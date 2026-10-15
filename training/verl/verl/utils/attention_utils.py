# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

_index_first_axis, _pad_input, _rearrange, _unpad_input = None, None, None, None


def _torch_index_first_axis(input: Tensor, indices: Tensor) -> Tensor:
    return input[indices]


def _torch_pad_input(hidden_states: Tensor, indices: Tensor, batch: int, seqlen: int) -> Tensor:
    output = hidden_states.new_zeros((batch * seqlen, *hidden_states.shape[1:]))
    output[indices] = hidden_states
    return output.view(batch, seqlen, *hidden_states.shape[1:])


def _torch_unpad_input(hidden_states: Tensor, attention_mask: Tensor, unused_mask: Tensor | None = None):
    batch, seqlen = attention_mask.shape
    all_masks = (attention_mask + unused_mask) if unused_mask is not None else attention_mask
    indices = torch.nonzero(all_masks.reshape(-1).bool(), as_tuple=False).flatten()
    unpadded = hidden_states.reshape(batch * seqlen, *hidden_states.shape[2:])[indices]
    seqlens = all_masks.sum(dim=-1, dtype=torch.int32)
    used_seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
    cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    max_seqlen = int(seqlens.max().item()) if seqlens.numel() else 0
    return unpadded, indices, cu_seqlens, max_seqlen, used_seqlens


def _get_attention_functions() -> tuple[Callable, Callable, Callable, Callable]:
    """Dynamically import attention functions based on available hardware."""

    from verl.utils.device import is_torch_npu_available

    global _index_first_axis, _pad_input, _rearrange, _unpad_input

    if is_torch_npu_available(check_device=False):
        from verl.utils.npu_flash_attn_utils import index_first_axis, pad_input, rearrange, unpad_input
    else:
        try:
            from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
        except ModuleNotFoundError as exc:
            if exc.name not in {"flash_attn", "flash_attn.bert_padding"}:
                raise
            from einops import rearrange

            index_first_axis = _torch_index_first_axis
            pad_input = _torch_pad_input
            unpad_input = _torch_unpad_input

    _index_first_axis, _pad_input, _rearrange, _unpad_input = index_first_axis, pad_input, rearrange, unpad_input

    return _index_first_axis, _pad_input, _rearrange, _unpad_input


def index_first_axis(*args, **kwargs):
    """
    Unified entry point for `index_first_axis` across CUDA and NPU backends.

    Dynamically dispatches to the appropriate device-specific implementation:
      - On CUDA: `flash_attn.bert_padding.index_first_axis`
      - On NPU: `transformers.integrations.npu_flash_attention.index_first_axis`
        (falls back to `transformers.modeling_flash_attention_utils._index_first_axis`
        in newer versions of transformers).

    Users can call this function directly without worrying about the underlying device.
    """
    func, *_ = _get_attention_functions()
    return func(*args, **kwargs)


def pad_input(*args, **kwargs):
    """
    Unified entry point for `pad_input` across CUDA and NPU backends.

    Dynamically dispatches to the appropriate device-specific implementation:
      - On CUDA: `flash_attn.bert_padding.pad_input`
      - On NPU: `transformers.integrations.npu_flash_attention.pad_input`
        (falls back to `transformers.modeling_flash_attention_utils._pad_input`
        in newer versions of transformers).

    Users can call this function directly without worrying about the underlying device.
    """
    _, func, *_ = _get_attention_functions()
    return func(*args, **kwargs)


def rearrange(*args, **kwargs):
    """
    Unified entry point for `rearrange` across CUDA and NPU backends.

    Dynamically dispatches to the appropriate device-specific implementation:
      - On CUDA: `flash_attn.bert_padding.rearrange`
      - On NPU: `transformers.integrations.npu_flash_attention.rearrange`
        (falls back to `einops.rearrange` if no dedicated NPU implementation exists).

    Users can call this function directly without worrying about the underlying device.
    """
    *_, func, _ = _get_attention_functions()
    return func(*args, **kwargs)


def unpad_input(*args, **kwargs):
    """
    Unified entry point for `unpad_input` across CUDA and NPU backends.

    Dynamically dispatches to the appropriate device-specific implementation:
      - On CUDA: `flash_attn.bert_padding.unpad_input`
      - On NPU: `transformers.integrations.npu_flash_attention.unpad_input`
        (falls back to `transformers.modeling_flash_attention_utils._unpad_input`
        in newer versions of transformers).

    Users can call this function directly without worrying about the underlying device.
    """
    *_, func = _get_attention_functions()
    return func(*args, **kwargs)


def get_diffusion_attention_mask(attention_mask: Tensor, L0: int, L1: int):
    """
    Build diffusion masks for *unpadded* (nnz) tokens.

    Input attention_mask: [B, L0 + 2*L1], where
      - segment 0 (len L0): padding on the LEFT
      - segment 1 (len L1): padding in its RIGHT half
      - segment 2 (len L1): padding in its RIGHT half

    Returns:
      segment_mask:  [1, total_nnz], values in {0,1,2}
      document_mask: [1, total_nnz], values in {0..B-1}
    """
    assert attention_mask.dim() == 2
    B, S = attention_mask.shape
    assert L0 + 2 * L1 == S

    device = attention_mask.device
    nnz_mask = attention_mask.to(dtype=torch.bool)

    # Segment id per position in the full (padded) sequence: [S]
    pos = torch.arange(S, device=device)
    seg_id = torch.empty(S, device=device, dtype=torch.long)
    seg_id[pos < L0] = 0
    seg_id[(pos >= L0) & (pos < L0 + L1)] = 1
    seg_id[pos >= L0 + L1] = 2

    # Expand to [B, S] then select nnz tokens in row-major order (b-major, then pos)
    seg_full = seg_id.unsqueeze(0).expand(B, S)
    doc_full = torch.arange(B, device=device, dtype=torch.long).unsqueeze(1).expand(B, S)

    segment_mask = seg_full[nnz_mask].unsqueeze(0)    # [1, total_nnz]
    document_mask = doc_full[nnz_mask].unsqueeze(0)   # [1, total_nnz]
    return segment_mask, document_mask



__all__ = ["index_first_axis", "pad_input", "rearrange", "unpad_input", "get_diffusion_attention_mask"]
