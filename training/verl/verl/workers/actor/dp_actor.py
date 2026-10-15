# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
Single Process Actor
"""

import gc
import hashlib
import inspect
import json
import logging
import math
import os
import threading
import time
import weakref
from contextlib import nullcontext
from copy import deepcopy
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from torch.nn.attention.flex_attention import create_block_mask
from tqdm import tqdm, trange
from verl import DataProto
from verl.trainer.ppo.block_tpf_ce_weight import (
    BLOCK_TPF_SOURCE,
    MissingBlockTPFMetadataError,
    build_block_tpf_token_tensors,
    normalize_block_tpf_weights,
    resolve_block_tpf_metadata_policy,
)
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.trainer.ppo.diffusion_tpf_replay import compute_tpf_reinforce_token_mean_loss
from verl.trainer.ppo.entropy_gated_rkl_probe import (
    DEFAULT_ENTROPY_THRESHOLDS,
    ENTROPY_GATED_RKL_REPORT_SLOTS,
    accumulate_entropy_gated_rkl_report_,
    all_reduce_entropy_gated_rkl_probe_stats_,
    compute_entropy_gated_ce_rkl_terms,
    compute_entropy_gated_rkl_probe_stats,
    entropy_gated_rkl_blocks_response_level_ce,
    entropy_gated_rkl_parity_terms,
    entropy_gated_rkl_probe_metrics,
    merge_entropy_gated_rkl_probe_stats_,
    validate_entropy_gated_rkl_composition,
)
from verl.trainer.ppo.gemma_incremental_prefix_cache import (
    DEFAULT_GEMMA_PREFIX_CACHE_MAX_BYTES,
    DEFAULT_GEMMA_PREFIX_CACHE_MAX_ENTRIES,
    GemmaIncrementalPrefixKVCache,
)
from verl.trainer.ppo.gemma_intermediate_replay import GEMMA_REPLAY_SCHEMA_VERSION
from verl.trainer.ppo.adjacent_entropy_contraction import (
    AdjacEntConfig,
    compute_adj_entropy_hinge_hidden,
    count_unique_valid_pairs_from_batch,
)
from verl.trainer.ppo.gemma_adjacent_entropy import (
    GemmaAdjacEntConfig,
    GemmaAecState,
    gemma_aec_metrics,
    gemma_aec_wave_terms,
)
from verl.trainer.ppo.hybrid_ce_negative_pg import (
    combine_hybrid_objective,
    compute_chunked_action_log_probs,
)
from verl.trainer.ppo.qwen_nostream_grpo import require_native_external_prefix_kv_backend
from verl.trainer.ppo.response_level_ce import (
    build_response_level_aec_pair_weights,
    build_response_level_ce_weights,
)
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input, get_diffusion_attention_mask
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.lifecycle import emit_lifecycle, lifecycle_memory
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits, probs_from_logits, extract_from_logits_v2
from verl.utils.ulysses import (
    all_gather_sequence_with_grad,
    gather_outputs_and_unpad,
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_rank,
    slice_input_tensor,
    ulysses_pad,
    ulysses_pad_and_slice_inputs,
    ulysses_pad_and_slice_tensor,
)
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_GEMMA_FP32_FINITE_CHECK_CHUNK_ELEMENTS = 8 * 1024 * 1024


def _all_finite_chunked(
    tensor: torch.Tensor,
    *,
    chunk_elements: int = _GEMMA_FP32_FINITE_CHECK_CHUNK_ELEMENTS,
) -> bool:
    """Check finiteness without materializing a tensor-sized bool buffer."""

    chunk_elements = max(int(chunk_elements), 1)
    flat = tensor.reshape(-1)
    all_finite = torch.ones((), dtype=torch.bool, device=tensor.device)
    for start in range(0, int(flat.numel()), chunk_elements):
        chunk = flat.narrow(
            0,
            start,
            min(chunk_elements, int(flat.numel()) - start),
        )
        all_finite.logical_and_(torch.isfinite(chunk).all())
    return bool(all_finite.item())


def _pack_block_local_response_hidden_impl(
    *,
    values,
    starts,
    lengths,
    dense_value,
    field_name,
    input_ids,
    attention_mask,
    indices,
    response_length,
):
    """Pack bounded block pages directly into the remove-padding layout."""
    if values is None:
        return None
    if dense_value is not None:
        raise RuntimeError(f"received both dense and block-local {field_name}")
    if not (
        torch.is_tensor(values)
        and torch.is_tensor(starts)
        and torch.is_tensor(lengths)
    ):
        raise RuntimeError(f"incomplete block-local {field_name} payload")
    batch_size, seqlen = input_ids.shape
    values = values.to(device=input_ids.device)
    starts = starts.to(device=input_ids.device, dtype=torch.long).view(-1)
    lengths = lengths.to(device=input_ids.device, dtype=torch.long).view(-1)
    if values.dim() != 3 or int(values.shape[0]) != batch_size:
        raise RuntimeError(
            f"block-local {field_name} must be [batch, block, hidden], "
            f"got {tuple(values.shape)}"
        )
    if int(starts.numel()) != batch_size or int(lengths.numel()) != batch_size:
        raise RuntimeError(
            f"block-local {field_name} metadata does not match batch={batch_size}: "
            f"starts={tuple(starts.shape)} lengths={tuple(lengths.shape)}"
        )
    block_size = int(values.shape[1])
    hidden_size = int(values.shape[2])
    if bool(
        (
            starts.lt(0)
            | starts.ge(response_length)
            | lengths.lt(0)
            | lengths.gt(block_size)
            | (starts + lengths).gt(response_length)
        ).any()
    ):
        raise RuntimeError(f"block-local {field_name} positions out of range")
    packed = torch.zeros(
        (indices.numel(), hidden_size),
        device=input_ids.device,
        dtype=values.dtype,
    )
    offsets = torch.arange(
        block_size,
        device=input_ids.device,
        dtype=torch.long,
    ).unsqueeze(0)
    positions = starts.unsqueeze(1) + offsets
    valid = offsets < lengths.unsqueeze(1)
    rows = (
        torch.arange(
            batch_size,
            device=input_ids.device,
            dtype=torch.long,
        )
        .unsqueeze(1)
        .expand_as(positions)
    )
    rows = rows[valid]
    positions = positions[valid]
    selected_values = values[valid]
    flat_attention_mask = attention_mask.reshape(-1).bool()
    flat_offsets = flat_attention_mask.to(torch.long).cumsum(dim=0) - 1
    response_abs_positions = seqlen - response_length + positions
    response_flat_idx = rows * seqlen + response_abs_positions
    response_valid = flat_attention_mask[response_flat_idx]
    if bool(response_valid.any()):
        packed_idx = flat_offsets[response_flat_idx[response_valid]]
        packed[packed_idx] = selected_values[response_valid]
    return packed.unsqueeze(0)


def _pack_multi_block_response_hidden_impl(
    *,
    values,
    row_indices,
    starts,
    lengths,
    dense_value,
    field_name,
    input_ids,
    attention_mask,
    indices,
    response_length,
):
    """Pack multiple bounded block pages per batch row into remove-padding layout."""
    if values is None:
        return None
    if dense_value is not None:
        raise RuntimeError(f"received both dense and multi-block {field_name}")
    if not all(torch.is_tensor(value) for value in (values, starts, lengths)):
        raise RuntimeError(f"incomplete multi-block {field_name} payload")
    batch_size, seqlen = input_ids.shape
    values = values.to(device=input_ids.device)
    starts = starts.to(device=input_ids.device, dtype=torch.long)
    lengths = lengths.to(device=input_ids.device, dtype=torch.long)
    if values.dim() == 4:
        if int(values.shape[0]) != batch_size:
            raise RuntimeError(
                f"padded multi-block {field_name} batch mismatch: "
                f"values={tuple(values.shape)} batch={batch_size}"
            )
        pages_per_row = int(values.shape[1])
        if starts.shape != (batch_size, pages_per_row) or lengths.shape != (
            batch_size,
            pages_per_row,
        ):
            raise RuntimeError(
                f"padded multi-block {field_name} metadata mismatch: "
                f"starts={tuple(starts.shape)} lengths={tuple(lengths.shape)} "
                f"expected={(batch_size, pages_per_row)}"
            )
        row_indices = (
            torch.arange(batch_size, device=input_ids.device, dtype=torch.long)
            .unsqueeze(1)
            .expand(batch_size, pages_per_row)
            .reshape(-1)
        )
        values = values.reshape(
            batch_size * pages_per_row,
            int(values.shape[2]),
            int(values.shape[3]),
        )
        starts = starts.reshape(-1)
        lengths = lengths.reshape(-1)
        present = lengths.gt(0)
        values = values[present]
        row_indices = row_indices[present]
        starts = starts[present]
        lengths = lengths[present]
    elif values.dim() == 3 and torch.is_tensor(row_indices):
        row_indices = row_indices.to(
            device=input_ids.device,
            dtype=torch.long,
        ).view(-1)
        starts = starts.view(-1)
        lengths = lengths.view(-1)
    else:
        raise RuntimeError(
            f"multi-block {field_name} must be [pages, block, hidden] with "
            f"row indices or [batch, pages, block, hidden], "
            f"got {tuple(values.shape)}"
        )
    page_count = int(values.shape[0])
    if not (
        int(row_indices.numel()) == page_count
        and int(starts.numel()) == page_count
        and int(lengths.numel()) == page_count
    ):
        raise RuntimeError(
            f"multi-block {field_name} metadata does not match pages={page_count}: "
            f"rows={tuple(row_indices.shape)} starts={tuple(starts.shape)} "
            f"lengths={tuple(lengths.shape)}"
        )
    block_size = int(values.shape[1])
    hidden_size = int(values.shape[2])
    if bool(
        (
            row_indices.lt(0)
            | row_indices.ge(batch_size)
            | starts.lt(0)
            | starts.ge(response_length)
            | lengths.lt(0)
            | lengths.gt(block_size)
            | (starts + lengths).gt(response_length)
        ).any()
    ):
        raise RuntimeError(f"multi-block {field_name} positions out of range")
    packed = torch.zeros(
        (indices.numel(), hidden_size),
        device=input_ids.device,
        dtype=values.dtype,
    )
    if page_count == 0:
        return packed.unsqueeze(0)
    offsets = torch.arange(
        block_size,
        device=input_ids.device,
        dtype=torch.long,
    ).unsqueeze(0)
    positions = starts.unsqueeze(1) + offsets
    valid = offsets < lengths.unsqueeze(1)
    rows = row_indices.unsqueeze(1).expand_as(positions)[valid]
    positions = positions[valid]
    selected_values = values[valid]
    linear = rows * int(response_length) + positions
    if int(torch.unique(linear).numel()) != int(linear.numel()):
        raise RuntimeError(
            f"multi-block {field_name} contains overlapping row/position pages"
        )
    flat_attention_mask = attention_mask.reshape(-1).bool()
    flat_offsets = flat_attention_mask.to(torch.long).cumsum(dim=0) - 1
    response_abs_positions = seqlen - response_length + positions
    response_flat_idx = rows * seqlen + response_abs_positions
    response_valid = flat_attention_mask[response_flat_idx]
    if bool(response_valid.any()):
        packed_idx = flat_offsets[response_flat_idx[response_valid]]
        packed[packed_idx] = selected_values[response_valid]
    return packed.unsqueeze(0)


def _pack_gpu_replay_lease_hidden_impl(
    *,
    descriptor_rows,
    starts,
    lengths,
    field_name,
    input_ids,
    attention_mask,
    indices,
    response_length,
    sequence_parallel_size=1,
):
    """Pack GPU-local IPC pages and share them within an SP group."""
    from sglang.srt.dllm.algorithm.sdar_gpu_replay_lease import (
        import_gpu_replay_lease,
        is_gpu_replay_lease,
        is_gpu_replay_lease_fanout,
        select_gpu_replay_lease_for_device,
    )

    if isinstance(descriptor_rows, np.ndarray):
        descriptor_rows = descriptor_rows.reshape(-1).tolist()
    elif isinstance(descriptor_rows, tuple):
        descriptor_rows = list(descriptor_rows)
    if not isinstance(descriptor_rows, list):
        return None
    batch_size, seqlen = input_ids.shape
    if len(descriptor_rows) != batch_size:
        raise RuntimeError(
            f"{field_name} GPU lease rows do not match batch: "
            f"{len(descriptor_rows)}/{batch_size}"
        )
    if not (torch.is_tensor(starts) and torch.is_tensor(lengths)):
        raise RuntimeError(
            f"{field_name} GPU leases require starts and lengths"
        )
    starts_cpu = starts.detach().cpu().long()
    lengths_cpu = lengths.detach().cpu().long()
    if starts_cpu.shape != lengths_cpu.shape:
        raise RuntimeError(
            f"{field_name} GPU lease starts/lengths mismatch"
        )
    flat_attention_mask = attention_mask.reshape(-1).bool()
    flat_offsets = flat_attention_mask.to(torch.long).cumsum(dim=0) - 1
    imported = []
    values = None
    actor_device_index = (
        input_ids.device.index
        if input_ids.device.index is not None
        else torch.cuda.current_device()
    )
    actor_device_uuid = str(
        torch.cuda.get_device_properties(actor_device_index).uuid
    )
    metadata_descriptor = next(
        (
            (
                descriptor["descriptors"][0]
                if is_gpu_replay_lease_fanout(descriptor)
                else descriptor
            )
            for descriptors in descriptor_rows
            if isinstance(descriptors, list)
            for descriptor in descriptors
            if (
                is_gpu_replay_lease(descriptor)
                or is_gpu_replay_lease_fanout(descriptor)
            )
        ),
        None,
    )
    if metadata_descriptor is None:
        return None
    dtype_name = str(metadata_descriptor.get("dtype", "")).split(".")[-1]
    packed_dtype = getattr(torch, dtype_name, None)
    metadata_shape = metadata_descriptor.get("shape", [])
    if packed_dtype is None or len(metadata_shape) != 2:
        raise RuntimeError(
            f"{field_name} GPU lease has invalid dtype/shape metadata"
        )
    packed = torch.zeros(
        (indices.numel(), int(metadata_shape[-1])),
        device=input_ids.device,
        dtype=packed_dtype,
    )
    uses_fanout = False
    seen_linear: set[int] = set()
    try:
        for row_index, descriptors in enumerate(descriptor_rows):
            if not isinstance(descriptors, list):
                raise RuntimeError(
                    f"{field_name} GPU lease row is not a list"
                )
            present_pages = int(
                lengths_cpu[row_index].gt(0).sum().item()
            )
            if len(descriptors) != present_pages:
                raise RuntimeError(
                    f"{field_name} GPU lease page count mismatch: "
                    f"row={row_index} descriptors={len(descriptors)} "
                    f"lengths={present_pages}"
                )
            for page_index, descriptor in enumerate(descriptors):
                if is_gpu_replay_lease_fanout(descriptor):
                    uses_fanout = True
                    descriptor = select_gpu_replay_lease_for_device(
                        descriptor,
                        device_uuid=actor_device_uuid,
                    )
                elif (
                    is_gpu_replay_lease(descriptor)
                    and str(descriptor.get("device_uuid"))
                    != actor_device_uuid
                ):
                    continue
                if not is_gpu_replay_lease(descriptor):
                    raise RuntimeError(
                        f"{field_name} contains an invalid GPU lease"
                    )
                lease = import_gpu_replay_lease(descriptor)
                imported.append(lease)
                values = lease.tensor
                if values.dim() != 2:
                    raise RuntimeError(
                        f"{field_name} GPU lease must be [tokens, hidden], "
                        f"got {tuple(values.shape)}"
                    )
                if (
                    int(values.shape[-1]) != int(packed.shape[-1])
                    or values.dtype != packed.dtype
                ):
                    raise RuntimeError(
                        f"{field_name} GPU lease metadata mismatch"
                    )
                start = int(starts_cpu[row_index, page_index].item())
                length = int(lengths_cpu[row_index, page_index].item())
                if (
                    start < 0
                    or length < 0
                    or start + length > response_length
                    or length > int(values.shape[0])
                ):
                    raise RuntimeError(
                        f"{field_name} GPU lease page is out of range"
                    )
                positions = torch.arange(
                    start,
                    start + length,
                    device=input_ids.device,
                    dtype=torch.long,
                )
                linear_values = [
                    row_index * int(response_length) + position
                    for position in range(start, start + length)
                ]
                if seen_linear.intersection(linear_values):
                    raise RuntimeError(
                        f"{field_name} GPU lease pages overlap"
                    )
                seen_linear.update(linear_values)
                response_abs_positions = (
                    seqlen - response_length + positions
                )
                response_flat_idx = (
                    row_index * seqlen + response_abs_positions
                )
                response_valid = flat_attention_mask[
                    response_flat_idx
                ]
                if bool(response_valid.any()):
                    packed_idx = flat_offsets[
                        response_flat_idx[response_valid]
                    ]
                    packed[packed_idx] = values[:length][response_valid]
        torch.cuda.current_stream(input_ids.device).synchronize()
        if int(sequence_parallel_size) > 1 and not uses_fanout:
            from verl.utils.ulysses import (
                get_ulysses_sequence_parallel_group,
            )

            group = get_ulysses_sequence_parallel_group()
            if group is None:
                raise RuntimeError(
                    f"{field_name} requires an initialized SP group"
                )
            torch.distributed.all_reduce(
                packed,
                op=torch.distributed.ReduceOp.SUM,
                group=group,
            )
        return packed.unsqueeze(0)
    finally:
        values = None
        for lease in imported:
            lease.release(
                synchronize=False,
                collect_garbage=False,
            )


def _shift_preexpanded_logits_source_impl(
    hidden_states: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    L0: torch.Tensor,
    L1: torch.Tensor,
    staircase_size: int,
) -> torch.Tensor:
    """Apply SDAR blockwise next-token hidden shift on a gathered sequence."""
    shifted = hidden_states.clone()
    cu = cu_seqlens.detach().cpu().long().tolist()
    l0_values = L0.detach().cpu().long().tolist()
    l1_values = L1.detach().cpu().long().tolist()
    for doc_index, (l0_value, l1_value) in enumerate(
        zip(l0_values, l1_values, strict=True)
    ):
        doc_start = int(cu[doc_index])
        clean_start = doc_start + int(l0_value)
        noisy_start = clean_start + int(l1_value)
        for block_start in range(
            0,
            int(l1_value),
            int(staircase_size),
        ):
            block_length = min(
                int(staircase_size),
                int(l1_value) - block_start,
            )
            noisy_pos = noisy_start + block_start
            clean_pos = clean_start + block_start
            if clean_pos > doc_start:
                shifted[noisy_pos] = hidden_states[clean_pos - 1]
            if block_length > 1:
                shifted[
                    noisy_pos + 1 : noisy_pos + block_length
                ] = hidden_states[
                    noisy_pos : noisy_pos + block_length - 1
                ]
    return shifted


_GEMMA_REPLAY_FSDP_CAPTURE_EXPECTED_TORCH_VERSION = (2, 11)
_GEMMA_REPLAY_FSDP_CAPTURE_EXPECTED_SIGNATURE = ("state", "handle", "sharded_grad")
_gemma_replay_fsdp_capture_lock = threading.RLock()
_gemma_replay_fsdp_capture_registry: dict[int, tuple[weakref.ReferenceType, int]] = {}
_gemma_replay_fsdp_accumulate_sharded_grad_original = None
_gemma_replay_fsdp_accumulate_sharded_grad_installed = False


def _gemma_replay_torch_major_minor() -> tuple[int, int]:
    version = torch.__version__.split("+", 1)[0]
    parts = version.split(".")
    if len(parts) < 2:
        raise RuntimeError(f"Unsupported PyTorch version string for DiffusionGemma FSDP capture: {torch.__version__}")
    return int(parts[0]), int(parts[1])


def _validate_gemma_replay_fsdp_accumulate_sharded_grad(fn) -> None:
    if _gemma_replay_torch_major_minor() != _GEMMA_REPLAY_FSDP_CAPTURE_EXPECTED_TORCH_VERSION:
        raise RuntimeError(
            "DiffusionGemma post-reduce FP32 grad capture is version-pinned to "
            f"PyTorch {_GEMMA_REPLAY_FSDP_CAPTURE_EXPECTED_TORCH_VERSION[0]}."
            f"{_GEMMA_REPLAY_FSDP_CAPTURE_EXPECTED_TORCH_VERSION[1]}.x; "
            f"found {torch.__version__}"
        )
    if getattr(fn, "__module__", None) != "torch.distributed.fsdp._runtime_utils" or getattr(fn, "__name__", None) != "_accumulate_sharded_grad":
        raise RuntimeError(
            "DiffusionGemma post-reduce FP32 grad capture expected "
            "torch.distributed.fsdp._runtime_utils._accumulate_sharded_grad"
        )
    signature = inspect.signature(fn)
    if tuple(signature.parameters) != _GEMMA_REPLAY_FSDP_CAPTURE_EXPECTED_SIGNATURE:
        raise RuntimeError(
            "DiffusionGemma post-reduce FP32 grad capture found an unsupported "
            f"_accumulate_sharded_grad signature: {signature}"
        )


def _gemma_replay_fsdp_accumulate_sharded_grad_wrapper(state, handle, sharded_grad):
    flat_param = getattr(handle, "flat_param", None)
    entry = None
    if flat_param is not None:
        flat_param_id = id(flat_param)
        with _gemma_replay_fsdp_capture_lock:
            entry = _gemma_replay_fsdp_capture_registry.get(flat_param_id)
    if entry is not None:
        actor_ref, window_id = entry
        actor = actor_ref()
        if actor is None:
            with _gemma_replay_fsdp_capture_lock:
                _gemma_replay_fsdp_capture_registry.pop(flat_param_id, None)
        else:
            actor._capture_gemma_replay_post_reduce_fp32_grad(handle, sharded_grad, window_id)
    return _gemma_replay_fsdp_accumulate_sharded_grad_original(state, handle, sharded_grad)


def _install_gemma_replay_fsdp_post_reduce_grad_capture() -> None:
    global _gemma_replay_fsdp_accumulate_sharded_grad_original
    global _gemma_replay_fsdp_accumulate_sharded_grad_installed
    from torch.distributed.fsdp import _runtime_utils

    with _gemma_replay_fsdp_capture_lock:
        current = _runtime_utils._accumulate_sharded_grad
        if _gemma_replay_fsdp_accumulate_sharded_grad_installed:
            if current is not _gemma_replay_fsdp_accumulate_sharded_grad_wrapper:
                raise RuntimeError(
                    "DiffusionGemma post-reduce FP32 grad capture was installed, "
                    "but torch.distributed.fsdp._runtime_utils._accumulate_sharded_grad was replaced"
                )
            return
        _validate_gemma_replay_fsdp_accumulate_sharded_grad(current)
        _gemma_replay_fsdp_accumulate_sharded_grad_original = current
        _runtime_utils._accumulate_sharded_grad = _gemma_replay_fsdp_accumulate_sharded_grad_wrapper
        _gemma_replay_fsdp_accumulate_sharded_grad_installed = True


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        if self._is_diffusion_gemma_actor():
            self._recompute_rotary_buffers()
        role = "Ref" if actor_optimizer is None else "Actor"

        self.block_length = self.config.block_length

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.use_dynamic_bsz = self.config.get("use_dynamic_bsz", False)

        self.use_prefix_grouper = self.config.get("use_prefix_grouper", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_prefix_grouper={self.use_prefix_grouper}")

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

        # Sum of squared probabilities computation (for optimal_token_baseline)
        # Only initialize if calculate_sum_pi_squared config is enabled
        if self.config.get("calculate_sum_pi_squared", False):
            self.calculate_sum_pi_squared_from_logits = (
                torch.compile(verl_F.calculate_sum_pi_squared_from_logits, dynamic=True)
                if self.config.get("use_torch_compile", True)
                else verl_F.calculate_sum_pi_squared_from_logits
            )
            assert not (self.use_fused_kernels or self.use_prefix_grouper), (
                "calculate_sum_pi_squared is not supported with "
                f"{self.use_fused_kernels=} or {self.use_prefix_grouper=} for now."
            )

        self._replica_grad_sync_group = None
        self._replica_grad_sync_group_size = 1
        self._last_replica_grad_sync_tensors = 0
        self._last_replica_grad_sync_bytes = 0
        self._last_replica_grad_sync_seconds = 0.0
        self._lifecycle_optimizer_step_count = 0
        self._last_optimizer_step_applied = False
        self._init_gemma_replay_fp32_grad_accum_state()
        self._install_gemma_replay_fp32_grad_accum_zero_grad_hook()
        self._gemma_incr_prefix_kv_cache: GemmaIncrementalPrefixKVCache | None = None

    def _recompute_rotary_buffers(self) -> None:
        """Recompute every rotary embedding's inverse-frequency buffers in float32 on this rank."""
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        for name, module in self.actor_module.named_modules():
            init_fns = getattr(module, "rope_init_fns", None)
            if not isinstance(init_fns, dict) or not init_fns:
                # one inv_freq buffer, computed by rope_init_fn (an attribute of some models, chosen from rope_type
                # at construction by others)
                if not torch.is_tensor(getattr(module, "inv_freq", None)):
                    continue
                single = getattr(module, "rope_init_fn", None)
                if not callable(single):
                    rope_type = getattr(module, "rope_type", None)
                    if rope_type == "default" and callable(getattr(module, "compute_default_rope_parameters", None)):
                        single = module.compute_default_rope_parameters
                    elif isinstance(rope_type, str):
                        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

                        single = ROPE_INIT_FUNCTIONS.get(rope_type)
                if not callable(single):
                    continue
                init_fns = {"": single}
            for layer_type, init_fn in init_fns.items():
                prefix = f"{layer_type}_" if layer_type else ""
                buffer = getattr(module, f"{prefix}inv_freq", None)
                if not torch.is_tensor(buffer):
                    continue
                if layer_type:
                    kwargs = {"device": buffer.device, "layer_type": layer_type}
                    if layer_type == "full_attention" and module.rope_type.get(layer_type) == "proportional":
                        kwargs["head_dim_key"] = "global_head_dim"
                    expected, scaling = init_fn(module.config, **kwargs)
                else:
                    expected, scaling = init_fn(module.config, buffer.device)
                expected = expected.to(device=buffer.device, dtype=torch.float32)
                held = buffer.shape == expected.shape and bool(torch.equal(buffer, expected.to(buffer.dtype)))
                print(
                    f"[ORL_ROTARY] rank {rank} {name.replace('_fsdp_wrapped_module.', '')} {layer_type or 'rope'}: "
                    f"buffer {'held' if held else 'did not hold'} the configured frequencies; recomputed in float32",
                    flush=True,
                )
                module.register_buffer(f"{prefix}inv_freq", expected.clone(), persistent=False)
                module.register_buffer(f"{prefix}original_inv_freq", expected.clone(), persistent=False)
                setattr(module, f"{prefix}attention_scaling", scaling)

    def _qwen_nostream_grpo_enabled(self) -> bool:
        qwen_cfg = self.config.get("qwen_nostream_grpo", {}) or {}
        return bool(qwen_cfg.get("enable", False))

    def _negative_action_pg_enabled(self) -> bool:
        jsd_cfg = self.config.get("generation_consistency_jsd", {}) or {}
        negative_cfg = jsd_cfg.get("negative_action_pg", {}) or {}
        return (
            bool(jsd_cfg.get("enable", False))
            and str(jsd_cfg.get("objective", "")) == "trace_all_ce"
            and bool(negative_cfg.get("enable", False))
        )

    def _call_qwen_external_prefix_kv_replay(self, data: DataProto) -> dict[str, float]:
        """Call the mandatory Qwen no-copy external-prefix-KV replay edge."""

        if not bool(self.config.get("use_qwen_external_prefix_kv_replay", False)):
            raise RuntimeError(
                "Qwen no-stream GRPO requires actor.use_qwen_external_prefix_kv_replay=true; "
                "refusing packed/repeated-prefix fallback."
            )

        base_actor_module = getattr(self.actor_module, "module", self.actor_module)
        hook = getattr(
            base_actor_module,
            "qwen_nostream_external_prefix_kv_replay",
            None,
        )
        context_keys = {"input_ids", "attention_mask", "responses"}
        if hook is None and context_keys.issubset(data.batch.keys()):
            from verl.trainer.ppo.qwen_nostream_external_kv_overlay import (
                install_qwen_nostream_external_prefix_kv_hook,
            )

            install_qwen_nostream_external_prefix_kv_hook(base_actor_module)
            hook = getattr(
                base_actor_module,
                "qwen_nostream_external_prefix_kv_replay",
                None,
            )
        if callable(hook):
            result = hook(data)
            if result is None:
                return {"actor/qwen_external_prefix_kv/model_hook_called": 1.0}
            if isinstance(result, dict):
                metrics = {
                    f"actor/qwen_external_prefix_kv/{str(key)}": float(value)
                    for key, value in result.items()
                    if isinstance(value, (int, float))
                }
                metrics["actor/qwen_external_prefix_kv/model_hook_called"] = 1.0
                return metrics
            raise RuntimeError("qwen_nostream_external_prefix_kv_replay must return None or a metrics dict")

        required = (
            "qwen_external_kv_q_canvas",
            "qwen_external_kv_k_canvas",
            "qwen_external_kv_v_canvas",
            "qwen_external_kv_k_prefix",
            "qwen_external_kv_v_prefix",
            "qwen_external_kv_cu_seqlens_q",
            "qwen_external_kv_cu_seqlens_prefix",
        )
        missing = [key for key in required if key not in data.batch]
        if missing:
            raise RuntimeError(
                "Qwen no-stream GRPO requires real external-prefix-KV replay tensors "
                "or a model qwen_nostream_external_prefix_kv_replay hook; missing "
                f"{missing}. Packed full-sequence fallback is disabled."
            )
        q_canvas = data.batch["qwen_external_kv_q_canvas"]
        k_canvas = data.batch["qwen_external_kv_k_canvas"]
        v_canvas = data.batch["qwen_external_kv_v_canvas"]
        k_prefix = data.batch["qwen_external_kv_k_prefix"]
        v_prefix = data.batch["qwen_external_kv_v_prefix"]
        cu_seqlens_q = data.batch["qwen_external_kv_cu_seqlens_q"]
        cu_seqlens_prefix = data.batch["qwen_external_kv_cu_seqlens_prefix"]
        prefix_segment_ids = data.batch.get("qwen_external_kv_prefix_segment_ids", None)
        external_kv_varlen_attention, _ = require_native_external_prefix_kv_backend(
            q_canvas=q_canvas,
            k_canvas=k_canvas,
            v_canvas=v_canvas,
            k_prefix=k_prefix,
            v_prefix=v_prefix,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_prefix=cu_seqlens_prefix,
            prefix_segment_ids=prefix_segment_ids,
        )
        _ = external_kv_varlen_attention(
            q_canvas,
            k_canvas,
            v_canvas,
            k_prefix,
            v_prefix,
            cu_seqlens_q,
            cu_seqlens_prefix,
            prefix_segment_ids=prefix_segment_ids,
            backend="triton",
        )
        return {"actor/qwen_external_prefix_kv/native_call": 1.0}

    def _actor_model_config(self):
        module = getattr(self.actor_module, "module", self.actor_module)
        return getattr(module, "config", None)

    def invalidate_gemma_prefix_kv_cache(self) -> None:
        """Invalidate the incremental prefix KV cache (if active).

        Must be called at the start of each update window, on abort, and after
        the optimizer step so stale KV entries from a previous model version
        are never reused.
        """
        cache = getattr(self, "_gemma_incr_prefix_kv_cache", None)
        if cache is not None:
            cache.invalidate()

    def _is_diffusion_gemma_actor(self) -> bool:
        config = self._actor_model_config()
        return getattr(config, "model_type", None) == "diffusion_gemma"

    def _new_diffusion_gemma_external_kv_replay_context(self):
        enabled = bool(
            self.config.get("use_diffusion_gemma_external_kv_replay", False)
        )
        if not enabled:
            return None, nullcontext
        try:
            from transformers.models.diffusion_gemma.replay_external_kv import (
                DiffusionGemmaExternalKVReplayStats,
                diffusion_gemma_external_kv_replay,
            )
        except ImportError as exc:
            raise RuntimeError(
                "actor.use_diffusion_gemma_external_kv_replay requires the "
                "DiffusionGemma external-KV overlay"
            ) from exc

        stats = DiffusionGemmaExternalKVReplayStats()

        def context_factory():
            return diffusion_gemma_external_kv_replay(stats=stats)

        return stats, context_factory

    @staticmethod
    def _diffusion_gemma_external_kv_replay_metrics(
        stats,
        *,
        decoder_forward_seconds: float,
        decoder_backward_seconds: float,
    ) -> dict[str, float]:
        prefix = "actor/diffusion_gemma/external_kv"
        if stats is None:
            metrics = {
                f"{prefix}/enabled": 0.0,
                f"{prefix}/calls": 0.0,
                f"{prefix}/backend_triton_calls": 0.0,
                f"{prefix}/fallbacks": 0.0,
                f"{prefix}/kernel_seconds": 0.0,
                f"{prefix}/shared_prefix_rows": 0.0,
                f"{prefix}/shared_prefix_bytes_avoided": 0.0,
                f"{prefix}/kv_pack_calls": 0.0,
                f"{prefix}/kv_repeat_calls": 0.0,
                f"{prefix}/prefix_append_calls": 0.0,
                f"{prefix}/prefix_append_seconds": 0.0,
            }
        else:
            stats.assert_no_kv_pack_or_repeat()
            metrics = stats.metrics(prefix)
        kernel_seconds = float(metrics[f"{prefix}/kernel_seconds"])
        metrics[f"{prefix}/moe_remaining_decoder_seconds"] = max(
            float(decoder_forward_seconds) - kernel_seconds,
            0.0,
        )
        metrics[f"{prefix}/backward_seconds"] = float(decoder_backward_seconds)
        return metrics

    def _forward_diffusion_gemma_micro_batch(
        self,
        micro_batch: dict[str, torch.Tensor],
        temperature: float,
        calculate_entropy: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Score one rollout block per sample with DiffusionGemma's encoder/decoder API.

        DiffusionGemma is not a decoder-only model: committed tokens are encoded
        causally into a KV cache and a separate canvas is denoised with
        bidirectional attention. The trainer transports the globally rotated
        block index and deterministic noise seed explicitly so old/new policy
        passes use identical inputs. The causal prefix cache is built under
        no-grad with an encoder-only model path and then consumed by a
        decoder-only gradient pass. This avoids putting tied encoder/decoder
        parameters behind two nested FSDP wrappers in the same backward graph
        while preserving prompt and committed-block conditioning.
        """
        input_ids = micro_batch["input_ids"]
        responses = micro_batch["responses"]
        attention_mask = micro_batch["attention_mask"].bool()
        batch_size, sequence_length = input_ids.shape
        response_length = responses.shape[-1]
        prompt_width = sequence_length - response_length
        if prompt_width < 0:
            raise RuntimeError(
                "DiffusionGemma actor expected input_ids=[prompt|response], "
                f"got input_shape={tuple(input_ids.shape)} response_shape={tuple(responses.shape)}"
            )

        model_config = self._actor_model_config()
        canvas_length = int(getattr(model_config, "canvas_length", self.block_length))
        text_config = getattr(model_config, "text_config", model_config)
        vocab_size = int(getattr(text_config, "vocab_size"))
        pad_token_id = int(micro_batch.get("pad_token_id", 0))
        effective_temperature = self._effective_logprob_temperature(temperature)
        metadata_keys = (
            "gemma_selected_block_indices",
            "gemma_available_block_counts",
            "gemma_denoise_noise_seeds",
        )
        missing_metadata = [key for key in metadata_keys if key not in micro_batch]
        if missing_metadata:
            raise RuntimeError(
                "DiffusionGemma actor requires explicit trainer-selected block metadata; "
                f"missing={missing_metadata}"
            )
        selected_block_indices = micro_batch["gemma_selected_block_indices"].long().reshape(-1)
        available_block_counts = micro_batch["gemma_available_block_counts"].long().reshape(-1)
        denoise_noise_seeds = micro_batch["gemma_denoise_noise_seeds"].long().reshape(-1)
        if any(int(value.numel()) != batch_size for value in (
            selected_block_indices,
            available_block_counts,
            denoise_noise_seeds,
        )):
            raise RuntimeError("DiffusionGemma block metadata must have one value per batch row")

        response_active = attention_mask[:, -response_length:] & responses.ge(0)
        if "response_mask" in micro_batch:
            if micro_batch["response_mask"].shape != responses.shape:
                raise RuntimeError("DiffusionGemma response_mask must match responses")
            response_active &= micro_batch["response_mask"].bool()

        log_probs = torch.zeros(responses.shape, device=input_ids.device, dtype=torch.float32)
        entropy = (
            torch.zeros(responses.shape, device=input_ids.device, dtype=torch.float32)
            if calculate_entropy
            else None
        )
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)
        sum_pi_squared = (
            torch.zeros(responses.shape, device=input_ids.device, dtype=torch.float32)
            if calculate_sum_pi_squared
            else None
        )
        ce_sum = None
        ce_count = 0
        selected_blocks = []
        prefix_cache_seconds = 0.0
        decoder_seconds = 0.0
        prefix_event_pairs = []
        decoder_event_pairs = []
        use_cuda_events = bool(input_ids.is_cuda)
        external_kv_stats, external_kv_context = (
            self._new_diffusion_gemma_external_kv_replay_context()
        )

        for batch_idx in range(batch_size):
            prompt_mask = attention_mask[batch_idx, :prompt_width]
            prompt_ids = input_ids[batch_idx, :prompt_width][prompt_mask]
            if prompt_ids.numel() == 0:
                raise RuntimeError("DiffusionGemma actor received an empty prompt")

            response_valid = response_active[batch_idx]
            valid_positions = torch.nonzero(response_valid, as_tuple=False).flatten()
            if valid_positions.numel():
                active_block_indices = torch.unique(valid_positions.div(canvas_length, rounding_mode="floor"))
            else:
                active_block_indices = torch.zeros(1, dtype=torch.long, device=input_ids.device)
            num_blocks = int(active_block_indices.numel())
            block_index = int(selected_block_indices[batch_idx].detach().cpu().item())
            transported_num_blocks = int(available_block_counts[batch_idx].detach().cpu().item())
            selected_is_active = bool(active_block_indices.eq(block_index).any().detach().cpu().item())
            if transported_num_blocks != num_blocks or not selected_is_active:
                raise RuntimeError(
                    "DiffusionGemma block metadata does not match active response extent: "
                    f"row={batch_idx} selected={block_index} transported_available={transported_num_blocks} "
                    f"derived_available={num_blocks}"
                )
            block_start = block_index * canvas_length
            block_end = min(block_start + canvas_length, response_length)
            block_width = max(0, block_end - block_start)
            selected_blocks.append(block_index)

            committed_mask = response_valid[:block_start]
            committed_ids = responses[batch_idx, :block_start][committed_mask]
            prefix_ids = torch.cat((prompt_ids, committed_ids), dim=0)
            prefix_length = int(prefix_ids.numel())

            clean_canvas = torch.full(
                (canvas_length,),
                pad_token_id,
                device=input_ids.device,
                dtype=input_ids.dtype,
            )
            active_canvas = torch.zeros(canvas_length, device=input_ids.device, dtype=torch.bool)
            if block_width:
                block_active = response_valid[block_start:block_end]
                block_labels = responses[batch_idx, block_start:block_end]
                clean_canvas[:block_width] = torch.where(
                    block_active,
                    block_labels,
                    torch.full_like(block_labels, pad_token_id),
                )
                active_canvas[:block_width] = block_active

            canvas_positions = torch.arange(canvas_length, device=input_ids.device, dtype=torch.long)
            # The seed is independent of clean labels, avoiding target leakage
            # while keeping old/new policy inputs byte-identical.
            noisy_canvas = torch.remainder(
                (canvas_positions + 1) * 1_103_515_245
                + denoise_noise_seeds[batch_idx] * 12_345
                + (block_start + 1) * 2_654_435_761,
                vocab_size,
            ).to(dtype=input_ids.dtype)

            prefix_positions = torch.arange(prefix_length, device=input_ids.device, dtype=torch.long)
            decoder_positions = torch.arange(
                prefix_length,
                prefix_length + canvas_length,
                device=input_ids.device,
                dtype=torch.long,
            )
            prefix_attention_mask = torch.ones(
                (1, prefix_length),
                device=input_ids.device,
                dtype=attention_mask.dtype,
            )

            # Encoder and decoder layers share parameters. Keep the encoder out
            # of the backward graph, but do not run the decoder while constructing
            # its detached causal cache.
            was_training = self.actor_module.training
            self.actor_module.eval()
            try:
                with torch.no_grad():
                    if use_cuda_events:
                        prefix_start = torch.cuda.Event(enable_timing=True)
                        prefix_end = torch.cuda.Event(enable_timing=True)
                        prefix_start.record()
                    else:
                        prefix_start = time.perf_counter()
                    prefix_output = self.actor_module(
                        input_ids=prefix_ids.unsqueeze(0),
                        attention_mask=prefix_attention_mask,
                        position_ids=prefix_positions.unsqueeze(0),
                        return_prefix_cache_only=True,
                    )
                    if use_cuda_events:
                        prefix_end.record()
                        prefix_event_pairs.append((prefix_start, prefix_end))
                    else:
                        prefix_cache_seconds += time.perf_counter() - prefix_start
                    prefix_cache = prefix_output.past_key_values
            finally:
                self.actor_module.train(was_training)

            if prefix_cache is None:
                raise RuntimeError("DiffusionGemma causal prefix pass did not return a KV cache")

            # Transformers drops past_key_values from checkpointed decoder
            # layers. The canvas is only 256 tokens, so retain its activations
            # and keep checkpointing enabled everywhere else.
            decoder_layers = [
                module
                for module in self.actor_module.modules()
                if module.__class__.__name__ == "DiffusionGemmaDecoderTextLayer"
            ]
            decoder_checkpointing = [
                bool(getattr(module, "gradient_checkpointing", False)) for module in decoder_layers
            ]
            for module in decoder_layers:
                module.gradient_checkpointing = False
            try:
                if use_cuda_events:
                    decoder_start = torch.cuda.Event(enable_timing=True)
                    decoder_end = torch.cuda.Event(enable_timing=True)
                    decoder_start.record()
                else:
                    decoder_start = time.perf_counter()
                with external_kv_context():
                    output = self.actor_module(
                        input_ids=None,
                        past_key_values=prefix_cache,
                        decoder_input_ids=noisy_canvas.unsqueeze(0),
                        decoder_position_ids=decoder_positions.unsqueeze(0),
                        self_conditioning_logits=None,
                    )
                if use_cuda_events:
                    decoder_end.record()
                    decoder_event_pairs.append((decoder_start, decoder_end))
                else:
                    decoder_seconds += time.perf_counter() - decoder_start
            finally:
                for module, checkpointing in zip(decoder_layers, decoder_checkpointing, strict=True):
                    module.gradient_checkpointing = checkpointing
            block_logits = output.logits.squeeze(0).float().div(effective_temperature)
            block_log_probs = logprobs_from_logits(
                block_logits.unsqueeze(0), clean_canvas.unsqueeze(0)
            ).squeeze(0)

            if block_width:
                active_width = active_canvas[:block_width]
                log_probs[batch_idx, block_start:block_end] = torch.where(
                    active_width,
                    block_log_probs[:block_width],
                    torch.zeros_like(block_log_probs[:block_width]),
                )
                if bool(active_width.any()):
                    block_ce = -block_log_probs[:block_width][active_width].sum()
                    ce_sum = block_ce if ce_sum is None else ce_sum + block_ce
                    ce_count += int(active_width.sum().item())
                if calculate_entropy:
                    block_entropy = verl_F.entropy_from_logits(block_logits)
                    entropy[batch_idx, block_start:block_end] = torch.where(
                        active_width,
                        block_entropy[:block_width],
                        torch.zeros_like(block_entropy[:block_width]),
                    )
                if calculate_sum_pi_squared:
                    block_probs = torch.softmax(block_logits, dim=-1)
                    block_sum_pi_squared = block_probs.square().sum(dim=-1)
                    sum_pi_squared[batch_idx, block_start:block_end] = torch.where(
                        active_width,
                        block_sum_pi_squared[:block_width],
                        torch.zeros_like(block_sum_pi_squared[:block_width]),
                    )

        if use_cuda_events:
            torch.cuda.synchronize(input_ids.device)
            prefix_cache_seconds = sum(start.elapsed_time(end) for start, end in prefix_event_pairs) / 1000.0
            decoder_seconds = sum(start.elapsed_time(end) for start, end in decoder_event_pairs) / 1000.0
        if ce_sum is None:
            ce_sum = next(self.actor_module.parameters()).sum() * 0.0
        denoise_ce = ce_sum / float(max(ce_count, 1))
        outputs = {"log_probs": log_probs}

        generation_jsd_config = self.config.get("generation_consistency_jsd", {})
        if generation_jsd_config.get("enable", False):
            objective = generation_jsd_config.get("objective", "trace_all_ce")
            if objective not in {"trace_all_ce", "gemma_denoise_ce"}:
                raise RuntimeError(
                    "DiffusionGemma smoke training supports only trace_all_ce/gemma_denoise_ce, "
                    f"got objective={objective!r}"
                )
            outputs["generation_jsd_loss"] = denoise_ce
            available_active_tokens = int(response_active.sum().detach().cpu().item())
            selected_histogram = {
                block_index: selected_blocks.count(block_index)
                for block_index in sorted(set(selected_blocks))
            }
            outputs["generation_jsd_metrics"] = {
                "generation_consistency_jsd/loss": float(denoise_ce.detach().item()),
                "generation_consistency_jsd/active_tokens": float(ce_count),
                "generation_consistency_jsd/active_frac": (
                    float(ce_count) / max(batch_size * response_length, 1)
                ),
                "generation_consistency_jsd/available_active_tokens": float(available_active_tokens),
                "generation_consistency_jsd/gemma_active_coverage_fraction": (
                    float(ce_count) / max(available_active_tokens, 1)
                ),
                "generation_consistency_jsd/gemma_block_denoise": 1.0,
                "generation_consistency_jsd/gemma_detached_prefix_cache": 1.0,
                "generation_consistency_jsd/gemma_encoder_only_prefix_cache": 1.0,
                "generation_consistency_jsd/gemma_selected_block_mean": (
                    float(sum(selected_blocks)) / max(len(selected_blocks), 1)
                ),
                "actor/diffusion_gemma/prefix_cache_seconds": float(prefix_cache_seconds),
                "actor/diffusion_gemma/decoder_seconds": float(decoder_seconds),
            }
            outputs["generation_jsd_metrics"].update(
                self._diffusion_gemma_external_kv_replay_metrics(
                    external_kv_stats,
                    decoder_forward_seconds=decoder_seconds,
                    decoder_backward_seconds=0.0,
                )
            )
            for block_index, count in selected_histogram.items():
                outputs["generation_jsd_metrics"][
                    f"generation_consistency_jsd/gemma_selected_block_{block_index}_count"
                ] = float(count)
        if calculate_entropy:
            outputs["entropys"] = entropy
        if calculate_sum_pi_squared:
            outputs["sum_pi_squared"] = sum_pi_squared
        return outputs

    @staticmethod
    def _repeat_diffusion_gemma_prefix_cache(prefix_cache, batch_size: int):
        batch_size = int(batch_size)
        if batch_size <= 1:
            return prefix_cache
        repeated = deepcopy(prefix_cache)
        repeat = getattr(repeated, "batch_repeat_interleave", None)
        if not callable(repeat):
            raise RuntimeError(
                "DiffusionGemma replay decoder chunks larger than one require a "
                "prefix Cache with batch_repeat_interleave()"
            )
        result = repeat(batch_size)
        return repeated if result is None else result

    @staticmethod
    def _clone_diffusion_gemma_prefix_cache_for_batch1(prefix_cache):
        try:
            return deepcopy(prefix_cache)
        except Exception as exc:
            raise RuntimeError(
                "DiffusionGemma full-row-isolated decoder replay could not clone "
                "the batch-1 prefix cache before a row forward"
            ) from exc

    @staticmethod
    def _diffusion_gemma_replay_prefix_grad_mode(
        generation_jsd_config: dict,
    ) -> str:
        mode = str(
            generation_jsd_config.get(
                "gemma_replay_prefix_grad_mode",
                "detached",
            )
        ).strip().lower()
        if mode not in {"detached", "full"}:
            raise ValueError(
                "generation_consistency_jsd.gemma_replay_prefix_grad_mode must "
                f"be 'detached' or 'full', got {mode!r}"
            )
        return mode

    def _diffusion_gemma_self_conditioning_module(self) -> nn.Module:
        modules = [
            module
            for module in self.actor_module.modules()
            if module.__class__.__name__ == "DiffusionGemmaSelfConditioning"
        ]
        if len(modules) != 1:
            raise RuntimeError(
                "DiffusionGemma replay expected exactly one self-conditioning module, "
                f"found {len(modules)}"
            )
        return modules[0]

    @staticmethod
    def _diffusion_gemma_replay_checkpoint_layers(
        model: nn.Module,
    ) -> list[nn.Module]:
        # The pinned overlay aliases encoder-layer module objects into the
        # decoder. Disable generic layer checkpointing on either class spelling
        # or the encoder drops the KV cache needed by the decoder.
        layer_names = {
            "DiffusionGemmaEncoderTextLayer",
            "DiffusionGemmaDecoderTextLayer",
        }
        return [
            module
            for module in model.modules()
            if module.__class__.__name__ in layer_names
        ]

    @staticmethod
    def _install_diffusion_gemma_self_conditioning_signal(
        module: nn.Module,
        signal: torch.Tensor,
    ):
        """Replace full-vocab logits with the exact rollout soft embedding."""

        def replace_signal(_module, args):
            if len(args) < 2:
                raise RuntimeError("DiffusionGemma self-conditioning hook received invalid arguments")
            exact_signal = signal.detach().to(device=args[0].device, dtype=args[0].dtype)
            return (args[0], exact_signal, *args[2:])

        return module.register_forward_pre_hook(replace_signal)

    def _update_diffusion_gemma_replay_tensor_chunk(
        self,
        model_inputs: dict[str, torch.Tensor],
        *,
        meta_info: dict,
        generation_jsd_config: dict,
        h2d_seconds: float,
        chunk_start: float,
    ) -> dict[str, float]:
        """Backward every captured Gemma row while retaining one graph chunk."""
        required = {
            "input_ids",
            "attention_mask",
            "responses",
            "response_mask",
            "gemma_replay_schema_version",
            "gemma_replay_canvas_ids",
            "gemma_replay_step_indices",
            "gemma_replay_row_valid",
            "gemma_replay_valid_masks",
            "gemma_replay_done_masks",
            "gemma_replay_self_conditioning_signal",
            "gemma_replay_self_conditioning_present",
            "gemma_replay_block_indices",
            "gemma_replay_block_real",
        }
        gemma_grpo_config = self.config.get("gemma_nostream_grpo", {}) or {}
        gemma_grpo_enabled = bool(gemma_grpo_config.get("enable", False))
        prefix_grad_mode = self._diffusion_gemma_replay_prefix_grad_mode(
            generation_jsd_config
        )
        gemma_baseline_config = self.config.get(
            "gemma_correct_only_baseline", {}
        ) or {}
        gemma_baseline_enabled = bool(
            gemma_baseline_config.get("enable", False)
        )
        if gemma_grpo_enabled and gemma_baseline_enabled:
            raise RuntimeError(
                "DiffusionGemma correct-only baseline cannot run with "
                "gemma_nostream_grpo"
            )
        if gemma_baseline_enabled:
            strict_contract = bool(gemma_baseline_config.get("strict_contract", True))
            if prefix_grad_mode != "full" and strict_contract:
                raise RuntimeError(
                    "DiffusionGemma correct-only baseline requires full prefix "
                    "and decoder gradients (set gemma_correct_only_baseline.strict_contract=false "
                    "to allow the detached prefix-KV replay)"
                )
            if not bool(
                self.config.get(
                    "use_diffusion_gemma_external_kv_replay", False
                )
            ):
                raise RuntimeError(
                    "DiffusionGemma correct-only baseline requires native "
                    "external-prefix KV replay"
                )
        if gemma_grpo_enabled:
            required.update(
                {
                    "gemma_replay_advantages",
                    "gemma_nostream_grpo_selected_rows",
                    "gemma_nostream_grpo_selected_action_tokens",
                }
            )
        missing = sorted(required - model_inputs.keys())
        if missing:
            raise RuntimeError(f"DiffusionGemma replay actor chunk is missing {missing}")
        if not self._is_diffusion_gemma_actor():
            raise RuntimeError("gemma_intermediate_replay_v1 was dispatched to a non-DiffusionGemma actor")

        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"].bool()
        targets = model_inputs["responses"].long()
        target_mask = model_inputs["response_mask"].bool()
        canvas_ids = model_inputs["gemma_replay_canvas_ids"].long()
        step_indices = model_inputs["gemma_replay_step_indices"].long()
        row_valid = model_inputs["gemma_replay_row_valid"].bool()
        valid_masks = model_inputs["gemma_replay_valid_masks"].bool()
        done_masks = model_inputs["gemma_replay_done_masks"].bool()
        self_conditioning = model_inputs["gemma_replay_self_conditioning_signal"]
        block_indices = model_inputs["gemma_replay_block_indices"].long()
        block_real = model_inputs["gemma_replay_block_real"].bool()
        row_advantages = (
            model_inputs["gemma_replay_advantages"].float()
            if gemma_grpo_enabled
            else None
        )
        selected_rows = (
            model_inputs["gemma_nostream_grpo_selected_rows"].bool()
            if gemma_grpo_enabled
            else None
        )
        schema_version = model_inputs["gemma_replay_schema_version"].long()
        if not bool(schema_version.eq(GEMMA_REPLAY_SCHEMA_VERSION).all()):
            raise RuntimeError("DiffusionGemma actor received an unsupported replay schema version")

        block_slots = int(input_ids.shape[0])
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise RuntimeError("DiffusionGemma replay prefix tensors must be rank-2 and aligned")
        if targets.ndim != 2 or target_mask.shape != targets.shape:
            raise RuntimeError("DiffusionGemma replay target tensors must be rank-2 and aligned")
        block_size = int(targets.shape[-1])
        if canvas_ids.ndim != 3 or tuple(canvas_ids.shape[::2]) != (block_slots, block_size):
            raise RuntimeError(
                "DiffusionGemma replay canvases must be [blocks, steps, block_size], "
                f"got {tuple(canvas_ids.shape)}"
            )
        max_steps = int(canvas_ids.shape[1])
        if tuple(step_indices.shape) != (block_slots, max_steps):
            raise RuntimeError("DiffusionGemma replay step indices do not align with canvases")
        if tuple(row_valid.shape) != (block_slots, max_steps):
            raise RuntimeError("DiffusionGemma replay row-valid mask does not align with canvases")
        if tuple(valid_masks.shape) != (block_slots, max_steps, block_size):
            raise RuntimeError("DiffusionGemma replay valid masks do not align with canvases")
        if done_masks.shape != valid_masks.shape:
            raise RuntimeError("DiffusionGemma replay done masks do not align with valid masks")
        if (
            self_conditioning.ndim != 4
            or tuple(self_conditioning.shape[:3]) != (block_slots, max_steps, block_size)
        ):
            raise RuntimeError(
                "DiffusionGemma replay self-conditioning must be "
                "[blocks, steps, block_size, hidden]"
            )
        if tuple(block_indices.shape) != (block_slots,) or tuple(block_real.shape) != (block_slots,):
            raise RuntimeError("DiffusionGemma replay block metadata must have one value per block slot")
        if gemma_grpo_enabled:
            if tuple(row_advantages.shape) != (block_slots, max_steps):
                raise RuntimeError(
                    "Gemma no-stream GRPO advantages must align with replay rows"
                )
            if tuple(selected_rows.shape) != (block_slots, max_steps):
                raise RuntimeError(
                    "Gemma no-stream GRPO selected-row mask must align with replay rows"
                )
            if not bool(self.config.get("use_diffusion_gemma_external_kv_replay", False)):
                raise RuntimeError(
                    "Gemma no-stream GRPO requires native external-prefix-KV replay"
                )
            if not bool(
                generation_jsd_config.get(
                    "gemma_replay_gpu_incremental_prefix_cache", False
                )
            ):
                raise RuntimeError(
                    "Gemma no-stream GRPO requires GPU incremental prefix cache"
                )
        self._dump_gemma_replay_block_fingerprint_debug(
            model_inputs,
            global_step=meta_info.get("global_step"),
            chunk_idx=meta_info.get("_t5_tensor_chunk_idx"),
        )

        row_valid = row_valid & block_real.unsqueeze(1)
        if gemma_grpo_enabled:
            selected_rows = selected_rows & block_real.unsqueeze(1)
            if not torch.equal(row_valid, selected_rows):
                raise RuntimeError(
                    "Gemma no-stream GRPO actor row-valid mask must contain exactly "
                    "the selected replay rows"
                )
            if not bool(torch.isfinite(row_advantages[row_valid]).all()):
                raise RuntimeError(
                    "Gemma no-stream GRPO selected advantages must be finite"
                )
        active_mask = valid_masks & target_mask.unsqueeze(1) & row_valid.unsqueeze(-1)
        expected_rows = int(row_valid.sum().detach().cpu().item())
        expected_active_tokens = int(active_mask.sum().detach().cpu().item())
        total_active_tokens = float(
            meta_info.get("_t5_tensor_total_active_tokens", 0.0) or 0.0
        )
        unknown_total_grad_rescale = bool(
            meta_info.get("_t5_tensor_unknown_total_grad_rescale", False)
        )
        provisional_denominator = float(
            meta_info.get("_t5_tensor_provisional_denominator", 0.0) or 0.0
        )
        use_provisional_denominator = (
            unknown_total_grad_rescale and provisional_denominator > 0.0
        )
        loss_scale_basis_active_tokens = 0.0
        loss_scale_basis_rows = 0.0
        if unknown_total_grad_rescale:
            basis = self._t5_tensor_grad_rescale_basis(meta_info)
            if basis == "rows":
                loss_scale_numerator = float(expected_rows)
                loss_scale_basis_rows = 1.0
            elif expected_active_tokens > 0:
                loss_scale_numerator = float(expected_active_tokens)
                loss_scale_basis_active_tokens = 1.0
            else:
                loss_scale_numerator = float(expected_rows)
                loss_scale_basis_rows = 1.0
            loss_scale_factor = (
                loss_scale_numerator / provisional_denominator
                if use_provisional_denominator
                else loss_scale_numerator
            )
        elif total_active_tokens > 0.0:
            loss_scale_factor = float(expected_active_tokens) / total_active_tokens
            loss_scale_basis_active_tokens = 1.0
        else:
            total_rows = max(
                int(meta_info.get("_t5_tensor_total_rows", expected_rows) or expected_rows),
                1,
            )
            loss_scale_factor = float(expected_rows) / float(total_rows)
            loss_scale_basis_rows = 1.0
        # Adjacent average entropy contraction on the canvas (see gemma_adjacent_entropy.py).
        # The finish step rescales every accumulated gradient by world_size * numerator /
        # final_active_tokens (unknown-total path) so the hinge term is pre-scaled to land on
        # coef * sum(hinge) / N_pairs_total after that rescale.
        gemma_aec_cfg = GemmaAdjacEntConfig.from_mapping(
            generation_jsd_config.get("gemma_adjacent_entropy_contraction", {}) or {}
        )
        gemma_aec_state = GemmaAecState()
        gemma_aec_loss_multiplier = 0.0
        gemma_aec_total_pairs = float(meta_info.get("_gemma_aec_total_pairs", 0.0) or 0.0)
        if gemma_aec_cfg.enable:
            if gemma_grpo_enabled:
                raise RuntimeError("gemma_adjacent_entropy_contraction cannot run with gemma_nostream_grpo")
            if gemma_aec_total_pairs <= 0.0:
                raise RuntimeError(
                    "gemma_adjacent_entropy_contraction is enabled but the trainer did not transport "
                    "_gemma_aec_total_pairs (> 0) in the replay meta_info"
                )
            if unknown_total_grad_rescale:
                aec_total_active_tokens = float(
                    meta_info.get("_gemma_aec_total_active_tokens", 0.0) or 0.0
                )
                if aec_total_active_tokens <= 0.0:
                    raise RuntimeError(
                        "gemma_adjacent_entropy_contraction needs _gemma_aec_total_active_tokens "
                        "when the replay uses the finish-time global rescale"
                    )
                numerator = provisional_denominator if use_provisional_denominator else 1.0
                gemma_aec_loss_multiplier = (
                    gemma_aec_cfg.coef * aec_total_active_tokens / (gemma_aec_total_pairs * numerator)
                )
            else:
                gemma_aec_loss_multiplier = gemma_aec_cfg.coef / gemma_aec_total_pairs
        real_rows_tensor = model_inputs.pop("_t5_tensor_real_rows", None)
        if torch.is_tensor(real_rows_tensor) and int(real_rows_tensor.numel()) > 0:
            transported_rows = int(real_rows_tensor.detach().cpu().long().sum().item())
            if transported_rows != expected_rows:
                raise RuntimeError(
                    "DiffusionGemma replay row count changed during FSDP dispatch: "
                    f"transported={transported_rows} derived={expected_rows}"
                )
        for block_slot in range(block_slots):
            real_steps = step_indices[block_slot][row_valid[block_slot]]
            if int(torch.unique(real_steps).numel()) != int(real_steps.numel()):
                raise RuntimeError(
                    f"DiffusionGemma replay block slot {block_slot} contains duplicate steps"
                )

        decoder_chunk_size = max(
            1,
            int(generation_jsd_config.get("gemma_replay_decoder_chunk_size", 1) or 1),
        )
        decoder_activation_offload = bool(
            generation_jsd_config.get(
                "gemma_replay_decoder_activation_offload",
                False,
            )
        )
        compact_step_tail = bool(
            generation_jsd_config.get("gemma_replay_compact_step_tail", True)
        )
        real_row_pack = bool(
            generation_jsd_config.get("gemma_replay_real_row_pack", False)
        )
        if gemma_aec_cfg.enable and not real_row_pack:
            raise RuntimeError(
                "gemma_adjacent_entropy_contraction requires gemma_replay_real_row_pack=true "
                "(waves must hold consecutive real states of one block)"
            )
        row_isolated_moe = bool(
            generation_jsd_config.get("gemma_replay_row_isolated_moe", False)
        )
        full_row_isolated_decoder = bool(
            generation_jsd_config.get("gemma_replay_full_row_isolated_decoder", False)
        )
        full_row_isolated_decoder_active = (
            full_row_isolated_decoder and decoder_chunk_size > 1
        )
        external_kv_stats, external_kv_context = (
            self._new_diffusion_gemma_external_kv_replay_context()
        )
        if gemma_grpo_enabled and external_kv_stats is None:
            raise RuntimeError(
                "Gemma no-stream GRPO requires the native no-copy "
                "external-prefix-KV replay context"
            )
        if prefix_grad_mode == "full" and external_kv_stats is None:
            raise RuntimeError(
                "DiffusionGemma full prefix-gradient replay requires the "
                "native no-copy external-prefix-KV replay context"
            )
        if external_kv_stats is not None and full_row_isolated_decoder_active:
            raise RuntimeError(
                "actor.use_diffusion_gemma_external_kv_replay cannot be combined "
                "with gemma_replay_full_row_isolated_decoder; external-KV replay "
                "already shares the original batch-1 prefix across decoder rows"
            )
        row_isolated_moe_active = row_isolated_moe and decoder_chunk_size > 1
        row_isolated_moe_context = nullcontext
        if row_isolated_moe_active:
            try:
                from transformers.models.diffusion_gemma.replay_row_isolated_moe import (
                    diffusion_gemma_decoder_row_isolated_moe,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "generation_consistency_jsd.gemma_replay_row_isolated_moe requires "
                    "a DiffusionGemma overlay that exports diffusion_gemma_decoder_row_isolated_moe"
                ) from exc
            row_isolated_moe_context = diffusion_gemma_decoder_row_isolated_moe
        # sc_hidden needed for ghost-chunk zero tensors in real_row_pack mode.
        sc_hidden = int(self_conditioning.shape[-1])
        raw_padded_rows = block_slots * max_steps
        if compact_step_tail:
            # Replay rows are sorted before transport, so padding is a suffix of
            # each block slot. FSDP ranks may own different samples, however,
            # and must execute the same number of model collectives. Synchronize
            # the last real row per slot and trim only the suffix that is padding
            # on every rank.
            row_positions = torch.arange(
                1,
                max_steps + 1,
                device=row_valid.device,
                dtype=torch.long,
            ).unsqueeze(0)
            synchronized_step_limits = torch.where(
                row_valid,
                row_positions,
                torch.zeros_like(row_positions),
            ).amax(dim=1)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(
                    synchronized_step_limits,
                    op=torch.distributed.ReduceOp.MAX,
                )
        else:
            synchronized_step_limits = torch.full(
                (block_slots,),
                max_steps,
                device=row_valid.device,
                dtype=torch.long,
            )
        if bool(
            synchronized_step_limits.lt(0).any()
            or synchronized_step_limits.gt(max_steps).any()
        ):
            raise RuntimeError(
                "DiffusionGemma replay synchronized step limits are outside "
                f"[0, {max_steps}]: {synchronized_step_limits.detach().cpu().tolist()}"
            )
        if real_row_pack:
            # For each block-slot, gather only the row indices where row_valid is
            # True (already accounts for block_real via the row_valid update at
            # line 565). Pack those real rows contiguously and chunk into waves of
            # decoder_chunk_size. Then synchronize the per-slot wave count across
            # FSDP ranks with ALL_REDUCE MAX so every rank executes the same
            # number of collectives (FSDP symmetry). Ranks that have fewer real
            # rows run ghost waves with all-False active_chunk and zero loss.
            slot_real_indices: list[torch.Tensor] = [
                torch.where(row_valid[s])[0]  # ascending, since row_valid is [0..max_steps-1]
                for s in range(block_slots)
            ]
            real_chunk_counts = torch.tensor(
                [
                    (len(idx) + decoder_chunk_size - 1) // decoder_chunk_size
                    if len(idx) > 0 else 0
                    for idx in slot_real_indices
                ],
                dtype=torch.long,
                device=row_valid.device,
            )
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(
                    real_chunk_counts,
                    op=torch.distributed.ReduceOp.MAX,
                )
            executed_decoder_rows = int(
                (real_chunk_counts * decoder_chunk_size).sum().item()
            )
        else:
            executed_decoder_rows = sum(
                (
                    (int(step_limit) + decoder_chunk_size - 1)
                    // decoder_chunk_size
                )
                * decoder_chunk_size
                for step_limit in synchronized_step_limits.detach().cpu().tolist()
            )
        effective_temperature = self._effective_logprob_temperature(
            float(meta_info.get("temperature", 1.0))
        )
        jsd_coef = float(generation_jsd_config.get("loss_coef", 1.0))
        ce_sum_loss_multiplier = (
            jsd_coef
            * float(loss_scale_factor)
            / float(max(expected_active_tokens, 1))
        )
        sc_module = self._diffusion_gemma_self_conditioning_module()
        sc_inject_handles, sc_inject_acc = _install_gemma_sc_inject_hooks(sc_module)
        decoder_layers = self._diffusion_gemma_replay_checkpoint_layers(
            self.actor_module
        )
        decoder_checkpointing = [
            bool(getattr(module, "gradient_checkpointing", False)) for module in decoder_layers
        ]
        for module in decoder_layers:
            module.gradient_checkpointing = False
        fp32_grad_accum_enabled = self._gemma_replay_fp32_grad_accum_enabled(generation_jsd_config)
        if fp32_grad_accum_enabled and not getattr(self, "_gemma_replay_fp32_grad_accum_active", False):
            self._reset_gemma_replay_fp32_grad_accum_window()

        use_cuda_events = bool(input_ids.is_cuda)
        timing_events: dict[str, list[tuple[Any, Any]]] = {
            "prefix_build": [],
            "prefix_extension": [],
            "decoder_forward": [],
            "decoder_backward": [],
        }
        timing_seconds = {key: 0.0 for key in timing_events}

        def timing_start():
            if use_cuda_events:
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                return event
            return time.perf_counter()

        def timing_end(name: str, start):
            if use_cuda_events:
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                timing_events[name].append((start, end))
            else:
                timing_seconds[name] += time.perf_counter() - start

        processed_rows = torch.zeros_like(row_valid)
        processed_active_tokens = 0
        ce_sum_value = 0.0
        prefix_cache_builds = 0
        full_prefix_decoder_forwards = 0
        decoder_forwards = 0
        decoder_backwards = 0
        full_row_isolated_decoder_row_forwards = 0
        real_blocks = int(block_real.sum().detach().cpu().item())
        block_row_counts: dict[int, int] = {}
        step_row_counts: dict[int, int] = {}
        ghost_chunk_rows_count = 0

        incr_cache_enabled = bool(
            generation_jsd_config.get("gemma_replay_gpu_incremental_prefix_cache", False)
        )
        if prefix_grad_mode == "full" and incr_cache_enabled:
            raise RuntimeError(
                "DiffusionGemma full prefix-gradient replay cannot use the "
                "detached incremental prefix KV cache"
            )
        existing_incr_cache = getattr(self, "_gemma_incr_prefix_kv_cache", None)
        if incr_cache_enabled and existing_incr_cache is None:
            max_entries = int(
                generation_jsd_config.get(
                    "gemma_replay_incremental_prefix_cache_max_entries",
                    DEFAULT_GEMMA_PREFIX_CACHE_MAX_ENTRIES,
                )
            )
            max_bytes = int(
                generation_jsd_config.get(
                    "gemma_replay_incremental_prefix_cache_max_bytes",
                    DEFAULT_GEMMA_PREFIX_CACHE_MAX_BYTES,
                )
            )
            self._gemma_incr_prefix_kv_cache = GemmaIncrementalPrefixKVCache(
                max_entries=max_entries,
                max_bytes=max_bytes,
            )
        incr_cache = (
            getattr(self, "_gemma_incr_prefix_kv_cache", None)
            if incr_cache_enabled
            else None
        )

        incr_metrics_before = (
            incr_cache.counters_snapshot()
            if incr_cache is not None
            else {}
        )
        cache_metadata: list[list[int]] | None = None
        if incr_cache is not None:
            identity_hashes = model_inputs.get("gemma_replay_identity_hashes")
            if not torch.is_tensor(identity_hashes) or tuple(identity_hashes.shape) != (
                block_slots,
                2,
            ):
                incr_cache.invalidate()
                raise RuntimeError(
                    "DiffusionGemma incremental prefix cache requires "
                    "gemma_replay_identity_hashes=[block_slots, 2]"
                )
            # One tiny D2H transfer/synchronization per replay chunk supplies all
            # Python dict keys. Prefix token validation itself remains on-device.
            try:
                cache_metadata = (
                    torch.cat(
                        (
                            identity_hashes.long(),
                            block_indices.unsqueeze(1),
                            block_real.long().unsqueeze(1),
                        ),
                        dim=1,
                    )
                    .detach()
                    .cpu()
                    .tolist()
                )
            except Exception:
                incr_cache.invalidate()
                raise

        try:
            for block_slot in range(block_slots):
                # Determine whether this slot has any work to do and set up the
                # inner chunk iterator. real_row_pack packs only real rows (all
                # row_valid=True positions) into contiguous chunks, skipping
                # interior padding gaps. The sequential path iterates step
                # positions 0..step_limit in decoder_chunk_size strides (the
                # existing compact_step_tail / full-range behaviour).
                if real_row_pack:
                    num_packed_chunks = int(real_chunk_counts[block_slot].item())
                    if num_packed_chunks == 0:
                        continue
                    real_idx_for_slot = slot_real_indices[block_slot]
                    if gemma_aec_cfg.enable and real_idx_for_slot.numel() > 1:
                        # Rows are sorted by ascending Gemma4Renoise `step`, which counts DOWN
                        # (48 -> 1), i.e. ascending index = reverse chronological order. AEC pairs
                        # (t -> t+1) need chronological waves (the "next" state must carry the
                        # gradient), so walk the real rows in descending step index. The CE is
                        # order-independent and the prefix cache is per block, so this is safe.
                        real_idx_for_slot = real_idx_for_slot.flip(0)
                    n_real_for_slot = len(real_idx_for_slot)
                    inner_range = range(num_packed_chunks)
                else:
                    step_limit = int(
                        synchronized_step_limits[block_slot].detach().cpu().item()
                    )
                    if step_limit <= 0:
                        continue
                    inner_range = range(0, step_limit, decoder_chunk_size)

                prefix_ids = input_ids[block_slot][attention_mask[block_slot]]
                if int(prefix_ids.numel()) <= 0:
                    raise RuntimeError("DiffusionGemma replay received an empty prefix slot")
                prefix_length = int(prefix_ids.numel())
                prefix_positions = torch.arange(
                    prefix_length,
                    device=input_ids.device,
                    dtype=torch.long,
                )
                prefix_attention_mask = torch.ones(
                    (1, prefix_length),
                    device=input_ids.device,
                    dtype=attention_mask.dtype,
                )

                if cache_metadata is not None:
                    lane0, lane1, block_index_for_slot, block_is_real = cache_metadata[
                        block_slot
                    ]
                    replay_identity = (int(lane0), int(lane1))
                    block_index_for_slot = int(block_index_for_slot)
                    block_is_real = bool(block_is_real)
                else:
                    replay_identity = (0, 0)
                    block_index_for_slot = int(block_indices[block_slot].item())
                    block_is_real = bool(block_real[block_slot])

                prefix_cache = None
                can_cache = (
                    incr_cache is not None
                    and block_is_real
                    and prefix_grad_mode == "detached"
                )
                if can_cache:
                    cached_entry = incr_cache.lookup_for_extension(
                        replay_identity,
                        block_index_for_slot,
                        prefix_ids,
                    )
                    if cached_entry is not None:
                        cached_len = int(cached_entry.prefix_ids.numel())
                        cached_kv = cached_entry.kv_cache
                        del cached_entry
                        delta_ids = prefix_ids[cached_len:]
                        delta_positions = prefix_positions[cached_len:]
                        was_training = self.actor_module.training
                        self.actor_module.eval()
                        try:
                            with torch.no_grad():
                                start = timing_start()
                                prefix_output = self.actor_module(
                                    input_ids=delta_ids.unsqueeze(0),
                                    # Mask covers cached tokens plus the new delta.
                                    attention_mask=prefix_attention_mask,
                                    position_ids=delta_positions.unsqueeze(0),
                                    past_key_values=cached_kv,
                                    return_prefix_cache_only=True,
                                )
                                timing_end("prefix_extension", start)
                                prefix_cache = prefix_output.past_key_values
                        finally:
                            self.actor_module.train(was_training)
                        if prefix_cache is not None:
                            incr_cache.record_extension(
                                appended_tokens=prefix_length - cached_len,
                                avoided_prefix_tokens=cached_len,
                            )
                            incr_cache.evict(
                                replay_identity,
                                block_index_for_slot - 1,
                            )
                        else:
                            incr_cache.invalidate_identity(replay_identity)
                        del cached_kv

                if prefix_cache is None and prefix_grad_mode == "detached":
                    was_training = self.actor_module.training
                    self.actor_module.eval()
                    try:
                        with torch.no_grad():
                            start = timing_start()
                            prefix_output = self.actor_module(
                                input_ids=prefix_ids.unsqueeze(0),
                                attention_mask=prefix_attention_mask,
                                position_ids=prefix_positions.unsqueeze(0),
                                return_prefix_cache_only=True,
                            )
                            timing_end("prefix_build", start)
                            prefix_cache = prefix_output.past_key_values
                    finally:
                        self.actor_module.train(was_training)
                    if can_cache:
                        incr_cache.record_build()

                if prefix_grad_mode == "detached" and prefix_cache is None:
                    raise RuntimeError("DiffusionGemma replay prefix pass returned no detached KV cache")
                if prefix_grad_mode == "detached":
                    prefix_cache_builds += 1

                if can_cache:
                    incr_cache.store(
                        replay_identity,
                        block_index_for_slot,
                        prefix_ids,
                        prefix_cache,
                    )

                decoder_positions = torch.arange(
                    prefix_length,
                    prefix_length + block_size,
                    device=input_ids.device,
                    dtype=torch.long,
                )

                for loop_idx in inner_range:
                    # Build canvas_chunk / sc_chunk / active_chunk for this wave.
                    # real_row_pack: select packed real rows (non-contiguous in
                    #   the step-axis); ghost waves (loop_idx * cs >= n_real)
                    #   use all-zero tensors so active_chunk is all-False and
                    #   contributes zero loss while keeping FSDP collective
                    #   symmetry with slower ranks.
                    # sequential path: take a contiguous step slice as before.
                    if real_row_pack:
                        pack_start = loop_idx * decoder_chunk_size
                        pack_end = min(pack_start + decoder_chunk_size, n_real_for_slot)
                        actual_rows = max(0, pack_end - pack_start)
                        if actual_rows > 0:
                            selected = real_idx_for_slot[pack_start:pack_end]
                            canvas_chunk = canvas_ids[block_slot][selected]
                            sc_chunk = self_conditioning[block_slot][selected]
                            active_chunk = active_mask[block_slot][selected]
                            index_chunk = step_indices[block_slot][selected]
                            advantage_chunk = (
                                row_advantages[block_slot][selected]
                                if gemma_grpo_enabled
                                else None
                            )
                        else:
                            # Ghost wave: this rank has fewer real rows than the
                            # FSDP-synchronized maximum for this slot. Run a
                            # fully-masked (all-False active_chunk) forward+backward
                            # to keep collective counts equal across ranks.
                            selected = real_idx_for_slot[:0]
                            canvas_chunk = canvas_ids.new_zeros(0, block_size)
                            sc_chunk = self_conditioning.new_zeros(0, block_size, sc_hidden)
                            active_chunk = torch.zeros(
                                0, block_size, dtype=torch.bool, device=row_valid.device
                            )
                            index_chunk = step_indices.new_zeros(0)
                            advantage_chunk = (
                                row_advantages.new_zeros(0)
                                if gemma_grpo_enabled
                                else None
                            )
                        pad_rows = decoder_chunk_size - actual_rows
                    else:
                        step_start = loop_idx
                        step_end = min(step_start + decoder_chunk_size, step_limit)
                        actual_rows = step_end - step_start
                        canvas_chunk = canvas_ids[block_slot, step_start:step_end]
                        sc_chunk = self_conditioning[block_slot, step_start:step_end]
                        active_chunk = active_mask[block_slot, step_start:step_end]
                        row_chunk_seq = row_valid[block_slot, step_start:step_end]
                        index_chunk = step_indices[block_slot, step_start:step_end]
                        advantage_chunk = (
                            row_advantages[block_slot, step_start:step_end]
                            if gemma_grpo_enabled
                            else None
                        )
                        pad_rows = decoder_chunk_size - actual_rows

                    if pad_rows > 0:
                        canvas_chunk = torch.cat(
                            (
                                canvas_chunk,
                                torch.zeros(
                                    (pad_rows, block_size),
                                    device=canvas_chunk.device,
                                    dtype=canvas_chunk.dtype,
                                ),
                            ),
                            dim=0,
                        )
                        sc_chunk = torch.cat(
                            (
                                sc_chunk,
                                torch.zeros(
                                    (pad_rows, block_size, sc_hidden),
                                    device=sc_chunk.device,
                                    dtype=sc_chunk.dtype,
                                ),
                            ),
                            dim=0,
                        )
                        active_chunk = torch.cat(
                            (
                                active_chunk,
                                torch.zeros(
                                    (pad_rows, block_size),
                                    device=active_chunk.device,
                                    dtype=torch.bool,
                                ),
                            ),
                            dim=0,
                        )
                        if gemma_grpo_enabled:
                            advantage_chunk = torch.cat(
                                (
                                    advantage_chunk,
                                    advantage_chunk.new_zeros(pad_rows),
                                ),
                                dim=0,
                            )

                    sc_module._sc_inject_rows = int(actual_rows)
                    if full_row_isolated_decoder_active:
                        row_logits = []
                        for row_offset in range(decoder_chunk_size):
                            row_cache = (
                                None
                                if prefix_grad_mode == "full"
                                else self._clone_diffusion_gemma_prefix_cache_for_batch1(
                                    prefix_cache
                                )
                            )
                            hook = self._install_diffusion_gemma_self_conditioning_signal(
                                sc_module,
                                sc_chunk[row_offset : row_offset + 1],
                            )
                            try:
                                start = timing_start()
                                with external_kv_context(), row_isolated_moe_context():
                                    if prefix_grad_mode == "full":
                                        row_output = self.actor_module(
                                            input_ids=prefix_ids.unsqueeze(0),
                                            attention_mask=prefix_attention_mask,
                                            position_ids=prefix_positions.unsqueeze(0),
                                            decoder_input_ids=canvas_chunk[
                                                row_offset : row_offset + 1
                                            ],
                                            decoder_position_ids=decoder_positions.unsqueeze(0),
                                            self_conditioning_logits=None,
                                            decoder_activation_offload=decoder_activation_offload,
                                        )
                                    else:
                                        row_output = self.actor_module(
                                            input_ids=None,
                                            past_key_values=row_cache,
                                            decoder_input_ids=canvas_chunk[
                                                row_offset : row_offset + 1
                                            ],
                                            decoder_position_ids=decoder_positions.unsqueeze(0),
                                            self_conditioning_logits=None,
                                            decoder_activation_offload=decoder_activation_offload,
                                        )
                                timing_end("decoder_forward", start)
                            finally:
                                hook.remove()
                            decoder_forwards += 1
                            if prefix_grad_mode == "full":
                                prefix_cache_builds += 1
                                full_prefix_decoder_forwards += 1
                            full_row_isolated_decoder_row_forwards += 1
                            row_logits.append(row_output.logits)
                            del row_output, row_cache
                        logits = torch.cat(row_logits, dim=0).float().div(
                            effective_temperature
                        )
                        del row_logits
                    else:
                        if prefix_grad_mode == "full":
                            chunk_cache = None
                        elif external_kv_stats is not None:
                            chunk_cache = prefix_cache
                            assert chunk_cache is prefix_cache
                        else:
                            chunk_cache = self._repeat_diffusion_gemma_prefix_cache(
                                prefix_cache,
                                decoder_chunk_size,
                            )
                        hook = self._install_diffusion_gemma_self_conditioning_signal(
                            sc_module,
                            sc_chunk,
                        )
                        try:
                            start = timing_start()
                            with external_kv_context(), row_isolated_moe_context():
                                if prefix_grad_mode == "full":
                                    output = self.actor_module(
                                        input_ids=prefix_ids.unsqueeze(0),
                                        attention_mask=prefix_attention_mask,
                                        position_ids=prefix_positions.unsqueeze(0),
                                        decoder_input_ids=canvas_chunk,
                                        decoder_position_ids=decoder_positions.unsqueeze(0).expand(
                                            decoder_chunk_size,
                                            -1,
                                        ),
                                        self_conditioning_logits=None,
                                        decoder_activation_offload=decoder_activation_offload,
                                    )
                                else:
                                    output = self.actor_module(
                                        input_ids=None,
                                        past_key_values=chunk_cache,
                                        decoder_input_ids=canvas_chunk,
                                        decoder_position_ids=decoder_positions.unsqueeze(0).expand(
                                            decoder_chunk_size,
                                            -1,
                                        ),
                                        self_conditioning_logits=None,
                                        decoder_activation_offload=decoder_activation_offload,
                                    )
                            timing_end("decoder_forward", start)
                        finally:
                            hook.remove()
                        decoder_forwards += 1
                        if prefix_grad_mode == "full":
                            prefix_cache_builds += 1
                            full_prefix_decoder_forwards += 1

                        logits = output.logits.float().div(effective_temperature)
                    target_chunk = targets[block_slot].unsqueeze(0).expand(
                        decoder_chunk_size,
                        -1,
                    )
                    safe_targets = torch.where(
                        active_chunk,
                        target_chunk,
                        torch.zeros_like(target_chunk),
                    )
                    token_log_probs = logprobs_from_logits(logits, safe_targets)
                    if gemma_grpo_enabled:
                        ce_sum = -(
                            token_log_probs
                            * active_chunk.to(token_log_probs.dtype)
                            * advantage_chunk.unsqueeze(-1).to(token_log_probs.dtype)
                        ).sum()
                    else:
                        ce_sum = -token_log_probs.masked_select(active_chunk).sum()
                    if full_row_isolated_decoder_active:
                        # Keep padded/ghost row graphs attached to this chunk's
                        # single backward without changing the CE value.
                        ce_sum = ce_sum + token_log_probs.sum() * 0.0
                    elif not bool(active_chunk.any()):
                        ce_sum = token_log_probs.sum() * 0.0
                    loss = ce_sum * ce_sum_loss_multiplier
                    if gemma_aec_cfg.enable and actual_rows > 0:
                        aec_hinge_sum = gemma_aec_wave_terms(
                            logits=logits,
                            canvas_chunk=canvas_chunk,
                            index_chunk=index_chunk,
                            actual_rows=int(actual_rows),
                            block_slot=int(block_slot),
                            cfg=gemma_aec_cfg,
                            state=gemma_aec_state,
                        )
                        if aec_hinge_sum is not None:
                            loss = loss + aec_hinge_sum * gemma_aec_loss_multiplier
                    start = timing_start()
                    if getattr(self, "scaler", None) is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    timing_end("decoder_backward", start)
                    decoder_backwards += 1
                    if fp32_grad_accum_enabled:
                        self._clear_gemma_replay_public_fsdp_grads()

                    chunk_active_tokens = int(active_chunk.sum().detach().cpu().item())
                    processed_active_tokens += chunk_active_tokens
                    ce_sum_value += float(ce_sum.detach().cpu().item())

                    if real_row_pack:
                        # Mark only the real rows we actually indexed; ghost
                        # waves contribute nothing to processed_rows.
                        if actual_rows > 0:
                            processed_rows[block_slot, selected] = True
                        # Accumulate ghost rows (intra-chunk padding + full
                        # ghost waves) for metrics.
                        ghost_chunk_rows_count += pad_rows
                        if bool(block_real[block_slot]) and actual_rows > 0:
                            block_index = int(block_indices[block_slot].detach().cpu().item())
                            block_row_counts[block_index] = (
                                block_row_counts.get(block_index, 0) + actual_rows
                            )
                            for step_index in index_chunk.detach().cpu().tolist():
                                step_row_counts[int(step_index)] = (
                                    step_row_counts.get(int(step_index), 0) + 1
                                )
                    else:
                        processed_rows[block_slot, step_start:step_end] = True
                        if bool(block_real[block_slot]):
                            block_index = int(block_indices[block_slot].detach().cpu().item())
                            block_row_counts[block_index] = (
                                block_row_counts.get(block_index, 0)
                                + int(row_chunk_seq.sum().detach().cpu().item())
                            )
                            for step_index in index_chunk[row_chunk_seq].detach().cpu().tolist():
                                step_row_counts[int(step_index)] = (
                                    step_row_counts.get(int(step_index), 0) + 1
                                )

                    if not full_row_isolated_decoder_active:
                        del output, chunk_cache
                    del logits, token_log_probs, ce_sum, loss
        except Exception:
            if fp32_grad_accum_enabled:
                self._cleanup_gemma_replay_fp32_grad_accum()
            # Invalidate the incremental prefix KV cache on any exception so
            # stale GPU tensors from a partially completed update are not reused.
            self.invalidate_gemma_prefix_kv_cache()
            raise
        finally:
            for module, checkpointing in zip(decoder_layers, decoder_checkpointing, strict=True):
                module.gradient_checkpointing = checkpointing

        if use_cuda_events:
            torch.cuda.synchronize(input_ids.device)
            for name, pairs in timing_events.items():
                timing_seconds[name] = sum(
                    start.elapsed_time(end) for start, end in pairs
                ) / 1000.0
        prefix_seconds = (
            timing_seconds["prefix_build"]
            + timing_seconds["prefix_extension"]
        )
        if incr_cache is not None:
            incr_cache.record_timing(
                build_seconds=timing_seconds["prefix_build"],
                extension_seconds=timing_seconds["prefix_extension"],
            )

        if external_kv_stats is not None and hasattr(
            external_kv_stats,
            "calls",
        ):
            external_calls = int(external_kv_stats.calls)
            triton_calls = int(external_kv_stats.backend_triton_calls)
            fallbacks = int(external_kv_stats.fallbacks)
            if (
                decoder_forwards > 0
                and (
                    external_calls <= 0
                    or triton_calls != external_calls
                    or fallbacks != 0
                )
            ):
                self.invalidate_gemma_prefix_kv_cache()
                raise RuntimeError(
                    "DiffusionGemma decoder replay did not stay on the no-copy "
                    "external-KV Triton path: "
                    f"decoder_forwards={decoder_forwards} "
                    f"calls={external_calls} triton_calls={triton_calls} "
                    f"fallbacks={fallbacks}"
                )

        if not torch.equal(processed_rows & row_valid, row_valid):
            missing_rows = int((row_valid & ~processed_rows).sum().detach().cpu().item())
            self.invalidate_gemma_prefix_kv_cache()
            raise RuntimeError(f"DiffusionGemma replay skipped {missing_rows} intermediate rows")
        processed_real_rows = int((processed_rows & row_valid).sum().detach().cpu().item())
        if processed_real_rows != expected_rows or processed_active_tokens != expected_active_tokens:
            self.invalidate_gemma_prefix_kv_cache()
            raise RuntimeError(
                "DiffusionGemma actor replay accounting mismatch: "
                f"rows={processed_real_rows}/{expected_rows} "
                f"tokens={processed_active_tokens}/{expected_active_tokens}"
            )

        loss_value = ce_sum_value / max(expected_active_tokens, 1)
        empty_cache_seconds = 0.0
        if bool(generation_jsd_config.get("stream_replay_empty_cache", True)):
            empty_cache_start = time.perf_counter()
            torch.cuda.empty_cache()
            gc.collect()
            empty_cache_seconds = time.perf_counter() - empty_cache_start
        chunk_total_seconds = time.perf_counter() - chunk_start
        selected_advantages = (
            row_advantages[row_valid]
            if gemma_grpo_enabled
            else None
        )
        metrics: dict[str, float] = {
            "actor/pg_loss": float(loss_value) if gemma_grpo_enabled else 0.0,
            "actor/t5_tensor_replay/loss": float(loss_value),
            "actor/t5_tensor_replay/active_tokens": float(expected_active_tokens),
            "actor/t5_tensor_replay/rows": float(expected_rows),
            "actor/t5_tensor_replay/padded_rows": float(executed_decoder_rows),
            "actor/t5_tensor_replay/raw_padded_rows": float(raw_padded_rows),
            "actor/t5_tensor_replay/trimmed_padding_rows": float(
                raw_padded_rows - executed_decoder_rows
            ),
            "actor/t5_tensor_replay/loss_scale": float(loss_scale_factor),
            "actor/t5_tensor_replay/unknown_total_grad_rescale": (
                1.0 if unknown_total_grad_rescale else 0.0
            ),
            "actor/t5_tensor_replay/provisional_denominator": float(
                provisional_denominator
            ),
            "actor/t5_tensor_replay/use_provisional_denominator": (
                1.0 if use_provisional_denominator else 0.0
            ),
            "actor/t5_tensor_replay/loss_scale_basis_active_tokens": float(
                loss_scale_basis_active_tokens
            ),
            "actor/t5_tensor_replay/loss_scale_basis_rows": float(
                loss_scale_basis_rows
            ),
            "actor/t5_tensor_replay/jsd_only": 0.0 if gemma_grpo_enabled else 1.0,
            "actor/t5_tensor_replay/h2d_seconds": float(h2d_seconds),
            "actor/t5_tensor_replay/forward_seconds": float(
                prefix_seconds + timing_seconds["decoder_forward"]
            ),
            "actor/t5_tensor_replay/backward_seconds": float(timing_seconds["decoder_backward"]),
            "actor/t5_tensor_replay/loss_prep_seconds": 0.0,
            "actor/t5_tensor_replay/empty_cache_seconds": float(empty_cache_seconds),
            "actor/t5_tensor_replay/chunk_total_seconds": float(chunk_total_seconds),
            "generation_consistency_jsd/loss": float(loss_value),
            "generation_consistency_jsd/active_tokens": float(expected_active_tokens),
            "generation_consistency_jsd/gemma_intermediate_replay": 1.0,
            "generation_consistency_jsd/gemma_detached_prefix_cache": (
                0.0 if prefix_grad_mode == "full" else 1.0
            ),
            "actor/gemma_correct_only_baseline/enabled": (
                1.0 if gemma_baseline_enabled else 0.0
            ),
            "actor/gemma_correct_only_baseline/grpo_enabled": 0.0,
            "actor/gemma_correct_only_baseline/tpf_weighting_enabled": 0.0,
            "actor/gemma_correct_only_baseline/global_active_tokens": (
                float(expected_active_tokens)
                if gemma_baseline_enabled
                else 0.0
            ),
            "actor/gemma_correct_only_baseline/global_token_mean_loss": (
                float(loss_value) if gemma_baseline_enabled else 0.0
            ),
            "actor/diffusion_gemma/total_blocks": float(real_blocks),
            "actor/diffusion_gemma/intermediate_rows": float(expected_rows),
            "actor/diffusion_gemma/active_token_steps": float(expected_active_tokens),
            "actor/diffusion_gemma/prefix_cache_builds": float(prefix_cache_builds),
            "actor/diffusion_gemma/prefix_cache_seconds": float(prefix_seconds),
            "actor/diffusion_gemma/prefix_grad_mode_full": (
                1.0 if prefix_grad_mode == "full" else 0.0
            ),
            "actor/diffusion_gemma/prefix_grad_mode_detached": (
                1.0 if prefix_grad_mode == "detached" else 0.0
            ),
            "actor/diffusion_gemma/full_prefix_decoder_forwards": float(
                full_prefix_decoder_forwards
            ),
            "actor/diffusion_gemma/decoder_forwards": float(decoder_forwards),
            "actor/diffusion_gemma/decoder_backwards": float(decoder_backwards),
            "actor/diffusion_gemma/decoder_forward_seconds": float(
                timing_seconds["decoder_forward"]
            ),
            "actor/diffusion_gemma/decoder_backward_seconds": float(
                timing_seconds["decoder_backward"]
            ),
            "actor/gemma_nostream_grpo/enabled": (
                1.0 if gemma_grpo_enabled else 0.0
            ),
            "actor/gemma_nostream_grpo/selected_rows": (
                float(expected_rows) if gemma_grpo_enabled else 0.0
            ),
            "actor/gemma_nostream_grpo/selected_action_tokens": (
                float(expected_active_tokens) if gemma_grpo_enabled else 0.0
            ),
            "actor/gemma_nostream_grpo/positive_rows": (
                float((selected_advantages > 0).sum().detach().cpu().item())
                if gemma_grpo_enabled
                else 0.0
            ),
            "actor/gemma_nostream_grpo/negative_rows": (
                float((selected_advantages < 0).sum().detach().cpu().item())
                if gemma_grpo_enabled
                else 0.0
            ),
            "actor/gemma_nostream_grpo/zero_rows": (
                float((selected_advantages == 0).sum().detach().cpu().item())
                if gemma_grpo_enabled
                else 0.0
            ),
            "actor/diffusion_gemma/decoder_chunk_size": float(decoder_chunk_size),
            "actor/diffusion_gemma/decoder_activation_offload_enabled": (
                1.0 if decoder_activation_offload else 0.0
            ),
            "actor/diffusion_gemma/compact_step_tail_enabled": (
                1.0 if compact_step_tail else 0.0
            ),
            "actor/diffusion_gemma/real_row_pack_enabled": (
                1.0 if real_row_pack else 0.0
            ),
            "actor/diffusion_gemma/row_isolated_moe_enabled": (
                1.0 if row_isolated_moe else 0.0
            ),
            "actor/diffusion_gemma/row_isolated_moe_active": (
                1.0 if row_isolated_moe_active else 0.0
            ),
            "actor/diffusion_gemma/full_row_isolated_decoder_configured": (
                1.0 if full_row_isolated_decoder else 0.0
            ),
            "actor/diffusion_gemma/full_row_isolated_decoder_active": (
                1.0 if full_row_isolated_decoder_active else 0.0
            ),
            "actor/diffusion_gemma/full_row_isolated_decoder_row_forwards": float(
                full_row_isolated_decoder_row_forwards
            ),
            "actor/diffusion_gemma/full_row_isolated_decoder_backward_calls": float(
                decoder_backwards if full_row_isolated_decoder_active else 0
            ),
            "actor/diffusion_gemma/raw_step_slots": float(raw_padded_rows),
            "actor/diffusion_gemma/executed_step_slots": float(
                executed_decoder_rows
            ),
            "actor/diffusion_gemma/trimmed_step_slots": float(
                raw_padded_rows - executed_decoder_rows
            ),
            # ghost_chunk_rows counts decoder rows executed with all-False
            # active_chunk (intra-chunk padding + FSDP ghost waves). Only
            # nonzero when real_row_pack=True.
            "actor/diffusion_gemma/ghost_chunk_rows": float(ghost_chunk_rows_count),
            "actor/diffusion_gemma/row_coverage": 1.0,
            "actor/diffusion_gemma/token_coverage": 1.0,
            "actor/diffusion_gemma/coverage_percent": 100.0,
            "actor/diffusion_gemma/clean_filter_enabled": 0.0,
            "actor/diffusion_gemma/reward_gating_enabled": 0.0,
            "actor/diffusion_gemma/advantage_gating_enabled": 0.0,
        }
        metrics.update(
            self._diffusion_gemma_external_kv_replay_metrics(
                external_kv_stats,
                decoder_forward_seconds=timing_seconds["decoder_forward"],
                decoder_backward_seconds=timing_seconds["decoder_backward"],
            )
        )
        for handle in sc_inject_handles:
            handle.remove()
        sc_module._sc_inject_rows = 0
        metrics.update(_gemma_sc_inject_metrics(sc_inject_acc))
        if gemma_aec_cfg.enable:
            metrics.update(
                gemma_aec_metrics(
                    gemma_aec_state,
                    cfg=gemma_aec_cfg,
                    total_pairs_hint=gemma_aec_total_pairs,
                    # this call's share of coef * sum(hinge) / N_pairs_total
                    loss_value=gemma_aec_cfg.coef * gemma_aec_state.hinge_sum / max(gemma_aec_total_pairs, 1.0),
                )
            )
        metrics.update(
            {
                "actor/diffusion_gemma/external_kv/prefix_grad_mode_full": (
                    1.0
                    if external_kv_stats is not None
                    and prefix_grad_mode == "full"
                    else 0.0
                ),
                "actor/diffusion_gemma/external_kv/prefix_grad_mode_detached": (
                    1.0
                    if external_kv_stats is not None
                    and prefix_grad_mode == "detached"
                    else 0.0
                ),
            }
        )
        if fp32_grad_accum_enabled:
            metrics.update(self._gemma_replay_fp32_grad_accum_metrics())
        for block_index, count in sorted(block_row_counts.items()):
            metrics[f"actor/diffusion_gemma/block_{block_index}/rows"] = float(count)
        for step_index, count in sorted(step_row_counts.items()):
            metrics[f"actor/diffusion_gemma/step_{step_index}/rows"] = float(count)
        if incr_cache is not None:
            metrics.update(incr_cache.metrics_snapshot())
            prefix = "actor/diffusion_gemma/prefix_kv_cache"
            for key, value in incr_cache.counters_snapshot().items():
                metrics[f"{prefix}/chunk_{key}"] = float(
                    value - incr_metrics_before[key]
                )
        else:
            metrics["actor/diffusion_gemma/prefix_kv_cache/enabled"] = 0.0
        return metrics

    def set_replica_grad_sync_group(self, process_group, group_size: int = 1):
        """Configure explicit delayed replica-gradient synchronization."""
        self._replica_grad_sync_group = process_group
        self._replica_grad_sync_group_size = max(1, int(group_size or 1))

    def _sync_replica_grads_once(self) -> None:
        group = getattr(self, "_replica_grad_sync_group", None)
        group_size = int(getattr(self, "_replica_grad_sync_group_size", 1) or 1)
        if group is None or group_size <= 1:
            self._last_replica_grad_sync_tensors = 0
            self._last_replica_grad_sync_bytes = 0
            self._last_replica_grad_sync_seconds = 0.0
            return
        sync_start = time.perf_counter()
        tensor_count = 0
        byte_count = 0
        for param in self.actor_module.parameters():
            grad = param.grad
            if grad is None:
                continue
            if isinstance(grad, DTensor):
                grad = grad.to_local()
            if getattr(grad, "is_sparse", False):
                raise RuntimeError("replica gradient sync does not support sparse gradients")
            torch.distributed.all_reduce(grad, op=torch.distributed.ReduceOp.SUM, group=group)
            grad.div_(group_size)
            tensor_count += 1
            byte_count += int(grad.numel() * grad.element_size())
        self._last_replica_grad_sync_tensors = tensor_count
        self._last_replica_grad_sync_bytes = byte_count
        self._last_replica_grad_sync_seconds = time.perf_counter() - sync_start

    def _init_gemma_replay_fp32_grad_accum_state(self) -> None:
        self._gemma_replay_fp32_grad_accum: dict[int, torch.Tensor] = {}
        self._gemma_replay_fp32_grad_accum_meta: dict[int, dict[str, Any]] = {}
        self._gemma_replay_fp32_grad_accum_templates: dict[int, Any] = {}
        self._gemma_replay_fp32_grad_accum_param_refs: dict[int, weakref.ReferenceType] = {}
        self._gemma_replay_fp32_grad_accum_registered_handles: set[int] = set()
        self._gemma_replay_fp32_grad_accum_captured_handles: set[int] = set()
        self._gemma_replay_fp32_grad_accum_capture_counts: dict[int, int] = {}
        self._gemma_replay_fp32_grad_accum_calls = 0
        self._gemma_replay_fp32_grad_accum_duplicate_captures = 0
        self._gemma_replay_fp32_grad_accum_materialized_grads = 0
        self._gemma_replay_fp32_grad_accum_materialize_seconds = 0.0
        self._gemma_replay_fp32_grad_accum_cleanup_seconds = 0.0
        self._gemma_replay_fp32_grad_accum_cleanup_count = 0
        self._gemma_replay_fp32_grad_accum_peak_bytes = 0
        self._gemma_replay_fp32_grad_accum_active = False
        self._gemma_replay_fp32_grad_accum_cpu_offload = False
        self._gemma_replay_fp32_grad_accum_window_id = 0
        self._gemma_replay_fp32_grad_accum_lock = threading.RLock()

    def _install_gemma_replay_fp32_grad_accum_zero_grad_hook(self) -> None:
        optimizer = getattr(self, "actor_optimizer", None)
        if optimizer is None or getattr(optimizer, "_verl_gemma_fp32_grad_accum_zero_grad_hooked", False):
            return
        original_zero_grad = optimizer.zero_grad

        def zero_grad_with_gemma_accum_cleanup(*args, **kwargs):
            self._cleanup_gemma_replay_fp32_grad_accum()
            return original_zero_grad(*args, **kwargs)

        optimizer.zero_grad = zero_grad_with_gemma_accum_cleanup
        optimizer._verl_gemma_fp32_grad_accum_zero_grad_hooked = True

    def _reset_gemma_replay_fp32_grad_accum_window(self) -> None:
        if not hasattr(self, "_gemma_replay_fp32_grad_accum_lock"):
            self._init_gemma_replay_fp32_grad_accum_state()
        if not self._is_diffusion_gemma_actor():
            raise RuntimeError("gemma_replay_fp32_grad_accum post-reduce capture is DiffusionGemma-only")
        _install_gemma_replay_fsdp_post_reduce_grad_capture()
        handles = list(self._iter_gemma_replay_fsdp_flat_param_handles())
        if not handles:
            raise RuntimeError(
                "gemma_replay_fp32_grad_accum now requires DiffusionGemma FSDP post-reduce capture; "
                "no FSDP flat-param handles were found"
            )
        self._unregister_gemma_replay_fsdp_capture_handles()
        with self._gemma_replay_fp32_grad_accum_lock:
            self._gemma_replay_fp32_grad_accum = {}
            self._gemma_replay_fp32_grad_accum_meta = {}
            self._gemma_replay_fp32_grad_accum_templates = {}
            self._gemma_replay_fp32_grad_accum_param_refs = {}
            self._gemma_replay_fp32_grad_accum_registered_handles = set()
            self._gemma_replay_fp32_grad_accum_captured_handles = set()
            self._gemma_replay_fp32_grad_accum_capture_counts = {}
            self._gemma_replay_fp32_grad_accum_calls = 0
            self._gemma_replay_fp32_grad_accum_duplicate_captures = 0
            self._gemma_replay_fp32_grad_accum_materialized_grads = 0
            self._gemma_replay_fp32_grad_accum_materialize_seconds = 0.0
            self._gemma_replay_fp32_grad_accum_cleanup_seconds = 0.0
            self._gemma_replay_fp32_grad_accum_cleanup_count = 0
            self._gemma_replay_fp32_grad_accum_peak_bytes = 0
            self._gemma_replay_fp32_grad_accum_active = True
            self._gemma_replay_fp32_grad_accum_cpu_offload = bool(
                self.config.get("generation_consistency_jsd", {}).get(
                    "gemma_replay_fp32_grad_accum_cpu_offload",
                    False,
                )
            )
            self._gemma_replay_fp32_grad_accum_window_id += 1
            window_id = self._gemma_replay_fp32_grad_accum_window_id
        actor_ref = weakref.ref(self)
        with _gemma_replay_fsdp_capture_lock:
            for handle in handles:
                flat_param = handle.flat_param
                flat_param_id = id(flat_param)
                _gemma_replay_fsdp_capture_registry[flat_param_id] = (actor_ref, window_id)
                self._gemma_replay_fp32_grad_accum_registered_handles.add(flat_param_id)
                self._gemma_replay_fp32_grad_accum_param_refs[flat_param_id] = weakref.ref(flat_param)

    def _gemma_replay_fp32_grad_accum_enabled(self, generation_jsd_config: dict | None = None) -> bool:
        if generation_jsd_config is None:
            generation_jsd_config = self.config.get("generation_consistency_jsd", {})
        return (
            bool(generation_jsd_config.get("gemma_replay_fp32_grad_accum", False))
            and generation_jsd_config.get("objective", "full_jsd") == "gemma_denoise_ce"
            and self._is_diffusion_gemma_actor()
        )

    @staticmethod
    def _gemma_replay_local_grad_tensor(grad: torch.Tensor) -> torch.Tensor:
        if isinstance(grad, DTensor):
            grad = grad.to_local()
        if getattr(grad, "is_sparse", False):
            raise RuntimeError("gemma_replay_fp32_grad_accum does not support sparse gradients")
        return grad

    def _iter_gemma_replay_fsdp_flat_param_handles(self):
        seen: set[int] = set()
        for module in self.actor_module.modules():
            handles = []
            handle = getattr(module, "_handle", None)
            if handle is not None:
                handles.append(handle)
            handles.extend(list(getattr(module, "_handles", []) or []))
            for handle in handles:
                flat_param = getattr(handle, "flat_param", None)
                if flat_param is None:
                    continue
                handle_id = id(handle)
                if handle_id in seen:
                    continue
                seen.add(handle_id)
                yield handle

    def _unregister_gemma_replay_fsdp_capture_handles(self) -> None:
        registered = set(getattr(self, "_gemma_replay_fp32_grad_accum_registered_handles", set()))
        if not registered:
            return
        actor_ref = weakref.ref(self)
        with _gemma_replay_fsdp_capture_lock:
            for flat_param_id in registered:
                entry = _gemma_replay_fsdp_capture_registry.get(flat_param_id)
                if entry is not None and entry[0] == actor_ref:
                    _gemma_replay_fsdp_capture_registry.pop(flat_param_id, None)

    def _capture_gemma_replay_post_reduce_fp32_grad(
        self,
        handle,
        sharded_grad: torch.Tensor,
        window_id: int,
    ) -> None:
        if not getattr(self, "_gemma_replay_fp32_grad_accum_active", False):
            return
        if int(window_id) != int(getattr(self, "_gemma_replay_fp32_grad_accum_window_id", -1)):
            return
        if not torch.is_tensor(sharded_grad):
            raise RuntimeError("DiffusionGemma FSDP post-reduce capture expected a tensor sharded_grad")
        if sharded_grad.dtype != torch.float32:
            raise RuntimeError(
                "DiffusionGemma FSDP post-reduce capture expected FP32 sharded_grad before FSDP casts it; "
                f"got {sharded_grad.dtype}"
            )
        if getattr(sharded_grad, "is_sparse", False):
            raise RuntimeError("gemma_replay_fp32_grad_accum does not support sparse gradients")
        flat_param = getattr(handle, "flat_param", None)
        if flat_param is None:
            return
        flat_param_id = id(flat_param)
        grad = sharded_grad.detach()
        if not _all_finite_chunked(grad):
            raise RuntimeError(
                "gemma_replay_fp32_grad_accum received a non-finite "
                "post-reduce shard"
            )
        cpu_offload = bool(
            getattr(
                self,
                "_gemma_replay_fp32_grad_accum_cpu_offload",
                False,
            )
        )
        with self._gemma_replay_fp32_grad_accum_lock:
            if flat_param_id not in self._gemma_replay_fp32_grad_accum_registered_handles:
                return
            self._gemma_replay_fp32_grad_accum_active = True
            capture_count = self._gemma_replay_fp32_grad_accum_capture_counts.get(flat_param_id, 0)
            if capture_count > 0:
                self._gemma_replay_fp32_grad_accum_duplicate_captures += 1
            self._gemma_replay_fp32_grad_accum_capture_counts[flat_param_id] = capture_count + 1
            self._gemma_replay_fp32_grad_accum_captured_handles.add(flat_param_id)
            self._gemma_replay_fp32_grad_accum_calls += 1
            meta = self._gemma_replay_fp32_grad_accum_meta.get(flat_param_id)
            if meta is None:
                meta = {
                    "shape": tuple(grad.shape),
                    "device": grad.device,
                    "buffer_device": torch.device(
                        "cpu" if cpu_offload else grad.device
                    ),
                    "dtype": flat_param.dtype,
                }
                self._gemma_replay_fp32_grad_accum_meta[flat_param_id] = meta
                self._gemma_replay_fp32_grad_accum_templates[flat_param_id] = None
                self._gemma_replay_fp32_grad_accum_param_refs[flat_param_id] = weakref.ref(flat_param)
                if cpu_offload:
                    buffer = torch.empty(
                        grad.shape,
                        dtype=torch.float32,
                        device="cpu",
                    )
                    buffer.copy_(grad, non_blocking=False)
                    self._gemma_replay_fp32_grad_accum[flat_param_id] = buffer
                else:
                    self._gemma_replay_fp32_grad_accum[flat_param_id] = (
                        torch.zeros_like(grad, dtype=torch.float32)
                    )
            elif tuple(grad.shape) != meta["shape"] or grad.device != meta["device"]:
                raise RuntimeError(
                    "gemma_replay_fp32_grad_accum post-reduce shard changed shape/device: "
                    f"expected shape={meta['shape']} device={meta['device']} "
                    f"got shape={tuple(grad.shape)} device={grad.device}"
                )
            buffer = self._gemma_replay_fp32_grad_accum[flat_param_id]
            if not (cpu_offload and capture_count == 0):
                grad_to_add = grad.to(
                    device=buffer.device,
                    dtype=torch.float32,
                )
                buffer.add_(grad_to_add)
            if not _all_finite_chunked(buffer):
                raise RuntimeError("gemma_replay_fp32_grad_accum encountered a non-finite accumulator")
            current_bytes = self._gemma_replay_fp32_grad_accum_current_bytes()
            self._gemma_replay_fp32_grad_accum_peak_bytes = max(
                int(getattr(self, "_gemma_replay_fp32_grad_accum_peak_bytes", 0)),
                current_bytes,
            )

    def _clear_gemma_replay_public_fsdp_grads(self) -> None:
        if not getattr(self, "_gemma_replay_fp32_grad_accum_active", False):
            return
        for param_ref in getattr(self, "_gemma_replay_fp32_grad_accum_param_refs", {}).values():
            param = param_ref()
            if param is not None:
                param.grad = None

    def _gemma_replay_fp32_grad_accum_current_bytes(self) -> int:
        return int(
            sum(
                int(buffer.numel() * buffer.element_size())
                for buffer in getattr(self, "_gemma_replay_fp32_grad_accum", {}).values()
            )
        )

    def _gemma_replay_fp32_grad_accum_metrics(self) -> dict[str, float]:
        accum = getattr(self, "_gemma_replay_fp32_grad_accum", {})
        registered = getattr(self, "_gemma_replay_fp32_grad_accum_registered_handles", set())
        captured = getattr(self, "_gemma_replay_fp32_grad_accum_captured_handles", set())
        current_bytes = self._gemma_replay_fp32_grad_accum_current_bytes()
        return {
            "actor/diffusion_gemma/fp32_grad_accum/enabled": (
                1.0 if getattr(self, "_gemma_replay_fp32_grad_accum_active", False) else 0.0
            ),
            "actor/diffusion_gemma/fp32_grad_accum/cpu_offload": (
                1.0
                if getattr(
                    self,
                    "_gemma_replay_fp32_grad_accum_cpu_offload",
                    False,
                )
                else 0.0
            ),
            "actor/diffusion_gemma/fp32_grad_accum/capture_mode_post_reduce_fp32": 1.0,
            "actor/diffusion_gemma/fp32_grad_accum/registered_handles": float(len(registered)),
            "actor/diffusion_gemma/fp32_grad_accum/missed_registered_handles": float(
                len(set(registered) - set(captured))
            ),
            "actor/diffusion_gemma/fp32_grad_accum/unique_captured_handles": float(len(captured)),
            "actor/diffusion_gemma/fp32_grad_accum/buffers": float(len(accum)),
            "actor/diffusion_gemma/fp32_grad_accum/bytes": float(current_bytes),
            "actor/diffusion_gemma/fp32_grad_accum/calls": float(
                getattr(self, "_gemma_replay_fp32_grad_accum_calls", 0)
            ),
            "actor/diffusion_gemma/fp32_grad_accum/duplicate_captures": float(
                getattr(self, "_gemma_replay_fp32_grad_accum_duplicate_captures", 0)
            ),
            "actor/diffusion_gemma/fp32_grad_accum/materialized_grads": float(
                getattr(self, "_gemma_replay_fp32_grad_accum_materialized_grads", 0)
            ),
            "actor/diffusion_gemma/fp32_grad_accum/materialize_seconds": float(
                getattr(self, "_gemma_replay_fp32_grad_accum_materialize_seconds", 0.0)
            ),
            "actor/diffusion_gemma/fp32_grad_accum/cleanup_seconds": float(
                getattr(self, "_gemma_replay_fp32_grad_accum_cleanup_seconds", 0.0)
            ),
            "actor/diffusion_gemma/fp32_grad_accum/cleanup_count": float(
                getattr(self, "_gemma_replay_fp32_grad_accum_cleanup_count", 0)
            ),
            "actor/diffusion_gemma/fp32_grad_accum/peak_bytes": float(
                getattr(self, "_gemma_replay_fp32_grad_accum_peak_bytes", 0)
            ),
        }

    def _accumulate_gemma_replay_fp32_local_grads(self) -> dict[str, float]:
        raise RuntimeError(
            "gemma_replay_fp32_grad_accum public .grad accumulation is disabled for parity safety; "
            "enable DiffusionGemma FSDP post-reduce FP32 capture instead"
        )

    @staticmethod
    def _gemma_replay_materialize_dtensor_grad(template: DTensor, local_grad: torch.Tensor):
        try:
            return DTensor.from_local(
                local_grad,
                device_mesh=template.device_mesh,
                placements=template.placements,
                shape=template.shape,
                stride=template.stride(),
            )
        except TypeError:
            return DTensor.from_local(
                local_grad,
                device_mesh=template.device_mesh,
                placements=template.placements,
            )

    def materialize_gemma_replay_fp32_grad_accum(self) -> dict[str, float]:
        if not hasattr(self, "_gemma_replay_fp32_grad_accum"):
            self._init_gemma_replay_fp32_grad_accum_state()
        if not getattr(self, "_gemma_replay_fp32_grad_accum_active", False):
            return self._gemma_replay_fp32_grad_accum_metrics()

        materialize_start = time.perf_counter()
        self._gemma_replay_fp32_grad_accum_materialized_grads = 0
        for param_id, buffer in self._gemma_replay_fp32_grad_accum.items():
            param_ref = self._gemma_replay_fp32_grad_accum_param_refs.get(param_id)
            param = param_ref() if param_ref is not None else None
            if param is None:
                raise RuntimeError("gemma_replay_fp32_grad_accum parameter disappeared before materialize")
            meta = self._gemma_replay_fp32_grad_accum_meta[param_id]
            if (
                tuple(buffer.shape) != meta["shape"]
                or buffer.device
                != meta.get(
                    "buffer_device",
                    meta["device"],
                )
            ):
                raise RuntimeError(
                    "gemma_replay_fp32_grad_accum buffer changed shape/device before materialize: "
                    f"expected shape={meta['shape']} "
                    f"device={meta.get('buffer_device', meta['device'])} "
                    f"got shape={tuple(buffer.shape)} device={buffer.device}"
                )
            if not _all_finite_chunked(buffer):
                raise RuntimeError("gemma_replay_fp32_grad_accum materialize found a non-finite accumulator")
            if bool(getattr(param, "_post_backward_called", False)):
                raise RuntimeError(
                    "gemma_replay_fp32_grad_accum materialize ran before FSDP post-backward finalization"
                )
            if hasattr(param, "_saved_grad_shard"):
                delattr(param, "_saved_grad_shard")
            local_grad = buffer.to(device=meta["device"], dtype=meta["dtype"])
            template = self._gemma_replay_fp32_grad_accum_templates.get(param_id)
            param.grad = (
                self._gemma_replay_materialize_dtensor_grad(template, local_grad)
                if isinstance(template, DTensor)
                else local_grad
            )
            self._gemma_replay_fp32_grad_accum_materialized_grads += 1
        self._gemma_replay_fp32_grad_accum_materialize_seconds += time.perf_counter() - materialize_start
        return self._gemma_replay_fp32_grad_accum_metrics()

    def _cleanup_gemma_replay_fp32_grad_accum(self) -> dict[str, float]:
        if not hasattr(self, "_gemma_replay_fp32_grad_accum"):
            return {}
        cleanup_start = time.perf_counter()
        had_state = bool(
            getattr(self, "_gemma_replay_fp32_grad_accum_active", False)
            or getattr(self, "_gemma_replay_fp32_grad_accum", {})
        )
        if had_state:
            self._gemma_replay_fp32_grad_accum_cleanup_count += 1
        self._clear_gemma_replay_public_fsdp_grads()
        self._unregister_gemma_replay_fsdp_capture_handles()
        self._gemma_replay_fp32_grad_accum = {}
        self._gemma_replay_fp32_grad_accum_meta = {}
        self._gemma_replay_fp32_grad_accum_templates = {}
        self._gemma_replay_fp32_grad_accum_param_refs = {}
        self._gemma_replay_fp32_grad_accum_registered_handles = set()
        self._gemma_replay_fp32_grad_accum_captured_handles = set()
        self._gemma_replay_fp32_grad_accum_capture_counts = {}
        self._gemma_replay_fp32_grad_accum_active = False
        self._gemma_replay_fp32_grad_accum_cleanup_seconds += time.perf_counter() - cleanup_start
        return self._gemma_replay_fp32_grad_accum_metrics()

    def _effective_logprob_temperature(self, temperature: float) -> float:
        if self.config.get("use_raw_replay_log_probs", False):
            return 1.0
        return temperature if temperature and temperature > 0 else 1.0

    @staticmethod
    def _actor_replay_token_debug_enabled() -> bool:
        return os.environ.get("VERL_ACTOR_REPLAY_TOKEN_DEBUG", "0") == "1"

    @staticmethod
    def _append_if_present(keys: list[str], batch, names: list[str]) -> None:
        for name in names:
            if name in batch.keys() and name not in keys:
                keys.append(name)

    def _actor_replay_token_debug_path(self) -> str:
        dump_dir = os.environ.get("VERL_ACTOR_REPLAY_TOKEN_DEBUG_DIR")
        if not dump_dir:
            trace_dir = os.environ.get("SGLANG_SDAR_TRACE_DIR")
            if trace_dir:
                dump_dir = os.path.join(os.path.dirname(trace_dir), "actor_replay_token_debug")
            else:
                dump_dir = os.path.join(os.getcwd(), "actor_replay_token_debug")
        os.makedirs(dump_dir, exist_ok=True)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        return os.path.join(dump_dir, f"rank{rank}_pid{os.getpid()}.jsonl")

    @staticmethod
    def _tensor_item(tensor: torch.Tensor, b: int, pos: int):
        value = tensor[b, pos].detach().cpu()
        if value.numel() != 1:
            return None
        if value.dtype == torch.bool:
            return bool(value.item())
        if torch.is_floating_point(value):
            return float(value.float().item())
        return int(value.long().item())

    @staticmethod
    def _hidden_status(tensor: torch.Tensor | None, b: int, pos: int) -> dict[str, bool | None]:
        if tensor is None:
            return {"present": False, "finite": None, "nonzero": None}
        value = tensor[b, pos].detach().float()
        return {
            "present": True,
            "finite": bool(torch.isfinite(value).all().item()),
            "nonzero": bool(value.abs().sum().item() > 0),
        }

    def _dump_actor_replay_token_debug(
        self,
        micro_batch: dict[str, torch.Tensor],
        log_probs: torch.Tensor,
        input_ids: torch.Tensor,
        response_length: int,
    ) -> None:
        try:
            response_mask = micro_batch.get("response_mask", torch.ones_like(log_probs, dtype=torch.bool)).bool()
            active = torch.nonzero(response_mask, as_tuple=False)
            debug_limit = int(os.environ.get("VERL_ACTOR_REPLAY_TOKEN_DEBUG_LIMIT", "0"))
            if debug_limit > 0:
                active = active[:debug_limit]

            rollout_lp = micro_batch.get("rollout_log_probs", None)
            step_maps = micro_batch.get("step_maps", None)
            current_mask = micro_batch.get("response_current_step_mask", None)
            future_mask = micro_batch.get("response_future_steps_mask", None)
            sc_signal = micro_batch.get("rollout_self_conditioning_signal", None)
            source_hidden = micro_batch.get("rollout_logits_source_hidden", None)
            phase = micro_batch.get("_actor_replay_debug_phase", "unknown")
            global_step = micro_batch.get("_actor_replay_debug_global_step", None)
            batch_idx = micro_batch.get("_actor_replay_debug_batch_idx", None)
            micro_idx = micro_batch.get("_actor_replay_debug_micro_idx", None)

            rows = []
            seqlen = input_ids.shape[1]
            clean_response_start = seqlen - 2 * response_length
            noisy_response_start = seqlen - response_length
            for b, pos in active.detach().cpu().tolist():
                actor_lp = float(log_probs[b, pos].detach().float().cpu().item())
                rollout_value = (
                    float(rollout_lp[b, pos].detach().float().cpu().item()) if rollout_lp is not None else None
                )
                diff = actor_lp - rollout_value if rollout_value is not None else None
                clean_abs_pos = clean_response_start + pos
                noisy_abs_pos = noisy_response_start + pos
                rows.append(
                    {
                        "phase": phase,
                        "global_step": global_step,
                        "batch_idx": batch_idx,
                        "micro_idx": micro_idx,
                        "sample_id": int(b),
                        "response_index": int(pos),
                        "token_id": int(micro_batch["responses"][b, pos].detach().cpu().item()),
                        "rollout_logprob": rollout_value,
                        "actor_logprob": actor_lp,
                        "diff": diff,
                        "abs_diff": abs(diff) if diff is not None else None,
                        "label_pos": int(noisy_abs_pos),
                        "input_pos": int(noisy_abs_pos),
                        "clean_input_pos": int(clean_abs_pos),
                        "noisy_input_pos": int(noisy_abs_pos),
                        "clean_input_id": int(input_ids[b, clean_abs_pos].detach().cpu().item())
                        if 0 <= clean_abs_pos < seqlen
                        else None,
                        "noisy_input_id": int(input_ids[b, noisy_abs_pos].detach().cpu().item())
                        if 0 <= noisy_abs_pos < seqlen
                        else None,
                        "active_mask": bool(response_mask[b, pos].detach().cpu().item()),
                        "step_key": self._tensor_item(step_maps, b, pos) if step_maps is not None else None,
                        "current_step_mask": self._tensor_item(current_mask, b, pos)
                        if current_mask is not None
                        else None,
                        "future_step_mask": self._tensor_item(future_mask, b, pos)
                        if future_mask is not None
                        else None,
                        "self_conditioning_signal": self._hidden_status(sc_signal, b, pos),
                        "logits_source_hidden": self._hidden_status(source_hidden, b, pos),
                    }
                )

            path = self._actor_replay_token_debug_path()
            with open(path, "a", encoding="utf-8") as fp:
                for row in rows:
                    fp.write(json.dumps(row, sort_keys=True) + "\n")
            print(f"[ActorReplayTokenDebug] wrote {len(rows)} rows to {path}", flush=True)
        except Exception as exc:
            print(f"[ActorReplayTokenDebug] failed: {exc}", flush=True)

    def _forward_micro_batch(
        self, micro_batch: dict[str, torch.Tensor], temperature: float, calculate_entropy: bool = False
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            dict[str, torch.Tensor]:
                log_probs: (bs, response_len)
                if calculate_entropy is True:
                    entropys: (bs, response_len)
                if calculate_sum_pi_squared is False:
                    sum_pi_squared: (bs, response_len)
        """
        self._dump_actor_input_checksum_debug(
            micro_batch,
            phase=micro_batch.get("_actor_replay_debug_phase", None),
            global_step=micro_batch.get("_actor_replay_debug_global_step", None),
            batch_idx=micro_batch.get("_actor_replay_debug_batch_idx", None),
            micro_idx=micro_batch.get("_actor_replay_debug_micro_idx", None),
        )
        if self._is_diffusion_gemma_actor():
            return self._forward_diffusion_gemma_micro_batch(
                micro_batch,
                temperature=temperature,
                calculate_entropy=calculate_entropy,
            )
        signed_trace_ce_enabled = False
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)
        sum_pi_squared_checkpointing = self.config.get("sum_pi_squared_checkpointing", False)
        generation_jsd_config_for_validation = self.config.get("generation_consistency_jsd", {})
        block_tpf_ce_config = generation_jsd_config_for_validation.get("block_tpf_ce_weight", {}) or {}
        block_tpf_ce_requested = bool(block_tpf_ce_config.get("enable", False))
        response_tpf_ce_config = (
            generation_jsd_config_for_validation.get(
                "response_tpf_ce_scale",
                {},
            )
            or {}
        )
        response_tpf_ce_requested = bool(
            response_tpf_ce_config.get("enable", False)
        )
        response_level_ce_config = (
            generation_jsd_config_for_validation.get(
                "response_level_ce",
                {},
            )
            or {}
        )
        response_level_ce_requested = bool(
            response_level_ce_config.get("enable", False)
        )
        if sum(
            (
                block_tpf_ce_requested,
                response_tpf_ce_requested,
                response_level_ce_requested,
            )
        ) > 1:
            raise ValueError(
                "block_tpf_ce_weight, response_tpf_ce_scale, and "
                "response_level_ce are mutually exclusive"
            )
        block_tpf_weight_rmpad = None
        response_tpf_ce_scale_rmpad = None
        response_level_ce_weight_rmpad = None
        if (
            block_tpf_ce_requested
            or response_tpf_ce_requested
            or response_level_ce_requested
        ):
            objective = str(generation_jsd_config_for_validation.get("objective", "full_jsd"))
            if objective != "trace_all_ce":
                raise ValueError(
                    "trace CE weighting is supported only with "
                    f"objective=trace_all_ce, got {objective!r}"
                )
            if not self.use_remove_padding:
                raise NotImplementedError(
                    "trace CE weighting requires actor.use_remove_padding=true; "
                    "the non-rmpad Qwen actor path is not implemented"
                )
        if response_tpf_ce_requested:
            normalization = str(
                response_tpf_ce_config.get(
                    "normalization",
                    "unweighted_token_mean",
                )
            )
            if normalization != "unweighted_token_mean":
                raise ValueError(
                    "response_tpf_ce_scale.normalization must be "
                    f"unweighted_token_mean, got {normalization!r}"
                )
        # PrefixGrouper path for shared-prefix optimization
        if self.use_prefix_grouper:
            can_use_pg = (
                not self.use_remove_padding
                and not self.use_ulysses_sp
                and not self.use_fused_kernels
                and not self.use_dynamic_bsz
            )
            if can_use_pg and "response_mask" in micro_batch and "uid" in micro_batch:
                from verl.trainer.ppo.prefix_grouper_utils import forward_micro_batch_with_prefix_grouper

                return forward_micro_batch_with_prefix_grouper(
                    micro_batch=micro_batch,
                    model=self.actor_module,
                    temperature=self._effective_logprob_temperature(temperature),
                    calculate_entropy=calculate_entropy,
                    device_name=self.device_name,
                    param_dtype=self.param_dtype,
                    use_chunking_entropy=self.config.get("entropy_from_logits_with_chunking", False),
                )

        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, enabled=False):
            input_ids = micro_batch["input_ids"]
            responses = micro_batch["responses"]

            # Check
            # log_probs = micro_batch["rollout_log_probs"]

            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)
                rollout_sc_signal = micro_batch.get("rollout_self_conditioning_signal", None)
                rollout_sc_signal_rmpad = None
                sparse_sc_block_values = micro_batch.get(
                    "rollout_self_conditioning_signal_block_values", None
                )
                sparse_sc_block_starts = micro_batch.get(
                    "rollout_self_conditioning_signal_block_starts", None
                )
                sparse_sc_block_lengths = micro_batch.get(
                    "rollout_self_conditioning_signal_block_lengths", None
                )
                sparse_sc_block_rows = micro_batch.get(
                    "rollout_self_conditioning_signal_block_rows", None
                )
                gpu_sc_descriptor_rows = micro_batch.get(
                    "_packed_sparse_sc_gpu_leases",
                    None,
                )
                rollout_logits_source_hidden = micro_batch.get("rollout_logits_source_hidden", None)
                rollout_logits_source_hidden_rmpad = None
                sparse_logits_block_values = micro_batch.get(
                    "rollout_logits_source_hidden_block_values", None
                )
                sparse_logits_block_starts = micro_batch.get(
                    "rollout_logits_source_hidden_block_starts", None
                )
                sparse_logits_block_lengths = micro_batch.get(
                    "rollout_logits_source_hidden_block_lengths", None
                )
                sparse_logits_block_rows = micro_batch.get(
                    "rollout_logits_source_hidden_block_rows", None
                )
                gpu_logits_descriptor_rows = micro_batch.get(
                    "_packed_sparse_logits_gpu_leases",
                    None,
                )
                sparse_logits_source_values = micro_batch.get("rollout_logits_source_hidden_sparse_values", None)
                sparse_logits_source_rows = micro_batch.get("rollout_logits_source_hidden_sparse_rows", None)
                sparse_logits_source_positions = micro_batch.get("rollout_logits_source_hidden_sparse_positions", None)
                sparse_logits_source_shape = micro_batch.get("rollout_logits_source_hidden_sparse_shape", None)
                rollout_log_probs = micro_batch.get("rollout_log_probs", None)
                rollout_log_probs_rmpad = None
                step_maps_values = micro_batch.get("step_maps", None)
                step_maps_rmpad = None
                adj_ent_pair_mask_values = micro_batch.get("adjacent_entropy_pair_mask", None)
                adj_ent_pair_mask_rmpad = None
                adj_ent_prev_mean_values = micro_batch.get("adjacent_entropy_prev_mean", None)
                adj_ent_prev_mean_rmpad = None
                adj_ent_pair_local_step_values = micro_batch.get(
                    "adjacent_entropy_pair_local_step", None
                )
                adj_ent_pair_local_step_rmpad = None
                adj_ent_rollout_next_mean_values = micro_batch.get(
                    "adjacent_entropy_rollout_next_mean", None
                )
                adj_ent_rollout_next_mean_rmpad = None
                adj_ent_vocab_size_values = micro_batch.get(
                    "adjacent_entropy_vocab_size", None
                )
                adj_ent_vocab_size_rmpad = None
                adj_ent_pair_ids_values = micro_batch.get(
                    "adjacent_entropy_pair_ids", None
                )
                adj_ent_pair_ids_rmpad = None
                rollout_quality_mask = micro_batch.get("rollout_quality_mask", None)
                rollout_quality_mask_rmpad = None
                signed_trace_ce_role = micro_batch.get(
                    "signed_trace_ce_role",
                    None,
                )
                signed_trace_ce_role_rmpad = None
                causal_wrong_ce_role = micro_batch.get(
                    "causal_wrong_ce_role",
                    None,
                )
                causal_wrong_ce_role_rmpad = None
                def _pack_response_hidden(response_hidden):
                    if response_hidden is None:
                        return None
                    response_valid_mask = attention_mask[:, -response_length:].bool()
                    flat_response_values = response_hidden.to(device=input_ids.device)[response_valid_mask]
                    packed = torch.zeros(
                        (indices.numel(), response_hidden.shape[-1]),
                        device=input_ids.device,
                        dtype=response_hidden.dtype,
                    )
                    if flat_response_values.numel() == 0:
                        return packed.unsqueeze(0)

                    flat_attention_mask = attention_mask.reshape(-1).bool()
                    flat_offsets = flat_attention_mask.to(torch.long).cumsum(dim=0) - 1
                    response_flat_idx = (
                        torch.arange(batch_size, device=input_ids.device).unsqueeze(1) * seqlen
                        + torch.arange(seqlen - response_length, seqlen, device=input_ids.device).unsqueeze(0)
                    )
                    packed_idx = flat_offsets[response_flat_idx[response_valid_mask]]
                    packed[packed_idx] = flat_response_values
                    return packed.unsqueeze(0)

                def _pack_sparse_response_hidden():
                    if sparse_logits_source_values is None:
                        return None
                    if rollout_logits_source_hidden is not None:
                        raise RuntimeError("received both dense and sparse rollout_logits_source_hidden payloads")
                    if not (
                        torch.is_tensor(sparse_logits_source_values)
                        and torch.is_tensor(sparse_logits_source_rows)
                        and torch.is_tensor(sparse_logits_source_positions)
                        and torch.is_tensor(sparse_logits_source_shape)
                    ):
                        raise RuntimeError("incomplete sparse rollout_logits_source_hidden payload")
                    values = sparse_logits_source_values.to(device=input_ids.device)
                    rows = sparse_logits_source_rows.to(device=input_ids.device, dtype=torch.long).view(-1)
                    positions = sparse_logits_source_positions.to(device=input_ids.device, dtype=torch.long).view(-1)
                    shape = sparse_logits_source_shape.detach().cpu().long().view(-1)
                    if int(shape.numel()) != 3:
                        raise RuntimeError(
                            "rollout_logits_source_hidden_sparse_shape must have 3 elements, "
                            f"got shape={tuple(sparse_logits_source_shape.shape)}"
                        )
                    row_count, sparse_response_len, hidden_size = (int(shape[i].item()) for i in range(3))
                    if row_count != batch_size or sparse_response_len != response_length:
                        raise RuntimeError(
                            "sparse rollout_logits_source_hidden shape does not match micro-batch: "
                            f"sparse_shape={(row_count, sparse_response_len, hidden_size)} "
                            f"input_shape={(batch_size, response_length)}"
                        )
                    if values.dim() != 2 or int(values.shape[-1]) != hidden_size:
                        raise RuntimeError(
                            "rollout_logits_source_hidden_sparse_values must be [values, hidden], "
                            f"got values_shape={tuple(values.shape)} hidden_size={hidden_size}"
                        )
                    value_count = int(values.shape[0])
                    if int(rows.numel()) != value_count or int(positions.numel()) != value_count:
                        raise RuntimeError(
                            "sparse rollout_logits_source_hidden index count mismatch: "
                            f"values={tuple(values.shape)} rows={tuple(rows.shape)} "
                            f"positions={tuple(positions.shape)}"
                        )
                    packed = torch.zeros(
                        (indices.numel(), hidden_size),
                        device=input_ids.device,
                        dtype=values.dtype,
                    )
                    if value_count <= 0:
                        return packed.unsqueeze(0)
                    valid = rows.ge(0) & rows.lt(batch_size) & positions.ge(0) & positions.lt(response_length)
                    if not bool(valid.all()):
                        raise RuntimeError(
                            "sparse rollout_logits_source_hidden indices out of range: "
                            f"sparse_shape={(row_count, sparse_response_len, hidden_size)}"
                        )
                    linear = rows * int(response_length) + positions
                    if int(torch.unique(linear).numel()) != value_count:
                        raise RuntimeError("sparse rollout_logits_source_hidden contains duplicate row/position entries")
                    flat_attention_mask = attention_mask.reshape(-1).bool()
                    flat_offsets = flat_attention_mask.to(torch.long).cumsum(dim=0) - 1
                    response_abs_positions = seqlen - response_length + positions
                    response_flat_idx = rows * seqlen + response_abs_positions
                    response_valid = flat_attention_mask[response_flat_idx]
                    if bool(response_valid.any()):
                        packed_idx = flat_offsets[response_flat_idx[response_valid]]
                        packed[packed_idx] = values[response_valid]
                    return packed.unsqueeze(0)

                def _pack_block_local_response_hidden(
                    *,
                    values,
                    row_indices,
                    starts,
                    lengths,
                    dense_value,
                    field_name,
                ):
                    if row_indices is not None or (
                        torch.is_tensor(values) and values.dim() == 4
                    ):
                        return _pack_multi_block_response_hidden_impl(
                            values=values,
                            row_indices=row_indices,
                            starts=starts,
                            lengths=lengths,
                            dense_value=dense_value,
                            field_name=field_name,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            indices=indices,
                            response_length=response_length,
                        )
                    return _pack_block_local_response_hidden_impl(
                        values=values,
                        starts=starts,
                        lengths=lengths,
                        dense_value=dense_value,
                        field_name=field_name,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        indices=indices,
                        response_length=response_length,
                    )

                def _pack_response_scalar(response_values, fill_value: float = 0.0):
                    if response_values is None:
                        return None
                    response_valid_mask = attention_mask[:, -response_length:].bool()
                    flat_response_values = response_values.to(device=input_ids.device)[response_valid_mask]
                    packed = torch.full(
                        (indices.numel(),),
                        fill_value,
                        device=input_ids.device,
                        dtype=response_values.dtype,
                    )
                    if flat_response_values.numel() == 0:
                        return packed.unsqueeze(0)
                    flat_attention_mask = attention_mask.reshape(-1).bool()
                    flat_offsets = flat_attention_mask.to(torch.long).cumsum(dim=0) - 1
                    response_flat_idx = (
                        torch.arange(batch_size, device=input_ids.device).unsqueeze(1) * seqlen
                        + torch.arange(seqlen - response_length, seqlen, device=input_ids.device).unsqueeze(0)
                    )
                    packed_idx = flat_offsets[response_flat_idx[response_valid_mask]]
                    packed[packed_idx] = flat_response_values
                    return packed.unsqueeze(0)

                rollout_sc_signal_rmpad = _pack_response_hidden(rollout_sc_signal)
                if rollout_sc_signal_rmpad is None:
                    rollout_sc_signal_rmpad = (
                        _pack_block_local_response_hidden(
                            values=sparse_sc_block_values,
                            row_indices=sparse_sc_block_rows,
                            starts=sparse_sc_block_starts,
                            lengths=sparse_sc_block_lengths,
                            dense_value=rollout_sc_signal,
                            field_name="rollout_self_conditioning_signal",
                        )
                    )
                if (
                    rollout_sc_signal_rmpad is None
                    and gpu_sc_descriptor_rows is not None
                ):
                    rollout_sc_signal_rmpad = (
                        _pack_gpu_replay_lease_hidden_impl(
                            descriptor_rows=gpu_sc_descriptor_rows,
                            starts=sparse_sc_block_starts,
                            lengths=sparse_sc_block_lengths,
                            field_name=(
                                "rollout_self_conditioning_signal"
                            ),
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            indices=indices,
                            response_length=response_length,
                            sequence_parallel_size=(
                                self.ulysses_sequence_parallel_size
                            ),
                        )
                    )
                rollout_logits_source_hidden_rmpad = _pack_response_hidden(rollout_logits_source_hidden)
                if rollout_logits_source_hidden_rmpad is None:
                    rollout_logits_source_hidden_rmpad = _pack_sparse_response_hidden()
                if rollout_logits_source_hidden_rmpad is None:
                    rollout_logits_source_hidden_rmpad = (
                        _pack_block_local_response_hidden(
                            values=sparse_logits_block_values,
                            row_indices=sparse_logits_block_rows,
                            starts=sparse_logits_block_starts,
                            lengths=sparse_logits_block_lengths,
                            dense_value=rollout_logits_source_hidden,
                            field_name="rollout_logits_source_hidden",
                        )
                    )
                if (
                    rollout_logits_source_hidden_rmpad is None
                    and gpu_logits_descriptor_rows is not None
                ):
                    rollout_logits_source_hidden_rmpad = (
                        _pack_gpu_replay_lease_hidden_impl(
                            descriptor_rows=gpu_logits_descriptor_rows,
                            starts=sparse_logits_block_starts,
                            lengths=sparse_logits_block_lengths,
                            field_name=(
                                "rollout_logits_source_hidden"
                            ),
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            indices=indices,
                            response_length=response_length,
                            sequence_parallel_size=(
                                self.ulysses_sequence_parallel_size
                            ),
                        )
                    )
                rollout_log_probs_rmpad = _pack_response_scalar(rollout_log_probs, fill_value=-float("inf"))
                step_maps_rmpad = _pack_response_scalar(step_maps_values, fill_value=-1)
                adj_ent_pair_mask_rmpad = _pack_response_scalar(
                    adj_ent_pair_mask_values.to(dtype=torch.bool)
                    if adj_ent_pair_mask_values is not None
                    else None,
                    fill_value=False,
                )
                adj_ent_prev_mean_rmpad = _pack_response_scalar(
                    adj_ent_prev_mean_values.to(dtype=torch.float32)
                    if adj_ent_prev_mean_values is not None
                    else None,
                    fill_value=0.0,
                )
                adj_ent_pair_local_step_rmpad = _pack_response_scalar(
                    adj_ent_pair_local_step_values.to(dtype=torch.float32)
                    if adj_ent_pair_local_step_values is not None
                    else None,
                    fill_value=-1.0,
                )
                adj_ent_rollout_next_mean_rmpad = _pack_response_scalar(
                    adj_ent_rollout_next_mean_values.to(
                        dtype=torch.float32
                    )
                    if adj_ent_rollout_next_mean_values is not None
                    else None,
                    fill_value=0.0,
                )
                adj_ent_vocab_size_rmpad = _pack_response_scalar(
                    adj_ent_vocab_size_values.to(dtype=torch.long)
                    if adj_ent_vocab_size_values is not None
                    else None,
                    fill_value=0,
                )
                adj_ent_pair_ids_rmpad = _pack_response_scalar(
                    adj_ent_pair_ids_values.to(dtype=torch.long)
                    if adj_ent_pair_ids_values is not None
                    else None,
                    fill_value=-1,
                )
                rollout_quality_mask_rmpad = _pack_response_scalar(rollout_quality_mask, fill_value=False)
                signed_trace_ce_role_rmpad = _pack_response_scalar(
                    signed_trace_ce_role,
                    fill_value=0,
                )
                causal_wrong_ce_role_rmpad = _pack_response_scalar(
                    causal_wrong_ce_role,
                    fill_value=0,
                )
                _block_tpf_weight_vals = micro_batch.get("block_tpf_weight", None)
                block_tpf_weight_rmpad = _pack_response_scalar(_block_tpf_weight_vals, fill_value=1.0)
                _response_tpf_ce_scale_vals = micro_batch.get(
                    "response_tpf_ce_scale",
                    None,
                )
                response_tpf_ce_scale_rmpad = _pack_response_scalar(
                    _response_tpf_ce_scale_vals,
                    fill_value=1.0,
                )
                _response_level_ce_weight_vals = micro_batch.get(
                    "response_level_ce_weight",
                    None,
                )
                response_level_ce_weight_rmpad = _pack_response_scalar(
                    _response_level_ce_weight_vals,
                    fill_value=0.0,
                )
                _response_level_aec_weight_vals = micro_batch.get(
                    "response_level_aec_pair_weight",
                    None,
                )
                response_level_aec_pair_weight_rmpad = _pack_response_scalar(
                    _response_level_aec_weight_vals,
                    fill_value=0.0,
                )

                token_debug_enabled = self._actor_replay_token_debug_enabled()
                teacher_token_debug_rows = []
                teacher_token_debug_limit = int(os.environ.get("VERL_ACTOR_REPLAY_TOKEN_DEBUG_LIMIT", "0"))
                response_rmpad_to_batch_pos = None
                if token_debug_enabled:
                    response_valid_mask = attention_mask[:, -response_length:].bool()
                    response_rmpad_to_batch_pos = torch.full(
                        (indices.numel(), 2),
                        -1,
                        device=input_ids.device,
                        dtype=torch.long,
                    )
                    flat_attention_mask = attention_mask.reshape(-1).bool()
                    flat_offsets = flat_attention_mask.to(torch.long).cumsum(dim=0) - 1
                    response_flat_idx = (
                        torch.arange(batch_size, device=input_ids.device).unsqueeze(1) * seqlen
                        + torch.arange(seqlen - response_length, seqlen, device=input_ids.device).unsqueeze(0)
                    )
                    if bool(response_valid_mask.any()):
                        packed_idx = flat_offsets[response_flat_idx[response_valid_mask]]
                        row_ids = (
                            torch.arange(batch_size, device=input_ids.device)
                            .unsqueeze(1)
                            .expand(batch_size, response_length)[response_valid_mask]
                        )
                        pos_ids = (
                            torch.arange(response_length, device=input_ids.device)
                            .unsqueeze(0)
                            .expand(batch_size, response_length)[response_valid_mask]
                        )
                        response_rmpad_to_batch_pos[packed_idx, 0] = row_ids
                        response_rmpad_to_batch_pos[packed_idx, 1] = pos_ids

                labels = torch.full_like(input_ids, -100)
                labels[:, -responses.shape[1]:] = responses
                labels_rmpad, _, _, *_ = unpad_input(
                    labels.unsqueeze(-1), attention_mask
                )
                labels_rmpad = labels_rmpad.transpose(0, 1)
                labels_rmpad_2d = labels_rmpad
                tpf_labels_rmpad = None
                if "tpf_action_ids" in micro_batch:
                    tpf_labels = torch.full_like(input_ids, -100)
                    tpf_labels[:, -response_length:] = micro_batch["tpf_action_ids"]
                    tpf_labels_rmpad, _, _, *_ = unpad_input(
                        tpf_labels.unsqueeze(-1), attention_mask
                    )
                    tpf_labels_rmpad = tpf_labels_rmpad.transpose(0, 1)
                dense_noisy_labels = micro_batch.get("dense_noisy_labels", None)
                dense_labels_rmpad = None
                if dense_noisy_labels is not None:
                    dense_labels = torch.full_like(input_ids, -100)
                    dense_labels[:, -response_length:] = dense_noisy_labels
                    dense_labels_rmpad, _, _, *_ = unpad_input(
                        dense_labels.unsqueeze(-1), attention_mask
                    )
                    dense_labels_rmpad = dense_labels_rmpad.transpose(0, 1)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                is_mask_all_zero = attention_mask.sum() == 0
                if is_mask_all_zero:
                    input_ids_rmpad = torch.zeros(
                        (1, self.ulysses_sequence_parallel_size),
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )
                    labels_rmpad = torch.full(
                        (1, self.ulysses_sequence_parallel_size), -100,
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )
                    if position_ids.dim() == 3:
                        position_ids_rmpad = torch.zeros(
                            (position_ids.shape[0], 1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )
                    else:
                        position_ids_rmpad = torch.zeros(
                            (1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )
                    if rollout_sc_signal_rmpad is not None:
                        rollout_sc_signal_rmpad = torch.zeros(
                            (
                                1,
                                self.ulysses_sequence_parallel_size,
                                rollout_sc_signal_rmpad.shape[-1],
                            ),
                            device=input_ids.device,
                            dtype=rollout_sc_signal_rmpad.dtype,
                        )
                    if rollout_logits_source_hidden_rmpad is not None:
                        rollout_logits_source_hidden_rmpad = torch.zeros(
                            (1, self.ulysses_sequence_parallel_size, rollout_logits_source_hidden_rmpad.shape[-1]),
                            device=input_ids.device,
                            dtype=rollout_logits_source_hidden_rmpad.dtype,
                        )

                if "image_bound" in multi_modal_inputs:
                    raise RuntimeError
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                    def _slice_sp_tensor(value, *, seq_dim, pad_value):
                        if value is None:
                            return None
                        sliced, value_pad_size = ulysses_pad_and_slice_tensor(
                            value,
                            seq_dim=seq_dim,
                            sp_size=self.ulysses_sequence_parallel_size,
                            pad_value=pad_value,
                        )
                        if int(value_pad_size) != int(pad_size):
                            raise RuntimeError(
                                "Ulysses replay tensor padding mismatch: "
                                f"input_pad={pad_size} value_pad={value_pad_size}"
                            )
                        return sliced

                    labels_rmpad = _slice_sp_tensor(
                        labels_rmpad,
                        seq_dim=1,
                        pad_value=-100,
                    )
                    tpf_labels_rmpad = _slice_sp_tensor(
                        tpf_labels_rmpad,
                        seq_dim=1,
                        pad_value=-100,
                    )
                    dense_labels_rmpad = _slice_sp_tensor(
                        dense_labels_rmpad,
                        seq_dim=1,
                        pad_value=-100,
                    )
                    rollout_sc_signal_rmpad = _slice_sp_tensor(
                        rollout_sc_signal_rmpad,
                        seq_dim=1,
                        pad_value=0,
                    )
                    rollout_logits_source_hidden_rmpad = _slice_sp_tensor(
                        rollout_logits_source_hidden_rmpad,
                        seq_dim=1,
                        pad_value=0,
                    )
                    rollout_log_probs_rmpad = _slice_sp_tensor(
                        rollout_log_probs_rmpad,
                        seq_dim=1,
                        pad_value=-float("inf"),
                    )
                    step_maps_rmpad = _slice_sp_tensor(
                        step_maps_rmpad,
                        seq_dim=1,
                        pad_value=-1,
                    )
                    adj_ent_pair_mask_rmpad = _slice_sp_tensor(
                        adj_ent_pair_mask_rmpad,
                        seq_dim=1,
                        pad_value=False,
                    )
                    adj_ent_prev_mean_rmpad = _slice_sp_tensor(
                        adj_ent_prev_mean_rmpad,
                        seq_dim=1,
                        pad_value=0.0,
                    )
                    adj_ent_pair_local_step_rmpad = _slice_sp_tensor(
                        adj_ent_pair_local_step_rmpad,
                        seq_dim=1,
                        pad_value=-1.0,
                    )
                    adj_ent_rollout_next_mean_rmpad = _slice_sp_tensor(
                        adj_ent_rollout_next_mean_rmpad,
                        seq_dim=1,
                        pad_value=0.0,
                    )
                    adj_ent_vocab_size_rmpad = _slice_sp_tensor(
                        adj_ent_vocab_size_rmpad,
                        seq_dim=1,
                        pad_value=0,
                    )
                    adj_ent_pair_ids_rmpad = _slice_sp_tensor(
                        adj_ent_pair_ids_rmpad,
                        seq_dim=1,
                        pad_value=-1,
                    )
                    rollout_quality_mask_rmpad = _slice_sp_tensor(
                        rollout_quality_mask_rmpad,
                        seq_dim=1,
                        pad_value=False,
                    )
                    signed_trace_ce_role_rmpad = _slice_sp_tensor(
                        signed_trace_ce_role_rmpad,
                        seq_dim=1,
                        pad_value=0,
                    )
                    causal_wrong_ce_role_rmpad = _slice_sp_tensor(
                        causal_wrong_ce_role_rmpad,
                        seq_dim=1,
                        pad_value=0,
                    )
                    block_tpf_weight_rmpad = _slice_sp_tensor(
                        block_tpf_weight_rmpad,
                        seq_dim=1,
                        pad_value=1.0,
                    )
                    response_tpf_ce_scale_rmpad = _slice_sp_tensor(
                        response_tpf_ce_scale_rmpad,
                        seq_dim=1,
                        pad_value=1.0,
                    )
                    response_level_ce_weight_rmpad = _slice_sp_tensor(
                        response_level_ce_weight_rmpad,
                        seq_dim=1,
                        pad_value=0.0,
                    )
                    response_level_aec_pair_weight_rmpad = _slice_sp_tensor(
                        response_level_aec_pair_weight_rmpad,
                        seq_dim=1,
                        pad_value=0.0,
                    )
                    response_rmpad_to_batch_pos = _slice_sp_tensor(
                        response_rmpad_to_batch_pos,
                        seq_dim=0,
                        pad_value=-1,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)
                labels_rmpad = labels_rmpad.squeeze(0)
                if tpf_labels_rmpad is not None:
                    tpf_labels_rmpad = tpf_labels_rmpad.squeeze(0)
                if rollout_log_probs_rmpad is not None:
                    rollout_log_probs_rmpad = rollout_log_probs_rmpad.squeeze(0)
                if step_maps_rmpad is not None:
                    step_maps_rmpad = step_maps_rmpad.squeeze(0)
                if adj_ent_pair_mask_rmpad is not None:
                    adj_ent_pair_mask_rmpad = adj_ent_pair_mask_rmpad.squeeze(0).bool()
                if adj_ent_prev_mean_rmpad is not None:
                    adj_ent_prev_mean_rmpad = adj_ent_prev_mean_rmpad.squeeze(0).float()
                if adj_ent_pair_local_step_rmpad is not None:
                    adj_ent_pair_local_step_rmpad = (
                        adj_ent_pair_local_step_rmpad.squeeze(0).float()
                    )
                if adj_ent_rollout_next_mean_rmpad is not None:
                    adj_ent_rollout_next_mean_rmpad = (
                        adj_ent_rollout_next_mean_rmpad.squeeze(0).float()
                    )
                if adj_ent_vocab_size_rmpad is not None:
                    adj_ent_vocab_size_rmpad = (
                        adj_ent_vocab_size_rmpad.squeeze(0).long()
                    )
                if adj_ent_pair_ids_rmpad is not None:
                    adj_ent_pair_ids_rmpad = (
                        adj_ent_pair_ids_rmpad.squeeze(0).long()
                    )
                if rollout_quality_mask_rmpad is not None:
                    rollout_quality_mask_rmpad = rollout_quality_mask_rmpad.squeeze(0).bool()
                if signed_trace_ce_role_rmpad is not None:
                    signed_trace_ce_role_rmpad = (
                        signed_trace_ce_role_rmpad.squeeze(0).to(
                            dtype=torch.int8
                        )
                    )
                if causal_wrong_ce_role_rmpad is not None:
                    causal_wrong_ce_role_rmpad = (
                        causal_wrong_ce_role_rmpad.squeeze(0).to(
                            dtype=torch.int8
                        )
                    )
                if dense_labels_rmpad is not None:
                    dense_labels_rmpad = dense_labels_rmpad.squeeze(0)
                if block_tpf_weight_rmpad is not None:
                    block_tpf_weight_rmpad = block_tpf_weight_rmpad.squeeze(0)
                if response_tpf_ce_scale_rmpad is not None:
                    response_tpf_ce_scale_rmpad = (
                        response_tpf_ce_scale_rmpad.squeeze(0)
                    )
                if response_level_ce_weight_rmpad is not None:
                    response_level_ce_weight_rmpad = (
                        response_level_ce_weight_rmpad.squeeze(0)
                    )
                if response_level_aec_pair_weight_rmpad is not None:
                    response_level_aec_pair_weight_rmpad = (
                        response_level_aec_pair_weight_rmpad.squeeze(0)
                    )

                def _append_trace_future_teacher_token_debug(
                    *,
                    rmpad_idx: torch.Tensor,
                    labels_keep: torch.Tensor,
                    teacher_prob_keep: torch.Tensor,
                    student_label_prob_keep: torch.Tensor | None,
                    teacher_source_keep: torch.Tensor | None,
                ) -> None:
                    if (
                        not token_debug_enabled
                        or response_rmpad_to_batch_pos is None
                        or teacher_prob_keep.numel() == 0
                    ):
                        return
                    remaining = (
                        teacher_token_debug_limit - len(teacher_token_debug_rows)
                        if teacher_token_debug_limit > 0
                        else int(rmpad_idx.numel())
                    )
                    if remaining <= 0:
                        return
                    take = min(int(rmpad_idx.numel()), remaining)
                    coords = response_rmpad_to_batch_pos[rmpad_idx[:take]].detach().cpu()
                    phase = micro_batch.get("_actor_replay_debug_phase", "unknown")
                    global_step_dbg = micro_batch.get("_actor_replay_debug_global_step", None)
                    batch_idx_dbg = micro_batch.get("_actor_replay_debug_batch_idx", None)
                    micro_idx_dbg = micro_batch.get("_actor_replay_debug_micro_idx", None)

                    def _prob_at(row_prob: torch.Tensor, token_id: int | None) -> float | None:
                        if token_id is None or token_id < 0 or token_id >= int(row_prob.shape[-1]):
                            return None
                        return float(row_prob[token_id].detach().float().cpu().item())

                    top_prob, top_id = teacher_prob_keep[:take].detach().float().max(dim=-1)
                    for local_i in range(take):
                        b = int(coords[local_i, 0].item())
                        pos = int(coords[local_i, 1].item())
                        if b < 0 or pos < 0:
                            continue
                        label_id = int(labels_keep[local_i].detach().cpu().item())
                        prev_id = (
                            int(responses[b, pos - 1].detach().cpu().item())
                            if pos > 0
                            else None
                        )
                        next_id = (
                            int(responses[b, pos + 1].detach().cpu().item())
                            if pos + 1 < response_length
                            else None
                        )
                        rollout_lp = None
                        if rollout_log_probs_rmpad is not None:
                            rollout_lp = float(
                                rollout_log_probs_rmpad[rmpad_idx[local_i]].detach().float().cpu().item()
                            )
                        step_key = None
                        if step_maps_rmpad is not None:
                            step_key = int(step_maps_rmpad[rmpad_idx[local_i]].detach().long().cpu().item())
                        source_norm = None
                        source_nonzero = None
                        if teacher_source_keep is not None:
                            source_value = teacher_source_keep[local_i].detach().float()
                            source_norm = float(source_value.norm().cpu().item())
                            source_nonzero = bool(source_value.abs().sum().cpu().item() > 0)
                        clean_abs_pos = seqlen - 2 * response_length + pos
                        noisy_abs_pos = seqlen - response_length + pos
                        row_prob = teacher_prob_keep[local_i]
                        teacher_label_prob = _prob_at(row_prob, label_id)
                        teacher_token_debug_rows.append(
                            {
                                "debug_kind": "trace_future_ce_teacher_token",
                                "phase": phase,
                                "global_step": global_step_dbg,
                                "batch_idx": batch_idx_dbg,
                                "micro_idx": micro_idx_dbg,
                                "sample_id": b,
                                "row_index": b,
                                "response_index": pos,
                                "rmpad_index": int(rmpad_idx[local_i].detach().cpu().item()),
                                "step_key": step_key,
                                "label_id": label_id,
                                "prev_label_id": prev_id,
                                "next_label_id": next_id,
                                "clean_input_id": int(input_ids[b, clean_abs_pos].detach().cpu().item())
                                if 0 <= clean_abs_pos < seqlen
                                else None,
                                "noisy_input_id": int(input_ids[b, noisy_abs_pos].detach().cpu().item())
                                if 0 <= noisy_abs_pos < seqlen
                                else None,
                                "clean_input_pos": int(clean_abs_pos),
                                "noisy_input_pos": int(noisy_abs_pos),
                                "rollout_logprob": rollout_lp,
                                "rollout_prob": float(torch.exp(torch.tensor(rollout_lp)).item())
                                if rollout_lp is not None and rollout_lp > -float("inf")
                                else None,
                                "teacher_label_prob": teacher_label_prob,
                                "teacher_prev_label_prob": _prob_at(row_prob, prev_id),
                                "teacher_next_label_prob": _prob_at(row_prob, next_id),
                                "teacher_top_id": int(top_id[local_i].cpu().item()),
                                "teacher_top_prob": float(top_prob[local_i].cpu().item()),
                                "teacher_top_matches_label": bool(
                                    int(top_id[local_i].cpu().item()) == label_id
                                ),
                                "student_label_prob": float(
                                    student_label_prob_keep[local_i].detach().float().cpu().item()
                                )
                                if student_label_prob_keep is not None
                                else None,
                                "logits_source_hidden_norm": source_norm,
                                "logits_source_hidden_nonzero": source_nonzero,
                            }
                        )

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = self._effective_logprob_temperature(temperature)
                    extra_args["return_dict"] = True
                if rollout_sc_signal_rmpad is not None:
                    # Same criterion as the model: modeling_llada2_moe.py checks
                    # `self.self_conditioning is None`. The L0 arm turns the adapter off on both sides,
                    # but SDARPrefixKV still captures sc_signal into the replay.
                    _sc_owner = getattr(self.actor_module, "module", self.actor_module)
                    if getattr(_sc_owner, "self_conditioning", None) is not None:
                        extra_args["self_conditioning_signal"] = rollout_sc_signal_rmpad.detach()
                    elif not globals().get("_SC_SIGNAL_DROP_ANNOUNCED"):
                        globals()["_SC_SIGNAL_DROP_ANNOUNCED"] = True
                        print("[SC-SIGNAL] the actor has no self_conditioning adapter; "
                              "dropping the signal sent by the rollout (expected for the L0 arm without a residual). "
                              "If this is not intended, the actor's use_self_conditioning is off.",
                              flush=True)
                generation_jsd_config = self.config.get("generation_consistency_jsd", {})
                generation_jsd_override = micro_batch.get("_generation_jsd_config_override", None)
                if generation_jsd_override:
                    generation_jsd_config = {**dict(generation_jsd_config), **generation_jsd_override}
                generation_jsd_objective = generation_jsd_config.get("objective", "full_jsd")
                fast_hidden_ce = (
                    generation_jsd_config.get("enable", False)
                    and generation_jsd_objective in {"clean_agree_ce", "trace_all_ce", "trace_all_rkl", "trace_future_ce"}
                )
                # Causal wrong CE requires raw (unshifted) hidden so we can derive
                # both the staircase source (positive) and causal source (wrong)
                # in one forward pass without a no_grad teacher path.
                _causal_wrong_cfg = (
                    (generation_jsd_config.get("causal_wrong_ce", {}) or {})
                )
                causal_wrong_ce_actor_enabled = (
                    bool(_causal_wrong_cfg.get("enable", False))
                    and generation_jsd_objective == "trace_all_ce"
                    and generation_jsd_config.get("enable", False)
                )
                if causal_wrong_ce_actor_enabled and causal_wrong_ce_role_rmpad is None:
                    raise RuntimeError(
                        "causal_wrong_ce actor forward is missing per-token roles; "
                        "ensure causal_wrong_ce_role is present in the micro-batch"
                    )
                if rollout_logits_source_hidden_rmpad is not None and not generation_jsd_config.get("enable", False):
                    extra_args["logits_source_hidden"] = rollout_logits_source_hidden_rmpad
                labels_for_model = None if generation_jsd_config.get("enable", False) else labels_rmpad_2d

                ####################
                # Prepare metadata for attention
                ####################
                if not is_mask_all_zero:
                    L1 = micro_batch["responses"].shape[1]
                    L0 = attention_mask.shape[1] - 2 * L1

                    L0_list = attention_mask[:, :L0].sum(dim=1).view(-1)
                    L1_list = attention_mask[:, L0:L0 + L1].sum(dim=1).view(-1)

                    max_seqlen = attention_mask.sum(dim=1).max().item()
                    extra_args["preexpanded_attention_metadata"] = {
                        "cu_seqlens": cu_seqlens,
                        "L0": L0_list.to(torch.int32),
                        "L1": L1_list.to(torch.int32),
                        "staircase_size": self.block_length,
                        "max_seqlen": max_seqlen,
                        "backend": "fast",
                        # SDAR's in-model logits-source shift uses full-sequence
                        # indices. Under Ulysses we apply it after a
                        # differentiable hidden gather, then slice it back.
                        # When causal_wrong_ce is enabled we need raw hidden to
                        # derive both the staircase and causal sources, so we
                        # always disable the in-model shift.
                        "shift_noisy_logits": (
                            False
                            if causal_wrong_ce_actor_enabled
                            else not self.use_ulysses_sp
                        ),
                    }
                    # The input is already laid out as
                    # [prompt | clean response | noisy response], so the
                    # target SDAR model should use the preexpanded diffusion
                    # attention path instead of a padded 2D mask.
                    attention_mask = None
                else:
                    attention_mask = None
                ####################
                # Prepare metadata for attention
                ####################

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=attention_mask,
                    position_ids=position_ids_rmpad,
                    labels=labels_for_model,
                    return_full_logits=not fast_hidden_ce,
                    return_logits_source_hidden=fast_hidden_ce,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating


                if self.use_fused_kernels:
                    raise RuntimeError
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size) or fast-path hidden source
                    if self.use_ulysses_sp and not is_mask_all_zero:
                        if not fast_hidden_ce:
                            raise NotImplementedError(
                                "SDAR preexpanded Ulysses currently supports "
                                "hidden-source CE objectives only"
                            )
                        metadata = extra_args[
                            "preexpanded_attention_metadata"
                        ]
                        gathered_hidden = all_gather_sequence_with_grad(
                            logits_rmpad,
                            gather_dim=0,
                        )
                        shifted_hidden = (
                            _shift_preexpanded_logits_source_impl(
                                gathered_hidden,
                                cu_seqlens=metadata["cu_seqlens"],
                                L0=metadata["L0"],
                                L1=metadata["L1"],
                                staircase_size=int(
                                    metadata["staircase_size"]
                                ),
                            )
                        )
                        # For causal_wrong_ce: derive causal (staircase_size=1)
                        # source from the same gathered raw hidden; slice back
                        # using the same SP partition as the staircase source.
                        causal_wrong_hidden_sp = None
                        if causal_wrong_ce_actor_enabled:
                            causal_wrong_hidden_full = (
                                _shift_preexpanded_logits_source_impl(
                                    gathered_hidden,
                                    cu_seqlens=metadata["cu_seqlens"],
                                    L0=metadata["L0"],
                                    L1=metadata["L1"],
                                    staircase_size=1,
                                )
                            )
                            causal_wrong_hidden_sp = slice_input_tensor(
                                causal_wrong_hidden_full,
                                dim=0,
                                padding=False,
                            )
                        logits_rmpad = slice_input_tensor(
                            shifted_hidden,
                            dim=0,
                            padding=False,
                        )
                    else:
                        # Non-SP path: derive causal source if needed.
                        causal_wrong_hidden_sp = None
                        if causal_wrong_ce_actor_enabled and not is_mask_all_zero:
                            metadata = extra_args.get(
                                "preexpanded_attention_metadata", {}
                            )
                            causal_wrong_hidden_sp = (
                                _shift_preexpanded_logits_source_impl(
                                    logits_rmpad,
                                    cu_seqlens=metadata["cu_seqlens"],
                                    L0=metadata["L0"],
                                    L1=metadata["L1"],
                                    staircase_size=1,
                                )
                            )
                        # Apply normal in-process staircase shift when not SP
                        # and shift_noisy_logits was NOT handled inside the model
                        # (the preexpanded_attention_metadata path already applied
                        # it via shift_noisy_logits=True; for causal_wrong_ce we
                        # set shift_noisy_logits=False so we must shift here).
                        if (
                            causal_wrong_ce_actor_enabled
                            and not is_mask_all_zero
                            and fast_hidden_ce
                        ):
                            metadata = extra_args.get(
                                "preexpanded_attention_metadata", {}
                            )
                            logits_rmpad = (
                                _shift_preexpanded_logits_source_impl(
                                    logits_rmpad,
                                    cu_seqlens=metadata["cu_seqlens"],
                                    L0=metadata["L0"],
                                    L1=metadata["L1"],
                                    staircase_size=int(
                                        metadata["staircase_size"]
                                    ),
                                )
                            )
                    logprob_temperature = self._effective_logprob_temperature(temperature)
                    valid_label_mask = labels_rmpad.ne(-100)
                    if generation_jsd_config.get("enable", False) and generation_jsd_objective == "tpf_reinforce":
                        valid_label_mask = torch.zeros_like(valid_label_mask)
                    generation_jsd_loss = None
                    generation_jsd_metrics = {}
                    signed_trace_ce_enabled = False
                    signed_positive_ce_sum = None
                    signed_negative_ce_sum = None
                    signed_positive_ce_count = 0
                    signed_negative_ce_count = 0
                    causal_wrong_positive_ce_sum = None
                    causal_wrong_neg_ce_sum = None
                    causal_wrong_positive_ce_count = 0
                    causal_wrong_neg_ce_count = 0
                    _adj_ent_hinge_sum = None
                    _adj_ent_pair_count = 0
                    _adj_ent_pair_weight_sum = 0.0
                    # Pre-renorm eligible pair-weight sum: chunking-independent
                    # quantity used for the prepass/actual denominator
                    # consistency assertion (equals post-renorm sum when no
                    # entropy gate is active).
                    _adj_ent_pair_weight_pre_renorm = 0.0
                    _adj_ent_metrics: dict = {}
                    _adj_ent_cfg_dict = generation_jsd_config.get(
                        "adjacent_entropy_contraction", {}
                    ) or {}
                    _adj_ent_enable = bool(
                        _adj_ent_cfg_dict.get("enable", False)
                    )
                    _adj_ent_response_level = bool(
                        _adj_ent_cfg_dict.get("response_level", False)
                    )
                    # AEC only ever demands rising confidence. Applying it to the
                    # rows the signed objective is pushing down would train the
                    # model to be confidently wrong, so drop them from the pair
                    # mask before any pair statistics are accumulated.
                    if (
                        _adj_ent_enable
                        and adj_ent_pair_mask_rmpad is not None
                        and signed_trace_ce_role_rmpad is not None
                    ):
                        _adj_role_mode = str(
                            _adj_ent_cfg_dict.get(
                                "signed_role_filter", "exclude_negative"
                            )
                            or "exclude_negative"
                        ).strip().lower()
                        if _adj_role_mode == "exclude_negative":
                            adj_ent_pair_mask_rmpad = (
                                adj_ent_pair_mask_rmpad
                                & signed_trace_ce_role_rmpad.ne(-1)
                            )
                        elif _adj_role_mode == "positive_only":
                            adj_ent_pair_mask_rmpad = (
                                adj_ent_pair_mask_rmpad
                                & signed_trace_ce_role_rmpad.eq(1)
                            )
                        elif _adj_role_mode != "none":
                            raise ValueError(
                                "adjacent_entropy_contraction."
                                "signed_role_filter must be exclude_negative, "
                                f"positive_only, or none, got {_adj_role_mode!r}"
                            )
                    if (
                        _adj_ent_enable
                        and _adj_ent_response_level
                        and response_level_aec_pair_weight_rmpad is None
                    ):
                        raise RuntimeError(
                            "response-level AEC forward is missing pair weights"
                        )
                    _entropy_rkl_probe_cfg = generation_jsd_config.get(
                        "entropy_gated_rkl_probe", {}
                    ) or {}
                    _entropy_gated_rkl_cfg = generation_jsd_config.get(
                        "entropy_gated_rkl", {}
                    ) or {}
                    _entropy_gated_rkl_enable = bool(
                        _entropy_gated_rkl_cfg.get("enable", False)
                    )
                    _entropy_gated_rkl_threshold = float(
                        _entropy_gated_rkl_cfg.get(
                            "entropy_threshold",
                            0.10,
                        )
                    )
                    _entropy_gated_rkl_cap = float(
                        _entropy_gated_rkl_cfg.get("rkl_cap", 5.0)
                    )
                    _entropy_gated_rkl_cad_mode = bool(
                        _entropy_gated_rkl_cfg.get("cad_mode", False)
                    )
                    _entropy_gated_rkl_fkl_coef = float(
                        _entropy_gated_rkl_cfg.get("fkl_coef", 1.0)
                    )
                    _entropy_gated_rkl_cj_power = float(
                        _entropy_gated_rkl_cfg.get("cj_power", 1.0)
                    )
                    _entropy_rkl_probe_enable = bool(
                        _entropy_rkl_probe_cfg.get("enable", False)
                    )
                    _entropy_rkl_teacher_enable = (
                        _entropy_rkl_probe_enable
                        or _entropy_gated_rkl_enable
                    )
                    _entropy_rkl_probe_thresholds_list = [
                        float(value)
                        for value in _entropy_rkl_probe_cfg.get(
                            "thresholds",
                            DEFAULT_ENTROPY_THRESHOLDS,
                        )
                    ]
                    if _entropy_gated_rkl_enable and not any(
                        math.isclose(
                            value,
                            _entropy_gated_rkl_threshold,
                            rel_tol=0.0,
                            abs_tol=1e-12,
                        )
                        for value in _entropy_rkl_probe_thresholds_list
                    ):
                        _entropy_rkl_probe_thresholds_list.append(
                            _entropy_gated_rkl_threshold
                        )
                    _entropy_rkl_probe_thresholds = tuple(
                        sorted(set(_entropy_rkl_probe_thresholds_list))
                    )
                    if generation_jsd_config.get("enable", False) and generation_jsd_objective != "tpf_reinforce":
                        objective = generation_jsd_objective
                        signed_trace_ce_enabled = (
                            objective == "trace_all_ce"
                            and bool(
                                (
                                    generation_jsd_config.get(
                                        "signed_trace_ce", {}
                                    )
                                    or {}
                                ).get("enable", False)
                            )
                        )
                        if (
                            signed_trace_ce_enabled
                            and signed_trace_ce_role_rmpad is None
                        ):
                            raise RuntimeError(
                                "signed_trace_ce actor forward is missing roles"
                            )
                        jsd_temperature = float(generation_jsd_config.get("temperature", logprob_temperature))
                        teacher_source = generation_jsd_config.get("teacher_source", "rollout_final")
                        teacher_index_map = None
                        trace_future_hard_label_ce = (
                            objective == "trace_future_ce"
                            and bool(generation_jsd_config.get("trace_future_hard_label_ce", False))
                        )
                        trace_future_clean_filter_requested = (
                            objective == "trace_future_ce"
                            and bool(generation_jsd_config.get("trace_future_clean_filter", False))
                        )
                        trace_future_clean_consistency_requested = (
                            objective == "trace_future_ce"
                            and bool(generation_jsd_config.get("trace_future_clean_consistency_enable", False))
                        )
                        if objective == "trace_all_ce" and teacher_source != "clean_causal":
                            jsd_mask = valid_label_mask
                        elif teacher_source == "clean_causal" and not trace_future_hard_label_ce:
                            teacher_extra_args = dict(extra_args)
                            teacher_metadata = dict(teacher_extra_args["preexpanded_attention_metadata"])
                            teacher_metadata["staircase_size"] = int(
                                generation_jsd_config.get("teacher_staircase_size", 1)
                            )
                            teacher_metadata["shift_noisy_logits"] = False
                            teacher_extra_args["preexpanded_attention_metadata"] = teacher_metadata
                            teacher_extra_args.pop("self_conditioning_signal", None)
                            teacher_extra_args.pop("logits_source_hidden", None)
                            with torch.no_grad():
                                teacher_output = self.actor_module(
                                    input_ids=input_ids_rmpad,
                                    attention_mask=attention_mask,
                                    position_ids=position_ids_rmpad,
                                    labels=None,
                                    return_full_logits=not fast_hidden_ce,
                                    return_logits_source_hidden=fast_hidden_ce,
                                    **multi_modal_inputs,
                                    use_cache=False,
                                    **teacher_extra_args,
                                )
                                teacher_logits_all = teacher_output.logits.squeeze(0).detach()
                            teacher_index_map = torch.full_like(labels_rmpad, -1, dtype=torch.long)
                            metadata = extra_args["preexpanded_attention_metadata"]
                            cu = metadata["cu_seqlens"].detach().cpu().tolist()
                            l0_values = metadata["L0"].detach().cpu().tolist()
                            l1_values = metadata["L1"].detach().cpu().tolist()
                            for doc_idx, (l0_i, l1_i) in enumerate(zip(l0_values, l1_values, strict=True)):
                                start_i = int(cu[doc_idx])
                                l0_i = int(l0_i)
                                l1_i = int(l1_i)
                                clean_start = start_i + l0_i
                                noisy_start = start_i + l0_i + l1_i
                                if l1_i > 0:
                                    positions = torch.arange(l1_i, device=labels_rmpad.device)
                                    teacher_index_map[noisy_start : noisy_start + l1_i] = clean_start + positions - 1
                            jsd_mask = valid_label_mask & teacher_index_map.ge(0)
                        else:
                            if rollout_logits_source_hidden_rmpad is None and not trace_future_hard_label_ce:
                                raise RuntimeError(
                                    "actor.generation_consistency_jsd requires rollout_logits_source_hidden in the batch"
                                )
                            teacher_logits_rmpad = None
                            if objective not in {"gated_ce", "trace_future_ce"}:
                                teacher_extra_args = dict(extra_args)
                                teacher_extra_args["logits_source_hidden"] = rollout_logits_source_hidden_rmpad.detach()
                                with torch.no_grad():
                                    teacher_output = self.actor_module(
                                        input_ids=input_ids_rmpad,
                                        attention_mask=attention_mask,
                                        position_ids=position_ids_rmpad,
                                        labels=None,
                                        return_full_logits=True,
                                        **multi_modal_inputs,
                                        use_cache=False,
                                        **teacher_extra_args,
                                    )
                                    teacher_logits_rmpad = teacher_output.logits.squeeze(0).detach()
                            if trace_future_hard_label_ce:
                                jsd_mask = valid_label_mask
                            elif rollout_logits_source_hidden_rmpad is not None:
                                source_hidden = rollout_logits_source_hidden_rmpad.squeeze(0).detach().to(
                                    device=logits_rmpad.device
                                )
                                jsd_mask = valid_label_mask & (source_hidden.float().abs().sum(dim=-1) > 0)
                            else:
                                jsd_mask = valid_label_mask
                        actor_module_unwrapped = getattr(self.actor_module, "module", self.actor_module)
                        lm_head = getattr(actor_module_unwrapped, "lm_head", None)
                        vocab_size = lm_head.weight.shape[0] if fast_hidden_ce and lm_head is not None else logits_rmpad.shape[-1]
                        if (
                            (trace_future_clean_filter_requested or trace_future_clean_consistency_requested)
                            and teacher_index_map is None
                        ):
                            teacher_extra_args = dict(extra_args)
                            teacher_metadata = dict(teacher_extra_args["preexpanded_attention_metadata"])
                            teacher_metadata["staircase_size"] = int(
                                generation_jsd_config.get("teacher_staircase_size", 1)
                            )
                            teacher_metadata["shift_noisy_logits"] = False
                            teacher_extra_args["preexpanded_attention_metadata"] = teacher_metadata
                            teacher_extra_args.pop("self_conditioning_signal", None)
                            teacher_extra_args.pop("logits_source_hidden", None)
                            with torch.no_grad():
                                teacher_output = self.actor_module(
                                    input_ids=input_ids_rmpad,
                                    attention_mask=attention_mask,
                                    position_ids=position_ids_rmpad,
                                    labels=None,
                                    return_full_logits=not fast_hidden_ce,
                                    return_logits_source_hidden=fast_hidden_ce,
                                    **multi_modal_inputs,
                                    use_cache=False,
                                    **teacher_extra_args,
                                )
                                teacher_logits_all = teacher_output.logits.squeeze(0).detach()
                            teacher_index_map = torch.full_like(labels_rmpad, -1, dtype=torch.long)
                            metadata = extra_args["preexpanded_attention_metadata"]
                            cu = metadata["cu_seqlens"].detach().cpu().tolist()
                            l0_values = metadata["L0"].detach().cpu().tolist()
                            l1_values = metadata["L1"].detach().cpu().tolist()
                            for doc_idx, (l0_i, l1_i) in enumerate(zip(l0_values, l1_values, strict=True)):
                                start_i = int(cu[doc_idx])
                                l0_i = int(l0_i)
                                l1_i = int(l1_i)
                                clean_start = start_i + l0_i
                                noisy_start = start_i + l0_i + l1_i
                                if l1_i > 0:
                                    positions = torch.arange(l1_i, device=labels_rmpad.device)
                                    teacher_index_map[noisy_start : noisy_start + l1_i] = clean_start + positions - 1
                            jsd_mask = jsd_mask & teacher_index_map.ge(0)
                        jsd_mask = jsd_mask & labels_rmpad.ge(0) & labels_rmpad.lt(vocab_size)
                        if rollout_quality_mask_rmpad is not None:
                            jsd_mask = jsd_mask & rollout_quality_mask_rmpad
                        if signed_trace_ce_enabled:
                            jsd_mask = (
                                jsd_mask
                                & signed_trace_ce_role_rmpad.ne(0)
                            )
                        if causal_wrong_ce_actor_enabled and causal_wrong_ce_role_rmpad is not None:
                            jsd_mask = (
                                jsd_mask
                                & causal_wrong_ce_role_rmpad.ne(0)
                            )
                        threshold = float(generation_jsd_config.get("teacher_max_prob_threshold", 0.0))
                        row_chunk = int(generation_jsd_config.get("row_chunk_size", 8))
                        active_idx = torch.nonzero(jsd_mask, as_tuple=False).flatten()
                        jsd_terms = []
                        teacher_max_values = []
                        teacher_entropy_values = []
                        student_entropy_values = []
                        teacher_label_prob_values = []
                        teacher_top_match_values = []
                        student_label_prob_values = []
                        trace_all_ce_loss_sum = None
                        trace_all_ce_count = 0
                        trace_all_ce_weight_sum = 0.0
                        trace_all_ce_weight_values = []
                        signed_positive_ce_sum = None
                        signed_negative_ce_sum = None
                        signed_positive_ce_count = 0
                        signed_negative_ce_count = 0
                        causal_wrong_positive_ce_sum = None
                        causal_wrong_neg_ce_sum = None
                        causal_wrong_positive_ce_count = 0
                        causal_wrong_neg_ce_count = 0
                        _block_tpf_ce_cfg = generation_jsd_config.get("block_tpf_ce_weight", {}) or {}
                        _block_tpf_ce_enabled = (
                            bool(_block_tpf_ce_cfg.get("enable", False))
                            and block_tpf_weight_rmpad is not None
                        )
                        _response_tpf_ce_cfg = (
                            generation_jsd_config.get(
                                "response_tpf_ce_scale",
                                {},
                            )
                            or {}
                        )
                        _response_tpf_ce_enabled = (
                            bool(_response_tpf_ce_cfg.get("enable", False))
                            and response_tpf_ce_scale_rmpad is not None
                        )
                        _response_level_ce_cfg = (
                            generation_jsd_config.get(
                                "response_level_ce",
                                {},
                            )
                            or {}
                        )
                        _response_level_ce_enabled = (
                            bool(_response_level_ce_cfg.get("enable", False))
                            and response_level_ce_weight_rmpad is not None
                        )
                        if (
                            bool(_response_level_ce_cfg.get("enable", False))
                            and not _response_level_ce_enabled
                        ):
                            raise RuntimeError(
                                "response_level_ce forward is missing token weights"
                            )
                        response_tpf_ce_scale_values = []
                        response_level_ce_weight_values = []
                        trace_clean_filter_total = 0
                        trace_clean_filter_kept = 0
                        trace_clean_filter_enabled = (
                            (
                                objective == "trace_all_ce"
                                and teacher_source == "clean_causal"
                            )
                            or trace_future_clean_filter_requested
                        ) and teacher_index_map is not None
                        trace_clean_consistency_enabled = (
                            trace_future_clean_consistency_requested
                            and teacher_index_map is not None
                        )
                        trace_clean_threshold = float(
                            generation_jsd_config.get("student_final_prob_threshold", threshold)
                        )
                        trace_clean_require_top_match = bool(
                            generation_jsd_config.get("clean_filter_require_top_match", True)
                        )
                        clean_consistency_high_threshold = float(
                            generation_jsd_config.get(
                                "clean_consistency_high_prob_threshold",
                                trace_clean_threshold,
                            )
                        )
                        clean_consistency_low_threshold = float(
                            generation_jsd_config.get("clean_consistency_low_prob_threshold", 0.1)
                        )
                        clean_consistency_pos_coef = float(
                            generation_jsd_config.get("clean_consistency_pos_coef", 0.0)
                        )
                        clean_consistency_neg_coef = float(
                            generation_jsd_config.get("clean_consistency_neg_coef", 0.0)
                        )
                        clean_consistency_neg_prob_clip = float(
                            generation_jsd_config.get("clean_consistency_neg_prob_clip", 0.99)
                        )
                        clean_consistency_neg_loss_clip = float(
                            generation_jsd_config.get("clean_consistency_neg_loss_clip", 5.0)
                        )
                        logit_clip = float(generation_jsd_config.get("logit_clip", 50.0))
                        prob_eps = float(generation_jsd_config.get("prob_eps", 1e-6))
                        selected_vocab_size = int(generation_jsd_config.get("selected_vocab_size", 0) or 0)
                        if _entropy_rkl_teacher_enable:
                            if objective != "trace_all_ce":
                                raise RuntimeError(
                                    "entropy-gated RKL requires objective=trace_all_ce"
                                )
                            if teacher_source != "rollout_final":
                                raise RuntimeError(
                                    "entropy-gated RKL requires teacher_source=rollout_final"
                                )
                            if not fast_hidden_ce or lm_head is None:
                                raise RuntimeError(
                                    "entropy-gated RKL requires the full-vocab "
                                    "hidden-source trace-all-CE path"
                                )
                            if selected_vocab_size != 0:
                                raise RuntimeError(
                                    "entropy-gated RKL requires selected_vocab_size=0"
                                )
                            if (
                                _entropy_rkl_probe_enable
                                and not _entropy_rkl_probe_thresholds
                            ):
                                raise RuntimeError(
                                    "entropy_gated_rkl_probe requires at least one threshold"
                                )
                            if _entropy_gated_rkl_enable:
                                if not (
                                    math.isfinite(
                                        _entropy_gated_rkl_threshold
                                    )
                                    and 0.0
                                    <= _entropy_gated_rkl_threshold
                                    <= 1.0
                                ):
                                    raise RuntimeError(
                                        "entropy_gated_rkl.entropy_threshold "
                                        "must be finite and in [0,1]"
                                    )
                                if not (
                                    math.isfinite(_entropy_gated_rkl_cap)
                                    and _entropy_gated_rkl_cap > 0.0
                                ):
                                    raise RuntimeError(
                                        "entropy_gated_rkl.rkl_cap must be "
                                        "finite and positive"
                                    )
                                if not (
                                    math.isfinite(
                                        _entropy_gated_rkl_fkl_coef
                                    )
                                    and _entropy_gated_rkl_fkl_coef >= 0.0
                                ):
                                    raise RuntimeError(
                                        "entropy_gated_rkl.fkl_coef must be "
                                        "finite and non-negative"
                                    )
                                if not (
                                    math.isfinite(
                                        _entropy_gated_rkl_cj_power
                                    )
                                    and _entropy_gated_rkl_cj_power >= 0.0
                                ):
                                    raise RuntimeError(
                                        "entropy_gated_rkl.cj_power must be "
                                        "finite and non-negative"
                                    )
                        trace_ce_entropy_coef = float(generation_jsd_config.get("entropy_coef", 0.0) or 0.0)
                        trace_ce_entropy_mode = str(
                            generation_jsd_config.get("entropy_mode", "penalty") or "penalty"
                        ).strip().lower()
                        raw_trace_ce_entropy_target = generation_jsd_config.get(
                            "entropy_target", None
                        )
                        trace_ce_entropy_target = (
                            None
                            if raw_trace_ce_entropy_target is None
                            else float(raw_trace_ce_entropy_target)
                        )
                        if trace_ce_entropy_coef < 0.0:
                            raise ValueError(
                                "generation_consistency_jsd.entropy_coef must be non-negative; "
                                "use entropy_mode=penalty or bonus to choose the sign"
                            )
                        if trace_ce_entropy_mode not in {"penalty", "bonus"}:
                            raise ValueError(
                                "generation_consistency_jsd.entropy_mode must be 'penalty' or 'bonus', "
                                f"got {trace_ce_entropy_mode!r}"
                            )
                        if trace_ce_entropy_target is not None:
                            if (
                                not math.isfinite(trace_ce_entropy_target)
                                or trace_ce_entropy_target < 0.0
                            ):
                                raise ValueError(
                                    "generation_consistency_jsd.entropy_target must be "
                                    "finite and non-negative"
                                )
                            if trace_ce_entropy_mode != "penalty":
                                raise ValueError(
                                    "generation_consistency_jsd.entropy_target is only "
                                    "supported with entropy_mode=penalty"
                                )
                        trace_ce_entropy_sign = 1.0 if trace_ce_entropy_mode == "penalty" else -1.0
                        trace_ce_entropy_enabled = (
                            trace_ce_entropy_coef != 0.0
                            and objective in {"trace_all_ce", "trace_future_ce"}
                        )
                        if trace_ce_entropy_enabled and selected_vocab_size > 0:
                            raise RuntimeError(
                                "generation_consistency_jsd.entropy_coef requires selected_vocab_size=0 "
                                "so entropy is computed over the full vocabulary"
                            )
                        if trace_clean_consistency_enabled and selected_vocab_size > 0:
                            raise RuntimeError(
                                "trace_future_clean_consistency_enable requires selected_vocab_size=0 "
                                "so unlikelihood uses full-vocab probability"
                            )
                        clean_pos_terms = []
                        clean_neg_terms = []
                        clean_pos_prob_values = []
                        clean_neg_prob_values = []
                        clean_consistency_total = 0
                        clean_consistency_pos_count = 0
                        clean_consistency_neg_count = 0
                        trace_ce_entropy_sum = None
                        trace_ce_entropy_count = 0
                        _entropy_rkl_probe_stats = None
                        _entropy_gated_rkl_report = torch.zeros(
                            ENTROPY_GATED_RKL_REPORT_SLOTS,
                            dtype=torch.float64,
                            device=logits_rmpad.device,
                        )
                        _adj_ent_num_pairs = 0
                        _adj_ent_entropy_sums = None
                        _adj_ent_token_counts = None
                        _adj_ent_prev_sums = None
                        _adj_ent_rollout_next_sums = None
                        _adj_ent_response_weight_sums = None
                        _adj_ent_local_step_sums = None
                        if _adj_ent_enable:
                            if not fast_hidden_ce or lm_head is None:
                                raise RuntimeError(
                                    "adjacent entropy contraction requires "
                                    "the trace-all-CE hidden-source fast path"
                                )
                            if selected_vocab_size > 0:
                                raise RuntimeError(
                                    "adjacent entropy contraction requires "
                                    "selected_vocab_size=0"
                                )
                            if (
                                adj_ent_pair_mask_rmpad is None
                                or adj_ent_prev_mean_rmpad is None
                                or adj_ent_rollout_next_mean_rmpad is None
                                or adj_ent_vocab_size_rmpad is None
                                or adj_ent_pair_ids_rmpad is None
                                or adj_ent_pair_ids_values is None
                            ):
                                raise RuntimeError(
                                    "adjacent entropy contraction forward is "
                                    "missing pair mask, previous mean, producer "
                                    "vocabulary size, or pair IDs"
                                )
                            if (
                                float(
                                    _adj_ent_cfg_dict.get("rho_ramp_per_step", 0.0)
                                    or 0.0
                                )
                                > 0.0
                                and adj_ent_pair_local_step_rmpad is None
                            ):
                                raise RuntimeError(
                                    "rho_ramp_per_step > 0 but "
                                    "adjacent_entropy_pair_local_step never "
                                    "reached the actor micro-batch (transport "
                                    "gap); refusing to train with silent k=0"
                                )
                            _full_pair_ids = adj_ent_pair_ids_values.long()
                            _valid_full_pair_ids = _full_pair_ids[
                                _full_pair_ids.ge(0)
                            ]
                            if _valid_full_pair_ids.numel() > 0:
                                _adj_ent_num_pairs = (
                                    int(_valid_full_pair_ids.max().item()) + 1
                                )
                            _adj_ent_entropy_sums = torch.zeros(
                                _adj_ent_num_pairs,
                                device=logits_rmpad.device,
                                dtype=torch.float32,
                            ) + logits_rmpad.float().sum() * 0.0
                            _adj_ent_token_counts = torch.zeros_like(
                                _adj_ent_entropy_sums
                            )
                            _adj_ent_prev_sums = torch.zeros_like(
                                _adj_ent_entropy_sums
                            )
                            _adj_ent_rollout_next_sums = torch.zeros_like(
                                _adj_ent_entropy_sums
                            )
                            _adj_ent_response_weight_sums = torch.zeros_like(
                                _adj_ent_entropy_sums
                            )
                            _adj_ent_local_step_sums = torch.zeros_like(
                                _adj_ent_entropy_sums
                            )

                        def _accumulate_trace_ce_entropy(entropy_values: torch.Tensor) -> None:
                            nonlocal trace_ce_entropy_sum, trace_ce_entropy_count
                            if not trace_ce_entropy_enabled or entropy_values.numel() == 0:
                                return
                            entropy_values = torch.nan_to_num(
                                entropy_values,
                                nan=0.0,
                                posinf=0.0,
                                neginf=0.0,
                            )
                            entropy_sum = entropy_values.sum()
                            trace_ce_entropy_sum = (
                                entropy_sum
                                if trace_ce_entropy_sum is None
                                else trace_ce_entropy_sum + entropy_sum
                            )
                            trace_ce_entropy_count += int(entropy_values.numel())

                        for idx_start in range(0, active_idx.numel(), row_chunk):
                            idx = active_idx[idx_start : idx_start + row_chunk]
                            clean_keep_mask = None
                            clean_pos_mask = None
                            clean_neg_mask = None
                            clean_prob_for_idx = None
                            if trace_clean_filter_enabled or trace_clean_consistency_enabled:
                                labels_valid_for_clean = labels_rmpad[idx].long()
                                label_in_range_for_clean = labels_valid_for_clean.ge(0) & labels_valid_for_clean.lt(vocab_size)
                                if trace_clean_filter_enabled:
                                    clean_keep_mask = torch.zeros_like(label_in_range_for_clean, dtype=torch.bool)
                                if trace_clean_consistency_enabled:
                                    clean_pos_mask = torch.zeros_like(label_in_range_for_clean, dtype=torch.bool)
                                    clean_neg_mask = torch.zeros_like(label_in_range_for_clean, dtype=torch.bool)
                                    clean_prob_for_idx = torch.zeros_like(
                                        label_in_range_for_clean,
                                        dtype=torch.float32,
                                    )
                                if label_in_range_for_clean.any():
                                    labels_clean = labels_valid_for_clean[label_in_range_for_clean]
                                    teacher_rows = teacher_logits_all[teacher_index_map[idx][label_in_range_for_clean]]
                                    if fast_hidden_ce and lm_head is not None and teacher_rows.shape[-1] == lm_head.weight.shape[1]:
                                        clean_logits = torch.nn.functional.linear(
                                            teacher_rows.to(dtype=lm_head.weight.dtype),
                                            lm_head.weight,
                                            lm_head.bias,
                                        ).float()
                                    else:
                                        clean_logits = teacher_rows.float()
                                    clean_logits = clean_logits.div(jsd_temperature).clamp(min=-logit_clip, max=logit_clip)
                                    clean_prob = torch.nn.functional.softmax(clean_logits, dim=-1)
                                    clean_final_prob = clean_prob.gather(
                                        dim=-1, index=labels_clean.unsqueeze(-1)
                                    ).squeeze(-1)
                                    if trace_clean_filter_enabled:
                                        keep_clean = clean_final_prob.gt(trace_clean_threshold)
                                        if trace_clean_require_top_match:
                                            keep_clean = keep_clean & clean_prob.argmax(dim=-1).eq(labels_clean)
                                        clean_keep_mask[label_in_range_for_clean] = keep_clean
                                        trace_clean_filter_total += int(labels_clean.numel())
                                        trace_clean_filter_kept += int(keep_clean.sum().detach().item())
                                    if trace_clean_consistency_enabled:
                                        pos_clean = clean_final_prob.gt(clean_consistency_high_threshold)
                                        neg_clean = clean_final_prob.lt(clean_consistency_low_threshold)
                                        clean_pos_mask[label_in_range_for_clean] = pos_clean
                                        clean_neg_mask[label_in_range_for_clean] = neg_clean
                                        clean_prob_for_idx[label_in_range_for_clean] = clean_final_prob.float()
                                        clean_consistency_total += int(labels_clean.numel())
                                        clean_consistency_pos_count += int(pos_clean.sum().detach().item())
                                        clean_consistency_neg_count += int(neg_clean.sum().detach().item())
                            if (
                                objective in {"trace_all_ce", "trace_all_rkl", "trace_future_ce"}
                                and fast_hidden_ce
                                and selected_vocab_size > 0
                                and lm_head is not None
                                and logits_rmpad.shape[-1] == lm_head.weight.shape[1]
                            ):
                                labels_valid = labels_rmpad[idx].long()
                                label_in_range = labels_valid.ge(0) & labels_valid.lt(vocab_size)
                                if clean_keep_mask is not None:
                                    label_in_range = label_in_range & clean_keep_mask
                                if label_in_range.any():
                                    labels_keep = labels_valid[label_in_range]
                                    n_random = max(selected_vocab_size - int(labels_keep.numel()), 0)
                                    if n_random > 0:
                                        random_vocab = torch.randint(
                                            low=0,
                                            high=vocab_size,
                                            size=(n_random,),
                                            device=labels_keep.device,
                                            dtype=labels_keep.dtype,
                                        )
                                        vocab_ids = torch.unique(torch.cat([labels_keep, random_vocab], dim=0))
                                    else:
                                        vocab_ids = torch.unique(labels_keep)
                                    source_keep = logits_rmpad[idx][label_in_range].to(dtype=lm_head.weight.dtype)
                                    weight = lm_head.weight.index_select(0, vocab_ids).to(dtype=source_keep.dtype)
                                    bias = (
                                        lm_head.bias.index_select(0, vocab_ids).to(dtype=source_keep.dtype)
                                        if lm_head.bias is not None
                                        else None
                                    )
                                    student_logits = torch.nn.functional.linear(source_keep, weight, bias).float()
                                    student_logits = student_logits.div(jsd_temperature).clamp(
                                        min=-logit_clip, max=logit_clip
                                    )
                                    target_idx = labels_keep.unsqueeze(-1).eq(vocab_ids.unsqueeze(0)).to(torch.long).argmax(dim=-1)
                                    if objective == "trace_future_ce":
                                        if trace_future_hard_label_ce:
                                            ce = torch.nn.functional.cross_entropy(
                                                student_logits,
                                                target_idx,
                                                reduction="none",
                                            )
                                            ce = torch.nan_to_num(ce, nan=0.0, posinf=0.0, neginf=0.0)
                                            jsd_terms.append(ce)
                                            teacher_max_values.append(torch.ones_like(ce.detach()))
                                            teacher_entropy_values.append(torch.zeros_like(ce.detach()))
                                            student_entropy_values.append(torch.zeros_like(ce.detach()))
                                            continue
                                        if rollout_logits_source_hidden_rmpad is None:
                                            raise RuntimeError(
                                                "generation_consistency_jsd.objective=trace_future_ce requires future rollout_logits_source_hidden"
                                            )
                                        teacher_source_rows = rollout_logits_source_hidden_rmpad.squeeze(0).detach()[idx][label_in_range]
                                        if teacher_source_rows.shape[-1] == lm_head.weight.shape[1]:
                                            teacher_logits = torch.nn.functional.linear(
                                                teacher_source_rows.to(dtype=lm_head.weight.dtype),
                                                weight,
                                                bias,
                                            ).float()
                                        else:
                                            teacher_logits = teacher_source_rows.index_select(-1, vocab_ids).float()
                                        teacher_logits = teacher_logits.div(jsd_temperature).clamp(
                                            min=-logit_clip, max=logit_clip
                                        )
                                        student_logp = torch.nn.functional.log_softmax(student_logits, dim=-1)
                                        teacher_logp = torch.nn.functional.log_softmax(teacher_logits.detach(), dim=-1)
                                        teacher_prob = teacher_logp.exp()
                                        ce = -(teacher_prob * student_logp).sum(dim=-1)
                                        teacher_entropy = -(teacher_prob * teacher_logp).sum(dim=-1)
                                        student_prob = student_logp.exp()
                                        student_entropy = -(student_prob * student_logp).sum(dim=-1)
                                        teacher_label_prob = teacher_prob.gather(
                                            dim=-1, index=target_idx.unsqueeze(-1)
                                        ).squeeze(-1)
                                        student_label_prob = student_prob.gather(
                                            dim=-1, index=target_idx.unsqueeze(-1)
                                        ).squeeze(-1)
                                        teacher_top_match = teacher_prob.argmax(dim=-1).eq(target_idx)
                                        jsd_terms.append(torch.nan_to_num(ce, nan=0.0, posinf=0.0, neginf=0.0))
                                        teacher_max_values.append(teacher_prob.max(dim=-1).values.detach())
                                        teacher_entropy_values.append(teacher_entropy.detach())
                                        student_entropy_values.append(student_entropy.detach())
                                        teacher_label_prob_values.append(teacher_label_prob.detach())
                                        teacher_top_match_values.append(teacher_top_match.to(dtype=torch.float32).detach())
                                        student_label_prob_values.append(student_label_prob.detach())
                                        continue
                                    if objective == "trace_all_rkl":
                                        if teacher_source == "clean_causal":
                                            teacher_source_rows = teacher_logits_all[teacher_index_map[idx][label_in_range]]
                                        else:
                                            teacher_source_rows = teacher_logits_rmpad[idx][label_in_range]
                                        if teacher_source_rows.shape[-1] == lm_head.weight.shape[1]:
                                            teacher_logits = torch.nn.functional.linear(
                                                teacher_source_rows.to(dtype=lm_head.weight.dtype),
                                                weight,
                                                bias,
                                            ).float()
                                        else:
                                            teacher_logits = teacher_source_rows.index_select(-1, vocab_ids).float()
                                        teacher_logits = teacher_logits.div(jsd_temperature).clamp(
                                            min=-logit_clip, max=logit_clip
                                        )
                                        student_logp = torch.nn.functional.log_softmax(student_logits, dim=-1)
                                        teacher_logp = torch.nn.functional.log_softmax(teacher_logits.detach(), dim=-1)
                                        student_p = student_logp.exp()
                                        ce = (student_p * (student_logp - teacher_logp)).sum(dim=-1)
                                    else:
                                        ce = torch.nn.functional.cross_entropy(
                                            student_logits,
                                            target_idx,
                                            reduction="none",
                                        )
                                    ce = torch.nan_to_num(ce, nan=0.0, posinf=0.0, neginf=0.0)
                                    if _response_level_ce_enabled:
                                        _response_weight = (
                                            response_level_ce_weight_rmpad[idx][
                                                label_in_range
                                            ].float()
                                        )
                                        if not bool(
                                            _response_weight.gt(0).all()
                                        ):
                                            raise RuntimeError(
                                                "response_level_ce found "
                                                "non-positive active-token weight"
                                            )
                                        ce_sum = (
                                            ce * _response_weight
                                        ).sum()
                                        trace_all_ce_weight_sum += float(
                                            _response_weight.sum().item()
                                        )
                                        response_level_ce_weight_values.append(
                                            _response_weight.detach()
                                        )
                                    elif _response_tpf_ce_enabled:
                                        _response_scale = (
                                            response_tpf_ce_scale_rmpad[idx][
                                                label_in_range
                                            ].float()
                                        )
                                        ce_sum = (ce * _response_scale).sum()
                                        trace_all_ce_weight_sum += float(
                                            ce.numel()
                                        )
                                        response_tpf_ce_scale_values.append(
                                            _response_scale.detach()
                                        )
                                    elif _block_tpf_ce_enabled:
                                        _tpf_w = block_tpf_weight_rmpad[idx][label_in_range].float()
                                        ce_sum = (ce * _tpf_w).sum()
                                        trace_all_ce_weight_sum += float(_tpf_w.sum().item())
                                        trace_all_ce_weight_values.append(_tpf_w.detach())
                                    else:
                                        ce_sum = ce.sum()
                                        trace_all_ce_weight_sum += float(ce.numel())
                                    trace_all_ce_loss_sum = ce_sum if trace_all_ce_loss_sum is None else trace_all_ce_loss_sum + ce_sum
                                    trace_all_ce_count += int(ce.numel())
                                continue
                            if fast_hidden_ce:
                                if lm_head is None:
                                    raise RuntimeError(f"{objective} fast path requires actor module lm_head")
                                student_source = logits_rmpad[idx]
                                if student_source.shape[-1] == lm_head.weight.shape[1]:
                                    student_logits = torch.nn.functional.linear(
                                        student_source.to(dtype=lm_head.weight.dtype),
                                        lm_head.weight,
                                        lm_head.bias,
                                    ).float().div(jsd_temperature)
                                else:
                                    student_logits = student_source.float().div(jsd_temperature)
                            else:
                                student_logits = logits_rmpad[idx].float().div(jsd_temperature)
                            student_logits = torch.nan_to_num(student_logits, nan=0.0, posinf=0.0, neginf=0.0)
                            student_logits = student_logits.clamp(min=-logit_clip, max=logit_clip)
                            if objective == "trace_all_ce":
                                labels_valid = labels_rmpad[idx].long()
                                label_in_range = labels_valid.ge(0) & labels_valid.lt(student_logits.shape[-1])
                                if clean_keep_mask is not None:
                                    label_in_range = label_in_range & clean_keep_mask
                                if (
                                    _adj_ent_enable
                                    and _adj_ent_num_pairs > 0
                                ):
                                    _adj_mask_chunk = (
                                        adj_ent_pair_mask_rmpad[idx].bool()
                                        & label_in_range
                                    )
                                    if bool(_adj_mask_chunk.any()):
                                        _adj_ids_chunk = (
                                            adj_ent_pair_ids_rmpad[idx][
                                                _adj_mask_chunk
                                            ].long()
                                        )
                                        if (
                                            int(_adj_ids_chunk.min().item()) < 0
                                            or int(_adj_ids_chunk.max().item())
                                            >= _adj_ent_num_pairs
                                        ):
                                            raise RuntimeError(
                                                "adjacent entropy pair ID is "
                                                "outside the packed batch range"
                                            )
                                        _adj_vocab_chunk = (
                                            adj_ent_vocab_size_rmpad[idx][
                                                _adj_mask_chunk
                                            ].long()
                                        )
                                        if (
                                            not bool(
                                                _adj_vocab_chunk.eq(
                                                    int(vocab_size)
                                                ).all()
                                            )
                                        ):
                                            raise RuntimeError(
                                                "adjacent entropy producer/actor "
                                                "vocabulary mismatch"
                                            )
                                        _adj_logits_chunk = student_logits[
                                            _adj_mask_chunk
                                        ]
                                        _adj_logp_chunk = (
                                            torch.nn.functional.log_softmax(
                                                _adj_logits_chunk,
                                                dim=-1,
                                            )
                                        )
                                        _adj_entropy_chunk = -(
                                            _adj_logp_chunk.exp()
                                            * _adj_logp_chunk
                                        ).sum(dim=-1) / math.log(
                                            max(int(vocab_size), 2)
                                        )
                                        _adj_prev_chunk = (
                                            adj_ent_prev_mean_rmpad[idx][
                                                _adj_mask_chunk
                                            ].float()
                                        )
                                        _adj_rollout_next_chunk = (
                                            adj_ent_rollout_next_mean_rmpad[idx][
                                                _adj_mask_chunk
                                            ].float()
                                        )
                                        if _adj_ent_response_level:
                                            _adj_response_weight_chunk = (
                                                response_level_aec_pair_weight_rmpad[
                                                    idx
                                                ][_adj_mask_chunk].float()
                                            )
                                            if not bool(
                                                _adj_response_weight_chunk.gt(
                                                    0
                                                ).all()
                                            ):
                                                raise RuntimeError(
                                                    "response-level AEC found "
                                                    "non-positive pair weight"
                                                )
                                        else:
                                            _adj_response_weight_chunk = (
                                                torch.ones_like(
                                                    _adj_entropy_chunk
                                                )
                                            )
                                        _adj_ent_entropy_sums = (
                                            _adj_ent_entropy_sums.index_add(
                                                0,
                                                _adj_ids_chunk,
                                                _adj_entropy_chunk,
                                            )
                                        )
                                        _adj_ent_token_counts = (
                                            _adj_ent_token_counts.index_add(
                                                0,
                                                _adj_ids_chunk,
                                                torch.ones_like(
                                                    _adj_entropy_chunk
                                                ),
                                            )
                                        )
                                        _adj_ent_prev_sums = (
                                            _adj_ent_prev_sums.index_add(
                                                0,
                                                _adj_ids_chunk,
                                                _adj_prev_chunk,
                                            )
                                        )
                                        _adj_ent_rollout_next_sums = (
                                            _adj_ent_rollout_next_sums.index_add(
                                                0,
                                                _adj_ids_chunk,
                                                _adj_rollout_next_chunk,
                                            )
                                        )
                                        _adj_ent_response_weight_sums = (
                                            _adj_ent_response_weight_sums.index_add(
                                                0,
                                                _adj_ids_chunk,
                                                _adj_response_weight_chunk,
                                            )
                                        )
                                        if adj_ent_pair_local_step_rmpad is not None:
                                            _adj_local_step_chunk = (
                                                adj_ent_pair_local_step_rmpad[idx][
                                                    _adj_mask_chunk
                                                ].float()
                                            )
                                            _adj_ent_local_step_sums = (
                                                _adj_ent_local_step_sums.index_add(
                                                    0,
                                                    _adj_ids_chunk,
                                                    _adj_local_step_chunk,
                                                )
                                            )
                                if label_in_range.any():
                                    if _entropy_rkl_teacher_enable:
                                        _probe_student_all = student_logits[
                                            label_in_range
                                        ]
                                        if (
                                            rollout_logits_source_hidden_rmpad
                                            is None
                                        ):
                                            _probe_source_valid = torch.zeros(
                                                _probe_student_all.shape[0],
                                                dtype=torch.bool,
                                                device=_probe_student_all.device,
                                            )
                                            _probe_teacher_source = (
                                                _probe_student_all.new_empty(
                                                    (
                                                        0,
                                                        int(
                                                            lm_head.weight.shape[
                                                                1
                                                            ]
                                                        ),
                                                    )
                                                )
                                            )
                                        else:
                                            _probe_teacher_source_all = (
                                                rollout_logits_source_hidden_rmpad.squeeze(
                                                    0
                                                )[idx][label_in_range]
                                            )
                                            _probe_source_valid = (
                                                _probe_teacher_source_all.detach()
                                                .float()
                                                .abs()
                                                .sum(dim=-1)
                                                .gt(0)
                                            )
                                            _probe_teacher_source = (
                                                _probe_teacher_source_all[
                                                    _probe_source_valid
                                                ]
                                            )
                                        if (
                                            _entropy_gated_rkl_enable
                                            and not bool(
                                                _probe_source_valid.all()
                                            )
                                        ):
                                            _egrkl_n_missing = int(
                                                (~_probe_source_valid).sum().item()
                                            )
                                            # LLaDA2: a few accepted tokens carry no
                                            # rollout hidden (tokens rewritten/refined after acceptance by the
                                            # SDARPrefixKV edit/refine passes are not captured). With
                                            # ENTROPY_GATED_RKL_ALLOW_MISSING_TEACHER=1 those tokens keep their
                                            # CE and get a zero RKL term (teacher := detached student) instead
                                            # of aborting the step; the count is printed for the log.
                                            if (
                                                os.environ.get(
                                                    "ENTROPY_GATED_RKL_ALLOW_MISSING_TEACHER",
                                                    "0",
                                                )
                                                != "1"
                                            ):
                                                raise RuntimeError(
                                                    "entropy_gated_rkl requires "
                                                    "teacher-source hidden for every "
                                                    "eligible token; missing="
                                                    f"{_egrkl_n_missing}"
                                                )
                                            print(
                                                "[EGRKL_TEACHER_MISSING] "
                                                f"{_egrkl_n_missing}/"
                                                f"{int(_probe_source_valid.numel())} "
                                                "eligible tokens have no rollout "
                                                "hidden -> CE only (RKL=0) for them",
                                                flush=True,
                                            )
                                        with torch.no_grad():
                                            if (
                                                _probe_teacher_source.shape[-1]
                                                == lm_head.weight.shape[1]
                                            ):
                                                _probe_teacher_logits = (
                                                    torch.nn.functional.linear(
                                                        _probe_teacher_source.to(
                                                            dtype=lm_head.weight.dtype
                                                        ),
                                                        lm_head.weight,
                                                        lm_head.bias,
                                                    )
                                                    .float()
                                                    .div(jsd_temperature)
                                                )
                                            elif (
                                                _probe_teacher_source.shape[-1]
                                                == vocab_size
                                            ):
                                                _probe_teacher_logits = (
                                                    _probe_teacher_source.float().div(
                                                        jsd_temperature
                                                    )
                                                )
                                            else:
                                                raise RuntimeError(
                                                    "entropy-gated RKL teacher "
                                                    "source has incompatible width: "
                                                    f"{_probe_teacher_source.shape[-1]}"
                                                )
                                        if _entropy_rkl_probe_enable:
                                            _probe_chunk_stats = (
                                                compute_entropy_gated_rkl_probe_stats(
                                                    _probe_student_all[
                                                        _probe_source_valid
                                                    ],
                                                    _probe_teacher_logits,
                                                    eligible_count=int(
                                                        label_in_range.sum().item()
                                                    ),
                                                    entropy_thresholds=(
                                                        _entropy_rkl_probe_thresholds
                                                    ),
                                                    logit_clip=logit_clip,
                                                )
                                            )
                                            _entropy_rkl_probe_stats = (
                                                merge_entropy_gated_rkl_probe_stats_(
                                                    _entropy_rkl_probe_stats,
                                                    _probe_chunk_stats,
                                                )
                                            )
                                    if _entropy_gated_rkl_enable:
                                        if (
                                            _probe_teacher_logits.shape[0]
                                            != _probe_student_all.shape[0]
                                        ):
                                            # rows without rollout hidden: teacher := detached student
                                            # (same scale as the real teacher logits) -> RKL term 0
                                            _hybrid_teacher_logits = (
                                                _probe_student_all.detach()
                                                .float()
                                                .div(jsd_temperature)
                                            )
                                            _hybrid_teacher_logits[
                                                _probe_source_valid
                                            ] = _probe_teacher_logits
                                        else:
                                            _hybrid_teacher_logits = (
                                                _probe_teacher_logits
                                            )
                                        _hybrid_terms = (
                                            compute_entropy_gated_ce_rkl_terms(
                                                _probe_student_all,
                                                _hybrid_teacher_logits,
                                                labels_valid[label_in_range],
                                                entropy_threshold=(
                                                    _entropy_gated_rkl_threshold
                                                ),
                                                rkl_cap=(
                                                    _entropy_gated_rkl_cap
                                                ),
                                                logit_clip=logit_clip,
                                                cad_mode=(
                                                    _entropy_gated_rkl_cad_mode
                                                ),
                                                fkl_coef=(
                                                    _entropy_gated_rkl_fkl_coef
                                                ),
                                                cj_power=(
                                                    _entropy_gated_rkl_cj_power
                                                ),
                                            )
                                        )
                                        ce = _hybrid_terms["loss"]
                                    else:
                                        ce = torch.nn.functional.cross_entropy(
                                            student_logits[label_in_range],
                                            labels_valid[label_in_range],
                                            reduction="none",
                                        )
                                    if (
                                        signed_trace_ce_enabled
                                        and not bool(torch.isfinite(ce).all())
                                    ):
                                        raise FloatingPointError(
                                            "signed_trace_ce produced non-finite CE"
                                        )
                                    ce = torch.nan_to_num(ce, nan=0.0, posinf=0.0, neginf=0.0)
                                    if signed_trace_ce_enabled:
                                        signed_roles = (
                                            signed_trace_ce_role_rmpad[idx][
                                                label_in_range
                                            ]
                                        )
                                        positive_mask = signed_roles.gt(0)
                                        negative_mask = signed_roles.lt(0)
                                        if bool(positive_mask.any()):
                                            positive_sum = ce[
                                                positive_mask
                                            ].sum()
                                            signed_positive_ce_sum = (
                                                positive_sum
                                                if signed_positive_ce_sum is None
                                                else signed_positive_ce_sum
                                                + positive_sum
                                            )
                                            signed_positive_ce_count += int(
                                                positive_mask.sum().item()
                                            )
                                        if bool(negative_mask.any()):
                                            negative_sum = ce[
                                                negative_mask
                                            ].sum()
                                            signed_negative_ce_sum = (
                                                negative_sum
                                                if signed_negative_ce_sum is None
                                                else signed_negative_ce_sum
                                                + negative_sum
                                            )
                                            signed_negative_ce_count += int(
                                                negative_mask.sum().item()
                                            )
                                        trace_all_ce_count += int(ce.numel())
                                        trace_all_ce_weight_sum += float(
                                            ce.numel()
                                        )
                                        continue
                                    if causal_wrong_ce_actor_enabled and causal_wrong_hidden_sp is not None:
                                        # causal_wrong_ce branch:
                                        #   positive (+1): staircase source (logits_rmpad) -> positive CE
                                        #   wrong (-1):    causal source (causal_wrong_hidden_sp) -> negative CE
                                        #   drop (0):      excluded from both (already excluded by jsd_mask)
                                        causal_roles_chunk = causal_wrong_ce_role_rmpad[idx][label_in_range]
                                        positive_cwce_mask = causal_roles_chunk.gt(0)
                                        wrong_cwce_mask = causal_roles_chunk.lt(0)
                                        # Positive branch: staircase student logits already computed
                                        if bool(positive_cwce_mask.any()):
                                            pos_ce = ce[positive_cwce_mask]
                                            if not bool(torch.isfinite(pos_ce).all()):
                                                raise FloatingPointError(
                                                    "causal_wrong_ce positive branch produced non-finite CE"
                                                )
                                            causal_wrong_positive_ce_sum = (
                                                pos_ce.sum()
                                                if causal_wrong_positive_ce_sum is None
                                                else causal_wrong_positive_ce_sum + pos_ce.sum()
                                            )
                                            causal_wrong_positive_ce_count += int(positive_cwce_mask.sum().item())
                                        # Wrong branch: causal source hidden -> LM head -> CE
                                        if bool(wrong_cwce_mask.any()):
                                            wrong_idx_full = idx[label_in_range][wrong_cwce_mask]
                                            wrong_source = causal_wrong_hidden_sp[wrong_idx_full]
                                            if wrong_source.shape[-1] == lm_head.weight.shape[1]:
                                                wrong_logits = torch.nn.functional.linear(
                                                    wrong_source.to(dtype=lm_head.weight.dtype),
                                                    lm_head.weight,
                                                    lm_head.bias,
                                                ).float().div(jsd_temperature)
                                            else:
                                                wrong_logits = wrong_source.float().div(jsd_temperature)
                                            wrong_logits = torch.nan_to_num(
                                                wrong_logits, nan=0.0, posinf=0.0, neginf=0.0
                                            ).clamp(min=-logit_clip, max=logit_clip)
                                            wrong_targets = labels_rmpad[wrong_idx_full].long()
                                            wrong_ce = torch.nn.functional.cross_entropy(
                                                wrong_logits,
                                                wrong_targets,
                                                reduction="none",
                                            )
                                            if not bool(torch.isfinite(wrong_ce).all()):
                                                raise FloatingPointError(
                                                    "causal_wrong_ce wrong branch produced non-finite CE"
                                                )
                                            causal_wrong_neg_ce_sum = (
                                                wrong_ce.sum()
                                                if causal_wrong_neg_ce_sum is None
                                                else causal_wrong_neg_ce_sum + wrong_ce.sum()
                                            )
                                            causal_wrong_neg_ce_count += int(wrong_cwce_mask.sum().item())
                                        trace_all_ce_count += int(ce.numel())
                                        trace_all_ce_weight_sum += float(ce.numel())
                                        continue
                                    if trace_ce_entropy_enabled:
                                        _accumulate_trace_ce_entropy(
                                            verl_F.entropy_from_logits(
                                                student_logits[
                                                    label_in_range
                                                ]
                                            )
                                        )
                                    if _response_level_ce_enabled:
                                        _response_weight = (
                                            response_level_ce_weight_rmpad[idx][
                                                label_in_range
                                            ].float()
                                        )
                                        if not bool(
                                            _response_weight.gt(0).all()
                                        ):
                                            raise RuntimeError(
                                                "response_level_ce found "
                                                "non-positive active-token weight"
                                            )
                                        ce_sum = (
                                            ce * _response_weight
                                        ).sum()
                                        trace_all_ce_weight_sum += float(
                                            _response_weight.sum().item()
                                        )
                                        response_level_ce_weight_values.append(
                                            _response_weight.detach()
                                        )
                                    elif _response_tpf_ce_enabled:
                                        _response_scale = (
                                            response_tpf_ce_scale_rmpad[idx][
                                                label_in_range
                                            ].float()
                                        )
                                        ce_sum = (ce * _response_scale).sum()
                                        trace_all_ce_weight_sum += float(
                                            ce.numel()
                                        )
                                        response_tpf_ce_scale_values.append(
                                            _response_scale.detach()
                                        )
                                    elif _block_tpf_ce_enabled:
                                        _tpf_w = block_tpf_weight_rmpad[idx][label_in_range].float()
                                        ce_sum = (ce * _tpf_w).sum()
                                        trace_all_ce_weight_sum += float(_tpf_w.sum().item())
                                        trace_all_ce_weight_values.append(_tpf_w.detach())
                                    else:
                                        ce_sum = ce.sum()
                                        trace_all_ce_weight_sum += float(ce.numel())
                                    if _entropy_gated_rkl_enable:
                                        accumulate_entropy_gated_rkl_report_(
                                            _entropy_gated_rkl_report,
                                            _hybrid_terms,
                                            cad_mode=(
                                                _entropy_gated_rkl_cad_mode
                                            ),
                                            token_weights=(
                                                _response_weight
                                                if _response_level_ce_enabled
                                                else None
                                            ),
                                        )
                                    trace_all_ce_loss_sum = ce_sum if trace_all_ce_loss_sum is None else trace_all_ce_loss_sum + ce_sum
                                    trace_all_ce_count += int(ce.numel())
                                continue
                            if objective == "trace_future_ce":
                                student_logp = torch.nan_to_num(
                                    torch.nn.functional.log_softmax(student_logits, dim=-1),
                                    nan=-100.0,
                                    posinf=0.0,
                                    neginf=-100.0,
                                )
                                if trace_clean_consistency_enabled:
                                    labels_valid = labels_rmpad[idx].long()
                                    label_in_range = labels_valid.ge(0) & labels_valid.lt(student_logp.shape[-1])
                                    if clean_pos_mask is not None:
                                        pos_mask = clean_pos_mask & label_in_range
                                        if pos_mask.any():
                                            pos_logp = student_logp[pos_mask].gather(
                                                dim=-1,
                                                index=labels_valid[pos_mask].unsqueeze(-1),
                                            ).squeeze(-1)
                                            pos_loss = torch.nan_to_num(
                                                -pos_logp,
                                                nan=0.0,
                                                posinf=0.0,
                                                neginf=0.0,
                                            )
                                            clean_pos_terms.append(pos_loss)
                                            if clean_prob_for_idx is not None:
                                                clean_pos_prob_values.append(clean_prob_for_idx[pos_mask].detach())
                                    if clean_neg_mask is not None:
                                        neg_mask = clean_neg_mask & label_in_range
                                        if neg_mask.any():
                                            neg_logp = student_logp[neg_mask].gather(
                                                dim=-1,
                                                index=labels_valid[neg_mask].unsqueeze(-1),
                                            ).squeeze(-1)
                                            neg_prob = neg_logp.exp().clamp(
                                                min=prob_eps,
                                                max=clean_consistency_neg_prob_clip,
                                            )
                                            neg_loss = torch.nan_to_num(
                                                -torch.log1p(-neg_prob),
                                                nan=0.0,
                                                posinf=0.0,
                                                neginf=0.0,
                                            )
                                            if clean_consistency_neg_loss_clip > 0:
                                                neg_loss = neg_loss.clamp(max=clean_consistency_neg_loss_clip)
                                            clean_neg_terms.append(neg_loss)
                                            if clean_prob_for_idx is not None:
                                                clean_neg_prob_values.append(clean_prob_for_idx[neg_mask].detach())
                                if trace_future_hard_label_ce:
                                    labels_valid = labels_rmpad[idx].long()
                                    keep = labels_valid.ge(0) & labels_valid.lt(student_logits.shape[-1])
                                    if clean_keep_mask is not None:
                                        keep = keep & clean_keep_mask
                                    if keep.any():
                                        ce = torch.nn.functional.cross_entropy(
                                            student_logits[keep],
                                            labels_valid[keep],
                                            reduction="none",
                                        )
                                        ce = torch.nan_to_num(ce, nan=0.0, posinf=0.0, neginf=0.0)
                                        if trace_ce_entropy_enabled:
                                            _accumulate_trace_ce_entropy(
                                                verl_F.entropy_from_logits(
                                                    student_logits[keep]
                                                )
                                            )
                                        jsd_terms.append(ce)
                                        teacher_max_values.append(torch.ones_like(ce.detach()))
                                        teacher_entropy_values.append(torch.zeros_like(ce.detach()))
                                        student_entropy_values.append(torch.zeros_like(ce.detach()))
                                    continue
                                if rollout_logits_source_hidden_rmpad is None:
                                    raise RuntimeError(
                                        "generation_consistency_jsd.objective=trace_future_ce requires future rollout_logits_source_hidden"
                                    )
                                teacher_source_rows = rollout_logits_source_hidden_rmpad.squeeze(0).detach()[idx]
                                if fast_hidden_ce:
                                    if teacher_source_rows.shape[-1] == lm_head.weight.shape[1]:
                                        teacher_logits = torch.nn.functional.linear(
                                            teacher_source_rows.to(device=student_logits.device, dtype=lm_head.weight.dtype),
                                            lm_head.weight,
                                            lm_head.bias,
                                        ).float().div(jsd_temperature)
                                    else:
                                        teacher_logits = teacher_source_rows.to(
                                            device=student_logits.device
                                        ).float().div(jsd_temperature)
                                else:
                                    teacher_logits = teacher_source_rows.to(device=student_logits.device).float().div(jsd_temperature)
                                teacher_logits = torch.nan_to_num(
                                    teacher_logits.detach(), nan=0.0, posinf=0.0, neginf=0.0
                                ).clamp(min=-logit_clip, max=logit_clip)
                                teacher_logp = torch.nan_to_num(
                                    torch.nn.functional.log_softmax(teacher_logits, dim=-1),
                                    nan=-100.0,
                                    posinf=0.0,
                                    neginf=-100.0,
                                )
                                teacher_prob = teacher_logp.exp()
                                teacher_max = teacher_prob.max(dim=-1).values
                                keep = teacher_max.gt(threshold) if threshold > 0 else torch.ones_like(
                                    teacher_max, dtype=torch.bool
                                )
                                if clean_keep_mask is not None:
                                    keep = keep & clean_keep_mask
                                if keep.any():
                                    kept_teacher_prob = teacher_prob[keep]
                                    kept_student_logp = student_logp[keep]
                                    labels_valid = labels_rmpad[idx].long()
                                    labels_keep = labels_valid[keep]
                                    ce = -(kept_teacher_prob * kept_student_logp).sum(dim=-1)
                                    ce = torch.nan_to_num(ce, nan=0.0, posinf=0.0, neginf=0.0)
                                    jsd_terms.append(ce)
                                    teacher_entropy_values.append(
                                        -(kept_teacher_prob * teacher_logp[keep]).sum(dim=-1).detach()
                                    )
                                    student_prob = kept_student_logp.exp()
                                    student_entropy = -(student_prob * kept_student_logp).sum(dim=-1)
                                    _accumulate_trace_ce_entropy(student_entropy)
                                    student_entropy_values.append(student_entropy.detach())
                                    teacher_max_values.append(teacher_max[keep].detach())
                                    teacher_label_prob = kept_teacher_prob.gather(
                                        dim=-1, index=labels_keep.unsqueeze(-1)
                                    ).squeeze(-1)
                                    student_label_prob = student_prob.gather(
                                        dim=-1, index=labels_keep.unsqueeze(-1)
                                    ).squeeze(-1)
                                    teacher_top_match = kept_teacher_prob.argmax(dim=-1).eq(labels_keep)
                                    _append_trace_future_teacher_token_debug(
                                        rmpad_idx=idx[keep],
                                        labels_keep=labels_keep,
                                        teacher_prob_keep=kept_teacher_prob,
                                        student_label_prob_keep=student_label_prob,
                                        teacher_source_keep=teacher_source_rows[keep],
                                    )
                                    teacher_label_prob_values.append(teacher_label_prob.detach())
                                    teacher_top_match_values.append(teacher_top_match.to(dtype=torch.float32).detach())
                                    student_label_prob_values.append(student_label_prob.detach())
                                continue
                            if objective == "gated_ce":
                                if rollout_log_probs_rmpad is None:
                                    raise RuntimeError(
                                        "generation_consistency_jsd.objective=gated_ce requires rollout_log_probs"
                                    )
                                labels_valid = labels_rmpad[idx].long()
                                rollout_teacher_prob = torch.nan_to_num(
                                    rollout_log_probs_rmpad[idx].float().exp(), nan=0.0, posinf=1.0, neginf=0.0
                                )
                                label_in_range = labels_valid.ge(0) & labels_valid.lt(student_logits.shape[-1])
                                teacher_threshold = float(
                                    generation_jsd_config.get("teacher_prob_threshold", threshold)
                                )
                                keep = label_in_range & rollout_teacher_prob.gt(teacher_threshold)
                                if keep.any():
                                    ce = torch.nn.functional.cross_entropy(
                                        student_logits[keep],
                                        labels_valid[keep],
                                        reduction="none",
                                    )
                                    ce = torch.nan_to_num(ce, nan=0.0, posinf=0.0, neginf=0.0)
                                    jsd_terms.append(ce)
                                    teacher_max_values.append(rollout_teacher_prob[keep].detach())
                                    teacher_entropy_values.append(torch.zeros_like(ce.detach()))
                                    student_entropy_values.append(torch.zeros_like(ce.detach()))
                                continue

                            if teacher_source == "clean_causal":
                                if fast_hidden_ce:
                                    teacher_hidden_or_logits = teacher_logits_all[teacher_index_map[idx]]
                                    if teacher_hidden_or_logits.shape[-1] == lm_head.weight.shape[1]:
                                        teacher_logits = torch.nn.functional.linear(
                                            teacher_hidden_or_logits.to(dtype=lm_head.weight.dtype),
                                            lm_head.weight,
                                            lm_head.bias,
                                        ).float().div(jsd_temperature)
                                    else:
                                        teacher_logits = teacher_hidden_or_logits.float().div(jsd_temperature)
                                else:
                                    teacher_logits = teacher_logits_all[teacher_index_map[idx]].float().div(jsd_temperature)
                            else:
                                teacher_logits = teacher_logits_rmpad[idx].float().div(jsd_temperature)
                            teacher_logits = torch.nan_to_num(teacher_logits.detach(), nan=0.0, posinf=0.0, neginf=0.0)
                            teacher_logits = teacher_logits.clamp(min=-logit_clip, max=logit_clip)
                            student_logp = torch.nan_to_num(
                                torch.nn.functional.log_softmax(student_logits, dim=-1),
                                nan=-100.0,
                                posinf=0.0,
                                neginf=-100.0,
                            )
                            teacher_logp = torch.nan_to_num(
                                torch.nn.functional.log_softmax(teacher_logits, dim=-1),
                                nan=-100.0,
                                posinf=0.0,
                                neginf=-100.0,
                            )
                            student_prob = student_logp.exp()
                            teacher_prob = teacher_logp.exp()
                            teacher_max = teacher_prob.max(dim=-1).values
                            if objective == "clean_agree_ce":
                                if rollout_log_probs_rmpad is None:
                                    raise RuntimeError(
                                        "generation_consistency_jsd.objective=clean_agree_ce requires rollout_log_probs"
                                    )
                                labels_valid = labels_rmpad[idx].long()
                                rollout_teacher_prob = torch.nan_to_num(
                                    rollout_log_probs_rmpad[idx].float().exp(), nan=0.0, posinf=1.0, neginf=0.0
                                )
                                label_in_range = labels_valid.ge(0) & labels_valid.lt(teacher_prob.shape[-1])
                                clean_final_prob = torch.zeros_like(rollout_teacher_prob)
                                if label_in_range.any():
                                    clean_final_prob[label_in_range] = torch.gather(
                                        teacher_prob[label_in_range],
                                        dim=-1,
                                        index=labels_valid[label_in_range].unsqueeze(-1),
                                    ).squeeze(-1)
                                teacher_top_prob, teacher_top_ids = teacher_prob.max(dim=-1)
                                rollout_threshold = float(
                                    generation_jsd_config.get("teacher_prob_threshold", threshold)
                                )
                                clean_threshold = float(
                                    generation_jsd_config.get("student_final_prob_threshold", threshold)
                                )
                                keep = (
                                    label_in_range
                                    & teacher_top_ids.eq(labels_valid)
                                    & rollout_teacher_prob.gt(rollout_threshold)
                                    & clean_final_prob.gt(clean_threshold)
                                )
                                if keep.any():
                                    ce = torch.nn.functional.cross_entropy(
                                        student_logits[keep],
                                        labels_valid[keep],
                                        reduction="none",
                                    )
                                    ce = torch.nan_to_num(ce, nan=0.0, posinf=0.0, neginf=0.0)
                                    jsd_terms.append(ce)
                                    teacher_max_values.append(clean_final_prob[keep].detach())
                                    teacher_entropy_values.append(torch.zeros_like(ce.detach()))
                                    student_entropy_values.append(torch.zeros_like(ce.detach()))
                                continue
                            if objective == "clean_causal_ce":
                                teacher_top_prob, teacher_top_ids = teacher_prob.max(dim=-1)
                                keep = teacher_top_prob.gt(threshold) if threshold > 0 else torch.ones_like(
                                    teacher_top_prob, dtype=torch.bool
                                )
                                if keep.any():
                                    ce = torch.nn.functional.cross_entropy(
                                        student_logits[keep],
                                        teacher_top_ids[keep],
                                        reduction="none",
                                    )
                                    ce = torch.nan_to_num(ce, nan=0.0, posinf=0.0, neginf=0.0)
                                    jsd_terms.append(ce)
                                    teacher_max_values.append(teacher_top_prob[keep].detach())
                                    teacher_entropy_values.append(torch.zeros_like(ce.detach()))
                                    student_entropy_values.append(torch.zeros_like(ce.detach()))
                                continue
                            if objective == "top_token_jsd":
                                if rollout_log_probs_rmpad is None:
                                    raise RuntimeError(
                                        "generation_consistency_jsd.objective=top_token_jsd requires rollout_log_probs"
                                    )
                                labels_valid = labels_rmpad[idx].long()
                                rollout_teacher_prob = torch.nan_to_num(
                                    rollout_log_probs_rmpad[idx].float().exp(), nan=0.0, posinf=1.0, neginf=0.0
                                )
                                label_in_range = labels_valid.ge(0) & labels_valid.lt(teacher_prob.shape[-1])
                                student_final_prob = torch.zeros_like(rollout_teacher_prob)
                                if label_in_range.any():
                                    student_final_prob[label_in_range] = torch.gather(
                                        teacher_prob[label_in_range],
                                        dim=-1,
                                        index=labels_valid[label_in_range].unsqueeze(-1),
                                    ).squeeze(-1)
                                teacher_threshold = float(generation_jsd_config.get("teacher_prob_threshold", threshold))
                                student_threshold = float(
                                    generation_jsd_config.get("student_final_prob_threshold", threshold)
                                )
                                keep = (
                                    label_in_range
                                    & rollout_teacher_prob.gt(teacher_threshold)
                                    & student_final_prob.gt(student_threshold)
                                )
                            else:
                                keep = teacher_max.gt(threshold) if threshold > 0 else torch.ones_like(teacher_max, dtype=torch.bool)

                            if keep.any() and objective == "top_token_jsd":
                                topk = int(generation_jsd_config.get("topk", 64))
                                jsd_rows = []
                                kept_teacher_max = []
                                kept_teacher_entropy = []
                                kept_student_entropy = []
                                for row_idx in torch.nonzero(keep, as_tuple=False).flatten().tolist():
                                    label_idx = labels_valid[row_idx].view(1)
                                    s_top = torch.topk(student_logits[row_idx], k=min(topk, student_logits.shape[-1])).indices
                                    t_top = torch.topk(teacher_logits[row_idx], k=min(topk, teacher_logits.shape[-1])).indices
                                    vocab_idx = torch.unique(torch.cat([s_top, t_top, label_idx], dim=0))
                                    s_sub = student_logits[row_idx, vocab_idx]
                                    t_sub = teacher_logits[row_idx, vocab_idx]
                                    s_logp = torch.nn.functional.log_softmax(s_sub, dim=-1)
                                    t_logp = torch.nn.functional.log_softmax(t_sub, dim=-1)
                                    s_prob = s_logp.exp()
                                    t_prob = t_logp.exp()
                                    m_prob = 0.5 * (s_prob + t_prob)
                                    m_logp = m_prob.clamp_min(prob_eps).log()
                                    m_entropy = -(m_prob * m_logp).sum()
                                    s_entropy = -(s_prob * s_logp).sum()
                                    t_entropy = -(t_prob * t_logp).sum()
                                    jsd_rows.append(torch.nan_to_num(m_entropy - 0.5 * s_entropy - 0.5 * t_entropy, nan=0.0, posinf=0.0, neginf=0.0).view(1))
                                    kept_teacher_max.append(student_final_prob[row_idx].detach().view(1))
                                    kept_teacher_entropy.append(t_entropy.detach().view(1))
                                    kept_student_entropy.append(s_entropy.detach().view(1))
                                if jsd_rows:
                                    jsd = torch.cat(jsd_rows, dim=0)
                                    jsd_terms.append(jsd)
                                    teacher_max_values.append(torch.cat(kept_teacher_max, dim=0))
                                    teacher_entropy_values.append(torch.cat(kept_teacher_entropy, dim=0))
                                    student_entropy_values.append(torch.cat(kept_student_entropy, dim=0))
                            elif keep.any():
                                student_prob_keep = student_prob[keep]
                                teacher_prob_keep = teacher_prob[keep]
                                mixed_prob = 0.5 * (student_prob_keep + teacher_prob_keep)
                                mixed_logp = mixed_prob.clamp_min(prob_eps).log()
                                mixed_entropy = -(mixed_prob * mixed_logp).sum(dim=-1)
                                student_entropy = -(student_prob_keep * student_logp[keep]).sum(dim=-1)
                                teacher_entropy = -(teacher_prob_keep * teacher_logp[keep]).sum(dim=-1)
                                jsd = mixed_entropy - 0.5 * student_entropy - 0.5 * teacher_entropy
                                jsd = torch.nan_to_num(jsd, nan=0.0, posinf=0.0, neginf=0.0)
                                jsd_terms.append(jsd)
                                teacher_max_values.append(teacher_max[keep].detach())
                                teacher_entropy_values.append(teacher_entropy.detach())
                                student_entropy_values.append(student_entropy.detach())
                        if objective == "trace_all_ce" and signed_trace_ce_enabled:
                            graph_zero = logits_rmpad.float().sum() * 0.0
                            positive_sum = (
                                signed_positive_ce_sum
                                if signed_positive_ce_sum is not None
                                else graph_zero
                            )
                            negative_sum = (
                                signed_negative_ce_sum
                                if signed_negative_ce_sum is not None
                                else graph_zero
                            )
                            positive_mean = (
                                positive_sum
                                / float(signed_positive_ce_count)
                                if signed_positive_ce_count > 0
                                else graph_zero
                            )
                            negative_mean = (
                                negative_sum
                                / float(signed_negative_ce_count)
                                if signed_negative_ce_count > 0
                                else graph_zero
                            )
                            generation_jsd_loss = (
                                positive_mean - negative_mean
                            )
                            if not bool(
                                torch.isfinite(generation_jsd_loss).all()
                            ):
                                raise FloatingPointError(
                                    "signed_trace_ce produced a non-finite objective"
                                )
                            generation_jsd_metrics = {
                                "generation_consistency_jsd/loss": float(
                                    generation_jsd_loss.detach().item()
                                ),
                                "generation_consistency_jsd/ce_loss": float(
                                    generation_jsd_loss.detach().item()
                                ),
                                "generation_consistency_jsd/total_objective": float(
                                    generation_jsd_loss.detach().item()
                                ),
                                "generation_consistency_jsd/active_tokens": float(
                                    signed_positive_ce_count
                                    + signed_negative_ce_count
                                ),
                                "generation_consistency_jsd/weighted_numerator": float(
                                    (
                                        positive_sum - negative_sum
                                    ).detach().item()
                                ),
                                "generation_consistency_jsd/weighted_denominator": float(
                                    signed_positive_ce_count
                                    + signed_negative_ce_count
                                ),
                                "signed_trace_ce/local_positive_ce_sum": float(
                                    positive_sum.detach().item()
                                ),
                                "signed_trace_ce/local_negative_ce_sum": float(
                                    negative_sum.detach().item()
                                ),
                                "signed_trace_ce/local_positive_tokens": float(
                                    signed_positive_ce_count
                                ),
                                "signed_trace_ce/local_negative_tokens": float(
                                    signed_negative_ce_count
                                ),
                                "signed_trace_ce/local_positive_ce_mean": float(
                                    positive_mean.detach().item()
                                ),
                                "signed_trace_ce/local_negative_ce_mean": float(
                                    negative_mean.detach().item()
                                ),
                                "signed_trace_ce/local_signed_loss": float(
                                    generation_jsd_loss.detach().item()
                                ),
                            }
                        elif objective == "trace_all_ce" and causal_wrong_ce_actor_enabled:
                            graph_zero = logits_rmpad.float().sum() * 0.0
                            cwce_pos_sum = (
                                causal_wrong_positive_ce_sum
                                if causal_wrong_positive_ce_sum is not None
                                else graph_zero
                            )
                            cwce_neg_sum = (
                                causal_wrong_neg_ce_sum
                                if causal_wrong_neg_ce_sum is not None
                                else graph_zero
                            )
                            cwce_pos_mean = (
                                cwce_pos_sum / float(causal_wrong_positive_ce_count)
                                if causal_wrong_positive_ce_count > 0
                                else graph_zero
                            )
                            cwce_neg_mean = (
                                cwce_neg_sum / float(causal_wrong_neg_ce_count)
                                if causal_wrong_neg_ce_count > 0
                                else graph_zero
                            )
                            generation_jsd_loss = cwce_pos_mean - cwce_neg_mean
                            if not bool(torch.isfinite(generation_jsd_loss).all()):
                                raise FloatingPointError(
                                    "causal_wrong_ce produced a non-finite objective"
                                )
                            generation_jsd_metrics = {
                                "generation_consistency_jsd/loss": float(
                                    generation_jsd_loss.detach().item()
                                ),
                                "generation_consistency_jsd/ce_loss": float(
                                    generation_jsd_loss.detach().item()
                                ),
                                "generation_consistency_jsd/total_objective": float(
                                    generation_jsd_loss.detach().item()
                                ),
                                "generation_consistency_jsd/active_tokens": float(
                                    causal_wrong_positive_ce_count
                                    + causal_wrong_neg_ce_count
                                ),
                                "causal_wrong_ce/local_positive_ce_sum": float(
                                    cwce_pos_sum.detach().item()
                                ),
                                "causal_wrong_ce/local_wrong_ce_sum": float(
                                    cwce_neg_sum.detach().item()
                                ),
                                "causal_wrong_ce/local_positive_tokens": float(
                                    causal_wrong_positive_ce_count
                                ),
                                "causal_wrong_ce/local_wrong_tokens": float(
                                    causal_wrong_neg_ce_count
                                ),
                                "causal_wrong_ce/local_positive_ce_mean": float(
                                    cwce_pos_mean.detach().item()
                                ),
                                "causal_wrong_ce/local_wrong_ce_mean": float(
                                    cwce_neg_mean.detach().item()
                                ),
                                "causal_wrong_ce/local_loss": float(
                                    generation_jsd_loss.detach().item()
                                ),
                                "causal_wrong_ce/has_positive": float(
                                    causal_wrong_positive_ce_count > 0
                                ),
                                "causal_wrong_ce/has_wrong": float(
                                    causal_wrong_neg_ce_count > 0
                                ),
                            }
                        elif objective == "trace_all_ce" and trace_all_ce_count > 0:
                            _tce_denominator = (
                                trace_all_ce_weight_sum
                                if trace_all_ce_weight_sum > 0
                                else float(trace_all_ce_count)
                            )
                            trace_all_ce_loss = trace_all_ce_loss_sum / _tce_denominator
                            generation_jsd_loss = trace_all_ce_loss
                            trace_ce_entropy_mean = None
                            trace_ce_entropy_objective = None
                            if trace_ce_entropy_count > 0:
                                trace_ce_entropy_mean = trace_ce_entropy_sum / float(trace_ce_entropy_count)
                                trace_ce_entropy_objective = trace_ce_entropy_mean
                                if trace_ce_entropy_target is not None:
                                    trace_ce_entropy_objective = torch.relu(
                                        trace_ce_entropy_mean - trace_ce_entropy_target
                                    )
                                generation_jsd_loss = (
                                    generation_jsd_loss
                                    + trace_ce_entropy_sign
                                    * trace_ce_entropy_coef
                                    * trace_ce_entropy_objective
                                )
                            trace_ce_entropy_contribution = (
                                trace_ce_entropy_sign
                                * trace_ce_entropy_coef
                                * trace_ce_entropy_objective
                                if trace_ce_entropy_objective is not None
                                else generation_jsd_loss.detach() * 0.0
                            )
                            generation_jsd_metrics = {
                                "generation_consistency_jsd/loss": generation_jsd_loss.detach().item(),
                                "generation_consistency_jsd/ce_loss": trace_all_ce_loss.detach().item(),
                                "generation_consistency_jsd/entropy_mean": (
                                    trace_ce_entropy_mean.detach().item()
                                    if trace_ce_entropy_mean is not None
                                    else 0.0
                                ),
                                "generation_consistency_jsd/entropy_contribution": (
                                    trace_ce_entropy_contribution.detach().item()
                                ),
                                "generation_consistency_jsd/total_objective": generation_jsd_loss.detach().item(),
                                "generation_consistency_jsd/entropy_coef": trace_ce_entropy_coef,
                                "generation_consistency_jsd/entropy_target": (
                                    trace_ce_entropy_target
                                    if trace_ce_entropy_target is not None
                                    else -1.0
                                ),
                                "generation_consistency_jsd/entropy_excess": (
                                    trace_ce_entropy_objective.detach().item()
                                    if trace_ce_entropy_objective is not None
                                    else 0.0
                                ),
                                "generation_consistency_jsd/entropy_mode_penalty": float(
                                    trace_ce_entropy_mode == "penalty"
                                ),
                                "generation_consistency_jsd/active_tokens": float(trace_all_ce_count),
                                "generation_consistency_jsd/weighted_numerator": float(
                                    trace_all_ce_loss_sum.detach().item()
                                ),
                                "generation_consistency_jsd/weighted_denominator": float(
                                    _tce_denominator
                                ),
                                "generation_consistency_jsd/active_frac": (
                                    trace_all_ce_count / max(int(valid_label_mask.sum().detach().item()), 1)
                                ),
                                "generation_consistency_jsd/teacher_max_prob_mean": 1.0,
                                "generation_consistency_jsd/teacher_entropy_mean": 0.0,
                                "generation_consistency_jsd/student_entropy_mean": 0.0,
                            }
                            if _block_tpf_ce_enabled and trace_all_ce_weight_values:
                                _active_w = torch.cat(trace_all_ce_weight_values)
                                generation_jsd_metrics.update(
                                    {
                                        "generation_consistency_jsd/block_tpf_weight_min": float(
                                            _active_w.min().item()
                                        ),
                                        "generation_consistency_jsd/block_tpf_weight_max": float(
                                            _active_w.max().item()
                                        ),
                                        "generation_consistency_jsd/block_tpf_weight_mean": float(
                                            _active_w.mean().item()
                                        ),
                                    }
                                )
                            if (
                                _response_tpf_ce_enabled
                                and response_tpf_ce_scale_values
                            ):
                                _active_scale = torch.cat(
                                    response_tpf_ce_scale_values
                                )
                                generation_jsd_metrics.update(
                                    {
                                        "generation_consistency_jsd/response_tpf_ce_scale_min": float(
                                            _active_scale.min().item()
                                        ),
                                        "generation_consistency_jsd/response_tpf_ce_scale_max": float(
                                            _active_scale.max().item()
                                        ),
                                        "generation_consistency_jsd/response_tpf_ce_scale_mean": float(
                                            _active_scale.mean().item()
                                        ),
                                    }
                                )
                            if (
                                _response_level_ce_enabled
                                and response_level_ce_weight_values
                            ):
                                _response_weights = torch.cat(
                                    response_level_ce_weight_values
                                )
                                generation_jsd_metrics.update(
                                    {
                                        "response_level_ce/local_weight_min": float(
                                            _response_weights.min().item()
                                        ),
                                        "response_level_ce/local_weight_max": float(
                                            _response_weights.max().item()
                                        ),
                                        "response_level_ce/local_weight_mean": float(
                                            _response_weights.mean().item()
                                        ),
                                        "response_level_ce/local_weight_sum": float(
                                            _response_weights.sum().item()
                                        ),
                                    }
                                )
                            if trace_clean_filter_enabled:
                                generation_jsd_metrics.update(
                                    {
                                        "generation_consistency_jsd/clean_filter_total": float(trace_clean_filter_total),
                                        "generation_consistency_jsd/clean_filter_kept": float(trace_clean_filter_kept),
                                        "generation_consistency_jsd/clean_filter_keep_frac": (
                                            float(trace_clean_filter_kept)
                                            / max(float(trace_clean_filter_total), 1.0)
                                        ),
                                    }
                                )
                        elif jsd_terms:
                            jsd_all = torch.cat(jsd_terms, dim=0)
                            base_generation_jsd_loss = jsd_all.mean()
                            generation_jsd_loss = base_generation_jsd_loss
                            clean_pos_loss = None
                            clean_neg_loss = None
                            if clean_pos_terms:
                                clean_pos_all = torch.cat(clean_pos_terms, dim=0)
                                clean_pos_loss = clean_pos_all.mean()
                                generation_jsd_loss = (
                                    generation_jsd_loss
                                    + clean_consistency_pos_coef * clean_pos_loss
                                )
                            if clean_neg_terms:
                                clean_neg_all = torch.cat(clean_neg_terms, dim=0)
                                clean_neg_loss = clean_neg_all.mean()
                                generation_jsd_loss = (
                                    generation_jsd_loss
                                    + clean_consistency_neg_coef * clean_neg_loss
                                )
                            trace_ce_entropy_mean = None
                            trace_ce_entropy_objective = None
                            if trace_ce_entropy_count > 0:
                                trace_ce_entropy_mean = trace_ce_entropy_sum / float(trace_ce_entropy_count)
                                trace_ce_entropy_objective = trace_ce_entropy_mean
                                if trace_ce_entropy_target is not None:
                                    trace_ce_entropy_objective = torch.relu(
                                        trace_ce_entropy_mean - trace_ce_entropy_target
                                    )
                                generation_jsd_loss = (
                                    generation_jsd_loss
                                    + trace_ce_entropy_sign
                                    * trace_ce_entropy_coef
                                    * trace_ce_entropy_objective
                                )
                            trace_ce_entropy_contribution = (
                                trace_ce_entropy_sign
                                * trace_ce_entropy_coef
                                * trace_ce_entropy_objective
                                if trace_ce_entropy_objective is not None
                                else generation_jsd_loss.detach() * 0.0
                            )
                            teacher_max_all = torch.cat(teacher_max_values, dim=0)
                            teacher_entropy_all = torch.cat(teacher_entropy_values, dim=0)
                            student_entropy_all = torch.cat(student_entropy_values, dim=0)
                            generation_jsd_metrics = {
                                "generation_consistency_jsd/loss": generation_jsd_loss.detach().item(),
                                "generation_consistency_jsd/base_loss": base_generation_jsd_loss.detach().item(),
                                "generation_consistency_jsd/ce_loss": base_generation_jsd_loss.detach().item(),
                                "generation_consistency_jsd/entropy_mean": (
                                    trace_ce_entropy_mean.detach().item()
                                    if trace_ce_entropy_mean is not None
                                    else 0.0
                                ),
                                "generation_consistency_jsd/entropy_contribution": (
                                    trace_ce_entropy_contribution.detach().item()
                                ),
                                "generation_consistency_jsd/total_objective": generation_jsd_loss.detach().item(),
                                "generation_consistency_jsd/entropy_coef": trace_ce_entropy_coef,
                                "generation_consistency_jsd/entropy_target": (
                                    trace_ce_entropy_target
                                    if trace_ce_entropy_target is not None
                                    else -1.0
                                ),
                                "generation_consistency_jsd/entropy_excess": (
                                    trace_ce_entropy_objective.detach().item()
                                    if trace_ce_entropy_objective is not None
                                    else 0.0
                                ),
                                "generation_consistency_jsd/entropy_mode_penalty": float(
                                    trace_ce_entropy_mode == "penalty"
                                ),
                                "generation_consistency_jsd/active_tokens": float(jsd_all.numel()),
                                "generation_consistency_jsd/active_frac": (
                                    jsd_all.numel() / max(int(valid_label_mask.sum().detach().item()), 1)
                                ),
                                "generation_consistency_jsd/teacher_max_prob_mean": teacher_max_all.mean().item(),
                                "generation_consistency_jsd/teacher_entropy_mean": teacher_entropy_all.mean().item(),
                                "generation_consistency_jsd/student_entropy_mean": student_entropy_all.mean().item(),
                            }
                            if teacher_label_prob_values:
                                teacher_label_prob_all = torch.cat(teacher_label_prob_values, dim=0)
                                teacher_top_match_all = torch.cat(teacher_top_match_values, dim=0)
                                student_label_prob_all = torch.cat(student_label_prob_values, dim=0)
                                generation_jsd_metrics.update(
                                    {
                                        "generation_consistency_jsd/teacher_label_prob_mean": teacher_label_prob_all.mean().item(),
                                        "generation_consistency_jsd/teacher_label_prob_p10": torch.quantile(
                                            teacher_label_prob_all.float(), 0.10
                                        ).item(),
                                        "generation_consistency_jsd/teacher_label_prob_p50": torch.quantile(
                                            teacher_label_prob_all.float(), 0.50
                                        ).item(),
                                        "generation_consistency_jsd/teacher_label_prob_p90": torch.quantile(
                                            teacher_label_prob_all.float(), 0.90
                                        ).item(),
                                        "generation_consistency_jsd/teacher_top_matches_label_frac": teacher_top_match_all.mean().item(),
                                        "generation_consistency_jsd/student_label_prob_mean": student_label_prob_all.mean().item(),
                                    }
                                )
                            if objective == "trace_future_ce":
                                generation_jsd_metrics.update(
                                    {
                                        "generation_consistency_jsd/future_logits_offset": float(
                                            generation_jsd_config.get("trace_future_logits_offset", 5)
                                        ),
                                        "generation_consistency_jsd/future_skip_tail_steps": float(
                                            generation_jsd_config.get("trace_future_skip_tail_steps", 0) or 0
                                        ),
                                        "generation_consistency_jsd/trace_future_hard_label_ce": (
                                            1.0 if trace_future_hard_label_ce else 0.0
                                        ),
                                    }
                                )
                            if trace_clean_consistency_enabled:
                                clean_consistency_metrics = {
                                    "generation_consistency_jsd/clean_consistency_enabled": 1.0,
                                    "generation_consistency_jsd/clean_consistency_total": float(clean_consistency_total),
                                    "generation_consistency_jsd/clean_consistency_pos_tokens": float(
                                        clean_consistency_pos_count
                                    ),
                                    "generation_consistency_jsd/clean_consistency_neg_tokens": float(
                                        clean_consistency_neg_count
                                    ),
                                    "generation_consistency_jsd/clean_consistency_pos_frac": (
                                        float(clean_consistency_pos_count)
                                        / max(float(clean_consistency_total), 1.0)
                                    ),
                                    "generation_consistency_jsd/clean_consistency_neg_frac": (
                                        float(clean_consistency_neg_count)
                                        / max(float(clean_consistency_total), 1.0)
                                    ),
                                    "generation_consistency_jsd/clean_consistency_pos_loss": (
                                        clean_pos_loss.detach().item() if clean_pos_loss is not None else 0.0
                                    ),
                                    "generation_consistency_jsd/clean_consistency_neg_loss": (
                                        clean_neg_loss.detach().item() if clean_neg_loss is not None else 0.0
                                    ),
                                    "generation_consistency_jsd/clean_consistency_pos_coef": clean_consistency_pos_coef,
                                    "generation_consistency_jsd/clean_consistency_neg_coef": clean_consistency_neg_coef,
                                    "generation_consistency_jsd/clean_consistency_high_threshold": (
                                        clean_consistency_high_threshold
                                    ),
                                    "generation_consistency_jsd/clean_consistency_low_threshold": (
                                        clean_consistency_low_threshold
                                    ),
                                }
                                if clean_pos_prob_values:
                                    clean_pos_prob_all = torch.cat(clean_pos_prob_values, dim=0)
                                    clean_consistency_metrics[
                                        "generation_consistency_jsd/clean_consistency_pos_prob_mean"
                                    ] = clean_pos_prob_all.mean().item()
                                else:
                                    clean_consistency_metrics[
                                        "generation_consistency_jsd/clean_consistency_pos_prob_mean"
                                    ] = 0.0
                                if clean_neg_prob_values:
                                    clean_neg_prob_all = torch.cat(clean_neg_prob_values, dim=0)
                                    clean_consistency_metrics[
                                        "generation_consistency_jsd/clean_consistency_neg_prob_mean"
                                    ] = clean_neg_prob_all.mean().item()
                                else:
                                    clean_consistency_metrics[
                                        "generation_consistency_jsd/clean_consistency_neg_prob_mean"
                                    ] = 0.0
                                generation_jsd_metrics.update(clean_consistency_metrics)
                            if trace_clean_filter_enabled:
                                generation_jsd_metrics.update(
                                    {
                                        "generation_consistency_jsd/clean_filter_total": float(trace_clean_filter_total),
                                        "generation_consistency_jsd/clean_filter_kept": float(trace_clean_filter_kept),
                                        "generation_consistency_jsd/clean_filter_keep_frac": (
                                            float(trace_clean_filter_kept)
                                            / max(float(trace_clean_filter_total), 1.0)
                                        ),
                                    }
                                )
                        else:
                            generation_jsd_loss = torch.nan_to_num(logits_rmpad, nan=0.0, posinf=0.0, neginf=0.0).sum() * 0.0
                            generation_jsd_metrics = {
                                "generation_consistency_jsd/loss": 0.0,
                                "generation_consistency_jsd/ce_loss": 0.0,
                                "generation_consistency_jsd/entropy_mean": 0.0,
                                "generation_consistency_jsd/entropy_contribution": 0.0,
                                "generation_consistency_jsd/total_objective": 0.0,
                                "generation_consistency_jsd/entropy_coef": trace_ce_entropy_coef,
                                "generation_consistency_jsd/entropy_target": (
                                    trace_ce_entropy_target
                                    if trace_ce_entropy_target is not None
                                    else -1.0
                                ),
                                "generation_consistency_jsd/entropy_excess": 0.0,
                                "generation_consistency_jsd/entropy_mode_penalty": float(
                                    trace_ce_entropy_mode == "penalty"
                                ),
                                "generation_consistency_jsd/active_tokens": 0.0,
                                "generation_consistency_jsd/weighted_numerator": 0.0,
                                "generation_consistency_jsd/weighted_denominator": 0.0,
                                "generation_consistency_jsd/active_frac": 0.0,
                                "generation_consistency_jsd/teacher_max_prob_mean": 0.0,
                                "generation_consistency_jsd/teacher_entropy_mean": 0.0,
                                "generation_consistency_jsd/student_entropy_mean": 0.0,
                            }
                            if objective == "trace_future_ce":
                                generation_jsd_metrics.update(
                                    {
                                        "generation_consistency_jsd/future_logits_offset": float(
                                            generation_jsd_config.get("trace_future_logits_offset", 5)
                                        ),
                                        "generation_consistency_jsd/future_skip_tail_steps": float(
                                            generation_jsd_config.get("trace_future_skip_tail_steps", 0) or 0
                                        ),
                                        "generation_consistency_jsd/trace_future_hard_label_ce": (
                                            1.0 if trace_future_hard_label_ce else 0.0
                                        ),
                                    }
                                )
                        # Aggregate entropy values already projected for CE.
                        # This avoids a second conditional lm-head call, whose
                        # FSDP unshard order can differ across SP shards.
                        if _adj_ent_enable:
                            if _adj_ent_num_pairs > 0:
                                _aec_diag_marker = float(
                                    getattr(self, "_aec_diag_chunk_idx", -1) + 1
                                )
                                _adj_ent_stats = torch.stack(
                                    [
                                        _adj_ent_entropy_sums.detach(),
                                        _adj_ent_token_counts,
                                        _adj_ent_prev_sums,
                                        _adj_ent_rollout_next_sums,
                                        _adj_ent_response_weight_sums,
                                        torch.full_like(
                                            _adj_ent_entropy_sums, _aec_diag_marker
                                        ),
                                        (
                                            _adj_ent_local_step_sums
                                            if _adj_ent_local_step_sums is not None
                                            else -_adj_ent_token_counts
                                        ),
                                    ],
                                    dim=-1,
                                )
                                _local_col1 = _adj_ent_stats[:, 1]
                                _local_frac = int(
                                    (_local_col1 - _local_col1.round()).abs().gt(1e-6).sum()
                                )
                                _local_zero_w = int(
                                    (_local_col1.gt(0) & _adj_ent_stats[:, 4].le(0)).sum()
                                )
                                if _local_frac or _local_zero_w:
                                    print(
                                        "[SDAR_STREAM_DIAG] aec_stats_local_anomaly "
                                        f"rank={torch.distributed.get_rank() if torch.distributed.is_initialized() else -1} "
                                        f"num_pairs={_adj_ent_num_pairs} "
                                        f"frac_count_slots={_local_frac} zero_weight_counted_slots={_local_zero_w} "
                                        f"pre_allreduce=1",
                                        flush=True,
                                    )
                                if self.use_ulysses_sp:
                                    torch.distributed.all_reduce(
                                        _adj_ent_stats,
                                        op=torch.distributed.ReduceOp.SUM,
                                        group=get_ulysses_sequence_parallel_group(),
                                    )
                                    _aec_marker_col = _adj_ent_stats[:, 5]
                                    _aec_sp_world = torch.distributed.get_world_size(
                                        group=get_ulysses_sequence_parallel_group()
                                    )
                                    if _adj_ent_stats.shape[0] > 0 and not bool(
                                        _aec_marker_col.eq(
                                            _aec_diag_marker * _aec_sp_world
                                        ).all()
                                    ):
                                        print(
                                            "[SDAR_STREAM_DIAG] aec_sp_desync "
                                            f"rank={torch.distributed.get_rank()} "
                                            f"local_marker={_aec_diag_marker} "
                                            f"reduced_marker_min={float(_aec_marker_col.min())} "
                                            f"reduced_marker_max={float(_aec_marker_col.max())} "
                                            f"num_pairs={_adj_ent_num_pairs}",
                                            flush=True,
                                        )
                                _adj_counts = _adj_ent_stats[:, 1]
                                _adj_valid_pairs = _adj_counts.gt(0)
                                if bool(_adj_valid_pairs.any()):
                                    _adj_h_next = (
                                        _adj_ent_stats[:, 0][
                                            _adj_valid_pairs
                                        ]
                                        / _adj_counts[_adj_valid_pairs]
                                    )
                                    _adj_h_prev = (
                                        _adj_ent_stats[:, 2][
                                            _adj_valid_pairs
                                        ]
                                        / _adj_counts[_adj_valid_pairs]
                                    ).detach()
                                    _adj_h_rollout_next = (
                                        _adj_ent_stats[:, 3][
                                            _adj_valid_pairs
                                        ]
                                        / _adj_counts[_adj_valid_pairs]
                                    ).detach()
                                    _adj_pair_weights = (
                                        _adj_ent_stats[:, 4][
                                            _adj_valid_pairs
                                        ]
                                        / _adj_counts[_adj_valid_pairs]
                                    ).detach()
                                    _adj_pair_local_step = (
                                        _adj_ent_stats[:, 6][
                                            _adj_valid_pairs
                                        ]
                                        / _adj_counts[_adj_valid_pairs]
                                    ).detach()
                                    if not bool(
                                        torch.isfinite(_adj_pair_weights).all()
                                        and _adj_pair_weights.gt(0).all()
                                    ):
                                        _adj_bad = (
                                            ~torch.isfinite(_adj_pair_weights)
                                        ) | _adj_pair_weights.le(0)
                                        print(
                                            "[SDAR_STREAM_DIAG] aec_aggregate_weight_anomaly "
                                            f"bad_pairs={int(_adj_bad.sum())}/{int(_adj_pair_weights.numel())} "
                                            f"bad_weight_sums={_adj_ent_stats[:, 4][_adj_valid_pairs][_adj_bad][:8].tolist()} "
                                            f"bad_counts={_adj_counts[_adj_valid_pairs][_adj_bad][:8].tolist()} "
                                            f"weight_min={float(_adj_pair_weights.min())} "
                                            f"weight_max={float(_adj_pair_weights.max())} "
                                            f"sp={int(self.use_ulysses_sp)}",
                                            flush=True,
                                        )
                                        raise RuntimeError(
                                            "adjacent entropy response weights "
                                            "must be finite and positive"
                                        )
                                    # Single-source gate mode: eligibility was
                                    # already applied upstream by editing the
                                    # pair mask (full-footprint H_prev); the
                                    # in-path gate must stay inert to avoid
                                    # double-gating with a different H_prev
                                    # scope (CE-window vs full footprint).
                                    if getattr(
                                        self, "_aec_gate_single_source", False
                                    ):
                                        _adj_ceiling = 0.0
                                        _adj_floor = 0.0
                                    else:
                                        _adj_ceiling = float(
                                            _adj_ent_cfg_dict.get("ceiling", 0.0)
                                            or 0.0
                                        )
                                        _adj_floor = float(
                                            _adj_ent_cfg_dict.get("floor", 0.0)
                                            or 0.0
                                        )
                                    # Entropy gate: build eligible mask.
                                    # ceiling > 0: exempt pairs with H_prev > ceiling (strict).
                                    # floor > 0: exempt pairs with H_prev < floor (strict).
                                    # Equality with threshold participates.
                                    # floor and ceiling are mutually exclusive (config validates).
                                    _adj_total_pairs_count = int(
                                        _adj_valid_pairs.sum().item()
                                    )
                                    _adj_eligible_mask = torch.ones(
                                        _adj_h_prev.shape,
                                        dtype=torch.bool,
                                        device=_adj_h_prev.device,
                                    )
                                    _adj_floor_exempt_count = 0
                                    _adj_ceiling_exempt_count = 0
                                    if _adj_ceiling > 0.0:
                                        _ceiling_mask = _adj_h_prev > _adj_ceiling
                                        _adj_eligible_mask = _adj_eligible_mask & ~_ceiling_mask
                                        _adj_ceiling_exempt_count = int(_ceiling_mask.sum().item())
                                    if _adj_floor > 0.0:
                                        _floor_mask = _adj_h_prev < _adj_floor
                                        _adj_eligible_mask = _adj_eligible_mask & ~_floor_mask
                                        _adj_floor_exempt_count = int(_floor_mask.sum().item())
                                    # Exempt pairs contribute zero excess; eligible pairs
                                    # are normalized over their own count/weight.
                                    # Per-pair rho ramp (block-local denoise depth):
                                    # rho_k = min(rho_ramp_per_step * k, rho), where k is
                                    # the pair's block-local iteration index. rho stays the
                                    # plateau/ceiling value. ramp<=0 keeps the original
                                    # scalar-rho expression bit-exactly.
                                    _adj_rho_max = float(
                                        _adj_ent_cfg_dict.get("rho", 0.05)
                                    )
                                    _adj_rho_ramp = float(
                                        _adj_ent_cfg_dict.get("rho_ramp_per_step", 0.0)
                                        or 0.0
                                    )
                                    if _adj_rho_ramp > 0.0:
                                        if bool(_adj_pair_local_step.lt(0.0).any()):
                                            raise RuntimeError(
                                                "rho_ramp_per_step requires "
                                                "adjacent_entropy_pair_local_step for "
                                                "every pair; found negative/missing "
                                                "local steps "
                                                f"(min={float(_adj_pair_local_step.min())})"
                                            )
                                        _adj_rho_vec = torch.clamp(
                                            _adj_rho_ramp * _adj_pair_local_step,
                                            max=_adj_rho_max,
                                        )
                                        print(
                                            "[AEC_RHO_RAMP] ramp="
                                            f"{_adj_rho_ramp} rho_max={_adj_rho_max} "
                                            f"pairs={int(_adj_pair_local_step.numel())} "
                                            f"k_min={float(_adj_pair_local_step.min()):.0f} "
                                            f"k_max={float(_adj_pair_local_step.max()):.0f} "
                                            f"rho_mean={float(_adj_rho_vec.mean()):.4f} "
                                            f"at_plateau={int(_adj_rho_vec.ge(_adj_rho_max).sum())}",
                                            flush=True,
                                        )
                                        _adj_excess = torch.where(
                                            _adj_eligible_mask,
                                            _adj_h_next
                                            - (1.0 - _adj_rho_vec) * _adj_h_prev,
                                            torch.zeros_like(_adj_h_next),
                                        )
                                    else:
                                        _adj_excess = torch.where(
                                            _adj_eligible_mask,
                                            _adj_h_next - (
                                                1.0 - float(_adj_ent_cfg_dict.get("rho", 0.05))
                                            ) * _adj_h_prev,
                                            torch.zeros_like(_adj_h_next),
                                        )
                                    _adj_positive_excess = torch.relu(
                                        _adj_excess
                                    )
                                    _adj_pair_losses = (
                                        _adj_positive_excess.square()
                                    )
                                    # Per-response eligible renormalization:
                                    # Scale eligible pair weights so each response
                                    # with >=1 eligible pair retains its full
                                    # aggregate weight (= 1.0 per response for
                                    # equal-weight scheme). Exempt pairs get
                                    # weight 0 via the eligible mask.
                                    # For batch_size=1 (our experiment) this is
                                    # exact per-response renorm; for larger
                                    # batches without per-response grouping it
                                    # is equivalent when all responses have the
                                    # same number of pairs.
                                    _adj_total_weight_sum = float(
                                        _adj_pair_weights.sum().item()
                                    )
                                    _adj_ent_pair_count = int(
                                        _adj_eligible_mask.sum().item()
                                    )
                                    _adj_eligible_weight_pre_renorm = float(
                                        (_adj_pair_weights * _adj_eligible_mask.float()).sum().item()
                                    )
                                    # renorm=false (config knob): keep original
                                    # per-pair weights and simply zero exempt
                                    # pairs ("proportional voice"): a heavily
                                    # gated response contributes proportionally
                                    # less, and AEC pressure fades naturally as
                                    # more pairs become exempt. Avoids the
                                    # chunk-scoped cross-response mass transfer
                                    # of the renorm scheme and makes
                                    # post-renorm == pre-renorm identically.
                                    _adj_do_renorm = bool(
                                        _adj_ent_cfg_dict.get("renorm", True)
                                    )
                                    if not _adj_do_renorm:
                                        _adj_renorm_factor = (
                                            1.0
                                            if _adj_ent_pair_count > 0
                                            else 0.0
                                        )
                                    elif (
                                        _adj_ent_pair_count > 0
                                        and _adj_eligible_weight_pre_renorm > 0.0
                                    ):
                                        _adj_renorm_factor = (
                                            _adj_total_weight_sum
                                            / _adj_eligible_weight_pre_renorm
                                        )
                                    else:
                                        _adj_renorm_factor = 0.0
                                    # Renormalized weights: eligible pairs scaled
                                    # up; exempt pairs zeroed out.
                                    _adj_renorm_weights = (
                                        _adj_pair_weights
                                        * _adj_renorm_factor
                                        * _adj_eligible_mask.float()
                                    )
                                    # The SP reduction above is deliberately
                                    # detached to avoid an autograd collective
                                    # interleaving with FSDP collectives. This
                                    # surrogate has the exact derivative of the
                                    # global pair loss with respect to each
                                    # shard-local entropy sum while preserving
                                    # the true detached scalar value.
                                    _adj_local_grad_term = (
                                        (
                                            2.0
                                            * _adj_positive_excess.detach()
                                            * _adj_renorm_weights
                                            / _adj_counts[_adj_valid_pairs]
                                        )
                                        * _adj_ent_entropy_sums[
                                            _adj_valid_pairs
                                        ]
                                    ).sum()
                                    _adj_true_hinge_sum = (
                                        (
                                            _adj_pair_losses
                                            * _adj_renorm_weights
                                        ).sum().detach()
                                    )
                                    _adj_ent_hinge_sum = (
                                        _adj_true_hinge_sum
                                        + _adj_local_grad_term
                                        - _adj_local_grad_term.detach()
                                    )
                                    # Eligible weight sum after renorm equals
                                    # total_weight_sum when eligible>0 (1.0 per
                                    # response), 0.0 when no eligible pairs.
                                    _adj_ent_pair_weight_sum = float(
                                        _adj_renorm_weights.sum().item()
                                    )
                                    # Pre-renorm eligible weight: the assertion
                                    # denominator (matches the prepass, which
                                    # cannot know micro-batch chunk boundaries
                                    # and therefore cannot reproduce the
                                    # post-renorm sum).
                                    _adj_ent_pair_weight_pre_renorm = float(
                                        _adj_eligible_weight_pre_renorm
                                    )
                                    _adj_h_prev_sum = float(
                                        _adj_h_prev[_adj_eligible_mask].sum().item()
                                        if _adj_eligible_mask.any()
                                        else 0.0
                                    )
                                    _adj_h_next_sum = float(
                                        _adj_h_next.detach()[_adj_eligible_mask].sum().item()
                                        if _adj_eligible_mask.any()
                                        else 0.0
                                    )
                                    _adj_violation_count = float(
                                        _adj_excess.detach().gt(0).sum().item()
                                    )
                                    _adj_mismatch = (
                                        _adj_h_next.detach()
                                        - _adj_h_rollout_next
                                    )
                                    _adj_rollout_next_sum = float(
                                        _adj_h_rollout_next.sum().item()
                                    )
                                    _adj_mismatch_signed_sum = float(
                                        _adj_mismatch.sum().item()
                                    )
                                    _adj_mismatch_abs_sum = float(
                                        _adj_mismatch.abs().sum().item()
                                    )
                                    _adj_eligible_fraction = (
                                        float(_adj_ent_pair_count)
                                        / float(_adj_total_pairs_count)
                                        if _adj_total_pairs_count > 0
                                        else 0.0
                                    )
                                    _adj_ent_metrics = {
                                        "adj_ent/avg_entropy_t": (
                                            _adj_h_prev_sum
                                            / max(_adj_ent_pair_count, 1)
                                        ),
                                        "adj_ent/avg_entropy_next": (
                                            _adj_h_next_sum
                                            / max(_adj_ent_pair_count, 1)
                                        ),
                                        "adj_ent/relative_entropy_drop": (
                                            _adj_h_prev_sum
                                            - _adj_h_next_sum
                                        )
                                        / max(
                                            _adj_h_prev_sum,
                                            float(
                                                _adj_ent_cfg_dict.get(
                                                    "eps", 1e-8
                                                )
                                            ),
                                        ),
                                        "adj_ent/violation_rate": (
                                            _adj_violation_count
                                            / max(_adj_ent_pair_count, 1)
                                        ),
                                        "adj_ent/valid_pair_count": float(
                                            _adj_ent_pair_count
                                        ),
                                        "adj_ent/total_pairs": float(
                                            _adj_total_pairs_count
                                        ),
                                        "adj_ent/eligible_pairs": float(
                                            _adj_ent_pair_count
                                        ),
                                        "adj_ent/floor_exempt": float(
                                            _adj_floor_exempt_count
                                        ),
                                        "adj_ent/ceiling_exempt": float(
                                            _adj_ceiling_exempt_count
                                        ),
                                        "adj_ent/eligible_fraction": (
                                            _adj_eligible_fraction
                                        ),
                                        "adj_ent/eligible_weight_mass": (
                                            _adj_eligible_weight_pre_renorm
                                            / _adj_total_weight_sum
                                            if _adj_total_weight_sum > 0.0
                                            else 0.0
                                        ),
                                        "adj_ent/eligible_hinge_violations": (
                                            _adj_violation_count
                                        ),
                                        "adj_ent/pair_weight_sum": (
                                            _adj_ent_pair_weight_sum
                                        ),
                                        "adj_ent/response_level": float(
                                            _adj_ent_response_level
                                        ),
                                        "adj_ent/hinge_sum": float(
                                            _adj_ent_hinge_sum.detach().item()
                                        ),
                                        "_adj_ent/h_t_sum": _adj_h_prev_sum,
                                        "_adj_ent/h_next_sum": _adj_h_next_sum,
                                        "_adj_ent/violation_count": (
                                            _adj_violation_count
                                        ),
                                        "_adj_ent/rollout_next_sum": (
                                            _adj_rollout_next_sum
                                        ),
                                        "_adj_ent/mismatch_signed_sum": (
                                            _adj_mismatch_signed_sum
                                        ),
                                        "_adj_ent/mismatch_abs_sum": (
                                            _adj_mismatch_abs_sum
                                        ),
                                    }
                                else:
                                    _adj_ent_hinge_sum = (
                                        logits_rmpad.float().sum() * 0.0
                                    )
                            else:
                                _adj_ent_hinge_sum = (
                                    logits_rmpad.float().sum() * 0.0
                                )
                        if fast_hidden_ce:
                            # The clean-agree objective replaces PPO loss.  Avoid
                            # materializing full-vocab logits again just to compute
                            # diagnostic log_probs that will not drive gradients.
                            valid_label_mask = torch.zeros_like(valid_label_mask)
                    needs_full_logits = calculate_entropy or calculate_sum_pi_squared
                    if needs_full_logits:
                        logits_rmpad.div_(logprob_temperature)
                        finite_logits = torch.isfinite(logits_rmpad)
                        logits_rmpad = torch.where(
                            finite_logits,
                            logits_rmpad,
                            torch.zeros_like(logits_rmpad).detach(),
                        )

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = torch.zeros(
                        labels_rmpad.shape,
                        device=logits_rmpad.device,
                        dtype=logits_rmpad.dtype,
                    )
                    if valid_label_mask.any():
                        valid_logits = logits_rmpad[valid_label_mask]
                        if not needs_full_logits:
                            valid_logits = valid_logits.float().div(logprob_temperature)
                            valid_logits = torch.where(
                                torch.isfinite(valid_logits),
                                valid_logits,
                                torch.zeros_like(valid_logits).detach(),
                            )
                        else:
                            valid_logits = valid_logits.float()
                        valid_labels = labels_rmpad[valid_label_mask]
                        valid_log_probs = torch.nn.functional.log_softmax(valid_logits, dim=-1)
                        gathered = valid_log_probs.gather(
                            dim=-1,
                            index=valid_labels.unsqueeze(-1),
                        ).squeeze(-1)
                        log_probs[valid_label_mask] = gathered.to(log_probs.dtype)

                    dense_log_probs_rmpad = None
                    dense_token_probs_rmpad = None
                    tpf_action_log_probs_rmpad = None
                    if tpf_labels_rmpad is not None:
                        tpf_valid_mask = tpf_labels_rmpad.ne(-100)
                        # Keep ghost-only rank-alignment micro-batches connected
                        # to the full model graph. Their objective mask is zero,
                        # so this changes no gradients, but backward must still
                        # traverse FSDP in the same order on every rank.
                        tpf_action_log_probs_rmpad = (
                            logits_rmpad.float().sum(dim=-1) * 0.0
                        )
                        if tpf_valid_mask.any():
                            tpf_sources = logits_rmpad[tpf_valid_mask]
                            tpf_labels_valid = tpf_labels_rmpad[
                                tpf_valid_mask
                            ]
                            projection_chunk_size = max(
                                int(
                                    generation_jsd_config.get(
                                        "row_chunk_size", 8
                                    )
                                    or 8
                                ),
                                1,
                            )
                            action_lm_head = None
                            if fast_hidden_ce:
                                actor_module_unwrapped = getattr(
                                    self.actor_module,
                                    "module",
                                    self.actor_module,
                                )
                                action_lm_head = getattr(
                                    actor_module_unwrapped,
                                    "lm_head",
                                    None,
                                )
                                if action_lm_head is None:
                                    raise RuntimeError(
                                        "negative_action_pg fast replay requires "
                                        "actor module lm_head"
                                    )
                            tpf_gathered = compute_chunked_action_log_probs(
                                tpf_sources,
                                tpf_labels_valid,
                                temperature=(
                                    logprob_temperature
                                    if (
                                        action_lm_head is not None
                                        or not needs_full_logits
                                    )
                                    else 1.0
                                ),
                                chunk_size=projection_chunk_size,
                                projection_weight=(
                                    action_lm_head.weight
                                    if action_lm_head is not None
                                    else None
                                ),
                                projection_bias=(
                                    action_lm_head.bias
                                    if action_lm_head is not None
                                    else None
                                ),
                            )
                            tpf_action_log_probs_rmpad[tpf_valid_mask] = tpf_gathered.to(
                                tpf_action_log_probs_rmpad.dtype
                            )
                    if dense_labels_rmpad is not None:
                        dense_valid_mask = dense_labels_rmpad.ne(-100)
                        dense_log_probs_rmpad = torch.zeros(
                            dense_labels_rmpad.shape,
                            device=logits_rmpad.device,
                            dtype=logits_rmpad.dtype,
                        )
                        dense_token_probs_rmpad = torch.zeros_like(dense_log_probs_rmpad)
                        if dense_valid_mask.any():
                            dense_logits = logits_rmpad[dense_valid_mask]
                            if not needs_full_logits:
                                dense_logits = dense_logits.float().div(logprob_temperature)
                                dense_logits = torch.where(
                                    torch.isfinite(dense_logits),
                                    dense_logits,
                                    torch.zeros_like(dense_logits).detach(),
                                )
                            else:
                                dense_logits = dense_logits.float()
                            dense_labels_valid = dense_labels_rmpad[dense_valid_mask]
                            dense_valid_log_probs = torch.nn.functional.log_softmax(dense_logits, dim=-1)
                            dense_gathered = dense_valid_log_probs.gather(
                                dim=-1,
                                index=dense_labels_valid.unsqueeze(-1),
                            ).squeeze(-1)
                            dense_log_probs_rmpad[dense_valid_mask] = dense_gathered.to(dense_log_probs_rmpad.dtype)
                            dense_token_probs_rmpad[dense_valid_mask] = dense_gathered.exp().to(dense_token_probs_rmpad.dtype)

                    # Check
                    # probs = probs_from_logits(
                    #     logits=logits_rmpad,
                    #     labels=labels_rmpad, #input_ids_rmpad_rolled,
                    #     inplace_backward=inplace_backward,
                    # )
                    # logits = extract_from_logits_v2(
                    #     logits=logits_rmpad,
                    #     labels=labels_rmpad,  # input_ids_rmpad_rolled,
                    # )

                    # compute entropy
                    if calculate_entropy:
                        # ((total_nnz / sp) + pad)
                        entropy_rmpad = (
                            self.compute_entropy_from_logits(logits_rmpad)
                            if not self.config.entropy_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits_rmpad)
                        )

                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = (
                            self.calculate_sum_pi_squared_from_logits(logits_rmpad)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(
                                self.calculate_sum_pi_squared_from_logits, logits_rmpad
                            )
                        )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = gather_outputs_and_unpad(
                            sum_pi_squared_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )

                if is_mask_all_zero:
                    log_probs = log_probs[:0]
                    if tpf_action_log_probs_rmpad is not None:
                        tpf_action_log_probs_rmpad = tpf_action_log_probs_rmpad[:0]
                    if dense_labels_rmpad is not None:
                        dense_log_probs_rmpad = dense_log_probs_rmpad[:0]
                        dense_token_probs_rmpad = dense_token_probs_rmpad[:0]

                    # Check
                    # probs = probs[:0]
                    # logits = logits[:0]

                    if calculate_entropy:
                        entropy_rmpad = entropy_rmpad[:0]

                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if calculate_sum_pi_squared:
                    full_sum_pi_squared = pad_input(
                        hidden_states=sum_pi_squared_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )

                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                if tpf_action_log_probs_rmpad is not None:
                    full_tpf_action_log_probs = pad_input(
                        hidden_states=tpf_action_log_probs_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if dense_labels_rmpad is not None:
                    full_dense_log_probs = pad_input(
                        hidden_states=dense_log_probs_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    full_dense_token_probs = pad_input(
                        hidden_states=dense_token_probs_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )

                # Check
                # full_probs = pad_input(
                #     hidden_states=probs.unsqueeze(-1),
                #     indices=indices,
                #     batch=batch_size,
                #     seqlen=seqlen,
                # )
                # full_logits = pad_input(
                #     hidden_states=logits.unsqueeze(-1),
                #     indices=indices,
                #     batch=batch_size,
                #     seqlen=seqlen,
                # )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length:]  # (bsz, response_length)
                if calculate_sum_pi_squared:
                    # (bsz, response_length)
                    sum_pi_squared = full_sum_pi_squared.squeeze(-1)[:, -response_length:]
                log_probs = full_log_probs.squeeze(-1)[:, -response_length:]  # (bsz, response_length)
                if tpf_action_log_probs_rmpad is not None:
                    tpf_action_log_probs = full_tpf_action_log_probs.squeeze(-1)[:, -response_length:]
                if dense_labels_rmpad is not None:
                    dense_log_probs = full_dense_log_probs.squeeze(-1)[:, -response_length:]
                    dense_token_probs = full_dense_token_probs.squeeze(-1)[:, -response_length:]

                # Check
                # probs = full_probs.squeeze(-1)[:, -response_length:]
                # logits = full_logits.squeeze(-1)[:, -response_length:]

            else:  # not using rmpad and no ulysses sp
                raise NotImplementedError
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = self._effective_logprob_temperature(temperature)
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(self._effective_logprob_temperature(temperature))
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared = (
                            self.calculate_sum_pi_squared_from_logits(logits)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.calculate_sum_pi_squared_from_logits, logits)
                        )
            # Check
            # response_mask = micro_batch["response_mask"]
            # print(responses.masked_select(response_mask)[:20], probs.masked_select(response_mask)[:20],micro_batch["rollout_log_probs"].masked_select(response_mask)[:20])
            if os.environ.get("VERL_ACTOR_REPLAY_DEBUG", "0") == "1":
                try:
                    debug_limit = int(os.environ.get("VERL_ACTOR_REPLAY_DEBUG_LIMIT", "12"))
                    response_mask = micro_batch.get("response_mask", torch.ones_like(log_probs, dtype=torch.bool)).bool()
                    rollout_lp = micro_batch.get("rollout_log_probs", None)
                    source_hidden = micro_batch.get("rollout_logits_source_hidden", None)
                    active = torch.nonzero(response_mask, as_tuple=False)
                    rows = []
                    for b, pos in active[:debug_limit].detach().cpu().tolist():
                        noisy_abs = input_ids.shape[1] - response_length + pos
                        lo = max(0, noisy_abs - 4)
                        hi = min(input_ids.shape[1], noisy_abs + 5)
                        source_nonzero = None
                        if source_hidden is not None:
                            source_nonzero = bool(source_hidden[b, pos].detach().float().abs().sum().item() > 0)
                        rows.append(
                            {
                                "batch": int(b),
                                "response_pos": int(pos),
                                "token_id": int(micro_batch["responses"][b, pos].detach().cpu().item()),
                                "actor_logprob": float(log_probs[b, pos].detach().float().cpu().item()),
                                "rollout_logprob": (
                                    float(rollout_lp[b, pos].detach().float().cpu().item())
                                    if rollout_lp is not None
                                    else None
                                ),
                                "noisy_abs_pos": int(noisy_abs),
                                "input_ids_around_noisy": [
                                    int(x) for x in input_ids[b, lo:hi].detach().cpu().tolist()
                                ],
                                "has_source_hidden": source_nonzero,
                            }
                        )
                    print(f"[ActorReplayDebug] rows={rows}", flush=True)
                except Exception as exc:
                    print(f"[ActorReplayDebug] failed: {exc}", flush=True)

            if self._actor_replay_token_debug_enabled():
                self._dump_actor_replay_token_debug(
                    micro_batch=micro_batch,
                    log_probs=log_probs,
                    input_ids=input_ids,
                    response_length=response_length,
                )
                if teacher_token_debug_rows:
                    try:
                        path = self._actor_replay_token_debug_path()
                        with open(path, "a", encoding="utf-8") as fp:
                            for row in teacher_token_debug_rows:
                                fp.write(json.dumps(row, sort_keys=True) + "\n")
                        print(
                            f"[ActorReplayTokenDebug] wrote {len(teacher_token_debug_rows)} teacher rows to {path}",
                            flush=True,
                        )
                    except Exception as exc:
                        print(f"[ActorReplayTokenDebug] teacher dump failed: {exc}", flush=True)

            outputs = {"log_probs": log_probs}
            if "tpf_action_ids" in micro_batch:
                outputs["tpf_action_log_probs"] = tpf_action_log_probs
            if generation_jsd_loss is not None:
                outputs["generation_jsd_loss"] = generation_jsd_loss
                outputs["generation_jsd_metrics"] = generation_jsd_metrics
                if _entropy_rkl_probe_enable:
                    if _entropy_rkl_probe_stats is None:
                        _probe_empty = torch.empty(
                            (0, int(vocab_size)),
                            dtype=torch.float32,
                            device=logits_rmpad.device,
                        )
                        _entropy_rkl_probe_stats = (
                            compute_entropy_gated_rkl_probe_stats(
                                _probe_empty,
                                _probe_empty,
                                eligible_count=0,
                                entropy_thresholds=(
                                    _entropy_rkl_probe_thresholds
                                ),
                                logit_clip=logit_clip,
                            )
                        )
                    outputs["entropy_gated_rkl_probe_stats"] = (
                        _entropy_rkl_probe_stats
                    )
                if _entropy_gated_rkl_enable:
                    outputs["entropy_gated_rkl_loss_stats"] = (
                        _entropy_gated_rkl_report
                    )
                if _adj_ent_hinge_sum is not None or _adj_ent_pair_count > 0:
                    outputs["adj_ent_hinge_sum"] = _adj_ent_hinge_sum
                    outputs["adj_ent_pair_count"] = _adj_ent_pair_count
                    outputs["adj_ent_pair_weight_sum"] = (
                        _adj_ent_pair_weight_sum
                    )
                    outputs["adj_ent_pair_weight_pre_renorm"] = (
                        _adj_ent_pair_weight_pre_renorm
                    )
                    outputs["adj_ent_metrics"] = _adj_ent_metrics
                if signed_trace_ce_enabled:
                    outputs["signed_trace_ce_positive_sum"] = positive_sum
                    outputs["signed_trace_ce_negative_sum"] = negative_sum
                    outputs["signed_trace_ce_positive_tokens"] = (
                        signed_positive_ce_count
                    )
                    outputs["signed_trace_ce_negative_tokens"] = (
                        signed_negative_ce_count
                    )
                if causal_wrong_ce_actor_enabled:
                    outputs["causal_wrong_ce_positive_sum"] = cwce_pos_sum
                    outputs["causal_wrong_ce_wrong_sum"] = cwce_neg_sum
                    outputs["causal_wrong_ce_positive_tokens"] = (
                        causal_wrong_positive_ce_count
                    )
                    outputs["causal_wrong_ce_wrong_tokens"] = (
                        causal_wrong_neg_ce_count
                    )
            if "dense_noisy_labels" in micro_batch and self.config.get("dense_noisy_pg", {}).get("enable", False):
                outputs["dense_log_probs"] = dense_log_probs
                outputs["dense_token_probs"] = dense_token_probs
            if calculate_entropy:
                outputs["entropys"] = entropy
            if calculate_sum_pi_squared:
                outputs["sum_pi_squared"] = sum_pi_squared
            return outputs

    def _causal_ar_rollout_score_micro_batch(
        self, micro_batch: dict[str, torch.Tensor], temperature: float
    ) -> dict[str, torch.Tensor]:
        """Teacher-force clean rollout tokens with an explicit causal mask.

        SDAR eval treats an all-ones mask as bidirectional, so the AR-prob
        advantage path must pass a real lower-triangular mask.
        """
        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            responses = micro_batch["responses"]
            attention_2d = micro_batch["attention_mask"].bool()
            response_mask = micro_batch.get("response_mask", torch.ones_like(responses, dtype=torch.bool)).bool()

            batch_size, seqlen = input_ids.shape
            response_length = responses.shape[-1]
            logprob_temperature = self._effective_logprob_temperature(temperature)
            chunk_size = int(self.config.get("causal_ar_score_chunk_size", 128))
            log_probs = torch.zeros(responses.shape, device=input_ids.device, dtype=torch.float32)
            prompt_width = seqlen - response_length
            for b in range(batch_size):
                prompt_valid = attention_2d[b, :prompt_width]
                resp_valid = response_mask[b]
                prompt_ids = input_ids[b, :prompt_width][prompt_valid]
                resp_ids = responses[b][resp_valid]
                if prompt_ids.numel() == 0 or resp_ids.numel() == 0:
                    continue
                noisy_ids = torch.full_like(resp_ids, micro_batch.get("pad_token_id", 0))
                compact_ids = torch.cat([prompt_ids, resp_ids, noisy_ids], dim=0).unsqueeze(0)
                compact_len = compact_ids.shape[1]
                prompt_len = prompt_ids.numel()
                resp_len = resp_ids.numel()
                compact_pos = torch.cat(
                    [
                        torch.arange(prompt_len + resp_len, device=input_ids.device),
                        torch.arange(prompt_len, prompt_len + resp_len, device=input_ids.device),
                    ],
                    dim=0,
                ).unsqueeze(0)
                score_model = self.actor_module
                was_training = score_model.training
                score_model.train()
                metadata = {
                    "cu_seqlens": torch.tensor([0, compact_len], device=input_ids.device, dtype=torch.int32),
                    "L0": torch.tensor([prompt_len], device=input_ids.device, dtype=torch.int32),
                    "L1": torch.tensor([resp_len], device=input_ids.device, dtype=torch.int32),
                    # staircase_size=1 makes noisy-position logits shift from
                    # prompt/clean previous-token states, i.e. AR teacher forcing.
                    "staircase_size": 1,
                    "max_seqlen": compact_len,
                    "backend": "fast",
                    "shift_noisy_logits": True,
                }
                try:
                    output = score_model(
                        input_ids=compact_ids,
                        attention_mask=None,
                        position_ids=compact_pos,
                        use_cache=False,
                        return_dict=True,
                        preexpanded_attention_metadata=metadata,
                    )
                finally:
                    score_model.train(was_training)
                logits = output.logits[:, prompt_len + resp_len :, :].squeeze(0).contiguous()
                if os.environ.get("VERL_CAUSAL_AR_SCORE_DEBUG", "0") == "1":
                    try:
                        finite = torch.isfinite(logits[: min(chunk_size, logits.shape[0])])
                        print(
                            "[CausalARScoreDebugLogits] "
                            f"sample={b} finite_frac={finite.float().mean().detach().cpu().item()} "
                            f"min={torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0).min().detach().cpu().item()} "
                            f"max={torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0).max().detach().cpu().item()} "
                            f"preexpanded_ar=1 model_training={score_model.training}",
                            flush=True,
                        )
                    except Exception as exc:
                        print(f"[CausalARScoreDebugLogits] failed: {exc}", flush=True)
                sample_log_probs = torch.zeros((resp_ids.numel(),), device=input_ids.device, dtype=torch.float32)
                for start in range(0, resp_ids.numel(), chunk_size):
                    end = min(start + chunk_size, resp_ids.numel())
                    chunk_logits = logits[start:end, :].float().div(logprob_temperature)
                    chunk_logits = torch.where(
                        torch.isfinite(chunk_logits),
                        chunk_logits,
                        torch.zeros_like(chunk_logits).detach(),
                    )
                    chunk_log_probs = torch.nn.functional.log_softmax(chunk_logits, dim=-1).gather(
                        dim=-1,
                        index=resp_ids[start:end, None],
                    ).squeeze(-1)
                    chunk_log_probs = torch.nan_to_num(chunk_log_probs, nan=-100.0, posinf=0.0, neginf=-100.0)
                    sample_log_probs[start:end] = chunk_log_probs
                log_probs[b, resp_valid] = sample_log_probs

            token_probs = log_probs.exp()
            token_probs = torch.nan_to_num(token_probs, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
            token_probs = torch.where(response_mask, token_probs, torch.zeros_like(token_probs))
            denom = response_mask.sum(dim=-1).clamp_min(1).to(dtype=torch.float32)
            avg_prob = token_probs.sum(dim=-1) / denom
            return {
                "ar_token_log_probs": log_probs,
                "ar_token_probs": token_probs,
                "ar_avg_prob": avg_prob,
            }

    def _process_pad(
        self,
        attn: torch.Tensor,   # [B, 1, N, N]
        input_ids: torch.Tensor,  # [B, N]
        line_pos: int,
        pad_token_id: int
    ) -> torch.Tensor:
        """Zero out attention columns for padding tokens before line_pos.

        Mirrors process_pad in rl_sdar.py.
        """
        N = input_ids.shape[1]
        device = input_ids.device
        cols = torch.arange(N, device=device)
        key_mask = (cols < line_pos).unsqueeze(0) & (input_ids == pad_token_id)  # [B, N]
        attn = attn.masked_fill(key_mask[:, None, None, :], 0)
        # Fix rows that became all-zero (avoid NaN in softmax)
        A = attn[:, 0]  # [B, N, N]
        bad = (A.sum(-1) == 0) & (torch.arange(A.size(1), device=device).unsqueeze(0) < line_pos)
        b_idx, r_idx = bad.nonzero(as_tuple=True)
        if len(b_idx):
            A[b_idx, r_idx, :] = 0.0
            A[b_idx, r_idx, r_idx] = 1.0
        return attn


    def _reset_stream_trace_all_ce_grad_scale(self):
        self._stream_trace_all_ce_grad_weighted_denominator = 0.0
        self._stream_trace_all_ce_grad_chunks = 0
        self._stream_trace_all_ce_grad_scale_pending = False

    def _accumulate_stream_trace_all_ce_grad_scale(self, weighted_denominator: float):
        weighted_denominator = max(float(weighted_denominator or 0.0), 0.0)
        self._stream_trace_all_ce_grad_weighted_denominator = (
            float(getattr(self, "_stream_trace_all_ce_grad_weighted_denominator", 0.0) or 0.0)
            + weighted_denominator
        )
        self._stream_trace_all_ce_grad_chunks = (
            int(getattr(self, "_stream_trace_all_ce_grad_chunks", 0) or 0) + 1
        )
        self._stream_trace_all_ce_grad_scale_pending = True

    @staticmethod
    def _actor_param_checksum_debug_enabled() -> bool:
        return os.environ.get("VERL_ACTOR_PARAM_CHECKSUM_DEBUG", "0") == "1"

    @staticmethod
    def _actor_chunk_loss_debug_enabled() -> bool:
        return os.environ.get("VERL_ACTOR_CHUNK_LOSS_DEBUG", "0") == "1"

    @staticmethod
    def _actor_input_checksum_debug_enabled() -> bool:
        return os.environ.get("VERL_ACTOR_INPUT_CHECKSUM_DEBUG", "0") == "1"

    @staticmethod
    def _actor_param_checksum_rank() -> int:
        if torch.distributed.is_initialized():
            return int(torch.distributed.get_rank())
        try:
            return int(os.environ.get("RANK", "0"))
        except ValueError:
            return 0

    @staticmethod
    def _actor_param_checksum_scalar(value):
        if value is None:
            return None
        if isinstance(value, DTensor):
            value = value.full_tensor()
        if torch.is_tensor(value):
            if value.numel() <= 0:
                return None
            value = value.detach().reshape(-1)[0].float().cpu()
            return float(value.item())
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _actor_param_checksum_tensor_stats(tensor: torch.Tensor | None) -> dict | None:
        if tensor is None:
            return None
        if isinstance(tensor, DTensor):
            tensor = tensor.to_local()
        if getattr(tensor, "is_sparse", False):
            tensor = tensor.coalesce().values()
        tensor = tensor.detach()
        numel = int(tensor.numel())
        if numel <= 0:
            return {
                "numel": 0,
                "sum": 0.0,
                "abs_sum": 0.0,
                "sq_sum": 0.0,
                "max_abs": 0.0,
                "nonfinite": 0,
            }
        values = tensor.float().reshape(-1)
        finite = torch.isfinite(values)
        nonfinite = 0 if bool(finite.all().detach().cpu().item()) else int((~finite).sum().detach().cpu().item())
        values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        abs_values = values.abs()
        return {
            "numel": numel,
            "sum": float(values.sum().detach().cpu().item()),
            "abs_sum": float(abs_values.sum().detach().cpu().item()),
            "sq_sum": float((values * values).sum().detach().cpu().item()),
            "max_abs": float(abs_values.max().detach().cpu().item()),
            "nonfinite": nonfinite,
        }

    def _actor_param_checksum_debug_path(self) -> str:
        dump_dir = os.environ.get("VERL_ACTOR_PARAM_CHECKSUM_DIR")
        if not dump_dir:
            dump_dir = os.path.join(os.getcwd(), "actor_param_checksum_debug")
        os.makedirs(dump_dir, exist_ok=True)
        rank = self._actor_param_checksum_rank()
        return os.path.join(dump_dir, f"rank{rank}_pid{os.getpid()}.jsonl")

    def _actor_chunk_loss_debug_path(self) -> str:
        dump_dir = os.environ.get("VERL_ACTOR_CHUNK_LOSS_DEBUG_DIR")
        if not dump_dir:
            dump_dir = os.path.join(os.getcwd(), "actor_chunk_loss_debug")
        os.makedirs(dump_dir, exist_ok=True)
        rank = self._actor_param_checksum_rank()
        return os.path.join(dump_dir, f"rank{rank}.jsonl")

    def _actor_input_checksum_debug_path(self) -> str:
        dump_dir = os.environ.get("VERL_ACTOR_INPUT_CHECKSUM_DIR")
        if not dump_dir:
            dump_dir = os.path.join(os.getcwd(), "actor_input_checksum_debug")
        os.makedirs(dump_dir, exist_ok=True)
        rank = self._actor_param_checksum_rank()
        return os.path.join(dump_dir, f"rank{rank}.jsonl")

    @staticmethod
    def _tensor_sha256(tensor: torch.Tensor) -> str:
        value = tensor.detach().contiguous().cpu()
        digest = hashlib.sha256()
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(json.dumps([int(dim) for dim in value.shape]).encode("utf-8"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    @classmethod
    def _gemma_replay_block_fingerprint_records(
        cls,
        model_inputs: dict[str, torch.Tensor],
    ) -> list[dict[str, Any]]:
        block_real = model_inputs["gemma_replay_block_real"].bool().reshape(-1)
        row_valid = model_inputs["gemma_replay_row_valid"].bool()
        attention_mask = model_inputs["attention_mask"].bool()
        records: list[dict[str, Any]] = []
        identity_keys = (
            "gemma_replay_request_uid_hashes",
            "gemma_replay_sample_uid_hashes",
            "gemma_replay_response_uid_hashes",
            "gemma_replay_identity_hashes",
        )
        for slot in torch.nonzero(block_real, as_tuple=False).reshape(-1).tolist():
            prefix_active = attention_mask[slot]
            active_rows = row_valid[slot]
            fields = {
                "prefix_ids": model_inputs["input_ids"][slot][prefix_active],
                "prefix_positions": model_inputs["position_ids"][slot][prefix_active],
                "responses": model_inputs["responses"][slot],
                "response_mask": model_inputs["response_mask"][slot],
                "canvas_ids": model_inputs["gemma_replay_canvas_ids"][slot][active_rows],
                "step_indices": model_inputs["gemma_replay_step_indices"][slot][active_rows],
                "valid_masks": model_inputs["gemma_replay_valid_masks"][slot][active_rows],
                "done_masks": model_inputs["gemma_replay_done_masks"][slot][active_rows],
                "self_conditioning_signal": model_inputs[
                    "gemma_replay_self_conditioning_signal"
                ][slot][active_rows],
                "self_conditioning_present": model_inputs[
                    "gemma_replay_self_conditioning_present"
                ][slot][active_rows],
            }
            tensor_hashes = {
                key: cls._tensor_sha256(value)
                for key, value in fields.items()
            }
            block_index = int(
                model_inputs["gemma_replay_block_indices"][slot]
                .detach()
                .cpu()
                .item()
            )
            sample_digest = hashlib.sha256()
            for key in ("prefix_ids", "responses", "response_mask"):
                sample_digest.update(key.encode("utf-8"))
                sample_digest.update(tensor_hashes[key].encode("ascii"))
            semantic_digest = hashlib.sha256()
            semantic_digest.update(str(block_index).encode("ascii"))
            for key in sorted(tensor_hashes):
                semantic_digest.update(key.encode("utf-8"))
                semantic_digest.update(tensor_hashes[key].encode("ascii"))
            identity_hashes = {}
            for key in identity_keys:
                value = model_inputs.get(key)
                if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) > slot:
                    identity_hashes[key] = [
                        int(item)
                        for item in value[slot].detach().cpu().reshape(-1).tolist()
                    ]
            records.append(
                {
                    "block_slot": int(slot),
                    "block_index": block_index,
                    "intermediate_rows": int(
                        active_rows.sum().detach().cpu().item()
                    ),
                    "active_token_steps": int(
                        (
                            model_inputs["gemma_replay_valid_masks"][slot][active_rows]
                            & model_inputs["response_mask"][slot].bool().unsqueeze(0)
                        )
                        .sum()
                        .detach()
                        .cpu()
                        .item()
                    ),
                    "sample_fingerprint": sample_digest.hexdigest(),
                    "semantic_fingerprint": semantic_digest.hexdigest(),
                    "tensor_hashes": tensor_hashes,
                    "identity_hashes": identity_hashes,
                }
            )
        return records

    def _dump_gemma_replay_block_fingerprint_debug(
        self,
        model_inputs: dict[str, torch.Tensor],
        *,
        global_step=None,
        chunk_idx=None,
    ) -> None:
        if not self._actor_input_checksum_debug_enabled():
            return
        try:
            common = {
                "rank": int(self._actor_param_checksum_rank()),
                "pid": int(os.getpid()),
                "phase": "gemma_replay_block",
                "global_step": self._actor_param_checksum_scalar(global_step),
                "chunk_idx": self._actor_param_checksum_scalar(chunk_idx),
            }
            records = self._gemma_replay_block_fingerprint_records(model_inputs)
            path = self._actor_input_checksum_debug_path()
            with open(path, "a", encoding="utf-8") as fp:
                for record in records:
                    fp.write(
                        json.dumps({**common, **record}, sort_keys=True) + "\n"
                    )
        except Exception as exc:
            print(
                f"[ActorInputChecksumDebug] failed phase=gemma_replay_block: {exc}",
                flush=True,
            )

    def _dump_actor_input_checksum_debug(
        self,
        micro_batch: dict[str, torch.Tensor],
        *,
        phase: str | None = None,
        global_step=None,
        batch_idx=None,
        micro_idx=None,
    ) -> None:
        if not self._actor_input_checksum_debug_enabled():
            return
        try:
            keys = [
                "input_ids",
                "attention_mask",
                "position_ids",
                "responses",
                "response_mask",
                "rollout_self_conditioning_signal",
                "rollout_logits_source_hidden",
                "rollout_log_probs",
                "rollout_quality_mask",
            ]
            tensors = {}
            with torch.no_grad():
                for key in keys:
                    value = micro_batch.get(key, None)
                    if not torch.is_tensor(value):
                        continue
                    tensors[key] = {
                        "dtype": str(value.dtype),
                        "shape": [int(dim) for dim in value.shape],
                        "stride": [int(dim) for dim in value.stride()],
                        "contiguous": bool(value.is_contiguous()),
                        "stats": self._actor_param_checksum_tensor_stats(value),
                    }
            record = {
                "rank": int(self._actor_param_checksum_rank()),
                "pid": int(os.getpid()),
                "phase": phase,
                "global_step": self._actor_param_checksum_scalar(global_step),
                "batch_idx": self._actor_param_checksum_scalar(batch_idx),
                "micro_idx": self._actor_param_checksum_scalar(micro_idx),
                "tensors": tensors,
            }
            path = self._actor_input_checksum_debug_path()
            with open(path, "a", encoding="utf-8") as fp:
                fp.write(json.dumps(record, sort_keys=True) + "\n")
        except Exception as exc:
            print(f"[ActorInputChecksumDebug] failed phase={phase}: {exc}", flush=True)

    def _dump_actor_chunk_loss_debug(
        self,
        *,
        phase: str,
        global_step=None,
        batch_idx=None,
        microbatch_idx=None,
        chunk_idx=None,
        rows=None,
        real_rows=None,
        active_tokens=None,
        loss_scale_factor=None,
        unscaled_loss=None,
        generation_jsd_loss=None,
        scaled_loss=None,
    ) -> None:
        if not self._actor_chunk_loss_debug_enabled():
            return
        try:
            global_step_scalar = self._actor_param_checksum_scalar(global_step)
            record = {
                "rank": int(self._actor_param_checksum_rank()),
                "pid": int(os.getpid()),
                "phase": phase,
                "global_step": int(global_step_scalar) if global_step_scalar is not None else None,
                "active_tokens": self._actor_param_checksum_scalar(active_tokens),
                "loss_scale_factor": self._actor_param_checksum_scalar(loss_scale_factor),
                "unscaled_loss": self._actor_param_checksum_scalar(unscaled_loss),
                "generation_jsd_loss": self._actor_param_checksum_scalar(generation_jsd_loss),
                "scaled_loss": self._actor_param_checksum_scalar(scaled_loss),
            }
            optional_fields = {
                "batch_idx": batch_idx,
                "microbatch_idx": microbatch_idx,
                "chunk_idx": chunk_idx,
                "rows": rows,
                "real_rows": real_rows,
            }
            for key, value in optional_fields.items():
                if value is None:
                    continue
                scalar = self._actor_param_checksum_scalar(value)
                record[key] = int(scalar) if scalar is not None else None
            path = self._actor_chunk_loss_debug_path()
            with open(path, "a", encoding="utf-8") as fp:
                fp.write(json.dumps(record, sort_keys=True) + "\n")
        except Exception as exc:
            print(f"[ActorChunkLossDebug] failed phase={phase}: {exc}", flush=True)

    def _actor_param_checksum_global_step(self):
        for attr in ("_last_update_policy_global_step", "global_step", "_global_step"):
            value = getattr(self, attr, None)
            if value is None:
                continue
            if torch.is_tensor(value):
                if value.numel() <= 0:
                    continue
                value = value.detach().reshape(-1)[0].cpu().item()
            try:
                return int(value)
            except (TypeError, ValueError):
                return value
        return None

    def _dump_actor_param_checksum_debug(
        self,
        *,
        phase: str,
        grad_scale_factor: float,
        stream_trace_all_ce_denominator: float,
        stream_trace_all_ce_scale: float,
        stream_trace_all_ce_pending: bool,
        grad_norm=None,
        include_grad: bool = False,
    ) -> None:
        if not self._actor_param_checksum_debug_enabled():
            return
        try:
            limit = int(os.environ.get("VERL_ACTOR_PARAM_CHECKSUM_LIMIT", "8"))
        except ValueError:
            limit = 8
        limit = max(limit, 0)
        try:
            rank = self._actor_param_checksum_rank()
            params = []
            with torch.no_grad():
                for idx, (name, param) in enumerate(self.actor_module.named_parameters()):
                    if idx >= limit:
                        break
                    row = {
                        "index": int(idx),
                        "name": name,
                        "dtype": str(param.dtype),
                        "shape": [int(dim) for dim in param.shape],
                        "param": self._actor_param_checksum_tensor_stats(param),
                    }
                    if include_grad:
                        row["grad"] = self._actor_param_checksum_tensor_stats(param.grad)
                    params.append(row)
            record = {
                "rank": int(rank),
                "pid": int(os.getpid()),
                "global_step": self._actor_param_checksum_global_step(),
                "phase": phase,
                "grad_scale_factor": float(grad_scale_factor),
                "stream_trace_all_ce_denominator": float(stream_trace_all_ce_denominator),
                "stream_trace_all_ce_scale": float(stream_trace_all_ce_scale),
                "stream_trace_all_ce_pending": bool(stream_trace_all_ce_pending),
                "grad_norm": self._actor_param_checksum_scalar(grad_norm),
                "param_limit": int(limit),
                "params": params,
            }
            path = self._actor_param_checksum_debug_path()
            with open(path, "a", encoding="utf-8") as fp:
                fp.write(json.dumps(record, sort_keys=True) + "\n")
        except Exception as exc:
            print(f"[ActorParamChecksumDebug] failed phase={phase}: {exc}", flush=True)

    def _sc_train_mode_registry(self):
        """FlatParameter frozen-slice registry for VERL_SC_TRAIN_MODE
        freeze-by-restore.

        freeze_sc -> frozen slices = the self_conditioning.* params;
        only_sc   -> frozen slices = the COMPLEMENT (everything else).
        Uses the same _fqns/_shard_param_infos mapping (and the same global
        presence check) as _maybe_record_sc_param_delta below: the SC module
        can land entirely inside one rank's shard, so presence is all-reduced.
        Freezing is implemented as snapshot-before-step + restore-after-step on
        the local flat shards — exact (also undoes AdamW weight decay), needs
        no use_orig_params, and leaves the fast-hidden-CE lm_head path intact
        (use_orig_params reshards weights into 1-D views and broke it).
        """
        import os as _os
        mode = _os.environ.get("VERL_SC_TRAIN_MODE", "").strip()
        if not mode:
            return None, None
        if mode not in ("freeze_sc", "only_sc"):
            raise ValueError(f"unknown VERL_SC_TRAIN_MODE={mode!r}")
        registry = []
        local_sc = 0
        for pname, p in self.actor_module.named_parameters():
            fqns = getattr(p, "_fqns", None)
            infos = getattr(p, "_shard_param_infos", None)
            if fqns is None or infos is None:
                continue
            shard = getattr(p, "_local_shard", None)
            if shard is None or shard.numel() != p.data.numel():
                shard = p.data
            shard_numel = shard.numel()
            sc_ranges = []
            for fqn, spi in zip(fqns, infos):
                if not getattr(spi, "in_shard", False):
                    continue
                if "self_conditioning" in (str(pname) + "." + str(fqn)):
                    a = int(spi.offset_in_shard)
                    sc_ranges.append((a, a + int(spi.numel_in_shard)))
            local_sc += len(sc_ranges)
            if mode == "freeze_sc":
                frozen = sc_ranges
            else:
                frozen, cur = [], 0
                for a, b in sorted(sc_ranges):
                    if cur < a:
                        frozen.append((cur, a))
                    cur = max(cur, b)
                if cur < shard_numel:
                    frozen.append((cur, shard_numel))
            registry.append((p, frozen))
        if not registry:
            raise RuntimeError(
                f"VERL_SC_TRAIN_MODE={mode} but no FlatParameter with "
                "_fqns/_shard_param_infos found (unsupported FSDP flavor)"
            )
        global_sc = local_sc
        if torch.distributed.is_initialized():
            t = torch.tensor([local_sc], device=get_device_id(), dtype=torch.int64)
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
            global_sc = int(t.item())
        if global_sc == 0:
            raise RuntimeError(
                f"VERL_SC_TRAIN_MODE={mode} but no self_conditioning slices "
                "found in any rank's flat-param shard"
            )
        return mode, registry

    def _maybe_sc_train_mode_snapshot(self):
        """Snapshot the frozen slices once, right before the FIRST optimizer
        step (i.e. after checkpoint load) — the values restore() pins forever."""
        if getattr(self, "_sc_tm_state", None) is not None:
            return
        mode, registry = self._sc_train_mode_registry()
        if mode is None:
            self._sc_tm_state = False
            return
        snaps = []
        local_frozen = 0
        for p, ranges in registry:
            shard = getattr(p, "_local_shard", None)
            if shard is None or shard.numel() != p.data.numel():
                shard = p.data
            buf = [(a, b, shard[a:b].detach().clone()) for a, b in ranges]
            local_frozen += sum(b - a for a, b in ranges)
            snaps.append((p, buf))
        self._sc_tm_state = (mode, snaps)
        global_frozen = local_frozen
        if torch.distributed.is_initialized():
            t = torch.tensor([local_frozen], device=get_device_id(), dtype=torch.int64)
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
            global_frozen = int(t.item())
        rank0 = (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0
        if rank0:
            print(
                f"[SC_TRAIN_MODE] mode={mode} frozen_numel_global={global_frozen} "
                f"(freeze-by-restore, snapshot taken pre-first-step)",
                flush=True,
            )

    def _maybe_sc_train_mode_restore(self):
        """Copy the snapshot back into the frozen slices after every applied
        optimizer step — exact freeze incl. undoing decoupled weight decay."""
        st = getattr(self, "_sc_tm_state", None)
        if not st:
            return
        _mode, snaps = st
        with torch.no_grad():
            for p, buf in snaps:
                shard = getattr(p, "_local_shard", None)
                if shard is None or shard.numel() != p.data.numel():
                    shard = p.data
                for a, b, w in buf:
                    shard[a:b].copy_(w)

    def _maybe_record_sc_param_delta(self):
        """Per-step parameter-delta telemetry for the self-conditioning
        (residual) adapter vs the rest of the network.

        Env-gated: VERL_ACTOR_SC_DELTA_METRICS=1 (default off = byte-equal).
        Works under flat-param FSDP where named_parameters() has no per-name
        tensors (the SC_LR_MULT trap): FlatParameter carries _fqns +
        _shard_param_infos, which locate every original param's slice inside
        the LOCAL shard. L2 sums over disjoint shards are all-reduced across
        the world (fsdp_size == world) so the reported norms are global.
        Ratio semantics: rel = ||dTheta||/||Theta|| per group; the sc/nonsc
        ratio is the "angular velocity" imbalance the alpha=1/8 reparam is
        meant to damp (~12x measured pre-reparam per the SC_LR_MULT note).
        """
        if not getattr(self, "_sc_delta_enabled", None):
            import os as _os
            self._sc_delta_enabled = _os.environ.get(
                "VERL_ACTOR_SC_DELTA_METRICS", "0"
            ).strip() == "1"
            self._sc_delta_registry = None
            self._sc_delta_prev = None
            if not self._sc_delta_enabled:
                return
        if not self._sc_delta_enabled:
            return
        if self._sc_delta_registry is None:
            registry = []
            for pname, p in self.actor_module.named_parameters():
                fqns = getattr(p, "_fqns", None)
                infos = getattr(p, "_shard_param_infos", None)
                if fqns is None or infos is None:
                    continue
                # _fqns are RELATIVE to the owning FSDP unit; if the SC module
                # is auto-wrapped as its own unit they carry no parent prefix,
                # so match on the flat-param's own dotted path + relative fqn.
                sc_ranges = []
                for fqn, spi in zip(fqns, infos):
                    if not getattr(spi, "in_shard", False):
                        continue
                    if "self_conditioning" in (str(pname) + "." + str(fqn)):
                        a = int(spi.offset_in_shard)
                        sc_ranges.append((a, a + int(spi.numel_in_shard)))
                registry.append((p, sc_ranges))
            if not registry:
                raise RuntimeError(
                    "VERL_ACTOR_SC_DELTA_METRICS=1 but no FlatParameter with "
                    "_fqns/_shard_param_infos found (unsupported FSDP flavor)"
                )
            # The SC adapter is a single model-level module: its flat-param
            # slice can land entirely inside ONE rank's shard, so presence
            # must be checked GLOBALLY (a per-rank check false-raises on the
            # ranks whose shard holds none of it).
            local_sc = sum(len(r) for _, r in registry)
            global_sc = local_sc
            if torch.distributed.is_initialized():
                t = torch.tensor([local_sc], device=get_device_id(), dtype=torch.int64)
                torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
                global_sc = int(t.item())
            if global_sc == 0:
                raise RuntimeError(
                    "VERL_ACTOR_SC_DELTA_METRICS=1 but no self_conditioning "
                    "slices found in any rank's flat-param shard"
                )
            self._sc_delta_registry = registry
            self._sc_delta_prev = [None] * len(registry)
        sc_d2 = nonsc_d2 = sc_n2 = nonsc_n2 = 0.0
        first_pass = self._sc_delta_prev[0] is None
        for k, (p, sc_ranges) in enumerate(self._sc_delta_registry):
            shard = getattr(p, "_local_shard", None)
            if shard is None or shard.numel() != p.data.numel():
                shard = p.data
            cur = shard.detach().to("cpu", dtype=torch.float32, copy=True)
            total_n2 = float(cur.square().sum())
            s_n2 = sum(float(cur[a:b].square().sum()) for a, b in sc_ranges)
            sc_n2 += s_n2
            nonsc_n2 += total_n2 - s_n2
            prev = self._sc_delta_prev[k]
            if prev is not None:
                d = cur - prev
                total_d2 = float(d.square().sum())
                s_d2 = sum(float(d[a:b].square().sum()) for a, b in sc_ranges)
                sc_d2 += s_d2
                nonsc_d2 += total_d2 - s_d2
            self._sc_delta_prev[k] = cur
        if torch.distributed.is_initialized():
            buf = torch.tensor(
                [sc_d2, nonsc_d2, sc_n2, nonsc_n2],
                device=get_device_id(), dtype=torch.float64,
            )
            torch.distributed.all_reduce(buf, op=torch.distributed.ReduceOp.SUM)
            sc_d2, nonsc_d2, sc_n2, nonsc_n2 = (float(x) for x in buf.tolist())
        if first_pass:
            if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
                print(
                    f"[SC_DELTA_METRICS] APPLIED sc_norm={sc_n2 ** 0.5:.4e} "
                    f"nonsc_norm={nonsc_n2 ** 0.5:.4e} (first pass, deltas start next step)",
                    flush=True,
                )
            self._last_sc_delta_metrics = {}
            return
        sc_rel = (sc_d2 ** 0.5) / max(sc_n2 ** 0.5, 1e-12)
        nonsc_rel = (nonsc_d2 ** 0.5) / max(nonsc_n2 ** 0.5, 1e-12)
        self._last_sc_delta_metrics = {
            "actor/sc_param_delta_l2": sc_d2 ** 0.5,
            "actor/nonsc_param_delta_l2": nonsc_d2 ** 0.5,
            "actor/sc_param_delta_rel": sc_rel,
            "actor/nonsc_param_delta_rel": nonsc_rel,
            "actor/sc_vs_nonsc_step_ratio": sc_rel / max(nonsc_rel, 1e-12),
        }

    def _optimizer_step(
        self,
        grad_scale_factor: float | None = None,
        *,
        force_stream_trace_all_ce_sync: bool = False,
    ):
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        grad_scale_factor = 1.0 if grad_scale_factor is None else float(grad_scale_factor)
        if grad_scale_factor < 0.0:
            raise ValueError(f"grad_scale_factor must be non-negative, got {grad_scale_factor}")
        stream_trace_all_ce_pending = bool(
            getattr(self, "_stream_trace_all_ce_grad_scale_pending", False)
            or force_stream_trace_all_ce_sync
        )
        stream_trace_all_ce_denominator = 0.0
        stream_trace_all_ce_scale = 1.0
        if stream_trace_all_ce_pending:
            local_denominator = float(
                getattr(self, "_stream_trace_all_ce_grad_weighted_denominator", 0.0) or 0.0
            )
            stream_trace_all_ce_denominator = local_denominator
            world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
            if torch.distributed.is_initialized():
                denominator_tensor = torch.tensor(
                    [local_denominator],
                    device=get_device_id(),
                    dtype=torch.float64,
                )
                torch.distributed.all_reduce(denominator_tensor, op=torch.distributed.ReduceOp.SUM)
                stream_trace_all_ce_denominator = float(denominator_tensor.item())
            fixed_batch_denominator = float(
                (self.config.get("generation_consistency_jsd", {}) or {}).get(
                    "fixed_batch_denominator", 0
                )
                or 0
            )
            if fixed_batch_denominator > 0.0:
                # Clamp from below by the pre-filter response count so the
                # update magnitude scales with keep_frac; the measured
                # denominator (= kept responses under response_level_ce) can
                # only be <= the pre-filter count, so max() never amplifies.
                stream_trace_all_ce_denominator = max(
                    stream_trace_all_ce_denominator, fixed_batch_denominator
                )
            if stream_trace_all_ce_denominator > 0.0:
                stream_trace_all_ce_scale = float(world_size) / stream_trace_all_ce_denominator
            grad_scale_factor *= stream_trace_all_ce_scale
        self._last_stream_trace_all_ce_grad_scale_pending = 1.0 if stream_trace_all_ce_pending else 0.0
        self._last_stream_trace_all_ce_grad_scale_factor = stream_trace_all_ce_scale
        self._last_stream_trace_all_ce_grad_scale_denominator = stream_trace_all_ce_denominator
        self._last_grad_scale_factor = grad_scale_factor
        if grad_scale_factor != 1.0:
            for param in self.actor_module.parameters():
                grad = param.grad
                if grad is not None:
                    grad.data.mul_(grad_scale_factor)
        nonfinite_grad_tensors = 0
        nonfinite_grad_values = 0
        total_grad_values = 0
        if self.config.get("sanitize_nonfinite_grads", False):
            for param in self.actor_module.parameters():
                grad = param.grad
                if grad is None:
                    continue
                finite = torch.isfinite(grad)
                total_grad_values += grad.numel()
                if not finite.all():
                    nonfinite_grad_tensors += 1
                    nonfinite_grad_values += int((~finite).sum().detach().cpu().item())
                    grad.data = torch.nan_to_num(grad.data, nan=0.0, posinf=0.0, neginf=0.0)
        self._last_nonfinite_grad_tensors = nonfinite_grad_tensors
        self._last_nonfinite_grad_values = nonfinite_grad_values
        self._last_total_grad_values = total_grad_values
        self._dump_actor_param_checksum_debug(
            phase="pre_step",
            grad_scale_factor=grad_scale_factor,
            stream_trace_all_ce_denominator=stream_trace_all_ce_denominator,
            stream_trace_all_ce_scale=stream_trace_all_ce_scale,
            stream_trace_all_ce_pending=stream_trace_all_ce_pending,
            grad_norm=None,
            include_grad=True,
        )
        self._sync_replica_grads_once()
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        if isinstance(self.actor_module, FSDP) and world_size > 1:
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # Snapshot the self-conditioning params (and a backbone sample) before
        # step(); ||dW||/||W|| is computed after the step (telemetry only).
        _sc_pre_norms = _sc_snapshot_norms(self)

        # if grad_norm is not finite, skip the update
        self._maybe_sc_train_mode_snapshot()
        optimizer_step_applied = False
        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
            optimizer_step_applied = True
        else:
            if not torch.isfinite(grad_norm):
                print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
                self.actor_optimizer.zero_grad()
            else:
                self.actor_optimizer.step()
                optimizer_step_applied = True
        self._last_optimizer_step_applied = optimizer_step_applied
        if optimizer_step_applied:
            self._maybe_sc_train_mode_restore()
            self._lifecycle_optimizer_step_count += 1
            _sc_record_update_magnitude(self, _sc_pre_norms)
            self._maybe_record_sc_param_delta()
        emit_lifecycle(
            "DataParallelPPOActor",
            "optimizer_step_done" if optimizer_step_applied else "optimizer_step_skipped",
            rank=int(torch.distributed.get_rank()) if torch.distributed.is_initialized() else 0,
            global_step=getattr(self, "_last_update_policy_global_step", None),
            optimizer_step_count=self._lifecycle_optimizer_step_count,
            applied=optimizer_step_applied,
            grad_norm=float(grad_norm.detach().cpu().item()),
            finite_grad_norm=bool(torch.isfinite(grad_norm).item()),
            memory=lifecycle_memory("after_optimizer_step"),
        )

        # Clear cached weight scales for QAT (weights changed)
        if getattr(self.actor_module, "_qat_fuse_enabled", False):
            from verl.utils.qat import invalidate_all_scales

            invalidate_all_scales(self.actor_module)

        self._dump_actor_param_checksum_debug(
            phase="post_step",
            grad_scale_factor=grad_scale_factor,
            stream_trace_all_ce_denominator=stream_trace_all_ce_denominator,
            stream_trace_all_ce_scale=stream_trace_all_ce_scale,
            stream_trace_all_ce_pending=stream_trace_all_ce_pending,
            grad_norm=grad_norm,
            include_grad=False,
        )

        if stream_trace_all_ce_pending:
            self._reset_stream_trace_all_ce_grad_scale()

        return grad_norm

    @staticmethod
    def _local_grad_tensor_for_signature(grad: torch.Tensor) -> torch.Tensor:
        if isinstance(grad, DTensor):
            grad = grad.to_local()
        if getattr(grad, "is_sparse", False):
            grad = grad.coalesce().values()
        return grad

    @staticmethod
    def _t5_tensor_grad_rescale_basis(meta_info: dict) -> str:
        basis = str(meta_info.get("_t5_tensor_grad_rescale_basis", "active_tokens") or "active_tokens").strip().lower()
        if basis in {"row", "rows"}:
            return "rows"
        if basis in {"active", "active_token", "active_tokens", "token", "tokens"}:
            return "active_tokens"
        raise ValueError(
            "_t5_tensor_grad_rescale_basis must be rows or active_tokens, "
            f"got {meta_info.get('_t5_tensor_grad_rescale_basis')!r}"
        )

    @staticmethod
    def _t5_tensor_debug_finish_grad_rescale(
        *,
        rows: int,
        active_tokens: float,
        meta_info: dict,
    ) -> tuple[float, float, float, float, float, float]:
        unknown_total_grad_rescale = bool(meta_info.get("_t5_tensor_unknown_total_grad_rescale", False))
        grad_rescale_denominator = 0.0
        grad_rescale_factor = 1.0
        grad_rescale_basis_active_tokens = 0.0
        grad_rescale_basis_rows = 0.0
        provisional_denominator = float(meta_info.get("_t5_tensor_provisional_denominator", 0.0) or 0.0)
        use_provisional_denominator = unknown_total_grad_rescale and provisional_denominator > 0.0
        if unknown_total_grad_rescale:
            basis = DataParallelPPOActor._t5_tensor_grad_rescale_basis(meta_info)
            if basis == "rows":
                grad_rescale_denominator = float(max(int(rows), 1))
                grad_rescale_basis_rows = 1.0
            elif active_tokens > 0:
                grad_rescale_denominator = float(active_tokens)
                grad_rescale_basis_active_tokens = 1.0
            else:
                grad_rescale_denominator = float(max(int(rows), 1))
                grad_rescale_basis_rows = 1.0
            if use_provisional_denominator:
                grad_rescale_factor = provisional_denominator / max(grad_rescale_denominator, 1.0)
            else:
                grad_rescale_factor = 1.0 / max(grad_rescale_denominator, 1.0)
        return (
            float(grad_rescale_factor),
            float(grad_rescale_denominator),
            float(provisional_denominator),
            1.0 if use_provisional_denominator else 0.0,
            float(grad_rescale_basis_active_tokens),
            float(grad_rescale_basis_rows),
        )

    @staticmethod
    def _prepare_t5_tensor_grad_parity_meta(meta_info: dict, total_rows: int) -> dict:
        parity_meta_info = dict(meta_info or {})
        expected_rows = int(parity_meta_info.get("_t5_tensor_total_rows", total_rows) or total_rows)
        expected_active_tokens = float(parity_meta_info.get("_t5_tensor_total_active_tokens", 0.0) or 0.0)
        normalize_at_finish = bool(parity_meta_info.get("_t5_tensor_normalize_at_finish", False))
        unknown_total_grad_rescale = normalize_at_finish or (
            expected_rows <= 0 and expected_active_tokens <= 0
        )
        provisional_denominator = float(parity_meta_info.get("_t5_tensor_provisional_denominator", 0.0) or 0.0)
        use_provisional_denominator = unknown_total_grad_rescale and provisional_denominator > 0.0
        if not use_provisional_denominator:
            provisional_denominator = 0.0
        parity_meta_info["_t5_tensor_total_rows"] = int(expected_rows)
        parity_meta_info["_t5_tensor_total_active_tokens"] = float(expected_active_tokens)
        parity_meta_info["_t5_tensor_unknown_total_grad_rescale"] = bool(unknown_total_grad_rescale)
        parity_meta_info["_t5_tensor_provisional_denominator"] = float(provisional_denominator)
        parity_meta_info["_t5_tensor_grad_rescale_basis"] = DataParallelPPOActor._t5_tensor_grad_rescale_basis(
            parity_meta_info
        )
        return parity_meta_info

    @staticmethod
    def _expand_sparse_rollout_logits_source_hidden(
        model_inputs: dict[str, torch.Tensor],
        *,
        expand_dense: bool = True,
    ) -> dict[str, float]:
        metrics = {
            "actor/t5_tensor_replay/sparse_logits_source_hidden_enabled": 0.0,
            "actor/t5_tensor_replay/sparse_logits_source_hidden_values": 0.0,
            "actor/t5_tensor_replay/sparse_logits_source_hidden_dense_bytes": 0.0,
            "actor/t5_tensor_replay/sparse_logits_source_hidden_sparse_bytes": 0.0,
            "actor/t5_tensor_replay/sparse_logits_source_hidden_expanded_dense": 0.0,
            "actor/t5_tensor_replay/sparse_logits_source_hidden_direct_pack": 0.0,
            "actor/t5_tensor_replay/sparse_logits_source_hidden_expand_seconds": 0.0,
        }
        sparse_keys = (
            "rollout_logits_source_hidden_sparse_values",
            "rollout_logits_source_hidden_sparse_rows",
            "rollout_logits_source_hidden_sparse_positions",
            "rollout_logits_source_hidden_sparse_shape",
        )
        present = [key for key in sparse_keys if key in model_inputs]
        if not present:
            return metrics
        if len(present) != len(sparse_keys):
            raise RuntimeError(
                "incomplete sparse rollout_logits_source_hidden payload: "
                f"present={present} expected={list(sparse_keys)}"
            )
        if "rollout_logits_source_hidden" in model_inputs:
            raise RuntimeError("received both dense and sparse rollout_logits_source_hidden payloads")

        expand_start = time.perf_counter()
        if expand_dense:
            values = model_inputs.pop("rollout_logits_source_hidden_sparse_values")
            rows = model_inputs.pop("rollout_logits_source_hidden_sparse_rows").long().view(-1)
            positions = model_inputs.pop("rollout_logits_source_hidden_sparse_positions").long().view(-1)
            shape = model_inputs.pop("rollout_logits_source_hidden_sparse_shape").detach().cpu().long().view(-1)
        else:
            values = model_inputs["rollout_logits_source_hidden_sparse_values"]
            rows = model_inputs["rollout_logits_source_hidden_sparse_rows"].long().view(-1)
            positions = model_inputs["rollout_logits_source_hidden_sparse_positions"].long().view(-1)
            shape = model_inputs["rollout_logits_source_hidden_sparse_shape"].detach().cpu().long().view(-1)
        if int(shape.numel()) != 3:
            raise RuntimeError(
                "rollout_logits_source_hidden_sparse_shape must have 3 elements, "
                f"got shape={tuple(shape.shape)}"
            )
        row_count, response_len, hidden_size = (int(shape[i].item()) for i in range(3))
        input_rows = int(model_inputs["input_ids"].shape[0])
        responses = model_inputs.get("responses")
        input_response_len = int(responses.shape[-1]) if torch.is_tensor(responses) else response_len
        if row_count != input_rows or response_len != input_response_len:
            raise RuntimeError(
                "sparse rollout_logits_source_hidden shape does not match tensor chunk: "
                f"sparse_shape={(row_count, response_len, hidden_size)} "
                f"input_rows={input_rows} response_len={input_response_len}"
            )
        if values.dim() != 2 or int(values.shape[-1]) != hidden_size:
            raise RuntimeError(
                "rollout_logits_source_hidden_sparse_values must be [values, hidden], "
                f"got values_shape={tuple(values.shape)} hidden_size={hidden_size}"
            )
        value_count = int(values.shape[0])
        if int(rows.numel()) != value_count or int(positions.numel()) != value_count:
            raise RuntimeError(
                "sparse rollout_logits_source_hidden index count mismatch: "
                f"values={tuple(values.shape)} rows={tuple(rows.shape)} positions={tuple(positions.shape)}"
            )
        if value_count > 0:
            if bool(rows.lt(0).any() or rows.ge(row_count).any() or positions.lt(0).any() or positions.ge(response_len).any()):
                raise RuntimeError(
                    "sparse rollout_logits_source_hidden indices out of range: "
                    f"sparse_shape={(row_count, response_len, hidden_size)}"
                )
            linear = rows * int(response_len) + positions
            if int(torch.unique(linear).numel()) != value_count:
                raise RuntimeError("sparse rollout_logits_source_hidden contains duplicate row/position entries")
        dense_bytes = row_count * response_len * hidden_size * int(values.element_size())
        sparse_bytes = (
            int(values.numel()) * int(values.element_size())
            + int(rows.numel()) * int(rows.element_size())
            + int(positions.numel()) * int(positions.element_size())
            + int(shape.numel()) * int(shape.element_size())
        )
        if expand_dense:
            dense = torch.zeros(
                (row_count, response_len, hidden_size),
                device=values.device,
                dtype=values.dtype,
            )
            if value_count > 0:
                dense[rows, positions] = values
            model_inputs["rollout_logits_source_hidden"] = dense
        metrics.update(
            {
                "actor/t5_tensor_replay/sparse_logits_source_hidden_enabled": 1.0,
                "actor/t5_tensor_replay/sparse_logits_source_hidden_values": float(value_count),
                "actor/t5_tensor_replay/sparse_logits_source_hidden_dense_bytes": float(dense_bytes),
                "actor/t5_tensor_replay/sparse_logits_source_hidden_sparse_bytes": float(sparse_bytes),
                "actor/t5_tensor_replay/sparse_logits_source_hidden_expanded_dense": 1.0 if expand_dense else 0.0,
                "actor/t5_tensor_replay/sparse_logits_source_hidden_direct_pack": 0.0 if expand_dense else 1.0,
                "actor/t5_tensor_replay/sparse_logits_source_hidden_expand_seconds": (
                    time.perf_counter() - expand_start
                ),
            }
        )
        return metrics

    @staticmethod
    def _slice_t5_tensor_batch_rows(
        tensor_batch: dict[str, torch.Tensor],
        start: int,
        end: int,
    ) -> dict[str, torch.Tensor]:
        total_rows = int(tensor_batch["input_ids"].shape[0])
        start = max(0, min(int(start), total_rows))
        end = max(start, min(int(end), total_rows))
        sparse_keys = {
            "rollout_logits_source_hidden_sparse_values",
            "rollout_logits_source_hidden_sparse_rows",
            "rollout_logits_source_hidden_sparse_positions",
            "rollout_logits_source_hidden_sparse_shape",
        }
        chunk: dict[str, torch.Tensor] = {}
        for key, value in tensor_batch.items():
            if key in sparse_keys:
                continue
            if torch.is_tensor(value) and value.dim() > 0 and int(value.shape[0]) == total_rows:
                chunk[key] = value[start:end].contiguous()
            else:
                chunk[key] = value

        values = tensor_batch.get("rollout_logits_source_hidden_sparse_values")
        rows = tensor_batch.get("rollout_logits_source_hidden_sparse_rows")
        positions = tensor_batch.get("rollout_logits_source_hidden_sparse_positions")
        shape = tensor_batch.get("rollout_logits_source_hidden_sparse_shape")
        present = [torch.is_tensor(x) for x in (values, rows, positions, shape)]
        if any(present):
            if not all(present):
                raise RuntimeError("incomplete sparse rollout_logits_source_hidden payload")
            shape_cpu = shape.detach().cpu().long().view(-1).clone()
            if int(shape_cpu.numel()) != 3:
                raise RuntimeError(
                    "rollout_logits_source_hidden_sparse_shape must have 3 elements, "
                    f"got shape={tuple(shape.shape)}"
                )
            if int(shape_cpu[0].item()) != total_rows:
                raise RuntimeError(
                    "sparse rollout_logits_source_hidden row count mismatch while slicing: "
                    f"sparse_rows={int(shape_cpu[0].item())} total_rows={total_rows}"
                )
            rows_flat = rows.detach().cpu().long().view(-1)
            row_mask = rows_flat.ge(start) & rows_flat.lt(end)
            selected = torch.nonzero(row_mask, as_tuple=False).flatten().to(device=rows.device)
            chunk["rollout_logits_source_hidden_sparse_values"] = values.index_select(0, selected).contiguous()
            chunk["rollout_logits_source_hidden_sparse_rows"] = (
                rows.index_select(0, selected).long() - int(start)
            ).contiguous()
            chunk["rollout_logits_source_hidden_sparse_positions"] = positions.index_select(0, selected).contiguous()
            shape_cpu[0] = int(end - start)
            chunk["rollout_logits_source_hidden_sparse_shape"] = shape_cpu
        return chunk

    def _count_t5_tensor_grad_parity_active_tokens(self, tensor_batch: dict[str, torch.Tensor]) -> float:
        responses = tensor_batch["responses"]
        attention_response_mask = tensor_batch["attention_mask"][:, -responses.shape[-1]:]
        active = attention_response_mask.bool() & responses.ge(0)
        generation_jsd_config = self.config.get("generation_consistency_jsd", {})
        requires_source_hidden = (
            bool(generation_jsd_config.get("enable", False))
            and generation_jsd_config.get("objective", "full_jsd") == "trace_future_ce"
            and not bool(generation_jsd_config.get("trace_future_hard_label_ce", False))
        )
        source_hidden = tensor_batch.get("rollout_logits_source_hidden")
        if requires_source_hidden and source_hidden is not None:
            active = active & source_hidden.detach().float().abs().sum(dim=-1).gt(0)
        elif requires_source_hidden and all(
            key in tensor_batch
            for key in (
                "rollout_logits_source_hidden_sparse_values",
                "rollout_logits_source_hidden_sparse_rows",
                "rollout_logits_source_hidden_sparse_positions",
            )
        ):
            rows = tensor_batch["rollout_logits_source_hidden_sparse_rows"].detach().cpu().long().view(-1)
            positions = tensor_batch["rollout_logits_source_hidden_sparse_positions"].detach().cpu().long().view(-1)
            sparse_active = torch.zeros_like(active, dtype=torch.bool, device="cpu")
            if int(rows.numel()) > 0:
                valid = rows.ge(0) & rows.lt(active.shape[0]) & positions.ge(0) & positions.lt(active.shape[1])
                if bool(valid.any()):
                    sparse_active[rows[valid], positions[valid]] = True
            active = active & sparse_active.to(device=active.device)
        quality_mask = tensor_batch.get("rollout_quality_mask")
        if quality_mask is not None:
            active = active & quality_mask.bool()
        return float(active.sum().detach().cpu().item())

    def _scale_actor_grads_for_debug(self, scale: float) -> None:
        scale = float(scale)
        if scale == 1.0:
            return
        for param in self.actor_module.parameters():
            grad = param.grad
            if grad is not None:
                grad.data.mul_(scale)

    def _sample_local_grad_signature(
        self,
        *,
        sample_params: int,
        sample_elems_per_param: int,
    ) -> dict:
        sample_params = max(int(sample_params), 0)
        sample_elems_per_param = max(int(sample_elems_per_param), 0)
        samples: dict[str, torch.Tensor] = {}
        grad_param_count = 0
        grad_elem_count = 0
        sampled_elem_count = 0
        nonfinite_values = 0
        norm_sq = 0.0
        for name, param in self.actor_module.named_parameters():
            grad = param.grad
            if grad is None:
                continue
            grad_tensor = self._local_grad_tensor_for_signature(grad.detach())
            if grad_tensor.numel() <= 0:
                continue
            grad_param_count += 1
            grad_elem_count += int(grad_tensor.numel())
            grad_flat = grad_tensor.float().reshape(-1)
            finite = torch.isfinite(grad_flat)
            if not bool(finite.all().detach().cpu().item()):
                nonfinite_values += int((~finite).sum().detach().cpu().item())
            grad_for_norm = torch.nan_to_num(grad_flat, nan=0.0, posinf=0.0, neginf=0.0)
            norm_sq += float(torch.sum(grad_for_norm * grad_for_norm).detach().cpu().item())
            if len(samples) >= sample_params or sample_elems_per_param <= 0:
                continue
            sample_count = min(int(sample_elems_per_param), int(grad_flat.numel()))
            if sample_count <= 0:
                continue
            if sample_count == int(grad_flat.numel()):
                sample = grad_flat
            elif sample_count == 1:
                sample = grad_flat[:1]
            else:
                indices = torch.arange(sample_count, device=grad_flat.device, dtype=torch.long)
                indices = indices * (int(grad_flat.numel()) - 1) // (sample_count - 1)
                sample = grad_flat.index_select(0, indices)
            samples[name] = sample.detach().cpu().clone()
            sampled_elem_count += int(sample.numel())
        return {
            "samples": samples,
            "grad_param_count": float(grad_param_count),
            "grad_elem_count": float(grad_elem_count),
            "sampled_param_count": float(len(samples)),
            "sampled_elem_count": float(sampled_elem_count),
            "grad_norm": float(norm_sq ** 0.5),
            "nonfinite_values": float(nonfinite_values),
        }

    def debug_t5_tensor_grad_parity(
        self,
        tensor_batch: dict[str, torch.Tensor],
        meta_info: dict | None = None,
        chunk_size: int = 0,
        max_rows: int = 16,
        sample_params: int = 64,
        sample_elems_per_param: int = 256,
        atol: float = 1.0e-4,
        rtol: float = 2.0e-3,
        raise_on_fail: bool = True,
    ) -> dict[str, float]:
        """Debug-only compact tensor full-vs-chunk gradient parity check."""
        if self.actor_optimizer is None:
            raise RuntimeError("debug_t5_tensor_grad_parity requires an actor optimizer")
        if self.scaler is not None:
            raise NotImplementedError("debug_t5_tensor_grad_parity does not support fp16 grad scaler")

        generation_jsd_config = self.config.get("generation_consistency_jsd", {})
        selected_vocab_size = int(generation_jsd_config.get("selected_vocab_size", 0) or 0)
        if selected_vocab_size > 0:
            raise NotImplementedError(
                "debug_t5_tensor_grad_parity does not support generation_consistency_jsd.selected_vocab_size > 0"
            )

        original_rows = int(tensor_batch["input_ids"].shape[0])
        if original_rows <= 0:
            raise RuntimeError("debug_t5_tensor_grad_parity requires at least one tensor row")
        max_rows = int(max_rows or 0)
        total_rows = original_rows
        truncated = bool(max_rows > 0 and original_rows > max_rows)
        if truncated:
            total_rows = max_rows
            tensor_batch = self._slice_t5_tensor_batch_rows(tensor_batch, 0, total_rows)
        chunk_size = int(chunk_size or 0)
        if chunk_size <= 0:
            chunk_size = total_rows
        chunk_size = max(1, min(chunk_size, total_rows))
        sample_params = int(sample_params)
        sample_elems_per_param = int(sample_elems_per_param)
        atol = float(atol)
        rtol = float(rtol)

        parity_meta_source = dict(meta_info or {})
        parity_meta_source["_t5_tensor_total_rows"] = int(total_rows)
        if truncated:
            parity_meta_source["_t5_tensor_total_active_tokens"] = (
                self._count_t5_tensor_grad_parity_active_tokens(tensor_batch)
            )
            provisional_denominator = float(
                parity_meta_source.get("_t5_tensor_provisional_denominator", 0.0) or 0.0
            )
            response_length = int(tensor_batch["responses"].shape[-1])
            if provisional_denominator > 0.0 and response_length > 0:
                parity_meta_source["_t5_tensor_provisional_denominator"] = float(total_rows * response_length)
        parity_meta_info = self._prepare_t5_tensor_grad_parity_meta(parity_meta_source, total_rows)
        was_training = bool(self.actor_module.training)

        def _chunk_tensor_batch(start: int, end: int) -> dict[str, torch.Tensor]:
            return self._slice_t5_tensor_batch_rows(tensor_batch, start, end)

        try:
            self.actor_optimizer.zero_grad()

            full_metrics = self.update_t5_tensor_chunk(
                tensor_batch=tensor_batch,
                meta_info={**parity_meta_info, "_t5_tensor_chunk_idx": 0},
            )
            full_active_tokens = float(full_metrics.get("actor/t5_tensor_replay/active_tokens", 0.0) or 0.0)
            full_loss = float(full_metrics.get("actor/t5_tensor_replay/loss", 0.0) or 0.0)
            (
                full_grad_rescale_factor,
                full_grad_rescale_denominator,
                full_provisional_denominator,
                full_use_provisional_denominator,
                full_grad_rescale_basis_active_tokens,
                full_grad_rescale_basis_rows,
            ) = self._t5_tensor_debug_finish_grad_rescale(
                rows=total_rows,
                active_tokens=full_active_tokens,
                meta_info=parity_meta_info,
            )
            self._scale_actor_grads_for_debug(full_grad_rescale_factor)
            full_signature = self._sample_local_grad_signature(
                sample_params=sample_params,
                sample_elems_per_param=sample_elems_per_param,
            )

            self.actor_optimizer.zero_grad()

            chunk_active_tokens = 0.0
            chunk_loss_weighted_sum = 0.0
            chunk_loss_sum = 0.0
            chunk_rows = 0
            chunks = 0
            for start in range(0, total_rows, chunk_size):
                end = min(start + chunk_size, total_rows)
                chunk_metrics = self.update_t5_tensor_chunk(
                    tensor_batch=_chunk_tensor_batch(start, end),
                    meta_info={**parity_meta_info, "_t5_tensor_chunk_idx": chunks},
                )
                rows = end - start
                active_tokens = float(chunk_metrics.get("actor/t5_tensor_replay/active_tokens", 0.0) or 0.0)
                loss_value = float(chunk_metrics.get("actor/t5_tensor_replay/loss", 0.0) or 0.0)
                chunk_rows += rows
                chunk_active_tokens += active_tokens
                chunk_loss_sum += loss_value
                chunk_loss_weighted_sum += loss_value * active_tokens
                chunks += 1

            if chunk_active_tokens > 0:
                chunk_loss = chunk_loss_weighted_sum / chunk_active_tokens
            else:
                chunk_loss = chunk_loss_sum / max(chunks, 1)
            (
                chunk_grad_rescale_factor,
                chunk_grad_rescale_denominator,
                chunk_provisional_denominator,
                chunk_use_provisional_denominator,
                chunk_grad_rescale_basis_active_tokens,
                chunk_grad_rescale_basis_rows,
            ) = self._t5_tensor_debug_finish_grad_rescale(
                rows=chunk_rows,
                active_tokens=chunk_active_tokens,
                meta_info=parity_meta_info,
            )
            self._scale_actor_grads_for_debug(chunk_grad_rescale_factor)
            chunk_signature = self._sample_local_grad_signature(
                sample_params=sample_params,
                sample_elems_per_param=sample_elems_per_param,
            )

            full_samples = full_signature["samples"]
            chunk_samples = chunk_signature["samples"]
            sample_names = list(full_samples.keys())
            missing_chunk_samples = sorted(set(sample_names) - set(chunk_samples.keys()))
            extra_chunk_samples = sorted(set(chunk_samples.keys()) - set(sample_names))
            common_sample_names = [name for name in sample_names if name in chunk_samples]
            if common_sample_names:
                full_vec = torch.cat([full_samples[name].reshape(-1) for name in common_sample_names], dim=0)
                chunk_vec = torch.cat([chunk_samples[name].reshape(-1) for name in common_sample_names], dim=0)
                diff = (full_vec - chunk_vec).abs()
                max_abs = float(diff.max().item()) if diff.numel() > 0 else 0.0
                mean_abs = float(diff.mean().item()) if diff.numel() > 0 else 0.0
                diff_l2 = float(torch.linalg.vector_norm(full_vec - chunk_vec).item())
                full_l2 = float(torch.linalg.vector_norm(full_vec).item())
                rel_l2 = diff_l2 / max(full_l2, 1.0e-12)
                sampled_allclose = bool(torch.allclose(full_vec, chunk_vec, atol=atol, rtol=rtol))
                full_sample_max_abs = float(full_vec.abs().max().item()) if full_vec.numel() > 0 else 0.0
                chunk_sample_max_abs = float(chunk_vec.abs().max().item()) if chunk_vec.numel() > 0 else 0.0
            else:
                max_abs = 0.0
                mean_abs = 0.0
                rel_l2 = 0.0
                sampled_allclose = False
                full_sample_max_abs = 0.0
                chunk_sample_max_abs = 0.0

            full_grad_norm = float(full_signature["grad_norm"])
            chunk_grad_norm = float(chunk_signature["grad_norm"])
            grad_norm_abs_diff = abs(full_grad_norm - chunk_grad_norm)
            grad_norm_rel_diff = grad_norm_abs_diff / max(abs(full_grad_norm), 1.0e-12)
            grad_norm_ok = grad_norm_abs_diff <= atol + rtol * abs(full_grad_norm)
            loss_abs_diff = abs(full_loss - chunk_loss)
            loss_ok = loss_abs_diff <= atol + rtol * abs(full_loss)
            active_token_diff = abs(float(full_active_tokens) - float(chunk_active_tokens))
            active_ok = active_token_diff == 0.0
            nonfinite_values = float(full_signature["nonfinite_values"]) + float(chunk_signature["nonfinite_values"])

            fail_reasons = []
            if missing_chunk_samples or extra_chunk_samples:
                fail_reasons.append(
                    "sampled parameter mismatch "
                    f"missing={missing_chunk_samples[:4]} extra={extra_chunk_samples[:4]}"
                )
            if float(full_signature["sampled_elem_count"]) <= 0 or float(chunk_signature["sampled_elem_count"]) <= 0:
                fail_reasons.append("no sampled gradient elements")
            if nonfinite_values > 0:
                fail_reasons.append(f"nonfinite sampled/local grad values={nonfinite_values:.0f}")
            if not sampled_allclose:
                fail_reasons.append(
                    f"sampled grads differ max_abs={max_abs:.6g} mean_abs={mean_abs:.6g} rel_l2={rel_l2:.6g}"
                )
            if not grad_norm_ok:
                fail_reasons.append(
                    f"grad norm differs full={full_grad_norm:.6g} chunk={chunk_grad_norm:.6g} "
                    f"rel_diff={grad_norm_rel_diff:.6g}"
                )
            if not loss_ok:
                fail_reasons.append(
                    f"loss differs full={full_loss:.6g} chunk={chunk_loss:.6g} abs_diff={loss_abs_diff:.6g}"
                )
            if not active_ok:
                fail_reasons.append(
                    f"active token count differs full={full_active_tokens:.0f} chunk={chunk_active_tokens:.0f}"
                )

            passed = len(fail_reasons) == 0
            metrics = {
                "actor/t5_tensor_grad_parity/passed": 1.0 if passed else 0.0,
                "actor/t5_tensor_grad_parity/chunks": float(chunks),
                "actor/t5_tensor_grad_parity/chunk_size": float(chunk_size),
                "actor/t5_tensor_grad_parity/original_rows": float(original_rows),
                "actor/t5_tensor_grad_parity/rows": float(total_rows),
                "actor/t5_tensor_grad_parity/truncated": 1.0 if truncated else 0.0,
                "actor/t5_tensor_grad_parity/sample_params": float(sample_params),
                "actor/t5_tensor_grad_parity/sample_elems_per_param": float(sample_elems_per_param),
                "actor/t5_tensor_grad_parity/sampled_params": float(full_signature["sampled_param_count"]),
                "actor/t5_tensor_grad_parity/sampled_elems": float(full_signature["sampled_elem_count"]),
                "actor/t5_tensor_grad_parity/atol": float(atol),
                "actor/t5_tensor_grad_parity/rtol": float(rtol),
                "actor/t5_tensor_grad_parity/max_abs": float(max_abs),
                "actor/t5_tensor_grad_parity/mean_abs": float(mean_abs),
                "actor/t5_tensor_grad_parity/rel_l2": float(rel_l2),
                "actor/t5_tensor_grad_parity/full_sample_max_abs": float(full_sample_max_abs),
                "actor/t5_tensor_grad_parity/chunk_sample_max_abs": float(chunk_sample_max_abs),
                "actor/t5_tensor_grad_parity/full_grad_norm": float(full_grad_norm),
                "actor/t5_tensor_grad_parity/chunk_grad_norm": float(chunk_grad_norm),
                "actor/t5_tensor_grad_parity/grad_norm_abs_diff": float(grad_norm_abs_diff),
                "actor/t5_tensor_grad_parity/grad_norm_rel_diff": float(grad_norm_rel_diff),
                "actor/t5_tensor_grad_parity/full_loss": float(full_loss),
                "actor/t5_tensor_grad_parity/chunk_loss": float(chunk_loss),
                "actor/t5_tensor_grad_parity/loss_abs_diff": float(loss_abs_diff),
                "actor/t5_tensor_grad_parity/full_active_tokens": float(full_active_tokens),
                "actor/t5_tensor_grad_parity/chunk_active_tokens": float(chunk_active_tokens),
                "actor/t5_tensor_grad_parity/active_token_diff": float(active_token_diff),
                "actor/t5_tensor_grad_parity/nonfinite_values": float(nonfinite_values),
                "actor/t5_tensor_grad_parity/full_finish_grad_rescale_factor": float(full_grad_rescale_factor),
                "actor/t5_tensor_grad_parity/chunk_finish_grad_rescale_factor": float(chunk_grad_rescale_factor),
                "actor/t5_tensor_grad_parity/full_finish_grad_rescale_denominator": float(
                    full_grad_rescale_denominator
                ),
                "actor/t5_tensor_grad_parity/chunk_finish_grad_rescale_denominator": float(
                    chunk_grad_rescale_denominator
                ),
                "actor/t5_tensor_grad_parity/full_provisional_denominator": float(full_provisional_denominator),
                "actor/t5_tensor_grad_parity/chunk_provisional_denominator": float(chunk_provisional_denominator),
                "actor/t5_tensor_grad_parity/full_use_provisional_denominator": float(
                    full_use_provisional_denominator
                ),
                "actor/t5_tensor_grad_parity/chunk_use_provisional_denominator": float(
                    chunk_use_provisional_denominator
                ),
                "actor/t5_tensor_grad_parity/full_grad_rescale_basis_active_tokens": float(
                    full_grad_rescale_basis_active_tokens
                ),
                "actor/t5_tensor_grad_parity/chunk_grad_rescale_basis_active_tokens": float(
                    chunk_grad_rescale_basis_active_tokens
                ),
                "actor/t5_tensor_grad_parity/full_grad_rescale_basis_rows": float(
                    full_grad_rescale_basis_rows
                ),
                "actor/t5_tensor_grad_parity/chunk_grad_rescale_basis_rows": float(
                    chunk_grad_rescale_basis_rows
                ),
            }
            if not passed and raise_on_fail:
                raise RuntimeError("t5 tensor grad parity failed: " + "; ".join(fail_reasons))
            return metrics
        finally:
            self.actor_optimizer.zero_grad()
            self.actor_module.train(was_training)

    def debug_t5_tensor_objective_grad_parity(
        self,
        tensor_batch: dict[str, torch.Tensor],
        meta_info: dict | None = None,
        max_rows: int = 16,
        sample_params: int = 64,
        sample_elems_per_param: int = 256,
        atol: float = 1.0e-4,
        rtol: float = 2.0e-3,
        raise_on_fail: bool = True,
    ) -> dict[str, float]:
        """Debug-only trace_all_ce vs trace_future hard-label CE grad parity."""
        if self.actor_optimizer is None:
            raise RuntimeError("debug_t5_tensor_objective_grad_parity requires an actor optimizer")
        if self.scaler is not None:
            raise NotImplementedError("debug_t5_tensor_objective_grad_parity does not support fp16 grad scaler")

        original_rows = int(tensor_batch["input_ids"].shape[0])
        if original_rows <= 0:
            raise RuntimeError("debug_t5_tensor_objective_grad_parity requires at least one tensor row")
        max_rows = int(max_rows or 0)
        total_rows = original_rows
        truncated = bool(max_rows > 0 and original_rows > max_rows)
        if truncated:
            total_rows = max_rows
            tensor_batch = self._slice_t5_tensor_batch_rows(tensor_batch, 0, total_rows)

        sample_params = int(sample_params)
        sample_elems_per_param = int(sample_elems_per_param)
        atol = float(atol)
        rtol = float(rtol)

        parity_meta_source = dict(meta_info or {})
        parity_meta_source["_t5_tensor_total_rows"] = int(total_rows)
        parity_meta_source["_t5_tensor_total_active_tokens"] = (
            self._count_t5_tensor_grad_parity_active_tokens(tensor_batch)
        )
        parity_meta_info = self._prepare_t5_tensor_grad_parity_meta(parity_meta_source, total_rows)
        was_training = bool(self.actor_module.training)
        generation_jsd_config = self.config.get("generation_consistency_jsd", {})
        sentinel = object()

        def _set_cfg(overrides: dict[str, object]) -> dict[str, object]:
            previous: dict[str, object] = {}
            for key, value in overrides.items():
                previous[key] = generation_jsd_config.get(key, sentinel)
                generation_jsd_config[key] = value
            return previous

        def _restore_cfg(previous: dict[str, object]) -> None:
            for key, value in previous.items():
                if value is sentinel:
                    generation_jsd_config.pop(key, None)
                else:
                    generation_jsd_config[key] = value

        def _run_signature(name: str, overrides: dict[str, object]) -> dict[str, object]:
            previous = _set_cfg(overrides)
            try:
                self.actor_optimizer.zero_grad()
                metrics = self.update_t5_tensor_chunk(
                    tensor_batch=tensor_batch,
                    meta_info={**parity_meta_info, "_t5_tensor_chunk_idx": 0},
                )
                active_tokens = float(metrics.get("actor/t5_tensor_replay/active_tokens", 0.0) or 0.0)
                loss_value = float(metrics.get("actor/t5_tensor_replay/loss", 0.0) or 0.0)
                (
                    grad_rescale_factor,
                    grad_rescale_denominator,
                    provisional_denominator,
                    use_provisional_denominator,
                    grad_rescale_basis_active_tokens,
                    grad_rescale_basis_rows,
                ) = self._t5_tensor_debug_finish_grad_rescale(
                    rows=total_rows,
                    active_tokens=active_tokens,
                    meta_info=parity_meta_info,
                )
                self._scale_actor_grads_for_debug(grad_rescale_factor)
                signature = self._sample_local_grad_signature(
                    sample_params=sample_params,
                    sample_elems_per_param=sample_elems_per_param,
                )
                return {
                    "name": name,
                    "metrics": metrics,
                    "active_tokens": active_tokens,
                    "loss": loss_value,
                    "grad_rescale_factor": grad_rescale_factor,
                    "grad_rescale_denominator": grad_rescale_denominator,
                    "provisional_denominator": provisional_denominator,
                    "use_provisional_denominator": use_provisional_denominator,
                    "grad_rescale_basis_active_tokens": grad_rescale_basis_active_tokens,
                    "grad_rescale_basis_rows": grad_rescale_basis_rows,
                    "signature": signature,
                }
            finally:
                _restore_cfg(previous)

        try:
            trace_all = _run_signature(
                "trace_all_ce",
                {
                    "enable": True,
                    "mode": "replace",
                    "objective": "trace_all_ce",
                    "trace_future_hard_label_ce": False,
                },
            )
            self.actor_optimizer.zero_grad()
            future_hard = _run_signature(
                "trace_future_hard_label_ce",
                {
                    "enable": True,
                    "mode": "replace",
                    "objective": "trace_future_ce",
                    "trace_future_hard_label_ce": True,
                },
            )

            trace_all_sig = trace_all["signature"]
            future_sig = future_hard["signature"]
            trace_all_samples = trace_all_sig["samples"]
            future_samples = future_sig["samples"]
            sample_names = list(trace_all_samples.keys())
            missing_future_samples = sorted(set(sample_names) - set(future_samples.keys()))
            extra_future_samples = sorted(set(future_samples.keys()) - set(sample_names))
            common_sample_names = [name for name in sample_names if name in future_samples]
            if common_sample_names:
                trace_all_vec = torch.cat(
                    [trace_all_samples[name].reshape(-1) for name in common_sample_names],
                    dim=0,
                )
                future_vec = torch.cat(
                    [future_samples[name].reshape(-1) for name in common_sample_names],
                    dim=0,
                )
                diff = (trace_all_vec - future_vec).abs()
                max_abs = float(diff.max().item()) if diff.numel() > 0 else 0.0
                mean_abs = float(diff.mean().item()) if diff.numel() > 0 else 0.0
                diff_l2 = float(torch.linalg.vector_norm(trace_all_vec - future_vec).item())
                trace_all_l2 = float(torch.linalg.vector_norm(trace_all_vec).item())
                rel_l2 = diff_l2 / max(trace_all_l2, 1.0e-12)
                sampled_allclose = bool(torch.allclose(trace_all_vec, future_vec, atol=atol, rtol=rtol))
                trace_all_sample_max_abs = float(trace_all_vec.abs().max().item()) if trace_all_vec.numel() > 0 else 0.0
                future_sample_max_abs = float(future_vec.abs().max().item()) if future_vec.numel() > 0 else 0.0
            else:
                max_abs = 0.0
                mean_abs = 0.0
                rel_l2 = 0.0
                sampled_allclose = False
                trace_all_sample_max_abs = 0.0
                future_sample_max_abs = 0.0

            trace_all_grad_norm = float(trace_all_sig["grad_norm"])
            future_grad_norm = float(future_sig["grad_norm"])
            grad_norm_abs_diff = abs(trace_all_grad_norm - future_grad_norm)
            grad_norm_rel_diff = grad_norm_abs_diff / max(abs(trace_all_grad_norm), 1.0e-12)
            grad_norm_ok = grad_norm_abs_diff <= atol + rtol * abs(trace_all_grad_norm)
            loss_abs_diff = abs(float(trace_all["loss"]) - float(future_hard["loss"]))
            loss_ok = loss_abs_diff <= atol + rtol * abs(float(trace_all["loss"]))
            active_token_diff = abs(float(trace_all["active_tokens"]) - float(future_hard["active_tokens"]))
            active_ok = active_token_diff == 0.0
            nonfinite_values = float(trace_all_sig["nonfinite_values"]) + float(future_sig["nonfinite_values"])

            fail_reasons = []
            if missing_future_samples or extra_future_samples:
                fail_reasons.append(
                    "sampled parameter mismatch "
                    f"missing={missing_future_samples[:4]} extra={extra_future_samples[:4]}"
                )
            if float(trace_all_sig["sampled_elem_count"]) <= 0 or float(future_sig["sampled_elem_count"]) <= 0:
                fail_reasons.append("no sampled gradient elements")
            if nonfinite_values > 0:
                fail_reasons.append(f"nonfinite sampled/local grad values={nonfinite_values:.0f}")
            if not sampled_allclose:
                fail_reasons.append(
                    f"sampled grads differ max_abs={max_abs:.6g} mean_abs={mean_abs:.6g} rel_l2={rel_l2:.6g}"
                )
            if not grad_norm_ok:
                fail_reasons.append(
                    f"grad norm differs trace_all={trace_all_grad_norm:.6g} "
                    f"future_hard={future_grad_norm:.6g} rel_diff={grad_norm_rel_diff:.6g}"
                )
            if not loss_ok:
                fail_reasons.append(
                    f"loss differs trace_all={trace_all['loss']:.6g} "
                    f"future_hard={future_hard['loss']:.6g} abs_diff={loss_abs_diff:.6g}"
                )
            if not active_ok:
                fail_reasons.append(
                    f"active token count differs trace_all={trace_all['active_tokens']:.0f} "
                    f"future_hard={future_hard['active_tokens']:.0f}"
                )

            passed = len(fail_reasons) == 0
            metrics = {
                "actor/t5_tensor_objective_grad_parity/passed": 1.0 if passed else 0.0,
                "actor/t5_tensor_objective_grad_parity/original_rows": float(original_rows),
                "actor/t5_tensor_objective_grad_parity/rows": float(total_rows),
                "actor/t5_tensor_objective_grad_parity/truncated": 1.0 if truncated else 0.0,
                "actor/t5_tensor_objective_grad_parity/sample_params": float(sample_params),
                "actor/t5_tensor_objective_grad_parity/sample_elems_per_param": float(sample_elems_per_param),
                "actor/t5_tensor_objective_grad_parity/sampled_params": float(trace_all_sig["sampled_param_count"]),
                "actor/t5_tensor_objective_grad_parity/sampled_elems": float(trace_all_sig["sampled_elem_count"]),
                "actor/t5_tensor_objective_grad_parity/atol": float(atol),
                "actor/t5_tensor_objective_grad_parity/rtol": float(rtol),
                "actor/t5_tensor_objective_grad_parity/max_abs": float(max_abs),
                "actor/t5_tensor_objective_grad_parity/mean_abs": float(mean_abs),
                "actor/t5_tensor_objective_grad_parity/rel_l2": float(rel_l2),
                "actor/t5_tensor_objective_grad_parity/trace_all_sample_max_abs": float(trace_all_sample_max_abs),
                "actor/t5_tensor_objective_grad_parity/future_hard_sample_max_abs": float(future_sample_max_abs),
                "actor/t5_tensor_objective_grad_parity/trace_all_grad_norm": float(trace_all_grad_norm),
                "actor/t5_tensor_objective_grad_parity/future_hard_grad_norm": float(future_grad_norm),
                "actor/t5_tensor_objective_grad_parity/grad_norm_abs_diff": float(grad_norm_abs_diff),
                "actor/t5_tensor_objective_grad_parity/grad_norm_rel_diff": float(grad_norm_rel_diff),
                "actor/t5_tensor_objective_grad_parity/trace_all_loss": float(trace_all["loss"]),
                "actor/t5_tensor_objective_grad_parity/future_hard_loss": float(future_hard["loss"]),
                "actor/t5_tensor_objective_grad_parity/loss_abs_diff": float(loss_abs_diff),
                "actor/t5_tensor_objective_grad_parity/trace_all_active_tokens": float(trace_all["active_tokens"]),
                "actor/t5_tensor_objective_grad_parity/future_hard_active_tokens": float(future_hard["active_tokens"]),
                "actor/t5_tensor_objective_grad_parity/active_token_diff": float(active_token_diff),
                "actor/t5_tensor_objective_grad_parity/nonfinite_values": float(nonfinite_values),
                "actor/t5_tensor_objective_grad_parity/trace_all_finish_grad_rescale_factor": float(
                    trace_all["grad_rescale_factor"]
                ),
                "actor/t5_tensor_objective_grad_parity/future_hard_finish_grad_rescale_factor": float(
                    future_hard["grad_rescale_factor"]
                ),
                "actor/t5_tensor_objective_grad_parity/trace_all_finish_grad_rescale_denominator": float(
                    trace_all["grad_rescale_denominator"]
                ),
                "actor/t5_tensor_objective_grad_parity/future_hard_finish_grad_rescale_denominator": float(
                    future_hard["grad_rescale_denominator"]
                ),
                "actor/t5_tensor_objective_grad_parity/trace_all_provisional_denominator": float(
                    trace_all["provisional_denominator"]
                ),
                "actor/t5_tensor_objective_grad_parity/future_hard_provisional_denominator": float(
                    future_hard["provisional_denominator"]
                ),
                "actor/t5_tensor_objective_grad_parity/trace_all_use_provisional_denominator": float(
                    trace_all["use_provisional_denominator"]
                ),
                "actor/t5_tensor_objective_grad_parity/future_hard_use_provisional_denominator": float(
                    future_hard["use_provisional_denominator"]
                ),
                "actor/t5_tensor_objective_grad_parity/missing_future_sample_params": float(
                    len(missing_future_samples)
                ),
                "actor/t5_tensor_objective_grad_parity/extra_future_sample_params": float(
                    len(extra_future_samples)
                ),
            }
            if fail_reasons:
                metrics["actor/t5_tensor_objective_grad_parity/fail_reason_count"] = float(len(fail_reasons))
                metrics["actor/t5_tensor_objective_grad_parity/fail_reason"] = "; ".join(fail_reasons[:8])
                print(
                    "[T5TensorObjectiveGradParity] FAILED "
                    + metrics["actor/t5_tensor_objective_grad_parity/fail_reason"],
                    flush=True,
                )
                if raise_on_fail:
                    raise RuntimeError(
                        "t5 tensor trace_all vs future-hard objective grad parity failed: "
                        + metrics["actor/t5_tensor_objective_grad_parity/fail_reason"]
                    )
            else:
                metrics["actor/t5_tensor_objective_grad_parity/fail_reason_count"] = 0.0
                print(
                    "[T5TensorObjectiveGradParity] passed "
                    f"rows={total_rows} active={trace_all['active_tokens']:.0f} "
                    f"max_abs={max_abs:.6g} rel_l2={rel_l2:.6g}",
                    flush=True,
                )
            return metrics
        finally:
            self.actor_optimizer.zero_grad()
            self.actor_module.train(was_training)

    def update_t5_tensor_chunk(self, tensor_batch: dict[str, torch.Tensor], meta_info: dict | None = None):
        """Accumulate generation-consistency gradients from a compact tensor chunk."""
        chunk_start = time.perf_counter()
        self.actor_module.train()
        meta_info = dict(meta_info or {})
        self._aec_diag_chunk_idx = int(meta_info.get("_t5_tensor_chunk_idx", -1))
        generation_jsd_config = self.config.get("generation_consistency_jsd", {})
        generation_jsd_override = {}
        if "_t5_tensor_row_chunk_size" in meta_info:
            generation_jsd_override["row_chunk_size"] = int(meta_info["_t5_tensor_row_chunk_size"])
        if "_t5_tensor_empty_cache" in meta_info:
            generation_jsd_override["stream_replay_empty_cache"] = bool(meta_info["_t5_tensor_empty_cache"])
        if generation_jsd_override:
            generation_jsd_config = {**dict(generation_jsd_config), **generation_jsd_override}
        if not generation_jsd_config.get("enable", False):
            raise RuntimeError("update_t5_tensor_chunk requires actor.generation_consistency_jsd.enable=true")
        objective = generation_jsd_config.get("objective", "full_jsd")
        if objective not in {"trace_future_ce", "trace_all_ce", "gemma_denoise_ce"}:
            raise RuntimeError(
                "update_t5_tensor_chunk currently supports "
                "generation_consistency_jsd.objective=trace_future_ce, trace_all_ce, "
                "or gemma_denoise_ce"
            )
        if generation_jsd_config.get("mode", "replace") != "replace":
            raise RuntimeError("update_t5_tensor_chunk currently supports generation_consistency_jsd.mode=replace")

        blocking_h2d = os.getenv("VERL_T5_TENSOR_BLOCKING_H2D") == "1"
        sync_after_h2d = os.getenv("VERL_T5_TENSOR_SYNC_AFTER_H2D") == "1"
        sync_before_backward = os.getenv("VERL_T5_TENSOR_SYNC_BEFORE_BACKWARD") == "1"
        sync_after_backward = os.getenv("VERL_T5_TENSOR_SYNC_AFTER_BACKWARD") == "1"

        device_id = get_device_id()
        h2d_start = time.perf_counter()
        model_inputs = {
            key: value.to(device_id, non_blocking=not blocking_h2d) if torch.is_tensor(value) else value
            for key, value in tensor_batch.items()
        }
        if sync_after_h2d:
            torch.cuda.synchronize()
        h2d_seconds = time.perf_counter() - h2d_start
        if "gemma_replay_schema_version" in model_inputs:
            if objective != "gemma_denoise_ce":
                raise RuntimeError(
                    "gemma_intermediate_replay_v1 requires "
                    "generation_consistency_jsd.objective=gemma_denoise_ce"
                )
            return self._update_diffusion_gemma_replay_tensor_chunk(
                model_inputs,
                meta_info=meta_info,
                generation_jsd_config=generation_jsd_config,
                h2d_seconds=h2d_seconds,
                chunk_start=chunk_start,
            )
        sparse_source_metrics = self._expand_sparse_rollout_logits_source_hidden(
            model_inputs,
            expand_dense=not self.use_remove_padding,
        )
        if generation_jsd_override:
            model_inputs["_generation_jsd_config_override"] = generation_jsd_override
        rows = int(model_inputs["input_ids"].shape[0])
        real_rows_tensor = model_inputs.pop("_t5_tensor_real_rows", None)
        real_rows = rows
        if torch.is_tensor(real_rows_tensor):
            if int(real_rows_tensor.numel()) == 1:
                real_rows = int(real_rows_tensor.detach().cpu().reshape(-1)[0].item())
            elif int(real_rows_tensor.numel()) > 0:
                real_rows = int(real_rows_tensor.detach().cpu().reshape(-1).sum().item())
            real_rows = max(real_rows, 0)
        temperature = meta_info["temperature"]
        global_step = meta_info.get("global_step", 0)
        self._last_update_policy_global_step = global_step

        token_debug_enabled = self._actor_replay_token_debug_enabled()
        input_checksum_debug_enabled = self._actor_input_checksum_debug_enabled()
        forward_inputs = {
            **model_inputs,
            "_actor_replay_debug_phase": "t5_tensor_update",
            "_actor_replay_debug_global_step": global_step,
            "_actor_replay_debug_batch_idx": int(meta_info.get("_t5_tensor_chunk_idx", 0)),
            "_actor_replay_debug_micro_idx": 0,
        } if (token_debug_enabled or input_checksum_debug_enabled) else model_inputs

        forward_start = time.perf_counter()
        outputs = self._forward_micro_batch(
            forward_inputs,
            temperature=temperature,
            calculate_entropy=False,
        )
        forward_seconds = time.perf_counter() - forward_start

        loss_prep_start = time.perf_counter()
        if "generation_jsd_loss" not in outputs:
            raise RuntimeError(f"{objective} tensor chunk did not produce generation_jsd_loss")

        jsd_metrics = outputs.get("generation_jsd_metrics", {})
        active_tokens = float(jsd_metrics.get("generation_consistency_jsd/active_tokens", 0.0) or 0.0)
        total_active_tokens = float(meta_info.get("_t5_tensor_total_active_tokens", 0.0) or 0.0)
        unknown_total_grad_rescale = bool(meta_info.get("_t5_tensor_unknown_total_grad_rescale", False))
        provisional_denominator = float(meta_info.get("_t5_tensor_provisional_denominator", 0.0) or 0.0)
        use_provisional_denominator = unknown_total_grad_rescale and provisional_denominator > 0.0
        loss_scale_basis_active_tokens = 0.0
        loss_scale_basis_rows = 0.0
        if unknown_total_grad_rescale:
            basis = self._t5_tensor_grad_rescale_basis(meta_info)
            if basis == "rows":
                loss_scale_numerator = float(real_rows)
                loss_scale_basis_rows = 1.0
            elif active_tokens > 0:
                loss_scale_numerator = active_tokens
                loss_scale_basis_active_tokens = 1.0
            else:
                loss_scale_numerator = float(real_rows)
                loss_scale_basis_rows = 1.0
            if use_provisional_denominator:
                loss_scale_factor = loss_scale_numerator / provisional_denominator
            else:
                loss_scale_factor = loss_scale_numerator
        elif total_active_tokens > 0:
            loss_scale_factor = active_tokens / total_active_tokens
            loss_scale_basis_active_tokens = 1.0
        else:
            total_rows = max(int(meta_info.get("_t5_tensor_total_rows", rows) or rows), 1)
            loss_scale_factor = real_rows / total_rows
            loss_scale_basis_rows = 1.0

        jsd_coef = float(generation_jsd_config.get("loss_coef", 1.0))
        unscaled_loss = outputs["generation_jsd_loss"] * jsd_coef
        loss = unscaled_loss * loss_scale_factor
        if objective in {"trace_all_ce", "gemma_denoise_ce"}:
            self._dump_actor_chunk_loss_debug(
                phase=f"split_t5_tensor_{objective}",
                global_step=global_step,
                chunk_idx=meta_info.get("_t5_tensor_chunk_idx", None),
                rows=rows,
                real_rows=real_rows,
                active_tokens=active_tokens,
                loss_scale_factor=loss_scale_factor,
                unscaled_loss=unscaled_loss,
                generation_jsd_loss=outputs["generation_jsd_loss"],
                scaled_loss=loss,
            )
        loss_prep_seconds = time.perf_counter() - loss_prep_start
        backward_start = time.perf_counter()
        if sync_before_backward:
            torch.cuda.synchronize()
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        if sync_after_backward:
            torch.cuda.synchronize()
        backward_seconds = time.perf_counter() - backward_start

        metrics = {
            "actor/pg_loss": 0.0,
            "actor/t5_tensor_replay/blocking_h2d": 1.0 if blocking_h2d else 0.0,
            "actor/t5_tensor_replay/sync_after_h2d": 1.0 if sync_after_h2d else 0.0,
            "actor/t5_tensor_replay/sync_before_backward": 1.0 if sync_before_backward else 0.0,
            "actor/t5_tensor_replay/sync_after_backward": 1.0 if sync_after_backward else 0.0,
            "actor/t5_tensor_replay/loss": float(outputs["generation_jsd_loss"].detach().item()),
            "actor/t5_tensor_replay/loss_scale": float(loss_scale_factor),
            "actor/t5_tensor_replay/unknown_total_grad_rescale": 1.0 if unknown_total_grad_rescale else 0.0,
            "actor/t5_tensor_replay/provisional_denominator": float(provisional_denominator),
            "actor/t5_tensor_replay/use_provisional_denominator": 1.0 if use_provisional_denominator else 0.0,
            "actor/t5_tensor_replay/loss_scale_basis_active_tokens": float(loss_scale_basis_active_tokens),
            "actor/t5_tensor_replay/loss_scale_basis_rows": float(loss_scale_basis_rows),
            "actor/t5_tensor_replay/active_tokens": float(active_tokens),
            "actor/t5_tensor_replay/rows": float(real_rows),
            "actor/t5_tensor_replay/padded_rows": float(rows),
            "actor/t5_tensor_replay/jsd_only": 1.0,
        }
        metrics.update(jsd_metrics)
        metrics.update(sparse_source_metrics)

        del outputs, model_inputs, forward_inputs, loss
        empty_cache_seconds = 0.0
        if bool(generation_jsd_config.get("stream_replay_empty_cache", True)):
            empty_cache_start = time.perf_counter()
            torch.cuda.empty_cache()
            gc.collect()
            empty_cache_seconds = time.perf_counter() - empty_cache_start
        chunk_total_seconds = time.perf_counter() - chunk_start
        metrics.update(
            {
                "actor/t5_tensor_replay/h2d_seconds": h2d_seconds,
                "actor/t5_tensor_replay/forward_seconds": forward_seconds,
                "actor/t5_tensor_replay/loss_prep_seconds": loss_prep_seconds,
                "actor/t5_tensor_replay/backward_seconds": backward_seconds,
                "actor/t5_tensor_replay/empty_cache_seconds": empty_cache_seconds,
                "actor/t5_tensor_replay/chunk_total_seconds": chunk_total_seconds,
                "actor/t5_tensor_replay/rows_per_second": (
                    float(rows) / chunk_total_seconds if chunk_total_seconds > 0 else 0.0
                ),
                "actor/t5_tensor_replay/active_tokens_per_second": (
                    float(active_tokens) / chunk_total_seconds if chunk_total_seconds > 0 else 0.0
                ),
            }
        )
        return metrics

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy: bool = False) -> dict[str, torch.Tensor]:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            dict[str, torch.Tensor]: a dict containing keys
                - ``log_probs``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``entropys``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``sum_pi_squared``: tensor of shape [batch_size, response_length]. torch.float32.
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)

        # Diffusion batches rely on the model's training forward path to build
        # the packed noisy/clean attention pattern.  Keep no_grad below, but do
        # not force eval mode for those batches.
        if "response_current_step_mask" in data.batch.keys():
            self.actor_module.train()
        else:
            self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        token_debug_enabled = self._actor_replay_token_debug_enabled()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        self._append_if_present(
            select_keys,
            data.batch,
            [
                "response_mask",
                "gemma_selected_block_indices",
                "gemma_available_block_counts",
                "gemma_denoise_noise_seeds",
            ],
        )
        if "rollout_self_conditioning_signal" in data.batch.keys():
            select_keys.append("rollout_self_conditioning_signal")
        if "rollout_logits_source_hidden" in data.batch.keys():
            select_keys.append("rollout_logits_source_hidden")
        if "response_tpf_ce_scale" in data.batch.keys():
            select_keys.append("response_tpf_ce_scale")
        self._append_if_present(
            select_keys,
            data.batch,
            [
                "rollout_self_conditioning_signal_block_values",
                "rollout_self_conditioning_signal_block_rows",
                "rollout_self_conditioning_signal_block_starts",
                "rollout_self_conditioning_signal_block_lengths",
                "rollout_logits_source_hidden_block_values",
                "rollout_logits_source_hidden_block_rows",
                "rollout_logits_source_hidden_block_starts",
                "rollout_logits_source_hidden_block_lengths",
            ],
        )
        if token_debug_enabled:
            self._append_if_present(
                select_keys,
                data.batch,
                [
                    "rollout_log_probs",
                    "response_mask",
                    "step_maps",
                    "response_current_step_mask",
                    "response_future_steps_mask",
                ],
            )
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        if self.use_prefix_grouper:
            self._append_if_present(select_keys, data.batch, ["prompts", "response_mask"])
            if "uid" in data.non_tensor_batch:
                non_tensor_select_keys.append("uid")
        # print(select_keys, non_tensor_select_keys) # ['responses', 'input_ids', 'attention_mask', 'position_ids'] ['multi_modal_inputs']

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        sum_pi_squared_lst = []
        show_progress = self.config.get("show_progress", True)
        for micro_idx, micro_batch in enumerate(
            tqdm(micro_batches, desc="Processing micro-batches", leave=False, disable=not show_progress)
        ):
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
            if token_debug_enabled:
                model_inputs["_actor_replay_debug_phase"] = "compute_log_prob"
                model_inputs["_actor_replay_debug_global_step"] = data.meta_info.get("global_step", None)
                model_inputs["_actor_replay_debug_micro_idx"] = int(micro_idx)
            with torch.no_grad():
                outputs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(outputs["log_probs"].cpu())
            if calculate_entropy:
                entropy_lst.append(outputs["entropys"])
            if calculate_sum_pi_squared:
                sum_pi_squared_lst.append(outputs["sum_pi_squared"])

        log_probs = torch.concat(log_probs_lst, dim=0)
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if calculate_sum_pi_squared:
            sum_pi_squared = torch.concat(sum_pi_squared_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if calculate_sum_pi_squared:
                sum_pi_squared = restore_dynamic_batch(sum_pi_squared, batch_idx_list)

        outputs = {"log_probs": log_probs}
        if calculate_entropy:
            outputs["entropys"] = entropys
        if calculate_sum_pi_squared:
            outputs["sum_pi_squared"] = sum_pi_squared
        return outputs

    @GPUMemoryLogger(role="dp actor causal ar score", logger=logger)
    def compute_causal_ar_rollout_scores(self, data: DataProto) -> dict[str, torch.Tensor]:
        """Compute AR probabilities for sampled rollout tokens without residuals."""
        self.actor_module.eval()
        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        self._append_if_present(select_keys, data.batch, ["response_mask"])
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        probs_lst = []
        avg_prob_lst = []
        show_progress = self.config.get("show_progress", True)
        for micro_batch in tqdm(
            micro_batches, desc="Causal AR scoring micro-batches", leave=False, disable=not show_progress
        ):
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                outputs = self._causal_ar_rollout_score_micro_batch(model_inputs, temperature=temperature)
            log_probs_lst.append(outputs["ar_token_log_probs"].cpu())
            probs_lst.append(outputs["ar_token_probs"].cpu())
            avg_prob_lst.append(outputs["ar_avg_prob"].cpu())

        ar_token_log_probs = torch.concat(log_probs_lst, dim=0)
        ar_token_probs = torch.concat(probs_lst, dim=0)
        ar_avg_prob = torch.concat(avg_prob_lst, dim=0)
        if use_dynamic_bsz:
            ar_token_log_probs = restore_dynamic_batch(ar_token_log_probs, batch_idx_list)
            ar_token_probs = restore_dynamic_batch(ar_token_probs, batch_idx_list)
            ar_avg_prob = restore_dynamic_batch(ar_avg_prob, batch_idx_list)
        if os.environ.get("VERL_CAUSAL_AR_SCORE_DEBUG", "0") == "1":
            try:
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                head = ar_token_probs[0, :8].detach().float().cpu().tolist() if ar_token_probs.numel() else []
                target_head = data.batch["responses"][0, :8].detach().cpu().tolist() if "responses" in data.batch.keys() else []
                input_head = data.batch["input_ids"][0, -data.batch["responses"].shape[-1] - 1 : -data.batch["responses"].shape[-1] + 7].detach().cpu().tolist() if "responses" in data.batch.keys() else []
                print(
                    "[CausalARScoreDebug] "
                    f"rank={rank} shape={tuple(ar_token_probs.shape)} "
                    f"avg={ar_avg_prob.detach().float().cpu().tolist()} first_probs={head} "
                    f"target_ids={target_head} source_ids={input_head}",
                    flush=True,
                )
            except Exception as exc:
                print(f"[CausalARScoreDebug] failed: {exc}", flush=True)

        return {
            "ar_token_log_probs": ar_token_log_probs,
            "ar_token_probs": ar_token_probs,
            "ar_avg_prob": ar_avg_prob,
        }

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        global_step = data.meta_info.get("global_step", 0)
        self._last_update_policy_global_step = data.meta_info.get("global_step", None)
        streaming_accumulate_only = bool(data.meta_info.get("_streaming_accumulate_only", False))
        streaming_total_rows_override = data.meta_info.get("_streaming_total_rows", None)
        streaming_disable_progress = bool(data.meta_info.get("_streaming_disable_progress", False))
        replay_jsd_config = self.config.get(
            "generation_consistency_jsd", {}
        ) or {}
        if bool(
            replay_jsd_config.get(
                "require_self_conditioning_replay", False
            )
        ):
            base_actor_module = getattr(
                self.actor_module, "module", self.actor_module
            )
            model_config = getattr(base_actor_module, "config", None)
            self_conditioning_probability = float(
                getattr(model_config, "self_conditioning_prob", 0.0)
                or 0.0
            )
            if self_conditioning_probability <= 0.0:
                raise RuntimeError(
                    "actor model did not enable self-conditioning"
                )
            has_self_conditioning_replay = (
                "rollout_self_conditioning_signal" in data.batch
                or "rollout_self_conditioning_signal_block_values"
                in data.batch
                or "_packed_sparse_sc_gpu_leases"
                in data.non_tensor_batch
            )
            if not has_self_conditioning_replay:
                raise RuntimeError(
                    "self-conditioning replay is required but captured signal "
                    "is missing"
                )

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        self._append_if_present(
            select_keys,
            data.batch,
            [
                "step_maps",
                "adjacent_entropy_pair_mask",
                "adjacent_entropy_prev_mean",
                "adjacent_entropy_pair_local_step",
                "adjacent_entropy_rollout_next_mean",
                "adjacent_entropy_vocab_size",
                "adjacent_entropy_pair_ids",
            ],
        )
        self._append_if_present(
            select_keys,
            data.batch,
            [
                "gemma_selected_block_indices",
                "gemma_available_block_counts",
                "gemma_denoise_noise_seeds",
            ],
        )
        trace_config = self.config.trace_config
        trace_style = trace_config.get("future_step_penalty_style", None)
        if trace_style == 1:
            trace_entropy_penalty = (
                trace_config.get("future_step_penalty", 0) != 0
                or trace_config.get("current_step_penalty", 0) != 0
            )
        elif trace_style == 2:
            trace_entropy_penalty = (
                trace_config.get("future_step_penalty_k", 0) != 0
                or trace_config.get("current_step_penalty_k", 0) != 0
            )
        else:
            trace_entropy_penalty = trace_style is not None
        select_entropy_step_masks = self.config.calculate_entropy or (self.config.entropy_coeff != 0) or trace_entropy_penalty
        if select_entropy_step_masks:
            select_keys.extend(["response_future_steps_mask", "response_current_step_mask"])

        token_debug_enabled = self._actor_replay_token_debug_enabled()
        if token_debug_enabled:
            self._append_if_present(
                select_keys,
                data.batch,
                [
                    "step_maps",
                    "response_future_steps_mask",
                    "response_current_step_mask",
                    "rollout_log_probs",
                ],
            )
        if self.use_prefix_grouper and "prompts" in data.batch.keys():
            select_keys.append("prompts")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs only for optional diagnostics; the PPO loss
        # uses old_log_probs and confidence weights on the stored-rollout path.
        report_rollout_corr_metrics = self.config.get("report_rollout_corr_metrics", True)
        if (
            (report_rollout_corr_metrics or self.config.get("generation_consistency_jsd", {}).get("enable", False))
            and "rollout_log_probs" in data.batch.keys()
        ):
            select_keys.append("rollout_log_probs")
        if "rollout_confidence_weights" in data.batch.keys():
            select_keys.append("rollout_confidence_weights")
        if "rollout_self_conditioning_signal" in data.batch.keys():
            select_keys.append("rollout_self_conditioning_signal")
        if "rollout_logits_source_hidden" in data.batch.keys():
            select_keys.append("rollout_logits_source_hidden")
        if "response_tpf_ce_scale" in data.batch.keys():
            select_keys.append("response_tpf_ce_scale")
        self._append_if_present(
            select_keys,
            data.batch,
            [
                "rollout_self_conditioning_signal_block_values",
                "rollout_self_conditioning_signal_block_rows",
                "rollout_self_conditioning_signal_block_starts",
                "rollout_self_conditioning_signal_block_lengths",
                "rollout_logits_source_hidden_block_values",
                "rollout_logits_source_hidden_block_rows",
                "rollout_logits_source_hidden_block_starts",
                "rollout_logits_source_hidden_block_lengths",
            ],
        )
        if "rollout_quality_mask" in data.batch.keys():
            select_keys.append("rollout_quality_mask")
        if "signed_trace_ce_role" in data.batch.keys():
            select_keys.append("signed_trace_ce_role")
        if "causal_wrong_ce_role" in data.batch.keys():
            select_keys.append("causal_wrong_ce_role")
        dense_noisy_pg_config = self.config.get("dense_noisy_pg", {})
        if dense_noisy_pg_config.get("enable", False) and "dense_noisy_labels" in data.batch.keys():
            select_keys.append("dense_noisy_labels")
        self._append_if_present(
            select_keys,
            data.batch,
            [
                "tpf_action_ids",
                "tpf_action_mask",
                "tpf_row_rewards",
                "tpf_row_advantages",
                "tpf_selected_mask",
                "tpf_final_mask",
            ],
        )
        if self._qwen_nostream_grpo_enabled():
            self._append_if_present(
                select_keys,
                data.batch,
                [
                    "qwen_external_kv_q_canvas",
                    "qwen_external_kv_k_canvas",
                    "qwen_external_kv_v_canvas",
                    "qwen_external_kv_k_prefix",
                    "qwen_external_kv_v_prefix",
                    "qwen_external_kv_cu_seqlens_q",
                    "qwen_external_kv_cu_seqlens_prefix",
                    "qwen_external_kv_prefix_segment_ids",
                ],
            )

        # Build exact per-token block identities before DataProto.select drops
        # rollout object metadata. step_maps are looked up as denoise-step keys;
        # they are never interpreted as block indices.
        _pre_jsd_config = self.config.get("generation_consistency_jsd", {})
        _block_tpf_cfg = _pre_jsd_config.get("block_tpf_ce_weight", {}) or {}
        _block_tpf_requested = bool(_block_tpf_cfg.get("enable", False))
        _block_tpf_objective = str(_pre_jsd_config.get("objective", "full_jsd"))
        if _block_tpf_requested and _block_tpf_objective != "trace_all_ce":
            raise ValueError(
                "generation_consistency_jsd.block_tpf_ce_weight is supported only "
                f"with objective=trace_all_ce, got {_block_tpf_objective!r}"
            )
        _block_tpf_enable = (
            _block_tpf_requested
            and bool(_pre_jsd_config.get("enable", False))
            and _block_tpf_objective == "trace_all_ce"
        )
        _block_tpf_build = None
        if _block_tpf_enable:
            _source = str(_block_tpf_cfg.get("source", BLOCK_TPF_SOURCE)).strip()
            if _source != BLOCK_TPF_SOURCE:
                raise ValueError(
                    "block_tpf_ce_weight.source must be "
                    f"{BLOCK_TPF_SOURCE!r}, got {_source!r}"
                )
            _normalize = str(_block_tpf_cfg.get("normalize", "global_mean")).strip()
            if _normalize != "global_mean":
                raise ValueError(
                    "block_tpf_ce_weight.normalize must be 'global_mean', "
                    f"got {_normalize!r}"
                )
            _missing_policy = str(
                _block_tpf_cfg.get("missing_metadata_policy", "error")
            ).strip().lower()
            if _missing_policy not in {"error", "disable"}:
                raise ValueError(
                    "block_tpf_ce_weight.missing_metadata_policy must be error or disable"
                )
            if streaming_accumulate_only:
                raise RuntimeError(
                    "block_tpf_ce_weight requires the worker to buffer external replay "
                    "chunks and normalize the complete rollout batch before backward"
                )
            if not self.use_remove_padding:
                raise NotImplementedError(
                    "block_tpf_ce_weight requires actor.use_remove_padding=true"
                )
            if _pre_jsd_config.get("mode", "replace") != "replace":
                raise ValueError("block_tpf_ce_weight currently requires generation_consistency_jsd.mode=replace")
            if (
                float(_pre_jsd_config.get("entropy_coef", 0.0) or 0.0) != 0.0
                or float(self.config.entropy_coeff) != 0.0
                or bool(self.config.use_kl_loss)
                or bool(dense_noisy_pg_config.get("enable", False))
                or bool(trace_entropy_penalty)
            ):
                raise ValueError(
                    "block_tpf_ce_weight must be isolated from entropy, KL, dense-noisy-PG, "
                    "and trace entropy terms"
                )
            if int(self.config.ppo_epochs) != 1:
                raise ValueError(
                    "block_tpf_ce_weight requires ppo_epochs=1 so one rollout batch "
                    "produces exactly one optimizer step"
                )

            _metadata_error = None
            try:
                _rollout_sc = data.non_tensor_batch.get("rollout_self_conditioning", None)
                _response_identities = data.non_tensor_batch.get("response_uid", None)
                _step_maps = data.batch.get("step_maps", None)
                _response_mask = data.batch.get("response_mask", None)
                _responses = data.batch.get("responses", None)
                if _rollout_sc is None:
                    raise MissingBlockTPFMetadataError("rollout_self_conditioning is missing")
                if _response_identities is None:
                    raise MissingBlockTPFMetadataError("response_uid is missing")
                if _step_maps is None or _response_mask is None or _responses is None:
                    raise MissingBlockTPFMetadataError(
                        "step_maps, response_mask, and responses are required"
                    )
                _active_ce_tokens = _response_mask.bool() & _responses.ge(0)
                _block_tpf_build = build_block_tpf_token_tensors(
                    rollout_self_conditioning=_rollout_sc,
                    response_identities=_response_identities,
                    step_maps=_step_maps,
                    active_token_mask=_active_ce_tokens,
                )
            except MissingBlockTPFMetadataError as exc:
                _metadata_error = exc
            except Exception as exc:
                _metadata_error = exc

            _missing_local = isinstance(_metadata_error, MissingBlockTPFMetadataError)
            _invalid_local = _metadata_error is not None and not _missing_local
            _failure_flags = torch.tensor(
                [int(_missing_local), int(_invalid_local)],
                device=get_device_id(),
                dtype=torch.long,
            )
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(
                    _failure_flags,
                    op=torch.distributed.ReduceOp.MAX,
                )
            _missing_any = bool(_failure_flags[0].item())
            _invalid_any = bool(_failure_flags[1].item())
            _block_tpf_enable = resolve_block_tpf_metadata_policy(
                missing=_missing_any,
                invalid=_invalid_any,
                policy=_missing_policy,
                detail=(
                    str(_metadata_error)
                    if _metadata_error is not None
                    else "metadata failure on another actor rank"
                ),
            )
            if not _block_tpf_enable:
                _block_tpf_enable = False
                _block_tpf_build = None

            if _block_tpf_enable:
                assert _block_tpf_build is not None
                data.batch["block_tpf_raw"] = _block_tpf_build.raw_tpf
                data.batch["block_tpf_response_hashes"] = _block_tpf_build.response_hashes
                data.batch["block_tpf_block_indices"] = _block_tpf_build.block_indices
                data.batch["block_tpf_active_mask"] = _block_tpf_build.active_mask
                select_keys.extend(
                    [
                        "block_tpf_raw",
                        "block_tpf_response_hashes",
                        "block_tpf_block_indices",
                        "block_tpf_active_mask",
                    ]
                )

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = []
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")
        if self.use_prefix_grouper and "uid" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("uid")
        if "response_uid" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("response_uid")
        for key in (
            "_packed_sparse_sc_gpu_leases",
            "_packed_sparse_logits_gpu_leases",
        ):
            if key in data.non_tensor_batch:
                non_tensor_select_keys.append(key)

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        block_tpf_metrics = {
            "actor/block_tpf_ce/raw_tpf_min": 0.0,
            "actor/block_tpf_ce/raw_tpf_mean": 0.0,
            "actor/block_tpf_ce/raw_tpf_max": 0.0,
            "actor/block_tpf_ce/weight_min": 0.0,
            "actor/block_tpf_ce/weight_mean": 0.0,
            "actor/block_tpf_ce/weight_max": 0.0,
            "actor/block_tpf_ce/active_blocks": 0.0,
            "actor/block_tpf_ce/active_tokens": 0.0,
            "actor/block_tpf_ce/ghost_rows_excluded": 0.0,
        }
        if _block_tpf_enable:
            assert _block_tpf_build is not None
            _normalization = normalize_block_tpf_weights(
                raw_tpf=data.batch["block_tpf_raw"],
                response_hashes=data.batch["block_tpf_response_hashes"],
                block_indices=data.batch["block_tpf_block_indices"],
                active_mask=data.batch["block_tpf_active_mask"],
                min_weight=float(_block_tpf_cfg.get("min", 0.25)),
                max_weight=float(_block_tpf_cfg.get("max", 4.0)),
                ghost_rows_excluded=_block_tpf_build.ghost_rows_excluded,
            )
            data.batch["block_tpf_weight"] = _normalization.weights
            block_tpf_metrics.update(
                {
                    "actor/block_tpf_ce/raw_tpf_min": _normalization.raw_min,
                    "actor/block_tpf_ce/raw_tpf_mean": _normalization.global_mean,
                    "actor/block_tpf_ce/raw_tpf_max": _normalization.raw_max,
                    "actor/block_tpf_ce/weight_min": _normalization.weight_min,
                    "actor/block_tpf_ce/weight_mean": _normalization.weight_mean,
                    "actor/block_tpf_ce/weight_max": _normalization.weight_max,
                    "actor/block_tpf_ce/active_blocks": float(
                        _normalization.global_active_blocks
                    ),
                    "actor/block_tpf_ce/active_tokens": float(
                        _normalization.global_active_tokens
                    ),
                    "actor/block_tpf_ce/ghost_rows_excluded": float(
                        _normalization.global_ghost_rows_excluded
                    ),
                }
            )

        generation_jsd_config = self.config.get("generation_consistency_jsd", {})
        tpf_reinforce_config = generation_jsd_config.get("tpf_reinforce", {}) or {}
        tpf_reinforce_update = (
            generation_jsd_config.get("enable", False)
            and generation_jsd_config.get("objective", "full_jsd") == "tpf_reinforce"
        )
        negative_action_pg_config = (
            generation_jsd_config.get("negative_action_pg", {}) or {}
        )
        negative_action_pg_update = self._negative_action_pg_enabled()
        signed_trace_ce_config = (
            generation_jsd_config.get("signed_trace_ce", {}) or {}
        )
        signed_trace_ce_update = (
            bool(generation_jsd_config.get("enable", False))
            and generation_jsd_config.get("objective", "") == "trace_all_ce"
            and bool(signed_trace_ce_config.get("enable", False))
        )
        causal_wrong_ce_config = (
            generation_jsd_config.get("causal_wrong_ce", {}) or {}
        )
        causal_wrong_ce_update = (
            bool(generation_jsd_config.get("enable", False))
            and generation_jsd_config.get("objective", "") == "trace_all_ce"
            and bool(causal_wrong_ce_config.get("enable", False))
        )
        _adj_ent_cfg_dict = (
            generation_jsd_config.get("adjacent_entropy_contraction", {}) or {}
        )
        adj_ent_cfg = AdjacEntConfig.from_mapping(_adj_ent_cfg_dict)
        adj_ent_enable = adj_ent_cfg.enable
        response_level_aec_enable = bool(
            adj_ent_enable and adj_ent_cfg.response_level
        )
        response_level_ce_config = (
            generation_jsd_config.get("response_level_ce", {}) or {}
        )
        response_level_ce_enable = (
            bool(generation_jsd_config.get("enable", False))
            and generation_jsd_config.get("objective", "") == "trace_all_ce"
            and bool(response_level_ce_config.get("enable", False))
        )
        causal_prob_ce_filter_config = (
            generation_jsd_config.get("causal_prob_ce_filter", {}) or {}
        )
        causal_prob_ce_filter_enable = (
            bool(generation_jsd_config.get("enable", False))
            and generation_jsd_config.get("objective", "") == "trace_all_ce"
            and bool(causal_prob_ce_filter_config.get("enable", False))
        )
        causal_prob_ce_filter_threshold = float(
            causal_prob_ce_filter_config.get("threshold", 0.9)
        )
        causal_prob_ce_filter_direction = str(
            causal_prob_ce_filter_config.get(
                "direction",
                "greater_than",
            )
        ).strip().lower()
        fixed_batch_denominator = int(
            generation_jsd_config.get("fixed_batch_denominator", 0) or 0
        )
        entropy_rkl_probe_config = (
            generation_jsd_config.get("entropy_gated_rkl_probe", {}) or {}
        )
        entropy_gated_rkl_config = (
            generation_jsd_config.get("entropy_gated_rkl", {}) or {}
        )
        entropy_gated_rkl_enable = bool(
            entropy_gated_rkl_config.get("enable", False)
        )
        entropy_gated_rkl_threshold = float(
            entropy_gated_rkl_config.get("entropy_threshold", 0.10)
        )
        entropy_gated_rkl_cap = float(
            entropy_gated_rkl_config.get("rkl_cap", 5.0)
        )
        entropy_gated_rkl_cad_mode = bool(
            entropy_gated_rkl_config.get("cad_mode", False)
        )
        entropy_gated_rkl_fkl_coef = float(
            entropy_gated_rkl_config.get("fkl_coef", 1.0)
        )
        entropy_gated_rkl_cj_power = float(
            entropy_gated_rkl_config.get("cj_power", 1.0)
        )
        entropy_gated_rkl_allow_aec = bool(
            entropy_gated_rkl_config.get("allow_aec_composition", False)
        )
        entropy_rkl_probe_enable = bool(
            entropy_rkl_probe_config.get("enable", False)
        )
        entropy_rkl_probe_threshold_list = [
            float(value)
            for value in entropy_rkl_probe_config.get(
                "thresholds",
                DEFAULT_ENTROPY_THRESHOLDS,
            )
        ]
        if entropy_gated_rkl_enable and not any(
            math.isclose(
                value,
                entropy_gated_rkl_threshold,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for value in entropy_rkl_probe_threshold_list
        ):
            entropy_rkl_probe_threshold_list.append(
                entropy_gated_rkl_threshold
            )
        entropy_rkl_probe_thresholds = tuple(
            sorted(set(entropy_rkl_probe_threshold_list))
        )
        if negative_action_pg_update:
            if generation_jsd_config.get("mode", "replace") != "replace":
                raise ValueError(
                    "negative_action_pg requires "
                    "generation_consistency_jsd.mode=replace"
                )
            required_hybrid_keys = {
                "rollout_quality_mask",
                "tpf_action_ids",
                "tpf_action_mask",
                "tpf_row_advantages",
            }
            missing_hybrid_keys = sorted(
                required_hybrid_keys.difference(data.batch.keys())
            )
            if missing_hybrid_keys:
                raise RuntimeError(
                    "negative_action_pg actor batch is missing "
                    f"{missing_hybrid_keys}"
                )
            negative_advantage = float(
                negative_action_pg_config.get("advantage", -1.0)
            )
            if not math.isfinite(negative_advantage) or negative_advantage >= 0:
                raise ValueError(
                    "negative_action_pg.advantage must be finite and negative"
                )
            if not bool(
                negative_action_pg_config.get(
                    "global_token_normalization", True
                )
            ):
                raise ValueError(
                    "negative_action_pg requires global_token_normalization=true"
                )
            if int(self.ulysses_sequence_parallel_size) != 1:
                raise NotImplementedError(
                    "negative_action_pg global normalization currently requires "
                    "ulysses_sequence_parallel_size=1"
                )
            if (
                int(self.config.ppo_epochs) != 1
                or bool(self.config.use_kl_loss)
                or bool(dense_noisy_pg_config.get("enable", False))
                or float(self.config.entropy_coeff) != 0.0
                or bool(trace_entropy_penalty)
                or float(
                    generation_jsd_config.get("entropy_coef", 0.0) or 0.0
                )
                != 0.0
                or _block_tpf_enable
                or bool(
                    generation_jsd_config.get(
                        "response_tpf_ce_scale", {}
                    ).get("enable", False)
                )
            ):
                raise ValueError(
                    "negative_action_pg first version requires ppo_epochs=1 "
                    "and no KL, entropy, dense-noisy-PG, or CE weighting"
                )
        if signed_trace_ce_update:
            if generation_jsd_config.get("mode", "replace") != "replace":
                raise ValueError(
                    "signed_trace_ce requires generation_consistency_jsd.mode=replace"
                )
            if "signed_trace_ce_role" not in data.batch.keys():
                raise RuntimeError(
                    "signed_trace_ce actor batch is missing signed_trace_ce_role"
                )
            if negative_action_pg_update or tpf_reinforce_update:
                raise ValueError(
                    "signed_trace_ce cannot be combined with sampled-action objectives"
                )
            if (
                int(self.config.ppo_epochs) != 1
                or bool(self.config.use_kl_loss)
                or bool(dense_noisy_pg_config.get("enable", False))
                or float(self.config.entropy_coeff) != 0.0
                or bool(trace_entropy_penalty)
                or float(
                    generation_jsd_config.get("entropy_coef", 0.0) or 0.0
                )
                != 0.0
                or _block_tpf_enable
                or bool(
                    generation_jsd_config.get(
                        "response_tpf_ce_scale", {}
                    ).get("enable", False)
                )
                or int(
                    generation_jsd_config.get(
                        "selected_vocab_size", 0
                    )
                    or 0
                )
                != 0
            ):
                raise ValueError(
                    "signed_trace_ce requires ppo_epochs=1, full-vocab CE, "
                    "and no KL, entropy, dense-noisy-PG, or CE weighting"
                )
            if not self.use_remove_padding:
                raise NotImplementedError(
                    "signed_trace_ce requires actor.use_remove_padding=true"
                )
        if causal_wrong_ce_update:
            if generation_jsd_config.get("mode", "replace") != "replace":
                raise ValueError(
                    "causal_wrong_ce requires generation_consistency_jsd.mode=replace"
                )
            if "causal_wrong_ce_role" not in data.batch.keys():
                raise RuntimeError(
                    "causal_wrong_ce actor batch is missing causal_wrong_ce_role"
                )
            if (
                negative_action_pg_update
                or tpf_reinforce_update
                or signed_trace_ce_update
            ):
                raise ValueError(
                    "causal_wrong_ce cannot be combined with another signed or "
                    "sampled-action objective"
                )
            if (
                int(self.config.ppo_epochs) != 1
                or bool(self.config.use_kl_loss)
                or bool(dense_noisy_pg_config.get("enable", False))
                or float(self.config.entropy_coeff) != 0.0
                or bool(trace_entropy_penalty)
                or float(
                    generation_jsd_config.get("entropy_coef", 0.0) or 0.0
                )
                != 0.0
                or _block_tpf_enable
                or bool(
                    generation_jsd_config.get(
                        "response_tpf_ce_scale", {}
                    ).get("enable", False)
                )
                or int(
                    generation_jsd_config.get(
                        "selected_vocab_size", 0
                    )
                    or 0
                )
                != 0
            ):
                raise ValueError(
                    "causal_wrong_ce requires ppo_epochs=1, full-vocab CE, "
                    "and no KL, entropy, dense-noisy-PG, or CE weighting"
                )
            if not self.use_remove_padding:
                raise NotImplementedError(
                    "causal_wrong_ce requires actor.use_remove_padding=true"
                )
        tpf_global_token_normalization = (
            tpf_reinforce_update
            and bool(tpf_reinforce_config.get("global_token_normalization", True))
        )
        stream_replay_chunk_size = int(generation_jsd_config.get("stream_replay_chunk_size", 0) or 0)
        stream_replay_empty_cache = bool(generation_jsd_config.get("stream_replay_empty_cache", True))
        stream_replay_update = (
            generation_jsd_config.get("enable", False)
            and generation_jsd_config.get("objective", "full_jsd") in {"clean_agree_ce", "trace_all_ce", "trace_future_ce"}
            and stream_replay_chunk_size > 0
        )
        if negative_action_pg_update and stream_replay_chunk_size <= 0:
            raise ValueError(
                "negative_action_pg requires stream_replay_chunk_size>0 for "
                "rank-aligned one-step gradient accumulation"
            )
        if streaming_accumulate_only:
            stream_replay_update = True
            if stream_replay_chunk_size <= 0:
                stream_replay_chunk_size = int(data.batch.batch_size[0])
        stream_trace_all_ce_extra_grad_terms = (
            bool(self.config.use_kl_loss)
            or bool(dense_noisy_pg_config.get("enable", False))
            or float(self.config.entropy_coeff) != 0.0
            or bool(trace_entropy_penalty)
        )
        trace_all_ce_global_denominator_scale = (
            (stream_replay_update or _block_tpf_enable)
            and generation_jsd_config.get("enable", False)
            and generation_jsd_config.get("objective", "full_jsd") == "trace_all_ce"
            and generation_jsd_config.get("mode", "replace") == "replace"
            and not stream_trace_all_ce_extra_grad_terms
            and not negative_action_pg_update
            and not signed_trace_ce_update
            and not causal_wrong_ce_update
        )
        if adj_ent_enable:
            adj_ent_cfg.validate()
            _adj_ent_objective_ok = (
                generation_jsd_config.get("enable", False)
                and generation_jsd_config.get("objective", "full_jsd") == "trace_all_ce"
                and generation_jsd_config.get("mode", "replace") == "replace"
            )
            if not _adj_ent_objective_ok:
                raise ValueError(
                    "adjacent_entropy_contraction requires "
                    "generation_consistency_jsd.enable=true, "
                    "objective=trace_all_ce, and mode=replace"
                )
            if int(self.config.ppo_epochs) != 1:
                raise ValueError(
                    "adjacent_entropy_contraction requires ppo_epochs=1"
                )
            if not (
                trace_all_ce_global_denominator_scale
                or signed_trace_ce_update
            ):
                raise ValueError(
                    "adjacent_entropy_contraction requires either the stream "
                    "global denominator path (stream_replay_chunk_size>0, no "
                    "extra grad terms) or signed_trace_ce"
                )
            if signed_trace_ce_update and not stream_replay_update:
                raise ValueError(
                    "adjacent_entropy_contraction with signed_trace_ce "
                    "requires stream_replay_chunk_size>0"
                )
            if not (
                "adjacent_entropy_pair_mask" in data.batch.keys()
                and "adjacent_entropy_prev_mean" in data.batch.keys()
                and "adjacent_entropy_rollout_next_mean" in data.batch.keys()
                and "adjacent_entropy_vocab_size" in data.batch.keys()
                and "adjacent_entropy_pair_ids" in data.batch.keys()
            ):
                raise RuntimeError(
                    "adjacent_entropy_contraction payload incomplete: "
                    "batch must contain adjacent_entropy_pair_mask, "
                    "adjacent_entropy_prev_mean, "
                    "adjacent_entropy_rollout_next_mean, "
                    "adjacent_entropy_vocab_size, and "
                    "adjacent_entropy_pair_ids"
                )
        if entropy_rkl_probe_enable:
            if not (
                generation_jsd_config.get("enable", False)
                and generation_jsd_config.get("objective", "") == "trace_all_ce"
                and generation_jsd_config.get("mode", "replace") == "replace"
            ):
                raise ValueError(
                    "entropy_gated_rkl_probe requires generation consistency "
                    "enable=true, objective=trace_all_ce, and mode=replace"
                )
            if not stream_replay_update:
                raise ValueError(
                    "entropy_gated_rkl_probe requires stream_replay_chunk_size>0"
                )
            if int(self.config.ppo_epochs) != 1:
                raise ValueError(
                    "entropy_gated_rkl_probe requires ppo_epochs=1"
                )
            if not entropy_rkl_probe_thresholds:
                raise ValueError(
                    "entropy_gated_rkl_probe requires at least one threshold"
                )
        if entropy_gated_rkl_enable:
            if not trace_all_ce_global_denominator_scale:
                raise ValueError(
                    "entropy_gated_rkl requires the trace-all-CE global token "
                    "denominator path"
                )
            if generation_jsd_config.get(
                "teacher_source",
                "rollout_final",
            ) != "rollout_final":
                raise ValueError(
                    "entropy_gated_rkl requires teacher_source=rollout_final"
                )
            if not (
                math.isfinite(entropy_gated_rkl_threshold)
                and 0.0 <= entropy_gated_rkl_threshold <= 1.0
            ):
                raise ValueError(
                    "entropy_gated_rkl.entropy_threshold must be finite and "
                    "in [0,1]"
                )
            if not (
                math.isfinite(entropy_gated_rkl_cap)
                and entropy_gated_rkl_cap > 0.0
            ):
                raise ValueError(
                    "entropy_gated_rkl.rkl_cap must be finite and positive"
                )
            validate_entropy_gated_rkl_composition(
                adj_ent_enable=adj_ent_enable,
                allow_aec_composition=entropy_gated_rkl_allow_aec,
                negative_action_pg_update=negative_action_pg_update,
                signed_trace_ce_update=signed_trace_ce_update,
                causal_wrong_ce_update=causal_wrong_ce_update,
                block_tpf_enable=_block_tpf_enable,
                response_tpf_ce_scale_enable=bool(
                    generation_jsd_config.get(
                        "response_tpf_ce_scale", {}
                    ).get("enable", False)
                ),
                selected_vocab_size=int(
                    generation_jsd_config.get(
                        "selected_vocab_size", 0
                    )
                    or 0
                ),
                entropy_coef=float(
                    generation_jsd_config.get("entropy_coef", 0.0) or 0.0
                ),
            )
        if response_level_ce_enable:
            if generation_jsd_config.get("mode", "replace") != "replace":
                raise ValueError(
                    "response_level_ce requires generation consistency mode=replace"
                )
            if int(self.config.ppo_epochs) != 1:
                raise ValueError("response_level_ce requires ppo_epochs=1")
            if not trace_all_ce_global_denominator_scale:
                raise ValueError(
                    "response_level_ce requires the trace-all-CE global "
                    "denominator path"
                )
            incompatible = (
                entropy_gated_rkl_blocks_response_level_ce(
                    entropy_gated_rkl_enable=entropy_gated_rkl_enable,
                    allow_aec_composition=entropy_gated_rkl_allow_aec,
                )
                or (adj_ent_enable and not response_level_aec_enable)
                or negative_action_pg_update
                or signed_trace_ce_update
                or causal_wrong_ce_update
                or _block_tpf_enable
                or bool(
                    generation_jsd_config.get(
                        "response_tpf_ce_scale", {}
                    ).get("enable", False)
                )
                or int(
                    generation_jsd_config.get(
                        "selected_vocab_size", 0
                    )
                    or 0
                )
                != 0
                or float(
                    generation_jsd_config.get("entropy_coef", 0.0) or 0.0
                )
                != 0.0
            )
            if incompatible:
                raise ValueError(
                    "response_level_ce requires full-vocab trace-all-CE "
                    "without incompatible auxiliary objectives or CE weighting"
                )
            if "response_uid" not in data.non_tensor_batch:
                raise RuntimeError(
                    "response_level_ce requires response_uid metadata"
                )
        if response_level_aec_enable and not response_level_ce_enable:
            raise ValueError(
                "response-level AEC currently requires response_level_ce"
            )
        if fixed_batch_denominator:
            if fixed_batch_denominator < 0:
                raise ValueError(
                    "generation_consistency_jsd.fixed_batch_denominator must be >= 0"
                )
            if not response_level_ce_enable:
                raise ValueError(
                    "fixed_batch_denominator requires response_level_ce "
                    "(the grad-scale denominator must be in response units)"
                )
        if causal_prob_ce_filter_enable:
            if generation_jsd_config.get("mode", "replace") != "replace":
                raise ValueError(
                    "causal_prob_ce_filter requires generation consistency mode=replace"
                )
            if int(self.config.ppo_epochs) != 1:
                raise ValueError("causal_prob_ce_filter requires ppo_epochs=1")
            if not trace_all_ce_global_denominator_scale:
                raise ValueError(
                    "causal_prob_ce_filter requires the trace-all-CE global "
                    "denominator path"
                )
            if not (
                math.isfinite(causal_prob_ce_filter_threshold)
                and 0.0 <= causal_prob_ce_filter_threshold <= 1.0
            ):
                raise ValueError(
                    "causal_prob_ce_filter.threshold must be finite and in [0,1]"
                )
            if causal_prob_ce_filter_direction not in {
                "greater_than",
                "less_than",
            }:
                raise ValueError(
                    "causal_prob_ce_filter.direction must be greater_than or "
                    "less_than"
                )
            incompatible = (
                response_level_ce_enable
                or entropy_gated_rkl_enable
                or adj_ent_enable
                or negative_action_pg_update
                or signed_trace_ce_update
                or causal_wrong_ce_update
                or _block_tpf_enable
                or bool(
                    generation_jsd_config.get(
                        "response_tpf_ce_scale", {}
                    ).get("enable", False)
                )
                or int(
                    generation_jsd_config.get(
                        "selected_vocab_size", 0
                    )
                    or 0
                )
                != 0
                or float(
                    generation_jsd_config.get("entropy_coef", 0.0) or 0.0
                )
                != 0.0
            )
            if incompatible:
                raise ValueError(
                    "causal_prob_ce_filter requires plain full-vocab "
                    "trace-all-CE without auxiliary objectives or weighting"
                )
        if tpf_reinforce_update and int(self.config.ppo_epochs) != 1:
            raise ValueError(
                "tpf_reinforce requires ppo_epochs=1 so each on-policy rollout "
                "batch produces exactly one optimizer step"
            )
        if tpf_reinforce_update:
            optimizer_lr_override = tpf_reinforce_config.get("optimizer_lr_override", None)
            if optimizer_lr_override is not None:
                optimizer_lr_override = float(optimizer_lr_override)
                if not math.isfinite(optimizer_lr_override) or optimizer_lr_override <= 0.0:
                    raise ValueError(
                        "tpf_reinforce.optimizer_lr_override must be finite and positive"
                    )
                for param_group in self.actor_optimizer.param_groups:
                    param_group["lr"] = optimizer_lr_override

        response_level_weighting = None
        response_level_aec_weighting = None
        # Streamed t5 tensor chunks arrive with consumer-prebuilt response-level
        # weights (bare tensor chunks carry no response_uid to rebuild from).
        _rl_ce_weight_prebuilt = "response_level_ce_weight" in data.batch.keys()
        _rl_aec_weight_prebuilt = (
            "response_level_aec_pair_weight" in data.batch.keys()
        )
        if response_level_ce_enable and not (
            _rl_ce_weight_prebuilt
            and (_rl_aec_weight_prebuilt or not response_level_aec_enable)
        ):
            response_uids = list(data.non_tensor_batch["response_uid"])
            sequence_parallel_rank = (
                int(get_ulysses_sequence_parallel_rank())
                if self.use_ulysses_sp
                else 0
            )
            if not _rl_ce_weight_prebuilt:
                response_level_weighting = build_response_level_ce_weights(
                    response_uids,
                    data.batch["response_mask"],
                    sequence_parallel_rank=sequence_parallel_rank,
                )
                data.batch["response_level_ce_weight"] = (
                    response_level_weighting.weights
                )
            if response_level_aec_enable and not _rl_aec_weight_prebuilt:
                required_aec_keys = {
                    "adjacent_entropy_pair_mask",
                    "adjacent_entropy_pair_ids",
                }
                missing_aec_keys = sorted(
                    required_aec_keys.difference(data.batch.keys())
                )
                if missing_aec_keys:
                    raise RuntimeError(
                        "response-level AEC actor batch is missing "
                        f"{missing_aec_keys}"
                    )
                response_level_aec_weighting = (
                    build_response_level_aec_pair_weights(
                        response_uids,
                        data.batch["adjacent_entropy_pair_mask"],
                        data.batch["adjacent_entropy_pair_ids"],
                        sequence_parallel_rank=sequence_parallel_rank,
                    )
                )
                data.batch["response_level_aec_pair_weight"] = (
                    response_level_aec_weighting.weights
                )

        # ── Single-source AEC entropy gate (canonical implementation) ──
        # Applied ONCE here, at batch level, using the FULL-FOOTPRINT
        # H_prev of each pair (mean of the transported
        # adjacent_entropy_prev_mean over ALL of the pair's positions).
        # Exempt pairs are removed from the pair MASK, so every downstream
        # consumer (prepass, forward hinge, renorm, assertions) sees only
        # eligible pairs through the unmodified base code path — the two
        # in-path gates are then kept inert via _aec_gate_single_source.
        # Weights were built BEFORE this edit (1/N_total per pair), so a
        # gated response contributes proportionally less ("proportional
        # voice" — renorm semantics intentionally disabled).
        self._aec_gate_single_source = False
        if response_level_aec_enable:
            _sg_cfg = generation_jsd_config.get(
                "adjacent_entropy_contraction", {}
            ) or {}
            _sg_floor = float(_sg_cfg.get("floor", 0.0) or 0.0)
            _sg_ceiling = float(_sg_cfg.get("ceiling", 0.0) or 0.0)
            # Adaptive quantile gate ("ceiling_topk_ratio"): instead of an
            # absolute H_prev threshold, keep exactly the top-X% MOST
            # CONFIDENT pairs (lowest full-footprint H_prev) of this rank's
            # batch each step. Coverage stays pinned at X% for the whole run
            # (the absolute ceiling's coverage drifts as the entropy
            # distribution moves — retention% collapse is a death
            # fingerprint). Per-rank quantile => exactly X% coverage
            # globally; deterministic tie-break by (H, pair_id).
            _sg_topk = float(_sg_cfg.get("ceiling_topk_ratio", 0.0) or 0.0)
            if _sg_topk > 0.0 and (_sg_floor > 0.0 or _sg_ceiling > 0.0):
                raise RuntimeError(
                    "ceiling_topk_ratio is mutually exclusive with "
                    f"floor/ceiling (got topk={_sg_topk} floor={_sg_floor} "
                    f"ceiling={_sg_ceiling})"
                )
            if not (0.0 <= _sg_topk <= 1.0):
                raise RuntimeError(
                    f"ceiling_topk_ratio must be in [0,1], got {_sg_topk}"
                )
            if (
                (_sg_floor > 0.0 or _sg_ceiling > 0.0 or _sg_topk > 0.0)
                and "adjacent_entropy_pair_mask" in data.batch.keys()
                and "adjacent_entropy_pair_ids" in data.batch.keys()
                and "adjacent_entropy_prev_mean" in data.batch.keys()
            ):
                _sg_mask = data.batch["adjacent_entropy_pair_mask"].bool()
                _sg_ids = data.batch["adjacent_entropy_pair_ids"]
                _sg_prev = data.batch["adjacent_entropy_prev_mean"].float()
                _sg_valid = _sg_mask & _sg_ids.ge(0)
                _sg_keep = _sg_mask.clone()
                _sg_pairs = []
                for _sg_pid in torch.unique(_sg_ids[_sg_valid]).tolist():
                    _sg_pid = int(_sg_pid)
                    _sg_pos = _sg_valid & _sg_ids.eq(_sg_pid)
                    _sg_h = float(_sg_prev[_sg_pos].mean().item())
                    _sg_pairs.append((_sg_pid, _sg_pos, _sg_h))
                _sg_total = len(_sg_pairs)
                _sg_exempt = 0
                _sg_thresh = float("nan")
                if _sg_topk > 0.0:
                    if _sg_total > 0:
                        _sg_k = max(1, int(round(_sg_topk * _sg_total)))
                        _sg_order = sorted(
                            range(_sg_total),
                            key=lambda i: (_sg_pairs[i][2], _sg_pairs[i][0]),
                        )
                        _sg_keep_idx = set(_sg_order[:_sg_k])
                        _sg_thresh = _sg_pairs[_sg_order[_sg_k - 1]][2]
                        for _sg_i, (_sg_pid, _sg_pos, _sg_h) in enumerate(
                            _sg_pairs
                        ):
                            if _sg_i not in _sg_keep_idx:
                                _sg_keep = _sg_keep & ~_sg_pos
                                _sg_exempt += 1
                else:
                    for _sg_pid, _sg_pos, _sg_h in _sg_pairs:
                        # Strict inequality exempts; equality participates.
                        if (_sg_floor > 0.0 and _sg_h < _sg_floor) or (
                            _sg_ceiling > 0.0 and _sg_h > _sg_ceiling
                        ):
                            _sg_keep = _sg_keep & ~_sg_pos
                            _sg_exempt += 1
                data.batch["adjacent_entropy_pair_mask"] = _sg_keep.to(
                    data.batch["adjacent_entropy_pair_mask"].dtype
                )
                self._aec_gate_single_source = True
                print(
                    "[AEC_GATE_SINGLE_SOURCE] total_pairs="
                    f"{_sg_total} exempt={_sg_exempt} "
                    f"eligible={_sg_total - _sg_exempt} "
                    f"floor={_sg_floor} ceiling={_sg_ceiling} "
                    f"topk_ratio={_sg_topk} topk_thresh={_sg_thresh:.4f}",
                    flush=True,
                )

        # ── AEC CE-ratio coefficient controller ────────────
        # When ce_ratio_target > 0, the AEC coefficient is no longer the
        # static cfg coef: each step uses
        #   coef_t = target * EMA(CE) / EMA(raw_AEC)
        # so the weighted AEC term tracks a fixed fraction of the CE
        # objective. EMAs come from the globally reduced per-step values of
        # PREVIOUS steps (identical on every rank), so coef_t is
        # deterministic across ranks; a same-step exact ratio would need a
        # second forward pass. The first step after (re)start bootstraps
        # with the static cfg coef; state is intentionally not checkpointed
        # (one bootstrap step per controller attempt).
        _aec_ratio_target = float(
            _adj_ent_cfg_dict.get("ce_ratio_target", 0.0) or 0.0
        )
        self._aec_effective_coef = float(adj_ent_cfg.coef)
        if adj_ent_enable and _aec_ratio_target > 0.0:
            _rc_state = getattr(self, "_aec_ce_ratio_state", None)
            if (
                _rc_state
                and _rc_state.get("ce_ema", 0.0) > 0.0
                and _rc_state.get("aec_ema", 0.0) > 0.0
            ):
                _rc_lo = float(
                    _adj_ent_cfg_dict.get("ce_ratio_coef_min", 1.0)
                )
                _rc_hi = float(
                    _adj_ent_cfg_dict.get("ce_ratio_coef_max", 1000.0)
                )
                _rc_dyn = (
                    _aec_ratio_target
                    * _rc_state["ce_ema"]
                    / max(_rc_state["aec_ema"], 1e-12)
                )
                self._aec_effective_coef = min(max(_rc_dyn, _rc_lo), _rc_hi)
            print(
                "[AEC_CE_RATIO] target="
                f"{_aec_ratio_target} effective_coef="
                f"{self._aec_effective_coef:.3f}",
                flush=True,
            )

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        # Expanded replay rows belong to one on-policy rollout batch. Splitting
        # TPF rows into PPO mini-batches used to take one optimizer step per
        # split, so a single rollout batch could update the model many times and
        # collapse immediately to high-confidence repetition. Accumulate all
        # rank-aligned TPF micro-batches and take exactly one optimizer step.
        mini_batches = (
            [data]
            if (
                stream_replay_update
                or tpf_reinforce_update
                or negative_action_pg_update
                or signed_trace_ce_update
                or causal_wrong_ce_update
                or _block_tpf_enable
            )
            else data.split(self.config.ppo_mini_batch_size)
        )

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {
            "actor/pg_loss": 0.0,
            "actor/kl_loss": 0.0,
        }
        trace_all_ce_report_local_numerator = 0.0
        trace_all_ce_report_local_denominator = 0.0
        signed_trace_ce_report_local_positive_sum = 0.0
        signed_trace_ce_report_local_negative_sum = 0.0
        signed_trace_ce_report_local_positive_tokens = 0.0
        signed_trace_ce_report_local_negative_tokens = 0.0
        causal_wrong_ce_report_local_positive_sum = 0.0
        causal_wrong_ce_report_local_wrong_sum = 0.0
        causal_wrong_ce_report_local_positive_tokens = 0.0
        causal_wrong_ce_report_local_wrong_tokens = 0.0
        entropy_rkl_probe_local_stats = None
        entropy_gated_rkl_local_report = torch.zeros(
            ENTROPY_GATED_RKL_REPORT_SLOTS,
            dtype=torch.float64,
            device=get_device_id(),
        )
        if tpf_reinforce_update:
            metrics["actor/tpf_reinforce/optimizer_steps_per_rollout_batch"] = float(
                self.config.ppo_epochs * len(mini_batches)
            )
            metrics["actor/tpf_reinforce/effective_optimizer_lr"] = float(
                self.actor_optimizer.param_groups[0]["lr"]
            )

        # One global progress bar at MINI-batch level (works well for dynamic bsz).
        total_minibatches = self.config.ppo_epochs * len(mini_batches)
        show_progress = self.config.get("show_progress", True) and not streaming_disable_progress
        pbar = tqdm(total=total_minibatches, desc="PPO update (minibatch)", leave=False, disable=not show_progress)
        try:
            for epoch in range(self.config.ppo_epochs):
                for batch_idx, mini_batch in enumerate(mini_batches):
                    stream_total_rows = None
                    tpf_global_action_tokens = None
                    tpf_actor_data_parallel_size = 1
                    hybrid_global_ce_tokens = None
                    hybrid_global_action_tokens = None
                    hybrid_actor_data_parallel_size = 1
                    signed_global_positive_tokens = None
                    signed_global_negative_tokens = None
                    signed_actor_world_size = 1
                    causal_wrong_global_positive_tokens = None
                    causal_wrong_global_wrong_tokens = None
                    causal_wrong_actor_world_size = 1
                    block_tpf_local_weighted_numerator = 0.0
                    block_tpf_local_weighted_denominator = 0.0
                    if negative_action_pg_update:
                        if not torch.distributed.is_initialized():
                            raise RuntimeError(
                                "negative_action_pg global normalization "
                                "requires distributed initialization"
                            )
                        hybrid_actor_data_parallel_size = (
                            torch.distributed.get_world_size()
                        )
                        positive_ce_mask = (
                            mini_batch.batch["response_mask"].bool()
                            & mini_batch.batch[
                                "rollout_quality_mask"
                            ].bool()
                        )
                        hybrid_counts = torch.tensor(
                            [
                                float(positive_ce_mask.sum().item()),
                                float(
                                    mini_batch.batch[
                                        "tpf_action_mask"
                                    ].bool().sum().item()
                                ),
                            ],
                            device=get_device_id(),
                            dtype=torch.float64,
                        )
                        torch.distributed.all_reduce(
                            hybrid_counts,
                            op=torch.distributed.ReduceOp.SUM,
                        )
                        hybrid_global_ce_tokens = hybrid_counts[0].float()
                        hybrid_global_action_tokens = (
                            hybrid_counts[1].float()
                        )
                        if (
                            float(hybrid_counts.sum().detach().cpu().item())
                            <= 0.0
                        ):
                            raise RuntimeError(
                                "negative_action_pg rollout batch has neither "
                                "positive CE tokens nor negative action tokens"
                            )
                    if signed_trace_ce_update:
                        if not torch.distributed.is_initialized():
                            raise RuntimeError(
                                "signed_trace_ce global normalization requires "
                                "distributed initialization"
                            )
                        roles = mini_batch.batch[
                            "signed_trace_ce_role"
                        ].to(dtype=torch.int8)
                        active_labels = mini_batch.batch[
                            "response_mask"
                        ].bool()
                        local_counts = torch.tensor(
                            [
                                float(
                                    (
                                        active_labels
                                        & roles.gt(0)
                                    ).sum().item()
                                ),
                                float(
                                    (
                                        active_labels
                                        & roles.lt(0)
                                    ).sum().item()
                                ),
                            ],
                            device=get_device_id(),
                            dtype=torch.float64,
                        )
                        torch.distributed.all_reduce(
                            local_counts,
                            op=torch.distributed.ReduceOp.SUM,
                        )
                        sequence_parallel_size = max(
                            int(self.ulysses_sequence_parallel_size),
                            1,
                        )
                        local_counts /= float(sequence_parallel_size)
                        signed_global_positive_tokens = local_counts[0].float()
                        signed_global_negative_tokens = local_counts[1].float()
                        signed_actor_world_size = (
                            torch.distributed.get_world_size()
                        )
                        if float(local_counts.sum().item()) <= 0.0:
                            raise RuntimeError(
                                "signed_trace_ce rollout batch has no positive "
                                "or negative CE tokens"
                            )
                    if causal_wrong_ce_update:
                        if not torch.distributed.is_initialized():
                            raise RuntimeError(
                                "causal_wrong_ce global normalization requires "
                                "distributed initialization"
                            )
                        roles = mini_batch.batch[
                            "causal_wrong_ce_role"
                        ].to(dtype=torch.int8)
                        active_labels = mini_batch.batch[
                            "response_mask"
                        ].bool()
                        local_counts = torch.tensor(
                            [
                                float(
                                    (
                                        active_labels
                                        & roles.gt(0)
                                    ).sum().item()
                                ),
                                float(
                                    (
                                        active_labels
                                        & roles.lt(0)
                                    ).sum().item()
                                ),
                            ],
                            device=get_device_id(),
                            dtype=torch.float64,
                        )
                        torch.distributed.all_reduce(
                            local_counts,
                            op=torch.distributed.ReduceOp.SUM,
                        )
                        sequence_parallel_size = max(
                            int(self.ulysses_sequence_parallel_size),
                            1,
                        )
                        local_counts /= float(sequence_parallel_size)
                        causal_wrong_global_positive_tokens = (
                            local_counts[0].float()
                        )
                        causal_wrong_global_wrong_tokens = (
                            local_counts[1].float()
                        )
                        causal_wrong_actor_world_size = (
                            torch.distributed.get_world_size()
                        )
                        if float(local_counts.sum().item()) <= 0.0:
                            raise RuntimeError(
                                "causal_wrong_ce rollout batch has no positive "
                                "or wrong CE tokens"
                            )
                    if tpf_global_token_normalization:
                        if "tpf_action_mask" not in mini_batch.batch:
                            raise RuntimeError(
                                "tpf_reinforce global token normalization requires tpf_action_mask"
                            )
                        if not torch.distributed.is_initialized():
                            raise RuntimeError(
                                "tpf_reinforce global token normalization requires distributed initialization"
                            )
                        tpf_actor_data_parallel_size = int(
                            data.meta_info.get(
                                "tpf_actor_data_parallel_size",
                                torch.distributed.get_world_size(),
                            )
                        )
                        distributed_world_size = torch.distributed.get_world_size()
                        if distributed_world_size != tpf_actor_data_parallel_size:
                            raise NotImplementedError(
                                "tpf_reinforce global token normalization currently requires "
                                "ulysses_sequence_parallel_size=1; "
                                f"distributed_world_size={distributed_world_size} "
                                f"actor_data_parallel_size={tpf_actor_data_parallel_size}"
                            )
                        tpf_global_action_tokens = mini_batch.batch["tpf_action_mask"].sum().to(
                            device=get_device_id(),
                            dtype=torch.float32,
                        )
                        torch.distributed.all_reduce(
                            tpf_global_action_tokens,
                            op=torch.distributed.ReduceOp.SUM,
                        )
                        if float(tpf_global_action_tokens.detach().cpu().item()) <= 0.0:
                            raise RuntimeError(
                                "tpf_reinforce distributed mini-batch has zero global action tokens"
                            )
                    if stream_replay_update:
                        stream_total_rows = (
                            int(streaming_total_rows_override)
                            if streaming_total_rows_override is not None
                            else int(mini_batch.batch.batch_size[0])
                        )
                        micro_batches = mini_batch.split(stream_replay_chunk_size)
                        rank_aligned_ghost_micro_batches = 0
                        if (
                            (
                                trace_all_ce_global_denominator_scale
                                or negative_action_pg_update
                                or signed_trace_ce_update
                                or causal_wrong_ce_update
                            )
                            and bool(
                                generation_jsd_config.get(
                                    "stream_replay_rank_align_micro_batches",
                                    True,
                                )
                            )
                            and torch.distributed.is_initialized()
                        ):
                            local_micro_batch_count = len(micro_batches)
                            max_micro_batch_count = torch.tensor(
                                [local_micro_batch_count],
                                device=get_device_id(),
                                dtype=torch.long,
                            )
                            torch.distributed.all_reduce(
                                max_micro_batch_count,
                                op=torch.distributed.ReduceOp.MAX,
                            )
                            target_micro_batch_count = int(
                                max_micro_batch_count.item()
                            )
                            # A rank can hold an EMPTY shard when the step has fewer
                            # replay rows than DP ranks (code_l0: one kept rollout that fits in a
                            # single 32-block gives 3-6 rows for 8 ranks). Ghosts are forwarded
                            # with loss_scale_factor=0 and excluded from every sum, so their
                            # content only needs valid shapes: borrow a template from the lowest
                            # rank that has data instead of raising (which made the controller
                            # replay the same step and crash again, deterministically). Every
                            # rank executes both collectives below; only the empty ones use the
                            # payload.
                            any_empty_shard = torch.tensor(
                                [1 if local_micro_batch_count == 0 else 0],
                                device=get_device_id(),
                                dtype=torch.long,
                            )
                            torch.distributed.all_reduce(
                                any_empty_shard,
                                op=torch.distributed.ReduceOp.MAX,
                            )
                            if target_micro_batch_count > 0 and int(any_empty_shard.item()) > 0:
                                _ghost_world = torch.distributed.get_world_size()
                                _ghost_rank = torch.distributed.get_rank()
                                _ghost_src = torch.tensor(
                                    [_ghost_rank if local_micro_batch_count > 0 else _ghost_world],
                                    device=get_device_id(),
                                    dtype=torch.long,
                                )
                                torch.distributed.all_reduce(
                                    _ghost_src,
                                    op=torch.distributed.ReduceOp.MIN,
                                )
                                _ghost_src = int(_ghost_src.item())
                                _ghost_payload = [
                                    deepcopy(micro_batches[0]).to("cpu")
                                    if _ghost_rank == _ghost_src
                                    else None
                                ]
                                torch.distributed.broadcast_object_list(
                                    _ghost_payload,
                                    src=_ghost_src,
                                    device=torch.device("cuda", get_device_id()),
                                )
                                if local_micro_batch_count == 0:
                                    ghost_micro_batch = _ghost_payload[0]
                                    ghost_micro_batch.meta_info = {
                                        **ghost_micro_batch.meta_info,
                                        "_stream_replay_rank_alignment_ghost": True,
                                    }
                                    micro_batches.append(ghost_micro_batch)
                                    rank_aligned_ghost_micro_batches += 1
                                    print(
                                        "[STREAM_REPLAY_RANK_ALIGN] empty local shard on rank "
                                        f"{_ghost_rank}: ghost template borrowed from rank "
                                        f"{_ghost_src}, target_micro_batches={target_micro_batch_count}",
                                        flush=True,
                                    )
                                del _ghost_payload
                            while len(micro_batches) < target_micro_batch_count:
                                ghost_micro_batch = deepcopy(micro_batches[-1])
                                ghost_micro_batch.meta_info = {
                                    **ghost_micro_batch.meta_info,
                                    "_stream_replay_rank_alignment_ghost": True,
                                }
                                micro_batches.append(ghost_micro_batch)
                                rank_aligned_ghost_micro_batches += 1
                        self.gradient_accumulation = max(len(micro_batches), 1)
                    elif self.config.use_dynamic_bsz:
                        max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                        micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                    else:
                        self.gradient_accumulation = (
                            self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                        )
                        micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                    if not streaming_accumulate_only:
                        self._reset_stream_trace_all_ce_grad_scale()
                        self.actor_optimizer.zero_grad()

                    # Adjacent entropy contraction: pre-pass to count valid pairs
                    # and CE tokens across all microbatches. These global counts
                    # are needed to compute the exact per-pair normalization scale
                    # before the backward loop.
                    _adj_ent_pre_N_pair_global = 0.0
                    _adj_ent_pre_global_C = 0.0
                    # Total valid pairs/responses before floor/ceiling gate.
                    # Used to distinguish schema errors (zero total = no pairs
                    # transported at all) from legitimate all-gated conditions
                    # (total > 0 but all pairs exempt by floor/ceiling).
                    _adj_ent_pre_N_total_global = 0.0
                    # True when gates are active and every pair is exempt;
                    # AEC backward contribution is zero but this is not an error.
                    _aec_all_gated = False
                    if adj_ent_enable and (
                        trace_all_ce_global_denominator_scale
                        or signed_trace_ce_update
                    ):
                        _pre_local_pairs = 0.0
                        _pre_local_ce = 0.0
                        _pre_local_total = 0.0
                        # Positive-role rows seen (signed mode): an all-negative
                        # /all-dropped batch legitimately has zero AEC pairs.
                        _pre_local_pos = 0.0
                        for _pre_mb in micro_batches:
                            if bool(_pre_mb.meta_info.get("_stream_replay_rank_alignment_ghost", False)):
                                continue
                            _pre_batch = _pre_mb.batch
                            _pre_adj_mask = _pre_batch.get("adjacent_entropy_pair_mask", None)
                            _pre_pair_ids = _pre_batch.get(
                                "adjacent_entropy_pair_ids", None
                            )
                            _pre_resp_mask = _pre_batch.get("response_mask", None)
                            # Mirror exactly which rows the forward will contract.
                            # signed_trace_ce already masks jsd_mask to role != 0,
                            # and signed_role_filter then removes role -1, so under
                            # both exclude_negative and positive_only only role +1
                            # rows survive. Counting anything else here breaks the
                            # pair-denominator check below.
                            _pre_role = None
                            if signed_trace_ce_update and _pre_adj_mask is not None:
                                _pre_role = _pre_batch.get(
                                    "signed_trace_ce_role", None
                                )
                                if _pre_role is None:
                                    _pre_role_np = (
                                        _pre_mb.non_tensor_batch or {}
                                    ).get("signed_trace_ce_role", None)
                                    if _pre_role_np is not None:
                                        _pre_role = torch.as_tensor(
                                            _pre_role_np,
                                            device=_pre_adj_mask.device,
                                        )
                            if _pre_role is not None:
                                _pre_local_pos += float(
                                    _pre_role.eq(1).sum().item()
                                )
                            if _pre_adj_mask is not None and _pre_role is not None:
                                _pre_role_mode = str(
                                    _adj_ent_cfg_dict.get(
                                        "signed_role_filter", "exclude_negative"
                                    )
                                    or "exclude_negative"
                                ).strip().lower()
                                if _pre_role_mode in (
                                    "exclude_negative",
                                    "positive_only",
                                ):
                                    _pre_keep = _pre_role.eq(1)
                                else:
                                    _pre_keep = _pre_role.ne(0)
                                while _pre_keep.dim() < _pre_adj_mask.dim():
                                    _pre_keep = _pre_keep.unsqueeze(-1)
                                _pre_adj_mask = (
                                    _pre_adj_mask.bool() & _pre_keep
                                )
                            # Gate config — same thresholds the actual AEC
                            # forward path applies, so the prepass denominator
                            # counts ONLY eligible pairs. In single-source
                            # gate mode the pair mask was already gated
                            # upstream, so the in-path gate stays inert here
                            # too (mask edits flow into _pre_adj_mask).
                            if getattr(self, "_aec_gate_single_source", False):
                                _pre_gate_floor = 0.0
                                _pre_gate_ceiling = 0.0
                            else:
                                _pre_gate_floor = float(
                                    _adj_ent_cfg_dict.get("floor", 0.0) or 0.0
                                )
                                _pre_gate_ceiling = float(
                                    _adj_ent_cfg_dict.get("ceiling", 0.0) or 0.0
                                )
                            _pre_prev_mean = _pre_batch.get(
                                "adjacent_entropy_prev_mean", None
                            )
                            if response_level_aec_enable:
                                _pre_pair_weights = _pre_batch.get(
                                    "response_level_aec_pair_weight",
                                    None,
                                )
                                _pre_ce_weights = _pre_batch.get(
                                    "response_level_ce_weight",
                                    None,
                                )
                                if (
                                    _pre_adj_mask is None
                                    or _pre_pair_ids is None
                                    or _pre_resp_mask is None
                                    or _pre_pair_weights is None
                                    or _pre_ce_weights is None
                                ):
                                    raise RuntimeError(
                                        "response-level AEC pre-pass is "
                                        "missing masks, IDs, or weights"
                                    )
                                # Iterate per batch item (= per response) so
                                # that re-normalization is response-scoped:
                                # a response with >=1 eligible pair contributes
                                # 1.0 (its full equal weight), consistent with
                                # the renorm applied in the actual forward path.
                                _pre_gate_active = (
                                    (_pre_gate_floor > 0.0 or _pre_gate_ceiling > 0.0)
                                    and _pre_prev_mean is not None
                                )
                                # Accumulate PRE-RENORM pair-weight sums (the
                                # base-lineage semantics: sum of the per-pair
                                # constant weight over unique valid pairs).
                                # _cnt_total: all valid pairs (pre-gate);
                                # _cnt_pairs: eligible pairs only (post-gate).
                                # This quantity is chunking-independent, matches
                                # the actual path's eligible_weight_pre_renorm,
                                # and reduces to the ungated base behavior when
                                # both gates are 0.0 (so coef calibration is
                                # unchanged).  NOTE: batch dim here is packed
                                # rows, NOT responses — do not count per-row.
                                _cnt_pairs = 0.0
                                _cnt_total = 0.0
                                _pre_valid_all = (
                                    _pre_adj_mask.bool()
                                    & _pre_pair_ids.ge(0)
                                )
                                _pre_unique_ids = torch.unique(
                                    _pre_pair_ids[_pre_valid_all]
                                )
                                for _pre_pid in _pre_unique_ids.tolist():
                                    _pre_pid = int(_pre_pid)
                                    _pre_pid_pos = (
                                        _pre_valid_all
                                        & _pre_pair_ids.eq(_pre_pid)
                                    )
                                    _pre_values = _pre_pair_weights[
                                        _pre_pid_pos
                                    ].float()
                                    if (
                                        _pre_values.numel() <= 0
                                        or not bool(_pre_values.gt(0).all())
                                        or not bool(
                                            torch.allclose(
                                                _pre_values,
                                                _pre_values[:1].expand_as(
                                                    _pre_values
                                                ),
                                                rtol=0.0,
                                                atol=1e-7,
                                            )
                                        )
                                    ):
                                        raise RuntimeError(
                                            "response-level AEC pair weight "
                                            "must be positive and constant"
                                        )
                                    _pre_w = float(_pre_values[0].item())
                                    _cnt_total += _pre_w
                                    # Apply floor/ceiling gate (strict
                                    # inequality exempts; equality participates)
                                    if _pre_gate_active:
                                        _pre_h_prev = float(
                                            _pre_prev_mean[
                                                _pre_pid_pos
                                            ].float().mean().item()
                                        )
                                        if (
                                            _pre_gate_floor > 0.0
                                            and _pre_h_prev < _pre_gate_floor
                                        ):
                                            continue  # floor-exempt
                                        if (
                                            _pre_gate_ceiling > 0.0
                                            and _pre_h_prev > _pre_gate_ceiling
                                        ):
                                            continue  # ceiling-exempt
                                    _cnt_pairs += _pre_w
                                _cnt_ce = float(
                                    _pre_ce_weights[
                                        _pre_resp_mask.bool()
                                    ].sum().item()
                                )
                            else:
                                _cnt_pairs, _cnt_ce = (
                                    count_unique_valid_pairs_from_batch(
                                        _pre_adj_mask,
                                        _pre_pair_ids,
                                        _pre_resp_mask,
                                        prev_mean=_pre_prev_mean,
                                        floor=_pre_gate_floor,
                                        ceiling=_pre_gate_ceiling,
                                    )
                                )
                                # Count total valid pairs before gate to
                                # distinguish schema errors from all-gated.
                                _cnt_total, _ = (
                                    count_unique_valid_pairs_from_batch(
                                        _pre_adj_mask,
                                        _pre_pair_ids,
                                        _pre_resp_mask,
                                    )
                                )
                            _pre_local_pairs += _cnt_pairs
                            _pre_local_ce += _cnt_ce
                            _pre_local_total += _cnt_total
                        _pre_stats = torch.tensor(
                            [
                                float(_pre_local_pairs),
                                float(_pre_local_ce),
                                float(_pre_local_total),
                                float(_pre_local_pos),
                            ],
                            dtype=torch.float64,
                            device=get_device_id(),
                        )
                        if torch.distributed.is_initialized():
                            torch.distributed.all_reduce(
                                _pre_stats,
                                op=torch.distributed.ReduceOp.SUM,
                            )
                        _adj_ent_pre_N_pair_global = float(_pre_stats[0].item())
                        _adj_ent_pre_global_C = float(_pre_stats[1].item())
                        _adj_ent_pre_N_total_global = float(_pre_stats[2].item())
                        _adj_ent_pre_pos_global = float(_pre_stats[3].item())
                        # Zero valid pairs transported: AEC contributes zero
                        # this step and CE still trains. This never raises,
                        # because a raise on a fixed-seed batch turns into a
                        # deterministic crash-resume loop. Expected cases are
                        # an empty transported mask (e.g. every page cropped
                        # away on a long-prompt batch) and, in signed mode, a
                        # batch with zero positive-role rows. A non-empty mask
                        # with zero countable pairs is reported loudly.
                        if _adj_ent_pre_N_total_global <= 0.0:
                            try:
                                _d_resp = int(_pre_resp_mask.sum().item()) if _pre_resp_mask is not None else -1
                                _d_adj = int(_pre_adj_mask.sum().item()) if _pre_adj_mask is not None else -1
                                _d_ids = int(_pre_pair_ids[_pre_pair_ids >= 0].unique().numel()) \
                                    if _pre_pair_ids is not None else -1
                                _d_shape = tuple(_pre_resp_mask.shape) if _pre_resp_mask is not None else None
                            except Exception as _e:
                                _d_resp = _d_adj = _d_ids = -2
                                _d_shape = str(_e)[:60]
                            print("[AEC-PAIRS-DIAG] rank=%s resp_positions=%s adj_mask_true=%s "
                                  "unique_pair_ids=%s resp_mask_shape=%s "
                                  "N_pair_global=%s N_total_global=%s C_global=%s"
                                  % (getattr(self, "rank", "?"), _d_resp, _d_adj, _d_ids, _d_shape,
                                     _adj_ent_pre_N_pair_global, _adj_ent_pre_N_total_global,
                                     _adj_ent_pre_global_C), flush=True)
                            _aec_all_gated = True
                            _signed_no_positive = bool(
                                signed_trace_ce_update
                                and _adj_ent_pre_pos_global <= 0.0
                            )
                            if _d_adj == 0 or _signed_no_positive:
                                if not globals().get("_AEC_NO_PAIRS_ANNOUNCED"):
                                    globals()["_AEC_NO_PAIRS_ANNOUNCED"] = True
                                    print(
                                        "[AEC-NO-PAIRS-SKIP] no pairable adjacent denoising "
                                        "states this step%s; skipping the AEC term "
                                        "(CE still trains)"
                                        % (
                                            " (signed mode, zero positive-role rows)"
                                            if _signed_no_positive
                                            else ""
                                        ),
                                        flush=True,
                                    )
                            else:
                                import logging as _logging
                                _logging.getLogger(__name__).warning(
                                    "[AEC-PAIRS-WARN] transported pair mask has %s "
                                    "positions but zero countable pairs globally "
                                    "(mask/pairing mismatch); skipping the AEC term "
                                    "this step (CE still trains)",
                                    _d_adj,
                                )
                        # All-gated: valid pairs exist but every one is exempt
                        # by the active floor/ceiling gate.  AEC contributes
                        # zero to the backward pass; this is not an error.
                        if _adj_ent_pre_N_pair_global <= 0.0:
                            _aec_all_gated = True
                            import logging as _logging
                            _logging.getLogger(__name__).info(
                                "AEC all-gated: total_valid=%.0f eligible=0 "
                                "(floor=%.4f ceiling=%.4f) — skipping AEC "
                                "backward contribution this step.",
                                _adj_ent_pre_N_total_global,
                                float(
                                    _adj_ent_cfg_dict.get("floor", 0.0) or 0.0
                                ),
                                float(
                                    _adj_ent_cfg_dict.get("ceiling", 0.0) or 0.0
                                ),
                            )
                        if (
                            not _aec_all_gated
                            and trace_all_ce_global_denominator_scale
                            and _adj_ent_pre_global_C <= 0.0
                        ):
                            raise RuntimeError(
                                "adjacent entropy contraction pre-pass found "
                                "zero global trace-all-CE tokens"
                            )
                    _adj_ent_local_hinge_sum_report = 0.0
                    _adj_ent_local_pair_count_report = 0
                    _adj_ent_local_pair_weight_sum_report = 0.0
                    _adj_ent_local_pair_weight_pre_renorm_report = 0.0
                    _adj_ent_local_h_t_sum_report = 0.0
                    _adj_ent_local_h_next_sum_report = 0.0
                    _adj_ent_local_violation_count_report = 0.0
                    _adj_ent_local_rollout_next_sum_report = 0.0
                    _adj_ent_local_mismatch_signed_sum_report = 0.0
                    _adj_ent_local_mismatch_abs_sum_report = 0.0
                    _adj_ent_sp_size = max(
                        int(self.ulysses_sequence_parallel_size),
                        1,
                    )
                    for micro_idx, micro_batch in enumerate(micro_batches):
                        is_rank_alignment_ghost = bool(
                            micro_batch.meta_info.get(
                                "_stream_replay_rank_alignment_ghost",
                                False,
                            )
                        )
                        micro_batch = micro_batch.to(get_device_id())
                        micro_batch_metrics = {}
                        model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
                        response_mask = model_inputs["response_mask"]
                        old_log_prob = model_inputs["old_log_probs"]
                        advantages = model_inputs["advantages"]

                        entropy_coeff = self.config.entropy_coeff
                        loss_agg_mode = self.config.loss_agg_mode

                        trace_config = self.config.trace_config
                        trace_style = trace_config.get("future_step_penalty_style", None)
                        if trace_style == 1:
                            trace_entropy_penalty = (
                                trace_config.get("future_step_penalty", 0) != 0
                                or trace_config.get("current_step_penalty", 0) != 0
                            )
                        elif trace_style == 2:
                            trace_entropy_penalty = (
                                trace_config.get("future_step_penalty_k", 0) != 0
                                or trace_config.get("current_step_penalty_k", 0) != 0
                            )
                        else:
                            trace_entropy_penalty = trace_style is not None
                        calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0) or trace_entropy_penalty

                        if (
                            negative_action_pg_update
                            or signed_trace_ce_update
                            or causal_wrong_ce_update
                        ):
                            # Branch numerators are divided by their own
                            # distributed denominators below.
                            loss_scale_factor = 1.0
                        elif stream_replay_update:
                            loss_scale_factor = response_mask.shape[0] / max(int(stream_total_rows or 0), 1)
                        elif tpf_global_token_normalization:
                            # The TPF numerator is summed across local micro-batches.
                            # FSDP averages gradients across ranks, so the objective
                            # applies the rank scale explicitly below.
                            loss_scale_factor = 1.0
                        elif self.config.use_dynamic_bsz:
                            loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                        else:
                            loss_scale_factor = 1 / self.gradient_accumulation

                        # all return: (bsz, response_length)
                        input_checksum_debug_enabled = self._actor_input_checksum_debug_enabled()
                        if tpf_reinforce_update and self._qwen_nostream_grpo_enabled():
                            micro_batch_metrics.update(self._call_qwen_external_prefix_kv_replay(micro_batch))
                        outputs = self._forward_micro_batch(
                            {
                                **model_inputs,
                                "_actor_replay_debug_phase": "update_policy",
                                "_actor_replay_debug_global_step": global_step,
                                "_actor_replay_debug_batch_idx": int(batch_idx),
                                "_actor_replay_debug_micro_idx": int(micro_idx),
                            }
                            if (token_debug_enabled or input_checksum_debug_enabled)
                            else model_inputs,
                            temperature=temperature,
                            calculate_entropy=calculate_entropy,
                        )
                        if entropy_rkl_probe_enable:
                            _probe_mb_stats = outputs.pop(
                                "entropy_gated_rkl_probe_stats",
                                None,
                            )
                            if _probe_mb_stats is None:
                                raise RuntimeError(
                                    "entropy_gated_rkl_probe forward did not "
                                    "return diagnostic statistics"
                                )
                            if not is_rank_alignment_ghost:
                                entropy_rkl_probe_local_stats = (
                                    merge_entropy_gated_rkl_probe_stats_(
                                        entropy_rkl_probe_local_stats,
                                        _probe_mb_stats,
                                    )
                                )
                        if entropy_gated_rkl_enable:
                            _hybrid_mb_report = outputs.pop(
                                "entropy_gated_rkl_loss_stats",
                                None,
                            )
                            if _hybrid_mb_report is None:
                                raise RuntimeError(
                                    "entropy_gated_rkl forward did not return "
                                    "loss statistics"
                                )
                            if not is_rank_alignment_ghost:
                                entropy_gated_rkl_local_report.add_(
                                    _hybrid_mb_report.detach().to(
                                        dtype=torch.float64
                                    )
                                )

                        log_prob = outputs["log_probs"]
                        entropy = outputs["entropys"] if calculate_entropy else None

                        # for fully_async_policy
                        if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                            old_log_prob = model_inputs["old_log_probs"]
                        else:
                            if on_policy:
                                old_log_prob = log_prob.detach()
                            else:
                                old_log_prob = model_inputs["old_log_probs"]

                        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                        # print(loss_mode) # vanilla
                        # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                        # Extract pre-computed rollout correction weights if present
                        # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                        rollout_is_weights = model_inputs.get("rollout_is_weights", None)
                        confidence_weights = model_inputs.get("rollout_confidence_weights", None)
                        advantages = torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)
                        if confidence_weights is not None:
                            confidence_weights = confidence_weights.to(device=advantages.device, dtype=advantages.dtype)
                            confidence_weights = torch.nan_to_num(
                                confidence_weights, nan=0.0, posinf=1.0, neginf=0.0
                            ).clamp_(0.0, 1.0)
                            confidence_weights = torch.where(
                                response_mask.bool(),
                                confidence_weights,
                                torch.zeros_like(confidence_weights),
                            )
                            advantages = advantages * confidence_weights
                            valid_confidence = confidence_weights[response_mask.bool()]
                            if valid_confidence.numel() > 0:
                                micro_batch_metrics["actor/confidence_weighted_ppo/weight_mean"] = (
                                    valid_confidence.mean().detach().item()
                                )
                                micro_batch_metrics["actor/confidence_weighted_ppo/weight_min"] = (
                                    valid_confidence.min().detach().item()
                                )
                                micro_batch_metrics["actor/confidence_weighted_ppo/weight_max"] = (
                                    valid_confidence.max().detach().item()
                                )
                                micro_batch_metrics["actor/confidence_weighted_ppo/active_frac"] = (
                                    valid_confidence.gt(0).to(dtype=torch.float32).mean().detach().item()
                                )

                        # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                        # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                        generation_jsd_config = self.config.get("generation_consistency_jsd", {})
                        jsd_replaces_pg = generation_jsd_config.get("enable", False) and (
                            generation_jsd_config.get("mode", "replace") == "replace"
                        )
                        if jsd_replaces_pg:
                            pg_loss = torch.nan_to_num(log_prob, nan=0.0, posinf=0.0, neginf=0.0).sum() * 0.0
                            pg_metrics = {}
                        else:
                            policy_loss_fn = get_policy_loss_fn(loss_mode) # compute_policy_loss_vanilla

                            # Check
                            # print(model_inputs["responses"].masked_select(response_mask)[:20], old_log_prob.masked_select(response_mask)[:20], log_prob.masked_select(response_mask)[:20],
                            #       model_inputs["rollout_log_probs"].masked_select(response_mask)[:20],advantages.masked_select(response_mask)[:20])

                            # Compute policy loss (any function is expected to return 2 values)
                            pg_loss, pg_metrics = policy_loss_fn(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=advantages,
                                response_mask=response_mask,
                                loss_agg_mode=loss_agg_mode,
                                config=self.config,
                                rollout_is_weights=rollout_is_weights,
                            )
                        micro_batch_metrics.update(pg_metrics)

                        # Skip if using bypass_mode loss (metrics already computed in pg_metrics)
                        rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                        if report_rollout_corr_metrics and loss_mode != "bypass_mode" and rollout_log_prob is not None:
                            # Compute metrics using CURRENT policy π_θ vs π_rollout
                            # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                            rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                                log_prob=log_prob,
                                rollout_log_prob=rollout_log_prob,
                                response_mask=response_mask,
                            )
                            micro_batch_metrics.update(rollout_corr_metrics)

                        policy_loss = pg_loss
                        if generation_jsd_config.get("enable", False):
                            objective = generation_jsd_config.get("objective", "full_jsd")
                            jsd_mode = generation_jsd_config.get("mode", "replace")
                            if objective == "tpf_reinforce":
                                if "tpf_action_log_probs" not in outputs:
                                    raise RuntimeError("tpf_reinforce missing tpf_action_log_probs from actor forward")
                                tpf_action_mask = model_inputs["tpf_action_mask"].bool()
                                tpf_advantages = torch.nan_to_num(
                                    model_inputs["tpf_row_advantages"].to(
                                        device=outputs["tpf_action_log_probs"].device,
                                        dtype=outputs["tpf_action_log_probs"].dtype,
                                    ),
                                    nan=0.0,
                                    posinf=0.0,
                                    neginf=0.0,
                                )
                                action_token_count = tpf_action_mask.sum().clamp_min(1)
                                tpf_loss = compute_tpf_reinforce_token_mean_loss(
                                    outputs["tpf_action_log_probs"],
                                    tpf_action_mask,
                                    tpf_advantages,
                                    normalization_denominator=tpf_global_action_tokens,
                                )
                                if tpf_global_token_normalization:
                                    tpf_loss = tpf_loss * tpf_actor_data_parallel_size
                                tpf_cfg = generation_jsd_config.get("tpf_reinforce", {}) or {}
                                tpf_coef = float(tpf_cfg.get("loss_coef", generation_jsd_config.get("loss_coef", 1.0)))
                                if jsd_mode == "replace":
                                    policy_loss = tpf_coef * tpf_loss
                                    micro_batch_metrics["actor/generation_consistency_jsd_replaces_pg"] = 1.0
                                elif jsd_mode == "add":
                                    policy_loss = policy_loss + tpf_coef * tpf_loss
                                    micro_batch_metrics["actor/generation_consistency_jsd_replaces_pg"] = 0.0
                                else:
                                    raise ValueError(f"Unsupported generation_consistency_jsd.mode: {jsd_mode}")
                                selected_adv = tpf_advantages.detach()
                                selected_rewards = model_inputs["tpf_row_rewards"].detach().float()
                                selected_mask = model_inputs.get("tpf_selected_mask", None)
                                final_mask = model_inputs.get("tpf_final_mask", None)
                                micro_batch_metrics.update(
                                    {
                                        "actor/tpf_reinforce/loss": float(tpf_loss.detach().item()),
                                        "actor/tpf_reinforce/action_tokens": float(action_token_count.detach().item()),
                                        "actor/tpf_reinforce/global_action_tokens": float(
                                            tpf_global_action_tokens.detach().item()
                                        )
                                        if tpf_global_action_tokens is not None
                                        else float(action_token_count.detach().item()),
                                        "actor/tpf_reinforce/global_token_normalization": float(
                                            tpf_global_token_normalization
                                        ),
                                        "actor/tpf_reinforce/advantage_mean": float(selected_adv.float().mean().item())
                                        if selected_adv.numel() > 0
                                        else 0.0,
                                        "actor/tpf_reinforce/advantage_min": float(selected_adv.float().min().item())
                                        if selected_adv.numel() > 0
                                        else 0.0,
                                        "actor/tpf_reinforce/advantage_max": float(selected_adv.float().max().item())
                                        if selected_adv.numel() > 0
                                        else 0.0,
                                        "actor/tpf_reinforce/reward_mean": float(selected_rewards.mean().item())
                                        if selected_rewards.numel() > 0
                                        else 0.0,
                                        "actor/tpf_reinforce/reward_min": float(selected_rewards.min().item())
                                        if selected_rewards.numel() > 0
                                        else 0.0,
                                        "actor/tpf_reinforce/reward_max": float(selected_rewards.max().item())
                                        if selected_rewards.numel() > 0
                                        else 0.0,
                                        "actor/tpf_reinforce/selected_positive_advantages": float(
                                            (selected_adv > 0).sum().item()
                                        ),
                                        "actor/tpf_reinforce/selected_zero_advantages": float(
                                            (selected_adv == 0).sum().item()
                                        ),
                                        "actor/tpf_reinforce/selected_negative_advantages": float(
                                            (selected_adv < 0).sum().item()
                                        ),
                                    }
                                )
                                if selected_mask is not None:
                                    micro_batch_metrics["actor/tpf_reinforce/selected_fraction"] = float(
                                        selected_mask.float().mean().detach().item()
                                    )
                                if final_mask is not None:
                                    micro_batch_metrics["actor/tpf_reinforce/final_rows"] = float(
                                        final_mask.bool().sum().detach().item()
                                    )
                            else:
                                if "generation_jsd_loss" not in outputs:
                                    raise RuntimeError(
                                        "actor.generation_consistency_jsd enabled but generation_jsd_loss is missing"
                                    )
                                jsd_loss = outputs["generation_jsd_loss"]
                                if signed_trace_ce_update:
                                    required_signed_outputs = {
                                        "signed_trace_ce_positive_sum",
                                        "signed_trace_ce_negative_sum",
                                        "signed_trace_ce_positive_tokens",
                                        "signed_trace_ce_negative_tokens",
                                    }
                                    missing_signed_outputs = sorted(
                                        required_signed_outputs.difference(
                                            outputs.keys()
                                        )
                                    )
                                    if missing_signed_outputs:
                                        raise RuntimeError(
                                            "signed_trace_ce actor forward is "
                                            f"missing {missing_signed_outputs}"
                                        )
                                    if (
                                        signed_global_positive_tokens is None
                                        or signed_global_negative_tokens is None
                                    ):
                                        raise RuntimeError(
                                            "signed_trace_ce global denominators "
                                            "were not initialized"
                                        )
                                    positive_sum = outputs[
                                        "signed_trace_ce_positive_sum"
                                    ]
                                    negative_sum = outputs[
                                        "signed_trace_ce_negative_sum"
                                    ]
                                    positive_objective = positive_sum * 0.0
                                    if (
                                        float(
                                            signed_global_positive_tokens.item()
                                        )
                                        > 0.0
                                    ):
                                        positive_objective = (
                                            positive_sum
                                            * signed_actor_world_size
                                            / signed_global_positive_tokens
                                        )
                                    negative_objective = negative_sum * 0.0
                                    if (
                                        float(
                                            signed_global_negative_tokens.item()
                                        )
                                        > 0.0
                                    ):
                                        negative_objective = (
                                            negative_sum
                                            * signed_actor_world_size
                                            / signed_global_negative_tokens
                                        )
                                    policy_loss = (
                                        positive_objective
                                        - negative_objective
                                    )
                                    if not bool(
                                        torch.isfinite(policy_loss).all()
                                    ):
                                        raise FloatingPointError(
                                            "signed_trace_ce produced a "
                                            "non-finite distributed objective"
                                        )
                                    if is_rank_alignment_ghost:
                                        policy_loss = (
                                            positive_sum + negative_sum
                                        ) * 0.0
                                    local_positive_tokens = int(
                                        outputs[
                                            "signed_trace_ce_positive_tokens"
                                        ]
                                    )
                                    local_negative_tokens = int(
                                        outputs[
                                            "signed_trace_ce_negative_tokens"
                                        ]
                                    )
                                    if not is_rank_alignment_ghost:
                                        signed_trace_ce_report_local_positive_sum += float(
                                            positive_sum.detach().item()
                                        )
                                        signed_trace_ce_report_local_negative_sum += float(
                                            negative_sum.detach().item()
                                        )
                                        signed_trace_ce_report_local_positive_tokens += float(
                                            local_positive_tokens
                                        )
                                        signed_trace_ce_report_local_negative_tokens += float(
                                            local_negative_tokens
                                        )
                                    micro_batch_metrics.update(
                                        outputs.get(
                                            "generation_jsd_metrics", {}
                                        )
                                    )
                                    micro_batch_metrics.update(
                                        {
                                            "actor/signed_trace_ce/positive_objective": float(
                                                positive_objective.detach().item()
                                            ),
                                            "actor/signed_trace_ce/negative_objective": float(
                                                negative_objective.detach().item()
                                            ),
                                            "actor/signed_trace_ce/total_objective": float(
                                                policy_loss.detach().item()
                                            ),
                                            "actor/signed_trace_ce/local_positive_tokens": float(
                                                local_positive_tokens
                                            ),
                                            "actor/signed_trace_ce/local_negative_tokens": float(
                                                local_negative_tokens
                                            ),
                                            "actor/signed_trace_ce/global_positive_tokens": float(
                                                signed_global_positive_tokens.item()
                                            ),
                                            "actor/signed_trace_ce/global_negative_tokens": float(
                                                signed_global_negative_tokens.item()
                                            ),
                                            "actor/signed_trace_ce/finite": 1.0,
                                        }
                                    )
                                    micro_batch_metrics[
                                        "actor/generation_consistency_jsd_replaces_pg"
                                    ] = 1.0
                                elif causal_wrong_ce_update:
                                    required_causal_outputs = {
                                        "causal_wrong_ce_positive_sum",
                                        "causal_wrong_ce_wrong_sum",
                                        "causal_wrong_ce_positive_tokens",
                                        "causal_wrong_ce_wrong_tokens",
                                    }
                                    missing_causal_outputs = sorted(
                                        required_causal_outputs.difference(
                                            outputs.keys()
                                        )
                                    )
                                    if missing_causal_outputs:
                                        raise RuntimeError(
                                            "causal_wrong_ce actor forward is "
                                            f"missing {missing_causal_outputs}"
                                        )
                                    if (
                                        causal_wrong_global_positive_tokens
                                        is None
                                        or causal_wrong_global_wrong_tokens
                                        is None
                                    ):
                                        raise RuntimeError(
                                            "causal_wrong_ce global denominators "
                                            "were not initialized"
                                        )
                                    positive_sum = outputs[
                                        "causal_wrong_ce_positive_sum"
                                    ]
                                    wrong_sum = outputs[
                                        "causal_wrong_ce_wrong_sum"
                                    ]
                                    positive_objective = positive_sum * 0.0
                                    if (
                                        float(
                                            causal_wrong_global_positive_tokens.item()
                                        )
                                        > 0.0
                                    ):
                                        positive_objective = (
                                            positive_sum
                                            * causal_wrong_actor_world_size
                                            / causal_wrong_global_positive_tokens
                                        )
                                    wrong_objective = wrong_sum * 0.0
                                    if (
                                        float(
                                            causal_wrong_global_wrong_tokens.item()
                                        )
                                        > 0.0
                                    ):
                                        wrong_objective = (
                                            wrong_sum
                                            * causal_wrong_actor_world_size
                                            / causal_wrong_global_wrong_tokens
                                        )
                                    policy_loss = (
                                        positive_objective - wrong_objective
                                    )
                                    if not bool(
                                        torch.isfinite(policy_loss).all()
                                    ):
                                        raise FloatingPointError(
                                            "causal_wrong_ce produced a "
                                            "non-finite distributed objective"
                                        )
                                    if is_rank_alignment_ghost:
                                        policy_loss = (
                                            positive_sum + wrong_sum
                                        ) * 0.0
                                    local_positive_tokens = int(
                                        outputs[
                                            "causal_wrong_ce_positive_tokens"
                                        ]
                                    )
                                    local_wrong_tokens = int(
                                        outputs[
                                            "causal_wrong_ce_wrong_tokens"
                                        ]
                                    )
                                    if not is_rank_alignment_ghost:
                                        causal_wrong_ce_report_local_positive_sum += float(
                                            positive_sum.detach().item()
                                        )
                                        causal_wrong_ce_report_local_wrong_sum += float(
                                            wrong_sum.detach().item()
                                        )
                                        causal_wrong_ce_report_local_positive_tokens += float(
                                            local_positive_tokens
                                        )
                                        causal_wrong_ce_report_local_wrong_tokens += float(
                                            local_wrong_tokens
                                        )
                                    micro_batch_metrics.update(
                                        outputs.get(
                                            "generation_jsd_metrics", {}
                                        )
                                    )
                                    micro_batch_metrics.update(
                                        {
                                            "actor/causal_wrong_ce/positive_objective": float(
                                                positive_objective.detach().item()
                                            ),
                                            "actor/causal_wrong_ce/wrong_objective": float(
                                                wrong_objective.detach().item()
                                            ),
                                            "actor/causal_wrong_ce/total_objective": float(
                                                policy_loss.detach().item()
                                            ),
                                            "actor/causal_wrong_ce/local_positive_tokens": float(
                                                local_positive_tokens
                                            ),
                                            "actor/causal_wrong_ce/local_wrong_tokens": float(
                                                local_wrong_tokens
                                            ),
                                            "actor/causal_wrong_ce/global_positive_tokens": float(
                                                causal_wrong_global_positive_tokens.item()
                                            ),
                                            "actor/causal_wrong_ce/global_wrong_tokens": float(
                                                causal_wrong_global_wrong_tokens.item()
                                            ),
                                            "actor/causal_wrong_ce/finite": 1.0,
                                        }
                                    )
                                    micro_batch_metrics[
                                        "actor/generation_consistency_jsd_replaces_pg"
                                    ] = 1.0
                                elif negative_action_pg_update:
                                    if "tpf_action_log_probs" not in outputs:
                                        raise RuntimeError(
                                            "negative_action_pg missing "
                                            "tpf_action_log_probs from actor forward"
                                        )
                                    if (
                                        hybrid_global_ce_tokens is None
                                        or hybrid_global_action_tokens is None
                                    ):
                                        raise RuntimeError(
                                            "negative_action_pg global denominators "
                                            "were not initialized"
                                        )
                                    jsd_metrics = outputs.get(
                                        "generation_jsd_metrics", {}
                                    )
                                    local_ce_tokens = float(
                                        jsd_metrics.get(
                                            "generation_consistency_jsd/"
                                            "weighted_denominator",
                                            0.0,
                                        )
                                        or 0.0
                                    )
                                    ce_loss_sum = jsd_loss * local_ce_tokens
                                    action_mask = model_inputs[
                                        "tpf_action_mask"
                                    ].bool()
                                    action_advantages = torch.nan_to_num(
                                        model_inputs[
                                            "tpf_row_advantages"
                                        ].to(
                                            device=outputs[
                                                "tpf_action_log_probs"
                                            ].device,
                                            dtype=outputs[
                                                "tpf_action_log_probs"
                                            ].dtype,
                                        ),
                                        nan=0.0,
                                        posinf=0.0,
                                        neginf=0.0,
                                    )
                                    negative_pg_sum = (
                                        compute_tpf_reinforce_token_mean_loss(
                                            outputs[
                                                "tpf_action_log_probs"
                                            ],
                                            action_mask,
                                            action_advantages,
                                            normalization_denominator=1.0,
                                        )
                                    )
                                    ce_coef = float(
                                        negative_action_pg_config.get(
                                            "ce_loss_coef", 1.0
                                        )
                                    )
                                    negative_coef = float(
                                        negative_action_pg_config.get(
                                            "loss_coef", 1.0
                                        )
                                    )
                                    hybrid_objective = (
                                        combine_hybrid_objective(
                                            ce_loss_sum,
                                            negative_pg_sum,
                                            global_ce_tokens=(
                                                hybrid_global_ce_tokens
                                            ),
                                            global_action_tokens=(
                                                hybrid_global_action_tokens
                                            ),
                                            data_parallel_size=(
                                                hybrid_actor_data_parallel_size
                                            ),
                                            ce_loss_coef=ce_coef,
                                            negative_loss_coef=negative_coef,
                                        )
                                    )
                                    ce_objective = hybrid_objective.ce
                                    negative_objective = (
                                        hybrid_objective.negative_pg
                                    )
                                    policy_loss = hybrid_objective.total
                                    if is_rank_alignment_ghost:
                                        policy_loss = (
                                            policy_loss
                                            + outputs[
                                                "tpf_action_log_probs"
                                            ].sum()
                                        ) * 0.0
                                    micro_batch_metrics.update(jsd_metrics)
                                    micro_batch_metrics.update(
                                        {
                                            "actor/hybrid_ce_negative_pg/"
                                            "ce_objective": float(
                                                ce_objective.detach().item()
                                            ),
                                            "actor/hybrid_ce_negative_pg/"
                                            "negative_pg_objective": float(
                                                negative_objective.detach().item()
                                            ),
                                            "actor/hybrid_ce_negative_pg/"
                                            "total_objective": float(
                                                policy_loss.detach().item()
                                            ),
                                            "actor/hybrid_ce_negative_pg/"
                                            "local_positive_ce_tokens": float(
                                                local_ce_tokens
                                            ),
                                            "actor/hybrid_ce_negative_pg/"
                                            "local_negative_action_tokens": float(
                                                action_mask.sum()
                                                .detach()
                                                .item()
                                            ),
                                            "actor/hybrid_ce_negative_pg/"
                                            "global_positive_ce_tokens": float(
                                                hybrid_global_ce_tokens.detach()
                                                .item()
                                            ),
                                            "actor/hybrid_ce_negative_pg/"
                                            "global_negative_action_tokens": float(
                                                hybrid_global_action_tokens.detach()
                                                .item()
                                            ),
                                            "actor/hybrid_ce_negative_pg/"
                                            "ce_loss_coef": ce_coef,
                                            "actor/hybrid_ce_negative_pg/"
                                            "negative_loss_coef": negative_coef,
                                        }
                                    )
                                    micro_batch_metrics[
                                        "actor/"
                                        "generation_consistency_jsd_replaces_pg"
                                    ] = 1.0
                                else:
                                    jsd_coef = float(
                                        generation_jsd_config.get(
                                            "loss_coef", 1.0
                                        )
                                    )
                                    if jsd_mode == "replace":
                                        policy_loss = jsd_coef * jsd_loss
                                        micro_batch_metrics[
                                            "actor/"
                                            "generation_consistency_jsd_replaces_pg"
                                        ] = 1.0
                                    elif jsd_mode == "add":
                                        policy_loss = (
                                            policy_loss
                                            + jsd_coef * jsd_loss
                                        )
                                        micro_batch_metrics[
                                            "actor/"
                                            "generation_consistency_jsd_replaces_pg"
                                        ] = 0.0
                                    else:
                                        raise ValueError(
                                            "Unsupported "
                                            "generation_consistency_jsd.mode: "
                                            f"{jsd_mode}"
                                        )
                                    micro_batch_metrics.update(
                                        outputs.get(
                                            "generation_jsd_metrics", {}
                                        )
                                    )
                                if (
                                    trace_all_ce_global_denominator_scale
                                    and not is_rank_alignment_ghost
                                ):
                                    _trace_ce_report_metrics = outputs.get(
                                        "generation_jsd_metrics", {}
                                    )
                                    trace_all_ce_report_local_numerator += float(
                                        _trace_ce_report_metrics.get(
                                            "generation_consistency_jsd/"
                                            "weighted_numerator",
                                            0.0,
                                        )
                                        or 0.0
                                    )
                                    trace_all_ce_report_local_denominator += float(
                                        _trace_ce_report_metrics.get(
                                            "generation_consistency_jsd/"
                                            "weighted_denominator",
                                            0.0,
                                        )
                                        or 0.0
                                    )
                                if _block_tpf_enable:
                                    _jsd_metrics = outputs.get("generation_jsd_metrics", {})
                                    block_tpf_local_weighted_numerator += float(
                                        _jsd_metrics.get(
                                            "generation_consistency_jsd/weighted_numerator",
                                            0.0,
                                        )
                                        or 0.0
                                    )
                                    block_tpf_local_weighted_denominator += float(
                                        _jsd_metrics.get(
                                            "generation_consistency_jsd/weighted_denominator",
                                            0.0,
                                        )
                                        or 0.0
                                    )
                            trace_all_ce_chunk_denominator = 0.0
                            if trace_all_ce_global_denominator_scale:
                                trace_all_ce_chunk_denominator = float(
                                    outputs.get("generation_jsd_metrics", {}).get(
                                        "generation_consistency_jsd/weighted_denominator",
                                        0.0,
                                    )
                                    or 0.0
                                )
                                if is_rank_alignment_ghost:
                                    loss_scale_factor = 0.0
                                else:
                                    loss_scale_factor = (
                                        trace_all_ce_chunk_denominator
                                    )
                                    self._accumulate_stream_trace_all_ce_grad_scale(
                                        trace_all_ce_chunk_denominator
                                    )
                                micro_batch_metrics[
                                    "actor/stream_replay/trace_all_ce_global_denominator_scale"
                                ] = 1.0
                                micro_batch_metrics[
                                    "actor/stream_replay/trace_all_ce_chunk_weighted_denominator"
                                ] = trace_all_ce_chunk_denominator
                            # Adjacent entropy contraction: add aux loss term.
                            # Scaling: policy_loss += coef * hinge_sum / N_pair_global
                            #          * global_C / C_i
                            # Existing FSDP rank averaging and the final
                            # world_size/global_C grad scale already supply
                            # world-size compensation. The hinge scalar is
                            # replicated on SP peers, but each peer carries
                            # only its shard-local exact gradient, so no SP
                            # divisor belongs in the gradient scale.
                            # After loss * C_i and final grad scale:
                            # net = coef * hinge_sum / N_pair_global  (equal-pair mean)
                            if (
                                adj_ent_enable
                                and _adj_ent_pre_N_pair_global > 0
                            ):
                                _mb_hinge = outputs.get("adj_ent_hinge_sum", None)
                                _mb_pair_cnt = outputs.get("adj_ent_pair_count", 0) or 0
                                _mb_pair_weight_sum = float(
                                    outputs.get(
                                        "adj_ent_pair_weight_sum",
                                        _mb_pair_cnt,
                                    )
                                    or 0.0
                                )
                                _mb_adj_metrics = outputs.get("adj_ent_metrics", {})
                                if not is_rank_alignment_ghost:
                                    _adj_ent_local_hinge_sum_report += float(
                                        _mb_hinge.detach().item() if _mb_hinge is not None else 0.0
                                    )
                                    _adj_ent_local_pair_count_report += int(_mb_pair_cnt)
                                    _adj_ent_local_pair_weight_sum_report += (
                                        _mb_pair_weight_sum
                                    )
                                    # Pre-renorm eligible weight for the
                                    # denominator assertion; falls back to the
                                    # post-renorm sum (identical when no gate).
                                    _adj_ent_local_pair_weight_pre_renorm_report += float(
                                        outputs.get(
                                            "adj_ent_pair_weight_pre_renorm",
                                            _mb_pair_weight_sum,
                                        )
                                        or 0.0
                                    )
                                    _adj_ent_local_h_t_sum_report += float(
                                        _mb_adj_metrics.get(
                                            "_adj_ent/h_t_sum", 0.0
                                        )
                                    )
                                    _adj_ent_local_h_next_sum_report += float(
                                        _mb_adj_metrics.get(
                                            "_adj_ent/h_next_sum", 0.0
                                        )
                                    )
                                    _adj_ent_local_violation_count_report += float(
                                        _mb_adj_metrics.get(
                                            "_adj_ent/violation_count", 0.0
                                        )
                                    )
                                    _adj_ent_local_rollout_next_sum_report += float(
                                        _mb_adj_metrics.get(
                                            "_adj_ent/rollout_next_sum", 0.0
                                        )
                                    )
                                    _adj_ent_local_mismatch_signed_sum_report += float(
                                        _mb_adj_metrics.get(
                                            "_adj_ent/mismatch_signed_sum", 0.0
                                        )
                                    )
                                    _adj_ent_local_mismatch_abs_sum_report += float(
                                        _mb_adj_metrics.get(
                                            "_adj_ent/mismatch_abs_sum", 0.0
                                        )
                                    )
                                    micro_batch_metrics.update(
                                        {
                                            key: value
                                            for key, value in _mb_adj_metrics.items()
                                            if not key.startswith("_adj_ent/")
                                        }
                                    )
                                if (
                                    _mb_hinge is not None
                                    and not _aec_all_gated
                                    and trace_all_ce_global_denominator_scale
                                    and trace_all_ce_chunk_denominator > 0
                                ):
                                    # _adj_ent_pre_N_pair_global > 0 guaranteed
                                    # here because _aec_all_gated is False.
                                    _adj_scale = (
                                        getattr(
                                            self,
                                            "_aec_effective_coef",
                                            adj_ent_cfg.coef,
                                        )
                                        * _adj_ent_pre_global_C
                                        / (
                                            _adj_ent_pre_N_pair_global
                                            * trace_all_ce_chunk_denominator
                                        )
                                    )
                                    policy_loss = policy_loss + _adj_scale * _mb_hinge
                                elif (
                                    _mb_hinge is not None
                                    and not _aec_all_gated
                                    and signed_trace_ce_update
                                    and not is_rank_alignment_ghost
                                ):
                                    # signed_trace_ce assembles policy_loss already at
                                    # final scale (positive/negative sums each carry
                                    # world_size / their own global token count), so
                                    # there is no downstream C_i multiply and no
                                    # world_size/global_C grad scale to pre-compensate.
                                    # Only FSDP rank averaging (1/world_size) remains,
                                    # which the explicit world_size factor cancels:
                                    # net = coef * hinge_sum / N_pair_global.
                                    # _adj_ent_pre_N_pair_global > 0 guaranteed.
                                    _adj_scale = (
                                        getattr(
                                            self,
                                            "_aec_effective_coef",
                                            adj_ent_cfg.coef,
                                        )
                                        * float(signed_actor_world_size)
                                        / _adj_ent_pre_N_pair_global
                                    )
                                    policy_loss = policy_loss + _adj_scale * _mb_hinge
                                elif not is_rank_alignment_ghost:
                                    # Empty shard OR all-gated: graph-connected
                                    # zero to preserve FSDP parameter graph
                                    # connectivity without adding any gradient
                                    # signal.
                                    _adj_graph_zero = (
                                        outputs["log_probs"].float().sum() * 0.0
                                    )
                                    policy_loss = policy_loss + _adj_graph_zero
                        dense_noisy_pg_config = self.config.get("dense_noisy_pg", {})
                        dense_noisy_pg_mode = dense_noisy_pg_config.get("mode", "add")
                        if dense_noisy_pg_mode not in {"add", "replace"}:
                            raise ValueError(f"Unsupported dense_noisy_pg.mode: {dense_noisy_pg_mode}")
                        if (
                            dense_noisy_pg_config.get("enable", False)
                            and "dense_noisy_labels" in model_inputs
                            and "dense_log_probs" in outputs
                            and "dense_token_probs" in outputs
                        ):
                            dense_labels = model_inputs["dense_noisy_labels"]
                            dense_mask = dense_labels.ne(-100)
                            dense_probs = torch.nan_to_num(
                                outputs["dense_token_probs"].detach(), nan=0.0, posinf=1.0, neginf=0.0
                            ).clamp_(0.0, 1.0)
                            dense_log_probs = torch.nan_to_num(
                                outputs["dense_log_probs"], nan=0.0, posinf=0.0, neginf=0.0
                            )
                            threshold = float(dense_noisy_pg_config.get("threshold", 0.1))
                            active_dense_mask = dense_mask & dense_probs.gt(threshold)
                            response_mask_f = response_mask.to(dtype=advantages.dtype)
                            seq_advantages = (
                                (advantages * response_mask_f).sum(dim=-1)
                                / response_mask_f.sum(dim=-1).clamp_min(1.0)
                            ).unsqueeze(-1)
                            dense_weights = dense_probs * active_dense_mask.to(dtype=dense_probs.dtype)
                            dense_loss_mat = -seq_advantages * dense_weights * dense_log_probs
                            if active_dense_mask.any():
                                dense_loss = agg_loss(
                                    loss_mat=dense_loss_mat,
                                    loss_mask=active_dense_mask,
                                    loss_agg_mode=loss_agg_mode,
                                )
                            else:
                                dense_loss = dense_loss_mat.sum() * 0.0
                            dense_loss_coef = float(dense_noisy_pg_config.get("loss_coef", 1.0))
                            if dense_noisy_pg_mode == "replace":
                                policy_loss = dense_loss_coef * dense_loss
                                micro_batch_metrics["actor/dense_noisy_pg_replaces_pg"] = 1.0
                            else:
                                policy_loss = policy_loss + dense_loss_coef * dense_loss
                                micro_batch_metrics["actor/dense_noisy_pg_replaces_pg"] = 0.0
                            dense_total = dense_mask.sum().clamp_min(1)
                            micro_batch_metrics["actor/dense_noisy_pg_loss"] = dense_loss.detach().item()
                            micro_batch_metrics["actor/dense_noisy_pg_active_frac"] = (
                                active_dense_mask.sum().to(dtype=torch.float32) / dense_total.to(dtype=torch.float32)
                            ).detach().item()
                            micro_batch_metrics["actor/dense_noisy_pg_prob_mean"] = (
                                dense_probs[dense_mask].mean().detach().item() if dense_mask.any() else 0.0
                            )
                        if calculate_entropy and entropy is not None:
                            entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                            micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                            if entropy_coeff != 0:
                                policy_loss -= entropy_agg * entropy_coeff


                            future_steps_entropy_agg = agg_loss(loss_mat=entropy,
                                                                loss_mask=model_inputs["response_future_steps_mask"],
                                                                loss_agg_mode=loss_agg_mode)
                            if model_inputs["response_future_steps_mask"].logical_not().all():
                                future_steps_entropy_agg = torch.zeros_like(future_steps_entropy_agg)
                            micro_batch_metrics["actor/future_steps_entropy"] = future_steps_entropy_agg.detach().item()

                            current_step_entropy_agg = agg_loss(loss_mat=entropy,
                                                                loss_mask=model_inputs["response_current_step_mask"],
                                                                loss_agg_mode=loss_agg_mode)
                            if model_inputs["response_current_step_mask"].logical_not().all():
                                current_step_entropy_agg = torch.zeros_like(current_step_entropy_agg)
                            micro_batch_metrics["actor/current_step_entropy"] = current_step_entropy_agg.detach().item()

                            if self.config.trace_config.get("future_step_penalty_style", None) == 1:
                                policy_loss -= future_steps_entropy_agg * self.config.trace_config.get("future_step_penalty", 0)
                                micro_batch_metrics["actor/future_step_penalty"] = self.config.trace_config.get("future_step_penalty", 0)

                                policy_loss -= current_step_entropy_agg * self.config.trace_config.get("current_step_penalty", 0)
                                micro_batch_metrics["actor/current_step_penalty"] = self.config.trace_config.get("current_step_penalty", 0)

                            if self.config.trace_config.get("future_step_penalty_style", None) == 2:
                                policy_loss -= future_steps_entropy_agg * self.config.trace_config.get("future_step_penalty_k", 0) * global_step
                                micro_batch_metrics["actor/future_step_penalty"] = self.config.trace_config.get("future_step_penalty_k", 0) * global_step

                                policy_loss -= current_step_entropy_agg * self.config.trace_config.get("current_step_penalty_k", 0) * global_step
                                micro_batch_metrics["actor/current_step_penalty"] = self.config.trace_config.get("current_step_penalty_k", 0) * global_step

                        if self.config.use_kl_loss:
                            ref_log_prob = model_inputs["ref_log_prob"]
                            # compute kl loss
                            kld = kl_penalty(
                                logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                            )
                            kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                            policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                            metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor
                            micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                        if self.config.use_dynamic_bsz:
                            # relative to the dynamic bsz
                            loss = policy_loss * loss_scale_factor
                        else:
                            loss = policy_loss * loss_scale_factor
                        if (
                            stream_replay_update
                            and generation_jsd_config.get("enable", False)
                            and generation_jsd_config.get("objective", "full_jsd") == "trace_all_ce"
                        ):
                            generation_jsd_metrics = outputs.get("generation_jsd_metrics", {})
                            self._dump_actor_chunk_loss_debug(
                                phase="baseline_stream_trace_all_ce",
                                global_step=global_step,
                                batch_idx=batch_idx,
                                microbatch_idx=micro_idx,
                                rows=response_mask.shape[0],
                                active_tokens=generation_jsd_metrics.get(
                                    "generation_consistency_jsd/active_tokens", 0.0
                                ),
                                loss_scale_factor=loss_scale_factor,
                                unscaled_loss=policy_loss,
                                generation_jsd_loss=outputs.get("generation_jsd_loss", None),
                                scaled_loss=loss,
                            )
                        if self.scaler is not None:
                            self.scaler.scale(loss).backward()
                        else:
                            loss.backward()

                        metrics["actor/pg_loss"] += pg_loss.detach().item() * loss_scale_factor
                        if not is_rank_alignment_ghost:
                            append_to_dict(metrics, micro_batch_metrics)
                        if stream_replay_update:
                            del outputs, model_inputs, micro_batch, log_prob, old_log_prob, advantages, response_mask, policy_loss, pg_loss, loss
                            if "entropy" in locals():
                                del entropy
                            if stream_replay_empty_cache:
                                torch.cuda.empty_cache()
                                gc.collect()

                    if signed_trace_ce_update:
                        signed_stats = torch.tensor(
                            [
                                signed_trace_ce_report_local_positive_sum,
                                signed_trace_ce_report_local_negative_sum,
                                signed_trace_ce_report_local_positive_tokens,
                                signed_trace_ce_report_local_negative_tokens,
                            ],
                            device=get_device_id(),
                            dtype=torch.float64,
                        )
                        if torch.distributed.is_initialized():
                            torch.distributed.all_reduce(
                                signed_stats,
                                op=torch.distributed.ReduceOp.SUM,
                            )
                        global_positive_sum = float(signed_stats[0].item())
                        global_negative_sum = float(signed_stats[1].item())
                        global_positive_count = float(
                            signed_stats[2].item()
                        )
                        global_negative_count = float(
                            signed_stats[3].item()
                        )
                        expected_positive_count = float(
                            signed_global_positive_tokens.item()
                            if signed_global_positive_tokens is not None
                            else 0.0
                        )
                        expected_negative_count = float(
                            signed_global_negative_tokens.item()
                            if signed_global_negative_tokens is not None
                            else 0.0
                        )
                        if (
                            abs(
                                global_positive_count
                                - expected_positive_count
                            )
                            > 0.5
                            or abs(
                                global_negative_count
                                - expected_negative_count
                            )
                            > 0.5
                        ):
                            raise RuntimeError(
                                "signed_trace_ce distributed token counts "
                                "changed during micro-batching: "
                                f"observed=({global_positive_count},"
                                f"{global_negative_count}) expected=("
                                f"{expected_positive_count},"
                                f"{expected_negative_count})"
                            )
                        positive_mean = (
                            global_positive_sum / global_positive_count
                            if global_positive_count > 0.0
                            else 0.0
                        )
                        negative_mean = (
                            global_negative_sum / global_negative_count
                            if global_negative_count > 0.0
                            else 0.0
                        )
                        signed_loss = positive_mean - negative_mean
                        if not math.isfinite(signed_loss):
                            raise FloatingPointError(
                                "signed_trace_ce global loss is non-finite"
                            )
                        append_to_dict(
                            metrics,
                            {
                                "actor/signed_trace_ce/global_positive_ce_sum": global_positive_sum,
                                "actor/signed_trace_ce/global_negative_ce_sum": global_negative_sum,
                                "actor/signed_trace_ce/global_positive_tokens": global_positive_count,
                                "actor/signed_trace_ce/global_negative_tokens": global_negative_count,
                                "actor/signed_trace_ce/global_positive_ce_mean": positive_mean,
                                "actor/signed_trace_ce/global_negative_ce_mean": negative_mean,
                                "actor/signed_trace_ce/global_signed_loss": signed_loss,
                                "actor/signed_trace_ce/finite": 1.0,
                            },
                        )

                    if causal_wrong_ce_update:
                        causal_stats = torch.tensor(
                            [
                                causal_wrong_ce_report_local_positive_sum,
                                causal_wrong_ce_report_local_wrong_sum,
                                causal_wrong_ce_report_local_positive_tokens,
                                causal_wrong_ce_report_local_wrong_tokens,
                            ],
                            device=get_device_id(),
                            dtype=torch.float64,
                        )
                        if torch.distributed.is_initialized():
                            torch.distributed.all_reduce(
                                causal_stats,
                                op=torch.distributed.ReduceOp.SUM,
                            )
                        global_positive_sum = float(causal_stats[0].item())
                        global_wrong_sum = float(causal_stats[1].item())
                        global_positive_count = float(causal_stats[2].item())
                        global_wrong_count = float(causal_stats[3].item())
                        expected_positive_count = float(
                            causal_wrong_global_positive_tokens.item()
                            if causal_wrong_global_positive_tokens is not None
                            else 0.0
                        )
                        expected_wrong_count = float(
                            causal_wrong_global_wrong_tokens.item()
                            if causal_wrong_global_wrong_tokens is not None
                            else 0.0
                        )
                        if (
                            abs(
                                global_positive_count
                                - expected_positive_count
                            )
                            > 0.5
                            or abs(
                                global_wrong_count
                                - expected_wrong_count
                            )
                            > 0.5
                        ):
                            raise RuntimeError(
                                "causal_wrong_ce distributed token counts "
                                "changed during micro-batching: "
                                f"observed=({global_positive_count},"
                                f"{global_wrong_count}) expected=("
                                f"{expected_positive_count},"
                                f"{expected_wrong_count})"
                            )
                        positive_mean = (
                            global_positive_sum / global_positive_count
                            if global_positive_count > 0.0
                            else 0.0
                        )
                        wrong_mean = (
                            global_wrong_sum / global_wrong_count
                            if global_wrong_count > 0.0
                            else 0.0
                        )
                        causal_loss = positive_mean - wrong_mean
                        if not math.isfinite(causal_loss):
                            raise FloatingPointError(
                                "causal_wrong_ce global loss is non-finite"
                            )
                        append_to_dict(
                            metrics,
                            {
                                "actor/causal_wrong_ce/global_positive_ce_sum": global_positive_sum,
                                "actor/causal_wrong_ce/global_wrong_ce_sum": global_wrong_sum,
                                "actor/causal_wrong_ce/global_positive_tokens": global_positive_count,
                                "actor/causal_wrong_ce/global_wrong_tokens": global_wrong_count,
                                "actor/causal_wrong_ce/global_positive_ce_mean": positive_mean,
                                "actor/causal_wrong_ce/global_wrong_ce_mean": wrong_mean,
                                "actor/causal_wrong_ce/global_loss": causal_loss,
                                "actor/causal_wrong_ce/finite": 1.0,
                            },
                        )

                    if _block_tpf_enable:
                        _weighted_objective_stats = torch.tensor(
                            [
                                block_tpf_local_weighted_numerator,
                                block_tpf_local_weighted_denominator,
                            ],
                            device=get_device_id(),
                            dtype=torch.float64,
                        )
                        if torch.distributed.is_initialized():
                            torch.distributed.all_reduce(
                                _weighted_objective_stats,
                                op=torch.distributed.ReduceOp.SUM,
                            )
                        _global_weighted_numerator = float(
                            _weighted_objective_stats[0].item()
                        )
                        _global_weighted_denominator = float(
                            _weighted_objective_stats[1].item()
                        )
                        block_tpf_metrics.update(
                            {
                                "actor/block_tpf_ce/weighted_numerator": _global_weighted_numerator,
                                "actor/block_tpf_ce/weighted_denominator": _global_weighted_denominator,
                                "actor/block_tpf_ce/weighted_loss": (
                                    _global_weighted_numerator
                                    / _global_weighted_denominator
                                    if _global_weighted_denominator > 0.0
                                    else 0.0
                                ),
                            }
                        )

                    if stream_replay_update or trace_all_ce_global_denominator_scale:
                        append_to_dict(
                            metrics,
                            {
                                "actor/stream_replay/chunk_size": float(stream_replay_chunk_size),
                                "actor/stream_replay/num_chunks": float(len(micro_batches)),
                                "actor/stream_replay/total_rows": float(stream_total_rows or 0),
                                "actor/stream_replay/rank_alignment_ghost_micro_batches": float(
                                    rank_aligned_ghost_micro_batches
                                    if stream_replay_update
                                    else 0
                                ),
                                "actor/stream_replay/trace_all_ce_global_denominator_scale": (
                                    1.0 if trace_all_ce_global_denominator_scale else 0.0
                                ),
                                "actor/stream_replay/trace_all_ce_local_weighted_denominator": float(
                                    getattr(
                                        self,
                                        "_stream_trace_all_ce_grad_weighted_denominator",
                                        0.0,
                                    )
                                    or 0.0
                                ),
                            },
                        )

                    if adj_ent_enable:
                        # The pre-pass sees full response masks replicated on
                        # every SP peer, while the actor's weighted CE
                        # denominator is accumulated after sequence slicing.
                        # They must match exactly after removing that one SP
                        # replication factor. Check before the optimizer step.
                        _adj_ent_actual_stats = torch.tensor(
                            [
                                float(
                                    getattr(
                                        self,
                                        "_stream_trace_all_ce_grad_weighted_denominator",
                                        0.0,
                                    )
                                    or 0.0
                                ),
                                float(
                                    # PRE-renorm eligible weight: matches the
                                    # prepass gate accounting exactly (the
                                    # prepass cannot reproduce post-renorm
                                    # micro-batch chunk boundaries).
                                    _adj_ent_local_pair_weight_pre_renorm_report
                                    if response_level_aec_enable
                                    else _adj_ent_local_pair_count_report
                                ),
                            ],
                            dtype=torch.float64,
                            device=get_device_id(),
                        )
                        if torch.distributed.is_initialized():
                            torch.distributed.all_reduce(
                                _adj_ent_actual_stats,
                                op=torch.distributed.ReduceOp.SUM,
                            )
                        _adj_ent_actual_global_C = float(
                            _adj_ent_actual_stats[0].item()
                        )
                        _adj_ent_actual_pair_denominator_replicated = float(
                            _adj_ent_actual_stats[1].item()
                        )
                        _adj_ent_expected_global_C = (
                            _adj_ent_pre_global_C / float(_adj_ent_sp_size)
                        )
                        # Tolerance must sit above fp32 summation-order noise
                        # (observed ~1e-6 abs on sums of ~19 at B8) yet below the
                        # smallest genuine mismatch quantum (one pair weight,
                        # >= ~1/64).
                        # Only the global-denominator path folds global_C into the
                        # AEC scale. signed_trace_ce normalizes its two branches
                        # independently and never accumulates a trace-all-CE
                        # weighted denominator, so this check does not apply.
                        if trace_all_ce_global_denominator_scale and not math.isclose(
                            _adj_ent_expected_global_C,
                            _adj_ent_actual_global_C,
                            rel_tol=1e-5,
                            abs_tol=1e-3,
                        ):
                            raise RuntimeError(
                                "adjacent entropy CE denominator mismatch: "
                                "prepass/sp="
                                f"{_adj_ent_expected_global_C} "
                                "actual_trace_all_ce_weighted="
                                f"{_adj_ent_actual_global_C} "
                                f"prepass_replicated={_adj_ent_pre_global_C} "
                                f"sp_size={_adj_ent_sp_size}"
                            )
                        # Eligible-pair denominator check: prepass (eligible)
                        # must match actual (eligible, pre-renorm weight).
                        # Both sides use pre-renorm pair-weight units.
                        #
                        # With an ACTIVE entropy gate the two sides can
                        # legitimately disagree on borderline pairs
                        # (H_prev ~ threshold): the prepass gates on the mean
                        # of the transported prev_mean over ALL pair
                        # positions, while the forward path gates on the mean
                        # over the CE-chunk-covered positions. Until the
                        # author picks a single canonical H_prev source, a
                        # bounded band ([0.5x, 2x]) downgrades this known
                        # boundary-flip discrepancy to a warning + metric,
                        # while gross accounting errors (the original
                        # rows-vs-weight-mass class was ~30-50x) still raise.
                        # With NO active gate the strict check is preserved
                        # (base-lineage parity).
                        _adj_gate_is_active = (
                            float(_adj_ent_cfg_dict.get("floor", 0.0) or 0.0) > 0.0
                            or float(_adj_ent_cfg_dict.get("ceiling", 0.0) or 0.0) > 0.0
                        )
                        _adj_denominator_consistent = math.isclose(
                            _adj_ent_pre_N_pair_global,
                            _adj_ent_actual_pair_denominator_replicated,
                            rel_tol=(1e-5 if response_level_aec_enable else 0.0),
                            abs_tol=(
                                1e-3
                                if response_level_aec_enable
                                else 0.5
                            ),
                        )
                        if not _adj_denominator_consistent:
                            _adj_denominator_ratio = (
                                _adj_ent_actual_pair_denominator_replicated
                                / _adj_ent_pre_N_pair_global
                                if _adj_ent_pre_N_pair_global > 0.0
                                else float("inf")
                            )
                            # With code_l0 data and no gate, steps that keep a
                            # single short rollout (kept=1 of 8, 2-3 blocks) hit the packing
                            # underfill fallback, and the forward then covers only part of the
                            # pairs the transported pair mask announced (observed ratios 0.50 and
                            # 9/17 -- exactly one block's share). The AEC term of that one step is
                            # under-weighted by that ratio; nothing else is affected. Downgrade to
                            # a warning + metric inside a wide band; gross accounting errors
                            # (the historical rows-vs-weight-mass class was 30-50x) still raise.
                            _adj_small_step_band = (
                                not _adj_gate_is_active
                                and 0.2 <= _adj_denominator_ratio <= 5.0
                            )
                            if (
                                _adj_gate_is_active
                                and 0.25 <= _adj_denominator_ratio <= 2.0
                            ) or _adj_small_step_band:
                                print(
                                    ("[AEC_GATE_WARN] eligible denominator "
                                     "boundary-flip gap (gate active): "
                                     if _adj_gate_is_active else
                                     "[AEC_UNDERFILL_WARN] eligible denominator "
                                     "gap on a small step (no gate; partial pair coverage): ")
                                    + f"prepass={_adj_ent_pre_N_pair_global} "
                                    "actual="
                                    f"{_adj_ent_actual_pair_denominator_replicated} "
                                    f"ratio={_adj_denominator_ratio:.4f}",
                                    flush=True,
                                )
                                metrics[
                                    "adj_ent/denominator_gap_ratio"
                                ] = [_adj_denominator_ratio]
                            else:
                                raise RuntimeError(
                                    "adjacent entropy eligible pair denominator "
                                    "mismatch (prepass eligible vs actual eligible): "
                                    f"prepass_eligible={_adj_ent_pre_N_pair_global} "
                                    "actual_eligible_sp_replicated="
                                    f"{_adj_ent_actual_pair_denominator_replicated}"
                                )
                        append_to_dict(
                            metrics,
                            {
                                "adj_ent/denominator_assertion_passed": 1.0,
                                "adj_ent/pre_ce_count_sp_replicated": _adj_ent_pre_global_C,
                                "adj_ent/pre_ce_count_unique": _adj_ent_expected_global_C,
                                "adj_ent/actual_ce_weighted_denominator": _adj_ent_actual_global_C,
                                "adj_ent/pair_count_assertion_passed": 1.0,
                                "adj_ent/pre_N_eligible_pair_global": _adj_ent_pre_N_pair_global,
                                "adj_ent/pre_N_total_pair_global": _adj_ent_pre_N_total_global,
                                "adj_ent/all_gated": 1.0 if _aec_all_gated else 0.0,
                                "adj_ent/response_level": float(
                                    response_level_aec_enable
                                ),
                            },
                        )

                    if _block_tpf_enable:
                        append_to_dict(metrics, block_tpf_metrics)

                    if not streaming_accumulate_only:
                        grad_norm = self._optimizer_step(
                            force_stream_trace_all_ce_sync=(
                                trace_all_ce_global_denominator_scale
                            )
                        )
                        total_grad_values = getattr(self, "_last_total_grad_values", 0)
                        nonfinite_grad_values = getattr(self, "_last_nonfinite_grad_values", 0)
                        mini_batch_metrics = {
                            "actor/grad_norm": grad_norm.detach().item(),
                            "actor/optimizer_steps": float(self._last_optimizer_step_applied),
                            "actor/optimizer_step_count": float(self._lifecycle_optimizer_step_count),
                            **getattr(self, "_last_sc_delta_metrics", {}),
                            "actor/nonfinite_grad_tensors": float(getattr(self, "_last_nonfinite_grad_tensors", 0)),
                            "actor/nonfinite_grad_values": float(nonfinite_grad_values),
                            "actor/nonfinite_grad_frac": (
                                float(nonfinite_grad_values) / float(total_grad_values) if total_grad_values else 0.0
                            ),
                            "actor/stream_replay/trace_all_ce_finish_grad_scale_pending": float(
                                getattr(self, "_last_stream_trace_all_ce_grad_scale_pending", 0.0)
                            ),
                            "actor/stream_replay/trace_all_ce_finish_grad_scale": float(
                                getattr(self, "_last_stream_trace_all_ce_grad_scale_factor", 1.0)
                            ),
                            "actor/stream_replay/trace_all_ce_finish_grad_scale_denominator": float(
                                getattr(self, "_last_stream_trace_all_ce_grad_scale_denominator", 0.0)
                            ),
                        }
                        # sc_update/* used to be written on the actor only and never read, so it
                        # was only visible as [SC-UPD] in the worker logs; report it through the
                        # actor metrics instead.
                        mini_batch_metrics.update(_sc_wandb_metrics(self))
                        append_to_dict(metrics, mini_batch_metrics)

                    # Update progress ONCE per mini-batch (no micro-batch progress bar).
                    pbar.update(1)
                    if show_progress:
                        pbar.set_postfix(
                            {
                                "epoch": f"{epoch + 1}/{self.config.ppo_epochs}",
                                "mini": f"{batch_idx + 1}/{len(mini_batches)}",
                            },
                            refresh=False,
                        )
        finally:
            pbar.close()
        if trace_all_ce_global_denominator_scale:
            _trace_ce_report_stats = torch.tensor(
                [
                    trace_all_ce_report_local_numerator,
                    trace_all_ce_report_local_denominator,
                ],
                device=get_device_id(),
                dtype=torch.float64,
            )
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(
                    _trace_ce_report_stats,
                    op=torch.distributed.ReduceOp.SUM,
                )
            _trace_ce_global_numerator = float(
                _trace_ce_report_stats[0].item()
            )
            _trace_ce_global_denominator = float(
                _trace_ce_report_stats[1].item()
            )
            _trace_ce_global_loss = (
                _trace_ce_global_numerator / _trace_ce_global_denominator
                if _trace_ce_global_denominator > 0.0
                else 0.0
            )
            metrics[
                "generation_consistency_jsd/ce_loss"
            ] = [_trace_ce_global_loss]
            if (
                float(
                    generation_jsd_config.get("entropy_coef", 0.0)
                    or 0.0
                )
                == 0.0
            ):
                metrics[
                    "generation_consistency_jsd/loss"
                ] = [_trace_ce_global_loss]
                metrics[
                    "generation_consistency_jsd/total_objective"
                ] = [_trace_ce_global_loss]
            metrics[
                "generation_consistency_jsd/weighted_numerator"
            ] = [_trace_ce_global_numerator]
            metrics[
                "generation_consistency_jsd/weighted_denominator"
            ] = [_trace_ce_global_denominator]
            metrics[
                "generation_consistency_jsd/active_tokens"
            ] = [_trace_ce_global_denominator]
            metrics[
                "actor/stream_replay/"
                "trace_all_ce_global_weighted_numerator"
            ] = [_trace_ce_global_numerator]
            metrics[
                "actor/stream_replay/"
                "trace_all_ce_global_weighted_denominator"
            ] = [_trace_ce_global_denominator]
            metrics[
                "actor/stream_replay/trace_all_ce_global_token_mean_loss"
            ] = [_trace_ce_global_loss]
            if response_level_ce_enable and response_level_weighting is None:
                if not _rl_ce_weight_prebuilt:
                    raise RuntimeError(
                        "response_level_ce weighting was not initialized"
                    )
                # Consumer-prebuilt weights: per-response denominators are baked
                # into the weight tensor; weighting-object metrics unavailable.
                metrics["response_level_ce/enabled"] = [1.0]
                metrics["response_level_ce/prebuilt_weights"] = [1.0]
                metrics["response_level_ce/global_weight_sum"] = [
                    _trace_ce_global_denominator
                ]
                metrics["response_level_ce/loss"] = [
                    _trace_ce_global_loss
                ]
            elif response_level_ce_enable:
                if (
                    abs(
                        _trace_ce_global_denominator
                        - float(response_level_weighting.global_responses)
                    )
                    > 1e-4
                ):
                    raise RuntimeError(
                        "response_level_ce global denominator mismatch: "
                        f"observed={_trace_ce_global_denominator} "
                        f"responses={response_level_weighting.global_responses}"
                    )
                metrics["response_level_ce/enabled"] = [1.0]
                metrics["response_level_ce/global_responses"] = [
                    float(response_level_weighting.global_responses)
                ]
                metrics["response_level_ce/global_active_tokens"] = [
                    float(response_level_weighting.global_active_tokens)
                ]
                metrics["response_level_ce/min_tokens_per_response"] = [
                    float(response_level_weighting.min_tokens_per_response)
                ]
                metrics["response_level_ce/max_tokens_per_response"] = [
                    float(response_level_weighting.max_tokens_per_response)
                ]
                metrics["response_level_ce/global_weight_sum"] = [
                    _trace_ce_global_denominator
                ]
                metrics["response_level_ce/loss"] = [
                    _trace_ce_global_loss
                ]
            if causal_prob_ce_filter_enable:
                metrics["causal_prob_ce_filter/enabled"] = [1.0]
                metrics["causal_prob_ce_filter/threshold"] = [
                    causal_prob_ce_filter_threshold
                ]
                metrics["causal_prob_ce_filter/direction_less_than"] = [
                    float(
                        causal_prob_ce_filter_direction == "less_than"
                    )
                ]
                metrics["causal_prob_ce_filter/global_active_trace_tokens"] = [
                    _trace_ce_global_denominator
                ]
                metrics["causal_prob_ce_filter/loss"] = [
                    _trace_ce_global_loss
                ]
            if adj_ent_enable and _adj_ent_pre_N_pair_global > 0:
                _adj_stats_local = torch.tensor(
                    [
                        float(_adj_ent_local_hinge_sum_report),
                        float(_adj_ent_local_pair_count_report),
                        float(_adj_ent_local_h_t_sum_report),
                        float(_adj_ent_local_h_next_sum_report),
                        float(_adj_ent_local_violation_count_report),
                        float(_adj_ent_local_rollout_next_sum_report),
                        float(_adj_ent_local_mismatch_signed_sum_report),
                        float(_adj_ent_local_mismatch_abs_sum_report),
                        float(_adj_ent_local_pair_weight_sum_report),
                    ],
                    dtype=torch.float64,
                    device=get_device_id(),
                )
                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(
                        _adj_stats_local,
                        op=torch.distributed.ReduceOp.SUM,
                    )
                _adj_global_hinge_sum = float(_adj_stats_local[0].item())
                _adj_global_pair_count = float(_adj_stats_local[1].item())
                _adj_global_h_t_sum = float(_adj_stats_local[2].item())
                _adj_global_h_next_sum = float(_adj_stats_local[3].item())
                _adj_global_violation_count = float(
                    _adj_stats_local[4].item()
                )
                _adj_global_rollout_next_sum = float(
                    _adj_stats_local[5].item()
                )
                _adj_global_mismatch_signed_sum = float(
                    _adj_stats_local[6].item()
                )
                _adj_global_mismatch_abs_sum = float(
                    _adj_stats_local[7].item()
                )
                _adj_global_pair_weight_sum = float(
                    _adj_stats_local[8].item()
                )
                _adj_global_hinge_sum_replicated = _adj_global_hinge_sum
                _adj_global_pair_count_replicated = _adj_global_pair_count
                _adj_global_hinge_sum /= float(_adj_ent_sp_size)
                _adj_global_pair_count /= float(_adj_ent_sp_size)
                _adj_global_h_t_sum /= float(_adj_ent_sp_size)
                _adj_global_h_next_sum /= float(_adj_ent_sp_size)
                _adj_global_violation_count /= float(_adj_ent_sp_size)
                _adj_global_rollout_next_sum /= float(_adj_ent_sp_size)
                _adj_global_mismatch_signed_sum /= float(
                    _adj_ent_sp_size
                )
                _adj_global_mismatch_abs_sum /= float(_adj_ent_sp_size)
                _adj_global_pair_weight_sum /= float(_adj_ent_sp_size)
                _adj_loss_denominator = (
                    _adj_global_pair_weight_sum
                    if response_level_aec_enable
                    else _adj_global_pair_count
                )
                _adj_raw_mean = (
                    _adj_global_hinge_sum / _adj_loss_denominator
                    if _adj_loss_denominator > 0
                    else 0.0
                )
                _aec_eff_coef = float(
                    getattr(self, "_aec_effective_coef", adj_ent_cfg.coef)
                )
                _adj_weighted_loss = _aec_eff_coef * _adj_raw_mean
                _total_objective_with_aec = (
                    _trace_ce_global_loss + _adj_weighted_loss
                )
                # CE-ratio controller state update: EMAs of the globally
                # reduced CE and raw AEC feed the NEXT step's coef_t. Skip
                # zero-valued steps so the coefficient holds instead of
                # exploding against a vanishing denominator.
                _rc_target = float(
                    _adj_ent_cfg_dict.get("ce_ratio_target", 0.0) or 0.0
                )
                if _rc_target > 0.0:
                    if _trace_ce_global_loss > 0.0 and _adj_raw_mean > 0.0:
                        _rc_beta = float(
                            _adj_ent_cfg_dict.get("ce_ratio_ema_beta", 0.9)
                        )
                        _rc_state = (
                            getattr(self, "_aec_ce_ratio_state", None) or {}
                        )
                        _rc_state["ce_ema"] = _rc_beta * _rc_state.get(
                            "ce_ema", _trace_ce_global_loss
                        ) + (1.0 - _rc_beta) * _trace_ce_global_loss
                        _rc_state["aec_ema"] = _rc_beta * _rc_state.get(
                            "aec_ema", _adj_raw_mean
                        ) + (1.0 - _rc_beta) * _adj_raw_mean
                        self._aec_ce_ratio_state = _rc_state
                    metrics["adj_ent/ce_ratio_target"] = [_rc_target]
                    metrics["adj_ent/ce_ratio_realized"] = [
                        _adj_weighted_loss / _trace_ce_global_loss
                        if _trace_ce_global_loss > 0.0
                        else 0.0
                    ]
                metrics["adj_ent/global_hinge_sum"] = [_adj_global_hinge_sum]
                metrics["adj_ent/global_pair_count"] = [_adj_global_pair_count]
                metrics["adj_ent/global_pair_weight_sum"] = [
                    _adj_global_pair_weight_sum
                ]
                metrics["adj_ent/response_level"] = [
                    float(response_level_aec_enable)
                ]
                metrics["adj_ent/pre_N_pair_global"] = [
                    _adj_ent_pre_N_pair_global / float(_adj_ent_sp_size)
                ]
                metrics["adj_ent/global_hinge_sum_sp_replicated"] = [
                    _adj_global_hinge_sum_replicated
                ]
                metrics["adj_ent/global_pair_count_sp_replicated"] = [
                    _adj_global_pair_count_replicated
                ]
                metrics["adj_ent/pre_N_pair_global_sp_replicated"] = [
                    _adj_ent_pre_N_pair_global
                ]
                metrics["adj_ent/raw_global_mean_loss"] = [_adj_raw_mean]
                metrics["adj_ent/weighted_loss"] = [_adj_weighted_loss]
                metrics["adj_ent/total_objective_with_ce"] = [
                    _total_objective_with_aec
                ]
                metrics[
                    "generation_consistency_jsd/total_objective"
                ] = [_total_objective_with_aec]
                metrics["adj_ent/coef"] = [_aec_eff_coef]
                metrics["adj_ent/rho"] = [adj_ent_cfg.rho]
                if response_level_aec_enable:
                    if response_level_aec_weighting is None:
                        raise RuntimeError(
                            "response-level AEC weighting was not initialized"
                        )
                    # _adj_ent_pre_N_pair_global is the ELIGIBLE pre-renorm
                    # pair-weight sum (≈ eligible responses for the 1.0-per-
                    # response equal-weight scheme), whereas
                    # response_level_aec_weighting.global_responses counts ALL
                    # responses (pre-gate). They legitimately differ when
                    # entropy gates are active, so we report both as
                    # diagnostics rather than raising on the difference.
                    _observed_aec_eligible_responses = (
                        _adj_ent_pre_N_pair_global
                        / float(_adj_ent_sp_size)
                    )
                    _aec_total_responses = float(
                        response_level_aec_weighting.global_responses
                    )
                    metrics["response_level_aec/global_responses_total"] = [
                        _aec_total_responses
                    ]
                    metrics["response_level_aec/global_responses_eligible"] = [
                        _observed_aec_eligible_responses
                    ]
                    metrics["response_level_aec/global_pairs"] = [
                        float(
                            response_level_aec_weighting.global_active_tokens
                        )
                    ]
                    metrics[
                        "response_level_aec/min_pairs_per_response"
                    ] = [
                        float(
                            response_level_aec_weighting.min_tokens_per_response
                        )
                    ]
                    metrics[
                        "response_level_aec/max_pairs_per_response"
                    ] = [
                        float(
                            response_level_aec_weighting.max_tokens_per_response
                        )
                    ]
                metrics["adj_ent/avg_entropy_t"] = [
                    _adj_global_h_t_sum / _adj_global_pair_count
                ]
                metrics["adj_ent/avg_entropy_next"] = [
                    _adj_global_h_next_sum / _adj_global_pair_count
                ]
                metrics["adj_ent/relative_entropy_drop"] = [
                    (
                        _adj_global_h_t_sum
                        - _adj_global_h_next_sum
                    )
                    / max(_adj_global_h_t_sum, adj_ent_cfg.eps)
                ]
                metrics["adj_ent/violation_rate"] = [
                    _adj_global_violation_count
                    / _adj_global_pair_count
                ]
                metrics["adj_ent/target_pass_rate"] = [
                    1.0
                    - _adj_global_violation_count
                    / _adj_global_pair_count
                ]
                metrics["adj_ent/rollout_avg_entropy_next"] = [
                    _adj_global_rollout_next_sum
                    / _adj_global_pair_count
                ]
                metrics["adj_ent/actor_rollout_next_bias"] = [
                    _adj_global_mismatch_signed_sum
                    / _adj_global_pair_count
                ]
                metrics["adj_ent/actor_rollout_next_mae"] = [
                    _adj_global_mismatch_abs_sum
                    / _adj_global_pair_count
                ]
                metrics["adj_ent/actor_rollout_next_relative_mae"] = [
                    _adj_global_mismatch_abs_sum
                    / max(
                        _adj_global_rollout_next_sum,
                        adj_ent_cfg.eps,
                    )
                ]
        if entropy_rkl_probe_enable:
            if entropy_rkl_probe_local_stats is None:
                raise RuntimeError(
                    "entropy_gated_rkl_probe collected no local statistics"
                )
            all_reduce_entropy_gated_rkl_probe_stats_(
                entropy_rkl_probe_local_stats
            )
            _probe_metrics = entropy_gated_rkl_probe_metrics(
                entropy_rkl_probe_local_stats,
                entropy_thresholds=entropy_rkl_probe_thresholds,
            )
            if (
                _probe_metrics[
                    "entropy_gate_rkl_probe/teacher_source_tokens"
                ]
                <= 0.0
            ):
                raise RuntimeError(
                    "entropy_gated_rkl_probe found zero global teacher-source "
                    "tokens"
                )
            for key, value in _probe_metrics.items():
                metrics[key] = [float(value)]
        if entropy_gated_rkl_enable:
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(
                    entropy_gated_rkl_local_report,
                    op=torch.distributed.ReduceOp.SUM,
                )
            _hybrid_ce_sum = float(
                entropy_gated_rkl_local_report[0].item()
            )
            _hybrid_ce_count = float(
                entropy_gated_rkl_local_report[1].item()
            )
            _hybrid_rkl_raw_sum = float(
                entropy_gated_rkl_local_report[2].item()
            )
            _hybrid_rkl_used_sum = float(
                entropy_gated_rkl_local_report[3].item()
            )
            _hybrid_rkl_count = float(
                entropy_gated_rkl_local_report[4].item()
            )
            _hybrid_cap_count = float(
                entropy_gated_rkl_local_report[5].item()
            )
            _hybrid_reported_sum = float(
                entropy_gated_rkl_local_report[6].item()
            )
            _hybrid_fkl_sum = float(
                entropy_gated_rkl_local_report[7].item()
            )
            _hybrid_cj_sum = float(
                entropy_gated_rkl_local_report[8].item()
            )
            _hybrid_total_count = _hybrid_ce_count + _hybrid_rkl_count
            if _hybrid_total_count <= 0.0:
                raise RuntimeError(
                    "entropy_gated_rkl found zero global objective tokens"
                )
            _hybrid_parity = entropy_gated_rkl_parity_terms(
                entropy_gated_rkl_local_report,
                cad_mode=entropy_gated_rkl_cad_mode,
                weighted=response_level_ce_enable,
            )
            if response_level_ce_enable:
                if _hybrid_parity["weight_sum"] <= 0.0:
                    raise RuntimeError(
                        "entropy_gated_rkl weighted accounting is empty"
                    )
                if abs(
                    _hybrid_parity["weight_sum"]
                    - _trace_ce_global_denominator
                ) > 1e-6 * max(_trace_ce_global_denominator, 1.0):
                    raise RuntimeError(
                        "entropy_gated_rkl weight sum does not match trace "
                        "denominator: "
                        f"hybrid={_hybrid_parity['weight_sum']} "
                        f"trace={_trace_ce_global_denominator}"
                    )
            elif (
                abs(_hybrid_total_count - _trace_ce_global_denominator)
                > 0.5
            ):
                raise RuntimeError(
                    "entropy_gated_rkl token count does not match trace "
                    "denominator: "
                    f"hybrid={_hybrid_total_count} "
                    f"trace={_trace_ce_global_denominator}"
                )
            _hybrid_numerator = _hybrid_parity["numerator"]
            _hybrid_loss = _hybrid_parity["loss"]
            _hybrid_parity_abs = abs(
                _hybrid_loss - _trace_ce_global_loss
            )
            if _hybrid_parity_abs > 1e-5 * max(
                abs(_trace_ce_global_loss),
                1.0,
            ):
                raise RuntimeError(
                    "entropy_gated_rkl global loss mismatch: "
                    f"components={_hybrid_loss} "
                    f"trace={_trace_ce_global_loss}"
                )
            _hybrid_metrics = {
                "entropy_gated_rkl/enabled": 1.0,
                "entropy_gated_rkl/entropy_threshold": (
                    entropy_gated_rkl_threshold
                ),
                "entropy_gated_rkl/entropy_threshold_nats": (
                    entropy_gated_rkl_threshold * math.log(151936)
                ),
                "entropy_gated_rkl/rkl_cap": entropy_gated_rkl_cap,
                "entropy_gated_rkl/ce_tokens": _hybrid_ce_count,
                "entropy_gated_rkl/rkl_tokens": _hybrid_rkl_count,
                "entropy_gated_rkl/rkl_fraction": (
                    _hybrid_rkl_count / _hybrid_total_count
                ),
                "entropy_gated_rkl/ce_mean": (
                    _hybrid_ce_sum / max(_hybrid_ce_count, 1.0)
                ),
                "entropy_gated_rkl/rkl_raw_mean": (
                    _hybrid_rkl_raw_sum
                    / max(_hybrid_rkl_count, 1.0)
                ),
                "entropy_gated_rkl/rkl_used_mean": (
                    _hybrid_rkl_used_sum
                    / max(_hybrid_rkl_count, 1.0)
                ),
                "entropy_gated_rkl/capped_tokens": _hybrid_cap_count,
                "entropy_gated_rkl/capped_fraction": (
                    _hybrid_cap_count / max(_hybrid_rkl_count, 1.0)
                ),
                "entropy_gated_rkl/hybrid_numerator": _hybrid_numerator,
                "entropy_gated_rkl/hybrid_loss": _hybrid_loss,
                "entropy_gated_rkl/reported_sum": _hybrid_reported_sum,
                "entropy_gated_rkl/global_loss_parity_abs": (
                    _hybrid_parity_abs
                ),
                "entropy_gated_rkl/cad_mode": (
                    1.0 if entropy_gated_rkl_cad_mode else 0.0
                ),
                "entropy_gated_rkl/response_level_weighted": (
                    1.0 if response_level_ce_enable else 0.0
                ),
                "entropy_gated_rkl/weighted_numerator": (
                    _hybrid_parity["weighted_loss_sum"]
                ),
                "entropy_gated_rkl/weight_sum": (
                    _hybrid_parity["weight_sum"]
                ),
            }
            if entropy_gated_rkl_cad_mode:
                _hybrid_metrics.update(
                    {
                        "entropy_gated_rkl/fkl_coef": (
                            entropy_gated_rkl_fkl_coef
                        ),
                        "entropy_gated_rkl/cj_power": (
                            entropy_gated_rkl_cj_power
                        ),
                        "entropy_gated_rkl/fkl_mean": (
                            _hybrid_fkl_sum
                            / max(_hybrid_rkl_count, 1.0)
                        ),
                        "entropy_gated_rkl/cj_mean": (
                            _hybrid_cj_sum
                            / max(_hybrid_rkl_count, 1.0)
                        ),
                    }
                )
            for key, value in _hybrid_metrics.items():
                metrics[key] = [float(value)]
        if not streaming_accumulate_only:
            self.actor_optimizer.zero_grad()
        return metrics


# ---------------------------------------------------------------------------
# Self-conditioning telemetry for any actor whose module tree has parameters
# named *self_conditioning*: sc_update/* = relative weight-update magnitude of
# the SC module vs. a backbone sample; sc_inject/* = ||SC residual|| /
# ||decoder input embeds||, accumulated either by the LLaDA2 overlay
# (`_sc_inject_acc` on modules) or by _install_gemma_sc_inject_hooks.
# Telemetry only: nothing here touches the loss or the optimizer.
# ---------------------------------------------------------------------------

_SC_UPDATE_MAX_BACKBONE_TENSORS = 8


def _sc_iter_named_params(actor):
    mod = getattr(actor, "actor_module", None)
    if mod is None:
        return []
    try:
        return list(mod.named_parameters())
    except Exception:  # noqa: BLE001
        return []


def _sc_snapshot_norms(actor):
    """Clone the SC parameters and a bounded backbone sample before optimizer.step()."""
    out = {"sc": [], "bb": []}
    try:
        for name, p in _sc_iter_named_params(actor):
            if p is None or not p.requires_grad:
                continue
            bucket = "sc" if "self_conditioning" in name else "bb"
            if bucket == "bb":
                # Rank-consistent sample: select by name only (sharded numel
                # differs per rank) and skip the giant tensors so the clone
                # stays small.
                if len(out["bb"]) >= _SC_UPDATE_MAX_BACKBONE_TENSORS or any(
                    k in name for k in ("embed_tokens", "lm_head", "experts", "vision")
                ):
                    continue
            out[bucket].append((name, p.detach().float().clone()))
    except Exception:  # noqa: BLE001
        return None
    return out


def _sc_record_update_magnitude(actor, pre):
    """After optimizer.step(): ||dW||/||W|| per bucket -> actor._sc_update_metrics."""
    if not pre:
        return
    try:
        cur = {n: p for n, p in _sc_iter_named_params(actor)}
        # Under FSDP(use_orig_params=True) each rank only holds its slice of
        # every parameter (often an empty slice for small tensors such as the
        # SC norms), so sum the squared norms across the process group before
        # dividing; every rank then reports the same global ratio.
        device = None
        for _name, p in cur.items():
            if p is not None and p.device.type == "cuda":
                device = p.device
                break
        sums = torch.zeros(4, dtype=torch.float64, device=device or "cpu")  # sc_num, sc_den, bb_num, bb_den
        for bi, bucket in enumerate(("sc", "bb")):
            for name, old in pre[bucket]:
                p = cur.get(name)
                if p is None:
                    continue
                new = p.detach().float()
                if new.shape != old.shape or new.numel() == 0:
                    continue
                sums[2 * bi] += (new - old).double().pow(2).sum()
                sums[2 * bi + 1] += old.double().pow(2).sum()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(sums, op=torch.distributed.ReduceOp.SUM)
        sc_num, sc_den, bb_num, bb_den = sums.tolist()
        rel_sc = (sc_num ** 0.5) / (sc_den ** 0.5) if sc_den > 0 else float("nan")
        rel_bb = (bb_num ** 0.5) / (bb_den ** 0.5) if bb_den > 0 else float("nan")
        m = getattr(actor, "_sc_update_metrics", None)
        if m is None:
            m = actor._sc_update_metrics = {}
        m["sc_update/rel_sc"] = rel_sc
        m["sc_update/rel_backbone"] = rel_bb
        m["sc_update/ratio"] = (
            rel_sc / rel_bb if rel_bb and rel_bb == rel_bb and rel_bb > 0
            else float("nan")
        )
        m["sc_update/sc_tensors"] = float(len(pre["sc"]))
        # Injection strength (residual : backbone activation), accumulated by
        # the LLaDA2 overlay on modules as `_sc_inject_acc`.
        try:
            _inj = None
            _mods = getattr(actor, "actor_module", None)
            for _mod in (_mods.modules() if _mods is not None else []):
                _a = getattr(_mod, "_sc_inject_acc", None)
                if _a is None:
                    continue
                if _inj is None:
                    _inj = [_a[0].clone(), _a[1].clone(), int(_a[2])]
                else:
                    _inj[0] = _inj[0] + _a[0]
                    _inj[1] = _inj[1] + _a[1]
                    _inj[2] = _inj[2] + int(_a[2])
                _mod._sc_inject_acc = None
            if _inj is not None and _inj[2] > 0:
                _rn = float(_inj[0].item()) ** 0.5
                _xn = float(_inj[1].item()) ** 0.5
                m["sc_inject/residual_norm"] = _rn
                m["sc_inject/embed_norm"] = _xn
                m["sc_inject/ratio"] = (_rn / _xn) if _xn > 0 else float("nan")
                m["sc_inject/calls"] = float(_inj[2])
                if int(torch.distributed.get_rank()) == 0:
                    print(
                        "[SC-INJ] ratio=%.4e residual=%.3e embed=%.3e calls=%d"
                        % (m["sc_inject/ratio"], _rn, _xn, _inj[2]),
                        flush=True,
                    )
        except Exception as _exc:  # noqa: BLE001
            print("[SC-INJ] skipped: %r" % (_exc,), flush=True)
        if int(torch.distributed.get_rank()) == 0:
            print(
                "[SC-UPD] rel_sc=%.3e rel_backbone=%.3e ratio=%.2f sc_tensors=%d"
                % (rel_sc, rel_bb, m["sc_update/ratio"], len(pre["sc"])),
                flush=True,
            )
    except Exception as exc:  # noqa: BLE001
        print("[SC-UPD] skipped: %r" % (exc,), flush=True)


def _sc_wandb_metrics(actor):
    """Hand _sc_update_metrics to the logger.

    Non-finite values are dropped (an arm without an adapter has rel_sc/ratio
    = nan), except sc_tensors, which is always reported: it is the validity
    switch for the other keys (0 means the name match found no SC params).
    """
    import math

    m = getattr(actor, "_sc_update_metrics", None)
    if not m:
        return {}
    out = {}
    for k, v in m.items():
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if k.endswith("sc_tensors") or math.isfinite(v):
            out[k] = v
    return out


def _install_gemma_sc_inject_hooks(sc_module):
    """Accumulate ||sc_signal||^2 (down_proj output) and ||inputs_embeds||^2 (SC module input) over
    the real rows of every replay wave; no host sync here. Returns the hook handles."""
    acc = {"r2": None, "x2": None, "calls": 0}

    def _rows(mod):
        return int(getattr(mod, "_sc_inject_rows", 0) or 0)

    def _pre(mod, args, kwargs=None):
        if not args or not torch.is_tensor(args[0]) or args[0].is_meta:
            return
        x = args[0]
        n = _rows(mod)
        x = x[:n] if 0 < n < x.shape[0] else x
        with torch.no_grad():
            v = x.detach().float().pow(2).sum()
        acc["x2"] = v if acc["x2"] is None else acc["x2"] + v
        acc["calls"] += 1

    def _down(mod, args, out):
        if not torch.is_tensor(out) or out.is_meta:
            return
        n = _rows(sc_module)
        r = out[:n] if 0 < n < out.shape[0] else out
        with torch.no_grad():
            v = r.detach().float().pow(2).sum()
        acc["r2"] = v if acc["r2"] is None else acc["r2"] + v

    handles = [sc_module.register_forward_pre_hook(_pre), sc_module.down_proj.register_forward_hook(_down)]
    return handles, acc


def _gemma_sc_inject_metrics(acc) -> dict[str, float]:
    if acc is None or acc["calls"] == 0 or acc["r2"] is None or acc["x2"] is None:
        return {}
    rn = float(acc["r2"].item()) ** 0.5
    xn = float(acc["x2"].item()) ** 0.5
    return {
        "sc_inject/residual_norm": rn,
        "sc_inject/embed_norm": xn,
        "sc_inject/ratio": (rn / xn) if xn > 0 else float("nan"),
        "sc_inject/calls": float(acc["calls"]),
    }
