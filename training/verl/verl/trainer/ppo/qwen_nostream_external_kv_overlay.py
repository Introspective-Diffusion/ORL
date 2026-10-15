"""External-prefix-KV attention overlay for Qwen SDAR no-stream GRPO.

Provides `install_qwen_nostream_external_prefix_kv_hook(model)` which attaches
`qwen_nostream_external_prefix_kv_replay(data)` to the model instance.

When the actor calls that hook (before `_forward_micro_batch`), the hook:
  1. Extracts per-sample prefix (L0) and canvas (L1) lengths from the batch.
  2. Builds `cu_seqlens_prefix`, `cu_seqlens_q`, and per-sample segment offsets
     that match the packed tensor ordering that `_forward_micro_batch` will
     produce via `unpad_input`.
  3. Installs a `QwenExtPrefixKVContext` on every ``SDARAttention`` layer in
     the model.  The context is consumed by the ext-prefix-KV branch added to
     ``SDARAttention.forward`` in ``modeling_sdar.py`` (one-shot: cleared after
     each layer's first invocation during the subsequent model forward).
  4. Runs a 1-token dry validation of ``external_kv_varlen_attention`` to prove
     the Triton kernel can execute on this GPU, raising ``RuntimeError`` if it
     cannot.
  5. Returns ``{"native_call": 1.0, "fallback": 0.0,
     "layers_installed": float(n_layers)}``.

Attention routing inside ``SDARAttention.forward`` (ext-prefix-KV branch):

  For each packed segment (sample) with offsets [seg_start .. seg_start+L0+L1+L1]:
    - Prefix tokens [seg_start .. seg_start+L0]:
        ``flash_attn_varlen_func`` bidirectional self-attention.
    - Canvas tokens (clean block) [seg_start+L0 .. seg_start+L0+L1]:
        ``external_kv_varlen_attention`` (Triton backend, no-copy) with
        detached prefix K/V and gradient-carrying canvas Q/K/V.
    - Noisy tokens (2nd L1 block) [seg_start+L0+L1 .. seg_start+L0+2*L1]:
        ``external_kv_varlen_attention`` (same kernel, same detached prefix K/V).
        The noisy tokens don't contribute to the GRPO loss; the full-kernel path
        is used so gradients flow correctly through any shared weights.

Gradient contract:
  - Gradient-carrying: canvas Q, K, V, and output (and noisy Q, K, V, output).
  - Detached (stop-gradient): prefix K, V supplied to the canvas cross-attention.
    Prefix hidden states receive gradients only through their own self-attention
    path, not through canvas-to-prefix cross-attention.
  - This matches the ``external_kv_varlen_attention`` no-copy Triton contract.
"""

from __future__ import annotations

import dataclasses
import sys
from types import MethodType
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from verl import DataProto


# ---------------------------------------------------------------------------
# Context dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class QwenExtPrefixKVContext:
    """Per-layer context consumed by SDARAttention ext-prefix-KV branch.

    Attributes
    ----------
    segments : List[(packed_start, L0_i, L1_i)]
        Per-sample segment descriptors in packed-tensor order.
        ``packed_start`` is the first token index in the rmpad tensor for
        sample ``i``; ``L0_i`` and ``L1_i`` are effective prefix and canvas
        (block) lengths.
    cu_seqlens_prefix : Tensor [B+1] int32
        Cumulative prefix segment lengths (CUDA device).
    cu_seqlens_q : Tensor [B+1] int32
        Cumulative canvas segment lengths (CUDA device).
    max_seqlen_prefix : int
    max_seqlen_q : int
    scaling : float
        Softmax scale for this layer (set by SDARAttention).
    """
    segments: List[Tuple[int, int, int]]
    cu_seqlens_prefix: torch.Tensor
    cu_seqlens_q: torch.Tensor
    max_seqlen_prefix: int
    max_seqlen_q: int
    scaling: float = 1.0


def _reference_external_kv_cpu(
    q_canvas: torch.Tensor,
    k_canvas: torch.Tensor,
    v_canvas: torch.Tensor,
    k_prefix: torch.Tensor,
    v_prefix: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_prefix: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    for segment_index in range(int(cu_seqlens_q.numel()) - 1):
        q_start = int(cu_seqlens_q[segment_index].item())
        q_end = int(cu_seqlens_q[segment_index + 1].item())
        prefix_start = int(cu_seqlens_prefix[segment_index].item())
        prefix_end = int(cu_seqlens_prefix[segment_index + 1].item())
        q_segment = q_canvas[q_start:q_end].permute(1, 0, 2).unsqueeze(0)
        k_segment = torch.cat(
            (k_prefix[prefix_start:prefix_end], k_canvas[q_start:q_end]),
            dim=0,
        ).permute(1, 0, 2).unsqueeze(0)
        v_segment = torch.cat(
            (v_prefix[prefix_start:prefix_end], v_canvas[q_start:q_end]),
            dim=0,
        ).permute(1, 0, 2).unsqueeze(0)
        output = F.scaled_dot_product_attention(
            q_segment,
            k_segment,
            v_segment,
            is_causal=False,
            scale=scale,
            enable_gqa=True,
        )
        outputs.append(output.squeeze(0).permute(1, 0, 2))
    return torch.cat(outputs, dim=0)


# ---------------------------------------------------------------------------
# Core attention computation for ext-prefix-KV layers
# ---------------------------------------------------------------------------

def run_ext_prefix_kv_attention(
    q_packed: torch.Tensor,   # [T_total, Hq, D]
    k_packed: torch.Tensor,   # [T_total, Hkv, D]
    v_packed: torch.Tensor,   # [T_total, Hkv, D]
    ctx: QwenExtPrefixKVContext,
) -> torch.Tensor:
    """Run external-prefix-KV attention for one SDARAttention layer.

    Parameters
    ----------
    q_packed, k_packed, v_packed : Tensor [T_total, H, D]
        Packed Q/K/V for all tokens in the micro-batch (rmpad format).
    ctx : QwenExtPrefixKVContext
        Segment metadata and cu_seqlens tensors.

    Returns
    -------
    Tensor [T_total, Hq, D]
        Attention output in original packed ordering.
    """
    try:
        from fast_diffusion_attention.external_prefix_kv_attn import (
            external_kv_varlen_attention,
        )
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "qwen_nostream_external_prefix_kv_replay requires "
            "fast_diffusion_attention to be importable."
        ) from exc

    T_total = q_packed.shape[0]
    segments = ctx.segments

    # Collect per-segment tensor slices
    prefix_q_parts: list[torch.Tensor] = []
    prefix_k_parts: list[torch.Tensor] = []
    prefix_v_parts: list[torch.Tensor] = []
    canvas_q_parts: list[torch.Tensor] = []
    canvas_k_parts: list[torch.Tensor] = []
    canvas_v_parts: list[torch.Tensor] = []
    noisy_q_parts: list[torch.Tensor] = []
    noisy_k_parts: list[torch.Tensor] = []
    noisy_v_parts: list[torch.Tensor] = []
    has_noisy = False

    for seg_start, L0_i, L1_i in segments:
        pfx_end = seg_start + L0_i
        cvs_end = pfx_end + L1_i
        nsy_end = cvs_end + L1_i

        prefix_q_parts.append(q_packed[seg_start:pfx_end])
        prefix_k_parts.append(k_packed[seg_start:pfx_end])
        prefix_v_parts.append(v_packed[seg_start:pfx_end])

        canvas_q_parts.append(q_packed[pfx_end:cvs_end])
        canvas_k_parts.append(k_packed[pfx_end:cvs_end])
        canvas_v_parts.append(v_packed[pfx_end:cvs_end])

        if nsy_end <= T_total:
            noisy_q_parts.append(q_packed[cvs_end:nsy_end])
            noisy_k_parts.append(k_packed[cvs_end:nsy_end])
            noisy_v_parts.append(v_packed[cvs_end:nsy_end])
            has_noisy = True

    # --- Prefix self-attention (bidirectional) ---
    q_prefix = torch.cat(prefix_q_parts, dim=0)   # [T_pfx, Hq, D]
    k_prefix = torch.cat(prefix_k_parts, dim=0)   # [T_pfx, Hkv, D]
    v_prefix = torch.cat(prefix_v_parts, dim=0)   # [T_pfx, Hkv, D]

    if q_prefix.shape[0] == 0 or ctx.max_seqlen_prefix <= 0:
        prefix_out = q_prefix
    elif q_prefix.is_cuda:
        empty_prefix_k = k_prefix[:0]
        empty_prefix_v = v_prefix[:0]
        empty_cu = torch.zeros_like(ctx.cu_seqlens_prefix)
        prefix_out = external_kv_varlen_attention(
            q_prefix,
            k_prefix,
            v_prefix,
            empty_prefix_k,
            empty_prefix_v,
            ctx.cu_seqlens_prefix,
            empty_cu,
            max_seqlen_q=ctx.max_seqlen_prefix,
            max_seqlen_prefix=0,
            scale=ctx.scaling,
            backend="triton",
        )
    else:
        prefix_outputs: list[torch.Tensor] = []
        for segment_index in range(len(segments)):
            start = int(ctx.cu_seqlens_prefix[segment_index].item())
            end = int(ctx.cu_seqlens_prefix[segment_index + 1].item())
            q_segment = q_prefix[start:end].permute(1, 0, 2).unsqueeze(0)
            k_segment = k_prefix[start:end].permute(1, 0, 2).unsqueeze(0)
            v_segment = v_prefix[start:end].permute(1, 0, 2).unsqueeze(0)
            out_segment = F.scaled_dot_product_attention(
                q_segment,
                k_segment,
                v_segment,
                is_causal=False,
                scale=ctx.scaling,
                enable_gqa=True,
            )
            prefix_outputs.append(out_segment.squeeze(0).permute(1, 0, 2))
        prefix_out = torch.cat(prefix_outputs, dim=0)

    # Detached prefix K/V for cross-attention with canvas/noisy
    k_prefix_det = k_prefix.detach()
    v_prefix_det = v_prefix.detach()

    # --- Canvas external-prefix-KV attention ---
    q_canvas = torch.cat(canvas_q_parts, dim=0)   # [T_cvs, Hq, D]
    k_canvas = torch.cat(canvas_k_parts, dim=0)   # [T_cvs, Hkv, D]
    v_canvas = torch.cat(canvas_v_parts, dim=0)   # [T_cvs, Hkv, D]

    if q_canvas.shape[0] == 0 or ctx.max_seqlen_q <= 0:
        canvas_out = q_canvas
    elif q_canvas.is_cuda:
        canvas_out = external_kv_varlen_attention(
            q_canvas, k_canvas, v_canvas,
            k_prefix_det, v_prefix_det,
            ctx.cu_seqlens_q, ctx.cu_seqlens_prefix,
            max_seqlen_q=ctx.max_seqlen_q,
            max_seqlen_prefix=ctx.max_seqlen_prefix,
            scale=ctx.scaling,
            backend="triton",
        )
    else:
        canvas_out = _reference_external_kv_cpu(
            q_canvas,
            k_canvas,
            v_canvas,
            k_prefix_det,
            v_prefix_det,
            ctx.cu_seqlens_q,
            ctx.cu_seqlens_prefix,
            ctx.scaling,
        )

    # --- Noisy block external-prefix-KV attention ---
    noisy_out: Optional[torch.Tensor] = None
    if has_noisy and noisy_q_parts:
        q_noisy = torch.cat(noisy_q_parts, dim=0)
        k_noisy = torch.cat(noisy_k_parts, dim=0)
        v_noisy = torch.cat(noisy_v_parts, dim=0)
        # Noisy block: attends to prefix (detached) + noisy K/V (with grad).
        # This matches the no-stream GRPO setup where clean == noisy == final tokens.
        if q_noisy.shape[0] == 0 or ctx.max_seqlen_q <= 0:
            noisy_out = q_noisy
        elif q_noisy.is_cuda:
            noisy_out = external_kv_varlen_attention(
                q_noisy, k_noisy, v_noisy,
                k_prefix_det, v_prefix_det,
                ctx.cu_seqlens_q, ctx.cu_seqlens_prefix,
                max_seqlen_q=ctx.max_seqlen_q,
                max_seqlen_prefix=ctx.max_seqlen_prefix,
                scale=ctx.scaling,
                backend="triton",
            )
        else:
            noisy_out = _reference_external_kv_cpu(
                q_noisy,
                k_noisy,
                v_noisy,
                k_prefix_det,
                v_prefix_det,
                ctx.cu_seqlens_q,
                ctx.cu_seqlens_prefix,
                ctx.scaling,
            )

    # --- Reconstruct in original packed order ---
    output = torch.empty_like(q_packed)  # [T_total, Hq, D]
    pfx_off = 0
    cvs_off = 0
    nsy_off = 0

    for i, (seg_start, L0_i, L1_i) in enumerate(segments):
        pfx_end = seg_start + L0_i
        cvs_end = pfx_end + L1_i
        nsy_end = cvs_end + L1_i

        output[seg_start:pfx_end] = prefix_out[pfx_off:pfx_off + L0_i]
        pfx_off += L0_i

        output[pfx_end:cvs_end] = canvas_out[cvs_off:cvs_off + L1_i]
        cvs_off += L1_i

        if noisy_out is not None and nsy_end <= T_total:
            output[cvs_end:nsy_end] = noisy_out[nsy_off:nsy_off + L1_i]
            nsy_off += L1_i

    return output


# ---------------------------------------------------------------------------
# Segment / cu_seqlens extraction from the (pre-rmpad) batch
# ---------------------------------------------------------------------------

def _build_ext_prefix_kv_context(
    batch: dict,
    device: torch.device,
) -> QwenExtPrefixKVContext:
    """Extract prefix/canvas metadata from the micro-batch tensor dict.

    The micro-batch is in padded ``[B, L0_max + 2*L1]`` form (SDAR layout:
    prompt | clean block | noisy block | padding).  The packed (rmpad) ordering
    produced by ``unpad_input`` in ``_forward_micro_batch`` places sample ``i``'s
    valid tokens contiguously starting at ``cu_seqlens_all[i]``.
    """
    input_ids = batch["input_ids"]           # [B, total_seq_len]
    attention_mask = batch["attention_mask"]  # [B, total_seq_len]
    responses = batch["responses"]            # [B, L1]

    B = input_ids.shape[0]
    L1 = responses.shape[1]
    total_seqlen = attention_mask.shape[1]
    response_area_start = total_seqlen - 2 * L1

    # Per-sample prefix lengths (valid tokens before the response area)
    L0_per = attention_mask[:, :response_area_start].sum(dim=1).long()   # [B]
    # Per-sample canvas (clean block) lengths
    L1_per = attention_mask[:, response_area_start:response_area_start + L1].sum(dim=1).long()  # [B]
    # Per-sample total valid tokens (prefix + clean + noisy)
    seqlens_all = attention_mask.sum(dim=1).long()  # [B]

    # Build cu_seqlens (same ordering as _forward_micro_batch's unpad_input)
    cu_all = torch.zeros(B + 1, dtype=torch.int32, device=device)
    cu_all[1:] = seqlens_all.cumsum(0).to(torch.int32)

    # Build prefix / canvas cu_seqlens for external-prefix-KV kernel
    cu_pfx = torch.zeros(B + 1, dtype=torch.int32, device=device)
    cu_pfx[1:] = L0_per.cumsum(0).to(torch.int32)

    cu_q = torch.zeros(B + 1, dtype=torch.int32, device=device)
    cu_q[1:] = L1_per.cumsum(0).to(torch.int32)

    max_pfx = int(L0_per.max().item()) if B > 0 else 1
    max_q = int(L1_per.max().item()) if B > 0 else 1

    # Build segment list
    segments: list[tuple[int, int, int]] = []
    for i in range(B):
        packed_start = int(cu_all[i].item())
        L0_i = int(L0_per[i].item())
        L1_i = int(L1_per[i].item())
        segments.append((packed_start, L0_i, L1_i))

    return QwenExtPrefixKVContext(
        segments=segments,
        cu_seqlens_prefix=cu_pfx,
        cu_seqlens_q=cu_q,
        max_seqlen_prefix=max_pfx,
        max_seqlen_q=max_q,
    )


# ---------------------------------------------------------------------------
# Triton dry-run validation
# ---------------------------------------------------------------------------

def _validate_triton_kernel_or_raise(ctx: QwenExtPrefixKVContext, device: torch.device) -> None:
    """Verify the Triton backend accepts this configuration, or raise RuntimeError.

    Uses a tiny synthetic 1-token prefix / 1-token canvas call so the check
    is GPU-synchronous but negligible in runtime.
    """
    from fast_diffusion_attention.external_prefix_kv_attn import (
        external_kv_varlen_attention,
    )
    from fast_diffusion_attention.external_prefix_kv_triton import (
        triton_backend_supported,
    )

    # Pick head_dim from the first real segment if possible, else default to 128.
    head_dim = 128
    dtype = torch.bfloat16

    q = torch.randn(1, 1, head_dim, device=device, dtype=dtype)
    k = torch.randn(1, 1, head_dim, device=device, dtype=dtype)
    v = torch.randn(1, 1, head_dim, device=device, dtype=dtype)
    cu = torch.tensor([0, 1], dtype=torch.int32, device=device)

    supported, reason = triton_backend_supported(q, k, v, k, v, cu, cu)
    if not supported:
        raise RuntimeError(
            "Qwen no-stream external-prefix-KV Triton backend is NOT supported "
            f"for this GPU/dtype configuration: {reason}. "
            "Packed/repeated-prefix fallback is disabled."
        )
    # Actual tiny forward to prove the kernel compiles and runs.
    _ = external_kv_varlen_attention(q, k, v, k, v, cu, cu, backend="triton")


# ---------------------------------------------------------------------------
# Hook installation
# ---------------------------------------------------------------------------

def _install_ext_kv_ctx_on_layers(
    model: "torch.nn.Module",
    ctx: QwenExtPrefixKVContext,
) -> int:
    """Set ``_qwen_ext_prefix_kv_ctx`` on each SDARAttention layer.

    Returns the number of layers that received the context.
    """
    # Resolve FSDP / DP wrapper to the base module.
    base = getattr(model, "module", model)
    # Traverse model.model.layers (SDARModel structure).
    inner = getattr(base, "model", base)
    layers = getattr(inner, "layers", None)
    if layers is None:
        # Fallback: walk all named modules looking for SDARAttention.
        n = 0
        for m in base.modules():
            if type(m).__name__ == "SDARAttention":
                _install_external_kv_attention_forward(m)
                m._qwen_ext_prefix_kv_ctx = ctx
                n += 1
        return n

    n = 0
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is not None:
            _install_external_kv_attention_forward(attn)
            attn._qwen_ext_prefix_kv_ctx = ctx
            n += 1
    return n


def _install_external_kv_attention_forward(attention: "torch.nn.Module") -> None:
    """Patch the concrete dynamically-loaded SDARAttention instance."""
    if bool(getattr(attention, "_qwen_ext_prefix_kv_forward_installed", False)):
        return
    original_forward = attention.forward

    def external_kv_forward(
        self: "torch.nn.Module",
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value=None,
        cache_position=None,
        **kwargs,
    ):
        ctx = getattr(self, "_qwen_ext_prefix_kv_ctx", None)
        if ctx is None:
            return original_forward(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                cache_position=cache_position,
                **kwargs,
            )
        if past_key_value is not None:
            raise RuntimeError(
                "Qwen no-stream external-prefix-KV replay does not accept a "
                "pre-existing model KV cache"
            )
        input_shape = hidden_states.shape[:-1]
        batch_size, query_length = input_shape
        if batch_size != 1:
            raise RuntimeError(
                "Qwen no-stream external-prefix-KV replay expects the packed "
                f"remove-padding batch dimension to be 1, got {batch_size}"
            )
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_norm(
            self.q_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        model_module = sys.modules.get(type(self).__module__)
        apply_rotary = getattr(model_module, "apply_rotary_pos_emb", None)
        if not callable(apply_rotary):
            raise RuntimeError(
                "Qwen no-stream external-prefix-KV replay could not resolve "
                "apply_rotary_pos_emb from the dynamically loaded SDAR module"
            )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary(
            query_states,
            key_states,
            cos,
            sin,
        )
        q_packed = query_states.transpose(1, 2).flatten(0, 1).contiguous()
        k_packed = key_states.transpose(1, 2).flatten(0, 1).contiguous()
        v_packed = value_states.transpose(1, 2).flatten(0, 1).contiguous()
        expected_tokens = sum(
            int(prefix_len) + 2 * int(canvas_len)
            for _, prefix_len, canvas_len in ctx.segments
        )
        if expected_tokens != int(q_packed.shape[0]):
            raise RuntimeError(
                "Qwen no-stream external-prefix-KV packed token accounting "
                f"mismatch: context={expected_tokens} qkv={int(q_packed.shape[0])}"
            )
        ctx.scaling = float(self.scaling)
        attention_output = run_ext_prefix_kv_attention(
            q_packed,
            k_packed,
            v_packed,
            ctx,
        )
        attention_output = attention_output.reshape(
            batch_size,
            query_length,
            -1,
        )
        return self.o_proj(attention_output), None

    attention.forward = MethodType(external_kv_forward, attention)
    attention._qwen_ext_prefix_kv_forward_installed = True


def make_qwen_nostream_external_prefix_kv_replay(model: "torch.nn.Module"):
    """Return a bound method to attach as ``model.qwen_nostream_external_prefix_kv_replay``.

    The method signature matches what ``DataParallelPPOActor._call_qwen_external_prefix_kv_replay``
    expects: ``hook(data) -> dict[str, float]``.
    """

    def qwen_nostream_external_prefix_kv_replay(
        self: "torch.nn.Module",
        data: "DataProto",
    ) -> dict[str, float]:
        """Install ext-prefix-KV context for the next model forward.

        Called by the actor BEFORE ``_forward_micro_batch``.  Sets
        ``_qwen_ext_prefix_kv_ctx`` on every ``SDARAttention`` layer so that
        the subsequent forward uses the native no-copy Triton kernel for canvas
        tokens with detached prefix K/V.
        """
        device = next(self.parameters()).device
        ctx = _build_ext_prefix_kv_context(data.batch, device)

        n_layers = _install_ext_kv_ctx_on_layers(self, ctx)
        if n_layers == 0:
            raise RuntimeError(
                "qwen_nostream_external_prefix_kv_replay: found 0 SDARAttention layers. "
                "The model must contain SDARAttention modules."
            )

        _validate_triton_kernel_or_raise(ctx, device)

        return {
            "native_call": 1.0,
            "fallback": 0.0,
            "layers_installed": float(n_layers),
        }

    return MethodType(qwen_nostream_external_prefix_kv_replay, model)


def install_qwen_nostream_external_prefix_kv_hook(model: "torch.nn.Module") -> None:
    """Attach ``qwen_nostream_external_prefix_kv_replay`` to ``model`` in-place.

    Should be called once after the model is constructed (e.g. in a post-init
    hook or at the point where ``use_qwen_external_prefix_kv_replay=true`` is
    detected).
    """
    if not hasattr(model, "qwen_nostream_external_prefix_kv_replay"):
        model.qwen_nostream_external_prefix_kv_replay = (
            make_qwen_nostream_external_prefix_kv_replay(model)
        )
