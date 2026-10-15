"""
External-prefix-KV varlen attention for packed canvas segments.

For each segment i, canvas queries attend bidirectionally to:
    [ prefix_i_windowed keys/values | canvas_i keys/values ]

Segments are never mixed.  Supports GQA (Hq % Hkv == 0), bf16/fp16,
and full forward + backward (prefix gradients included).

Typical caller layout (DiffusionGemma, Qwen):
    - q_canvas  : canvas tokens only     [Tq_total,  Hq,  D]
    - k_canvas  : canvas KV              [Tq_total,  Hkv, D]
    - k_prefix  : shared read-only KV    [Tp_total,  Hkv, D]
    - cu_seqlens_q      : [B+1] cumulative canvas lengths
    - cu_seqlens_prefix : [B+1] cumulative prefix lengths

Implementation strategy
-----------------------
1. ``external_kv_varlen_attention_reference``  — pure PyTorch, float32,
   used for correctness testing.
2. ``external_kv_varlen_attention_flash``      — flash-attn
   ``flash_attn_varlen_func`` with separate cu_seqlens_q / cu_seqlens_k.
   Materialises combined [prefix | canvas] K/V per segment once; reports
   KV-pack time separately in benchmark mode.
3. ``external_kv_varlen_attention_triton``      — native no-copy Triton
   forward/backward backend.
4. ``external_kv_varlen_attention``             — public API with explicit
   ``auto``, ``triton``, ``flash``, and ``reference`` backend selection.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch

from .external_prefix_kv_triton import (
    external_kv_varlen_attention_triton,
    triton_backend_supported,
)

__all__ = [
    "external_kv_varlen_attention",
    "external_kv_varlen_attention_reference",
    "external_kv_varlen_attention_flash",
    "external_kv_varlen_attention_triton",
]

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_inputs(
    q_canvas: torch.Tensor,
    k_canvas: torch.Tensor,
    v_canvas: torch.Tensor,
    k_prefix: torch.Tensor,
    v_prefix: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_prefix: torch.Tensor,
    prefix_segment_ids: Optional[torch.Tensor],
) -> None:
    if q_canvas.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"q_canvas dtype must be float16 or bfloat16, got {q_canvas.dtype}"
        )
    if k_canvas.dtype != q_canvas.dtype or v_canvas.dtype != q_canvas.dtype:
        raise ValueError(
            "k_canvas and v_canvas must have the same dtype as q_canvas "
            f"({q_canvas.dtype}), got k={k_canvas.dtype}, v={v_canvas.dtype}"
        )
    if k_prefix.dtype != q_canvas.dtype or v_prefix.dtype != q_canvas.dtype:
        raise ValueError(
            "k_prefix and v_prefix must have the same dtype as q_canvas "
            f"({q_canvas.dtype}), got k_prefix={k_prefix.dtype}, v_prefix={v_prefix.dtype}"
        )
    if q_canvas.dim() != 3 or k_canvas.dim() != 3 or v_canvas.dim() != 3:
        raise ValueError("q_canvas, k_canvas, v_canvas must be 3-D [T, H, D]")
    if k_prefix.dim() != 3 or v_prefix.dim() != 3:
        raise ValueError("k_prefix, v_prefix must be 3-D [Tp, Hkv, D]")

    Hq = q_canvas.shape[1]
    Hkv = k_canvas.shape[1]
    if Hq % Hkv != 0:
        raise ValueError(
            f"n_q_heads ({Hq}) must be divisible by n_kv_heads ({Hkv}) (GQA)"
        )
    if k_canvas.shape != v_canvas.shape:
        raise ValueError(
            f"k_canvas and v_canvas must have the same shape, "
            f"got k={k_canvas.shape}, v={v_canvas.shape}"
        )
    if k_prefix.shape[1] != Hkv or v_prefix.shape[1] != Hkv:
        raise ValueError(
            f"k_prefix/v_prefix head dim must match k_canvas head dim ({Hkv}), "
            f"got {k_prefix.shape[1]}, {v_prefix.shape[1]}"
        )
    if q_canvas.shape[2] != k_canvas.shape[2]:
        raise ValueError(
            f"head_dim mismatch: q_canvas={q_canvas.shape[2]}, k_canvas={k_canvas.shape[2]}"
        )
    if k_prefix.shape != v_prefix.shape:
        raise ValueError(
            f"k_prefix and v_prefix must have the same shape, "
            f"got k_prefix={k_prefix.shape}, v_prefix={v_prefix.shape}"
        )
    if k_prefix.shape[2] != q_canvas.shape[2]:
        raise ValueError(
            f"prefix head_dim must match q_canvas ({q_canvas.shape[2]}), "
            f"got {k_prefix.shape[2]}"
        )
    if q_canvas.shape[0] != k_canvas.shape[0]:
        raise ValueError(
            f"canvas Q and K/V token counts must match, "
            f"got q={q_canvas.shape[0]}, k/v={k_canvas.shape[0]}"
        )
    tensors = (q_canvas, k_canvas, v_canvas, k_prefix, v_prefix)
    if any(t.device != q_canvas.device for t in tensors[1:]):
        raise ValueError("all Q/K/V tensors must be on the same device")
    if cu_seqlens_q.dim() != 1 or cu_seqlens_prefix.dim() != 1:
        raise ValueError("cu_seqlens_q and cu_seqlens_prefix must be 1-D")
    if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_prefix.dtype != torch.int32:
        raise ValueError("cu_seqlens_q and cu_seqlens_prefix must have dtype int32")
    if cu_seqlens_q.device != q_canvas.device or cu_seqlens_prefix.device != q_canvas.device:
        raise ValueError("cu_seqlens tensors must be on the same device as Q/K/V")
    _validate_prefix_segment_ids(
        cu_seqlens_q.numel() - 1,
        cu_seqlens_prefix,
        prefix_segment_ids,
        q_canvas.device,
    )


def _validate_prefix_segment_ids(
    num_query_segments: int,
    cu_seqlens_prefix: torch.Tensor,
    prefix_segment_ids: Optional[torch.Tensor],
    device: torch.device,
) -> None:
    """Validate query-segment to prefix-pool mapping."""
    num_prefix_segments = cu_seqlens_prefix.numel() - 1
    if prefix_segment_ids is None:
        if num_query_segments != num_prefix_segments:
            raise ValueError(
                "cu_seqlens_q and cu_seqlens_prefix must describe the same "
                "number of segments when prefix_segment_ids is None"
            )
        return
    if prefix_segment_ids.dim() != 1:
        raise ValueError("prefix_segment_ids must be 1-D")
    if prefix_segment_ids.numel() != num_query_segments:
        raise ValueError(
            f"prefix_segment_ids must have Bq={num_query_segments} entries, "
            f"got {prefix_segment_ids.numel()}"
        )
    if prefix_segment_ids.dtype != torch.int32:
        raise ValueError("prefix_segment_ids must have dtype int32")
    if prefix_segment_ids.device != device:
        raise ValueError("prefix_segment_ids must be on the same device as Q/K/V")
    if not prefix_segment_ids.is_contiguous():
        raise ValueError("prefix_segment_ids must be contiguous")


# ---------------------------------------------------------------------------
# Helper: apply optional prefix window, return per-segment (k, v) slices
# ---------------------------------------------------------------------------

def _prefix_slices(
    k_prefix: torch.Tensor,
    v_prefix: torch.Tensor,
    cu_seqlens_prefix: torch.Tensor,
    prefix_window: Optional[int],
    B: int,
    prefix_segment_ids: Optional[torch.Tensor] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Return per-segment prefix K/V slices, respecting prefix_window."""
    if prefix_window is not None and prefix_window < 0:
        raise ValueError(
            f"prefix_window must be non-negative or None, got {prefix_window}"
        )
    _validate_prefix_segment_ids(
        B, cu_seqlens_prefix, prefix_segment_ids, k_prefix.device
    )
    k_list: List[torch.Tensor] = []
    v_list: List[torch.Tensor] = []
    for i in range(B):
        prefix_segment = (
            i
            if prefix_segment_ids is None
            else int(prefix_segment_ids[i].item())
        )
        ps = int(cu_seqlens_prefix[prefix_segment].item())
        pe = int(cu_seqlens_prefix[prefix_segment + 1].item())
        if prefix_window is not None and (pe - ps) > prefix_window:
            ps = pe - prefix_window
        k_list.append(k_prefix[ps:pe])
        v_list.append(v_prefix[ps:pe])
    return k_list, v_list


# ---------------------------------------------------------------------------
# Pure-PyTorch reference (float32 accumulation, per-segment loop)
# ---------------------------------------------------------------------------

def external_kv_varlen_attention_reference(
    q_canvas: torch.Tensor,
    k_canvas: torch.Tensor,
    v_canvas: torch.Tensor,
    k_prefix: torch.Tensor,
    v_prefix: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_prefix: torch.Tensor,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_prefix: Optional[int] = None,
    scale: Optional[float] = None,
    prefix_window: Optional[int] = None,
    prefix_segment_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pure-PyTorch reference implementation of external_kv_varlen_attention.

    Processes each segment independently in a Python loop.  Float32 accumulation
    makes this suitable as a correctness oracle.  Not intended for production use.

    Parameters
    ----------
    q_canvas : Tensor [Tq, Hq, D]
        Packed canvas queries.
    k_canvas, v_canvas : Tensor [Tq, Hkv, D]
        Packed canvas keys/values.
    k_prefix, v_prefix : Tensor [Tp, Hkv, D]
        Packed prefix keys/values.
    cu_seqlens_q : Tensor [B+1] int32
        Cumulative canvas segment lengths.
    cu_seqlens_prefix : Tensor [M+1] int32
        Cumulative lengths for the unique prefix-segment pool.
    max_seqlen_q, max_seqlen_prefix : int or None
        Unused by reference; accepted for API symmetry.
    scale : float or None
        Attention softmax scale.  Default: ``1/sqrt(D)``.
    prefix_window : int or None
        If set, each segment only attends to its last ``prefix_window``
        prefix tokens.
    prefix_segment_ids : Tensor [B] int32 or None
        Maps each query segment to a prefix-pool segment.  ``None`` uses the
        identity mapping and requires ``M == B``.

    Returns
    -------
    Tensor [Tq, Hq, D]
        Attention output, same shape and dtype as q_canvas.
    """
    B = cu_seqlens_q.shape[0] - 1
    head_dim = q_canvas.shape[-1]
    Hq = q_canvas.shape[1]
    Hkv = k_canvas.shape[1]
    n_kv_groups = Hq // Hkv

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    k_p_list, v_p_list = _prefix_slices(
        k_prefix,
        v_prefix,
        cu_seqlens_prefix,
        prefix_window,
        B,
        prefix_segment_ids,
    )

    out_parts: List[torch.Tensor] = []
    for i in range(B):
        qs = int(cu_seqlens_q[i].item())
        qe = int(cu_seqlens_q[i + 1].item())

        # [Tq_i, Hq, D] in float32
        q_i = q_canvas[qs:qe].float()

        # Combined KV: [Tp_i_win + Tq_i, Hkv, D] in float32
        k_i = torch.cat([k_p_list[i], k_canvas[qs:qe]], dim=0).float()
        v_i = torch.cat([v_p_list[i], v_canvas[qs:qe]], dim=0).float()

        # GQA expansion
        if n_kv_groups > 1:
            k_i = k_i.repeat_interleave(n_kv_groups, dim=1)
            v_i = v_i.repeat_interleave(n_kv_groups, dim=1)

        # Bidirectional attention
        # qk: [Hq, Tq_i, Tkv_i]
        qk = torch.einsum("ihd,jhd->hij", q_i, k_i) * scale
        attn = torch.softmax(qk, dim=-1)
        # out_i: [Tq_i, Hq, D]
        out_i = torch.einsum("hij,jhd->ihd", attn, v_i)
        out_parts.append(out_i.to(q_canvas.dtype))

    return torch.cat(out_parts, dim=0)


# ---------------------------------------------------------------------------
# Flash-attn baseline (interleaves K/V per segment, one flash-attn call)
# ---------------------------------------------------------------------------

def external_kv_varlen_attention_flash(
    q_canvas: torch.Tensor,
    k_canvas: torch.Tensor,
    v_canvas: torch.Tensor,
    k_prefix: torch.Tensor,
    v_prefix: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_prefix: torch.Tensor,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_prefix: Optional[int] = None,
    scale: Optional[float] = None,
    prefix_window: Optional[int] = None,
    prefix_segment_ids: Optional[torch.Tensor] = None,
    _return_pack_ms: bool = False,
) -> torch.Tensor:
    """Flash-attn baseline for external_kv_varlen_attention.

    Materialises combined ``[prefix_mapped_i_windowed | canvas_i]`` K/V per
    query segment into a single packed tensor, then calls
    ``flash_attn_varlen_func`` with separate cumulative lengths.  Repeated
    mapping IDs therefore repeat the prefix data in this baseline.

    KV-pack cost: one ``torch.cat`` per segment plus a final ``torch.cat``
    across segments.  Set ``_return_pack_ms=True`` (internal benchmark flag)
    to also return the pack time in milliseconds.

    Requires: ``flash-attn >= 2.0``.
    """
    try:
        from flash_attn import flash_attn_varlen_func  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "flash-attn is required for external_kv_varlen_attention_flash. "
            "Install with: pip install flash-attn"
        ) from exc

    B = cu_seqlens_q.shape[0] - 1
    device = q_canvas.device
    head_dim = q_canvas.shape[-1]

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    if max_seqlen_q is None:
        max_seqlen_q = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item())

    # ---- KV-pack phase: build combined [prefix_i | canvas_i] per segment ----
    if _return_pack_ms and torch.cuda.is_available():
        pack_start = torch.cuda.Event(enable_timing=True)
        pack_end = torch.cuda.Event(enable_timing=True)
        pack_start.record()

    k_p_list, v_p_list = _prefix_slices(
        k_prefix,
        v_prefix,
        cu_seqlens_prefix,
        prefix_window,
        B,
        prefix_segment_ids,
    )

    k_parts: List[torch.Tensor] = []
    v_parts: List[torch.Tensor] = []
    kv_lens: List[int] = []
    for i in range(B):
        qs = int(cu_seqlens_q[i].item())
        qe = int(cu_seqlens_q[i + 1].item())
        k_i = torch.cat([k_p_list[i], k_canvas[qs:qe]], dim=0)
        v_i = torch.cat([v_p_list[i], v_canvas[qs:qe]], dim=0)
        k_parts.append(k_i)
        v_parts.append(v_i)
        kv_lens.append(k_i.shape[0])

    k_combined = torch.cat(k_parts, dim=0)
    v_combined = torch.cat(v_parts, dim=0)

    # Build cu_seqlens_k on the same device as q_canvas
    cu_seqlens_k = torch.zeros(B + 1, dtype=torch.int32, device=device)
    cumlen = 0
    for i, l in enumerate(kv_lens):
        cumlen += l
        cu_seqlens_k[i + 1] = cumlen
    max_seqlen_k = max(kv_lens)

    if _return_pack_ms and torch.cuda.is_available():
        pack_end.record()
        torch.cuda.synchronize()
        pack_ms = pack_start.elapsed_time(pack_end)
    else:
        pack_ms = 0.0

    # ---- Attention phase ----
    out = flash_attn_varlen_func(
        q_canvas,
        k_combined,
        v_combined,
        cu_seqlens_q.to(device),
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale=scale,
        causal=False,
    )

    if _return_pack_ms:
        return out, pack_ms  # type: ignore[return-value]
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def external_kv_varlen_attention(
    q_canvas: torch.Tensor,
    k_canvas: torch.Tensor,
    v_canvas: torch.Tensor,
    k_prefix: torch.Tensor,
    v_prefix: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_prefix: torch.Tensor,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_prefix: Optional[int] = None,
    scale: Optional[float] = None,
    prefix_window: Optional[int] = None,
    backend: str = "auto",
    prefix_segment_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Varlen attention where each canvas segment attends to its own prefix KV.

    For query segment ``i``, every canvas query sees all canvas keys/values in
    segment ``i`` **and** the selected tokens from prefix-pool segment
    ``prefix_segment_ids[i]`` — bidirectionally.  Cross-canvas-segment
    attention is never computed.

    ``backend="auto"`` selects the no-copy Triton backend for supported CUDA
    inputs.  Unsupported inputs fall back to flash-attn when possible, then
    to the pure-PyTorch reference.

    Parameters
    ----------
    q_canvas : Tensor [Tq, Hq, D]
        Packed canvas query tokens across all segments.
        dtype: ``float16`` or ``bfloat16``.
    k_canvas, v_canvas : Tensor [Tq, Hkv, D]
        Packed canvas key/value tokens.  GQA: ``Hq`` must be divisible by
        ``Hkv``.  Same dtype as ``q_canvas``.
    k_prefix, v_prefix : Tensor [Tp, Hkv, D]
        Packed prefix key/value tokens (read-only; gradients still flow
        unless the caller detaches them).  Same dtype as ``q_canvas``.
    cu_seqlens_q : Tensor [B+1] int32
        Cumulative canvas segment lengths, e.g. ``[0, Tq_0, Tq_0+Tq_1]``.
    cu_seqlens_prefix : Tensor [M+1] int32
        Cumulative lengths for ``M`` unique prefix-pool segments.
    max_seqlen_q : int or None
        Maximum canvas segment length.  Providing this avoids a GPU→CPU
        sync; **always pass it during training / benchmarking**.
    max_seqlen_prefix : int or None
        Maximum prefix segment length.  Providing this avoids a GPU→CPU sync
        in the Triton backend.  The flash backend computes the combined-KV
        maximum while packing.
    scale : float or None
        Softmax scale.  Default: ``1 / sqrt(D)``.
    prefix_window : int or None
        If set, each segment attends only to its **last** ``prefix_window``
        prefix tokens rather than the full prefix.  Useful for long-context
        efficiency.  ``None`` means full prefix.
    backend : {"auto", "triton", "flash", "reference"}
        Backend selection.  ``"triton"`` never materializes combined K/V;
        ``"flash"`` uses the materializing flash-attn baseline;
        ``"reference"`` uses pure PyTorch.
    prefix_segment_ids : Tensor [B] int32 or None
        Maps each query/canvas segment to one of the ``M`` prefix-pool
        segments.  IDs may repeat.  ``None`` preserves the identity mapping
        and requires ``M == B``.

    Returns
    -------
    Tensor [Tq, Hq, D]
        Attention output, same shape and dtype as ``q_canvas``.

    Notes
    -----
    * **Empty prefix**: a query segment mapped to an empty prefix-pool segment
      falls back to plain canvas self-attention.
    * **GQA**: ``Hq % Hkv == 0`` is required.
    * **Gradients**: backward is supported for all five Q/K/V tensors.
      DiffusionGemma may pass detached prefix K/V to skip their gradient work.

    Examples
    --------
    Single segment (B=1)::

        import torch
        from fast_diffusion_attention import external_kv_varlen_attention

        B, Tq, Tp, Hq, Hkv, D = 1, 256, 128, 16, 8, 128
        q = torch.randn(Tq, Hq, D, device="cuda", dtype=torch.bfloat16)
        k_c = torch.randn(Tq, Hkv, D, device="cuda", dtype=torch.bfloat16)
        v_c = torch.randn_like(k_c)
        k_p = torch.randn(Tp, Hkv, D, device="cuda", dtype=torch.bfloat16)
        v_p = torch.randn_like(k_p)

        cu_q = torch.tensor([0, Tq], dtype=torch.int32, device="cuda")
        cu_p = torch.tensor([0, Tp], dtype=torch.int32, device="cuda")

        out = external_kv_varlen_attention(q, k_c, v_c, k_p, v_p, cu_q, cu_p)
        # out: [256, 16, 128]

    Two segments with different prefix lengths and prefix_window::

        seglens_q  = [256, 128]
        seglens_p  = [512, 256]
        cu_q = torch.tensor([0, 256, 384], dtype=torch.int32, device="cuda")
        cu_p = torch.tensor([0, 512, 768], dtype=torch.int32, device="cuda")

        out = external_kv_varlen_attention(
            q, k_c, v_c, k_p, v_p, cu_q, cu_p,
            max_seqlen_q=256, prefix_window=128,
        )
    """
    _validate_inputs(
        q_canvas, k_canvas, v_canvas, k_prefix, v_prefix,
        cu_seqlens_q, cu_seqlens_prefix, prefix_segment_ids,
    )
    if prefix_window is not None and (
        not isinstance(prefix_window, int) or prefix_window < 0
    ):
        raise ValueError(
            f"prefix_window must be a non-negative int or None, got {prefix_window}"
        )
    if max_seqlen_q is not None and max_seqlen_q <= 0:
        raise ValueError("max_seqlen_q must be positive")
    if max_seqlen_prefix is not None and max_seqlen_prefix < 0:
        raise ValueError("max_seqlen_prefix must be non-negative")
    if backend not in {"auto", "triton", "flash", "reference"}:
        raise ValueError(
            "backend must be one of 'auto', 'triton', 'flash', or 'reference', "
            f"got {backend!r}"
        )

    common_args = (
        q_canvas,
        k_canvas,
        v_canvas,
        k_prefix,
        v_prefix,
        cu_seqlens_q,
        cu_seqlens_prefix,
        max_seqlen_q,
        max_seqlen_prefix,
        scale,
        prefix_window,
        prefix_segment_ids,
    )
    if backend == "triton":
        return external_kv_varlen_attention_triton(*common_args)
    if backend == "flash":
        return external_kv_varlen_attention_flash(*common_args)
    if backend == "reference":
        return external_kv_varlen_attention_reference(*common_args)

    supported, _ = triton_backend_supported(
        q_canvas,
        k_canvas,
        v_canvas,
        k_prefix,
        v_prefix,
        cu_seqlens_q,
        cu_seqlens_prefix,
        prefix_segment_ids,
    )
    if supported:
        return external_kv_varlen_attention_triton(*common_args)

    if q_canvas.is_cuda:
        try:
            return external_kv_varlen_attention_flash(*common_args)
        except ImportError:
            pass
    return external_kv_varlen_attention_reference(*common_args)
