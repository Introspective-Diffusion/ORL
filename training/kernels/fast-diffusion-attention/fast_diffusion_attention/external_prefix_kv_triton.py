"""No-copy Triton backend for external-prefix-KV varlen attention.

The kernels operate directly on separate packed prefix and canvas K/V tensors.
No combined ``[prefix | canvas]`` K/V tensor is materialized.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


_TRITON_HEAD_DIMS = (16, 32, 64, 128, 256, 512)


def triton_backend_supported(
    q_canvas: torch.Tensor,
    k_canvas: torch.Tensor,
    v_canvas: torch.Tensor,
    k_prefix: torch.Tensor,
    v_prefix: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_prefix: torch.Tensor,
    prefix_segment_ids: Optional[torch.Tensor] = None,
) -> Tuple[bool, str]:
    """Return whether the native Triton backend supports these inputs."""
    tensors = (q_canvas, k_canvas, v_canvas, k_prefix, v_prefix)
    if not all(t.is_cuda for t in tensors):
        return False, "all Q/K/V tensors must be CUDA tensors"
    if q_canvas.device.type != "cuda":
        return False, "q_canvas must be on CUDA"
    if any(t.device != q_canvas.device for t in tensors[1:]):
        return False, "all Q/K/V tensors must be on the same CUDA device"
    if q_canvas.dtype not in (torch.float16, torch.bfloat16):
        return False, "only float16 and bfloat16 are supported"
    if any(t.dtype != q_canvas.dtype for t in tensors[1:]):
        return False, "all Q/K/V tensors must have the same dtype"
    if q_canvas.ndim != 3 or any(t.ndim != 3 for t in tensors[1:]):
        return False, "all Q/K/V tensors must be three-dimensional"
    if k_canvas.shape != v_canvas.shape:
        return False, "canvas K and V shapes must match"
    if k_prefix.shape != v_prefix.shape:
        return False, "prefix K and V shapes must match"
    if q_canvas.shape[0] != k_canvas.shape[0]:
        return False, "canvas Q and K/V token counts must match"
    if k_canvas.shape[1] == 0 or q_canvas.shape[1] % k_canvas.shape[1] != 0:
        return False, "the query-head count must be divisible by the KV-head count"
    if (
        q_canvas.shape[2] != k_canvas.shape[2]
        or q_canvas.shape[2] != k_prefix.shape[2]
        or k_canvas.shape[1] != k_prefix.shape[1]
    ):
        return False, "canvas and prefix K/V head shapes must match Q"
    if q_canvas.shape[-1] not in _TRITON_HEAD_DIMS:
        return False, f"head_dim must be one of {_TRITON_HEAD_DIMS}"
    if any(t.stride(-1) != 1 for t in tensors):
        return False, "the head dimension must be contiguous"
    if not cu_seqlens_q.is_cuda or not cu_seqlens_prefix.is_cuda:
        return False, "cu_seqlens tensors must be CUDA tensors"
    if (
        cu_seqlens_q.device != q_canvas.device
        or cu_seqlens_prefix.device != q_canvas.device
    ):
        return False, "cu_seqlens tensors must be on the Q/K/V device"
    if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_prefix.dtype != torch.int32:
        return False, "cu_seqlens tensors must have dtype int32"
    if not cu_seqlens_q.is_contiguous() or not cu_seqlens_prefix.is_contiguous():
        return False, "cu_seqlens tensors must be contiguous"
    if cu_seqlens_q.numel() < 2:
        return False, "at least one segment is required"
    num_query_segments = cu_seqlens_q.numel() - 1
    if prefix_segment_ids is None:
        if cu_seqlens_q.numel() != cu_seqlens_prefix.numel():
            return False, "identity mapping requires matching segment counts"
    else:
        if not prefix_segment_ids.is_cuda:
            return False, "prefix_segment_ids must be a CUDA tensor"
        if prefix_segment_ids.device != q_canvas.device:
            return False, "prefix_segment_ids must be on the Q/K/V device"
        if prefix_segment_ids.dtype != torch.int32:
            return False, "prefix_segment_ids must have dtype int32"
        if prefix_segment_ids.ndim != 1:
            return False, "prefix_segment_ids must be one-dimensional"
        if prefix_segment_ids.numel() != num_query_segments:
            return False, "prefix_segment_ids must have one entry per query segment"
        if not prefix_segment_ids.is_contiguous():
            return False, "prefix_segment_ids must be contiguous"
        if k_prefix.requires_grad or v_prefix.requires_grad:
            return (
                False,
                "mapped trainable prefix K/V requires the flash or reference backend",
            )
    if q_canvas.shape[0] == 0:
        return False, "at least one canvas query is required"
    if torch.cuda.get_device_capability(q_canvas.device)[0] < 8:
        return False, "compute capability 8.0 or newer is required"
    return True, ""


def _forward_config(
    head_dim: int,
    n_q_heads: int,
    n_kv_heads: int,
    device: torch.device,
) -> Tuple[int, int, int, int]:
    """Return BLOCK_M, BLOCK_N, warps, stages for forward."""
    cc_major, _ = torch.cuda.get_device_capability(device)
    if head_dim >= 512:
        return 16, 16, 8, 1
    if head_dim >= 256:
        return 32, 32, 8, 1
    if cc_major >= 9 and head_dim == 128:
        return 128, 128, 8, 2
    if head_dim == 128:
        return 128, 64, 8, 2
    if head_dim == 64:
        return 128, 64, 4, 2
    return 64, 64, 4, 2


def _dq_config(
    head_dim: int,
    n_q_heads: int,
    n_kv_heads: int,
    device: torch.device,
) -> Tuple[int, int, int, int]:
    """Return BLOCK_M, BLOCK_N, warps, stages for dQ."""
    cc_major, _ = torch.cuda.get_device_capability(device)
    if head_dim >= 512:
        return 16, 16, 8, 1
    if head_dim >= 256:
        return 32, 32, 8, 1
    if cc_major >= 9 and head_dim == 128:
        return 128, 128, 8, 2
    if head_dim == 128:
        return 64, 64, 8, 2
    if head_dim == 64:
        return 64, 64, 4, 2
    return 64, 64, 4, 1


def _dkdv_config(
    head_dim: int,
    n_q_heads: int,
    n_kv_heads: int,
    device: torch.device,
) -> Tuple[int, int, int, int]:
    """Return BLOCK_M, BLOCK_N, warps, stages for dK/dV."""
    if head_dim >= 512:
        return 16, 16, 8, 1
    if head_dim >= 256:
        return 32, 32, 8, 1
    if head_dim == 128:
        return 64, 64, 4, 1
    return 64, 64, 4, 1


@triton.jit
def _external_prefix_fwd_kernel(
    Q,
    KCanvas,
    VCanvas,
    KPrefix,
    VPrefix,
    Out,
    LSE,
    CuQ,
    CuPrefix,
    PrefixSegmentIds,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kct,
    stride_kch,
    stride_kcd,
    stride_vct,
    stride_vch,
    stride_vcd,
    stride_kpt,
    stride_kph,
    stride_kpd,
    stride_vpt,
    stride_vph,
    stride_vpd,
    stride_ot,
    stride_oh,
    stride_od,
    n_q_heads: tl.constexpr,
    sm_scale,
    HAS_PREFIX_SEGMENT_IDS: tl.constexpr,
    PREFIX_WINDOW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    N_KV_GROUPS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_segment = tl.program_id(2)
    pid_kv = pid_h // N_KV_GROUPS

    q_start = tl.load(CuQ + pid_segment).to(tl.int32)
    q_end = tl.load(CuQ + pid_segment + 1).to(tl.int32)
    q_len = q_end - q_start
    if HAS_PREFIX_SEGMENT_IDS:
        prefix_segment = tl.load(PrefixSegmentIds + pid_segment).to(tl.int32)
    else:
        prefix_segment = pid_segment
    p_start = tl.load(CuPrefix + prefix_segment).to(tl.int32)
    p_end = tl.load(CuPrefix + prefix_segment + 1).to(tl.int32)
    p_len_raw = p_end - p_start
    if PREFIX_WINDOW >= 0:
        p_len = tl.minimum(p_len_raw, PREFIX_WINDOW)
        p_selected_start = p_end - p_len
    else:
        p_len = p_len_raw
        p_selected_start = p_start

    start_m = pid_m * BLOCK_M
    if start_m >= q_len:
        return

    offs_m = start_m + tl.arange(0, BLOCK_M)
    m_valid = offs_m < q_len
    q_abs = q_start + start_m

    q_base = Q + pid_h * stride_qh
    q_block_ptr = tl.make_block_ptr(
        base=q_base,
        shape=(q_end, BLOCK_D),
        strides=(stride_qt, stride_qd),
        offsets=(q_abs, 0),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0),
    )
    q = tl.load(q_block_ptr, boundary_check=(0,))
    q = (q * sm_scale).to(q.dtype)

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    kp_base = KPrefix + pid_kv * stride_kph
    vp_base = VPrefix + pid_kv * stride_vph
    num_prefix_tiles = (p_len + BLOCK_N - 1) // BLOCK_N
    for tile_n in range(num_prefix_tiles):
        start_n = tile_n * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_valid = offs_n < p_len
        key_abs = p_selected_start + start_n

        k_block_ptr = tl.make_block_ptr(
            base=kp_base,
            shape=(p_end, BLOCK_D),
            strides=(stride_kpt, stride_kpd),
            offsets=(key_abs, 0),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1, 0),
        )
        v_block_ptr = tl.make_block_ptr(
            base=vp_base,
            shape=(p_end, BLOCK_D),
            strides=(stride_vpt, stride_vpd),
            offsets=(key_abs, 0),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1, 0),
        )
        k = tl.load(k_block_ptr, boundary_check=(0,))
        v = tl.load(v_block_ptr, boundary_check=(0,))

        qk = tl.dot(q, tl.trans(k))
        mask = m_valid[:, None] & n_valid[None, :]
        qk = tl.where(mask, qk, -1.0e6)

        m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp2(m_i - m_i_new)
        p = tl.exp2(qk - m_i_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_i_new

    kc_base = KCanvas + pid_kv * stride_kch
    vc_base = VCanvas + pid_kv * stride_vch
    num_canvas_tiles = (q_len + BLOCK_N - 1) // BLOCK_N
    for tile_n in range(num_canvas_tiles):
        start_n = tile_n * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_valid = offs_n < q_len
        key_abs = q_start + start_n

        k_block_ptr = tl.make_block_ptr(
            base=kc_base,
            shape=(q_end, BLOCK_D),
            strides=(stride_kct, stride_kcd),
            offsets=(key_abs, 0),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1, 0),
        )
        v_block_ptr = tl.make_block_ptr(
            base=vc_base,
            shape=(q_end, BLOCK_D),
            strides=(stride_vct, stride_vcd),
            offsets=(key_abs, 0),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1, 0),
        )
        k = tl.load(k_block_ptr, boundary_check=(0,))
        v = tl.load(v_block_ptr, boundary_check=(0,))

        qk = tl.dot(q, tl.trans(k))
        mask = m_valid[:, None] & n_valid[None, :]
        qk = tl.where(mask, qk, -1.0e6)

        m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp2(m_i - m_i_new)
        p = tl.exp2(qk - m_i_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_i_new

    l_i_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_i_safe[:, None]

    out_base = Out + pid_h * stride_oh
    out_block_ptr = tl.make_block_ptr(
        base=out_base,
        shape=(q_end, BLOCK_D),
        strides=(stride_ot, stride_od),
        offsets=(q_abs, 0),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0),
    )
    tl.store(out_block_ptr, acc.to(q.dtype), boundary_check=(0,))

    lse = m_i + tl.log(l_i_safe) * 1.4426950408889634
    lse_offsets = (q_start + offs_m) * n_q_heads + pid_h
    tl.store(LSE + lse_offsets, lse, mask=m_valid)


@triton.jit
def _external_prefix_delta_kernel(
    Out,
    dOut,
    Delta,
    BLOCK_D: tl.constexpr,
):
    """Compute Delta[t,h] = sum_d(Out[t,h,d] * dOut[t,h,d])."""
    pid = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    base = pid * BLOCK_D
    out = tl.load(Out + base + offs_d).to(tl.float32)
    dout = tl.load(dOut + base + offs_d).to(tl.float32)
    tl.store(Delta + pid, tl.sum(out * dout))


@triton.jit
def _external_prefix_bwd_dq_kernel(
    Q,
    KCanvas,
    VCanvas,
    KPrefix,
    VPrefix,
    dOut,
    dQ,
    LSE,
    Delta,
    CuQ,
    CuPrefix,
    PrefixSegmentIds,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kct,
    stride_kch,
    stride_kcd,
    stride_vct,
    stride_vch,
    stride_vcd,
    stride_kpt,
    stride_kph,
    stride_kpd,
    stride_vpt,
    stride_vph,
    stride_vpd,
    stride_dot,
    stride_doh,
    stride_dod,
    stride_dqt,
    stride_dqh,
    stride_dqd,
    n_q_heads: tl.constexpr,
    sm_scale,
    scale,
    HAS_PREFIX_SEGMENT_IDS: tl.constexpr,
    PREFIX_WINDOW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    N_KV_GROUPS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_segment = tl.program_id(2)
    pid_kv = pid_h // N_KV_GROUPS

    q_start = tl.load(CuQ + pid_segment).to(tl.int32)
    q_end = tl.load(CuQ + pid_segment + 1).to(tl.int32)
    q_len = q_end - q_start
    if HAS_PREFIX_SEGMENT_IDS:
        prefix_segment = tl.load(PrefixSegmentIds + pid_segment).to(tl.int32)
    else:
        prefix_segment = pid_segment
    p_start = tl.load(CuPrefix + prefix_segment).to(tl.int32)
    p_end = tl.load(CuPrefix + prefix_segment + 1).to(tl.int32)
    p_len_raw = p_end - p_start
    if PREFIX_WINDOW >= 0:
        p_len = tl.minimum(p_len_raw, PREFIX_WINDOW)
        p_selected_start = p_end - p_len
    else:
        p_len = p_len_raw
        p_selected_start = p_start

    start_m = pid_m * BLOCK_M
    if start_m >= q_len:
        return

    offs_m = start_m + tl.arange(0, BLOCK_M)
    m_valid = offs_m < q_len
    q_abs = q_start + start_m

    q_base = Q + pid_h * stride_qh
    q_block_ptr = tl.make_block_ptr(
        base=q_base,
        shape=(q_end, BLOCK_D),
        strides=(stride_qt, stride_qd),
        offsets=(q_abs, 0),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0),
    )
    q = tl.load(q_block_ptr, boundary_check=(0,))
    q_scaled = (q * sm_scale).to(q.dtype)

    do_base = dOut + pid_h * stride_doh
    do_block_ptr = tl.make_block_ptr(
        base=do_base,
        shape=(q_end, BLOCK_D),
        strides=(stride_dot, stride_dod),
        offsets=(q_abs, 0),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0),
    )
    do = tl.load(do_block_ptr, boundary_check=(0,))

    stats_offsets = (q_start + offs_m) * n_q_heads + pid_h
    lse = tl.load(LSE + stats_offsets, mask=m_valid, other=0.0)
    delta = tl.load(Delta + stats_offsets, mask=m_valid, other=0.0)
    dq = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    kp_base = KPrefix + pid_kv * stride_kph
    vp_base = VPrefix + pid_kv * stride_vph
    num_prefix_tiles = (p_len + BLOCK_N - 1) // BLOCK_N
    for tile_n in range(num_prefix_tiles):
        start_n = tile_n * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_valid = offs_n < p_len
        key_abs = p_selected_start + start_n

        k_block_ptr = tl.make_block_ptr(
            base=kp_base,
            shape=(p_end, BLOCK_D),
            strides=(stride_kpt, stride_kpd),
            offsets=(key_abs, 0),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1, 0),
        )
        v_block_ptr = tl.make_block_ptr(
            base=vp_base,
            shape=(p_end, BLOCK_D),
            strides=(stride_vpt, stride_vpd),
            offsets=(key_abs, 0),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1, 0),
        )
        k = tl.load(k_block_ptr, boundary_check=(0,))
        v = tl.load(v_block_ptr, boundary_check=(0,))

        qk = tl.dot(q_scaled, tl.trans(k))
        mask = m_valid[:, None] & n_valid[None, :]
        qk = tl.where(mask, qk, -1.0e6)
        p = tl.exp2(qk - lse[:, None])
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(k.dtype), k)

    kc_base = KCanvas + pid_kv * stride_kch
    vc_base = VCanvas + pid_kv * stride_vch
    num_canvas_tiles = (q_len + BLOCK_N - 1) // BLOCK_N
    for tile_n in range(num_canvas_tiles):
        start_n = tile_n * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_valid = offs_n < q_len
        key_abs = q_start + start_n

        k_block_ptr = tl.make_block_ptr(
            base=kc_base,
            shape=(q_end, BLOCK_D),
            strides=(stride_kct, stride_kcd),
            offsets=(key_abs, 0),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1, 0),
        )
        v_block_ptr = tl.make_block_ptr(
            base=vc_base,
            shape=(q_end, BLOCK_D),
            strides=(stride_vct, stride_vcd),
            offsets=(key_abs, 0),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1, 0),
        )
        k = tl.load(k_block_ptr, boundary_check=(0,))
        v = tl.load(v_block_ptr, boundary_check=(0,))

        qk = tl.dot(q_scaled, tl.trans(k))
        mask = m_valid[:, None] & n_valid[None, :]
        qk = tl.where(mask, qk, -1.0e6)
        p = tl.exp2(qk - lse[:, None])
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(k.dtype), k)

    dq *= scale
    dq_base = dQ + pid_h * stride_dqh
    dq_block_ptr = tl.make_block_ptr(
        base=dq_base,
        shape=(q_end, BLOCK_D),
        strides=(stride_dqt, stride_dqd),
        offsets=(q_abs, 0),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0),
    )
    tl.store(dq_block_ptr, dq.to(q.dtype), boundary_check=(0,))


@triton.jit
def _external_prefix_bwd_dkdv_kernel(
    Q,
    K,
    V,
    dOut,
    dK,
    dV,
    LSE,
    Delta,
    CuQ,
    CuPrefix,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_dot,
    stride_doh,
    stride_dod,
    stride_dkt,
    stride_dkh,
    stride_dkd,
    stride_dvt,
    stride_dvh,
    stride_dvd,
    n_q_heads: tl.constexpr,
    sm_scale,
    scale,
    PREFIX_WINDOW: tl.constexpr,
    IS_PREFIX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    N_KV_GROUPS: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_kv = tl.program_id(1)
    pid_segment = tl.program_id(2)

    q_start = tl.load(CuQ + pid_segment).to(tl.int32)
    q_end = tl.load(CuQ + pid_segment + 1).to(tl.int32)
    q_len = q_end - q_start

    if IS_PREFIX:
        p_start = tl.load(CuPrefix + pid_segment).to(tl.int32)
        p_end = tl.load(CuPrefix + pid_segment + 1).to(tl.int32)
        p_len_raw = p_end - p_start
        if PREFIX_WINDOW >= 0:
            key_len = tl.minimum(p_len_raw, PREFIX_WINDOW)
            key_start = p_end - key_len
        else:
            key_len = p_len_raw
            key_start = p_start
        key_end = p_end
    else:
        key_len = q_len
        key_start = q_start
        key_end = q_end

    start_n = pid_n * BLOCK_N
    if start_n >= key_len:
        return

    offs_n = start_n + tl.arange(0, BLOCK_N)
    n_valid = offs_n < key_len
    key_abs = key_start + start_n

    k_base = K + pid_kv * stride_kh
    v_base = V + pid_kv * stride_vh
    k_block_ptr = tl.make_block_ptr(
        base=k_base,
        shape=(key_end, BLOCK_D),
        strides=(stride_kt, stride_kd),
        offsets=(key_abs, 0),
        block_shape=(BLOCK_N, BLOCK_D),
        order=(1, 0),
    )
    v_block_ptr = tl.make_block_ptr(
        base=v_base,
        shape=(key_end, BLOCK_D),
        strides=(stride_vt, stride_vd),
        offsets=(key_abs, 0),
        block_shape=(BLOCK_N, BLOCK_D),
        order=(1, 0),
    )
    k = tl.load(k_block_ptr, boundary_check=(0,))
    v = tl.load(v_block_ptr, boundary_check=(0,))

    dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    num_query_tiles = (q_len + BLOCK_M - 1) // BLOCK_M

    for group in range(N_KV_GROUPS):
        pid_h = pid_kv * N_KV_GROUPS + group
        q_base = Q + pid_h * stride_qh
        do_base = dOut + pid_h * stride_doh

        for tile_m in range(num_query_tiles):
            start_m = tile_m * BLOCK_M
            offs_m = start_m + tl.arange(0, BLOCK_M)
            m_valid = offs_m < q_len
            q_abs = q_start + start_m

            q_block_ptr = tl.make_block_ptr(
                base=q_base,
                shape=(q_end, BLOCK_D),
                strides=(stride_qt, stride_qd),
                offsets=(q_abs, 0),
                block_shape=(BLOCK_M, BLOCK_D),
                order=(1, 0),
            )
            do_block_ptr = tl.make_block_ptr(
                base=do_base,
                shape=(q_end, BLOCK_D),
                strides=(stride_dot, stride_dod),
                offsets=(q_abs, 0),
                block_shape=(BLOCK_M, BLOCK_D),
                order=(1, 0),
            )
            q = tl.load(q_block_ptr, boundary_check=(0,))
            do = tl.load(do_block_ptr, boundary_check=(0,))

            qk = tl.dot(q, tl.trans(k)) * sm_scale
            mask = m_valid[:, None] & n_valid[None, :]
            qk = tl.where(mask, qk, -1.0e6)

            stats_offsets = (q_start + offs_m) * n_q_heads + pid_h
            lse = tl.load(LSE + stats_offsets, mask=m_valid, other=0.0)
            delta = tl.load(Delta + stats_offsets, mask=m_valid, other=0.0)
            p = tl.exp2(qk - lse[:, None])

            dv += tl.dot(tl.trans(p.to(do.dtype)), do)
            dp = tl.dot(do, tl.trans(v))
            ds = p * (dp - delta[:, None])
            dk += tl.dot(tl.trans(ds.to(q.dtype)), q)

    dk *= scale
    dk_base = dK + pid_kv * stride_dkh
    dv_base = dV + pid_kv * stride_dvh
    dk_block_ptr = tl.make_block_ptr(
        base=dk_base,
        shape=(key_end, BLOCK_D),
        strides=(stride_dkt, stride_dkd),
        offsets=(key_abs, 0),
        block_shape=(BLOCK_N, BLOCK_D),
        order=(1, 0),
    )
    dv_block_ptr = tl.make_block_ptr(
        base=dv_base,
        shape=(key_end, BLOCK_D),
        strides=(stride_dvt, stride_dvd),
        offsets=(key_abs, 0),
        block_shape=(BLOCK_N, BLOCK_D),
        order=(1, 0),
    )
    tl.store(dk_block_ptr, dk.to(k.dtype), boundary_check=(0,))
    tl.store(dv_block_ptr, dv.to(v.dtype), boundary_check=(0,))


class _ExternalPrefixKVTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q_canvas,
        k_canvas,
        v_canvas,
        k_prefix,
        v_prefix,
        cu_seqlens_q,
        cu_seqlens_prefix,
        prefix_segment_ids,
        has_prefix_segment_ids,
        max_seqlen_q,
        max_seqlen_prefix,
        scale,
        prefix_window,
    ):
        total_q, n_q_heads, head_dim = q_canvas.shape
        n_kv_heads = k_canvas.shape[1]
        n_kv_groups = n_q_heads // n_kv_heads
        batch_size = cu_seqlens_q.numel() - 1
        sm_scale = scale * 1.4426950408889634

        block_m, block_n, num_warps, num_stages = _forward_config(
            head_dim, n_q_heads, n_kv_heads, q_canvas.device
        )
        out = torch.empty(
            q_canvas.shape, dtype=q_canvas.dtype, device=q_canvas.device
        )
        lse = torch.empty(
            total_q, n_q_heads, dtype=torch.float32, device=q_canvas.device
        )
        grid = (
            triton.cdiv(max_seqlen_q, block_m),
            n_q_heads,
            batch_size,
        )
        _external_prefix_fwd_kernel[grid](
            q_canvas,
            k_canvas,
            v_canvas,
            k_prefix,
            v_prefix,
            out,
            lse,
            cu_seqlens_q,
            cu_seqlens_prefix,
            prefix_segment_ids,
            q_canvas.stride(0),
            q_canvas.stride(1),
            q_canvas.stride(2),
            k_canvas.stride(0),
            k_canvas.stride(1),
            k_canvas.stride(2),
            v_canvas.stride(0),
            v_canvas.stride(1),
            v_canvas.stride(2),
            k_prefix.stride(0),
            k_prefix.stride(1),
            k_prefix.stride(2),
            v_prefix.stride(0),
            v_prefix.stride(1),
            v_prefix.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            n_q_heads=n_q_heads,
            sm_scale=sm_scale,
            HAS_PREFIX_SEGMENT_IDS=has_prefix_segment_ids,
            PREFIX_WINDOW=prefix_window,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=head_dim,
            N_KV_GROUPS=n_kv_groups,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        ctx.save_for_backward(
            q_canvas,
            k_canvas,
            v_canvas,
            k_prefix,
            v_prefix,
            out,
            lse,
            cu_seqlens_q,
            cu_seqlens_prefix,
            prefix_segment_ids,
        )
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_prefix = max_seqlen_prefix
        ctx.scale = scale
        ctx.sm_scale = sm_scale
        ctx.prefix_window = prefix_window
        ctx.has_prefix_segment_ids = has_prefix_segment_ids
        ctx.n_q_heads = n_q_heads
        ctx.n_kv_heads = n_kv_heads
        ctx.head_dim = head_dim
        return out

    @staticmethod
    def backward(ctx, d_out):
        (
            q_canvas,
            k_canvas,
            v_canvas,
            k_prefix,
            v_prefix,
            out,
            lse,
            cu_seqlens_q,
            cu_seqlens_prefix,
            prefix_segment_ids,
        ) = ctx.saved_tensors
        d_out = d_out.contiguous()

        total_q, n_q_heads, head_dim = q_canvas.shape
        n_kv_heads = ctx.n_kv_heads
        n_kv_groups = n_q_heads // n_kv_heads
        batch_size = cu_seqlens_q.numel() - 1
        needs = ctx.needs_input_grad
        if ctx.has_prefix_segment_ids and (needs[3] or needs[4]):
            raise RuntimeError(
                "Mapped trainable prefix K/V is not supported by the Triton "
                "backward; use backend='flash' or backend='reference'."
            )

        delta = torch.empty(
            total_q, n_q_heads, dtype=torch.float32, device=q_canvas.device
        )
        _external_prefix_delta_kernel[(total_q * n_q_heads,)](
            out,
            d_out,
            delta,
            BLOCK_D=head_dim,
            num_warps=min(8, max(1, head_dim // 32)),
        )

        dq = None
        if needs[0]:
            block_m, block_n, num_warps, num_stages = _dq_config(
                head_dim, n_q_heads, n_kv_heads, q_canvas.device
            )
            dq = torch.empty(
                q_canvas.shape, dtype=q_canvas.dtype, device=q_canvas.device
            )
            grid = (
                triton.cdiv(ctx.max_seqlen_q, block_m),
                n_q_heads,
                batch_size,
            )
            _external_prefix_bwd_dq_kernel[grid](
                q_canvas,
                k_canvas,
                v_canvas,
                k_prefix,
                v_prefix,
                d_out,
                dq,
                lse,
                delta,
                cu_seqlens_q,
                cu_seqlens_prefix,
                prefix_segment_ids,
                q_canvas.stride(0),
                q_canvas.stride(1),
                q_canvas.stride(2),
                k_canvas.stride(0),
                k_canvas.stride(1),
                k_canvas.stride(2),
                v_canvas.stride(0),
                v_canvas.stride(1),
                v_canvas.stride(2),
                k_prefix.stride(0),
                k_prefix.stride(1),
                k_prefix.stride(2),
                v_prefix.stride(0),
                v_prefix.stride(1),
                v_prefix.stride(2),
                d_out.stride(0),
                d_out.stride(1),
                d_out.stride(2),
                dq.stride(0),
                dq.stride(1),
                dq.stride(2),
                n_q_heads=n_q_heads,
                sm_scale=ctx.sm_scale,
                scale=ctx.scale,
                HAS_PREFIX_SEGMENT_IDS=ctx.has_prefix_segment_ids,
                PREFIX_WINDOW=ctx.prefix_window,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_D=head_dim,
                N_KV_GROUPS=n_kv_groups,
                num_warps=num_warps,
                num_stages=num_stages,
            )

        dkc = dvc = None
        if needs[1] or needs[2]:
            block_m, block_n, num_warps, num_stages = _dkdv_config(
                head_dim, n_q_heads, n_kv_heads, q_canvas.device
            )
            dkc_work = torch.zeros(
                k_canvas.shape, dtype=k_canvas.dtype, device=k_canvas.device
            )
            dvc_work = torch.zeros(
                v_canvas.shape, dtype=v_canvas.dtype, device=v_canvas.device
            )
            grid = (
                triton.cdiv(ctx.max_seqlen_q, block_n),
                n_kv_heads,
                batch_size,
            )
            _external_prefix_bwd_dkdv_kernel[grid](
                q_canvas,
                k_canvas,
                v_canvas,
                d_out,
                dkc_work,
                dvc_work,
                lse,
                delta,
                cu_seqlens_q,
                cu_seqlens_prefix,
                q_canvas.stride(0),
                q_canvas.stride(1),
                q_canvas.stride(2),
                k_canvas.stride(0),
                k_canvas.stride(1),
                k_canvas.stride(2),
                v_canvas.stride(0),
                v_canvas.stride(1),
                v_canvas.stride(2),
                d_out.stride(0),
                d_out.stride(1),
                d_out.stride(2),
                dkc_work.stride(0),
                dkc_work.stride(1),
                dkc_work.stride(2),
                dvc_work.stride(0),
                dvc_work.stride(1),
                dvc_work.stride(2),
                n_q_heads=n_q_heads,
                sm_scale=ctx.sm_scale,
                scale=ctx.scale,
                PREFIX_WINDOW=ctx.prefix_window,
                IS_PREFIX=False,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_D=head_dim,
                N_KV_GROUPS=n_kv_groups,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            dkc = dkc_work if needs[1] else None
            dvc = dvc_work if needs[2] else None

        dkp = dvp = None
        if needs[3] or needs[4]:
            dkp_work = torch.zeros(
                k_prefix.shape, dtype=k_prefix.dtype, device=k_prefix.device
            )
            dvp_work = torch.zeros(
                v_prefix.shape, dtype=v_prefix.dtype, device=v_prefix.device
            )
            max_selected_prefix = ctx.max_seqlen_prefix
            if ctx.prefix_window >= 0:
                max_selected_prefix = min(
                    max_selected_prefix, ctx.prefix_window
                )
            if max_selected_prefix > 0:
                block_m, block_n, num_warps, num_stages = _dkdv_config(
                    head_dim, n_q_heads, n_kv_heads, q_canvas.device
                )
                grid = (
                    triton.cdiv(max_selected_prefix, block_n),
                    n_kv_heads,
                    batch_size,
                )
                _external_prefix_bwd_dkdv_kernel[grid](
                    q_canvas,
                    k_prefix,
                    v_prefix,
                    d_out,
                    dkp_work,
                    dvp_work,
                    lse,
                    delta,
                    cu_seqlens_q,
                    cu_seqlens_prefix,
                    q_canvas.stride(0),
                    q_canvas.stride(1),
                    q_canvas.stride(2),
                    k_prefix.stride(0),
                    k_prefix.stride(1),
                    k_prefix.stride(2),
                    v_prefix.stride(0),
                    v_prefix.stride(1),
                    v_prefix.stride(2),
                    d_out.stride(0),
                    d_out.stride(1),
                    d_out.stride(2),
                    dkp_work.stride(0),
                    dkp_work.stride(1),
                    dkp_work.stride(2),
                    dvp_work.stride(0),
                    dvp_work.stride(1),
                    dvp_work.stride(2),
                    n_q_heads=n_q_heads,
                    sm_scale=ctx.sm_scale,
                    scale=ctx.scale,
                    PREFIX_WINDOW=ctx.prefix_window,
                    IS_PREFIX=True,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    BLOCK_D=head_dim,
                    N_KV_GROUPS=n_kv_groups,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
            dkp = dkp_work if needs[3] else None
            dvp = dvp_work if needs[4] else None

        return (
            dq,
            dkc,
            dvc,
            dkp,
            dvp,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def external_kv_varlen_attention_triton(
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
    """Run no-copy external-prefix-KV attention with native Triton kernels."""
    supported, reason = triton_backend_supported(
        q_canvas,
        k_canvas,
        v_canvas,
        k_prefix,
        v_prefix,
        cu_seqlens_q,
        cu_seqlens_prefix,
        prefix_segment_ids,
    )
    if not supported:
        raise ValueError(f"Triton external-prefix-KV backend unsupported: {reason}")
    if prefix_window is not None and prefix_window < 0:
        raise ValueError(f"prefix_window must be non-negative or None, got {prefix_window}")
    q_lengths = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    prefix_lengths = cu_seqlens_prefix[1:] - cu_seqlens_prefix[:-1]
    if max_seqlen_q is None:
        max_seqlen_q = int(q_lengths.max().item())
    if max_seqlen_prefix is None:
        max_seqlen_prefix = int(prefix_lengths.max().item())
    if max_seqlen_q <= 0:
        raise ValueError("max_seqlen_q must be positive")
    if max_seqlen_prefix < 0:
        raise ValueError("max_seqlen_prefix must be non-negative")
    if scale is None:
        scale = 1.0 / math.sqrt(q_canvas.shape[-1])

    window = -1 if prefix_window is None else int(prefix_window)
    prefix_ids_arg = (
        prefix_segment_ids
        if prefix_segment_ids is not None
        else cu_seqlens_q
    )
    return _ExternalPrefixKVTritonFunction.apply(
        q_canvas,
        k_canvas,
        v_canvas,
        k_prefix,
        v_prefix,
        cu_seqlens_q,
        cu_seqlens_prefix,
        prefix_ids_arg,
        prefix_segment_ids is not None,
        int(max_seqlen_q),
        int(max_seqlen_prefix),
        float(scale),
        window,
    )
