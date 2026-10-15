"""
Triton Flash Attention for Diffusion Language Model with noisy input.

Provides ``diff_attn_varlen_with_noisy_input`` — a fused forward + backward kernel
that implements the document-level staircase attention mask used in diffusion
language model training.

Token layout per document
-------------------------
::

    [ Prompt (L0 tokens) | Clean Response (L1 tokens) | Noisy Response (L1 tokens) ]

Total tokens per document = L0 + 2 * L1.  Multiple documents are packed into a
single flat tensor and separated by ``cu_seqlens`` (cumulative sequence lengths).

Visibility rules (staircase_size = S, block index = token_offset // S)
----------------------------------------------------------------------
- **Prompt** query in block *b*  → sees prompt blocks [0 .. b]
- **Clean**  query in block *b*  → sees ALL prompt + clean blocks [0 .. b]
- **Noisy**  query in block *b*  → sees ALL prompt + clean blocks [0 .. b-1]
                                    + own noisy block *b*  (NOT own clean block)

Supported dtypes: ``torch.float16``, ``torch.bfloat16``.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import triton
import triton.language as tl

__all__ = ["diff_attn_varlen_with_noisy_input", "idlm_attn_varlen"]

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)
_VALID_STAIRCASE_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256)

# ---------------------------------------------------------------------------
# Tuned kernel configs: (cc_major, cc_minor, head_dim, n_q_heads, n_kv_heads) -> kernel -> (BM, BN, warps, stages)
# Matching order: exact match on all 5 params -> match on (cc_major, cc_minor, head_dim) -> match on (cc_major, cc_minor) -> default
# ---------------------------------------------------------------------------
_TUNED_CONFIGS = {
    (8, 0, 128, 16, 8): {
        "fwd":  (128, 32, 4, 1),
        "dq":   (64, 64, 4, 1),
        "dkdv": (32, 64, 4, 3),
    },
    (8, 0, 128, 32, 8): {
        "fwd":  (128, 32, 4, 1),
        "dq":   (64, 64, 4, 1),
        "dkdv": (32, 64, 4, 3),
    },
    (8, 0, 128, 32, 4): {
        "fwd":  (128, 32, 4, 1),
        "dq":   (64, 64, 4, 1),
        "dkdv": (32, 64, 4, 3),
    },

    (8, 6, 128, 16, 8): {
        "fwd":  (128, 32, 4, 1),
        "dq":   (64, 32, 4, 1),
        "dkdv": (32, 32, 4, 1),
    },
    (8, 6, 128, 32, 8): {
        "fwd":  (128, 32, 4, 1),
        "dq":   (64, 32, 4, 1),
        "dkdv": (32, 32, 4, 1),
    },
    (8, 6, 128, 32, 4): {
        "fwd":  (128, 32, 4, 1),
        "dq":   (64, 32, 4, 1),
        "dkdv": (32, 32, 4, 1),
    },

    (8, 9, 128, 16, 8): {
        "fwd":  (128, 32, 4, 2),
        "dq":   (128, 32, 8, 1),
        "dkdv": (64, 64, 8, 3),
    },
    (8, 9, 128, 32, 8): {
        "fwd":  (128, 32, 4, 2),
        "dq":   (128, 32, 8, 1),
        "dkdv": (64, 64, 8, 3),
    },
    (8, 9, 128, 32, 4): {
        "fwd":  (128, 32, 4, 2),
        "dq":   (128, 32, 8, 1),
        "dkdv": (64, 64, 8, 2),
    },

    (9, 0, 128, 16, 8): {
        "fwd":  (128, 128, 8, 2),
        "dq":   (128, 128, 8, 2),
        "dkdv": (64, 64, 4, 1),
    },
    (9, 0, 128, 32, 8): {
        "fwd":  (128, 128, 8, 2),
        "dq":   (128, 128, 8, 2),
        "dkdv": (64, 64, 4, 3),
    },
    (9, 0, 128, 32, 4): {
        "fwd":  (128, 128, 8, 2),
        "dq":   (128, 128, 8, 2),
        "dkdv": (64, 64, 4, 1),
    },
}

# Fallback configs for architectures without tuned values
_DEFAULT_CONFIGS = {
    "fwd":  (128, 32, 4, 1),
    "dq":   (64, 32, 4, 1),
    "dkdv": (32, 32, 4, 1),
}


def _get_kernel_config(head_dim: int, n_q_heads: int, n_kv_heads: int, kernel: str):
    """Return (BLOCK_M, BLOCK_N, num_warps, num_stages) for *kernel*.

    Matching order:
    1. Exact match on (cc_major, cc_minor, head_dim, n_q_heads, n_kv_heads)
    2. Match on (cc_major, cc_minor, head_dim) with any nh/nkv
    3. Match on (cc_major, cc_minor) with any head_dim/nh/nkv
    4. Fallback to default config
    """
    import warnings

    cc_major, cc_minor = torch.cuda.get_device_capability()

    # Try exact match on all 5 params
    key = (cc_major, cc_minor, head_dim, n_q_heads, n_kv_heads)
    if key in _TUNED_CONFIGS and kernel in _TUNED_CONFIGS[key]:
        return _TUNED_CONFIGS[key][kernel]

    # Fallback to default
    warnings.warn(
        f"No tuned config for ({cc_major}.{cc_minor}, d={head_dim}, "
        f"nh={n_q_heads}, nkv={n_kv_heads}, kernel={kernel}). "
        f"Using default config: {_DEFAULT_CONFIGS[kernel]}. Consider running tuning."
    )
    return _DEFAULT_CONFIGS[kernel]


# ---------------------------------------------------------------------------
# Helper — compute per-row mask ranges
# ---------------------------------------------------------------------------
@triton.jit
def _compute_ranges(offs_m, L0, L1, doc_len, LOG2S: tl.constexpr):
    """Compute two visibility ranges for prompt/clean/noisy layout."""
    is_prompt = offs_m < L0
    is_clean = (offs_m >= L0) & (offs_m < L0 + L1)

    q_off = tl.where(is_prompt, offs_m,
                     tl.where(is_clean, offs_m - L0, offs_m - L0 - L1))
    b = q_off >> LOG2S
    next_block_end = (b + 1) << LOG2S

    lower1 = tl.zeros_like(offs_m)
    upper1 = tl.where(is_prompt, tl.minimum(next_block_end, L0),
                      tl.where(is_clean, tl.minimum(L0 + next_block_end, L0 + L1),
                               L0 + (b << LOG2S)))

    lower2 = tl.where(is_prompt | is_clean, doc_len, L0 + L1 + (b << LOG2S))
    upper2 = tl.where(is_prompt | is_clean, doc_len, tl.minimum(L0 + L1 + next_block_end, doc_len))

    return lower1, upper1, lower2, upper2


@triton.jit
def _compute_idlm_ranges(
    offs_m,
    doc_len,
    LOG2S: tl.constexpr,
    USE_REGULAR_CAUSAL: tl.constexpr,
    NOISY_BIDIRECTIONAL: tl.constexpr,
):
    """Compute two visibility ranges for I-DLM native [xt | x0] layout."""
    n = doc_len // 2
    is_xt = offs_m < n
    q_off = tl.where(is_xt, offs_m, offs_m - n)
    b = q_off >> LOG2S
    block_start = b << LOG2S
    block_end = tl.minimum((b + 1) << LOG2S, n)

    if USE_REGULAR_CAUSAL and not NOISY_BIDIRECTIONAL:
        xt_upper = q_off + 1
    else:
        xt_upper = block_end

    if USE_REGULAR_CAUSAL:
        clean_upper = n + q_off + 1
    else:
        clean_upper = n + block_end

    lower1 = tl.where(is_xt, block_start, n)
    upper1 = tl.where(is_xt, xt_upper, clean_upper)
    lower2 = tl.where(is_xt, n, doc_len)
    upper2 = tl.where(is_xt, n + block_start, doc_len)
    return lower1, upper1, lower2, upper2


@triton.jit
def _compute_ranges_for_mode(
    offs_m,
    L0,
    L1,
    doc_len,
    LOG2S: tl.constexpr,
    MODE: tl.constexpr,
    USE_REGULAR_CAUSAL: tl.constexpr,
    NOISY_BIDIRECTIONAL: tl.constexpr,
):
    if MODE == 1:
        return _compute_idlm_ranges(
            offs_m,
            doc_len,
            LOG2S,
            USE_REGULAR_CAUSAL,
            NOISY_BIDIRECTIONAL,
        )

    return _compute_ranges(offs_m, L0, L1, doc_len, LOG2S)


# ---------------------------------------------------------------------------
# Forward kernel
# ---------------------------------------------------------------------------

@triton.jit
def _diff_noisy_attn_fwd_kernel(
    Q, K, V, Out, L_buf,
    cu_seqlens, L0_arr, L1_arr,
    stride_qt, stride_qh, stride_qd,
    stride_kt, stride_kh, stride_kd,
    stride_vt, stride_vh, stride_vd,
    stride_ot, stride_oh, stride_od,
    n_heads: tl.constexpr,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    LOG2S: tl.constexpr,
    N_KV_GROUPS: tl.constexpr,
    MODE: tl.constexpr,
    USE_REGULAR_CAUSAL: tl.constexpr,
    NOISY_BIDIRECTIONAL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_doc = tl.program_id(2)
    pid_kv = pid_h // N_KV_GROUPS

    seq_start = tl.load(cu_seqlens + pid_doc).to(tl.int32)
    seq_end = tl.load(cu_seqlens + pid_doc + 1).to(tl.int32)
    doc_len = seq_end - seq_start
    L0 = tl.load(L0_arr + pid_doc).to(tl.int32)
    L1 = tl.load(L1_arr + pid_doc).to(tl.int32)

    start_m = pid_m * BLOCK_M
    if start_m >= doc_len:
        return

    offs_m = start_m + tl.arange(0, BLOCK_M)
    m_valid = offs_m < doc_len

    lower1, upper1, lower2, upper2 = _compute_ranges_for_mode(
        offs_m,
        L0,
        L1,
        doc_len,
        LOG2S,
        MODE,
        USE_REGULAR_CAUSAL,
        NOISY_BIDIRECTIONAL,
    )

    # Load Q tile
    q_abs = seq_start + start_m
    Q_ptr = Q + pid_h * stride_qh
    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr, shape=(seq_end, BLOCK_D),
        strides=(stride_qt, stride_qd), offsets=(q_abs, 0),
        block_shape=(BLOCK_M, BLOCK_D), order=(1, 0),
    )
    q = tl.load(Q_block_ptr, boundary_check=(0,))
    q = (q * sm_scale).to(q.dtype)

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    K_ptr = K + pid_kv * stride_kh
    V_ptr = V + pid_kv * stride_vh

    # --- Range1 loop ---
    min_lower1 = tl.min(tl.where(m_valid, lower1, doc_len))
    max_upper1 = tl.max(tl.where(m_valid, upper1, 0))
    max_upper1 = tl.minimum(max_upper1, doc_len)
    range1_start = (min_lower1 // BLOCK_N) * BLOCK_N
    num_n1 = (max_upper1 - range1_start + BLOCK_N - 1) // BLOCK_N
    num_n1 = tl.where(min_lower1 < max_upper1, num_n1, 0)

    for it_n in range(num_n1):
        start_n = range1_start + it_n * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_valid = offs_n < doc_len

        k_abs = seq_start + start_n
        K_block_ptr = tl.make_block_ptr(
            base=K_ptr, shape=(seq_end, BLOCK_D),
            strides=(stride_kt, stride_kd), offsets=(k_abs, 0),
            block_shape=(BLOCK_N, BLOCK_D), order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            base=V_ptr, shape=(seq_end, BLOCK_D),
            strides=(stride_vt, stride_vd), offsets=(k_abs, 0),
            block_shape=(BLOCK_N, BLOCK_D), order=(1, 0),
        )
        k = tl.load(K_block_ptr, boundary_check=(0,))
        v = tl.load(V_block_ptr, boundary_check=(0,))

        qk = tl.dot(q, tl.trans(k))
        # Range1 mask: lower1[qi] <= ki < upper1[qi]
        cols = offs_n[None, :]
        mask = (cols >= lower1[:, None]) & (cols < upper1[:, None]) & m_valid[:, None] & n_valid[None, :]
        qk = tl.where(mask, qk, -1.0e6)

        m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp2(m_i - m_i_new)
        p = tl.exp2(qk - m_i_new[:, None])
        l_ij = tl.sum(p, axis=1)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        m_i = m_i_new
        acc = tl.dot(p.to(v.dtype), v, acc)

    # --- Range2 loop: [min_lower2, max_upper2) --- only for noisy rows
    min_lower2 = tl.min(tl.where(m_valid, lower2, doc_len))
    max_upper2 = tl.max(tl.where(m_valid, upper2, 0))
    max_upper2 = tl.minimum(max_upper2, doc_len)

    # Align min_lower2 down to BLOCK_N boundary
    range2_start = (min_lower2 // BLOCK_N) * BLOCK_N
    num_n2 = (max_upper2 - range2_start + BLOCK_N - 1) // BLOCK_N
    # Only iterate if range2 is non-empty
    num_n2 = tl.where(min_lower2 < max_upper2, num_n2, 0)

    for it_n in range(num_n2):
        start_n = range2_start + it_n * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_valid = offs_n < doc_len

        k_abs = seq_start + start_n
        K_block_ptr = tl.make_block_ptr(
            base=K_ptr, shape=(seq_end, BLOCK_D),
            strides=(stride_kt, stride_kd), offsets=(k_abs, 0),
            block_shape=(BLOCK_N, BLOCK_D), order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            base=V_ptr, shape=(seq_end, BLOCK_D),
            strides=(stride_vt, stride_vd), offsets=(k_abs, 0),
            block_shape=(BLOCK_N, BLOCK_D), order=(1, 0),
        )
        k = tl.load(K_block_ptr, boundary_check=(0,))
        v = tl.load(V_block_ptr, boundary_check=(0,))

        qk = tl.dot(q, tl.trans(k))
        # Range2 mask: lower2[qi] <= ki < upper2[qi]
        cols = offs_n[None, :]
        mask = (cols >= lower2[:, None]) & (cols < upper2[:, None]) & m_valid[:, None] & n_valid[None, :]
        qk = tl.where(mask, qk, -1.0e6)

        m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp2(m_i - m_i_new)
        p = tl.exp2(qk - m_i_new[:, None])
        l_ij = tl.sum(p, axis=1)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        m_i = m_i_new
        acc = tl.dot(p.to(v.dtype), v, acc)

    # Normalize & store
    l_i_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_i_safe[:, None]

    O_ptr = Out + pid_h * stride_oh
    O_block_ptr = tl.make_block_ptr(
        base=O_ptr, shape=(seq_end, BLOCK_D),
        strides=(stride_ot, stride_od), offsets=(q_abs, 0),
        block_shape=(BLOCK_M, BLOCK_D), order=(1, 0),
    )
    tl.store(O_block_ptr, acc.to(q.dtype), boundary_check=(0,))

    lse = m_i + tl.log(l_i_safe) * 1.44269504  # store in log2 domain
    lse = tl.where(m_valid, lse, 0.0)
    L_base = L_buf + pid_h
    offs_l = (seq_start + offs_m) * n_heads
    tl.store(L_base + offs_l, lse, mask=m_valid)


# ---------------------------------------------------------------------------
# Backward kernel - dK, dV
# ---------------------------------------------------------------------------

@triton.jit
def _diff_noisy_attn_bwd_dkdv_kernel(
    Q, K, V, Out, dO, dK, dV, L_buf, D_buf,
    cu_seqlens, L0_arr, L1_arr,
    stride_qt, stride_qh, stride_qd,
    stride_kt, stride_kh, stride_kd,
    stride_vt, stride_vh, stride_vd,
    stride_ot, stride_oh, stride_od,
    stride_dkt, stride_dkh, stride_dkd,
    n_heads: tl.constexpr,
    sm_scale,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    LOG2S: tl.constexpr,
    N_KV_GROUPS: tl.constexpr,
    MODE: tl.constexpr,
    USE_REGULAR_CAUSAL: tl.constexpr,
    NOISY_BIDIRECTIONAL: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)     # Q head index
    pid_doc = tl.program_id(2)
    pid_kv = pid_h // N_KV_GROUPS  # corresponding KV head

    seq_start = tl.load(cu_seqlens + pid_doc).to(tl.int32)
    seq_end = tl.load(cu_seqlens + pid_doc + 1).to(tl.int32)
    doc_len = seq_end - seq_start
    L0 = tl.load(L0_arr + pid_doc).to(tl.int32)
    L1 = tl.load(L1_arr + pid_doc).to(tl.int32)

    start_n = pid_n * BLOCK_N
    if start_n >= doc_len:
        return

    offs_n = start_n + tl.arange(0, BLOCK_N)
    n_valid = offs_n < doc_len

    k_abs = seq_start + start_n
    K_ptr = K + pid_kv * stride_kh
    V_ptr = V + pid_kv * stride_vh

    K_block_ptr = tl.make_block_ptr(
        base=K_ptr, shape=(seq_end, BLOCK_D),
        strides=(stride_kt, stride_kd), offsets=(k_abs, 0),
        block_shape=(BLOCK_N, BLOCK_D), order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V_ptr, shape=(seq_end, BLOCK_D),
        strides=(stride_vt, stride_vd), offsets=(k_abs, 0),
        block_shape=(BLOCK_N, BLOCK_D), order=(1, 0),
    )
    k = tl.load(K_block_ptr, boundary_check=(0,))
    v = tl.load(V_block_ptr, boundary_check=(0,))

    dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

    Q_ptr = Q + pid_h * stride_qh
    dO_ptr = dO + pid_h * stride_oh

    if MODE == 1:
        # I-DLM [xt | x0] inverse bounds.
        # xt keys are only visible to queries in their own xt block.
        # clean keys are visible to later xt blocks and to causal clean suffix.
        n = doc_len // 2
        is_xt_k = offs_n < n
        is_clean_k = ~is_xt_k
        k_off = tl.where(is_xt_k, offs_n, offs_n - n)
        b_k = k_off >> LOG2S

        any_valid = tl.max(tl.where(n_valid, 1, 0).to(tl.int32)) > 0
        any_xt = tl.max(tl.where(n_valid & is_xt_k, 1, 0).to(tl.int32)) > 0
        any_clean = tl.max(tl.where(n_valid & is_clean_k, 1, 0).to(tl.int32)) > 0
        all_xt = any_valid & any_xt & (~any_clean)
        all_clean = any_valid & any_clean & (~any_xt)

        min_xt_block_start = tl.min(tl.where(n_valid & is_xt_k, b_k << LOG2S, doc_len))
        max_xt_block_end = tl.max(tl.where(n_valid & is_xt_k, tl.minimum((b_k + 1) << LOG2S, n), 0))

        min_clean_xt_start = tl.min(tl.where(n_valid & is_clean_k, tl.minimum((b_k + 1) << LOG2S, n), doc_len))
        min_clean_off = tl.min(tl.where(n_valid & is_clean_k, k_off, n))
        min_clean_block_start = tl.min(tl.where(n_valid & is_clean_k, b_k << LOG2S, n))
        if USE_REGULAR_CAUSAL:
            clean_suffix_start = n + min_clean_off
        else:
            clean_suffix_start = n + min_clean_block_start

        m_lower = tl.where(all_xt, min_xt_block_start, tl.where(all_clean, min_clean_xt_start, 0))
        m_upper = tl.where(all_xt, max_xt_block_end, doc_len)
        gap_start = tl.where(all_clean, n, doc_len)
        gap_end = tl.where(all_clean, clean_suffix_start, doc_len)
    else:
        # Compute tight M-iteration bounds based on which query rows attend to keys in this N-tile.
        is_prompt_k = offs_n < L0
        is_clean_k = (offs_n >= L0) & (offs_n < L0 + L1)
        is_noisy_k = offs_n >= L0 + L1
        k_off = tl.where(is_prompt_k, offs_n,
                         tl.where(is_clean_k, offs_n - L0, offs_n - L0 - L1))
        b_k = k_off >> LOG2S

        lower_q = tl.where(is_prompt_k, b_k << LOG2S,
                           tl.where(is_clean_k, L0 + (b_k << LOG2S),
                                    L0 + L1 + (b_k << LOG2S)))
        upper_q = tl.where(is_noisy_k,
                           tl.minimum(L0 + L1 + ((b_k + 1) << LOG2S), doc_len),
                           doc_len)

        m_lower = tl.min(tl.where(n_valid, lower_q, doc_len))
        m_upper = tl.max(tl.where(n_valid, upper_q, 0))
        m_upper = tl.minimum(m_upper, doc_len)

        # Compute gap bounds for clean keys to skip useless M-tiles.
        any_prompt = tl.max(tl.where(n_valid & is_prompt_k, 1, 0).to(tl.int32)) > 0
        any_noisy = tl.max(tl.where(n_valid & is_noisy_k, 1, 0).to(tl.int32)) > 0
        all_clean = (~any_prompt) & (~any_noisy) & (tl.max(tl.where(n_valid, 1, 0).to(tl.int32)) > 0)

        min_b_clean = tl.min(tl.where(n_valid & is_clean_k, b_k, 999999))
        gap_start = tl.where(all_clean, L0 + L1, m_upper)
        gap_end = tl.where(all_clean, tl.minimum(L0 + L1 + ((min_b_clean + 1) << LOG2S), doc_len), m_upper)

    # Align to BLOCK_M
    m_start_tile = (m_lower // BLOCK_M) * BLOCK_M
    num_m_iters = (m_upper - m_start_tile + BLOCK_M - 1) // BLOCK_M
    num_m_iters = tl.where(m_lower < m_upper, num_m_iters, 0)

    for it_m in range(num_m_iters):
        start_m = m_start_tile + it_m * BLOCK_M

        # Skip if this M-tile is entirely within the gap
        in_gap = (start_m >= gap_start) & (start_m + BLOCK_M <= gap_end)
        if not in_gap:
            offs_m = start_m + tl.arange(0, BLOCK_M)
            m_valid = offs_m < doc_len

            lower1, upper1, lower2, upper2_q = _compute_ranges_for_mode(
                offs_m,
                L0,
                L1,
                doc_len,
                LOG2S,
                MODE,
                USE_REGULAR_CAUSAL,
                NOISY_BIDIRECTIONAL,
            )

            # Build combined mask
            cols = offs_n[None, :]
            in_r1 = (cols >= lower1[:, None]) & (cols < upper1[:, None])
            in_r2 = (cols >= lower2[:, None]) & (cols < upper2_q[:, None])
            mask = (in_r1 | in_r2) & m_valid[:, None] & n_valid[None, :]

            q_abs = seq_start + start_m
            Q_block_ptr = tl.make_block_ptr(
                base=Q_ptr, shape=(seq_end, BLOCK_D),
                strides=(stride_qt, stride_qd), offsets=(q_abs, 0),
                block_shape=(BLOCK_M, BLOCK_D), order=(1, 0),
            )
            q = tl.load(Q_block_ptr, boundary_check=(0,))

            qk = tl.dot(q, tl.trans(k))
            qk = qk * sm_scale
            qk = tl.where(mask, qk, -1.0e6)

            L_base = L_buf + pid_h
            offs_l = (seq_start + offs_m) * n_heads
            lse = tl.load(L_base + offs_l, mask=m_valid, other=0.0)

            p = tl.exp2(qk - lse[:, None])

            dO_block_ptr = tl.make_block_ptr(
                base=dO_ptr, shape=(seq_end, BLOCK_D),
                strides=(stride_ot, stride_od), offsets=(q_abs, 0),
                block_shape=(BLOCK_M, BLOCK_D), order=(1, 0),
            )
            do = tl.load(dO_block_ptr, boundary_check=(0,))
            dv += tl.dot(tl.trans(p.to(do.dtype)), do)

            D_base = D_buf + pid_h
            Di = tl.load(D_base + offs_l, mask=m_valid, other=0.0)

            dpT = tl.dot(do, tl.trans(v))
            ds = p * (dpT - Di[:, None])
            dk += tl.dot(tl.trans(ds.to(q.dtype)), q)

    # Apply scale for correct gradient magnitude (not sm_scale which includes log2(e))
    dk = dk * scale

    # Write partial dK/dV for this Q head (reduced across Q heads on host)
    dK_ptr = dK + pid_h * stride_dkh
    dV_ptr = dV + pid_h * stride_dkh
    dK_block_ptr = tl.make_block_ptr(
        base=dK_ptr, shape=(seq_end, BLOCK_D),
        strides=(stride_dkt, stride_dkd), offsets=(k_abs, 0),
        block_shape=(BLOCK_N, BLOCK_D), order=(1, 0),
    )
    dV_block_ptr = tl.make_block_ptr(
        base=dV_ptr, shape=(seq_end, BLOCK_D),
        strides=(stride_dkt, stride_dkd), offsets=(k_abs, 0),
        block_shape=(BLOCK_N, BLOCK_D), order=(1, 0),
    )
    tl.store(dK_block_ptr, dk.to(k.dtype), boundary_check=(0,))
    tl.store(dV_block_ptr, dv.to(v.dtype), boundary_check=(0,))


# ---------------------------------------------------------------------------
# Backward kernel - dQ
# ---------------------------------------------------------------------------

@triton.jit
def _diff_noisy_attn_bwd_dq_kernel(
    Q, K, V, Out, dO, dQ, L_buf, D_buf,
    cu_seqlens, L0_arr, L1_arr,
    stride_qt, stride_qh, stride_qd,
    stride_kt, stride_kh, stride_kd,
    stride_vt, stride_vh, stride_vd,
    stride_ot, stride_oh, stride_od,
    n_heads: tl.constexpr,
    sm_scale,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    LOG2S: tl.constexpr,
    N_KV_GROUPS: tl.constexpr,
    MODE: tl.constexpr,
    USE_REGULAR_CAUSAL: tl.constexpr,
    NOISY_BIDIRECTIONAL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_doc = tl.program_id(2)
    pid_kv = pid_h // N_KV_GROUPS

    seq_start = tl.load(cu_seqlens + pid_doc).to(tl.int32)
    seq_end = tl.load(cu_seqlens + pid_doc + 1).to(tl.int32)
    doc_len = seq_end - seq_start
    L0 = tl.load(L0_arr + pid_doc).to(tl.int32)
    L1 = tl.load(L1_arr + pid_doc).to(tl.int32)

    start_m = pid_m * BLOCK_M
    if start_m >= doc_len:
        return

    offs_m = start_m + tl.arange(0, BLOCK_M)
    m_valid = offs_m < doc_len

    lower1, upper1, lower2, upper2 = _compute_ranges_for_mode(
        offs_m,
        L0,
        L1,
        doc_len,
        LOG2S,
        MODE,
        USE_REGULAR_CAUSAL,
        NOISY_BIDIRECTIONAL,
    )

    q_abs = seq_start + start_m
    Q_ptr = Q + pid_h * stride_qh
    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr, shape=(seq_end, BLOCK_D),
        strides=(stride_qt, stride_qd), offsets=(q_abs, 0),
        block_shape=(BLOCK_M, BLOCK_D), order=(1, 0),
    )
    q = tl.load(Q_block_ptr, boundary_check=(0,))
    q = (q * sm_scale).to(q.dtype)

    dO_ptr = dO + pid_h * stride_oh
    dO_block_ptr = tl.make_block_ptr(
        base=dO_ptr, shape=(seq_end, BLOCK_D),
        strides=(stride_ot, stride_od), offsets=(q_abs, 0),
        block_shape=(BLOCK_M, BLOCK_D), order=(1, 0),
    )
    do = tl.load(dO_block_ptr, boundary_check=(0,))

    L_base = L_buf + pid_h
    D_base = D_buf + pid_h
    offs_l = (seq_start + offs_m) * n_heads
    lse = tl.load(L_base + offs_l, mask=m_valid, other=0.0)
    Di = tl.load(D_base + offs_l, mask=m_valid, other=0.0)

    dq = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    K_ptr = K + pid_kv * stride_kh
    V_ptr = V + pid_kv * stride_vh

    # --- Range1 loop ---
    min_lower1 = tl.min(tl.where(m_valid, lower1, doc_len))
    max_upper1 = tl.max(tl.where(m_valid, upper1, 0))
    max_upper1 = tl.minimum(max_upper1, doc_len)
    range1_start = (min_lower1 // BLOCK_N) * BLOCK_N
    num_n1 = (max_upper1 - range1_start + BLOCK_N - 1) // BLOCK_N
    num_n1 = tl.where(min_lower1 < max_upper1, num_n1, 0)

    for it_n in range(num_n1):
        start_n = range1_start + it_n * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_valid = offs_n < doc_len

        k_abs = seq_start + start_n
        K_block_ptr = tl.make_block_ptr(
            base=K_ptr, shape=(seq_end, BLOCK_D),
            strides=(stride_kt, stride_kd), offsets=(k_abs, 0),
            block_shape=(BLOCK_N, BLOCK_D), order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            base=V_ptr, shape=(seq_end, BLOCK_D),
            strides=(stride_vt, stride_vd), offsets=(k_abs, 0),
            block_shape=(BLOCK_N, BLOCK_D), order=(1, 0),
        )
        k = tl.load(K_block_ptr, boundary_check=(0,))
        v = tl.load(V_block_ptr, boundary_check=(0,))

        cols = offs_n[None, :]
        mask = (cols >= lower1[:, None]) & (cols < upper1[:, None]) & m_valid[:, None] & n_valid[None, :]
        qk = tl.dot(q, tl.trans(k))
        qk = tl.where(mask, qk, -1.0e6)

        p = tl.exp2(qk - lse[:, None])

        dpT = tl.dot(do, tl.trans(v))
        ds = p * (dpT - Di[:, None])
        dq += tl.dot(ds.to(k.dtype), k)

    # --- Range2 loop ---
    min_lower2 = tl.min(tl.where(m_valid, lower2, doc_len))
    max_upper2 = tl.max(tl.where(m_valid, upper2, 0))
    max_upper2 = tl.minimum(max_upper2, doc_len)

    range2_start = (min_lower2 // BLOCK_N) * BLOCK_N
    num_n2 = (max_upper2 - range2_start + BLOCK_N - 1) // BLOCK_N
    num_n2 = tl.where(min_lower2 < max_upper2, num_n2, 0)

    for it_n in range(num_n2):
        start_n = range2_start + it_n * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_valid = offs_n < doc_len

        k_abs = seq_start + start_n
        K_block_ptr = tl.make_block_ptr(
            base=K_ptr, shape=(seq_end, BLOCK_D),
            strides=(stride_kt, stride_kd), offsets=(k_abs, 0),
            block_shape=(BLOCK_N, BLOCK_D), order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            base=V_ptr, shape=(seq_end, BLOCK_D),
            strides=(stride_vt, stride_vd), offsets=(k_abs, 0),
            block_shape=(BLOCK_N, BLOCK_D), order=(1, 0),
        )
        k = tl.load(K_block_ptr, boundary_check=(0,))
        v = tl.load(V_block_ptr, boundary_check=(0,))

        cols = offs_n[None, :]
        mask = (cols >= lower2[:, None]) & (cols < upper2[:, None]) & m_valid[:, None] & n_valid[None, :]
        qk = tl.dot(q, tl.trans(k))
        qk = tl.where(mask, qk, -1.0e6)

        p = tl.exp2(qk - lse[:, None])

        dpT = tl.dot(do, tl.trans(v))
        ds = p * (dpT - Di[:, None])
        dq += tl.dot(ds.to(k.dtype), k)

    dq = dq * scale

    dQ_ptr = dQ + pid_h * stride_qh
    dQ_block_ptr = tl.make_block_ptr(
        base=dQ_ptr, shape=(seq_end, BLOCK_D),
        strides=(stride_qt, stride_qd), offsets=(q_abs, 0),
        block_shape=(BLOCK_M, BLOCK_D), order=(1, 0),
    )
    tl.store(dQ_block_ptr, dq.to(q.dtype), boundary_check=(0,))


# ---------------------------------------------------------------------------
# Backward helper — fused D_i = rowsum(dO * O) kernel
# ---------------------------------------------------------------------------

@triton.jit
def _compute_delta_kernel(
    O, dO, Delta,
    BLOCK_D: tl.constexpr,
):
    """Compute D_i = sum_d(O[t,h,d] * dO[t,h,d]) for one (token, head) pair."""
    pid = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    base = pid * BLOCK_D
    o = tl.load(O + base + offs_d).to(tl.float32)
    do_val = tl.load(dO + base + offs_d).to(tl.float32)
    delta = tl.sum(o * do_val)
    tl.store(Delta + pid, delta)


# ---------------------------------------------------------------------------
# Autograd wrapper
# ---------------------------------------------------------------------------

class _DiffAttnNoisyFunction(torch.autograd.Function):
    """Autograd function — not part of the public API; use :func:`diff_attn_varlen_with_noisy_input`."""

    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens, L0_arr, L1_arr, staircase_size, max_seqlen,
                scale, dropout_p, mode, use_regular_causal, noisy_bidirectional):
        total_tokens, n_heads, head_dim = q.shape
        n_kv_heads = k.shape[1]
        n_kv_groups = n_heads // n_kv_heads
        log2s = int(math.log2(staircase_size))

        sm_scale = scale * 1.44269504  # scale * log2(e) for exp2-based softmax

        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        o = torch.empty_like(q)

        batch_size = cu_seqlens.numel() - 1
        BLOCK_D = head_dim
        BLOCK_M, BLOCK_N, fwd_num_warps, fwd_num_stages = _get_kernel_config(
            head_dim, n_heads, n_kv_heads, "fwd"
        )

        # LSE is always stored in float32 for numerical stability.
        lse = torch.empty(total_tokens, n_heads, dtype=torch.float32, device=q.device)

        num_m_tiles = (max_seqlen + BLOCK_M - 1) // BLOCK_M
        grid = (num_m_tiles, n_heads, batch_size)

        _diff_noisy_attn_fwd_kernel[grid](
            q, k, v, o, lse,
            cu_seqlens, L0_arr, L1_arr,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            n_heads=n_heads, sm_scale=sm_scale,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, LOG2S=log2s,
            N_KV_GROUPS=n_kv_groups,
            MODE=mode,
            USE_REGULAR_CAUSAL=use_regular_causal,
            NOISY_BIDIRECTIONAL=noisy_bidirectional,
            num_warps=fwd_num_warps, num_stages=fwd_num_stages,
        )

        ctx.save_for_backward(q, k, v, o, lse, cu_seqlens, L0_arr, L1_arr)
        ctx.sm_scale = sm_scale
        ctx.scale = scale
        ctx.log2s = log2s
        ctx.n_heads = n_heads
        ctx.BLOCK_D = BLOCK_D
        ctx.max_seqlen = max_seqlen
        ctx.n_kv_heads = n_kv_heads
        ctx.mode = mode
        ctx.use_regular_causal = use_regular_causal
        ctx.noisy_bidirectional = noisy_bidirectional
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse, cu_seqlens, L0_arr, L1_arr = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        scale = ctx.scale
        log2s = ctx.log2s
        n_heads = ctx.n_heads
        BLOCK_D = ctx.BLOCK_D
        n_kv_heads = ctx.n_kv_heads
        n_kv_groups = n_heads // n_kv_heads
        mode = ctx.mode
        use_regular_causal = ctx.use_regular_causal
        noisy_bidirectional = ctx.noisy_bidirectional

        do = do.contiguous()
        total_tokens = q.shape[0]
        batch_size = cu_seqlens.numel() - 1
        max_doc_len = ctx.max_seqlen

        BLOCK_M_DQ, BLOCK_N_DQ, dq_num_warps, dq_num_stages = _get_kernel_config(
            BLOCK_D, n_heads, n_kv_heads, "dq"
        )
        BLOCK_M_DKDV, BLOCK_N_DKDV, dkdv_num_warps, dkdv_num_stages = _get_kernel_config(
            BLOCK_D, n_heads, n_kv_heads, "dkdv"
        )

        # D_i = sum_d (dO_i * O_i)  — fused kernel avoids fp32 intermediate tensors.
        D = torch.empty(total_tokens, n_heads, dtype=torch.float32, device=q.device)
        _compute_delta_kernel[(total_tokens * n_heads,)](
            o, do, D,
            BLOCK_D=BLOCK_D,
            num_warps=max(1, BLOCK_D // 32),
        )

        dq = torch.empty_like(q)

        # --- dQ kernel ---
        num_m_tiles = (max_doc_len + BLOCK_M_DQ - 1) // BLOCK_M_DQ
        grid_dq = (num_m_tiles, n_heads, batch_size)
        _diff_noisy_attn_bwd_dq_kernel[grid_dq](
            q, k, v, o, do, dq, lse, D,
            cu_seqlens, L0_arr, L1_arr,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            n_heads=n_heads, sm_scale=sm_scale, scale=scale,
            BLOCK_M=BLOCK_M_DQ, BLOCK_N=BLOCK_N_DQ, BLOCK_D=BLOCK_D, LOG2S=log2s,
            N_KV_GROUPS=n_kv_groups,
            MODE=mode,
            USE_REGULAR_CAUSAL=use_regular_causal,
            NOISY_BIDIRECTIONAL=noisy_bidirectional,
            num_warps=dq_num_warps, num_stages=dq_num_stages,
        )

        # --- dK, dV kernel ---
        # Each Q head writes partial dK/dV; reduced across Q heads after.
        dk_buf = torch.empty(total_tokens, n_heads, BLOCK_D,
                             dtype=k.dtype, device=k.device)
        dv_buf = torch.empty(total_tokens, n_heads, BLOCK_D,
                             dtype=v.dtype, device=v.device)
        num_n_tiles = (max_doc_len + BLOCK_N_DKDV - 1) // BLOCK_N_DKDV
        grid_dkdv = (num_n_tiles, n_heads, batch_size)
        _diff_noisy_attn_bwd_dkdv_kernel[grid_dkdv](
            q, k, v, o, do, dk_buf, dv_buf, lse, D,
            cu_seqlens, L0_arr, L1_arr,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            dk_buf.stride(0), dk_buf.stride(1), dk_buf.stride(2),
            n_heads=n_heads, sm_scale=sm_scale, scale=scale,
            BLOCK_M=BLOCK_M_DKDV, BLOCK_N=BLOCK_N_DKDV, BLOCK_D=BLOCK_D, LOG2S=log2s,
            N_KV_GROUPS=n_kv_groups,
            MODE=mode,
            USE_REGULAR_CAUSAL=use_regular_causal,
            NOISY_BIDIRECTIONAL=noisy_bidirectional,
            num_warps=dkdv_num_warps, num_stages=dkdv_num_stages,
        )

        # Reduce partial dK/dV across Q heads sharing each KV head
        if n_kv_groups == 1:
            dk = dk_buf
            dv = dv_buf
        else:
            dk = dk_buf.view(total_tokens, n_kv_heads, n_kv_groups, BLOCK_D).sum(dim=2)
            dv = dv_buf.view(total_tokens, n_kv_heads, n_kv_groups, BLOCK_D).sum(dim=2)

        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def diff_attn_varlen_with_noisy_input(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    L0_arr: torch.Tensor,
    L1_arr: torch.Tensor,
    staircase_size: int = 4,
    max_seqlen: Optional[int] = None,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Fused attention with the diffusion noisy-input staircase mask.

    Computes multi-head attention with a structured block-causal mask designed
    for diffusion language model training where each document is laid out as
    ``[Prompt, CleanResponse, NoisyResponse]``.

    This function supports both forward and backward passes and is a drop-in
    replacement for ``flash_attn_varlen_func`` when the staircase mask is needed.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor of shape ``(total_tokens, n_heads, head_dim)``.
        Must be ``float16`` or ``bfloat16``.
    k : torch.Tensor
        Key tensor, same shape and dtype as *q*.
    v : torch.Tensor
        Value tensor, same shape and dtype as *q*.
    cu_seqlens : torch.Tensor
        Cumulative sequence lengths, shape ``(batch + 1,)``, dtype ``int32``.
        For example, two documents of lengths 100 and 200 give
        ``cu_seqlens = [0, 100, 300]``.
    L0_arr : torch.Tensor
        Prompt length for each document, shape ``(batch,)``, dtype ``int32``.
    L1_arr : torch.Tensor
        Response length for each document, shape ``(batch,)``, dtype ``int32``.
        Each document has ``L0 + 2 * L1`` total tokens.
    staircase_size : int, default 4
        Block size for the staircase attention pattern.
        Must be a power of 2 in ``{1, 2, 4, 8, 16, 32, 64}``.
        ``L0`` and ``L1`` must each be divisible by ``staircase_size``.
    max_seqlen : int or None, default None
        Maximum document length across the batch.  When provided, avoids an
        expensive GPU → CPU synchronisation (``cu_seqlens.max().item()``).
        **Always pass this in training / benchmarking.**
    scale : float or None, default None
        Attention scaling factor.  Defaults to ``1 / sqrt(head_dim)``.

    Returns
    -------
    torch.Tensor
        Output tensor, same shape and dtype as *q*:
        ``(total_tokens, n_heads, head_dim)``.

    Raises
    ------
    ValueError
        If *staircase_size* is not a valid power of 2, or if tensor shapes /
        dtypes are invalid.

    Examples
    --------
    Single document, staircase_size=4::

        import torch
        from diff_attn_noisy import diff_attn_varlen_with_noisy_input

        L0, L1, S = 128, 64, 4        # prompt=128, response=64, block=4
        total = L0 + 2 * L1           # 256 tokens
        n_heads, head_dim = 16, 64

        q = torch.randn(total, n_heads, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        cu_seqlens = torch.tensor([0, total], dtype=torch.int32, device="cuda")
        L0_arr = torch.tensor([L0], dtype=torch.int32, device="cuda")
        L1_arr = torch.tensor([L1], dtype=torch.int32, device="cuda")

        out = diff_attn_varlen_with_noisy_input(
            q, k, v, cu_seqlens, L0_arr, L1_arr,
            staircase_size=S, max_seqlen=total,
        )

    Two documents packed into one batch::

        L0s, L1s = [128, 64], [64, 32]
        doc_lens = [l0 + 2 * l1 for l0, l1 in zip(L0s, L1s)]
        total = sum(doc_lens)

        q = torch.randn(total, n_heads, head_dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        cu_seqlens = torch.tensor([0, doc_lens[0], total], dtype=torch.int32, device="cuda")
        L0_arr = torch.tensor(L0s, dtype=torch.int32, device="cuda")
        L1_arr = torch.tensor(L1s, dtype=torch.int32, device="cuda")

        out = diff_attn_varlen_with_noisy_input(
            q, k, v, cu_seqlens, L0_arr, L1_arr,
            staircase_size=4, max_seqlen=max(doc_lens),
        )
    """
    # ---- Input validation ----
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError(
            f"q, k, v must be 3-D (total_tokens, n_heads, head_dim), "
            f"got dims {q.dim()}, {k.dim()}, {v.dim()}"
        )
    if k.shape != v.shape:
        raise ValueError(
            f"k and v must have the same shape, got k={k.shape}, v={v.shape}"
        )
    if q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2]:
        raise ValueError(
            f"q and k/v must match in total_tokens (dim 0) and head_dim (dim 2), "
            f"got q={q.shape}, k={k.shape}"
        )
    n_q_heads, n_kv_heads = q.shape[1], k.shape[1]
    if n_q_heads % n_kv_heads != 0:
        raise ValueError(
            f"n_q_heads ({n_q_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
        )
    if q.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"q.dtype must be float16 or bfloat16, got {q.dtype}"
        )
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError(
            f"k and v must have the same dtype as q ({q.dtype}), "
            f"got k={k.dtype}, v={v.dtype}"
        )
    if staircase_size not in _VALID_STAIRCASE_SIZES:
        raise ValueError(
            f"staircase_size must be one of {_VALID_STAIRCASE_SIZES}, "
            f"got {staircase_size}"
        )

    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    if max_seqlen is None:
        max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())

    return _DiffAttnNoisyFunction.apply(
        q, k, v, cu_seqlens, L0_arr, L1_arr,
        staircase_size, max_seqlen, scale, 0.0, 0, True, False,
    )


def idlm_attn_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    block_size: int = 4,
    max_seqlen: Optional[int] = None,
    scale: Optional[float] = None,
    use_regular_causal: bool = True,
    noisy_block_bidirectional: bool = False,
) -> torch.Tensor:
    """Fused attention for I-DLM native ``[xt | x0]`` documents.

    Visibility matches ``training/model/*/modeling_sdar.py::block_diff_mask``:
    noisy/xt rows see their own xt block plus previous clean blocks; clean/x0
    rows see clean tokens causally (or block-causally when
    ``use_regular_causal=False``).
    """
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError(
            f"q, k, v must be 3-D (total_tokens, n_heads, head_dim), "
            f"got dims {q.dim()}, {k.dim()}, {v.dim()}"
        )
    if k.shape != v.shape:
        raise ValueError(f"k and v must have the same shape, got k={k.shape}, v={v.shape}")
    if q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2]:
        raise ValueError(f"q and k/v must match in total_tokens and head_dim, got q={q.shape}, k={k.shape}")
    n_q_heads, n_kv_heads = q.shape[1], k.shape[1]
    if n_q_heads % n_kv_heads != 0:
        raise ValueError(f"n_q_heads ({n_q_heads}) must be divisible by n_kv_heads ({n_kv_heads})")
    if q.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"q.dtype must be float16 or bfloat16, got {q.dtype}")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError(f"k and v must have the same dtype as q ({q.dtype}), got k={k.dtype}, v={v.dtype}")
    if block_size not in _VALID_STAIRCASE_SIZES:
        raise ValueError(f"block_size must be one of {_VALID_STAIRCASE_SIZES}, got {block_size}")

    doc_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    if torch.any(doc_lens % 2 != 0):
        raise ValueError("Each I-DLM document length must be even because layout is [xt | x0].")
    if max_seqlen is None:
        max_seqlen = int(doc_lens.max().item())

    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    dummy = torch.empty(cu_seqlens.numel() - 1, dtype=torch.int32, device=cu_seqlens.device)
    return _DiffAttnNoisyFunction.apply(
        q,
        k,
        v,
        cu_seqlens,
        dummy,
        dummy,
        block_size,
        max_seqlen,
        scale,
        0.0,
        1,
        bool(use_regular_causal),
        bool(noisy_block_bidirectional),
    )
