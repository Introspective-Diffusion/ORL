"""Fused per-row logits kernels for the greedy dLLM decode step.

One denoise step needs, per block position (row of the [tokens, vocab]
logits): the greedy proposal (argmax), the row entropy at temperature 1
(controller signal), and the confidence probability of the proposal under the
per-position confidence temperature T_i, i.e. softmax(x / T_i)[argmax]. The
torch formulation materialises log_softmax / softmax over the full vocab
several times (~8 passes over [32, 157k] fp32 and ~25 launches per step). It
is the same numbers computed in two passes here:

  row_stats(x)           -> argmax, x_max, logsumexp(x), entropy(x)
  row_conf_prob(x, T, .) -> softmax(x / T)[argmax] == 1 / sum_j exp((x_j - x_max) / T)

Values agree with the torch path up to fp32 summation-order noise (a few
ulp); greedy trajectories can therefore diverge on near-tie decisions,
exactly as any kernel change does (see LLADA2_FUSED_NEXT handoff, numerics).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _row_stats_kernel(
    X,
    stride_x,
    V,
    AMAX,
    XMAX,
    LSE,
    ENT,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    xp = X + row.to(tl.int64) * stride_x
    # running max; a large finite sentinel instead of -inf so the first-chunk
    # rescale (m - m_new) * s is (finite * 0) = 0 rather than (-inf * 0) = nan
    m = tl.full((), -1.0e30, tl.float32)
    s = tl.zeros((), tl.float32)  # sum exp(x - m)
    t = tl.zeros((), tl.float32)  # sum (x - m) exp(x - m)
    best_idx = tl.full((), 2**31 - 1, tl.int32)
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < V
        x = tl.load(xp + offs, mask=mask, other=float("-inf")).to(tl.float32)
        cm = tl.max(x, 0)
        m_new = tl.maximum(m, cm)
        alpha = tl.exp(m - m_new)  # 0 on the first chunk (m = -inf)
        u = x - m_new  # <= 0; -inf on masked lanes
        e = tl.exp(u)  # 0 on masked lanes
        s_chunk = tl.sum(e, 0)
        t_chunk = tl.sum(tl.where(mask, u * e, 0.0), 0)
        # rescale the running sums to the new max: sum (x - m_new) e^(x - m_new)
        #   = alpha * (t_old + (m_old - m_new) * s_old)
        t = alpha * (t + (m - m_new) * s) + t_chunk
        s = alpha * s + s_chunk
        # first index attaining the maximum (torch.argmax semantics)
        idx_c = tl.min(tl.where(x == cm, offs, 2**31 - 1), 0)
        take = cm > m  # strictly greater than the previous running max
        best_idx = tl.where(take, idx_c, best_idx)
        m = m_new
    log_s = tl.log(s)
    tl.store(AMAX + row, best_idx.to(tl.int64))
    tl.store(XMAX + row, m)
    tl.store(LSE + row, m + log_s)
    tl.store(ENT + row, log_s - t / s)


@triton.jit
def _row_conf_prob_kernel(
    X,
    stride_x,
    V,
    T,
    stride_t,
    t_scalar,
    XMAX,
    OUT,
    T_FROM_PTR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    xp = X + row.to(tl.int64) * stride_x
    if T_FROM_PTR:
        temp = tl.load(T + row * stride_t).to(tl.float32)
    else:
        # uniform temperature passed as a kernel scalar: no per-step
        # torch.tensor(float, device=cuda) (a pageable H2D copy that blocks the
        # host until the stream -- i.e. the model forward -- has drained)
        temp = t_scalar
    xm = tl.load(XMAX + row)
    s = tl.zeros((), tl.float32)
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < V
        x = tl.load(xp + offs, mask=mask, other=float("-inf")).to(tl.float32)
        s += tl.sum(tl.exp((x - xm) / temp), 0)
    tl.store(OUT + row, 1.0 / s)


@triton.jit
def _row_conf_prob_ctrl_kernel(
    X,
    stride_x,
    V,
    ENT,
    HREF,
    PREV_T,
    PROMPT,
    XMAX,
    OUT_P,
    OUT_T,
    t_min,
    t_max,
    gamma,
    HAS_PREV: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """row_conf_prob with the per-position entropy-gated confidence temperature
    computed inline (adaptive_confidence_temperature.py, per_position light
    path): T_i = clamp(t_min + (t_max - t_min) * clip(h_i / H_ref, 0, 1)^gamma),
    prompt rows pinned to t_max, monotone no-reheat min with the previous
    step's vector when HAS_PREV. Writes T_i (the next step's PREV_T) and
    softmax(x / T_i)[argmax]."""
    row = tl.program_id(0)
    xp = X + row.to(tl.int64) * stride_x
    h = tl.load(ENT + row)
    href = tl.maximum(tl.load(HREF), 1e-8)
    ratio = tl.minimum(tl.maximum(h / href, 0.0), 1.0)
    shaped = tl.where(ratio > 0.0, tl.exp2(gamma * tl.log2(ratio)), 0.0)
    prop = tl.minimum(tl.maximum(t_min + (t_max - t_min) * shaped, t_min), t_max)
    is_prompt = tl.load(PROMPT + row) != 0
    prop = tl.where(is_prompt, t_max, prop)
    if HAS_PREV:
        t = tl.minimum(tl.load(PREV_T + row), prop)
        t = tl.where(is_prompt, t_max, t)
    else:
        t = prop
    tl.store(OUT_T + row, t)
    temp = tl.maximum(t, 1e-6)
    xm = tl.load(XMAX + row)
    s = tl.zeros((), tl.float32)
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < V
        x = tl.load(xp + offs, mask=mask, other=float("-inf")).to(tl.float32)
        s += tl.sum(tl.exp((x - xm) / temp), 0)
    tl.store(OUT_P + row, 1.0 / s)


_BLOCK = 4096


def row_stats(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """argmax[int64], x_max, logsumexp, entropy (nats, T=1) per row; fp32."""
    assert logits.ndim == 2 and logits.is_cuda
    if logits.stride(1) != 1:
        logits = logits.contiguous()
    n, v = logits.shape
    amax = torch.empty(n, dtype=torch.int64, device=logits.device)
    xmax = torch.empty(n, dtype=torch.float32, device=logits.device)
    lse = torch.empty_like(xmax)
    ent = torch.empty_like(xmax)
    _row_stats_kernel[(n,)](
        logits, logits.stride(0), v, amax, xmax, lse, ent, BLOCK=_BLOCK, num_warps=8
    )
    return amax, xmax, lse, ent


def row_conf_prob(
    logits: torch.Tensor, temperature: torch.Tensor | float, xmax: torch.Tensor
) -> torch.Tensor:
    """softmax(x_i / T_i)[argmax_i] for every row; T may be a python float, a
    0-dim tensor (uniform) or a [rows] tensor (per position). Clamped at 1e-6
    like confidence_probs_from_logits."""
    assert logits.ndim == 2 and logits.is_cuda
    if logits.stride(1) != 1:
        logits = logits.contiguous()
    n, v = logits.shape
    out = torch.empty(n, dtype=torch.float32, device=logits.device)
    if not isinstance(temperature, torch.Tensor):
        t_scalar = max(float(temperature), 1e-6)
        _row_conf_prob_kernel[(n,)](
            logits, logits.stride(0), v, out, 0, t_scalar, xmax, out,
            T_FROM_PTR=False, BLOCK=_BLOCK, num_warps=8,
        )
        return out
    temperature = temperature.to(device=logits.device, dtype=torch.float32).clamp_min(1e-6)
    if temperature.dim() == 0:
        t_ptr, stride_t = temperature.reshape(1), 0
    else:
        t_ptr, stride_t = temperature.reshape(-1).contiguous(), 1
        assert t_ptr.numel() == n
    _row_conf_prob_kernel[(n,)](
        logits, logits.stride(0), v, t_ptr, stride_t, 0.0, xmax, out,
        T_FROM_PTR=True, BLOCK=_BLOCK, num_warps=8,
    )
    return out


def row_conf_prob_ctrl(
    logits: torch.Tensor,
    xmax: torch.Tensor,
    entropy: torch.Tensor,
    h_ref: torch.Tensor,
    prev_t: torch.Tensor | None,
    prompt_mask: torch.Tensor,
    t_min: float,
    t_max: float,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(proposal confidence prob, per-position temperature) with the
    entropy-gated controller folded into the pass over the logits."""
    assert logits.ndim == 2 and logits.is_cuda
    if logits.stride(1) != 1:
        logits = logits.contiguous()
    n, v = logits.shape
    out_p = torch.empty(n, dtype=torch.float32, device=logits.device)
    out_t = torch.empty(n, dtype=torch.float32, device=logits.device)
    prompt_u8 = prompt_mask.reshape(-1).to(torch.uint8)
    assert prompt_u8.numel() == n and entropy.numel() == n
    href = h_ref.reshape(1).to(torch.float32)
    has_prev = prev_t is not None and prev_t.numel() == n
    _row_conf_prob_ctrl_kernel[(n,)](
        logits, logits.stride(0), v, entropy, href,
        prev_t.reshape(-1).to(torch.float32) if has_prev else out_t,  # dummy ptr when unused
        prompt_u8, xmax, out_p, out_t,
        float(t_min), float(t_max), float(gamma),
        HAS_PREV=has_prev, BLOCK=_BLOCK, num_warps=8,
    )
    return out_p, out_t
