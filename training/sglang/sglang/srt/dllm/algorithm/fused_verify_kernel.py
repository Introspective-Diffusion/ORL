"""
Fused Triton kernel for speculative verification + correction sampling.

Combines softmax + p/q gather + ratio + accept/reject + Gumbel-max correction
into a single kernel, eliminating 4+ separate GPU kernel launches and avoiding
materialization of full [nv, V] intermediate tensors.

For accepted rows (~78% for N=2), the kernel early-exits after the softmax pass,
skipping the expensive correction computation entirely.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_verify_kernel(
    # Inputs
    clean_logits_ptr,  # [nv, V] - model logits at verify positions
    draft_probs_ptr,   # [nv, V] - draft probability distributions
    spec_vals_ptr,     # [nv] int64 - speculative token IDs
    rand_accept_ptr,   # [nv] float32 - uniform random for accept/reject
    gumbel_seed,       # int64 - seed for Gumbel noise RNG
    # Outputs
    accepted_ptr,      # [nv] int32 - 1 if accepted, 0 if rejected
    corr_tokens_ptr,   # [nv] int64 - correction token (only valid if rejected)
    # Params
    V: tl.constexpr,
    INV_TEMP: tl.constexpr,  # 1.0 / temperature (pre-computed)
    ALPHA: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Fused speculative verification kernel.

    For each row (verify request):
    1. Online softmax of clean_logits to get p(x)
    2. Gather p(spec_val) and q(spec_val) from draft_probs
    3. Accept/reject: accept if rand < min(1, p/(alpha*q))
    4. If rejected: compute correction dist = max(0, p-q), sample via Gumbel-max
    """
    row = tl.program_id(0)
    base = row * V

    # === Pass 1: Online softmax (max + sum_exp in one pass) ===
    row_max = float("-inf")
    row_sum = 0.0

    for start in range(0, V, BLOCK_V):
        offs = start + tl.arange(0, BLOCK_V)
        mask = offs < V
        logits = tl.load(clean_logits_ptr + base + offs, mask=mask, other=float("-inf"))
        logits = logits.to(tl.float32) * INV_TEMP

        block_max = tl.max(logits, axis=0)
        new_max = tl.maximum(row_max, block_max)
        # Rescale running sum and add new block
        row_sum = row_sum * tl.exp(row_max - new_max) + tl.sum(
            tl.exp(logits - new_max), axis=0
        )
        row_max = new_max

    # === Gather p(spec_val) and q(spec_val) ===
    spec_val = tl.load(spec_vals_ptr + row)
    spec_logit = tl.load(clean_logits_ptr + base + spec_val).to(tl.float32) * INV_TEMP
    p_spec = tl.exp(spec_logit - row_max) / row_sum

    q_spec = tl.load(draft_probs_ptr + row * V + spec_val).to(tl.float32)

    # === Accept/reject decision ===
    denom = q_spec * ALPHA
    ratio = tl.where(denom > 0, p_spec / denom, 0.0)
    rand_val = tl.load(rand_accept_ptr + row)
    accepted = (ratio >= 1.0) | (rand_val < ratio)

    tl.store(accepted_ptr + row, accepted.to(tl.int32))

    # Early exit for accepted rows (skip expensive correction)
    if accepted:
        tl.store(corr_tokens_ptr + row, 0)
        return

    # === Pass 2: Correction sampling via Gumbel-max (rejected rows only) ===
    # Sample from max(0, p(x) - q(x)) using the Gumbel-max trick:
    #   argmax_x [log(max(0, p(x) - q(x))) + Gumbel(0,1)]
    # where Gumbel(0,1) = -log(-log(U)), U ~ Uniform(0,1)
    best_score = float("-inf")
    best_token = 0

    for start in range(0, V, BLOCK_V):
        offs = start + tl.arange(0, BLOCK_V)
        mask = offs < V

        logits = tl.load(clean_logits_ptr + base + offs, mask=mask, other=float("-inf"))
        logits = logits.to(tl.float32) * INV_TEMP
        p_vals = tl.exp(logits - row_max) / row_sum

        q_vals = tl.load(draft_probs_ptr + row * V + offs, mask=mask, other=0.0)
        q_vals = q_vals.to(tl.float32)

        # Correction distribution: max(0, p - q)
        corr = tl.maximum(p_vals - q_vals, 0.0)

        # Gumbel-max: log(corr) + Gumbel noise
        log_corr = tl.where(corr > 1e-20, tl.log(corr), float("-inf"))
        # tl.rand generates uniform [0, 1) with Philox RNG
        u = tl.rand(gumbel_seed + row, offs)
        gumbel = -tl.log(-tl.log(tl.clamp(u, min=1e-10, max=1.0 - 1e-7)) + 1e-20)
        scores = log_corr + gumbel
        scores = tl.where(mask, scores, float("-inf"))

        # Track block-level argmax
        block_max_score = tl.max(scores, axis=0)
        if block_max_score > best_score:
            # Find the token with the max score in this block
            is_max = scores == block_max_score
            # Use minimum of matching offsets (deterministic tie-breaking)
            max_off = tl.min(tl.where(is_max, offs, V), axis=0)
            best_score = block_max_score
            best_token = max_off

    tl.store(corr_tokens_ptr + row, best_token.to(tl.int64))


def fused_spec_verify(
    clean_logits: torch.Tensor,   # [nv, V] model logits
    draft_probs: torch.Tensor,    # [nv, V] draft probability distributions
    spec_vals: torch.Tensor,      # [nv] int64 speculative token IDs
    temperature: float = 1.0,
    alpha: float = 1.0,
    gumbel_seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run fused speculative verification.

    Returns:
        accepted: [nv] int32 tensor (1=accepted, 0=rejected)
        corr_tokens: [nv] int64 tensor (correction token, valid only for rejected rows)
    """
    nv, V = clean_logits.shape
    device = clean_logits.device

    # Pre-generate uniform random numbers for accept/reject
    rand_accept = torch.rand(nv, device=device, dtype=torch.float32)

    # Output tensors
    accepted = torch.empty(nv, device=device, dtype=torch.int32)
    corr_tokens = torch.empty(nv, device=device, dtype=torch.int64)

    # Choose BLOCK_V based on vocab size
    if V <= 32768:
        BLOCK_V = 1024
    elif V <= 65536:
        BLOCK_V = 2048
    else:
        BLOCK_V = 4096

    inv_temp = 1.0 / temperature if temperature > 0 else 1.0

    # Ensure draft_probs is float32 and contiguous
    if draft_probs.dtype != torch.float32:
        draft_probs = draft_probs.to(torch.float32)
    if not draft_probs.is_contiguous():
        draft_probs = draft_probs.contiguous()
    if not clean_logits.is_contiguous():
        clean_logits = clean_logits.contiguous()

    grid = (nv,)
    _fused_verify_kernel[grid](
        clean_logits,
        draft_probs,
        spec_vals,
        rand_accept,
        gumbel_seed,
        accepted,
        corr_tokens,
        V=V,
        INV_TEMP=inv_temp,
        ALPHA=alpha,
        BLOCK_V=BLOCK_V,
        num_warps=8,
        num_stages=2,
    )

    return accepted, corr_tokens


@triton.jit
def _fused_verify_from_logits_kernel(
    # Inputs
    clean_logits_ptr,  # [nv, V] - model logits at verify positions
    draft_logits_ptr,  # [nv, V] - draft logits (NOT probs)
    spec_vals_ptr,     # [nv] int64
    rand_accept_ptr,   # [nv] float32
    gumbel_seed,       # int64
    # Outputs
    accepted_ptr,      # [nv] int32
    corr_tokens_ptr,   # [nv] int64
    # Params
    V: tl.constexpr,
    INV_TEMP: tl.constexpr,
    ALPHA: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Fused verify that takes draft LOGITS (not probs).
    Computes softmax(clean) and softmax(draft) internally.
    """
    row = tl.program_id(0)
    base = row * V

    # === Pass 1: Dual online softmax (clean + draft in one pass) ===
    c_max = float("-inf")
    c_sum = 0.0
    d_max = float("-inf")
    d_sum = 0.0

    for start in range(0, V, BLOCK_V):
        offs = start + tl.arange(0, BLOCK_V)
        mask = offs < V
        # Clean logits
        c_logits = tl.load(clean_logits_ptr + base + offs, mask=mask, other=float("-inf"))
        c_logits = c_logits.to(tl.float32) * INV_TEMP
        c_bmax = tl.max(c_logits, axis=0)
        c_new_max = tl.maximum(c_max, c_bmax)
        c_sum = c_sum * tl.exp(c_max - c_new_max) + tl.sum(tl.exp(c_logits - c_new_max), axis=0)
        c_max = c_new_max
        # Draft logits
        d_logits = tl.load(draft_logits_ptr + base + offs, mask=mask, other=float("-inf"))
        d_logits = d_logits.to(tl.float32) * INV_TEMP
        d_bmax = tl.max(d_logits, axis=0)
        d_new_max = tl.maximum(d_max, d_bmax)
        d_sum = d_sum * tl.exp(d_max - d_new_max) + tl.sum(tl.exp(d_logits - d_new_max), axis=0)
        d_max = d_new_max

    # Gather p(spec) and q(spec)
    spec_val = tl.load(spec_vals_ptr + row)
    c_spec = tl.load(clean_logits_ptr + base + spec_val).to(tl.float32) * INV_TEMP
    p_spec = tl.exp(c_spec - c_max) / c_sum
    d_spec = tl.load(draft_logits_ptr + base + spec_val).to(tl.float32) * INV_TEMP
    q_spec = tl.exp(d_spec - d_max) / d_sum

    # Accept/reject
    denom = q_spec * ALPHA
    ratio = tl.where(denom > 0, p_spec / denom, 0.0)
    rand_val = tl.load(rand_accept_ptr + row)
    accepted = (ratio >= 1.0) | (rand_val < ratio)
    tl.store(accepted_ptr + row, accepted.to(tl.int32))

    if accepted:
        tl.store(corr_tokens_ptr + row, 0)
        return

    # === Pass 2: Correction via Gumbel-max ===
    best_score = float("-inf")
    best_token = 0
    for start in range(0, V, BLOCK_V):
        offs = start + tl.arange(0, BLOCK_V)
        mask = offs < V
        c_logits = tl.load(clean_logits_ptr + base + offs, mask=mask, other=float("-inf"))
        c_logits = c_logits.to(tl.float32) * INV_TEMP
        p_vals = tl.exp(c_logits - c_max) / c_sum
        d_logits = tl.load(draft_logits_ptr + base + offs, mask=mask, other=float("-inf"))
        d_logits = d_logits.to(tl.float32) * INV_TEMP
        q_vals = tl.exp(d_logits - d_max) / d_sum
        corr = tl.maximum(p_vals - q_vals, 0.0)
        log_corr = tl.where(corr > 1e-20, tl.log(corr), float("-inf"))
        u = tl.rand(gumbel_seed + row, offs)
        gumbel = -tl.log(-tl.log(tl.clamp(u, min=1e-10, max=1.0 - 1e-7)) + 1e-20)
        scores = log_corr + gumbel
        scores = tl.where(mask, scores, float("-inf"))
        block_max_score = tl.max(scores, axis=0)
        if block_max_score > best_score:
            is_max = scores == block_max_score
            max_off = tl.min(tl.where(is_max, offs, V), axis=0)
            best_score = block_max_score
            best_token = max_off

    tl.store(corr_tokens_ptr + row, best_token.to(tl.int64))


def fused_spec_verify_from_logits(
    clean_logits: torch.Tensor,   # [nv, V] model logits
    draft_logits: torch.Tensor,   # [nv, V] draft logits (NOT probs)
    spec_vals: torch.Tensor,      # [nv] int64
    temperature: float = 1.0,
    alpha: float = 1.0,
    gumbel_seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused verify that takes draft logits instead of probs.
    Eliminates the need for Python-side F.softmax on draft logits.
    """
    nv, V = clean_logits.shape
    device = clean_logits.device

    rand_accept = torch.rand(nv, device=device, dtype=torch.float32)
    accepted = torch.empty(nv, device=device, dtype=torch.int32)
    corr_tokens = torch.empty(nv, device=device, dtype=torch.int64)

    if V <= 32768:
        BLOCK_V = 1024
    elif V <= 65536:
        BLOCK_V = 2048
    else:
        BLOCK_V = 4096

    inv_temp = 1.0 / temperature if temperature > 0 else 1.0

    if not clean_logits.is_contiguous():
        clean_logits = clean_logits.contiguous()
    if not draft_logits.is_contiguous():
        draft_logits = draft_logits.contiguous()

    grid = (nv,)
    _fused_verify_from_logits_kernel[grid](
        clean_logits, draft_logits, spec_vals, rand_accept, gumbel_seed,
        accepted, corr_tokens,
        V=V, INV_TEMP=inv_temp, ALPHA=alpha, BLOCK_V=BLOCK_V,
        num_warps=8, num_stages=2,
    )
    return accepted, corr_tokens


@triton.jit
def _fused_verify_multi_spec_kernel(
    # Inputs - interleaved layout: [nv * num_specs, V]
    clean_logits_ptr,  # [nv * num_specs, V]
    draft_probs_ptr,   # [nv * num_specs, V]
    spec_vals_ptr,     # [nv * num_specs] int64
    rand_accept_ptr,   # [nv * num_specs] float32
    gumbel_seed,       # int64
    # Outputs
    accepted_ptr,      # [nv * num_specs] int32
    corr_tokens_ptr,   # [nv * num_specs] int64
    # Params
    V: tl.constexpr,
    INV_TEMP: tl.constexpr,
    ALPHA: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Same as _fused_verify_kernel but for multi-spec layout.

    Each row is independently verified. The caller handles left-to-right
    rejection logic on the CPU side.
    """
    row = tl.program_id(0)
    base = row * V

    # Online softmax
    row_max = float("-inf")
    row_sum = 0.0
    for start in range(0, V, BLOCK_V):
        offs = start + tl.arange(0, BLOCK_V)
        mask = offs < V
        logits = tl.load(clean_logits_ptr + base + offs, mask=mask, other=float("-inf"))
        logits = logits.to(tl.float32) * INV_TEMP
        block_max = tl.max(logits, axis=0)
        new_max = tl.maximum(row_max, block_max)
        row_sum = row_sum * tl.exp(row_max - new_max) + tl.sum(
            tl.exp(logits - new_max), axis=0
        )
        row_max = new_max

    spec_val = tl.load(spec_vals_ptr + row)
    spec_logit = tl.load(clean_logits_ptr + base + spec_val).to(tl.float32) * INV_TEMP
    p_spec = tl.exp(spec_logit - row_max) / row_sum
    q_spec = tl.load(draft_probs_ptr + row * V + spec_val).to(tl.float32)

    denom = q_spec * ALPHA
    ratio = tl.where(denom > 0, p_spec / denom, 0.0)
    rand_val = tl.load(rand_accept_ptr + row)
    accepted = (ratio >= 1.0) | (rand_val < ratio)
    tl.store(accepted_ptr + row, accepted.to(tl.int32))

    if accepted:
        tl.store(corr_tokens_ptr + row, 0)
        return

    # Correction via Gumbel-max
    best_score = float("-inf")
    best_token = 0
    for start in range(0, V, BLOCK_V):
        offs = start + tl.arange(0, BLOCK_V)
        mask = offs < V
        logits = tl.load(clean_logits_ptr + base + offs, mask=mask, other=float("-inf"))
        logits = logits.to(tl.float32) * INV_TEMP
        p_vals = tl.exp(logits - row_max) / row_sum
        q_vals = tl.load(draft_probs_ptr + row * V + offs, mask=mask, other=0.0).to(tl.float32)
        corr = tl.maximum(p_vals - q_vals, 0.0)
        log_corr = tl.where(corr > 1e-20, tl.log(corr), float("-inf"))
        u = tl.rand(gumbel_seed + row, offs)
        gumbel = -tl.log(-tl.log(tl.clamp(u, min=1e-10, max=1.0 - 1e-7)) + 1e-20)
        scores = log_corr + gumbel
        scores = tl.where(mask, scores, float("-inf"))
        block_max_score = tl.max(scores, axis=0)
        if block_max_score > best_score:
            is_max = scores == block_max_score
            max_off = tl.min(tl.where(is_max, offs, V), axis=0)
            best_score = block_max_score
            best_token = max_off

    tl.store(corr_tokens_ptr + row, best_token.to(tl.int64))
