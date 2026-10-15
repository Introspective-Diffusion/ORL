# Triton fused MoE kernels
# Credit source reference maintained in original location
from typing import Tuple
import torch
import triton
import triton.language as tl
from liger_kernel.ops.swiglu import LigerSiLUMulFunction

# ---------------- Utility -----------------

def cdiv(a: int, b: int) -> int:
    return -(a // -b)

# --------------- Triton Kernels (copied) ---------------
@triton.jit
def moe_align_block_size_stage1(topk_ids_ptr, tokens_cnts_ptr, num_experts: tl.constexpr, numel: tl.constexpr, tokens_per_thread: tl.constexpr):
    pid = tl.program_id(0)
    start_idx = pid * tokens_per_thread
    off_c = (pid + 1) * num_experts
    for i in range(tokens_per_thread):
        if start_idx + i < numel:
            idx = tl.load(topk_ids_ptr + start_idx + i)
            token_cnt = tl.load(tokens_cnts_ptr + off_c + idx)
            tl.store(tokens_cnts_ptr + off_c + idx, token_cnt + 1)

@triton.jit
def moe_align_block_size_stage2(tokens_cnts_ptr, num_experts: tl.constexpr):
    pid = tl.program_id(0)
    last_cnt = 0
    for i in range(1, num_experts + 1):
        token_cnt = tl.load(tokens_cnts_ptr + i * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + i * num_experts + pid, last_cnt)

@triton.jit
def moe_align_block_size_stage3(total_tokens_post_pad_ptr, tokens_cnts_ptr, cumsum_ptr, num_experts: tl.constexpr, block_size: tl.constexpr):
    last_cumsum = 0
    off_cnt = num_experts * num_experts
    for i in range(1, num_experts + 1):
        token_cnt = tl.load(tokens_cnts_ptr + off_cnt + i - 1)
        last_cumsum = last_cumsum + tl.cdiv(token_cnt, block_size) * block_size
        tl.store(cumsum_ptr + i, last_cumsum)
    tl.store(total_tokens_post_pad_ptr, last_cumsum)

@triton.jit
def moe_align_block_size_stage4(topk_ids_ptr, sorted_token_ids_ptr, expert_ids_ptr, tokens_cnts_ptr, cumsum_ptr, num_experts: tl.constexpr, block_size: tl.constexpr, numel: tl.constexpr, tokens_per_thread: tl.constexpr):
    pid = tl.program_id(0)
    start_idx = tl.load(cumsum_ptr + pid)
    end_idx = tl.load(cumsum_ptr + pid + 1)
    for i in range(start_idx, end_idx, block_size):
        tl.store(expert_ids_ptr + i // block_size, pid)
    start_idx = pid * tokens_per_thread
    off_t = pid * num_experts
    for i in range(start_idx, tl.minimum(start_idx + tokens_per_thread, numel)):
        expert_id = tl.load(topk_ids_ptr + i)
        token_cnt = tl.load(tokens_cnts_ptr + off_t + expert_id)
        rank_post_pad = token_cnt + tl.load(cumsum_ptr + expert_id)
        tl.store(sorted_token_ids_ptr + rank_post_pad, i)
        tl.store(tokens_cnts_ptr + off_t + expert_id, token_cnt + 1)

# Helper
@triton.jit()
def col_major(pid, m, n, block_m: tl.constexpr, block_n: tl.constexpr):
    grid_m = tl.cdiv(m, block_m)
    pid_m = (pid % grid_m)
    pid_n = pid // grid_m
    return pid_m, pid_n

@triton.jit
def fused_moe_kernel(a_ptr, b_ptr, c_ptr, topk_weights_ptr, sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr, N, K, EM, num_valid_tokens, stride_am, stride_ak, stride_be, stride_bk, stride_bn, stride_cm, stride_cn, stride_weight, stride_token_id, block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr, MUL_ROUTED_WEIGHT: tl.constexpr, top_k: tl.constexpr, compute_type: tl.constexpr):
    pid = tl.program_id(axis=0)
    pid_m, pid_n = col_major(pid, EM, N, block_m, block_n,)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * block_m >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * block_m + tl.arange(0, block_m)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens
    offs_bn = (pid_n * block_n + tl.arange(0, block_n)) % N
    offs_k = tl.arange(0, block_k)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)
    off_experts = tl.load(expert_ids_ptr + pid_m)
    b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, block_k)):
        a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < K - k * block_k), other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * block_k, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += block_k * stride_ak
        b_ptrs += block_k * stride_bk
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token * stride_weight, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]
    accumulator = accumulator.to(compute_type)
    offs_cn = pid_n * block_n + tl.arange(0, block_n)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

# --------------- Python helpers ---------------

def moe_align_block_size(topk_ids: torch.Tensor, block_size: int, num_experts: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sorted_ids = torch.empty((topk_ids.numel() + num_experts * (block_size - 1), ), dtype=torch.int32, device=topk_ids.device)
    expert_ids = torch.empty((topk_ids.numel() + num_experts, ), dtype=torch.int32, device=topk_ids.device)
    sorted_ids.fill_(topk_ids.numel())
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)
    numel = topk_ids.numel()
    grid = (num_experts, )
    tokens_cnts = torch.zeros((num_experts + 1, num_experts), dtype=torch.int32, device=topk_ids.device)
    cumsum = torch.zeros((num_experts + 1,), dtype=torch.int32, device=topk_ids.device)
    tokens_per_thread = cdiv(numel, num_experts)
    moe_align_block_size_stage1[grid](topk_ids, tokens_cnts, num_experts, numel, tokens_per_thread)
    moe_align_block_size_stage2[grid](tokens_cnts, num_experts)
    moe_align_block_size_stage3[(1, )](num_tokens_post_pad, tokens_cnts, cumsum, num_experts, block_size)
    moe_align_block_size_stage4[grid](topk_ids, sorted_ids, expert_ids, tokens_cnts, cumsum, num_experts, block_size, numel, tokens_per_thread)
    return sorted_ids, expert_ids, num_tokens_post_pad

def invoke_fused_moe_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor, sorted_token_ids: torch.Tensor, expert_ids: torch.Tensor, num_tokens_post_padded: torch.Tensor, mul_routed_weight: bool, top_k: int, config: dict):
    EM = sorted_token_ids.shape[0]
    N = B.shape[1]
    grid = lambda META: (triton.cdiv(EM, META['block_m']) * triton.cdiv(N, META['block_n']), )
    fused_moe_kernel[grid](A, B, C, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded, B.shape[1], B.shape[2], sorted_token_ids.shape[0], topk_ids.numel(), A.stride(0), A.stride(1), B.stride(0), B.stride(2), B.stride(1), C.stride(1), C.stride(2), topk_weights.stride(1), sorted_token_ids.stride(0), MUL_ROUTED_WEIGHT=mul_routed_weight, top_k=top_k, compute_type=tl.bfloat16 if A.dtype == torch.bfloat16 else tl.float16, **config)

# --------------- Public API ---------------

def fused_moe(hidden_states: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor, inplace=False):
    assert hidden_states.is_contiguous(), "hidden_states must be contiguous"
    assert w1.is_contiguous() and w2.is_contiguous(), "expert weights must be contiguous"
    assert hidden_states.dtype in [torch.float16, torch.bfloat16]
    assert hidden_states.shape[1] == w1.shape[2], f"Hidden dim mismatch {hidden_states.shape} vs {w1.shape}"
    # Additional shape safety
    assert w1.shape[0] == w2.shape[0], "Expert count mismatch"
    assert w1.shape[1] // 2 == w2.shape[2], "Intermediate dim mismatch between w1 and w2"
    M, _ = hidden_states.shape
    E, N, _ = w1.shape
    config = {'block_m': 64, 'block_n': 64, 'block_k': 32}
    if topk_ids.numel() <= w1.shape[0]:
        config = {'block_m': 16, 'block_n': 32, 'block_k': 64}
    intermediate_cache1 = torch.empty((M, topk_ids.shape[1], N), device=hidden_states.device, dtype=hidden_states.dtype)
    intermediate_cache2 = torch.empty((M * topk_ids.shape[1], N // 2), device=hidden_states.device, dtype=hidden_states.dtype)
    intermediate_cache3 = torch.empty((M, topk_ids.shape[1], w2.shape[1]), device=hidden_states.device, dtype=hidden_states.dtype)
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(topk_ids, config['block_m'], E)
    invoke_fused_moe_kernel(hidden_states, w1, intermediate_cache1, topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded, False, topk_ids.shape[1], config)
    intermediate_cache2 = LigerSiLUMulFunction.apply(intermediate_cache1.view(-1, N)[:, :N // 2].contiguous(), intermediate_cache1.view(-1, N)[:, N // 2:].contiguous())
    invoke_fused_moe_kernel(intermediate_cache2, w2, intermediate_cache3, topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded, True, 1, config)
    if inplace:
        return torch.sum(intermediate_cache3, dim=1, out=hidden_states)
    return torch.sum(intermediate_cache3, dim=1)
