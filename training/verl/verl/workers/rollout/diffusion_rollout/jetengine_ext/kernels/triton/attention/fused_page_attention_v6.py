import torch
import triton
import triton.language as tl
import math
from typing import Optional

LOG2E = math.log2(math.e)

@triton.jit
def fused_kv_cache_attention_kernel(
    q_ptr, q_stride_seq, q_stride_head, q_stride_dim,
    k_ptr, k_stride_seq, k_stride_head, k_stride_dim,
    v_ptr, v_stride_seq, v_stride_head, v_stride_dim,
    k_cache_ptr, k_cache_stride_block, k_cache_stride_pos, k_cache_stride_head, k_cache_stride_dim,
    v_cache_ptr, v_cache_stride_block, v_cache_stride_pos, v_cache_stride_head, v_cache_stride_dim,
    block_tables_ptr, block_tables_stride_seq, block_tables_stride_block,
    seq_lens_k_ptr,
    o_ptr, o_stride_seq, o_stride_head, o_stride_dim,
    BLOCK_LEN: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr, 
    KV_GROUP_SIZE: tl.constexpr,
    SM_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    seq_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    seq_start_q = seq_idx * BLOCK_LEN
    seq_len_k_cache = tl.load(seq_lens_k_ptr + seq_idx)
    kv_head_idx = head_idx // KV_GROUP_SIZE
    
    d_idx = tl.arange(0, BLOCK_DMODEL)
    n_idx = tl.arange(0, BLOCK_N)
    
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    q_offs = tl.arange(0, BLOCK_M)
    q_mask = q_offs < BLOCK_LEN
    
    q_ptrs = (
        q_ptr + (seq_start_q + q_offs[:, None]) * q_stride_seq + head_idx * q_stride_head + d_idx[None, :] * q_stride_dim
    )
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0, cache_modifier=".ca")
    
    # This part handles attention within the new tokens themselves
    kv_offs = (seq_start_q + n_idx[:, None]) * k_stride_seq + kv_head_idx * k_stride_head + d_idx[None, :] * k_stride_dim

    k_ptrs = k_ptr + kv_offs
    v_ptrs = v_ptr + kv_offs
    kv_mask = (n_idx[:, None] < BLOCK_LEN)
    k = tl.load(k_ptrs, mask=kv_mask, other=0.0, cache_modifier=".ca")
    v = tl.load(v_ptrs, mask=kv_mask, other=0.0, cache_modifier=".ca")
    
    qk = tl.dot(q, tl.trans(k), allow_tf32=True)
    qk = qk.to(tl.float32) * SM_SCALE
    
    nmask = (n_idx[None, :] < BLOCK_LEN)
    qk = tl.where(nmask, qk, -float('inf'))
    
    m_new = tl.maximum(m_i, tl.max(qk, axis=1))
    alpha = tl.exp2((m_i - m_new))
    p = tl.exp2((qk - m_new[:, None]))
    
    acc = acc * alpha[:, None] + tl.dot(p, v.to(p.dtype), allow_tf32=True)
    l_i = l_i * alpha + tl.sum(p, axis=1)
    m_i = m_new

    num_logical_blocks = (seq_len_k_cache + BLOCK_SIZE - 1) // BLOCK_SIZE
    kv_pos = 0
    
    # Loop over blocks of keys and values in the KV cache
    for logical_block_num in range(num_logical_blocks):
        physical_block_id = tl.load(
            block_tables_ptr + seq_idx * block_tables_stride_seq + logical_block_num * block_tables_stride_block
        )
        tokens_in_block = tl.minimum(seq_len_k_cache - kv_pos, BLOCK_SIZE)
        
        for tile_n_start in range(0, tokens_in_block, BLOCK_N):
            actual_n = tl.minimum(tokens_in_block - tile_n_start, BLOCK_N)
            
            k_cache_base = k_cache_ptr + physical_block_id * k_cache_stride_block + kv_head_idx * k_cache_stride_head
            v_cache_base = v_cache_ptr + physical_block_id * v_cache_stride_block + kv_head_idx * v_cache_stride_head
            
            offs_n = (tile_n_start + n_idx)[:, None] * k_cache_stride_pos
            k_ptrs = k_cache_base + offs_n + d_idx[None, :] * k_cache_stride_dim
            v_ptrs = v_cache_base + offs_n + d_idx[None, :] * v_cache_stride_dim
            
            kv_mask = (n_idx[:, None] < actual_n)
            k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
            v = tl.load(v_ptrs, mask=kv_mask, other=0.0)
            
            qk = tl.dot(q, tl.trans(k), allow_tf32=True)
            qk = qk.to(tl.float32) * SM_SCALE
            
            nmask = n_idx[None, :] < actual_n
            qk = tl.where(nmask, qk, -float('inf'))
            
            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.exp2((m_i - m_new))
            p = tl.exp2((qk - m_new[:, None]))
            
            acc = acc * alpha[:, None] + tl.dot(p, v.to(p.dtype), allow_tf32=True)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new

        kv_pos += tokens_in_block

    

    l_safe = tl.where(l_i == 0, 1.0, l_i)
    acc = acc / l_safe[:, None]
    
    o_ptrs = (
        o_ptr + (seq_start_q + q_offs[:, None]) * o_stride_seq + head_idx * o_stride_head + d_idx[None, :] * o_stride_dim
    )
    tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), mask=q_mask[:, None])
    
def fused_kv_cache_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens_k: torch.Tensor, # New argument
    block_len: int,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    q_total_tokens, num_q_heads, head_dim = q.shape
    _, num_kv_heads, _ = k.shape
    _, block_size, _, _ = k_cache.shape
    num_seqs = block_tables.shape[0]

    assert seq_lens_k.shape == (num_seqs,)
    assert q_total_tokens == num_seqs * block_len

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim) * LOG2E
        
    o = torch.empty_like(q)
    
    _BLOCK_M = min(triton.next_power_of_2(block_len), 128)
    BLOCK_M = max(16, _BLOCK_M)
    _BLOCK_N = 128 if block_size > 128 else 64
    BLOCK_N = max(16, _BLOCK_N)

    k_cache, v_cache = k_cache.contiguous(), v_cache.contiguous()
    block_tables = block_tables.contiguous()
    seq_lens_k = seq_lens_k.to(torch.int32).contiguous()

    grid = (num_seqs, num_q_heads)
    
    fused_kv_cache_attention_kernel[grid](
        q, q.stride(0), q.stride(1), q.stride(2),
        k, k.stride(0), k.stride(1), k.stride(2),
        v, v.stride(0), v.stride(1), v.stride(2),
        k_cache, k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache, v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        block_tables, block_tables.stride(0), block_tables.stride(1),
        seq_lens_k,
        o, o.stride(0), o.stride(1), o.stride(2),
        BLOCK_LEN=block_len, 
        BLOCK_SIZE=block_size, 
        KV_GROUP_SIZE=num_q_heads//num_kv_heads,
        SM_SCALE=float(sm_scale),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=head_dim,
        num_warps=8
    )
    return o

def test_fused_kv_cache_attention():
    torch.manual_seed(42)
    num_seqs = 4
    block_len = 4
    block_size = 256
    num_q_heads = 16
    num_kv_heads = 8
    head_dim = 128
    num_blocks = 128
    max_blocks_per_seq = 32

    device = 'cuda'
    dtype = torch.float16

    total_tokens = num_seqs * block_len
    q = torch.randn(total_tokens, num_q_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(total_tokens, num_kv_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(total_tokens, num_kv_heads, head_dim, device=device, dtype=dtype)

    k_cache = torch.randn(num_blocks, num_kv_heads, head_dim, block_size, device=device, dtype=dtype).permute(0, 3, 1, 2).contiguous()
    v_cache = torch.randn(num_blocks, num_kv_heads, head_dim, block_size, device=device, dtype=dtype).permute(0, 3, 1, 2).contiguous()

    block_tables = torch.randint(0, num_blocks, (num_seqs, max_blocks_per_seq), device=device, dtype=torch.int32)

    cached_lens = torch.tensor([48, 32, 64, 16], device=device)
    cu_seqlens_k = torch.cumsum(torch.cat([torch.tensor([0], device=device), cached_lens]), dim=0).to(torch.int32)
    cu_seqlens_q = torch.arange(0, (num_seqs + 1) * block_len, block_len, device=device, dtype=torch.int32)

    def reference_implementation():
        outputs = []
        scale = 1.0 / math.sqrt(head_dim)
        for seq_idx in range(num_seqs):
            q_start, q_end = cu_seqlens_q[seq_idx].item(), cu_seqlens_q[seq_idx+1].item()
            cache_len = cached_lens[seq_idx].item()
            q_seq = q[q_start:q_end]
            k_seq = k[q_start:q_end]
            v_seq = v[q_start:q_end]
            k_cached_full = []
            v_cached_full = []
            if cache_len > 0:
                num_cache_blocks = (cache_len + block_size - 1) // block_size
                rem_len = cache_len
                for i in range(num_cache_blocks):
                    physical_block_id = block_tables[seq_idx, i].item()
                    len_to_get = min(rem_len, block_size)
                    k_cached_full.append(k_cache[physical_block_id, :len_to_get])
                    v_cached_full.append(v_cache[physical_block_id, :len_to_get])
                    rem_len -= len_to_get
            if k_cached_full:
                k_full = torch.cat(k_cached_full + [k_seq], dim=0)
                v_full = torch.cat(v_cached_full + [v_seq], dim=0)
            else:
                k_full = k_seq
                v_full = v_seq
            q_in = q_seq.unsqueeze(0).transpose(1, 2)
            k_in = k_full.unsqueeze(0).transpose(1, 2)
            v_in = v_full.unsqueeze(0).transpose(1, 2)
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q_in, k_in, v_in, scale=scale, enable_gqa=True, is_causal=False
            )
            outputs.append(attn_output.transpose(1, 2).squeeze(0))
        return torch.cat(outputs, dim=0)

    ref_output = reference_implementation()
    fused_output = fused_kv_cache_attention(
        q, k, v, k_cache, v_cache, block_tables, cached_lens, block_len
    )
    
    print(f"Max diff: {(ref_output - fused_output).abs().max().item()}")
    print(f"Mean diff: {(ref_output - fused_output).abs().mean().item()}")
    
    rtol = 1e-2
    atol = 1e-2
    assert torch.allclose(ref_output, fused_output, rtol=rtol, atol=atol)

    print('v1 test passed')

if __name__ == "__main__":
    test_fused_kv_cache_attention()