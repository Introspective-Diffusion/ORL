import torch
from torch import nn
import triton
import triton.language as tl
import torch.nn.functional as F

from jetengine_ext.utils.context import get_context
from jetengine_ext.engine.sequence import RunType
from jetengine_ext.kernels.triton.attention import sparse_attn_varlen
# from jetengine.kernels.triton.attention import fused_kv_cache_attention
# from jetengine.kernels.triton.attention import fused_kv_cache_attention_v5
try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
except (ImportError, AttributeError):
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None


def _repeat_kv_for_query_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    if x.shape[-2] == num_heads:
        return x
    repeat = num_heads // x.shape[-2]
    return x.repeat_interleave(repeat, dim=-2)


def _torch_flash_attn_varlen_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool = False,
):
    del max_seqlen_q, max_seqlen_k
    outputs = []
    num_heads = q.shape[-2]
    cu_q = cu_seqlens_q.detach().cpu().tolist()
    cu_k = cu_seqlens_k.detach().cpu().tolist()
    for i in range(len(cu_q) - 1):
        q_i = q[cu_q[i] : cu_q[i + 1]]
        k_i = _repeat_kv_for_query_heads(k[cu_k[i] : cu_k[i + 1]], num_heads)
        v_i = _repeat_kv_for_query_heads(v[cu_k[i] : cu_k[i + 1]], num_heads)
        out_i = F.scaled_dot_product_attention(
            q_i.transpose(0, 1).unsqueeze(0),
            k_i.transpose(0, 1).unsqueeze(0),
            v_i.transpose(0, 1).unsqueeze(0),
            is_causal=causal,
        )
        outputs.append(out_i.squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0) if outputs else q.new_empty(q.shape)


def _torch_flash_attn_with_kvcache_fallback(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cache_seqlens: torch.Tensor,
    block_table: torch.Tensor,
    causal: bool = False,
):
    batch, block_len, num_heads, _ = q.shape
    outputs = []
    cache_lens = cache_seqlens.detach().cpu().tolist()
    block_table_cpu = block_table.detach().cpu().tolist() if block_table is not None else [[] for _ in range(batch)]
    for i in range(batch):
        prefix_k, prefix_v = [], []
        remaining = int(cache_lens[i])
        for block_id in block_table_cpu[i]:
            if block_id < 0 or remaining <= 0:
                break
            take = min(int(remaining), int(k_cache.shape[1]))
            prefix_k.append(k_cache[int(block_id), :take])
            prefix_v.append(v_cache[int(block_id), :take])
            remaining -= take
        if prefix_k:
            k_i = torch.cat(prefix_k + [k[i]], dim=0)
            v_i = torch.cat(prefix_v + [v[i]], dim=0)
        else:
            k_i = k[i]
            v_i = v[i]
        k_i = _repeat_kv_for_query_heads(k_i, num_heads)
        v_i = _repeat_kv_for_query_heads(v_i, num_heads)
        q_i = q[i]
        attn_mask = None
        if causal:
            prefix_len = max(k_i.shape[0] - block_len, 0)
            q_pos = torch.arange(block_len, device=q.device).unsqueeze(1)
            k_pos = torch.arange(k_i.shape[0], device=q.device).unsqueeze(0)
            attn_mask = k_pos <= (prefix_len + q_pos)
        out_i = F.scaled_dot_product_attention(
            q_i.transpose(0, 1).unsqueeze(0),
            k_i.transpose(0, 1).unsqueeze(0),
            v_i.transpose(0, 1).unsqueeze(0),
            attn_mask=attn_mask,
            is_causal=False,
        )
        outputs.append(out_i.squeeze(0).transpose(0, 1))
    return torch.stack(outputs, dim=0)


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    slot = tl.load(slot_mapping_ptr + idx)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        pass

class BlockAttention(Attention):
    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__(num_heads, head_dim, scale, num_kv_heads)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        should_store_whole = (context.run_type == RunType.PREFILL)
        if should_store_whole and k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
            
        if context.run_type == RunType.PREFILL:
            if context.prefill_causal:
                attn_varlen = flash_attn_varlen_func or _torch_flash_attn_varlen_fallback
                o = attn_varlen(
                    q,
                    k,
                    v,
                    context.cu_seqlens_q,
                    context.cu_seqlens_k,
                    context.max_seqlen_q,
                    context.max_seqlen_k,
                    causal=True,
                )
            else:
                o = sparse_attn_varlen(q, k, v,
                                    cu_seqlens_q=context.cu_seqlens_q,
                                    cu_seqlens_k=context.cu_seqlens_k,
                                    staircase_size=context.block_length)
        else:
            q = q.view(-1, context.block_length, self.num_heads, self.head_dim)
            k = k.view(-1, context.block_length, self.num_kv_heads, self.head_dim)
            v = v.view(-1, context.block_length, self.num_kv_heads, self.head_dim)
            attn_kvcache = flash_attn_with_kvcache or _torch_flash_attn_with_kvcache_fallback
            o = attn_kvcache(q, k_cache=k_cache, v_cache=v_cache, k=k, v=v,
                             cache_seqlens=context.context_lens,
                             block_table=context.block_tables,
                             causal=bool(context.commit_causal))
        o = o.view(-1, self.num_heads * self.head_dim)
        return o

        
