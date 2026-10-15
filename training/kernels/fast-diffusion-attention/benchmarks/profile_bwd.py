"""Profile backward pass breakdown: D_i, dQ, dKdV kernel, dKdV reduce, buffer alloc."""
import torch
import math
import gc
from fast_diffusion_attention.diff_attn_noisy import (
    _get_kernel_config, _diff_noisy_attn_fwd_kernel,
    _diff_noisy_attn_bwd_dq_kernel, _diff_noisy_attn_bwd_dkdv_kernel,
    _compute_delta_kernel,
)


def bench_fn(fn, warmup=20, repeat=60):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(repeat)]
    for s, e in events:
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    times = sorted([s.elapsed_time(e) for s, e in events])
    return times[len(times) // 2]


def profile_backward(doc_seqlen, num_docs, S, n_heads, n_kv_heads, head_dim):
    device = "cuda"
    n_kv_groups = n_heads // n_kv_heads
    log2s = int(math.log2(S))

    L1 = (doc_seqlen // 3 // S) * S
    if L1 == 0:
        L1 = S
    L0 = doc_seqlen - 2 * L1
    L0 = (L0 // S) * S
    actual = L0 + 2 * L1
    total = actual * num_docs

    q = torch.randn(total, n_heads, head_dim, device=device, dtype=torch.float16).contiguous()
    k = torch.randn(total, n_kv_heads, head_dim, device=device, dtype=torch.float16).contiguous()
    v = torch.randn(total, n_kv_heads, head_dim, device=device, dtype=torch.float16).contiguous()
    o = torch.randn_like(q)
    do = torch.randn_like(q)

    cu = torch.tensor([actual * i for i in range(num_docs + 1)], dtype=torch.int32, device=device)
    l0a = torch.full((num_docs,), L0, dtype=torch.int32, device=device)
    l1a = torch.full((num_docs,), L1, dtype=torch.int32, device=device)

    scale = 1.0 / math.sqrt(head_dim)
    sm_scale = scale * 1.44269504

    BLOCK_D = head_dim
    batch_size = num_docs

    # Run forward to get lse
    BM_f, BN_f, nw_f, ns_f = _get_kernel_config(head_dim, "fwd")
    lse = torch.empty(total, n_heads, dtype=torch.float32, device=device)
    num_m_tiles = (actual + BM_f - 1) // BM_f
    _diff_noisy_attn_fwd_kernel[(num_m_tiles, n_heads, batch_size)](
        q, k, v, o, lse, cu, l0a, l1a,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        n_heads=n_heads, sm_scale=sm_scale,
        BLOCK_M=BM_f, BLOCK_N=BN_f, BLOCK_D=BLOCK_D, LOG2S=log2s,
        N_KV_GROUPS=n_kv_groups, num_warps=nw_f, num_stages=ns_f,
    )
    torch.cuda.synchronize()

    BM_dq, BN_dq, nw_dq, ns_dq = _get_kernel_config(BLOCK_D, "dq")
    BM_dk, BN_dk, nw_dk, ns_dk = _get_kernel_config(BLOCK_D, "dkdv")

    scale = 1.0 / math.sqrt(head_dim)

    # 1. D_i computation (fused kernel)
    def do_Di():
        D_out = torch.empty(total, n_heads, dtype=torch.float32, device=device)
        _compute_delta_kernel[(total * n_heads,)](
            o, do, D_out, BLOCK_D=head_dim,
            num_warps=max(1, head_dim // 32),
        )
        return D_out
    t_Di = bench_fn(do_Di)
    D = do_Di()

    # 2. dQ kernel
    dq = torch.empty_like(q)
    num_m = (actual + BM_dq - 1) // BM_dq
    def do_dq():
        _diff_noisy_attn_bwd_dq_kernel[(num_m, n_heads, batch_size)](
            q, k, v, o, do, dq, lse, D, cu, l0a, l1a,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            n_heads=n_heads, sm_scale=sm_scale, scale=scale,
            BLOCK_M=BM_dq, BLOCK_N=BN_dq, BLOCK_D=BLOCK_D, LOG2S=log2s,
            N_KV_GROUPS=n_kv_groups, num_warps=nw_dq, num_stages=ns_dq,
        )
    t_dq = bench_fn(do_dq)

    # 3. Buffer allocation
    def do_alloc():
        a = torch.empty(total, n_heads, BLOCK_D, dtype=k.dtype, device=device)
        b = torch.empty(total, n_heads, BLOCK_D, dtype=v.dtype, device=device)
        return a, b
    t_alloc = bench_fn(do_alloc)

    # 4. dKdV kernel
    dk_buf = torch.empty(total, n_heads, BLOCK_D, dtype=k.dtype, device=device)
    dv_buf = torch.empty(total, n_heads, BLOCK_D, dtype=v.dtype, device=device)
    num_n = (actual + BN_dk - 1) // BN_dk
    def do_dkdv():
        dk_buf.zero_()
        dv_buf.zero_()
        _diff_noisy_attn_bwd_dkdv_kernel[(num_n, n_heads, batch_size)](
            q, k, v, o, do, dk_buf, dv_buf, lse, D, cu, l0a, l1a,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            dk_buf.stride(0), dk_buf.stride(1), dk_buf.stride(2),
            n_heads=n_heads, sm_scale=sm_scale, scale=scale,
            BLOCK_M=BM_dk, BLOCK_N=BN_dk, BLOCK_D=BLOCK_D, LOG2S=log2s,
            N_KV_GROUPS=n_kv_groups, num_warps=nw_dk, num_stages=ns_dk,
        )
    t_dkdv = bench_fn(do_dkdv)

    # 5. Reduce
    def do_reduce():
        if n_kv_groups > 1:
            return (dk_buf.view(total, n_kv_heads, n_kv_groups, BLOCK_D).sum(dim=2),
                    dv_buf.view(total, n_kv_heads, n_kv_groups, BLOCK_D).sum(dim=2))
        return dk_buf, dv_buf
    t_reduce = bench_fn(do_reduce)

    t_total = t_Di + t_dq + t_alloc + t_dkdv + t_reduce
    print(f"  D_i:      {t_Di:7.3f} ms ({t_Di/t_total*100:5.1f}%)")
    print(f"  dQ:       {t_dq:7.3f} ms ({t_dq/t_total*100:5.1f}%)")
    print(f"  alloc:    {t_alloc:7.3f} ms ({t_alloc/t_total*100:5.1f}%)")
    print(f"  dKdV:     {t_dkdv:7.3f} ms ({t_dkdv/t_total*100:5.1f}%)")
    print(f"  reduce:   {t_reduce:7.3f} ms ({t_reduce/t_total*100:5.1f}%)")
    print(f"  TOTAL:    {t_total:7.3f} ms")

    del q, k, v, o, do, dq, dk_buf, dv_buf, lse, D
    gc.collect()
    torch.cuda.empty_cache()


def main():
    configs = [
        # (doc_seqlen, num_docs, S, n_heads, n_kv_heads, head_dim)
        (1024, 32, 4, 16, 16, 64,  "MHA d=64 32×1024=32k"),
        (2048, 16, 4, 16, 16, 64,  "MHA d=64 16×2048=32k"),
        (4096, 8,  4, 16, 16, 64,  "MHA d=64 8×4096=32k"),
        (8192, 4,  4, 16, 16, 64,  "MHA d=64 4×8192=32k"),
        (1024, 32, 4, 16, 8, 128,  "GQA 2x d=128 32×1024=32k"),
        (2048, 16, 4, 16, 8, 128,  "GQA 2x d=128 16×2048=32k"),
        (4096, 8,  4, 16, 8, 128,  "GQA 2x d=128 8×4096=32k"),
        (8192, 4,  4, 16, 8, 128,  "GQA 2x d=128 4×8192=32k"),
    ]

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"{'=' * 60}")

    for doc_seq, ndocs, S, nh, nkv, hd, desc in configs:
        print(f"\n{desc}")
        print("-" * 50)
        profile_backward(doc_seq, ndocs, S, nh, nkv, hd)


if __name__ == "__main__":
    main()
