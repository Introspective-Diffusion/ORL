"""Tune backward kernel block sizes for multi-doc scenarios (30-50k total tokens)."""
import torch
import math
import gc
from fast_diffusion_attention.diff_attn_noisy import (
    _diff_noisy_attn_fwd_kernel, _diff_noisy_attn_bwd_dq_kernel,
    _diff_noisy_attn_bwd_dkdv_kernel, _compute_delta_kernel,
)


def bench_fn(fn, warmup=10, repeat=40):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(repeat)]
    for s, e in events:
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    times = sorted([s.elapsed_time(e) for s, e in events])
    return times[len(times) // 2]


def tune_kernel(kernel_name, doc_seqlen_list, num_docs_list, S, n_heads, n_kv_heads, head_dim, configs):
    device = "cuda"
    n_kv_groups = n_heads // n_kv_heads
    log2s = int(math.log2(S))
    scale = 1.0 / math.sqrt(head_dim)
    sm_scale = scale * 1.44269504
    BLOCK_D = head_dim

    best_config = None
    best_total = float('inf')

    for BM, BN, NW, NS in configs:
        total_time = 0.0
        valid = True

        for doc_seqlen, num_docs in zip(doc_seqlen_list, num_docs_list):
            L1 = (doc_seqlen // 3 // S) * S
            if L1 == 0:
                L1 = S
            L0 = doc_seqlen - 2 * L1
            L0 = (L0 // S) * S
            actual = L0 + 2 * L1
            total = actual * num_docs

            q = torch.randn(total, n_heads, head_dim, device=device, dtype=torch.float16).contiguous()
            k = torch.randn(total, n_kv_heads, head_dim, device=device, dtype=torch.float16).contiguous()
            v = torch.randn_like(k)
            o = torch.randn_like(q)
            do = torch.randn_like(q)
            cu = torch.tensor([actual * i for i in range(num_docs + 1)], dtype=torch.int32, device=device)
            l0a = torch.full((num_docs,), L0, dtype=torch.int32, device=device)
            l1a = torch.full((num_docs,), L1, dtype=torch.int32, device=device)

            # Forward to get lse
            fwd_BM, fwd_BN = 128, 32
            lse = torch.empty(total, n_heads, dtype=torch.float32, device=device)
            _diff_noisy_attn_fwd_kernel[((actual + fwd_BM - 1) // fwd_BM, n_heads, num_docs)](
                q, k, v, o, lse, cu, l0a, l1a,
                q.stride(0), q.stride(1), q.stride(2),
                k.stride(0), k.stride(1), k.stride(2),
                v.stride(0), v.stride(1), v.stride(2),
                o.stride(0), o.stride(1), o.stride(2),
                n_heads=n_heads, sm_scale=sm_scale,
                BLOCK_M=fwd_BM, BLOCK_N=fwd_BN, BLOCK_D=BLOCK_D, LOG2S=log2s,
                N_KV_GROUPS=n_kv_groups, num_warps=4, num_stages=1,
            )
            D = torch.empty(total, n_heads, dtype=torch.float32, device=device)
            _compute_delta_kernel[(total * n_heads,)](o, do, D, BLOCK_D=BLOCK_D, num_warps=max(1, BLOCK_D // 32))
            torch.cuda.synchronize()

            try:
                if kernel_name == "dq":
                    dq = torch.empty_like(q)
                    num_tiles = (actual + BM - 1) // BM
                    def run(dq=dq):
                        _diff_noisy_attn_bwd_dq_kernel[(num_tiles, n_heads, num_docs)](
                            q, k, v, o, do, dq, lse, D, cu, l0a, l1a,
                            q.stride(0), q.stride(1), q.stride(2),
                            k.stride(0), k.stride(1), k.stride(2),
                            v.stride(0), v.stride(1), v.stride(2),
                            o.stride(0), o.stride(1), o.stride(2),
                            n_heads=n_heads, sm_scale=sm_scale, scale=scale,
                            BLOCK_M=BM, BLOCK_N=BN, BLOCK_D=BLOCK_D, LOG2S=log2s,
                            N_KV_GROUPS=n_kv_groups, num_warps=NW, num_stages=NS,
                        )
                else:
                    dk_buf = torch.empty(total, n_heads, BLOCK_D, dtype=k.dtype, device=device)
                    dv_buf = torch.empty(total, n_heads, BLOCK_D, dtype=v.dtype, device=device)
                    num_tiles = (actual + BN - 1) // BN
                    def run(dk_buf=dk_buf, dv_buf=dv_buf):
                        _diff_noisy_attn_bwd_dkdv_kernel[(num_tiles, n_heads, num_docs)](
                            q, k, v, o, do, dk_buf, dv_buf, lse, D, cu, l0a, l1a,
                            q.stride(0), q.stride(1), q.stride(2),
                            k.stride(0), k.stride(1), k.stride(2),
                            v.stride(0), v.stride(1), v.stride(2),
                            o.stride(0), o.stride(1), o.stride(2),
                            dk_buf.stride(0), dk_buf.stride(1), dk_buf.stride(2),
                            n_heads=n_heads, sm_scale=sm_scale, scale=scale,
                            BLOCK_M=BM, BLOCK_N=BN, BLOCK_D=BLOCK_D, LOG2S=log2s,
                            N_KV_GROUPS=n_kv_groups, num_warps=NW, num_stages=NS,
                        )
                t = bench_fn(run)
                total_time += t
            except Exception as e:
                valid = False
                break

            del q, k, v, o, do, lse, D
            gc.collect()
            torch.cuda.empty_cache()

        if valid and total_time < best_total:
            best_total = total_time
            best_config = (BM, BN, NW, NS)
        if valid:
            print(f"  ({BM:>3}, {BN:>3}, W={NW}, S={NS}): {total_time:8.3f} ms {'<-- BEST' if total_time == best_total else ''}")

    return best_config, best_total


def main():
    print(f"GPU: {torch.cuda.get_device_name()}")

    # Multi-doc scenarios to optimize for
    doc_seqlens = [1024, 2048, 4096, 8192]
    num_docs_l = [32, 16, 8, 4]

    dq_configs = [
        (BM, BN, W, S)
        for BM in [32, 64, 128]
        for BN in [32, 64, 128]
        for W in [2, 4, 8]
        for S in [1, 2, 3]
    ]
    dkdv_configs = [
        (BM, BN, W, S)
        for BM in [32, 64, 128]
        for BN in [32, 64, 128]
        for W in [2, 4, 8]
        for S in [1, 2, 3]
    ]

    for head_dim, nh, nkv, desc in [
        (128, 16, 8, "GQA 2x d=128"),
        (128, 32, 8, "GQA 4x d=128"),
        (128, 32, 4, "GQA 8x d=128"),
    ]:
        print(f"\n{'='*70}")
        print(f"{desc}: Tuning dQ kernel")
        print(f"{'='*70}")
        best, t = tune_kernel("dq", doc_seqlens, num_docs_l, 4, nh, nkv, head_dim, dq_configs)
        print(f"BEST dQ: {best} total={t:.3f}ms")

        print(f"\n{'='*70}")
        print(f"{desc}: Tuning dKdV kernel")
        print(f"{'='*70}")
        best, t = tune_kernel("dkdv", doc_seqlens, num_docs_l, 4, nh, nkv, head_dim, dkdv_configs)
        print(f"BEST dKdV: {best} total={t:.3f}ms")


if __name__ == "__main__":
    main()
