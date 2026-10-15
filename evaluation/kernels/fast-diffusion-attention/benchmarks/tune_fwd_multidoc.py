"""Tune forward kernel block sizes for multi-doc scenarios (30-50k total tokens)."""
import torch
import math
import gc
from fast_diffusion_attention.diff_attn_noisy import (
    _diff_noisy_attn_fwd_kernel,
)


def bench_fn(fn, warmup=5, repeat=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(repeat)]
    for s, e in events:
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    times = sorted([s.elapsed_time(e) for s, e in events])
    return times[len(times) // 2]


def tune_kernel(doc_seqlen_list, num_docs_list, S, n_heads, n_kv_heads, head_dim, configs):
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
            o = torch.empty_like(q)
            lse = torch.empty(total, n_heads, dtype=torch.float32, device=device)
            cu = torch.tensor([actual * i for i in range(num_docs + 1)], dtype=torch.int32, device=device)
            l0a = torch.full((num_docs,), L0, dtype=torch.int32, device=device)
            l1a = torch.full((num_docs,), L1, dtype=torch.int32, device=device)

            try:
                num_tiles = (actual + BM - 1) // BM

                def run():
                    _diff_noisy_attn_fwd_kernel[(num_tiles, n_heads, num_docs)](
                        q, k, v, o, lse, cu, l0a, l1a,
                        q.stride(0), q.stride(1), q.stride(2),
                        k.stride(0), k.stride(1), k.stride(2),
                        v.stride(0), v.stride(1), v.stride(2),
                        o.stride(0), o.stride(1), o.stride(2),
                        n_heads=n_heads, sm_scale=sm_scale,
                        BLOCK_M=BM, BLOCK_N=BN, BLOCK_D=BLOCK_D, LOG2S=log2s,
                        N_KV_GROUPS=n_kv_groups, num_warps=NW, num_stages=NS,
                    )

                t = bench_fn(run)
                total_time += t
            except Exception as e:
                valid = False
                break

            del q, k, v, o, lse
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

    doc_seqlens = [1024, 2048, 4096, 8192]
    num_docs_l = [32, 16, 8, 4]

    fwd_configs = [
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
        print(f"{desc}: Tuning forward kernel")
        print(f"{'='*70}")
        best, t = tune_kernel(doc_seqlens, num_docs_l, 4, nh, nkv, head_dim, fwd_configs)
        print(f"BEST fwd: {best} total={t:.3f}ms")


if __name__ == "__main__":
    main()
