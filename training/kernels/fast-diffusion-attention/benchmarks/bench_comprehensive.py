"""Comprehensive benchmark with diverse configurations.

Fair comparison: flash_attn uses the SAME per-document cu_seqlens as our
implementation, so both skip cross-document attention equally.

Tests with:
- Various sequence lengths (including non-power-of-2)
- Different staircase sizes
- Different document counts
"""
import torch
import gc
import json
from fast_diffusion_attention import diff_attn_varlen_with_noisy_input
from flash_attn import flash_attn_varlen_func


def bench(fn, warmup=25, repeat=60):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(repeat)]
    for s, e in events:
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    times = sorted([s.elapsed_time(e) for s, e in events])
    return times[len(times) // 2]


def get_configs():
    configs = []

    # Single-doc configs with various seqlens and staircase sizes
    for total_seqlen in [256, 512, 768, 1024, 1536, 2048, 3072, 4096, 5120, 6144, 7168, 8192]:
        for staircase_size in [1, 2, 4, 8, 16, 32, 64]:
            L1 = (total_seqlen // 3 // staircase_size) * staircase_size
            if L1 == 0:
                L1 = staircase_size
            L0 = total_seqlen - 2 * L1
            L0 = (L0 // staircase_size) * staircase_size
            actual = L0 + 2 * L1
            if actual == 0 or L0 == 0 or L1 == 0:
                continue
            configs.append({
                "total": actual, "S": staircase_size,
                "L0": L0, "L1": L1, "n_docs": 1,
            })

    # Multi-doc configs
    for n_docs in [2, 3, 4, 8]:
        for staircase_size in [1, 4, 16, 64]:
            for per_doc_seqlen in [256, 512, 1024, 2048]:
                L1 = (per_doc_seqlen // 3 // staircase_size) * staircase_size
                if L1 == 0:
                    L1 = staircase_size
                L0 = per_doc_seqlen - 2 * L1
                L0 = (L0 // staircase_size) * staircase_size
                actual_per = L0 + 2 * L1
                if actual_per == 0 or L0 == 0 or L1 == 0:
                    continue
                configs.append({
                    "total": actual_per * n_docs, "S": staircase_size,
                    "L0": L0, "L1": L1, "n_docs": n_docs,
                })

    # Realistic non-uniform multi-doc configs
    for staircase_size in [4, 16]:
        L0_list = [128, 256, 64]
        L1_list = [96, 128, 48]
        L0_list = [(l // staircase_size) * staircase_size for l in L0_list]
        L1_list = [(l // staircase_size) * staircase_size for l in L1_list]
        if all(l > 0 for l in L0_list) and all(l > 0 for l in L1_list):
            total = sum(l0 + 2 * l1 for l0, l1 in zip(L0_list, L1_list))
            configs.append({
                "total": total, "S": staircase_size,
                "L0": L0_list, "L1": L1_list, "n_docs": len(L0_list),
                "non_uniform": True,
            })

    return configs


def run_benchmark(mode="fwd"):
    device = "cuda"
    n_heads = 16
    head_dim = 64
    title = "FORWARD" if mode == "fwd" else "FWD+BWD"

    configs = get_configs()
    results = []

    print(f"\n{'=' * 130}")
    print(f"{title} BENCHMARK: diff_attn_varlen_with_noisy_input vs flash_attn_varlen_func (per-doc causal)")
    print("NOTE: flash_attn uses SAME per-document cu_seqlens (fair comparison)")
    print(f"n_heads={n_heads}, head_dim={head_dim}, GPU: {torch.cuda.get_device_name()}")
    print(f"{'=' * 130}")
    header = (f"{'total':>7} {'S':>3} {'L0':>6} {'L1':>6} {'docs':>4} {'doc_len':>7} | "
              f"{'ours(ms)':>10} {'flash(ms)':>10} {'ratio':>8}")
    print(header)
    print("-" * 130)

    for cfg in configs:
        n_docs = cfg["n_docs"]
        S = cfg["S"]
        total = cfg["total"]

        is_non_uniform = cfg.get("non_uniform", False)

        if is_non_uniform:
            L0_list = cfg["L0"]
            L1_list = cfg["L1"]
            doc_lens = [l0 + 2 * l1 for l0, l1 in zip(L0_list, L1_list)]
            cu_list = [0]
            for dl in doc_lens:
                cu_list.append(cu_list[-1] + dl)
            cu = torch.tensor(cu_list, dtype=torch.int32, device=device)
            l0a = torch.tensor(L0_list, dtype=torch.int32, device=device)
            l1a = torch.tensor(L1_list, dtype=torch.int32, device=device)
            max_doc_len = max(doc_lens)
            L0_display = str(L0_list)
            L1_display = str(L1_list)
            doc_len_display = str(doc_lens)
        else:
            L0 = cfg["L0"]
            L1 = cfg["L1"]
            doc_len = L0 + 2 * L1
            if n_docs == 1:
                cu = torch.tensor([0, total], dtype=torch.int32, device=device)
            else:
                cu_list = [0] + [doc_len * (i + 1) for i in range(n_docs)]
                cu = torch.tensor(cu_list, dtype=torch.int32, device=device)
            l0a = torch.tensor([L0] * n_docs, dtype=torch.int32, device=device)
            l1a = torch.tensor([L1] * n_docs, dtype=torch.int32, device=device)
            max_doc_len = doc_len
            L0_display = str(L0)
            L1_display = str(L1)
            doc_len_display = str(doc_len)

        qd = torch.randn(total, n_heads, head_dim, device=device, dtype=torch.float16)
        kd = torch.randn(total, n_heads, head_dim, device=device, dtype=torch.float16)
        vd = torch.randn(total, n_heads, head_dim, device=device, dtype=torch.float16)

        # FAIR: flash uses the SAME cu_seqlens as ours (per-document boundaries)
        cu_flash = cu
        max_seqlen_flash = max_doc_len

        if mode == "fwd":
            def run_ours(q=qd, k=kd, v=vd, c=cu, a=l0a, b=l1a, s=S, m=total):
                return diff_attn_varlen_with_noisy_input(q, k, v, c, a, b, s, max_seqlen=m)

            def run_flash(q=qd, k=kd, v=vd, c=cu_flash, t=max_seqlen_flash):
                return flash_attn_varlen_func(q, k, v, c, c, t, t, causal=True)
        else:
            def run_ours(q=qd, k=kd, v=vd, c=cu, a=l0a, b=l1a, s=S, m=total):
                qq = q.detach().requires_grad_(True)
                kk = k.detach().requires_grad_(True)
                vv = v.detach().requires_grad_(True)
                diff_attn_varlen_with_noisy_input(qq, kk, vv, c, a, b, s, max_seqlen=m).sum().backward()

            def run_flash(q=qd, k=kd, v=vd, c=cu_flash, t=max_seqlen_flash):
                qq = q.detach().requires_grad_(True)
                kk = k.detach().requires_grad_(True)
                vv = v.detach().requires_grad_(True)
                flash_attn_varlen_func(qq, kk, vv, c, c, t, t, causal=True).sum().backward()

        ours_ms = bench(run_ours)
        flash_ms = bench(run_flash)
        ratio = ours_ms / flash_ms

        print(f"{total:>7} {S:>3} {L0_display:>6} {L1_display:>6} {n_docs:>4} {doc_len_display:>7} | "
              f"{ours_ms:>10.3f} {flash_ms:>10.3f} {ratio:>8.2f}x")

        results.append({
            "mode": mode,
            "total_tokens": total,
            "staircase_size": S,
            "L0": cfg["L0"],
            "L1": cfg["L1"],
            "n_docs": n_docs,
            "max_doc_len": max_doc_len,
            "ours_ms": round(ours_ms, 4),
            "flash_ms": round(flash_ms, 4),
            "ratio": round(ratio, 3),
        })

        del qd, kd, vd
        gc.collect()
        torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    all_results = []
    all_results.extend(run_benchmark("fwd"))
    all_results.extend(run_benchmark("fwd_bwd"))

    with open("benchmark_results_fair.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to benchmark_results_fair.json ({len(all_results)} configs)")
