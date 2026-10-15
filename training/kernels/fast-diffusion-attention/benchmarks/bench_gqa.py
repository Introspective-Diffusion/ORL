"""Benchmark GQA support: diff_attn_varlen_with_noisy_input vs flash_attn_varlen_func.

Tests both MHA (n_heads == n_kv_heads) and GQA (n_heads > n_kv_heads) configs.
flash_attn uses the SAME per-document cu_seqlens for fair comparison.
"""
import torch
import gc
from fast_diffusion_attention import diff_attn_varlen_with_noisy_input
from flash_attn import flash_attn_varlen_func


def bench(fn, warmup=30, repeat=80):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(repeat)]
    for s, e in events:
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    times = sorted([s.elapsed_time(e) for s, e in events])
    return times[len(times) // 2]


def main():
    device = "cuda"

    configs = [
        # (seqlen, staircase_size, n_heads, n_kv_heads, head_dim, description)
        # MHA baseline
        (1024, 4, 16, 16, 64, "MHA d=64"),
        (2048, 4, 16, 16, 64, "MHA d=64"),
        (4096, 4, 16, 16, 64, "MHA d=64"),
        (8192, 4, 16, 16, 64, "MHA d=64"),
        # GQA with user's model config
        (1024, 4, 16, 8, 128, "GQA 2x d=128"),
        (2048, 4, 16, 8, 128, "GQA 2x d=128"),
        (4096, 4, 16, 8, 128, "GQA 2x d=128"),
        (8192, 4, 16, 8, 128, "GQA 2x d=128"),
        # GQA 4x
        (1024, 4, 16, 4, 128, "GQA 4x d=128"),
        (2048, 4, 16, 4, 128, "GQA 4x d=128"),
        (4096, 4, 16, 4, 128, "GQA 4x d=128"),
        (8192, 4, 16, 4, 128, "GQA 4x d=128"),
        # Larger staircase sizes
        (4096, 16, 16, 8, 128, "GQA 2x d=128 S=16"),
        (8192, 16, 16, 8, 128, "GQA 2x d=128 S=16"),
    ]

    for mode in ["fwd", "fwd+bwd"]:
        print(f"\n{'=' * 120}")
        print(f"{mode.upper()} BENCHMARK: diff_attn vs flash_attn | GPU: {torch.cuda.get_device_name()}")
        print(f"{'=' * 120}")
        hdr = f"{'desc':>18} {'seq':>6} {'S':>3} {'Hq':>3} {'Hkv':>3} {'D':>4} | {'ours(ms)':>10} {'flash(ms)':>10} {'ratio':>8}"
        print(hdr)
        print("-" * len(hdr))

        for seqlen, S, n_heads, n_kv_heads, head_dim, desc in configs:
            L1 = (seqlen // 3 // S) * S
            if L1 == 0:
                L1 = S
            L0 = seqlen - 2 * L1
            L0 = (L0 // S) * S
            actual = L0 + 2 * L1
            if actual == 0 or L0 == 0:
                continue

            q = torch.randn(actual, n_heads, head_dim, device=device, dtype=torch.float16)
            k = torch.randn(actual, n_kv_heads, head_dim, device=device, dtype=torch.float16)
            v = torch.randn(actual, n_kv_heads, head_dim, device=device, dtype=torch.float16)
            cu = torch.tensor([0, actual], dtype=torch.int32, device=device)
            l0a = torch.tensor([L0], dtype=torch.int32, device=device)
            l1a = torch.tensor([L1], dtype=torch.int32, device=device)

            if mode == "fwd":
                def run_ours(q=q, k=k, v=v, cu=cu, a=l0a, b=l1a, s=S, m=actual):
                    return diff_attn_varlen_with_noisy_input(q.detach(), k.detach(), v.detach(), cu, a, b, s, max_seqlen=m)

                def run_flash(q=q, k=k, v=v, cu=cu, t=actual):
                    return flash_attn_varlen_func(q.detach(), k.detach(), v.detach(), cu, cu, t, t, causal=True)
            else:
                def run_ours(q=q, k=k, v=v, cu=cu, a=l0a, b=l1a, s=S, m=actual):
                    qq = q.detach().requires_grad_(True)
                    kk = k.detach().requires_grad_(True)
                    vv = v.detach().requires_grad_(True)
                    diff_attn_varlen_with_noisy_input(qq, kk, vv, cu, a, b, s, max_seqlen=m).sum().backward()

                def run_flash(q=q, k=k, v=v, cu=cu, t=actual):
                    qq = q.detach().requires_grad_(True)
                    kk = k.detach().requires_grad_(True)
                    vv = v.detach().requires_grad_(True)
                    flash_attn_varlen_func(qq, kk, vv, cu, cu, t, t, causal=True).sum().backward()

            ours_ms = bench(run_ours)
            flash_ms = bench(run_flash)
            ratio = ours_ms / flash_ms

            print(f"{desc:>18} {actual:>6} {S:>3} {n_heads:>3} {n_kv_heads:>3} {head_dim:>4} | "
                  f"{ours_ms:>10.3f} {flash_ms:>10.3f} {ratio:>8.2f}x")

            del q, k, v
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
