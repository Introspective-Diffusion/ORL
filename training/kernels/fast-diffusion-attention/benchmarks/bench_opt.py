"""Benchmark optimized backward vs flash_attn."""
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
    return times[len(times)//2]


def main():
    device = "cuda"
    n_heads = 16
    head_dim = 64

    configs = [
        (1024, 4), (1024, 16),
        (2048, 4), (2048, 16),
        (4096, 4), (4096, 16),
        (8192, 4), (8192, 16),
    ]

    hdr = f"{'seq':>8} {'S':>4} | {'FWD_v1':>8} {'FWD_fl':>8} {'ratio':>6} | {'FB_v1':>8} {'FB_fl':>8} {'ratio':>6} | {'BWD_v1':>8} {'BWD_fl':>8} {'ratio':>6}"
    print(hdr)
    print("-" * len(hdr))

    for seqlen, S in configs:
        L1 = (seqlen // 3 // S) * S
        if L1 == 0:
            L1 = S
        L0 = seqlen - 2 * L1
        L0 = (L0 // S) * S
        actual = L0 + 2 * L1
        if actual == 0 or L0 == 0:
            continue

        q = torch.randn(actual, n_heads, head_dim, device=device, dtype=torch.float16)
        k = torch.randn(actual, n_heads, head_dim, device=device, dtype=torch.float16)
        v = torch.randn(actual, n_heads, head_dim, device=device, dtype=torch.float16)
        cu = torch.tensor([0, actual], dtype=torch.int32, device=device)
        l0a = torch.tensor([L0], dtype=torch.int32, device=device)
        l1a = torch.tensor([L1], dtype=torch.int32, device=device)

        def run_v1_fwd(q=q, k=k, v=v, cu=cu, a=l0a, b=l1a, s=S, m=actual):
            return diff_attn_varlen_with_noisy_input(q.detach(), k.detach(), v.detach(), cu, a, b, s, max_seqlen=m)

        def run_flash_fwd(q=q, k=k, v=v, cu=cu, t=actual):
            return flash_attn_varlen_func(q.detach(), k.detach(), v.detach(), cu, cu, t, t, causal=True)

        def run_v1_fb(q=q, k=k, v=v, cu=cu, a=l0a, b=l1a, s=S, m=actual):
            qq = q.detach().requires_grad_(True)
            kk = k.detach().requires_grad_(True)
            vv = v.detach().requires_grad_(True)
            diff_attn_varlen_with_noisy_input(qq, kk, vv, cu, a, b, s, max_seqlen=m).sum().backward()

        def run_flash_fb(q=q, k=k, v=v, cu=cu, t=actual):
            qq = q.detach().requires_grad_(True)
            kk = k.detach().requires_grad_(True)
            vv = v.detach().requires_grad_(True)
            flash_attn_varlen_func(qq, kk, vv, cu, cu, t, t, causal=True).sum().backward()

        fwd_v1 = bench(run_v1_fwd)
        fwd_fl = bench(run_flash_fwd)
        fb_v1 = bench(run_v1_fb)
        fb_fl = bench(run_flash_fb)
        bwd_v1 = fb_v1 - fwd_v1
        bwd_fl = fb_fl - fwd_fl

        print(f"{actual:>8} {S:>4} | {fwd_v1:>8.3f} {fwd_fl:>8.3f} {fwd_v1/fwd_fl:>6.2f}x | "
              f"{fb_v1:>8.3f} {fb_fl:>8.3f} {fb_v1/fb_fl:>6.2f}x | "
              f"{bwd_v1:>8.3f} {bwd_fl:>8.3f} {bwd_v1/bwd_fl:>6.2f}x")

        del q, k, v
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
