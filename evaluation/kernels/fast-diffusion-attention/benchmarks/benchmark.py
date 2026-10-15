"""Benchmark diff_attn_varlen_with_noisy_input vs flash_attn causal."""
import torch
import gc
from fast_diffusion_attention import diff_attn_varlen_with_noisy_input
from flash_attn import flash_attn_varlen_func


def flops_attention(seqlen, n_heads, head_dim, causal=True):
    f = 4 * seqlen * seqlen * head_dim * n_heads
    if causal:
        f //= 2
    return f


def benchmark_fn(fn, warmup=20, repeat=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(repeat)]
    for s, e in events:
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    times = sorted([s.elapsed_time(e) for s, e in events])
    return times[len(times)//2]


def get_configs():
    configs = []
    for total_seqlen in [512, 1024, 2048, 4096, 8192]:
        for staircase_size in [1, 4, 16, 64]:
            L1 = (total_seqlen // 3 // staircase_size) * staircase_size
            if L1 == 0:
                L1 = staircase_size
            L0 = total_seqlen - 2 * L1
            L0 = (L0 // staircase_size) * staircase_size
            actual_total = L0 + 2 * L1
            if actual_total == 0 or L0 == 0:
                continue
            configs.append((actual_total, staircase_size, L0, L1))
    return configs


def run_benchmark(mode="fwd"):
    device = "cuda"
    n_heads = 16
    head_dim = 64
    title = "FORWARD" if mode == "fwd" else "FWD+BWD"

    print(f"\n{'=' * 95}")
    print(f"{title} BENCHMARK: diff_attn_varlen_with_noisy_input vs flash_attn_varlen_func (causal)")
    print(f"n_heads={n_heads}, head_dim={head_dim}")
    print(f"{'=' * 95}")
    print(f"{'seqlen':>8} {'staircase':>10} {'L0':>6} {'L1':>6} | {'ours(ms)':>10} {'flash(ms)':>10} | "
          f"{'ours TFLOPS':>12} {'flash TFLOPS':>12} {'ratio':>8}")
    print("-" * 95)

    for actual_total, staircase_size, L0, L1 in get_configs():
        cu = torch.tensor([0, actual_total], dtype=torch.int32, device=device)
        l0a = torch.tensor([L0], dtype=torch.int32, device=device)
        l1a = torch.tensor([L1], dtype=torch.int32, device=device)
        qd = torch.randn(actual_total, n_heads, head_dim, device=device, dtype=torch.float16)
        kd = torch.randn(actual_total, n_heads, head_dim, device=device, dtype=torch.float16)
        vd = torch.randn(actual_total, n_heads, head_dim, device=device, dtype=torch.float16)

        ss = staircase_size
        at = actual_total

        if mode == "fwd":
            def run_ours(q=qd, k=kd, v=vd, c=cu, a=l0a, b=l1a, s=ss, m=at):
                return diff_attn_varlen_with_noisy_input(q, k, v, c, a, b, s, max_seqlen=m)
            def run_flash(q=qd, k=kd, v=vd, c=cu, t=at):
                return flash_attn_varlen_func(q, k, v, c, c, t, t, causal=True)
            mult = 1.0
        else:
            def run_ours(q=qd, k=kd, v=vd, c=cu, a=l0a, b=l1a, s=ss, m=at):
                qq, kk, vv = q.detach().requires_grad_(True), k.detach().requires_grad_(True), v.detach().requires_grad_(True)
                diff_attn_varlen_with_noisy_input(qq, kk, vv, c, a, b, s, max_seqlen=m).sum().backward()
            def run_flash(q=qd, k=kd, v=vd, c=cu, t=at):
                qq, kk, vv = q.detach().requires_grad_(True), k.detach().requires_grad_(True), v.detach().requires_grad_(True)
                flash_attn_varlen_func(qq, kk, vv, c, c, t, t, causal=True).sum().backward()
            mult = 3.5

        ours_ms = benchmark_fn(run_ours)
        flash_ms = benchmark_fn(run_flash)

        f = flops_attention(actual_total, n_heads, head_dim, causal=True) * mult
        ours_tflops = f / (ours_ms * 1e-3) / 1e12
        flash_tflops = f / (flash_ms * 1e-3) / 1e12
        ratio = ours_ms / flash_ms

        print(f"{actual_total:>8} {ss:>10} {L0:>6} {L1:>6} | {ours_ms:>10.3f} {flash_ms:>10.3f} | "
              f"{ours_tflops:>12.2f} {flash_tflops:>12.2f} {ratio:>8.2f}x")

        del qd, kd, vd
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run_benchmark("fwd")
    run_benchmark("fwd_bwd")
