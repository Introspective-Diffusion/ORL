"""Benchmark multi-document batches with total tokens 30k-50k.

Each batch packs multiple documents (each L0+2*L1 tokens) into a single
flat tensor, matching realistic training scenarios.
"""
import torch
import gc
from fast_diffusion_attention import diff_attn_varlen_with_noisy_input
from flash_attn import flash_attn_varlen_func


def bench(fn, warmup=20, repeat=60):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(repeat)]
    for s, e in events:
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    times = sorted([s.elapsed_time(e) for s, e in events])
    return times[len(times) // 2]


def make_batch(doc_seqlen, num_docs, staircase_size, n_heads, n_kv_heads, head_dim, device):
    """Create a multi-doc batch where each doc has doc_seqlen tokens."""
    S = staircase_size
    L1 = (doc_seqlen // 3 // S) * S
    if L1 == 0:
        L1 = S
    L0 = doc_seqlen - 2 * L1
    L0 = (L0 // S) * S
    actual_per_doc = L0 + 2 * L1
    if actual_per_doc == 0 or L0 == 0:
        return None
    total = actual_per_doc * num_docs

    q = torch.randn(total, n_heads, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(total, n_kv_heads, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(total, n_kv_heads, head_dim, device=device, dtype=torch.float16)

    cu = torch.tensor([actual_per_doc * i for i in range(num_docs + 1)], dtype=torch.int32, device=device)
    l0a = torch.full((num_docs,), L0, dtype=torch.int32, device=device)
    l1a = torch.full((num_docs,), L1, dtype=torch.int32, device=device)

    return q, k, v, cu, l0a, l1a, actual_per_doc, total, L0, L1


def main():
    device = "cuda"

    # (doc_seqlen, num_docs, staircase_size, n_heads, n_kv_heads, head_dim, description)
    configs = [
        # MHA d=64: ~32k total
        (1024, 32, 4, 16, 16, 64, "MHA d=64"),
        (2048, 16, 4, 16, 16, 64, "MHA d=64"),
        (4096, 8,  4, 16, 16, 64, "MHA d=64"),
        (8192, 4,  4, 16, 16, 64, "MHA d=64"),
        # GQA 2x d=128: ~32k total
        (1024, 32, 4, 16, 8, 128, "GQA 2x d=128"),
        (2048, 16, 4, 16, 8, 128, "GQA 2x d=128"),
        (4096, 8,  4, 16, 8, 128, "GQA 2x d=128"),
        (8192, 4,  4, 16, 8, 128, "GQA 2x d=128"),
        # Mixed doc lengths: ~40k total
        # We simulate this by varying doc count at a given seqlen
        (1024, 40, 4, 16, 8, 128, "GQA 2x ~40k"),
        (2048, 20, 4, 16, 8, 128, "GQA 2x ~40k"),
        (4096, 10, 4, 16, 8, 128, "GQA 2x ~40k"),
        (8192, 5,  4, 16, 8, 128, "GQA 2x ~40k"),
    ]

    for mode in ["fwd", "fwd+bwd"]:
        print(f"\n{'=' * 130}")
        print(f"{mode.upper()} BENCHMARK (multi-doc, long total seq) | GPU: {torch.cuda.get_device_name()}")
        print(f"{'=' * 130}")
        hdr = (f"{'desc':>16} {'doc_len':>8} {'#docs':>6} {'total':>7} {'S':>3} "
               f"{'Hq':>3} {'Hkv':>3} {'D':>4} | {'ours(ms)':>10} {'flash(ms)':>10} {'ratio':>8}")
        print(hdr)
        print("-" * len(hdr))

        for doc_seqlen, num_docs, S, n_heads, n_kv_heads, head_dim, desc in configs:
            batch = make_batch(doc_seqlen, num_docs, S, n_heads, n_kv_heads, head_dim, device)
            if batch is None:
                continue
            q, k, v, cu, l0a, l1a, actual_per_doc, total, L0, L1 = batch

            max_sl = actual_per_doc

            if mode == "fwd":
                def run_ours(q=q, k=k, v=v, cu=cu, a=l0a, b=l1a, s=S, m=max_sl):
                    return diff_attn_varlen_with_noisy_input(q.detach(), k.detach(), v.detach(), cu, a, b, s, max_seqlen=m)

                def run_flash(q=q, k=k, v=v, cu=cu, t=max_sl):
                    return flash_attn_varlen_func(q.detach(), k.detach(), v.detach(), cu, cu, t, t, causal=True)
            else:
                def run_ours(q=q, k=k, v=v, cu=cu, a=l0a, b=l1a, s=S, m=max_sl):
                    qq = q.detach().requires_grad_(True)
                    kk = k.detach().requires_grad_(True)
                    vv = v.detach().requires_grad_(True)
                    diff_attn_varlen_with_noisy_input(qq, kk, vv, cu, a, b, s, max_seqlen=m).sum().backward()

                def run_flash(q=q, k=k, v=v, cu=cu, t=max_sl):
                    qq = q.detach().requires_grad_(True)
                    kk = k.detach().requires_grad_(True)
                    vv = v.detach().requires_grad_(True)
                    flash_attn_varlen_func(qq, kk, vv, cu, cu, t, t, causal=True).sum().backward()

            ours_ms = bench(run_ours)
            flash_ms = bench(run_flash)
            ratio = ours_ms / flash_ms

            print(f"{desc:>16} {actual_per_doc:>8} {num_docs:>6} {total:>7} {S:>3} "
                  f"{n_heads:>3} {n_kv_heads:>3} {head_dim:>4} | "
                  f"{ours_ms:>10.3f} {flash_ms:>10.3f} {ratio:>8.2f}x")

            del q, k, v, cu, l0a, l1a
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
