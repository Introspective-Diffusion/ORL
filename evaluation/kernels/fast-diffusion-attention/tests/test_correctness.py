"""Correctness tests for diff_attn_varlen_with_noisy_input."""
import torch
import math

from fast_diffusion_attention import diff_attn_varlen_with_noisy_input
from fast_diffusion_attention.ref_mask import build_full_mask


def naive_attention_with_mask(q, k, v, mask, scale=None):
    """Reference attention: q,k,v are [total_tokens, n_heads, head_dim], mask is [total_tokens, total_tokens]."""
    total, n_heads, d = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(d)
    qk = torch.einsum("ihd,jhd->hij", q.float(), k.float()) * scale
    mask_expanded = mask.unsqueeze(0).expand(n_heads, -1, -1)
    qk = qk.masked_fill(~mask_expanded, -1e9)
    attn = torch.softmax(qk, dim=-1)
    out = torch.einsum("hij,jhd->ihd", attn, v.float())
    return out.to(q.dtype)


def test_forward_small():
    """Test with the example.txt configuration: staircase_size=2, doc1(L0=4,L1=6), doc2(L0=2,L1=2)."""
    torch.manual_seed(42)
    device = "cuda"
    n_heads = 2
    head_dim = 64
    staircase_size = 2

    L0_list = [4, 2]
    L1_list = [6, 2]
    doc_lens = [L0 + 2 * L1 for L0, L1 in zip(L0_list, L1_list)]
    total_tokens = sum(doc_lens)

    q = torch.randn(total_tokens, n_heads, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(total_tokens, n_heads, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(total_tokens, n_heads, head_dim, device=device, dtype=torch.float16)

    cu_seqlens = torch.tensor([0] + [sum(doc_lens[:i+1]) for i in range(len(doc_lens))],
                               dtype=torch.int32, device=device)
    L0_arr = torch.tensor(L0_list, dtype=torch.int32, device=device)
    L1_arr = torch.tensor(L1_list, dtype=torch.int32, device=device)

    mask = build_full_mask(L0_list, L1_list, staircase_size).to(device)
    ref_out = naive_attention_with_mask(q, k, v, mask)
    tri_out = diff_attn_varlen_with_noisy_input(q, k, v, cu_seqlens, L0_arr, L1_arr, staircase_size)

    max_diff = (ref_out.float() - tri_out.float()).abs().max().item()
    mean_diff = (ref_out.float() - tri_out.float()).abs().mean().item()
    print(f"[Forward small] max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    assert max_diff < 0.01, f"Forward small test FAILED: max_diff={max_diff}"
    print("[Forward small] PASSED")


def test_forward_various_staircase():
    """Test with various staircase sizes."""
    torch.manual_seed(123)
    device = "cuda"
    n_heads = 4
    head_dim = 64

    for staircase_size in [1, 2, 4, 8, 16, 32, 64]:
        L0 = staircase_size * 4
        L1 = staircase_size * 3
        total = L0 + 2 * L1

        q = torch.randn(total, n_heads, head_dim, device=device, dtype=torch.float16)
        k = torch.randn(total, n_heads, head_dim, device=device, dtype=torch.float16)
        v = torch.randn(total, n_heads, head_dim, device=device, dtype=torch.float16)

        cu_seqlens = torch.tensor([0, total], dtype=torch.int32, device=device)
        L0_arr = torch.tensor([L0], dtype=torch.int32, device=device)
        L1_arr = torch.tensor([L1], dtype=torch.int32, device=device)

        mask = build_full_mask([L0], [L1], staircase_size).to(device)
        ref_out = naive_attention_with_mask(q, k, v, mask)
        tri_out = diff_attn_varlen_with_noisy_input(q, k, v, cu_seqlens, L0_arr, L1_arr, staircase_size)

        max_diff = (ref_out.float() - tri_out.float()).abs().max().item()
        print(f"[Forward staircase={staircase_size}] max_diff={max_diff:.6f}")
        assert max_diff < 0.02, f"Forward staircase={staircase_size} FAILED: max_diff={max_diff}"
    print("[Forward various staircase] ALL PASSED")


def test_forward_bfloat16():
    """Test forward pass with bfloat16."""
    torch.manual_seed(42)
    device = "cuda"
    n_heads = 4
    head_dim = 64
    staircase_size = 4
    L0, L1 = 64, 48
    total = L0 + 2 * L1

    q = torch.randn(total, n_heads, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    cu_seqlens = torch.tensor([0, total], dtype=torch.int32, device=device)
    L0_arr = torch.tensor([L0], dtype=torch.int32, device=device)
    L1_arr = torch.tensor([L1], dtype=torch.int32, device=device)

    mask = build_full_mask([L0], [L1], staircase_size).to(device)
    ref_out = naive_attention_with_mask(q, k, v, mask)
    tri_out = diff_attn_varlen_with_noisy_input(q, k, v, cu_seqlens, L0_arr, L1_arr, staircase_size)

    cos = torch.nn.functional.cosine_similarity(
        tri_out.float().reshape(-1), ref_out.float().reshape(-1), dim=0
    ).item()
    print(f"[Forward bfloat16] cos_sim={cos:.6f}")
    assert cos > 0.9999, f"Forward bfloat16 FAILED: cos_sim={cos}"
    print("[Forward bfloat16] PASSED")


def _check_grad(name, g_ref, g_tri, cos_thresh=0.99, mag_lo=0.95, mag_hi=1.05):
    """Check both direction (cosine similarity) and magnitude of gradients."""
    g_ref_flat = g_ref.float().reshape(-1)
    g_tri_flat = g_tri.float().reshape(-1)
    cos_sim = torch.nn.functional.cosine_similarity(g_ref_flat, g_tri_flat, dim=0).item()
    mag_ratio = g_tri_flat.norm().item() / max(g_ref_flat.norm().item(), 1e-12)
    print(f"  {name}: cos_sim={cos_sim:.6f}, mag_ratio={mag_ratio:.4f}")
    assert cos_sim > cos_thresh, f"{name} cos_sim={cos_sim:.6f} < {cos_thresh}"
    assert mag_lo < mag_ratio < mag_hi, f"{name} mag_ratio={mag_ratio:.4f} outside [{mag_lo}, {mag_hi}]"


def test_backward():
    """Test backward pass by comparing gradients with naive implementation."""
    torch.manual_seed(999)
    device = "cuda"
    n_heads = 2
    head_dim = 64
    staircase_size = 4

    L0 = 16
    L1 = 12
    total = L0 + 2 * L1

    q_ref = torch.randn(total, n_heads, head_dim, device=device, dtype=torch.float32, requires_grad=True)
    k_ref = torch.randn(total, n_heads, head_dim, device=device, dtype=torch.float32, requires_grad=True)
    v_ref = torch.randn(total, n_heads, head_dim, device=device, dtype=torch.float32, requires_grad=True)

    mask = build_full_mask([L0], [L1], staircase_size).to(device)
    scale = 1.0 / math.sqrt(head_dim)
    qk = torch.einsum("ihd,jhd->hij", q_ref, k_ref) * scale
    mask_expanded = mask.unsqueeze(0).expand(n_heads, -1, -1)
    qk = qk.masked_fill(~mask_expanded, -1e9)
    attn = torch.softmax(qk, dim=-1)
    ref_out = torch.einsum("hij,jhd->ihd", attn, v_ref)
    ref_out.sum().backward()

    for dtype_name, dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16)]:
        q_tri = q_ref.detach().clone().to(dtype).requires_grad_(True)
        k_tri = k_ref.detach().clone().to(dtype).requires_grad_(True)
        v_tri = v_ref.detach().clone().to(dtype).requires_grad_(True)

        cu_seqlens = torch.tensor([0, total], dtype=torch.int32, device=device)
        L0_arr = torch.tensor([L0], dtype=torch.int32, device=device)
        L1_arr = torch.tensor([L1], dtype=torch.int32, device=device)

        tri_out = diff_attn_varlen_with_noisy_input(q_tri, k_tri, v_tri, cu_seqlens, L0_arr, L1_arr, staircase_size)
        tri_out.float().sum().backward()

        print(f"[Backward {dtype_name}]")
        for name, g_ref_g, g_tri_g in [("dQ", q_ref.grad, q_tri.grad),
                                         ("dK", k_ref.grad, k_tri.grad),
                                         ("dV", v_ref.grad, v_tri.grad)]:
            _check_grad(f"{dtype_name} {name}", g_ref_g, g_tri_g)
    print("[Backward] PASSED")


def test_gqa_forward():
    """Test forward pass with grouped-query attention (n_kv_heads < n_heads)."""
    torch.manual_seed(77)
    device = "cuda"
    head_dim = 64
    staircase_size = 4
    L0, L1 = 16, 12
    total = L0 + 2 * L1

    for n_heads, n_kv_heads in [(4, 2), (8, 2), (8, 4), (16, 8)]:
        n_kv_groups = n_heads // n_kv_heads
        q = torch.randn(total, n_heads, head_dim, device=device, dtype=torch.float16)
        k = torch.randn(total, n_kv_heads, head_dim, device=device, dtype=torch.float16)
        v = torch.randn(total, n_kv_heads, head_dim, device=device, dtype=torch.float16)

        cu_seqlens = torch.tensor([0, total], dtype=torch.int32, device=device)
        L0_arr = torch.tensor([L0], dtype=torch.int32, device=device)
        L1_arr = torch.tensor([L1], dtype=torch.int32, device=device)

        # Reference: expand k/v to match q heads
        k_exp = k.repeat_interleave(n_kv_groups, dim=1)
        v_exp = v.repeat_interleave(n_kv_groups, dim=1)
        mask = build_full_mask([L0], [L1], staircase_size).to(device)
        ref_out = naive_attention_with_mask(q, k_exp, v_exp, mask)

        tri_out = diff_attn_varlen_with_noisy_input(q, k, v, cu_seqlens, L0_arr, L1_arr, staircase_size)

        max_diff = (ref_out.float() - tri_out.float()).abs().max().item()
        print(f"[GQA Forward h={n_heads},kv={n_kv_heads}] max_diff={max_diff:.6f}")
        assert max_diff < 0.01, f"GQA Forward h={n_heads},kv={n_kv_heads} FAILED: max_diff={max_diff}"
    print("[GQA Forward] ALL PASSED")


def test_gqa_backward():
    """Test backward pass with grouped-query attention."""
    torch.manual_seed(42)
    device = "cuda"
    staircase_size = 4
    L0, L1 = 16, 12
    total = L0 + 2 * L1

    for n_heads, n_kv_heads, head_dim in [(4, 2, 64), (16, 8, 128)]:
        n_kv_groups = n_heads // n_kv_heads

        # Reference in float32 with expanded k/v
        q_ref = torch.randn(total, n_heads, head_dim, device=device, dtype=torch.float32, requires_grad=True)
        k_ref_small = torch.randn(total, n_kv_heads, head_dim, device=device, dtype=torch.float32, requires_grad=True)
        v_ref_small = torch.randn(total, n_kv_heads, head_dim, device=device, dtype=torch.float32, requires_grad=True)

        k_ref = k_ref_small.repeat_interleave(n_kv_groups, dim=1)
        v_ref = v_ref_small.repeat_interleave(n_kv_groups, dim=1)

        mask = build_full_mask([L0], [L1], staircase_size).to(device)
        scale = 1.0 / math.sqrt(head_dim)
        qk = torch.einsum("ihd,jhd->hij", q_ref, k_ref) * scale
        mask_expanded = mask.unsqueeze(0).expand(n_heads, -1, -1)
        qk = qk.masked_fill(~mask_expanded, -1e9)
        attn = torch.softmax(qk, dim=-1)
        ref_out = torch.einsum("hij,jhd->ihd", attn, v_ref)
        ref_out.sum().backward()

        # Triton kernel
        for dtype_name, dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16)]:
            q_tri = q_ref.detach().clone().to(dtype).requires_grad_(True)
            k_tri = k_ref_small.detach().clone().to(dtype).requires_grad_(True)
            v_tri = v_ref_small.detach().clone().to(dtype).requires_grad_(True)

            cu_seqlens = torch.tensor([0, total], dtype=torch.int32, device=device)
            L0_arr = torch.tensor([L0], dtype=torch.int32, device=device)
            L1_arr = torch.tensor([L1], dtype=torch.int32, device=device)

            tri_out = diff_attn_varlen_with_noisy_input(q_tri, k_tri, v_tri, cu_seqlens, L0_arr, L1_arr, staircase_size)
            tri_out.float().sum().backward()

            tag = f"h={n_heads},kv={n_kv_heads},d={head_dim} {dtype_name}"
            print(f"[GQA Backward {tag}]")
            for name, g_ref_g, g_tri_g in [("dQ", q_ref.grad, q_tri.grad),
                                             ("dK", k_ref_small.grad, k_tri.grad),
                                             ("dV", v_ref_small.grad, v_tri.grad)]:
                _check_grad(f"{tag} {name}", g_ref_g, g_tri_g)
    print("[GQA Backward] ALL PASSED")


def test_gqa_multi_doc():
    """Test GQA with multiple documents packed in one batch."""
    torch.manual_seed(123)
    device = "cuda"
    n_heads, n_kv_heads = 8, 4
    n_kv_groups = n_heads // n_kv_heads
    head_dim = 64
    staircase_size = 2

    L0_list = [4, 2]
    L1_list = [6, 2]
    doc_lens = [L0 + 2 * L1 for L0, L1 in zip(L0_list, L1_list)]
    total_tokens = sum(doc_lens)

    q = torch.randn(total_tokens, n_heads, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(total_tokens, n_kv_heads, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(total_tokens, n_kv_heads, head_dim, device=device, dtype=torch.float16)

    cu_seqlens = torch.tensor([0] + [sum(doc_lens[:i+1]) for i in range(len(doc_lens))],
                               dtype=torch.int32, device=device)
    L0_arr = torch.tensor(L0_list, dtype=torch.int32, device=device)
    L1_arr = torch.tensor(L1_list, dtype=torch.int32, device=device)

    # Reference with expanded k/v
    k_exp = k.repeat_interleave(n_kv_groups, dim=1)
    v_exp = v.repeat_interleave(n_kv_groups, dim=1)
    mask = build_full_mask(L0_list, L1_list, staircase_size).to(device)
    ref_out = naive_attention_with_mask(q, k_exp, v_exp, mask)

    tri_out = diff_attn_varlen_with_noisy_input(q, k, v, cu_seqlens, L0_arr, L1_arr, staircase_size)

    max_diff = (ref_out.float() - tri_out.float()).abs().max().item()
    print(f"[GQA Multi-doc] max_diff={max_diff:.6f}")
    assert max_diff < 0.01, f"GQA Multi-doc FAILED: max_diff={max_diff}"
    print("[GQA Multi-doc] PASSED")


def test_backward_multi_doc():
    """Test backward with multiple documents packed in one batch."""
    torch.manual_seed(314)
    device = "cuda"
    n_heads, n_kv_heads = 4, 2
    n_kv_groups = n_heads // n_kv_heads
    head_dim = 64
    staircase_size = 4

    L0_list = [16, 8, 12]
    L1_list = [12, 8, 8]
    doc_lens = [L0 + 2 * L1 for L0, L1 in zip(L0_list, L1_list)]
    total_tokens = sum(doc_lens)

    q_ref = torch.randn(total_tokens, n_heads, head_dim, device=device, dtype=torch.float32, requires_grad=True)
    k_ref_small = torch.randn(total_tokens, n_kv_heads, head_dim, device=device, dtype=torch.float32, requires_grad=True)
    v_ref_small = torch.randn(total_tokens, n_kv_heads, head_dim, device=device, dtype=torch.float32, requires_grad=True)

    k_ref = k_ref_small.repeat_interleave(n_kv_groups, dim=1)
    v_ref = v_ref_small.repeat_interleave(n_kv_groups, dim=1)

    mask = build_full_mask(L0_list, L1_list, staircase_size).to(device)
    scale = 1.0 / math.sqrt(head_dim)
    qk = torch.einsum("ihd,jhd->hij", q_ref, k_ref) * scale
    mask_expanded = mask.unsqueeze(0).expand(n_heads, -1, -1)
    qk = qk.masked_fill(~mask_expanded, -1e9)
    attn = torch.softmax(qk, dim=-1)
    ref_out = torch.einsum("hij,jhd->ihd", attn, v_ref)
    ref_out.sum().backward()

    for dtype_name, dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16)]:
        q_tri = q_ref.detach().clone().to(dtype).requires_grad_(True)
        k_tri = k_ref_small.detach().clone().to(dtype).requires_grad_(True)
        v_tri = v_ref_small.detach().clone().to(dtype).requires_grad_(True)

        cu_seqlens = torch.tensor([0] + [sum(doc_lens[:i+1]) for i in range(len(doc_lens))],
                                   dtype=torch.int32, device=device)
        L0_arr = torch.tensor(L0_list, dtype=torch.int32, device=device)
        L1_arr = torch.tensor(L1_list, dtype=torch.int32, device=device)

        tri_out = diff_attn_varlen_with_noisy_input(q_tri, k_tri, v_tri, cu_seqlens, L0_arr, L1_arr, staircase_size)
        tri_out.float().sum().backward()

        print(f"[Multi-doc Backward {dtype_name}]")
        for name, g_ref_g, g_tri_g in [("dQ", q_ref.grad, q_tri.grad),
                                         ("dK", k_ref_small.grad, k_tri.grad),
                                         ("dV", v_ref_small.grad, v_tri.grad)]:
            _check_grad(f"multi-doc {dtype_name} {name}", g_ref_g, g_tri_g)
    print("[Multi-doc Backward] PASSED")


def test_backward_longer_seq():
    """Test backward with longer sequences to check numerical stability.

    Uses per-document checking to avoid reference OOM for very large masks.
    """
    torch.manual_seed(2024)
    device = "cuda"
    n_heads, n_kv_heads = 8, 4
    n_kv_groups = n_heads // n_kv_heads
    head_dim = 64
    staircase_size = 4

    for L0, L1 in [(128, 64), (256, 128)]:
        total = L0 + 2 * L1
        print(f"[Backward longer L0={L0} L1={L1} total={total}]")

        q_ref = torch.randn(total, n_heads, head_dim, device=device, dtype=torch.float32, requires_grad=True)
        k_ref_small = torch.randn(total, n_kv_heads, head_dim, device=device, dtype=torch.float32, requires_grad=True)
        v_ref_small = torch.randn(total, n_kv_heads, head_dim, device=device, dtype=torch.float32, requires_grad=True)

        k_ref = k_ref_small.repeat_interleave(n_kv_groups, dim=1)
        v_ref = v_ref_small.repeat_interleave(n_kv_groups, dim=1)

        mask = build_full_mask([L0], [L1], staircase_size).to(device)
        scale = 1.0 / math.sqrt(head_dim)
        qk = torch.einsum("ihd,jhd->hij", q_ref, k_ref) * scale
        mask_expanded = mask.unsqueeze(0).expand(n_heads, -1, -1)
        qk = qk.masked_fill(~mask_expanded, -1e9)
        attn = torch.softmax(qk, dim=-1)
        ref_out = torch.einsum("hij,jhd->ihd", attn, v_ref)
        ref_out.sum().backward()

        for dtype_name, dtype in [("fp16", torch.float16)]:
            q_tri = q_ref.detach().clone().to(dtype).requires_grad_(True)
            k_tri = k_ref_small.detach().clone().to(dtype).requires_grad_(True)
            v_tri = v_ref_small.detach().clone().to(dtype).requires_grad_(True)

            cu_seqlens = torch.tensor([0, total], dtype=torch.int32, device=device)
            L0_arr = torch.tensor([L0], dtype=torch.int32, device=device)
            L1_arr = torch.tensor([L1], dtype=torch.int32, device=device)

            tri_out = diff_attn_varlen_with_noisy_input(q_tri, k_tri, v_tri, cu_seqlens, L0_arr, L1_arr, staircase_size)
            tri_out.float().sum().backward()

            # Slightly relaxed thresholds for longer sequences (more fp16 accumulation error)
            for name, g_ref_g, g_tri_g in [("dQ", q_ref.grad, q_tri.grad),
                                             ("dK", k_ref_small.grad, k_tri.grad),
                                             ("dV", v_ref_small.grad, v_tri.grad)]:
                _check_grad(f"L0={L0} {dtype_name} {name}", g_ref_g, g_tri_g,
                            cos_thresh=0.98, mag_lo=0.90, mag_hi=1.10)
    print("[Backward longer seq] PASSED")


if __name__ == "__main__":
    test_forward_small()
    test_forward_various_staircase()
    test_forward_bfloat16()
    test_backward()
    test_gqa_forward()
    test_gqa_backward()
    test_gqa_multi_doc()
    test_backward_multi_doc()
    test_backward_longer_seq()
    print("\nAll tests passed!")
