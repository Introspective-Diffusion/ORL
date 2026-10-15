import pytest
import torch
import torch.nn.functional as F

from fast_diffusion_attention.external_prefix_kv_attn import (
    external_kv_varlen_attention,
)


def _cu_seqlens(lengths: list[int], device: torch.device) -> torch.Tensor:
    values = [0]
    for length in lengths:
        values.append(values[-1] + int(length))
    return torch.tensor(values, dtype=torch.int32, device=device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_eight_query_segments_share_one_detached_prefix(dtype: torch.dtype) -> None:
    """Mapped Triton attention reuses one prefix without changing query gradients."""
    torch.manual_seed(400)
    device = torch.device("cuda")
    query_segments, query_length, prefix_length = 8, 64, 256
    query_heads, kv_heads, head_dim = 8, 4, 64
    total_query_tokens = query_segments * query_length

    base_query = torch.randn(
        total_query_tokens, query_heads, head_dim, device=device, dtype=dtype
    )
    base_key = torch.randn(
        total_query_tokens, kv_heads, head_dim, device=device, dtype=dtype
    )
    base_value = torch.randn_like(base_key)
    prefix_key = torch.randn(
        prefix_length, kv_heads, head_dim, device=device, dtype=dtype
    )
    prefix_value = torch.randn_like(prefix_key)
    cu_query = _cu_seqlens([query_length] * query_segments, device)
    cu_prefix = _cu_seqlens([prefix_length], device)
    prefix_segment_ids = torch.zeros(
        query_segments, dtype=torch.int32, device=device
    )
    output_gradient = torch.randn_like(base_query)

    results: dict[str, tuple[torch.Tensor, list[torch.Tensor]]] = {}
    for backend in ("reference", "triton"):
        query = base_query.detach().clone().requires_grad_(True)
        key = base_key.detach().clone().requires_grad_(True)
        value = base_value.detach().clone().requires_grad_(True)
        output = external_kv_varlen_attention(
            query,
            key,
            value,
            prefix_key,
            prefix_value,
            cu_query,
            cu_prefix,
            max_seqlen_q=query_length,
            max_seqlen_prefix=prefix_length,
            prefix_segment_ids=prefix_segment_ids,
            backend=backend,
        )
        output.backward(output_gradient)
        results[backend] = (
            output.detach(),
            [query.grad.detach(), key.grad.detach(), value.grad.detach()],
        )

    reference_output, reference_gradients = results["reference"]
    triton_output, triton_gradients = results["triton"]
    tolerance = 0.03 if dtype == torch.bfloat16 else 0.015
    torch.testing.assert_close(
        triton_output.float(),
        reference_output.float(),
        atol=tolerance,
        rtol=0.03,
    )
    for triton_gradient, reference_gradient in zip(
        triton_gradients, reference_gradients, strict=True
    ):
        cosine = F.cosine_similarity(
            triton_gradient.float().flatten(),
            reference_gradient.float().flatten(),
            dim=0,
        )
        assert float(cosine.item()) >= 0.995
        torch.testing.assert_close(
            triton_gradient.float(),
            reference_gradient.float(),
            atol=tolerance,
            rtol=0.03,
        )

    assert prefix_key.grad is None
    assert prefix_value.grad is None
