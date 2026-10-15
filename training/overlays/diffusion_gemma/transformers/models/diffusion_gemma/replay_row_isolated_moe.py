from contextlib import contextmanager
from contextvars import ContextVar

import torch

_DIFFUSION_GEMMA_DECODER_ROW_ISOLATED_MOE = ContextVar(
    "diffusion_gemma_decoder_row_isolated_moe",
    default=False,
)


@contextmanager
def diffusion_gemma_decoder_row_isolated_moe(enabled: bool = True):
    token = _DIFFUSION_GEMMA_DECODER_ROW_ISOLATED_MOE.set(bool(enabled))
    try:
        yield
    finally:
        _DIFFUSION_GEMMA_DECODER_ROW_ISOLATED_MOE.reset(token)


def diffusion_gemma_decoder_moe_forward(layer, residual: torch.Tensor) -> torch.Tensor:
    def run_moe(residual_rows: torch.Tensor) -> torch.Tensor:
        hidden_states_flat = residual_rows.reshape(-1, residual_rows.shape[-1])
        hidden_states_2_for_routing = hidden_states_flat
        hidden_states_2_for_experts = layer.pre_feedforward_layernorm_2(hidden_states_flat)
        _, top_k_weights, top_k_index = layer.router(hidden_states_2_for_routing)
        hidden_states_2 = layer.experts(hidden_states_2_for_experts, top_k_index, top_k_weights)
        return hidden_states_2.reshape(residual_rows.shape)

    if not _DIFFUSION_GEMMA_DECODER_ROW_ISOLATED_MOE.get() or residual.shape[0] <= 1:
        return run_moe(residual)
    return torch.cat([run_moe(row) for row in residual.split(1, dim=0)], dim=0)
