from typing import Optional, Tuple

import torch
from torch import nn
import torch.distributed as dist

from jetengine_ext.layers.activation import SiluAndMul
from jetengine_ext.layers.layernorm import RMSNorm
from jetengine_ext.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
    ReplicatedLinear,
)
from jetengine_ext.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from jetengine_ext.kernels import fused_moe
from jetengine_ext.models.sdar import SDARAttention as SDARMoeAttention
# --------------------------------------------------------------------------- #
#                               LOW-LEVEL BLOCKS                              #
# --------------------------------------------------------------------------- #
class SDARMoeMLP(nn.Module):
    """The dense MLP used inside both experts and non-MoE layers."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        # gate + up projection fused
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)          # (B, S, 2*I)
        x = self.act_fn(gate_up)                # SILU(x[:I]) * x[I:]
        return self.down_proj(x)                # (B, S, H)


class SDARMoeSparseMoeBlock(nn.Module):
    """Top-k sparse MoE block (Switch-Transformer routing)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        rms_norm_eps: float,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        # Gating must see the full hidden dimension on every rank, so replicate instead of RowParallel
        self.gate = ReplicatedLinear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [SDARMoeMLP(hidden_size, intermediate_size) for _ in range(num_experts)]
        )
        # Cache for fused weights - will be populated on first forward pass
        # self._w1 = None
        # self._w2 = None
        self._prepare_fused_weights = False


    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        S, H = hidden_states.shape
        if not hasattr(self, '_w1') or self._w1 is None:
            raise RuntimeError("Fused MoE weights _w1 not initialized. Ensure load_model was called.")
        if not hasattr(self, '_w2') or self._w2 is None:
            raise RuntimeError("Fused MoE weights _w2 not initialized. Ensure load_model was called.")
        flat = hidden_states.view(-1, H)
        router_logits = self.gate(flat)
        probs = torch.softmax(router_logits, dim=-1, dtype=torch.float)
        top_p, top_i = torch.topk(probs, self.top_k, dim=-1)
        top_p = top_p / top_p.sum(dim=-1, keepdim=True)
        out = fused_moe(hidden_states=hidden_states, w1=self._w1, w2=self._w2, topk_weights=top_p, topk_ids=top_i, inplace=False)
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(out)
        return out.view(S, H), router_logits





# --------------------------------------------------------------------------- #
#                               DECODER LAYER                                 #
# --------------------------------------------------------------------------- #
class SDARMoeDecoderLayer(nn.Module):
    """Decoder layer that can be either dense or MoE depending on config."""

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = SDARMoeAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            head_dim=getattr(config, "head_dim", None),
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            rope_theta=getattr(config, "rope_theta", 10000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )

        is_moe_layer = (
            config.num_experts > 0
            and (layer_idx + 1) % config.decoder_sparse_step == 0
        )
        if is_moe_layer:
            self.mlp = SDARMoeSparseMoeBlock(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                num_experts=config.num_experts,
                top_k=config.num_experts_per_tok,
                rms_norm_eps=config.rms_norm_eps,
            )
        else:
            self.mlp = SDARMoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # ---- Attention ----
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions, hidden_states)

        # ---- FFN / MoE ----
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        mlp_out = self.mlp(hidden_states)

        if isinstance(mlp_out, tuple):   # MoE returns (H, logits)
            mlp_out, router_logits = mlp_out
        else:
            router_logits = None

        hidden_states = mlp_out          # residual is added inside layernorms
        return hidden_states, residual, router_logits


# --------------------------------------------------------------------------- #
#                                 FULL MODEL                                  #
# --------------------------------------------------------------------------- #
class SDARMoeModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [SDARMoeDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...] | None]:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        router_logits_accum: list[torch.Tensor] = []

        for layer in self.layers:
            hidden_states, residual, router_logits = layer(positions, hidden_states, residual)
            if router_logits is not None:
                router_logits_accum.append(router_logits)

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states, tuple(router_logits_accum) if router_logits_accum else None


class SDARMoeForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config) -> None:
        super().__init__()
        self.model = SDARMoeModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    # --------------------------------------------------------------------- #
    #                        PUBLIC INFERENCE API                           #
    # --------------------------------------------------------------------- #
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hidden_states, _ = self.model(input_ids, positions)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        return logits