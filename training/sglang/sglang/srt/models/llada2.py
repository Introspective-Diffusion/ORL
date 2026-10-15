# coding=utf-8
# Copyright 2023 Antgroup and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SGLang LLaDA2MoeModelLM model."""

import logging
from typing import Iterable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    parallel_state,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.communicator import (
    LayerCommunicator,
    LayerScatterModes,
    enable_moe_dense_fully_dp,
)
from sglang.srt.layers.dp_attention import (
    attn_tp_all_reduce,
    get_attention_dp_size,
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import get_deepep_mode, get_moe_a2a_backend
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.token_dispatcher import DeepEPDispatcher
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.utils import (
    apply_qk_norm,
    create_fused_set_kv_buffer_arg,
    enable_fused_set_kv_buffer,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, is_cuda, is_non_idle_and_non_empty, make_layers

LoraConfig = None
logger = logging.getLogger(__name__)
_is_cuda = is_cuda()


class LLaDA2SelfConditioning(nn.Module):
    """Gated residual adapter over the previous denoise step's soft embeddings.

    Shape is this fork's SDARSelfConditioning (sdar.py:94-142), not the jetengine
    one: plain nn.Linear so the adapter stays out of sglang's TP registry and is
    bit-identical on every rank, and post_norm is a forward ARGUMENT rather than a
    config read, so the algorithm can force it per call.

    The prenorm/out reparametrization knobs must move together. Half-configured is
    a silent 8-32x output error -- the same class of failure as a checkpoint that
    stores residual_scale 1.0 next to weights trained at 8x -- so it raises.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = getattr(
            config, "self_conditioning_intermediate_size", config.intermediate_size
        )
        self.pre_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, inputs_embeds, signal, post_norm: bool = False):
        _prenorm_scale = float(
            getattr(self.config, "self_conditioning_prenorm_scale", 1.0)
        )
        _reparam_out_scale = float(
            getattr(self.config, "self_conditioning_reparam_out_scale", 1.0)
        )
        if (_prenorm_scale == 1.0) != (_reparam_out_scale == 1.0):
            raise RuntimeError(
                "self_conditioning reparam knobs must be set together: "
                f"prenorm_scale={_prenorm_scale} out_scale={_reparam_out_scale}"
            )
        signal = self.pre_norm(signal) * _prenorm_scale
        gated = F.silu(self.gate_proj(signal)) * self.up_proj(signal)
        residual_scale = float(
            getattr(self.config, "self_conditioning_residual_scale", 1.0)
        )
        combined = inputs_embeds + residual_scale * _reparam_out_scale * self.down_proj(gated)
        if post_norm:
            scale = float(getattr(self.config, "self_conditioning_post_norm_scale", 1.0))
            return F.rms_norm(combined, (combined.size(-1),), eps=self.pre_norm.variance_epsilon) * scale
        return combined


class LLaDA2MoeMLP(nn.Module):
    def __init__(
        self,
        intermediate_size: int,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: Optional[bool] = True,
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.tp_size = tp_size

        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [intermediate_size] * 2,
            bias=config.use_bias,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=config.use_bias,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )

        if config.hidden_act != "silu":
            raise ValueError("Unsupported activation. Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        if (self.tp_size == 1) and hidden_states.shape[0] == 0:
            return hidden_states

        gate_up, _ = self.gate_up_proj(hidden_states)
        hidden_states = self.act_fn(gate_up)
        hidden_states, _ = self.down_proj(
            hidden_states, skip_all_reduce=use_reduce_scatter
        )
        return hidden_states


class LLaDA2MoeGate(nn.Module):
    def __init__(
        self,
        config,
        params_dtype: Optional[torch.dtype] = None,
        prefix: str = "",
    ):
        super().__init__()
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        self.weight = nn.Parameter(
            torch.empty(
                (config.num_experts, config.hidden_size),
                dtype=self.params_dtype,
            ),
        )
        if getattr(config, "moe_router_enable_expert_bias", False):
            self.expert_bias = nn.Parameter(
                torch.empty((config.num_experts,), dtype=torch.float32),
            )
        else:
            self.expert_bias = None

    def forward(self, hidden_states):
        logits = F.linear(hidden_states.to(self.weight.dtype), self.weight, None).to(
            hidden_states.dtype
        )
        return logits


class LLaDA2MoeSparseMoeBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_id = layer_id
        self.alt_stream = alt_stream
        self.tp_size = get_tensor_model_parallel_world_size()
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_size = config.hidden_size
        self.num_shared_experts = config.num_shared_experts
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.score_function = getattr(config, "score_function", None)

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )

        # Gate always runs at half / full precision for now.
        router_dtype = getattr(config, "router_dtype", None)
        if router_dtype is None:
            self.router_dtype = None
        elif router_dtype == "fp32":
            self.router_dtype = torch.float32
        else:
            self.router_dtype = torch.bfloat16

        # TODO global_server_args.ep_num_redundant_experts is used for eplb, not supported now
        assert get_global_server_args().ep_num_redundant_experts == 0
        # check group topk
        self.num_expert_group = getattr(config, "n_group", 0)
        self.topk_group = getattr(config, "topk_group", 0)
        if self.num_expert_group > 0 or self.topk_group > 0:
            assert (
                self.num_expert_group > 0
                and 0 < self.topk_group <= self.num_expert_group
            )
            self.use_grouped_topk = True
        else:
            self.num_expert_group = self.topk_group = None
            self.use_grouped_topk = False

        self.num_experts = (
            config.num_experts + get_global_server_args().ep_num_redundant_experts
        )

        self.gate = LLaDA2MoeGate(
            config=config,
            params_dtype=self.router_dtype,
            prefix=add_prefix("gate", prefix),
        )
        self.correction_bias = (
            self.gate.expert_bias.data if self.gate.expert_bias is not None else None
        )

        if self.score_function is not None:
            assert (
                self.score_function == "softmax" and self.correction_bias is None
            ) or (
                self.score_function == "sigmoid" and self.correction_bias is not None
            ), "score_function and correction_bias should be in 2 combination (softmax, None) or (sigmoid, not None)"

        self.topk = TopK(
            top_k=self.top_k,
            renormalize=self.norm_topk_prob,
            use_grouped_topk=self.use_grouped_topk,
            num_expert_group=self.num_expert_group,
            # num_fused_shared_experts=self.num_fused_shared_experts,
            topk_group=self.topk_group,
            correction_bias=self.correction_bias,
            routed_scaling_factor=self.routed_scaling_factor,
        )

        self.experts = get_moe_impl_class(quant_config)(
            num_experts=self.num_experts,
            top_k=self.top_k,
            layer_id=self.layer_id,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            routed_scaling_factor=self.routed_scaling_factor,
            prefix=add_prefix("experts", prefix),
        )
        # shared expert
        if config.num_shared_experts is not None:
            if hasattr(config, "moe_shared_expert_intermediate_size"):
                intermediate_size = config.moe_shared_expert_intermediate_size
            else:
                intermediate_size = config.moe_intermediate_size
            intermediate_size *= config.num_shared_experts
            # disable tp for shared experts when enable deepep moe
            self.shared_experts = LLaDA2MoeMLP(
                intermediate_size=intermediate_size,
                config=config,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_experts", prefix),
                **(
                    dict(tp_rank=0, tp_size=1)
                    if get_moe_a2a_backend().is_deepep()
                    else {}
                ),
            )
        # dispatcher
        if get_moe_a2a_backend().is_deepep():
            # TODO: we will support tp < ep in the future
            self.ep_size = get_tensor_model_parallel_world_size()

            self.deepep_dispatcher = DeepEPDispatcher(
                group=parallel_state.get_tp_group().device_group,
                router_topk=self.top_k,
                permute_fusion=True,
                num_experts=self.num_experts,
                num_local_experts=config.num_experts // self.tp_size,
                hidden_size=config.hidden_size,
                params_dtype=config.torch_dtype,
                deepep_mode=get_deepep_mode(),
                async_finish=True,  # TODO
                return_recv_hook=True,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        if not get_moe_a2a_backend().is_deepep():
            return self.forward_normal(hidden_states, use_reduce_scatter)
        else:
            return self.forward_deepep(hidden_states, forward_batch)

    def get_moe_weights(self):
        return [
            x.data
            for name, x in self.experts.named_parameters()
            if name not in ["correction_bias"]
        ]

    def _forward_shared_experts(self, hidden_states: torch.Tensor):
        shared_output = None
        if self.num_shared_experts > 0:
            shared_output = self.shared_experts(hidden_states)
        return shared_output

    def _forward_router_experts(self, hidden_states: torch.Tensor):
        # router_logits: (num_tokens, n_experts)
        router_logits = self.gate(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)
        return self.experts(hidden_states, topk_output)

    def forward_normal_dual_stream(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        current_stream = torch.cuda.current_stream()
        self.alt_stream.wait_stream(current_stream)
        shared_output = self._forward_shared_experts(hidden_states.clone())

        with torch.cuda.stream(self.alt_stream):
            router_output = self._forward_router_experts(hidden_states)
        current_stream.wait_stream(self.alt_stream)

        return router_output, shared_output

    def forward_normal(
        self,
        hidden_states: torch.Tensor,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)

        if (
            self.alt_stream is not None
            and hidden_states.shape[0] > 0
            and get_is_capture_mode()
        ):
            final_hidden_states, shared_output = self.forward_normal_dual_stream(
                hidden_states
            )
        else:
            shared_output = self._forward_shared_experts(hidden_states)
            final_hidden_states = self._forward_router_experts(hidden_states)

        if self.num_shared_experts > 0:
            final_hidden_states = final_hidden_states + shared_output

        if self.tp_size > 1 and not use_reduce_scatter:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        return final_hidden_states.view(num_tokens, hidden_size)

    def forward_deepep(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        shared_output = None
        forward_mode = forward_batch.forward_mode
        if is_non_idle_and_non_empty(forward_mode, hidden_states):
            router_logits = self.gate(hidden_states)
            if self.num_shared_experts > 0:
                shared_output = self.shared_experts(hidden_states)

            topk_output = self.topk(
                hidden_states,
                router_logits,
                num_token_non_padded=forward_batch.num_token_non_padded,
                expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                    layer_id=self.layer_id,
                ),
            )
        else:
            topk_output = self.topk.empty_topk_output(hidden_states.device)

        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )

        if shared_output is not None:
            final_hidden_states += shared_output
        return final_hidden_states


class LLaDA2MoeAttention(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_kv_heads = config.num_key_value_heads
        self.dp_size = get_attention_dp_size()
        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        assert self.total_num_heads % attn_tp_size == 0
        if self.total_kv_heads >= attn_tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_kv_heads % attn_tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert attn_tp_size % self.total_kv_heads == 0
        assert self.total_num_heads >= self.total_kv_heads

        self.num_heads = self.total_num_heads // attn_tp_size
        self.head_dim = config.head_dim or (self.hidden_size // self.total_num_heads)
        self.q_size = self.head_dim * self.num_heads

        self.num_kv_heads = max(1, self.total_kv_heads // attn_tp_size)
        self.kv_size = max(1, self.num_kv_heads * self.head_dim)

        self.scale = self.head_dim**-0.5

        self.use_qk_norm = getattr(config, "use_qk_norm", True)

        self.query_key_value = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_kv_heads,
            bias=(config.use_bias or config.use_qkv_bias),
            quant_config=quant_config,
            prefix=add_prefix("query_key_value", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )

        if self.use_qk_norm:
            self.query_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.key_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.dense = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=config.use_bias,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("dense", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )

        if hasattr(config, "partial_rotary_factor"):
            self.rotary_dim = int(self.head_dim * config.partial_rotary_factor)
        elif hasattr(config, "rotary_dim"):
            self.rotary_dim = config.rotary_dim
        else:
            self.rotary_dim = self.head_dim
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.rotary_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )

        # _DUAL_ATTN: mirror of sdar.py -- the bidirectional branch is what this
        # model has always used; the causal branch is what the dllm protocol asks
        # for during prefill and the clean-x0 pass.
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scale,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            attn_type=AttentionType.ENCODER_ONLY,
            prefix=add_prefix("attn", prefix),
        )
        self.attn_bidir = self.attn
        self.attn_causal = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scale,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix),
        )

        self.alt_stream = alt_stream


    def _select_attention(self, forward_batch):
        """Which mask this phase needs. Copied from sdar.py:248-260.

        JointThreshold sets none of these flags, so it always lands on
        attn_bidir -- the single path this model had before this patch.
        """
        if (
            getattr(forward_batch, "dllm_force_causal", False)
            or getattr(forward_batch, "dllm_is_encoder", False)
            or getattr(forward_batch, "dllm_is_prefill", False)
        ):
            return self.attn_causal
        if getattr(forward_batch, "dllm_is_denoise", False):
            return self.attn_bidir
        fm = getattr(forward_batch, "forward_mode", None)
        if fm is not None and getattr(fm, "is_dllm_extend", None) and fm.is_dllm_extend():
            return self.attn_bidir
        return self.attn_bidir

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states
        qkv, _ = self.query_key_value(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.use_qk_norm:
            q, k = apply_qk_norm(
                q=q,
                k=k,
                q_norm=self.query_layernorm,
                k_norm=self.key_layernorm,
                head_dim=self.head_dim,
                alt_stream=self.alt_stream,
            )
        can_fuse_set_kv = (
            self.head_dim == self.rotary_emb.rotary_dim
            and enable_fused_set_kv_buffer(forward_batch)
        )
        q, k = self.rotary_emb(
            positions,
            q,
            k,
            fused_set_kv_buffer_arg=(
                create_fused_set_kv_buffer_arg(
                    value=v,
                    layer=self._select_attention(forward_batch),
                    forward_batch=forward_batch,
                )
                if can_fuse_set_kv
                else None
            ),
        )
        context_layer = self._select_attention(forward_batch)(
            q,
            k,
            v,
            forward_batch,
            save_kv_cache=not can_fuse_set_kv,
        )
        attn_output, _ = self.dense(context_layer)
        return attn_output


class LLaDA2MoeBlock(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        hidden_size = config.hidden_size

        self.input_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.dp_size = get_attention_dp_size()
        self.attention = LLaDA2MoeAttention(
            config,
            layer_id,
            quant_config,
            reduce_results=False,
            prefix=add_prefix("attention", prefix),
            alt_stream=alt_stream,
        )
        self.layer_id = layer_id
        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()

        self.is_layer_sparse = self._is_layer_sparse(config, layer_id=layer_id)
        is_previous_layer_sparse = self._is_layer_sparse(config, layer_id=layer_id - 1)
        is_next_layer_sparse = self._is_layer_sparse(config, layer_id=layer_id + 1)

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=is_next_layer_sparse,
        )

        self.is_last_layer = self.layer_id == config.num_hidden_layers - 1

        if self.is_layer_sparse:
            self.mlp = LLaDA2MoeSparseMoeBlock(
                layer_id=layer_id,
                config=config,
                quant_config=quant_config,
                alt_stream=alt_stream,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            if enable_moe_dense_fully_dp():
                mlp_tp_rank, mlp_tp_size = 0, 1
            else:
                mlp_tp_rank, mlp_tp_size = None, None
            self.mlp = LLaDA2MoeMLP(
                intermediate_size=config.intermediate_size,
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                tp_rank=mlp_tp_rank,
                tp_size=mlp_tp_size,
            )

        self.post_attention_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
        )

    def _is_layer_sparse(self, config: PretrainedConfig, layer_id: int) -> bool:
        return (
            config.num_experts is not None and layer_id >= config.first_k_dense_replace
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> torch.Tensor:
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=forward_batch,
        )

        hidden_states = self.attention(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=forward_batch,
        )

        # For DP with padding, reduce scatter can be used instead of all-reduce.
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )

        hidden_states = self.mlp(hidden_states, forward_batch, use_reduce_scatter)

        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=forward_batch,
        )

        return hidden_states, residual


class LLaDA2MoeModel(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_dim = config.hidden_size
        if self.pp_group.is_first_rank:
            self.word_embeddings = VocabParallelEmbedding(
                self.vocab_size,
                self.embed_dim,
                quant_config=quant_config,
                prefix=add_prefix("word_embeddings", prefix),
                use_attn_tp_group=is_dp_attention_enabled(),
            )
        else:
            self.word_embeddings = PPMissingLayer()

        self.embedding_dropout = torch.nn.Dropout(config.embedding_dropout)

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: LLaDA2MoeBlock(
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(self.embed_dim, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.word_embeddings(input_ids)
            else:
                hidden_states = input_embeds
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        for i in range(self.start_layer, self.end_layer):
            with get_global_expert_distribution_recorder().with_current_layer(i):
                layer = self.layers[i]
                hidden_states, residual = layer(
                    positions,
                    hidden_states,
                    forward_batch,
                    residual,
                )
        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
        else:
            if not forward_batch.forward_mode.is_idle():
                if residual is None:
                    hidden_states = self.norm(hidden_states)
                else:
                    hidden_states, _ = self.norm(hidden_states, residual)
            return hidden_states


class LLaDA2MoeModelLM(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        alt_stream = torch.cuda.Stream() if _is_cuda else None

        self.model = LLaDA2MoeModel(
            config,
            quant_config,
            alt_stream=alt_stream,
            prefix=add_prefix("model", ""),
        )

        if config.tie_word_embeddings:
            self.lm_head = self.model.word_embeddings
        else:
            # TODO something wrong with ParallelLMHead with DP attention enabled
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
                use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
            )
        self.logits_processor = LogitsProcessor(config, return_full_logits=True)

        # Self-conditioning. Hung on self.model so the state-dict path is
        # `model.self_conditioning.*` -- the same path the FSDP-side adapter
        # (llada_residual.py attaches at inner.self_conditioning) produces. If the
        # two disagree, load_weights' final `if name not in params_dict: continue`
        # drops the whole adapter silently, and a zero-init adapter that never
        # loaded is indistinguishable from a working one on step 0.
        self.use_self_conditioning = bool(
            getattr(config, "use_self_conditioning", False)
        )
        if self.use_self_conditioning:
            sc = LLaDA2SelfConditioning(config)
            # down_proj zero-init: step 0 is then bit-identical to stock llada2, so
            # the parity gate can be an exact-equality check rather than a tolerance.
            if bool(getattr(config, "self_conditioning_zero_init", True)):
                with torch.no_grad():
                    sc.down_proj.weight.zero_()
            self.model.self_conditioning = sc
        else:
            self.model.self_conditioning = None

    @torch.no_grad()
    def build_self_conditioned_input_embeds(
        self,
        input_ids: torch.Tensor,
        prev_logits: Optional[torch.Tensor],
        self_conditioning_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Previous denoise step's prediction, pushed back into embedding space.

        Ported from this fork's sdar.py:637-760. Three properties are load-bearing
        and each was got wrong in our HF implementation before this:

        * the signal is the PREVIOUS step's logits, handed in by the algorithm --
          never recomputed with a second forward at the same step (that cost
          exactly 2x forwards and conditioned on a different quantity);
        * only `self_conditioning_mask` positions receive it, everything else is
          exactly zero;
        * `prev_logits is None` -- the first step of a block -- returns None, so the
          adapter does not run at all and the block starts from a zero signal.

        Vocab-parallel embeddings are handled the way sdar.py does: remap the probs
        into the local shard via shard_indices and all-reduce, because a softmax
        over a sharded vocab is not separable.
        """
        if (
            not self.use_self_conditioning
            or prev_logits is None
            or self.model.self_conditioning is None
            or not bool(self_conditioning_mask.any())
        ):
            # Never leave a stale signal behind: a reader that finds last step's
            # tensor here would be silently wrong, which is worse than finding None.
            # Never leave a stale signal behind. The trace writer reads these
            # attributes off the model; finding last step's tensor here would be
            # silently wrong, which is worse than finding nothing.
            self._last_self_conditioning_signal = None
            self._last_self_conditioning_valid = None
            return None

        inputs_embeds = self.model.word_embeddings(input_ids)
        selected = torch.nonzero(self_conditioning_mask, as_tuple=False).flatten()
        if selected.numel() == 0:
            return inputs_embeds

        signal = torch.zeros_like(inputs_embeds)
        flat_signal = signal.view(-1, signal.size(-1))
        flat_logits = prev_logits.view(-1, prev_logits.size(-1))
        embed_tokens = self.model.word_embeddings
        embed_weight = embed_tokens.weight
        # Chunked so the fp32 softmax over 157184 columns never materialises for
        # more rows than this at once.
        chunk = max(1, int(getattr(self.config, "self_conditioning_chunk_size", 256)))
        for start in range(0, selected.numel(), chunk):
            idx = selected[start : start + chunk]
            logits = flat_logits.index_select(0, idx).float()
            probs = F.softmax(logits, dim=-1).to(embed_weight.dtype)
            if probs.size(-1) == embed_weight.size(0):
                soft = probs.matmul(embed_weight)
            elif hasattr(embed_tokens, "shard_indices"):
                si = embed_tokens.shard_indices
                local = probs.new_zeros((probs.size(0), embed_weight.size(0)))
                if si.org_vocab_end_index > si.org_vocab_start_index:
                    local[:, : si.num_org_elements] = probs[
                        :, si.org_vocab_start_index : si.org_vocab_end_index
                    ]
                if si.added_vocab_end_index > si.added_vocab_start_index:
                    ls = si.num_org_elements_padded
                    local[:, ls : ls + si.num_added_elements] = probs[
                        :, si.added_vocab_start_index : si.added_vocab_end_index
                    ]
                soft = local.matmul(embed_weight)
                if getattr(embed_tokens, "tp_size", 1) > 1:
                    if getattr(embed_tokens, "use_attn_tp_group", False):
                        soft = attn_tp_all_reduce(soft)
                    else:
                        soft = tensor_model_parallel_all_reduce(soft)
            else:
                raise RuntimeError(
                    "self-conditioning vocab mismatch: logits_vocab=%d embed_rows=%d"
                    % (probs.size(-1), embed_weight.size(0))
                )
            flat_signal.index_copy_(0, idx, soft.to(inputs_embeds.dtype))

        post_norm = bool(
            getattr(self, "_force_self_conditioning_post_norm", False)
            or getattr(self.config, "self_conditioning_post_norm", False)
        )
        self._last_self_conditioning_signal = signal
        self._last_self_conditioning_valid = self_conditioning_mask
        return self.model.self_conditioning(
            inputs_embeds, signal.to(inputs_embeds.dtype), post_norm=post_norm
        )

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def get_embed_and_head(self):
        """Used by the eagle_worker."""
        return self.model.word_embeddings.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        """Used by the eagle_worker."""
        del self.model.word_embeddings.weight
        del self.lm_head.weight
        self.model.word_embeddings.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch
            )
        else:
            return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

        params_dict = dict(self.named_parameters())
        # O(1) expert lookup. The stock loop below scans all
        # num_experts*3 mappings for every incoming tensor with a substring test;
        # on LLaDA2.1-mini that is ~1.5e4 expert tensors x 768 checks per weight
        # sync. Key the mappings by their "experts.<id>.<proj>." token instead.
        import re as _re

        _expert_lookup = {weight_name: m for m in expert_params_mapping for weight_name in [m[1]]}
        _expert_key_re = _re.compile(r"(experts\.\d+\.(?:gate_proj|up_proj|down_proj)\.)")
        # Silent-skip accounting. Every `name not in params_dict: continue` below
        # drops a tensor with no error. On a weight-sync path that is the worst
        # failure available: the run trains, logs normally, and rolls out a policy
        # frozen at its initial weights. Count what arrives, what lands, and what is
        # dropped, then refuse to continue if the shape of that is wrong.
        _n_seen = 0
        _n_loaded = 0
        _skipped = []
        for name, loaded_weight in weights:
            _n_seen += 1
            if (
                ("v_head" in name)
                or ("inv_freq" in name)
                or (self.config.tie_word_embeddings and "lm_head" in name)
            ):
                continue

            if (
                hasattr(self.config, "norm_head")
                and self.config.norm_head
                and "lm_head.weight" in name
            ):
                import torch.nn.functional as F

                loaded_weight = F.normalize(loaded_weight, dim=0, p=2, eps=1e-7)

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:
                    continue
                # Skip the self-conditioning adapter for the same reason. It is a separate gated MLP with its own
                # gate_proj / up_proj / down_proj, whose names collide with the fused MLP's;
                # letting it through this rewrite yields model.self_conditioning.gate_up_proj,
                # and a second pass gives gate_gate_up_proj; neither name exists, so the adapter
                # would not be loaded at all.
                if "self_conditioning" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    _skipped.append(name)
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                _n_loaded += 1
                break
            else:
                _m = _expert_key_re.search(name) if "experts." in name else None
                _direct = _expert_lookup.get(_m.group(1)) if _m else None
                for mapping in ([_direct] if _direct is not None else expert_params_mapping):
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    if name not in params_dict:
                        _skipped.append(name)
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    _n_loaded += 1
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name not in params_dict:
                        _skipped.append(name)
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    _n_loaded += 1

        _sc_expected = sum(1 for n in params_dict if ".self_conditioning." in n)
        _sc_skipped = [n for n in _skipped if ".self_conditioning." in n]
        _exp_skipped = [n for n in _skipped if "experts" in n]
        logger.info(
            "[llada2 load_weights] seen=%d loaded=%d skipped=%d "
            "(experts_skipped=%d) self_conditioning params=%d",
            _n_seen, _n_loaded, len(_skipped), len(_exp_skipped), _sc_expected,
        )
        if _exp_skipped:
            raise RuntimeError(
                "llada2 load_weights dropped %d expert tensor(s) -- e.g. %s. This is "
                "silent weight loss: the engine would serve a policy frozen at its "
                "initial expert weights while the run looks healthy. Check the "
                "expert name mapping against the checkpoint's actual layout."
                % (len(_exp_skipped), _exp_skipped[:3])
            )
        if _sc_skipped:
            # A zero-init adapter that never loaded is indistinguishable from a
            # working one on step 0 -- the parity gate cannot see it. So check the
            # names, not the output.
            raise RuntimeError(
                "self_conditioning tensor(s) arrived but did not land: %s. The engine "
                "module must hang at the same state-dict path the trainer writes "
                "(model.self_conditioning.*)." % _sc_skipped[:3]
            )

        self.routed_experts_weights_of_layer = {
            layer_id: layer.mlp.get_moe_weights()
            for layer_id, layer in enumerate(self.model.layers)
            if not isinstance(layer, PPMissingLayer)
            and isinstance(layer.mlp, LLaDA2MoeSparseMoeBlock)
        }

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        num_groups = getattr(config, "n_group", 0)
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.num_experts,
            num_groups=None if num_groups == 0 else num_groups,
        )


EntryClass = LLaDA2MoeModelLM
