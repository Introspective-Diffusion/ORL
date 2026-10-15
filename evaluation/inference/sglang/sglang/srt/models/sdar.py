# coding=utf-8
"""
SGLang SDARModelLM (block diffusion / dLLM-style forward).
"""

import logging
from typing import Iterable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes
from sglang.srt.layers.dp_attention import (
    attn_tp_all_reduce,
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
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.models.utils import (
    apply_qk_norm,
    create_fused_set_kv_buffer_arg,
    enable_fused_set_kv_buffer,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, is_cuda, make_layers

logger = logging.getLogger(__name__)
_is_cuda = is_cuda()


class SDARMLP(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config=None,
        reduce_results: bool = True,
        prefix: str = "",
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor, use_reduce_scatter: bool = False):
        gate_up, _ = self.gate_up_proj(hidden_states)
        hidden_states = self.act_fn(gate_up)
        hidden_states, _ = self.down_proj(
            hidden_states, skip_all_reduce=use_reduce_scatter
        )
        return hidden_states


class SDARSelfConditioning(nn.Module):
    """Gated residual adapter for previous-iteration soft embeddings."""

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = getattr(
            config, "self_conditioning_intermediate_size", config.intermediate_size
        )
        self.pre_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=False
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        signal: torch.Tensor,
        post_norm: bool = False,
    ) -> torch.Tensor:
        _prenorm_scale = float(
            getattr(self.config, "self_conditioning_prenorm_scale", 1.0)
        )
        _reparam_out_scale = float(
            getattr(self.config, "self_conditioning_reparam_out_scale", 1.0)
        )
        # Reparametrization invariant: both knobs move together (see
        # training/tools/make_scrp_base.py). Half-configured state is a
        # silent 8-32x output error, so fail loudly instead.
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
            return F.rms_norm(
                combined,
                (combined.size(-1),),
                eps=self.pre_norm.variance_epsilon,
            ) * scale
        return combined


class SDARAttention(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config=None,
        reduce_results: bool = True,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.tp_size = get_tensor_model_parallel_world_size()
        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= attn_tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)

        self.head_dim = getattr(
            config, "head_dim", self.hidden_size // self.total_num_heads
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scale = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=getattr(config, "attention_bias", False),
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=getattr(config, "attention_bias", False),
            quant_config=quant_config,
            reduce_results=reduce_results,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("o_proj", prefix),
        )

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        rope_theta = getattr(config, "rope_theta", 10000.0)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_pos = getattr(config, "max_position_embeddings", 32768)
        self.rotary_dim = self.head_dim
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.rotary_dim,
            max_position=max_pos,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )

        # Shared q/k/v/o projections with two attention wrappers. This mirrors
        # DiffusionGemma's serving structure: denoise uses bidirectional block
        # attention; prompt prefill and clean commit use causal attention.
        self.attn_bidir = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scale,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            attn_type=AttentionType.ENCODER_ONLY,
            prefix=add_prefix("attn", prefix),
        )
        self.attn_causal = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scale,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix),
        )
        self.alt_stream = alt_stream

    def _select_attention(self, forward_batch: ForwardBatch) -> RadixAttention:
        if (
            getattr(forward_batch, "dllm_force_causal", False)
            or getattr(forward_batch, "dllm_is_encoder", False)
            or getattr(forward_batch, "dllm_is_prefill", False)
        ):
            return self.attn_causal
        if getattr(forward_batch, "dllm_is_denoise", False):
            return self.attn_bidir
        if forward_batch.forward_mode.is_dllm_extend():
            return self.attn_bidir
        # Normal generation without a dLLM algorithm is standard AR decoding.
        return self.attn_causal

    def forward_prepare_native(self, positions, hidden_states, forward_batch):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = apply_qk_norm(
            q=q,
            k=k,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            head_dim=self.head_dim,
            alt_stream=self.alt_stream,
        )
        attn = self._select_attention(forward_batch)
        # Disable fused rope+KV store for DLLM (dtype mismatch: int32 vs int64)
        use_fused = (
            enable_fused_set_kv_buffer(forward_batch)
            and not forward_batch.forward_mode.is_dllm_extend()
        )
        q, k = self.rotary_emb(
            positions,
            q,
            k,
            fused_set_kv_buffer_arg=(
                create_fused_set_kv_buffer_arg(
                    value=v,
                    layer=attn,
                    forward_batch=forward_batch,
                )
                if use_fused
                else None
            ),
        )
        return q, k, v

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        if get_global_server_args().rl_on_policy_target is not None:
            hidden_states = hidden_states.bfloat16()

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = apply_qk_norm(
            q=q,
            k=k,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            head_dim=self.head_dim,
            alt_stream=self.alt_stream,
        )
        # Disable fused rope+KV store for DLLM: the fused kernel requires
        # positions and out_cache_loc to share the same dtype, but DLLM
        # positions are int32 while out_cache_loc is int64.
        use_fused = (
            enable_fused_set_kv_buffer(forward_batch)
            and not forward_batch.forward_mode.is_dllm_extend()
        )
        attn = self._select_attention(forward_batch)
        q, k = self.rotary_emb(
            positions,
            q,
            k,
            fused_set_kv_buffer_arg=(
                create_fused_set_kv_buffer_arg(
                    value=v,
                    layer=attn,
                    forward_batch=forward_batch,
                )
                if use_fused
                else None
            ),
        )

        if get_global_server_args().rl_on_policy_target is not None:
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)

        context_layer = attn(
            q,
            k,
            v,
            forward_batch,
            save_kv_cache=not use_fused,
        )
        out, _ = self.o_proj(context_layer)
        return out


class SDARBlock(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_id = layer_id

        norm_kwargs = (
            dict(
                weight_dtype=torch.float32,
                cast_x_before_out_mul=True,
                override_orig_dtype=torch.float32,
                fp32_residual=True,
            )
            if get_global_server_args().rl_on_policy_target is not None
            else {}
        )
        self.input_layernorm = RMSNorm(
            self.hidden_size, eps=config.rms_norm_eps, **norm_kwargs
        )
        self.post_attention_layernorm = RMSNorm(
            self.hidden_size, eps=config.rms_norm_eps, **norm_kwargs
        )

        self.self_attn = SDARAttention(
            layer_id=layer_id,
            config=config,
            quant_config=quant_config,
            reduce_results=False,
            prefix=add_prefix("self_attn", prefix),
            alt_stream=alt_stream,
        )

        self.mlp = SDARMLP(
            config=config,
            quant_config=quant_config,
            reduce_results=True,
            prefix=add_prefix("mlp", prefix),
        )

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=False,
            is_previous_layer_sparse=False,
            is_next_layer_sparse=False,
        )
        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states,
            residual,
            forward_batch,
        )

        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states,
            residual,
            forward_batch,
        )

        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )
        hidden_states = self.mlp(hidden_states, use_reduce_scatter=use_reduce_scatter)

        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )

        return hidden_states, residual


class SDARModel(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_dim = config.hidden_size
        self.pp_group = get_pp_group()

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                self.embed_dim,
                quant_config=quant_config,
                use_attn_tp_group=is_dp_attention_enabled(),
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: SDARBlock(
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
            norm_kwargs = (
                dict(
                    weight_dtype=torch.float32,
                    cast_x_before_out_mul=True,
                    override_orig_dtype=torch.float32,
                    fp32_residual=True,
                )
                if get_global_server_args().rl_on_policy_target is not None
                else {}
            )
            self.norm = RMSNorm(self.embed_dim, eps=config.rms_norm_eps, **norm_kwargs)
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
            hidden_states = (
                self.embed_tokens(input_ids) if input_embeds is None else input_embeds
            )
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors.get("residual", None)

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions, hidden_states, forward_batch, residual
            )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        else:
            if not forward_batch.forward_mode.is_idle():
                hidden_states, residual = self.norm(hidden_states, residual)
            return hidden_states


class SDARForCausalLM(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.pp_group = get_pp_group()
        assert self.pp_group.world_size == 1, (
            f"SDARMoeForCausalLM does not support pipeline parallel (pp_size={self.pp_group.world_size}). "
            "Please set pp_size=1."
        )

        self.config = config
        self.quant_config = quant_config
        alt_stream = torch.cuda.Stream() if _is_cuda else None

        self.model = SDARModel(
            config,
            quant_config=quant_config,
            prefix=add_prefix("model", ""),
            alt_stream=alt_stream,
        )

        if self.pp_group.is_last_rank:
            tp_size = get_tensor_model_parallel_world_size()
            if (
                self.pp_group.world_size == 1
                and config.tie_word_embeddings
                and tp_size == 1
            ):
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
                    prefix=add_prefix("lm_head", prefix),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.use_self_conditioning = (
            float(getattr(config, "self_conditioning_prob", 0.0)) > 0.0
        )
        if self.use_self_conditioning:
            self.self_conditioning = SDARSelfConditioning(config)
        else:
            self.self_conditioning = None

        self.logits_processor = LogitsProcessor(config, return_full_logits=True)

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if (
            input_embeds is None
            and self.use_self_conditioning
            and self.self_conditioning is not None
            and bool(getattr(self.config, "self_conditioning_post_norm", False))
        ):
            input_embeds = self.model.embed_tokens(input_ids)
            input_embeds = self.self_conditioning(
                input_embeds,
                torch.zeros_like(input_embeds),
                post_norm=True,
            )
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        if getattr(self, "_capture_sdar_trace_tensors", False):
            self._last_sdar_hidden_states = hidden_states.detach()
        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch
            )
        else:
            return hidden_states

    @torch.no_grad()
    def build_self_conditioned_input_embeds(
        self,
        input_ids: torch.Tensor,
        prev_logits: Optional[torch.Tensor],
        self_conditioning_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if (
            not self.use_self_conditioning
            or prev_logits is None
            or self.self_conditioning is None
            or not bool(self_conditioning_mask.any())
        ):
            self._last_self_conditioning_stats = None
            if getattr(self, "_capture_sdar_trace_tensors", False):
                self._last_self_conditioning_signal = None
                self._last_self_conditioning_valid = None
                self._last_self_conditioning_signal_is_compact = False
                self._last_self_conditioning_signal_span = None
            return None

        inputs_embeds = self.model.embed_tokens(input_ids)
        selected_indices = torch.nonzero(
            self_conditioning_mask, as_tuple=False
        ).flatten()
        if selected_indices.numel() == 0:
            return inputs_embeds

        direct_capture_span = getattr(
            self,
            "_sdar_direct_self_conditioning_capture_span",
            None,
        )
        direct_capture = (
            getattr(
                self,
                "_capture_sdar_gpu_replay_page",
                False,
            )
            and isinstance(direct_capture_span, (tuple, list))
            and len(direct_capture_span) == 2
        )
        if direct_capture:
            capture_start = int(direct_capture_span[0])
            capture_end = int(direct_capture_span[1])
            direct_capture = (
                0 <= capture_start < capture_end <= int(inputs_embeds.shape[0])
                and bool(
                    (
                        selected_indices.ge(capture_start)
                        & selected_indices.lt(capture_end)
                    ).all()
                )
            )
        signal = torch.zeros_like(inputs_embeds)
        if direct_capture:
            signal_page = torch.zeros(
                (
                    capture_end - capture_start,
                    int(inputs_embeds.shape[-1]),
                ),
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            )
        else:
            signal_page = None

        chunk_size = int(getattr(self.config, "self_conditioning_chunk_size", 256))
        chunk_size = max(1, chunk_size)
        embed_tokens = self.model.embed_tokens
        embed_weight = embed_tokens.weight
        flat_signal = signal.view(-1, signal.size(-1))
        flat_logits = prev_logits.view(-1, prev_logits.size(-1))
        for start in range(0, selected_indices.numel(), chunk_size):
            indices = selected_indices[start : start + chunk_size]
            logits = flat_logits.index_select(0, indices).float()
            probs = F.softmax(logits, dim=-1).to(embed_weight.dtype)
            if probs.size(-1) == embed_weight.size(0):
                soft_embeddings = probs.matmul(embed_weight)
            elif hasattr(embed_tokens, "shard_indices"):
                shard_indices = embed_tokens.shard_indices
                local_probs = probs.new_zeros((probs.size(0), embed_weight.size(0)))
                org_start = shard_indices.org_vocab_start_index
                org_end = shard_indices.org_vocab_end_index
                if org_end > org_start:
                    local_probs[:, : shard_indices.num_org_elements] = probs[:, org_start:org_end]
                added_start = shard_indices.added_vocab_start_index
                added_end = shard_indices.added_vocab_end_index
                if added_end > added_start:
                    local_start = shard_indices.num_org_elements_padded
                    local_end = local_start + shard_indices.num_added_elements
                    local_probs[:, local_start:local_end] = probs[:, added_start:added_end]
                soft_embeddings = local_probs.matmul(embed_weight)
                if getattr(embed_tokens, "tp_size", 1) > 1:
                    if getattr(embed_tokens, "use_attn_tp_group", False):
                        soft_embeddings = attn_tp_all_reduce(soft_embeddings)
                    else:
                        soft_embeddings = tensor_model_parallel_all_reduce(soft_embeddings)
            else:
                raise RuntimeError(
                    "Self-conditioning logits vocab does not match embedding weight: "
                    f"logits_vocab={probs.size(-1)} embed_rows={embed_weight.size(0)}"
                )
            soft_embeddings = soft_embeddings.to(inputs_embeds.dtype)
            flat_signal.index_copy_(0, indices, soft_embeddings)
            if signal_page is not None:
                signal_page.index_copy_(
                    0,
                    indices - capture_start,
                    soft_embeddings,
                )

        post_norm = bool(
            getattr(self, "_force_self_conditioning_post_norm", False)
            or getattr(self.config, "self_conditioning_post_norm", False)
        )
        conditioned = self.self_conditioning(
            inputs_embeds,
            signal.to(inputs_embeds.dtype),
            post_norm=post_norm,
        )
        captured_signal = (
            signal_page
            if signal_page is not None
            else signal
        )
        captured_valid = (
            self_conditioning_mask[capture_start:capture_end]
            if signal_page is not None
            else self_conditioning_mask
        )
        if getattr(self, "_capture_sdar_trace_tensors", False):
            self._last_self_conditioning_signal = captured_signal.detach()
            self._last_self_conditioning_valid = captured_valid.detach()
            self._last_self_conditioning_signal_is_compact = bool(
                signal_page is not None
            )
            self._last_self_conditioning_signal_span = (
                (capture_start, capture_end)
                if signal_page is not None
                else None
            )
        if not bool(getattr(self, "_self_conditioning_stats_enabled", True)):
            self._last_self_conditioning_stats = None
            return conditioned
        with torch.no_grad():
            residual = conditioned - inputs_embeds
            if signal_page is not None:
                signal_square_mean = (
                    signal_page.float().square().sum()
                    / float(max(inputs_embeds.numel(), 1))
                )
                signal_absmax = signal_page.float().abs().max()
            else:
                signal_square_mean = signal.float().square().mean()
                signal_absmax = signal.float().abs().max()
            self._last_self_conditioning_stats = {
                "signal_rms": float(signal_square_mean.sqrt().item()),
                "signal_absmax": float(signal_absmax.item()),
                "residual_rms": float(residual.float().square().mean().sqrt().item()),
                "residual_absmax": float(residual.float().abs().max().item()),
                "residual_to_input_rms": float(
                    residual.float().square().mean().sqrt().item()
                    / (inputs_embeds.float().square().mean().sqrt().item() + 1e-12)
                ),
            }
        return conditioned

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if not name.startswith("model.") and (
                name.startswith("layers.")
                or name.startswith("embed_tokens.")
                or name.startswith("norm.")
            ):
                name = add_prefix(name, "model")

            if name == "model.embed_tokens.weight":
                if (
                    self.pp_group.is_last_rank
                    and self.config.tie_word_embeddings
                    and "lm_head.weight" in params_dict
                ):
                    param = params_dict["lm_head.weight"]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)

            if name.startswith("self_conditioning."):
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
                    logger.info(
                        "Loaded %s shape=%s rms=%.6g absmax=%.6g",
                        name,
                        tuple(loaded_weight.shape),
                        float(loaded_weight.float().square().mean().sqrt().item()),
                        float(loaded_weight.float().abs().max().item()),
                    )
                else:
                    logger.warning(f"Parameter {name} not found in params_dict")
                continue

            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue

            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if name.startswith("model.vision_tower") and name not in params_dict:
                continue
            if "scale" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name in params_dict.keys():
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                else:
                    logger.warning(f"Parameter {name} not found in params_dict")


EntryClass = SDARForCausalLM
