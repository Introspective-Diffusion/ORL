# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""
Compatibility utilities for different versions of transformers library.
"""

from __future__ import annotations

import importlib.metadata
from functools import lru_cache
from collections.abc import Iterable
from typing import Optional

import torch
from packaging import version
from transformers import PretrainedConfig

# Handle version compatibility for flash_attn_supports_top_left_mask
# This function was added in newer versions of transformers
try:
    from transformers.modeling_flash_attention_utils import flash_attn_supports_top_left_mask
except ImportError:
    # For older versions of transformers that don't have this function
    # Default to False as a safe fallback for older versions
    def flash_attn_supports_top_left_mask():
        """Fallback implementation for older transformers versions.
        Returns False to disable features that require this function.
        """
        return False


def ensure_transformers_runtime_compat() -> None:
    """Restore small compatibility hooks expected by older remote model code."""
    _ensure_legacy_tied_weights_compat()
    _ensure_diffusion_gemma_auto_classes()

    try:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    except Exception:
        return

    def _compute_default_rope_parameters(
        config: Optional[PretrainedConfig] = None,
        device: Optional[torch.device] = None,
        seq_len: Optional[int] = None,
        **_: object,
    ) -> tuple[torch.Tensor, float]:
        del seq_len
        base = float(getattr(config, "rope_theta", 10000.0))
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
        )
        return inv_freq, 1.0

    # Transformers 5.x ships a "default" RoPE initializer, but old SDAR remote
    # code calls the registry through the legacy `(config, device)` signature.
    # The stock initializer can produce invalid inv_freq values for this path,
    # so override it with the legacy-compatible implementation.
    ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters


def _ensure_diffusion_gemma_auto_classes() -> None:
    """Register the vendored DiffusionGemma overlay with HF auto classes."""

    try:
        from transformers import AutoConfig, AutoModel
        from transformers.models.diffusion_gemma.configuration_diffusion_gemma import (
            DiffusionGemmaConfig,
        )
        from transformers.models.diffusion_gemma import (
            modeling_diffusion_gemma as diffusion_gemma_modeling,
            replay_external_kv as diffusion_gemma_external_kv,
        )
        from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
            DiffusionGemmaDecoderModel,
            DiffusionGemmaEncoderModel,
            DiffusionGemmaEncoderTextModel,
            DiffusionGemmaForBlockDiffusion,
        )
    except (ImportError, ModuleNotFoundError):
        return

    AutoConfig.register(
        DiffusionGemmaConfig.model_type,
        DiffusionGemmaConfig,
        exist_ok=True,
    )
    AutoModel.register(
        DiffusionGemmaConfig,
        DiffusionGemmaForBlockDiffusion,
        exist_ok=True,
    )

    def _get_embed_tokens(self):
        return self.embed_tokens

    def _set_embed_tokens(self, value):
        self.embed_tokens = value

    for model_class in (
        DiffusionGemmaEncoderTextModel,
        DiffusionGemmaDecoderModel,
    ):
        if "get_input_embeddings" not in model_class.__dict__:
            model_class.get_input_embeddings = _get_embed_tokens
        if "set_input_embeddings" not in model_class.__dict__:
            model_class.set_input_embeddings = _set_embed_tokens

    def _get_encoder_embeddings(self):
        return self.language_model.embed_tokens

    def _set_encoder_embeddings(self, value):
        self.language_model.embed_tokens = value

    if "get_input_embeddings" not in DiffusionGemmaEncoderModel.__dict__:
        DiffusionGemmaEncoderModel.get_input_embeddings = (
            _get_encoder_embeddings
        )
    if "set_input_embeddings" not in DiffusionGemmaEncoderModel.__dict__:
        DiffusionGemmaEncoderModel.set_input_embeddings = (
            _set_encoder_embeddings
        )

    external_kv_attention = (
        diffusion_gemma_external_kv.diffusion_gemma_external_kv_attention
    )
    if not getattr(
        external_kv_attention,
        "_verl_encoder_passthrough_compat",
        False,
    ):
        def _external_kv_decoder_only(
            module,
            query_states,
            key_states,
            value_states,
            attention_mask,
            past_key_values,
        ):
            # A combined full-gradient replay first runs the causal encoder to
            # produce this cache. Only the subsequent decoder call may enter
            # the external-prefix attention kernel.
            if past_key_values is None:
                return None
            return external_kv_attention(
                module,
                query_states,
                key_states,
                value_states,
                attention_mask,
                past_key_values,
            )

        _external_kv_decoder_only._verl_encoder_passthrough_compat = True
        diffusion_gemma_external_kv.diffusion_gemma_external_kv_attention = (
            _external_kv_decoder_only
        )
        diffusion_gemma_modeling.diffusion_gemma_external_kv_attention = (
            _external_kv_decoder_only
        )


def repair_sdar_rotary_embedding(model: torch.nn.Module) -> None:
    """Repair SDAR RoPE buffers after loading legacy checkpoints.

    Some SDAR checkpoints include an invalid `rotary_emb.inv_freq` buffer under
    Transformers 5.x. Even if the RoPE registry is patched before model
    construction, `from_pretrained` can restore that stale buffer from the
    checkpoint. Recompute it from config after loading.
    """
    config = getattr(model, "config", None)
    model_body = getattr(model, "model", None)
    rotary_emb = getattr(model_body, "rotary_emb", None)
    if config is None or rotary_emb is None or getattr(config, "model_type", None) != "sdar":
        return

    device = getattr(rotary_emb.inv_freq, "device", None)
    base = float(getattr(config, "rope_theta", 10000.0))
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    )
    rotary_emb.inv_freq = inv_freq
    rotary_emb.original_inv_freq = inv_freq


def _ensure_legacy_tied_weights_compat() -> None:
    """Allow legacy SDAR remote code to run with Transformers' dict tied-weight API."""
    try:
        from transformers.modeling_utils import PreTrainedModel
    except Exception:
        return

    if getattr(PreTrainedModel, "_verl_legacy_tied_weights_compat", False):
        return

    original_get_expanded_tied_weights_keys = getattr(PreTrainedModel, "get_expanded_tied_weights_keys", None)
    if original_get_expanded_tied_weights_keys is None:
        return

    def _get_expanded_tied_weights_keys(self, all_submodels: bool = False):
        tied_weights_keys = getattr(self, "_tied_weights_keys", None)
        tied_mapping = _sdar_legacy_tied_weights_mapping(self, tied_weights_keys)
        if tied_mapping is None:
            return original_get_expanded_tied_weights_keys(self, all_submodels=all_submodels)

        original_tied_weights_keys = self._tied_weights_keys
        self._tied_weights_keys = tied_mapping
        try:
            return original_get_expanded_tied_weights_keys(self, all_submodels=all_submodels)
        finally:
            self._tied_weights_keys = original_tied_weights_keys

    PreTrainedModel.get_expanded_tied_weights_keys = _get_expanded_tied_weights_keys
    PreTrainedModel._verl_legacy_tied_weights_compat = True


def _sdar_legacy_tied_weights_mapping(model, tied_weights_keys) -> Optional[dict[str, str]]:
    if isinstance(tied_weights_keys, (str, bytes)) or not isinstance(tied_weights_keys, Iterable):
        return None

    config = getattr(model, "config", None)
    is_sdar_model = model.__class__.__name__ == "SDARForCausalLM" or getattr(config, "model_type", None) == "sdar"
    if not is_sdar_model:
        return None

    if not getattr(config, "tie_word_embeddings", False):
        return {}

    tied_weights_keys = list(tied_weights_keys)
    if tied_weights_keys != ["lm_head.weight"]:
        return None

    input_embeddings = model.get_input_embeddings()
    input_embeddings_name = _module_name(model, input_embeddings)
    if input_embeddings_name is None:
        return None

    return {"lm_head.weight": f"{input_embeddings_name}.weight"}


def _module_name(model, target_module) -> Optional[str]:
    for name, module in model.named_modules(remove_duplicate=False):
        if module is target_module:
            return name
    return None


@lru_cache
def is_transformers_version_in_range(min_version: Optional[str] = None, max_version: Optional[str] = None) -> bool:
    try:
        # Get the installed version of the transformers library
        transformers_version_str = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError as e:
        raise ModuleNotFoundError("The `transformers` package is not installed.") from e

    transformers_version = version.parse(transformers_version_str)

    lower_bound_check = True
    if min_version is not None:
        lower_bound_check = version.parse(min_version) <= transformers_version

    upper_bound_check = True
    if max_version is not None:
        upper_bound_check = transformers_version <= version.parse(max_version)

    return lower_bound_check and upper_bound_check
