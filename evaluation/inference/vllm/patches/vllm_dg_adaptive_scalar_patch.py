#!/usr/bin/env python3
"""Add the sglang ADs scalar adaptive-temperature controller to vLLM.

The script first applies `vllm_dg_hf_align_patch.py` when needed, then adds
stateful scalar adaptive annealing driven by raw-logit normalized entropy.
Configuration is read from generation_config.json:

    "temperature_mode": "adaptive_scalar",
    "adaptive_power": 0.5

The existing t_min/t_max and sampler_config.entropy_bound fields remain the
temperature endpoints and entropy-bound setting. Scope is unresolved positions,
and reheating is disabled, matching the ADs fanout variant.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


DEFAULT_SOURCE = (
    "/usr/local/lib/python3.12/dist-packages/vllm/"
    "model_executor/models/diffusion_gemma.py"
)
PATH = Path(os.environ.get("DG_VLLM_SOURCE", DEFAULT_SOURCE))
BASE_PATCH = Path(__file__).with_name("vllm_dg_hf_align_patch.py")


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one source match, found {count}")
    return source.replace(old, new)


def apply_base_alignment() -> None:
    source = PATH.read_text()
    if "HF-reference sampling" in source:
        return
    spec = importlib.util.spec_from_file_location("vllm_dg_hf_align_patch", BASE_PATCH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import base patch: {BASE_PATCH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PATH = PATH
    module.main()


def main() -> None:
    apply_base_alignment()
    source = PATH.read_text()

    source = replace_once(
        source,
        """    history: torch.Tensor,  # [max_num_reqs, ST, CL]
    history_len_tensor: torch.Tensor,  # [max_num_reqs]
    # Output tensors (modified in-place)
""",
        """    history: torch.Tensor,  # [max_num_reqs, ST, CL]
    history_len_tensor: torch.Tensor,  # [max_num_reqs]
    adaptive_h_ref: torch.Tensor,  # [max_num_reqs]
    adaptive_t_prev: torch.Tensor,  # [max_num_reqs]
    adaptive_accepted_prev: torch.Tensor,  # [max_num_reqs, CL]
    # Output tensors (modified in-place)
""",
        "compiled adaptive state arguments",
    )

    source = replace_once(
        source,
        """    ST: int,
    # Sampler config
    entropy_bound: float,
) -> torch.Tensor:
""",
        """    ST: int,
    # Sampler config
    entropy_bound: float,
    adaptive_scalar: bool,
    adaptive_power: float,
) -> torch.Tensor:
""",
        "compiled adaptive config arguments",
    )

    source = replace_once(
        source,
        """    # ---- Phase 1: Temperature schedule ----
    steps_f = step_tensor[decode_slots].float()
    remaining = (max_denoising_steps - steps_f).clamp(min=1.0)
    temp = t_min + (t_max - t_min) * (remaining / max_denoising_steps)

    # ---- Phase 2: Temperature scaling + HF-reference sampling ----
    logits_3d = logits.reshape(num_decode, CL, -1).float()
    scaled = logits_3d / temp[:, None, None].clamp(min=1e-10)
""",
        """    is_commit = is_encoder_phase[decode_slots]
    is_denoise = ~is_commit

    # ---- Phase 1: Temperature schedule ----
    steps_f = step_tensor[decode_slots].float()
    remaining = (max_denoising_steps - steps_f).clamp(min=1.0)
    linear_temp = t_min + (t_max - t_min) * (
        remaining / max_denoising_steps
    )

    logits_3d = logits.reshape(num_decode, CL, -1).float()
    if adaptive_scalar:
        raw_log_probs = logits_3d.log_softmax(dim=-1)
        raw_probs = raw_log_probs.exp()
        raw_entropy = -(raw_probs * raw_log_probs).sum(dim=-1)
        raw_entropy = raw_entropy / torch.log(
            logits_3d.new_tensor(float(max(vocab_size, 2)))
        )
        scope = ~adaptive_accepted_prev[decode_slots]
        empty_scope = ~scope.any(dim=-1)
        scope = torch.where(
            empty_scope[:, None], torch.ones_like(scope), scope
        )
        h_scope = (raw_entropy * scope).sum(dim=-1) / scope.sum(
            dim=-1
        ).clamp_min(1)
        first_step = steps_f == 0
        old_h_ref = adaptive_h_ref[decode_slots]
        h_ref = torch.where(first_step, h_scope.clamp_min(1e-8), old_h_ref)
        old_t_prev = adaptive_t_prev[decode_slots]
        t_prev = torch.where(
            first_step, torch.full_like(old_t_prev, t_max), old_t_prev
        )
        ratio = (h_scope / h_ref.clamp_min(1e-8)).clamp(0.0, 1.0)
        proposed = t_min + (t_max - t_min) * ratio.pow(adaptive_power)
        adaptive_temp = torch.where(
            first_step,
            torch.full_like(proposed, t_max),
            torch.minimum(t_prev, proposed),
        ).clamp(t_min, t_max)
        adaptive_h_ref[decode_slots] = torch.where(
            is_denoise, h_ref, torch.zeros_like(h_ref)
        )
        adaptive_t_prev[decode_slots] = torch.where(
            is_denoise, adaptive_temp, torch.full_like(adaptive_temp, t_max)
        )
        temp = adaptive_temp
    else:
        temp = linear_temp

    # ---- Phase 2: Temperature scaling + HF-reference sampling ----
    scaled = logits_3d / temp[:, None, None].clamp(min=1e-10)
""",
        "adaptive temperature calculation",
    )

    source = replace_once(
        source,
        """    # ---- Phase 5: Post-sample ----
    is_commit = is_encoder_phase[decode_slots]  # [num_decode]
    is_denoise = ~is_commit
    cur_step = step_tensor[decode_slots].float()
""",
        """    # ---- Phase 5: Post-sample ----
    if adaptive_scalar:
        adaptive_accepted_prev[decode_slots] = torch.where(
            is_denoise[:, None], eb_mask, torch.zeros_like(eb_mask)
        )
    cur_step = step_tensor[decode_slots].float()
""",
        "adaptive accepted-mask update",
    )

    source = replace_once(
        source,
        """        self.self_conditioning_embeds = torch.zeros(
            max_num_reqs, canvas_length, hidden_size, dtype=torch.float32, device=device
        )

    def init_canvas(self, slot_indices_np: np.ndarray) -> None:
""",
        """        self.self_conditioning_embeds = torch.zeros(
            max_num_reqs, canvas_length, hidden_size, dtype=torch.float32, device=device
        )
        self.adaptive_h_ref = torch.zeros(
            max_num_reqs, dtype=torch.float32, device=device
        )
        self.adaptive_t_prev = torch.zeros(
            max_num_reqs, dtype=torch.float32, device=device
        )
        self.adaptive_accepted_prev = torch.zeros(
            max_num_reqs, canvas_length, dtype=torch.bool, device=device
        )

    def init_canvas(self, slot_indices_np: np.ndarray) -> None:
""",
        "adaptive state buffers",
    )

    source = replace_once(
        source,
        """        self.accepted_canvas_history_len[slot_idx] = 0
        self.self_conditioning_embeds[slot_idx] = 0

    def remove_request(self, slot_idx: int) -> None:
""",
        """        self.accepted_canvas_history_len[slot_idx] = 0
        self.self_conditioning_embeds[slot_idx] = 0
        self.adaptive_h_ref[slot_idx] = 0
        self.adaptive_t_prev[slot_idx] = 0
        self.adaptive_accepted_prev[slot_idx] = False

    def remove_request(self, slot_idx: int) -> None:
""",
        "adaptive request initialization",
    )

    source = replace_once(
        source,
        """        self.accepted_canvas_history_len[slot_idx] = 0
        self.self_conditioning_embeds[slot_idx] = 0


class DiffusionGemmaModelState(ModelState):
""",
        """        self.accepted_canvas_history_len[slot_idx] = 0
        self.self_conditioning_embeds[slot_idx] = 0
        self.adaptive_h_ref[slot_idx] = 0
        self.adaptive_t_prev[slot_idx] = 0
        self.adaptive_accepted_prev[slot_idx] = False


class DiffusionGemmaModelState(ModelState):
""",
        "adaptive request removal",
    )

    source = replace_once(
        source,
        """            entropy_bound=entropy_bound,
            confidence_threshold=gen["confidence_threshold"],
            embed_weight=self.model.model.embed_tokens.weight,
            normalizer=self.model.model.normalizer,
""",
        """            entropy_bound=entropy_bound,
            confidence_threshold=gen["confidence_threshold"],
            adaptive_scalar=gen.get("temperature_mode") == "adaptive_scalar",
            adaptive_power=float(gen.get("adaptive_power", 0.5)),
            embed_weight=self.model.model.embed_tokens.weight,
            normalizer=self.model.model.normalizer,
""",
        "adaptive sampler construction",
    )

    source = replace_once(
        source,
        """        entropy_bound: float,
        embed_weight: torch.Tensor,
        normalizer: torch.Tensor,
""",
        """        entropy_bound: float,
        adaptive_scalar: bool,
        adaptive_power: float,
        embed_weight: torch.Tensor,
        normalizer: torch.Tensor,
""",
        "adaptive sampler init signature",
    )

    source = replace_once(
        source,
        """        self.diffusion_states = diffusion_states
        self.entropy_bound = entropy_bound

        max_num_reqs = diffusion_states.max_num_reqs
""",
        """        self.diffusion_states = diffusion_states
        self.entropy_bound = entropy_bound
        self.adaptive_scalar = adaptive_scalar
        self.adaptive_power = adaptive_power

        max_num_reqs = diffusion_states.max_num_reqs
""",
        "adaptive sampler fields",
    )

    source = replace_once(
        source,
        """            states.accepted_canvas_history,
            states.accepted_canvas_history_len,
            # Output
""",
        """            states.accepted_canvas_history,
            states.accepted_canvas_history_len,
            states.adaptive_h_ref,
            states.adaptive_t_prev,
            states.adaptive_accepted_prev,
            # Output
""",
        "adaptive compiled call state",
    )

    source = replace_once(
        source,
        """            ST=states.stability_threshold,
            entropy_bound=self.entropy_bound,
        )
""",
        """            ST=states.stability_threshold,
            entropy_bound=self.entropy_bound,
            adaptive_scalar=self.adaptive_scalar,
            adaptive_power=self.adaptive_power,
        )
""",
        "adaptive compiled call config",
    )

    PATH.write_text(source)
    print(f"patched scalar adaptive temperature into {PATH}")


if __name__ == "__main__":
    main()
