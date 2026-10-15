"""SDAR/I-DLM block decode policy aligned with prefix-KV visualization.

This algorithm is intentionally additive: it does not change the existing
LowConfidence or JointThreshold algorithms. It mirrors the knobs used by
``visualization/math500_prefix_kv_commit_sweep.py`` for no-self-conditioning
SDAR checkpoints:

* probability / entropy based acceptance
* mask-token or random-vocab unresolved inputs
* sticky accept or allow accepted tokens to change before block commit
* force-accept-leftmost when a step would otherwise accept nothing
* one final commit forward after the block is filled

Self-conditioning is deliberately not handled here yet.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from contextlib import nullcontext
from typing import List, Optional, Tuple, Union

import torch

_SDAR_LOGITS_SHIFT_ENV = (
    None if __import__("os").environ.get("SGLANG_SDAR_LOGITS_SHIFT") is None
    else __import__("os").environ["SGLANG_SDAR_LOGITS_SHIFT"] == "1"
)
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.adaptive_confidence_temperature import (
    ADAPTIVE_CONFIDENCE_TEMPERATURE_MODES,
    compute_adaptive_confidence_temperature,
    confidence_probs_from_logits,
    summarize_adaptive_confidence_temperature_diagnostics,
    validate_adaptive_confidence_temperature_config,
)
from sglang.srt.dllm.algorithm.adaptive_temperature import (
    ADAPTIVE_TEMPERATURE_MODES,
    compute_adaptive_temperature,
    summarize_adaptive_temperature_diagnostics,
    validate_adaptive_temperature_config,
)
from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.algorithm.sdar_cuda_graph_metrics import SdarCudaGraphMetrics
from sglang.srt.dllm.algorithm.sdar_replay_preselection import (
    DEFAULT_DIRECTION as DEFAULT_REPLAY_PRESELECT_DIRECTION,
    DEFAULT_FINAL_POLICY as DEFAULT_REPLAY_PRESELECT_FINAL_POLICY,
    DEFAULT_GROUP_FIELDS as DEFAULT_REPLAY_PRESELECT_GROUP_FIELDS,
    DEFAULT_SCORE as DEFAULT_REPLAY_PRESELECT_SCORE,
    LEGACY_PRESELECT_MODE,
    RANKED_PRESELECT_MODE,
    ReplayPreselectionPolicy,
    cleanup_discarded_replay_rows,
    select_completed_block_replay_rows,
)
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

_MEMORY_TRACE_TENSOR_STORES = {
    "tensor",
    "memory",
    "inline",
    "none",
    "off",
    "table",
    "memory_table",
    "handle",
    "gpu_table",
    "cuda_ipc",
}


def _transfer_schedule(block_size: int, steps: int) -> torch.Tensor:
    steps = max(int(steps), 1)
    base = block_size // steps
    remainder = block_size % steps
    schedule = torch.zeros(steps, dtype=torch.int64)
    schedule[:] = base
    schedule[:remainder] += 1
    return schedule


# Process-lifetime count of every model forward this algorithm issues; see
# _forward_without_generic_logprob. Mirrored to a file only when the env var is set.
_TRUE_FWD = [0]
_TRUE_FWD_PATH = os.environ.get("SDAR_TRUE_FWD_COUNTER_PATH", "")
# Batched decode: end a request's block after a refine pass that changed nothing
# (same rule as the single-request path). SDAR_BATCH_REFINE_EXIT=0 restores the
# earlier behaviour of refining until post_accept_refine_steps is exhausted.
_BATCH_REFINE_EXIT = os.environ.get("SDAR_BATCH_REFINE_EXIT", "1") != "0"


class SDARPrefixKV(DllmAlgorithm):
    def __init__(self, config: DllmConfig):
        super().__init__(config)
        ac = config.algorithm_config or {}

        self.accept_rule = ac.get("accept_rule", ac.get("confidence_metric", "prob_threshold"))
        if self.accept_rule == "prob":
            self.accept_rule = "prob_threshold"
        elif self.accept_rule == "entropy":
            self.accept_rule = "entropy_threshold"
        elif self.accept_rule == "entropy_budget":
            self.accept_rule = "entropy_bound"
        elif self.accept_rule in {"topk", "topk_schedule"}:
            self.accept_rule = "topk_schedule"
        if self.accept_rule not in {
            "prob_threshold",
            "entropy_threshold",
            "entropy_bound",
            "topk_schedule",
        }:
            raise ValueError(f"Unsupported accept_rule: {self.accept_rule}")

        self.prob_threshold = float(ac.get("prob_threshold", ac.get("threshold", 0.6)))
        self.entropy_threshold = float(ac.get("entropy_threshold", 1.0))
        self.entropy_bound = float(ac.get("entropy_bound", ac.get("eb_threshold", 0.1)))
        self.noise_mode = ac.get("noise_mode", "mask")
        if self.noise_mode not in {"mask", "random_vocab"}:
            raise ValueError(f"Unsupported noise_mode: {self.noise_mode}")
        self.random_token_high = int(ac.get("random_token_high", self.mask_id))
        self.self_conditioning_post_norm = bool(ac.get("self_conditioning_post_norm", False))
        self.self_conditioning_residual_scale = float(
            ac.get("self_conditioning_residual_scale", 1.0)
        )
        self.active_input_mode = str(
            ac.get("active_input_mode", "exact")
        ).strip().lower()
        if self.active_input_mode not in {"exact", "random_active_block"}:
            raise ValueError(
                "active_input_mode must be exact or random_active_block, "
                f"got {self.active_input_mode!r}"
            )

        self.allow_accepted_change = bool(
            ac.get("allow_accepted_change", ac.get("spec_verify", False))
        )
        self.edit_threshold = float(ac.get("edit_threshold", self.prob_threshold))
        # --- EOS edit shield -------------------------------------------------------------
        # Forbid the edit rule from overwriting an already-committed terminal token.
        # Measured motivation (one training run, step 250): 83.8% of committed EOS in non-terminating
        # responses are overwritten by the edit rule, 93% of them by a newline.
        # OFF by default -> the decode path is byte-identical to the pre-patch behaviour.
        #   edit_shield_eos_window <= 0 : permanent protection
        #   edit_shield_eos_window  > 0 : protect only for the first N denoise steps after
        #       commit, then release. The overwrite hazard is front-loaded (median lag 3
        #       steps, mean 6.3), so a short window captures most of the benefit while
        #       leaving the edit rule able to correct a genuinely misplaced EOS later.
        self.edit_shield_eos = bool(ac.get("edit_shield_eos", False))
        self.edit_shield_eos_window = int(ac.get("edit_shield_eos_window", 0))
        self.eos_token_ids: tuple[int, ...] = tuple(
            getattr(config, "eos_token_ids", ()) or ()
        )
        if self.edit_shield_eos and not self.eos_token_ids:
            raise ValueError(
                "edit_shield_eos=true but no EOS token ids reached DllmConfig. Refusing to "
                "run a shield that would silently protect nothing (an inert shield is "
                "indistinguishable from a null result)."
            )
        self._eos_ids_tensor: torch.Tensor | None = None
        self.force_accept_leftmost = bool(ac.get("force_accept_leftmost", True))
        self.max_steps_per_block = int(ac.get("max_steps_per_block", self.block_size))
        self.post_accept_refine_steps = int(ac.get("post_accept_refine_steps", 0))
        # Attention pattern of the per-block commit pass that rewrites the finished
        # block's K/V into the prefix cache.
        #   causal        token-causal within the block (I-DLM idlm_block_generate
        #                 commit_mask; what SDAR-family checkpoints were adapted with)
        #   bidirectional block-causal, i.e. bidirectional within the block -- the
        #                 pattern the training replay kernel uses for clean prefix
        #                 blocks (fast_diffusion_attention ref_mask: k_block <= q_block)
        #                 and what JointThreshold leaves in the cache.
        # Same forward count either way; only the mask differs. Default keeps the
        # historical behaviour.
        #   reuse_refine  no commit forward at all: the last refine pass already ran
        #                 on the final clean tokens with the bidirectional mask and
        #                 wrote their K/V into the cache (JointThreshold semantics);
        #                 its logits are reused for boundary/p_probs. Falls back to a
        #                 bidirectional commit forward when the final forward was not
        #                 clean (post_accept_refine_steps=0, or an edit on the last
        #                 refine step), so the cache is never left dirty.
        self.commit_mode = str(ac.get("commit_mode", "causal")).strip().lower()
        if self.commit_mode not in {"causal", "bidirectional", "reuse_refine"}:
            raise ValueError(
                "commit_mode must be 'causal', 'bidirectional' or 'reuse_refine', "
                f"got {self.commit_mode!r}"
            )
        logger.info("[SDAR-COMMIT] commit_mode=%s", self.commit_mode)
        # Set by the decode loops: did the final forward of the block run on the
        # final token ids (no accept/edit happened after it)? Per-request refine
        # counts are stashed for the honest total_forwards accounting.
        self._last_block_final_forward_clean: bool = False
        self._last_refine_steps_by_batch: list[int] = []
        self.temperature = ac.get("temperature", None)
        self.temperature_max = float(ac.get("temperature_max", 0.8 if self.temperature is None else self.temperature))
        self.temperature_min = float(ac.get("temperature_min", self.temperature_max))
        self.confidence_temperature = float(ac.get("confidence_temperature", 1.0))
        self.confidence_temperature_mode = str(
            ac.get("confidence_temperature_mode", "fixed")
        ).strip().lower()
        _valid_confidence_temperature_modes = {
            "fixed",
            "step_linear",
            *ADAPTIVE_CONFIDENCE_TEMPERATURE_MODES,
        }
        if (
            self.confidence_temperature_mode
            not in _valid_confidence_temperature_modes
        ):
            raise ValueError(
                "confidence_temperature_mode must be one of "
                f"{sorted(_valid_confidence_temperature_modes)}, got "
                f"{self.confidence_temperature_mode!r}"
            )
        self.confidence_temperature_max = float(
            ac.get(
                "confidence_temperature_max",
                self.confidence_temperature,
            )
        )
        self.confidence_temperature_min = float(
            ac.get(
                "confidence_temperature_min",
                self.confidence_temperature_max,
            )
        )
        self.adaptive_confidence_temperature_power = float(
            ac.get("adaptive_confidence_temperature_power", 1.0)
        )
        self.adaptive_confidence_temperature_allow_reheat = bool(
            ac.get(
                "adaptive_confidence_temperature_allow_reheat",
                False,
            )
        )
        # Entropy-gated per-position confidence temperature: cool only
        # positions whose local entropy has fallen; keep ambiguous positions
        # at T_max (decode-side analog of the AEC ceiling).
        self.confidence_temperature_per_position = bool(
            ac.get("confidence_temperature_per_position", False)
        )
        # step_linear: a fixed confidence-temperature schedule instead of the
        # entropy controller. The iteration counter is block-local (restarts
        # every block, keeps counting through post-accept refine passes),
        # iter = 1, 2, ...:
        #     first_step_full=False: T(iter) = max(T_min, T_max - decrement * iter)
        #     first_step_full=True:  T(iter) = max(T_min, T_max - decrement * (iter - 1))
        # With first_step_full the first forward of a block uses T_max. Only
        # the confidence distribution (accept / edit decisions) is affected.
        self.confidence_temperature_step_decrement = float(
            ac.get("confidence_temperature_step_decrement", 0.0)
        )
        self.confidence_temperature_first_step_full = bool(
            ac.get("confidence_temperature_first_step_full", False)
        )
        if self.confidence_temperature_mode == "step_linear":
            if self.confidence_temperature_step_decrement < 0.0:
                raise ValueError("confidence_temperature_step_decrement must be >= 0")
            if self.confidence_temperature_min > self.confidence_temperature_max:
                raise ValueError("confidence_temperature_min must be <= confidence_temperature_max")
        if (
            self.confidence_temperature_mode
            in ADAPTIVE_CONFIDENCE_TEMPERATURE_MODES
        ):
            validate_adaptive_confidence_temperature_config(
                self.confidence_temperature_min,
                self.confidence_temperature_max,
                self.adaptive_confidence_temperature_power,
            )
        # temperature_mode: "fixed" | "linear" | "adaptive_global_entropy" | "adaptive_unresolved_entropy"
        # Backward compat: if temperature is set and mode is unspecified → fixed; else → linear.
        _raw_mode = ac.get("temperature_mode", None)
        if _raw_mode is not None:
            self.temperature_mode: str = str(_raw_mode).strip().lower()
        elif self.temperature is not None:
            self.temperature_mode = "fixed"
        else:
            self.temperature_mode = "linear"
        _valid_modes = {"fixed", "linear", "adaptive_global_entropy", "adaptive_unresolved_entropy"}
        if self.temperature_mode not in _valid_modes:
            raise ValueError(
                f"temperature_mode must be one of {sorted(_valid_modes)}, got {self.temperature_mode!r}"
            )
        self.adaptive_temperature_allow_reheat = bool(
            ac.get("adaptive_temperature_allow_reheat", False)
        )
        self.adaptive_temperature_power = float(
            ac.get("adaptive_temperature_power", 1.0)
        )
        validate_adaptive_temperature_config(
            self.temperature_min,
            self.temperature_max,
            self.adaptive_temperature_power,
        )
        # global AEC: compute H over ALL valid (non-prompt) response positions in addition to
        # the existing unresolved-only scope.
        self.global_aec_enabled: bool = bool(
            ac.get("global_adjacent_entropy_contraction", False)
        )
        # Per-request adaptive temperature state: H_ref (first-step entropy) and T_prev (previous T).
        # Key: req_pool_idx (int). Cleared when a block completes.
        self._adaptive_temp_state_by_req: dict[int, dict] = {}
        # Confidence-temperature adaptation is an independent controller. It
        # must never share H_ref/T_prev state with proposal-temperature modes.
        self._adaptive_confidence_temp_state_by_req: dict[int, dict] = {}
        self.top_k = int(ac.get("top_k", 0))
        self.top_p = float(ac.get("top_p", 1.0))
        self.denoising_steps = int(ac.get("denoising_steps", self.block_size))
        # One-position logits shift, applied at three sites (decode logits,
        # replay logits_source_hidden, commit probs). True is the causal-LM
        # convention the SDAR family was trained under; LLaDA2 is a masked
        # diffusion denoiser whose logits at i predict the token AT i, and the
        # shift silently costs it every point (89.17 -> 0.00, GSM8K n=120).
        # Default True so SDAR arms are bit-unchanged.
        self.logits_shift = bool(ac.get("logits_shift", True))
        if _SDAR_LOGITS_SHIFT_ENV is not None and _SDAR_LOGITS_SHIFT_ENV != self.logits_shift:
            print(
                "[SDARPrefixKV] SGLANG_SDAR_LOGITS_SHIFT overrides the config: "
                f"logits_shift {self.logits_shift} -> {_SDAR_LOGITS_SHIFT_ENV}",
                flush=True,
            )
            self.logits_shift = _SDAR_LOGITS_SHIFT_ENV

        self.adjacent_entropy_contraction: bool = bool(
            ac.get("adjacent_entropy_contraction", False)
        )
        # True vocabulary size for entropy normalization (from hf_config via DllmConfig).
        # Do NOT use random_token_high — it excludes special IDs and is not the true vocab.
        self.aec_vocab_size: int | None = (
            int(config.vocab_size) if getattr(config, "vocab_size", None) is not None else None
        )
        self.deterministic_randtok = bool(ac.get("deterministic_randtok", False))
        env_randtok_seed = os.environ.get(
            "SGLANG_SDAR_RANDTOK_SEED",
            "",
        ).strip()
        self.randtok_seed = int(
            env_randtok_seed
            if env_randtok_seed
            else ac.get("randtok_seed", 0)
        )
        self._transfer_schedule = _transfer_schedule(self.block_size, self.denoising_steps)
        self._boundary_logits_by_req: dict[int, torch.Tensor] = {}
        self._boundary_hidden_by_req: dict[int, torch.Tensor] = {}
        self._selected_logprobs_by_req: dict[int, dict[str, list[float] | list[int]]] = {}
        self._request_stats_by_req: dict[int, list[dict[str, int | float]]] = {}
        self._request_replay_tensors_by_req: dict[int, list[dict]] = {}
        self._request_generated_tokens_by_req: dict[int, int] = {}
        self._stats_lock = threading.Lock()

        self._stats = {
            "total_forwards": 0,
            "denoise_forwards": 0,
            "commit_forwards": 0,
            "accepted_tokens": 0,
            "new_accept_tokens": 0,
            "edit_change_tokens": 0,
            "shield_blocked_tokens": 0,
            "total_tokens": 0,
            "generated_tokens": 0,
            "replay_source_rows": 0,
            "replay_selected_rows": 0,
            "replay_pruned_rows": 0,
            "replay_pruned_gpu_lease_pages": 0,
            "replay_pruned_tensor_files": 0,
            "replay_peak_rows_per_request": 0,
            "replay_capture_source_steps": 0,
            "replay_capture_selected_steps": 0,
            "replay_capture_final_steps": 0,
        }
        # CUDA graph acceleration for the complete SDAR training rollout path.
        # Enabled via sdar_rollout_cuda_graph server arg (preferred) or env var
        # SGLANG_SDAR_ROLLOUT_CUDA_GRAPH=1. When enabled, graph-safe forward passes
        # Trace tensors are returned as stable graph outputs and dynamic
        # self-conditioning embeddings are copied into captured input buffers.
        _cfg_cuda_graph = getattr(config, "sdar_rollout_cuda_graph", False)
        _env_cuda_graph = os.environ.get(
            "SGLANG_SDAR_ROLLOUT_CUDA_GRAPH", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._sdar_cuda_graph_enabled: bool = _cfg_cuda_graph or _env_cuda_graph
        if self._sdar_cuda_graph_enabled:
            logger.info(
                "[SDAR-CudaGraph] CUDA graph acceleration enabled "
                "(cfg=%s env=%s).",
                _cfg_cuda_graph,
                _env_cuda_graph,
            )
        self._cuda_graph_metrics = SdarCudaGraphMetrics()
        # Log every N steps to avoid spam; reduce to 1 for debugging
        self._cuda_graph_log_interval: int = int(
            os.environ.get("SGLANG_SDAR_CUDA_GRAPH_LOG_INTERVAL", "500")
        )
        self._cuda_graph_log_counter: int = 0

        self._trace_dir = os.environ.get("SGLANG_SDAR_TRACE_DIR")
        self._trace_full = os.environ.get("SGLANG_SDAR_TRACE_FULL", "0") == "1"
        self._trace_tensors = os.environ.get("SGLANG_SDAR_TRACE_TENSORS", "0") == "1"
        # 0 means unlimited. Tensor tracing is already opt-in and full replay
        # requires every denoise step, so do not silently sample by default.
        self._trace_tensor_max_records = int(os.environ.get("SGLANG_SDAR_TRACE_TENSOR_MAX_RECORDS", "0"))
        self._trace_tensor_max_tokens = int(os.environ.get("SGLANG_SDAR_TRACE_TENSOR_MAX_TOKENS", str(self.block_size)))
        self._trace_tensor_inline = os.environ.get("SGLANG_SDAR_TRACE_TENSOR_INLINE", "0") == "1"
        self._trace_tensor_store = os.environ.get("SGLANG_SDAR_TRACE_TENSOR_STORE", "file").lower()
        def _replay_preselect_setting(
            env_name: str,
            config_name: str,
            default,
        ):
            if env_name in os.environ:
                return os.environ[env_name]
            return ac.get(config_name, default)

        replay_preselect_value = _replay_preselect_setting(
            "SGLANG_SDAR_REPLAY_PRESELECT_FRACTION",
            "replay_preselect_fraction",
            None,
        )
        replay_preselect_text = (
            str(replay_preselect_value).strip()
            if replay_preselect_value is not None
            else ""
        )
        self._trace_replay_preselect_fraction = (
            float(replay_preselect_text) if replay_preselect_text else None
        )
        self._trace_replay_preselect_mode = str(
            _replay_preselect_setting(
                "SGLANG_SDAR_REPLAY_PRESELECT_MODE",
                "replay_preselect_mode",
                LEGACY_PRESELECT_MODE,
            )
            or LEGACY_PRESELECT_MODE
        ).strip().lower()
        self._trace_replay_preselect_direction = str(
            _replay_preselect_setting(
                "SGLANG_SDAR_REPLAY_PRESELECT_DIRECTION",
                "replay_preselect_direction",
                DEFAULT_REPLAY_PRESELECT_DIRECTION,
            )
            or DEFAULT_REPLAY_PRESELECT_DIRECTION
        ).strip().lower()
        self._trace_replay_preselect_score = str(
            _replay_preselect_setting(
                "SGLANG_SDAR_REPLAY_PRESELECT_SCORE",
                "replay_preselect_score",
                DEFAULT_REPLAY_PRESELECT_SCORE,
            )
            or DEFAULT_REPLAY_PRESELECT_SCORE
        ).strip()
        self._trace_replay_preselect_group_fields = (
            _replay_preselect_setting(
                "SGLANG_SDAR_REPLAY_PRESELECT_GROUP_FIELDS",
                "replay_preselect_group_fields",
                DEFAULT_REPLAY_PRESELECT_GROUP_FIELDS,
            )
        )
        self._trace_replay_preselect_final_policy = str(
            _replay_preselect_setting(
                "SGLANG_SDAR_REPLAY_PRESELECT_FINAL_POLICY",
                "replay_preselect_final_policy",
                DEFAULT_REPLAY_PRESELECT_FINAL_POLICY,
            )
            or DEFAULT_REPLAY_PRESELECT_FINAL_POLICY
        ).strip().lower()
        self._trace_replay_preselect_policy = None
        if self._trace_replay_preselect_fraction is not None:
            self._trace_replay_preselect_policy = (
                ReplayPreselectionPolicy.from_values(
                    fraction=self._trace_replay_preselect_fraction,
                    mode=self._trace_replay_preselect_mode,
                    direction=self._trace_replay_preselect_direction,
                    score=self._trace_replay_preselect_score,
                    group_fields=(
                        self._trace_replay_preselect_group_fields
                    ),
                    final_policy=(
                        self._trace_replay_preselect_final_policy
                    ),
                )
            )
        self._replay_step_sample_fraction = float(
            os.environ.get(
                "SGLANG_SDAR_REPLAY_STEP_SAMPLE_FRACTION",
                "1.0",
            )
        )
        if not 0.0 <= self._replay_step_sample_fraction <= 1.0:
            raise ValueError(
                "SGLANG_SDAR_REPLAY_STEP_SAMPLE_FRACTION must be in [0, 1], "
                f"got {self._replay_step_sample_fraction}"
            )
        if (
            self._trace_replay_preselect_policy is not None
            and self._trace_replay_preselect_policy.mode
            == RANKED_PRESELECT_MODE
        ):
            if self._replay_step_sample_fraction != 1.0:
                raise ValueError(
                    "ranked SDAR replay preselection requires "
                    "SGLANG_SDAR_REPLAY_STEP_SAMPLE_FRACTION=1.0"
                )
            if self._trace_tensor_max_records > 0:
                raise ValueError(
                    "ranked SDAR replay preselection requires "
                    "SGLANG_SDAR_TRACE_TENSOR_MAX_RECORDS=0 so every "
                    "denoise row reaches block ranking"
                )
        self._replay_step_sample_seed = int(
            os.environ.get(
                "SGLANG_SDAR_REPLAY_STEP_SAMPLE_SEED",
                "20260722",
            )
        )
        self._replay_step_sample_mode = os.environ.get(
            "SGLANG_SDAR_REPLAY_STEP_SAMPLE_MODE",
            "bernoulli",
        ).strip().lower()
        if self._replay_step_sample_mode not in {
            "bernoulli",
            "stratified",
        }:
            raise ValueError(
                "SGLANG_SDAR_REPLAY_STEP_SAMPLE_MODE must be bernoulli "
                f"or stratified, got {self._replay_step_sample_mode!r}"
            )
        self._trace_tpf_metadata_first = os.environ.get("SGLANG_SDAR_TPF_METADATA_FIRST", "0").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if self._trace_tpf_metadata_first and self._trace_tensor_store in _MEMORY_TRACE_TENSOR_STORES:
            self._trace_tensor_store = "file"
        self._trace_async_cpu = os.environ.get("SGLANG_SDAR_TRACE_ASYNC_CPU", "1") != "0"
        self._gpu_replay_lease_enabled = os.environ.get(
            "SGLANG_SDAR_GPU_REPLAY_LEASE",
            "0",
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._gpu_replay_lease_table = None
        self._gpu_replay_force_recycle_at_idle_prefill = (
            os.environ.get(
                "SGLANG_SDAR_GPU_REPLAY_FORCE_RECYCLE_AT_IDLE_PREFILL",
                "0",
            ).strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self._gpu_replay_export_logits_source = os.environ.get(
            "SGLANG_SDAR_GPU_REPLAY_EXPORT_LOGITS_SOURCE",
            "1",
        ).strip().lower() in {"1", "true", "yes", "on"}
        if self._gpu_replay_lease_enabled:
            from sglang.srt.dllm.algorithm.sdar_gpu_replay_lease import (
                GpuReplayLeaseTable,
            )

            self._gpu_replay_lease_table = GpuReplayLeaseTable(
                stale_warning_seconds=float(
                    os.environ.get(
                        "SGLANG_SDAR_GPU_REPLAY_STALE_WARNING_S",
                        "300",
                    )
                ),
                max_bytes=int(
                    os.environ.get(
                        "SGLANG_SDAR_GPU_REPLAY_MAX_BYTES",
                        str(16 * 1024**3),
                    )
                ),
            )
            logger.info(
                "[SDAR_GPU_REPLAY] GPU-resident CUDA IPC lease table enabled"
            )
        self._trace_tensor_records = 0
        self._trace_block_idx = {}
        self._trace_preselect_log_counts: dict[int, int] = {}
        logger.info(
            "[QWEN_REPLAY_PRESELECT] init store=%s fraction=%s mode=%s "
            "direction=%s score=%s group_fields=%s final_policy=%s",
            self._trace_tensor_store,
            self._trace_replay_preselect_fraction,
            self._trace_replay_preselect_mode,
            self._trace_replay_preselect_direction,
            self._trace_replay_preselect_score,
            self._trace_replay_preselect_group_fields,
            self._trace_replay_preselect_final_policy,
        )

        # ------------------------------------------------------------------
        # Async capture pipeline (SGLANG_SDAR_ASYNC_CAPTURE=1, default off)
        # ------------------------------------------------------------------
        self._async_capture_enabled = (
            os.environ.get("SGLANG_SDAR_ASYNC_CAPTURE", "0").lower()
            in {"1", "true", "yes", "on"}
        )
        if (
            self._gpu_replay_lease_enabled
            and not self._async_capture_enabled
        ):
            raise RuntimeError(
                "SGLANG_SDAR_GPU_REPLAY_LEASE requires "
                "SGLANG_SDAR_ASYNC_CAPTURE=1"
            )
        self._async_capture_worker = None
        # Protect all stores written by the background consumer and atomically
        # transferred to the scheduler at the rollout boundary.
        self._replay_tensors_lock = threading.Lock()
        self._capture_generation_by_req: dict[int, int] = {}
        self._async_capture_boundary_count = 0
        self._async_capture_metrics_log_every = max(
            1, int(os.environ.get("SGLANG_SDAR_ASYNC_METRICS_LOG_EVERY", "32"))
        )
        self._async_capture_drain_timeout_s = float(
            os.environ.get("SGLANG_SDAR_ASYNC_DRAIN_TIMEOUT_S", "30")
        )
        if self._async_capture_enabled:
            if not self._trace_dir:
                logger.warning(
                    "[SDAR_ASYNC_CAPTURE] SGLANG_SDAR_ASYNC_CAPTURE=1 but "
                    "SGLANG_SDAR_TRACE_DIR is unset; using the synchronous "
                    "non-capture path."
                )
            else:
                from sglang.srt.dllm.algorithm.sdar_async_capture import (
                    AsyncCaptureWorker,
                )

                self._async_capture_worker = AsyncCaptureWorker(
                    algorithm_ref=self,
                    queue_depth=int(
                        os.environ.get("SGLANG_SDAR_ASYNC_QUEUE_DEPTH", "64")
                    ),
                    queue_max_bytes=int(
                        os.environ.get(
                            "SGLANG_SDAR_ASYNC_QUEUE_MAX_BYTES",
                            str(512 * 1024 * 1024),
                        )
                    ),
                    backpressure_policy=os.environ.get(
                        "SGLANG_SDAR_ASYNC_BACKPRESSURE", "error"
                    ).strip().lower(),
                    enqueue_timeout_s=float(
                        os.environ.get(
                            "SGLANG_SDAR_ASYNC_ENQUEUE_TIMEOUT_S", "0"
                        )
                    ),
                )
                logger.info(
                    "[SDAR_ASYNC_CAPTURE] Async capture enabled "
                    "(store=%s trace_tensors=%s trace_dir=%s)",
                    self._trace_tensor_store,
                    self._trace_tensors,
                    self._trace_dir,
                )

    def cleanup_request(self, req_pool_idx: int):
        req_pool_idx = int(req_pool_idx)
        generation = self._capture_generation_for_request(req_pool_idx)
        if self._async_capture_worker is not None:
            self._async_capture_worker.cancel_request(
                req_pool_idx, generation=generation, wait=False
            )
        self._capture_generation_by_req[req_pool_idx] = generation + 1
        self._trace_block_idx.pop(req_pool_idx, None)
        self._boundary_logits_by_req.pop(req_pool_idx, None)
        self._boundary_hidden_by_req.pop(req_pool_idx, None)
        if self._async_capture_worker is not None:
            with self._replay_tensors_lock:
                self._selected_logprobs_by_req.pop(req_pool_idx, None)
                self._request_stats_by_req.pop(req_pool_idx, None)
                self._request_replay_tensors_by_req.pop(req_pool_idx, None)
        else:
            self._selected_logprobs_by_req.pop(req_pool_idx, None)
            self._request_stats_by_req.pop(req_pool_idx, None)
            self._request_replay_tensors_by_req.pop(req_pool_idx, None)
        self._request_generated_tokens_by_req.pop(req_pool_idx, None)
        self._trace_preselect_log_counts.pop(req_pool_idx, None)
        self._clear_adaptive_temp_state(req_pool_idx)
        self._clear_adaptive_confidence_temp_state(req_pool_idx)

    def _eos_hold(self, block_input_ids: torch.Tensor) -> torch.Tensor:
        """Boolean mask: positions currently holding a terminal token."""
        if (
            self._eos_ids_tensor is None
            or self._eos_ids_tensor.device != block_input_ids.device
            or self._eos_ids_tensor.dtype != block_input_ids.dtype
        ):
            self._eos_ids_tensor = torch.tensor(
                self.eos_token_ids,
                dtype=block_input_ids.dtype,
                device=block_input_ids.device,
            )
        return torch.isin(block_input_ids, self._eos_ids_tensor)

    def _apply_edit_shield(
        self,
        edit_mask: torch.Tensor,
        block_input_ids: torch.Tensor,
        eos_commit_step: torch.Tensor | None,
        step: int,
    ) -> tuple[torch.Tensor, int]:
        """Narrow `edit_mask` so committed terminal tokens survive the edit rule.

        `block_input_ids` must still hold the PRE-edit committed tokens, which is the same
        state the caller's `proposal != block_input_ids` clause reads.

        With a positive `edit_shield_eos_window`, protection lapses once a committed EOS has
        survived that many denoise steps, so the edit rule can still repair a genuinely
        misplaced EOS late in the block.

        Returns the narrowed mask and the number of edits it suppressed.
        """
        hold = self._eos_hold(block_input_ids)
        if self.edit_shield_eos_window > 0 and eos_commit_step is not None:
            age = step - eos_commit_step
            hold = hold & (eos_commit_step >= 0) & (age <= self.edit_shield_eos_window)
        blocked = int((edit_mask & hold).sum().item())
        if blocked:
            self._stats["shield_blocked_tokens"] += blocked
        return edit_mask & ~hold, blocked

    def _update_eos_commit_step(
        self,
        eos_commit_step: torch.Tensor,
        block_input_ids: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        """Stamp the step at which each position became EOS; clear it when it stops being EOS.

        Clearing matters: a position that loses EOS and later re-commits one gets a fresh
        protection window rather than inheriting an expired stamp.
        """
        now_eos = self._eos_hold(block_input_ids)
        stamped = torch.where(
            now_eos & (eos_commit_step < 0),
            torch.full_like(eos_commit_step, int(step)),
            eos_commit_step,
        )
        return torch.where(now_eos, stamped, torch.full_like(eos_commit_step, -1))

    @staticmethod
    def _selected_trace_entries(
        mask: torch.Tensor,
        proposal: torch.Tensor,
        raw_logprobs: torch.Tensor,
        sample_logprobs: torch.Tensor,
        proposal_probs: torch.Tensor,
        limit: int = 64,
    ) -> list[dict]:
        positions = torch.nonzero(mask, as_tuple=False).flatten()[:limit]
        entries = []
        for pos in positions.detach().cpu().tolist():
            entries.append(
                {
                    "pos": int(pos),
                    "token_id": int(proposal[pos].detach().cpu().item()),
                    "raw_logprob": float(raw_logprobs[pos].detach().cpu().item()),
                    "sample_logprob": float(sample_logprobs[pos].detach().cpu().item()),
                    "sample_prob": float(proposal_probs[pos].detach().cpu().item()),
                }
            )
        return entries

    @staticmethod
    def _forward_without_generic_logprob(
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ):
        old_return_logprob = getattr(forward_batch, "return_logprob", False)
        # DLLM decode returns an entire block, so SGLang's generic next-token
        # logprob metadata is incomplete. SDARPrefixKV records selected-token
        # logprobs itself and hands them to the output processor.
        forward_batch.return_logprob = False
        try:
            return model_runner.forward(forward_batch, pp_proxy_tensors=None)
        finally:
            forward_batch.return_logprob = old_return_logprob
            # Opt-in true forward counter (SDAR_TRUE_FWD_COUNTER_PATH). Every model
            # forward this algorithm issues -- prefill, denoise, post-accept refine,
            # commit -- passes through here. The per-request `total_forwards`
            # metric is denoise + commit only and does not count refine steps, so
            # an external serial driver reading this file before/after a request
            # gets the honest count (same mechanism as JointThreshold's
            # TPF_COUNTER_PATH). No-op unless the env var is set.
            if _TRUE_FWD_PATH:
                _TRUE_FWD[0] += 1
                try:
                    with open(_TRUE_FWD_PATH + ".tmp", "w") as _f:
                        _f.write("%d\n" % _TRUE_FWD[0])
                    os.replace(_TRUE_FWD_PATH + ".tmp", _TRUE_FWD_PATH)
                except Exception:
                    pass

    def get_stats(self) -> dict:
        with self._stats_lock:
            s = dict(self._stats)
        result = {
            **s,
            "tpf": s["total_tokens"] / max(s["total_forwards"], 1),
            "tokens_per_denoise_forward": s["total_tokens"]
            / max(s["denoise_forwards"], 1),
            "tpf_new_accept": s["new_accept_tokens"]
            / max(s["denoise_forwards"], 1),
            "tpf_returned": s["total_tokens"]
            / max(s["denoise_forwards"], 1),
            "tpf_generated": s["generated_tokens"]
            / max(s["denoise_forwards"], 1),
        }
        if self._sdar_cuda_graph_enabled:
            result.update(self._cuda_graph_metrics.get_stats(synchronize=True))
        return result

    def get_async_capture_metrics(self) -> dict:
        """Return async capture pipeline metrics (empty dict if feature is off)."""
        if self._async_capture_worker is None:
            return {}
        return self._async_capture_worker.get_metrics()

    def _capture_generation_for_request(self, req_pool_idx: int) -> int:
        return self._capture_generation_by_req.setdefault(int(req_pool_idx), 0)

    def _should_capture_replay_step(
        self,
        *,
        req_pool_idx: int,
        generation: int,
        block_idx: int,
        local_step: int,
        is_final_denoise_row: bool,
    ) -> bool:
        from sglang.srt.dllm.algorithm.sdar_async_capture import (
            sample_replay_step,
        )

        selected = sample_replay_step(
            fraction=float(self._replay_step_sample_fraction),
            mode=self._replay_step_sample_mode,
            seed=int(self._replay_step_sample_seed),
            req_pool_idx=req_pool_idx,
            generation=generation,
            block_idx=block_idx,
            local_step=local_step,
            is_final_denoise_row=is_final_denoise_row,
        )
        with self._stats_lock:
            self._stats["replay_capture_source_steps"] += 1
            self._stats["replay_capture_selected_steps"] += int(selected)
            self._stats["replay_capture_final_steps"] += int(
                is_final_denoise_row
            )
        return selected

    def _is_capture_generation_current(
        self, req_pool_idx: int, generation: int
    ) -> bool:
        return self._capture_generation_for_request(req_pool_idx) == int(generation)

    def _record_async_block_counts(
        self, new_accept_tokens: int, edit_change_tokens: int
    ) -> None:
        with self._stats_lock:
            self._stats["accepted_tokens"] += int(new_accept_tokens)
            self._stats["new_accept_tokens"] += int(new_accept_tokens)
            self._stats["edit_change_tokens"] += int(edit_change_tokens)

    def drain_async_capture(
        self, req_pool_idx: int = -1, timeout: float | None = None
    ) -> dict:
        """Drain completed capture work or raise a typed lifecycle error."""
        if self._async_capture_worker is None:
            return {}
        timeout = (
            self._async_capture_drain_timeout_s if timeout is None else float(timeout)
        )
        if req_pool_idx >= 0:
            generation = self._capture_generation_for_request(req_pool_idx)
            return self._async_capture_worker.drain_request(
                req_pool_idx, generation=generation, timeout=timeout
            )
        return self._async_capture_worker.drain_all(timeout=timeout)

    def collect_completed_capture(
        self, req_pool_idx: int, timeout: float | None = None
    ) -> dict:
        """Mandatory scheduler boundary for all rollout capture consumers."""
        req_pool_idx = int(req_pool_idx)
        try:
            lifecycle = self.drain_async_capture(req_pool_idx, timeout=timeout)
        except Exception:
            if self._async_capture_worker is not None:
                self._async_capture_worker.cancel_request(
                    req_pool_idx,
                    generation=self._capture_generation_for_request(req_pool_idx),
                    wait=False,
                )
            with self._replay_tensors_lock:
                self._selected_logprobs_by_req.pop(req_pool_idx, None)
                self._request_stats_by_req.pop(req_pool_idx, None)
                self._request_replay_tensors_by_req.pop(req_pool_idx, None)
            raise
        with self._replay_tensors_lock:
            selected_logprobs = self._selected_logprobs_by_req.pop(
                req_pool_idx, None
            )
            request_stats = self._request_stats_by_req.pop(req_pool_idx, None)
            replay_tensors = self._request_replay_tensors_by_req.pop(
                req_pool_idx, None
            )
        if replay_tensors:
            replay_tensors = self.finalize_async_replay_tensors(replay_tensors)
        if request_stats and lifecycle:
            global_metrics = self.get_async_capture_metrics()
            request_stats[-1]["async_capture_lifecycle"] = {
                **lifecycle,
                "global": global_metrics,
            }
            self._async_capture_boundary_count += 1
            if (
                self._async_capture_boundary_count == 1
                or self._async_capture_boundary_count
                % self._async_capture_metrics_log_every
                == 0
                or global_metrics.get("backpressure_count", 0) > 0
            ):
                logger.info(
                    "[SDAR_ASYNC_CAPTURE_METRICS] boundary=%d req=%d %s",
                    self._async_capture_boundary_count,
                    req_pool_idx,
                    global_metrics,
                )
        if self._async_capture_worker is not None:
            self._async_capture_worker.forget_request(
                req_pool_idx,
                generation=self._capture_generation_for_request(req_pool_idx),
            )
        return {
            "selected_logprobs": selected_logprobs,
            "request_stats": request_stats,
            "replay_tensors": replay_tensors,
            "lifecycle": lifecycle,
        }

    def _determine_denoise_graph_fallback_reason(
        self, forward_batch: ForwardBatch, has_input_embeds: bool
    ) -> str | None:
        """Return the eager fallback reason string, or None if CUDA graph is eligible.

        This method encodes the SDAR graph safety policy:
        - Trace tensors: must fall back because Python-side hidden-state refresh
          (_last_sdar_hidden_states = hidden_states.detach()) is NOT replayed by
          the CUDA graph; replay would leave the tensor stale.
        - Self-conditioning input_embeds: the embeds tensor is dynamically computed
          from previous logits; CUDA graphs are captured without input_embeds, so
          in-flight input_embeds cannot be injected into the graph.
        - disable_cuda_graph_once: another caller explicitly requested eager mode
          for this specific forward (e.g. LM-head debug hooks, LoRA swap).

        Note: batch_size_miss is handled transparently by CudaGraphRunner.can_run();
        it returns False and the model_runner falls back to eager automatically.
        We record it separately by checking graph_runner.can_run() before the forward.
        """
        if not self._sdar_cuda_graph_enabled:
            return None  # metrics disabled; don't report
        if self._trace_tensors:
            return "trace_tensors"
        if has_input_embeds:
            return "self_conditioning"
        if getattr(forward_batch, "disable_cuda_graph_once", False):
            return "disable_once"
        return None

    def _forward_denoise_with_graph_metrics(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ):
        """Run one denoise forward and record the actual replay branch."""
        if not self._sdar_cuda_graph_enabled:
            return self._forward_without_generic_logprob(model_runner, forward_batch)

        start_event = end_event = None
        if forward_batch.input_ids.device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        out = self._forward_without_generic_logprob(model_runner, forward_batch)
        if end_event is not None:
            end_event.record()

        if self._trace_tensors and isinstance(
            out.logits_output.hidden_states, torch.Tensor
        ):
            # The graph runner returns a view of the static captured output.
            # Rebinding after every replay guarantees trace consumers never see
            # a warmup/capture-time tensor.
            model_runner.model._last_sdar_hidden_states = (
                out.logits_output.hidden_states
            )

        self._cuda_graph_metrics.record_forward(
            graph_was_replayed=bool(out.graph_was_replayed),
            miss_reason=out.graph_replay_miss_reason,
            start_event=start_event,
            end_event=end_event,
            num_tokens=int(forward_batch.input_ids.numel()),
        )
        self._cuda_graph_metrics.update_cache_stats(model_runner)

        # Periodic summary log
        self._cuda_graph_log_counter += 1
        if self._cuda_graph_log_counter % self._cuda_graph_log_interval == 0:
            self._cuda_graph_metrics.log_summary(logger)

        return out

    def shutdown_async_capture(self, timeout: float = 5.0) -> bool:
        if self._async_capture_worker is None:
            return True
        return self._async_capture_worker.stop(
            drain=True, join_timeout=float(timeout)
        )

    def _enable_model_tensor_capture(self, model_runner: ModelRunner) -> None:
        if hasattr(model_runner, "model"):
            setattr(model_runner.model, "_capture_sdar_trace_tensors", self._trace_tensors)
            setattr(
                model_runner.model,
                "_capture_sdar_gpu_replay_page",
                bool(self._gpu_replay_lease_enabled),
            )

    @staticmethod
    def _tensor_to_trace_list(tensor: torch.Tensor, max_tokens: int) -> list:
        return tensor[:max_tokens].detach().to(device="cpu", dtype=torch.float32).tolist()

    def _trace_tensor_record_allowed(self) -> bool:
        return (
            self._trace_tensor_max_records <= 0
            or self._trace_tensor_records < self._trace_tensor_max_records
        )

    def _enqueue_step_handle_async(
        self,
        model_runner,
        block_start: int,
        block_end: int,
        prompt_mask: torch.Tensor,
        block_idx: int,
        req_pool_idx: int,
        step: int,
        input_ids_before: torch.Tensor,
        current_ids: torch.Tensor,
        proposal: torch.Tensor,
        proposal_probs: torch.Tensor,
        raw_logprobs: torch.Tensor,
        sample_logprobs: torch.Tensor,
        entropy: torch.Tensor,
        accepted_mask: torch.Tensor,
        new_accept_mask: torch.Tensor,
        edit_mask: torch.Tensor,
        action_mask: torch.Tensor,
        is_final_denoise_row: bool,
        was_completed: bool,
        start_offset: int,
        sc_stats,
    ) -> None:
        """
        Critical-path async capture: snapshot GPU tensors, record a CUDA event,
        and enqueue a lightweight :class:`StepHandle`.  No D2H, no file writes.
        """
        from sglang.srt.dllm.algorithm.sdar_async_capture import StepHandle

        token_limit = min(self.block_size, max(1, self._trace_tensor_max_tokens))
        capture_replay = (
            self._trace_tensors
            and bool(self._trace_dir)
            and self._trace_tensor_record_allowed()
        )
        if capture_replay:
            self._trace_tensor_records += 1
        model = getattr(model_runner, "model", None)
        signal_full = (
            getattr(model, "_last_self_conditioning_signal", None)
            if model is not None and capture_replay
            else None
        )
        valid_full = (
            getattr(model, "_last_self_conditioning_valid", None)
            if model is not None and capture_replay
            else None
        )
        hidden_full = (
            getattr(model, "_last_sdar_hidden_states", None)
            if model is not None and capture_replay
            else None
        )

        sc_signal: Optional[torch.Tensor] = None
        valid_mask: Optional[torch.Tensor] = None
        sc_signal_is_compact = bool(
            model is not None
            and getattr(
                model,
                "_last_self_conditioning_signal_is_compact",
                False,
            )
        )
        if isinstance(signal_full, torch.Tensor):
            if sc_signal_is_compact:
                compact_span = getattr(
                    model,
                    "_last_self_conditioning_signal_span",
                    None,
                )
                if tuple(compact_span or ()) != (
                    int(block_start),
                    int(block_end),
                ):
                    raise RuntimeError(
                        "compact self-conditioning span does not match "
                        f"capture block: compact={compact_span} "
                        f"block={(block_start, block_end)}"
                    )
                sc_signal = signal_full.detach()
            else:
                # `.clone()`, not a bare slice: a view pins the whole signal_full in GPU memory
                # until the handle is dequeued, and estimate_handle_bytes charges by
                # untyped_storage().nbytes(), i.e. for the whole base tensor.
                sc_signal = signal_full[
                    block_start:block_end
                ].detach().clone()
        if isinstance(valid_full, torch.Tensor):
            valid_mask = (
                valid_full.detach()
                if sc_signal_is_compact
                else valid_full[block_start:block_end].detach().clone()
            )

        logits_source_hidden: Optional[torch.Tensor] = None
        if isinstance(hidden_full, torch.Tensor):
            block_hidden = hidden_full[block_start:block_end]
            boundary_h = None
            if block_idx > 0:
                boundary_h = self._boundary_hidden_by_req.get(int(req_pool_idx))
            if boundary_h is not None:
                boundary_h = boundary_h.to(device=block_hidden.device, dtype=block_hidden.dtype)
            else:
                boundary_h = block_hidden[0]
            if self.logits_shift:
                logits_source_hidden = torch.cat(
                    [boundary_h.unsqueeze(0), block_hidden[:-1]], dim=0
                ).detach()
            else:
                logits_source_hidden = block_hidden.detach()

        position_start = int(block_idx) * self.block_size
        step_stride = max(self.max_steps_per_block, 1) + self.block_size
        step_key = int(block_idx) * step_stride + int(step)

        # Record CUDA event on current compute stream for safe D2H in consumer
        cuda_event: Optional[torch.cuda.Event] = None
        if torch.cuda.is_available() and proposal.is_cuda:
            cuda_event = torch.cuda.Event()
            cuda_event.record(torch.cuda.current_stream(proposal.device))

        generation = self._capture_generation_for_request(req_pool_idx)
        handle = StepHandle(
            # accepted_mask is mutated by later denoise iterations.  The other
            # tensors are per-step immutable allocations or an existing snapshot.
            input_ids_snapshot=input_ids_before.detach(),
            current_ids_snapshot=current_ids.detach().clone(),
            proposal=proposal.detach(),
            proposal_probs=proposal_probs.detach(),
            raw_logprobs=raw_logprobs.detach(),
            sample_logprobs=sample_logprobs.detach(),
            entropy=entropy.detach(),
            accepted_mask=accepted_mask.detach().clone(),
            new_accept_mask=new_accept_mask.detach(),
            edit_mask=edit_mask.detach(),
            action_mask=action_mask.detach(),
            prompt_mask=prompt_mask.detach(),
            sc_signal=sc_signal,
            logits_source_hidden=logits_source_hidden,
            valid_mask=valid_mask,
            cuda_event=cuda_event,
            step=int(step),
            block_idx=int(block_idx),
            req_pool_idx=int(req_pool_idx),
            block_start=int(block_start),
            block_end=int(block_end),
            was_completed=bool(was_completed),
            response_start_index=int(self._request_generated_tokens_by_req.get(int(req_pool_idx), 0)),
            start_offset=int(start_offset),
            block_size=int(self.block_size),
            sc_stats=sc_stats,
            trace_full=self._trace_full,
            trace_tensors=self._trace_tensors,
            capture_replay=capture_replay,
            trace_tensor_records=int(self._trace_tensor_records),
            trace_tensor_store=self._trace_tensor_store,
            trace_tensor_max_tokens=int(token_limit),
            position_start=position_start,
            step_key=step_key,
            is_final_denoise_row=bool(is_final_denoise_row),
            enqueue_ns=time.monotonic_ns(),
            generation=generation,
            sc_signal_is_compact=sc_signal_is_compact,
        )
        try:
            self._async_capture_worker.enqueue_step(handle)
        except Exception:
            self._async_capture_worker.cancel_request(
                req_pool_idx, generation=generation, wait=False
            )
            raise

    def _enqueue_block_handle_async(
        self,
        *,
        final_ids: torch.Tensor,
        commit_argmax: torch.Tensor,
        commit_argmax_probs: torch.Tensor,
        p_probs: torch.Tensor,
        q_probs: torch.Tensor,
        q_raw_logprobs: torch.Tensor,
        q_sample_logprobs: torch.Tensor,
        prompt_mask: torch.Tensor,
        req_pool_idx: int,
        block_idx: int,
        block_start: int,
        block_end: int,
        start_offset: int,
        denoise_forwards: int,
        batch_denoise_forwards: int,
        response_clean_start: int,
        adaptive_temperature_diagnostics: list[dict],
        adaptive_confidence_temperature_diagnostics: list[dict],
    ) -> None:
        """Producer-only BlockHandle enqueue with no host materialization."""
        from sglang.srt.dllm.algorithm.sdar_async_capture import BlockHandle

        cuda_event: Optional[torch.cuda.Event] = None
        if torch.cuda.is_available() and final_ids.is_cuda:
            cuda_event = torch.cuda.Event()
            cuda_event.record(torch.cuda.current_stream(final_ids.device))
        generation = self._capture_generation_for_request(req_pool_idx)
        handle = BlockHandle(
            # `final_ids` is a view of forward_batch.input_ids, which later forwards
            # overwrite; the consumer copies it after this point, so snapshot it here.
            final_ids=final_ids.detach().clone(),
            commit_argmax=commit_argmax.detach(),
            commit_argmax_probs=commit_argmax_probs.detach(),
            p_probs=p_probs.detach(),
            q_probs=q_probs.detach(),
            q_raw_logprobs=q_raw_logprobs.detach(),
            q_sample_logprobs=q_sample_logprobs.detach(),
            prompt_mask=prompt_mask.detach(),
            cuda_event=cuda_event,
            req_pool_idx=int(req_pool_idx),
            block_idx=int(block_idx),
            block_start=int(block_start),
            block_end=int(block_end),
            start_offset=int(start_offset),
            denoise_forwards=int(denoise_forwards),
            batch_denoise_forwards=int(batch_denoise_forwards),
            response_clean_start=int(response_clean_start),
            adaptive_temperature_diagnostics=[
                dict(item) for item in adaptive_temperature_diagnostics
            ],
            adaptive_confidence_temperature_diagnostics=[
                dict(item)
                for item in adaptive_confidence_temperature_diagnostics
            ],
            trace_dir=self._trace_dir,
            enqueue_ns=time.monotonic_ns(),
            generation=generation,
        )
        try:
            self._async_capture_worker.enqueue_block(handle)
        except Exception:
            self._async_capture_worker.cancel_request(
                req_pool_idx, generation=generation, wait=False
            )
            raise

    def _tensor_to_cpu(
        self,
        tensor: torch.Tensor,
        token_limit: int,
        async_events: list[torch.cuda.Event] | None = None,
    ) -> torch.Tensor:
        view = tensor[:token_limit].detach()
        if (
            self._trace_async_cpu
            and async_events is not None
            and view.is_cuda
            and torch.cuda.is_available()
        ):
            try:
                cpu_tensor = torch.empty(
                    tuple(view.shape),
                    dtype=view.dtype,
                    device="cpu",
                    pin_memory=True,
                )
                cpu_tensor.copy_(view, non_blocking=True)
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(view.device))
                async_events.append(event)
                return cpu_tensor
            except Exception:
                # Fall back to the original synchronous path if pinned memory or
                # event creation is unavailable in a deployment environment.
                pass
        return view.cpu()

    @staticmethod
    def finalize_async_replay_tensors(replay_tensors):
        """Synchronize internal async CPU copies before replay leaves scheduler."""
        def _finalize(value):
            if isinstance(value, dict):
                events = value.pop("_async_cpu_events", None)
                if events:
                    for event in events:
                        event.synchronize()
                for child in value.values():
                    _finalize(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    _finalize(child)

        _finalize(replay_tensors)
        return replay_tensors

    def _write_replay_tensor_file(
        self,
        payload: dict,
        req_pool_idx: int,
        block_idx: int,
        step: int,
    ) -> str:
        tensor_dir = os.path.join(self._trace_dir, "tensors")
        os.makedirs(tensor_dir, exist_ok=True)
        filename = (
            f"rank0_req{int(req_pool_idx)}_block{int(block_idx)}_"
            f"step{int(step)}_{time.time_ns()}.pt"
        )
        path = os.path.abspath(os.path.join(tensor_dir, filename))
        torch.save(payload, path)
        return path

    def _build_replay_tensor_trace(
        self,
        model_runner: ModelRunner,
        block_start: int,
        block_end: int,
        prompt_mask: torch.Tensor,
        block_idx: int,
        req_pool_idx: int,
        step: int,
        input_ids_before: torch.Tensor | None = None,
        action_ids: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        accepted_mask: torch.Tensor | None = None,
        new_accept_mask: torch.Tensor | None = None,
        edit_change_mask: torch.Tensor | None = None,
        response_start_index: int = 0,
        start_offset: int = 0,
    ) -> dict | None:
        """Write and describe opt-in tensors needed by RL replay."""
        if (
            not self._trace_tensors
            or not self._trace_tensor_record_allowed()
            or not hasattr(model_runner, "model")
            or not self._trace_dir
        ):
            return None

        token_limit = min(self.block_size, max(1, self._trace_tensor_max_tokens))
        model = model_runner.model
        signal_full = getattr(model, "_last_self_conditioning_signal", None)
        valid_full = getattr(model, "_last_self_conditioning_valid", None)
        hidden_full = getattr(model, "_last_sdar_hidden_states", None)

        signal = None
        valid = None
        if isinstance(signal_full, torch.Tensor):
            signal = signal_full[block_start:block_end]
        if isinstance(valid_full, torch.Tensor):
            valid = valid_full[block_start:block_end]

        logits_source_hidden = None
        if isinstance(hidden_full, torch.Tensor):
            block_hidden = hidden_full[block_start:block_end]
            boundary = None
            if block_idx > 0:
                boundary = self._boundary_hidden_by_req.get(int(req_pool_idx))
            if boundary is not None:
                boundary = boundary.to(device=block_hidden.device, dtype=block_hidden.dtype)
            else:
                boundary = block_hidden[0]
            if self.logits_shift:
                logits_source_hidden = torch.cat([boundary.unsqueeze(0), block_hidden[:-1]], dim=0)
            else:
                logits_source_hidden = block_hidden

        if signal is None and logits_source_hidden is None and input_ids_before is None:
            return None
        used_self_conditioning_signal = signal is not None

        if valid is None:
            valid = ~prompt_mask

        async_cpu_events: list[torch.cuda.Event] = []
        action_ids_cpu = None
        action_mask_cpu = None
        new_accept_mask_cpu = None
        edit_change_mask_cpu = None
        replay_active_mask_cpu = None
        random_vocab_mask_cpu = None
        if isinstance(action_ids, torch.Tensor):
            action_ids_cpu = self._tensor_to_cpu(action_ids.to(dtype=torch.long), token_limit, async_cpu_events)
        if isinstance(action_mask, torch.Tensor):
            action_mask_cpu = self._tensor_to_cpu(action_mask.to(dtype=torch.bool), token_limit, async_cpu_events)
        if isinstance(new_accept_mask, torch.Tensor):
            new_accept_mask_bool = new_accept_mask.to(dtype=torch.bool)
            new_accept_mask_cpu = self._tensor_to_cpu(new_accept_mask_bool, token_limit, async_cpu_events)
            replay_active_mask = new_accept_mask_bool
            if isinstance(edit_change_mask, torch.Tensor):
                edit_change_mask_bool = edit_change_mask.to(dtype=torch.bool)
                edit_change_mask_cpu = self._tensor_to_cpu(edit_change_mask_bool, token_limit, async_cpu_events)
                replay_active_mask = replay_active_mask | edit_change_mask_bool
            replay_active_mask_cpu = self._tensor_to_cpu(replay_active_mask, token_limit, async_cpu_events)
        elif isinstance(edit_change_mask, torch.Tensor):
            edit_change_mask_bool = edit_change_mask.to(dtype=torch.bool)
            edit_change_mask_cpu = self._tensor_to_cpu(edit_change_mask_bool, token_limit, async_cpu_events)
            replay_active_mask_cpu = self._tensor_to_cpu(edit_change_mask_bool, token_limit, async_cpu_events)
        if (
            isinstance(accepted_mask, torch.Tensor)
            and isinstance(new_accept_mask, torch.Tensor)
        ):
            accepted_before = (
                accepted_mask.to(dtype=torch.bool)
                & ~new_accept_mask.to(dtype=torch.bool)
            )
            random_vocab_mask = (
                ~accepted_before
                & ~prompt_mask.to(dtype=torch.bool)
            )
            random_vocab_mask_cpu = self._tensor_to_cpu(
                random_vocab_mask,
                token_limit,
                async_cpu_events,
            )

        # Global AEC: mask of ALL valid (non-prompt, non-padding) response positions.
        all_valid_mask_cpu = None
        if self.global_aec_enabled and isinstance(valid, torch.Tensor):
            all_valid_mask = valid.to(dtype=torch.bool) & (~prompt_mask.to(dtype=torch.bool))
            all_valid_mask_cpu = self._tensor_to_cpu(all_valid_mask, token_limit, async_cpu_events)

        hidden_size = None
        dtype = None
        device = None
        if isinstance(signal, torch.Tensor):
            hidden_size = int(signal.shape[-1])
            dtype = signal.dtype
            device = signal.device
        elif isinstance(logits_source_hidden, torch.Tensor):
            hidden_size = int(logits_source_hidden.shape[-1])
            dtype = logits_source_hidden.dtype
            device = logits_source_hidden.device
        if signal is None and hidden_size is not None:
            signal = torch.zeros(
                (self.block_size, hidden_size),
                dtype=dtype,
                device=device,
            )

        self._trace_tensor_records += 1
        position_start = int(block_idx) * self.block_size
        step_stride = max(self.max_steps_per_block, 1) + self.block_size
        step_key = int(block_idx) * step_stride + int(step)
        response_start_index = int(response_start_index)
        start_offset = int(start_offset)
        tensor_payload = {
            "position": (position_start, position_start + self.block_size),
            "valid": self._tensor_to_cpu(valid.to(dtype=torch.bool), token_limit, async_cpu_events),
            "prompt_mask": self._tensor_to_cpu(prompt_mask.to(dtype=torch.bool), token_limit, async_cpu_events),
            "action_ids": action_ids_cpu,
            "action_mask": action_mask_cpu,
            "new_accept_mask": new_accept_mask_cpu,
            "edit_change_mask": edit_change_mask_cpu,
            "replay_active_mask": replay_active_mask_cpu,
            "random_vocab_mask": random_vocab_mask_cpu,
            "all_valid_mask": all_valid_mask_cpu,
            "input_ids": self._tensor_to_cpu(input_ids_before, token_limit, async_cpu_events)
            if isinstance(input_ids_before, torch.Tensor)
            else None,
            "self_conditioning_signal": self._tensor_to_cpu(signal, token_limit, async_cpu_events)
            if isinstance(signal, torch.Tensor)
            else None,
            "logits_source_hidden": self._tensor_to_cpu(logits_source_hidden, token_limit, async_cpu_events)
            if isinstance(logits_source_hidden, torch.Tensor)
            else None,
            "metadata": {
                "format": "sglang_sdar_replay_tensor_v1",
                "step_key": step_key,
                "local_step": int(step),
                "req_pool_idx": int(req_pool_idx),
                "block_idx": int(block_idx),
                "start_offset": start_offset,
                "response_start_index": response_start_index,
                "block_tokens": int(self.block_size),
                "stored_tokens": int(token_limit),
                "truncated": bool(token_limit != self.block_size),
                "new_accept_count": int(new_accept_mask_cpu.sum().item())
                if isinstance(new_accept_mask_cpu, torch.Tensor)
                else 0,
                "edit_change_count": int(
                    edit_change_mask_cpu.sum().item()
                )
                if isinstance(edit_change_mask_cpu, torch.Tensor)
                else 0,
                "action_token_count": int(action_mask_cpu.sum().item())
                if isinstance(action_mask_cpu, torch.Tensor)
                else 0,
                "is_final_denoise_row": False,
                "is_terminal_commit": False,
                "has_self_conditioning_signal": isinstance(signal, torch.Tensor),
                "used_self_conditioning_signal": bool(used_self_conditioning_signal),
                "has_logits_source_hidden": isinstance(logits_source_hidden, torch.Tensor),
            },
        }
        if async_cpu_events:
            tensor_payload["_async_cpu_events"] = async_cpu_events
        tensor_path = None
        tensor_file_bytes = 0
        if self._trace_tensor_store not in _MEMORY_TRACE_TENSOR_STORES:
            self.finalize_async_replay_tensors(tensor_payload)
            tensor_path = self._write_replay_tensor_file(
                tensor_payload,
                req_pool_idx=req_pool_idx,
                block_idx=block_idx,
                step=step,
            )
            try:
                tensor_file_bytes = int(os.path.getsize(tensor_path))
            except OSError:
                tensor_file_bytes = 0
        payload = {
            "format": "sglang_sdar_replay_tensor_v1",
            "step_key": step_key,
            "local_step": int(step),
            "block_idx": int(block_idx),
            "req_pool_idx": int(req_pool_idx),
            "position": [position_start, position_start + self.block_size],
            "start_offset": start_offset,
            "response_start_index": response_start_index,
            "stored_tokens": int(token_limit),
            "block_tokens": int(self.block_size),
            "truncated": bool(token_limit != self.block_size),
            "prompt_mask": [bool(x) for x in prompt_mask[:token_limit].detach().cpu().tolist()],
            "action_ids": (
                [int(x) for x in action_ids[:token_limit].detach().cpu().tolist()]
                if isinstance(action_ids, torch.Tensor)
                else []
            ),
            "action_mask": (
                [bool(x) for x in action_mask[:token_limit].detach().cpu().tolist()]
                if isinstance(action_mask, torch.Tensor)
                else []
            ),
            "new_accept_count": int(new_accept_mask.to(dtype=torch.bool).sum().item())
            if isinstance(new_accept_mask, torch.Tensor)
            else 0,
            "edit_change_count": int(
                edit_change_mask.to(dtype=torch.bool).sum().item()
            )
            if isinstance(edit_change_mask, torch.Tensor)
            else 0,
            "action_token_count": int(action_mask.to(dtype=torch.bool).sum().item())
            if isinstance(action_mask, torch.Tensor)
            else 0,
            "is_final_denoise_row": False,
            "is_terminal_commit": False,
            "new_accept_positions": (
                [
                    int(x)
                    for x in torch.nonzero(new_accept_mask.to(dtype=torch.bool), as_tuple=False)
                    .flatten()[:token_limit]
                    .detach()
                    .cpu()
                    .tolist()
                ]
                if isinstance(new_accept_mask, torch.Tensor)
                else []
            ),
            "edit_change_positions": (
                [
                    int(x)
                    for x in torch.nonzero(edit_change_mask.to(dtype=torch.bool), as_tuple=False)
                    .flatten()[:token_limit]
                    .detach()
                    .cpu()
                    .tolist()
                ]
                if isinstance(edit_change_mask, torch.Tensor)
                else []
            ),
            "has_self_conditioning_signal": signal is not None,
            "used_self_conditioning_signal": bool(used_self_conditioning_signal),
            "has_logits_source_hidden": logits_source_hidden is not None,
            "has_input_ids": isinstance(input_ids_before, torch.Tensor),
        }
        if tensor_path is not None:
            payload["tensor_path"] = tensor_path
            payload["trace_root"] = os.path.abspath(self._trace_dir)
            payload["tensor_file_bytes"] = int(tensor_file_bytes)
        if self._trace_tensor_inline and signal is not None:
            payload["self_conditioning_signal"] = self._tensor_to_trace_list(signal, token_limit)
        if self._trace_tensor_inline and valid is not None:
            payload["valid"] = [bool(x) for x in valid[:token_limit].detach().cpu().tolist()]
        if self._trace_tensor_inline and logits_source_hidden is not None:
            payload["logits_source_hidden"] = self._tensor_to_trace_list(logits_source_hidden, token_limit)
        if tensor_path is not None:
            self._request_replay_tensors_by_req.setdefault(int(req_pool_idx), []).append(dict(payload))
        else:
            self._request_replay_tensors_by_req.setdefault(int(req_pool_idx), []).append(tensor_payload)
        return payload

    def _mark_final_denoise_row(self, req_pool_idx: int, trace_steps: list[dict]) -> None:
        if not trace_steps:
            return
        block_step_keys = set()
        for trace_step in trace_steps:
            trace_step["is_final_denoise_row"] = False
            replay_meta = trace_step.get("replay_tensors")
            if isinstance(replay_meta, dict):
                replay_meta["is_final_denoise_row"] = False
                if replay_meta.get("step_key") is not None:
                    block_step_keys.add(replay_meta["step_key"])
        final_step = trace_steps[-1]
        final_step["is_final_denoise_row"] = True
        replay_meta = final_step.get("replay_tensors")
        if isinstance(replay_meta, dict):
            replay_meta["is_final_denoise_row"] = True
            step_key = replay_meta.get("step_key")
            if step_key is not None:
                for tensor_payload in self._request_replay_tensors_by_req.get(int(req_pool_idx), []):
                    if not isinstance(tensor_payload, dict):
                        continue
                    metadata = tensor_payload.get("metadata") if isinstance(tensor_payload.get("metadata"), dict) else tensor_payload
                    if (
                        isinstance(metadata, dict)
                        and metadata.get("step_key") in block_step_keys
                    ):
                        metadata["is_final_denoise_row"] = False
                        tensor_payload["is_final_denoise_row"] = False
                    if (
                        isinstance(metadata, dict)
                        and metadata.get("step_key") == step_key
                    ):
                        metadata["is_final_denoise_row"] = True
                        tensor_payload["is_final_denoise_row"] = True

    def _attach_block_tpf_summary(
        self,
        req_pool_idx: int,
        block_idx: int,
        *,
        new_accept_tokens: int,
        denoise_forwards: int,
    ) -> None:
        """Attach the canonical block TPF numerator/denominator to replay rows."""

        for tensor_payload in self._request_replay_tensors_by_req.get(int(req_pool_idx), []):
            if not isinstance(tensor_payload, dict):
                continue
            metadata = (
                tensor_payload.get("metadata")
                if isinstance(tensor_payload.get("metadata"), dict)
                else tensor_payload
            )
            if int(metadata.get("block_idx", tensor_payload.get("block_idx", -1))) != int(block_idx):
                continue
            metadata["new_accept_tokens"] = int(new_accept_tokens)
            metadata["denoise_forwards"] = int(denoise_forwards)
            tensor_payload["new_accept_tokens"] = int(new_accept_tokens)
            tensor_payload["denoise_forwards"] = int(denoise_forwards)

    def _preselect_completed_block_replay(
        self,
        req_pool_idx: int,
        block_idx: int,
    ) -> None:
        """Prune one completed block before capture advances to later blocks."""
        fraction = self._trace_replay_preselect_fraction
        if fraction is None:
            return
        policy = getattr(
            self,
            "_trace_replay_preselect_policy",
            None,
        )
        if policy is None:
            # Preserve tests and callers that construct the algorithm without
            # running __init__; the historical default remains earliest.
            policy = ReplayPreselectionPolicy.from_values(
                fraction=fraction,
                mode=getattr(
                    self,
                    "_trace_replay_preselect_mode",
                    LEGACY_PRESELECT_MODE,
                ),
                direction=getattr(
                    self,
                    "_trace_replay_preselect_direction",
                    DEFAULT_REPLAY_PRESELECT_DIRECTION,
                ),
                score=getattr(
                    self,
                    "_trace_replay_preselect_score",
                    DEFAULT_REPLAY_PRESELECT_SCORE,
                ),
                group_fields=getattr(
                    self,
                    "_trace_replay_preselect_group_fields",
                    DEFAULT_REPLAY_PRESELECT_GROUP_FIELDS,
                ),
                final_policy=getattr(
                    self,
                    "_trace_replay_preselect_final_policy",
                    DEFAULT_REPLAY_PRESELECT_FINAL_POLICY,
                ),
            )
        req_pool_idx = int(req_pool_idx)
        block_idx = int(block_idx)
        all_rows = self._request_replay_tensors_by_req.get(req_pool_idx, [])
        indexed_block_rows: list[tuple[int, dict]] = []
        for index, row in enumerate(all_rows):
            if not isinstance(row, dict):
                continue
            metadata = (
                row.get("metadata")
                if isinstance(row.get("metadata"), dict)
                else row
            )
            row_block_idx = int(
                metadata.get(
                    "block_idx",
                    metadata.get("block_index", row.get("block_idx", -1)),
                )
            )
            if row_block_idx == block_idx:
                indexed_block_rows.append((index, row))
        if not indexed_block_rows:
            return

        self.finalize_async_replay_tensors(
            [row for _, row in indexed_block_rows]
        )
        block_rows = [row for _, row in indexed_block_rows]
        result = select_completed_block_replay_rows(
            block_rows,
            policy=policy,
        )
        selected_local_indices = set(result.selected_indices)
        discarded_local_indices = set(result.discarded_indices)
        selected_indices = {
            indexed_block_rows[local_index][0]
            for local_index in selected_local_indices
        }
        block_row_indices = {
            index for index, _ in indexed_block_rows
        }

        self._request_replay_tensors_by_req[req_pool_idx] = [
            row
            for index, row in enumerate(all_rows)
            if index not in block_row_indices
            or index in selected_indices
        ]
        discarded_rows = [
            indexed_block_rows[local_index][1]
            for local_index in sorted(discarded_local_indices)
        ]
        lease_table = getattr(
            self,
            "_gpu_replay_lease_table",
            None,
        )
        recycle_acknowledged = getattr(
            lease_table,
            "recycle_acknowledged",
            None,
        )
        cleanup = cleanup_discarded_replay_rows(
            discarded_rows,
            recycle_acknowledged=(
                recycle_acknowledged
                if callable(recycle_acknowledged)
                else None
            ),
        )
        with getattr(self, "_stats_lock", nullcontext()):
            self._stats["replay_source_rows"] += result.source_rows
            self._stats["replay_selected_rows"] += len(
                result.selected_indices
            )
            self._stats["replay_pruned_rows"] += len(
                result.discarded_indices
            )
            self._stats["replay_pruned_gpu_lease_pages"] = int(
                self._stats.get(
                    "replay_pruned_gpu_lease_pages",
                    0,
                )
            ) + int(cleanup.acknowledged_gpu_lease_pages)
            self._stats["replay_pruned_tensor_files"] = int(
                self._stats.get(
                    "replay_pruned_tensor_files",
                    0,
                )
            ) + int(cleanup.deleted_tensor_files)
            self._stats["replay_peak_rows_per_request"] = max(
                int(self._stats["replay_peak_rows_per_request"]),
                len(all_rows),
            )
        log_count = self._trace_preselect_log_counts.get(req_pool_idx, 0)
        if log_count == 0 or block_idx % 32 == 0:
            logger.info(
                "[QWEN_REPLAY_PRESELECT] req=%d block=%d mode=%s "
                "source=%d selected=%d pruned_leases=%d pruned_files=%d "
                "request_rows_before=%d request_rows_after=%d",
                req_pool_idx,
                block_idx,
                policy.mode,
                result.source_rows,
                len(result.selected_indices),
                cleanup.acknowledged_gpu_lease_pages,
                cleanup.deleted_tensor_files,
                len(all_rows),
                len(self._request_replay_tensors_by_req[req_pool_idx]),
            )
        self._trace_preselect_log_counts[req_pool_idx] = log_count + 1

    def _temperature(self, step: int) -> float:
        """Return sampling temperature for fixed and linear modes.

        For adaptive modes this returns T_max (the caller must use
        _compute_adaptive_temperature() and pass it explicitly).
        """
        if self.temperature_mode == "fixed":
            return float(self.temperature) if self.temperature is not None else self.temperature_max
        # linear (and fallback for adaptive when no per-req state is available)
        if self.max_steps_per_block <= 1:
            return self.temperature_min
        frac = (self.max_steps_per_block - step) / (self.max_steps_per_block - 1)
        return float(
            self.temperature_min + (self.temperature_max - self.temperature_min) * frac
        )

    def _compute_adaptive_temperature(
        self,
        req_pool_idx: int,
        step: int,
        controller_logits: "torch.Tensor",
        prompt_mask: "torch.Tensor",
        accepted_mask: "torch.Tensor",
    ) -> tuple[float, dict]:
        return compute_adaptive_temperature(
            state_by_request=self._adaptive_temp_state_by_req,
            req_pool_idx=req_pool_idx,
            local_step=step,
            controller_logits=controller_logits,
            prompt_mask=prompt_mask,
            accepted_mask=accepted_mask,
            temperature_mode=self.temperature_mode,
            temperature_min=self.temperature_min,
            temperature_max=self.temperature_max,
            power=self.adaptive_temperature_power,
            allow_reheat=self.adaptive_temperature_allow_reheat,
        )

    def _clear_adaptive_temp_state(self, req_pool_idx: int) -> None:
        """Clear per-block adaptive temperature state after block completion."""
        self._adaptive_temp_state_by_req.pop(int(req_pool_idx), None)

    def _compute_adaptive_confidence_temperature(
        self,
        req_pool_idx: int,
        step: int,
        controller_logits: "torch.Tensor",
        prompt_mask: "torch.Tensor",
        accepted_mask: "torch.Tensor",
    ) -> "tuple[float | torch.Tensor, dict]":
        return compute_adaptive_confidence_temperature(
            state_by_request=self._adaptive_confidence_temp_state_by_req,
            req_pool_idx=req_pool_idx,
            local_step=step,
            controller_logits=controller_logits,
            prompt_mask=prompt_mask,
            accepted_mask=accepted_mask,
            confidence_temperature_mode=self.confidence_temperature_mode,
            confidence_temperature_min=self.confidence_temperature_min,
            confidence_temperature_max=self.confidence_temperature_max,
            power=self.adaptive_confidence_temperature_power,
            allow_reheat=(
                self.adaptive_confidence_temperature_allow_reheat
            ),
            per_position=self.confidence_temperature_per_position,
        )

    def _clear_adaptive_confidence_temp_state(
        self, req_pool_idx: int
    ) -> None:
        """Clear independent per-block confidence controller state."""
        self._adaptive_confidence_temp_state_by_req.pop(
            int(req_pool_idx), None
        )

    @staticmethod
    def _complete_adaptive_temperature_metrics(
        metrics: dict | None,
        *,
        prompt_mask: torch.Tensor,
        accepted_mask: torch.Tensor,
        new_accept_mask: torch.Tensor,
        edit_mask: torch.Tensor,
    ) -> dict | None:
        if metrics is None:
            return None
        completed = dict(metrics)
        nonprompt_mask = ~prompt_mask.bool()
        completed.update(
            {
                "accepted_count": int(
                    (accepted_mask.bool() & nonprompt_mask).sum().item()
                ),
                "new_accept_count": int(new_accept_mask.bool().sum().item()),
                "edit_count": int(edit_mask.bool().sum().item()),
            }
        )
        return completed

    @staticmethod
    def _complete_adaptive_confidence_temperature_metrics(
        metrics: dict | None,
        *,
        sampling_temperature: float,
        prompt_mask: torch.Tensor,
        accepted_mask: torch.Tensor,
        new_accept_mask: torch.Tensor,
        edit_mask: torch.Tensor,
    ) -> dict | None:
        if metrics is None:
            return None
        completed = dict(metrics)
        nonprompt_mask = ~prompt_mask.bool()
        completed.update(
            {
                "sampling_temperature": float(sampling_temperature),
                "accepted_count": int(
                    (accepted_mask.bool() & nonprompt_mask).sum().item()
                ),
                "new_accept_count": int(
                    new_accept_mask.bool().sum().item()
                ),
                "edit_count": int(edit_mask.bool().sum().item()),
            }
        )
        return completed

    @staticmethod
    def _top_k_logits(logits: torch.Tensor, top_k: int) -> torch.Tensor:
        if top_k <= 0 or top_k >= logits.size(-1):
            return logits
        kth = torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
        return logits.masked_fill(logits < kth, torch.finfo(logits.dtype).min)

    @staticmethod
    def _top_p_logits(logits: torch.Tensor, top_p: float) -> torch.Tensor:
        if top_p >= 1.0:
            return logits
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative_probs > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, sorted_idx, remove)
        return logits.masked_fill(mask, torch.finfo(logits.dtype).min)

    def _sample_from_logits(
        self,
        logits: torch.Tensor,
        step: int,
        temperature_override: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample from logits with optional temperature override.

        temperature_override: when provided, skips _temperature(step) lookup.
        Used by adaptive temperature modes to pass the pre-computed temperature.
        """
        sample_logits = logits.float()
        sample_logits = self._top_k_logits(sample_logits, self.top_k)
        sample_logits = self._top_p_logits(sample_logits, self.top_p)
        temperature = (
            temperature_override if temperature_override is not None else self._temperature(step)
        )
        if temperature > 0:
            sample_logits = sample_logits / temperature
        probs = F.softmax(sample_logits, dim=-1)
        if temperature > 0:
            proposal = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            proposal = torch.argmax(probs, dim=-1)
        return proposal, probs, sample_logits

    def _request_thresholds(
        self, forward_batch: ForwardBatch, batch_id: int
    ) -> tuple[float, float]:
        """(prob_threshold, edit_threshold) for one request.

        Per-request sampling-params overrides win; otherwise the instance
        defaults from the algorithm yaml apply, so batches without overrides
        stay byte-identical to the pre-patch engine.
        """
        prob_threshold = self.prob_threshold
        edit_threshold = self.edit_threshold
        sampling_info = getattr(forward_batch, "sampling_info", None)
        if sampling_info is None:
            return prob_threshold, edit_threshold
        prob_list = getattr(sampling_info, "dllm_prob_thresholds", None)
        if prob_list is not None and batch_id < len(prob_list):
            value = prob_list[batch_id]
            if value is not None:
                prob_threshold = float(value)
        edit_list = getattr(sampling_info, "dllm_edit_thresholds", None)
        if edit_list is not None and batch_id < len(edit_list):
            value = edit_list[batch_id]
            if value is not None:
                edit_threshold = float(value)
        return prob_threshold, edit_threshold

    def _confidence_probs(
        self,
        logits: torch.Tensor,
        temperature_override: float | None = None,
    ) -> torch.Tensor:
        temperature = (
            self.confidence_temperature
            if temperature_override is None
            else temperature_override
        )
        return confidence_probs_from_logits(logits, temperature)

    def _trace_probability_sample(
        self,
        confidence_probs: torch.Tensor,
        req_pool_idx: int,
        block_idx: int,
        step: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Draw a trace-only sample without perturbing generation RNG."""
        if not (self._trace_dir and self._trace_full):
            return None, None
        generator = torch.Generator(device=confidence_probs.device)
        seed = (
            int(self.randtok_seed) * 1_000_003
            + (int(req_pool_idx) + 1) * 97_409
            + (int(block_idx) + 1) * 10_009
            + int(step)
        ) % (2**63 - 1)
        generator.manual_seed(seed)
        sampled_ids = torch.multinomial(
            confidence_probs,
            num_samples=1,
            generator=generator,
        ).squeeze(-1)
        sampled_probs = confidence_probs.gather(
            -1, sampled_ids.unsqueeze(-1)
        ).squeeze(-1)
        return sampled_ids, sampled_probs

    def _random_unresolved(
        self,
        target: torch.Tensor,
        block_idx: int = 0,
        step_idx: int = 0,
    ) -> torch.Tensor:
        if self.deterministic_randtok:
            pos = (
                torch.arange(target.numel(), device=target.device, dtype=torch.long)
                + int(block_idx) * self.block_size
            )
            return (
                (pos + 1) * 1103515245
                + (int(step_idx) + 1) * 12345
                + self.randtok_seed
            ).remainder(self.random_token_high).to(target.dtype).view_as(target)
        return torch.randint(
            low=0,
            high=self.random_token_high,
            size=target.shape,
            dtype=target.dtype,
            device=target.device,
        )

    def _build_denoise_model_input_ids(
        self,
        logical_input_ids: torch.Tensor,
        request_spans: list[tuple[int, int]],
        prompt_masks: list[torch.Tensor],
        block_indices: list[int],
        step_idx: int,
        active: list[bool] | None = None,
    ) -> torch.Tensor:
        """Build the model-only active-block view without mutating logical state.

        In exact mode returns logical_input_ids unchanged (zero-copy).
        In random_active_block mode every non-prompt position in the active
        block is replaced with deterministic random-vocabulary IDs for the
        model forward; logical state is not mutated.
        """
        if self.active_input_mode == "exact":
            return logical_input_ids

        count = len(request_spans)
        if len(prompt_masks) != count or len(block_indices) != count:
            raise ValueError(
                "request_spans, prompt_masks, and block_indices must have "
                "matching lengths"
            )
        if active is None:
            active = [True] * count
        elif len(active) != count:
            raise ValueError("active must match request_spans")

        model_input_ids = logical_input_ids.clone()
        for batch_id, (block_start, block_end) in enumerate(request_spans):
            if not active[batch_id]:
                continue
            span_len = int(block_end) - int(block_start)
            prompt_mask = prompt_masks[batch_id]
            if span_len < 0 or int(prompt_mask.numel()) != span_len:
                raise ValueError(
                    "prompt mask must exactly cover its active-block span"
                )
            randomize_mask = ~prompt_mask
            if not bool(randomize_mask.any()):
                continue
            logical_span = logical_input_ids[block_start:block_end]
            random_span = self._random_unresolved(
                logical_span,
                block_idx=int(block_indices[batch_id]),
                step_idx=int(step_idx),
            )
            model_span = model_input_ids[block_start:block_end]
            model_span[randomize_mask] = random_span[randomize_mask]
        return model_input_ids

    def _capture_replay_input_ids(
        self,
        model_input_ids: torch.Tensor,
        logical_block_ids: torch.Tensor,
        block_start: int,
        block_end: int,
    ) -> torch.Tensor:
        """Record the exact IDs the model consumed for this denoise forward."""
        if self.active_input_mode == "exact":
            return logical_block_ids.detach().clone()
        return model_input_ids[block_start:block_end].detach().clone()

    def _refresh_unaccepted_inputs(
        self,
        block_input_ids: torch.Tensor,
        accepted_mask: torch.Tensor,
        block_idx: int = 0,
        step_idx: int = 0,
    ) -> None:
        unresolved = ~accepted_mask
        if not bool(unresolved.any()):
            return
        if self.noise_mode == "mask":
            block_input_ids[unresolved] = self.mask_id
        elif self.noise_mode == "random_vocab":
            random_block = self._random_unresolved(
                block_input_ids,
                block_idx=block_idx,
                step_idx=step_idx,
            )
            block_input_ids[unresolved] = random_block[unresolved]
        else:
            raise ValueError(f"Unsupported noise_mode: {self.noise_mode}")

    def _request_input_spans(
        self, forward_batch: ForwardBatch
    ) -> list[tuple[int, int]]:
        batch_size = int(forward_batch.batch_size)
        total_tokens = int(forward_batch.input_ids.shape[0])
        extend_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
        if extend_lens is None:
            seq_lens = getattr(forward_batch, "seq_lens_cpu", None)
            prefix_lens = getattr(forward_batch, "extend_prefix_lens_cpu", None)
            if seq_lens is not None and prefix_lens is not None:
                if isinstance(seq_lens, torch.Tensor):
                    seq_lens = [int(x) for x in seq_lens.detach().cpu().tolist()]
                else:
                    seq_lens = [int(x) for x in seq_lens]
                prefix_lens = [int(x) for x in prefix_lens]
                extend_lens = [
                    seq_len - prefix_len
                    for seq_len, prefix_len in zip(seq_lens, prefix_lens)
                ]
        if extend_lens is None:
            expected = batch_size * self.block_size
            if total_tokens != expected:
                raise RuntimeError(
                    "SDARPrefixKV cannot infer per-request input spans: "
                    f"batch index 0, slice length {total_tokens}, "
                    f"block_size {self.block_size}, batch_size {batch_size}"
                )
            lengths = [self.block_size for _ in range(batch_size)]
        elif isinstance(extend_lens, torch.Tensor):
            lengths = [int(x) for x in extend_lens.detach().cpu().tolist()]
        else:
            lengths = [int(x) for x in extend_lens]

        if len(lengths) != batch_size:
            raise RuntimeError(
                "SDARPrefixKV span metadata length mismatch: "
                f"batch index 0, slice length {len(lengths)}, "
                f"block_size {self.block_size}, batch_size {batch_size}"
            )

        spans = []
        offset = 0
        for batch_id, length in enumerate(lengths):
            if length <= 0:
                raise RuntimeError(
                    "SDARPrefixKV invalid request span: "
                    f"batch index {batch_id}, slice length {length}, "
                    f"block_size {self.block_size}"
                )
            end = offset + length
            if end > total_tokens:
                raise RuntimeError(
                    "SDARPrefixKV request span exceeds input_ids: "
                    f"batch index {batch_id}, slice length {length}, "
                    f"block_size {self.block_size}, total_input_ids {total_tokens}"
                )
            spans.append((offset, end))
            offset = end
        if offset != total_tokens:
            raise RuntimeError(
                "SDARPrefixKV request spans do not cover input_ids: "
                f"batch index {batch_size - 1}, slice length {total_tokens - offset}, "
                f"block_size {self.block_size}, total_input_ids {total_tokens}"
            )
        return spans

    def _check_decode_span(
        self, batch_id: int, block_start: int, block_end: int
    ) -> None:
        slice_len = int(block_end - block_start)
        if slice_len != self.block_size:
            raise RuntimeError(
                "SDARPrefixKV decode span length mismatch: "
                f"batch index {batch_id}, slice length {slice_len}, "
                f"block_size {self.block_size}"
            )

    def _build_accept_mask(
        self,
        eligible_mask: torch.Tensor,
        proposal_probs: torch.Tensor,
        entropy: torch.Tensor,
        step_idx: int,
        prob_threshold: float | None = None,
    ) -> torch.Tensor:
        accept = torch.zeros_like(eligible_mask, dtype=torch.bool)
        if not bool(eligible_mask.any()):
            return accept

        if self.accept_rule == "prob_threshold":
            effective_prob_threshold = (
                self.prob_threshold if prob_threshold is None else prob_threshold
            )
            high_conf = (proposal_probs > effective_prob_threshold) & eligible_mask
            floor = int(self._transfer_schedule[min(step_idx, self.denoising_steps - 1)].item())
            floor = min(max(floor, 1), int(eligible_mask.sum().item()))
            if int(high_conf.sum().item()) >= floor:
                accept = high_conf
            else:
                confidence = torch.where(
                    eligible_mask,
                    proposal_probs,
                    torch.full_like(proposal_probs, -float("inf")),
                )
                accept[torch.topk(confidence, k=floor).indices] = True
        elif self.accept_rule == "topk_schedule":
            floor = int(self._transfer_schedule[min(step_idx, self.denoising_steps - 1)].item())
            floor = min(max(floor, 1), int(eligible_mask.sum().item()))
            confidence = torch.where(
                eligible_mask,
                proposal_probs,
                torch.full_like(proposal_probs, -float("inf")),
            )
            accept[torch.topk(confidence, k=floor).indices] = True
        elif self.accept_rule == "entropy_threshold":
            high_conf = (entropy < self.entropy_threshold) & eligible_mask
            floor = int(self._transfer_schedule[min(step_idx, self.denoising_steps - 1)].item())
            floor = min(max(floor, 1), int(eligible_mask.sum().item()))
            if int(high_conf.sum().item()) >= floor:
                accept = high_conf
            else:
                score = torch.where(
                    eligible_mask,
                    -entropy,
                    torch.full_like(entropy, -float("inf")),
                )
                accept[torch.topk(score, k=floor).indices] = True
        elif self.accept_rule == "entropy_bound":
            eligible_indices = torch.nonzero(eligible_mask, as_tuple=False).flatten()
            eligible_entropy = entropy[eligible_indices]
            sorted_entropy, order = torch.sort(eligible_entropy, descending=False)
            selected_sorted = (
                torch.cumsum(sorted_entropy, dim=-1) - sorted_entropy
            ) <= self.entropy_bound
            selected = eligible_indices[order[selected_sorted]]
            accept[selected] = True
        else:
            raise ValueError(f"Unsupported accept_rule: {self.accept_rule}")

        if not bool(accept.any()) and self.force_accept_leftmost and bool(eligible_mask.any()):
            accept[torch.nonzero(eligible_mask, as_tuple=False).flatten()[0]] = True
        return accept

    def _decode_one_block(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        block_start: int,
        block_end: int,
        prompt_mask: torch.Tensor,
        batch_id: int,
        block_idx: int,
        req_pool_idx: int,
        start_offset: int,
    ) -> tuple[
        LogitsProcessorOutput | torch.Tensor,
        bool,
        int,
        int,
        int,
        list[dict],
        list[dict],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        self._check_decode_span(batch_id, block_start, block_end)
        req_prob_threshold, req_edit_threshold = self._request_thresholds(
            forward_batch, batch_id
        )
        block_input_ids = forward_batch.input_ids[block_start:block_end]
        accepted_mask = prompt_mask.clone()
        self._refresh_unaccepted_inputs(block_input_ids, accepted_mask, block_idx, 0)
        q_probs = torch.full(
            (self.block_size,),
            float("nan"),
            dtype=torch.float32,
            device=block_input_ids.device,
        )
        q_raw_logprobs = torch.full_like(q_probs, float("nan"))
        q_sample_logprobs = torch.full_like(q_probs, float("nan"))

        logits_output = None
        can_run_cuda_graph = False
        denoise_forwards = 0
        new_accept_tokens = 0
        edit_change_tokens = 0
        shield_blocked_tokens = 0
        # Per-position step at which the current EOS was committed (-1 = not EOS).
        # Only needed by the time-limited shield variant; None otherwise.
        eos_commit_step = (
            torch.full(
                (block_input_ids.numel(),),
                -1,
                dtype=torch.long,
                device=block_input_ids.device,
            )
            if (self.edit_shield_eos and self.edit_shield_eos_window > 0)
            else None
        )
        refine_steps = 0
        completed_once = bool(accepted_mask.all())
        trace_steps = []
        adaptive_temperature_diagnostics: list[dict] = []
        adaptive_confidence_temperature_diagnostics: list[dict] = []
        prev_raw_logits = None

        max_iterations = self.max_steps_per_block + self.post_accept_refine_steps
        denoise_step = 0
        changed = False
        for step in range(1, max_iterations + 1):
            if completed_once and refine_steps >= self.post_accept_refine_steps:
                break

            was_completed = completed_once
            accepted_before = accepted_mask.clone()
            old_is_denoise = getattr(forward_batch, "dllm_is_denoise", False)
            old_input_ids = forward_batch.input_ids
            old_input_embeds = getattr(forward_batch, "input_embeds", None)
            old_disable_graph = getattr(forward_batch, "disable_cuda_graph_once", False)
            model_input_ids = self._build_denoise_model_input_ids(
                old_input_ids,
                [(block_start, block_end)],
                [prompt_mask],
                [block_idx],
                step_idx=step,
            )
            forward_batch.dllm_is_denoise = True
            sc_stats = None
            if self._trace_tensors and hasattr(model_runner, "model"):
                model_runner.model._last_self_conditioning_signal = None
                model_runner.model._last_self_conditioning_valid = None
            if (
                prev_raw_logits is not None
                and getattr(model_runner.model, "use_self_conditioning", False)
                and hasattr(model_runner.model, "build_self_conditioned_input_embeds")
            ):
                full_prev_logits = torch.zeros(
                    (
                        model_input_ids.shape[0],
                        prev_raw_logits.shape[-1],
                    ),
                    dtype=prev_raw_logits.dtype,
                    device=prev_raw_logits.device,
                )
                full_prev_logits[block_start:block_end] = prev_raw_logits
                sc_mask = torch.zeros_like(model_input_ids, dtype=torch.bool)
                sc_mask[block_start:block_end] = ~prompt_mask
                old_post_norm = getattr(
                    model_runner.model, "_force_self_conditioning_post_norm", False
                )
                model_runner.model._force_self_conditioning_post_norm = (
                    self.self_conditioning_post_norm
                )
                model_runner.model._sdar_direct_self_conditioning_capture_span = (
                    (int(block_start), int(block_end))
                    if self._gpu_replay_lease_enabled
                    else None
                )
                input_embeds = model_runner.model.build_self_conditioned_input_embeds(
                    model_input_ids,
                    full_prev_logits,
                    sc_mask,
                )
                model_runner.model._force_self_conditioning_post_norm = old_post_norm
                if input_embeds is not None:
                    forward_batch.input_embeds = input_embeds.contiguous()
                    sc_stats = getattr(
                        model_runner.model, "_last_self_conditioning_stats", None
                    )
            forward_batch.input_ids = model_input_ids
            try:
                out = self._forward_denoise_with_graph_metrics(
                    model_runner, forward_batch
                )
                logits_output = out.logits_output
                can_run_cuda_graph = out.graph_was_replayed
            finally:
                forward_batch.dllm_is_denoise = old_is_denoise
                forward_batch.input_ids = old_input_ids
                forward_batch.input_embeds = old_input_embeds
                forward_batch.disable_cuda_graph_once = old_disable_graph
            if not was_completed:
                denoise_forwards += 1

            input_ids_before = self._capture_replay_input_ids(
                model_input_ids,
                block_input_ids,
                block_start,
                block_end,
            )
            raw_logits = logits_output.full_logits[block_start:block_end].float()
            prev_raw_logits = raw_logits.detach()
            boundary = None
            if block_idx > 0:
                boundary = self._boundary_logits_by_req.get(int(req_pool_idx))
            if boundary is not None:
                boundary = boundary.to(device=raw_logits.device, dtype=raw_logits.dtype)
            if boundary is None:
                # Degraded fallback for unusual scheduler states: use in-place
                # logits for position 0 if no prefix boundary has been observed.
                boundary = raw_logits[0]
            if self.logits_shift:
                curr_logits = torch.cat([boundary.unsqueeze(0), raw_logits[:-1]], dim=0)
            else:
                curr_logits = raw_logits
            _step_capped = min(step, self.max_steps_per_block)
            _sample_temp_override = None
            _adaptive_temp_metrics: dict | None = None
            if self.temperature_mode in ADAPTIVE_TEMPERATURE_MODES:
                _sample_temp_override, _adaptive_temp_metrics = self._compute_adaptive_temperature(
                    req_pool_idx, _step_capped, curr_logits, prompt_mask, accepted_mask
                )
            sampling_temperature = float(
                _sample_temp_override
                if _sample_temp_override is not None
                else self._temperature(_step_capped)
            )
            proposal, probs, _ = self._sample_from_logits(
                curr_logits,
                _step_capped,
                temperature_override=_sample_temp_override,
            )
            _confidence_temp_override = None
            _adaptive_confidence_temp_metrics: dict | None = None
            if (
                self.confidence_temperature_mode
                in ADAPTIVE_CONFIDENCE_TEMPERATURE_MODES
            ):
                (
                    _confidence_temp_override,
                    _adaptive_confidence_temp_metrics,
                ) = self._compute_adaptive_confidence_temperature(
                    req_pool_idx,
                    _step_capped,
                    curr_logits,
                    prompt_mask,
                    accepted_before,
                )
            elif self.confidence_temperature_mode == "step_linear":
                _confidence_temp_override = max(
                    self.confidence_temperature_min,
                    self.confidence_temperature_max
                    - self.confidence_temperature_step_decrement
                    * float(
                        max(_step_capped - 1, 0)
                        if self.confidence_temperature_first_step_full
                        else _step_capped
                    ),
                )
            confidence_probs = self._confidence_probs(
                curr_logits,
                temperature_override=_confidence_temp_override,
            )
            diagnostic_sample_ids, diagnostic_sample_probs = (
                self._trace_probability_sample(
                    confidence_probs,
                    req_pool_idx,
                    block_idx,
                    step,
                )
            )
            proposal_probs = confidence_probs.gather(-1, proposal.unsqueeze(-1)).squeeze(-1)
            raw_logprobs = F.log_softmax(curr_logits.float(), dim=-1).gather(
                -1, proposal.unsqueeze(-1)
            ).squeeze(-1)
            sample_logprobs = proposal_probs.clamp_min(
                torch.finfo(proposal_probs.dtype).tiny
            ).log()
            entropy = -(
                confidence_probs * confidence_probs.clamp_min(torch.finfo(confidence_probs.dtype).tiny).log()
            ).sum(dim=-1)

            changed = False
            edit_mask = torch.zeros_like(accepted_mask)
            step_shield_blocked = 0
            unresolved_mask = (~accepted_mask) & (~prompt_mask)
            if not was_completed:
                accept_mask = self._build_accept_mask(
                    unresolved_mask,
                    proposal_probs,
                    entropy,
                    denoise_step,
                    prob_threshold=req_prob_threshold,
                )
                if bool(accept_mask.any()):
                    block_input_ids[accept_mask] = proposal[accept_mask].to(block_input_ids.dtype)
                    q_probs[accept_mask] = proposal_probs[accept_mask].float()
                    q_raw_logprobs[accept_mask] = raw_logprobs[accept_mask].float()
                    q_sample_logprobs[accept_mask] = sample_logprobs[accept_mask].float()
                    accepted_mask |= accept_mask
                    changed = True
                denoise_step += 1
            else:
                refine_steps += 1

            if self.allow_accepted_change:
                edit_mask = (
                    accepted_before
                    & (~prompt_mask)
                    & (proposal_probs > req_edit_threshold)
                    & (proposal.to(block_input_ids.dtype) != block_input_ids)
                )
                if self.edit_shield_eos:
                    edit_mask, step_shield_blocked = self._apply_edit_shield(
                        edit_mask, block_input_ids, eos_commit_step, step
                    )
                    shield_blocked_tokens += step_shield_blocked
                if bool(edit_mask.any()):
                    block_input_ids[edit_mask] = proposal[edit_mask].to(block_input_ids.dtype)
                    q_probs[edit_mask] = proposal_probs[edit_mask].float()
                    q_raw_logprobs[edit_mask] = raw_logprobs[edit_mask].float()
                    q_sample_logprobs[edit_mask] = sample_logprobs[edit_mask].float()
                    changed = True

            if eos_commit_step is not None:
                eos_commit_step = self._update_eos_commit_step(
                    eos_commit_step, block_input_ids, step
                )

            new_accept_mask = accepted_mask & ~accepted_before
            action_mask = (new_accept_mask | edit_mask) & (~prompt_mask)
            block_completed_now = bool(accepted_mask.all())
            _adaptive_temp_metrics = self._complete_adaptive_temperature_metrics(
                _adaptive_temp_metrics,
                prompt_mask=prompt_mask,
                accepted_mask=accepted_mask,
                new_accept_mask=new_accept_mask,
                edit_mask=edit_mask,
            )
            if _adaptive_temp_metrics is not None:
                adaptive_temperature_diagnostics.append(_adaptive_temp_metrics)
            _adaptive_confidence_temp_metrics = (
                self._complete_adaptive_confidence_temperature_metrics(
                    _adaptive_confidence_temp_metrics,
                    sampling_temperature=sampling_temperature,
                    prompt_mask=prompt_mask,
                    accepted_mask=accepted_mask,
                    new_accept_mask=new_accept_mask,
                    edit_mask=edit_mask,
                )
            )
            if _adaptive_confidence_temp_metrics is not None:
                adaptive_confidence_temperature_diagnostics.append(
                    _adaptive_confidence_temp_metrics
                )
            if self._async_capture_worker is None:
                # Sync path: compute counts needed for trace record and stats
                new_accept_count = int(new_accept_mask.sum().item())
                edit_change_count = int(edit_mask.sum().item())
                new_accept_tokens += new_accept_count
                edit_change_tokens += edit_change_count
            # In async path counts are computed by consumer from GPU tensors

            if self._trace_dir:
                if self._async_capture_worker is not None:
                    # Async path: enqueue lightweight handle; no D2H here
                    self._enqueue_step_handle_async(
                        model_runner=model_runner,
                        block_start=block_start,
                        block_end=block_end,
                        prompt_mask=prompt_mask,
                        block_idx=block_idx,
                        req_pool_idx=req_pool_idx,
                        step=step,
                        input_ids_before=input_ids_before,
                        current_ids=block_input_ids,
                        proposal=proposal,
                        proposal_probs=proposal_probs,
                        raw_logprobs=raw_logprobs,
                        sample_logprobs=sample_logprobs,
                        entropy=entropy,
                        accepted_mask=accepted_mask,
                        new_accept_mask=new_accept_mask,
                        edit_mask=edit_mask,
                        action_mask=action_mask,
                        is_final_denoise_row=block_completed_now,
                        was_completed=bool(was_completed),
                        start_offset=start_offset,
                        sc_stats=sc_stats,
                    )
                else:
                    # Sync path: build trace_step dict inline (existing behavior)
                    new_accept_count_local = int(new_accept_mask.sum().item()) if self._async_capture_worker is None else 0
                    edit_change_count_local = int(edit_mask.sum().item()) if self._async_capture_worker is None else 0
                    trace_step = {
                        "trace_tensors_enabled": bool(self._trace_tensors),
                        "trace_tensor_records": int(self._trace_tensor_records),
                        "trace_dir_set": bool(self._trace_dir),
                        "step": step,
                        "was_completed": bool(was_completed),
                        "accepted_count": int(accepted_mask.sum().item()),
                        "new_accept_count": new_accept_count_local,
                        "edit_change_count": edit_change_count_local,
                        "shield_blocked_count": int(step_shield_blocked),
                        "action_token_count": int(action_mask.sum().item()),
                        "accepted_positions": [
                            int(x)
                            for x in torch.nonzero(accepted_mask, as_tuple=False)
                            .flatten()[:64]
                            .detach()
                            .cpu()
                            .tolist()
                        ],
                        "new_accept_positions": [
                            int(x)
                            for x in torch.nonzero(accepted_mask & ~accepted_before, as_tuple=False)
                            .flatten()[:64]
                            .detach()
                            .cpu()
                            .tolist()
                        ],
                        "edit_change_positions": [
                            int(x)
                            for x in torch.nonzero(edit_mask, as_tuple=False)
                            .flatten()[:64]
                            .detach()
                            .cpu()
                            .tolist()
                        ],
                        "action_positions": [
                            int(x)
                            for x in torch.nonzero(action_mask, as_tuple=False)
                            .flatten()[:64]
                            .detach()
                            .cpu()
                            .tolist()
                        ],
                        "action_token_ids": [
                            int(proposal[int(x)].detach().cpu().item())
                            for x in torch.nonzero(action_mask, as_tuple=False)
                            .flatten()[:64]
                            .detach()
                            .cpu()
                            .tolist()
                        ],
                        "new_accept_token_logprobs": self._selected_trace_entries(
                            new_accept_mask,
                            proposal,
                            raw_logprobs,
                            sample_logprobs,
                            proposal_probs,
                        ),
                        "edit_token_logprobs": self._selected_trace_entries(
                            edit_mask,
                            proposal,
                            raw_logprobs,
                            sample_logprobs,
                            proposal_probs,
                        ),
                        "max_prob_mean": float(proposal_probs.mean().item()),
                        "entropy_mean": float(entropy.mean().item()),
                        "head_tokens": [int(x) for x in block_input_ids[:16].detach().cpu().tolist()],
                        "proposal_head": [int(x) for x in proposal[:32].detach().cpu().tolist()],
                        "maxp_head": [
                            float(x)
                            for x in proposal_probs[:16].detach().cpu().tolist()
                        ],
                        "self_conditioning": sc_stats,
                    }
                    if _adaptive_temp_metrics is not None:
                        trace_step["adaptive_temperature"] = dict(
                            _adaptive_temp_metrics
                        )
                    if _adaptive_confidence_temp_metrics is not None:
                        trace_step[
                            "adaptive_confidence_temperature"
                        ] = dict(_adaptive_confidence_temp_metrics)
                    replay_tensors = self._build_replay_tensor_trace(
                        model_runner,
                        block_start,
                        block_end,
                        prompt_mask,
                        block_idx,
                        req_pool_idx,
                        step,
                        input_ids_before=input_ids_before,
                        action_ids=proposal,
                        action_mask=action_mask,
                        accepted_mask=accepted_mask,
                        new_accept_mask=new_accept_mask,
                        edit_change_mask=edit_mask,
                        response_start_index=self._request_generated_tokens_by_req.get(int(req_pool_idx), 0),
                        start_offset=int(prompt_mask.sum().item()),
                    )
                    if replay_tensors is not None:
                        trace_step["replay_tensors"] = replay_tensors
                    if self._trace_full:
                        trace_step.update(
                            {
                                "current_ids": [
                                    int(x) for x in block_input_ids.detach().cpu().tolist()
                                ],
                                "proposal_ids": [
                                    int(x) for x in proposal.detach().cpu().tolist()
                                ],
                                "diagnostic_sample_ids": (
                                    [
                                        int(x)
                                        for x in diagnostic_sample_ids.detach()
                                        .cpu()
                                        .tolist()
                                    ]
                                    if diagnostic_sample_ids is not None
                                    else []
                                ),
                                "diagnostic_sample_probs": (
                                    [
                                        float(x)
                                        for x in diagnostic_sample_probs.detach()
                                        .cpu()
                                        .tolist()
                                    ]
                                    if diagnostic_sample_probs is not None
                                    else []
                                ),
                                "action_mask": [
                                    bool(x) for x in action_mask.detach().cpu().tolist()
                                ],
                                "accepted_mask": [
                                    bool(x) for x in accepted_mask.detach().cpu().tolist()
                                ],
                                "new_accept_mask": [
                                    bool(x)
                                    for x in (accepted_mask & ~accepted_before)
                                    .detach()
                                    .cpu()
                                    .tolist()
                                ],
                                "edit_mask": [
                                    bool(x) for x in edit_mask.detach().cpu().tolist()
                                ],
                                "proposal_probs": [
                                    float(x)
                                    for x in proposal_probs.detach().cpu().tolist()
                                ],
                            }
                        )
                    trace_steps.append(trace_step)

            if block_completed_now:
                completed_once = True
                if not self.allow_accepted_change or self.post_accept_refine_steps <= 0:
                    break

            self._refresh_unaccepted_inputs(
                block_input_ids,
                accepted_mask,
                block_idx,
                denoise_step,
            )
            if was_completed and not changed:
                break

        # The K/V and logits left behind by the last forward describe the final
        # token ids only if that forward saw them, i.e. nothing was accepted or
        # edited after it. True on the normal exit above; false when the block
        # completed with post_accept_refine_steps<=0 or the refine cap cut off an
        # edit. commit_mode=reuse_refine keys off this.
        self._last_block_final_forward_clean = bool(completed_once and not changed)
        self._last_refine_steps_by_batch = [int(refine_steps)]

        self._mark_final_denoise_row(req_pool_idx, trace_steps)
        # Clear adaptive temperature state for this request after block completes.
        self._clear_adaptive_temp_state(req_pool_idx)
        self._clear_adaptive_confidence_temp_state(req_pool_idx)
        return (
            logits_output,
            can_run_cuda_graph,
            denoise_forwards,
            new_accept_tokens,
            edit_change_tokens,
            trace_steps,
            adaptive_temperature_diagnostics,
            adaptive_confidence_temperature_diagnostics,
            q_probs,
            q_raw_logprobs,
            q_sample_logprobs,
        )

    def _decode_batch_blocks(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        request_spans: list[tuple[int, int]],
        decode_block_mask: list[bool],
        prompt_masks: list[torch.Tensor],
        start_offsets: list[int],
        block_indices: list[int],
        req_pool_indices: list[int],
    ) -> tuple:
        batch_size = forward_batch.batch_size
        request_thresholds = [
            self._request_thresholds(forward_batch, b) for b in range(batch_size)
        ]
        block_inputs = []
        accepted_masks = []
        for batch_id in range(batch_size):
            block_start, block_end = request_spans[batch_id]
            if not decode_block_mask[batch_id]:
                block_inputs.append(None)
                accepted_masks.append(prompt_masks[batch_id].clone())
                continue
            self._check_decode_span(batch_id, block_start, block_end)
            block_input_ids = forward_batch.input_ids[block_start:block_end]
            accepted_mask = prompt_masks[batch_id].clone()
            self._refresh_unaccepted_inputs(
                block_input_ids,
                accepted_mask,
                block_indices[batch_id],
                0,
            )
            block_inputs.append(block_input_ids)
            accepted_masks.append(accepted_mask)

        completed_once = [
            (not decode_block_mask[i]) or bool(mask.all())
            for i, mask in enumerate(accepted_masks)
        ]
        q_probs_by_batch = [
            torch.full(
                (self.block_size,),
                float("nan"),
                dtype=torch.float32,
                device=forward_batch.input_ids.device,
            )
            for _ in range(batch_size)
        ]
        q_raw_logprobs_by_batch = [torch.full_like(q_probs, float("nan")) for q_probs in q_probs_by_batch]
        # Per-position EOS commit step, one tensor per batch entry. Time-limited shield only.
        eos_commit_step_by_batch = [
            (
                torch.full(
                    (self.block_size,),
                    -1,
                    dtype=torch.long,
                    device=forward_batch.input_ids.device,
                )
                if (self.edit_shield_eos and self.edit_shield_eos_window > 0)
                else None
            )
            for _ in range(batch_size)
        ]
        q_sample_logprobs_by_batch = [torch.full_like(q_probs, float("nan")) for q_probs in q_probs_by_batch]
        refine_steps = [0 for _ in range(batch_size)]
        denoise_steps = [0 for _ in range(batch_size)]
        prev_raw_logits: list[torch.Tensor | None] = [None for _ in range(batch_size)]
        trace_steps_by_batch: list[list[dict]] = [[] for _ in range(batch_size)]
        adaptive_temperature_diagnostics_by_batch: list[list[dict]] = [
            [] for _ in range(batch_size)
        ]
        adaptive_confidence_temperature_diagnostics_by_batch: list[
            list[dict]
        ] = [[] for _ in range(batch_size)]
        new_accept_tokens_by_batch = [0 for _ in range(batch_size)]
        edit_change_tokens_by_batch = [0 for _ in range(batch_size)]
        logits_output = None
        can_run_cuda_graph = False
        denoise_forwards = 0
        max_iterations = self.max_steps_per_block + self.post_accept_refine_steps

        # Did any request accept or edit a token in the most recent forward? The
        # whole batch is forwarded every iteration, so when this is False at loop
        # exit the cache and logits of the last forward describe every request's
        # final tokens (see commit_mode=reuse_refine).
        last_iter_changed = False
        refine_done = [False for _ in range(batch_size)]
        for step in range(1, max_iterations + 1):
            active = [
                decode_block_mask[i]
                and not refine_done[i]
                and not (
                    completed_once[i]
                    and refine_steps[i] >= self.post_accept_refine_steps
                )
                for i in range(batch_size)
            ]
            if not any(active):
                break
            last_iter_changed = False

            accepted_before = [mask.clone() for mask in accepted_masks]
            was_completed = list(completed_once)
            old_is_denoise = getattr(forward_batch, "dllm_is_denoise", False)
            old_input_ids = forward_batch.input_ids
            old_input_embeds = getattr(forward_batch, "input_embeds", None)
            old_disable_graph = getattr(forward_batch, "disable_cuda_graph_once", False)
            model_input_ids = self._build_denoise_model_input_ids(
                old_input_ids,
                request_spans,
                prompt_masks,
                block_indices,
                step_idx=step,
                active=active,
            )
            forward_batch.dllm_is_denoise = True
            sc_stats_by_batch = [None for _ in range(batch_size)]
            if self._trace_tensors and hasattr(model_runner, "model"):
                model_runner.model._last_self_conditioning_signal = None
                model_runner.model._last_self_conditioning_valid = None

            if (
                any(x is not None for x in prev_raw_logits)
                and getattr(model_runner.model, "use_self_conditioning", False)
                and hasattr(model_runner.model, "build_self_conditioned_input_embeds")
            ):
                sample_logits = next(x for x in prev_raw_logits if x is not None)
                full_prev_logits = torch.zeros(
                    (
                        model_input_ids.shape[0],
                        sample_logits.shape[-1],
                    ),
                    dtype=sample_logits.dtype,
                    device=sample_logits.device,
                )
                sc_mask = torch.zeros_like(model_input_ids, dtype=torch.bool)
                for batch_id, prev in enumerate(prev_raw_logits):
                    if prev is None or not active[batch_id]:
                        continue
                    block_start, block_end = request_spans[batch_id]
                    full_prev_logits[block_start:block_end] = prev
                    sc_mask[block_start:block_end] = ~prompt_masks[batch_id]
                old_post_norm = getattr(
                    model_runner.model, "_force_self_conditioning_post_norm", False
                )
                model_runner.model._force_self_conditioning_post_norm = (
                    self.self_conditioning_post_norm
                )
                active_spans = [
                    request_spans[batch_id]
                    for batch_id in range(batch_size)
                    if active[batch_id]
                    and prev_raw_logits[batch_id] is not None
                ]
                model_runner.model._sdar_direct_self_conditioning_capture_span = (
                    tuple(int(value) for value in active_spans[0])
                    if self._gpu_replay_lease_enabled
                    and len(active_spans) == 1
                    else None
                )
                input_embeds = model_runner.model.build_self_conditioned_input_embeds(
                    model_input_ids,
                    full_prev_logits,
                    sc_mask,
                )
                model_runner.model._force_self_conditioning_post_norm = old_post_norm
                if input_embeds is not None:
                    forward_batch.input_embeds = input_embeds.contiguous()
                    shared_stats = getattr(
                        model_runner.model, "_last_self_conditioning_stats", None
                    )
                    sc_stats_by_batch = [shared_stats for _ in range(batch_size)]

            forward_batch.input_ids = model_input_ids
            try:
                out = self._forward_denoise_with_graph_metrics(
                    model_runner, forward_batch
                )
                logits_output = out.logits_output
                can_run_cuda_graph = out.graph_was_replayed
            finally:
                forward_batch.dllm_is_denoise = old_is_denoise
                forward_batch.input_ids = old_input_ids
                forward_batch.input_embeds = old_input_embeds
                forward_batch.disable_cuda_graph_once = old_disable_graph

            if any(active[i] and not was_completed[i] for i in range(batch_size)):
                denoise_forwards += 1

            for batch_id in range(batch_size):
                if not active[batch_id]:
                    continue
                block_start, block_end = request_spans[batch_id]
                block_input_ids = block_inputs[batch_id]
                if block_input_ids is None:
                    continue
                accepted_mask = accepted_masks[batch_id]
                prompt_mask = prompt_masks[batch_id]
                raw_logits = logits_output.full_logits[block_start:block_end].float()
                prev_raw_logits[batch_id] = raw_logits.detach()
                input_ids_before = self._capture_replay_input_ids(
                    model_input_ids,
                    block_input_ids,
                    block_start,
                    block_end,
                )
                boundary = None
                if block_indices[batch_id] > 0:
                    boundary = self._boundary_logits_by_req.get(int(req_pool_indices[batch_id]))
                if boundary is not None:
                    boundary = boundary.to(device=raw_logits.device, dtype=raw_logits.dtype)
                if boundary is None:
                    boundary = raw_logits[0]
                if self.logits_shift:
                    curr_logits = torch.cat([boundary.unsqueeze(0), raw_logits[:-1]], dim=0)
                else:
                    curr_logits = raw_logits
                _step_capped_b = min(step, self.max_steps_per_block)
                _sample_temp_override_b = None
                _adaptive_temp_metrics_b: dict | None = None
                if self.temperature_mode in ADAPTIVE_TEMPERATURE_MODES:
                    (
                        _sample_temp_override_b,
                        _adaptive_temp_metrics_b,
                    ) = self._compute_adaptive_temperature(
                        req_pool_indices[batch_id],
                        _step_capped_b,
                        curr_logits,
                        prompt_mask,
                        accepted_mask,
                    )
                sampling_temperature_b = float(
                    _sample_temp_override_b
                    if _sample_temp_override_b is not None
                    else self._temperature(_step_capped_b)
                )
                proposal, probs, _ = self._sample_from_logits(
                    curr_logits,
                    _step_capped_b,
                    temperature_override=_sample_temp_override_b,
                )
                _confidence_temp_override_b = None
                _adaptive_confidence_temp_metrics_b: dict | None = None
                if (
                    self.confidence_temperature_mode
                    in ADAPTIVE_CONFIDENCE_TEMPERATURE_MODES
                ):
                    (
                        _confidence_temp_override_b,
                        _adaptive_confidence_temp_metrics_b,
                    ) = self._compute_adaptive_confidence_temperature(
                        req_pool_indices[batch_id],
                        _step_capped_b,
                        curr_logits,
                        prompt_mask,
                        accepted_before[batch_id],
                    )
                elif self.confidence_temperature_mode == "step_linear":
                    _confidence_temp_override_b = max(
                        self.confidence_temperature_min,
                        self.confidence_temperature_max
                        - self.confidence_temperature_step_decrement
                        * float(
                            max(_step_capped_b - 1, 0)
                            if self.confidence_temperature_first_step_full
                            else _step_capped_b
                        ),
                    )
                confidence_probs = self._confidence_probs(
                    curr_logits,
                    temperature_override=_confidence_temp_override_b,
                )
                diagnostic_sample_ids, diagnostic_sample_probs = (
                    self._trace_probability_sample(
                        confidence_probs,
                        req_pool_indices[batch_id],
                        block_indices[batch_id],
                        step,
                    )
                )
                proposal_probs = confidence_probs.gather(-1, proposal.unsqueeze(-1)).squeeze(-1)
                raw_logprobs = F.log_softmax(curr_logits.float(), dim=-1).gather(
                    -1, proposal.unsqueeze(-1)
                ).squeeze(-1)
                sample_logprobs = proposal_probs.clamp_min(
                    torch.finfo(proposal_probs.dtype).tiny
                ).log()
                entropy = -(
                    confidence_probs * confidence_probs.clamp_min(torch.finfo(confidence_probs.dtype).tiny).log()
                ).sum(dim=-1)

                changed = False
                edit_mask = torch.zeros_like(accepted_mask)
                step_shield_blocked = 0
                unresolved_mask = (~accepted_mask) & (~prompt_mask)
                if not was_completed[batch_id]:
                    accept_mask = self._build_accept_mask(
                        unresolved_mask,
                        proposal_probs,
                        entropy,
                        denoise_steps[batch_id],
                        prob_threshold=request_thresholds[batch_id][0],
                    )
                    if bool(accept_mask.any()):
                        block_input_ids[accept_mask] = proposal[accept_mask].to(
                            block_input_ids.dtype
                        )
                        q_probs_by_batch[batch_id][accept_mask] = proposal_probs[
                            accept_mask
                        ].float()
                        q_raw_logprobs_by_batch[batch_id][accept_mask] = raw_logprobs[
                            accept_mask
                        ].float()
                        q_sample_logprobs_by_batch[batch_id][accept_mask] = sample_logprobs[
                            accept_mask
                        ].float()
                        accepted_mask |= accept_mask
                        changed = True
                    denoise_steps[batch_id] += 1
                else:
                    refine_steps[batch_id] += 1

                if self.allow_accepted_change:
                    edit_mask = (
                        accepted_before[batch_id]
                        & (~prompt_mask)
                        & (proposal_probs > request_thresholds[batch_id][1])
                        & (proposal.to(block_input_ids.dtype) != block_input_ids)
                    )
                    if self.edit_shield_eos:
                        edit_mask, step_shield_blocked = self._apply_edit_shield(
                            edit_mask,
                            block_input_ids,
                            eos_commit_step_by_batch[batch_id],
                            step,
                        )
                    if bool(edit_mask.any()):
                        block_input_ids[edit_mask] = proposal[edit_mask].to(
                            block_input_ids.dtype
                        )
                        q_probs_by_batch[batch_id][edit_mask] = proposal_probs[
                            edit_mask
                        ].float()
                        q_raw_logprobs_by_batch[batch_id][edit_mask] = raw_logprobs[
                            edit_mask
                        ].float()
                        q_sample_logprobs_by_batch[batch_id][edit_mask] = sample_logprobs[
                            edit_mask
                        ].float()
                        changed = True

                if eos_commit_step_by_batch[batch_id] is not None:
                    eos_commit_step_by_batch[batch_id] = self._update_eos_commit_step(
                        eos_commit_step_by_batch[batch_id], block_input_ids, step
                    )

                new_accept_mask = accepted_mask & ~accepted_before[batch_id]
                action_mask = (new_accept_mask | edit_mask) & (~prompt_mask)
                block_completed_now = bool(accepted_mask.all())
                _adaptive_temp_metrics_b = (
                    self._complete_adaptive_temperature_metrics(
                        _adaptive_temp_metrics_b,
                        prompt_mask=prompt_mask,
                        accepted_mask=accepted_mask,
                        new_accept_mask=new_accept_mask,
                        edit_mask=edit_mask,
                    )
                )
                if _adaptive_temp_metrics_b is not None:
                    adaptive_temperature_diagnostics_by_batch[batch_id].append(
                        _adaptive_temp_metrics_b
                    )
                _adaptive_confidence_temp_metrics_b = (
                    self._complete_adaptive_confidence_temperature_metrics(
                        _adaptive_confidence_temp_metrics_b,
                        sampling_temperature=sampling_temperature_b,
                        prompt_mask=prompt_mask,
                        accepted_mask=accepted_mask,
                        new_accept_mask=new_accept_mask,
                        edit_mask=edit_mask,
                    )
                )
                if _adaptive_confidence_temp_metrics_b is not None:
                    adaptive_confidence_temperature_diagnostics_by_batch[
                        batch_id
                    ].append(_adaptive_confidence_temp_metrics_b)
                if self._async_capture_worker is None:
                    # Sync path: compute counts needed for trace record and stats
                    new_accept_count = int(new_accept_mask.sum().item())
                    edit_change_count = int(edit_mask.sum().item())
                    new_accept_tokens_by_batch[batch_id] += new_accept_count
                    edit_change_tokens_by_batch[batch_id] += edit_change_count
                # In async path counts are computed by consumer from GPU tensors

                if self._trace_dir:
                    if self._async_capture_worker is not None:
                        # Async path: enqueue lightweight handle; no D2H here
                        self._enqueue_step_handle_async(
                            model_runner=model_runner,
                            block_start=block_start,
                            block_end=block_end,
                            prompt_mask=prompt_mask,
                            block_idx=block_indices[batch_id],
                            req_pool_idx=req_pool_indices[batch_id],
                            step=step,
                            input_ids_before=input_ids_before,
                            current_ids=block_input_ids,
                            proposal=proposal,
                            proposal_probs=proposal_probs,
                            raw_logprobs=raw_logprobs,
                            sample_logprobs=sample_logprobs,
                            entropy=entropy,
                            accepted_mask=accepted_mask,
                            new_accept_mask=new_accept_mask,
                            edit_mask=edit_mask,
                            action_mask=action_mask,
                            is_final_denoise_row=block_completed_now,
                            was_completed=bool(was_completed[batch_id]),
                            start_offset=start_offsets[batch_id],
                            sc_stats=sc_stats_by_batch[batch_id],
                        )
                    else:
                        # Sync path: build trace_step dict inline (existing behavior)
                        new_accept_count_local = int(new_accept_mask.sum().item())
                        edit_change_count_local = int(edit_mask.sum().item())
                        trace_step = {
                            "trace_tensors_enabled": bool(self._trace_tensors),
                            "trace_tensor_records": int(self._trace_tensor_records),
                            "trace_dir_set": bool(self._trace_dir),
                            "step": step,
                            "was_completed": bool(was_completed[batch_id]),
                            "accepted_count": int(accepted_mask.sum().item()),
                            "new_accept_count": new_accept_count_local,
                            "edit_change_count": edit_change_count_local,
                            "shield_blocked_count": int(step_shield_blocked),
                            "action_token_count": int(action_mask.sum().item()),
                            "accepted_positions": [
                                int(x)
                                for x in torch.nonzero(accepted_mask, as_tuple=False)
                                .flatten()[:64]
                                .detach()
                                .cpu()
                                .tolist()
                            ],
                            "new_accept_positions": [
                                int(x)
                                for x in torch.nonzero(
                                    accepted_mask & ~accepted_before[batch_id],
                                    as_tuple=False,
                                )
                                .flatten()[:64]
                                .detach()
                                .cpu()
                                .tolist()
                            ],
                            "edit_change_positions": [
                                int(x)
                                for x in torch.nonzero(edit_mask, as_tuple=False)
                                .flatten()[:64]
                                .detach()
                                .cpu()
                                .tolist()
                            ],
                            "action_positions": [
                                int(x)
                                for x in torch.nonzero(action_mask, as_tuple=False)
                                .flatten()[:64]
                                .detach()
                                .cpu()
                                .tolist()
                            ],
                            "action_token_ids": [
                                int(proposal[int(x)].detach().cpu().item())
                                for x in torch.nonzero(action_mask, as_tuple=False)
                                .flatten()[:64]
                                .detach()
                                .cpu()
                                .tolist()
                            ],
                            "new_accept_token_logprobs": self._selected_trace_entries(
                                new_accept_mask,
                                proposal,
                                raw_logprobs,
                                sample_logprobs,
                                proposal_probs,
                            ),
                            "edit_token_logprobs": self._selected_trace_entries(
                                edit_mask,
                                proposal,
                                raw_logprobs,
                                sample_logprobs,
                                proposal_probs,
                            ),
                            "max_prob_mean": float(proposal_probs.mean().item()),
                            "entropy_mean": float(entropy.mean().item()),
                            "head_tokens": [
                                int(x) for x in block_input_ids[:16].detach().cpu().tolist()
                            ],
                            "proposal_head": [
                                int(x) for x in proposal[:32].detach().cpu().tolist()
                            ],
                            "maxp_head": [
                                float(x)
                                for x in proposal_probs[:16].detach().cpu().tolist()
                            ],
                            "self_conditioning": sc_stats_by_batch[batch_id],
                        }
                        if _adaptive_temp_metrics_b is not None:
                            trace_step["adaptive_temperature"] = dict(
                                _adaptive_temp_metrics_b
                            )
                        if _adaptive_confidence_temp_metrics_b is not None:
                            trace_step[
                                "adaptive_confidence_temperature"
                            ] = dict(
                                _adaptive_confidence_temp_metrics_b
                            )
                        replay_tensors = self._build_replay_tensor_trace(
                            model_runner,
                            block_start,
                            block_end,
                            prompt_mask,
                            block_indices[batch_id],
                            req_pool_indices[batch_id],
                            step,
                            input_ids_before=input_ids_before,
                            action_ids=proposal,
                            action_mask=action_mask,
                            accepted_mask=accepted_mask,
                            new_accept_mask=new_accept_mask,
                            edit_change_mask=edit_mask,
                            response_start_index=self._request_generated_tokens_by_req.get(
                                int(req_pool_indices[batch_id]),
                                0,
                            ),
                            start_offset=int(prompt_masks[batch_id].sum().item()),
                        )
                        if replay_tensors is not None:
                            trace_step["replay_tensors"] = replay_tensors
                        if self._trace_full:
                            trace_step.update(
                                {
                                    "current_ids": [
                                        int(x)
                                        for x in block_input_ids.detach()
                                        .cpu()
                                        .tolist()
                                    ],
                                    "proposal_ids": [
                                        int(x)
                                        for x in proposal.detach().cpu().tolist()
                                    ],
                                    "diagnostic_sample_ids": (
                                        [
                                            int(x)
                                            for x in diagnostic_sample_ids.detach()
                                            .cpu()
                                            .tolist()
                                        ]
                                        if diagnostic_sample_ids is not None
                                        else []
                                    ),
                                    "diagnostic_sample_probs": (
                                        [
                                            float(x)
                                            for x in diagnostic_sample_probs.detach()
                                            .cpu()
                                            .tolist()
                                        ]
                                        if diagnostic_sample_probs is not None
                                        else []
                                    ),
                                    "action_mask": [
                                        bool(x)
                                        for x in action_mask.detach().cpu().tolist()
                                    ],
                                    "accepted_mask": [
                                        bool(x)
                                        for x in accepted_mask.detach().cpu().tolist()
                                    ],
                                    "new_accept_mask": [
                                        bool(x)
                                        for x in (
                                            accepted_mask
                                            & ~accepted_before[batch_id]
                                        )
                                        .detach()
                                        .cpu()
                                        .tolist()
                                    ],
                                    "edit_mask": [
                                        bool(x)
                                        for x in edit_mask.detach().cpu().tolist()
                                    ],
                                    "proposal_probs": [
                                        float(x)
                                        for x in proposal_probs.detach().cpu().tolist()
                                    ],
                                }
                            )
                        trace_steps_by_batch[batch_id].append(trace_step)

                if block_completed_now:
                    completed_once[batch_id] = True
                    if not self.allow_accepted_change or self.post_accept_refine_steps <= 0:
                        active[batch_id] = False

                self._refresh_unaccepted_inputs(
                    block_input_ids,
                    accepted_mask,
                    block_indices[batch_id],
                    denoise_steps[batch_id],
                )
                if was_completed[batch_id] and not changed:
                    completed_once[batch_id] = True
                    # A refine pass that changed nothing ends the block, exactly as
                    # the single-request path breaks here. Previously the
                    # batched path kept refining until post_accept_refine_steps (16)
                    # was exhausted, so training rollouts (n=8 -> this path) paid up
                    # to 16 uncounted forwards per block that the serial eval path
                    # never ran: two decoders under one yaml. `refine_done` drops this
                    # request from `active` on the next iteration; refine_steps itself
                    # stays an honest forward count for the accounting.
                    if _BATCH_REFINE_EXIT:
                        refine_done[batch_id] = True
                last_iter_changed = last_iter_changed or bool(changed)

        self._last_block_final_forward_clean = bool(
            all(completed_once[i] for i in range(batch_size) if decode_block_mask[i])
            and not last_iter_changed
        )
        self._last_refine_steps_by_batch = [int(x) for x in refine_steps]

        for batch_id, trace_steps in enumerate(trace_steps_by_batch):
            if trace_steps:
                self._mark_final_denoise_row(req_pool_indices[batch_id], trace_steps)
            # Clear adaptive temperature state for each request after block completes.
            self._clear_adaptive_temp_state(req_pool_indices[batch_id])
            self._clear_adaptive_confidence_temp_state(
                req_pool_indices[batch_id]
            )

        return (
            logits_output,
            can_run_cuda_graph,
            denoise_forwards,
            denoise_steps,
            new_accept_tokens_by_batch,
            edit_change_tokens_by_batch,
            trace_steps_by_batch,
            adaptive_temperature_diagnostics_by_batch,
            adaptive_confidence_temperature_diagnostics_by_batch,
            q_probs_by_batch,
            q_raw_logprobs_by_batch,
            q_sample_logprobs_by_batch,
        )

    def _write_trace_record(self, record: dict) -> None:
        if not self._trace_dir:
            return
        os.makedirs(self._trace_dir, exist_ok=True)
        path = os.path.join(self._trace_dir, f"rank0_trace.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _write_introspection_record(self, record: dict) -> None:
        if not self._trace_dir:
            return
        os.makedirs(self._trace_dir, exist_ok=True)
        path = os.path.join(self._trace_dir, "rank0_introspection.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        overlap_fn=None,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        self._enable_model_tensor_capture(model_runner)
        if self._trace_tensors:
            # DLLM logits return full hidden states as a stable CUDA graph output.
            # CudaGraphRunner recaptures once if startup used NULL mode.
            forward_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        if self.self_conditioning_post_norm and hasattr(model_runner.model, "config"):
            # Post-norm checkpoints encode the architecture in remote code, but
            # some saved configs lack an explicit flag. Serving uses its own
            # SDAR implementation, so propagate the algorithm knob to all model
            # forward paths, including prompt prefill and clean commit.
            setattr(model_runner.model.config, "self_conditioning_post_norm", True)
        if hasattr(model_runner.model, "config"):
            setattr(
                model_runner.model.config,
                "self_conditioning_residual_scale",
                self.self_conditioning_residual_scale,
            )

        batch_size = forward_batch.batch_size
        device = forward_batch.input_ids.device
        request_spans = self._request_input_spans(forward_batch)

        mask_index = forward_batch.input_ids == self.mask_id
        if not bool(mask_index.any()):
            if (
                self._gpu_replay_force_recycle_at_idle_prefill
                and self._gpu_replay_lease_table is not None
            ):
                recycled = (
                    self._gpu_replay_lease_table.force_recycle_all()
                )
                if recycled["released_leases"]:
                    logger.info(
                        "[SDAR_GPU_REPLAY] quiescent prefill recycled "
                        "leases=%d bytes=%d",
                        recycled["released_leases"],
                        recycled["released_bytes"],
                    )
            # Pure prefill/commit-like block: run once to populate KV and retain
            # the last-position next-token logits as boundary logits for the next
            # denoise block. This mirrors HF idlm_block_generate.prev_last_logits.
            out = self._forward_without_generic_logprob(model_runner, forward_batch)
            logits_output = out.logits_output
            can_run_cuda_graph = out.graph_was_replayed
            full_logits = logits_output.full_logits.float()
            full_hidden = (
                logits_output.hidden_states
                if isinstance(logits_output.hidden_states, torch.Tensor)
                else getattr(model_runner.model, "_last_sdar_hidden_states", None)
            )
            if self._trace_tensors and isinstance(full_hidden, torch.Tensor):
                model_runner.model._last_sdar_hidden_states = full_hidden
            for batch_id, (block_start, block_end) in enumerate(request_spans):
                req_pool_idx = int(forward_batch.req_pool_indices[batch_id].item())
                self._boundary_logits_by_req[req_pool_idx] = (
                    full_logits[block_end - 1].detach().clone()
                )
                if self._trace_tensors and isinstance(full_hidden, torch.Tensor):
                    self._boundary_hidden_by_req[req_pool_idx] = (
                        full_hidden[block_end - 1].detach().clone()
                    )
                self._trace_block_idx[req_pool_idx] = (
                    self._trace_block_idx.get(req_pool_idx, 0) + 1
                )
            return logits_output, [], can_run_cuda_graph

        start_list = []
        prompt_masks = []
        req_pool_indices = []
        block_indices = []
        decode_block_mask = []
        for batch_id, (block_start, block_end) in enumerate(request_spans):
            block_input_ids = forward_batch.input_ids[block_start:block_end]
            span_len = int(block_end - block_start)
            has_mask = bool((block_input_ids == self.mask_id).any())
            # SGLang can schedule partial extend spans before a full SDAR block
            # is available. Those spans may contain mask tokens, but SDAR block
            # denoising requires exactly block_size tokens. Treat partial spans
            # as commit/prefill-like spans instead of crashing the scheduler.
            if has_mask and span_len != self.block_size:
                has_mask = False
            if has_mask:
                self._check_decode_span(batch_id, block_start, block_end)
            decode_block_mask.append(has_mask)
            # Initial non-mask tokens are prompt tail tokens, not generated tokens.
            prompt_mask = block_input_ids != self.mask_id
            prompt_masks.append(prompt_mask)
            start_list.append(
                int(prompt_mask.sum().item())
                if has_mask
                else span_len
            )
            req_pool_idx = int(forward_batch.req_pool_indices[batch_id].item())
            req_pool_indices.append(req_pool_idx)
            block_indices.append(self._trace_block_idx.get(req_pool_idx, 0))

        logits_output = None
        can_run_cuda_graph = False
        total_denoise_forwards = 0
        q_probs_by_batch = None
        q_raw_logprobs_by_batch = None
        q_sample_logprobs_by_batch = None
        request_denoise_forwards = [0 for _ in range(batch_size)]
        request_new_accept_tokens = [0 for _ in range(batch_size)]
        request_edit_change_tokens = [0 for _ in range(batch_size)]
        request_trace_records = [None for _ in range(batch_size)]
        request_adaptive_temperature_diagnostics: list[list[dict]] = [
            [] for _ in range(batch_size)
        ]
        request_adaptive_confidence_temperature_diagnostics: list[
            list[dict]
        ] = [[] for _ in range(batch_size)]
        if batch_size > 1:
            (
                logits_output,
                can_run_cuda_graph,
                total_denoise_forwards,
                request_denoise_forwards,
                request_new_accept_tokens,
                request_edit_change_tokens,
                trace_steps_by_batch,
                request_adaptive_temperature_diagnostics,
                request_adaptive_confidence_temperature_diagnostics,
                q_probs_by_batch,
                q_raw_logprobs_by_batch,
                q_sample_logprobs_by_batch,
            ) = self._decode_batch_blocks(
                model_runner,
                forward_batch,
                request_spans,
                decode_block_mask,
                prompt_masks,
                start_list,
                block_indices,
                req_pool_indices,
            )
            request_denoise_forwards = [int(x) for x in request_denoise_forwards]
            request_new_accept_tokens = [int(x) for x in request_new_accept_tokens]
            request_edit_change_tokens = [int(x) for x in request_edit_change_tokens]
            if self._trace_dir and self._async_capture_worker is None:
                for batch_id in range(batch_size):
                    if not decode_block_mask[batch_id]:
                        continue
                    block_start, block_end = request_spans[batch_id]
                    new_accept_count = int(request_new_accept_tokens[batch_id])
                    edit_change_count = int(request_edit_change_tokens[batch_id])
                    _thr_prob, _thr_edit = self._request_thresholds(
                        forward_batch, batch_id
                    )
                    trace_record = {
                        "time": time.time(),
                        "req_pool_idx": req_pool_indices[batch_id],
                        "block_idx": block_indices[batch_id],
                        "prob_threshold_effective": float(_thr_prob),
                        "edit_threshold_effective": float(_thr_edit),
                        "block_start": int(block_start),
                        "block_end": int(block_end),
                        "start_offset": int(start_list[batch_id]),
                        "denoise_forwards": int(request_denoise_forwards[batch_id]),
                        "batch_denoise_forwards": int(total_denoise_forwards),
                        "accepted_tokens": new_accept_count,
                        "new_accept_count": new_accept_count,
                        "new_accept_tokens": new_accept_count,
                        "edit_change_count": edit_change_count,
                        "edit_change_tokens": edit_change_count,
                        "adaptive_temperature_diagnostics": (
                            request_adaptive_temperature_diagnostics[batch_id]
                        ),
                        "adaptive_temperature_summary": (
                            summarize_adaptive_temperature_diagnostics(
                                request_adaptive_temperature_diagnostics[batch_id]
                            )
                        ),
                        "adaptive_confidence_temperature_diagnostics": (
                            request_adaptive_confidence_temperature_diagnostics[
                                batch_id
                            ]
                        ),
                        "adaptive_confidence_temperature_summary": (
                            summarize_adaptive_confidence_temperature_diagnostics(
                                request_adaptive_confidence_temperature_diagnostics[
                                    batch_id
                                ]
                            )
                        ),
                        "final_head_tokens": [
                            int(x)
                            for x in forward_batch.input_ids[block_start:block_end]
                            .detach()
                            .cpu()
                            .tolist()
                        ],
                        "steps": trace_steps_by_batch[batch_id],
                    }
                    request_trace_records[batch_id] = trace_record
                    self._write_trace_record(trace_record)
        else:
            q_probs_by_batch = []
            q_raw_logprobs_by_batch = []
            q_sample_logprobs_by_batch = []
            for batch_id in range(batch_size):
                block_start, block_end = request_spans[batch_id]
                if not decode_block_mask[batch_id]:
                    empty_q = torch.empty(
                        (0,), dtype=torch.float32, device=forward_batch.input_ids.device
                    )
                    q_probs_by_batch.append(empty_q)
                    q_raw_logprobs_by_batch.append(empty_q)
                    q_sample_logprobs_by_batch.append(empty_q)
                    continue
                (
                    logits_output,
                    can_run_cuda_graph,
                    denoise_forwards,
                    new_accept_tokens,
                    edit_change_tokens,
                    trace_steps,
                    adaptive_temperature_diagnostics,
                    adaptive_confidence_temperature_diagnostics,
                    q_probs,
                    q_raw_logprobs,
                    q_sample_logprobs,
                ) = self._decode_one_block(
                    model_runner,
                    forward_batch,
                    block_start,
                    block_end,
                    prompt_masks[batch_id],
                    batch_id,
                    block_indices[batch_id],
                    req_pool_indices[batch_id],
                    start_list[batch_id],
                )
                q_probs_by_batch.append(q_probs)
                q_raw_logprobs_by_batch.append(q_raw_logprobs)
                q_sample_logprobs_by_batch.append(q_sample_logprobs)
                total_denoise_forwards += denoise_forwards
                request_denoise_forwards[batch_id] = int(denoise_forwards)
                request_new_accept_tokens[batch_id] = int(new_accept_tokens)
                request_edit_change_tokens[batch_id] = int(edit_change_tokens)
                request_adaptive_temperature_diagnostics[batch_id] = (
                    adaptive_temperature_diagnostics
                )
                request_adaptive_confidence_temperature_diagnostics[
                    batch_id
                ] = adaptive_confidence_temperature_diagnostics
                if self._trace_dir and self._async_capture_worker is None:
                    req_pool_idx = req_pool_indices[batch_id]
                    block_idx = block_indices[batch_id]
                    new_accept_count = int(new_accept_tokens)
                    edit_change_count = int(edit_change_tokens)
                    _thr_prob, _thr_edit = self._request_thresholds(
                        forward_batch, batch_id
                    )
                    trace_record = {
                        "time": time.time(),
                        "req_pool_idx": req_pool_idx,
                        "block_idx": block_idx,
                        "prob_threshold_effective": float(_thr_prob),
                        "edit_threshold_effective": float(_thr_edit),
                        "block_start": int(block_start),
                        "block_end": int(block_end),
                        "start_offset": int(start_list[batch_id]) if batch_id < len(start_list) else 0,
                        "denoise_forwards": int(denoise_forwards),
                        "accepted_tokens": new_accept_count,
                        "new_accept_count": new_accept_count,
                        "new_accept_tokens": new_accept_count,
                        "edit_change_count": edit_change_count,
                        "edit_change_tokens": edit_change_count,
                        "adaptive_temperature_diagnostics": (
                            adaptive_temperature_diagnostics
                        ),
                        "adaptive_temperature_summary": (
                            summarize_adaptive_temperature_diagnostics(
                                adaptive_temperature_diagnostics
                            )
                        ),
                        "adaptive_confidence_temperature_diagnostics": (
                            adaptive_confidence_temperature_diagnostics
                        ),
                        "adaptive_confidence_temperature_summary": (
                            summarize_adaptive_confidence_temperature_diagnostics(
                                adaptive_confidence_temperature_diagnostics
                            )
                        ),
                        "final_head_tokens": [
                            int(x)
                            for x in forward_batch.input_ids[block_start:block_end]
                            .detach()
                            .cpu()
                            .tolist()
                        ],
                        "steps": trace_steps,
                    }
                    request_trace_records[batch_id] = trace_record
                    self._write_trace_record(trace_record)

        # Commit forward: write final clean tokens into KV cache. With
        # commit_mode=causal this is a causal clean-x0 pass, matching HF
        # idlm_block_generate's commit_mask; with commit_mode=bidirectional the
        # block is recomputed with the same block-bidirectional pattern the denoise
        # and refine passes (and the training replay kernel) use.
        # commit_mode=reuse_refine: skip the pass entirely when the decode loop's
        # last forward already ran on the final tokens (it wrote their K/V with the
        # bidirectional mask, exactly what a bidirectional commit would recompute);
        # its logits_output is still in hand and serves as the commit output.
        reuse_refine = (
            self.commit_mode == "reuse_refine"
            and bool(self._last_block_final_forward_clean)
            and logits_output is not None
        )
        commit_forwards_this_call = 0 if reuse_refine else 1
        if not reuse_refine:
            old_force_causal = getattr(forward_batch, "dllm_force_causal", False)
            old_is_commit = getattr(forward_batch, "dllm_is_commit", False)
            old_is_denoise = getattr(forward_batch, "dllm_is_denoise", False)
            old_disable_graph = getattr(forward_batch, "disable_cuda_graph_once", False)
            forward_batch.dllm_force_causal = self.commit_mode == "causal"
            forward_batch.dllm_is_commit = True
            forward_batch.dllm_is_denoise = False
            forward_batch.disable_cuda_graph_once = True
            try:
                out = self._forward_without_generic_logprob(model_runner, forward_batch)
                logits_output = out.logits_output
                can_run_cuda_graph = out.graph_was_replayed
            finally:
                forward_batch.dllm_force_causal = old_force_causal
                forward_batch.dllm_is_commit = old_is_commit
                forward_batch.dllm_is_denoise = old_is_denoise
                forward_batch.disable_cuda_graph_once = old_disable_graph
        commit_logits = logits_output.full_logits.float()
        commit_hidden = (
            logits_output.hidden_states
            if isinstance(logits_output.hidden_states, torch.Tensor)
            else getattr(model_runner.model, "_last_sdar_hidden_states", None)
        )
        if self._trace_tensors and isinstance(commit_hidden, torch.Tensor):
            model_runner.model._last_sdar_hidden_states = commit_hidden
        for batch_id, (block_start, block_end) in enumerate(request_spans):
            req_pool_idx = req_pool_indices[batch_id]
            if not decode_block_mask[batch_id]:
                self._boundary_logits_by_req[req_pool_idx] = (
                    commit_logits[block_end - 1].detach().clone()
                )
                if self._trace_tensors and isinstance(commit_hidden, torch.Tensor):
                    self._boundary_hidden_by_req[req_pool_idx] = (
                        commit_hidden[block_end - 1].detach().clone()
                    )
                self._trace_block_idx[req_pool_idx] = block_indices[batch_id] + 1
                continue
            self._check_decode_span(batch_id, block_start, block_end)
            final_ids = forward_batch.input_ids[block_start:block_end]
            raw_commit = commit_logits[block_start:block_end]
            boundary = None
            if block_indices[batch_id] > 0:
                boundary = self._boundary_logits_by_req.get(int(req_pool_idx))
            if boundary is not None:
                boundary = boundary.to(device=raw_commit.device, dtype=raw_commit.dtype)
            else:
                boundary = raw_commit[0]
            if self.logits_shift:
                shifted_commit = torch.cat([boundary.unsqueeze(0), raw_commit[:-1]], dim=0)
            else:
                shifted_commit = raw_commit
            commit_probs = torch.softmax(shifted_commit.float(), dim=-1)
            commit_argmax = commit_probs.argmax(dim=-1)
            commit_argmax_probs = commit_probs.gather(
                -1, commit_argmax.unsqueeze(-1)
            ).squeeze(-1)
            p_probs = commit_probs.gather(-1, final_ids.unsqueeze(-1)).squeeze(-1)
            q_probs = (
                q_probs_by_batch[batch_id]
                if q_probs_by_batch is not None
                else torch.full_like(p_probs, float("nan"), dtype=torch.float32)
            )
            q_raw_logprobs = (
                q_raw_logprobs_by_batch[batch_id]
                if q_raw_logprobs_by_batch is not None
                else torch.full_like(p_probs, float("nan"), dtype=torch.float32)
            )
            q_sample_logprobs = (
                q_sample_logprobs_by_batch[batch_id]
                if q_sample_logprobs_by_batch is not None
                else torch.full_like(p_probs, float("nan"), dtype=torch.float32)
            )
            nonprompt_start = int(start_list[batch_id])
            emitted_tokens = int(self.block_size - nonprompt_start)
            response_clean_start = int(
                self._request_generated_tokens_by_req.get(int(req_pool_idx), 0)
            )

            if self._async_capture_worker is not None and self._trace_dir:
                self._enqueue_block_handle_async(
                    final_ids=final_ids.detach(),
                    commit_argmax=commit_argmax.detach(),
                    commit_argmax_probs=commit_argmax_probs.detach(),
                    p_probs=p_probs.detach(),
                    q_probs=q_probs.detach(),
                    q_raw_logprobs=q_raw_logprobs.detach(),
                    q_sample_logprobs=q_sample_logprobs.detach(),
                    prompt_mask=prompt_masks[batch_id].detach(),
                    req_pool_idx=int(req_pool_idx),
                    block_idx=int(block_indices[batch_id]),
                    block_start=int(block_start),
                    block_end=int(block_end),
                    start_offset=nonprompt_start,
                    denoise_forwards=int(request_denoise_forwards[batch_id]),
                    batch_denoise_forwards=int(total_denoise_forwards),
                    response_clean_start=response_clean_start,
                    adaptive_temperature_diagnostics=(
                        request_adaptive_temperature_diagnostics[batch_id]
                    ),
                    adaptive_confidence_temperature_diagnostics=(
                        request_adaptive_confidence_temperature_diagnostics[
                            batch_id
                        ]
                    ),
                )
            else:
                # ----------------------------------------------------------------
                # Sync path: existing behavior (introspection + file writes)
                # ----------------------------------------------------------------
                valid = torch.isfinite(q_probs) & (q_probs > 0)
                ratios = torch.ones_like(p_probs, dtype=torch.float32)
                ratios[valid] = torch.minimum(
                    torch.ones_like(p_probs[valid], dtype=torch.float32),
                    p_probs[valid].float() / q_probs[valid].float().clamp_min(1e-12),
                )
                nonprompt = torch.arange(self.block_size, device=final_ids.device) >= nonprompt_start
                avg_ratio = (
                    float(ratios[nonprompt & valid].mean().item())
                    if bool((nonprompt & valid).any())
                    else float("nan")
                )
                self._write_introspection_record(
                    {
                        "time": time.time(),
                        "req_pool_idx": int(req_pool_idx),
                        "block_idx": int(block_indices[batch_id]),
                        "block_start": int(block_start),
                        "block_end": int(block_end),
                        "start_offset": int(start_list[batch_id]),
                        "avg_introspective_consistency": avg_ratio,
                        "positions": [
                            {
                                "pos": int(i),
                                "token_id": int(final_ids[i].item()),
                                "q": (
                                    float(q_probs[i].item())
                                    if bool(torch.isfinite(q_probs[i]).item())
                                    else None
                                ),
                                "p": float(p_probs[i].item()),
                                "commit_argmax_id": int(commit_argmax[i].item()),
                                "commit_argmax_p": float(commit_argmax_probs[i].item()),
                                "commit_argmax_matches_final": bool(
                                    int(commit_argmax[i].item()) == int(final_ids[i].item())
                                ),
                                "ratio": float(ratios[i].item()),
                                "is_prompt": bool(i < nonprompt_start),
                            }
                            for i in range(self.block_size)
                        ],
                    }
                )
                out_slice = slice(nonprompt_start, self.block_size)
                clean_token_probs = [
                    float(x) for x in p_probs[out_slice].detach().float().cpu().tolist()
                ]
                self._selected_logprobs_by_req[int(req_pool_idx)] = {
                    "token_ids": [
                        int(x) for x in final_ids[out_slice].detach().cpu().tolist()
                    ],
                    "raw_logprobs": [
                        float(x) for x in q_raw_logprobs[out_slice].detach().cpu().tolist()
                    ],
                    "sample_logprobs": [
                        float(x)
                        for x in q_sample_logprobs[out_slice].detach().cpu().tolist()
                    ],
                    "sample_probs": [
                        float(x) for x in q_probs[out_slice].detach().cpu().tolist()
                    ],
                }
                denoise_count = int(request_denoise_forwards[batch_id])
                refine_count = (
                    int(self._last_refine_steps_by_batch[batch_id])
                    if batch_id < len(self._last_refine_steps_by_batch)
                    else 0
                )
                new_accept_count = int(request_new_accept_tokens[batch_id])
                edit_change_count = int(request_edit_change_tokens[batch_id])
                self._attach_block_tpf_summary(
                    req_pool_idx,
                    int(block_indices[batch_id]),
                    new_accept_tokens=new_accept_count,
                    denoise_forwards=denoise_count,
                )
                self._preselect_completed_block_replay(
                    req_pool_idx,
                    int(block_indices[batch_id]),
                )
                request_stats = {
                    "block_idx": int(block_indices[batch_id]),
                    "block_size": int(self.block_size),
                    "start_offset": int(nonprompt_start),
                    "denoise_forwards": denoise_count,
                    # Post-accept refine passes are real forwards too.
                    # commit_forwards is 0 under reuse_refine.
                    "refine_forwards": refine_count,
                    "commit_forwards": commit_forwards_this_call,
                    "total_forwards": denoise_count + refine_count + commit_forwards_this_call,
                    "accepted_tokens": new_accept_count,
                    "new_accept_count": new_accept_count,
                    "new_accept_tokens": new_accept_count,
                    "edit_change_count": edit_change_count,
                    "edit_change_tokens": edit_change_count,
                    "returned_tokens": emitted_tokens,
                    "total_tokens": emitted_tokens,
                    "generated_tokens": emitted_tokens,
                    "true_forward_tokens": int(self.block_size * denoise_count),
                    "accepted_tokens_per_denoise_forward": new_accept_count / max(denoise_count, 1),
                    "new_accept_tokens_per_denoise_forward": new_accept_count / max(denoise_count, 1),
                    "edit_change_tokens_per_denoise_forward": edit_change_count / max(denoise_count, 1),
                    "tokens_per_denoise_forward": emitted_tokens / max(denoise_count, 1),
                    "clean_token_start_index": response_clean_start,
                    "clean_token_probs": clean_token_probs,
                }
                adaptive_temperature_diagnostics = (
                    request_adaptive_temperature_diagnostics[batch_id]
                )
                if adaptive_temperature_diagnostics:
                    request_stats["adaptive_temperature_diagnostics"] = [
                        dict(item)
                        for item in adaptive_temperature_diagnostics
                    ]
                    request_stats["adaptive_temperature_summary"] = (
                        summarize_adaptive_temperature_diagnostics(
                            adaptive_temperature_diagnostics
                        )
                    )
                adaptive_confidence_temperature_diagnostics = (
                    request_adaptive_confidence_temperature_diagnostics[
                        batch_id
                    ]
                )
                if adaptive_confidence_temperature_diagnostics:
                    request_stats[
                        "adaptive_confidence_temperature_diagnostics"
                    ] = [
                        dict(item)
                        for item in (
                            adaptive_confidence_temperature_diagnostics
                        )
                    ]
                    request_stats[
                        "adaptive_confidence_temperature_summary"
                    ] = (
                        summarize_adaptive_confidence_temperature_diagnostics(
                            adaptive_confidence_temperature_diagnostics
                        )
                    )
                if self._sdar_cuda_graph_enabled:
                    graph_stats = self._cuda_graph_metrics.get_stats(synchronize=False)
                    for key in (
                        "cuda_graph_total_forwards",
                        "cuda_graph_replays",
                        "cuda_graph_eager_fallbacks",
                        "cuda_graph_hit_rate",
                    ):
                        request_stats[key] = graph_stats[key]
                if request_trace_records[batch_id] is not None:
                    request_stats["replay_trace_record"] = request_trace_records[batch_id]
                self._request_stats_by_req.setdefault(int(req_pool_idx), []).append(request_stats)

            # Always update these synchronously - needed by next block's critical path
            self._request_generated_tokens_by_req[int(req_pool_idx)] = (
                int(self._request_generated_tokens_by_req.get(int(req_pool_idx), 0))
                + int(emitted_tokens)
            )
            self._boundary_logits_by_req[req_pool_indices[batch_id]] = (
                commit_logits[block_end - 1].detach().clone()
            )
            if self._trace_tensors and isinstance(commit_hidden, torch.Tensor):
                self._boundary_hidden_by_req[req_pool_indices[batch_id]] = (
                    commit_hidden[block_end - 1].detach().clone()
                )
            self._trace_block_idx[req_pool_indices[batch_id]] = block_indices[batch_id] + 1

        next_token_ids_list = []
        total_new_tokens = 0
        for batch_id, (block_start, block_end) in enumerate(request_spans):
            if not decode_block_mask[batch_id]:
                next_token_ids_list.append(forward_batch.input_ids.new_empty((0,)))
                continue
            self._check_decode_span(batch_id, block_start, block_end)
            start = start_list[batch_id]
            next_token_ids = forward_batch.input_ids[
                block_start + start : block_end
            ].clone()
            next_token_ids_list.append(next_token_ids)
            total_new_tokens += int(next_token_ids.numel())

        with self._stats_lock:
            _total_refine = int(sum(self._last_refine_steps_by_batch))
            self._stats["denoise_forwards"] += total_denoise_forwards
            self._stats["refine_forwards"] = (
                self._stats.get("refine_forwards", 0) + _total_refine
            )
            self._stats["commit_forwards"] += commit_forwards_this_call
            self._stats["total_forwards"] += (
                total_denoise_forwards + _total_refine + commit_forwards_this_call
            )
            if self._async_capture_worker is None:
                self._stats["accepted_tokens"] += sum(
                    int(x) for x in request_new_accept_tokens
                )
                self._stats["new_accept_tokens"] += sum(
                    int(x) for x in request_new_accept_tokens
                )
                self._stats["edit_change_tokens"] += sum(
                    int(x) for x in request_edit_change_tokens
                )
            self._stats["total_tokens"] += total_new_tokens
            self._stats["generated_tokens"] += total_new_tokens

        return logits_output, next_token_ids_list, can_run_cuda_graph


Algorithm = SDARPrefixKV
