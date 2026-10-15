"""Utilities for on-policy distillation training."""

from .dataset import trajectory_record_to_examples, trajectory_records_to_examples
from .losses import topk_distillation_loss
from .masks import build_prompt_clean_noisy_mask
from .rollout import DenoiseBlockState, PrefixNoisyBatch, apply_denoise_step, build_prefix_noisy_batch, select_transfer_indices
from .rollout_metrics import RolloutForwardEvent, RolloutMetrics, RolloutTimer
from .rollout_logging import build_rollout_sample_rows, log_rollout_samples_to_wandb
from .teacher import score_teacher_topk, score_teacher_topk_from_logits
from .trajectory import OPDExtendedRound, OPDRound, build_extended_rounds, expand_first_k_rounds, masked_mean_per_state

__all__ = [
    "OPDRound",
    "OPDExtendedRound",
    "build_rollout_sample_rows",
    "build_extended_rounds",
    "build_prompt_clean_noisy_mask",
    "DenoiseBlockState",
    "PrefixNoisyBatch",
    "expand_first_k_rounds",
    "log_rollout_samples_to_wandb",
    "masked_mean_per_state",
    "apply_denoise_step",
    "build_prefix_noisy_batch",
    "RolloutForwardEvent",
    "RolloutMetrics",
    "RolloutTimer",
    "score_teacher_topk",
    "score_teacher_topk_from_logits",
    "select_transfer_indices",
    "topk_distillation_loss",
    "trajectory_record_to_examples",
    "trajectory_records_to_examples",
]
