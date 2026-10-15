"""Qwen no-stream rule-based GRPO path with block-level TPF advantage scaling.

This module provides CPU-side orchestration for the Qwen SDAR no-stream GRPO
training path. It is intentionally independent of rollout and SGLang classes
and contains no streaming components (no replay queues, no split-stream CE, no
early-stream dispatch).

Algorithm:
  1. Collect four sampled responses per prompt (two prompts = eight responses).
  2. Compute math rule reward for each response (external, e.g. math_verify).
  3. GRPO outcome advantage grouped by prompt UID:
       advantage[r] = (reward[r] - mean_g) / (std_g + eps)
     Groups with < min_group_size or zero variance receive zero advantage.
  4. Block-level TPF from exact rollout metadata (not action_token_count):
       TPF[r, b] = new_accept_tokens[r, b] / denoise_forwards[r, b]
     Normalize by mean over unique (response_uid, block_index) pairs across the
     distributed rollout batch. Clip to [tpf_min_weight, tpf_max_weight].
  5. Scale only correct positive advantages for sampled denoise actions:
       scaled_advantage[r, b] = grpo_advantage[r] * tpf_weight[r, b]
     Wrong or non-positive advantages remain unchanged.
  6. Metadata-first replay row selection:
       - Always include each block's final denoise row.
       - Deterministically include ceil(fraction * group_size) non-final rows
         per (response_uid, block_index) group, ranked by outcome reward then
         step_index tie-break.
  7. Rank-align ghost rows; globally normalize action-token loss. One optimizer
     step per rollout batch.

External-prefix-KV kernel (Qwen):
  ``enable_external_prefix_kv=True`` marks the actor replay as requiring the
  native no-copy external-prefix-KV backend.  The pure CPU orchestration below
  still computes row selection and advantages; the actor path is responsible
  for calling the kernel and must fail fast rather than falling back to packed
  or repeated prefix tensors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch

from verl.trainer.ppo.adaptive_grpo_grouping import (
    AdaptiveGRPOGroupStats,
    adaptive_group_metrics,
    aggregate_response_tpf_rewards,
    compute_adaptive_grpo_advantages,
)
from verl.trainer.ppo.block_tpf_ce_weight import (
    BlockTPFRecords,
    MissingBlockTPFMetadataError,
    extract_rollout_block_tpf_summaries,
    extract_unique_block_tpf_records,
    merge_unique_block_tpf_records,
    stable_response_identity_hash,
)
from verl.trainer.ppo.diffusion_tpf_replay import (
    AdvantageEstimator,
    AdvantageResult,
    DiffusionTPFGroupStats,
    DiffusionTPFReplayConfig,
    DiffusionTPFReplayOutput,
    DiffusionTPFReplayRow,
    RewardProvider,
    compute_diffusion_tpf_replay,
    compute_tpf_reinforce_token_mean_loss,
)

__all__ = [
    "QwenGRPOConfig",
    "GRPOGroupStats",
    "QwenGRPOOutput",
    "OutcomeRewardProvider",
    "OutcomeAdvantageEstimator",
    "compute_grpo_advantages",
    "compute_block_tpf_response_weights",
    "compute_response_tpf_rewards",
    "compute_qwen_nostream_grpo",
    "compute_tpf_reinforce_token_mean_loss",
    "load_external_prefix_kv_kernel",
    "require_native_external_prefix_kv_backend",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QwenGRPOConfig:
    """Configuration for Qwen no-stream GRPO with block-level TPF scaling.

    All defaults match the specification: rollout.n=4, block_size=64,
    response_max_length=8192, TPF clip [0.25, 4.0], 10% non-final rows.
    """

    rollout_n: int = 4
    block_size: int = 64
    response_max_length: int = 8192
    tpf_min_weight: float = 0.25
    tpf_max_weight: float = 4.0
    fraction: float = 0.10
    selector_seed: int = 42
    min_group_size: int = 2
    eps: float = 1e-8
    missing_tpf_policy: str = "disable"
    enable_external_prefix_kv: bool = False
    metric_prefix: str = "qwen_nostream_grpo"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


GRPOGroupStats = AdaptiveGRPOGroupStats


@dataclass(frozen=True)
class QwenGRPOOutput:
    """Output of the Qwen no-stream GRPO orchestration."""

    rows: list[dict[str, Any]]
    rewards_by_response: dict[str, float]
    advantages_by_response: dict[str, float]
    block_tpf_weights: dict[tuple[str, int], float]
    row_scaled_advantages: list[float]
    selected_mask: torch.Tensor
    final_mask: torch.Tensor
    group_stats: list[GRPOGroupStats]
    metrics: dict[str, float]


# ---------------------------------------------------------------------------
# Reward and advantage providers (plug-in to diffusion_tpf_replay)
# ---------------------------------------------------------------------------


class OutcomeRewardProvider(RewardProvider):
    """Maps each denoise row's response_uid to the corresponding math reward.

    All rows belonging to the same response receive the same scalar reward.
    This allows row selection within diffusion_tpf_replay to use the outcome
    reward as the ranking signal; rows within the same group will tie-break
    deterministically on step_index.
    """

    def __init__(self, rewards_by_response: dict[str, float]) -> None:
        self._rewards = rewards_by_response

    def rewards(
        self,
        rows: Sequence[Mapping[str, Any]],
        config: DiffusionTPFReplayConfig,
    ) -> list[float]:
        return [float(self._rewards.get(str(row.get("response_uid", "")), 0.0)) for row in rows]


class OutcomeAdvantageEstimator(AdvantageEstimator):
    """Maps each denoise row's response_uid to the precomputed GRPO advantage.

    Advantage is assigned at response granularity (all rows of a response share
    the same advantage value before block-TPF scaling). The ``estimate`` method
    wraps the precomputed values so that row-selection within
    ``compute_diffusion_tpf_replay`` uses them for top-fraction ranking.
    """

    def __init__(self, advantages_by_response: dict[str, float]) -> None:
        self._advantages = advantages_by_response

    def estimate(
        self,
        rows: Sequence[Mapping[str, Any]],
        rewards: Sequence[float],
        config: DiffusionTPFReplayConfig,
    ) -> AdvantageResult:
        advantages = [
            float(self._advantages.get(str(row.get("response_uid", "")), 0.0)) for row in rows
        ]
        groups = _group_by(rows, config.group_key_fields)
        group_stats: list[DiffusionTPFGroupStats] = []
        for key, indices in groups.items():
            group_advs = [advantages[i] for i in indices]
            group_rwds = [float(rewards[i]) for i in indices]
            mean = sum(group_rwds) / len(group_rwds)
            var = sum((r - mean) ** 2 for r in group_rwds) / len(group_rwds)
            std = math.sqrt(var + config.eps)
            group_stats.append(DiffusionTPFGroupStats(
                key=key,
                key_dict={f: v for f, v in zip(config.group_key_fields, key, strict=True)},
                row_indices=tuple(indices),
                size=len(indices),
                reward_mean=mean,
                reward_variance=var,
                reward_std=std,
                short_group=len(indices) < config.min_group_size,
                zero_variance=var <= config.eps,
            ))
        return AdvantageResult(advantages=advantages, group_stats=group_stats)


# ---------------------------------------------------------------------------
# GRPO advantage computation
# ---------------------------------------------------------------------------


def compute_grpo_advantages(
    rewards_by_response: dict[str, float],
    prompt_by_response: dict[str, str],
    *,
    response_tpf_rewards: Mapping[str, float] | None = None,
    min_group_size: int = 2,
    eps: float = 1e-8,
) -> tuple[dict[str, float], list[GRPOGroupStats]]:
    """Compute adaptive GRPO advantages grouped by prompt UID.

    Args:
        rewards_by_response: Mapping response_uid → scalar math rule reward.
        prompt_by_response: Mapping response_uid → prompt_uid.
        response_tpf_rewards: Required for all-correct groups. If omitted, an
            all-correct group is treated as zero-signal for compatibility with
            direct callers that only test rule-reward grouping.
        min_group_size: Groups smaller than this receive zero advantage.
        eps: Variance floor for numerical stability.

    Returns:
        advantages_by_response: response_uid → GRPO advantage.
        group_stats: Per-prompt group statistics.
    """
    prompt_to_responses: dict[str, list[str]] = {}
    for response_uid, prompt_uid in prompt_by_response.items():
        prompt_to_responses.setdefault(str(prompt_uid), []).append(
            str(response_uid)
        )
    if response_tpf_rewards is None:
        # Preserve the historical direct-helper behavior while the top-level
        # runtime always supplies exact response TPF metadata.
        response_tpf_rewards = {
            response_uid: 0.0 for response_uid in rewards_by_response
        }
    result = compute_adaptive_grpo_advantages(
        rewards_by_response,
        prompt_to_responses,
        response_tpf_rewards=response_tpf_rewards,
        min_group_size=min_group_size,
        eps=eps,
    )
    return result.advantages_by_response, result.group_stats


# ---------------------------------------------------------------------------
# Block-level TPF weight computation
# ---------------------------------------------------------------------------


def compute_block_tpf_response_weights(
    rollout_states_by_response: dict[str, Mapping[Any, Any] | None],
    *,
    tpf_min_weight: float = 0.25,
    tpf_max_weight: float = 4.0,
    missing_policy: str = "disable",
    gathered_record_batches: Sequence[BlockTPFRecords] | None = None,
    metric_prefix: str = "qwen_grpo",
) -> tuple[dict[tuple[str, int], float], dict[str, float]]:
    """Compute block-level TPF weights normalized across the distributed batch.

    Uses ``extract_rollout_block_tpf_summaries`` to read
    ``new_accept_tokens / denoise_forwards`` per block (not action_token_count).
    Normalization deduplicates across ranks so each unique (response_uid,
    block_index) pair is counted exactly once in the global mean.

    Args:
        rollout_states_by_response: response_uid → rollout self-conditioning dict.
            None entries are treated as missing and handled per missing_policy.
        tpf_min_weight: Lower clip bound after normalization.
        tpf_max_weight: Upper clip bound after normalization.
        missing_policy: ``"error"`` raises; ``"disable"`` returns weight=1 fallback.
        gathered_record_batches: Pre-gathered records for testing or custom collectives.

    Returns:
        weights: (response_uid, block_index) → weight in [min, max].
        metrics: Diagnostic scalar metrics.
    """
    block_raw: list[tuple[str, int, float]] = []
    missing_count = 0

    for response_uid, states in rollout_states_by_response.items():
        if states is None:
            missing_count += 1
            continue
        try:
            summaries = extract_rollout_block_tpf_summaries(states)
        except MissingBlockTPFMetadataError:
            if missing_policy == "error":
                raise
            missing_count += 1
            continue
        for block_index, summary in summaries.items():
            block_raw.append((response_uid, block_index, summary.raw_tpf))

    if not block_raw:
        if missing_policy == "error":
            raise MissingBlockTPFMetadataError("no valid block TPF data found in rollout states")
        return {}, {
            f"{metric_prefix}/tpf_global_mean": 0.0,
            f"{metric_prefix}/tpf_local_blocks": 0.0,
            f"{metric_prefix}/tpf_missing_responses": float(missing_count),
        }

    # Build shape-[n, 1] token tensors so existing record utilities can deduplicate.
    n = len(block_raw)
    raw_tpf_t = torch.tensor([[v] for _, _, v in block_raw], dtype=torch.float32)
    hashes = [stable_response_identity_hash(r) for r, _, _ in block_raw]
    hashes_t = torch.tensor([[[h0, h1]] for h0, h1 in hashes], dtype=torch.long)
    block_idx_t = torch.tensor([[bi] for _, bi, _ in block_raw], dtype=torch.long)
    active_t = torch.ones([n, 1], dtype=torch.bool)

    local_records = extract_unique_block_tpf_records(raw_tpf_t, hashes_t, block_idx_t, active_t)

    all_batches: list[BlockTPFRecords]
    if gathered_record_batches is not None:
        all_batches = list(gathered_record_batches)
    else:
        all_batches = _gather_tpf_records(local_records)

    global_records = merge_unique_block_tpf_records(all_batches)

    if global_records.values.numel() == 0:
        if missing_policy == "error":
            raise MissingBlockTPFMetadataError("distributed rollout batch has no active TPF blocks")
        return {}, {
            f"{metric_prefix}/tpf_global_mean": 0.0,
            f"{metric_prefix}/tpf_local_blocks": float(n),
            f"{metric_prefix}/tpf_missing_responses": float(missing_count),
        }

    global_mean = float(global_records.values.mean().item())
    if not math.isfinite(global_mean) or global_mean < 0.0:
        raise ValueError(f"global block TPF mean must be finite and positive, got {global_mean}")

    weights: dict[tuple[str, int], float] = {}
    global_by_key = {
        tuple(int(item) for item in key): float(value)
        for key, value in zip(
            global_records.keys.detach().cpu().tolist(),
            global_records.values.detach().cpu().tolist(),
            strict=True,
        )
    }
    for response_uid, block_index, raw_tpf in block_raw:
        h0, h1 = stable_response_identity_hash(response_uid)
        global_raw = global_by_key.get((h0, h1, int(block_index)))
        if global_raw is None:
            raise ValueError(
                f"global block TPF records are missing {(response_uid, block_index)!r}"
            )
        if not math.isclose(global_raw, raw_tpf, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(
                f"local/global block TPF mismatch for {(response_uid, block_index)!r}: "
                f"{raw_tpf} vs {global_raw}"
            )
        w = (
            1.0
            if global_mean == 0.0
            else max(tpf_min_weight, min(tpf_max_weight, global_raw / global_mean))
        )
        weights[(response_uid, block_index)] = w

    v = global_records.values
    metrics = {
        f"{metric_prefix}/tpf_global_mean": global_mean,
        f"{metric_prefix}/tpf_global_blocks": float(v.numel()),
        f"{metric_prefix}/tpf_local_blocks": float(n),
        f"{metric_prefix}/tpf_missing_responses": float(missing_count),
        f"{metric_prefix}/tpf_raw_min": float(v.min().item()),
        f"{metric_prefix}/tpf_raw_max": float(v.max().item()),
    }
    return weights, metrics


def compute_response_tpf_rewards(
    rollout_states_by_response: dict[str, Mapping[Any, Any] | None],
    *,
    missing_policy: str,
) -> dict[str, float]:
    """Compute exact response-level TPF rewards from canonical block summaries."""

    block_summaries: list[dict[str, Any]] = []
    missing: list[str] = []
    for response_uid, states in rollout_states_by_response.items():
        if states is None:
            missing.append(str(response_uid))
            continue
        try:
            summaries = extract_rollout_block_tpf_summaries(states)
        except MissingBlockTPFMetadataError:
            if missing_policy == "error":
                raise
            missing.append(str(response_uid))
            continue
        for block_index, summary in summaries.items():
            block_summaries.append(
                {
                    "response_uid": str(response_uid),
                    "block_index": int(block_index),
                    "new_accept_tokens": int(summary.new_accept_tokens),
                    "denoise_forwards": int(summary.denoise_forwards),
                }
            )
    if missing and missing_policy == "error":
        raise MissingBlockTPFMetadataError(
            "responses are missing exact TPF metadata: "
            f"{sorted(missing)[:5]}"
        )
    return aggregate_response_tpf_rewards(block_summaries)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def _select_final_plus_fraction(
    rows: Sequence[Mapping[str, Any]],
    *,
    fraction: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select producer-pruned rows or final plus earliest deterministic fraction."""

    selected = [False] * len(rows)
    final = [bool(row.get("is_final_denoise_row", False)) for row in rows]
    groups = _group_by(rows, ("response_uid", "block_index"))
    for _, indices in groups.items():
        if indices and all(
            bool(rows[index].get("producer_preselected", False))
            for index in indices
        ):
            for index in indices:
                selected[index] = True
            continue
        non_final = [index for index in indices if not final[index]]
        for index in indices:
            if final[index]:
                selected[index] = True
        n_pick = math.ceil(float(fraction) * len(non_final))
        if n_pick <= 0:
            continue
        ordered_non_final = sorted(
            non_final,
            key=lambda index: (
                int(rows[index].get("step_index", 0)),
                str(rows[index].get("request_uid", "")),
                str(rows[index].get("sample_uid", "")),
            ),
        )
        for index in ordered_non_final[:n_pick]:
            selected[index] = True
    return torch.tensor(selected, dtype=torch.bool), torch.tensor(final, dtype=torch.bool)


def compute_qwen_nostream_grpo(
    rows: Sequence[DiffusionTPFReplayRow | Mapping[str, Any]],
    rewards_by_response: dict[str, float],
    prompt_by_response: dict[str, str],
    rollout_states_by_response: dict[str, Mapping[Any, Any] | None],
    *,
    config: QwenGRPOConfig | None = None,
    gathered_tpf_record_batches: Sequence[BlockTPFRecords] | None = None,
) -> QwenGRPOOutput:
    """Compute Qwen no-stream GRPO advantages with block-level TPF scaling.

    This is the top-level CPU orchestration function. No streaming components
    are referenced or instantiated.

    Args:
        rows: All denoise forward rows for this rollout batch. Each row must
            carry ``request_uid``, ``sample_uid``, ``response_uid``,
            ``block_index``, ``step_index``, and optionally
            ``is_final_denoise_row``.
        rewards_by_response: response_uid → math rule scalar reward (-1 or +1).
        prompt_by_response: response_uid → prompt_uid for GRPO grouping.
        rollout_states_by_response: response_uid → rollout self-conditioning
            metadata dict (keyed by denoise step). None means missing.
        config: Algorithm hyper-parameters.
        gathered_tpf_record_batches: Optional pre-gathered TPF records for
            deterministic tests or custom distributed collectives.

    Returns:
        QwenGRPOOutput containing selected rows, per-row scaled advantages,
        selection masks, GRPO group stats, and diagnostic metrics.

    Raises:
        MissingBlockTPFMetadataError: If ``config.missing_tpf_policy == "error"``
            and any response is missing block TPF metadata.
    """
    if config is None:
        config = QwenGRPOConfig()

    # --- Step 1: Exact block and response TPF ---
    block_tpf_weights, tpf_metrics = compute_block_tpf_response_weights(
        rollout_states_by_response,
        tpf_min_weight=config.tpf_min_weight,
        tpf_max_weight=config.tpf_max_weight,
        missing_policy=config.missing_tpf_policy,
        gathered_record_batches=gathered_tpf_record_batches,
        metric_prefix=config.metric_prefix,
    )
    response_tpf_rewards = compute_response_tpf_rewards(
        rollout_states_by_response,
        missing_policy=config.missing_tpf_policy,
    )
    if config.missing_tpf_policy != "error":
        for response_uid in rewards_by_response:
            response_tpf_rewards.setdefault(str(response_uid), 0.0)

    # --- Step 2: Three-way prompt-group advantages ---
    prompt_to_responses: dict[str, list[str]] = {}
    for response_uid, prompt_uid in prompt_by_response.items():
        prompt_to_responses.setdefault(str(prompt_uid), []).append(
            str(response_uid)
        )
    adaptive_result = compute_adaptive_grpo_advantages(
        rewards_by_response,
        prompt_to_responses,
        response_tpf_rewards=response_tpf_rewards,
        min_group_size=config.min_group_size,
        eps=config.eps,
    )
    advantages_by_response = adaptive_result.advantages_by_response
    grpo_group_stats = adaptive_result.group_stats

    # --- Step 3: Row selection — final + fraction non-final per block ---
    # OutcomeRewardProvider feeds math rewards as row scores so that tie-breaking
    # (within groups of equal reward) is deterministic via step_index.
    replay_config = DiffusionTPFReplayConfig(
        group_key_fields=("response_uid", "block_index"),
        identity_fields=("request_uid", "sample_uid", "response_uid", "block_index", "step_index"),
        tie_break_fields=("response_uid", "block_index", "step_index", "request_uid", "sample_uid"),
        fraction=config.fraction,
        min_group_size=1,
        final_step_policy="flag",
        eps=config.eps,
        metric_prefix=config.metric_prefix,
    )
    replay_output: DiffusionTPFReplayOutput = compute_diffusion_tpf_replay(
        rows,
        config=replay_config,
        reward_provider=OutcomeRewardProvider(rewards_by_response),
        advantage_estimator=OutcomeAdvantageEstimator(advantages_by_response),
    )
    selected_mask, final_mask = _select_final_plus_fraction(
        replay_output.rows,
        fraction=config.fraction,
        seed=config.selector_seed,
    )
    active_row_mask = torch.tensor(
        [
            abs(float(advantages_by_response.get(str(row.get("response_uid", "")), 0.0)))
            > config.eps
            for row in replay_output.rows
        ],
        dtype=torch.bool,
    )
    selected_mask &= active_row_mask

    # --- Step 4: Scale only correct positive advantages by per-block TPF ---
    row_scaled_advantages: list[float] = []
    positive_tpf_scaled_rows = 0
    negative_unscaled_rows = 0
    for row in replay_output.rows:
        response_uid = str(row.get("response_uid", ""))
        block_index = int(row.get("block_index", 0))
        grpo_adv = float(advantages_by_response.get(response_uid, 0.0))
        rule_reward = float(rewards_by_response.get(response_uid, 0.0))
        tpf_w = float(block_tpf_weights.get((response_uid, block_index), 1.0))
        if rule_reward > 0.0 and grpo_adv > config.eps:
            row_scaled_advantages.append(grpo_adv * tpf_w)
            positive_tpf_scaled_rows += 1
        else:
            row_scaled_advantages.append(grpo_adv)
            if grpo_adv < -config.eps:
                negative_unscaled_rows += 1

    # --- Step 5: Aggregate metrics ---
    grpo_metrics = _build_grpo_metrics(grpo_group_stats, advantages_by_response, rewards_by_response, config)
    all_metrics: dict[str, float] = {
        **grpo_metrics,
        **adaptive_group_metrics(adaptive_result, prefix=config.metric_prefix),
        **tpf_metrics,
        **replay_output.metrics,
    }
    all_metrics[f"{config.metric_prefix}/selected_rows"] = float(selected_mask.sum().item())
    all_metrics[f"{config.metric_prefix}/final_rows"] = float(final_mask.sum().item())
    all_metrics[f"{config.metric_prefix}/selected_fraction"] = (
        float(selected_mask.float().mean().item()) if selected_mask.numel() else 0.0
    )
    all_metrics[f"{config.metric_prefix}/external_prefix_kv_required"] = float(
        bool(config.enable_external_prefix_kv)
    )
    all_metrics[f"{config.metric_prefix}/positive_tpf_scaled_rows"] = float(
        positive_tpf_scaled_rows
    )
    all_metrics[f"{config.metric_prefix}/negative_unscaled_rows"] = float(
        negative_unscaled_rows
    )
    all_metrics[f"{config.metric_prefix}/response_tpf_min"] = (
        min(response_tpf_rewards.values()) if response_tpf_rewards else 0.0
    )
    all_metrics[f"{config.metric_prefix}/response_tpf_max"] = (
        max(response_tpf_rewards.values()) if response_tpf_rewards else 0.0
    )

    return QwenGRPOOutput(
        rows=replay_output.rows,
        rewards_by_response=rewards_by_response,
        advantages_by_response=advantages_by_response,
        block_tpf_weights=block_tpf_weights,
        row_scaled_advantages=row_scaled_advantages,
        selected_mask=selected_mask,
        final_mask=final_mask,
        group_stats=grpo_group_stats,
        metrics=all_metrics,
    )


# ---------------------------------------------------------------------------
# Loss computation helpers
# ---------------------------------------------------------------------------


def compute_qwen_grpo_loss(
    action_log_probs: torch.Tensor,
    action_mask: torch.Tensor,
    row_scaled_advantages: torch.Tensor,
    *,
    normalization_denominator: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Compute the GRPO policy gradient loss with global token normalization.

    Wraps ``compute_tpf_reinforce_token_mean_loss`` with Qwen-specific
    documentation. The loss is:

        L = -sum_i A_i * sum_t log pi(a_it | s_i) / global_token_count

    where A_i is the block-TPF-scaled GRPO advantage for row i.

    Args:
        action_log_probs: Shape [rows, tokens], log probabilities.
        action_mask: Shape [rows, tokens], bool mask for valid action tokens.
        row_scaled_advantages: Shape [rows], per-row GRPO * TPF advantage.
        normalization_denominator: Global action token count across all ranks.
            If None, uses local masked token count.

    Returns:
        Scalar loss tensor.
    """
    return compute_tpf_reinforce_token_mean_loss(
        action_log_probs,
        action_mask,
        row_scaled_advantages,
        normalization_denominator=normalization_denominator,
    )


def load_external_prefix_kv_kernel():
    """Load the native external-prefix-KV attention entry points."""

    try:
        from fast_diffusion_attention.external_prefix_kv_attn import (  # type: ignore[import]
            external_kv_varlen_attention,
        )
        from fast_diffusion_attention.external_prefix_kv_triton import (  # type: ignore[import]
            triton_backend_supported,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Qwen no-stream GRPO requires the fast-diffusion-attention "
            "external-prefix-KV kernel package. Install it and ensure "
            "fast_diffusion_attention is importable."
        ) from exc
    return external_kv_varlen_attention, triton_backend_supported


def require_native_external_prefix_kv_backend(
    *,
    q_canvas: torch.Tensor,
    k_canvas: torch.Tensor,
    v_canvas: torch.Tensor,
    k_prefix: torch.Tensor,
    v_prefix: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_prefix: torch.Tensor,
    prefix_segment_ids: torch.Tensor | None = None,
) -> tuple[Any, Any]:
    """Return kernel functions only if the no-copy Triton backend can run."""

    external_kv_varlen_attention, triton_backend_supported = load_external_prefix_kv_kernel()
    supported, reason = triton_backend_supported(
        q_canvas,
        k_canvas,
        v_canvas,
        k_prefix,
        v_prefix,
        cu_seqlens_q,
        cu_seqlens_prefix,
        prefix_segment_ids,
    )
    if not supported:
        raise RuntimeError(
            "Qwen no-stream GRPO requires the native no-copy external-prefix-KV "
            f"Triton backend; refusing packed/repeated-prefix fallback: {reason}"
        )
    return external_kv_varlen_attention, triton_backend_supported


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _group_by(
    rows: Sequence[Mapping[str, Any]],
    key_fields: Sequence[str],
) -> dict[tuple[Any, ...], list[int]]:
    groups: dict[tuple[Any, ...], list[int]] = {}
    for i, row in enumerate(rows):
        key = tuple(row.get(f) for f in key_fields)
        groups.setdefault(key, []).append(i)
    return groups


def _gather_tpf_records(local_records: BlockTPFRecords) -> list[BlockTPFRecords]:
    """Gather BlockTPFRecords across all ranks (no-op if not initialized)."""
    if not torch.distributed.is_initialized() or torch.distributed.get_world_size() <= 1:
        return [local_records]

    world_size = torch.distributed.get_world_size()
    device = local_records.keys.device
    local_n = torch.tensor([local_records.keys.shape[0]], dtype=torch.long, device=device)
    gathered_ns = [torch.zeros_like(local_n) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_ns, local_n)
    counts = [int(v.item()) for v in gathered_ns]
    max_n = max(counts, default=0)

    if max_n <= 0:
        empty = BlockTPFRecords(
            keys=torch.empty((0, 3), dtype=torch.long, device=device),
            values=torch.empty((0,), dtype=torch.float64, device=device),
        )
        return [empty] * world_size

    padded_keys = torch.full((max_n, 3), -1, dtype=torch.long, device=device)
    padded_vals = torch.zeros((max_n,), dtype=torch.float64, device=device)
    if local_records.keys.shape[0] > 0:
        padded_keys[: local_records.keys.shape[0]] = local_records.keys
        padded_vals[: local_records.values.shape[0]] = local_records.values

    gathered_keys = [torch.empty_like(padded_keys) for _ in range(world_size)]
    gathered_vals = [torch.empty_like(padded_vals) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_keys, padded_keys)
    torch.distributed.all_gather(gathered_vals, padded_vals)

    return [
        BlockTPFRecords(keys=k[:c], values=v[:c])
        for k, v, c in zip(gathered_keys, gathered_vals, counts, strict=True)
    ]


def _build_grpo_metrics(
    group_stats: list[GRPOGroupStats],
    advantages_by_response: dict[str, float],
    rewards_by_response: dict[str, float],
    config: QwenGRPOConfig,
) -> dict[str, float]:
    prefix = config.metric_prefix
    all_rewards = list(rewards_by_response.values())
    all_advantages = list(advantages_by_response.values())
    return {
        f"{prefix}/grpo_groups": float(len(group_stats)),
        f"{prefix}/grpo_total_responses": float(len(rewards_by_response)),
        f"{prefix}/grpo_short_groups": float(sum(1 for s in group_stats if s.short_group)),
        f"{prefix}/grpo_zero_variance_groups": float(sum(1 for s in group_stats if s.zero_variance)),
        f"{prefix}/grpo_reward_mean": float(sum(all_rewards) / len(all_rewards)) if all_rewards else 0.0,
        f"{prefix}/grpo_reward_max": float(max(all_rewards)) if all_rewards else 0.0,
        f"{prefix}/grpo_reward_min": float(min(all_rewards)) if all_rewards else 0.0,
        f"{prefix}/grpo_advantage_mean": float(sum(all_advantages) / len(all_advantages)) if all_advantages else 0.0,
        f"{prefix}/grpo_advantage_max": float(max(all_advantages)) if all_advantages else 0.0,
        f"{prefix}/grpo_advantage_min": float(min(all_advantages)) if all_advantages else 0.0,
    }
