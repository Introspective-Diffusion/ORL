"""Shared prompt-group policy for rule-reward and TPF GRPO."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


ALL_WRONG = "all_wrong_skip"
ALL_CORRECT_TPF = "all_correct_tpf"
MIXED_RULE = "mixed_rule"
SHORT_GROUP = "short_group_skip"


@dataclass(frozen=True)
class AdaptiveGRPOGroupStats:
    prompt_uid: str
    response_uids: tuple[str, ...]
    rewards: tuple[float, ...]
    effective_rewards: tuple[float, ...]
    advantages: tuple[float, ...]
    mean: float
    std: float
    short_group: bool
    zero_variance: bool
    mode: str


@dataclass(frozen=True)
class AdaptiveGRPOResult:
    advantages_by_response: dict[str, float]
    mode_by_response: dict[str, str]
    response_tpf_rewards: dict[str, float]
    group_stats: list[AdaptiveGRPOGroupStats]


def aggregate_response_tpf_rewards(
    block_summaries: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    """Aggregate exact block summaries into response TPF rewards.

    Response TPF is the ratio of total newly accepted tokens to total denoise
    forwards. Duplicate TP-rank copies are counted once by
    ``(response_uid, block_index)``.
    """

    unique: dict[tuple[str, int], tuple[float, float]] = {}
    for summary in block_summaries:
        response_uid = str(summary["response_uid"])
        block_index = int(summary["block_index"])
        new_accept_tokens = float(summary.get("new_accept_tokens", 0) or 0)
        denoise_forwards = float(summary.get("denoise_forwards", 0) or 0)
        if (
            not math.isfinite(new_accept_tokens)
            or not math.isfinite(denoise_forwards)
            or new_accept_tokens < 0.0
            or denoise_forwards <= 0.0
        ):
            raise ValueError(
                "response TPF requires finite non-negative new_accept_tokens "
                f"and positive denoise_forwards, got response_uid={response_uid!r} "
                f"block_index={block_index} values=({new_accept_tokens}, {denoise_forwards})"
            )
        key = (response_uid, block_index)
        value = (new_accept_tokens, denoise_forwards)
        previous = unique.get(key)
        if previous is not None and previous != value:
            raise ValueError(
                "inconsistent duplicate block TPF summary for "
                f"response_uid={response_uid!r} block_index={block_index}: "
                f"{previous} vs {value}"
            )
        unique[key] = value

    totals: dict[str, list[float]] = {}
    for (response_uid, _), (new_accept_tokens, denoise_forwards) in unique.items():
        aggregate = totals.setdefault(response_uid, [0.0, 0.0])
        aggregate[0] += new_accept_tokens
        aggregate[1] += denoise_forwards
    return {
        response_uid: values[0] / values[1]
        for response_uid, values in totals.items()
        if values[1] > 0.0
    }


def compute_adaptive_grpo_advantages(
    rule_rewards_by_response: Mapping[str, float],
    prompt_to_responses: Mapping[str, Sequence[str]],
    *,
    response_tpf_rewards: Mapping[str, float] | None,
    min_group_size: int,
    eps: float,
    norm_by_std: bool = True,
    correct_threshold: float = 0.0,
) -> AdaptiveGRPOResult:
    """Apply all-wrong skip, all-correct TPF, and mixed-rule semantics."""

    if min_group_size < 1:
        raise ValueError("min_group_size must be positive")
    if not math.isfinite(eps) or eps < 0.0:
        raise ValueError(f"eps must be finite and non-negative, got {eps}")
    tpf_rewards = {
        str(response_uid): float(value)
        for response_uid, value in (response_tpf_rewards or {}).items()
    }
    advantages: dict[str, float] = {}
    modes: dict[str, str] = {}
    stats: list[AdaptiveGRPOGroupStats] = []

    for prompt_uid in sorted(prompt_to_responses):
        response_uids = tuple(sorted(str(uid) for uid in prompt_to_responses[prompt_uid]))
        rule_rewards = tuple(
            float(rule_rewards_by_response.get(response_uid, 0.0))
            for response_uid in response_uids
        )
        if any(not math.isfinite(reward) for reward in rule_rewards):
            raise ValueError(f"non-finite rule reward in prompt group {prompt_uid!r}")

        short_group = len(response_uids) < min_group_size
        correct = tuple(reward > correct_threshold for reward in rule_rewards)
        if short_group:
            mode = SHORT_GROUP
            effective_rewards = rule_rewards
        elif not any(correct):
            mode = ALL_WRONG
            effective_rewards = rule_rewards
        elif all(correct):
            mode = ALL_CORRECT_TPF
            missing = [uid for uid in response_uids if uid not in tpf_rewards]
            if missing:
                raise ValueError(
                    "all-correct GRPO group is missing response TPF rewards: "
                    f"prompt_uid={prompt_uid!r} responses={missing[:5]}"
                )
            effective_rewards = tuple(tpf_rewards[uid] for uid in response_uids)
            if any(not math.isfinite(value) or value < 0.0 for value in effective_rewards):
                raise ValueError(
                    f"invalid response TPF reward in prompt group {prompt_uid!r}"
                )
        else:
            mode = MIXED_RULE
            effective_rewards = rule_rewards

        count = len(effective_rewards)
        mean = sum(effective_rewards) / count if count else 0.0
        variance = (
            sum((value - mean) ** 2 for value in effective_rewards) / count
            if count
            else 0.0
        )
        zero_variance = variance <= eps
        std = math.sqrt(variance + eps)
        if short_group or mode == ALL_WRONG or zero_variance:
            group_advantages = tuple(0.0 for _ in effective_rewards)
        elif norm_by_std:
            group_advantages = tuple(
                (value - mean) / std for value in effective_rewards
            )
        else:
            group_advantages = tuple(value - mean for value in effective_rewards)

        for response_uid, advantage in zip(
            response_uids, group_advantages, strict=True
        ):
            advantages[response_uid] = float(advantage)
            modes[response_uid] = mode
        stats.append(
            AdaptiveGRPOGroupStats(
                prompt_uid=str(prompt_uid),
                response_uids=response_uids,
                rewards=rule_rewards,
                effective_rewards=effective_rewards,
                advantages=group_advantages,
                mean=float(mean),
                std=float(std),
                short_group=short_group,
                zero_variance=zero_variance,
                mode=mode,
            )
        )

    return AdaptiveGRPOResult(
        advantages_by_response=advantages,
        mode_by_response=modes,
        response_tpf_rewards=tpf_rewards,
        group_stats=stats,
    )


def adaptive_group_metrics(
    result: AdaptiveGRPOResult,
    *,
    prefix: str,
) -> dict[str, float]:
    """Return common diagnostics for both model paths."""

    stats = result.group_stats
    active_responses = sum(
        abs(value) > 0.0 for value in result.advantages_by_response.values()
    )
    return {
        f"{prefix}/all_wrong_groups_skipped": float(
            sum(group.mode == ALL_WRONG for group in stats)
        ),
        f"{prefix}/all_correct_tpf_groups": float(
            sum(group.mode == ALL_CORRECT_TPF for group in stats)
        ),
        f"{prefix}/mixed_rule_groups": float(
            sum(group.mode == MIXED_RULE for group in stats)
        ),
        f"{prefix}/short_groups_skipped": float(
            sum(group.mode == SHORT_GROUP for group in stats)
        ),
        f"{prefix}/zero_signal_groups": float(
            sum(group.zero_variance or group.mode in {ALL_WRONG, SHORT_GROUP} for group in stats)
        ),
        f"{prefix}/active_responses": float(active_responses),
    }


__all__ = [
    "ALL_CORRECT_TPF",
    "ALL_WRONG",
    "MIXED_RULE",
    "SHORT_GROUP",
    "AdaptiveGRPOGroupStats",
    "AdaptiveGRPOResult",
    "adaptive_group_metrics",
    "aggregate_response_tpf_rewards",
    "compute_adaptive_grpo_advantages",
]
