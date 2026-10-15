"""Signed all-token CE response routing.

This module computes per-response roles (+1 positive, -1 negative, 0 drop)
grouped by prompt uid for the signed CE objective:
  L = mean_global_tokens(CE_correct) - mean_global_tokens(CE_wrong)

Baseline (top-TPF positive, top-25% wrong negative CE):
  - Per prompt uid: exactly one highest response-TPF correct response -> +1
  - ceil(0.25 * n_wrong) highest response-TPF wrong responses -> -1
  - All others -> 0 (drop, leases acknowledged)

Residual variant:
  - Same response/state selection and signed labels as the baseline.
  - random_active_block applied to both positive and negative pages at actor
    replay (configured at launch time, transparent to this module).

State-level sampling for both uses SGLANG_SDAR_REPLAY_STEP_SAMPLE_FRACTION=0.25
stratified mode (set in the launcher environment), which is the existing per-block
state capture infrastructure and requires no code changes here.

This module is intentionally NOT negative_action_pg and must not share its
config keys, guards, or semantics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


RESPONSE_ROLE_POSITIVE = 1
RESPONSE_ROLE_NEGATIVE = -1
RESPONSE_ROLE_DROP = 0


@dataclass(frozen=True)
class SignedTraceCEConfig:
    """Configuration for signed trace CE response routing."""

    enable: bool = False
    wrong_fraction: float = 0.25
    tpf_tie_key: str = "rollout_index"
    # "top1": one highest-TPF correct response per prompt gets +1. "all":
    # every correct response gets +1.
    positive_mode: str = "top1"
    # "drop": length-capped rollouts are forced to role 0 and train nothing.
    # "negative": they stay in the wrong pool and can
    # take -CE — plugs the "never terminate = never punished" escape hatch
    # that collapsed earlier runs (models learned to run to the
    # length cap: 8/8 length-dropped, TPF 42->60, eval acc -> 12).
    length_role: str = "drop"

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any] | None,
    ) -> "SignedTraceCEConfig":
        values = values or {}
        if not isinstance(values, Mapping):
            raise ValueError("signed_trace_ce must be a mapping")
        enable = bool(values.get("enable", False))
        wrong_fraction = float(values.get("wrong_fraction", 0.25))
        if not math.isfinite(wrong_fraction) or not 0.0 < wrong_fraction <= 1.0:
            raise ValueError(
                "signed_trace_ce.wrong_fraction must be in (0, 1], "
                f"got {wrong_fraction!r}"
            )
        tpf_tie_key = str(values.get("tpf_tie_key", "rollout_index"))
        positive_mode = str(values.get("positive_mode", "top1")).strip().lower()
        length_role = str(values.get("length_role", "drop")).strip().lower()
        return cls(
            enable=enable,
            wrong_fraction=wrong_fraction,
            tpf_tie_key=tpf_tie_key,
            positive_mode=positive_mode,
            length_role=length_role,
        )

    def validate(self) -> None:
        if not math.isfinite(self.wrong_fraction) or not 0.0 < self.wrong_fraction <= 1.0:
            raise ValueError(
                "signed_trace_ce.wrong_fraction must be in (0, 1], "
                f"got {self.wrong_fraction!r}"
            )
        if self.positive_mode not in ("top1", "all"):
            raise ValueError(
                "signed_trace_ce.positive_mode must be top1 or all, "
                f"got {self.positive_mode!r}"
            )
        if self.length_role not in ("drop", "negative"):
            raise ValueError(
                "signed_trace_ce.length_role must be drop or negative, "
                f"got {self.length_role!r}"
            )


@dataclass
class SignedResponseRoutingResult:
    """Per-response roles and routing metrics."""

    roles: list[int]
    metrics: dict[str, float] = field(default_factory=dict)


def build_signed_response_roles(
    *,
    prompt_uids: Sequence[Any],
    correct_flags: Sequence[bool],
    response_tpf: Sequence[float] | None,
    config: SignedTraceCEConfig,
    drop_flags: Sequence[bool] | None = None,
) -> SignedResponseRoutingResult:
    """Assign +1/-1/0 roles per response, grouped by prompt uid.

    All n rollouts are generated; this function filters post-generation.

    Per prompt uid:
    - drop_flags[i] True (e.g. finish_reason==length: the response never
      terminated, so it is neither right nor wrong) -> 0 unconditionally,
      excluded from both candidate pools.
    - correct responses: positive_mode "top1" -> exactly one highest
      response-TPF gets +1, rest 0; "all" -> every correct response gets +1.
    - wrong responses: ceil(wrong_fraction * n_wrong) highest TPF -> -1; rest -> 0

    TPF ties broken by stable rollout-index ascending (lower index preferred).
    If response_tpf is None, uses rollout-index order (lowest index first).
    """
    config.validate()
    n = len(prompt_uids)
    if len(correct_flags) != n:
        raise ValueError(
            f"signed_trace_ce alignment: prompt_uids={n} correct_flags={len(correct_flags)}"
        )
    if response_tpf is not None and len(response_tpf) != n:
        raise ValueError(
            f"signed_trace_ce alignment: prompt_uids={n} response_tpf={len(response_tpf)}"
        )
    if drop_flags is not None and len(drop_flags) != n:
        raise ValueError(
            f"signed_trace_ce alignment: prompt_uids={n} drop_flags={len(drop_flags)}"
        )

    roles = [RESPONSE_ROLE_DROP] * n

    # Group by prompt uid
    uid_groups: dict[Any, list[int]] = {}
    for i, uid in enumerate(prompt_uids):
        uid_groups.setdefault(uid, []).append(i)

    total_positive = 0
    total_negative = 0
    total_drop = 0
    total_forced_drop = 0
    total_groups_with_correct = 0
    total_groups_with_wrong = 0
    total_correct_candidates = 0
    total_wrong_candidates = 0

    for uid, indices in uid_groups.items():
        if drop_flags is not None and config.length_role == "drop":
            forced = [i for i in indices if drop_flags[i]]
            total_forced_drop += len(forced)
            indices = [i for i in indices if not drop_flags[i]]
        # length_role == "negative": length-capped rows stay in the pool and
        # are forced into the wrong bucket regardless of scorer verdict.
        if drop_flags is not None and config.length_role == "negative":
            correct_indices = [
                i for i in indices if correct_flags[i] and not drop_flags[i]
            ]
            wrong_indices = [
                i for i in indices if (not correct_flags[i]) or drop_flags[i]
            ]
        else:
            correct_indices = [i for i in indices if correct_flags[i]]
            wrong_indices = [i for i in indices if not correct_flags[i]]
        total_correct_candidates += len(correct_indices)
        total_wrong_candidates += len(wrong_indices)

        def _tpf_key(idx: int) -> tuple[float, int]:
            tpf = float(response_tpf[idx]) if response_tpf is not None else 0.0
            return (-tpf, idx)

        # positive_mode "all": every correct -> +1; "top1": one top-TPF correct.
        if correct_indices:
            total_groups_with_correct += 1
            if config.positive_mode == "all":
                for i in correct_indices:
                    roles[i] = RESPONSE_ROLE_POSITIVE
                total_positive += len(correct_indices)
            else:
                best_correct = min(correct_indices, key=_tpf_key)
                roles[best_correct] = RESPONSE_ROLE_POSITIVE
                total_positive += 1

        # Select ceil(wrong_fraction * n_wrong) top-TPF wrong -> -1
        if wrong_indices:
            total_groups_with_wrong += 1
            k_wrong = math.ceil(config.wrong_fraction * len(wrong_indices))
            sorted_wrong = sorted(wrong_indices, key=_tpf_key)
            for i in sorted_wrong[:k_wrong]:
                roles[i] = RESPONSE_ROLE_NEGATIVE
            total_negative += k_wrong

    total_drop = n - total_positive - total_negative

    metrics = {
        "signed_trace_ce/total_responses": float(n),
        "signed_trace_ce/prompt_uids": float(len(uid_groups)),
        "signed_trace_ce/correct_candidates": float(total_correct_candidates),
        "signed_trace_ce/wrong_candidates": float(total_wrong_candidates),
        "signed_trace_ce/positive_selected": float(total_positive),
        "signed_trace_ce/negative_selected": float(total_negative),
        "signed_trace_ce/dropped": float(total_drop),
        "signed_trace_ce/forced_drop_length": float(total_forced_drop),
        "signed_trace_ce/groups_with_correct": float(total_groups_with_correct),
        "signed_trace_ce/groups_with_wrong": float(total_groups_with_wrong),
        "signed_trace_ce/enabled": 1.0,
    }
    return SignedResponseRoutingResult(roles=roles, metrics=metrics)


def compute_signed_trace_ce_loss(
    *,
    positive_ce_sum: float,
    positive_token_count: int,
    negative_ce_sum: float,
    negative_token_count: int,
) -> tuple[float | None, dict[str, float]]:
    """Compute L = mean(CE_positive) - mean(CE_wrong).

    Returns (loss, metrics).  loss is None when both branches are empty
    (zero-signal batch); a batch with only positive or only negative is valid.
    Uses independent global denominators; coefficient exactly 1 for each branch.
    """
    has_positive = positive_token_count > 0
    has_negative = negative_token_count > 0

    if not has_positive and not has_negative:
        metrics = {
            "signed_trace_ce/positive_tokens": 0.0,
            "signed_trace_ce/negative_tokens": 0.0,
            "signed_trace_ce/positive_ce_mean": 0.0,
            "signed_trace_ce/negative_ce_mean": 0.0,
            "signed_trace_ce/signed_loss": 0.0,
            "signed_trace_ce/has_positive": 0.0,
            "signed_trace_ce/has_negative": 0.0,
            "signed_trace_ce/zero_signal_batch": 1.0,
        }
        return None, metrics

    pos_mean = positive_ce_sum / positive_token_count if has_positive else 0.0
    neg_mean = negative_ce_sum / negative_token_count if has_negative else 0.0
    loss = pos_mean - neg_mean

    metrics = {
        "signed_trace_ce/positive_tokens": float(positive_token_count),
        "signed_trace_ce/negative_tokens": float(negative_token_count),
        "signed_trace_ce/positive_ce_mean": float(pos_mean),
        "signed_trace_ce/negative_ce_mean": float(neg_mean),
        "signed_trace_ce/signed_loss": float(loss),
        "signed_trace_ce/has_positive": float(has_positive),
        "signed_trace_ce/has_negative": float(has_negative),
        "signed_trace_ce/zero_signal_batch": 0.0,
    }
    return loss, metrics
