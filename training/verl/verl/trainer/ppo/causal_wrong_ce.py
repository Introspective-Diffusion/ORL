"""Causal wrong-response negative CE with final-row-only state filtering.

Objective:
    L = mean_global_tokens(CE_positive_trace) - mean_global_tokens(CE_causal_wrong)

Independent denominators, coefficient 1.

Positive branch (correct responses):
    All producer-selected states + final row per block, exactly as selected by
    trace_all_ce_state_selection.  No changes to that selection logic.

Wrong branch (wrong responses):
    Exactly ONE is_final_denoise_row per (response_uid, block_index).
    The causal source for every wrong-branch token is computed with
    staircase_size=1 applied to the raw preexpanded hidden (shift_noisy_logits=False),
    which maps each noisy response token j to the preceding clean x0 hidden
    (token 0 -> last prompt hidden, token j -> clean token j-1).

Uncertain responses: dropped entirely.

Positive-only, negative-only, mixed, and zero-signal batches are all valid.
No response-level top-TPF selection is applied.

Compatibility:
    - Compatible with: trace_all_ce_state_selection (controls correct branch)
    - Incompatible with: signed_trace_ce, negative_action_pg, tpf_reinforce,
      block_tpf_ce_weight, response_tpf_ce_scale
    - Requires: trace_all_ce objective, trace_replay_all_steps=true
    - Requires: shift_noisy_logits=false in actor (handled automatically when
      causal_wrong_ce.enable=true)
    - Requires: verification_status non-tensor field from enhanced verifier reward

This module is intentionally NOT signed_trace_ce and must not share its config
keys, guards, or semantics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

# Per-response tri-state roles
TOKEN_ROLE_POSITIVE = 1       # correct response -> positive CE branch
TOKEN_ROLE_WRONG_CAUSAL = -1  # wrong response   -> causal wrong CE branch
TOKEN_ROLE_DROP = 0           # uncertain        -> excluded from both branches


@dataclass(frozen=True)
class CausalWrongCEConfig:
    """Configuration for the causal wrong CE objective."""

    enable: bool = False

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any] | None,
    ) -> "CausalWrongCEConfig":
        values = values or {}
        if not isinstance(values, Mapping):
            raise ValueError("causal_wrong_ce must be a mapping")
        return cls(enable=bool(values.get("enable", False)))

    def validate(self) -> None:
        pass


def build_causal_wrong_ce_response_roles(
    *,
    verification_statuses: Sequence[Any],
) -> list[int]:
    """Map per-response verification_status to +1/-1/0 roles.

    Returns a list of int8-compatible ints, one per response in the batch.
    'correct'  -> TOKEN_ROLE_POSITIVE (+1)
    'wrong'    -> TOKEN_ROLE_WRONG_CAUSAL (-1)
    anything else (including 'uncertain') -> TOKEN_ROLE_DROP (0)
    """
    roles: list[int] = []
    for status in verification_statuses:
        s = str(status).strip().lower() if status is not None else ""
        if s == "correct":
            roles.append(TOKEN_ROLE_POSITIVE)
        elif s == "wrong":
            roles.append(TOKEN_ROLE_WRONG_CAUSAL)
        else:
            roles.append(TOKEN_ROLE_DROP)
    return roles


def select_wrong_response_final_step_keys(
    *,
    rollout_self_conditioning: Any,
    block_size: int,
) -> set[int]:
    """Return step_keys where is_final_denoise_row=True, one per block_index.

    For wrong responses, only the final denoising state of each block is trained on.
    Every represented block must have exactly one final row. Missing or duplicate
    final markers are replay corruption and fail closed.

    Args:
        rollout_self_conditioning: The sc_b dict for one response
            (maps step_key -> item dict with metadata).
        block_size: Tokens per block, used to infer block_index from step_key
            when explicit block_idx metadata is absent.

    Returns:
        Set of integer step_keys corresponding to final denoise rows.
    """
    if not isinstance(rollout_self_conditioning, dict) or not rollout_self_conditioning:
        raise ValueError(
            "causal_wrong_ce requires non-empty replay metadata for a wrong response"
        )

    block_size = max(1, int(block_size))
    rows_by_block: dict[int, list[int]] = {}
    finals_by_block: dict[int, list[int]] = {}

    for raw_key, item in rollout_self_conditioning.items():
        if not isinstance(item, dict):
            continue
        metadata = (
            item.get("metadata")
            if isinstance(item.get("metadata"), dict)
            else {}
        )
        is_final = bool(
            item.get(
                "is_final_denoise_row",
                metadata.get("is_final_denoise_row", False),
            )
        )
        try:
            step_key = int(raw_key)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"causal_wrong_ce replay step key is not an integer: {raw_key!r}"
            ) from exc

        # Resolve block_index from metadata, falling back to step_key // block_size
        block_idx = item.get("block_idx")
        if block_idx is None:
            block_idx = metadata.get("block_idx")
        if block_idx is None:
            block_idx = metadata.get("block_index")
        if block_idx is None:
            block_idx = metadata.get("producer_block_index")
        if block_idx is None:
            block_idx = step_key // block_size
        try:
            block_idx = int(block_idx)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "causal_wrong_ce replay block index is not an integer: "
                f"{block_idx!r}"
            ) from exc
        if block_idx < 0:
            raise ValueError(
                f"causal_wrong_ce replay block index must be non-negative: {block_idx}"
            )

        rows_by_block.setdefault(block_idx, []).append(step_key)
        if is_final:
            finals_by_block.setdefault(block_idx, []).append(step_key)

    if not rows_by_block:
        raise ValueError(
            "causal_wrong_ce found no valid replay rows for a wrong response"
        )

    selected: set[int] = set()
    for block_idx in sorted(rows_by_block):
        final_keys = finals_by_block.get(block_idx, [])
        if len(final_keys) != 1:
            raise ValueError(
                "causal_wrong_ce requires exactly one final denoise row per "
                f"represented block; block={block_idx} finals={len(final_keys)} "
                f"rows={len(rows_by_block[block_idx])}"
            )
        selected.add(final_keys[0])
    return selected


def compute_causal_wrong_ce_loss(
    *,
    positive_ce_sum: float,
    positive_token_count: int,
    wrong_causal_ce_sum: float,
    wrong_causal_token_count: int,
) -> tuple[float | None, dict[str, float]]:
    """Compute L = mean(CE_positive) - mean(CE_causal_wrong).

    Independent denominators; coefficient 1 for each branch.

    Returns (loss, metrics).  loss is None when both branches are empty
    (zero-signal batch).  Positive-only or negative-only batches are valid
    and return finite losses.
    """
    has_positive = positive_token_count > 0
    has_wrong = wrong_causal_token_count > 0

    if not has_positive and not has_wrong:
        metrics: dict[str, float] = {
            "causal_wrong_ce/positive_tokens": 0.0,
            "causal_wrong_ce/wrong_tokens": 0.0,
            "causal_wrong_ce/positive_ce_mean": 0.0,
            "causal_wrong_ce/wrong_ce_mean": 0.0,
            "causal_wrong_ce/loss": 0.0,
            "causal_wrong_ce/has_positive": 0.0,
            "causal_wrong_ce/has_wrong": 0.0,
            "causal_wrong_ce/zero_signal_batch": 1.0,
        }
        return None, metrics

    pos_mean = positive_ce_sum / positive_token_count if has_positive else 0.0
    wrong_mean = (
        wrong_causal_ce_sum / wrong_causal_token_count if has_wrong else 0.0
    )
    if not math.isfinite(pos_mean) or not math.isfinite(wrong_mean):
        raise FloatingPointError(
            f"causal_wrong_ce produced non-finite means: "
            f"pos_mean={pos_mean} wrong_mean={wrong_mean}"
        )
    loss = pos_mean - wrong_mean

    metrics = {
        "causal_wrong_ce/positive_tokens": float(positive_token_count),
        "causal_wrong_ce/wrong_tokens": float(wrong_causal_token_count),
        "causal_wrong_ce/positive_ce_mean": float(pos_mean),
        "causal_wrong_ce/wrong_ce_mean": float(wrong_mean),
        "causal_wrong_ce/loss": float(loss),
        "causal_wrong_ce/has_positive": float(has_positive),
        "causal_wrong_ce/has_wrong": float(has_wrong),
        "causal_wrong_ce/zero_signal_batch": 0.0,
    }
    return loss, metrics
