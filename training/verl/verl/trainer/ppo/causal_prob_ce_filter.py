"""Token-label filter driven by the existing forced-causal commit forward."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class CausalProbFilterResult:
    keep_mask: Tensor
    response_tokens: int
    kept_tokens: int


def build_causal_prob_keep_mask(
    clean_token_probs: Sequence[float | None],
    *,
    local_tail_start: int,
    local_tail_length: int,
    response_start: int,
    response_end: int,
    threshold: float,
    direction: str = "greater_than",
) -> CausalProbFilterResult:
    """Build a local-tail mask from a fixed causal-probability threshold."""

    if not math.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise ValueError(
            f"causal probability threshold must be in [0,1], got {threshold}"
        )
    if local_tail_start < 0 or local_tail_length < 0:
        raise ValueError("local tail start/length must be non-negative")
    if response_start < 0 or response_end < response_start:
        raise ValueError("invalid response span")
    direction = str(direction).strip().lower()
    if direction not in {"greater_than", "less_than"}:
        raise ValueError(
            "causal probability direction must be greater_than or less_than"
        )
    keep_mask = torch.ones((local_tail_length,), dtype=torch.bool)
    total = 0
    kept = 0
    for local_pos, tail_pos in enumerate(
        range(local_tail_start, local_tail_start + local_tail_length)
    ):
        if not (response_start <= tail_pos < response_end):
            continue
        response_pos = int(tail_pos - response_start)
        if response_pos >= len(clean_token_probs):
            raise RuntimeError(
                "causal-prob CE filter is missing commit probability for "
                f"response position {response_pos}"
            )
        raw_prob = clean_token_probs[response_pos]
        if raw_prob is None:
            raise RuntimeError(
                "causal-prob CE filter received None for response position "
                f"{response_pos}"
            )
        try:
            probability = float(raw_prob)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "causal-prob CE filter received a non-numeric probability at "
                f"response position {response_pos}: {raw_prob!r}"
            ) from exc
        if not math.isfinite(probability) or probability < 0.0 or probability > 1.0:
            raise RuntimeError(
                "causal-prob CE filter received an invalid probability at "
                f"response position {response_pos}: {probability}"
            )
        keep = (
            probability > threshold
            if direction == "greater_than"
            else probability < threshold
        )
        keep_mask[local_pos] = keep
        total += 1
        kept += int(keep)
    return CausalProbFilterResult(
        keep_mask=keep_mask,
        response_tokens=total,
        kept_tokens=kept,
    )


__all__ = [
    "CausalProbFilterResult",
    "build_causal_prob_keep_mask",
]
