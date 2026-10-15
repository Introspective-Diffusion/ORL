"""Entropy-controlled confidence-temperature helpers for SDAR decoding.

This controller is deliberately separate from proposal-temperature adaptation:
it never samples proposal tokens and computes its entropy signal from raw
temperature-1 logits.
"""

from __future__ import annotations

import math
from collections.abc import MutableMapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F


ADAPTIVE_CONFIDENCE_TEMPERATURE_MODES = frozenset(
    {"adaptive_global_entropy", "adaptive_unresolved_entropy"}
)


def validate_adaptive_confidence_temperature_config(
    confidence_temperature_min: float,
    confidence_temperature_max: float,
    power: float,
) -> None:
    values = {
        "confidence_temperature_min": float(confidence_temperature_min),
        "confidence_temperature_max": float(confidence_temperature_max),
        "adaptive_confidence_temperature_power": float(power),
    }
    for name, value in values.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
    if values["confidence_temperature_min"] <= 0.0:
        raise ValueError(
            "confidence_temperature_min must be greater than zero, "
            f"got {confidence_temperature_min!r}"
        )
    if (
        values["confidence_temperature_max"]
        < values["confidence_temperature_min"]
    ):
        raise ValueError(
            "confidence_temperature_max must be greater than or equal to "
            "confidence_temperature_min, "
            f"got min={confidence_temperature_min!r} "
            f"max={confidence_temperature_max!r}"
        )
    if values["adaptive_confidence_temperature_power"] <= 0.0:
        raise ValueError(
            "adaptive_confidence_temperature_power must be greater than zero, "
            f"got {power!r}"
        )


def confidence_probs_from_logits(
    logits: torch.Tensor,
    confidence_temperature: "float | torch.Tensor",
) -> torch.Tensor:
    """Return confidence probabilities without affecting proposal sampling.

    ``confidence_temperature`` may be a scalar (uniform temperature) or a
    [tokens] tensor (per-position temperature from the entropy-gated
    controller); the vector form scales each row's logits independently.
    """

    confidence_logits = logits.float()
    if isinstance(confidence_temperature, torch.Tensor):
        t_vec = confidence_temperature.to(
            device=confidence_logits.device, dtype=confidence_logits.dtype
        ).clamp_min(1e-6)
        if t_vec.numel() != confidence_logits.shape[0]:
            raise ValueError(
                "per-position confidence temperature length mismatch: "
                f"temps={t_vec.numel()} tokens={confidence_logits.shape[0]}"
            )
        return F.softmax(confidence_logits / t_vec.view(-1, 1), dim=-1)
    temperature = max(float(confidence_temperature), 1e-6)
    if temperature != 1.0:
        confidence_logits = confidence_logits / temperature
    return F.softmax(confidence_logits, dim=-1)


def compute_adaptive_confidence_temperature(
    *,
    state_by_request: MutableMapping[int, dict[str, float | int]],
    req_pool_idx: int,
    local_step: int,
    controller_logits: torch.Tensor,
    prompt_mask: torch.Tensor,
    accepted_mask: torch.Tensor,
    confidence_temperature_min: float,
    confidence_temperature_max: float,
    power: float = 1.0,
    allow_reheat: bool = False,
    confidence_temperature_mode: str = "adaptive_global_entropy",
    per_position: bool = False,
) -> tuple["float | torch.Tensor", dict[str, Any]]:
    """Compute one request's adaptive confidence temperature.

    With ``per_position=True`` the returned temperature is a [tokens] tensor:
    T_i = T_min + (T_max - T_min) * clip(h_i / H_ref, 0, 1) ** gamma, so only
    positions whose LOCAL entropy has fallen get cooled (accepted faster)
    while still-ambiguous positions keep an honest confidence distribution.
    This is the decode-side analog of the AEC ceiling: never manufacture
    confidence where the model has legitimate uncertainty. The scalar
    controller state (H_ref anchoring, reheat bookkeeping, diagnostics)
    is maintained exactly as in scalar mode; the vector itself follows h_i
    every step, so ``allow_reheat`` does not constrain it.

    Controller entropy always uses raw logits at temperature 1.0. Global mode
    averages over every non-prompt position. Unresolved mode averages only over
    positions that are both non-prompt and unaccepted at the start of this
    denoising step, before its acceptance/edit update. Changing scope size does
    not reset H_ref: the first step's mean remains the per-block reference.
    After initialization, an empty unresolved scope has H=0, H/Href=0, and
    therefore proposes T_min. The returned temperature is used only to
    construct the confidence distribution for acceptance and edit decisions.
    """

    if (
        confidence_temperature_mode
        not in ADAPTIVE_CONFIDENCE_TEMPERATURE_MODES
    ):
        raise ValueError(
            "unsupported adaptive confidence temperature mode: "
            f"{confidence_temperature_mode!r}"
        )
    validate_adaptive_confidence_temperature_config(
        confidence_temperature_min,
        confidence_temperature_max,
        power,
    )
    if controller_logits.ndim != 2:
        raise ValueError(
            "controller_logits must have shape [tokens, vocab], "
            f"got {tuple(controller_logits.shape)}"
        )
    token_count = int(controller_logits.shape[0])
    if (
        prompt_mask.numel() != token_count
        or accepted_mask.numel() != token_count
    ):
        raise ValueError(
            "adaptive confidence temperature mask/logit shape mismatch: "
            f"tokens={token_count} prompt={prompt_mask.numel()} "
            f"accepted={accepted_mask.numel()}"
        )

    t_min = float(confidence_temperature_min)
    t_max = float(confidence_temperature_max)
    gamma = float(power)
    request_key = int(req_pool_idx)
    step = int(local_step)

    scope_mask = ~prompt_mask.reshape(-1).bool()
    if confidence_temperature_mode == "adaptive_unresolved_entropy":
        scope_mask &= ~accepted_mask.reshape(-1).bool()
    scope_count = int(scope_mask.sum().item())

    # This is the controller signal. Do not divide logits by T_conf here:
    # doing so would introduce circular feedback into the controller.
    log_probs = F.log_softmax(controller_logits.float(), dim=-1)
    controller_probs = log_probs.exp()
    token_entropy_nats = -(controller_probs * log_probs).sum(dim=-1)
    log_vocab = math.log(max(int(controller_logits.shape[-1]), 2))
    if scope_count:
        raw_entropy_nats = float(
            token_entropy_nats[scope_mask].mean().item()
        )
        normalized_entropy = raw_entropy_nats / log_vocab
    else:
        raw_entropy_nats = 0.0
        normalized_entropy = 0.0

    state = state_by_request.get(request_key)
    if state is None or step <= 1:
        h_ref_raw_nats = raw_entropy_nats
        h_ref_normalized = normalized_entropy
        previous_temperature = t_max
        raw_h_over_h_ref = 1.0
        h_over_h_ref = 1.0
        shaped_ratio = 1.0
        proposed_temperature = t_max
        effective_temperature = t_max
        reheated = False
        state_by_request[request_key] = {
            "h_ref_raw_nats": h_ref_raw_nats,
            "h_ref_normalized": h_ref_normalized,
            "previous_temperature": effective_temperature,
            "local_step": step,
            # per_position state: reset on every new block (step<=1)
            "prev_t_vec": None,
        }
    else:
        h_ref_raw_nats = float(state["h_ref_raw_nats"])
        h_ref_normalized = float(state["h_ref_normalized"])
        previous_temperature = float(state["previous_temperature"])
        if scope_count == 0:
            raw_h_over_h_ref = 0.0
        else:
            raw_h_over_h_ref = raw_entropy_nats / max(
                h_ref_raw_nats, 1e-8
            )
        h_over_h_ref = max(0.0, min(1.0, raw_h_over_h_ref))
        shaped_ratio = h_over_h_ref**gamma
        proposed_temperature = (
            t_min + (t_max - t_min) * shaped_ratio
        )
        proposed_temperature = max(
            t_min, min(t_max, proposed_temperature)
        )
        if allow_reheat:
            effective_temperature = proposed_temperature
        else:
            effective_temperature = min(
                previous_temperature, proposed_temperature
            )
        effective_temperature = max(
            t_min, min(t_max, effective_temperature)
        )
        reheated = bool(
            allow_reheat
            and effective_temperature > previous_temperature + 1e-12
        )
        state["previous_temperature"] = effective_temperature
        state["local_step"] = step

    # Refresh state reference: the step<=1 branch may have replaced the dict
    # in state_by_request; we need the current one for the per-position path.
    state = state_by_request.get(request_key)

    metrics: dict[str, Any] = {
        "local_step": step,
        "scope_count": scope_count,
        "raw_mean_entropy_nats": raw_entropy_nats,
        "normalized_entropy": normalized_entropy,
        "h_ref_raw_nats": h_ref_raw_nats,
        "h_ref_normalized": h_ref_normalized,
        "raw_h_over_h_ref": raw_h_over_h_ref,
        "h_over_h_ref": h_over_h_ref,
        "shaped_ratio": shaped_ratio,
        "proposed_confidence_temperature": proposed_temperature,
        "effective_confidence_temperature": effective_temperature,
        "previous_confidence_temperature": previous_temperature,
        "reheated": reheated,
        "adaptive_confidence_temperature_power": gamma,
        "adaptive_confidence_temperature_allow_reheat": bool(allow_reheat),
        "confidence_temperature_mode": confidence_temperature_mode,
        "controller_scope": (
            "unresolved_unaccepted_nonprompt"
            if confidence_temperature_mode
            == "adaptive_unresolved_entropy"
            else "global_all_nonprompt"
        ),
        "controller_temperature": 1.0,
        "controller_entropy_logits": "unscaled",
        # Familiar aliases make controller plots easier to compare while the
        # enclosing diagnostics field keeps proposal and confidence data
        # unambiguous.
        "raw_ratio": raw_h_over_h_ref,
        "ratio": h_over_h_ref,
    }
    if per_position:
        # Entropy-gated per-position temperature. Positions with low local
        # entropy (already decided) cool toward T_min; ambiguous positions
        # stay near T_max. Prompt positions are pinned to T_max: their
        # confidence values never drive accept/edit decisions, but pinning
        # keeps the vector semantics unambiguous.
        h_ref_pin = max(h_ref_raw_nats, 1e-8)
        ratio_vec = (token_entropy_nats / h_ref_pin).clamp(0.0, 1.0)
        proposed_t_vec = t_min + (t_max - t_min) * ratio_vec.pow(gamma)
        proposed_t_vec = proposed_t_vec.clamp(t_min, t_max)
        # Prompt positions are pinned to T_max regardless of allow_reheat.
        prompt_flat = prompt_mask.reshape(-1).bool()
        proposed_t_vec = torch.where(
            prompt_flat,
            torch.full_like(proposed_t_vec, t_max),
            proposed_t_vec,
        )

        # Monotonic no-reheat for per-position vectors.
        # When allow_reheat=False, each position's temperature must not rise
        # above its previous value. Reset on new block (step<=1) or shape mismatch.
        # Prompt positions are always pinned to T_max regardless.
        # This is the per-position analog of the scalar
        #   effective_temperature = min(previous_temperature, proposed_temperature)
        # path above.
        prev_t_vec = state.get("prev_t_vec") if state is not None else None
        if (
            not allow_reheat
            and prev_t_vec is not None
            and prev_t_vec.shape == proposed_t_vec.shape
        ):
            # Apply element-wise monotonicity: temperature can only decrease.
            t_vec = torch.minimum(prev_t_vec.to(proposed_t_vec.device), proposed_t_vec)
            # Re-pin prompt positions to T_max after the min so prompt slots
            # are never accidentally clamped by an old low-temperature entry.
            t_vec = torch.where(prompt_flat, torch.full_like(t_vec, t_max), t_vec)
        else:
            t_vec = proposed_t_vec

        # Persist the effective vector for the next step (keyed per request).
        # state is always non-None here: either it already existed or was
        # freshly created by the step<=1 branch above and re-fetched via the
        # refresh below.  The else branch is unreachable and was removed.
        state["prev_t_vec"] = t_vec.detach().clone()

        metrics["per_position"] = True
        metrics["per_position_t_mean"] = float(t_vec.mean().item())
        metrics["per_position_t_min"] = float(t_vec.min().item())
        metrics["per_position_t_max"] = float(t_vec.max().item())
        return t_vec, metrics
    return effective_temperature, metrics


def summarize_adaptive_confidence_temperature_diagnostics(
    diagnostics: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Return a JSON-safe per-block summary of confidence diagnostics."""

    if not diagnostics:
        return {}

    def values(name: str) -> list[float]:
        return [
            float(item[name])
            for item in diagnostics
            if item.get(name) is not None
            and math.isfinite(float(item[name]))
        ]

    def stats(items: list[float]) -> dict[str, float]:
        if not items:
            return {}
        return {
            "min": min(items),
            "max": max(items),
            "mean": sum(items) / len(items),
            "first": items[0],
            "last": items[-1],
        }

    names = (
        "effective_confidence_temperature",
        "proposed_confidence_temperature",
        "sampling_temperature",
        "normalized_entropy",
        "raw_mean_entropy_nats",
        "raw_h_over_h_ref",
        "h_over_h_ref",
        "shaped_ratio",
        "scope_count",
        "accepted_count",
        "new_accept_count",
        "edit_count",
    )
    summary: dict[str, Any] = {
        "step_count": len(diagnostics),
        "reheat_count": sum(
            bool(item.get("reheated")) for item in diagnostics
        ),
        "h_ref_normalized": float(
            diagnostics[0]["h_ref_normalized"]
        ),
        "h_ref_raw_nats": float(diagnostics[0]["h_ref_raw_nats"]),
        "confidence_temperature_mode": diagnostics[0].get(
            "confidence_temperature_mode"
        ),
        "controller_scope": diagnostics[0].get("controller_scope"),
        "adaptive_confidence_temperature_power": float(
            diagnostics[0]["adaptive_confidence_temperature_power"]
        ),
        "adaptive_confidence_temperature_allow_reheat": bool(
            diagnostics[0][
                "adaptive_confidence_temperature_allow_reheat"
            ]
        ),
        "controller_temperature": 1.0,
        "controller_entropy_logits": "unscaled",
    }
    for name in names:
        summary[name] = stats(values(name))
    return summary
