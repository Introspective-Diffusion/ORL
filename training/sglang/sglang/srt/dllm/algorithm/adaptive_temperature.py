"""Entropy-controlled proposal-temperature helpers for SDAR decoding."""

from __future__ import annotations

import math
from collections.abc import MutableMapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F


ADAPTIVE_TEMPERATURE_MODES = frozenset(
    {"adaptive_global_entropy", "adaptive_unresolved_entropy"}
)


def validate_adaptive_temperature_config(
    temperature_min: float,
    temperature_max: float,
    power: float,
) -> None:
    values = {
        "temperature_min": float(temperature_min),
        "temperature_max": float(temperature_max),
        "adaptive_temperature_power": float(power),
    }
    for name, value in values.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
    if values["temperature_min"] < 0.0:
        raise ValueError(
            f"temperature_min must be non-negative, got {temperature_min!r}"
        )
    if values["temperature_max"] < values["temperature_min"]:
        raise ValueError(
            "temperature_max must be greater than or equal to temperature_min, "
            f"got min={temperature_min!r} max={temperature_max!r}"
        )
    if values["adaptive_temperature_power"] <= 0.0:
        raise ValueError(
            "adaptive_temperature_power must be greater than zero, "
            f"got {power!r}"
        )


def compute_adaptive_temperature(
    *,
    state_by_request: MutableMapping[int, dict[str, float | int]],
    req_pool_idx: int,
    local_step: int,
    controller_logits: torch.Tensor,
    prompt_mask: torch.Tensor,
    accepted_mask: torch.Tensor,
    temperature_mode: str,
    temperature_min: float,
    temperature_max: float,
    power: float = 1.0,
    allow_reheat: bool = False,
) -> tuple[float, dict[str, Any]]:
    """Compute one request's adaptive proposal temperature.

    ``controller_logits`` are always converted to probabilities at temperature
    1.0 here. Acceptance confidence and proposal sampling distributions remain
    separate concerns.
    """

    if temperature_mode not in ADAPTIVE_TEMPERATURE_MODES:
        raise ValueError(
            f"unsupported adaptive temperature mode: {temperature_mode!r}"
        )
    validate_adaptive_temperature_config(
        temperature_min, temperature_max, power
    )
    if controller_logits.ndim != 2:
        raise ValueError(
            "controller_logits must have shape [tokens, vocab], "
            f"got {tuple(controller_logits.shape)}"
        )
    token_count = int(controller_logits.shape[0])
    if prompt_mask.numel() != token_count or accepted_mask.numel() != token_count:
        raise ValueError(
            "adaptive temperature mask/logit shape mismatch: "
            f"tokens={token_count} prompt={prompt_mask.numel()} "
            f"accepted={accepted_mask.numel()}"
        )

    t_min = float(temperature_min)
    t_max = float(temperature_max)
    gamma = float(power)
    request_key = int(req_pool_idx)
    step = int(local_step)

    scope_mask = ~prompt_mask.reshape(-1).bool()
    if temperature_mode == "adaptive_unresolved_entropy":
        scope_mask &= ~accepted_mask.reshape(-1).bool()
    scope_count = int(scope_mask.sum().item())

    # Controller entropy is intentionally based on unscaled logits.
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
        raw_ratio = 1.0
        ratio = 1.0
        shaped_ratio = 1.0
        proposed_temperature = t_max
        effective_temperature = t_max
        reheated = False
        state = {
            "h_ref_raw_nats": h_ref_raw_nats,
            "h_ref_normalized": h_ref_normalized,
            "previous_temperature": effective_temperature,
            "local_step": step,
        }
        state_by_request[request_key] = state
    else:
        h_ref_raw_nats = float(state["h_ref_raw_nats"])
        h_ref_normalized = float(state["h_ref_normalized"])
        previous_temperature = float(state["previous_temperature"])
        if scope_count == 0:
            raw_ratio = 0.0
            ratio = 0.0
        else:
            raw_ratio = normalized_entropy / max(h_ref_normalized, 1e-8)
            ratio = max(0.0, min(1.0, raw_ratio))
        shaped_ratio = ratio**gamma
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

    metrics: dict[str, Any] = {
        "local_step": step,
        "scope_count": scope_count,
        "raw_mean_entropy_nats": raw_entropy_nats,
        "normalized_entropy": normalized_entropy,
        "h_ref_raw_nats": h_ref_raw_nats,
        "h_ref_normalized": h_ref_normalized,
        "raw_ratio": raw_ratio,
        "ratio": ratio,
        "shaped_ratio": shaped_ratio,
        "proposed_temperature": proposed_temperature,
        "effective_temperature": effective_temperature,
        "previous_temperature": previous_temperature,
        "reheated": reheated,
        "adaptive_temperature_power": gamma,
        "adaptive_temperature_allow_reheat": bool(allow_reheat),
        "temperature_mode": temperature_mode,
        # Compatibility aliases for the original instrumentation names.
        "controller_scope_count": scope_count,
        "raw_mean_entropy": raw_entropy_nats,
        "ref_entropy": h_ref_normalized,
    }
    return effective_temperature, metrics


def summarize_adaptive_temperature_diagnostics(
    diagnostics: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Return a JSON-safe per-block summary of adaptive step diagnostics."""

    if not diagnostics:
        return {}

    def values(name: str) -> list[float]:
        return [
            float(item[name])
            for item in diagnostics
            if item.get(name) is not None
            and math.isfinite(float(item[name]))
        ]

    effective = values("effective_temperature")
    proposed = values("proposed_temperature")
    entropy = values("normalized_entropy")
    raw_entropy = values("raw_mean_entropy_nats")
    raw_ratio = values("raw_ratio")
    shaped_ratio = values("shaped_ratio")

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

    return {
        "step_count": len(diagnostics),
        "reheat_count": sum(bool(item.get("reheated")) for item in diagnostics),
        "effective_temperature": stats(effective),
        "proposed_temperature": stats(proposed),
        "normalized_entropy": stats(entropy),
        "raw_mean_entropy_nats": stats(raw_entropy),
        "raw_ratio": stats(raw_ratio),
        "shaped_ratio": stats(shaped_ratio),
        "h_ref_normalized": float(diagnostics[0]["h_ref_normalized"]),
        "h_ref_raw_nats": float(diagnostics[0]["h_ref_raw_nats"]),
        "temperature_mode": diagnostics[0].get("temperature_mode"),
        "adaptive_temperature_power": float(
            diagnostics[0]["adaptive_temperature_power"]
        ),
        "adaptive_temperature_allow_reheat": bool(
            diagnostics[0]["adaptive_temperature_allow_reheat"]
        ),
    }
