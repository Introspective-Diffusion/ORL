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
    if isinstance(confidence_temperature, torch.Tensor) and confidence_temperature.dim() == 0:
        # 0-dim device scalar from the sync-free controller: same arithmetic as
        # the Python-float path (scalar cast to the logits dtype, then divide),
        # without a device->host copy.
        t0 = confidence_temperature.to(confidence_logits.dtype).clamp_min(1e-6)
        return F.softmax(confidence_logits / t0, dim=-1)
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
    light: bool = False,
    token_entropy_nats: "torch.Tensor | None" = None,
) -> tuple["float | torch.Tensor", dict[str, Any]]:
    """Compute one request's adaptive confidence temperature.

    ``token_entropy_nats`` ([tokens] fp32): the controller signal precomputed
    by the fused row kernel (fused_step_kernels.row_stats); when given, the
    full-vocab log_softmax pass here is skipped.

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

    # ------------------------------------------------------------------
    # Sync-free formulation. Every quantity that used to be a
    # Python float pulled off the device with .item() (scope count, mean
    # entropy, H_ref, the scalar temperature chain) is now a 0-dim device
    # tensor, and the masked mean is a masked sum / count instead of boolean
    # indexing (which needs nonzero() and therefore a sync). The arithmetic is
    # otherwise the one the float version performed: the scalar chain runs in
    # float64 (Python float semantics), the per-position vector in float32
    # with the same operation order. Nothing here blocks the CPU, so the
    # kernels queue up behind the model forward instead of serialising after
    # it; the per-step cost drops from ~1 ms to the ~0.1 ms of GPU work.
    # Diagnostics are 0-dim tensors too -- call
    # materialize_adaptive_confidence_temperature_diagnostics() once per block
    # before they are summarised / serialised.
    # ------------------------------------------------------------------
    dev = controller_logits.device
    t_min = float(confidence_temperature_min)
    t_max = float(confidence_temperature_max)
    gamma = float(power)
    request_key = int(req_pool_idx)
    step = int(local_step)
    f64 = dict(dtype=torch.float64, device=dev)

    scope_mask = ~prompt_mask.reshape(-1).bool()
    if confidence_temperature_mode == "adaptive_unresolved_entropy":
        scope_mask = scope_mask & ~accepted_mask.reshape(-1).bool()
    scope_count = scope_mask.sum()  # 0-dim int64, stays on device

    # Controller signal: raw logits at temperature 1.0 (never divided by T_conf).
    if token_entropy_nats is None:
        log_probs = F.log_softmax(controller_logits.float(), dim=-1)
        token_entropy_nats = -(log_probs.exp() * log_probs).sum(dim=-1)  # [tokens] fp32
    else:
        token_entropy_nats = token_entropy_nats.reshape(-1).to(torch.float32)
    log_vocab = math.log(max(int(controller_logits.shape[-1]), 2))
    # masked mean; exactly 0.0 when the scope is empty (old behaviour)
    scope_f = scope_mask.to(token_entropy_nats.dtype)
    raw_entropy_nats = torch.where(
        scope_count > 0,
        (token_entropy_nats * scope_f).sum() / scope_count.clamp_min(1).to(token_entropy_nats.dtype),
        torch.zeros((), dtype=token_entropy_nats.dtype, device=dev),
    )
    state = state_by_request.get(request_key)

    if light and per_position:
        # Decision-only fast path (no diagnostics consumer): the per-position
        # vector depends on H_ref and h_i alone, so the scalar temperature chain
        # and the metrics dict are skipped -- ~12 fewer tiny kernel launches per
        # step. State layout stays identical to the full path.
        if state is None or step <= 1:
            state = {
                "h_ref_raw_nats": raw_entropy_nats.detach().clone(),
                "h_ref_normalized": None,
                "previous_temperature": None,
                "local_step": step,
                "prev_t_vec": None,
            }
            state_by_request[request_key] = state
        else:
            state["local_step"] = step
        h_ref_pin = torch.clamp_min(state["h_ref_raw_nats"], 1e-8)
        ratio_vec = (token_entropy_nats / h_ref_pin).clamp(0.0, 1.0)
        proposed_t_vec = (t_min + (t_max - t_min) * ratio_vec.pow(gamma)).clamp(t_min, t_max)
        prompt_flat = prompt_mask.reshape(-1).bool()
        proposed_t_vec = torch.where(prompt_flat, torch.full_like(proposed_t_vec, t_max), proposed_t_vec)
        prev_t_vec = state.get("prev_t_vec")
        if not allow_reheat and prev_t_vec is not None and prev_t_vec.shape == proposed_t_vec.shape:
            t_vec = torch.minimum(prev_t_vec.to(proposed_t_vec.device), proposed_t_vec)
            t_vec = torch.where(prompt_flat, torch.full_like(t_vec, t_max), t_vec)
        else:
            t_vec = proposed_t_vec
        state["prev_t_vec"] = t_vec.detach().clone()
        return t_vec, {
            "local_step": step,
            "per_position": True,
            "light": True,
            "confidence_temperature_mode": confidence_temperature_mode,
        }

    static_metrics: dict[str, Any] = {
        "local_step": step,
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
    }

    if per_position:
        # Per-position mode: decisions depend on H_ref and h_i only. The scalar
        # temperature chain (ratio / shaped / proposed / effective / reheated)
        # is pure diagnostics here, so it is NOT computed on the device per
        # step; materialize_...() replays it in Python floats at block end
        # from the per-step raw entropies -- bitwise the numbers the old
        # implementation produced, at zero per-step launches.
        if state is None or step <= 1:
            state = {
                "h_ref_raw_nats": raw_entropy_nats.detach().clone(),
                "local_step": step,
                "prev_t_vec": None,
            }
            state_by_request[request_key] = state
        else:
            state["local_step"] = step
        h_ref_raw_nats = state["h_ref_raw_nats"]
        # (Same fp32 arithmetic as before: H_ref was a Python float that torch
        # cast to fp32 at the division; clamp_min on the fp32 tensor is identical.)
        h_ref_pin = torch.clamp_min(h_ref_raw_nats, 1e-8)
        ratio_vec = (token_entropy_nats / h_ref_pin).clamp(0.0, 1.0)
        proposed_t_vec = (t_min + (t_max - t_min) * ratio_vec.pow(gamma)).clamp(t_min, t_max)
        # Prompt positions are pinned to T_max regardless of allow_reheat.
        prompt_flat = prompt_mask.reshape(-1).bool()
        proposed_t_vec = torch.where(
            prompt_flat, torch.full_like(proposed_t_vec, t_max), proposed_t_vec
        )
        # Monotonic no-reheat (element-wise analog of min(previous, proposed)),
        # reset on a new block / shape mismatch; prompt re-pinned after the min.
        prev_t_vec = state.get("prev_t_vec")
        if (
            not allow_reheat
            and prev_t_vec is not None
            and prev_t_vec.shape == proposed_t_vec.shape
        ):
            t_vec = torch.minimum(prev_t_vec.to(proposed_t_vec.device), proposed_t_vec)
            t_vec = torch.where(prompt_flat, torch.full_like(t_vec, t_max), t_vec)
        else:
            t_vec = proposed_t_vec
        state["prev_t_vec"] = t_vec.detach().clone()

        metrics: dict[str, Any] = dict(static_metrics)
        metrics.update(
            {
                "scope_count": scope_count,
                "raw_mean_entropy_nats": raw_entropy_nats,
                "h_ref_raw_nats": h_ref_raw_nats,
                "per_position": True,
                "_lazy_scalar_chain": {
                    "t_min": t_min,
                    "t_max": t_max,
                    "gamma": gamma,
                    "allow_reheat": bool(allow_reheat),
                    "log_vocab": log_vocab,
                },
            }
        )
        if not light:
            metrics["per_position_t_mean"] = t_vec.mean()
            metrics["per_position_t_min"] = t_vec.min()
            metrics["per_position_t_max"] = t_vec.max()
        return t_vec, metrics

    # Scalar mode: the effective temperature IS the decision, so the chain runs
    # on the device every step (float64 == the former Python-float arithmetic).
    normalized_entropy = raw_entropy_nats.double() / log_vocab
    if state is None or step <= 1:
        h_ref_raw_nats = raw_entropy_nats.detach().clone()  # fp32 0-dim
        h_ref_normalized = normalized_entropy.detach().clone()
        previous_temperature = torch.full((), t_max, **f64)
        raw_h_over_h_ref = torch.ones((), **f64)
        h_over_h_ref = torch.ones((), **f64)
        shaped_ratio = torch.ones((), **f64)
        proposed_temperature = torch.full((), t_max, **f64)
        effective_temperature = torch.full((), t_max, **f64)
        reheated = torch.zeros((), dtype=torch.bool, device=dev)
        state_by_request[request_key] = {
            "h_ref_raw_nats": h_ref_raw_nats,
            "h_ref_normalized": h_ref_normalized,
            "previous_temperature": effective_temperature,
            "local_step": step,
            "prev_t_vec": None,
        }
    else:
        h_ref_raw_nats = state["h_ref_raw_nats"]
        h_ref_normalized = state["h_ref_normalized"]
        previous_temperature = state["previous_temperature"]
        raw_h_over_h_ref = torch.where(
            scope_count > 0,
            raw_entropy_nats.double() / torch.clamp_min(h_ref_raw_nats.double(), 1e-8),
            torch.zeros((), **f64),
        )
        h_over_h_ref = raw_h_over_h_ref.clamp(0.0, 1.0)
        shaped_ratio = h_over_h_ref.pow(gamma)
        proposed_temperature = (t_min + (t_max - t_min) * shaped_ratio).clamp(t_min, t_max)
        if allow_reheat:
            effective_temperature = proposed_temperature
        else:
            effective_temperature = torch.minimum(previous_temperature, proposed_temperature)
        effective_temperature = effective_temperature.clamp(t_min, t_max)
        reheated = (
            (effective_temperature > previous_temperature + 1e-12)
            if allow_reheat
            else torch.zeros((), dtype=torch.bool, device=dev)
        )
        state["previous_temperature"] = effective_temperature
        state["local_step"] = step

    metrics = dict(static_metrics)
    metrics.update(
        {
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
            # Familiar aliases make controller plots easier to compare.
            "raw_ratio": raw_h_over_h_ref,
            "ratio": h_over_h_ref,
        }
    )
    return effective_temperature, metrics


def _replay_scalar_chain(diagnostics: Sequence[dict[str, Any]]) -> None:
    """Fill the scalar-chain diagnostics of a per-position block from its
    per-step raw entropies, in Python floats -- exactly the arithmetic the
    earlier controller performed on the device round-trip each step."""
    prev_t = None
    h_ref = None
    h_ref_norm = None
    for i, item in enumerate(diagnostics):
        cfg = item.pop("_lazy_scalar_chain", None)
        if cfg is None:
            continue
        t_min, t_max, gamma = cfg["t_min"], cfg["t_max"], cfg["gamma"]
        allow_reheat, log_vocab = cfg["allow_reheat"], cfg["log_vocab"]
        raw = float(item["raw_mean_entropy_nats"])
        scope_count = int(item["scope_count"])
        normalized = raw / log_vocab
        if h_ref is None or int(item["local_step"]) <= 1:
            h_ref, h_ref_norm = raw, normalized
            previous = t_max
            raw_ratio = ratio = shaped = 1.0
            proposed = effective = t_max
            reheated = False
        else:
            previous = prev_t
            raw_ratio = 0.0 if scope_count == 0 else raw / max(h_ref, 1e-8)
            ratio = max(0.0, min(1.0, raw_ratio))
            shaped = ratio**gamma
            proposed = max(t_min, min(t_max, t_min + (t_max - t_min) * shaped))
            effective = proposed if allow_reheat else min(previous, proposed)
            effective = max(t_min, min(t_max, effective))
            reheated = bool(allow_reheat and effective > previous + 1e-12)
        prev_t = effective
        item.update(
            {
                "normalized_entropy": normalized,
                "h_ref_raw_nats": h_ref,
                "h_ref_normalized": h_ref_norm,
                "raw_h_over_h_ref": raw_ratio,
                "h_over_h_ref": ratio,
                "shaped_ratio": shaped,
                "proposed_confidence_temperature": proposed,
                "effective_confidence_temperature": effective,
                "previous_confidence_temperature": previous,
                "reheated": reheated,
                "raw_ratio": raw_ratio,
                "ratio": ratio,
            }
        )


def materialize_adaptive_confidence_temperature_diagnostics(
    *diagnostic_lists: Sequence[dict[str, Any]],
) -> None:
    """Turn the 0-dim device tensors inside per-step diagnostics into Python
    numbers in place, with a single device->host copy for everything passed,
    then fill the lazily-deferred scalar chain of each per-position block.
    Each argument must be ONE block's steps in order (the chain is sequential).
    Idempotent; call once per block before summarising / serialising."""
    slots: list[tuple[dict, str]] = []
    values: list[torch.Tensor] = []
    for diagnostics in diagnostic_lists:
        for item in diagnostics:
            if not isinstance(item, dict):
                continue
            for key, val in item.items():
                if isinstance(val, torch.Tensor) and val.dim() == 0:
                    slots.append((item, key))
                    values.append(val.detach().to(torch.float64))
    if values:
        flat = torch.stack(values).cpu().tolist()
        for (item, key), num in zip(slots, flat):
            if key in ("scope_count", "accepted_count", "new_accept_count", "edit_count"):
                item[key] = int(num)
            elif key == "reheated":
                item[key] = bool(num)
            else:
                item[key] = float(num)
    for diagnostics in diagnostic_lists:
        _replay_scalar_chain([d for d in diagnostics if isinstance(d, dict)])


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
