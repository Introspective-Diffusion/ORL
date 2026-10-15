"""Conflict-free TPF replay scoring utilities for non-stream diffusion RL.

This module is intentionally independent of actor, rollout, and SGLang classes.
It consumes Python row records, computes row rewards/advantages on CPU, and
returns tensor-friendly outputs that can later be attached to trainer batches.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Mapping, Protocol, Sequence

import torch


DEFAULT_IDENTITY_FIELDS = (
    "request_uid",
    "sample_uid",
    "response_uid",
    "block_index",
    "step_index",
)
DEFAULT_GROUP_KEY_FIELDS = ("response_uid", "block_index")
DEFAULT_TIE_BREAK_FIELDS = (
    "response_uid",
    "block_index",
    "step_index",
    "request_uid",
    "sample_uid",
)
DEFAULT_REWARD_COUNT_FIELDS = (
    "new_accept_count",
    "new_accept_tokens",
    "accepted_token_count",
    "accepted_tokens",
)
DEFAULT_FINAL_STEP_FIELDS = (
    "is_final_denoise_row",
    "final_denoise_row",
    "is_final_step",
)
DEFAULT_TERMINAL_COMMIT_FIELDS = (
    "is_terminal_commit",
    "terminal_commit",
    "is_commit_pass",
    "commit_pass",
)


@dataclass(frozen=True)
class DiffusionTPFReplayRow:
    """Minimal row contract for one denoise forward.

    The default reward is ``new_accept_count`` unless this row is marked as a
    terminal commit pass. Extra producer metadata can be carried in ``metadata``.
    """

    request_uid: str
    sample_uid: str
    response_uid: str
    block_index: int
    step_index: int
    new_accept_count: int | float
    is_final_denoise_row: bool = False
    is_terminal_commit: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_mapping(self) -> dict[str, Any]:
        """Return a plain mapping with metadata preserved under explicit fields."""

        values = dict(self.metadata)
        values.update(
            {
                "request_uid": self.request_uid,
                "sample_uid": self.sample_uid,
                "response_uid": self.response_uid,
                "block_index": self.block_index,
                "step_index": self.step_index,
                "new_accept_count": self.new_accept_count,
                "is_final_denoise_row": self.is_final_denoise_row,
                "is_terminal_commit": self.is_terminal_commit,
            }
        )
        return values


RowRecord = Mapping[str, Any] | DiffusionTPFReplayRow


@dataclass(frozen=True)
class DiffusionTPFReplayConfig:
    """Configuration for the row-level filtered REINFORCE core."""

    group_key_fields: tuple[str, ...] = DEFAULT_GROUP_KEY_FIELDS
    identity_fields: tuple[str, ...] = DEFAULT_IDENTITY_FIELDS
    tie_break_fields: tuple[str, ...] = DEFAULT_TIE_BREAK_FIELDS
    reward_count_fields: tuple[str, ...] = DEFAULT_REWARD_COUNT_FIELDS
    final_step_fields: tuple[str, ...] = DEFAULT_FINAL_STEP_FIELDS
    terminal_commit_fields: tuple[str, ...] = DEFAULT_TERMINAL_COMMIT_FIELDS
    fraction: float = 0.10
    eps: float = 1e-8
    min_group_size: int = 2
    final_step_policy: str = "flag"
    quality_gate_field: str = ""
    nonzero_advantage_only: bool = False
    metric_prefix: str = "diffusion_tpf_replay"


@dataclass(frozen=True)
class DiffusionTPFGroupStats:
    """Group-level reward statistics used to compute advantages."""

    key: tuple[Any, ...]
    key_dict: dict[str, Any]
    row_indices: tuple[int, ...]
    size: int
    reward_mean: float
    reward_variance: float
    reward_std: float
    short_group: bool
    zero_variance: bool


@dataclass(frozen=True)
class DiffusionTPFReplayOutput:
    """Tensor-friendly output from the orchestration function."""

    rows: list[dict[str, Any]]
    rewards: torch.Tensor
    advantages: torch.Tensor
    selected_mask: torch.Tensor
    final_mask: torch.Tensor
    top_fraction_mask: torch.Tensor
    forced_final_mask: torch.Tensor
    group_stats: list[DiffusionTPFGroupStats]
    metrics: dict[str, float]


@dataclass(frozen=True)
class DiffusionTPFMaterializedReplay:
    """Selected replay payloads loaded after metadata-only row selection."""

    selected_indices: list[int]
    payloads: list[dict[str, Any]]
    metrics: dict[str, float]


@dataclass(frozen=True)
class AdvantageResult:
    """Advantage values and the group statistics used to produce them."""

    advantages: list[float]
    group_stats: list[DiffusionTPFGroupStats]


@dataclass(frozen=True)
class SelectionResult:
    """Selector masks before conversion to tensors."""

    selected_mask: list[bool]
    final_mask: list[bool]
    top_fraction_mask: list[bool]
    forced_final_mask: list[bool]


def compute_tpf_reinforce_token_mean_loss(
    action_log_probs: torch.Tensor,
    action_mask: torch.Tensor,
    row_advantages: torch.Tensor,
    *,
    normalization_denominator: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Return ``sum_i -A_i * sum_t log pi(a_it|s_i)`` averaged by action tokens."""

    if action_log_probs.shape != action_mask.shape:
        raise ValueError(
            "action_log_probs and action_mask must have the same shape, got "
            f"{tuple(action_log_probs.shape)} and {tuple(action_mask.shape)}"
        )
    if row_advantages.dim() != 1 or int(row_advantages.shape[0]) != int(action_log_probs.shape[0]):
        raise ValueError(
            "row_advantages must be [rows] and match action_log_probs rows, got "
            f"{tuple(row_advantages.shape)} for action_log_probs={tuple(action_log_probs.shape)}"
        )
    mask = action_mask.to(device=action_log_probs.device, dtype=action_log_probs.dtype)
    advantages = torch.nan_to_num(
        row_advantages.to(device=action_log_probs.device, dtype=action_log_probs.dtype),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if normalization_denominator is None:
        token_count = mask.sum().clamp_min(1.0)
    else:
        token_count = torch.as_tensor(
            normalization_denominator,
            device=action_log_probs.device,
            dtype=action_log_probs.dtype,
        )
        if token_count.numel() != 1:
            raise ValueError("normalization_denominator must be a scalar")
        if not bool(torch.isfinite(token_count).item()) or float(token_count.item()) <= 0.0:
            raise ValueError("normalization_denominator must be finite and positive")
    return -(action_log_probs * mask * advantages.unsqueeze(-1)).sum() / token_count


class RewardProvider(Protocol):
    """Interface for pluggable row reward algorithms."""

    def rewards(self, rows: Sequence[Mapping[str, Any]], config: DiffusionTPFReplayConfig) -> list[float]:
        """Return one finite reward per row."""


class AdvantageEstimator(Protocol):
    """Interface for pluggable advantage algorithms."""

    def estimate(
        self,
        rows: Sequence[Mapping[str, Any]],
        rewards: Sequence[float],
        config: DiffusionTPFReplayConfig,
    ) -> AdvantageResult:
        """Return one advantage per row and per-group statistics."""


class RowSelector(Protocol):
    """Interface for pluggable row filters."""

    def select(
        self,
        rows: Sequence[Mapping[str, Any]],
        advantages: Sequence[float],
        group_stats: Sequence[DiffusionTPFGroupStats],
        config: DiffusionTPFReplayConfig,
    ) -> SelectionResult:
        """Return selected/top/final masks."""


class NewAcceptCountRewardProvider:
    """Default TPF reward: newly accepted action tokens per denoise forward."""

    def rewards(self, rows: Sequence[Mapping[str, Any]], config: DiffusionTPFReplayConfig) -> list[float]:
        values: list[float] = []
        for row_index, row in enumerate(rows):
            if _row_bool(row, config.terminal_commit_fields, default=False, row_index=row_index):
                values.append(0.0)
                continue

            raw_count = _first_present(row, config.reward_count_fields)
            if raw_count is _MISSING:
                raise ValueError(
                    "missing reward count field for row "
                    f"{row_index}; expected one of {config.reward_count_fields}"
                )
            count = _validate_count(raw_count, row_index=row_index)
            values.append(float(count))
        return values


class GroupRelativeZScoreAdvantageEstimator:
    """Group-normalized heuristic advantage estimator.

    For each group, advantages are ``(reward - mean) / sqrt(variance + eps)``.
    Groups smaller than ``min_group_size`` or with variance ``<= eps`` receive
    zero advantages.
    """

    def estimate(
        self,
        rows: Sequence[Mapping[str, Any]],
        rewards: Sequence[float],
        config: DiffusionTPFReplayConfig,
    ) -> AdvantageResult:
        _validate_reward_values(rewards, expected_len=len(rows))
        groups = _group_indices(rows, config.group_key_fields)
        advantages = [0.0 for _ in rows]
        group_stats: list[DiffusionTPFGroupStats] = []

        for key, indices in groups.items():
            group_rewards = [float(rewards[index]) for index in indices]
            mean = sum(group_rewards) / len(group_rewards)
            variance = sum((reward - mean) ** 2 for reward in group_rewards) / len(group_rewards)
            short_group = len(indices) < config.min_group_size
            zero_variance = variance <= config.eps
            std = math.sqrt(variance + config.eps)

            if not short_group and not zero_variance:
                for index in indices:
                    advantages[index] = (float(rewards[index]) - mean) / std

            group_stats.append(
                DiffusionTPFGroupStats(
                    key=key,
                    key_dict={field_name: value for field_name, value in zip(config.group_key_fields, key, strict=True)},
                    row_indices=tuple(indices),
                    size=len(indices),
                    reward_mean=mean,
                    reward_variance=variance,
                    reward_std=std,
                    short_group=short_group,
                    zero_variance=zero_variance,
                )
            )

        return AdvantageResult(advantages=advantages, group_stats=group_stats)


class TopFractionPlusFinalRowSelector:
    """Select per-group top rows by advantage, then union final denoise rows."""

    def select(
        self,
        rows: Sequence[Mapping[str, Any]],
        advantages: Sequence[float],
        group_stats: Sequence[DiffusionTPFGroupStats],
        config: DiffusionTPFReplayConfig,
    ) -> SelectionResult:
        if len(advantages) != len(rows):
            raise ValueError(f"expected {len(rows)} advantages, got {len(advantages)}")
        _validate_reward_values(advantages, expected_len=len(rows), label="advantage")

        final_mask = _compute_final_mask(rows, group_stats, config)
        eligible_mask = _compute_eligibility_mask(rows, config)
        top_mask = [False for _ in rows]

        for stats in group_stats:
            k = int(math.ceil(config.fraction * stats.size))
            ranked = sorted(
                (
                    index
                    for index in stats.row_indices
                    if eligible_mask[index]
                    and (
                        not config.nonzero_advantage_only
                        or abs(float(advantages[index])) > float(config.eps)
                    )
                ),
                key=lambda index: (
                    -float(advantages[index]),
                    *(_tie_value(rows[index], field_name, index) for field_name in config.tie_break_fields),
                    index,
                ),
            )
            for index in ranked[:k]:
                top_mask[index] = True

        selected_final_mask = [
            final
            and eligible
            and (
                not config.nonzero_advantage_only
                or abs(float(advantage)) > float(config.eps)
            )
            for final, eligible, advantage in zip(final_mask, eligible_mask, advantages, strict=True)
        ]
        selected_mask = [
            top or final
            for top, final in zip(top_mask, selected_final_mask, strict=True)
        ]
        forced_final_mask = [
            selected and final and not top
            for selected, final, top in zip(selected_mask, selected_final_mask, top_mask, strict=True)
        ]
        return SelectionResult(
            selected_mask=selected_mask,
            final_mask=final_mask,
            top_fraction_mask=top_mask,
            forced_final_mask=forced_final_mask,
        )


def compute_diffusion_tpf_replay(
    rows: Sequence[RowRecord],
    *,
    config: DiffusionTPFReplayConfig | None = None,
    reward_provider: RewardProvider | None = None,
    advantage_estimator: AdvantageEstimator | None = None,
    row_selector: RowSelector | None = None,
) -> DiffusionTPFReplayOutput:
    """Compute rewards, advantages, row-selection masks, stats, and metrics.

    This is a pure CPU orchestration point. It performs validation up front,
    delegates reward/advantage/selection behavior to pluggable components, and
    returns CPU tensors for trainer-side attachment.
    """

    config = config or DiffusionTPFReplayConfig()
    _validate_config(config)
    normalized_rows = _normalize_rows(rows)
    _validate_rows(normalized_rows, config)

    reward_provider = reward_provider or NewAcceptCountRewardProvider()
    advantage_estimator = advantage_estimator or GroupRelativeZScoreAdvantageEstimator()
    row_selector = row_selector or TopFractionPlusFinalRowSelector()

    rewards = reward_provider.rewards(normalized_rows, config)
    _validate_reward_values(rewards, expected_len=len(normalized_rows))

    advantage_result = advantage_estimator.estimate(normalized_rows, rewards, config)
    _validate_reward_values(advantage_result.advantages, expected_len=len(normalized_rows), label="advantage")

    selection = row_selector.select(
        normalized_rows,
        advantage_result.advantages,
        advantage_result.group_stats,
        config,
    )
    _validate_selection(selection, expected_len=len(normalized_rows))

    reward_tensor = torch.tensor(rewards, dtype=torch.float32, device="cpu")
    advantage_tensor = torch.tensor(advantage_result.advantages, dtype=torch.float32, device="cpu")
    selected_tensor = torch.tensor(selection.selected_mask, dtype=torch.bool, device="cpu")
    final_tensor = torch.tensor(selection.final_mask, dtype=torch.bool, device="cpu")
    top_tensor = torch.tensor(selection.top_fraction_mask, dtype=torch.bool, device="cpu")
    forced_final_tensor = torch.tensor(selection.forced_final_mask, dtype=torch.bool, device="cpu")
    eligible_tensor = torch.tensor(_compute_eligibility_mask(normalized_rows, config), dtype=torch.bool, device="cpu")

    metrics = _build_metrics(
        rewards=reward_tensor,
        advantages=advantage_tensor,
        selected_mask=selected_tensor,
        top_fraction_mask=top_tensor,
        forced_final_mask=forced_final_tensor,
        eligible_mask=eligible_tensor,
        group_stats=advantage_result.group_stats,
        prefix=config.metric_prefix,
    )
    return DiffusionTPFReplayOutput(
        rows=normalized_rows,
        rewards=reward_tensor,
        advantages=advantage_tensor,
        selected_mask=selected_tensor,
        final_mask=final_tensor,
        top_fraction_mask=top_tensor,
        forced_final_mask=forced_final_tensor,
        group_stats=advantage_result.group_stats,
        metrics=metrics,
    )


def materialize_selected_tpf_replay_handles(
    handles: Sequence[Mapping[str, Any]],
    replay_output: DiffusionTPFReplayOutput,
    *,
    trace_root: str | None = None,
    load_fn: Any = torch.load,
    cleanup: bool = True,
) -> DiffusionTPFMaterializedReplay:
    """Load only rows selected by ``compute_diffusion_tpf_replay``.

    Metadata-first replay keeps one lightweight handle per denoise row until
    row selection is known.  This helper validates every handle cheaply, loads
    only selected tensor files, then deletes all per-step files in the scoped
    trace directory.
    """

    if len(handles) != int(replay_output.selected_mask.numel()):
        raise ValueError(
            "metadata handle count must match TPF replay rows, got "
            f"{len(handles)} handles for {int(replay_output.selected_mask.numel())} rows"
        )

    selected_mask = replay_output.selected_mask.detach().cpu().bool().view(-1)
    selected_indices = [idx for idx, selected in enumerate(selected_mask.tolist()) if selected]
    final_rows = int(replay_output.final_mask.detach().cpu().bool().sum().item())
    if final_rows < len(replay_output.group_stats):
        raise ValueError(
            "TPF metadata-first final-row coverage is incomplete: "
            f"{final_rows}/{len(replay_output.group_stats)} groups"
        )

    root = _resolve_trace_root(trace_root, handles)
    paths = [_resolve_handle_tensor_path(handle, root, row_index=i) for i, handle in enumerate(handles)]
    identities: set[tuple[Any, ...]] = set()
    total_bytes = 0
    selected_declared_bytes = 0
    for row_index, (handle, path) in enumerate(zip(handles, paths, strict=True)):
        identity = _handle_identity(handle, row_index=row_index)
        if identity in identities:
            raise ValueError(f"duplicate TPF metadata replay handle identity {identity!r}")
        identities.add(identity)
        declared_bytes = _handle_declared_bytes(handle, path)
        total_bytes += declared_bytes
        if bool(selected_mask[row_index]):
            selected_declared_bytes += declared_bytes

    payloads: list[dict[str, Any]] = []
    load_start = time.perf_counter()
    peak_selected_bytes = 0
    try:
        for row_index in selected_indices:
            handle = handles[row_index]
            path = paths[row_index]
            if not os.path.exists(path):
                raise FileNotFoundError(f"selected TPF replay tensor file is missing: {path}")
            payload = load_fn(path, map_location="cpu")
            if not isinstance(payload, dict):
                raise ValueError(f"selected TPF replay tensor file did not contain a mapping: {path}")
            _validate_materialized_tpf_payload(payload, handle, row_index=row_index)
            payloads.append(payload)
            peak_selected_bytes = max(peak_selected_bytes, _payload_tensor_bytes(payload))
    finally:
        deleted, delete_errors = _cleanup_tpf_replay_files(paths, cleanup=cleanup)

    load_seconds = time.perf_counter() - load_start
    selected_files = len(selected_indices)
    metrics = {
        "diffusion_tpf_replay/metadata_rows": float(len(handles)),
        "diffusion_tpf_replay/total_files": float(len(paths)),
        "diffusion_tpf_replay/selected_files": float(selected_files),
        "diffusion_tpf_replay/total_file_bytes": float(total_bytes),
        "diffusion_tpf_replay/selected_file_bytes": float(selected_declared_bytes),
        "diffusion_tpf_replay/avoided_file_bytes": float(max(0, total_bytes - selected_declared_bytes)),
        "diffusion_tpf_replay/avoided_file_fraction": (
            float(max(0, total_bytes - selected_declared_bytes) / total_bytes)
            if total_bytes > 0
            else 0.0
        ),
        "diffusion_tpf_replay/selected_file_fraction": (
            float(selected_files / len(paths)) if paths else 0.0
        ),
        "diffusion_tpf_replay/materialize_load_seconds": float(load_seconds),
        "diffusion_tpf_replay/peak_selected_file_bytes": float(peak_selected_bytes),
        "diffusion_tpf_replay/deleted_files": float(deleted),
        "diffusion_tpf_replay/delete_errors": float(delete_errors),
        "diffusion_tpf_replay/final_rows": float(final_rows),
        "diffusion_tpf_replay/per_block_final_coverage": (
            float(final_rows / max(1, len(replay_output.group_stats)))
        ),
    }
    return DiffusionTPFMaterializedReplay(
        selected_indices=selected_indices,
        payloads=payloads,
        metrics=metrics,
    )


_MISSING = object()


def _normalize_rows(rows: Sequence[RowRecord]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, DiffusionTPFReplayRow):
            normalized.append(row.as_mapping())
        elif isinstance(row, Mapping):
            normalized.append(dict(row))
        elif is_dataclass(row):
            normalized.append(asdict(row))
        else:
            raise TypeError(f"row records must be mappings or dataclasses, got {type(row)!r}")
    return normalized


def _validate_config(config: DiffusionTPFReplayConfig) -> None:
    if not config.group_key_fields:
        raise ValueError("group_key_fields must be non-empty")
    if not config.identity_fields:
        raise ValueError("identity_fields must be non-empty")
    if not config.tie_break_fields:
        raise ValueError("tie_break_fields must be non-empty")
    if not 0.0 < float(config.fraction) <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {config.fraction!r}")
    if not math.isfinite(float(config.eps)) or config.eps <= 0:
        raise ValueError(f"eps must be finite and > 0, got {config.eps!r}")
    if config.min_group_size < 1:
        raise ValueError(f"min_group_size must be >= 1, got {config.min_group_size!r}")
    if config.final_step_policy not in {"none", "flag", "max_step_index", "flag_or_max_step_index"}:
        raise ValueError(
            "final_step_policy must be one of none, flag, max_step_index, "
            f"flag_or_max_step_index; got {config.final_step_policy!r}"
        )
    if not isinstance(config.quality_gate_field, str):
        raise ValueError(f"quality_gate_field must be a string, got {config.quality_gate_field!r}")


def _validate_rows(rows: Sequence[Mapping[str, Any]], config: DiffusionTPFReplayConfig) -> None:
    identities: set[tuple[Any, ...]] = set()
    for row_index, row in enumerate(rows):
        identity = tuple(_require_field(row, field_name, row_index) for field_name in config.identity_fields)
        if identity in identities:
            raise ValueError(f"duplicate TPF replay row identity {identity!r}")
        identities.add(identity)
        for field_name in config.group_key_fields:
            _require_field(row, field_name, row_index)

        for field_name in config.terminal_commit_fields:
            if field_name in row:
                _validate_bool_like(row[field_name], field_name=field_name, row_index=row_index)
        for field_name in config.final_step_fields:
            if field_name in row:
                _validate_bool_like(row[field_name], field_name=field_name, row_index=row_index)
        if config.quality_gate_field and config.quality_gate_field in row:
            _validate_bool_like(
                row[config.quality_gate_field],
                field_name=config.quality_gate_field,
                row_index=row_index,
            )


def _require_field(row: Mapping[str, Any], field_name: str, row_index: int) -> Any:
    if field_name not in row or row[field_name] is None:
        raise ValueError(f"row {row_index} is missing required field {field_name!r}")
    return row[field_name]


def _first_present(row: Mapping[str, Any], field_names: Sequence[str]) -> Any:
    for field_name in field_names:
        if field_name in row and row[field_name] is not None:
            return row[field_name]
    return _MISSING


def _validate_count(value: Any, *, row_index: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"row {row_index} reward count must be numeric, got bool")
    if not isinstance(value, int | float):
        raise ValueError(f"row {row_index} reward count must be numeric, got {type(value)!r}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"row {row_index} reward count must be finite, got {value!r}")
    if numeric < 0:
        raise ValueError(f"row {row_index} reward count must be non-negative, got {value!r}")
    if not numeric.is_integer():
        raise ValueError(f"row {row_index} reward count must be an integer token count, got {value!r}")
    return int(numeric)


def _validate_reward_values(
    values: Sequence[float],
    *,
    expected_len: int,
    label: str = "reward",
) -> None:
    if len(values) != expected_len:
        raise ValueError(f"expected {expected_len} {label}s, got {len(values)}")
    for index, value in enumerate(values):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{label} for row {index} must be finite, got {value!r}")


def _group_indices(rows: Sequence[Mapping[str, Any]], group_key_fields: Sequence[str]) -> dict[tuple[Any, ...], list[int]]:
    groups: dict[tuple[Any, ...], list[int]] = {}
    for row_index, row in enumerate(rows):
        key = tuple(_require_field(row, field_name, row_index) for field_name in group_key_fields)
        try:
            groups.setdefault(key, []).append(row_index)
        except TypeError as exc:
            raise ValueError(f"group key for row {row_index} must be hashable, got {key!r}") from exc
    return groups


def _row_bool(
    row: Mapping[str, Any],
    field_names: Sequence[str],
    *,
    default: bool,
    row_index: int,
) -> bool:
    value = _first_present(row, field_names)
    if value is _MISSING:
        return default
    return _validate_bool_like(value, field_name="/".join(field_names), row_index=row_index)


def _compute_eligibility_mask(
    rows: Sequence[Mapping[str, Any]],
    config: DiffusionTPFReplayConfig,
) -> list[bool]:
    if not config.quality_gate_field:
        return [True for _ in rows]
    return [
        _row_bool(
            row,
            (config.quality_gate_field,),
            default=True,
            row_index=row_index,
        )
        for row_index, row in enumerate(rows)
    ]


def _validate_bool_like(value: Any, *, field_name: str, row_index: int) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    raise ValueError(f"row {row_index} field {field_name!r} must be boolean-like, got {value!r}")


def _compute_final_mask(
    rows: Sequence[Mapping[str, Any]],
    group_stats: Sequence[DiffusionTPFGroupStats],
    config: DiffusionTPFReplayConfig,
) -> list[bool]:
    if config.final_step_policy == "none":
        return [False for _ in rows]

    final_mask = [False for _ in rows]
    if config.final_step_policy in {"flag", "flag_or_max_step_index"}:
        for row_index, row in enumerate(rows):
            if _row_bool(row, config.terminal_commit_fields, default=False, row_index=row_index):
                continue
            if _row_bool(row, config.final_step_fields, default=False, row_index=row_index):
                final_mask[row_index] = True
        _validate_at_most_one_final_per_group(final_mask, group_stats)

    if config.final_step_policy in {"max_step_index", "flag_or_max_step_index"}:
        max_step_mask = _max_step_final_mask(rows, group_stats, config)
        final_mask = [
            flagged or max_step
            for flagged, max_step in zip(final_mask, max_step_mask, strict=True)
        ]
        _validate_at_most_one_final_per_group(final_mask, group_stats)

    return final_mask


def _validate_at_most_one_final_per_group(
    final_mask: Sequence[bool],
    group_stats: Sequence[DiffusionTPFGroupStats],
) -> None:
    for stats in group_stats:
        final_count = sum(1 for index in stats.row_indices if final_mask[index])
        if final_count > 1:
            raise ValueError(f"group {stats.key!r} has multiple final denoise rows")


def _max_step_final_mask(
    rows: Sequence[Mapping[str, Any]],
    group_stats: Sequence[DiffusionTPFGroupStats],
    config: DiffusionTPFReplayConfig,
) -> list[bool]:
    final_mask = [False for _ in rows]
    for stats in group_stats:
        candidates = [
            index
            for index in stats.row_indices
            if not _row_bool(rows[index], config.terminal_commit_fields, default=False, row_index=index)
        ]
        if not candidates:
            continue
        ranked = sorted(
            candidates,
            key=lambda index: (
                -_numeric_step_index(rows[index], index),
                *(_tie_value(rows[index], field_name, index) for field_name in config.tie_break_fields),
                index,
            ),
        )
        final_mask[ranked[0]] = True
    return final_mask


def _numeric_step_index(row: Mapping[str, Any], row_index: int) -> float:
    value = _require_field(row, "step_index", row_index)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"row {row_index} step_index must be numeric, got {value!r}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"row {row_index} step_index must be finite, got {value!r}")
    return numeric


def _tie_value(row: Mapping[str, Any], field_name: str, row_index: int) -> tuple[int, Any]:
    value = _require_field(row, field_name, row_index)
    if isinstance(value, bool):
        return (0, int(value))
    if isinstance(value, int | float):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"row {row_index} tie-break field {field_name!r} must be finite")
        return (1, numeric)
    if isinstance(value, str):
        return (2, value)
    return (3, repr(value))


def _validate_selection(selection: SelectionResult, *, expected_len: int) -> None:
    for label, values in (
        ("selected_mask", selection.selected_mask),
        ("final_mask", selection.final_mask),
        ("top_fraction_mask", selection.top_fraction_mask),
        ("forced_final_mask", selection.forced_final_mask),
    ):
        if len(values) != expected_len:
            raise ValueError(f"expected {expected_len} values for {label}, got {len(values)}")
        if not all(isinstance(value, bool) for value in values):
            raise ValueError(f"{label} must contain booleans")


def _build_metrics(
    *,
    rewards: torch.Tensor,
    advantages: torch.Tensor,
    selected_mask: torch.Tensor,
    top_fraction_mask: torch.Tensor,
    forced_final_mask: torch.Tensor,
    eligible_mask: torch.Tensor,
    group_stats: Sequence[DiffusionTPFGroupStats],
    prefix: str,
) -> dict[str, float]:
    total_rows = int(rewards.numel())
    selected_rows = int(selected_mask.sum().item())
    eligible_rows = int(eligible_mask.sum().item())
    selected_advantages = advantages[selected_mask]

    metrics = {
        f"{prefix}/total_rows": float(total_rows),
        f"{prefix}/groups": float(len(group_stats)),
        f"{prefix}/eligible_rows": float(eligible_rows),
        f"{prefix}/quality_rejected_rows": float(total_rows - eligible_rows),
        f"{prefix}/selected_rows": float(selected_rows),
        f"{prefix}/selected_fraction": float(selected_rows / total_rows) if total_rows else 0.0,
        f"{prefix}/selected_fraction_of_eligible": (
            float(selected_rows / eligible_rows) if eligible_rows else 0.0
        ),
        f"{prefix}/top_fraction_selected_rows": float(int(top_fraction_mask.sum().item())),
        f"{prefix}/forced_final_selected_rows": float(int(forced_final_mask.sum().item())),
        f"{prefix}/zero_variance_groups": float(sum(1 for stats in group_stats if stats.zero_variance)),
        f"{prefix}/short_groups": float(sum(1 for stats in group_stats if stats.short_group)),
        f"{prefix}/selected_advantages_positive": float(int((selected_advantages > 0).sum().item())),
        f"{prefix}/selected_advantages_zero": float(int((selected_advantages == 0).sum().item())),
        f"{prefix}/selected_advantages_negative": float(int((selected_advantages < 0).sum().item())),
    }
    metrics.update(_tensor_summary_metrics(rewards, prefix=f"{prefix}/reward"))
    metrics.update(_tensor_summary_metrics(advantages, prefix=f"{prefix}/advantage"))
    return metrics


def _tensor_summary_metrics(values: torch.Tensor, *, prefix: str) -> dict[str, float]:
    if int(values.numel()) == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
        }
    return {
        f"{prefix}_mean": float(values.mean().item()),
        f"{prefix}_min": float(values.min().item()),
        f"{prefix}_max": float(values.max().item()),
    }


def _handle_meta(handle: Mapping[str, Any]) -> Mapping[str, Any]:
    meta = handle.get("metadata") if isinstance(handle.get("metadata"), Mapping) else handle
    if not isinstance(meta, Mapping):
        raise ValueError(f"TPF replay handle metadata must be a mapping, got {type(meta)!r}")
    return meta


def _resolve_trace_root(trace_root: str | None, handles: Sequence[Mapping[str, Any]]) -> str:
    if trace_root:
        return os.path.realpath(os.path.abspath(trace_root))
    for handle in handles:
        value = _handle_meta(handle).get("trace_root") or handle.get("trace_root")
        if value:
            return os.path.realpath(os.path.abspath(str(value)))
        tensor_path = _handle_meta(handle).get("tensor_path") or handle.get("tensor_path")
        if tensor_path:
            path = os.path.realpath(os.path.abspath(str(tensor_path)))
            parent = os.path.dirname(path)
            if os.path.basename(parent) == "tensors":
                return os.path.dirname(parent)
            return parent
    raise ValueError("TPF metadata replay requires trace_root or tensor_path handles")


def _resolve_handle_tensor_path(handle: Mapping[str, Any], root: str, *, row_index: int) -> str:
    meta = _handle_meta(handle)
    raw_path = meta.get("tensor_path") or handle.get("tensor_path")
    if not raw_path:
        raise ValueError(f"row {row_index} is missing TPF replay tensor_path")
    raw_path = str(raw_path)
    if os.path.isabs(raw_path):
        path = raw_path
    else:
        path = os.path.join(root, raw_path)
    real_root = os.path.realpath(os.path.abspath(root))
    real_path = os.path.realpath(os.path.abspath(path))
    try:
        common = os.path.commonpath([real_root, real_path])
    except ValueError as exc:
        raise ValueError(f"row {row_index} TPF replay tensor_path escapes trace root") from exc
    if common != real_root:
        raise ValueError(f"row {row_index} TPF replay tensor_path escapes trace root: {raw_path}")
    return real_path


def _handle_identity(handle: Mapping[str, Any], *, row_index: int) -> tuple[Any, ...]:
    meta = _handle_meta(handle)
    try:
        return (
            str(handle.get("_tpf_response_uid", meta.get("_tpf_response_uid", ""))),
            int(meta.get("step_key")),
            int(meta.get("local_step")),
            int(meta.get("block_idx", meta.get("block_index", 0))),
            int(meta.get("req_pool_idx", -1)),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"row {row_index} has malformed TPF replay handle identity") from exc


def _handle_declared_bytes(handle: Mapping[str, Any], path: str) -> int:
    meta = _handle_meta(handle)
    for key in ("file_bytes", "tensor_file_bytes", "byte_size", "nbytes", "tensor_bytes"):
        value = meta.get(key, handle.get(key))
        if value is None:
            continue
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            continue
        return max(0, numeric)
    try:
        return max(0, int(os.path.getsize(path)))
    except OSError:
        return 0


def _validate_materialized_tpf_payload(
    payload: Mapping[str, Any],
    handle: Mapping[str, Any],
    *,
    row_index: int,
) -> None:
    meta = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else payload
    if not isinstance(meta, Mapping):
        raise ValueError(f"selected TPF replay row {row_index} payload metadata is malformed")
    handle_meta = _handle_meta(handle)
    for field_name in ("step_key", "local_step"):
        if int(meta.get(field_name, -1)) != int(handle_meta.get(field_name, -2)):
            raise ValueError(
                f"selected TPF replay row {row_index} {field_name} mismatch: "
                f"payload={meta.get(field_name)!r} handle={handle_meta.get(field_name)!r}"
            )
    payload_block = int(meta.get("block_idx", meta.get("block_index", -1)))
    handle_block = int(handle_meta.get("block_idx", handle_meta.get("block_index", -2)))
    if payload_block != handle_block:
        raise ValueError(
            f"selected TPF replay row {row_index} block mismatch: "
            f"payload={payload_block} handle={handle_block}"
        )
    action_ids = payload.get("action_ids")
    action_mask = payload.get("action_mask")
    if not isinstance(action_ids, torch.Tensor) or not isinstance(action_mask, torch.Tensor):
        raise ValueError(f"selected TPF replay row {row_index} is missing action_ids/action_mask tensors")
    if action_ids.view(-1).numel() != action_mask.view(-1).numel():
        raise ValueError(f"selected TPF replay row {row_index} action_ids/action_mask shape mismatch")

    handle_action_ids = handle_meta.get("action_ids", handle.get("action_ids"))
    handle_action_mask = handle_meta.get("action_mask", handle.get("action_mask"))
    if handle_action_ids is not None:
        expected = torch.as_tensor(handle_action_ids, dtype=torch.long).view(-1)
        actual = action_ids.detach().cpu().long().view(-1)[: expected.numel()]
        if int(actual.numel()) != int(expected.numel()) or not torch.equal(actual, expected):
            raise ValueError(f"selected TPF replay row {row_index} action_ids mismatch")
    if handle_action_mask is not None:
        expected_mask = torch.as_tensor(handle_action_mask, dtype=torch.bool).view(-1)
        actual_mask = action_mask.detach().cpu().bool().view(-1)[: expected_mask.numel()]
        if int(actual_mask.numel()) != int(expected_mask.numel()) or not torch.equal(actual_mask, expected_mask):
            raise ValueError(f"selected TPF replay row {row_index} action_mask mismatch")


def _payload_tensor_bytes(payload: Mapping[str, Any]) -> int:
    total = 0
    for value in payload.values():
        if isinstance(value, torch.Tensor):
            total += int(value.numel()) * int(value.element_size())
    return total


def _cleanup_tpf_replay_files(paths: Sequence[str], *, cleanup: bool) -> tuple[int, int]:
    if not cleanup:
        return 0, 0
    deleted = 0
    errors = 0
    seen: set[str] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        try:
            os.remove(path)
            deleted += 1
        except FileNotFoundError:
            continue
        except OSError:
            errors += 1
    return deleted, errors
