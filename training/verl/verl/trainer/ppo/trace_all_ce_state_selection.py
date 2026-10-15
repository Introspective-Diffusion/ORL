"""Deterministic state selection for correct-only trace-all-CE replay."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


DEFAULT_GROUP_FIELDS = ("response_uid", "block_index")
DEFAULT_SCORE = "new_accept_count"
EDIT_THEN_NEW_ACCEPT_SCORE = "edit_then_new_accept"
DEFAULT_FINAL_POLICY = "always_include"
DEFAULT_METRIC_PREFIX = "trace_all_ce_state_selection"
PRODUCER_STATE_SELECTION_VERSION = 1
EDIT_RANKED_PROTOCOL_VERSION = 2
PRODUCER_STATE_SELECTION_MODE = "ranked"
EDIT_RANKED_SELECTION_MODE = "edit_then_new_accept"
VALID_SCORES = frozenset({DEFAULT_SCORE, EDIT_THEN_NEW_ACCEPT_SCORE})
VALID_MODES = frozenset({PRODUCER_STATE_SELECTION_MODE, EDIT_RANKED_SELECTION_MODE})


@dataclass(frozen=True)
class TraceAllCEStateSelectionConfig:
    """Accepted selective-state policy for trace-all-CE."""

    fraction: float = 0.25
    direction: str = "highest"
    score: str = DEFAULT_SCORE
    group_fields: tuple[str, ...] = DEFAULT_GROUP_FIELDS
    final_policy: str = DEFAULT_FINAL_POLICY
    metric_prefix: str = DEFAULT_METRIC_PREFIX

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any] | None,
    ) -> "TraceAllCEStateSelectionConfig":
        values = values or {}
        if not isinstance(values, Mapping):
            raise ValueError("trace_all_ce_state_selection must be a mapping")
        raw_group_fields = values.get("group_fields", DEFAULT_GROUP_FIELDS)
        if isinstance(raw_group_fields, str) or not isinstance(
            raw_group_fields,
            Sequence,
        ):
            raise ValueError(
                "trace_all_ce_state_selection.group_fields must be a sequence"
            )
        score_raw = str(values.get("score", DEFAULT_SCORE) or DEFAULT_SCORE).strip()
        if score_raw not in VALID_SCORES:
            raise ValueError(
                f"trace_all_ce_state_selection.score must be one of "
                f"{sorted(VALID_SCORES)}, got {score_raw!r}"
            )
        return cls(
            fraction=values.get("fraction", 0.25),
            direction=str(values.get("direction", "highest") or "highest")
            .strip()
            .lower(),
            score=score_raw,
            group_fields=tuple(str(field) for field in raw_group_fields),
            final_policy=str(
                values.get("final_policy", DEFAULT_FINAL_POLICY)
                or DEFAULT_FINAL_POLICY
            )
            .strip()
            .lower(),
        )


@dataclass(frozen=True)
class TraceAllCEStateSelectionGroup:
    key: tuple[Any, ...]
    row_indices: tuple[int, ...]
    rank_selected_indices: tuple[int, ...]
    selected_indices: tuple[int, ...]
    final_index: int
    score_min: int
    score_max: int
    zero_variance: bool


@dataclass(frozen=True)
class TraceAllCEStateSelectionResult:
    selected_mask: tuple[bool, ...]
    rank_selected_mask: tuple[bool, ...]
    final_mask: tuple[bool, ...]
    forced_final_mask: tuple[bool, ...]
    groups: tuple[TraceAllCEStateSelectionGroup, ...]
    metrics: dict[str, float]


def select_trace_all_ce_states(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: TraceAllCEStateSelectionConfig | None = None,
) -> TraceAllCEStateSelectionResult:
    """Select ceil(fraction * rows) per block by score, plus its final row."""

    config = config or TraceAllCEStateSelectionConfig()
    _validate_config(config)
    normalized_rows = _validate_rows(rows, config)
    groups = _group_indices(normalized_rows, config.group_fields)

    selected_mask = [False] * len(normalized_rows)
    rank_selected_mask = [False] * len(normalized_rows)
    final_mask = [False] * len(normalized_rows)
    forced_final_mask = [False] * len(normalized_rows)
    group_results: list[TraceAllCEStateSelectionGroup] = []

    for key, indices in groups.items():
        final_indices = [
            index
            for index in indices
            if normalized_rows[index]["is_final_denoise_row"]
        ]
        if len(final_indices) != 1:
            raise ValueError(
                "trace_all_ce_state_selection requires exactly one final "
                f"denoise row per group; group={key!r} finals={len(final_indices)}"
            )
        final_index = final_indices[0]
        final_mask[final_index] = True

        ranked = sorted(
            indices,
            key=lambda index: _ranking_key(
                normalized_rows[index],
                direction=config.direction,
                score=config.score,
            ),
        )
        select_count = int(math.ceil(float(config.fraction) * len(indices)))
        rank_selected = ranked[:select_count]
        for index in rank_selected:
            rank_selected_mask[index] = True
            selected_mask[index] = True

        if not selected_mask[final_index]:
            selected_mask[final_index] = True
            forced_final_mask[final_index] = True

        selected_indices = tuple(
            index for index in indices if selected_mask[index]
        )
        _score_field = DEFAULT_SCORE if config.score == EDIT_THEN_NEW_ACCEPT_SCORE else config.score
        scores = [normalized_rows[index][_score_field] for index in indices]
        group_results.append(
            TraceAllCEStateSelectionGroup(
                key=key,
                row_indices=tuple(indices),
                rank_selected_indices=tuple(rank_selected),
                selected_indices=selected_indices,
                final_index=final_index,
                score_min=min(scores),
                score_max=max(scores),
                zero_variance=min(scores) == max(scores),
            )
        )

    metrics = _build_metrics(
        rows=normalized_rows,
        selected_mask=selected_mask,
        rank_selected_mask=rank_selected_mask,
        final_mask=final_mask,
        forced_final_mask=forced_final_mask,
        groups=group_results,
        config=config,
    )
    return TraceAllCEStateSelectionResult(
        selected_mask=tuple(selected_mask),
        rank_selected_mask=tuple(rank_selected_mask),
        final_mask=tuple(final_mask),
        forced_final_mask=tuple(forced_final_mask),
        groups=tuple(group_results),
        metrics=metrics,
    )


def validate_producer_preselected_trace_all_ce_states(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: TraceAllCEStateSelectionConfig | None = None,
) -> TraceAllCEStateSelectionResult:
    """Validate complete producer-ranked groups without selecting them again."""

    config = config or TraceAllCEStateSelectionConfig()
    _validate_config(config)
    normalized_rows = _validate_rows(rows, config)
    if not normalized_rows:
        result = select_trace_all_ce_states([], config=config)
        metrics = dict(result.metrics)
        metrics[f"{config.metric_prefix}/producer_preselected"] = 1.0
        metrics[
            f"{config.metric_prefix}/producer_preselected_groups"
        ] = 0.0
        return TraceAllCEStateSelectionResult(
            selected_mask=result.selected_mask,
            rank_selected_mask=result.rank_selected_mask,
            final_mask=result.final_mask,
            forced_final_mask=result.forced_final_mask,
            groups=result.groups,
            metrics=metrics,
        )

    producer_flags = [
        _bool_value(
            row.get("producer_state_selection", False),
            field="producer_state_selection",
            row_index=index,
        )
        for index, row in enumerate(normalized_rows)
    ]
    if any(producer_flags) and not all(producer_flags):
        raise ValueError(
            "trace_all_ce_state_selection received mixed/partial producer "
            "state-selection rows"
        )
    if not all(producer_flags):
        raise ValueError(
            "trace_all_ce_state_selection producer validation requires every "
            "row to be producer-preselected"
        )

    grouped_indices = _group_indices(
        normalized_rows,
        config.group_fields,
    )
    rank_selected_mask = [False] * len(normalized_rows)
    final_mask = [False] * len(normalized_rows)
    forced_final_mask = [False] * len(normalized_rows)
    group_results: list[TraceAllCEStateSelectionGroup] = []
    source_group_sizes: list[int] = []
    source_score_sums: list[int] = []

    for key, indices in grouped_indices.items():
        canonical_summary: tuple[Any, ...] | None = None
        for index in indices:
            row = normalized_rows[index]
            _validate_producer_policy_row(
                row,
                row_index=index,
                config=config,
            )
            summary = _producer_group_summary(row, row_index=index)
            if canonical_summary is None:
                canonical_summary = summary
            elif summary != canonical_summary:
                raise ValueError(
                    "trace_all_ce_state_selection received inconsistent "
                    f"producer group metadata for group={key!r}"
                )

        assert canonical_summary is not None
        (
            source_rows,
            rank_selected_rows,
            selected_rows,
            producer_final_rows,
            forced_final_rows,
            final_already_selected_rows,
            source_score_min,
            source_score_sum,
            source_score_max,
            zero_variance,
            producer_block_index,
        ) = canonical_summary
        expected_rank_rows = math.ceil(
            float(config.fraction) * source_rows
        )
        if source_rows <= 0 or source_rows < selected_rows:
            raise ValueError(
                "trace_all_ce_state_selection producer source row count is "
                f"invalid for group={key!r}: source={source_rows} "
                f"selected={selected_rows}"
            )
        if rank_selected_rows != expected_rank_rows:
            raise ValueError(
                "trace_all_ce_state_selection producer rank count mismatch "
                f"for group={key!r}: expected={expected_rank_rows} "
                f"actual={rank_selected_rows}"
            )
        if selected_rows != len(indices):
            raise ValueError(
                "trace_all_ce_state_selection received a partial producer "
                f"group={key!r}: expected selected rows={selected_rows} "
                f"observed={len(indices)}"
            )
        if (
            producer_final_rows != 1
            or forced_final_rows not in {0, 1}
            or final_already_selected_rows not in {0, 1}
            or forced_final_rows + final_already_selected_rows != 1
            or selected_rows != rank_selected_rows + forced_final_rows
        ):
            raise ValueError(
                "trace_all_ce_state_selection producer final-union summary "
                f"is invalid for group={key!r}"
            )
        if producer_block_index != int(key[1]):
            raise ValueError(
                "trace_all_ce_state_selection producer block index mismatch "
                f"for group={key!r}: producer={producer_block_index}"
            )
        if (
            source_score_min < 0
            or source_score_min > source_score_max
            or source_score_sum < source_score_min * source_rows
            or source_score_sum > source_score_max * source_rows
            or zero_variance
            != (source_score_min == source_score_max)
        ):
            raise ValueError(
                "trace_all_ce_state_selection producer source score summary "
                f"is invalid for group={key!r}"
            )

        final_indices = [
            index
            for index in indices
            if normalized_rows[index]["is_final_denoise_row"]
        ]
        rank_indices = [
            index
            for index in indices
            if _bool_value(
                normalized_rows[index].get(
                    "producer_state_selection_rank_selected",
                    False,
                ),
                field="producer_state_selection_rank_selected",
                row_index=index,
            )
        ]
        forced_indices = [
            index
            for index in indices
            if _bool_value(
                normalized_rows[index].get(
                    "producer_state_selection_forced_final",
                    False,
                ),
                field="producer_state_selection_forced_final",
                row_index=index,
            )
        ]
        if len(final_indices) != 1:
            raise ValueError(
                "trace_all_ce_state_selection producer group requires exactly "
                f"one retained final row; group={key!r} "
                f"finals={len(final_indices)}"
            )
        if len(rank_indices) != rank_selected_rows:
            raise ValueError(
                "trace_all_ce_state_selection producer retained rank-row "
                f"count mismatch for group={key!r}"
            )
        if (
            len(forced_indices) != forced_final_rows
            or set(forced_indices)
            != (set(final_indices) - set(rank_indices))
        ):
            raise ValueError(
                "trace_all_ce_state_selection producer forced-final markers "
                f"are invalid for group={key!r}"
            )
        if (
            len(set(final_indices) & set(rank_indices))
            != final_already_selected_rows
        ):
            raise ValueError(
                "trace_all_ce_state_selection producer final/rank overlap "
                f"is invalid for group={key!r}"
            )
        if config.score != EDIT_THEN_NEW_ACCEPT_SCORE:
            selected_scores = [
                int(normalized_rows[index][config.score])
                for index in indices
            ]
            if any(
                sc < source_score_min or sc > source_score_max
                for sc in selected_scores
            ):
                raise ValueError(
                    "trace_all_ce_state_selection producer retained score "
                    f"lies outside the source range for group={key!r}"
                )

        for index in rank_indices:
            rank_selected_mask[index] = True
        for index in final_indices:
            final_mask[index] = True
        for index in forced_indices:
            forced_final_mask[index] = True
        group_results.append(
            TraceAllCEStateSelectionGroup(
                key=key,
                row_indices=tuple(indices),
                rank_selected_indices=tuple(rank_indices),
                selected_indices=tuple(indices),
                final_index=final_indices[0],
                score_min=source_score_min,
                score_max=source_score_max,
                zero_variance=zero_variance,
            )
        )
        source_group_sizes.append(source_rows)
        source_score_sums.append(source_score_sum)

    selected_mask = [True] * len(normalized_rows)
    metrics = _build_producer_metrics(
        rows=normalized_rows,
        rank_selected_mask=rank_selected_mask,
        final_mask=final_mask,
        forced_final_mask=forced_final_mask,
        groups=group_results,
        source_group_sizes=source_group_sizes,
        source_score_sums=source_score_sums,
        config=config,
    )
    return TraceAllCEStateSelectionResult(
        selected_mask=tuple(selected_mask),
        rank_selected_mask=tuple(rank_selected_mask),
        final_mask=tuple(final_mask),
        forced_final_mask=tuple(forced_final_mask),
        groups=tuple(group_results),
        metrics=metrics,
    )


def _validate_config(config: TraceAllCEStateSelectionConfig) -> None:
    if isinstance(config.fraction, bool):
        raise ValueError(
            "trace_all_ce_state_selection.fraction must be numeric"
        )
    try:
        fraction = float(config.fraction)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "trace_all_ce_state_selection.fraction must be numeric"
        ) from exc
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError(
            "trace_all_ce_state_selection.fraction must be in (0, 1], "
            f"got {config.fraction!r}"
        )
    if config.direction not in {"highest", "lowest"}:
        raise ValueError(
            "trace_all_ce_state_selection.direction must be highest or lowest, "
            f"got {config.direction!r}"
        )
    if config.score not in VALID_SCORES:
        raise ValueError(
            "trace_all_ce_state_selection.score must be new_accept_count or "
            f"edit_then_new_accept, got {config.score!r}"
        )
    if tuple(config.group_fields) != DEFAULT_GROUP_FIELDS:
        raise ValueError(
            "trace_all_ce_state_selection.group_fields must be "
            "[response_uid, block_index], "
            f"got {list(config.group_fields)!r}"
        )
    if config.final_policy != DEFAULT_FINAL_POLICY:
        raise ValueError(
            "trace_all_ce_state_selection.final_policy must be always_include, "
            f"got {config.final_policy!r}"
        )
    if not config.metric_prefix:
        raise ValueError(
            "trace_all_ce_state_selection.metric_prefix must be non-empty"
        )


def _validate_rows(
    rows: Sequence[Mapping[str, Any]],
    config: TraceAllCEStateSelectionConfig,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    identities: set[tuple[Any, ...]] = set()
    for row_index, raw_row in enumerate(rows):
        if not isinstance(raw_row, Mapping):
            raise ValueError(
                "trace_all_ce_state_selection rows must be mappings, "
                f"got row {row_index}: {type(raw_row)!r}"
            )
        row = dict(raw_row)
        key = tuple(
            _required(row, field, row_index)
            for field in config.group_fields
        )
        try:
            hash(key)
        except TypeError as exc:
            raise ValueError(
                "trace_all_ce_state_selection group values must be hashable, "
                f"got row {row_index}: {key!r}"
            ) from exc
        step_index = _integer_value(
            _required(row, "step_index", row_index),
            field="step_index",
            row_index=row_index,
            non_negative=True,
        )
        trace_step_key = _integer_value(
            _required(row, "trace_step_key", row_index),
            field="trace_step_key",
            row_index=row_index,
            non_negative=False,
        )
        if config.score == EDIT_THEN_NEW_ACCEPT_SCORE:
            # For edit_then_new_accept, validate both component fields.
            edit_count = _integer_value(
                _required(row, "edit_change_count", row_index),
                field="edit_change_count",
                row_index=row_index,
                non_negative=True,
            )
            score = _integer_value(
                _required(row, DEFAULT_SCORE, row_index),
                field=DEFAULT_SCORE,
                row_index=row_index,
                non_negative=True,
            )
        else:
            edit_count = None
            score = _integer_value(
                _required(row, config.score, row_index),
                field=config.score,
                row_index=row_index,
                non_negative=True,
            )
        final = _bool_value(
            _required(row, "is_final_denoise_row", row_index),
            field="is_final_denoise_row",
            row_index=row_index,
        )
        identity = (*key, step_index, trace_step_key)
        if identity in identities:
            raise ValueError(
                "duplicate trace_all_ce_state_selection row identity "
                f"{identity!r}"
            )
        identities.add(identity)
        updates: dict[str, Any] = {
            "step_index": step_index,
            "trace_step_key": trace_step_key,
            "is_final_denoise_row": final,
        }
        if config.score == EDIT_THEN_NEW_ACCEPT_SCORE:
            updates["edit_change_count"] = edit_count
            updates[DEFAULT_SCORE] = score
        else:
            updates[config.score] = score
        row.update(updates)
        normalized.append(row)
    return normalized


def _group_indices(
    rows: Sequence[Mapping[str, Any]],
    group_fields: Sequence[str],
) -> dict[tuple[Any, ...], list[int]]:
    groups: dict[tuple[Any, ...], list[int]] = {}
    for index, row in enumerate(rows):
        key = tuple(row[field] for field in group_fields)
        groups.setdefault(key, []).append(index)
    return groups


def _ranking_key(
    row: Mapping[str, Any],
    *,
    direction: str,
    score: str = DEFAULT_SCORE,
) -> tuple[int, int, int, int]:
    if score == EDIT_THEN_NEW_ACCEPT_SCORE:
        # (-edit_change_count, -new_accept_count, local_step, step_key)
        edit_count = int(row["edit_change_count"])
        accept_count = int(row[DEFAULT_SCORE])
        return (
            -edit_count,
            -accept_count,
            int(row["step_index"]),
            int(row["trace_step_key"]),
        )
    val = int(row[score])
    score_key = -val if direction == "highest" else val
    return (
        score_key,
        int(row["step_index"]),
        int(row["trace_step_key"]),
        0,
    )


def _required(
    row: Mapping[str, Any],
    field: str,
    row_index: int,
) -> Any:
    if field not in row or row[field] is None:
        raise ValueError(
            f"trace_all_ce_state_selection row {row_index} is missing {field!r}"
        )
    return row[field]


def _integer_value(
    value: Any,
    *,
    field: str,
    row_index: int,
    non_negative: bool,
) -> int:
    if isinstance(value, bool):
        raise ValueError(
            f"trace_all_ce_state_selection row {row_index} {field} "
            "must be an integer"
        )
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"trace_all_ce_state_selection row {row_index} {field} "
            "must be an integer"
        ) from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(
            f"trace_all_ce_state_selection row {row_index} {field} "
            f"must be a finite integer, got {value!r}"
        )
    integer = int(numeric)
    if non_negative and integer < 0:
        raise ValueError(
            f"trace_all_ce_state_selection row {row_index} {field} "
            f"must be non-negative, got {value!r}"
        )
    return integer


def _bool_value(
    value: Any,
    *,
    field: str,
    row_index: int,
) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    raise ValueError(
        f"trace_all_ce_state_selection row {row_index} {field} "
        f"must be boolean-like, got {value!r}"
    )


def _validate_producer_policy_row(
    row: Mapping[str, Any],
    *,
    row_index: int,
    config: TraceAllCEStateSelectionConfig,
) -> None:
    version = _integer_value(
        _required(
            row,
            "producer_state_selection_version",
            row_index,
        ),
        field="producer_state_selection_version",
        row_index=row_index,
        non_negative=True,
    )
    complete = _bool_value(
        _required(
            row,
            "producer_state_selection_complete",
            row_index,
        ),
        field="producer_state_selection_complete",
        row_index=row_index,
    )
    mode = str(
        _required(
            row,
            "producer_state_selection_mode",
            row_index,
        )
    ).strip().lower()
    direction = str(
        _required(
            row,
            "producer_state_selection_direction",
            row_index,
        )
    ).strip().lower()
    score = str(
        _required(
            row,
            "producer_state_selection_score",
            row_index,
        )
    ).strip()
    final_policy = str(
        _required(
            row,
            "producer_state_selection_final_policy",
            row_index,
        )
    ).strip().lower()
    raw_group_fields = _required(
        row,
        "producer_state_selection_group_fields",
        row_index,
    )
    if isinstance(raw_group_fields, str):
        group_fields = tuple(
            field.strip()
            for field in raw_group_fields.split(",")
            if field.strip()
        )
    elif isinstance(raw_group_fields, Sequence):
        group_fields = tuple(str(field) for field in raw_group_fields)
    else:
        raise ValueError(
            "trace_all_ce_state_selection producer group fields must be "
            f"a sequence, got row {row_index}"
        )
    raw_fraction = _required(
        row,
        "producer_state_selection_fraction",
        row_index,
    )
    if isinstance(raw_fraction, bool):
        raise ValueError(
            "trace_all_ce_state_selection producer fraction must be numeric"
        )
    try:
        fraction = float(raw_fraction)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "trace_all_ce_state_selection producer fraction must be numeric"
        ) from exc
    # Accept version 1 (new_accept_count ranked) or version 2 (edit_then_new_accept)
    expected_version = (
        EDIT_RANKED_PROTOCOL_VERSION
        if config.score == EDIT_THEN_NEW_ACCEPT_SCORE
        else PRODUCER_STATE_SELECTION_VERSION
    )
    expected_mode = (
        EDIT_RANKED_SELECTION_MODE
        if config.score == EDIT_THEN_NEW_ACCEPT_SCORE
        else PRODUCER_STATE_SELECTION_MODE
    )
    if (
        version != expected_version
        or not complete
        or mode != expected_mode
        or not math.isfinite(fraction)
        or not math.isclose(
            fraction,
            float(config.fraction),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or direction != config.direction
        or score != config.score
        or group_fields != tuple(config.group_fields)
        or final_policy != config.final_policy
        or not _bool_value(
            row.get("producer_preselected", False),
            field="producer_preselected",
            row_index=row_index,
        )
    ):
        raise ValueError(
            "trace_all_ce_state_selection producer policy mismatch at row "
            f"{row_index}: version={version!r} complete={complete!r} "
            f"mode={mode!r} fraction={fraction!r} "
            f"direction={direction!r} score={score!r} "
            f"group_fields={list(group_fields)!r} "
            f"final_policy={final_policy!r}"
        )


def _producer_group_summary(
    row: Mapping[str, Any],
    *,
    row_index: int,
) -> tuple[Any, ...]:
    integer_fields = (
        "producer_source_rows",
        "producer_rank_selected_rows",
        "producer_selected_rows",
        "producer_final_rows",
        "producer_forced_final_rows",
        "producer_final_already_selected_rows",
        "producer_source_score_min",
        "producer_source_score_sum",
        "producer_source_score_max",
        "producer_block_index",
    )
    values = [
        _integer_value(
            _required(row, field, row_index),
            field=field,
            row_index=row_index,
            non_negative=True,
        )
        for field in integer_fields
    ]
    zero_variance = _bool_value(
        _required(
            row,
            "producer_zero_variance",
            row_index,
        ),
        field="producer_zero_variance",
        row_index=row_index,
    )
    return (*values[:9], zero_variance, values[9])


def _build_producer_metrics(
    *,
    rows: Sequence[Mapping[str, Any]],
    rank_selected_mask: Sequence[bool],
    final_mask: Sequence[bool],
    forced_final_mask: Sequence[bool],
    groups: Sequence[TraceAllCEStateSelectionGroup],
    source_group_sizes: Sequence[int],
    source_score_sums: Sequence[int],
    config: TraceAllCEStateSelectionConfig,
) -> dict[str, float]:
    prefix = config.metric_prefix
    source_rows = sum(source_group_sizes)
    selected_rows = len(rows)
    rank_selected_rows = sum(rank_selected_mask)
    final_rows = sum(final_mask)
    forced_final_rows = sum(forced_final_mask)
    final_already_selected_rows = sum(
        final and rank_selected
        for final, rank_selected in zip(
            final_mask,
            rank_selected_mask,
            strict=True,
        )
    )
    score_field = DEFAULT_SCORE if config.score == EDIT_THEN_NEW_ACCEPT_SCORE else config.score
    selected_scores = [int(row[score_field]) for row in rows]

    def _mean(values: Sequence[int]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    return {
        f"{prefix}/enabled": 1.0,
        f"{prefix}/source_rows": float(source_rows),
        f"{prefix}/groups": float(len(groups)),
        f"{prefix}/rank_selected_rows": float(rank_selected_rows),
        f"{prefix}/final_rows": float(final_rows),
        f"{prefix}/final_already_selected_rows": float(
            final_already_selected_rows
        ),
        f"{prefix}/forced_final_rows": float(forced_final_rows),
        f"{prefix}/selected_rows": float(selected_rows),
        f"{prefix}/dropped_rows": float(source_rows - selected_rows),
        f"{prefix}/selected_fraction": (
            float(selected_rows / source_rows) if source_rows else 0.0
        ),
        f"{prefix}/zero_variance_groups": float(
            sum(group.zero_variance for group in groups)
        ),
        f"{prefix}/singleton_groups": float(
            sum(size == 1 for size in source_group_sizes)
        ),
        f"{prefix}/group_rows_min": float(
            min(source_group_sizes, default=0)
        ),
        f"{prefix}/group_rows_mean": _mean(source_group_sizes),
        f"{prefix}/group_rows_max": float(
            max(source_group_sizes, default=0)
        ),
        f"{prefix}/direction_highest": float(
            config.direction == "highest"
        ),
        f"{prefix}/direction_lowest": float(
            config.direction == "lowest"
        ),
        f"{prefix}/source_score_min": float(
            min(
                (group.score_min for group in groups),
                default=0,
            )
        ),
        f"{prefix}/source_score_mean": (
            float(sum(source_score_sums) / source_rows)
            if source_rows
            else 0.0
        ),
        f"{prefix}/source_score_max": float(
            max(
                (group.score_max for group in groups),
                default=0,
            )
        ),
        f"{prefix}/selected_score_min": float(
            min(selected_scores, default=0)
        ),
        f"{prefix}/selected_score_mean": _mean(selected_scores),
        f"{prefix}/selected_score_max": float(
            max(selected_scores, default=0)
        ),
        f"{prefix}/producer_preselected": 1.0,
        f"{prefix}/producer_preselected_groups": float(len(groups)),
        f"{prefix}/producer_preselected_rows": float(selected_rows),
    }


def _build_metrics(
    *,
    rows: Sequence[Mapping[str, Any]],
    selected_mask: Sequence[bool],
    rank_selected_mask: Sequence[bool],
    final_mask: Sequence[bool],
    forced_final_mask: Sequence[bool],
    groups: Sequence[TraceAllCEStateSelectionGroup],
    config: TraceAllCEStateSelectionConfig,
) -> dict[str, float]:
    prefix = config.metric_prefix
    source_rows = len(rows)
    selected_rows = sum(selected_mask)
    rank_selected_rows = sum(rank_selected_mask)
    final_rows = sum(final_mask)
    forced_final_rows = sum(forced_final_mask)
    final_already_selected_rows = sum(
        final and rank_selected
        for final, rank_selected in zip(
            final_mask,
            rank_selected_mask,
            strict=True,
        )
    )
    score_field = DEFAULT_SCORE if config.score == EDIT_THEN_NEW_ACCEPT_SCORE else config.score
    source_scores = [int(row[score_field]) for row in rows]
    selected_scores = [
        sc
        for sc, selected in zip(
            source_scores,
            selected_mask,
            strict=True,
        )
        if selected
    ]
    group_sizes = [len(group.row_indices) for group in groups]

    def _mean(values: Sequence[int]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    return {
        f"{prefix}/enabled": 1.0,
        f"{prefix}/source_rows": float(source_rows),
        f"{prefix}/groups": float(len(groups)),
        f"{prefix}/rank_selected_rows": float(rank_selected_rows),
        f"{prefix}/final_rows": float(final_rows),
        f"{prefix}/final_already_selected_rows": float(
            final_already_selected_rows
        ),
        f"{prefix}/forced_final_rows": float(forced_final_rows),
        f"{prefix}/selected_rows": float(selected_rows),
        f"{prefix}/dropped_rows": float(source_rows - selected_rows),
        f"{prefix}/selected_fraction": (
            float(selected_rows / source_rows) if source_rows else 0.0
        ),
        f"{prefix}/zero_variance_groups": float(
            sum(group.zero_variance for group in groups)
        ),
        f"{prefix}/singleton_groups": float(
            sum(len(group.row_indices) == 1 for group in groups)
        ),
        f"{prefix}/group_rows_min": float(min(group_sizes, default=0)),
        f"{prefix}/group_rows_mean": _mean(group_sizes),
        f"{prefix}/group_rows_max": float(max(group_sizes, default=0)),
        f"{prefix}/direction_highest": float(
            config.direction == "highest"
        ),
        f"{prefix}/direction_lowest": float(config.direction == "lowest"),
        f"{prefix}/source_score_min": float(min(source_scores, default=0)),
        f"{prefix}/source_score_mean": _mean(source_scores),
        f"{prefix}/source_score_max": float(max(source_scores, default=0)),
        f"{prefix}/selected_score_min": float(
            min(selected_scores, default=0)
        ),
        f"{prefix}/selected_score_mean": _mean(selected_scores),
        f"{prefix}/selected_score_max": float(
            max(selected_scores, default=0)
        ),
        f"{prefix}/producer_preselected": 0.0,
        f"{prefix}/producer_preselected_groups": 0.0,
        f"{prefix}/producer_preselected_rows": 0.0,
    }
