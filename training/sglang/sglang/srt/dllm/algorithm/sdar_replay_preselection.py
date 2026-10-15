"""Deterministic completed-block replay preselection and cleanup."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Mapping, MutableMapping, Sequence


LEGACY_PRESELECT_MODE = "earliest"
RANKED_PRESELECT_MODE = "ranked"
EDIT_RANKED_PRESELECT_MODE = "edit_then_new_accept"
DEFAULT_DIRECTION = "highest"
DEFAULT_SCORE = "new_accept_count"
EDIT_THEN_NEW_ACCEPT_SCORE = "edit_then_new_accept"
DEFAULT_GROUP_FIELDS = ("response_uid", "block_index")
DEFAULT_FINAL_POLICY = "always_include"
STATE_SELECTION_PROTOCOL_VERSION = 1
EDIT_RANKED_PROTOCOL_VERSION = 2
GPU_REPLAY_LEASE_FORMAT = "sglang_sdar_gpu_replay_lease_v1"


@dataclass(frozen=True)
class ReplayPreselectionPolicy:
    fraction: float
    mode: str = LEGACY_PRESELECT_MODE
    direction: str = DEFAULT_DIRECTION
    score: str = DEFAULT_SCORE
    group_fields: tuple[str, ...] = DEFAULT_GROUP_FIELDS
    final_policy: str = DEFAULT_FINAL_POLICY

    @classmethod
    def from_values(
        cls,
        *,
        fraction: Any,
        mode: Any = LEGACY_PRESELECT_MODE,
        direction: Any = DEFAULT_DIRECTION,
        score: Any = DEFAULT_SCORE,
        group_fields: Any = DEFAULT_GROUP_FIELDS,
        final_policy: Any = DEFAULT_FINAL_POLICY,
    ) -> "ReplayPreselectionPolicy":
        if isinstance(group_fields, str):
            group_fields = tuple(
                field.strip()
                for field in group_fields.split(",")
                if field.strip()
            )
        elif isinstance(group_fields, Sequence):
            group_fields = tuple(str(field).strip() for field in group_fields)
        else:
            raise ValueError(
                "SDAR replay preselection group fields must be a sequence"
            )
        policy = cls(
            fraction=fraction,
            mode=str(mode or LEGACY_PRESELECT_MODE).strip().lower(),
            direction=str(direction or DEFAULT_DIRECTION).strip().lower(),
            score=str(score or DEFAULT_SCORE).strip(),
            group_fields=tuple(group_fields),
            final_policy=str(
                final_policy or DEFAULT_FINAL_POLICY
            ).strip().lower(),
        )
        policy.validate()
        return policy

    def validate(self) -> None:
        if isinstance(self.fraction, bool):
            raise ValueError(
                "SDAR replay preselection fraction must be numeric"
            )
        try:
            fraction = float(self.fraction)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "SDAR replay preselection fraction must be numeric"
            ) from exc
        if (
            not math.isfinite(fraction)
            or fraction < 0.0
            or fraction > 1.0
        ):
            raise ValueError(
                "SDAR replay preselection fraction must be in [0, 1], "
                f"got {self.fraction!r}"
            )
        if self.mode not in {
            LEGACY_PRESELECT_MODE,
            RANKED_PRESELECT_MODE,
            EDIT_RANKED_PRESELECT_MODE,
        }:
            raise ValueError(
                "SDAR replay preselection mode must be earliest, ranked, or "
                f"edit_then_new_accept, got {self.mode!r}"
            )
        if self.mode == LEGACY_PRESELECT_MODE:
            return
        if fraction <= 0.0:
            raise ValueError(
                "ranked SDAR replay preselection fraction must be in (0, 1]"
            )
        if self.mode == RANKED_PRESELECT_MODE:
            if self.direction not in {"highest", "lowest"}:
                raise ValueError(
                    "ranked SDAR replay preselection direction must be highest "
                    f"or lowest, got {self.direction!r}"
                )
            if self.score != DEFAULT_SCORE:
                raise ValueError(
                    "ranked SDAR replay preselection score must be "
                    f"new_accept_count, got {self.score!r}"
                )
        elif self.mode == EDIT_RANKED_PRESELECT_MODE:
            if self.score != EDIT_THEN_NEW_ACCEPT_SCORE:
                raise ValueError(
                    "edit_then_new_accept SDAR replay preselection score must "
                    f"be edit_then_new_accept, got {self.score!r}"
                )
        if tuple(self.group_fields) != DEFAULT_GROUP_FIELDS:
            raise ValueError(
                "ranked SDAR replay preselection group fields must be "
                "[response_uid, block_index], "
                f"got {list(self.group_fields)!r}"
            )
        if self.final_policy != DEFAULT_FINAL_POLICY:
            raise ValueError(
                "ranked SDAR replay preselection final policy must be "
                f"always_include, got {self.final_policy!r}"
            )


@dataclass(frozen=True)
class ReplayPreselectionResult:
    source_rows: int
    rank_selected_indices: tuple[int, ...]
    selected_indices: tuple[int, ...]
    discarded_indices: tuple[int, ...]
    final_indices: tuple[int, ...]
    forced_final_indices: tuple[int, ...]
    source_score_min: int
    source_score_sum: int
    source_score_max: int
    zero_variance: bool
    # edit_then_new_accept extras (0 for other modes)
    source_edit_count_sum: int = 0
    positive_edit_rank_rows: int = 0
    zero_edit_backfill_rows: int = 0


@dataclass(frozen=True)
class ReplayCleanupResult:
    acknowledged_gpu_lease_pages: int
    deleted_tensor_files: int


def replay_row_metadata(
    row: Mapping[str, Any],
) -> MutableMapping[str, Any]:
    metadata = row.get("metadata")
    if isinstance(metadata, MutableMapping):
        return metadata
    if isinstance(row, MutableMapping):
        return row
    raise TypeError("SDAR replay rows must be mutable mappings")


def select_completed_block_replay_rows(
    rows: Sequence[MutableMapping[str, Any]],
    *,
    policy: ReplayPreselectionPolicy,
) -> ReplayPreselectionResult:
    """Select one completed block while preserving input row order."""

    policy.validate()
    indexed_rows = [
        (index, row, replay_row_metadata(row))
        for index, row in enumerate(rows)
    ]
    ordered = sorted(
        indexed_rows,
        key=lambda entry: (
            _integer_metadata(
                entry[2],
                "local_step",
                fallback_key="step_index",
                fallback=entry[0],
            ),
            _integer_metadata(
                entry[2],
                "step_key",
                fallback_key="trace_step_key",
                fallback=entry[0],
            ),
            entry[0],
        ),
    )
    if not ordered:
        return ReplayPreselectionResult(
            source_rows=0,
            rank_selected_indices=(),
            selected_indices=(),
            discarded_indices=(),
            final_indices=(),
            forced_final_indices=(),
            source_score_min=0,
            source_score_sum=0,
            source_score_max=0,
            zero_variance=False,
        )

    if policy.mode == LEGACY_PRESELECT_MODE:
        return _select_legacy_earliest(ordered, policy=policy)
    if policy.mode == EDIT_RANKED_PRESELECT_MODE:
        return _select_edit_ranked(ordered, policy=policy)
    return _select_ranked(ordered, policy=policy)


def cleanup_discarded_replay_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    acknowledge_gpu_replay_lease: Callable[[dict[str, Any]], None]
    | None = None,
    recycle_acknowledged: Callable[[], Any] | None = None,
) -> ReplayCleanupResult:
    """Acknowledge discarded leases and unlink discarded file payloads."""

    descriptors: dict[str, dict[str, Any]] = {}
    tensor_paths: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        for descriptor in _iter_gpu_replay_leases(row):
            lease_id = descriptor.get("lease_id")
            if lease_id is None:
                raise ValueError(
                    "discarded SDAR GPU replay lease is missing lease_id"
                )
            descriptors.setdefault(str(lease_id), descriptor)
        metadata = (
            row.get("metadata")
            if isinstance(row.get("metadata"), Mapping)
            else row
        )
        tensor_path = row.get("tensor_path") or metadata.get("tensor_path")
        if tensor_path:
            path = str(tensor_path)
            if not os.path.isabs(path):
                trace_root = row.get("trace_root") or metadata.get(
                    "trace_root"
                )
                if trace_root:
                    path = os.path.join(str(trace_root), path)
            tensor_paths.add(os.path.abspath(path))

    if descriptors and acknowledge_gpu_replay_lease is None:
        from sglang.srt.dllm.algorithm.sdar_gpu_replay_lease import (
            acknowledge_gpu_replay_lease as acknowledge,
        )

        acknowledge_gpu_replay_lease = acknowledge
    if acknowledge_gpu_replay_lease is not None:
        for descriptor in descriptors.values():
            acknowledge_gpu_replay_lease(descriptor)
    if descriptors and recycle_acknowledged is not None:
        recycle_acknowledged()

    deleted_files = 0
    for tensor_path in sorted(tensor_paths):
        try:
            os.unlink(tensor_path)
            deleted_files += 1
        except FileNotFoundError:
            pass
    return ReplayCleanupResult(
        acknowledged_gpu_lease_pages=len(descriptors),
        deleted_tensor_files=deleted_files,
    )


def _select_legacy_earliest(
    ordered: Sequence[
        tuple[int, MutableMapping[str, Any], MutableMapping[str, Any]]
    ],
    *,
    policy: ReplayPreselectionPolicy,
) -> ReplayPreselectionResult:
    final_rows = [
        entry
        for entry in ordered
        if bool(entry[2].get("is_final_denoise_row", False))
    ]
    if not final_rows:
        final_rows = [ordered[-1]]
        final_rows[0][2]["is_final_denoise_row"] = True
        final_rows[0][1]["is_final_denoise_row"] = True
    final_indices = {entry[0] for entry in final_rows}
    non_final = [
        entry
        for entry in ordered
        if entry[0] not in final_indices
        and not bool(entry[2].get("is_terminal_commit", False))
    ]
    rank_selected = non_final[
        : math.ceil(float(policy.fraction) * len(non_final))
    ]
    selected_indices = final_indices | {
        entry[0] for entry in rank_selected
    }
    for index, row, metadata in ordered:
        if index not in selected_indices:
            continue
        block_index = _integer_metadata(
            metadata,
            "block_idx",
            fallback_key="block_index",
        )
        metadata["producer_preselected"] = True
        metadata["producer_preselection_mode"] = LEGACY_PRESELECT_MODE
        metadata["producer_source_rows"] = len(ordered)
        metadata["producer_selected_rows"] = len(selected_indices)
        metadata["producer_block_index"] = block_index
        row["producer_preselected"] = True
    return ReplayPreselectionResult(
        source_rows=len(ordered),
        rank_selected_indices=tuple(entry[0] for entry in rank_selected),
        selected_indices=tuple(
            index for index, _, _ in ordered if index in selected_indices
        ),
        discarded_indices=tuple(
            index for index, _, _ in ordered if index not in selected_indices
        ),
        final_indices=tuple(
            index for index, _, _ in ordered if index in final_indices
        ),
        forced_final_indices=tuple(final_indices),
        source_score_min=0,
        source_score_sum=0,
        source_score_max=0,
        zero_variance=False,
    )


def _select_edit_ranked(
    ordered: Sequence[
        tuple[int, MutableMapping[str, Any], MutableMapping[str, Any]]
    ],
    *,
    policy: ReplayPreselectionPolicy,
) -> ReplayPreselectionResult:
    """Select edit-ranked states: positive-edit first, zero-edit backfill.

    Lexicographic key: (-edit_change_count, -new_accept_count, local_step, step_key).
    k = ceil(fraction * N) across all denoise rows.  The final row is always
    unioned under the always_include policy so realized selection may be k+1.
    """
    denoise_rows = [
        entry
        for entry in ordered
        if not bool(entry[2].get("is_terminal_commit", False))
    ]
    if not denoise_rows:
        raise ValueError(
            "edit_then_new_accept SDAR replay preselection requires denoise rows"
        )
    final_rows = [
        entry
        for entry in denoise_rows
        if bool(entry[2].get("is_final_denoise_row", False))
    ]
    if len(final_rows) != 1:
        raise ValueError(
            "edit_then_new_accept SDAR replay preselection requires exactly "
            f"one final denoise row, got {len(final_rows)}"
        )

    identities: set[tuple[int, int]] = set()
    edit_counts: dict[int, int] = {}
    accept_counts: dict[int, int] = {}
    for index, _, metadata in denoise_rows:
        local_step = _integer_metadata(
            metadata,
            "local_step",
            fallback_key="step_index",
            required=True,
        )
        step_key = _integer_metadata(
            metadata,
            "step_key",
            fallback_key="trace_step_key",
            required=True,
        )
        identity = (local_step, step_key)
        if identity in identities:
            raise ValueError(
                "edit_then_new_accept SDAR replay preselection received "
                f"duplicate (local_step, step_key) identity {identity!r}"
            )
        identities.add(identity)
        edit_counts[index] = _integer_metadata(
            metadata,
            "edit_change_count",
            required=True,
            non_negative=True,
        )
        accept_counts[index] = _integer_metadata(
            metadata,
            "new_accept_count",
            required=True,
            non_negative=True,
        )

    k = math.ceil(float(policy.fraction) * len(denoise_rows))
    final_index = final_rows[0][0]

    def _sort_key(entry: tuple[int, Any, Any]) -> tuple[int, int, int, int]:
        idx = entry[0]
        meta = entry[2]
        return (
            -edit_counts[idx],
            -accept_counts[idx],
            _integer_metadata(
                meta,
                "local_step",
                fallback_key="step_index",
                required=True,
            ),
            _integer_metadata(
                meta,
                "step_key",
                fallback_key="trace_step_key",
                required=True,
            ),
        )

    ranked = sorted(denoise_rows, key=_sort_key)

    positive_edit_entries = [e for e in ranked if edit_counts[e[0]] > 0]
    zero_edit_entries = [e for e in ranked if edit_counts[e[0]] == 0]

    rank_selected: list[tuple[int, Any, Any]] = []
    # Fill from positive-edit first
    rank_selected.extend(positive_edit_entries[:k])
    remaining = k - len(rank_selected)
    positive_edit_rank_count = len(rank_selected)
    # Backfill with zero-edit by (-new_accept_count, local_step, step_key)
    zero_edit_backfill_count = 0
    if remaining > 0:
        zero_edit_sorted = sorted(
            zero_edit_entries,
            key=_sort_key,
        )
        backfill = zero_edit_sorted[:remaining]
        rank_selected.extend(backfill)
        zero_edit_backfill_count = len(backfill)

    rank_selected_indices = {entry[0] for entry in rank_selected}
    selected_indices = rank_selected_indices | {final_index}
    forced_final_indices = (
        {final_index} if final_index not in rank_selected_indices else set()
    )

    source_edit_sum = sum(edit_counts.values())
    source_accept_scores = list(accept_counts.values())
    source_score_min = min(source_accept_scores)
    source_score_sum = sum(source_accept_scores)
    source_score_max = max(source_accept_scores)
    zero_variance = source_score_min == source_score_max
    group_fields = ",".join(policy.group_fields)

    for index, row, metadata in denoise_rows:
        if index not in selected_indices:
            continue
        block_index = _integer_metadata(
            metadata,
            "block_idx",
            fallback_key="block_index",
        )
        rank_selected_row = index in rank_selected_indices
        forced_final_row = index in forced_final_indices
        metadata.update(
            {
                "producer_preselected": True,
                "producer_preselection_mode": EDIT_RANKED_PRESELECT_MODE,
                "producer_source_rows": len(denoise_rows),
                "producer_rank_selected_rows": k,
                "producer_selected_rows": len(selected_indices),
                "producer_final_rows": 1,
                "producer_forced_final_rows": len(forced_final_indices),
                "producer_final_already_selected_rows": int(
                    not forced_final_indices
                ),
                "producer_source_score_min": source_score_min,
                "producer_source_score_sum": source_score_sum,
                "producer_source_score_max": source_score_max,
                "producer_zero_variance": zero_variance,
                "producer_block_index": block_index,
                "producer_state_selection": True,
                "producer_state_selection_version": EDIT_RANKED_PROTOCOL_VERSION,
                "producer_state_selection_complete": True,
                "producer_state_selection_mode": EDIT_RANKED_PRESELECT_MODE,
                "producer_state_selection_fraction": float(policy.fraction),
                "producer_state_selection_direction": policy.direction,
                "producer_state_selection_score": EDIT_THEN_NEW_ACCEPT_SCORE,
                "producer_state_selection_group_fields": group_fields,
                "producer_state_selection_final_policy": policy.final_policy,
                "producer_state_selection_rank_selected": rank_selected_row,
                "producer_state_selection_forced_final": forced_final_row,
                # edit-ranked extras
                "producer_source_edit_count_sum": source_edit_sum,
                "producer_positive_edit_rank_rows": positive_edit_rank_count,
                "producer_zero_edit_backfill_rows": zero_edit_backfill_count,
                "producer_row_edit_count": edit_counts[index],
                "producer_row_new_accept_count": accept_counts[index],
            }
        )
        row["producer_preselected"] = True
        row["producer_state_selection"] = True

    return ReplayPreselectionResult(
        source_rows=len(denoise_rows),
        rank_selected_indices=tuple(entry[0] for entry in rank_selected),
        selected_indices=tuple(
            index for index, _, _ in ordered if index in selected_indices
        ),
        discarded_indices=tuple(
            index for index, _, _ in ordered if index not in selected_indices
        ),
        final_indices=(final_index,),
        forced_final_indices=tuple(forced_final_indices),
        source_score_min=source_score_min,
        source_score_sum=source_score_sum,
        source_score_max=source_score_max,
        zero_variance=zero_variance,
        source_edit_count_sum=source_edit_sum,
        positive_edit_rank_rows=positive_edit_rank_count,
        zero_edit_backfill_rows=zero_edit_backfill_count,
    )


def _select_ranked(
    ordered: Sequence[
        tuple[int, MutableMapping[str, Any], MutableMapping[str, Any]]
    ],
    *,
    policy: ReplayPreselectionPolicy,
) -> ReplayPreselectionResult:
    denoise_rows = [
        entry
        for entry in ordered
        if not bool(entry[2].get("is_terminal_commit", False))
    ]
    if not denoise_rows:
        raise ValueError(
            "ranked SDAR replay preselection requires denoise rows"
        )
    final_rows = [
        entry
        for entry in denoise_rows
        if bool(entry[2].get("is_final_denoise_row", False))
    ]
    if len(final_rows) != 1:
        raise ValueError(
            "ranked SDAR replay preselection requires exactly one final "
            f"denoise row, got {len(final_rows)}"
        )

    identities: set[tuple[int, int]] = set()
    scores: dict[int, int] = {}
    for index, _, metadata in denoise_rows:
        local_step = _integer_metadata(
            metadata,
            "local_step",
            fallback_key="step_index",
            required=True,
        )
        step_key = _integer_metadata(
            metadata,
            "step_key",
            fallback_key="trace_step_key",
            required=True,
        )
        identity = (local_step, step_key)
        if identity in identities:
            raise ValueError(
                "ranked SDAR replay preselection received duplicate "
                f"(local_step, step_key) identity {identity!r}"
            )
        identities.add(identity)
        scores[index] = _integer_metadata(
            metadata,
            policy.score,
            required=True,
            non_negative=True,
        )

    ranked = sorted(
        denoise_rows,
        key=lambda entry: (
            -scores[entry[0]]
            if policy.direction == "highest"
            else scores[entry[0]],
            _integer_metadata(
                entry[2],
                "local_step",
                fallback_key="step_index",
                required=True,
            ),
            _integer_metadata(
                entry[2],
                "step_key",
                fallback_key="trace_step_key",
                required=True,
            ),
        ),
    )
    rank_count = math.ceil(float(policy.fraction) * len(denoise_rows))
    rank_selected = ranked[:rank_count]
    rank_selected_indices = {entry[0] for entry in rank_selected}
    final_index = final_rows[0][0]
    selected_indices = rank_selected_indices | {final_index}
    forced_final_indices = (
        {final_index}
        if final_index not in rank_selected_indices
        else set()
    )
    source_scores = list(scores.values())
    group_fields = ",".join(policy.group_fields)
    source_score_min = min(source_scores)
    source_score_sum = sum(source_scores)
    source_score_max = max(source_scores)
    zero_variance = source_score_min == source_score_max

    for index, row, metadata in denoise_rows:
        if index not in selected_indices:
            continue
        block_index = _integer_metadata(
            metadata,
            "block_idx",
            fallback_key="block_index",
        )
        rank_selected_row = index in rank_selected_indices
        forced_final_row = index in forced_final_indices
        metadata.update(
            {
                "producer_preselected": True,
                "producer_preselection_mode": RANKED_PRESELECT_MODE,
                "producer_source_rows": len(denoise_rows),
                "producer_rank_selected_rows": rank_count,
                "producer_selected_rows": len(selected_indices),
                "producer_final_rows": 1,
                "producer_forced_final_rows": len(
                    forced_final_indices
                ),
                "producer_final_already_selected_rows": int(
                    not forced_final_indices
                ),
                "producer_source_score_min": source_score_min,
                "producer_source_score_sum": source_score_sum,
                "producer_source_score_max": source_score_max,
                "producer_zero_variance": zero_variance,
                "producer_block_index": block_index,
                "producer_state_selection": True,
                "producer_state_selection_version": (
                    STATE_SELECTION_PROTOCOL_VERSION
                ),
                "producer_state_selection_complete": True,
                "producer_state_selection_mode": (
                    RANKED_PRESELECT_MODE
                ),
                "producer_state_selection_fraction": float(
                    policy.fraction
                ),
                "producer_state_selection_direction": policy.direction,
                "producer_state_selection_score": policy.score,
                "producer_state_selection_group_fields": group_fields,
                "producer_state_selection_final_policy": (
                    policy.final_policy
                ),
                "producer_state_selection_rank_selected": (
                    rank_selected_row
                ),
                "producer_state_selection_forced_final": (
                    forced_final_row
                ),
            }
        )
        row["producer_preselected"] = True
        row["producer_state_selection"] = True

    return ReplayPreselectionResult(
        source_rows=len(denoise_rows),
        rank_selected_indices=tuple(
            entry[0] for entry in rank_selected
        ),
        selected_indices=tuple(
            index for index, _, _ in ordered if index in selected_indices
        ),
        discarded_indices=tuple(
            index for index, _, _ in ordered if index not in selected_indices
        ),
        final_indices=(final_index,),
        forced_final_indices=tuple(forced_final_indices),
        source_score_min=source_score_min,
        source_score_sum=source_score_sum,
        source_score_max=source_score_max,
        zero_variance=zero_variance,
    )


def _integer_metadata(
    metadata: Mapping[str, Any],
    key: str,
    *,
    fallback_key: str | None = None,
    fallback: int | None = None,
    required: bool = False,
    non_negative: bool = False,
) -> int:
    value = metadata.get(key)
    if value is None and fallback_key is not None:
        value = metadata.get(fallback_key)
    if value is None:
        if required:
            raise ValueError(
                f"SDAR replay preselection row is missing {key!r}"
            )
        if fallback is None:
            fallback = 0
        value = fallback
    if isinstance(value, bool):
        raise ValueError(
            f"SDAR replay preselection {key} must be an integer"
        )
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"SDAR replay preselection {key} must be an integer"
        ) from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(
            f"SDAR replay preselection {key} must be a finite integer, "
            f"got {value!r}"
        )
    integer = int(numeric)
    if non_negative and integer < 0:
        raise ValueError(
            f"SDAR replay preselection {key} must be non-negative, "
            f"got {value!r}"
        )
    return integer


def _iter_gpu_replay_leases(
    value: Any,
):
    if (
        isinstance(value, Mapping)
        and value.get("format") == GPU_REPLAY_LEASE_FORMAT
    ):
        yield dict(value)
        return
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _iter_gpu_replay_leases(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _iter_gpu_replay_leases(child)
