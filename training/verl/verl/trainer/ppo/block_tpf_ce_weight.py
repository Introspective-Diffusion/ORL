"""Block-level TPF weighting utilities for Qwen trace replay CE.

This module is intentionally separate from row-level ``tpf_reinforce``.
The canonical block TPF is:

    sum(new_accept_count over non-terminal denoise rows) / denoise_forwards

Producer block summaries (``new_accept_tokens`` and ``denoise_forwards``)
take precedence when present and are checked against the denoise rows.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch


BLOCK_TPF_SOURCE = "rollout_new_accept_per_denoise_forward"


class MissingBlockTPFMetadataError(ValueError):
    """Raised when an active replay token has no exact block TPF metadata."""


@dataclass(frozen=True)
class BlockTPFSummary:
    block_index: int
    new_accept_tokens: int
    denoise_forwards: int

    @property
    def raw_tpf(self) -> float:
        return float(self.new_accept_tokens) / float(self.denoise_forwards)


@dataclass(frozen=True)
class BlockTPFTokenTensors:
    raw_tpf: torch.Tensor
    response_hashes: torch.Tensor
    block_indices: torch.Tensor
    active_mask: torch.Tensor
    ghost_rows_excluded: int


@dataclass(frozen=True)
class BlockTPFRecords:
    keys: torch.Tensor
    values: torch.Tensor


@dataclass(frozen=True)
class BlockTPFNormalization:
    weights: torch.Tensor
    global_mean: float
    global_active_blocks: int
    global_active_tokens: int
    global_ghost_rows_excluded: int
    raw_min: float
    raw_max: float
    weight_min: float
    weight_mean: float
    weight_max: float


def stable_response_identity_hash(response_identity: Any) -> tuple[int, int]:
    """Return two stable 63-bit hash lanes for one response identity."""

    if response_identity is None or str(response_identity) == "":
        raise MissingBlockTPFMetadataError("block TPF weighting requires a non-empty response_uid")
    digest = hashlib.blake2b(
        str(response_identity).encode("utf-8"),
        digest_size=16,
        person=b"verl-block-tpf",
    ).digest()
    mask = (1 << 63) - 1
    return (
        int.from_bytes(digest[:8], "little", signed=False) & mask,
        int.from_bytes(digest[8:], "little", signed=False) & mask,
    )


def _mapping_value(state: Mapping[str, Any], key: str) -> Any:
    if key in state:
        return state[key]
    metadata = state.get("metadata")
    if isinstance(metadata, Mapping):
        return metadata.get(key)
    return None


def _int_scalar(value: Any, *, field: str) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{field} must be scalar, got shape={tuple(value.shape)}")
        value = value.detach().cpu().item()
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer, got {value!r}") from exc


def rollout_state_block_index(state: Mapping[str, Any]) -> int:
    """Read the producer block index without interpreting a denoise step key."""

    value = _mapping_value(state, "block_index")
    if value is None:
        value = _mapping_value(state, "block_idx")
    if value is None:
        raise MissingBlockTPFMetadataError(
            "rollout self-conditioning row is missing block_index/block_idx; "
            "step keys are denoise-step identities and cannot be used as block IDs"
        )
    block_index = _int_scalar(value, field="block_index")
    if block_index < 0:
        raise MissingBlockTPFMetadataError(f"block_index must be non-negative, got {block_index}")
    return block_index


def extract_rollout_block_tpf_summaries(
    rollout_states: Mapping[Any, Any],
) -> dict[int, BlockTPFSummary]:
    """Extract and validate one canonical TPF summary per response block."""

    if not isinstance(rollout_states, Mapping) or not rollout_states:
        raise MissingBlockTPFMetadataError("rollout_self_conditioning is missing or empty")

    row_new_accept: dict[int, int] = {}
    row_forwards: dict[int, int] = {}
    exact_summaries: dict[int, set[tuple[int, int]]] = {}

    for state in rollout_states.values():
        if not isinstance(state, Mapping):
            continue
        block_index = rollout_state_block_index(state)

        exact_new_accept = _mapping_value(state, "new_accept_tokens")
        exact_forwards = _mapping_value(state, "denoise_forwards")
        if (exact_new_accept is None) != (exact_forwards is None):
            raise ValueError(
                f"block {block_index} must provide both new_accept_tokens and denoise_forwards"
            )
        if exact_new_accept is not None:
            exact_summaries.setdefault(block_index, set()).add(
                (
                    _int_scalar(exact_new_accept, field="new_accept_tokens"),
                    _int_scalar(exact_forwards, field="denoise_forwards"),
                )
            )

        if bool(_mapping_value(state, "is_terminal_commit") or False):
            continue
        new_accept_count = _mapping_value(state, "new_accept_count")
        if new_accept_count is None:
            raise MissingBlockTPFMetadataError(
                f"non-terminal denoise row for block {block_index} is missing new_accept_count"
            )
        row_new_accept[block_index] = row_new_accept.get(block_index, 0) + _int_scalar(
            new_accept_count,
            field="new_accept_count",
        )
        row_forwards[block_index] = row_forwards.get(block_index, 0) + 1

    block_indices = sorted(set(row_forwards) | set(exact_summaries))
    summaries: dict[int, BlockTPFSummary] = {}
    for block_index in block_indices:
        exact_values = exact_summaries.get(block_index, set())
        if len(exact_values) > 1:
            raise ValueError(
                f"inconsistent producer block summaries for block {block_index}: "
                f"{sorted(exact_values)}"
            )
        if exact_values:
            new_accept_tokens, denoise_forwards = next(iter(exact_values))
            if denoise_forwards <= 0:
                raise ValueError(
                    f"producer block summary for block {block_index} has "
                    f"denoise_forwards={denoise_forwards}"
                )
            observed_new_accept = row_new_accept.get(block_index, 0)
            observed_forwards = row_forwards.get(block_index, 0)
            if observed_forwards > denoise_forwards:
                raise ValueError(
                    f"producer block summary mismatch for block {block_index}: "
                    f"summary=({new_accept_tokens}, {denoise_forwards}) "
                    f"rows=({observed_new_accept}, {observed_forwards})"
                )
            # Metadata-first callers may retain only a subset of per-step rows
            # while carrying the exact producer block summary on each retained
            # row. Validate row sums only when the complete denoise trace is
            # present; otherwise the exact summary is authoritative.
            if (
                observed_forwards == denoise_forwards
                and observed_new_accept != new_accept_tokens
            ):
                raise ValueError(
                    f"producer block summary mismatch for block {block_index}: "
                    f"summary=({new_accept_tokens}, {denoise_forwards}) "
                    f"rows=({observed_new_accept}, {observed_forwards})"
                )
        else:
            new_accept_tokens = row_new_accept.get(block_index, 0)
            denoise_forwards = row_forwards.get(block_index, 0)
            if denoise_forwards <= 0:
                continue
        summaries[block_index] = BlockTPFSummary(
            block_index=block_index,
            new_accept_tokens=new_accept_tokens,
            denoise_forwards=denoise_forwards,
        )

    if not summaries:
        raise MissingBlockTPFMetadataError("no non-terminal denoise block summaries are available")
    return summaries


def _row_value(values: Any, row_index: int) -> Any:
    try:
        return values[row_index]
    except (IndexError, KeyError, TypeError):
        return None


def _state_for_step(rollout_states: Mapping[Any, Any], step_key: int) -> Mapping[str, Any] | None:
    state = rollout_states.get(step_key)
    if state is None:
        state = rollout_states.get(str(step_key))
    return state if isinstance(state, Mapping) else None


def build_block_tpf_token_tensors(
    *,
    rollout_self_conditioning: Sequence[Any],
    response_identities: Sequence[Any],
    step_maps: torch.Tensor,
    active_token_mask: torch.Tensor,
) -> BlockTPFTokenTensors:
    """Map exact rollout block metadata to active CE token positions."""

    if step_maps.ndim != 2 or active_token_mask.shape != step_maps.shape:
        raise ValueError(
            "step_maps and active_token_mask must be shape-aligned rank-2 tensors, "
            f"got {tuple(step_maps.shape)} and {tuple(active_token_mask.shape)}"
        )

    rows, response_length = step_maps.shape
    device = step_maps.device
    raw_tpf = torch.full((rows, response_length), -1.0, dtype=torch.float32, device=device)
    response_hashes = torch.full(
        (rows, response_length, 2),
        -1,
        dtype=torch.long,
        device=device,
    )
    block_indices = torch.full(
        (rows, response_length),
        -1,
        dtype=torch.long,
        device=device,
    )
    active_token_mask = active_token_mask.to(device=device, dtype=torch.bool)
    ghost_rows_excluded = 0

    for row_index in range(rows):
        active_positions = torch.nonzero(active_token_mask[row_index], as_tuple=False).flatten()
        if active_positions.numel() == 0:
            ghost_rows_excluded += 1
            continue

        rollout_states = _row_value(rollout_self_conditioning, row_index)
        if not isinstance(rollout_states, Mapping):
            raise MissingBlockTPFMetadataError(
                f"row {row_index} has active CE tokens but no rollout_self_conditioning"
            )
        response_identity = _row_value(response_identities, row_index)
        hash_lanes = stable_response_identity_hash(response_identity)
        summaries = extract_rollout_block_tpf_summaries(rollout_states)

        step_values = step_maps[row_index, active_positions].detach().cpu().tolist()
        positions_by_step: dict[int, list[int]] = {}
        for position, raw_step_key in zip(active_positions.detach().cpu().tolist(), step_values, strict=True):
            step_key = int(raw_step_key)
            if step_key < 0:
                raise MissingBlockTPFMetadataError(
                    f"row {row_index} active response position {position} has no denoise step key"
                )
            positions_by_step.setdefault(step_key, []).append(int(position))

        for step_key, positions in positions_by_step.items():
            state = _state_for_step(rollout_states, step_key)
            if state is None:
                raise MissingBlockTPFMetadataError(
                    f"row {row_index} denoise step {step_key} is absent from rollout metadata"
                )
            block_index = rollout_state_block_index(state)
            summary = summaries.get(block_index)
            if summary is None:
                raise MissingBlockTPFMetadataError(
                    f"row {row_index} block {block_index} has no canonical TPF summary"
                )
            index = torch.tensor(positions, dtype=torch.long, device=device)
            raw_tpf[row_index, index] = float(summary.raw_tpf)
            response_hashes[row_index, index, 0] = hash_lanes[0]
            response_hashes[row_index, index, 1] = hash_lanes[1]
            block_indices[row_index, index] = block_index

    return BlockTPFTokenTensors(
        raw_tpf=raw_tpf,
        response_hashes=response_hashes,
        block_indices=block_indices,
        active_mask=active_token_mask,
        ghost_rows_excluded=ghost_rows_excluded,
    )


def extract_unique_block_tpf_records(
    raw_tpf: torch.Tensor,
    response_hashes: torch.Tensor,
    block_indices: torch.Tensor,
    active_mask: torch.Tensor,
) -> BlockTPFRecords:
    """Deduplicate repeated token/row values into unique block records."""

    if raw_tpf.ndim != 2:
        raise ValueError(f"raw_tpf must be rank 2, got shape={tuple(raw_tpf.shape)}")
    if response_hashes.shape != (*raw_tpf.shape, 2):
        raise ValueError("response_hashes must have shape [rows, response_length, 2]")
    if block_indices.shape != raw_tpf.shape or active_mask.shape != raw_tpf.shape:
        raise ValueError("block_indices and active_mask must match raw_tpf")

    valid = (
        active_mask.bool()
        & raw_tpf.ge(0.0)
        & block_indices.ge(0)
        & response_hashes[..., 0].ge(0)
        & response_hashes[..., 1].ge(0)
    )
    table: dict[tuple[int, int, int], float] = {}
    for index in torch.nonzero(valid, as_tuple=False).detach().cpu().tolist():
        row, position = index
        key = (
            int(response_hashes[row, position, 0].detach().cpu().item()),
            int(response_hashes[row, position, 1].detach().cpu().item()),
            int(block_indices[row, position].detach().cpu().item()),
        )
        value = float(raw_tpf[row, position].detach().cpu().item())
        previous = table.get(key)
        if previous is not None and not math.isclose(previous, value, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"inconsistent raw TPF for block key {key}: {previous} vs {value}")
        table[key] = value

    keys = sorted(table)
    if not keys:
        return BlockTPFRecords(
            keys=torch.empty((0, 3), dtype=torch.long, device=raw_tpf.device),
            values=torch.empty((0,), dtype=torch.float64, device=raw_tpf.device),
        )
    return BlockTPFRecords(
        keys=torch.tensor(keys, dtype=torch.long, device=raw_tpf.device),
        values=torch.tensor([table[key] for key in keys], dtype=torch.float64, device=raw_tpf.device),
    )


def merge_unique_block_tpf_records(
    record_batches: Sequence[BlockTPFRecords],
    *,
    device: torch.device | None = None,
) -> BlockTPFRecords:
    """Merge per-rank records while counting each block identity once."""

    table: dict[tuple[int, int, int], float] = {}
    for records in record_batches:
        if records.keys.ndim != 2 or records.keys.shape[-1] != 3:
            raise ValueError("block TPF record keys must have shape [blocks, 3]")
        if records.values.shape != records.keys.shape[:1]:
            raise ValueError("block TPF record values must match the key count")
        for key_values, raw_value in zip(
            records.keys.detach().cpu().tolist(),
            records.values.detach().cpu().tolist(),
            strict=True,
        ):
            key = tuple(int(value) for value in key_values)
            value = float(raw_value)
            previous = table.get(key)
            if previous is not None and not math.isclose(previous, value, rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError(
                    f"inconsistent cross-rank raw TPF for block key {key}: {previous} vs {value}"
                )
            table[key] = value

    if device is None:
        device = record_batches[0].keys.device if record_batches else torch.device("cpu")
    keys = sorted(table)
    if not keys:
        return BlockTPFRecords(
            keys=torch.empty((0, 3), dtype=torch.long, device=device),
            values=torch.empty((0,), dtype=torch.float64, device=device),
        )
    return BlockTPFRecords(
        keys=torch.tensor(keys, dtype=torch.long, device=device),
        values=torch.tensor([table[key] for key in keys], dtype=torch.float64, device=device),
    )


def resolve_block_tpf_metadata_policy(
    *,
    missing: bool,
    invalid: bool,
    policy: str,
    detail: str = "",
) -> bool:
    """Return whether weighting remains enabled, or raise for fatal metadata."""

    policy = str(policy).strip().lower()
    if policy not in {"error", "disable"}:
        raise ValueError("block TPF missing metadata policy must be error or disable")
    suffix = f": {detail}" if detail else ""
    if invalid:
        raise RuntimeError(f"invalid block TPF metadata{suffix}")
    if missing:
        if policy == "error":
            raise RuntimeError(f"missing block TPF metadata{suffix}")
        return False
    return True


def _gather_block_tpf_records(local_records: BlockTPFRecords) -> list[BlockTPFRecords]:
    if not torch.distributed.is_initialized() or torch.distributed.get_world_size() <= 1:
        return [local_records]

    world_size = torch.distributed.get_world_size()
    device = local_records.keys.device
    local_count = torch.tensor([local_records.keys.shape[0]], dtype=torch.long, device=device)
    gathered_counts = [torch.zeros_like(local_count) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_counts, local_count)
    counts = [int(value.item()) for value in gathered_counts]
    max_count = max(counts, default=0)
    if max_count <= 0:
        return [
            BlockTPFRecords(
                keys=torch.empty((0, 3), dtype=torch.long, device=device),
                values=torch.empty((0,), dtype=torch.float64, device=device),
            )
            for _ in range(world_size)
        ]

    padded_keys = torch.full((max_count, 3), -1, dtype=torch.long, device=device)
    padded_values = torch.zeros((max_count,), dtype=torch.float64, device=device)
    if local_records.keys.shape[0] > 0:
        padded_keys[: local_records.keys.shape[0]] = local_records.keys
        padded_values[: local_records.values.shape[0]] = local_records.values
    gathered_keys = [torch.empty_like(padded_keys) for _ in range(world_size)]
    gathered_values = [torch.empty_like(padded_values) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_keys, padded_keys)
    torch.distributed.all_gather(gathered_values, padded_values)
    return [
        BlockTPFRecords(keys=keys[:count], values=values[:count])
        for keys, values, count in zip(gathered_keys, gathered_values, counts, strict=True)
    ]


def _collective_device(fallback: torch.device) -> torch.device:
    if not torch.distributed.is_initialized():
        return fallback
    if str(torch.distributed.get_backend()).lower() == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return fallback


def normalize_block_tpf_weights(
    *,
    raw_tpf: torch.Tensor,
    response_hashes: torch.Tensor,
    block_indices: torch.Tensor,
    active_mask: torch.Tensor,
    min_weight: float,
    max_weight: float,
    ghost_rows_excluded: int = 0,
    gathered_record_batches: Sequence[BlockTPFRecords] | None = None,
) -> BlockTPFNormalization:
    """Normalize by the distributed mean over unique real response blocks."""

    min_weight = float(min_weight)
    max_weight = float(max_weight)
    if not math.isfinite(min_weight) or not math.isfinite(max_weight):
        raise ValueError("block TPF weight bounds must be finite")
    if min_weight <= 0.0 or max_weight < min_weight:
        raise ValueError(
            f"block TPF weight bounds must satisfy 0 < min <= max, got {min_weight}, {max_weight}"
        )

    local_records = extract_unique_block_tpf_records(
        raw_tpf,
        response_hashes,
        block_indices,
        active_mask,
    )
    collective_device = _collective_device(raw_tpf.device)
    if local_records.keys.device != collective_device:
        local_records = BlockTPFRecords(
            keys=local_records.keys.to(collective_device),
            values=local_records.values.to(collective_device),
        )
    record_batches = (
        list(gathered_record_batches)
        if gathered_record_batches is not None
        else _gather_block_tpf_records(local_records)
    )
    global_records = merge_unique_block_tpf_records(
        record_batches,
        device=collective_device,
    )
    if global_records.values.numel() <= 0:
        raise MissingBlockTPFMetadataError("distributed rollout batch has no active real TPF blocks")

    global_mean = float(global_records.values.mean().item())
    if not math.isfinite(global_mean) or global_mean <= 0.0:
        raise ValueError(f"global block TPF mean must be finite and positive, got {global_mean}")

    valid = (
        active_mask.bool()
        & raw_tpf.ge(0.0)
        & block_indices.ge(0)
        & response_hashes[..., 0].ge(0)
        & response_hashes[..., 1].ge(0)
    )
    weights = torch.ones_like(raw_tpf, dtype=torch.float32)
    weights[valid] = (raw_tpf[valid].float() / global_mean).clamp(min_weight, max_weight)

    block_weights = (global_records.values / global_mean).clamp(min_weight, max_weight)
    metric_counts = torch.tensor(
        [
            int(valid.sum().detach().cpu().item()),
            int(ghost_rows_excluded),
        ],
        dtype=torch.long,
        device=collective_device,
    )
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        torch.distributed.all_reduce(metric_counts, op=torch.distributed.ReduceOp.SUM)

    return BlockTPFNormalization(
        weights=weights.detach(),
        global_mean=global_mean,
        global_active_blocks=int(global_records.values.numel()),
        global_active_tokens=int(metric_counts[0].item()),
        global_ghost_rows_excluded=int(metric_counts[1].item()),
        raw_min=float(global_records.values.min().item()),
        raw_max=float(global_records.values.max().item()),
        weight_min=float(block_weights.min().item()),
        weight_mean=float(block_weights.mean().item()),
        weight_max=float(block_weights.max().item()),
    )
