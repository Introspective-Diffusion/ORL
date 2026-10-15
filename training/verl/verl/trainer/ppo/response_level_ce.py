"""Equal-response weighting for packed trace-all-CE training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor

from verl.trainer.ppo.block_tpf_ce_weight import (
    MissingBlockTPFMetadataError,
    stable_response_identity_hash,
)


@dataclass(frozen=True)
class ResponseCountRecords:
    keys: Tensor
    counts: Tensor


@dataclass(frozen=True)
class ResponseLevelCEWeights:
    weights: Tensor
    global_responses: int
    global_active_tokens: int
    min_tokens_per_response: int
    max_tokens_per_response: int


def extract_local_response_count_records(
    response_uids: Sequence[Any],
    response_mask: Tensor,
    *,
    contribute: bool = True,
) -> ResponseCountRecords:
    """Sum this rank's active-token ownership by original response."""

    if response_mask.ndim != 2:
        raise ValueError(
            f"response_mask must be [rows, length], got {tuple(response_mask.shape)}"
        )
    if len(response_uids) != int(response_mask.shape[0]):
        raise ValueError(
            f"response_uid rows={len(response_uids)} do not match "
            f"mask rows={response_mask.shape[0]}"
        )
    table: dict[tuple[int, int], int] = {}
    if contribute:
        counts = response_mask.bool().sum(dim=-1).detach().cpu().tolist()
        local_identities: dict[tuple[int, int], str] = {}
        for uid, count in zip(response_uids, counts, strict=True):
            if int(count) <= 0:
                continue
            try:
                key = stable_response_identity_hash(uid)
            except MissingBlockTPFMetadataError as exc:
                raise RuntimeError(
                    "response-level CE requires a non-empty response_uid"
                ) from exc
            identity = str(uid)
            previous_identity = local_identities.get(key)
            if previous_identity is not None and previous_identity != identity:
                raise RuntimeError(
                    "response-level CE detected a local response hash collision"
                )
            local_identities[key] = identity
            table[key] = table.get(key, 0) + int(count)

    device = response_mask.device
    keys = sorted(table)
    if not keys:
        return ResponseCountRecords(
            keys=torch.empty((0, 2), dtype=torch.long, device=device),
            counts=torch.empty((0,), dtype=torch.float64, device=device),
        )
    return ResponseCountRecords(
        keys=torch.tensor(keys, dtype=torch.long, device=device),
        counts=torch.tensor(
            [table[key] for key in keys],
            dtype=torch.float64,
            device=device,
        ),
    )


def merge_response_count_records(
    record_batches: Sequence[ResponseCountRecords],
    *,
    device: torch.device | None = None,
) -> ResponseCountRecords:
    """Sum disjoint rank-local token ownership for each response."""

    table: dict[tuple[int, int], float] = {}
    for records in record_batches:
        if records.keys.ndim != 2 or records.keys.shape[-1] != 2:
            raise ValueError("response count keys must have shape [responses, 2]")
        if records.counts.shape != records.keys.shape[:1]:
            raise ValueError("response counts must match the key count")
        for raw_key, raw_count in zip(
            records.keys.detach().cpu().tolist(),
            records.counts.detach().cpu().tolist(),
            strict=True,
        ):
            key = (int(raw_key[0]), int(raw_key[1]))
            count = float(raw_count)
            if count <= 0.0:
                raise ValueError(
                    f"response token count must be positive, got {count}"
                )
            table[key] = table.get(key, 0.0) + count

    if device is None:
        device = (
            record_batches[0].keys.device
            if record_batches
            else torch.device("cpu")
        )
    keys = sorted(table)
    if not keys:
        return ResponseCountRecords(
            keys=torch.empty((0, 2), dtype=torch.long, device=device),
            counts=torch.empty((0,), dtype=torch.float64, device=device),
        )
    return ResponseCountRecords(
        keys=torch.tensor(keys, dtype=torch.long, device=device),
        counts=torch.tensor(
            [table[key] for key in keys],
            dtype=torch.float64,
            device=device,
        ),
    )


def _collective_device(fallback: torch.device) -> torch.device:
    if not torch.distributed.is_initialized():
        return fallback
    if str(torch.distributed.get_backend()).lower() == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return fallback


def gather_response_count_records(
    local_records: ResponseCountRecords,
) -> list[ResponseCountRecords]:
    """All-gather variable-length response-count tables."""

    if (
        not torch.distributed.is_initialized()
        or torch.distributed.get_world_size() <= 1
    ):
        return [local_records]
    world_size = torch.distributed.get_world_size()
    device = _collective_device(local_records.keys.device)
    local_keys = local_records.keys.to(device=device)
    local_counts = local_records.counts.to(device=device)
    local_size = torch.tensor(
        [local_keys.shape[0]],
        dtype=torch.long,
        device=device,
    )
    gathered_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_sizes, local_size)
    sizes = [int(value.item()) for value in gathered_sizes]
    max_size = max(sizes, default=0)
    if max_size <= 0:
        return [
            ResponseCountRecords(
                keys=torch.empty((0, 2), dtype=torch.long, device=device),
                counts=torch.empty((0,), dtype=torch.float64, device=device),
            )
            for _ in range(world_size)
        ]

    padded_keys = torch.full(
        (max_size, 2),
        -1,
        dtype=torch.long,
        device=device,
    )
    padded_counts = torch.zeros(
        (max_size,),
        dtype=torch.float64,
        device=device,
    )
    if local_keys.shape[0] > 0:
        padded_keys[: local_keys.shape[0]] = local_keys
        padded_counts[: local_counts.shape[0]] = local_counts
    gathered_keys = [torch.empty_like(padded_keys) for _ in range(world_size)]
    gathered_counts = [
        torch.empty_like(padded_counts) for _ in range(world_size)
    ]
    torch.distributed.all_gather(gathered_keys, padded_keys)
    torch.distributed.all_gather(gathered_counts, padded_counts)
    return [
        ResponseCountRecords(
            keys=keys[:size],
            counts=counts[:size],
        )
        for keys, counts, size in zip(
            gathered_keys,
            gathered_counts,
            sizes,
            strict=True,
        )
    ]


def build_response_level_ce_weights(
    response_uids: Sequence[Any],
    response_mask: Tensor,
    *,
    sequence_parallel_rank: int = 0,
    gathered_record_batches: Sequence[ResponseCountRecords] | None = None,
) -> ResponseLevelCEWeights:
    """Build ``1/N_response`` weights for every active token."""

    local = extract_local_response_count_records(
        response_uids,
        response_mask,
        contribute=int(sequence_parallel_rank) == 0,
    )
    gathered = (
        list(gathered_record_batches)
        if gathered_record_batches is not None
        else gather_response_count_records(local)
    )
    merged = merge_response_count_records(
        gathered,
        device=response_mask.device,
    )
    if merged.keys.shape[0] <= 0:
        raise RuntimeError("response-level CE found zero globally active responses")
    table = {
        (int(key[0]), int(key[1])): float(count)
        for key, count in zip(
            merged.keys.detach().cpu().tolist(),
            merged.counts.detach().cpu().tolist(),
            strict=True,
        )
    }
    weights = torch.zeros_like(response_mask, dtype=torch.float32)
    for row, uid in enumerate(response_uids):
        active = response_mask[row].bool()
        if not bool(active.any()):
            continue
        key = stable_response_identity_hash(uid)
        count = table.get(key)
        if count is None or count <= 0.0:
            raise RuntimeError(
                f"response-level CE is missing global count for response {uid!r}"
            )
        weights[row, active] = 1.0 / count

    global_counts = merged.counts.detach().cpu()
    return ResponseLevelCEWeights(
        weights=weights,
        global_responses=int(merged.keys.shape[0]),
        global_active_tokens=int(round(float(global_counts.sum().item()))),
        min_tokens_per_response=int(round(float(global_counts.min().item()))),
        max_tokens_per_response=int(round(float(global_counts.max().item()))),
    )


# Running count of steps with zero globally valid AEC responses.
_AEC_ZERO_RESP_SKIP_COUNT = 0


def build_response_level_aec_pair_weights(
    response_uids: Sequence[Any],
    pair_mask: Tensor,
    pair_ids: Tensor,
    *,
    sequence_parallel_rank: int = 0,
    gathered_record_batches: Sequence[ResponseCountRecords] | None = None,
) -> ResponseLevelCEWeights:
    """Build ``1/N_pairs(response)`` weights at every valid AEC pair token."""

    if pair_mask.shape != pair_ids.shape or pair_mask.ndim != 2:
        raise ValueError(
            "AEC pair mask/ids must be shape-aligned rank-2 tensors"
        )
    if len(response_uids) != int(pair_mask.shape[0]):
        raise ValueError("response_uid rows do not match AEC pair rows")
    local_sets: dict[tuple[int, int], set[int]] = {}
    if int(sequence_parallel_rank) == 0:
        for row, uid in enumerate(response_uids):
            valid_ids = pair_ids[row][
                pair_mask[row].bool() & pair_ids[row].ge(0)
            ].detach().cpu().unique().tolist()
            if not valid_ids:
                continue
            key = stable_response_identity_hash(uid)
            local_sets.setdefault(key, set()).update(
                int(value) for value in valid_ids
            )
    device = pair_mask.device
    local_keys = sorted(local_sets)
    local_records = ResponseCountRecords(
        keys=(
            torch.tensor(local_keys, dtype=torch.long, device=device)
            if local_keys
            else torch.empty((0, 2), dtype=torch.long, device=device)
        ),
        counts=(
            torch.tensor(
                [len(local_sets[key]) for key in local_keys],
                dtype=torch.float64,
                device=device,
            )
            if local_keys
            else torch.empty((0,), dtype=torch.float64, device=device)
        ),
    )
    gathered = (
        list(gathered_record_batches)
        if gathered_record_batches is not None
        else gather_response_count_records(local_records)
    )
    merged = merge_response_count_records(gathered, device=device)
    if merged.keys.shape[0] <= 0:
        # No response in this step has a pairable adjacent denoising state.
        # Zero weights make the AEC term contribute exactly zero to backward;
        # the dp_actor zero-pair pre-pass then disables AEC for this step only.
        # merged.counts is empty here, so the count fields are filled with 0.
        global _AEC_ZERO_RESP_SKIP_COUNT
        try:
            _AEC_ZERO_RESP_SKIP_COUNT += 1
        except NameError:
            _AEC_ZERO_RESP_SKIP_COUNT = 1
        print(
            "[AEC-ZERO-RESP-SKIP] zero globally valid AEC responses this step; "
            "AEC contributes zero (occurrences so far: %d)"
            % (_AEC_ZERO_RESP_SKIP_COUNT,),
            flush=True,
        )
        return ResponseLevelCEWeights(
            weights=torch.zeros_like(pair_mask, dtype=torch.float32),
            global_responses=0,
            global_active_tokens=0,
            min_tokens_per_response=0,
            max_tokens_per_response=0,
        )
    table = {
        (int(key[0]), int(key[1])): float(count)
        for key, count in zip(
            merged.keys.detach().cpu().tolist(),
            merged.counts.detach().cpu().tolist(),
            strict=True,
        )
    }
    weights = torch.zeros_like(pair_mask, dtype=torch.float32)
    for row, uid in enumerate(response_uids):
        valid = pair_mask[row].bool() & pair_ids[row].ge(0)
        if not bool(valid.any()):
            continue
        count = table.get(stable_response_identity_hash(uid))
        if count is None or count <= 0.0:
            raise RuntimeError(
                f"response-level AEC is missing pair count for {uid!r}"
            )
        weights[row, valid] = 1.0 / count
    counts = merged.counts.detach().cpu()
    return ResponseLevelCEWeights(
        weights=weights,
        global_responses=int(merged.keys.shape[0]),
        global_active_tokens=int(round(float(counts.sum().item()))),
        min_tokens_per_response=int(round(float(counts.min().item()))),
        max_tokens_per_response=int(round(float(counts.max().item()))),
    )


__all__ = [
    "ResponseCountRecords",
    "ResponseLevelCEWeights",
    "build_response_level_aec_pair_weights",
    "build_response_level_ce_weights",
    "extract_local_response_count_records",
    "gather_response_count_records",
    "merge_response_count_records",
]
