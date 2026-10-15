"""Manifest primitives for hybrid packed replay scheduling.

This module is intentionally independent of Ray, actor, rollout, and SGLang
classes.  All structures are pure-Python, frozen dataclasses that round-trip
cleanly through JSON/msgpack non-tensor payloads.  No dense hidden tensors
are embedded.

Public API
----------
BlockPageRef
    One contiguous page of a response block within a packed sequence.
PackedRowManifest
    Descriptor for one response row laid out in a packed round.
page_ref_cost
    Scalar cost estimate for a single page reference.
manifest_cost
    Scalar cost estimate for a single manifest row.
assign_manifests_to_ranks
    Greedy cost-balanced assignment of manifests across N ranks.
validate_non_overlapping_spans
    Raise ValueError if any two pages within a row overlap.
manifest_to_mapping / manifest_from_mapping
    Serializable roundtrip suitable for Ray non_tensor_batch payloads.
page_ref_to_mapping / page_ref_from_mapping
    Same for individual page references.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

# ---------------------------------------------------------------------------
# Tuneable cost constants (module-level, not config-driven, to keep the
# interface minimal while remaining easy to override in tests).
# ---------------------------------------------------------------------------

_BYTES_PER_PAGE_TOKEN: int = 2  # conservative estimate; one bfloat16 element
_PAGE_OVERHEAD_TOKENS: int = 4  # fixed scheduling overhead per page reference


# ---------------------------------------------------------------------------
# BlockPageRef
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockPageRef:
    """One contiguous page of a response block within a packed sequence.

    Parameters
    ----------
    block_index:
        Index of the source diffusion block (``>= 0``).
    step_key:
        Opaque string that identifies the denoising step within the block.
    response_start:
        Token offset of this page within the *response* span of the packed row
        (``>= 0``).
    page_length:
        Number of tokens in this page (``> 0``).
    source_start:
        Token offset within the *source* block replay buffer (``>= 0``).
    tensor_path:
        Optional filesystem path to the backing tensor store.
    trace_root:
        Optional root directory for block trace artefacts.
    """

    block_index: int
    step_key: str
    response_start: int
    page_length: int
    source_start: int
    tensor_path: str | None = None
    trace_root: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.block_index, int) or self.block_index < 0:
            raise ValueError(
                f"block_index must be a non-negative int, got {self.block_index!r}"
            )
        if not isinstance(self.step_key, str):
            raise ValueError(
                f"step_key must be a str, got {type(self.step_key)!r}"
            )
        if not isinstance(self.response_start, int) or self.response_start < 0:
            raise ValueError(
                f"response_start must be a non-negative int, got {self.response_start!r}"
            )
        if not isinstance(self.page_length, int) or self.page_length <= 0:
            raise ValueError(
                f"page_length must be a positive int, got {self.page_length!r}"
            )
        if not isinstance(self.source_start, int) or self.source_start < 0:
            raise ValueError(
                f"source_start must be a non-negative int, got {self.source_start!r}"
            )
        if self.tensor_path is not None and not isinstance(self.tensor_path, str):
            raise ValueError(
                f"tensor_path must be str or None, got {type(self.tensor_path)!r}"
            )
        if self.trace_root is not None and not isinstance(self.trace_root, str):
            raise ValueError(
                f"trace_root must be str or None, got {type(self.trace_root)!r}"
            )
        if self.tensor_path is not None and self.trace_root is not None:
            root = os.path.realpath(self.trace_root)
            raw_path = self.tensor_path
            path = os.path.realpath(
                raw_path
                if os.path.isabs(raw_path)
                else os.path.join(root, raw_path)
            )
            try:
                contained = os.path.commonpath((root, path)) == root
            except ValueError:
                contained = False
            if not contained:
                raise ValueError(
                    "tensor_path escapes trace_root: "
                    f"path={self.tensor_path!r} root={self.trace_root!r}"
                )
            object.__setattr__(self, "trace_root", root)
            object.__setattr__(self, "tensor_path", path)

    @property
    def response_end(self) -> int:
        """Exclusive end offset within the response span."""
        return self.response_start + self.page_length


# ---------------------------------------------------------------------------
# PackedRowManifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PackedRowManifest:
    """Descriptor for one response row laid out in a hybrid packed round.

    Parameters
    ----------
    sample_index:
        Position of this sample in the originating data batch (``>= 0``).
    response_uid:
        Opaque string that uniquely identifies the generation response.
    packed_round_index:
        Index of the packing round this row belongs to (``>= 0``).
    pages:
        Ordered tuple of :class:`BlockPageRef` objects describing the pages
        that compose this row's response span.
    active_label_tokens:
        Number of tokens that carry a cross-entropy label (``>= 0``).
    sequence_tokens:
        Total token count for the full sequence (prompt + response) (``> 0``).
    """

    sample_index: int
    response_uid: str
    packed_round_index: int
    pages: tuple[BlockPageRef, ...]
    active_label_tokens: int
    sequence_tokens: int

    def __post_init__(self) -> None:
        if not isinstance(self.sample_index, int) or self.sample_index < 0:
            raise ValueError(
                f"sample_index must be a non-negative int, got {self.sample_index!r}"
            )
        if not isinstance(self.response_uid, str):
            raise ValueError(
                f"response_uid must be a str, got {type(self.response_uid)!r}"
            )
        if not isinstance(self.packed_round_index, int) or self.packed_round_index < 0:
            raise ValueError(
                f"packed_round_index must be a non-negative int, "
                f"got {self.packed_round_index!r}"
            )
        if not isinstance(self.pages, tuple):
            raise ValueError(
                f"pages must be a tuple, got {type(self.pages)!r}"
            )
        for i, p in enumerate(self.pages):
            if not isinstance(p, BlockPageRef):
                raise ValueError(
                    f"pages[{i}] must be a BlockPageRef, got {type(p)!r}"
                )
        if not isinstance(self.active_label_tokens, int) or self.active_label_tokens < 0:
            raise ValueError(
                f"active_label_tokens must be a non-negative int, "
                f"got {self.active_label_tokens!r}"
            )
        if not isinstance(self.sequence_tokens, int) or self.sequence_tokens <= 0:
            raise ValueError(
                f"sequence_tokens must be a positive int, got {self.sequence_tokens!r}"
            )

    # ------------------------------------------------------------------
    # Stable identity key (deterministic, hashable)
    # ------------------------------------------------------------------

    @property
    def identity_key(self) -> tuple[str, int, int]:
        """Deterministic key for equality / tie-breaking: (response_uid, packed_round_index, sample_index)."""
        return (self.response_uid, self.packed_round_index, self.sample_index)


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------


def page_ref_cost(page: BlockPageRef) -> int:
    """Scalar cost estimate for one page reference.

    Cost accounts for per-token bytes and a fixed scheduling overhead.
    """
    return page.page_length * _BYTES_PER_PAGE_TOKEN + _PAGE_OVERHEAD_TOKENS


def manifest_cost(manifest: PackedRowManifest) -> int:
    """Scalar cost estimate for one manifest row.

    Combines sequence layout cost, active label overhead, and page I/O cost.
    """
    seq_cost = manifest.sequence_tokens
    label_cost = manifest.active_label_tokens
    page_io_cost = sum(page_ref_cost(p) for p in manifest.pages)
    return seq_cost + label_cost + page_io_cost


# ---------------------------------------------------------------------------
# Validation: non-overlapping response spans
# ---------------------------------------------------------------------------


def validate_non_overlapping_spans(manifest: PackedRowManifest) -> None:
    """Raise :exc:`ValueError` if any two pages within *manifest* overlap.

    Overlap is defined as having intersecting half-open intervals
    ``[response_start, response_end)``.
    """
    pages = sorted(manifest.pages, key=lambda p: (p.response_start, p.response_end))
    for i in range(len(pages) - 1):
        a, b = pages[i], pages[i + 1]
        if a.response_end > b.response_start:
            raise ValueError(
                f"Overlapping response spans in manifest {manifest.response_uid!r}: "
                f"page {i} covers [{a.response_start}, {a.response_end}) "
                f"and page {i + 1} covers [{b.response_start}, {b.response_end})"
            )


# ---------------------------------------------------------------------------
# Greedy cost-balanced rank assignment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RankAssignment:
    """Result of assigning one manifest to a rank."""

    rank: int
    manifest: PackedRowManifest


@dataclass(frozen=True)
class AssignmentResult:
    """Output of :func:`assign_manifests_to_ranks`."""

    assignments: tuple[RankAssignment, ...]
    rank_costs: tuple[int, ...]
    rank_row_counts: tuple[int, ...]

    @property
    def num_ranks(self) -> int:
        return len(self.rank_costs)

    def manifests_for_rank(self, rank: int) -> list[PackedRowManifest]:
        return [a.manifest for a in self.assignments if a.rank == rank]


def assign_manifests_to_ranks(
    manifests: Sequence[PackedRowManifest],
    num_ranks: int,
) -> AssignmentResult:
    """Deterministic greedy cost-balanced assignment of manifests to ranks.

    Rows are processed in descending order of cost.  Ties are broken by
    ``identity_key`` (lexicographic) to guarantee a stable, reproducible
    assignment across workers.

    Parameters
    ----------
    manifests:
        Sequence of :class:`PackedRowManifest` objects to assign.
    num_ranks:
        Number of target ranks (``>= 1``).

    Returns
    -------
    AssignmentResult
        Immutable result containing per-manifest rank assignments and per-rank
        cost/row-count metrics.
    """
    if num_ranks < 1:
        raise ValueError(f"num_ranks must be >= 1, got {num_ranks!r}")

    costs = [(manifest_cost(m), m.identity_key, m) for m in manifests]
    # Stable sort: descending cost, then ascending identity_key for ties.
    costs.sort(key=lambda x: (-x[0], x[1]))

    rank_costs: list[int] = [0] * num_ranks
    rank_row_counts: list[int] = [0] * num_ranks
    assignments: list[RankAssignment] = []

    for cost, _key, m in costs:
        # Assign to the rank with the lowest accumulated cost; use rank index
        # as a tie-break to keep assignments stable.
        best_rank = min(range(num_ranks), key=lambda r: (rank_costs[r], r))
        assignments.append(RankAssignment(rank=best_rank, manifest=m))
        rank_costs[best_rank] += cost
        rank_row_counts[best_rank] += 1

    return AssignmentResult(
        assignments=tuple(assignments),
        rank_costs=tuple(rank_costs),
        rank_row_counts=tuple(rank_row_counts),
    )


# ---------------------------------------------------------------------------
# Serialization roundtrip (no dense tensors)
# ---------------------------------------------------------------------------


def page_ref_to_mapping(page: BlockPageRef) -> dict[str, Any]:
    """Convert a :class:`BlockPageRef` to a plain JSON-serialisable dict."""
    return {
        "block_index": page.block_index,
        "step_key": page.step_key,
        "response_start": page.response_start,
        "page_length": page.page_length,
        "source_start": page.source_start,
        "tensor_path": page.tensor_path,
        "trace_root": page.trace_root,
    }


def page_ref_from_mapping(d: Mapping[str, Any]) -> BlockPageRef:
    """Reconstruct a :class:`BlockPageRef` from a plain mapping."""
    return BlockPageRef(
        block_index=int(d["block_index"]),
        step_key=str(d["step_key"]),
        response_start=int(d["response_start"]),
        page_length=int(d["page_length"]),
        source_start=int(d["source_start"]),
        tensor_path=d.get("tensor_path"),
        trace_root=d.get("trace_root"),
    )


def manifest_to_mapping(manifest: PackedRowManifest) -> dict[str, Any]:
    """Convert a :class:`PackedRowManifest` to a plain JSON-serialisable dict.

    Dense tensors are **never** included; the output is safe for use in Ray
    ``non_tensor_batch`` payloads.
    """
    return {
        "sample_index": manifest.sample_index,
        "response_uid": manifest.response_uid,
        "packed_round_index": manifest.packed_round_index,
        "pages": [page_ref_to_mapping(p) for p in manifest.pages],
        "active_label_tokens": manifest.active_label_tokens,
        "sequence_tokens": manifest.sequence_tokens,
    }


def manifest_from_mapping(d: Mapping[str, Any]) -> PackedRowManifest:
    """Reconstruct a :class:`PackedRowManifest` from a plain mapping."""
    return PackedRowManifest(
        sample_index=int(d["sample_index"]),
        response_uid=str(d["response_uid"]),
        packed_round_index=int(d["packed_round_index"]),
        pages=tuple(page_ref_from_mapping(p) for p in d["pages"]),
        active_label_tokens=int(d["active_label_tokens"]),
        sequence_tokens=int(d["sequence_tokens"]),
    )
