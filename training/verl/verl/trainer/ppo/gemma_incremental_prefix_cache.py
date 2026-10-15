"""Rank-local GPU-resident prefix KV cache for DiffusionGemma replay."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

DEFAULT_GEMMA_PREFIX_CACHE_MAX_ENTRIES = 8
DEFAULT_GEMMA_PREFIX_CACHE_MAX_BYTES = 8 * 1024**3

GemmaReplayIdentity = tuple[int, int]


def kv_cache_nbytes(kv_cache: Any) -> int:
    """Return the bytes owned by tensors reachable from a KV cache object."""
    total = 0
    seen: set[int] = set()

    def walk(value: Any) -> None:
        nonlocal total
        value_id = id(value)
        if value_id in seen:
            return
        seen.add(value_id)
        if torch.is_tensor(value):
            total += int(value.numel()) * int(value.element_size())
        elif isinstance(value, Mapping):
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)
        elif value is not None and hasattr(value, "__dict__"):
            for item in vars(value).values():
                walk(item)

    walk(kv_cache)
    return total


@dataclass(frozen=True)
class GemmaPrefixKVCacheEntry:
    prefix_ids: torch.Tensor
    kv_cache: Any
    nbytes: int
    segment_lengths: tuple[int, ...]


class GemmaIncrementalPrefixKVCache:
    """Bounded cache of detached prefix tokens and encoder KV tensors.

    Prefix tokens remain on their input device. Exact extension validation is
    performed on that device; on CUDA, converting the final one-element result
    to a Python bool causes one synchronization per validated cache hit.
    """

    def __init__(
        self,
        max_entries: int = DEFAULT_GEMMA_PREFIX_CACHE_MAX_ENTRIES,
        max_bytes: int = DEFAULT_GEMMA_PREFIX_CACHE_MAX_BYTES,
    ) -> None:
        self.max_entries = int(max_entries)
        self.max_bytes = int(max_bytes)
        if self.max_entries <= 0:
            raise ValueError("DiffusionGemma prefix cache max_entries must be positive")
        if self.max_bytes <= 0:
            raise ValueError("DiffusionGemma prefix cache max_bytes must be positive")

        self._entries: dict[tuple[GemmaReplayIdentity, int], GemmaPrefixKVCacheEntry] = {}
        self._entries_bytes = 0
        self._hits = 0
        self._misses = 0
        self._rebuilds = 0
        self._mismatches = 0
        self._order_mismatches = 0
        self._appended_tokens = 0
        self._avoided_prefix_tokens = 0
        self._build_seconds = 0.0
        self._extension_seconds = 0.0
        self._invalidations = 0
        self._validation_checks = 0
        self._validation_cuda_syncs = 0
        self._evictions = 0
        self._rejections = 0
        self._peak_entries = 0
        self._peak_bytes = 0

    @staticmethod
    def _identity(identity: GemmaReplayIdentity) -> GemmaReplayIdentity:
        if len(identity) != 2:
            raise ValueError("DiffusionGemma replay identity must contain two int64 lanes")
        return int(identity[0]), int(identity[1])

    @property
    def entries(self) -> int:
        return len(self._entries)

    @property
    def bytes(self) -> int:
        return self._entries_bytes

    def peek(
        self,
        identity: GemmaReplayIdentity,
        block_index: int,
    ) -> GemmaPrefixKVCacheEntry | None:
        """Return an entry without changing hit/miss metrics."""
        return self._entries.get((self._identity(identity), int(block_index)))

    def lookup_for_extension(
        self,
        identity: GemmaReplayIdentity,
        block_index: int,
        prefix_ids: torch.Tensor,
    ) -> GemmaPrefixKVCacheEntry | None:
        """Return block ``N-1`` only when ``prefix_ids`` extends it exactly."""
        identity = self._identity(identity)
        block_index = int(block_index)
        expected_key = (identity, block_index - 1)
        entry = self._entries.get(expected_key)
        if entry is None:
            self._misses += 1
            if any(key_identity == identity for key_identity, _ in self._entries):
                self._order_mismatches += 1
                self.invalidate_identity(identity)
            return None

        cached_length = int(entry.prefix_ids.numel())
        if (
            prefix_ids.device != entry.prefix_ids.device
            or prefix_ids.dtype != entry.prefix_ids.dtype
            or int(prefix_ids.numel()) <= cached_length
        ):
            self._mismatches += 1
            self.invalidate_identity(identity)
            return None

        self._validation_checks += 1
        matches = prefix_ids[:cached_length].eq(entry.prefix_ids).all()
        if prefix_ids.is_cuda:
            self._validation_cuda_syncs += 1
        if not bool(matches.item()):
            self._mismatches += 1
            self.invalidate_identity(identity)
            return None

        self._hits += 1
        return entry

    def evict(self, identity: GemmaReplayIdentity, block_index: int) -> bool:
        key = (self._identity(identity), int(block_index))
        entry = self._entries.pop(key, None)
        if entry is None:
            return False
        self._entries_bytes -= entry.nbytes
        self._evictions += 1
        del entry
        return True

    def invalidate_identity(self, identity: GemmaReplayIdentity) -> int:
        identity = self._identity(identity)
        keys = [key for key in self._entries if key[0] == identity]
        for key in keys:
            entry = self._entries.pop(key)
            self._entries_bytes -= entry.nbytes
            self._evictions += 1
            del entry
        return len(keys)

    def store(
        self,
        identity: GemmaReplayIdentity,
        block_index: int,
        prefix_ids: torch.Tensor,
        kv_cache: Any,
        segment_lengths: tuple[int, ...] | list[int] | None = None,
    ) -> bool:
        """Store a detached device-local token copy and the KV cache reference."""
        identity = self._identity(identity)
        key = (identity, int(block_index))
        self.evict(identity, block_index)

        stored_prefix_ids = prefix_ids.detach().contiguous().clone()
        normalized_segment_lengths = tuple(
            int(length)
            for length in (
                segment_lengths
                if segment_lengths is not None
                else (int(stored_prefix_ids.numel()),)
            )
        )
        if (
            not normalized_segment_lengths
            or any(length <= 0 for length in normalized_segment_lengths)
            or sum(normalized_segment_lengths) != int(stored_prefix_ids.numel())
        ):
            raise ValueError(
                "DiffusionGemma prefix cache segment lengths must be positive "
                "and sum to the stored prefix length: "
                f"segments={normalized_segment_lengths} "
                f"prefix_length={int(stored_prefix_ids.numel())}"
            )
        prefix_nbytes = int(stored_prefix_ids.numel()) * int(stored_prefix_ids.element_size())
        entry_nbytes = kv_cache_nbytes(kv_cache) + prefix_nbytes

        if entry_nbytes > self.max_bytes:
            self._rejections += 1
            del stored_prefix_ids
            return False

        while self._entries and (
            len(self._entries) >= self.max_entries
            or self._entries_bytes + entry_nbytes > self.max_bytes
        ):
            oldest_key = next(iter(self._entries))
            oldest = self._entries.pop(oldest_key)
            self._entries_bytes -= oldest.nbytes
            self._evictions += 1
            del oldest

        self._entries[key] = GemmaPrefixKVCacheEntry(
            prefix_ids=stored_prefix_ids,
            kv_cache=kv_cache,
            nbytes=entry_nbytes,
            segment_lengths=normalized_segment_lengths,
        )
        self._entries_bytes += entry_nbytes
        self._peak_entries = max(self._peak_entries, len(self._entries))
        self._peak_bytes = max(self._peak_bytes, self._entries_bytes)
        return True

    def invalidate(self) -> None:
        while self._entries:
            _, entry = self._entries.popitem()
            self._entries_bytes -= entry.nbytes
            del entry
        self._entries_bytes = 0
        self._invalidations += 1

    def record_build(self) -> None:
        self._rebuilds += 1

    def record_extension(
        self,
        *,
        appended_tokens: int,
        avoided_prefix_tokens: int,
    ) -> None:
        self._appended_tokens += int(appended_tokens)
        self._avoided_prefix_tokens += int(avoided_prefix_tokens)

    def record_timing(self, *, build_seconds: float, extension_seconds: float) -> None:
        self._build_seconds += float(build_seconds)
        self._extension_seconds += float(extension_seconds)

    def counters_snapshot(self) -> dict[str, float]:
        return {
            "hits": float(self._hits),
            "misses": float(self._misses),
            "rebuilds": float(self._rebuilds),
            "mismatches": float(self._mismatches),
            "order_mismatches": float(self._order_mismatches),
            "appended_tokens": float(self._appended_tokens),
            "avoided_prefix_tokens": float(self._avoided_prefix_tokens),
            "avoided_prefix_forwards": float(self._hits),
            "build_seconds": float(self._build_seconds),
            "extension_seconds": float(self._extension_seconds),
            "invalidations": float(self._invalidations),
            "validation_checks": float(self._validation_checks),
            "validation_cuda_syncs": float(self._validation_cuda_syncs),
            "evictions": float(self._evictions),
            "rejections": float(self._rejections),
            "peak_entries": float(self._peak_entries),
            "peak_bytes": float(self._peak_bytes),
        }

    def metrics_snapshot(self) -> dict[str, float]:
        prefix = "actor/diffusion_gemma/prefix_kv_cache"
        metrics = {
            f"{prefix}/{key}": value
            for key, value in self.counters_snapshot().items()
        }
        metrics.update(
            {
                f"{prefix}/enabled": 1.0,
                f"{prefix}/entries": float(self.entries),
                f"{prefix}/bytes": float(self.bytes),
            }
        )
        return metrics
