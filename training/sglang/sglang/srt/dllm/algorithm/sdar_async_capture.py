"""
Async capture pipeline for SDAR rollout trace / residual / replay tensors.

Feature flag: ``SGLANG_SDAR_ASYNC_CAPTURE=1``  (default **off**).

Architecture
------------
Critical path (each denoise step)
  1. Detach GPU tensor slices – no copy, just a refcount bump.
  2. Record a CUDA event on the current compute stream.
  3. Enqueue a :class:`StepHandle` with the tensor refs, event, and cheap
     Python scalars.  No ``.cpu()``, ``.item()``, ``torch.save``, or file
     writes occur here.

Consumer thread (one daemon per :class:`AsyncCaptureWorker` instance)
  1. Dequeue handles in FIFO order.
  2. ``cuda_event.synchronize()`` before any D2H access.
  3. Copy tensors to pinned CPU memory.
  4. Build ``trace_step`` dicts, replay tensor payloads, trace records.
  5. Write files / populate in-memory stores.

Backpressure policy
  Count and retained-storage byte budgets cover queued and active handles.
  Default policy raises a measured error immediately rather than dropping a
  replay row or waiting for background I/O.  ``block`` is opt-in and bounded
  by ``SGLANG_SDAR_ASYNC_ENQUEUE_TIMEOUT_S``.

Safety invariants
  - GPU tensors are kept alive by the handle's Python refcount until the
    consumer dereferences the handle.
  - A per-request generation counter rejects stale handles after request-pool
    reuse and prevents use-after-free/allocator-reuse races.
  - Consumer errors are retained and raised at the scheduler rollout boundary.
"""

from __future__ import annotations

import atexit
import dataclasses
import json
import logging
import math
import os
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

import torch

from sglang.srt.dllm.algorithm.adaptive_confidence_temperature import (
    summarize_adaptive_confidence_temperature_diagnostics,
)
from sglang.srt.dllm.algorithm.adaptive_temperature import (
    summarize_adaptive_temperature_diagnostics,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Env-var constants
# ---------------------------------------------------------------------------
_ASYNC_CAPTURE_DEFAULT_QUEUE_DEPTH = int(
    os.environ.get("SGLANG_SDAR_ASYNC_QUEUE_DEPTH", "64")
)
_ASYNC_CAPTURE_DEFAULT_QUEUE_BYTES = int(
    os.environ.get(
        "SGLANG_SDAR_ASYNC_QUEUE_MAX_BYTES", str(512 * 1024 * 1024)
    )
)

# Mirror of sdar_prefix_kv._MEMORY_TRACE_TENSOR_STORES – kept in sync here to
# avoid a circular / heavy import from the consumer thread.
_MEMORY_TRACE_TENSOR_STORES = {
    "tensor",
    "memory",
    "inline",
    "none",
    "off",
    "table",
    "memory_table",
    "handle",
    "gpu_table",
    "cuda_ipc",
}


def _annotate_step_payload_invalid(
    step_dict: dict,
    step_key_to_payload: Dict[int, dict],
    step_key: int,
) -> None:
    """Mark a replay row as not part of a valid adjacent entropy pair."""
    invalid_annotation = {
        "adjacent_entropy_pair_valid": False,
        "adjacent_entropy_prev_step_key": -1,
        "adjacent_entropy_pair_local_step": -1,
        "adjacent_entropy_prev_mean": 0.0,
        "adjacent_entropy_rollout_next_mean": 0.0,
        "adjacent_entropy_vocab_size": 0,
    }
    rp = step_dict.get("replay_payload")
    tp = step_key_to_payload.get(int(step_key))
    current_mask = step_dict.get("_aec_random_vocab_mask_cpu")
    current_global_mask = step_dict.get("_aec_all_valid_mask_cpu")
    if not isinstance(current_mask, torch.Tensor):
        for payload in (tp, rp):
            if not isinstance(payload, dict):
                continue
            current_mask = payload.get("random_vocab_mask")
            if current_mask is not None:
                break
    if not isinstance(current_global_mask, torch.Tensor):
        for payload in (tp, rp):
            if not isinstance(payload, dict):
                continue
            current_global_mask = payload.get("all_valid_mask")
            if current_global_mask is not None:
                break
    if current_mask is not None and not isinstance(current_mask, torch.Tensor):
        current_mask = torch.as_tensor(current_mask, dtype=torch.bool)
    if current_global_mask is not None and not isinstance(
        current_global_mask, torch.Tensor
    ):
        current_global_mask = torch.as_tensor(
            current_global_mask, dtype=torch.bool
        )
    if isinstance(current_mask, torch.Tensor):
        invalid_next_mask = torch.zeros_like(
            current_mask.detach().cpu(),
            dtype=torch.bool,
        ).reshape(-1)
    else:
        block_tokens = 0
        for payload in (tp, rp, step_dict):
            if isinstance(payload, dict):
                block_tokens = int(payload.get("block_tokens", 0) or 0)
                if block_tokens > 0:
                    break
        invalid_next_mask = torch.zeros(
            (block_tokens,),
            dtype=torch.bool,
        )
    invalid_global_annotation = {
        "global_aec_pair_valid": False,
        "global_aec_prev_step_key": -1,
        "global_aec_prev_mean": 0.0,
        "global_aec_rollout_next_mean": 0.0,
        "global_aec_vocab_size": 0,
    }
    invalid_global_next_mask = None
    if isinstance(current_global_mask, torch.Tensor):
        invalid_global_next_mask = torch.zeros_like(
            current_global_mask.detach().cpu(),
            dtype=torch.bool,
        ).reshape(-1)
    if isinstance(rp, dict):
        rp.update(invalid_annotation)
        rp["adjacent_entropy_next_mask"] = invalid_next_mask.tolist()
        if invalid_global_next_mask is not None:
            rp.update(invalid_global_annotation)
            rp["global_aec_next_mask"] = (
                invalid_global_next_mask.tolist()
            )
        meta_in_rp = rp.get("metadata")
        if isinstance(meta_in_rp, dict):
            meta_in_rp.update(invalid_annotation)
            meta_in_rp["adjacent_entropy_next_mask"] = (
                invalid_next_mask.tolist()
            )
            if invalid_global_next_mask is not None:
                meta_in_rp.update(invalid_global_annotation)
                meta_in_rp["global_aec_next_mask"] = (
                    invalid_global_next_mask.tolist()
                )
    if isinstance(tp, dict):
        tp.update(invalid_annotation)
        tp["adjacent_entropy_next_mask"] = invalid_next_mask
        if invalid_global_next_mask is not None:
            tp.update(invalid_global_annotation)
            tp["global_aec_next_mask"] = invalid_global_next_mask
        meta_in_tp = tp.get("metadata")
        if isinstance(meta_in_tp, dict):
            meta_in_tp.update(invalid_annotation)
            meta_in_tp["adjacent_entropy_next_mask"] = (
                invalid_next_mask.tolist()
            )
            if invalid_global_next_mask is not None:
                meta_in_tp.update(invalid_global_annotation)
                meta_in_tp["global_aec_next_mask"] = (
                    invalid_global_next_mask.tolist()
                )


def _clip_global_aec_masks_at_eos(
    step_dicts: List[dict],
    final_ids: Optional[torch.Tensor],
    start_offset: int,
    eos_token_ids: set[int],
) -> Optional[int]:
    """Exclude post-EOS block positions from every global-AEC state."""
    if (
        not isinstance(final_ids, torch.Tensor)
        or not eos_token_ids
        or final_ids.ndim != 1
    ):
        return None
    start = max(0, min(int(start_offset), int(final_ids.numel())))
    eos_mask = torch.zeros_like(final_ids, dtype=torch.bool)
    for token_id in eos_token_ids:
        eos_mask |= final_ids.eq(int(token_id))
    eos_positions = torch.nonzero(
        eos_mask
        & (torch.arange(final_ids.numel()) >= start),
        as_tuple=False,
    ).flatten()
    if not eos_positions.numel():
        return None
    response_end = int(eos_positions[0].item()) + 1
    for step_dict in step_dicts:
        mask = step_dict.get("_aec_all_valid_mask_cpu")
        if not isinstance(mask, torch.Tensor):
            continue
        clipped = mask.bool().clone()
        if response_end < clipped.numel():
            clipped[response_end:] = False
        step_dict["_aec_all_valid_mask_cpu"] = clipped
    return response_end


def _mix_replay_sample_key(value: int) -> int:
    value &= (1 << 64) - 1
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & ((1 << 64) - 1)
    value ^= value >> 31
    return value


def sample_replay_step(
    *,
    fraction: float,
    mode: str = "bernoulli",
    seed: int,
    req_pool_idx: int,
    generation: int,
    block_idx: int,
    local_step: int,
    is_final_denoise_row: bool,
) -> bool:
    """Deterministically Bernoulli-sample a row, always retaining final rows."""
    if is_final_denoise_row:
        return True
    if fraction <= 0.0:
        return False
    if fraction >= 1.0:
        return True
    block_key = (
        int(seed)
        ^ (int(req_pool_idx) * 0x9E3779B185EBCA87)
        ^ (int(generation) * 0xC2B2AE3D27D4EB4F)
        ^ (int(block_idx) * 0x165667B19E3779F9)
    ) & ((1 << 64) - 1)
    if str(mode).strip().lower() == "stratified":
        phase = _mix_replay_sample_key(block_key) / float(1 << 64)
        before = math.floor(int(local_step) * float(fraction) + phase)
        after = math.floor(
            (int(local_step) + 1) * float(fraction) + phase
        )
        return after > before
    value = _mix_replay_sample_key(
        block_key ^ (int(local_step) * 0x85EBCA77C2B2AE63)
    )
    return value < int(float(fraction) * (1 << 64))


class AsyncCaptureError(RuntimeError):
    """Base class for capture failures propagated to the rollout boundary."""


class AsyncCaptureBackpressureError(AsyncCaptureError):
    """The bounded queue could not accept a lossless capture handle."""


class AsyncCaptureConsumerError(AsyncCaptureError):
    """A background consumer failed while materializing a capture handle."""


class AsyncCaptureDrainTimeout(AsyncCaptureError):
    """Capture data was not complete before the rollout-boundary timeout."""


class AsyncCaptureMissingBlockError(AsyncCaptureError):
    """Step rows completed without their matching end-of-block handle."""


def _tensor_storage_bytes(tensor: Optional[torch.Tensor]) -> int:
    """Return retained storage capacity without touching tensor contents."""
    if not isinstance(tensor, torch.Tensor):
        return 0
    try:
        return int(tensor.untyped_storage().nbytes())
    except Exception:
        return int(tensor.numel() * tensor.element_size())


def estimate_handle_bytes(handle: Any) -> int:
    """Conservative upper bound for tensor storage retained by a handle.

    Shared storages are intentionally counted once per tensor reference.  This
    can overestimate retained memory, but it guarantees the configured byte
    budget is never exceeded by accounting.
    """
    total = 0
    for field in dataclasses.fields(handle):
        total += _tensor_storage_bytes(getattr(handle, field.name))
    return total


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
class AsyncCaptureMetrics:
    """Thread-safe counters and timers for the async capture pipeline."""

    def __init__(self):
        self._lock = threading.Lock()
        self.enqueue_count: int = 0
        self.process_count: int = 0
        self.drop_count: int = 0
        self.fallback_sync_count: int = 0
        self.backpressure_count: int = 0
        self.backpressure_timeout_count: int = 0
        self.enqueue_error_count: int = 0
        self.enqueue_us_total: float = 0.0
        self.enqueue_us_max: float = 0.0
        self.queue_depth_max: int = 0
        self.queued_bytes: int = 0
        self.queued_bytes_max: int = 0
        self.pending_depth: int = 0
        self.pending_depth_max: int = 0
        self.pending_bytes: int = 0
        self.pending_bytes_max: int = 0
        self.step_enqueue_count: int = 0
        self.step_enqueue_bytes: int = 0
        self.block_enqueue_count: int = 0
        self.block_enqueue_bytes: int = 0
        self.event_wait_us_total: float = 0.0
        self.pack_us_total: float = 0.0
        self.d2h_us_total: float = 0.0
        self.write_us_total: float = 0.0
        self.critical_path_blocked_us_total: float = 0.0
        self.bytes_written_total: int = 0
        self.replay_rows_written: int = 0
        self.consumer_errors: int = 0
        self.drain_count: int = 0
        self.drain_timeout_count: int = 0
        self.drain_wait_us_total: float = 0.0
        self.drain_wait_us_max: float = 0.0
        self.cancel_request_count: int = 0
        self.cancelled_handle_count: int = 0
        self.cancelled_bytes: int = 0
        self.missing_block_count: int = 0
        self.shutdown_count: int = 0
        self.shutdown_timeout_count: int = 0
        self.copy_stream_fallback_count: int = 0
        # Per-handle latency (enqueue → process)
        self.e2e_us_total: float = 0.0
        # Adjacent entropy contraction
        self.adjacent_entropy_valid_pairs: int = 0
        self.adjacent_entropy_gapped_rows: int = 0
        self.adjacent_entropy_empty_mask_rows: int = 0
        self.adjacent_entropy_unpaired_rows: int = 0

    def record_enqueue(
        self,
        *,
        kind: str,
        retained_bytes: int,
        enqueue_us: float,
        queued_depth: int,
        queued_bytes: int,
        pending_depth: int,
        pending_bytes: int,
    ) -> None:
        with self._lock:
            self.enqueue_count += 1
            self.enqueue_us_total += enqueue_us
            self.enqueue_us_max = max(self.enqueue_us_max, enqueue_us)
            self.queue_depth_max = max(self.queue_depth_max, queued_depth)
            self.queued_bytes = queued_bytes
            self.queued_bytes_max = max(self.queued_bytes_max, queued_bytes)
            self.pending_depth = pending_depth
            self.pending_depth_max = max(self.pending_depth_max, pending_depth)
            self.pending_bytes = pending_bytes
            self.pending_bytes_max = max(self.pending_bytes_max, pending_bytes)
            if kind == _HANDLE_TYPE_STEP:
                self.step_enqueue_count += 1
                self.step_enqueue_bytes += retained_bytes
            elif kind == _HANDLE_TYPE_BLOCK:
                self.block_enqueue_count += 1
                self.block_enqueue_bytes += retained_bytes

    def record_dequeue(self, queued_bytes: int) -> None:
        with self._lock:
            self.queued_bytes = queued_bytes

    def record_release(self, pending_depth: int, pending_bytes: int) -> None:
        with self._lock:
            self.pending_depth = pending_depth
            self.pending_bytes = pending_bytes

    def record_process(self, event_wait_us: float, pack_us: float, d2h_us: float,
                       write_us: float, bytes_written: int, e2e_us: float) -> None:
        with self._lock:
            self.process_count += 1
            self.event_wait_us_total += event_wait_us
            self.pack_us_total += pack_us
            self.d2h_us_total += d2h_us
            self.write_us_total += write_us
            self.bytes_written_total += bytes_written
            self.e2e_us_total += e2e_us

    def record_replay_row(self) -> None:
        with self._lock:
            self.replay_rows_written += 1

    def record_drop(self) -> None:
        with self._lock:
            self.drop_count += 1

    def record_backpressure(self, us: float, *, timed_out: bool) -> None:
        with self._lock:
            self.backpressure_count += 1
            self.critical_path_blocked_us_total += us
            if timed_out:
                self.backpressure_timeout_count += 1
                self.enqueue_error_count += 1

    def record_consumer_error(self) -> None:
        with self._lock:
            self.consumer_errors += 1

    def record_drain(self, wait_us: float, *, timed_out: bool) -> None:
        with self._lock:
            self.drain_count += 1
            self.drain_wait_us_total += wait_us
            self.drain_wait_us_max = max(self.drain_wait_us_max, wait_us)
            if timed_out:
                self.drain_timeout_count += 1

    def record_cancel(self, handles: int, retained_bytes: int, missing_blocks: int) -> None:
        with self._lock:
            self.cancel_request_count += 1
            self.cancelled_handle_count += handles
            self.cancelled_bytes += retained_bytes
            self.missing_block_count += missing_blocks

    def record_shutdown(self, *, timed_out: bool) -> None:
        with self._lock:
            self.shutdown_count += 1
            if timed_out:
                self.shutdown_timeout_count += 1

    def record_copy_stream_fallback(self) -> None:
        with self._lock:
            self.copy_stream_fallback_count += 1

    def record_adjacent_entropy_pairs(
        self,
        *,
        valid_pairs: int,
        gapped_rows: int,
        empty_mask_rows: int,
        unpaired_rows: int,
    ) -> None:
        with self._lock:
            self.adjacent_entropy_valid_pairs += valid_pairs
            self.adjacent_entropy_gapped_rows += gapped_rows
            self.adjacent_entropy_empty_mask_rows += empty_mask_rows
            self.adjacent_entropy_unpaired_rows += unpaired_rows

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            n = max(self.process_count, 1)
            enqueued = max(self.enqueue_count, 1)
            steps = max(self.step_enqueue_count, 1)
            blocks = max(self.block_enqueue_count, 1)
            return {
                "enqueue_count": self.enqueue_count,
                "process_count": self.process_count,
                "drop_count": self.drop_count,
                "fallback_sync_count": self.fallback_sync_count,
                "backpressure_count": self.backpressure_count,
                "backpressure_timeout_count": self.backpressure_timeout_count,
                "enqueue_error_count": self.enqueue_error_count,
                "enqueue_us_avg": self.enqueue_us_total / enqueued,
                "enqueue_us_max": self.enqueue_us_max,
                "queue_depth_max": self.queue_depth_max,
                "queued_bytes": self.queued_bytes,
                "queued_bytes_max": self.queued_bytes_max,
                "pending_depth": self.pending_depth,
                "pending_depth_max": self.pending_depth_max,
                "pending_bytes": self.pending_bytes,
                "pending_bytes_max": self.pending_bytes_max,
                "retained_bytes_per_handle_avg": (
                    self.step_enqueue_bytes + self.block_enqueue_bytes
                )
                / enqueued,
                "bytes_per_step_avg": self.step_enqueue_bytes / steps,
                "bytes_per_block_avg": self.block_enqueue_bytes / blocks,
                "event_wait_us_avg": self.event_wait_us_total / n,
                "pack_us_avg": self.pack_us_total / n,
                "d2h_us_avg": self.d2h_us_total / n,
                "write_us_avg": self.write_us_total / n,
                "e2e_us_avg": self.e2e_us_total / n,
                "critical_path_blocked_us_total": self.critical_path_blocked_us_total,
                "bytes_written_total": self.bytes_written_total,
                "replay_rows_written": self.replay_rows_written,
                "consumer_errors": self.consumer_errors,
                "drain_count": self.drain_count,
                "drain_timeout_count": self.drain_timeout_count,
                "drain_wait_us_avg": self.drain_wait_us_total
                / max(self.drain_count, 1),
                "drain_wait_us_max": self.drain_wait_us_max,
                "cancel_request_count": self.cancel_request_count,
                "cancelled_handle_count": self.cancelled_handle_count,
                "cancelled_bytes": self.cancelled_bytes,
                "missing_block_count": self.missing_block_count,
                "shutdown_count": self.shutdown_count,
                "shutdown_timeout_count": self.shutdown_timeout_count,
                "copy_stream_fallback_count": self.copy_stream_fallback_count,
                "adjacent_entropy_valid_pairs": self.adjacent_entropy_valid_pairs,
                "adjacent_entropy_gapped_rows": self.adjacent_entropy_gapped_rows,
                "adjacent_entropy_empty_mask_rows": self.adjacent_entropy_empty_mask_rows,
                "adjacent_entropy_unpaired_rows": self.adjacent_entropy_unpaired_rows,
            }


# ---------------------------------------------------------------------------
# Handle types
# ---------------------------------------------------------------------------
_HANDLE_TYPE_STEP = "step"
_HANDLE_TYPE_BLOCK = "block"
_HANDLE_TYPE_DRAIN = "drain"


@dataclasses.dataclass
class StepHandle:
    """
    Lightweight per-step capture handle enqueued from the generation critical
    path.  All tensor fields must be already detached (no grad tracking).
    Mutable masks must be cloned so the decode loop can continue modifying
    the originals.
    """
    # GPU tensor snapshots (ownership kept alive by Python refcount)
    input_ids_snapshot: torch.Tensor    # clone of block_input_ids before this step
    current_ids_snapshot: torch.Tensor  # clone after this step's accepted edits
    proposal: torch.Tensor              # detach only – immutable after accept
    proposal_probs: torch.Tensor        # detach only
    raw_logprobs: torch.Tensor          # detach only
    sample_logprobs: torch.Tensor       # detach only
    entropy: torch.Tensor               # detach only
    accepted_mask: torch.Tensor         # clone (loop mutates original)
    new_accept_mask: torch.Tensor       # clone
    edit_mask: torch.Tensor             # clone
    action_mask: torch.Tensor           # clone
    prompt_mask: torch.Tensor           # stable (not mutated); detach
    # Optional replay tensors
    sc_signal: Optional[torch.Tensor]   # self-conditioning signal slice
    logits_source_hidden: Optional[torch.Tensor]
    valid_mask: Optional[torch.Tensor]
    # CUDA event: consumer must synchronize before D2H
    cuda_event: Optional[torch.cuda.Event]
    # Cheap Python scalars
    step: int
    block_idx: int
    req_pool_idx: int
    block_start: int
    block_end: int
    was_completed: bool
    response_start_index: int
    start_offset: int
    block_size: int
    sc_stats: Any
    trace_full: bool
    trace_tensors: bool
    capture_replay: bool
    trace_tensor_records: int
    trace_tensor_store: str
    trace_tensor_max_tokens: int
    position_start: int   # block_idx * block_size
    step_key: int
    is_final_denoise_row: bool
    enqueue_ns: int       # time.monotonic_ns() at enqueue
    # Generation counter: consumer validates tensor storage is unchanged
    generation: int = 0
    sc_signal_is_compact: bool = False


@dataclasses.dataclass
class BlockHandle:
    """
    End-of-block capture handle for introspection record and trace record
    assembly.  Enqueued once per block after the commit forward completes.
    """
    # GPU tensors for introspection
    final_ids: torch.Tensor
    commit_argmax: torch.Tensor
    commit_argmax_probs: torch.Tensor
    p_probs: torch.Tensor
    q_probs: torch.Tensor
    q_raw_logprobs: torch.Tensor
    q_sample_logprobs: torch.Tensor
    prompt_mask: torch.Tensor
    cuda_event: Optional[torch.cuda.Event]
    # Scalars
    req_pool_idx: int
    block_idx: int
    block_start: int
    block_end: int
    start_offset: int
    denoise_forwards: int
    batch_denoise_forwards: int
    response_clean_start: int
    trace_dir: str
    enqueue_ns: int
    adaptive_temperature_diagnostics: list[dict] = dataclasses.field(
        default_factory=list
    )
    adaptive_confidence_temperature_diagnostics: list[
        dict
    ] = dataclasses.field(default_factory=list)
    generation: int = 0


@dataclasses.dataclass
class DrainHandle:
    """Sentinel: consumer sets an Event when it is done with this marker."""
    event: threading.Event
    req_pool_idx: int  # -1 means drain all


@dataclasses.dataclass
class _QueueEntry:
    kind: str
    handle: Any
    retained_bytes: int
    key: tuple[int, int]


# ---------------------------------------------------------------------------
# Async capture worker
# ---------------------------------------------------------------------------
class AsyncCaptureWorker:
    """
    Background daemon thread that drains the capture queue and performs all
    D2H copies, tensor serialization, and file writes.

    Parameters
    ----------
    algorithm_ref:
        The :class:`SDARPrefixKV` instance.  The worker holds a reference to
        it for access to ``_request_replay_tensors_by_req``,
        ``_request_stats_by_req``, ``_write_trace_record``,
        ``_write_introspection_record``, ``_write_replay_tensor_file``, and
        ``finalize_async_replay_tensors``.
    Both queue count and retained tensor storage bytes are bounded.  The
    limits include the item currently being consumed, so references cannot
    exceed the configured budget while a D2H copy is in flight.
    """

    def __init__(
        self,
        algorithm_ref,
        queue_depth: int = _ASYNC_CAPTURE_DEFAULT_QUEUE_DEPTH,
        queue_max_bytes: int = _ASYNC_CAPTURE_DEFAULT_QUEUE_BYTES,
        backpressure_policy: str = "error",
        enqueue_timeout_s: float = 0.0,
        *,
        start_consumer: bool = True,
    ):
        if queue_depth <= 0:
            raise ValueError("queue_depth must be positive")
        if queue_max_bytes <= 0:
            raise ValueError("queue_max_bytes must be positive")
        if backpressure_policy not in {"error", "block"}:
            raise ValueError("backpressure_policy must be 'error' or 'block'")
        self._algo = algorithm_ref
        self._max_count = int(queue_depth)
        self._max_bytes = int(queue_max_bytes)
        self._backpressure_policy = backpressure_policy
        self._enqueue_timeout_s = max(float(enqueue_timeout_s), 0.0)
        self._entries: Deque[_QueueEntry] = deque()
        self._condition = threading.Condition()
        self._queued_bytes = 0
        self._pending_count = 0
        self._pending_bytes = 0
        self._pending_by_key: Dict[tuple[int, int], int] = {}
        self._pending_bytes_by_key: Dict[tuple[int, int], int] = {}
        # Store only text, never exception tracebacks that can retain handle
        # frames and therefore GPU tensors after budget release.
        self._errors_by_key: Dict[tuple[int, int], str] = {}
        self._cancelled_keys: set[tuple[int, int]] = set()
        self._request_lifecycle: Dict[tuple[int, int], Dict[str, Any]] = {}
        self._accepting = True
        self._stopping = False
        self._metrics = AsyncCaptureMetrics()
        self._req_step_handles: Dict[tuple[int, int, int], List[dict]] = {}
        self._copy_streams: Dict[int, torch.cuda.Stream] = {}
        self._thread: Optional[threading.Thread] = None
        if start_consumer:
            self._thread = threading.Thread(
                target=self._consumer_loop,
                name="sdar-async-capture",
                daemon=True,
            )
            self._thread.start()
        atexit.register(self.stop, drain=False, join_timeout=2.0)
        logger.info(
            "[SDAR_ASYNC_CAPTURE] Worker started "
            "(max_count=%d max_bytes=%d policy=%s timeout_s=%.3f)",
            self._max_count,
            self._max_bytes,
            self._backpressure_policy,
            self._enqueue_timeout_s,
        )

    @staticmethod
    def _handle_key(handle: Any) -> tuple[int, int]:
        return int(handle.req_pool_idx), int(handle.generation)

    def _lifecycle_for(self, key: tuple[int, int]) -> Dict[str, Any]:
        return self._request_lifecycle.setdefault(
            key,
            {
                "enqueue_count": 0,
                "enqueue_bytes": 0,
                "processed_count": 0,
                "consumer_errors": 0,
                "backpressure_count": 0,
                "drain_wait_us": 0.0,
                "cancelled": False,
            },
        )

    def _enqueue(self, kind: str, handle: Any) -> None:
        start_ns = time.monotonic_ns()
        retained_bytes = estimate_handle_bytes(handle)
        key = self._handle_key(handle)
        if retained_bytes > self._max_bytes:
            elapsed_us = (time.monotonic_ns() - start_ns) / 1_000
            self._metrics.record_backpressure(elapsed_us, timed_out=True)
            raise AsyncCaptureBackpressureError(
                f"{kind} handle retains {retained_bytes} bytes, exceeding "
                f"SGLANG_SDAR_ASYNC_QUEUE_MAX_BYTES={self._max_bytes}"
            )

        deadline = (
            time.monotonic() + self._enqueue_timeout_s
            if self._backpressure_policy == "block"
            else None
        )
        backpressured = False
        with self._condition:
            while True:
                if not self._accepting:
                    self._metrics.record_backpressure(0.0, timed_out=True)
                    raise AsyncCaptureBackpressureError("async capture worker is stopping")
                if self._thread is not None and not self._thread.is_alive():
                    self._metrics.record_backpressure(0.0, timed_out=True)
                    raise AsyncCaptureBackpressureError(
                        "async capture consumer is not alive"
                    )
                if key in self._cancelled_keys:
                    self._metrics.record_backpressure(0.0, timed_out=True)
                    raise AsyncCaptureBackpressureError(
                        f"request {key[0]} generation {key[1]} was cancelled"
                    )
                has_capacity = (
                    self._pending_count < self._max_count
                    and self._pending_bytes + retained_bytes <= self._max_bytes
                )
                if has_capacity:
                    break
                backpressured = True
                lifecycle = self._lifecycle_for(key)
                lifecycle["backpressure_count"] += 1
                if self._backpressure_policy == "error":
                    elapsed_us = (time.monotonic_ns() - start_ns) / 1_000
                    self._metrics.record_backpressure(elapsed_us, timed_out=True)
                    raise AsyncCaptureBackpressureError(
                        "async capture queue budget exhausted: "
                        f"pending={self._pending_count}/{self._max_count} "
                        f"bytes={self._pending_bytes}/{self._max_bytes} "
                        f"incoming_bytes={retained_bytes}"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    elapsed_us = (time.monotonic_ns() - start_ns) / 1_000
                    self._metrics.record_backpressure(elapsed_us, timed_out=True)
                    raise AsyncCaptureBackpressureError(
                        "timed out waiting for async capture queue capacity: "
                        f"pending={self._pending_count}/{self._max_count} "
                        f"bytes={self._pending_bytes}/{self._max_bytes}"
                    )
                self._condition.wait(timeout=remaining)

            entry = _QueueEntry(kind, handle, retained_bytes, key)
            self._entries.append(entry)
            self._queued_bytes += retained_bytes
            self._pending_count += 1
            self._pending_bytes += retained_bytes
            self._pending_by_key[key] = self._pending_by_key.get(key, 0) + 1
            self._pending_bytes_by_key[key] = (
                self._pending_bytes_by_key.get(key, 0) + retained_bytes
            )
            lifecycle = self._lifecycle_for(key)
            lifecycle["enqueue_count"] += 1
            lifecycle["enqueue_bytes"] += retained_bytes
            enqueue_us = (time.monotonic_ns() - start_ns) / 1_000
            self._metrics.record_enqueue(
                kind=kind,
                retained_bytes=retained_bytes,
                enqueue_us=enqueue_us,
                queued_depth=len(self._entries),
                queued_bytes=self._queued_bytes,
                pending_depth=self._pending_count,
                pending_bytes=self._pending_bytes,
            )
            if backpressured:
                self._metrics.record_backpressure(enqueue_us, timed_out=False)
            self._condition.notify()

    def enqueue_step(
        self, handle: StepHandle, *, is_replay_critical: bool = True
    ) -> None:
        del is_replay_critical
        self._enqueue(_HANDLE_TYPE_STEP, handle)

    def enqueue_block(self, handle: BlockHandle) -> None:
        self._enqueue(_HANDLE_TYPE_BLOCK, handle)

    def _generation_for(self, req_pool_idx: int, generation: Optional[int]) -> int:
        if generation is not None:
            return int(generation)
        getter = getattr(self._algo, "_capture_generation_for_request", None)
        return int(getter(req_pool_idx) if getter is not None else 0)

    def drain_request(
        self,
        req_pool_idx: int,
        *,
        generation: Optional[int] = None,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        key = (int(req_pool_idx), self._generation_for(req_pool_idx, generation))
        start_ns = time.monotonic_ns()
        deadline = time.monotonic() + max(float(timeout), 0.0)
        with self._condition:
            while self._pending_by_key.get(key, 0) > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    wait_us = (time.monotonic_ns() - start_ns) / 1_000
                    self._metrics.record_drain(wait_us, timed_out=True)
                    raise AsyncCaptureDrainTimeout(
                        f"request {key[0]} generation {key[1]} still has "
                        f"{self._pending_by_key.get(key, 0)} handles / "
                        f"{self._pending_bytes_by_key.get(key, 0)} bytes pending"
                    )
                self._condition.wait(timeout=remaining)

            wait_us = (time.monotonic_ns() - start_ns) / 1_000
            self._metrics.record_drain(wait_us, timed_out=False)
            lifecycle = dict(self._lifecycle_for(key))
            lifecycle["drain_wait_us"] = wait_us
            lifecycle["pending_count"] = 0
            lifecycle["pending_bytes"] = 0
            error = self._errors_by_key.pop(key, None)
            missing_keys = [k for k in self._req_step_handles if k[:2] == key]
            if missing_keys:
                for missing_key in missing_keys:
                    self._req_step_handles.pop(missing_key, None)
                self._metrics.record_cancel(0, 0, len(missing_keys))
                raise AsyncCaptureMissingBlockError(
                    f"request {key[0]} generation {key[1]} completed without "
                    f"{len(missing_keys)} BlockHandle(s)"
                )
            if error is not None:
                raise AsyncCaptureConsumerError(
                    f"async capture failed for request {key[0]} "
                    f"generation {key[1]}: {error}"
                )
            return lifecycle

    def drain_all(self, timeout: float = 60.0) -> Dict[str, Any]:
        start_ns = time.monotonic_ns()
        deadline = time.monotonic() + max(float(timeout), 0.0)
        with self._condition:
            while self._pending_count > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    wait_us = (time.monotonic_ns() - start_ns) / 1_000
                    self._metrics.record_drain(wait_us, timed_out=True)
                    raise AsyncCaptureDrainTimeout(
                        f"{self._pending_count} handles / "
                        f"{self._pending_bytes} bytes still pending"
                    )
                self._condition.wait(timeout=remaining)
            wait_us = (time.monotonic_ns() - start_ns) / 1_000
            self._metrics.record_drain(wait_us, timed_out=False)
            if self._errors_by_key:
                key, error = next(iter(self._errors_by_key.items()))
                raise AsyncCaptureConsumerError(
                    f"async capture failed for request {key[0]} "
                    f"generation {key[1]}: {error}"
                )
            if self._req_step_handles:
                count = len(self._req_step_handles)
                self._req_step_handles.clear()
                self._metrics.record_cancel(0, 0, count)
                raise AsyncCaptureMissingBlockError(
                    f"{count} step group(s) completed without BlockHandle"
                )
            return self._metrics.snapshot()

    def get_metrics(self) -> Dict[str, Any]:
        with self._condition:
            current = {
                "queued_depth": len(self._entries),
                "queued_bytes": self._queued_bytes,
                "pending_depth": self._pending_count,
                "pending_bytes": self._pending_bytes,
            }
        snapshot = self._metrics.snapshot()
        snapshot.update(current)
        snapshot["queue_max_count"] = self._max_count
        snapshot["queue_max_bytes"] = self._max_bytes
        snapshot["backpressure_policy"] = self._backpressure_policy
        return snapshot

    def request_metrics(
        self, req_pool_idx: int, generation: Optional[int] = None
    ) -> Dict[str, Any]:
        key = (int(req_pool_idx), self._generation_for(req_pool_idx, generation))
        with self._condition:
            result = dict(self._lifecycle_for(key))
            result["pending_count"] = self._pending_by_key.get(key, 0)
            result["pending_bytes"] = self._pending_bytes_by_key.get(key, 0)
            return result

    def forget_request(
        self, req_pool_idx: int, generation: Optional[int] = None
    ) -> None:
        key = (int(req_pool_idx), self._generation_for(req_pool_idx, generation))
        with self._condition:
            if self._pending_by_key.get(key, 0) != 0:
                return
            self._errors_by_key.pop(key, None)
            self._request_lifecycle.pop(key, None)
            self._cancelled_keys.discard(key)

    def _release_entry_locked(self, entry: _QueueEntry) -> None:
        key = entry.key
        self._pending_count -= 1
        self._pending_bytes -= entry.retained_bytes
        remaining = self._pending_by_key.get(key, 0) - 1
        if remaining <= 0:
            self._pending_by_key.pop(key, None)
            self._pending_bytes_by_key.pop(key, None)
            if key in self._cancelled_keys and key not in self._errors_by_key:
                self._request_lifecycle.pop(key, None)
                self._cancelled_keys.discard(key)
        else:
            self._pending_by_key[key] = remaining
            self._pending_bytes_by_key[key] = max(
                0,
                self._pending_bytes_by_key.get(key, 0) - entry.retained_bytes,
            )
        self._metrics.record_release(self._pending_count, self._pending_bytes)
        self._condition.notify_all()

    def _purge_key_locked(self, key: tuple[int, int]) -> tuple[int, int]:
        kept: Deque[_QueueEntry] = deque()
        purged_count = 0
        purged_bytes = 0
        while self._entries:
            entry = self._entries.popleft()
            if entry.key == key:
                purged_count += 1
                purged_bytes += entry.retained_bytes
                self._queued_bytes -= entry.retained_bytes
                self._release_entry_locked(entry)
            else:
                kept.append(entry)
        self._entries = kept
        self._metrics.record_dequeue(self._queued_bytes)
        return purged_count, purged_bytes

    def cancel_request(
        self,
        req_pool_idx: int,
        *,
        generation: Optional[int] = None,
        wait: bool = False,
        timeout: float = 5.0,
    ) -> bool:
        key = (int(req_pool_idx), self._generation_for(req_pool_idx, generation))
        with self._condition:
            self._cancelled_keys.add(key)
            lifecycle = self._lifecycle_for(key)
            lifecycle["cancelled"] = True
            purged_count, purged_bytes = self._purge_key_locked(key)
            missing_keys = [k for k in self._req_step_handles if k[:2] == key]
            for missing_key in missing_keys:
                self._req_step_handles.pop(missing_key, None)
            self._metrics.record_cancel(
                purged_count, purged_bytes, len(missing_keys)
            )
            if not wait:
                complete = self._pending_by_key.get(key, 0) == 0
                if complete:
                    self._errors_by_key.pop(key, None)
                    self._request_lifecycle.pop(key, None)
                    self._cancelled_keys.discard(key)
                return complete
            deadline = time.monotonic() + max(float(timeout), 0.0)
            while self._pending_by_key.get(key, 0) > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(timeout=remaining)
            self._errors_by_key.pop(key, None)
            self._request_lifecycle.pop(key, None)
            self._cancelled_keys.discard(key)
            return True

    def stop(self, *, drain: bool = True, join_timeout: float = 5.0) -> bool:
        try:
            atexit.unregister(self.stop)
        except Exception:
            pass
        with self._condition:
            if self._stopping:
                return self._thread is None or not self._thread.is_alive()
            self._accepting = False
        timed_out = False
        if drain:
            try:
                self.drain_all(timeout=join_timeout)
            except AsyncCaptureError:
                timed_out = True
        with self._condition:
            if self._entries:
                keys = {entry.key for entry in self._entries}
                for key in keys:
                    self._cancelled_keys.add(key)
                    self._purge_key_locked(key)
            self._stopping = True
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=max(float(join_timeout), 0.0))
            timed_out = timed_out or self._thread.is_alive()
        self._req_step_handles.clear()
        if self._thread is None or not self._thread.is_alive():
            self._copy_streams.clear()
            self._algo = None
        self._metrics.record_shutdown(timed_out=timed_out)
        return not timed_out

    # ------------------------------------------------------------------
    # Consumer loop
    # ------------------------------------------------------------------
    def _consumer_loop(self) -> None:
        logger.info("[SDAR_ASYNC_CAPTURE] Consumer thread started")
        while True:
            with self._condition:
                while not self._entries and not self._stopping:
                    self._condition.wait(timeout=0.5)
                if not self._entries:
                    if self._stopping:
                        break
                    continue
                entry = self._entries.popleft()
                self._queued_bytes -= entry.retained_bytes
                self._metrics.record_dequeue(self._queued_bytes)

            handle = entry.handle
            try:
                if entry.key in self._cancelled_keys or not self._is_current(handle):
                    continue
                if entry.kind == _HANDLE_TYPE_STEP:
                    self._process_step(handle)
                elif entry.kind == _HANDLE_TYPE_BLOCK:
                    self._process_block(handle)
                with self._condition:
                    self._lifecycle_for(entry.key)["processed_count"] += 1
            except Exception as exc:
                self._metrics.record_consumer_error()
                error_text = f"{type(exc).__name__}: {exc}"
                with self._condition:
                    self._errors_by_key.setdefault(entry.key, error_text)
                    self._lifecycle_for(entry.key)["consumer_errors"] += 1
                    self._cancelled_keys.add(entry.key)
                    purged_count, purged_bytes = self._purge_key_locked(entry.key)
                    missing_keys = [
                        key for key in self._req_step_handles if key[:2] == entry.key
                    ]
                    for missing_key in missing_keys:
                        self._req_step_handles.pop(missing_key, None)
                    if purged_count or missing_keys:
                        self._metrics.record_cancel(
                            purged_count, purged_bytes, len(missing_keys)
                        )
                logger.error(
                    "[SDAR_ASYNC_CAPTURE] Consumer error processing %s: %s",
                    entry.kind,
                    error_text,
                )
            finally:
                entry.handle = None
                handle = None
                with self._condition:
                    self._release_entry_locked(entry)

        logger.info("[SDAR_ASYNC_CAPTURE] Consumer thread exiting")

    def _is_current(self, handle: Any) -> bool:
        checker = getattr(self._algo, "_is_capture_generation_current", None)
        if checker is None:
            return True
        return bool(checker(handle.req_pool_idx, handle.generation))

    @property
    def _adjacent_entropy_contraction_enabled(self) -> bool:
        return bool(getattr(self._algo, "adjacent_entropy_contraction", False))

    def _get_adjacent_entropy_vocab_size(self) -> Optional[int]:
        """Return the true vocab size for entropy normalization, or None if unavailable.

        Uses algo.aec_vocab_size which is populated from model_config.hf_config.vocab_size
        via DllmConfig.  Never falls back to random_token_high (which excludes special IDs
        and is not the true vocabulary size).
        """
        val = getattr(self._algo, "aec_vocab_size", None)
        if val is not None and int(val) >= 2:
            return int(val)
        return None

    def _annotate_adjacent_entropy_pairs(
        self,
        step_dicts: List[dict],
        req_id: int,
        block_idx: int,
    ) -> None:
        """Compute and attach adjacent entropy pair annotations to replay payloads.

        Operates on all accumulated step_dicts for a completed block:
        - True adjacency: local_step_next == local_step + 1, never across missing steps.
        - M_t and M_next are independent masks; they must NOT be intersected.
        - Normalizes by log(vocab_size). If vocab_size is unavailable or either
          mask is empty, the pair is marked invalid (fail-closed).
        - Adds fields to:
          (a) the tensor_payload in algo._request_replay_tensors_by_req (memory mode)
          (b) the meta_payload in step_dicts[i]["replay_payload"] (all modes)
        """
        if not step_dicts:
            return
        vocab_size = self._get_adjacent_entropy_vocab_size()
        if vocab_size is None:
            logger.warning(
                "[SDAR_AEC] adjacent_entropy_contraction enabled but "
                "aec_vocab_size not available (hf_config.vocab_size not set); "
                "all pairs will be invalid"
            )
        log_vocab = math.log(float(vocab_size)) if vocab_size is not None else float("nan")

        # Sort by local_step for correct adjacency detection.
        sorted_steps = sorted(step_dicts, key=lambda d: int(d.get("step", 0)))

        # Build lookup from step_key -> tensor_payload in algo store
        step_key_to_payload: Dict[int, dict] = {}
        with self._algo._replay_tensors_lock:
            all_rows = self._algo._request_replay_tensors_by_req.get(req_id, [])
            for tp in all_rows:
                if not isinstance(tp, dict):
                    continue
                meta = tp.get("metadata") if isinstance(tp.get("metadata"), dict) else tp
                sk = meta.get("step_key", tp.get("step_key"))
                if sk is not None and int(meta.get("block_idx", tp.get("block_idx", -1))) == block_idx:
                    step_key_to_payload[int(sk)] = tp

        valid_pairs = 0
        gapped_rows = 0
        empty_mask_rows = 0
        unpaired_rows = 0

        # For each step t+1, look for a valid adjacent predecessor t
        prev: Optional[dict] = None
        for cur in sorted_steps:
            cur_step = int(cur.get("step", 0))
            cur_step_key = None
            rp = cur.get("replay_payload")
            if isinstance(rp, dict):
                cur_step_key = rp.get("step_key")

            if prev is None:
                # No predecessor for first step — mark unpaired
                if cur_step_key is not None:
                    _annotate_step_payload_invalid(
                        cur, step_key_to_payload, cur_step_key
                    )
                unpaired_rows += 1
                prev = cur
                continue

            prev_step = int(prev.get("step", -999))
            prev_step_key = None
            prev_rp = prev.get("replay_payload")
            if isinstance(prev_rp, dict):
                prev_step_key = prev_rp.get("step_key")

            # Check strict adjacency
            if cur_step != prev_step + 1:
                if cur_step > prev_step + 1:
                    gapped_rows += 1
                if cur_step_key is not None:
                    _annotate_step_payload_invalid(
                        cur, step_key_to_payload, cur_step_key
                    )
                prev = cur
                continue

            # Get entropy and mask tensors for prev (t) and cur (t+1)
            entropy_t: Optional[torch.Tensor] = prev.get("_aec_entropy_cpu")
            mask_t: Optional[torch.Tensor] = prev.get("_aec_random_vocab_mask_cpu")
            entropy_next: Optional[torch.Tensor] = cur.get("_aec_entropy_cpu")
            mask_next: Optional[torch.Tensor] = cur.get("_aec_random_vocab_mask_cpu")

            # Entropy and vocabulary normalization are shared prerequisites
            # for both scopes. The masks are not: global/all-valid AEC must
            # remain valid when the unresolved mask is empty.
            if (
                entropy_t is None
                or entropy_next is None
                or vocab_size is None
                or not math.isfinite(log_vocab)
                or log_vocab <= 0.0
            ):
                if cur_step_key is not None:
                    _annotate_step_payload_invalid(
                        cur, step_key_to_payload, cur_step_key
                    )
                empty_mask_rows += 1
                prev = cur
                continue

            gall_mask_t: Optional[torch.Tensor] = prev.get(
                "_aec_all_valid_mask_cpu"
            )
            gall_mask_next: Optional[torch.Tensor] = cur.get(
                "_aec_all_valid_mask_cpu"
            )
            block_tokens = 0
            if isinstance(rp, dict):
                block_tokens = int(
                    rp.get("stored_tokens")
                    or rp.get("block_tokens")
                    or 0
                )
            if block_tokens <= 0:
                for candidate in (mask_next, gall_mask_next, entropy_next):
                    if isinstance(candidate, torch.Tensor):
                        block_tokens = int(candidate.shape[0])
                        break

            # Unresolved/random-vocabulary annotation. Invalid rows retain a
            # shape-stable all-false mask so compact transport can stack every
            # row without dropping the field.
            annotation = {
                "adjacent_entropy_pair_valid": False,
                "adjacent_entropy_prev_step_key": int(prev_step_key) if prev_step_key is not None else -1,
                # Block-local denoise iteration index of the pair's first
                # step (t). Drives the per-pair rho ramp in the actor.
                "adjacent_entropy_pair_local_step": int(prev_step),
                "adjacent_entropy_prev_mean": 0.0,
                "adjacent_entropy_rollout_next_mean": 0.0,
                "adjacent_entropy_vocab_size": int(vocab_size),
            }
            next_mask_tensor = torch.zeros(
                (max(block_tokens, 0),),
                dtype=torch.bool,
            )
            unresolved_valid = False
            if (
                isinstance(mask_t, torch.Tensor)
                and isinstance(mask_next, torch.Tensor)
                and mask_t.shape[0] == entropy_t.shape[0]
                and mask_next.shape[0] == entropy_next.shape[0]
                and bool(mask_t.bool().any())
                and bool(mask_next.bool().any())
            ):
                prev_mean_raw = float(
                    entropy_t[mask_t.bool()].mean().item()
                )
                next_mean_raw = float(
                    entropy_next[mask_next.bool()].mean().item()
                )
                if (
                    math.isfinite(prev_mean_raw)
                    and math.isfinite(next_mean_raw)
                ):
                    annotation.update(
                        {
                            "adjacent_entropy_pair_valid": True,
                            "adjacent_entropy_prev_mean": (
                                prev_mean_raw / log_vocab
                            ),
                            "adjacent_entropy_rollout_next_mean": (
                                next_mean_raw / log_vocab
                            ),
                        }
                    )
                    next_mask_tensor = mask_next.bool().clone()
                    unresolved_valid = True
            if not unresolved_valid:
                empty_mask_rows += 1

            # --- Global AEC annotation (all-valid scope) ---
            # Uses all_valid_mask instead of random_vocab_mask. Both H_prev and H_next
            # must use the SAME position domain (all valid non-prompt positions).
            # Fail-closed: missing or empty all_valid_mask → global pair invalid.
            global_annotation: dict = {
                "global_aec_pair_valid": False,
                "global_aec_prev_step_key": int(prev_step_key) if prev_step_key is not None else -1,
                "global_aec_prev_mean": 0.0,
                "global_aec_rollout_next_mean": 0.0,
                "global_aec_vocab_size": int(vocab_size) if vocab_size is not None else 0,
            }
            global_next_mask_tensor = torch.zeros(
                (max(block_tokens, 0),),
                dtype=torch.bool,
            )
            global_valid = False
            if (
                isinstance(gall_mask_t, torch.Tensor)
                and isinstance(gall_mask_next, torch.Tensor)
            ):
                # Fail-closed if size mismatch between entropy and mask
                _nt_g = int(gall_mask_t.bool().sum().item())
                _nn_g = int(gall_mask_next.bool().sum().item())
                if _nt_g > 0 and _nn_g > 0:
                    if (
                        gall_mask_t.shape[0] != entropy_t.shape[0]
                        or gall_mask_next.shape[0] != entropy_next.shape[0]
                    ):
                        # size mismatch → fail-closed (do not silently crop)
                        logger.warning(
                            "[SDAR_GAEC] global AEC mask/entropy shape mismatch at "
                            "req=%d block=%d step=%d: mask_t=%s ent_t=%s mask_next=%s ent_next=%s",
                            req_id, block_idx, cur_step,
                            gall_mask_t.shape, entropy_t.shape,
                            gall_mask_next.shape, entropy_next.shape,
                        )
                    else:
                        g_prev_mean_raw = float(entropy_t[gall_mask_t.bool()].mean().item())
                        g_next_mean_raw = float(entropy_next[gall_mask_next.bool()].mean().item())
                        if (
                            math.isfinite(g_prev_mean_raw)
                            and math.isfinite(g_next_mean_raw)
                        ):
                            g_prev_mean = g_prev_mean_raw / log_vocab
                            g_next_mean = g_next_mean_raw / log_vocab
                            global_annotation.update({
                                "global_aec_pair_valid": True,
                                "global_aec_prev_mean": float(g_prev_mean),
                                "global_aec_rollout_next_mean": float(g_next_mean),
                            })
                            global_next_mask_tensor = gall_mask_next.bool().clone()
                            global_valid = True

            annotation.update(global_annotation)

            # Annotate meta_payload (for all transport modes)
            if isinstance(rp, dict):
                rp.update(annotation)
                rp["adjacent_entropy_next_mask"] = next_mask_tensor.tolist()
                if bool(getattr(self._algo, "global_aec_enabled", False)):
                    rp["global_aec_next_mask"] = global_next_mask_tensor.tolist()
                meta_in_rp = rp.get("metadata")
                if isinstance(meta_in_rp, dict):
                    meta_in_rp.update(annotation)
                    meta_in_rp["adjacent_entropy_next_mask"] = next_mask_tensor.tolist()
                    if bool(getattr(self._algo, "global_aec_enabled", False)):
                        meta_in_rp["global_aec_next_mask"] = global_next_mask_tensor.tolist()

            # Annotate tensor_payload in algo store (for memory mode)
            if cur_step_key is not None:
                tp = step_key_to_payload.get(int(cur_step_key))
                if isinstance(tp, dict):
                    tp.update(annotation)
                    tp["adjacent_entropy_next_mask"] = next_mask_tensor
                    if bool(getattr(self._algo, "global_aec_enabled", False)):
                        tp["global_aec_next_mask"] = global_next_mask_tensor
                    meta_in_tp = tp.get("metadata")
                    if isinstance(meta_in_tp, dict):
                        meta_in_tp.update(annotation)
                        meta_in_tp["adjacent_entropy_next_mask"] = next_mask_tensor.tolist()
                        if bool(getattr(self._algo, "global_aec_enabled", False)):
                            meta_in_tp["global_aec_next_mask"] = global_next_mask_tensor.tolist()

            if unresolved_valid or global_valid:
                valid_pairs += 1
            prev = cur

        self._metrics.record_adjacent_entropy_pairs(
            valid_pairs=valid_pairs,
            gapped_rows=gapped_rows,
            empty_mask_rows=empty_mask_rows,
            unpaired_rows=unpaired_rows,
        )
        if gapped_rows or empty_mask_rows:
            logger.debug(
                "[SDAR_AEC] block req=%d block=%d valid_pairs=%d gapped=%d "
                "empty_mask=%d unpaired=%d",
                req_id,
                block_idx,
                valid_pairs,
                gapped_rows,
                empty_mask_rows,
                unpaired_rows,
            )

    def _can_commit(self, handle: Any) -> bool:
        key = self._handle_key(handle)
        with self._condition:
            return key not in self._cancelled_keys and self._is_current(handle)

    def _copy_to_cpu(
        self,
        tensors: Dict[str, Optional[torch.Tensor]],
        producer_event: Optional[torch.cuda.Event],
    ) -> tuple[Dict[str, Optional[torch.Tensor]], float, float]:
        """Materialize tensors on a dedicated per-device CUDA copy stream."""
        cuda_tensor = next(
            (
                tensor
                for tensor in tensors.values()
                if isinstance(tensor, torch.Tensor) and tensor.is_cuda
            ),
            None,
        )
        event_start = time.monotonic_ns()
        if producer_event is not None:
            producer_event.synchronize()
        event_wait_us = (time.monotonic_ns() - event_start) / 1_000
        d2h_start = time.monotonic_ns()
        result: Dict[str, Optional[torch.Tensor]] = {}
        if cuda_tensor is None:
            for name, tensor in tensors.items():
                result[name] = (
                    tensor.detach().clone()
                    if isinstance(tensor, torch.Tensor)
                    else None
                )
            return (
                result,
                event_wait_us,
                (time.monotonic_ns() - d2h_start) / 1_000,
            )

        device_index = int(cuda_tensor.device.index or 0)
        try:
            stream = self._copy_streams.get(device_index)
            if stream is None:
                with torch.cuda.device(device_index):
                    stream = torch.cuda.Stream(device=device_index)
                self._copy_streams[device_index] = stream
            done = torch.cuda.Event()
            with torch.cuda.device(device_index), torch.cuda.stream(stream):
                for name, tensor in tensors.items():
                    if tensor is None:
                        result[name] = None
                    elif tensor.is_cuda:
                        cpu_tensor = torch.empty(
                            tuple(tensor.shape),
                            dtype=tensor.dtype,
                            device="cpu",
                            pin_memory=True,
                        )
                        cpu_tensor.copy_(tensor.detach(), non_blocking=True)
                        result[name] = cpu_tensor
                    else:
                        result[name] = tensor.detach().clone()
                done.record(stream)
            done.synchronize()
        except Exception as exc:
            self._metrics.record_copy_stream_fallback()
            error_text = f"{type(exc).__name__}: {exc}"
            logger.error(
                "[SDAR_ASYNC_CAPTURE] Dedicated D2H stream failed; "
                "falling back in consumer: %s",
                error_text,
            )
            result = {
                name: tensor.detach().cpu().clone() if tensor is not None else None
                for name, tensor in tensors.items()
            }
        d2h_us = (time.monotonic_ns() - d2h_start) / 1_000
        return result, event_wait_us, d2h_us

    # ------------------------------------------------------------------
    # Internal: process step handle
    # ------------------------------------------------------------------
    def _process_step(self, handle: StepHandle) -> None:
        t_start = time.monotonic_ns()
        token_limit = min(handle.block_size, max(1, handle.trace_tensor_max_tokens))
        gpu_lease_table = getattr(
            self._algo,
            "_gpu_replay_lease_table",
            None,
        )
        gpu_lease_enabled = gpu_lease_table is not None
        capture_replay_row = bool(handle.capture_replay)
        should_capture = getattr(
            self._algo,
            "_should_capture_replay_step",
            None,
        )
        if capture_replay_row and callable(should_capture):
            capture_replay_row = bool(
                should_capture(
                    req_pool_idx=handle.req_pool_idx,
                    generation=handle.generation,
                    block_idx=handle.block_idx,
                    local_step=handle.step,
                    is_final_denoise_row=(
                        handle.is_final_denoise_row
                    ),
                )
            )
        cpu, event_wait_us, d2h_us = self._copy_to_cpu(
            {
                "input_ids": handle.input_ids_snapshot[:token_limit],
                "current_ids": handle.current_ids_snapshot[:token_limit],
                "proposal": handle.proposal[:token_limit],
                "proposal_probs": handle.proposal_probs[:token_limit],
                "raw_logprobs": handle.raw_logprobs[:token_limit],
                "sample_logprobs": handle.sample_logprobs[:token_limit],
                "entropy": handle.entropy[:token_limit],
                "accepted_mask": handle.accepted_mask[:token_limit],
                "new_accept_mask": handle.new_accept_mask[:token_limit],
                "edit_mask": handle.edit_mask[:token_limit],
                "action_mask": handle.action_mask[:token_limit],
                "prompt_mask": handle.prompt_mask[:token_limit],
                "sc_signal": (
                    handle.sc_signal[:token_limit]
                    if handle.sc_signal is not None
                    and not gpu_lease_enabled
                    else None
                ),
                "logits_source_hidden": (
                    handle.logits_source_hidden[:token_limit]
                    if handle.logits_source_hidden is not None
                    and not gpu_lease_enabled
                    else None
                ),
                "valid": (
                    handle.valid_mask[:token_limit]
                    if handle.valid_mask is not None
                    else None
                ),
            },
            handle.cuda_event,
        )
        input_ids_cpu = cpu["input_ids"]
        current_ids_cpu = cpu["current_ids"]
        proposal_cpu = cpu["proposal"]
        proposal_probs_cpu = cpu["proposal_probs"]
        raw_logprobs_cpu = cpu["raw_logprobs"]
        sample_logprobs_cpu = cpu["sample_logprobs"]
        entropy_cpu = cpu["entropy"]
        accepted_mask_cpu = cpu["accepted_mask"]
        new_accept_mask_cpu = cpu["new_accept_mask"]
        edit_mask_cpu = cpu["edit_mask"]
        action_mask_cpu = cpu["action_mask"]
        prompt_mask_cpu = cpu["prompt_mask"]
        sc_signal_payload: Any = cpu["sc_signal"]
        logits_source_hidden_payload: Any = cpu[
            "logits_source_hidden"
        ]
        if gpu_lease_enabled and capture_replay_row:
            logits_source_hidden_tensor = (
                handle.logits_source_hidden[:token_limit]
                if handle.logits_source_hidden is not None
                else None
            )
            sc_signal_tensor = (
                handle.sc_signal[:token_limit]
                if handle.sc_signal is not None
                else None
            )
            if (
                sc_signal_tensor is not None
                and not handle.sc_signal_is_compact
            ):
                # A block slice can retain the full growing rollout storage.
                # Lease only a compact GPU allocation for this block.
                sc_signal_tensor = sc_signal_tensor.clone()
            if (
                sc_signal_tensor is None
                and logits_source_hidden_tensor is not None
            ):
                sc_signal_tensor = torch.zeros_like(
                    logits_source_hidden_tensor
                )
            if sc_signal_tensor is not None:
                sc_signal_payload = gpu_lease_table.export(
                    sc_signal_tensor,
                    logical_name=(
                        f"req{handle.req_pool_idx}:"
                        f"step{handle.step_key}:self_conditioning_signal"
                    ),
                )
            if (
                logits_source_hidden_tensor is not None
                and bool(
                    getattr(
                        self._algo,
                        "_gpu_replay_export_logits_source",
                        False,
                    )
                )
            ):
                logits_source_hidden_payload = (
                    gpu_lease_table.export(
                        logits_source_hidden_tensor,
                        logical_name=(
                            f"req{handle.req_pool_idx}:"
                            f"step{handle.step_key}:"
                            "logits_source_hidden"
                        ),
                    )
                )

        # --- Build trace step dict ---
        t_pack_start = time.monotonic_ns()

        new_accept_count = int(new_accept_mask_cpu.sum().item()) if new_accept_mask_cpu is not None else 0
        edit_change_count = int(edit_mask_cpu.sum().item()) if edit_mask_cpu is not None else 0
        action_token_count = int(action_mask_cpu.sum().item()) if action_mask_cpu is not None else 0

        def _nonzero_list(mask_cpu, limit=64):
            if mask_cpu is None:
                return []
            return [int(x) for x in torch.nonzero(mask_cpu, as_tuple=False).flatten()[:limit].tolist()]

        def _selected_entries(mask_cpu, proposal_c, raw_lp_c, sample_lp_c, pp_c, limit=64):
            if mask_cpu is None:
                return []
            positions = torch.nonzero(mask_cpu, as_tuple=False).flatten()[:limit]
            entries = []
            for pos in positions.tolist():
                pos = int(pos)
                entries.append({
                    "pos": pos,
                    "token_id": int(proposal_c[pos].item()) if proposal_c is not None else 0,
                    "raw_logprob": float(raw_lp_c[pos].item()) if raw_lp_c is not None else float("nan"),
                    "sample_logprob": float(sample_lp_c[pos].item()) if sample_lp_c is not None else float("nan"),
                    "sample_prob": float(pp_c[pos].item()) if pp_c is not None else float("nan"),
                })
            return entries

        trace_step: Dict[str, Any] = {
            "trace_tensors_enabled": handle.trace_tensors,
            "trace_tensor_records": handle.trace_tensor_records,
            "trace_dir_set": bool(self._algo._trace_dir),
            "replay_step_selected": capture_replay_row,
            "step": handle.step,
            "was_completed": handle.was_completed,
            "accepted_count": int(accepted_mask_cpu.sum().item()) if accepted_mask_cpu is not None else 0,
            "new_accept_count": new_accept_count,
            "edit_change_count": edit_change_count,
            "action_token_count": action_token_count,
            "accepted_positions": _nonzero_list(accepted_mask_cpu),
            "new_accept_positions": _nonzero_list(new_accept_mask_cpu),
            "edit_change_positions": _nonzero_list(edit_mask_cpu),
            "action_positions": _nonzero_list(action_mask_cpu),
            "action_token_ids": (
                [int(proposal_cpu[int(x)].item())
                 for x in torch.nonzero(action_mask_cpu, as_tuple=False).flatten()[:64].tolist()]
                if (action_mask_cpu is not None and proposal_cpu is not None) else []
            ),
            "new_accept_token_logprobs": _selected_entries(
                new_accept_mask_cpu, proposal_cpu, raw_logprobs_cpu,
                sample_logprobs_cpu,
                proposal_probs_cpu,
            ),
            "edit_token_logprobs": _selected_entries(
                edit_mask_cpu, proposal_cpu, raw_logprobs_cpu,
                sample_logprobs_cpu,
                proposal_probs_cpu,
            ),
            "max_prob_mean": float(proposal_probs_cpu.mean().item()) if proposal_probs_cpu is not None else float("nan"),
            "entropy_mean": float(entropy_cpu.mean().item()) if entropy_cpu is not None else float("nan"),
            "head_tokens": [int(x) for x in (current_ids_cpu[:16].tolist() if current_ids_cpu is not None else [])],
            "proposal_head": [int(x) for x in (proposal_cpu[:32].tolist() if proposal_cpu is not None else [])],
            "maxp_head": [float(x) for x in (proposal_probs_cpu[:16].tolist() if proposal_probs_cpu is not None else [])],
            "self_conditioning": handle.sc_stats,
        }

        if handle.trace_full:
            trace_step.update({
                "current_ids": current_ids_cpu.tolist() if current_ids_cpu is not None else [],
                "proposal_ids": proposal_cpu.tolist() if proposal_cpu is not None else [],
                "action_mask": action_mask_cpu.tolist() if action_mask_cpu is not None else [],
                "accepted_mask": accepted_mask_cpu.tolist() if accepted_mask_cpu is not None else [],
                "new_accept_mask": new_accept_mask_cpu.tolist() if new_accept_mask_cpu is not None else [],
                "edit_mask": edit_mask_cpu.tolist() if edit_mask_cpu is not None else [],
                "proposal_probs": proposal_probs_cpu.tolist() if proposal_probs_cpu is not None else [],
            })

        # --- Build replay tensor payload ---
        replay_payload: Optional[Dict] = None
        if capture_replay_row and self._algo._trace_dir:
            replay_payload = self._build_replay_payload(
                handle, token_limit,
                input_ids_cpu, proposal_cpu, action_mask_cpu,
                accepted_mask_cpu, new_accept_mask_cpu,
                edit_mask_cpu, prompt_mask_cpu,
                sc_signal_payload,
                logits_source_hidden_payload,
                cpu["valid"],
            )
            if replay_payload is not None:
                trace_step["replay_tensors"] = replay_payload

        # --- Capture entropy/mask for adjacent entropy pair annotation ---
        # Extract before _maybe_write_replay_file pops _tensor_payload.
        aec_entropy_cpu: Optional[torch.Tensor] = None
        aec_random_vocab_mask_cpu: Optional[torch.Tensor] = None
        aec_all_valid_mask_cpu: Optional[torch.Tensor] = None
        _global_aec_enabled = bool(getattr(self._algo, "global_aec_enabled", False))
        if capture_replay_row and replay_payload is not None:
            _tp = replay_payload.get("_tensor_payload")
            if isinstance(_tp, dict):
                if self._adjacent_entropy_contraction_enabled:
                    rvm = _tp.get("random_vocab_mask")
                    if isinstance(rvm, torch.Tensor):
                        aec_random_vocab_mask_cpu = rvm.bool()
                if _global_aec_enabled:
                    avm = _tp.get("all_valid_mask")
                    if isinstance(avm, torch.Tensor):
                        aec_all_valid_mask_cpu = avm.bool()
            if isinstance(entropy_cpu, torch.Tensor) and (
                self._adjacent_entropy_contraction_enabled or _global_aec_enabled
            ):
                aec_entropy_cpu = entropy_cpu.float()

        pack_us = (time.monotonic_ns() - t_pack_start) / 1_000

        # --- Write files ---
        t_write_start = time.monotonic_ns()
        bytes_written = 0
        if not self._can_commit(handle):
            return
        if replay_payload is not None and handle.capture_replay:
            bytes_written = self._maybe_write_replay_file(replay_payload, handle)
        write_us = (time.monotonic_ns() - t_write_start) / 1_000

        # --- Accumulate per-request step handles (as processed dicts) ---
        step_group_key = (
            int(handle.req_pool_idx),
            int(handle.generation),
            int(handle.block_idx),
        )
        with self._condition:
            if step_group_key[:2] not in self._cancelled_keys:
                step_entry: Dict[str, Any] = {
                    "trace_step": trace_step,
                    "replay_payload": replay_payload,
                    "new_accept_count": new_accept_count,
                    "edit_change_count": edit_change_count,
                    "step": handle.step,
                    "block_idx": handle.block_idx,
                    "is_final_denoise_row": handle.is_final_denoise_row,
                }
                if aec_entropy_cpu is not None:
                    step_entry["_aec_entropy_cpu"] = aec_entropy_cpu
                if aec_random_vocab_mask_cpu is not None:
                    step_entry["_aec_random_vocab_mask_cpu"] = aec_random_vocab_mask_cpu
                if aec_all_valid_mask_cpu is not None:
                    step_entry["_aec_all_valid_mask_cpu"] = aec_all_valid_mask_cpu
                self._req_step_handles.setdefault(step_group_key, []).append(step_entry)

        e2e_us = (time.monotonic_ns() - t_start) / 1_000
        self._metrics.record_process(event_wait_us, pack_us, d2h_us, write_us, bytes_written, e2e_us)
        if replay_payload is not None:
            self._metrics.record_replay_row()

    def _build_replay_payload(
        self,
        handle: StepHandle,
        token_limit: int,
        input_ids_cpu: Optional[torch.Tensor],
        proposal_cpu: Optional[torch.Tensor],
        action_mask_cpu: Optional[torch.Tensor],
        accepted_mask_cpu: Optional[torch.Tensor],
        new_accept_mask_cpu: Optional[torch.Tensor],
        edit_mask_cpu: Optional[torch.Tensor],
        prompt_mask_cpu: Optional[torch.Tensor],
        sc_signal_cpu: Any,
        lsh_cpu: Any,
        valid_cpu: Optional[torch.Tensor],
    ) -> Optional[Dict]:
        """Build the replay tensor payload on the consumer thread."""

        if sc_signal_cpu is None and lsh_cpu is None and input_ids_cpu is None:
            return None

        used_self_conditioning_signal = sc_signal_cpu is not None
        if (
            sc_signal_cpu is None
            and isinstance(lsh_cpu, torch.Tensor)
        ):
            sc_signal_cpu = torch.zeros(
                (handle.block_size, int(lsh_cpu.shape[-1])),
                dtype=lsh_cpu.dtype,
                device="cpu",
            )[:token_limit]
        if valid_cpu is None:
            valid_cpu = (
                ~prompt_mask_cpu
                if prompt_mask_cpu is not None
                else torch.ones(token_limit, dtype=torch.bool)
            )

        replay_active_mask_cpu = None
        if new_accept_mask_cpu is not None:
            replay_active_mask_cpu = new_accept_mask_cpu.clone()
            if edit_mask_cpu is not None:
                replay_active_mask_cpu = replay_active_mask_cpu | edit_mask_cpu
        elif edit_mask_cpu is not None:
            replay_active_mask_cpu = edit_mask_cpu.clone()
        random_vocab_mask_cpu = None
        if (
            accepted_mask_cpu is not None
            and new_accept_mask_cpu is not None
        ):
            accepted_before_cpu = (
                accepted_mask_cpu.bool()
                & ~new_accept_mask_cpu.bool()
            )
            random_vocab_mask_cpu = ~accepted_before_cpu
            if prompt_mask_cpu is not None:
                random_vocab_mask_cpu &= ~prompt_mask_cpu.bool()
        all_valid_mask_cpu = None
        if (
            bool(getattr(self._algo, "global_aec_enabled", False))
            and isinstance(valid_cpu, torch.Tensor)
        ):
            all_valid_mask_cpu = valid_cpu.bool().clone()
            if prompt_mask_cpu is not None:
                all_valid_mask_cpu &= ~prompt_mask_cpu.bool()

        new_accept_count = int(new_accept_mask_cpu.sum().item()) if new_accept_mask_cpu is not None else 0
        edit_change_count = int(edit_mask_cpu.sum().item()) if edit_mask_cpu is not None else 0
        action_count = int(action_mask_cpu.sum().item()) if action_mask_cpu is not None else 0

        tensor_payload: Dict[str, Any] = {
            "position": (handle.position_start, handle.position_start + handle.block_size),
            "valid": valid_cpu,
            "prompt_mask": prompt_mask_cpu,
            "action_ids": proposal_cpu,
            "action_mask": action_mask_cpu,
            "new_accept_mask": new_accept_mask_cpu,
            "edit_change_mask": edit_mask_cpu,
            "replay_active_mask": replay_active_mask_cpu,
            "random_vocab_mask": random_vocab_mask_cpu,
            "all_valid_mask": all_valid_mask_cpu,
            "input_ids": input_ids_cpu,
            "self_conditioning_signal": sc_signal_cpu,
            "logits_source_hidden": lsh_cpu,
            "metadata": {
                "format": "sglang_sdar_replay_tensor_v1",
                "step_key": handle.step_key,
                "local_step": handle.step,
                "req_pool_idx": handle.req_pool_idx,
                "block_idx": handle.block_idx,
                "start_offset": handle.start_offset,
                "response_start_index": handle.response_start_index,
                "block_tokens": handle.block_size,
                "stored_tokens": token_limit,
                "truncated": token_limit != handle.block_size,
                "new_accept_count": new_accept_count,
                "edit_change_count": edit_change_count,
                "action_token_count": action_count,
                "is_final_denoise_row": handle.is_final_denoise_row,
                "is_terminal_commit": False,
                "has_self_conditioning_signal": sc_signal_cpu is not None,
                "used_self_conditioning_signal": used_self_conditioning_signal,
                "has_logits_source_hidden": lsh_cpu is not None,
                "gpu_replay_lease": isinstance(
                    sc_signal_cpu,
                    dict,
                )
                or isinstance(lsh_cpu, dict),
            },
        }

        meta_payload: Dict[str, Any] = {
            "format": "sglang_sdar_replay_tensor_v1",
            "step_key": handle.step_key,
            "local_step": handle.step,
            "block_idx": handle.block_idx,
            "req_pool_idx": handle.req_pool_idx,
            "position": [handle.position_start, handle.position_start + handle.block_size],
            "start_offset": handle.start_offset,
            "response_start_index": handle.response_start_index,
            "stored_tokens": token_limit,
            "block_tokens": handle.block_size,
            "truncated": token_limit != handle.block_size,
            "prompt_mask": prompt_mask_cpu.tolist() if prompt_mask_cpu is not None else [],
            "action_ids": proposal_cpu.tolist() if proposal_cpu is not None else [],
            "action_mask": action_mask_cpu.tolist() if action_mask_cpu is not None else [],
            "new_accept_count": new_accept_count,
            "edit_change_count": edit_change_count,
            "action_token_count": action_count,
            "is_final_denoise_row": handle.is_final_denoise_row,
            "is_terminal_commit": False,
            "new_accept_positions": (
                [int(x) for x in torch.nonzero(new_accept_mask_cpu, as_tuple=False).flatten()[:token_limit].tolist()]
                if new_accept_mask_cpu is not None else []
            ),
            "edit_change_positions": (
                [int(x) for x in torch.nonzero(edit_mask_cpu, as_tuple=False).flatten()[:token_limit].tolist()]
                if edit_mask_cpu is not None else []
            ),
            "has_self_conditioning_signal": sc_signal_cpu is not None,
            "used_self_conditioning_signal": used_self_conditioning_signal,
            "has_logits_source_hidden": lsh_cpu is not None,
            "gpu_replay_lease": isinstance(sc_signal_cpu, dict)
            or isinstance(lsh_cpu, dict),
            "has_input_ids": input_ids_cpu is not None,
            "_tensor_payload": tensor_payload,
        }
        cleanup_descriptors = []
        for descriptor in (
            sc_signal_cpu,
            lsh_cpu,
        ):
            if (
                isinstance(descriptor, dict)
                and descriptor.get("format")
                == "sglang_sdar_gpu_replay_lease_v1"
            ):
                cleanup_descriptors.append(
                    {
                        "format": descriptor["format"],
                        "lease_id": descriptor.get("lease_id"),
                        "ack_shm_name": descriptor.get("ack_shm_name"),
                    }
                )
        if cleanup_descriptors:
            # File-backed rows otherwise hide the lease descriptors inside
            # the .pt payload. Keep only acknowledgement-safe scalar fields
            # in metadata so block pruning can release before unlinking.
            meta_payload[
                "_gpu_replay_cleanup_descriptors"
            ] = cleanup_descriptors
        if getattr(self._algo, "_trace_tensor_inline", False):
            if isinstance(sc_signal_cpu, torch.Tensor):
                meta_payload["self_conditioning_signal"] = (
                    sc_signal_cpu.float().tolist()
                )
            if valid_cpu is not None:
                meta_payload["valid"] = valid_cpu.bool().tolist()
            if isinstance(lsh_cpu, torch.Tensor):
                meta_payload["logits_source_hidden"] = lsh_cpu.float().tolist()
        return meta_payload

    def _maybe_write_replay_file(self, meta_payload: Dict, handle: StepHandle) -> int:
        """Write replay tensors to file if store mode == file; return bytes."""
        if handle.trace_tensor_store in _MEMORY_TRACE_TENSOR_STORES:
            # Memory mode: store the tensor_payload in the algorithm's dict
            tensor_payload = meta_payload.pop("_tensor_payload", None)
            algo = self._algo
            if tensor_payload is not None:
                with self._condition:
                    if not self._can_commit(handle):
                        return 0
                    with algo._replay_tensors_lock:
                        algo._request_replay_tensors_by_req.setdefault(
                            handle.req_pool_idx, []
                        ).append(tensor_payload)
            return 0
        else:
            # File mode: write .pt file
            tensor_payload = meta_payload.pop("_tensor_payload", None)
            if tensor_payload is None:
                return 0
            path = self._algo._write_replay_tensor_file(
                tensor_payload,
                req_pool_idx=handle.req_pool_idx,
                block_idx=handle.block_idx,
                step=handle.step,
            )
            meta_payload["tensor_path"] = path
            meta_payload["trace_root"] = os.path.abspath(self._algo._trace_dir)
            try:
                file_bytes = int(os.path.getsize(path))
            except OSError:
                file_bytes = 0
            meta_payload["tensor_file_bytes"] = file_bytes
            # Also store the meta_payload (without tensor_payload) in the algo dict
            with self._condition:
                if not self._can_commit(handle):
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
                    return 0
                with self._algo._replay_tensors_lock:
                    self._algo._request_replay_tensors_by_req.setdefault(
                        handle.req_pool_idx, []
                    ).append(meta_payload)
            return file_bytes

    # ------------------------------------------------------------------
    # Internal: process block handle
    # ------------------------------------------------------------------
    def _process_block(self, handle: BlockHandle) -> None:
        t_start = time.monotonic_ns()
        block_size = handle.final_ids.shape[0] if handle.final_ids is not None else 0
        cpu, event_wait_us, d2h_us = self._copy_to_cpu(
            {
                "final_ids": handle.final_ids,
                "commit_argmax": handle.commit_argmax,
                "commit_argmax_probs": handle.commit_argmax_probs,
                "p_probs": handle.p_probs,
                "q_probs": handle.q_probs,
                "q_raw_logprobs": handle.q_raw_logprobs,
                "q_sample_logprobs": handle.q_sample_logprobs,
                "prompt_mask": handle.prompt_mask,
            },
            handle.cuda_event,
        )
        final_ids_cpu = cpu["final_ids"]
        commit_argmax_cpu = cpu["commit_argmax"]
        commit_argmax_probs_cpu = cpu["commit_argmax_probs"]
        p_probs_cpu = cpu["p_probs"]
        q_probs_cpu = cpu["q_probs"]
        q_raw_logprobs_cpu = cpu["q_raw_logprobs"]
        q_sample_logprobs_cpu = cpu["q_sample_logprobs"]

        t_pack_start = time.monotonic_ns()

        # Collect accumulated step handles for this block
        req_id = handle.req_pool_idx
        step_group_key = (
            int(handle.req_pool_idx),
            int(handle.generation),
            int(handle.block_idx),
        )
        with self._condition:
            if not self._can_commit(handle):
                self._req_step_handles.pop(step_group_key, None)
                return
            step_dicts = self._req_step_handles.pop(step_group_key, [])

        # Compute new_accept_tokens and edit_change_tokens from accumulated steps
        total_new_accept = sum(s["new_accept_count"] for s in step_dicts)
        total_edit_change = sum(s["edit_change_count"] for s in step_dicts)

        # Mark final denoise row in step_dicts
        if step_dicts:
            for step_dict in step_dicts:
                trace_step = step_dict["trace_step"]
                trace_step["is_final_denoise_row"] = False
                replay_meta = trace_step.get("replay_tensors")
                if isinstance(replay_meta, dict):
                    replay_meta["is_final_denoise_row"] = False
            final_step = step_dicts[-1]["trace_step"]
            final_step["is_final_denoise_row"] = True
            replay_meta = final_step.get("replay_tensors")
            if isinstance(replay_meta, dict):
                replay_meta["is_final_denoise_row"] = True
                final_step_key = replay_meta.get("step_key")
            else:
                final_step_key = None
        else:
            final_step_key = None

        # Adjacent entropy pair annotation: runs after all step rows for the
        # block are available and before preselection/transport.
        # Runs for standard unresolved AEC and/or global (all-valid) AEC.
        _global_aec_active = bool(getattr(self._algo, "global_aec_enabled", False))
        if _global_aec_active:
            _global_response_end = _clip_global_aec_masks_at_eos(
                step_dicts,
                final_ids_cpu,
                handle.start_offset,
                set(getattr(self._algo, "eos_token_ids", set())),
            )
            _clip_logs = int(
                getattr(self, "_global_aec_clip_log_count", 0)
            )
            if _clip_logs < 8:
                eos_ids = sorted(
                    set(getattr(self._algo, "eos_token_ids", set()))
                )
                logger.warning(
                    "[SDAR_GAEC_CLIP] block=%d start=%d eos_ids=%s "
                    "response_end=%s final_tokens=%d tail=%s",
                    handle.block_idx,
                    handle.start_offset,
                    eos_ids,
                    _global_response_end,
                    int(final_ids_cpu.numel())
                    if isinstance(final_ids_cpu, torch.Tensor)
                    else 0,
                    (
                        final_ids_cpu[-8:].tolist()
                        if isinstance(final_ids_cpu, torch.Tensor)
                        else []
                    ),
                )
                self._global_aec_clip_log_count = _clip_logs + 1
        if self._adjacent_entropy_contraction_enabled or _global_aec_active:
            self._annotate_adjacent_entropy_pairs(
                step_dicts, req_id, handle.block_idx
            )

        # Attach block TPF summary to accumulated replay rows for this block
        if self._algo._trace_tensors:
            with self._algo._replay_tensors_lock:
                all_rows = self._algo._request_replay_tensors_by_req.get(req_id, [])
                for tensor_payload in all_rows:
                    if not isinstance(tensor_payload, dict):
                        continue
                    metadata = tensor_payload.get("metadata") if isinstance(tensor_payload.get("metadata"), dict) else tensor_payload
                    if not isinstance(metadata, dict):
                        continue
                    if int(metadata.get("block_idx", tensor_payload.get("block_idx", -1))) != int(handle.block_idx):
                        continue
                    metadata["new_accept_tokens"] = total_new_accept
                    metadata["denoise_forwards"] = handle.denoise_forwards
                    metadata["is_final_denoise_row"] = False
                    tensor_payload["new_accept_tokens"] = total_new_accept
                    tensor_payload["denoise_forwards"] = handle.denoise_forwards
                    tensor_payload["is_final_denoise_row"] = False
                    if final_step_key is not None and metadata.get("step_key") == final_step_key:
                        metadata["is_final_denoise_row"] = True
                        tensor_payload["is_final_denoise_row"] = True

            # Preselect replay rows
            with self._algo._replay_tensors_lock:
                self._algo._preselect_completed_block_replay(
                    req_id, handle.block_idx
                )

        # Build introspection record
        introspection_record = None
        if (
            final_ids_cpu is not None
            and p_probs_cpu is not None
            and handle.trace_dir
        ):
            start_offset = handle.start_offset
            nonprompt_start = start_offset
            introspection_record = {
                "time": time.time(),
                "req_pool_idx": handle.req_pool_idx,
                "block_idx": handle.block_idx,
                "block_start": handle.block_start,
                "block_end": handle.block_end,
                "start_offset": start_offset,
                "avg_introspective_consistency": float("nan"),
                "positions": [],
            }
            if commit_argmax_cpu is not None and commit_argmax_probs_cpu is not None:
                valid = (
                    torch.isfinite(q_probs_cpu) & (q_probs_cpu > 0)
                    if q_probs_cpu is not None else torch.zeros(block_size, dtype=torch.bool)
                )
                ratios = torch.ones(block_size, dtype=torch.float32)
                if q_probs_cpu is not None:
                    valid_idx = valid.nonzero(as_tuple=False).flatten()
                    for idx_item in valid_idx:
                        i = int(idx_item.item())
                        q_val = float(q_probs_cpu[i].item())
                        p_val = float(p_probs_cpu[i].item()) if p_probs_cpu is not None else 1.0
                        ratios[i] = min(1.0, p_val / max(q_val, 1e-12))
                nonprompt = torch.arange(block_size) >= nonprompt_start
                valid_np = nonprompt & valid
                avg_ratio = float(ratios[valid_np].mean().item()) if bool(valid_np.any()) else float("nan")
                introspection_record["avg_introspective_consistency"] = avg_ratio
                positions = []
                for i in range(block_size):
                    positions.append({
                        "pos": i,
                        "token_id": int(final_ids_cpu[i].item()) if final_ids_cpu is not None else 0,
                        "q": float(q_probs_cpu[i].item()) if (q_probs_cpu is not None and bool(torch.isfinite(q_probs_cpu[i]).item())) else None,
                        "p": float(p_probs_cpu[i].item()) if p_probs_cpu is not None else float("nan"),
                        "commit_argmax_id": int(commit_argmax_cpu[i].item()),
                        "commit_argmax_p": float(commit_argmax_probs_cpu[i].item()),
                        "commit_argmax_matches_final": bool(
                            commit_argmax_cpu[i].item() == final_ids_cpu[i].item()
                            if final_ids_cpu is not None else False
                        ),
                        "ratio": float(ratios[i].item()),
                        "is_prompt": bool(i < nonprompt_start),
                    })
                introspection_record["positions"] = positions

        # Build trace record from accumulated step dicts
        trace_record = None
        if handle.trace_dir:
            final_head_tokens = (
                [int(x) for x in final_ids_cpu[:block_size].tolist()]
                if final_ids_cpu is not None else []
            )
            trace_record = {
                "time": time.time(),
                "req_pool_idx": handle.req_pool_idx,
                "block_idx": handle.block_idx,
                "block_start": handle.block_start,
                "block_end": handle.block_end,
                "start_offset": handle.start_offset,
                "denoise_forwards": handle.denoise_forwards,
                "batch_denoise_forwards": handle.batch_denoise_forwards,
                "accepted_tokens": total_new_accept,
                "new_accept_count": total_new_accept,
                "new_accept_tokens": total_new_accept,
                "edit_change_count": total_edit_change,
                "edit_change_tokens": total_edit_change,
                "adaptive_temperature_diagnostics": (
                    handle.adaptive_temperature_diagnostics
                ),
                "adaptive_temperature_summary": (
                    summarize_adaptive_temperature_diagnostics(
                        handle.adaptive_temperature_diagnostics
                    )
                ),
                "adaptive_confidence_temperature_diagnostics": (
                    handle.adaptive_confidence_temperature_diagnostics
                ),
                "adaptive_confidence_temperature_summary": (
                    summarize_adaptive_confidence_temperature_diagnostics(
                        handle.adaptive_confidence_temperature_diagnostics
                    )
                ),
                "final_head_tokens": final_head_tokens,
                "steps": [s["trace_step"] for s in step_dicts],
            }

        # Build request stats
        emitted_tokens = block_size - handle.start_offset
        clean_token_probs = (
            [float(x) for x in p_probs_cpu[handle.start_offset:].float().tolist()]
            if p_probs_cpu is not None else []
        )
        selected_logprobs = {
            "token_ids": (
                [int(x) for x in final_ids_cpu[handle.start_offset:].tolist()]
                if final_ids_cpu is not None else []
            ),
            "raw_logprobs": (
                [float(x) for x in q_raw_logprobs_cpu[handle.start_offset:].tolist()]
                if q_raw_logprobs_cpu is not None else []
            ),
            "sample_logprobs": (
                [float(x) for x in q_sample_logprobs_cpu[handle.start_offset:].tolist()]
                if q_sample_logprobs_cpu is not None else []
            ),
            "sample_probs": (
                [float(x) for x in q_probs_cpu[handle.start_offset:].tolist()]
                if q_probs_cpu is not None else []
            ),
        }

        pack_us = (time.monotonic_ns() - t_pack_start) / 1_000

        # Write files
        t_write_start = time.monotonic_ns()
        bytes_written = 0
        if trace_record is not None:
            self._algo._write_trace_record(trace_record)
            bytes_written += len(json.dumps(trace_record).encode())
        if introspection_record is not None:
            self._algo._write_introspection_record(introspection_record)
            bytes_written += len(json.dumps(introspection_record).encode())
        write_us = (time.monotonic_ns() - t_write_start) / 1_000

        # Store selected logprobs (thread-safe: only consumer writes; main thread reads after drain)
        request_stats = {
            "block_idx": handle.block_idx,
            "block_size": block_size,
            "start_offset": handle.start_offset,
            "denoise_forwards": handle.denoise_forwards,
            "commit_forwards": 1,
            "total_forwards": handle.denoise_forwards + 1,
            "accepted_tokens": total_new_accept,
            "new_accept_count": total_new_accept,
            "new_accept_tokens": total_new_accept,
            "edit_change_count": total_edit_change,
            "edit_change_tokens": total_edit_change,
            "returned_tokens": emitted_tokens,
            "total_tokens": emitted_tokens,
            "generated_tokens": emitted_tokens,
            "true_forward_tokens": int(block_size * handle.denoise_forwards),
            "accepted_tokens_per_denoise_forward": total_new_accept / max(handle.denoise_forwards, 1),
            "new_accept_tokens_per_denoise_forward": total_new_accept / max(handle.denoise_forwards, 1),
            "edit_change_tokens_per_denoise_forward": total_edit_change / max(handle.denoise_forwards, 1),
            "tokens_per_denoise_forward": emitted_tokens / max(handle.denoise_forwards, 1),
            "clean_token_start_index": handle.response_clean_start,
            "clean_token_probs": clean_token_probs,
        }
        if handle.adaptive_temperature_diagnostics:
            request_stats["adaptive_temperature_diagnostics"] = [
                dict(item)
                for item in handle.adaptive_temperature_diagnostics
            ]
            request_stats["adaptive_temperature_summary"] = (
                summarize_adaptive_temperature_diagnostics(
                    handle.adaptive_temperature_diagnostics
                )
            )
        if handle.adaptive_confidence_temperature_diagnostics:
            request_stats[
                "adaptive_confidence_temperature_diagnostics"
            ] = [
                dict(item)
                for item in (
                    handle.adaptive_confidence_temperature_diagnostics
                )
            ]
            request_stats[
                "adaptive_confidence_temperature_summary"
            ] = (
                summarize_adaptive_confidence_temperature_diagnostics(
                    handle.adaptive_confidence_temperature_diagnostics
                )
            )
        if getattr(self._algo, "_sdar_cuda_graph_enabled", False):
            graph_stats = self._algo._cuda_graph_metrics.get_stats(
                synchronize=False
            )
            for key in (
                "cuda_graph_total_forwards",
                "cuda_graph_replays",
                "cuda_graph_eager_fallbacks",
                "cuda_graph_hit_rate",
            ):
                request_stats[key] = graph_stats[key]
        if trace_record is not None:
            request_stats["replay_trace_record"] = trace_record
        with self._condition:
            if not self._can_commit(handle):
                return
            with self._algo._replay_tensors_lock:
                self._algo._selected_logprobs_by_req[
                    handle.req_pool_idx
                ] = selected_logprobs
                self._algo._request_stats_by_req.setdefault(
                    handle.req_pool_idx, []
                ).append(request_stats)
        record_counts = getattr(self._algo, "_record_async_block_counts", None)
        if record_counts is not None:
            record_counts(total_new_accept, total_edit_change)

        e2e_us = (time.monotonic_ns() - t_start) / 1_000
        self._metrics.record_process(event_wait_us, pack_us, d2h_us, write_us, bytes_written, e2e_us)

    # ------------------------------------------------------------------
    # Public: expose accumulated step data for requests that skip BlockHandle
    # ------------------------------------------------------------------
    def flush_request_step_handles(self, req_pool_idx: int) -> None:
        """Compatibility wrapper for older callers; performs real cancellation."""
        self.cancel_request(req_pool_idx)
