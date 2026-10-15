"""Bounded asynchronous intermediate-replay capture for DiffusionGemma.

The rollout thread owns only the work needed to make immutable GPU snapshots,
record a producer-stream event, and enqueue a bounded handle.  A single FIFO
consumer waits for that event, performs D2H copies, validates row ordering and
cardinality, and assembles the existing ``gemma_intermediate_replay_v1``
payload.

Capture is lossless by construction: queue exhaustion, malformed rows,
consumer failures, and incomplete block finalization are surfaced to the
rollout caller.  No path drops replay rows or returns a partial payload.
"""

from __future__ import annotations

import atexit
import dataclasses
import logging
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, Optional, Sequence

import torch

logger = logging.getLogger(__name__)


class GemmaReplayCaptureError(RuntimeError):
    """Base class for fail-closed intermediate-replay capture errors."""


class GemmaReplayCaptureBackpressureError(GemmaReplayCaptureError):
    """The bounded queue could not accept a lossless capture handle."""


class GemmaReplayCaptureConsumerError(GemmaReplayCaptureError):
    """The background consumer failed while materializing replay."""


class GemmaReplayCaptureDrainTimeout(GemmaReplayCaptureError):
    """A block did not finish capture before its rollout boundary."""


class GemmaReplayCaptureIntegrityError(GemmaReplayCaptureError):
    """Captured replay violated a row or block integrity invariant."""


class GemmaReplayCaptureOrderingError(GemmaReplayCaptureIntegrityError):
    """A replay row was duplicated or processed out of order."""


class GemmaReplayCaptureMissingRowError(GemmaReplayCaptureIntegrityError):
    """Block finalization found fewer or more rows than rollout reported."""


class GemmaReplayCaptureMissingBlockError(GemmaReplayCaptureIntegrityError):
    """Rows were drained without a matching block-finalization handle."""


class GemmaReplayCaptureDuplicateBlockError(GemmaReplayCaptureIntegrityError):
    """A capture block was finalized more than once."""


def _tensor_bytes(value: torch.Tensor) -> int:
    return int(value.numel()) * int(value.element_size())


def _retained_tensor_bytes(value: Any) -> int:
    if not isinstance(value, torch.Tensor):
        return 0
    try:
        return int(value.untyped_storage().nbytes())
    except Exception:
        return _tensor_bytes(value)


def estimate_gemma_capture_handle_bytes(handle: Any) -> int:
    """Conservatively account tensor storage retained by a queue handle."""

    return sum(
        _retained_tensor_bytes(getattr(handle, field.name))
        for field in dataclasses.fields(handle)
    )


def build_gemma_intermediate_replay_item(
    *,
    canvas_length: int,
    req_pool_idx: int,
    block_index: int,
    target_ids: torch.Tensor,
    step_indices: list[int],
    canvases: list[torch.Tensor],
    self_conditioning: list[torch.Tensor],
    self_conditioning_present: list[bool],
    done_after_step: list[bool],
    steps_used: int,
    capture_started_wall_ts: float,
    capture_seconds: float,
    capture_buffer_bytes: int,
    block_committed_wall_ts: Optional[float] = None,
) -> dict:
    """Build the pre-existing Gemma replay schema from CPU row tensors."""

    rows = len(step_indices)
    if rows != int(steps_used):
        raise GemmaReplayCaptureMissingRowError(
            "DiffusionGemma replay row/steps parity failed: "
            f"captured={rows} steps_used={steps_used}"
        )
    if not (
        rows
        == len(canvases)
        == len(self_conditioning)
        == len(self_conditioning_present)
        == len(done_after_step)
    ):
        raise GemmaReplayCaptureMissingRowError(
            "DiffusionGemma replay has missing per-step inputs"
        )
    if rows < 1:
        raise GemmaReplayCaptureMissingRowError(
            "DiffusionGemma replay captured no denoise rows"
        )

    canvas_tensor = torch.stack(canvases, dim=0).to(dtype=torch.long)
    signal_tensor = torch.stack(self_conditioning, dim=0)
    step_tensor = torch.tensor(step_indices, dtype=torch.int32)
    present_tensor = torch.tensor(self_conditioning_present, dtype=torch.bool)
    done_rows = torch.tensor(done_after_step, dtype=torch.bool)
    valid_masks = torch.ones((rows, canvas_length), dtype=torch.bool)
    done_masks = done_rows[:, None].expand(-1, canvas_length).clone()
    terminal_mask = torch.zeros_like(valid_masks)
    terminal_mask[-1].fill_(True)
    row_valid_mask = torch.ones(rows, dtype=torch.bool)
    target_valid_mask = torch.ones(canvas_length, dtype=torch.bool)
    target_tensor = target_ids.detach().to(
        device="cpu", dtype=torch.long, copy=True
    )

    expected_canvas_shape = (rows, canvas_length)
    if tuple(canvas_tensor.shape) != expected_canvas_shape:
        raise GemmaReplayCaptureIntegrityError(
            "DiffusionGemma replay canvas shape mismatch: "
            f"{tuple(canvas_tensor.shape)} != {expected_canvas_shape}"
        )
    if (
        signal_tensor.ndim != 3
        or tuple(signal_tensor.shape[:2]) != expected_canvas_shape
    ):
        raise GemmaReplayCaptureIntegrityError(
            "DiffusionGemma replay self-conditioning shape mismatch: "
            f"{tuple(signal_tensor.shape)}"
        )
    if target_tensor.numel() != canvas_length:
        raise GemmaReplayCaptureIntegrityError(
            "DiffusionGemma replay target length mismatch: "
            f"{target_tensor.numel()} != {canvas_length}"
        )
    if bool(present_tensor[0]):
        raise GemmaReplayCaptureIntegrityError(
            "DiffusionGemma replay first step unexpectedly has "
            "self-conditioning"
        )
    if bool(torch.count_nonzero(signal_tensor[0])):
        raise GemmaReplayCaptureIntegrityError(
            "DiffusionGemma replay first-step self-conditioning is not zero"
        )
    if rows > 1 and not bool(present_tensor[1:].all()):
        raise GemmaReplayCaptureIntegrityError(
            "DiffusionGemma replay is missing later self-conditioning"
        )

    tensors = (
        target_tensor,
        step_tensor,
        canvas_tensor,
        signal_tensor,
        present_tensor,
        valid_masks,
        done_masks,
        terminal_mask,
        row_valid_mask,
        target_valid_mask,
    )
    tensor_bytes = sum(_tensor_bytes(value) for value in tensors)
    producer_peak_bytes = int(capture_buffer_bytes) + int(tensor_bytes)
    committed_ts = (
        time.time()
        if block_committed_wall_ts is None
        else float(block_committed_wall_ts)
    )
    return {
        "format": "gemma_intermediate_replay_v1",
        "schema_version": 1,
        "req_pool_idx": int(req_pool_idx),
        "block_index": int(block_index),
        "block_size": int(canvas_length),
        "block_finalized": True,
        "expected_committed_response_prefix_length": int(block_index)
        * int(canvas_length),
        "final_target_ids": target_tensor,
        "target_valid_mask": target_valid_mask,
        "step_indices": step_tensor,
        "pre_forward_canvas_ids": canvas_tensor,
        "self_conditioning_signal": signal_tensor,
        "self_conditioning_present": present_tensor,
        "valid_masks": valid_masks,
        "done_masks": done_masks,
        "terminal_mask": terminal_mask,
        "row_valid_mask": row_valid_mask,
        "steps_used": int(steps_used),
        "captured_rows": int(rows),
        "captured_token_steps": int(rows) * int(canvas_length),
        "valid_token_count": int(canvas_length),
        "clean_filter_enabled": False,
        "active_mask_semantics": "all_valid_step_tokens",
        "capture_started_wall_ts": float(capture_started_wall_ts),
        "block_committed_wall_ts": committed_ts,
        "producer_capture_seconds": float(capture_seconds),
        "producer_tensor_bytes": int(tensor_bytes),
        "producer_peak_buffer_bytes": int(producer_peak_bytes),
        "timing": {
            "capture_seconds": float(capture_seconds),
        },
    }


@dataclasses.dataclass(frozen=True)
class GemmaReplayStepHandle:
    """Immutable producer snapshot for one batched denoise iteration."""

    capture_id: int
    sequence_index: int
    step_index: int
    req_pool_ids: tuple[int, ...]
    block_indices: tuple[int, ...]
    pre_forward_canvas_ids: torch.Tensor
    self_conditioning_signal: torch.Tensor
    self_conditioning_present: bool
    expected_self_conditioning_present: bool
    active_mask: torch.Tensor
    done_after_step: torch.Tensor
    cuda_event: Optional[Any] = None


@dataclasses.dataclass(frozen=True)
class GemmaReplayBlockHandle:
    """End-of-block snapshot used to validate and assemble replay payloads."""

    capture_id: int
    sequence_index: int
    req_pool_ids: tuple[int, ...]
    block_indices: tuple[int, ...]
    final_target_ids: torch.Tensor
    steps_used: torch.Tensor
    canvas_length: int
    capture_started_wall_ts: float
    capture_seconds: float
    block_committed_wall_ts: float
    cuda_event: Optional[Any] = None


@dataclasses.dataclass
class _RequestRows:
    step_indices: list[int] = dataclasses.field(default_factory=list)
    canvases: list[torch.Tensor] = dataclasses.field(default_factory=list)
    self_conditioning: list[torch.Tensor] = dataclasses.field(
        default_factory=list
    )
    self_conditioning_present: list[bool] = dataclasses.field(
        default_factory=list
    )
    done_after_step: list[bool] = dataclasses.field(default_factory=list)
    terminated: bool = False


@dataclasses.dataclass
class _CaptureState:
    req_pool_ids: tuple[int, ...]
    block_indices: tuple[int, ...]
    canvas_length: int
    rows: list[_RequestRows]
    next_sequence_index: int = 0
    previous_step_index: Optional[int] = None


@dataclasses.dataclass
class _QueueEntry:
    handle: Any
    retained_bytes: int

    @property
    def capture_id(self) -> int:
        return int(self.handle.capture_id)


@dataclasses.dataclass(frozen=True)
class _ErrorRecord:
    error_class: type[GemmaReplayCaptureError]
    message: str


def _default_event_factory(
    tensors: Sequence[torch.Tensor],
) -> Optional[torch.cuda.Event]:
    cuda_tensors = [
        tensor
        for tensor in tensors
        if isinstance(tensor, torch.Tensor) and tensor.is_cuda
    ]
    if not cuda_tensors:
        return None
    devices = {tensor.device for tensor in cuda_tensors}
    if len(devices) != 1:
        raise GemmaReplayCaptureIntegrityError(
            f"Gemma replay snapshots span CUDA devices: {sorted(map(str, devices))}"
        )
    device = cuda_tensors[0].device
    with torch.cuda.device(device):
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
    return event


def _default_event_waiter(event: Any) -> None:
    event.synchronize()


def _default_tensor_copier(
    tensors: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return {
        name: tensor.detach().to(device="cpu", copy=True)
        for name, tensor in tensors.items()
    }


class GemmaAsyncReplayCaptureWorker:
    """FIFO worker for bounded, lossless Gemma intermediate replay capture."""

    def __init__(
        self,
        *,
        queue_depth: int = 64,
        queue_max_bytes: int = 512 * 1024 * 1024,
        backpressure_policy: str = "error",
        enqueue_timeout_s: float = 0.0,
        event_factory: Optional[
            Callable[[Sequence[torch.Tensor]], Optional[Any]]
        ] = None,
        event_waiter: Optional[Callable[[Any], None]] = None,
        tensor_copier: Optional[
            Callable[
                [Dict[str, torch.Tensor]], Dict[str, torch.Tensor]
            ]
        ] = None,
        start_consumer: bool = True,
    ):
        if int(queue_depth) <= 0:
            raise ValueError("Gemma async replay queue_depth must be positive")
        if int(queue_max_bytes) <= 0:
            raise ValueError(
                "Gemma async replay queue_max_bytes must be positive"
            )
        if backpressure_policy not in {"error", "block"}:
            raise ValueError(
                "Gemma async replay backpressure_policy must be 'error' or "
                "'block'"
            )
        if float(enqueue_timeout_s) < 0:
            raise ValueError(
                "Gemma async replay enqueue_timeout_s must be nonnegative"
            )

        self._max_count = int(queue_depth)
        self._max_bytes = int(queue_max_bytes)
        self._backpressure_policy = str(backpressure_policy)
        self._enqueue_timeout_s = float(enqueue_timeout_s)
        self._event_factory = event_factory or _default_event_factory
        self._event_waiter = event_waiter or _default_event_waiter
        self._tensor_copier = tensor_copier or _default_tensor_copier

        self._entries: Deque[_QueueEntry] = deque()
        self._condition = threading.Condition()
        self._queued_bytes = 0
        self._pending_count = 0
        self._pending_bytes = 0
        self._pending_by_capture: dict[int, int] = {}
        self._pending_bytes_by_capture: dict[int, int] = {}
        self._states: dict[int, _CaptureState] = {}
        self._results: dict[int, list[dict]] = {}
        self._errors: dict[int, _ErrorRecord] = {}
        self._cancelled: set[int] = set()
        self._finalizing: set[int] = set()
        self._completed: set[int] = set()
        self._accepting = True
        self._stopping = False
        self._thread: Optional[threading.Thread] = None
        self._next_capture_id = 1
        self._metrics = {
            "enqueue_count": 0,
            "process_count": 0,
            "backpressure_count": 0,
            "consumer_errors": 0,
            "drop_count": 0,
            "queue_depth_max": 0,
            "queued_bytes_max": 0,
            "pending_depth_max": 0,
            "pending_bytes_max": 0,
        }
        if start_consumer:
            self.start()
        atexit.register(self.stop)

    def start(self) -> None:
        """Start the consumer once; exposed for deterministic CPU tests."""

        with self._condition:
            if self._thread is not None:
                if not self._thread.is_alive():
                    raise RuntimeError(
                        "Gemma async replay consumer terminated unexpectedly"
                    )
                return
            if self._stopping:
                raise RuntimeError("Gemma async replay worker is stopping")
            self._thread = threading.Thread(
                target=self._consumer_loop,
                name="gemma-async-replay-capture",
                daemon=True,
            )
            self._thread.start()

    def allocate_capture_id(self) -> int:
        with self._condition:
            capture_id = self._next_capture_id
            self._next_capture_id += 1
            return int(capture_id)

    def make_producer_event(
        self, *tensors: torch.Tensor
    ) -> Optional[Any]:
        """Record completion after immutable snapshots on the producer stream."""

        return self._event_factory(tuple(tensors))

    def _raise_record(self, capture_id: int, record: _ErrorRecord) -> None:
        raise record.error_class(
            f"Gemma async replay capture {capture_id} failed: "
            f"{record.message}"
        )

    def _enqueue(self, handle: Any) -> None:
        capture_id = int(handle.capture_id)
        retained_bytes = estimate_gemma_capture_handle_bytes(handle)
        if retained_bytes > self._max_bytes:
            with self._condition:
                self._metrics["backpressure_count"] += 1
            raise GemmaReplayCaptureBackpressureError(
                f"Gemma replay handle retains {retained_bytes} bytes, "
                f"exceeding queue_max_bytes={self._max_bytes}"
            )

        deadline = (
            time.monotonic() + self._enqueue_timeout_s
            if self._backpressure_policy == "block"
            else None
        )
        with self._condition:
            while True:
                error = self._errors.get(capture_id)
                if error is not None:
                    self._raise_record(capture_id, error)
                if capture_id in self._cancelled:
                    raise GemmaReplayCaptureError(
                        f"Gemma replay capture {capture_id} was cancelled"
                    )
                if not self._accepting:
                    raise GemmaReplayCaptureBackpressureError(
                        "Gemma async replay worker is stopping"
                    )
                if self._thread is not None and not self._thread.is_alive():
                    raise GemmaReplayCaptureConsumerError(
                        "Gemma async replay consumer is not alive"
                    )

                has_capacity = (
                    self._pending_count < self._max_count
                    and self._pending_bytes + retained_bytes <= self._max_bytes
                )
                if has_capacity:
                    break
                self._metrics["backpressure_count"] += 1
                if self._backpressure_policy == "error":
                    raise GemmaReplayCaptureBackpressureError(
                        "Gemma async replay queue budget exhausted: "
                        f"pending={self._pending_count}/{self._max_count} "
                        f"bytes={self._pending_bytes}/{self._max_bytes} "
                        f"incoming_bytes={retained_bytes}"
                    )
                remaining = float(deadline) - time.monotonic()
                if remaining <= 0:
                    raise GemmaReplayCaptureBackpressureError(
                        "Timed out waiting for Gemma async replay queue "
                        f"capacity: pending={self._pending_count}/"
                        f"{self._max_count} bytes={self._pending_bytes}/"
                        f"{self._max_bytes}"
                    )
                self._condition.wait(timeout=remaining)

            entry = _QueueEntry(handle=handle, retained_bytes=retained_bytes)
            self._entries.append(entry)
            self._queued_bytes += retained_bytes
            self._pending_count += 1
            self._pending_bytes += retained_bytes
            self._pending_by_capture[capture_id] = (
                self._pending_by_capture.get(capture_id, 0) + 1
            )
            self._pending_bytes_by_capture[capture_id] = (
                self._pending_bytes_by_capture.get(capture_id, 0)
                + retained_bytes
            )
            self._metrics["enqueue_count"] += 1
            self._metrics["queue_depth_max"] = max(
                self._metrics["queue_depth_max"], len(self._entries)
            )
            self._metrics["queued_bytes_max"] = max(
                self._metrics["queued_bytes_max"], self._queued_bytes
            )
            self._metrics["pending_depth_max"] = max(
                self._metrics["pending_depth_max"], self._pending_count
            )
            self._metrics["pending_bytes_max"] = max(
                self._metrics["pending_bytes_max"], self._pending_bytes
            )
            self._condition.notify()

    def enqueue_step(self, handle: GemmaReplayStepHandle) -> None:
        self._enqueue(handle)

    def enqueue_block(self, handle: GemmaReplayBlockHandle) -> None:
        self._enqueue(handle)

    def _release_entry_locked(self, entry: _QueueEntry) -> None:
        capture_id = entry.capture_id
        self._pending_count -= 1
        self._pending_bytes -= entry.retained_bytes
        remaining = self._pending_by_capture.get(capture_id, 0) - 1
        if remaining <= 0:
            self._pending_by_capture.pop(capture_id, None)
            self._pending_bytes_by_capture.pop(capture_id, None)
            if (
                capture_id in self._cancelled
                and capture_id not in self._errors
            ):
                self._cancelled.discard(capture_id)
        else:
            self._pending_by_capture[capture_id] = remaining
            self._pending_bytes_by_capture[capture_id] = max(
                0,
                self._pending_bytes_by_capture.get(capture_id, 0)
                - entry.retained_bytes,
            )
        self._condition.notify_all()

    def _purge_capture_locked(self, capture_id: int) -> None:
        retained: Deque[_QueueEntry] = deque()
        while self._entries:
            entry = self._entries.popleft()
            if entry.capture_id == capture_id:
                self._queued_bytes -= entry.retained_bytes
                self._release_entry_locked(entry)
                entry.handle = None
            else:
                retained.append(entry)
        self._entries = retained

    def _fail_capture_locked(self, capture_id: int, exc: Exception) -> None:
        if capture_id not in self._errors:
            if isinstance(exc, GemmaReplayCaptureError):
                error_class = type(exc)
                message = str(exc)
            else:
                error_class = GemmaReplayCaptureConsumerError
                message = f"{type(exc).__name__}: {exc}"
            self._errors[capture_id] = _ErrorRecord(
                error_class=error_class,
                message=message,
            )
            self._metrics["consumer_errors"] += 1
        self._cancelled.add(capture_id)
        self._states.pop(capture_id, None)
        self._results.pop(capture_id, None)
        self._finalizing.discard(capture_id)
        self._completed.discard(capture_id)
        self._purge_capture_locked(capture_id)

    def cancel_capture(self, capture_id: int) -> None:
        """Discard queued/assembled data after the producer has already failed."""

        capture_id = int(capture_id)
        with self._condition:
            self._cancelled.add(capture_id)
            self._states.pop(capture_id, None)
            self._results.pop(capture_id, None)
            self._finalizing.discard(capture_id)
            self._completed.discard(capture_id)
            self._errors.pop(capture_id, None)
            self._purge_capture_locked(capture_id)
            if self._pending_by_capture.get(capture_id, 0) == 0:
                self._cancelled.discard(capture_id)
            self._condition.notify_all()

    def _materialize(
        self,
        tensors: Dict[str, torch.Tensor],
        cuda_event: Optional[Any],
    ) -> Dict[str, torch.Tensor]:
        if cuda_event is not None:
            self._event_waiter(cuda_event)
        result = self._tensor_copier(tensors)
        if set(result) != set(tensors):
            raise GemmaReplayCaptureConsumerError(
                "Gemma replay tensor copier returned incomplete fields"
            )
        if not all(isinstance(value, torch.Tensor) for value in result.values()):
            raise GemmaReplayCaptureConsumerError(
                "Gemma replay tensor copier returned a non-tensor field"
            )
        return result

    @staticmethod
    def _validate_batch_identity(
        state: _CaptureState,
        req_pool_ids: tuple[int, ...],
        block_indices: tuple[int, ...],
    ) -> None:
        if state.req_pool_ids != req_pool_ids:
            raise GemmaReplayCaptureIntegrityError(
                "Gemma replay request ordering changed within a block: "
                f"{req_pool_ids} != {state.req_pool_ids}"
            )
        if state.block_indices != block_indices:
            raise GemmaReplayCaptureIntegrityError(
                "Gemma replay block indices changed during capture: "
                f"{block_indices} != {state.block_indices}"
            )

    def _process_step(self, handle: GemmaReplayStepHandle) -> None:
        cpu = self._materialize(
            {
                "pre_forward_canvas_ids": handle.pre_forward_canvas_ids,
                "self_conditioning_signal": handle.self_conditioning_signal,
                "active_mask": handle.active_mask,
                "done_after_step": handle.done_after_step,
            },
            handle.cuda_event,
        )
        canvases = cpu["pre_forward_canvas_ids"]
        signals = cpu["self_conditioning_signal"]
        active_mask = cpu["active_mask"].bool().reshape(-1)
        done_after = cpu["done_after_step"].bool().reshape(-1)
        req_pool_ids = tuple(int(value) for value in handle.req_pool_ids)
        block_indices = tuple(int(value) for value in handle.block_indices)
        batch_size = len(req_pool_ids)

        if len(set(req_pool_ids)) != batch_size:
            raise GemmaReplayCaptureIntegrityError(
                f"Gemma replay has duplicate req_pool_ids: {req_pool_ids}"
            )
        if len(block_indices) != batch_size:
            raise GemmaReplayCaptureIntegrityError(
                "Gemma replay request/block cardinality mismatch"
            )
        if canvases.ndim != 2 or int(canvases.shape[0]) != batch_size:
            raise GemmaReplayCaptureIntegrityError(
                "Gemma replay pre-forward canvas has invalid shape: "
                f"{tuple(canvases.shape)}"
            )
        canvas_length = int(canvases.shape[1])
        if signals.ndim == 2 and int(signals.shape[0]) == (
            batch_size * canvas_length
        ):
            signals = signals.view(batch_size, canvas_length, -1)
        if (
            signals.ndim != 3
            or tuple(signals.shape[:2]) != (batch_size, canvas_length)
        ):
            raise GemmaReplayCaptureIntegrityError(
                "Gemma replay self-conditioning has invalid shape: "
                f"{tuple(signals.shape)}"
            )
        if active_mask.numel() != batch_size or done_after.numel() != batch_size:
            raise GemmaReplayCaptureIntegrityError(
                "Gemma replay activity mask has invalid batch cardinality"
            )
        if (
            bool(handle.self_conditioning_present)
            != bool(handle.expected_self_conditioning_present)
        ):
            raise GemmaReplayCaptureIntegrityError(
                "DiffusionGemma replay self-conditioning presence mismatch: "
                f"step={handle.step_index} "
                f"consumed={handle.self_conditioning_present} "
                f"expected={handle.expected_self_conditioning_present}"
            )
        if int(handle.step_index) < 1:
            raise GemmaReplayCaptureOrderingError(
                f"Gemma replay has invalid step index {handle.step_index}"
            )

        active_values = [bool(value) for value in active_mask.tolist()]
        done_values = [bool(value) for value in done_after.tolist()]
        compact_rows: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for batch_id, active in enumerate(active_values):
            if active:
                compact_rows[batch_id] = (
                    canvases[batch_id].to(dtype=torch.long).clone(),
                    signals[batch_id].clone(),
                )

        capture_id = int(handle.capture_id)
        with self._condition:
            if capture_id in self._cancelled:
                return
            if capture_id in self._completed or capture_id in self._finalizing:
                raise GemmaReplayCaptureOrderingError(
                    f"Gemma replay row arrived after block {capture_id} finalized"
                )
            state = self._states.get(capture_id)
            if state is None:
                state = _CaptureState(
                    req_pool_ids=req_pool_ids,
                    block_indices=block_indices,
                    canvas_length=canvas_length,
                    rows=[_RequestRows() for _ in range(batch_size)],
                )
                self._states[capture_id] = state
            self._validate_batch_identity(state, req_pool_ids, block_indices)
            if state.canvas_length != canvas_length:
                raise GemmaReplayCaptureIntegrityError(
                    "Gemma replay canvas length changed during capture"
                )
            if int(handle.sequence_index) != state.next_sequence_index:
                raise GemmaReplayCaptureOrderingError(
                    "Gemma replay duplicate/out-of-order sequence: "
                    f"received={handle.sequence_index} "
                    f"expected={state.next_sequence_index}"
                )
            if (
                state.previous_step_index is not None
                and int(handle.step_index) != state.previous_step_index - 1
            ):
                raise GemmaReplayCaptureOrderingError(
                    "Gemma replay denoise steps are not strictly adjacent and "
                    "descending: "
                    f"received={handle.step_index} "
                    f"after={state.previous_step_index}"
                )

            for batch_id, rows in enumerate(state.rows):
                active = active_values[batch_id]
                is_done = done_values[batch_id]
                if rows.terminated and active:
                    raise GemmaReplayCaptureOrderingError(
                        "Gemma replay request became active after its terminal "
                        f"row: req_pool_idx={req_pool_ids[batch_id]}"
                    )
                if not active:
                    if not is_done:
                        raise GemmaReplayCaptureIntegrityError(
                            "Gemma replay inactive request is not marked done: "
                            f"req_pool_idx={req_pool_ids[batch_id]}"
                        )
                    continue
                if (
                    rows.step_indices
                    and int(handle.step_index) != rows.step_indices[-1] - 1
                ):
                    raise GemmaReplayCaptureOrderingError(
                        "Gemma replay per-request rows are duplicate or "
                        "out-of-order: "
                        f"req_pool_idx={req_pool_ids[batch_id]} "
                        f"step={handle.step_index}"
                    )
                canvas_row, signal_row = compact_rows[batch_id]
                rows.step_indices.append(int(handle.step_index))
                rows.canvases.append(canvas_row)
                rows.self_conditioning.append(signal_row)
                rows.self_conditioning_present.append(
                    bool(handle.self_conditioning_present)
                )
                rows.done_after_step.append(is_done)
                rows.terminated = is_done

            state.next_sequence_index += 1
            state.previous_step_index = int(handle.step_index)

    def _process_block(self, handle: GemmaReplayBlockHandle) -> None:
        cpu = self._materialize(
            {
                "final_target_ids": handle.final_target_ids,
                "steps_used": handle.steps_used,
            },
            handle.cuda_event,
        )
        targets = cpu["final_target_ids"]
        steps_used_tensor = cpu["steps_used"].long().reshape(-1)
        req_pool_ids = tuple(int(value) for value in handle.req_pool_ids)
        block_indices = tuple(int(value) for value in handle.block_indices)
        batch_size = len(req_pool_ids)
        if (
            targets.ndim != 2
            or tuple(targets.shape)
            != (batch_size, int(handle.canvas_length))
        ):
            raise GemmaReplayCaptureIntegrityError(
                "Gemma replay final target has invalid shape: "
                f"{tuple(targets.shape)}"
            )
        if steps_used_tensor.numel() != batch_size:
            raise GemmaReplayCaptureIntegrityError(
                "Gemma replay steps_used has invalid batch cardinality"
            )
        steps_used = [int(value) for value in steps_used_tensor.tolist()]

        capture_id = int(handle.capture_id)
        with self._condition:
            if capture_id in self._cancelled:
                return
            if capture_id in self._completed or capture_id in self._finalizing:
                raise GemmaReplayCaptureDuplicateBlockError(
                    f"Gemma replay block {capture_id} finalized more than once"
                )
            state = self._states.pop(capture_id, None)
            if state is None:
                raise GemmaReplayCaptureMissingRowError(
                    f"Gemma replay block {capture_id} has no captured rows"
                )
            self._validate_batch_identity(state, req_pool_ids, block_indices)
            if state.canvas_length != int(handle.canvas_length):
                raise GemmaReplayCaptureIntegrityError(
                    "Gemma replay final canvas length does not match rows"
                )
            if int(handle.sequence_index) != state.next_sequence_index:
                raise GemmaReplayCaptureOrderingError(
                    "Gemma replay block finalized out of sequence: "
                    f"received={handle.sequence_index} "
                    f"expected={state.next_sequence_index}"
                )
            for batch_id, expected_rows in enumerate(steps_used):
                captured_rows = len(state.rows[batch_id].step_indices)
                if captured_rows != expected_rows:
                    raise GemmaReplayCaptureMissingRowError(
                        "DiffusionGemma replay row/steps parity failed: "
                        f"req_pool_idx={req_pool_ids[batch_id]} "
                        f"captured={captured_rows} steps_used={expected_rows}"
                    )
            self._finalizing.add(capture_id)

        try:
            replay_items = []
            for batch_id, rows in enumerate(state.rows):
                capture_buffer_bytes = sum(
                    _tensor_bytes(canvas) + _tensor_bytes(signal)
                    for canvas, signal in zip(
                        rows.canvases, rows.self_conditioning
                    )
                )
                replay_items.append(
                    build_gemma_intermediate_replay_item(
                        canvas_length=int(handle.canvas_length),
                        req_pool_idx=req_pool_ids[batch_id],
                        block_index=block_indices[batch_id],
                        target_ids=targets[batch_id],
                        step_indices=rows.step_indices,
                        canvases=rows.canvases,
                        self_conditioning=rows.self_conditioning,
                        self_conditioning_present=(
                            rows.self_conditioning_present
                        ),
                        done_after_step=rows.done_after_step,
                        steps_used=steps_used[batch_id],
                        capture_started_wall_ts=(
                            handle.capture_started_wall_ts
                        ),
                        capture_seconds=handle.capture_seconds,
                        capture_buffer_bytes=capture_buffer_bytes,
                        block_committed_wall_ts=(
                            handle.block_committed_wall_ts
                        ),
                    )
                )
        except Exception:
            with self._condition:
                self._finalizing.discard(capture_id)
            raise

        with self._condition:
            self._finalizing.discard(capture_id)
            if capture_id in self._cancelled:
                return
            self._results[capture_id] = replay_items
            self._completed.add(capture_id)
            self._condition.notify_all()

    def _consumer_loop(self) -> None:
        while True:
            with self._condition:
                while not self._entries and not self._stopping:
                    self._condition.wait(timeout=0.5)
                if not self._entries:
                    if self._stopping:
                        return
                    continue
                entry = self._entries.popleft()
                self._queued_bytes -= entry.retained_bytes
                capture_id = entry.capture_id

            handle = entry.handle
            try:
                with self._condition:
                    cancelled = capture_id in self._cancelled
                if cancelled:
                    continue
                if isinstance(handle, GemmaReplayStepHandle):
                    self._process_step(handle)
                elif isinstance(handle, GemmaReplayBlockHandle):
                    self._process_block(handle)
                else:
                    raise GemmaReplayCaptureConsumerError(
                        f"Unsupported Gemma replay handle: {type(handle)!r}"
                    )
                with self._condition:
                    self._metrics["process_count"] += 1
            except Exception as exc:
                with self._condition:
                    self._fail_capture_locked(capture_id, exc)
                logger.error(
                    "Gemma async replay consumer failed capture %d: %s: %s",
                    capture_id,
                    type(exc).__name__,
                    exc,
                )
            finally:
                handle = None
                with self._condition:
                    self._release_entry_locked(entry)
                entry.handle = None

    def drain_capture(
        self, capture_id: int, *, timeout: float
    ) -> list[dict]:
        """Wait for one block and return only a complete validated payload."""

        capture_id = int(capture_id)
        if float(timeout) < 0:
            raise ValueError("Gemma replay drain timeout must be nonnegative")
        deadline = time.monotonic() + float(timeout)
        with self._condition:
            while self._pending_by_capture.get(capture_id, 0) > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    pending = self._pending_by_capture.get(capture_id, 0)
                    pending_bytes = self._pending_bytes_by_capture.get(
                        capture_id, 0
                    )
                    self._cancelled.add(capture_id)
                    self._states.pop(capture_id, None)
                    self._results.pop(capture_id, None)
                    self._finalizing.discard(capture_id)
                    self._completed.discard(capture_id)
                    self._purge_capture_locked(capture_id)
                    raise GemmaReplayCaptureDrainTimeout(
                        f"Gemma replay capture {capture_id} still has "
                        f"{pending} handles / {pending_bytes} bytes pending"
                    )
                self._condition.wait(timeout=remaining)

            error = self._errors.pop(capture_id, None)
            if error is not None:
                self._cancelled.discard(capture_id)
                self._states.pop(capture_id, None)
                self._results.pop(capture_id, None)
                self._finalizing.discard(capture_id)
                self._completed.discard(capture_id)
                self._raise_record(capture_id, error)

            result = self._results.pop(capture_id, None)
            if result is None:
                self._states.pop(capture_id, None)
                self._cancelled.discard(capture_id)
                self._finalizing.discard(capture_id)
                self._completed.discard(capture_id)
                raise GemmaReplayCaptureMissingBlockError(
                    f"Gemma replay capture {capture_id} drained without a "
                    "block-finalization payload"
                )
            self._completed.discard(capture_id)
            self._cancelled.discard(capture_id)
            return result

    def get_metrics(self) -> dict:
        with self._condition:
            return {
                **self._metrics,
                "queued_depth": len(self._entries),
                "queued_bytes": self._queued_bytes,
                "pending_depth": self._pending_count,
                "pending_bytes": self._pending_bytes,
                "queue_max_count": self._max_count,
                "queue_max_bytes": self._max_bytes,
                "backpressure_policy": self._backpressure_policy,
            }

    def stop(
        self, *, drain: bool = False, join_timeout: float = 5.0
    ) -> bool:
        try:
            atexit.unregister(self.stop)
        except Exception:
            pass

        deadline = time.monotonic() + max(float(join_timeout), 0.0)
        with self._condition:
            if self._stopping:
                thread = self._thread
            else:
                self._accepting = False
                if drain:
                    while self._pending_count > 0:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        self._condition.wait(timeout=remaining)
                capture_ids = set(self._pending_by_capture)
                for capture_id in capture_ids:
                    self._cancelled.add(capture_id)
                    self._purge_capture_locked(capture_id)
                self._states.clear()
                self._results.clear()
                self._errors.clear()
                self._finalizing.clear()
                self._completed.clear()
                self._stopping = True
                self._condition.notify_all()
                thread = self._thread

        if thread is not None:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        return thread is None or not thread.is_alive()
