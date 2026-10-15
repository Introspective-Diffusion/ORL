"""Same-host CUDA IPC leases for GPU-resident SDAR replay tensors."""

from __future__ import annotations

import fcntl
import gc
import logging
import os
import socket
import threading
import time
import uuid
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any

import numpy as np
import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor


logger = logging.getLogger(__name__)

GPU_REPLAY_LEASE_FORMAT = "sglang_sdar_gpu_replay_lease_v1"
GPU_REPLAY_LEASE_FANOUT_FORMAT = "sglang_sdar_gpu_replay_lease_fanout_v1"
_ACK_LOCK_PATH = "/tmp/sglang_sdar_gpu_replay_ack.lock"
_DTYPES = {
    str(dtype): dtype
    for dtype in (
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    )
}


def is_gpu_replay_lease(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("format") == GPU_REPLAY_LEASE_FORMAT
    )


def is_gpu_replay_lease_fanout(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("format") == GPU_REPLAY_LEASE_FANOUT_FORMAT
        and isinstance(value.get("descriptors"), list)
    )


def select_gpu_replay_lease_for_device(
    value: dict[str, Any],
    *,
    device_uuid: str,
) -> dict[str, Any]:
    """Select the one-shot lease relayed to a specific actor GPU."""
    if is_gpu_replay_lease(value):
        if str(value.get("device_uuid")) != str(device_uuid):
            raise RuntimeError(
                "GPU replay lease is not local to the actor device: "
                f"lease={value.get('device_uuid')} actor={device_uuid}"
            )
        return value
    if not is_gpu_replay_lease_fanout(value):
        raise TypeError("expected an SDAR GPU replay lease or fan-out descriptor")
    matches = [
        descriptor
        for descriptor in value["descriptors"]
        if str(descriptor.get("device_uuid")) == str(device_uuid)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "GPU replay lease fan-out must contain exactly one actor-local "
            f"descriptor, got {len(matches)} for uuid={device_uuid}"
        )
    return matches[0]


def slice_gpu_replay_lease(
    descriptor: dict[str, Any],
    start: int,
    length: int,
) -> dict[str, Any]:
    """Return metadata for a first-dimension view of one leased tensor."""
    if not is_gpu_replay_lease(descriptor):
        raise TypeError("expected an SDAR GPU replay lease descriptor")
    shape = [int(value) for value in descriptor["shape"]]
    stride = [int(value) for value in descriptor["stride"]]
    start = int(start)
    length = int(length)
    if not shape or start < 0 or length < 0 or start + length > shape[0]:
        raise ValueError(
            f"invalid lease slice start={start} length={length} shape={shape}"
        )
    sliced = dict(descriptor)
    sliced["shape"] = [length, *shape[1:]]
    sliced["tensor_storage_offset_elems"] = int(
        descriptor["tensor_storage_offset_elems"]
    ) + start * stride[0]
    sliced["logical_slice_start"] = int(
        descriptor.get("logical_slice_start", 0)
    ) + start
    return sliced


@dataclass
class _ArenaChunk:
    storage: torch.Tensor
    capacity_bytes: int
    offset_bytes: int = 0
    active_leases: int = 0


@dataclass
class _ProducerLease:
    tensor: torch.Tensor
    producer_event: torch.cuda.Event
    ack: shared_memory.SharedMemory
    ack_view: np.ndarray
    expected_consumers: int
    nbytes: int
    created_at: float
    device_index: int
    arena_chunk: _ArenaChunk | None = None


class GpuReplayLeaseTable:
    """Own GPU tensors until every CUDA IPC consumer acknowledges completion."""

    def __init__(
        self,
        *,
        stale_warning_seconds: float = 300.0,
        recycle_interval_seconds: float = 0.25,
        max_bytes: int = 16 * 1024**3,
        trim_recycled_cache: bool | None = None,
        trim_min_bytes: int = 256 * 1024**2,
    ):
        self._stale_warning_seconds = max(
            float(stale_warning_seconds),
            1.0,
        )
        self._recycle_interval_seconds = max(
            float(recycle_interval_seconds),
            0.01,
        )
        self._max_bytes = max(int(max_bytes), 0)
        if trim_recycled_cache is None:
            trim_recycled_cache = str(
                os.environ.get(
                    "SGLANG_SDAR_GPU_REPLAY_TRIM_RECYCLED_CACHE",
                    "0",
                )
            ).strip().lower() in {"1", "true", "yes", "on"}
        self._trim_recycled_cache = bool(trim_recycled_cache)
        self._trim_min_bytes = max(int(trim_min_bytes), 0)
        self._arena_enabled = str(
            os.environ.get(
                "SGLANG_SDAR_GPU_REPLAY_ARENA",
                "1",
            )
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._arena_chunk_bytes = max(
            int(
                os.environ.get(
                    "SGLANG_SDAR_GPU_REPLAY_ARENA_CHUNK_BYTES",
                    str(256 * 1024**2),
                )
            ),
            1,
        )
        self._arena_chunks: dict[int, list[_ArenaChunk]] = {}
        self._recycled_bytes_since_trim = 0
        self._trim_count = 0
        self._leases: dict[str, _ProducerLease] = {}
        self._lock = threading.Lock()
        self._closed = False
        self._thread = threading.Thread(
            target=self._recycle_loop,
            name="sdar-gpu-replay-lease-recycler",
            daemon=True,
        )
        self._thread.start()

    @staticmethod
    def _align_bytes(value: int, alignment: int = 256) -> int:
        alignment = max(int(alignment), 1)
        return (
            (int(value) + alignment - 1)
            // alignment
            * alignment
        )

    def _stage_in_arena(
        self,
        tensor: torch.Tensor,
        *,
        tensor_nbytes: int,
    ) -> tuple[torch.Tensor, _ArenaChunk | None]:
        if not self._arena_enabled:
            return tensor.detach(), None
        device_index = int(
            tensor.device.index
            if tensor.device.index is not None
            else torch.cuda.current_device()
        )
        alignment = max(256, int(tensor.element_size()))
        with self._lock:
            chunks = self._arena_chunks.setdefault(
                device_index,
                [],
            )
            selected = None
            selected_offset = 0
            for chunk in chunks:
                offset = self._align_bytes(
                    chunk.offset_bytes,
                    alignment,
                )
                if offset + tensor_nbytes <= chunk.capacity_bytes:
                    selected = chunk
                    selected_offset = offset
                    break
            if selected is None:
                arena_bytes = sum(
                    chunk.capacity_bytes
                    for device_chunks in self._arena_chunks.values()
                    for chunk in device_chunks
                )
                remaining = (
                    self._max_bytes - arena_bytes
                    if self._max_bytes > 0
                    else self._arena_chunk_bytes
                )
                chunk_bytes = max(
                    self._arena_chunk_bytes,
                    self._align_bytes(tensor_nbytes, alignment),
                )
                if self._max_bytes > 0:
                    chunk_bytes = min(chunk_bytes, remaining)
                if chunk_bytes < tensor_nbytes:
                    raise RuntimeError(
                        "SDAR GPU replay arena exceeded its byte budget: "
                        f"arena={arena_bytes} requested={tensor_nbytes} "
                        f"limit={self._max_bytes}"
                    )
                with torch.cuda.device(device_index):
                    storage = torch.empty(
                        chunk_bytes,
                        dtype=torch.uint8,
                        device=device_index,
                    )
                selected = _ArenaChunk(
                    storage=storage,
                    capacity_bytes=chunk_bytes,
                )
                chunks.append(selected)
                selected_offset = 0
            selected.offset_bytes = (
                selected_offset + tensor_nbytes
            )
            selected.active_leases += 1
            storage_slice = selected.storage.narrow(
                0,
                selected_offset,
                tensor_nbytes,
            )
        staged = storage_slice.view(tensor.dtype).view(
            tensor.shape
        )
        staged.copy_(tensor)
        return staged, selected

    def _release_arena_reservation(
        self,
        chunk: _ArenaChunk | None,
    ) -> None:
        if chunk is None:
            return
        with self._lock:
            chunk.active_leases -= 1
            if chunk.active_leases < 0:
                raise RuntimeError(
                    "SDAR GPU replay arena lease count underflow"
                )
            if chunk.active_leases == 0:
                chunk.offset_bytes = 0

    def export(
        self,
        tensor: torch.Tensor,
        *,
        logical_name: str,
        expected_consumers: int = 1,
    ) -> dict[str, Any]:
        if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
            raise TypeError("GPU replay leases require a CUDA tensor")
        if tensor.requires_grad:
            raise ValueError("GPU replay lease tensors must be detached")
        expected_consumers = int(expected_consumers)
        if expected_consumers != 1:
            raise ValueError(
                "first-version GPU replay leases are one-shot; "
                "export a fresh descriptor per consumer"
            )
        tensor = tensor.detach()
        tensor_nbytes = int(
            tensor.numel() * tensor.element_size()
        )
        with self._lock:
            active_bytes = sum(
                active.nbytes for active in self._leases.values()
            )
            if (
                self._max_bytes > 0
                and active_bytes + tensor_nbytes > self._max_bytes
            ):
                raise RuntimeError(
                    "SDAR GPU replay lease table exceeded its byte budget: "
                    f"active={active_bytes} requested={tensor_nbytes} "
                    f"limit={self._max_bytes}"
                )
        arena_chunk = None
        ack = None
        try:
            tensor, arena_chunk = self._stage_in_arena(
                tensor,
                tensor_nbytes=tensor_nbytes,
            )
            producer_event = torch.cuda.Event(interprocess=True)
            producer_event.record(
                torch.cuda.current_stream(tensor.device)
            )

            storage = tensor._typed_storage()
            (
                _storage_device,
                storage_handle,
                storage_size_bytes,
                storage_offset_bytes,
                ref_counter_handle,
                ref_counter_offset,
                storage_event_handle,
                storage_event_sync_required,
            ) = storage._share_cuda_()

            ack = shared_memory.SharedMemory(
                create=True,
                size=np.dtype(np.int64).itemsize,
            )
            ack_view = np.ndarray(
                (1,),
                dtype=np.int64,
                buffer=ack.buf,
            )
            ack_view[0] = 0
            lease_id = uuid.uuid4().hex
            lease = _ProducerLease(
                tensor=tensor,
                producer_event=producer_event,
                ack=ack,
                ack_view=ack_view,
                expected_consumers=expected_consumers,
                nbytes=tensor_nbytes,
                created_at=time.monotonic(),
                device_index=int(
                    tensor.device.index
                    if tensor.device.index is not None
                    else torch.cuda.current_device()
                ),
                arena_chunk=arena_chunk,
            )
            with self._lock:
                if self._closed:
                    raise RuntimeError(
                        "GPU replay lease table is closed"
                    )
                self._leases[lease_id] = lease
        except Exception:
            if ack is not None:
                ack.close()
                try:
                    ack.unlink()
                except FileNotFoundError:
                    pass
            self._release_arena_reservation(arena_chunk)
            raise

        return {
            "format": GPU_REPLAY_LEASE_FORMAT,
            "version": 1,
            "lease_id": lease_id,
            "logical_name": str(logical_name),
            "producer_host": socket.gethostname(),
            "producer_pid": int(os.getpid()),
            "torch_version": str(torch.__version__),
            "cuda_version": str(torch.version.cuda),
            "device_uuid": str(
                torch.cuda.get_device_properties(tensor.device.index).uuid
            ),
            "shape": [int(value) for value in tensor.shape],
            "stride": [int(value) for value in tensor.stride()],
            "tensor_storage_offset_elems": int(tensor.storage_offset()),
            "dtype": str(tensor.dtype),
            "requires_grad": False,
            "storage_handle": storage_handle,
            "storage_size_bytes": int(storage_size_bytes),
            "storage_offset_bytes": int(storage_offset_bytes),
            "ref_counter_handle": ref_counter_handle,
            "ref_counter_offset": int(ref_counter_offset),
            "event_handle": storage_event_handle,
            "event_sync_required": bool(storage_event_sync_required),
            "producer_event_handle": producer_event.ipc_handle(),
            "ack_shm_name": ack.name,
            "expected_consumers": expected_consumers,
            "nbytes": lease.nbytes,
            "logical_slice_start": 0,
        }

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "active_leases": len(self._leases),
                "active_bytes": sum(
                    lease.nbytes for lease in self._leases.values()
                ),
                "recycled_bytes_since_trim": int(
                    self._recycled_bytes_since_trim
                ),
                "trim_count": int(self._trim_count),
                "arena_chunks": sum(
                    len(chunks)
                    for chunks in self._arena_chunks.values()
                ),
                "arena_bytes": sum(
                    chunk.capacity_bytes
                    for chunks in self._arena_chunks.values()
                    for chunk in chunks
                ),
            }

    def recycle_acknowledged(self) -> None:
        """Synchronously release leases whose acknowledgements are complete."""

        self._recycle_once()

    def force_recycle_all(self) -> dict[str, int]:
        """Release all leases at a caller-proven quiescent boundary."""
        arena_chunks_to_release: dict[int, list[_ArenaChunk]] = {}
        with self._lock:
            leases = list(self._leases.values())
            self._leases.clear()
            if self._trim_recycled_cache:
                arena_chunks_to_release = self._arena_chunks
                self._arena_chunks = {}
                self._recycled_bytes_since_trim = 0
                self._trim_count += 1
            else:
                for chunks in self._arena_chunks.values():
                    for chunk in chunks:
                        chunk.active_leases = 0
                        chunk.offset_bytes = 0
        released_bytes = sum(lease.nbytes for lease in leases)
        released_arena_bytes = sum(
            chunk.capacity_bytes
            for chunks in arena_chunks_to_release.values()
            for chunk in chunks
        )
        device_indices = {
            int(lease.device_index) for lease in leases
        }
        device_indices.update(arena_chunks_to_release)
        for lease in leases:
            lease.tensor = None
            lease.producer_event = None
            lease.arena_chunk = None
            lease.ack.close()
            try:
                lease.ack.unlink()
            except FileNotFoundError:
                pass
        arena_chunks_to_release.clear()
        for device_index in sorted(device_indices):
            with torch.cuda.device(device_index):
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()
        return {
            "released_leases": len(leases),
            "released_bytes": int(released_bytes),
            "released_arena_bytes": int(released_arena_bytes),
        }

    def close(self) -> None:
        self._closed = True
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _recycle_loop(self) -> None:
        while not self._closed:
            self._recycle_once()
            time.sleep(self._recycle_interval_seconds)

    def _recycle_once(self) -> None:
        now = time.monotonic()
        completed: list[str] = []
        with self._lock:
            for lease_id, lease in self._leases.items():
                if int(lease.ack_view[0]) >= lease.expected_consumers:
                    completed.append(lease_id)
                elif now - lease.created_at >= self._stale_warning_seconds:
                    logger.warning(
                        "SDAR GPU replay lease is still active after %.1fs: "
                        "lease=%s bytes=%d acknowledgements=%d/%d",
                        now - lease.created_at,
                        lease_id,
                        lease.nbytes,
                        int(lease.ack_view[0]),
                        lease.expected_consumers,
                    )
            leases = [
                self._leases.pop(lease_id)
                for lease_id in completed
            ]
            for lease in leases:
                chunk = lease.arena_chunk
                if chunk is None:
                    continue
                chunk.active_leases -= 1
                if chunk.active_leases < 0:
                    raise RuntimeError(
                        "SDAR GPU replay arena lease count underflow"
                    )
                if chunk.active_leases == 0:
                    chunk.offset_bytes = 0
            has_active_leases = bool(self._leases)
        recycled_devices = {
            int(lease.device_index)
            for lease in leases
        }
        recycled_bytes = sum(
            int(lease.nbytes)
            for lease in leases
        )
        for lease in leases:
            lease.tensor = None
            lease.producer_event = None
            lease.ack.close()
            try:
                lease.ack.unlink()
            except FileNotFoundError:
                pass
        if leases:
            self._recycled_bytes_since_trim += recycled_bytes
            should_trim = (
                self._trim_recycled_cache
                and (
                    not has_active_leases
                    or self._recycled_bytes_since_trim
                    >= self._trim_min_bytes
                )
            )
            for device_index in sorted(recycled_devices):
                with torch.cuda.device(device_index):
                    torch.cuda.ipc_collect()
                    if should_trim:
                        torch.cuda.empty_cache()
            if should_trim:
                self._recycled_bytes_since_trim = 0
                self._trim_count += 1


class ImportedGpuReplayLease:
    """A mapped CUDA tensor plus an explicit producer-release acknowledgement."""

    def __init__(
        self,
        tensor: torch.Tensor,
        descriptor: dict[str, Any],
    ):
        self.tensor = tensor
        self.descriptor = descriptor
        self._released = False

    def release(
        self,
        *,
        synchronize: bool = True,
        collect_garbage: bool = True,
    ) -> None:
        if self._released:
            return
        if (
            synchronize
            and self.tensor is not None
            and self.tensor.is_cuda
        ):
            torch.cuda.current_stream(self.tensor.device).synchronize()
        self.tensor = None
        if collect_garbage:
            gc.collect()
        acknowledge_gpu_replay_lease(self.descriptor)
        self._released = True

    def __enter__(self) -> torch.Tensor:
        return self.tensor

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()


def acknowledge_gpu_replay_lease(
    descriptor: dict[str, Any],
) -> None:
    """Acknowledge a descriptor that was consumed or deliberately discarded."""
    if not is_gpu_replay_lease(descriptor):
        raise TypeError("expected an SDAR GPU replay lease descriptor")
    ack = shared_memory.SharedMemory(
        name=str(descriptor["ack_shm_name"])
    )
    try:
        ack_view = np.ndarray((1,), dtype=np.int64, buffer=ack.buf)
        open(_ACK_LOCK_PATH, "a").close()
        with open(_ACK_LOCK_PATH, "r+") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            ack_view[0] += 1
            fcntl.flock(lock_file, fcntl.LOCK_UN)
    finally:
        ack.close()


def import_gpu_replay_lease(
    descriptor: dict[str, Any],
) -> ImportedGpuReplayLease:
    if not is_gpu_replay_lease(descriptor):
        raise TypeError("expected an SDAR GPU replay lease descriptor")
    if socket.gethostname() != str(descriptor["producer_host"]):
        raise RuntimeError("CUDA IPC replay leases cannot cross hosts")
    if str(torch.__version__) != str(descriptor["torch_version"]):
        raise RuntimeError(
            "CUDA IPC replay lease PyTorch version mismatch: "
            f"{torch.__version__} != {descriptor['torch_version']}"
        )
    dtype = _DTYPES.get(str(descriptor["dtype"]))
    if dtype is None:
        raise TypeError(
            f"unsupported CUDA IPC replay dtype {descriptor['dtype']}"
        )

    device_index = next(
        (
            index
            for index in range(torch.cuda.device_count())
            if str(torch.cuda.get_device_properties(index).uuid)
            == str(descriptor["device_uuid"])
        ),
        None,
    )
    if device_index is None:
        raise RuntimeError(
            "producer GPU is not visible in the replay actor process: "
            f"uuid={descriptor['device_uuid']}"
        )

    with torch.cuda.device(device_index):
        producer_event = torch.cuda.Event.from_ipc_handle(
            device_index,
            descriptor["producer_event_handle"],
        )
        torch.cuda.current_stream(device_index).wait_event(producer_event)
        tensor = rebuild_cuda_tensor(
            torch.Tensor,
            tuple(int(value) for value in descriptor["shape"]),
            tuple(int(value) for value in descriptor["stride"]),
            int(descriptor["tensor_storage_offset_elems"]),
            torch.storage.TypedStorage,
            dtype,
            device_index,
            descriptor["storage_handle"],
            int(descriptor["storage_size_bytes"]),
            int(descriptor["storage_offset_bytes"]),
            False,
            descriptor["ref_counter_handle"],
            int(descriptor["ref_counter_offset"]),
            descriptor["event_handle"],
            bool(descriptor["event_sync_required"]),
        )
    return ImportedGpuReplayLease(tensor, descriptor)


def relay_gpu_replay_lease(
    descriptor: dict[str, Any],
    *,
    target_device_index: int,
    target_table: GpuReplayLeaseTable,
) -> dict[str, Any]:
    """Move one lease GPU-to-GPU and export a target-local one-shot lease."""
    target_device_index = int(target_device_index)
    target_uuid = str(
        torch.cuda.get_device_properties(target_device_index).uuid
    )
    if str(descriptor.get("device_uuid")) == target_uuid:
        relayed = dict(descriptor)
        relayed["producer_replica_rank"] = target_device_index
        return relayed

    source = import_gpu_replay_lease(descriptor)
    try:
        with torch.cuda.device(target_device_index):
            target = source.tensor.to(
                device=target_device_index,
                non_blocking=True,
                copy=True,
            ).contiguous()
            torch.cuda.current_stream(target_device_index).synchronize()
            relayed = target_table.export(
                target,
                logical_name=(
                    f"relay:{descriptor.get('logical_name', '')}"
                ),
            )
        relayed["producer_replica_rank"] = target_device_index
        return relayed
    finally:
        source.release(
            synchronize=False,
            collect_garbage=False,
        )


def relay_gpu_replay_lease_fanout(
    descriptor: dict[str, Any],
    *,
    target_device_indices: list[int],
    target_table: GpuReplayLeaseTable,
) -> dict[str, Any]:
    """Relay one source lease to one one-shot lease per SP actor GPU."""
    target_device_indices = list(
        dict.fromkeys(int(index) for index in target_device_indices)
    )
    if not target_device_indices:
        raise ValueError("GPU replay lease fan-out requires a target device")
    source = import_gpu_replay_lease(descriptor)
    outputs: list[dict[str, Any]] = []
    try:
        for target_device_index in target_device_indices:
            with torch.cuda.device(target_device_index):
                target = source.tensor.to(
                    device=target_device_index,
                    non_blocking=True,
                    copy=True,
                ).contiguous()
                torch.cuda.current_stream(target_device_index).synchronize()
                relayed = target_table.export(
                    target,
                    logical_name=(
                        f"fanout:{target_device_index}:"
                        f"{descriptor.get('logical_name', '')}"
                    ),
                )
            relayed["producer_replica_rank"] = target_device_index
            outputs.append(relayed)
        return {
            "format": GPU_REPLAY_LEASE_FANOUT_FORMAT,
            "version": 1,
            "source_lease_id": descriptor.get("lease_id"),
            "descriptors": outputs,
            "nbytes": sum(
                int(output.get("nbytes", 0) or 0)
                for output in outputs
            ),
        }
    finally:
        source.release(
            synchronize=False,
            collect_garbage=False,
        )
