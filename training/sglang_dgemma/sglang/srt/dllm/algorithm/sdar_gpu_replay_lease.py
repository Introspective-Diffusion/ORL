"""Compatibility surface for adjacent-branch CUDA replay descriptors.

The DiffusionGemma baseline transports final-response replay tensors through
Ray and never creates same-host CUDA-IPC leases.  Adjacent's shared VERL
materializer still imports this module eagerly, so descriptor-only helpers are
provided here while producer/consumer CUDA-IPC operations fail closed.
"""

from __future__ import annotations

import fcntl
import os
from multiprocessing import shared_memory
from typing import Any

import numpy as np


GPU_REPLAY_LEASE_FORMAT = "sglang_sdar_gpu_replay_lease_v1"
GPU_REPLAY_LEASE_FANOUT_FORMAT = "sglang_sdar_gpu_replay_lease_fanout_v1"
# One lock per user and host, shared by every process of a run; a fixed
# name would collide with (and be unwritable for) other users on the host.
_ACK_LOCK_PATH = os.environ.get(
    "SGLANG_SDAR_GPU_REPLAY_ACK_LOCK", f"/tmp/sglang_sdar_gpu_replay_ack_{os.getuid()}.lock"
)


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


def acknowledge_gpu_replay_lease(descriptor: dict[str, Any]) -> None:
    if not is_gpu_replay_lease(descriptor):
        raise TypeError("expected an SDAR GPU replay lease descriptor")
    ack = shared_memory.SharedMemory(name=str(descriptor["ack_shm_name"]))
    try:
        ack_view = np.ndarray((1,), dtype=np.int64, buffer=ack.buf)
        open(_ACK_LOCK_PATH, "a").close()
        with open(_ACK_LOCK_PATH, "r+") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            ack_view[0] += 1
            fcntl.flock(lock_file, fcntl.LOCK_UN)
    finally:
        ack.close()


def _unsupported() -> RuntimeError:
    return RuntimeError(
        "same-host SDAR CUDA-IPC replay leases are unavailable in the "
        "DiffusionGemma cross-node tensor-replay runtime"
    )


class GpuReplayLeaseTable:
    def __init__(self, *args, **kwargs):
        del args, kwargs
        raise _unsupported()


def import_gpu_replay_lease(descriptor: dict[str, Any]):
    del descriptor
    raise _unsupported()


def relay_gpu_replay_lease(
    descriptor: dict[str, Any],
    *,
    target_device_index: int,
    target_table: GpuReplayLeaseTable,
) -> dict[str, Any]:
    del descriptor, target_device_index, target_table
    raise _unsupported()


def relay_gpu_replay_lease_fanout(
    descriptor: dict[str, Any],
    *,
    target_device_indices: list[int],
    target_table: GpuReplayLeaseTable,
) -> dict[str, Any]:
    del descriptor, target_device_indices, target_table
    raise _unsupported()
