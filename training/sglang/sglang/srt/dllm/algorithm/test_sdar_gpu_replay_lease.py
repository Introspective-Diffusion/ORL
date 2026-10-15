from multiprocessing import shared_memory

import numpy as np
import pytest
import torch

from sglang.srt.dllm.algorithm.sdar_gpu_replay_lease import (
    GpuReplayLeaseTable,
    GPU_REPLAY_LEASE_FANOUT_FORMAT,
    GPU_REPLAY_LEASE_FORMAT,
    acknowledge_gpu_replay_lease,
    is_gpu_replay_lease,
    is_gpu_replay_lease_fanout,
    select_gpu_replay_lease_for_device,
    slice_gpu_replay_lease,
)


def _descriptor() -> dict:
    return {
        "format": GPU_REPLAY_LEASE_FORMAT,
        "lease_id": "test",
        "shape": [8, 4],
        "stride": [4, 1],
        "tensor_storage_offset_elems": 2,
        "logical_slice_start": 0,
    }


def test_slice_gpu_replay_lease_updates_view_metadata():
    descriptor = _descriptor()
    sliced = slice_gpu_replay_lease(descriptor, 3, 2)

    assert is_gpu_replay_lease(sliced)
    assert sliced["shape"] == [2, 4]
    assert sliced["tensor_storage_offset_elems"] == 14
    assert sliced["logical_slice_start"] == 3
    assert descriptor["shape"] == [8, 4]


def test_acknowledge_gpu_replay_lease_increments_shared_counter():
    ack = shared_memory.SharedMemory(
        create=True,
        size=np.dtype(np.int64).itemsize,
    )
    view = np.ndarray((1,), dtype=np.int64, buffer=ack.buf)
    view[0] = 0
    descriptor = {
        **_descriptor(),
        "ack_shm_name": ack.name,
    }
    try:
        acknowledge_gpu_replay_lease(descriptor)
        assert int(view[0]) == 1
    finally:
        ack.close()
        ack.unlink()


def test_select_gpu_replay_lease_fanout_by_device_uuid():
    first = {**_descriptor(), "device_uuid": "gpu-a"}
    second = {**_descriptor(), "device_uuid": "gpu-b"}
    fanout = {
        "format": GPU_REPLAY_LEASE_FANOUT_FORMAT,
        "descriptors": [first, second],
    }

    assert is_gpu_replay_lease_fanout(fanout)
    assert (
        select_gpu_replay_lease_for_device(
            fanout,
            device_uuid="gpu-b",
        )
        is second
    )


def test_select_gpu_replay_lease_fanout_rejects_missing_device():
    fanout = {
        "format": GPU_REPLAY_LEASE_FANOUT_FORMAT,
        "descriptors": [
            {**_descriptor(), "device_uuid": "gpu-a"},
        ],
    }

    try:
        select_gpu_replay_lease_for_device(
            fanout,
            device_uuid="gpu-b",
        )
    except RuntimeError as error:
        assert "exactly one actor-local descriptor" in str(error)
    else:
        raise AssertionError("missing fan-out device must fail")


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA unavailable",
)
def test_gpu_replay_arena_reuses_bounded_storage(monkeypatch):
    monkeypatch.setenv("SGLANG_SDAR_GPU_REPLAY_ARENA", "1")
    monkeypatch.setenv(
        "SGLANG_SDAR_GPU_REPLAY_ARENA_CHUNK_BYTES",
        str(4 * 1024**2),
    )
    table = GpuReplayLeaseTable(
        max_bytes=8 * 1024**2,
        trim_recycled_cache=False,
    )
    first = torch.arange(
        256 * 64,
        device="cuda",
        dtype=torch.bfloat16,
    ).reshape(256, 64)
    second = first + 1
    first_desc = table.export(first, logical_name="first")
    second_desc = table.export(second, logical_name="second")
    first_lease = table._leases[first_desc["lease_id"]]
    second_lease = table._leases[second_desc["lease_id"]]

    torch.testing.assert_close(first_lease.tensor, first)
    torch.testing.assert_close(second_lease.tensor, second)
    assert (
        first_lease.tensor.untyped_storage().data_ptr()
        == second_lease.tensor.untyped_storage().data_ptr()
    )
    assert table.stats()["arena_chunks"] == 1
    assert table.stats()["arena_bytes"] == 4 * 1024**2

    acknowledge_gpu_replay_lease(first_desc)
    acknowledge_gpu_replay_lease(second_desc)
    table._recycle_once()
    assert table.stats()["active_leases"] == 0
    table.close()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA unavailable",
)
def test_force_recycle_all_trims_arena_storage(monkeypatch):
    monkeypatch.setenv("SGLANG_SDAR_GPU_REPLAY_ARENA", "1")
    monkeypatch.setenv(
        "SGLANG_SDAR_GPU_REPLAY_ARENA_CHUNK_BYTES",
        str(4 * 1024**2),
    )
    table = GpuReplayLeaseTable(
        max_bytes=8 * 1024**2,
        trim_recycled_cache=True,
    )
    tensor = torch.arange(
        256 * 64,
        device="cuda",
        dtype=torch.bfloat16,
    ).reshape(256, 64)
    table.export(tensor, logical_name="trim")
    assert table.stats()["arena_bytes"] == 4 * 1024**2

    recycled = table.force_recycle_all()

    assert recycled["released_leases"] == 1
    assert recycled["released_arena_bytes"] == 4 * 1024**2
    assert table.stats()["active_leases"] == 0
    assert table.stats()["arena_chunks"] == 0
    assert table.stats()["arena_bytes"] == 0
    table.close()
