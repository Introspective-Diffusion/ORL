"""Low-overhead, machine-readable RL lifecycle diagnostics."""

from __future__ import annotations

import json
import os
import time
from typing import Any

import torch


def lifecycle_memory(tag: str) -> dict[str, float | str | int]:
    """Return process allocator and device-wide free-memory counters."""
    record: dict[str, float | str | int] = {
        "tag": tag,
        "pid": os.getpid(),
        "device": -1,
        "allocated": 0,
        "reserved": 0,
        "free": 0,
        "total": 0,
    }
    if not torch.cuda.is_available():
        return record
    try:
        device = torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(device)
        record.update(
            {
                "device": int(device),
                "allocated": int(torch.cuda.memory_allocated(device)),
                "reserved": int(torch.cuda.memory_reserved(device)),
                "free": int(free),
                "total": int(total),
            }
        )
    except Exception as exc:  # diagnostics must never break the lifecycle
        record["memory_error"] = repr(exc)
    return record


def emit_lifecycle(component: str, event: str, **fields: Any) -> dict[str, Any]:
    """Print one stable JSON record and return it for callers/tests."""
    record = {
        "component": component,
        "event": event,
        "monotonic_seconds": time.monotonic(),
        "pid": os.getpid(),
        **fields,
    }
    print(
        "[RL_LIFECYCLE] "
        + json.dumps(record, sort_keys=True, default=str, separators=(",", ":")),
        flush=True,
    )
    return record
