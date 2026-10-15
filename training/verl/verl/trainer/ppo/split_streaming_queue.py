import asyncio
import time
from collections import deque
from typing import Any

import ray


@ray.remote(num_cpus=0, max_concurrency=64)
class SplitStreamingReplayQueue:
    """Tiny async Ray queue for split-GPU rollout-to-actor replay chunks."""

    def __init__(self, queue_name: str = "split_streaming", max_items: int = 0):
        self._queue_name = str(queue_name)
        self._max_items = max(0, int(max_items or 0))
        self._items = deque()
        self._closed = False
        self._cond = asyncio.Condition()
        self._pushed = 0
        self._popped = 0
        self._max_queued = 0
        self._push_wait_seconds = 0.0
        self._push_wait_seconds_max = 0.0
        self._push_wait_count = 0
        self._first_push_ts = None
        self._last_push_ts = None
        self._first_pop_ts = None
        self._last_pop_ts = None

    async def push(self, item: Any, source: str | None = None) -> int:
        async with self._cond:
            wait_start = None
            while self._max_items > 0 and len(self._items) >= self._max_items and not self._closed:
                if wait_start is None:
                    wait_start = time.perf_counter()
                await self._cond.wait()
            if self._closed:
                raise RuntimeError("SplitStreamingReplayQueue is closed")
            if wait_start is not None:
                wait_seconds = time.perf_counter() - wait_start
                self._push_wait_seconds += wait_seconds
                self._push_wait_seconds_max = max(self._push_wait_seconds_max, wait_seconds)
                self._push_wait_count += 1
            idx = self._pushed
            self._pushed += 1
            rows = len(item) if hasattr(item, "__len__") else -1
            ts = time.time()
            if self._first_push_ts is None:
                self._first_push_ts = ts
            self._last_push_ts = ts
            self._items.append((idx, item, source, ts))
            self._max_queued = max(self._max_queued, len(self._items))
            print(
                f"[SplitStreaming] queue_push ts={ts:.6f} queue={self._queue_name} "
                f"idx={idx} rows={rows} source={source}",
                flush=True,
            )
            self._cond.notify()
            return idx

    async def pop(self) -> dict[str, Any] | None:
        async with self._cond:
            while not self._items and not self._closed:
                await self._cond.wait()
            if not self._items:
                return None
            idx, item, source, push_ts = self._items.popleft()
            self._cond.notify_all()
            self._popped += 1
            ts = time.time()
            if self._first_pop_ts is None:
                self._first_pop_ts = ts
            self._last_pop_ts = ts
            return {"idx": idx, "item": item, "source": source, "push_ts": push_ts}

    async def pop_many(self, max_items: int = 1, timeout_s: float | None = None) -> list[dict[str, Any]] | None:
        max_items = max(int(max_items), 1)
        timeout_s = None if timeout_s is None else max(0.0, float(timeout_s))
        async with self._cond:
            while not self._items and not self._closed:
                if timeout_s is None:
                    await self._cond.wait()
                else:
                    try:
                        await asyncio.wait_for(self._cond.wait(), timeout=timeout_s)
                    except asyncio.TimeoutError:
                        return []
            if not self._items:
                return None
            items = []
            while self._items and len(items) < max_items:
                idx, item, source, push_ts = self._items.popleft()
                self._popped += 1
                ts = time.time()
                if self._first_pop_ts is None:
                    self._first_pop_ts = ts
                self._last_pop_ts = ts
                items.append({"idx": idx, "item": item, "source": source, "push_ts": push_ts})
            self._cond.notify_all()
            return items

    async def close(self) -> dict[str, Any]:
        async with self._cond:
            self._closed = True
            self._cond.notify_all()
            return await self.stats()

    async def stats(self) -> dict[str, Any]:
        return {
            "pushed": self._pushed,
            "popped": self._popped,
            "queued": len(self._items),
            "max_items": self._max_items,
            "max_queued": self._max_queued,
            "push_wait_seconds": float(self._push_wait_seconds),
            "push_wait_seconds_max": float(self._push_wait_seconds_max),
            "push_wait_count": self._push_wait_count,
            "closed": int(self._closed),
            "queue_name": self._queue_name,
            "first_push_ts": float(self._first_push_ts or 0.0),
            "last_push_ts": float(self._last_push_ts or 0.0),
            "first_pop_ts": float(self._first_pop_ts or 0.0),
            "last_pop_ts": float(self._last_pop_ts or 0.0),
        }
