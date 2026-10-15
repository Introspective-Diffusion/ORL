"""Per-algorithm SDAR CUDA graph metrics with CUDA-event GPU timing."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Optional


class SdarCudaGraphMetrics:
    """Aggregate metrics for SDAR CUDA graph acceleration."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total_forwards: int = 0
        self.graph_replays: int = 0
        self.eager_fallbacks: int = 0
        self.fallback_reasons: Counter[str] = Counter()

        self.total_graph_time_ms: float = 0.0
        self.total_eager_time_ms: float = 0.0
        self.timed_graph_steps: int = 0
        self.timed_eager_steps: int = 0
        self.total_graph_tokens: int = 0
        self.total_eager_tokens: int = 0
        self._latency_window_size: int = 256
        self._latency_samples_ms: list[float] = []
        self._latency_is_graph: list[bool] = []
        self._pending_cuda_events: list[tuple[Any, Any, bool, int]] = []

        self.capture_count: int = 0
        self.capture_time_s: float = 0.0
        self.graph_cache_entries: int = 0
        self.graph_cache_hbm_bytes: int = 0
        self.graph_cache_evictions: int = 0
        self.policy_version: str = "initial"

    # ------------------------------------------------------------------
    # Recording helpers
    # ------------------------------------------------------------------

    def record_forward(
        self,
        *,
        graph_was_replayed: bool,
        miss_reason: Optional[str],
        start_event: Optional[Any],
        end_event: Optional[Any],
        num_tokens: int,
    ) -> None:
        self.total_forwards += 1
        if graph_was_replayed:
            self.graph_replays += 1
        else:
            self.eager_fallbacks += 1
            self.fallback_reasons[miss_reason or "unknown"] += 1
        if start_event is not None and end_event is not None:
            self._pending_cuda_events.append(
                (start_event, end_event, graph_was_replayed, int(num_tokens))
            )
        self.flush_cuda_events(blocking=False)

    def _push_latency(self, latency_ms: float, is_graph: bool, num_tokens: int) -> None:
        self._latency_samples_ms.append(latency_ms)
        self._latency_is_graph.append(is_graph)
        if is_graph:
            self.total_graph_time_ms += latency_ms
            self.total_graph_tokens += num_tokens
            self.timed_graph_steps += 1
        else:
            self.total_eager_time_ms += latency_ms
            self.total_eager_tokens += num_tokens
            self.timed_eager_steps += 1
        if len(self._latency_samples_ms) > self._latency_window_size:
            self._latency_samples_ms.pop(0)
            self._latency_is_graph.pop(0)

    def flush_cuda_events(self, *, blocking: bool) -> None:
        remaining = []
        for start, end, is_graph, num_tokens in self._pending_cuda_events:
            if blocking:
                end.synchronize()
            elif not end.query():
                remaining.append((start, end, is_graph, num_tokens))
                continue
            self._push_latency(
                float(start.elapsed_time(end)),
                is_graph=is_graph,
                num_tokens=num_tokens,
            )
        self._pending_cuda_events = remaining

    def update_cache_stats(self, model_runner: Any) -> None:
        runner = getattr(model_runner, "graph_runner", None)
        if runner is None:
            return
        self.capture_count = int(getattr(runner, "capture_count", 0))
        self.capture_time_s = float(getattr(runner, "capture_time_s", 0.0))
        self.graph_cache_entries = len(getattr(runner, "graphs", {}))
        self.graph_cache_evictions = int(getattr(runner, "cache_evictions", 0))
        self.graph_cache_hbm_bytes = int(
            max(float(getattr(model_runner, "graph_mem_usage", 0.0)), 0.0)
            * (1024**3)
        )
        self.policy_version = str(
            getattr(model_runner, "_sdar_policy_version", "initial")
        )

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def get_stats(self, *, synchronize: bool = False) -> Dict[str, object]:
        self.flush_cuda_events(blocking=synchronize)
        graph_hit_rate = (
            self.graph_replays / self.total_forwards if self.total_forwards > 0 else 0.0
        )
        avg_graph_latency_ms = (
            self.total_graph_time_ms / self.timed_graph_steps
            if self.timed_graph_steps > 0
            else float("nan")
        )
        avg_eager_latency_ms = (
            self.total_eager_time_ms / self.timed_eager_steps
            if self.timed_eager_steps > 0
            else float("nan")
        )
        if self._latency_samples_ms:
            sorted_lat = sorted(self._latency_samples_ms)
            n = len(sorted_lat)
            p50_ms = sorted_lat[n // 2]
            p99_ms = sorted_lat[min(n - 1, int(n * 0.99))]
        else:
            p50_ms = float("nan")
            p99_ms = float("nan")

        result = {
            "cuda_graph_total_forwards": self.total_forwards,
            "cuda_graph_replays": self.graph_replays,
            "cuda_graph_eager_fallbacks": self.eager_fallbacks,
            "cuda_graph_hit_rate": round(graph_hit_rate, 4),
            "cuda_graph_avg_graph_latency_ms": round(avg_graph_latency_ms, 3),
            "cuda_graph_avg_eager_latency_ms": round(avg_eager_latency_ms, 3),
            "cuda_graph_p50_latency_ms": round(p50_ms, 3),
            "cuda_graph_p99_latency_ms": round(p99_ms, 3),
            "cuda_graph_true_forward_tps": (
                round(self.total_graph_tokens / (self.total_graph_time_ms / 1000.0), 3)
                if self.total_graph_time_ms > 0
                else float("nan")
            ),
            "cuda_graph_eager_true_forward_tps": (
                round(self.total_eager_tokens / (self.total_eager_time_ms / 1000.0), 3)
                if self.total_eager_time_ms > 0
                else float("nan")
            ),
            "cuda_graph_capture_count": self.capture_count,
            "cuda_graph_capture_time_s": round(self.capture_time_s, 3),
            "cuda_graph_cache_entries": self.graph_cache_entries,
            "cuda_graph_cache_hbm_bytes": self.graph_cache_hbm_bytes,
            "cuda_graph_cache_evictions": self.graph_cache_evictions,
            "cuda_graph_policy_version": self.policy_version,
            "cuda_graph_pending_latency_events": len(self._pending_cuda_events),
        }
        for reason, count in sorted(self.fallback_reasons.items()):
            result[f"cuda_graph_fallback_{reason}"] = count
        return result

    def log_summary(self, logger, prefix: str = "[SDAR-CudaGraph]") -> None:
        s = self.get_stats(synchronize=False)
        logger.info(
            "%s forwards=%d graph=%d(%.1f%%) eager=%d "
            "avg_graph=%.2fms avg_eager=%.2fms p50=%.2fms p99=%.2fms "
            "fallbacks=%s cache_entries=%d hbm_bytes=%d policy=%s",
            prefix,
            s["cuda_graph_total_forwards"],
            s["cuda_graph_replays"],
            100.0 * s["cuda_graph_hit_rate"],
            s["cuda_graph_eager_fallbacks"],
            s["cuda_graph_avg_graph_latency_ms"],
            s["cuda_graph_avg_eager_latency_ms"],
            s["cuda_graph_p50_latency_ms"],
            s["cuda_graph_p99_latency_ms"],
            dict(self.fallback_reasons),
            s["cuda_graph_cache_entries"],
            s["cuda_graph_cache_hbm_bytes"],
            s["cuda_graph_policy_version"],
        )
