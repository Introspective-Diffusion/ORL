from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter


@dataclass
class RolloutForwardEvent:
    kind: str
    tokens: int
    seconds: float
    active_sequences: int = 0
    active_blocks: int = 0
    accepted_tokens: int = 0
    masked_tokens_remaining: int = 0

    @property
    def tpf(self) -> int:
        return int(self.tokens)

    @property
    def tokens_per_second(self) -> float:
        return float(self.tokens) / max(float(self.seconds), 1e-12)

    @property
    def forward_time_ms(self) -> float:
        return float(self.seconds) * 1000.0


@dataclass
class RolloutMetrics:
    events: list[RolloutForwardEvent] = field(default_factory=list)

    def add(self, event: RolloutForwardEvent) -> None:
        self.events.append(event)

    def summary(self, prefix: str = "rollout") -> dict[str, float]:
        if not self.events:
            return {}
        total_tokens = sum(event.tokens for event in self.events)
        total_seconds = sum(event.seconds for event in self.events)
        denoise = [event for event in self.events if event.kind == "denoise"]
        prefill = [event for event in self.events if event.kind == "prefill"]

        metrics = {
            f"{prefix}/forward_count": float(len(self.events)),
            f"{prefix}/tpf": float(total_tokens) / len(self.events),
            f"{prefix}/tokens_per_second": float(total_tokens) / max(total_seconds, 1e-12),
            f"{prefix}/forward_time_ms": 1000.0 * total_seconds / len(self.events),
            f"{prefix}/accepted_tokens": float(sum(event.accepted_tokens for event in self.events)),
            f"{prefix}/masked_tokens_remaining": float(self.events[-1].masked_tokens_remaining),
        }
        if prefill:
            metrics[f"{prefix}/prefill_tpf"] = float(sum(event.tokens for event in prefill)) / len(prefill)
            metrics[f"{prefix}/prefill_time_ms"] = 1000.0 * sum(event.seconds for event in prefill) / len(prefill)
        if denoise:
            metrics[f"{prefix}/denoise_tpf"] = float(sum(event.tokens for event in denoise)) / len(denoise)
            metrics[f"{prefix}/denoise_time_ms"] = 1000.0 * sum(event.seconds for event in denoise) / len(denoise)
        return metrics


class RolloutTimer:
    def __init__(self, metrics: RolloutMetrics, kind: str, tokens: int, **metadata):
        self.metrics = metrics
        self.kind = kind
        self.tokens = int(tokens)
        self.metadata = metadata
        self.start = 0.0

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        seconds = perf_counter() - self.start
        self.metrics.add(
            RolloutForwardEvent(
                kind=self.kind,
                tokens=self.tokens,
                seconds=seconds,
                active_sequences=int(self.metadata.get("active_sequences", 0)),
                active_blocks=int(self.metadata.get("active_blocks", 0)),
                accepted_tokens=int(self.metadata.get("accepted_tokens", 0)),
                masked_tokens_remaining=int(self.metadata.get("masked_tokens_remaining", 0)),
            )
        )
        return False
