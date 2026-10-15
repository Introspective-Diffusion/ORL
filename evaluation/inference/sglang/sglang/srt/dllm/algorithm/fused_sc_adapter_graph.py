"""Bitwise-guarded CUDA graphs for the eager self-conditioning adapter."""

from __future__ import annotations

import dataclasses
import logging
import os

import torch


logger = logging.getLogger(__name__)
_ENABLED = os.environ.get("SDAR_FUSED_SC_ADAPTER_GRAPH", "1") == "1"
_CHECK_EVERY = int(os.environ.get("SDAR_FUSED_SC_ADAPTER_GRAPH_CHECK_EVERY", "0"))


@dataclasses.dataclass
class _Entry:
    graph: torch.cuda.CUDAGraph
    inputs: torch.Tensor
    signal: torch.Tensor
    output: torch.Tensor
    replays: int = 0


class ExactSCAdapterGraphs:
    """Capture adapter calls by exact `(rows, dtype, post_norm)` shape."""

    def __init__(self):
        self.entries: dict[tuple, _Entry] = {}
        self.disabled_keys: set[tuple] = set()
        self.stream: torch.cuda.Stream | None = None

    @staticmethod
    def _key(inputs: torch.Tensor, post_norm: bool) -> tuple:
        return (
            int(inputs.shape[0]),
            int(inputs.shape[1]),
            str(inputs.dtype),
            bool(post_norm),
        )

    def run(
        self,
        adapter,
        inputs: torch.Tensor,
        signal: torch.Tensor,
        *,
        post_norm: bool,
    ) -> torch.Tensor | None:
        if (
            not _ENABLED
            or not inputs.is_cuda
            or inputs.shape != signal.shape
            or inputs.requires_grad
            or signal.requires_grad
        ):
            return None
        key = self._key(inputs, post_norm)
        if key in self.disabled_keys:
            return None
        entry = self.entries.get(key)
        if entry is None:
            try:
                return self._capture(
                    key, adapter, inputs, signal, post_norm=post_norm
                )
            except Exception as exc:  # noqa: BLE001
                self.disabled_keys.add(key)
                logger.warning(
                    "[SDAR-SC-GRAPH] disabled key=%s after capture failure: %r",
                    key,
                    exc,
                )
                return None
        entry.inputs.copy_(inputs)
        entry.signal.copy_(signal)
        entry.graph.replay()
        entry.replays += 1
        if _CHECK_EVERY and entry.replays % _CHECK_EVERY == 0:
            torch.cuda.synchronize()
            replay = entry.output.clone()
            eager = adapter(inputs, signal, post_norm=post_norm)
            torch.cuda.synchronize()
            if not torch.equal(replay, eager):
                self.disabled_keys.add(key)
                self.entries.pop(key, None)
                logger.warning(
                    "[SDAR-SC-GRAPH] guard mismatch at replay=%d key=%s",
                    entry.replays,
                    key,
                )
                return eager
        return entry.output

    def _capture(
        self,
        key: tuple,
        adapter,
        inputs: torch.Tensor,
        signal: torch.Tensor,
        *,
        post_norm: bool,
    ) -> torch.Tensor:
        eager = adapter(inputs, signal, post_norm=post_norm)
        eager_ref = eager.clone()
        torch.cuda.synchronize()

        static_inputs = inputs.clone()
        static_signal = signal.clone()
        graph = torch.cuda.CUDAGraph()
        if self.stream is None:
            self.stream = torch.cuda.Stream(device=inputs.device)
        stream = self.stream
        stream.wait_stream(torch.cuda.current_stream(inputs.device))
        with torch.cuda.stream(stream):
            with torch.cuda.graph(graph, stream=stream):
                static_output = adapter(
                    static_inputs,
                    static_signal,
                    post_norm=post_norm,
                )
        torch.cuda.current_stream(inputs.device).wait_stream(stream)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        if not torch.equal(static_output, eager_ref):
            self.disabled_keys.add(key)
            raise RuntimeError(f"adapter replay differs from eager for key={key}")
        self.entries[key] = _Entry(
            graph=graph,
            inputs=static_inputs,
            signal=static_signal,
            output=static_output,
        )
        logger.info("[SDAR-SC-GRAPH] captured key=%s", key)
        return eager
