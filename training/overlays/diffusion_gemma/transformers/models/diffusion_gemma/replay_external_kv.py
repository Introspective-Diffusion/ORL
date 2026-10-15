from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar

import torch


class DiffusionGemmaExternalKVReplayStats:
    """Per-context counters for the no-copy replay attention path."""

    def __init__(self, prefix_grad_mode: str = "detached"):
        prefix_grad_mode = str(prefix_grad_mode).strip().lower()
        if prefix_grad_mode not in {"detached", "full"}:
            raise ValueError(
                "DiffusionGemma external-KV prefix_grad_mode must be "
                f"'detached' or 'full', got {prefix_grad_mode!r}"
            )
        self.prefix_grad_mode = prefix_grad_mode
        self.calls = 0
        self.backend_triton_calls = 0
        self.native_forward_calls = 0
        self.native_backward_calls = 0
        self.fallbacks = 0
        self.shared_prefix_rows = 0
        self.shared_prefix_bytes_avoided = 0
        self.kv_pack_calls = 0
        self.kv_repeat_calls = 0
        self.prefix_append_calls = 0
        self._kernel_cpu_seconds = 0.0
        self._kernel_cuda_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self.last_fallback_reason: str | None = None

    def record_fallback(self, reason: str) -> None:
        self.fallbacks += 1
        self.last_fallback_reason = reason

    def record_native_backward(self, grad: torch.Tensor) -> torch.Tensor:
        self.native_backward_calls += 1
        return grad

    def record_kernel_start(self, device: torch.device):
        if device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event
        return time.perf_counter()

    def record_kernel_end(self, start) -> None:
        if isinstance(start, torch.cuda.Event):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self._kernel_cuda_events.append((start, end))
        else:
            self._kernel_cpu_seconds += time.perf_counter() - start

    def kernel_seconds(self) -> float:
        cuda_seconds = sum(start.elapsed_time(end) for start, end in self._kernel_cuda_events) / 1000.0
        return self._kernel_cpu_seconds + cuda_seconds

    def assert_no_kv_pack_or_repeat(self) -> None:
        assert self.kv_pack_calls == 0, "DiffusionGemma external-KV replay packed prefix/canvas KV"
        assert self.kv_repeat_calls == 0, "DiffusionGemma external-KV replay repeated the prefix KV cache"
        assert self.prefix_append_calls == 0, "DiffusionGemma external-KV replay appended prefix and canvas KV"

    def metrics(self, prefix: str = "actor/diffusion_gemma/external_kv") -> dict[str, float]:
        self.assert_no_kv_pack_or_repeat()
        return {
            f"{prefix}/enabled": 1.0,
            f"{prefix}/calls": float(self.calls),
            f"{prefix}/backend_triton_calls": float(self.backend_triton_calls),
            f"{prefix}/native_forward_calls": float(self.native_forward_calls),
            f"{prefix}/native_backward_calls": float(self.native_backward_calls),
            f"{prefix}/prefix_grad_mode": (
                1.0 if self.prefix_grad_mode == "full" else 0.0
            ),
            f"{prefix}/prefix_grad_mode_full": (
                1.0 if self.prefix_grad_mode == "full" else 0.0
            ),
            f"{prefix}/prefix_grad_mode_detached": (
                1.0 if self.prefix_grad_mode == "detached" else 0.0
            ),
            f"{prefix}/fallbacks": float(self.fallbacks),
            f"{prefix}/kernel_seconds": float(self.kernel_seconds()),
            f"{prefix}/shared_prefix_rows": float(self.shared_prefix_rows),
            f"{prefix}/shared_prefix_bytes_avoided": float(self.shared_prefix_bytes_avoided),
            f"{prefix}/kv_pack_calls": float(self.kv_pack_calls),
            f"{prefix}/kv_repeat_calls": float(self.kv_repeat_calls),
            f"{prefix}/prefix_append_calls": float(self.prefix_append_calls),
            f"{prefix}/prefix_append_seconds": 0.0,
        }


_DIFFUSION_GEMMA_EXTERNAL_KV_REPLAY = ContextVar(
    "diffusion_gemma_external_kv_replay",
    default=None,
)

_external_kv_varlen_attention = None
_triton_backend_supported = None


@contextmanager
def diffusion_gemma_external_kv_replay(
    enabled: bool = True,
    *,
    stats: DiffusionGemmaExternalKVReplayStats | None = None,
    prefix_grad_mode: str | None = None,
):
    """Enable external-prefix KV attention only within this replay context."""

    if not enabled:
        yield stats
        return
    if prefix_grad_mode is None:
        prefix_grad_mode = (
            stats.prefix_grad_mode if stats is not None else "detached"
        )
    prefix_grad_mode = str(prefix_grad_mode).strip().lower()
    if stats is None:
        stats = DiffusionGemmaExternalKVReplayStats(
            prefix_grad_mode=prefix_grad_mode
        )
    elif stats.prefix_grad_mode != prefix_grad_mode:
        raise ValueError(
            "DiffusionGemma external-KV stats/context prefix_grad_mode mismatch: "
            f"{stats.prefix_grad_mode!r} != {prefix_grad_mode!r}"
        )
    token = _DIFFUSION_GEMMA_EXTERNAL_KV_REPLAY.set(stats)
    try:
        yield stats
    finally:
        _DIFFUSION_GEMMA_EXTERNAL_KV_REPLAY.reset(token)
        stats.assert_no_kv_pack_or_repeat()


def _load_external_kv_kernel():
    global _external_kv_varlen_attention
    global _triton_backend_supported
    if _external_kv_varlen_attention is None:
        from fast_diffusion_attention import external_kv_varlen_attention
        from fast_diffusion_attention.external_prefix_kv_triton import triton_backend_supported

        _external_kv_varlen_attention = external_kv_varlen_attention
        _triton_backend_supported = triton_backend_supported
    return _external_kv_varlen_attention, _triton_backend_supported


def _reject_external_kv(stats: DiffusionGemmaExternalKVReplayStats, reason: str):
    stats.record_fallback(reason)
    raise RuntimeError(
        "DiffusionGemma external-KV replay requires the no-copy Triton backend; "
        f"refusing a packing/repeat fallback: {reason}"
    )


def _packed_canvas_view(states: torch.Tensor, name: str) -> torch.Tensor:
    batch_size, _, canvas_length, _ = states.shape
    packed = states.transpose(1, 2).reshape(batch_size * canvas_length, states.shape[1], states.shape[3])
    assert packed.data_ptr() == states.data_ptr(), f"{name} packing unexpectedly copied canvas states"
    return packed


def diffusion_gemma_external_kv_attention(
    module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values,
) -> torch.Tensor | None:
    """Run no-copy shared-prefix attention when the replay context is active."""

    stats = _DIFFUSION_GEMMA_EXTERNAL_KV_REPLAY.get()
    if stats is None:
        return None
    if past_key_values is None:
        return _reject_external_kv(stats, "past_key_values is None")
    if past_key_values.__class__.__name__ != "DynamicCache":
        return _reject_external_kv(
            stats,
            f"expected a DynamicCache, got {past_key_values.__class__.__name__}",
        )
    if getattr(past_key_values, "is_compileable", False):
        return _reject_external_kv(stats, "compileable/static caches are not supported")
    if attention_mask is not None:
        return _reject_external_kv(stats, "masked/padded decoder replay is not supported")
    if module.training and float(module.attention_dropout) != 0.0:
        return _reject_external_kv(stats, "attention dropout must be zero")
    if query_states.ndim != 4 or key_states.ndim != 4 or value_states.ndim != 4:
        return _reject_external_kv(stats, "canvas Q/K/V must be rank-4")

    batch_size, _, canvas_length, _ = query_states.shape
    if key_states.shape[0] != batch_size or value_states.shape[0] != batch_size:
        return _reject_external_kv(stats, "canvas Q/K/V batch dimensions differ")
    if key_states.shape[2] != canvas_length or value_states.shape[2] != canvas_length:
        return _reject_external_kv(stats, "canvas Q/K/V sequence lengths differ")
    if module.layer_idx >= len(past_key_values.layers):
        return _reject_external_kv(stats, f"prefix cache has no layer {module.layer_idx}")

    cache_layer = past_key_values.layers[module.layer_idx]
    encoder_key_states = cache_layer.keys
    encoder_value_states = cache_layer.values
    if encoder_key_states.ndim != 4 or encoder_value_states.ndim != 4:
        return _reject_external_kv(stats, "prefix K/V must be rank-4")
    if encoder_key_states.shape[0] != 1 or encoder_value_states.shape[0] != 1:
        return _reject_external_kv(
            stats,
            "the original prefix cache must retain batch size 1",
        )
    if encoder_key_states.shape != encoder_value_states.shape:
        return _reject_external_kv(stats, "prefix K/V shapes differ")

    q_canvas = _packed_canvas_view(query_states, "query")
    k_canvas = _packed_canvas_view(key_states, "key")
    v_canvas = _packed_canvas_view(value_states, "value")
    k_prefix = encoder_key_states[0].transpose(0, 1)
    v_prefix = encoder_value_states[0].transpose(0, 1)
    if stats.prefix_grad_mode == "detached":
        k_prefix = k_prefix.detach()
        v_prefix = v_prefix.detach()
    elif not k_prefix.requires_grad or not v_prefix.requires_grad:
        return _reject_external_kv(
            stats,
            "full prefix-gradient mode requires gradient-receiving prefix K/V leaves",
        )
    assert k_prefix.data_ptr() == encoder_key_states.data_ptr(), "prefix K packing unexpectedly copied states"
    assert v_prefix.data_ptr() == encoder_value_states.data_ptr(), "prefix V packing unexpectedly copied states"

    prefix_length = int(k_prefix.shape[0])
    if prefix_length <= 0:
        return _reject_external_kv(stats, "prefix cache is empty")
    cu_seqlens_q = torch.arange(
        0,
        (batch_size + 1) * canvas_length,
        canvas_length,
        dtype=torch.int32,
        device=q_canvas.device,
    )
    cu_seqlens_prefix = torch.tensor(
        [0, prefix_length],
        dtype=torch.int32,
        device=q_canvas.device,
    )
    prefix_segment_ids = torch.zeros(batch_size, dtype=torch.int32, device=q_canvas.device)
    prefix_window = int(module.sliding_window) if module.is_sliding else None

    try:
        external_attention, backend_supported = _load_external_kv_kernel()
    except (ImportError, ModuleNotFoundError) as exc:
        return _reject_external_kv(stats, f"kernel import failed: {exc}")
    supported, reason = backend_supported(
        q_canvas,
        k_canvas,
        v_canvas,
        k_prefix,
        v_prefix,
        cu_seqlens_q,
        cu_seqlens_prefix,
        prefix_segment_ids,
    )
    if not supported:
        return _reject_external_kv(stats, reason)

    key_ptr = encoder_key_states.data_ptr()
    value_ptr = encoder_value_states.data_ptr()
    key_version = encoder_key_states._version
    value_version = encoder_value_states._version
    start = stats.record_kernel_start(q_canvas.device)
    output = external_attention(
        q_canvas,
        k_canvas,
        v_canvas,
        k_prefix,
        v_prefix,
        cu_seqlens_q,
        cu_seqlens_prefix,
        max_seqlen_q=canvas_length,
        max_seqlen_prefix=prefix_length,
        scale=module.scaling,
        prefix_window=prefix_window,
        backend="triton",
        prefix_segment_ids=prefix_segment_ids,
    )
    stats.record_kernel_end(start)
    if output.requires_grad:
        output.register_hook(stats.record_native_backward)

    expected_shape = (batch_size * canvas_length, query_states.shape[1], query_states.shape[3])
    if tuple(output.shape) != expected_shape:
        raise RuntimeError(
            "DiffusionGemma external-KV kernel returned the wrong shape: " f"{tuple(output.shape)} != {expected_shape}"
        )
    assert encoder_key_states.data_ptr() == key_ptr and encoder_key_states._version == key_version
    assert encoder_value_states.data_ptr() == value_ptr and encoder_value_states._version == value_version

    effective_prefix_length = min(prefix_length, prefix_window) if prefix_window is not None else prefix_length
    prefix_row_bytes = (
        2 * effective_prefix_length * int(k_prefix.shape[1]) * int(k_prefix.shape[2]) * k_prefix.element_size()
    )
    stats.calls += 1
    stats.backend_triton_calls += 1
    stats.native_forward_calls += 1
    stats.shared_prefix_rows += batch_size
    stats.shared_prefix_bytes_avoided += max(batch_size - 1, 0) * prefix_row_bytes
    stats.assert_no_kv_pack_or_repeat()
    return output.reshape(batch_size, canvas_length, query_states.shape[1], query_states.shape[3])
