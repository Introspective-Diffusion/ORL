from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import pytest
import torch

from sglang.srt.dllm.algorithm.sdar_cuda_graph_metrics import SdarCudaGraphMetrics


class FakeEvent:
    def __init__(self, elapsed_ms: float, ready: bool = True):
        self.elapsed_ms = elapsed_ms
        self.ready = ready

    def query(self):
        return self.ready

    def synchronize(self):
        self.ready = True

    def elapsed_time(self, _other):
        return self.elapsed_ms


def test_metrics_use_actual_replay_signal_and_reason():
    metrics = SdarCudaGraphMetrics()
    metrics.record_forward(
        graph_was_replayed=True,
        miss_reason=None,
        start_event=FakeEvent(1.25),
        end_event=FakeEvent(1.25),
        num_tokens=512,
    )
    metrics.record_forward(
        graph_was_replayed=False,
        miss_reason="mode_mismatch",
        start_event=FakeEvent(2.5),
        end_event=FakeEvent(2.5),
        num_tokens=512,
    )
    metrics.record_forward(
        graph_was_replayed=False,
        miss_reason="self_conditioning_shape_miss",
        start_event=None,
        end_event=None,
        num_tokens=512,
    )
    stats = metrics.get_stats(synchronize=True)
    assert stats["cuda_graph_replays"] == 1
    assert stats["cuda_graph_eager_fallbacks"] == 2
    assert stats["cuda_graph_fallback_mode_mismatch"] == 1
    assert stats["cuda_graph_fallback_self_conditioning_shape_miss"] == 1
    assert stats["cuda_graph_avg_graph_latency_ms"] == 1.25
    assert stats["cuda_graph_avg_eager_latency_ms"] == 2.5


def test_metrics_instances_do_not_share_policy_state():
    first = SdarCudaGraphMetrics()
    second = SdarCudaGraphMetrics()
    first.record_forward(
        graph_was_replayed=True,
        miss_reason=None,
        start_event=None,
        end_event=None,
        num_tokens=64,
    )
    assert first.total_forwards == 1
    assert second.total_forwards == 0


@dataclass(frozen=True)
class TinyGraphKey:
    batch_size: int
    block_size: int
    use_input_embeds: bool
    trace_hidden: bool
    dtype: torch.dtype


class TinySdar(torch.nn.Module):
    """Small deterministic SDAR analogue used for real CUDA graph parity."""

    def __init__(self, vocab_size=97, hidden_size=32):
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(20260713)
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.denoise = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.copy_(
                    torch.randn(
                        parameter.shape,
                        generator=generator,
                        dtype=parameter.dtype,
                    )
                    * 0.05
                )

    def self_condition(
        self, input_ids: torch.Tensor, previous_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base = self.embedding(input_ids)
        signal = previous_logits.float().softmax(-1).to(base.dtype) @ self.embedding.weight
        return (base + 0.2 * signal).contiguous(), signal

    def forward(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self.embedding(input_ids) if input_embeds is None else input_embeds
        hidden = torch.tanh(self.denoise(inputs))
        return self.lm_head(hidden).float(), hidden


class TinyGraphCache:
    """Captures token-id and self-conditioned input variants with an LRU cap."""

    def __init__(self, model: TinySdar, block_size=8, cap=4):
        self.model = model
        self.block_size = block_size
        self.cap = cap
        self.entries: OrderedDict[TinyGraphKey, dict] = OrderedDict()
        self.evictions = 0
        self.policy_version = "initial"

    def capture(self, key: TinyGraphKey):
        if key in self.entries:
            return
        while len(self.entries) >= self.cap:
            self.entries.popitem(last=False)
            self.evictions += 1
        num_tokens = key.batch_size * key.block_size
        input_ids = torch.zeros(num_tokens, dtype=torch.long, device="cuda")
        input_embeds = torch.zeros(
            num_tokens,
            self.model.embedding.embedding_dim,
            dtype=self.model.embedding.weight.dtype,
            device="cuda",
        )
        logits_buffer = torch.empty(
            num_tokens,
            self.model.embedding.num_embeddings,
            dtype=torch.float32,
            device="cuda",
        )
        hidden_buffer = torch.empty(
            num_tokens,
            self.model.embedding.embedding_dim,
            dtype=self.model.embedding.weight.dtype,
            device="cuda",
        )
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                self.model(
                    input_ids,
                    input_embeds if key.use_input_embeds else None,
                )
        torch.cuda.current_stream().wait_stream(warmup_stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            logits, hidden = self.model(
                input_ids,
                input_embeds if key.use_input_embeds else None,
            )
            logits_buffer.copy_(logits)
            if key.trace_hidden:
                hidden_buffer.copy_(hidden)
        self.entries[key] = {
            "graph": graph,
            "input_ids": input_ids,
            "input_embeds": input_embeds,
            "logits": logits_buffer,
            "hidden": hidden_buffer,
            "parameter_ptrs": self.parameter_ptrs(),
        }

    def parameter_ptrs(self):
        return tuple(parameter.data_ptr() for parameter in self.model.parameters())

    def replay(
        self,
        key: TinyGraphKey,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
    ):
        if key not in self.entries:
            raise KeyError(key)
        entry = self.entries[key]
        if tuple(input_ids.shape) != (key.batch_size * key.block_size,):
            raise ValueError("token_shape_miss")
        if key.use_input_embeds:
            expected = (
                key.batch_size * key.block_size,
                self.model.embedding.embedding_dim,
            )
            if input_embeds is None or tuple(input_embeds.shape) != expected:
                raise ValueError("self_conditioning_shape_miss")
            if not input_embeds.is_contiguous():
                raise ValueError("self_conditioning_layout_miss")
            entry["input_embeds"].copy_(input_embeds)
        entry["input_ids"].copy_(input_ids)
        entry["graph"].replay()
        return entry["logits"].clone(), entry["hidden"].clone()

    def validate_weight_pointers_or_recapture(self, key: TinyGraphKey):
        entry = self.entries[key]
        if entry["parameter_ptrs"] == self.parameter_ptrs():
            return True
        self.entries.pop(key)
        self.evictions += 1
        self.capture(key)
        return False

    def pointer_fingerprint(self, key: TinyGraphKey):
        entry = self.entries[key]
        return (
            tuple(value.data_ptr() for value in (entry["input_ids"], entry["input_embeds"])),
            entry["parameter_ptrs"],
        )


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


@pytest.fixture
def tiny_model():
    return TinySdar().to(device="cuda", dtype=torch.float32).eval()


def _assert_step_parity(
    model: TinySdar,
    cache: TinyGraphCache,
    key: TinyGraphKey,
    input_ids: torch.Tensor,
    input_embeds: torch.Tensor | None,
):
    with torch.no_grad():
        eager_logits, eager_hidden = model(input_ids, input_embeds)
        graph_logits, graph_hidden = cache.replay(key, input_ids, input_embeds)
    torch.testing.assert_close(graph_logits, eager_logits, rtol=0, atol=0)
    torch.testing.assert_close(graph_hidden, eager_hidden, rtol=0, atol=0)
    assert torch.equal(graph_logits.argmax(-1), eager_logits.argmax(-1))
    eager_edit = eager_logits.softmax(-1).amax(-1) < 0.6
    graph_edit = graph_logits.softmax(-1).amax(-1) < 0.6
    assert torch.equal(graph_edit, eager_edit)
    return eager_logits, eager_hidden


def test_cuda_graph_eager_parity_bsz8_trace_self_conditioning_multiple_blocks(
    tiny_model,
):
    cache = TinyGraphCache(tiny_model, block_size=8, cap=4)
    token_key = TinyGraphKey(8, 8, False, True, torch.float32)
    embed_key = TinyGraphKey(8, 8, True, True, torch.float32)
    cache.capture(token_key)
    cache.capture(embed_key)
    generator = torch.Generator(device="cuda").manual_seed(17)
    input_ids = torch.randint(
        0, tiny_model.embedding.num_embeddings, (64,), generator=generator, device="cuda"
    )

    previous_logits, _ = _assert_step_parity(
        tiny_model, cache, token_key, input_ids, None
    )
    for _block in range(3):
        conditioned, eager_signal = tiny_model.self_condition(
            input_ids, previous_logits
        )
        graph_logits, graph_hidden = _assert_step_parity(
            tiny_model, cache, embed_key, input_ids, conditioned
        )
        # Trace payload fields used by SDAR: hidden output and SC signal.
        torch.testing.assert_close(graph_hidden, tiny_model(input_ids, conditioned)[1])
        graph_signal = previous_logits.float().softmax(-1) @ tiny_model.embedding.weight
        torch.testing.assert_close(graph_signal, eager_signal)
        previous_logits = graph_logits
        input_ids = graph_logits.argmax(-1)


def test_hot_weight_copy_is_visible_without_recapture(tiny_model):
    cache = TinyGraphCache(tiny_model, block_size=8, cap=2)
    key = TinyGraphKey(8, 8, False, True, torch.float32)
    cache.capture(key)
    input_ids = torch.arange(64, device="cuda") % tiny_model.embedding.num_embeddings
    before, _ = cache.replay(key, input_ids, None)
    pointers_before = cache.parameter_ptrs()
    with torch.no_grad():
        tiny_model.denoise.weight.copy_(tiny_model.denoise.weight + 0.01)
    assert cache.parameter_ptrs() == pointers_before
    assert cache.validate_weight_pointers_or_recapture(key)
    graph_after, _ = cache.replay(key, input_ids, None)
    eager_after, _ = tiny_model(input_ids)
    torch.testing.assert_close(graph_after, eager_after, rtol=0, atol=0)
    assert not torch.equal(before, graph_after)


def test_pointer_change_and_memory_wake_force_recapture(tiny_model):
    cache = TinyGraphCache(tiny_model, block_size=8, cap=2)
    key = TinyGraphKey(8, 8, False, True, torch.float32)
    cache.capture(key)
    before_sleep = cache.pointer_fingerprint(key)
    assert before_sleep == cache.pointer_fingerprint(key)
    with torch.no_grad():
        replacement = torch.nn.Parameter(tiny_model.denoise.weight.detach().clone())
    tiny_model.denoise.weight = replacement
    assert not cache.validate_weight_pointers_or_recapture(key)
    assert cache.evictions == 1
    assert before_sleep != cache.pointer_fingerprint(key)


def test_cache_cap_eviction_and_shape_failures(tiny_model):
    cache = TinyGraphCache(tiny_model, block_size=8, cap=2)
    keys = [
        TinyGraphKey(1, 8, False, True, torch.float32),
        TinyGraphKey(4, 8, False, True, torch.float32),
        TinyGraphKey(8, 8, True, True, torch.float32),
    ]
    for key in keys:
        cache.capture(key)
    assert len(cache.entries) == 2
    assert cache.evictions == 1
    with pytest.raises(ValueError, match="token_shape_miss"):
        cache.replay(keys[-1], torch.zeros(8, device="cuda", dtype=torch.long), None)
    with pytest.raises(ValueError, match="self_conditioning_shape_miss"):
        cache.replay(
            keys[-1],
            torch.zeros(64, device="cuda", dtype=torch.long),
            torch.zeros(1, 32, device="cuda"),
        )
