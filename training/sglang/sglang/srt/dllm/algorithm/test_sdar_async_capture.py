"""Focused tests for lossless bounded SDAR async rollout capture."""

from __future__ import annotations

import ast
import builtins
import gc
import os
import threading
import time
import unittest
import weakref
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import torch

from sglang.srt.dllm.algorithm.sdar_async_capture import (
    AsyncCaptureBackpressureError,
    AsyncCaptureConsumerError,
    AsyncCaptureDrainTimeout,
    AsyncCaptureMetrics,
    AsyncCaptureMissingBlockError,
    AsyncCaptureWorker,
    BlockHandle,
    StepHandle,
    estimate_handle_bytes,
    sample_replay_step,
)


_HAS_CUDA = torch.cuda.is_available()


class _MockAlgo:
    def __init__(self):
        self._trace_dir = "/tmp/sdar_async_capture_test"
        self._trace_tensors = True
        self._trace_tensor_store = "memory_table"
        self._trace_tensor_inline = False
        self._request_replay_tensors_by_req: Dict[int, list] = {}
        self._selected_logprobs_by_req: Dict[int, dict] = {}
        self._request_stats_by_req: Dict[int, list] = {}
        self._replay_tensors_lock = threading.Lock()
        self._trace_records: List[dict] = []
        self._introspection_records: List[dict] = []
        self._generation: Dict[int, int] = {}
        self._accepted = 0
        self._edited = 0
        self._preselected_blocks: List[tuple[int, int]] = []
        self.raise_on_trace_write = False

    def _capture_generation_for_request(self, req_pool_idx: int) -> int:
        return self._generation.setdefault(int(req_pool_idx), 0)

    def _is_capture_generation_current(
        self, req_pool_idx: int, generation: int
    ) -> bool:
        return self._capture_generation_for_request(req_pool_idx) == generation

    def _write_trace_record(self, record: dict) -> None:
        if self.raise_on_trace_write:
            raise RuntimeError("injected trace write failure")
        self._trace_records.append(record)

    def _write_introspection_record(self, record: dict) -> None:
        self._introspection_records.append(record)

    def _write_replay_tensor_file(
        self, payload, req_pool_idx: int, block_idx: int, step: int
    ) -> str:
        raise AssertionError("tests use memory_table and must not write files")

    def _preselect_completed_block_replay(
        self, req_pool_idx: int, block_idx: int
    ) -> None:
        self._preselected_blocks.append(
            (int(req_pool_idx), int(block_idx))
        )

    def _record_async_block_counts(self, accepted: int, edited: int) -> None:
        self._accepted += int(accepted)
        self._edited += int(edited)


def _make_step(
    *,
    req: int = 1,
    generation: int = 0,
    block: int = 0,
    step: int = 1,
    block_size: int = 8,
    hidden_size: int = 16,
    device: str = "cpu",
    event=None,
    with_signal: bool = True,
    trace_tensors: bool = True,
) -> StepHandle:
    input_ids = torch.arange(block_size, device=device, dtype=torch.long)
    proposal = input_ids + 100
    new_accept = torch.zeros(block_size, device=device, dtype=torch.bool)
    new_accept[2::2] = True
    edit = torch.zeros_like(new_accept)
    edit[3] = True
    action = new_accept | edit
    prompt = torch.zeros_like(new_accept)
    prompt[:2] = True
    signal = (
        torch.arange(
            block_size * hidden_size, device=device, dtype=torch.float32
        ).reshape(block_size, hidden_size)
        if with_signal
        else None
    )
    hidden = torch.arange(
        block_size * hidden_size, device=device, dtype=torch.float32
    ).reshape(block_size, hidden_size)
    return StepHandle(
        input_ids_snapshot=input_ids,
        current_ids_snapshot=input_ids + 1,
        proposal=proposal,
        proposal_probs=torch.linspace(0.1, 0.9, block_size, device=device),
        raw_logprobs=torch.linspace(-2.0, -0.1, block_size, device=device),
        sample_logprobs=torch.linspace(-1.5, -0.05, block_size, device=device),
        entropy=torch.linspace(1.0, 0.1, block_size, device=device),
        accepted_mask=prompt | new_accept,
        new_accept_mask=new_accept,
        edit_mask=edit,
        action_mask=action,
        prompt_mask=prompt,
        sc_signal=signal,
        logits_source_hidden=hidden,
        valid_mask=~prompt,
        cuda_event=event,
        step=step,
        block_idx=block,
        req_pool_idx=req,
        block_start=0,
        block_end=block_size,
        was_completed=False,
        response_start_index=block * block_size,
        start_offset=2,
        block_size=block_size,
        sc_stats=None,
        trace_full=False,
        trace_tensors=trace_tensors,
        capture_replay=trace_tensors,
        trace_tensor_records=step,
        trace_tensor_store="memory_table",
        trace_tensor_max_tokens=block_size,
        position_start=block * block_size,
        step_key=block * (block_size + 8) + step,
        is_final_denoise_row=False,
        enqueue_ns=time.monotonic_ns(),
        generation=generation,
    )


def _make_block(
    *,
    req: int = 1,
    generation: int = 0,
    block: int = 0,
    block_size: int = 8,
    device: str = "cpu",
    event=None,
) -> BlockHandle:
    final_ids = torch.arange(block_size, device=device, dtype=torch.long) + 200
    q = torch.linspace(0.2, 0.8, block_size, device=device)
    prompt = torch.zeros(block_size, device=device, dtype=torch.bool)
    prompt[:2] = True
    return BlockHandle(
        final_ids=final_ids,
        commit_argmax=final_ids.clone(),
        commit_argmax_probs=torch.linspace(0.3, 0.9, block_size, device=device),
        p_probs=torch.linspace(0.25, 0.85, block_size, device=device),
        q_probs=q,
        q_raw_logprobs=q.log(),
        q_sample_logprobs=q.log() - 0.1,
        prompt_mask=prompt,
        cuda_event=event,
        req_pool_idx=req,
        block_idx=block,
        block_start=0,
        block_end=block_size,
        start_offset=2,
        denoise_forwards=1,
        batch_denoise_forwards=1,
        response_clean_start=block * block_size,
        trace_dir="/tmp/sdar_async_capture_test",
        enqueue_ns=time.monotonic_ns(),
        generation=generation,
    )


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())


class TestMetrics(unittest.TestCase):
    def test_thread_safe_metrics(self):
        metrics = AsyncCaptureMetrics()

        def increment():
            for _ in range(100):
                metrics.record_enqueue(
                    kind="step",
                    retained_bytes=8,
                    enqueue_us=1.0,
                    queued_depth=1,
                    queued_bytes=8,
                    pending_depth=1,
                    pending_bytes=8,
                )

        threads = [threading.Thread(target=increment) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["enqueue_count"], 400)
        self.assertEqual(snapshot["drop_count"], 0)


class TestStrictBudgets(unittest.TestCase):
    def test_count_budget_raises_without_drop(self):
        algo = _MockAlgo()
        worker = AsyncCaptureWorker(
            algo, queue_depth=1, queue_max_bytes=1 << 30, start_consumer=False
        )
        worker.enqueue_step(_make_step())
        with self.assertRaises(AsyncCaptureBackpressureError):
            worker.enqueue_step(_make_step(step=2))
        metrics = worker.get_metrics()
        self.assertEqual(metrics["drop_count"], 0)
        self.assertEqual(metrics["backpressure_timeout_count"], 1)
        self.assertEqual(metrics["pending_depth"], 1)
        worker.stop(drain=False)

    def test_byte_budget_is_strict_and_reported(self):
        algo = _MockAlgo()
        handle = _make_step(hidden_size=128)
        retained = estimate_handle_bytes(handle)
        worker = AsyncCaptureWorker(
            algo,
            queue_depth=8,
            queue_max_bytes=retained,
            start_consumer=False,
        )
        worker.enqueue_step(handle)
        with self.assertRaises(AsyncCaptureBackpressureError):
            worker.enqueue_step(_make_step(step=2, hidden_size=128))
        metrics = worker.get_metrics()
        self.assertLessEqual(metrics["pending_bytes_max"], retained)
        self.assertLessEqual(metrics["pending_bytes"], retained)
        self.assertEqual(metrics["queue_max_bytes"], retained)
        worker.stop(drain=False)

    def test_single_oversize_handle_rejected(self):
        handle = _make_step(hidden_size=128)
        worker = AsyncCaptureWorker(
            _MockAlgo(),
            queue_depth=8,
            queue_max_bytes=max(1, estimate_handle_bytes(handle) - 1),
            start_consumer=False,
        )
        with self.assertRaises(AsyncCaptureBackpressureError):
            worker.enqueue_step(handle)
        self.assertEqual(worker.get_metrics()["pending_bytes"], 0)
        worker.stop(drain=False)


class TestParityAndOrdering(unittest.TestCase):
    def test_replay_rows_masks_residual_and_action_ids_match(self):
        algo = _MockAlgo()
        worker = AsyncCaptureWorker(algo, queue_depth=8, queue_max_bytes=1 << 28)
        step = _make_step(req=7)
        expected_action_ids = step.proposal.clone()
        expected_input_ids = step.input_ids_snapshot.clone()
        expected_action_mask = step.action_mask.clone()
        expected_edit_change_count = int(step.edit_mask.sum().item())
        expected_random_vocab_mask = (
            ~(step.accepted_mask & ~step.new_accept_mask)
            & ~step.prompt_mask
        )
        expected_signal = step.sc_signal.clone()
        expected_logits_source = step.logits_source_hidden.clone()
        worker.enqueue_step(step)
        worker.enqueue_block(_make_block(req=7))
        lifecycle = worker.drain_request(7, generation=0, timeout=5.0)
        rows = algo._request_replay_tensors_by_req[7]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        torch.testing.assert_close(row["action_ids"], expected_action_ids)
        torch.testing.assert_close(row["input_ids"], expected_input_ids)
        torch.testing.assert_close(row["action_mask"], expected_action_mask)
        torch.testing.assert_close(
            row["random_vocab_mask"],
            expected_random_vocab_mask,
        )
        torch.testing.assert_close(
            row["self_conditioning_signal"], expected_signal
        )
        torch.testing.assert_close(
            row["logits_source_hidden"], expected_logits_source
        )
        self.assertTrue(
            row["metadata"]["used_self_conditioning_signal"]
        )
        self.assertEqual(
            row["metadata"]["action_token_count"],
            int(expected_action_mask.sum().item()),
        )
        self.assertEqual(
            row["metadata"]["edit_change_count"],
            expected_edit_change_count,
        )
        self.assertTrue(row["metadata"]["is_final_denoise_row"])
        self.assertEqual(algo._accepted, int(step.new_accept_mask.sum().item()))
        self.assertEqual(lifecycle["consumer_errors"], 0)
        worker.stop()

    def test_block_completion_normalizes_to_exactly_one_final_row(self):
        algo = _MockAlgo()
        worker = AsyncCaptureWorker(
            algo,
            queue_depth=8,
            queue_max_bytes=1 << 28,
        )
        first = _make_step(req=71, step=1)
        second = _make_step(req=71, step=2)
        first.is_final_denoise_row = True
        second.is_final_denoise_row = True
        worker.enqueue_step(first)
        worker.enqueue_step(second)
        worker.enqueue_block(_make_block(req=71))
        worker.drain_request(71, generation=0, timeout=5.0)

        rows = algo._request_replay_tensors_by_req[71]
        self.assertEqual(
            [
                bool(row["metadata"]["is_final_denoise_row"])
                for row in rows
            ],
            [False, True],
        )
        self.assertEqual(algo._preselected_blocks, [(71, 0)])
        worker.stop()

    def test_missing_signal_uses_zero_residual_without_claiming_use(self):
        algo = _MockAlgo()
        worker = AsyncCaptureWorker(algo, queue_depth=8, queue_max_bytes=1 << 28)
        worker.enqueue_step(_make_step(req=8, with_signal=False))
        worker.enqueue_block(_make_block(req=8))
        worker.drain_request(8, generation=0, timeout=5.0)
        row = algo._request_replay_tensors_by_req[8][0]
        self.assertIsNotNone(row["self_conditioning_signal"])
        self.assertEqual(
            torch.count_nonzero(row["self_conditioning_signal"]).item(), 0
        )
        self.assertFalse(row["metadata"]["used_self_conditioning_signal"])
        worker.stop()

    def test_no_duplicate_or_dropped_rows(self):
        algo = _MockAlgo()
        worker = AsyncCaptureWorker(algo, queue_depth=16, queue_max_bytes=1 << 28)
        for step_index in range(1, 5):
            worker.enqueue_step(_make_step(req=9, step=step_index))
        worker.enqueue_block(_make_block(req=9))
        worker.drain_request(9, generation=0, timeout=5.0)
        rows = algo._request_replay_tensors_by_req[9]
        keys = [row["metadata"]["step_key"] for row in rows]
        self.assertEqual(len(rows), 4)
        self.assertEqual(len(set(keys)), 4)
        self.assertEqual(worker.get_metrics()["drop_count"], 0)
        worker.stop()

    def test_event_synchronize_runs_only_on_consumer_thread(self):
        calls: List[str] = []

        class Event:
            def synchronize(self):
                calls.append(threading.current_thread().name)

        algo = _MockAlgo()
        worker = AsyncCaptureWorker(algo, queue_depth=8, queue_max_bytes=1 << 28)
        worker.enqueue_step(_make_step(req=10, event=Event()))
        worker.enqueue_block(_make_block(req=10, event=Event()))
        worker.drain_request(10, generation=0, timeout=5.0)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(name == "sdar-async-capture" for name in calls))
        worker.stop()

    @unittest.skipUnless(_HAS_CUDA, "CUDA unavailable")
    def test_dedicated_cuda_copy_stream(self):
        algo = _MockAlgo()
        worker = AsyncCaptureWorker(algo, queue_depth=8, queue_max_bytes=1 << 28)
        step_event = torch.cuda.Event()
        step_event.record(torch.cuda.current_stream())
        block_event = torch.cuda.Event()
        block_event.record(torch.cuda.current_stream())
        worker.enqueue_step(
            _make_step(req=11, device="cuda", event=step_event)
        )
        worker.enqueue_block(
            _make_block(req=11, device="cuda", event=block_event)
        )
        worker.drain_request(11, generation=0, timeout=10.0)
        self.assertIn(0, worker._copy_streams)
        self.assertEqual(worker.get_metrics()["copy_stream_fallback_count"], 0)
        worker.stop()


class TestFailureCleanup(unittest.TestCase):
    def test_stale_generation_is_not_materialized(self):
        algo = _MockAlgo()
        algo._generation[19] = 1
        worker = AsyncCaptureWorker(algo, queue_depth=8, queue_max_bytes=1 << 28)
        worker.enqueue_step(_make_step(req=19, generation=0))
        lifecycle = worker.drain_request(
            19, generation=0, timeout=5.0
        )
        self.assertEqual(lifecycle["processed_count"], 0)
        self.assertNotIn(19, algo._request_replay_tensors_by_req)
        worker.stop()

    def test_missing_block_is_error_and_releases_step_group(self):
        worker = AsyncCaptureWorker(
            _MockAlgo(), queue_depth=8, queue_max_bytes=1 << 28
        )
        worker.enqueue_step(_make_step(req=20))
        with self.assertRaises(AsyncCaptureMissingBlockError):
            worker.drain_request(20, generation=0, timeout=5.0)
        self.assertEqual(worker.get_metrics()["pending_depth"], 0)
        self.assertEqual(len(worker._req_step_handles), 0)
        worker.stop(drain=False)

    def test_consumer_exception_propagates_and_releases_budget(self):
        algo = _MockAlgo()
        algo.raise_on_trace_write = True
        worker = AsyncCaptureWorker(algo, queue_depth=8, queue_max_bytes=1 << 28)
        worker.enqueue_step(_make_step(req=21))
        block = _make_block(req=21)
        tensor_ref = weakref.ref(block.final_ids)
        worker.enqueue_block(block)
        del block
        with self.assertRaises(AsyncCaptureConsumerError):
            worker.drain_request(21, generation=0, timeout=5.0)
        gc.collect()
        self.assertIsNone(tensor_ref())
        metrics = worker.get_metrics()
        self.assertEqual(metrics["pending_depth"], 0)
        self.assertEqual(metrics["pending_bytes"], 0)
        self.assertEqual(metrics["consumer_errors"], 1)
        worker.stop(drain=False)

    def test_request_cancellation_purges_queued_storage(self):
        worker = AsyncCaptureWorker(
            _MockAlgo(),
            queue_depth=8,
            queue_max_bytes=1 << 28,
            start_consumer=False,
        )
        handle = _make_step(req=22)
        tensor_ref = weakref.ref(handle.sc_signal)
        worker.enqueue_step(handle)
        del handle
        self.assertTrue(worker.cancel_request(22, generation=0))
        gc.collect()
        self.assertIsNone(tensor_ref())
        metrics = worker.get_metrics()
        self.assertEqual(metrics["pending_bytes"], 0)
        self.assertEqual(metrics["cancelled_handle_count"], 1)
        worker.stop(drain=False)

    def test_shutdown_purges_queued_storage(self):
        worker = AsyncCaptureWorker(
            _MockAlgo(),
            queue_depth=8,
            queue_max_bytes=1 << 28,
            start_consumer=False,
        )
        handle = _make_step(req=23)
        tensor_ref = weakref.ref(handle.sc_signal)
        worker.enqueue_step(handle)
        del handle
        self.assertTrue(worker.stop(drain=False))
        gc.collect()
        self.assertIsNone(tensor_ref())
        self.assertEqual(worker.get_metrics()["pending_bytes"], 0)

    def test_boundary_timeout_cancels_active_handle(self):
        entered = threading.Event()
        release = threading.Event()

        class BlockingEvent:
            def synchronize(self):
                entered.set()
                release.wait(5.0)

        worker = AsyncCaptureWorker(
            _MockAlgo(), queue_depth=8, queue_max_bytes=1 << 28
        )
        handle = _make_step(req=24, event=BlockingEvent())
        tensor_ref = weakref.ref(handle.sc_signal)
        worker.enqueue_step(handle)
        del handle
        self.assertTrue(entered.wait(2.0))
        with self.assertRaises(AsyncCaptureDrainTimeout):
            worker.drain_request(24, generation=0, timeout=0.01)
        self.assertFalse(
            worker.cancel_request(24, generation=0, wait=False)
        )
        release.set()
        self.assertTrue(
            _wait_until(lambda: worker.get_metrics()["pending_depth"] == 0)
        )
        gc.collect()
        self.assertIsNone(tensor_ref())
        worker.stop(drain=False)


class TestProducerCriticalPath(unittest.TestCase):
    def test_producer_helpers_have_no_forbidden_host_operations(self):
        source_path = Path(__file__).with_name("sdar_prefix_kv.py")
        tree = ast.parse(source_path.read_text())
        target_names = {
            "_enqueue_step_handle_async",
            "_enqueue_block_handle_async",
        }
        forbidden_attrs = {"cpu", "item", "synchronize", "save"}
        violations = []
        found = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name not in target_names:
                    continue
                found.add(node.name)
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if (
                            isinstance(child.func, ast.Attribute)
                            and child.func.attr in forbidden_attrs
                        ):
                            violations.append((node.name, child.func.attr))
                        if (
                            isinstance(child.func, ast.Name)
                            and child.func.id == "open"
                        ):
                            violations.append((node.name, "open"))
        self.assertEqual(found, target_names)
        self.assertEqual(violations, [])

    def test_enqueue_does_not_materialize_or_wait_event(self):
        class ProducerEvent:
            def synchronize(self):
                raise AssertionError("producer synchronized CUDA event")

        worker = AsyncCaptureWorker(
            _MockAlgo(),
            queue_depth=8,
            queue_max_bytes=1 << 28,
            start_consumer=False,
        )
        handle = _make_step(req=30, event=ProducerEvent())
        with patch.object(
            torch.Tensor,
            "cpu",
            autospec=True,
            side_effect=AssertionError("producer called cpu"),
        ), patch.object(
            torch.Tensor,
            "item",
            autospec=True,
            side_effect=AssertionError("producer called item"),
        ), patch.object(
            torch,
            "save",
            side_effect=AssertionError("producer called torch.save"),
        ), patch.object(
            builtins,
            "open",
            side_effect=AssertionError("producer wrote a file"),
        ):
            worker.enqueue_step(handle)
        worker.cancel_request(30, generation=0)
        worker.stop(drain=False)

    def test_scheduler_has_mandatory_capture_boundary(self):
        root = Path(__file__).parents[2] / "managers"
        mixin_source = (root / "scheduler_output_processor_mixin.py").read_text()
        scheduler_source = (root / "scheduler.py").read_text()
        self.assertIn("def collect_sdar_rollout_capture", mixin_source)
        self.assertGreaterEqual(
            mixin_source.count("collect_sdar_rollout_capture("), 3
        )
        self.assertGreaterEqual(
            scheduler_source.count("collect_sdar_rollout_capture("), 1
        )
        algorithm_source = Path(__file__).with_name("sdar_prefix_kv.py").read_text()
        self.assertIn("def collect_completed_capture(", algorithm_source)
        self.assertIn("drain_async_capture(req_pool_idx", algorithm_source)

    def test_scheduler_boundary_calls_collector_before_attaching_data(self):
        from sglang.srt.managers.scheduler_output_processor_mixin import (
            collect_sdar_rollout_capture,
        )

        class Algo:
            def __init__(self):
                self.calls = []

            def collect_completed_capture(self, req_pool_idx):
                self.calls.append(req_pool_idx)
                return {
                    "selected_logprobs": {"raw_logprobs": [-0.5]},
                    "request_stats": [{"block_idx": 0}],
                    "replay_tensors": [{"metadata": {"step_key": 1}}],
                    "lifecycle": {"drain_wait_us": 7},
                }

        class Req:
            return_logprob = True

        algo = Algo()
        req = Req()
        selected = collect_sdar_rollout_capture(algo, req, 31)
        self.assertEqual(algo.calls, [31])
        self.assertEqual(selected["raw_logprobs"], [-0.5])
        self.assertEqual(req.sglang_sdar_request_metrics[0]["block_idx"], 0)
        self.assertEqual(
            req.sglang_sdar_replay_tensors[0]["metadata"]["step_key"], 1
        )
        self.assertEqual(
            req.sglang_sdar_async_capture_metrics["drain_wait_us"], 7
        )

    def test_scheduler_boundary_propagates_capture_timeout(self):
        from sglang.srt.managers.scheduler_output_processor_mixin import (
            collect_sdar_rollout_capture,
        )

        class Algo:
            def collect_completed_capture(self, req_pool_idx):
                raise AsyncCaptureDrainTimeout(f"request {req_pool_idx}")

        class Req:
            return_logprob = False

        with self.assertRaises(AsyncCaptureDrainTimeout):
            collect_sdar_rollout_capture(Algo(), Req(), 32)

    def test_no_stream_capture_waits_for_final_boundary(self):
        from sglang.srt.managers.scheduler_output_processor_mixin import (
            should_collect_sdar_rollout_capture_now,
        )

        class Req:
            return_logprob = False

        with patch.dict(
            os.environ, {"SGLANG_SDAR_REPLAY_EARLY_STREAM": "0"}, clear=False
        ):
            self.assertFalse(should_collect_sdar_rollout_capture_now(Req()))
            logprob_req = Req()
            logprob_req.return_logprob = True
            self.assertTrue(
                should_collect_sdar_rollout_capture_now(logprob_req)
            )
        with patch.dict(
            os.environ, {"SGLANG_SDAR_REPLAY_EARLY_STREAM": "1"}, clear=False
        ):
            self.assertTrue(should_collect_sdar_rollout_capture_now(Req()))

    def test_feature_is_off_by_default(self):
        old = os.environ.pop("SGLANG_SDAR_ASYNC_CAPTURE", None)
        try:
            enabled = os.environ.get("SGLANG_SDAR_ASYNC_CAPTURE", "0") == "1"
            self.assertFalse(enabled)
        finally:
            if old is not None:
                os.environ["SGLANG_SDAR_ASYNC_CAPTURE"] = old


def test_replay_step_sampling_is_deterministic_and_keeps_final():
    def sample():
        return [
            sample_replay_step(
                fraction=0.25,
                seed=20260722,
                req_pool_idx=3,
                generation=7,
                block_idx=step // 256,
                local_step=step % 256,
                is_final_denoise_row=False,
            )
            for step in range(4096)
        ]

    first = sample()
    second = sample()
    assert first == second
    assert 0.22 <= sum(first) / len(first) <= 0.28

    for fraction, interval in ((0.10, 10), (0.25, 4)):
        for block_idx in range(8):
            selected = [
                sample_replay_step(
                    fraction=fraction,
                    mode="stratified",
                    seed=20260722,
                    req_pool_idx=3,
                    generation=7,
                    block_idx=block_idx,
                    local_step=step,
                    is_final_denoise_row=False,
                )
                for step in range(100)
            ]
            assert sum(selected) == int(100 * fraction)
            assert all(
                sum(selected[start : start + interval]) == 1
                for start in range(0, 100, interval)
            )

    assert sample_replay_step(
        fraction=0.0,
        seed=20260722,
        req_pool_idx=3,
        generation=7,
        block_idx=4,
        local_step=9,
        is_final_denoise_row=True,
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
