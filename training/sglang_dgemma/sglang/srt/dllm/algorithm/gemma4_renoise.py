"""Uniform-state (renoising) block-diffusion algorithm for DiffusionGemma.

Two phases, distinguished by `forward_batch.dllm_is_encoder`:
  * encoder/prefill: one forward to write the context KV cache (no denoising).
  * decoder/denoise: for one canvas, run `max_denoising_steps` reverse steps,
    feeding the previous step's logits back as self-conditioning.

Ported from the HF `diffusion_gemma` EntropyBoundSampler + linear temperature
schedule + StableAndConfident stopping. Defaults follow the checkpoint's
generation_config.json and may be overridden via --dllm-algorithm-config.
"""

import logging
import os
import time
from typing import List, Tuple, Union

import torch

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.algorithm.gemma_async_capture import (
    GemmaAsyncReplayCaptureWorker,
    GemmaReplayBlockHandle,
    GemmaReplayStepHandle,
    build_gemma_intermediate_replay_item,
)
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class Gemma4Renoise(DllmAlgorithm):
    def __init__(self, config: DllmConfig):
        super().__init__(config)
        self.canvas_length = config.block_size

        ac = config.algorithm_config or {}
        self.max_denoising_steps = ac.get("max_denoising_steps", 48)
        if self.max_denoising_steps < 1:
            raise ValueError("max_denoising_steps must be >= 1")
        s = ac.get("sampler_config", {})
        self.entropy_bound = s.get("entropy_bound", 0.1)
        t = ac.get("temperature_schedule", {})
        self.t_min = t.get("t_min", 0.4)
        self.t_max = t.get("t_max", 0.8)
        st = ac.get("stopping_config", {})
        self.confidence_threshold = st.get("confidence_threshold", 0.005)
        self.stability_threshold = st.get("stability_threshold", 1)
        self._accepted_mask = None

        self.vocab_size = None
        self._base_seed = ac.get("seed", None)
        self._generator = None
        replay_env = os.environ.get(
            "SGLANG_GEMMA_INTERMEDIATE_REPLAY",
            os.environ.get("SGLANG_SDAR_TRACE_TENSORS", "0"),
        )
        replay_default = str(replay_env).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        replay_enabled = ac.get("intermediate_replay", replay_default)
        self._intermediate_replay_enabled = (
            str(replay_enabled).strip().lower()
            in {"1", "true", "yes", "on"}
            if isinstance(replay_enabled, str)
            else bool(replay_enabled)
        )
        self._replay_capture_worker = None
        self._replay_capture_drain_timeout_s = 30.0
        if self._intermediate_replay_enabled:
            async_config = ac.get("intermediate_replay_async", {}) or {}
            if not isinstance(async_config, dict):
                raise ValueError(
                    "intermediate_replay_async must be a configuration mapping"
                )
            queue_depth = int(
                async_config.get(
                    "queue_depth",
                    os.environ.get(
                        "SGLANG_GEMMA_REPLAY_ASYNC_QUEUE_DEPTH", "64"
                    ),
                )
            )
            queue_max_bytes = int(
                async_config.get(
                    "queue_max_bytes",
                    os.environ.get(
                        "SGLANG_GEMMA_REPLAY_ASYNC_QUEUE_MAX_BYTES",
                        str(512 * 1024 * 1024),
                    ),
                )
            )
            backpressure_policy = str(
                async_config.get(
                    "backpressure_policy",
                    os.environ.get(
                        "SGLANG_GEMMA_REPLAY_ASYNC_BACKPRESSURE", "error"
                    ),
                )
            ).strip().lower()
            enqueue_timeout_s = float(
                async_config.get(
                    "enqueue_timeout_s",
                    os.environ.get(
                        "SGLANG_GEMMA_REPLAY_ASYNC_ENQUEUE_TIMEOUT_S", "0"
                    ),
                )
            )
            self._replay_capture_drain_timeout_s = float(
                async_config.get(
                    "drain_timeout_s",
                    os.environ.get(
                        "SGLANG_GEMMA_REPLAY_ASYNC_DRAIN_TIMEOUT_S", "30"
                    ),
                )
            )
            if self._replay_capture_drain_timeout_s <= 0:
                raise ValueError(
                    "Gemma intermediate replay drain_timeout_s must be positive"
                )
            self._replay_capture_worker = GemmaAsyncReplayCaptureWorker(
                queue_depth=queue_depth,
                queue_max_bytes=queue_max_bytes,
                backpressure_policy=backpressure_policy,
                enqueue_timeout_s=enqueue_timeout_s,
            )
        self._replay_block_idx_by_req: dict[int, int] = {}
        self._stats = {
            "total_forwards": 0,
            "denoise_forwards": 0,
            "commit_forwards": 0,
            "accepted_tokens": 0,
            "new_accept_tokens": 0,
            "edit_change_tokens": 0,
            "total_tokens": 0,
            "generated_tokens": 0,
            "replay_captured_blocks": 0,
            "replay_captured_rows": 0,
            "replay_captured_token_steps": 0,
            "replay_captured_bytes": 0,
            "replay_producer_peak_bytes": 0,
        }

    def get_stats(self) -> dict:
        return dict(self._stats)

    def _temperature(self, step: int) -> float:
        return self.t_min + (self.t_max - self.t_min) * (
            step / self.max_denoising_steps
        )

    def _seed_generator(self, device) -> torch.Generator:
        # One generator drives the whole decode so every canvas draw is reproducible
        # (when `seed` is set) and, under TP, identical across ranks: each rank samples
        # the canvas from all-gathered logits, so divergent draws would corrupt
        # attention. Re-seed per decode and broadcast rank 0's seed so ranks agree.
        if self._generator is None or self._generator.device != device:
            self._generator = torch.Generator(device=device)
        if self._base_seed is not None:
            seed = int(self._base_seed)
        else:
            seed = int(torch.randint(0, torch.iinfo(torch.int32).max, ()).item())
        if get_tensor_model_parallel_world_size() > 1:
            t = torch.tensor(
                [seed if get_tensor_model_parallel_rank() == 0 else 0],
                dtype=torch.int64,
                device=device,
            )
            seed = int(tensor_model_parallel_all_reduce(t).item())
        self._generator.manual_seed(seed)
        return self._generator

    def _init_canvas(self, shape, device, gen) -> torch.Tensor:
        return torch.randint(
            low=0, high=self.vocab_size, size=shape, device=device, generator=gen
        )

    def _accept(self, current, denoiser, token_entropy) -> torch.Tensor:
        # EntropyBound: accept the lowest-entropy positions within entropy_bound.
        # Store the mask so _renoise re-noises exactly its complement.
        sorted_e, idx = torch.sort(token_entropy, dim=-1, descending=False)
        cum = torch.cumsum(sorted_e, dim=-1)
        sel = (cum - sorted_e) <= self.entropy_bound
        self._accepted_mask = torch.zeros_like(sel).scatter(-1, idx, sel)
        return torch.where(self._accepted_mask, denoiser, current)

    def _renoise(self, accepted, gen) -> torch.Tensor:
        # Re-noise every non-accepted position with a fresh uniform token.
        rand = self._init_canvas(accepted.shape, accepted.device, gen)
        return torch.where(self._accepted_mask, accepted, rand)

    def _stop(self, history, argmax, token_entropy) -> torch.Tensor:
        # Per-request [bs] bool: each request stops on its own stability/confidence.
        bs = argmax.shape[0]
        if len(history) == self.stability_threshold:
            stable = torch.ones(bs, dtype=torch.bool, device=argmax.device)
            for c in history:
                stable &= (argmax == c).all(dim=1)
        else:
            stable = torch.zeros(bs, dtype=torch.bool, device=argmax.device)
        history.append(argmax)
        if len(history) > self.stability_threshold:
            history.pop(0)
        confident = token_entropy.mean(dim=1) < self.confidence_threshold
        return stable & confident

    @staticmethod
    def _req_pool_ids(forward_batch: ForwardBatch) -> list[int]:
        req_pool_indices = getattr(forward_batch, "req_pool_indices", None)
        if not isinstance(req_pool_indices, torch.Tensor):
            raise RuntimeError(
                "DiffusionGemma intermediate replay requires req_pool_indices"
            )
        return [
            int(value)
            for value in req_pool_indices.detach().cpu().long().view(-1).tolist()
        ]

    @staticmethod
    def _set_model_replay_capture(model_runner: ModelRunner, enabled: bool) -> None:
        setter = getattr(
            getattr(model_runner, "model", None),
            "set_intermediate_replay_capture",
            None,
        )
        if not callable(setter):
            if enabled:
                raise RuntimeError(
                    "DiffusionGemma model lacks compact self-conditioning capture API"
                )
            return
        setter(enabled)

    @staticmethod
    def _last_model_self_conditioning(model_runner: ModelRunner):
        getter = getattr(
            getattr(model_runner, "model", None),
            "get_last_self_conditioning_signal",
            None,
        )
        if not callable(getter):
            raise RuntimeError(
                "DiffusionGemma model lacks compact self-conditioning capture API"
            )
        signal, present = getter()
        if not isinstance(signal, torch.Tensor):
            raise RuntimeError(
                "DiffusionGemma forward did not expose consumed self-conditioning"
            )
        return signal, bool(present)

    def close_intermediate_replay_capture(self) -> None:
        worker = self._replay_capture_worker
        if worker is not None:
            worker.stop(drain=False)
            self._replay_capture_worker = None

    def _build_intermediate_replay_item(
        self,
        *,
        req_pool_idx: int,
        block_index: int,
        target_ids: torch.Tensor,
        step_indices: list[int],
        canvases: list[torch.Tensor],
        self_conditioning: list[torch.Tensor],
        self_conditioning_present: list[bool],
        done_after_step: list[bool],
        steps_used: int,
        capture_started_wall_ts: float,
        capture_seconds: float,
        capture_buffer_bytes: int,
    ) -> dict:
        return build_gemma_intermediate_replay_item(
            canvas_length=self.canvas_length,
            req_pool_idx=req_pool_idx,
            block_index=block_index,
            target_ids=target_ids,
            step_indices=step_indices,
            canvases=canvases,
            self_conditioning=self_conditioning,
            self_conditioning_present=self_conditioning_present,
            done_after_step=done_after_step,
            steps_used=steps_used,
            capture_started_wall_ts=capture_started_wall_ts,
            capture_seconds=capture_seconds,
            capture_buffer_bytes=capture_buffer_bytes,
        )

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        if self.vocab_size is None:
            self.vocab_size = model_runner.model_config.hf_config.text_config.vocab_size

        if forward_batch.dllm_is_encoder:
            if self._intermediate_replay_enabled:
                for req_pool_idx in self._req_pool_ids(forward_batch):
                    self._replay_block_idx_by_req[req_pool_idx] = 0
                self._set_model_replay_capture(model_runner, False)
            # Skip an empty (fully-cached) encode round, rope can't reshape 0 tokens.
            if forward_batch.input_ids.numel() == 0:
                return None, [], False
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            return out.logits_output, [], out.can_run_graph

        bs = forward_batch.batch_size
        L = self.canvas_length
        device = forward_batch.input_ids.device
        gen = self._seed_generator(device)
        req_pool_ids = (
            self._req_pool_ids(forward_batch)
            if self._intermediate_replay_enabled
            else []
        )
        if self._intermediate_replay_enabled and len(req_pool_ids) != bs:
            raise RuntimeError(
                "DiffusionGemma replay req_pool_indices/batch mismatch: "
                f"{len(req_pool_ids)} != {bs}"
            )
        block_indices = [
            int(self._replay_block_idx_by_req.get(req_pool_idx, 0))
            for req_pool_idx in req_pool_ids
        ]
        capture_started_wall_ts = time.time()
        capture_started = time.perf_counter()
        capture_worker = self._replay_capture_worker
        capture_id = None
        capture_sequence_index = 0
        if self._intermediate_replay_enabled:
            if capture_worker is None:
                raise RuntimeError(
                    "DiffusionGemma intermediate replay worker is unavailable"
                )
            capture_id = capture_worker.allocate_capture_id()
            self._set_model_replay_capture(model_runner, True)

        try:
            current = self._init_canvas((bs, L), device, gen)
            forward_batch.input_ids[:] = current.reshape(-1)
            self_cond = None
            history: List[torch.Tensor] = []
            out = None
            # Each request freezes its canvas once it stops while the rest keep denoising.
            done = torch.zeros(bs, dtype=torch.bool, device=device)
            steps_used = torch.zeros(
                bs, dtype=torch.int32, device=device
            )  # TPF logging only
            # Emit the greedy argmax (not the multinomial canvas), frozen per-item on stop.
            argmax = current.clone()

            for step in reversed(range(1, self.max_denoising_steps + 1)):
                active_before = ~done
                steps_used += active_before.int()
                pre_forward_canvas = None
                if self._intermediate_replay_enabled:
                    # Snapshot the actual decoder input immediately before forward.
                    pre_forward_canvas = (
                        forward_batch.input_ids.view(bs, L).detach().clone()
                    )
                forward_batch.dllm_self_conditioning_logits = self_cond
                out = model_runner.forward(
                    forward_batch, pp_proxy_tensors=None
                )
                consumed_signal_snapshot = None
                consumed_present = False
                expected_present = self_cond is not None
                if self._intermediate_replay_enabled:
                    consumed_signal, consumed_present = (
                        self._last_model_self_conditioning(model_runner)
                    )
                    consumed_signal_snapshot = (
                        consumed_signal.detach().clone()
                    )

                logits = (
                    out.logits_output.full_logits / self._temperature(step)
                ).view(bs, L, -1)
                token_entropy = torch.distributions.Categorical(
                    logits=logits
                ).entropy()
                probs = torch.softmax(logits, dim=-1)
                denoiser = torch.multinomial(
                    probs.reshape(bs * L, -1),
                    num_samples=1,
                    generator=gen,
                ).view(bs, L)

                keep = done.view(bs, 1)
                argmax = torch.where(
                    keep, argmax, torch.argmax(logits, dim=-1)
                )
                accepted = self._accept(current, denoiser, token_entropy)
                current = torch.where(
                    keep, current, self._renoise(accepted, gen)
                )
                forward_batch.input_ids[:] = current.reshape(-1)
                if self_cond is None:
                    self_cond = logits.reshape(bs * L, -1)
                else:
                    self_cond = torch.where(
                        done.view(bs, 1, 1),
                        self_cond.view(bs, L, -1),
                        logits,
                    ).reshape(bs * L, -1)

                newly = self._stop(history, argmax, token_entropy) & (~done)
                if bool(newly.any()):
                    done |= newly
                if self._intermediate_replay_enabled:
                    active_snapshot = active_before.detach().clone()
                    done_snapshot = done.detach().clone()
                    producer_event = capture_worker.make_producer_event(
                        pre_forward_canvas,
                        consumed_signal_snapshot,
                        active_snapshot,
                        done_snapshot,
                    )
                    capture_worker.enqueue_step(
                        GemmaReplayStepHandle(
                            capture_id=capture_id,
                            sequence_index=capture_sequence_index,
                            step_index=int(step),
                            req_pool_ids=tuple(req_pool_ids),
                            block_indices=tuple(block_indices),
                            pre_forward_canvas_ids=pre_forward_canvas,
                            self_conditioning_signal=(
                                consumed_signal_snapshot
                            ),
                            self_conditioning_present=bool(
                                consumed_present
                            ),
                            expected_self_conditioning_present=bool(
                                expected_present
                            ),
                            active_mask=active_snapshot,
                            done_after_step=done_snapshot,
                            cuda_event=producer_event,
                        )
                    )
                    capture_sequence_index += 1
                if bool(done.all()):
                    break

            # Commit: one causal re-encode of the emitted argmax canvas. Every denoise
            # forward above overwrites the canvas KV slots with K/V of the *noisy*
            # current canvas under the bidirectional mask; whatever the last step left
            # there (renoised junk included when max_denoising_steps is exhausted)
            # silently becomes the context for all later canvases and compounds into
            # degenerate text. The HF reference re-encodes the committed canvas
            # causally (encoder phase) before the next one; mirror that here.
            forward_batch.input_ids[:] = argmax.reshape(-1)
            forward_batch.dllm_self_conditioning_logits = None
            forward_batch.dllm_is_encoder = True
            try:
                model_runner.forward(forward_batch, pp_proxy_tensors=None)
            finally:
                forward_batch.dllm_is_encoder = False
            if self._intermediate_replay_enabled:
                target_snapshot = argmax.detach().clone()
                steps_used_snapshot = steps_used.detach().clone()
                block_committed_wall_ts = time.time()
                producer_event = capture_worker.make_producer_event(
                    target_snapshot, steps_used_snapshot
                )
                capture_worker.enqueue_block(
                    GemmaReplayBlockHandle(
                        capture_id=capture_id,
                        sequence_index=capture_sequence_index,
                        req_pool_ids=tuple(req_pool_ids),
                        block_indices=tuple(block_indices),
                        final_target_ids=target_snapshot,
                        steps_used=steps_used_snapshot,
                        canvas_length=int(L),
                        capture_started_wall_ts=capture_started_wall_ts,
                        capture_seconds=(
                            time.perf_counter() - capture_started
                        ),
                        block_committed_wall_ts=block_committed_wall_ts,
                        cuda_event=producer_event,
                    )
                )
                capture_sequence_index += 1
        except BaseException:
            if capture_id is not None and capture_worker is not None:
                capture_worker.cancel_capture(capture_id)
            raise
        finally:
            if self._intermediate_replay_enabled:
                try:
                    self._set_model_replay_capture(model_runner, False)
                except BaseException:
                    if capture_id is not None and capture_worker is not None:
                        capture_worker.cancel_capture(capture_id)
                    raise

        replay_items = []
        replay_rows_total = 0
        replay_token_steps_total = 0
        replay_bytes_total = 0
        replay_peak_bytes = 0
        if self._intermediate_replay_enabled:
            try:
                replay_items = capture_worker.drain_capture(
                    capture_id,
                    timeout=self._replay_capture_drain_timeout_s,
                )
            except BaseException:
                capture_worker.cancel_capture(capture_id)
                raise
            for batch_id, replay_item in enumerate(replay_items):
                replay_rows_total += int(replay_item["captured_rows"])
                replay_token_steps_total += int(
                    replay_item["captured_token_steps"]
                )
                replay_bytes_total += int(replay_item["producer_tensor_bytes"])
                replay_peak_bytes = max(
                    replay_peak_bytes,
                    int(replay_item["producer_peak_buffer_bytes"]),
                )
                self._replay_block_idx_by_req[req_pool_ids[batch_id]] = (
                    block_indices[batch_id] + 1
                )

        request_metrics = []
        denoise_forwards_total = 0
        for batch_id in range(bs):
            denoise_forwards = int(steps_used[batch_id].item())
            denoise_forwards_total += denoise_forwards
            metrics = {
                "algorithm": type(self).__name__,
                "block_size": int(L),
                "denoise_forwards": denoise_forwards,
                "commit_forwards": 1,
                "generated_tokens": int(L),
                "total_tokens": int(L),
                "true_forward_tokens": int(L) * denoise_forwards,
            }
            if self._intermediate_replay_enabled:
                replay_item = replay_items[batch_id]
                metrics.update(
                    {
                        "replay_schema": "gemma_intermediate_replay_v1",
                        "replay_block_index": int(
                            replay_item["block_index"]
                        ),
                        "replay_captured_blocks": 1,
                        "replay_captured_rows": int(
                            replay_item["captured_rows"]
                        ),
                        "replay_captured_token_steps": int(
                            replay_item["captured_token_steps"]
                        ),
                        "replay_captured_bytes": int(
                            replay_item["producer_tensor_bytes"]
                        ),
                        "replay_producer_peak_bytes": int(
                            replay_item["producer_peak_buffer_bytes"]
                        ),
                    }
                )
            request_metrics.append(metrics)
        if out.logits_output.customized_info is None:
            out.logits_output.customized_info = {}
        out.logits_output.customized_info["sglang_sdar_request_metrics"] = request_metrics
        if self._intermediate_replay_enabled:
            # One batch element per request; the scheduler's customized-info
            # collector binds each block item to the matching request.
            out.logits_output.customized_info[
                "gemma_intermediate_replay"
            ] = replay_items

        generated_tokens = int(bs) * int(L)
        commit_forwards = int(bs)
        self._stats["denoise_forwards"] += denoise_forwards_total
        self._stats["commit_forwards"] += commit_forwards
        self._stats["total_forwards"] += denoise_forwards_total + commit_forwards
        self._stats["accepted_tokens"] += generated_tokens
        self._stats["new_accept_tokens"] += generated_tokens
        self._stats["total_tokens"] += generated_tokens
        self._stats["generated_tokens"] += generated_tokens
        if self._intermediate_replay_enabled:
            self._stats["replay_captured_blocks"] += int(bs)
            self._stats["replay_captured_rows"] += replay_rows_total
            self._stats[
                "replay_captured_token_steps"
            ] += replay_token_steps_total
            self._stats["replay_captured_bytes"] += replay_bytes_total
            self._stats["replay_producer_peak_bytes"] = max(
                int(self._stats["replay_producer_peak_bytes"]),
                int(replay_peak_bytes),
            )

        logger.info(
            "[dllm-tpf] algo=%s steps=%s replay_blocks=%d replay_rows=%d "
            "replay_token_steps=%d replay_bytes=%d replay_peak_bytes=%d",
            type(self).__name__,
            steps_used.tolist(),
            int(bs) if self._intermediate_replay_enabled else 0,
            replay_rows_total,
            replay_token_steps_total,
            replay_bytes_total,
            replay_peak_bytes,
        )

        next_token_ids = [argmax[i] for i in range(bs)]
        return out.logits_output, next_token_ids, out.can_run_graph


Algorithm = Gemma4Renoise
