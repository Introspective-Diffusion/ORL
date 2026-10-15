from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.dllm.mixin.req import DllmReqPhase
from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe.routed_experts_capturer import get_global_experts_capturer
from sglang.srt.managers.io_struct import (
    AbortReq,
    BatchEmbeddingOutput,
    BatchTokenIDOutput,
)
from sglang.srt.managers.schedule_batch import (
    BaseFinishReason,
    Req,
    ScheduleBatch,
)
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.server_args import get_global_server_args

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import (
        EmbeddingBatchResult,
        GenerationBatchResult,
        ScheduleBatch,
        Scheduler,
    )

logger = logging.getLogger(__name__)

DEFAULT_FORCE_STREAM_INTERVAL = 50


_early_stream_diag_printed = False


def _sdar_replay_early_stream_enabled() -> bool:
    value = os.getenv("SGLANG_SDAR_REPLAY_EARLY_STREAM", "0").strip().lower()
    enabled = value in {"1", "true", "yes", "on"}
    global _early_stream_diag_printed
    if not _early_stream_diag_printed:
        _early_stream_diag_printed = True
        print(f"[SDAR_STREAM_DIAG] scheduler early_stream_enabled={enabled}", flush=True)
    return enabled


def should_collect_sdar_rollout_capture_now(req: Req) -> bool:
    """Whether the current block has a consumer before request completion."""
    return bool(req.return_logprob or _sdar_replay_early_stream_enabled())


def collect_sdar_rollout_capture(dllm_algo, req: Req, req_pool_idx: int):
    """Drain and atomically transfer SDAR capture data at the rollout boundary.

    Async capture failures and timeouts deliberately propagate from here.  A
    request must never continue with partial residual/replay metadata.
    """
    if dllm_algo is None:
        return None
    collector = getattr(dllm_algo, "collect_completed_capture", None)
    if collector is not None:
        capture = collector(req_pool_idx)
        selected_logprobs = capture.get("selected_logprobs")
        request_stats = capture.get("request_stats")
        replay_tensors = capture.get("replay_tensors")
        lifecycle = capture.get("lifecycle")
    else:
        selected_store = getattr(dllm_algo, "_selected_logprobs_by_req", {})
        selected_logprobs = selected_store.pop(req_pool_idx, None)
        request_stats_store = getattr(dllm_algo, "_request_stats_by_req", {})
        request_stats = request_stats_store.pop(req_pool_idx, None)
        replay_store = getattr(dllm_algo, "_request_replay_tensors_by_req", {})
        replay_tensors = replay_store.pop(req_pool_idx, None)
        lifecycle = None
        if replay_tensors:
            finalize = getattr(dllm_algo, "finalize_async_replay_tensors", None)
            if finalize is not None:
                replay_tensors = finalize(replay_tensors)

    if request_stats:
        existing_stats = getattr(req, "sglang_sdar_request_metrics", [])
        req.sglang_sdar_request_metrics = existing_stats + request_stats
    if replay_tensors:
        existing_replay = getattr(req, "sglang_sdar_replay_tensors", [])
        req.sglang_sdar_replay_tensors = existing_replay + replay_tensors
    if lifecycle:
        req.sglang_sdar_async_capture_metrics = lifecycle
    return selected_logprobs if req.return_logprob else None


def apply_dllm_carry_write(req, dllm_algo) -> None:
    """SDARPrefixKV commit_mode=fused_next: the previous block's T2T (edit) check
    ran inside this step's fused forward, after that block's tokens had already
    been written to dllm_ids / output_ids. If it changed any of them, patch both
    in place before this step's tokens are appended.

    The algorithm leaves ``(rel_start, tokens)`` in ``_fused_carry_write``:
    ``tokens`` are the previous block's final non-prompt ids (block_size -
    rel_start of them), which occupy dllm_ids[dllm_block_offset - block_size +
    rel_start : dllm_block_offset] and the last len(tokens) entries of
    output_ids (nothing else was emitted for this request in between)."""
    if dllm_algo is None:
        return
    store = getattr(dllm_algo, "_fused_carry_write", None)
    if not store:
        return
    item = store.pop(req.req_pool_idx, None)
    if item is None:
        return
    rel_start, tokens = item
    tokens = [int(t) for t in tokens]
    n = len(tokens)
    if n == 0:
        return
    block_size = req.dllm_config.block_size
    base = req.dllm_block_offset - block_size + int(rel_start)
    if req.dllm_ids and 0 <= base and base + n <= len(req.dllm_ids):
        req.dllm_ids[base : base + n] = tokens
    if len(req.output_ids) >= n:
        req.output_ids[-n:] = tokens
    else:
        logger.warning(
            "[SDAR-FUSED] carry write: output_ids shorter than carry (%d < %d), "
            "req_pool_idx=%s",
            len(req.output_ids),
            n,
            req.req_pool_idx,
        )


class SchedulerOutputProcessorMixin:
    """
    This class implements the output processing logic for Scheduler.
    We put them into a separate file to make the `scheduler.py` shorter.
    """

    def _get_storage_backend_type(self) -> str:
        """Get storage backend type from tree_cache."""
        storage_backend_type = "none"
        cache_controller = getattr(self.tree_cache, "cache_controller", None)
        if cache_controller and hasattr(cache_controller, "storage_backend"):
            storage_backend = cache_controller.storage_backend
            if storage_backend is not None:
                storage_backend_type = type(storage_backend).__name__
        return storage_backend_type

    def _get_cached_tokens_details(self, req: Req) -> Optional[dict]:
        """Get detailed cache breakdown for a request, if available.

        Returns:
            - None if HiCache is not enabled
            - {"device": X, "host": Y} if HiCache enabled but L3 storage is not
            - {"device": X, "host": Y, "storage": Z, "storage_backend": "..."} if L3 enabled
        """
        # Only show details if HiCache is enabled
        if not getattr(self, "enable_hierarchical_cache", False):
            return None

        # Only show if there are any cached tokens
        if (
            req.cached_tokens_device > 0
            or req.cached_tokens_host > 0
            or req.cached_tokens_storage > 0
        ):
            details = {
                "device": req.cached_tokens_device,
                "host": req.cached_tokens_host,
            }
            # Only include storage fields if L3 storage is enabled
            if getattr(self, "enable_hicache_storage", False):
                details["storage"] = req.cached_tokens_storage
                details["storage_backend"] = self._get_storage_backend_type()
            return details
        return None

    def process_batch_result_prebuilt(self: Scheduler, batch: ScheduleBatch):
        assert self.disaggregation_mode == DisaggregationMode.DECODE
        for req in batch.reqs:
            req.time_stats.set_decode_prebuilt_finish_time()
            req.check_finished()
            if req.finished():
                req.time_stats.set_quick_finish_time()
                release_kv_cache(req, self.tree_cache)

        # Note: Logprobs should be handled on the prefill engine.
        self.stream_output(batch.reqs, batch.return_logprob)

    def maybe_collect_routed_experts(self: Scheduler, req: Req):
        """Collect routed experts for a finished request."""
        req.routed_experts = get_global_experts_capturer().get_routed_experts(
            req_pool_idx=req.req_pool_idx,
            seqlen=req.seqlen,
            req_to_token_pool=self.req_to_token_pool,
        )

    def maybe_collect_customized_info(
        self: Scheduler, i: int, req: Req, logits_output: LogitsProcessorOutput
    ):
        if logits_output is not None and logits_output.customized_info is not None:
            if req.customized_info is None:
                req.customized_info = {}
            for k, v in logits_output.customized_info.items():
                if k not in req.customized_info:
                    req.customized_info[k] = []
                # Copy the element so it doesn't retain the entire batch
                # tensor/array via a view reference.
                elem = v[i]
                if isinstance(elem, torch.Tensor):
                    elem = elem.clone()
                elif hasattr(elem, "copy") and callable(elem.copy):
                    elem = elem.copy()
                req.customized_info[k].append(elem)

    def process_batch_result_prefill(
        self: Scheduler,
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
    ):
        skip_stream_req = None

        if self.is_generation:
            if result.copy_done is not None:
                result.copy_done.synchronize()

            (
                logits_output,
                next_token_ids,
                extend_input_len_per_req,
                extend_logprob_start_len_per_req,
            ) = (
                result.logits_output,
                result.next_token_ids,
                result.extend_input_len_per_req,
                result.extend_logprob_start_len_per_req,
            )

            # Move next_token_ids and logprobs to cpu
            next_token_ids = next_token_ids.tolist()
            if batch.return_logprob:
                if logits_output.next_token_logprobs is not None:
                    logits_output.next_token_logprobs = (
                        logits_output.next_token_logprobs.tolist()
                    )
                if logits_output.input_token_logprobs is not None:
                    logits_output.input_token_logprobs = tuple(
                        logits_output.input_token_logprobs.tolist()
                    )

            hidden_state_offset = 0

            # Check finish conditions
            logprob_pt = 0

            for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids)):
                if req.finished() or req.is_retracted:
                    # decode req in mixed batch or retracted req
                    continue

                if req.is_chunked <= 0:
                    req.time_stats.set_prefill_finished_time()

                    # req output_ids are set here
                    req.output_ids.append(next_token_id)
                    req.check_finished()

                    if req.finished():
                        self.maybe_collect_routed_experts(req)
                        release_kv_cache(req, self.tree_cache)
                        req.time_stats.set_completion_time()
                    elif not batch.decoding_reqs or req not in batch.decoding_reqs:
                        # This updates radix so others can match
                        self.tree_cache.cache_unfinished_req(req)

                    self.maybe_collect_customized_info(i, req, logits_output)

                    if batch.return_logprob:
                        assert extend_logprob_start_len_per_req is not None
                        assert extend_input_len_per_req is not None
                        extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                        extend_input_len = extend_input_len_per_req[i]

                        num_input_logprobs = self._calculate_num_input_logprobs(
                            req, extend_input_len, extend_logprob_start_len
                        )

                        if req.return_logprob:
                            self.add_logprob_return_values(
                                i,
                                req,
                                logprob_pt,
                                next_token_ids,
                                num_input_logprobs,
                                logits_output,
                            )
                        logprob_pt += num_input_logprobs

                    if (
                        req.return_hidden_states
                        and logits_output.hidden_states is not None
                    ):
                        req.hidden_states.append(
                            logits_output.hidden_states[
                                hidden_state_offset : (
                                    hidden_state_offset := hidden_state_offset
                                    + len(req.origin_input_ids)
                                )
                            ]
                            .cpu()
                            .clone()
                            .tolist()
                        )

                    if req.grammar is not None:
                        # FIXME: this try-except block is for handling unexpected xgrammar issue.
                        try:
                            req.grammar.accept_token(next_token_id)
                        except ValueError as e:
                            # Grammar accept_token can raise ValueError if the token is not in the grammar.
                            # This can happen if the grammar is not set correctly or the token is invalid.
                            logger.error(
                                f"Grammar accept_token failed for req {req.rid} with token {next_token_id}: {e}"
                            )
                            self.abort_request(AbortReq(rid=req.rid))
                        req.grammar.finished = req.finished()

                else:
                    # being chunked reqs' prefill is not finished
                    req.is_chunked -= 1
                    # There is only at most one request being currently chunked.
                    # Because this request does not finish prefill,
                    # we don't want to stream the request currently being chunked.
                    skip_stream_req = req

                    # Incrementally update input logprobs.
                    if batch.return_logprob:
                        extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                        extend_input_len = extend_input_len_per_req[i]
                        if extend_logprob_start_len < extend_input_len:
                            # Update input logprobs.
                            num_input_logprobs = self._calculate_num_input_logprobs(
                                req, extend_input_len, extend_logprob_start_len
                            )
                            if req.return_logprob:
                                self.add_input_logprob_return_values(
                                    i,
                                    req,
                                    logits_output,
                                    logprob_pt,
                                    num_input_logprobs,
                                    last_prefill_chunk=False,
                                )
                            logprob_pt += num_input_logprobs

                    req.time_stats.set_last_chunked_prefill_finish_time()

        else:  # embedding or reward model
            if result.copy_done is not None:
                result.copy_done.synchronize()

            is_sparse = envs.SGLANG_EMBEDDINGS_SPARSE_HEAD.is_set()

            embeddings = result.embeddings

            if is_sparse:
                batch_ids, token_ids = embeddings.indices()
                values = embeddings.values()

                embeddings = [{} for _ in range(embeddings.size(0))]
                for i in range(batch_ids.shape[0]):
                    embeddings[batch_ids[i].item()][token_ids[i].item()] = values[
                        i
                    ].item()
            else:
                if isinstance(embeddings, torch.Tensor):
                    embeddings = embeddings.tolist()
                else:
                    embeddings = [tensor.tolist() for tensor in embeddings]

            # Check finish conditions
            for i, req in enumerate(batch.reqs):
                if req.is_retracted:
                    continue

                req.embedding = embeddings[i]
                if req.is_chunked <= 0:
                    req.time_stats.set_prefill_finished_time()
                    # Dummy output token for embedding models
                    req.output_ids.append(0)
                    req.check_finished()

                    if req.finished():
                        release_kv_cache(req, self.tree_cache)
                        req.time_stats.set_completion_time()
                    else:
                        self.tree_cache.cache_unfinished_req(req)
                else:
                    # being chunked reqs' prefill is not finished
                    req.is_chunked -= 1
                    req.time_stats.set_last_chunked_prefill_finish_time()

        self.stream_output(batch.reqs, batch.return_logprob, skip_stream_req)

        if self.current_scheduler_metrics_enabled:
            can_run_cuda_graph = getattr(result, "can_run_cuda_graph", False)
            self.log_prefill_stats(
                prefill_stats=batch.prefill_stats,
                can_run_cuda_graph=can_run_cuda_graph,
                dp_cooperation_info=batch.dp_cooperation_info,
            )

    def _resolve_spec_overlap_token_ids(
        self: Scheduler, result: GenerationBatchResult, batch: ScheduleBatch
    ) -> List[List[int]]:
        """Resolve the padding next token ids for speculative decoding with overlap."""
        assert result.next_token_ids.is_cpu
        assert result.accept_lens.is_cpu

        next_token_ids = result.next_token_ids.tolist()
        accept_lens = result.accept_lens.tolist()
        result.num_accepted_tokens = sum(accept_lens) - len(batch.reqs)
        result.accept_length_per_req_cpu = [x - 1 for x in accept_lens]

        predict_tokens = []
        stride = self.draft_worker.speculative_num_draft_tokens

        for i, req in enumerate(batch.reqs):
            req.kv_committed_len += accept_lens[i]
            predict_tokens.append(
                next_token_ids[i * stride : i * stride + accept_lens[i]]
            )
            req.spec_verify_ct += 1

            accepted_draft_tokens = result.accept_length_per_req_cpu[i]
            req.spec_accepted_tokens += accepted_draft_tokens
            req.update_spec_acceptance_histogram(accepted_draft_tokens)

        return predict_tokens

    def process_batch_result_idle(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ):
        if result.copy_done is not None:
            result.copy_done.synchronize()

        self.stream_output_generation(
            batch.reqs, batch.return_logprob, is_idle_batch=True
        )

    def process_batch_result_dllm_critical(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ):
        """Critical-path processing: KV free + state updates + output_ids + finished check.
        Must run BEFORE prepare_for_dllm_decode of next step."""
        if result.copy_done is not None:
            result.copy_done.synchronize()

        self.token_to_kv_pool_allocator.free_group_begin()

        dllm_algo = getattr(self.tp_worker, "dllm_algorithm", None)
        kv_trim_info = getattr(dllm_algo, "_kv_trim_info", {}) if dllm_algo else {}
        advance_override = getattr(dllm_algo, "_advance_override", {}) if dllm_algo else {}
        dllm_write_override = getattr(dllm_algo, "_dllm_write_override", {}) if dllm_algo else {}

        # KV free (needed so prep can allocate new slots)
        kv_gpu_parts = []
        for idx in range(batch.batch_size()):
            if not result.next_token_ids:
                break
            trim_info = kv_trim_info.get(batch.reqs[idx].req_pool_idx)
            if trim_info is not None:
                gpu_indices = trim_info.get("kv_indices_gpu")
                if gpu_indices is not None:
                    kv_gpu_parts.append(gpu_indices)
        if kv_gpu_parts:
            self.token_to_kv_pool_allocator.free(torch.cat(kv_gpu_parts))

        # Update state + output_ids + finished check (all needed before next step)
        for idx in range(batch.batch_size()):
            if not result.next_token_ids:
                break
            req = batch.reqs[idx]
            req_pool_idx = req.req_pool_idx
            raw = result.next_token_ids[idx]
            next_token_ids = raw if isinstance(raw, list) else raw.tolist()

            # KV state updates
            trim_info = kv_trim_info.get(req_pool_idx)
            if trim_info is not None:
                trim_count = trim_info["trim_count"]
                req.kv_committed_len -= trim_count
                req.kv_allocated_len -= trim_count
                req.dllm_kv_valid_len = req.kv_committed_len
            else:
                req.dllm_kv_valid_len = None
            adv = advance_override.get(req_pool_idx)
            if adv is not None:
                req.dllm_next_advance = adv

            # dllm_ids write
            is_inline_pf = getattr(req, '_inline_prefill', False)
            if next_token_ids and not is_inline_pf:
                self.num_generated_tokens += len(next_token_ids)
                capture_collected = should_collect_sdar_rollout_capture_now(req)
                selected_logprobs = (
                    collect_sdar_rollout_capture(dllm_algo, req, req_pool_idx)
                    if capture_collected
                    else None
                )
                apply_dllm_carry_write(req, dllm_algo)
                dllm_tokens = dllm_write_override.pop(req_pool_idx, None)
                if req.dllm_ids and dllm_tokens is not None:
                    write_start = req.dllm_block_offset
                    req.dllm_ids[write_start:write_start + len(dllm_tokens)] = dllm_tokens

                # output_ids + finished check (needed for filter_batch)
                raw_logprobs = []
                if isinstance(selected_logprobs, dict):
                    raw_logprobs = selected_logprobs.get("raw_logprobs", []) or []
                for pos, next_token_id in enumerate(next_token_ids):
                    if req.return_logprob and pos < len(raw_logprobs):
                        req.output_token_logprobs_val.append(float(raw_logprobs[pos]))
                        req.output_token_logprobs_idx.append(int(next_token_id))
                    req.output_ids.append(next_token_id)
                    req.check_finished()
                    if req.finished():
                        try:
                            if not capture_collected:
                                collect_sdar_rollout_capture(
                                    dllm_algo, req, req_pool_idx
                                )
                        finally:
                            from sglang.srt.managers.scheduler_output_processor_mixin import release_kv_cache
                            release_kv_cache(req, self.tree_cache)
                            req.time_stats.set_completion_time()
                            if dllm_algo is not None:
                                dllm_algo.cleanup_request(req_pool_idx)
                        break

        self.token_to_kv_pool_allocator.free_group_end()
        # stream_output must run in critical path (clients wait for response)
        self.stream_output(batch.reqs, batch.return_logprob)

    def process_batch_result_dllm_deferred(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ):
        """Deferred processing: logging + cleanup.
        Runs during overlap window (while GPU is computing next step)."""
        self.log_batch_result_stats(batch, result)
        self._maybe_clear_mm_inputs(batch)
        self.maybe_send_health_check_signal()

    def process_batch_result_dllm(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ):
        import time as _time
        _tp0 = _time.perf_counter()
        if result.copy_done is not None:
            result.copy_done.synchronize()
        _tp1 = _time.perf_counter()

        self.token_to_kv_pool_allocator.free_group_begin()

        # Read algorithm signals for KV trim / dllm_ids override
        dllm_algo = getattr(self.tp_worker, "dllm_algorithm", None)
        dllm_write_override = (
            getattr(dllm_algo, "_dllm_write_override", {}) if dllm_algo else {}
        )
        kv_trim_info = (
            getattr(dllm_algo, "_kv_trim_info", {}) if dllm_algo else {}
        )
        advance_override = (
            getattr(dllm_algo, "_advance_override", {}) if dllm_algo else {}
        )

        if not result.next_token_ids:
            # Pure prefill batch (no decode requests)
            for req in batch.reqs:
                self.tree_cache.cache_unfinished_req(req)
                if req.is_dllm() and req.is_dllm_prefill():
                    origin_len = req.dllm_prefill_len()
                    cached_len = (
                        len(req.prefix_indices)
                        if req.prefix_indices is not None
                        else 0
                    )
                    if cached_len >= origin_len:
                        req.dllm_phase = DllmReqPhase.STAGING_DECODE
                        req.dllm_next_advance = origin_len
                        req._inline_prefill = False

        # Batch KV free: collect GPU tensor slices directly (no CPU round-trip)
        kv_gpu_parts = []
        kv_cpu_parts = []
        for idx in range(batch.batch_size()):
            if not result.next_token_ids:
                break
            trim_info = kv_trim_info.get(batch.reqs[idx].req_pool_idx)
            if trim_info is not None:
                gpu_indices = trim_info.get("kv_indices_gpu")
                if gpu_indices is not None:
                    kv_gpu_parts.append(gpu_indices)
                else:
                    cpu_indices = trim_info.get("kv_indices")
                    if cpu_indices:
                        kv_cpu_parts.extend(cpu_indices)
        if kv_gpu_parts:
            self.token_to_kv_pool_allocator.free(torch.cat(kv_gpu_parts))
        if kv_cpu_parts:
            self.token_to_kv_pool_allocator.free(torch.tensor(
                kv_cpu_parts, dtype=torch.int64,
                device=self.token_to_kv_pool_allocator.device,
            ))

        for idx in range(batch.batch_size()):
            if not result.next_token_ids:
                break

            req = batch.reqs[idx]
            raw = result.next_token_ids[idx]
            next_token_ids = raw if isinstance(raw, list) else raw.tolist()

            # Handle inline prefill requests in mixed batch:
            # Force-treat _inline_prefill requests as pure prefill regardless
            # of what the algorithm returned (it may not know about inline prefill).
            is_inline_pf = getattr(req, '_inline_prefill', False)
            if not next_token_ids or is_inline_pf:
                self.tree_cache.cache_unfinished_req(req)
                # Transition prefill → decode
                if req.is_dllm() and (req.is_dllm_prefill() or is_inline_pf):
                    origin_len = req.dllm_prefill_len()
                    cached_len = (
                        len(req.prefix_indices)
                        if req.prefix_indices is not None
                        else 0
                    )
                    if cached_len >= origin_len:
                        req.dllm_phase = DllmReqPhase.STAGING_DECODE
                        req.dllm_next_advance = origin_len
                        req._inline_prefill = False
                continue

            self.num_generated_tokens += len(next_token_ids)

            req_pool_idx = req.req_pool_idx
            capture_collected = should_collect_sdar_rollout_capture_now(req)
            selected_logprobs = (
                collect_sdar_rollout_capture(dllm_algo, req, req_pool_idx)
                if capture_collected
                else None
            )
            apply_dllm_carry_write(req, dllm_algo)
            dllm_tokens = dllm_write_override.pop(req_pool_idx, None)
            if req.dllm_ids:
                if dllm_tokens is not None:
                    write_start = req.dllm_block_offset
                    req.dllm_ids[
                        write_start : write_start + len(dllm_tokens)
                    ] = dllm_tokens
                elif next_token_ids:
                    block_size = req.dllm_config.block_size
                    write_start = (
                        req.dllm_block_offset
                        + block_size
                        - len(next_token_ids)
                    )
                    req.dllm_ids[
                        write_start : write_start + len(next_token_ids)
                    ] = next_token_ids

            trim_info = kv_trim_info.pop(req_pool_idx, None)
            if trim_info is not None:
                trim_count = trim_info["trim_count"]
                req.kv_committed_len -= trim_count
                req.kv_allocated_len -= trim_count
                req.dllm_kv_valid_len = req.kv_committed_len
            else:
                req.dllm_kv_valid_len = None

            adv = advance_override.pop(req_pool_idx, None)
            if adv is not None:
                req.dllm_next_advance = adv

            finished = False
            raw_logprobs = []
            if isinstance(selected_logprobs, dict):
                raw_logprobs = selected_logprobs.get("raw_logprobs", []) or []
            for pos, next_token_id in enumerate(next_token_ids):
                if req.return_logprob and pos < len(raw_logprobs):
                    req.output_token_logprobs_val.append(float(raw_logprobs[pos]))
                    req.output_token_logprobs_idx.append(int(next_token_id))
                req.output_ids.append(next_token_id)
                req.check_finished()
                if req.finished():
                    try:
                        if not capture_collected:
                            collect_sdar_rollout_capture(
                                dllm_algo, req, req_pool_idx
                            )
                    finally:
                        release_kv_cache(req, self.tree_cache)
                        req.time_stats.set_completion_time()
                        if dllm_algo is not None:
                            dllm_algo.cleanup_request(req_pool_idx)
                    finished = True
                    break
            if not finished:
                # In fast decode loop, skip expensive cache_unfinished_req
                # (GPU tensor copy per req). prepare_for_dllm_decode uses
                # kv_committed_len instead of prefix_indices for decode reqs.
                if not getattr(batch, '_dllm_decode_mode', False) or is_inline_pf:
                    self.tree_cache.cache_unfinished_req(req)

        _tp2 = _time.perf_counter()
        self.stream_output(batch.reqs, batch.return_logprob)
        _tp3 = _time.perf_counter()
        self.token_to_kv_pool_allocator.free_group_end()
        _tp4 = _time.perf_counter()

        # Accumulate process timing
        _proc_timing = getattr(self, '_dllm_proc_timing', None)
        if _proc_timing is None:
            _proc_timing = {'sync': 0, 'loop': 0, 'stream': 0, 'free': 0, 'count': 0}
            self._dllm_proc_timing = _proc_timing
        _proc_timing['sync'] += (_tp1 - _tp0)
        _proc_timing['loop'] += (_tp2 - _tp1)
        _proc_timing['stream'] += (_tp3 - _tp2)
        _proc_timing['free'] += (_tp4 - _tp3)
        _proc_timing['count'] += 1
        if _proc_timing['count'] % 500 == 0:
            c = _proc_timing['count']
            logger.info(
                f"[DLLM process] avg(ms): sync={_proc_timing['sync']/c*1000:.2f} "
                f"loop={_proc_timing['loop']/c*1000:.2f} "
                f"stream={_proc_timing['stream']/c*1000:.2f} "
                f"free={_proc_timing['free']/c*1000:.2f}"
            )

        if self.current_scheduler_metrics_enabled:
            can_run_cuda_graph = getattr(result, "can_run_cuda_graph", False)
            self.log_prefill_stats(
                prefill_stats=batch.prefill_stats,
                can_run_cuda_graph=can_run_cuda_graph,
                dp_cooperation_info=batch.dp_cooperation_info,
            )

    def process_batch_result_decode(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ):
        if result.copy_done is not None:
            result.copy_done.synchronize()

        logits_output, next_token_ids, can_run_cuda_graph = (
            result.logits_output,
            result.next_token_ids,
            result.can_run_cuda_graph,
        )

        if batch.spec_algorithm.is_none():
            next_token_ids = next_token_ids.tolist()
            if batch.return_logprob:
                next_token_logprobs = logits_output.next_token_logprobs.tolist()
        elif batch.is_spec_v2:
            next_token_ids = self._resolve_spec_overlap_token_ids(result, batch)

        self.num_generated_tokens += len(batch.reqs)
        if not batch.spec_algorithm.is_none():
            self.update_spec_metrics(batch.batch_size(), result.num_accepted_tokens)
        if self.enable_metrics:
            self.metrics_collector.increment_cuda_graph_pass(value=can_run_cuda_graph)

        self.token_to_kv_pool_allocator.free_group_begin()

        # NOTE: in any case, we should check finish here
        # if finished, also clean up committed kv cache and over-allocated kv cache here

        # Check finish condition
        for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids)):
            req: Req

            if self.enable_overlap and (req.finished() or req.is_retracted):
                # NOTE: This (req.finished() or req.is_retracted) should only happen when overlap scheduling is enabled.
                # (currently not, e.g. Eagle V1 still check finish during forward)
                # And all the over-allocated tokens will be freed in `release_kv_cache`.
                continue

            new_accepted_len = 1
            if batch.spec_algorithm.is_none():
                req.output_ids.append(next_token_id)
            elif batch.is_spec_v2:
                # Only spec v2's output_ids are updated here.
                req.output_ids.extend(next_token_id)
                new_accepted_len = len(next_token_id)

            # Update Mamba last track seqlen
            self._mamba_prefix_cache_update(req, batch, result, i)

            req.time_stats.set_last_decode_finish_time()

            req.check_finished(new_accepted_len)

            if req.finished():
                # delete feature to save memory
                if req.multimodal_inputs is not None:
                    for mm_item in req.multimodal_inputs.mm_items:
                        pixel_values = mm_item.feature
                        if isinstance(pixel_values, torch.Tensor):
                            mm_item.feature = None
                            del pixel_values
                self.maybe_collect_routed_experts(req)

                if self.server_args.disaggregation_decode_enable_offload_kvcache:
                    # Asynchronously offload KV cache; release_kv_cache will be called after Device->Host transfer completes
                    if not self.decode_offload_manager.offload_kv_cache(req):
                        release_kv_cache(req, self.tree_cache)
                else:
                    release_kv_cache(req, self.tree_cache)

                req.time_stats.set_completion_time()

            self.maybe_collect_customized_info(i, req, logits_output)

            if req.return_logprob and batch.spec_algorithm.is_none():
                # speculative worker handles logprob in speculative decoding
                req.output_token_logprobs_val.append(next_token_logprobs[i])
                req.output_token_logprobs_idx.append(next_token_id)
                if req.top_logprobs_num > 0:
                    req.output_top_logprobs_val.append(
                        logits_output.next_token_top_logprobs_val[i]
                    )
                    req.output_top_logprobs_idx.append(
                        logits_output.next_token_top_logprobs_idx[i]
                    )
                if req.token_ids_logprob is not None:
                    req.output_token_ids_logprobs_val.append(
                        logits_output.next_token_token_ids_logprobs_val[i]
                    )
                    req.output_token_ids_logprobs_idx.append(
                        logits_output.next_token_token_ids_logprobs_idx[i]
                    )

            if req.return_hidden_states and logits_output.hidden_states is not None:
                req.hidden_states.append(
                    logits_output.hidden_states[i].cpu().clone().tolist()
                )

            if req.grammar is not None:
                # FIXME: this try-except block is for handling unexpected xgrammar issue.
                try:
                    if batch.spec_algorithm.is_none():
                        # Normal decode: single token
                        req.grammar.accept_token(next_token_id)
                    elif batch.is_spec_v2:
                        # Speculative decode: next_token_id is a list of accepted tokens
                        for token_id in next_token_id:
                            req.grammar.accept_token(token_id)
                except ValueError as e:
                    # Grammar accept_token can raise ValueError if the token is not in the grammar.
                    # This can happen if the grammar is not set correctly or the token is invalid.
                    logger.error(
                        f"Grammar accept_token failed for req {req.rid} with token {next_token_id}: {e}"
                    )
                    self.abort_request(AbortReq(rid=req.rid))
                req.grammar.finished = req.finished()

        self.stream_output(batch.reqs, batch.return_logprob)
        self.token_to_kv_pool_allocator.free_group_end()

        self.forward_ct_decode = (self.forward_ct_decode + 1) % (1 << 30)
        if self.current_scheduler_metrics_enabled:
            if self.forward_ct_decode % self.server_args.decode_log_interval == 0:
                self.log_decode_stats(can_run_cuda_graph, running_batch=batch)
            self.log_decode_stats_every_iteration(
                batch, num_accepted_tokens=result.num_accepted_tokens
            )

    def _mamba_prefix_cache_update(
        self, req: Req, batch: ScheduleBatch, result: GenerationBatchResult, i: int
    ) -> None:
        seq_len = len(req.origin_input_ids) + len(req.output_ids) - 1
        if req.mamba_ping_pong_track_buffer is not None:
            mamba_track_interval = get_global_server_args().mamba_track_interval
            if batch.spec_algorithm.is_none() and seq_len % mamba_track_interval == 0:
                # for non-spec decode, we update mamba_last_track_seqlen at the end of each track interval
                req.mamba_next_track_idx = (
                    batch.req_to_token_pool.get_mamba_ping_pong_other_idx(
                        req.mamba_next_track_idx
                    )
                )
                req.mamba_last_track_seqlen = seq_len
            elif (
                not batch.spec_algorithm.is_none()
                and result.accept_length_per_req_cpu is not None
            ):
                # for spec decode, update mamba_last_track_seqlen if this iteration crosses a track interval
                actual_seq_len = req.seqlen - 1
                if (
                    actual_seq_len // mamba_track_interval
                    != (actual_seq_len - result.accept_length_per_req_cpu[i])
                    // mamba_track_interval
                ):
                    req.mamba_next_track_idx = (
                        batch.req_to_token_pool.get_mamba_ping_pong_other_idx(
                            req.mamba_next_track_idx
                        )
                    )
                    req.mamba_last_track_seqlen = (
                        actual_seq_len // mamba_track_interval * mamba_track_interval
                    )

    def _process_input_token_logprobs(
        self, req: Req, input_token_logprobs: List
    ) -> None:
        """Process input token logprobs values and indices."""
        is_multi_item_scoring = self._is_multi_item_scoring(req)

        # Process logprob values - handle multi-item scoring vs regular requests
        if is_multi_item_scoring:
            # Multi-item scoring: use all logprobs as-is
            req.input_token_logprobs_val = input_token_logprobs
        else:
            # Regular request: add None at start, remove last (sampling token)
            req.input_token_logprobs_val = [None] + input_token_logprobs[:-1]

        # Process logprob indices based on scoring type
        if is_multi_item_scoring:
            # Multi-item scoring: only include delimiter token positions
            relevant_tokens = req.origin_input_ids[req.logprob_start_len :]
            input_token_logprobs_idx = [
                token_id
                for token_id in relevant_tokens
                if token_id == self.server_args.multi_item_scoring_delimiter
            ]
        else:
            # Regular request: include all tokens from logprob_start_len onwards
            input_token_logprobs_idx = req.origin_input_ids[req.logprob_start_len :]

        # Clip padded hash values from image tokens to prevent detokenization errors
        req.input_token_logprobs_idx = [
            x if x < self.model_config.vocab_size - 1 else 0
            for x in input_token_logprobs_idx
        ]

    def _process_input_top_logprobs(self, req: Req) -> None:
        """Process input top logprobs."""
        if req.top_logprobs_num <= 0:
            return

        is_multi_item_scoring = self._is_multi_item_scoring(req)

        # Initialize arrays - multi-item scoring starts empty, others start with None
        req.input_top_logprobs_val = [] if is_multi_item_scoring else [None]
        req.input_top_logprobs_idx = [] if is_multi_item_scoring else [None]

        # Extend arrays with temp values
        for val, idx in zip(
            req.temp_input_top_logprobs_val,
            req.temp_input_top_logprobs_idx,
            strict=True,
        ):
            req.input_top_logprobs_val.extend(val)
            req.input_top_logprobs_idx.extend(idx)

        # Remove last token (sampling token) for non multi-item scoring requests
        if not is_multi_item_scoring:
            req.input_top_logprobs_val.pop()
            req.input_top_logprobs_idx.pop()

        # Clean up temp storage
        req.temp_input_top_logprobs_idx = None
        req.temp_input_top_logprobs_val = None

    def _process_input_token_ids_logprobs(self, req: Req) -> None:
        """Process input token IDs logprobs."""
        if req.token_ids_logprob is None:
            return

        is_multi_item_scoring = self._is_multi_item_scoring(req)

        # Initialize arrays - multi-item scoring starts empty, others start with None
        req.input_token_ids_logprobs_val = [] if is_multi_item_scoring else [None]
        req.input_token_ids_logprobs_idx = [] if is_multi_item_scoring else [None]

        # Process temp values - convert tensors to lists and extend arrays
        for val, idx in zip(
            req.temp_input_token_ids_logprobs_val,
            req.temp_input_token_ids_logprobs_idx,
            strict=True,
        ):
            val_list = val.tolist() if isinstance(val, torch.Tensor) else val
            req.input_token_ids_logprobs_val.extend(
                val_list if isinstance(val_list, list) else [val_list]
            )
            req.input_token_ids_logprobs_idx.extend(idx)

        # Remove last token (sampling token) for non multi-item scoring requests
        if not is_multi_item_scoring:
            req.input_token_ids_logprobs_val.pop()
            req.input_token_ids_logprobs_idx.pop()

        # Clean up temp storage
        req.temp_input_token_ids_logprobs_idx = None
        req.temp_input_token_ids_logprobs_val = None

    def _calculate_relevant_tokens_len(self, req: Req) -> int:
        """Calculate the expected length of logprob arrays based on whether multi-item scoring is enabled.

        For multi-item scoring, only delimiter positions have logprobs.
        For regular requests, all positions from logprob_start_len onwards have logprobs.
        """
        is_multi_item_scoring = self._is_multi_item_scoring(req)
        relevant_tokens = req.origin_input_ids[req.logprob_start_len :]

        if is_multi_item_scoring:
            # Multi-item scoring: count delimiter tokens from logprob_start_len onwards
            return sum(
                1
                for token_id in relevant_tokens
                if token_id == self.server_args.multi_item_scoring_delimiter
            )
        else:
            # Regular request: all tokens from logprob_start_len onwards
            return len(relevant_tokens)

    def _calculate_num_input_logprobs(
        self, req: Req, extend_input_len: int, extend_logprob_start_len: int
    ) -> int:
        """Calculate the number of input logprobs based on whether multi-item scoring is enabled.

        For multi-item scoring, only delimiter positions have logprobs.
        For regular requests, all positions in the range have logprobs.
        """
        is_multi_item_scoring = self._is_multi_item_scoring(req)

        if is_multi_item_scoring:
            # Multi-item scoring: count delimiter tokens in the relevant portion
            relevant_tokens = req.origin_input_ids[
                extend_logprob_start_len:extend_input_len
            ]
            return sum(
                1
                for token_id in relevant_tokens
                if token_id == self.server_args.multi_item_scoring_delimiter
            )
        else:
            # Regular request: all tokens in the range
            return extend_input_len - extend_logprob_start_len

    def _is_multi_item_scoring(self, req: Req) -> bool:
        """Check if request uses multi-item scoring.

        Multi-item scoring applies to prefill-only requests when a delimiter
        token is configured. In this mode, only positions containing the
        delimiter token receive logprobs.
        """
        return req.is_prefill_only and self.server_args.multi_item_scoring_delimiter

    def add_input_logprob_return_values(
        self: Scheduler,
        i: int,
        req: Req,
        output: LogitsProcessorOutput,
        logprob_pt: int,
        num_input_logprobs: int,
        last_prefill_chunk: bool,  # If True, it means prefill is finished.
    ):
        """Incrementally add input logprobs to `req`.

        Args:
            i: The request index in a batch.
            req: The request. Input logprobs inside req are modified as a
                consequence of the API
            fill_ids: The prefill ids processed.
            output: Logit processor output that's used to compute input logprobs
            last_prefill_chunk: True if it is the last prefill (when chunked).
                Some of input logprob operation should only happen at the last
                prefill (e.g., computing input token logprobs).
        """
        assert output.input_token_logprobs is not None
        if req.input_token_logprobs is None:
            req.input_token_logprobs = []
        if req.temp_input_top_logprobs_val is None:
            req.temp_input_top_logprobs_val = []
        if req.temp_input_top_logprobs_idx is None:
            req.temp_input_top_logprobs_idx = []
        if req.temp_input_token_ids_logprobs_val is None:
            req.temp_input_token_ids_logprobs_val = []
        if req.temp_input_token_ids_logprobs_idx is None:
            req.temp_input_token_ids_logprobs_idx = []

        if req.input_token_logprobs_val is not None:
            # The input logprob has been already computed. It only happens
            # upon retract.
            if req.top_logprobs_num > 0:
                assert req.input_token_logprobs_val is not None
            return

        # Important for the performance.
        assert isinstance(output.input_token_logprobs, tuple)
        input_token_logprobs: Tuple[int] = output.input_token_logprobs
        input_token_logprobs = input_token_logprobs[
            logprob_pt : logprob_pt + num_input_logprobs
        ]
        req.input_token_logprobs.extend(input_token_logprobs)

        if req.top_logprobs_num > 0:
            req.temp_input_top_logprobs_val.append(output.input_top_logprobs_val[i])
            req.temp_input_top_logprobs_idx.append(output.input_top_logprobs_idx[i])

        if req.token_ids_logprob is not None:
            req.temp_input_token_ids_logprobs_val.append(
                output.input_token_ids_logprobs_val[i]
            )
            req.temp_input_token_ids_logprobs_idx.append(
                output.input_token_ids_logprobs_idx[i]
            )

        if last_prefill_chunk:
            input_token_logprobs = req.input_token_logprobs
            req.input_token_logprobs = None
            assert req.input_token_logprobs_val is None
            assert req.input_token_logprobs_idx is None
            assert req.input_top_logprobs_val is None
            assert req.input_top_logprobs_idx is None

            # Process all input logprob types using helper functions
            self._process_input_token_logprobs(req, input_token_logprobs)
            self._process_input_top_logprobs(req)

            self._process_input_token_ids_logprobs(req)

            if req.return_logprob:
                relevant_tokens_len = self._calculate_relevant_tokens_len(req)
                assert len(req.input_token_logprobs_val) == relevant_tokens_len
                assert len(req.input_token_logprobs_idx) == relevant_tokens_len
                if req.top_logprobs_num > 0:
                    assert len(req.input_top_logprobs_val) == relevant_tokens_len
                    assert len(req.input_top_logprobs_idx) == relevant_tokens_len
                if req.token_ids_logprob is not None:
                    assert len(req.input_token_ids_logprobs_val) == relevant_tokens_len
                    assert len(req.input_token_ids_logprobs_idx) == relevant_tokens_len

    def add_logprob_return_values(
        self: Scheduler,
        i: int,
        req: Req,
        pt: int,
        next_token_ids: List[int],
        num_input_logprobs: int,
        output: LogitsProcessorOutput,
    ):
        """Attach logprobs to the return values."""
        if output.next_token_logprobs is not None:
            req.output_token_logprobs_val.append(output.next_token_logprobs[i])
            req.output_token_logprobs_idx.append(next_token_ids[i])

        # Only add input logprobs if there are input tokens to process
        # Note: For prefill-only requests with default logprob_start_len, this will be 0,
        # meaning we only compute output logprobs (which is the intended behavior)
        if num_input_logprobs > 0:
            self.add_input_logprob_return_values(
                i, req, output, pt, num_input_logprobs, last_prefill_chunk=True
            )
        else:
            self._initialize_empty_logprob_containers(req)

        if req.top_logprobs_num > 0:
            req.output_top_logprobs_val.append(output.next_token_top_logprobs_val[i])
            req.output_top_logprobs_idx.append(output.next_token_top_logprobs_idx[i])

        if (
            req.token_ids_logprob is not None
            and output.next_token_token_ids_logprobs_val is not None
        ):
            # Convert GPU tensor to list if needed
            logprobs_val = output.next_token_token_ids_logprobs_val[i]
            if isinstance(logprobs_val, torch.Tensor):
                logprobs_val = logprobs_val.tolist()
            req.output_token_ids_logprobs_val.append(logprobs_val)
            req.output_token_ids_logprobs_idx.append(
                output.next_token_token_ids_logprobs_idx[i]
            )

        return num_input_logprobs

    def _initialize_empty_logprob_containers(self, req: Req) -> None:
        """
        Initialize logprob fields to empty lists if unset.

        This is needed for prefill-only requests where the normal initialization
        flow might be bypassed, but downstream code expects these fields to be lists.
        """
        if req.input_token_logprobs_val is None:
            req.input_token_logprobs_val = []
        if req.input_token_logprobs_idx is None:
            req.input_token_logprobs_idx = []
        if req.input_top_logprobs_val is None:
            req.input_top_logprobs_val = []
        if req.input_top_logprobs_idx is None:
            req.input_top_logprobs_idx = []
        if req.input_token_ids_logprobs_val is None:
            req.input_token_ids_logprobs_val = []
        if req.input_token_ids_logprobs_idx is None:
            req.input_token_ids_logprobs_idx = []

    def stream_output(
        self: Scheduler,
        reqs: List[Req],
        return_logprob: bool,
        skip_req: Optional[Req] = None,
    ):
        """Stream the output to detokenizer."""
        if self.is_generation:
            self.stream_output_generation(reqs, return_logprob, skip_req)
        else:  # embedding or reward model
            self.stream_output_embedding(reqs)

        if envs.SGLANG_TEST_CRASH_AFTER_STREAM_OUTPUTS.get() > 0:
            self._trigger_crash_for_tests(
                envs.SGLANG_TEST_CRASH_AFTER_STREAM_OUTPUTS.get()
            )

    def _trigger_crash_for_tests(self, crash_threshold: int):
        # Crash trigger: crash after stream_output is called N times
        # This is used for testing purposes.
        if not hasattr(self, "_test_stream_output_count"):
            self._test_stream_output_count = 0
        self._test_stream_output_count += 1
        if self._test_stream_output_count >= crash_threshold:
            raise RuntimeError(
                f"Test crash after stream_output called {self._test_stream_output_count} times"
            )

    def stream_output_generation(
        self: Scheduler,
        reqs: List[Req],
        return_logprob: bool,
        skip_req: Optional[Req] = None,
        is_idle_batch: bool = False,
    ):
        rids = []
        http_worker_ipcs = []
        finished_reasons: List[BaseFinishReason] = []

        decoded_texts = []
        decode_ids_list = []
        read_offsets = []
        output_ids = []

        skip_special_tokens = []
        spaces_between_special_tokens = []
        no_stop_trim = []
        prompt_tokens = []
        completion_tokens = []
        cached_tokens = []
        cached_tokens_details = []  # Detailed breakdown by cache source
        spec_verify_ct = []
        spec_accepted_tokens = []
        spec_acceptance_histogram = []
        retraction_counts = []
        output_hidden_states = None
        load = self.get_load()
        routed_experts = None
        customized_info = {}
        include_sdar_request_metrics = any(
            hasattr(req, "sglang_sdar_request_metrics")
            for req in reqs
            if req is not skip_req
        )
        include_sdar_replay_tensors = any(
            hasattr(req, "sglang_sdar_replay_tensors")
            for req in reqs
            if req is not skip_req
        )

        time_stats = []

        if return_logprob:
            input_token_logprobs_val = []
            input_token_logprobs_idx = []
            output_token_logprobs_val = []
            output_token_logprobs_idx = []
            input_top_logprobs_val = []
            input_top_logprobs_idx = []
            output_top_logprobs_val = []
            output_top_logprobs_idx = []
            input_token_ids_logprobs_val = []
            input_token_ids_logprobs_idx = []
            output_token_ids_logprobs_val = []
            output_token_ids_logprobs_idx = []
        else:
            input_token_logprobs_val = input_token_logprobs_idx = (
                output_token_logprobs_val
            ) = output_token_logprobs_idx = input_top_logprobs_val = (
                input_top_logprobs_idx
            ) = output_top_logprobs_val = output_top_logprobs_idx = (
                input_token_ids_logprobs_val
            ) = input_token_ids_logprobs_idx = output_token_ids_logprobs_val = (
                output_token_ids_logprobs_idx
            ) = None

        for req in reqs:
            if req is skip_req:
                continue

            # Multimodal partial stream chunks break the detokenizer, so drop aborted requests here.
            if self.model_config.is_multimodal_gen and req.to_finish:
                continue

            if req.finished():
                if req.finished_output:
                    # With the overlap schedule, a request will try to output twice and hit this line twice
                    # because of the one additional delayed token. This "continue" prevented the dummy output.
                    continue
                req.finished_output = True
                if req.finished_len is None:
                    req.finished_len = len(req.output_ids)
                should_output = True
            else:
                if req.stream:
                    stream_interval = (
                        req.sampling_params.stream_interval or self.stream_interval
                    )

                    # origin stream_interval logic
                    should_output = (
                        len(req.output_ids) % stream_interval == 1
                        if not self.model_config.is_multimodal_gen
                        and stream_interval > 1
                        else len(req.output_ids) % stream_interval == 0
                    )

                    if should_output:
                        # check_match_stop_str_prefix if  tail_str's suffix match stop_str prefix
                        should_output &= not req.check_match_stop_str_prefix()
                    elif (
                        _sdar_replay_early_stream_enabled()
                        and hasattr(req, "sglang_sdar_replay_tensors")
                    ):
                        # SDAR replay can become available before the normal
                        # token stream interval. If at least one token is
                        # also ready, stream now so the trainer can consume
                        # replay without sending empty chunks to detokenizer.
                        replay_tensors = getattr(req, "sglang_sdar_replay_tensors", [])
                        replay_offset = int(getattr(req, "send_sdar_replay_tensors_offset", 0) or 0)
                        has_unsent_replay = replay_offset < len(replay_tensors)
                        has_new_token = req.send_token_offset < len(req.output_ids_through_stop)
                        if has_unsent_replay and has_new_token:
                            should_output = not req.check_match_stop_str_prefix()
                else:
                    should_output = (
                        len(req.output_ids) % DEFAULT_FORCE_STREAM_INTERVAL == 0
                        if not self.model_config.is_multimodal_gen
                        else False
                    )

            # Skip if no new tokens since last stream (avoids empty pipeline round-trip,
            # critical for DLLM where prefill produces 0 tokens)
            if should_output and not req.finished():
                if req.send_token_offset >= len(req.output_ids):
                    should_output = False

            if should_output:
                send_token_offset = req.send_token_offset
                send_output_token_logprobs_offset = (
                    req.send_output_token_logprobs_offset
                )
                rids.append(req.rid)
                http_worker_ipcs.append(req.http_worker_ipc)
                finished_reasons.append(
                    req.finished_reason.to_json() if req.finished_reason else None
                )
                decoded_texts.append(req.decoded_text)
                decode_ids, read_offset = req.init_incremental_detokenize()

                if self.model_config.is_multimodal_gen:
                    decode_ids_list.append(decode_ids)
                else:
                    decode_ids_list.append(decode_ids[req.send_decode_id_offset :])

                # Exclude the tokens after stop condition
                output_ids_ = req.output_ids_through_stop

                req.send_decode_id_offset = len(decode_ids)
                read_offsets.append(read_offset)
                output_ids.append(output_ids_[send_token_offset:])
                req.send_token_offset = len(output_ids_)
                skip_special_tokens.append(req.sampling_params.skip_special_tokens)
                spaces_between_special_tokens.append(
                    req.sampling_params.spaces_between_special_tokens
                )
                no_stop_trim.append(req.sampling_params.no_stop_trim)
                prompt_tokens.append(len(req.origin_input_ids))
                completion_tokens.append(len(output_ids_))
                cached_tokens.append(req.cached_tokens)

                # Collect detailed cache breakdown if available
                cached_tokens_details.append(self._get_cached_tokens_details(req))

                retraction_counts.append(req.retraction_count)

                time_stats.append(req.time_stats)

                if not self.spec_algorithm.is_none():
                    spec_verify_ct.append(req.spec_verify_ct)
                    spec_accepted_tokens.append(req.spec_accepted_tokens)
                    spec_acceptance_histogram.append(req.spec_acceptance_histogram)

                if return_logprob:
                    if (
                        req.return_logprob
                        and not req.input_logprob_sent
                        # Decode server does not send input logprobs
                        and self.disaggregation_mode != DisaggregationMode.DECODE
                    ):
                        input_token_logprobs_val.append(req.input_token_logprobs_val)
                        input_token_logprobs_idx.append(req.input_token_logprobs_idx)
                        input_top_logprobs_val.append(req.input_top_logprobs_val)
                        input_top_logprobs_idx.append(req.input_top_logprobs_idx)
                        input_token_ids_logprobs_val.append(
                            req.input_token_ids_logprobs_val
                        )
                        input_token_ids_logprobs_idx.append(
                            req.input_token_ids_logprobs_idx
                        )
                        req.input_logprob_sent = True
                    else:
                        input_token_logprobs_val.append([])
                        input_token_logprobs_idx.append([])
                        input_top_logprobs_val.append([])
                        input_top_logprobs_idx.append([])
                        input_token_ids_logprobs_val.append([])
                        input_token_ids_logprobs_idx.append([])

                    if req.return_logprob:
                        output_token_logprobs_val.append(
                            req.output_token_logprobs_val[
                                send_output_token_logprobs_offset:
                            ]
                        )
                        output_token_logprobs_idx.append(
                            req.output_token_logprobs_idx[
                                send_output_token_logprobs_offset:
                            ]
                        )
                        output_top_logprobs_val.append(
                            req.output_top_logprobs_val[
                                send_output_token_logprobs_offset:
                            ]
                        )
                        output_top_logprobs_idx.append(
                            req.output_top_logprobs_idx[
                                send_output_token_logprobs_offset:
                            ]
                        )
                        output_token_ids_logprobs_val.append(
                            req.output_token_ids_logprobs_val[
                                send_output_token_logprobs_offset:
                            ]
                        )
                        output_token_ids_logprobs_idx.append(
                            req.output_token_ids_logprobs_idx[
                                send_output_token_logprobs_offset:
                            ]
                        )
                        req.send_output_token_logprobs_offset = len(
                            req.output_token_logprobs_val
                        )
                    else:
                        output_token_logprobs_val.append([])
                        output_token_logprobs_idx.append([])
                        output_top_logprobs_val.append([])
                        output_top_logprobs_idx.append([])
                        output_token_ids_logprobs_val.append([])
                        output_token_ids_logprobs_idx.append([])

                if req.return_hidden_states:
                    if output_hidden_states is None:
                        output_hidden_states = []
                    output_hidden_states.append(req.hidden_states)
                if req.return_routed_experts:
                    if routed_experts is None:
                        routed_experts = []
                    routed_experts.append(req.routed_experts)

                if req.customized_info is not None:
                    for k, v in req.customized_info.items():
                        if k not in customized_info:
                            customized_info[k] = []
                        customized_info[k].append(v[send_token_offset:])
                if include_sdar_request_metrics:
                    customized_info.setdefault("sglang_sdar_request_metrics", []).append(
                        getattr(req, "sglang_sdar_request_metrics", [])
                    )
                if include_sdar_replay_tensors:
                    replay_tensors_for_output = getattr(req, "sglang_sdar_replay_tensors", [])
                    if getattr(req, "stream", False):
                        replay_offset = int(getattr(req, "send_sdar_replay_tensors_offset", 0) or 0)
                        replay_offset = max(0, min(replay_offset, len(replay_tensors_for_output)))
                        replay_tensors_for_output = replay_tensors_for_output[replay_offset:]
                        req.send_sdar_replay_tensors_offset = len(
                            getattr(req, "sglang_sdar_replay_tensors", [])
                        )
                    customized_info.setdefault("sglang_sdar_replay_tensors", []).append(
                        replay_tensors_for_output
                    )

            if (
                req.finished()
                and self.attn_tp_rank == 0
                and self.server_args.enable_request_time_stats_logging
            ):
                req.log_time_stats()

        dp_ranks = [self.dp_rank] * len(rids) if rids else None

        # Send to detokenizer
        if reqs or is_idle_batch:
            if self.model_config.is_multimodal_gen:
                return
            self.send_to_detokenizer.send_output(
                BatchTokenIDOutput(
                    rids=rids,
                    http_worker_ipcs=http_worker_ipcs,
                    spec_verify_ct=spec_verify_ct,
                    spec_accepted_tokens=spec_accepted_tokens,
                    spec_acceptance_histogram=spec_acceptance_histogram,
                    time_stats=time_stats,
                    finished_reasons=finished_reasons,
                    decoded_texts=decoded_texts,
                    decode_ids=decode_ids_list,
                    read_offsets=read_offsets,
                    output_ids=output_ids,
                    skip_special_tokens=skip_special_tokens,
                    spaces_between_special_tokens=spaces_between_special_tokens,
                    no_stop_trim=no_stop_trim,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cached_tokens=cached_tokens,
                    cached_tokens_details=cached_tokens_details,
                    input_token_logprobs_val=input_token_logprobs_val,
                    input_token_logprobs_idx=input_token_logprobs_idx,
                    output_token_logprobs_val=output_token_logprobs_val,
                    output_token_logprobs_idx=output_token_logprobs_idx,
                    input_top_logprobs_val=input_top_logprobs_val,
                    input_top_logprobs_idx=input_top_logprobs_idx,
                    output_top_logprobs_val=output_top_logprobs_val,
                    output_top_logprobs_idx=output_top_logprobs_idx,
                    input_token_ids_logprobs_val=input_token_ids_logprobs_val,
                    input_token_ids_logprobs_idx=input_token_ids_logprobs_idx,
                    output_token_ids_logprobs_val=output_token_ids_logprobs_val,
                    output_token_ids_logprobs_idx=output_token_ids_logprobs_idx,
                    output_token_entropy_val=None,
                    output_hidden_states=output_hidden_states,
                    routed_experts=routed_experts,
                    customized_info=customized_info,
                    placeholder_tokens_idx=None,
                    placeholder_tokens_val=None,
                    retraction_counts=retraction_counts,
                    load=load,
                    dp_ranks=dp_ranks,
                )
            )

    def stream_output_embedding(self: Scheduler, reqs: List[Req]):
        rids = []
        http_worker_ipcs = []
        finished_reasons: List[BaseFinishReason] = []

        embeddings = []
        prompt_tokens = []
        cached_tokens = []
        cached_tokens_details = []  # Detailed breakdown by cache source
        time_stats = []
        retraction_counts = []
        for req in reqs:
            if req.finished():
                rids.append(req.rid)
                http_worker_ipcs.append(req.http_worker_ipc)
                finished_reasons.append(req.finished_reason.to_json())
                embeddings.append(req.embedding)
                prompt_tokens.append(len(req.origin_input_ids))
                cached_tokens.append(req.cached_tokens)

                # Collect detailed cache breakdown if available
                cached_tokens_details.append(self._get_cached_tokens_details(req))
                time_stats.append(req.time_stats)
                retraction_counts.append(req.retraction_count)
        self.send_to_detokenizer.send_output(
            BatchEmbeddingOutput(
                rids=rids,
                http_worker_ipcs=http_worker_ipcs,
                time_stats=time_stats,
                finished_reasons=finished_reasons,
                embeddings=embeddings,
                prompt_tokens=prompt_tokens,
                cached_tokens=cached_tokens,
                cached_tokens_details=cached_tokens_details,
                placeholder_tokens_idx=None,
                placeholder_tokens_val=None,
                retraction_counts=retraction_counts,
            )
        )
