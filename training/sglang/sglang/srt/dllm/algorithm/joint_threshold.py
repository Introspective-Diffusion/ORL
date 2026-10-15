import os
import threading as _threading

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


def _get_num_transfer_tokens(block_size: int, steps: int):
    """Distribute block_size tokens evenly across steps (matches official HF generate)."""
    base = block_size // steps
    remainder = block_size % steps
    schedule = torch.zeros(steps, dtype=torch.int64)
    schedule[:] = base
    schedule[:remainder] += 1
    return schedule



# --- TPF instrumentation ---
# The published LLaDA2.1 table reports Score | TPF, and TPF = generated tokens /
# forward passes. base.py already defines get_stats(), but nothing here ever wrote
# to _stats -- only SDARPrefixKV populates it -- so TPF read 0 under this algorithm.
#
# Two sinks, because they answer different questions:
#   _stats["total_forwards"]  -- process-lifetime total, via get_stats()
#   TPF_COUNTER_PATH          -- same total mirrored to a file at the end of each
#                                run() call, i.e. once per block rather than once
#                                per forward (per-forward writes would dominate the
#                                step time). With --max-running-requests 1 the
#                                server is serial, so an external driver can read
#                                the file before and after one request and get that
#                                request's forward count exactly. That is how the
#                                repro harness attributes TPF per question.
_TPF_PATH = os.environ.get("TPF_COUNTER_PATH", "")
_TPF_LOCK = _threading.Lock()

# Opt-in per-iteration trace (SGLANG_JT_TRACE_PATH): one JSON line per block
# iteration with the M2T / T2T decisions, so a JointThreshold run can be diffed
# step by step against SDARPrefixKV's SGLANG_SDAR_TRACE_DIR trace. Off by default.
_JT_TRACE_PATH = os.environ.get("SGLANG_JT_TRACE_PATH", "")


def _jt_trace(record: dict) -> None:
    if not _JT_TRACE_PATH:
        return
    try:
        import json

        with open(_JT_TRACE_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass


def _tpf_flush(stats):
    if not _TPF_PATH:
        return
    try:
        os.makedirs(os.path.dirname(_TPF_PATH), exist_ok=True)
        tmp = _TPF_PATH + ".tmp"
        with open(tmp, "w") as f:
            f.write("%d %d\n" % (stats["total_forwards"], stats.get("run_calls", 0)))
        os.replace(tmp, _TPF_PATH)
    except Exception:
        # Telemetry must never take the server down.
        pass
# --- end TPF instrumentation ---

class JointThreshold(DllmAlgorithm):

    def __init__(
        self,
        config: DllmConfig,
    ):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.5)
        self.edit_threshold = config.algorithm_config.get("edit_threshold", 0)
        self.max_post_edit_steps = config.algorithm_config.get(
            "max_post_edit_steps", 16
        )
        self.penalty_lambda = config.algorithm_config.get("penalty_lambda", 0)
        # steps: number of denoising steps per block.
        # Default = block_size → 1 token per step (most conservative, matches HF default).
        # Smaller steps = faster but potentially lower quality.
        self.steps = config.algorithm_config.get("steps", self.block_size)
        # Pre-compute per-step transfer schedule
        self._transfer_schedule = _get_num_transfer_tokens(self.block_size, self.steps)

    def cleanup_request(self, req_pool_idx: int):
        pass

    def _fwd(self, model_runner, forward_batch):
        with _TPF_LOCK:
            self._stats["total_forwards"] += 1
        return model_runner.forward(forward_batch, pp_proxy_tensors=None)

    def run(self, model_runner, forward_batch, overlap_fn=None):
        # tp_worker._forward_batch_generation_dllm always passes overlap_fn (as it
        # does for SDARPrefixKV.run). The TPF wrapper added in the +58-line
        # instrumentation commit dropped it, so every JointThreshold request died
        # with "unexpected keyword argument 'overlap_fn'"; an earlier benchmark
        # only ran because its copy of this file had been patched by hand.
        # Forward it.
        with _TPF_LOCK:
            self._stats["run_calls"] = self._stats.get("run_calls", 0) + 1
        try:
            return self._run_inner(model_runner, forward_batch, overlap_fn=overlap_fn)
        finally:
            _tpf_flush(self._stats)

    def _run_inner(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        overlap_fn=None,
    ) -> tuple[LogitsProcessorOutput | torch.Tensor, torch.Tensor | None, bool]:
        batch_size = forward_batch.batch_size
        device = forward_batch.input_ids.device

        mask_index = forward_batch.input_ids == self.mask_id
        if not mask_index.any():
            out = self._fwd(model_runner, forward_batch)
            return out.logits_output, [], out.can_run_graph

        start_list = []
        prompt_masks = []
        for i in range(batch_size):
            block_start = i * self.block_size
            block_end = block_start + self.block_size
            block_input_ids = forward_batch.input_ids[block_start:block_end]

            prompt_mask = block_input_ids != self.mask_id
            prompt_masks.append(prompt_mask)
            start_list.append(prompt_mask.sum().item())

        post_edit_steps = torch.zeros(batch_size, dtype=torch.int32, device=device)
        # Track denoising step index per batch item (for scheduled transfer)
        denoise_step = torch.zeros(batch_size, dtype=torch.int32, device=device)

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        any_changed_in_last_step = False

        schedule = self._transfer_schedule.to(device)
        max_iterations = self.steps + self.max_post_edit_steps

        for _ in range(max_iterations):
            if finished.all():
                break

            out = self._fwd(model_runner, forward_batch)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

            any_changed_in_last_step = False

            for i in range(batch_size):
                if finished[i]:
                    continue

                block_start = i * self.block_size
                block_end = block_start + self.block_size

                curr_input_ids = forward_batch.input_ids[block_start:block_end]
                curr_logits = logits_output.full_logits[block_start:block_end]
                curr_prompt_mask = prompt_masks[i]

                if self.penalty_lambda > 0:
                    prev_ids = curr_input_ids[:-1]
                    curr_logits[1:, :].scatter_(
                        1, prev_ids.unsqueeze(-1), -self.penalty_lambda, reduce="add"
                    )

                x = torch.argmax(curr_logits, dim=-1)
                p = torch.squeeze(
                    torch.gather(
                        F.softmax(curr_logits, dim=-1),
                        dim=-1,
                        index=torch.unsqueeze(x, -1),
                    ),
                    -1,
                )

                curr_mask_index = curr_input_ids == self.mask_id
                has_mask = curr_mask_index.any()

                # Mask to token (M2T) with scheduled transfer
                mask_transfer_index = torch.zeros_like(curr_mask_index)
                if has_mask:
                    step_idx = min(int(denoise_step[i].item()), self.steps - 1)
                    num_to_transfer = int(schedule[step_idx].item())
                    num_available = int(curr_mask_index.sum().item())
                    num_to_transfer = min(num_to_transfer, num_available)

                    confidence = torch.where(curr_mask_index, p, torch.tensor(-float('inf'), device=device))
                    high_conf = (confidence > self.threshold) & curr_mask_index
                    num_high = int(high_conf.sum().item())

                    if num_high >= num_to_transfer:
                        # Enough high-confidence tokens: commit all of them
                        mask_transfer_index = high_conf
                    else:
                        # Not enough: take top-k by confidence
                        _, idx = torch.topk(confidence, k=num_to_transfer)
                        mask_transfer_index[idx] = True

                    denoise_step[i] += 1
                else:
                    post_edit_steps[i] += 1
                    if post_edit_steps[i] > self.max_post_edit_steps:
                        finished[i] = True
                        continue

                # Token to token (T2T) editing
                edit_mask = ~curr_mask_index & ~curr_prompt_mask
                edit_transfer_index = (
                    (p > self.edit_threshold) & (curr_input_ids != x) & edit_mask
                )

                transfer_index = mask_transfer_index | edit_transfer_index
                if _JT_TRACE_PATH:
                    _before = curr_input_ids.tolist()
                    _jt_trace({
                        "batch_id": i,
                        "block_pos0": int(forward_batch.positions[block_start].item())
                        if getattr(forward_batch, "positions", None) is not None else None,
                        "denoise_step": int(denoise_step[i].item()),
                        "post_edit_steps": int(post_edit_steps[i].item()),
                        "num_mask_before": int(curr_mask_index.sum().item()),
                        "tokens_before": _before,
                        "proposal_ids": x.tolist(),
                        "proposal_probs": [round(float(v), 4) for v in p.tolist()],
                        "accept_positions": torch.nonzero(mask_transfer_index).flatten().tolist(),
                        "edit_positions": torch.nonzero(edit_transfer_index).flatten().tolist(),
                        "edits": [[int(j), int(_before[j]), int(x[j].item()), round(float(p[j].item()), 4)]
                                  for j in torch.nonzero(edit_transfer_index).flatten().tolist()],
                    })
                if not transfer_index.any():
                    finished[i] = True
                    continue

                curr_input_ids[transfer_index] = x[transfer_index]
                any_changed_in_last_step = True

        if any_changed_in_last_step:
            out = self._fwd(model_runner, forward_batch)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

        next_token_ids_list = []
        for i in range(batch_size):
            block_start = i * self.block_size
            block_end = block_start + self.block_size
            block_ids = forward_batch.input_ids[block_start:block_end]
            next_token_ids_list.append(block_ids[start_list[i]:])

        return logits_output, next_token_ids_list, can_run_cuda_graph


Algorithm = JointThreshold
