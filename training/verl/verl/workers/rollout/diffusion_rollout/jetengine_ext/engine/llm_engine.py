import atexit
import json
import os
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp
# Added imports for profiling
import torch
from torch import nn
from contextlib import nullcontext
import torch.profiler as torch_profiler

from jetengine_ext.config import Config
from jetengine_ext.sampling_params import SamplingParams
from jetengine_ext.engine.sequence import Sequence, RunType
from jetengine_ext.engine.scheduler import Scheduler
from jetengine_ext.engine.model_runner import ModelRunner
from jetengine_ext.utils.loader import load_from_hf_model


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        explicit_mask_token_id = config.mask_token_id
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True, trust_remote_code=True)
        config.eos = self.tokenizer.eos_token_id
        if explicit_mask_token_id != -1:
            config.mask_token_id = explicit_mask_token_id
        else:
            config.mask_token_id = self.tokenizer.mask_token_id if self.tokenizer.mask_token_id is not None else self.tokenizer.pad_token_id
        assert config.mask_token_id is not None, "Model tokenizer must have a mask_token_id or pad_token_id"

        self.config = config
        self.scheduler = Scheduler(config)
        self.scheduler.consistent_sampling_params = False
        self.step_events = []
        # self.scheduler.block_manager.p = self.model_runner.p
        self.scheduler.block_manager.valid = self.model_runner.valid
        self.scheduler.block_manager.residual_hidden_states = self.model_runner.residual_hidden_states
        atexit.register(self.exit)
        self._last_step_had_work = False
        self._last_step_run_type = None
        self._last_no_work_diagnostics = None
        self.no_progress_step_limit = max(1, int(os.environ.get("JETENGINE_NO_PROGRESS_LIMIT", "100")))

    def offload_parameters(self, include_buffers: bool = False):
        """
        Replace all parameter (and buffer) storages with meta tensors.
        Keeps shapes/dtypes, frees GPU/CPU memory.
        """

        def offload_parameters_keep_buffers(model: torch.nn.Module):
            """
            Move *parameters* to meta to free memory while keeping buffers unchanged.
            Works for any module tree.
            """
            # 1) Snapshot real buffers (module reference + buffer name + tensor)
            saved_buffers = []
            for mod in model.modules():
                for bname, buf in list(mod._buffers.items()):
                    if buf is not None:
                        saved_buffers.append((mod, bname, buf))

            # 2) Move everything to meta
            model.to_empty(device=torch.device("meta"))

            # 3) Restore the saved, real buffers
            for mod, bname, buf in saved_buffers:
                # Reattach the original tensor (device/dtype preserved)
                mod._buffers[bname] = buf

            torch.cuda.empty_cache()
        if include_buffers:
            self.model_runner.model.to_empty(device=torch.device("meta"))
        else:
            offload_parameters_keep_buffers(self.model_runner.model)

        print("Successfully cleaned old parameters (buffers kept)." if not include_buffers
              else "Successfully cleaned old parameters and buffers.")

    def reload_parameters(self, hf_model: nn.Module):
        load_from_hf_model(self.model_runner.model, hf_model=hf_model)

    def exit(self):
        model_runner = getattr(self, "model_runner", None)
        if model_runner is not None:
            model_runner.call("exit")
            del self.model_runner
        for p in getattr(self, "ps", []):
            p.join()
        self.ps = []

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        if isinstance(prompt, list):
            if self.tokenizer.pad_token_id in prompt:
                start = prompt.index(self.tokenizer.pad_token_id) + 1
                prompt = prompt[start:]
        seq = Sequence(prompt, self.config.mask_token_id, sampling_params)
        seq.eos_token_id = self.tokenizer.eos_token_id
        self.scheduler.add(seq)

    def _scheduler_debug_state(self, pending_idx: int | None = None, total: int | None = None) -> dict:
        debug_state = getattr(self.scheduler, "debug_state", None)
        if callable(debug_state):
            return debug_state(pending_idx=pending_idx, total=total)
        running = getattr(self.scheduler, "running", [])
        block_manager = getattr(self.scheduler, "block_manager", None)
        return {
            "pending_idx": pending_idx,
            "total": total,
            "running": len(running),
            "free_blocks": len(getattr(block_manager, "free_block_ids", []) or []),
            "used_blocks": len(getattr(block_manager, "used_block_ids", []) or []),
            "sequences": [
                {
                    "seq_id": int(getattr(seq, "seq_id", -1)),
                    "status": getattr(getattr(seq, "status", None), "name", str(getattr(seq, "status", None))),
                    "num_tokens": int(getattr(seq, "num_tokens", -1)),
                    "num_completion_tokens": int(getattr(seq, "num_completion_tokens", -1)),
                    "max_tokens": int(getattr(seq, "max_tokens", -1)),
                    "block_table_len": len(getattr(seq, "block_table", []) or []),
                }
                for seq in running
            ],
        }

    def _raise_no_progress(
        self,
        where: str,
        consecutive_no_work_steps: int,
        pending_idx: int | None = None,
        total: int | None = None,
    ):
        diagnostics = self._scheduler_debug_state(pending_idx=pending_idx, total=total)
        self._last_no_work_diagnostics = diagnostics
        raise RuntimeError(
            "JetEngine scheduler made no progress "
            f"for {consecutive_no_work_steps} consecutive steps in {where}. "
            "This would otherwise busy-spin. Diagnostics: "
            f"{json.dumps(diagnostics, sort_keys=True)}"
        )

    def step(self):
        self._last_step_had_work = False
        self._last_step_run_type = None
        scheduled_seqs, run_type = self.scheduler.schedule()
        if scheduled_seqs is None:
            self._last_no_work_diagnostics = self._scheduler_debug_state()
            return [], 0 # Nothing to run

        self._last_step_had_work = True
        self._last_step_run_type = run_type
        if run_type == RunType.PREFILL:
            forward_tokens = sum(len(seq) for seq in scheduled_seqs)
            active_blocks = sum(seq.num_blocks for seq in scheduled_seqs)
        else:
            forward_tokens = len(scheduled_seqs) * self.config.block_length
            active_blocks = len(scheduled_seqs)

        start = perf_counter()
        logits = self.model_runner.call("run", scheduled_seqs, run_type)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = perf_counter() - start
        self.scheduler.postprocess(scheduled_seqs, logits, run_type)
        
        #finished_outputs = [(seq.seq_id, seq.completion_token_ids) for seq in scheduled_seqs if seq.is_finished]
        
        finished_outputs = [
            (
                seq.seq_id,
                seq.prompt_token_ids,
                seq.completion_token_ids,
                seq.first_unmask_steps,
                seq.token_logprobs,
                seq.input_hidden_states,
                seq.replay_trace if getattr(seq, "record_replay_trace", False) else None,
            )
            for seq in scheduled_seqs
            if seq.is_finished
        ]

        accepted_tokens = sum(getattr(seq, "num_to_transfer", 0) for seq in scheduled_seqs)
        masked_tokens_remaining = 0
        if run_type == RunType.DENOISE:
            for seq in scheduled_seqs:
                masked_tokens_remaining += sum(1 for token_id in seq.intermediate_block_tokens if token_id == self.config.mask_token_id)
        self.step_events.append(
            {
                "kind": f"jetengine_{run_type.name.lower()}",
                "tokens": int(forward_tokens),
                "seconds": float(elapsed),
                "active_sequences": int(len(scheduled_seqs)),
                "active_blocks": int(active_blocks),
                "accepted_tokens": int(accepted_tokens),
                "masked_tokens_remaining": int(masked_tokens_remaining),
            }
        )
        return finished_outputs, accepted_tokens

    def is_finished(self):
        return self.scheduler.is_finished()




    def _clean_token_ids(self, token_ids):
        # Accept tensors, numpy ints, etc.
        try:
            token_ids = list(token_ids)
        except Exception:
            token_ids = [token_ids]
        
        vocab_size = getattr(self.tokenizer, "vocab_size", None)
        special_ids = set(getattr(self.tokenizer, "all_special_ids", []) or [])
        mask_id = getattr(self.config, "mask_token_id", None)

        cleaned = []
        for t in token_ids:
            if t is None or t < 0 or t == mask_id or t >= vocab_size:
                if t not in special_ids:
                    cleaned.append(0)
                    continue
            cleaned.append(t)
        return cleaned

    def _clean_response_triplet(self, response_ids, unmask_times, token_logprobs):
        """Keep response ids, step ids, and logprobs aligned after cleaning."""
        vocab_size = getattr(self.tokenizer, "vocab_size", None)
        special_ids = set(getattr(self.tokenizer, "all_special_ids", []) or [])
        mask_id = getattr(self.config, "mask_token_id", None)
        clean_ids, clean_steps, clean_lps = [], [], []
        for token_id, step, logprob in zip(response_ids, unmask_times, token_logprobs):
            invalid = token_id is None or token_id < 0 or token_id == mask_id or token_id >= vocab_size
            if invalid and token_id not in special_ids:
                token_id = 0
            clean_ids.append(token_id)
            clean_steps.append(step)
            clean_lps.append(logprob)
        return clean_ids, clean_steps, clean_lps

    def _safe_decode(self, token_ids):
        ids = self._clean_token_ids(token_ids)
        # skip_special_tokens can be True or False; doesn't affect the None issue
        return self.tokenizer.decode(ids, skip_special_tokens=False)
    


    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
        # New optional profiling controls
        profile: bool = False,
        profile_dir: str | None = None,
    ) -> list[str]:
        # ... (This method remains largely the same, but the progress bar will update differently) ...
        # The logic inside the `while not self.is_finished()` loop correctly calls `self.step()`
        # and collects outputs.
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
            self.scheduler.consistent_sampling_params = True
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        
        total_generated_tokens = 0
        start_time = perf_counter()

        # Setup profiler context
        activities = [torch_profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch_profiler.ProfilerActivity.CUDA)
        trace_dir = profile_dir or "profiler_traces"
        prof_ctx = (
            torch_profiler.profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                on_trace_ready=torch_profiler.tensorboard_trace_handler(trace_dir),
            )
            if profile else nullcontext()
        )

        with prof_ctx as prof:
            consecutive_no_work_steps = 0
            while not self.is_finished():
                output, num_processed = self.step()
                if not self._last_step_had_work and getattr(self.scheduler, "running", []):
                    consecutive_no_work_steps += 1
                    if consecutive_no_work_steps >= self.no_progress_step_limit:
                        self._raise_no_progress("generate", consecutive_no_work_steps)
                else:
                    consecutive_no_work_steps = 0
                if profile:
                    prof.step()
                total_generated_tokens += num_processed
                
                throughput = total_generated_tokens / (perf_counter() - start_time)
                if use_tqdm:
                    pbar.set_postfix({"Throughput": f"{int(throughput)} tok/s"})

                #for seq_id, token_ids in output:
                #    outputs[seq_id] = token_ids
                for seq_id, prompt_ids, response_ids, unmask_times, token_logprobs, residual_hidden_states, replay_trace in output:
                    outputs[seq_id] = {
                        "prompt_ids": prompt_ids,
                        "response_ids": response_ids,
                        "unmask_times": unmask_times,
                        "token_logprobs": token_logprobs,
                        "residual_hidden_states": residual_hidden_states,
                        "replay_trace": replay_trace,
                    }
                    if use_tqdm:
                        pbar.update(1)

        #outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        #outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        cleaned_outputs = []
        for item in outputs:
            response_ids, unmask_times, token_logprobs = self._clean_response_triplet(
                item["response_ids"], item["unmask_times"], item["token_logprobs"]
            )
            cleaned_outputs.append(
                {
                    "text": self._safe_decode(response_ids),
                    "prompt_ids": item["prompt_ids"],
                    "response_ids": response_ids,
                    "first_unmask_times": unmask_times,
                    "token_logprobs": token_logprobs,
                    "residual_hidden_states": item["residual_hidden_states"],
                    "replay_trace": item.get("replay_trace"),
                }
            )
        outputs = cleaned_outputs

        if use_tqdm:
            pbar.close()
        return outputs

    def generate_streaming(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        max_active: int | None = None,
        use_tqdm: bool = True,
        # New optional profiling controls
        profile: bool = False,
        profile_dir: str | None = None,
    ) -> list[str]:
        """
        Stream prompts through the engine while keeping up to `max_active` sequences running.
        As sequences finish, new prompts are added from the pending list to maximize GPU utilization.
        """
        total = len(prompts)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * total
            self.scheduler.consistent_sampling_params = True

        if max_active is None:
            max_active = getattr(self.scheduler, "max_num_seqs", 32)

        if use_tqdm:
            pbar = tqdm(total=total, desc="Generating", dynamic_ncols=True)

        outputs: dict[int, list[int]] = {}
        pending_idx = 0

        # Prime initial requests up to capacity
        initial = min(max_active, total)
        for i in range(initial):
            self.add_request(prompts[i], sampling_params[i])
        pending_idx = initial

        total_generated_tokens = 0
        start_time = perf_counter()

        # Setup profiler context
        activities = [torch_profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch_profiler.ProfilerActivity.CUDA)
        trace_dir = profile_dir or "profiler_traces"
        prof_ctx = (
            torch_profiler.profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                on_trace_ready=torch_profiler.tensorboard_trace_handler(trace_dir),
            )
            if profile else nullcontext()
        )

        with prof_ctx as prof:
            consecutive_no_work_steps = 0
            while not self.is_finished() or pending_idx < total:
                # Top up to capacity before each step
                running = getattr(self.scheduler, "running", [])
                deficit = max_active - len(running)
                while deficit > 0 and pending_idx < total:
                    self.add_request(prompts[pending_idx], sampling_params[pending_idx])
                    pending_idx += 1
                    deficit -= 1

                output, num_processed = self.step()
                if not self._last_step_had_work and getattr(self.scheduler, "running", []):
                    consecutive_no_work_steps += 1
                    if consecutive_no_work_steps >= self.no_progress_step_limit:
                        self._raise_no_progress(
                            "generate_streaming",
                            consecutive_no_work_steps,
                            pending_idx=pending_idx,
                            total=total,
                        )
                else:
                    consecutive_no_work_steps = 0
                if profile:
                    prof.step()
                total_generated_tokens += num_processed

                if use_tqdm:
                    throughput = total_generated_tokens / (perf_counter() - start_time + 1e-6)
                    pbar.set_postfix({"Throughput": f"{int(throughput)} tok/s"})
                    pbar.update(len(output))

                #for seq_id, token_ids in output:
                #    outputs[seq_id] = token_ids
                for seq_id, prompt_ids, response_ids, unmask_times, token_logprobs, residual_hidden_states, replay_trace in output:
                    outputs[seq_id] = {
                        "prompt_ids": prompt_ids,
                        "response_ids": response_ids,
                        "unmask_times": unmask_times,
                        "token_logprobs": token_logprobs,
                        "residual_hidden_states": residual_hidden_states,
                        "replay_trace": replay_trace,
                    }

        #outputs_list = [outputs[seq_id] for seq_id in sorted(outputs)]
        #results = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs_list]
        outputs_list = [outputs[seq_id] for seq_id in sorted(outputs)]
        results = []
        for item in outputs_list:
            response_ids, unmask_times, token_logprobs = self._clean_response_triplet(
                item["response_ids"], item["unmask_times"], item["token_logprobs"]
            )
            results.append(
                {
                    "text": self._safe_decode(response_ids),
                    "prompt_ids": item["prompt_ids"],
                    "response_ids": response_ids,
                    "first_unmask_times": unmask_times,
                    "token_logprobs": token_logprobs,
                    "residual_hidden_states": item["residual_hidden_states"],
                    "replay_trace": item.get("replay_trace"),
                }
            )

        if use_tqdm:
            pbar.close()
        return results