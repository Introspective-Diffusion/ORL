"""
DiffusionRolloutWorker – SDAR block-diffusion rollout using jetengine_ext.

Mirrors the logic of ~/diffusion-reasoning/sample/sdar_rl_rollout.py but runs
inline (no subprocess) and returns the results as a verl DataProto so the
trainer can pass them directly to the reward and actor workers.

Per-sequence outputs stored in DataProto.non_tensor_batch:
  "prompt_token_ids"      – list[list[int]]
  "completion_token_ids"  – list[list[int]]
  "step_maps"             – list[list[int]]  (first_unmask_times per token)
  "hidden_states_paths"   – list[str | None] (.pt file path, or None)
  "full_texts"            – list[str]         decoded generation text
  "rollout_replay_trace"  – list[list[dict]]  opt-in strict replay trace
"""

import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from verl.workers.config import FSDPCriticConfig, FSDPEngineConfig, HFModelConfig, RolloutConfig

# Make the vendored jetengine_ext importable as a top-level package.
# Its internal modules use `from jetengine_ext.xxx import ...` so the
# parent directory of the vendored copy must be on sys.path.
_je_parent = str(Path(__file__).resolve().parent)
if _je_parent not in sys.path:
    sys.path.insert(0, _je_parent)

from verl import DataProto


def to_1d_object_array_of_items(items) -> np.ndarray:
    """
    Force a 1-D object array where each element is exactly one item from `items`.
    Works well for list-of-lists (variable or fixed length).
    """
    items_list = list(items)  # important: materialize iterator
    arr = np.empty((len(items_list),), dtype=object)
    arr[:] = items_list
    return arr


def _detach_cpu_tensor(tensor: torch.Tensor, pin_memory: bool = False) -> torch.Tensor:
    tensor = tensor.detach().cpu()
    if pin_memory and torch.cuda.is_available():
        try:
            tensor = tensor.pin_memory()
        except RuntimeError:
            pass
    return tensor


def compact_self_conditioning_states(states, pin_memory: bool = False):
    """Keep only rollout self-conditioning tensors needed by replay."""
    if not isinstance(states, dict):
        return {}

    compact = {}
    for step, item in states.items():
        if not isinstance(item, dict):
            continue
        signal = item.get("self_conditioning_signal", None)
        logits_source_hidden = item.get("logits_source_hidden", None)
        if signal is None and logits_source_hidden is None:
            continue

        valid = item.get("valid", None)
        prompt_mask = item.get("prompt_mask", None)
        input_ids = item.get("input_ids", None)
        if isinstance(signal, torch.Tensor):
            signal = _detach_cpu_tensor(signal, pin_memory=pin_memory)
        if isinstance(logits_source_hidden, torch.Tensor):
            logits_source_hidden = _detach_cpu_tensor(logits_source_hidden, pin_memory=pin_memory)
        if isinstance(valid, torch.Tensor):
            valid = _detach_cpu_tensor(valid, pin_memory=pin_memory)
        if isinstance(prompt_mask, torch.Tensor):
            prompt_mask = _detach_cpu_tensor(prompt_mask, pin_memory=pin_memory)
        if isinstance(input_ids, torch.Tensor):
            input_ids = _detach_cpu_tensor(input_ids, pin_memory=pin_memory)

        compact[int(step)] = {
            "position": item.get("position", None),
            "valid": valid,
            "prompt_mask": prompt_mask,
            "input_ids": input_ids,
            "self_conditioning_signal": signal,
            "logits_source_hidden": logits_source_hidden,
        }
    return compact


def summarize_step_events(events, num_sequences: int) -> dict[str, float]:
    """Summarize JetEngine denoise forwards as tokens-per-forward metrics."""
    if not events:
        return {}

    denoise_events = [event for event in events if event.get("kind") == "jetengine_denoise"]
    prefill_events = [event for event in events if event.get("kind") == "jetengine_prefill"]
    if not denoise_events:
        return {}

    accepted = np.asarray([event.get("accepted_tokens", 0) for event in denoise_events], dtype=float)
    forward_tokens = np.asarray([event.get("tokens", 0) for event in denoise_events], dtype=float)
    seconds = np.asarray([event.get("seconds", 0.0) for event in denoise_events], dtype=float)
    denom_sequences = max(int(num_sequences), 1)
    total_seconds = float(seconds.sum())

    summary = {
        "rollout/tpf/accepted_per_denoise_forward_mean": float(accepted.mean()),
        "rollout/tpf/accepted_per_denoise_forward_min": float(accepted.min()),
        "rollout/tpf/accepted_per_denoise_forward_max": float(accepted.max()),
        "rollout/tpf/forward_tokens_per_denoise_forward_mean": float(forward_tokens.mean()),
        "rollout/tpf/denoise_forwards_per_sequence": float(len(denoise_events) / denom_sequences),
        "rollout/tpf/prefill_forwards_per_sequence": float(len(prefill_events) / denom_sequences),
    }
    if total_seconds > 0:
        summary["rollout/tpf/accepted_tokens_per_second"] = float(accepted.sum() / total_seconds)
        summary["rollout/tpf/forward_tokens_per_second"] = float(forward_tokens.sum() / total_seconds)
    return summary


class DiffusionRolloutWorker:
    """Rollout worker for SDAR block-diffusion models.

    Args:
        model_path: HF model path or local directory.
        block_size: Tokens denoised per step (``block_length`` in jetengine).
        denoising_steps: Denoising iterations per block.
        tensor_parallel_size: TP degree for jetengine_ext.
        max_active: Max concurrent sequences inside the engine.
        temperature: Sampling temperature.
        top_p / top_k: Nucleus / top-k sampling parameters.
        remasking_strategy: ``"low_confidence_dynamic"`` or ``"low_confidence_static"``.
        dynamic_threshold: Confidence threshold for dynamic remasking.
        hidden_states_dir: Directory to save residual hidden state `.pt` files.
            Falls back to ``/dev/shm/<project_name>/temp_data/`` if None.
        mask_token_id: Override mask token id (defaults to tokenizer.mask_token_id).
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig
    ):
        self.config = config
        self.model_config = model_config
        self._init_engine()

    # ------------------------------------------------------------------
    def _init_engine(self):
        import torch.distributed as dist
        from transformers import AutoTokenizer
        from jetengine_ext.llm import LLM

        # With tensor_parallel_size=1, ModelRunner runs in the main process and
        # hardcodes `torch.cuda.set_device(rank=0)`.  In a multi-GPU setup each
        # verl actor worker is assigned a different LOCAL_RANK GPU.  We redirect
        # set_device so ModelRunner allocates its tensors on the correct device.
        _target_device = torch.cuda.current_device()
        _real_set_device = torch.cuda.set_device
        def _redirect_set_device(device):
            _real_set_device(_target_device)
        torch.cuda.set_device = _redirect_set_device

        # jetengine_ext's ModelRunner unconditionally calls dist.init_process_group.
        # When running inside a verl Ray worker the group is already initialized,
        # so we temporarily make it a no-op to avoid "initialized twice" errors.
        _real_init = dist.init_process_group
        def _noop_if_initialized(backend, init_method=None, *args, **kwargs):
            if dist.is_initialized():
                return   # already initialized — skip
            return _real_init(backend, init_method, *args, **kwargs)
        dist.init_process_group = _noop_if_initialized

        # jetengine_ext's SDARAttention and ColumnParallelLinear call
        # dist.get_world_size() / dist.get_rank() at __init__ time to determine
        # tensor-parallel (TP) sharding.  When running inside a verl Ray worker
        # the default process group has world_size == n_gpus (data-parallel, not
        # TP), so those calls would return 4 (or however many GPUs) instead of
        # tensor_parallel_size (1).  That causes:
        #   • ColumnParallelLinear to allocate 1/n_gpus weight shards → each
        #     worker gets a different 1/4 of the model weights.
        #   • SDARAttention to use num_kv_heads // n_gpus → KV-cache stride
        #     mismatch → AssertionError in store_kvcache.
        # Fix: temporarily report world_size=tensor_parallel_size and rank=0
        # during the LLM constructor so the engine builds a fully-replicated
        # (TP=1) model on every data-parallel worker.
        _real_get_world_size = dist.get_world_size
        _real_get_rank = dist.get_rank
        _tp = self.config.tensor_model_parallel_size
        dist.get_world_size = lambda group=None: _tp if group is None else _real_get_world_size(group)
        dist.get_rank = lambda group=None: 0 if group is None else _real_get_rank(group)
        model_path = getattr(self.model_config, "local_path", None) or self.model_config.path

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, use_fast=True, trust_remote_code=True
            )
            if torch.cuda.is_available() and self.config.engine_kwargs.get("log_memory_on_init", True):
                # allocated memory (currently used by tensors)
                allocated = torch.cuda.memory_allocated()

                # peak allocated memory
                max_allocated = torch.cuda.max_memory_allocated()

                # cached memory (held by the PyTorch allocator)
                cached = torch.cuda.memory_reserved()

                # peak cached memory
                max_cached = torch.cuda.max_memory_reserved()

                # convert to MB
                print(f"allocated: {allocated / 1024 ** 2:.2f} MB")
                print(f"max allocated: {max_allocated / 1024 ** 2:.2f} MB")
                print(f"cached: {cached / 1024 ** 2:.2f} MB")
                print(f"max cached: {max_cached / 1024 ** 2:.2f} MB")
            self.llm = LLM(
                model_path,
                max_num_batched_tokens=self.config.max_num_batched_tokens,
                max_num_seqs=self.config.max_num_seqs,
                max_model_len=self.config.max_model_len,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                tensor_parallel_size=self.config.tensor_model_parallel_size,
                enforce_eager=self.config.enforce_eager,
                mask_token_id=self.config.engine_kwargs.get("mask_token_id", 151669),
                block_length=self.config.engine_kwargs.get("block_length", 4),
                use_self_conditioning=self.config.engine_kwargs.get("use_self_conditioning", False),
                self_conditioning_residual_scale=self.config.engine_kwargs.get("self_conditioning_residual_scale", None),
                self_conditioning_post_norm=self.config.engine_kwargs.get("self_conditioning_post_norm", None),
                self_conditioning_post_norm_scale=self.config.engine_kwargs.get("self_conditioning_post_norm_scale", None),
                self_conditioning_mask_mode=self.config.engine_kwargs.get("self_conditioning_mask_mode", "mask_token"),
                record_self_conditioning_debug=(
                    self.config.engine_kwargs.get("record_self_conditioning_debug", False)
                    or self.config.engine_kwargs.get("allow_accepted_change", False)
                ),
                prefill_causal=self.config.engine_kwargs.get("prefill_causal", True),
                commit_causal=self.config.engine_kwargs.get("commit_causal", False),
            )
        finally:
            torch.cuda.set_device = _real_set_device      # always restore
            dist.init_process_group = _real_init           # always restore
            dist.get_world_size = _real_get_world_size     # always restore
            dist.get_rank = _real_get_rank                 # always restore
        print(
            f"[DiffusionRolloutWorker] model={model_path}  "
            f"block={self.config.engine_kwargs.get('block_length', 4)}  steps={self.config.engine_kwargs.get('denoising_steps', 4)}  "
            f"tp={self.config.tensor_model_parallel_size}"
        )

    # ------------------------------------------------------------------
    def generate_sequences(
        self,
        prompts: DataProto,
    ) -> DataProto:
        """Generate sequences for a batch of prompts.
        # ... existing docstring ...
        """
        from jetengine_ext.sampling_params import SamplingParams
        import uuid

        # ---- decode prompt texts ----
        if prompts.non_tensor_batch is not None and "raw_prompt" in prompts.non_tensor_batch:
            # Modern verl RLHFDataset returns raw chat messages; apply chat template.
            prompt_texts = [
                self.tokenizer.apply_chat_template(
                    list(messages), tokenize=False, add_generation_prompt=True
                )
                for messages in prompts.non_tensor_batch["raw_prompt"]
            ]
        elif prompts.non_tensor_batch is not None and "prompts" in prompts.non_tensor_batch:
            prompt_texts: List[str] = list(prompts.non_tensor_batch["prompts"])
        elif prompts.batch is not None and "input_ids" in prompts.batch:
            prompt_texts = [
                self.tokenizer.decode(row.tolist(), skip_special_tokens=False)
                for row in prompts.batch["input_ids"]
            ]
        else:
            raise ValueError(
                "prompts DataProto must have non_tensor_batch['raw_prompt'], "
                "non_tensor_batch['prompts'], or batch['input_ids']"
            )

        N = len(prompt_texts)

        # ---- generate ----
        sp = SamplingParams(
            temperature=self.config.temperature,
            max_tokens=self.config.response_length,
            ignore_eos=self.config.get("ignore_eos", False),
            block_length=self.config.engine_kwargs.get("block_length", 4),
            denoising_steps=self.config.engine_kwargs.get("denoising_steps", 4),
            max_steps_per_block=self.config.engine_kwargs.get("max_steps_per_block", None),
            num_transfer_per_step=self.config.engine_kwargs.get("num_transfer_per_step", 0),
            dynamic_threshold=self.config.engine_kwargs.get("dynamic_threshold", 0.9),
            eb_threshold=self.config.engine_kwargs.get("eb_threshold", 0.35),
            topk=self.config.top_k,
            topp=self.config.top_p,
            remasking_strategy=self.config.engine_kwargs.get("remasking_strategy", "low_confidence_dynamic"),
            force_accept_leftmost=self.config.engine_kwargs.get("force_accept_leftmost", False),
            prefill_full_prompt=self.config.engine_kwargs.get("prefill_full_prompt", False),
            noise_mode=self.config.engine_kwargs.get("noise_mode", "mask"),
            random_token_high=self.config.engine_kwargs.get("random_token_high", None),
            deterministic_randtok=self.config.engine_kwargs.get("deterministic_randtok", False),
            randtok_seed=self.config.engine_kwargs.get("randtok_seed", 0),
            allow_accepted_change=self.config.engine_kwargs.get("allow_accepted_change", False),
            edit_threshold=self.config.engine_kwargs.get("edit_threshold", None),
            exclude_prompt_tail_from_stop=self.config.engine_kwargs.get("exclude_prompt_tail_from_stop", False),
            truncate_on_stop=self.config.engine_kwargs.get("truncate_on_stop", False),
            record_replay_trace=self.config.engine_kwargs.get("record_replay_trace", False),
            replay_trace_full=self.config.engine_kwargs.get("replay_trace_full", False),
            stop_words=[]
        )
        self.llm.step_events = []
        _t0 = time.perf_counter()
        outputs = self.llm.generate_streaming(
            prompt_texts,
            sp,
            max_active=self.config.engine_kwargs.get("max_active", 32),
            use_tqdm=self.config.engine_kwargs.get("show_progress", True),
        )
        _generate_sequences_s = time.perf_counter() - _t0
        step_events = list(getattr(self.llm, "step_events", []))
        tpf_metrics = summarize_step_events(step_events, N)


        # ---- collect results (maintain original order via index) ----
        prompt_ids: List = [None] * N
        response_ids: List = [None] * N
        step_maps: List = [None] * N
        log_probs: List = [None] * N
        full_texts: List = [""] * N
        residual_hidden_states = [None] * N
        replay_traces = [None] * N

        # NEW: per-generated-sequence uuid
        response_uuid_list: List[str] = [str(uuid.uuid4()) for _ in range(N)]

        for idx, out in enumerate(outputs):
            text = out.get("text", "")
            p_ids = out.get("prompt_ids", [])
            c_ids = out.get("response_ids", [])
            steps = out.get("first_unmask_times", None)
            logprobs = out.get("token_logprobs", None)
            residual_h = out.get("residual_hidden_states", None)
            replay_trace = out.get("replay_trace", None)
            if steps is None:
                steps = list(range(len(c_ids)))

            if logprobs is None:
                logprobs = [-float("inf") for _ in range(len(c_ids))]
            assert len(c_ids) == len(steps), (
                f"completion_token_ids length {len(c_ids)} != "
                f"step_map length {len(steps)}"
            )

            prompt_ids[idx] = p_ids
            response_ids[idx] = c_ids
            step_maps[idx] = steps
            log_probs[idx] = logprobs
            residual_hidden_states[idx] = compact_self_conditioning_states(
                residual_h,
                pin_memory=self.config.engine_kwargs.get("pin_self_conditioning_cpu", False),
            )
            replay_traces[idx] = replay_trace
            full_texts[idx] = text

        # replace None with empty defaults
        for i in range(N):
            if prompt_ids[i] is None:
                prompt_ids[i] = []
            if response_ids[i] is None:
                response_ids[i] = []
            if step_maps[i] is None:
                step_maps[i] = []
            if log_probs[i] is None:
                log_probs[i] = []
        meta_info = {
            "metrics": [
                {
                    "generate_sequences": _generate_sequences_s,
                    "tool_calls": 0.0,
                    "num_preempted": -1,
                    **tpf_metrics,
                }
            ] * N,
        }
        non_tensors = {
            "prompt_ids": to_1d_object_array_of_items(prompt_ids),  # list[list[int]]
            "response_ids": to_1d_object_array_of_items(response_ids),  # list[list[int]]
            "step_maps": to_1d_object_array_of_items(step_maps),  # list[list[int]]
            "full_texts": to_1d_object_array_of_items(full_texts),  # list[str]
            "response_uid": to_1d_object_array_of_items(response_uuid_list),  # list[str]
            "token_logprobs": to_1d_object_array_of_items(log_probs),
            "rollout_self_conditioning": to_1d_object_array_of_items(residual_hidden_states),
            "rollout_step_events": to_1d_object_array_of_items([step_events] * N),
        }
        if self.config.engine_kwargs.get("record_replay_trace", False):
            non_tensors["rollout_replay_trace"] = to_1d_object_array_of_items(replay_traces)
        return DataProto.from_dict(tensors={}, non_tensors=non_tensors, meta_info=meta_info)


    def cleanup(self):
        """Free all GPU memory held by the rollout engine.

        With tensor_parallel_size=1, ModelRunner runs in the MAIN PROCESS.
        Calling exit() would invoke dist.destroy_process_group(), which breaks
        any active FSDP process group.  Instead we manually delete every large
        GPU tensor and let the atexit handler call exit() at program shutdown
        (when destroying the group is safe).

        Tensors freed here:
          model_runner.model                  – SDAR model weights (~15 GB)
          model_runner.kv_cache               – KV-cache blocks
          model_runner.p                      – logit distribution store (~23 GB)
          model_runner.residual_hidden_states
          model_runner.valid
          scheduler.block_manager.{p,valid,residual_hidden_states}
              (shared references to the same tensors above)
        """
        import gc
        if self.llm is not None:
            try:
                mr = self.llm.model_runner
                # 1. Clear k_cache / v_cache views inside each attention layer.
                #    They are slices of mr.kv_cache and must be released first.
                try:
                    for layer in mr.model.model.layers:
                        for attr in ("k_cache", "v_cache"):
                            if hasattr(layer.self_attn, attr):
                                delattr(layer.self_attn, attr)
                except Exception:
                    pass
                # 2. Delete the model object (releases its weights + internal refs).
                try:
                    del mr.model
                except Exception:
                    pass
                # 3. Delete remaining large GPU tensors from model_runner.
                for attr in ("kv_cache", "p", "residual_hidden_states", "valid"):
                    try:
                        if hasattr(mr, attr):
                            delattr(mr, attr)
                    except Exception:
                        pass
            except Exception:
                pass
            # 4. Release shared references still held by the scheduler.
            try:
                bm = self.llm.scheduler.block_manager
                for attr in ("p", "valid", "residual_hidden_states"):
                    try:
                        if hasattr(bm, attr):
                            delattr(bm, attr)
                    except Exception:
                        pass
            except Exception:
                pass
            self.llm = None
        gc.collect()
        torch.cuda.empty_cache()
