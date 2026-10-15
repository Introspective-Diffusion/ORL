"""Diffusion FSDP actor-rollout-ref worker.

Subclasses the standard ``ActorRolloutRefWorker`` from ``fsdp_workers.py``,
overriding only the rollout-related methods.  The actor training path
(``DataParallelPPOActor.compute_log_prob`` / ``update_policy``) is inherited
as-is, which gives us correct mini-batching, PPO epochs, and loss
normalization via ``agg_loss``.

The key insight is that the SDAR model is a standard causal LM — only
the *generator* (rollout) is diffusion-specific.
"""

import gc
import logging
import random
import time

import numpy
import numpy as np
import ray
import torch
from omegaconf import open_dict
from tensordict import TensorDict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType
from verl import DataProto
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.single_controller.base.decorator import (
    Dispatch,
    make_nd_compute_dataproto_dispatch_fn,
    register,
)
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.fsdp_utils import fsdp_version, load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
from verl.utils.model import convert_weight_keys
from verl.utils.profiler import DistProfiler, DistProfilerExtension, ProfilerConfig, log_gpu_memory_usage, simple_timer
from verl.workers.config import FSDPCriticConfig, FSDPEngineConfig, HFModelConfig, RolloutConfig
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.workers.rollout.diffusion_rollout import DiffusionRolloutWorker

logger = logging.getLogger(__name__)


class DiffusionFSDPActorRolloutRefWorker(ActorRolloutRefWorker):
    """FSDP actor worker with a diffusion-specific rollout engine.

    Inherits the full standard training pipeline (FSDP model, optimizer,
    ``DataParallelPPOActor``, ``compute_log_prob``, ``update_actor``,
    checkpointing) from ``ActorRolloutRefWorker``.

    Overrides only:
    - ``init_model``: patches strategy to "fsdp" and registers a forward
      pre-hook for 2D→4D attention mask expansion.
    - ``_build_rollout``: instantiates ``DiffusionRolloutWorker`` instead of
      vLLM / SGLang.
    - ``generate_sequences``: uses the diffusion rollout engine with GPU
      memory management (free after generation, reinit before next).
    - ``compute_log_prob``: frees rollout GPU memory before delegating to
      the standard pipeline.
    """

    # ------------------------------------------------------------------
    # Override: init_model to patch strategy + register mask hook
    # ------------------------------------------------------------------

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # _build_model_optimizer checks self.config.actor.strategy to decide
        # FSDP vs FSDP2 wrapping.  We use "diffusion" only for worker routing
        # in main_ppo.py; the actual FSDP wrapping should use "fsdp".
        with open_dict(self.config):
            self.config.actor.strategy = "fsdp"
        result = super().init_model()

        if self._is_actor and self.config.actor.get("freeze_embed_lm_head", False):
            frozen = 0
            for name, param in self.actor_module_fsdp.named_parameters():
                if "embed_tokens" in name or "lm_head" in name:
                    param.requires_grad_(False)
                    frozen += 1
            if self.rank == 0:
                print(f"[DiffusionWorker] Froze {frozen} embed/lm_head parameters")

        # Register a forward pre-hook on the FSDP-wrapped model to expand
        # 2D attention masks to 4D causal masks.  This is needed because the
        # SDAR model comments out _update_causal_mask and passes the raw
        # attention_mask directly to F.scaled_dot_product_attention.
        # if self._is_actor:
        #     self.actor_module_fsdp.register_forward_pre_hook(
        #         _sdar_forward_pre_hook, with_kwargs=True
        #     )
        #     logger.info("Registered SDAR 2D→4D attention mask forward pre-hook on actor model.")

        return result

    # ------------------------------------------------------------------
    # Override: build the diffusion rollout instead of vLLM / SGLang
    # ------------------------------------------------------------------

    def _build_rollout(self, trust_remote_code=False):
        # 1. parse rollout and huggingface model config
        rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.rollout)
        model_config: HFModelConfig = omega_conf_to_dataclass(self.config.model, dataclass_type=HFModelConfig)
        # self.model_config = model_config

        # 2. build rollout device mesh
        assert self.config.rollout.data_parallel_size == 1
        assert self.config.rollout.pipeline_model_parallel_size == 1
        assert self.config.rollout.tensor_model_parallel_size == 1

        self.rollout_device_mesh = None

        # 3. Register dispatch info
        # Standard logic: collect if rank is 0 in the TP/PP group
        is_collect = True
        self._register_dispatch_collect_info(
            "rollout", dp_rank=self.rank, is_collect=is_collect
        )

        # 4. Instantiate DiffusionRolloutWorker
        log_gpu_memory_usage(f"Before building {self.config.rollout.name} rollout", logger=logger)
        if self._is_actor and hasattr(self, "actor_module_fsdp"):
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            gc.collect()
            torch.cuda.empty_cache()
            log_gpu_memory_usage(f"After offloading actor before {self.config.rollout.name} rollout", logger=logger)
        self.diffusion_rollout = DiffusionRolloutWorker(
            config=rollout_config, model_config=model_config
        )
        log_gpu_memory_usage(f"Before building {self.config.rollout.name} rollout", logger=logger)

        # Full params
        if fsdp_version(self.actor_module_fsdp) == 1:
            FSDP.set_state_dict_type(
                self.actor_module_fsdp,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(rank0_only=False),
            )

        # No LoRA weight sync needed — the diffusion rollout maintains its
        # own separate model instance.
        self.base_sync_done = True

        # CPU cache for actor weights to sync to engine after reinit.
        self._cached_actor_state_dict = None
        self._initial_weight_sync_skipped = False

    # ------------------------------------------------------------------
    # Weight sync helpers
    # ------------------------------------------------------------------

    # def _extract_actor_state_dict(self):
    #     """Extract a full HF-format state dict from the FSDP actor model.
    #
    #     Returns a dict of CPU tensors suitable for later in-place copy into
    #     the jetengine model.
    #     """
    #     # Load FSDP model to GPU if it was offloaded
    #     if self._is_offload_param:
    #         load_fsdp_model_to_gpu(self.actor_module_fsdp)
    #
    #     # Extract state dict (FSDP handles the all-gather internally)
    #     params = self.actor_module_fsdp.state_dict()
    #
    #     # Apply checkpoint key conversion (e.g. HF transformers key remapping)
    #     params = convert_weight_keys(
    #         params,
    #         getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp),
    #     )
    #
    #     # Move everything to CPU to avoid holding GPU memory
    #     cpu_params = {k: v.detach().cpu() for k, v in params.items()}
    #     del params
    #
    #     # Offload FSDP model back to CPU if needed
    #     if self._is_offload_param:
    #         offload_fsdp_model_to_cpu(self.actor_module_fsdp)
    #
    #     logger.info("Extracted actor state dict with %d keys to CPU.", len(cpu_params))
    #     return cpu_params

    def _extract_actor_state_dict(self):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        extracted_state_dict = {}

        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        if world_size == 1:
            if self._is_offload_param:
                load_fsdp_model_to_gpu(self.actor_module_fsdp)
            try:
                for key, value in self.actor_module_fsdp.state_dict().items():
                    extracted_state_dict[key] = value.detach().cpu()
            finally:
                if self._is_offload_param:
                    offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            return extracted_state_dict

        # Pull the offloaded parameters with summon_full_params, unshard them on GPU, then move them to CPU
        with FSDP.summon_full_params(
                self.actor_module_fsdp,
                writeback=False,
                rank0_only=False,  # every worker needs the full parameters
                offload_to_cpu=True
        ):
            # Note: the unsharded memory is freed when the summon_full_params context exits,
            # so a deep copy via .clone() is required; otherwise the returned tensor becomes invalid.
            for key, value in self.actor_module_fsdp.state_dict().items():
                extracted_state_dict[key] = value.cpu().clone()

        return extracted_state_dict

    def _apply_cached_weights_to_engine(self):
        """Copy cached HF state dict into the jetengine model parameters in-place.

        Handles fused projections (qkv_proj, gate_up_proj) that differ between
        the HF model (separate projections) and jetengine (fused projections).
        """
        sd = self._cached_actor_state_dict
        engine_model = self.diffusion_rollout.llm.model_runner.model
        applied = 0
        skipped = []

        for name, param in engine_model.named_parameters():
            try:
                if "qkv_proj.weight" in name:
                    base = name.replace("qkv_proj.weight", "")
                    q = sd[base + "q_proj.weight"]
                    k = sd[base + "k_proj.weight"]
                    v = sd[base + "v_proj.weight"]
                    param.data.copy_(
                        torch.cat([q, k, v], dim=0).to(device=param.device, dtype=param.dtype)
                    )
                elif "gate_up_proj.weight" in name and "experts." not in name:
                    base = name.replace("gate_up_proj.weight", "")
                    gate = sd[base + "gate_proj.weight"]
                    up = sd[base + "up_proj.weight"]
                    param.data.copy_(
                        torch.cat([gate, up], dim=0).to(device=param.device, dtype=param.dtype)
                    )
                else:
                    # Direct mapping — try name as-is, then strip 'model.' prefix
                    if name in sd:
                        param.data.copy_(sd[name].to(device=param.device, dtype=param.dtype))
                    elif name.startswith("model.") and name[6:] in sd:
                        param.data.copy_(sd[name[6:]].to(device=param.device, dtype=param.dtype))
                    else:
                        skipped.append(name)
                        continue
                applied += 1
            except KeyError as e:
                skipped.append(name)
                logger.warning("Weight sync: missing key for engine param '%s': %s", name, e)

        logger.info(
            "Applied %d actor weights to engine model (%d skipped).",
            applied, len(skipped),
        )
        if skipped:
            logger.debug("Skipped engine params: %s", skipped[:10])

    # ------------------------------------------------------------------
    # Override: generate_sequences using the diffusion rollout
    # ------------------------------------------------------------------

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Generate sequences using the diffusion rollout engine.

        Unlike the parent implementation, we do NOT call ``rollout_mode()`` /
        ``trainer_mode()`` because the diffusion rollout has its own separate
        model — there is no hybrid-engine context switching.
        """
        assert self._is_rollout

        # Reinitialize rollout engine if it was freed after the previous
        # training step (GPU memory management).
        if self.diffusion_rollout.llm is None:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            _fr, _tot = torch.cuda.mem_get_info()
            _stats = torch.cuda.memory_stats()
            _peak = _stats.get("allocated_bytes.all.peak", 0)
            _cur = _stats.get("allocated_bytes.all.current", 0)
            _free_G = _fr / 1e9
            _tot_G = _tot / 1e9
            _overhead_G = 21.0
            _min_kv_G = 2.0
            _gpu_util = 1.0 - (_free_G - _overhead_G - _min_kv_G) / _tot_G + 0.05
            _gpu_util = float(min(0.95, max(0.80, _gpu_util)))
            logger.info(
                "[reinit-diag] free=%.2fG total=%.2fG peak=%.2fG cur=%.2fG gpu_util_new=%.3f",
                _free_G, _tot_G, _peak / 1e9, _cur / 1e9, _gpu_util,
            )
            self.diffusion_rollout.gpu_memory_utilization = _gpu_util
            logger.info("Reinitializing diffusion rollout engine after training step.")
            self.diffusion_rollout._init_engine()

            # Apply cached actor weights to the freshly initialized engine
            if self._cached_actor_state_dict is not None:
                self._apply_cached_weights_to_engine()

        # Generate via the diffusion engine.
        output = self.diffusion_rollout.generate_sequences(prompts=prompts)
        torch.cuda.empty_cache()
        return output

    # ------------------------------------------------------------------
    # Helpers: free rollout GPU memory
    # ------------------------------------------------------------------

    def _free_rollout_gpu(self):
        """Free the diffusion rollout engine from GPU memory."""
        if self._is_rollout and hasattr(self, "diffusion_rollout") and self.diffusion_rollout is not None:
            _pre = torch.cuda.memory_allocated() / 1e9
            logger.info("GPU before rollout free: %.2f GB alloc", _pre)
            self.diffusion_rollout.cleanup()
            gc.collect()
            torch.cuda.empty_cache()
            _post = torch.cuda.memory_allocated() / 1e9
            logger.info("GPU after rollout free: %.2f GB alloc", _post)

    # ------------------------------------------------------------------
    # Override: compute_log_prob — free rollout before training
    # ------------------------------------------------------------------

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def compute_log_prob(self, data: DataProto):
        """Free diffusion rollout, then delegate to standard pipeline."""
        self._free_rollout_gpu()
        return ActorRolloutRefWorker.compute_log_prob(self, data)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def update_actor(self, data: DataProto):
        """Free diffusion rollout before actor update when old-logprob recompute is skipped."""
        self._free_rollout_gpu()
        return ActorRolloutRefWorker.update_actor(self, data)


class AsyncDiffusionFSDPActorRolloutRefWorker(DiffusionFSDPActorRolloutRefWorker):
    """Async variant that syncs actor weights to the diffusion rollout engine.

    After PPO training updates the FSDP actor, ``update_weights()`` extracts
    the full state dict to a CPU cache.  The next ``generate_sequences()``
    call copies those weights into the freshly reinitialized engine in-place.
    """

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self):
        """Extract updated FSDP actor weights and cache them on CPU.

        If the engine is still alive (e.g., first call after checkpoint load),
        apply weights immediately.
        """
        if not hasattr(self, "_initial_weight_sync_skipped"):
            self._initial_weight_sync_skipped = False
        if not hasattr(self, "_cached_actor_state_dict"):
            self._cached_actor_state_dict = None
        if (
                not self._initial_weight_sync_skipped
                and self._cached_actor_state_dict is None
                and self.config.rollout.engine_kwargs.get("skip_initial_weight_sync", False)
                and hasattr(self, "diffusion_rollout")
                and self.diffusion_rollout is not None
                and self.diffusion_rollout.llm is not None
        ):
            # The rollout engine has just loaded the same base checkpoint.  For
            # smoke tests this avoids an expensive full FSDP state extraction
            # before any optimizer step has changed the actor weights.
            print("[DiffusionWorker] Skipping initial rollout weight sync", flush=True)
            self._initial_weight_sync_skipped = True
            return True

        print("[DiffusionWorker] Extracting actor state for rollout weight sync", flush=True)
        self._cached_actor_state_dict = self._extract_actor_state_dict()

        # If engine is alive (e.g., first call before any cleanup), apply now
        if (
                hasattr(self, "diffusion_rollout")
                and self.diffusion_rollout is not None
                and self.diffusion_rollout.llm is not None
        ):
            self._apply_cached_weights_to_engine()

        return True


class DiffusionAgentLoopManager:
    """Minimal AgentLoopManager replacement for SDAR diffusion rollout.

    The standard AgentLoopManager initialises vLLM/sglang LLM servers and
    uses dedicated AgentLoopWorker actors for generation.  The diffusion
    worker handles rollout entirely inside
    DiffusionActorRolloutRefWorker.generate_sequences (via jetengine_ext),
    so we bypass all of that and route generate_sequences straight to the
    worker group.

    Plug in via config:
        actor_rollout_ref.rollout.agent.agent_loop_manager_class:
            verl.workers.diffusion_workers.DiffusionAgentLoopManager
    """

    def __init__(
            self,
            config,
            worker_group,
            rollout_resource_pool=None,
            reward_loop_worker_handles=None,
    ):
        self.config = config
        self.worker_group = worker_group
        self.reward_loop_worker_handles = reward_loop_worker_handles
        # RayPPOTrainer.init_workers accesses async_rollout_manager.rollout_replicas
        # to pass to CheckpointEngineManager; an empty list is correct here because
        # the diffusion rollout has no standalone LLM server replicas.
        self.rollout_replicas = []

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Delegate directly to the diffusion worker group, then compute rewards."""
        n = int(self.config.actor_rollout_ref.rollout.get("n", 1))
        rollout_prompts = prompts.repeat(repeat_times=n, interleave=True) if n > 1 else prompts

        output = self.worker_group.generate_sequences(rollout_prompts)
        metrics = output.meta_info["metrics"]
        batch = self._build_tensor_batch(output, rollout_prompts)
        non_tensor_batch = self._build_non_tensor_batch(output, rollout_prompts)
        output = DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
        output.batch["rm_scores"], acc = self._compute_rewards(output)
        output.non_tensor_batch["acc"] = acc
        output.meta_info["reward_extra_keys"] = ["acc"]
        timing = self._performance_metrics(metrics, output)
        rollout_metrics = self._rollout_metrics(metrics, output)
        output.meta_info = {"timing": timing, "rollout_metrics": rollout_metrics}
        return output

    def _performance_metrics(self, metrics, output: DataProto) -> dict[str, float]:
        timing: dict[str, float] = {}

        # New structure: metrics is a dict[str, list], e.g.
        # {"generate_sequences": [...], "tool_calls": [...], "num_preempted": [...]}
        # Old structure: metrics is list[list[dict]]
        if isinstance(metrics, dict):
            t_generate_sequences = np.asarray(metrics.get("generate_sequences", []), dtype=float)
            t_tool_calls = np.asarray(metrics.get("tool_calls", []), dtype=float)
            num_preempted = np.asarray(metrics.get("num_preempted", []), dtype=float)
        else:
            t_generate_sequences = np.array([m["generate_sequences"] for chunk in metrics for m in chunk], dtype=float)
            t_tool_calls = np.array([m["tool_calls"] for chunk in metrics for m in chunk], dtype=float)
            num_preempted = np.array([m["num_preempted"] for chunk in metrics for m in chunk], dtype=float)

        # empty-guard
        if t_generate_sequences.size == 0 or t_tool_calls.size == 0 or num_preempted.size == 0:
            return timing

        timing["agent_loop/num_preempted/min"] = num_preempted.min()
        timing["agent_loop/num_preempted/max"] = num_preempted.max()
        timing["agent_loop/num_preempted/mean"] = num_preempted.mean()

        timing["agent_loop/generate_sequences/min"] = t_generate_sequences.min()
        timing["agent_loop/generate_sequences/max"] = t_generate_sequences.max()
        timing["agent_loop/generate_sequences/mean"] = t_generate_sequences.mean()

        timing["agent_loop/tool_calls/min"] = t_tool_calls.min()
        timing["agent_loop/tool_calls/max"] = t_tool_calls.max()
        timing["agent_loop/tool_calls/mean"] = t_tool_calls.mean()

        # batch sequence generation is bounded by the slowest sample
        slowest = int(np.argmax(t_generate_sequences + t_tool_calls))
        attention_mask = output.batch["attention_mask"][slowest]
        prompt_length = output.batch["prompts"].shape[1]

        timing["agent_loop/slowest/generate_sequences"] = t_generate_sequences[slowest]
        timing["agent_loop/slowest/tool_calls"] = t_tool_calls[slowest]
        timing["agent_loop/slowest/prompt_length"] = attention_mask[:prompt_length].sum().item()
        timing["agent_loop/slowest/response_length"] = attention_mask[prompt_length:].sum().item()
        timing["agent_loop/slowest/num_preempted"] = num_preempted[slowest]

        return timing

    def _rollout_metrics(self, metrics, output: DataProto | None = None) -> dict[str, float]:
        if not isinstance(metrics, dict):
            return {}

        rollout_metrics: dict[str, float] = {}
        for key, values in metrics.items():
            if not key.startswith("rollout/"):
                continue
            arr = np.asarray(values, dtype=float)
            if arr.size == 0:
                continue
            rollout_metrics[key] = float(arr.mean())

        if output is None or output.batch is None:
            return rollout_metrics

        response_mask = output.batch.get("response_mask")
        rollout_step_events = output.non_tensor_batch.get("rollout_step_events")
        if response_mask is None or rollout_step_events is None:
            return rollout_metrics

        valid_response_tokens_total = int(response_mask.bool().sum().item())
        seen_event_streams: set[int] = set()
        denoise_forwards_total = 0
        accepted_tokens_total = 0
        forward_tokens_total = 0
        accepting_denoise_forwards_total = 0
        for events in rollout_step_events:
            # DiffusionRolloutWorker stores the same per-worker event stream on
            # each row it generated. Count shared streams once while preserving
            # distinct workers that happen to have identical event contents.
            stream_id = id(events)
            if stream_id in seen_event_streams:
                continue
            seen_event_streams.add(stream_id)
            if events is None:
                continue
            for event in events:
                if not isinstance(event, dict) or event.get("kind") != "jetengine_denoise":
                    continue
                denoise_forwards_total += 1
                accepted_tokens = int(event.get("accepted_tokens", 0))
                accepted_tokens_total += accepted_tokens
                forward_tokens_total += int(event.get("tokens", 0))
                if accepted_tokens > 0:
                    accepting_denoise_forwards_total += 1

        rollout_metrics["rollout/tpf/derived/valid_response_tokens_total"] = float(valid_response_tokens_total)
        rollout_metrics["rollout/tpf/derived/denoise_forwards_total"] = float(denoise_forwards_total)
        rollout_metrics["rollout/tpf/derived/accepted_tokens_total"] = float(accepted_tokens_total)
        rollout_metrics["rollout/tpf/derived/forward_tokens_total"] = float(forward_tokens_total)
        if denoise_forwards_total > 0:
            rollout_metrics["rollout/tpf/derived/event_weighted_accepted_per_denoise_forward"] = (
                float(accepted_tokens_total) / float(denoise_forwards_total)
            )
            rollout_metrics["rollout/tpf/derived/valid_response_tokens_per_denoise_forward"] = (
                float(valid_response_tokens_total) / float(denoise_forwards_total)
            )
        if accepting_denoise_forwards_total > 0:
            rollout_metrics["rollout/tpf/derived/accepted_per_accepting_denoise_forward"] = (
                float(accepted_tokens_total) / float(accepting_denoise_forwards_total)
            )
        return rollout_metrics

    def _build_non_tensor_batch(self, output: DataProto, prompts: DataProto):
        non_tensor_batch = {}
        passthrough_keys = ["raw_prompt", "data_source", "reward_model"]
        for key in passthrough_keys:
            if key in prompts.non_tensor_batch:
                non_tensor_batch[key] = prompts.non_tensor_batch[key]

        n = len(output)
        non_tensor_batch["multi_modal_inputs"] = np.array([{} for _ in range(n)], dtype=object)
        non_tensor_batch["prompt_ids"] = output.non_tensor_batch["prompt_ids"]
        non_tensor_batch["response_ids"] = output.non_tensor_batch["response_ids"]
        non_tensor_batch["step_maps"] = output.non_tensor_batch["step_maps"]
        non_tensor_batch["response_uid"] = output.non_tensor_batch["response_uid"]
        if "rollout_step_events" in output.non_tensor_batch:
            non_tensor_batch["rollout_step_events"] = output.non_tensor_batch["rollout_step_events"]
        if "rollout_self_conditioning" in output.non_tensor_batch:
            non_tensor_batch["rollout_self_conditioning"] = output.non_tensor_batch["rollout_self_conditioning"]
        if "rollout_replay_trace" in output.non_tensor_batch:
            non_tensor_batch["rollout_replay_trace"] = output.non_tensor_batch["rollout_replay_trace"]
        return non_tensor_batch

    def _build_tensor_batch(self, output: DataProto, prompts: DataProto) -> TensorDict:
        block_length = self.config.actor_rollout_ref.rollout.engine_kwargs.get("block_length", 4)
        prompt_length = int(self.config.data.max_prompt_length)
        response_length = int(self.config.actor_rollout_ref.rollout.response_length) + block_length

        # Prefer pad_token_id from prompts.meta_info (trainer usually sets it); fallback to 0.
        pad_token_id = int(prompts.meta_info.get("pad_token_id", 0))

        prompt_ids_arr = output.non_tensor_batch.get("prompt_ids", None)
        response_ids_arr = output.non_tensor_batch.get("response_ids", None)
        step_maps_arr = output.non_tensor_batch.get("step_maps", None)
        token_logprobs_arr = output.non_tensor_batch.get("token_logprobs", None)

        if prompt_ids_arr is None or response_ids_arr is None:
            raise KeyError(
                "Diffusion rollout output is missing required keys in non_tensor_batch: "
                f"{list(output.non_tensor_batch.keys())}"
            )

        # Avoid truncation
        prompt_length = max(prompt_length, max(len(p_i) for p_i in prompt_ids_arr))
        response_length = max(response_length, max(len(r_i) for r_i in response_ids_arr))

        assert prompt_length % block_length == 0
        assert response_length % block_length == 0

        batch_size = len(output)
        prompts_ids = torch.empty((batch_size, prompt_length), dtype=torch.long)
        responses_ids = torch.empty((batch_size, response_length), dtype=torch.long)
        prompt_attn = torch.zeros((batch_size, prompt_length), dtype=torch.bool)
        resp_attn = torch.zeros((batch_size, response_length), dtype=torch.bool)
        position_ids = torch.zeros((batch_size, prompt_length + response_length), dtype=torch.long)

        step_maps = torch.zeros((batch_size, response_length), dtype=torch.long)
        token_logprobs = torch.zeros((batch_size, response_length), dtype=torch.float)

        for i in range(batch_size):
            p_ids = list(prompt_ids_arr[i]) if prompt_ids_arr is not None else []
            r_ids = list(response_ids_arr[i]) if response_ids_arr is not None else []
            s_maps = list(step_maps_arr[i]) if step_maps_arr is not None else []
            t_logprobs = list(token_logprobs_arr[i]) if token_logprobs_arr is not None else []

            p_pad, p_mask, p_valid = self._left_pad_1d(p_ids, prompt_length, pad_token_id)
            r_pad, r_mask, r_valid = self._right_pad_1d(r_ids, response_length, pad_token_id)

            # step_maps uses right padding to response_length
            # pad_id=0 by default (padding positions will be 0)
            s_pad, _s_mask, _s_valid = self._right_pad_1d(s_maps, response_length, pad_id=max(s_maps) + 1)
            t_logprobs_pad, _t_logprobs_mask, _t_logprobs_valid = self._right_pad_1d(t_logprobs, response_length, pad_id=-float("inf"), dtype=torch.float)

            prompts_ids[i] = p_pad
            responses_ids[i] = r_pad
            prompt_attn[i] = p_mask
            resp_attn[i] = r_mask
            step_maps[i] = s_pad
            token_logprobs[i] = t_logprobs_pad

            # position_ids: padding positions stay 0
            # prompt valid positions: 0..p_valid-1 (aligned to the right due to left padding)
            if p_valid > 0:
                start = prompt_length - p_valid
                position_ids[i, start:prompt_length] = torch.arange(p_valid, dtype=torch.long)

            # response valid positions: continue counting from p_valid
            if r_valid > 0:
                position_ids[i, prompt_length: prompt_length + r_valid] = torch.arange(
                    p_valid, p_valid + r_valid, dtype=torch.long
                )

        attention_mask = torch.cat([prompt_attn, resp_attn], dim=1)
        input_ids = torch.cat([prompts_ids, responses_ids], dim=1)

        # response_mask: which response tokens are valid (same shape as "responses")
        response_mask = resp_attn

        batch = TensorDict(
            {
                "prompts": prompts_ids,
                "responses": responses_ids,
                "response_mask": response_mask,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "step_maps": step_maps,
                "log_probs": token_logprobs,
                "rollout_log_probs": token_logprobs,
            },
            batch_size=batch_size,
        )
        return batch

    @staticmethod
    def _left_pad_1d(seq: list[int], length: int, pad_id: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Returns (padded_ids[length], attn_mask[length], valid_len)."""
        if seq is None:
            seq = []
        # truncate from the left (keep the most recent tokens)
        seq = seq[-length:] if len(seq) > length else seq
        valid_len = len(seq)
        pad_len = length - valid_len
        ids = torch.full((length,), pad_id, dtype=torch.long)
        mask = torch.zeros((length,), dtype=torch.long)
        if valid_len > 0:
            ids[pad_len:] = torch.tensor(seq, dtype=torch.long)
            mask[pad_len:] = 1
        return ids, mask, valid_len

    @staticmethod
    def _right_pad_1d(seq: list[int], length: int, pad_id: int, dtype=torch.long) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Returns (padded_ids[length], attn_mask[length], valid_len)."""
        if seq is None:
            seq = []
        # truncate from the right (keep the earliest generated tokens)
        seq = seq[:length] if len(seq) > length else seq
        valid_len = len(seq)
        ids = torch.full((length,), pad_id, dtype=dtype)
        mask = torch.zeros((length,), dtype=torch.long)
        if valid_len > 0:
            ids[:valid_len] = torch.tensor(seq, dtype=dtype)
            mask[:valid_len] = 1
        return ids, mask, valid_len

    def _compute_rewards(self, output: DataProto) -> DataProto:
        batch_size = len(output)
        reward_keys = ["data_source", "reward_model", "extra_info"]

        # Call reward worker per sequence (synchronous via ray.get)
        scores = []
        for i in range(batch_size):
            item_batch = TensorDict(
                {
                    "prompts": output.batch["prompts"][i: i + 1],
                    "responses": output.batch["responses"][i: i + 1],
                    "attention_mask": output.batch["attention_mask"][i: i + 1],
                    "input_ids": output.batch["input_ids"][i: i + 1],
                    "position_ids": output.batch["position_ids"][i: i + 1],
                },
                batch_size=1,
            )
            item_non_tensor = {
                k: np.array([output.non_tensor_batch[k][i]])
                for k in reward_keys
                if k in output.non_tensor_batch
            }
            item_data = DataProto(batch=item_batch, non_tensor_batch=item_non_tensor)

            handle = random.choice(self.reward_loop_worker_handles)
            result = ray.get(handle.compute_score.remote(item_data))
            scores.append(float(result.get("reward_score", 0.0)))

        # Place score at the last valid response token position
        prompt_length = output.batch["prompts"].size(1)
        response_attn = output.batch["attention_mask"][:, prompt_length:]
        response_length = (response_attn.sum(dim=1).long() - 1).clamp(min=0)
        rm_scores = torch.zeros_like(output.batch["responses"], dtype=torch.float32)
        rm_scores[torch.arange(batch_size), response_length] = torch.tensor(scores, dtype=torch.float32)

        acc = numpy.array(scores) > 0
        acc = acc.astype(float)

        return rm_scores, acc

    def start_profile(self):
        self.worker_group.start_profile()

    def stop_profile(self):
        self.worker_group.stop_profile()
