# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The main entry point to run the PPO algorithm
"""

import datetime
import json
import logging
import os
import time
import uuid
import warnings
from dataclasses import asdict
from multiprocessing.shared_memory import SharedMemory
from typing import Optional

import numpy as np
import psutil
import torch
import torch.distributed
import torch.distributed as dist
from codetiming import Timer
from omegaconf import DictConfig, OmegaConf, open_dict
from omegaconf.errors import ConfigAttributeError
from peft import LoraConfig, TaskType, get_peft_model
from safetensors.torch import save_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType

try:
    # for torch 2.5+
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineRegistry
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.activation_offload import enable_activation_offloading
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import (
    get_device_id,
    get_device_name,
    get_nccl_backend,
    get_torch_device,
    set_expandable_segments,
)
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    collect_lora_params,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    get_shard_placement_fn,
    init_fn,
    layered_summon_lora_params,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
    replace_lora_wrapper,
)
from verl.utils.import_utils import import_external_libs
from verl.utils.lifecycle import emit_lifecycle, lifecycle_memory
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.shared_weight import (
    FSDP1QwenParamStream,
    SharedWeightMetrics,
    StreamingExportUnavailable,
    _InstrumentedParamStream,
    assert_stream_offload_safe,
    shared_weights_enabled,
    stream_export_enabled,
    stream_fsdp1_params_layered,
    update_weights_with_failure_invalidation,
)
from verl.utils.model import convert_weight_keys, deduplicate_diffusion_gemma_tied_weights
from verl.utils.profiler import DistProfiler, DistProfilerExtension, ProfilerConfig, log_gpu_memory_usage, simple_timer
from verl.utils.profiler.performance import reduce_timing, topk_reduce_ratio_min_max
from verl.utils.py_functional import append_to_dict, convert_to_regular_types

# QAT support
from verl.utils.qat import apply_qat, enable_qat_fuse
from verl.utils.ray_utils import get_event_loop
from verl.workers.config import FSDPCriticConfig, FSDPEngineConfig, HFModelConfig, RolloutConfig
from verl.workers.config.optimizer import build_optimizer
from verl.workers.rollout import get_rollout_class
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager
from verl.workers.flex_attn_utils import WrappedFlexAttention

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()
_ACTOR_WORKER_CONSTRUCTION_COUNT = 0


def create_device_mesh(world_size, fsdp_size):
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh(device_name, mesh_shape=(world_size,), mesh_dim_names=["fsdp"])
    else:
        device_mesh = init_device_mesh(
            device_name, mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=["ddp", "fsdp"]
        )
    return device_mesh


def get_sharding_strategy(device_mesh, zero3_enable=True):
    from torch.distributed.fsdp import ShardingStrategy

    if zero3_enable:
        fsdp_strategy = ShardingStrategy.FULL_SHARD
        hsdp_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        fsdp_strategy = ShardingStrategy.SHARD_GRAD_OP
        hsdp_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2

    if device_mesh.ndim == 1:
        sharding_strategy = fsdp_strategy
    elif device_mesh.ndim == 2:
        sharding_strategy = hsdp_strategy
    else:
        raise NotImplementedError(f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2")
    return sharding_strategy


def get_vl_model_vision_tower(vl_model_instance):
    """
    Util to extract Vision Tower from a VL model instance
    """
    if hasattr(vl_model_instance, "model") and hasattr(vl_model_instance.model, "visual"):
        # transformers >= 4.52.0
        return vl_model_instance.model.visual
    elif hasattr(vl_model_instance, "visual"):
        # transformers < 4.52.0
        return vl_model_instance.visual
    return None


class ActorRolloutRefWorker(Worker, DistProfilerExtension):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str, **kwargs):
        Worker.__init__(self)

        self.config = config
        import torch.distributed

        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(
                backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}",
                rank=rank,
                world_size=world_size,
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )

        # Apply NPU patches for FSDP backend
        from verl.workers.engine.fsdp.utils import apply_npu_fsdp_patches

        apply_npu_fsdp_patches()

        # build device mesh for FSDP
        world_size = torch.distributed.get_world_size()
        # TODO(sgm): support FSDP hybrid shard for larger model
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=self.config.actor.fsdp_config.fsdp_size)
        self.actor_fsdp_local_shard_group_only = False
        self.actor_fsdp_local_process_group = None
        self.actor_replica_grad_process_group = None
        self.actor_replica_grad_group_size = 1
        actor_fsdp_config = self.config.actor.fsdp_config
        actor_fsdp_size = int(actor_fsdp_config.get("fsdp_size", -1) or -1)
        if (
            role == "actor"
            and bool(actor_fsdp_config.get("local_shard_group_only", False))
            and actor_fsdp_size > 0
            and actor_fsdp_size < world_size
        ):
            if world_size % actor_fsdp_size != 0:
                raise ValueError(
                    "actor.fsdp_config.local_shard_group_only requires world_size divisible by fsdp_size, "
                    f"got world_size={world_size} fsdp_size={actor_fsdp_size}"
                )
            self.actor_fsdp_local_shard_group_only = True
            rank = torch.distributed.get_rank()
            local_node_ip = str(self._get_node_ip()).strip("[]")
            node_ips = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(node_ips, local_node_ip)
            node_ip_order = []
            for node_ip in node_ips:
                if node_ip not in node_ip_order:
                    node_ip_order.append(node_ip)
            node_rank_groups = [[idx for idx, node_ip in enumerate(node_ips) if node_ip == group_ip] for group_ip in node_ip_order]
            for ranks in node_rank_groups:
                if len(ranks) != actor_fsdp_size:
                    raise ValueError(
                        "actor.fsdp_config.local_shard_group_only requires each actor node to have fsdp_size ranks, "
                        f"got node_groups={node_rank_groups} fsdp_size={actor_fsdp_size}"
                    )
            local_group_index = next(idx for idx, ranks in enumerate(node_rank_groups) if rank in ranks)
            local_shard_idx = node_rank_groups[local_group_index].index(rank)
            replica_count = len(node_rank_groups)
            for ranks in node_rank_groups:
                group = torch.distributed.new_group(ranks=ranks, backend=get_nccl_backend())
                if rank in ranks:
                    self.actor_fsdp_local_process_group = group
            for shard_idx in range(actor_fsdp_size):
                ranks = [node_ranks[shard_idx] for node_ranks in node_rank_groups]
                group = torch.distributed.new_group(ranks=ranks, backend=get_nccl_backend())
                if local_shard_idx == shard_idx:
                    self.actor_replica_grad_process_group = group
                    self.actor_replica_grad_group_size = len(ranks)
            if self.actor_fsdp_local_process_group is None or self.actor_replica_grad_process_group is None:
                raise RuntimeError(
                    "failed to initialize local FSDP shard/replica groups for "
                    f"rank={rank} world_size={world_size} fsdp_size={actor_fsdp_size}"
                )
            print(
                "[SplitStreaming] actor_local_fsdp_groups "
                f"rank={rank} fsdp_size={actor_fsdp_size} replica_count={replica_count} "
                f"node_ip={local_node_ip} local_shard_idx={local_shard_idx} "
                f"node_rank_groups={node_rank_groups}",
                flush=True,
            )

        # build device mesh for Ulysses Sequence Parallel
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.actor.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
            )

        # create training dispatch
        if self.ulysses_device_mesh is not None:
            is_collect = self.ulysses_device_mesh["sp"].get_local_rank() == 0
            self._register_dispatch_collect_info(
                "actor", dp_rank=self.ulysses_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )
        else:
            self._register_dispatch_collect_info("actor", dp_rank=self.rank, is_collect=True)

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        self._lora_rank = self.config.model.get("lora_rank", 0)
        self._is_lora = self.config.model.get("lora_adapter_path") is not None or self._lora_rank > 0

        self.role = role
        assert self.role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]

        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]
        global _ACTOR_WORKER_CONSTRUCTION_COUNT
        _ACTOR_WORKER_CONSTRUCTION_COUNT += 1
        self._lifecycle_actor_uuid = uuid.uuid4().hex
        self._lifecycle_construction_count = _ACTOR_WORKER_CONSTRUCTION_COUNT
        self._lifecycle_model_init_count = 0
        self._lifecycle_checkpoint_load_count = 0
        self._lifecycle_hot_update_count = 0
        try:
            import ray

            self._lifecycle_ray_actor_id = str(ray.get_runtime_context().get_actor_id())
        except Exception:
            self._lifecycle_ray_actor_id = "unavailable"
        emit_lifecycle(
            "ActorRolloutRefWorker",
            "constructed",
            actor_uuid=self._lifecycle_actor_uuid,
            ray_actor_id=self._lifecycle_ray_actor_id,
            rank=int(self.rank),
            role=self.role,
            construction_count=self._lifecycle_construction_count,
            model_init_count=self._lifecycle_model_init_count,
            checkpoint_load_count=self._lifecycle_checkpoint_load_count,
            memory=lifecycle_memory("actor_worker_constructed"),
        )
        self.use_orig_params = self.config.actor.fsdp_config.get("use_orig_params", False)

        # TODO(haibin.lin):
        # As of now the type of config is DictConfig, if we assign config.profiler with ProfilerConfig,
        # it will actually convert the ProfilerConfig dataclass back to a DictConfig.
        # We can still use ProfilerConfig for testing purpose (tests/utils/test_nvtx_profile.py)
        # as they provides DictConfig-like interface
        # The benefit of creating the dataclass config is to perform validation during __post_init__
        if self._is_actor:
            omega_profiler_config = config.actor.get("profiler", {})
        elif self._is_rollout:
            # NOTE: In colocation mode, rollout config may not take effect (follow the actor config)
            # This is for extendability in AsyncRL cases
            omega_profiler_config = config.rollout.get("profiler", {})
        elif self._is_ref:
            omega_profiler_config = config.ref.get("profiler", {})
        else:
            raise ValueError(
                f"Invalid role {self.role}, should be one of "
                "['actor', 'rollout', 'ref', 'actor_rollout', 'actor_rollout_ref']"
            )
        # omega_profiler_config is DictConfig
        # profiler_config is a ProfilerConfig dataclass
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )

        self._is_offload_param = False
        self._is_offload_optimizer = False
        if self._is_actor:
            self._is_offload_param = self.config.actor.fsdp_config.get("param_offload", False)
            self._is_offload_optimizer = self.config.actor.fsdp_config.get("optimizer_offload", False)
        elif self._is_ref:
            # TODO: it seems that manual offload is slowly than FSDP offload
            self._is_offload_param = self.config.ref.fsdp_config.get("param_offload", False)

        # normalize config
        if self._is_actor:
            self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
            self.config.actor.ppo_mini_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
            assert self.config.actor.ppo_mini_batch_size > 0, (
                f"ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be larger than 0 after "
                f"normalization"
            )
            # micro bsz
            if self.config.actor.ppo_micro_batch_size is not None:
                self.config.actor.ppo_micro_batch_size //= (
                    self.device_mesh.size() // self.ulysses_sequence_parallel_size
                )
                self.config.actor.ppo_micro_batch_size_per_gpu = self.config.actor.ppo_micro_batch_size

            if self.config.actor.ppo_micro_batch_size_per_gpu is not None:
                assert self.config.actor.ppo_mini_batch_size % self.config.actor.ppo_micro_batch_size_per_gpu == 0, (
                    f"normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be divisible by "
                    f"ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}"
                )
                assert self.config.actor.ppo_mini_batch_size // self.config.actor.ppo_micro_batch_size_per_gpu > 0, (
                    f"normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be larger than "
                    f"ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}"
                )

        # normalize rollout config
        if self._is_rollout and self.config.rollout.log_prob_micro_batch_size is not None:
            self.config.rollout.log_prob_micro_batch_size //= (
                self.device_mesh.size() // self.ulysses_sequence_parallel_size
            )
            self.config.rollout.log_prob_micro_batch_size_per_gpu = self.config.rollout.log_prob_micro_batch_size
        # normalize ref config
        if self._is_ref and self.config.ref.log_prob_micro_batch_size is not None:
            self.config.ref.log_prob_micro_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
            self.config.ref.log_prob_micro_batch_size_per_gpu = self.config.ref.log_prob_micro_batch_size

    def _init_qat_config(self):
        """Initialize QAT configuration from actor.qat."""
        try:
            self.qat_config = self.config.actor.qat
            self._qat_enabled = self.qat_config.enable
            if self._qat_enabled:
                logger.info(
                    f"QAT enabled: mode={self.qat_config.mode}, config_path={self.qat_config.quantization_config_path}"
                )
        except (AttributeError, KeyError, ConfigAttributeError):
            # QAT config not provided, disable QAT
            self._qat_enabled = False
            self.qat_config = None

    def _restore_w4a4_input_scales(self, model, model_path):
        """Restore input_global_scale and input_amax from checkpoint for W4A4 mode."""
        import glob

        from safetensors import safe_open

        safetensor_files = glob.glob(f"{model_path}/model*.safetensors")
        loaded_count = 0

        for sf_path in safetensor_files:
            with safe_open(sf_path, framework="pt") as f:
                for key in f.keys():
                    if "input_global_scale" in key:
                        module_path = key.replace(".input_global_scale", "")
                        amax_key = f"{module_path}.input_amax"

                        module = model
                        for part in module_path.split("."):
                            module = getattr(module, part)

                        scale_val = f.get_tensor(key)
                        val = scale_val.item() if scale_val.numel() == 1 else scale_val.max().item()
                        module.input_global_scale.fill_(val)

                        amax_val = f.get_tensor(amax_key)
                        amax = amax_val.item() if amax_val.numel() == 1 else amax_val.max().item()
                        module.input_amax.fill_(amax)
                        loaded_count += 1

        if self.rank == 0:
            logger.info(f"[W4A4] Loaded {loaded_count} input scales from checkpoint")

    def _build_model_optimizer(
        self,
        model_path,
        fsdp_config: FSDPEngineConfig,
        optim_config,
        override_model_config,
        use_remove_padding=False,
        use_fused_kernels=False,
        enable_gradient_checkpointing=False,
        trust_remote_code=False,
        use_liger=False,
        role="actor",
        enable_activation_offload=False,
        use_prefix_grouper=False,
        use_tiled_mlp=False,
        tiled_mlp_shards=4,
    ):
        from torch.distributed.fsdp import CPUOffload, MixedPrecision
        from transformers import (
            AutoConfig,
            AutoModel,
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
        )
        try:
            from transformers import AutoModelForVision2Seq
        except ImportError:
            AutoModelForVision2Seq = AutoModelForImageTextToText

        from verl.utils.model import get_generation_config, print_model_size, update_model_config
        from verl.utils.torch_dtypes import PrecisionType
        from verl.utils.transformers_compat import ensure_transformers_runtime_compat, repair_sdar_rotary_embedding

        assert role in ["actor", "ref"]
        ensure_transformers_runtime_compat()

        # TiledMLP requires FSDP2 for correct gradient computation
        if use_tiled_mlp and self.config.actor.strategy == "fsdp":
            raise ValueError("TiledMLP requires FSDP2. Set `actor_rollout_ref.actor.strategy=fsdp2`.")

        log_gpu_memory_usage(f"Before init {role} from HF AutoModel", logger=logger)
        local_path = model_path

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code)

        if self.config.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.model.custom_chat_template
            else:
                self.tokenizer.chat_template = self.config.model.custom_chat_template

        torch_dtype = fsdp_config.get("model_dtype", None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # override model kwargs
        attn_implementation = override_model_config.get("attn_implementation", "flash_attention_2")
        actor_model_config = AutoConfig.from_pretrained(
            local_path, trust_remote_code=trust_remote_code, attn_implementation=attn_implementation
        )
        # TODO: VL models use VisionAttention, which directly uses flash_attention in transformers>=4.53
        # which will be patched by _ulysses_flash_attention_forward, but errorly misses position_ids
        # Maybe support Ulysses in VisionAttention in the future and remove this patch
        if self.ulysses_sequence_parallel_size > 1 and hasattr(actor_model_config, "vision_config"):
            actor_model_config.vision_config._attn_implementation = "eager"

        # patch for qwen2.5-vl: when using flash_attention_3, set vision tower to use flash_attention_2
        # because the vision tower does not support flash_attention_3
        if (
            getattr(actor_model_config, "model_type", None) == "qwen2_5_vl"
            and attn_implementation == "flash_attention_3"
            and hasattr(actor_model_config, "vision_config")
        ):
            actor_model_config.vision_config._attn_implementation = "flash_attention_2"

        # patch for kimi-vl
        if getattr(actor_model_config, "model_type", None) == "kimi_vl":
            actor_model_config.text_config.topk_method = "greedy"

        self.generation_config = get_generation_config(local_path, trust_remote_code=trust_remote_code)

        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        if self.config.model.get("mtp", {}).get("enable", False):
            raise NotImplementedError("Right now,  MTP is not supported in FSDP")
        else:
            if hasattr(actor_model_config, "num_nextn_predict_layers"):
                actor_model_config.num_nextn_predict_layers = 0

        # Worker-side switches carried in override_config must not leak into the HF
        # model config (they would be written into every saved checkpoint's config.json).
        override_model_config = {
            k: v
            for k, v in dict(override_model_config).items()
            if not (str(k).startswith("env__") or str(k).startswith("stack_sampler") or str(k) == "rollout_mode_light_empty_cache")
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
        if self.rank == 0:
            print(f"Model config after override: {actor_model_config}")

        # NOTE(fix me): tie_word_embedding causes meta_tensor init to hang
        init_context = get_init_weight_context_manager(
            use_meta_tensor=(
                not actor_model_config.tie_word_embeddings
                and not bool(role == "actor" and getattr(self, "actor_fsdp_local_shard_group_only", False))
            ),
            mesh=self.device_mesh,
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            has_remote_code = hasattr(actor_model_config, "auto_map") and any(
                actor_model_config.architectures[0] in val for val in actor_model_config.auto_map.values()
            )
            if has_remote_code:
                auto_class = next(
                    k for k, v in actor_model_config.auto_map.items() if actor_model_config.architectures[0] in v
                )
                match auto_class:
                    case "AutoModelForVision2Seq":
                        actor_module_class = AutoModelForVision2Seq
                    case "AutoModelForCausalLM":
                        actor_module_class = AutoModelForCausalLM
                    case "AutoModelForImageTextToText":
                        actor_module_class = AutoModelForImageTextToText
                    case _:
                        actor_module_class = AutoModel
            else:
                if type(actor_model_config) in AutoModelForVision2Seq._model_mapping.keys():
                    actor_module_class = AutoModelForVision2Seq
                elif type(actor_model_config) in AutoModelForCausalLM._model_mapping.keys():
                    actor_module_class = AutoModelForCausalLM
                elif type(actor_model_config) in AutoModelForImageTextToText._model_mapping.keys():
                    actor_module_class = AutoModelForImageTextToText
                else:
                    actor_module_class = AutoModel

            actor_module = actor_module_class.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch_dtype,
                config=actor_model_config,
                trust_remote_code=trust_remote_code,
                attn_implementation=attn_implementation,
            )
            repair_sdar_rotary_embedding(actor_module)
            _repair_self_conditioning_init(actor_module)

            # Apply Liger kernel to the model if use_liger is set to True
            if use_liger:
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(model=actor_module)

            fused_kernel_options = self.config.model.get("fused_kernel_options", None)
            fused_kernels_backend = (
                fused_kernel_options.get("impl_backend", None) if fused_kernel_options is not None else None
            )

            apply_monkey_patch(
                model=actor_module,
                use_remove_padding=use_remove_padding,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
                use_fused_kernels=use_fused_kernels,
                fused_kernels_backend=fused_kernels_backend,
                use_prefix_grouper=use_prefix_grouper,
                use_tiled_mlp=use_tiled_mlp,
                tiled_mlp_shards=tiled_mlp_shards,
            )

            from transformers.integrations import flex_attention
            flex_attention.WrappedFlexAttention = WrappedFlexAttention

            # some parameters may not in torch_dtype. TODO(zhangchi.usc1992) remove this after we switch to fsdp2
            actor_module.to(torch_dtype)

            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        if self._is_lora:
            print("Applying LoRA to actor module")
            actor_module.enable_input_require_grads()

            lora_adapter_path = self.config.model.get("lora_adapter_path")
            if lora_adapter_path is not None:
                from peft import PeftModel

                print(f"Loading pre-trained LoRA adapter to {role} from: {lora_adapter_path}")

                # Copy adapter to local if needed
                local_adapter_path = copy_to_local(lora_adapter_path, use_shm=self.config.model.get("use_shm", False))

                actor_module = PeftModel.from_pretrained(actor_module, local_adapter_path, is_trainable=True)
                peft_config = actor_module.peft_config["default"]
                # Ensure task_type is TaskType enum, not string
                if isinstance(peft_config.task_type, str):
                    peft_config.task_type = TaskType.CAUSAL_LM

            else:
                # Convert config to regular Python types before creating PEFT model
                lora_config = {
                    "task_type": TaskType.CAUSAL_LM,
                    "r": self.config.model.lora_rank,
                    "lora_alpha": self.config.model.lora_alpha,
                    "target_modules": convert_to_regular_types(self.config.model.target_modules),
                    "exclude_modules": convert_to_regular_types(self.config.model.exclude_modules),
                    "bias": "none",
                }
                actor_module = get_peft_model(actor_module, LoraConfig(**lora_config))

        self.use_orig_params = fsdp_config.get("use_orig_params", False)
        # SC_LR_MULT builds per-parameter optimizer groups by matching the
        # substring "self_conditioning" in actor_module_fsdp.named_parameters().
        # FSDP1 with use_orig_params=False returns FLAT parameters -- no module
        # names survive -- so the grouping would find nothing. Force the
        # original names unless the adapter is its own FSDP unit, whose flat
        # parameter keeps the name: the LLaDA2 recipe sets SC_LR_MULT=0.3 and
        # wraps LLaDA2SelfConditioning separately. Same reason the
        # freeze_vision_tower branch below forces it: any per-parameter
        # distinction needs the original names. No-op when SC_LR_MULT is 1.0.
        if float(os.environ.get("SC_LR_MULT", "1.0")) != 1.0 and not self.use_orig_params \
                and not _sc_adapter_is_separately_wrapped(fsdp_config):
            self.use_orig_params = True
            if self.rank == 0:
                print(
                    "[SC_LR_MULT] forcing fsdp use_orig_params=True; flat parameters "
                    "carry no module names and the LR grouping would find nothing",
                    flush=True,
                )
        if self.config.actor.get("freeze_vision_tower", False):
            vision_tower = get_vl_model_vision_tower(actor_module)
            if vision_tower is not None:
                vision_tower.requires_grad_(False)
                self.use_orig_params = True
                if self.rank == 0:
                    print("[actor model] Vision tower is set to not trainable.")
            else:
                if self.rank == 0:
                    print("[actor model] No vision tower found.")

        # NOTE: VERL_SC_TRAIN_MODE freezing is
        # implemented in dp_actor (_maybe_sc_train_mode_snapshot/_restore,
        # flat-shard snapshot-restore) — NOT here. A requires_grad_(False) +
        # use_orig_params=True surgery at this site was smoke-tested and
        # reverted: use_orig_params reshards weights into 1-D views between
        # forwards and breaks the fast-hidden-CE lm_head path in dp_actor.

        # Apply QAT before FSDP wrapping (actor only)
        if role == "actor" and self._qat_enabled:
            actor_module = apply_qat(actor_module, self.qat_config)
            enable_qat_fuse(actor_module)
            if self.qat_config.mode == "w4a4":
                self._restore_w4a4_input_scales(actor_module, self.config.model.path)

        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage(f"After init {role} from HF AutoModel", logger=logger)

        # We wrap FSDP for rollout as well
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = PrecisionType.to_dtype(fsdp_config.dtype)
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        # Store param_dtype for QAT quantizer
        self._param_dtype = param_dtype

        auto_wrap_policy = get_fsdp_wrap_policy(
            module=actor_module,
            config=fsdp_config.get("wrap_policy", None),
            is_lora=self._is_lora,
        )

        # if self._is_rollout and self.config.rollout.name == "hf":
        #     # TODO(zhangchi.usc1992, shengguangming) fix me.
        #     Current, auto_wrap_policy causes HFRollout to hang in Gemma
        #     auto_wrap_policy = None

        if self.rank == 0:
            print(f"wrap_policy: {auto_wrap_policy}")

        fsdp_mesh = self.device_mesh
        fsdp_enable_zero3 = fsdp_config.reshard_after_forward
        sharding_strategy = get_sharding_strategy(fsdp_mesh, fsdp_enable_zero3)
        use_local_shard_group = bool(role == "actor" and getattr(self, "actor_fsdp_local_shard_group_only", False))
        if use_local_shard_group:
            from torch.distributed.fsdp import ShardingStrategy

            sharding_strategy = ShardingStrategy.FULL_SHARD if fsdp_enable_zero3 else ShardingStrategy.SHARD_GRAD_OP

        # TODO: add transformer policy
        # We force reference policy to use CPUOffload to save memory.
        # We force turn off CPUOffload for actor because it causes incorrect results when using grad accumulation
        cpu_offload = None if role == "actor" else CPUOffload(offload_params=True)
        fsdp_strategy = self.config.actor.strategy
        if fsdp_strategy == "fsdp":
            fsdp_kwargs = {
                "cpu_offload": cpu_offload,
                "param_init_fn": init_fn,
                "auto_wrap_policy": auto_wrap_policy,
                "device_id": get_device_id(),
                "sharding_strategy": sharding_strategy,
                "mixed_precision": mixed_precision,
                "sync_module_states": True,
                "use_orig_params": self.use_orig_params,
                "forward_prefetch": fsdp_config.get("forward_prefetch", False),
            }
            if use_local_shard_group:
                fsdp_kwargs["process_group"] = self.actor_fsdp_local_process_group
            else:
                fsdp_kwargs["device_mesh"] = self.device_mesh
            actor_module_fsdp = FSDP(actor_module, **fsdp_kwargs)
        elif fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(
                param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True
            )
            if role == "actor" and fsdp_config.offload_policy:
                cpu_offload = CPUOffloadPolicy(pin_memory=True)
                self._is_offload_param = False
                self._is_offload_optimizer = False
            else:
                cpu_offload = None if role == "actor" else CPUOffloadPolicy(pin_memory=True)

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": fsdp_config.reshard_after_forward,
                "shard_placement_fn": get_shard_placement_fn(fsdp_size=self.device_mesh.shape[-1]),
            }
            full_state = actor_module.state_dict()
            apply_fsdp2(actor_module, fsdp_kwargs, fsdp_config)
            fsdp2_load_full_state_dict(actor_module, full_state, fsdp_mesh, cpu_offload)
            actor_module_fsdp = actor_module
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        if enable_activation_offload:
            enable_activation_offloading(actor_module_fsdp, fsdp_strategy, enable_gradient_checkpointing)

        log_gpu_memory_usage(f"After {role} FSDP init", logger=logger)

        # TODO: add more optimizer args into config
        if role == "actor" and optim_config is not None:
            from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

            # SC_LR_MULT: give the self-conditioning adapter its own LR scale.
            # Its weights are ~10x smaller than mature transformer weights, so
            # Adam's scale-invariant steps rotate it ~10x faster in relative
            # terms (measured 12x); a mult < 1 brings its angular velocity in
            # line with the rest of the network. Default 1.0 = base behavior.
            _sc_lr_mult = float(os.environ.get("SC_LR_MULT", "1.0"))
            if _sc_lr_mult != 1.0:
                _sc_params, _other_params = [], []
                for _pname, _p in actor_module_fsdp.named_parameters():
                    (_sc_params if "self_conditioning" in _pname else _other_params).append(_p)
                if not _sc_params:
                    raise RuntimeError(
                        "SC_LR_MULT set but no self_conditioning parameters found "
                        "in actor_module_fsdp.named_parameters()"
                    )
                _base_lr = float(optim_config.lr)
                print(
                    f"[SC_LR_MULT] mult={_sc_lr_mult} sc_tensors={len(_sc_params)} "
                    f"sc_lr={_base_lr * _sc_lr_mult:.2e} base_lr={_base_lr:.2e}",
                    flush=True,
                )
                actor_optimizer = build_optimizer(
                    [
                        {"params": _other_params},
                        {"params": _sc_params, "lr": _base_lr * _sc_lr_mult},
                    ],
                    optim_config,
                )
            else:
                actor_optimizer = build_optimizer(actor_module_fsdp.parameters(), optim_config)

            total_steps = optim_config.get("total_training_steps", 0)
            num_warmup_steps = int(optim_config.get("lr_warmup_steps", -1))
            lr_scheduler_type = optim_config.get("lr_scheduler_type", "constant")
            min_lr_ratio = optim_config.get("min_lr_ratio", 0.0)
            num_cycles = optim_config.get("num_cycles", 0.5)
            if num_warmup_steps < 0:
                num_warmup_steps_ratio = optim_config.get("lr_warmup_steps_ratio", 0.0)
                num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            if self.rank == 0:
                print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

            if lr_scheduler_type == "constant":
                actor_lr_scheduler = get_constant_schedule_with_warmup(
                    optimizer=actor_optimizer, num_warmup_steps=num_warmup_steps
                )
            elif lr_scheduler_type == "cosine":
                actor_lr_scheduler = get_cosine_schedule_with_warmup(
                    optimizer=actor_optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=total_steps,
                    min_lr_ratio=min_lr_ratio,
                    num_cycles=num_cycles,
                )
            else:
                raise NotImplementedError(f"LR scheduler type {lr_scheduler_type} is not supported")

            log_gpu_memory_usage(f"After {role} optimizer init", logger=logger)
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config

    def _build_rollout(self, trust_remote_code=False):
        from torch.distributed.device_mesh import init_device_mesh

        # 1. parse rollout and huggingface model config
        rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.rollout)
        model_config: HFModelConfig = omega_conf_to_dataclass(self.config.model, dataclass_type=HFModelConfig)
        self.model_config = model_config

        # 2. build rollout device mesh
        infer_tp = self.config.rollout.tensor_model_parallel_size * self.config.rollout.data_parallel_size
        infer_pp = self.config.rollout.pipeline_model_parallel_size
        infer_world_size = infer_tp * infer_pp
        dp = self.world_size // infer_world_size
        assert self.world_size % infer_world_size == 0, (
            f"rollout world_size: {self.world_size} is not divisible by infer_world_size: {infer_world_size}"
        )
        rollout_device_mesh = init_device_mesh(
            device_name, mesh_shape=(dp, infer_tp, infer_pp), mesh_dim_names=["dp", "infer_tp", "infer_pp"]
        )
        rollout_name = self.config.rollout.name

        self.rollout_device_mesh = rollout_device_mesh

        if rollout_name == "hf":
            self._register_dispatch_collect_info("rollout", dp_rank=self.rank, is_collect=True)
        else:
            is_collect = (
                rollout_device_mesh["infer_tp"].get_local_rank() == 0
                and rollout_device_mesh["infer_pp"].get_local_rank() == 0
            )
            self._register_dispatch_collect_info(
                "rollout", dp_rank=rollout_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )

        # 4. build rollout model
        log_gpu_memory_usage(f"Before building {self.config.rollout.name} rollout", logger=logger)
        self.rollout = get_rollout_class(rollout_config.name, rollout_config.mode)(
            config=rollout_config, model_config=model_config, device_mesh=rollout_device_mesh
        )
        log_gpu_memory_usage(f"After building {self.config.rollout.name} rollout", logger=logger)

        # Full params
        if torch.distributed.get_world_size() == 1 and fsdp_version(self.actor_module_fsdp) == 1:
            FSDP.set_state_dict_type(
                self.actor_module_fsdp,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(),
            )
        elif fsdp_version(self.actor_module_fsdp) == 1:
            FSDP.set_state_dict_type(
                self.actor_module_fsdp,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        # used for LoRA
        self.base_sync_done: bool = "dummy" not in self.config.rollout.load_format
        self.layered_summon = self.config.rollout.get("layered_summon", False)

        # 5. switch to trainer mode
        # NOTE: It's critical that hybrid engine in trainer mode initially to load checkpoint.
        # For async mode, we can't call run_until_complete here, so we will switch to trainer mode in AgentLoopManager.
        # Note: sync mode is deprecated and rejected in RolloutConfig.__post_init__

    def _export_actor_weights(self, *, base_sync_done: bool):
        invalidate_prefix_cache = getattr(
            getattr(self, "actor", None),
            "invalidate_gemma_prefix_kv_cache",
            None,
        )
        if callable(invalidate_prefix_cache):
            invalidate_prefix_cache()
        peft_config = None
        peft_model = getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        if hasattr(peft_model, "peft_config"):  # LoRA
            peft_config = peft_model.peft_config.get("default", None)
            params = collect_lora_params(
                module=self.actor_module_fsdp,
                layered_summon=self.config.rollout.get("layered_summon", False),
                base_sync_done=base_sync_done,
            )
            if not base_sync_done:
                params = {replace_lora_wrapper(k, peft_config): v for k, v in params.items()}
        else:
            # Split-streaming actor-only workers do not call _build_rollout(),
            # which normally switches FSDP1 away from FULL_STATE_DICT. The nccl
            # checkpoint engine's send_weights() consumes plain tensors
            # (.nbytes/.view) and ShardedTensor rejects torch-function dispatch,
            # so a SHARDED_STATE_DICT export crashes it. Export a full state
            # dict offloaded to CPU on every rank instead — correct for models
            # up to tens of GB; revisit sharded streaming only for models where
            # a full CPU copy per rank is prohibitive.
            # DiffusionGemma (26B) keeps its paper branch's sharded export: under the
            # device-mesh FSDP1 wrap it yields DTensors, gathered one at a time below.
            # A full CPU copy per rank makes its weight sync 2-5x slower.
            if (
                self.config.rollout.checkpoint_engine.backend != "naive"
                and fsdp_version(self.actor_module_fsdp) == 1
            ):
                if getattr(getattr(peft_model, "config", None), "model_type", None) == "diffusion_gemma":
                    FSDP.set_state_dict_type(
                        self.actor_module_fsdp,
                        state_dict_type=StateDictType.SHARDED_STATE_DICT,
                        state_dict_config=ShardedStateDictConfig(),
                    )
                else:
                    FSDP.set_state_dict_type(
                        self.actor_module_fsdp,
                        state_dict_type=StateDictType.FULL_STATE_DICT,
                        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
                    )
            params = self.actor_module_fsdp.state_dict()

        from torch.distributed._shard.sharded_tensor import ShardedTensor as _ShardedTensor

        def _weight_nbytes(param):
            # ShardedTensor rejects torch-function dispatch for numel/element_size
            # (raises RuntimeError); size the full logical tensor from metadata.
            if isinstance(param, _ShardedTensor):
                total = 1
                for dim in param.metadata().size:
                    total *= int(dim)
                return total * param.dtype.itemsize
            return param.numel() * param.element_size()

        original_keys = len(params)
        original_bytes = sum(_weight_nbytes(param) for param in params.values())
        params = convert_weight_keys(params, peft_model)
        converted_keys = len(params)
        converted_bytes = sum(_weight_nbytes(param) for param in params.values())
        params, skipped_tied_aliases = deduplicate_diffusion_gemma_tied_weights(params, peft_model)

        if getattr(getattr(peft_model, "config", None), "model_type", None) == "diffusion_gemma":
            sent_bytes = sum(_weight_nbytes(param) for param in params.values())
            dedup_ratio = converted_bytes / sent_bytes if sent_bytes else 1.0
            if self.rank == 0:
                logger.info(
                    "DiffusionGemma weight sync: original_keys=%d original_bytes=%d "
                    "converted_keys=%d converted_bytes=%d sent_keys=%d sent_bytes=%d "
                    "skipped_tied_aliases=%d skipped_tied_bytes=%d dedup_ratio=%.6fx",
                    original_keys,
                    original_bytes,
                    converted_keys,
                    converted_bytes,
                    len(params),
                    sent_bytes,
                    len(skipped_tied_aliases),
                    converted_bytes - sent_bytes,
                    dedup_ratio,
                )

        if peft_config is not None and base_sync_done:
            per_tensor_param = params.items() if isinstance(params, dict) else params
        else:
            device = get_device_id()  # used when fsdp2 set cpu_offload_policy
            per_tensor_param = (
                (name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param)
                for name, param in params.items()
            )

        # QAT: quantize weights before sending to rollout.
        if self._qat_enabled:
            from verl.utils.qat.quantizer import QATQuantizer

            quantizer = QATQuantizer(
                mode=self.qat_config.mode,
                group_size=self.qat_config.group_size,
                ignore_patterns=self.qat_config.ignore_patterns,
                device=torch.device(get_device_id()),
                param_dtype=self._param_dtype,
            )
            per_tensor_param = quantizer.quantize_with_fusion(
                per_tensor_param,
                target_device=torch.device("cpu"),
            )
            aggressive_empty_cache(force_sync=True)

        return per_tensor_param, peft_config, params

    def _export_actor_weights_streaming(self, device: int):
        """Build the narrow Qwen/SDAR FSDP1 stream before any collectives."""
        peft_model = getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        if hasattr(peft_model, "peft_config"):
            return None, None, "lora_not_supported"

        if fsdp_version(self.actor_module_fsdp) != 1:
            return None, None, "fsdp_version_not_1"

        if self._qat_enabled:
            return None, None, "qat_not_supported"

        if not self.config.rollout.free_cache_engine:
            return None, None, "free_cache_engine_required"

        invalidate_prefix_cache = getattr(
            getattr(self, "actor", None),
            "invalidate_gemma_prefix_kv_cache",
            None,
        )
        if callable(invalidate_prefix_cache):
            invalidate_prefix_cache()

        try:
            stream = stream_fsdp1_params_layered(self.actor_module_fsdp, device)
        except StreamingExportUnavailable as exc:
            return None, None, exc.reason
        return stream, None, None

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def execute_checkpoint_engine(self, method: str, *args, **kwargs):
        checkpoint_engine = getattr(self, "checkpoint_engine", None)
        if checkpoint_engine is None:
            raise RuntimeError(
                "execute_checkpoint_engine requires a non-naive checkpoint engine initialized on ActorRolloutRefWorker"
            )
        return getattr(checkpoint_engine, method)(*args, **kwargs)

    async def rollout_mode(self):
        """Context switch hybridengine to rollout mode."""
        rollout_mode_start = time.perf_counter()
        emit_lifecycle(
            "ActorRolloutRefWorker",
            "rollout_mode_start",
            actor_uuid=self._lifecycle_actor_uuid,
            ray_actor_id=self._lifecycle_ray_actor_id,
            rank=int(self.rank),
            base_sync_done=bool(self.base_sync_done),
            hot_update_count=self._lifecycle_hot_update_count,
            model_init_count=self._lifecycle_model_init_count,
            checkpoint_load_count=self._lifecycle_checkpoint_load_count,
            memory=lifecycle_memory("before_rollout_weight_export"),
        )
        _rollout_mode_empty_cache()

        log_gpu_memory_usage("Before load_fsdp_model_to_gpu", logger=logger)
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        log_gpu_memory_usage("After load_fsdp_model_to_gpu", logger=logger)
        emit_lifecycle(
            "ActorRolloutRefWorker",
            "actor_weights_loaded_for_export",
            actor_uuid=self._lifecycle_actor_uuid,
            rank=int(self.rank),
            actor_module_id=id(self.actor_module_fsdp),
            memory=lifecycle_memory("actor_weights_loaded_for_export"),
        )

        # ---- shared-weight HBM snapshot (taken before any export) -----------
        _sw_metrics = getattr(self, "_shared_weight_metrics", None)
        _sw_enabled = shared_weights_enabled() and _sw_metrics is not None
        _sw_use_stream = _sw_enabled and stream_export_enabled()
        _sw_begin = _sw_metrics.begin_hot_sync() if _sw_enabled else {}
        if _sw_enabled:
            emit_lifecycle(
                "ActorRolloutRefWorker",
                "shared_weight_hot_sync_begin",
                actor_uuid=self._lifecycle_actor_uuid,
                rank=int(self.rank),
                stream_export=_sw_use_stream,
                **_sw_begin,
            )
        # ---------------------------------------------------------------------

        # Non-naive backends always use the full state_dict export path and return
        # early.  For the naive backend with streaming enabled, we defer the
        # state_dict call and try the streaming generator instead (lower peak HBM).
        if self.config.rollout.checkpoint_engine.backend != "naive":
            per_tensor_param, peft_config, params = self._export_actor_weights(base_sync_done=self.base_sync_done)
            assert getattr(self, "checkpoint_engine", None) is not None
            try:
                await self.checkpoint_engine.send_weights(per_tensor_param)
            finally:
                log_gpu_memory_usage("Before offload_fsdp_model_to_cpu", logger=logger)
                if self._is_offload_param:
                    offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                log_gpu_memory_usage("After offload_fsdp_model_to_cpu", logger=logger)
                emit_lifecycle(
                    "ActorRolloutRefWorker",
                    "actor_weights_offloaded_after_export",
                    actor_uuid=self._lifecycle_actor_uuid,
                    rank=int(self.rank),
                    memory=lifecycle_memory("actor_weights_offloaded_after_export"),
                )
                del params, per_tensor_param
                aggressive_empty_cache(force_sync=True)
            self.base_sync_done = True
            emit_lifecycle(
                "ActorRolloutRefWorker",
                "rollout_mode_done",
                actor_uuid=self._lifecycle_actor_uuid,
                rank=int(self.rank),
                backend=str(self.config.rollout.checkpoint_engine.backend),
                base_sync_done=bool(self.base_sync_done),
                seconds=time.perf_counter() - rollout_mode_start,
                memory=lifecycle_memory("rollout_mode_done"),
            )
            return

        device = get_device_id()

        # Build the Qwen/SDAR unit plan before any SGLang bucket can be updated.
        # Unsupported configurations take the unchanged full-state-dict path.
        _sw_stream: Optional[FSDP1QwenParamStream] = None
        _sw_fallback_reason = None
        _sw_baseline_export_seconds = 0.0
        per_tensor_param = None
        peft_config = None
        params = None

        if _sw_use_stream:
            _sw_stream, _, _sw_fallback_reason = self._export_actor_weights_streaming(device)
            if _sw_stream is not None:
                emit_lifecycle(
                    "ActorRolloutRefWorker",
                    "shared_weight_stream_plan_ready",
                    actor_uuid=self._lifecycle_actor_uuid,
                    rank=int(self.rank),
                    plan_signature=_sw_stream.plan.signature,
                    expected_params=len(_sw_stream.plan.expected_names),
                    fsdp_layers=len(_sw_stream.plan.units) - 1,
                    memory=lifecycle_memory("stream_plan_ready"),
                )
            else:
                export_started_at = time.perf_counter()
                per_tensor_param, peft_config, params = self._export_actor_weights(
                    base_sync_done=self.base_sync_done
                )
                _sw_baseline_export_seconds = time.perf_counter() - export_started_at
        else:
            export_started_at = time.perf_counter()
            per_tensor_param, peft_config, params = self._export_actor_weights(base_sync_done=self.base_sync_done)
            _sw_baseline_export_seconds = time.perf_counter() - export_started_at

        # Special handling for LoRA with sleep_level=2:
        # When sleep_level=2, base model weights are destroyed during each sleep cycle.
        # separately collect and update LoRA weights and base model weights through their respective interfaces.
        # Here: params contains LoRA weights, base_model_params contains base model weights.
        # Only needed if the rollout engine actually sleeps/frees weights (free_cache_engine=True).
        if (
            peft_config is not None
            and getattr(self.rollout, "sleep_level", None) == 2
            and self.config.rollout.free_cache_engine
        ):
            base_model_params = collect_lora_params(
                module=self.actor_module_fsdp,
                layered_summon=self.layered_summon,
                base_sync_done=False,
            )
            base_model_params = {replace_lora_wrapper(k, peft_config): v for k, v in base_model_params.items()}
            base_model_params = convert_weight_keys(
                base_model_params, getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
            )

        _sw_defer_actor_offload = _sw_stream is not None
        if _sw_defer_actor_offload:
            emit_lifecycle(
                "ActorRolloutRefWorker",
                "actor_weight_offload_deferred_for_stream",
                actor_uuid=self._lifecycle_actor_uuid,
                rank=int(self.rank),
                plan_signature=_sw_stream.plan.signature,
                memory=lifecycle_memory("actor_offload_deferred_for_stream"),
            )
        else:
            log_gpu_memory_usage("Before offload_fsdp_model_to_cpu", logger=logger)
            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload_fsdp_model_to_cpu", logger=logger)
            emit_lifecycle(
                "ActorRolloutRefWorker",
                "actor_weights_offloaded_before_rollout_wake",
                actor_uuid=self._lifecycle_actor_uuid,
                rank=int(self.rank),
                memory=lifecycle_memory("actor_weights_offloaded_before_rollout_wake"),
            )

        set_expandable_segments(False)

        if self.config.rollout.free_cache_engine and not self.config.rollout.enforce_eager:
            emit_lifecycle(
                "ActorRolloutRefWorker",
                "rollout_resume_start",
                actor_uuid=self._lifecycle_actor_uuid,
                rank=int(self.rank),
                tags=["cuda_graph"],
                memory=lifecycle_memory("before_rollout_resume_cuda_graph"),
            )
            await self.rollout.resume(tags=["cuda_graph"])
            log_gpu_memory_usage("After resume cuda_graph", logger=logger)
            emit_lifecycle(
                "ActorRolloutRefWorker",
                "rollout_resume_done",
                actor_uuid=self._lifecycle_actor_uuid,
                rank=int(self.rank),
                tags=["cuda_graph"],
                memory=lifecycle_memory("after_rollout_resume_cuda_graph"),
            )
        if self.config.rollout.free_cache_engine:
            emit_lifecycle(
                "ActorRolloutRefWorker",
                "rollout_resume_start",
                actor_uuid=self._lifecycle_actor_uuid,
                rank=int(self.rank),
                tags=["weights"],
                memory=lifecycle_memory("before_rollout_resume_weights"),
            )
            await self.rollout.resume(tags=["weights"])
        log_gpu_memory_usage("After resume weights", logger=logger)
        emit_lifecycle(
            "ActorRolloutRefWorker",
            "rollout_resume_done",
            actor_uuid=self._lifecycle_actor_uuid,
            rank=int(self.rank),
            tags=["weights"],
            memory=lifecycle_memory("after_rollout_resume_weights"),
        )

        if (
            peft_config is not None
            and getattr(self.rollout, "sleep_level", None) == 2
            and self.config.rollout.free_cache_engine
        ):
            per_tensor_base_params = (
                (name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param)
                for name, param in base_model_params.items()
            )
            await self.rollout.update_weights(per_tensor_base_params, base_sync_done=False)
            del base_model_params, per_tensor_base_params

        _sw_update_source = _sw_stream if _sw_stream is not None else per_tensor_param
        if _sw_enabled:
            _sw_copy_stream = _InstrumentedParamStream(_sw_update_source)
            _sw_update_source = _sw_copy_stream
        else:
            _sw_copy_stream = None

        _sw_update_returned = False
        _sw_update_started_at = time.perf_counter()
        if _sw_stream is not None:
            self._shared_weight_policy_valid = False
            _sw_invalidation_attempted = False
            _sw_partial_invalidated = False

            async def _invalidate_partial_rollout_weights():
                nonlocal _sw_invalidation_attempted, _sw_partial_invalidated
                _sw_invalidation_attempted = True
                emit_lifecycle(
                    "ActorRolloutRefWorker",
                    "shared_weight_partial_update_invalidation_start",
                    actor_uuid=self._lifecycle_actor_uuid,
                    rank=int(self.rank),
                    plan_signature=_sw_stream.plan.signature,
                    memory=lifecycle_memory("before_partial_weight_invalidation"),
                )
                await self.rollout.release()
                _sw_partial_invalidated = True
                emit_lifecycle(
                    "ActorRolloutRefWorker",
                    "shared_weight_partial_update_invalidated",
                    actor_uuid=self._lifecycle_actor_uuid,
                    rank=int(self.rank),
                    plan_signature=_sw_stream.plan.signature,
                    memory=lifecycle_memory("after_partial_weight_invalidation"),
                )

            try:
                await update_weights_with_failure_invalidation(
                    self.rollout.update_weights,
                    _invalidate_partial_rollout_weights,
                    _sw_update_source,
                    peft_config=peft_config,
                    base_sync_done=self.base_sync_done,
                )
                _sw_update_returned = True
                assert_stream_offload_safe(
                    _sw_stream,
                    update_returned=_sw_update_returned,
                )
                if _sw_copy_stream is None or not _sw_copy_stream.completed:
                    raise RuntimeError(
                        "instrumented stream did not finish before rollout.update_weights() returned"
                    )
            except BaseException as exc:
                self._shared_weight_policy_valid = False
                if not _sw_invalidation_attempted:
                    try:
                        await _invalidate_partial_rollout_weights()
                    except BaseException as cleanup_exc:
                        if hasattr(exc, "add_note"):
                            exc.add_note(
                                "rollout weight invalidation also failed: "
                                f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                            )
                if _sw_enabled:
                    failure_fields = _sw_metrics.on_hot_sync_failed(
                        f"{type(exc).__name__}:{exc}"
                    )
                    emit_lifecycle(
                        "ActorRolloutRefWorker",
                        "shared_weight_hot_sync_failed",
                        actor_uuid=self._lifecycle_actor_uuid,
                        rank=int(self.rank),
                        plan_signature=_sw_stream.plan.signature,
                        partial_policy_active=not _sw_partial_invalidated,
                        policy_generation_allowed=False,
                        actor_offload_attempted=False,
                        **failure_fields,
                    )
                set_expandable_segments(True)
                raise
        else:
            await self.rollout.update_weights(
                _sw_update_source,
                peft_config=peft_config,
                base_sync_done=self.base_sync_done,
            )
            _sw_update_returned = True

        _sw_update_seconds = time.perf_counter() - _sw_update_started_at
        log_gpu_memory_usage("After update_weights", logger=logger)

        emit_lifecycle(
            "ActorRolloutRefWorker",
            "rollout_existing_engine_weights_updated",
            actor_uuid=self._lifecycle_actor_uuid,
            rank=int(self.rank),
            actor_module_id=id(self.actor_module_fsdp),
            base_sync_done_before=bool(self.base_sync_done),
            checkpoint_load_count=self._lifecycle_checkpoint_load_count,
            memory=lifecycle_memory("after_existing_rollout_engine_weight_update"),
        )

        if _sw_enabled and _sw_copy_stream is not None:
            if _sw_stream is not None:
                _sw_allgather_bytes = _sw_stream.allgather_bytes
                _sw_allgather_seconds = _sw_stream.allgather_seconds
                _sw_fsdp_layers_streamed = _sw_stream.fsdp_layers_streamed
                _sw_fsdp_units_streamed = _sw_stream.fsdp_units_streamed
            else:
                _sw_allgather_bytes = _sw_copy_stream.bytes_yielded
                _sw_allgather_seconds = _sw_baseline_export_seconds
                _sw_fsdp_layers_streamed = 0
                _sw_fsdp_units_streamed = 0

            _sw_allgather_fields = _sw_metrics.on_allgather_done(
                allgather_bytes=_sw_allgather_bytes,
                allgather_seconds=_sw_allgather_seconds,
                stream_export=_sw_stream is not None,
                fsdp_layers_streamed=_sw_fsdp_layers_streamed,
                fsdp_units_streamed=_sw_fsdp_units_streamed,
                params_streamed=_sw_copy_stream.params_yielded,
                fallback_reason=_sw_fallback_reason,
            )
            emit_lifecycle(
                "ActorRolloutRefWorker",
                "shared_weight_allgather_done",
                actor_uuid=self._lifecycle_actor_uuid,
                rank=int(self.rank),
                generator_consumed=bool(
                    _sw_stream is None or _sw_stream.completed
                ),
                update_returned=_sw_update_returned,
                **_sw_allgather_fields,
            )

        if _sw_stream is not None:
            assert_stream_offload_safe(
                _sw_stream,
                update_returned=_sw_update_returned,
            )
            emit_lifecycle(
                "ActorRolloutRefWorker",
                "shared_weight_stream_offload_assertion_passed",
                actor_uuid=self._lifecycle_actor_uuid,
                rank=int(self.rank),
                generator_consumed=_sw_stream.completed,
                update_returned=_sw_update_returned,
                fsdp_units_streamed=_sw_stream.fsdp_units_streamed,
                plan_signature=_sw_stream.plan.signature,
            )
            log_gpu_memory_usage(
                "Before deferred offload_fsdp_model_to_cpu",
                logger=logger,
            )
            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage(
                "After deferred offload_fsdp_model_to_cpu",
                logger=logger,
            )
            emit_lifecycle(
                "ActorRolloutRefWorker",
                "actor_weights_offloaded_after_stream_update",
                actor_uuid=self._lifecycle_actor_uuid,
                rank=int(self.rank),
                generator_consumed=True,
                update_returned=True,
                memory=lifecycle_memory("actor_offloaded_after_stream_update"),
            )

        if _sw_enabled and _sw_copy_stream is not None:
            _sw_allgather_seconds = (
                _sw_stream.allgather_seconds
                if _sw_stream is not None
                else _sw_baseline_export_seconds
            )
            _sw_copy_seconds = max(
                0.0,
                _sw_update_seconds - _sw_allgather_seconds,
            )
            _sw_done_fields = _sw_metrics.on_hot_sync_done(
                copy_bytes=_sw_copy_stream.bytes_yielded,
                copy_seconds=_sw_copy_seconds,
                update_seconds=_sw_update_seconds,
            )
            emit_lifecycle(
                "ActorRolloutRefWorker",
                "shared_weight_hot_sync_done",
                actor_uuid=self._lifecycle_actor_uuid,
                rank=int(self.rank),
                stream_export_used=_sw_stream is not None,
                copy_stream_params=getattr(_sw_copy_stream, "params_yielded", 0),
                **_sw_done_fields,
            )

        if params is not None:
            del params
        if per_tensor_param is not None:
            del per_tensor_param
        _sw_stream = None
        _rollout_mode_empty_cache()
        if self.config.rollout.free_cache_engine:
            emit_lifecycle(
                "ActorRolloutRefWorker",
                "rollout_resume_start",
                actor_uuid=self._lifecycle_actor_uuid,
                rank=int(self.rank),
                tags=["kv_cache"],
                memory=lifecycle_memory("before_rollout_resume_kv_cache"),
            )
            await self.rollout.resume(tags=["kv_cache"])
        log_gpu_memory_usage("After resume kv_cache", logger=logger)
        emit_lifecycle(
            "ActorRolloutRefWorker",
            "rollout_resume_done",
            actor_uuid=self._lifecycle_actor_uuid,
            rank=int(self.rank),
            tags=["kv_cache"],
            memory=lifecycle_memory("after_rollout_resume_kv_cache"),
        )

        self._shared_weight_policy_valid = True
        self.base_sync_done = True
        set_expandable_segments(True)
        emit_lifecycle(
            "ActorRolloutRefWorker",
            "rollout_mode_done",
            actor_uuid=self._lifecycle_actor_uuid,
            ray_actor_id=self._lifecycle_ray_actor_id,
            rank=int(self.rank),
            backend="naive",
            base_sync_done=bool(self.base_sync_done),
            hot_update_count=self._lifecycle_hot_update_count,
            model_init_count=self._lifecycle_model_init_count,
            checkpoint_load_count=self._lifecycle_checkpoint_load_count,
            seconds=time.perf_counter() - rollout_mode_start,
            memory=lifecycle_memory("after_hot_wake"),
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl.workers.actor import DataParallelPPOActor

        _start_stack_sampler_if_configured(self.config, int(self.rank))
        # Perf switches that must reach the worker regardless of env propagation:
        # `+actor_rollout_ref.model.override_config.rollout_mode_light_empty_cache=true`.
        try:
            global _LIGHT_EMPTY_CACHE_FROM_CONFIG
            _oc = self.config.model.get("override_config", {}) or {}
            _LIGHT_EMPTY_CACHE_FROM_CONFIG = bool(_oc.get("rollout_mode_light_empty_cache", False))
            # Generic worker-env injection: `+actor_rollout_ref.model.override_config.env__FOO=bar`
            # sets os.environ["FOO"]="bar" inside every worker process before the model is
            # built (env vars passed to the driver do not reliably reach Ray actors).
            for _k in list(_oc.keys()):
                if str(_k).startswith("env__"):
                    os.environ[str(_k)[5:]] = str(_oc.get(_k))
                    print(f"[WORKER-ENV] {str(_k)[5:]}={_oc.get(_k)}", flush=True)
        except Exception:
            pass
        self._lifecycle_model_init_count += 1
        if self._lifecycle_model_init_count != 1:
            raise RuntimeError(
                "persistent actor model was initialized more than once: "
                f"actor_uuid={self._lifecycle_actor_uuid} count={self._lifecycle_model_init_count}"
            )
        emit_lifecycle(
            "ActorRolloutRefWorker",
            "model_init_start",
            actor_uuid=self._lifecycle_actor_uuid,
            ray_actor_id=self._lifecycle_ray_actor_id,
            rank=int(self.rank),
            role=self.role,
            construction_count=self._lifecycle_construction_count,
            model_init_count=self._lifecycle_model_init_count,
            checkpoint_load_count=self._lifecycle_checkpoint_load_count,
            model_path=str(self.config.model.path),
            memory=lifecycle_memory("before_model_init"),
        )

        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        # Initialize QAT config before _build_model_optimizer
        self._init_qat_config()

        override_model_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))
        use_remove_padding = self.config.model.get("use_remove_padding", False)
        use_shm = self.config.model.get("use_shm", False)
        use_fused_kernels = self.config.model.get("use_fused_kernels", False)

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = omega_conf_to_dataclass(self.config.actor.fsdp_config)
            else:
                optim_config = None
                fsdp_config = FSDPEngineConfig()

            local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            # TiledMLP configuration for memory-efficient MLP computation
            tiled_mlp_config = self.config.model.get("tiled_mlp", {})
            use_tiled_mlp = tiled_mlp_config.get("enabled", False)
            tiled_mlp_shards = tiled_mlp_config.get("num_shards", 4)

            (
                self.actor_module_fsdp,
                self.actor_optimizer,
                self.actor_lr_scheduler,
                self.actor_model_config,
            ) = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                enable_gradient_checkpointing=self.config.model.get("enable_gradient_checkpointing", False),
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="actor",
                enable_activation_offload=self.config.model.get("enable_activation_offload", False),
                use_prefix_grouper=self.config.actor.get("use_prefix_grouper", False),
                use_tiled_mlp=use_tiled_mlp,
                tiled_mlp_shards=tiled_mlp_shards,
            )

            # get the original unwrapped module
            if fsdp_version(self.actor_module_fsdp) == 1:
                self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                log_gpu_memory_usage("After offload actor model during init", logger=logger)

            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage("After offload actor optimizer during init", logger=logger)

        if self._is_actor:
            actor_cfg = omega_conf_to_dataclass(self.config.actor)
            self.actor = DataParallelPPOActor(
                config=actor_cfg, actor_module=self.actor_module_fsdp, actor_optimizer=self.actor_optimizer
            )
            if getattr(self, "actor_fsdp_local_shard_group_only", False):
                self.actor.set_replica_grad_sync_group(
                    self.actor_replica_grad_process_group,
                    group_size=int(self.actor_replica_grad_group_size),
                )

        if self._is_rollout:
            self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))

        if self._is_ref:
            ref_model_path = self.config.model.path
            ref_model = self.config.ref.get("model", None)
            if ref_model is not None:
                ref_model_path = ref_model.get("path", self.config.model.path)

            if self.rank == 0:
                print("reference model:", ref_model_path)
            local_path = copy_to_local(ref_model_path, use_shm=use_shm)
            use_prefix_grouper = hasattr(self.config, "actor") and self.config.actor.get("use_prefix_grouper", False)

            # TiledMLP for ref model: use ref config if specified, otherwise use actor config
            ref_tiled_mlp_config = self.config.ref.get("tiled_mlp", None)
            if ref_tiled_mlp_config is None:
                ref_tiled_mlp_config = self.config.model.get("tiled_mlp", {})
            ref_use_tiled_mlp = ref_tiled_mlp_config.get("enabled", False)
            ref_tiled_mlp_shards = ref_tiled_mlp_config.get("num_shards", 4)

            self.ref_module_fsdp = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=omega_conf_to_dataclass(self.config.ref.fsdp_config),
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="ref",
                use_prefix_grouper=use_prefix_grouper,
                use_tiled_mlp=ref_use_tiled_mlp,
                tiled_mlp_shards=ref_tiled_mlp_shards,
            )[0]
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
                self.config.ref.use_fused_kernels = use_fused_kernels
                if use_prefix_grouper:
                    self.config.ref.use_prefix_grouper = use_prefix_grouper
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)

        if self._is_actor:
            self.base_sync_done = "dummy" not in self.config.rollout.load_format
            self.layered_summon = self.config.rollout.get("layered_summon", False)
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_config=self.config.actor.checkpoint,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
            )
            checkpoint_engine_config = omega_conf_to_dataclass(self.config.rollout.checkpoint_engine)
            if checkpoint_engine_config.backend != "naive":
                bucket_size = checkpoint_engine_config.update_weights_bucket_megabytes << 20
                engine_kwargs = checkpoint_engine_config.engine_kwargs.get(checkpoint_engine_config.backend, {})
                self.checkpoint_engine = CheckpointEngineRegistry.new(
                    checkpoint_engine_config.backend,
                    is_master=(torch.distributed.get_rank() == 0),
                    bucket_size=bucket_size,
                    **engine_kwargs,
                )

        if not self._is_actor and self._is_rollout:
            # If ActorRolloutRefWorker is initialized as a standalone rollout,
            # create a checkpoint manager for FSDP model to allow loading FSDP checkpoints for rollout.

            checkpoint_contents = OmegaConf.create({"load_contents": ["model"], "save_contents": []})
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=None,
                lr_scheduler=None,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_config=checkpoint_contents,
            )
        # Shared-weight reuse metrics (feature-flagged; initialised regardless so
        # that rollout_mode() can always reference self._shared_weight_metrics).
        self._shared_weight_metrics: SharedWeightMetrics = SharedWeightMetrics(
            actor_uuid=self._lifecycle_actor_uuid,
        )
        self._shared_weight_policy_valid = False
        emit_lifecycle(
            "ActorRolloutRefWorker",
            "shared_weight_metrics_init",
            actor_uuid=self._lifecycle_actor_uuid,
            rank=int(self.rank),
            shared_weights_enabled=shared_weights_enabled(),
            stream_export_enabled=stream_export_enabled(),
            metrics_uuid=self._shared_weight_metrics.instance_uuid,
        )
        emit_lifecycle(
            "ActorRolloutRefWorker",
            "model_init_done",
            actor_uuid=self._lifecycle_actor_uuid,
            ray_actor_id=self._lifecycle_ray_actor_id,
            rank=int(self.rank),
            role=self.role,
            actor_module_id=id(self.actor_module_fsdp),
            construction_count=self._lifecycle_construction_count,
            model_init_count=self._lifecycle_model_init_count,
            checkpoint_load_count=self._lifecycle_checkpoint_load_count,
            base_sync_done=bool(getattr(self, "base_sync_done", False)),
            memory=lifecycle_memory("after_model_init"),
        )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="red", role="actor_update")
    def update_actor(self, data: DataProto):
        assert self._is_actor
        emit_lifecycle(
            "ActorRolloutRefWorker",
            "actor_update_start",
            actor_uuid=self._lifecycle_actor_uuid,
            ray_actor_id=self._lifecycle_ray_actor_id,
            rank=int(self.rank),
            actor_module_id=id(self.actor_module_fsdp),
            global_step=data.meta_info.get("global_step"),
            model_init_count=self._lifecycle_model_init_count,
            checkpoint_load_count=self._lifecycle_checkpoint_load_count,
            memory=lifecycle_memory("before_actor_forward_backward"),
        )
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_device_id())

        with self.ulysses_sharding_manager:
            data = data.to("cpu")  # data will to device with each micro batch on actor.update_policy
            data.meta_info.setdefault("pad_token_id", self.tokenizer.pad_token_id)
            # perform training
            with Timer(name="update_policy", logger=None) as timer:
                metrics = self.actor.update_policy(data=data)
            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            images_seqlens = data.meta_info.get("images_seqlens", None)
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(
                global_num_tokens, delta_time, images_seqlens=images_seqlens
            )
            metrics["perf/mfu/actor"] = (
                estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
            )
            metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
            metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr.item() if torch.is_tensor(lr) else lr
            self.actor_lr_scheduler.step()

            # TODO: here, we should return all metrics
            output = DataProto(meta_info={"metrics": metrics})

            output = output.to("cpu")

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during update_actor", logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
            log_gpu_memory_usage("After offload actor optimizer during update_actor", logger=logger)

        emit_lifecycle(
            "ActorRolloutRefWorker",
            "actor_update_done",
            actor_uuid=self._lifecycle_actor_uuid,
            ray_actor_id=self._lifecycle_ray_actor_id,
            rank=int(self.rank),
            actor_module_id=id(self.actor_module_fsdp),
            global_step=data.meta_info.get("global_step"),
            optimizer_step_count=getattr(self.actor, "_lifecycle_optimizer_step_count", 0),
            grad_norm=metrics.get("actor/grad_norm"),
            optimizer_steps=metrics.get("actor/optimizer_steps"),
            model_init_count=self._lifecycle_model_init_count,
            checkpoint_load_count=self._lifecycle_checkpoint_load_count,
            memory=lifecycle_memory("after_actor_update_and_offload"),
        )
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def begin_replay_accum(self, expected_rows: int | None = None, meta_info: dict | None = None):
        """Begin an experimental replay accumulation window."""
        assert self._is_actor
        if getattr(self, "_streaming_replay_active", False):
            raise RuntimeError("begin_replay_accum called while a replay accumulation window is already active")
        if int(self.config.actor.get("ppo_epochs", 1)) != 1:
            raise NotImplementedError("split_gpu_streaming immediate replay accumulation currently requires ppo_epochs=1")

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_device_id())

        self.ulysses_sharding_manager.__enter__()
        self.actor.actor_optimizer.zero_grad()
        self._streaming_replay_active = True
        self._streaming_replay_started_at = time.perf_counter()
        self._streaming_replay_metrics = {}
        self._streaming_replay_chunks = 0
        self._streaming_replay_rows = 0
        self._streaming_replay_expected_rows = int(expected_rows or 0)
        self._streaming_replay_meta_info = dict(meta_info or {})
        generation_jsd_cfg = self.config.actor.get("generation_consistency_jsd", {}) or {}
        block_tpf_cfg = generation_jsd_cfg.get("block_tpf_ce_weight", {}) or {}
        self._streaming_replay_buffer_block_tpf = bool(
            generation_jsd_cfg.get("enable", False)
            and generation_jsd_cfg.get("objective", "full_jsd") == "trace_all_ce"
            and block_tpf_cfg.get("enable", False)
        )
        adjacent_entropy_cfg = (
            generation_jsd_cfg.get("adjacent_entropy_contraction", {}) or {}
        )
        self._streaming_replay_buffer_adjacent_entropy = bool(
            generation_jsd_cfg.get("enable", False)
            and generation_jsd_cfg.get("objective", "full_jsd")
            == "trace_all_ce"
            and adjacent_entropy_cfg.get("enable", False)
        )
        self._streaming_replay_buffer_global_objective = bool(
            self._streaming_replay_buffer_block_tpf
            or self._streaming_replay_buffer_adjacent_entropy
        )
        self._streaming_replay_buffered_chunks = []
        return {"streaming_replay/begin": 1.0}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def consume_replay_chunk(self, data: DataProto):
        assert self._is_actor
        if not getattr(self, "_streaming_replay_active", False):
            self.begin_replay_accum(
                expected_rows=data.batch.batch_size[0] if data.batch is not None else 0,
                meta_info=data.meta_info,
            )
        chunk_rows = int(data.batch.batch_size[0] if data.batch is not None else 0)
        data = data.to("cpu")
        data.meta_info.update(self._streaming_replay_meta_info)
        data.meta_info.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        if getattr(
            self,
            "_streaming_replay_buffer_global_objective",
            False,
        ):
            # Clipped block weights and adjacent-entropy contraction both need
            # exact whole-batch denominators before backward. Keep externally
            # materialized chunks on CPU and run one normal actor update after
            # the final chunk arrives.
            self._streaming_replay_buffered_chunks.append(data)
            self._streaming_replay_chunks += 1
            self._streaming_replay_rows += chunk_rows
            return {
                "streaming_replay/chunks": float(self._streaming_replay_chunks),
                "streaming_replay/latest_rows": float(chunk_rows),
                "streaming_replay/global_objective_buffered": 1.0,
                "streaming_replay/block_tpf_buffered": float(
                    self._streaming_replay_buffer_block_tpf
                ),
                "streaming_replay/adjacent_entropy_buffered": float(
                    self._streaming_replay_buffer_adjacent_entropy
                ),
            }
        data.meta_info["_streaming_accumulate_only"] = True
        data.meta_info["_streaming_disable_progress"] = True
        data.meta_info["_streaming_total_rows"] = int(self._streaming_replay_expected_rows or chunk_rows)

        with Timer(name="streaming_replay_consume", logger=None) as timer:
            metrics = self.actor.update_policy(data=data)
        append_to_dict(self._streaming_replay_metrics, metrics)
        self._streaming_replay_chunks += 1
        self._streaming_replay_rows += chunk_rows
        return {
            "streaming_replay/chunks": float(self._streaming_replay_chunks),
            "streaming_replay/latest_rows": float(chunk_rows),
            "streaming_replay/consume_seconds": timer.last,
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def finish_replay_update(self):
        assert self._is_actor
        if not getattr(self, "_streaming_replay_active", False):
            return DataProto(meta_info={"metrics": {"streaming_replay/chunks": 0.0}})

        metrics = self._streaming_replay_metrics
        try:
            if getattr(
                self,
                "_streaming_replay_buffer_global_objective",
                False,
            ):
                buffered_chunks = getattr(self, "_streaming_replay_buffered_chunks", [])
                if not buffered_chunks:
                    raise RuntimeError(
                        "global replay objective window finished without buffered chunks"
                    )
                combined = DataProto.concat(buffered_chunks)
                combined.meta_info.update(self._streaming_replay_meta_info)
                combined.meta_info.setdefault("pad_token_id", self.tokenizer.pad_token_id)
                combined.meta_info["_streaming_disable_progress"] = True
                combined.meta_info.pop("_streaming_accumulate_only", None)
                combined.meta_info.pop("_streaming_total_rows", None)
                buffered_metrics = self.actor.update_policy(data=combined)
                metrics.update(buffered_metrics)
                if self._streaming_replay_buffer_block_tpf:
                    metrics[
                        "actor/block_tpf_ce/external_stream_buffered"
                    ] = 1.0
                if self._streaming_replay_buffer_adjacent_entropy:
                    metrics[
                        "adj_ent/external_stream_buffered"
                    ] = 1.0
            else:
                grad_norm = self.actor._optimizer_step()
                total_grad_values = getattr(self.actor, "_last_total_grad_values", 0)
                nonfinite_grad_values = getattr(self.actor, "_last_nonfinite_grad_values", 0)
                append_to_dict(
                    metrics,
                    {
                        "actor/grad_norm": grad_norm.detach().item(),
                        "actor/nonfinite_grad_tensors": float(
                            getattr(self.actor, "_last_nonfinite_grad_tensors", 0)
                        ),
                        "actor/nonfinite_grad_values": float(nonfinite_grad_values),
                        "actor/nonfinite_grad_frac": (
                            float(nonfinite_grad_values) / float(total_grad_values)
                            if total_grad_values
                            else 0.0
                        ),
                    },
                )
            delta_time = time.perf_counter() - self._streaming_replay_started_at
            self.actor.actor_optimizer.zero_grad()

            global_num_tokens = self._streaming_replay_meta_info.get("global_token_num", None)
            if global_num_tokens is not None:
                images_seqlens = self._streaming_replay_meta_info.get("images_seqlens", None)
                estimated_flops, promised_flops = self.flops_counter.estimate_flops(
                    global_num_tokens, delta_time, images_seqlens=images_seqlens
                )
                metrics["perf/mfu/actor"] = (
                    estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
                )
            metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
            metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr.item() if torch.is_tensor(lr) else lr
            self.actor_lr_scheduler.step()
            metrics["streaming_replay/chunks"] = float(self._streaming_replay_chunks)
            metrics["streaming_replay/rows"] = float(self._streaming_replay_rows)
            metrics["streaming_replay/expected_rows"] = float(self._streaming_replay_expected_rows)
            metrics["streaming_replay/update_seconds"] = delta_time

            output = DataProto(meta_info={"metrics": metrics})
            output = output.to("cpu")
            return output
        finally:
            self.ulysses_sharding_manager.__exit__(None, None, None)
            self._streaming_replay_active = False
            self._streaming_replay_metrics = {}
            self._streaming_replay_buffer_block_tpf = False
            self._streaming_replay_buffer_adjacent_entropy = False
            self._streaming_replay_buffer_global_objective = False
            self._streaming_replay_buffered_chunks = []
            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                log_gpu_memory_usage("After offload actor model during streaming replay update", logger=logger)
            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage("After offload actor optimizer during streaming replay update", logger=logger)

    def _reset_t5_tensor_replay_state(self):
        self._t5_tensor_replay_active = False
        self._t5_tensor_replay_started_at = 0.0
        self._t5_tensor_replay_metrics = {}
        self._t5_tensor_replay_chunks = 0
        self._t5_tensor_replay_rows = 0
        self._t5_tensor_replay_active_tokens = 0.0
        self._t5_tensor_replay_loss_weighted_sum = 0.0
        self._t5_tensor_replay_expected_rows = 0
        self._t5_tensor_replay_expected_active_tokens = 0.0
        self._t5_tensor_replay_meta_info = {}
        self._t5_tensor_replay_unknown_total_grad_rescale = False
        self._t5_tensor_replay_normalize_at_finish = False
        self._t5_tensor_replay_provisional_denominator = 0.0
        self._t5_tensor_replay_use_provisional_denominator = False
        self._t5_tensor_replay_grad_rescale_basis = "active_tokens"
        self._t5_tensor_replay_last_chunk_idx = -1
        self._t5_tensor_replay_seen_chunk_indices = set()

    def _cleanup_t5_tensor_replay_window(self, offload_log_suffix: str):
        try:
            if hasattr(self, "actor"):
                self.actor._cleanup_gemma_replay_fp32_grad_accum()
                # Invalidate incremental prefix KV cache: weights may change after
                # the optimizer step or the window aborted without completing.
                self.actor.invalidate_gemma_prefix_kv_cache()
            self.ulysses_sharding_manager.__exit__(None, None, None)
        finally:
            self._reset_t5_tensor_replay_state()
            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                log_gpu_memory_usage(f"After offload actor model during {offload_log_suffix}", logger=logger)
            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage(f"After offload actor optimizer during {offload_log_suffix}", logger=logger)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def debug_t5_tensor_grad_parity(
        self,
        tensor_batch: dict[str, torch.Tensor],
        meta_info: dict | None = None,
        chunk_size: int = 0,
        max_rows: int = 16,
        sample_params: int = 64,
        sample_elems_per_param: int = 256,
        atol: float = 1.0e-4,
        rtol: float = 2.0e-3,
        raise_on_fail: bool = True,
    ):
        """Run debug-only compact tensor full-vs-chunk grad parity without a step."""
        assert self._is_actor
        if getattr(self, "_t5_tensor_replay_active", False):
            raise RuntimeError("debug_t5_tensor_grad_parity called while a tensor replay window is active")

        entered_sharding_manager = False
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_device_id())
        try:
            self.ulysses_sharding_manager.__enter__()
            entered_sharding_manager = True
            return self.actor.debug_t5_tensor_grad_parity(
                tensor_batch=tensor_batch,
                meta_info=meta_info,
                chunk_size=chunk_size,
                max_rows=max_rows,
                sample_params=sample_params,
                sample_elems_per_param=sample_elems_per_param,
                atol=atol,
                rtol=rtol,
                raise_on_fail=raise_on_fail,
            )
        finally:
            try:
                self.actor.actor_optimizer.zero_grad()
            finally:
                if entered_sharding_manager:
                    self.ulysses_sharding_manager.__exit__(None, None, None)
                if self._is_offload_param:
                    offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                    log_gpu_memory_usage("After offload actor model during t5 tensor grad parity", logger=logger)
                if self._is_offload_optimizer:
                    offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                    log_gpu_memory_usage("After offload actor optimizer during t5 tensor grad parity", logger=logger)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def debug_t5_tensor_objective_grad_parity(
        self,
        tensor_batch: dict[str, torch.Tensor],
        meta_info: dict | None = None,
        max_rows: int = 16,
        sample_params: int = 64,
        sample_elems_per_param: int = 256,
        atol: float = 1.0e-4,
        rtol: float = 2.0e-3,
        raise_on_fail: bool = True,
    ):
        """Run debug-only trace_all_ce vs trace_future hard-label CE grad parity."""
        assert self._is_actor
        if getattr(self, "_t5_tensor_replay_active", False):
            raise RuntimeError("debug_t5_tensor_objective_grad_parity called while a tensor replay window is active")

        entered_sharding_manager = False
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_device_id())
        try:
            self.ulysses_sharding_manager.__enter__()
            entered_sharding_manager = True
            return self.actor.debug_t5_tensor_objective_grad_parity(
                tensor_batch=tensor_batch,
                meta_info=meta_info,
                max_rows=max_rows,
                sample_params=sample_params,
                sample_elems_per_param=sample_elems_per_param,
                atol=atol,
                rtol=rtol,
                raise_on_fail=raise_on_fail,
            )
        finally:
            try:
                self.actor.actor_optimizer.zero_grad()
            finally:
                if entered_sharding_manager:
                    self.ulysses_sharding_manager.__exit__(None, None, None)
                if self._is_offload_param:
                    offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                    log_gpu_memory_usage(
                        "After offload actor model during t5 tensor objective grad parity",
                        logger=logger,
                    )
                if self._is_offload_optimizer:
                    offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                    log_gpu_memory_usage(
                        "After offload actor optimizer during t5 tensor objective grad parity",
                        logger=logger,
                    )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def begin_t5_tensor_accum(
        self,
        expected_rows: int | None = None,
        expected_active_tokens: int | None = None,
        meta_info: dict | None = None,
    ):
        """Begin compact tensor replay accumulation for trace_future_ce."""
        assert self._is_actor
        if getattr(self, "_t5_tensor_replay_active", False):
            raise RuntimeError("begin_t5_tensor_accum called while a tensor replay window is already active")
        if int(self.config.actor.get("ppo_epochs", 1)) != 1:
            raise NotImplementedError("t5 tensor replay accumulation currently requires ppo_epochs=1")

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_device_id())

        self.ulysses_sharding_manager.__enter__()
        self.actor.actor_optimizer.zero_grad()
        if self.actor._gemma_replay_fp32_grad_accum_enabled(self.actor.config.get("generation_consistency_jsd", {})):
            self.actor._reset_gemma_replay_fp32_grad_accum_window()
        # Invalidate the incremental prefix KV cache at the start of each update
        # window so entries from a previous global step are never reused.
        self.actor.invalidate_gemma_prefix_kv_cache()
        self._t5_tensor_replay_active = True
        self._t5_tensor_replay_started_at = time.perf_counter()
        self._t5_tensor_replay_metrics = {}
        self._t5_tensor_replay_chunks = 0
        self._t5_tensor_replay_rows = 0
        self._t5_tensor_replay_active_tokens = 0.0
        self._t5_tensor_replay_loss_weighted_sum = 0.0
        self._t5_tensor_replay_last_chunk_idx = -1
        self._t5_tensor_replay_expected_rows = int(expected_rows or 0)
        self._t5_tensor_replay_expected_active_tokens = float(expected_active_tokens or 0)
        self._t5_tensor_replay_meta_info = dict(meta_info or {})
        self._t5_tensor_replay_normalize_at_finish = bool(
            self._t5_tensor_replay_meta_info.get("_t5_tensor_normalize_at_finish", False)
        )
        self._t5_tensor_replay_unknown_total_grad_rescale = (
            self._t5_tensor_replay_normalize_at_finish
            or (
                self._t5_tensor_replay_expected_rows <= 0
                and self._t5_tensor_replay_expected_active_tokens <= 0
            )
        )
        self._t5_tensor_replay_provisional_denominator = float(
            self._t5_tensor_replay_meta_info.get("_t5_tensor_provisional_denominator", 0.0) or 0.0
        )
        grad_rescale_basis = str(
            self._t5_tensor_replay_meta_info.get("_t5_tensor_grad_rescale_basis", "active_tokens")
            or "active_tokens"
        ).strip().lower()
        if grad_rescale_basis in {"row", "rows"}:
            grad_rescale_basis = "rows"
        elif grad_rescale_basis in {"active", "active_token", "active_tokens", "token", "tokens"}:
            grad_rescale_basis = "active_tokens"
        else:
            raise ValueError(
                "_t5_tensor_grad_rescale_basis must be rows or active_tokens, "
                f"got {self._t5_tensor_replay_meta_info.get('_t5_tensor_grad_rescale_basis')!r}"
            )
        self._t5_tensor_replay_grad_rescale_basis = grad_rescale_basis
        self._t5_tensor_replay_use_provisional_denominator = (
            self._t5_tensor_replay_unknown_total_grad_rescale
            and self._t5_tensor_replay_provisional_denominator > 0.0
        )
        if not self._t5_tensor_replay_use_provisional_denominator:
            self._t5_tensor_replay_provisional_denominator = 0.0
        self._t5_tensor_replay_meta_info["_t5_tensor_total_rows"] = self._t5_tensor_replay_expected_rows
        self._t5_tensor_replay_meta_info["_t5_tensor_total_active_tokens"] = (
            self._t5_tensor_replay_expected_active_tokens
        )
        self._t5_tensor_replay_meta_info["_t5_tensor_unknown_total_grad_rescale"] = (
            self._t5_tensor_replay_unknown_total_grad_rescale
        )
        self._t5_tensor_replay_meta_info["_t5_tensor_provisional_denominator"] = (
            self._t5_tensor_replay_provisional_denominator
        )
        self._t5_tensor_replay_meta_info["_t5_tensor_grad_rescale_basis"] = grad_rescale_basis
        print(
            "[SplitStreaming] actor_begin "
            f"ts={time.time():.6f} "
            f"expected_rows={self._t5_tensor_replay_expected_rows} "
            f"expected_active_tokens={self._t5_tensor_replay_expected_active_tokens} "
            f"normalize_at_finish={self._t5_tensor_replay_normalize_at_finish} "
            f"unknown_total_grad_rescale={self._t5_tensor_replay_unknown_total_grad_rescale} "
            f"provisional_denominator={self._t5_tensor_replay_provisional_denominator:.6f} "
            f"use_provisional_denominator={self._t5_tensor_replay_use_provisional_denominator}",
            flush=True,
        )
        return {"t5_tensor_replay/begin": 1.0}

    @staticmethod
    def _synthesize_response_level_weights_(tensor_dict: dict[str, torch.Tensor]) -> None:
        # Streamed chunks carry per-row response-level COUNT scalars instead of
        # positional [N, L] weight tensors (alignment-proof across permutation/
        # slicing/shm). Expand them here from the chunk's own masks so all
        # downstream packing sees the usual weight tensors.
        ce_counts = tensor_dict.pop("response_level_ce_token_count", None)
        if ce_counts is not None and "response_level_ce_weight" not in tensor_dict:
            response_mask = tensor_dict["response_mask"].to(dtype=torch.float32)
            counts = ce_counts.view(-1, 1).to(
                dtype=torch.float32, device=response_mask.device
            )
            tensor_dict["response_level_ce_weight"] = torch.where(
                counts > 0,
                response_mask / counts.clamp(min=1.0),
                torch.zeros_like(response_mask),
            )
        pair_counts = tensor_dict.pop("response_level_aec_pair_count", None)
        if pair_counts is not None and "response_level_aec_pair_weight" not in tensor_dict:
            pair_mask = tensor_dict.get("adjacent_entropy_pair_mask")
            pair_ids = tensor_dict.get("adjacent_entropy_pair_ids")
            if pair_mask is not None and pair_ids is not None:
                valid = (pair_mask.bool() & pair_ids.ge(0)).to(dtype=torch.float32)
                counts = pair_counts.view(-1, 1).to(
                    dtype=torch.float32, device=valid.device
                )
                weight = torch.where(
                    counts > 0,
                    valid / counts.clamp(min=1.0),
                    torch.zeros_like(valid),
                )
                bad_rows = ((valid.sum(dim=-1) > 0) & (counts.view(-1) <= 0)).sum()
                nonfinite = (~torch.isfinite(weight)).sum()
                if int(bad_rows) or int(nonfinite):
                    print(
                        "[SDAR_STREAM_DIAG] aec_weight_synthesis_anomaly "
                        f"rows={int(valid.shape[0])} bad_rows={int(bad_rows)} "
                        f"nonfinite={int(nonfinite)} counts_min={float(counts.min())} "
                        f"counts_max={float(counts.max())}",
                        flush=True,
                    )
                tensor_dict["response_level_aec_pair_weight"] = weight

    def _consume_t5_tensor_chunk_impl(self, tensor_dict: dict[str, torch.Tensor], chunk_idx: int | None = None):
        assert self._is_actor
        self._synthesize_response_level_weights_(tensor_dict)
        actor_method_enter_ts = time.time()
        submit_wall_ts_tensor = tensor_dict.pop("_node_local_submit_wall_ts", None)
        actor_method_queue_seconds = 0.0
        if torch.is_tensor(submit_wall_ts_tensor) and int(submit_wall_ts_tensor.numel()) > 0:
            try:
                submit_wall_ts = float(submit_wall_ts_tensor.detach().cpu().reshape(-1)[0].item())
                if submit_wall_ts > 0.0:
                    actor_method_queue_seconds = max(0.0, actor_method_enter_ts - submit_wall_ts)
            except Exception:
                actor_method_queue_seconds = 0.0
        if not getattr(self, "_t5_tensor_replay_active", False):
            first_value = next(iter(tensor_dict.values()))
            self.begin_t5_tensor_accum(
                expected_rows=int(first_value.shape[0]) if torch.is_tensor(first_value) else 0,
                expected_active_tokens=0,
                meta_info={},
            )

        chunk_rows = int(tensor_dict["input_ids"].shape[0])
        real_rows_tensor = tensor_dict.get("_t5_tensor_real_rows")
        real_chunk_rows = chunk_rows
        if torch.is_tensor(real_rows_tensor):
            if int(real_rows_tensor.numel()) == 1:
                real_chunk_rows = int(real_rows_tensor.detach().cpu().reshape(-1)[0].item())
            elif int(real_rows_tensor.numel()) > 0:
                real_chunk_rows = int(real_rows_tensor.detach().cpu().reshape(-1).sum().item())
            real_chunk_rows = max(real_chunk_rows, 0)
        meta_info = dict(self._t5_tensor_replay_meta_info)
        meta_info.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        meta_info["_t5_tensor_chunk_idx"] = int(chunk_idx if chunk_idx is not None else self._t5_tensor_replay_chunks)
        out_of_order_chunk = False
        if chunk_idx is not None:
            previous_chunk_idx = int(getattr(self, "_t5_tensor_replay_last_chunk_idx", -1))
            current_chunk_idx = int(chunk_idx)
            seen_chunk_indices = getattr(self, "_t5_tensor_replay_seen_chunk_indices", set())
            if current_chunk_idx in seen_chunk_indices:
                raise RuntimeError(
                    "t5 tensor replay chunk_idx must be unique within an update window: "
                    f"current_chunk_idx={current_chunk_idx} previous_chunk_idx={previous_chunk_idx}"
                )
            seen_chunk_indices.add(current_chunk_idx)
            self._t5_tensor_replay_seen_chunk_indices = seen_chunk_indices
            out_of_order_chunk = previous_chunk_idx >= 0 and current_chunk_idx < previous_chunk_idx

        print(
            "[SplitStreaming] actor_consume_start "
            f"ts={time.time():.6f} rank={self.rank} chunk_idx={meta_info['_t5_tensor_chunk_idx']} "
            f"rows={real_chunk_rows} padded_rows={chunk_rows} "
            f"actor_method_queue_s={actor_method_queue_seconds:.6f} "
            f"out_of_order={int(out_of_order_chunk)}",
            flush=True,
        )
        try:
            with Timer(name="t5_tensor_replay_consume", logger=None) as timer:
                metrics = self.actor.update_t5_tensor_chunk(tensor_batch=tensor_dict, meta_info=meta_info)
        except Exception:
            self.actor.invalidate_gemma_prefix_kv_cache()
            raise
        append_to_dict(self._t5_tensor_replay_metrics, metrics)
        self._t5_tensor_replay_chunks += 1
        self._t5_tensor_replay_rows += real_chunk_rows
        active_tokens = float(metrics.get("actor/t5_tensor_replay/active_tokens", 0.0) or 0.0)
        loss_value = float(metrics.get("actor/t5_tensor_replay/loss", 0.0) or 0.0)
        self._t5_tensor_replay_active_tokens += active_tokens
        self._t5_tensor_replay_loss_weighted_sum += loss_value * active_tokens
        if chunk_idx is not None:
            self._t5_tensor_replay_last_chunk_idx = max(
                int(getattr(self, "_t5_tensor_replay_last_chunk_idx", -1)),
                int(chunk_idx),
            )
        print(
            "[SplitStreaming] actor_consume_done "
            f"ts={time.time():.6f} rank={self.rank} chunk_idx={meta_info['_t5_tensor_chunk_idx']} "
            f"rows={real_chunk_rows} padded_rows={chunk_rows} seconds={timer.last:.6f}",
            flush=True,
        )
        return_metrics = {
            key: value
            for key, value in metrics.items()
            if str(key).startswith("actor/t5_tensor_replay/")
        }
        return_metrics.update(
            {
                "t5_tensor_replay/chunks": float(self._t5_tensor_replay_chunks),
                "t5_tensor_replay/latest_rows": float(real_chunk_rows),
                "t5_tensor_replay/latest_padded_rows": float(chunk_rows),
                "t5_tensor_replay/consume_seconds": timer.last,
                "actor/t5_tensor_replay/actor_method_queue_seconds": float(actor_method_queue_seconds),
                "actor/t5_tensor_replay/out_of_order_chunk": float(out_of_order_chunk),
            }
        )
        return return_metrics

    @staticmethod
    def _is_t5_tensor_chunk_shm_payload(tensor_dict: dict) -> bool:
        return (
            isinstance(tensor_dict, dict)
            and tensor_dict.get("format") == "split_streaming_t5_tensor_chunk_shm_v1"
        )

    @staticmethod
    def _t5_tensor_chunk_from_shm_handle(handle: dict) -> torch.Tensor:
        dtype_map = {
            "torch.bool": torch.bool,
            "torch.uint8": torch.uint8,
            "torch.int8": torch.int8,
            "torch.int16": torch.int16,
            "torch.int32": torch.int32,
            "torch.int64": torch.int64,
            "torch.float16": torch.float16,
            "torch.bfloat16": torch.bfloat16,
            "torch.float32": torch.float32,
            "torch.float64": torch.float64,
        }
        dtype = dtype_map[str(handle["dtype"])]
        shape = tuple(int(dim) for dim in handle["shape"])
        nbytes = int(handle["nbytes"])
        numel = int(np.prod(shape)) if shape else 1
        shm = SharedMemory(name=str(handle["name"]))
        try:
            tensor = torch.empty(numel, dtype=dtype)
            if nbytes > 0:
                raw_np = np.ndarray((nbytes,), dtype=np.uint8, buffer=shm.buf)
                raw_t = torch.from_numpy(raw_np)
                tensor.view(torch.uint8).reshape(-1).copy_(raw_t)
            return tensor.view(shape)
        finally:
            shm.close()

    @classmethod
    def _materialize_t5_tensor_chunk_payload(cls, tensor_dict: dict) -> tuple[dict, dict[str, float]]:
        if not cls._is_t5_tensor_chunk_shm_payload(tensor_dict):
            return tensor_dict, {}

        materialize_start = time.perf_counter()
        materialized = {}
        non_tensors = tensor_dict.get("non_tensors", {})
        if isinstance(non_tensors, dict):
            materialized.update(non_tensors)

        tensors = tensor_dict.get("tensors", {})
        if not isinstance(tensors, dict):
            raise RuntimeError("split streaming tensor chunk SHM payload has invalid tensors field")

        total_bytes = 0
        for key, handle in tensors.items():
            if not isinstance(handle, dict):
                raise RuntimeError(f"split streaming tensor chunk SHM handle for {key!r} is invalid")
            tensor_key = str(handle.get("key", key))
            materialized[tensor_key] = cls._t5_tensor_chunk_from_shm_handle(handle)
            total_bytes += int(handle.get("nbytes", 0) or 0)

        materialize_seconds = time.perf_counter() - materialize_start
        return materialized, {
            "t5_tensor_replay/shm_materialize_chunks": 1.0,
            "t5_tensor_replay/shm_materialize_items": float(len(tensors)),
            "t5_tensor_replay/shm_materialize_bytes": float(total_bytes),
            "t5_tensor_replay/shm_materialize_seconds": materialize_seconds,
        }

    def _consume_t5_tensor_chunk_payload(self, tensor_dict: dict, chunk_idx: int | None = None):
        tensor_dict, materialize_metrics = self._materialize_t5_tensor_chunk_payload(tensor_dict)
        metrics = self._consume_t5_tensor_chunk_impl(tensor_dict, chunk_idx=chunk_idx)
        if materialize_metrics:
            metrics.update(materialize_metrics)
        return metrics

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def consume_t5_tensor_chunk(self, tensor_dict: dict[str, torch.Tensor], chunk_idx: int | None = None):
        return self._consume_t5_tensor_chunk_payload(tensor_dict, chunk_idx=chunk_idx)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def consume_t5_tensor_chunk_async(self, tensor_dict: dict[str, torch.Tensor], chunk_idx: int | None = None):
        return self._consume_t5_tensor_chunk_payload(tensor_dict, chunk_idx=chunk_idx)

    @register(dispatch_mode=Dispatch.DP_COMPUTE)
    def consume_t5_tensor_chunk_dp(self, tensor_dict: dict[str, torch.Tensor], chunk_idx: int | None = None):
        return self._consume_t5_tensor_chunk_payload(tensor_dict, chunk_idx=chunk_idx)

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def consume_t5_tensor_chunk_dp_async(self, tensor_dict: dict[str, torch.Tensor], chunk_idx: int | None = None):
        return self._consume_t5_tensor_chunk_payload(tensor_dict, chunk_idx=chunk_idx)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def abort_t5_tensor_accum(self, reason: str = ""):
        assert self._is_actor
        active = bool(getattr(self, "_t5_tensor_replay_active", False))
        if not active:
            return {
                "t5_tensor_replay/abort": 0.0,
                "t5_tensor_replay/abort_active": 0.0,
            }

        started_at = float(getattr(self, "_t5_tensor_replay_started_at", 0.0) or 0.0)
        delta_time = time.perf_counter() - started_at if started_at > 0 else 0.0
        metrics = {
            "t5_tensor_replay/abort": 1.0,
            "t5_tensor_replay/abort_active": 1.0,
            "t5_tensor_replay/abort_chunks": float(getattr(self, "_t5_tensor_replay_chunks", 0) or 0),
            "t5_tensor_replay/abort_rows": float(getattr(self, "_t5_tensor_replay_rows", 0) or 0),
            "t5_tensor_replay/abort_active_tokens": float(
                getattr(self, "_t5_tensor_replay_active_tokens", 0.0) or 0.0
            ),
            "t5_tensor_replay/abort_seconds": float(delta_time),
        }
        print(
            "[SplitStreaming] actor_abort "
            f"ts={time.time():.6f} reason={reason!r} "
            f"chunks={metrics['t5_tensor_replay/abort_chunks']:.0f} "
            f"rows={metrics['t5_tensor_replay/abort_rows']:.0f} "
            f"seconds={delta_time:.6f}",
            flush=True,
        )
        try:
            self.actor.actor_optimizer.zero_grad()
        finally:
            self._cleanup_t5_tensor_replay_window("t5 tensor replay abort")
        return metrics

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def finish_t5_tensor_update(
        self,
        final_rows: int | None = None,
        final_active_tokens: float | None = None,
        global_mean_rescale: bool = False,
    ):
        assert self._is_actor
        if not getattr(self, "_t5_tensor_replay_active", False):
            return DataProto(meta_info={"metrics": {"t5_tensor_replay/chunks": 0.0}})

        metrics = self._t5_tensor_replay_metrics
        delta_time = time.perf_counter() - self._t5_tensor_replay_started_at
        try:
            print(
                "[SplitStreaming] actor_finish_start "
                f"ts={time.time():.6f} chunks={self._t5_tensor_replay_chunks} rows={self._t5_tensor_replay_rows}",
                flush=True,
            )
            unknown_total_grad_rescale = bool(self._t5_tensor_replay_unknown_total_grad_rescale)
            grad_rescale_denominator = 0.0
            grad_rescale_factor = 1.0
            grad_rescale_basis_active_tokens = 0.0
            grad_rescale_basis_rows = 0.0
            provisional_denominator = float(self._t5_tensor_replay_provisional_denominator)
            use_provisional_denominator = bool(self._t5_tensor_replay_use_provisional_denominator)
            if unknown_total_grad_rescale:
                grad_rescale_basis = getattr(self, "_t5_tensor_replay_grad_rescale_basis", "active_tokens")
                if global_mean_rescale:
                    if grad_rescale_basis == "rows":
                        grad_rescale_denominator = float(max(int(final_rows or self._t5_tensor_replay_rows), 1))
                        grad_rescale_basis_rows = 1.0
                    else:
                        final_active = float(
                            final_active_tokens
                            if final_active_tokens is not None
                            else self._t5_tensor_replay_active_tokens
                        )
                        if final_active > 0:
                            grad_rescale_denominator = final_active
                            grad_rescale_basis_active_tokens = 1.0
                        else:
                            grad_rescale_denominator = float(max(int(final_rows or self._t5_tensor_replay_rows), 1))
                            grad_rescale_basis_rows = 1.0
                    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
                    numerator = provisional_denominator if use_provisional_denominator else 1.0
                    grad_rescale_factor = float(world_size) * float(numerator) / max(grad_rescale_denominator, 1.0)
                elif grad_rescale_basis == "rows":
                    grad_rescale_denominator = float(max(int(self._t5_tensor_replay_rows), 1))
                    grad_rescale_basis_rows = 1.0
                elif self._t5_tensor_replay_active_tokens > 0:
                    grad_rescale_denominator = float(self._t5_tensor_replay_active_tokens)
                    grad_rescale_basis_active_tokens = 1.0
                else:
                    grad_rescale_denominator = float(max(int(self._t5_tensor_replay_rows), 1))
                    grad_rescale_basis_rows = 1.0
                if not global_mean_rescale:
                    if use_provisional_denominator:
                        grad_rescale_factor = provisional_denominator / max(grad_rescale_denominator, 1.0)
                    else:
                        grad_rescale_factor = 1.0 / max(grad_rescale_denominator, 1.0)
            fp32_grad_accum_active = bool(getattr(self.actor, "_gemma_replay_fp32_grad_accum_active", False))
            fp32_grad_accum_metrics = (
                self.actor.materialize_gemma_replay_fp32_grad_accum()
                if fp32_grad_accum_active
                else {}
            )
            grad_norm = self.actor._optimizer_step(grad_scale_factor=grad_rescale_factor)
            sc_update_metrics = dict(getattr(self.actor, "_sc_update_metrics", {}) or {})
            self.actor._sc_update_metrics = {}
            fp32_grad_accum_cleanup_metrics = (
                self.actor._cleanup_gemma_replay_fp32_grad_accum()
                if fp32_grad_accum_active
                else {}
            )
            if fp32_grad_accum_metrics and fp32_grad_accum_cleanup_metrics:
                fp32_grad_accum_metrics[
                    "actor/diffusion_gemma/fp32_grad_accum/cleanup_count"
                ] = fp32_grad_accum_cleanup_metrics.get(
                    "actor/diffusion_gemma/fp32_grad_accum/cleanup_count",
                    fp32_grad_accum_metrics.get("actor/diffusion_gemma/fp32_grad_accum/cleanup_count", 0.0),
                )
                fp32_grad_accum_metrics[
                    "actor/diffusion_gemma/fp32_grad_accum/cleanup_seconds"
                ] = fp32_grad_accum_cleanup_metrics.get(
                    "actor/diffusion_gemma/fp32_grad_accum/cleanup_seconds",
                    fp32_grad_accum_metrics.get("actor/diffusion_gemma/fp32_grad_accum/cleanup_seconds", 0.0),
                )
            total_grad_values = getattr(self.actor, "_last_total_grad_values", 0)
            nonfinite_grad_values = getattr(self.actor, "_last_nonfinite_grad_values", 0)
            if sc_update_metrics:
                append_to_dict(metrics, sc_update_metrics)
            append_to_dict(
                metrics,
                {
                    "actor/grad_norm": grad_norm.detach().item(),
                    "actor/optimizer_steps": float(
                        getattr(
                            self.actor,
                            "_last_optimizer_step_applied",
                            False,
                        )
                    ),
                    "actor/optimizer_step_count": float(
                        getattr(
                            self.actor,
                            "_lifecycle_optimizer_step_count",
                            0,
                        )
                    ),
                    "actor/replica_grad_sync_tensors": float(
                        getattr(self.actor, "_last_replica_grad_sync_tensors", 0)
                    ),
                    "actor/replica_grad_sync_bytes": float(
                        getattr(self.actor, "_last_replica_grad_sync_bytes", 0)
                    ),
                    "actor/replica_grad_sync_seconds": float(
                        getattr(self.actor, "_last_replica_grad_sync_seconds", 0.0)
                    ),
                    "actor/t5_tensor_replay/finish_grad_rescale_factor": float(grad_rescale_factor),
                    "actor/t5_tensor_replay/finish_grad_rescale_denominator": float(grad_rescale_denominator),
                    "actor/t5_tensor_replay/finish_grad_rescale_unknown_total": (
                        1.0 if unknown_total_grad_rescale else 0.0
                    ),
                    "actor/t5_tensor_replay/finish_provisional_denominator": float(provisional_denominator),
                    "actor/t5_tensor_replay/finish_use_provisional_denominator": (
                        1.0 if use_provisional_denominator else 0.0
                    ),
                    "actor/t5_tensor_replay/finish_grad_rescale_basis_active_tokens": float(
                        grad_rescale_basis_active_tokens
                    ),
                    "actor/t5_tensor_replay/finish_grad_rescale_basis_rows": float(grad_rescale_basis_rows),
                    "actor/t5_tensor_replay/finish_global_mean_rescale": (
                        1.0 if global_mean_rescale else 0.0
                    ),
                    "actor/t5_tensor_replay/finish_final_rows": float(
                        final_rows if final_rows is not None else self._t5_tensor_replay_rows
                    ),
                    "actor/t5_tensor_replay/finish_final_active_tokens": float(
                        final_active_tokens
                        if final_active_tokens is not None
                        else self._t5_tensor_replay_active_tokens
                    ),
                    "actor/nonfinite_grad_tensors": float(getattr(self.actor, "_last_nonfinite_grad_tensors", 0)),
                    "actor/nonfinite_grad_values": float(nonfinite_grad_values),
                    "actor/nonfinite_grad_frac": (
                        float(nonfinite_grad_values) / float(total_grad_values) if total_grad_values else 0.0
                    ),
                },
            )
            if fp32_grad_accum_metrics:
                metrics.update(fp32_grad_accum_metrics)
            self.actor.actor_optimizer.zero_grad()

            global_num_tokens = self._t5_tensor_replay_meta_info.get("global_token_num", None)
            if global_num_tokens is not None:
                images_seqlens = self._t5_tensor_replay_meta_info.get("images_seqlens", None)
                estimated_flops, promised_flops = self.flops_counter.estimate_flops(
                    global_num_tokens, delta_time, images_seqlens=images_seqlens
                )
                metrics["perf/mfu/actor"] = (
                    estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
                )
            metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
            metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr.item() if torch.is_tensor(lr) else lr
            self.actor_lr_scheduler.step()
            metrics["t5_tensor_replay/chunks"] = float(self._t5_tensor_replay_chunks)
            metrics["t5_tensor_replay/rows"] = float(self._t5_tensor_replay_rows)
            metrics["t5_tensor_replay/expected_rows"] = float(self._t5_tensor_replay_expected_rows)
            metrics["t5_tensor_replay/active_tokens"] = float(self._t5_tensor_replay_active_tokens)
            metrics["t5_tensor_replay/expected_active_tokens"] = float(
                self._t5_tensor_replay_expected_active_tokens
            )
            metrics["t5_tensor_replay/unknown_total_grad_rescale"] = (
                1.0 if unknown_total_grad_rescale else 0.0
            )
            metrics["t5_tensor_replay/normalize_at_finish"] = (
                1.0 if self._t5_tensor_replay_normalize_at_finish else 0.0
            )
            metrics["t5_tensor_replay/grad_rescale_denominator"] = float(grad_rescale_denominator)
            metrics["t5_tensor_replay/grad_rescale_factor"] = float(grad_rescale_factor)
            metrics["t5_tensor_replay/provisional_denominator"] = float(provisional_denominator)
            metrics["t5_tensor_replay/use_provisional_denominator"] = (
                1.0 if use_provisional_denominator else 0.0
            )
            metrics["t5_tensor_replay/grad_rescale_basis_active_tokens"] = float(
                grad_rescale_basis_active_tokens
            )
            metrics["t5_tensor_replay/grad_rescale_basis_rows"] = float(grad_rescale_basis_rows)
            metrics["t5_tensor_replay/global_mean_rescale"] = 1.0 if global_mean_rescale else 0.0
            metrics["t5_tensor_replay/final_rows"] = float(
                final_rows if final_rows is not None else self._t5_tensor_replay_rows
            )
            metrics["t5_tensor_replay/final_active_tokens"] = float(
                final_active_tokens
                if final_active_tokens is not None
                else self._t5_tensor_replay_active_tokens
            )
            if self._t5_tensor_replay_active_tokens > 0:
                full_batch_loss = float(self._t5_tensor_replay_loss_weighted_sum) / float(
                    self._t5_tensor_replay_active_tokens
                )
                metrics["actor/t5_tensor_replay/full_batch_loss"] = full_batch_loss
                metrics["generation_consistency_jsd/loss"] = full_batch_loss
            metrics["generation_consistency_jsd/active_tokens"] = float(self._t5_tensor_replay_active_tokens)
            metrics["t5_tensor_replay/update_seconds"] = delta_time

            output = DataProto(meta_info={"metrics": metrics})
            output = output.to("cpu")
            print(
                "[SplitStreaming] actor_finish_done "
                f"ts={time.time():.6f} chunks={self._t5_tensor_replay_chunks} "
                f"rows={self._t5_tensor_replay_rows} "
                f"grad_rescale_denominator={grad_rescale_denominator:.6f} "
                f"grad_rescale_factor={grad_rescale_factor:.6f} "
                f"provisional_denominator={provisional_denominator:.6f} "
                f"seconds={delta_time:.6f}",
                flush=True,
            )
            return output
        finally:
            self._cleanup_t5_tensor_replay_window("t5 tensor replay update")

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    @DistProfiler.annotate(color="red", role="rollout_generate")
    def generate_sequences(self, prompts: DataProto):
        # Support all hardwares
        assert self._is_rollout
        prompts = prompts.to(get_device_id())

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)

        timing_generate = {}
        if self._is_actor:  # For rollout only, we do not switch context.
            loop = get_event_loop()
            loop.run_until_complete(self.rollout_mode())
            if shared_weights_enabled() and not self._shared_weight_policy_valid:
                raise RuntimeError(
                    "rollout generation rejected because the last shared-weight "
                    "hot-sync did not produce a complete active policy"
                )
            log_gpu_memory_usage("After switch to rollout mode", logger=logger)

        with simple_timer("generate_sequences", timing_generate):
            output = self.rollout.generate_sequences(prompts=prompts)

        if self._is_actor:
            loop.run_until_complete(self.trainer_mode())
            log_gpu_memory_usage("After switch to trainer mode", logger=logger)

        # We calculate the average timing across all ranks
        # to make sure meta_info["timing"] is the same
        timing_generate_topk_ratio, timing_generate_min, timing_generate_max = topk_reduce_ratio_min_max(
            timing_generate["generate_sequences"]
        )
        timing_generate = reduce_timing(timing_generate)
        timing_generate.update(
            {
                "generation_timing/max": timing_generate_max,
                "generation_timing/min": timing_generate_min,
                "generation_timing/topk_ratio": timing_generate_topk_ratio,
            }
        )
        output.meta_info["timing"] = timing_generate
        output = output.to("cpu")

        # clear kv cache
        get_torch_device().empty_cache()
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="blue", role="actor_compute_log_prob")
    def compute_log_prob(self, data: DataProto):
        # when is_lora is True, we use the actor without lora applied to calculate the log_prob
        # which is mostly used for ref log_prob calculation
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        # Support all hardwares
        from contextlib import nullcontext

        is_lora = data.meta_info.pop("is_lora", False)
        adapter_ctx = self.actor.actor_module.disable_adapter() if is_lora else nullcontext()
        # we should always recompute old_log_probs when it is HybridEngine
        config_source = self.config.ref if is_lora else self.config.rollout
        data.meta_info["micro_batch_size"] = config_source.log_prob_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = config_source.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = config_source.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature
        data.meta_info.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        # perform recompute log_prob
        calculate_entropy = (not is_lora) and self.config.actor.get("compute_old_log_prob_entropy", True)
        with self.ulysses_sharding_manager:
            with adapter_ctx:
                outputs = self.actor.compute_log_prob(data=data, calculate_entropy=calculate_entropy)
            if not is_lora:
                tensors = {"old_log_probs": outputs["log_probs"]}
            else:
                tensors = {"ref_log_prob": outputs["log_probs"]}
            if calculate_entropy:
                tensors["entropys"] = outputs["entropys"]
            if "sum_pi_squared" in outputs:
                tensors["sum_pi_squared"] = outputs["sum_pi_squared"]
            output = DataProto.from_dict(
                tensors=tensors,
                meta_info={"temperature": self.config.rollout.temperature},
            )

        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
            self.actor.actor_module._handle.reshard(True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during compute_log_prob", logger=logger)

        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="blue", role="actor_causal_ar_rollout_score")
    def compute_causal_ar_rollout_scores(self, data: DataProto):
        """Score sampled rollout tokens under no-residual causal AR teacher forcing."""
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = self.config.rollout.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature
        data.meta_info.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        with self.ulysses_sharding_manager:
            outputs = self.actor.compute_causal_ar_rollout_scores(data=data)
            output = DataProto.from_dict(
                tensors={
                    "ar_token_log_probs": outputs["ar_token_log_probs"],
                    "ar_token_probs": outputs["ar_token_probs"],
                    "ar_avg_prob": outputs["ar_avg_prob"],
                },
                meta_info={"temperature": self.config.rollout.temperature},
            )

        output = output.to("cpu")

        if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
            self.actor.actor_module._handle.reshard(True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during causal AR rollout score", logger=logger)

        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="olive", role="ref_compute_log_prob")
    def compute_ref_log_prob(self, data: DataProto):
        if self._is_lora:
            # if _is_lora, actor without lora applied is the ref
            data.meta_info["is_lora"] = True
            return self.compute_log_prob(data)
        assert self._is_ref
        # else:
        # otherwise, the class have a standalone ref model

        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["temperature"] = self.config.rollout.temperature
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz
        data.meta_info.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        with self.ulysses_sharding_manager:
            data = data.to("cpu")  # data will to device with each micro batch on ref.compute_log_prob
            outputs = self.ref_policy.compute_log_prob(data=data, calculate_entropy=False)
            output = DataProto.from_dict(tensors={"ref_log_prob": outputs["log_probs"]})

        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1:
            if fsdp_version(self.ref_policy.actor_module) == 1:
                self.ref_policy.actor_module._handle.reshard(True)
            elif fsdp_version(self.ref_policy.actor_module) == 2:
                self.ref_policy.actor_module.reshard()

        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        from verl.utils.logger import log_with_rank

        # only support save and load ckpt for actor
        assert self._is_actor

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        self.checkpoint_manager.save_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
        )
        dist.barrier()

        if self._is_lora and hasattr(getattr(self, "actor_module", self.actor_module_fsdp), "peft_config"):
            lora_save_path = os.path.join(local_path, "lora_adapter")
            peft_model = getattr(self, "actor_module", self.actor_module_fsdp)
            peft_config = {}
            if dist.get_rank() == 0:
                os.makedirs(lora_save_path, exist_ok=True)
                peft_config = asdict(peft_model.peft_config.get("default", {}))
                peft_config["task_type"] = peft_config["task_type"].value
                peft_config["peft_type"] = peft_config["peft_type"].value
                peft_config["target_modules"] = list(peft_config["target_modules"])
            try:
                if fsdp_version(self.actor_module_fsdp) > 0:
                    self.actor_module_fsdp = self.actor_module_fsdp.to(get_device_name())
                    lora_params = layered_summon_lora_params(self.actor_module_fsdp)
                    if dist.get_rank() == 0:
                        save_file(lora_params, os.path.join(lora_save_path, "adapter_model.safetensors"))
                        with open(os.path.join(lora_save_path, "adapter_config.json"), "w", encoding="utf-8") as f:
                            json.dump(peft_config, f, ensure_ascii=False, indent=4)
            except Exception as e:
                log_with_rank(
                    f"Save LoRA Adapter Error ({e})", rank=dist.get_rank(), logger=logger, log_only_rank_0=True
                )

            dist.barrier()
            log_with_rank(
                f"[rank-{self.rank}]: Saved LoRA adapter to: {lora_save_path}",
                rank=dist.get_rank(),
                logger=logger,
                log_only_rank_0=True,
            )

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        assert self._is_actor or (not self._is_actor and self._is_rollout), (
            f"Checkpoint loading is only supported for Actor or standalone Rollout Workers, but got "
            f"{self._is_actor} and {self._is_rollout}"
        )
        invalidate_prefix_cache = getattr(
            getattr(self, "actor", None),
            "invalidate_gemma_prefix_kv_cache",
            None,
        )
        if self._is_actor and callable(invalidate_prefix_cache):
            invalidate_prefix_cache()

        # No checkpoint to load, just offload the model and optimizer to CPU
        if local_path is None:
            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            if self._is_offload_optimizer:
                offload_fsdp_optimizer(self.actor_optimizer)
            return

        self._lifecycle_checkpoint_load_count += 1
        emit_lifecycle(
            "ActorRolloutRefWorker",
            "checkpoint_load_start",
            actor_uuid=self._lifecycle_actor_uuid,
            ray_actor_id=self._lifecycle_ray_actor_id,
            rank=int(self.rank),
            checkpoint_load_count=self._lifecycle_checkpoint_load_count,
            local_path=str(local_path),
            memory=lifecycle_memory("before_checkpoint_load"),
        )
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        self.checkpoint_manager.load_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )

        resume_lr_override_raw = os.environ.get(
            "VERL_ACTOR_RESUME_LR_OVERRIDE",
            "",
        ).strip()
        if self._is_actor and resume_lr_override_raw:
            resume_lr_override = float(resume_lr_override_raw)
            if not np.isfinite(resume_lr_override) or resume_lr_override <= 0:
                raise ValueError(
                    "VERL_ACTOR_RESUME_LR_OVERRIDE must be finite and positive, "
                    f"got {resume_lr_override_raw!r}"
                )
            for param_group in self.actor_optimizer.param_groups:
                param_group["lr"] = resume_lr_override
                if "initial_lr" in param_group:
                    param_group["initial_lr"] = resume_lr_override
            if self.actor_lr_scheduler is not None:
                if hasattr(self.actor_lr_scheduler, "base_lrs"):
                    self.actor_lr_scheduler.base_lrs = [
                        resume_lr_override
                        for _ in self.actor_lr_scheduler.base_lrs
                    ]
                if hasattr(self.actor_lr_scheduler, "_last_lr"):
                    self.actor_lr_scheduler._last_lr = [
                        resume_lr_override
                        for _ in self.actor_optimizer.param_groups
                    ]
            if self.rank == 0:
                print(
                    "VERL_ACTOR_RESUME_LR_OVERRIDE_APPLIED "
                    f"lr={resume_lr_override}"
                )

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.actor_optimizer)
        emit_lifecycle(
            "ActorRolloutRefWorker",
            "checkpoint_load_done",
            actor_uuid=self._lifecycle_actor_uuid,
            ray_actor_id=self._lifecycle_ray_actor_id,
            rank=int(self.rank),
            checkpoint_load_count=self._lifecycle_checkpoint_load_count,
            local_path=str(local_path),
            memory=lifecycle_memory("after_checkpoint_load"),
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def start_profile(self, **kwargs) -> None:
        """Start profiling for the current rank in the current training step."""
        self.profiler.start(**kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def stop_profile(self) -> None:
        """Stop profiling for the current rank in the current training step."""
        self.profiler.stop()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def dump_memory_snapshot(self, tag: str = "manual", sub_dir: str = None) -> None:
        """Manually trigger a CUDA memory snapshot dump on all ranks."""
        # Memory snapshot is now handled by the profiler system
        # This method is kept for backward compatibility but delegates to profiler
        if hasattr(self, "profiler") and hasattr(self.profiler, "_impl"):
            try:
                # Try to use the profiler's memory snapshot functionality
                if hasattr(self.profiler._impl, "sampler"):
                    out_dir = OmegaConf.select(self.config, "actor.profiler.save_path") or "."
                    self.profiler._impl.sampler.dump_memory_snapshot(out_dir=out_dir, tag=tag, sub_dir=sub_dir)
            except Exception:
                # silently ignore if profiler doesn't support memory snapshots
                pass


class CriticWorker(Worker, DistProfilerExtension):
    def __init__(self, config: FSDPCriticConfig):
        Worker.__init__(self)
        omega_profiler_config = config.get("profiler", {})
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )
        import torch.distributed

        self.config = config
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=get_nccl_backend(),
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )
        self.config: FSDPCriticConfig = config

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
            )

        # create training dispatch
        if self.ulysses_device_mesh is not None:
            is_collect = self.ulysses_device_mesh["sp"].get_local_rank() == 0
            self._register_dispatch_collect_info(
                "critic", dp_rank=self.ulysses_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )
        else:
            self._register_dispatch_collect_info("critic", dp_rank=self.rank, is_collect=True)

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # set FSDP offload params
        self._is_offload_param = self.config.model.fsdp_config.param_offload
        self._is_offload_optimizer = self.config.model.fsdp_config.optimizer_offload

        # normalize config
        self.config.ppo_mini_batch_size *= self.config.rollout_n
        self.config.ppo_mini_batch_size //= torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
        if self.config.ppo_micro_batch_size is not None:
            self.config.ppo_micro_batch_size //= (
                torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
            )
            self.config.forward_micro_batch_size //= (
                torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
            )
            self.config.ppo_micro_batch_size_per_gpu = self.config.ppo_micro_batch_size
            self.config.forward_micro_batch_size_per_gpu = self.config.forward_micro_batch_size

        if self.config.ppo_micro_batch_size_per_gpu is not None:
            assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size_per_gpu == 0, (
                f"normalized ppo_mini_batch_size {self.config.ppo_mini_batch_size} should be divisible by "
                f"ppo_micro_batch_size_per_gpu {self.config.ppo_micro_batch_size_per_gpu}"
            )
            assert self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu > 0, (
                f"normalized ppo_mini_batch_size {self.config.ppo_mini_batch_size} should be larger than "
                f"ppo_micro_batch_size_per_gpu {self.config.ppo_micro_batch_size_per_gpu}"
            )
        self._is_lora = (
            self.config.model.get("lora_adapter_path") is not None or self.config.model.get("lora_rank", 0) > 0
        )
        self.use_orig_params = self.config.model.fsdp_config.get("use_orig_params", False)

    def _build_critic_model_optimizer(self, config: FSDPCriticConfig):
        # the following line is necessary
        from torch.distributed.fsdp import MixedPrecision

        from verl.utils.model import load_valuehead_model, print_model_size
        from verl.utils.torch_dtypes import PrecisionType

        use_shm = config.model.get("use_shm", False)
        local_path = copy_to_local(config.model.path, use_shm=use_shm)
        # note that the tokenizer between actor and critic may be different. So override tokenizer info with actor info
        # using random initialized model from any architecture. May not be the same as Actor.

        tokenizer_path = copy_to_local(config.model.tokenizer_path, use_shm=use_shm)
        self.tokenizer = hf_tokenizer(tokenizer_path, trust_remote_code=config.model.get("trust_remote_code", False))
        self.processor = hf_processor(tokenizer_path, trust_remote_code=config.model.get("trust_remote_code", False))

        if self.config.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.model.custom_chat_template
            else:
                self.tokenizer.chat_template = self.config.model.custom_chat_template
        override_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_config)
        if self.rank == 0:
            print(f"Critic overriding config {override_config_kwargs}")

        torch_dtype = self.config.model.fsdp_config.get("model_dtype", "fp32")
        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        from transformers import AutoConfig

        # override model kwargs
        attn_implementation = override_config.get("attn_implementation", "flash_attention_2")
        critic_model_config = AutoConfig.from_pretrained(
            local_path,
            attn_implementation=attn_implementation,
            trust_remote_code=config.model.get("trust_remote_code", False),
        )
        # TODO: VL models use VisionAttention, which directly uses flash_attention in transformers>=4.53
        # which will be patched by _ulysses_flash_attention_forward, but errorly misses position_ids
        # Maybe support Ulysses in VisionAttention in the future and remove this patch
        if self.ulysses_sequence_parallel_size > 1 and hasattr(critic_model_config, "vision_config"):
            critic_model_config.vision_config._attn_implementation = "eager"

        critic_model_config.num_labels = 1
        # patch for kimi-vl
        if getattr(critic_model_config, "model_type", None) == "kimi_vl":
            critic_model_config.text_config.topk_method = "greedy"

        init_context = get_init_weight_context_manager(
            use_meta_tensor=not critic_model_config.tie_word_embeddings, mesh=self.device_mesh
        )

        # TiledMLP configuration for memory-efficient MLP computation
        tiled_mlp_config = config.model.get("tiled_mlp", {})
        use_tiled_mlp = tiled_mlp_config.get("enabled", False)
        tiled_mlp_shards = tiled_mlp_config.get("num_shards", 4)

        # TiledMLP requires FSDP2 for correct gradient computation
        if use_tiled_mlp and config.strategy == "fsdp":
            raise ValueError("TiledMLP requires FSDP2. Set `critic.strategy=fsdp2`.")

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            critic_model_config.classifier_dropout = 0.0
            critic_model_config.hidden_dropout = "0"
            critic_model_config.summary_dropout_prob = 0.0

            critic_module = load_valuehead_model(
                local_path,
                torch_dtype,
                critic_model_config,
                config.model.get("trust_remote_code", False),
            )

            use_remove_padding = config.model.get("use_remove_padding", False)

            apply_monkey_patch(
                model=critic_module,
                use_remove_padding=use_remove_padding,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
                use_tiled_mlp=use_tiled_mlp,
                tiled_mlp_shards=tiled_mlp_shards,
            )

            # some parameters may not in torch_dtype
            critic_module.to(torch_dtype)

            if config.model.get("enable_gradient_checkpointing", False):
                critic_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        if self._is_lora:
            print("Applying LoRA to critic module")
            critic_module.enable_input_require_grads()

            # Check if we should load a pre-trained LoRA adapter
            lora_adapter_path = self.config.model.get("lora_adapter_path")
            if lora_adapter_path is not None:
                from peft import PeftModel

                print(f"Loading pre-trained LoRA adapter to critic from: {lora_adapter_path}")

                # Copy adapter to local if needed
                local_adapter_path = copy_to_local(lora_adapter_path, use_shm=self.config.model.get("use_shm", False))

                critic_module = PeftModel.from_pretrained(critic_module, local_adapter_path, is_trainable=True)
                peft_config = critic_module.peft_config["default"]
                # Ensure task_type is TaskType enum, not string
                # Use TOKEN_CLS for Critic since it's loaded as AutoModelForTokenClassification
                if isinstance(peft_config.task_type, str):
                    peft_config.task_type = TaskType.TOKEN_CLS

            else:
                # Convert config to regular Python types before creating PEFT model
                # Use TOKEN_CLS for Critic since it's loaded as AutoModelForTokenClassification
                lora_config = {
                    "task_type": TaskType.TOKEN_CLS,
                    "r": self.config.model.lora_rank,
                    "lora_alpha": self.config.model.lora_alpha,
                    "target_modules": convert_to_regular_types(self.config.model.target_modules),
                    "bias": "none",
                }
                critic_module = get_peft_model(critic_module, LoraConfig(**lora_config))

        if self.rank == 0:
            print_model_size(critic_module)

        self.critic_model_config = critic_model_config

        fsdp_config = self.config.model.fsdp_config
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(
            module=critic_module,
            config=self.config.model.fsdp_config.wrap_policy,
            is_lora=self._is_lora,
        )

        log_gpu_memory_usage("Before critic FSDP", logger=None)

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        self.use_orig_params = fsdp_config.get("use_orig_params", False)
        if self.config.model.get("freeze_vision_tower", False):
            vision_tower = get_vl_model_vision_tower(critic_module)
            if vision_tower is not None:
                vision_tower.requires_grad_(False)
                self.use_orig_params = True
                if self.rank == 0:
                    print("[critic model] Vision tower is set to not trainable.")
            else:
                if self.rank == 0:
                    print("[critic model] No vision tower found.")

        # Note: We force turn off CPUOffload for critic because it causes incorrect results when using grad accumulation
        if config.strategy == "fsdp":
            critic_module = FSDP(
                critic_module,
                param_init_fn=init_fn,
                use_orig_params=self.use_orig_params,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                forward_prefetch=self.config.model.fsdp_config.forward_prefetch,
                device_mesh=self.device_mesh,
                cpu_offload=None,
            )
        elif config.strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(
                param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True
            )
            offload_policy = None
            if fsdp_config.offload_policy:
                self._is_offload_param = False
                self._is_offload_optimizer = False
                offload_policy = CPUOffloadPolicy(pin_memory=True)

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": offload_policy,
                "reshard_after_forward": fsdp_config.reshard_after_forward,
                "shard_placement_fn": get_shard_placement_fn(fsdp_size=self.device_mesh.shape[-1]),
            }
            full_state = critic_module.state_dict()
            apply_fsdp2(critic_module, fsdp_kwargs, fsdp_config)
            fsdp2_load_full_state_dict(critic_module, full_state, fsdp_mesh, offload_policy)
        else:
            raise NotImplementedError(f"Unknown strategy {config.strategy}")

        if config.model.get("enable_activation_offload", False):
            enable_gradient_checkpointing = config.model.get("enable_gradient_checkpointing", False)
            enable_activation_offloading(critic_module, config.strategy, enable_gradient_checkpointing)

        log_gpu_memory_usage("After critic FSDP", logger=None)

        critic_optimizer = build_optimizer(critic_module.parameters(), config.optim)

        total_steps = config.optim.get("total_training_steps", 0)
        num_warmup_steps = int(config.optim.get("lr_warmup_steps", -1))

        lr_scheduler_type = config.optim.get("lr_scheduler_type", "constant")
        if num_warmup_steps < 0:
            num_warmup_steps_ratio = config.optim.get("lr_warmup_steps_ratio", 0.0)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        if self.rank == 0:
            print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

        from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

        if lr_scheduler_type == "constant":
            critic_lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=critic_optimizer, num_warmup_steps=num_warmup_steps
            )
        elif lr_scheduler_type == "cosine":
            min_lr_ratio = config.optim.get("min_lr_ratio", 0.0)
            num_cycles = config.optim.get("num_cycles", 0.5)
            critic_lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=critic_optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps,
                min_lr_ratio=min_lr_ratio,
                num_cycles=num_cycles,
            )
        else:
            raise NotImplementedError(f"LR scheduler type {lr_scheduler_type} is not supported")

        return critic_module, critic_optimizer, critic_lr_scheduler

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        from verl.workers.critic import DataParallelPPOCritic

        self.critic_module, self.critic_optimizer, self.critic_lr_scheduler = self._build_critic_model_optimizer(
            self.config
        )

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
            log_gpu_memory_usage("After offload critic model during init", logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)
            log_gpu_memory_usage("After offload critic optimizer during init", logger=logger)

        self.critic = DataParallelPPOCritic(
            config=self.config, critic_module=self.critic_module, critic_optimizer=self.critic_optimizer
        )

        self.flops_counter = FlopsCounter(self.critic_model_config)
        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.critic_module,
            optimizer=self.critic_optimizer,
            lr_scheduler=self.critic_lr_scheduler,
            processing_class=self.processor if self.processor is not None else self.tokenizer,
            checkpoint_config=self.config.checkpoint,
            trust_remote_code=self.config.model.get("trust_remote_code", False),
        )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="critic"))
    @DistProfiler.annotate(color="cyan", role="compute_values")
    def compute_values(self, data: DataProto):
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)
        micro_batch_size = self.config.forward_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["max_token_len"] = self.config.forward_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz
        # perform forward computation
        with self.ulysses_sharding_manager:
            data = data.to("cpu")  # data will to device with each micro batch on critic.compute_values
            values = self.critic.compute_values(data=data)
            output = DataProto.from_dict(tensors={"values": values})

        output = output.to("cpu")
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="critic"))
    @DistProfiler.annotate(color="pink", role="critic_update")
    def update_critic(self, data: DataProto):
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.critic_optimizer, device_id=get_device_id())

        # perform forward computation
        with self.ulysses_sharding_manager:
            data = data.to("cpu")  # data will to device with each micro batch on critic.update_critic
            with Timer(name="update_critic", logger=None) as timer:
                metrics = self.critic.update_critic(data=data)
            delta_time = timer.last

            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu/critic"] = estimated_flops * self.config.ppo_epochs / promised_flops / self.world_size

            lr = self.critic_lr_scheduler.get_last_lr()[0]
            metrics["critic/lr"] = lr
            self.critic_lr_scheduler.step()

            output = DataProto(batch=None, meta_info={"metrics": metrics})

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        import torch

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)

        self.checkpoint_manager.save_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
        )

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=True):
        import torch

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)

        self.checkpoint_manager.load_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.critic_optimizer)


# ================================= Async related workers =================================
class AsyncActorRolloutRefWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self):
        self._lifecycle_hot_update_count += 1
        start = time.perf_counter()
        emit_lifecycle(
            "AsyncActorRolloutRefWorker",
            "hot_update_start",
            actor_uuid=self._lifecycle_actor_uuid,
            ray_actor_id=self._lifecycle_ray_actor_id,
            rank=int(self.rank),
            actor_module_id=id(self.actor_module_fsdp),
            hot_update_count=self._lifecycle_hot_update_count,
            base_sync_done=bool(self.base_sync_done),
            model_init_count=self._lifecycle_model_init_count,
            checkpoint_load_count=self._lifecycle_checkpoint_load_count,
            memory=lifecycle_memory("hot_update_start"),
        )
        await self.rollout_mode()
        emit_lifecycle(
            "AsyncActorRolloutRefWorker",
            "hot_update_done",
            actor_uuid=self._lifecycle_actor_uuid,
            ray_actor_id=self._lifecycle_ray_actor_id,
            rank=int(self.rank),
            actor_module_id=id(self.actor_module_fsdp),
            hot_update_count=self._lifecycle_hot_update_count,
            base_sync_done=bool(self.base_sync_done),
            model_init_count=self._lifecycle_model_init_count,
            checkpoint_load_count=self._lifecycle_checkpoint_load_count,
            seconds=time.perf_counter() - start,
            memory=lifecycle_memory("hot_update_done"),
        )
        return True



_LIGHT_EMPTY_CACHE_FROM_CONFIG = False


def _rollout_mode_empty_cache() -> None:
    """The two cache flushes around the weight export in rollout_mode.

    `aggressive_empty_cache(force_sync=True)` runs gc.collect() + empty_cache +
    synchronize up to three times; on the LLaDA2 actor (1.5e4 params, Ray) the two
    calls cost ~4.3 s per training step (stack samples, rank 0). Set
    VERL_ROLLOUT_MODE_LIGHT_EMPTY_CACHE=1 to do a single empty_cache()+synchronize
    without gc; default keeps the old behaviour."""
    if os.environ.get("VERL_ROLLOUT_MODE_LIGHT_EMPTY_CACHE", "0") == "1" or _LIGHT_EMPTY_CACHE_FROM_CONFIG:
        device = get_torch_device()
        if device.is_available():
            device.empty_cache()
            device.synchronize()
        return
    aggressive_empty_cache(force_sync=True)


def _start_stack_sampler_if_configured(config, rank: int) -> None:
    """Poor man's profiler for nodes without ptrace (py-spy needs
    kernel.yama.ptrace_scope=0 or sudo, we have neither).

    `+actor_rollout_ref.model.override_config.stack_sampler_dir=<dir>` (optionally
    `+actor_rollout_ref.model.override_config.stack_sampler_interval_s=0.5`) starts a daemon thread in
    every worker that appends the main thread's Python stack to
    `<dir>/rank<r>.log` at the given interval. Each record is
    `### <unix_time>` followed by one `file:line function` per frame, innermost
    last. Off unless the key is set; zero cost otherwise."""
    # Read from the free-form model.override_config (the actor config is a strict
    # dataclass and rejects unknown keys). Also honoured: env VERL_STACK_SAMPLER_DIR.
    sampler_dir = os.environ.get("VERL_STACK_SAMPLER_DIR") or None
    interval_cfg = None
    try:
        oc = config.model.get("override_config", {}) or {}
        sampler_dir = sampler_dir or oc.get("stack_sampler_dir", None)
        interval_cfg = oc.get("stack_sampler_interval_s", None)
    except Exception:
        pass
    if not sampler_dir:
        return
    import sys
    import threading
    import traceback

    interval = float(interval_cfg or os.environ.get("VERL_STACK_SAMPLER_INTERVAL_S", 0.5) or 0.5)
    os.makedirs(sampler_dir, exist_ok=True)
    path = os.path.join(sampler_dir, f"rank{rank}.log")

    def _run():
        me = threading.get_ident()
        with open(path, "a", buffering=1) as fh:
            while True:
                time.sleep(interval)
                names = {t.ident: t.name for t in threading.enumerate()}
                now = time.time()
                # All threads: Ray async actors run coroutines (e.g. rollout_mode /
                # update_weights) on an event-loop thread, not on MainThread.
                for ident, frame in list(sys._current_frames().items()):
                    if ident == me:
                        continue
                    fh.write(f"### {now:.3f} thread={names.get(ident, ident)}\n")
                    for fr in traceback.extract_stack(frame):
                        fh.write(f"{fr.filename}:{fr.lineno} {fr.name}\n")

    threading.Thread(target=_run, name="verl-stack-sampler", daemon=True).start()
    print(f"[STACK-SAMPLER] rank {rank} -> {path} every {interval}s", flush=True)


def _repair_self_conditioning_init(module):
    """Re-run the self-conditioning init that meta-device loading silently skipped.

    `LLaDA2MoeForCausalLM.__init__` calls `_init_self_conditioning_weights()`, but
    transformers 5.8.1 materialises the four adapter keys -- absent from the
    llada2.1-mini base checkpoint -- at zero *after* `__init__` has run, so the
    intended `normal_(0, 1e-3)` on gate/up is lost. gate=up=down=0 is an exact fixed
    point of the gradient, so the adapter can never train out of it.

    Rank layout matters here. `fsdp_workers.py:535` wraps `from_pretrained` in
    `get_init_weight_context_manager(use_meta_tensor=True, mesh=...)`, which gives
    **rank 0 real CPU tensors and every other rank meta tensors**. FSDP then wraps
    with `sync_module_states=True` (fsdp_workers.py:738), broadcasting rank 0's
    weights to the rest. So:

      * a meta rank cannot be repaired and must not be judged -- it has no storage
        to read or write, and it will receive rank 0's repaired weights anyway;
      * rank 0 is the single source of truth, so repairing and gating there covers
        the whole group.

    The first version of this patch reported a meta tensor as 0.0, could not tell
    "not yet materialised" from "dead", and so refused to launch on all 8 ranks.

    Only gate/up decide whether the adapter is dead. down_proj at zero is intentional
    -- it makes step 0 bit-identical to stock LLaDA -- so testing down would reject
    every healthy fresh start.
    """
    import math
    import os

    sc = getattr(module, "self_conditioning", None)
    if sc is None:
        return  # no-residual arm; nothing to repair and nothing to gate

    tensors = {
        "gate": sc.gate_proj.weight,
        "up": sc.up_proj.weight,
        "down": sc.down_proj.weight,
    }
    meta = [n for n, t in tensors.items() if t is None or getattr(t, "is_meta", False)]
    if meta:
        # Not this rank's job. Say so explicitly rather than reading 0.0 off a
        # storage-less tensor and calling it dead.
        print(
            "[SC-INIT] rank holds meta tensors (%s); skipping -- rank 0 repairs and "
            "sync_module_states broadcasts." % ",".join(meta),
            flush=True,
        )
        return

    def _amax(t):
        return float(t.detach().float().abs().max())

    gate0, up0 = _amax(tensors["gate"]), _amax(tensors["up"])
    # 1e-12 sits far above the ~1e-32 garbage that to_empty leaves and far below
    # the ~1e-3 the intended normal init produces, so it separates the two cleanly.
    # The materialised-but-uninitialised storage is not always ~0: two
    # launches in a row came up with gate=nan/up=0 and gate=0/up=nan, and `nan < 1e-12`
    # is False, so the old test skipped the repair and the launch died on the
    # dead-adapter gate below. Anything non-finite is uninitialised memory and
    # must be repaired exactly like zeros.
    def _healthy(v):
        return math.isfinite(v) and v >= 1e-12

    dead = not (_healthy(gate0) and _healthy(up0))

    if dead:
        if hasattr(module, "_init_self_conditioning_weights"):
            module._init_self_conditioning_weights()
        else:  # adapter present but the helper is not: init inline, same recipe
            std = float(getattr(module.config, "self_conditioning_init_std", 1e-3))
            sc.gate_proj.weight.data.normal_(mean=0.0, std=std)
            sc.up_proj.weight.data.normal_(mean=0.0, std=std)
            if bool(getattr(module.config, "self_conditioning_zero_init", True)):
                sc.down_proj.weight.data.zero_()
            else:
                sc.down_proj.weight.data.normal_(mean=0.0, std=std)
            sc.pre_norm.weight.data.fill_(1.0)

    gate1, up1 = _amax(tensors["gate"]), _amax(tensors["up"])
    print(
        "[SC-INIT] real tensors on this rank. before: gate=%.3e up=%.3e (dead=%s) "
        "after: gate=%.3e up=%.3e down=%.3e" % (
            gate0, up0, dead, gate1, up1, _amax(tensors["down"])),
        flush=True,
    )

    if not (_healthy(gate1) and _healthy(up1)) and os.environ.get("SC_ALLOW_DEAD_ADAPTER") != "1":
        # Do not train. gate=up=0 means every adapter gradient is exactly zero;
        # the run would look completely healthy and learn no residual at all,
        # which is precisely how earlier runs were lost.
        raise RuntimeError(
            "self-conditioning adapter is at the dead fixed point after repair "
            "(gate=%.3e up=%.3e). Every adapter gradient is exactly zero, so the "
            "residual cannot train. Set SC_ALLOW_DEAD_ADAPTER=1 only if you "
            "deliberately want a residual arm with no residual." % (gate1, up1)
        )


def _sc_adapter_is_separately_wrapped(fsdp_config):
    """True when the self-conditioning adapter is its own FSDP unit.

    If it is, its FlatParameter is named
    `..._fsdp_wrapped_module.self_conditioning._fsdp_wrapped_module._flat_param`,
    which the SC_LR_MULT substring match finds even with use_orig_params=False.
    Forcing use_orig_params=True in that case buys nothing and costs the lm_head
    reshard that breaks ten reads of `lm_head.weight.shape[1]`.

    Measured across the three configurations before choosing this behavior.
    """
    try:
        wrap = (fsdp_config or {}).get("wrap_policy") or {}
        names = wrap.get("transformer_layer_cls_to_wrap") or []
        if isinstance(names, str):
            names = [names]
        return any("SelfConditioning" in str(n) for n in names)
    except Exception:
        return False
