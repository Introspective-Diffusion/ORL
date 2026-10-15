# Copyright 2023-2024 SGLang Team
# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import asyncio
import dataclasses
import inspect
import json
import logging
import math
import os
import time
import uuid
from importlib import import_module
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Optional

import numpy as np
import ray
import torch
from packaging import version
from ray.actor import ActorHandle

from verl.checkpoint_engine import CheckpointEngineWorker
from verl.single_controller.ray import RayClassWithInitArgs
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_visible_devices_keyword
from verl.utils.lifecycle import emit_lifecycle, lifecycle_memory
from verl.utils.net_utils import get_free_port, is_valid_ipv6_address
from verl.utils.profiler import DistProfiler, build_sglang_profiler_args
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.replica import RolloutMode, RolloutReplica, TokenOutput
from verl.workers.rollout.sglang_rollout.import_utils import (
    PATCHED_SGLANG_PATH_ENV,
    PATCHED_SGLANG_PATH_KEY,
    configure_sglang_import,
    get_patched_sglang_path,
)
from verl.workers.rollout.sglang_rollout.sglang_rollout import (
    SGLANG_REPLICA_RANK_ENV,
    ServerAdapter,
    _set_envs_and_config,
)
from verl.workers.rollout.utils import get_max_position_embeddings, run_unvicorn

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

visible_devices_keyword = get_visible_devices_keyword()
_SOURCE_MARKER_LOGGED = False
_SGLANG_BATCH_INPUT_IDS_ENV = os.getenv("VERL_SGLANG_BATCH_INPUT_IDS")
_SGLANG_MQ_SHM_UNLINK_PATCH_ATTR = "_verl_mq_shm_unlink_after_ready"
_SGLANG_SDAR_STREAM_REPLAY_TENSOR_STORE_ENV = "SGLANG_SDAR_STREAM_REPLAY_TENSOR_STORE"
_SGLANG_SDAR_SHM_TENSOR_STORE_VALUES = {"shm", "shared_memory", "1", "true"}
_SGLANG_SDAR_TABLE_TENSOR_STORE_VALUES = {"table", "memory_table", "handle"}
_SGLANG_SDAR_TPF_METADATA_FIRST_ENV = "SGLANG_SDAR_TPF_METADATA_FIRST"
_SGLANG_SERVER_CONSTRUCTION_COUNT = 0


def _compact_producer_state_selection_metadata(
    metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    if not any(
        bool(row.get("producer_state_selection", False))
        or str(row.get("producer_preselection_mode", "")).strip().lower()
        == "ranked"
        for row in metadata
    ):
        return {}
    bool_fields = {
        "producer_state_selection_rows": "producer_state_selection",
        "producer_state_selection_complete_rows": (
            "producer_state_selection_complete"
        ),
        "producer_state_selection_rank_selected_flags": (
            "producer_state_selection_rank_selected"
        ),
        "producer_state_selection_forced_final_flags": (
            "producer_state_selection_forced_final"
        ),
        "producer_zero_variance_groups": "producer_zero_variance",
    }
    int_fields = {
        "producer_state_selection_versions": (
            "producer_state_selection_version"
        ),
        "producer_source_rows": "producer_source_rows",
        "producer_rank_selected_rows": "producer_rank_selected_rows",
        "producer_selected_rows": "producer_selected_rows",
        "producer_final_rows": "producer_final_rows",
        "producer_forced_final_rows": "producer_forced_final_rows",
        "producer_final_already_selected_rows": (
            "producer_final_already_selected_rows"
        ),
        "producer_source_score_min": "producer_source_score_min",
        "producer_source_score_sum": "producer_source_score_sum",
        "producer_source_score_max": "producer_source_score_max",
        "producer_block_indices": "producer_block_index",
    }
    string_fields = {
        "producer_preselection_modes": "producer_preselection_mode",
        "producer_state_selection_modes": (
            "producer_state_selection_mode"
        ),
        "producer_state_selection_directions": (
            "producer_state_selection_direction"
        ),
        "producer_state_selection_scores": (
            "producer_state_selection_score"
        ),
        "producer_state_selection_group_fields": (
            "producer_state_selection_group_fields"
        ),
        "producer_state_selection_final_policies": (
            "producer_state_selection_final_policy"
        ),
    }
    compact: dict[str, Any] = {
        output_key: torch.tensor(
            [bool(row.get(metadata_key, False)) for row in metadata],
            dtype=torch.bool,
        )
        for output_key, metadata_key in bool_fields.items()
    }
    compact.update(
        {
            output_key: torch.tensor(
                [int(row.get(metadata_key, 0) or 0) for row in metadata],
                dtype=torch.long,
            )
            for output_key, metadata_key in int_fields.items()
        }
    )
    compact["producer_state_selection_fractions"] = torch.tensor(
        [
            float(row.get("producer_state_selection_fraction", 0.0) or 0.0)
            for row in metadata
        ],
        dtype=torch.float64,
    )
    compact.update(
        {
            output_key: [
                str(row.get(metadata_key, "") or "")
                for row in metadata
            ]
            for output_key, metadata_key in string_fields.items()
        }
    )
    return compact


def _import_sglang_module(module_name: str, config: Any = None):
    configure_sglang_import(config)
    return import_module(module_name)


def _server_args_field_names(server_args_cls: type) -> set[str]:
    return {field.name for field in dataclasses.fields(server_args_cls)}


def _derive_paired_server_seed(seed_base: int, replica_rank: int) -> int:
    """Give each rollout replica a stable, paired, but distinct server seed."""
    return int((int(seed_base) + int(replica_rank) * 1_000_003) % (2**31 - 1))


def _prepare_engine_kwargs(raw_engine_kwargs: dict[str, Any], server_args_cls: type) -> tuple[dict[str, Any], Optional[str]]:
    """Normalize optional SGLang kwargs and fail clearly for patched DLLM args."""
    engine_kwargs = dict(raw_engine_kwargs or {})
    engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
    engine_kwargs.pop(PATCHED_SGLANG_PATH_KEY, None)
    attention_backend = engine_kwargs.pop("attention_backend", None)

    dllm_keys = {"dllm_algorithm", "dllm_algorithm_config"}
    if dllm_keys.intersection(engine_kwargs):
        server_arg_fields = _server_args_field_names(server_args_cls)
        missing_fields = sorted(dllm_keys.intersection(engine_kwargs) - server_arg_fields)
        if missing_fields:
            raise RuntimeError(
                "SGLang DLLM rollout requested, but the imported sglang.srt.server_args.ServerArgs "
                f"does not support {missing_fields}. Configure actor_rollout_ref.rollout.engine_kwargs.sglang."
                f"{PATCHED_SGLANG_PATH_KEY}, or set {PATCHED_SGLANG_PATH_ENV} before launching verl."
            )

        config_path = engine_kwargs.get("dllm_algorithm_config")
        if config_path is not None and not os.path.exists(config_path):
            raise FileNotFoundError(f"SGLang DLLM algorithm config not found: {config_path}")

    return engine_kwargs, attention_backend


def _is_dllm_rollout(config: RolloutConfig) -> bool:
    """Return true for the opt-in patched SGLang DLLM route."""
    engine_kwargs = config.get("engine_kwargs", {}) or {}
    sglang_kwargs = engine_kwargs.get("sglang", {}) or {}
    return bool(sglang_kwargs.get("dllm_algorithm"))


def _memory_occupation_tags(config: RolloutConfig) -> list[str]:
    """Release graph memory too when CUDA graph is enabled."""
    tags = ["kv_cache", "weights"]
    if not config.enforce_eager:
        tags.append("cuda_graph")
    return tags


def _batch_input_ids_enabled(config: RolloutConfig) -> bool:
    """New DLLM runtimes expect token-id prompts through the batch-shaped field."""
    if _SGLANG_BATCH_INPUT_IDS_ENV is not None:
        return _SGLANG_BATCH_INPUT_IDS_ENV.lower() not in {"0", "false", "no"}
    return _is_dllm_rollout(config)


def _install_sglang_mq_shm_unlink_patch() -> None:
    """Unlink SGLang TP message-queue shm once all local readers opened it.

    SGLang's TP MessageQueue creates a named multiprocessing SharedMemory ring
    buffer in the scheduler writer process.  VERL may later stop SGLang
    subprocesses with SIGKILL, so relying on ShmRingBuffer.__del__ can leave a
    closed but still named /dev/shm/psm_* file.  POSIX shm stays usable through
    existing file descriptors after unlink, so unlinking after wait_until_ready()
    keeps the running queue intact while making hard-kill cleanup safe.
    """
    try:
        shm_broadcast = import_module("sglang.srt.distributed.device_communicators.shm_broadcast")
    except Exception as exc:
        logger.warning("Unable to install SGLang MQ shm unlink patch: %r", exc)
        return

    message_queue_cls = getattr(shm_broadcast, "MessageQueue", None)
    ring_buffer_cls = getattr(shm_broadcast, "ShmRingBuffer", None)
    if message_queue_cls is None or ring_buffer_cls is None:
        return
    if getattr(message_queue_cls, _SGLANG_MQ_SHM_UNLINK_PATCH_ATTR, False):
        return

    original_wait_until_ready = message_queue_cls.wait_until_ready

    def _unlink_creator_buffer(buffer: Any) -> None:
        if buffer is None or not getattr(buffer, "is_creator", False):
            return
        if getattr(buffer, "_verl_unlinked", False):
            return
        shared_mem = getattr(buffer, "shared_memory", None)
        if shared_mem is None:
            return
        try:
            shared_mem.unlink()
            buffer._verl_unlinked = True
            logger.info(
                "Unlinked SGLang MQ shm ring buffer name=%s size=%s",
                getattr(shared_mem, "name", "unknown"),
                getattr(shared_mem, "size", "unknown"),
            )
        except FileNotFoundError:
            buffer._verl_unlinked = True
        except Exception as exc:
            logger.warning(
                "Failed to unlink SGLang MQ shm ring buffer name=%s: %r",
                getattr(shared_mem, "name", "unknown"),
                exc,
            )

    def wait_until_ready_and_unlink(self, *args, **kwargs):
        result = original_wait_until_ready(self, *args, **kwargs)
        if getattr(self, "_is_writer", False):
            _unlink_creator_buffer(getattr(self, "buffer", None))
        return result

    def safe_ring_buffer_del(self):
        shared_mem = getattr(self, "shared_memory", None)
        if shared_mem is None:
            return
        try:
            shared_mem.close()
        except Exception:
            pass
        if getattr(self, "is_creator", False) and not getattr(self, "_verl_unlinked", False):
            try:
                shared_mem.unlink()
            except FileNotFoundError:
                pass

    message_queue_cls.wait_until_ready = wait_until_ready_and_unlink
    ring_buffer_cls.__del__ = safe_ring_buffer_del
    setattr(message_queue_cls, _SGLANG_MQ_SHM_UNLINK_PATCH_ATTR, True)


def _run_sglang_scheduler_process_with_mq_shm_unlink(*args, **kwargs):
    """Scheduler subprocess entrypoint with the SGLang MQ shm unlink patch."""
    configure_sglang_import()
    _install_sglang_mq_shm_unlink_patch()
    engine_mod = import_module("sglang.srt.entrypoints.engine")
    return engine_mod.run_scheduler_process(*args, **kwargs)


def _sum_dllm_stats(internal_states: list[dict[str, Any]] | None) -> dict[str, float]:
    totals: dict[str, float] = {}
    for state in internal_states or []:
        if not isinstance(state, dict):
            continue
        stats = state.get("dllm_stats") or {}
        if not isinstance(stats, dict):
            continue
        for key, value in stats.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                totals[key] = totals.get(key, 0.0) + float(value)
    return totals


def _diff_dllm_stats(after: dict[str, float], before: dict[str, float], block_size: int) -> dict[str, float]:
    keys = {
        "total_forwards",
        "denoise_forwards",
        "commit_forwards",
        "accepted_tokens",
        "new_accept_tokens",
        "edit_change_tokens",
        "total_tokens",
        "generated_tokens",
        "replay_captured_blocks",
        "replay_captured_rows",
        "replay_captured_token_steps",
        "replay_captured_bytes",
        "replay_producer_peak_bytes",
        "replay_source_rows",
        "replay_selected_rows",
        "replay_pruned_rows",
        "replay_peak_rows_per_request",
    }
    delta = {key: float(after.get(key, 0.0) - before.get(key, 0.0)) for key in keys}
    if not any(delta.values()):
        return {}
    denoise_forwards = max(delta.get("denoise_forwards", 0.0), 0.0)
    total_tokens = max(delta.get("total_tokens", 0.0), 0.0)
    new_accept_tokens = max(delta.get("new_accept_tokens", delta.get("accepted_tokens", 0.0)), 0.0)
    delta["block_size"] = float(block_size)
    delta["true_forward_tokens"] = float(block_size) * denoise_forwards
    if denoise_forwards > 0:
        delta["tokens_per_denoise_forward"] = total_tokens / denoise_forwards
        delta["new_accept_tokens_per_denoise_forward"] = new_accept_tokens / denoise_forwards
    return delta


class SGLangHttpServer:
    """SGLang http server in single node, this is equivalent to launch server with command line:
    ```
    python -m sglang.launch_server --node-rank 0 --nnode 1 ...
    ```

    Args:
        config (DictConfig): full config.
        rollout_mode (RolloutMode): rollout mode.
        replica_rank (int): replica rank, a replica may contain multiple nodes.
        node_rank (int): node rank.
        nnodes (int): number of nodes.
        cuda_visible_devices (str): cuda visible devices.
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        rollout_mode: RolloutMode,
        workers: list[ActorHandle],
        replica_rank: int,
        node_rank: int,
        nnodes: int,
        cuda_visible_devices: str,
        base_gpu_id: int,
    ):
        print(f"SGLang http server: {rollout_mode=}, {replica_rank=}, {node_rank=}, {nnodes=}, {cuda_visible_devices=}")
        os.environ[visible_devices_keyword] = cuda_visible_devices
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        global _SGLANG_SERVER_CONSTRUCTION_COUNT
        _SGLANG_SERVER_CONSTRUCTION_COUNT += 1
        self._lifecycle_engine_uuid = uuid.uuid4().hex
        self._lifecycle_construction_count = _SGLANG_SERVER_CONSTRUCTION_COUNT
        self._lifecycle_launch_count = 0
        self._lifecycle_sleep_count = 0
        self._lifecycle_wake_count = 0

        self.config: RolloutConfig = omega_conf_to_dataclass(config)
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config, dataclass_type=HFModelConfig)
        max_position_embeddings = get_max_position_embeddings(self.model_config.hf_config)
        if self.config.max_model_len is None:
            self.config.max_model_len = max_position_embeddings
        else:
            if self.config.max_model_len > max_position_embeddings:
                raise ValueError(
                    f"max_model_len ({self.config.max_model_len}) should be less than or equal to "
                    f"max_position_embeddings ({max_position_embeddings})"
                )
        self.rollout_mode = rollout_mode
        self.workers = workers

        self.replica_rank = replica_rank
        self.node_rank = node_rank
        self.nnodes = nnodes
        self.base_gpu_id = base_gpu_id
        emit_lifecycle(
            "SGLangHttpServer",
            "constructed",
            engine_uuid=self._lifecycle_engine_uuid,
            construction_count=self._lifecycle_construction_count,
            launch_count=self._lifecycle_launch_count,
            rollout_mode=str(self.rollout_mode),
            replica_rank=int(self.replica_rank),
            node_rank=int(self.node_rank),
            nnodes=int(self.nnodes),
            cuda_visible_devices=cuda_visible_devices,
            memory=lifecycle_memory("rollout_server_constructed"),
        )

        if self.rollout_mode != RolloutMode.HYBRID and self.config.load_format == "dummy":
            logger.warning(f"rollout mode is {self.rollout_mode}, load_format is dummy, set to auto")
            self.config.load_format = "auto"

        self._sglang = None
        self._sglang_engine = None
        self._http_server = None
        self._io_struct = None
        self._tokenizer_manager = None
        self._ServerArgs = None
        self._sdar_replay_tensor_store: dict[str, list[dict]] = {}
        self._gemma_replay_stream_metrics: dict[str, dict[str, Any]] = {}

        # used for http server
        self._server_address = ray.util.get_node_ip_address().strip("[]")
        self._server_port = None

        # used for controlling sglang server profiler
        profiler_config = self.config.profiler
        tool_config = None
        if profiler_config is not None:
            if profiler_config.tool in ["torch", "npu"]:
                tool_config = omega_conf_to_dataclass((profiler_config.tool_config or {}).get(profiler_config.tool))
            else:
                logger.warning(f"agent loop only support torch and npu profiler, got {profiler_config.tool}")
                profiler_config = None
        self.profiler_controller = DistProfiler(self.replica_rank, config=profiler_config, tool_config=tool_config)

        # used for NCCL process group
        if self.node_rank == 0:
            self._master_address = self._server_address
            self._master_port, self._master_sock = get_free_port(self._server_address)
            logger.info(
                f"SGLangHttpServer, replica_rank: {self.replica_rank}, "
                f"master address: {self._master_address}, port: {self._master_port}"
            )
        else:
            self._master_address = None
            self._master_port = None

    def _ensure_sglang_imports(self):
        if self._sglang is not None:
            return
        configure_sglang_import(self.config)
        self._sglang = _import_sglang_module("sglang", self.config)
        self._sglang_engine = _import_sglang_module("sglang.srt.entrypoints.engine", self.config)
        sglang_kwargs = (self.config.get("engine_kwargs", {}) or {}).get("sglang", {}) or {}
        if not str(sglang_kwargs.get("dllm_algorithm", "")).startswith("Gemma4"):
            self._sglang_engine._set_envs_and_config = _set_envs_and_config
        self._http_server = _import_sglang_module("sglang.srt.entrypoints.http_server", self.config)
        self._io_struct = _import_sglang_module("sglang.srt.managers.io_struct", self.config)
        self._tokenizer_manager = _import_sglang_module("sglang.srt.managers.tokenizer_manager", self.config)
        self._ServerArgs = self._http_server.ServerArgs

    def get_master_address(self):
        """Get master address and port for init NCCL process group."""
        return self._master_address, self._master_port

    def get_server_address(self):
        """Get http server address and port."""
        assert self._server_port is not None, "http server is not launched, port is None"
        return self._server_address, self._server_port

    async def launch_server(self, master_address: str = None, master_port: int = None):
        self._lifecycle_launch_count += 1
        if self._lifecycle_launch_count != 1:
            raise RuntimeError(
                "persistent SGLang server launched more than once: "
                f"engine_uuid={self._lifecycle_engine_uuid} count={self._lifecycle_launch_count}"
            )
        launch_start = time.perf_counter()
        emit_lifecycle(
            "SGLangHttpServer",
            "launch_start",
            engine_uuid=self._lifecycle_engine_uuid,
            construction_count=self._lifecycle_construction_count,
            launch_count=self._lifecycle_launch_count,
            rollout_mode=str(self.rollout_mode),
            replica_rank=int(self.replica_rank),
            node_rank=int(self.node_rank),
            memory=lifecycle_memory("before_rollout_server_launch"),
        )
        self._ensure_sglang_imports()
        if self.node_rank != 0:
            assert master_address and master_port, "non-master node should provide master address and port"
            self._master_address = master_address
            self._master_port = master_port

        raw_sglang_engine_kwargs = dict(
            self.config.get("engine_kwargs", {}).get("sglang", {}) or {}
        )
        paired_seed_base = raw_sglang_engine_kwargs.pop(
            "paired_seed_base",
            None,
        )
        engine_kwargs, attention_backend = _prepare_engine_kwargs(
            raw_sglang_engine_kwargs,
            self._ServerArgs,
        )
        if paired_seed_base is not None:
            engine_kwargs["random_seed"] = _derive_paired_server_seed(
                int(paired_seed_base),
                int(self.replica_rank),
            )
            logger.info(
                "Resolved paired SGLang server seed base=%s replica=%s seed=%s",
                paired_seed_base,
                self.replica_rank,
                engine_kwargs["random_seed"],
            )
        quantization = self.config.get("quantization", None)
        if quantization is not None:
            if quantization == "fp8":
                assert version.parse(self._sglang.__version__) >= version.parse("0.5.5"), (
                    "sglang>=0.5.5 is required for FP8 quantization"
                )
                FP8_BLOCK_QUANT_KWARGS = {
                    "activation_scheme": "dynamic",
                    "fmt": "e4m3",
                    "quant_method": "fp8",
                    "weight_block_size": [128, 128],
                }
                fp8_block_quant_kwargs = dict(FP8_BLOCK_QUANT_KWARGS)
            else:
                raise ValueError(f"Currently only support fp8 quantization, got: {quantization}")
        model_override_args: dict[str, Any] = {}
        if quantization == "fp8":
            model_override_args["quantization_config"] = (
                fp8_block_quant_kwargs
            )
        configured_model_override = engine_kwargs.pop(
            "json_model_override_args", None
        )
        if configured_model_override:
            if isinstance(configured_model_override, str):
                configured_model_override = json.loads(
                    configured_model_override
                )
            if not isinstance(configured_model_override, dict):
                raise TypeError(
                    "engine_kwargs.sglang.json_model_override_args must be "
                    "a JSON object"
                )
            model_override_args.update(configured_model_override)
        env_model_override = os.environ.get(
            "SGLANG_JSON_MODEL_OVERRIDE_ARGS", ""
        ).strip()
        if env_model_override:
            parsed_env_override = json.loads(env_model_override)
            if not isinstance(parsed_env_override, dict):
                raise TypeError(
                    "SGLANG_JSON_MODEL_OVERRIDE_ARGS must decode to an object"
                )
            model_override_args.update(parsed_env_override)
        if model_override_args:
            logger.info(
                "SGLang model overrides for replica %s: %s",
                self.replica_rank,
                model_override_args,
            )
        dist_init_addr = (
            f"[{self._master_address}]:{self._master_port}"
            if is_valid_ipv6_address(self._master_address)
            else f"{self._master_address}:{self._master_port}"
        )
        infer_tp = self.config.tensor_model_parallel_size * self.config.data_parallel_size
        args = {
            "model_path": self.model_config.local_path,
            "dtype": self.config.dtype,
            "mem_fraction_static": self.config.gpu_memory_utilization,
            "disable_cuda_graph": self.config.enforce_eager,
            "enable_memory_saver": True,
            "base_gpu_id": self.base_gpu_id,
            "gpu_id_step": 1,
            "tp_size": infer_tp,
            "dp_size": self.config.data_parallel_size,
            "ep_size": self.config.expert_parallel_size,
            "node_rank": self.node_rank,
            "load_format": self.config.load_format,
            "dist_init_addr": dist_init_addr,
            "nnodes": self.nnodes,
            "trust_remote_code": self.model_config.trust_remote_code,
            "max_running_requests": self.config.get("max_num_seqs", None),
            "log_level": "error",
            "mm_attention_backend": "fa3",
            "attention_backend": attention_backend if attention_backend is not None else "fa3",
            "skip_tokenizer_init": self.config.skip_tokenizer_init,
            "skip_server_warmup": True,
            "quantization": quantization,
            "json_model_override_args": json.dumps(model_override_args),
            **engine_kwargs,
        }

        if self.config.prometheus.enable:
            if self.config.prometheus.served_model_name:
                # Extract model name from path if it's a full path
                served_model_name = self.config.prometheus.served_model_name
                if "/" in served_model_name:
                    # If it's a full path, extract the last part as model name
                    served_model_name = served_model_name.split("/")[-1]
                args["served_model_name"] = served_model_name

            # start sglang metrics
            args["enable_metrics"] = True

        # enable_weights_cpu_backup is supported in sglang>=0.5.3
        if "enable_weights_cpu_backup" in [f.name for f in dataclasses.fields(self._ServerArgs)]:
            enable_weights_cpu_backup = True if self.rollout_mode == RolloutMode.COLOCATED else False
            args["enable_weights_cpu_backup"] = enable_weights_cpu_backup

        if self.config.enable_rollout_routing_replay:
            args.update({"enable_return_routed_experts": True})

        # mtp
        if self.config.mtp.enable and self.config.mtp.enable_rollout:
            # Enable weights CPU backup for sglang >= 0.5.6
            if self._sglang.__version__ < "0.5.6":
                raise ValueError(f"sglang version {self._sglang.__version__} is not supported for MTP rollout")

            args["speculative_algorithm"] = self.config.mtp.speculative_algorithm
            args["speculative_num_steps"] = self.config.mtp.speculative_num_steps
            args["speculative_eagle_topk"] = self.config.mtp.speculative_eagle_topk
            args["speculative_num_draft_tokens"] = self.config.mtp.speculative_num_draft_tokens

            args["enable_weights_cpu_backup"] = True
            args["enable_draft_weights_cpu_backup"] = True

        # NOTE: We can't directly call SGLang's launch_server since it's not an async function.
        # https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/http_server.py
        sglang_kwargs = (self.config.get("engine_kwargs", {}) or {}).get("sglang", {}) or {}
        if not str(sglang_kwargs.get("dllm_algorithm", "")).startswith("Gemma4"):
            self._sglang_engine._set_envs_and_config = _set_envs_and_config
        os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
        server_args = self._ServerArgs(**args)
        launch_subprocesses = getattr(self._http_server, "_launch_subprocesses", None)
        if launch_subprocesses is None:
            launch_subprocesses = self._sglang_engine.Engine._launch_subprocesses
        launch_subprocess_params = set(inspect.signature(launch_subprocesses).parameters)
        if {
            "init_tokenizer_manager_func",
            "run_scheduler_process_func",
            "run_detokenizer_process_func",
        }.issubset(launch_subprocess_params):
            self.tokenizer_manager, self.template_manager, self.scheduler_info, *_ = launch_subprocesses(
                server_args=server_args,
                init_tokenizer_manager_func=self._sglang_engine.init_tokenizer_manager,
                run_scheduler_process_func=_run_sglang_scheduler_process_with_mq_shm_unlink,
                run_detokenizer_process_func=self._sglang_engine.run_detokenizer_process,
            )
        else:
            launch_kwargs = {"server_args": server_args}
            if "run_scheduler_process_func" in launch_subprocess_params:
                launch_kwargs["run_scheduler_process_func"] = _run_sglang_scheduler_process_with_mq_shm_unlink
            self.tokenizer_manager, self.template_manager, self.scheduler_info, *_ = launch_subprocesses(**launch_kwargs)

        # In multi-node cases, non-zero rank nodes should not launch http server.
        if self.node_rank > 0:
            return

        self._http_server.set_global_state(
            self._http_server._GlobalState(
                tokenizer_manager=self.tokenizer_manager,
                template_manager=self.template_manager,
                scheduler_info=self.scheduler_info,
            )
        )
        self._http_server.app.is_single_tokenizer_mode = True

        # Set warmup_thread_{kw}args to avoid AttributeError in lifespan function
        self._http_server.app.server_args = server_args
        self._http_server.app.warmup_thread_kwargs = {"server_args": server_args}
        self._http_server.app.warmup_thread_args = (server_args, None, None)

        # Manually add Prometheus middleware before starting server
        # This ensures /metrics endpoint is available immediately
        if server_args.enable_metrics:
            from sglang.srt.utils.common import add_prometheus_middleware

            add_prometheus_middleware(self._http_server.app)

        self._server_port, self._server_task = await run_unvicorn(self._http_server.app, server_args, self._server_address)
        self.tokenizer_manager.server_status = self._tokenizer_manager.ServerStatus.Up
        emit_lifecycle(
            "SGLangHttpServer",
            "launch_done",
            engine_uuid=self._lifecycle_engine_uuid,
            construction_count=self._lifecycle_construction_count,
            launch_count=self._lifecycle_launch_count,
            rollout_mode=str(self.rollout_mode),
            replica_rank=int(self.replica_rank),
            node_rank=int(self.node_rank),
            server_address=self._server_address,
            server_port=int(self._server_port),
            seconds=time.perf_counter() - launch_start,
            memory=lifecycle_memory("after_rollout_server_launch"),
        )

    async def wake_up(self):
        self._lifecycle_wake_count += 1
        tags = _memory_occupation_tags(self.config)
        start = time.perf_counter()
        emit_lifecycle(
            "SGLangHttpServer",
            "wake_start",
            engine_uuid=self._lifecycle_engine_uuid,
            launch_count=self._lifecycle_launch_count,
            wake_count=self._lifecycle_wake_count,
            rollout_mode=str(self.rollout_mode),
            replica_rank=int(self.replica_rank),
            node_rank=int(self.node_rank),
            tags=tags,
            memory=lifecycle_memory("before_rollout_wake"),
        )
        if self.node_rank != 0:
            result = {"resumed": False, "reason": "nonzero_node_rank", "tags": tags}
            emit_lifecycle(
                "SGLangHttpServer",
                "wake_done",
                engine_uuid=self._lifecycle_engine_uuid,
                wake_count=self._lifecycle_wake_count,
                seconds=time.perf_counter() - start,
                result=result,
                memory=lifecycle_memory("after_rollout_wake_skipped"),
            )
            return result

        if self.rollout_mode == RolloutMode.HYBRID:
            # In hybrid mode, rollout is wake up in `update_weights`
            emit_lifecycle(
                "SGLangHttpServer",
                "wake_delegated_to_hot_update",
                engine_uuid=self._lifecycle_engine_uuid,
                wake_count=self._lifecycle_wake_count,
                rollout_mode=str(self.rollout_mode),
                tags=tags,
                memory=lifecycle_memory("hybrid_wake_delegated"),
            )
            raise ValueError(f"wake_up not support rollout_mode {self.rollout_mode}")
        elif self.rollout_mode == RolloutMode.COLOCATED:
            # Directly call engine to wake up without sync weights.
            obj = self._io_struct.ResumeMemoryOccupationReqInput(tags=tags)
            await self.tokenizer_manager.resume_memory_occupation(obj, None)
            await self.tokenizer_manager.flush_cache()
            result = {"resumed": True, "reason": "resumed", "tags": tags}
        elif self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip wake_up in standalone mode")
            result = {"resumed": False, "reason": "standalone", "tags": tags}
        emit_lifecycle(
            "SGLangHttpServer",
            "wake_done",
            engine_uuid=self._lifecycle_engine_uuid,
            launch_count=self._lifecycle_launch_count,
            wake_count=self._lifecycle_wake_count,
            rollout_mode=str(self.rollout_mode),
            seconds=time.perf_counter() - start,
            result=result,
            memory=lifecycle_memory("after_rollout_wake"),
        )
        return result

    async def sleep(self):
        self._lifecycle_sleep_count += 1
        tags = _memory_occupation_tags(self.config)
        start = time.perf_counter()
        emit_lifecycle(
            "SGLangHttpServer",
            "sleep_start",
            engine_uuid=self._lifecycle_engine_uuid,
            construction_count=self._lifecycle_construction_count,
            launch_count=self._lifecycle_launch_count,
            sleep_count=self._lifecycle_sleep_count,
            rollout_mode=str(self.rollout_mode),
            replica_rank=int(self.replica_rank),
            node_rank=int(self.node_rank),
            tags=tags,
            memory=lifecycle_memory("before_rollout_sleep"),
        )
        if self.node_rank != 0:
            result = {
                "released": False,
                "reason": "nonzero_node_rank",
                "rollout_mode": str(self.rollout_mode),
                "tags": tags,
            }
            emit_lifecycle(
                "SGLangHttpServer",
                "sleep_done",
                engine_uuid=self._lifecycle_engine_uuid,
                sleep_count=self._lifecycle_sleep_count,
                seconds=time.perf_counter() - start,
                result=result,
                memory=lifecycle_memory("after_rollout_sleep_skipped"),
            )
            return result
        if not self.config.free_cache_engine:
            result = {
                "released": False,
                "reason": "free_cache_engine_disabled",
                "rollout_mode": str(self.rollout_mode),
                "tags": tags,
            }
            emit_lifecycle(
                "SGLangHttpServer",
                "sleep_done",
                engine_uuid=self._lifecycle_engine_uuid,
                sleep_count=self._lifecycle_sleep_count,
                seconds=time.perf_counter() - start,
                result=result,
                memory=lifecycle_memory("after_rollout_sleep_skipped"),
            )
            return result

        if self.rollout_mode == RolloutMode.HYBRID:
            obj = self._io_struct.ReleaseMemoryOccupationReqInput(tags=tags)
            await self.tokenizer_manager.release_memory_occupation(obj, None)
        elif self.rollout_mode == RolloutMode.COLOCATED:
            obj = self._io_struct.ReleaseMemoryOccupationReqInput(tags=tags)
            await self.tokenizer_manager.release_memory_occupation(obj, None)
        elif self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip sleep in standalone mode")
            result = {
                "released": False,
                "reason": "standalone",
                "rollout_mode": str(self.rollout_mode),
                "tags": tags,
            }
            emit_lifecycle(
                "SGLangHttpServer",
                "sleep_done",
                engine_uuid=self._lifecycle_engine_uuid,
                sleep_count=self._lifecycle_sleep_count,
                seconds=time.perf_counter() - start,
                result=result,
                memory=lifecycle_memory("after_rollout_sleep_skipped"),
            )
            return result
        result = {
            "released": True,
            "reason": "released",
            "rollout_mode": str(self.rollout_mode),
            "tags": tags,
        }
        emit_lifecycle(
            "SGLangHttpServer",
            "sleep_done",
            engine_uuid=self._lifecycle_engine_uuid,
            construction_count=self._lifecycle_construction_count,
            launch_count=self._lifecycle_launch_count,
            sleep_count=self._lifecycle_sleep_count,
            rollout_mode=str(self.rollout_mode),
            seconds=time.perf_counter() - start,
            result=result,
            memory=lifecycle_memory("after_rollout_sleep"),
        )
        return result

    async def clear_kv_cache(self):
        if self.node_rank == 0:
            await self.tokenizer_manager.flush_cache()

    async def _get_dllm_stats_snapshot(self) -> dict[str, float]:
        if not _is_dllm_rollout(self.config):
            return {}
        if os.environ.get("VERL_SGLANG_PER_REQUEST_STATS", "0").strip().lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return {}
        timeout_seconds = float(os.environ.get("VERL_SGLANG_STATS_TIMEOUT_SECONDS", "5"))
        try:
            state = await asyncio.wait_for(
                self.tokenizer_manager.get_internal_state(),
                timeout=max(timeout_seconds, 0.1),
            )
            return _sum_dllm_stats(state)
        except TimeoutError:
            logger.warning(
                "Timed out after %.1fs collecting SGLang DLLM stats; continuing without snapshot",
                timeout_seconds,
            )
            emit_lifecycle(
                "SGLangHttpServer",
                "stats_snapshot_timeout",
                engine_uuid=self._lifecycle_engine_uuid,
                node_rank=int(self.node_rank),
                replica_rank=int(self.replica_rank),
                timeout_seconds=float(timeout_seconds),
            )
            return {}
        except Exception as exc:
            logger.debug("Failed to collect SGLang DLLM stats snapshot: %s", exc)
            return {}

    @staticmethod
    def _normalize_prompt_ids(prompt_ids: Any) -> list[int]:
        def to_python(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().tolist()
            if hasattr(value, "tolist") and not isinstance(value, (dict, list, tuple)):
                return value.tolist()
            return value

        def prompt_mapping_key(value: Any) -> Optional[str]:
            if not hasattr(value, "keys") or not hasattr(value, "__getitem__"):
                return None
            keys = value.keys()
            for key in ("input_ids", "prompt_ids", "prompts"):
                if key in keys:
                    return key
            return None

        def unwrap(value: Any) -> Any:
            value = to_python(value)
            key = prompt_mapping_key(value)
            while key is not None:
                value = to_python(value[key])
                key = prompt_mapping_key(value)
            if hasattr(value, "keys") and hasattr(value, "__getitem__"):
                if not isinstance(value, (list, tuple)):
                    raise ValueError(
                        "SGLangHttpServer.generate received a prompt dict without input_ids, prompt_ids, or prompts"
                    )
            return value

        def is_token_scalar(value: Any) -> bool:
            value = to_python(value)
            if isinstance(value, (dict, list, tuple, str, bytes)):
                return False
            if hasattr(value, "item") and not isinstance(value, int):
                value = value.item()
            try:
                int(value)
            except (TypeError, ValueError):
                return False
            return True

        def normalize_token(token_id: Any) -> int:
            token_id = to_python(token_id)
            if hasattr(token_id, "item") and not isinstance(token_id, int):
                token_id = token_id.item()
            return int(token_id)

        def extract_single_prompt(value: Any) -> list[int]:
            value = unwrap(value)
            if isinstance(value, tuple):
                value = list(value)
            if not isinstance(value, list):
                if is_token_scalar(value):
                    return [normalize_token(value)]
                raise TypeError(f"SGLangHttpServer.generate expected token ids, got {type(value).__name__}")
            if not value:
                return []

            if all(is_token_scalar(item) for item in value):
                return [normalize_token(item) for item in value]

            if len(value) == 1:
                return extract_single_prompt(value[0])

            raise ValueError(f"SGLangHttpServer.generate expects one prompt, got nested batch size {len(value)}")

        return extract_single_prompt(prompt_ids)

    @staticmethod
    def _unwrap_single_sglang_output(output: Any) -> dict[str, Any]:
        if isinstance(output, list):
            if len(output) != 1:
                raise ValueError(f"SGLangHttpServer.generate expected one output, got {len(output)}")
            output = output[0]
        if not isinstance(output, dict):
            raise TypeError(f"SGLangHttpServer.generate expected dict output, got {type(output).__name__}")
        return output

    @staticmethod
    def _normalize_output_ids(output_ids: Any) -> list[int]:
        if output_ids is None:
            return []
        if isinstance(output_ids, torch.Tensor):
            output_ids = output_ids.detach().cpu().tolist()
        elif hasattr(output_ids, "tolist") and not isinstance(output_ids, (dict, list, tuple)):
            output_ids = output_ids.tolist()
        if isinstance(output_ids, tuple):
            output_ids = list(output_ids)
        if not isinstance(output_ids, list):
            return [int(output_ids)]
        return [int(token_id) for token_id in output_ids]

    @staticmethod
    def _is_prefix(prefix: list[int], values: list[int]) -> bool:
        return len(prefix) <= len(values) and values[: len(prefix)] == prefix

    @classmethod
    def _merge_stream_output_ids(cls, cumulative_ids: list[int], chunk_ids: Any) -> tuple[list[int], list[int]]:
        """Merge SGLang stream chunks whether output_ids are deltas or cumulative."""
        chunk_ids = cls._normalize_output_ids(chunk_ids)
        if not chunk_ids:
            return list(cumulative_ids), []
        if cls._is_prefix(cumulative_ids, chunk_ids):
            return list(chunk_ids), chunk_ids[len(cumulative_ids) :]
        if cls._is_prefix(chunk_ids, cumulative_ids):
            return list(cumulative_ids), []
        return [*cumulative_ids, *chunk_ids], chunk_ids

    @staticmethod
    def _merge_stream_text(cumulative_text: str, chunk_text: Any) -> tuple[str, str]:
        if chunk_text is None:
            return cumulative_text, ""
        chunk_text = str(chunk_text)
        if chunk_text.startswith(cumulative_text):
            return chunk_text, chunk_text[len(cumulative_text) :]
        if cumulative_text.startswith(chunk_text):
            return cumulative_text, ""
        return cumulative_text + chunk_text, chunk_text

    @staticmethod
    def _as_sdar_replay_items(replay_tensors: Any) -> list[dict]:
        if not replay_tensors:
            return []
        if isinstance(replay_tensors, list):
            return replay_tensors
        if isinstance(replay_tensors, tuple):
            return list(replay_tensors)
        return [replay_tensors]

    @classmethod
    def _preselect_qwen_replay_items(
        cls,
        replay_tensors: Any,
    ) -> tuple[list[dict], int]:
        """Bound no-stream replay memory before tensors cross Ray boundaries."""
        items = cls._as_sdar_replay_items(replay_tensors)
        fraction_text = os.environ.get(
            "SGLANG_SDAR_REPLAY_PRESELECT_FRACTION",
            "",
        ).strip()
        if not items or not fraction_text:
            return items, len(items)
        from sglang.srt.dllm.algorithm.sdar_replay_preselection import (
            LEGACY_PRESELECT_MODE,
            ReplayPreselectionPolicy,
            cleanup_discarded_replay_rows,
            select_completed_block_replay_rows,
        )

        policy = ReplayPreselectionPolicy.from_values(
            fraction=fraction_text,
            mode=os.environ.get(
                "SGLANG_SDAR_REPLAY_PRESELECT_MODE",
                LEGACY_PRESELECT_MODE,
            ),
            direction=os.environ.get(
                "SGLANG_SDAR_REPLAY_PRESELECT_DIRECTION",
                "highest",
            ),
            score=os.environ.get(
                "SGLANG_SDAR_REPLAY_PRESELECT_SCORE",
                "new_accept_count",
            ),
            group_fields=os.environ.get(
                "SGLANG_SDAR_REPLAY_PRESELECT_GROUP_FIELDS",
                "response_uid,block_index",
            ),
            final_policy=os.environ.get(
                "SGLANG_SDAR_REPLAY_PRESELECT_FINAL_POLICY",
                "always_include",
            ),
        )
        metadata_rows = [
            (
                item.get("metadata")
                if isinstance(item, dict)
                and isinstance(item.get("metadata"), dict)
                else item
            )
            for item in items
        ]
        preselected_rows = [
            isinstance(metadata, dict)
            and bool(metadata.get("producer_preselected", False))
            for metadata in metadata_rows
        ]
        if any(preselected_rows) and not all(preselected_rows):
            raise RuntimeError(
                "mixed producer-preselected and unselected SDAR replay rows"
            )
        if preselected_rows and all(preselected_rows):
            source_rows_by_block: dict[int, int] = {}
            for metadata in metadata_rows:
                block_index = int(
                    metadata.get(
                        "block_idx",
                        metadata.get("block_index", 0),
                    )
                )
                source_rows_by_block[block_index] = max(
                    source_rows_by_block.get(block_index, 0),
                    int(metadata.get("producer_source_rows", 0)),
                )
            return items, sum(source_rows_by_block.values())

        groups: dict[int, list[tuple[int, dict, dict]]] = {}
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            metadata = (
                item.get("metadata")
                if isinstance(item.get("metadata"), dict)
                else item
            )
            block_index = int(
                metadata.get(
                    "block_idx",
                    metadata.get("block_index", item.get("block_idx", 0)),
                )
            )
            groups.setdefault(block_index, []).append((index, item, metadata))

        keep_indices: set[int] = set()
        discarded_items: list[dict] = []
        for entries in groups.values():
            group_items = [entry[1] for entry in entries]
            result = select_completed_block_replay_rows(
                group_items,
                policy=policy,
            )
            for local_index in result.selected_indices:
                keep_indices.add(entries[local_index][0])
            discarded_items.extend(
                entries[local_index][1]
                for local_index in result.discarded_indices
            )

        cleanup_discarded_replay_rows(discarded_items)

        return [
            item for index, item in enumerate(items) if index in keep_indices
        ], len(items)

    @staticmethod
    def _is_gemma_intermediate_replay(payload: Any) -> bool:
        if isinstance(payload, dict):
            return payload.get("format") in {
                "gemma_intermediate_replay_v1",
                "gemma_intermediate_replay_table_handle_v1",
            }
        if isinstance(payload, (list, tuple)) and payload:
            return all(SGLangHttpServer._is_gemma_intermediate_replay(item) for item in payload)
        return False

    @staticmethod
    def _tensor_nbytes(value: torch.Tensor) -> int:
        return int(value.numel()) * int(value.element_size())

    @classmethod
    def _bind_gemma_intermediate_replay_item(
        cls,
        item: dict[str, Any],
        *,
        request_id: str,
        prompt_ids: list[int],
        output_ids: list[int],
        stream_context: Optional[dict[str, Any]],
        response_length: Optional[int] = None,
        response_finished: bool = False,
    ) -> dict[str, Any]:
        if item.get("format") != "gemma_intermediate_replay_v1":
            raise RuntimeError(
                "Invalid DiffusionGemma replay format: "
                f"{item.get('format')!r}"
            )
        block_size = int(item.get("block_size", 0) or 0)
        block_index = int(item.get("block_index", -1))
        steps_used = int(item.get("steps_used", -1))
        if block_size <= 0 or block_index < 0 or steps_used <= 0:
            raise RuntimeError(
                "Invalid DiffusionGemma replay block metadata: "
                f"block_size={block_size} block_index={block_index} "
                f"steps_used={steps_used}"
            )

        tensor_shapes = {
            "final_target_ids": (block_size,),
            "target_valid_mask": (block_size,),
            "step_indices": (steps_used,),
            "pre_forward_canvas_ids": (steps_used, block_size),
            "self_conditioning_present": (steps_used,),
            "valid_masks": (steps_used, block_size),
            "done_masks": (steps_used, block_size),
            "terminal_mask": (steps_used, block_size),
            "row_valid_mask": (steps_used,),
        }
        for key, expected_shape in tensor_shapes.items():
            value = item.get(key)
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(
                    f"DiffusionGemma replay missing tensor {key}"
                )
            if tuple(value.shape) != expected_shape:
                raise RuntimeError(
                    f"DiffusionGemma replay {key} shape mismatch: "
                    f"{tuple(value.shape)} != {expected_shape}"
                )
        signal = item.get("self_conditioning_signal")
        if (
            not isinstance(signal, torch.Tensor)
            or signal.ndim != 3
            or tuple(signal.shape[:2]) != (steps_used, block_size)
        ):
            raise RuntimeError(
                "DiffusionGemma replay self_conditioning_signal shape mismatch: "
                f"{getattr(signal, 'shape', None)}"
            )
        if int(item.get("captured_rows", steps_used)) != steps_used:
            raise RuntimeError(
                "DiffusionGemma replay captured rows do not equal steps_used"
            )
        committed_response_length = block_index * block_size
        available_tokens = min(
            block_size,
            max(0, len(output_ids) - committed_response_length),
        )
        if available_tokens <= 0:
            raise RuntimeError(
                "DiffusionGemma replay block arrived after the natural response "
                f"end: block={block_index} output_tokens={len(output_ids)}"
            )
        if response_finished and available_tokens < block_size:
            partial_valid = (
                torch.arange(block_size, dtype=torch.long) < available_tokens
            )
            item["target_valid_mask"] = partial_valid
            item["valid_masks"] = (
                item["valid_masks"].detach().cpu().bool()
                & partial_valid.unsqueeze(0)
            )
            item["done_masks"] = (
                item["done_masks"].detach().cpu().bool()
                & partial_valid.unsqueeze(0)
            )
            item["terminal_mask"] = (
                item["terminal_mask"].detach().cpu().bool()
                & partial_valid.unsqueeze(0)
            )
            item["valid_token_count"] = available_tokens
        if int(item.get("valid_token_count", available_tokens)) != available_tokens:
            raise RuntimeError(
                "DiffusionGemma replay valid target length does not match the "
                f"committed output: {item.get('valid_token_count')} != {available_tokens}"
            )
        if bool(item.get("clean_filter_enabled", False)):
            raise RuntimeError(
                "DiffusionGemma intermediate replay must disable clean filtering"
            )
        expected_valid_masks = item["target_valid_mask"].detach().cpu().bool().unsqueeze(0).expand(
            steps_used,
            -1,
        )
        if not torch.equal(
            item["valid_masks"].detach().cpu().bool(),
            expected_valid_masks,
        ):
            raise RuntimeError(
                "DiffusionGemma replay valid step-token masks do not match "
                "the committed target mask"
            )
        if not bool(item["row_valid_mask"].all()):
            raise RuntimeError(
                "DiffusionGemma replay must train every captured row"
            )
        if bool(item["self_conditioning_present"][0]):
            raise RuntimeError(
                "DiffusionGemma replay first step has self-conditioning"
            )
        if bool(torch.count_nonzero(signal[0])):
            raise RuntimeError(
                "DiffusionGemma replay first-step self-conditioning is not zero"
            )
        if steps_used > 1 and not bool(
            item["self_conditioning_present"][1:].all()
        ):
            raise RuntimeError(
                "DiffusionGemma replay is missing later self-conditioning"
            )
        if not bool(torch.isfinite(signal.float()).all()):
            raise RuntimeError(
                "DiffusionGemma replay self-conditioning contains non-finite values"
            )

        declared_prefix_length = int(
            item.get(
                "expected_committed_response_prefix_length",
                committed_response_length,
            )
        )
        if declared_prefix_length != committed_response_length:
            raise RuntimeError(
                "DiffusionGemma replay committed-prefix length mismatch: "
                f"{declared_prefix_length} != {committed_response_length}"
            )
        block_end = committed_response_length + available_tokens
        committed_target = torch.tensor(
            output_ids[committed_response_length:block_end],
            dtype=torch.long,
        )
        if not torch.equal(
            item["final_target_ids"].detach().cpu().long()[:available_tokens],
            committed_target,
        ):
            raise RuntimeError(
                "DiffusionGemma replay target differs from committed output block"
            )

        context = dict(stream_context or {})
        request_uid = str(
            context.get("request_uid")
            or context.get("request_id")
            or request_id
        )
        sample_uid = str(
            context.get("sample_uid")
            or context.get("sample_id")
            or request_uid
        )
        response_uid = str(
            context.get("response_uid")
            or context.get("uid")
            or request_uid
        )
        response_length = (
            len(output_ids)
            if response_finished
            else int(response_length or len(output_ids))
        )
        if response_length < block_end:
            raise RuntimeError(
                "DiffusionGemma configured response length is shorter than "
                f"committed block end: {response_length} < {block_end}"
            )
        expected_blocks = (
            response_length + block_size - 1
        ) // block_size
        committed_response_prefix = [
            int(value)
            for value in output_ids[:committed_response_length]
        ]
        full_prefix = [
            *[int(value) for value in prompt_ids],
            *committed_response_prefix,
        ]
        tensor_bytes = sum(
            cls._tensor_nbytes(value)
            for value in item.values()
            if isinstance(value, torch.Tensor)
        )
        bound = dict(item)
        bound.update(
            {
                "request_id": str(request_id),
                "request_uid": request_uid,
                "sample_uid": sample_uid,
                "response_uid": response_uid,
                "prompt_ids": [int(value) for value in prompt_ids],
                "committed_response_prefix_ids": committed_response_prefix,
                "committed_prefix_ids": committed_response_prefix,
                "prefix_ids": full_prefix,
                "response_length": response_length,
                "expected_blocks": expected_blocks,
                "response_finished": bool(response_finished),
                "block_finalized": True,
                "captured_rows": steps_used,
                "captured_token_steps": steps_used * block_size,
                "replay_tensor_bytes": int(tensor_bytes),
                "transport_bound_wall_ts": float(time.time()),
            }
        )
        if len(bound["prefix_ids"]) != len(prompt_ids) + committed_response_length:
            raise RuntimeError(
                "DiffusionGemma replay committed prefix contains future tokens"
            )
        return bound

    @classmethod
    def _gemma_replay_handle_metadata(
        cls, item: dict[str, Any]
    ) -> dict[str, Any]:
        tensor_shapes = {}
        tensor_dtypes = {}
        for key, value in item.items():
            if isinstance(value, torch.Tensor):
                tensor_shapes[key] = list(value.shape)
                tensor_dtypes[key] = str(value.dtype)
        return {
            "format": "gemma_intermediate_replay_handle_metadata_v1",
            "schema": "gemma_intermediate_replay_v1",
            "schema_version": int(item.get("schema_version", 1)),
            "request_id": str(item.get("request_id", "")),
            "request_uid": str(item.get("request_uid", "")),
            "sample_uid": str(item.get("sample_uid", "")),
            "response_uid": str(item.get("response_uid", "")),
            "block_index": int(item["block_index"]),
            "block_size": int(item["block_size"]),
            "steps_used": int(item["steps_used"]),
            "captured_rows": int(item["captured_rows"]),
            "captured_token_steps": int(item["captured_token_steps"]),
            "replay_tensor_bytes": int(item["replay_tensor_bytes"]),
            "committed_prefix_length": len(item["committed_prefix_ids"]),
            "tensor_shapes": tensor_shapes,
            "tensor_dtypes": tensor_dtypes,
        }

    @staticmethod
    def _logprob_token_ids(output_token_logprobs: list[Any]) -> Optional[list[int]]:
        token_ids = []
        for item in output_token_logprobs:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                return None
            try:
                token_ids.append(int(item[1]))
            except (TypeError, ValueError):
                return None
        return token_ids

    @classmethod
    def _merge_stream_output_token_logprobs(
        cls,
        cumulative_logprobs: list[Any],
        chunk_logprobs: Any,
        previous_ids: list[int],
        chunk_ids: list[int],
        cumulative_ids: list[int],
        chunk_delta_ids: list[int],
    ) -> list[Any]:
        if not chunk_logprobs:
            return cumulative_logprobs
        chunk_logprobs = list(chunk_logprobs)
        logprob_token_ids = cls._logprob_token_ids(chunk_logprobs)
        if logprob_token_ids == cumulative_ids:
            return chunk_logprobs
        if logprob_token_ids == chunk_delta_ids:
            return [*cumulative_logprobs, *chunk_logprobs]
        if logprob_token_ids == chunk_ids:
            if cls._is_prefix(previous_ids, chunk_ids):
                return chunk_logprobs
            if cls._is_prefix(chunk_ids, previous_ids):
                return cumulative_logprobs
            return [*cumulative_logprobs, *chunk_logprobs]
        if len(chunk_logprobs) == len(cumulative_ids):
            return chunk_logprobs
        if len(chunk_logprobs) == len(chunk_delta_ids):
            return [*cumulative_logprobs, *chunk_logprobs]
        return chunk_logprobs if len(chunk_logprobs) > len(cumulative_logprobs) else cumulative_logprobs

    @staticmethod
    def _meta_info_without_sdar_replay_tensors(meta_info: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in meta_info.items()
            if key
            not in {
                "sglang_sdar_replay_tensors",
                "gemma_intermediate_replay",
            }
        }

    @staticmethod
    def _extract_sdar_replay_trace(meta_info: dict[str, Any], output: Optional[dict[str, Any]] = None) -> Optional[list[Any]]:
        output = output or {}
        replay_trace = meta_info.get("replay_trace", output.get("replay_trace"))
        if replay_trace is not None:
            if isinstance(replay_trace, list):
                return replay_trace
            return [replay_trace]

        trace_records = []
        for stat in meta_info.get("sglang_sdar_request_metrics", []) or []:
            if not isinstance(stat, dict):
                continue
            record = stat.get("replay_trace_record")
            if record is not None:
                trace_records.append(record)
        return trace_records or None

    async def _push_sdar_stream_chunk(
        self,
        *,
        sdar_stream_queue: Any,
        sdar_stream_source: Optional[str],
        sdar_stream_context: Optional[dict[str, Any]],
        request_id: str,
        prompt_ids: list[int],
        output_ids: list[int],
        chunk_output_ids: list[int],
        chunk_index: int,
        output: dict[str, Any],
        meta_info: dict[str, Any],
        replay_items: list[dict],
        text: Optional[str],
        chunk_text: Optional[str],
    ) -> None:
        total_start = time.perf_counter()
        tensor_store = os.environ.get(_SGLANG_SDAR_STREAM_REPLAY_TENSOR_STORE_ENV, "").strip().lower()
        compact_seconds = 0.0
        shm_seconds = 0.0
        table_key = None
        gemma_replay = bool(replay_items) and all(
            isinstance(item, dict)
            and item.get("format") == "gemma_intermediate_replay_v1"
            for item in replay_items
        )
        if any(self._is_gemma_intermediate_replay(item) for item in replay_items) and not gemma_replay:
            raise RuntimeError(
                "SGLang stream chunk mixed DiffusionGemma and generic replay payloads"
            )
        gemma_rows = 0
        gemma_token_steps = 0
        gemma_bytes = 0
        gemma_producer_peak_bytes = 0
        gemma_producer_to_queue_seconds = 0.0
        gemma_store_seconds = 0.0
        if gemma_replay:
            if len(replay_items) != 1:
                raise RuntimeError(
                    "DiffusionGemma streaming requires one committed block per chunk, "
                    f"got {len(replay_items)}"
                )
            replay_items = [
                self._bind_gemma_intermediate_replay_item(
                    replay_items[0],
                    request_id=request_id,
                    prompt_ids=prompt_ids,
                    output_ids=output_ids,
                    stream_context=sdar_stream_context,
                    response_length=int(self.config.response_length),
                    response_finished=bool(
                        meta_info.get(
                            "finish_reason", output.get("finish_reason")
                        )
                        is not None
                    ),
                )
            ]
            replay_item = replay_items[0]
            block_index = int(replay_item["block_index"])
            table_key = f"{request_id}:gemma:block:{block_index}"
            store_start = time.perf_counter()
            ray_ref = ray.put(replay_item) if ray.is_initialized() else None
            if ray_ref is None:
                self._sdar_replay_tensor_store[table_key] = replay_items
            gemma_store_seconds = time.perf_counter() - store_start
            compact_replay = {
                "format": "gemma_intermediate_replay_table_handle_v1",
                "schema": "gemma_intermediate_replay_v1",
                "request_id": table_key,
                "source_request_id": str(request_id),
                "chunk_index": int(chunk_index),
                "block_index": block_index,
                "num_blocks": 1,
                "metadata_payload": self._gemma_replay_handle_metadata(
                    replay_item
                ),
                "transport": "ray_object_store",
                "store_seconds": float(gemma_store_seconds),
                "stored_wall_ts": float(time.time()),
                "fabric_interface": os.environ.get(
                    "SGLANG_SDAR_RESEARCH_FABRIC_INTERFACE", "unset"
                ),
                "fabric_hca": os.environ.get(
                    "SGLANG_SDAR_RESEARCH_FABRIC_HCA", "unset"
                ),
            }
            if ray_ref is not None:
                compact_replay["ray_ref"] = ray_ref
            gemma_rows = int(replay_item["captured_rows"])
            gemma_token_steps = int(replay_item["captured_token_steps"])
            gemma_bytes = int(replay_item["replay_tensor_bytes"])
            gemma_producer_peak_bytes = int(
                replay_item.get("producer_peak_buffer_bytes", gemma_bytes)
            )
            committed_wall_ts = float(
                replay_item.get("block_committed_wall_ts", 0.0) or 0.0
            )
            if committed_wall_ts > 0.0:
                gemma_producer_to_queue_seconds = max(
                    0.0, time.time() - committed_wall_ts
                )
            table_handle = True
        else:
            table_handle = tensor_store in _SGLANG_SDAR_TABLE_TENSOR_STORE_VALUES
        if table_handle and not gemma_replay:
            table_key = f"{request_id}:{int(chunk_index)}"
            self._sdar_replay_tensor_store[table_key] = list(replay_items)
            compact_replay = {
                "format": "sglang_sdar_replay_table_handle_v1",
                "request_id": table_key,
                "source_request_id": str(request_id),
                "chunk_index": int(chunk_index),
                "num_tensors": int(len(replay_items)),
                "metadata_payload": self._compact_sdar_replay_metadata(replay_items),
            }
        elif not gemma_replay:
            compact_start = time.perf_counter()
            compact_replay = self._compact_sdar_replay_tensors(replay_items)
            compact_seconds = time.perf_counter() - compact_start
            if tensor_store in _SGLANG_SDAR_SHM_TENSOR_STORE_VALUES:
                shm_start = time.perf_counter()
                compact_replay = self._compact_replay_to_shm_handles(compact_replay)
                shm_seconds = time.perf_counter() - shm_start
        replay_tensor_format = (
            str(compact_replay.get("format"))
            if isinstance(compact_replay, dict) and compact_replay.get("format") is not None
            else "sglang_sdar_replay_tensor_batch_v1"
        )
        replay_tensor_store = (
            "shm" if replay_tensor_format == "sglang_sdar_replay_shm_batch_v1" else "tensor"
        )
        if table_handle:
            replay_tensor_store = "table"

        raw: dict[str, Any] = {
            "format": "sglang_sdar_stream_chunk_v1",
            "request_id": str(request_id),
            "prompt_ids": list(prompt_ids),
            "output_ids": list(output_ids),
            "chunk_output_ids": list(chunk_output_ids),
            "chunk_index": int(chunk_index),
            "meta_info": self._meta_info_without_sdar_replay_tensors(meta_info),
            "stream_context": dict(sdar_stream_context or {}),
            "sglang_sdar_stream_replay_tensor_store": replay_tensor_store,
            "sglang_sdar_stream_replay_tensor_format": replay_tensor_format,
            "sglang_sdar_replay_tensors_compact": compact_replay,
            "sglang_sdar_replay_tensors_count": len(replay_items),
            "sglang_stream_replay_compact_seconds": float(compact_seconds),
            "sglang_stream_replay_shm_seconds": float(shm_seconds),
            "sglang_stream_replay_items": int(len(replay_items)),
            "sglang_stream_chunk_output_tokens": int(len(chunk_output_ids)),
            "sglang_stream_pre_queue_seconds": float(time.perf_counter() - total_start),
            "sglang_stream_queue_push_start_wall_ts": float(time.time()),
        }
        if gemma_replay:
            raw.update(
                {
                    "gemma_replay_blocks_captured": 1,
                    "gemma_replay_blocks_pushed": 1,
                    "gemma_replay_rows_pushed": gemma_rows,
                    "gemma_replay_token_steps_pushed": gemma_token_steps,
                    "gemma_replay_bytes_pushed": gemma_bytes,
                    "gemma_replay_producer_peak_bytes": gemma_producer_peak_bytes,
                    "gemma_replay_producer_to_queue_seconds": gemma_producer_to_queue_seconds,
                    "gemma_replay_store_seconds": gemma_store_seconds,
                    "gemma_replay_queue_transport": "ray_object_store",
                    "gemma_replay_cross_node_shm_handles": 0,
                }
            )
        replay_trace = self._extract_sdar_replay_trace(meta_info, output)
        if replay_trace is not None:
            raw["rollout_replay_trace"] = replay_trace
        finish_reason = meta_info.get("finish_reason", output.get("finish_reason"))
        if finish_reason is not None:
            raw["finish_reason"] = finish_reason
        if text is not None:
            raw["text"] = text
        if chunk_text is not None:
            raw["chunk_text"] = chunk_text
        source = sdar_stream_source or f"sglang_sdar_stream:{request_id}"
        queue_push_start = time.perf_counter()
        try:
            await sdar_stream_queue.push.remote(raw, source=source)
        except Exception:
            if table_key is not None:
                self._sdar_replay_tensor_store.pop(table_key, None)
            raise
        queue_push_seconds = time.perf_counter() - queue_push_start
        if gemma_replay:
            metrics = self._gemma_replay_stream_metrics.setdefault(
                str(request_id),
                {
                    "captured_blocks": 0.0,
                    "pushed_blocks": 0.0,
                    "pushed_block_indices": [],
                    "pushed_rows": 0.0,
                    "pushed_token_steps": 0.0,
                    "pushed_bytes": 0.0,
                    "queue_backpressure_seconds": 0.0,
                    "store_seconds": 0.0,
                    "producer_to_queue_seconds": 0.0,
                    "producer_peak_bytes": 0.0,
                    "cross_node_shm_handles": 0.0,
                },
            )
            metrics["captured_blocks"] += 1.0
            metrics["pushed_blocks"] += 1.0
            metrics["pushed_block_indices"].append(
                int(replay_items[0]["block_index"])
            )
            metrics["pushed_rows"] += float(gemma_rows)
            metrics["pushed_token_steps"] += float(gemma_token_steps)
            metrics["pushed_bytes"] += float(gemma_bytes)
            metrics["queue_backpressure_seconds"] += float(
                queue_push_seconds
            )
            metrics["store_seconds"] += float(gemma_store_seconds)
            metrics["producer_to_queue_seconds"] += float(
                gemma_producer_to_queue_seconds
            )
            metrics["producer_peak_bytes"] = max(
                metrics["producer_peak_bytes"],
                float(gemma_producer_peak_bytes),
            )
            logger.info(
                "DiffusionGemma replay pushed request=%s block=%d rows=%d "
                "token_steps=%d bytes=%d queue_backpressure=%.6fs "
                "producer_to_queue=%.6fs producer_peak_bytes=%d "
                "transport=ray_object_store fabric=%s/%s",
                request_id,
                int(replay_items[0]["block_index"]),
                gemma_rows,
                gemma_token_steps,
                gemma_bytes,
                queue_push_seconds,
                gemma_producer_to_queue_seconds,
                gemma_producer_peak_bytes,
                compact_replay["fabric_interface"],
                compact_replay["fabric_hca"],
            )

    async def _consume_streaming_generate_request(
        self,
        generate_request: Any,
        *,
        request_id: str,
        prompt_ids: list[int],
        sdar_stream_queue: Any,
        sdar_stream_source: Optional[str],
        sdar_stream_context: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        final_output: Optional[dict[str, Any]] = None
        cumulative_output_ids: list[int] = []
        cumulative_text = ""
        has_text = False
        cumulative_logprobs: list[Any] = []

        chunk_index = -1
        replay_chunk_count = 0
        async for chunk in self.tokenizer_manager.generate_request(generate_request, None):
            chunk_index += 1
            output = self._unwrap_single_sglang_output(chunk)
            meta_info = output.get("meta_info", {}) or {}
            previous_output_ids = list(cumulative_output_ids)
            chunk_ids = self._normalize_output_ids(output.get("output_ids", []))
            cumulative_output_ids, chunk_output_ids = self._merge_stream_output_ids(cumulative_output_ids, chunk_ids)

            chunk_text: Optional[str] = None
            if "text" in output:
                has_text = True
                cumulative_text, chunk_text = self._merge_stream_text(cumulative_text, output.get("text"))

            cumulative_logprobs = self._merge_stream_output_token_logprobs(
                cumulative_logprobs,
                meta_info.get("output_token_logprobs"),
                previous_output_ids,
                chunk_ids,
                cumulative_output_ids,
                chunk_output_ids,
            )

            replay_payload = meta_info.get("gemma_intermediate_replay")
            if replay_payload is None:
                replay_payload = meta_info.get("sglang_sdar_replay_tensors")
            replay_items = self._as_sdar_replay_items(replay_payload)
            if replay_items:
                replay_chunk_count += 1
                gemma_items = all(
                    self._is_gemma_intermediate_replay(replay_item)
                    for replay_item in replay_items
                )
                # A Gemma scheduler callback may coalesce committed blocks, and
                # each block is an actor delivery unit. Generic SDAR/Qwen replay
                # contains many step rows per chunk and must remain one compact
                # queue item.
                if gemma_items:
                    request_metrics = self._gemma_replay_stream_metrics.get(
                        str(request_id), {}
                    )
                    seen_block_indices = {
                        int(index)
                        for index in request_metrics.get(
                            "pushed_block_indices", []
                        )
                    }
                    unique_replay_items = []
                    for replay_item in replay_items:
                        block_index = int(replay_item["block_index"])
                        if block_index in seen_block_indices:
                            continue
                        unique_replay_items.append(replay_item)
                        seen_block_indices.add(block_index)
                    replay_groups = [
                        [replay_item] for replay_item in unique_replay_items
                    ]
                else:
                    replay_groups = [replay_items]
                for replay_group in replay_groups:
                    await self._push_sdar_stream_chunk(
                        sdar_stream_queue=sdar_stream_queue,
                        sdar_stream_source=sdar_stream_source,
                        sdar_stream_context=sdar_stream_context,
                        request_id=request_id,
                        prompt_ids=prompt_ids,
                        output_ids=cumulative_output_ids,
                        chunk_output_ids=chunk_output_ids,
                        chunk_index=chunk_index,
                        output=output,
                        meta_info=meta_info,
                        replay_items=replay_group,
                        text=cumulative_text if has_text else None,
                        chunk_text=chunk_text,
                    )

            final_output = dict(output)
            final_output["output_ids"] = list(cumulative_output_ids)
            final_output["meta_info"] = dict(meta_info)
            if has_text:
                final_output["text"] = cumulative_text

        if final_output is None:
            raise RuntimeError("SGLang streaming generate_request yielded no chunks")

        final_meta_info = dict(final_output.get("meta_info", {}) or {})
        final_meta_info.pop("sglang_sdar_replay_tensors", None)
        if cumulative_logprobs:
            final_meta_info["output_token_logprobs"] = cumulative_logprobs
        final_output["meta_info"] = final_meta_info
        final_output["output_ids"] = list(cumulative_output_ids)
        if has_text:
            final_output["text"] = cumulative_text
        print(
            f"[SDAR_STREAM_DIAG] request_id={request_id} stream_chunks={chunk_index + 1} "
            f"replay_chunks={replay_chunk_count}",
            flush=True,
        )
        return final_output

    async def generate(
        self,
        prompt_ids: torch.Tensor,
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        stream_sdar_replay: bool = False,
        sdar_stream_queue: Optional[Any] = None,
        sdar_stream_source: Optional[str] = None,
        sdar_stream_context: Optional[dict[str, Any]] = None,
    ) -> TokenOutput:
        """Generate sequence with token-in-token-out."""
        generate_total_start = time.perf_counter()
        global _SOURCE_MARKER_LOGGED
        if not _SOURCE_MARKER_LOGGED:
            logger.warning(
                "SGLangHttpServer source marker: %s batch_input_ids=%s",
                __file__,
                _batch_input_ids_enabled(self.config),
            )
            _SOURCE_MARKER_LOGGED = True
        # TODO(@wuxibin): switch to `/generate` http endpoint once multi-modal support ready.
        prompt_ids = self._normalize_prompt_ids(prompt_ids)

        max_possible_tokens = self.config.max_model_len - len(prompt_ids)

        if max_possible_tokens < 0:
            raise ValueError(
                f"Prompt length ({len(prompt_ids)}) exceeds the model's maximum context length "
                f"({self.config.max_model_len})."
            )

        if "max_new_tokens" in sampling_params:
            max_new_tokens = sampling_params.pop("max_new_tokens")
        elif "max_tokens" in sampling_params:
            # support vllm-style 'max_tokens' param
            max_new_tokens = sampling_params.pop("max_tokens")
        elif _is_dllm_rollout(self.config):
            # The token-in-token-out agent loop passes an unpadded prompt.  The
            # AR default below fills prompt_length+response_length and makes
            # SDAR decode extra diffusion blocks that are later truncated.
            max_new_tokens = self.config.response_length
        else:
            max_new_tokens = self.config.response_length + self.config.prompt_length - len(prompt_ids)

        # Clamp max_new_tokens to the valid range [0, max_possible_tokens]
        max_new_tokens = max(0, min(max_new_tokens, max_possible_tokens))

        assert max_new_tokens <= max_possible_tokens, (
            f"max_new_tokens {max_new_tokens} exceeds available context space {max_possible_tokens}"
        )
        sampling_params["max_new_tokens"] = max_new_tokens
        _apply_dllm_stop_token_ids(self, sampling_params)
        return_logprob = sampling_params.pop("logprobs", False)
        emit_lifecycle(
            "SGLangHttpServer",
            "generate_start",
            engine_uuid=self._lifecycle_engine_uuid,
            node_rank=int(self.node_rank),
            replica_rank=int(self.replica_rank),
            request_id=str(request_id),
            prompt_tokens=int(len(prompt_ids)),
            max_new_tokens=int(max_new_tokens),
            stream_sdar_replay=bool(stream_sdar_replay),
            memory=lifecycle_memory("before_generate"),
        )

        request = {
            "rid": request_id,
            "input_ids": [prompt_ids] if _batch_input_ids_enabled(self.config) else prompt_ids,
            "sampling_params": sampling_params,
            "return_logprob": return_logprob,
            "image_data": image_data,
            # TODO: support video input for sglang
            # video_data=video_data,
        }

        if self.config.enable_rollout_routing_replay:
            request.update({"return_routed_experts": True})

        should_stream_sdar_replay = bool(stream_sdar_replay and sdar_stream_queue is not None)
        if should_stream_sdar_replay:
            request["stream"] = True
            self._gemma_replay_stream_metrics.pop(str(request_id), None)

        stats_before_start = time.perf_counter()
        stats_before = await self._get_dllm_stats_snapshot()
        stats_before_seconds = time.perf_counter() - stats_before_start
        emit_lifecycle(
            "SGLangHttpServer",
            "generate_request_start",
            engine_uuid=self._lifecycle_engine_uuid,
            node_rank=int(self.node_rank),
            replica_rank=int(self.replica_rank),
            request_id=str(request_id),
            stats_before_seconds=float(stats_before_seconds),
        )
        request_build_seconds = 0.0
        request_generate_seconds = 0.0
        try:
            request_build_start = time.perf_counter()
            generate_request = self._io_struct.GenerateReqInput(**request)
            request_build_seconds += time.perf_counter() - request_build_start
            request_generate_start = time.perf_counter()
            if should_stream_sdar_replay:
                output = await self._consume_streaming_generate_request(
                    generate_request,
                    request_id=request_id,
                    prompt_ids=prompt_ids,
                    sdar_stream_queue=sdar_stream_queue,
                    sdar_stream_source=sdar_stream_source,
                    sdar_stream_context=sdar_stream_context,
                )
            else:
                output = await self.tokenizer_manager.generate_request(generate_request, None).__anext__()
            request_generate_seconds += time.perf_counter() - request_generate_start
        except ValueError as exc:
            if _batch_input_ids_enabled(self.config) or "input_ids should be a list of lists" not in str(exc):
                raise
            request["input_ids"] = [prompt_ids]
            request_build_start = time.perf_counter()
            generate_request = self._io_struct.GenerateReqInput(**request)
            request_build_seconds += time.perf_counter() - request_build_start
            request_generate_start = time.perf_counter()
            if should_stream_sdar_replay:
                output = await self._consume_streaming_generate_request(
                    generate_request,
                    request_id=request_id,
                    prompt_ids=prompt_ids,
                    sdar_stream_queue=sdar_stream_queue,
                    sdar_stream_source=sdar_stream_source,
                    sdar_stream_context=sdar_stream_context,
                )
            else:
                output = await self.tokenizer_manager.generate_request(generate_request, None).__anext__()
            request_generate_seconds += time.perf_counter() - request_generate_start
        emit_lifecycle(
            "SGLangHttpServer",
            "generate_request_done",
            engine_uuid=self._lifecycle_engine_uuid,
            node_rank=int(self.node_rank),
            replica_rank=int(self.replica_rank),
            request_id=str(request_id),
            request_generate_seconds=float(request_generate_seconds),
            memory=lifecycle_memory("after_generate_request"),
        )
        unwrap_start = time.perf_counter()
        output = self._unwrap_single_sglang_output(output)
        unwrap_seconds = time.perf_counter() - unwrap_start
        stats_after_start = time.perf_counter()
        stats_after = await self._get_dllm_stats_snapshot()
        stats_after_seconds = time.perf_counter() - stats_after_start
        stats_diff_start = time.perf_counter()
        engine_kwargs = self.config.get("engine_kwargs", {}) or {}
        block_size = int(engine_kwargs.get("block_length", 256) or 256)
        stats_delta = _diff_dllm_stats(stats_after, stats_before, block_size)
        stats_diff_seconds = time.perf_counter() - stats_diff_start
        meta_info = output.get("meta_info", {}) or {}
        output_ids = output["output_ids"]
        if _is_dllm_rollout(self.config) and len(output_ids) > int(self.config.response_length):
            raise RuntimeError(
                "SGLang DLLM returned more output tokens than rollout.response_length: "
                f"{len(output_ids)} > {self.config.response_length}; "
                f"{max_new_tokens=}, {meta_info.get('finish_reason')=}"
            )
        rollout_meta_info = {
            key: meta_info[key]
            for key in (
                "id",
                "finish_reason",
                "prompt_tokens",
                "completion_tokens",
                "cached_tokens",
                "e2e_latency",
                "decode_throughput",
                "time_to_first_token",
                "time_to_last_token",
                "queue_time",
                "sglang_sdar_request_metrics",
            )
            if key in meta_info
        }
        replay_tensors = None if should_stream_sdar_replay else meta_info.get("sglang_sdar_replay_tensors")
        gemma_replay = None if should_stream_sdar_replay else meta_info.get("gemma_intermediate_replay")
        replay_source_rows = len(self._as_sdar_replay_items(replay_tensors))
        if replay_tensors is not None and gemma_replay is None:
            replay_tensors, replay_source_rows = self._preselect_qwen_replay_items(
                replay_tensors
            )
        if self._is_gemma_intermediate_replay(gemma_replay):
            gemma_replay = [
                self._bind_gemma_intermediate_replay_item(
                    item,
                    request_id=request_id,
                    prompt_ids=prompt_ids,
                    output_ids=output_ids,
                    stream_context=sdar_stream_context,
                    response_length=int(self.config.response_length),
                    response_finished=True,
                )
                for item in self._as_sdar_replay_items(gemma_replay)
            ]
        stored_replay = gemma_replay if gemma_replay is not None else replay_tensors
        if stored_replay:
            self._sdar_replay_tensor_store[str(request_id)] = stored_replay
            rollout_meta_info["sglang_sdar_replay_handle"] = {
                "request_id": str(request_id),
                "num_tensors": len(stored_replay),
                "source_num_tensors": int(replay_source_rows),
                "schema_format": (
                    "gemma_intermediate_replay_v1"
                    if self._is_gemma_intermediate_replay(stored_replay)
                    else "sglang_sdar_replay_tensor_batch_v1"
                ),
            }
            rollout_meta_info["sglang_sdar_replay_source_rows"] = int(
                replay_source_rows
            )
            rollout_meta_info["sglang_sdar_replay_selected_rows"] = int(
                len(stored_replay)
            )
        replay_trace = self._extract_sdar_replay_trace(meta_info)
        if replay_trace is not None:
            rollout_meta_info["replay_trace"] = replay_trace
        rollout_meta_info["requested_max_new_tokens"] = int(max_new_tokens)
        rollout_meta_info["raw_output_token_count"] = int(len(output_ids))
        rollout_meta_info["verl_sglang_generate_total_seconds"] = float(time.perf_counter() - generate_total_start)
        rollout_meta_info["verl_sglang_stats_before_seconds"] = float(stats_before_seconds)
        rollout_meta_info["verl_sglang_request_build_seconds"] = float(request_build_seconds)
        rollout_meta_info["verl_sglang_request_generate_seconds"] = float(request_generate_seconds)
        rollout_meta_info["verl_sglang_unwrap_seconds"] = float(unwrap_seconds)
        rollout_meta_info["verl_sglang_stats_after_seconds"] = float(stats_after_seconds)
        rollout_meta_info["verl_sglang_stats_diff_seconds"] = float(stats_diff_seconds)
        rollout_meta_info["verl_sglang_stream_replay_enabled"] = float(bool(should_stream_sdar_replay))
        rollout_meta_info["verl_sglang_batch_input_ids_enabled"] = float(bool(_batch_input_ids_enabled(self.config)))
        rollout_meta_info["verl_sglang_return_logprob"] = float(bool(return_logprob))
        gemma_stream_metrics = self._gemma_replay_stream_metrics.pop(
            str(request_id), None
        )
        if gemma_stream_metrics is not None:
            expected_stream_blocks = (
                (int(len(output_ids)) + block_size - 1) // block_size
                if output_ids
                else 0
            )
            pushed_block_indices = sorted(
                int(index)
                for index in gemma_stream_metrics.get(
                    "pushed_block_indices", []
                )
            )
            expected_block_indices = list(range(expected_stream_blocks))
            if pushed_block_indices != expected_block_indices:
                raise RuntimeError(
                    "DiffusionGemma streaming replay identity mismatch: "
                    f"request_id={request_id} "
                    f"expected_blocks={expected_block_indices} "
                    f"pushed_blocks={pushed_block_indices}"
                )
            rollout_meta_info[
                "gemma_intermediate_replay_stream_metrics"
            ] = dict(gemma_stream_metrics)
        if stats_delta:
            rollout_meta_info["sglang_sdar_global_stats_delta"] = stats_delta

        if return_logprob:
            output_token_logprobs = meta_info.get("output_token_logprobs") or []
            if output_token_logprobs:
                log_probs, token_ids = zip(
                    *[(log_prob, token_ids) for log_prob, token_ids, _ in output_token_logprobs], strict=True
                )
                if _is_dllm_rollout(self.config):
                    token_ids = list(token_ids)
                    log_probs = list(log_probs)
                    if token_ids != list(output_ids):
                        raise RuntimeError(
                            "SGLang DLLM output_token_logprobs are not aligned with output_ids: "
                            f"{len(token_ids)=}, {len(output_ids)=}"
                        )
                    if not all(math.isfinite(float(log_prob)) for log_prob in log_probs):
                        raise RuntimeError("SGLang DLLM output_token_logprobs contain non-finite values.")
            elif _is_dllm_rollout(self.config):
                raise RuntimeError("SGLang DLLM rollout returned no output_token_logprobs despite return_logprob=True")
            else:
                raise RuntimeError("SGLang returned no output_token_logprobs despite return_logprob=True")
        else:
            token_ids = output["output_ids"]
            log_probs = None

        routed_experts = None
        if self.config.enable_rollout_routing_replay:
            if self.config.skip_tokenizer_init:
                routed_experts = meta_info.get("routed_experts", None)
            else:
                from sglang.srt.layers.moe.routed_experts_capturer import extract_routed_experts_from_meta_info

                hf_config = self.model_config.hf_config
                if not hasattr(hf_config, "num_hidden_layers") or not hasattr(hf_config, "num_experts_per_tok"):
                    raise AttributeError(
                        "enable_rollout_routing_replay is set, but hf_config is missing "
                        "'num_hidden_layers' or 'num_experts_per_tok'. This feature requires an MoE model "
                        "configuration that defines these attributes."
                    )
                routed_experts = extract_routed_experts_from_meta_info(output).reshape(
                    -1, hf_config.num_hidden_layers, hf_config.num_experts_per_tok
                )

        emit_lifecycle(
            "SGLangHttpServer",
            "generate_done",
            engine_uuid=self._lifecycle_engine_uuid,
            node_rank=int(self.node_rank),
            replica_rank=int(self.replica_rank),
            request_id=str(request_id),
            output_tokens=int(len(token_ids)),
            total_seconds=float(time.perf_counter() - generate_total_start),
            request_generate_seconds=float(request_generate_seconds),
            stats_after_seconds=float(stats_after_seconds),
            memory=lifecycle_memory("generate_done"),
        )
        return TokenOutput(
            token_ids=token_ids,
            log_probs=log_probs,
            meta_info=rollout_meta_info,
            routed_experts=routed_experts,
        )

    async def generate_batch(
        self,
        requests: list[dict[str, Any]],
    ) -> list[TokenOutput]:
        """Generate for multiple non-streaming, non-multimodal requests in a single SGLang call.

        Each dict in *requests* must have:
          - ``request_id``     (str)
          - ``prompt_ids``     (list[int])
          - ``sampling_params`` (dict) — may contain ``max_new_tokens``/``max_tokens``/``logprobs``

        Returns a list of :class:`TokenOutput` in the same order as *requests*.
        This method is the server-side half of the VERL_SGLANG_BATCH_RPC optimisation.
        """
        if not requests:
            return []

        batch_total_start = time.perf_counter()

        request_ids: list[str] = []
        norm_prompt_ids: list[list[int]] = []
        norm_sampling_params: list[dict[str, Any]] = []
        return_logprobs: list[bool] = []
        max_new_tokens_list: list[int] = []

        for req in requests:
            request_id = str(req["request_id"])
            prompt_ids = self._normalize_prompt_ids(req["prompt_ids"])
            sampling_params = dict(req["sampling_params"])

            max_possible_tokens = self.config.max_model_len - len(prompt_ids)
            if max_possible_tokens < 0:
                raise ValueError(
                    f"Prompt length ({len(prompt_ids)}) exceeds model max context "
                    f"({self.config.max_model_len}) for request_id={request_id}"
                )

            if "max_new_tokens" in sampling_params:
                max_new_tokens = sampling_params.pop("max_new_tokens")
            elif "max_tokens" in sampling_params:
                max_new_tokens = sampling_params.pop("max_tokens")
            elif _is_dllm_rollout(self.config):
                max_new_tokens = self.config.response_length
            else:
                max_new_tokens = self.config.response_length + self.config.prompt_length - len(prompt_ids)

            max_new_tokens = max(0, min(int(max_new_tokens), max_possible_tokens))
            sampling_params["max_new_tokens"] = max_new_tokens
            _apply_dllm_stop_token_ids(self, sampling_params)
            return_logprob = bool(sampling_params.pop("logprobs", False))

            request_ids.append(request_id)
            norm_prompt_ids.append(prompt_ids)
            norm_sampling_params.append(sampling_params)
            return_logprobs.append(return_logprob)
            max_new_tokens_list.append(max_new_tokens)

        batch_req_dict: dict[str, Any] = {
            "rid": request_ids,
            "input_ids": norm_prompt_ids,
            "sampling_params": norm_sampling_params,
            "return_logprob": return_logprobs,
        }

        request_build_seconds = time.perf_counter() - batch_total_start

        emit_lifecycle(
            "SGLangHttpServer",
            "generate_batch_start",
            engine_uuid=self._lifecycle_engine_uuid,
            node_rank=int(self.node_rank),
            replica_rank=int(self.replica_rank),
            batch_size=len(requests),
        )

        request_generate_start = time.perf_counter()
        batch_req = self._io_struct.GenerateReqInput(**batch_req_dict)
        raw_outputs = await self.tokenizer_manager.generate_request(batch_req, None).__anext__()
        request_generate_seconds = time.perf_counter() - request_generate_start

        if not isinstance(raw_outputs, list):
            raw_outputs = [raw_outputs]
        if len(raw_outputs) != len(requests):
            raise ValueError(
                f"SGLang generate_batch returned {len(raw_outputs)} outputs "
                f"for {len(requests)} requests"
            )

        emit_lifecycle(
            "SGLangHttpServer",
            "generate_batch_done",
            engine_uuid=self._lifecycle_engine_uuid,
            node_rank=int(self.node_rank),
            replica_rank=int(self.replica_rank),
            batch_size=len(requests),
            request_generate_seconds=float(request_generate_seconds),
        )

        results: list[TokenOutput] = []
        for i, raw_output in enumerate(raw_outputs):
            request_id = request_ids[i]
            prompt_ids = norm_prompt_ids[i]
            max_new_tokens = max_new_tokens_list[i]
            return_logprob = return_logprobs[i]

            output: dict[str, Any] = raw_output if isinstance(raw_output, dict) else dict(raw_output)
            meta_info: dict[str, Any] = output.get("meta_info", {}) or {}
            output_ids = self._normalize_output_ids(output.get("output_ids", []))

            if _is_dllm_rollout(self.config) and len(output_ids) > int(self.config.response_length):
                raise RuntimeError(
                    f"SGLang DLLM batch returned more output tokens than rollout.response_length for "
                    f"request_id={request_id}: {len(output_ids)} > {self.config.response_length}"
                )

            rollout_meta_info: dict[str, Any] = {
                key: meta_info[key]
                for key in (
                    "id",
                    "finish_reason",
                    "prompt_tokens",
                    "completion_tokens",
                    "cached_tokens",
                    "e2e_latency",
                    "decode_throughput",
                    "time_to_first_token",
                    "time_to_last_token",
                    "queue_time",
                    "sglang_sdar_request_metrics",
                )
                if key in meta_info
            }

            # Non-streaming SDAR replay tensors — store per-request so
            # fetch_sdar_replay_tensors still works unchanged.
            replay_tensors = meta_info.get("sglang_sdar_replay_tensors")
            gemma_replay = meta_info.get("gemma_intermediate_replay")
            replay_source_rows = len(self._as_sdar_replay_items(replay_tensors))
            if replay_tensors is not None and gemma_replay is None:
                replay_tensors, replay_source_rows = self._preselect_qwen_replay_items(replay_tensors)
            if self._is_gemma_intermediate_replay(gemma_replay):
                gemma_replay = [
                    self._bind_gemma_intermediate_replay_item(
                        item,
                        request_id=request_id,
                        prompt_ids=prompt_ids,
                        output_ids=output_ids,
                        stream_context=None,
                        response_length=int(self.config.response_length),
                        response_finished=True,
                    )
                    for item in self._as_sdar_replay_items(gemma_replay)
                ]
            stored_replay = gemma_replay if gemma_replay is not None else replay_tensors
            if stored_replay:
                self._sdar_replay_tensor_store[str(request_id)] = stored_replay
                rollout_meta_info["sglang_sdar_replay_handle"] = {
                    "request_id": str(request_id),
                    "num_tensors": len(stored_replay),
                    "source_num_tensors": int(replay_source_rows),
                    "schema_format": (
                        "gemma_intermediate_replay_v1"
                        if self._is_gemma_intermediate_replay(stored_replay)
                        else "sglang_sdar_replay_tensor_batch_v1"
                    ),
                }
                rollout_meta_info["sglang_sdar_replay_source_rows"] = int(replay_source_rows)
                rollout_meta_info["sglang_sdar_replay_selected_rows"] = int(len(stored_replay))

            replay_trace = self._extract_sdar_replay_trace(meta_info)
            if replay_trace is not None:
                rollout_meta_info["replay_trace"] = replay_trace

            rollout_meta_info["requested_max_new_tokens"] = int(max_new_tokens)
            rollout_meta_info["raw_output_token_count"] = int(len(output_ids))
            rollout_meta_info["verl_sglang_batch_rpc"] = True
            rollout_meta_info["verl_sglang_batch_size"] = int(len(requests))
            rollout_meta_info["verl_sglang_request_build_seconds"] = float(request_build_seconds / len(requests))
            rollout_meta_info["verl_sglang_request_generate_seconds"] = float(request_generate_seconds / len(requests))

            if return_logprob:
                output_token_logprobs = meta_info.get("output_token_logprobs") or []
                if output_token_logprobs:
                    log_probs_vals, token_ids = zip(
                        *[(lp, tid) for lp, tid, _ in output_token_logprobs], strict=True
                    )
                    token_ids = list(token_ids)
                    log_probs_vals = list(log_probs_vals)
                    if _is_dllm_rollout(self.config):
                        if token_ids != list(output_ids):
                            raise RuntimeError(
                                f"SGLang DLLM batch output_token_logprobs misaligned with output_ids "
                                f"for request_id={request_id}: "
                                f"{len(token_ids)=}, {len(output_ids)=}"
                            )
                        if not all(math.isfinite(float(lp)) for lp in log_probs_vals):
                            raise RuntimeError(
                                f"SGLang DLLM batch output_token_logprobs contain non-finite values "
                                f"for request_id={request_id}"
                            )
                else:
                    raise RuntimeError(
                        f"SGLang batch returned no output_token_logprobs despite return_logprob=True "
                        f"for request_id={request_id}"
                    )
            else:
                token_ids = output_ids
                log_probs_vals = None

            routed_experts = None
            if self.config.enable_rollout_routing_replay:
                # skip_tokenizer_init path: routed_experts already in meta_info
                if self.config.skip_tokenizer_init:
                    routed_experts = meta_info.get("routed_experts", None)

            results.append(
                TokenOutput(
                    token_ids=token_ids,
                    log_probs=log_probs_vals,
                    meta_info=rollout_meta_info,
                    routed_experts=routed_experts,
                )
            )

        return results

    @staticmethod
    def _compact_sdar_replay_metadata(items: list[dict]) -> dict[str, Any]:
        if not items:
            return {}
        metadata = [
            item.get("metadata", item) if isinstance(item.get("metadata", item), dict) else {}
            for item in items
        ]
        block_tpf_summary_available = [
            meta.get("new_accept_tokens") is not None
            and meta.get("denoise_forwards") is not None
            for meta in metadata
        ]
        positions = [item.get("position", meta.get("position", (0, 0))) for item, meta in zip(items, metadata, strict=True)]
        positions = [
            tuple(int(x) for x in (pos.detach().cpu().tolist() if isinstance(pos, torch.Tensor) else pos))
            if isinstance(pos, (tuple, list, torch.Tensor)) and len(pos) == 2
            else (0, 0)
            for pos in positions
        ]
        compact = {
            "format": "sglang_sdar_replay_metadata_batch_v1",
            "step_keys": torch.tensor(
                [int(meta.get("step_key", idx)) for idx, meta in enumerate(metadata)],
                dtype=torch.long,
            ),
            "local_steps": torch.tensor([int(meta.get("local_step", 0)) for meta in metadata], dtype=torch.long),
            "block_indices": torch.tensor([int(meta.get("block_idx", 0)) for meta in metadata], dtype=torch.long),
            "req_pool_indices": torch.tensor([int(meta.get("req_pool_idx", -1)) for meta in metadata], dtype=torch.long),
            "start_offsets": torch.tensor([int(meta.get("start_offset", 0)) for meta in metadata], dtype=torch.long),
            "response_start_indices": torch.tensor(
                [int(meta.get("response_start_index", 0)) for meta in metadata],
                dtype=torch.long,
            ),
            "new_accept_counts": torch.tensor([int(meta.get("new_accept_count", 0)) for meta in metadata], dtype=torch.long),
            "new_accept_tokens": torch.tensor(
                [int(meta.get("new_accept_tokens", 0)) for meta in metadata],
                dtype=torch.long,
            ),
            "denoise_forwards": torch.tensor(
                [int(meta.get("denoise_forwards", 0)) for meta in metadata],
                dtype=torch.long,
            ),
            "block_tpf_summary_available": torch.tensor(
                block_tpf_summary_available,
                dtype=torch.bool,
            ),
            "action_token_counts": torch.tensor([int(meta.get("action_token_count", 0)) for meta in metadata], dtype=torch.long),
            "is_final_denoise_rows": torch.tensor(
                [bool(meta.get("is_final_denoise_row", False)) for meta in metadata],
                dtype=torch.bool,
            ),
            "is_terminal_commits": torch.tensor(
                [bool(meta.get("is_terminal_commit", False)) for meta in metadata],
                dtype=torch.bool,
            ),
            "producer_preselected_rows": torch.tensor(
                [bool(meta.get("producer_preselected", False)) for meta in metadata],
                dtype=torch.bool,
            ),
            "positions": torch.tensor(positions, dtype=torch.long),
            "tensor_paths": [str(meta.get("tensor_path", "")) for meta in metadata],
            "trace_roots": [str(meta.get("trace_root", "")) for meta in metadata],
            "tensor_file_bytes": torch.tensor(
                [int(meta.get("tensor_file_bytes", meta.get("file_bytes", 0)) or 0) for meta in metadata],
                dtype=torch.long,
            ),
        }
        compact.update(
            _compact_producer_state_selection_metadata(metadata)
        )
        action_ids = [meta.get("action_ids") for meta in metadata]
        action_mask = [meta.get("action_mask") for meta in metadata]
        if all(isinstance(value, (list, tuple, torch.Tensor)) for value in action_ids):
            compact["action_ids"] = torch.stack([torch.as_tensor(value, dtype=torch.long) for value in action_ids], dim=0)
        if all(isinstance(value, (list, tuple, torch.Tensor)) for value in action_mask):
            compact["action_mask"] = torch.stack([torch.as_tensor(value, dtype=torch.bool) for value in action_mask], dim=0)

        # Adjacent entropy contraction scalar fields (metadata-only path).
        _aec_pair_valid_m = [meta.get("adjacent_entropy_pair_valid") for meta in metadata]
        if all(v is not None for v in _aec_pair_valid_m):
            compact["adjacent_entropy_pair_valid"] = torch.tensor(
                [bool(v) for v in _aec_pair_valid_m], dtype=torch.bool
            )
            compact["adjacent_entropy_pair_local_step"] = torch.tensor(
                [int(meta.get("adjacent_entropy_pair_local_step", -1)) for meta in metadata],
                dtype=torch.long,
            )
            compact["adjacent_entropy_prev_step_key"] = torch.tensor(
                [int(meta.get("adjacent_entropy_prev_step_key", -1)) for meta in metadata],
                dtype=torch.long,
            )
            compact["adjacent_entropy_prev_mean"] = torch.tensor(
                [float(meta.get("adjacent_entropy_prev_mean", 0.0)) for meta in metadata],
                dtype=torch.float32,
            )
            compact["adjacent_entropy_rollout_next_mean"] = torch.tensor(
                [float(meta.get("adjacent_entropy_rollout_next_mean", 0.0)) for meta in metadata],
                dtype=torch.float32,
            )
            compact["adjacent_entropy_vocab_size"] = torch.tensor(
                [int(meta.get("adjacent_entropy_vocab_size", 0)) for meta in metadata],
                dtype=torch.long,
            )
            _aec_next_masks_m = [meta.get("adjacent_entropy_next_mask") for meta in metadata]
            if all(v is not None for v in _aec_next_masks_m):
                try:
                    compact["adjacent_entropy_next_mask"] = torch.stack(
                        [torch.as_tensor(m, dtype=torch.bool).detach().cpu() for m in _aec_next_masks_m], dim=0
                    )
                except Exception:
                    pass

        # Global/all-valid adjacent entropy contraction fields.  These are
        # intentionally separate from the unresolved/random-vocabulary fields
        # above because both annotations may coexist on the same replay row.
        _global_aec_pair_valid_m = [
            meta.get("global_aec_pair_valid") for meta in metadata
        ]
        if all(v is not None for v in _global_aec_pair_valid_m):
            compact["global_aec_pair_valid"] = torch.tensor(
                [bool(v) for v in _global_aec_pair_valid_m],
                dtype=torch.bool,
            )
            compact["global_aec_prev_step_key"] = torch.tensor(
                [
                    int(meta.get("global_aec_prev_step_key", -1))
                    for meta in metadata
                ],
                dtype=torch.long,
            )
            compact["global_aec_prev_mean"] = torch.tensor(
                [
                    float(meta.get("global_aec_prev_mean", 0.0))
                    for meta in metadata
                ],
                dtype=torch.float32,
            )
            compact["global_aec_rollout_next_mean"] = torch.tensor(
                [
                    float(
                        meta.get(
                            "global_aec_rollout_next_mean",
                            0.0,
                        )
                    )
                    for meta in metadata
                ],
                dtype=torch.float32,
            )
            compact["global_aec_vocab_size"] = torch.tensor(
                [
                    int(meta.get("global_aec_vocab_size", 0))
                    for meta in metadata
                ],
                dtype=torch.long,
            )
            _global_aec_next_masks_m = [
                meta.get("global_aec_next_mask") for meta in metadata
            ]
            if all(v is not None for v in _global_aec_next_masks_m):
                try:
                    compact["global_aec_next_mask"] = torch.stack(
                        [
                            torch.as_tensor(
                                mask,
                                dtype=torch.bool,
                            )
                            .detach()
                            .cpu()
                            for mask in _global_aec_next_masks_m
                        ],
                        dim=0,
                    )
                except Exception:
                    pass

        return compact

    @staticmethod
    def _compact_sdar_replay_tensors(items: list[dict]) -> dict[str, Any]:
        """Pack per-step replay dictionaries into a compact tensor batch.

        This keeps the full replay algorithm unchanged while avoiding a large
        list of small Python dictionaries crossing process boundaries.
        """
        if not items:
            return {}

        def _stack_tensor(key: str):
            values = []
            for item in items:
                value = item.get(key)
                if value is None and key == "prompt_mask":
                    metadata = (
                        item.get("metadata", item)
                        if isinstance(item.get("metadata", item), dict)
                        else {}
                    )
                    value = metadata.get("prompt_mask")
                    if value is not None and not isinstance(value, torch.Tensor):
                        value = torch.as_tensor(value, dtype=torch.bool)
                values.append(value)
            if not all(isinstance(value, torch.Tensor) for value in values):
                return None
            return torch.stack([value.detach().cpu() for value in values], dim=0)

        # Memory/table replay rows are themselves metadata dictionaries, while
        # file-backed rows carry the same fields in item["metadata"]. Support
        # both layouts; using {} for the top-level form silently discarded step
        # keys and adjacent-entropy annotations during table materialization.
        metadata = [
            (
                item.get("metadata", item)
                if isinstance(item.get("metadata", item), dict)
                else {}
            )
            for item in items
        ]
        block_tpf_summary_available = [
            meta.get("new_accept_tokens") is not None
            and meta.get("denoise_forwards") is not None
            for meta in metadata
        ]
        step_keys = [int(meta.get("step_key", idx)) for idx, meta in enumerate(metadata)]
        local_steps = [int(meta.get("local_step", 0)) for meta in metadata]
        block_indices = [int(meta.get("block_idx", 0)) for meta in metadata]
        req_pool_indices = [int(meta.get("req_pool_idx", -1)) for meta in metadata]
        start_offsets = [int(meta.get("start_offset", 0)) for meta in metadata]
        response_start_indices = [int(meta.get("response_start_index", 0)) for meta in metadata]
        new_accept_counts = [int(meta.get("new_accept_count", 0)) for meta in metadata]
        new_accept_tokens = [int(meta.get("new_accept_tokens", 0)) for meta in metadata]
        denoise_forwards = [int(meta.get("denoise_forwards", 0)) for meta in metadata]
        action_token_counts = [int(meta.get("action_token_count", 0)) for meta in metadata]
        is_final_denoise_rows = [bool(meta.get("is_final_denoise_row", False)) for meta in metadata]
        is_terminal_commits = [bool(meta.get("is_terminal_commit", False)) for meta in metadata]
        producer_preselected_rows = [
            bool(meta.get("producer_preselected", False)) for meta in metadata
        ]
        positions = [item.get("position", meta.get("position", (0, 0))) for item, meta in zip(items, metadata, strict=True)]
        positions = [
            tuple(int(x) for x in (pos.detach().cpu().tolist() if isinstance(pos, torch.Tensor) else pos))
            if isinstance(pos, (tuple, list, torch.Tensor)) and len(pos) == 2
            else (0, 0)
            for pos in positions
        ]

        compact: dict[str, Any] = {
            "format": "sglang_sdar_replay_tensor_batch_v1",
            "step_keys": torch.tensor(step_keys, dtype=torch.long),
            "local_steps": torch.tensor(local_steps, dtype=torch.long),
            "block_indices": torch.tensor(block_indices, dtype=torch.long),
            "req_pool_indices": torch.tensor(req_pool_indices, dtype=torch.long),
            "start_offsets": torch.tensor(start_offsets, dtype=torch.long),
            "response_start_indices": torch.tensor(response_start_indices, dtype=torch.long),
            "new_accept_counts": torch.tensor(new_accept_counts, dtype=torch.long),
            "new_accept_tokens": torch.tensor(new_accept_tokens, dtype=torch.long),
            "denoise_forwards": torch.tensor(denoise_forwards, dtype=torch.long),
            "block_tpf_summary_available": torch.tensor(
                block_tpf_summary_available,
                dtype=torch.bool,
            ),
            "action_token_counts": torch.tensor(action_token_counts, dtype=torch.long),
            "is_final_denoise_rows": torch.tensor(is_final_denoise_rows, dtype=torch.bool),
            "is_terminal_commits": torch.tensor(is_terminal_commits, dtype=torch.bool),
            "producer_preselected_rows": torch.tensor(
                producer_preselected_rows,
                dtype=torch.bool,
            ),
            "positions": torch.tensor(positions, dtype=torch.long),
        }
        compact.update(
            _compact_producer_state_selection_metadata(metadata)
        )
        for out_key, item_key in (
            ("valid", "valid"),
            ("action_ids", "action_ids"),
            ("action_mask", "action_mask"),
            ("new_accept_mask", "new_accept_mask"),
            ("edit_change_mask", "edit_change_mask"),
            ("replay_active_mask", "replay_active_mask"),
            ("random_vocab_mask", "random_vocab_mask"),
            ("prompt_mask", "prompt_mask"),
            ("input_ids", "input_ids"),
            ("self_conditioning_signal", "self_conditioning_signal"),
            ("logits_source_hidden", "logits_source_hidden"),
        ):
            value = _stack_tensor(item_key)
            if value is not None:
                compact[out_key] = value
        from sglang.srt.dllm.algorithm.sdar_gpu_replay_lease import (
            is_gpu_replay_lease,
        )

        for item_key in (
            "self_conditioning_signal",
            "logits_source_hidden",
        ):
            lease_values = [item.get(item_key) for item in items]
            if all(
                is_gpu_replay_lease(value)
                for value in lease_values
            ):
                compact[f"{item_key}_gpu_leases"] = lease_values
            elif any(
                is_gpu_replay_lease(value)
                for value in lease_values
            ):
                raise RuntimeError(
                    "mixed CUDA IPC lease and tensor replay rows for "
                    f"{item_key}"
                )

        # Adjacent entropy contraction scalars (from metadata) and next_mask tensor.
        # Fields are present only when adjacent_entropy_contraction is enabled on
        # the producer.  If any item lacks them the whole block is skipped so the
        # disabled path is unchanged.
        _aec_pair_valid = [
            meta.get("adjacent_entropy_pair_valid")
            for meta in metadata
        ]
        if all(v is not None for v in _aec_pair_valid):
            compact["adjacent_entropy_pair_valid"] = torch.tensor(
                [bool(v) for v in _aec_pair_valid], dtype=torch.bool
            )
            compact["adjacent_entropy_pair_local_step"] = torch.tensor(
                [int(meta.get("adjacent_entropy_pair_local_step", -1)) for meta in metadata],
                dtype=torch.long,
            )
            compact["adjacent_entropy_prev_step_key"] = torch.tensor(
                [int(meta.get("adjacent_entropy_prev_step_key", -1)) for meta in metadata],
                dtype=torch.long,
            )
            compact["adjacent_entropy_prev_mean"] = torch.tensor(
                [float(meta.get("adjacent_entropy_prev_mean", 0.0)) for meta in metadata],
                dtype=torch.float32,
            )
            compact["adjacent_entropy_rollout_next_mean"] = torch.tensor(
                [float(meta.get("adjacent_entropy_rollout_next_mean", 0.0)) for meta in metadata],
                dtype=torch.float32,
            )
            compact["adjacent_entropy_vocab_size"] = torch.tensor(
                [int(meta.get("adjacent_entropy_vocab_size", 0)) for meta in metadata],
                dtype=torch.long,
            )
            # next_mask: try item-level tensor first (memory mode), then metadata list
            _aec_next_masks = []
            _all_valid = True
            for item, meta in zip(items, metadata, strict=True):
                nm = item.get("adjacent_entropy_next_mask")
                if nm is None:
                    nm = meta.get("adjacent_entropy_next_mask")
                if nm is None:
                    _all_valid = False
                    break
                if not isinstance(nm, torch.Tensor):
                    try:
                        nm = torch.as_tensor(nm, dtype=torch.bool)
                    except Exception:
                        _all_valid = False
                        break
                _aec_next_masks.append(nm.bool())
            if _all_valid and _aec_next_masks:
                try:
                    compact["adjacent_entropy_next_mask"] = torch.stack(
                        [m.detach().cpu() for m in _aec_next_masks], dim=0
                    )
                except Exception:
                    pass  # shape mismatch: skip tensor; scalars still transported

        # Global/all-valid AEC fields use the same compact transport contract,
        # but remain independently gated so unresolved-only producers are
        # unchanged.
        _global_aec_pair_valid = [
            meta.get("global_aec_pair_valid") for meta in metadata
        ]
        if all(v is not None for v in _global_aec_pair_valid):
            compact["global_aec_pair_valid"] = torch.tensor(
                [bool(v) for v in _global_aec_pair_valid],
                dtype=torch.bool,
            )
            compact["global_aec_prev_step_key"] = torch.tensor(
                [
                    int(meta.get("global_aec_prev_step_key", -1))
                    for meta in metadata
                ],
                dtype=torch.long,
            )
            compact["global_aec_prev_mean"] = torch.tensor(
                [
                    float(meta.get("global_aec_prev_mean", 0.0))
                    for meta in metadata
                ],
                dtype=torch.float32,
            )
            compact["global_aec_rollout_next_mean"] = torch.tensor(
                [
                    float(
                        meta.get(
                            "global_aec_rollout_next_mean",
                            0.0,
                        )
                    )
                    for meta in metadata
                ],
                dtype=torch.float32,
            )
            compact["global_aec_vocab_size"] = torch.tensor(
                [
                    int(meta.get("global_aec_vocab_size", 0))
                    for meta in metadata
                ],
                dtype=torch.long,
            )
            _global_aec_next_masks = []
            _global_all_valid = True
            for item, meta in zip(items, metadata, strict=True):
                next_mask = item.get("global_aec_next_mask")
                if next_mask is None:
                    next_mask = meta.get("global_aec_next_mask")
                if next_mask is None:
                    _global_all_valid = False
                    break
                if not isinstance(next_mask, torch.Tensor):
                    try:
                        next_mask = torch.as_tensor(
                            next_mask,
                            dtype=torch.bool,
                        )
                    except Exception:
                        _global_all_valid = False
                        break
                _global_aec_next_masks.append(next_mask.bool())
            if _global_all_valid and _global_aec_next_masks:
                try:
                    compact["global_aec_next_mask"] = torch.stack(
                        [
                            mask.detach().cpu()
                            for mask in _global_aec_next_masks
                        ],
                        dim=0,
                    )
                except Exception:
                    pass

        return compact

    @staticmethod
    def _tensor_to_shm_handle(tensor: torch.Tensor) -> dict[str, Any]:
        tensor = tensor.detach().cpu().contiguous()
        raw = tensor.view(torch.uint8).numpy().reshape(-1)
        shm = SharedMemory(create=True, size=int(raw.nbytes))
        success = False
        try:
            shm_arr = np.ndarray((raw.nbytes,), dtype=np.uint8, buffer=shm.buf)
            shm_arr[:] = raw
            handle = {
                "name": shm.name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "nbytes": int(raw.nbytes),
            }
            success = True
            return handle
        finally:
            if not success:
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
                except Exception:
                    logger.warning("Unable to unlink failed SDAR replay shm name=%s", shm.name, exc_info=True)
            shm.close()

    @classmethod
    def _compact_replay_to_shm_handles(cls, compact: dict[str, Any]) -> dict[str, Any]:
        if not compact:
            return {}
        handles: dict[str, Any] = {
            "format": "sglang_sdar_replay_shm_batch_v1",
            "tensors": {},
        }
        try:
            for key, value in compact.items():
                if isinstance(value, torch.Tensor):
                    handles["tensors"][key] = cls._tensor_to_shm_handle(value)
                elif key not in {"format", "tensors"}:
                    handles[key] = value
            return handles
        except Exception:
            for handle in handles.get("tensors", {}).values():
                if not isinstance(handle, dict) or "name" not in handle:
                    continue
                try:
                    shm = SharedMemory(name=str(handle["name"]))
                except FileNotFoundError:
                    continue
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
                except Exception:
                    logger.warning(
                        "Unable to unlink partial SDAR replay shm name=%s",
                        handle.get("name"),
                        exc_info=True,
                    )
                finally:
                    shm.close()
            raise

    async def fetch_sdar_replay_tensors(self, request_id: str, cleanup: bool = True) -> dict[str, Any]:
        request_id = str(request_id)
        if cleanup:
            items = self._sdar_replay_tensor_store.pop(request_id, [])
        else:
            items = self._sdar_replay_tensor_store.get(request_id, [])
        if self._is_gemma_intermediate_replay(items):
            # Native DiffusionGemma replay never uses Qwen GPU-lease
            # descriptors. Keep it in Ray tensor/object transport and avoid
            # importing producer-specific Qwen modules from the external
            # Gemma SGLang checkout.
            if (
                isinstance(items, list)
                and len(items) == 1
                and isinstance(items[0], dict)
                and items[0].get("format")
                == "gemma_intermediate_replay_v1"
            ):
                return items[0]
            return items
        if isinstance(items, list):
            from sglang.srt.dllm.algorithm.sdar_gpu_replay_lease import (
                is_gpu_replay_lease,
            )

            for item in items:
                if not isinstance(item, dict):
                    continue
                for key in (
                    "self_conditioning_signal",
                    "logits_source_hidden",
                ):
                    descriptor = item.get(key)
                    if is_gpu_replay_lease(descriptor):
                        descriptor = dict(descriptor)
                        descriptor["producer_replica_rank"] = int(
                            self.replica_rank
                        )
                        item[key] = descriptor
        metadata_first = os.environ.get(_SGLANG_SDAR_TPF_METADATA_FIRST_ENV, "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        has_tensor_paths = isinstance(items, list) and any(
            isinstance(item, dict)
            and (item.get("tensor_path") or (isinstance(item.get("metadata"), dict) and item["metadata"].get("tensor_path")))
            for item in items
        )
        loaded_paths: list[str] = []
        if metadata_first and has_tensor_paths:
            compact = self._compact_sdar_replay_metadata(items)
        elif has_tensor_paths:
            loaded_items: list[dict[str, Any]] = []
            metadata_keys = (
                "step_key",
                "local_step",
                "block_idx",
                "block_index",
                "req_pool_idx",
                "start_offset",
                "response_start_index",
                "stored_tokens",
                "block_tokens",
                "truncated",
                "new_accept_count",
                "new_accept_tokens",
                "denoise_forwards",
                "action_token_count",
                "is_final_denoise_row",
                "is_terminal_commit",
                "producer_preselected",
                "producer_preselection_mode",
                "producer_source_rows",
                "producer_rank_selected_rows",
                "producer_selected_rows",
                "producer_final_rows",
                "producer_forced_final_rows",
                "producer_final_already_selected_rows",
                "producer_source_score_min",
                "producer_source_score_sum",
                "producer_source_score_max",
                "producer_zero_variance",
                "producer_block_index",
                "producer_state_selection",
                "producer_state_selection_version",
                "producer_state_selection_complete",
                "producer_state_selection_mode",
                "producer_state_selection_fraction",
                "producer_state_selection_direction",
                "producer_state_selection_score",
                "producer_state_selection_group_fields",
                "producer_state_selection_final_policy",
                "producer_state_selection_rank_selected",
                "producer_state_selection_forced_final",
                "adjacent_entropy_pair_valid",
                "adjacent_entropy_prev_step_key",
                "adjacent_entropy_pair_local_step",
                "adjacent_entropy_prev_mean",
                "adjacent_entropy_rollout_next_mean",
                "adjacent_entropy_vocab_size",
                "adjacent_entropy_next_mask",
                "global_aec_pair_valid",
                "global_aec_prev_step_key",
                "global_aec_prev_mean",
                "global_aec_rollout_next_mean",
                "global_aec_vocab_size",
                "global_aec_next_mask",
            )
            for item in items:
                if not isinstance(item, dict):
                    continue
                row_metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else item
                tensor_path = item.get("tensor_path") or row_metadata.get("tensor_path")
                if tensor_path:
                    tensor_path = str(tensor_path)
                    loaded = torch.load(tensor_path, map_location="cpu", weights_only=False)
                    if not isinstance(loaded, dict):
                        raise TypeError(
                            f"SDAR replay tensor file must contain a dictionary: {tensor_path}"
                        )
                    loaded_metadata = (
                        dict(loaded.get("metadata"))
                        if isinstance(loaded.get("metadata"), dict)
                        else {}
                    )
                    loaded_metadata.update(
                        {
                            key: row_metadata[key]
                            for key in metadata_keys
                            if key in row_metadata
                        }
                    )
                    loaded["metadata"] = loaded_metadata
                    loaded_items.append(loaded)
                    loaded_paths.append(tensor_path)
                else:
                    loaded_items.append(item)
            compact = self._compact_sdar_replay_tensors(loaded_items)
            compact_bytes = sum(
                int(value.numel() * value.element_size())
                for value in compact.values()
                if isinstance(value, torch.Tensor)
            )
            logger.info(
                "[QWEN_REPLAY_FETCH] request=%s rows=%d files=%d compact_bytes=%d",
                request_id,
                len(loaded_items),
                len(loaded_paths),
                compact_bytes,
            )
        elif metadata_first:
            compact = self._compact_sdar_replay_metadata(items)
        else:
            compact = self._compact_sdar_replay_tensors(items)
        if cleanup:
            for tensor_path in loaded_paths:
                try:
                    os.unlink(tensor_path)
                except FileNotFoundError:
                    pass
        if os.environ.get("SGLANG_SDAR_TRACE_TENSOR_STORE", "file").lower() in {"shm", "shared_memory"}:
            return self._compact_replay_to_shm_handles(compact)
        return compact

    async def start_profile(self, **kwargs):
        if (
            self.profiler_controller.check_enable()
            and self.profiler_controller.check_this_rank()
            and self.profiler_controller.is_discrete_mode()
        ):
            profile_args = build_sglang_profiler_args(
                self.profiler_controller.config, self.profiler_controller.tool_config, self.replica_rank
            )
            await self.tokenizer_manager.start_profile(**profile_args)

    async def stop_profile(self):
        if (
            self.profiler_controller.check_enable()
            and self.profiler_controller.check_this_rank()
            and self.profiler_controller.is_discrete_mode()
        ):
            await self.tokenizer_manager.stop_profile()


_rollout_worker_actor_cls = ray.remote(ServerAdapter)
_checkpoint_engine_worker_actor_cls = ray.remote(CheckpointEngineWorker)


class SGLangReplica(RolloutReplica):
    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: HFModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
    ):
        super().__init__(replica_rank, config, model_config, gpus_per_node, is_reward_model)
        self.server_class = ray.remote(SGLangHttpServer)

    def get_ray_class_with_init_args(self) -> RayClassWithInitArgs:
        """Get rollout worker actor class for colocated and standalone mode."""
        if self.config.checkpoint_engine.backend != "naive":
            worker_dict_cls = RayClassWithInitArgs(
                cls=_checkpoint_engine_worker_actor_cls,
                rollout_config=self.config,
                model_config=self.model_config,
                server_adapter=None,
            )
        else:
            worker_dict_cls = RayClassWithInitArgs(
                cls=_rollout_worker_actor_cls,
                config=self.config,
                model_config=self.model_config,
                device_mesh=None,
            )
        patched_sglang_path = get_patched_sglang_path(self.config)
        env_vars = self.get_worker_env()
        if patched_sglang_path and PATCHED_SGLANG_PATH_ENV not in env_vars:
            env_vars[PATCHED_SGLANG_PATH_ENV] = patched_sglang_path
        # RayWorkerGroup later replaces runtime_env.env_vars with its own rank
        # env plus worker_env.  This update is still useful for detached actor
        # paths, while get_worker_env() is the authoritative path here.
        if env_vars:
            worker_dict_cls.update_options(
                {"runtime_env": {"env_vars": env_vars}}
            )
        return worker_dict_cls

    def get_worker_env(self) -> dict[str, str]:
        env_vars = {
            key: value
            for key, value in os.environ.items()
            if key.startswith(
                (
                    "SGLANG_SDAR_TRACE_",
                    "SGLANG_SDAR_TPF_",
                    "SGLANG_SDAR_GPU_",
                    "SGLANG_SDAR_RESEARCH_FABRIC_",
                    "SGLANG_GEMMA_",
                )
            )
            or key == _SGLANG_SDAR_STREAM_REPLAY_TENSOR_STORE_ENV
        }
        env_vars[SGLANG_REPLICA_RANK_ENV] = str(self.replica_rank)
        patched_sglang_path = get_patched_sglang_path(self.config)
        if patched_sglang_path:
            env_vars[PATCHED_SGLANG_PATH_ENV] = patched_sglang_path
        return env_vars

    async def launch_servers(self):
        """Launch http server in each node."""
        assert len(self.workers) == self.world_size, (
            f"worker number {len(self.workers)} not equal to world size {self.world_size}"
        )

        # get (node_id, CUDA_VISIBLE_DEVICES) of all workers
        worker_infos = await asyncio.gather(
            *[
                worker.__ray_call__.remote(
                    lambda self: (ray.get_runtime_context().get_node_id(), os.environ[visible_devices_keyword])
                )
                for worker in self.workers
            ]
        )
        worker_cuda_visible_devices = [worker_info[1] for worker_info in worker_infos]
        worker_node_ids = [worker_info[0] for worker_info in worker_infos]
        base_gpu_id = 0
        infer_tp = self.config.tensor_model_parallel_size * self.config.data_parallel_size
        replica_world_size = infer_tp * self.config.pipeline_model_parallel_size
        if os.environ.get(f"RAY_EXPERIMENTAL_NOSET_{visible_devices_keyword}", None):
            logger.warning(f"RAY_EXPERIMENTAL_NOSET_{visible_devices_keyword} is set True!")
            base_gpu_id = (0 + self.replica_rank * replica_world_size) % self.gpus_per_node
        # create server actor in each node with node affinity and cuda visible devices
        for node_rank in range(self.nnodes):
            workers = self.workers[
                node_rank * self.gpus_per_replica_node : (node_rank + 1) * self.gpus_per_replica_node
            ]
            node_cuda_visible_devices_set = worker_cuda_visible_devices[
                node_rank * self.gpus_per_replica_node : (node_rank + 1) * self.gpus_per_replica_node
            ]
            node_cuda_visible_devices = ",".join(
                map(
                    str,
                    sorted(
                        set(
                            int(device)
                            for worker_devices_set in node_cuda_visible_devices_set
                            for device in worker_devices_set.split(",")
                            if device.strip()
                        )
                    ),
                )
            )

            node_id = worker_node_ids[node_rank * self.gpus_per_replica_node]
            name = (
                f"sglang_server_{self.replica_rank}_{node_rank}"
                if not self.is_reward_model
                else f"sglang_server_reward_{self.replica_rank}_{node_rank}"
            )
            env_vars = {f"RAY_EXPERIMENTAL_NOSET_{visible_devices_keyword}": "1"}
            # The patched SDAR rollout can write exact replay tensors under
            # SGLANG_SDAR_TRACE_* and can opt streaming raw replay into shm
            # handles.  Server actors use an explicit runtime_env, so forward
            # these opt-in settings instead of silently dropping them at the
            # Ray actor boundary.
            env_vars.update(
                {
                    key: value
                    for key, value in os.environ.items()
                    if key.startswith(
                        (
                            "SGLANG_SDAR_TRACE_",
                            "SGLANG_SDAR_TPF_",
                            "SGLANG_SDAR_RESEARCH_FABRIC_",
                            "SGLANG_GEMMA_",
                        )
                    )
                    or key == _SGLANG_SDAR_STREAM_REPLAY_TENSOR_STORE_ENV
                }
            )
            patched_sglang_path = get_patched_sglang_path(self.config)
            if patched_sglang_path:
                env_vars[PATCHED_SGLANG_PATH_ENV] = patched_sglang_path
            server = self.server_class.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                ),
                runtime_env={"env_vars": env_vars},
                name=name,
            ).remote(
                config=self.config,
                model_config=self.model_config,
                rollout_mode=self.rollout_mode,
                workers=workers,
                replica_rank=self.replica_rank,
                node_rank=node_rank,
                nnodes=self.nnodes,
                cuda_visible_devices=node_cuda_visible_devices,
                base_gpu_id=base_gpu_id,
            )
            self.servers.append(server)

        # launch http server in each node
        master_address, master_port = await self.servers[0].get_master_address.remote()
        await asyncio.gather(
            *[
                server.launch_server.remote(master_address=master_address, master_port=master_port)
                for server in self.servers
            ]
        )

        # get http server address from first server
        server_address, server_port = await self.servers[0].get_server_address.remote()
        self._server_handle = self.servers[0]
        self._server_address = (
            f"[{server_address}]:{server_port}"
            if is_valid_ipv6_address(server_address)
            else f"{server_address}:{server_port}"
        )


_DLLM_STOP_IDS_CACHE = None


def _resolve_dllm_stop_token_ids(server):
    """Stop ids for a tokenizer-less DLLM engine, resolved once per process.

    The engine runs with skip_tokenizer_init=True, so it cannot look up an eos itself,
    and llada2.1-mini has no generation_config.json to fall back on. Without this the
    rollout cannot terminate any sequence.
    """
    global _DLLM_STOP_IDS_CACHE
    if _DLLM_STOP_IDS_CACHE is not None:
        return _DLLM_STOP_IDS_CACHE

    import os
    raw = os.environ.get("DLLM_STOP_TOKEN_IDS")
    if raw is not None:
        # Explicit empty string means "no stop ids" on purpose, so honour it.
        ids = [int(x) for x in raw.replace(",", " ").split() if x.strip()]
        _DLLM_STOP_IDS_CACHE = ids
        print("[DLLM-STOP] from DLLM_STOP_TOKEN_IDS: %s" % ids, flush=True)
        return ids

    ids = []
    try:
        from transformers import AutoTokenizer
        path = server.model_config.local_path
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        # The chat template's assistant terminator is what the model was trained to
        # emit; the tokenizer eos is the generic fallback. Keep only single-token
        # markers -- a multi-token marker cannot act as a stop id at all.
        for marker in ("<|role_end|>", "<|im_end|>", "<|endoftext|>"):
            try:
                enc = tok.encode(marker, add_special_tokens=False)
            except Exception:
                continue
            if len(enc) == 1 and enc[0] not in ids:
                ids.append(enc[0])
        if tok.eos_token_id is not None and tok.eos_token_id not in ids:
            ids.append(int(tok.eos_token_id))
    except Exception as exc:
        print("[DLLM-STOP] could not resolve stop ids from the tokenizer: %r" % (exc,),
              flush=True)

    _DLLM_STOP_IDS_CACHE = ids
    print("[DLLM-STOP] resolved stop_token_ids=%s" % ids, flush=True)
    if not ids:
        # Say it loudly. A rollout with no stop id looks completely healthy and
        # silently spends its whole budget past the answer.
        print("[DLLM-STOP] WARNING: no stop ids resolved -- every rollout row will "
              "run to response_length and the loss will train on post-answer filler.",
              flush=True)
    return ids


def _apply_dllm_stop_token_ids(server, sampling_params):
    """Attach stop ids to a DLLM request, without overriding an explicit caller."""
    try:
        if not _is_dllm_rollout(server.config):
            return
        if sampling_params.get("stop_token_ids"):
            return  # caller knows better; leave it alone
        ids = _resolve_dllm_stop_token_ids(server)
        if ids:
            sampling_params["stop_token_ids"] = list(ids)
    except Exception as exc:
        # Never let stop-id resolution take down a rollout.
        print("[DLLM-STOP] skipped: %r" % (exc,), flush=True)
