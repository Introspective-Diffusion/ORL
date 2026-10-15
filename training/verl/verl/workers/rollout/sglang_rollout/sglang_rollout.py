# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
from __future__ import annotations

import logging
import multiprocessing as mp
import os
from importlib import import_module
from typing import Any, Generator

import ray
import torch
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from verl.utils.net_utils import is_valid_ipv6_address
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.sglang_rollout.import_utils import configure_sglang_import
from verl.workers.rollout.sglang_rollout.utils import get_named_tensor_buckets

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

SGLANG_REPLICA_RANK_ENV = "VERL_SGLANG_REPLICA_RANK"


def _import_sglang_module(module_name: str, config: Any = None):
    configure_sglang_import(config)
    return import_module(module_name)


# patch to avoid issue https://github.com/sgl-project/sglang/issues/6723
def _set_envs_and_config(server_args: Any):
    sglang_utils = _import_sglang_module("sglang.srt.utils")

    # Set global environments
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = str(int(server_args.enable_nccl_nvls))
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "4"
    os.environ["CUDA_MODULE_LOADING"] = "AUTO"
    # Enable faulthandler in subprocesses
    os.environ["PYTHONFAULTHANDLER"] = "1"

    # Set prometheus env vars
    if server_args.enable_metrics:
        sglang_utils.set_prometheus_multiproc_dir()

    # Set ulimit
    sglang_utils.set_ulimit()

    # Check flashinfer version
    if server_args.attention_backend == "flashinfer":
        sglang_utils.assert_pkg_version(
            "flashinfer_python",
            "0.2.5",
            "Please uninstall the old version and reinstall the latest version by following the instructions at https://docs.flashinfer.ai/installation.html.",
        )
    if sglang_utils.is_cuda():
        sglang_utils.assert_pkg_version(
            "sgl-kernel",
            "0.1.1",
            "Please reinstall the latest version with `pip install sgl-kernel --force-reinstall`",
        )

    # Set mp start method
    mp.set_start_method("spawn", force=True)


# because chatCompletion is an async method, it makes the whole ray actor be an async actor
# which can not call loop.run_until_complete. So we need to make the engine to be an async class
class ServerAdapter(BaseRollout):
    """SGLang server adapter used in native http server mode, serve as http client to request SGLang server
    to resume/release/update weights and kv_cache.

    - hybrid mode: reside in each hybrid worker to sync weights between training engine and SGLang server.
    - standalone/colocated mode: just a dummy placeholder to occupy the GPU to prevent ray scheduling new GPU actor.
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        configure_sglang_import(config)
        sglang = _import_sglang_module("sglang", config)
        sglang_engine = _import_sglang_module("sglang.srt.entrypoints.engine", config)
        sglang_engine._set_envs_and_config = _set_envs_and_config
        self._sgl_update_weights = _import_sglang_module("sglang.srt.weight_sync.utils", config).update_weights

        if config.get("quantization", None) == "fp8":
            from packaging import version

            assert version.parse(sglang.__version__) >= version.parse("0.5.5"), (
                "sglang>=0.5.5 is required for FP8 quantization"
            )
            FP8_BLOCK_QUANT_KWARGS = {
                "activation_scheme": "dynamic",
                "fmt": "e4m3",
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
            }
            fp8_block_quant_kwargs = dict(FP8_BLOCK_QUANT_KWARGS)
            model_config.hf_config.quantization_config = fp8_block_quant_kwargs
        super().__init__(config, model_config, device_mesh)
        self._weight_policy_version = 0
        self._engine: Any = None

        rank = int(os.environ["RANK"])
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        rollout_world_size = self.config.tensor_model_parallel_size * self.config.data_parallel_size
        self.rollout_rank = rank % rollout_world_size
        replica_rank_override = os.environ.get(SGLANG_REPLICA_RANK_ENV)
        self.replica_rank = int(replica_rank_override) if replica_rank_override is not None else rank // rollout_world_size
        self.node_rank = self.rollout_rank // local_world_size
        self.local_rank = self.rollout_rank % local_world_size
        logger.info(
            "SGLang ServerAdapter rank mapping: env_rank=%s local_world_size=%s rollout_world_size=%s "
            "replica_rank=%s rollout_rank=%s node_rank=%s local_rank=%s replica_override=%s",
            rank,
            local_world_size,
            rollout_world_size,
            self.replica_rank,
            self.rollout_rank,
            self.node_rank,
            self.local_rank,
            replica_rank_override,
        )

    def _memory_occupation_tags(self) -> list[str]:
        tags = ["kv_cache", "weights"]
        if not self.config.enforce_eager:
            tags.append("cuda_graph")
        return tags

    async def _init_server_adapter(self):
        if self._engine is not None:
            return

        # device_mesh is needed to gather cuda ipc handle to update weights
        if self.device_mesh is None:
            assert torch.distributed.is_initialized(), "torch distributed must be initialized"
            infer_tp = self.config.tensor_model_parallel_size * self.config.data_parallel_size
            infer_pp = self.config.pipeline_model_parallel_size
            infer_world_size = infer_tp * infer_pp
            dp = torch.distributed.get_world_size() // infer_world_size
            self.device_mesh = init_device_mesh(
                "cpu", mesh_shape=(dp, infer_tp, infer_pp), mesh_dim_names=["dp", "infer_tp", "infer_pp"]
            )

        # Only init http server adapter in tp rank 0
        if self.device_mesh["infer_tp"].get_local_rank() != 0:
            return

        # Lazy init http server adapter because http server is launched after hybrid engine.
        actor_name = f"sglang_server_{self.replica_rank}_{self.node_rank}"
        logger.info(
            "SGLang ServerAdapter resolving server actor: actor_name=%s replica_rank=%s node_rank=%s",
            actor_name,
            self.replica_rank,
            self.node_rank,
        )
        self.server_actor = ray.get_actor(actor_name)
        server_address, server_port = await self.server_actor.get_server_address.remote()
        logger.debug(
            f"replica_rank={self.replica_rank} node_rank={self.node_rank}, "
            f"server address: {server_address}, port: {server_port}"
        )
        host = f"[{server_address}]" if is_valid_ipv6_address(server_address) else server_address
        configure_sglang_import(self.config)
        AsyncHttpServerAdapter = import_module(
            "verl.workers.rollout.sglang_rollout.http_server_engine"
        ).AsyncHttpServerAdapter
        self._engine = AsyncHttpServerAdapter(
            model_path=self.model_config.local_path,
            host=host,
            port=server_port,
            launch_server=False,
            trust_remote_code=self.model_config.trust_remote_code,
        )

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tag: weights or kv_cache.
        """
        await self._init_server_adapter()
        if self.device_mesh["infer_tp"].get_local_rank() == 0 and self.config.free_cache_engine:
            await self._engine.resume_memory_occupation(tags=tags)

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        await self._init_server_adapter()
        if self.device_mesh["infer_tp"].get_local_rank() == 0 and self.config.free_cache_engine:
            await self._engine.release_memory_occupation(tags=self._memory_occupation_tags())

    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        """
        Update model weights using tensor buckets, similar to THUDM/slime's implementation.

        Notes:
          - For the best performance of `rebuild_cuda_tensor`, it is recommended to:
              1. Enable `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES`.
              2. Manually set `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
            when using Tensor Parallelism (TP >= 8).
          - See reference implementations in SLIME:
            - Main logic: https://github.com/THUDM/slime/blob/fb7605cc5fb09af0f9369d37f7192f12bddee577/slime/ray/ppo_actor.py#L452
            - runtime envs: https://github.com/THUDM/slime/blob/fb7605cc5fb09af0f9369d37f7192f12bddee577/slime/ray/ppo_actor.py#L39
        """
        await self._init_server_adapter()
        self._weight_policy_version += 1
        weight_version = str(
            kwargs.get("weight_version", f"rollout-{self._weight_policy_version}")
        )

        update_weights_bucket_bytes = int(self.config.checkpoint_engine.update_weights_bucket_megabytes) << 20
        if self.config.get("quantization", None) == "fp8":
            from verl.utils.sglang.sglang_fp8_utils import quant_weights_by_name

            logger.info("Convert bf16 weights to fp8 format before loading")
            weights = quant_weights_by_name(
                weights,
                self.model_config.hf_config.quantization_config,
                dtype=self.model_config.hf_config.dtype,
            )
        else:
            weights = weights

        async for params_batch in get_named_tensor_buckets(weights, update_weights_bucket_bytes):
            await self._sgl_update_weights(
                engine=self._engine,
                params_batch=params_batch,
                device_mesh_key="infer_tp",
                device_mesh=self.device_mesh,
                weight_version=weight_version,
            )

        if self.device_mesh["infer_tp"].get_local_rank() == 0:
            await self._engine.flush_cache()
