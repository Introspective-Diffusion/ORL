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
import asyncio
import heapq
import json
import logging
import os
import random
import time
from multiprocessing.shared_memory import SharedMemory
from abc import ABC, abstractmethod
from typing import Any, Optional
from uuid import uuid4

import hydra
import numpy as np
import ray
import torch
from cachetools import LRUCache
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from pydantic import BaseModel, ConfigDict
from tensordict import TensorDict
from transformers import AutoProcessor, AutoTokenizer

from verl.experimental.agent_loop.prometheus_utils import update_prometheus_config
from verl.experimental.agent_loop.utils import resolve_config_path
from verl.protocol import DataProto
from verl.single_controller.ray.base import RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import split_resource_pool
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.chat_template import initialize_system_prompt
from verl.utils.dataset.rl_dataset import RLHFDataset, get_dataset_class
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.utils.ray_utils import get_event_loop
from verl.utils.rollout_trace import (
    RolloutTraceConfig,
    rollout_trace_attr,
    rollout_trace_op,
)
from verl.utils.transferqueue_utils import tqbridge
from verl.workers.rollout.replica import TokenOutput, get_rollout_replica_class

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_BATCH_RPC_ENABLED: bool = os.getenv("VERL_SGLANG_BATCH_RPC", "0").strip().lower() in {"1", "true", "yes", "on"}
_SDAR_ATTACH_IN_AGENT_WORKER: bool = os.getenv(
    "VERL_SDAR_ATTACH_IN_AGENT_WORKER", "0"
).strip().lower() in {"1", "true", "yes", "on"}

# Replay of a recorded run (used by scripts/reproduce_qwen_dlm.sh): listed blocks get an all-MASK final-token snapshot
# before trace-map matching, and listed responses take the recorded reward verdict.
_SNAPSHOT_REPLAY_PATH: str = os.getenv("ORL_QWEN_DLM_REPLAY_SNAPSHOTS", "")
_SNAPSHOT_REPLAY_PLAN: dict | None = None


def _snapshot_replay_plan() -> dict | None:
    global _SNAPSHOT_REPLAY_PLAN
    if not _SNAPSHOT_REPLAY_PATH:
        return None
    if _SNAPSHOT_REPLAY_PLAN is None:
        with open(_SNAPSHOT_REPLAY_PATH, encoding="utf-8") as f:
            _SNAPSHOT_REPLAY_PLAN = json.load(f)
    return _SNAPSHOT_REPLAY_PLAN


def _derive_rollout_sampling_seed(
    *,
    seed_base: int,
    global_step: int,
    sample_position: int,
) -> int:
    """Derive a stable per-step/per-response SGLang sampling seed."""
    return int(
        (
            int(seed_base)
            + max(int(global_step), 0) * 1_000_003
            + int(sample_position) * 97_409
        )
        % (2**31 - 1)
    )


def _parse_mixed_prob_thresholds(value: Any) -> Optional[list[float]]:
    """Parse rollout_extra_sampling_params.mixed_prob_thresholds.

    Accepts a comma-separated string ("0.6,0.7,0.9") or a list of numbers.
    Returns None when unset/empty so the injection stays inert by default.
    """
    if value is None:
        return None
    if hasattr(value, "to_container"):
        value = value.to_container()
    if isinstance(value, str):
        value = [part for part in (piece.strip() for piece in value.split(",")) if part]
    thresholds = [float(x) for x in value]
    for threshold in thresholds:
        if not (0.0 < threshold <= 1.0):
            raise ValueError(
                f"mixed_prob_thresholds entries must be in (0, 1], got {threshold}"
            )
    return thresholds or None


def _mixed_prob_threshold_for_position(
    thresholds: list[float], sample_position: int
) -> float:
    """Round-robin threshold assignment: list[sample_position % len(list)]."""
    return float(thresholds[int(sample_position) % len(thresholds)])


def _assign_mixed_prob_thresholds(batch, rollout_config) -> None:
    """Assign the per-trace mixed_prob_threshold column on a batch in place.

    Batch position is the assignment key (list[i % len(list)]); with the
    prompt-major R-rollout layout of the FULL batch every GRPO group spans the
    whole threshold list, so this must run at the AgentLoopManager level
    before chunking — worker chunks restart positions and would starve the
    tail of the list (observed: 8 workers x 4 samples never reached 0.8/0.9).
    Idempotent (column already present -> no-op) so the worker-level call is
    a fallback for non-manager invocations only. No-op when the hydra key is
    unset or on validation batches (eval stays on the engine-default lens).
    """
    if "mixed_prob_threshold" in batch.non_tensor_batch:
        return
    if batch.meta_info.get("validate", False):
        return
    rollout_engine_kwargs = rollout_config.get("engine_kwargs", {}) or {}
    extra_sampling_params = (
        rollout_engine_kwargs.get("rollout_extra_sampling_params", {}) or {}
    )
    thresholds = _parse_mixed_prob_thresholds(
        extra_sampling_params.get("mixed_prob_thresholds", None)
    )
    if not thresholds:
        return
    batch.non_tensor_batch["mixed_prob_threshold"] = np.array(
        [
            _mixed_prob_threshold_for_position(thresholds, i)
            for i in range(len(batch))
        ],
        dtype=np.float64,
    )
    logger.info(
        "[MIXED_THR] per-request thresholds assigned: list=%s samples=%d",
        thresholds,
        len(batch),
    )


def _batch_rpc_fallback_reason(
    *,
    image_data: Any,
    video_data: Any,
    stream_sdar_replay: bool,
) -> Optional[str]:
    """Return a non-None reason string when the request cannot be batched."""
    if stream_sdar_replay:
        return "stream_sdar_replay"
    if image_data is not None:
        return "image_data"
    if video_data is not None:
        return "video_data"
    return None


class AsyncLLMServerManager:
    """
    A class to manage multiple OpenAI compatible LLM servers. This class provides
    - Load balance: least requests load balancing
    - Sticky session: send multi-turn chat completions to same server for automatic prefix caching
    """

    def __init__(
        self,
        config: DictConfig,
        server_handles: list[ray.actor.ActorHandle],
        max_cache_size: int = 10000,
        preferred_server_index: int | None = None,
    ):
        """Initialize the AsyncLLMServerManager.

        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
            max_cache_size (int, optional): max cache size for request_id to server mapping. Defaults to 10000.
        """
        self.config = config
        self.server_handles = list(server_handles)
        self._preferred_server = None
        if self.server_handles and preferred_server_index is not None:
            preferred_server_index = int(preferred_server_index) % len(
                self.server_handles
            )
            preferred = self.server_handles.pop(preferred_server_index)
            self.server_handles.insert(0, preferred)
            self._preferred_server = preferred
        else:
            random.shuffle(self.server_handles)

        # Least requests load balancing
        self.weighted_serveres = [[0, idx, server] for idx, server in enumerate(self.server_handles)]
        heapq.heapify(self.weighted_serveres)

        # LRU cache to map request_id to server
        self.request_id_to_server = LRUCache(maxsize=max_cache_size)

        # Batch-RPC coalescing state (VERL_SGLANG_BATCH_RPC)
        self._batch_rpc_enabled: bool = _BATCH_RPC_ENABLED
        # server_key (id(handle)) -> list of pending entries
        self._pending_batch: dict[int, list[dict[str, Any]]] = {}
        # server_key -> True when a flush callback is already scheduled
        self._batch_flush_scheduled: set[int] = set()
        # Aggregate metrics (monotonically increasing counters / appended lists)
        self._batch_metrics: dict[str, Any] = {
            "batch_rpc_count": 0,
            "fallback_rpc_count": 0,
            "batch_sizes": [],
            "batch_generate_seconds": [],
            "fallback_reasons": {},
        }

    def _choose_server(self, request_id: str) -> ray.actor.ActorHandle:
        # TODO: implement server pressure awareness load balancing
        if request_id in self.request_id_to_server:
            return self.request_id_to_server[request_id]

        if self._preferred_server is not None:
            self.request_id_to_server[request_id] = (
                self._preferred_server
            )
            return self._preferred_server

        _, _, server = self.weighted_serveres[0]
        self.weighted_serveres[0][0] += 1
        heapq.heapreplace(self.weighted_serveres, self.weighted_serveres[0])
        self.request_id_to_server[request_id] = server
        return server

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        stream_sdar_replay: bool = False,
        sdar_stream_queue: ray.actor.ActorHandle | None = None,
        sdar_stream_source: str | None = None,
        sdar_stream_context: dict[str, Any] | None = None,
    ) -> TokenOutput:
        """Generate tokens from prompt ids.

        Args:
            request_id (str): request id for sticky session.
            prompt_ids (List[int]): List of prompt token ids.
            sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.

        Returns:
            TokenOutput: token output
        """
        server = self._choose_server(request_id)

        if self._batch_rpc_enabled:
            fallback_reason = _batch_rpc_fallback_reason(
                image_data=image_data,
                video_data=video_data,
                stream_sdar_replay=stream_sdar_replay,
            )
            if fallback_reason is None:
                return await self._enqueue_and_await_batch(
                    server=server,
                    request_id=request_id,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                )
            # Not batchable — fall through to per-request RPC and track reason
            self._batch_metrics["fallback_rpc_count"] += 1
            self._batch_metrics["fallback_reasons"][fallback_reason] = (
                self._batch_metrics["fallback_reasons"].get(fallback_reason, 0) + 1
            )

        output = await server.generate.remote(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            video_data=video_data,
            stream_sdar_replay=stream_sdar_replay,
            sdar_stream_queue=sdar_stream_queue,
            sdar_stream_source=sdar_stream_source,
            sdar_stream_context=sdar_stream_context,
        )
        return output

    async def _enqueue_and_await_batch(
        self,
        server: ray.actor.ActorHandle,
        request_id: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
    ) -> TokenOutput:
        """Coalesce this request with concurrent same-server requests into one batch RPC."""
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        server_key = id(server)

        if server_key not in self._pending_batch:
            self._pending_batch[server_key] = []

        self._pending_batch[server_key].append(
            {
                "future": future,
                "server": server,
                "request_id": request_id,
                "prompt_ids": prompt_ids,
                "sampling_params": dict(sampling_params),
            }
        )

        if server_key not in self._batch_flush_scheduled:
            self._batch_flush_scheduled.add(server_key)
            # Schedule flush after all currently-pending coroutines have had a
            # chance to enqueue their requests (next event-loop callback slot).
            loop.call_soon(
                lambda sk=server_key, sv=server: asyncio.ensure_future(
                    self._flush_server_batch(sk, sv)
                )
            )

        return await future

    async def _flush_server_batch(self, server_key: int, server: ray.actor.ActorHandle) -> None:
        """Dispatch all pending requests for *server* as a single batch RPC."""
        self._batch_flush_scheduled.discard(server_key)
        pending = self._pending_batch.pop(server_key, [])
        if not pending:
            return

        batch_size = len(pending)
        self._batch_metrics["batch_rpc_count"] += 1
        self._batch_metrics["batch_sizes"].append(batch_size)

        futures: list[asyncio.Future] = [item["future"] for item in pending]
        requests = [
            {
                "request_id": item["request_id"],
                "prompt_ids": item["prompt_ids"],
                "sampling_params": item["sampling_params"],
            }
            for item in pending
        ]

        batch_start = time.time()
        try:
            outputs: list[TokenOutput] = await server.generate_batch.remote(requests)
            batch_seconds = time.time() - batch_start
            self._batch_metrics["batch_generate_seconds"].append(batch_seconds)

            if len(outputs) != len(futures):
                raise ValueError(
                    f"generate_batch returned {len(outputs)} outputs for {batch_size} requests"
                )
            for future, output in zip(futures, outputs, strict=True):
                if not future.done():
                    future.set_result(output)
        except Exception as exc:
            # Propagate the exception to every waiting coroutine so the caller
            # gets a clear per-request failure rather than a silent hang.
            for future in futures:
                if not future.done():
                    future.set_exception(exc)

    async def fetch_sdar_replay_tensors(
        self,
        request_id: str,
        handle: dict[str, Any],
        cleanup: bool = True,
    ) -> dict[str, Any]:
        server = self._choose_server(request_id)
        handle_request_id = str(handle.get("request_id", request_id))
        return await server.fetch_sdar_replay_tensors.remote(handle_request_id, cleanup=cleanup)

    def fetch_sdar_replay_tensors_ref(
        self,
        request_id: str,
        handle: dict[str, Any],
        cleanup: bool = True,
    ):
        server = self._choose_server(request_id)
        handle_request_id = str(handle.get("request_id", request_id))
        return server.fetch_sdar_replay_tensors.remote(handle_request_id, cleanup=cleanup)


class AgentLoopMetrics(BaseModel):
    """Agent loop performance metrics."""

    generate_sequences: float = 0.0
    tool_calls: float = 0.0
    postprocess: float = 0.0
    num_preempted: int = -1  # -1 means not available


class AgentLoopOutput(BaseModel):
    """Agent loop output."""

    prompt_ids: list[int]
    """Prompt token ids."""
    response_ids: list[int]
    """Response token ids including LLM generated token, tool response token."""
    response_mask: list[int]
    """Response mask, 1 for LLM generated token, 0 for tool response token."""
    response_logprobs: Optional[list[float]] = None
    """Log probabilities for the response tokens."""
    routed_experts: Optional[Any] = None
    """Routed experts for the total tokens."""
    multi_modal_data: Optional[dict[str, Any]] = None
    """Multi-modal data for multi-modal tools."""
    reward_score: Optional[float] = None
    """Reward score for the trajectory."""
    num_turns: int = 0
    """Number of chat turns, including user, assistant, tool."""
    metrics: AgentLoopMetrics
    """Auxiliary performance metrics"""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


class _InternalAgentLoopOutput(AgentLoopOutput):
    """Internal agent loop output with padded sequences."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_ids: torch.Tensor
    """Padded prompt token ids."""
    response_ids: torch.Tensor
    """Padded response token ids."""
    input_ids: torch.Tensor
    """Padded input ids(prompt_ids + response_ids)."""
    position_ids: torch.Tensor
    """Padded position ids."""
    response_mask: torch.Tensor
    """Padded response mask."""
    attention_mask: torch.Tensor
    """Padded attention mask."""
    response_logprobs: Optional[torch.Tensor] = None
    """Padded log probabilities for the response tokens."""
    routed_experts: Optional[torch.Tensor] = None
    """Padded routed experts for the total tokens."""
    multi_modal_inputs: Optional[dict[str, torch.Tensor]] = None
    """Multi-modal inputs for processors (e.g., pixel_values, image_grid_thw)."""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


class DictConfigWrap:
    """Wrapper for DictConfig to avoid hydra.utils.instantiate recursive resolve."""

    def __init__(self, config: DictConfig):
        self.config = config


class AgentLoopBase(ABC):
    """An agent loop takes an input message, chat with OpenAI compatible LLM server and interact with various
    environments."""

    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: AsyncLLMServerManager,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        dataset_cls: type[RLHFDataset],
        dataset_config: DictConfigWrap,
        **kwargs,
    ):
        """Initialize agent loop, each sample will have its own loop instance.

        Args:
            trainer_config (DictConfigWrap): trainer config.
            server_manager (AsyncLLMServerManager): OpenAI compatible LLM server manager.
            tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
            processor (AutoProcessor): Processor for process messages.
            dataset_cls (type[Dataset]): Dataset class for creating dataset, Defaults to RLHFDataset.
            dataset_config (DictConfigWrap): Dataset config.
        """
        self.config = trainer_config.config
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.processor = processor
        self.dataset_cls = dataset_cls
        self.dataset_config = dataset_config.config
        self.apply_chat_template_kwargs = self.dataset_config.get("apply_chat_template_kwargs", {})
        self.system_prompt = initialize_system_prompt(self.tokenizer, **self.apply_chat_template_kwargs)
        self.loop = get_event_loop()

    async def process_vision_info(self, messages: list[dict]) -> dict:
        """Extract images and videos from messages.

        Args:
            messages (list[dict]): Input messages.

        Returns:
            dict: Multi-modal data with keys "images" and "videos".
        """
        multi_modal_data = {}
        if self.processor is not None:
            images, videos = await self.dataset_cls.process_vision_info(
                messages, image_patch_size=self.processor.image_processor.patch_size, config=self.dataset_config
            )
            if images is not None:
                multi_modal_data["images"] = images
            if videos is not None:
                multi_modal_data["videos"] = videos

        return multi_modal_data

    async def apply_chat_template(
        self,
        messages: list[dict],
        tools: list[dict] = None,
        images: list[Image.Image] = None,
        videos: list[tuple[torch.Tensor, dict]] = None,
        remove_system_prompt: bool = False,
    ):
        """Apply chat template to messages with optional tools, images, and videos.

        Args:
            messages (list[dict]): Input messages.
            tools (list[dict], optional): Tools schemas. Defaults to None.
            images (list[Image.Image], optional): Input images. Defaults to None.
            videos (list[tuple[torch.Tensor, dict]], optional): Input videos. Defaults to None.
            remove_system_prompt (bool, optional): Whether to remove system prompt. Defaults to False.

        Returns:
            list[int]: Prompt token ids.
        """
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )

            # split the videos and according metadatas
            if videos is not None:
                videos, video_metadatas = zip(*videos, strict=False)
                videos, video_metadatas = list(videos), list(video_metadatas)
            else:
                video_metadatas = None

            model_inputs = self.processor(
                text=[raw_prompt],
                images=images,
                videos=videos,
                video_metadata=video_metadatas,
                return_tensors="pt",
                do_sample_frames=False,
            )
            prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
            if hasattr(prompt_ids, "input_ids"):
                prompt_ids = prompt_ids.input_ids
            if hasattr(prompt_ids, "squeeze"):
                prompt_ids = prompt_ids.squeeze(0).tolist()

        if remove_system_prompt:
            prompt_ids = prompt_ids[len(self.system_prompt) :]

        response_prefix = self.dataset_config.get("response_prefix", "") or os.environ.get("VERL_RESPONSE_PREFIX", "")
        if response_prefix:
            prefix_ids = self.tokenizer.encode(response_prefix, add_special_tokens=False)
            prompt_ids = prompt_ids + prefix_ids

        return prompt_ids

    @abstractmethod
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run agent loop to interact with LLM server and environment.

        Args:
            sampling_params (Dict[str, Any]): LLM sampling params.
            **kwargs: dataset fields from `verl.utils.dataset.RLHFDataset`.

        Returns:
            AgentLoopOutput: Agent loop output.
        """
        raise NotImplementedError


"""Agent loop registry: key is agent_name, value is a dict of agent loop config
used by hydra.utils.instantiate to initialize agent loop instance.

https://hydra.cc/docs/advanced/instantiate_objects/overview/
"""
_agent_loop_registry: dict[str, dict] = {}


def register(agent_name: str):
    """Register agent loop class."""

    def decorator(subclass: type[AgentLoopBase]) -> type[AgentLoopBase]:
        fqdn = f"{subclass.__module__}.{subclass.__qualname__}"
        _agent_loop_registry[agent_name] = {"_target_": fqdn}
        return subclass

    return decorator


class AgentLoopWorker:
    """Agent loop worker takes a batch of messages and run each message in an agent loop."""

    def __init__(
        self,
        config: DictConfig,
        server_handles: list[ray.actor.ActorHandle],
        reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
        worker_index: int | None = None,
    ):
        """Initialize agent loop manager.
        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
            reward_loop_worker_handles (List[ray.actor.ActorHandle]): Actor handles for streaming reward computation.
        """
        self.config = config
        self.worker_index = worker_index

        # for recipe to change
        if not hasattr(self, "server_manager"):
            self.server_manager = AsyncLLMServerManager(
                config,
                server_handles,
                preferred_server_index=worker_index,
            )

        self.dataset_cls = get_dataset_class(config.data)
        self.reward_loop_worker_handles = reward_loop_worker_handles

        model_path = config.actor_rollout_ref.model.path
        self.model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        self.processor = hf_processor(local_path, trust_remote_code=True)

        agent_loop_config_path = config.actor_rollout_ref.rollout.agent.agent_loop_config_path
        if agent_loop_config_path:
            resolved_path = resolve_config_path(agent_loop_config_path)
            agent_loop_configs = OmegaConf.load(resolved_path)
            for agent_loop_config in agent_loop_configs:
                _agent_loop_registry[agent_loop_config.name] = agent_loop_config
        if self.config.actor_rollout_ref.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.actor_rollout_ref.model.custom_chat_template
            self.tokenizer.chat_template = self.config.actor_rollout_ref.model.custom_chat_template

        trace_config = self.config.actor_rollout_ref.rollout.get("trace", {})
        RolloutTraceConfig.init(
            self.config.trainer.project_name,
            self.config.trainer.experiment_name,
            trace_config.get("backend"),
            trace_config.get("token2text", False),
            trace_config.get("max_samples_per_step_per_worker", None),
        )

    def ready(self) -> int:
        """Barrier used to keep lazy worker startup outside rollout timing."""
        from sglang.srt.dllm.algorithm.sdar_gpu_replay_lease import (
            is_gpu_replay_lease,
            slice_gpu_replay_lease,
        )

        # Resolve imports and private CUDA-IPC schema helpers before the first
        # timed rollout. References keep the module hot in this worker.
        self._gpu_replay_lease_helpers = (
            is_gpu_replay_lease,
            slice_gpu_replay_lease,
        )
        return int(self.worker_index if self.worker_index is not None else -1)

    @tqbridge()
    async def generate_sequences(self, batch: DataProto, replay_queue: ray.actor.ActorHandle | None = None) -> DataProto:
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        if len(batch) == 0:
            return DataProto(meta_info={"metrics": []})

        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )
        rollout_engine_kwargs = config.get("engine_kwargs", {}) or {}
        rollout_extra_sampling_params = rollout_engine_kwargs.get("rollout_extra_sampling_params", {}) or {}
        sampling_seed_base = rollout_extra_sampling_params.get(
            "sampling_seed_base",
            None,
        )
        if sampling_seed_base is not None:
            sampling_seed_base = int(sampling_seed_base)
        stop_token_ids = rollout_extra_sampling_params.get("stop_token_ids", None)
        if stop_token_ids is not None:
            if hasattr(stop_token_ids, "to_container"):
                stop_token_ids = stop_token_ids.to_container()
            sampling_params["stop_token_ids"] = [int(x) for x in stop_token_ids]
        # Fallback assignment for non-manager invocations; the manager assigns
        # this column on the full batch before chunking (idempotent here).
        _assign_mixed_prob_thresholds(batch, config)
        mixed_thr_column = batch.non_tensor_batch.get("mixed_prob_threshold")

        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature
            # Mixed per-request thresholds are a train-rollout treatment; eval
            # must stay on the engine-default decode lens.
            mixed_thr_column = None

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch.non_tensor_batch:
            default_agent_loop = config.agent.default_agent_loop
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_loop] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        max_samples_per_worker = RolloutTraceConfig.get_instance().max_samples_per_step_per_worker

        # For n rollouts per sample, we trace all n rollouts for selected samples
        # Note: This sampling happens per-worker, so total traces = max_samples_per_worker * num_workers * n
        if max_samples_per_worker is not None:
            unique_sample_indices = np.unique(index)
            if max_samples_per_worker < len(unique_sample_indices):
                selected_samples = set(
                    np.random.choice(unique_sample_indices, max_samples_per_worker, replace=False).tolist()
                )
                traced_indices = set(i for i in range(len(batch)) if index[i] in selected_samples)
            else:
                traced_indices = set(range(len(batch)))
        else:
            traced_indices = set(range(len(batch)))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index.tolist(), batch.meta_info.get("validate", False)
        )

        use_sdar_streaming_replay = (
            replay_queue is not None
            and self._is_sdar_diffusion_rollout()
        )
        sdar_stream_tasks: list[asyncio.Task | None] = [None] * len(batch)
        raw_replay_queue_max_items = 0
        if use_sdar_streaming_replay:
            from verl.trainer.ppo.split_streaming_queue import SplitStreamingReplayQueue

            split_streaming_cfg = self.config.trainer.get("split_gpu_streaming", {})
            try:
                raw_replay_queue_max_items = int(split_streaming_cfg.get("raw_replay_queue_max_items", 0) or 0)
            except (TypeError, ValueError):
                raw_replay_queue_max_items = 0
            raw_replay_queue_max_items = max(0, raw_replay_queue_max_items)

        tasks = []
        rollout_global_step = int(batch.meta_info.get("global_steps", -1))
        self._sdar_replay_global_step = (
            None if batch.meta_info.get("validate", False) else rollout_global_step
        )
        for i in range(len(batch)):
            trace_this_sample = i in traced_indices
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            if use_sdar_streaming_replay:
                raw_stream_queue = SplitStreamingReplayQueue.remote(
                    queue_name=f"sdar_raw_sample_{i}",
                    max_items=raw_replay_queue_max_items,
                )
                sample_non_tensor = self._slice_non_tensor_batch(batch.non_tensor_batch, i)
                sdar_stream_tasks[i] = asyncio.create_task(
                    self._consume_sdar_streaming_replay(
                        sample_idx=i,
                        raw_stream_queue=raw_stream_queue,
                        replay_queue=replay_queue,
                        sample_kwargs=dict(kwargs),
                        sample_non_tensor=sample_non_tensor,
                    )
                )
                kwargs = {
                    **kwargs,
                    "_sdar_stream_replay": True,
                    "_sdar_stream_queue": raw_stream_queue,
                    "_sdar_stream_source": f"agent_worker_sample_{i}",
                }
            tasks.append(
                asyncio.create_task(
                    self._run_agent_loop_with_index(
                        i,
                        {
                            **sampling_params,
                            **(
                                {
                                    "sampling_seed": (
                                        _derive_rollout_sampling_seed(
                                            seed_base=sampling_seed_base,
                                            global_step=rollout_global_step,
                                            sample_position=i,
                                        )
                                    )
                                }
                                if sampling_seed_base is not None
                                else {}
                            ),
                            **(
                                {
                                    "dllm_prob_threshold": float(mixed_thr_column[i]),
                                    # edit follows prob for the mixed arm.
                                    "dllm_edit_threshold": float(mixed_thr_column[i]),
                                }
                                if mixed_thr_column is not None
                                else {}
                            ),
                        },
                        trajectory_info[i],
                        trace=trace_this_sample,
                        **kwargs,
                    )
                )
            )

        if replay_queue is None:
            indexed_outputs = await asyncio.gather(*tasks)
            outputs = [output for _, output in sorted(indexed_outputs, key=lambda item: item[0])]
            output = self._postprocess(
                outputs,
                input_non_tensor_batch=batch.non_tensor_batch,
                # With one sample per agent worker, attaching here lets each
                # worker prepare replay state independently while slower remote
                # generations are still running.  The legacy manager-side
                # attach remains the default for compatibility.
                defer_sdar_attach=not _SDAR_ATTACH_IN_AGENT_WORKER,
            )
        else:
            outputs: list[_InternalAgentLoopOutput | None] = [None] * len(tasks)
            streamed_outputs: list[DataProto | None] = [None] * len(tasks)
            pending_stream_tasks: list[tuple[int, asyncio.Task, DataProto]] = []
            for task in asyncio.as_completed(tasks):
                sample_idx, sample_output = await task
                outputs[sample_idx] = sample_output
                sample_non_tensor = self._slice_non_tensor_batch(batch.non_tensor_batch, sample_idx)
                sample_data = self._postprocess([sample_output], input_non_tensor_batch=sample_non_tensor)
                streamed_outputs[sample_idx] = sample_data
                stream_task = sdar_stream_tasks[sample_idx] if sample_idx < len(sdar_stream_tasks) else None
                if stream_task is not None:
                    pending_stream_tasks.append((sample_idx, stream_task, sample_data))
                else:
                    await replay_queue.push.remote(sample_data, source=f"agent_worker_sample_{sample_idx}")
            if pending_stream_tasks:
                stream_counts = await asyncio.gather(
                    *(stream_task for _, stream_task, _ in pending_stream_tasks)
                )
                for (sample_idx, _stream_task, sample_data), streamed_chunk_count in zip(
                    pending_stream_tasks,
                    stream_counts,
                    strict=True,
                ):
                    streamed_chunk_count = int(streamed_chunk_count)
                    if streamed_chunk_count <= 0:
                        print(
                            f"[SDAR_STREAM_DIAG] zero_chunk_fallback sample_idx={sample_idx}",
                            flush=True,
                        )
                        await replay_queue.push.remote(sample_data, source=f"agent_worker_sample_{sample_idx}")
                    else:
                        self._drop_streamed_sdar_replay_state_(sample_data)
            ordered_streamed_outputs = [item for item in streamed_outputs if item is not None]
            AgentLoopManager._align_worker_output_non_tensor_batch_(ordered_streamed_outputs)
            # Per-sample attach timings differ across chunks and DataProto.concat
            # asserts meta_info equality; fold them into one scalar (the manager
            # consumes a single per-worker value, see sdar_worker_attach_seconds
            # pop in AgentLoopManager).
            _sdar_attach_seconds = sum(
                float(item.meta_info.pop("sdar_worker_attach_seconds", 0.0) or 0.0)
                for item in ordered_streamed_outputs
            )
            output = DataProto.concat(ordered_streamed_outputs)
            if _sdar_attach_seconds > 0.0:
                output.meta_info["sdar_worker_attach_seconds"] = _sdar_attach_seconds
            output.meta_info["metrics"] = [
                item.metrics.model_dump()
                for item in outputs
                if item is not None
            ]

        return output

    async def _run_agent_loop_with_index(
        self,
        sample_pos: int,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        trace: bool = True,
        **kwargs,
    ) -> tuple[int, _InternalAgentLoopOutput]:
        raw_stream_queue = kwargs.get("_sdar_stream_queue")
        try:
            output = await self._run_agent_loop(
                sampling_params,
                trajectory,
                agent_name=agent_name,
                trace=trace,
                **kwargs,
            )
            return sample_pos, output
        finally:
            if raw_stream_queue is not None:
                await raw_stream_queue.close.remote()

    @staticmethod
    def _slice_non_tensor_batch(non_tensor_batch: dict[str, Any], index: int) -> dict[str, np.ndarray]:
        sliced = {}
        for key, value in non_tensor_batch.items():
            if isinstance(value, np.ndarray):
                sliced[key] = value[index : index + 1]
            else:
                sliced[key] = np.array([value[index]], dtype=object)
        return sliced

    @staticmethod
    def _strip_internal_agent_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in kwargs.items()
            if not key.startswith("_sdar_stream_") and key != "_skip_compute_score"
        }

    @staticmethod
    def _as_int_list(value: Any) -> list[int]:
        if value is None:
            return []
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, tuple):
            value = list(value)
        while isinstance(value, list) and len(value) == 1 and isinstance(value[0], (list, tuple)):
            value = list(value[0])
        if not isinstance(value, list):
            value = list(value)
        return [int(item.item() if hasattr(item, "item") else item) for item in value]

    @staticmethod
    def _tensor_from_sdar_shm_handle(handle: dict[str, Any]) -> torch.Tensor:
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
        shape = tuple(int(x) for x in handle["shape"])
        nbytes = int(handle["nbytes"])
        shm = SharedMemory(name=str(handle["name"]))
        try:
            raw_np = np.ndarray((nbytes,), dtype=np.uint8, buffer=shm.buf)
            raw_t = torch.from_numpy(raw_np)
            tensor = torch.empty(int(np.prod(shape)), dtype=dtype)
            tensor.view(torch.uint8).copy_(raw_t)
            return tensor.view(shape)
        finally:
            shm.close()
            try:
                shm.unlink()
            except FileNotFoundError:
                pass

    @classmethod
    def _materialize_sdar_replay_payload(cls, payload: Any) -> Any:
        if isinstance(payload, dict) and payload.get("format") == "sglang_sdar_replay_shm_batch_v1":
            materialized = {
                **{key: value for key, value in payload.items() if key not in {"format", "tensors"}},
                "format": "sglang_sdar_replay_tensor_batch_v1",
            }
            tensors = payload.get("tensors", {})
            if isinstance(tensors, dict):
                for key, handle in tensors.items():
                    if isinstance(handle, dict):
                        materialized[key] = cls._tensor_from_sdar_shm_handle(handle)
            return materialized
        if isinstance(payload, list):
            return [cls._materialize_sdar_replay_payload(item) for item in payload]
        if isinstance(payload, tuple):
            return tuple(cls._materialize_sdar_replay_payload(item) for item in payload)
        return payload

    @staticmethod
    def _tensor_to_sdar_shm_handle(tensor: torch.Tensor) -> dict[str, Any]:
        tensor = tensor.detach().cpu().contiguous()
        raw = tensor.view(torch.uint8).numpy().reshape(-1)
        shm = SharedMemory(create=True, size=max(1, int(raw.nbytes)))
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
                    logger.warning(
                        "Unable to unlink failed direct compact SDAR replay shm name=%s",
                        shm.name,
                        exc_info=True,
                    )
            shm.close()

    @staticmethod
    def _unlink_sdar_shm_payload(payload: Any) -> None:
        if isinstance(payload, dict) and payload.get("format") == "sglang_sdar_replay_shm_batch_v1":
            tensors = payload.get("tensors", {})
            if not isinstance(tensors, dict):
                return
            for handle in tensors.values():
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
                        "Unable to unlink direct compact SDAR replay shm name=%s",
                        handle.get("name"),
                        exc_info=True,
                    )
                finally:
                    shm.close()
            return
        if isinstance(payload, (list, tuple)):
            for item in payload:
                AgentLoopWorker._unlink_sdar_shm_payload(item)

    @classmethod
    def _sdar_replay_payload_to_shm_handles(cls, payload: Any) -> tuple[Any, int]:
        if not isinstance(payload, dict):
            return payload, 0
        handles: dict[str, Any] = {
            "format": "sglang_sdar_replay_shm_batch_v1",
            "tensors": {},
        }
        shm_bytes = 0
        try:
            for key, value in payload.items():
                if isinstance(value, torch.Tensor):
                    handle = cls._tensor_to_sdar_shm_handle(value)
                    handles["tensors"][key] = handle
                    shm_bytes += int(handle.get("nbytes", 0) or 0)
                elif key not in {"format", "tensors"}:
                    handles[key] = value
            return handles, shm_bytes
        except Exception:
            cls._unlink_sdar_shm_payload(handles)
            raise

    def _direct_compact_replay_tensor_store(self) -> str:
        split_streaming_cfg = self.config.trainer.get("split_gpu_streaming", {})
        store = str(split_streaming_cfg.get("direct_compact_replay_tensor_store", "tensor") or "tensor").strip().lower()
        if store in {"", "tensor"}:
            return "tensor"
        if store in {"shm", "shared_memory"}:
            return "shm"
        if store in {"table", "memory_table", "handle"}:
            return "table"
        raise RuntimeError(
            "trainer.split_gpu_streaming.direct_compact_replay_tensor_store "
            f"must be tensor, shm, or table, got {store!r}"
        )

    def _direct_compact_enabled(self) -> bool:
        env_value = os.environ.get("VERL_SPLIT_DIRECT_COMPACT_ENABLE")
        if env_value is not None:
            return str(env_value).strip().lower() not in {"0", "false", "no", "off", "disabled"}
        split_streaming_cfg = self.config.trainer.get("split_gpu_streaming", {})
        return bool(split_streaming_cfg.get("direct_compact_enable", True))

    @staticmethod
    def _extract_sdar_replay_payload(raw_chunk: dict[str, Any]) -> Any:
        for key in (
            "sglang_sdar_replay_tensors",
            "sglang_sdar_replay_tensors_compact",
            "replay_tensors",
            "replay_tensors_compact",
            "compact_replay_tensors",
        ):
            payload = raw_chunk.get(key)
            if payload is not None and (
                not isinstance(payload, (dict, list, tuple)) or len(payload) > 0
            ):
                return payload
        meta_info = raw_chunk.get("meta_info")
        if isinstance(meta_info, dict):
            payload = meta_info.get("sglang_sdar_replay_tensors")
            if payload is not None and (
                not isinstance(payload, (dict, list, tuple)) or len(payload) > 0
            ):
                return payload
        return None

    @staticmethod
    def _is_sdar_replay_table_handle(payload: Any) -> bool:
        return isinstance(payload, dict) and payload.get("format") == "sglang_sdar_replay_table_handle_v1"

    def _sdar_stream_future_logits_offset(self) -> int:
        jsd_cfg = self.config.actor_rollout_ref.actor.get("generation_consistency_jsd", {})
        if not bool(jsd_cfg.get("enable", False)):
            return 0
        if jsd_cfg.get("objective", "full_jsd") != "trace_future_ce":
            return 0
        return int(jsd_cfg.get("trace_future_logits_offset", 5) or 0)

    def _sdar_stream_future_skip_tail_steps(self) -> int:
        future_offset = self._sdar_stream_future_logits_offset()
        if future_offset <= 0:
            return 0
        jsd_cfg = self.config.actor_rollout_ref.actor.get("generation_consistency_jsd", {})
        return int(jsd_cfg.get("trace_future_skip_tail_steps", future_offset) or future_offset)

    @staticmethod
    def _normalize_trace_future_tail_policy(value: Any) -> str:
        policy = str(value or "strict_skip").strip().lower()
        aliases = {
            "strict": "strict_skip",
            "skip": "strict_skip",
            "skip_tail": "strict_skip",
            "last": "last_available_skip_final",
            "last_available": "last_available_skip_final",
            "last_available_teacher": "last_available_skip_final",
        }
        policy = aliases.get(policy, policy)
        if policy not in {"strict_skip", "last_available_skip_final"}:
            raise ValueError(
                "actor.generation_consistency_jsd.trace_future_tail_policy must be "
                "strict_skip or last_available_skip_final, got "
                f"{value!r}"
            )
        return policy

    def _sdar_stream_future_tail_policy(self) -> str:
        jsd_cfg = self.config.actor_rollout_ref.actor.get("generation_consistency_jsd", {})
        return self._normalize_trace_future_tail_policy(
            jsd_cfg.get("trace_future_tail_policy", "strict_skip")
        )

    def _sdar_stream_future_train_all_missing_with_final(self) -> bool:
        jsd_cfg = self.config.actor_rollout_ref.actor.get("generation_consistency_jsd", {})
        return bool(jsd_cfg.get("trace_future_train_all_missing_with_final", False))

    def _sdar_stream_future_train_final_step_with_final(self) -> bool:
        jsd_cfg = self.config.actor_rollout_ref.actor.get("generation_consistency_jsd", {})
        return bool(jsd_cfg.get("trace_future_train_final_step_with_final", False))

    def _sdar_stream_requires_teacher_logits(self) -> bool:
        jsd_cfg = self.config.actor_rollout_ref.actor.get("generation_consistency_jsd", {})
        if not bool(jsd_cfg.get("enable", False)):
            return False
        if jsd_cfg.get("objective", "full_jsd") != "trace_future_ce":
            return False
        return not bool(jsd_cfg.get("trace_future_hard_label_ce", False))

    def _sdar_stream_accumulate_teacher_blocks(self) -> bool:
        jsd_cfg = self.config.actor_rollout_ref.actor.get("generation_consistency_jsd", {})
        return (
            self._sdar_stream_requires_teacher_logits()
            and bool(jsd_cfg.get("trace_future_accumulate_teacher_blocks", True))
        )

    def _sdar_stream_delta_teacher_cache(self) -> bool:
        jsd_cfg = self.config.actor_rollout_ref.actor.get("generation_consistency_jsd", {})
        return (
            self._sdar_stream_accumulate_teacher_blocks()
            and bool(jsd_cfg.get("trace_future_delta_teacher_cache", False))
        )

    def _sdar_stream_replay_step_stride(self) -> int:
        jsd_cfg = self.config.actor_rollout_ref.actor.get("generation_consistency_jsd", {})
        try:
            stride = int(jsd_cfg.get("stream_replay_step_stride", 1) or 1)
        except (TypeError, ValueError):
            stride = 1
        return max(1, stride)

    def _sdar_stream_replay_step_offset(self) -> int:
        jsd_cfg = self.config.actor_rollout_ref.actor.get("generation_consistency_jsd", {})
        try:
            return int(jsd_cfg.get("stream_replay_step_offset", 0) or 0)
        except (TypeError, ValueError):
            return 0

    def _sdar_stream_block_level_final_flush(self) -> bool:
        jsd_cfg = self.config.actor_rollout_ref.actor.get("generation_consistency_jsd", {})
        return bool(
            jsd_cfg.get("stream_block_level_final_flush", False)
            or jsd_cfg.get("trace_future_block_level_final_flush", False)
        )

    @staticmethod
    def _sdar_replay_payload_step_keys(payload: Any) -> set[int]:
        if isinstance(payload, dict) and payload.get("format") == "sglang_sdar_replay_tensor_batch_v1":
            step_keys = payload.get("step_keys")
            if isinstance(step_keys, torch.Tensor):
                return {int(step_key) for step_key in step_keys.detach().cpu().long().view(-1).tolist()}
            return set()
        if isinstance(payload, (list, tuple)):
            out: set[int] = set()
            for item in payload:
                if not isinstance(item, dict):
                    continue
                meta = item.get("metadata", item)
                step_key = meta.get("step_key") if isinstance(meta, dict) else None
                if step_key is not None:
                    out.add(int(step_key))
            return out
        if isinstance(payload, dict):
            meta = payload.get("metadata", payload)
            step_key = meta.get("step_key") if isinstance(meta, dict) else None
            return {int(step_key)} if step_key is not None else set()
        return set()

    @staticmethod
    def _sdar_replay_payload_block_indices(payload: Any) -> set[int]:
        if isinstance(payload, dict) and payload.get("format") == "sglang_sdar_replay_tensor_batch_v1":
            block_indices = payload.get("block_indices")
            if isinstance(block_indices, torch.Tensor):
                return {int(block_idx) for block_idx in block_indices.detach().cpu().long().view(-1).tolist()}
            return set()
        if isinstance(payload, (list, tuple)):
            out: set[int] = set()
            for item in payload:
                if not isinstance(item, dict):
                    continue
                meta = item.get("metadata", item)
                block_idx = meta.get("block_idx") if isinstance(meta, dict) else None
                if block_idx is not None:
                    out.add(int(block_idx))
            return out
        if isinstance(payload, dict):
            meta = payload.get("metadata", payload)
            block_idx = meta.get("block_idx") if isinstance(meta, dict) else None
            return {int(block_idx)} if block_idx is not None else set()
        return set()

    @staticmethod
    def _filter_sdar_replay_payload_step_keys(payload: Any, keep_step_keys: set[int]) -> Any:
        keep_step_keys = {int(step_key) for step_key in keep_step_keys}
        if (
            not keep_step_keys
            or not isinstance(payload, dict)
            or payload.get("format") != "sglang_sdar_replay_tensor_batch_v1"
        ):
            return None
        step_keys = payload.get("step_keys")
        if not isinstance(step_keys, torch.Tensor):
            return payload
        step_keys_cpu = step_keys.detach().cpu().long().view(-1)
        n = int(step_keys_cpu.numel())
        keep_indices = [
            idx
            for idx, step_key in enumerate(step_keys_cpu.tolist())
            if int(step_key) in keep_step_keys
        ]
        if not keep_indices:
            return None
        filtered = dict(payload)
        if len(keep_indices) != n:
            index = torch.tensor(keep_indices, dtype=torch.long)
            for key, value in payload.items():
                if isinstance(value, torch.Tensor) and value.dim() >= 1 and int(value.shape[0]) == n:
                    filtered[key] = value.index_select(0, index.to(value.device))
                elif isinstance(value, list) and len(value) == n:
                    filtered[key] = [value[i] for i in keep_indices]
        return filtered

    @staticmethod
    def _shallow_copy_sdar_replay_payload(payload: Any) -> Any:
        if isinstance(payload, dict):
            return dict(payload)
        if isinstance(payload, list):
            return list(payload)
        if isinstance(payload, tuple):
            return tuple(payload)
        return payload

    @classmethod
    def _sdar_payload_raw_push_ts_bounds(
        cls,
        payload: Any,
        step_push_ts: dict[int, float],
        *,
        fallback_first_ts: float = 0.0,
        fallback_last_ts: float = 0.0,
    ) -> tuple[float, float]:
        timestamps = [
            float(step_push_ts.get(step_key, 0.0) or 0.0)
            for step_key in cls._sdar_replay_payload_step_keys(payload)
        ]
        timestamps = [ts for ts in timestamps if ts > 0.0]
        first_ts = min(timestamps) if timestamps else float(fallback_first_ts or 0.0)
        last_ts = max(timestamps) if timestamps else float(fallback_last_ts or first_ts or 0.0)
        return first_ts, last_ts

    @staticmethod
    def _sdar_stream_timing_metadata(
        *,
        raw_first_push_ts: float,
        raw_last_push_ts: float,
        raw_queue_wait_seconds: float,
        now_ts: float,
        sample_first_raw_push_ts: float = 0.0,
        request_first_raw_push_ts: float = 0.0,
    ) -> dict[str, float]:
        raw_first_push_ts = float(raw_first_push_ts or 0.0)
        raw_last_push_ts = float(raw_last_push_ts or 0.0)
        sample_first_raw_push_ts = float(sample_first_raw_push_ts or raw_first_push_ts or 0.0)
        request_first_raw_push_ts = float(request_first_raw_push_ts or raw_first_push_ts or 0.0)
        now_ts = float(now_ts)
        raw_ready_push_ts = max(raw_first_push_ts, raw_last_push_ts)
        raw_first_to_trainer_seconds = (
            max(0.0, now_ts - raw_first_push_ts)
            if raw_first_push_ts > 0.0
            else 0.0
        )
        raw_ready_to_trainer_seconds = (
            max(0.0, now_ts - raw_ready_push_ts)
            if raw_ready_push_ts > 0.0
            else 0.0
        )
        raw_support_wait_seconds = max(0.0, raw_first_to_trainer_seconds - raw_ready_to_trainer_seconds)
        return {
            "raw_first_push_ts": raw_first_push_ts,
            "raw_last_push_ts": raw_last_push_ts,
            "raw_ready_push_ts": raw_ready_push_ts,
            "sample_first_raw_push_ts": sample_first_raw_push_ts,
            "request_first_raw_push_ts": request_first_raw_push_ts,
            "raw_first_to_trainer_push_seconds": raw_first_to_trainer_seconds,
            "raw_ready_to_trainer_push_seconds": raw_ready_to_trainer_seconds,
            "raw_support_wait_seconds": raw_support_wait_seconds,
            "raw_to_trainer_push_seconds": raw_ready_to_trainer_seconds,
            "raw_queue_wait_seconds": max(0.0, float(raw_queue_wait_seconds or 0.0)),
            "direct_ready_wait_seconds": raw_ready_to_trainer_seconds,
        }

    @staticmethod
    def _update_sdar_stream_chunk_timing_(
        data: DataProto,
        timing_metadata: dict[str, float],
    ) -> None:
        stream_markers = data.non_tensor_batch.get("sglang_sdar_stream_chunk")
        if isinstance(stream_markers, np.ndarray):
            marker_iter = stream_markers.reshape(-1).tolist()
        elif isinstance(stream_markers, (list, tuple)):
            marker_iter = list(stream_markers)
        else:
            marker_iter = [stream_markers]
        for marker in marker_iter:
            if isinstance(marker, dict):
                marker.update(timing_metadata)

    @staticmethod
    def _merge_sdar_replay_payloads(existing: Any, incoming: Any) -> Any:
        if existing is None:
            return incoming
        if incoming is None:
            return existing
        if (
            isinstance(existing, dict)
            and isinstance(incoming, dict)
            and existing.get("format") == "sglang_sdar_replay_tensor_batch_v1"
            and incoming.get("format") == "sglang_sdar_replay_tensor_batch_v1"
        ):
            existing_steps = existing.get("step_keys")
            incoming_steps = incoming.get("step_keys")
            if not isinstance(existing_steps, torch.Tensor) or not isinstance(incoming_steps, torch.Tensor):
                return incoming
            n_existing = int(existing_steps.detach().view(-1).numel())
            n_incoming = int(incoming_steps.detach().view(-1).numel())
            merged = dict(existing)
            keys = set(existing) | set(incoming)
            for key in keys:
                left = existing.get(key)
                right = incoming.get(key)
                if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
                    if left.dim() >= 1 and right.dim() >= 1 and int(left.shape[0]) == n_existing and int(right.shape[0]) == n_incoming:
                        merged[key] = torch.cat([left, right], dim=0)
                    elif key not in merged:
                        merged[key] = right
                elif isinstance(left, list) and isinstance(right, list) and len(left) == n_existing and len(right) == n_incoming:
                    merged[key] = [*left, *right]
                elif key not in merged and right is not None:
                    merged[key] = right
            return merged
        if isinstance(existing, list):
            return [*existing, *list(incoming if isinstance(incoming, (list, tuple)) else [incoming])]
        if isinstance(existing, tuple):
            return tuple([*existing, *list(incoming if isinstance(incoming, (list, tuple)) else [incoming])])
        return [existing, *list(incoming if isinstance(incoming, (list, tuple)) else [incoming])]

    @staticmethod
    def _expand_sdar_replay_payload_normalized_step_keys(payload: Any, *, block_length: int) -> Any:
        if not (
            isinstance(payload, dict)
            and payload.get("format") == "sglang_sdar_replay_tensor_batch_v1"
        ):
            return payload
        step_keys = payload.get("step_keys")
        local_steps = payload.get("local_steps")
        block_indices = payload.get("block_indices")
        if not (
            isinstance(step_keys, torch.Tensor)
            and isinstance(local_steps, torch.Tensor)
            and isinstance(block_indices, torch.Tensor)
        ):
            return payload
        step_keys_flat = step_keys.detach().cpu().long().view(-1)
        local_steps_flat = local_steps.detach().cpu().long().view(-1)
        block_indices_flat = block_indices.detach().cpu().long().view(-1)
        n = int(step_keys_flat.numel())
        if n <= 0 or int(local_steps_flat.numel()) != n or int(block_indices_flat.numel()) != n:
            return payload

        block_length = max(1, int(block_length or 1))
        existing_step_keys = {int(step_key) for step_key in step_keys_flat.tolist()}
        alias_indices: list[int] = []
        alias_step_keys: list[int] = []
        for idx, (raw_step_key, local_step, block_idx) in enumerate(
            zip(
                step_keys_flat.tolist(),
                local_steps_flat.tolist(),
                block_indices_flat.tolist(),
                strict=True,
            )
        ):
            normalized_step_key = int(block_idx) * block_length + int(local_step)
            if normalized_step_key == int(raw_step_key) or normalized_step_key in existing_step_keys:
                continue
            existing_step_keys.add(normalized_step_key)
            alias_indices.append(idx)
            alias_step_keys.append(normalized_step_key)
        if not alias_indices:
            return payload

        index_cpu = torch.tensor(alias_indices, dtype=torch.long)
        expanded = dict(payload)
        for key, value in payload.items():
            if isinstance(value, torch.Tensor):
                if value.dim() >= 1 and int(value.shape[0]) == n:
                    index = index_cpu.to(value.device)
                    expanded[key] = torch.cat([value, value.index_select(0, index)], dim=0)
            elif isinstance(value, list) and len(value) == n:
                expanded[key] = [*value, *[value[idx] for idx in alias_indices]]
        alias_tensor = torch.tensor(alias_step_keys, dtype=step_keys.dtype, device=step_keys.device)
        expanded["step_keys"] = torch.cat([step_keys.view(-1), alias_tensor], dim=0)
        expanded["normalized_step_key_aliases_added"] = int(len(alias_step_keys))
        return expanded

    @staticmethod
    def _sdar_same_block_future_teacher_step_key(
        step_key: int,
        available_step_keys: list[int] | set[int],
        future_offset: int,
        block_length: int,
        *,
        allow_final_fallback: bool,
        require_closed_block_for_final: bool = False,
    ) -> int | None:
        step_key = int(step_key)
        block_length = max(1, int(block_length or 1))
        block_idx = step_key // block_length
        available_step_keys_sorted = sorted(
            int(k)
            for k in available_step_keys
            if int(k) // block_length == block_idx
        )
        if not available_step_keys_sorted or step_key not in available_step_keys_sorted:
            return None
        current_idx = available_step_keys_sorted.index(step_key)
        target_idx = current_idx + max(0, int(future_offset))
        if target_idx < len(available_step_keys_sorted):
            return int(available_step_keys_sorted[target_idx])
        if not allow_final_fallback:
            return None
        if require_closed_block_for_final and len(available_step_keys_sorted) < block_length:
            return None
        return int(available_step_keys_sorted[-1])

    @staticmethod
    def _sdar_replay_payload_future_teacher_step_key(
        payload: Any,
        step_key: int,
        future_offset: int,
        block_length: int,
        *,
        allow_final_fallback: bool,
        require_closed_block_for_final: bool = False,
    ) -> int | None:
        if (
            isinstance(payload, dict)
            and payload.get("format") == "sglang_sdar_replay_tensor_batch_v1"
            and isinstance(payload.get("step_keys"), torch.Tensor)
            and isinstance(payload.get("local_steps"), torch.Tensor)
            and isinstance(payload.get("block_indices"), torch.Tensor)
        ):
            step_keys = payload["step_keys"].detach().cpu().long().view(-1).tolist()
            local_steps = payload["local_steps"].detach().cpu().long().view(-1).tolist()
            block_indices = payload["block_indices"].detach().cpu().long().view(-1).tolist()
            if len(step_keys) == len(local_steps) == len(block_indices):
                entries = [
                    (int(s), int(local), int(block))
                    for s, local, block in zip(step_keys, local_steps, block_indices, strict=True)
                ]
                matches = [entry for entry in entries if entry[0] == int(step_key)]
                if matches:
                    if future_offset <= 0:
                        return int(step_key)
                    current_block_idx = int(matches[0][2])
                    ordered_entries = sorted(
                        (
                            (block_idx, local_step, candidate_step_key)
                            for candidate_step_key, local_step, block_idx in entries
                            if int(block_idx) == current_block_idx
                        ),
                        key=lambda item: (item[0], item[1]),
                    )
                    ordered_step_keys = [int(candidate_step_key) for _, _, candidate_step_key in ordered_entries]
                    if int(step_key) in ordered_step_keys:
                        current_idx = ordered_step_keys.index(int(step_key))
                    else:
                        return None
                    target_idx = current_idx + max(0, int(future_offset))
                    if target_idx < len(ordered_step_keys):
                        return int(ordered_step_keys[target_idx])
                    if require_closed_block_for_final and len(ordered_step_keys) < max(1, int(block_length or 1)):
                        return None
                    if allow_final_fallback:
                        return int(ordered_step_keys[-1])
                    return None

        return AgentLoopWorker._sdar_same_block_future_teacher_step_key(
            int(step_key),
            AgentLoopWorker._sdar_replay_payload_step_keys(payload),
            int(future_offset),
            int(block_length),
            allow_final_fallback=allow_final_fallback,
            require_closed_block_for_final=require_closed_block_for_final,
        )

    @staticmethod
    def _filter_sdar_replay_payload_new_steps(
        payload: Any,
        emitted_step_keys: set[int],
        *,
        future_offset: int = 0,
        block_length: int = 1,
        step_stride: int = 1,
        step_offset: int = 0,
        train_all_missing_with_final: bool = False,
        accumulate_teacher_blocks: bool = False,
    ) -> Any:
        def _payload_step_keys(value: Any) -> set[int]:
            if isinstance(value, dict) and value.get("format") == "sglang_sdar_replay_tensor_batch_v1":
                step_keys = value.get("step_keys")
                if isinstance(step_keys, torch.Tensor):
                    return {int(step_key) for step_key in step_keys.detach().cpu().long().view(-1).tolist()}
                return set()
            if isinstance(value, (list, tuple)):
                out: set[int] = set()
                for item in value:
                    if not isinstance(item, dict):
                        continue
                    meta = item.get("metadata", item)
                    step_key = meta.get("step_key") if isinstance(meta, dict) else None
                    if step_key is not None:
                        out.add(int(step_key))
                return out
            if isinstance(value, dict):
                meta = value.get("metadata", value)
                step_key = meta.get("step_key") if isinstance(meta, dict) else None
                return {int(step_key)} if step_key is not None else set()
            return set()

        available_step_keys = _payload_step_keys(payload)
        available_step_keys_sorted = sorted(int(step_key) for step_key in available_step_keys)
        block_length = max(1, int(block_length or 1))

        def _teacher_step_key(step_key: int) -> int | None:
            if future_offset <= 0:
                return None
            return AgentLoopWorker._sdar_replay_payload_future_teacher_step_key(
                payload,
                int(step_key),
                int(future_offset),
                int(block_length),
                allow_final_fallback=False,
                require_closed_block_for_final=True,
            )

        def _is_ready(step_key: int) -> bool:
            return future_offset <= 0 or _teacher_step_key(step_key) is not None

        ready_step_keys = sorted(
            step_key
            for step_key in available_step_keys
            if step_key not in emitted_step_keys and _is_ready(step_key)
        )
        if not ready_step_keys:
            print(
                "[SDAR_STREAM_DIAG] no_ready_steps "
                f"available={len(available_step_keys)} emitted={len(emitted_step_keys)} "
                f"future_offset={future_offset}",
                flush=True,
            )
            return None

        step_stride = max(1, int(step_stride or 1))
        step_offset = int(step_offset or 0)
        train_step_keys = [
            step_key
            for step_key in ready_step_keys
            if step_stride <= 1 or (step_key - step_offset) % step_stride == 0
        ]
        if not train_step_keys:
            if not train_all_missing_with_final:
                emitted_step_keys.update(ready_step_keys)
            print(
                "[SDAR_STREAM_DIAG] stride_eats "
                f"ready={len(ready_step_keys)} stride={step_stride} offset={step_offset} "
                f"train_all_missing_with_final={train_all_missing_with_final}",
                flush=True,
            )
            return None
        train_step_key_set = set(train_step_keys)
        teacher_step_keys = (
            [int(_teacher_step_key(step_key)) for step_key in train_step_keys]
            if future_offset > 0
            else []
        )
        support_step_key_set = set(teacher_step_keys)
        if accumulate_teacher_blocks and teacher_step_keys:
            for train_step_key, teacher_step_key in zip(train_step_keys, teacher_step_keys):
                lo = min(int(train_step_key), int(teacher_step_key))
                hi = max(int(train_step_key), int(teacher_step_key))
                support_step_key_set.update(
                    int(step_key)
                    for step_key in available_step_keys
                    if lo <= int(step_key) <= hi
                )
        keep_step_key_set = train_step_key_set | support_step_key_set

        if isinstance(payload, dict) and payload.get("format") == "sglang_sdar_replay_tensor_batch_v1":
            step_keys = payload.get("step_keys")
            if not isinstance(step_keys, torch.Tensor):
                return payload
            step_keys_cpu = step_keys.detach().cpu().long().view(-1)
            n = int(step_keys_cpu.numel())
            keep_indices = []
            for idx, step_key in enumerate(step_keys_cpu.tolist()):
                step_key = int(step_key)
                if step_key not in keep_step_key_set:
                    continue
                keep_indices.append(idx)
            if not keep_indices:
                return None
            emitted_step_keys.update(train_step_keys if train_all_missing_with_final else ready_step_keys)
            if len(keep_indices) == n:
                filtered = dict(payload)
                filtered["train_step_keys"] = torch.tensor(train_step_keys, dtype=torch.long)
                if teacher_step_keys:
                    filtered["teacher_step_keys"] = torch.tensor(teacher_step_keys, dtype=torch.long)
                if accumulate_teacher_blocks:
                    filtered["trace_future_accumulate_teacher_blocks"] = True
                return filtered
            index = torch.tensor(keep_indices, dtype=torch.long)
            filtered = dict(payload)
            for key, value in payload.items():
                if isinstance(value, torch.Tensor) and value.dim() >= 1 and int(value.shape[0]) == n:
                    filtered[key] = value.index_select(0, index.to(value.device))
                elif isinstance(value, list) and len(value) == n:
                    filtered[key] = [value[i] for i in keep_indices]
            filtered["train_step_keys"] = torch.tensor(train_step_keys, dtype=torch.long)
            if teacher_step_keys:
                filtered["teacher_step_keys"] = torch.tensor(teacher_step_keys, dtype=torch.long)
            if accumulate_teacher_blocks:
                filtered["trace_future_accumulate_teacher_blocks"] = True
            return filtered

        if isinstance(payload, (list, tuple)):
            filtered_items = []
            for item in payload:
                if not isinstance(item, dict):
                    filtered_items.append(item)
                    continue
                meta = item.get("metadata", item)
                step_key = meta.get("step_key") if isinstance(meta, dict) else None
                if step_key is None:
                    filtered_items.append(item)
                    continue
                step_key = int(step_key)
                if step_key not in keep_step_key_set:
                    continue
                filtered_items.append(item)
            if not filtered_items:
                return None
            emitted_step_keys.update(train_step_keys if train_all_missing_with_final else ready_step_keys)
            return type(payload)(filtered_items) if isinstance(payload, tuple) else filtered_items

        if isinstance(payload, dict):
            meta = payload.get("metadata", payload)
            step_key = meta.get("step_key") if isinstance(meta, dict) else None
            if step_key is not None:
                step_key = int(step_key)
                if step_key not in train_step_key_set:
                    return None
                emitted_step_keys.update(train_step_keys if train_all_missing_with_final else ready_step_keys)
        return payload

    @staticmethod
    def _filter_sdar_replay_payload_tail_steps(
        payload: Any,
        emitted_step_keys: set[int],
        *,
        future_offset: int = 0,
        block_length: int = 1,
        step_stride: int = 1,
        step_offset: int = 0,
        tail_policy: str = "strict_skip",
        train_all_missing_with_final: bool = False,
        train_final_step_with_final: bool = False,
        accumulate_teacher_blocks: bool = False,
    ) -> Any:
        if future_offset <= 0:
            return None
        if AgentLoopWorker._normalize_trace_future_tail_policy(tail_policy) != "last_available_skip_final":
            return None
        if not (
            isinstance(payload, dict)
            and payload.get("format") == "sglang_sdar_replay_tensor_batch_v1"
        ):
            return None
        step_keys = payload.get("step_keys")
        if not isinstance(step_keys, torch.Tensor):
            return None
        step_keys_cpu = step_keys.detach().cpu().long().view(-1)
        if int(step_keys_cpu.numel()) <= 1:
            return None

        available_step_keys = sorted({int(step_key) for step_key in step_keys_cpu.tolist()})
        last_step_key = int(available_step_keys[-1])
        block_length = max(1, int(block_length or 1))

        def _same_block_tail_teacher_step_key(step_key: int) -> int | None:
            return AgentLoopWorker._sdar_replay_payload_future_teacher_step_key(
                payload,
                int(step_key),
                int(future_offset),
                int(block_length),
                allow_final_fallback=True,
            )

        step_stride = max(1, int(step_stride or 1))
        step_offset = int(step_offset or 0)
        train_step_keys = [
            step_key
            for step_key in available_step_keys
            if step_key not in emitted_step_keys
            and (
                step_key < last_step_key
                or (train_all_missing_with_final and train_final_step_with_final and step_key == last_step_key)
            )
            and (
                train_all_missing_with_final
                or _same_block_tail_teacher_step_key(step_key) is not None
            )
            and (
                train_all_missing_with_final
                or step_stride <= 1
                or (step_key - step_offset) % step_stride == 0
            )
        ]
        tail_coverage_metrics: dict[str, Any] = {}
        if train_all_missing_with_final:
            expected_nonfinal_step_keys = [
                step_key
                for step_key in available_step_keys
                if step_key not in emitted_step_keys and step_key < last_step_key
            ]
            expected_final_step_keys = [
                step_key
                for step_key in available_step_keys
                if (
                    train_final_step_with_final
                    and step_key not in emitted_step_keys
                    and step_key == last_step_key
                )
            ]
            train_step_key_set_for_metrics = {int(step_key) for step_key in train_step_keys}
            emitted_nonfinal_step_keys = [
                step_key
                for step_key in expected_nonfinal_step_keys
                if step_key in train_step_key_set_for_metrics
            ]
            emitted_final_step_keys = [
                step_key
                for step_key in expected_final_step_keys
                if step_key in train_step_key_set_for_metrics
            ]
            missing_nonfinal_step_keys = [
                step_key
                for step_key in expected_nonfinal_step_keys
                if step_key not in train_step_key_set_for_metrics
            ]
            tail_coverage_metrics = {
                "tail_expected_nonfinal_steps": int(len(expected_nonfinal_step_keys)),
                "tail_emitted_nonfinal_steps": int(len(emitted_nonfinal_step_keys)),
                "tail_missing_nonfinal_steps": int(len(missing_nonfinal_step_keys)),
                "tail_expected_final_steps": int(len(expected_final_step_keys)),
                "tail_emitted_final_steps": int(len(emitted_final_step_keys)),
                "tail_final_step_key": int(last_step_key),
                "tail_expected_nonfinal_step_keys": expected_nonfinal_step_keys,
                "tail_emitted_nonfinal_step_keys": emitted_nonfinal_step_keys,
                "tail_missing_nonfinal_step_keys": missing_nonfinal_step_keys,
                "tail_expected_final_step_keys": expected_final_step_keys,
                "tail_emitted_final_step_keys": emitted_final_step_keys,
            }
            if missing_nonfinal_step_keys:
                raise RuntimeError(
                    "SGLang streaming SDAR trace_future tail coverage invariant failed: "
                    f"expected_nonfinal_step_keys={expected_nonfinal_step_keys} "
                    f"train_step_keys={train_step_keys} "
                    f"missing_nonfinal_step_keys={missing_nonfinal_step_keys} "
                    f"final_step_key={last_step_key}"
                )
        if not train_step_keys:
            return None

        teacher_step_keys_maybe = [
            _same_block_tail_teacher_step_key(step_key) for step_key in train_step_keys
        ]
        if any(step_key is None for step_key in teacher_step_keys_maybe):
            return None
        teacher_step_keys = [int(step_key) for step_key in teacher_step_keys_maybe if step_key is not None]
        keep_step_key_set = set(train_step_keys)
        keep_step_key_set.update(teacher_step_keys)
        if accumulate_teacher_blocks:
            for train_step_key in train_step_keys:
                keep_step_key_set.update(
                    int(step_key)
                    for step_key in available_step_keys
                    if int(train_step_key) <= int(step_key) <= int(last_step_key)
                )
        keep_indices = [
            idx
            for idx, step_key in enumerate(step_keys_cpu.tolist())
            if int(step_key) in keep_step_key_set
        ]
        if not keep_indices:
            return None

        filtered = dict(payload)
        if len(keep_indices) != int(step_keys_cpu.numel()):
            index = torch.tensor(keep_indices, dtype=torch.long)
            for key, value in payload.items():
                if (
                    isinstance(value, torch.Tensor)
                    and value.dim() >= 1
                    and int(value.shape[0]) == int(step_keys_cpu.numel())
                ):
                    filtered[key] = value.index_select(0, index.to(value.device))
                elif isinstance(value, list) and len(value) == int(step_keys_cpu.numel()):
                    filtered[key] = [value[i] for i in keep_indices]
        filtered["train_step_keys"] = torch.tensor(train_step_keys, dtype=torch.long)
        filtered["teacher_step_keys"] = torch.tensor(teacher_step_keys, dtype=torch.long)
        filtered["trace_future_tail_policy"] = "last_available_skip_final"
        if accumulate_teacher_blocks:
            filtered["trace_future_accumulate_teacher_blocks"] = True
        if train_all_missing_with_final:
            filtered["trace_future_train_all_missing_with_final"] = True
            filtered.update(tail_coverage_metrics)
        if train_final_step_with_final:
            filtered["trace_future_train_final_step_with_final"] = True
        return filtered

    @staticmethod
    def _filter_sdar_replay_payload_completed_block_steps(
        payload: Any,
        emitted_step_keys: set[int],
        *,
        response_ids: list[int],
        prompt_ids: list[int] | None = None,
        prompt_length: int | None = None,
        block_length: int = 1,
        step_stride: int = 1,
        step_offset: int = 0,
        seen_block_indices: set[int] | None = None,
        allow_partial_response_block: bool = False,
        train_all_missing_with_final: bool = False,
        train_final_step_with_final: bool = False,
        accumulate_teacher_blocks: bool = False,
    ) -> Any:
        if not train_all_missing_with_final:
            return None
        if not (
            isinstance(payload, dict)
            and payload.get("format") == "sglang_sdar_replay_tensor_batch_v1"
        ):
            return None
        step_keys = payload.get("step_keys")
        local_steps = payload.get("local_steps")
        block_indices = payload.get("block_indices")
        if not (
            isinstance(step_keys, torch.Tensor)
            and isinstance(local_steps, torch.Tensor)
            and isinstance(block_indices, torch.Tensor)
        ):
            return None

        step_keys_cpu = step_keys.detach().cpu().long().view(-1)
        local_steps_cpu = local_steps.detach().cpu().long().view(-1)
        block_indices_cpu = block_indices.detach().cpu().long().view(-1)
        n = int(step_keys_cpu.numel())
        if n <= 0 or int(local_steps_cpu.numel()) != n or int(block_indices_cpu.numel()) != n:
            return None

        block_length = max(1, int(block_length or 1))
        if len(response_ids) <= 0:
            return None

        entries = [
            {
                "idx": int(idx),
                "step_key": int(step_key),
                "local_step": int(local_step),
                "block_idx": int(block_idx),
            }
            for idx, (step_key, local_step, block_idx) in enumerate(
                zip(
                    step_keys_cpu.tolist(),
                    local_steps_cpu.tolist(),
                    block_indices_cpu.tolist(),
                    strict=True,
                )
            )
        ]

        step_stride = max(1, int(step_stride or 1))
        step_offset = int(step_offset or 0)
        final_step_by_block: dict[int, int] = {}

        def _final_entry_for_block(block_idx: int) -> dict[str, int] | None:
            block_entries = [
                entry
                for entry in entries
                if int(entry["block_idx"]) == int(block_idx)
            ]
            if not block_entries:
                return None
            return max(
                block_entries,
                key=lambda item: (int(item["local_step"]), int(item["step_key"])),
            )

        if prompt_ids is not None:
            prompt_ids = list(prompt_ids)
            prompt_length_i = int(prompt_length if prompt_length is not None else len(prompt_ids))
            line_i = (len(prompt_ids) // block_length) * block_length
            if line_i > prompt_length_i:
                line_i = prompt_length_i
            prompt_tail_length = max(0, len(prompt_ids) - line_i)
            tail_len = prompt_tail_length + len(response_ids)
            full_L1 = 0 if tail_len <= 0 else ((tail_len + block_length - 1) // block_length) * block_length
            response_start = int(prompt_tail_length)
            response_end = response_start + len(response_ids)
            states = AgentLoopWorker._sdar_compact_replay_to_states(payload)
            for block_idx in sorted({int(entry["block_idx"]) for entry in entries}):
                final_entry = _final_entry_for_block(block_idx)
                if final_entry is None:
                    continue
                item = states.get(int(final_entry["step_key"]))
                if item is None:
                    continue
                span = AgentLoopWorker._sdar_trace_item_tail_span(
                    item,
                    rollout_prefix_len=line_i,
                    L1=full_L1,
                    block_size=block_length,
                )
                if span is None:
                    continue
                dst_start, copy_len = span
                dst_end = int(dst_start) + int(copy_len)
                label_start = max(int(dst_start), response_start)
                label_end = min(dst_end, response_end)
                if label_end <= label_start:
                    continue
                if not allow_partial_response_block and dst_end > response_end:
                    continue
                final_step_by_block[int(block_idx)] = int(final_entry["step_key"])
        else:
            completed_block_count = int(len(response_ids)) // block_length
            if completed_block_count <= 0:
                return None
            if seen_block_indices:
                completed_block_set = set(
                    sorted(int(block_idx) for block_idx in seen_block_indices)[:completed_block_count]
                )
            else:
                observed_blocks = sorted({int(entry["block_idx"]) for entry in entries})
                completed_block_set = set(observed_blocks[:completed_block_count])
            for block_idx in sorted(
                {
                    int(entry["block_idx"])
                    for entry in entries
                    if int(entry["block_idx"]) in completed_block_set
                }
            ):
                final_entry = _final_entry_for_block(block_idx)
                if final_entry is not None:
                    final_step_by_block[int(block_idx)] = int(final_entry["step_key"])
        if not final_step_by_block:
            return None

        train_step_keys: list[int] = []
        teacher_step_keys: list[int] = []
        expected_nonfinal_step_keys: list[int] = []
        expected_final_step_keys: list[int] = []
        emitted_nonfinal_step_keys: list[int] = []
        emitted_final_step_keys: list[int] = []
        train_block_by_step: dict[int, int] = {}
        ordered_entries = sorted(
            entries,
            key=lambda item: (
                int(item["block_idx"]),
                int(item["local_step"]),
                int(item["step_key"]),
            ),
        )
        for entry in ordered_entries:
            block_idx = int(entry["block_idx"])
            if block_idx not in final_step_by_block:
                continue
            step_key = int(entry["step_key"])
            if step_key in emitted_step_keys:
                continue
            final_step_key = int(final_step_by_block[block_idx])
            is_final_step = step_key == final_step_key
            if is_final_step:
                if not train_final_step_with_final:
                    continue
                expected_final_step_keys.append(step_key)
            else:
                expected_nonfinal_step_keys.append(step_key)
            if (
                not train_all_missing_with_final
                and step_stride > 1
                and (step_key - step_offset) % step_stride != 0
            ):
                continue
            train_step_keys.append(step_key)
            teacher_step_keys.append(final_step_key)
            train_block_by_step[step_key] = block_idx
            if is_final_step:
                emitted_final_step_keys.append(step_key)
            else:
                emitted_nonfinal_step_keys.append(step_key)

        if not train_step_keys:
            return None

        keep_step_key_set = set(train_step_keys)
        keep_step_key_set.update(teacher_step_keys)
        if accumulate_teacher_blocks:
            available_step_keys_by_block: dict[int, list[int]] = {}
            for entry in entries:
                available_step_keys_by_block.setdefault(int(entry["block_idx"]), []).append(
                    int(entry["step_key"])
                )
            for train_step_key, teacher_step_key in zip(train_step_keys, teacher_step_keys, strict=True):
                block_idx = int(train_block_by_step[int(train_step_key)])
                lo = min(int(train_step_key), int(teacher_step_key))
                hi = max(int(train_step_key), int(teacher_step_key))
                keep_step_key_set.update(
                    int(step_key)
                    for step_key in available_step_keys_by_block.get(block_idx, [])
                    if lo <= int(step_key) <= hi
                )

        keep_indices = [
            int(entry["idx"])
            for entry in entries
            if int(entry["step_key"]) in keep_step_key_set
        ]
        if not keep_indices:
            return None

        emitted_step_keys.update(train_step_keys)
        filtered = dict(payload)
        if len(keep_indices) != n:
            index = torch.tensor(keep_indices, dtype=torch.long)
            for key, value in payload.items():
                if (
                    isinstance(value, torch.Tensor)
                    and value.dim() >= 1
                    and int(value.shape[0]) == n
                ):
                    filtered[key] = value.index_select(0, index.to(value.device))
                elif isinstance(value, list) and len(value) == n:
                    filtered[key] = [value[i] for i in keep_indices]
        filtered["train_step_keys"] = torch.tensor(train_step_keys, dtype=torch.long)
        filtered["teacher_step_keys"] = torch.tensor(teacher_step_keys, dtype=torch.long)
        filtered["trace_future_tail_policy"] = "last_available_skip_final"
        filtered["trace_future_train_all_missing_with_final"] = True
        filtered["trace_future_block_level_final_flush"] = True
        completed_blocks = sorted(int(block_idx) for block_idx in final_step_by_block.keys())
        filtered["block_level_completed_block_count"] = int(len(completed_blocks))
        filtered["block_level_completed_block_indices"] = completed_blocks
        if accumulate_teacher_blocks:
            filtered["trace_future_accumulate_teacher_blocks"] = True
        if train_final_step_with_final:
            filtered["trace_future_train_final_step_with_final"] = True
        missing_nonfinal_step_keys = [
            step_key
            for step_key in expected_nonfinal_step_keys
            if step_key not in set(emitted_nonfinal_step_keys)
        ]
        filtered.update(
            {
                "tail_expected_nonfinal_steps": int(len(expected_nonfinal_step_keys)),
                "tail_emitted_nonfinal_steps": int(len(emitted_nonfinal_step_keys)),
                "tail_missing_nonfinal_steps": int(len(missing_nonfinal_step_keys)),
                "tail_expected_final_steps": int(len(expected_final_step_keys)),
                "tail_emitted_final_steps": int(len(emitted_final_step_keys)),
                "tail_expected_nonfinal_step_keys": expected_nonfinal_step_keys,
                "tail_emitted_nonfinal_step_keys": emitted_nonfinal_step_keys,
                "tail_missing_nonfinal_step_keys": missing_nonfinal_step_keys,
                "tail_expected_final_step_keys": expected_final_step_keys,
                "tail_emitted_final_step_keys": emitted_final_step_keys,
            }
        )
        return filtered

    @staticmethod
    def _prune_sdar_replay_payload_emitted_steps(payload: Any, emitted_step_keys: set[int]) -> Any:
        if payload is None or not emitted_step_keys:
            return payload
        if isinstance(payload, dict) and payload.get("format") == "sglang_sdar_replay_tensor_batch_v1":
            step_keys = payload.get("step_keys")
            if not isinstance(step_keys, torch.Tensor):
                return payload
            step_keys_cpu = step_keys.detach().cpu().long().view(-1)
            n = int(step_keys_cpu.numel())
            keep_indices = [
                idx
                for idx, step_key in enumerate(step_keys_cpu.tolist())
                if int(step_key) not in emitted_step_keys
            ]
            if not keep_indices:
                return None
            if len(keep_indices) == n:
                return payload
            index = torch.tensor(keep_indices, dtype=torch.long)
            pruned = dict(payload)
            for key, value in payload.items():
                if isinstance(value, torch.Tensor) and value.dim() >= 1 and int(value.shape[0]) == n:
                    pruned[key] = value.index_select(0, index.to(value.device))
                elif isinstance(value, list) and len(value) == n:
                    pruned[key] = [value[i] for i in keep_indices]
            return pruned
        if isinstance(payload, (list, tuple)):
            kept = []
            for item in payload:
                if not isinstance(item, dict):
                    kept.append(item)
                    continue
                meta = item.get("metadata", item)
                step_key = meta.get("step_key") if isinstance(meta, dict) else None
                if step_key is not None and int(step_key) in emitted_step_keys:
                    continue
                kept.append(item)
            if not kept:
                return None
            return type(payload)(kept) if isinstance(payload, tuple) else kept
        if isinstance(payload, dict):
            meta = payload.get("metadata", payload)
            step_key = meta.get("step_key") if isinstance(meta, dict) else None
            if step_key is not None and int(step_key) in emitted_step_keys:
                return None
        return payload

    @staticmethod
    def _sdar_tensor(value: Any, dtype: torch.dtype | None = None) -> torch.Tensor | None:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu()
        elif isinstance(value, np.ndarray):
            tensor = torch.as_tensor(value)
        else:
            try:
                tensor = torch.as_tensor(value)
            except Exception:
                return None
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        return tensor

    @staticmethod
    def _sdar_owned_tensor_row(tensor: torch.Tensor, index: int) -> torch.Tensor:
        """Detach a row from a compact batch's backing storage."""
        return tensor[index].detach().cpu().clone()

    @classmethod
    def _sdar_compact_replay_to_states(cls, payload: Any) -> dict[int, dict[str, Any]]:
        payload = cls._materialize_sdar_replay_payload(payload)
        if not (
            isinstance(payload, dict)
            and payload.get("format") == "sglang_sdar_replay_tensor_batch_v1"
        ):
            return {}
        step_keys = cls._sdar_tensor(payload.get("step_keys"), dtype=torch.long)
        if step_keys is None:
            return {}
        step_keys = step_keys.view(-1)
        n = int(step_keys.numel())

        def _batched_tensor(key: str, dtype: torch.dtype | None = None) -> torch.Tensor | None:
            tensor = cls._sdar_tensor(payload.get(key), dtype=dtype)
            if tensor is None or tensor.dim() < 1 or int(tensor.shape[0]) != n:
                return None
            return tensor

        positions = _batched_tensor("positions", dtype=torch.long)
        valid = _batched_tensor("valid", dtype=torch.bool)
        input_ids = _batched_tensor("input_ids", dtype=torch.long)
        self_conditioning_signal = _batched_tensor("self_conditioning_signal")
        logits_source_hidden = _batched_tensor("logits_source_hidden")
        adjacent_entropy_pair_valid = _batched_tensor(
            "adjacent_entropy_pair_valid",
            dtype=torch.bool,
        )
        adjacent_entropy_prev_step_key = _batched_tensor(
            "adjacent_entropy_prev_step_key",
            dtype=torch.long,
        )
        adjacent_entropy_pair_local_step = _batched_tensor(
            "adjacent_entropy_pair_local_step",
            dtype=torch.long,
        )
        adjacent_entropy_prev_mean = _batched_tensor(
            "adjacent_entropy_prev_mean",
            dtype=torch.float32,
        )
        adjacent_entropy_rollout_next_mean = _batched_tensor(
            "adjacent_entropy_rollout_next_mean",
            dtype=torch.float32,
        )
        adjacent_entropy_vocab_size = _batched_tensor(
            "adjacent_entropy_vocab_size",
            dtype=torch.long,
        )
        adjacent_entropy_next_mask = _batched_tensor(
            "adjacent_entropy_next_mask",
            dtype=torch.bool,
        )
        global_aec_pair_valid = _batched_tensor(
            "global_aec_pair_valid",
            dtype=torch.bool,
        )
        global_aec_prev_step_key = _batched_tensor(
            "global_aec_prev_step_key",
            dtype=torch.long,
        )
        global_aec_prev_mean = _batched_tensor(
            "global_aec_prev_mean",
            dtype=torch.float32,
        )
        global_aec_rollout_next_mean = _batched_tensor(
            "global_aec_rollout_next_mean",
            dtype=torch.float32,
        )
        global_aec_vocab_size = _batched_tensor(
            "global_aec_vocab_size",
            dtype=torch.long,
        )
        global_aec_next_mask = _batched_tensor(
            "global_aec_next_mask",
            dtype=torch.bool,
        )

        def _value(key: str, index: int, default: Any) -> Any:
            values = payload.get(key)
            if (
                isinstance(values, torch.Tensor)
                and values.dim() >= 1
                and int(values.shape[0]) == n
            ):
                return values[index].item()
            return default

        def _string(key: str, index: int) -> str:
            values = payload.get(key)
            if isinstance(values, list) and len(values) == n:
                return str(values[index])
            return ""

        states: dict[int, dict[str, Any]] = {}
        for i, step_key_value in enumerate(step_keys.tolist()):
            position = None
            if positions is not None:
                pos_i = positions[i].view(-1)
                if int(pos_i.numel()) == 1:
                    position = int(pos_i.item())
                elif int(pos_i.numel()) >= 2:
                    position = tuple(int(x) for x in pos_i[:2].tolist())
            block_index = int(_value("block_indices", i, 0))
            item: dict[str, Any] = {
                "position": position,
                "local_step": int(_value("local_steps", i, 0)),
                "block_index": block_index,
                "start_offset": int(
                    _value("start_offsets", i, 0)
                ),
                "response_start_index": int(
                    _value("response_start_indices", i, 0)
                ),
                "new_accept_count": int(
                    _value("new_accept_counts", i, 0)
                ),
                "is_final_denoise_row": bool(
                    _value("is_final_denoise_rows", i, False)
                ),
                "is_terminal_commit": bool(
                    _value("is_terminal_commits", i, False)
                ),
                "producer_preselected": bool(
                    _value("producer_preselected_rows", i, False)
                ),
                "producer_preselection_mode": _string(
                    "producer_preselection_modes",
                    i,
                ),
                "producer_source_rows": int(
                    _value("producer_source_rows", i, 0)
                ),
                "producer_rank_selected_rows": int(
                    _value("producer_rank_selected_rows", i, 0)
                ),
                "producer_selected_rows": int(
                    _value("producer_selected_rows", i, 0)
                ),
                "producer_final_rows": int(
                    _value("producer_final_rows", i, 0)
                ),
                "producer_forced_final_rows": int(
                    _value("producer_forced_final_rows", i, 0)
                ),
                "producer_final_already_selected_rows": int(
                    _value(
                        "producer_final_already_selected_rows",
                        i,
                        0,
                    )
                ),
                "producer_source_score_min": int(
                    _value("producer_source_score_min", i, 0)
                ),
                "producer_source_score_sum": int(
                    _value("producer_source_score_sum", i, 0)
                ),
                "producer_source_score_max": int(
                    _value("producer_source_score_max", i, 0)
                ),
                "producer_zero_variance": bool(
                    _value(
                        "producer_zero_variance_groups",
                        i,
                        False,
                    )
                ),
                "producer_block_index": int(
                    _value("producer_block_indices", i, block_index)
                ),
                "producer_state_selection": bool(
                    _value(
                        "producer_state_selection_rows",
                        i,
                        False,
                    )
                ),
                "producer_state_selection_version": int(
                    _value(
                        "producer_state_selection_versions",
                        i,
                        0,
                    )
                ),
                "producer_state_selection_complete": bool(
                    _value(
                        "producer_state_selection_complete_rows",
                        i,
                        False,
                    )
                ),
                "producer_state_selection_mode": _string(
                    "producer_state_selection_modes",
                    i,
                ),
                "producer_state_selection_fraction": float(
                    _value(
                        "producer_state_selection_fractions",
                        i,
                        0.0,
                    )
                ),
                "producer_state_selection_direction": _string(
                    "producer_state_selection_directions",
                    i,
                ),
                "producer_state_selection_score": _string(
                    "producer_state_selection_scores",
                    i,
                ),
                "producer_state_selection_group_fields": _string(
                    "producer_state_selection_group_fields",
                    i,
                ),
                "producer_state_selection_final_policy": _string(
                    "producer_state_selection_final_policies",
                    i,
                ),
                "producer_state_selection_rank_selected": bool(
                    _value(
                        "producer_state_selection_rank_selected_flags",
                        i,
                        False,
                    )
                ),
                "producer_state_selection_forced_final": bool(
                    _value(
                        "producer_state_selection_forced_final_flags",
                        i,
                        False,
                    )
                ),
            }
            if valid is not None:
                item["valid"] = cls._sdar_owned_tensor_row(valid, i)
            if input_ids is not None:
                item["input_ids"] = cls._sdar_owned_tensor_row(input_ids, i)
            if self_conditioning_signal is not None:
                item["self_conditioning_signal"] = cls._sdar_owned_tensor_row(
                    self_conditioning_signal,
                    i,
                )
            if logits_source_hidden is not None:
                item["logits_source_hidden"] = cls._sdar_owned_tensor_row(
                    logits_source_hidden,
                    i,
                )
            if adjacent_entropy_pair_valid is not None:
                item["adjacent_entropy_pair_valid"] = bool(
                    adjacent_entropy_pair_valid[i].view(-1)[0].item()
                )
            if adjacent_entropy_prev_step_key is not None:
                item["adjacent_entropy_prev_step_key"] = int(
                    adjacent_entropy_prev_step_key[i].view(-1)[0].item()
                )
            if adjacent_entropy_pair_local_step is not None:
                item["adjacent_entropy_pair_local_step"] = int(
                    adjacent_entropy_pair_local_step[i].view(-1)[0].item()
                )
            if adjacent_entropy_prev_mean is not None:
                item["adjacent_entropy_prev_mean"] = float(
                    adjacent_entropy_prev_mean[i].view(-1)[0].item()
                )
            if adjacent_entropy_rollout_next_mean is not None:
                item["adjacent_entropy_rollout_next_mean"] = float(
                    adjacent_entropy_rollout_next_mean[i].view(-1)[0].item()
                )
            if adjacent_entropy_vocab_size is not None:
                item["adjacent_entropy_vocab_size"] = int(
                    adjacent_entropy_vocab_size[i].view(-1)[0].item()
                )
            if adjacent_entropy_next_mask is not None:
                item["adjacent_entropy_next_mask"] = cls._sdar_owned_tensor_row(
                    adjacent_entropy_next_mask,
                    i,
                )
            if global_aec_pair_valid is not None:
                item["global_aec_pair_valid"] = bool(
                    global_aec_pair_valid[i].view(-1)[0].item()
                )
            if global_aec_prev_step_key is not None:
                item["global_aec_prev_step_key"] = int(
                    global_aec_prev_step_key[i].view(-1)[0].item()
                )
            if global_aec_prev_mean is not None:
                item["global_aec_prev_mean"] = float(
                    global_aec_prev_mean[i].view(-1)[0].item()
                )
            if global_aec_rollout_next_mean is not None:
                item["global_aec_rollout_next_mean"] = float(
                    global_aec_rollout_next_mean[i].view(-1)[0].item()
                )
            if global_aec_vocab_size is not None:
                item["global_aec_vocab_size"] = int(
                    global_aec_vocab_size[i].view(-1)[0].item()
                )
            if global_aec_next_mask is not None:
                item["global_aec_next_mask"] = cls._sdar_owned_tensor_row(
                    global_aec_next_mask,
                    i,
                )
            states[int(step_key_value)] = item
        return states

    @staticmethod
    def _sdar_position_start(position: Any) -> int | None:
        if position is None:
            return None
        if isinstance(position, torch.Tensor):
            position = position.detach().cpu().tolist()
        if isinstance(position, (tuple, list)) and len(position) > 0:
            return int(position[0])
        return int(position)

    @classmethod
    def _sdar_replay_dst_start(
        cls,
        position: Any,
        *,
        rollout_prefix_len: int,
        L1: int,
        block_size: int,
        block_idx: int = 0,
    ) -> int:
        pos_start = cls._sdar_position_start(position)
        if pos_start is None:
            return int(block_idx) * int(block_size)
        if 0 <= pos_start < L1:
            return pos_start
        return pos_start - int(rollout_prefix_len)

    @classmethod
    def _sdar_trace_item_tail_span(
        cls,
        item: dict[str, Any],
        *,
        rollout_prefix_len: int,
        L1: int,
        block_size: int,
    ) -> tuple[int, int] | None:
        dst_start = cls._sdar_replay_dst_start(
            item.get("position", None),
            rollout_prefix_len=rollout_prefix_len,
            L1=L1,
            block_size=block_size,
            block_idx=0,
        )
        tensor_len = None
        for key in ("input_ids", "self_conditioning_signal", "logits_source_hidden"):
            value = item.get(key)
            if isinstance(value, torch.Tensor) and value.dim() >= 1:
                tensor_len = int(value.shape[0])
                break
        if tensor_len is None:
            tensor_len = int(block_size)
        src_start = 0
        if dst_start < 0:
            src_start = -dst_start
            dst_start = 0
        if dst_start >= L1 or src_start >= tensor_len:
            return None
        copy_len = min(tensor_len - src_start, L1 - dst_start)
        if copy_len <= 0:
            return None
        return int(dst_start), int(copy_len)

    @classmethod
    def _sdar_source_replay_metrics(
        cls,
        replay_payload: Any,
        *,
        prompt_ids: list[int],
        response_ids: list[int],
        prompt_length: int,
        block_length: int,
        train_step_keys: list[int],
    ) -> dict[str, Any]:
        block_length = max(int(block_length or 1), 1)
        line_i = (len(prompt_ids) // block_length) * block_length
        if line_i > int(prompt_length):
            line_i = int(prompt_length)
        p_tail = prompt_ids[line_i:]
        tail_len = len(p_tail) + len(response_ids)
        full_L1 = 0 if tail_len <= 0 else ((tail_len + block_length - 1) // block_length) * block_length
        label_tail = torch.zeros((full_L1,), dtype=torch.bool)
        response_start = len(p_tail)
        response_end = response_start + len(response_ids)
        if response_end > response_start:
            label_tail[response_start:response_end] = True

        states = cls._sdar_compact_replay_to_states(replay_payload)
        rows = 0
        active_tokens = 0
        active_step_keys: list[int] = []
        for train_step_key in train_step_keys:
            item = states.get(int(train_step_key))
            if item is None:
                continue
            span = cls._sdar_trace_item_tail_span(
                item,
                rollout_prefix_len=line_i,
                L1=full_L1,
                block_size=block_length,
            )
            if span is None:
                continue
            dst_start, copy_len = span
            span_mask = label_tail[dst_start : dst_start + copy_len]
            active_count = int(span_mask.sum().item())
            if active_count <= 0:
                continue
            rows += 1
            active_tokens += active_count
            active_step_keys.append(int(train_step_key))

        return {
            "source_replay_rows": int(rows),
            "source_replay_active_tokens": int(active_tokens),
            "source_replay_step_count": int(len(train_step_keys)),
            "source_replay_active_step_count": int(len(active_step_keys)),
            "source_replay_active_step_keys": active_step_keys,
        }

    def _sdar_pending_replay_metrics(
        self,
        *,
        pending_replay_payload: Any,
        all_seen_step_keys: set[int],
        emitted_step_keys: set[int],
        future_offset: int,
        future_skip_tail_steps: int,
        step_stride: int = 1,
        step_offset: int = 0,
        sample_idx: int,
        streamed_chunks: int,
        prompt_ids: list[int],
        response_ids: list[int],
        prompt_length: int,
        block_length: int,
        tail_policy: str = "strict_skip",
        train_all_missing_with_final: bool = False,
        train_final_step_with_final: bool = False,
    ) -> dict[str, Any]:
        pending_step_keys = sorted(self._sdar_replay_payload_step_keys(pending_replay_payload))
        replay_metrics = self._sdar_source_replay_metrics(
            pending_replay_payload,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            prompt_length=prompt_length,
            block_length=block_length,
            train_step_keys=pending_step_keys,
        )

        ready_emitted_step_keys = set(int(k) for k in emitted_step_keys)
        ready_payload = self._filter_sdar_replay_payload_new_steps(
            self._shallow_copy_sdar_replay_payload(pending_replay_payload),
            ready_emitted_step_keys,
            future_offset=future_offset,
            block_length=block_length,
            step_stride=step_stride,
            step_offset=step_offset,
            train_all_missing_with_final=train_all_missing_with_final,
        )
        ready_step_keys = sorted(set(ready_emitted_step_keys) - set(int(k) for k in emitted_step_keys))
        ready_metrics = self._sdar_source_replay_metrics(
            ready_payload,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            prompt_length=prompt_length,
            block_length=block_length,
            train_step_keys=ready_step_keys,
        )
        ready_active_step_keys = sorted(
            int(k) for k in ready_metrics.get("source_replay_active_step_keys", []) or []
        )

        unsupported_step_keys = sorted(set(pending_step_keys) - set(ready_step_keys))
        unsupported_metrics = self._sdar_source_replay_metrics(
            pending_replay_payload,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            prompt_length=prompt_length,
            block_length=block_length,
            train_step_keys=unsupported_step_keys,
        )
        tail_coverage_metrics: dict[str, Any] = {}
        if (
            train_all_missing_with_final
            and future_offset > 0
            and self._normalize_trace_future_tail_policy(tail_policy) == "last_available_skip_final"
        ):
            available_step_keys = pending_step_keys
            last_step_key = int(available_step_keys[-1]) if available_step_keys else None
            expected_nonfinal_step_keys = (
                [
                    step_key
                    for step_key in available_step_keys
                    if step_key not in emitted_step_keys and step_key < int(last_step_key)
                ]
                if last_step_key is not None
                else []
            )
            tail_coverage_metrics = {
                "tail_expected_nonfinal_steps": int(len(expected_nonfinal_step_keys)),
                "tail_emitted_nonfinal_steps": 0,
                "tail_missing_nonfinal_steps": int(len(expected_nonfinal_step_keys)),
                "tail_expected_final_steps": int(
                    1
                    if train_final_step_with_final
                    and last_step_key is not None
                    and int(last_step_key) not in emitted_step_keys
                    else 0
                ),
                "tail_emitted_final_steps": 0,
                "tail_expected_nonfinal_step_keys": expected_nonfinal_step_keys,
                "tail_emitted_nonfinal_step_keys": [],
                "tail_missing_nonfinal_step_keys": expected_nonfinal_step_keys,
            }
            if last_step_key is not None:
                tail_coverage_metrics["tail_final_step_key"] = int(last_step_key)
        metrics = {
            "format": "split_sdar_stream_metrics_v1",
            "sample_index": int(sample_idx),
            "streamed_chunks": int(streamed_chunks),
            "source_pending_replay_steps": int(len(pending_step_keys)),
            "source_pending_replay_rows": int(replay_metrics.get("source_replay_rows", 0)),
            "source_pending_replay_active_tokens": int(
                replay_metrics.get("source_replay_active_tokens", 0)
            ),
            "source_pending_ready_steps": int(len(ready_step_keys)),
            "source_pending_ready_rows": int(ready_metrics.get("source_replay_rows", 0)),
            "source_pending_ready_active_tokens": int(
                ready_metrics.get("source_replay_active_tokens", 0)
            ),
            "source_pending_ready_active_steps": int(
                ready_metrics.get("source_replay_active_step_count", 0)
            ),
            "source_pending_unsupported_steps": int(len(unsupported_step_keys)),
            "source_pending_unsupported_rows": int(
                unsupported_metrics.get("source_replay_rows", 0)
            ),
            "source_pending_unsupported_active_tokens": int(
                unsupported_metrics.get("source_replay_active_tokens", 0)
            ),
            "source_pending_skipped_tail_steps": int(len(unsupported_step_keys)),
            "source_pending_unexpected_steps": int(len(ready_active_step_keys)),
            "source_pending_future_offset": int(future_offset),
            "source_pending_future_skip_tail_steps": int(future_skip_tail_steps),
            "source_pending_stream_replay_step_stride": int(max(1, step_stride or 1)),
            "source_pending_stream_replay_step_offset": int(step_offset or 0),
            "source_pending_seen_steps": int(len(all_seen_step_keys)),
            "pending_step_keys": pending_step_keys,
            "ready_step_keys": ready_step_keys,
            "ready_active_step_keys": ready_active_step_keys,
            "unsupported_step_keys": unsupported_step_keys,
            "skipped_tail_step_keys": unsupported_step_keys,
            "unexpected_step_keys": ready_active_step_keys,
        }
        metrics.update(tail_coverage_metrics)
        return metrics

    @staticmethod
    def _drop_streamed_sdar_replay_state_(data: DataProto) -> None:
        if data.batch is not None:
            drop_batch_keys = [
                key
                for key in (
                    "rollout_self_conditioning_signal",
                    "rollout_logits_source_hidden",
                )
                if key in data.batch.keys()
            ]
            if drop_batch_keys:
                data.pop(batch_keys=drop_batch_keys)
        for key in (
            "rollout_self_conditioning",
            "sglang_sdar_replay_tensors",
            "sglang_sdar_replay_tensors_compact",
            "replay_tensors",
            "replay_tensors_compact",
            "compact_replay_tensors",
        ):
            data.non_tensor_batch.pop(key, None)

    @staticmethod
    def _trim_streamed_sdar_training_state_(data: DataProto, future_offset: int) -> None:
        stream_markers = data.non_tensor_batch.get("sglang_sdar_stream_chunk")
        train_step_keys: set[int] = set()
        teacher_step_keys: set[int] = set()
        if isinstance(stream_markers, np.ndarray):
            marker_iter = stream_markers.tolist()
        elif isinstance(stream_markers, (list, tuple)):
            marker_iter = stream_markers
        else:
            marker_iter = [stream_markers]
        for marker in marker_iter:
            if not isinstance(marker, dict):
                continue
            for step_key in marker.get("train_step_keys", []) or []:
                train_step_keys.add(int(step_key))
            for step_key in marker.get("teacher_step_keys", []) or []:
                teacher_step_keys.add(int(step_key))
        if not train_step_keys:
            return
        keep_step_keys = set(train_step_keys)
        if teacher_step_keys:
            keep_step_keys.update(teacher_step_keys)
        elif future_offset > 0:
            keep_step_keys.update(step_key + int(future_offset) for step_key in train_step_keys)

        states_arr = data.non_tensor_batch.get("rollout_self_conditioning")
        if states_arr is None:
            return
        if isinstance(states_arr, np.ndarray):
            states_iter = states_arr.tolist()
        elif isinstance(states_arr, (list, tuple)):
            states_iter = list(states_arr)
        else:
            states_iter = [states_arr]
        trimmed_states = []
        for states in states_iter:
            if not isinstance(states, dict):
                trimmed_states.append(states)
                continue
            trimmed_states.append(
                {
                    step_key: value
                    for step_key, value in states.items()
                    if int(step_key) in keep_step_keys
                }
            )
        trimmed_arr = np.empty((len(trimmed_states),), dtype=object)
        trimmed_arr[:] = trimmed_states
        data.non_tensor_batch["rollout_self_conditioning"] = trimmed_arr

    def _build_sdar_streaming_output(self, raw_chunk: dict[str, Any]) -> AgentLoopOutput:
        prompt_ids = self._as_int_list(raw_chunk.get("prompt_ids"))
        response_ids = self._as_int_list(raw_chunk.get("output_ids") or raw_chunk.get("response_ids"))
        response_length = int(self.config.actor_rollout_ref.rollout.response_length)
        if len(response_ids) > response_length:
            raise RuntimeError(
                "SGLang streaming SDAR replay chunk exceeded rollout.response_length: "
                f"{len(response_ids)} > {response_length}"
            )
        stream_context = raw_chunk.get("stream_context") if isinstance(raw_chunk.get("stream_context"), dict) else {}
        multi_modal_data = stream_context.get("multi_modal_data") or {}
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=[1] * len(response_ids),
            response_logprobs=None,
            routed_experts=None,
            multi_modal_data=multi_modal_data,
            reward_score=None,
            num_turns=2,
            metrics=AgentLoopMetrics(),
        )
        output.extra_fields.update({"turn_scores": [], "tool_rewards": []})
        meta_info = raw_chunk.get("meta_info") if isinstance(raw_chunk.get("meta_info"), dict) else {}
        output.extra_fields["sglang_meta_info"] = meta_info
        output.extra_fields["sglang_sdar_metrics"] = meta_info.get("sglang_sdar_request_metrics")
        output.extra_fields["sglang_sdar_stats_delta"] = meta_info.get(
            "sglang_sdar_stats_delta",
            meta_info.get("sglang_sdar_global_stats_delta"),
        )
        replay_payload = self._extract_sdar_replay_payload(raw_chunk)
        output.extra_fields["sglang_sdar_replay_tensors"] = replay_payload
        output.extra_fields["sglang_sdar_stream_chunk"] = {
            "request_id": raw_chunk.get("request_id"),
            "chunk_index": raw_chunk.get("chunk_index"),
            "queue_index": raw_chunk.get("queue_index"),
            "source": raw_chunk.get("source"),
        }
        for key in (
            "raw_first_push_ts",
            "raw_last_push_ts",
            "raw_ready_push_ts",
            "sample_first_raw_push_ts",
            "request_first_raw_push_ts",
            "raw_first_to_trainer_push_seconds",
            "raw_ready_to_trainer_push_seconds",
            "raw_support_wait_seconds",
            "raw_to_trainer_push_seconds",
            "raw_queue_wait_seconds",
            "direct_ready_wait_seconds",
            "source_replay_rows",
            "source_replay_active_tokens",
            "source_replay_items",
            "source_replay_chunks",
            "source_replay_step_count",
            "source_replay_active_step_count",
            "source_direct_compact_fallback_items",
            "source_direct_compact_fallback_rows",
            "source_direct_compact_fallback_active_tokens",
        ):
            if key in raw_chunk:
                output.extra_fields["sglang_sdar_stream_chunk"][key] = float(raw_chunk.get(key) or 0.0)
        if raw_chunk.get("direct_compact_fallback_reason") is not None:
            output.extra_fields["sglang_sdar_stream_chunk"]["direct_compact_fallback_reason"] = str(
                raw_chunk.get("direct_compact_fallback_reason")
            )
        if isinstance(replay_payload, dict):
            train_step_keys = replay_payload.get("train_step_keys")
            if isinstance(train_step_keys, torch.Tensor):
                output.extra_fields["sglang_sdar_stream_chunk"]["train_step_keys"] = [
                    int(step_key)
                    for step_key in train_step_keys.detach().cpu().long().view(-1).tolist()
                ]
            elif isinstance(train_step_keys, (list, tuple)):
                output.extra_fields["sglang_sdar_stream_chunk"]["train_step_keys"] = [
                    int(step_key)
                    for step_key in train_step_keys
                ]
            teacher_step_keys = replay_payload.get("teacher_step_keys")
            if isinstance(teacher_step_keys, torch.Tensor):
                output.extra_fields["sglang_sdar_stream_chunk"]["teacher_step_keys"] = [
                    int(step_key)
                    for step_key in teacher_step_keys.detach().cpu().long().view(-1).tolist()
                ]
            elif isinstance(teacher_step_keys, (list, tuple)):
                output.extra_fields["sglang_sdar_stream_chunk"]["teacher_step_keys"] = [
                    int(step_key)
                    for step_key in teacher_step_keys
                ]
            if bool(replay_payload.get("trace_future_train_all_missing_with_final", False)):
                output.extra_fields["sglang_sdar_stream_chunk"][
                    "trace_future_train_all_missing_with_final"
                ] = True
        return output

    def _build_sdar_direct_compact_replay_item(
        self,
        raw_chunk: dict[str, Any],
        replay_payload: Any,
        *,
        sample_idx: int,
        streamed_chunks: int,
        future_offset: int,
        stream_cache_key: str | None = None,
        delta_teacher_cache_sent_step_keys: set[int] | None = None,
    ) -> tuple[dict[str, Any] | None, str]:
        if not (
            isinstance(replay_payload, dict)
            and replay_payload.get("format") == "sglang_sdar_replay_tensor_batch_v1"
        ):
            return None, "invalid_replay_payload"
        step_keys = replay_payload.get("step_keys")
        if not isinstance(step_keys, torch.Tensor):
            return None, "missing_step_keys"
        train_step_keys = replay_payload.get("train_step_keys")
        if isinstance(train_step_keys, torch.Tensor):
            train_step_keys = [
                int(step_key)
                for step_key in train_step_keys.detach().cpu().long().view(-1).tolist()
            ]
        elif isinstance(train_step_keys, np.ndarray):
            train_step_keys = [
                int(step_key.item() if hasattr(step_key, "item") else step_key)
                for step_key in train_step_keys.reshape(-1).tolist()
            ]
        elif isinstance(train_step_keys, (list, tuple, set)):
            train_step_keys = [
                int(step_key.item() if hasattr(step_key, "item") else step_key)
                for step_key in train_step_keys
            ]
        elif train_step_keys is not None:
            train_step_keys = [int(train_step_keys.item() if hasattr(train_step_keys, "item") else train_step_keys)]
        else:
            train_step_keys = []
        train_step_keys = list(dict.fromkeys(train_step_keys))
        if not train_step_keys:
            return None, "missing_train_step_keys"
        rollout_cfg = self.config.actor_rollout_ref.rollout
        block_length = int((rollout_cfg.get("engine_kwargs", {}) or {}).get("block_length", 1) or 1)
        block_length = max(block_length, 1)
        teacher_step_keys = self._as_int_list(replay_payload.get("teacher_step_keys"))
        if teacher_step_keys and len(teacher_step_keys) != len(train_step_keys):
            return None, "teacher_step_key_count_mismatch"

        replay_step_keys = {
            int(step_key)
            for step_key in step_keys.detach().cpu().long().view(-1).tolist()
        }
        if not teacher_step_keys and future_offset > 0:
            teacher_step_keys_maybe = [
                self._sdar_replay_payload_future_teacher_step_key(
                    replay_payload,
                    int(step_key),
                    int(future_offset),
                    int(block_length),
                    allow_final_fallback=True,
                )
                for step_key in train_step_keys
            ]
            if any(step_key is None for step_key in teacher_step_keys_maybe):
                return None, "missing_future_support"
            teacher_step_keys = [
                int(step_key) for step_key in teacher_step_keys_maybe if step_key is not None
            ]
        requires_teacher_logits = self._sdar_stream_requires_teacher_logits()
        accumulate_teacher_blocks = bool(
            replay_payload.get("trace_future_accumulate_teacher_blocks", False)
            if isinstance(replay_payload, dict)
            else False
        )
        if teacher_step_keys:
            required_support_step_keys = {int(step_key) for step_key in teacher_step_keys}
            if requires_teacher_logits and not required_support_step_keys.issubset(replay_step_keys):
                return None, "missing_future_support"
        required_step_key_set = {int(step_key) for step_key in train_step_keys}
        required_step_key_set.update(int(step_key) for step_key in teacher_step_keys)
        if accumulate_teacher_blocks and teacher_step_keys:
            for train_step_key, teacher_step_key in zip(train_step_keys, teacher_step_keys):
                lo = min(int(train_step_key), int(teacher_step_key))
                hi = max(int(train_step_key), int(teacher_step_key))
                required_step_key_set.update(
                    int(step_key)
                    for step_key in replay_step_keys
                    if lo <= int(step_key) <= hi
                )

        prompt_ids = self._as_int_list(raw_chunk.get("prompt_ids"))
        response_ids = self._as_int_list(raw_chunk.get("output_ids") or raw_chunk.get("response_ids"))
        prompt_length = int(rollout_cfg.prompt_length)
        response_length = int(rollout_cfg.response_length)
        if len(prompt_ids) > prompt_length:
            return None, "prompt_length_exceeded"
        if len(response_ids) > response_length:
            return None, "response_length_exceeded"
        source = raw_chunk.get("source")
        if source is None:
            source = f"agent_worker_sample_{sample_idx}"
        request_id = raw_chunk.get("request_id")
        chunk_index = raw_chunk.get("chunk_index")
        if chunk_index is None:
            chunk_index = streamed_chunks
        queue_index = raw_chunk.get("queue_index")
        if queue_index is None:
            queue_index = streamed_chunks

        source_metrics = self._sdar_source_replay_metrics(
            replay_payload,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            prompt_length=prompt_length,
            block_length=block_length,
            train_step_keys=train_step_keys,
        )
        clean_token_probs: list[float | None] = [None] * len(response_ids)
        meta_info = raw_chunk.get("meta_info") if isinstance(raw_chunk.get("meta_info"), dict) else {}
        metric_blocks = meta_info.get("sglang_sdar_request_metrics", [])
        if isinstance(metric_blocks, dict):
            metric_blocks = [metric_blocks]
        if isinstance(metric_blocks, list):
            for block_metrics in metric_blocks:
                if not isinstance(block_metrics, dict):
                    continue
                probs = block_metrics.get("clean_token_probs")
                start = block_metrics.get("clean_token_start_index")
                if probs is None or start is None:
                    continue
                try:
                    start_i = int(start)
                except (TypeError, ValueError):
                    continue
                if hasattr(probs, "tolist"):
                    probs = probs.tolist()
                if not isinstance(probs, list):
                    continue
                for offset, prob in enumerate(probs):
                    pos = start_i + int(offset)
                    if 0 <= pos < len(clean_token_probs):
                        try:
                            clean_token_probs[pos] = float(prob)
                        except (TypeError, ValueError):
                            pass

        direct_replay_payload = replay_payload
        delta_teacher_cache_enabled = (
            self._sdar_stream_delta_teacher_cache()
            and accumulate_teacher_blocks
            and bool(stream_cache_key)
            and delta_teacher_cache_sent_step_keys is not None
        )
        delta_step_keys: list[int] = []
        cached_step_keys: list[int] = []
        if delta_teacher_cache_enabled:
            sent_step_keys = {int(step_key) for step_key in (delta_teacher_cache_sent_step_keys or set())}
            keep_step_key_set = {int(step_key) for step_key in train_step_keys}
            keep_step_key_set.update(
                int(step_key)
                for step_key in required_step_key_set
                if int(step_key) not in sent_step_keys
            )
            direct_replay_payload = self._filter_sdar_replay_payload_step_keys(
                replay_payload,
                keep_step_key_set,
            )
            if direct_replay_payload is None:
                return None, "empty_delta_teacher_cache_payload"
            delta_step_keys = sorted(
                int(step_key)
                for step_key in self._sdar_replay_payload_step_keys(direct_replay_payload)
            )
            cached_step_keys = sorted(
                int(step_key)
                for step_key in required_step_key_set
                if int(step_key) not in set(delta_step_keys)
            )
            direct_replay_payload["trace_future_delta_teacher_cache"] = True
            direct_replay_payload["trace_future_cache_key"] = str(stream_cache_key)
            direct_replay_payload["trace_future_required_step_keys"] = torch.tensor(
                sorted(int(step_key) for step_key in required_step_key_set),
                dtype=torch.long,
            )
            direct_replay_payload["trace_future_delta_step_keys"] = torch.tensor(
                delta_step_keys,
                dtype=torch.long,
            )
            direct_replay_payload["trace_future_cached_step_keys"] = torch.tensor(
                cached_step_keys,
                dtype=torch.long,
            )
            direct_replay_payload["train_step_keys"] = torch.tensor(train_step_keys, dtype=torch.long)
            if teacher_step_keys:
                direct_replay_payload["teacher_step_keys"] = torch.tensor(teacher_step_keys, dtype=torch.long)
            direct_replay_payload["trace_future_accumulate_teacher_blocks"] = True

        direct_item = {
            "format": "split_sdar_direct_compact_v1",
            "prompt_ids": prompt_ids,
            "response_ids": response_ids,
            "prompt_length": prompt_length,
            "response_length": response_length,
            "block_length": block_length,
            "future_offset": int(future_offset),
            "sample_index": int(sample_idx),
            "chunk_index": int(chunk_index),
            "queue_index": int(queue_index),
            "source": source,
            "request_id": str(request_id) if request_id is not None else None,
            "response_uid": str(raw_chunk.get("response_uid") or stream_cache_key or request_id or ""),
            "stream_cache_key": str(stream_cache_key) if stream_cache_key else None,
            "train_step_keys": train_step_keys,
            "teacher_step_keys": teacher_step_keys,
            "trace_future_tail_policy": str(
                replay_payload.get("trace_future_tail_policy", "strict_skip")
                if isinstance(replay_payload, dict)
                else "strict_skip"
            ),
            "trace_future_accumulate_teacher_blocks": bool(
                replay_payload.get("trace_future_accumulate_teacher_blocks", False)
                if isinstance(replay_payload, dict)
                else False
            ),
            "source_replay_rows": int(source_metrics.get("source_replay_rows", 0)),
            "source_replay_active_tokens": int(source_metrics.get("source_replay_active_tokens", 0)),
            "source_replay_items": 1,
            "source_replay_chunks": 1,
            "source_replay_step_count": int(source_metrics.get("source_replay_step_count", len(train_step_keys))),
            "source_replay_active_step_count": int(
                source_metrics.get("source_replay_active_step_count", 0)
            ),
            "clean_token_probs": clean_token_probs,
            "sglang_sdar_stream_replay_tensor_store": raw_chunk.get(
                "sglang_sdar_stream_replay_tensor_store"
            ),
            "sglang_sdar_stream_replay_tensor_format": raw_chunk.get(
                "sglang_sdar_stream_replay_tensor_format"
            ),
            "replay": direct_replay_payload,
        }
        replay_trace = raw_chunk.get("rollout_replay_trace", raw_chunk.get("replay_trace"))
        if replay_trace is not None:
            direct_item["rollout_replay_trace"] = replay_trace
        if delta_teacher_cache_enabled:
            direct_item.update(
                {
                    "trace_future_delta_teacher_cache": True,
                    "trace_future_cache_key": str(stream_cache_key),
                    "trace_future_required_step_keys": sorted(
                        int(step_key) for step_key in required_step_key_set
                    ),
                    "trace_future_delta_step_keys": delta_step_keys,
                    "trace_future_cached_step_keys": cached_step_keys,
                }
            )
        for key in (
            "tail_expected_nonfinal_steps",
            "tail_emitted_nonfinal_steps",
            "tail_missing_nonfinal_steps",
            "tail_expected_final_steps",
            "tail_emitted_final_steps",
            "tail_final_step_key",
            "tail_expected_nonfinal_step_keys",
            "tail_emitted_nonfinal_step_keys",
            "tail_missing_nonfinal_step_keys",
            "tail_expected_final_step_keys",
            "tail_emitted_final_step_keys",
            "trace_future_block_level_final_flush",
            "block_level_completed_block_count",
            "block_level_completed_block_indices",
        ):
            if key in replay_payload:
                direct_item[key] = replay_payload[key]
        direct_item["sglang_meta_info"] = meta_info
        return direct_item, "ok"

    async def _consume_sdar_streaming_replay(
        self,
        *,
        sample_idx: int,
        raw_stream_queue: ray.actor.ActorHandle,
        replay_queue: ray.actor.ActorHandle,
        sample_kwargs: dict[str, Any],
        sample_non_tensor: dict[str, np.ndarray],
    ) -> int:
        streamed_chunks = 0
        raw_items_seen = 0
        emitted_step_keys: set[int] = set()
        all_seen_step_keys: set[int] = set()
        all_seen_block_indices: set[int] = set()
        pending_replay_payload = None
        pending_step_push_ts: dict[int, float] = {}
        pending_raw_first_push_ts = 0.0
        sample_first_raw_push_ts = 0.0
        sample_last_raw_push_ts = 0.0
        request_first_raw_push_ts: dict[str, float] = {}
        request_last_raw_push_ts: dict[str, float] = {}
        future_offset = self._sdar_stream_future_logits_offset()
        future_skip_tail_steps = self._sdar_stream_future_skip_tail_steps()
        future_tail_policy = self._sdar_stream_future_tail_policy()
        future_train_all_missing_with_final = self._sdar_stream_future_train_all_missing_with_final()
        future_train_final_step_with_final = self._sdar_stream_future_train_final_step_with_final()
        future_accumulate_teacher_blocks = self._sdar_stream_accumulate_teacher_blocks()
        future_delta_teacher_cache = self._sdar_stream_delta_teacher_cache()
        future_block_level_final_flush = self._sdar_stream_block_level_final_flush()
        stream_cache_key = f"sdar:{uuid4().hex}"
        tail_acc_pushed = False
        delta_teacher_cache_sent_step_keys: set[int] = set()
        future_train_all_missing_log = (
            " trace_future_train_all_missing_with_final=True"
            if future_train_all_missing_with_final
            else ""
        )
        future_train_final_log = (
            " trace_future_train_final_step_with_final=True"
            if future_train_final_step_with_final
            else ""
        )
        future_block_level_final_log = (
            " trace_future_block_level_final_flush=True"
            if future_block_level_final_flush
            else ""
        )
        stream_replay_step_stride = self._sdar_stream_replay_step_stride()
        stream_replay_step_offset = self._sdar_stream_replay_step_offset()
        split_streaming_cfg = self.config.trainer.get("split_gpu_streaming", {})
        try:
            raw_replay_pull_max_items = int(split_streaming_cfg.get("raw_replay_queue_pull_max_items", 8) or 8)
        except (TypeError, ValueError):
            raw_replay_pull_max_items = 8
        raw_replay_pull_max_items = max(1, raw_replay_pull_max_items)
        rollout_cfg = self.config.actor_rollout_ref.rollout
        prompt_length = int(rollout_cfg.prompt_length)
        block_length = int((rollout_cfg.get("engine_kwargs", {}) or {}).get("block_length", 1) or 1)
        block_length = max(block_length, 1)
        last_prompt_ids: list[int] = []
        last_response_ids: list[int] = []
        last_request_id: str | None = None
        last_meta_info: dict = {}
        while True:
            pop_many_start = time.perf_counter()
            items = await raw_stream_queue.pop_many.remote(raw_replay_pull_max_items)
            raw_pop_many_seconds = time.perf_counter() - pop_many_start
            if items is None:
                break
            raw_items_seen += len(items)
            for queue_item in items:
                item_process_start = time.perf_counter()
                raw_pop_ts = time.time()
                raw_push_ts = float(queue_item.get("push_ts", 0.0) or 0.0)
                raw_queue_wait_seconds = (
                    max(0.0, raw_pop_ts - raw_push_ts)
                    if raw_push_ts > 0.0
                    else 0.0
                )
                raw_chunk = dict(queue_item.get("item") or {})
                raw_chunk["queue_index"] = int(queue_item.get("idx", streamed_chunks))
                raw_chunk["source"] = queue_item.get("source")
                request_id = raw_chunk.get("request_id")
                if request_id not in (None, ""):
                    last_request_id = str(request_id)
                stream_request_key = (
                    str(request_id)
                    if request_id not in (None, "")
                    else f"sample:{sample_idx}"
                )
                extract_start = time.perf_counter()
                replay_payload = self._extract_sdar_replay_payload(raw_chunk)
                extract_seconds = time.perf_counter() - extract_start
                table_fetch_seconds = 0.0
                table_replay_handle = None
                table_replay_fetch_ref = None
                if self._is_sdar_replay_table_handle(replay_payload):
                    table_replay_handle = dict(replay_payload)
                    fetch_request_id = str(
                        raw_chunk.get("request_id")
                        or replay_payload.get("source_request_id")
                        or replay_payload.get("request_id")
                    )
                    table_replay_fetch_ref = self.server_manager.fetch_sdar_replay_tensors_ref(
                        fetch_request_id,
                        replay_payload,
                        cleanup=True,
                    )
                    # The metadata_batch_v1 preview carries no readable step keys,
                    # so every consumer-side filter reads it as empty and the chunk
                    # is silently dropped. Await the table
                    # fetch here and consume the canonical per-step item list the
                    # central lane uses; fall back to the preview only on failure.
                    table_fetch_start = time.perf_counter()
                    fetched_payload = None
                    try:
                        fetched_payload = await table_replay_fetch_ref
                    except Exception as exc:
                        print(
                            "[SDAR_STREAM_DIAG] table_fetch_failed "
                            f"request_id={fetch_request_id} exc={exc!r}",
                            flush=True,
                        )
                    table_fetch_seconds = time.perf_counter() - table_fetch_start
                    if fetched_payload is not None:
                        table_replay_fetch_ref = None
                        replay_payload = fetched_payload
                    else:
                        replay_payload = replay_payload.get("metadata_payload")
                        if isinstance(replay_payload, dict):
                            replay_payload = dict(replay_payload)
                            replay_payload["_table_replay_handle"] = table_replay_handle
                            replay_payload["_table_replay_fetch_ref"] = table_replay_fetch_ref
                    raw_chunk["sglang_sdar_replay_tensors_compact"] = replay_payload
                    raw_chunk["sglang_sdar_replay_table_fetch_seconds"] = float(table_fetch_seconds)
                materialize_start = time.perf_counter()
                replay_payload = self._materialize_sdar_replay_payload(replay_payload)
                materialize_seconds = time.perf_counter() - materialize_start
                if raw_push_ts > 0.0:
                    sample_first_raw_push_ts = (
                        min(sample_first_raw_push_ts, raw_push_ts)
                        if sample_first_raw_push_ts > 0.0
                        else raw_push_ts
                    )
                    sample_last_raw_push_ts = max(sample_last_raw_push_ts, raw_push_ts)
                    existing_request_first_ts = float(request_first_raw_push_ts.get(stream_request_key, 0.0) or 0.0)
                    request_first_raw_push_ts[stream_request_key] = (
                        min(existing_request_first_ts, raw_push_ts)
                        if existing_request_first_ts > 0.0
                        else raw_push_ts
                    )
                    request_last_raw_push_ts[stream_request_key] = max(
                        float(request_last_raw_push_ts.get(stream_request_key, 0.0) or 0.0),
                        raw_push_ts,
                )
                raw_step_keys = self._sdar_replay_payload_step_keys(replay_payload)
                print(
                    "[SDAR_STREAM_DIAG] item "
                    f"sample_idx={sample_idx} qidx={raw_chunk.get('queue_index')} "
                    f"payload_type={type(replay_payload).__name__} "
                    f"payload_fmt={replay_payload.get('format') if isinstance(replay_payload, dict) else None} "
                    f"payload_keys={sorted(replay_payload.keys())[:14] if isinstance(replay_payload, dict) else None} "
                    f"raw_step_keys={len(raw_step_keys)} "
                    f"table_handle={table_replay_handle is not None} "
                    f"wire_store={raw_chunk.get('sglang_sdar_stream_replay_tensor_store')} "
                    f"wire_fmt={raw_chunk.get('sglang_sdar_stream_replay_tensor_format')}",
                    flush=True,
                )
                all_seen_step_keys.update(raw_step_keys)
                all_seen_block_indices.update(self._sdar_replay_payload_block_indices(replay_payload))
                last_prompt_ids = self._as_int_list(raw_chunk.get("prompt_ids"))
                last_response_ids = self._as_int_list(raw_chunk.get("output_ids") or raw_chunk.get("response_ids"))
                # Merge, never overwrite: a trailing tensor-only chunk carries no
                # meta_info and would wipe the finish_reason the quality filter
                # needs at close (reject_length raise).
                _chunk_meta_info = raw_chunk.get("meta_info")
                if isinstance(_chunk_meta_info, dict) and _chunk_meta_info:
                    last_meta_info = {**last_meta_info, **_chunk_meta_info}
                if raw_push_ts > 0.0:
                    for step_key in raw_step_keys:
                        existing_push_ts = float(pending_step_push_ts.get(step_key, 0.0) or 0.0)
                        pending_step_push_ts[step_key] = (
                            min(existing_push_ts, raw_push_ts)
                            if existing_push_ts > 0.0
                            else raw_push_ts
                        )
                merge_filter_start = time.perf_counter()
                pending_replay_payload = self._merge_sdar_replay_payloads(pending_replay_payload, replay_payload)
                pending_raw_first_push_ts, _ = self._sdar_payload_raw_push_ts_bounds(
                    pending_replay_payload,
                    pending_step_push_ts,
                    fallback_first_ts=pending_raw_first_push_ts or raw_push_ts,
                    fallback_last_ts=raw_push_ts,
                )
                if future_block_level_final_flush:
                    replay_payload = self._filter_sdar_replay_payload_completed_block_steps(
                        pending_replay_payload,
                        emitted_step_keys,
                        response_ids=last_response_ids,
                        prompt_ids=last_prompt_ids,
                        prompt_length=prompt_length,
                        block_length=block_length,
                        step_stride=stream_replay_step_stride,
                        step_offset=stream_replay_step_offset,
                        seen_block_indices=all_seen_block_indices,
                        allow_partial_response_block=False,
                        train_all_missing_with_final=future_train_all_missing_with_final,
                        train_final_step_with_final=future_train_final_step_with_final,
                        accumulate_teacher_blocks=future_accumulate_teacher_blocks,
                    )
                else:
                    replay_payload = self._filter_sdar_replay_payload_new_steps(
                        pending_replay_payload,
                        emitted_step_keys,
                        future_offset=future_offset,
                        block_length=block_length,
                        step_stride=stream_replay_step_stride,
                        step_offset=stream_replay_step_offset,
                        train_all_missing_with_final=future_train_all_missing_with_final,
                        accumulate_teacher_blocks=future_accumulate_teacher_blocks,
                    )
                payload_first_push_ts, payload_last_push_ts = self._sdar_payload_raw_push_ts_bounds(
                    replay_payload,
                    pending_step_push_ts,
                    fallback_first_ts=pending_raw_first_push_ts or raw_push_ts,
                    fallback_last_ts=raw_push_ts,
                )
                request_first_ts = float(request_first_raw_push_ts.get(stream_request_key, 0.0) or 0.0)
                request_last_ts = float(request_last_raw_push_ts.get(stream_request_key, 0.0) or 0.0)
                stream_first_raw_push_ts = request_first_ts or sample_first_raw_push_ts
                stream_last_raw_push_ts = request_last_ts or sample_last_raw_push_ts
                raw_first_push_ts = (
                    stream_first_raw_push_ts
                    or pending_raw_first_push_ts
                    or payload_first_push_ts
                    or raw_push_ts
                )
                raw_last_push_ts = max(payload_last_push_ts, stream_last_raw_push_ts, raw_push_ts)
                pending_replay_payload = self._prune_sdar_replay_payload_emitted_steps(
                    pending_replay_payload,
                    emitted_step_keys,
                )
                pending_keys = self._sdar_replay_payload_step_keys(pending_replay_payload)
                pending_step_push_ts = {
                    step_key: push_ts
                    for step_key, push_ts in pending_step_push_ts.items()
                    if step_key in pending_keys
                }
                pending_raw_first_push_ts, _ = self._sdar_payload_raw_push_ts_bounds(
                    pending_replay_payload,
                    pending_step_push_ts,
                )
                merge_filter_seconds = time.perf_counter() - merge_filter_start
                if replay_payload is None:
                    print(
                        "[SDAR_STREAM_DIAG] drop_at_merge_filter "
                        f"sample_idx={sample_idx} qidx={raw_chunk.get('queue_index')} "
                        f"pending_fmt={pending_replay_payload.get('format') if isinstance(pending_replay_payload, dict) else type(pending_replay_payload).__name__} "
                        f"pending_keys={len(pending_keys)} raw_step_keys={len(raw_step_keys)} "
                        f"all_seen={len(all_seen_step_keys)} emitted={len(emitted_step_keys)} "
                        f"block_flush={future_block_level_final_flush} "
                        f"stride={stream_replay_step_stride} offset={stream_replay_step_offset} "
                        f"future_offset={future_offset}",
                        flush=True,
                    )
                    continue
                timing_metadata = self._sdar_stream_timing_metadata(
                    raw_first_push_ts=raw_first_push_ts,
                    raw_last_push_ts=raw_last_push_ts,
                    raw_queue_wait_seconds=raw_queue_wait_seconds,
                    now_ts=time.time(),
                    sample_first_raw_push_ts=sample_first_raw_push_ts,
                    request_first_raw_push_ts=request_first_ts,
                )
                direct_item = None
                direct_fallback_reason = "direct_compact_disabled"
                direct_build_seconds = 0.0
                if self._direct_compact_enabled():
                    direct_build_start = time.perf_counter()
                    direct_item, direct_fallback_reason = self._build_sdar_direct_compact_replay_item(
                        raw_chunk,
                        replay_payload,
                        sample_idx=sample_idx,
                        streamed_chunks=streamed_chunks,
                        future_offset=future_offset,
                        stream_cache_key=stream_cache_key,
                        delta_teacher_cache_sent_step_keys=(
                            delta_teacher_cache_sent_step_keys if future_delta_teacher_cache else None
                        ),
                    )
                    direct_build_seconds = time.perf_counter() - direct_build_start
                if direct_item is not None:
                    timing_metadata = self._sdar_stream_timing_metadata(
                        raw_first_push_ts=raw_first_push_ts,
                        raw_last_push_ts=raw_last_push_ts,
                        raw_queue_wait_seconds=raw_queue_wait_seconds,
                        now_ts=time.time(),
                        sample_first_raw_push_ts=sample_first_raw_push_ts,
                        request_first_raw_push_ts=request_first_ts,
                    )
                    direct_item.update(timing_metadata)
                    direct_item.update(
                        {
                            "stream_consumer_raw_pop_many_seconds": float(raw_pop_many_seconds),
                            "stream_consumer_raw_queue_wait_seconds": float(raw_queue_wait_seconds),
                            "stream_consumer_extract_seconds": float(extract_seconds),
                            "stream_consumer_table_fetch_seconds": float(table_fetch_seconds),
                            "stream_consumer_materialize_seconds": float(materialize_seconds),
                            "stream_consumer_merge_filter_seconds": float(merge_filter_seconds),
                            "stream_consumer_direct_build_seconds": float(direct_build_seconds),
                            "stream_consumer_item_process_pre_push_seconds": float(
                                time.perf_counter() - item_process_start
                            ),
                            "sglang_stream_replay_compact_seconds": float(
                                raw_chunk.get("sglang_stream_replay_compact_seconds", 0.0) or 0.0
                            ),
                            "sglang_stream_replay_shm_seconds": float(
                                raw_chunk.get("sglang_stream_replay_shm_seconds", 0.0) or 0.0
                            ),
                            "sglang_stream_pre_queue_seconds": float(
                                raw_chunk.get("sglang_stream_pre_queue_seconds", 0.0) or 0.0
                            ),
                            "sglang_stream_replay_items": int(
                                raw_chunk.get("sglang_stream_replay_items", 0) or 0
                            ),
                            "sglang_stream_chunk_output_tokens": int(
                                raw_chunk.get("sglang_stream_chunk_output_tokens", 0) or 0
                            ),
                        }
                    )
                    direct_replay = direct_item.get("replay")
                    direct_table_handle = table_replay_handle
                    direct_table_fetch_ref = table_replay_fetch_ref
                    if isinstance(direct_replay, dict):
                        if direct_table_handle is None:
                            direct_table_handle = direct_replay.get("_table_replay_handle")
                        if direct_table_fetch_ref is None:
                            direct_table_fetch_ref = direct_replay.get("_table_replay_fetch_ref")
                    if direct_table_handle is not None:
                        table_direct_replay = dict(direct_table_handle)
                        table_metadata_payload = dict(direct_replay) if isinstance(direct_replay, dict) else direct_replay
                        if isinstance(table_metadata_payload, dict):
                            table_metadata_payload.pop("_table_replay_handle", None)
                            table_metadata_payload.pop("_table_replay_fetch_ref", None)
                        table_direct_replay["metadata_payload"] = table_metadata_payload
                        table_direct_replay["fetch_ref"] = direct_table_fetch_ref
                        direct_item["replay"] = table_direct_replay
                        direct_replay = table_direct_replay
                    direct_replay_format = (
                        str(direct_replay.get("format"))
                        if isinstance(direct_replay, dict) and direct_replay.get("format") is not None
                        else "sglang_sdar_replay_tensor_batch_v1"
                    )
                    direct_item.update(
                        {
                            "direct_compact_replay_tensor_store": (
                                "table" if self._is_sdar_replay_table_handle(direct_replay) else "tensor"
                            ),
                            "direct_compact_replay_tensor_format": direct_replay_format,
                            "direct_compact_replay_shm_bytes": 0,
                            "direct_compact_replay_shm_create_seconds": 0.0,
                        }
                    )
                    if (
                        self._direct_compact_replay_tensor_store() == "shm"
                        and not self._is_sdar_replay_table_handle(direct_replay)
                    ):
                        shm_create_start = time.perf_counter()
                        direct_replay, shm_bytes = self._sdar_replay_payload_to_shm_handles(direct_replay)
                        shm_create_seconds = time.perf_counter() - shm_create_start
                        direct_item["replay"] = direct_replay
                        direct_item.update(
                            {
                                "direct_compact_replay_tensor_store": "shm",
                                "direct_compact_replay_tensor_format": "sglang_sdar_replay_shm_batch_v1",
                                "direct_compact_replay_shm_bytes": int(shm_bytes),
                                "direct_compact_replay_shm_create_seconds": float(shm_create_seconds),
                            }
                        )
                    direct_item["stream_consumer_trainer_queue_push_start_wall_ts"] = float(time.time())
                    try:
                        await replay_queue.push.remote(
                            direct_item,
                            source=f"agent_worker_sample_{sample_idx}_sdar_chunk_{streamed_chunks}",
                        )
                    except Exception:
                        self._unlink_sdar_shm_payload(direct_item.get("replay"))
                        raise
                    if future_delta_teacher_cache:
                        delta_teacher_cache_sent_step_keys.update(
                            int(step_key)
                            for step_key in direct_item.get("trace_future_delta_step_keys", []) or []
                        )
                    streamed_chunks += 1
                    continue
                fallback_train_step_keys: list[int] = []
                if isinstance(replay_payload, dict):
                    fallback_train_step_keys = self._as_int_list(replay_payload.get("train_step_keys"))
                if not fallback_train_step_keys:
                    fallback_train_step_keys = sorted(self._sdar_replay_payload_step_keys(replay_payload))
                fallback_source_metrics = self._sdar_source_replay_metrics(
                    replay_payload,
                    prompt_ids=last_prompt_ids,
                    response_ids=last_response_ids,
                    prompt_length=prompt_length,
                    block_length=block_length,
                    train_step_keys=fallback_train_step_keys,
                )
                print(
                    "[SplitStreaming] direct_compact_fallback "
                    f"sample_idx={sample_idx} chunk={streamed_chunks} reason={direct_fallback_reason} "
                    f"rows={fallback_source_metrics.get('source_replay_rows', 0)} "
                    f"active_tokens={fallback_source_metrics.get('source_replay_active_tokens', 0)} "
                    f"step_count={fallback_source_metrics.get('source_replay_step_count', 0)}",
                    flush=True,
                )
                for payload_key in (
                    "sglang_sdar_replay_tensors",
                    "replay_tensors",
                    "replay_tensors_compact",
                    "compact_replay_tensors",
                ):
                    raw_chunk.pop(payload_key, None)
                raw_chunk["sglang_sdar_replay_tensors_compact"] = replay_payload
                raw_chunk.update(timing_metadata)
                raw_chunk.update(
                    {
                        "direct_compact_fallback_reason": direct_fallback_reason,
                        "source_replay_rows": int(fallback_source_metrics.get("source_replay_rows", 0)),
                        "source_replay_active_tokens": int(
                            fallback_source_metrics.get("source_replay_active_tokens", 0)
                        ),
                        "source_replay_items": 1,
                        "source_replay_chunks": 1,
                        "source_replay_step_count": int(
                            fallback_source_metrics.get("source_replay_step_count", 0)
                        ),
                        "source_replay_active_step_count": int(
                            fallback_source_metrics.get("source_replay_active_step_count", 0)
                        ),
                        "source_direct_compact_fallback_items": 1,
                        "source_direct_compact_fallback_rows": int(
                            fallback_source_metrics.get("source_replay_rows", 0)
                        ),
                        "source_direct_compact_fallback_active_tokens": int(
                            fallback_source_metrics.get("source_replay_active_tokens", 0)
                        ),
                    }
                )
                chunk_output = self._build_sdar_streaming_output(raw_chunk)
                chunk_internal = await self._agent_loop_postprocess(
                    chunk_output,
                    _skip_compute_score=True,
                    **sample_kwargs,
                )
                chunk_data = self._postprocess([chunk_internal], input_non_tensor_batch=sample_non_tensor)
                self._trim_streamed_sdar_training_state_(chunk_data, future_offset)
                timing_metadata = self._sdar_stream_timing_metadata(
                    raw_first_push_ts=raw_first_push_ts,
                    raw_last_push_ts=raw_last_push_ts,
                    raw_queue_wait_seconds=raw_queue_wait_seconds,
                    now_ts=time.time(),
                    sample_first_raw_push_ts=sample_first_raw_push_ts,
                    request_first_raw_push_ts=request_first_ts,
                )
                self._update_sdar_stream_chunk_timing_(chunk_data, timing_metadata)
                await replay_queue.push.remote(
                    chunk_data,
                    source=f"agent_worker_sample_{sample_idx}_sdar_chunk_{streamed_chunks}",
                )
                streamed_chunks += 1
        if pending_replay_payload is not None:
            if future_block_level_final_flush:
                tail_replay_payload = self._filter_sdar_replay_payload_completed_block_steps(
                    pending_replay_payload,
                    emitted_step_keys,
                    response_ids=last_response_ids,
                    prompt_ids=last_prompt_ids,
                    prompt_length=prompt_length,
                    block_length=block_length,
                    step_stride=stream_replay_step_stride,
                    step_offset=stream_replay_step_offset,
                    seen_block_indices=all_seen_block_indices,
                    allow_partial_response_block=True,
                    train_all_missing_with_final=future_train_all_missing_with_final,
                    train_final_step_with_final=future_train_final_step_with_final,
                    accumulate_teacher_blocks=future_accumulate_teacher_blocks,
                )
            else:
                tail_replay_payload = self._filter_sdar_replay_payload_tail_steps(
                    pending_replay_payload,
                    emitted_step_keys,
                    future_offset=future_offset,
                    block_length=block_length,
                    step_stride=stream_replay_step_stride,
                    step_offset=stream_replay_step_offset,
                    tail_policy=future_tail_policy,
                    train_all_missing_with_final=future_train_all_missing_with_final,
                    train_final_step_with_final=future_train_final_step_with_final,
                    accumulate_teacher_blocks=future_accumulate_teacher_blocks,
                )
            if tail_replay_payload is None:
                print(
                    "[SDAR_STREAM_DIAG] tail_none "
                    f"sample_idx={sample_idx} "
                    f"pending_fmt={pending_replay_payload.get('format') if isinstance(pending_replay_payload, dict) else type(pending_replay_payload).__name__} "
                    f"pending_keys={len(self._sdar_replay_payload_step_keys(pending_replay_payload))} "
                    f"emitted={len(emitted_step_keys)} future_offset={future_offset} "
                    f"tail_policy={future_tail_policy} block_flush={future_block_level_final_flush}",
                    flush=True,
                )
            if tail_replay_payload is not None:
                tail_raw_chunk = {
                    "prompt_ids": last_prompt_ids,
                    "output_ids": last_response_ids,
                    "meta_info": last_meta_info,
                    "request_id": last_request_id or f"sample:{sample_idx}:tail",
                    "chunk_index": streamed_chunks,
                    "queue_index": streamed_chunks,
                    "source": f"agent_worker_sample_{sample_idx}_sdar_tail",
                }
                direct_item = None
                direct_fallback_reason = "direct_compact_disabled"
                if self._direct_compact_enabled():
                    direct_item, direct_fallback_reason = self._build_sdar_direct_compact_replay_item(
                        tail_raw_chunk,
                        tail_replay_payload,
                        sample_idx=sample_idx,
                        streamed_chunks=streamed_chunks,
                        future_offset=future_offset,
                        stream_cache_key=stream_cache_key,
                        delta_teacher_cache_sent_step_keys=(
                            delta_teacher_cache_sent_step_keys if future_delta_teacher_cache else None
                        ),
                    )
                if direct_item is not None:
                    payload_first_push_ts, payload_last_push_ts = self._sdar_payload_raw_push_ts_bounds(
                        tail_replay_payload,
                        pending_step_push_ts,
                        fallback_first_ts=pending_raw_first_push_ts or sample_first_raw_push_ts,
                        fallback_last_ts=sample_last_raw_push_ts,
                    )
                    timing_metadata = self._sdar_stream_timing_metadata(
                        raw_first_push_ts=payload_first_push_ts or sample_first_raw_push_ts,
                        raw_last_push_ts=max(payload_last_push_ts, sample_last_raw_push_ts),
                        raw_queue_wait_seconds=0.0,
                        now_ts=time.time(),
                        sample_first_raw_push_ts=sample_first_raw_push_ts,
                        request_first_raw_push_ts=sample_first_raw_push_ts,
                    )
                    direct_item.update(timing_metadata)
                    direct_item["trace_future_tail_policy"] = future_tail_policy
                    if future_train_all_missing_with_final:
                        direct_item["trace_future_train_all_missing_with_final"] = True
                    if future_train_final_step_with_final:
                        direct_item["trace_future_train_final_step_with_final"] = True
                    direct_replay = direct_item.get("replay")
                    direct_table_handle = None
                    direct_table_fetch_ref = None
                    if isinstance(direct_replay, dict):
                        direct_table_handle = direct_replay.get("_table_replay_handle")
                        direct_table_fetch_ref = direct_replay.get("_table_replay_fetch_ref")
                    if direct_table_handle is not None:
                        table_direct_replay = dict(direct_table_handle)
                        table_metadata_payload = dict(direct_replay) if isinstance(direct_replay, dict) else direct_replay
                        if isinstance(table_metadata_payload, dict):
                            table_metadata_payload.pop("_table_replay_handle", None)
                            table_metadata_payload.pop("_table_replay_fetch_ref", None)
                        table_direct_replay["metadata_payload"] = table_metadata_payload
                        table_direct_replay["fetch_ref"] = direct_table_fetch_ref
                        direct_item["replay"] = table_direct_replay
                        direct_replay = table_direct_replay
                    direct_replay_format = (
                        str(direct_replay.get("format"))
                        if isinstance(direct_replay, dict) and direct_replay.get("format") is not None
                        else "sglang_sdar_replay_tensor_batch_v1"
                    )
                    direct_item.update(
                        {
                            "direct_compact_replay_tensor_store": (
                                "table" if self._is_sdar_replay_table_handle(direct_replay) else "tensor"
                            ),
                            "direct_compact_replay_tensor_format": direct_replay_format,
                            "direct_compact_replay_shm_bytes": 0,
                            "direct_compact_replay_shm_create_seconds": 0.0,
                        }
                    )
                    if (
                        self._direct_compact_replay_tensor_store() == "shm"
                        and not self._is_sdar_replay_table_handle(direct_replay)
                    ):
                        shm_create_start = time.perf_counter()
                        direct_replay, shm_bytes = self._sdar_replay_payload_to_shm_handles(direct_replay)
                        shm_create_seconds = time.perf_counter() - shm_create_start
                        direct_item["replay"] = direct_replay
                        direct_item.update(
                            {
                                "direct_compact_replay_tensor_store": "shm",
                                "direct_compact_replay_tensor_format": "sglang_sdar_replay_shm_batch_v1",
                                "direct_compact_replay_shm_bytes": int(shm_bytes),
                                "direct_compact_replay_shm_create_seconds": float(shm_create_seconds),
                            }
                        )
                    _acc_out = self._build_sdar_streaming_output(tail_raw_chunk)
                    _acc_internal = await self._agent_loop_postprocess(_acc_out, **sample_kwargs)
                    if _acc_internal.reward_score is not None:
                        direct_item["acc"] = float(_acc_internal.reward_score)
                    try:
                        await replay_queue.push.remote(
                            direct_item,
                            source=f"agent_worker_sample_{sample_idx}_sdar_tail_{streamed_chunks}",
                        )
                    except Exception:
                        self._unlink_sdar_shm_payload(direct_item.get("replay"))
                        raise
                    tail_acc_pushed = "acc" in direct_item
                    if future_delta_teacher_cache:
                        delta_teacher_cache_sent_step_keys.update(
                            int(step_key)
                            for step_key in direct_item.get("trace_future_delta_step_keys", []) or []
                        )
                    print(
                        "[SplitStreaming] sdar_stream_tail_replay_flush "
                        f"sample_idx={sample_idx} chunk={streamed_chunks} "
                        f"train_step_keys={direct_item.get('train_step_keys', [])} "
                        f"teacher_step_keys={direct_item.get('teacher_step_keys', [])} "
                        f"tail_expected_nonfinal_steps={direct_item.get('tail_expected_nonfinal_steps', 0)} "
                        f"tail_emitted_nonfinal_steps={direct_item.get('tail_emitted_nonfinal_steps', 0)} "
                        f"tail_missing_nonfinal_steps={direct_item.get('tail_missing_nonfinal_steps', 0)} "
                        f"tail_expected_final_steps={direct_item.get('tail_expected_final_steps', 0)} "
                        f"tail_emitted_final_steps={direct_item.get('tail_emitted_final_steps', 0)} "
                        f"tail_policy={future_tail_policy}{future_train_all_missing_log}{future_train_final_log}{future_block_level_final_log}",
                        flush=True,
                    )
                    streamed_chunks += 1
                    emitted_step_keys.update(
                        int(step_key)
                        for step_key in direct_item.get("train_step_keys", []) or []
                    )
                    pending_replay_payload = self._prune_sdar_replay_payload_emitted_steps(
                        pending_replay_payload,
                        emitted_step_keys,
                    )
                    pending_keys = self._sdar_replay_payload_step_keys(pending_replay_payload)
                    pending_step_push_ts = {
                        step_key: push_ts
                        for step_key, push_ts in pending_step_push_ts.items()
                        if step_key in pending_keys
                    }
                    pending_raw_first_push_ts, _ = self._sdar_payload_raw_push_ts_bounds(
                        pending_replay_payload,
                        pending_step_push_ts,
                    )
                else:
                    fallback_train_step_keys: list[int] = []
                    if isinstance(tail_replay_payload, dict):
                        fallback_train_step_keys = self._as_int_list(
                            tail_replay_payload.get("train_step_keys")
                        )
                    if not fallback_train_step_keys:
                        fallback_train_step_keys = sorted(
                            self._sdar_replay_payload_step_keys(tail_replay_payload)
                        )
                    fallback_source_metrics = self._sdar_source_replay_metrics(
                        tail_replay_payload,
                        prompt_ids=last_prompt_ids,
                        response_ids=last_response_ids,
                        prompt_length=prompt_length,
                        block_length=block_length,
                        train_step_keys=fallback_train_step_keys,
                    )
                    print(
                        "[SplitStreaming] sdar_stream_tail_replay_flush_skipped "
                        f"sample_idx={sample_idx} reason={direct_fallback_reason} "
                        f"tail_expected_nonfinal_steps={tail_replay_payload.get('tail_expected_nonfinal_steps', 0)} "
                        f"tail_emitted_nonfinal_steps={tail_replay_payload.get('tail_emitted_nonfinal_steps', 0)} "
                        f"tail_missing_nonfinal_steps={tail_replay_payload.get('tail_missing_nonfinal_steps', 0)} "
                        f"tail_expected_final_steps={tail_replay_payload.get('tail_expected_final_steps', 0)} "
                        f"tail_emitted_final_steps={tail_replay_payload.get('tail_emitted_final_steps', 0)} "
                        f"tail_policy={future_tail_policy}{future_train_all_missing_log}{future_train_final_log}{future_block_level_final_log}",
                        flush=True,
                    )
                    payload_first_push_ts, payload_last_push_ts = self._sdar_payload_raw_push_ts_bounds(
                        tail_replay_payload,
                        pending_step_push_ts,
                        fallback_first_ts=pending_raw_first_push_ts or sample_first_raw_push_ts,
                        fallback_last_ts=sample_last_raw_push_ts,
                    )
                    timing_metadata = self._sdar_stream_timing_metadata(
                        raw_first_push_ts=payload_first_push_ts or sample_first_raw_push_ts,
                        raw_last_push_ts=max(payload_last_push_ts, sample_last_raw_push_ts),
                        raw_queue_wait_seconds=0.0,
                        now_ts=time.time(),
                        sample_first_raw_push_ts=sample_first_raw_push_ts,
                        request_first_raw_push_ts=sample_first_raw_push_ts,
                    )
                    for payload_key in (
                        "sglang_sdar_replay_tensors",
                        "replay_tensors",
                        "replay_tensors_compact",
                        "compact_replay_tensors",
                    ):
                        tail_raw_chunk.pop(payload_key, None)
                    tail_raw_chunk["sglang_sdar_replay_tensors_compact"] = tail_replay_payload
                    tail_raw_chunk.update(timing_metadata)
                    tail_raw_chunk.update(
                        {
                            "direct_compact_fallback_reason": direct_fallback_reason,
                            "source_replay_rows": int(fallback_source_metrics.get("source_replay_rows", 0)),
                            "source_replay_active_tokens": int(
                                fallback_source_metrics.get("source_replay_active_tokens", 0)
                            ),
                            "source_replay_items": 1,
                            "source_replay_chunks": 1,
                            "source_replay_step_count": int(
                                fallback_source_metrics.get("source_replay_step_count", 0)
                            ),
                            "source_replay_active_step_count": int(
                                fallback_source_metrics.get("source_replay_active_step_count", 0)
                            ),
                            "source_direct_compact_fallback_items": 1,
                            "source_direct_compact_fallback_rows": int(
                                fallback_source_metrics.get("source_replay_rows", 0)
                            ),
                            "source_direct_compact_fallback_active_tokens": int(
                                fallback_source_metrics.get("source_replay_active_tokens", 0)
                            ),
                        }
                    )
                    chunk_output = self._build_sdar_streaming_output(tail_raw_chunk)
                    chunk_internal = await self._agent_loop_postprocess(
                        chunk_output,
                        _skip_compute_score=True,
                        **sample_kwargs,
                    )
                    chunk_data = self._postprocess([chunk_internal], input_non_tensor_batch=sample_non_tensor)
                    self._trim_streamed_sdar_training_state_(chunk_data, future_offset)
                    self._update_sdar_stream_chunk_timing_(chunk_data, timing_metadata)
                    await replay_queue.push.remote(
                        chunk_data,
                        source=f"agent_worker_sample_{sample_idx}_sdar_tail_{streamed_chunks}",
                    )
                    streamed_chunks += 1
                    emitted_step_keys.update(int(step_key) for step_key in fallback_train_step_keys)
                    pending_replay_payload = self._prune_sdar_replay_payload_emitted_steps(
                        pending_replay_payload,
                        emitted_step_keys,
                    )
                    pending_keys = self._sdar_replay_payload_step_keys(pending_replay_payload)
                    pending_step_push_ts = {
                        step_key: push_ts
                        for step_key, push_ts in pending_step_push_ts.items()
                        if step_key in pending_keys
                    }
                    pending_raw_first_push_ts, _ = self._sdar_payload_raw_push_ts_bounds(
                        pending_replay_payload,
                        pending_step_push_ts,
                    )

        if streamed_chunks > 0 and not tail_acc_pushed:
            # L2b hold-until-tail: mid-stream items are held by stream_cache_key
            # until an acc-carrying item arrives, but strict_skip/future_offset<=0
            # always yields tail None, so no acc item ever comes and the consumer
            # raises orphaned-hold at close. Push a row-less closer that carries
            # acc + meta so the held group is stamped and released.
            closer_out = self._build_sdar_streaming_output(
                {
                    "prompt_ids": last_prompt_ids,
                    "output_ids": last_response_ids,
                    "meta_info": last_meta_info,
                    "request_id": last_request_id or f"sample:{sample_idx}:closer",
                }
            )
            closer_internal = await self._agent_loop_postprocess(closer_out, **sample_kwargs)
            closer_item = {
                "format": "split_sdar_direct_compact_v1",
                "closer_only": True,
                "stream_cache_key": str(stream_cache_key),
                "response_uid": str(stream_cache_key),
                "sample_index": int(sample_idx),
                "sglang_meta_info": last_meta_info,
            }
            if closer_internal.reward_score is not None:
                closer_item["acc"] = float(closer_internal.reward_score)
            await replay_queue.push.remote(
                closer_item,
                source=f"agent_worker_sample_{sample_idx}_sdar_tail_closer",
            )
            print(
                "[SplitStreaming] sdar_stream_tail_closer "
                f"sample_idx={sample_idx} acc={closer_item.get('acc')} "
                f"finish={last_meta_info.get('finish_reason') if isinstance(last_meta_info, dict) else None}",
                flush=True,
            )

        if pending_replay_payload is not None:
            pending_metrics = self._sdar_pending_replay_metrics(
                pending_replay_payload=pending_replay_payload,
                all_seen_step_keys=all_seen_step_keys,
                emitted_step_keys=emitted_step_keys,
                future_offset=future_offset,
                future_skip_tail_steps=future_skip_tail_steps,
                step_stride=stream_replay_step_stride,
                step_offset=stream_replay_step_offset,
                sample_idx=sample_idx,
                streamed_chunks=streamed_chunks,
                prompt_ids=last_prompt_ids,
                response_ids=last_response_ids,
                prompt_length=prompt_length,
                block_length=block_length,
                tail_policy=future_tail_policy,
                train_all_missing_with_final=future_train_all_missing_with_final,
                train_final_step_with_final=future_train_final_step_with_final,
            )
            if future_train_all_missing_with_final:
                pending_metrics["trace_future_train_all_missing_with_final"] = True
            if future_train_final_step_with_final:
                pending_metrics["trace_future_train_final_step_with_final"] = True
            pending_steps = int(pending_metrics.get("source_pending_replay_steps", 0) or 0)
            tail_missing_nonfinal_steps = int(pending_metrics.get("tail_missing_nonfinal_steps", 0) or 0)
            print(
                "[SDAR_STREAM_DIAG] close "
                f"sample_idx={sample_idx} streamed_chunks={streamed_chunks} "
                f"pending_steps={pending_steps} all_seen={len(all_seen_step_keys)} "
                f"emitted={len(emitted_step_keys)}",
                flush=True,
            )
            if pending_steps > 0:
                print(
                    "[SplitStreaming] sdar_stream_pending_replay_close "
                    f"sample_idx={sample_idx} pending_steps={pending_steps} "
                    f"pending_rows={pending_metrics.get('source_pending_replay_rows', 0)} "
                    f"ready_steps={pending_metrics.get('source_pending_ready_steps', 0)} "
                    f"ready_rows={pending_metrics.get('source_pending_ready_rows', 0)} "
                    f"unsupported_steps={pending_metrics.get('source_pending_unsupported_steps', 0)} "
                    f"unsupported_rows={pending_metrics.get('source_pending_unsupported_rows', 0)} "
                    f"unexpected_steps={pending_metrics.get('source_pending_unexpected_steps', 0)} "
                    f"tail_expected_nonfinal_steps={pending_metrics.get('tail_expected_nonfinal_steps', 0)} "
                    f"tail_emitted_nonfinal_steps={pending_metrics.get('tail_emitted_nonfinal_steps', 0)} "
                    f"tail_missing_nonfinal_steps={tail_missing_nonfinal_steps} "
                    f"future_offset={future_offset} future_skip_tail_steps={future_skip_tail_steps} "
                    f"tail_expected_final_steps={pending_metrics.get('tail_expected_final_steps', 0)} "
                    f"tail_emitted_final_steps={pending_metrics.get('tail_emitted_final_steps', 0)} "
                    f"tail_policy={future_tail_policy}{future_train_all_missing_log}{future_train_final_log}{future_block_level_final_log} "
                    f"stream_step_stride={stream_replay_step_stride} "
                    f"stream_step_offset={stream_replay_step_offset} "
                    f"pending_step_keys={pending_metrics.get('pending_step_keys', [])}",
                    flush=True,
                )
                await replay_queue.push.remote(
                    pending_metrics,
                    source=f"agent_worker_sample_{sample_idx}_sdar_pending_summary",
                )
                if tail_missing_nonfinal_steps > 0:
                    raise RuntimeError(
                        "SGLang streaming SDAR trace_future tail coverage invariant failed at close: "
                        f"sample_idx={sample_idx} "
                        f"tail_expected_nonfinal_step_keys="
                        f"{pending_metrics.get('tail_expected_nonfinal_step_keys', [])} "
                        f"tail_missing_nonfinal_step_keys="
                        f"{pending_metrics.get('tail_missing_nonfinal_step_keys', [])} "
                        f"pending_step_keys={pending_metrics.get('pending_step_keys', [])} "
                        f"tail_policy={future_tail_policy}{future_train_all_missing_log}{future_train_final_log}{future_block_level_final_log}"
                    )
                if int(pending_metrics.get("source_pending_ready_rows", 0) or 0) > 0:
                    raise RuntimeError(
                        "SGLang streaming SDAR replay closed with ready pending replay rows: "
                        f"sample_idx={sample_idx} pending_step_keys={pending_metrics.get('pending_step_keys', [])} "
                        f"ready_step_keys={pending_metrics.get('ready_step_keys', [])} "
                        f"ready_active_step_keys={pending_metrics.get('ready_active_step_keys', [])} "
                        f"ready_rows={pending_metrics.get('source_pending_ready_rows', 0)} "
                        f"ready_active_tokens={pending_metrics.get('source_pending_ready_active_tokens', 0)} "
                        f"future_offset={future_offset} future_skip_tail_steps={future_skip_tail_steps}"
                    )
        print(
            f"[SDAR_STREAM_DIAG] stream_consumer_exit sample_idx={sample_idx} "
            f"streamed_chunks={streamed_chunks} raw_items_seen={raw_items_seen}",
            flush=True,
        )
        return streamed_chunks

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        trace: bool = True,
        **kwargs,
    ) -> _InternalAgentLoopOutput:
        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
            trace=trace,
        ):
            assert agent_name in _agent_loop_registry, (
                f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
            )

            agent_loop_config = _agent_loop_registry[agent_name]
            agent_loop = hydra.utils.instantiate(
                config=agent_loop_config,
                trainer_config=DictConfigWrap(config=self.config),
                server_manager=self.server_manager,
                tokenizer=self.tokenizer,
                processor=self.processor,
                dataset_cls=self.dataset_cls,
                dataset_config=DictConfigWrap(self.config.data),
            )
            output: AgentLoopOutput = await agent_loop.run(sampling_params, **kwargs)
            postprocess_start = time.perf_counter()
            processed = await self._agent_loop_postprocess(output, **self._strip_internal_agent_kwargs(kwargs))
            output.metrics.postprocess = time.perf_counter() - postprocess_start
            processed.metrics = output.metrics
            return processed

    async def _agent_loop_postprocess(self, output, **kwargs) -> _InternalAgentLoopOutput:
        """Perform post-processing operations on the output of each individual agent loop."""
        skip_compute_score = bool(kwargs.get("_skip_compute_score", False))
        score_kwargs = self._strip_internal_agent_kwargs(kwargs)
        output.extra_fields["raw_prompt"] = kwargs["raw_prompt"]

        # Some AgentLoop may have already computed the reward score, e.g SWE-agent.

        # NOTE: consistent with the legacy batch version of generate_sequences that existed in the
        # deprecated vLLM SPMD rollout implementation.
        # prompt_ids: left padded with zeros (e.g., [0,0,0,0,1,2,3,4])
        # response_ids: right padded with zeros (e.g., [5,6,7,8,0,0,0,0])
        # input_ids: concatenation of prompt + response
        # Mask:
        # For example, if the prompt is [1,2,3,4] and the response is [5,6,7,(tool start)8,9(tool end),10,11,12]
        # - prompt_attention_mask: 0s for padding, 1s for tokens
        #   e.g., [0,0,0,0,1,1,1,1]
        # - response_attention_mask: 0s for padding, 1s for tokens
        #   e.g., [1,1,1,1,1,1,1,1,1,1,1,0,0,0,0]
        # attention_mask: concatenation of prompt_attention_mask and response_attention_mask
        #   e.g., [0,0,0,0,1,1,1,1(prompt),1,1,1,1,1,1,1,1,1,1,1,0,0,0,0(response)]
        # - response_mask: 1s for LLM generated tokens, 0 for tool response/padding tokens
        #   e.g., [1,1,1,1,1,1,1,(tool start),0,0(tool end),1,1,0,0,0,0]
        # - position_ids: sequential positions for tokens, starting at 0
        #   e.g., [0,0,0,0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,0,0,0,0]

        # TODO(wuxibin): remove padding and use tensordict.
        prompt_max_length = int(self.config.actor_rollout_ref.rollout.prompt_length)
        response_max_length = int(self.config.actor_rollout_ref.rollout.response_length)
        prompt_ids = list(output.prompt_ids)
        if len(prompt_ids) > prompt_max_length:
            # Match left-truncation semantics: keep the prompt tail that is closest
            # to generation when a streamed backend returns an overlong prompt.
            prompt_ids = prompt_ids[-prompt_max_length:]
        response_ids = list(output.response_ids)
        response_mask_ids = list(output.response_mask)
        if len(response_ids) > response_max_length:
            response_ids = response_ids[:response_max_length]
            response_mask_ids = response_mask_ids[:response_max_length]

        self.tokenizer.padding_side = "left"
        prompt_output = self.tokenizer.pad(
            {"input_ids": prompt_ids},
            padding="max_length",
            max_length=prompt_max_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if prompt_output["input_ids"].dim() == 1:
            prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
            prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)

        self.tokenizer.padding_side = "right"
        response_output = self.tokenizer.pad(
            {"input_ids": response_ids},
            padding="max_length",
            max_length=response_max_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if response_output["input_ids"].dim() == 1:
            response_output["input_ids"] = response_output["input_ids"].unsqueeze(0)
            response_output["attention_mask"] = response_output["attention_mask"].unsqueeze(0)

        response_mask_output = self.tokenizer.pad(
            {"input_ids": response_mask_ids},
            padding="max_length",
            max_length=response_max_length,
            return_tensors="pt",
            return_attention_mask=False,
        )
        if response_mask_output["input_ids"].dim() == 1:
            response_mask_output["input_ids"] = response_mask_output["input_ids"].unsqueeze(0)

        response_logprobs = None
        if output.response_logprobs is not None:
            truncated_logprobs = list(output.response_logprobs)[:response_max_length]
            pad_size = response_max_length - len(truncated_logprobs)
            response_logprobs = torch.tensor(truncated_logprobs + [0.0] * pad_size).unsqueeze(0)

        response_mask = response_mask_output["input_ids"] * response_output["attention_mask"]
        attention_mask = torch.cat([prompt_output["attention_mask"], response_output["attention_mask"]], dim=1)
        input_ids = torch.cat([prompt_output["input_ids"], response_output["input_ids"]], dim=1)

        routed_experts = None
        if output.routed_experts is not None:
            total_length = input_ids.shape[1]
            length, layer_num, topk_num = output.routed_experts.shape
            if isinstance(output.routed_experts, np.ndarray):
                experts_tensor = torch.from_numpy(output.routed_experts)
            elif isinstance(output.routed_experts, torch.Tensor):
                experts_tensor = output.routed_experts
            else:
                raise TypeError(f"Unsupported type for routed_experts: {type(output.routed_experts)}")
            routed_experts = torch.zeros(1, total_length, layer_num, topk_num, dtype=experts_tensor.dtype)

            # Calculate start position: left padding means original prompt starts at the end
            start_pos = prompt_output["input_ids"].shape[1] - len(prompt_ids)
            end_pos = min(start_pos + length, total_length)

            # Add boundary checks for robustness
            if start_pos < 0 or end_pos > total_length:
                raise ValueError(
                    f"Invalid position range: start_pos={start_pos}, end_pos={end_pos}, total_length={total_length}"
                )

            routed_experts[:, start_pos:end_pos] = experts_tensor.unsqueeze(0)

        multi_modal_inputs = self._compute_multi_modal_inputs(output, input_ids)
        position_ids = self._compute_position_ids(input_ids, attention_mask, multi_modal_inputs)
        if not skip_compute_score:
            await self._compute_score(
                output,
                prompts=prompt_output["input_ids"],
                responses=response_output["input_ids"],
                attention_mask=attention_mask,
                input_ids=input_ids,
                position_ids=position_ids,
                kwargs=score_kwargs,
            )

        return _InternalAgentLoopOutput(
            prompt_ids=prompt_output["input_ids"],
            response_ids=response_output["input_ids"],
            input_ids=input_ids,
            position_ids=position_ids,
            response_mask=response_mask,
            attention_mask=attention_mask,
            response_logprobs=response_logprobs,
            routed_experts=routed_experts,
            multi_modal_inputs=multi_modal_inputs,
            multi_modal_data=output.multi_modal_data,
            reward_score=output.reward_score,
            num_turns=output.num_turns,
            metrics=output.metrics,
            extra_fields=output.extra_fields,
        )

    def _compute_multi_modal_inputs(self, output, input_ids) -> dict[str, torch.Tensor]:
        """Compute multi-modal inputs with image and video."""
        multi_modal_inputs = {}
        if self.processor is None:
            return multi_modal_inputs

        images = output.multi_modal_data.get("images")
        videos = output.multi_modal_data.get("videos")
        # split the videos and according metadatas
        if videos is not None:
            videos, video_metadatas = zip(*videos, strict=False)
            videos, video_metadatas = list(videos), list(video_metadatas)
        else:
            video_metadatas = None
        current_text = self.tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)
        multi_modal_inputs = self.processor(
            text=[current_text],
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            return_tensors="pt",
            do_sample_frames=False,
        )
        multi_modal_inputs.pop("input_ids", None)
        multi_modal_inputs.pop("attention_mask", None)

        # We must use dict(multi_modal_inputs) to convert BatchFeature values to a new dict
        # because np.array() only keeps the keys for BatchFeature.
        multi_modal_inputs = dict(multi_modal_inputs.convert_to_tensors("pt"))
        image_grid_thw = multi_modal_inputs.get("image_grid_thw")
        if image_grid_thw is not None:
            images_seqlens = torch.repeat_interleave(image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0])
            multi_modal_inputs["images_seqlens"] = images_seqlens
        return multi_modal_inputs

    def _compute_position_ids(self, input_ids, attention_mask, multi_modal_inputs) -> torch.Tensor:
        """Compute position ids for multi-modal inputs."""
        if self.processor is None:
            return compute_position_id_with_mask(attention_mask)  # (1, seq_len)

        image_grid_thw = multi_modal_inputs.get("image_grid_thw")
        video_grid_thw = multi_modal_inputs.get("video_grid_thw")

        # Model's get_rope_index has been dynamically bind to the processor.
        vision_position_ids, _ = self.processor.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )
        vision_position_ids = vision_position_ids.transpose(0, 1)  # (3, 1, seq_len) => (1, 3, seq_len)

        valid_mask = attention_mask[0].bool()
        text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
        text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
        text_position_ids = text_position_ids.unsqueeze(0)
        position_ids = torch.cat((text_position_ids, vision_position_ids), dim=1)  # (1, 4, seq_length)
        return position_ids

    async def _compute_score(self, output, prompts, responses, attention_mask, input_ids, position_ids, kwargs):
        """Compute reward score for single sample."""
        enable_async_reward = self.reward_loop_worker_handles is not None

        if output.reward_score is None and enable_async_reward:
            batch = TensorDict(
                {
                    "prompts": prompts,  # [1, prompt_length]
                    "responses": responses,  # [1, response_length]
                    "attention_mask": attention_mask,  # [1, prompt_length + response_length]
                    "input_ids": input_ids,  # [1, prompt_length + response_length]
                    "position_ids": position_ids,
                },
                batch_size=1,
            )
            non_tensor_batch = {
                **{k: np.array([v]) for k, v in kwargs.items()},
                "__num_turns__": np.array([output.num_turns]),
                "tool_extra_fields": np.array([output.extra_fields], dtype=object),
            }

            data = DataProto(
                batch=batch,
                non_tensor_batch=non_tensor_batch,
            )
            selected_reward_loop_worker_handle = random.choice(self.reward_loop_worker_handles)
            result = await selected_reward_loop_worker_handle.compute_score.remote(data)
            output.reward_score = result["reward_score"]
            output.extra_fields["reward_extra_info"] = result["reward_extra_info"]
        if os.environ.get("ORL_QWEN_DLM_REPLAY_REWARD", "0") == "1" and output.reward_score is not None:
            valid = attention_mask[0, prompts.shape[1] :].bool()
            self._apply_sdar_reward_replay(output, responses[0][valid].tolist())

    def _apply_sdar_reward_replay(self, output, response: list[int]) -> None:
        """Take the reference run's verdict for a response it also produced: code tests can depend on hash order
        (e.g. a list built from a set of strings) and judge the same program differently from one run to the next."""
        import hashlib

        from verl.trainer.ppo.ray_trainer import _summarize_sglang_sdar_blocks

        plan = _snapshot_replay_plan()
        step = getattr(self, "_sdar_replay_global_step", None)
        info = output.extra_fields.get("reward_extra_info")
        if plan is None or step is None or getattr(self, "tokenizer", None) is None or not isinstance(info, dict):
            return
        text = self.tokenizer.decode(response, skip_special_tokens=True)
        entries = plan["entries"].get(f"{step}:{hashlib.sha1(text.encode()).hexdigest()}") or []
        if len(entries) > 1:
            summary = _summarize_sglang_sdar_blocks(
                output.extra_fields.get("sglang_sdar_metrics"), returned_tokens=len(response)
            )
            entries = [
                e
                for e in entries
                if e["denoise_forwards"] == int(summary.get("denoise_forwards", -1))
                and e["edit_change_tokens"] == int(summary.get("edit_change_tokens", -1))
            ] or entries
        verdicts = {json.dumps(e["reward"], sort_keys=True) for e in entries if e.get("reward")}
        if len(verdicts) != 1:
            return
        reference = json.loads(verdicts.pop())
        before = (float(output.reward_score), info.get("acc"))
        output.reward_score = float(reference["score"])
        for key, value in reference.items():
            if key in info:
                info[key] = value
        if before != (output.reward_score, info.get("acc")):
            print(
                f"[SDAR_REWARD_REPLAY] step={step} score {before[0]} -> {output.reward_score} "
                f"acc {before[1]} -> {info.get('acc')}",
                flush=True,
            )

    def _postprocess(
        self,
        inputs: list[_InternalAgentLoopOutput],
        input_non_tensor_batch: dict | None = None,
        defer_sdar_attach: bool = False,
    ) -> DataProto:
        """Process the padded outputs from _run_agent_loop and combine them into a batch."""
        # Convert lists back to tensors and stack them to create a batch.
        prompt_ids = torch.cat([input.prompt_ids for input in inputs], dim=0)
        response_ids = torch.cat([input.response_ids for input in inputs], dim=0)
        response_mask = torch.cat([input.response_mask for input in inputs], dim=0)
        attention_mask = torch.cat([input.attention_mask for input in inputs], dim=0)
        input_ids = torch.cat([input.input_ids for input in inputs], dim=0)
        position_ids = torch.cat([input.position_ids for input in inputs], dim=0)
        optional_outputs = {}
        if inputs[0].response_logprobs is not None:
            optional_outputs["rollout_log_probs"] = torch.cat([input.response_logprobs for input in inputs], dim=0)
        elif self.config.actor_rollout_ref.rollout.get("speed_only_dummy_rollout_logprobs", False):
            optional_outputs["rollout_log_probs"] = torch.zeros_like(response_mask, dtype=torch.float32)
        if inputs[0].routed_experts is not None:
            optional_outputs["routed_experts"] = torch.cat([input.routed_experts for input in inputs], dim=0)

        batch = TensorDict(
            {
                "prompts": prompt_ids,  # [bsz, prompt_length]
                "responses": response_ids,  # [bsz, response_length]
                "response_mask": response_mask,  # [bsz, response_length]
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                # position_ids: [bsz, 3, prompt_length + response_length] or [bsz, prompt_length + response_length]
                "position_ids": position_ids,
                **optional_outputs,
            },
            batch_size=len(inputs),
        )

        scores = [input.reward_score for input in inputs]
        if all(score is not None for score in scores):
            prompt_length = prompt_ids.size(1)
            response_length = attention_mask[:, prompt_length:].sum(dim=1) - 1
            rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
            rm_scores[torch.arange(response_mask.size(0)), response_length] = torch.tensor(scores, dtype=torch.float32)
            batch["rm_scores"] = rm_scores

        non_tensor_batch = {
            "__num_turns__": np.array([input.num_turns for input in inputs], dtype=np.int32),
        }
        if self.reward_loop_worker_handles is None and input_non_tensor_batch:
            non_tensor_batch.update(input_non_tensor_batch)
        elif input_non_tensor_batch and "mixed_prob_threshold" in input_non_tensor_batch:
            # The async-reward path skips the wholesale input merge (reward
            # fields own the output non_tensor); the per-trace threshold
            # column is ours alone, so carry it through for group analysis.
            non_tensor_batch["mixed_prob_threshold"] = input_non_tensor_batch[
                "mixed_prob_threshold"
            ]

        # add reward_extra_info to non_tensor_batch
        reward_extra_infos = [input.extra_fields.get("reward_extra_info", {}) for input in inputs]
        reward_extra_keys = list(reward_extra_infos[0].keys())
        for key in reward_extra_keys:
            non_tensor_batch[key] = np.array([info[key] for info in reward_extra_infos])

        # Add multi_modal_inputs to non_tensor_batch if any samples have them
        multi_modal_inputs_list = [input.multi_modal_inputs for input in inputs]
        if any(mmi is not None for mmi in multi_modal_inputs_list):
            non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs_list, dtype=object)

        metrics = [input.metrics.model_dump() for input in inputs]
        # Collect extra fields from all inputs and convert them to np.ndarray
        # Keep a stable set of keys so downstream batch concat stays consistent across agent loops.
        extra_fields = {}
        default_extra_keys = {
            "turn_scores",
            "tool_rewards",
            "is_cancel",
            "param_version_start",
            "param_version_end",
            "extras",
            "request_uid",
            "sample_uid",
            "response_uid",
        }
        all_keys = set(key for input_item in inputs for key in input_item.extra_fields) | default_extra_keys
        for key in all_keys:
            temp_arr = np.empty(len(inputs), dtype=object)
            temp_arr[:] = [input.extra_fields.get(key) for input in inputs]
            extra_fields[key] = temp_arr

        non_tensor_batch.update(extra_fields)

        # Only include reward_extra_keys in meta_info if rm_scores is in batch
        # This avoids conflicts when reward_tensor is merged later in ray_trainer.py
        if "rm_scores" in batch.keys():
            meta_info = {"metrics": metrics, "reward_extra_keys": reward_extra_keys}
        else:
            meta_info = {"metrics": metrics}

        if self._is_sdar_diffusion_rollout():
            if defer_sdar_attach:
                meta_info["sdar_attach_deferred"] = True
            else:
                sdar_attach_start = time.perf_counter()
                self._attach_sdar_diffusion_fields(batch, non_tensor_batch)
                meta_info["sdar_worker_attach_seconds"] = (
                    time.perf_counter() - sdar_attach_start
                )

        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info=meta_info,
        )

    def _is_sdar_diffusion_rollout(self) -> bool:
        rollout_cfg = self.config.actor_rollout_ref.rollout
        sglang_kwargs = (rollout_cfg.get("engine_kwargs", {}) or {}).get("sglang", {}) or {}
        return rollout_cfg.name == "sglang" and sglang_kwargs.get("dllm_algorithm") == "SDARPrefixKV"

    @staticmethod
    def _object_array(items: list[Any]) -> np.ndarray:
        arr = np.empty((len(items),), dtype=object)
        arr[:] = items
        return arr

    @staticmethod
    def _sdar_trace_record_final_tokens(
        record: dict[str, Any],
        block_length: int,
    ) -> list[int | None]:
        """Return the best available final token ids for an SDAR trace block."""
        block_tokens = int(record.get("block_end", block_length)) - int(record.get("block_start", 0))
        block_tokens = max(block_tokens, block_length, 1)
        final_tokens: list[int | None] = [None] * block_tokens

        for pos, token_id in enumerate(record.get("final_head_tokens", []) or []):
            if pos >= block_tokens:
                break
            final_tokens[pos] = int(token_id)

        for step in record.get("steps", []) or []:
            if not isinstance(step, dict):
                continue
            for trace_key in ("new_accept_token_logprobs", "edit_token_logprobs"):
                for item in step.get(trace_key, []) or []:
                    if not isinstance(item, dict) or "pos" not in item or "token_id" not in item:
                        continue
                    pos = int(item["pos"])
                    if 0 <= pos < block_tokens:
                        final_tokens[pos] = int(item["token_id"])
        return final_tokens

    def _load_sdar_trace_records(self) -> list[dict[str, Any]]:
        trace_dir = os.environ.get("SGLANG_SDAR_TRACE_DIR")
        if not trace_dir:
            return []
        trace_path = os.path.join(trace_dir, "rank0_trace.jsonl")
        if not os.path.exists(trace_path):
            return []
        records = []
        with open(trace_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed SGLang SDAR trace line in %s", trace_path)
        return records

    def _load_sdar_self_conditioning_by_request(self) -> dict[int, dict[int, dict[str, Any]]]:
        trace_dir = os.environ.get("SGLANG_SDAR_TRACE_DIR")
        if not trace_dir:
            return {}
        trace_path = os.path.join(trace_dir, "rank0_trace.jsonl")
        if not os.path.exists(trace_path):
            return {}

        def _resolve_tensor_path(tensor_path: str) -> str:
            if os.path.isabs(tensor_path):
                return tensor_path
            return os.path.join(os.path.dirname(os.path.abspath(trace_path)), tensor_path)

        def _as_tensor(value: Any, dtype: torch.dtype | None = None) -> torch.Tensor | None:
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                return value.detach().cpu()
            return torch.as_tensor(value, dtype=dtype)

        by_request: dict[int, dict[int, dict[str, Any]]] = {}
        try:
            with open(trace_path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    req_pool_idx = record.get("req_pool_idx")
                    if req_pool_idx is None:
                        continue
                    request_states = by_request.setdefault(int(req_pool_idx), {})
                    for step in record.get("steps", []):
                        if not isinstance(step, dict):
                            continue
                        meta = step.get("replay_tensors")
                        if not isinstance(meta, dict):
                            continue
                        item = meta
                        tensor_path = meta.get("tensor_path")
                        if tensor_path:
                            item = torch.load(_resolve_tensor_path(str(tensor_path)), map_location="cpu")
                        if not isinstance(item, dict):
                            continue
                        signal = _as_tensor(item.get("self_conditioning_signal"), dtype=torch.float32)
                        logits_source_hidden = _as_tensor(item.get("logits_source_hidden"), dtype=torch.float32)
                        input_ids = _as_tensor(item.get("input_ids"), dtype=torch.long)
                        if signal is None and logits_source_hidden is None and input_ids is None:
                            continue
                        valid = _as_tensor(item.get("valid"), dtype=torch.bool)
                        position = item.get("position", meta.get("position"))
                        if isinstance(position, torch.Tensor):
                            position = tuple(int(x) for x in position.detach().cpu().tolist())
                        elif isinstance(position, list):
                            position = tuple(int(x) for x in position)
                        step_key = int(meta.get("step_key", step.get("step", 0)))
                        request_states[step_key] = {
                            "position": position,
                            "valid": valid,
                            "input_ids": input_ids,
                            "self_conditioning_signal": signal,
                            "logits_source_hidden": logits_source_hidden,
                        }
        except Exception as exc:
            logger.warning("Failed to load SGLang SDAR replay tensor trace: %s", exc)
            return {}
        return by_request

    def _build_sdar_trace_maps(
        self,
        response_lists: list[list[int]],
        prompt_tail_lengths: list[int],
        replay_tensors_by_sample: list[Any] | None = None,
        trace_records: list[dict[str, Any]] | None = None,
        replay_samples: list[tuple[int, str, int, int]] | None = None,
    ) -> tuple[list[list[int]], list[dict[int, dict[str, Any]]], list[dict[str, Any]]]:
        from sglang.srt.dllm.algorithm.sdar_gpu_replay_lease import (
            is_gpu_replay_lease,
            slice_gpu_replay_lease,
        )

        rollout_cfg = self.config.actor_rollout_ref.rollout
        block_length = int((rollout_cfg.get("engine_kwargs", {}) or {}).get("block_length", 1))
        block_length = max(block_length, 1)
        records = (
            trace_records
            if trace_records is not None
            else self._load_sdar_trace_records()
        )
        replay_status: list[dict[str, Any]] = [{} for _ in response_lists]
        if replay_samples is not None and records:
            records, replay_status = self._apply_snapshot_replay(
                records, response_lists, prompt_tail_lengths, block_length, replay_samples
            )
        trace_dir = os.environ.get("SGLANG_SDAR_TRACE_DIR")
        trace_path = os.path.join(trace_dir, "rank0_trace.jsonl") if trace_dir else None

        def _resolve_tensor_path(tensor_path: str) -> str:
            if os.path.isabs(tensor_path):
                return tensor_path
            if trace_path is None:
                return tensor_path
            return os.path.join(os.path.dirname(os.path.abspath(trace_path)), tensor_path)

        def _as_tensor(value: Any, dtype: torch.dtype | None = None) -> torch.Tensor | None:
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                return value.detach().cpu()
            return torch.as_tensor(value, dtype=dtype)

        def _as_replay_hidden(
            value: Any,
        ) -> torch.Tensor | dict[str, Any] | None:
            if is_gpu_replay_lease(value):
                return dict(value)
            return _as_tensor(value, dtype=torch.float32)

        def _normalize_replay_item(item: Any, meta: dict[str, Any] | None = None) -> dict[str, Any] | None:
            meta = meta if isinstance(meta, dict) else {}
            if not isinstance(meta, dict):
                meta = {}
            if not isinstance(item, dict):
                return None
            signal = _as_replay_hidden(
                item.get("self_conditioning_signal")
            )
            logits_source_hidden = _as_replay_hidden(
                item.get("logits_source_hidden")
            )
            input_ids = _as_tensor(item.get("input_ids"), dtype=torch.long)
            tensor_path = item.get("tensor_path", meta.get("tensor_path"))
            if (
                signal is None
                and logits_source_hidden is None
                and input_ids is None
                and not tensor_path
            ):
                return None
            valid = _as_tensor(item.get("valid"), dtype=torch.bool)
            prompt_mask = _as_tensor(item.get("prompt_mask"), dtype=torch.bool)
            action_ids = _as_tensor(item.get("action_ids"), dtype=torch.long)
            action_mask = _as_tensor(item.get("action_mask"), dtype=torch.bool)
            random_vocab_mask = _as_tensor(
                item.get("random_vocab_mask"),
                dtype=torch.bool,
            )
            position = item.get("position", meta.get("position"))
            if isinstance(position, torch.Tensor):
                position = tuple(int(x) for x in position.detach().cpu().tolist())
            elif isinstance(position, list):
                position = tuple(int(x) for x in position)
            def _producer_value(key: str, default: Any = None) -> Any:
                return item.get(key, meta.get(key, default))

            return {
                "position": position,
                "valid": valid,
                "prompt_mask": prompt_mask,
                "input_ids": input_ids,
                "self_conditioning_signal": signal,
                "logits_source_hidden": logits_source_hidden,
                "action_ids": action_ids,
                "action_mask": action_mask,
                "random_vocab_mask": random_vocab_mask,
                "tensor_path": str(tensor_path) if tensor_path else None,
                "trace_root": item.get("trace_root", meta.get("trace_root")),
                "tensor_file_bytes": int(
                    item.get(
                        "tensor_file_bytes",
                        meta.get("tensor_file_bytes", 0),
                    )
                    or 0
                ),
                "step_key": int(
                    item.get("step_key", meta.get("step_key", -1))
                ),
                "req_pool_idx": int(
                    item.get("req_pool_idx", meta.get("req_pool_idx", -1))
                ),
                "local_step": int(
                    item.get("local_step", meta.get("local_step", 0))
                ),
                "block_index": int(
                    item.get(
                        "block_index",
                        item.get(
                            "block_idx",
                            meta.get("block_index", meta.get("block_idx", 0)),
                        ),
                    )
                ),
                "start_offset": int(
                    item.get(
                        "start_offset",
                        meta.get("start_offset", 0),
                    )
                ),
                "response_start_index": int(
                    item.get(
                        "response_start_index",
                        meta.get("response_start_index", 0),
                    )
                ),
                "new_accept_count": int(
                    item.get(
                        "new_accept_count",
                        meta.get("new_accept_count", 0),
                    )
                ),
                "new_accept_tokens": int(
                    item.get(
                        "new_accept_tokens",
                        meta.get("new_accept_tokens", 0),
                    )
                ),
                "denoise_forwards": int(
                    item.get(
                        "denoise_forwards",
                        meta.get("denoise_forwards", 0),
                    )
                ),
                "action_token_count": int(
                    item.get(
                        "action_token_count",
                        meta.get("action_token_count", 0),
                    )
                ),
                "is_final_denoise_row": bool(
                    item.get(
                        "is_final_denoise_row",
                        meta.get("is_final_denoise_row", False),
                    )
                ),
                "is_terminal_commit": bool(
                    item.get(
                        "is_terminal_commit",
                        meta.get("is_terminal_commit", False),
                    )
                ),
                "producer_preselected": bool(
                    item.get(
                        "producer_preselected",
                        meta.get("producer_preselected", False),
                    )
                ),
                "producer_preselection_mode": str(
                    _producer_value("producer_preselection_mode", "")
                    or ""
                ),
                "producer_source_rows": int(
                    _producer_value("producer_source_rows", 0) or 0
                ),
                "producer_rank_selected_rows": int(
                    _producer_value(
                        "producer_rank_selected_rows",
                        0,
                    )
                    or 0
                ),
                "producer_selected_rows": int(
                    _producer_value("producer_selected_rows", 0) or 0
                ),
                "producer_final_rows": int(
                    _producer_value("producer_final_rows", 0) or 0
                ),
                "producer_forced_final_rows": int(
                    _producer_value(
                        "producer_forced_final_rows",
                        0,
                    )
                    or 0
                ),
                "producer_final_already_selected_rows": int(
                    _producer_value(
                        "producer_final_already_selected_rows",
                        0,
                    )
                    or 0
                ),
                "producer_source_score_min": int(
                    _producer_value("producer_source_score_min", 0)
                    or 0
                ),
                "producer_source_score_sum": int(
                    _producer_value("producer_source_score_sum", 0)
                    or 0
                ),
                "producer_source_score_max": int(
                    _producer_value("producer_source_score_max", 0)
                    or 0
                ),
                "producer_zero_variance": bool(
                    _producer_value("producer_zero_variance", False)
                ),
                "producer_block_index": int(
                    _producer_value(
                        "producer_block_index",
                        item.get(
                            "block_index",
                            item.get(
                                "block_idx",
                                meta.get(
                                    "block_index",
                                    meta.get("block_idx", 0),
                                ),
                            ),
                        ),
                    )
                ),
                "producer_state_selection": bool(
                    _producer_value("producer_state_selection", False)
                ),
                "producer_state_selection_version": int(
                    _producer_value(
                        "producer_state_selection_version",
                        0,
                    )
                    or 0
                ),
                "producer_state_selection_complete": bool(
                    _producer_value(
                        "producer_state_selection_complete",
                        False,
                    )
                ),
                "producer_state_selection_mode": str(
                    _producer_value(
                        "producer_state_selection_mode",
                        "",
                    )
                    or ""
                ),
                "producer_state_selection_fraction": float(
                    _producer_value(
                        "producer_state_selection_fraction",
                        0.0,
                    )
                    or 0.0
                ),
                "producer_state_selection_direction": str(
                    _producer_value(
                        "producer_state_selection_direction",
                        "",
                    )
                    or ""
                ),
                "producer_state_selection_score": str(
                    _producer_value(
                        "producer_state_selection_score",
                        "",
                    )
                    or ""
                ),
                "producer_state_selection_group_fields": str(
                    _producer_value(
                        "producer_state_selection_group_fields",
                        "",
                    )
                    or ""
                ),
                "producer_state_selection_final_policy": str(
                    _producer_value(
                        "producer_state_selection_final_policy",
                        "",
                    )
                    or ""
                ),
                "producer_state_selection_rank_selected": bool(
                    _producer_value(
                        "producer_state_selection_rank_selected",
                        False,
                    )
                ),
                "producer_state_selection_forced_final": bool(
                    _producer_value(
                        "producer_state_selection_forced_final",
                        False,
                    )
                ),
                "adjacent_entropy_pair_valid": bool(
                    _producer_value("adjacent_entropy_pair_valid", False)
                ),
                "adjacent_entropy_prev_step_key": int(
                    _producer_value("adjacent_entropy_prev_step_key", -1) or -1
                ),
                "adjacent_entropy_pair_local_step": int(
                    _producer_value("adjacent_entropy_pair_local_step", -1)
                    if _producer_value("adjacent_entropy_pair_local_step", -1) is not None
                    else -1
                ),
                "adjacent_entropy_prev_mean": float(
                    _producer_value("adjacent_entropy_prev_mean", 0.0) or 0.0
                ),
                "adjacent_entropy_rollout_next_mean": float(
                    _producer_value("adjacent_entropy_rollout_next_mean", 0.0) or 0.0
                ),
                "adjacent_entropy_vocab_size": int(
                    _producer_value("adjacent_entropy_vocab_size", 0) or 0
                ),
                "adjacent_entropy_next_mask": (
                    _as_tensor(
                        item.get("adjacent_entropy_next_mask"),
                        dtype=torch.bool,
                    )
                ),
                # Global AEC (all-valid scope) fields
                "global_aec_pair_valid": bool(
                    _producer_value("global_aec_pair_valid", False)
                ),
                "global_aec_prev_step_key": int(
                    _producer_value("global_aec_prev_step_key", -1) or -1
                ),
                "global_aec_prev_mean": float(
                    _producer_value("global_aec_prev_mean", 0.0) or 0.0
                ),
                "global_aec_rollout_next_mean": float(
                    _producer_value("global_aec_rollout_next_mean", 0.0) or 0.0
                ),
                "global_aec_vocab_size": int(
                    _producer_value("global_aec_vocab_size", 0) or 0
                ),
                "global_aec_next_mask": (
                    _as_tensor(
                        item.get("global_aec_next_mask"),
                        dtype=torch.bool,
                    )
                ),
            }

        def _load_step_replay_item(step: dict[str, Any]) -> dict[str, Any] | None:
            meta = step.get("replay_tensors")
            if not isinstance(meta, dict):
                return None
            item: Any = meta
            tensor_path = meta.get("tensor_path")
            if tensor_path:
                item = torch.load(_resolve_tensor_path(str(tensor_path)), map_location="cpu")
            return _normalize_replay_item(item, meta)

        def _trace_step_key(record: dict[str, Any], step: dict[str, Any]) -> int:
            meta = step.get("replay_tensors")
            if isinstance(meta, dict) and meta.get("step_key") is not None:
                return int(meta["step_key"])
            block_idx = int(record.get("block_idx", 0))
            local_step = int(step.get("step", 0))
            return block_idx * block_length + local_step

        def _compact_replay_by_step(payload: Any) -> dict[int, dict[str, Any]]:
            payload = self._materialize_sdar_replay_payload(payload)
            if (
                not isinstance(payload, dict)
                or payload.get("format")
                not in {
                    "sglang_sdar_replay_tensor_batch_v1",
                    "sglang_sdar_replay_metadata_batch_v1",
                }
            ):
                return {}
            step_keys = _as_tensor(payload.get("step_keys"), dtype=torch.long)
            positions = _as_tensor(payload.get("positions"), dtype=torch.long)
            if step_keys is None:
                return {}
            n = int(step_keys.numel())
            out: dict[int, dict[str, Any]] = {}
            for i in range(n):
                def _owned_row(key: str) -> torch.Tensor | None:
                    value = payload.get(key)
                    if not isinstance(value, torch.Tensor):
                        return None
                    return self._sdar_owned_tensor_row(value, i)

                def _owned_hidden(
                    key: str,
                ) -> torch.Tensor | dict[str, Any] | None:
                    tensor = _owned_row(key)
                    if tensor is not None:
                        return tensor
                    leases = payload.get(f"{key}_gpu_leases")
                    if (
                        isinstance(leases, list)
                        and i < len(leases)
                        and is_gpu_replay_lease(leases[i])
                    ):
                        return dict(leases[i])
                    return None

                def _compact_scalar(
                    key: str,
                    default: Any,
                ) -> Any:
                    values = payload.get(key)
                    if isinstance(values, torch.Tensor) and i < int(
                        values.shape[0]
                    ):
                        return values[i].item()
                    return default

                def _compact_string(key: str) -> str:
                    values = payload.get(key)
                    if isinstance(values, list) and i < len(values):
                        return str(values[i])
                    return ""

                item = {
                    "step_key": int(step_keys[i].item()),
                    "position": tuple(int(x) for x in positions[i].tolist()) if positions is not None else None,
                    "valid": _owned_row("valid"),
                    "prompt_mask": _owned_row("prompt_mask"),
                    "input_ids": _owned_row("input_ids"),
                    "action_ids": _owned_row("action_ids"),
                    "action_mask": _owned_row("action_mask"),
                    "random_vocab_mask": _owned_row(
                        "random_vocab_mask"
                    ),
                    "self_conditioning_signal": _owned_hidden(
                        "self_conditioning_signal"
                    ),
                    "logits_source_hidden": _owned_hidden(
                        "logits_source_hidden"
                    ),
                    "tensor_path": (
                        payload["tensor_paths"][i]
                        if isinstance(payload.get("tensor_paths"), list)
                        and i < len(payload["tensor_paths"])
                        else None
                    ),
                    "trace_root": (
                        payload["trace_roots"][i]
                        if isinstance(payload.get("trace_roots"), list)
                        and i < len(payload["trace_roots"])
                        else None
                    ),
                    "tensor_file_bytes": (
                        int(payload["tensor_file_bytes"][i].item())
                        if isinstance(
                            payload.get("tensor_file_bytes"),
                            torch.Tensor,
                        )
                        else 0
                    ),
                    "local_step": (
                        int(payload["local_steps"][i].item())
                        if isinstance(payload.get("local_steps"), torch.Tensor)
                        else 0
                    ),
                    "req_pool_idx": (
                        int(payload["req_pool_indices"][i].item())
                        if isinstance(
                            payload.get("req_pool_indices"),
                            torch.Tensor,
                        )
                        else -1
                    ),
                    "block_index": (
                        int(payload["block_indices"][i].item())
                        if isinstance(payload.get("block_indices"), torch.Tensor)
                        else 0
                    ),
                    "start_offset": (
                        int(payload["start_offsets"][i].item())
                        if isinstance(
                            payload.get("start_offsets"),
                            torch.Tensor,
                        )
                        else 0
                    ),
                    "response_start_index": (
                        int(payload["response_start_indices"][i].item())
                        if isinstance(
                            payload.get("response_start_indices"),
                            torch.Tensor,
                        )
                        else 0
                    ),
                    "new_accept_count": (
                        int(payload["new_accept_counts"][i].item())
                        if isinstance(payload.get("new_accept_counts"), torch.Tensor)
                        else 0
                    ),
                    "new_accept_tokens": (
                        int(payload["new_accept_tokens"][i].item())
                        if isinstance(payload.get("new_accept_tokens"), torch.Tensor)
                        else 0
                    ),
                    "denoise_forwards": (
                        int(payload["denoise_forwards"][i].item())
                        if isinstance(payload.get("denoise_forwards"), torch.Tensor)
                        else 0
                    ),
                    "action_token_count": (
                        int(payload["action_token_counts"][i].item())
                        if isinstance(payload.get("action_token_counts"), torch.Tensor)
                        else 0
                    ),
                    "is_final_denoise_row": (
                        bool(payload["is_final_denoise_rows"][i].item())
                        if isinstance(
                            payload.get("is_final_denoise_rows"),
                            torch.Tensor,
                        )
                        else False
                    ),
                    "is_terminal_commit": (
                        bool(payload["is_terminal_commits"][i].item())
                        if isinstance(
                            payload.get("is_terminal_commits"),
                            torch.Tensor,
                        )
                        else False
                    ),
                    "producer_preselected": (
                        bool(payload["producer_preselected_rows"][i].item())
                        if isinstance(
                            payload.get("producer_preselected_rows"),
                            torch.Tensor,
                        )
                        else False
                    ),
                    "producer_preselection_mode": _compact_string(
                        "producer_preselection_modes"
                    ),
                    "producer_source_rows": int(
                        _compact_scalar("producer_source_rows", 0)
                    ),
                    "producer_rank_selected_rows": int(
                        _compact_scalar(
                            "producer_rank_selected_rows",
                            0,
                        )
                    ),
                    "producer_selected_rows": int(
                        _compact_scalar("producer_selected_rows", 0)
                    ),
                    "producer_final_rows": int(
                        _compact_scalar("producer_final_rows", 0)
                    ),
                    "producer_forced_final_rows": int(
                        _compact_scalar(
                            "producer_forced_final_rows",
                            0,
                        )
                    ),
                    "producer_final_already_selected_rows": int(
                        _compact_scalar(
                            "producer_final_already_selected_rows",
                            0,
                        )
                    ),
                    "producer_source_score_min": int(
                        _compact_scalar(
                            "producer_source_score_min",
                            0,
                        )
                    ),
                    "producer_source_score_sum": int(
                        _compact_scalar(
                            "producer_source_score_sum",
                            0,
                        )
                    ),
                    "producer_source_score_max": int(
                        _compact_scalar(
                            "producer_source_score_max",
                            0,
                        )
                    ),
                    "producer_zero_variance": bool(
                        _compact_scalar(
                            "producer_zero_variance_groups",
                            False,
                        )
                    ),
                    "producer_block_index": int(
                        _compact_scalar(
                            "producer_block_indices",
                            (
                                int(payload["block_indices"][i].item())
                                if isinstance(
                                    payload.get("block_indices"),
                                    torch.Tensor,
                                )
                                else 0
                            ),
                        )
                    ),
                    "producer_state_selection": bool(
                        _compact_scalar(
                            "producer_state_selection_rows",
                            False,
                        )
                    ),
                    "producer_state_selection_version": int(
                        _compact_scalar(
                            "producer_state_selection_versions",
                            0,
                        )
                    ),
                    "producer_state_selection_complete": bool(
                        _compact_scalar(
                            "producer_state_selection_complete_rows",
                            False,
                        )
                    ),
                    "producer_state_selection_mode": _compact_string(
                        "producer_state_selection_modes"
                    ),
                    "producer_state_selection_fraction": float(
                        _compact_scalar(
                            "producer_state_selection_fractions",
                            0.0,
                        )
                    ),
                    "producer_state_selection_direction": (
                        _compact_string(
                            "producer_state_selection_directions"
                        )
                    ),
                    "producer_state_selection_score": _compact_string(
                        "producer_state_selection_scores"
                    ),
                    "producer_state_selection_group_fields": (
                        _compact_string(
                            "producer_state_selection_group_fields"
                        )
                    ),
                    "producer_state_selection_final_policy": (
                        _compact_string(
                            "producer_state_selection_final_policies"
                        )
                    ),
                    "producer_state_selection_rank_selected": bool(
                        _compact_scalar(
                            "producer_state_selection_rank_selected_flags",
                            False,
                        )
                    ),
                    "producer_state_selection_forced_final": bool(
                        _compact_scalar(
                            "producer_state_selection_forced_final_flags",
                            False,
                        )
                    ),
                    "adjacent_entropy_pair_valid": bool(
                        _compact_scalar("adjacent_entropy_pair_valid", False)
                    ),
                    "adjacent_entropy_prev_step_key": int(
                        _compact_scalar("adjacent_entropy_prev_step_key", -1)
                    ),
                    "adjacent_entropy_pair_local_step": int(
                        _compact_scalar("adjacent_entropy_pair_local_step", -1)
                    ),
                    "adjacent_entropy_prev_mean": float(
                        _compact_scalar("adjacent_entropy_prev_mean", 0.0)
                    ),
                    "adjacent_entropy_rollout_next_mean": float(
                        _compact_scalar("adjacent_entropy_rollout_next_mean", 0.0)
                    ),
                    "adjacent_entropy_vocab_size": int(
                        _compact_scalar("adjacent_entropy_vocab_size", 0)
                    ),
                    "adjacent_entropy_next_mask": _owned_row(
                        "adjacent_entropy_next_mask"
                    ),
                    # Global AEC (all-valid scope) fields
                    "global_aec_pair_valid": bool(
                        _compact_scalar("global_aec_pair_valid", False)
                    ),
                    "global_aec_prev_step_key": int(
                        _compact_scalar("global_aec_prev_step_key", -1)
                    ),
                    "global_aec_prev_mean": float(
                        _compact_scalar("global_aec_prev_mean", 0.0)
                    ),
                    "global_aec_rollout_next_mean": float(
                        _compact_scalar("global_aec_rollout_next_mean", 0.0)
                    ),
                    "global_aec_vocab_size": int(
                        _compact_scalar("global_aec_vocab_size", 0)
                    ),
                    "global_aec_next_mask": _owned_row(
                        "global_aec_next_mask"
                    ),
                }
                normalized = _normalize_replay_item(item, item)
                if normalized is not None:
                    out[int(step_keys[i].item())] = normalized
            return out

        used_records: set[int] = set()
        all_step_maps: list[list[int]] = []
        all_self_conditioning: list[dict[int, dict[str, Any]]] = []
        all_trace_map_status: list[dict[str, Any]] = []

        for sample_idx, (response, prompt_tail_len) in enumerate(zip(response_lists, prompt_tail_lengths)):
            sample_replay_by_step: dict[int, dict[str, Any]] = {}
            if replay_tensors_by_sample is not None and sample_idx < len(replay_tensors_by_sample):
                payloads = replay_tensors_by_sample[sample_idx]
                sample_replay_by_step.update(_compact_replay_by_step(payloads))
                if not sample_replay_by_step and isinstance(payloads, (list, tuple)):
                    for payload in payloads or []:
                        if not isinstance(payload, dict):
                            continue
                        meta = payload.get("metadata", payload)
                        if not isinstance(meta, dict) or meta.get("step_key") is None:
                            continue
                        normalized = _normalize_replay_item(payload, meta)
                        if normalized is not None:
                            sample_replay_by_step[int(meta["step_key"])] = normalized
            producer_preselected = bool(sample_replay_by_step) and all(
                bool(item.get("producer_preselected", False))
                for item in sample_replay_by_step.values()
            )
            step_map: list[int] = []
            self_conditioning: dict[int, dict[str, Any]] = dict(sample_replay_by_step)
            exact_chunks = 0
            exact_tokens = 0
            synthetic_chunks = 0
            synthetic_tokens = 0
            start = 0
            while start < len(response):
                match_idx = None
                match_candidates: list[tuple[int, int, int]] = []
                expected_start_offset = prompt_tail_len if start == 0 else 0
                for idx, record in enumerate(records):
                    if idx in used_records:
                        continue
                    final_ids = self._sdar_trace_record_final_tokens(record, block_length)
                    start_offset = int(record.get("start_offset", 0))
                    final_response_ids = final_ids[start_offset:]
                    compare_len = min(block_length - start_offset, len(final_response_ids), len(response) - start)
                    if (
                        compare_len > 0
                        and all(token_id is not None for token_id in final_response_ids[:compare_len])
                        and response[start : start + compare_len] == final_response_ids[:compare_len]
                    ):
                        match_candidates.append((idx, start_offset, compare_len))
                if match_candidates:
                    offset_matches = [
                        candidate for candidate in match_candidates if candidate[1] == expected_start_offset
                    ]
                    match_idx = max(offset_matches or match_candidates, key=lambda candidate: (candidate[2], candidate[0]))[
                        0
                    ]

                if match_idx is None:
                    # Untraced fallback: train the whole diffusion block as one round.
                    chunk_len = min(block_length, len(response) - start)
                    step_map.extend([start // block_length] * chunk_len)
                    synthetic_chunks += 1
                    synthetic_tokens += chunk_len
                    start += chunk_len
                    continue

                used_records.add(match_idx)
                record = records[match_idx]
                block_idx = int(record.get("block_idx", start // block_length))
                start_offset = int(record.get("start_offset", 0))
                generated_capacity = max(block_length - start_offset, 1)
                chunk_len = min(generated_capacity, len(response) - start)
                local_steps = [-1] * chunk_len
                last_step_key = block_idx * block_length
                step_keys = []
                for step in record.get("steps", []):
                    step_key = _trace_step_key(record, step)
                    step_keys.append(step_key)
                    last_step_key = step_key
                    for pos in step.get("new_accept_positions", []):
                        pos = int(pos)
                        response_pos = pos - start_offset
                        if 0 <= response_pos < len(local_steps) and local_steps[response_pos] < 0:
                            local_steps[response_pos] = step_key
                local_steps = [value if value >= 0 else last_step_key for value in local_steps]
                step_map.extend(local_steps)
                exact_chunks += 1
                exact_tokens += chunk_len

                for step in record.get("steps", []):
                    if not isinstance(step, dict):
                        continue
                    step_key = _trace_step_key(record, step)
                    if step_key not in step_keys:
                        continue
                    item = sample_replay_by_step.get(step_key)
                    if item is None and not producer_preselected:
                        item = _load_step_replay_item(step)
                    if not isinstance(item, dict):
                        continue
                    item = dict(item)
                    for tensor_key in (
                        "valid",
                        "prompt_mask",
                        "input_ids",
                        "action_ids",
                        "action_mask",
                        "random_vocab_mask",
                        "adjacent_entropy_next_mask",
                        "self_conditioning_signal",
                        "logits_source_hidden",
                    ):
                        tensor_value = item.get(tensor_key)
                        if isinstance(tensor_value, torch.Tensor) and tensor_value.shape[0] > start_offset:
                            item[tensor_key] = tensor_value[start_offset : start_offset + chunk_len]
                        elif is_gpu_replay_lease(tensor_value):
                            item[tensor_key] = slice_gpu_replay_lease(
                                tensor_value,
                                start_offset,
                                min(
                                    chunk_len,
                                    int(tensor_value["shape"][0])
                                    - start_offset,
                                ),
                            )
                    item["position"] = (prompt_tail_len + start, prompt_tail_len + start + chunk_len)
                    self_conditioning[step_key] = item
                start += chunk_len

            all_step_maps.append(step_map)
            all_self_conditioning.append(self_conditioning)
            if synthetic_tokens:
                source = "mixed_trace_and_synthetic" if exact_tokens else "synthetic_block_fallback"
            elif exact_tokens:
                source = "exact_trace"
            else:
                source = "empty"
            all_trace_map_status.append(
                {
                    "source": source,
                    "exact_chunks": exact_chunks,
                    "exact_tokens": exact_tokens,
                    "synthetic_chunks": synthetic_chunks,
                    "synthetic_tokens": synthetic_tokens,
                    "response_tokens": len(response),
                    "replay_tensor_steps": len(self_conditioning),
                    "has_replay_tensors": bool(self_conditioning),
                    **replay_status[sample_idx],
                }
            )
            if replay_samples is not None:
                print(
                    f"[SDAR_REPLAY] step={replay_samples[sample_idx][0]} "
                    f"synthetic_tokens={synthetic_tokens} {json.dumps(replay_status[sample_idx], sort_keys=True)}",
                    flush=True,
                )

        return all_step_maps, all_self_conditioning, all_trace_map_status

    def _sdar_replay_samples(
        self, response_lists: list[list[int]], non_tensor_batch: dict[str, Any]
    ) -> list[tuple[int, str, int, int]] | None:
        """(global step, dumped response text, denoise forwards, edit-changed tokens) per sample, or None when the
        snapshot replay is off or this is not a training rollout."""
        step = getattr(self, "_sdar_replay_global_step", None)
        tokenizer = getattr(self, "tokenizer", None)
        if _snapshot_replay_plan() is None or step is None or tokenizer is None:
            return None
        from verl.trainer.ppo.ray_trainer import _summarize_sglang_sdar_blocks

        metrics = non_tensor_batch.get("sglang_sdar_metrics")
        samples = []
        for i, response in enumerate(response_lists):
            summary = _summarize_sglang_sdar_blocks(metrics[i], returned_tokens=len(response)) if metrics is not None else {}
            samples.append(
                (
                    int(step),
                    tokenizer.decode(response, skip_special_tokens=True),
                    int(summary.get("denoise_forwards", -1)),
                    int(summary.get("edit_change_tokens", -1)),
                )
            )
        return samples

    def _apply_snapshot_replay(
        self,
        records: list[dict[str, Any]],
        response_lists: list[list[int]],
        prompt_tail_lengths: list[int],
        block_length: int,
        replay_samples: list[tuple[int, str, int, int]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Give the records of the blocks listed by the replay plan an all-MASK snapshot.

        Chunks are assigned to records with the intact snapshots by the same rule as the trace-map matching; the
        reference plan names chunks by block index relative to the response's first block."""
        import hashlib

        plan = _snapshot_replay_plan()
        mask_id = int(plan.get("mask_token_id", 151669))
        records = list(records)

        def chunk_matches(record: dict[str, Any], response: list[int], start: int) -> tuple[bool, int, int]:
            final_ids = self._sdar_trace_record_final_tokens(record, block_length)
            start_offset = int(record.get("start_offset", 0))
            final_response_ids = final_ids[start_offset:]
            compare_len = min(block_length - start_offset, len(final_response_ids), len(response) - start)
            ok = (
                compare_len > 0
                and all(token_id is not None for token_id in final_response_ids[:compare_len])
                and response[start : start + compare_len] == final_response_ids[:compare_len]
            )
            return ok, start_offset, compare_len

        def masked(record: dict[str, Any]) -> dict[str, Any]:
            out = dict(record)
            out["final_head_tokens"] = [mask_id] * len(record.get("final_head_tokens") or [])
            return out

        used: set[int] = set()
        targets: list[int] = []
        statuses: list[dict[str, Any]] = []
        for sample_idx, (response, prompt_tail_len) in enumerate(zip(response_lists, prompt_tail_lengths)):
            chunks: list[tuple[int, int, int]] = []
            start = 0
            while start < len(response):
                expected_start_offset = prompt_tail_len if start == 0 else 0
                candidates = []
                for idx, record in enumerate(records):
                    if idx in used:
                        continue
                    ok, start_offset, compare_len = chunk_matches(record, response, start)
                    if ok:
                        candidates.append((idx, start_offset, compare_len))
                if candidates:
                    offset_matches = [c for c in candidates if c[1] == expected_start_offset]
                    idx = max(offset_matches or candidates, key=lambda c: (c[2], c[0]))[0]
                    used.add(idx)
                    chunk_len = min(max(block_length - int(records[idx].get("start_offset", 0)), 1), len(response) - start)
                    chunks.append((idx, start, chunk_len))
                else:
                    chunk_len = min(block_length, len(response) - start)
                start += chunk_len

            status: dict[str, Any] = {"replay_paper_match": False}
            statuses.append(status)
            if sample_idx >= len(replay_samples) or not chunks:
                continue
            step, text, denoise_forwards, edit_change_tokens = replay_samples[sample_idx]
            entries = plan["entries"].get(f"{step}:{hashlib.sha1(text.encode()).hexdigest()}") or []
            entry = next(
                (
                    e
                    for e in entries
                    if e["denoise_forwards"] == denoise_forwards and e["edit_change_tokens"] == edit_change_tokens
                ),
                None,
            )
            if entry is None:
                status["replay_text_match"] = bool(entries)
                continue
            first_block = min(int(records[idx].get("block_idx", 0)) for idx, _, _ in chunks)
            mask_blocks = set(entry.get("mask_blocks") or [])
            mask_candidates = set(entry.get("mask_candidates") or [])
            chosen, predicted = [], 0
            for idx, chunk_start, chunk_len in chunks:
                if int(records[idx].get("block_idx", 0)) - first_block in mask_blocks:
                    chosen.append(idx)
                    if not chunk_matches(masked(records[idx]), response, chunk_start)[0]:
                        predicted += chunk_len
            for idx, chunk_start, chunk_len in chunks:
                if predicted >= int(entry["synthetic_tokens"]):
                    break
                if int(records[idx].get("block_idx", 0)) - first_block in mask_candidates and not chunk_matches(
                    masked(records[idx]), response, chunk_start
                )[0]:
                    chosen.append(idx)
                    predicted += chunk_len
            targets.extend(chosen)
            status.update(
                replay_paper_match=True,
                replay_paper_dump_idx=int(entry["dump_idx"]),
                replay_masked_chunks=len(chosen),
                replay_predicted_synthetic_tokens=predicted,
                replay_paper_synthetic_tokens=int(entry["synthetic_tokens"]),
            )
        for idx in targets:
            records[idx] = masked(records[idx])
        return records, statuses

    def _attach_sdar_diffusion_fields(
        self,
        batch: TensorDict,
        non_tensor_batch: dict[str, Any],
        *,
        trace_records: list[dict[str, Any]] | None = None,
    ) -> None:
        if trace_records is None:
            trace_records = self._trace_records_from_sdar_metrics(
                non_tensor_batch.get("sglang_sdar_metrics")
            )
        prompt_length = batch["prompts"].shape[1]
        prompt_ids_tensor = batch["prompts"]
        response_ids_tensor = batch["responses"]
        prompt_mask = batch["attention_mask"][:, :prompt_length].bool()
        response_mask = batch["response_mask"].bool()

        prompt_lists = [
            prompt_ids_tensor[i][prompt_mask[i]].tolist()
            for i in range(prompt_ids_tensor.shape[0])
        ]
        response_lists = [
            response_ids_tensor[i][response_mask[i]].tolist()
            for i in range(response_ids_tensor.shape[0])
        ]
        rollout_cfg = self.config.actor_rollout_ref.rollout
        block_length = int((rollout_cfg.get("engine_kwargs", {}) or {}).get("block_length", 1))
        block_length = max(block_length, 1)
        prompt_tail_lengths = [len(prompt_ids) % block_length for prompt_ids in prompt_lists]
        replay_tensors_by_sample = non_tensor_batch.pop("sglang_sdar_replay_tensors", None)
        if replay_tensors_by_sample is None:
            sglang_meta_infos = non_tensor_batch.get("sglang_meta_info")
            replay_tensors_by_sample = [
                (meta.get("sglang_sdar_replay_tensors") if isinstance(meta, dict) else None)
                for meta in sglang_meta_infos
            ] if sglang_meta_infos is not None else None
        step_map_lists, rollout_self_conditioning, trace_map_status = self._build_sdar_trace_maps(
            response_lists,
            prompt_tail_lengths,
            replay_tensors_by_sample=replay_tensors_by_sample,
            trace_records=trace_records,
            replay_samples=self._sdar_replay_samples(response_lists, non_tensor_batch),
        )

        step_maps = torch.zeros_like(response_ids_tensor, dtype=torch.long)
        log_probs = torch.full(response_ids_tensor.shape, -float("inf"), dtype=torch.float32)
        if "rollout_log_probs" in batch.keys():
            log_probs.copy_(batch["rollout_log_probs"].to(dtype=torch.float32))

        token_logprob_lists = []
        for i, response in enumerate(response_lists):
            valid_len = len(response)
            if valid_len:
                steps = step_map_lists[i][:valid_len]
                step_maps[i, :valid_len] = torch.tensor(steps, dtype=torch.long)
            token_logprob_lists.append(log_probs[i, :valid_len].tolist())

        batch["step_maps"] = step_maps
        batch["log_probs"] = log_probs
        non_tensor_batch["prompt_ids"] = self._object_array(prompt_lists)
        non_tensor_batch["response_ids"] = self._object_array(response_lists)
        non_tensor_batch["step_maps"] = self._object_array(step_map_lists)
        existing_response_uids = non_tensor_batch.get("response_uid")
        if (
            isinstance(existing_response_uids, np.ndarray)
            and len(existing_response_uids) == len(response_lists)
        ):
            response_uids = [
                str(value) if value is not None and str(value) else uuid4().hex
                for value in existing_response_uids.tolist()
            ]
        else:
            response_uids = [uuid4().hex for _ in response_lists]
        non_tensor_batch["response_uid"] = self._object_array(response_uids)
        non_tensor_batch["token_logprobs"] = self._object_array(token_logprob_lists)
        non_tensor_batch["rollout_self_conditioning"] = self._object_array(rollout_self_conditioning)
        non_tensor_batch["sdar_trace_map_status"] = self._object_array(trace_map_status)

    @staticmethod
    def _trace_records_from_sdar_metrics(value: Any) -> list[dict[str, Any]] | None:
        """Extract request-local trace records already returned by SGLang."""
        records: list[dict[str, Any]] = []

        def _visit(item: Any) -> None:
            if isinstance(item, np.ndarray):
                for child in item.reshape(-1).tolist():
                    _visit(child)
                return
            if isinstance(item, (list, tuple)):
                for child in item:
                    _visit(child)
                return
            if not isinstance(item, dict):
                return
            record = item.get("replay_trace_record")
            if isinstance(record, dict):
                records.append(record)

        _visit(value)
        return records or None

    @classmethod
    def attach_deferred_sdar_fields(
        cls,
        config: DictConfig,
        data: DataProto,
        *,
        trace_records: list[dict[str, Any]] | None = None,
    ) -> None:
        helper = cls.__new__(cls)
        helper.config = config
        helper._attach_sdar_diffusion_fields(
            data.batch,
            data.non_tensor_batch,
            trace_records=trace_records,
        )

    def create_transferqueue_client(
        self,
    ):
        """Create a client for data system (TransferQueue)."""
        from verl.single_controller.ray.base import get_random_string
        from verl.utils.transferqueue_utils import create_transferqueue_client

        client_name = get_random_string(length=6)

        self.tq_client = create_transferqueue_client(
            client_id=f"AgentLoopWorker_{client_name}",
            config=self.config.transfer_queue,
        )


async def get_trajectory_info(step, index, validate):
    """Get trajectory info.

    Args:
        step (int): global steps in the trainer.
        index (list): form datastore extra_info.index column.
        validate (bool): whether is a validate step.

    Returns:
        list: trajectory.
    """
    trajectory_info = []
    rollout_n = 0
    for i in range(len(index)):
        if i > 0 and index[i - 1] == index[i]:
            rollout_n += 1
        else:
            rollout_n = 0
        trajectory_info.append({"step": step, "sample_index": index[i], "rollout_n": rollout_n, "validate": validate})
    return trajectory_info


class AgentLoopManager:
    """Agent loop manager that manages a group of agent loop workers."""

    def __init__(
        self,
        config: DictConfig,
        worker_group: RayWorkerGroup = None,
        rollout_resource_pool: RayResourcePool = None,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
    ):
        """Initialize agent loop manager.

        Args:
            config (DictConfig): trainer config.
            worker_group (RayWorkerGroup): ActorRolloutRef worker group for hybrid mode; None for standalone mode.
            rollout_resource_pool (RayResourcePool): Resource pool for actor rollout (Colocate or Standalone mode).
            reward_loop_worker_handles (List[ray.actor.ActorHandle]): Actor handles for streaming reward computation.
        """
        self.config = config
        self.worker_group = worker_group
        self.reward_loop_worker_handles = reward_loop_worker_handles

        # for recipe to change
        if not hasattr(self, "rollout_replica_class"):
            self.rollout_replica_class = get_rollout_replica_class(self.config.actor_rollout_ref.rollout.name)
        if not hasattr(self, "agent_loop_workers_class"):
            self.agent_loop_workers_class = ray.remote(AgentLoopWorker)

        self._initialize_llm_servers(rollout_resource_pool)
        self._init_agent_loop_workers()

    def _initialize_llm_servers(self, rollout_resource_pool: RayResourcePool):
        rollout_world_size = (
            self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
            * self.config.actor_rollout_ref.rollout.data_parallel_size
            * self.config.actor_rollout_ref.rollout.pipeline_model_parallel_size
        )
        if self.worker_group:
            world_size = self.worker_group.world_size
        elif rollout_resource_pool is not None:
            world_size = rollout_resource_pool.world_size
        else:
            world_size = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        num_replicas = world_size // rollout_world_size

        rollout_config = self.config.actor_rollout_ref.rollout
        model_config = self.config.actor_rollout_ref.model
        rollout_gpus_per_node = self.config.trainer.n_gpus_per_node
        if rollout_resource_pool is not None and len(rollout_resource_pool.store) > 0:
            rollout_gpus_per_node = rollout_resource_pool.store[0]
        self.rollout_replicas = [
            self.rollout_replica_class(
                replica_rank=replica_rank,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=rollout_gpus_per_node,
            )
            for replica_rank in range(num_replicas)
        ]

        if self.worker_group and rollout_config.name != "trtllm":
            self._run_all([server.init_hybrid(self.worker_group) for server in self.rollout_replicas])
        elif self.worker_group and rollout_config.name == "trtllm":
            self._run_all(
                [
                    server.init_hybrid_colocated(self.worker_group, rollout_resource_pool)
                    for server in self.rollout_replicas
                ]
            )
        elif rollout_resource_pool is not None:
            if num_replicas > 1:
                rollout_resource_pools = split_resource_pool(
                    rollout_resource_pool,
                    [rollout_world_size] * num_replicas,
                )
            else:
                rollout_resource_pools = [rollout_resource_pool]
            self._run_all(
                [
                    server.init_colocated(replica_resource_pool)
                    for server, replica_resource_pool in zip(
                        self.rollout_replicas, rollout_resource_pools, strict=True
                    )
                ]
            )
        else:
            self._run_all([server.init_standalone() for server in self.rollout_replicas])

        self.server_handles = [server._server_handle for server in self.rollout_replicas]
        self.server_addresses = [server._server_address for server in self.rollout_replicas]

        print(f"AgentLoopManager: {self.server_addresses}")

        # Update Prometheus configuration with server addresses
        if rollout_config.prometheus.enable:
            if rollout_config.disable_log_stats:
                raise ValueError("PROMETHEUS needs disable_log_stats==False, but it is currently True.")
            update_prometheus_config(rollout_config.prometheus, self.server_addresses, rollout_config.name)

    def _init_agent_loop_workers(self):
        self.agent_loop_workers = []
        self.agent_loop_worker_node_ids = []
        self.agent_loop_worker_node_ips = []
        num_workers = self.config.actor_rollout_ref.rollout.agent.num_workers

        alive_nodes = [node for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
        node_ids = [node["NodeID"] for node in alive_nodes]
        node_ip_by_id = {node["NodeID"]: node.get("NodeManagerAddress") for node in alive_nodes}
        for i in range(num_workers):
            # Round-robin scheduling over the all nodes
            node_id = node_ids[i % len(node_ids)]
            node_ip = node_ip_by_id.get(node_id)
            env_vars = {
                key: value
                for key, value in os.environ.items()
                if key.startswith("SGLANG_SDAR_TRACE_")
                or key == "VERL_SDAR_ATTACH_IN_AGENT_WORKER"
            }
            worker_options = {
                "name": f"agent_loop_worker_{i}" + f"_{uuid4().hex[:8]}",
                "scheduling_strategy": ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id, soft=True
                ),
            }
            if env_vars:
                worker_options["runtime_env"] = {"env_vars": env_vars}
            self.agent_loop_workers.append(
                self.agent_loop_workers_class.options(**worker_options).remote(
                    self.config,
                    self.server_handles,
                    self.reward_loop_worker_handles,
                    i,
                )
            )
            self.agent_loop_worker_node_ids.append(node_id)
            self.agent_loop_worker_node_ips.append(node_ip)
        ready_workers = ray.get(
            [worker.ready.remote() for worker in self.agent_loop_workers]
        )
        if len(ready_workers) != len(self.agent_loop_workers):
            raise RuntimeError(
                "Agent loop worker readiness barrier returned the wrong count: "
                f"{len(ready_workers)}/{len(self.agent_loop_workers)}"
            )

    def _rotate_sdar_trace_files(self) -> None:
        if os.environ.get("VERL_SDAR_ROTATE_TRACE_PER_BATCH", "1").strip().lower() in {
            "0",
            "false",
            "no",
            "off",
        }:
            return
        rollout_config = self.config.actor_rollout_ref.rollout
        sglang_config = (rollout_config.get("engine_kwargs", {}) or {}).get(
            "sglang",
            {},
        ) or {}
        if (
            rollout_config.name != "sglang"
            or sglang_config.get("dllm_algorithm") != "SDARPrefixKV"
        ):
            return
        trace_dir = os.environ.get("SGLANG_SDAR_TRACE_DIR")
        if not trace_dir:
            return

        removed_files = 0
        removed_bytes = 0
        trace_path = os.path.join(trace_dir, "rank0_trace.jsonl")
        try:
            removed_bytes += os.path.getsize(trace_path)
        except OSError:
            pass
        os.makedirs(trace_dir, exist_ok=True)
        with open(trace_path, "w", encoding="utf-8"):
            pass

        tensor_dir = os.path.join(trace_dir, "tensors")
        try:
            entries = list(os.scandir(tensor_dir))
        except FileNotFoundError:
            entries = []
        for entry in entries:
            if not entry.is_file() or not entry.name.endswith(".pt"):
                continue
            try:
                removed_bytes += entry.stat().st_size
                os.unlink(entry.path)
                removed_files += 1
            except FileNotFoundError:
                pass
        print(
            "[SDAR_TRACE_ROTATE] "
            f"trace={trace_path} removed_files={removed_files} "
            f"removed_bytes={removed_bytes}",
            flush=True,
        )

    def generate_sequences(self, prompts: DataProto, replay_queue: ray.actor.ActorHandle | None = None) -> DataProto:
        """Split input batch and dispatch to agent loop workers.

        Args:
            prompts (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
        """

        if replay_queue is None:
            self._rotate_sdar_trace_files()

        # Assign per-trace mixed thresholds on the FULL batch (global batch
        # position) before chunking; worker-local positions restart per chunk.
        _assign_mixed_prob_thresholds(
            prompts, self.config.actor_rollout_ref.rollout
        )
        chunkes = prompts.chunk(len(self.agent_loop_workers))
        worker_chunks = [
            (worker_idx, worker, chunk)
            for worker_idx, (worker, chunk) in enumerate(zip(self.agent_loop_workers, chunkes, strict=True))
            if len(chunk) > 0
        ]
        if not worker_chunks:
            raise RuntimeError("AgentLoopManager.generate_sequences received an empty prompt batch")
        def _replay_queue_for_worker(worker_idx: int):
            if isinstance(replay_queue, dict):
                worker_queues = replay_queue.get("worker_queues")
                if isinstance(worker_queues, (list, tuple)) and worker_idx < len(worker_queues):
                    return worker_queues[worker_idx]
                node_queues = replay_queue.get("node_queues")
                worker_node_ids = replay_queue.get("worker_node_ids")
                if (
                    isinstance(node_queues, dict)
                    and isinstance(worker_node_ids, (list, tuple))
                    and worker_idx < len(worker_node_ids)
                ):
                    return node_queues.get(worker_node_ids[worker_idx])
            if isinstance(replay_queue, (list, tuple)):
                if worker_idx < len(replay_queue):
                    return replay_queue[worker_idx]
            return replay_queue

        manager_ray_get_start = time.perf_counter()
        outputs = ray.get(
            [
                worker.generate_sequences.remote(chunk, _replay_queue_for_worker(worker_idx))
                for worker_idx, worker, chunk in worker_chunks
            ]
        )
        manager_ray_get_seconds = time.perf_counter() - manager_ray_get_start
        outputs = [worker_output for worker_output in outputs if len(worker_output) > 0]
        if not outputs:
            raise RuntimeError("AgentLoopManager.generate_sequences received no non-empty worker outputs")
        manager_sdar_attach_start = time.perf_counter()
        deferred_outputs = [
            worker_output
            for worker_output in outputs
            if worker_output.meta_info.pop("sdar_attach_deferred", False)
        ]
        deferred_output_ids = {id(worker_output) for worker_output in deferred_outputs}
        for worker_output in outputs:
            if id(worker_output) in deferred_output_ids:
                AgentLoopWorker.attach_deferred_sdar_fields(
                    self.config,
                    worker_output,
                )
        manager_sdar_attach_seconds = (
            time.perf_counter() - manager_sdar_attach_start
        )
        worker_sdar_attach_seconds = [
            float(worker_output.meta_info.pop("sdar_worker_attach_seconds", 0.0) or 0.0)
            for worker_output in outputs
        ]
        manager_concat_start = time.perf_counter()
        self._align_worker_output_non_tensor_batch_(outputs)
        output = DataProto.concat(outputs)
        manager_concat_seconds = time.perf_counter() - manager_concat_start

        # calculate performance metrics
        metrics = [worker_output.meta_info.pop("metrics") for worker_output in outputs]  # List[List[Dict[str, str]]]
        timing = self._performance_metrics(metrics, output)
        timing.update(
            {
                "manager_ray_get": manager_ray_get_seconds,
                "manager_sdar_attach": manager_sdar_attach_seconds,
                "manager_concat": manager_concat_seconds,
                "agent_worker_sdar_attach_sum": float(sum(worker_sdar_attach_seconds)),
                "agent_worker_sdar_attach_max": float(max(worker_sdar_attach_seconds, default=0.0)),
                "agent_worker_sdar_attach_workers": float(
                    sum(seconds > 0.0 for seconds in worker_sdar_attach_seconds)
                ),
            }
        )

        output.meta_info = {"timing": timing, **outputs[0].meta_info}
        del outputs, deferred_outputs
        return output

    @staticmethod
    def _align_worker_output_non_tensor_batch_(outputs: list[DataProto]) -> None:
        """Make optional agent-loop non-tensor fields stable before DataProto.concat."""
        all_keys = set()
        for output in outputs:
            all_keys.update(output.non_tensor_batch.keys())
        if not all_keys:
            return
        for output in outputs:
            rows = len(output)
            for key in all_keys:
                if key in output.non_tensor_batch:
                    continue
                values = np.empty((rows,), dtype=object)
                values[:] = None
                output.non_tensor_batch[key] = values

    def _performance_metrics(self, metrics: list[list[dict[str, str]]], output: DataProto) -> dict[str, float]:
        timing = {}
        t_generate_sequences = np.array([metric["generate_sequences"] for chunk in metrics for metric in chunk])
        t_tool_calls = np.array([metric["tool_calls"] for chunk in metrics for metric in chunk])
        t_postprocess = np.array([metric.get("postprocess", 0.0) for chunk in metrics for metric in chunk])
        num_preempted = np.array([metric["num_preempted"] for chunk in metrics for metric in chunk])
        timing["agent_loop/num_preempted/min"] = num_preempted.min()
        timing["agent_loop/num_preempted/max"] = num_preempted.max()
        timing["agent_loop/num_preempted/mean"] = num_preempted.mean()
        timing["agent_loop/generate_sequences/min"] = t_generate_sequences.min()
        timing["agent_loop/generate_sequences/max"] = t_generate_sequences.max()
        timing["agent_loop/generate_sequences/mean"] = t_generate_sequences.mean()
        timing["agent_loop/tool_calls/min"] = t_tool_calls.min()
        timing["agent_loop/tool_calls/max"] = t_tool_calls.max()
        timing["agent_loop/tool_calls/mean"] = t_tool_calls.mean()
        timing["agent_loop/postprocess/min"] = t_postprocess.min()
        timing["agent_loop/postprocess/max"] = t_postprocess.max()
        timing["agent_loop/postprocess/mean"] = t_postprocess.mean()

        # batch sequence generation is bounded by the slowest sample
        slowest = np.argmax(t_generate_sequences + t_tool_calls)
        attention_mask = output.batch["attention_mask"][slowest]
        prompt_length = output.batch["prompts"].shape[1]
        timing["agent_loop/slowest/generate_sequences"] = t_generate_sequences[slowest]
        timing["agent_loop/slowest/tool_calls"] = t_tool_calls[slowest]
        timing["agent_loop/slowest/postprocess"] = t_postprocess[slowest]
        timing["agent_loop/slowest/prompt_length"] = attention_mask[:prompt_length].sum().item()
        timing["agent_loop/slowest/response_length"] = attention_mask[prompt_length:].sum().item()
        timing["agent_loop/slowest/num_preempted"] = num_preempted[slowest]

        return timing

    def clear_kv_cache(self):
        """Clear all rollout kv cache, but don`t sleep."""
        self._run_all([replica.clear_kv_cache() for replica in self.rollout_replicas])

    def start_profile(self, **kwargs):
        """Start profiling on all rollout replicas."""
        self._run_all([replica.start_profile(**kwargs) for replica in self.rollout_replicas])

    def stop_profile(self):
        """Stop profiling on all rollout replicas."""
        self._run_all([replica.stop_profile() for replica in self.rollout_replicas])

    def _run_all(self, tasks: list[asyncio.Task]):
        async def run_all():
            await asyncio.gather(*tasks)

        asyncio.run(run_all())
