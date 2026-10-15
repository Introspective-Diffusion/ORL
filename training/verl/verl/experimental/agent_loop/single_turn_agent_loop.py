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
import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.trainer.ppo.gemma_intermediate_replay import (
    GEMMA_REPLAY_FORMAT,
    finalize_gemma_replay_payload,
)
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _normalize_prompt_ids(prompt_ids: Any) -> list[int]:
    def unwrap(value: Any) -> Any:
        while hasattr(value, "keys") and hasattr(value, "__getitem__"):
            for key in ("input_ids", "prompt_ids"):
                if key in value.keys():
                    value = value[key]
                    break
            else:
                return value
        if hasattr(value, "tolist") and not isinstance(value, (list, tuple)):
            value = value.tolist()
        return value

    prompt_ids = unwrap(prompt_ids)
    if isinstance(prompt_ids, tuple):
        prompt_ids = list(prompt_ids)
    elif not isinstance(prompt_ids, list):
        prompt_ids = list(prompt_ids)

    while prompt_ids and isinstance(unwrap(prompt_ids[0]), (list, tuple)):
        if len(prompt_ids) != 1:
            raise ValueError(f"SingleTurnAgentLoop expected one prompt, got batch size {len(prompt_ids)}")
        prompt_ids = list(unwrap(prompt_ids[0]))

    return [int(unwrap(token_id).item() if hasattr(unwrap(token_id), "item") else unwrap(token_id)) for token_id in prompt_ids]


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length

        tool_config_path = self.config.data.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]

    def _gemma_replay_config(self) -> tuple[bool, str]:
        jsd_config = self.config.actor_rollout_ref.actor.get(
            "generation_consistency_jsd", {}
        )
        enabled = (
            bool(jsd_config.get("enable", False))
            and str(jsd_config.get("mode", "replace")) == "replace"
            and str(jsd_config.get("objective", "")) == "gemma_denoise_ce"
        )
        delivery = str(
            jsd_config.get("gemma_replay_delivery", "final_response")
        ).strip().lower()
        if delivery == "rotating_block":
            return False, delivery
        if delivery not in {"final_response", "block_stream"}:
            raise ValueError(
                "actor.generation_consistency_jsd.gemma_replay_delivery must be "
                f"final_response, block_stream, or rotating_block, got {delivery!r}"
            )
        return enabled, delivery

    def _finalize_gemma_replay(
        self,
        payload: Any,
        *,
        request_uid: str,
        sample_uid: str,
        response_uid: str,
        delivery: str,
        actual_response_tokens: int,
    ) -> dict[str, Any]:
        block_size = int(
            self.config.actor_rollout_ref.rollout.get("engine_kwargs", {}).get(
                "block_length",
                self.config.actor_rollout_ref.actor.get("block_length", 256),
            )
            or 256
        )
        expected_blocks = max(
            1,
            (int(actual_response_tokens) + block_size - 1) // block_size,
        )

        def complete(item: Any) -> Any:
            if isinstance(item, list):
                return [complete(value) for value in item]
            if isinstance(item, tuple):
                return tuple(complete(value) for value in item)
            if not isinstance(item, dict):
                return item
            item = dict(item)
            if item.get("format") == GEMMA_REPLAY_FORMAT:
                item.setdefault("request_uid", request_uid)
                item.setdefault("sample_uid", sample_uid)
                item.setdefault("response_uid", response_uid)
                item.setdefault("response_length", int(self.response_length))
                item.setdefault("block_size", block_size)
                item["expected_blocks"] = expected_blocks
                item["response_finished"] = True
                if "blocks" in item and isinstance(
                    item["blocks"], (list, tuple)
                ):
                    item["blocks"] = [
                        complete(block) for block in item["blocks"]
                    ]
            return item

        return finalize_gemma_replay_payload(
            complete(payload),
            delivery=delivery,
        )

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        # 1. extract images and videos from messages
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        # 2. apply chat template and tokenize
        prompt_ids = await self.apply_chat_template(
            messages,
            tools=self.tool_schemas,
            images=images,
            videos=videos,
        )
        prompt_ids = _normalize_prompt_ids(prompt_ids)

        # 3. generate sequences
        metrics = {}
        request_id = uuid4().hex
        sample_uid = str(kwargs.get("uid", kwargs.get("index", request_id)))
        replay_context = {
            "request_uid": request_id,
            "sample_uid": sample_uid,
            "response_uid": request_id,
        }
        if multi_modal_data:
            replay_context["multi_modal_data"] = multi_modal_data
        gemma_replay_enabled, gemma_replay_delivery = (
            self._gemma_replay_config()
        )
        stream_sdar_replay = (
            bool(kwargs.get("_sdar_stream_replay", False))
            and kwargs.get("_sdar_stream_queue") is not None
        )
        if gemma_replay_enabled and gemma_replay_delivery == "final_response":
            stream_sdar_replay = False
        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
                stream_sdar_replay=stream_sdar_replay,
                sdar_stream_queue=kwargs.get("_sdar_stream_queue"),
                sdar_stream_source=kwargs.get("_sdar_stream_source"),
                sdar_stream_context=replay_context,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        if len(output.token_ids) > self.response_length:
            raise RuntimeError(
                "SGLang rollout returned more tokens than configured response_length: "
                f"{len(output.token_ids)} > {self.response_length}. "
                "Refusing to silently truncate generation because this corrupts rollout TPF accounting."
            )
        if output.log_probs is not None and len(output.log_probs) != len(output.token_ids):
            raise RuntimeError(
                "SGLang rollout logprobs are not aligned with output tokens: "
                f"{len(output.log_probs)=}, {len(output.token_ids)=}"
            )
        response_mask = [1] * len(output.token_ids)
        rollout_meta_info = output.meta_info or {}

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=output.token_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
            routed_experts=(
                output.routed_experts[: len(prompt_ids) + self.response_length]
                if output.routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics=metrics,
        )

        # keeping the schema consistent with tool_agent_loop
        output.extra_fields.update({"turn_scores": [], "tool_rewards": []})
        output.extra_fields["sglang_meta_info"] = rollout_meta_info
        output.extra_fields["sglang_sdar_metrics"] = rollout_meta_info.get("sglang_sdar_request_metrics")
        output.extra_fields["sglang_sdar_stats_delta"] = rollout_meta_info.get(
            "sglang_sdar_stats_delta",
            rollout_meta_info.get("sglang_sdar_global_stats_delta"),
        )
        output.extra_fields["sglang_sdar_replay_tensors"] = None
        output.extra_fields["sglang_sdar_stream_chunk"] = None
        output.extra_fields["gemma_intermediate_replay"] = None
        replay_payload = rollout_meta_info.get("gemma_intermediate_replay")
        output.extra_fields["request_uid"] = request_id
        output.extra_fields["sample_uid"] = sample_uid
        output.extra_fields["response_uid"] = request_id
        replay_handle = rollout_meta_info.get("sglang_sdar_replay_handle")
        if isinstance(replay_handle, dict) and not stream_sdar_replay:
            fetched_replay = await self.server_manager.fetch_sdar_replay_tensors(
                request_id,
                replay_handle,
                cleanup=True,
            )
            output.extra_fields["sglang_sdar_replay_tensors"] = fetched_replay
            if replay_payload is None:
                replay_payload = fetched_replay

        response_uid = request_id
        if (
            isinstance(replay_payload, dict)
            and replay_payload.get("response_uid") is not None
        ):
            response_uid = str(replay_payload["response_uid"])
        output.extra_fields["response_uid"] = response_uid
        if gemma_replay_enabled and replay_payload is not None:
            output.extra_fields["gemma_intermediate_replay"] = (
                self._finalize_gemma_replay(
                    replay_payload,
                    request_uid=request_id,
                    sample_uid=sample_uid,
                    response_uid=response_uid,
                    delivery=gemma_replay_delivery,
                    actual_response_tokens=len(output.response_ids),
                )
            )

        return output
