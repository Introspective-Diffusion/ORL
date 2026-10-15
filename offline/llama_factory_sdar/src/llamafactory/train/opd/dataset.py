from __future__ import annotations

from typing import Any

import torch

from .trajectory import build_extended_rounds


def _as_long_tensor(values: Any) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.long) if not torch.is_tensor(values) else values.to(dtype=torch.long)


def _as_float_tensor(values: Any) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float) if not torch.is_tensor(values) else values.to(dtype=torch.float)


def trajectory_record_to_examples(
    record: dict[str, Any],
    mask_token_id: int,
    max_states: int,
    ignore_index: int = -100,
) -> list[dict[str, Any]]:
    """Expand one compact OPD trajectory record into training examples.

    Required fields:
      - prompt_input_ids or prompt_ids
      - response_ids or final_response_ids
      - step_map
      - teacher_topk_ids
      - teacher_topk_logprobs

    Teacher top-k tensors are optional. When present they are stored block-local
    for the clean response and attached unchanged to each round. When absent,
    the trainer can use clean_student self-consistency targets.
    """

    prompt_ids = _as_long_tensor(record.get("prompt_input_ids", record.get("prompt_ids", [])))
    response_ids = _as_long_tensor(record.get("response_ids", record.get("final_response_ids", [])))
    step_map = _as_long_tensor(record["step_map"])
    has_teacher_topk = "teacher_topk_ids" in record and "teacher_topk_logprobs" in record
    teacher_topk_ids = _as_long_tensor(record["teacher_topk_ids"]) if has_teacher_topk else None
    teacher_topk_logprobs = _as_float_tensor(record["teacher_topk_logprobs"]) if has_teacher_topk else None

    if has_teacher_topk:
        if teacher_topk_ids.shape[:1] != response_ids.shape[:1]:
            raise ValueError("teacher_topk_ids must have one top-k row per response token.")
        if teacher_topk_ids.shape != teacher_topk_logprobs.shape:
            raise ValueError("teacher_topk_ids and teacher_topk_logprobs must have the same shape.")

    examples: list[dict[str, Any]] = []
    rollout_block_size = int(record.get("block_size", response_ids.numel()))
    if rollout_block_size <= 0:
        raise ValueError("block_size must be positive when present.")

    for block_start in range(0, response_ids.numel(), rollout_block_size):
        block_end = min(block_start + rollout_block_size, response_ids.numel())
        block_response_ids = response_ids[block_start:block_end]
        block_step_map = step_map[block_start:block_end]
        block_teacher_topk_ids = teacher_topk_ids[block_start:block_end] if has_teacher_topk else None
        block_teacher_topk_logprobs = teacher_topk_logprobs[block_start:block_end] if has_teacher_topk else None
        block_prefix_ids = torch.cat([prompt_ids, response_ids[:block_start]], dim=0)

        rounds = build_extended_rounds(
            block_prefix_ids,
            block_response_ids,
            block_step_map,
            mask_token_id=mask_token_id,
            max_states=max_states,
            ignore_index=ignore_index,
        )
        prefix_len = block_prefix_ids.numel()
        block_len = block_response_ids.numel()
        for round_info in rounds:
            example = {
                "input_ids": round_info.input_ids.tolist(),
                "labels": round_info.labels.tolist(),
                "attention_mask": round_info.attention_mask.long().tolist(),
                "position_ids": round_info.position_ids.tolist(),
                "segment_ids": round_info.segment_ids.tolist(),
                "current_step_mask": round_info.current_step_mask.long().tolist(),
                "unresolved_mask": round_info.unresolved_mask.long().tolist(),
                "step_map": block_step_map.tolist(),
                "prefix_len": prefix_len,
                "block_len": block_len,
                "block_index": block_start // rollout_block_size,
                "step": round_info.step,
            }
            if has_teacher_topk:
                example["teacher_topk_ids"] = block_teacher_topk_ids.tolist()
                example["teacher_topk_logprobs"] = block_teacher_topk_logprobs.tolist()
            examples.append(example)

    return examples


def trajectory_records_to_examples(
    records: list[dict[str, Any]],
    mask_token_id: int,
    max_states: int,
    ignore_index: int = -100,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for record in records:
        examples.extend(
            trajectory_record_to_examples(
                record,
                mask_token_id=mask_token_id,
                max_states=max_states,
                ignore_index=ignore_index,
            )
        )
    return examples
