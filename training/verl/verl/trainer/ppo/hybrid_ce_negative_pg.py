"""Pure helpers for positive trace-CE and negative sampled-action training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class EosAbsorbingTarget:
    """Training-only target obtained by making EOS locally absorbing."""

    token_ids: tuple[int, ...]
    real_token_count: int
    synthetic_eos_count: int
    truncated_post_eos_count: int
    first_eos_index: int | None
    clipped_by_block: bool
    clipped_by_response_limit: bool


@dataclass(frozen=True)
class NegativeActionTargets:
    """Exact sampled-action tensors for the negative-PG branch."""

    action_ids: torch.Tensor
    action_mask: torch.Tensor
    row_rewards: torch.Tensor
    row_advantages: torch.Tensor
    selected_rows: torch.Tensor


@dataclass(frozen=True)
class HybridObjective:
    """Separately normalized positive-CE and negative-PG objectives."""

    total: torch.Tensor
    ce: torch.Tensor
    negative_pg: torch.Tensor


@dataclass(frozen=True)
class TraceCETargets:
    """Positive CE labels for one exact captured denoising state."""

    token_ids: torch.Tensor
    token_mask: torch.Tensor


def build_trace_ce_label_mask(
    active_block_mask: torch.Tensor,
    *,
    token_scope: str,
    action_mask: torch.Tensor | None = None,
    replay_input_ids: torch.Tensor | None = None,
    mask_token_id: int | None = None,
) -> torch.Tensor:
    """Select positive trace-CE labels within one captured block state."""

    if active_block_mask.dim() != 1:
        raise ValueError("active_block_mask must be one-dimensional")
    active_block_mask = active_block_mask.bool()
    token_scope = str(token_scope).strip().lower()
    if token_scope == "active_block_all":
        return active_block_mask
    if token_scope == "masked_input":
        if replay_input_ids is None or mask_token_id is None:
            raise ValueError(
                "masked_input trace CE requires replay_input_ids and "
                "mask_token_id"
            )
        if replay_input_ids.shape != active_block_mask.shape:
            raise ValueError(
                "replay_input_ids must align with active_block_mask"
            )
        return active_block_mask & replay_input_ids.to(
            device=active_block_mask.device
        ).eq(int(mask_token_id))
    if token_scope != "current_action":
        raise ValueError(
            "trace_all_ce token_scope must be active_block_all, "
            f"masked_input, or current_action, got {token_scope!r}"
        )
    if action_mask is None:
        raise ValueError("current_action trace CE requires an exact action_mask")
    if action_mask.shape != active_block_mask.shape:
        raise ValueError("action_mask must align with active_block_mask")
    return active_block_mask & action_mask.to(
        device=active_block_mask.device,
        dtype=torch.bool,
    )


def build_trace_ce_targets(
    final_target_ids: torch.Tensor,
    active_block_mask: torch.Tensor,
    *,
    token_scope: str,
    action_ids: torch.Tensor | None = None,
    action_mask: torch.Tensor | None = None,
    replay_input_ids: torch.Tensor | None = None,
    mask_token_id: int | None = None,
    ignore_id: int = -100,
) -> TraceCETargets:
    """Build either all-block final targets or exact current-action targets."""

    if final_target_ids.dim() != 1:
        raise ValueError("final_target_ids must be one-dimensional")
    if final_target_ids.shape != active_block_mask.shape:
        raise ValueError("final_target_ids must align with active_block_mask")
    active_block_mask = active_block_mask.to(
        device=final_target_ids.device,
        dtype=torch.bool,
    )
    token_scope = str(token_scope).strip().lower()
    token_mask = build_trace_ce_label_mask(
        active_block_mask,
        token_scope=token_scope,
        action_mask=action_mask,
        replay_input_ids=replay_input_ids,
        mask_token_id=mask_token_id,
    )
    if token_scope == "current_action":
        if action_ids is None:
            raise ValueError("current_action trace CE requires exact action_ids")
        if action_ids.shape != final_target_ids.shape:
            raise ValueError("action_ids must align with final_target_ids")
        source_ids = action_ids.to(
            device=final_target_ids.device,
            dtype=final_target_ids.dtype,
        )
    else:
        source_ids = final_target_ids
    if bool((token_mask & source_ids.eq(int(ignore_id))).any()):
        raise ValueError("selected trace CE targets cannot contain ignore_id")
    return TraceCETargets(
        token_ids=torch.where(
            token_mask,
            source_ids,
            torch.full_like(source_ids, int(ignore_id)),
        ),
        token_mask=token_mask,
    )


def build_eos_absorbing_target(
    response_ids: Sequence[int],
    *,
    eos_token_id: int,
    block_size: int,
    block_offset: int = 0,
    max_extra_eos: int = 16,
    max_response_length: int | None = None,
) -> EosAbsorbingTarget:
    """Append trainable EOS targets without crossing the current response block.

    The returned tokens are a training-only copy. Text, reward, correctness, and
    rollout-quality checks must continue to use the original response.
    """

    block_size = int(block_size)
    block_offset = int(block_offset)
    max_extra_eos = int(max_extra_eos)
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if block_offset < 0:
        raise ValueError(
            f"block_offset must be non-negative, got {block_offset}"
        )
    if max_extra_eos < 0:
        raise ValueError(
            f"max_extra_eos must be non-negative, got {max_extra_eos}"
        )
    if max_response_length is not None and int(max_response_length) < 0:
        raise ValueError(
            "max_response_length must be non-negative when provided, "
            f"got {max_response_length}"
        )

    tokens = tuple(int(token_id) for token_id in response_ids)
    try:
        first_eos_index = tokens.index(int(eos_token_id))
    except ValueError:
        return EosAbsorbingTarget(
            token_ids=tokens,
            real_token_count=len(tokens),
            synthetic_eos_count=0,
            truncated_post_eos_count=0,
            first_eos_index=None,
            clipped_by_block=False,
            clipped_by_response_limit=False,
        )

    real_token_count = first_eos_index + 1
    prefix = tokens[:real_token_count]
    truncated_post_eos_count = len(tokens) - real_token_count
    occupied_in_block = (
        block_offset + real_token_count
    ) % block_size
    block_room = block_size - occupied_in_block
    if occupied_in_block == 0:
        block_room = 0

    response_room = max_extra_eos
    if max_response_length is not None:
        response_room = max(0, int(max_response_length) - real_token_count)
    synthetic_eos_count = min(max_extra_eos, block_room, response_room)

    return EosAbsorbingTarget(
        token_ids=prefix + (int(eos_token_id),) * synthetic_eos_count,
        real_token_count=real_token_count,
        synthetic_eos_count=synthetic_eos_count,
        truncated_post_eos_count=truncated_post_eos_count,
        first_eos_index=first_eos_index,
        clipped_by_block=block_room < max_extra_eos,
        clipped_by_response_limit=response_room < min(max_extra_eos, block_room),
    )


def build_negative_action_targets(
    action_ids: torch.Tensor,
    action_mask: torch.Tensor,
    semantic_response_mask: torch.Tensor,
    positive_rows: torch.Tensor,
    *,
    negative_advantage: float = -1.0,
    ignore_id: int = -100,
) -> NegativeActionTargets:
    """Keep only accepted/edited actions from negative response rows."""

    if action_ids.shape != action_mask.shape:
        raise ValueError("action_ids and action_mask must have identical shapes")
    if action_ids.shape != semantic_response_mask.shape:
        raise ValueError(
            "semantic_response_mask must align with sampled action tensors"
        )
    if action_ids.dim() != 2:
        raise ValueError("sampled action tensors must have shape [rows, tokens]")
    if positive_rows.dim() != 1 or positive_rows.shape[0] != action_ids.shape[0]:
        raise ValueError("positive_rows must have one value per replay row")
    negative_advantage = float(negative_advantage)
    if not math.isfinite(negative_advantage) or negative_advantage >= 0:
        raise ValueError("negative_advantage must be finite and negative")

    positive_rows = positive_rows.to(
        device=action_ids.device,
        dtype=torch.bool,
    )
    active_mask = (
        action_mask.to(device=action_ids.device, dtype=torch.bool)
        & semantic_response_mask.to(device=action_ids.device, dtype=torch.bool)
        & positive_rows.logical_not().unsqueeze(-1)
    )
    filtered_ids = torch.where(
        active_mask,
        action_ids,
        torch.full_like(action_ids, int(ignore_id)),
    )
    row_rewards = positive_rows.logical_not().to(dtype=torch.float32)
    row_advantages = torch.where(
        positive_rows,
        torch.zeros_like(row_rewards),
        torch.full_like(row_rewards, negative_advantage),
    )
    return NegativeActionTargets(
        action_ids=filtered_ids,
        action_mask=active_mask,
        row_rewards=row_rewards,
        row_advantages=row_advantages,
        selected_rows=active_mask.any(dim=-1),
    )


def combine_hybrid_objective(
    ce_loss_sum: torch.Tensor,
    negative_pg_sum: torch.Tensor,
    *,
    global_ce_tokens: torch.Tensor | float,
    global_action_tokens: torch.Tensor | float,
    data_parallel_size: int,
    ce_loss_coef: float = 1.0,
    negative_loss_coef: float = 1.0,
) -> HybridObjective:
    """Return branch-wise global token means before FSDP gradient averaging."""

    if ce_loss_sum.numel() != 1 or negative_pg_sum.numel() != 1:
        raise ValueError("hybrid loss sums must be scalar tensors")
    data_parallel_size = int(data_parallel_size)
    if data_parallel_size <= 0:
        raise ValueError("data_parallel_size must be positive")
    ce_loss_coef = float(ce_loss_coef)
    negative_loss_coef = float(negative_loss_coef)
    if (
        not math.isfinite(ce_loss_coef)
        or ce_loss_coef < 0.0
        or not math.isfinite(negative_loss_coef)
        or negative_loss_coef < 0.0
    ):
        raise ValueError("hybrid loss coefficients must be finite and non-negative")

    ce_count = torch.as_tensor(
        global_ce_tokens,
        device=ce_loss_sum.device,
        dtype=ce_loss_sum.dtype,
    )
    action_count = torch.as_tensor(
        global_action_tokens,
        device=negative_pg_sum.device,
        dtype=negative_pg_sum.dtype,
    )
    if (
        ce_count.numel() != 1
        or action_count.numel() != 1
        or not bool(torch.isfinite(ce_count).item())
        or not bool(torch.isfinite(action_count).item())
        or float(ce_count.item()) < 0.0
        or float(action_count.item()) < 0.0
    ):
        raise ValueError("global hybrid token counts must be finite and non-negative")

    ce_objective = ce_loss_sum * 0.0
    if float(ce_count.item()) > 0.0:
        ce_objective = ce_loss_sum * data_parallel_size / ce_count
    negative_objective = negative_pg_sum * 0.0
    if float(action_count.item()) > 0.0:
        negative_objective = (
            negative_pg_sum * data_parallel_size / action_count
        )
    return HybridObjective(
        total=(
            ce_loss_coef * ce_objective
            + negative_loss_coef * negative_objective
        ),
        ce=ce_objective,
        negative_pg=negative_objective,
    )


def compute_chunked_action_log_probs(
    replay_sources: torch.Tensor,
    action_labels: torch.Tensor,
    *,
    temperature: float,
    chunk_size: int,
    projection_weight: torch.Tensor | None = None,
    projection_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather exact action log-probabilities from logits or replay hidden rows."""

    if replay_sources.dim() != 2:
        raise ValueError("replay_sources must have shape [actions, features]")
    if action_labels.dim() != 1 or action_labels.shape[0] != replay_sources.shape[0]:
        raise ValueError("action_labels must have one value per replay source")
    temperature = float(temperature)
    chunk_size = int(chunk_size)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if projection_weight is not None:
        if projection_weight.dim() != 2:
            raise ValueError("projection_weight must be rank two")
        if replay_sources.shape[-1] != projection_weight.shape[-1]:
            raise ValueError(
                "replay hidden width does not match projection weight"
            )
    elif projection_bias is not None:
        raise ValueError("projection_bias requires projection_weight")

    gathered_chunks = []
    for start in range(0, int(replay_sources.shape[0]), chunk_size):
        end = start + chunk_size
        source_chunk = replay_sources[start:end]
        if projection_weight is not None:
            logits = F.linear(
                source_chunk.to(dtype=projection_weight.dtype),
                projection_weight,
                projection_bias,
            )
        else:
            logits = source_chunk
        logits = logits.float().div(temperature)
        logits = torch.where(
            torch.isfinite(logits),
            logits,
            torch.zeros_like(logits).detach(),
        )
        labels = action_labels[start:end]
        gathered_chunks.append(
            F.log_softmax(logits, dim=-1)
            .gather(dim=-1, index=labels.unsqueeze(-1))
            .squeeze(-1)
        )
    if not gathered_chunks:
        return replay_sources.new_empty((0,), dtype=torch.float32)
    return torch.cat(gathered_chunks, dim=0)


__all__ = [
    "EosAbsorbingTarget",
    "HybridObjective",
    "NegativeActionTargets",
    "TraceCETargets",
    "build_eos_absorbing_target",
    "build_negative_action_targets",
    "build_trace_ce_label_mask",
    "build_trace_ce_targets",
    "combine_hybrid_objective",
    "compute_chunked_action_log_probs",
]
