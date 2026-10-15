from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class OPDRound:
    """One reconstructed OPD training state from a rollout trajectory."""

    step: int
    xt_ids: torch.Tensor
    current_step_mask: torch.Tensor
    unresolved_mask: torch.Tensor
    visible_mask: torch.Tensor


@dataclass(frozen=True)
class OPDExtendedRound:
    """One OPD round in `[prefix | clean_block | noisy_block]` layout."""

    step: int
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    current_step_mask: torch.Tensor
    unresolved_mask: torch.Tensor
    segment_ids: torch.Tensor


def _unique_sorted_steps(step_map: torch.Tensor) -> list[int]:
    valid_steps = step_map[step_map >= 0]
    if valid_steps.numel() == 0:
        return []
    return sorted(int(x) for x in torch.unique(valid_steps).detach().cpu().tolist())


def expand_first_k_rounds(
    x0_ids: torch.Tensor,
    step_map: torch.Tensor,
    mask_token_id: int,
    max_states: int,
) -> list[OPDRound]:
    """Reconstruct the first K denoising training rounds from `x0` and step ids.

    For a selected step `s`, tokens with `step < s` are already committed and
    stay visible, tokens with `step >= s` are masked, and the loss mask is
    exactly `step == s`. Negative step ids are treated as padding/invalid.
    """

    if x0_ids.dim() != 1 or step_map.dim() != 1:
        raise ValueError("x0_ids and step_map must be 1D tensors.")
    if x0_ids.numel() != step_map.numel():
        raise ValueError("x0_ids and step_map must have the same length.")
    rounds: list[OPDRound] = []
    steps = _unique_sorted_steps(step_map)
    if max_states > 0:
        steps = steps[:max_states]
    for step in steps:
        current_step_mask = step_map == step
        if not current_step_mask.any():
            continue

        valid_mask = step_map >= 0
        unresolved_mask = valid_mask & (step_map >= step)
        visible_mask = valid_mask & (step_map < step)
        xt_ids = x0_ids.clone()
        xt_ids[unresolved_mask] = mask_token_id
        rounds.append(
            OPDRound(
                step=step,
                xt_ids=xt_ids,
                current_step_mask=current_step_mask,
                unresolved_mask=unresolved_mask,
                visible_mask=visible_mask,
            )
        )

    return rounds


def build_extended_rounds(
    prefix_ids: torch.Tensor,
    clean_block_ids: torch.Tensor,
    step_map: torch.Tensor,
    mask_token_id: int,
    max_states: int,
    ignore_index: int = -100,
) -> list[OPDExtendedRound]:
    """Build first-K OPD rounds as `[prefix | clean_block | noisy_block]`.

    `labels` are only populated on the noisy segment at current-step positions;
    teacher distillation targets can use the same current-step mask to gather
    teacher logits from the clean block.
    """

    if prefix_ids.dim() != 1 or clean_block_ids.dim() != 1 or step_map.dim() != 1:
        raise ValueError("prefix_ids, clean_block_ids, and step_map must be 1D tensors.")
    if clean_block_ids.numel() != step_map.numel():
        raise ValueError("clean_block_ids and step_map must have the same length.")

    rounds = expand_first_k_rounds(clean_block_ids, step_map, mask_token_id, max_states)
    ext_rounds: list[OPDExtendedRound] = []
    prefix_len = prefix_ids.numel()
    block_len = clean_block_ids.numel()
    device = clean_block_ids.device

    for round_info in rounds:
        input_ids = torch.cat([prefix_ids, clean_block_ids, round_info.xt_ids], dim=0)
        labels = torch.full_like(input_ids, ignore_index)
        noisy_offset = prefix_len + block_len
        labels[noisy_offset : noisy_offset + block_len] = torch.where(
            round_info.current_step_mask,
            clean_block_ids,
            torch.full_like(clean_block_ids, ignore_index),
        )
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        position_ids = torch.cat(
            [
                torch.arange(prefix_len, device=device),
                torch.arange(prefix_len, prefix_len + block_len, device=device),
                torch.arange(prefix_len, prefix_len + block_len, device=device),
            ],
            dim=0,
        )
        segment_ids = torch.cat(
            [
                torch.zeros(prefix_len, device=device, dtype=torch.long),
                torch.ones(block_len, device=device, dtype=torch.long),
                torch.full((block_len,), 2, device=device, dtype=torch.long),
            ],
            dim=0,
        )
        ext_rounds.append(
            OPDExtendedRound(
                step=round_info.step,
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                position_ids=position_ids,
                current_step_mask=round_info.current_step_mask,
                unresolved_mask=round_info.unresolved_mask,
                segment_ids=segment_ids,
            )
        )

    return ext_rounds


def masked_mean_per_state(losses: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Average masked losses per state, then average states equally.

    This prevents states with more selected tokens from dominating states with
    fewer selected tokens.
    """

    if losses.shape != masks.shape:
        raise ValueError("losses and masks must have the same shape.")
    if losses.dim() != 2:
        raise ValueError("losses and masks must be shaped [num_states, seq_len].")

    masks = masks.to(dtype=torch.bool, device=losses.device)
    counts = masks.sum(dim=-1)
    valid_states = counts > 0
    if not valid_states.any():
        return losses.new_zeros(()), losses.new_empty((0,))

    per_state = (losses.masked_fill(~masks, 0).sum(dim=-1)[valid_states] / counts[valid_states].to(losses.dtype))
    return per_state.mean(), per_state
