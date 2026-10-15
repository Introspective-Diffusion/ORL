from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch


RemaskStrategy = Literal[
    "sequential",
    "low_confidence_static",
    "low_confidence_dynamic",
    "confidence_threshold_leftmost",
    "entropy_static",
    "entropy_dynamic",
    "entropy_bound",
]


@dataclass
class DenoiseBlockState:
    """Mutable state for one active rollout block."""

    block_ids: torch.Tensor
    step_map: torch.Tensor
    token_logprobs: torch.Tensor
    current_step: int = 0
    previous_hidden: torch.Tensor | None = field(default=None, repr=False)

    @classmethod
    def make(cls, block_size: int, mask_token_id: int, device=None) -> "DenoiseBlockState":
        return cls(
            block_ids=torch.full((block_size,), mask_token_id, dtype=torch.long, device=device),
            step_map=torch.zeros(block_size, dtype=torch.long, device=device),
            token_logprobs=torch.full((block_size,), -float("inf"), dtype=torch.float, device=device),
        )

    @property
    def unresolved_mask(self) -> torch.Tensor:
        return self.step_map == 0

    @property
    def is_finished(self) -> bool:
        return not bool(self.unresolved_mask.any().item())


@dataclass(frozen=True)
class PrefixNoisyBatch:
    """Padded `[prompt | noisy_block]` batch used by HF rollout."""

    input_ids: torch.Tensor
    position_ids: torch.Tensor
    block_positions: torch.Tensor
    prefix_lens: torch.Tensor
    attention_mask: torch.Tensor


def build_prefix_noisy_batch(
    prefix_ids: list[list[int]],
    block_ids: torch.Tensor,
    pad_token_id: int,
    device=None,
    pad_to_multiple: int = 1,
) -> PrefixNoisyBatch:
    """Build a variable-prefix batch for `[prompt | noisy_block]` rollout.

    Blocks are placed immediately after each unpadded prompt. The returned dense
    mask keeps prompt tokens causal and lets noisy block tokens attend to the
    prompt plus the whole current noisy block, while ignoring right padding.
    """

    if block_ids.dim() != 2:
        raise ValueError("block_ids must be shaped [batch, block_len].")
    if len(prefix_ids) != block_ids.size(0):
        raise ValueError("prefix_ids length must match block batch size.")

    batch_size, block_len = block_ids.shape
    device = device if device is not None else block_ids.device
    prefix_lens = torch.tensor([len(ids) for ids in prefix_ids], dtype=torch.long, device=device)
    total_len = int((prefix_lens.max() + block_len).item()) if batch_size > 0 else block_len
    if pad_to_multiple > 1 and total_len % pad_to_multiple:
        total_len = ((total_len + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple

    input_ids = torch.full((batch_size, total_len), pad_token_id, dtype=torch.long, device=device)
    position_ids = torch.zeros((batch_size, total_len), dtype=torch.long, device=device)
    block_positions = torch.empty((batch_size, block_len), dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, total_len, total_len), dtype=torch.bool, device=device)

    for batch_idx, ids in enumerate(prefix_ids):
        prefix_len = len(ids)
        seq_len = prefix_len + block_len
        if prefix_len > 0:
            input_ids[batch_idx, :prefix_len] = torch.tensor(ids, dtype=torch.long, device=device)
            position_ids[batch_idx, :prefix_len] = torch.arange(prefix_len, device=device)
            attention_mask[batch_idx, :prefix_len, :prefix_len] = torch.tril(
                torch.ones(prefix_len, prefix_len, dtype=torch.bool, device=device)
            )

        block_slice = slice(prefix_len, seq_len)
        input_ids[batch_idx, block_slice] = block_ids[batch_idx].to(device=device)
        position_ids[batch_idx, block_slice] = torch.arange(prefix_len, seq_len, device=device)
        block_positions[batch_idx] = torch.arange(prefix_len, seq_len, device=device)
        attention_mask[batch_idx, prefix_len:seq_len, :seq_len] = True

    return PrefixNoisyBatch(
        input_ids=input_ids,
        position_ids=position_ids,
        block_positions=block_positions,
        prefix_lens=prefix_lens,
        attention_mask=attention_mask,
    )


def select_transfer_indices(
    unresolved_mask: torch.Tensor,
    confidence: torch.Tensor,
    num_to_transfer: int,
    strategy: RemaskStrategy = "low_confidence_static",
    dynamic_threshold: float = 0.9,
    force_accept_leftmost: bool = False,
) -> torch.Tensor:
    """Select token positions to commit from a denoise step."""

    if unresolved_mask.shape != confidence.shape:
        raise ValueError("unresolved_mask and confidence must have the same shape.")
    transfer = torch.zeros_like(unresolved_mask, dtype=torch.bool)
    if num_to_transfer <= 0 or not unresolved_mask.any():
        return transfer

    unresolved_indices = torch.nonzero(unresolved_mask, as_tuple=False).flatten()
    num_to_transfer = min(int(num_to_transfer), unresolved_indices.numel())

    if strategy == "sequential":
        selected = unresolved_indices[:num_to_transfer]
    elif strategy == "entropy_bound":
        unresolved_entropy = confidence[unresolved_indices]
        sorted_entropy, sorted_order = torch.sort(unresolved_entropy, descending=False)
        selected_sorted = (torch.cumsum(sorted_entropy, dim=-1) - sorted_entropy) <= dynamic_threshold
        selected = unresolved_indices[sorted_order[selected_sorted]]
    else:
        unresolved_confidence = torch.where(unresolved_mask, confidence, torch.full_like(confidence, -float("inf")))
        if strategy == "confidence_threshold_leftmost":
            dynamic_selected = torch.nonzero(unresolved_confidence > dynamic_threshold, as_tuple=False).flatten()
            if dynamic_selected.numel() > 0:
                selected = dynamic_selected
            else:
                selected = unresolved_indices[:num_to_transfer]
        elif strategy in {"low_confidence_dynamic", "entropy_dynamic"}:
            dynamic_selected = torch.nonzero(unresolved_confidence > dynamic_threshold, as_tuple=False).flatten()
            if dynamic_selected.numel() >= num_to_transfer:
                selected = dynamic_selected
            else:
                selected = torch.topk(unresolved_confidence, k=num_to_transfer).indices
        elif strategy in {"low_confidence_static", "entropy_static"}:
            selected = torch.topk(unresolved_confidence, k=num_to_transfer).indices
        else:
            raise ValueError(f"Unsupported remasking strategy: {strategy}")

    if force_accept_leftmost and selected.numel() == 0:
        selected = unresolved_indices[:1]

    transfer[selected] = True
    return transfer & unresolved_mask


def apply_denoise_step(
    state: DenoiseBlockState,
    sampled_ids: torch.Tensor,
    confidence: torch.Tensor,
    token_logprobs: torch.Tensor,
    num_to_transfer: int,
    strategy: RemaskStrategy = "low_confidence_static",
    dynamic_threshold: float = 0.9,
    next_hidden: torch.Tensor | None = None,
    force_accept_leftmost: bool = False,
) -> torch.Tensor:
    """Commit selected sampled tokens and update rollout trajectory metadata."""

    if sampled_ids.shape != state.block_ids.shape:
        raise ValueError("sampled_ids must match block shape.")
    if confidence.shape != state.block_ids.shape or token_logprobs.shape != state.block_ids.shape:
        raise ValueError("confidence and token_logprobs must match block shape.")

    state.current_step += 1
    transfer = select_transfer_indices(
        state.unresolved_mask,
        confidence,
        num_to_transfer=num_to_transfer,
        strategy=strategy,
        dynamic_threshold=dynamic_threshold,
        force_accept_leftmost=force_accept_leftmost,
    )
    state.block_ids = torch.where(transfer, sampled_ids.to(state.block_ids.dtype), state.block_ids)
    state.step_map = torch.where(
        transfer,
        torch.full_like(state.step_map, state.current_step),
        state.step_map,
    )
    state.token_logprobs = torch.where(transfer, token_logprobs.to(state.token_logprobs.dtype), state.token_logprobs)
    state.previous_hidden = next_hidden.detach() if next_hidden is not None else None
    return transfer
