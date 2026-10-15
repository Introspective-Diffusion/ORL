from __future__ import annotations

import torch


def score_teacher_topk_from_logits(
    teacher_logits: torch.Tensor,
    response_start: int,
    response_len: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract AR teacher top-k log-probs for response tokens.

    `teacher_logits[:, t-1]` predicts token at position `t`. Returned tensors
    are shaped `[batch, response_len, top_k]`.
    """

    if teacher_logits.dim() != 3:
        raise ValueError("teacher_logits must be [batch, seq_len, vocab].")
    if response_start <= 0:
        raise ValueError("response_start must be > 0 for next-token teacher scoring.")
    if response_len <= 0:
        raise ValueError("response_len must be positive.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    start = response_start - 1
    end = start + response_len
    selected_logits = teacher_logits[:, start:end, :]
    teacher_logprobs = selected_logits.float().log_softmax(dim=-1)
    topk_logprobs, topk_ids = teacher_logprobs.topk(k=top_k, dim=-1)
    return topk_ids, topk_logprobs


@torch.no_grad()
def score_teacher_topk(
    teacher_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_start: int,
    response_len: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a frozen AR teacher and extract response-position top-k targets."""

    teacher_model.eval()
    outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    return score_teacher_topk_from_logits(
        outputs.logits,
        response_start=response_start,
        response_len=response_len,
        top_k=top_k,
    )
