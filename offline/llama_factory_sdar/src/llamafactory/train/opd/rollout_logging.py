from __future__ import annotations

from typing import Any, Iterable


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def _decode(tokenizer: Any, token_ids: Iterable[int], max_chars: int) -> str:
    text = tokenizer.decode(list(token_ids), skip_special_tokens=False)
    return _truncate_text(text, max_chars)


def build_rollout_sample_rows(
    records: list[dict[str, Any]],
    tokenizer: Any,
    max_samples: int = 8,
    max_states_per_sample: int = 4,
    max_chars: int = 2048,
) -> list[dict[str, Any]]:
    """Build human-readable rows for W&B rollout sample logging.

    Expected record keys are intentionally simple so rollout backends can use
    this helper without depending on a specific dataset class:
      - prompt_input_ids: list[int]
      - response_ids: list[int]
      - states: list[dict] with xt_ids, step, loss_mask/current_step_mask
    """

    rows: list[dict[str, Any]] = []
    for sample_idx, record in enumerate(records[:max_samples]):
        prompt_ids = record.get("prompt_input_ids") or record.get("prompt_ids") or []
        response_ids = record.get("response_ids") or record.get("final_response_ids") or []
        prompt_text = _decode(tokenizer, prompt_ids, max_chars)
        response_text = _decode(tokenizer, response_ids, max_chars)
        states = record.get("states", [])

        for state_idx, state in enumerate(states[:max_states_per_sample]):
            xt_ids = state.get("xt_ids", [])
            loss_mask = state.get("loss_mask", state.get("current_step_mask", []))
            unresolved_mask = state.get("unresolved_mask", [])
            loss_positions = [i for i, flag in enumerate(loss_mask) if flag]
            rows.append(
                {
                    "sample_idx": sample_idx,
                    "state_idx": state_idx,
                    "step": state.get("step", state_idx),
                    "prompt": prompt_text,
                    "final_response": response_text,
                    "xt_state": _decode(tokenizer, xt_ids, max_chars),
                    "loss_positions": ",".join(map(str, loss_positions[:128])),
                    "num_loss_tokens": len(loss_positions),
                    "num_unresolved_tokens": sum(1 for flag in unresolved_mask if flag),
                }
            )

        if not states:
            rows.append(
                {
                    "sample_idx": sample_idx,
                    "state_idx": -1,
                    "step": -1,
                    "prompt": prompt_text,
                    "final_response": response_text,
                    "xt_state": "",
                    "loss_positions": "",
                    "num_loss_tokens": 0,
                    "num_unresolved_tokens": 0,
                }
            )

    return rows


def log_rollout_samples_to_wandb(
    records: list[dict[str, Any]],
    tokenizer: Any,
    step: int | None = None,
    table_name: str = "rollout/samples",
    max_samples: int = 8,
    max_states_per_sample: int = 4,
    max_chars: int = 2048,
) -> bool:
    """Log plaintext rollout samples to W&B if available.

    Returns True when a table is logged and False when W&B is unavailable or
    there is nothing to log.
    """

    rows = build_rollout_sample_rows(
        records,
        tokenizer,
        max_samples=max_samples,
        max_states_per_sample=max_states_per_sample,
        max_chars=max_chars,
    )
    if not rows:
        return False

    try:
        import wandb  # type: ignore
    except Exception:
        return False

    if wandb.run is None:
        return False

    columns = [
        "sample_idx",
        "state_idx",
        "step",
        "prompt",
        "final_response",
        "xt_state",
        "loss_positions",
        "num_loss_tokens",
        "num_unresolved_tokens",
    ]
    table = wandb.Table(columns=columns)
    for row in rows:
        table.add_data(*(row[column] for column in columns))

    payload = {table_name: table}
    if step is None:
        wandb.log(payload)
    else:
        wandb.log(payload, step=step)
    return True
