"""Import-safe compatibility surface for optional DiffusionGemma GRPO."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GemmaNoStreamGRPOConfig:
    response_length: int = 8192
    block_size: int = 256
    rollout_n: int = 8
    tpf_clip_low: float = 0.25
    tpf_clip_high: float = 4.0
    selector_fraction: float = 0.10
    selector_seed: int = 42
    use_external_prefix_kv: bool = False
    use_incremental_prefix_cache: bool = False


def _gemma_unavailable(*args, **kwargs):
    raise RuntimeError(
        "DiffusionGemma GRPO support was not bootstrapped into this "
        "Qwen-only isolated worktree"
    )


compute_gemma_nostream_grpo_from_replay = _gemma_unavailable
restrict_tensor_batch_to_selected_rows = _gemma_unavailable
