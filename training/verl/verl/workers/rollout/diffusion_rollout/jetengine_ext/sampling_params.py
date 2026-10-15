from dataclasses import dataclass
from typing import Literal

RemaskStrategy = Literal[
    'sequential',
    'low_confidence_static',
    'low_confidence_dynamic',
    'confidence_threshold_leftmost',
    'entropy_static',
    'entropy_dynamic',
    'entropy_bound',
    'entropy_bounded',
]

@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    # Block Diffusion Parameters
    block_length: int = 4
    denoising_steps: int = 4
    max_steps_per_block: int | None = None
    num_transfer_per_step: int = 0
    dynamic_threshold: float = 0.9
    eb_threshold: float = 0.35
    topk: int = 0
    topp: float = 1
    remasking_strategy: RemaskStrategy = 'low_confidence_static'
    force_accept_leftmost: bool = False
    prefill_full_prompt: bool = False
    noise_mode: str = "mask"
    random_token_high: int | None = None
    deterministic_randtok: bool = False
    randtok_seed: int = 0
    allow_accepted_change: bool = False
    edit_threshold: float | None = None
    exclude_prompt_tail_from_stop: bool = False
    truncate_on_stop: bool = False
    record_replay_trace: bool = False
    replay_trace_full: bool = False
    stop_words: list[int] | None = None
