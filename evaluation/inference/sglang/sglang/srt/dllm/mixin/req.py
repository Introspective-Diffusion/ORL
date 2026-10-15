from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Optional

from sglang.srt.dllm.config import DllmConfig

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class DllmReqPhase(str, enum.Enum):
    STAGING_PREFILL = "staging_prefill"
    STAGING_DECODE = "staging_decode"
    INCOMING_PREFILL = "incoming_prefill"
    INCOMING_DECODE = "incoming_decode"


class ReqDllmMixin:
    def init_diffusion_llm(self: Req, dllm_config: DllmConfig):
        self.dllm_phase: Optional[DllmReqPhase] = None
        self.dllm_ids = []
        self.dllm_block_offset = 0
        self.dllm_denoise_step = 0
        self.dllm_needs_commit = False
        self.dllm_config = dllm_config
        # KV trim support: when set, cache_unfinished_req uses this
        # instead of len(fill_ids) to bound prefix_indices.
        self.dllm_kv_valid_len: Optional[int] = None
        # Variable advance: override how much dllm_block_offset advances
        # in the next _init_fill_ids_for_dllm call (default = block_size).
        self.dllm_next_advance: Optional[int] = None

        if self.dllm_config is not None:
            if self.dllm_prefill_len() == 0:
                self.dllm_phase = DllmReqPhase.INCOMING_DECODE
            else:
                self.dllm_phase = DllmReqPhase.INCOMING_PREFILL

    def dllm_prefill_len(self: Req) -> int:
        """Number of prompt tokens handled by prefill (the rest, if any, is the
        prompt tail that rides inside the first decode block as fixed tokens).

        Relative grid (default): the whole prompt. Absolute grid
        (`prompt_tail_in_first_block`): the whole prompt blocks only, i.e.
        `(len(prompt) // block_size) * block_size`, so every decode block starts
        at a multiple of block_size like LLaDA2's official `generate` and verl's
        training layout. Both modes give 0 for prompts shorter than one block."""
        n = len(self.origin_input_ids)
        block_size = self.dllm_config.block_size
        if n < block_size:
            return 0
        if getattr(self.dllm_config, "prompt_tail_in_first_block", False):
            return (n // block_size) * block_size
        return n

    def is_dllm(self: Req) -> bool:
        return self.dllm_config is not None

    def is_dllm_prefill(self: Req) -> bool:
        return self.dllm_phase in [
            DllmReqPhase.STAGING_PREFILL,
            DllmReqPhase.INCOMING_PREFILL,
        ]

    def determine_dllm_phase(self: Req):
        prefix_length = len(self.prefix_indices)
        min_required_length = prefix_length + self.dllm_config.block_size

        if len(self.fill_ids) < min_required_length:
            return

        input_block = self.fill_ids[prefix_length:min_required_length]
        is_prefill_phase = self.dllm_config.mask_id not in input_block

        if is_prefill_phase:
            self.dllm_phase = DllmReqPhase.STAGING_PREFILL
        else:
            self.dllm_phase = DllmReqPhase.STAGING_DECODE

    def _init_fill_ids_for_dllm(self: Req):
        if not self.dllm_ids:
            self.dllm_ids = (
                self.origin_input_ids
                + [self.dllm_config.mask_id] * self.dllm_config.block_size
            )
        else:
            # Use variable advance if set (for KV trim), else block_size.
            advance = (
                self.dllm_next_advance
                if self.dllm_next_advance is not None
                else self.dllm_config.block_size
            )
            self.dllm_next_advance = None  # reset
            self.dllm_block_offset += advance
            self.dllm_ids += [self.dllm_config.mask_id] * self.dllm_config.block_size

        self.fill_ids = self.dllm_ids
