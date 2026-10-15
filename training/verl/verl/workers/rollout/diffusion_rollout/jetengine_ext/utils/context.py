from dataclasses import dataclass, field
from typing import List
import torch

from jetengine_ext.engine.sequence import RunType
from jetengine_ext.engine.sequence import Sequence

@dataclass
class Context:
    run_type: RunType | None = None
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    is_last_denoise_step: List[bool] = field(default_factory=lambda: [False])
    block_length: int = 4
    seqs: list[Sequence] | None = None
    use_self_conditioning: bool = False
    self_conditioning_mask_mode: str = "mask_token"
    prefill_causal: bool = True
    commit_causal: bool = False

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(run_type, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, is_last_denoise_step=[False], block_length=4, seqs=None, use_self_conditioning=False, self_conditioning_mask_mode="mask_token", prefill_causal=True, commit_causal=False):
    global _CONTEXT
    _CONTEXT = Context(run_type, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables, is_last_denoise_step, block_length, seqs, use_self_conditioning, self_conditioning_mask_mode, prefill_causal, commit_causal)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
