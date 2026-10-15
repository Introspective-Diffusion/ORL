"""Fast Diffusion Attention — GPU-accelerated attention kernels for diffusion LM training."""

from .diff_attn_noisy import diff_attn_varlen_with_noisy_input, idlm_attn_varlen
from .external_prefix_kv_attn import (
    external_kv_varlen_attention,
    external_kv_varlen_attention_flash,
    external_kv_varlen_attention_reference,
    external_kv_varlen_attention_triton,
)

__all__ = [
    "diff_attn_varlen_with_noisy_input",
    "idlm_attn_varlen",
    "external_kv_varlen_attention",
    "external_kv_varlen_attention_flash",
    "external_kv_varlen_attention_reference",
    "external_kv_varlen_attention_triton",
]
