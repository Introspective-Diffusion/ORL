"""Fast Diffusion Attention — GPU-accelerated attention kernels for diffusion LM training."""

from .diff_attn_noisy import diff_attn_varlen_with_noisy_input, idlm_attn_varlen

__all__ = [
    "diff_attn_varlen_with_noisy_input",
    "idlm_attn_varlen",
]
