import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.5
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    mask_token_id: int = -1
    block_length: int = 4
    use_self_conditioning: bool = False
    self_conditioning_residual_scale: float | None = None
    self_conditioning_post_norm: bool | None = None
    self_conditioning_post_norm_scale: float | None = None
    self_conditioning_mask_mode: str = "mask_token"
    record_self_conditioning_debug: bool = False
    prefill_causal: bool = True
    commit_causal: bool = False
    distributed_init_method: str | None = None

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
        if self.self_conditioning_residual_scale is not None:
            self.hf_config.self_conditioning_residual_scale = float(self.self_conditioning_residual_scale)
        if self.self_conditioning_post_norm is not None:
            self.hf_config.self_conditioning_post_norm = bool(self.self_conditioning_post_norm)
        if self.self_conditioning_post_norm_scale is not None:
            self.hf_config.self_conditioning_post_norm_scale = float(self.self_conditioning_post_norm_scale)
        if self.self_conditioning_mask_mode not in {"mask_token", "nonprompt"}:
            raise ValueError(
                "self_conditioning_mask_mode must be 'mask_token' or 'nonprompt'"
            )
        self.hf_config.record_self_conditioning_debug = bool(self.record_self_conditioning_debug)
        self.use_self_conditioning = bool(
            self.use_self_conditioning
            or float(getattr(self.hf_config, "self_conditioning_prob", 0.0)) > 0.0
        )
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
        assert self.mask_token_id != -1, "Mask token ID must be set"
