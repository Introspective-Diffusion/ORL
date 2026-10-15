import torch
import torch.distributed as dist
import triton
import triton.language as tl
import math
import torch.nn.functional as F
from jetengine_ext.engine.sequence import RunType
from jetengine_ext.layers.activation import SiluAndMul
from jetengine_ext.layers.attention import BlockAttention
from jetengine_ext.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from jetengine_ext.layers.layernorm import RMSNorm
from jetengine_ext.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from jetengine_ext.layers.rotary_embedding import get_rope
from jetengine_ext.utils.context import get_context
from torch import nn


class SDARAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = BlockAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.view(-1, self.num_heads, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(-1, self.num_kv_heads, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o)
        return output


class SDARMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class SDARSelfConditioning(nn.Module):
    """Gated residual adapter for previous denoise-step soft embeddings."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = getattr(config, "self_conditioning_intermediate_size", config.intermediate_size)
        self.pre_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, inputs_embeds: torch.Tensor, signal: torch.Tensor) -> torch.Tensor:
        normed = self.pre_norm(signal)
        gated = F.silu(self.gate_proj(normed)) * self.up_proj(normed)
        residual_scale = float(getattr(self.config, "self_conditioning_residual_scale", 1.0))
        combined = inputs_embeds + residual_scale * self.down_proj(gated)
        if bool(getattr(self.config, "self_conditioning_post_norm", False)):
            post_norm_scale = float(getattr(self.config, "self_conditioning_post_norm_scale", 1.0))
            return F.rms_norm(combined, (combined.size(-1),), eps=self.pre_norm.eps) * post_norm_scale
        return combined


class SDARDecoderLayer(nn.Module):

    def __init__(
        self,
        config
    ) -> None:
        super().__init__()
        self.self_attn = SDARAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', False),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=None
            if isinstance(getattr(config, "rope_scaling", None), dict)
            else getattr(config, "rope_scaling", None),
        )
        self.mlp = SDARMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

@triton.jit
def gather_residual_hidden_states_2d_kernel(
    hs_ptr, hs_s0, hs_s1, hs_s2,
    context_lens_ptr,
    block_tables_ptr, bt_s0, bt_s1,
    out_ptr, out_s0, out_s1,
    hidden_size: tl.constexpr,
    block_size: tl.constexpr,
    block_length: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid_tok = tl.program_id(0)
    pid_hb = tl.program_id(1)

    # index of the sequence
    s = pid_tok // block_length

    # index of the token within the block
    t = pid_tok - s * block_length

    # KV cache length
    ctx = tl.load(context_lens_ptr + s).to(tl.int32)

    # position of the current token in its sequence
    pos = ctx + t

    # virtual page
    logical_block = pos // block_size

    # page offset
    off = pos - logical_block * block_size

    # physical page
    phys = tl.load(block_tables_ptr + s * bt_s0 + logical_block * bt_s1).to(tl.int32)
    tl.device_assert(phys >= 0, "phys out of bounds")

    h_offs = pid_hb * BLOCK_V + tl.arange(0, BLOCK_V)
    h_mask = h_offs < hidden_size
    valid = h_mask & (phys >= 0)

    # ---- IMPORTANT: do address arithmetic in int64 ----
    phys_i64 = phys.to(tl.int64)
    off_i64 = off.to(tl.int64)
    h_i64 = h_offs.to(tl.int64)

    hs_s0_i64 = tl.full((), hs_s0, tl.int64)
    hs_s1_i64 = tl.full((), hs_s1, tl.int64)
    hs_s2_i64 = tl.full((), hs_s2, tl.int64)

    # Avoid negative address math when phys < 0
    phys_safe = tl.maximum(phys_i64, 0)

    hs_offs = phys_safe * hs_s0_i64 + off_i64 * hs_s1_i64 + h_i64 * hs_s2_i64

    x = tl.load(hs_ptr + hs_offs, mask=valid, other=0.0)

    out_offs = pid_tok * out_s0 + h_offs * out_s1
    tl.store(out_ptr + out_offs, x, mask=h_mask)


@triton.jit
def gather_residual_valid_1d_kernel(
    valid_ptr, v_s0, v_s1,
    context_lens_ptr,
    block_tables_ptr, bt_s0, bt_s1,
    out_ptr, out_s0,
    block_size: tl.constexpr,
    block_length: tl.constexpr,
):
    pid_tok = tl.program_id(0)

    # index of the sequence
    s = pid_tok // block_length

    # index of the token within the block
    t = pid_tok - s * block_length

    # KV cache length
    ctx = tl.load(context_lens_ptr + s).to(tl.int32)

    # position of the current token in its sequence
    pos = ctx + t

    # virtual page
    logical_block = pos // block_size

    # page offset
    off = pos - logical_block * block_size

    # physical page
    phys = tl.load(block_tables_ptr + s * bt_s0 + logical_block * bt_s1).to(tl.int32)
    tl.device_assert(phys >= 0, "phys out of bounds")

    # ---- address arithmetic in int64 ----
    phys_i64 = phys.to(tl.int64)
    off_i64 = off.to(tl.int64)
    v_s0_i64 = tl.full((), v_s0, tl.int64)
    v_s1_i64 = tl.full((), v_s1, tl.int64)

    phys_safe = tl.maximum(phys_i64, 0)
    v_offs = phys_safe * v_s0_i64 + off_i64 * v_s1_i64

    x = tl.load(valid_ptr + v_offs, mask=(phys >= 0), other=0).to(tl.int1)

    tl.store(out_ptr + pid_tok * out_s0, x)

def gather_residual_hidden_states(
    residual_hidden_states,
    valid,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    block_length: int,
    BLOCK_V: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Faster 2D-parallel gather + TP-correct normalized entropy over global vocab.
    """
    assert context_lens.is_cuda and block_tables.is_cuda
    assert context_lens.dtype == torch.int32
    assert block_tables.dtype == torch.int32

    _, block_size, _ = residual_hidden_states.shape
    bs = context_lens.numel()
    n_tok = bs * block_length

    hidden_size = residual_hidden_states.shape[-1]
    residual_hidden_states_slice =  torch.empty((n_tok, hidden_size), device=residual_hidden_states.device, dtype=residual_hidden_states.dtype)
    valid_slice = torch.empty((n_tok,), device=residual_hidden_states.device, dtype=torch.bool)

    # Simple heuristic: bigger BLOCK_V -> more warps
    num_warps = 8 if BLOCK_V >= 1024 else 4

    gather_residual_hidden_states_2d_kernel[(n_tok, triton.cdiv(hidden_size, BLOCK_V))](
        residual_hidden_states, residual_hidden_states.stride(0), residual_hidden_states.stride(1), residual_hidden_states.stride(2),
        context_lens,
        block_tables, block_tables.stride(0), block_tables.stride(1),
        residual_hidden_states_slice, residual_hidden_states_slice.stride(0), residual_hidden_states_slice.stride(1),
        hidden_size=hidden_size,
        block_size=block_size,
        block_length=block_length,
        BLOCK_V=BLOCK_V,
        num_warps=num_warps,
    )

    gather_residual_valid_1d_kernel[(n_tok,)](
        valid, valid.stride(0), valid.stride(1),
        context_lens,
        block_tables, block_tables.stride(0), block_tables.stride(1),
        valid_slice, valid_slice.stride(0),
        block_size=block_size,
        block_length=block_length,
        num_warps=4,
    )

    return residual_hidden_states_slice, valid_slice



@triton.jit
def _store_to_paged_residual_hidden_states_kernel(
    rhs_ptr, rhs_s0, rhs_s1, rhs_s2,
    context_lens_ptr,
    block_tables_ptr, bt_s0, bt_s1,
    hs_ptr, hs_s0, hs_s1,
    hidden_size: tl.constexpr,
    block_size: tl.constexpr,
    block_length: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid_tok = tl.program_id(0)
    pid_hb = tl.program_id(1)

    # index of the sequence
    s = pid_tok // block_length
    # index of the token within the block
    t = pid_tok - s * block_length

    # KV cache length
    ctx = tl.load(context_lens_ptr + s).to(tl.int32)

    # position of the current token in its sequence
    pos = ctx + t

    # virtual page
    logical_block = pos // block_size

    # page offset
    off = pos - logical_block * block_size

    # physical page
    phys = tl.load(block_tables_ptr + s * bt_s0 + logical_block * bt_s1).to(tl.int32)
    tl.device_assert(phys >= 0, "phys out of bounds")

    h_offs = pid_hb * BLOCK_V + tl.arange(0, BLOCK_V)
    h_mask = h_offs < hidden_size
    valid = h_mask & (phys >= 0)

    # ---- load from contiguous hidden_states[n_tok, hidden_size] ----
    hs_offs = pid_tok * hs_s0 + h_offs * hs_s1
    x = tl.load(hs_ptr + hs_offs, mask=h_mask, other=0.0)

    # ---- store into paged residual_hidden_states[phys, off, hidden] ----
    phys_i64 = phys.to(tl.int64)
    off_i64 = off.to(tl.int64)
    h_i64 = h_offs.to(tl.int64)

    rhs_s0_i64 = tl.full((), rhs_s0, tl.int64)
    rhs_s1_i64 = tl.full((), rhs_s1, tl.int64)
    rhs_s2_i64 = tl.full((), rhs_s2, tl.int64)

    phys_safe = tl.maximum(phys_i64, 0)
    rhs_offs = phys_safe * rhs_s0_i64 + off_i64 * rhs_s1_i64 + h_i64 * rhs_s2_i64

    tl.store(rhs_ptr + rhs_offs, x, mask=valid)


@triton.jit
def _store_to_paged_valid_kernel(
    valid_ptr, v_s0, v_s1,
    context_lens_ptr,
    block_tables_ptr, bt_s0, bt_s1,
    block_size: tl.constexpr,
    block_length: tl.constexpr,
):
    pid_tok = tl.program_id(0)

    # index of the sequence
    s = pid_tok // block_length

    # index of the token within the block
    t = pid_tok - s * block_length

    # KV cache length
    ctx = tl.load(context_lens_ptr + s).to(tl.int32)

    # position of the current token in its sequence
    pos = ctx + t

    # virtual page
    logical_block = pos // block_size

    # page offset
    off = pos - logical_block * block_size

    # physical page
    phys = tl.load(block_tables_ptr + s * bt_s0 + logical_block * bt_s1).to(tl.int32)
    tl.device_assert(phys >= 0, "phys out of bounds")

    # ---- store True into valid[phys, off] ----
    phys_i64 = phys.to(tl.int64)
    off_i64 = off.to(tl.int64)
    v_s0_i64 = tl.full((), v_s0, tl.int64)
    v_s1_i64 = tl.full((), v_s1, tl.int64)

    phys_safe = tl.maximum(phys_i64, 0)
    v_offs = phys_safe * v_s0_i64 + off_i64 * v_s1_i64

    one = tl.full((), 1, tl.int1)
    tl.store(valid_ptr + v_offs, one, mask=(phys >= 0))




def store_residual_hidden_states(
    residual_hidden_states,
    valid,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    block_length: int,
    hidden_states = None,
    *,
    BLOCK_V: int = 1024,
) -> None:
    """
    High-performance softmax(logits) + store into paged 'p'.

    Design:
      - 2D grid (token, vocab_block)
      - 3 passes: block-max -> row-max, block-sum -> row-sum, normalize+store
    """
    assert context_lens.is_cuda and block_tables.is_cuda
    assert context_lens.dtype == torch.int32
    assert block_tables.dtype == torch.int32

    _, block_size, _ = residual_hidden_states.shape
    bs = context_lens.numel()
    n_tok = bs * block_length

    hidden_size = residual_hidden_states.shape[-1]

    num_warps = 8 if BLOCK_V >= 1024 else 4

    _store_to_paged_residual_hidden_states_kernel[(n_tok, triton.cdiv(hidden_size, BLOCK_V))](
        residual_hidden_states, residual_hidden_states.stride(0), residual_hidden_states.stride(1), residual_hidden_states.stride(2),
        context_lens,
        block_tables, block_tables.stride(0), block_tables.stride(1),
        hidden_states, hidden_states.stride(0), hidden_states.stride(1),
        hidden_size=hidden_size,
        block_size=block_size,
        block_length=block_length,
        BLOCK_V=BLOCK_V,
        num_warps=num_warps,
    )

    _store_to_paged_valid_kernel[(n_tok,)](
        valid, valid.stride(0), valid.stride(1),
        context_lens,
        block_tables, block_tables.stride(0), block_tables.stride(1),
        block_size=block_size,
        block_length=block_length,
        num_warps=4,
    )


def normalized_entropy(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Compute normalized Shannon entropy for each row of a probability tensor.

    Args:
        p: Tensor of shape (N, vocab_len), rows should sum to 1 (e.g., softmax output)
        eps: small value to avoid log(0); also clips p for numerical stability

    Returns:
        entropy_norm: Tensor of shape (N,), values in [0, 1]
                      1 → perfectly uniform, 0 → fully concentrated (e.g., one-hot)
    """
    # Ensure numerical stability: clamp to [eps, 1]
    p = torch.clamp(p, min=eps)

    # Optional: renormalize in case input isn't exactly sum-to-1
    p = p / p.sum(dim=-1, keepdim=True)

    vocab_len = p.size(1)
    if vocab_len == -1:
        return torch.ones(p.size(0), device=p.device, dtype=p.dtype)

    # Compute Shannon entropy: H = -sum(p * log p)
    # Use log base e (natural log); any base works since we normalize
    entropy = -(p * torch.log(p)).sum(dim=-1)  # (N,)

    # Normalize by max possible entropy: log(vocab_len)
    max_entropy = math.log(float(vocab_len))
    entropy_norm = entropy / max_entropy

    # Clip to [0, 1] due to floating errors
    return torch.clamp(entropy_norm, 0.0, 1.0)


class SDARModel(nn.Module):

    def __init__(
        self,
        config,
        mask_token_id
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([SDARDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # self.p = torch.tensor([])
        self.residual_hidden_states = torch.tensor([])
        self.valid = torch.tensor([])
        self.mask_token_id = mask_token_id

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = input_embeds if input_embeds is not None else self.embed_tokens(input_ids)

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class SDARForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config,
        mask_token_id=-1
    ) -> None:
        super().__init__()
        self.config = config
        self.model = SDARModel(config, mask_token_id)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.mask_token_id = mask_token_id
        self.use_self_conditioning = float(getattr(config, "self_conditioning_prob", 0.0)) > 0.0
        self.self_conditioning = SDARSelfConditioning(config) if self.use_self_conditioning else None
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def build_self_conditioned_input_embeds(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor | None:
        context = get_context()
        if not (
            context.use_self_conditioning
            and self.use_self_conditioning
            and self.self_conditioning is not None
        ):
            return None

        inputs_embeds = self.model.embed_tokens(input_ids)
        signal = torch.zeros_like(inputs_embeds)
        if context.run_type == RunType.DENOISE:
            residual_hidden_states_slice, valid_slice = gather_residual_hidden_states(
                self.model.residual_hidden_states,
                self.model.valid,
                context.context_lens,
                context.block_tables,
                block_length=context.block_length,
            )
            bs = context.context_lens.numel()
            assert context.block_length * bs == input_ids.numel()
            prompt_mask = torch.zeros_like(valid_slice, dtype=torch.bool)

            if context.commit_causal:
                # The clean commit forward should use only the zero-signal
                # post-norm path, not previous denoise-step refinement signal.
                selected = torch.zeros_like(valid_slice, dtype=torch.bool)
            elif context.self_conditioning_mask_mode == "nonprompt":
                selected = valid_slice.clone()
                if context.seqs is not None:
                    assert len(context.seqs) == bs
                    assert len(context.seqs) == len(set(context.seqs))
                    for i, seq in enumerate(context.seqs):
                        start = i * context.block_length
                        end = start + context.block_length
                        block_prompt_mask = getattr(seq, "block_prompt_mask", None)
                        if block_prompt_mask:
                            prompt_mask[start:end] = torch.tensor(
                                block_prompt_mask[:context.block_length],
                                device=selected.device,
                                dtype=torch.bool,
                            )
                    selected = selected & ~prompt_mask
            else:
                selected = valid_slice & input_ids.eq(self.mask_token_id)

            if bool(selected.any()):
                assert self.model.embed_tokens.tp_size == 1
                logits_slice = residual_hidden_states_slice @ self.lm_head.weight.t()
                probs = logits_slice.float().softmax(dim=-1).to(logits_slice.dtype)
                soft_embeddings = probs @ self.model.embed_tokens.weight
                signal = torch.where(selected.unsqueeze(-1), soft_embeddings, signal)

            if context.seqs is not None:
                assert len(context.seqs) == bs
                assert len(context.seqs) == len(set(context.seqs))
                record_debug = bool(getattr(self.config, "record_self_conditioning_debug", False))
                for i, seq in enumerate(context.seqs):
                    start = i * context.block_length
                    end = start + context.block_length
                    if selected[start:end].any():
                        item = {
                            "position": (context.context_lens[i].item(), context.context_lens[i].item() + context.block_length),
                            "valid": (
                                selected[start:end]
                                if context.self_conditioning_mask_mode == "nonprompt"
                                else valid_slice[start:end]
                            ).cpu(),
                            "prompt_mask": prompt_mask[start:end].cpu(),
                            "self_conditioning_signal": signal[start:end, :].cpu(),
                        }
                        if record_debug:
                            item["residual_hidden_states"] = residual_hidden_states_slice[start:end, :].cpu()
                            item["input_ids"] = input_ids[start:end].cpu()
                        seq.input_hidden_states[seq.global_denoising_step + 1] = item

        return self.self_conditioning(inputs_embeds, signal.to(inputs_embeds.dtype))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        input_embeds = self.build_self_conditioned_input_embeds(input_ids)
        hidden_states = self.model(input_ids, positions, input_embeds=input_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        context = get_context()
        logits_source = hidden_states
        if context.run_type == RunType.DENOISE and context.seqs is not None:
            shifted_hidden_states = torch.empty_like(hidden_states)
            for i, seq in enumerate(context.seqs):
                start = i * context.block_length
                end = start + context.block_length
                block_hidden = hidden_states[start:end]
                if seq.prefix_last_hidden is None:
                    shifted_hidden_states[start] = block_hidden[0]
                else:
                    shifted_hidden_states[start] = seq.prefix_last_hidden.to(
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                    )
                if context.block_length > 1:
                    shifted_hidden_states[start + 1:end] = block_hidden[:-1]
                seq.last_denoise_hidden = block_hidden.detach()
                if context.use_self_conditioning:
                    step_key = seq.global_denoising_step + 1
                    item = seq.input_hidden_states.setdefault(step_key, {})
                    if "prompt_mask" not in item:
                        block_prompt_mask = getattr(seq, "block_prompt_mask", None)
                        if block_prompt_mask:
                            item["prompt_mask"] = torch.tensor(
                                block_prompt_mask[:context.block_length],
                                dtype=torch.bool,
                            )
                        else:
                            item["prompt_mask"] = torch.zeros(context.block_length, dtype=torch.bool)
                    item["logits_source_hidden"] = shifted_hidden_states[start:end].detach().cpu()
            logits_source = shifted_hidden_states

        logits = self.lm_head(logits_source)
        if context.run_type == RunType.DENOISE and context.use_self_conditioning:
            store_residual_hidden_states(
                self.model.residual_hidden_states,
                self.model.valid,
                context.context_lens,
                context.block_tables,
                block_length=context.block_length,
                hidden_states=hidden_states
            )
        return logits
