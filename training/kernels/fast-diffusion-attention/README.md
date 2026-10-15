# Fast Diffusion Attention

> This is the version of the kernel used for the ORL paper's training runs and needed to reproduce them. The latest
> version, with kernels derived from FlashAttention-2 and FlashAttention-3, is at
> https://github.com/yuezhouhu/fast-diffusion-attention; we are still working on integrating it into ORL training.

GPU-accelerated attention kernels for diffusion language model training, implemented in [Triton](https://github.com/triton-lang/triton).

## Features

- **Staircase block-causal attention** tailored for the prompt / clean-response / noisy-response layout used in diffusion LM training
- Forward + backward support for end-to-end training
- **Grouped-Query Attention (GQA)**: `n_q_heads` can differ from `n_kv_heads` (e.g., 16 Q heads, 8 KV heads)
- Supports `float16` and `bfloat16`
- Variable-length sequences packed in a single batch (`varlen` style, same as `flash_attn_varlen_func`)

## Installation

```bash
pip install -e .
```

Requirements: PyTorch >= 2.0, Triton >= 3.0, NVIDIA GPU (Ampere or newer).

## Quick Start

```python
import torch
from fast_diffusion_attention import diff_attn_varlen_with_noisy_input

# Single document: prompt=128, response=64, staircase_size=4
L0, L1, S = 128, 64, 4
total = L0 + 2 * L1           # 256 tokens
n_heads, head_dim = 16, 64

q = torch.randn(total, n_heads, head_dim, device="cuda", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)

cu_seqlens = torch.tensor([0, total], dtype=torch.int32, device="cuda")
L0_arr = torch.tensor([L0], dtype=torch.int32, device="cuda")
L1_arr = torch.tensor([L1], dtype=torch.int32, device="cuda")

out = diff_attn_varlen_with_noisy_input(
    q, k, v, cu_seqlens, L0_arr, L1_arr,
    staircase_size=S,
    max_seqlen=total,
)
# out: [256, 16, 64], same dtype as input
```

## API Reference

### `diff_attn_varlen_with_noisy_input`

```python
from fast_diffusion_attention import diff_attn_varlen_with_noisy_input

out = diff_attn_varlen_with_noisy_input(
    q, k, v,
    cu_seqlens,
    L0_arr, L1_arr,
    staircase_size=4,
    max_seqlen=None,
    scale=None,
)
```

Fused multi-head attention with the diffusion noisy-input staircase mask. Supports both forward and backward (for training).

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `q` | `Tensor [T, H, D]` | Query tensor. dtype: `float16` or `bfloat16`. |
| `k` | `Tensor [T, Hkv, D]` | Key tensor. `H` must be divisible by `Hkv` (GQA). Same dtype as `q`. |
| `v` | `Tensor [T, Hkv, D]` | Value tensor. Same shape as `k`. |
| `cu_seqlens` | `Tensor [B+1]` | Cumulative sequence lengths, `int32`. E.g., `[0, 256, 512]` for two 256-token docs. |
| `L0_arr` | `Tensor [B]` | Prompt length per document, `int32`. |
| `L1_arr` | `Tensor [B]` | Response length per document, `int32`. Each doc has `L0 + 2*L1` total tokens. |
| `staircase_size` | `int` | Block size for the staircase pattern. Must be a power of 2 in `{1, 2, 4, 8, 16, 32, 64, 128, 256}`. `L0` and `L1` must be divisible by this value. |
| `max_seqlen` | `int` or `None` | Maximum document length in the batch. Pass explicitly to avoid GPU-CPU sync. **Always set this in training.** |
| `scale` | `float` or `None` | Attention scaling factor. Default: `1 / sqrt(head_dim)`. |

#### Returns

| Type | Description |
|------|-------------|
| `Tensor [T, H, D]` | Output tensor, same shape and dtype as `q`. |

## Attention Mask Structure

Each document is laid out as three contiguous segments:

```
[ Prompt (L0 tokens) | Clean Response (L1 tokens) | Noisy Response (L1 tokens) ]
```

Tokens within each segment are grouped into blocks of `staircase_size` (S) tokens. The visibility rules are:

| Query type | Visible keys |
|------------|-------------|
| **Prompt** block *b* | Prompt blocks 0 .. *b* |
| **Clean** block *b* | ALL prompt + clean blocks 0 .. *b* |
| **Noisy** block *b* | ALL prompt + clean blocks 0 .. *b*-1 + own noisy block *b* |

Visual example (`L0=4, L1=6, S=2`):

```
         Prompt   Clean     Noisy
         b0 b1   b0 b1 b2  b0 b1 b2
Prompt  [XX ..   .. .. ..  .. .. ..]
 b0     [XX ..   .. .. ..  .. .. ..]
 b1     [XX XX   .. .. ..  .. .. ..]
        [XX XX   .. .. ..  .. .. ..]
Clean   [XX XX   XX .. ..  .. .. ..]
 b0     [XX XX   XX .. ..  .. .. ..]
 b1     [XX XX   XX XX ..  .. .. ..]
        [XX XX   XX XX ..  .. .. ..]
 b2     [XX XX   XX XX XX  .. .. ..]
        [XX XX   XX XX XX  .. .. ..]
Noisy   [XX XX   .. .. ..  XX .. ..]
 b0     [XX XX   .. .. ..  XX .. ..]
 b1     [XX XX   XX .. ..  .. XX ..]
        [XX XX   XX .. ..  .. XX ..]
 b2     [XX XX   XX XX ..  .. .. XX]
        [XX XX   XX XX ..  .. .. XX]
```

Key property: noisy block *b* sees clean blocks **strictly before** *b* (not *b* itself), preventing information leakage from the clean response to its noisy counterpart.

## Multi-Document Batching

Multiple documents are packed into a single flat tensor, separated by `cu_seqlens`. Each document has independent `L0` and `L1` values:

```python
L0_list = [128, 64, 96]
L1_list = [64, 32, 48]
doc_lens = [L0 + 2 * L1 for L0, L1 in zip(L0_list, L1_list)]
total = sum(doc_lens)  # 576

cu_seqlens = torch.tensor([0, 256, 384, 576], dtype=torch.int32, device="cuda")
L0_arr = torch.tensor(L0_list, dtype=torch.int32, device="cuda")
L1_arr = torch.tensor(L1_list, dtype=torch.int32, device="cuda")

out = diff_attn_varlen_with_noisy_input(
    q, k, v, cu_seqlens, L0_arr, L1_arr,
    staircase_size=4,
    max_seqlen=max(doc_lens),
)
```

Cross-document attention is **never** computed — each document is fully independent.

## Training Example

```python
import torch
from fast_diffusion_attention import diff_attn_varlen_with_noisy_input

L0, L1, S = 512, 256, 8
total = L0 + 2 * L1
n_q_heads, n_kv_heads, head_dim = 16, 8, 128  # GQA: 2 Q heads per KV head

q = torch.randn(total, n_q_heads, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(total, n_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(total, n_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)

cu = torch.tensor([0, total], dtype=torch.int32, device="cuda")
l0 = torch.tensor([L0], dtype=torch.int32, device="cuda")
l1 = torch.tensor([L1], dtype=torch.int32, device="cuda")

# Forward
out = diff_attn_varlen_with_noisy_input(q, k, v, cu, l0, l1, staircase_size=S, max_seqlen=total)

# Backward
loss = out.sum()
loss.backward()
# q.grad, k.grad, v.grad are now populated
```

## Performance

Benchmarked on NVIDIA RTX A6000 (Ampere, sm_86), float16, staircase_size=4.
Compared against `flash_attn_varlen_func(causal=True)` using **identical** per-document `cu_seqlens` (fair comparison — both kernels skip cross-document attention).

Kernel block sizes and launch parameters are **manually tuned per compute capability**. Tuned configs are currently available for Ampere (sm_8x: A6000, A100, RTX 3090, etc.). Other architectures fall back to conservative defaults.

Ratio = ours / flash (lower is better).

### Single-document

**MHA** (n_heads=16, head_dim=64, 1 doc):

| Seq length | Forward | Fwd+Bwd |
|------------|---------|---------|
| 1024 | **1.08x** | **1.16x** |
| 2048 | **1.01x** | 1.25x |
| 4096 | **1.04x** | 1.26x |
| 8192 | **1.06x** | 1.34x |

**GQA** (n_q_heads=16, n_kv_heads=8, head_dim=128, 1 doc):

| Seq length | Forward | Fwd+Bwd |
|------------|---------|---------|
| 1024 | **0.41x** | 1.24x |
| 2048 | **0.92x** | 1.28x |
| 4096 | **1.38x** | 1.32x |
| 8192 | **1.14x** | 1.33x |

### Multi-document (~32k total tokens)

**MHA** (n_heads=16, head_dim=64):

| Doc length | #Docs | Forward | Fwd+Bwd |
|------------|-------|---------|---------|
| 1024 | 32 | **0.85x** | **1.19x** |
| 2048 | 16 | **1.06x** | 1.27x |
| 4096 | 8 | **1.09x** | 1.33x |
| 8192 | 4 | **1.06x** | 1.34x |

**GQA** (n_q_heads=16, n_kv_heads=8, head_dim=128):

| Doc length | #Docs | Forward | Fwd+Bwd |
|------------|-------|---------|---------|
| 1024 | 32 | 1.35x | 1.24x |
| 2048 | 16 | **1.19x** | 1.28x |
| 4096 | 8 | **1.09x** | 1.32x |
| 8192 | 4 | **1.02x** | 1.33x |

Run `python benchmarks/bench_gqa.py` or `python benchmarks/bench_long.py` to reproduce.

## Kernel Tuning

Block sizes (`BLOCK_M`, `BLOCK_N`), `num_warps`, and `num_stages` are tuned offline per GPU architecture (keyed by compute capability major version and head_dim). This avoids `@triton.autotune` overhead from variable sequence lengths during training.

Tuned configs are stored in `_TUNED_CONFIGS` in `fast_diffusion_attention/diff_attn_noisy.py`. To tune for a new GPU,
run the forward and backward sweeps and enter the best block sizes they print into `_TUNED_CONFIGS`:

```bash
python benchmarks/tune_fwd_multidoc.py
python benchmarks/tune_bwd_multidoc.py
```

## Tips

1. **Always pass `max_seqlen`** to avoid GPU-CPU sync overhead.
2. **Use `bfloat16`** for training stability (same speed as `float16`).
3. The forward pass exploits mask sparsity — staircase attention skips key blocks that standard causal would compute.
