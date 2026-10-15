# Offline training

Builds QwenDLM-4B, the start model of the QwenDLM + ORL runs, from
[Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507): a block-diffusion model with the SDAR
architecture and a residual (self-conditioning) adapter. Four stages of supervised denoising training grow the block
from 7 to 256 tokens and then add the adapter. Checkpoint 38750 of the last stage, chosen among its checkpoints, is
the released `yifanyu/QwenDLM-4B`, which [training/README.md](../training/README.md) rescales with
`tools/make_scrp_base.py --gamma 8` before ORL.

## Stages

| Stage | Block | Starts from | Training rows | Steps | GPUs (global batch) |
|---|---|---|---|---|---|
| S1 | 7 | Qwen3-4B-Thinking-2507 | 0 to 790,016 | 26,500 | 8 (32) |
| S2 | 15 | S1 `checkpoint-26500` | 1,000,000 to 2,000,000 | 31,250 | 8 (32) |
| S3 | 256 | S2 `checkpoint-31250` | 8,000,000 to 9,000,000 | 31,250 | 8 (32) |
| S4 | 256, self-conditioning | S3 `checkpoint-31250` | all | 78,417 (QwenDLM-4B: 38,750) | 64 (128) |

Shared by all stages: full fine-tuning at a constant learning rate of 1e-5 (AdamW, bf16, DeepSpeed ZeRO-2), sequences
cut at 8,192 tokens without packing, flex attention. Each training sequence is paired with a noisy copy: a random
fraction of its response tokens, drawn per sequence, is replaced by random vocabulary tokens. Noisy tokens attend
bidirectionally within their block and to the clean tokens before it; clean tokens attend causally. The loss is the
denoising cross-entropy on the noisy copy (on every response position from S3 on) plus next-token cross-entropy on the
clean copy, auto-balanced to the denoising loss.

S4 adds self-conditioning: the model's first-pass prediction, as an expected embedding, passes through an RMSNorm and
a SwiGLU MLP and is added to the input embeddings of a second pass; the sum is RMS-normalized and scaled to 0.02,
Qwen's embedding scale. The adapter's output projection starts at zero, so the second pass first sees only the
normalized input embeddings.

The paper's S4 run trained 78,417 steps. 118 of its checkpoints were decoded on the first 100 GSM8K test problems and
on HumanEval at two confidence thresholds (0.6 and 0.9); accuracy does not keep rising with later checkpoints, and
step 38,750 became QwenDLM-4B.

## Environment

The S4 run used Python 3.10, PyTorch 2.11.0 (CUDA 12.8), transformers 4.55.0, DeepSpeed 0.16.9, datasets 3.6.0 and
accelerate 1.7.0 on H100 80GB GPUs:

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r offline/llama_factory_sdar/requirements.txt transformers==4.55.0 deepspeed==0.16.9
```

## Data

The training data cannot be released. It is self-generated: the paper's 10,037,345 rows (about 41B tokens) are
responses that Qwen3-4B-Thinking-2507 sampled itself, with thinking, to a mixture of reasoning and general instruction
prompts (top-p 0.95, top-k 20, at most 8,192 new tokens). Each row is a pretokenized conversation in the Qwen3 chat
format: parquet files (any number, read in file-name order) with the columns `input_ids`, `attention_mask` and
`labels`, where `labels` is `-100` on prompt tokens.
Each stage reads the row range of its table entry (`llama_factory_sdar/data/dataset_info.json`).

## Reproducing QwenDLM-4B

`tools/make_stage_model.py` builds each stage's start model from the previous stage's checkpoint; `scripts/train.sh`
trains it on one node (all local GPUs) or, with `NNODES`, `NODE_RANK` and `MASTER_ADDR` set, on several nodes, and
resumes from the latest checkpoint when rerun. Run both with the Python of the training environment:

```bash
export MODELS=/path/to/models RUNS=/path/to/runs DATA=/path/to/pretokenized-rows
hf download Qwen/Qwen3-4B-Thinking-2507 --revision 768f209d9ea81521153ed38c47d515654e938aea \
  --local-dir $MODELS/Qwen3-4B-Thinking-2507

python3 offline/tools/make_stage_model.py s1 $MODELS/Qwen3-4B-Thinking-2507 $MODELS/offline-s1
bash offline/scripts/train.sh offline/recipes/s1_b7.yaml $MODELS/offline-s1 $DATA $RUNS/s1

python3 offline/tools/make_stage_model.py s2 $RUNS/s1/checkpoint-26500 $MODELS/offline-s2
bash offline/scripts/train.sh offline/recipes/s2_b15.yaml $MODELS/offline-s2 $DATA $RUNS/s2

python3 offline/tools/make_stage_model.py s3 $RUNS/s2/checkpoint-31250 $MODELS/offline-s3
bash offline/scripts/train.sh offline/recipes/s3_b256.yaml $MODELS/offline-s3 $DATA $RUNS/s3

python3 offline/tools/make_stage_model.py s4 $RUNS/s3/checkpoint-31250 $MODELS/offline-s4
# process all rows once on one node, then train on 8 nodes (the same command on each, i = 0..7)
TOKENIZED=$RUNS/s4-tokenized bash offline/scripts/train.sh offline/recipes/s4_b256_sc.yaml $MODELS/offline-s4 $DATA \
  $RUNS/s4-prep max_steps=1
# up to QwenDLM-4B's step, set it aside, then on to the run's last step (each command on all 8 nodes)
export NNODES=8 NODE_RANK=$i MASTER_ADDR=<address of node 0> TOKENIZED=$RUNS/s4-tokenized
bash offline/scripts/train.sh offline/recipes/s4_b256_sc.yaml $MODELS/offline-s4 $DATA $RUNS/s4 max_steps=38750
rsync -a --exclude 'global_step*' $RUNS/s4/checkpoint-38750/ $MODELS/QwenDLM-4B-offline/   # on one node
bash offline/scripts/train.sh offline/recipes/s4_b256_sc.yaml $MODELS/offline-s4 $DATA $RUNS/s4
```

`$MODELS/QwenDLM-4B-offline` is the QwenDLM-4B of the paper runs; give it to `training/tools/make_scrp_base.py` in
place of the released checkpoint to train ORL from it. The learning rate is constant, so stopping at step 38,750 and
resuming trains the same steps as one run; only five checkpoints are kept, which is why step 38,750 is copied before
the run goes on. The S4 run took about 5.6 s per step on 8 nodes of 8 × H100 (about 122 h for 78,417 steps).
Multi-node runs need `$RUNS` on storage that every node sees, so that each checkpoint is complete in
one place. A stuck collective aborts within 30 minutes (`ddp_timeout: 1800` with NCCL async error handling), and
`train.sh` relaunches from the latest checkpoint. Checkpoints come every 250 steps (five kept) with their optimizer
state; add `report_to=wandb run_name=<name>` to log to W&B.

## Files

| Path | Contents |
|---|---|
| `llama_factory_sdar/` | Trainer: fork of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 0.9.4.dev0 with block-diffusion SFT (pretokenized data, block-aware collation, denoising + clean CE in `src/llamafactory/train/sft/trainer.py`) |
| `recipes/` | One recipe per stage |
| `tools/make_stage_model.py` | Stage start models: SDAR config, `<\|MASK\|>` token, adapter weights |
| `model/chat_template.jinja` | Chat template of the S1 start model, carried by every later stage |
| `scripts/train.sh` | Single- and multi-node launcher with automatic resume |

The model code of every stage is [training/overlays/sdar](../training/overlays/sdar); [PROVENANCE.md](PROVENANCE.md)
maps the recipes to the paper runs.
