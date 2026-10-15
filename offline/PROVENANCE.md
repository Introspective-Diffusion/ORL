# Provenance of `offline/`

`llama_factory_sdar/` was imported from the authors' private research repository at commit `e37e5e1`
(`training/llama_factory_sdar`, a fork of LLaMA-Factory 0.9.4.dev0). The recipes, the stage-model tool and the
launcher were written for the release from that commit's recipes, notes and launch scripts.

## Paper runs

| Stage | Recipe in the research repository | Started from | Paper checkpoint |
|---|---|---|---|
| S1 | `qwen3_4b_thinking_2507_b7_bidir_randmask_randtok_reasonmix_tokenized.yaml` | Qwen3-4B-Thinking-2507 | `checkpoint-26500` |
| S2 | `qwen3_4b_thinking_2507_b15_bidir_randmask_randtok_reasonmix_stage2_new1m_1m-2m.yaml` | S1 `checkpoint-26500` | `checkpoint-31250` (one epoch) |
| S3 | `qwen3_4b_thinking_2507_b256_randtok_allloss_baseline_from_b15_bidir_randmask_randtok_ckpt31250_8m-9m.yaml` | S2 `checkpoint-31250` | `checkpoint-31250` (one epoch) |
| S4 | `qwen3_4b_thinking_2507_b256_baseline_sc_postnorm002_zero_p100_from_baseline_ckpt31250_full.yaml` | S3 `checkpoint-31250` | last `checkpoint-78417`; `checkpoint-38750` = QwenDLM-4B |

- S1 read the dataset's train split as it then stood, 790,016 rows (one epoch = 24,688 steps at batch 32); its
  recipe asked for three epochs, the run was stopped after the first, and S2 started from its `checkpoint-26500`.
- S1 to S3 ran on one node of 8 × H100 each with per-device batch 2 and gradient accumulation 2.
- S4 ran on 8 nodes of 8 × H100 with per-device batch 2 and no gradient accumulation (set at launch over the
  recipe's 2 × 2), a global batch of 128 on the full 10,037,345-row train split, for one epoch (78,417 steps). Of
  its checkpoints, 118 were decoded on the first 100 GSM8K test problems and HumanEval at confidence thresholds 0.6
  and 0.9, and step 38,750 was selected as QwenDLM-4B (`evaluation/configs/paper/training.yaml`, `qwen_dlm_offline_initializer`).
- Every stage used a constant learning rate, so a checkpoint does not depend on where the run stops later: S1 to S3
  stop at the checkpoint the next stage started from, and S4 at its last step.

## Changes for the release

- `llama_factory_sdar/`: only `src/`, the license, the packaging files and the ZeRO-2 config are kept. The dataset
  registry is four local entries (`offline_s1` to `offline_s4`) with each stage's row range. A local dataset
  directory is read in sorted file order (`src/llamafactory/data/loader.py`), so row ranges do not depend on the
  filesystem's listing order. Three comments in `src/llamafactory/data/mm_plugin.py` are in English.
- Recipes: one per stage; `max_steps` is the checkpoint the next stage started from (S1 to S3) or the run's last
  step (S4); the model and output paths come from the launcher; `report_to: none`; `ddp_timeout: 1800` (it only bounds how long a stuck collective waits). The clean-CE
  weight `ce_alpha` keeps the paper value although `loss_auto_balance` sets the clean-CE scale instead.
- Model code: every stage uses `training/overlays/sdar`, the S4 model code. The S1 to S3 runs used earlier versions
  of the same files; the later code adds self-conditioning, beta masking, `loss_on_all_noisy_tokens` and
  `scale_loss_by_p_mask`, whose defaults keep the earlier behaviour, and has the same `use_regular_causal` masks.
- Stage models: the paper runs built them by hand (copy the checkpoint's weights and files, edit `config.json`,
  add an adapter shard). `tools/make_stage_model.py` does the same; the S4 `config.json` it writes matches the paper
  run's in every SDAR field. The adapter's gate and up projections are drawn with the tool's seed.
- Tokenizer: S1 adds `<|MASK|>` (id 151669) to Qwen3-4B-Thinking-2507's tokenizer and sets
  `model/chat_template.jinja`, the chat template of the paper's S1 start model (Qwen3's hybrid-thinking template,
  which leaves the reply open where the 2507 repository's own template starts it with `<think>`).
- Launcher: `scripts/train.sh` follows the paper's per-node S4 launcher (c10d rendezvous, NCCL async error handling,
  relaunch from the latest checkpoint, a per-node warm-up of the `trust_remote_code` module cache) without
  cluster-specific paths.
- The training data is not included.

## Environment of the S4 run

Python 3.10, PyTorch 2.11.0+cu128, transformers 4.55.0, DeepSpeed 0.16.9 (recorded in the checkpoints), datasets
3.6.0, accelerate 1.7.0, trl 0.9.6.
