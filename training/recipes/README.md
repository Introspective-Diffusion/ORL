# Training recipes

QwenDLM trains in two stages; LLaDA-2.1-mini and DiffusionGemma train in one:

| Recipe | Paper run | Start | Data |
|---|---|---|---|
| `qwen_dlm/stage1.yaml` | QwenDLM stage 1, steps 1-325 | `checkpoint-38750-scrp8` | `code_l0-20260828/train.parquet` |
| `qwen_dlm/stage2_rct.yaml` | QwenDLM + ORL stage 2, steps 326-350 (paper checkpoint: step 350) | stage-1 FSDP checkpoint `global_step_325` | same |
| `llada2_mini/orl.yaml` | LLaDA-2.1-mini + ORL, steps 1-625 (paper checkpoint: step 625) | `LLaDA2.1-mini-actor` | `gsm8k-math-code_l0-mix-20260909/train.parquet` |
| `diffusion_gemma/aec_c100.yaml` | DiffusionGemma + ORL, steps 1-900 (paper checkpoint: step 900) | `diffusiongemma-26B-A4B-it` | `gsm8k-math-code_l0-mix-20260909/train.parquet` |

The LLaDA-2.1-mini and DiffusionGemma paper runs were relaunched once from their step-450 checkpoints with the same
settings (LLaDA-2.1-mini on `train_from450.parquet`, the rows it had not reached). Both data loaders read the corpus in
order, one prompt per step, at a constant learning rate, so a single launch trains on the same rows in the same order.

## Format

```yaml
name: ...
description: ...
nodes: 1                    # 2 for DiffusionGemma (actor node + rollout node)
gpus_per_node: 8
base_config: ...            # training/verl/examples/diffusion_trainer/<base_config>.yaml
sglang_dir: sglang_dgemma   # optional; default training/sglang
extra_python_path: [...]    # optional; e.g. the DiffusionGemma Transformers overlay
inputs: {PLACEHOLDER: description, ...}
env: {NAME: value, ...}     # environment of the trainer and rollout processes
overrides: [...]            # Hydra overrides, applied in order after base_config
```

`scripts/orl_train.py` resolves `${PLACEHOLDER}` values from `--set KEY=VALUE`
or the environment and fills in `ORL_TRAINING`, `ORL_RECIPE_DIR`, `RUN_DIR`,
`EXPERIMENT_NAME`, `NET_IFNAME`, `RAY_TMP_DIR` and `NODE_TMP_DIR` itself. The
launcher also owns `PYTHONPATH`, `CUDA_VISIBLE_DEVICES`, the cache directories
and the W&B directories; set `WANDB_ENTITY`/`WANDB_PROJECT`/`WANDB_API_KEY` in
your environment if you keep the W&B logger.

`decode/` holds the rollout decoder configuration of each family.

## How the recipes were made

The paper runs were launched through layered research scripts. For QwenDLM and
LLaDA-2.1-mini, the original launch chains (at the training-code commits listed in
`../PROVENANCE.md`) were executed with an `apptainer` shim that records the
environment and arguments of the final `python -m verl.trainer.main_ppo` call
instead of running it; for DiffusionGemma, the trainer script that the original
launcher generated into the paper run directory was replayed the same way. The
recorded launch was then rewritten with placeholders for paths and hosts, and
logging identities were removed.

The QwenDLM and DiffusionGemma recipes build on the base config
`QwenDLM-4B-postnorm002-DAPO-sglang-noedit-cwppo-smoke`, a renamed copy of the one
their paper runs were launched with; the `postnorm002-DAPO-sglang-noedit-cwppo-smoke`
part of its name is left over from the research repository.

Every recipe was checked by composing its Hydra config (base config + overrides,
plus the same legacy-key migration `main_ppo` applies) and comparing it with the
config the paper run printed at startup. Remaining differences:

- keys added by the engine fusion that the paper run's code did not have; all of
  them default to off for the other families;
- paths, Ray temp directories and run names;
- the default API-key variable of the rollout judge
  (`...rollout_quality_filter.server_judge.api_key_env`), which no recipe enables;
- QwenDLM: the reward entry point is `rewards/code_exec.py:compute_score_lenient`,
  which matches the paper run's reward on every tested input;
- LLaDA-2.1-mini and DiffusionGemma: the step count of the whole run (625 and 900)
  instead of the first launch's 450. For DiffusionGemma the trainer script of the
  paper run's second launch (`028ce292`) is used; it differs from the first
  launch's script only in launch-level details.

The decoder configurations are the paper ones with comments removed. The
LLaDA-2.1-mini decoder additionally sets `confidence_temperature_first_step_full:
true`, which selects the step-linear schedule the LLaDA run used in the fused
SGLang tree (see `../PROVENANCE.md`).
