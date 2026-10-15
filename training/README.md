# ORL training

On-policy residual learning (ORL) for diffusion language models. This folder
holds everything needed to train the three model families of the paper with one
engine: QwenDLM (an SDAR-style block DLM converted from Qwen3-4B-Thinking-2507),
LLaDA-2.1-mini, and DiffusionGemma-26B-A4B.

## Layout

| Path | Contents |
|---|---|
| `verl/` | Trainer (fork of VERL): on-policy rollouts with exact residual-state replay, response-normalized CE, adjacent entropy contraction (AEC), entropy-gated CE / reverse KL, packed and cost-balanced replay, DiffusionGemma canvas replay |
| `sglang/` | Rollout engine for QwenDLM and LLaDA-2.1-mini (patched SGLang; `SDARPrefixKV` decoding with asynchronous exact-state capture) |
| `sglang_dgemma/` | Rollout engine for DiffusionGemma (SGLang `diffusion-gemma4-support` @ `11ffa55` plus `Gemma4Renoise` intermediate-state capture) |
| `kernels/fast-diffusion-attention/` | Block-diffusion ("staircase") attention Triton kernel used by replay |
| `overlays/` | Model code: `sdar/` (QwenDLM), `llada2/` (LLaDA-2.1-mini with the residual adapter), `diffusion_gemma/` (Transformers overlay) |
| `rewards/` | Verifiers that decide which rollouts are trained on |
| `recipes/` | The paper's training recipes (see `recipes/README.md`) |
| `scripts/` | `orl_train.py` (launcher), `ray_cluster.py` (two-node recipes), `reproduce_*.sh` (paper runs), `install_dgemma_transformers_overlay.sh` |
| `tools/` | Start-model preparation (`make_scrp_base.py`, `prepare_llada2_actor.sh`), Hugging Face export (`export_hf.py`, `compare_exports.py`) |
| `env/` | Package lists of the runtime environment of the paper runs |
| `PROVENANCE.md` | Source commits of every imported component and how the engine was fused |

## Runtime

The paper runs used an Apptainer image of `lmsysorg/sglang:nightly-dev-cu12-20260611-99ab90c5` with a Python 3.12
virtual environment on top (torch 2.11.0+cu129, transformers 5.8.1, ray 2.55.1). The commands below use directories of
your choice for `VENV` (the virtual environment), `DATA`, `MODELS`, `RUNS` and `OVERLAY` (the DiffusionGemma
Transformers overlay):

```bash
apptainer build sglang.sif docker://docker.io/lmsysorg/sglang@sha256:065d403274324cc9dec4f830ca62540e6ac586746e1aed268c98a2b347d5853f
mkdir -p $VENV
apptainer exec --nv -B $VENV sglang.sif python3 -m venv --system-site-packages $VENV
apptainer exec --nv -B $VENV sglang.sif $VENV/bin/pip install -r training/env/requirements-venv.txt
SIF=$PWD/sglang.sif PY=$VENV/bin/python3
```

Apptainer mounts only a few host directories by default (usually `$HOME`, `/tmp` and the current directory), so the
commands in this file bind the others they use with `-B`, `$VENV` included. A host directory outside these and the
`-B` list is not visible inside the container: with `$VENV` unbound, `$PY` does not exist there and the command fails.

`env/pip-freeze-paper-env.txt` is the full package list of that environment. The launcher puts `kernels/`, the
recipe's SGLang tree and `verl/` on `PYTHONPATH`, so neither SGLang nor VERL is installed, and binds the top-level
directories of the training tree, the run directory, the Python interpreter and every absolute `--set` path into the
container (add others with `--bind`). `orl_train.py --no-container --python <python>` runs a single-node recipe
without Apptainer.

Requirements per node: 8 x H100 80GB; `/dev/shm` large enough for the Ray object store (150 GB for QwenDLM, 120 GB
for LLaDA-2.1-mini and for each DiffusionGemma node); disk for every checkpoint with its optimizer state (all are
kept); `python3` with PyYAML on the host. The paper runs took 7.45 h (QwenDLM, 8 GPUs), 4.52 h (LLaDA-2.1-mini to
step 625, 8 GPUs) and 17.77 h (DiffusionGemma, 16 GPUs).

The code reward (`rewards/code_exec.py`) runs model-generated programs on the training nodes, each in a child Python
process with a timeout and limits on memory, process count and file size. Nothing else isolates them: there is no
sandbox for the filesystem or the network, so train on machines where that is acceptable.

## Start models and data

The training data is the Hugging Face dataset `yifanyu/ORL-training-data`; the reproduce scripts check its `SHA256SUMS`
before training.

| File | Rows | Used by | SHA-256 |
|---|---|---|---|
| `code_l0-20260828/train.parquet` | 819 | QwenDLM | `81096d39e59a40ed4001503dfcacf25dc45c5f3c84600afa198b4e0b11e781da` |
| `gsm8k-math-code_l0-mix-20260909/train.parquet` | 2457 | LLaDA-2.1-mini, DiffusionGemma | `751547ba1a3faa3729c9f7db00b22b984127f4d837097906e927363fd1a81fae` |
| `code_he_clean_rebuild/val.parquet` | 164 | validation file of QwenDLM and LLaDA-2.1-mini (never evaluated) | `9d32c5091435ed0de674853eee4ba3010b4a2390837f9bce75a92a0ef1698878` |

The dataset also holds `gsm8k-math-code_l0-mix-20260909/train_from450.parquet`, the rows from 450 onward with which the
paper's LLaDA-2.1-mini run was relaunched from its step-450 checkpoint; the release trains in one launch and does not
read it.

| Family | Start model |
|---|---|
| QwenDLM | `yifanyu/QwenDLM-4B` (checkpoint 38750 of the offline training, [offline/README.md](../offline/README.md)) after `tools/make_scrp_base.py --gamma 8`, which rescales its residual adapter by γ = 8 without changing what the model computes (the `-scrp8` directory below); its weights and configuration are byte-identical to the start model of the paper runs |
| LLaDA-2.1-mini | `inclusionAI/LLaDA2.1-mini` (revision `20e64e2ad21644d0e5248586ed9c942cdd45de0f`), prepared with `tools/prepare_llada2_actor.sh` |
| DiffusionGemma | `google/diffusiongemma-26B-A4B-it` (revision `52de6b914ee1749a7d4933202505ddf5b414ec43`) plus the runtime Transformers overlay |

```bash
hf download yifanyu/ORL-training-data --repo-type dataset --local-dir $DATA
hf download yifanyu/QwenDLM-4B --revision 94c4a956af73bb7be5ae5474e85a490a96ccb2b2 --local-dir $MODELS/QwenDLM-4B
apptainer exec -B $MODELS,$VENV $SIF $PY training/tools/make_scrp_base.py $MODELS/QwenDLM-4B $MODELS/QwenDLM-4B-scrp8 --gamma 8
hf download inclusionAI/LLaDA2.1-mini --revision 20e64e2ad21644d0e5248586ed9c942cdd45de0f --local-dir $MODELS/LLaDA2.1-mini
bash training/tools/prepare_llada2_actor.sh $MODELS/LLaDA2.1-mini $MODELS/LLaDA2.1-mini-actor
hf download google/diffusiongemma-26B-A4B-it --revision 52de6b914ee1749a7d4933202505ddf5b414ec43 --local-dir $MODELS/diffusiongemma-26B-A4B-it
bash training/scripts/install_dgemma_transformers_overlay.sh $OVERLAY --sif $SIF --python $PY
# the released ORL models: each export follows the tensor layout of its family's model
hf download yifanyu/QwenDLM-4B-ORL --revision e12b90bf8267aa5d994bcf73996f00e00d484239 \
  --local-dir $MODELS/QwenDLM-4B-ORL
hf download yifanyu/LLaDA2.1-mini-ORL --revision eb9e57b7ced51a750ec4d94aa7d4cee46243f597 \
  --local-dir $MODELS/LLaDA2.1-mini-ORL
hf download yifanyu/DiffusionGemma-26B-A4B-it-ORL --revision a5f9c6d23c585eefacf3986a12c3e809061f01ce \
  --local-dir $MODELS/DiffusionGemma-26B-A4B-it-ORL
```

The check at the end of `install_dgemma_transformers_overlay.sh` may print `[ERROR] ... auto_docstring` lines; they
come from Transformers' docstring checks and do not affect the overlay, which is ready once `overlay ready` is printed.

## Reproducing the paper runs

Each recipe below trains with one script, exports the final checkpoint to a Hugging Face directory and evaluates the
export with `evaluation/scripts/main_table.sh`, which prints every cell next to the paper's number (set up the
evaluation as in [evaluation/README.md](../evaluation/README.md)).

| Script | Paper checkpoint | Nodes | Launches |
|---|---|---|---|
| `scripts/reproduce_qwen_dlm.sh` | QwenDLM + ORL (step 350), bit-exact on 8 x H100 80GB with the runtime of `env/` | 1 | stage 1 `qwen_dlm/stage1.yaml` steps 1-300, the same recipe resumed from step 300 for 301-325, as in the paper run; stage 2 `qwen_dlm/stage2_rct.yaml` 326-350 |
| `scripts/reproduce_llada2.sh` | LLaDA-2.1-mini + ORL (step 625) | 1 | `llada2_mini/orl.yaml` steps 1-625 in one launch |
| `scripts/reproduce_diffusion_gemma.sh` | DiffusionGemma + ORL (step 900) | 2 | `diffusion_gemma/aec_c100.yaml` steps 1-900 in one launch |

The LLaDA-2.1-mini and DiffusionGemma paper runs were relaunched once from their step-450 checkpoints with the same
recipe. Their data loaders read the corpus in order, one prompt per step, at a constant learning rate, so one launch
trains on the same rows in the same order.

The scripts skip finished runs and segments, so they can be rerun after an interruption. Options after the positional
arguments go to `orl_train.py` (for example `--sif $SIF --python $PY`). They log to the console; add
`--logger console,wandb` and set `WANDB_API_KEY` (and `WANDB_ENTITY`) to log to W&B as well. The exports run with the
Python of the training environment (see [Exporting a checkpoint](#exporting-a-checkpoint)); `$OUT` is an existing
directory of your choice.

### QwenDLM + ORL

```bash
bash training/scripts/reproduce_qwen_dlm.sh $RUNS/qwen_dlm $MODELS/QwenDLM-4B-scrp8 $DATA --sif $SIF --python $PY
apptainer exec --nv -B $RUNS,$MODELS,$OUT,$VENV $SIF env PYTHONPATH=training/verl $PY training/tools/export_hf.py \
  --target $OUT/qwen-dlm-orl --actor-dir $RUNS/qwen_dlm/stage2/checkpoints/global_step_350/actor \
  --reference $MODELS/QwenDLM-4B-ORL
bash evaluation/scripts/main_table.sh qwen-dlm-orl $OUT/qwen-dlm-orl qwen-dlm-orl-repro
```

Like the paper run, stage 1 stops at step 300 and is resumed from that checkpoint. The resume restarts the rollout
servers and the steps after it depend on that, so the script keeps it.

### LLaDA-2.1-mini + ORL

```bash
bash training/scripts/reproduce_llada2.sh $RUNS/llada2 $MODELS/LLaDA2.1-mini-actor $DATA --sif $SIF --python $PY
apptainer exec --nv -B $RUNS,$MODELS,$OUT,$VENV $SIF env PYTHONPATH=training/verl $PY training/tools/export_hf.py \
  --target $OUT/llada-orl --actor-dir $RUNS/llada2/checkpoints/global_step_625/actor \
  --reference $MODELS/LLaDA2.1-mini-ORL
bash evaluation/scripts/main_table.sh llada-orl $OUT/llada-orl llada-orl-repro
```

### DiffusionGemma + ORL (two nodes)

```bash
# training: the same arguments on both nodes, `head` on the actor node and `worker` on the rollout node;
# $HEAD_IP and $WORKER_IP are the addresses of those two nodes
bash training/scripts/reproduce_diffusion_gemma.sh head $RUNS/dgemma $SIF $PY $MODELS/diffusiongemma-26B-A4B-it \
  $DATA $OVERLAY $HEAD_IP $WORKER_IP
bash training/scripts/reproduce_diffusion_gemma.sh worker $RUNS/dgemma $SIF $PY $MODELS/diffusiongemma-26B-A4B-it \
  $DATA $OVERLAY $HEAD_IP $WORKER_IP
# export (the Transformers overlay goes first on the path) and evaluation, on one node
apptainer exec --nv -B $RUNS,$MODELS,$OUT,$OVERLAY,$VENV $SIF env PYTHONPATH=$OVERLAY:training/verl $PY \
  training/tools/export_hf.py --target $OUT/dg-orl --actor-dir $RUNS/dgemma/checkpoints/global_step_900/actor \
  --reference $MODELS/DiffusionGemma-26B-A4B-it-ORL
bash evaluation/scripts/main_table.sh dg-orl $OUT/dg-orl dg-orl-repro
```

Both nodes must see the training tree, the start model, the data, the overlay, the venv and the run directory at the
same paths. The recipe passes the head's network interface name to every Ray worker, so set `NET_IFNAME` when the
nodes' default-route interfaces have different names. Only the recipe's `ray_kwargs.ray_init.runtime_env.env_vars`
reach the Ray workers; its `env` block reaches the driver only, as in the paper run. The Ray nodes always run in
Apptainer.

### What to expect

On 8 x H100 80GB with the runtime of `env/`, the QwenDLM recipe reproduces the paper checkpoint exactly: its export is
byte-identical to `yifanyu/QwenDLM-4B-ORL` (`python3 training/tools/compare_exports.py $OUT/qwen-dlm-orl
$MODELS/QwenDLM-4B-ORL` compares them tensor by tensor). LLaDA-2.1-mini and DiffusionGemma runs, and QwenDLM runs on
other hardware or software, land close to the paper's checkpoints but not bit for bit: the MoE kernels of
LLaDA-2.1-mini and DiffusionGemma accumulate in a nondeterministic order.

### Manual launches

The scripts only chain `orl_train.py` calls, which can also be run directly:

```bash
python3 training/scripts/orl_train.py training/recipes/llada2_mini/orl.yaml --run-dir $RUNS/llada2 \
  --set MODEL_PATH=$MODELS/LLaDA2.1-mini-actor --set TRAIN_FILE=$DATA/gsm8k-math-code_l0-mix-20260909/train.parquet \
  --set VAL_FILE=$DATA/code_he_clean_rebuild/val.parquet --sif $SIF --python $PY
```

`--steps N --no-save --logger console` turns any recipe into a short smoke test;
`--fresh` starts a resume recipe from `MODEL_PATH`. The LLaDA-2.1-mini and
DiffusionGemma reproduce scripts pass these options on, but
`reproduce_qwen_dlm.sh` sets the step count of each of its segments itself, so
with `--steps` its stage 1 still runs all 300 steps: smoke-test QwenDLM with
`orl_train.py` and `recipes/qwen_dlm/stage1.yaml` directly.

The QwenDLM stage-1 recipe and the LLaDA-2.1-mini recipe start fresh and pin
`trainer.initial_global_step=0`, which the trainer checks against the step of a
resumed checkpoint. To continue one of them from its checkpoint at step `N`
(for example after an interruption), add

```bash
--override ++trainer.resume_mode=resume_path \
--override ++trainer.resume_from_path=$RUNS/<run>/checkpoints/global_step_N \
--override ++trainer.initial_global_step=N
```

A resumed run restores the model, optimizer, data order and trainer RNG, but
not the rollout servers' state, so its steps differ from those of an
uninterrupted run from the first resumed step on.

## Exporting a checkpoint

Run `tools/export_hf.py` with the Python of the training environment (inside `$SIF` with `$PY`, or the
`--no-container` Python) and `training/verl` on `PYTHONPATH`; DiffusionGemma needs its Transformers overlay first on
the path, as in its recipe above. `--reference` is the released ORL model of the same family (`yifanyu/QwenDLM-4B-ORL`,
`yifanyu/LLaDA2.1-mini-ORL`, `yifanyu/DiffusionGemma-26B-A4B-it-ORL`): the export takes its key layout and fails if the
checkpoint cannot supply one of its tensors or holds tensors the reference does not name (`--allow-drop` drops
those). `tools/compare_exports.py A B` compares two exports tensor by tensor.

The export loads the checkpoint's model code through Transformers, which keeps a module cache under `HF_HOME`
(default `~/.cache/huggingface`). If that directory cannot be created in the container, for example because
`~/.cache` links to a file system outside the binds, the export fails with
`FileNotFoundError: ... huggingface/modules`; point `HF_HOME` at a bound directory, as in
`env HF_HOME=$OUT/hf_home PYTHONPATH=training/verl $PY training/tools/export_hf.py ...`. Training is not affected:
`orl_train.py` sets its own `HF_HOME`.
