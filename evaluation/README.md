# ORL evaluation

Generation, isolated scoring and aggregation for the paper's main table: GSM8K, MATH-500, HumanEval and MBPP on
QwenDLM, LLaDA-2.1-mini and DiffusionGemma, each as base model and with
ORL, using the decoders and samplers recorded for the paper.

## Requirements

- Linux with NVIDIA GPUs; every job uses one GPU (DiffusionGemma needs 80 GB).
- [Apptainer](https://apptainer.org) (tested with 1.5.0) with user namespaces and `nvidia-container-cli` for `--nvccli`;
  jobs run with only their GPU visible and no network. `build_images.sh` needs unprivileged `apptainer build`
  (fakeroot or user namespaces), about 60 GB of disk for the images, and about 50 GB free in Apptainer's temporary
  directory while it builds (`APPTAINER_TMPDIR`, default `/tmp`; also set `APPTAINER_CACHEDIR` if your home directory
  is small).
- For scoring: a user systemd session (`systemd-run --user`) with the memory, pids and cpu cgroup controllers
  delegated, plus `unshare` and `prlimit`; batch-job shells often lack the systemd session.
- Python 3.10+ with PyYAML, `git`, `pip` and `curl` on the host.
- Work and image paths without `,` or `:`.

## Evaluate a model

```bash
export ORL_EVAL_WORK=/path/to/eval-work   # images, runtimes, inputs, model copies, generations and scores
export MODELS=/path/to/models              # downloaded models
bash scripts/build_images.sh               # SGLang, vLLM (DiffusionGemma) and scorer images, pinned by digest
bash scripts/prepare.sh                    # engine runtimes, frozen inputs and decoder configs, scorer runtime
hf download yifanyu/QwenDLM-4B-ORL --revision e12b90bf8267aa5d994bcf73996f00e00d484239 \
  --local-dir $MODELS/QwenDLM-4B-ORL
bash scripts/main_table.sh qwen-dlm-orl $MODELS/QwenDLM-4B-ORL
```

The six models and their pinned Hugging Face revisions are listed in
[docs/reproduction/CHECKPOINTS.md](docs/reproduction/CHECKPOINTS.md); evaluate each at its listed revision (the
DiffusionGemma base model's evaluation revision differs from its training-start revision).

`main_table.sh ROLE MODEL_DIR [NAME]` copies the model into the work directory, runs every generation job on the GPUs in
`GPUS` (default: `CUDA_VISIBLE_DEVICES`, else all), scores each benchmark and prints accuracy and TPF next to the
paper's numbers. Rerunning it resumes: finished jobs are kept, unfinished ones are moved aside (`*.unfinished-*`) and
rerun, and a benchmark is scored again when its generations changed. `ROLE` selects the recorded decoder and sampler:

| role | model | jobs per benchmark |
|---|---|---|
| `qwen-dlm-base`, `qwen-dlm-orl` | QwenDLM, QwenDLM + ORL | 8 shards, seed 1234 |
| `llada-base`, `llada-orl` | LLaDA-2.1-mini, LLaDA-2.1-mini + ORL | 1, seed 12345 |
| `dg-base`, `dg-orl` | DiffusionGemma, DiffusionGemma + ORL | seeds 1234, 1235, 1236 (averaged); GSM8K in 4 parts |

A checkpoint trained with `training/` is evaluated after exporting it to a Hugging Face directory with
`training/tools/export_hf.py`. Single steps are available as `scripts/generate.sh GPU SPEC` (one job; `DRY_RUN=1` only
builds its recipe), `scripts/score.sh FAMILY NAME BENCH [SEED]` and `scripts/summarize.py`.

Accuracy counts every task. TPF (tokens per forward) is the mean over responses that stop before the length cap of
generated tokens divided by denoising forwards; each family keeps the counter the paper used. MBPP is the sanitized
427-task set, scored with the paper's multi-candidate extractor (`runtime/scoring/sources/score_mbpp427_our_extractor.py`).

## Expected results

`main_table.sh` prints every cell next to the paper's number. Reproduced numbers can differ slightly from the paper's
(sampling, the GPU and its software all play a part); the differences are small. To reproduce the ORL models
themselves, follow the recipes in [training/README.md](../training/README.md#reproducing-the-paper-runs).

The transfer evaluation (the paper's Table 2) is not covered by these scripts. Instead,
`repro/eval/transfer_generate.py` generates it from the frozen task files and checks each against its SHA256. The MMLU-Pro and IFEval inputs are
included in `inputs/transfer/`, and `repro/eval/build_transfer_inputs.py` rebuilds AIME, LiveCodeBench and GPQA, the
last from your own Hugging Face access, since its terms rule out sharing it in plain text.
[repro/paper/TRANSFER_GENERATION.md](repro/paper/TRANSFER_GENERATION.md) documents its protocol and input hashes.

## Layout

| path | contents |
|---|---|
| `scripts/` | the commands above |
| `inference/` | request clients and the engine patches (`inference/vllm/patches` for DiffusionGemma) |
| `repro/` | recipe building, generation, isolated scoring, aggregation |
| `runtime/` | staging of the SGLang/vLLM runtimes and of the scorer (`runtime/scoring`) |
| `configs/paper/` | recorded protocols: decoders, samplers, checkpoints |
| `inputs/` | frozen benchmark inputs ([sources and licenses](inputs/README.md)) |
| `results/reference/` | the paper's numbers |
| `docs/reproduction/` | protocol details: [settings](docs/reproduction/SETTINGS.md), [evaluation](docs/reproduction/EVALUATION.md), [checkpoints](docs/reproduction/CHECKPOINTS.md) |

The CPU tests are listed under [local tests](docs/reproduction/QUICKSTART.md#local-tests).
