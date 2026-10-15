# Experiment settings

Start with [`configs/paper/experiments.yaml`](../../configs/paper/experiments.yaml).
It links the selected checkpoints, training recipes, evaluation matrices,
decoder profiles, scorer records and the paper's numbers. The manuscript retains
the method, comparison scope and metric definitions; this directory records
the implementation settings needed to inspect or reproduce each experiment.

## Configuration map

| Experiment information | File |
| --- | --- |
| Selected model roles and weight inputs | [checkpoints.yaml](../../configs/paper/checkpoints.yaml) |
| Offline initializer and three online training recipes | [training.yaml](../../configs/paper/training.yaml) |
| QwenDLM runtime and Base/ORL decoders | [qwen-dlm.yaml](../../configs/paper/decoding/qwen-dlm.yaml) |
| LLaDA runtime and Base/ORL decoders | [llada.yaml](../../configs/paper/decoding/llada.yaml) |
| DiffusionGemma runtime and Base/ORL decoders | [dg.yaml](../../configs/paper/decoding/dg.yaml) |
| Main-table tasks, seeds, partitions and scoring | [main.yaml](../../repro/paper/main.yaml) |
| Transfer tasks, per-task repeats and scoring | [transfer.yaml](../../repro/paper/transfer.yaml) |
| Separate small same-GPU throughput check | [tps.yaml](../../repro/paper/tps.yaml) |
| Speculative targets, drafters and verification sweep | [speculative.yaml](../../configs/paper/speculative.yaml) |
| Input identities and portable protocols | [protocols.json](../../repro/eval/portable/protocols.json) |
| Scorer identities | [scoring_registry.json](../../repro/eval/portable/scoring_registry.json) |
| Transfer input hashes and generation protocol | [TRANSFER_GENERATION.md](../../repro/paper/TRANSFER_GENERATION.md) |
| The paper's main-table numbers | [main.json](../../results/reference/main.json) |

Settings are JSON-compatible YAML and can be inspected with the Python standard library.
The training and speculative files record recipes; the bundled evaluation
planner does not execute those training or speculative jobs.

## What runs end to end

- Main table: `scripts/main_table.sh ROLE MODEL_DIR` runs and scores every
  generation job of one model in a single command; see the
  [quickstart](QUICKSTART.md).
- Transfer: `repro/eval/transfer_generate.py` generates the outputs from the
  frozen task files, and checks each file against its SHA256 in
  [TRANSFER_GENERATION.md](../../repro/paper/TRANSFER_GENERATION.md).
  MMLU-Pro and IFEval are bundled, and `repro/eval/build_transfer_inputs.py`
  rebuilds AIME, LiveCodeBench and GPQA; GPQA cannot be redistributed in
  plain text, so it is built from the gated Hugging Face dataset with your own
  access.
- Same-GPU TPS check: for now it can only be dry-run, with
  `python3 repro/paper/plan.py --suite tps --list`. A measurement runs each
  family's Base and ORL arms serially on one physical GPU and counts only
  runs that finished without a retry.

## Inspect an evaluation cell

Run from `evaluation/`:

```bash
python3 repro/paper/plan.py --suite main --model qwen-dlm-orl --benchmark gsm8k --json
python3 repro/paper/plan.py --suite transfer --model dg-orl --benchmark aime24 --list
python3 repro/paper/render_config.py --suite main --model llada-orl --benchmark humaneval --seed 12345 --out /tmp/orl-llada-he.yaml
```

The planner resolves benchmark-specific output caps and decoder overrides
before expanding seeds and task shards. Model weights and frozen task inputs
are supplied separately, as described in the [quickstart](QUICKSTART.md).
The render command requires a new output path and handles QwenDLM/LLaDA native
decoders; DiffusionGemma uses the vLLM runtime adapter.

## Evaluation and repeated runs

Main-table QwenDLM and LLaDA cells each use one generation run. DiffusionGemma
cells average three generation seeds (1234--1236). Transfer AIME uses eight
seeds per model, GPQA three, and the remaining transfer tasks one LLaDA or
three DiffusionGemma seeds. Each seed produces one completion per task;
repeated-run quality is the equal-weight mean of run accuracies. It is not
best-of-k or majority voting.

QwenDLM uses canvas seed 12345 and server seed 1234.
LLaDA uses random-token seed 12345 for the main suite and the listed run seed
for transfer. DiffusionGemma resets its canvas RNG per request. Zero request
sampling temperature does not remove canvas randomness.

## Decoder and timing scope

The selected main-table systems use their recorded decoder policies:

- QwenDLM Base uses a probability threshold; ORL uses an entropy budget and
  confidence annealing. Edit controls and the math output caps also differ.
- LLaDA Base/ORL use thresholds 0.70/0.85 with the same edit, refinement,
  temperature and commit settings.
- DiffusionGemma Base/ORL use entropy bounds 0.10/0.12 and different sampler
  temperature policies.

ORL is evaluated as a joint training-and-decoding method: the rollout policy
determines which residual-conditioned states are available for learning,
and each method is evaluated with its specified generation policy. The manuscript's LLaDA
fixed-control threshold grid is a separate comparison. Sweep and ablation
settings must not be inferred from the selected main-table profile.

Timing includes prefill, decode, scheduling and host overhead on one H100,
with one request in flight. Loading/startup and preprocessing are excluded.
Original full-task Base/ORL accounting has no warmup exclusion; speculative
servers use two warmups. Clean TPS excludes length failures; all-output and
completed-output TPS charge their request time. Completed-output TPS still
credits completed incorrect answers. TPF counts denoising forwards; refinement
and commit overhead remain included in wall time.

Engine revisions and container digests are in the decoder profiles. Digests
pin userspace, not the host driver; the recorded host drivers are `590.48.01`
and `595.91.07`. Other recorded versions are DiffusionGemma vLLM
`0.22.1rc1.dev357+g74b5964f0` and a later Qwen reproduction environment with
PyTorch `2.11.0+cu129`, Transformers `5.8.1` and FlashInfer `0.6.12`.

## Input and scoring records

Main tasks are GSM8K test (1,319), MATH-500 (500), HumanEval (164) and the
canonical MBPP collection (427). HumanEval uses base tests, not expanded
EvalPlus tests. QwenDLM retains its CodeTACO extraction/execution path;
LLaDA/DiffusionGemma use EvalPlus 0.3.1 base tests. MBPP retains the original
multi-candidate extraction policy and is neither MBPP378 nor the raw 500-task
split. Cross-family and quoted literature scores are not scorer-matched.

Qwen numeric extraction uses the last boxed answer, with a last-numeric-match
fallback after the thinking delimiter. GPQA/MMLU-Pro retain the original
multiple-choice parser, including its unanchored fallback. Math scorer pins
are math-verify 0.8.0 and SymPy 1.14.0. Full hashes are in the linked protocols;
the following manuscript prefixes identify the original records:

| Record | Revision or SHA-256 prefix |
| --- | --- |
| Qwen GSM8K upstream revision | `740312add88f` |
| Qwen MATH-500 upstream revision | `6e4ed1a2a79a` |
| Qwen GSM8K serialized input | `40805b714b72` |
| Qwen MATH-500 serialized input | `2d084c9d3ec6` |
| Qwen HumanEval prompt/gold rows | `0f0ea89ab42f` |
| HumanEvalPlus v0.1.10 data | `42526ec0e7d5` |
| Canonical MBPP427 tasks | `b5443f461dec` |
| Shared AIME 2024/2025 input | `ec677f2990d5` |
| GPQA-Diamond input | `7cf55ae09e62` |
| MMLU-Pro input | `8f10a6eb35f6` |
| LiveCodeBench-v6 input | `26f01f74d4b0` |
| IFEval input | `69c2d38e1f5b` |

Frozen content hashes identify the input files. DiffusionGemma serves AIME and
GPQA with a 32,768-token context and every other benchmark with 8,192.

The offline initializer's auxiliary checkpoint sweep evaluated 118
checkpoints on GSM8K100 and HumanEval164. The selected initializer is step
38,750; the original auxiliary plot's nearest evaluated marker is step
38,500. This sweep is distinct from the main-table generation repeats.
