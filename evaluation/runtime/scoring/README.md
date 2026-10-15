# Isolated scoring environment

`scripts/prepare.sh scorer` assembles the runtime of all seven main-table profiles from `sources/`, `wheels/`,
PyPI and the HumanEval+ v0.1.10 release, checking every hash.

Generation and scoring are separate. Never run a code benchmark's generated
solutions on the host or inside a generation container containing credentials.
The portable scorer requires Docker, Podman or Apptainer, a pre-pulled image
pinned by digest, and checksum-pinned scorer sources. It does not implicitly
build an image or download dependencies.

The paper's scoring used Python 3.11 and base image
`python@sha256:a3ab0b966bc4e91546a033e22093cb840908979487a9fc0e6e38295747e49ac0`.
That digest alone is not the full dependency environment for every benchmark.
QwenDLM HumanEval uses the original CodeTACO-backed scorer; LLaDA/DG HumanEval use
their separately pinned EvalPlus protocol. Do not interchange them. Main math
dependencies are recorded in `repro/eval/portable/protocols.json`.

`repro.eval.score` stages only manifest-listed inputs and runtime sources.
It supplies no cloud credentials or model weights and disables container
networking, privilege escalation and writable root filesystems. A dedicated
container host/VM remains recommended.

Each scoring container gets 2 CPUs and 8 GB, except LCB, which gets 8 CPUs: it
judges 8 problems at once with a 6 s limit per test, and with 2 CPUs passing
solutions time out (7–10 problems per DiffusionGemma run).

## Complete Benchmark Runner

`repro.eval.portable.score_benchmarks` implements all evaluated main benchmark
families and LLaDA/DiffusionGemma transfer benchmarks. It calls checksum-pinned
historical implementations, not a common replacement parser. It is intended to
run only through `repro.eval.score`: Linux, Python3.11, UID65534, with `/runtime`
and `/inputs` mounted by the reviewed isolated container stage. Code benchmarks
must never run directly on the host.

The authoritative closure is `repro/eval/portable/scoring_registry.json`.
`source_manifest.json` describes source/data provenance; the two requirements
locks describe exact wheel hashes. Wheel installation happens offline inside
the scorer via `pip --no-index --no-deps --target`, after hash validation. No
dependency fallback, source build or implicit download is allowed.

Supply a runtime root with the profile's exact listed relative paths:

- `repo/repro/eval/benchmarks.py` and `repo/repro/reward/code_taco_reward.py`:
  original QwenDLM classes and shared MBPP CodeTACO dependency.
- `repo/repro/experiments/paper_repro_20260917/score_mbpp427_our_extractor.py`:
  original MBPP-427 multi-candidate extraction.
- `repo/training/diffusion-reasoning-dapo-latent/verl/verl/utils/reward_score/math_verify.py`:
  paper4 math verifier; distinct from QwenDLM's MATH500 class.
- `he_repo/repro/experiments/diffusion_gemma_aec_extra_eval_20260913/dgemma_core_score_point.py`
  and `he_data/HumanEvalPlus-v0.1.10.jsonl`: paper4 HumanEval EvalPlus0.3.1,
  reporting base-test verdicts, not plus-test accuracy.
- `transfer_repo/score_full_extra_results.py`, `transfer_repo/lcb_score.py` and
  `transfer_repo/lcb_lib/`: bundled in `sources/`; `lcb_lib/` is LiveCodeBench's
  test harness as packaged in OpenCompass. `transfer_repo/instruction_following_eval/`
  (IFEval) is not bundled.
- `transfer_data/lcb_v6_tests.json` and `transfer_data/nltk_data/`: not
  bundled. `repro/eval/build_transfer_inputs.py` rebuilds `lcb_v6_tests.json`,
  private tests included, from LiveCodeBench's release; `nltk_data/` is
  supplied separately.
- `he_wheels/` or `transfer_wheels/`: exact Linux x86_64 CPython3.11 wheels.

Only a profile's required files are copied into its scoring capsule; unlisted
files, credentials and weights are excluded. Most wheel pins correspond to
public distributions, but the recorded langdetect wheel was locally built.
Do not assume a newly rebuilt wheel has the same hash. Obtain the exact
recorded artifact or define a new dependency protocol explicitly. The
main-table runtime is complete (`scripts/prepare.sh scorer`).
`scripts/stage_scorer.py --transfer --wheels DIR --lcb-tests FILE --out DIR`
stages AIME, GPQA, MMLU-Pro and LCB; its only wheel, numpy, is among those
`prepare.sh scorer` downloads. IFEval also needs the recorded langdetect wheel
and NLTK data, which are not included.

## Binding And Execution

Prepare the required directory layout without hand-building source trees:

```sh
python -m repro.eval.portable.stage_scoring \
  --profile qwen-dlm.he --bundle /authorized/scorer-sources.tar.gz \
  --bundle-sha256 EXPECTED_BUNDLE_SHA256 --out /local/scoring-assets
```

The profile names and required relative paths are listed in `scoring_registry.json`.
Multiple `--profile`, `--bundle`, and matching `--bundle-sha256` arguments are
supported. Existing authorized reproduction/source bundles can contain many
other files: only the selected profile's exact source/data/wheel closure is read
into the output, and every required member is checked again against its fixed
registry hash. Raw answers, credentials and unrelated scripts are not copied.
Missing inputs fail explicitly. `--file REGISTRY_PATH=/authorized/local/file`
also maps individually supplied sources, licensed tests or wheels to the exact
layout, with the same fixed hash check. No archive code is imported or executed.

For paper4 GSM8K the parser is self-contained, so staging needs only
`--profile paper4.gsm8k --out /local/scoring-assets` and no external bundle.

```sh
python -m repro.eval.portable.bind_benchmark_score \
  --family llada --benchmark gsm8k --evaluation-id llada-base-s12345 \
  --generation-protocol-id llada-base-gsm8k-main-v1-s12345 \
  --records /local/generation/records.jsonl \
  --original-problems /local/inputs/gsm8k.jsonl \
  --expected-problems-sha256 EXPECTED_INPUT_SHA256 \
  --runtime-root /local/scoring-assets \
  --scorer-image python@sha256:IMAGE_DIGEST --out /local/bound-score

python -m repro.eval.score \
  --recipe /local/bound-score/score_recipe.json \
  --runtime-root /local/bound-score/runtime \
  --input-root /local/bound-score/inputs --model-root /local/unused-model \
  --output-root /local/score-attempt --container-engine docker --execute

python -m repro.eval.aggregate \
  --manifest /local/score-attempt/artifacts/aggregation_manifest.json \
  --input-root /local/score-attempt/artifacts
```

Replace the explicitly labeled input hash/image placeholders with actual pinned
values. `--records` can repeat for disjoint shards. `--records-root` selects
normalized `records.jsonl`, or native `shard*.jsonl` when none are present. A
complete single-seed benchmark is required; missing, duplicate or mixed task
coverage fails. AIME requires the full60 tasks and emits two30-task year cells.
The generic binder hashes supplied records and original gold; it does not infer
execution/model attestation. The stronger QwenDLM HumanEval recipe/receipt bridge
remains available as `bind_qwen_dlm_he_score`.

## Apptainer Without A Container Daemon

On hosts without Docker or Podman, pull the pinned digest once and pass the SIF:

```sh
apptainer pull scorer.sif \
  docker://python@sha256:a3ab0b966bc4e91546a033e22093cb840908979487a9fc0e6e38295747e49ac0

python -m repro.eval.score \
  --recipe /local/bound-score/score_recipe.json \
  --runtime-root /local/bound-score/runtime \
  --input-root /local/bound-score/inputs --model-root /local/unused-model \
  --output-root /local/score-attempt \
  --container-engine apptainer --apptainer-image scorer.sif --execute
```

The SIF must record the recipe digest in its definition (`apptainer inspect
--deffile`); its SHA-256 is added to the execution receipt. The caller is
mapped to UID/GID 65534 in a new user and network namespace (`unshare`), and
`apptainer exec --userns --containall` runs without network, home directory or
site bind paths, with read-only `/runtime` and `/inputs`. CPU-time and
file-size rlimits are set with `prlimit`; memory, process-count and CPU limits
come from a `systemd-run --user` scope, so the host must delegate those cgroup
controllers to users. On timeout the scope is killed. Unlike Docker, no seccomp
profile is applied.

IFEval uses strict-prompt credit plus strict/loose instruction metrics, resetting
Python random and langdetect seeds to 0 before each run. For IFEval (its source
and RNG) and LiveCodeBench (its exact source), the scorer is the closest match
found, not a confirmed copy of the one the paper used. AIME and MC retain the
historical parser's known edge cases; no silent correction is made.

The full runner requires an amd64 image even on an ARM host. The CPU fixture
tests check the scorers' logic only; passing them is not a rescore of the full
table.
