# Transfer Generation CLI

`repro/eval/transfer_generate.py` runs the five transfer benchmarks through the
same native clients as the paper's transfer runs. It never imports a scorer,
executes a generated answer, downloads weights or submits a cluster job.
Without `--execute`, it validates the local inputs/runtime and prints a plan.

## Inputs

Supply local checkpoint weights, a staged runtime, and the original full JSONL
benchmark file. Files must contain the original frozen bytes, not a reserialized
copy or a pre-sharded subset. The CLI checks SHA256, full row count, unique IDs
and messages before selecting a shard. AIME additionally requires both original
30-task `meta.year` groups (`aime24` and `aime25`).

MMLU-Pro and IFEval are bundled in `inputs/transfer/`. `repro/eval/build_transfer_inputs.py --out DIR` rebuilds
AIME, LiveCodeBench and GPQA from the pinned sources below, together with the LiveCodeBench tests the scorer reads
(`lcb_v6_tests.json`), copies the two bundled files, and writes a file only if its SHA256 matches. GPQA is not
bundled because its terms ask that examples not be shared in plain text: the script stores only the order of each
question's options and reads the questions from the gated dataset, so accept its terms on Hugging Face and run
`hf auth login` first.

| Benchmark | Full rows | SHA256 | Partitions |
| --- | ---: | --- | ---: |
| aime | 60 | ec677f2990d5a0dd78e4d631d0dca03ea29f9e0c8103141be6f065cd031018b5 | 1 |
| gpqa | 198 | 7cf55ae09e627820d492577114d0b7838082869f004df5fe48b875002af2034c | 1 |
| mmlu_pro | 12032 | 8f10a6eb35f6279f2a0a30bf4631551d8c12b5c55a2f8d245662fb624c4902cb | 16 |
| lcb | 175 | 26f01f74d4b0e8d74d7f97f482a2456b181c3bc3f289677831399b82dc0c3b8e | 1 |
| ifeval | 541 | 69c2d38e1f5b0e22376f7f3f9184de9ccbdf5e2061fb74bb4c55541fb8f3ca60 | 1 |

| File | Source | Revision |
| --- | --- | --- |
| `aime.jsonl` | `HuggingFaceH4/aime_2024` train and `math-ai/aime25` test, shuffled with seed 20260909 | `2fe88a2f1091d5048c0f36abc874fb997b3dd99a`, `563bb8404243c5f09de6ec262f2db674fe5bce9b` |
| `lcb.jsonl`, `lcb_v6_tests.json` | `livecodebench/code_generation_lite` `test6.jsonl` (release v6, 175 problems) | `0fe84c3912ea0c4d4a78037083943e8f0c4dd505` |
| `mmlu_pro.jsonl` | `TIGER-Lab/MMLU-Pro` test, bundled | `b189ec765aa7ed75c8acfea42df31fdae71f97be` |
| `ifeval.jsonl` | `google-research/google-research` `instruction_following_eval/data/input_data.jsonl`, bundled | `08a8d6736475776f42ffac23b2c13111a28e5795` |
| `gpqa.jsonl` | `Idavidrein/gpqa` `gpqa_diamond.csv` (gated), options in the stored order | `83022cefff930aea54f654c0b282e74b9eeda5c6` |

The paper input predates a recorded shuffle seed. I-DLM's later
question-MD5 permutation does not reproduce its bytes, so the builder retains
the 198 four-digit option permutations needed for parity. Those digits contain
no question or answer text; the gated source is still required locally.

Partitioning is `rows[shard_index::shard_count]`. The count is fixed by the
matrix, not user-overridden. Seeds must belong to the selected model/benchmark
schedule in `transfer.yaml`. Decoding, caps and engine settings come from
`configs/paper/decoding/`; no generic cap override is accepted.

## LLaDA

Stage the native client and hash-verified engine with `runtime/stage.py`. Install
the pinned image/runtime dependencies separately. The CLI checks every engine
source against its bundled source manifest and writes the exact greedy,
Quality-mode decoder with the requested run seed.

```bash
python3 runtime/stage.py --family llada --out /local/runtime-llada
python3 repro/eval/transfer_generate.py \
  --model-id llada-base --runtime /local/runtime-llada \
  --model /local/llada-base --problems /local/inputs/ifeval.jsonl \
  --benchmark ifeval --seed 12345 --shard-index 0 \
  --out /local/output/llada-base-ifeval --execute
```

The ORL command uses `--model-id llada-orl` and its local checkpoint. Model load,
prompt rendering, attention backend, graph capture, counters and native timing
remain in `multi_gen.py`; there is no candidate-code evaluation path.

## DiffusionGemma

Use a separately prepared Base or ORL vLLM image/server. The generator does not
patch the user's installed vLLM. `runtime/vllm/serve.py` owns role-specific
server startup and writes a receipt only after health succeeds. AIME/GPQA need
context 32768; other transfer benchmarks use 8192.

```bash
python3 runtime/stage.py --family dg --out /local/runtime-dg
python3 runtime/vllm/serve.py --model-root /local/dg-base \
  --role base --benchmark ifeval --out /local/server-dg --execute
# In a second shell after server_receipt.json exists:
python3 repro/eval/transfer_generate.py \
  --model-id dg-base --runtime /local/runtime-dg --model /local/dg-base \
  --problems /local/inputs/ifeval.jsonl --benchmark ifeval --seed 1234 \
  --server-receipt /local/server-dg/server_receipt.json \
  --base http://127.0.0.1:8000 --out /local/output/dg-base-ifeval --execute
```

The CLI verifies the receipt's role, exact model-content inventory, image pin,
context length, diffusion config, sampler overrides, per-request seed protocol
and local endpoint. It requires a ready receipt and rejects a sibling
`server_exit.json` before execution. A supplied receipt is not signed hardware
attestation and cannot authenticate a replacement listener on the same port;
the operator must keep that server exclusive and unchanged during generation.

## Smoke and Outputs

Add `--smoke-limit 2` to either generation command for a two-request GPU smoke.
Full input identity is still verified before taking the first two requests of
the exact selected shard. Output scope becomes `smoke_only`, never a full-cell
result. This is the shortest requested acceptance run, not a new benchmark.

Output directories must be new; there is no implicit resume or overwrite.
Successful output includes:

- `records.jsonl`: normalized task ID, raw answer, original gold/meta, finish,
  completion tokens, denoise/total forwards and native request wall time.
- `selected_problems.jsonl`, `input_manifest.json`, `model_manifest.json`:
  selected prompt bytes, full/selected input hashes, parameters and weight hashes.
- `native/`: unchanged native records; `generation.log`: engine/client output.
- `execution_receipt.json`: emitted only after successful client termination,
  exact task coverage and record validation; `scoring_status` remains `not_run`.

## Scoring

Each complete run (one model, benchmark and seed) is scored in the isolated
scorer described in `runtime/scoring/README.md`. Stage the AIME, GPQA, MMLU-Pro
and LCB scorer once, then bind and score every run:

```bash
python3 scripts/stage_scorer.py --transfer --wheels "$ORL_EVAL_WORK/scoring/wheels" \
  --lcb-tests /local/inputs/lcb_v6_tests.json --out /local/transfer-scorer
python3 -m repro.eval.portable.bind_benchmark_score --family dg --benchmark lcb \
  --evaluation-id dg-base-lcb-s1234 --generation-protocol-id dg-base-lcb-transfer-v1-s1234 \
  --records /local/output/dg-base-lcb/records.jsonl --original-problems /local/inputs/lcb.jsonl \
  --expected-problems-sha256 26f01f74d4b0e8d74d7f97f482a2456b181c3bc3f289677831399b82dc0c3b8e \
  --runtime-root /local/transfer-scorer \
  --scorer-image python@sha256:a3ab0b966bc4e91546a033e22093cb840908979487a9fc0e6e38295747e49ac0 \
  --out /local/bound-dg-base-lcb
python3 -m repro.eval.score --recipe /local/bound-dg-base-lcb/score_recipe.json \
  --runtime-root /local/bound-dg-base-lcb/runtime --input-root /local/bound-dg-base-lcb/inputs \
  --model-root /local/unused-model --output-root /local/score-dg-base-lcb \
  --container-engine docker --execute
```

A run's accuracy is `correct / n` from `scoring_receipt.json`; length-capped
answers score wrong. Clean TPF is the mean of `completion_tokens / fwd_denoise`
over records that did not hit the cap. Tables average the seeds equally. AIME
year metadata, IFEval instruction specifications and LCB IDs are preserved
exactly. The existing MC parser is not altered by generation.

CPU inert tests:

```bash
python3 -m unittest repro.eval.tests.test_transfer_generate
```
