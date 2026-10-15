# Paper Evaluation And Inference Quickstart

The commands below are the components that `scripts/main_table.sh` chains; see
the [evaluation README](../../README.md) for an end-to-end run.

Run commands from `evaluation/` with Python 3.10+ and Git. The planner,
runtime staging and raw inference client use the standard library. Evaluation
recipes additionally require `runtime/requirements-eval.txt`; GPU engines use
their own pinned environments, not the host Python environment.

## Inspect The Experiments

```sh
python3 -m repro.paper.plan --suite main
python3 -m repro.paper.plan --suite transfer --model dg-orl --benchmark aime24 --list
python3 -m repro.paper.plan --suite tps --json
```

These are plans only: no download, server, scoring or cluster job is started.
See [checkpoints](CHECKPOINTS.md) for the six model identities, and
[`configs/paper/checkpoints.yaml`](../../configs/paper/checkpoints.yaml) for the
environment variable that names each model's weights (`weights_env`, for
example `QWEN_DLM_ORL_WEIGHTS`). The planner reports those variable names; it
does not automatically locate or verify weights. Every model is pinned to a
Hugging Face revision; download it, or point the variable at a local copy.

## Prepare Runtime Sources

```sh
python3 -m runtime.stage --family qwen-dlm --out outputs/runtime-qwen-dlm
python3 -m runtime.stage --family llada --out outputs/runtime-llada
python3 -m runtime.stage --family dg --out outputs/runtime-dg
```

Every output must be new. This copies source, verifies both SGLang profiles and
writes SHA-256 inventories. It neither installs dependencies nor loads a model.
A failed preparation can leave a partial output; inspect it and choose a fresh
directory for the next attempt. QwenDLM and LLaDA recipe `--engine-subdir` is
`engine`. See [SGLang](../../runtime/sglang/README.md),
[vLLM](../../runtime/vllm/README.md) and
[scoring](../../runtime/scoring/README.md) for environment boundaries.

## Send A Prompt To A Running Server

After separately provisioning a local server, a raw request looks like:

```sh
python3 -m inference.generate --api openai-chat \
  --base-url http://127.0.0.1:8000 --model dg \
  --prompt 'What is 2 + 2?' \
  --sampling-params '{"temperature":0,"max_tokens":128}' \
  --output outputs/example.jsonl
```

This is a connectivity example, not a paper benchmark or decoder configuration.
The server must already have the appropriate model and sampler configured.
`examples/generate.py` exposes the same CLI. The client returns the untouched
native response and request latency, does not score it, and refuses overwriting
an output. Only credential-free loopback endpoints are accepted. For SGLang use
`--api sglang`, an explicitly chat-formatted raw prompt and SGLang sampling
parameters; the client does not choose a chat template.

## Run Paper Recipes

```sh
python3 -m repro.eval.portable.build_recipe --help
python3 -m repro.eval.generate --help
python3 -m repro.eval.score --help
python3 -m repro.eval.aggregate --help
```

Supply the frozen benchmark inputs (`inputs/`), local weights, an exact native
decoder config, staged runtime and a pinned scorer. The registry YAML
files describe protocols; they are not themselves native engine configs or
runnable recipes. Render QwenDLM/LLaDA decoder settings with:

```sh
python3 -m repro.paper.render_config --model qwen-dlm-orl --benchmark he \
  --seed 1234 --out outputs/decoder.yaml
```

For DG, use `runtime.vllm.serve` inside the role-specific patched container.
`build_recipe` hashes concrete files into a runnable recipe. Benchmark
names are `gsm8k`, `he`, `math500`, `mbpp`; use explicit `--seed` (QwenDLM 1234,
LLaDA 12345, DG 1234/1235/1236). QwenDLM also requires `--shard-index` 0 through 7.
All eight disjoint QwenDLM shards are needed for a complete benchmark cell.

Native runtime scripts are `repro/eval/run_eval.py` for QwenDLM and, in
`inference/clients/`, `multi_gen.py` for LLaDA, `vllm_paper4.py` for DG
GSM8K/HumanEval/MATH500 and `vllm_full_extra_client.py` for DG MBPP427. DG
requires a separately running local server and server manifest. Scoring is a
separate container-only stage; never treat successful generation as a passing
benchmark result.

Transfer has its own complete generation entrypoint:

```sh
python3 -m repro.eval.transfer_generate --help
python3 -m repro.eval.portable.stage_scoring --help
python3 -m repro.eval.portable.bind_benchmark_score --help
```

See [transfer generation](../../repro/paper/TRANSFER_GENERATION.md) for exact
model/benchmark/seed/shard commands. It validates the full frozen input before
partitioning; `--smoke-limit` explicitly marks incomplete diagnostics.
The unified scorer supports main and transfer benchmarks, with family-specific
historical parsers retained. The asset staging tool verifies local
source/data/wheel files without downloading or executing them;
`scripts/prepare.sh scorer` fetches and supplies those of the main table. See
[scoring setup](../../runtime/scoring/README.md) and
[evaluation workflow](../../repro/eval/portable/README.md) for binding, coverage
checks and isolation. Model weights are not bundled.

## Local Tests

```sh
python3 -m unittest discover -s inference/tests -v
python3 -m unittest discover -s repro/eval/portable/tests -v
python3 -m unittest discover -s repro/eval/tests -p 'test_*.py' -v
python3 -m unittest discover -s repro/paper -p 'test_*.py' -v
python3 -m unittest discover -s runtime -p 'test_*.py' -v
```

Fixtures are trusted and inert; they do not execute generated solutions on the
host. Actual container acceptance requires an explicitly configured immutable
image and is otherwise skipped. No commands above launch cluster jobs.
