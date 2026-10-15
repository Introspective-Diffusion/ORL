# Portable Evaluation

The stable commands are:

```sh
python -m repro.eval.generate --help
python -m repro.eval.score --help
python -m repro.eval.aggregate --help
python -m repro.eval.portable.build_recipe --help
```

Run module commands from `evaluation/`, or put the checkout on
`PYTHONPATH`. Absolute script paths such as `/checkout/repro/eval/generate.py`
also work from an unrelated working directory. Recipes use relative contained
artifact paths, not machine-specific repository paths.

## Workflow

1. Supply licensed frozen inputs, materialized model weights, trusted runtime
   sources and an immutable pre-pulled scorer image. No downloads are implicit.
2. `portable.build_recipe` creates a checksum inventory and generation/scoring
   recipe. Generation adapters live only in `inference/adapters/`.
3. `repro.eval.generate` prints the checked command by default. `--execute`
   runs the supplied trusted generation runtime in a new attempt directory.
4. `portable.stage_scoring` prepares a fixed-hash source/data/wheel closure from
   explicitly supplied local bundles/files. `portable.bind_benchmark_score`
   binds complete main/transfer records to original gold and creates an executable
   isolated score recipe. `portable.bind_qwen_dlm_he_score` additionally
   binds QwenDLM HumanEval original gold, source, recipe, execution receipt, settings,
   normalized/native responses, model inventory and exact eight-shard partition.
5. `repro.eval.score --execute` stages only allowlisted, checksum-pinned runtime
   and input files into a credential-free, network-disabled, read-only container
   (Docker, Podman, or Apptainer for daemonless hosts; see `runtime/scoring/README.md`).
   Model weights and unlisted files are not mounted. Resource limits and timeout
   cleanup remain active. A fresh output directory is required for each attempt.
6. `repro.eval.aggregate --manifest MANIFEST --input-root INPUTS` validates exact
   coverage, hashes and optional Boolean verdicts, then emits metrics. It does not
   execute a scorer or generated programs; supplied-verdict aggregation is labeled
   separately from rescoring.

## Scope And Limits

- Generation recipes support LLaDA, DiffusionGemma and QwenDLM main benchmarks.
  DiffusionGemma requires an already provisioned local server and a supplied
  server manifest. No server or cluster provisioning is hidden in these commands.
- QwenDLM generation recipes represent one of eight fresh-server shards, not a
  complete benchmark. Only all eight disjoint shards make a full QwenDLM cell.
- Scoring dispatch covers all evaluated main family/benchmark combinations and
  all LLaDA/DiffusionGemma transfer benchmarks. `scoring_registry.json` pins the
  required original source, data and offline wheels per protocol;
  `scripts/prepare.sh scorer` supplies those of the main table.
- The generic complete-benchmark binder hashes supplied records and original
  gold; it does not invent generation/model attestation. The separate QwenDLM
  HumanEval binder retains its stronger recipe/receipt binding.
- Transfer generation has a separate `repro/eval/transfer_generate.py` entrypoint.
  AIME scoring emits distinct30-task year cells; repeated seeds are never merged
  into one accuracy cell implicitly.
- Original family-specific prompts, scorers and parser behavior remain distinct.
- Container isolation is a defense layer; run untrusted programs on a dedicated
  host.

## Tests

```sh
python -m unittest discover -s repro/eval/portable/tests -v
```

Trusted fixtures cover clean-directory CLIs, checksum/coverage rejection,
credential filtering, allowlisted staging, timeout cleanup, generation bindings,
and the QwenDLM HumanEval verdict/aggregation bridge. Generated solutions are never
executed by host tests. Real-container acceptance is explicitly skipped unless
an immutable pre-pulled test image and working container engine are supplied.
