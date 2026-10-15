# Paper Evaluation

Evaluation separates generation, isolated scoring, and aggregation; `scripts/main_table.sh` chains them. Follow [Quickstart](QUICKSTART.md), [portable evaluation](../../repro/eval/portable/README.md), [transfer generation](../../repro/paper/TRANSFER_GENERATION.md), and [scorer provisioning](../../runtime/scoring/README.md).

- `repro/paper/plan.py` expands the main, transfer, or quick-TPS matrix without starting jobs. Planner `humaneval` maps to recipe-builder `he`.
- `repro/eval/generate.py` validates a checksum-bound main recipe and prints the generation command; execution requires explicit `--execute`.
- `repro/eval/transfer_generate.py` supplies the distinct native transfer generation contract.
- `repro/eval/score.py` invokes an isolated scorer with pinned inputs/runtime. Never run generated candidate programs on the host.
- `repro/eval/aggregate.py` verifies complete task coverage and artifact hashes, then aggregates supplied verdicts. It does not independently rescore responses.

Main and transfer scorer routing preserves family-specific historical parsers. Exact source/data/wheel prerequisites are explicit and hash-pinned; `scripts/prepare.sh scorer` assembles those of the main table. QwenDLM HumanEval also has the stronger original generation-recipe/receipt bridge.

Accuracy includes all expected tasks; missing attempts cannot be silently dropped. Main MBPP has 427 tasks. TPF is a sample mean under the recorded family-specific counter and clean-selection definition, not a pooled token ratio. Canvas TPF excludes edit/refine/commit work and is not wall-clock TPS. TPS uses measured generation time; pooled TPS, response-mean TPS, and benchmark-macro TPS are distinct estimands.

Preserve frozen prompts, original partitions/seeds, decoding caps, and extraction behavior. Changing any of them defines a new protocol.
