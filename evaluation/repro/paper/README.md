# Paper Experiment Plans

The [experiment index](../../configs/paper/experiments.yaml) links these
evaluation plans to the [recorded training recipes](../../configs/paper/training.yaml),
[speculative comparison settings](../../configs/paper/speculative.yaml), and
[settings guide](../../docs/reproduction/SETTINGS.md). Training/speculative
records are descriptive settings; they are not launched by this planner.

These JSON-compatible YAML files describe the completed six-model main-table,
four-model transfer-table and quick TPS protocols. The standard-library planner
does not import engines/scorers, download assets, execute responses or launch jobs.

```bash
python3 repro/paper/plan.py --suite main
python3 repro/paper/plan.py --suite transfer --model dg-orl --benchmark aime24 --list
python3 repro/paper/plan.py --suite tps --json
python3 -m unittest discover -s repro/paper -p 'test_*.py'
```

Main: 40 generation/scored runs, 24 displayed cells. QwenDLM uses eight balanced
contiguous partitions per benchmark. LLaDA/DG use four modulo partitions, with
four benchmarks sharing each worker. Thus 192 benchmark shards correspond to
96 worker groups, not 192 simultaneously requested GPUs.

Transfer: 68 generation runs yield 100 scored run/year cells and 24 displayed
cells. Each AIME run includes both 30-task years; filtering `aime24` still
describes the original 60-task generation but selects only that year's score.
AIME uses eight seeds, GPQA three, others LLaDA one/DG three. MMLU-Pro uses
16 modulo partitions and every other transfer benchmark one: 188 worker groups.

TPS: the quick-v1 protocol has three serial same-GPU pairs, 108 total requests
including 12 warmups, and 96 measured requests. It measures throughput, not
full-benchmark accuracy.

`configs/paper/checkpoints.yaml` names local-weight environment variables and
pins the Hugging Face revision of every model; no internal storage paths or
credentials are distributed. Decoding configs retain benchmark-specific caps,
family-specific counters/timing, confidence schedules, engine pins and the DG
long-context setting.

The matrices are plans, not launchers: `scripts/main_table.sh` runs the cells of
the main matrix; the transfer and TPS matrices record the paper's runs. The
original MC parser is intentionally unchanged. Literature-quoted baselines are
outside these measured-model matrices.
