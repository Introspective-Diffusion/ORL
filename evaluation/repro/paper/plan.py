#!/usr/bin/env python3
"""Expand the paper matrices offline. No downloads, GPU jobs or scoring."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUITES = ("main", "transfer", "tps")
ALIASES = {"he": "humaneval", "mbpp427": "mbpp"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(relative, root=ROOT):
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), "Config escapes repository")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate config key: " + key)
            result[key] = value
        return result
    data = json.loads(path.read_text(), object_pairs_hook=unique)
    require(data.get("schema_version") == 1, "Unsupported config schema")
    return data


def partition_indices(n, count, kind):
    require(type(n) is int and n > 0 and type(count) is int and 0 < count <= n,
            "Invalid partition size")
    require(kind in ("modulo", "contiguous_balanced"), "Unknown partition kind")
    if kind == "modulo":
        parts = [list(range(i, n, count)) for i in range(count)]
    else:
        q, r = divmod(n, count)
        parts = [list(range(i*q+min(i, r), i*q+min(i, r)+q+(i < r))) for i in range(count)]
    require(sorted(x for part in parts for x in part) == list(range(n)), "Partition coverage mismatch")
    return parts


def quick_indices(benchmark, n, selection):
    require(selection["repeats"] == 1 and selection["warmup_per_benchmark"] == 1
            and selection["measured_per_benchmark"] == 8, "Only frozen quick-v1 selection supported")
    order = sorted(range(n), key=lambda i: hashlib.sha256(f"tps-quick-v1|{benchmark}|{i}".encode()).hexdigest())
    return order[:1], order[1:9]


def resolve_parameters(model, benchmark, seed, suite_name, root=ROOT):
    config = load(model["decoding"], root)
    require(config["family"] == model["family"], "Checkpoint/decoder family mismatch")
    role = config["roles"][model["role"]]
    require(benchmark in role["max_new_tokens"], "Unsupported decoder benchmark")
    cap = role["max_new_tokens"][benchmark]
    require(type(cap) is int and cap > 0, "Invalid token cap")
    decoder = {**config["decoder_common"], **role["decoder"], **role["benchmark_decoder"].get(benchmark, {})}
    if model["family"] == "llada":
        decoder["randtok_seed"] = seed
    sampling = dict(config["sampling"])
    if model["family"] in ("qwen-dlm", "llada"):
        decoder.update({k: v for k, v in sampling.items()
                        if k in ("temperature", "temperature_min", "temperature_max", "top_k", "top_p")})
    if model["family"] == "dg":
        sampling["seed"] = seed
    engine = dict(config["engine"])
    engine["context_length"] = engine.get("benchmark_context_length", {}).get(benchmark, engine.get("context_length"))
    return {"max_new_tokens": cap, "decoder": decoder, "sampling": sampling, "engine": engine,
            "status": config["status"], "weights_env": model["weights_env"]}


def expand(suite_name, models=None, benchmarks=None, root=ROOT):
    require(suite_name in SUITES, "Unknown suite")
    suite = load(f"repro/paper/{suite_name}.yaml", root)
    require(suite["suite"] == suite_name, "Suite name mismatch")
    checkpoints = load("configs/paper/checkpoints.yaml", root)["models"]
    require(len(suite["models"]) == len(set(suite["models"])), "Duplicate suite models")
    requested_models = set(models or suite["models"])
    require(requested_models and requested_models <= set(suite["models"]), "Unknown/unavailable model filter")
    available = set(suite["benchmarks"])
    available |= {k for v in suite["benchmarks"].values() for k in v.get("score_subsets", {})}
    requested_benchmarks = {ALIASES.get(b, b) for b in benchmarks} if benchmarks else available
    require(requested_benchmarks and requested_benchmarks <= available, "Unknown benchmark filter")
    runs, score_cells, worker_groups = [], [], set()
    for name in suite["models"]:
        if name not in requested_models:
            continue
        require(name in checkpoints, "Missing checkpoint definition")
        model = checkpoints[name]
        require(name == model["family"] + "-" + model["role"], "Invalid model identity")
        if model["public_hf"]:
            revision = model["public_hf"]["revision"]
            require(len(revision) == 40 and all(c in "0123456789abcdef" for c in revision), "Unpinned HF revision")
        for benchmark, spec in suite["benchmarks"].items():
            score_names = list(spec.get("score_subsets", {benchmark: spec["n"]}))
            selected_scores = score_names if benchmark in requested_benchmarks else [s for s in score_names if s in requested_benchmarks]
            if not selected_scores:
                continue
            seeds = suite.get("seed_overrides", {}).get(benchmark, {}).get(model["family"], suite["seeds_by_family"][model["family"]])
            require(seeds and len(seeds) == len(set(seeds)) and all(type(s) is int and s >= 0 for s in seeds), "Invalid/duplicate seeds")
            partition = suite.get("partition_by_family", {}).get(model["family"], suite.get("partition_default"))
            partition = suite.get("partition_overrides", {}).get(benchmark, partition)
            require(partition is not None, "Missing partition policy")
            if suite_name == "tps":
                warmup, measured = quick_indices(benchmark, spec["n"], suite["selection"])
                parts = [warmup + measured]
            else:
                warmup, measured = [], []
                parts = partition_indices(spec["n"], partition["count"], partition["kind"])
            for seed in seeds:
                key = f"{name}.{benchmark}.s{seed}"
                shards = [{"index": i, "count": len(parts), "row_count": len(indices),
                           "input_positions": indices} for i, indices in enumerate(parts)]
                runs.append({"id": key, "model": name, "family": model["family"], "role": model["role"],
                             "benchmark": benchmark, "seed": seed, "dataset_n": spec["n"],
                             "generated_rows": sum(map(len, parts)), "partition_kind": partition["kind"],
                             "shards": shards, "warmup_indices": warmup, "measured_indices": measured,
                             "parameters": resolve_parameters(model, benchmark, seed, suite_name, root),
                             "assets_env": suite["assets_env"], "public_execution_ready": False})
                for score_name in selected_scores:
                    score_cells.append({"id": f"{name}.{score_name}.s{seed}", "generation_run": key,
                                        "model": name, "benchmark": score_name, "seed": seed,
                                        "n": len(measured) if suite_name == "tps" else spec.get("score_subsets", {}).get(score_name, spec["n"]),
                                        "subset": {"field": spec["subset_field"], "equals": score_name}
                                        if "score_subsets" in spec else None})
                for i in range(len(parts)):
                    group = model["family"] if suite_name == "tps" else (
                        f"{name}.s{seed}.p{i}" if suite_name == "main" and model["family"] != "qwen-dlm" else f"{key}.p{i}")
                    worker_groups.add(group)
    require(len({r["id"] for r in runs}) == len(runs), "Duplicate generated run IDs")
    require(len({r["id"] for r in score_cells}) == len(score_cells), "Duplicate scored cell IDs")
    counts = {"generation_runs": len(runs), "scored_run_cells": len(score_cells),
              "displayed_cells": len({(c["model"], c["benchmark"]) for c in score_cells}),
              "benchmark_shards": sum(len(r["shards"]) for r in runs), "worker_groups": len(worker_groups),
              "generated_rows": sum(r["generated_rows"] for r in runs)}
    if suite_name == "tps":
        counts["measured_rows"] = sum(len(r["measured_indices"]) for r in runs)
    if not models and not benchmarks:
        require(counts == suite["expected"], f"Matrix count mismatch: {counts} != {suite['expected']}")
    return {"suite": suite_name, "status": suite["status"], "dry_run": True,
            "launches_resources": False, "counts": counts,
            "scoring_or_timing_contract": suite.get("scoring", suite.get("timing")),
            "runs": runs, "score_cells": score_cells}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITES, default="main")
    parser.add_argument("--model", action="append", help="Exact checkpoint ID; repeat to select several")
    parser.add_argument("--benchmark", action="append", help="Benchmark or AIME year; repeat to select several")
    parser.add_argument("--json", action="store_true", help="Print expanded plan including exact input positions")
    parser.add_argument("--list", action="store_true", help="List generation runs after counts")
    args = parser.parse_args()
    try:
        result = expand(args.suite, args.model, args.benchmark)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps({k: v for k, v in result.items() if k not in ("runs", "score_cells")}, indent=2))
        if args.list:
            for run in result["runs"]:
                print(f"{run['id']} rows={run['generated_rows']} shards={len(run['shards'])} cap={run['parameters']['max_new_tokens']}")


if __name__ == "__main__":
    main()
