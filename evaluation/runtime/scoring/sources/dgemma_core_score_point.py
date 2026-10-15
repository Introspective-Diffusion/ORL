#!/usr/bin/env python3
"""Score one HE+/MBPP+/MATH step-cap point safely."""

import argparse
import importlib.util
import json
import os
import signal
from pathlib import Path


def rows(directory):
    output = []
    for path in directory.glob("shard*.jsonl"):
        output.extend(json.loads(line) for line in path.open() if line.strip())
    return output


def score_code(data, dataset, directory):
    from evalplus.data import get_human_eval_plus, get_mbpp_plus
    from evalplus.evaluate import evaluate
    from evalplus.sanitize import sanitize

    expected = get_human_eval_plus() if dataset == "humaneval" else get_mbpp_plus()
    assert len(data) == len(expected)
    samples = directory / "samples-sanitized.jsonl"

    def bounded(row):
        if row["finish"] == "length":
            return ""
        def timeout(_signum, _frame):
            raise TimeoutError
        previous = signal.signal(signal.SIGALRM, timeout)
        signal.setitimer(signal.ITIMER_REAL, 2)
        try:
            return sanitize(row.get("raw") or "", row.get("entry_point"))
        except (TimeoutError, RecursionError, SyntaxError, ValueError):
            return ""
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)

    with samples.open("w") as handle:
        for row in data:
            handle.write(json.dumps({
                "task_id": row["task_id"],
                "solution": bounded(row),
            }) + "\n")
    result_path = Path(str(samples).replace(".jsonl", "_eval_results.json"))
    result_path.unlink(missing_ok=True)
    evaluate(
        dataset=dataset,
        samples=str(samples),
        parallel=int(os.environ.get("EVAL_PARALLEL", "4")),
        i_just_wanna_run=True,
    )
    result = json.loads(result_path.read_text())["eval"]
    base = plus = 0
    for row in data:
        item = result[row["task_id"]][0]
        b = item.get("base_status") == "pass" and row["finish"] != "length"
        p = b and item.get("plus_status") == "pass"
        base += b
        plus += p
    return {"base": base / len(data), "plus": plus / len(data)}


def score_math(data, repo):
    path = (
        repo
        / "training/diffusion-reasoning-dapo-latent/verl/verl"
        / "utils/reward_score/math_verify.py"
    )
    spec = importlib.util.spec_from_file_location("campaign_math_verify", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    correct = sum(
        row["finish"] != "length"
        and bool(module.compute_score(row.get("raw") or "", row["gold"]))
        for row in data
    )
    return correct / len(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--point", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    he = rows(args.point / "he")
    mbpp = rows(args.point / "mbpp")
    math = rows(args.point / "math500")
    assert len(he) == 164 and len(mbpp) == 378 and len(math) == 500
    summary = {
        "humaneval_plus": score_code(he, "humaneval", args.point / "he"),
        "mbpp_plus": score_code(mbpp, "mbpp", args.point / "mbpp"),
        "math500": {"accuracy": score_math(math, args.repo)},
    }
    (args.point / "score_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
