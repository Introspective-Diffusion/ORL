#!/usr/bin/env python3
"""Score sanitized MBPP-427 with our multi-candidate CodeTACO extractor."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def clean(row: dict) -> bool:
    finish = row.get("finish", row.get("finish_reason"))
    if isinstance(finish, dict):
        finish = finish.get("type")
    return finish != "length"


def denoise_forwards(row: dict) -> int:
    return int(row.get("fwd_denoise") or (row.get("metrics") or {})["denoise_forwards"])


def total_forwards(row: dict) -> int:
    return int(row.get("fwd_total") or (row.get("metrics") or {})["total_forwards"])


def wall_seconds(row: dict) -> float:
    return float(row.get("wall_s") or row["elapsed"])


def score_one(payload: tuple[str, dict]) -> tuple[str, bool]:
    repo, row = payload
    if not clean(row):
        return row["task_id"], False
    sys.path.insert(0, str(Path(repo) / "repro/eval"))
    from benchmarks import MBPP

    tests = row["gold"]["tests"]
    joined = "\n".join(tests)
    match = re.search(r"assert\s+\(?\s*([A-Za-z_]\w*)\s*\(", joined)
    entry_point = match.group(1) if match else "solution"
    lines = ["def check(candidate):"]
    for test in tests:
        lines.extend("    " + line for line in str(test).splitlines())
    gold = {"functional": "\n".join(lines), "entry_point": entry_point}
    return row["task_id"], MBPP().score(row.get("raw") or row.get("text") or "", {"gold": gold})


def load_rows(root: Path, entity: str) -> list[dict]:
    directories = list(root.glob(entity)) + list(root.glob(f"{entity}_*"))
    rows = [
        json.loads(line)
        for directory in sorted(set(directories))
        for path in directory.rglob("*.jsonl")
        for line in path.open()
        if line.strip()
    ]
    for directory in sorted(set(directories)):
        for path in directory.rglob("*.json"):
            if path.name == "assignments.json":
                continue
            try:
                payload = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            rows.extend(payload.get("results") or [])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--entity", action="append", required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    expected = {f"mbpp_sanitized/{index}" for index in range(427)}
    output = {}
    for entity in args.entity:
        rows = load_rows(args.root, entity)
        ids = [row["task_id"] for row in rows]
        if len(ids) != len(set(ids)) or set(ids) != expected:
            raise ValueError((entity, len(ids), len(set(ids)), sorted(expected - set(ids))[:20]))
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            scored = dict(pool.map(score_one, ((str(args.repo), row) for row in rows)))
        usable = [row for row in rows if clean(row)]
        output[entity] = {
            "n": 427,
            "correct": sum(scored.values()),
            "accuracy": sum(scored.values()) / 427,
            "collapse": 427 - len(usable),
            "m2t_tpf_mean_clean": statistics.fmean(
                row["completion_tokens"] / denoise_forwards(row) for row in usable
            ),
            "all_forward_tpf_mean_clean": statistics.fmean(
                row["completion_tokens"] / total_forwards(row) for row in usable
            ),
            "tps_pooled_clean": sum(row["completion_tokens"] for row in usable)
            / sum(wall_seconds(row) for row in usable),
        }
        print(entity, json.dumps(output[entity], indent=2), flush=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
