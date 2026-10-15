#!/usr/bin/env python3
"""Generate one full extra benchmark shard from a vLLM DiffusionGemma server."""

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path


COUNTERS = (
    "vllm:diffusion_num_denoising_steps_total",
    "vllm:diffusion_num_committed_tokens_total",
)
MAX_NEW = {
    "aime": 16384,
    "gpqa": 16384,
    "mbpp500": 2048,
    "mbpp427": 4096,
    "mbpp427_48k": 49152,
    "mbpp427_32k": 32768,
    "lcb": 4096,
    "ifeval": 2048,
    "mmlu_pro": 4096,
}


def metrics(base: str) -> tuple[float, float]:
    text = urllib.request.urlopen(base + "/metrics", timeout=30).read().decode()
    values = []
    for name in COUNTERS:
        matches = re.findall(
            rf"^{re.escape(name)}(?:{{[^}}]*}})?\s+([0-9.e+]+)",
            text,
            re.MULTILINE,
        )
        values.append(sum(float(value) for value in matches) if matches else 0.0)
    return values[0], values[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", default="dg")
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--benchmark", choices=tuple(MAX_NEW), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--partition-index", type=int, default=0)
    parser.add_argument("--partition-count", type=int, default=1)
    parser.add_argument("--max-denoising-steps", type=int, default=48)
    parser.add_argument("--ar", action="store_true")
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.problems.open()
        if line.strip()
    ]
    lo = len(rows) * args.partition_index // args.partition_count
    hi = len(rows) * (args.partition_index + 1) // args.partition_count
    rows = rows[lo:hi][args.worker::args.workers]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    completed = {
        json.loads(line)["task_id"]
        for line in args.out.open()
        if line.strip()
    } if args.out.exists() else set()

    with args.out.open("a") as output:
        for index, problem in enumerate(rows):
            if problem["task_id"] in completed:
                continue
            steps0, committed0 = (0.0, 0.0) if args.ar else metrics(args.base)
            body = json.dumps(
                {
                    "model": args.model,
                    "messages": problem["messages"],
                    "temperature": 0.0,
                    "max_tokens": MAX_NEW[args.benchmark],
                    "seed": args.seed,
                }
            ).encode()
            request = urllib.request.Request(
                args.base + "/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            started = time.perf_counter()
            response = json.loads(
                urllib.request.urlopen(request, timeout=7200).read()
            )
            elapsed = time.perf_counter() - started
            steps1, committed1 = (0.0, 0.0) if args.ar else metrics(args.base)
            choice = response["choices"][0]
            if args.ar:
                completion = int(response["usage"]["completion_tokens"])
                blocks = completion
                denoise = completion
                counter_steps = completion
                committed = completion
            else:
                counter_steps = int(round(steps1 - steps0))
                committed = int(round(committed1 - committed0))
                assert committed > 0 and committed % 256 == 0
                blocks = committed // 256
                denoise = counter_steps - blocks
                assert 0 < denoise <= blocks * args.max_denoising_steps
            raw = choice["message"].get("content") or ""
            reasoning = choice["message"].get("reasoning_content")
            if reasoning and reasoning not in raw:
                raw = f"<think>\n{reasoning}\n</think>\n{raw}"
            row = {
                "benchmark": args.benchmark,
                "task_id": problem["task_id"],
                "entry_point": problem.get("entry_point"),
                "raw": raw,
                "finish": (
                    "length"
                    if choice.get("finish_reason") == "length"
                    else "stop"
                ),
                "completion_tokens": int(response["usage"]["completion_tokens"]),
                "wall_s": elapsed,
                "n_blocks": blocks,
                "fwd_denoise": denoise,
                "fwd_commit": blocks,
                "fwd_total": counter_steps,
                "committed_canvas_tokens": committed,
                "seed": args.seed,
            }
            for key in ("gold", "meta"):
                if key in problem:
                    row[key] = problem[key]
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            if index % 25 == 0:
                print(
                    args.benchmark,
                    index,
                    problem["task_id"],
                    row["finish"],
                    row["completion_tokens"],
                    denoise,
                    flush=True,
                )


if __name__ == "__main__":
    main()
