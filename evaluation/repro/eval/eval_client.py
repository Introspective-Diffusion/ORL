#!/usr/bin/env python3
"""Per-shard SGLang evaluation client for GSM8K, MATH-500, HumanEval, MBPP.

Launched as a subprocess by run_eval.py.  All configuration is read from
environment variables so the orchestrator can set them per-shard without
CLI argument quoting edge-cases.

Required env vars:
    EVAL_BENCHMARK      gsm8k | math500 | humaneval | mbpp
    SGLANG_BASE_URL     e.g. http://127.0.0.1:46000
    SGLANG_START_Q      first question index (global, 0-based)
    SGLANG_NUM_Q        number of questions for this shard
    SGLANG_MAX_NEW_TOKENS
    SGLANG_EVAL_OUT     path to write JSON output (atomic)

Optional env vars:
    SGLANG_WORKERS          parallelism within shard (default: 1)
    SGLANG_SAVE_FULL_TEXT   1 = store full model output in each row (default: 0)
    HF_DATASETS_CACHE       datasets cache dir

Output JSON per shard::
    {
      "benchmark": "gsm8k",
      "start_q": 0,
      "num_q": 100,
      "max_new_tokens": 32768,
      "results": [
        {
          "idx": 0,
          "correct": true,
          "pred": "42",          # GSM8K only
          "gold": "42",
          "problem": "...",
          "finish_reason": {"type": "stop"},
          "prompt_tokens": 123,
          "completion_tokens": 456,
          "metrics": {
            "total_forwards": ...,
            "denoise_forwards": ...,
            "commit_forwards": ...,
            "total_tokens": ...
          },
          "elapsed": 1.23,
          "text_head": "...",
          "text_tail": "..."
          # "text": "..."  if SGLANG_SAVE_FULL_TEXT=1
        }
      ]
    }
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmarks import GSM8K, MATH500, HumanEval, MBPP  # noqa: E402

_STAT_KEYS = (
    "total_forwards",
    "denoise_forwards",
    "refine_forwards",
    "commit_forwards",
    "total_tokens",
)

_BENCHMARKS = {
    "gsm8k": GSM8K(),
    "math500": MATH500(),
    "humaneval": HumanEval(),
    "mbpp": MBPP(),
}


# ---------------------------------------------------------------------------
# SGLang HTTP helpers
# ---------------------------------------------------------------------------


def _generate(base_url: str, prompt: str, max_new_tokens: int) -> dict:
    payload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
        },
    }
    timeout = max(900, int(max_new_tokens * 0.8))
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            resp = requests.post(
                f"{base_url}/generate",
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
            payload["_eval_request_attempts"] = attempt + 1
            return payload
        except Exception as exc:
            last_exc = exc
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
    assert last_exc is not None
    raise last_exc


def _request_metrics(meta: dict) -> dict:
    """Aggregate per-request DLLM stats from sglang_sdar_request_metrics list."""
    items = meta.get("sglang_sdar_request_metrics") or []
    return {
        key: sum(int(item.get(key) or 0) for item in items)
        for key in _STAT_KEYS
    }


def _adaptive_temperature_blocks(meta: dict) -> list[dict]:
    """Extract JSON-safe per-block adaptive-temperature diagnostics."""
    items = meta.get("sglang_sdar_request_metrics") or []
    if isinstance(items, dict):
        items = [items]
    blocks = []
    for item in items:
        if not isinstance(item, dict):
            continue
        steps = item.get("adaptive_temperature_diagnostics") or []
        summary = item.get("adaptive_temperature_summary") or {}
        if not steps and not summary:
            continue
        blocks.append(
            {
                "block_idx": int(item.get("block_idx") or 0),
                "block_size": int(item.get("block_size") or 0),
                "start_offset": int(item.get("start_offset") or 0),
                "summary": summary,
                "steps": steps,
            }
        )
    return blocks


def _adaptive_confidence_temperature_blocks(meta: dict) -> list[dict]:
    """Extract adaptive confidence data without conflating proposal data."""

    items = meta.get("sglang_sdar_request_metrics") or []
    if isinstance(items, dict):
        items = [items]
    blocks = []
    for item in items:
        if not isinstance(item, dict):
            continue
        steps = (
            item.get("adaptive_confidence_temperature_diagnostics")
            or []
        )
        summary = (
            item.get("adaptive_confidence_temperature_summary") or {}
        )
        if not steps and not summary:
            continue
        blocks.append(
            {
                "block_idx": int(item.get("block_idx") or 0),
                "block_size": int(item.get("block_size") or 0),
                "start_offset": int(item.get("start_offset") or 0),
                "summary": summary,
                "steps": steps,
            }
        )
    return blocks


# ---------------------------------------------------------------------------
# Per-question evaluation
# ---------------------------------------------------------------------------


def _build_prompt(benchmark: str, item: dict) -> str:
    return (
        f"<|im_start|>user\n{item['prompt']}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
    )


def run_one(
    base_url: str,
    benchmark_obj,
    idx: int,
    item: dict,
    max_new_tokens: int,
    save_full_text: bool,
) -> dict:
    prompt = _build_prompt(benchmark_obj.name, item)
    t0 = time.time()
    response = _generate(base_url, prompt, max_new_tokens)
    elapsed = time.time() - t0

    text = response.get("text", "")
    meta = response.get("meta_info", {})
    correct = bool(benchmark_obj.score(text, item))

    row: dict = {
        "idx": idx,
        "correct": correct,
        "gold": item["gold"],
        "problem": item["prompt"],
        "finish_reason": meta.get("finish_reason"),
        "prompt_tokens": int(meta.get("prompt_tokens") or 0),
        "completion_tokens": int(meta.get("completion_tokens") or 0),
        "metrics": _request_metrics(meta),
        "adaptive_temperature_diagnostics": _adaptive_temperature_blocks(meta),
        "adaptive_confidence_temperature_diagnostics": (
            _adaptive_confidence_temperature_blocks(meta)
        ),
        "elapsed": elapsed,
        "eval_run_id": os.environ.get("SGLANG_EVAL_RUN_ID"),
        "request_attempts": int(response.get("_eval_request_attempts") or 1),
        "text_head": text[:500],
        "text_tail": text[-500:],
    }
    if benchmark_obj.name == "gsm8k":
        from benchmarks import extract_gsm8k_answer

        row["pred"] = extract_gsm8k_answer(text)
    if save_full_text:
        row["text"] = text
    return row


# ---------------------------------------------------------------------------
# Atomic shard file write
# ---------------------------------------------------------------------------


def _write_shard(
    out_path: Path,
    by_idx: dict[int, dict],
    *,
    benchmark: str,
    start_q: int,
    num_q: int,
    max_new_tokens: int,
) -> None:
    payload = {
        "benchmark": benchmark,
        "start_q": start_q,
        "num_q": num_q,
        "max_new_tokens": max_new_tokens,
        "results": [by_idx[i] for i in sorted(by_idx)],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    os.replace(tmp, out_path)


# ---------------------------------------------------------------------------
# Resume: load previously completed rows
# ---------------------------------------------------------------------------


def _load_existing(out_path: Path, expected_indices: set[int]) -> dict[int, dict]:
    if not out_path.exists():
        return {}
    try:
        payload = json.loads(out_path.read_text())
        return {
            int(row["idx"]): row
            for row in payload.get("results", [])
            if int(row["idx"]) in expected_indices
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    benchmark_name = os.environ["EVAL_BENCHMARK"].strip().lower()
    if benchmark_name not in _BENCHMARKS:
        sys.exit(
            f"Unknown EVAL_BENCHMARK={benchmark_name!r}. "
            f"Valid: {sorted(_BENCHMARKS)}"
        )
    bm = _BENCHMARKS[benchmark_name]

    base_url = os.environ.get("SGLANG_BASE_URL", "http://127.0.0.1:30000")
    start_q = int(os.environ["SGLANG_START_Q"])
    num_q = int(os.environ["SGLANG_NUM_Q"])
    max_new_tokens = int(os.environ.get("SGLANG_MAX_NEW_TOKENS", "32768"))
    workers = int(os.environ.get("SGLANG_WORKERS", "1"))
    save_full_text = os.environ.get("SGLANG_SAVE_FULL_TEXT", "0") == "1"
    out_path = Path(os.environ["SGLANG_EVAL_OUT"])

    items = bm.load(num_q, start_q)
    expected = list(range(start_q, start_q + len(items)))
    expected_set = set(expected)

    by_idx = _load_existing(out_path, expected_set)
    if by_idx:
        print(
            f"[resume] {benchmark_name} shard start={start_q}: "
            f"{len(by_idx)}/{len(expected)} already done",
            flush=True,
        )

    pending = [
        (global_idx, items[global_idx - start_q])
        for global_idx in expected
        if global_idx not in by_idx
    ]

    def save() -> None:
        _write_shard(
            out_path,
            by_idx,
            benchmark=benchmark_name,
            start_q=start_q,
            num_q=num_q,
            max_new_tokens=max_new_tokens,
        )

    def on_complete(row: dict) -> None:
        by_idx[row["idx"]] = row
        save()
        status = "OK" if row["correct"] else "X"
        fr = row.get("finish_reason")
        fr_type = fr.get("type") if isinstance(fr, dict) else str(fr)
        print(
            f"[{benchmark_name}] Q{row['idx']}: {status} "
            f"finish={fr_type} comp={row['completion_tokens']}",
            flush=True,
        )

    if workers == 1:
        for idx, item in pending:
            row = run_one(base_url, bm, idx, item, max_new_tokens, save_full_text)
            on_complete(row)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    run_one, base_url, bm, idx, item, max_new_tokens, save_full_text
                ): idx
                for idx, item in pending
            }
            for fut in as_completed(futures):
                on_complete(fut.result())

    final_correct = sum(bool(r["correct"]) for r in by_idx.values())
    final_total = len(by_idx)
    print(
        f"[{benchmark_name}] SHARD DONE start={start_q} n={final_total} "
        f"correct={final_correct}",
        flush=True,
    )
    if final_total != len(expected):
        sys.exit(
            f"[{benchmark_name}] incomplete shard: "
            f"{final_total}/{len(expected)} rows"
        )


if __name__ == "__main__":
    main()
