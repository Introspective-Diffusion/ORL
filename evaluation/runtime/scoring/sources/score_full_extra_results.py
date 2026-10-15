#!/usr/bin/env python3
"""Score full AIME/MMLU-Pro/GPQA-D/LCB-v6/IFEval outputs."""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from pathlib import Path


ENTITY_PARTS = {
    "llada_base": ("llada_base_q_small_full", "llada_base_q_mmlu_p0", "llada_base_q_mmlu_p1"),
    "llada_orl": ("llada_ours_q_small", "llada_ours_q_mmlu_p0", "llada_ours_q_mmlu_p1"),
    "dg_base_s1234": ("dg_base_seed1234_small_full", "dg_base_s1234_mmlu_p0", "dg_base_s1234_mmlu_p1"),
    "dg_base_s1235": ("dg_base_s1235_small", "dg_base_s1235_mmlu_p0", "dg_base_s1235_mmlu_p1"),
    "dg_base_s1236": ("dg_base_s1236_small", "dg_base_s1236_mmlu_p0", "dg_base_s1236_mmlu_p1"),
    "dg_orl_s1234": ("dg_ours_s1234_small", "dg_ours_s1234_mmlu_p0", "dg_ours_s1234_mmlu_p1"),
    "dg_orl_s1235": ("dg_ours_s1235_small", "dg_ours_s1235_mmlu_p0", "dg_ours_s1235_mmlu_p1"),
    "dg_orl_s1236": ("dg_ours_s1236_small", "dg_ours_s1236_mmlu_p0", "dg_ours_s1236_mmlu_p1"),
}
LETTER_RE = re.compile(r"(?i)Answer\s*:\s*([A-J])")
BOX_RE = re.compile(r"\\boxed\{([^{}]+)\}")


def rows_for(root: Path, parts: tuple[str, ...], benchmark: str) -> list[dict]:
    rows = []
    for part in parts:
        directory = root / part / benchmark
        if not directory.is_dir():
            continue
        for path in directory.glob("*.jsonl"):
            rows.extend(json.loads(line) for line in path.open() if line.strip())
    ids = [row["task_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError((parts, benchmark, "duplicate task ids"))
    return rows


def clean(row: dict) -> bool:
    return row.get("finish") != "length"


def normalized_integer(value) -> str | None:
    text = str(value).strip().strip("$").replace(",", "")
    if not re.fullmatch(r"[+-]?\d{1,3}", text):
        return None
    number = int(text)
    return str(number) if 0 <= number <= 999 else None


def aime_correct(row: dict) -> bool:
    text = row.get("raw") or ""
    values = BOX_RE.findall(text)
    if not values:
        values = re.findall(r"(?i)Answer\s*:\s*([^\n]+)", text)
    predicted = normalized_integer(values[-1]) if values else None
    return clean(row) and predicted == normalized_integer(row["gold"])


def mc_correct(row: dict) -> bool:
    values = LETTER_RE.findall(row.get("raw") or "")
    if not values:
        values = re.findall(r"\\boxed\{([A-J])\}", row.get("raw") or "", re.I)
    return clean(row) and bool(values) and values[-1].upper() == str(row["gold"]).upper()


def ifeval_scores(rows: list[dict]) -> tuple[dict[str, bool], dict]:
    from instruction_following_eval import evaluation_lib

    output = {}
    strict_instruction = strict_total = loose_instruction = loose_total = 0
    for row in rows:
        meta = row["meta"]
        example = evaluation_lib.InputExample(
            key=meta["key"],
            instruction_id_list=meta["instruction_id_list"],
            prompt=meta["prompt"],
            kwargs=meta["kwargs"],
        )
        responses = {meta["prompt"]: row.get("raw") or ""}
        strict = evaluation_lib.test_instruction_following_strict(example, responses)
        loose = evaluation_lib.test_instruction_following_loose(example, responses)
        ok = clean(row) and bool(strict.follow_all_instructions)
        output[row["task_id"]] = ok
        strict_instruction += sum(strict.follow_instruction_list) if clean(row) else 0
        strict_total += len(strict.follow_instruction_list)
        loose_instruction += sum(loose.follow_instruction_list) if clean(row) else 0
        loose_total += len(loose.follow_instruction_list)
    return output, {
        "strict_instruction_accuracy": strict_instruction / strict_total,
        "loose_instruction_accuracy": loose_instruction / loose_total,
        "instruction_n": strict_total,
    }


def lcb_scores(rows: list[dict], module_root: Path, tests: Path) -> dict[str, bool]:
    sys.path.insert(0, str(module_root))
    os.environ["LCB_TESTS"] = str(tests)
    from lcb_score import score_lcb_rows

    scored = score_lcb_rows([row for row in rows if clean(row)])
    return {
        row["task_id"]: clean(row) and bool(scored.get(row["task_id"], {}).get("base"))
        for row in rows
    }


def summarize(rows: list[dict], correct: dict[str, bool], extra: dict | None = None) -> dict:
    usable = [row for row in rows if clean(row)]
    result = {
        "n": len(rows),
        "correct": sum(correct.values()),
        "accuracy": sum(correct.values()) / len(rows),
        "collapse": len(rows) - len(usable),
        "m2t_tpf_mean_clean": statistics.fmean(
            row["completion_tokens"] / row["fwd_denoise"] for row in usable
        ),
        "all_forward_tpf_mean_clean": statistics.fmean(
            row["completion_tokens"] / row["fwd_total"] for row in usable
        ),
        "tps_pooled_clean": sum(row["completion_tokens"] for row in usable)
        / sum(row["wall_s"] for row in usable),
    }
    result.update(extra or {})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--ifeval-root", type=Path, required=True)
    parser.add_argument("--lcb-module", type=Path, required=True)
    parser.add_argument("--lcb-tests", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.ifeval_root))

    results = {}
    for entity, parts in ENTITY_PARTS.items():
        entity_result = {}
        aime = rows_for(args.root, parts, "aime")
        for year in ("aime24", "aime25"):
            selected = [row for row in aime if row["meta"]["year"] == year]
            entity_result[year] = summarize(
                selected, {row["task_id"]: aime_correct(row) for row in selected}
            )
        for benchmark in ("mmlu_pro", "gpqa"):
            rows = rows_for(args.root, parts, benchmark)
            entity_result[benchmark] = summarize(
                rows, {row["task_id"]: mc_correct(row) for row in rows}
            )
        rows = rows_for(args.root, parts, "ifeval")
        scores, extra = ifeval_scores(rows)
        entity_result["ifeval"] = summarize(rows, scores, extra)
        rows = rows_for(args.root, parts, "lcb")
        entity_result["lcb"] = summarize(
            rows, lcb_scores(rows, args.lcb_module, args.lcb_tests)
        )
        results[entity] = entity_result
        print(entity, {key: round(value["accuracy"], 4) for key, value in entity_result.items()}, flush=True)

    for role in ("base", "orl"):
        seeds = [results[f"dg_{role}_s{seed}"] for seed in (1234, 1235, 1236)]
        results[f"dg_{role}_mean"] = {
            benchmark: {
                metric: statistics.fmean(seed[benchmark][metric] for seed in seeds)
                for metric in (
                    "accuracy",
                    "m2t_tpf_mean_clean",
                    "all_forward_tpf_mean_clean",
                    "tps_pooled_clean",
                    "collapse",
                )
            }
            for benchmark in ("aime24", "aime25", "mmlu_pro", "gpqa", "lcb", "ifeval")
        }
    args.out.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
