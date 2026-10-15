"""
benchmarks.py — GSM8K, MATH-500, HumanEval-164, and MBPP-500.

GSM8K / MATH-500 keep the boxed math_verify protocol.
HumanEval / MBPP are the locked SDAR coding harness (code_he_clean parquet
+ code_taco_reward), not EvalPlus.

MATH-500 scoring requires ``math_verify`` and ``latex2sympy2_extended``.
An ImportError is raised at import time if they are missing so callers never
accidentally fall back to a wrong scoring semantic.

Interface per benchmark class:

    load(n: int, start: int = 0) -> list[dict]
        Each dict: {"prompt": <user-message str>,
                    "gold":   <answer str>,
                    "meta":   <dict>}

    score(output_text: str, item: dict) -> bool
        item is one element returned by load().
        output_text is the model's raw decoded completion.

    name: str
    gen_length: int  (suggested max new tokens)
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------------

GSM8K_DATASET = "openai/gsm8k"
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"


def extract_gsm8k_answer(text: str) -> str:
    """Extract the predicted answer from a model completion.

    Priority:
      1. Last \\boxed{...} in the text.
      2. Last number after </think> (or in whole text if no </think>).

    Returns "?" when nothing is found.
    """
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        return (
            boxed[-1]
            .replace(",", "")
            .replace("$", "")
            .replace("\\", "")
            .strip()
        )
    after = text.split("</think>")[-1] if "</think>" in text else text
    nums = re.findall(r"[\d,]+", after)
    return nums[-1].replace(",", "") if nums else "?"


def _norm_gsm8k(value: str) -> str:
    value = str(value).strip().replace(",", "").replace("$", "")
    try:
        number = float(value)
        return str(int(number)) if number == int(number) else str(number)
    except Exception:
        return value


class GSM8K:
    name = "gsm8k"
    gen_length = 32768

    def load(self, n: int, start: int = 0) -> list[dict]:
        from datasets import load_dataset  # deferred — tests mock this

        ds = load_dataset(
            GSM8K_DATASET,
            "main",
            split="test",
            revision=GSM8K_REVISION,
        )
        end = min(start + n, len(ds))
        items = []
        for item in ds.select(range(start, end)):
            gold = item["answer"].split("####")[-1].strip().replace(",", "")
            prompt = (
                f"{item['question']}\n"
                "Please reason step by step, and put your final answer within \\boxed{}."
            )
            items.append({"prompt": prompt, "gold": gold, "meta": {}})
        return items

    def score(self, output_text: str, item: dict) -> bool:
        pred = extract_gsm8k_answer(output_text or "")
        return _norm_gsm8k(pred) == _norm_gsm8k(item["gold"])


# ---------------------------------------------------------------------------
# MATH-500
# ---------------------------------------------------------------------------

MATH500_DATASET = "HuggingFaceH4/MATH-500"
MATH500_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"


def _require_math_verify() -> tuple:
    """Import math_verify and latex2sympy2_extended; raise clearly if absent."""
    try:
        from latex2sympy2_extended import NormalizationConfig  # noqa: F401
        from math_verify import (
            ExprExtractionConfig,
            LatexExtractionConfig,
            parse,
            verify,
        )
    except ImportError as exc:
        raise ImportError(
            "math_verify / latex2sympy2_extended are required for MATH-500 scoring "
            "but could not be imported. Install them before evaluating MATH-500.\n"
            f"Original error: {exc}"
        ) from exc
    return ExprExtractionConfig, LatexExtractionConfig, parse, verify


def _strip_thinking(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"^.*</think>\s*", "", text, count=1, flags=re.DOTALL)


def _verify_math500(prediction: str, reference: str) -> bool:
    ExprExtractionConfig, LatexExtractionConfig, parse, verify = _require_math_verify()
    prediction = _strip_thinking(prediction)
    gold_parsed = parse(
        f"${reference}$",
        extraction_mode="first_match",
        extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()],
    )
    if not gold_parsed:
        return False
    pred_parsed = parse(
        prediction,
        extraction_config=[
            LatexExtractionConfig(boxed_match_priority=0),
            ExprExtractionConfig(),
        ],
    )
    if not pred_parsed:
        return False
    try:
        return bool(verify(gold_parsed, pred_parsed))
    except Exception:
        return False


class MATH500:
    name = "math500"
    gen_length = 32768

    def load(self, n: int, start: int = 0) -> list[dict]:
        from datasets import load_dataset  # deferred

        ds = load_dataset(
            MATH500_DATASET,
            split="test",
            revision=MATH500_REVISION,
        )
        end = min(start + n, len(ds))
        items = []
        for item in ds.select(range(start, end)):
            prompt = (
                f"{item['problem']}\n"
                "Please reason step by step, and put your final answer within \\boxed{}."
            )
            items.append(
                {"prompt": prompt, "gold": item["answer"], "meta": {}}
            )
        return items

    def score(self, output_text: str, item: dict) -> bool:
        return _verify_math500(output_text or "", item["gold"])


# ---------------------------------------------------------------------------
# HumanEval-164 — locked SDAR coding harness.
# Prompts come from artifacts/data/code_he_clean/val.parquet so they match
# the training-time format. Scoring is code_taco_reward (last closed
# ```python fence + check(candidate) subprocess). Do not use EvalPlus here.
# ---------------------------------------------------------------------------

HUMANEVAL_PARQUET_ENV = "HUMANEVAL_VAL_PARQUET"
HUMANEVAL_PARQUET_DEFAULT = str(
    Path(__file__).resolve().parents[2]
    / "artifacts/data/code_he_clean/val.parquet"
)


def _load_code_taco_reward():
    import importlib.util
    from pathlib import Path as _Path

    path = _Path(__file__).resolve().parents[1] / "reward" / "code_taco_reward.py"
    spec = importlib.util.spec_from_file_location("_code_taco_reward_for_eval", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class HumanEval:
    name = "humaneval"
    gen_length = 8192

    def load(self, n: int, start: int = 0) -> list[dict]:
        import json
        import os

        import pyarrow.parquet as pq

        path = os.environ.get(HUMANEVAL_PARQUET_ENV, HUMANEVAL_PARQUET_DEFAULT)
        rows = pq.read_table(path).to_pylist()
        end = min(start + n, len(rows))
        items = []
        for r in rows[start:end]:
            prompt = (
                r["prompt"][0]["content"]
                if isinstance(r["prompt"], list)
                else r["prompt"]
            )
            gt = r["reward_model"]["ground_truth"]
            if isinstance(gt, str):
                gt = json.loads(gt)
            meta = {"entry_point": gt.get("entry_point", "")}
            items.append({"prompt": prompt, "gold": gt, "meta": meta})
        return items

    def score(self, output_text: str, item: dict) -> bool:
        mod = _load_code_taco_reward()
        result = mod._code_score(output_text or "", item["gold"])
        return bool(result.get("acc"))


# ---------------------------------------------------------------------------
# MBPP-500 — google-research-datasets/mbpp "full" test split.
# Same code_taco functional harness as HumanEval. Not EvalPlus's 378 set.
# ---------------------------------------------------------------------------

MBPP_DATASET = "google-research-datasets/mbpp"
MBPP_REVISION = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"


class MBPP:
    name = "mbpp"
    gen_length = 8192

    def load(self, n: int, start: int = 0) -> list[dict]:
        import re

        from datasets import load_dataset  # deferred

        ds = load_dataset(MBPP_DATASET, "full", split="test", revision=MBPP_REVISION)
        end = min(start + n, len(ds))
        items = []
        for item in ds.select(range(start, end)):
            tests = "\n".join(item["test_list"])
            prompt = (
                "You are an expert Python programmer. Write a solution to the "
                "following task, and return it in a single ```python code "
                "block.\n\n"
                f"Task: {item['text']}\n\n"
                f"Your code must pass these tests:\n\n{tests}"
            )
            defs = re.findall(r"def\s+([A-Za-z_]\w*)\s*\(", item["code"] or "")
            joined = tests
            ep = None
            for d in defs:
                if re.search(rf"\b{re.escape(d)}\s*\(", joined):
                    ep = d
                    break
            if ep is None:
                m = re.search(r"assert\s+\(?\s*([A-Za-z_]\w*)\s*\(", joined)
                ep = m.group(1) if m else "solution"
            lines = []
            setup = (item.get("test_setup_code") or "").rstrip()
            if setup:
                lines.append(setup)
            lines.append("def check(candidate):")
            for t in item["test_list"]:
                for ln in str(t).rstrip().split("\n"):
                    lines.append("    " + ln)
            gold = {"functional": "\n".join(lines), "entry_point": ep}
            items.append(
                {"prompt": prompt, "gold": gold, "meta": {"task_id": item["task_id"]}}
            )
        return items

    def score(self, output_text: str, item: dict) -> bool:
        import re

        mod = _load_code_taco_reward()
        body = output_text or ""
        if "</think>" in body:
            body = body.split("</think>")[-1]
        segs = [s.strip() for s in re.split(r"`{3,}", body) if s.strip()]
        cands = []
        for s in segs:
            t = re.sub(
                r"^(?:python3|python|py)[ \t]*\n?", "", s, count=1, flags=re.I
            ).strip()
            for v in (t, s):
                if v and "def " in v and v not in cands:
                    cands.append(v)
        if not cands and "def " in body:
            cands = [body.strip()]
        for cand in cands[:8]:
            result = mod._code_score("```python\n" + cand + "\n```", item["gold"])
            if result.get("acc"):
                return True
        return False

