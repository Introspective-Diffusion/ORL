#!/usr/bin/env python3
"""Build the two 2026-09-06 code RL corpora in the *eval-aligned, one-visible-test* format.

Prompt (identical to what the HumanEval/MBPP evaluation sends: SYSTEM_MSG + INSTRUCTION_PREFIX,
problem wrapped in a ```python fence; MBPP+ shape = docstring with the description + ONE assert):

    system: You are a helpful assistant good at coding.
    user:   Please provide a self-contained Python script that solves the following problem in a markdown code block:
            ```python
            \"\"\"
            <task description>
            <first assert>
            \"\"\"
            ```

Hidden tests (reward): ALL asserts (the visible one included, as in MBPP+) as `def check(candidate)`.

Sources:
  code_l0   code_l0-20260828/train.parquet (819 rows). Description = the text between
            "Task:" and "Your code must pass these tests"; tests = the row's existing `functional`.
  oci5k     nvidia/OpenCodeInstruct shard data/train-00000-of-00050.parquet (100k rows), a plain random
            sample of 5000 rows (seed 20260906), NO quality/difficulty selection (as requested);
            description = `input` verbatim; tests = the 10 LLM-generated `unit_tests`.

Output parquet schema mirrors code_l0 exactly (data_source, prompt[list<{role,content}>], ability,
reward_model{style, ground_truth: JSON string}, extra_info{split,index,task_id,reference_solution}).
"""
import argparse, json, os, random, re
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# The HumanEval/MBPP evaluation prompt.
INSTRUCTION_PREFIX = ("Please provide a self-contained Python script that solves "
                      "the following problem in a markdown code block:")
SYSTEM_MSG = "You are a helpful assistant good at coding."

SCHEMA = pa.schema([
    ("data_source", pa.string()),
    ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
    ("ability", pa.string()),
    ("reward_model", pa.struct([("style", pa.string()), ("ground_truth", pa.string())])),
    ("extra_info", pa.struct([("split", pa.string()), ("index", pa.string()), ("task_id", pa.string()),
                              ("reference_solution", pa.string())])),
])


FENCE_LINE = re.compile(r"^\s*```[A-Za-z0-9_+-]*\s*$", re.M)


def render_user(description: str, visible_assert: str) -> str:
    # same layout as the evaluation prompt: INSTRUCTION_PREFIX + "\n```python\n" + prompt.strip() + "\n```"
    # The source descriptions carry their own ``` fences around Sample Input/Output (722/819 code_l0,
    # 4392/5000 OCI rows); nested inside the outer ```python fence they would close it early, and the
    # HE/MBPP+ eval prompts have no inner fences. Drop the fence LINES only -- the sample content stays
    # as plain text under its **Sample Input/Output:** header.
    description = FENCE_LINE.sub("", description)
    description = re.sub(r"\n{3,}", "\n\n", description)
    problem = f'"""\n{description.strip()}\n{visible_assert.strip()}\n"""'
    return INSTRUCTION_PREFIX + f"\n```python\n{problem.strip()}\n```"


def row(data_source, description, visible_assert, functional, entry_point, index, task_id, reference):
    return {
        "data_source": data_source,
        "prompt": [{"role": "system", "content": SYSTEM_MSG},
                   {"role": "user", "content": render_user(description, visible_assert)}],
        "ability": "code",
        "reward_model": {"style": "rule", "ground_truth": json.dumps({"functional": functional, "entry_point": entry_point})},
        "extra_info": {"split": "train", "index": str(index), "task_id": task_id, "reference_solution": reference},
    }


def first_assert_from_check(functional: str) -> str:
    for line in functional.splitlines():
        s = line.strip()
        if s.startswith("assert "):
            return s
    raise ValueError("no assert in functional check")


def build_code_l0(src, out):
    df = pd.read_parquet(src)
    rows = []
    for i, r in df.iterrows():
        content = r["prompt"][0]["content"]
        assert "Task:" in content and "Your code must pass these tests" in content, i
        desc = content.split("Task:", 1)[1].split("Your code must pass these tests", 1)[0].strip()
        gt = json.loads(r["reward_model"]["ground_truth"])
        rows.append(row("humaneval", desc, first_assert_from_check(gt["functional"]), gt["functional"],
                        gt.get("entry_point"), r["extra_info"]["index"], r["extra_info"]["task_id"],
                        r["extra_info"]["reference_solution"]))
    write(rows, out)
    return rows


def build_oci(src, out, n, seed):
    df = pd.read_parquet(src)
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(df)), n))
    rows, dropped = [], 0
    for i in idx:
        r = df.iloc[i]
        try:
            tests = json.loads(r["unit_tests"])
        except Exception:
            dropped += 1; continue
        tests = [t.strip() for t in tests if t and t.strip()]
        if not tests:
            dropped += 1; continue
        functional = "def check(candidate):\n" + "\n".join(
            "\n".join("    " + ln for ln in t.splitlines()) for t in tests)
        rows.append(row("opencodeinstruct", r["input"], tests[0], functional, None, i, f"OCI/shard0/{r['id']}", r["output"]))
    write(rows, out)
    print(f"oci: sampled {n}, dropped {dropped} (unparseable/empty unit_tests), wrote {len(rows)}")
    scores = pd.to_numeric(df.iloc[idx]["average_test_score"], errors="coerce")
    print(f"oci: reference solution passes all its tests in {(scores == 1.0).mean():.1%} of sampled rows "
          f"(>=0.8: {(scores >= 0.8).mean():.1%}); kept regardless (no selection requested)")
    return rows


def write(rows, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    pq.write_table(table, out)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--code_l0", required=True)
    ap.add_argument("--oci_shard", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--n_oci", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260906)
    a = ap.parse_args()
    l0 = build_code_l0(a.code_l0, f"{a.out_root}/code_l0_1assert-20260906/train.parquet")
    oci = build_oci(a.oci_shard, f"{a.out_root}/oci5k_1assert-20260906/train.parquet", a.n_oci, a.seed)
    print("\n=== sample prompt (code_l0) ===\n" + l0[0]["prompt"][1]["content"][:900])
    print("\n=== sample prompt (oci) ===\n" + oci[0]["prompt"][1]["content"][:900])
