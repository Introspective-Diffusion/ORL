#!/usr/bin/env python3
"""build_code_level.py — build the code training set of one level tier.

Deterministic pipeline (source dataset with train and
bench parquet files, prompt_tokens <=1900 gate, functional check suite from
tests, he_clean preamble wrap, contamination counters vs bench + he_clean), with
the level filter and output dir as CLI args. task_id = "<LEVEL>/train/<i>".

Usage: build_code_level.py --source DATASET --level L0 --out DIR --he-clean HE_CLEAN_TRAIN_PARQUET
"""
import argparse, json, re, hashlib
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
from huggingface_hub import snapshot_download

ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True,
                help="source code-problem dataset: Hugging Face dataset id or local directory with train/bench parquet files")
ap.add_argument("--level", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--he-clean", required=True, help="code_he_clean/train.parquet (contamination reference)")
args = ap.parse_args()
LEVEL = args.level
OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
REP = {"level": LEVEL}

snap = Path(args.source) if Path(args.source).is_dir() else Path(snapshot_download(args.source, repo_type="dataset"))
files = sorted(snap.rglob("*.parquet"))
train_f = [f for f in files if "train" in f.name or "train" in str(f.parent)]
bench_f = [f for f in files if "bench" in f.name or "bench" in str(f.parent)]
assert train_f and bench_f, (train_f, bench_f)
tr = pq.read_table(train_f[0]).to_pylist()
be = pq.read_table(bench_f[0]).to_pylist()
REP["train_rows_total"] = len(tr)

sel = [r for r in tr if str(r.get("level")) == LEVEL]
REP["level_rows"] = len(sel)
assert sel, f"no rows with level=={LEVEL}"

he = pq.read_table(args.he_clean).to_pylist()
he_prompt = he[0]["prompt"][0]["content"]
preamble = he_prompt.split("\n\n")[0]
needs_wrap = "```python" not in sel[0]["prompt"]
REP["needs_wrap"] = needs_wrap

def build_functional(tests_raw):
    tests = json.loads(tests_raw) if isinstance(tests_raw, str) else tests_raw
    if isinstance(tests, dict): tests = tests.get("asserts") or tests.get("tests") or []
    lines = ["def check(candidate):"]
    for t in tests:
        t = str(t).rstrip()
        for ln in t.split("\n"):
            lines.append("    " + ln)
    if len(lines) == 1: lines.append("    pass")
    return "\n".join(lines)

out, dropped, badtests = [], 0, 0
for i, r in enumerate(sel):
    pt = r.get("prompt_tokens") or 0
    if pt and pt > 1900:
        dropped += 1; continue
    try:
        functional = build_functional(r["tests"])
    except Exception:
        badtests += 1; continue
    content = r["prompt"] if not needs_wrap else (preamble + "\n\n" + r["prompt"])
    gt = json.dumps({"functional": functional, "entry_point": r["entry_point"]})
    out.append({
        "data_source": "humaneval",
        "prompt": [{"role": "user", "content": content}],
        "ability": "code",
        "reward_model": {"style": "rule", "ground_truth": gt},
        "extra_info": {"split": "train", "index": str(i), "task_id": "%s/train/%d" % (LEVEL, i),
                       "reference_solution": str(r.get("solution", ""))[:4000]},
    })
REP["converted"] = len(out); REP["dropped_token"] = dropped; REP["bad_tests"] = badtests

def norm(s): return re.sub(r"\s+", " ", str(s)).strip().lower()
sel_ph = {hashlib.md5(norm(r["prompt"]).encode()).hexdigest() for r in sel}
be_ph = {hashlib.md5(norm(r["prompt"]).encode()).hexdigest() for r in be}
he_ph = {hashlib.md5(norm(h["prompt"][0]["content"]).encode()).hexdigest() for h in he}
sel_ep = {str(r.get("entry_point")) for r in sel}
be_ep = {str(r.get("entry_point")) for r in be}
he_ep = set()
for h in he:
    m = re.search(r"def\s+(\w+)\s*\(", h["prompt"][0]["content"])
    if m: he_ep.add(m.group(1))
REP["contam_vs_bench_prompt"] = len(sel_ph & be_ph)
REP["contam_vs_heclean_prompt"] = len(sel_ph & he_ph)
REP["contam_vs_bench_entrypoint"] = len(sel_ep & be_ep)
REP["contam_vs_heclean_entrypoint"] = len(sel_ep & he_ep)

pq.write_table(pa.Table.from_pylist(out), OUT / "train.parquet")
h = hashlib.sha256(open(OUT / "train.parquet", "rb").read()).hexdigest()
REP["output_sha256"] = h
(Path(OUT) / "manifest.json").write_text(json.dumps(REP, indent=2))
print("REPORT:", json.dumps(REP, ensure_ascii=False))
print("BUILD-%s-DONE" % LEVEL)
