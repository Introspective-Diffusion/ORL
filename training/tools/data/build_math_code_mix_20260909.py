#!/usr/bin/env python3
"""GSM8K + MATH + code_l0 training mix (2026-09-09), 1:1:1 by rows, shuffled with a fixed seed.

    python3 build_math_code_mix_20260909.py GSM8K_MATH_MIX_PARQUET CODE_L0_1ASSERT_PARQUET OUT_DIR

Sources:
    gsm8k / math : gsm8k-math-mix-shuffled-20260824/train.parquet  (MATH + GSM8K 1:1 mix;
                   data_source "openai/gsm8k" / "DigitalLearningGmbH/MATH-lighteval", '####' / \\boxed{} prompts)
    code         : code_l0_1assert-20260906/train.parquet          (819 code_l0 problems, one visible assert,
                   data_source "humaneval", hidden tests in reward_model.ground_truth)
The code set is the smallest (819), so 819 rows are sampled from each of gsm8k and math -> 2457 rows. Reward routing:
training/rewards/math_code_mix.py. Writes train.parquet + manifest.json (seed, counts, sha256).
"""
import hashlib, json, os, sys
import pandas as pd

SEED = 20260725


def main(MX, CODE, out_dir):
    mx = pd.read_parquet(MX); code = pd.read_parquet(CODE)
    n = len(code)
    gsm = mx[mx.data_source.str.contains("gsm8k")].sample(n=n, random_state=SEED)
    math = mx[~mx.data_source.str.contains("gsm8k")].sample(n=n, random_state=SEED + 1)
    df = pd.concat([gsm, math, code], ignore_index=True).sample(frac=1.0, random_state=SEED + 2).reset_index(drop=True)
    # extra_info structs differ per source; normalise to one string-valued dict so parquet gets a single struct type
    def norm_extra(e):
        d = dict(e) if e is not None else {}
        return {"split": str(d.get("split", "train")), "index": str(d.get("index", "")), "domain": str(d.get("domain", "")),
                "task_id": str(d.get("task_id", "")), "reference_solution": str(d.get("reference_solution", ""))}
    df["extra_info"] = df["extra_info"].map(norm_extra)
    df["prompt"] = df["prompt"].map(lambda p: [dict(m) for m in p])   # numpy arrays -> lists
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "train.parquet"); df.to_parquet(out, index=False)
    counts = df.data_source.value_counts().to_dict()
    man = {"seed": SEED, "rows": len(df), "counts": counts, "sources": {"gsm8k+math": MX, "code": CODE},
           "sha256": hashlib.sha256(open(out, "rb").read()).hexdigest(), "reward": "training/rewards/math_code_mix.py"}
    json.dump(man, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
    print(json.dumps(man, indent=1))
    # schema check: every row has the fields the trainer/reward need
    back = pd.read_parquet(out)
    assert list(back.columns) == ["data_source", "prompt", "ability", "reward_model", "extra_info"], list(back.columns)
    assert all(isinstance(r["ground_truth"], str) and r["ground_truth"] for r in back.reward_model), "empty ground_truth"
    print("first rows:", [(r.data_source, r.ability) for r in back.head(6).itertuples()])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
