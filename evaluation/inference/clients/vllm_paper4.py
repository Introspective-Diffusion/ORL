#!/usr/bin/env python3
"""Run the locked DiffusionGemma four-benchmark cell against a vLLM DiffusionGemma server and write rows in the
multi_gen shard schema so `score_paper4.py --entity <out>` scores them exactly like our sglang cells.

Prompts are built exactly as dgx_runtime/multi_gen.py renders them (chat messages for gsm8k/math500; for HumanEval/MBPP a
system message "You are a helpful assistant good at coding." plus the INSTRUCTION_PREFIX + fenced code block), and handed to
/v1/chat/completions so the server applies the same chat_template.jinja. Per-dataset max_tokens = the locked plan
(gsm8k 4096, he 2048, mbpp 2048, math500 4096); temperature 0; one request at a time.

Forward accounting from the Prometheus diffusion counters: the vLLM counter called
`diffusion_num_denoising_steps_total` includes one commit step per canvas. Therefore:
`n_blocks = committed_canvas_tokens / 256`, `fwd_denoise = counter_steps - n_blocks`,
`fwd_commit = n_blocks`. The paper's TPF uses `fwd_denoise` (`counter_steps - n_blocks`).

    python3 vllm_paper4.py --assets ./inputs --gsm ./inputs/gsm8k.jsonl --out ./outputs/paper4
Resumable: task_ids already present in <out>/<name>/shard_w0.jsonl are skipped.
"""
import argparse, json, os, re, time, urllib.request

INSTRUCTION_PREFIX = ("Please provide a self-contained Python script that solves "
                      "the following problem in a markdown code block:")
SYSTEM_MSG = "You are a helpful assistant good at coding."
PLAN = [("gsm8k", "gsm", 4096), ("he", "humaneval.jsonl", 2048), ("mbpp", "mbpp.jsonl", 2048), ("math500", "math500.jsonl", 4096)]
COUNTERS = ("vllm:diffusion_num_denoising_steps_total", "vllm:diffusion_num_committed_tokens_total")


def metrics(base):
    txt = urllib.request.urlopen(base + "/metrics", timeout=30).read().decode()
    out = []
    for name in COUNTERS:
        m = re.findall(rf"^{re.escape(name)}(?:{{[^}}]*}})?\s+([0-9.e+]+)", txt, re.M)
        out.append(sum(float(x) for x in m) if m else 0.0)
    return out


def render(p):
    if p.get("messages"):
        assert not p.get("assistant_prefill"), "assistant_prefill is not supported over the chat API"
        return [dict(m) for m in p["messages"]]
    return [{"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": INSTRUCTION_PREFIX + f"\n```python\n{p['prompt'].strip()}\n```"}]


def done_ids(path):
    ids = set()
    if os.path.exists(path):
        for l in open(path):
            if l.strip():
                ids.add(json.loads(l)["task_id"])
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000"); ap.add_argument("--model", default="dg")
    ap.add_argument("--assets", required=True); ap.add_argument("--gsm", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234,
                    help="Per-request diffusion seed (requires the HF-alignment vLLM patch to seed before init_canvas)")
    ap.add_argument("--max-denoising-steps", type=int)
    ap.add_argument("--only", default=None, help="comma list of dataset names to run")
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None
    for name, src, max_new in PLAN:
        if only and name not in only:
            continue
        problems_path = a.gsm if src == "gsm" else os.path.join(a.assets, src)
        probs = [json.loads(l) for l in open(problems_path) if l.strip()]
        out_dir = os.path.join(a.out, name); os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "shard_w0.jsonl")
        have = done_ids(out_path); todo = [p for p in probs if p["task_id"] not in have]
        print(f"[{name}] n={len(probs)} already={len(have)} todo={len(todo)} max_new={max_new}", flush=True)
        tot_tok = tot_wall = tot_steps = 0.0
        with open(out_path, "a") as out:
            for i, p in enumerate(todo):
                s0, c0 = metrics(a.base)
                body = json.dumps({"model": a.model, "messages": render(p), "temperature": 0.0,
                                   "max_tokens": max_new, "seed": a.seed}).encode()
                req = urllib.request.Request(a.base + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
                t0 = time.perf_counter(); resp = json.loads(urllib.request.urlopen(req, timeout=3600).read()); dt = time.perf_counter() - t0
                s1, c1 = metrics(a.base)
                counter_steps = int(round(s1 - s0)); committed = int(round(c1 - c0))
                assert committed > 0 and committed % 256 == 0, (
                    p["task_id"], committed
                )
                n_blocks = committed // 256
                denoise_forwards = counter_steps - n_blocks
                assert denoise_forwards > 0, (p["task_id"], counter_steps, n_blocks)
                if a.max_denoising_steps is not None:
                    assert denoise_forwards <= n_blocks * a.max_denoising_steps, (
                        p["task_id"], denoise_forwards, n_blocks,
                        a.max_denoising_steps,
                    )
                ch = resp["choices"][0]; ct = resp["usage"]["completion_tokens"]
                fr = "length" if ch["finish_reason"] == "length" else "stop"
                row = {"task_id": p["task_id"], "entry_point": p.get("entry_point"), "raw": ch["message"]["content"] or "",
                       "finish": fr, "completion_tokens": ct, "wall_s": round(dt, 4)}
                for k in ("gold", "meta"):
                    if k in p: row[k] = p[k]
                row.update({"n_blocks": n_blocks, "gen_tok": ct, "fwd_denoise": denoise_forwards,
                            "fwd_commit": n_blocks, "fwd_total": counter_steps,
                            "fwd_nocommit": denoise_forwards,
                            "vllm_counter_steps": counter_steps, "committed_canvas_tokens": committed,
                            "completion_tpf": ct / denoise_forwards,
                            "canvas_tpf_noedit": committed / denoise_forwards,
                            "all_forward_tpf": ct / counter_steps,
                            "completion_tps": ct / dt,
                            "vllm_finish_reason": ch["finish_reason"], "seed": a.seed})
                out.write(json.dumps(row) + "\n"); out.flush()
                tot_tok += ct; tot_wall += dt; tot_steps += denoise_forwards
                if i % 25 == 0 or i == len(todo) - 1:
                    print(f"  [{name}] {i+1}/{len(todo)} {p['task_id']} finish={fr} tok={ct} "
                          f"denoise={denoise_forwards} counter={counter_steps} "
                          f"pooledTPS={tot_tok/max(tot_wall,1e-9):.0f} TPF={tot_tok/max(tot_steps,1):.2f}", flush=True)
        print(f"[{name}] DONE rows={len(done_ids(out_path))}", flush=True)
    open(os.path.join(a.out, "GEN_DONE"), "w").write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    print("GEN_DONE", flush=True)


if __name__ == "__main__":
    main()
