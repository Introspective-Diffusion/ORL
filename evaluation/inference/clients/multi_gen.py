#!/usr/bin/env python3
"""Persistent-engine generation worker: ONE sglang engine per GPU serves SEVERAL datasets of one
(checkpoint, decode yaml) job, with node-local dynamic row dispatch and resume.

Why: the per-(dataset) job design spent ~2/3 of every job starting the engine
(container + imports + weights + init ~1.2 min per shard, 10 s stagger, static-shard tail, CPU
scoring with idle GPUs). Building the engine once per (ckpt, decoder) and streaming all datasets
through it removes 3 of the 5 engine starts per checkpoint and the shard tail.

Contract (identical rows to humaneval_gen.py, so humaneval_score.py and every table script work):
  * prompt rendering, engine construction (mem_fraction 0.60, graphs [1,2], radix cache off),
    temperature written into a per-process yaml copy, row schema, forward counters -- unchanged.
  * per-dataset max_new_tokens through sampling_params (honoured by sglang core; the algorithm only
    ignores temperature).
Dispatch: every worker on the node walks the same problem list and claims a row with
os.mkdir(CLAIM_DIR/<dataset>/<task_id>) (node-local tmpfs, atomic). Rows already present in
<out_dir>/shard*.jsonl (previous run, seeded rows) are skipped -> a retried job only fills gaps.
Each worker appends to <out_dir>/shard_w<WORKER>.jsonl, one json per line, flushed per row.

Env: ORL_ENGINE_ROOT, MODEL_DIR, CFG, TEMP (0), ALGO (SDARPrefixKV), PLAN (json: {"datasets":[{"name","problems","max_new","out_dir"}]}),
     WORKER (int), CLAIM_DIR, SDAR_TRUE_FWD_COUNTER_PATH (optional), ENGINE_RETRIES (3).
"""
import glob, json, os, re, sys, time, traceback

R = os.environ["ORL_ENGINE_ROOT"]
INSTRUCTION_PREFIX = ("Please provide a self-contained Python script that solves "
                      "the following problem in a markdown code block:")
SYSTEM_MSG = "You are a helpful assistant good at coding."


def read_counter(path):
    if not os.path.exists(path):
        return (0, 0)
    try:
        with open(path) as f:
            vals = [int(x) for x in f.read().split()]
        return (vals[0], vals[1] if len(vals) > 1 else 0)
    except Exception:
        return None


def log(msg):
    print(f"[multi_gen w{os.environ.get('WORKER', '?')} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def done_ids(out_dir):
    ids = set()
    for f in glob.glob(os.path.join(out_dir, "shard*.jsonl")):
        try:
            with open(f) as fh:
                for l in fh:
                    l = l.strip()
                    if l:
                        ids.add(json.loads(l)["task_id"])
        except Exception as e:   # a half-written line from a concurrent worker is not fatal
            log(f"warn: reading {f}: {e}")
    return ids


def main():
    if os.environ.get("ALGO", "SDARPrefixKV") == "SDARPrefixKV":
        # LLaDA2 / SDAR: the idlm-streaming sglang tree. Other algorithms (Gemma4Renoise) come from a separate fork
        # given on PYTHONPATH by the caller; inserting the idlm tree first would shadow it and lose its HF registrations.
        sys.path.insert(0, f"{R}/kernels/fast-diffusion-attention")
        sys.path.insert(0, f"{R}/inference/sglang")
    import sglang as sgl
    from transformers import AutoTokenizer

    algo = os.environ.get("ALGO", "SDARPrefixKV")
    cfg_path = os.environ["CFG"]
    M = os.environ["MODEL_DIR"]
    temp = float(os.environ.get("TEMP", "0"))
    worker = int(os.environ["WORKER"])
    claim_root = os.environ["CLAIM_DIR"]
    plan = json.load(open(os.environ["PLAN"]))
    counter = (os.environ.get("SDAR_TRUE_FWD_COUNTER_PATH", "") if algo == "SDARPrefixKV"
               else os.environ.get("TPF_COUNTER_PATH", ""))

    tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)

    def render(p):
        if p.get("messages"):
            msgs = [dict(m) for m in p["messages"]]
            if p.get("assistant_prefill"):
                magic = "-[[]]-this-is-really-our-highest-priority-[[]]-"
                full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": p["assistant_prefill"] + magic}], tokenize=False)
                return full.split(magic)[0]
        else:
            msgs = [{"role": "system", "content": SYSTEM_MSG},
                    {"role": "user", "content": INSTRUCTION_PREFIX + f"\n```python\n{p['prompt'].strip()}\n```"}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    if algo == "SDARPrefixKV":
        import yaml, tempfile
        with open(cfg_path) as fh:
            d = yaml.safe_load(fh)
        d["temperature"] = temp; d["temperature_max"] = temp; d["temperature_min"] = temp
        cfg_path = os.path.join(tempfile.gettempdir(), f"mg_cfg_T{temp}_{os.getpid()}_{os.path.basename(cfg_path)}")
        with open(cfg_path, "w") as fh:
            yaml.safe_dump(d, fh)
        assert yaml.safe_load(open(cfg_path))["temperature"] == temp

    eng = None
    for attempt in range(1, int(os.environ.get("ENGINE_RETRIES", "3")) + 1):
        try:
            # TP / MEM_FRAC / ATTN_BACKEND are env-overridable for other dLLMs (DiffusionGemma 26B: TP=1 needs
            # MEM_FRAC>=0.85 on 80 GB; its server args force triton attention + no cuda graph anyway)
            eng = sgl.Engine(model_path=M, trust_remote_code=True, tp_size=int(os.environ.get("TP", "1")),
                             mem_fraction_static=float(os.environ.get("MEM_FRAC", "0.60")), disable_cuda_graph=False, sdar_rollout_cuda_graph=True, sdar_rollout_cuda_graph_capture_sizes=[1, 2],
                             log_level="error", attention_backend=os.environ.get("ATTN_BACKEND", "flashinfer"),
                             disable_radix_cache=True,
                             dllm_algorithm=algo, dllm_algorithm_config=cfg_path)
            break
        except Exception:
            traceback.print_exc()
            log(f"engine start failed (attempt {attempt}); retrying in 30 s")
            time.sleep(30)
    if eng is None:
        log("ENGINE_FAILED"); sys.exit(3)
    log(f"engine up model={M} cfg={os.path.basename(cfg_path)} datasets={[d['name'] for d in plan['datasets']]}")

    for ds in plan["datasets"]:
        name, out_dir, maxnew = ds["name"], ds["out_dir"], int(ds["max_new"])
        with open(ds["problems"]) as f:
            probs = [json.loads(l) for l in f if l.strip()]
        have = done_ids(out_dir)
        todo = [p for p in probs if p["task_id"] not in have]
        cdir = os.path.join(claim_root, name); os.makedirs(cdir, exist_ok=True)
        out_path = os.path.join(out_dir, f"shard_w{worker}.jsonl")
        sp = {"max_new_tokens": maxnew, "temperature": temp}
        log(f"[{name}] n={len(probs)} already={len(have)} todo={len(todo)} max_new={maxnew}")
        n_mine = term = 0
        out = None                      # opened on first claimed row (no empty shard files)
        try:
            for p in todo:
                key = re.sub(r"[^A-Za-z0-9_.-]", "_", str(p["task_id"]))
                try:
                    os.mkdir(os.path.join(cdir, key))
                except FileExistsError:
                    continue            # another worker on this node took it
                if out is None:
                    out = open(out_path, "a")
                before = read_counter(counter) if counter else None
                t0 = time.time()
                o = eng.generate(render(p), sp)
                dt = time.time() - t0
                after = read_counter(counter) if counter else None
                mi = o.get("meta_info", {}) or {}
                fr = mi.get("finish_reason"); fr = fr.get("type") if isinstance(fr, dict) else fr
                term += (fr == "stop")
                row = {"task_id": p["task_id"], "entry_point": p.get("entry_point"), "raw": o.get("text", ""),
                       "finish": fr, "completion_tokens": mi.get("completion_tokens"), "wall_s": round(dt, 2)}
                for k in ("gold", "meta"):
                    if k in p: row[k] = p[k]
                rm = mi.get("sglang_sdar_request_metrics")
                recs = rm if isinstance(rm, list) else [rm] if isinstance(rm, dict) else []
                if recs:
                    agg = {}
                    for rec in recs:
                        for k, v in rec.items():
                            if isinstance(v, (int, float)) and not isinstance(v, bool):
                                agg[k] = agg.get(k, 0) + v
                    if agg.get("total_forwards") is None and agg.get("denoise_forwards") is not None:
                        # Gemma4Renoise records denoise/commit per canvas but no total: total = denoise + commit
                        agg["total_forwards"] = agg.get("denoise_forwards", 0) + agg.get("commit_forwards", 0)
                    row.update({"n_blocks": len(recs), "gen_tok": agg.get("generated_tokens"),
                                "fwd_total": agg.get("total_forwards"), "fwd_denoise": agg.get("denoise_forwards"),
                                "fwd_refine": agg.get("refine_forwards"), "fwd_commit": agg.get("commit_forwards")})
                    if agg.get("denoise_forwards") is not None:
                        row["fwd_nocommit"] = int(agg.get("denoise_forwards") or 0)
                if before is not None and after is not None:
                    row.update({"fwd_total_counter": after[0] - before[0], "n_blocks_counter": after[1] - before[1]})
                out.write(json.dumps(row) + "\n"); out.flush()
                n_mine += 1
                print(f"  [{name}] {p['task_id']} finish={fr} tok={row['completion_tokens']} "
                      f"fwd={row.get('fwd_total', row.get('fwd_total_counter'))} {dt:.1f}s", flush=True)
        finally:
            if out is not None:
                out.close()
        log(f"[{name}] DONE mine={n_mine} terminated={term}/{n_mine} out={out_path}")
    log("ALL_DATASETS_DONE")
    eng.shutdown()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
