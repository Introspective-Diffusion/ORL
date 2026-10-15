#!/usr/bin/env python3
"""Pinned transfer generation: local inputs/runtime/weights, no implicit scoring."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlparse

from repro.paper.plan import ROOT, expand
from repro.eval.portable.build_recipe import inventory
from ._paths import contained

INPUTS = {
    "aime": (60, "ec677f2990d5a0dd78e4d631d0dca03ea29f9e0c8103141be6f065cd031018b5"),
    "gpqa": (198, "7cf55ae09e627820d492577114d0b7838082869f004df5fe48b875002af2034c"),
    "mmlu_pro": (12032, "8f10a6eb35f6279f2a0a30bf4631551d8c12b5c55a2f8d245662fb624c4902cb"),
    "lcb": (175, "26f01f74d4b0e8d74d7f97f482a2456b181c3bc3f289677831399b82dc0c3b8e"),
    "ifeval": (541, "69c2d38e1f5b0e22376f7f3f9184de9ccbdf5e2061fb74bb4c55541fb8f3ca60"),
}
CLIENTS = {
    "llada": ("multi_gen.py", "105f676df662ac5cbd293f3bb9fb674fe2f57cac26764709cef3c9e946be6f41"),
    "dg": ("vllm_full_extra_client.py", "dc864eba1f2e88d24b404bdf8711aa63d7eb66bd07daadba10fdfd8526c25d1b"),
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def require(value, message):
    if not value:
        raise ValueError(message)


def read_inputs(path, benchmark):
    raw = path.read_bytes()
    n, expected = INPUTS[benchmark]
    require(sha(raw) == expected, "Frozen full input SHA256 mismatch; do not substitute prompts or pre-shard")
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    require(len(rows) == n, "Wrong full benchmark count")
    require(len({r["task_id"] for r in rows}) == n, "Duplicate input task IDs")
    require(all(isinstance(r.get("messages"), list) and r["messages"] for r in rows), "Frozen messages required")
    if benchmark == "aime":
        require(all(sum(r.get("meta", {}).get("year") == year for r in rows) == 30
                    for year in ("aime24", "aime25")), "AIME requires both frozen 30-task years")
    return rows, expected


def validate_runtime(runtime, family):
    name, expected = CLIENTS[family]
    path = contained(runtime, name)
    require(sha(path.read_bytes()) == expected, "Pinned native client source mismatch")
    if family == "llada":
        manifest_path = ROOT / "runtime/sglang/source_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        engine = (runtime / "engine").resolve()
        require(engine.is_relative_to(runtime.resolve()) and engine.is_dir(), "Missing/escaping engine directory")
        receipt = json.loads(contained(engine, "engine_manifest.json").read_text())
        require(receipt["profile"] == "llada" and receipt["engine_revision"] == manifest["base_revision"],
                "Wrong engine profile/revision")
        require(receipt["source_manifest_sha256"] == sha(manifest_path.read_bytes()), "Wrong engine source manifest")
        for relative, record in manifest["files"].items():
            if record["kind"] == "symlink":
                require(relative == manifest["nonruntime_format_symlink"], "Unexpected engine symlink")
                continue
            source = contained(engine, relative)
            require(sha(source.read_bytes()) == record["sha256"], "Engine source mismatch: " + relative)
    return path


def model_identity(model):
    require((model / "config.json").is_file(), "Local model config.json required")
    files = inventory(model, "model")
    require(any(r["path"].endswith(".safetensors") for r in files), "Local model weight files required")
    return {"files": files, "sha256": sha(canonical(files))}


def validate_server(receipt, parameters, role, base, model_sha):
    url = urlparse(base)
    require(url.scheme == "http" and url.hostname in ("localhost", "127.0.0.1", "::1")
            and not url.username and not url.password and not url.query and not url.fragment
            and url.path in ("", "/"), "DG server must be a credential-free local HTTP endpoint")
    decoder = parameters["decoder"]
    overrides = {k: v for k, v in decoder.items() if k not in ("canvas_length", "max_denoising_steps", "sampler_patch")}
    expected = {"schema_version": 1, "family": "dg", "role": role,
                "image_digest": parameters["engine"]["image"], "model_manifest_sha256": model_sha,
                "diffusion_config": {"canvas_length": 256, "max_denoising_steps": 48},
                "max_model_len": parameters["engine"]["context_length"], "seed_protocol": "reset_each_request",
                "sampler_role": "alignment" if role == "base" else "adaptive_scalar",
                "sampling_overrides": overrides, "gpu_memory_utilization": 0.95,
                "attention_backend": "TRITON_ATTN", "base_url": base.rstrip("/")}
    for key, value in expected.items():
        require(key in receipt and type(receipt[key]) is type(value) and receipt[key] == value,
                "DG server receipt mismatch: " + key)
    require(receipt.get("server_started") is True and receipt.get("server_ready") is True,
            "DG receipt must describe a started and ready server, not a dry-run")
    return expected


def read_server_receipt(path):
    require(not path.with_name("server_exit.json").exists(), "DG server has a recorded exit; start a new server")
    return json.loads(path.read_text())


def prepare(args):
    planned = expand("transfer", [args.model_id], [args.benchmark])["runs"]
    matches = [r for r in planned if r["seed"] == args.seed]
    require(len(matches) == 1, "Seed not in the validated benchmark/model matrix")
    run = matches[0]
    require(type(args.shard_index) is int and 0 <= args.shard_index < len(run["shards"]), "Invalid shard index")
    rows, source_sha = read_inputs(args.problems, args.benchmark)
    shard = run["shards"][args.shard_index]
    selected = [rows[i] for i in shard["input_positions"]]
    if args.smoke_limit is not None:
        require(type(args.smoke_limit) is int and 0 < args.smoke_limit <= len(selected), "Invalid smoke limit")
        selected = selected[:args.smoke_limit]
    require(not args.out.exists() and not args.out.is_symlink(), "Use a new output directory; no resume or overwrite")
    client = validate_runtime(args.runtime, run["family"])
    require(args.model.is_dir() and (args.model / "config.json").is_file(), "Supply a local model directory")
    if run["family"] == "dg":
        require(args.server_receipt is not None and args.server_receipt.is_file(), "DG requires --server-receipt from runtime/vllm/serve.py")
    return run, selected, source_sha, client


def normalize(native, problems, benchmark, seed):
    expected = {r["task_id"]: r for r in problems}
    require(len(expected) == len(problems), "Duplicate selected input IDs")
    require(len(native) == len(expected) and len({r["task_id"] for r in native}) == len(native), "Incomplete/duplicate output")
    require({r["task_id"] for r in native} == set(expected), "Wrong generated task IDs")
    by_id = {r["task_id"]: r for r in native}
    records = []
    for problem in problems:
        row = by_id[problem["task_id"]]
        for key in ("gold", "meta"):
            require(row.get(key) == problem.get(key), "Native output changed original " + key)
        require(isinstance(row.get("raw"), str), "Missing raw response")
        require(row.get("finish") in ("stop", "length"), "Unrecognized finish reason")
        for key in ("completion_tokens", "fwd_denoise", "fwd_total"):
            require(type(row.get(key)) is int and row[key] >= (0 if key == "completion_tokens" else 1), "Invalid counter " + key)
        require(row["fwd_total"] >= row["fwd_denoise"], "Inconsistent forward counters")
        require(type(row.get("wall_s")) in (int, float) and math.isfinite(row["wall_s"]) and row["wall_s"] > 0,
                "Invalid native wall time")
        record = {"task_id": problem["task_id"], "benchmark": benchmark, "seed": seed,
                  "entry_point": problem.get("entry_point"), "gold": problem.get("gold"),
                  "meta": problem.get("meta", {}), "raw": row["raw"], "finish": row["finish"],
                  "completion_tokens": row["completion_tokens"], "fwd_denoise": row["fwd_denoise"],
                  "fwd_total": row["fwd_total"], "wall_s": row["wall_s"], "scoring_status": "not_run"}
        records.append(record)
    return records


def execute(args, prepared):
    run, problems, source_sha, client = prepared
    identity = model_identity(args.model)
    server = None
    if run["family"] == "dg":
        server = read_server_receipt(args.server_receipt)
        validate_server(server, run["parameters"], run["role"], args.base, identity["sha256"])
    args.out.mkdir(parents=True)
    payload = "".join(json.dumps(r, sort_keys=True, ensure_ascii=True) + "\n" for r in problems)
    inputs = args.out / "selected_problems.jsonl"
    inputs.write_text(payload)
    manifest = {"schema_version": 1, "family": run["family"], "model_id": args.model_id,
                "benchmark": args.benchmark, "seed": args.seed, "shard_index": args.shard_index,
                "shard_count": len(run["shards"]), "scope": "smoke_only" if args.smoke_limit else "full_cell" if len(run["shards"]) == 1 else "shard",
                "full_dataset_n": run["dataset_n"], "expected_task_ids": [r["task_id"] for r in problems],
                "original_problems_sha256": source_sha, "selected_problems_sha256": sha(payload.encode()),
                "native_client_sha256": sha(client.read_bytes()), "parameters": run["parameters"],
                "model_manifest_sha256": identity["sha256"], "scoring_status": "not_run"}
    if server is not None:
        manifest["server_receipt_sha256"] = sha(args.server_receipt.read_bytes())
        (args.out / "server_receipt.json").write_bytes(args.server_receipt.read_bytes())
    (args.out / "input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out / "model_manifest.json").write_text(json.dumps(identity, indent=2) + "\n")
    environment = {k: v for k, v in os.environ.items() if k in
                   ("PATH", "LANG", "LD_LIBRARY_PATH", "CUDA_HOME", "CUDA_VISIBLE_DEVICES", "VIRTUAL_ENV")}
    cache = args.out / "cache"
    cache.mkdir()
    environment.update(HOME=str(cache), HF_HOME=str(cache / "hf"), HF_HUB_OFFLINE="1",
                       HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1",
                       TOKENIZERS_PARALLELISM="false")
    native_dir = args.out / "native"
    native_dir.mkdir()
    if run["family"] == "llada":
        cfg = args.out / "decoder.json"
        cfg.write_text(json.dumps(run["parameters"]["decoder"], indent=2) + "\n")
        plan = args.out / "native_plan.json"
        plan.write_text(json.dumps({"datasets": [{"name": args.benchmark, "problems": str(inputs),
            "max_new": run["parameters"]["max_new_tokens"], "out_dir": str(native_dir)}]}, indent=2) + "\n")
        claims = args.out / "claims"
        claims.mkdir()
        environment.update(ORL_ENGINE_ROOT=str(args.runtime / "engine"), MODEL_DIR=str(args.model), CFG=str(cfg),
                           PLAN=str(plan), TEMP="0", ALGO="SDARPrefixKV", WORKER="0", CLAIM_DIR=str(claims),
                           MEM_FRAC="0.60", ATTN_BACKEND="flashinfer", ENGINE_RETRIES="1")
        command = [sys.executable, str(client)]
    else:
        command = [sys.executable, str(client), "--base", args.base, "--model", "dg", "--problems", str(inputs),
                   "--benchmark", args.benchmark, "--seed", str(args.seed), "--max-denoising-steps", "48",
                   "--out", str(native_dir / "shard_w0.jsonl")]
    with (args.out / "generation.log").open("w") as log:
        subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    native = [json.loads(line) for path in sorted(native_dir.glob("*.jsonl"))
              for line in path.read_text().splitlines() if line.strip()]
    records = normalize(native, problems, args.benchmark, args.seed)
    output = args.out / "records.jsonl"
    output.write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=True) + "\n" for r in records))
    receipt = {"schema_version": 1, "status": "complete", "exit_code": 0, "scope": manifest["scope"],
               "input_manifest_sha256": sha((args.out / "input_manifest.json").read_bytes()),
               "records_sha256": sha(output.read_bytes()), "n": len(records), "expected_task_ids": manifest["expected_task_ids"],
               "scoring_status": "not_run", "generated_code_executed": False,
               "source_bound_not_signed_attestation": True}
    (args.out / "execution_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", choices=("llada-base", "llada-orl", "dg-base", "dg-orl"), required=True)
    for name in ("runtime", "model", "problems", "out"):
        parser.add_argument("--" + name, type=lambda s: Path(s).absolute(), required=True)
    parser.add_argument("--benchmark", choices=tuple(INPUTS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--server-receipt", type=Path)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--smoke-limit", type=int, help="Explicit incomplete smoke scope; never a full benchmark score")
    parser.add_argument("--execute", action="store_true", help="Run local generation; default only validates and prints plan")
    args = parser.parse_args(argv)
    prepared = prepare(args)
    if args.execute:
        result = execute(args, prepared)
    else:
        run, problems, source_sha, _ = prepared
        result = {"dry_run": True, "launches_resources": False, "model_id": args.model_id, "benchmark": args.benchmark,
                  "seed": args.seed, "shard_index": args.shard_index, "shard_count": len(run["shards"]),
                  "n": len(problems), "original_problems_sha256": source_sha, "parameters": run["parameters"],
                  "scope": "smoke_only" if args.smoke_limit else "full_cell" if len(run["shards"]) == 1 else "shard",
                  "scoring_status": "not_run", "note": "Model byte inventory and DG live-server receipt checked on --execute"}
    print(json.dumps(result, indent=2))
    return result
