#!/usr/bin/env python3
"""Evaluate one exported HF model with the repo's patched SGLang backend.

Launches one SGLang server per shard/GPU via Apptainer, waits for all servers
to become healthy, then launches the eval_client.py worker for each shard
**inside the same Apptainer SIF** (no --nv; client is CPU-only).
After all clients finish (or on error), servers are shut down.

DEFAULT (portable) mode — client runs inside the SIF:
    python repro/eval/run_eval.py \\
        --benchmark gsm8k \\
        --model /path/to/exported/model \\
        --output-dir /path/to/results \\
        --algo-config repro/eval/configs/p06.yaml \\
        --repo-root /path/to/repo \\
        --sif /path/to/sglang.sif \\
        --hf-cache /path/to/hf-cache

    The in-SIF Python used for the client defaults to:
        <repo-root>/runtime/verl-sglang-cu129-sif-venv/bin/python3
    Override with --client-python or EVAL_CLIENT_PYTHON.

HOST CLIENT mode (opt-in, requires a separate host Python env with datasets):
    python repro/eval/run_eval.py ... --host-client [--python /path/to/python]

Supports --dry-run: prints all server/client commands and validates paths
without starting any processes or requiring GPUs.

All paths can also be supplied via environment variables (CLI takes precedence):
    REPO_ROOT, SGLANG_SIF, HF_CACHE, EVAL_CLIENT_PYTHON, MODEL_PATH,
    EVAL_OUTPUT_DIR, GPU_OFFSET, BASE_PORT, ALGO_CONFIG, EVAL_BENCHMARK,
    EVAL_CUDA_GRAPH, EVAL_CUDA_GRAPH_CAPTURE_SIZES,
    EVAL_HOST_CLIENT (set to "1" to enable host-client mode),
    EVAL_PYTHON (host-client mode only)
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_CLIENT_SCRIPT = _THIS_DIR / "eval_client.py"

_BENCHMARK_DEFAULTS = {
    "gsm8k": {"num_q": 400, "num_shards": 4},
    "math500": {"num_q": 100, "num_shards": 4},
    "humaneval": {"num_q": 164, "num_shards": 8},
    "mbpp": {"num_q": 500, "num_shards": 8},
}


# ---------------------------------------------------------------------------
# Shard ranges
# ---------------------------------------------------------------------------


def shard_ranges(
    total: int, num_shards: int, start: int = 0
) -> list[tuple[int, int]]:
    """Return list of (start_idx, count) for each shard.

    Distributes remainder questions to the first shards (round-robin).
    Empty shards are omitted.
    """
    if total <= 0 or num_shards <= 0:
        raise ValueError(f"Invalid: total={total}, num_shards={num_shards}")
    base, remainder = divmod(total, num_shards)
    result = []
    cursor = start
    for shard_id in range(num_shards):
        size = base + (1 if shard_id < remainder else 0)
        if size > 0:
            result.append((cursor, size))
            cursor += size
    return result


# ---------------------------------------------------------------------------
# SGLang server health check
# ---------------------------------------------------------------------------


def health_ok(port: int) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/get_model_info",
            timeout=5,
        ) as response:
            return 200 <= int(response.status) < 300
    except (OSError, urllib.error.URLError):
        return False


def wait_for_health(
    port: int,
    timeout_s: float = 900.0,
    poll_s: float = 5.0,
) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if health_ok(port):
            return
        time.sleep(poll_s)
    raise RuntimeError(
        f"SGLang server on port {port} did not become healthy within {timeout_s}s"
    )


# ---------------------------------------------------------------------------
# Process lifecycle
# ---------------------------------------------------------------------------


def stop_processes(procs: list[subprocess.Popen]) -> None:
    for proc in procs:
        if proc.poll() is None:
            try:
                proc.send_signal(signal.SIGTERM)
            except Exception:
                pass
    time.sleep(5)
    for proc in procs:
        if proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------


def cuda_graph_args(enabled: bool, capture_sizes: str) -> list[str]:
    if not enabled:
        return ["--disable-cuda-graph"]
    sizes = [int(value) for value in capture_sizes.split(",")]
    if not sizes or any(value <= 0 for value in sizes):
        raise ValueError(f"invalid CUDA graph capture sizes: {capture_sizes}")
    return [
        "--sdar-rollout-cuda-graph",
        "--sdar-rollout-cuda-graph-capture-sizes",
        ",".join(str(value) for value in sorted(set(sizes))),
    ]


def build_local_server_cmd(
    *,
    repo_root: Path,
    model_path: Path,
    algo_config: Path,
    gpu: int,
    port: int,
    context_length: int,
    max_running_requests: int,
    cuda_graph: bool = False,
    cuda_graph_capture_sizes: str = "1",
) -> list[str]:
    """Return a local (no-apptainer) command that launches one SGLang server shard.

    Uses ``env`` to inject CUDA_VISIBLE_DEVICES and PYTHONPATH without
    polluting the parent process environment.  The sglang overlay from
    <repo_root>/inference/sglang is prepended to PYTHONPATH.
    """
    sglang_overlay = repo_root / "inference" / "sglang"
    cmd = [
        "env",
        f"CUDA_VISIBLE_DEVICES={gpu}",
        f"PYTHONPATH={sglang_overlay}",
    ]
    # One JIT cache per shard: concurrent servers sharing a Triton/Inductor cache dir race on
    # lazily compiled kernels (corrupt cubins -> "illegal memory access" in every shard but the
    # first to start; seen 2026-09-17 with the fused SDAR kernels).
    for var in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
        base = os.environ.get(var)
        if base:
            cmd.append(f"{var}={base.rstrip('/')}/gpu{gpu}")
    cmd += [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(model_path),
        "--trust-remote-code",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--served-model-name",
        "sdar",
        "--skip-server-warmup",
        "--tp-size",
        "1",
        "--max-running-requests",
        str(max_running_requests),
        "--context-length",
        str(context_length),
        "--attention-backend",
        "flashinfer",
        "--chunked-prefill-size",
        "-1",
        "--dllm-algorithm",
        "SDARPrefixKV",
        "--dllm-algorithm-config",
        str(algo_config),
    ]
    cmd.extend(cuda_graph_args(cuda_graph, cuda_graph_capture_sizes))
    extra = os.environ.get("SGLANG_EXTRA_SERVER_ARGS", "").strip()
    if extra:
        cmd.extend(shlex.split(extra))
    return cmd


def build_server_cmd(
    *,
    sif: Path,
    repo_root: Path,
    hf_cache: Path,
    sglang_overlay: Path,
    model_path: Path,
    algo_config: Path,
    gpu: int,
    port: int,
    context_length: int,
    max_running_requests: int,
    extra_bind_paths: list[Path] | None = None,
    cuda_graph: bool = False,
    cuda_graph_capture_sizes: str = "1",
) -> list[str]:
    """Return the apptainer command that launches one SGLang server shard."""
    bind_paths = [repo_root, model_path, sglang_overlay.parent.parent]
    # Hugging Face snapshot files are often symlinks into a sibling ``blobs``
    # directory. Binding only the snapshot directory leaves those targets
    # unresolved (or resolves them against stale paths inside the image).
    # Bind each external target directory at its host absolute path as well.
    for child in model_path.iterdir():
        if child.is_symlink():
            bind_paths.append(child.resolve(strict=True).parent)
    if extra_bind_paths:
        bind_paths.extend(extra_bind_paths)
    bind_paths = list(dict.fromkeys(bind_paths))

    cmd = [
        "apptainer",
        "exec",
        "--nv",
        "--env",
        f"CUDA_VISIBLE_DEVICES={gpu}",
        "--env",
        "HF_HOME=/hf",
        "--env",
        f"PYTHONPATH={sglang_overlay}",
    ]
    for bp in bind_paths:
        cmd += ["-B", str(bp)]
    cmd += [
        "-B",
        f"{hf_cache}:/hf",
        str(sif),
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(model_path),
        "--trust-remote-code",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--served-model-name",
        "sdar",
        "--skip-server-warmup",
        "--tp-size",
        "1",
        "--max-running-requests",
        str(max_running_requests),
        "--context-length",
        str(context_length),
        "--attention-backend",
        "flashinfer",
        "--chunked-prefill-size",
        "-1",
        "--dllm-algorithm",
        "SDARPrefixKV",
        "--dllm-algorithm-config",
        str(algo_config),
    ]
    cmd.extend(cuda_graph_args(cuda_graph, cuda_graph_capture_sizes))
    extra = os.environ.get("SGLANG_EXTRA_SERVER_ARGS", "").strip()
    if extra:
        cmd.extend(shlex.split(extra))
    return cmd


def build_client_env(
    *,
    base_env: dict,
    benchmark: str,
    port: int,
    start_q: int,
    num_q: int,
    max_new_tokens: int,
    out_path: Path,
    hf_cache: Path,
    eval_run_id: str = "",
) -> dict:
    env = dict(base_env)
    env.update(
        {
            "EVAL_BENCHMARK": benchmark,
            "SGLANG_BASE_URL": f"http://127.0.0.1:{port}",
            "SGLANG_START_Q": str(start_q),
            "SGLANG_NUM_Q": str(num_q),
            "SGLANG_MAX_NEW_TOKENS": str(max_new_tokens),
            "SGLANG_EVAL_OUT": str(out_path),
            "SGLANG_WORKERS": os.environ.get("SGLANG_WORKERS", "1"),
            "SGLANG_SAVE_FULL_TEXT": os.environ.get("SGLANG_SAVE_FULL_TEXT", "0"),
            "HF_HOME": str(hf_cache),
            "HF_HUB_CACHE": str(hf_cache / "hub"),
            "HF_DATASETS_CACHE": str(hf_cache / f"hfds_{benchmark}_{port}"),
            "SGLANG_EVAL_RUN_ID": eval_run_id,
        }
    )
    return env


def build_sif_client_cmd(
    *,
    sif: Path,
    repo_root: Path,
    hf_cache: Path,
    output_dir: Path,
    client_python: Path,
    shard_env: dict[str, str],
) -> list[str]:
    """Return the apptainer command that runs eval_client.py inside the SIF.

    No ``--nv``: the client is CPU-only (it posts HTTP to a local server).
    Shard-specific env vars are passed explicitly via ``--env`` so the host
    environment is not injected into the container.
    repo_root, hf_cache, and output_dir are bound at their host absolute paths.
    """
    client_venv = client_python.parent.parent
    cmd = ["apptainer", "exec"]
    for bind_path in (repo_root, client_venv, hf_cache, output_dir):
        cmd += ["-B", str(bind_path)]
    for key, value in shard_env.items():
        cmd += ["--env", f"{key}={value}"]
    cmd += [str(sif), str(client_python), str(_CLIENT_SCRIPT)]
    return cmd


# ---------------------------------------------------------------------------
# Path / argument validation
# ---------------------------------------------------------------------------


def _resolve_path(value: str | None, env_key: str, label: str) -> Path | None:
    raw = value or os.environ.get(env_key, "").strip() or None
    return Path(raw).resolve() if raw else None


def _require_path(value: str | None, env_key: str, label: str) -> Path:
    p = _resolve_path(value, env_key, label)
    if p is None:
        sys.exit(
            f"Error: {label} is required. Pass --{label.lower().replace(' ', '-')} "
            f"or set {env_key}."
        )
    return p


def resolve_client_python(value: str | None, repo_root: Path) -> Path:
    """Preserve a venv symlink path instead of resolving it to system Python."""
    if value:
        return Path(os.path.abspath(value))
    return (
        repo_root
        / "runtime"
        / "verl-sglang-cu129-sif-venv"
        / "bin"
        / "python3"
    )


def validate_paths(
    *,
    repo_root: Path,
    sif: Path | None,
    hf_cache: Path,
    model_path: Path,
    algo_config: Path,
    host_python: Path | None = None,
    no_sif: bool = False,
) -> None:
    """Validate all host-side paths before launching any processes.

    host_python: when provided (host-client mode), verified to exist on the
    host.  In default SIF-client mode, the client Python lives inside the
    container and is not checked here.
    no_sif: when True, sif is not required and its path is not checked.
    """
    errors = []
    if not repo_root.is_dir():
        errors.append(f"--repo-root not a directory: {repo_root}")
    if not no_sif:
        if sif is None or not sif.is_file():
            errors.append(f"--sif not found: {sif}")
    if not hf_cache.is_dir():
        errors.append(f"--hf-cache not a directory: {hf_cache}")
    if not (model_path / "config.json").is_file():
        errors.append(
            f"--model does not look like an HF model dir "
            f"(config.json not found): {model_path}"
        )
    if not algo_config.is_file():
        errors.append(f"--algo-config not found: {algo_config}")
    if host_python is not None and not host_python.is_file():
        errors.append(f"--python not found: {host_python}")
    if errors:
        sys.exit("Path validation failed:\n" + "\n".join(f"  {e}" for e in errors))


# ---------------------------------------------------------------------------
# Row counting (for progress monitoring)
# ---------------------------------------------------------------------------


def count_completed(out_dir: Path, benchmark: str) -> int:
    total = 0
    for path in out_dir.glob("*.json"):
        if "assignments" in path.name:
            continue
        try:
            data = json.loads(path.read_text())
            total += len(data.get("results", []))
        except Exception:
            pass
    return total


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------


def run_eval(
    *,
    benchmark: str,
    model_path: Path,
    output_dir: Path,
    algo_config: Path,
    repo_root: Path,
    sif: Path | None,
    hf_cache: Path,
    client_python: Path,
    num_q: int,
    num_shards: int,
    start_q: int,
    gpu_offset: int,
    base_port: int,
    port_stride: int,
    max_new_tokens: int,
    context_length: int,
    max_running_requests: int,
    cuda_graph: bool,
    cuda_graph_capture_sizes: str,
    health_timeout_s: float,
    dry_run: bool,
    host_client: bool = False,
    host_python: Path | None = None,
    no_sif: bool = False,
) -> None:
    # --no-sif: run servers locally and use host-client mode for eval_client.py
    if no_sif:
        host_client = True
        if host_python is None:
            host_python = Path(sys.executable)

    sglang_overlay = repo_root / "inference" / "sglang"
    ranges = shard_ranges(num_q, num_shards, start_q)
    eval_run_id = str(uuid.uuid4())
    bm_out = output_dir / benchmark
    bm_out.mkdir(parents=True, exist_ok=True)

    # Build manifest (server + client command pairs)
    manifest = []
    for shard_id, (sq, nq) in enumerate(ranges):
        gpu = gpu_offset + shard_id
        port = base_port + shard_id * port_stride
        shard_label = f"{benchmark}_s{sq:04d}"
        out_path = bm_out / f"{shard_label}.json"
        server_log = bm_out / f"{shard_label}.server.log"
        client_log = bm_out / f"{shard_label}.client.log"

        if no_sif:
            server_cmd = build_local_server_cmd(
                repo_root=repo_root,
                model_path=model_path,
                algo_config=algo_config,
                gpu=gpu,
                port=port,
                context_length=context_length,
                max_running_requests=max_running_requests,
                cuda_graph=cuda_graph,
                cuda_graph_capture_sizes=cuda_graph_capture_sizes,
            )
        else:
            server_cmd = build_server_cmd(
                sif=sif,
                repo_root=repo_root,
                hf_cache=hf_cache,
                sglang_overlay=sglang_overlay,
                model_path=model_path,
                algo_config=algo_config,
                gpu=gpu,
                port=port,
                context_length=context_length,
                max_running_requests=max_running_requests,
                cuda_graph=cuda_graph,
                cuda_graph_capture_sizes=cuda_graph_capture_sizes,
            )
        # Shard-specific env vars (clean dict, no host env leakage for SIF mode)
        shard_env = build_client_env(
            base_env={},
            benchmark=benchmark,
            port=port,
            start_q=sq,
            num_q=nq,
            max_new_tokens=max_new_tokens,
            out_path=out_path,
            hf_cache=hf_cache,
            eval_run_id=eval_run_id,
        )
        if host_client:
            assert host_python is not None
            client_cmd = [str(host_python), str(_CLIENT_SCRIPT)]
        else:
            client_cmd = build_sif_client_cmd(
                sif=sif,
                repo_root=repo_root,
                hf_cache=hf_cache,
                output_dir=bm_out,
                client_python=client_python,
                shard_env=shard_env,
            )

        manifest.append(
            {
                "shard_id": shard_id,
                "start_q": sq,
                "num_q": nq,
                "gpu": gpu,
                "port": port,
                "out_path": str(out_path),
                "server_log": str(server_log),
                "client_log": str(client_log),
                "server_cmd": server_cmd,
                "client_cmd": client_cmd,
                "shard_env": shard_env,
                "host_client": host_client,
                "eval_run_id": eval_run_id,
                "client_workers": int(shard_env["SGLANG_WORKERS"]),
            }
        )

    # Write assignments file
    assignments_path = bm_out / "assignments.json"
    safe_manifest = [
        {k: v for k, v in m.items() if k != "shard_env"} for m in manifest
    ]
    previous = []
    if assignments_path.exists():
        try:
            previous = json.loads(assignments_path.read_text())
        except json.JSONDecodeError:
            previous = []
    assignments_path.write_text(
        json.dumps([*previous, *safe_manifest], indent=2)
    )

    if dry_run:
        mode = "host-client" if host_client else "sif-client"
        print(f"[dry-run] benchmark={benchmark} shards={len(manifest)} mode={mode}")
        for m in manifest:
            print(f"\n--- shard {m['shard_id']} (GPU={m['gpu']}, port={m['port']}) ---")
            print("SERVER:", shlex.join(m["server_cmd"]))
            print("CLIENT:", shlex.join(m["client_cmd"]))
            if m["host_client"]:
                print(
                    "CLIENT ENV:",
                    "  ".join(f"{k}={v}" for k, v in m["shard_env"].items()),
                )
        return

    servers: list[subprocess.Popen] = []
    clients: list[subprocess.Popen] = []
    server_log_handles = []
    client_log_handles = []

    try:
        # Launch servers
        for m in manifest:
            fh = open(m["server_log"], "w")
            server_log_handles.append(fh)
            print(
                f"[server] launching shard {m['shard_id']} "
                f"gpu={m['gpu']} port={m['port']}",
                flush=True,
            )
            servers.append(
                subprocess.Popen(
                    m["server_cmd"], stdout=fh, stderr=subprocess.STDOUT
                )
            )

        # Wait for health
        for m in manifest:
            print(f"[health] waiting for port {m['port']} ...", flush=True)
            wait_for_health(m["port"], timeout_s=health_timeout_s)
            print(f"[health] port {m['port']} ready", flush=True)

        # Launch clients
        for m in manifest:
            fh = open(m["client_log"], "a")
            client_log_handles.append(fh)
            if m["host_client"]:
                # Host mode: pass shard vars via process env
                client_process_env = build_client_env(
                    base_env=os.environ.copy(),
                    benchmark=benchmark,
                    port=m["port"],
                    start_q=m["start_q"],
                    num_q=m["num_q"],
                    max_new_tokens=max_new_tokens,
                    out_path=Path(m["out_path"]),
                    hf_cache=hf_cache,
                    eval_run_id=m["eval_run_id"],
                )
                clients.append(
                    subprocess.Popen(
                        m["client_cmd"],
                        stdout=fh,
                        stderr=subprocess.STDOUT,
                        env=client_process_env,
                    )
                )
            else:
                # SIF mode: env vars are embedded in the apptainer command
                clients.append(
                    subprocess.Popen(
                        m["client_cmd"],
                        stdout=fh,
                        stderr=subprocess.STDOUT,
                    )
                )
        print(f"[clients] launched {len(clients)}", flush=True)

        # Monitor until done
        while True:
            done = count_completed(bm_out, benchmark)
            print(f"[progress] {benchmark} {done}/{num_q}", flush=True)
            if done >= num_q:
                print(f"[done] {benchmark} all {num_q} questions complete", flush=True)
                break
            all_exited = all(c.poll() is not None for c in clients)
            if all_exited:
                codes = [c.returncode for c in clients]
                raise RuntimeError(
                    f"All clients exited before completion: "
                    f"{done}/{num_q} done, codes={codes}"
                )
            time.sleep(60)

    finally:
        print(f"[cleanup] stopping clients and servers for {benchmark}", flush=True)
        stop_processes(clients)
        stop_processes(servers)
        for fh in client_log_handles + server_log_handles:
            try:
                fh.close()
            except Exception:
                pass
        final = count_completed(bm_out, benchmark)
        print(f"[final] {benchmark}: {final}/{num_q} rows written", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = {}  # populated from env below

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--benchmark",
        choices=list(_BENCHMARK_DEFAULTS),
        default=os.environ.get("EVAL_BENCHMARK", "").strip() or None,
        required=not os.environ.get("EVAL_BENCHMARK", "").strip(),
        help="Benchmark: gsm8k, math500, humaneval, or mbpp.",
    )
    ap.add_argument(
        "--model",
        default=os.environ.get("MODEL_PATH", "").strip() or None,
        required=not os.environ.get("MODEL_PATH", "").strip(),
        metavar="DIR",
        help="Path to merged HF SafeTensors model (must contain config.json). Env: MODEL_PATH",
    )
    ap.add_argument(
        "--output-dir",
        default=os.environ.get("EVAL_OUTPUT_DIR", "").strip() or None,
        required=not os.environ.get("EVAL_OUTPUT_DIR", "").strip(),
        metavar="DIR",
        help="Root output directory. Results go to <output-dir>/<benchmark>/. Env: EVAL_OUTPUT_DIR",
    )
    ap.add_argument(
        "--algo-config",
        default=os.environ.get("ALGO_CONFIG", "").strip() or None,
        required=not os.environ.get("ALGO_CONFIG", "").strip(),
        metavar="YAML",
        help=(
            "SDARPrefixKV algorithm config YAML (e.g. repro/eval/configs/p06.yaml). "
            "Env: ALGO_CONFIG"
        ),
    )
    ap.add_argument(
        "--repo-root",
        default=os.environ.get("REPO_ROOT", "").strip() or None,
        required=not os.environ.get("REPO_ROOT", "").strip(),
        metavar="DIR",
        help=(
            "Repo root directory (bound into the Apptainer container). "
            "inference/sglang overlay is expected at <repo-root>/inference/sglang. "
            "Env: REPO_ROOT"
        ),
    )
    ap.add_argument(
        "--sif",
        default=os.environ.get("SGLANG_SIF", "").strip() or None,
        required=False,
        metavar="FILE",
        help=(
            "Path to the SGLang Apptainer .sif image. Required unless --no-sif "
            "is set. Env: SGLANG_SIF"
        ),
    )
    ap.add_argument(
        "--no-sif",
        action="store_true",
        default=os.environ.get("EVAL_NO_SIF", "0").strip() == "1",
        help=(
            "Run SGLang servers directly on the host (no Apptainer). "
            "Implies host-client mode. --sif is not required. "
            "Use when the eval pod already has sglang installed. "
            "Env: EVAL_NO_SIF=1"
        ),
    )
    ap.add_argument(
        "--hf-cache",
        default=os.environ.get("HF_CACHE", "").strip() or None,
        required=not os.environ.get("HF_CACHE", "").strip(),
        metavar="DIR",
        help=(
            "HuggingFace cache directory. Mounted as /hf in container (HF_HOME=/hf) "
            "and used for HF_DATASETS_CACHE on the host. Env: HF_CACHE"
        ),
    )
    ap.add_argument(
        "--client-python",
        default=os.environ.get("EVAL_CLIENT_PYTHON", "").strip() or None,
        metavar="BIN",
        help=(
            "Path to the Python interpreter inside the SIF container used to run "
            "eval_client.py (default SIF-client mode). "
            "Default: <repo-root>/runtime/verl-sglang-cu129-sif-venv/bin/python3. "
            "Env: EVAL_CLIENT_PYTHON"
        ),
    )
    ap.add_argument(
        "--host-client",
        action="store_true",
        default=os.environ.get("EVAL_HOST_CLIENT", "0").strip() == "1",
        help=(
            "Run eval_client.py on the HOST Python instead of inside the SIF. "
            "Requires a host Python env with datasets/requests installed. "
            "Use --python to specify the host interpreter. Env: EVAL_HOST_CLIENT=1"
        ),
    )
    ap.add_argument(
        "--python",
        default=os.environ.get("EVAL_PYTHON", "").strip() or sys.executable,
        metavar="BIN",
        help=(
            "Host Python interpreter for eval_client.py (HOST CLIENT mode only). "
            "Ignored in default SIF-client mode. "
            f"Defaults to sys.executable ({sys.executable}). Env: EVAL_PYTHON"
        ),
    )
    env_num_q = os.environ.get("EVAL_NUM_Q", "").strip()
    env_num_shards = os.environ.get("EVAL_NUM_SHARDS", "").strip()
    ap.add_argument(
        "--num-q",
        type=int,
        default=int(env_num_q) if env_num_q else None,
        help="Total number of questions (default: 400 for gsm8k, 100 for math500).",
    )
    ap.add_argument(
        "--num-shards",
        type=int,
        default=int(env_num_shards) if env_num_shards else None,
        help="Number of parallel shards / GPUs (default: 4).",
    )
    ap.add_argument(
        "--start-q",
        type=int,
        default=int(os.environ.get("EVAL_START_Q", "0")),
        help="First question index (0-based, default: 0).",
    )
    ap.add_argument(
        "--gpu-offset",
        type=int,
        default=int(os.environ.get("GPU_OFFSET", "0")),
        help="CUDA_VISIBLE_DEVICES offset for first shard GPU (default: 0). Env: GPU_OFFSET",
    )
    ap.add_argument(
        "--base-port",
        type=int,
        default=int(os.environ.get("BASE_PORT", "46000")),
        help="Base TCP port (default: 46000). Env: BASE_PORT",
    )
    ap.add_argument(
        "--port-stride",
        type=int,
        default=int(os.environ.get("PORT_STRIDE", "10")),
        help=(
            "Port spacing between shards; avoids SGLang auxiliary-port "
            "collisions (default: 10). Env: PORT_STRIDE"
        ),
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(os.environ.get("EVAL_MAX_NEW_TOKENS", "32768")),
        help="Maximum new tokens per generation (default: 32768).",
    )
    ap.add_argument(
        "--context-length",
        type=int,
        default=int(os.environ.get("EVAL_CONTEXT_LENGTH", "65536")),
        help="SGLang server context length (default: 65536).",
    )
    ap.add_argument(
        "--max-running-requests",
        type=int,
        default=int(os.environ.get("SGLANG_MAX_RUNNING_REQUESTS", "1")),
        help="SGLang --max-running-requests (default: 1).",
    )
    ap.add_argument(
        "--cuda-graph",
        action="store_true",
        default=os.environ.get("EVAL_CUDA_GRAPH", "0") == "1",
        help="Enable SDAR rollout CUDA graphs. Env: EVAL_CUDA_GRAPH=1",
    )
    ap.add_argument(
        "--cuda-graph-capture-sizes",
        default=os.environ.get("EVAL_CUDA_GRAPH_CAPTURE_SIZES", "1"),
        help=(
            "Comma-separated rollout graph batch sizes (default: 1). "
            "LLaDA fused_next needs 1,2 and a server capacity >=2 while "
            "SGLANG_WORKERS remains 1."
        ),
    )
    ap.add_argument(
        "--health-timeout",
        type=float,
        default=float(os.environ.get("HEALTH_TIMEOUT_S", "900")),
        metavar="SECONDS",
        help="Seconds to wait for each server to become healthy (default: 900).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print server/client commands and validate paths without starting "
            "any processes or requiring GPUs."
        ),
    )
    return ap


def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()

    benchmark = args.benchmark
    model_path = Path(args.model).resolve()
    output_dir = Path(args.output_dir).resolve()
    algo_config = Path(args.algo_config).resolve()
    repo_root = Path(args.repo_root).resolve()
    no_sif: bool = args.no_sif
    if args.sif:
        sif: Path | None = Path(args.sif).resolve()
    elif not no_sif:
        sys.exit(
            "Error: --sif is required unless --no-sif is set. "
            "Pass --sif or set SGLANG_SIF, or use --no-sif / EVAL_NO_SIF=1."
        )
    else:
        sif = None
    hf_cache = Path(args.hf_cache).resolve()
    host_client: bool = args.host_client
    host_python = Path(args.python).absolute() if (host_client or no_sif) else None  # absolute(), not resolve(): a venv bin/python symlink must stay inside the venv

    # In-SIF client Python: explicit flag > env > default relative to repo_root
    client_python = resolve_client_python(args.client_python, repo_root)

    benchmark_defaults = _BENCHMARK_DEFAULTS[benchmark]
    num_q = (
        args.num_q
        if args.num_q is not None
        else int(benchmark_defaults["num_q"])
    )
    num_shards = (
        args.num_shards
        if args.num_shards is not None
        else int(benchmark_defaults["num_shards"])
    )
    if args.port_stride < 4:
        raise ValueError(
            "--port-stride must be at least 4 to avoid SGLang auxiliary ports"
        )

    print(
        f"[config] benchmark={benchmark} num_q={num_q} num_shards={num_shards} "
        f"start_q={args.start_q} gpu_offset={args.gpu_offset} "
        f"base_port={args.base_port} port_stride={args.port_stride}",
        flush=True,
    )
    print(f"[config] model        = {model_path}", flush=True)
    print(f"[config] output       = {output_dir}", flush=True)
    print(f"[config] config       = {algo_config}", flush=True)
    print(f"[config] repo_root    = {repo_root}", flush=True)
    print(f"[config] sif          = {sif}", flush=True)
    print(f"[config] hf_cache     = {hf_cache}", flush=True)
    if host_client:
        print(f"[config] host_python  = {host_python}  [host-client mode]", flush=True)
    else:
        print(f"[config] client_python = {client_python}  [sif-client mode]", flush=True)

    if not args.dry_run:
        validate_paths(
            repo_root=repo_root,
            sif=sif,
            hf_cache=hf_cache,
            model_path=model_path,
            algo_config=algo_config,
            host_python=host_python,
            no_sif=no_sif,
        )

    run_eval(
        benchmark=benchmark,
        model_path=model_path,
        output_dir=output_dir,
        algo_config=algo_config,
        repo_root=repo_root,
        sif=sif,
        hf_cache=hf_cache,
        client_python=client_python,
        num_q=num_q,
        num_shards=num_shards,
        start_q=args.start_q,
        gpu_offset=args.gpu_offset,
        base_port=args.base_port,
        port_stride=args.port_stride,
        max_new_tokens=args.max_new_tokens,
        context_length=args.context_length,
        max_running_requests=args.max_running_requests,
        cuda_graph=args.cuda_graph,
        cuda_graph_capture_sizes=args.cuda_graph_capture_sizes,
        health_timeout_s=args.health_timeout,
        dry_run=args.dry_run,
        host_client=host_client,
        host_python=host_python,
        no_sif=no_sif,
    )


if __name__ == "__main__":
    main()
