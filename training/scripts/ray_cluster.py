#!/usr/bin/env python3
"""Start, wait for or stop one node of the Ray cluster used by multi-node recipes.

Recipes with `nodes: 2` (training/recipes/diffusion_gemma) run the FSDP actor on
the Ray head node and the SGLang rollout engine on a worker node. Start the head
on the actor node, then the worker on the rollout node, wait until the cluster
has all GPUs, then launch scripts/orl_train.py on the head node with
RAY_ADDRESS=HEAD_IP:PORT (scripts/reproduce_diffusion_gemma.sh does all of this).

    python3 training/scripts/ray_cluster.py head   RECIPE --ip HEAD_IP --sif SIF --python PY --set DG_TRANSFORMERS_OVERLAY=DIR
    python3 training/scripts/ray_cluster.py worker RECIPE --ip HEAD_IP --node-ip WORKER_IP --sif SIF --python PY --set DG_TRANSFORMERS_OVERLAY=DIR
    python3 training/scripts/ray_cluster.py wait   RECIPE --ip HEAD_IP --sif SIF --python PY --gpus 16
    python3 training/scripts/ray_cluster.py stop   RECIPE --sif SIF --python PY

`head` and `worker` block until the Ray node exits; `wait` exits 0 once the
cluster reports --gpus GPUs and 1 after --timeout seconds. The containers bind
the top-level directories of the training tree, the Python interpreter and every
absolute --set path, plus any --bind; these paths must be the same on both nodes.
"""
import argparse
import getpass
import os
import re
import subprocess
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orl_train import bind_roots, default_ifname  # noqa: E402

TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAY_CLI = 'import sys; from ray.scripts.scripts import main; sys.argv=["ray"]+sys.argv[1:]; main()'
WAIT_FOR_GPUS = """
import sys, time
import ray
address, gpus, timeout = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
deadline = time.time() + timeout
while True:
    try:
        ray.init(address=address, logging_level="ERROR")
        have = int(ray.cluster_resources().get("GPU", 0))
        ray.shutdown()
    except Exception as error:
        have = 0
        print(f"waiting for the Ray head at {address}: {error}", flush=True)
    if have >= gpus:
        print(f"the Ray cluster has {have} GPUs", flush=True)
        sys.exit(0)
    if time.time() > deadline:
        print(f"the Ray cluster has {have} GPUs after {timeout:.0f}s, expected {gpus}", flush=True)
        sys.exit(1)
    time.sleep(10)
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("role", choices=["head", "worker", "wait", "stop"])
    ap.add_argument("recipe")
    ap.add_argument("--ip", help="IP of the head node (its own IP for `head`)")
    ap.add_argument("--node-ip", help="this worker's own IP (default: chosen by Ray)")
    ap.add_argument("--port", type=int, default=6606)
    ap.add_argument("--sif", required=True)
    ap.add_argument("--python", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--num-cpus", type=int, default=64)
    ap.add_argument("--object-store-memory", type=int, default=120_000_000_000)
    ap.add_argument("--temp-dir", default=f"/dev/shm/orl-raynode-{getpass.getuser()}")
    ap.add_argument("--bind", action="append", default=[], help="extra Apptainer bind (src[:dst])")
    ap.add_argument("--gpus", type=int, help="`wait`: number of GPUs to wait for (default: 2 x gpus_per_node)")
    ap.add_argument("--timeout", type=float, default=1200, help="`wait`: seconds before giving up")
    args = ap.parse_args()

    recipe = yaml.safe_load(open(args.recipe))
    values = dict(kv.split("=", 1) for kv in args.set)
    gpus = int(recipe.get("gpus_per_node", 8))
    sglang_dir = f"{TRAINING_ROOT}/{recipe.get('sglang_dir', 'sglang')}"
    extra = [re.sub(r"\$\{([A-Z0-9_]+)\}", lambda m: values[m.group(1)], p) for p in recipe.get("extra_python_path", [])]
    python_path = ":".join([f"{TRAINING_ROOT}/kernels/fast-diffusion-attention", sglang_dir, f"{TRAINING_ROOT}/verl"] + extra)
    ifname = values.get("NET_IFNAME", default_ifname())
    env = {
        "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(gpus)),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": python_path,
        "VERL_LOGGING_LEVEL": "INFO",
        "VERL_SGLANG_PATCHED_PATH": sglang_dir,
        "VERL_SGLANG_BATCH_INPUT_IDS": "0",
        "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "1",
        "RAY_USAGE_STATS_ENABLED": "0",
        "NCCL_DEBUG": "WARN",
        "NCCL_IB_DISABLE": "1",
        "NCCL_SOCKET_IFNAME": ifname,
        "GLOO_SOCKET_IFNAME": ifname,
    }
    paths = [TRAINING_ROOT, args.python] + [v for v in values.values() if v.startswith("/")]
    cmd = ["apptainer", "exec", "--nv"]
    for b in [f"{r}:{r}" for r in bind_roots(paths)] + args.bind:
        cmd += ["-B", b]
    for k, v in env.items():
        cmd += ["--env", f"{k}={v}"]
    cmd += [args.sif, args.python]
    if args.role == "stop":
        return subprocess.call(cmd + ["-c", RAY_CLI, "stop", "--force"])
    if not args.ip:
        sys.exit("--ip is required for head, worker and wait")
    if args.role == "wait":
        return subprocess.call(cmd + ["-c", WAIT_FOR_GPUS, f"{args.ip}:{args.port}", str(args.gpus or 2 * gpus),
                                      str(args.timeout)])
    os.makedirs(args.temp_dir, exist_ok=True)
    # Recipes point TMPDIR of every Ray worker at this node-local directory
    # (same default as orl_train.py); it has to exist on each node.
    os.makedirs(values.get("NODE_TMP_DIR", f"/dev/shm/orl-tmp-{getpass.getuser()}"), exist_ok=True)
    common = [f"--num-gpus={gpus}", f"--num-cpus={args.num_cpus}", f"--object-store-memory={args.object_store_memory}",
              f"--temp-dir={args.temp_dir}", "--block"]
    cmd += ["-c", RAY_CLI]
    if args.role == "head":
        cmd += ["start", "--head", f"--node-ip-address={args.ip}", f"--port={args.port}", "--include-dashboard=false"] + common
    else:
        node_ip = [f"--node-ip-address={args.node_ip}"] if args.node_ip else []
        cmd += ["start", f"--address={args.ip}:{args.port}"] + node_ip + common
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    sys.exit(main())
