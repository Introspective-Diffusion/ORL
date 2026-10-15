#!/usr/bin/env python3
"""Launch one ORL training recipe (training/recipes/<family>/<recipe>.yaml).

Example (single node, 8 GPUs, Apptainer runtime):

    python3 training/scripts/orl_train.py training/recipes/llada2_mini/orl.yaml \
        --run-dir runs/llada2 \
        --set MODEL_PATH=/models/LLaDA2.1-mini-actor \
        --set TRAIN_FILE=/data/gsm8k-math-code_l0-mix-20260909/train.parquet \
        --set VAL_FILE=/data/code_he_clean_rebuild/val.parquet \
        --sif /images/sglang-dev-cu12.sif --python /venvs/verl-sglang/bin/python3

A recipe holds the base Hydra config, the trainer environment and the ordered
Hydra overrides of the paper run, with ${PLACEHOLDER} values that are resolved
here. The resolved command is written to RUN_DIR/launch.json before launch.
"""
import argparse
import getpass
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys

import yaml

TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLACEHOLDER = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")


def default_ifname():
    try:
        with open("/proc/net/route") as f:
            for line in f.readlines()[1:]:
                fields = line.split()
                if fields[1] == "00000000":
                    return fields[0]
    except OSError:
        pass
    return "lo"


def resolve(value, values, missing):
    def rep(m):
        key = m.group(1)
        if key not in values:
            missing.add(key)
            return m.group(0)
        return str(values[key])

    return PLACEHOLDER.sub(rep, value)


def bind_roots(paths):
    """Top-level directories to bind so that `paths` resolve inside the container, both as given (a venv's
    bin/python3 is a symlink into the image) and resolved (model directories may hold symlinks into a cache)."""
    roots = set()

    def add(path):
        parts = path.split(os.sep)
        if len(parts) > 1 and parts[1] not in ("", "dev", "proc", "sys", "tmp", "usr", "etc", "bin", "lib", "lib64"):
            roots.add(os.sep + parts[1])

    for p in paths:
        if not p or not os.path.isabs(p):
            continue
        add(os.path.abspath(p))
        add(os.path.realpath(p))
        if os.path.isdir(p):
            for entry in os.scandir(p):
                if entry.is_symlink():
                    add(os.path.realpath(entry.path))
    return sorted(roots)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recipe")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="resolve a recipe placeholder")
    ap.add_argument("--override", action="append", default=[], metavar="HYDRA_OVERRIDE", help="extra Hydra override, applied last")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE", help="extra environment variable of the trainer and rollout processes")
    ap.add_argument("--sif", help="Apptainer image; omit together with --no-container to run --python directly")
    ap.add_argument("--python", default="python3", help="Python interpreter of the training environment")
    ap.add_argument("--no-container", action="store_true")
    ap.add_argument("--bind", action="append", default=[], help="extra Apptainer bind (src[:dst])")
    ap.add_argument("--cache-dir", help="HF/Triton/Inductor caches (default: RUN_DIR/../.cache)")
    ap.add_argument("--gpus", default=None, help="CUDA_VISIBLE_DEVICES (default: 0..gpus_per_node-1)")
    ap.add_argument("--steps", type=int, help="stop after this global step (smoke tests)")
    ap.add_argument("--fresh", action="store_true", help="start from MODEL_PATH instead of RESUME_PATH")
    ap.add_argument("--no-save", action="store_true", help="disable checkpointing")
    ap.add_argument("--logger", help="trainer.logger, e.g. console or console,wandb")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    recipe_path = os.path.abspath(args.recipe)
    recipe = yaml.safe_load(open(recipe_path))
    run_dir = os.path.abspath(args.run_dir)
    user = getpass.getuser()
    values = {
        "ORL_TRAINING": TRAINING_ROOT,
        "ORL_RECIPE_DIR": os.path.dirname(recipe_path),
        "RUN_DIR": run_dir,
        "EXPERIMENT_NAME": os.path.basename(run_dir),
        "NET_IFNAME": default_ifname(),
        "RAY_TMP_DIR": f"/dev/shm/orl-ray-{user}-{hashlib.md5(run_dir.encode()).hexdigest()[:8]}",
        "NODE_TMP_DIR": f"/dev/shm/orl-tmp-{user}",
    }
    for kv in args.set:
        k, _, v = kv.partition("=")
        values[k] = v
    for k in recipe.get("inputs", {}):
        if k not in values and os.environ.get(k):
            values[k] = os.environ[k]
    if args.fresh or "RESUME_PATH" not in recipe.get("inputs", {}):
        values.setdefault("RESUME_PATH", values.get("MODEL_PATH", ""))

    missing = set()
    env = {k: resolve(str(v), values, missing) for k, v in recipe.get("env", {}).items()}
    env.update(kv.partition("=")[::2] for kv in args.env)
    overrides = [resolve(o, values, missing) for o in recipe["overrides"]]
    if missing:
        sys.exit(f"unresolved recipe inputs: {sorted(missing)} (pass --set KEY=VALUE)")

    cache = os.path.abspath(args.cache_dir or os.path.join(os.path.dirname(run_dir), ".cache"))
    gpus = args.gpus or ",".join(str(i) for i in range(int(recipe.get("gpus_per_node", 8))))
    sglang_dir = f"{TRAINING_ROOT}/{recipe.get('sglang_dir', 'sglang')}"
    python_path = [f"{TRAINING_ROOT}/kernels/fast-diffusion-attention", sglang_dir, f"{TRAINING_ROOT}/verl"]
    python_path += [resolve(p, values, missing) for p in recipe.get("extra_python_path", [])]
    if missing:
        sys.exit(f"unresolved recipe inputs: {sorted(missing)} (pass --set KEY=VALUE)")
    env.update(
        {
            "PYTHONPATH": ":".join(python_path),
            "VERL_SGLANG_PATCHED_PATH": sglang_dir,
            "CUDA_VISIBLE_DEVICES": gpus,
            "WANDB_DIR": f"{run_dir}/wandb",
            "WANDB_CACHE_DIR": f"{run_dir}/wandb/cache",
            "WANDB_CONFIG_DIR": f"{run_dir}/wandb/config",
            "WANDB_DATA_DIR": f"{run_dir}/wandb/data",
            "HF_HOME": f"{cache}/hf_home",
            "XDG_CACHE_HOME": f"{cache}/xdg",
            "TRITON_CACHE_DIR": f"{cache}/triton",
            "TORCHINDUCTOR_CACHE_DIR": f"{cache}/inductor",
        }
    )
    if args.fresh:
        overrides += ["++trainer.resume_mode=disable", f"++trainer.resume_from_path={values['MODEL_PATH']}"]
        env.pop("VERL_ACTOR_RESUME_LR_OVERRIDE", None)
    if args.steps is not None:
        overrides.append(f"++trainer.total_training_steps={args.steps}")
    if args.no_save:
        overrides.append("++trainer.save_freq=-1")
    if args.logger:
        overrides.append(f"++trainer.logger=[{args.logger}]")
    overrides += args.override
    # keep Hydra's per-run output directory out of the source tree
    overrides.append(f"hydra.run.dir={run_dir}/hydra")

    verl_dir = f"{TRAINING_ROOT}/verl"
    trainer = [
        args.python, "-m", "verl.trainer.main_ppo",
        f"--config-path={verl_dir}/examples/diffusion_trainer",
        f"--config-name={recipe['base_config']}",
    ] + overrides
    if args.no_container:
        cmd = trainer
        proc_env = dict(os.environ, **env)
    else:
        if not args.sif:
            sys.exit("--sif is required unless --no-container is given")
        paths = [TRAINING_ROOT, run_dir, cache, args.python] + [
            v for v in values.values() if isinstance(v, str) and v.startswith("/")
        ]
        binds = [f"{r}:{r}" for r in bind_roots(paths)] + args.bind
        cmd = ["apptainer", "exec", "--nv"]
        for b in binds:
            cmd += ["-B", b]
        for k, v in sorted(env.items()):
            cmd += ["--env", f"{k}={v}"]
        cmd += [args.sif] + trainer
        proc_env = dict(os.environ)

    os.makedirs(run_dir, exist_ok=True)
    with open(f"{run_dir}/launch.json", "w") as f:
        json.dump({"recipe": recipe_path, "values": values, "env": env, "cwd": verl_dir, "cmd": cmd}, f, indent=1)
    if args.dry_run:
        print(" ".join(shlex.quote(c) for c in cmd))
        return 0
    for d in (cache, values["RAY_TMP_DIR"], values["NODE_TMP_DIR"]):
        os.makedirs(d, exist_ok=True)
    with open(f"{run_dir}/run.log", "ab") as log:
        p = subprocess.Popen(cmd, cwd=verl_dir, env=proc_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for chunk in iter(lambda: p.stdout.readline(), b""):
            sys.stdout.buffer.write(chunk)
            sys.stdout.flush()
            log.write(chunk)
        return p.wait()


if __name__ == "__main__":
    sys.exit(main())
