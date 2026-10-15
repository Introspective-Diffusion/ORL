#!/usr/bin/env bash
# Reproduce the paper's DiffusionGemma + ORL training on two nodes (8 x H100 80GB each). Run the same command with
# `head` on the actor node and `worker` on the rollout node; RUN_DIR and the other paths must be visible at the same
# location on both nodes. The head trains steps 1-900 in one launch; the worker serves rollouts until the head is done.
# The paper run was relaunched at step 450 with the same recipe and restored data-loader state, which trains on the same
# rows in the same order (corpus order, constant learning rate). A finished run is skipped.
#
# usage: reproduce_diffusion_gemma.sh head|worker RUN_DIR SIF PYTHON START_MODEL DATA_DIR OVERLAY HEAD_IP WORKER_IP [orl_train.py options]
#   START_MODEL  google/diffusiongemma-26B-A4B-it at revision 52de6b914ee1749a7d4933202505ddf5b414ec43
#   DATA_DIR     the training data (yifanyu/ORL-training-data); OVERLAY  from install_dgemma_transformers_overlay.sh
#   NET_IFNAME   (environment) network interface for NCCL and Gloo (default: each node's default-route interface)
set -uo pipefail
usage() { sed -n '7,10p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }
[ $# -ge 9 ] || usage
ROLE=$1 RUN=$2 SIF=$3 PY=$4 MODEL=$5 DATA=$6 OVERLAY=$7 HEAD_IP=$8 WORKER_IP=$9
shift 9
case "$ROLE" in head|worker) ;; *) usage ;; esac
TRAINING=$(cd "$(dirname "$0")/.." && pwd)
RECIPE=$TRAINING/recipes/diffusion_gemma/aec_c100.yaml
TRAIN=$DATA/gsm8k-math-code_l0-mix-20260909/train.parquet
C=$RUN/coord
mkdir -p "$C"
paths=(--set DG_TRANSFORMERS_OVERLAY="$OVERLAY" --set MODEL_PATH="$MODEL" --set TRAIN_FILE="$TRAIN" --set RUN_DIR="$RUN")
[ -n "${NET_IFNAME:-}" ] && paths+=(--set NET_IFNAME="$NET_IFNAME")
ray_node() { python3 "$TRAINING/scripts/ray_cluster.py" "$1" "$RECIPE" --sif "$SIF" --python "$PY" "${paths[@]}" "${@:2}"; }

if [ "$ROLE" = worker ]; then
  trap 'ray_node stop > "$C/worker_stop.log" 2>&1' EXIT
  # head_ready and done hold the head's session id; a matching pair is a finished run, so wait for a new head
  while :; do
    session=$(cat "$C/head_ready" 2>/dev/null)
    [ -n "$session" ] && [ "$session" != "$(cat "$C/done" 2>/dev/null)" ] && break
    sleep 5
  done
  ray_node worker --ip "$HEAD_IP" --node-ip "$WORKER_IP" > "$C/ray_worker.log" 2>&1 &
  until [ "$(cat "$C/done" 2>/dev/null)" = "$session" ]; do sleep 10; done
  exit 0
fi

(cd "$DATA" && sha256sum --quiet -c SHA256SUMS) || { echo "DATA_DIR does not hold the paper's training data" >&2; exit 2; }
session=$(date +%s%N)
rm -f "$C/head_ready" "$C/done"
trap 'echo "$session" > "$C/done"; ray_node stop > "$C/head_stop.log" 2>&1' EXIT
ray_node head --ip "$HEAD_IP" > "$C/ray_head.log" 2>&1 &
for _ in $(seq 60); do grep -q "Ray runtime started" "$C/ray_head.log" 2>/dev/null && break; sleep 5; done
grep -q "Ray runtime started" "$C/ray_head.log" || { echo "the Ray head did not start; see $C/ray_head.log" >&2; exit 3; }
echo "$session" > "$C/head_ready"
ray_node wait --ip "$HEAD_IP" --gpus 16 || exit 4
args=(--run-dir "$RUN" --sif "$SIF" --python "$PY" --set MODEL_PATH="$MODEL" --set TRAIN_FILE="$TRAIN"
      --set RAY_ADDRESS="$HEAD_IP:6606" --set ACTOR_NODE_IP="$HEAD_IP" --set ROLLOUT_NODE_IP="$WORKER_IP"
      --set DG_TRANSFORMERS_OVERLAY="$OVERLAY" --logger console)
[ -n "${NET_IFNAME:-}" ] && args+=(--set NET_IFNAME="$NET_IFNAME")
args+=("$@")
if [ ! -d "$RUN/checkpoints/global_step_900" ]; then
  python3 "$TRAINING/scripts/orl_train.py" "$RECIPE" "${args[@]}" || exit 1
fi
if [ -d "$RUN/checkpoints/global_step_900" ]; then
  echo "DiffusionGemma + ORL checkpoint: $RUN/checkpoints/global_step_900"
else
  echo "no checkpoint at $RUN/checkpoints/global_step_900 (expected after --steps or --no-save)"
fi
