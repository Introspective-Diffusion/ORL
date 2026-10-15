#!/usr/bin/env bash
# Reproduce the paper's LLaDA-2.1-mini + ORL training on one node (8 x H100 80GB): llada2_mini/orl.yaml, steps 1-625 in
# one launch; step 625 is the paper checkpoint. The paper run was launched twice (steps 1-450, then from step 451 on the
# rows it had not reached yet), which trains on the same rows in the same order: the corpus is read in order, one prompt
# per step, at a constant learning rate. A finished run is skipped.
#
# usage: reproduce_llada2.sh RUN_DIR ACTOR_MODEL DATA_DIR [orl_train.py options]
#   ACTOR_MODEL  from tools/prepare_llada2_actor.sh; DATA_DIR  the training data (yifanyu/ORL-training-data)
set -euo pipefail
RUN=$1 MODEL=$2 DATA=$3
shift 3
TRAINING=$(cd "$(dirname "$0")/.." && pwd)
(cd "$DATA" && sha256sum --quiet -c SHA256SUMS) || { echo "DATA_DIR does not hold the paper's training data" >&2; exit 2; }
if [ ! -d "$RUN/checkpoints/global_step_625" ]; then
  python3 "$TRAINING/scripts/orl_train.py" "$TRAINING/recipes/llada2_mini/orl.yaml" --run-dir "$RUN" \
    --set MODEL_PATH="$MODEL" --set TRAIN_FILE="$DATA/gsm8k-math-code_l0-mix-20260909/train.parquet" \
    --set VAL_FILE="$DATA/code_he_clean_rebuild/val.parquet" --logger console "$@"
fi
if [ -d "$RUN/checkpoints/global_step_625" ]; then
  echo "LLaDA-2.1-mini + ORL checkpoint: $RUN/checkpoints/global_step_625"
else
  echo "no checkpoint at $RUN/checkpoints/global_step_625 (expected after --steps or --no-save)"
fi
