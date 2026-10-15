#!/usr/bin/env bash
# Reproduce the paper's QwenDLM + ORL checkpoint on one node; on 8 x H100 80GB with the runtime of training/env/ the
# result is bit for bit the paper's. Stage 1 steps 1-300, stage 1 resumed from step 300 for steps 301-325, stage 2 steps
# 326-350, each segment launched from the previous checkpoint as the paper run was. qwen_dlm_paper_run.json replays the
# rollout randomness of the paper run that its seed does not fix.
# Finished segments are skipped, so the script can be rerun after an interruption.
#
# usage: reproduce_qwen_dlm.sh RUN_ROOT START_MODEL DATA_DIR [orl_train.py options, e.g. --sif S --python P]
#   START_MODEL  QwenDLM-4B after tools/make_scrp_base.py --gamma 8
#   DATA_DIR     the training data (yifanyu/ORL-training-data)
set -euo pipefail
RUNS=$1 MODEL=$2 DATA=$3
shift 3
TRAINING=$(cd "$(dirname "$0")/.." && pwd)
RECIPES=$TRAINING/recipes/qwen_dlm
python3 -c 'import json, sys; c = json.load(open(sys.argv[1])); sys.exit(c.get("self_conditioning_prenorm_scale") != 0.125 or c.get("self_conditioning_reparam_out_scale") != 0.125)' \
  "$MODEL/config.json" || { echo "START_MODEL must be QwenDLM-4B after tools/make_scrp_base.py --gamma 8" >&2; exit 2; }
(cd "$DATA" && sha256sum --quiet -c SHA256SUMS) || { echo "DATA_DIR does not hold the paper's training data" >&2; exit 2; }
common=(--set MODEL_PATH="$MODEL" --set TRAIN_FILE="$DATA/code_l0-20260828/train.parquet"
        --set VAL_FILE="$DATA/code_he_clean_rebuild/val.parquet" --logger console
        --env ORL_QWEN_DLM_REPLAY_SNAPSHOTS="$TRAINING/scripts/qwen_dlm_paper_run.json" --env ORL_QWEN_DLM_REPLAY_REWARD=1)

if [ ! -d "$RUNS/stage1/checkpoints/global_step_300" ]; then
  python3 "$TRAINING/scripts/orl_train.py" "$RECIPES/stage1.yaml" --run-dir "$RUNS/stage1" "${common[@]}" \
    --override ++trainer.total_training_steps=300 "$@"
fi
if [ ! -d "$RUNS/stage1b/checkpoints/global_step_325" ]; then
  python3 "$TRAINING/scripts/orl_train.py" "$RECIPES/stage1.yaml" --run-dir "$RUNS/stage1b" "${common[@]}" \
    --override ++trainer.resume_mode=resume_path \
    --override "++trainer.resume_from_path=$RUNS/stage1/checkpoints/global_step_300" \
    --override ++trainer.initial_global_step=300 "$@"
fi
if [ ! -d "$RUNS/stage2/checkpoints/global_step_350" ]; then
  python3 "$TRAINING/scripts/orl_train.py" "$RECIPES/stage2_rct.yaml" --run-dir "$RUNS/stage2" "${common[@]}" \
    --set RESUME_PATH="$RUNS/stage1b/checkpoints/global_step_325" "$@"
fi
echo "QwenDLM + ORL checkpoint: $RUNS/stage2/checkpoints/global_step_350"
