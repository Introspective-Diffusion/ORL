#!/usr/bin/env bash
# usage: train.sh RECIPE MODEL_DIR DATA_DIR OUT_DIR [KEY=VALUE ...]
#   RECIPE    offline/recipes/<stage>.yaml
#   MODEL_DIR the stage's start model (tools/make_stage_model.py)
#   DATA_DIR  the pretokenized training rows (parquet files with input_ids, attention_mask, labels)
#   OUT_DIR   checkpoints; a rerun resumes from the latest one
# One node by default, all local GPUs. Multi-node: run the same command on every node with NNODES, NODE_RANK and
# MASTER_ADDR (rank 0's address) set; OUT_DIR must be on storage that every node sees.
# Optional: TOKENIZED=<dir> saves the processed dataset there on the first run and reads it afterwards;
# GPUS (default: all local), RESTARTS (default 20) relaunches after a non-zero exit.
set -euo pipefail
[ $# -ge 4 ] || { sed -n '2,10p' "$0"; exit 1; }
RECIPE=$(realpath "$1") MODEL=$(realpath "$2") DATA=$(realpath "$3") OUT=$(realpath -m "$4")
shift 4
HERE=$(cd "$(dirname "$0")/.." && pwd)
LF=$HERE/llama_factory_sdar
NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0}
GPUS=${GPUS:-$(nvidia-smi -L | wc -l)}

mkdir -p "$OUT/.data" "$OUT/.cache"
cp "$LF/data/dataset_info.json" "$OUT/.data/dataset_info.json"
ln -sfn "$DATA" "$OUT/.data/offline_sft"
export PYTHONPATH="$LF/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_MODULES_CACHE=${HF_MODULES_CACHE:-$OUT/.cache/hf_modules.$NODE_RANK}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Populate the trust_remote_code module cache once per node before the ranks race to write it.
python3 - "$MODEL" <<'EOF'
import sys
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
AutoConfig.from_pretrained(sys.argv[1], trust_remote_code=True)
get_class_from_dynamic_module("modeling_sdar.SDARForCausalLM", sys.argv[1])
EOF

args=(model_name_or_path="$MODEL" output_dir="$OUT" dataset_dir="$OUT/.data")
[ -n "${TOKENIZED:-}" ] && args+=(tokenized_path="$TOKENIZED" data_shared_file_system=true)
if [ "$NNODES" -gt 1 ]; then
  launch=(--nnodes="$NNODES" --node_rank="$NODE_RANK" --rdzv_backend=c10d --rdzv_id=offline
          --rdzv_endpoint="${MASTER_ADDR:?set MASTER_ADDR to the address of node 0}:${MASTER_PORT:-29500}")
else
  launch=(--standalone)
fi

cd "$LF"
for attempt in $(seq 0 "${RESTARTS:-20}"); do
  echo "[train.sh] node $NODE_RANK attempt $attempt $(date -u +%FT%TZ)"
  torchrun "${launch[@]}" --nproc_per_node="$GPUS" src/llamafactory/launcher.py "$RECIPE" "${args[@]}" "$@" && exit 0
  echo "[train.sh] node $NODE_RANK exited with $?; resuming from the latest checkpoint in 30 s"
  sleep 30
done
exit 1
