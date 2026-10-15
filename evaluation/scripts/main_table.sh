#!/usr/bin/env bash
# Evaluate one model on the paper's main table with the recorded protocol: copy the model into the work directory, run
# every generation job across the GPUs, score each cell and print accuracy and TPF next to the paper's numbers.
# Rerunning resumes: finished generations and up-to-date scores are kept, unfinished jobs are rerun.
# usage: main_table.sh ROLE MODEL_DIR [NAME]
#   ROLE  qwen-dlm-base | qwen-dlm-orl | llada-base | llada-orl | dg-base | dg-orl  (selects decoder and sampler)
#   NAME  label for the outputs, starting with ROLE (default: ROLE)
# env: GPUS (comma-separated, default CUDA_VISIBLE_DEVICES or all GPUs), BENCHMARKS (default "gsm8k math500 he mbpp")
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/config.sh"
[ $# -ge 2 ] || { sed -n '5,8p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }
ROLE=$1 SRC=$2 NAME=${3:-$1}
case "$ROLE" in qwen-dlm-base|qwen-dlm-orl|llada-base|llada-orl|dg-base|dg-orl) ;; *) echo "unknown role $ROLE" >&2; exit 2 ;; esac
[[ $NAME == "$ROLE"* ]] || { echo "NAME must start with $ROLE" >&2; exit 2; }
[[ $NAME =~ ^[A-Za-z0-9._-]+$ ]] || { echo "NAME may only contain letters, digits, '.', '_' and '-'" >&2; exit 2; }
FAMILY=${ROLE%-*}
GPUS=${GPUS:-${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd,)}}
[ -n "$GPUS" ] || { echo "no GPUs found; set GPUS (e.g. GPUS=0,1,2,3)" >&2; exit 2; }
BENCHMARKS=${BENCHMARKS:-gsm8k math500 he mbpp}

if [ ! -e "$MODELS/$NAME" ]; then  # a plain copy: the recipes hash every file of the model directory
  mkdir -p "$MODELS"
  rm -rf "$MODELS/$NAME.partial"
  if command -v rsync > /dev/null; then
    rsync -aL --exclude=.git --exclude=__pycache__ "$SRC/" "$MODELS/$NAME.partial/"
  else
    cp -rL "$SRC" "$MODELS/$NAME.partial" && rm -rf "$MODELS/$NAME.partial/.git" &&
      find "$MODELS/$NAME.partial" -name __pycache__ -prune -exec rm -rf {} +
  fi || { echo "copying $SRC failed" >&2; rm -rf "$MODELS/$NAME.partial"; exit 1; }
  mv "$MODELS/$NAME.partial" "$MODELS/$NAME"
fi

specs=() cells=()
for bench in $BENCHMARKS; do
  case $FAMILY in
    qwen-dlm) for shard in 0 1 2 3 4 5 6 7; do specs+=("qwen-dlm:$NAME:$ROLE:$bench:$shard"); done; cells+=("$bench") ;;
    llada)    specs+=("llada:$NAME:$ROLE:$bench"); cells+=("$bench") ;;
    dg)       for seed in 1234 1235 1236; do
                if [ "$bench" = gsm8k ]; then
                  for rem in 0 1 2 3; do specs+=("dg:$NAME:$ROLE:$seed:gsm8k-m4r$rem"); done
                else
                  specs+=("dg:$NAME:$ROLE:$seed:$bench")
                fi
                cells+=("$bench $seed")
              done ;;
  esac
done

log() { echo "$RUNS/logs/${1//:/_}.log"; }
job() {  # GPU SPEC
  bash "$HERE/generate.sh" "$1" "$2" > "$(log "$2")" 2>&1
  echo $? > "$(log "$2").status"
}
mkdir -p "$RUNS/logs"
set -m  # every job gets its own process group, so an interrupt can stop its container too
declare -A busy
trap 'for p in "${busy[@]}"; do kill -TERM -- "-$p" 2>/dev/null; done; exit 130' INT TERM
for spec in "${specs[@]}"; do
  rm -f "$(log "$spec").status"
  while :; do
    for gpu in ${GPUS//,/ }; do
      pid=${busy[$gpu]:-}
      if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
        echo "$(date +%T) GPU $gpu: $spec"
        job "$gpu" "$spec" &
        busy[$gpu]=$!
        continue 3
      fi
    done
    sleep 5
  done
done
wait
set +m
failed=0
for spec in "${specs[@]}"; do
  status=$(cat "$(log "$spec").status" 2>/dev/null)
  # 11: the job had already finished
  case $status in 0|11) ;; *) echo "generation failed (exit ${status:-?}): $spec, see $(log "$spec")"; failed=1 ;; esac
done

for cell in "${cells[@]}"; do
  read -r bench seed <<< "$cell"
  latest=$(ls -t "$SCORING/attempts/$NAME-$bench-s${seed:-}"*.metrics.json 2>/dev/null | head -1)
  # keep a score that is newer than every generation record of the cell
  [ -n "$latest" ] && [ -z "$(find "$RUNS/$FAMILY" -path "*/$NAME-$bench*" -name '*.jsonl' -newer "$latest" 2>/dev/null | head -1)" ] && continue
  bash "$HERE/score.sh" "$FAMILY" "$NAME" "$bench" ${seed:-} || { echo "scoring $NAME $bench ${seed:-} failed"; failed=1; }
done
python3 "$HERE/summarize.py" --name "$NAME"
exit $failed
