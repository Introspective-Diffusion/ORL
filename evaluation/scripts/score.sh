#!/usr/bin/env bash
# Score one main-table cell (every generation run of a model on one benchmark and seed) in the isolated scorer, then
# aggregate it. Writes $SCORING/attempts/NAME-BENCH-sSEED-TIME.metrics.json and prints the metrics. A cell whose
# generation records changed since it was bound is bound again.
# usage: score.sh FAMILY NAME BENCH [SEED]   (SEED is for DiffusionGemma: 1234|1235|1236)
set -euo pipefail
source "$(dirname "$0")/config.sh"
family=$1 name=$2 bench=$3
case $family in
  qwen-dlm) seed=1234 problems=qwen-dlm/$bench.jsonl
            glob="$RUNS/qwen-dlm/$name-$bench-s[0-7]/artifacts/records.jsonl" ;;
  llada)    seed=12345 problems=paper4/$bench.jsonl
            glob="$RUNS/llada/$name-$bench-s12345/artifacts/$bench/shard*.jsonl" ;;
  dg)       seed=${4:?DiffusionGemma cells need a seed} problems=paper4/$bench.jsonl
            tag=$bench; [ "$bench" = gsm8k ] && tag="gsm8k-m4r[0-3]"
            glob="$RUNS/dg/$name-$tag-s$seed/artifacts/$bench/shard*.jsonl" ;;
  *) echo "unknown family $family" >&2; exit 2 ;;
esac
[ "$family" != qwen-dlm ] && [ "$bench" = he ] && problems=paper4/humaneval.jsonl
expected=$(awk -v f="$problems" '$2 == f {print $1}' "$ORL_EVAL/inputs/SHA256SUMS")
cell=$name-$bench-s$seed
bound=$SCORING/bound/$cell
export PYTHONPATH=$ORL_EVAL
cd "$ORL_EVAL"
records=() files=()
for path in $glob; do
  if [ -e "$path" ]; then records+=(--records "$path"); files+=("$path"); fi
done
[ ${#files[@]} -gt 0 ] || { echo "no generation records match $glob" >&2; exit 3; }
if [ -e "$bound" ] && [ -n "$(find "${files[@]}" -newer "$bound/score_recipe.json" 2>/dev/null | head -1)" ]; then
  rm -rf "$bound"  # regenerated since the last binding
fi
[ -e "$bound/score_recipe.json" ] || { rm -rf "$bound"; python3 -m repro.eval.portable.bind_benchmark_score --family "$family" \
  --benchmark "$bench" --evaluation-id "$cell" --generation-protocol-id "$name-$bench-main-v1-s$seed" "${records[@]}" \
  --original-problems "$INPUTS/$problems" --expected-problems-sha256 "$expected" \
  --runtime-root "$SCORING/runtime" --scorer-image "$SCORER_IMAGE" --out "$bound"; }
mkdir -p "$SCORING/unused-model" "$SCORING/attempts"
attempt=$SCORING/attempts/$cell-$(date +%Y%m%d-%H%M%S)
if ! python3 -m repro.eval.score --recipe "$bound/score_recipe.json" --runtime-root "$bound/runtime" \
    --input-root "$bound/inputs" --model-root "$SCORING/unused-model" --output-root "$attempt" \
    --container-engine apptainer --apptainer-image "$IMAGES/python-scorer.sif" --execute > "$attempt.cli.log" 2>&1; then
  echo "scoring failed: $attempt" >&2
  tail -5 "$attempt.cli.log" >&2
  tail -5 "$attempt/stage.log" >&2 2>/dev/null || true
  exit 1
fi
python3 -m repro.eval.aggregate --manifest "$attempt/artifacts/aggregation_manifest.json" \
  --input-root "$attempt/artifacts" > "$attempt.metrics.partial"
mv "$attempt.metrics.partial" "$attempt.metrics.json"
echo "$cell $(tr -d '\n' < "$attempt.metrics.json" | cut -c1-400)"
