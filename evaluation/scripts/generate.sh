#!/usr/bin/env bash
# Run one main-table generation job on one GPU in an isolated container (only that GPU visible, no network).
# usage: generate.sh GPU SPEC
#   qwen-dlm:NAME:ROLE:BENCH:SHARD    BENCH gsm8k|math500|he|mbpp, SHARD 0-7
#   llada:NAME:ROLE:BENCH
#   dg:NAME:ROLE:SEED:BENCH           SEED 1234|1235|1236, BENCH gsm8k-m4r0..gsm8k-m4r3|math500|he|mbpp
# NAME is a model directory under $MODELS and labels the outputs ($RUNS/FAMILY/NAME-BENCH-sSHARD_OR_SEED);
# ROLE (qwen-dlm-base, qwen-dlm-orl, llada-base, llada-orl, dg-base, dg-orl) selects the recorded decoder and sampler.
# Exits 11 without running when the job already finished; an unfinished earlier attempt is moved aside and rerun.
# DRY_RUN=1 only builds the recipe.
set -uo pipefail
source "$(dirname "$0")/config.sh"
GPU=$1 SPEC=$2
IFS=: read -r FAMILY NAME ROLE A1 A2 <<< "$SPEC"
MODEL=$MODELS/$NAME
GEN_SIF=$IMAGES/sglang-gen.sif

build_recipe() {  # OUT_JSON args...
  local out=$1; shift
  [ -s "$out" ] && return 0
  mkdir -p "$(dirname "$out")"
  rm -f "$out.partial"
  apptainer exec -B "$BINDS" --pwd "$ORL_EVAL" --env PYTHONDONTWRITEBYTECODE=1,PYTHONPATH=$ORL_EVAL "$GEN_SIF" \
    python3 -m repro.eval.portable.build_recipe "$@" --scorer-image "$SCORER_IMAGE" --score-args-json '[]' --out "$out.partial" \
    && mv "$out.partial" "$out"
}

container() {  # SIF CMD...  (this GPU only, as GPU 0; private loopback network)
  local sif=$1; shift
  NVIDIA_VISIBLE_DEVICES=$GPU apptainer exec --userns --nvccli --contain --net --network none --workdir "$scratch" \
    -B "$BINDS" --pwd "$ORL_EVAL" --env PYTHONDONTWRITEBYTECODE=1,PYTHONPATH=$ORL_EVAL,CUDA_VISIBLE_DEVICES=0 "$sif" "$@"
}

finished() {  # RUN_DIR
  python3 -c 'import json, sys; sys.exit(json.load(open(sys.argv[1])).get("exit_code") != 0)' \
    "$1/execution_receipt.json" 2>/dev/null || return 1
  [ "$FAMILY" != qwen-dlm ] || [ -e "$1/artifacts/GENERATION_COMPLETE.json" ]
}

dg_manifest() {  # ROLE_SHORT -> server manifest of $MODEL served as that role, written once per model and role
  local short=$1
  local manifest=$INPUTS/paper4/server-$NAME-$short.json lock=$INPUTS/paper4/.lock-server-$NAME-$short
  until [ -e "$manifest" ]; do
    if mkdir "$lock" 2>/dev/null; then
      local tmp; tmp=$(mktemp -d); rmdir "$tmp"
      if (cd "$ORL_EVAL" && PYTHONPATH=$ORL_EVAL python3 -m runtime.vllm.serve --model-root "$MODEL" --role "$short" \
            --benchmark gsm8k --out "$tmp" > "$tmp.log" 2>&1) &&
         python3 -c "import json, sys; json.dump(json.load(open(sys.argv[1]))['receipt'], open(sys.argv[2], 'w'), indent=2)" \
           "$tmp/launch_plan.json" "$manifest.tmp"; then
        mv "$manifest.tmp" "$manifest"
      else
        echo "planning the DiffusionGemma server for $MODEL failed:" >&2
        tail -20 "$tmp.log" >&2
        rm -rf "$tmp" "$tmp.log" "$manifest.tmp"; rmdir "$lock"
        return 1
      fi
      rm -rf "$tmp" "$tmp.log"; rmdir "$lock"
    else
      # a lock older than 20 minutes belongs to a dead job: hashing the model takes a few minutes at most
      [ -n "$(find "$lock" -maxdepth 0 -mmin +20 2>/dev/null)" ] && rmdir "$lock" 2>/dev/null
      sleep 15
    fi
  done
  echo "$manifest"
}

case "$FAMILY" in
  qwen-dlm)
    BENCH=$A1 SHARD=$A2 OUT=$NAME-$A1-s$A2
    plan=$BENCH; [ "$BENCH" = he ] && plan=humaneval
    RECIPE=$RECIPES/qwen-dlm/$OUT.json RUN=$RUNS/qwen-dlm/$OUT
    build_recipe "$RECIPE" --role "$ROLE" --benchmark "$BENCH" --seed 1234 --shard-index "$SHARD" \
      --runtime-root "$RUNTIMES/qwen-dlm" --input-root "$INPUTS/qwen-dlm" --model-root "$MODEL" \
      --runtime-script repro/eval/run_eval.py --problems "$BENCH.jsonl" --decoder-config "decoder-$ROLE-$plan.yaml" \
      --score-script score_qwen_dlm_he.py --engine-subdir engine || exit 10
    SIF=$GEN_SIF RUNTIME=$RUNTIMES/qwen-dlm INPUT=$INPUTS/qwen-dlm ;;
  llada)
    BENCH=$A1 OUT=$NAME-$A1-s12345
    plan=$BENCH file=$BENCH.jsonl; [ "$BENCH" = he ] && plan=humaneval file=humaneval.jsonl
    RECIPE=$RECIPES/llada/$OUT.json RUN=$RUNS/llada/$OUT
    build_recipe "$RECIPE" --role "$ROLE" --benchmark "$BENCH" --seed 12345 \
      --runtime-root "$RUNTIMES/llada" --input-root "$INPUTS/paper4" --model-root "$MODEL" \
      --runtime-script multi_gen.py --problems "$file" --decoder-config "decoder-$ROLE-$plan.yaml" \
      --score-script multi_gen.py --engine-subdir engine || exit 10
    SIF=$GEN_SIF RUNTIME=$RUNTIMES/llada INPUT=$INPUTS/paper4 ;;
  dg)
    SEED=$A1 BENCH_TAG=$A2 BENCH=${A2%%-*} OUT=$NAME-$A2-s$A1
    short=${ROLE#dg-}
    file=$BENCH.jsonl; [ "$BENCH" = he ] && file=humaneval.jsonl
    part=(); [ "$BENCH_TAG" != "$BENCH" ] && { file=${BENCH}_${BENCH_TAG#*-}.jsonl; part=(--smoke); }
    script=vllm_paper4.py; [ "$BENCH" = mbpp ] && script=vllm_full_extra_client.py
    manifest=$(dg_manifest "$short") || exit 10
    RECIPE=$RECIPES/dg/$OUT.json RUN=$RUNS/dg/$OUT
    build_recipe "$RECIPE" --role "$ROLE" --benchmark "$BENCH" --seed "$SEED" "${part[@]}" \
      --runtime-root "$RUNTIMES/dg" --input-root "$INPUTS/paper4" --model-root "$MODEL" \
      --problems "$file" --runtime-script "$script" --score-script vllm_paper4.py \
      --server-manifest "$(basename "$manifest")" || exit 10
    SIF=$IMAGES/vllm-$ROLE.sif RUNTIME=$RUNTIMES/dg INPUT=$INPUTS/paper4 ;;
  *) echo "unknown job $SPEC" >&2; exit 2 ;;
esac

[ -n "${DRY_RUN:-}" ] && { echo "recipe ready: $RECIPE"; exit 0; }
if [ -e "$RUN" ]; then
  finished "$RUN" && { echo "already generated: $RUN"; exit 11; }
  aside=$RUN.unfinished-$(date +%Y%m%d-%H%M%S)
  mv "$RUN" "$aside" && echo "moved an unfinished earlier attempt to $aside"
fi
scratch=$(mktemp -d "${TMPDIR:-/tmp}/orl-eval-$OUT.XXXX")
if [ "$FAMILY" = dg ]; then
  server=$RUNS/dg/servers/$OUT-gpu$GPU-$(date +%Y%m%d-%H%M%S)
  mkdir -p "$(dirname "$server")"
  container "$SIF" bash -c '
    set -u
    model=$1 short=$2 server=$3 recipe=$4 runtime=$5 inputs=$6 run=$7
    python3 -m runtime.vllm.serve --model-root "$model" --role "$short" --benchmark gsm8k --out "$server" --execute > "$server.log" 2>&1 &
    sp=$!
    for i in $(seq 900); do [ -e "$server/server_receipt.json" ] && break; kill -0 $sp 2>/dev/null || exit 21; sleep 2; done
    [ -e "$server/server_receipt.json" ] || { kill $sp; exit 22; }
    python3 -m repro.eval.generate --recipe "$recipe" --runtime-root "$runtime" --input-root "$inputs" \
      --model-root "$model" --output-root "$run" --execute; status=$?
    cp "$server/server_receipt.json" "$run/server_receipt.json" 2>/dev/null
    kill $sp; wait $sp; exit $status' _ "$MODEL" "$short" "$server" "$RECIPE" "$RUNTIME" "$INPUT" "$RUN"
else
  container "$SIF" python3 -m repro.eval.generate --recipe "$RECIPE" --runtime-root "$RUNTIME" \
    --input-root "$INPUT" --model-root "$MODEL" --output-root "$RUN" --execute
fi
status=$?
rm -rf "$scratch"
exit $status
