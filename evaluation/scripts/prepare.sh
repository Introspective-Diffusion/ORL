#!/usr/bin/env bash
# Prepare the evaluation work directory: staged engine runtimes, the frozen benchmark inputs with rendered decoder
# configs, and the isolated scorer runtime (57 wheels from PyPI and HumanEval+ from its release, all hash-checked).
# usage: prepare.sh [runtimes] [inputs] [scorer]   (default: all three; finished parts are skipped, partial ones redone)
set -euo pipefail
source "$(dirname "$0")/config.sh"
cd "$ORL_EVAL"
export PYTHONPATH=$ORL_EVAL
HUMANEVAL_PLUS_SHA256=42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f

runtimes() {
  for family in qwen-dlm llada dg; do
    [ -e "$RUNTIMES/$family/runtime_manifest.json" ] && continue  # written last by runtime.stage
    rm -rf "${RUNTIMES:?}/$family"
    python3 -m runtime.stage --family "$family" --out "$RUNTIMES/$family"
  done
}

inputs() {
  for f in $(cd inputs && ls */*.jsonl); do
    [ -e "$INPUTS/$f" ] || install -D -m 644 "inputs/$f" "$INPUTS/$f"
  done
  (cd "$INPUTS" && sha256sum --quiet -c "$ORL_EVAL/inputs/SHA256SUMS")
  # DiffusionGemma's GSM8K runs as four interleaved parts, as in the paper
  python3 - "$INPUTS/paper4" <<'EOF'
import sys
from pathlib import Path
root = Path(sys.argv[1])
lines = (root / 'gsm8k.jsonl').read_text().splitlines(keepends=True)
for rem in range(4):
    (root / f'gsm8k_m4r{rem}.jsonl').write_text(''.join(lines[rem::4]))
EOF
  for role in qwen-dlm-base qwen-dlm-orl llada-base llada-orl; do
    case $role in qwen-dlm-*) dir=qwen-dlm seed=1234 ;; *) dir=paper4 seed=12345 ;; esac
    for bench in gsm8k math500 humaneval mbpp; do
      out=$INPUTS/$dir/decoder-$role-$bench.yaml
      [ -s "$out" ] && continue
      rm -f "$out"
      python3 -m repro.paper.render_config --suite main --model "$role" --benchmark "$bench" --seed "$seed" --out "$out"
    done
  done
}

scorer() {
  [ -e "$SCORING/runtime/staging_receipt.json" ] && return  # written last by stage_scoring
  rm -rf "$SCORING/runtime"
  local wheels=$SCORING/wheels he=$SCORING/data/HumanEvalPlus-v0.1.10.jsonl
  mkdir -p "$wheels" "$SCORING/data"
  cp runtime/scoring/wheels/*.whl "$wheels/"
  # tempdir and wget publish no wheels; their recorded wheels ship in runtime/scoring/wheels
  grep -v -e '^#' -e '^tempdir==' -e '^wget==' runtime/scoring/requirements-main.lock | grep . > "$SCORING/pypi.lock"
  python3 -m pip download --quiet --no-deps --only-binary=:all: --implementation cp --python-version 3.11 \
    --platform manylinux2014_x86_64 --platform manylinux_2_17_x86_64 --platform manylinux_2_28_x86_64 \
    --platform manylinux_2_12_x86_64 --platform manylinux_2_5_x86_64 --platform manylinux2010_x86_64 \
    --platform manylinux1_x86_64 --platform any --require-hashes -r "$SCORING/pypi.lock" -d "$wheels"
  if ! echo "$HUMANEVAL_PLUS_SHA256  $he" | sha256sum --quiet -c 2>/dev/null; then
    curl -fsSL https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus.jsonl.gz \
      | gunzip > "$he.partial"
    echo "$HUMANEVAL_PLUS_SHA256  $he.partial" | sha256sum --quiet -c
    mv "$he.partial" "$he"
  fi
  python3 "$ORL_EVAL/scripts/stage_scorer.py" --wheels "$wheels" --humaneval-plus "$he" --out "$SCORING/runtime"
}

parts=("$@"); [ ${#parts[@]} -gt 0 ] || parts=(runtimes inputs scorer)
for part in "${parts[@]}"; do
  case "$part" in runtimes|inputs|scorer) "$part"; echo "prepared $part" ;; *) echo "unknown part: $part" >&2; exit 2 ;; esac
done
