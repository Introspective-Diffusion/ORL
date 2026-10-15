#!/usr/bin/env bash
# Prepare the LLaDA-2.1-mini start model for ORL training.
#
# Links every file of the Hugging Face snapshot inclusionAI/LLaDA2.1-mini
# (revision 20e64e2ad21644d0e5248586ed9c942cdd45de0f) into DST and replaces
# modeling_llada2_moe.py with training/overlays/llada2/modeling_llada2_moe.py,
# which adds the zero-initialized residual (self-conditioning) adapter, the
# block-diffusion training path and the hooks ORL replay needs. The stock file
# is kept as modeling_llada2_moe.stock_backup.py.
#
# usage: prepare_llada2_actor.sh HF_SNAPSHOT_DIR DST_DIR
set -euo pipefail
SRC="$(cd "${1:?usage: $0 HF_SNAPSHOT_DIR DST_DIR}" && pwd)"
DST="${2:?usage: $0 HF_SNAPSHOT_DIR DST_DIR}"
OVERLAY="$(cd "$(dirname "${BASH_SOURCE[0]}")/../overlays/llada2" && pwd)/modeling_llada2_moe.py"
mkdir -p "$DST"
for f in "$SRC"/*; do
  name="$(basename "$f")"
  [[ "$name" == "modeling_llada2_moe.py" ]] && continue
  ln -sfn "$(readlink -f "$f")" "$DST/$name"
done
cp "$(readlink -f "$SRC/modeling_llada2_moe.py")" "$DST/modeling_llada2_moe.stock_backup.py"
cp "$OVERLAY" "$DST/modeling_llada2_moe.py"
echo "modeling_llada2_moe.py md5: $(md5sum "$DST/modeling_llada2_moe.py" | cut -c1-12) (the paper runs' copy, 0952affd8fbd, differs in comments only)"
