#!/usr/bin/env bash
# Build the runtime Transformers overlay that DiffusionGemma training imports.
#
# Copies the transformers package of the training environment (expected 5.8.1)
# into OUT/transformers and overlays training/overlays/diffusion_gemma/transformers
# (DiffusionGemma model, replay external-KV attention, conversion mapping).
# Recipes put OUT on PYTHONPATH via DG_TRANSFORMERS_OVERLAY=OUT.
#
# usage: install_dgemma_transformers_overlay.sh OUT --sif SIF --python PY [--allow-version-mismatch]
#        install_dgemma_transformers_overlay.sh OUT --no-container --python PY
set -euo pipefail
OUT="${1:?usage: $0 OUT --sif SIF --python PY}"; shift
SIF=""; PY=""; CONTAINER=1; ALLOW=false; EXPECTED=5.8.1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sif) SIF="$2"; shift 2 ;;
    --python) PY="$2"; shift 2 ;;
    --no-container) CONTAINER=0; shift ;;
    --allow-version-mismatch) ALLOW=true; shift ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ -n "$PY" ]] || { echo "--python is required" >&2; exit 2; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "${HERE}/../overlays/diffusion_gemma/transformers" && pwd)"
OUT="$(mkdir -p "$OUT" && cd "$OUT" && pwd)"
BINDS="/$(echo "$OUT" | cut -d/ -f2)"
[[ "$PY" == /* && "/$(echo "$PY" | cut -d/ -f2)" != "$BINDS" ]] && BINDS+=",/$(echo "$PY" | cut -d/ -f2)"
run_py() {
  if [[ "$CONTAINER" == 1 ]]; then
    apptainer exec -B "$BINDS" "$SIF" "$PY" "$@"
  else
    "$PY" "$@"
  fi
}
if [[ ! -f "$OUT/transformers/__init__.py" ]]; then
  STAGE="$OUT/.staging-$$"
  trap 'rm -rf "$STAGE"' EXIT
  run_py - "$STAGE/transformers" "$EXPECTED" "$ALLOW" <<'PY'
import pathlib, shutil, sys
import transformers
dst, expected, allow = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3] == "true"
if transformers.__version__ != expected and not allow:
    sys.exit(f"transformers {transformers.__version__} != expected {expected}; review the overlay or pass --allow-version-mismatch")
shutil.copytree(pathlib.Path(transformers.__file__).resolve().parent, dst)
print(f"copied transformers {transformers.__version__}")
PY
  mv "$STAGE/transformers" "$OUT/transformers"
  rm -rf "$STAGE"
  trap - EXIT
fi
mkdir -p "$OUT/transformers/models/diffusion_gemma"
cp "$SRC/conversion_mapping.py" "$OUT/transformers/conversion_mapping.py"
cp "$SRC/models/diffusion_gemma/"*.py "$OUT/transformers/models/diffusion_gemma/"
PYTHONPATH="$OUT" run_py - "$OUT" <<'PY'
import pathlib, sys
import transformers
from transformers.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaForBlockDiffusion  # noqa: F401
loaded = pathlib.Path(transformers.__file__).resolve()
assert str(loaded).startswith(str(pathlib.Path(sys.argv[1]).resolve())), f"overlay not first on PYTHONPATH: {loaded}"
print(f"overlay ready: {loaded.parent}")
PY
