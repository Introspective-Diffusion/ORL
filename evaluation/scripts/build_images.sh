#!/usr/bin/env bash
# Build the Apptainer images the evaluation runs in, from digest-pinned public images:
#   sglang-gen.sif     SGLang, for QwenDLM and LLaDA generation
#   vllm-dg-base.sif   vLLM with the DiffusionGemma HF-alignment patch
#   vllm-dg-orl.sif    the same plus the adaptive-scalar patch, for DiffusionGemma + ORL
#   python-scorer.sif  the isolated scorer
# usage: build_images.sh   (existing images are kept)
set -euo pipefail
source "$(dirname "$0")/config.sh"
mkdir -p "$IMAGES"

room() {  # Apptainer unpacks every image it builds in $APPTAINER_TMPDIR (default /tmp)
  local tmp=${APPTAINER_TMPDIR:-/tmp} free
  free=$(df -P -BG "$tmp" | awk 'NR == 2 {print int($4)}')
  [ "$free" -ge 50 ] && return
  echo "$tmp has $free GB free, but building the SGLang image unpacks about 50 GB there; set APPTAINER_TMPDIR (and" \
       "APPTAINER_CACHEDIR, where pulled layers are cached) to directories on a larger local disk" >&2
  exit 2
}
pull() { [ -e "$IMAGES/$1" ] || { room; apptainer build "$IMAGES/$1" "docker://$2"; }; }
derive() {  # NAME BASE POST_COMMANDS
  [ -e "$IMAGES/$1" ] && return
  room
  local def; def=$(mktemp)
  printf 'Bootstrap: localimage\nFrom: %s\n\n%%files\n    %s /opt/idlm-patches\n\n%%post\n%s\n' \
    "$IMAGES/$2" "$ORL_EVAL/inference/vllm/patches" "$3" > "$def"
  apptainer build "$IMAGES/$1" "$def"
  rm -f "$def"
}

pull sglang.sif docker.io/lmsysorg/sglang@sha256:065d403274324cc9dec4f830ca62540e6ac586746e1aed268c98a2b347d5853f
pull vllm.sif docker.io/vllm/vllm-openai@sha256:9c719fc0c869092c7d0533f8357d6985a38d5ff03b20ffb6a4620c2b4806dd4b
pull python-scorer.sif "docker.io/library/$SCORER_IMAGE"
derive sglang-gen.sif sglang.sif '    python3 -m pip install --no-cache-dir PyYAML==6.0.3'
derive vllm-dg-base.sif vllm.sif '    python3 /opt/idlm-patches/vllm_dg_hf_align_patch.py'
derive vllm-dg-orl.sif vllm.sif '    python3 /opt/idlm-patches/vllm_dg_hf_align_patch.py
    python3 /opt/idlm-patches/vllm_dg_adaptive_scalar_patch.py'
ls -la "$IMAGES"
