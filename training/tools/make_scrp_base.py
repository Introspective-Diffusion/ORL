#!/usr/bin/env python3
"""Build the QwenDLM start model with the gamma-rescaled residual adapter.

The QwenDLM SFT checkpoint (checkpoint-38750) carries a gated-MLP residual
adapter `self_conditioning.{gate,up,down}_proj`. ORL rescales it so that its
relative update size under Adam matches the backbone without changing the
function: the three weight matrices are multiplied by gamma and the adapter's
normalized input and output are divided by gamma
(config.self_conditioning_prenorm_scale = self_conditioning_reparam_out_scale = 1/gamma).
Powers of two keep the bf16 rescaling exact.

The model code of the result is training/overlays/sdar/: the code the paper
runs loaded, with its comments in English.

usage: make_scrp_base.py SRC_CHECKPOINT_DIR DST_DIR [--gamma 8]
"""
import argparse
import json
import os
import shutil

from safetensors.torch import load_file, save_file

SC_KEYS = {
    "self_conditioning.gate_proj.weight",
    "self_conditioning.up_proj.weight",
    "self_conditioning.down_proj.weight",
}
OVERLAY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "overlays", "sdar")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--gamma", type=int, default=8)
    args = ap.parse_args()
    if args.gamma <= 0 or args.gamma & (args.gamma - 1):
        raise SystemExit("--gamma must be a power of two")
    os.makedirs(args.dst, exist_ok=True)
    for name in os.listdir(args.src):
        path = os.path.join(args.src, name)
        if os.path.isfile(path) and not name.endswith(".safetensors"):
            shutil.copy2(path, os.path.join(args.dst, name))
    for name in os.listdir(OVERLAY):
        shutil.copyfile(os.path.join(OVERLAY, name), os.path.join(args.dst, name))

    index = json.load(open(os.path.join(args.src, "model.safetensors.index.json")))
    scaled = 0
    for shard in sorted(set(index["weight_map"].values())):
        tensors = load_file(os.path.join(args.src, shard))
        for key in tensors:
            if key in SC_KEYS:
                tensors[key] = tensors[key] * args.gamma
                scaled += 1
        save_file(tensors, os.path.join(args.dst, shard), metadata={"format": "pt"})
    if scaled != len(SC_KEYS):
        raise SystemExit(f"scaled {scaled} adapter tensors, expected {len(SC_KEYS)}")

    cfg_path = os.path.join(args.dst, "config.json")
    cfg = json.load(open(cfg_path))
    cfg["self_conditioning_prenorm_scale"] = 1.0 / args.gamma
    cfg["self_conditioning_reparam_out_scale"] = 1.0 / args.gamma
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"wrote {args.dst} (gamma={args.gamma})")


if __name__ == "__main__":
    main()
