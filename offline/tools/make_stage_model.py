#!/usr/bin/env python3
"""Build the model directory an offline stage starts from.

    make_stage_model.py s1 QWEN3_4B_THINKING_DIR OUT   # SDAR config, 7-token blocks, <|MASK|> token
    make_stage_model.py s2 S1_CHECKPOINT OUT           # 15-token blocks
    make_stage_model.py s3 S2_CHECKPOINT OUT           # 256-token blocks, loss on every noisy position
    make_stage_model.py s4 S3_CHECKPOINT OUT           # self-conditioning adapter with a zero residual

Weights are hardlinked from the source when the filesystem allows it, otherwise copied. Trainer state
(optimizer, scheduler, RNG, trainer_state.json) is never copied: every stage starts from weights only.
Every stage gets the model code of training/overlays/sdar, which keeps earlier stages' behaviour behind
config flags.
"""
import argparse
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_CODE = ["configuration_sdar.py", "modeling_sdar.py", "fused_linear_diffusion_cross_entropy.py"]
CHAT_TEMPLATE = ROOT / "offline/model/chat_template.jinja"
MASK_TOKEN, MASK_ID = "<|MASK|>", 151669
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
                   "vocab.json", "merges.txt", "chat_template.jinja"]
SC_SHARD = "model-self-conditioning.safetensors"

STAGE_CONFIG = {
    "s1": {
        "architectures": ["SDARForCausalLM"],
        "auto_map": {"AutoConfig": "configuration_sdar.SDARConfig", "AutoModel": "modeling_sdar.SDARModel",
                     "AutoModelForCausalLM": "modeling_sdar.SDARForCausalLM"},
        "model_type": "sdar",
        "torch_dtype": "bfloat16",
        "use_cache": False,
        "attn_implementation": "flex_attention",
        "training_attention_backend": "flex_attention",
        "fuse_cross_entropy": True,
        "block_size": 7,
        "mask_token_id": MASK_ID,
        "random_token_high": MASK_ID,
        "noise_strategy": "random",
        "masking_strategy": "random",
        "noisy_block_bidirectional": True,
    },
    "s2": {"block_size": 15},
    "s3": {
        "block_size": 256,
        "loss_on_all_noisy_tokens": True,
        "scale_loss_by_p_mask": False,
        "use_regular_causal": True,
        "local_random_window_size": 5,
    },
    "s4": {
        "self_conditioning_prob": 1.0,
        "self_conditioning_chunk_size": 128,
        "self_conditioning_init": "zero",
        "self_conditioning_init_std": 0.001,
        "self_conditioning_post_norm": True,
        "self_conditioning_post_norm_scale": 0.02,
    },
}


def link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def link_weights(src: Path, out: Path) -> dict:
    index = json.loads((src / "model.safetensors.index.json").read_text())
    for shard in sorted(set(index["weight_map"].values())):
        link_or_copy(src / shard, out / shard)
    return index


def build_tokenizer(base: Path, out: Path) -> None:
    from transformers import AddedToken, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base)
    tok.add_tokens([AddedToken(MASK_TOKEN, special=False, normalized=False)])
    if tok.convert_tokens_to_ids(MASK_TOKEN) != MASK_ID:
        raise SystemExit(f"{MASK_TOKEN} got id {tok.convert_tokens_to_ids(MASK_TOKEN)}, expected {MASK_ID}")
    tok.chat_template = CHAT_TEMPLATE.read_text()
    tok.save_pretrained(out)


def self_conditioning_shard(config: dict, out: Path, seed: int) -> dict:
    """The adapter as SDARForCausalLM._init_self_conditioning_weights initializes it for init="zero"."""
    import torch
    from safetensors.torch import save_file

    hidden = config["hidden_size"]
    inter = config.get("self_conditioning_intermediate_size", config["intermediate_size"])
    std = config["self_conditioning_init_std"]
    gen = torch.Generator().manual_seed(seed)
    tensors = {
        "self_conditioning.pre_norm.weight": torch.ones(hidden),
        "self_conditioning.gate_proj.weight": torch.randn(inter, hidden, generator=gen) * std,
        "self_conditioning.up_proj.weight": torch.randn(inter, hidden, generator=gen) * std,
        "self_conditioning.down_proj.weight": torch.zeros(hidden, inter),
    }
    tensors = {name: t.to(torch.bfloat16).contiguous() for name, t in tensors.items()}
    save_file(tensors, out / SC_SHARD, metadata={"format": "pt"})
    return {name: t.numel() * t.element_size() for name, t in tensors.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=sorted(STAGE_CONFIG))
    ap.add_argument("source", type=Path, help="Qwen3-4B-Thinking-2507 for s1, the previous stage's checkpoint otherwise")
    ap.add_argument("out", type=Path)
    ap.add_argument("--seed", type=int, default=0, help="seed of the adapter's gate/up initialization (s4)")
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; use a new directory")
    args.out.mkdir(parents=True)

    config = json.loads((args.source / "config.json").read_text())
    index = link_weights(args.source, args.out)
    if args.stage == "s1":
        build_tokenizer(args.source, args.out)
    else:
        for name in TOKENIZER_FILES:
            if (args.source / name).exists():
                shutil.copy2(args.source / name, args.out / name)
    if (args.source / "generation_config.json").exists():
        shutil.copy2(args.source / "generation_config.json", args.out / "generation_config.json")
    for name in MODEL_CODE:
        shutil.copy2(ROOT / "training/overlays/sdar" / name, args.out / name)

    config.update(STAGE_CONFIG[args.stage])
    if args.stage == "s4":
        if any(name.startswith("self_conditioning.") for name in index["weight_map"]):
            raise SystemExit(f"{args.source} already holds self-conditioning weights")
        sizes = self_conditioning_shard(config, args.out, args.seed)
        index["weight_map"].update({name: SC_SHARD for name in sizes})
        index.setdefault("metadata", {})["total_size"] = index.get("metadata", {}).get("total_size", 0) + sum(sizes.values())
    (args.out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    changed = ", ".join(f"{k}={v}" for k, v in STAGE_CONFIG[args.stage].items() if k not in ("auto_map", "architectures"))
    print(f"{args.stage}: {args.out} ({changed})")


if __name__ == "__main__":
    main()
