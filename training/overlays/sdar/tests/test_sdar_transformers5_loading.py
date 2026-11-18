import json
from pathlib import Path
import shutil
import tempfile
import unittest

import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel


OVERLAY_DIR = Path(__file__).resolve().parents[1]


def _write_minified_sdar_checkpoint(dst: Path, *, include_stale_inv_freq: bool = False) -> dict:
    for name in ("configuration_sdar.py", "modeling_sdar.py", "fused_linear_diffusion_cross_entropy.py"):
        shutil.copy2(OVERLAY_DIR / name, dst / name)

    # Mirrors yifanyu/QwenDLM-4B config.json structure (notably omitting `pad_token_id`
    # and enabling `tie_word_embeddings=True` with `rope_theta=5000000`).
    cfg_dict = {
        "architectures": ["SDARForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_sdar.SDARConfig",
            "AutoModel": "modeling_sdar.SDARForCausalLM",
            "AutoModelForCausalLM": "modeling_sdar.SDARForCausalLM",
        },
        "attention_bias": False,
        "attention_dropout": 0.0,
        "block_size": 8,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "fuse_cross_entropy": True,
        "head_dim": 16,
        "hidden_act": "silu",
        "hidden_size": 64,
        "initializer_range": 0.02,
        "intermediate_size": 128,
        "loss_on_all_noisy_tokens": True,
        "mask_token_id": 255,
        "masking_strategy": "random",
        "max_position_embeddings": 1024,
        "max_window_layers": 2,
        "model_type": "sdar",
        "noise_strategy": "random",
        "noisy_block_bidirectional": True,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "random_token_high": 255,
        "rms_norm_eps": 1e-6,
        "rope_scaling": None,
        "rope_theta": 5000000.0,
        "scale_loss_by_p_mask": False,
        "self_conditioning_chunk_size": 16,
        "self_conditioning_init": "zero",
        "self_conditioning_init_std": 0.001,
        "self_conditioning_post_norm": True,
        "self_conditioning_post_norm_scale": 0.02,
        "self_conditioning_prob": 1.0,
        "sliding_window": None,
        "tie_word_embeddings": True,
        "torch_dtype": "float32",
        "training_attention_backend": "flex_attention",
        "use_cache": False,
        "use_regular_causal": True,
        "use_sliding_window": False,
        "vocab_size": 256,
    }
    (dst / "config.json").write_text(json.dumps(cfg_dict, indent=2), encoding="utf-8")

    torch.manual_seed(0)
    cfg = AutoConfig.from_pretrained(str(dst), trust_remote_code=True)
    ref_model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)

    # Save weights without tied `lm_head.weight` and without non-persistent `inv_freq`.
    state_dict = {
        k: v.detach().clone().contiguous()
        for k, v in ref_model.state_dict().items()
        if k != "lm_head.weight"
    }
    if include_stale_inv_freq:
        state_dict["model.rotary_emb.inv_freq"] = torch.zeros(cfg.head_dim // 2, dtype=torch.float32)
    save_file(state_dict, str(dst / "model.safetensors"), metadata={"format": "pt"})
    return {
        "cfg_dict": cfg_dict,
        "expected_inv_freq": ref_model.model.rotary_emb.inv_freq.detach().clone(),
        "expected_embed": ref_model.model.embed_tokens.weight.detach().clone(),
    }


class SDARTransformers5LoadingTests(unittest.TestCase):
    def test_from_pretrained_initializes_rope_inv_freq_and_tied_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ref = _write_minified_sdar_checkpoint(tmp_path)

            cfg = AutoConfig.from_pretrained(str(tmp_path), trust_remote_code=True)
            self.assertIsNone(cfg.pad_token_id)

            # Dirty allocator memory before meta-device materialization in `from_pretrained`.
            _ = torch.full((4096,), 987654.0)
            loaded = AutoModelForCausalLM.from_pretrained(str(tmp_path), trust_remote_code=True).eval()

            expected_inv_freq = 1.0 / (
                5000000.0 ** (torch.arange(0, 16, 2, dtype=torch.int64).float() / 16.0)
            )
            inv_freq = loaded.model.rotary_emb.inv_freq
            orig_inv_freq = loaded.model.rotary_emb.original_inv_freq

            self.assertTrue(torch.isfinite(inv_freq).all().item())
            self.assertEqual(inv_freq.shape, expected_inv_freq.shape)
            torch.testing.assert_close(inv_freq, expected_inv_freq, rtol=0.0, atol=0.0)
            torch.testing.assert_close(orig_inv_freq, expected_inv_freq, rtol=0.0, atol=0.0)
            torch.testing.assert_close(inv_freq, ref["expected_inv_freq"], rtol=0.0, atol=0.0)

            # Tied lm_head must share storage with embed_tokens and match saved weights.
            self.assertIs(loaded.lm_head.weight, loaded.model.embed_tokens.weight)
            torch.testing.assert_close(loaded.lm_head.weight, ref["expected_embed"], rtol=0.0, atol=0.0)

            if hasattr(PreTrainedModel, "get_expanded_tied_weights_keys"):
                self.assertEqual(
                    loaded._tied_weights_keys,
                    {"lm_head.weight": "model.embed_tokens.weight"},
                )
            else:
                self.assertEqual(loaded._tied_weights_keys, ["lm_head.weight"])

    def test_rotary_embedding_is_position_sensitive_after_from_pretrained(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_minified_sdar_checkpoint(tmp_path, include_stale_inv_freq=True)

            loaded = AutoModelForCausalLM.from_pretrained(str(tmp_path), trust_remote_code=True).eval()
            # Identical token repeated across positions with full bidirectional visibility:
            # if `inv_freq` collapsed to ~0, Q/K rotations would be position-invariant;
            # with valid RoPE frequencies and distinct token values, changing `position_ids`
            # changes attention outputs.
            input_ids = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
            full_vis = torch.ones((1, 1, 4, 4), dtype=torch.bool)
            pos_a = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
            pos_b = torch.tensor([[0, 3, 2, 1]], dtype=torch.long)

            with torch.no_grad():
                out_a = loaded(input_ids=input_ids, attention_mask=full_vis, position_ids=pos_a).logits
                out_b = loaded(input_ids=input_ids, attention_mask=full_vis, position_ids=pos_b).logits

            max_diff = (out_a - out_b).abs().max().item()
            self.assertGreater(max_diff, 1e-4)


if __name__ == "__main__":
    unittest.main()
