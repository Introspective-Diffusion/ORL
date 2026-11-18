from pathlib import Path
import sys
import unittest

import torch
from transformers import PreTrainedModel
from transformers.cache_utils import DynamicCache
from transformers import modeling_rope_utils as mru


OVERLAYS_PARENT = Path(__file__).resolve().parents[2]
if str(OVERLAYS_PARENT) not in sys.path:
    sys.path.insert(0, str(OVERLAYS_PARENT))

from sdar.configuration_sdar import SDARConfig  # noqa: E402
from sdar.modeling_sdar import SDARForCausalLM  # noqa: E402


def _build_test_model(use_regular_causal: bool = True) -> SDARForCausalLM:
    # Allow this test suite to run standalone on transformers 5.x even before
    # the companion RoPE/tied-weights PR is merged.
    if "default" not in mru.ROPE_INIT_FUNCTIONS:
        def _default_rope(config, device=None, seq_len=None, **kwargs):
            del seq_len, kwargs
            base = float(getattr(config, "rope_theta", 10000.0))
            dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
            return inv_freq, 1.0

        mru.ROPE_INIT_FUNCTIONS["default"] = _default_rope

    cfg = SDARConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        block_size=4,
        mask_token_id=255,
        use_cache=False,
        use_regular_causal=use_regular_causal,
        noisy_block_bidirectional=True,
    )
    if not hasattr(cfg, "pad_token_id"):
        cfg.pad_token_id = None
    if hasattr(PreTrainedModel, "get_expanded_tied_weights_keys") and isinstance(
        SDARForCausalLM._tied_weights_keys, list
    ):
        SDARForCausalLM._tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    torch.manual_seed(42)
    return SDARForCausalLM(cfg).eval()


class SDAREvalAttentionMaskTests(unittest.TestCase):
    def test_batched_2d_padding_mask_non_square_does_not_crash_and_matches_unpadded(self):
        """Regression test: 2D `(batch_size, seq_len)` with `batch_size != seq_len` previously crashed."""
        model = _build_test_model(use_regular_causal=True)

        # Batch of 2, seq_len = 5: sample 0 is left-padded by 2 tokens; sample 1 is unpadded.
        input_ids = torch.tensor(
            [
                [0, 0, 11, 22, 33],
                [44, 55, 66, 77, 88],
            ],
            dtype=torch.long,
        )
        attention_mask = torch.tensor(
            [
                [0, 0, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ],
            dtype=torch.long,
        )
        position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)

        with torch.no_grad():
            batched_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            ).logits
            ref_0 = model(
                input_ids=torch.tensor([[11, 22, 33]], dtype=torch.long),
                attention_mask=torch.ones((1, 3), dtype=torch.long),
            ).logits
            ref_1 = model(
                input_ids=torch.tensor([[44, 55, 66, 77, 88]], dtype=torch.long),
                attention_mask=torch.ones((1, 5), dtype=torch.long),
            ).logits

        torch.testing.assert_close(batched_logits[0, 2:], ref_0[0], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(batched_logits[1], ref_1[0], rtol=1e-5, atol=1e-5)

    def test_batched_2d_padding_mask_square_batch_eq_seq_does_not_swap_axes(self):
        """Regression test: when `batch_size == seq_len`, `(B, L)` was silently broadcast as `(1, 1, q_len, kv_len)`."""
        model = _build_test_model(use_regular_causal=True)

        # B = 4, L = 4 with distinct per-sample left-padding lengths.
        input_ids = torch.tensor(
            [
                [99, 99, 99, 10],
                [88, 88, 20, 21],
                [77, 30, 31, 32],
                [40, 41, 42, 43],
            ],
            dtype=torch.long,
        )
        attention_mask = torch.tensor(
            [
                [0, 0, 0, 1],
                [0, 0, 1, 1],
                [0, 1, 1, 1],
                [1, 1, 1, 1],
            ],
            dtype=torch.long,
        )
        position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)

        # Mutate masked pad token IDs: valid token logits must be completely invariant to pad-token identities.
        input_ids_perturbed_pads = input_ids.clone()
        input_ids_perturbed_pads[0, :3] = torch.tensor([1, 2, 3])
        input_ids_perturbed_pads[1, :2] = torch.tensor([4, 5])
        input_ids_perturbed_pads[2, :1] = torch.tensor([6])

        with torch.no_grad():
            logits_a = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            ).logits
            logits_b = model(
                input_ids=input_ids_perturbed_pads,
                attention_mask=attention_mask,
                position_ids=position_ids,
            ).logits

        for b_idx, start in enumerate([3, 2, 1, 0]):
            torch.testing.assert_close(
                logits_a[b_idx, start:],
                logits_b[b_idx, start:],
                rtol=1e-5,
                atol=1e-5,
            )

    def test_eval_causal_masking_prevents_future_token_leakage(self):
        """In regular causal eval forwards (`store_kv` not False), future tokens must not leak into earlier positions."""
        model = _build_test_model(use_regular_causal=True)

        seq_a = torch.tensor([[12, 24, 36, 48, 60]], dtype=torch.long)
        seq_b = torch.tensor([[12, 24, 36, 99, 101]], dtype=torch.long)

        with torch.no_grad():
            out_none_a = model(input_ids=seq_a).logits
            out_none_b = model(input_ids=seq_b).logits
            out_ones_a = model(input_ids=seq_a, attention_mask=torch.ones_like(seq_a)).logits

        torch.testing.assert_close(out_none_a, out_ones_a, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(out_none_a[:, :3], out_none_b[:, :3], rtol=1e-5, atol=1e-5)
        self.assertGreater((out_none_a[:, 3:] - out_none_b[:, 3:]).abs().max().item(), 1e-4)

    def test_block_diffusion_4d_bool_mask_and_store_kv_false_remain_bidirectional(self):
        """Custom SDAR 4D bool masks and `store_kv=False` noisy-block steps preserve bidirectional block attention."""
        model = _build_test_model(use_regular_causal=True)

        # 1) Explicit 4D bool staircase mask (2 causal prefix tokens + 4 bidirectional block tokens)
        mask_4d = torch.zeros((1, 1, 6, 6), dtype=torch.bool)
        mask_4d[:, :, :2, :2] = torch.tril(torch.ones((2, 2), dtype=torch.bool))
        mask_4d[:, :, 2:, :] = True

        ids_1 = torch.tensor([[5, 6, 10, 11, 12, 13]], dtype=torch.long)
        ids_2 = torch.tensor([[5, 6, 10, 11, 12, 99]], dtype=torch.long)

        with torch.no_grad():
            h1 = model.model(input_ids=ids_1, attention_mask=mask_4d).last_hidden_state
            h2 = model.model(input_ids=ids_2, attention_mask=mask_4d).last_hidden_state

        # Prefix stays causal; noisy block position 2 sees change at position 5 bidirectionally.
        torch.testing.assert_close(h1[:, :2], h2[:, :2], rtol=1e-5, atol=1e-5)
        self.assertGreater((h1[:, 2] - h2[:, 2]).abs().max().item(), 1e-4)

        # 2) Batched prefix prefill (`store_kv=True`) + active noisy block (`store_kv=False`) with 2D padding mask
        prefix_ids = torch.tensor([[0, 5, 6], [7, 8, 9]], dtype=torch.long)
        prefix_mask = torch.tensor([[0, 1, 1], [1, 1, 1]], dtype=torch.long)
        prefix_pos = (prefix_mask.cumsum(dim=-1) - 1).clamp(min=0)

        with torch.no_grad():
            prefill = model.model(
                input_ids=prefix_ids,
                attention_mask=prefix_mask,
                position_ids=prefix_pos,
                past_key_values=DynamicCache(),
                use_cache=True,
                store_kv=True,
            )
            cache = prefill.past_key_values
            noisy_a = torch.tensor([[20, 21, 22, 23], [30, 31, 32, 33]], dtype=torch.long)
            noisy_b = torch.tensor([[20, 21, 22, 99], [30, 31, 32, 33]], dtype=torch.long)
            full_kv_mask = torch.cat([prefix_mask, torch.ones((2, 4), dtype=torch.long)], dim=-1)
            block_pos = prefix_mask.sum(dim=-1, keepdim=True) + torch.arange(4)

            denoise_a = model.model(
                input_ids=noisy_a,
                attention_mask=full_kv_mask,
                position_ids=block_pos,
                past_key_values=cache,
                use_cache=True,
                store_kv=False,
            ).last_hidden_state
            denoise_b = model.model(
                input_ids=noisy_b,
                attention_mask=full_kv_mask,
                position_ids=block_pos,
                past_key_values=cache,
                use_cache=True,
                store_kv=False,
            ).last_hidden_state

        # Sample 0 noisy token 0 attends bidirectionally to noisy token 3; sample 1 is unaffected.
        self.assertGreater((denoise_a[0, 0] - denoise_b[0, 0]).abs().max().item(), 1e-4)
        torch.testing.assert_close(denoise_a[1], denoise_b[1], rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
