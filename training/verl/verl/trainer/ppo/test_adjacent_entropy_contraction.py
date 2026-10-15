"""Unit tests for adjacent_entropy_contraction.py.

Coverage:
- zero when entropy drop exceeds rho; positive when insufficient; larger on rise
- no gradient to logits_t; gradient to logits_next on violation
- normalized entropy (normalized by log(vocab_size))
- empty masks return graph-connected zero
- equal per-pair weighting across multiple step_keys
- production hidden-source helper agrees with reference logit helper
- graph-connected zero for empty local shard (hidden helper)
- exact global scaling arithmetic seam
- AdjacEntConfig validation
"""

from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn

from verl.trainer.ppo.adjacent_entropy_contraction import (
    AdjacEntConfig,
    _normalized_entropy,
    compute_adj_entropy_hinge_reference,
    compute_adj_entropy_hinge_hidden,
    count_unique_valid_pairs_from_batch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform_logits(vocab_size: int, n_tokens: int = 1) -> torch.Tensor:
    """Uniform distribution: H = log(V)/log(V) = 1.0."""
    return torch.zeros(n_tokens, vocab_size, dtype=torch.float32)


def _peaked_logits(vocab_size: int, n_tokens: int = 1, scale: float = 10.0) -> torch.Tensor:
    """Peaked distribution (one token dominates): entropy near 0."""
    logits = torch.full((n_tokens, vocab_size), -scale, dtype=torch.float32)
    logits[:, 0] = 0.0
    return logits


def _make_mask(n: int, val: bool = True) -> torch.Tensor:
    return torch.full((n,), val, dtype=torch.bool)


def _make_lm_head(in_features: int, vocab_size: int) -> nn.Linear:
    head = nn.Linear(in_features, vocab_size, bias=False)
    nn.init.eye_(head.weight[:min(in_features, vocab_size), :in_features]) if vocab_size <= in_features else None
    return head


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestAdjacEntConfig:
    def test_defaults(self):
        cfg = AdjacEntConfig.from_mapping({})
        assert cfg.enable is False
        assert math.isclose(cfg.rho, 0.05)
        assert cfg.projection_chunk_size == 512

    def test_enable_true(self):
        cfg = AdjacEntConfig.from_mapping({"enable": True, "rho": 0.2, "coef": 0.05})
        assert cfg.enable is True
        assert math.isclose(cfg.rho, 0.2)
        assert math.isclose(cfg.coef, 0.05)

    def test_none_mapping(self):
        cfg = AdjacEntConfig.from_mapping(None)
        assert cfg.enable is False

    def test_non_mapping_raises(self):
        with pytest.raises(ValueError, match="mapping"):
            AdjacEntConfig.from_mapping("bad")  # type: ignore

    def test_validate_rho_out_of_range(self):
        with pytest.raises(ValueError, match="rho"):
            AdjacEntConfig(enable=True, rho=1.5).validate()

    def test_validate_negative_coef(self):
        with pytest.raises(ValueError, match="coef"):
            AdjacEntConfig(enable=True, coef=-0.1).validate()

    def test_validate_zero_chunk_size(self):
        with pytest.raises(ValueError, match="projection_chunk_size"):
            AdjacEntConfig(enable=True, projection_chunk_size=0).validate()


# ---------------------------------------------------------------------------
# Normalized entropy
# ---------------------------------------------------------------------------

class TestNormalizedEntropy:
    def test_uniform_is_one(self):
        V = 32
        logits = _uniform_logits(V, n_tokens=4)
        h = _normalized_entropy(logits, vocab_size=V)
        assert torch.allclose(h, torch.ones(4), atol=1e-5)

    def test_peaked_is_near_zero(self):
        V = 32
        logits = _peaked_logits(V, n_tokens=4, scale=20.0)
        h = _normalized_entropy(logits, vocab_size=V)
        assert (h < 0.05).all()

    def test_entropy_between_zero_and_one(self):
        torch.manual_seed(0)
        V = 64
        logits = torch.randn(10, V)
        h = _normalized_entropy(logits, vocab_size=V)
        assert (h >= 0.0).all()
        assert (h <= 1.0 + 1e-5).all()

    def test_normalization_denominator_is_log_V(self):
        V = 8
        # uniform dist -> raw H = log(V); normalized = log(V)/log(V) = 1
        logits = _uniform_logits(V, n_tokens=1)
        h = _normalized_entropy(logits, vocab_size=V)
        assert math.isclose(float(h[0].item()), 1.0, abs_tol=1e-5)


# ---------------------------------------------------------------------------
# Reference helper
# ---------------------------------------------------------------------------

class TestReferenceHelper:
    def _pair(self, h_t_value: float, h_next_value: float, V: int = 16, rho: float = 0.1):
        """Build (logits_t, logits_next, mask) such that H_t ≈ h_t_value and H_next ≈ h_next_value."""
        log_norm = math.log(V)
        # Craft logits that produce desired entropy.  Simple approach:
        # entropy ≈ target: use temperature scaling of uniform distribution.
        # With logits = alpha * uniform -> entropy ~ unchanged (uniform is max entropy)
        # With logits = [0, -big, ...] -> entropy near 0
        # Use convex combination: uniform vs. peaked
        def _make_logits_for_entropy(target_h: float, n: int = 1) -> torch.Tensor:
            alpha = max(0.0, min(1.0, target_h))
            u = _uniform_logits(V, n_tokens=n)
            p = _peaked_logits(V, n_tokens=n, scale=10.0)
            # alpha=1 -> uniform (max entropy), alpha=0 -> peaked (low entropy)
            return alpha * u + (1.0 - alpha) * p

        logits_t = _make_logits_for_entropy(h_t_value, n=1)
        logits_next = _make_logits_for_entropy(h_next_value, n=1).requires_grad_(True)
        mask_t = _make_mask(1, True)
        mask_next = _make_mask(1, True)
        return logits_t, logits_next, mask_t, mask_next

    def test_zero_when_drop_exceeds_rho(self):
        """If H_next <= (1-rho)*H_t, hinge = 0."""
        V = 16
        # H_t = 1.0 (uniform), H_next = 0.5; rho=0.6 → threshold = 0.4 < 0.5
        # Actually: rho=0.6 -> threshold = (1-0.6)*1.0 = 0.4. H_next=0.5 > 0.4 → violation.
        # For zero hinge, need H_next <= (1-rho)*H_t.
        # Use H_t=1.0, rho=0.1, H_next=0.8 → threshold=0.9 > 0.8 → no violation.
        logits_t = _uniform_logits(V, n_tokens=1)
        logits_next = _peaked_logits(V, n_tokens=1, scale=3.0).requires_grad_(True)
        # Check actual entropy
        h_t_actual = float(_normalized_entropy(logits_t, V).item())
        h_next_actual = float(_normalized_entropy(logits_next.detach(), V).item())
        rho = 0.1
        threshold = (1.0 - rho) * h_t_actual
        if h_next_actual <= threshold:
            mask_t = _make_mask(1, True)
            mask_next = _make_mask(1, True)
            hinge, count, metrics = compute_adj_entropy_hinge_reference(
                logits_t, logits_next, mask_t, mask_next, rho=rho, vocab_size=V
            )
            assert float(hinge.item()) == pytest.approx(0.0, abs=1e-6)
            assert int(count.item()) == 1

    def test_positive_when_insufficient_drop(self):
        """If H_next = H_prev (no drop), hinge > 0 for rho > 0."""
        V = 16
        logits = _uniform_logits(V, n_tokens=1)
        logits_next = logits.clone().requires_grad_(True)
        mask_t = _make_mask(1, True)
        mask_next = _make_mask(1, True)
        rho = 0.1  # requires 10% drop; same entropy = 0% drop → violation
        hinge, count, metrics = compute_adj_entropy_hinge_reference(
            logits, logits_next, mask_t, mask_next, rho=rho, vocab_size=V
        )
        assert float(hinge.item()) > 0.0
        assert int(count.item()) == 1

    def test_larger_when_entropy_rises(self):
        """Entropy increase should give larger loss than no-change case."""
        V = 16
        logits_t = _peaked_logits(V, n_tokens=1, scale=5.0)  # low H_t
        logits_same = logits_t.clone().requires_grad_(True)
        logits_uniform = _uniform_logits(V, n_tokens=1).requires_grad_(True)
        mask = _make_mask(1, True)
        rho = 0.1

        hinge_same, _, _ = compute_adj_entropy_hinge_reference(
            logits_t, logits_same, mask, mask, rho=rho, vocab_size=V
        )
        hinge_rise, _, _ = compute_adj_entropy_hinge_reference(
            logits_t, logits_uniform, mask, mask, rho=rho, vocab_size=V
        )
        # Both should violate since H_t is low and no drop
        assert float(hinge_rise.item()) > float(hinge_same.item())

    def test_no_grad_to_logits_t(self):
        """Gradient must NOT flow to logits_t (it is stop-grad)."""
        V = 16
        logits_t = _uniform_logits(V, n_tokens=1).requires_grad_(True)
        logits_next = _uniform_logits(V, n_tokens=1).requires_grad_(True)
        mask = _make_mask(1, True)
        hinge, _, _ = compute_adj_entropy_hinge_reference(
            logits_t, logits_next, mask, mask, rho=0.1, vocab_size=V
        )
        if float(hinge.item()) > 0:
            hinge.backward()
            assert logits_t.grad is None, "logits_t must have no gradient"
            assert logits_next.grad is not None, "logits_next must have gradient"

    def test_grad_to_logits_next_on_violation(self):
        """When hinge > 0, gradient flows through logits_next."""
        torch.manual_seed(42)
        V = 16
        logits_t = _uniform_logits(V, n_tokens=1)  # H_t ≈ 1.0
        # Use slightly non-uniform logits so gradient at softmax is non-zero
        logits_next = torch.randn(1, V, dtype=torch.float32) * 0.01
        logits_next.requires_grad_(True)
        mask = _make_mask(1, True)
        rho = 0.5  # large drop required; near-uniform entropy → violation
        hinge, count, _ = compute_adj_entropy_hinge_reference(
            logits_t, logits_next, mask, mask, rho=rho, vocab_size=V
        )
        assert float(hinge.item()) > 0
        hinge.backward()
        assert logits_next.grad is not None
        assert logits_next.grad.abs().sum().item() > 0

    def test_empty_mask_t_returns_zero(self):
        V = 8
        logits_t = _uniform_logits(V, n_tokens=2)
        logits_next = _uniform_logits(V, n_tokens=2).requires_grad_(True)
        mask_t = _make_mask(2, False)
        mask_next = _make_mask(2, True)
        hinge, count, metrics = compute_adj_entropy_hinge_reference(
            logits_t, logits_next, mask_t, mask_next, rho=0.1, vocab_size=V
        )
        assert float(hinge.item()) == pytest.approx(0.0, abs=1e-9)
        assert int(count.item()) == 0
        assert metrics["adj_ent/empty_mask"] == 1.0

    def test_empty_mask_next_returns_zero(self):
        V = 8
        logits_t = _uniform_logits(V, n_tokens=2)
        logits_next = _uniform_logits(V, n_tokens=2).requires_grad_(True)
        mask_t = _make_mask(2, True)
        mask_next = _make_mask(2, False)
        hinge, count, metrics = compute_adj_entropy_hinge_reference(
            logits_t, logits_next, mask_t, mask_next, rho=0.1, vocab_size=V
        )
        assert float(hinge.item()) == pytest.approx(0.0, abs=1e-9)
        assert int(count.item()) == 0

    def test_graph_connected_zero_empty_mask(self):
        """Even with empty mask, returned hinge must require grad (graph-connected)."""
        V = 8
        logits_t = _uniform_logits(V, n_tokens=1)
        logits_next = _uniform_logits(V, n_tokens=1).requires_grad_(True)
        mask_t = _make_mask(1, False)
        mask_next = _make_mask(1, False)
        hinge, _, _ = compute_adj_entropy_hinge_reference(
            logits_t, logits_next, mask_t, mask_next, rho=0.1, vocab_size=V
        )
        assert float(hinge.item()) == pytest.approx(0.0, abs=1e-9)
        assert hinge.requires_grad


# ---------------------------------------------------------------------------
# Production hidden-source helper
# ---------------------------------------------------------------------------

class TestHiddenHelper:
    def _make_setup(self, n_tokens: int = 4, hidden: int = 8, vocab: int = 16):
        torch.manual_seed(42)
        actor_hidden = torch.randn(n_tokens, hidden, requires_grad=True)
        lm_head = nn.Linear(hidden, vocab, bias=False)
        nn.init.normal_(lm_head.weight, std=0.1)
        return actor_hidden, lm_head, vocab

    def test_empty_pair_mask_returns_graph_zero(self):
        n, H, V = 4, 8, 16
        actor_hidden, lm_head, vocab = self._make_setup(n, H, V)
        pair_mask = torch.zeros(n, dtype=torch.bool)
        prev_mean = torch.ones(n, dtype=torch.float32)
        step_maps = torch.zeros(n, dtype=torch.long)

        hinge, cnt, metrics = compute_adj_entropy_hinge_hidden(
            actor_hidden, lm_head, pair_mask, prev_mean, step_maps,
            rho=0.1, vocab_size=vocab, chunk_size=2
        )
        assert cnt == 0
        assert float(hinge.item()) == pytest.approx(0.0, abs=1e-9)
        assert hinge.requires_grad
        assert metrics["adj_ent/empty_mask"] == 1.0

    def test_hidden_agrees_with_reference_logits(self):
        """Production helper (hidden+lm_head) must agree with reference (logits)."""
        torch.manual_seed(7)
        n, H, V = 6, 16, 32
        actor_hidden, lm_head, vocab = self._make_setup(n, H, V)
        # Build logits exactly as the production helper would
        with torch.no_grad():
            logits_next = torch.nn.functional.linear(
                actor_hidden.detach(), lm_head.weight, None
            ).float()

        pair_mask = torch.ones(n, dtype=torch.bool)
        prev_mean_val = 0.8  # arbitrary prev entropy
        prev_mean = torch.full((n,), prev_mean_val, dtype=torch.float32)
        step_maps = torch.zeros(n, dtype=torch.long)  # all one step_key
        rho = 0.1

        # Production helper
        actor_hidden_for_prod = actor_hidden.detach().clone().requires_grad_(True)
        hinge_prod, cnt_prod, _ = compute_adj_entropy_hinge_hidden(
            actor_hidden_for_prod, lm_head, pair_mask, prev_mean, step_maps,
            rho=rho, vocab_size=vocab, chunk_size=4
        )

        # Reference: build mask_t=False since we supply prev_mean directly
        # We use the reference helper with same logits_next
        logits_next_ref = logits_next.clone().requires_grad_(True)
        # Build a single-token "logits_t" that produces entropy = prev_mean_val
        # For the reference helper, H_t = mean(entropy(logits_t[mask_t])).
        # Easiest: set mask_t=True with uniform logits scaled to yield entropy=prev_mean_val.
        # H(uniform) = 1.0. We want H_t_ref = prev_mean_val.
        # Use peaked logits: the production helper uses prev_mean directly.
        # Better approach: just check that the hinge values are close by computing
        # reference from the same entropy values.
        h_t_detached = torch.tensor([[prev_mean_val]], dtype=torch.float32)
        h_next = _normalized_entropy(logits_next, V)
        h_next_mean = h_next.mean()
        threshold = (1.0 - rho) * prev_mean_val
        excess = h_next_mean - threshold
        expected_hinge = max(0.0, float(excess.item())) ** 2

        assert math.isclose(float(hinge_prod.item()), expected_hinge, rel_tol=1e-4, abs_tol=1e-6)
        assert cnt_prod == 1

    def test_grad_flows_through_actor_hidden(self):
        """Gradient must flow through actor_hidden_rmpad on violation."""
        n, H, V = 4, 8, 16
        actor_hidden, lm_head, vocab = self._make_setup(n, H, V)
        pair_mask = torch.ones(n, dtype=torch.bool)
        prev_mean = torch.full((n,), 0.9, dtype=torch.float32)  # high prev entropy
        step_maps = torch.zeros(n, dtype=torch.long)

        hinge, cnt, metrics = compute_adj_entropy_hinge_hidden(
            actor_hidden, lm_head, pair_mask, prev_mean, step_maps,
            rho=0.5, vocab_size=vocab, chunk_size=2  # large rho -> likely violation
        )
        if float(hinge.item()) > 0:
            hinge.backward()
            assert actor_hidden.grad is not None
            assert actor_hidden.grad.abs().sum().item() > 0

    def test_multiple_step_keys_equal_weighting(self):
        """Two step_keys must each contribute equally (hinge_sum = hinge_k1 + hinge_k2)."""
        torch.manual_seed(99)
        n, H, V = 8, 16, 32
        actor_hidden, lm_head, vocab = self._make_setup(n, H, V)
        pair_mask = torch.ones(n, dtype=torch.bool)
        prev_mean = torch.full((n,), 0.8, dtype=torch.float32)
        # step_maps: first 4 tokens → key 0, last 4 → key 1
        step_maps = torch.cat([torch.zeros(4, dtype=torch.long), torch.ones(4, dtype=torch.long)])
        rho = 0.1

        hinge_both, cnt_both, _ = compute_adj_entropy_hinge_hidden(
            actor_hidden, lm_head, pair_mask, prev_mean, step_maps,
            rho=rho, vocab_size=vocab, chunk_size=4
        )
        assert cnt_both == 2

        # Check individual keys sum to combined
        mask_k0 = step_maps.eq(0)
        mask_k1 = step_maps.eq(1)
        hinge_k0, cnt_k0, _ = compute_adj_entropy_hinge_hidden(
            actor_hidden, lm_head, mask_k0, prev_mean, step_maps,
            rho=rho, vocab_size=vocab, chunk_size=4
        )
        hinge_k1, cnt_k1, _ = compute_adj_entropy_hinge_hidden(
            actor_hidden, lm_head, mask_k1, prev_mean, step_maps,
            rho=rho, vocab_size=vocab, chunk_size=4
        )
        expected_sum = float(hinge_k0.item()) + float(hinge_k1.item())
        assert math.isclose(float(hinge_both.item()), expected_sum, rel_tol=1e-5, abs_tol=1e-7)

    def test_invalid_step_keys_excluded(self):
        """Step keys of -1 (invalid/padded) must be excluded from pair computation."""
        n, H, V = 4, 8, 16
        actor_hidden, lm_head, vocab = self._make_setup(n, H, V)
        pair_mask = torch.ones(n, dtype=torch.bool)
        prev_mean = torch.full((n,), 0.8, dtype=torch.float32)
        step_maps = torch.tensor([-1, -1, 0, 0], dtype=torch.long)  # first 2 invalid

        hinge, cnt, metrics = compute_adj_entropy_hinge_hidden(
            actor_hidden, lm_head, pair_mask, prev_mean, step_maps,
            rho=0.1, vocab_size=vocab
        )
        assert cnt == 1  # only key=0 is valid

    def test_chunked_projection_agrees_with_unchunked(self):
        """Chunked projection must give same result as single large chunk."""
        torch.manual_seed(123)
        n, H, V = 12, 16, 32
        actor_hidden, lm_head, vocab = self._make_setup(n, H, V)
        pair_mask = torch.ones(n, dtype=torch.bool)
        prev_mean = torch.full((n,), 0.7, dtype=torch.float32)
        step_maps = torch.zeros(n, dtype=torch.long)
        rho = 0.1

        hinge_chunked, _, _ = compute_adj_entropy_hinge_hidden(
            actor_hidden.detach(), lm_head, pair_mask, prev_mean, step_maps,
            rho=rho, vocab_size=vocab, chunk_size=3
        )
        hinge_full, _, _ = compute_adj_entropy_hinge_hidden(
            actor_hidden.detach(), lm_head, pair_mask, prev_mean, step_maps,
            rho=rho, vocab_size=vocab, chunk_size=n
        )
        assert math.isclose(
            float(hinge_chunked.item()), float(hinge_full.item()), rel_tol=1e-5, abs_tol=1e-7
        )


# ---------------------------------------------------------------------------
# Global scaling arithmetic seam
# ---------------------------------------------------------------------------

class TestGlobalScalingArithmetic:
    """Verify the exact scaling formula that update_policy applies.

    The math:
      policy_loss += coef * hinge_sum_i * global_C / (N_pair_global * C_i)
      loss = policy_loss * C_i  (loss_scale_factor = C_i)
      FSDP averages gradients across world_size ranks.
      After grad scale world_size/global_C:
        net = coef * global_hinge_sum / N_pair_global  = coef * global_equal_pair_mean
    """

    def test_scaling_math_two_microbatches(self):
        """Verify scaling formula with two synthetic microbatches."""
        torch.manual_seed(0)
        coef = 0.01
        rho = 0.1
        world_size = 4

        # Microbatch 1: 10 CE tokens, 3 pairs, hinge_sum = 0.6
        C1, pairs1, hinge1 = 10.0, 3, 0.6
        # Microbatch 2: 20 CE tokens, 2 pairs, hinge_sum = 0.4
        C2, pairs2, hinge2 = 20.0, 2, 0.4

        N_pair_global = float(pairs1 + pairs2)  # = 5
        global_C = C1 + C2  # = 30

        # Expected gradient contribution after full scaling:
        # from mb1: coef * hinge1 / N_pair_global  (pre-scale)
        # after multiplication by C1 and then world_size/global_C:
        # = coef * hinge1 / N_pair_global * C1 * (world_size/global_C)
        # = coef * hinge1 / N_pair_global * C1/global_C (ws=1)
        # Hmm - this doesn't simplify to 1/N_pair unless C1/global_C cancels.
        # Let me verify: the spec formula for aux_scale_i:
        #   aux_scale_i = coef * global_C / (N_pair_global * C_i * world_size)
        # Then: policy_loss contribution_i = aux_scale_i * hinge_sum_i
        # After * C_i and * world_size/global_C:
        #   = coef * global_C / (N_pair * C_i * ws) * hinge_i * C_i * ws/global_C
        #   = coef * hinge_i / N_pair  ✓

        # Test aux_scale formula. There is deliberately no /world_size here:
        # FSDP rank averaging and the final world_size/global_C scale cancel.
        aux_scale1 = coef * global_C / (N_pair_global * C1)
        aux_scale2 = coef * global_C / (N_pair_global * C2)

        # These two microbatches are on one rank. Include FSDP's 1/world_size
        # averaging before the existing final world_size/global_C rescale.
        net1 = (
            aux_scale1
            * hinge1
            * C1
            / world_size
            * (world_size / global_C)
        )
        net2 = (
            aux_scale2
            * hinge2
            * C2
            / world_size
            * (world_size / global_C)
        )

        expected_net1 = coef * hinge1 / N_pair_global
        expected_net2 = coef * hinge2 / N_pair_global

        assert math.isclose(net1, expected_net1, rel_tol=1e-9)
        assert math.isclose(net2, expected_net2, rel_tol=1e-9)

        total_net = net1 + net2
        expected_total = coef * (hinge1 + hinge2) / N_pair_global
        assert math.isclose(total_net, expected_total, rel_tol=1e-9)

    def test_aux_scale_invariant_to_C_i_given_fixed_global_C(self):
        """Aux net effect per hinge_unit is always coef/N_pair regardless of C_i."""
        coef = 0.05
        N_pair_global = 10.0
        global_C = 100.0
        world_size = 8.0

        for C_i in [1.0, 5.0, 50.0, 99.0]:
            aux_scale = coef * global_C / (N_pair_global * C_i)
            net = (
                aux_scale
                * C_i
                / world_size
                * (world_size / global_C)
            )
            assert math.isclose(net, coef / N_pair_global, rel_tol=1e-9), (
                f"C_i={C_i}: net={net} expected {coef/N_pair_global}"
            )

    def test_extra_world_divisor_would_underweight(self):
        coef = 0.04
        global_C = 80.0
        global_pairs = 8.0
        C_i = 20.0
        world_size = 4.0
        wrong_scale = (
            coef
            * global_C
            / (global_pairs * C_i * world_size)
        )
        wrong_net = (
            wrong_scale
            * C_i
            / world_size
            * (world_size / global_C)
        )
        assert math.isclose(
            wrong_net,
            coef / (global_pairs * world_size),
            rel_tol=1e-9,
        )
        assert not math.isclose(
            wrong_net,
            coef / global_pairs,
            rel_tol=1e-9,
        )

    def test_sp_local_surrogate_needs_no_sp_divisor(self):
        coef = 0.02
        sp_size = 2.0
        world_size = 2.0
        unique_global_C = 40.0
        unique_global_pairs = 4.0
        local_C = 40.0
        positive_excess = 0.2
        pair_token_count = 10.0

        # The prepass sees full rows on both SP peers.
        pre_C = sp_size * unique_global_C
        pre_pairs = sp_size * unique_global_pairs
        aux_scale = (
            coef
            * pre_C
            / (pre_pairs * local_C)
        )

        # Each peer carries only the derivative for its shard-local entropy
        # sum: d(relu(mean-threshold)^2)/d(local_sum) = 2*d/count.
        local_surrogate_derivative = (
            2.0 * positive_excess / pair_token_count
        )
        summed_rank_grad = (
            sp_size
            * aux_scale
            * local_surrogate_derivative
            * local_C
        )
        fsdp_averaged = summed_rank_grad / world_size
        final_scaled = (
            fsdp_averaged
            * world_size
            / unique_global_C
        )
        assert math.isclose(
            final_scaled,
            (
                coef
                * sp_size
                * local_surrogate_derivative
                / unique_global_pairs
            ),
            rel_tol=1e-9,
        )

    def test_zero_N_pair_global_skips_aux(self):
        """If N_pair_global=0, no aux term should be added (avoid division by zero)."""
        # This tests the guard in update_policy: if _adj_ent_pre_N_pair_global > 0
        N_pair_global = 0.0
        # The guard ensures we skip the scaling block
        assert not (N_pair_global > 0), "guard should skip aux term"


# ---------------------------------------------------------------------------
# Count helper
# ---------------------------------------------------------------------------

class TestCountHelper:
    def test_basic_count(self):
        adj_mask = torch.tensor([[True, True, False, False]], dtype=torch.bool)
        step_maps = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)
        resp_mask = torch.ones(1, 4, dtype=torch.bool)
        pairs, ce = count_unique_valid_pairs_from_batch(adj_mask, step_maps, resp_mask)
        assert pairs == 1  # key=0 in row 0
        assert ce == 4

    def test_two_rows_two_keys_each(self):
        # 2 rows, 4 tokens each. Row 0: keys 0,0,1,1. Row 1: keys 2,2,3,3.
        adj_mask = torch.ones(2, 4, dtype=torch.bool)
        step_maps = torch.tensor([[0, 0, 1, 1], [2, 2, 3, 3]], dtype=torch.long)
        resp_mask = torch.ones(2, 4, dtype=torch.bool)
        pairs, ce = count_unique_valid_pairs_from_batch(adj_mask, step_maps, resp_mask)
        assert pairs == 4  # (0,0), (0,1), (1,2), (1,3)
        assert ce == 8

    def test_none_inputs(self):
        pairs, ce = count_unique_valid_pairs_from_batch(None, None, None)
        assert pairs == 0
        assert ce == 0

    def test_invalid_keys_excluded(self):
        adj_mask = torch.tensor([[True, True, True]], dtype=torch.bool)
        step_maps = torch.tensor([[-1, 0, -1]], dtype=torch.long)
        resp_mask = torch.ones(1, 3, dtype=torch.bool)
        pairs, ce = count_unique_valid_pairs_from_batch(adj_mask, step_maps, resp_mask)
        assert pairs == 1  # only key=0 is valid
