"""Unit tests for causal_wrong_ce.py.

Coverage:
- Tri-state role routing from verification_status
- Final-only row filtering per (block_index)
- Loss computation: independent global denominators
- Positive-only / negative-only / mixed / zero-signal batches
- Non-finite CE rejection
- Config incompatibility validation (via _causal_wrong_ce_config)
- Ghost/underfill ownership: roles zero out on non-label tokens
- Causal source invariance: staircase_size=1 on a toy fixture
"""

from __future__ import annotations

import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# module under test
# ---------------------------------------------------------------------------
from verl.trainer.ppo.causal_wrong_ce import (
    TOKEN_ROLE_DROP,
    TOKEN_ROLE_POSITIVE,
    TOKEN_ROLE_WRONG_CAUSAL,
    CausalWrongCEConfig,
    build_causal_wrong_ce_response_roles,
    compute_causal_wrong_ce_loss,
    select_wrong_response_final_step_keys,
)


# ===========================================================================
# Tri-state role routing
# ===========================================================================

class TestBuildRoles:
    def test_correct_maps_positive(self):
        roles = build_causal_wrong_ce_response_roles(
            verification_statuses=["correct"]
        )
        assert roles == [TOKEN_ROLE_POSITIVE]

    def test_wrong_maps_wrong_causal(self):
        roles = build_causal_wrong_ce_response_roles(
            verification_statuses=["wrong"]
        )
        assert roles == [TOKEN_ROLE_WRONG_CAUSAL]

    def test_uncertain_maps_drop(self):
        roles = build_causal_wrong_ce_response_roles(
            verification_statuses=["uncertain"]
        )
        assert roles == [TOKEN_ROLE_DROP]

    def test_none_maps_drop(self):
        roles = build_causal_wrong_ce_response_roles(
            verification_statuses=[None]
        )
        assert roles == [TOKEN_ROLE_DROP]

    def test_unknown_maps_drop(self):
        roles = build_causal_wrong_ce_response_roles(
            verification_statuses=["maybe", "idk", ""]
        )
        assert roles == [TOKEN_ROLE_DROP, TOKEN_ROLE_DROP, TOKEN_ROLE_DROP]

    def test_case_insensitive(self):
        roles = build_causal_wrong_ce_response_roles(
            verification_statuses=["CORRECT", "WRONG", "UNCERTAIN"]
        )
        assert roles == [TOKEN_ROLE_POSITIVE, TOKEN_ROLE_WRONG_CAUSAL, TOKEN_ROLE_DROP]

    def test_mixed_batch(self):
        statuses = ["correct", "wrong", "uncertain", "correct", "wrong"]
        roles = build_causal_wrong_ce_response_roles(
            verification_statuses=statuses
        )
        expected = [
            TOKEN_ROLE_POSITIVE,
            TOKEN_ROLE_WRONG_CAUSAL,
            TOKEN_ROLE_DROP,
            TOKEN_ROLE_POSITIVE,
            TOKEN_ROLE_WRONG_CAUSAL,
        ]
        assert roles == expected

    def test_empty_batch(self):
        assert build_causal_wrong_ce_response_roles(verification_statuses=[]) == []


# ===========================================================================
# Final-only row filtering
# ===========================================================================

def _make_sc_b(steps: list[dict]) -> dict:
    """Build a rollout_self_conditioning[b] dict from step specs."""
    return {
        str(step["step_key"]): {
            "is_final_denoise_row": step.get("is_final", False),
            "block_idx": step.get("block_idx", None),
        }
        for step in steps
    }


class TestSelectFinalStepKeys:
    def test_single_block_single_final(self):
        sc_b = _make_sc_b([
            {"step_key": 0, "block_idx": 0, "is_final": False},
            {"step_key": 1, "block_idx": 0, "is_final": False},
            {"step_key": 2, "block_idx": 0, "is_final": True},
        ])
        keys = select_wrong_response_final_step_keys(
            rollout_self_conditioning=sc_b, block_size=4
        )
        assert keys == {2}

    def test_two_blocks_one_final_each(self):
        sc_b = _make_sc_b([
            {"step_key": 0, "block_idx": 0, "is_final": False},
            {"step_key": 3, "block_idx": 0, "is_final": True},
            {"step_key": 4, "block_idx": 1, "is_final": False},
            {"step_key": 7, "block_idx": 1, "is_final": True},
        ])
        keys = select_wrong_response_final_step_keys(
            rollout_self_conditioning=sc_b, block_size=4
        )
        assert keys == {3, 7}

    def test_duplicate_finals_fail_closed(self):
        sc_b = _make_sc_b([
            {"step_key": 2, "block_idx": 0, "is_final": True},
            {"step_key": 5, "block_idx": 0, "is_final": True},
        ])
        with pytest.raises(ValueError, match="exactly one final"):
            select_wrong_response_final_step_keys(
                rollout_self_conditioning=sc_b, block_size=4
            )

    def test_missing_final_fails_closed(self):
        sc_b = _make_sc_b([
            {"step_key": 0, "block_idx": 0, "is_final": False},
            {"step_key": 1, "block_idx": 0, "is_final": False},
        ])
        with pytest.raises(ValueError, match="exactly one final"):
            select_wrong_response_final_step_keys(
                rollout_self_conditioning=sc_b, block_size=4
            )

    def test_block_idx_inferred_from_step_key_when_absent(self):
        # block_size=4 -> step_key=3 -> block_idx=0, step_key=7 -> block_idx=1
        sc_b = {
            "3": {"is_final_denoise_row": True},   # block 0
            "7": {"is_final_denoise_row": True},   # block 1
            "0": {"is_final_denoise_row": False},
        }
        keys = select_wrong_response_final_step_keys(
            rollout_self_conditioning=sc_b, block_size=4
        )
        assert keys == {3, 7}

    def test_empty_sc_b_fails_closed(self):
        with pytest.raises(ValueError, match="non-empty replay"):
            select_wrong_response_final_step_keys(
                rollout_self_conditioning={}, block_size=4
            )

    def test_non_dict_sc_b_fails_closed(self):
        with pytest.raises(ValueError, match="non-empty replay"):
            select_wrong_response_final_step_keys(
                rollout_self_conditioning=None, block_size=4
            )

    def test_metadata_block_idx_field(self):
        # is_final_denoise_row in nested metadata dict
        sc_b = {
            "5": {
                "metadata": {
                    "is_final_denoise_row": True,
                    "block_idx": 2,
                }
            }
        }
        keys = select_wrong_response_final_step_keys(
            rollout_self_conditioning=sc_b, block_size=4
        )
        assert keys == {5}


# ===========================================================================
# Loss computation
# ===========================================================================

class TestComputeLoss:
    def test_zero_signal_returns_none(self):
        loss, metrics = compute_causal_wrong_ce_loss(
            positive_ce_sum=0.0,
            positive_token_count=0,
            wrong_causal_ce_sum=0.0,
            wrong_causal_token_count=0,
        )
        assert loss is None
        assert metrics["causal_wrong_ce/zero_signal_batch"] == 1.0

    def test_positive_only(self):
        loss, metrics = compute_causal_wrong_ce_loss(
            positive_ce_sum=3.0,
            positive_token_count=3,
            wrong_causal_ce_sum=0.0,
            wrong_causal_token_count=0,
        )
        assert loss is not None
        assert math.isclose(loss, 1.0)
        assert metrics["causal_wrong_ce/wrong_tokens"] == 0.0

    def test_negative_only(self):
        loss, metrics = compute_causal_wrong_ce_loss(
            positive_ce_sum=0.0,
            positive_token_count=0,
            wrong_causal_ce_sum=4.0,
            wrong_causal_token_count=2,
        )
        assert loss is not None
        assert math.isclose(loss, -2.0)
        assert metrics["causal_wrong_ce/positive_tokens"] == 0.0

    def test_mixed_independent_denominators(self):
        # pos_mean = 6/3 = 2.0, neg_mean = 8/4 = 2.0 -> loss = 0.0
        loss, _ = compute_causal_wrong_ce_loss(
            positive_ce_sum=6.0,
            positive_token_count=3,
            wrong_causal_ce_sum=8.0,
            wrong_causal_token_count=4,
        )
        assert loss is not None
        assert math.isclose(loss, 0.0)

    def test_mixed_positive_dominates(self):
        # pos_mean = 6/2 = 3.0, neg_mean = 1/2 = 0.5 -> loss = 2.5
        loss, _ = compute_causal_wrong_ce_loss(
            positive_ce_sum=6.0,
            positive_token_count=2,
            wrong_causal_ce_sum=1.0,
            wrong_causal_token_count=2,
        )
        assert loss is not None
        assert math.isclose(loss, 2.5)

    def test_non_finite_pos_mean_raises(self):
        with pytest.raises(FloatingPointError):
            compute_causal_wrong_ce_loss(
                positive_ce_sum=float("inf"),
                positive_token_count=1,
                wrong_causal_ce_sum=0.0,
                wrong_causal_token_count=0,
            )

    def test_non_finite_wrong_mean_raises(self):
        with pytest.raises(FloatingPointError):
            compute_causal_wrong_ce_loss(
                positive_ce_sum=1.0,
                positive_token_count=1,
                wrong_causal_ce_sum=float("nan"),
                wrong_causal_token_count=1,
            )

    def test_metrics_has_signal_batch(self):
        _, metrics = compute_causal_wrong_ce_loss(
            positive_ce_sum=1.0,
            positive_token_count=1,
            wrong_causal_ce_sum=0.0,
            wrong_causal_token_count=0,
        )
        assert metrics["causal_wrong_ce/zero_signal_batch"] == 0.0
        assert metrics["causal_wrong_ce/has_positive"] == 1.0
        assert metrics["causal_wrong_ce/has_wrong"] == 0.0


# ===========================================================================
# Config
# ===========================================================================

class TestConfig:
    def test_disabled_by_default(self):
        cfg = CausalWrongCEConfig.from_mapping({})
        assert cfg.enable is False

    def test_enable_true(self):
        cfg = CausalWrongCEConfig.from_mapping({"enable": True})
        assert cfg.enable is True

    def test_none_mapping(self):
        cfg = CausalWrongCEConfig.from_mapping(None)
        assert cfg.enable is False

    def test_non_mapping_raises(self):
        with pytest.raises(ValueError):
            CausalWrongCEConfig.from_mapping("not_a_dict")  # type: ignore[arg-type]


# ===========================================================================
# Causal source (staircase_size=1) numeric fixture
# ===========================================================================

# Import the internal shift function directly for numeric validation.
try:
    from verl.workers.actor.dp_actor import _shift_preexpanded_logits_source_impl
    _ACTOR_AVAILABLE = True
except Exception:
    _ACTOR_AVAILABLE = False


@pytest.mark.skipif(not _ACTOR_AVAILABLE, reason="dp_actor not importable in test env")
class TestCausalSourceNumeric:
    """Verify that staircase_size=1 maps noisy token j to clean token j-1
    (and token 0 to last prompt token), which is the causal wrong CE source.
    """

    def _make_fixture(self, L0: int = 3, L1: int = 4, D: int = 2, device: str = "cpu"):
        """Single document: [prompt(L0) | clean_response(L1) | noisy_response(L1)]
        All ones hidden so we can check index mapping by modifying specific positions.
        """
        total_len = L0 + 2 * L1
        hidden = torch.zeros(total_len, D, device=device)
        # Unique values per token to trace them
        for i in range(total_len):
            hidden[i] = float(i + 1)
        cu_seqlens = torch.tensor([0, total_len], dtype=torch.int32)
        L0_t = torch.tensor([L0], dtype=torch.int32)
        L1_t = torch.tensor([L1], dtype=torch.int32)
        return hidden, cu_seqlens, L0_t, L1_t

    def test_causal_source_maps_noisy_to_clean_causal(self):
        L0, L1, D = 3, 4, 2
        hidden, cu_seqlens, L0_t, L1_t = self._make_fixture(L0, L1, D)
        # staircase_size=1 -> causal shift
        shifted = _shift_preexpanded_logits_source_impl(
            hidden,
            cu_seqlens=cu_seqlens,
            L0=L0_t,
            L1=L1_t,
            staircase_size=1,
        )
        # noisy region starts at offset L0+L1
        noisy_start = L0 + L1
        # token 0 in noisy -> should map to last prompt token (index L0-1)
        assert torch.allclose(
            shifted[noisy_start], hidden[L0 - 1]
        ), f"token0 expected hidden[{L0-1}]={hidden[L0-1]} got {shifted[noisy_start]}"
        # token j in noisy -> should map to clean token j-1 (index L0 + j - 1)
        for j in range(1, L1):
            expected = hidden[L0 + j - 1]
            actual = shifted[noisy_start + j]
            assert torch.allclose(actual, expected), (
                f"noisy token {j}: expected hidden[{L0+j-1}]={expected} got {actual}"
            )

    def test_staircase_normal_maps_to_block_starts(self):
        """staircase_size=L1 should produce DIFFERENT output from staircase_size=1
        on the noisy region (they model different source positions).

        This is a regression test: the causal source (size=1) and the block
        staircase source (size=L1) must differ on non-trivial inputs so that the
        two CE branches see genuinely different logit sources.
        """
        L0, L1, D = 3, 4, 2
        hidden, cu_seqlens, L0_t, L1_t = self._make_fixture(L0, L1, D)
        shifted_causal = _shift_preexpanded_logits_source_impl(
            hidden,
            cu_seqlens=cu_seqlens,
            L0=L0_t,
            L1=L1_t,
            staircase_size=1,
        )
        shifted_block = _shift_preexpanded_logits_source_impl(
            hidden,
            cu_seqlens=cu_seqlens,
            L0=L0_t,
            L1=L1_t,
            staircase_size=L1,
        )
        noisy_start = L0 + L1
        # Token 0 in noisy always maps to the last prompt token (same for both).
        assert torch.allclose(
            shifted_block[noisy_start], shifted_causal[noisy_start]
        ), "token 0: both staircase sizes map to last prompt token"
        # Tokens 1..L1-1 must DIFFER between causal (size=1) and block (size=L1).
        for j in range(1, L1):
            assert not torch.allclose(
                shifted_block[noisy_start + j], shifted_causal[noisy_start + j]
            ), (
                f"token {j}: causal and block staircases must produce different "
                f"sources (causal={shifted_causal[noisy_start+j].tolist()}, "
                f"block={shifted_block[noisy_start+j].tolist()})"
            )

    def test_causal_source_has_grad(self):
        """Grad must flow through causal source (no no_grad path)."""
        L0, L1, D = 2, 3, 4
        hidden_base, cu_seqlens, L0_t, L1_t = self._make_fixture(L0, L1, D)
        hidden = hidden_base.clone().requires_grad_(True)
        shifted = _shift_preexpanded_logits_source_impl(
            hidden,
            cu_seqlens=cu_seqlens,
            L0=L0_t,
            L1=L1_t,
            staircase_size=1,
        )
        noisy_start = L0 + L1
        loss = shifted[noisy_start:noisy_start + L1].sum()
        loss.backward()
        assert hidden.grad is not None
        assert hidden.grad.abs().sum().item() > 0


# ===========================================================================
# Masks disjoint: positive & wrong tokens cannot overlap
# ===========================================================================

class TestMaskDisjoint:
    def test_positive_and_wrong_masks_disjoint(self):
        statuses = ["correct", "wrong", "uncertain", "correct", "wrong"]
        roles = torch.tensor(
            build_causal_wrong_ce_response_roles(verification_statuses=statuses),
            dtype=torch.int8,
        )
        positive_mask = roles.gt(0)
        wrong_mask = roles.lt(0)
        assert not (positive_mask & wrong_mask).any(), (
            "positive and wrong masks must be disjoint"
        )


# ===========================================================================
# Ghost / underfill ownership
# ===========================================================================

class TestGhostOwnership:
    """Verify that role=0 is assigned on non-label positions (simulated)."""

    def _simulate_role_expansion(
        self,
        roles: list[int],
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Replicate the dp_actor _pack_response_scalar -> squeeze -> response_mask
        zeroing logic for testing purposes.
        """
        role_tensor = torch.tensor(roles, dtype=torch.int8)
        # Broadcast per-response role to per-token and zero non-label positions
        expanded = role_tensor.unsqueeze(-1).expand_as(response_mask)
        result = torch.where(
            response_mask.bool(),
            expanded,
            torch.zeros_like(expanded),
        )
        return result

    def test_ghost_responses_get_zero_role(self):
        roles = [TOKEN_ROLE_POSITIVE, TOKEN_ROLE_WRONG_CAUSAL, TOKEN_ROLE_DROP]
        # Ghost row: second response has all False response_mask
        response_mask = torch.tensor([
            [1, 1, 0, 0],
            [0, 0, 0, 0],  # ghost
            [1, 0, 0, 0],
        ], dtype=torch.bool)
        result = self._simulate_role_expansion(roles, response_mask)
        # Ghost row must all be 0
        assert result[1].eq(0).all(), "ghost row must have role=0"

    def test_non_label_tokens_get_zero_role(self):
        roles = [TOKEN_ROLE_WRONG_CAUSAL]
        response_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
        result = self._simulate_role_expansion(roles, response_mask)
        # Post-response tokens must be 0
        assert result[0, 2].item() == 0
        assert result[0, 3].item() == 0
        # Label tokens carry the role
        assert result[0, 0].item() == TOKEN_ROLE_WRONG_CAUSAL
        assert result[0, 1].item() == TOKEN_ROLE_WRONG_CAUSAL


# ===========================================================================
# Integration: full loss tensor with torch CE
# ===========================================================================

class TestFullLossTensor:
    """End-to-end differentiable loss with small vocabulary."""

    def _setup(self):
        torch.manual_seed(42)
        V = 8
        # Two tokens positive, two tokens wrong
        hidden_pos = torch.randn(2, V, requires_grad=True)
        hidden_wrong = torch.randn(2, V, requires_grad=True)
        labels_pos = torch.tensor([3, 5])
        labels_wrong = torch.tensor([2, 6])
        return hidden_pos, hidden_wrong, labels_pos, labels_wrong

    def test_loss_is_differentiable(self):
        hidden_pos, hidden_wrong, labels_pos, labels_wrong = self._setup()
        ce_pos = torch.nn.functional.cross_entropy(
            hidden_pos, labels_pos, reduction="none"
        )
        ce_wrong = torch.nn.functional.cross_entropy(
            hidden_wrong, labels_wrong, reduction="none"
        )
        pos_sum = float(ce_pos.sum().item())
        wrong_sum = float(ce_wrong.sum().item())
        loss_val, metrics = compute_causal_wrong_ce_loss(
            positive_ce_sum=pos_sum,
            positive_token_count=2,
            wrong_causal_ce_sum=wrong_sum,
            wrong_causal_token_count=2,
        )
        assert loss_val is not None
        assert math.isfinite(loss_val)
        # Reconstruct differentiable loss and check backward
        pos_mean = ce_pos.mean()
        wrong_mean = ce_wrong.mean()
        differentiable_loss = pos_mean - wrong_mean
        differentiable_loss.backward()
        assert hidden_pos.grad is not None
        assert hidden_wrong.grad is not None
        assert math.isclose(float(differentiable_loss.detach().item()), loss_val, rel_tol=1e-5)


def _collect_replay_item(
    *,
    block_idx: int,
    position_start: int,
    token_base: int,
    is_final: bool,
) -> dict:
    block = 4
    return {
        "block_idx": block_idx,
        "position": [position_start, position_start + block],
        "input_ids": torch.arange(
            token_base, token_base + block, dtype=torch.long
        ),
        "self_conditioning_signal": torch.arange(
            block * 3, dtype=torch.float32
        ).reshape(block, 3),
        "is_final_denoise_row": is_final,
    }


def test_collect_training_data_wrong_response_keeps_only_final_rows(
    monkeypatch,
):
    """Production collector packs one final state per wrong-response block."""
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    monkeypatch.delenv("VERL_QWEN_BLOCK_LOCAL_REPLAY", raising=False)
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            actor={
                "dense_noisy_pg": {"enable": False},
                "generation_consistency_jsd": {
                    "enable": True,
                    "objective": "trace_all_ce",
                    "mode": "replace",
                    "trace_replay_all_steps": True,
                    "trace_all_ce_token_scope": "active_block_all",
                    "trace_all_ce_replay_input_mode": "exact",
                    "trace_all_ce_state_selection": {"enable": False},
                    "trace_all_ce_packing": "all_blocks_per_round",
                    "trace_all_ce_packing_materialization": "dense",
                    "trace_all_ce_parallelism": "none",
                    "trace_all_ce_underfill_fallback": "block_only",
                    "trace_all_ce_packing_require_multi_block": False,
                    "stream_replay_chunk_size": 1,
                    "causal_wrong_ce": {"enable": True},
                },
                "forward_add_noise_kwargs": {
                    "only_train_current_step": True
                },
            },
            rollout=SimpleNamespace(
                engine_kwargs={"block_length": 4, "mask_token_id": 99}
            ),
        ),
        algorithm={},
    )
    trainer._causal_wrong_ce_config = lambda: CausalWrongCEConfig(
        enable=True
    )

    input_ids = torch.tensor(
        [[1, 2, 10, 11, 12, 13, 20, 21, 22, 23]]
    )
    labels = torch.tensor(
        [[-100, -100, 10, 11, 12, 13, 20, 21, 22, 23]]
    )
    states = {
        0: _collect_replay_item(
            block_idx=0,
            position_start=0,
            token_base=50,
            is_final=False,
        ),
        1: _collect_replay_item(
            block_idx=0,
            position_start=0,
            token_base=54,
            is_final=True,
        ),
        4: _collect_replay_item(
            block_idx=1,
            position_start=4,
            token_base=60,
            is_final=False,
        ),
        5: _collect_replay_item(
            block_idx=1,
            position_start=4,
            token_base=64,
            is_final=True,
        ),
    }
    result = trainer._collect_training_data(
        input_ids_aligned=input_ids,
        labels_aligned=labels,
        attention_mask_aligned=torch.ones_like(
            input_ids, dtype=torch.bool
        ),
        prompt_end_pos=torch.tensor([2]),
        seq_start_pos=torch.tensor([0]),
        seq_end_pos=torch.tensor([10]),
        step_maps_aligned=torch.zeros((1, 8), dtype=torch.long),
        log_probs_aligned=torch.zeros((1, 8)),
        ar_probs_aligned=None,
        ar_sequence_advantages=None,
        rollout_self_conditioning=np.asarray([states], dtype=object),
        L0=2,
        L1=8,
        non_tensor_batch={
            "causal_wrong_ce_role": np.asarray(
                [TOKEN_ROLE_WRONG_CAUSAL], dtype=np.int8
            ),
            "rollout_quality_keep": np.asarray([True]),
            "response_uid": np.asarray(["wrong-response"], dtype=object),
        },
        rm_scores=torch.ones((1, 8)),
        dummy_tensor=torch.zeros(1),
    )
    ext_labels = result[1]
    expanded_non_tensor = result[15]
    metrics = result[-1]

    assert ext_labels.shape[0] == 1
    assert int(ext_labels[:, 10:].ne(-100).sum().item()) == 8
    assert list(expanded_non_tensor["causal_wrong_ce_role"]) == [
        TOKEN_ROLE_WRONG_CAUSAL
    ]
    assert metrics["causal_wrong_ce/final_state_rows_selected"] == 2.0
    assert metrics["causal_wrong_ce/nonfinal_state_rows_selected"] == 0.0
    assert metrics["causal_wrong_ce/duplicate_label_tokens"] == 0.0
    assert metrics["causal_wrong_ce/negative_ce_label_tokens"] == 8.0
