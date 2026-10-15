"""Focused tests for active_input_mode closed-loop SDAR rollout.

Covers: exact identity, deterministic random model view / logical-state
preservation, per-step value changes, prompt preservation, batched
independence, inactive batch members, and exact captured model view.

Scope matches the portable residual-rollout100 recipe:
  - active_input_mode: "exact" | "random_active_block"
  - No active_random_fraction, no uniform_per_block, no capture budget.
"""

from __future__ import annotations

import unittest

import torch

from sglang.srt.dllm.algorithm.sdar_prefix_kv import SDARPrefixKV


def _make_algorithm(mode: str) -> SDARPrefixKV:
    algorithm = object.__new__(SDARPrefixKV)
    algorithm.active_input_mode = mode
    algorithm.deterministic_randtok = True
    algorithm.randtok_seed = 12345
    algorithm.random_token_high = 151669
    algorithm.block_size = 8
    algorithm._request_generated_tokens_by_req = {}
    return algorithm


class ExactModeTest(unittest.TestCase):
    """exact mode must be behavior-identical to the pre-feature baseline."""

    def test_exact_mode_returns_original_tensor(self) -> None:
        """_build_denoise_model_input_ids returns the same object in exact mode."""
        algorithm = _make_algorithm("exact")
        logical = torch.tensor([10, 11, 12, 13])

        result = algorithm._build_denoise_model_input_ids(
            logical,
            [(0, 4)],
            [torch.tensor([True, False, False, False])],
            [0],
            step_idx=1,
        )

        self.assertIs(result, logical)

    def test_exact_mode_capture_returns_logical_clone(self) -> None:
        """_capture_replay_input_ids in exact mode returns a clone of the logical block."""
        algorithm = _make_algorithm("exact")
        logical = torch.tensor([10, 11, 12, 13])

        captured = algorithm._capture_replay_input_ids(logical, logical, 0, 4)

        torch.testing.assert_close(captured, logical)
        self.assertIsNot(captured, logical)


class RandomActiveBlockModelViewTest(unittest.TestCase):
    """random_active_block mode: model view is deterministic; logical state is unchanged."""

    def test_deterministic_random_view_and_logical_state_preserved(self) -> None:
        """Two calls with the same args produce the same view; logical tensor unchanged."""
        algorithm = _make_algorithm("random_active_block")
        logical = torch.tensor([10, 11, 12, 13, 14, 15])
        original = logical.clone()
        prompt_mask = torch.tensor([True, True, False, False, False, False])

        view = algorithm._build_denoise_model_input_ids(
            logical,
            [(0, 6)],
            [prompt_mask],
            [2],
            step_idx=3,
        )
        repeated = algorithm._build_denoise_model_input_ids(
            logical,
            [(0, 6)],
            [prompt_mask],
            [2],
            step_idx=3,
        )
        expected = algorithm._random_unresolved(logical, block_idx=2, step_idx=3)

        # logical tensor must not be mutated
        torch.testing.assert_close(logical, original)
        # prompt positions pass through unchanged
        torch.testing.assert_close(view[:2], logical[:2])
        # non-prompt positions match _random_unresolved output
        torch.testing.assert_close(view[2:], expected[2:])
        # deterministic: same call produces same view
        torch.testing.assert_close(view, repeated)

    def test_view_is_a_new_tensor_not_aliased_to_logical(self) -> None:
        """Model view must not share storage with the logical tensor."""
        algorithm = _make_algorithm("random_active_block")
        logical = torch.tensor([20, 21, 22, 23, 24])
        prompt_mask = torch.tensor([True, False, False, False, False])

        view = algorithm._build_denoise_model_input_ids(
            logical,
            [(0, 5)],
            [prompt_mask],
            [0],
            step_idx=1,
        )

        self.assertIsNot(view, logical)
        self.assertFalse(view.data_ptr() == logical.data_ptr())

    def test_per_step_values_change_and_nonprompt_always_randomized(self) -> None:
        """Different step_idx values yield different random IDs for non-prompt positions."""
        algorithm = _make_algorithm("random_active_block")
        logical = torch.tensor([20, 21, 22, 23, 24])
        prompt_mask = torch.tensor([True, False, False, False, False])

        step_one = algorithm._build_denoise_model_input_ids(
            logical, [(0, 5)], [prompt_mask], [0], step_idx=1
        )
        step_two = algorithm._build_denoise_model_input_ids(
            logical, [(0, 5)], [prompt_mask], [0], step_idx=2
        )

        # prompt preserved
        self.assertTrue(torch.equal(step_one[:1], logical[:1]))
        self.assertTrue(torch.equal(step_two[:1], logical[:1]))
        # non-prompt IDs differ across steps
        self.assertFalse(torch.equal(step_one[1:], step_two[1:]))
        # non-prompt IDs differ from original logical IDs
        self.assertFalse(torch.equal(step_one[1:], logical[1:]))
        self.assertFalse(torch.equal(step_two[1:], logical[1:]))

    def test_prompt_positions_always_pass_through_unchanged(self) -> None:
        """All-prompt span: model view equals logical (no randomization possible)."""
        algorithm = _make_algorithm("random_active_block")
        logical = torch.tensor([5, 6, 7, 8])
        all_prompt = torch.ones(4, dtype=torch.bool)

        view = algorithm._build_denoise_model_input_ids(
            logical, [(0, 4)], [all_prompt], [0], step_idx=1
        )

        torch.testing.assert_close(view, logical)


class BatchedViewTest(unittest.TestCase):
    """Batched _build_denoise_model_input_ids produces the same result as per-request calls."""

    def test_batched_view_matches_independent_single_views(self) -> None:
        algorithm = _make_algorithm("random_active_block")
        logical = torch.tensor([31, 32, 33, 34, 41, 42, 43, 44])
        masks = [
            torch.tensor([True, False, False, False]),
            torch.tensor([False, False, False, False]),
        ]

        batched = algorithm._build_denoise_model_input_ids(
            logical,
            [(0, 4), (4, 8)],
            masks,
            [0, 1],
            step_idx=7,
        )
        first = algorithm._build_denoise_model_input_ids(
            logical[:4],
            [(0, 4)],
            [masks[0]],
            [0],
            step_idx=7,
        )
        second = algorithm._build_denoise_model_input_ids(
            logical[4:],
            [(0, 4)],
            [masks[1]],
            [1],
            step_idx=7,
        )

        torch.testing.assert_close(batched, torch.cat([first, second]))

    def test_inactive_batch_member_is_not_randomized(self) -> None:
        """Inactive entries (active=False) must not be touched."""
        algorithm = _make_algorithm("random_active_block")
        logical = torch.tensor([31, 32, 33, 34, 41, 42, 43, 44])

        view = algorithm._build_denoise_model_input_ids(
            logical,
            [(0, 4), (4, 8)],
            [
                torch.tensor([True, False, False, False]),
                torch.tensor([False, False, False, False]),
            ],
            [0, 1],
            step_idx=4,
            active=[True, False],
        )

        # inactive second span must equal logical
        torch.testing.assert_close(view[4:], logical[4:])


class CapturedReplayInputTest(unittest.TestCase):
    """_capture_replay_input_ids records the exact model-view IDs consumed."""

    def test_closed_loop_captures_model_view_not_logical(self) -> None:
        algorithm = _make_algorithm("random_active_block")
        logical = torch.tensor([10, 11, 12, 13, 14, 15])
        prompt_mask = torch.tensor([True, True, False, False, False, False])

        model_view = algorithm._build_denoise_model_input_ids(
            logical,
            [(0, 6)],
            [prompt_mask],
            [2],
            step_idx=3,
        )
        captured = algorithm._capture_replay_input_ids(
            model_view,
            logical,
            0,
            6,
        )

        # captured equals the model view
        torch.testing.assert_close(captured, model_view)
        # captured differs from logical for non-prompt positions
        self.assertFalse(torch.equal(captured[2:], logical[2:]))
        # captured is a fresh copy
        self.assertIsNot(captured, model_view)

    def test_exact_mode_captured_ids_equal_logical(self) -> None:
        algorithm = _make_algorithm("exact")
        logical = torch.tensor([10, 11, 12, 13])

        captured = algorithm._capture_replay_input_ids(logical, logical, 0, 4)

        torch.testing.assert_close(captured, logical)


if __name__ == "__main__":
    unittest.main()
