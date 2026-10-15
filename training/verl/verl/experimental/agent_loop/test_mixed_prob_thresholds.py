"""CPU tests for the mixed_prob_thresholds hydra-key parsing and assignment.

rollout_extra_sampling_params.mixed_prob_thresholds distributes a comma
separated threshold list round-robin over sample positions. The
injection must be inert when the key is unset.
"""

import pytest

pytest.importorskip("ray")

from verl.experimental.agent_loop.agent_loop import (  # noqa: E402
    _assign_mixed_prob_thresholds,
    _mixed_prob_threshold_for_position,
    _parse_mixed_prob_thresholds,
)


class _FakeBatch:
    def __init__(self, n, validate=False):
        self._n = n
        self.non_tensor_batch = {}
        self.meta_info = {"validate": validate}

    def __len__(self):
        return self._n


_ROLLOUT_CFG = {
    "engine_kwargs": {
        "rollout_extra_sampling_params": {
            "mixed_prob_thresholds": "0.6,0.6,0.7,0.7,0.8,0.8,0.9,0.9"
        }
    }
}


def test_parse_none_and_empty_are_inert():
    assert _parse_mixed_prob_thresholds(None) is None
    assert _parse_mixed_prob_thresholds("") is None
    assert _parse_mixed_prob_thresholds([]) is None


def test_parse_comma_separated_string():
    parsed = _parse_mixed_prob_thresholds("0.6,0.6,0.7,0.7,0.8,0.8,0.9,0.9")
    assert parsed == [0.6, 0.6, 0.7, 0.7, 0.8, 0.8, 0.9, 0.9]


def test_parse_tolerates_whitespace_and_trailing_comma():
    assert _parse_mixed_prob_thresholds(" 0.6 , 0.9 ,") == [0.6, 0.9]


def test_parse_list_of_numbers():
    assert _parse_mixed_prob_thresholds([0.6, "0.9"]) == [0.6, 0.9]


def test_parse_rejects_out_of_range():
    with pytest.raises(ValueError):
        _parse_mixed_prob_thresholds("0.6,1.5")
    with pytest.raises(ValueError):
        _parse_mixed_prob_thresholds("0.0,0.9")


def test_assign_covers_full_list_on_full_batch():
    batch = _FakeBatch(32)
    _assign_mixed_prob_thresholds(batch, _ROLLOUT_CFG)
    col = batch.non_tensor_batch["mixed_prob_threshold"]
    assert len(col) == 32
    # Every GRPO group of 8 consecutive rollouts spans the full dictionary
    # (this is why assignment must happen on the FULL batch, pre-chunking:
    # 8 workers x 4 samples with worker-local positions never reach 0.8/0.9).
    for g in range(4):
        assert sorted(set(col[g * 8 : (g + 1) * 8])) == [0.6, 0.7, 0.8, 0.9]


def test_assign_is_idempotent_and_skips_validate():
    batch = _FakeBatch(8)
    _assign_mixed_prob_thresholds(batch, _ROLLOUT_CFG)
    first = batch.non_tensor_batch["mixed_prob_threshold"]
    _assign_mixed_prob_thresholds(batch, _ROLLOUT_CFG)
    assert batch.non_tensor_batch["mixed_prob_threshold"] is first

    val_batch = _FakeBatch(8, validate=True)
    _assign_mixed_prob_thresholds(val_batch, _ROLLOUT_CFG)
    assert "mixed_prob_threshold" not in val_batch.non_tensor_batch

    unset = _FakeBatch(8)
    _assign_mixed_prob_thresholds(unset, {"engine_kwargs": {}})
    assert "mixed_prob_threshold" not in unset.non_tensor_batch


def test_round_robin_assignment():
    thresholds = [0.6, 0.6, 0.7, 0.7, 0.8, 0.8, 0.9, 0.9]
    assigned = [
        _mixed_prob_threshold_for_position(thresholds, i) for i in range(16)
    ]
    assert assigned == thresholds + thresholds
    # Every ROLLOUT_N=8 group over one prompt sees the full threshold dictionary.
    assert sorted(set(assigned)) == [0.6, 0.7, 0.8, 0.9]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
