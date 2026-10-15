import pytest

from verl.trainer.ppo.causal_prob_ce_filter import (
    build_causal_prob_keep_mask,
)


def test_strict_absolute_threshold_filters_only_response_positions():
    result = build_causal_prob_keep_mask(
        [0.95, 0.90, 0.91],
        local_tail_start=2,
        local_tail_length=5,
        response_start=3,
        response_end=6,
        threshold=0.90,
    )
    # tail positions 2 and 6 are outside the response and remain inert/true.
    assert result.keep_mask.tolist() == [True, True, False, True, True]
    assert result.response_tokens == 3
    assert result.kept_tokens == 2


def test_threshold_is_strictly_greater_than():
    result = build_causal_prob_keep_mask(
        [0.9],
        local_tail_start=0,
        local_tail_length=1,
        response_start=0,
        response_end=1,
        threshold=0.9,
    )
    assert result.keep_mask.tolist() == [False]


def test_less_than_direction_keeps_low_confidence_tokens_only():
    result = build_causal_prob_keep_mask(
        [0.2, 0.9, 0.95],
        local_tail_start=0,
        local_tail_length=3,
        response_start=0,
        response_end=3,
        threshold=0.9,
        direction="less_than",
    )
    assert result.keep_mask.tolist() == [True, False, False]
    assert result.kept_tokens == 1


@pytest.mark.parametrize("value", [None, float("nan"), -0.1, 1.1, "bad"])
def test_missing_or_invalid_probability_fails_closed(value):
    with pytest.raises(RuntimeError):
        build_causal_prob_keep_mask(
            [value],
            local_tail_start=0,
            local_tail_length=1,
            response_start=0,
            response_end=1,
            threshold=0.9,
        )


def test_missing_probability_position_fails_closed():
    with pytest.raises(RuntimeError, match="missing commit probability"):
        build_causal_prob_keep_mask(
            [],
            local_tail_start=0,
            local_tail_length=1,
            response_start=0,
            response_end=1,
            threshold=0.9,
        )


def test_threshold_validation():
    with pytest.raises(ValueError):
        build_causal_prob_keep_mask(
            [0.9],
            local_tail_start=0,
            local_tail_length=1,
            response_start=0,
            response_end=1,
            threshold=1.1,
        )


def test_direction_validation():
    with pytest.raises(ValueError, match="direction"):
        build_causal_prob_keep_mask(
            [0.9],
            local_tail_start=0,
            local_tail_length=1,
            response_start=0,
            response_end=1,
            threshold=0.9,
            direction="sideways",
        )
