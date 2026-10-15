import pytest
import torch

from verl.trainer.ppo.response_level_ce import (
    ResponseCountRecords,
    build_response_level_aec_pair_weights,
    build_response_level_ce_weights,
    extract_local_response_count_records,
    merge_response_count_records,
)


def test_response_level_aec_counts_unique_pairs_not_pair_tokens():
    pair_ids = torch.tensor(
        [
            [0, 0, 1, -1],
            [2, 2, -1, -1],
            [3, 3, 3, -1],
        ]
    )
    pair_mask = pair_ids.ge(0)
    result = build_response_level_aec_pair_weights(
        ["a", "a", "b"],
        pair_mask,
        pair_ids,
    )
    assert result.global_responses == 2
    assert result.global_active_tokens == 4
    assert result.min_tokens_per_response == 1
    assert result.max_tokens_per_response == 3
    assert result.weights[0, :3].tolist() == pytest.approx(
        [1 / 3, 1 / 3, 1 / 3]
    )
    assert result.weights[2, :3].tolist() == pytest.approx([1.0, 1.0, 1.0])
    unique_weight_sum = 0.0
    for pair_id in (0, 1, 2, 3):
        positions = pair_ids.eq(pair_id)
        unique_weight_sum += float(result.weights[positions][0].item())
    assert unique_weight_sum == pytest.approx(2.0)


def test_original_response_is_equal_weight_across_multiple_rows():
    mask = torch.tensor(
        [
            [1, 1, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    result = build_response_level_ce_weights(["a", "a", "b"], mask)
    assert result.global_responses == 2
    assert result.global_active_tokens == 7
    assert result.min_tokens_per_response == 3
    assert result.max_tokens_per_response == 4
    assert result.weights[:2].sum().item() == pytest.approx(1.0)
    assert result.weights[2].sum().item() == pytest.approx(1.0)
    assert result.weights.sum().item() == pytest.approx(2.0)
    assert result.weights[0, 0].item() == pytest.approx(1.0 / 3.0)
    assert result.weights[2, 0].item() == pytest.approx(1.0 / 4.0)


def test_cross_rank_partial_counts_are_summed_by_response():
    rank0 = extract_local_response_count_records(
        ["a", "b"],
        torch.tensor([[1, 1], [1, 0]], dtype=torch.bool),
    )
    rank1 = extract_local_response_count_records(
        ["a", "b"],
        torch.tensor([[1, 0], [1, 1]], dtype=torch.bool),
    )
    merged = merge_response_count_records([rank0, rank1])
    assert sorted(merged.counts.tolist()) == [3.0, 3.0]


def test_nonzero_sp_rank_uses_gathered_rank0_counts_without_contributing():
    mask = torch.tensor([[1, 1, 1]], dtype=torch.bool)
    canonical = extract_local_response_count_records(["a"], mask)
    result = build_response_level_ce_weights(
        ["a"],
        mask,
        sequence_parallel_rank=1,
        gathered_record_batches=[canonical],
    )
    assert result.global_responses == 1
    assert result.weights.sum().item() == pytest.approx(1.0)


def test_missing_response_uid_fails_closed():
    with pytest.raises(RuntimeError, match="non-empty response_uid"):
        build_response_level_ce_weights(
            [None],
            torch.tensor([[1]], dtype=torch.bool),
        )


def test_record_shapes_are_validated():
    bad = ResponseCountRecords(
        keys=torch.ones((1, 3), dtype=torch.long),
        counts=torch.ones((1,), dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="shape"):
        merge_response_count_records([bad])
