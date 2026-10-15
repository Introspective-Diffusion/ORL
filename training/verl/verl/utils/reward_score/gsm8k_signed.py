"""Signed GSM8K rule reward for GRPO."""

from verl.utils.reward_score.gsm8k import compute_score as gsm8k_compute_score


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
):
    del data_source, extra_info
    score = gsm8k_compute_score(
        solution_str=solution_str,
        ground_truth=str(ground_truth),
        method="strict",
        format_score=-1.0,
        score=1.0,
    )
    return 1.0 if float(score) > 0.0 else -1.0
