"""Signed rule reward for EleutherAI/Hendrycks MATH."""

from verl.utils.reward_score.math_reward import compute_score as math_compute_score


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
):
    del data_source, extra_info
    score = math_compute_score(
        solution_str=solution_str,
        ground_truth=str(ground_truth),
    )
    return 1.0 if float(score) > 0.0 else -1.0
