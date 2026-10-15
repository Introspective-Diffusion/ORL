"""Strict 0/1 GSM8K reward for correct-only DiffusionGemma training."""

from verl.utils.reward_score import gsm8k


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
):
    del data_source, extra_info
    correct = (
        float(
            gsm8k.compute_score(
                solution_str=solution_str,
                ground_truth=str(ground_truth),
                method="strict",
                format_score=0.0,
                score=1.0,
            )
        )
        >= 1.0
    )
    score = 1.0 if correct else 0.0
    return {
        "score": score,
        "acc": score,
        "verification_status": "correct" if correct else "wrong",
        "verified_wrong": not correct,
        "uncertain": False,
    }
