"""Per-data_source reward router for mixed-dataset cells (GSM8K + DAPO/MATH).

GSM8K prompts instruct '#### <answer>' outputs (strict extraction via
gsm8k_training_reward); DAPO/MATH prompts instruct \\boxed{} outputs
(enhanced_math_reward: legacy verifier OR math_verify, tri-state).
Routing on data_source keeps each format on its correct verifier —
enhanced_math_reward scores 0/20 on well-formed '#### n' solutions.

Both branches MUST return the identical key set: AgentLoopWorkers stamp the
reward dict's key list into chunk meta_info as `reward_extra_keys`, and
DataProto.concat asserts meta_info equality across chunks — heterogeneous
key sets crash generate_sequences (protocol.py:967).
"""

from verl.utils.reward_score import enhanced_math_reward, gsm8k_training_reward


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
):
    if "gsm8k" in str(data_source):
        r = gsm8k_training_reward.compute_score(
            data_source, solution_str, ground_truth, extra_info
        )
        correct = float(r["score"]) >= 1.0
        return {
            "score": r["score"],
            "acc": r["acc"],
            "verification_status": r["verification_status"],
            "legacy_correct": correct,
            "enhanced_correct": correct,
            "verified_wrong": r["verified_wrong"],
            "uncertain": r["uncertain"],
            "recovered_false_negative": False,
            "math_verify_available": False,
        }
    r = enhanced_math_reward.compute_score(
        data_source, solution_str, ground_truth, extra_info
    )
    return {
        "score": r["score"],
        "acc": r["acc"],
        "verification_status": r["verification_status"],
        "legacy_correct": r["legacy_correct"],
        "enhanced_correct": r["enhanced_correct"],
        "verified_wrong": r["verified_wrong"],
        "uncertain": r["uncertain"],
        "recovered_false_negative": r["recovered_false_negative"],
        "math_verify_available": r["math_verify_available"],
    }
