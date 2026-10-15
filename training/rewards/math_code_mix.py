"""Per-data_source reward router for the GSM8K + MATH + code mixture.

    code sources -> code_exec.code_score (strict: closed ```python block,
                    top-level tests stripped, all sampled tests must pass)
    "gsm8k"      -> verl gsm8k_training_reward (strict '#### <answer>' extraction)
    otherwise    -> verl enhanced_math_reward (\\boxed{} + math_verify, tri-state)

Every branch returns the same five keys; see code_exec.py for why.
"""

import importlib.util
import os

from verl.utils.reward_score import enhanced_math_reward, gsm8k_training_reward

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("orl_code_exec", os.path.join(_here, "code_exec.py"))
_code = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_code)

KEYS = ("score", "acc", "verification_status", "verified_wrong", "uncertain")


def _project(r):
    return {k: r[k] for k in KEYS}


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if _code.is_code_source(data_source):
        return _project(_code.code_score(solution_str, ground_truth, require_closed_block=True, strip_toplevel_tests=True))
    if "gsm8k" in (data_source or "").lower():
        return _project(gsm8k_training_reward.compute_score(data_source, solution_str, ground_truth, extra_info))
    return _project(enhanced_math_reward.compute_score(data_source, solution_str, ground_truth, extra_info))
