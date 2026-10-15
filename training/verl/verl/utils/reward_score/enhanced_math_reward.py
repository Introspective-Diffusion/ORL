"""Enhanced conservative three-state MATH verifier.

Tri-state outputs
-----------------
correct   : legacy verifier OR math_verify accepts
uncertain : extraction failed, math_verify could not parse either side,
            timeout, or any other ambiguous outcome
wrong     : math_verify parsed BOTH gold and prediction into structured
            symbolic/numeric artifacts AND confidently rejected the match

Design guarantees
-----------------
- Unknown / uncertain responses are NEVER negative (score = 0.0).
- Standard score semantics: 1.0 for correct, 0.0 otherwise.
- Fixes two known false-negative classes from gs900-925 signed-residual runs:
    (1) ``\\text{ and }`` used as conjunction in multi-value answers was
        mistakenly treated as a unit suffix by ``remove_right_units``.
        math_verify correctly handles this pattern.
    (2) Equivalent polynomial factorizations with the minus sign distributed
        differently were rejected by string comparison.
        math_verify resolves these via symbolic algebra.
- Does NOT mutate math_reward.py semantics.
- Uses the HuggingFace math_verify metric as the enhanced OR verifier;
  falls back gracefully when math_verify is not installed (all enhanced
  results become uncertain, preserving legacy behavior).
- No raw eval / parse_expr on untrusted model output.
- No long-lived daemon threads; math_verify's own timeout mechanism is used.

Confidence detection
--------------------
Math-Verify parsing returns structured and raw fallback artifacts.
A parse is "successful" when the artifact list has at least 2 elements
(a parsed symbolic/numeric object + the raw string).  A list of length 1
contains only the raw string (malformed / not parseable).  Empty means
nothing was extracted at all.

Confidently wrong: math_verify returns 0.0 AND both sides have len >= 2.
Uncertain: either side has len < 2, or any exception / timeout occurs.

Threaded reward workers
-----------------------
DAPO calls reward functions from a thread pool. Signal-based parser timeouts
are therefore disabled, after first extracting and length-bounding only the
last boxed answer. This avoids parsing the full generated reasoning and keeps
the thread-safe parse surface small.

Return value
------------
dict with keys:
    score                  float  1.0 correct, 0.0 otherwise
    acc                    float  same as score (backward-compat alias)
    verification_status    str    'correct' | 'uncertain' | 'wrong'
    legacy_correct         bool   legacy math_reward.py accepted
    enhanced_correct       bool   math_verify OR legacy accepted
    verified_wrong         bool   both parsed + confidently rejected
    uncertain              bool   parse / timeout / ambiguous
    recovered_false_negative bool  math_verify accepted but legacy did not
    math_verify_available   bool   dependency loaded in this reward worker
"""

from typing import Optional

from verl.utils.reward_score import math_reward as _legacy

# ---------------------------------------------------------------------------
# Optional math_verify import and module-scope cache
# ---------------------------------------------------------------------------

_MATH_VERIFY_AVAILABLE = False
_MAX_EXTRACTED_ANSWER_CHARS = 4096

try:
    from math_verify import parse as _math_verify_parse
    from math_verify import verify as _math_verify_compare
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

    _GOLD_EXTRACTION = (LatexExtractionConfig(),)
    _PRED_EXTRACTION = (
        ExprExtractionConfig(),
        LatexExtractionConfig(),
    )
    _MATH_VERIFY_AVAILABLE = True
except ImportError:
    _GOLD_EXTRACTION = ()
    _PRED_EXTRACTION = ()


def _extract_last_boxed(solution_str: str) -> str | None:
    """Extract one bounded final boxed answer without parsing the reasoning."""
    marker = r"\boxed"
    marker_index = solution_str.rfind(marker)
    if marker_index < 0:
        return None
    brace_start = marker_index + len(marker)
    while (
        brace_start < len(solution_str)
        and solution_str[brace_start].isspace()
    ):
        brace_start += 1
    if (
        brace_start >= len(solution_str)
        or solution_str[brace_start] != "{"
    ):
        return None
    depth = 0
    for index in range(brace_start, len(solution_str)):
        value = solution_str[index]
        if value == "{":
            depth += 1
        elif value == "}":
            depth -= 1
            if depth == 0:
                answer = solution_str[brace_start + 1 : index]
                if len(answer) > _MAX_EXTRACTED_ANSWER_CHARS:
                    return None
                return answer
    return None


# ---------------------------------------------------------------------------
# Enhanced equivalence via math_verify
# ---------------------------------------------------------------------------

def _math_verify_result(solution_str: str, ground_truth: str):
    """Call math_metric and return (score, gold_artifact_count, pred_artifact_count).

    Returns None when math_verify is unavailable or an unrecoverable error
    occurs (caller treats this as uncertain).
    """
    if not _MATH_VERIFY_AVAILABLE:
        return None

    extracted = _extract_last_boxed(solution_str)
    if extracted is None:
        return None
    gold_boxed = r"\boxed{" + ground_truth + "}"
    pred_boxed = r"\boxed{" + extracted + "}"
    try:
        # DAPO executes reward functions in a ThreadPoolExecutor. Math-Verify's
        # default signal.alarm timeout is invalid outside the main thread, so
        # disable it after bounding input to one short boxed expression.
        gold_arts = _math_verify_parse(
            gold_boxed,
            _GOLD_EXTRACTION,
            parsing_timeout=None,
        )
        pred_arts = _math_verify_parse(
            pred_boxed,
            _PRED_EXTRACTION,
            parsing_timeout=None,
        )
        score = _math_verify_compare(
            gold_arts,
            pred_arts,
            timeout_seconds=None,
        )
        return float(score), len(gold_arts), len(pred_arts)
    except Exception:  # noqa: BLE001
        return None


def _enhanced_check(solution_str: str, ground_truth: str) -> Optional[bool]:
    """Return True (correct), False (confident wrong), or None (uncertain).

    Confidence rule: confidently wrong only when math_verify parsed BOTH sides
    into structured artifacts (artifact list len >= 2 for gold and pred).
    """
    result = _math_verify_result(solution_str, ground_truth)
    if result is None:
        return None  # math_verify unavailable or exception → uncertain

    score, gold_len, pred_len = result

    if score >= 1.0:
        return True

    # score == 0.0: determine confidence
    both_parsed = gold_len >= 2 and pred_len >= 2
    if both_parsed:
        return False  # confidently wrong

    return None  # uncertain (one or both sides did not parse into a structured artifact)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_score(
    data_source,
    solution_str: str,
    ground_truth: str,
    extra_info=None,
) -> dict:
    """Enhanced three-state MATH verifier.

    Parameters
    ----------
    data_source:
        Dataset identifier (unused; kept for API compatibility).
    solution_str:
        Full model response string.
    ground_truth:
        Gold answer string.
    extra_info:
        Ignored; kept for API compatibility.

    Returns
    -------
    dict
        score (float), acc (float), verification_status (str),
        legacy_correct (bool), enhanced_correct (bool),
        verified_wrong (bool), uncertain (bool),
        recovered_false_negative (bool).
    """
    del data_source, extra_info

    ground_truth = str(ground_truth)

    # ------------------------------------------------------------------
    # 1. Legacy verifier
    # ------------------------------------------------------------------
    legacy_correct = False
    try:
        legacy_score = _legacy.compute_score(
            solution_str=solution_str,
            ground_truth=ground_truth,
        )
        legacy_correct = float(legacy_score) >= 1.0
    except Exception:  # noqa: BLE001
        legacy_correct = False

    # ------------------------------------------------------------------
    # 2. Enhanced check via math_verify
    # ------------------------------------------------------------------
    enhanced_result: Optional[bool] = None
    if not legacy_correct:
        enhanced_result = _enhanced_check(solution_str, ground_truth)

    # ------------------------------------------------------------------
    # 3. Tri-state decision
    # ------------------------------------------------------------------
    if legacy_correct:
        status = "correct"
        enhanced_correct = True
        verified_wrong = False
        uncertain = False
        recovered_false_negative = False

    elif enhanced_result is True:
        # math_verify accepted; legacy did not → false negative recovered
        status = "correct"
        enhanced_correct = True
        verified_wrong = False
        uncertain = False
        recovered_false_negative = True

    elif enhanced_result is False:
        # Both sides parsed + math_verify confidently rejected → wrong
        status = "wrong"
        enhanced_correct = False
        verified_wrong = True
        uncertain = False
        recovered_false_negative = False

    else:
        # enhanced_result is None: math_verify unavailable / uncertain parse
        status = "uncertain"
        enhanced_correct = False
        verified_wrong = False
        uncertain = True
        recovered_false_negative = False

    score = 1.0 if status == "correct" else 0.0

    return {
        "score": score,
        "acc": score,
        "verification_status": status,
        "legacy_correct": legacy_correct,
        "enhanced_correct": enhanced_correct,
        "verified_wrong": verified_wrong,
        "uncertain": uncertain,
        "recovered_false_negative": recovered_false_negative,
        "math_verify_available": _MATH_VERIFY_AVAILABLE,
    }
