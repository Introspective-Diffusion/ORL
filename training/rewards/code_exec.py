"""Binary code-execution reward for verifier-filtered on-policy training.

Each sampled test runs the extracted ```python block in a fresh `python -I`
child process (process isolation, per-test wall timeout, child-side rlimits).
Both stdio and function-call problems are supported, with numeric-tolerant
output comparison. Score is 1.0 iff ALL sampled tests pass.

Two options select the paper recipes' semantics:

- ``require_closed_block``: the last ```python block must be closed. An
  unclosed block scores 0. (LLaDA-2.1-mini recipe: on; QwenDLM recipe: off.)
- ``strip_toplevel_tests``: in functional (assert-suite) mode, drop column-0
  ``assert`` statements, test calls and ``if __name__ == "__main__":`` blocks
  from the solution before running it. (LLaDA-2.1-mini recipe: on; QwenDLM
  recipe: off.)

The returned key set must stay identical across all reward functions used in
one batch: the correctness filter reads verification_status / verified_wrong /
uncertain, and DataProto.concat requires equal key sets.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import time

CODE_SOURCE_KEYS = ("taco", "apps", "codecontests", "codeforces", "leetcode", "code", "humaneval", "mbpp")
PER_TEST_TIMEOUT = 6.0
FUNCTIONAL_TIMEOUT = 20.0
TOTAL_BUDGET = 90.0
NUM_TOL = 1e-6

_RLIMIT_PRELUDE = (
    "import resource\n"
    "try:\n"
    "    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))\n"
    "    resource.setrlimit(resource.RLIMIT_NPROC, (128, 128))\n"
    "    resource.setrlimit(resource.RLIMIT_FSIZE, (16 << 20, 16 << 20))\n"
    "except Exception:\n"
    "    pass\n"
)

_FN_RUNNER = (
    "\nimport json as _json, sys as _sys\n"
    "_args = _json.loads(_sys.stdin.read())\n"
    "_res = {fn}(*_args)\n"
    "print(_json.dumps(_res))\n"
)

_TOPLEVEL_TEST_CALL = re.compile(
    r"^(assert\b|check\s*\(|test\w*\s*\(|unittest\.main\s*\(|doctest\.testmod\s*\(|pytest\.main\s*\()"
)


def is_code_source(data_source):
    ds = (data_source or "").lower()
    return any(k in ds for k in CODE_SOURCE_KEYS)


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    require_closed_block=True,
    strip_toplevel_tests=True,
):
    """Code rows are executed; every other row is scored by the strict GSM8K reward."""
    if is_code_source(data_source):
        return code_score(
            solution_str,
            ground_truth,
            require_closed_block=require_closed_block,
            strip_toplevel_tests=strip_toplevel_tests,
        )
    from verl.utils.reward_score import gsm8k_training_reward

    return gsm8k_training_reward.compute_score(data_source, solution_str, ground_truth, extra_info)


def compute_score_strict(data_source, solution_str, ground_truth, extra_info=None):
    """LLaDA-2.1-mini recipe semantics."""
    return compute_score(data_source, solution_str, ground_truth, extra_info, True, True)


def compute_score_lenient(data_source, solution_str, ground_truth, extra_info=None):
    """QwenDLM (SDAR) recipe semantics."""
    return compute_score(data_source, solution_str, ground_truth, extra_info, False, False)


def extract_code(solution_str, require_closed_block=True):
    if "```python" not in (solution_str or ""):
        return None
    tail = solution_str.split("```python")[-1]
    if require_closed_block and "```" not in tail:
        return None
    return tail.split("```")[0]


def _strip_toplevel_tests(code):
    """Remove top-level test scaffolding; definitions, imports and indented code are kept."""
    out, lines, i = [], code.splitlines(), 0
    while i < len(lines):
        ln = lines[i]
        stripped = ln.lstrip()
        top = (len(ln) - len(stripped)) == 0 and stripped != ""
        if top and re.match(r"if\s+__name__\s*==\s*[\"']__main__[\"']\s*:", stripped):
            i += 1
            while i < len(lines) and (lines[i].strip() == "" or lines[i][:1] in (" ", "\t")):
                i += 1
            continue
        if top and _TOPLEVEL_TEST_CALL.match(stripped):
            depth = stripped.count("(") + stripped.count("[") - stripped.count(")") - stripped.count("]")
            i += 1
            while depth > 0 and i < len(lines):  # multi-line assert/call: swallow continuation lines
                depth += lines[i].count("(") + lines[i].count("[") - lines[i].count(")") - lines[i].count("]")
                i += 1
            continue
        out.append(ln)
        i += 1
    return "\n".join(out)


def _tokens_match(got, exp):
    if got == exp:
        return True
    try:
        g, e = float(got), float(exp)
    except (TypeError, ValueError):
        return False
    return abs(g - e) <= NUM_TOL * max(1.0, abs(e))


def _stdio_match(got, exp):
    gl = [l.rstrip() for l in got.strip().splitlines()]
    el = [l.rstrip() for l in exp.strip().splitlines()]
    gl = [l for l in gl if l != ""]
    el = [l for l in el if l != ""]
    if gl == el:
        return True
    if len(gl) != len(el):
        return False
    for g, e in zip(gl, el):
        gt_, et_ = g.split(), e.split()
        if len(gt_) != len(et_):
            return False
        if not all(_tokens_match(a, b) for a, b in zip(gt_, et_)):
            return False
    return True


def _norm_json(x):
    return json.dumps(x, sort_keys=True)


def _fn_match(res, exp):
    cands = [exp]
    if isinstance(exp, list) and len(exp) == 1:
        cands.append(exp[0])
    for c in cands:
        if _norm_json(res) == _norm_json(c) or _norm_json([res]) == _norm_json(c):
            return True
        if isinstance(res, (int, float)) and isinstance(c, (int, float)):
            if _tokens_match(res, c):
                return True
    return False


def _run_child(script, stdin_text, timeout, tmpdir):
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": tmpdir}
    return subprocess.run(
        [sys.executable, "-I", "-c", script],
        input=stdin_text,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=tmpdir,
        env=env,
        start_new_session=True,
    )


def _result(correct, uncertain=False):
    score = 1.0 if correct else 0.0
    return {
        "score": score,
        "acc": score,
        "verification_status": "correct" if correct else "wrong",
        "verified_wrong": (not correct) and (not uncertain),
        "uncertain": uncertain,
    }


def code_score(solution_str, ground_truth, require_closed_block=True, strip_toplevel_tests=True):
    correct = False
    uncertain = False
    try:
        code = extract_code(solution_str, require_closed_block=require_closed_block)
        if code is not None:
            gt = ground_truth if isinstance(ground_truth, dict) else json.loads(ground_truth)
            if "functional" in gt:
                # HumanEval-style: run the assert-based check(candidate) suite once.
                if strip_toplevel_tests:
                    code = _strip_toplevel_tests(code)
                with tempfile.TemporaryDirectory() as tmpdir:
                    script = (
                        _RLIMIT_PRELUDE + code + "\n" + gt["functional"]
                        + "\ncheck(" + (gt.get("entry_point") or "None") + ")\n"
                    )
                    try:
                        p = _run_child(script, "", FUNCTIONAL_TIMEOUT, tmpdir)
                        correct = p.returncode == 0
                    except Exception:
                        correct = False
                return _result(correct)
            inputs, outputs = gt["inputs"], gt["outputs"]
            fn = gt.get("fn_name")
            t0 = time.time()
            with tempfile.TemporaryDirectory() as tmpdir:
                all_pass = len(inputs) > 0
                for inp, exp in zip(inputs, outputs):
                    if time.time() - t0 > TOTAL_BUDGET:
                        all_pass = False
                        break
                    try:
                        if fn:
                            script = _RLIMIT_PRELUDE + code + _FN_RUNNER.format(fn=fn)
                            p = _run_child(script, json.dumps(inp), PER_TEST_TIMEOUT, tmpdir)
                            ok = (p.returncode == 0) and _fn_match(json.loads(p.stdout.strip()), exp)
                        else:
                            script = _RLIMIT_PRELUDE + code
                            p = _run_child(script, inp, PER_TEST_TIMEOUT, tmpdir)
                            ok = (p.returncode == 0) and _stdio_match(p.stdout, exp)
                    except Exception:
                        ok = False
                    if not ok:
                        all_pass = False
                        break
                correct = all_pass
    except Exception:
        correct = False
        uncertain = True
    return _result(correct, uncertain)
