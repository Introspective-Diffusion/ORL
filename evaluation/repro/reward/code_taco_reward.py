"""Binary code-execution training reward for TACO-style problems (coding wave-1).

Self-contained subprocess harness — no pyext/prime_code dependency (the vendored
prime_code is import-broken in the trainer venv: pyext is unbuildable on py3.12+).
Each sampled test runs the extracted ```python block in a fresh `python -I` child
(process isolation, per-test wall timeout, child-side rlimits), stdio and fn_name
modes, numeric-tolerant output compare. Strict: score 1.0 iff ALL sampled tests pass.

The return key set MUST stay identical to gsm8k_training_reward.compute_score —
ROLLOUT_QUALITY_MODE=correctness_only consumes verification_status/verified_wrong/
uncertain, and DataProto.concat asserts equal key sets across a batch.
"""
import json
import os
import re
import subprocess
import sys
import tempfile

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


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    ds = (data_source or "").lower()
    if any(k in ds for k in CODE_SOURCE_KEYS):
        return _code_score(solution_str, ground_truth)
    from verl.utils.reward_score import gsm8k_training_reward
    return gsm8k_training_reward.compute_score(data_source, solution_str, ground_truth, extra_info)


def _extract_code(solution_str):
    """Last ```python block, and it MUST be closed (2026-09-06).

    The 09-05 code_l0 wave collapsed into "write the function, then repeat `assert ...` until the
    2048-token cap" outputs; an unclosed block used to be accepted up to end-of-text and, whenever the
    echoed asserts happened to hold, rewarded. No closing fence -> no code -> score 0 -> filtered out.
    """
    if "```python" not in (solution_str or ""):
        return None
    tail = solution_str.split("```python")[-1]
    if "```" not in tail:
        return None
    return tail.split("```")[0]


_TOPLEVEL_TEST_CALL = re.compile(
    r"^(assert\b|check\s*\(|test\w*\s*\(|unittest\.main\s*\(|doctest\.testmod\s*\(|pytest\.main\s*\()"
)


def _strip_toplevel_tests(code):
    """Drop top-level test scaffolding from the extracted solution before it is executed
    (2026-09-06). Functional-mode rewards only; stdio-mode programs keep every line.

    The training prompts show one example assert and the model tends to echo it (and to invent
    more). Executed as part of the solution, a single hallucinated expected value fails an otherwise
    correct function, and copying tests is rewarded/penalised by accident. Removed: column-0
    `assert ...` statements (with their paren-continued lines), column-0 calls to check()/test*()/
    unittest.main()/doctest.testmod()/pytest.main(), and `if __name__ == "__main__":` blocks with their
    indented bodies. Definitions, imports and everything indented are untouched.
    """
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
        input=stdin_text, capture_output=True, text=True,
        timeout=timeout, cwd=tmpdir, env=env, start_new_session=True,
    )


def _code_score(solution_str, ground_truth):
    import time
    correct = False
    uncertain = False
    try:
        code = _extract_code(solution_str)
        if code is not None:
            gt = ground_truth if isinstance(ground_truth, dict) else json.loads(ground_truth)
            if "functional" in gt:
                # HumanEval-style: run the assert-based check(candidate) suite once.
                code = _strip_toplevel_tests(code)
                with tempfile.TemporaryDirectory() as tmpdir:
                    script = (_RLIMIT_PRELUDE + code + "\n" + gt["functional"]
                              + "\ncheck(" + (gt.get("entry_point") or "None") + ")\n")
                    try:
                        p = _run_child(script, "", FUNCTIONAL_TIMEOUT, tmpdir)
                        correct = p.returncode == 0
                    except Exception:
                        correct = False
                score = 1.0 if correct else 0.0
                return {
                    "score": score,
                    "acc": score,
                    "verification_status": "correct" if correct else "wrong",
                    "verified_wrong": not correct,
                    "uncertain": False,
                }
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
    score = 1.0 if correct else 0.0
    return {
        "score": score,
        "acc": score,
        "verification_status": "correct" if correct else "wrong",
        "verified_wrong": (not correct) and (not uncertain),
        "uncertain": uncertain,
    }
