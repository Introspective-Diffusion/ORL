#!/usr/bin/env python3
"""LiveCodeBench scoring for score_full_extra_results.py.

Follows the I-DLM reference evaluator (inference/eval/eval_lcb.py): code = last ``` block of the completion
(lcb_lib/extract_utils.extract_code_generation_v2); each problem is judged by the OpenCompass testing_util.run_test over
public + private tests (per-test timeout 6 s, all must pass) inside a fresh python subprocess with a 300 s global timeout;
EVAL_PARALLEL problems judged concurrently (default 16; the scoring sandbox sets 8). Tests come from <LCB_TESTS>, the
lcb_v6_tests.json built by repro/eval/build_transfer_inputs.py.
"""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "lcb_lib")
sys.path.insert(0, LIB)
from extract_utils import extract_code_generation_v2  # noqa: E402

_JUDGE = r'''
import sys, types, importlib.util, json, os, resource
LIB = sys.argv[1]
# Generated programs inherit this limit through testing_util subprocesses.
# Without it, adversarial/degenerate completions can OOM the scorer pod before
# the wall-time timeout has a chance to fire.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
_MEM = 8 * 1024 ** 3
resource.setrlimit(resource.RLIMIT_AS, (_MEM, _MEM))
def _imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod; spec.loader.exec_module(mod); return mod
_pyext = types.ModuleType("pyext")
class _RM:
    @staticmethod
    def from_string(name, path, c):
        mod = types.ModuleType(name); exec(c, mod.__dict__); return mod
_pyext.RuntimeModule = _RM; sys.modules["pyext"] = _pyext
tu = _imp("tu", LIB + "/testing_util.py")
data = json.loads(sys.stdin.read())
sample = {"input_output": json.dumps(data["tests"])}
try:
    results, _ = tu.run_test(sample, test=data["code"], debug=False, timeout=6)
    ok = bool(results) and all(r is True for r in results)
    print(json.dumps({"ok": ok, "reason": None if ok else "wa", "n": len(results or [])}))
except Exception as e:
    print(json.dumps({"ok": False, "reason": f"error:{type(e).__name__}"}))
'''


def judge(code, tests, timeout=300):
    if not code or not code.strip():
        return False, "empty"
    if not tests or not tests.get("inputs"):
        return False, "no_tests"
    try:
        r = subprocess.run([sys.executable, "-c", _JUDGE, LIB], input=json.dumps({"code": code, "tests": tests}),
                           capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0 or not r.stdout.strip():
            return False, "re:" + (r.stderr or "")[-120:].replace("\n", " ")
        out = json.loads(r.stdout.strip().splitlines()[-1])
        return bool(out["ok"]), out.get("reason")
    except subprocess.TimeoutExpired:
        return False, "global_timeout"
    except Exception as e:
        return False, f"error:{type(e).__name__}"


def score_lcb_rows(rows):
    tests_path = os.environ.get("LCB_TESTS")
    if not tests_path:
        raise RuntimeError("LCB_TESTS must name lcb_v6_tests.json, built by repro/eval/build_transfer_inputs.py")
    with open(tests_path) as f:
        tests = json.load(f)
    codes = [extract_code_generation_v2(r.get("raw") or "", model_type="chat") for r in rows]
    with ThreadPoolExecutor(max_workers=int(os.environ.get("EVAL_PARALLEL", "16"))) as pool:
        res = list(pool.map(lambda a: judge(a[0], tests.get(a[1]["task_id"])), zip(codes, rows)))
    out = {}
    by = {}
    for r, code, (ok, reason) in zip(rows, codes, res):
        out[r["task_id"]] = {"base": ok, "plus": ok, "reason": reason, "code_len": len(code)}
        m = r.get("meta") or {}
        k = (m.get("difficulty", "?"), "call" if m.get("fn_name") else "stdin")
        by.setdefault(k, [0, 0]); by[k][1] += 1; by[k][0] += ok
    for k in sorted(by):
        print(f"lcb {k[0]:6s} {k[1]:5s}: {by[k][0]}/{by[k][1]}")
    print(f"lcb empty_code={sum(1 for c in codes if not c.strip())} "
          f"reasons={dict(sorted(((v['reason'] or 'pass'), sum(1 for x in out.values() if x['reason'] == v['reason'])) for v in out.values()))}")
    return out
