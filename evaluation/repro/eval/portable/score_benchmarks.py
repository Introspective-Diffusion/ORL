#!/usr/bin/env python3
"""Versioned main/transfer scoring. Execute only in the isolated score stage."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import sys

BENCHMARKS = ('gsm8k', 'he', 'math500', 'mbpp', 'aime', 'gpqa', 'mmlu_pro', 'lcb', 'ifeval')
METRICS = ('completion_tokens', 'fwd_denoise', 'fwd_total', 'wall_s')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def profile_key(family, benchmark):
    if family not in ('qwen-dlm', 'llada', 'dg') or benchmark not in BENCHMARKS:
        raise ValueError('Unknown family/benchmark')
    if benchmark in BENCHMARKS[4:]:
        if family == 'qwen-dlm':
            raise ValueError('No QwenDLM transfer protocol was evaluated')
        return 'transfer.' + benchmark
    if benchmark == 'mbpp':
        return 'main.mbpp'
    return ('qwen-dlm' if family == 'qwen-dlm' else 'paper4') + '.' + benchmark


def registry():
    return json.loads(Path(__file__).with_name('scoring_registry.json').read_text())


def contained(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Asset must have a contained relative path')
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('Missing/escaping scoring asset: ' + str(relative))
    return path


def read_asset(root, record):
    path = contained(root, record['path'])
    data = path.read_bytes()
    if sha(data) != record['sha256']:
        raise ValueError('Scoring asset digest mismatch: ' + record['path'])
    return data


def required_files(profile):
    return {key: value for group in ('source_files', 'data_files', 'wheels')
            for key, value in profile[group].items()}


def validate_runtime(root, profile):
    for path, digest in required_files(profile).items():
        read_asset(root, {'path': path, 'sha256': digest})


def validate_inputs(manifest, root, profile):
    if manifest.get('schema_version') != 1 or manifest.get('scorer_protocol_id') != profile['protocol_id']:
        raise ValueError('Wrong score manifest protocol')
    expected = manifest['expected_task_ids']
    if not expected or len(expected) != len(set(expected)) or any(not isinstance(i, str) or not i for i in expected):
        raise ValueError('Invalid expected task IDs')
    if len(expected) != profile['full_cell_n'] or manifest.get('scope') != 'full_cell':
        raise ValueError('This scorer requires one complete benchmark run')
    rows = [json.loads(line) for line in read_asset(root, manifest['records']).splitlines() if line.strip()]
    gold_rows = [json.loads(line) for line in read_asset(root, manifest['original_problems']).splitlines() if line.strip()]
    gold = {r['task_id']: r for r in gold_rows}
    ids = [r['task_id'] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(expected) or len(gold_rows) != len(gold) or set(gold) != set(expected):
        raise ValueError('Record/original problem coverage mismatch')
    benchmark = manifest['benchmark']
    for row in rows:
        original = gold[row['task_id']]
        if row.get('gold') != original.get('gold'):
            raise ValueError('Original gold drift')
        if benchmark in ('aime', 'gpqa', 'mmlu_pro', 'lcb', 'ifeval') and row.get('meta', {}) != original.get('meta', {}):
            raise ValueError('Original benchmark metadata drift')
        if not isinstance(row.get('raw'), str) or row.get('finish') not in ('stop', 'length'):
            raise ValueError('Invalid normalized response/finish')
        for key in METRICS:
            value = row.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError('Invalid numeric metric: ' + key)
            if row['finish'] == 'stop' and key != 'completion_tokens' and value <= 0:
                raise ValueError('Nonpositive clean denominator')
        if benchmark == 'he' and row.get('entry_point') != original.get('entry_point'):
            raise ValueError('HumanEval entry-point drift')
    if benchmark == 'aime':
        years = [r.get('meta', {}).get('year') for r in rows]
        if years.count('aime24') != 30 or years.count('aime25') != 30:
            raise ValueError('AIME requires exactly 30 tasks per year')
    by_id = {r['task_id']: r for r in rows}
    return [by_id[i] for i in expected]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def normalize(value):
    value = str(value).strip().replace(',', '').replace('$', '')
    try:
        number = float(value)
        return str(int(number)) if number == int(number) else str(number)
    except (ValueError, OverflowError):
        return value


def gsm_answer(text):
    # Exact paper4 parser from the independently verified main-table scorer.
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text or '')
    if boxed:
        return normalize(boxed[-1].replace('\\', ''))
    visible = text.split('</think>')[-1] if '</think>' in text else text
    numbers = re.findall(r'[-+]?\d[\d,.]*', visible)
    return normalize(numbers[-1]) if numbers else '?'


def prepare_dependencies(runtime, profile, out):
    wheels = [str(contained(runtime, name)) for name in sorted(profile['wheels'])]
    if not wheels:
        return {}
    site = out / 'site'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
                    '--target', str(site), *wheels], check=True)
    sys.path.insert(0, str(site))
    os.environ['PYTHONPATH'] = str(site)
    return {d.metadata['Name']: d.version for d in importlib.metadata.distributions(path=[str(site)])}


def score_rows(rows, family, benchmark, runtime, out, loader=load_module):
    extra = {}
    if benchmark in BENCHMARKS[4:]:
        source = runtime / 'transfer_repo/score_full_extra_results.py'
        scorer = loader('portable_historical_transfer', source)
        ordered = sorted(rows, key=lambda r: r['task_id'])
        if benchmark in ('aime', 'gpqa', 'mmlu_pro'):
            judge = scorer.aime_correct if benchmark == 'aime' else scorer.mc_correct
            values = {r['task_id']: bool(judge(r)) for r in ordered}
        elif benchmark == 'ifeval':
            sys.path.insert(0, str(runtime / 'transfer_repo'))
            os.environ['NLTK_DATA'] = str(runtime / 'transfer_data/nltk_data')
            from langdetect import DetectorFactory
            random.seed(0)
            DetectorFactory.seed = 0
            values, extra = scorer.ifeval_scores(ordered)
        else:
            os.environ['EVAL_PARALLEL'] = '8'
            os.environ['OPENBLAS_NUM_THREADS'] = '1'
            values = scorer.lcb_scores(ordered, runtime / 'transfer_repo', runtime / 'transfer_data/lcb_v6_tests.json')
    elif benchmark == 'gsm8k' and family != 'qwen-dlm':
        values = {r['task_id']: r['finish'] != 'length' and gsm_answer(r['raw']) == normalize(r['gold']) for r in rows}
    elif benchmark == 'math500' and family != 'qwen-dlm':
        scorer = loader('portable_historical_math', runtime / 'repo/training/diffusion-reasoning-dapo-latent/verl/verl/utils/reward_score/math_verify.py')
        values = {r['task_id']: r['finish'] != 'length' and bool(scorer.compute_score(r['raw'], r['gold'])) for r in rows}
    elif benchmark == 'mbpp':
        scorer = loader('portable_historical_mbpp', runtime / 'repo/repro/experiments/paper_repro_20260917/score_mbpp427_our_extractor.py')
        values = dict(scorer.score_one((str(runtime / 'repo'), row)) for row in rows)
    elif benchmark == 'he' and family != 'qwen-dlm':
        os.environ.update(HUMANEVAL_OVERRIDE_PATH=str(runtime / 'he_data/HumanEvalPlus-v0.1.10.jsonl'),
                          XDG_CACHE_HOME=str(out / 'he_cache'), EVAL_PARALLEL='4')
        from evalplus.data import get_human_eval_plus_hash
        if get_human_eval_plus_hash() != 'fe585eb4df8c88d844eeb463ea4d0302':
            raise ValueError('HumanEvalPlus dataset identity mismatch')
        scorer = loader('portable_historical_he', runtime / 'he_repo/repro/experiments/diffusion_gemma_aec_extra_eval_20260913/dgemma_core_score_point.py')
        work = out / 'he_work'
        work.mkdir()
        scorer.score_code(rows, 'humaneval', work)
        evaluated = json.loads((work / 'samples-sanitized_eval_results.json').read_text())['eval']
        values = {r['task_id']: r['finish'] != 'length' and evaluated[r['task_id']][0]['base_status'] == 'pass' for r in rows}
    else:
        scorer = loader('portable_historical_benchmarks', runtime / 'repo/repro/eval/benchmarks.py')
        judge = getattr(scorer, {'gsm8k': 'GSM8K', 'math500': 'MATH500', 'he': 'HumanEval'}[benchmark])()
        values = {r['task_id']: r['finish'] != 'length' and bool(judge.score(r['raw'], {'gold': r['gold']})) for r in rows}
    if set(values) != {r['task_id'] for r in rows} or any(type(v) is not bool for v in values.values()):
        raise ValueError('Original scorer returned invalid verdict coverage/types')
    if any(values[r['task_id']] for r in rows if r['finish'] == 'length'):
        raise ValueError('Collapsed response unexpectedly received credit')
    return values, extra


def emit(manifest, rows, values, extra, out, profile, dependencies):
    evaluations = []
    groups = ('aime24', 'aime25') if manifest['benchmark'] == 'aime' else (manifest['benchmark'],)
    outputs = {}
    for benchmark in groups:
        chosen = [r for r in rows if manifest['benchmark'] != 'aime' or r['meta']['year'] == benchmark]
        verdicts = [{'task_id': r['task_id'], 'correct': values[r['task_id']]} for r in chosen]
        metrics = [{key: r[key] for key in ('task_id', 'finish', *METRICS)} for r in chosen]
        files = {}
        for kind, records in (('verdicts', verdicts), ('metrics', metrics)):
            name = benchmark + '_' + kind + '.jsonl'
            path = out / name
            if path.exists(): raise ValueError('Refusing existing scorer output')
            data = ''.join(json.dumps(r, sort_keys=True) + '\n' for r in records).encode()
            path.write_bytes(data)
            files[kind] = {'path': name, 'sha256': sha(data)}
        outputs[benchmark] = files
        evaluations.append({'id': manifest['evaluation_id'] + '.' + benchmark,
            'protocol_id': manifest['generation_protocol_id'], 'scope': 'full_cell', 'full_cell_n': len(chosen),
            'expected_task_ids': [r['task_id'] for r in chosen], 'generation_shards': [files['metrics']],
            'verdicts': dict(files['verdicts'], scorer_protocol_id=profile['protocol_id'])})
    (out / 'aggregation_manifest.json').write_text(json.dumps({'schema_version': 1, 'evaluations': evaluations}, indent=2))
    (out / 'scoring_receipt.json').write_text(json.dumps({'scorer_protocol_id': profile['protocol_id'],
        'family': manifest['family'], 'benchmark': manifest['benchmark'], 'n': len(rows), 'correct': sum(values.values()),
        'source_and_dependency_files': required_files(profile), 'dependencies': dependencies,
        'python': platform.python_version(), 'extra_metrics': extra, 'binding': manifest['binding'],
        'identity_note': profile['identity_note'], 'outputs': outputs,
        'isolation_note': 'Run through reviewed score stage; this receipt alone is not sandbox attestation'}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'input-root', 'runtime-root', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(argv)
    if sys.version_info[:2] != (3, 11) or sys.platform != 'linux' or os.geteuid() != 65534:
        raise ValueError('Use the isolated Linux Python3.11 score stage as UID65534; do not execute on host')
    if args.runtime_root.resolve() != Path('/runtime') or args.input_root.resolve() != Path('/inputs'):
        raise ValueError('Use the reviewed /runtime and /inputs isolated mounts')
    manifest = json.loads(args.manifest.read_text())
    key = profile_key(manifest['family'], manifest['benchmark'])
    profile = registry()['profiles'][key]
    validate_runtime(args.runtime_root, profile)
    rows = validate_inputs(manifest, args.input_root, profile)
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()): raise ValueError('Scoring requires a fresh output directory')
    os.environ.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', OPENBLAS_NUM_THREADS='1')
    dependencies = prepare_dependencies(args.runtime_root, profile, args.out)
    values, extra = score_rows(rows, manifest['family'], manifest['benchmark'], args.runtime_root, args.out)
    emit(manifest, rows, values, extra, args.out, profile, dependencies)


if __name__ == '__main__':
    main()
