#!/usr/bin/env python3
"""QwenDLM HumanEval adapter. Run model code only through an isolated score stage."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys

PROTOCOL = 'qwen-dlm-humaneval-codetaco-f2230a5c-py311-v1'
SOURCES = {
    'repro/eval/benchmarks.py': '5240fa8858b6890b81dd94014515d2bff9366d32aed4b3cf8922c20ba8b9a72a',
    'repro/reward/code_taco_reward.py': 'b1681c441f27d187108261144d8c8a9e8637e3ff65c8eb2ea20f1b597d5bb581',
}
METRICS = ('completion_tokens', 'fwd_denoise', 'fwd_total', 'wall_s')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_asset(root, record):
    relative = Path(record['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Artifact path must be relative and contained')
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('Missing or escaping artifact')
    data = path.read_bytes()
    if sha(data) != record['sha256']:
        raise ValueError('Artifact digest mismatch: ' + str(relative))
    return data


def validate_records(manifest, root):
    if manifest.get('schema_version') != 1 or manifest.get('scorer_protocol_id') != PROTOCOL:
        raise ValueError('Wrong scorer protocol')
    expected = manifest['expected_task_ids']
    if not expected or len(expected) > 164 or any(not isinstance(value, str) or not value for value in expected) or len(expected) != len(set(expected)):
        raise ValueError('Invalid expected task IDs')
    if manifest['full_cell_n'] != 164 or manifest['scope'] not in ('shard', 'partial_shard_set', 'full_cell'):
        raise ValueError('Invalid HumanEval coverage declaration')
    if (manifest['scope'] == 'full_cell') != (len(expected) == 164):
        raise ValueError('Scope does not match full-cell coverage')
    rows = [json.loads(line) for line in read_asset(root, manifest['records']).splitlines() if line.strip()]
    gold_rows = [json.loads(line) for line in read_asset(root, manifest['gold']).splitlines() if line.strip()]
    gold = {r['task_id']: r['gold'] for r in gold_rows}
    ids = [r['task_id'] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(expected) or len(gold_rows) != len(gold) or set(gold) != set(expected):
        raise ValueError('Wrong or duplicate record/gold IDs')
    for row in rows:
        reference = gold[row['task_id']]
        if row.get('gold') != reference:
            raise ValueError('Gold differs from original frozen problem')
        if not isinstance(reference, dict) or not isinstance(reference.get('functional'), str) or not reference['functional']:
            raise ValueError('Missing original functional tests')
        if not isinstance(reference.get('entry_point'), str) or not reference['entry_point'].isidentifier():
            raise ValueError('Invalid original entry point')
        if not isinstance(row.get('raw'), str) or row.get('finish') not in ('stop', 'length'):
            raise ValueError('Invalid response/finish schema')
        for key in METRICS:
            value = row.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError('Invalid numeric field: ' + key)
            if row['finish'] == 'stop' and key != 'completion_tokens' and value <= 0:
                raise ValueError('Nonpositive clean denominator')
    by_id = {row['task_id']: row for row in rows}
    return [by_id[task] for task in expected]


def validate_sources(runtime):
    for path, expected in SOURCES.items():
        read_asset(runtime, {'path': path, 'sha256': expected})


def score_records(rows, judge):
    verdicts = []
    for row in rows:
        correct = False if row['finish'] == 'length' else judge(row['raw'], {'gold': row['gold']})
        if type(correct) is not bool:
            raise ValueError('Original scorer must return Boolean correctness')
        verdicts.append({'task_id': row['task_id'], 'correct': correct})
    return verdicts


def emit(manifest, rows, verdicts, out, manifest_sha):
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('Refusing an occupied scorer output directory')
    expected = manifest['expected_task_ids']
    if [r['task_id'] for r in verdicts] != expected or any(type(r.get('correct')) is not bool for r in verdicts):
        raise ValueError('Invalid verdict coverage or types')
    metric_rows = [{key: row[key] for key in ('task_id', 'finish', *METRICS)} for row in rows]
    files = {}
    for name, values in (('verdicts.jsonl', verdicts), ('metric_records.jsonl', metric_rows)):
        data = (''.join(json.dumps(row, sort_keys=True) + '\n' for row in values)).encode()
        (out / name).write_bytes(data)
        files[name] = {'path': name, 'sha256': sha(data)}
    evaluation = {'id': manifest['evaluation_id'], 'protocol_id': manifest['generation_protocol_id'],
                  'scope': manifest['scope'], 'full_cell_n': 164,
                  'expected_task_ids': expected, 'generation_shards': [files['metric_records.jsonl']],
                  'verdicts': {**files['verdicts.jsonl'], 'scorer_protocol_id': PROTOCOL}}
    (out / 'aggregation_manifest.json').write_text(json.dumps({'schema_version': 1, 'evaluations': [evaluation]}, indent=2) + '\n')
    receipt = {'scorer_protocol_id': PROTOCOL, 'score_input_manifest_sha256': manifest_sha,
               'python': platform.python_version(), 'source_sha256': SOURCES,
               'adapter_sha256': sha(Path(__file__).read_bytes()), 'scope': manifest['scope'],
               'n': len(rows), 'correct': sum(v['correct'] for v in verdicts),
               'binding': manifest['binding'], 'outputs': files,
               'isolation_note': 'Caller must use the reviewed isolated container stage; this receipt is not sandbox attestation'}
    (out / 'scoring_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--runtime-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if sys.version_info[:2] != (3, 11):
        raise ValueError('This frozen scorer protocol requires Python3.11')
    payload = args.manifest.read_bytes()
    manifest = json.loads(payload)
    rows = validate_records(manifest, args.input_root)
    validate_sources(args.runtime_root)
    source = args.runtime_root / 'repro/eval/benchmarks.py'
    spec = importlib.util.spec_from_file_location('frozen_qwen_dlm_he_benchmarks', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    verdicts = score_records(rows, module.HumanEval().score)
    emit(manifest, rows, verdicts, args.out, sha(payload))


if __name__ == '__main__':
    main()
