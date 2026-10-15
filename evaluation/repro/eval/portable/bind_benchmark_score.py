#!/usr/bin/env python3
"""Bind complete benchmark records, original gold, and an offline scorer capsule."""
import argparse
import json
from pathlib import Path
import re
import shutil
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = 'repro.eval.portable'

from .build_recipe import inventory
from .score_benchmarks import (BENCHMARKS, contained, profile_key, registry, required_files,
                               sha, validate_inputs, validate_runtime)


def rows_from(path):
    return [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def bind(args):
    if args.out.exists():
        raise ValueError('Refusing an existing score capsule')
    if not re.fullmatch(r'[^\s]+@sha256:[a-f0-9]{64}', args.scorer_image):
        raise ValueError('Supply an immutable scorer image digest')
    profile = registry()['profiles'][profile_key(args.family, args.benchmark)]
    validate_runtime(args.runtime_root, profile)
    problem_bytes = args.original_problems.read_bytes()
    if sha(problem_bytes) != args.expected_problems_sha256:
        raise ValueError('Original problem file hash mismatch')
    problems = rows_from(args.original_problems)
    ids = [r['task_id'] for r in problems]
    if len(ids) != len(set(ids)) or len(ids) != profile['full_cell_n']:
        raise ValueError('Expected one full original benchmark with unique IDs')
    if args.records_root:
        paths = sorted(args.records_root.rglob('records.jsonl'))
        if not paths:
            paths = sorted(args.records_root.rglob('shard*.jsonl'))
    else:
        paths = args.records
    if not paths:
        raise ValueError('No generation record files selected')
    rows = [row for path in paths for row in rows_from(path)]
    by_id = {r['task_id']: r for r in rows}
    if len(rows) != len(by_id) or set(by_id) != set(ids):
        raise ValueError('Incomplete or duplicate generation coverage')
    ordered = []
    for original in problems:
        row = dict(by_id[original['task_id']])
        # Native QwenDLM rows use text/metrics/elapsed, while its adapter normalizes them.
        if 'raw' not in row and 'text' in row:
            row['raw'] = row['text']
            finish = row.get('finish_reason')
            row['finish'] = finish.get('type') if isinstance(finish, dict) else finish
            row['fwd_denoise'] = row['metrics']['denoise_forwards']
            row['fwd_total'] = row['metrics']['total_forwards']
            row['wall_s'] = row['elapsed']
        if args.benchmark == 'he' and 'entry_point' not in row:
            row['entry_point'] = original.get('entry_point')
        ordered.append(row)
    data = ''.join(json.dumps(r, sort_keys=True) + '\n' for r in ordered).encode()
    manifest = {'schema_version': 1, 'family': args.family, 'benchmark': args.benchmark,
        'scope': 'full_cell', 'scorer_protocol_id': profile['protocol_id'],
        'generation_protocol_id': args.generation_protocol_id, 'evaluation_id': args.evaluation_id,
        'expected_task_ids': ids, 'records': {'path': 'records.jsonl', 'sha256': sha(data)},
        'original_problems': {'path': 'original_problems.jsonl', 'sha256': sha(problem_bytes)},
        'binding': {'original_problem_sha256': sha(problem_bytes),
            'source_generation_records': [{'name': p.name, 'sha256': sha(p.read_bytes())} for p in paths],
            'scope': 'Supplied records and original gold are hash-bound; no execution receipt or model attestation inferred',
            'historical_source_identity': profile['identity_note']}}
    inputs = args.out / 'inputs'
    runtime = args.out / 'runtime'
    inputs.mkdir(parents=True)
    runtime.mkdir()
    (inputs / 'records.jsonl').write_bytes(data)
    (inputs / 'original_problems.jsonl').write_bytes(problem_bytes)
    validate_inputs(manifest, inputs, profile)
    (inputs / 'score_inputs.json').write_text(json.dumps(manifest, indent=2))
    for name, digest in required_files(profile).items():
        source = contained(args.runtime_root, name)
        target = runtime / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if sha(target.read_bytes()) != digest:
            raise ValueError('Runtime changed during scoring capsule copy')
    for name in ('score_benchmarks.py', 'scoring_registry.json'):
        shutil.copyfile(Path(__file__).with_name(name), runtime / name)
    recipe = {'schema_version': 1, 'protocol_id': args.evaluation_id + '-score',
        'scope': 'full_cell', 'historical_binding': profile['identity_note'], 'expected_task_ids': ids,
        'artifacts': inventory(runtime, 'runtime') + inventory(inputs, 'inputs'),
        'stages': {'score': {'script': 'score_benchmarks.py', 'image': args.scorer_image,
            'platform': 'linux/amd64', 'cpus': profile.get('cpus', 2), 'args': ['--manifest', '{inputs}/score_inputs.json',
                '--input-root', '{inputs}', '--runtime-root', '{runtime}', '--out', '{output}']}}}
    (args.out / 'score_recipe.json').write_text(json.dumps(recipe, indent=2))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=('qwen-dlm', 'llada', 'dg'), required=True)
    parser.add_argument('--benchmark', choices=BENCHMARKS, required=True)
    parser.add_argument('--evaluation-id', required=True, help='Unique model/seed/run identifier')
    parser.add_argument('--generation-protocol-id', required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--records-root', type=Path)
    group.add_argument('--records', type=Path, action='append')
    for name in ('original-problems', 'runtime-root', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-problems-sha256', required=True)
    parser.add_argument('--scorer-image', required=True)
    manifest = bind(parser.parse_args(argv))
    print(json.dumps({'scorer_protocol_id': manifest['scorer_protocol_id'],
                      'n': len(manifest['expected_task_ids']), 'quality_scored': False}))


if __name__ == '__main__':
    main()
