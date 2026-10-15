#!/usr/bin/env python3
"""Bind QwenDLM HE generations to original gold and a supplied frozen scorer."""
import argparse
import json
from pathlib import Path
import re

import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = 'repro.eval.portable'

from .build_recipe import inventory
from .reproduce import contained
from .score_qwen_dlm_he import PROTOCOL, SOURCES, sha, validate_records, validate_sources
from inference.adapters import qwen_dlm


def flag(args, name):
    if args.count(name) != 1:
        raise ValueError('Missing or duplicate generation argument: ' + name)
    return args[args.index(name) + 1]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def read_rows(data):
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def check_generation(recipe_data, attempt, original_inputs):
    recipe = json.loads(recipe_data)
    generate = recipe['stages']['generate']
    args = generate['args']
    script = Path(generate['script'])
    if script.name != 'run_eval.py' or flag(args, '--script') != str(script):
        raise ValueError('Generation entry point is not the pinned runner')
    runtime_inventory = {a['path']: a['sha256'] for a in recipe['artifacts'] if a['root'] == 'runtime'}
    if len(runtime_inventory) != sum(a['root'] == 'runtime' for a in recipe['artifacts']):
        raise ValueError('Duplicate runtime inventory paths')
    for name, digest in qwen_dlm.SOURCES.items():
        if runtime_inventory.get(str(script.parent / name)) != digest:
            raise ValueError('Generation recipe does not pin original runtime sources')
    role = flag(args, '--role')
    if generate.get('driver') != 'qwen-dlm' or flag(args, '--benchmark') != 'he' or role not in ('qwen-dlm-base', 'qwen-dlm-orl'):
        raise ValueError('Expected a QwenDLM HumanEval generation recipe')
    if recipe.get('scope') != 'shard' or int(flag(args, '--seed')) != 1234:
        raise ValueError('Wrong QwenDLM generation scope or seed')
    shard = int(flag(args, '--shard-index'))
    start, n = qwen_dlm.partition('he', shard)
    if recipe['shard'] != {'index': shard, 'count': 8, 'start_q': start, 'num_q': n, 'full_cell_n': 164}:
        raise ValueError('Recipe shard declaration mismatch')
    problem_arg = flag(args, '--problems')
    if not problem_arg.startswith('{inputs}/'):
        raise ValueError('Original problems must come from the pinned input root')
    relative = problem_arg[len('{inputs}/'):]
    problem_path = contained(original_inputs, relative)
    problem_bytes = problem_path.read_bytes()
    original_record = [a for a in recipe['artifacts'] if a['root'] == 'inputs' and a['path'] == relative]
    if len(original_record) != 1 or sha(problem_bytes) != original_record[0]['sha256']:
        raise ValueError('Original problem inventory mismatch')
    problems, encoded = qwen_dlm.validate_inputs(problem_path, 'he')
    selected = problems[start:start + n]
    expected = [r['task_id'] for r in selected]
    if recipe['expected_task_ids'] != expected:
        raise ValueError('Recipe IDs differ from original frozen partition')
    receipt_data = contained(attempt, 'execution_receipt.json').read_bytes()
    receipt = json.loads(receipt_data)
    if receipt.get('recipe_sha256') != sha(recipe_data) or receipt.get('protocol_id') != recipe['protocol_id'] or receipt.get('stage') != 'generate' or receipt.get('dry_run') is not False:
        raise ValueError('Generation receipt does not bind this executed recipe')
    if type(receipt.get('exit_code')) is not int or receipt['exit_code'] != 0 or receipt.get('timed_out', False):
        raise ValueError('Generation did not finish successfully')
    artifacts = attempt / 'artifacts'
    protocol_data = contained(artifacts, 'generation_protocol.json').read_bytes()
    protocol = json.loads(protocol_data)
    task = {'id': f"{'sft' if role == 'qwen-dlm-base' else 'rct'}-humaneval-s{start:04d}",
            'entity': 'sft' if role == 'qwen-dlm-base' else 'rct', 'benchmark': 'humaneval',
            'dataset_n': 164, 'max_new_tokens': 32768, 'seed': 1234,
            'shard_index': shard, 'start_q': start, 'num_q': n}
    if protocol['task'] != task or protocol['decoder'] != qwen_dlm.decoder(role, 'he') or protocol['input_task_ids'] != expected:
        raise ValueError('Generation protocol differs from frozen recipe')
    if protocol['source_sha256'] != qwen_dlm.SOURCES or protocol['harness_commit'] != qwen_dlm.HARNESS_COMMIT or protocol['engine_commit'] != qwen_dlm.ENGINE_COMMIT:
        raise ValueError('Generation source identity mismatch')
    if protocol['inputs_sha256'] != sha(encoded) or protocol.get('scoring_status') != 'not_run':
        raise ValueError('Wrong generation input digest or scoring status')
    core = {key: protocol[key] for key in ('task', 'decoder', 'inputs_sha256', 'harness_commit', 'engine_commit', 'scoring_status')}
    if sha(canonical(core)) != protocol['settings_sha256']:
        raise ValueError('Generation settings digest mismatch')
    for name, digest in protocol['patched_source_sha256'].items():
        if sha(contained(artifacts, 'generation_runtime/' + name).read_bytes()) != digest:
            raise ValueError('Patched generation source digest mismatch')
    if set(protocol['patched_source_sha256']) != set(qwen_dlm.SOURCES):
        raise ValueError('Missing patched generation source')
    complete_data = contained(artifacts, 'GENERATION_COMPLETE.json').read_bytes()
    complete = json.loads(complete_data)
    if complete != {'scope': 'shard', 'n': n, 'settings_sha256': protocol['settings_sha256'], 'scoring_run': False}:
        raise ValueError('Missing complete unscored shard marker')
    records_data = contained(artifacts, 'records.jsonl').read_bytes()
    records = read_rows(records_data)
    native_data = contained(artifacts, f'humaneval/humaneval_s{start:04d}.json').read_bytes()
    native = json.loads(native_data)['results']
    if [r['task_id'] for r in records] != expected or [r['task_id'] for r in native] != expected:
        raise ValueError('Wrong/duplicate normalized or native task IDs')
    for row, original, raw in zip(records, selected, native):
        if row['gold'] != original['gold'] or raw['gold'] != original['gold'] or raw['problem'] != original['prompt']:
            raise ValueError('Gold or prompt drift from original frozen problem')
        if raw['idx'] != original['idx'] or raw['correct'] is not None or raw['scoring_status'] != 'not_run':
            raise ValueError('Native generation identity or scoring drift')
        if row['generation_settings_sha256'] != protocol['settings_sha256'] or raw['generation_settings_sha256'] != protocol['settings_sha256']:
            raise ValueError('Response does not bind generation settings')
        normalized = {'task_id': raw['task_id'], 'raw': raw['text'], 'gold': raw['gold'],
                      'finish': raw['finish_reason']['type'], 'completion_tokens': raw['completion_tokens'],
                      'fwd_denoise': raw['metrics']['denoise_forwards'], 'fwd_total': raw['metrics']['total_forwards'],
                      'wall_s': raw['elapsed'], 'generation_settings_sha256': protocol['settings_sha256']}
        if row != normalized:
            raise ValueError('Normalized records differ from native generation')
    model_inventory = sorted((a['path'], a['sha256']) for a in recipe['artifacts'] if a['root'] == 'model')
    if not model_inventory:
        raise ValueError('Generation recipe has no model inventory')
    binding = {'shard_index': shard, 'generation_recipe_sha256': sha(recipe_data),
               'execution_receipt_sha256': sha(receipt_data), 'generation_protocol_sha256': sha(protocol_data),
               'completion_marker_sha256': sha(complete_data), 'native_rows_sha256': sha(native_data),
               'normalized_records_sha256': sha(records_data), 'original_problem_sha256': sha(problem_bytes),
               'runtime_inventory_sha256': sha(canonical(sorted(runtime_inventory.items()))),
               'model_inventory_sha256': sha(canonical(model_inventory))}
    evidence = {'generation_recipe.json': recipe_data, 'execution_receipt.json': receipt_data,
                'generation_protocol.json': protocol_data, 'GENERATION_COMPLETE.json': complete_data}
    return role, shard, records, selected, binding, evidence


def bind(args):
    if len(args.generation_recipe) != len(args.generation_root) or not args.generation_recipe:
        raise ValueError('Each generation recipe requires its attempt root')
    if args.out.exists():
        raise ValueError('Refusing an existing bound scoring directory')
    if not re.fullmatch(r'[^\s]+@sha256:[a-f0-9]{64}', args.scorer_image):
        raise ValueError('Supply an immutable scorer image digest')
    validate_sources(args.runtime_root)
    checked = [check_generation(path.read_bytes(), root, args.original_input_root)
               for path, root in zip(args.generation_recipe, args.generation_root)]
    roles = {row[0] for row in checked}
    shards = [row[1] for row in checked]
    if len(roles) != 1 or len(shards) != len(set(shards)):
        raise ValueError('Mixed roles or duplicated generation shards')
    for key in ('original_problem_sha256', 'model_inventory_sha256', 'runtime_inventory_sha256'):
        if len({row[4][key] for row in checked}) != 1:
            raise ValueError('Mixed generation input/model identity')
    checked.sort(key=lambda row: row[1])
    rows = [r for entry in checked for r in entry[2]]
    gold = [{'task_id': r['task_id'], 'gold': r['gold']} for entry in checked for r in entry[3]]
    role = next(iter(roles))
    scope = 'full_cell' if len(shards) == 8 else 'shard' if len(shards) == 1 else 'partial_shard_set'
    output, runtime = args.out / 'inputs', args.out / 'runtime'
    output.mkdir(parents=True)
    runtime.mkdir()
    for name, digest in SOURCES.items():
        target = runtime / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(contained(args.runtime_root, name).read_bytes())
        if sha(target.read_bytes()) != digest:
            raise ValueError('Runtime changed while copying')
    (runtime / 'score_qwen_dlm_he.py').write_bytes(Path(__file__).with_name('score_qwen_dlm_he.py').read_bytes())
    records = {}
    for name, values in (('records.jsonl', rows), ('gold.jsonl', gold)):
        data = ''.join(json.dumps(row, sort_keys=True) + '\n' for row in values).encode()
        (output / name).write_bytes(data)
        records[name] = {'path': name, 'sha256': sha(data)}
    for _, shard, _, _, _, evidence in checked:
        directory = output / 'evidence' / f'shard{shard}'
        directory.mkdir(parents=True)
        for name, data in evidence.items():
            (directory / name).write_bytes(data)
    manifest = {'schema_version': 1, 'scorer_protocol_id': PROTOCOL, 'scope': scope, 'full_cell_n': 164,
                'evaluation_id': role + '-humaneval-' + '-'.join(map(str, sorted(shards))),
                'generation_protocol_id': role + '-humaneval-qwen-dlm-v1-server1234-noise12345',
                'expected_task_ids': [r['task_id'] for r in rows], 'records': records['records.jsonl'], 'gold': records['gold.jsonl'],
                'binding': {'shards': [r[4] for r in checked], 'source_sha256': SOURCES,
                            'receipt_scope': 'Hash-bound supplied execution evidence; not remote hardware attestation'}}
    validate_records(manifest, output)
    (output / 'score_inputs.json').write_text(json.dumps(manifest, indent=2) + '\n')
    recipe = {'schema_version': 1, 'protocol_id': manifest['evaluation_id'] + '-score', 'scope': scope,
              'historical_binding': 'Exact supplied scorer source; original gold and generation evidence hash-bound',
              'expected_task_ids': manifest['expected_task_ids'],
              'artifacts': inventory(runtime, 'runtime') + inventory(output, 'inputs'),
              'stages': {'score': {'script': 'score_qwen_dlm_he.py', 'image': args.scorer_image,
                         'args': ['--manifest', '{inputs}/score_inputs.json', '--input-root', '{inputs}',
                                  '--runtime-root', '{runtime}', '--out', '{output}']}}}
    (args.out / 'score_recipe.json').write_text(json.dumps(recipe, indent=2) + '\n')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation-recipe', type=Path, action='append', required=True)
    parser.add_argument('--generation-root', type=Path, action='append', required=True, help='Attempt root containing execution_receipt.json and artifacts/')
    parser.add_argument('--original-input-root', type=Path, required=True)
    parser.add_argument('--runtime-root', type=Path, required=True, help='Supplied historical scorer source tree; never executed by binding')
    parser.add_argument('--scorer-image', required=True)
    parser.add_argument('--out', type=Path, required=True)
    manifest = bind(parser.parse_args(argv))
    print(json.dumps({'scope': manifest['scope'], 'n': len(manifest['expected_task_ids']), 'quality_scored': False}))


if __name__ == '__main__':
    main()
