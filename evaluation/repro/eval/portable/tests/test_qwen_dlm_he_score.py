"""Trusted fixture-only tests: no historical scorer or model code runs here."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from repro.eval.portable import bind_qwen_dlm_he_score as binder
from repro.eval.portable.build_recipe import build
from repro.eval.portable import evaluate
from repro.eval.portable import score_qwen_dlm_he as score
from inference.adapters import qwen_dlm
from repro.eval.portable.tests import test_qwen_dlm_generate


class ScoringBridgeTests(unittest.TestCase):
    def fixture(self, root, shards=(0,)):
        args, generation_hashes, _ = test_qwen_dlm_generate.QwenDlmTests().fixture(root)
        problems = [json.loads(line) for line in args.problems.read_text().splitlines()]
        for item in problems:
            item['gold'] = {'functional': 'inert original test fixture', 'entry_point': 'fixture_fn'}
        data = ''.join(json.dumps(row, sort_keys=True, ensure_ascii=True) + '\n' for row in problems).encode()
        args.problems.write_bytes(data)
        source_hashes = {}
        for name in score.SOURCES:
            path = args.runtime / name
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = b"raise RuntimeError('Historical scorer must never execute on host')\n"
            path.write_bytes(payload)
            source_hashes[name] = score.sha(payload)
        for context in (patch.dict(qwen_dlm.SOURCES, generation_hashes), patch.dict(qwen_dlm.INPUT_SHA, he=score.sha(data)),
                        patch.dict(score.SOURCES, source_hashes)):
            context.start()
            self.addCleanup(context.stop)
        recipes, attempts = [], []
        for shard in shards:
            attempt = root / ('attempt' + str(shard))
            attempt.mkdir()
            recipe = build(argparse.Namespace(runtime_root=args.runtime, input_root=args.problems.parent,
                model_root=args.model, runtime_script='run_eval.py', score_script='score.py',
                decoder_config='decoder.yaml', problems='he.jsonl', benchmark='he', seed=1234,
                smoke=False, role='qwen-dlm-base', engine_subdir='engine', shard_index=shard,
                score_args_json='[]', scorer_image='fixture@sha256:' + 'a' * 64))
            recipe_path = root / ('recipe' + str(shard) + '.json')
            recipe_path.write_text(json.dumps(recipe))
            args.out = attempt / 'artifacts'
            args.shard_index = shard
            command, environment, settings = qwen_dlm.prepare(args)
            # RUNNER is a fixed trusted fixture, not a generated response or historical scorer.
            subprocess.run(command, env=environment, check=True, capture_output=True)
            qwen_dlm.verify_output(args.out, settings)
            receipt = {'protocol_id': recipe['protocol_id'], 'recipe_sha256': score.sha(recipe_path.read_bytes()),
                       'stage': 'generate', 'dry_run': False, 'exit_code': 0}
            (attempt / 'execution_receipt.json').write_text(json.dumps(receipt))
            recipes.append(recipe_path)
            attempts.append(attempt)
        return argparse.Namespace(generation_recipe=recipes, generation_root=attempts,
            original_input_root=args.problems.parent, runtime_root=args.runtime,
            scorer_image='fixture@sha256:' + 'a' * 64, out=root / 'bound')

    def test_partial_bridge_never_imports_original_scorer(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.fixture(Path(temp))
            manifest = binder.bind(args)
            self.assertEqual(manifest['scope'], 'shard')
            self.assertEqual(len(manifest['expected_task_ids']), 21)
            rows = score.validate_records(manifest, args.out / 'inputs')
            rows[0]['finish'] = 'length'
            verdicts = score.score_records(rows, lambda text, item: text == 'inert trusted fixture')
            self.assertFalse(verdicts[0]['correct'])
            out = Path(temp) / 'fixture_verdicts'
            score.emit(manifest, rows, verdicts, out, score.sha(b'fixture manifest'))
            aggregation = json.loads((out / 'aggregation_manifest.json').read_text())
            result = evaluate.audit(aggregation, out, aggregate=True)['results'][0]
            self.assertEqual(result['correct'], 20)
            self.assertTrue(result['complete_task_coverage'])
            self.assertFalse(result['complete_benchmark_coverage'])
            self.assertEqual(result['full_cell_n'], 164)
            self.assertNotIn('raw', (out / 'metric_records.jsonl').read_text())
            with self.assertRaisesRegex(ValueError, 'occupied'):
                score.emit(manifest, rows, verdicts, out, 'fixture')

    def test_full_eight_shard_bridge(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.fixture(Path(temp), range(8))
            manifest = binder.bind(args)
            self.assertEqual(manifest['scope'], 'full_cell')
            rows = score.validate_records(manifest, args.out / 'inputs')
            self.assertEqual(len(rows), 164)
            out = Path(temp) / 'fixture_verdicts'
            score.emit(manifest, rows, score.score_records(rows, lambda *_: True), out, 'fixture')
            result = evaluate.audit(json.loads((out / 'aggregation_manifest.json').read_text()), out, True)['results'][0]
            self.assertTrue(result['complete_benchmark_coverage'])
            self.assertEqual(result['correct'], 164)

    def test_mutated_generation_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.fixture(Path(temp))
            attempt = args.generation_root[0]
            targets = [
                ('execution_receipt.json', lambda v: v.update(exit_code=True)),
                ('execution_receipt.json', lambda v: v.update(recipe_sha256='0' * 64)),
                ('execution_receipt.json', lambda v: v.update(protocol_id='wrong')),
                ('artifacts/GENERATION_COMPLETE.json', lambda v: v.update(n=20)),
                ('artifacts/humaneval/humaneval_s0000.json', lambda v: v['results'][0].update(text='changed')),
                ('artifacts/humaneval/humaneval_s0000.json', lambda v: v['results'][0].update(gold={'functional': 'wrong'})),
            ]
            for relative, mutate in targets:
                with self.subTest(relative=relative, mutate=mutate):
                    path = attempt / relative
                    original = path.read_bytes()
                    value = json.loads(original)
                    mutate(value)
                    path.write_text(json.dumps(value))
                    with self.assertRaises(ValueError):
                        binder.bind(args)
                    path.write_bytes(original)
            args.generation_recipe *= 2
            args.generation_root *= 2
            with self.assertRaisesRegex(ValueError, 'duplicated'):
                binder.bind(args)

    def test_input_source_runtime_and_record_invariants(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.fixture(Path(temp))
            source = args.runtime_root / next(iter(score.SOURCES))
            original = source.read_bytes()
            source.write_bytes(original + b'# changed\n')
            with self.assertRaisesRegex(ValueError, 'digest'):
                binder.bind(args)
            source.write_bytes(original)
            recipe_path = args.generation_recipe[0]
            original_recipe = recipe_path.read_bytes()
            changed = json.loads(original_recipe)
            next(a for a in changed['artifacts'] if a['path'] == 'run_eval.py')['sha256'] = '0' * 64
            recipe_path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'original runtime'):
                binder.bind(args)
            recipe_path.write_bytes(original_recipe)
            manifest = binder.bind(args)
            root = args.out / 'inputs'
            records_path = root / 'records.jsonl'
            original_records = records_path.read_bytes()
            for key, value in [('task_id', 'wrong-id'), ('gold', {}), ('wall_s', float('nan')), ('fwd_denoise', 0)]:
                rows = [json.loads(line) for line in original_records.splitlines()]
                rows[0][key] = value
                data = ''.join(json.dumps(row) + '\n' for row in rows).encode()
                records_path.write_bytes(data)
                changed = copy.deepcopy(manifest)
                changed['records']['sha256'] = score.sha(data)
                with self.subTest(key=key), self.assertRaises(ValueError):
                    score.validate_records(changed, root)
            records_path.write_bytes(original_records)
            changed = copy.deepcopy(manifest)
            changed['scope'] = 'full_cell'
            with self.assertRaisesRegex(ValueError, 'Scope'):
                score.validate_records(changed, root)
            with self.assertRaisesRegex(ValueError, 'Boolean'):
                score.score_records(score.validate_records(manifest, root), lambda *_: 1)


if __name__ == '__main__':
    unittest.main()
