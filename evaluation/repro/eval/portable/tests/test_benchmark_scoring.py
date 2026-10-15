"""Scorer routing and binding with trusted inert fixtures, never model code."""
import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from repro.eval.portable import bind_benchmark_score as binder
from repro.eval.portable import evaluate
from repro.eval.portable import score_benchmarks as scorer
from repro.eval.portable.reproduce import command


def row(task='fixture/0', finish='stop'):
    return {'task_id': task, 'raw': 'inert trusted fixture', 'gold': '42', 'meta': {},
            'finish': finish, 'completion_tokens': 5, 'fwd_denoise': 2, 'fwd_total': 3, 'wall_s': 0.5}


class ScoringTests(unittest.TestCase):
    def test_registry_covers_all_evaluated_family_benchmarks(self):
        profiles = scorer.registry()['profiles']
        for family in ('qwen-dlm', 'llada', 'dg'):
            for benchmark in scorer.BENCHMARKS:
                if family == 'qwen-dlm' and benchmark in scorer.BENCHMARKS[4:]:
                    with self.assertRaises(ValueError): scorer.profile_key(family, benchmark)
                else:
                    profile = profiles[scorer.profile_key(family, benchmark)]
                    self.assertGreater(profile['full_cell_n'], 0)
                    self.assertTrue(profile['protocol_id'])
                    self.assertTrue(all(len(d) == 64 for d in scorer.required_files(profile).values()))

    def test_paper_gsm_historical_signed_number_parser(self):
        self.assertEqual(scorer.gsm_answer('thinking 4 </think> answer: -12.5'), '-12.5')
        self.assertEqual(scorer.gsm_answer(r'first \boxed{1} then \boxed{2,000}'), '2000')
        rows = [dict(row(), raw='Answer: 42'), dict(row('fixture/1', 'length'), raw='Answer: 42')]
        verdicts, _ = scorer.score_rows(rows, 'llada', 'gsm8k', Path('/not-used'), Path('/not-used'))
        self.assertEqual(verdicts, {'fixture/0': True, 'fixture/1': False})

    def test_family_specific_dispatch_and_collapse(self):
        calls = []
        class Judge:
            def score(self, text, item):
                calls.append(('qwen-dlm', text, item))
                return True
        def score_code(rows, benchmark, work):
            calls.append(('evalplus', benchmark))
            (work / 'samples-sanitized_eval_results.json').write_text(json.dumps(
                {'eval': {r['task_id']: [{'base_status': 'pass', 'plus_status': 'fail'}] for r in rows}}))
        module = types.SimpleNamespace(GSM8K=Judge, MATH500=Judge, HumanEval=Judge,
            compute_score=lambda text, gold: calls.append(('paper_math', gold)) or True,
            score_one=lambda item: (item[1]['task_id'], item[1]['finish'] != 'length'),
            score_code=score_code)
        fake_data = types.ModuleType('evalplus.data')
        fake_data.get_human_eval_plus_hash = lambda: 'fe585eb4df8c88d844eeb463ea4d0302'
        for family in ('qwen-dlm', 'llada', 'dg'):
            for benchmark in ('gsm8k', 'math500', 'he', 'mbpp'):
                with self.subTest(family=family, benchmark=benchmark), tempfile.TemporaryDirectory() as temp:
                    rows = [dict(row(), raw='42'), row('fixture/1', 'length')]
                    with patch.dict(sys.modules, {'evalplus.data': fake_data}):
                        values, _ = scorer.score_rows(rows, family, benchmark, Path(temp), Path(temp), loader=lambda *_: module)
                    self.assertEqual(values, {'fixture/0': True, 'fixture/1': False})
        self.assertTrue(any(c[0] == 'qwen-dlm' for c in calls))
        self.assertTrue(any(c[0] == 'paper_math' for c in calls))
        self.assertTrue(any(c[0] == 'evalplus' for c in calls))

    def test_transfer_uses_original_functions_and_sorted_rng_order(self):
        seen = []
        def numeric(r):
            seen.append(r['task_id'])
            return r['finish'] != 'length'
        def instruction(rows):
            self.assertEqual([r['task_id'] for r in rows], sorted(r['task_id'] for r in rows))
            return {r['task_id']: r['finish'] != 'length' for r in rows}, {'strict_instruction_accuracy': 0.5}
        module = types.SimpleNamespace(aime_correct=numeric, mc_correct=numeric,
            ifeval_scores=instruction,
            lcb_scores=lambda rows, root, tests: {r['task_id']: r['finish'] != 'length' for r in rows})
        fake = types.ModuleType('langdetect')
        fake.DetectorFactory = types.SimpleNamespace(seed=None)
        for benchmark in ('aime', 'gpqa', 'mmlu_pro', 'ifeval', 'lcb'):
            with self.subTest(benchmark=benchmark), patch.dict(sys.modules, {'langdetect': fake}):
                values, extra = scorer.score_rows([row('b'), row('a', 'length')], 'dg', benchmark,
                    Path('/unused'), Path('/unused'), loader=lambda *_: module)
                self.assertEqual(values, {'a': False, 'b': True})
                if benchmark == 'ifeval':
                    self.assertEqual(fake.DetectorFactory.seed, 0)
                    self.assertEqual(extra['strict_instruction_accuracy'], 0.5)
        self.assertEqual(seen, ['a', 'b'] * 3)

    def fixture(self, root):
        original = root / 'problems.jsonl'
        records = root / 'records.jsonl'
        problems = [{'task_id': 'fixture/' + str(i), 'gold': '42', 'meta': {}} for i in range(2)]
        original.write_text(''.join(json.dumps(r) + '\n' for r in problems))
        records.write_text(''.join(json.dumps(row(r['task_id'])) + '\n' for r in problems))
        runtime = root / 'supplied-runtime'
        runtime.mkdir()
        (runtime / '.env').write_text('UNLISTED_PRIVATE=must-not-be-mounted')
        profile = copy.deepcopy(scorer.registry()['profiles']['paper4.gsm8k'])
        profile['full_cell_n'] = 2
        args = argparse.Namespace(family='llada', benchmark='gsm8k', records=[records], records_root=None,
            original_problems=original, expected_problems_sha256=scorer.sha(original.read_bytes()),
            runtime_root=runtime, scorer_image='fixture@sha256:' + 'a' * 64, out=root / 'bound',
            evaluation_id='fixture-s12345', generation_protocol_id='fixture-generation-v1')
        return args, profile

    def test_binding_and_aggregation_are_data_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args, profile = self.fixture(root)
            with patch.object(binder, 'registry', return_value={'profiles': {'paper4.gsm8k': profile}}):
                manifest = binder.bind(args)
            self.assertFalse((args.out / 'runtime/.env').exists())
            rows = scorer.validate_inputs(manifest, args.out / 'inputs', profile)
            output = root / 'scores'
            output.mkdir()
            values = {r['task_id']: True for r in rows}
            scorer.emit(manifest, rows, values, {}, output, profile, {})
            aggregate = evaluate.audit(json.loads((output / 'aggregation_manifest.json').read_text()), output, True)
            self.assertEqual(aggregate['results'][0]['correct'], 2)
            self.assertEqual(aggregate['results'][0]['scorer_protocol_id'], profile['protocol_id'])
            self.assertFalse(aggregate['generated_code_executed'])
            recipe = json.loads((args.out / 'score_recipe.json').read_text())
            argv, _ = command(recipe, 'score', {k: root / k for k in ('runtime', 'inputs', 'output')}, 'python3', 'docker')
            self.assertIn('--platform=linux/amd64', argv)
            self.assertIn('--cpus=2', argv)
            self.assertEqual(scorer.registry()['profiles']['transfer.lcb']['cpus'], 8)

    def test_hash_gold_duplicate_and_metadata_drift_rejected(self):
        for mode in ('hash', 'gold', 'duplicate'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                args, profile = self.fixture(Path(temp))
                rows = binder.rows_from(args.records[0])
                if mode == 'hash': args.expected_problems_sha256 = '0' * 64
                if mode == 'gold': rows[0]['gold'] = 'wrong'
                if mode == 'duplicate': rows.append(rows[0])
                args.records[0].write_text(''.join(json.dumps(r) + '\n' for r in rows))
                with patch.object(binder, 'registry', return_value={'profiles': {'paper4.gsm8k': profile}}):
                    with self.assertRaises(ValueError): binder.bind(args)

    def test_aime_emits_two_distinct_complete_year_cells(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [dict(row(f'{year}/{i}'), meta={'year': year})
                    for year in ('aime24', 'aime25') for i in range(30)]
            profile = scorer.registry()['profiles']['transfer.aime']
            manifest = {'benchmark': 'aime', 'family': 'dg', 'evaluation_id': 'dg-base-s1234',
                        'generation_protocol_id': 'fixture', 'binding': {}}
            scorer.emit(manifest, rows, {r['task_id']: True for r in rows}, {}, root, profile, {})
            result = evaluate.audit(json.loads((root / 'aggregation_manifest.json').read_text()), root, True)
            self.assertEqual([c['n'] for c in result['results']], [30, 30])
            self.assertEqual(len({c['id'] for c in result['results']}), 2)


if __name__ == '__main__':
    unittest.main()
