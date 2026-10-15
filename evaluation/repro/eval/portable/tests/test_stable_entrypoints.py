"""Clean-directory CLI tests using inert trusted fixtures only."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[4]
EVAL = REPO / 'repro/eval'


class StableEntrypointTests(unittest.TestCase):
    def run_cli(self, cwd, path, *args, execute=False):
        command = [sys.executable, str(EVAL / path), *args]
        if execute:
            command.append('--execute')
        result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_absolute_scripts_work_without_repository_cwd(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ('generate.py', 'score.py', 'aggregate.py', 'portable/build_recipe.py',
                         'portable/bind_score_inputs.py', 'portable/bind_qwen_dlm_he_score.py',
                         'portable/bind_benchmark_score.py', 'portable/stage_scoring.py',
                         'portable/reproduce.py', 'portable/evaluate.py', 'portable/score_qwen_dlm_he.py'):
                with self.subTest(script=name):
                    self.assertIn('usage:', self.run_cli(directory, name, '--help').stdout)

    def test_module_clis_from_clean_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = dict(os.environ, PYTHONPATH=str(REPO))
            for name in ('generate', 'score', 'aggregate', 'portable.build_recipe',
                         'portable.bind_score_inputs', 'portable.bind_qwen_dlm_he_score',
                         'portable.bind_benchmark_score', 'portable.stage_scoring'):
                with self.subTest(module=name):
                    result = subprocess.run([sys.executable, '-m', 'repro.eval.' + name, '--help'],
                        cwd=directory, env=environment, text=True, capture_output=True, timeout=15)
                    self.assertEqual(result.returncode, 0, result.stderr)

    def test_generation_dry_run_then_trusted_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / 'runtime'
            runtime.mkdir()
            script = runtime / 'trusted_fixture.py'
            script.write_text('from pathlib import Path\nimport sys\nPath(sys.argv[1]).write_text("inert fixture")\n')
            recipe = {'schema_version': 1, 'protocol_id': 'trusted-cli-fixture',
                'artifacts': [{'root': 'runtime', 'path': script.name,
                               'sha256': hashlib.sha256(script.read_bytes()).hexdigest()}],
                'stages': {'generate': {'script': script.name, 'args': ['{output}/fixture.txt']}}}
            path = root / 'recipe.json'
            path.write_text(json.dumps(recipe))
            output = root / 'attempt'
            arguments = ['--recipe', str(path), '--runtime-root', str(runtime),
                '--input-root', str(root / 'inputs'), '--model-root', str(root / 'weights'),
                '--output-root', str(output)]
            self.run_cli(root, 'generate.py', *arguments)
            self.assertFalse(output.exists())
            self.run_cli(root, 'generate.py', *arguments, execute=True)
            self.assertEqual((output / 'artifacts/fixture.txt').read_text(), 'inert fixture')

    def test_score_dry_run_remains_isolated_and_never_mounts_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / 'runtime'
            runtime.mkdir()
            scorer = runtime / 'score.py'
            scorer.write_text('raise RuntimeError("must not execute during dry run")\n')
            recipe = {'schema_version': 1, 'protocol_id': 'score-dry-run-fixture',
                'artifacts': [{'root': 'runtime', 'path': scorer.name,
                               'sha256': hashlib.sha256(scorer.read_bytes()).hexdigest()}],
                'stages': {'score': {'script': scorer.name, 'image': 'fixture@sha256:' + 'a' * 64}}}
            path = root / 'recipe.json'
            path.write_text(json.dumps(recipe))
            result = self.run_cli(root, 'score.py', '--recipe', str(path),
                '--runtime-root', str(runtime), '--input-root', str(root / 'inputs'),
                '--model-root', str(root / 'weights'), '--output-root', str(root / 'out'))
            command = json.loads(result.stdout)['command']
            self.assertIn('--network=none', command)
            self.assertIn('--cap-drop=ALL', command)
            self.assertIn('--read-only', command)
            self.assertNotIn(str(root / 'weights'), ' '.join(command))
            self.assertFalse((root / 'out').exists())

    def test_aggregation_remains_data_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = root / 'metrics.jsonl'
            rows.write_text(json.dumps({'task_id': 'fixture/0', 'finish': 'stop',
                'completion_tokens': 10, 'fwd_denoise': 2, 'fwd_total': 4, 'wall_s': 0.5}) + '\n')
            manifest = {'schema_version': 1, 'evaluations': [{'id': 'fixture',
                'protocol_id': 'fixture-v1', 'expected_task_ids': ['fixture/0'],
                'generation_shards': [{'path': rows.name, 'sha256': hashlib.sha256(rows.read_bytes()).hexdigest()}]}]}
            path = root / 'manifest.json'
            path.write_text(json.dumps(manifest))
            result = self.run_cli(root, 'aggregate.py', '--manifest', str(path), '--input-root', str(root))
            output = json.loads(result.stdout)
            self.assertFalse(output['generated_code_executed'])
            self.assertFalse(output['results'][0]['quality_scored'])
            self.assertEqual(output['results'][0]['tps_pooled_clean'], 20)


if __name__ == '__main__':
    unittest.main()
