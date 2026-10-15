import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from runtime.stage import stage


class StageTests(unittest.TestCase):
    def test_all_profiles_and_inventories(self):
        with tempfile.TemporaryDirectory() as directory:
            for family in ('qwen-dlm', 'llada', 'dg'):
                with self.subTest(family=family):
                    output = Path(directory) / family
                    receipt = stage(family, output)
                    manifest = json.loads((output / 'runtime_manifest.json').read_text())
                    self.assertFalse(receipt['model_execution'])
                    self.assertFalse(manifest['dependencies_installed'])
                    self.assertEqual(manifest['family'], family)
                    self.assertEqual(receipt['files'], len(manifest['files']))
                    for relative, digest in manifest['files'].items():
                        self.assertEqual(hashlib.sha256((output / relative).read_bytes()).hexdigest(), digest)
                    if family != 'dg':
                        self.assertTrue((output / 'engine/inference/sglang').is_dir())
                    if family == 'qwen-dlm':
                        from inference.adapters.qwen_dlm import validate_runtime
                        validate_runtime(output, 'repro/eval/run_eval.py')
                    with self.assertRaises(ValueError):
                        stage(family, output)

    def test_invalid_family_creates_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'invalid'
            with self.assertRaises(ValueError):
                stage('invalid', output)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
