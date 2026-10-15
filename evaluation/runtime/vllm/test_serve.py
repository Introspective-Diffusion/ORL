from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from runtime.vllm.serve import plan, validate_patch, terminate


class ServeTests(unittest.TestCase):
    def test_roles_context_and_model_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / 'model'
            model.mkdir()
            (model / 'generation_config.json').write_text('{"max_new_tokens":256}')
            for role, temp, entropy in [('base', .4, .1), ('orl', .5, .12)]:
                command, config, receipt = plan(model, role, 'aime', Path(directory) / role)
                self.assertEqual(config['t_min'], temp)
                self.assertEqual(config['sampler_config']['entropy_bound'], entropy)
                self.assertIsNone(config['max_new_tokens'])
                self.assertEqual(receipt['max_model_len'], 32768)
                self.assertFalse(receipt['server_ready'])
                self.assertIn('127.0.0.1', command)
                self.assertEqual(len(receipt['model_manifest_sha256']), 64)
            _, _, receipt = plan(model, 'base', 'humaneval', Path(directory) / 'he')
            self.assertEqual(receipt['max_model_len'], 8192)
            before = receipt['model_manifest_sha256']
            (model / 'config.json').write_text('{}')
            self.assertNotEqual(plan(model, 'base', 'humaneval', Path(directory) / 'he')[2]['model_manifest_sha256'], before)

    def test_wrong_sampler_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'sampler.py'
            source.write_text('HF-reference sampling')
            validate_patch(source, 'base')
            with self.assertRaises(ValueError):
                validate_patch(source, 'orl')
            source.write_text('HF-reference sampling adaptive_h_ref adaptive_scalar=gen.get')
            validate_patch(source, 'orl')
            with self.assertRaises(ValueError):
                validate_patch(source, 'base')

    def test_output_cannot_modify_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / 'model'
            model.mkdir()
            with self.assertRaises(ValueError):
                plan(model, 'base', 'humaneval', model / 'outputs')

    def test_cleanup_handles_already_exited_process_group(self):
        process = Mock(pid=123, returncode=0)
        process.poll.return_value = None
        with patch('runtime.vllm.serve.os.killpg', side_effect=ProcessLookupError):
            terminate(process)
        process.wait.assert_called_once_with(timeout=30)


if __name__ == '__main__':
    unittest.main()
