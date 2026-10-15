import argparse
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from repro.eval.portable import stage_scoring as staging
from repro.eval.portable.score_benchmarks import sha


class StageScoringTests(unittest.TestCase):
    def fixture(self, root, extra=(), omit=False):
        source = b'raise RuntimeError("scoring source must never execute during staging")\n'
        tests = b'{"licensed_test_fixture":true}\n'
        profile = {'source_files': {'repo/ref.py': sha(source)},
                   'data_files': {'private/tests.json': sha(tests)}, 'wheels': {}}
        members = [('repo/ref.py', source), ('raw/generated.py', b'raise RuntimeError("never copied")'),
                   ('.env', b'PRIVATE=never-copied')]
        if not omit:
            members.append(('private/tests.json', tests))
        bundle = root / 'authorized.tar.gz'
        with tarfile.open(bundle, 'w:gz') as archive:
            for name, data in [*members, *extra]:
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
        args = argparse.Namespace(profile=['fixture'], bundle=[bundle],
            bundle_sha256=[staging.file_sha(bundle)], file=[], out=root / 'runtime')
        return args, {'profiles': {'fixture': profile}}

    def test_selected_closure_only_without_importing_source(self):
        with tempfile.TemporaryDirectory() as temp:
            args, registry = self.fixture(Path(temp))
            with patch.object(staging, 'registry', return_value=registry):
                result = staging.stage(args)
            self.assertEqual(set(result['files']), {'repo/ref.py', 'private/tests.json'})
            self.assertFalse((args.out / '.env').exists())
            self.assertFalse((args.out / 'raw').exists())
            self.assertFalse(result['generated_code_executed'])
            self.assertEqual(json.loads((args.out / 'staging_receipt.json').read_text())['sources'][0]['sha256'], args.bundle_sha256[0])

    def test_bundle_hash_missing_and_duplicate_fail(self):
        for mode in ('hash', 'missing', 'duplicate', 'traversal'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                extra = [('repo/ref.py', b'wrong')] if mode == 'duplicate' else [('../escape', b'x')] if mode == 'traversal' else []
                args, registry = self.fixture(Path(temp), extra, omit=mode == 'missing')
                if mode == 'hash': args.bundle_sha256 = ['0' * 64]
                with patch.object(staging, 'registry', return_value=registry):
                    with self.assertRaises(ValueError): staging.stage(args)

    def test_explicit_file_mapping_completes_missing_bundle_asset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args, registry = self.fixture(root, omit=True)
            supplied = root / 'licensed.json'
            supplied.write_bytes(b'{"licensed_test_fixture":true}\n')
            args.file = ['private/tests.json=' + str(supplied)]
            with patch.object(staging, 'registry', return_value=registry):
                result = staging.stage(args)
            self.assertEqual(len(result['files']), 2)
            self.assertEqual(result['sources'][-1]['kind'], 'file')


if __name__ == '__main__':
    unittest.main()
