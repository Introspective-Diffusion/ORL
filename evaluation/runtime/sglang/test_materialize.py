import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from runtime.sglang.materialize import prepare


class MaterializeTests(unittest.TestCase):
    def test_nested_git_repositories_do_not_capture_patch_application(self):
        with tempfile.TemporaryDirectory() as directory:
            outer = Path(directory)
            inner = outer / 'project'
            inner.mkdir()
            for repository in (outer, inner):
                subprocess.run(['git', 'init', '-q', str(repository)], check=True)
            source, manifest = self.fixture(inner)
            # A decoy with the patch's relative name exposes writes to an ancestor.
            (outer / 'engine.py').write_text('old\n')
            subprocess.run(['git', '-C', str(outer), 'add', 'engine.py'], check=True)
            subprocess.run(['git', '-C', str(inner), 'add', 'source/engine.py'], check=True)
            indices = {repo: (repo / '.git/index').read_bytes() for repo in (outer, inner)}
            contaminated = {'GIT_DIR': str(outer / '.git'), 'GIT_WORK_TREE': str(outer),
                            'GIT_INDEX_FILE': str(outer / '.git/index')}
            with patch.dict(os.environ, contaminated):
                for profile, expected in [('llada', 'old\n'), ('sdar', 'new\n')]:
                    output = inner / 'generated' / profile
                    prepare(source, output, profile, manifest)
                    self.assertEqual((output / 'engine.py').read_text(), expected)
            self.assertEqual((inner / 'generated/llada/engine.py').read_text(), 'old\n')
            self.assertEqual((source / 'engine.py').read_text(), 'old\n')
            self.assertEqual((outer / 'engine.py').read_text(), 'old\n')
            for repo, index in indices.items():
                self.assertEqual((repo / '.git/index').read_bytes(), index)

    def test_bundled_profiles_differ_only_at_eight_reviewed_paths_inside_repo(self):
        source = Path(__file__).resolve().parents[2]
        manifest_path = Path(__file__).with_name('source_manifest.json')
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(len(manifest['sdar_changes']), 8)
        before = {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                  for name, entry in manifest['files'].items() if entry['kind'] == 'file'}
        with tempfile.TemporaryDirectory(prefix='.materialize-regression-', dir=source) as directory:
            root = Path(directory)
            for profile in ('llada', 'sdar'):
                prepare(source, root / profile, profile, manifest_path)
            paths = set(manifest['files']) | set(manifest['sdar_changes'])
            changed = set()
            for name in paths:
                if manifest['files'].get(name, {}).get('kind') == 'symlink':
                    self.assertFalse((root / 'llada' / name).exists())
                    self.assertFalse((root / 'sdar' / name).exists())
                    continue
                base, sdar = root / 'llada' / name, root / 'sdar' / name
                if not base.exists() or base.read_bytes() != sdar.read_bytes():
                    changed.add(name)
                if name in before:
                    self.assertEqual(hashlib.sha256(base.read_bytes()).hexdigest(), before[name])
                    self.assertFalse(base.samefile(source / name))
            self.assertEqual(changed, set(manifest['sdar_changes']))
        after = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in before}
        self.assertEqual(before, after)

    def fixture(self, root):
        source = root / 'source'
        source.mkdir()
        (source / 'engine.py').write_text('old\n')
        patch = root / 'sdar-engine.patch'
        patch.write_text('diff --git a/engine.py b/engine.py\n--- a/engine.py\n'
                         '+++ b/engine.py\n@@ -1 +1 @@\n-old\n+new\n')
        entry = lambda text: {'kind': 'file', 'mode': 0o644,
                              'sha256': hashlib.sha256(text.encode()).hexdigest()}
        manifest = root / 'source_manifest.json'
        manifest.write_text(json.dumps({'base_revision': 'base', 'sdar_revision': 'sdar',
            'patch_sha256': hashlib.sha256(patch.read_bytes()).hexdigest(),
            'nonruntime_format_symlink': 'format-link',
            'files': {'engine.py': entry('old\n')},
            'sdar_changes': {'engine.py': entry('new\n')}}))
        return source, manifest

    def test_two_profiles_leave_source_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, manifest = self.fixture(root)
            for profile, text in [('llada', 'old\n'), ('sdar', 'new\n')]:
                receipt = prepare(source, root / profile, profile, manifest)
                self.assertEqual((root / profile / 'engine.py').read_text(), text)
                self.assertEqual(receipt['engine_revision'], 'base' if profile == 'llada' else 'sdar')
            self.assertEqual((source / 'engine.py').read_text(), 'old\n')

    def test_source_change_and_occupied_output_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, manifest = self.fixture(root)
            with self.assertRaises(ValueError):
                prepare(source, source, 'sdar', manifest)
            (source / 'engine.py').write_text('changed\n')
            with self.assertRaisesRegex(ValueError, 'Source content'):
                prepare(source, root / 'output', 'llada', manifest)
            self.assertFalse((root / 'output').exists())

    def test_patch_change_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, manifest = self.fixture(root)
            (root / 'sdar-engine.patch').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'patch changed'):
                prepare(source, root / 'output', 'sdar', manifest)


if __name__ == '__main__':
    unittest.main()
