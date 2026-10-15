#!/usr/bin/env python3
"""Create a checksum-verified local engine tree; no network or GPU operations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contained(root, relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Unsafe source path: ' + relative)
    target = root / path
    for parent in target.parents:
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError('Symlink parent in source: ' + relative)
    return target


def prepare(source, output, profile, manifest_path=HERE / 'source_manifest.json'):
    source, output = source.resolve(), output.absolute()
    if output.exists() or output.is_symlink():
        raise ValueError('Use a new output directory; existing paths are never replaced')
    manifest = json.loads(manifest_path.read_text())
    if profile not in ('sdar', 'llada'):
        raise ValueError('Unknown engine profile')
    patch = manifest_path.parent / 'sdar-engine.patch'
    if profile == 'sdar' and sha(patch) != manifest['patch_sha256']:
        raise ValueError('SDAR engine patch changed')
    for relative, entry in manifest['files'].items():
        path = contained(source, relative)
        if entry['kind'] == 'symlink':
            if not path.is_symlink() or str(path.readlink()) != entry['target']:
                raise ValueError('Source symlink mismatch: ' + relative)
        elif path.is_symlink() or not path.is_file() or sha(path) != entry['sha256']:
            raise ValueError('Source content mismatch: ' + relative)
    output.mkdir(parents=True)
    omitted = []
    for relative, entry in manifest['files'].items():
        if entry['kind'] == 'symlink':
            if relative != manifest['nonruntime_format_symlink']:
                raise ValueError('Unreviewed source symlink: ' + relative)
            omitted.append(relative)
            continue
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
        target.chmod(entry['mode'])
    if profile == 'sdar':
        # Apply only the bundled, hash-checked patch inside the new engine tree.
        environment = {k: v for k, v in os.environ.items()
                       if k not in ('GIT_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE')}
        environment['GIT_CEILING_DIRECTORIES'] = str(output.parent.resolve())
        for options in (['--check'], []):
            subprocess.run(['git', 'apply', *options, str(patch.resolve())], cwd=output,
                           env=environment, check=True)
    expected = dict(manifest['files'])
    if profile == 'sdar':
        expected.update(manifest['sdar_changes'])
    for relative, entry in expected.items():
        if entry['kind'] != 'symlink' and sha(output / relative) != entry['sha256']:
            raise ValueError('Materialized engine mismatch: ' + relative)
    receipt = {'schema_version': 1, 'profile': profile,
               'source_revision': manifest['base_revision'],
               'engine_revision': manifest['sdar_revision'] if profile == 'sdar' else manifest['base_revision'],
               'source_manifest_sha256': sha(manifest_path),
               'verified_files': len(expected) - len(omitted),
               'omitted_nonruntime_symlinks': omitted,
               'scope': 'Source preparation only; no dependency installation or GPU acceptance'}
    (output / 'engine_manifest.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('sdar', 'llada'), required=True)
    parser.add_argument('--source-root', type=Path, default=HERE.parents[1])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_root, args.out, args.profile), indent=2))


if __name__ == '__main__':
    main()
