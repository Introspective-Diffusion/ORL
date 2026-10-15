#!/usr/bin/env python3
"""Prepare a hash-verified scorer runtime from explicitly supplied local bundles."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = 'repro.eval.portable'

from .score_benchmarks import registry, required_files, validate_runtime


def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def copy_checked(stream, target, expected):
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as output:
        shutil.copyfileobj(stream, output, length=1024 * 1024)
    if file_sha(target) != expected:
        raise ValueError('Scoring artifact digest mismatch: ' + target.name)


def stage(args):
    profiles = registry()['profiles']
    if len(args.bundle) != len(args.bundle_sha256):
        raise ValueError('Every authorized bundle requires its expected SHA256')
    if args.out.exists():
        raise ValueError('Scoring runtime output must be new')
    needed = {}
    for key in args.profile:
        if key not in profiles:
            raise ValueError('Unknown scoring profile: ' + key)
        for path, digest in required_files(profiles[key]).items():
            if path in needed and needed[path] != digest:
                raise ValueError('Conflicting profile artifact identities')
            needed[path] = digest
    sources = []
    for bundle, expected in zip(args.bundle, args.bundle_sha256):
        if file_sha(bundle) != expected:
            raise ValueError('Supplied bundle SHA256 mismatch')
    args.out.mkdir(parents=True)
    for bundle, expected in zip(args.bundle, args.bundle_sha256):
        selected = []
        with tarfile.open(bundle) as archive:
            seen = set()
            for member in archive:
                relative = PurePosixPath(member.name)
                if relative.is_absolute() or '..' in relative.parts:
                    raise ValueError('Unsafe archive path')
                name = str(relative)
                if name not in needed:
                    continue
                if name in seen or not member.isfile() or member.issym() or member.islnk():
                    raise ValueError('Duplicate/nonregular selected archive member')
                seen.add(name)
                target = args.out / name
                if target.exists():
                    # Multiple authorized bundles may contain the same immutable source.
                    digest = hashlib.sha256()
                    with archive.extractfile(member) as stream:
                        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                            digest.update(block)
                    if digest.hexdigest() != needed[name]:
                        raise ValueError('Conflicting artifact across supplied bundles')
                else:
                    with archive.extractfile(member) as stream:
                        copy_checked(stream, target, needed[name])
                selected.append(name)
        sources.append({'kind': 'bundle', 'name': bundle.name, 'sha256': expected, 'selected': selected})
    for specification in args.file:
        name, separator, value = specification.partition('=')
        if not separator or name not in needed:
            raise ValueError('File mapping must be a required registry path=local file')
        source = Path(value)
        if file_sha(source) != needed[name]:
            raise ValueError('Supplied source file SHA256 mismatch')
        target = args.out / name
        if not target.exists():
            with source.open('rb') as stream:
                copy_checked(stream, target, needed[name])
        sources.append({'kind': 'file', 'path': name, 'sha256': needed[name]})
    missing = [name for name in needed if not (args.out / name).is_file()]
    if missing:
        raise ValueError('Missing explicitly supplied artifacts: ' + ', '.join(missing))
    for key in args.profile:
        validate_runtime(args.out, profiles[key])
    receipt = {'schema_version': 1, 'profiles': args.profile, 'files': needed, 'sources': sources,
               'network_access': False, 'generated_code_executed': False,
               'scope': 'Source/data/wheel staging only; no scorer imports or execution'}
    (args.out / 'staging_receipt.json').write_text(json.dumps(receipt, indent=2))
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', action='append', required=True, choices=tuple(registry()['profiles']))
    parser.add_argument('--bundle', type=Path, action='append', default=[])
    parser.add_argument('--bundle-sha256', action='append', default=[])
    parser.add_argument('--file', action='append', default=[], help='Required registry path=local source/data/wheel file')
    parser.add_argument('--out', type=Path, required=True)
    result = stage(parser.parse_args(argv))
    print(json.dumps({'profiles': result['profiles'], 'files': len(result['files']), 'generated_code_executed': False}))


if __name__ == '__main__':
    main()
