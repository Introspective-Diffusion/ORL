"""Byte-level comparison of two safetensors exports, tensor by tensor (dtype, shape and a sha256 of the raw bytes),
without torch: compare_exports.py EXPORT_A EXPORT_B"""
import hashlib, json, struct, sys
from pathlib import Path


def tensors(directory):
    directory = Path(directory)
    index = directory / 'model.safetensors.index.json'
    files = sorted(set(json.loads(index.read_text())['weight_map'].values())) if index.exists() else ['model.safetensors']
    out = {}
    for name in files:
        with open(directory / name, 'rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            header = json.loads(f.read(n))
            base = 8 + n
            for key, meta in header.items():
                if key == '__metadata__':
                    continue
                start, end = meta['data_offsets']
                f.seek(base + start)
                digest = hashlib.sha256()
                remaining = end - start
                while remaining:
                    chunk = f.read(min(remaining, 1 << 24))
                    digest.update(chunk)
                    remaining -= len(chunk)
                out[key] = (meta['dtype'], tuple(meta['shape']), digest.hexdigest())
    return out


a, b = tensors(sys.argv[1]), tensors(sys.argv[2])
only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
common = sorted(set(a) & set(b))
meta_diff = [k for k in common if a[k][:2] != b[k][:2]]
byte_diff = [k for k in common if a[k][:2] == b[k][:2] and a[k][2] != b[k][2]]
print(f'tensors: {len(a)} vs {len(b)}; common {len(common)}; only in first {len(only_a)}; only in second {len(only_b)}')
print(f'dtype/shape differ: {len(meta_diff)}; bytes differ: {len(byte_diff)}; identical: {len(common) - len(meta_diff) - len(byte_diff)}')
for k in (only_a[:3] + only_b[:3] + meta_diff[:3] + byte_diff[:5]):
    print('  e.g.', k, a.get(k, ('-',))[:2], b.get(k, ('-',))[:2])
print('VERDICT:', 'IDENTICAL' if not (only_a or only_b or meta_diff or byte_diff) else 'DIFFERENT')
