#!/usr/bin/env python3
"""Stage a portable, local source tree without downloads or model execution."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from runtime.sglang.materialize import prepare


ENGINE_PROFILES = {'qwen-dlm': 'sdar', 'llada': 'llada'}  # QwenDLM runs on the SDAR-patched SGLang profile


def stage(family, output):
    if family not in ('qwen-dlm', 'llada', 'dg'):
        raise ValueError('Unknown family')
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise ValueError('Use a fresh runtime directory')
    output.mkdir(parents=True)
    if family in ENGINE_PROFILES:
        prepare(ROOT, output / 'engine', ENGINE_PROFILES[family])
    sources = {
        'qwen-dlm': ['repro/eval/eval_client.py', 'repro/eval/run_eval.py', 'repro/eval/benchmarks.py',
                     'repro/reward/code_taco_reward.py', 'repro/eval/portable/score_qwen_dlm_he.py'],
        'llada': ['inference/clients/multi_gen.py'],
        'dg': ['inference/clients/vllm_paper4.py', 'inference/clients/vllm_full_extra_client.py',
               'inference/vllm/patches/vllm_dg_hf_align_patch.py',
               'inference/vllm/patches/vllm_dg_adaptive_scalar_patch.py'],
    }[family]
    for relative in sources:
        target = output / relative if relative.startswith('repro/') and not relative.startswith('repro/eval/portable/') else output / Path(relative).name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    files = {str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(output.rglob('*')) if path.is_file()}
    receipt = {'schema_version': 1, 'family': family, 'files': files,
               'scope': 'Local source preparation only; dependencies, weights and licensed inputs are separate',
               'model_execution': False, 'dependencies_installed': False}
    (output / 'runtime_manifest.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return {'family': family, 'output': str(output), 'files': len(files), 'model_execution': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=('qwen-dlm', 'llada', 'dg'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(stage(args.family, args.out), indent=2))


if __name__ == '__main__':
    main()
