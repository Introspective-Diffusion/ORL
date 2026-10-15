"""Assemble the hash-verified scorer runtime for the main-table or transfer profiles from this repository and downloaded files."""
import argparse
import subprocess
import sys
from pathlib import Path

EVAL = Path(__file__).resolve().parents[1]
TRAINING = EVAL.parent / 'training'
SCORING_SOURCES = EVAL / 'runtime/scoring/sources'
PROFILES = ['qwen-dlm.gsm8k', 'qwen-dlm.math500', 'qwen-dlm.he', 'paper4.gsm8k', 'paper4.math500', 'paper4.he', 'main.mbpp']
# IFEval also needs the recorded langdetect wheel and NLTK data, which are not distributed.
TRANSFER_PROFILES = ['transfer.aime', 'transfer.gpqa', 'transfer.mmlu_pro', 'transfer.lcb']
SOURCES = {
    'repo/repro/eval/benchmarks.py': EVAL / 'repro/eval/benchmarks.py',
    'repo/repro/reward/code_taco_reward.py': EVAL / 'repro/reward/code_taco_reward.py',
    'repo/training/diffusion-reasoning-dapo-latent/verl/verl/utils/reward_score/math_verify.py':
        TRAINING / 'verl/verl/utils/reward_score/math_verify.py',
    'he_repo/repro/experiments/diffusion_gemma_aec_extra_eval_20260913/dgemma_core_score_point.py':
        SCORING_SOURCES / 'dgemma_core_score_point.py',
    'repo/repro/experiments/paper_repro_20260917/score_mbpp427_our_extractor.py':
        SCORING_SOURCES / 'score_mbpp427_our_extractor.py',
    'transfer_repo/score_full_extra_results.py': SCORING_SOURCES / 'score_full_extra_results.py',
    'transfer_repo/lcb_score.py': SCORING_SOURCES / 'lcb_score.py',
    'transfer_repo/lcb_lib/testing_util.py': SCORING_SOURCES / 'lcb_lib/testing_util.py',
    'transfer_repo/lcb_lib/extract_utils.py': SCORING_SOURCES / 'lcb_lib/extract_utils.py',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheels', type=Path, required=True)
    parser.add_argument('--humaneval-plus', type=Path, help='HumanEvalPlus-v0.1.10.jsonl (main-table profiles)')
    parser.add_argument('--transfer', action='store_true', help='Stage AIME, GPQA, MMLU-Pro and LCB instead')
    parser.add_argument('--lcb-tests', type=Path, help='lcb_v6_tests.json from repro/eval/build_transfer_inputs.py')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if (args.lcb_tests if args.transfer else args.humaneval_plus) is None:
        parser.error('--transfer needs --lcb-tests; the main-table profiles need --humaneval-plus')
    sys.path.insert(0, str(EVAL))
    from repro.eval.portable.stage_scoring import registry

    profiles = TRANSFER_PROFILES if args.transfer else PROFILES
    files = {}
    for profile in profiles:
        entry = registry()['profiles'][profile]
        for path in [*entry['source_files'], *entry['data_files'], *entry['wheels']]:
            if path in SOURCES:
                files[path] = SOURCES[path]
            elif path.endswith('.whl'):
                files[path] = args.wheels / Path(path).name
            elif path.endswith('HumanEvalPlus-v0.1.10.jsonl') and not args.transfer:
                files[path] = args.humaneval_plus
            elif path.endswith('lcb_v6_tests.json') and args.transfer:
                files[path] = args.lcb_tests
            else:
                raise SystemExit(f'{profile} needs {path}, which this script does not know how to supply')
    command = [sys.executable, '-m', 'repro.eval.portable.stage_scoring']
    for profile in profiles:
        command += ['--profile', profile]
    for path, local in sorted(files.items()):
        command += ['--file', f'{path}={local}']
    command += ['--out', str(args.out)]
    print(f'{len(profiles)} profiles, {len(files)} files', flush=True)
    subprocess.run(command, check=True, cwd=EVAL,
                   env={'PATH': '/usr/bin:/bin', 'PYTHONPATH': str(EVAL), 'PYTHONDONTWRITEBYTECODE': '1'})


if __name__ == '__main__':
    main()
