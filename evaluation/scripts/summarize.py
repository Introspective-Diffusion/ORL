"""Collect the scored main-table cells in $ORL_EVAL_WORK and print them next to the paper's numbers
(results/reference/main.json). DiffusionGemma cells average their three seeds; the latest attempt of a cell counts."""
import argparse
import glob
import json
import math
import os
import re
import statistics
import sys
from pathlib import Path

EVAL = Path(__file__).resolve().parents[1]
BENCHMARKS = {'gsm8k': 'gsm8k', 'math500': 'math500', 'he': 'humaneval', 'mbpp': 'mbpp'}
CELL = re.compile(r'((qwen-dlm|llada|dg)-(base|orl)(?:-[A-Za-z0-9._-]+)?)-(gsm8k|math500|he|mbpp)-s(\d+)\.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', help='only this model (the NAME given to main_table.sh)')
    parser.add_argument('--out', type=Path, help='also write the rows as JSON')
    args = parser.parse_args()
    if not os.environ.get('ORL_EVAL_WORK'):
        sys.exit('set ORL_EVAL_WORK to the evaluation work directory')
    reference = json.loads((EVAL / 'results/reference/main.json').read_text())['table']
    cells = {}
    for path in sorted(glob.glob(os.path.join(os.environ['ORL_EVAL_WORK'], 'scoring/attempts/*.metrics.json')),
                       key=os.path.getmtime):
        try:
            results = json.loads(Path(path).read_text())['results']
        except (OSError, ValueError, KeyError):
            print(f'skipping unreadable {path}', file=sys.stderr)
            continue
        for result in results:
            match = CELL.match(result.get('id', ''))
            if not match:
                continue
            name, family, role, bench, seed = match.groups()
            if args.name and name != args.name:
                continue
            cells[(name, family, role, bench, int(seed))] = dict(
                accuracy_pct=result['accuracy'] * 100, tpf=result.get('m2t_tpf_mean_clean', math.nan), n=result['n'])
    rows = []
    for name, family, role, bench in sorted({key[:4] for key in cells}):
        ref = reference[f'{family}.{role}.{BENCHMARKS[bench]}']['paper']
        seeds = sorted(seed for (n, f, r, b, seed) in cells if (n, b) == (name, bench))
        ours = [cells[(name, family, role, bench, seed)] for seed in seeds]
        rows.append(dict(model=name, benchmark=BENCHMARKS[bench], seeds=seeds,
                         accuracy_pct=statistics.mean(c['accuracy_pct'] for c in ours),
                         tpf=statistics.mean(c['tpf'] for c in ours),
                         paper_accuracy_pct=ref['accuracy_pct'], paper_tpf=ref['tpf']))
    print(f"{'model':24s} {'benchmark':10s} {'seeds':>5s} {'acc':>7s} {'paper':>6s} {'TPF':>7s} {'paper':>6s}")
    for row in rows:
        print(f"{row['model']:24s} {row['benchmark']:10s} {len(row['seeds']):5d} {row['accuracy_pct']:7.2f} "
              f"{row['paper_accuracy_pct']:6.1f} {row['tpf']:7.3f} {row['paper_tpf']:6.1f}")
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    main()
