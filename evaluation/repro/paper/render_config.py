"""Render the native QwenDLM/LLaDA decoder from a single validated paper cell."""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repro.paper.plan import expand


def render(suite, model, benchmark, seed):
    runs = [run for run in expand(suite, [model], [benchmark])['runs'] if run['seed'] == seed]
    if len(runs) != 1:
        raise ValueError('Select one validated model/benchmark/seed')
    run = runs[0]
    if run['family'] == 'dg':
        raise ValueError('DG uses runtime.vllm.serve to merge the checkpoint generation_config.json')
    return run['parameters']['decoder']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=('main', 'transfer', 'tps'), default='main')
    parser.add_argument('--model', required=True)
    parser.add_argument('--benchmark', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    config = render(args.suite, args.model, args.benchmark, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # JSON is valid YAML and avoids requiring a serializer in the planning environment.
    with args.out.open('x') as stream:
        stream.write(json.dumps(config, indent=2) + '\n')
    print(json.dumps({'output': str(args.out), 'model': args.model, 'benchmark': args.benchmark}))


if __name__ == '__main__':
    main()
