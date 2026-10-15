#!/usr/bin/env python3
"""Bind a new scoring recipe to complete generation records without executing them."""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = 'repro.eval.portable'

from .build_recipe import inventory


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation-recipe',type=Path,required=True)
    parser.add_argument('--records-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.out.exists():raise ValueError('Refusing an existing scoring recipe')
    recipe=json.loads(args.generation_recipe.read_text())
    ids=[]
    for path in args.records_root.rglob('*.jsonl'):
        for line in path.read_text().splitlines():
            if line.strip():ids.append(json.loads(line)['task_id'])
    if len(ids)!=len(set(ids)) or set(ids)!=set(recipe['expected_task_ids']):
        raise ValueError('Generation record coverage is not an exact unique match')
    recipe['artifacts']=[item for item in recipe['artifacts'] if item['root']!='inputs']
    recipe['artifacts']+=inventory(args.records_root.resolve(),'inputs')
    recipe['stages'].pop('generate',None)
    recipe['source_generation_recipe']=args.generation_recipe.name
    args.out.write_text(json.dumps(recipe,indent=2)+'\n')
    print(json.dumps({'scoring_rows':len(ids),'quality_scored':False}))


if __name__=='__main__':main()
