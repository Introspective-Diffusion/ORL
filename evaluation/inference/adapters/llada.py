#!/usr/bin/env python3
"""Generation-only adapter for a provided checksum-pinned multi_gen runtime."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('runtime','script','engine','model','config','problems','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--benchmark',choices=('gsm8k','he','math500','mbpp'),required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--max-new',type=int,required=True)
    args=parser.parse_args()
    import yaml
    config=yaml.safe_load(args.config.read_text())
    if not isinstance(config,dict):raise ValueError('Decoder config must be a mapping')
    config['randtok_seed']=args.seed;config['temperature']=0.0
    args.out.mkdir(parents=True,exist_ok=True)
    cfg=args.out/'decoder.yaml';cfg.write_text(yaml.safe_dump(config,sort_keys=False))
    result=args.out/args.benchmark;result.mkdir()
    plan=args.out/'plan.json'
    plan.write_text(json.dumps({'datasets':[{'name':args.benchmark,'problems':str(args.problems),
                    'max_new':args.max_new,'out_dir':str(result)}]},indent=2)+'\n')
    claim=args.out/'claims';claim.mkdir()
    environment=dict(os.environ,MODEL_DIR=str(args.model),ORL_ENGINE_ROOT=str(args.engine),CFG=str(cfg),PLAN=str(plan),
                     TEMP='0',ALGO='SDARPrefixKV',WORKER='0',CLAIM_DIR=str(claim),MEM_FRAC='0.60',
                     ATTN_BACKEND='flashinfer',CUDA_VISIBLE_DEVICES='0')
    subprocess.run([sys.executable,str(args.runtime/args.script)],env=environment,check=True)


if __name__=='__main__':main()
