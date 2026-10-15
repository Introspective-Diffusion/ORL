#!/usr/bin/env python3
"""Generation-only adapter to a separately pinned local DG server."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import urlparse


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('runtime','script','problems','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--benchmark',choices=('gsm8k','he','math500','mbpp'),required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--base',default='http://127.0.0.1:8000')
    args=parser.parse_args()
    url=urlparse(args.base)
    if url.scheme!='http' or url.hostname not in ('localhost','127.0.0.1','::1') or url.username or url.password:
        raise ValueError('Generation server must be a credential-free local HTTP endpoint')
    args.out.mkdir(parents=True,exist_ok=True)
    common=[sys.executable,str(args.runtime/args.script),'--base',args.base,'--model','dg',
            '--seed',str(args.seed),'--max-denoising-steps','48']
    if args.benchmark=='mbpp':
        rows=[json.loads(line) for line in args.problems.read_text().splitlines() if line.strip()]
        if not all(row.get('messages') for row in rows):raise ValueError('MBPP427 requires frozen canonical messages; no prompt conversion is inferred')
        command=common+['--problems',str(args.problems),'--benchmark','mbpp427',
                        '--out',str(args.out/'mbpp/shard_w0.jsonl')]
    else:
        with tempfile.TemporaryDirectory(prefix='inputs-',dir=args.out) as temp:
            assets=Path(temp)
            name={'gsm8k':'gsm8k.jsonl','he':'humaneval.jsonl','math500':'math500.jsonl'}[args.benchmark]
            source=assets/name;shutil.copyfile(args.problems,source)
            command=common+['--assets',str(assets),'--gsm',str(source),'--only',args.benchmark,'--out',str(args.out)]
            subprocess.run(command,check=True)
        return
    subprocess.run(command,check=True)


if __name__=='__main__':main()
