#!/usr/bin/env python3
"""Build concrete checksum inventories and runnable paper4 main-cell recipes."""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = 'repro.eval.portable'

from .reproduce import contained,sha256

COUNTS={'gsm8k':1319,'he':164,'math500':500,'mbpp':427}


def inventory(root,name):
    result=[]
    for path in sorted(root.rglob('*')):
        relative=path.relative_to(root)
        if '.git' in relative.parts or '__pycache__' in relative.parts:continue
        if path.is_symlink():raise ValueError('Materialize runtime/input/model symlinks before inventory')
        if not path.is_file():continue
        if path.name=='.env' or path.name.startswith('.env.') or path.name in ('.netrc','credentials','token'):
            raise ValueError('Credential-like file in supplied artifact root: '+str(relative))
        result.append({'root':name,'path':str(relative),'sha256':sha256(path)})
    if not result:raise ValueError('Empty artifact root: '+name)
    return result


def build(args):
    runtime=args.runtime_root.resolve();inputs=args.input_root.resolve();model=args.model_root.resolve()
    engine=(runtime/args.engine_subdir).resolve()
    if not engine.is_relative_to(runtime) or not engine.is_dir():raise ValueError('Engine must be a contained runtime directory')
    contained(runtime,args.runtime_script);contained(runtime,args.score_script)
    family=args.role.rsplit('-',1)[0]
    if family in ('llada','qwen-dlm'):contained(inputs,args.decoder_config)
    else:
        server=json.loads(contained(inputs,args.server_manifest).read_text())
        if not all(key in server for key in ('image_digest','model_manifest_sha256','diffusion_config','max_model_len','seed_protocol')):
            raise ValueError('DG server manifest is incomplete')
    rows=[json.loads(line) for line in contained(inputs,args.problems).read_text().splitlines() if line.strip()]
    ids=[r['task_id'] for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate task IDs')
    if not ids or (len(ids)!=COUNTS[args.benchmark] and not args.smoke):raise ValueError('Wrong full-cell task count')
    if args.seed not in ([12345] if family=='llada' else [1234] if family=='qwen-dlm' else [1234,1235,1236]):raise ValueError('Invalid main-cell seed')
    shard=None
    if family=='qwen-dlm':
        import yaml
        from inference.adapters import qwen_dlm as qwen_dlm_generate
        if args.smoke:raise ValueError('QwenDLM recipes preserve exact shards; arbitrary smoke subsets are not main shards')
        qwen_dlm_generate.validate_runtime(runtime,args.runtime_script)
        rows,_=qwen_dlm_generate.validate_inputs(contained(inputs,args.problems),args.benchmark)
        if yaml.safe_load(contained(inputs,args.decoder_config).read_text())!=qwen_dlm_generate.decoder(args.role,args.benchmark):
            raise ValueError('QwenDLM decoder configuration mismatch')
        index=getattr(args,'shard_index',None)
        start,count=qwen_dlm_generate.partition(args.benchmark,index)
        ids=[row['task_id'] for row in rows[start:start+count]]
        shard={'index':index,'count':8,'start_q':start,'num_q':count,'full_cell_n':COUNTS[args.benchmark]}
    artifacts=[]
    for name,root in [('runtime',runtime),('inputs',inputs),('model',model)]:artifacts+=inventory(root,name)
    score_args=json.loads(args.score_args_json)
    if not isinstance(score_args,list) or not all(isinstance(x,str) for x in score_args):raise ValueError('Scorer args must be a JSON string list')
    generation={'driver':'llada','script':args.runtime_script,'args':[
        '--runtime','{runtime}','--script',args.runtime_script,'--engine','{runtime}/'+args.engine_subdir,
        '--model','{model}','--config','{inputs}/'+str(args.decoder_config),
        '--problems','{inputs}/'+args.problems,'--benchmark',args.benchmark,
        '--seed',str(args.seed),'--max-new','4096','--out','{output}']}
    if family=='dg':generation={'driver':'dg','script':args.runtime_script,'args':[
        '--runtime','{runtime}','--script',args.runtime_script,'--problems','{inputs}/'+args.problems,
        '--benchmark',args.benchmark,'--seed',str(args.seed),'--out','{output}','--base',args.base]}
    if family=='qwen-dlm':generation={'driver':'qwen-dlm','script':args.runtime_script,'args':[
        '--runtime','{runtime}','--script',args.runtime_script,'--engine','{runtime}/'+args.engine_subdir,
        '--model','{model}','--config','{inputs}/'+args.decoder_config,
        '--problems','{inputs}/'+args.problems,'--benchmark',args.benchmark,'--role',args.role,
        '--seed',str(args.seed),'--shard-index',str(shard['index']),'--out','{output}']}
    recipe={'schema_version':1,'protocol_id':args.role+'-'+args.benchmark+'-main-v1-s'+str(args.seed),
            'scope':'shard' if shard else 'smoke_only' if args.smoke else 'full_cell',
            'historical_binding':'provided frozen runtime/config/weights',
            'expected_task_ids':ids,'artifacts':artifacts,'stages':{
                'generate':generation,
                'score':{'script':args.score_script,'image':args.scorer_image,'args':score_args}}}
    if shard:
        recipe.update(shard=shard,server_seed=1234,decoder_randtok_seed=12345)
        recipe['protocol_id']=args.role+'-'+args.benchmark+'-qwen-dlm-v1-shard'+str(shard['index'])+'-server1234-noise12345'
    return recipe


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role',choices=('llada-base','llada-orl','dg-base','dg-orl','qwen-dlm-base','qwen-dlm-orl'),required=True)
    parser.add_argument('--benchmark',choices=tuple(COUNTS),required=True)
    parser.add_argument('--seed',type=int,default=12345)
    for name in ('runtime-root','input-root','model-root'):parser.add_argument('--'+name,type=Path,required=True)
    for name in ('runtime-script','problems','score-script','scorer-image','score-args-json'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--engine-subdir',default='.')
    parser.add_argument('--decoder-config')
    parser.add_argument('--server-manifest')
    parser.add_argument('--base',default='http://127.0.0.1:8000')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--shard-index',type=int,help='Required for QwenDLM: one of its eight fixed fresh-server partitions')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.out.exists():raise ValueError('Refusing to overwrite recipe')
    recipe=build(args);args.out.write_text(json.dumps(recipe,indent=2)+'\n')
    print(json.dumps({'protocol_id':recipe['protocol_id'],'scope':recipe['scope'],
                      'artifacts':len(recipe['artifacts']),'task_count':len(recipe['expected_task_ids'])}))


if __name__=='__main__':main()
