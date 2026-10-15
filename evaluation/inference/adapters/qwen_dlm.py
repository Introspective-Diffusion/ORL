#!/usr/bin/env python3
"""Offline generation-only adapter for one exact QwenDLM main-paper shard.

Supply trusted, checksum-pinned runtime, model, decoder and frozen input roots.
No model download, historical scorer import, or candidate execution occurs here.
"""
import argparse
import ast
import hashlib
import json
import os
import socket
from pathlib import Path
import subprocess
import sys

HARNESS_COMMIT = 'f2230a5c2fcca72440d6c3fea4b50441b7d7516c'
ENGINE_COMMIT = 'b974f47818b5c8e6ab855da8f19e1cad0c92464f'
SOURCES = {
    'eval_client.py':'811dd2abac9f64a8d036baae008ed8c7d55cbe8402d4886765bb7a7c2429870f',
    'run_eval.py':'b04b4ebf54ce9d003b8d42125342217b07cc063b1c0ce819c4cf3fc9b597ae92',
    'benchmarks.py':'5240fa8858b6890b81dd94014515d2bff9366d32aed4b3cf8922c20ba8b9a72a',
}
COUNTS = {'gsm8k':1319,'he':164,'math500':500,'mbpp':427}
INPUT_SHA = {
    'gsm8k':'40805b714b72e65dde956b352960af7f608138a58db13abb519c3d245386784c',
    'math500':'2d084c9d3ec66b364633531bd1519f38f7f331115c0198ccc87151d83f8cae32',
    'mbpp':'b5443f461deca1546253c5fda84dc01e6fb9486b6510832a39c135474520939a',
    'he':'0f0ea89ab42f2debe39317b97bb5441431985da8a2854858539455f0f7012db0',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':')).encode()


def partition(benchmark,index):
    if type(index) is not int or not 0 <= index < 8:
        raise ValueError('QwenDLM requires an explicit shard index in [0,7]')
    size,remainder = divmod(COUNTS[benchmark],8)
    return index*size+min(index,remainder),size+(index<remainder)


def token_cap(role,benchmark):
    return 49152 if benchmark=='mbpp' or (role=='qwen-dlm-base' and benchmark in ('gsm8k','math500')) else 32768


def decoder(role,benchmark):
    result = dict(block_size=256,denoising_steps=256,confidence_temperature=1.0,
                  noise_mode='random_vocab',random_token_high=151669,deterministic_randtok=True,
                  randtok_seed=12345,allow_accepted_change=True,edit_threshold=.9,
                  force_accept_leftmost=True,post_accept_refine_steps=0,max_steps_per_block=256,
                  temperature=0.0,top_k=0,top_p=1.0,causal_prefill=True,self_conditioning_post_norm=True)
    if role=='qwen-dlm-base':
        threshold=.9 if benchmark=='he' else .85
        result.update(accept_rule='prob_threshold',prob_threshold=threshold,edit_threshold=threshold)
    elif role=='qwen-dlm-orl':
        result.update(accept_rule='entropy_budget',entropy_bound=.4 if benchmark in ('gsm8k','math500') else .6,
                      confidence_temperature_mode='step_linear',confidence_temperature_max=1.0,
                      confidence_temperature_min=.6,confidence_temperature_step_decrement=.01)
    else:
        raise ValueError('Unknown QwenDLM role')
    return result


def validate_inputs(path,benchmark):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    encoded = ''.join(json.dumps(row,sort_keys=True,ensure_ascii=True)+'\n' for row in rows).encode()
    if sha(encoded)!=INPUT_SHA[benchmark] or len(rows)!=COUNTS[benchmark]:
        raise ValueError('Wrong QwenDLM frozen input identity; paper4 prompts are not interchangeable')
    ids = [row['task_id'] for row in rows]
    if len(ids)!=len(set(ids)) or any(row['meta']['task_id']!=row['task_id'] for row in rows):
        raise ValueError('Frozen task ID mismatch')
    return rows,encoded


def validate_runtime(runtime,script):
    if __package__:
        from ._paths import contained
    else:
        from _paths import contained
    runner = contained(runtime,str(script))
    sources = {}
    for name,expected in SOURCES.items():
        relative = str((Path(script).parent/name))
        path = contained(runtime,relative)
        payload = path.read_bytes()
        if sha(payload)!=expected:
            raise ValueError('Locked QwenDLM source SHA256 mismatch: '+name)
        sources[name]=payload.decode()
    if runner.name!='run_eval.py':
        raise ValueError('QwenDLM runtime-script must name the pinned run_eval.py')
    return sources


def generation_only(source):
    # Same fail-closed transformation used by the validated cluster generation client.
    old = '    correct = bool(benchmark_obj.score(text, item))'
    if source.count(old)!=1:
        raise ValueError('Locked client shape changed; refuse to run')
    source=source.replace(old,'    correct = None  # Deferred isolated scoring; never execute candidate code here.')
    old='        "correct": correct,'
    if source.count(old)!=1:
        raise ValueError('Locked client result shape changed')
    source=source.replace(old,old+'\n        "scoring_status": "not_run",\n        "source_metadata": item.get("meta", {}),')
    source=source.replace('status = "OK" if row["correct"] else "X"','status = "UNSCORED"')
    if any(isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='score' for node in ast.walk(ast.parse(source))):
        raise ValueError('Scoring remains in generation client')
    return source


def frozen_benchmarks(source):
    # Extract only the pinned pure regex helper; never import the historical scorer module.
    tree=ast.parse(source)
    function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='extract_gsm8k_answer')
    helper=ast.get_source_segment(source,function)
    return '''import json, os, re
class _Frozen:
    def load(self, n, start=0):
        with open(os.environ['QWEN_DLM_FROZEN_INPUTS']) as stream:
            rows=[json.loads(line) for line in stream if line.strip()]
        return rows[start:start+n]
class GSM8K(_Frozen): name='gsm8k'
class MATH500(_Frozen): name='math500'
class HumanEval(_Frozen): name='humaneval'
class MBPP(_Frozen): name='mbpp'
''' + helper+'\n'


def prepare(args):
    import yaml
    sources=validate_runtime(args.runtime,args.script)
    rows,encoded=validate_inputs(args.problems,args.benchmark)
    expected_config=decoder(args.role,args.benchmark)
    if yaml.safe_load(args.config.read_text())!=expected_config:
        raise ValueError('Decoder config differs from the exact QwenDLM cell, including noise seed12345')
    if args.seed!=1234:
        raise ValueError('QwenDLM server seed must be1234, distinct from decoder noise seed12345')
    start,count=partition(args.benchmark,args.shard_index)
    args.out.mkdir(parents=True,exist_ok=True)
    if any(args.out.iterdir()):
        raise ValueError('Refusing an occupied output directory')
    runtime_root=args.runtime.resolve()
    engine=args.engine.resolve()
    if not engine.is_relative_to(runtime_root) or not (engine/'inference/sglang').is_dir():
        raise ValueError('Engine must be a contained supplied pinned QwenDLM runtime')
    if not (args.model/'config.json').is_file():
        raise ValueError('Supply a local model; downloads are disabled')
    staging=args.out/'generation_runtime';staging.mkdir()
    inputs=args.out/'frozen_inputs.ndjson';inputs.write_bytes(encoded)
    config=args.out/'decoder.yaml';config.write_bytes(args.config.read_bytes())
    bench='humaneval' if args.benchmark=='he' else args.benchmark
    task=dict(id=f"{'sft' if args.role=='qwen-dlm-base' else 'rct'}-{bench}-s{start:04d}",
              entity='sft' if args.role=='qwen-dlm-base' else 'rct',benchmark=bench,
              dataset_n=COUNTS[args.benchmark],max_new_tokens=token_cap(args.role,args.benchmark),
              seed=args.seed,shard_index=args.shard_index,start_q=start,num_q=count)
    settings=dict(task=task,decoder=expected_config,inputs_sha256=sha(encoded),harness_commit=HARNESS_COMMIT,
                  engine_commit=ENGINE_COMMIT,scoring_status='not_run')
    settings_hash=sha(canonical(settings))
    client=generation_only(sources['eval_client.py'])
    key='        "scoring_status": "not_run",'
    client=client.replace(key,key+'\n        "task_id": item["meta"]["task_id"],\n        "generation_settings_sha256": '+repr(settings_hash)+',\n        "generation_shard_id": '+repr(task['id'])+',')
    runner=sources['run_eval.py']
    needle='        "--skip-server-warmup",'
    if runner.count(needle)!=2 or runner.count('        "0.0.0.0",')!=2:
        raise ValueError('Locked runner command shape changed')
    runner=runner.replace(needle,needle+'\n        "--random-seed", os.environ["GENERATION_SEED"],')
    runner=runner.replace('        "0.0.0.0",','        "127.0.0.1",')
    shim=frozen_benchmarks(sources['benchmarks.py'])
    for name,source in [('run_eval.py',runner),('eval_client.py',client),('benchmarks.py',shim)]:
        ast.parse(source);(staging/name).write_text(source)
    settings.update(settings_sha256=settings_hash,scope='shard',shard_count=8,
                    input_task_ids=[row['task_id'] for row in rows[start:start+count]],
                    source_sha256=SOURCES,patched_source_sha256={name:sha((staging/name).read_bytes()) for name in SOURCES},
                    engine_identity_status='Expected source revision declared',
                    seed_note='Server seed 1234; decoder random-token seed 12345')
    (args.out/'generation_protocol.json').write_text(json.dumps(settings,indent=2)+'\n')
    cache=args.out/'cache';cache.mkdir()
    environment={key:os.environ[key] for key in ('PATH','LANG','SYSTEMROOT') if key in os.environ}
    environment.update(HOME=str(cache),HF_HOME=str(cache/'hf'),HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',
                       TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',
                       QWEN_DLM_FROZEN_INPUTS=str(inputs),GENERATION_SEED='1234',SGLANG_WORKERS='1',SGLANG_SAVE_FULL_TEXT='1',
                       SGLANG_FLASHINFER_PREFILL_BACKEND='fa3',SDAR_FUSED_ROW_STATS='1',SDAR_COMMIT_GRAPH='1',
                       TRITON_CACHE_DIR=str(cache/'triton'),TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'))
    command=[sys.executable,str(staging/'run_eval.py'),'--benchmark',bench,'--model',str(args.model),
             '--output-dir',str(args.out),'--algo-config',str(config),'--repo-root',str(engine),
             '--no-sif','--python',sys.executable,'--hf-cache',str(cache),
             '--start-q',str(start),'--num-q',str(count),'--num-shards','1','--gpu-offset','0',
             '--base-port',str(args.port),'--max-new-tokens',str(task['max_new_tokens']),
             '--context-length','65536','--max-running-requests','1','--cuda-graph',
             '--cuda-graph-capture-sizes','1','--health-timeout','900']
    return command,environment,settings


def verify_output(out,settings):
    task=settings['task'];bench=task['benchmark']
    path=out/bench/(bench+f"_s{task['start_q']:04d}.json")
    rows=json.loads(path.read_text())['results']
    frozen=[json.loads(line) for line in (out/'frozen_inputs.ndjson').read_text().splitlines()]
    if [row['task_id'] for row in rows]!=settings['input_task_ids']:
        raise ValueError('Incomplete or reordered shard output')
    normalized=[]
    for row in rows:
        if row['correct'] is not None or row['scoring_status']!='not_run' or row['generation_settings_sha256']!=settings['settings_sha256']:
            raise ValueError('Unexpected scoring or generation identity')
        if not isinstance(row.get('text'),str) or not row['text']:
            raise ValueError('Missing full generation text')
        index=row['idx']
        if not task['start_q'] <= index < task['start_q']+task['num_q'] or row['task_id']!=frozen[index]['task_id'] or row['gold']!=frozen[index]['gold'] or row['problem']!=frozen[index]['prompt']:
            raise ValueError('Output/input task binding mismatch')
        if row['finish_reason'].get('type') not in ('stop','length'):
            raise ValueError('Unknown generation finish status')
        normalized.append(dict(task_id=row['task_id'],raw=row['text'],gold=row['gold'],
                               finish=row['finish_reason']['type'],completion_tokens=row['completion_tokens'],
                               fwd_denoise=row['metrics']['denoise_forwards'],fwd_total=row['metrics']['total_forwards'],
                               wall_s=row['elapsed'],generation_settings_sha256=settings['settings_sha256']))
    (out/'records.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=True)+'\n' for row in normalized))
    (out/'GENERATION_COMPLETE.json').write_text(json.dumps({'scope':'shard','n':len(rows),'settings_sha256':settings['settings_sha256'],'scoring_run':False})+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('runtime','script','engine','model','config','problems','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--role',choices=('qwen-dlm-base','qwen-dlm-orl'),required=True)
    parser.add_argument('--benchmark',choices=tuple(COUNTS),required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--shard-index',type=int,required=True)
    parser.add_argument('--port',type=int,default=46000)
    args=parser.parse_args()
    if not 1024 <= args.port <= 65531:
        raise ValueError('Invalid loopback server port')
    for name in ('runtime','engine','model','config','problems','out'):
        setattr(args,name,getattr(args,name).resolve())
    for port in range(args.port,args.port+4):
        with socket.socket() as probe:
            try:probe.bind(('127.0.0.1',port))
            except OSError as error:raise ValueError('Requested loopback server port is occupied') from error
    command,environment,settings=prepare(args)
    subprocess.run(command,env=environment,check=True)
    verify_output(args.out,settings)


if __name__=='__main__':
    main()
