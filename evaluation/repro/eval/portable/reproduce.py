#!/usr/bin/env python3
"""Pinned local generation, isolated scoring, and aggregation entrypoints.

Dry-run by default. The caller supplies licensed inputs, weights, and runtime.
No downloads, cluster access, or implicit dataset substitutions are performed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import uuid


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def contained(root,relative):
    relative=Path(relative)
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('Artifact path must be contained and relative')
    path=(root/relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():raise ValueError('Missing or escaping artifact')
    return path


def validate(recipe,roots):
    if recipe.get('schema_version')!=1 or not recipe.get('protocol_id'):raise ValueError('Missing versioned protocol')
    if not recipe.get('artifacts'):raise ValueError('An immutable artifact inventory is required')
    covered=set()
    for item in recipe['artifacts']:
        root=item['root'];path=contained(roots[root],item['path'])
        if sha256(path)!=item['sha256']:raise ValueError('Artifact SHA256 mismatch: '+item['path'])
        covered.add((root,item['path']))
    for stage in ('generate','score'):
        command=recipe['stages'].get(stage)
        if not command:continue
        if ('runtime',command['script']) not in covered:raise ValueError('Stage script is not checksum-pinned')
    return covered


def expand(value,roots):
    if not isinstance(value,str):raise ValueError('Command arguments must be strings')
    for name,root in roots.items():value=value.replace('{'+name+'}',str(root))
    if re.search(r'\{(?:runtime|inputs|model|output)\}',value):raise ValueError('Unresolved path placeholder')
    return value


def stage_scoring_artifacts(recipe,roots,destination):
    staged={name:destination/name for name in ('runtime','inputs')}
    for root in staged.values():root.mkdir(parents=True,exist_ok=False)
    for item in recipe['artifacts']:
        name=item['root']
        if name not in staged:continue
        source=contained(roots[name],item['path'])
        target=staged[name]/item['path'];target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        if sha256(target)!=item['sha256']:raise ValueError('Artifact changed during isolated staging')
        target.chmod(0o444)
    # Container UID65534 must be able to traverse staged trees despite host umask077.
    for root in staged.values():
        root.chmod(0o755)
        for path in root.rglob('*'):
            if path.is_dir():path.chmod(0o755)
    return staged


# Enforced through a systemd user scope: the host must delegate memory, pids and cpu controllers.
APPTAINER_LIMITS=('MemoryMax=8G','MemorySwapMax=0','TasksMax=128')


def score_cpus(config):
    # LCB runs 8 test workers with 6 s per-test timeouts; fewer CPUs turn passing solutions into timeouts.
    cpus=config.get('cpus',2)
    if type(cpus) is not int or not 1<=cpus<=64:raise ValueError('Scoring CPU count must be an integer in [1,64]')
    return cpus


def apptainer_image(path,image,platform=None):
    path=Path(path).resolve()
    if not path.is_file():raise ValueError('Supply a local SIF pulled from the recipe image digest')
    deffile=subprocess.run(['apptainer','inspect','--deffile',str(path)],capture_output=True,text=True,check=True).stdout
    header={key.strip().lower():value.strip() for key,value in
            (line.split(':',1) for line in deffile.splitlines() if ':' in line)}
    def reference(value):
        name,digest=value.rsplit('@',1)
        return name.removeprefix('docker.io/').removeprefix('library/')+'@'+digest
    source=header.get('from','')
    if header.get('bootstrap')!='docker' or '@' not in source or reference(source)!=reference(image):
        raise ValueError('SIF was not pulled from the recipe image digest')
    if platform:
        layout=subprocess.run(['apptainer','sif','list',str(path)],capture_output=True,text=True,check=True).stdout
        if 'System/'+platform.split('/')[1]+')' not in layout:raise ValueError('SIF platform differs from the recipe')
    return {'path':str(path),'sha256':sha256(path),'source':source}


def apptainer_command(config,roots,mapped,image,unit,work):
    if any(re.search('[:,]',str(roots[name])) for name in mapped):
        raise ValueError('Apptainer bind sources cannot contain ":" or ","')
    argv=['systemd-run','--user','--scope','--quiet','--collect','--unit='+unit,
          *['--property='+limit for limit in (*APPTAINER_LIMITS,f'CPUQuota={100*score_cpus(config)}%')],'--',
          'prlimit','--cpu=600:600','--fsize=1073741824:1073741824','--',
          'unshare','--user','--map-user=65534','--map-group=65534','--net','--',
          'apptainer','exec','--userns','--containall','--no-home','--no-mount','bind-paths,hostfs',
          '--net','--network','none','--workdir',str(work),'--pwd','/tmp']
    for name in mapped:
        argv += ['--bind',f'{roots[name]}:/{name}'+('' if name=='output' else ':ro')]
    argv += [str(Path(image).resolve()),config.get('python','python3'),'/runtime/'+config['script']]
    return argv+[expand(a,mapped) for a in config.get('args',[])]


def command(recipe,stage,roots,python,container_engine,apptainer_sif=None,unit=None,work=None):
    config=recipe['stages'][stage]
    if stage=='generate':
        script=contained(roots['runtime'],config['script'])
        if config.get('driver') in ('llada','dg','qwen-dlm'):
            adapter={'llada':'llada','dg':'diffusion_gemma','qwen-dlm':'qwen_dlm'}[config['driver']]
            script=Path(__file__).resolve().parents[3]/'inference'/'adapters'/(adapter+'.py')
        elif config.get('driver') is not None:raise ValueError('Unknown generation driver')
        argv=[python,str(script)]
        argv += [expand(a,roots) for a in config.get('args',[])]
        env={k:expand(v,roots) for k,v in config.get('env',{}).items()}
        if any(re.search('TOKEN|SECRET|PASSWORD|CREDENTIAL|AWS_',k,re.I) for k in env):
            raise ValueError('Credentials are not part of a reproduction recipe')
        return argv,env
    image=config['image']
    if not re.fullmatch(r'[^\s]+@sha256:[a-f0-9]{64}',image):raise ValueError('Scoring requires an immutable image digest')
    mapped={name:Path('/'+name) for name in ('runtime','inputs','output')}
    if container_engine=='apptainer':
        if config.get('platform') not in (None,'linux/amd64'):raise ValueError('Unsupported frozen scoring platform')
        if apptainer_sif is None or unit is None or work is None:
            raise ValueError('Apptainer scoring requires a local SIF, scope unit and work directory')
        return apptainer_command(config,roots,mapped,apptainer_sif,unit,work),{}
    argv=[container_engine,'run','--rm','--pull=never','--network=none','--read-only',
          '--cap-drop=ALL','--security-opt=no-new-privileges','--pids-limit=128',
          '--memory=8g',f'--cpus={score_cpus(config)}','--user=65534:65534','--ipc=private',
          '--ulimit=cpu=600:600','--ulimit=fsize=1073741824:1073741824',
          '--tmpfs=/tmp:rw,nosuid,nodev,size=1g','--workdir=/tmp']
    if config.get('platform'):
        if config['platform'] != 'linux/amd64':raise ValueError('Unsupported frozen scoring platform')
        argv += ['--platform=linux/amd64']
    for name in mapped:
        option='' if name=='output' else ',readonly'
        argv += ['--mount',f'type=bind,src={roots[name]},dst=/{name}'+option]
    argv += [image,config.get('python','python3'),'/runtime/'+config['script']]
    argv += [expand(a,mapped) for a in config.get('args',[])]
    return argv,{}


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=('generate','score','aggregate'))
    ap.add_argument('--recipe',type=Path,required=True)
    ap.add_argument('--runtime-root',type=Path,required=True)
    ap.add_argument('--input-root',type=Path,required=True)
    ap.add_argument('--model-root',type=Path,required=True)
    ap.add_argument('--output-root',type=Path,required=True)
    ap.add_argument('--python',default=sys.executable)
    ap.add_argument('--container-engine',choices=('podman','docker','apptainer'),default='podman')
    ap.add_argument('--apptainer-image',type=Path,help='Local SIF pulled from the recipe image digest (apptainer only)')
    ap.add_argument('--execute',action='store_true')
    ap.add_argument('--timeout',type=int,default=7200)
    args=ap.parse_args(argv)
    if not 1<=args.timeout<=86400:raise ValueError('Timeout must be in [1,86400] seconds')
    roots={name:getattr(args,argument).resolve() for name,argument in
           [('runtime','runtime_root'),('inputs','input_root'),('model','model_root'),('output','output_root')]}
    if any(roots['output']==roots[k] or roots['output'].is_relative_to(roots[k]) or roots[k].is_relative_to(roots['output'])
           for k in ('runtime','inputs','model')):raise ValueError('Output must be separate from all read-only source roots')
    recipe=json.loads(args.recipe.read_text());validate(recipe,roots)
    execution_roots=dict(roots,output=roots['output']/'artifacts')
    if args.stage=='score':
        execution_roots.update({name:roots['output']/'pinned'/name for name in ('runtime','inputs')})
    if args.stage=='aggregate':
        config=recipe['stages']['aggregate']
        argv=[sys.executable,str(Path(__file__).with_name('evaluate.py')),'aggregate',
              '--manifest',str(contained(roots['inputs'],config['manifest'])),'--input-root',str(roots['inputs'])]
        env={}
    container_name='eval-'+uuid.uuid4().hex if args.stage=='score' else None
    apptainer=args.stage=='score' and args.container_engine=='apptainer'
    if args.stage!='aggregate':
        argv,env=command(recipe,args.stage,execution_roots,args.python,args.container_engine,
                         args.apptainer_image,container_name,roots['output']/'apptainer_work')
    if container_name and not apptainer:argv[2:2]=['--name',container_name]
    receipt={'protocol_id':recipe['protocol_id'],'recipe_sha256':sha256(args.recipe),
             'stage':args.stage,'dry_run':not args.execute,'command':argv,'environment':env,
             'historical_binding':recipe.get('historical_binding','')}
    print(json.dumps(receipt,indent=2),flush=True)
    if not args.execute:return
    if roots['output'].exists():raise ValueError('Refusing an existing output directory; every attempt needs a new directory')
    if apptainer:
        config=recipe['stages']['score']
        receipt['apptainer_image']=apptainer_image(args.apptainer_image,config['image'],config.get('platform'))
    execution_roots['output'].mkdir(parents=True)
    if args.stage=='score':
        execution_roots['output'].chmod(0o777)
        stage_scoring_artifacts(recipe,roots,roots['output']/'pinned')
    if apptainer:(roots['output']/'apptainer_work').mkdir()
    (roots['output']/'execution_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    clean_env={k:os.environ[k] for k in ('PATH','LANG','SYSTEMROOT') if k in os.environ}
    if args.stage=='score':
        # These configure only the host engine; they are never passed or mounted into the container.
        clean_env.update({k:os.environ[k] for k in ('HOME','XDG_RUNTIME_DIR') if k in os.environ})
    clean_env.update(env)
    with (roots['output']/'stage.log').open('w') as log:
        process=subprocess.Popen(argv,env=clean_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        timeout=args.timeout
        if args.stage=='generate' and recipe['stages']['generate'].get('driver')=='qwen-dlm':timeout=min(timeout,6600)
        try:status=process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            receipt['timed_out']=True
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait()
            status=124
            if container_name:
                remove=(['systemctl','--user','kill','--signal=SIGKILL',container_name+'.scope'] if apptainer
                        else [args.container_engine,'rm','--force',container_name])
                cleanup=subprocess.run(remove,env=clean_env,stdout=log,stderr=subprocess.STDOUT,timeout=30,check=False)
                receipt['container_cleanup_exit_code']=cleanup.returncode
    receipt['exit_code']=status
    (roots['output']/'execution_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    raise SystemExit(status)


if __name__=='__main__':main()
