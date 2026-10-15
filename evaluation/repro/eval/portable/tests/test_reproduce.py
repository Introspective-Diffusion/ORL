import hashlib
from pathlib import Path
import tempfile
import unittest
import argparse
import json
import os
import subprocess
import sys
from unittest.mock import patch
from inference.adapters import diffusion_gemma as dg_generate
from repro.eval.portable.reproduce import apptainer_image,command,validate,stage_scoring_artifacts
from repro.eval.portable.build_recipe import build


class EntrypointTests(unittest.TestCase):
    def test_dg_mbpp_uses_original_427_client_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);problems=root/'mbpp.jsonl'
            problems.write_text('{"task_id":"1","messages":[{"role":"user","content":"fixture"}]}\n')
            argv=['dg_generate.py','--runtime',str(root),'--script','vllm_full_extra_client.py',
                  '--problems',str(problems),'--out',str(root/'out'),'--benchmark','mbpp','--seed','1234']
            with patch.object(sys,'argv',argv),patch.object(dg_generate.subprocess, 'run') as run:
                dg_generate.main()
            command=run.call_args.args[0]
            self.assertEqual(command[command.index('--benchmark')+1],'mbpp427')
            self.assertEqual(command[command.index('--max-denoising-steps')+1],'48')
            self.assertIn(str(root/'out/mbpp/shard_w0.jsonl'),command)

    def test_dg_nonlocal_endpoint_is_rejected(self):
        argv=['dg_generate.py','--runtime','.','--script','client.py','--problems','problems.jsonl',
              '--out','new-out','--benchmark','he','--seed','1234','--base','http://external.invalid']
        with patch.object(sys,'argv',argv),self.assertRaisesRegex(ValueError,'local HTTP'):
            dg_generate.main()

    def test_trusted_toy_generation_and_score_input_binding(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();runtime=root/'runtime';inputs=root/'inputs';model=root/'model'
            for p in (runtime,inputs,model):p.mkdir()
            (runtime/'multi_gen.py').write_text("import json,os,pathlib\np=json.load(open(os.environ['PLAN']))['datasets'][0]\nrows=[json.loads(x) for x in open(p['problems'])]\nout=pathlib.Path(p['out_dir'])/'shard_w0.jsonl'\nout.write_text(''.join(json.dumps(dict(task_id=r['task_id'],raw='inert fixture',finish='stop',completion_tokens=1,fwd_denoise=1,fwd_total=2,wall_s=1))+'\\n' for r in rows))\n")
            (runtime/'score.py').write_text('pass\n')
            (inputs/'problems.jsonl').write_text('{"task_id":"toy/0","prompt":"fixture"}\n')
            (inputs/'decoder.yaml').write_text('temperature: 0\nrandtok_seed: 12345\n')
            (model/'config.json').write_text('{}\n')
            args=argparse.Namespace(runtime_root=runtime,input_root=inputs,model_root=model,
                runtime_script='multi_gen.py',score_script='score.py',decoder_config='decoder.yaml',
                problems='problems.jsonl',benchmark='gsm8k',seed=12345,smoke=True,role='llada-base',
                engine_subdir='.',score_args_json='[]',scorer_image='fixture@sha256:'+'a'*64)
            recipe=build(args);self.assertEqual(recipe['scope'],'smoke_only')
            path=root/'recipe.json';path.write_text(json.dumps(recipe))
            package=Path(__file__).resolve().parents[1];output=root/'attempt'
            result=subprocess.run([sys.executable,str(package/'reproduce.py'),'generate','--recipe',str(path),
                '--runtime-root',str(runtime),'--input-root',str(inputs),'--model-root',str(model),
                '--output-root',str(output),'--execute'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr+result.stdout)
            self.assertTrue((output/'artifacts/gsm8k/shard_w0.jsonl').exists())
            scored=root/'score-recipe.json'
            bound=subprocess.run([sys.executable,str(package/'bind_score_inputs.py'),'--generation-recipe',str(path),
                '--records-root',str(output/'artifacts'),'--out',str(scored)],capture_output=True,text=True)
            self.assertEqual(bound.returncode,0,bound.stderr)
            self.assertNotIn('generate',json.loads(scored.read_text())['stages'])

    def test_timeout_records_failure_without_untrusted_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();runtime=root/'runtime';runtime.mkdir()
            script=runtime/'wait.py';script.write_text('import time\ntime.sleep(60)\n')
            recipe={'schema_version':1,'protocol_id':'trusted-timeout-fixture','artifacts':[
                {'root':'runtime','path':'wait.py','sha256':hashlib.sha256(script.read_bytes()).hexdigest()}],
                'stages':{'generate':{'script':'wait.py'}}}
            path=root/'recipe.json';path.write_text(json.dumps(recipe));output=root/'out'
            result=subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'reproduce.py'),'generate',
                '--recipe',str(path),'--runtime-root',str(runtime),'--input-root',str(root/'inputs'),
                '--model-root',str(root/'model'),'--output-root',str(output),'--timeout','1','--execute'],
                capture_output=True,text=True)
            self.assertEqual(result.returncode,124,result.stderr)
            self.assertTrue(json.loads((output/'execution_receipt.json').read_text())['timed_out'])

    def test_scoring_stages_only_pinned_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);runtime=root/'runtime';inputs=root/'inputs'
            runtime.mkdir();inputs.mkdir()
            (runtime/'score.py').write_text('pass\n');(runtime/'.env').write_text('SECRET=private\n')
            (inputs/'records.jsonl').write_text('{}\n');(inputs/'unlisted.txt').write_text('private\n')
            recipe={'artifacts':[{'root':name,'path':filename,'sha256':hashlib.sha256((folder/filename).read_bytes()).hexdigest()}
                    for name,folder,filename in [('runtime',runtime,'score.py'),('inputs',inputs,'records.jsonl')]]}
            staged=stage_scoring_artifacts(recipe,{'runtime':runtime,'inputs':inputs},root/'staged')
            self.assertEqual(sorted(p.name for p in staged['runtime'].iterdir()),['score.py'])
            self.assertEqual(sorted(p.name for p in staged['inputs'].iterdir()),['records.jsonl'])

    def test_generation_uses_pinned_script_without_shell(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();(root/'generate.py').write_text('print(1)\n')
            recipe={'schema_version':1,'protocol_id':'test',
                    'artifacts':[{'root':'runtime','path':'generate.py','sha256':hashlib.sha256(b'print(1)\n').hexdigest()}],
                    'stages':{'generate':{'script':'generate.py','args':['--model','{model}'],'env':{'TEMP':'0'}}}}
            roots={key:root for key in ('runtime','model','inputs','output')}
            validate(recipe,roots)
            argv,env=command(recipe,'generate',roots,'python3','podman')
            self.assertEqual(argv,['python3',str(root/'generate.py'),'--model',str(root)])
            self.assertEqual(env,{'TEMP':'0'})
            (root/'generate.py').write_text('print(2)\n')
            with self.assertRaisesRegex(ValueError,'SHA256'):validate(recipe,roots)

    def test_score_always_isolated_and_pinned(self):
        recipe={'stages':{'score':{'script':'score.py','image':'local/runtime@sha256:'+'a'*64,
                                  'args':['--in','{inputs}','--out','{output}']}}}
        roots={key:Path('/local')/key for key in ('runtime','model','inputs','output')}
        argv,env=command(recipe,'score',roots,'python3','podman')
        for option in ('--network=none','--read-only','--cap-drop=ALL','--user=65534:65534','--pids-limit=128','--cpus=2'):
            self.assertIn(option,argv)
        self.assertNotIn('/local/model',' '.join(argv));self.assertEqual(env,{})
        self.assertEqual(argv[-4:],['--in','/inputs','--out','/output'])
        recipe['stages']['score']['cpus']=8
        self.assertIn('--cpus=8',command(recipe,'score',roots,'python3','podman')[0])
        for cpus in (0,'8',True):
            recipe['stages']['score']['cpus']=cpus
            with self.subTest(cpus=cpus),self.assertRaisesRegex(ValueError,'CPU'):command(recipe,'score',roots,'python3','podman')
        recipe['stages']['score'].pop('cpus')
        recipe['stages']['score']['image']='local/runtime:latest'
        with self.assertRaisesRegex(ValueError,'immutable'):command(recipe,'score',roots,'python3','podman')

    def test_apptainer_score_isolated_without_daemon(self):
        recipe={'stages':{'score':{'script':'score.py','image':'local/runtime@sha256:'+'a'*64,'platform':'linux/amd64',
                                  'args':['--in','{inputs}','--out','{output}']}}}
        roots={key:Path('/local')/key for key in ('runtime','model','inputs','output')}
        argv,env=command(recipe,'score',roots,'python3','apptainer',Path('/local/scorer.sif'),'eval-test',Path('/local/work'))
        for option in ('--unit=eval-test','--property=MemoryMax=8G','--property=MemorySwapMax=0','--property=TasksMax=128',
                       '--property=CPUQuota=200%','--cpu=600:600','--fsize=1073741824:1073741824','--map-user=65534',
                       '--map-group=65534','--containall','--no-home','bind-paths,hostfs','none','/local/work',
                       '/local/runtime:/runtime:ro','/local/inputs:/inputs:ro','/local/output:/output'):
            self.assertIn(option,argv)
        self.assertNotIn('--name',argv);self.assertNotIn('/local/model',' '.join(argv));self.assertEqual(env,{})
        self.assertEqual(argv[-7:],['/local/scorer.sif','python3','/runtime/score.py','--in','/inputs','--out','/output'])
        recipe['stages']['score']['cpus']=8
        self.assertIn('--property=CPUQuota=800%',
                      command(recipe,'score',roots,'python3','apptainer',Path('/local/scorer.sif'),'eval-test',Path('/local/work'))[0])
        with self.assertRaisesRegex(ValueError,'local SIF'):command(recipe,'score',roots,'python3','apptainer')
        recipe['stages']['score']['platform']='linux/arm64'
        with self.assertRaisesRegex(ValueError,'platform'):
            command(recipe,'score',roots,'python3','apptainer',Path('/local/scorer.sif'),'eval-test',Path('/local/work'))

    def test_apptainer_image_bound_to_recipe_digest(self):
        digest='a'*64
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();sif=root/'scorer.sif';sif.write_bytes(b'inert fixture')
            fake=root/'apptainer'
            fake.write_text('#!/bin/sh\nif [ "$1" = inspect ]; then printf "bootstrap: docker\\nfrom: %s\\n" "$FAKE_SOURCE";'
                            ' else echo "4 |1 |NONE |0-1 |FS (Squashfs/*System/$FAKE_ARCH)"; fi\n')
            fake.chmod(0o755)
            fixture={'PATH':str(root)+os.pathsep+os.environ.get('PATH','/usr/bin:/bin'),
                     'FAKE_SOURCE':'docker.io/library/python@sha256:'+digest,'FAKE_ARCH':'amd64'}
            with patch.dict(os.environ,fixture):
                record=apptainer_image(sif,'python@sha256:'+digest,'linux/amd64')
                self.assertEqual(record['sha256'],hashlib.sha256(b'inert fixture').hexdigest())
                with self.assertRaisesRegex(ValueError,'digest'):apptainer_image(sif,'python@sha256:'+'b'*64)
                with self.assertRaisesRegex(ValueError,'digest'):apptainer_image(sif,'other/python@sha256:'+digest)
                with patch.dict(os.environ,{'FAKE_ARCH':'arm64'}):
                    with self.assertRaisesRegex(ValueError,'platform'):apptainer_image(sif,'python@sha256:'+digest,'linux/amd64')
                with patch.dict(os.environ,{'FAKE_SOURCE':'docker.io/library/python:3.11'}):
                    with self.assertRaisesRegex(ValueError,'digest'):apptainer_image(sif,'python@sha256:'+digest)
                with self.assertRaisesRegex(ValueError,'local SIF'):apptainer_image(root/'missing.sif','python@sha256:'+digest)

    def test_credential_environment_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'g.py').write_text('')
            recipe={'stages':{'generate':{'script':'g.py','env':{'HF_TOKEN':'not-allowed'}}}}
            with self.assertRaisesRegex(ValueError,'Credentials'):
                command(recipe,'generate',{'runtime':root},'python3','podman')


if __name__=='__main__':unittest.main()
