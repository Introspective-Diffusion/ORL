import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from inference.adapters import qwen_dlm
from repro.eval.portable.build_recipe import build

CLIENT = '''def run_one(benchmark_obj, text, item):
    correct = bool(benchmark_obj.score(text, item))
    return {
        "correct": correct,
    }
'''
RUNNER = '''import argparse,json,os
from pathlib import Path
one=[
        "0.0.0.0",
        "--skip-server-warmup",
]
two=[
        "0.0.0.0",
        "--skip-server-warmup",
]
p=argparse.ArgumentParser()
for name in ('benchmark','output-dir','start-q','num-q'):p.add_argument('--'+name)
a,_=p.parse_known_args()
out=Path(a.output_dir)
protocol=json.loads((out/'generation_protocol.json').read_text())
items=[json.loads(line) for line in open(os.environ['QWEN_DLM_FROZEN_INPUTS'])]
start,count=int(a.start_q),int(a.num_q)
rows=[dict(idx=i,task_id=item['task_id'],gold=item['gold'],problem=item['prompt'],text='inert trusted fixture',correct=None,scoring_status='not_run',generation_settings_sha256=protocol['settings_sha256'],finish_reason={'type':'stop'},completion_tokens=1,metrics={'denoise_forwards':1,'total_forwards':2},elapsed=1.0) for i,item in enumerate(items) if start<=i<start+count]
(out/a.benchmark).mkdir()
(out/a.benchmark/(a.benchmark+f'_s{start:04d}.json')).write_text(json.dumps({'results':rows}))
'''
BENCHMARKS = '''raise RuntimeError('Original scorer module must never be imported')
def extract_gsm8k_answer(text):
    return "?"
'''


class QwenDlmTests(unittest.TestCase):
    def fixture(self,root):
        import yaml
        runtime=root/'runtime';inputs=root/'inputs';model=root/'model'
        for directory in (runtime,inputs,model):directory.mkdir()
        (runtime/'engine/inference/sglang').mkdir(parents=True)
        sources={'eval_client.py':CLIENT,'run_eval.py':RUNNER,'benchmarks.py':BENCHMARKS}
        for name,source in sources.items():(runtime/name).write_text(source)
        (runtime/'score.py').write_text("raise RuntimeError('Scorer must not run on host')\n")
        (model/'config.json').write_text('{}\n')
        rows=[dict(idx=i,task_id=f'trusted/{i}',meta={'task_id':f'trusted/{i}'},prompt='fixture',gold={'tests':['inert']}) for i in range(164)]
        data=''.join(json.dumps(row,sort_keys=True,ensure_ascii=True)+'\n' for row in rows).encode()
        problems=inputs/'he.jsonl';problems.write_bytes(data)
        config=inputs/'decoder.yaml';config.write_text(yaml.safe_dump(qwen_dlm.decoder('qwen-dlm-base','he')))
        args=argparse.Namespace(runtime=runtime,script=Path('run_eval.py'),engine=runtime/'engine',model=model,
                                config=config,problems=problems,out=root/'out',benchmark='he',role='qwen-dlm-base',seed=1234,shard_index=0,port=46000)
        hashes={name:hashlib.sha256(source.encode()).hexdigest() for name,source in sources.items()}
        return args,hashes,hashlib.sha256(data).hexdigest()

    def test_all_partitions_and_distinct_seeds(self):
        for bench,n in qwen_dlm.COUNTS.items():
            indices=[]
            for i in range(8):
                start,count=qwen_dlm.partition(bench,i);indices.extend(range(start,start+count))
            self.assertEqual(indices,list(range(n)))
            for role in ('qwen-dlm-base','qwen-dlm-orl'):
                self.assertEqual(qwen_dlm.decoder(role,bench)['randtok_seed'],12345)
        self.assertEqual(qwen_dlm.token_cap('qwen-dlm-base','gsm8k'),49152)
        self.assertEqual(qwen_dlm.token_cap('qwen-dlm-orl','gsm8k'),32768)
        with self.assertRaises(ValueError):qwen_dlm.partition('he',8)

    def test_trusted_fixture_generation_is_offline_and_never_imports_scorer(self):
        with tempfile.TemporaryDirectory() as temp:
            args,hashes,input_sha=self.fixture(Path(temp))
            with patch.dict(qwen_dlm.SOURCES,hashes),patch.dict(qwen_dlm.INPUT_SHA,he=input_sha),patch.dict(os.environ,{'AWS_SECRET_ACCESS_KEY':'inert','SGLANG_EXTRA_SERVER_ARGS':'--host remote.invalid'}):
                command,environment,settings=qwen_dlm.prepare(args)
            self.assertNotIn('AWS_SECRET_ACCESS_KEY',environment)
            self.assertNotIn('SGLANG_EXTRA_SERVER_ARGS',environment)
            self.assertEqual(environment['HF_HUB_OFFLINE'],'1')
            runner=(args.out/'generation_runtime/run_eval.py').read_text()
            self.assertNotIn('0.0.0.0',runner)
            self.assertEqual(runner.count('"127.0.0.1"'),2)
            self.assertNotIn('.score(', (args.out/'generation_runtime/eval_client.py').read_text())
            self.assertNotIn('raise RuntimeError',(args.out/'generation_runtime/benchmarks.py').read_text())
            self.assertEqual(settings['task']['seed'],1234)
            self.assertEqual(settings['decoder']['randtok_seed'],12345)
            subprocess.run(command,env=environment,check=True)
            qwen_dlm.verify_output(args.out,settings)
            rows=[json.loads(line) for line in (args.out/'records.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),21)
            self.assertEqual(rows[0]['raw'],'inert trusted fixture')
            self.assertEqual(json.loads((args.out/'GENERATION_COMPLETE.json').read_text())['scope'],'shard')
            native=args.out/'humaneval/humaneval_s0000.json'
            original=json.loads(native.read_text())
            for field,value in [('gold',{'tests':['different']}),('correct',True),('idx',163)]:
                changed=json.loads(json.dumps(original));changed['results'][0][field]=value
                native.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):qwen_dlm.verify_output(args.out,settings)
            native.write_text(json.dumps(original))

    def test_tampered_runtime_input_and_decoder_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            args,hashes,input_sha=self.fixture(Path(temp))
            with patch.dict(qwen_dlm.SOURCES,hashes),patch.dict(qwen_dlm.INPUT_SHA,he=input_sha):
                with self.assertRaises(ValueError):qwen_dlm.validate_inputs(args.problems,'gsm8k')
                args.config.write_text('randtok_seed: 1234\n')
                with self.assertRaisesRegex(ValueError,'Decoder'):qwen_dlm.prepare(args)
                (args.runtime/'eval_client.py').write_text(CLIENT+'\n# changed\n')
                with self.assertRaisesRegex(ValueError,'source SHA256'):qwen_dlm.prepare(args)
        with self.assertRaises(ValueError):qwen_dlm.generation_only('print("not the locked client")')

    def test_recipe_is_shard_not_full_cell(self):
        with tempfile.TemporaryDirectory() as temp:
            a,hashes,input_sha=self.fixture(Path(temp))
            args=argparse.Namespace(runtime_root=a.runtime,input_root=a.problems.parent,model_root=a.model,
                runtime_script='run_eval.py',score_script='score.py',decoder_config='decoder.yaml',problems='he.jsonl',
                benchmark='he',seed=1234,smoke=False,role='qwen-dlm-base',engine_subdir='engine',shard_index=7,
                score_args_json='[]',scorer_image='fixture@sha256:'+'a'*64)
            with patch.dict(qwen_dlm.SOURCES,hashes),patch.dict(qwen_dlm.INPUT_SHA,he=input_sha):
                recipe=build(args)
                self.assertEqual(recipe['scope'],'shard')
                self.assertEqual(recipe['shard']['start_q'],144)
                self.assertEqual(recipe['expected_task_ids'],[f'trusted/{i}' for i in range(144,164)])
                self.assertEqual(recipe['stages']['generate']['driver'],'qwen-dlm')
                args.shard_index=None
                with self.assertRaises(ValueError):build(args)
                args.shard_index=0;args.smoke=True
                with self.assertRaises(ValueError):build(args)


if __name__=='__main__':unittest.main()
