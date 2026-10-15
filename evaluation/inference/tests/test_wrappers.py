import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from inference.adapters import diffusion_gemma as dg
from inference.adapters import llada
from inference.adapters._paths import contained


class WrapperTests(unittest.TestCase):
    def test_dg_mbpp_uses_original_427_client_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);problems=root/'mbpp.jsonl'
            problems.write_text('{"task_id":"1","messages":[{"role":"user","content":"fixture"}]}\n')
            argv=['diffusion_gemma.py','--runtime',str(root),'--script','client.py',
                  '--problems',str(problems),'--out',str(root/'out'),'--benchmark','mbpp','--seed','1234']
            with patch.object(sys,'argv',argv),patch.object(dg.subprocess,'run') as run:
                dg.main()
            command=run.call_args.args[0]
            self.assertEqual(command[command.index('--benchmark')+1],'mbpp427')
            self.assertEqual(command[command.index('--max-denoising-steps')+1],'48')
            self.assertIn(str(root/'out/mbpp/shard_w0.jsonl'),command)

    def test_dg_nonlocal_endpoint_is_rejected(self):
        argv=['diffusion_gemma.py','--runtime','.','--script','client.py','--problems','problems.jsonl',
              '--out','new-out','--benchmark','he','--seed','1234','--base','http://external.invalid']
        with patch.object(sys,'argv',argv),self.assertRaisesRegex(ValueError,'local HTTP'):
            dg.main()

    def test_llada_trusted_fixture_retains_original_environment_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);runtime=root/'runtime';runtime.mkdir()
            script=runtime/'multi_gen.py'
            script.write_text("import json,os,pathlib\np=json.load(open(os.environ['PLAN']))['datasets'][0]\nassert os.environ['MEM_FRAC']=='0.60'\nassert os.environ['ATTN_BACKEND']=='flashinfer'\nassert os.environ['ALGO']=='SDARPrefixKV'\nrows=[json.loads(x) for x in open(p['problems'])]\n(pathlib.Path(p['out_dir'])/'shard_w0.jsonl').write_text(''.join(json.dumps(dict(task_id=r['task_id'],raw='inert fixture'))+'\\n' for r in rows))\n")
            problems=root/'problems.jsonl';problems.write_text('{"task_id":"toy/0","prompt":"inert"}\n')
            config=root/'decoder.yaml';config.write_text('temperature: 1\nrandtok_seed: 7\ncustom: retained\n')
            argv=['llada.py','--runtime',str(runtime),'--script','multi_gen.py','--engine',str(root/'engine'),
                  '--model',str(root/'model'),'--config',str(config),'--problems',str(problems),
                  '--out',str(root/'out'),'--benchmark','gsm8k','--seed','12345','--max-new','4096']
            with patch.object(sys,'argv',argv):llada.main()
            import yaml
            decoder=yaml.safe_load((root/'out/decoder.yaml').read_text())
            self.assertEqual(decoder,{'temperature':0.,'randtok_seed':12345,'custom':'retained'})
            plan=json.loads((root/'out/plan.json').read_text())
            self.assertEqual(plan['datasets'][0]['max_new'],4096)
            self.assertEqual(json.loads((root/'out/gsm8k/shard_w0.jsonl').read_text())['raw'],'inert fixture')

    def test_containment_rejects_traversal_and_escaping_symlink(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'runtime';root.mkdir()
            outside=root.parent/'outside.py';outside.write_text('inert')
            (root/'link.py').symlink_to(outside)
            for path in ('../outside.py',str(outside),'link.py','missing.py'):
                with self.assertRaises(ValueError):contained(root,path)


if __name__=='__main__':unittest.main()
