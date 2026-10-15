"""Exercise the actual migrated clients with inert CPU-only services/engines."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from inference.adapters import diffusion_gemma, llada

ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def diffusion_server():
    calls=[]
    class Handler(BaseHTTPRequestHandler):
        requests=0
        def do_GET(self):
            data=(f'vllm:diffusion_num_denoising_steps_total {self.requests*49}\n'
                  f'vllm:diffusion_num_committed_tokens_total {self.requests*256}\n').encode()
            self.send_response(200);self.end_headers();self.wfile.write(data)
        def do_POST(self):
            request=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            calls.append(request);type(self).requests+=1
            data=json.dumps({'choices':[{'message':{'content':'inert fixture'},'finish_reason':'stop'}],
                             'usage':{'completion_tokens':7}}).encode()
            self.send_response(200);self.end_headers();self.wfile.write(data)
        def log_message(self,*args):pass
    http=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=http.serve_forever,daemon=True);thread.start()
    try:yield 'http://127.0.0.1:'+str(http.server_port),calls
    finally:http.shutdown();http.server_close();thread.join()


class ClientIntegrationTests(unittest.TestCase):
    def test_actual_dg_clients_through_adapter(self):
        with tempfile.TemporaryDirectory() as temp,diffusion_server() as (base,calls):
            root=Path(temp)
            for benchmark,script,cap in [('gsm8k','vllm_paper4.py',4096),('mbpp','vllm_full_extra_client.py',4096)]:
                problems=root/(benchmark+'.jsonl')
                problems.write_text(json.dumps({'task_id':'inert/0','messages':[{'role':'user','content':'inert'}],'gold':{'tests':['not executed']}})+'\n')
                out=root/benchmark
                argv=['dg','--runtime',str(ROOT/'inference/clients'),'--script',script,'--problems',str(problems),
                      '--out',str(out),'--benchmark',benchmark,'--seed','1234','--base',base]
                with patch.object(sys,'argv',argv):diffusion_gemma.main()
                row=json.loads((out/benchmark/'shard_w0.jsonl').read_text())
                self.assertEqual(row['raw'],'inert fixture')
                self.assertEqual(row['fwd_denoise'],48)
                self.assertEqual(row['fwd_total'],49)
                self.assertEqual(row['gold'],{'tests':['not executed']})
                self.assertEqual(calls[-1]['max_tokens'],cap)
                self.assertEqual(calls[-1]['seed'],1234)

    def test_actual_llada_client_with_inert_engine(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);engine=root/'engine';imports=engine/'inference/sglang'
            (imports/'sglang').mkdir(parents=True)
            (imports/'sglang/__init__.py').write_text('''class Engine:
    def __init__(self,**kwargs):
        assert kwargs['mem_fraction_static']==.60
        assert kwargs['sdar_rollout_cuda_graph_capture_sizes']==[1,2]
        assert kwargs['disable_radix_cache'] is True
    def generate(self,prompt,params):
        assert params=={'max_new_tokens':4096,'temperature':0.}
        return {'text':'inert fixture','meta_info':{'finish_reason':{'type':'stop'},'completion_tokens':7,
                'sglang_sdar_request_metrics':[{'total_forwards':5,'denoise_forwards':4,'commit_forwards':1}]}}
    def shutdown(self): pass
''')
            (imports/'transformers.py').write_text('''class AutoTokenizer:
    @classmethod
    def from_pretrained(cls,*args,**kwargs): return cls()
    def apply_chat_template(self,*args,**kwargs): return 'inert rendered prompt'
''')
            problems=root/'inputs.jsonl';problems.write_text('{"task_id":"inert/0","prompt":"fixture"}\n')
            config=root/'decoder.yaml';config.write_text('temperature: 0\n')
            argv=['llada','--runtime',str(ROOT/'inference/clients'),'--script','multi_gen.py',
                  '--engine',str(engine),'--model',str(root/'model'),'--config',str(config),'--problems',str(problems),
                  '--out',str(root/'out'),'--benchmark','he','--seed','12345','--max-new','4096']
            with patch.object(sys,'argv',argv):llada.main()
            row=json.loads((root/'out/he/shard_w0.jsonl').read_text())
            self.assertEqual(row['raw'],'inert fixture')
            self.assertEqual((row['fwd_denoise'],row['fwd_total']),(4,5))


if __name__=='__main__':unittest.main()
