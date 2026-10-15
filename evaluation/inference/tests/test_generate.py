from contextlib import contextmanager, redirect_stderr
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from inference.generate import endpoint, generate, main, payload


@contextmanager
def server(redirect=False):
    received=[]
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append((self.path,json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
            if redirect:
                self.send_response(302);self.send_header('Location','http://external.invalid/');self.end_headers()
            else:
                data=json.dumps({'text':'inert generated text','meta_info':{'completion_tokens':3}}).encode()
                self.send_response(200);self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def log_message(self,*args):pass
    http=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    worker=threading.Thread(target=http.serve_forever,daemon=True);worker.start()
    try:yield 'http://127.0.0.1:'+str(http.server_port),received
    finally:http.shutdown();http.server_close();worker.join()


class HttpGenerationTests(unittest.TestCase):
    def test_raw_prompt_request_has_only_explicit_sampling(self):
        with server() as (base,received),patch.dict('os.environ',{'HTTP_PROXY':'http://external.invalid:9999'}):
            row=generate({'id':'x','prompt':'raw prompt'},base_url=base,api='sglang',sampling_params={'max_new_tokens':8})
        self.assertEqual(received,[('/generate',{'text':'raw prompt','sampling_params':{'max_new_tokens':8}})])
        self.assertEqual(row['response']['text'],'inert generated text')
        self.assertGreater(row['elapsed_ns'],0)

    def test_chat_messages_and_model_are_explicit(self):
        item={'messages':[{'role':'user','content':'inert'}]}
        result=payload(item,'openai-chat',{'max_tokens':8},'served-model')
        self.assertEqual(result,{'messages':item['messages'],'model':'served-model','stream':False,'max_tokens':8})
        self.assertEqual(endpoint('http://localhost:8000/v1','openai-chat'),'http://localhost:8000/v1/chat/completions')
        with self.assertRaises(ValueError):payload(item,'openai-chat',{'max_tokens':8})

    def test_batch_is_serial_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as temp,server() as (base,received):
            inputs=Path(temp)/'inputs.jsonl'
            inputs.write_text('{"id":"a","prompt":"one"}\n{"id":"b","prompt":"two"}\n')
            out=Path(temp)/'out.jsonl'
            argv=['--api','sglang','--base-url',base,'--input',str(inputs),'--sampling-params','{"max_new_tokens":8}','--output',str(out)]
            main(argv)
            self.assertEqual([row[1]['text'] for row in received],['one','two'])
            self.assertEqual([json.loads(line)['id'] for line in out.read_text().splitlines()],['a','b'])
            with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):main(argv)
            self.assertEqual(len(received),2)

    def test_invalid_later_input_is_rejected_before_any_request(self):
        with tempfile.TemporaryDirectory() as temp,patch('inference.generate.generate') as request:
            inputs=Path(temp)/'inputs.jsonl'
            inputs.write_text('{"id":"a","prompt":"one"}\n{"id":"a","prompt":"two"}\n')
            with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                main(['--api','sglang','--input',str(inputs),'--sampling-params','{"max_new_tokens":8}'])
            request.assert_not_called()

    def test_remote_credentials_redirect_and_reserved_fields_rejected(self):
        for url in ('http://remote.invalid','http://user:pass@localhost','https://localhost','http://localhost/?token=x'):
            with self.assertRaises(ValueError):endpoint(url,'sglang')
        for params in ({},{'model':'other'},{'temperature':float('nan')}):
            with self.assertRaises(ValueError):payload({'prompt':'inert'},'sglang',params)
        with server(redirect=True) as (base,received),self.assertRaises(HTTPError):
            generate({'prompt':'inert'},base_url=base,api='sglang',sampling_params={'max_new_tokens':8})
        self.assertEqual(len(received),1)


if __name__=='__main__':unittest.main()
