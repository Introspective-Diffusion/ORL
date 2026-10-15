"""Opt-in real container check; uses only a trusted inert fixture.

Set PAPER_EVAL_CONTAINER_IMAGE to a pre-pulled Python image@sha256 digest.
PAPER_EVAL_CONTAINER_ENGINE defaults to docker; podman is also supported.
For apptainer, also set PAPER_EVAL_APPTAINER_IMAGE to the SIF pulled from that digest.
No image download or daemon startup is attempted. A skipped test is not
evidence that container execution or isolation works on this platform.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCORER = '''import errno,json,os,socket
from pathlib import Path

assert os.geteuid() == 65534
assert 'PAPER_EVAL_PRIVATE_SENTINEL' not in os.environ
assert not Path('/runtime/unlisted-private.txt').exists()
assert not Path('/inputs/unlisted-private.txt').exists()
assert json.loads(Path('/inputs/records.jsonl').read_text()) == {'raw':'inert fixture'}
interfaces = sorted(name for _,name in socket.if_nameindex())
flags = {name:int((Path('/sys/class/net')/name/'flags').read_text(),16) for name in interfaces}
assert 'lo' in flags and flags['lo'] & 1
# Docker Desktop can retain DOWN tunnel devices in an isolated net namespace.
assert not any(value & 1 for name,value in flags.items() if name != 'lo'), flags
assert all(row.split()[0] == 'lo' for row in Path('/proc/net/route').read_text().splitlines()[1:])
assert all(row.split()[-1] == 'lo' for row in Path('/proc/net/ipv6_route').read_text().splitlines())
readonly = []
for name in ('/runtime/scorer.py','/inputs/records.jsonl','/rootfs-write-probe'):
    try:
        with open(name,'a') as stream:
            stream.write('unexpected write')
    except OSError as error:
        assert error.errno in (errno.EACCES,errno.EPERM,errno.EROFS)
        readonly.append(name)
    else:
        raise AssertionError('Read-only boundary failed: '+name)
with socket.socket() as sock:
    sock.settimeout(2)
    try:
        sock.connect(('192.0.2.1',443))
    except OSError as error:
        network_error = type(error).__name__
    else:
        raise AssertionError('Network unexpectedly connected')
Path('/output/acceptance.json').write_text(json.dumps({
    'uid':os.geteuid(),'read_only_paths':readonly,
    'external_connect_failed':True,'network_error':network_error,'interfaces':interfaces,
    'active_interfaces':[name for name,value in flags.items() if value & 1],
    'generated_code_executed':False,'fixture':'trusted_inert_container_acceptance'}))
'''


@unittest.skipUnless(os.environ.get('PAPER_EVAL_CONTAINER_IMAGE'),
                     'Requires an explicitly supplied pre-pulled immutable Python image')
class ContainerAcceptanceTests(unittest.TestCase):
    def test_real_scoring_container_with_inert_fixture(self):
        engine=os.environ.get('PAPER_EVAL_CONTAINER_ENGINE','docker')
        if engine not in ('docker','podman','apptainer'):
            self.fail('PAPER_EVAL_CONTAINER_ENGINE must be docker, podman or apptainer')
        sif=['--apptainer-image',os.environ.get('PAPER_EVAL_APPTAINER_IMAGE','')] if engine=='apptainer' else []
        if sif and not sif[1]:
            self.fail('PAPER_EVAL_APPTAINER_IMAGE must name the SIF pulled from PAPER_EVAL_CONTAINER_IMAGE')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve()
            runtime=root/'runtime';inputs=root/'inputs';model=root/'model'
            for folder in (runtime,inputs,model):folder.mkdir()
            (runtime/'scorer.py').write_text(SCORER)
            (inputs/'records.jsonl').write_text('{"raw":"inert fixture"}\n')
            for folder in (runtime,inputs):
                (folder/'unlisted-private.txt').write_text('must not enter container\n')
            artifacts=[]
            for name,path in (('runtime',runtime/'scorer.py'),('inputs',inputs/'records.jsonl')):
                artifacts.append({'root':name,'path':path.name,
                                  'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            recipe={'schema_version':1,'protocol_id':'trusted-container-acceptance-v1',
                    'artifacts':artifacts,'stages':{'score':{'script':'scorer.py',
                    'image':os.environ['PAPER_EVAL_CONTAINER_IMAGE'],'args':[]}}}
            recipe_path=root/'recipe.json';recipe_path.write_text(json.dumps(recipe))
            output=root/'attempt'
            result=subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'reproduce.py'),
                'score','--recipe',str(recipe_path),'--runtime-root',str(runtime),
                '--input-root',str(inputs),'--model-root',str(model),'--output-root',str(output),
                '--container-engine',engine,*sif,'--timeout','120','--execute'],
                env=dict(os.environ,PAPER_EVAL_PRIVATE_SENTINEL='inert-sentinel'),
                capture_output=True,text=True,timeout=180)
            log=(output/'stage.log').read_text() if (output/'stage.log').exists() else ''
            self.assertEqual(result.returncode,0,result.stdout+result.stderr+log)
            receipt=json.loads((output/'execution_receipt.json').read_text())
            self.assertEqual(receipt['exit_code'],0)
            proof=json.loads((output/'artifacts/acceptance.json').read_text())
            self.assertEqual(proof['uid'],65534)
            self.assertTrue(proof['external_connect_failed'])
            self.assertEqual(proof['active_interfaces'],['lo'])
            self.assertEqual(len(proof['read_only_paths']),3)
            self.assertFalse(proof['generated_code_executed'])


if __name__=='__main__':unittest.main()
