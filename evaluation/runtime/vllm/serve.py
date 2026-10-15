#!/usr/bin/env python3
"""Prepare and optionally run a loopback paper DG server in a patched image."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, ProxyHandler, build_opener

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from repro.eval.portable.build_recipe import inventory


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        raise HTTPError(request.full_url, code, 'Server health redirects are disabled', headers, fp)


def plan(model, role, benchmark, output, port=8000):
    if role not in ('base', 'orl') or not 1 <= port <= 65535:
        raise ValueError('Invalid role or port')
    registry = json.loads((ROOT / 'configs/paper/decoding/dg.yaml').read_text())
    if benchmark not in registry['roles'][role]['max_new_tokens']:
        raise ValueError('Unknown benchmark')
    model = model.resolve()
    resolved_output = output.resolve()
    if resolved_output == model or resolved_output.is_relative_to(model) or model.is_relative_to(resolved_output):
        raise ValueError('Server output must be separate from model inputs')
    files = inventory(model, 'model')
    model_digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    config = json.loads((model / 'generation_config.json').read_text())
    decoder = registry['roles'][role]['decoder']
    overrides = {key: value for key, value in registry['decoder_common'].items()
                 if key not in ('canvas_length', 'max_denoising_steps')}
    overrides.update({key: value for key, value in decoder.items() if key != 'sampler_patch'})
    config = copy.deepcopy(config)
    for key, value in overrides.items():
        if isinstance(value, dict):
            config[key] = dict(config.get(key, {}), **value)
        else:
            config[key] = value
    config['max_new_tokens'] = None
    if role == 'base' and config.get('temperature_mode') == 'adaptive_scalar':
        raise ValueError('Base model contains an adaptive sampler configuration')
    context = registry['engine']['benchmark_context_length'].get(benchmark, 8192)
    diffusion = {key: registry['decoder_common'][key] for key in ('canvas_length', 'max_denoising_steps')}
    output = output.absolute()
    command = ['vllm', 'serve', str(model), '--served-model-name', 'dg', '--host', '127.0.0.1',
               '--port', str(port), '--tensor-parallel-size', '1', '--max-num-seqs', '1',
               '--max-model-len', str(context), '--gpu-memory-utilization', '0.95',
               '--attention-backend', 'TRITON_ATTN', '--no-enable-prefix-caching',
               '--generation-config', str(output / 'config'), '--diffusion-config', json.dumps(diffusion)]
    receipt = {'schema_version': 1, 'family': 'dg', 'role': role,
               'image_digest': registry['engine']['image'], 'image_identity': 'declared; verify externally',
               'model_manifest_sha256': model_digest, 'diffusion_config': diffusion,
               'max_model_len': context, 'seed_protocol': 'reset_each_request',
               'sampler_role': decoder['sampler_patch'], 'sampling_overrides': overrides,
               'gpu_memory_utilization': 0.95, 'attention_backend': 'TRITON_ATTN',
               'base_url': 'http://127.0.0.1:' + str(port), 'server_ready': False}
    return command, config, receipt


def validate_patch(path, role):
    source = path.read_bytes()
    text = source.decode()
    if 'HF-reference sampling' not in text:
        raise ValueError('Use the pinned, HF-aligned DG image')
    adaptive = 'adaptive_h_ref' in text and 'adaptive_scalar=gen.get' in text
    if adaptive != (role == 'orl'):
        raise ValueError('Use separate Base and ORL sampler images')
    return hashlib.sha256(source).hexdigest()


def terminate(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def serve(command, receipt, output, startup_timeout, source):
    receipt['sampler_source_sha256'] = validate_patch(source, receipt['role'])
    opener = build_opener(ProxyHandler({}), NoRedirect())
    from urllib.parse import urlsplit
    port = urlsplit(receipt['base_url']).port
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', port))
    process = None
    previous = {}
    def stop(signum, frame):
        raise KeyboardInterrupt
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.signal(sig, stop)
        with (output / 'server.log').open('xb') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + startup_timeout
            while True:
                if process.poll() is not None:
                    raise RuntimeError('Server exited before becoming healthy; inspect server.log')
                try:
                    with opener.open(receipt['base_url'] + '/health', timeout=2) as response:
                        ready = response.status == 200
                except OSError:
                    ready = False
                if ready:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError('Server startup timed out')
                time.sleep(1)
            receipt.update(server_ready=True, server_started=True, server_pid=process.pid)
            (output / 'server_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
            print(json.dumps({'server_ready': True, 'receipt': str(output / 'server_receipt.json')}), flush=True)
            status = process.wait()
            if status:
                raise RuntimeError('Server exited with status ' + str(status))
    finally:
        try:
            if process:
                terminate(process)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            (output / 'server_exit.json').write_text(json.dumps({'server_ready': False,
                'exit_code': process.returncode if process else None}) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--role', choices=('base', 'orl'), required=True)
    parser.add_argument('--benchmark', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--startup-timeout', type=int, default=900)
    parser.add_argument('--sampler-source', type=Path, default=Path(
        '/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/diffusion_gemma.py'))
    args = parser.parse_args()
    if args.startup_timeout <= 0:
        parser.error('Startup timeout must be positive')
    command, config, receipt = plan(args.model_root, args.role, args.benchmark, args.out, args.port)
    if args.out.exists() or args.out.is_symlink():
        raise ValueError('Use a fresh output directory')
    args.out.mkdir(parents=True)
    (args.out / 'config').mkdir()
    (args.out / 'config/generation_config.json').write_text(json.dumps(config, indent=2) + '\n')
    (args.out / 'launch_plan.json').write_text(json.dumps({'command': command, 'receipt': receipt}, indent=2) + '\n')
    print(json.dumps({'dry_run': not args.execute, 'command': command}))
    if args.execute:
        serve(command, receipt, args.out, args.startup_timeout, args.sampler_source)


if __name__ == '__main__':
    main()
