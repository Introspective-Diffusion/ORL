#!/usr/bin/env python3
"""Send raw prompts or messages to an already running local generation server.

This client does not load models, choose a decoder, apply a model chat template,
or run generated programs. Sampling parameters must be supplied explicitly.
Use the paper adapters separately for frozen benchmark protocols.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

MAX_RESPONSE_BYTES = 32 * 1024 * 1024


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        raise HTTPError(request.full_url, code, 'Generation redirects are disabled', headers, fp)


def endpoint(base_url, api):
    url = urlsplit(base_url)
    if (url.scheme != 'http' or url.hostname not in ('127.0.0.1', 'localhost', '::1')
            or url.username or url.password or url.query or url.fragment):
        raise ValueError('Use a credential-free loopback HTTP server; remote endpoints are disabled')
    if url.port is not None and not 1 <= url.port <= 65535:
        raise ValueError('Invalid server port')
    path = url.path.rstrip('/')
    if api == 'sglang' and path == '':
        return base_url.rstrip('/') + '/generate'
    if api == 'openai-chat' and path in ('', '/v1'):
        return base_url.rstrip('/') + ('' if path else '/v1') + '/chat/completions'
    raise ValueError('Use an origin URL for SGLang, or an origin or /v1 URL for OpenAI chat')


def payload(item, api, sampling_params, model=None):
    if not isinstance(sampling_params, dict) or not sampling_params:
        raise ValueError('Supply an explicit nonempty sampling-parameter JSON object')
    if not all(isinstance(key, str) for key in sampling_params):
        raise ValueError('Sampling parameter names must be strings')
    if set(sampling_params) & {'text', 'prompt', 'messages', 'model', 'stream', 'sampling_params'}:
        raise ValueError('Sampling parameters cannot override request inputs, model, or streaming')
    json.dumps(sampling_params, allow_nan=False)
    if not isinstance(item, dict) or set(item) - {'id', 'prompt', 'messages'}:
        raise ValueError('Input rows accept only id and either prompt or messages')
    if ('prompt' in item) == ('messages' in item):
        raise ValueError('Each input must contain exactly one of prompt or messages')
    if 'prompt' in item and not isinstance(item['prompt'], str):
        raise ValueError('Prompt must be a string')
    if api == 'sglang':
        if 'prompt' not in item:
            raise ValueError('SGLang /generate requires an explicitly formatted raw prompt')
        return {'text': item['prompt'], 'sampling_params': dict(sampling_params)}
    if api != 'openai-chat' or not isinstance(model, str) or not model:
        raise ValueError('OpenAI chat requires an explicit served model name')
    messages = item.get('messages', [{'role': 'user', 'content': item.get('prompt')}])
    if (not isinstance(messages, list) or not messages
            or any(not isinstance(message, dict) or not isinstance(message.get('role'), str)
                   or not isinstance(message.get('content'), str) for message in messages)):
        raise ValueError('Messages must be a nonempty list of role/content string objects')
    return dict(sampling_params, model=model, messages=messages, stream=False)


def generate(item, *, base_url, api, sampling_params, model=None, timeout=1200):
    """Make one serial request with no retries; return unmodified JSON and latency."""
    if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('Timeout must be positive and finite')
    url = endpoint(base_url, api)
    body = json.dumps(payload(item, api, sampling_params, model), allow_nan=False).encode()
    request = Request(url, data=body, headers={'Content-Type': 'application/json'}, method='POST')
    # Ignore proxy credentials and prevent a local service redirecting prompts elsewhere.
    opener = build_opener(ProxyHandler({}), NoRedirect())
    start = time.perf_counter_ns()
    with opener.open(request, timeout=timeout) as response:
        encoded = response.read(MAX_RESPONSE_BYTES + 1)
        if len(encoded) > MAX_RESPONSE_BYTES:
            raise ValueError('Server response exceeds the client size limit')
        result = json.loads(encoded)
    elapsed_ns = time.perf_counter_ns() - start
    if not isinstance(result, dict):
        raise ValueError('Expected a JSON object from the generation server')
    return {'id': item.get('id', '0'), 'api': api, 'elapsed_s': elapsed_ns / 1e9,
            'elapsed_ns': elapsed_ns, 'response': result}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--api', choices=('sglang', 'openai-chat'), required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000', help='Already running local server')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--prompt', help='Raw SGLang prompt; one user message for OpenAI chat')
    source.add_argument('--messages-json', help='Explicit OpenAI chat messages JSON array')
    source.add_argument('--input', type=Path, help='JSONL rows: {id, prompt} or {id, messages}')
    parser.add_argument('--sampling-params', required=True, help='Explicit JSON object; no decoder defaults are inferred')
    parser.add_argument('--model', help='Required served model name for OpenAI chat')
    parser.add_argument('--timeout', type=float, default=1200)
    parser.add_argument('--output', type=Path, help='New JSONL file; existing files are never overwritten')
    args = parser.parse_args(argv)
    try:
        params = json.loads(args.sampling_params)
        if args.input:
            rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
        elif args.messages_json is not None:
            rows = [{'messages': json.loads(args.messages_json)}]
        else:
            rows = [{'prompt': args.prompt}]
        if not rows:
            raise ValueError('Input batch is empty')
        endpoint(args.base_url, args.api)
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            raise ValueError('Timeout must be positive and finite')
        ids = set()
        for index, item in enumerate(rows):
            payload(item, args.api, params, args.model)
            item.setdefault('id', str(index))
            if not isinstance(item['id'], str) or item['id'] in ids:
                raise ValueError('Input IDs must be unique strings')
            ids.add(item['id'])
        stream = args.output.open('x') if args.output else sys.stdout
        try:
            for item in rows:
                result = generate(item, base_url=args.base_url, api=args.api, sampling_params=params,
                                  model=args.model, timeout=args.timeout)
                stream.write(json.dumps(result, ensure_ascii=True) + '\n')
                stream.flush()
        finally:
            if args.output:
                stream.close()
    except (ValueError, OSError, URLError) as error:
        parser.exit(2, f'Generation failed: {error}\n')


if __name__ == '__main__':
    main()
