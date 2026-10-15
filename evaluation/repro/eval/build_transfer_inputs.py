"""Rebuild the transfer benchmark inputs and check each file against the SHA256 the paper's runs used.

AIME, LiveCodeBench and GPQA are rebuilt from pinned Hugging Face revisions; MMLU-Pro and IFEval are copied from the
bundled inputs/transfer/. A file is written only when its bytes match the recorded hash. GPQA is gated: accept its terms
at https://huggingface.co/datasets/Idavidrein/gpqa and log in with `hf auth login` first. Needs network access and the
`datasets`, `numpy` and `huggingface_hub` packages.
"""
import argparse
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import pickle
import shutil
import sys
import zlib

NEWLINE = b'\n'
BUNDLED = Path(__file__).resolve().parents[2] / 'inputs' / 'transfer'
EXPECTED = {
    'aime.jsonl': 'ec677f2990d5a0dd78e4d631d0dca03ea29f9e0c8103141be6f065cd031018b5',
    'lcb.jsonl': '26f01f74d4b0e8d74d7f97f482a2456b181c3bc3f289677831399b82dc0c3b8e',
    'lcb_v6_tests.json': 'd3b2da8f72842657dbc6809b76acfbcaaf7505b7cc4ea6c5a4f51afd40e0ac04',
    'gpqa.jsonl': '7cf55ae09e627820d492577114d0b7838082869f004df5fe48b875002af2034c',
    'mmlu_pro.jsonl': '8f10a6eb35f6279f2a0a30bf4631551d8c12b5c55a2f8d245662fb624c4902cb',
    'ifeval.jsonl': '69c2d38e1f5b0e22376f7f3f9184de9ccbdf5e2061fb74bb4c55541fb8f3ca60',
}

# These inputs are bundled because their licenses permit redistribution, but
# their exact upstream revisions are still part of the generation contract.
MMLU_PRO_SOURCE = ('TIGER-Lab/MMLU-Pro', 'b189ec765aa7ed75c8acfea42df31fdae71f97be', 'test')
IFEVAL_SOURCE = ('google-research/google-research', '08a8d6736475776f42ffac23b2c13111a28e5795',
                 'instruction_following_eval/data/input_data.jsonl')

# AIME 2024 and 2025, 30 problems each, shuffled together once with a fixed seed.
AIME_SOURCES = (('HuggingFaceH4/aime_2024', '2fe88a2f1091d5048c0f36abc874fb997b3dd99a', 'train', 'aime24'),
                ('math-ai/aime25', '563bb8404243c5f09de6ec262f2db674fe5bce9b', 'test', 'aime25'))
AIME_INSTRUCTION = " Let's think step by step and output the final answer within \\boxed{}."
AIME_SHUFFLE_SEED = 20260909

# LiveCodeBench release v6: the 175 problems added after release v5, with public and private tests.
LCB_SOURCE = ('livecodebench/code_generation_lite', '0fe84c3912ea0c4d4a78037083943e8f0c4dd505', 'test6.jsonl')
LCB_SYSTEM = ('You are an expert Python programmer. You will be given a question (problem specification) and will '
              'generate a correct Python program that matches the specification and passes all tests. You will NOT '
              'return anything except for the program.')
LCB_FORMAT_STARTER = ('You will use the following starter code to write the solution to the problem and enclose your '
                      'code within delimiters.')
LCB_FORMAT_STDIN = ('Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test '
                    'on the sample inputs). Enclose your code within delimiters as follows.')

# GPQA Diamond must not be shared in plain text, so only its option order is kept here: for each question, in file
# order, the CSV columns shown as A-D (0 is the correct answer, 1-3 are incorrect answers 1-3).
GPQA_SOURCE = ('Idavidrein/gpqa', '83022cefff930aea54f654c0b282e74b9eeda5c6', 'gpqa_diamond.csv')
GPQA_COLUMNS = ('Correct Answer', 'Incorrect Answer 1', 'Incorrect Answer 2', 'Incorrect Answer 3')
GPQA_INSTRUCTION = ("Answer the following multiple choice question. The last line of your response should be of the "
                    "following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by "
                    "step before answering.")
GPQA_ORDER = ('0132 1320 1302 2013 0312 1023 0312 1230 2103 1032 1230 0123 0213 0123 1203 0321 2031 0321 3210 3201 '
              '2103 2103 3120 3102 2031 3210 2031 0231 1203 2013 1203 1302 0321 1032 2301 2301 3210 2031 0213 0213 '
              '0231 3102 0231 0231 0312 2130 2103 0231 1203 0312 0123 2130 3201 1023 0213 2013 0132 1023 3210 1203 '
              '1032 2130 2103 2103 2310 1302 2103 0132 2301 2031 1230 0123 0132 0321 3120 0123 3201 0132 0312 0213 '
              '1302 0231 0213 2031 1032 3210 0213 0132 3102 2103 0231 3201 3120 1230 3012 0132 2301 1302 2103 0213 '
              '3012 1203 3102 3021 3021 2103 1302 0312 1230 0132 3120 3102 3012 0123 2031 0213 0312 1320 3120 3021 '
              '3120 0321 2310 0213 1302 2031 0213 3210 0312 1023 1230 1203 1032 0213 1302 2301 2130 1230 0231 2031 '
              '0321 1032 3201 0123 0213 1023 1320 0312 2013 1302 3201 1320 3012 2013 0132 3210 0321 2103 3120 0132 '
              '3021 2031 0123 1023 0132 3201 2103 3201 2013 3210 3120 0312 3201 2031 3201 2310 0123 0321 3021 2130 '
              '0132 1302 0321 0321 1032 2301 2031 0231 2103 3012 2310 3201 1023 2301 1302 1032 0213 1023').split()


def build_aime():
    from datasets import load_dataset
    import numpy as np
    problems = []
    for repo, revision, split, year in AIME_SOURCES:
        for index, row in enumerate(load_dataset(repo, split=split, revision=revision)):
            problems.append((row['problem'], str(row['answer']), repo, f'{year}-{index:02d}', year))
    rows = []
    for position, pick in enumerate(np.random.RandomState(AIME_SHUFFLE_SEED).permutation(len(problems))):
        problem, gold, repo, qid, year = problems[pick]
        meta = {'index': str(position), 'split': 'test', 'source': repo, 'qid': qid, 'year': year, 'rep': '0'}
        rows.append({'task_id': 'aime/' + qid, 'messages': [{'role': 'user', 'content': problem + AIME_INSTRUCTION}],
                     'gold': gold, 'meta': meta})
    return {'aime.jsonl': ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows).encode()}


class _NoGlobals(pickle.Unpickler):
    """LCB stores private tests as a pickled JSON string; refuse anything that would need a class or function."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f'refusing {module}.{name}')


def _private_tests(raw):
    if not raw:
        return []
    try:
        return json.loads(_NoGlobals(io.BytesIO(zlib.decompress(base64.b64decode(raw.encode('utf-8'))))).load())
    except Exception:
        try:
            return json.loads(raw)
        except Exception:
            return []


def _lcb_prompt(item):
    question, starter = item['question_content'], (item.get('starter_code') or '').strip()
    if starter:
        return (f'### Question:\n{question}\n\n### Format: {LCB_FORMAT_STARTER}\n```python\n{starter}\n```\n\n'
                '### Answer: (use the provided format with backticks)\n\n')
    return (f'### Question:\n{question}\n\n### Format: {LCB_FORMAT_STDIN}\n```python\n# YOUR CODE HERE\n```\n\n'
            '### Answer: (use the provided format with backticks)\n\n')


def build_lcb():
    from huggingface_hub import hf_hub_download
    repo, revision, filename = LCB_SOURCE
    source = hf_hub_download(repo, filename, repo_type='dataset', revision=revision)
    rows, tests = [], {}
    for line in open(source, encoding='utf-8'):
        if not line.strip():
            continue
        item = json.loads(line)
        meta = json.loads(item['metadata']) if isinstance(item['metadata'], str) else (item['metadata'] or {})
        fn_name = meta.get('func_name')
        public = item['public_test_cases']
        cases = list(json.loads(public) if isinstance(public, str) else public)
        cases += _private_tests(item.get('private_test_cases'))
        task_id = item['question_id']
        tests[task_id] = {'inputs': [c['input'] for c in cases], 'outputs': [c['output'] for c in cases],
                          'fn_name': fn_name}
        rows.append({'task_id': task_id, 'entry_point': fn_name,
                     'messages': [{'role': 'system', 'content': LCB_SYSTEM},
                                  {'role': 'user', 'content': _lcb_prompt(item)}],
                     'meta': {'fn_name': fn_name, 'difficulty': item['difficulty'], 'platform': item['platform'],
                              'contest_date': item['contest_date'][:10], 'n_tests': len(cases)}})
    return {'lcb.jsonl': ''.join(json.dumps(row) + '\n' for row in rows).encode(),
            'lcb_v6_tests.json': json.dumps(tests).encode()}


def _gpqa_jsonl(source):
    with open(source, newline='', encoding='utf-8') as f:
        questions = list(csv.DictReader(f))
    if len(questions) != len(GPQA_ORDER):
        raise RuntimeError(f'{source}: {len(questions)} questions, expected {len(GPQA_ORDER)}')
    rows = []
    for index, (question, order) in enumerate(zip(questions, GPQA_ORDER)):
        options = '\n'.join(f'{letter}) {question[GPQA_COLUMNS[int(column)]]}' for letter, column in zip('ABCD', order))
        content = f'{GPQA_INSTRUCTION}\n\n{question["Question"]}\n\n{options}'
        rows.append({'task_id': f'gpqa/{index}', 'messages': [{'role': 'user', 'content': content}],
                     'gold': 'ABCD'[order.index('0')]})
    return ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows).encode()


def build_gpqa():
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import GatedRepoError
    repo, revision, filename = GPQA_SOURCE
    try:
        source = hf_hub_download(repo, filename, repo_type='dataset', revision=revision)
    except GatedRepoError:
        raise RuntimeError(f'{repo} is gated: accept its terms at https://huggingface.co/datasets/{repo} and log in '
                           'with `hf auth login`') from None
    return {'gpqa.jsonl': _gpqa_jsonl(source)}


def copy_bundled(name):
    return lambda: {name: (BUNDLED / name).read_bytes()}


BUILDERS = {'aime': build_aime, 'lcb': build_lcb, 'gpqa': build_gpqa,
            'mmlu_pro': copy_bundled('mmlu_pro.jsonl'), 'ifeval': copy_bundled('ifeval.jsonl')}


def write_verified(root, name, data):
    digest = hashlib.sha256(data).hexdigest()
    if digest != EXPECTED[name]:
        print(f'{name}: sha256 {digest} != {EXPECTED[name]}; not written', file=sys.stderr)
        return False
    target = root / name
    if target.exists():
        if target.is_file() and target.read_bytes() == data:
            print(f'{name}: existing file has the expected SHA256 -> {target}')
            return True
        print(f'{name}: {target} exists with different content; not overwritten', file=sys.stderr)
        return False
    target.write_bytes(data)
    size = f'{data.count(NEWLINE)} rows' if name.endswith('.jsonl') else f'{len(data)} bytes'
    print(f'{name}: {size}, sha256 ok -> {target}')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='Directory for the inputs; existing files are kept')
    parser.add_argument('--benchmark', action='append', choices=sorted(BUILDERS),
                        help='Repeat to select several; default all')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    failed = False
    for benchmark in args.benchmark or sorted(BUILDERS):
        try:
            outputs = BUILDERS[benchmark]()
        except RuntimeError as error:
            print(f'{benchmark}: {error}', file=sys.stderr)
            failed = True
            continue
        for name, data in outputs.items():
            if not write_verified(args.out, name, data):
                failed = True
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
