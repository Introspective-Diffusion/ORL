import base64
import collections
import csv
import hashlib
import json
from pathlib import Path
import pickle
import re
import tempfile
import unittest
import zlib

from repro.eval import build_transfer_inputs as build

EVALUATION = Path(__file__).resolve().parents[3]


class BuildTransferInputsTests(unittest.TestCase):
    def test_bundled_files_match_recorded_hashes(self):
        for name in ('mmlu_pro.jsonl', 'ifeval.jsonl'):
            data = (build.BUNDLED / name).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), build.EXPECTED[name])

    def test_hashes_agree_with_generation_doc_and_checksums(self):
        document = (EVALUATION / 'repro/paper/TRANSFER_GENERATION.md').read_text()
        table = dict(re.findall(r'^\| (\w+) \| \d+ \| ([0-9a-f]{64}) \|', document, re.M))
        for benchmark in ('aime', 'lcb', 'gpqa', 'mmlu_pro', 'ifeval'):
            self.assertEqual(table[benchmark], build.EXPECTED[benchmark + '.jsonl'])
        sums = dict(line.split()[::-1] for line in (EVALUATION / 'inputs/SHA256SUMS').read_text().splitlines())
        for name in ('mmlu_pro.jsonl', 'ifeval.jsonl'):
            self.assertEqual(sums['transfer/' + name], build.EXPECTED[name])
        for _, revision, _ in (build.MMLU_PRO_SOURCE, build.IFEVAL_SOURCE):
            self.assertIn(revision, document)

    def test_exact_existing_output_is_idempotent(self):
        data = (build.BUNDLED / 'mmlu_pro.jsonl').read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertTrue(build.write_verified(root, 'mmlu_pro.jsonl', data))
            self.assertTrue(build.write_verified(root, 'mmlu_pro.jsonl', data))
            (root / 'mmlu_pro.jsonl').write_bytes(data + b'\n')
            self.assertFalse(build.write_verified(root, 'mmlu_pro.jsonl', data))

    def test_private_tests_decode_without_globals(self):
        cases = [{'input': '1\n', 'output': '2\n', 'testtype': 'stdin'}]
        raw = base64.b64encode(zlib.compress(pickle.dumps(json.dumps(cases)))).decode()
        self.assertEqual(build._private_tests(raw), cases)
        hostile = base64.b64encode(zlib.compress(pickle.dumps(collections.OrderedDict()))).decode()
        self.assertEqual(build._private_tests(hostile), [])
        self.assertEqual(build._private_tests(''), [])

    def test_gpqa_order_is_one_permutation_per_question(self):
        self.assertEqual(len(build.GPQA_ORDER), 198)
        for order in build.GPQA_ORDER:
            self.assertEqual(sorted(order), ['0', '1', '2', '3'])

    def test_gpqa_rows_follow_the_stored_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'gpqa_diamond.csv'
            with open(source, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=('Question',) + build.GPQA_COLUMNS)
                writer.writeheader()
                for index in range(len(build.GPQA_ORDER)):
                    writer.writerow({'Question': f'question {index}\n',
                                     **{column: f'{column} {index}' for column in build.GPQA_COLUMNS}})
            rows = [json.loads(line) for line in build._gpqa_jsonl(source).decode().split('\n') if line]
            with open(source, 'a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(['extra'] * 5)
            with self.assertRaises(RuntimeError):
                build._gpqa_jsonl(source)
        self.assertEqual(len(rows), 198)
        for index, (row, order) in enumerate(zip(rows, build.GPQA_ORDER)):
            self.assertEqual(row['task_id'], f'gpqa/{index}')
            self.assertEqual(row['gold'], 'ABCD'[order.index('0')])
            options = '\n'.join(f'{letter}) {build.GPQA_COLUMNS[int(column)]} {index}'
                                for letter, column in zip('ABCD', order))
            self.assertEqual(row['messages'], [{'role': 'user', 'content': f'{build.GPQA_INSTRUCTION}\n\n'
                                                                           f'question {index}\n\n\n{options}'}])


if __name__ == '__main__':
    unittest.main()
