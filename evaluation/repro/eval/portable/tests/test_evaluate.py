import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from repro.eval.portable.evaluate import audit


class EvaluationTests(unittest.TestCase):
    def verdict_fixture(self,root,manifest,rows):
        payload=''.join(json.dumps(row)+'\n' for row in rows).encode()
        (root/'verdicts.jsonl').write_bytes(payload)
        manifest['evaluations'][0]['verdicts']={'path':'verdicts.jsonl',
            'sha256':hashlib.sha256(payload).hexdigest(),'scorer_protocol_id':'synthetic-scorer-v1'}

    def fixture(self,root):
        row={'task_id':'synthetic/0','finish':'stop','completion_tokens':20,'fwd_denoise':4,'fwd_total':5,'wall_s':2}
        payload=(json.dumps(row)+'\n').encode();(root/'rows.jsonl').write_bytes(payload)
        shard={'path':'rows.jsonl','sha256':hashlib.sha256(payload).hexdigest()}
        return {'schema_version':1,'evaluations':[{'id':'synthetic','protocol_id':'synthetic-v1',
                'expected_task_ids':['synthetic/0'],'generation_shards':[shard]}]}

    def test_aggregate_without_quality_claim(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);manifest=self.fixture(root);result=audit(manifest,root,True)['results'][0]
            self.assertEqual(result['m2t_tpf_mean_clean'],5)
            self.assertEqual(result['tps_pooled_clean'],10)
            self.assertFalse(result['quality_scored'])

    def test_hash_drift_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);manifest=self.fixture(root);(root/'rows.jsonl').write_text('{}\n')
            with self.assertRaises(ValueError):audit(manifest,root)

    def test_path_escape_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);manifest=self.fixture(root)
            manifest['evaluations'][0]['generation_shards'][0]['path']='../outside.jsonl'
            with self.assertRaises(ValueError):audit(manifest,root)

    def test_duplicate_coverage_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);manifest=self.fixture(root)
            manifest['evaluations'][0]['generation_shards']*=2
            with self.assertRaises(ValueError):audit(manifest,root)

    def test_duplicate_evaluation_ids_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);manifest=self.fixture(root);manifest['evaluations']*=2
            with self.assertRaisesRegex(ValueError,'Duplicate evaluation IDs'):audit(manifest,root)

    def test_empty_manifest_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError,'Nonempty evaluation list'):
                audit({'schema_version':1,'evaluations':[]},Path(temp))

    def test_check_validates_verdicts_without_aggregating_quality(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);manifest=self.fixture(root)
            self.verdict_fixture(root,manifest,[{'task_id':'synthetic/0','correct':True}])
            checked=audit(manifest,root)['results'][0]
            self.assertFalse(checked['quality_scored']);self.assertNotIn('accuracy',checked)
            self.assertEqual(audit(manifest,root,True)['results'][0]['accuracy'],1)
            (root/'verdicts.jsonl').write_text('{}\n')
            with self.assertRaisesRegex(ValueError,'SHA256'):audit(manifest,root)

    def test_check_rejects_invalid_verdict_contracts(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);manifest=self.fixture(root)
            for rows in ([],[{'task_id':'wrong/0','correct':True}],
                         [{'task_id':'synthetic/0','correct':1}],
                         [{'task_id':'synthetic/0','correct':True}]*2):
                with self.subTest(rows=rows):
                    self.verdict_fixture(root,manifest,rows)
                    with self.assertRaises(ValueError):audit(manifest,root)
            self.verdict_fixture(root,manifest,[{'task_id':'synthetic/0','correct':True}])
            del manifest['evaluations'][0]['verdicts']['scorer_protocol_id']
            with self.assertRaisesRegex(ValueError,'scorer_protocol_id'):audit(manifest,root)


if __name__=='__main__':unittest.main()
