#!/usr/bin/env python3
"""Offline, protocol-explicit validation and aggregation of evaluation records."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def read_asset(root,record):
    relative=Path(record['path'])
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('Artifact paths must be relative and contained')
    path=(root/relative).resolve()
    if not path.is_relative_to(root.resolve()):raise ValueError('Artifact path escapes input root')
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=record['sha256']:raise ValueError('Artifact SHA256 mismatch: '+str(relative))
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def audit(manifest,root,aggregate=False):
    if manifest.get('schema_version')!=1:raise ValueError('Unsupported manifest version')
    evaluations=manifest['evaluations']
    if not isinstance(evaluations,list) or not evaluations:raise ValueError('Nonempty evaluation list required')
    evaluation_ids=[task['id'] for task in evaluations]
    if any(not isinstance(value,str) or not value for value in evaluation_ids):raise ValueError('Nonempty evaluation IDs required')
    if len(evaluation_ids)!=len(set(evaluation_ids)):raise ValueError('Duplicate evaluation IDs')
    records=[]
    for task in evaluations:
        if not task.get('protocol_id'):raise ValueError('Explicit protocol_id required')
        expected=task['expected_task_ids']
        if not expected:raise ValueError('Empty expected task set')
        if len(expected)!=len(set(expected)):raise ValueError('Duplicate expected task IDs')
        rows=[row for shard in task['generation_shards'] for row in read_asset(root,shard)]
        ids=[r['task_id'] for r in rows]
        if len(ids)!=len(set(ids)) or set(ids)!=set(expected):raise ValueError('Incomplete or duplicated generation coverage')
        for row in rows:
            if row.get('finish') not in ('stop','length'):raise ValueError('Unknown finish status')
            for key in ('completion_tokens','fwd_denoise','fwd_total','wall_s'):
                value=row.get(key)
                if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value) or value<0:
                    raise ValueError('Invalid numeric field: '+key)
            if row['finish']=='stop' and min(row['fwd_denoise'],row['fwd_total'],row['wall_s'])<=0:
                raise ValueError('Nonpositive clean denominator')
        clean=[r for r in rows if r['finish']=='stop']
        record={'id':task['id'],'protocol_id':task['protocol_id'],'n':len(rows),'collapse':len(rows)-len(clean),
                'complete_task_coverage':True,'quality_scored':False}
        if 'scope' in task or 'full_cell_n' in task:
            scope,full_n=task.get('scope'),task.get('full_cell_n')
            if scope not in ('shard','partial_shard_set','full_cell') or type(full_n) is not int or full_n<=0:
                raise ValueError('Invalid benchmark coverage declaration')
            if len(expected)>full_n or (scope=='full_cell')!=(len(expected)==full_n):
                raise ValueError('Scope disagrees with full benchmark coverage')
            record.update(scope=scope,full_cell_n=full_n,complete_benchmark_coverage=scope=='full_cell')
        by_id=None
        if task.get('verdicts'):
            verdicts=read_asset(root,task['verdicts']);by_id={r['task_id']:r for r in verdicts}
            if len(verdicts)!=len(by_id) or set(by_id)!=set(expected):raise ValueError('Incomplete or duplicated verdict coverage')
            if any(type(v.get('correct')) is not bool for v in verdicts):raise ValueError('Verdicts require Boolean correct values')
            scorer=task['verdicts'].get('scorer_protocol_id')
            if not scorer:raise ValueError('Explicit scorer_protocol_id required')
        if aggregate:
            if clean:
                record.update(m2t_tpf_mean_clean=statistics.fmean(r['completion_tokens']/r['fwd_denoise'] for r in clean),
                              all_forward_tpf_mean_clean=statistics.fmean(r['completion_tokens']/r['fwd_total'] for r in clean),
                              tps_pooled_clean=sum(r['completion_tokens'] for r in clean)/sum(r['wall_s'] for r in clean))
            else:record['efficiency_status']='all_collapsed'
            if by_id is not None:
                correct=sum(r['finish']=='stop' and by_id[r['task_id']]['correct'] for r in rows)
                record.update(correct=correct,accuracy=correct/len(rows),quality_scored=True,
                              scorer_protocol_id=scorer,evidence='aggregation_of_supplied_verdicts_not_rescoring')
        records.append(record)
    return {'schema_version':1,'manifest_sha256':hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            'results':records,'generated_code_executed':False,'models_called':False}


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=('check','aggregate'))
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--input-root',type=Path,required=True)
    ap.add_argument('--out',type=Path,help='Optional fresh output path; omitted means dry run/stdout only')
    args=ap.parse_args(argv)
    result=audit(json.loads(args.manifest.read_text()),args.input_root,args.command=='aggregate')
    text=json.dumps(result,indent=2)+'\n'
    if args.out:
        if args.out.exists():raise ValueError('Refusing to overwrite existing output')
        args.out.write_text(text)
    print(text,end='')


if __name__=='__main__':main()
