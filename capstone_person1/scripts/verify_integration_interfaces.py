"""Explicit synthetic protocol fixtures; never actual Person 2 training evidence."""
import json
import tempfile
from pathlib import Path
import pandas as pd
from capstone_person1.integration_checks import verify_person2_tokens,bind_parent
from capstone_person1.person1_data_contract import CONTRACT_VERSION
from capstone_person1.pipeline_controls import file_sha256

def main():
    results=[]
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary);(root/'manifests').mkdir()
        rows=[dict(row_id='fixture_a',problem_group_id='fixture_group_a',supervised_tokens=12),dict(row_id='fixture_b',problem_group_id='fixture_group_b',supervised_tokens=20)]
        manifest=root/'manifests/debug_32.jsonl';manifest.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        parent=root/'manifests/train_n1000.jsonl';parent.write_bytes(manifest.read_bytes())
        (root/'manifests/selection_bank.jsonl').write_text(json.dumps(dict(row_id='fixture_c',problem_group_id='fixture_group_c'))+'\n')
        actual=root/'actual.csv';pd.DataFrame([dict(row_id=row['row_id'],actual_supervised_tokens=row['supervised_tokens']) for row in rows]).to_csv(actual,index=False)
        metadata=root/'tokens.json';metadata.write_text(json.dumps(dict(trainer_version='synthetic_fixture_only',contract_version=CONTRACT_VERSION,
            manifest_sha256=file_sha256(manifest),prompt_mask_verified=True,eos_label_verified=True,padding_by_position_verified=True,
            confirmed_by='synthetic fixture',confirmed_utc='2026-01-01T00:00:00Z')))
        assert verify_person2_tokens(root,'debug_32',actual,metadata)['rows']==2
        results.append('complete synthetic label scope passes')
        for name,records in [('empty',[]),('partial',[dict(row_id='fixture_a',actual_supervised_tokens=12)]),
                ('duplicate',[dict(row_id='fixture_a',actual_supervised_tokens=12)]*2),
                ('unexpected',[dict(row_id='wrong',actual_supervised_tokens=12),dict(row_id='fixture_b',actual_supervised_tokens=20)]),
                ('mismatch',[dict(row_id='fixture_a',actual_supervised_tokens=11),dict(row_id='fixture_b',actual_supervised_tokens=20)])]:
            pd.DataFrame(records,columns=['row_id','actual_supervised_tokens']).to_csv(actual,index=False)
            try:verify_person2_tokens(root,'debug_32',actual,metadata)
            except ValueError:results.append(name+' label report rejected')
            else:raise AssertionError(name+' report unexpectedly passed')
        seen=root/'seen.csv';pd.DataFrame({'problem_group_id':['fixture_group_a','fixture_group_b']}).to_csv(seen,index=False)
        meta=root/'parent.json';meta.write_text(json.dumps(dict(actual_run_id='synthetic_fixture_only',checkpoint_id='synthetic_checkpoint',checkpoint_sha256='a'*64,
            actual_manifest_sha256=file_sha256(parent),confirmed_by='synthetic fixture',confirmed_utc='2026-01-01T00:00:00Z')))
        assert bind_parent(root,parent,seen,meta)['status']=='passed';results.append('consistent synthetic parent binding passes')
        pd.DataFrame({'problem_group_id':['wrong']}).to_csv(seen,index=False)
        try:bind_parent(root,parent,seen,meta)
        except ValueError:results.append('inconsistent seen groups rejected')
        else:raise AssertionError('incorrect seen groups passed')
    report=dict(status='passed',scope='synthetic protocol fixtures only; not actual trainer/parent evidence',checks=results)
    path=Path(__file__).resolve().parents[1]/'evidence/INTEGRATION_INTERFACE_FIXTURES.json'
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
