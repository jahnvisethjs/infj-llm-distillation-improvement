"""Conservative replacement-screen proposal; question screening, not math certification.

All nearest-neighbor results are deterministic retrieval flags. No evaluation
accuracy, NLL, generation outcome or selection membership enters the rule.
"""
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent
PREV = ROOT.parent / 'protection_evidence'
bank = json.loads((PREV/'candidate_questions.json').read_text())
parent = json.loads((PREV/'parent_questions.json').read_text())
with (PREV/'B_evidence/inputs/protected_questions.csv').open(newline='') as f:
    protected = list(csv.DictReader(f))

def norm(text):
    return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',text)).strip()

def relaxed(text):
    value = norm(text)
    for left,right in ((r'\left',''),(r'\right',''),(r'\dfrac',r'\frac'),(r'\tfrac',r'\frac'),('$',''),(r'\(',' '),(r'\)',' ')):
        value=value.replace(left,right)
    return re.sub(r'\s+','',value)

refs=[]
seen={}
for row in protected:
    key=norm(row['problem'])
    ref_id={'set':row['set_id'],'id':row['problem_id'],'role':row['role']}
    if key in seen:
        refs[seen[key]]['identities'].append(ref_id)
    else:
        seen[key]=len(refs)
        refs.append({'problem':row['problem'],'boundary':'benchmark_protected','identities':[ref_id]})
for r in parent:
    refs.append({'problem':r['problem'],'boundary':'parent','identities':[{'id':r['row_id']}]})
for filename,boundary in [('clean_development_reference_questions.json','reserved_development'),
                          ('transfer_numina_bank_questions.json','reserved_transfer')]:
    for row in json.loads((ROOT/filename).read_text()):
        refs.append({'problem':row['problem'],'boundary':boundary,'identities':[{'id':row['row_id']}]})
print(f'Independent proposal: {len(bank)} candidates, {len(refs)} reference questions',flush=True)
vectorizer=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=2,dtype=np.float32)
x=vectorizer.fit_transform([r['problem'] for r in bank+refs])
external=NearestNeighbors(n_neighbors=1,metric='cosine',algorithm='brute',n_jobs=2).fit(x[len(bank):])
internal=NearestNeighbors(n_neighbors=2,metric='cosine',algorithm='brute',n_jobs=2).fit(x[:len(bank)])
flags=[]
for start in range(0,len(bank),300):
    end=min(start+300,len(bank))
    ds,ix=external.kneighbors(x[start:end])
    internal_ds,internal_ix=internal.kneighbors(x[start:end])
    for pos in range(end-start):
        number=start+pos
        row=bank[number]
        score=1-float(ds[pos,0])
        if score>=.65:
            ref=refs[int(ix[pos,0])]
            flags.append({'row_id':row['row_id'],'rule':'external_cosine_ge_065','boundary':ref['boundary'],
                          'cosine':score,'counterpart_ids':ref['identities'],'candidate':row['problem'],'reference':ref['problem']})
        for distance,index in zip(internal_ds[pos],internal_ix[pos]):
            if int(index)==number:
                continue
            score=1-float(distance)
            if score>=.95:
                other=bank[int(index)]
                flags.append({'row_id':row['row_id'],'rule':'internal_cosine_ge_095','boundary':'internal',
                              'cosine':score,'counterpart_ids':[{'id':other['row_id']}],
                              'candidate':row['problem'],'reference':other['problem']})
            break
    print(f'Screened {end}/{len(bank)}; flag records={len(flags)}',flush=True)

# Internal matches are symmetric: quarantine both ends even if the other end's
# own closest neighbor differs. Cosmetic equality is checked separately.
held=set()
for f in flags:
    held.add(f['row_id'])
    if f['boundary']=='internal':
        held.update(i['id'] for i in f['counterpart_ids'])
exact_flags=[]
for name,fun in [('NFKC_whitespace',norm),('latex_spacing',relaxed)]:
    ref_index=defaultdict(list)
    for r in refs:ref_index[fun(r['problem'])].append(r)
    internal_index=defaultdict(list)
    for row in bank:
        internal_index[fun(row['problem'])].append(row['row_id'])
        for ref in ref_index.get(fun(row['problem']),[]):
            held.add(row['row_id'])
            exact_flags.append({'row_id':row['row_id'],'rule':name,'boundary':ref['boundary'],
                                'counterpart_ids':ref['identities'],'candidate':row['problem'],'reference':ref['problem']})
    for members in internal_index.values():
        if len(members)>1:
            held.update(members)
            exact_flags.append({'rule':name,'boundary':'internal_duplicate_cluster','row_ids':sorted(members)})

known=json.loads((ROOT.parent/'output/C0_C1_Followup_Review_and_Preview_20261110/known_exclusions.json').read_text())
known_ids={d['row_id'] for d in known['decisions']}
held.update(known_ids)
groups={r['problem_group_id'] for r in bank if r['row_id'] in held}
held.update(r['row_id'] for r in bank if r['problem_group_id'] in groups)
retained=[r for r in bank if r['row_id'] not in held]
cap=Counter((r['dataset_id'],r['length_bin']) for r in retained)
b_ids=set(json.loads((PREV/'B_selected_ids.json').read_text()))
c_ids=set(json.loads((PREV/'C0_selected_ids.json').read_text()))
summary={
 'version':'independent-conservative-screen-proposal-v1',
 'status':'new_protocol_proposal_not_original_matcher_clearance',
 'candidate_rows':len(bank),'reference_rows':len(refs),'held_out_rows':len(held),'retained_rows':len(retained),
 'retained_sources':dict(Counter(r['dataset_id'] for r in retained)),
 'minimum_stratum_capacity':min(cap.values()),
 'strata':[{'source':s,'length_bin':b,'candidate_rows':cap[s,b],'required_rows':50} for s,b in sorted(cap)],
 'count_capacity_passed':len(cap)==20 and min(cap.values())>=50,
 'original_selected_B_flagged':len(b_ids&held),'original_selected_C0_flagged':len(c_ids&held),
 'flag_records_by_rule':dict(Counter(f['rule'] for f in flags)),
 'rules':{'external_threshold':.65,'internal_threshold':.95,'max_distance_metric':'cosine',
          'vectorizer':"char_wb 3-5 grams, min_df=2, float32, fit once over candidates plus all 7319 supplied reference rows",
          'external_neighbors':1,'internal_neighbors_including_self':2,
          'flags_are_quarantined_without_identity_claim':True,'quality_scope':'Original structural eligibility plus removal of the two known targeted defects; no mathematical correctness certification'},
 'limitations':['Not a reproduction or resolution of the unavailable 25 exclusions/572 quarantines.',
                'Conservative similarity quarantine removes some distinct questions and changes the eligible population.',
                'Below-threshold semantic matches can remain; this is a bounded text screen, not complete decontamination.',
                'Only the provided R1000 parent, benchmark protected questions, development-reference and transfer reserves are screened; other unseen reserve manifests are not certified.',
                'Seven inherited parent/development relationships remain disclosed.',
                'Token-constrained feasibility and final selections still require the Sol solver; count capacity alone does not establish feasibility.'],
 'main_training_cleared':False,'original_matcher_flags_resolved':False,
 'environment':{'numpy':np.__version__,'sklearn':sklearn.__version__},
 'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'reference_content_sha256':hashlib.sha256(json.dumps(refs,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
 'held_out_ids':sorted(held),
}
(ROOT/'independent_screen_proposal.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
(ROOT/'independent_screen_flags.json').write_text(json.dumps(flags+exact_flags,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ('held_out_ids','strata')},indent=2),flush=True)
