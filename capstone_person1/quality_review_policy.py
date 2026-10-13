"""Conservative post-canonical review filters, derived from inspected mathematical defects.

These flags mean independent review is required, not that every flagged row is wrong.
Raw source records and original validity labels remain unchanged.
"""
import json
import re
from collections import Counter
from pathlib import Path
import pandas as pd
from capstone_person1.pipeline_controls import file_sha256,sha256_text
from capstone_person1.snapshot_stream import stage_batches,atomic_json

QUALITY_REVIEW_VERSION='person1-math-review-v6'

def cohort_flags(row,config):
    problem,solution=row['problem'],row['solution']
    flags=[]
    if row['dataset_id']=='numina' and row['original_source'].casefold() in set(config.get('review_source_blacklist_numina',[])):
        flags.append('numina_source_requires_independent_correctness_review')
    label_text=re.sub(r'\\+[()]','',problem)
    label_text=re.sub(r'\$\s*(\([1-9IVXivxⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\))\s*\$',r'\1',label_text)
    labels=list(re.finditer(r'(?m)(?:(?<![A-Za-z0-9_])\(([1-9IVXivxⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+)\)|^[ \t]*([1-9IVXivxⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+)[.)])[ \t]*',label_text))
    sections={match.group(1) or match.group(2) for match in labels}
    # Section labels can introduce tasks or givens. Withhold both conservatively;
    # verb lists missed Complete/Factorize and cannot establish a complete target.
    if len(sections)>=2:flags.append('explicit_numbered_multipart_requires_full_target_review')
    if len(re.findall(r'(?:\\?_){2,}',problem))>=2:flags.append('multiple_answer_slots_require_full_target_review')
    if re.search(r'\\boxed\s*\{',problem):flags.append('boxed_question_content_requires_answer_leakage_review')
    if re.search(r'(?:\\square|□)\s*(?:\\(?:rightarrow|to)|→)\s*(?:\\square|□)',problem):
        flags.append('placeholder_function_domain_requires_source_review')
    if re.search(r'\\_[A-Za-z0-9]',problem):flags.append('escaped_math_index_requires_source_review')
    if re.search(r'\btangent\b',problem,re.I) and re.search(r"[a-zA-Z]\s*(?:''|\\prime\s*\\prime)",problem):
        flags.append('higher_derivative_from_tangent_requires_independent_review')
    if re.search(r'\\(?:cup|cap|subset|subseteq)\b|[∪∩]',problem) and re.search(r'\bparameter\b|\bvalues?\s+of\s+\$?[a-zA-Z]',problem,re.I):
        flags.append('set_parameter_alternatives_require_independent_review')
    if (re.search(r'round(?:ing|ed)?[^.\n]{0,80}nearest[^.\n]{0,30}(?:whole|integer)',solution,re.I)
        and re.search(r'\b(?:levels?|balls?|people|children|students?|marbles?|items?|books?|cars?|trains?|tickets?|integers?)\b',problem,re.I)
        and not re.search(r'\b(?:estimate|approximate|approximately|about|round|rounded|rounding)\b',problem,re.I)):
        flags.append('exact_discrete_rounding_requires_independent_review')
    return flags

def review_cohorts(stages,config,release):
    path=Path(release)/'reports/review_cohort_quarantines.csv'
    state_path=path.with_suffix('.json')
    binding=dict(stage_sha256={source:file_sha256(Path(stage)/'checkpoint.json') for source,stage in stages.items()},
        policy_sha256=file_sha256(__file__),source_blacklist=config.get('review_source_blacklist_numina',[]))
    if state_path.exists():
        state=json.loads(state_path.read_text(encoding='utf-8'))
        if state['binding']!=binding or state['sha256']!=file_sha256(path):raise ValueError('Review cohort inputs/report changed')
        frame=pd.read_csv(path,dtype=str,keep_default_na=False)
        return frame,state['report']
    records=[];counts=Counter()
    columns=['row_id','dataset_id','original_source','problem_hash','problem','solution']
    for source,stage in stages.items():
        for batch in stage_batches(stage,columns,batch_size=4096):
            for row in batch.to_pylist():
                flags=cohort_flags(row,config)
                if flags:
                    counts.update(flags)
                    records.append(dict(row_id=row['row_id'],dataset_id=source,problem_hash=row['problem_hash'],
                        flags_json=json.dumps(flags),policy_version=QUALITY_REVIEW_VERSION))
        print(source+': global review-cohort scan completed',flush=True)
    frame=pd.DataFrame(records,columns=['row_id','dataset_id','problem_hash','flags_json','policy_version'])
    frame.to_csv(path,index=False)
    report=dict(policy_version=QUALITY_REVIEW_VERSION,flagged_rows=len(frame),affected_problem_groups=frame['problem_hash'].nunique(),
        nonexclusive_flag_counts=dict(counts),source_blacklist=config.get('review_source_blacklist_numina',[]),
        scope='Complete canonical sources; all members of implicated exact/near groups are excluded from release pools.',
        mathematical_correctness_claim='No claim every flagged row is wrong; unverified cohorts are withheld conservatively.',
        rationale='Actual inspections found synthetic mathematics defects, omitted numbered answers, ambiguous OCR domains/indices and incomplete set-parameter answers.')
    atomic_json(state_path,dict(binding=binding,sha256=file_sha256(path),report=report))
    return frame,report
