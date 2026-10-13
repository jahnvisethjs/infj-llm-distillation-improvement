"""Wire the Person 3 question-manifest interface into v2 only."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
path = ROOT / 'Person1_Full_Run_v2.ipynb'
notebook = json.loads(path.read_text(encoding='utf-8'))
source = ''.join(notebook['cells'][8]['source'])
if "'benchmark_manifest_source_sha256'" not in source:
    source = source.replace("'reference_encoder_sha256': sha256_text(CONTRACT_SOURCE),",
        "'reference_encoder_sha256': sha256_text(CONTRACT_SOURCE),\n    'benchmark_manifest_source_sha256': sha256_text(BENCHMARK_SOURCE),")
notebook['cells'][8]['source'] = source.splitlines(keepends=True)
source = """# Roles/provenance come from metadata, not filenames. Person 3 owns these data.
protection_bundle = load_protection_inputs(
    EVAL_MANIFEST_PATH, INPUT_DIR / 'reserved_evaluation_metadata.json',
    DEV_MANIFEST_PATH, DEV_METADATA_PATH,
)
reserved_test_df = protection_bundle['reserved']
development_df = protection_bundle['development']
eval_df = protection_bundle['questions'].copy()
protection_inputs = protection_bundle['inputs']
DEVELOPMENT_PROTECTED = protection_bundle['development_protected']
if not DEVELOPMENT_PROTECTED:
    INTEGRATION_BLOCKERS.append('Person 3 frozen development coverage pending')
    print('Person 1 data preparation continues; development protection remains pending.')
protection_scope = export_protection_inputs(protection_bundle, PROCESSED_DIR, REPORT_DIR)
eval_df['normalized_problem'] = eval_df['problem'].map(normalize_problem_for_matching)
eval_hashes = set(eval_df['problem_hash'])

checked_candidate_count = len(candidate_df)
candidate_df['exact_eval_overlap'] = candidate_df['problem_hash'].isin(eval_hashes)
exact_overlap_report = candidate_df[candidate_df['exact_eval_overlap']].copy()
exact_eval_overlap_ids = set(exact_overlap_report['row_id'])
exact_overlap_report.to_csv(REPORT_DIR / 'evaluation_exact_overlap.csv', index=False)
candidate_df = candidate_df[~candidate_df['exact_eval_overlap']].reset_index(drop=True)
protection_scope['exact_candidate_checks'] = {
    'checked_rows': checked_candidate_count, 'overlapping_rows_removed': len(exact_overlap_report),
}
protection_scope['overlap_checks_status'] = 'exact_candidates_checked; near_candidates_pending'
(REPORT_DIR / 'evaluation_protection_scope.json').write_text(json.dumps(protection_scope, indent=2), encoding='utf-8')
print('Exact protected development/test overlaps removed:', len(exact_overlap_report))
"""
notebook['cells'][20]['source'] = source.splitlines(keepends=True)
source = ''.join(notebook['cells'][24]['source'])
if '# Record the precise protection-check scope' not in source:
    source += """
# Record the precise protection-check scope; legacy matching remains diagnostic.
protection_scope['near_candidate_checks'] = {
    'matching_policy': REVIEW_POLICY['overlap_scope'],
    'checked_candidate_scope_per_source': universe_per_source,
    'protected_question_entries': len(eval_df),
    'candidate_pairs': len(eval_pair_candidates_df),
    'removed_groups': len(near_eval_groups),
    'recall_guaranteed': False,
}
protection_scope['overlap_checks_status'] = 'bounded_exact_and_approximate_near_candidate_checks_completed'
(REPORT_DIR / 'evaluation_protection_scope.json').write_text(json.dumps(protection_scope, indent=2), encoding='utf-8')
"""
notebook['cells'][24]['source'] = source.splitlines(keepends=True)
source = source.replace('len(eval_overlap_pairs)', 'len(eval_pair_candidates_df)')
notebook['cells'][24]['source'] = source.splitlines(keepends=True)
for cell in notebook['cells']:
    if cell['cell_type'] == 'code': cell['outputs'], cell['execution_count'] = [], None
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print('Wired v2 frozen question input validation and known/pending protection scope.')
