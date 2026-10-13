"""Apply the first alignment batch to v2 only; preserve all original notebook bytes."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / 'capstone_person1'
path = ROOT / 'Person1_Full_Run_v2.ipynb'
notebook = json.loads(path.read_text(encoding='utf-8'))


def source(index): return ''.join(notebook['cells'][index]['source'])
def replace(index, text): notebook['cells'][index]['source'] = text.splitlines(keepends=True)


# Pin the locally validated dependency versions rather than a broad Hub constraint.
replace(1, '%pip install -q "datasets==5.1.0" "transformers==4.57.6" "tokenizers==0.22.2" "datasketch==2.0.0" "huggingface_hub==0.36.0" "pandas>=2.2" "pyarrow>=17" "tqdm>=4.66"\n')
s = source(3)
start = s.index('try:\n    from google.colab import drive') if 'try:\n    from google.colab import drive' in s else -1
if start >= 0:
    end = s.index('# New runs', start)
    s = s[:start] + """if os.environ.get('CAPSTONE_MOUNT_DRIVE', '0') == '1':
    from google.colab import drive
    drive.mount('/content/drive')
    default_root = Path('/content/drive/MyDrive/capstone_person1')
else:
    default_root = Path.cwd() / 'capstone_person1'
PROJECT_ROOT = Path(os.environ.get('CAPSTONE_PROJECT_ROOT', str(default_root))).expanduser().resolve()

""" + s[end:]
replace(3, s)
s = source(5).split('\n# BEGIN STRUCTURED DATA CONFIGURATION', 1)[0]
config = json.loads((PROJECT / 'configs/person1_data_config.json').read_text(encoding='utf-8'))
config_code = (PROJECT / 'configuration.py').read_text(encoding='utf-8')
s += '\n# BEGIN STRUCTURED DATA CONFIGURATION\nimport json\n' + config_code + '\nDATA_CONFIG = ' + repr(config) + '\n'
s += """DATA_CONFIG_PATH = PROJECT_ROOT / 'configs/person1_data_config.json'
if DATA_CONFIG_PATH.exists():
    DATA_CONFIG = json.loads(DATA_CONFIG_PATH.read_text(encoding='utf-8'))
globals().update(notebook_settings(DATA_CONFIG))
REVIEW_POLICY['near_duplicate_threshold'] = NEAR_DUPLICATE_THRESHOLD
REVIEW_POLICY['decontamination_threshold'] = DECONTAMINATION_THRESHOLD
REVIEW_POLICY['diagram_policy'] = 'quarantine_external_images_and_inline_asy_tikz_svg'
# The old threshold-only engine remains diagnostic until the later matching stage is aligned.
INTEGRATION_BLOCKERS.append('Instance-level duplicate/decontamination policy alignment pending')
# END STRUCTURED DATA CONFIGURATION
"""
replace(5, s)
s = source(8)
if "'structured_data_configuration': DATA_CONFIG" not in s:
    s = s.replace("'control_source_sha256': sha256_text(CONTROL_SOURCE),", "'control_source_sha256': sha256_text(CONTROL_SOURCE),\n    'structured_data_configuration': DATA_CONFIG,\n    'reference_encoder_sha256': sha256_text(CONTRACT_SOURCE),")
replace(8, s)
s = source(14)
s = s.replace("row.get('question', '') or ''", "raw_text(row.get('question', ''))")
s = s.replace("row.get('answer', '') or ''", "raw_text(row.get('answer', ''))")
s = s.replace("row.get('problem', '') or ''", "raw_text(row.get('problem', ''))")
s = s.replace("row.get('solution', '') or ''", "raw_text(row.get('solution', ''))")
s = s.replace('final_answer, answer_status = extract_orca_final_answer(problem, solution)',
              'extraction = extract_answer_metadata(problem, solution)\n    final_answer, answer_status = extraction.candidate, extraction.mode')
s = s.replace("final_answer = compact_answer(row.get('answer', ''))", "final_answer = compact_answer(row.get('answer', ''))\n    extraction = extract_answer_metadata(problem, solution)")
duplicate = '    extraction = extract_answer_metadata(problem, solution)\n    extraction = extract_answer_metadata(problem, solution)\n'
while duplicate in s:
    s = s.replace(duplicate, '    extraction = extract_answer_metadata(problem, solution)\n')
s = s.replace("{'', 'proof', 'notfound'}", 'MISSING_ANSWER_VALUES')
s = s.replace("if has_image_reference(problem, solution): flags.append('missing_or_external_diagram')", 'flags.extend(diagram_flags(problem, solution))')
metadata = """'answer_parse_status': answer_status, 'correctness_status': 'unknown',
        'answer_extraction_json': json.dumps(extraction.to_dict(), ensure_ascii=False),
        'answer_evidence_start': extraction.evidence_start, 'answer_evidence_end': extraction.evidence_end,
        'answer_evidence_text': extraction.evidence_text, 'answer_confidence': extraction.confidence,
        'answer_type': answer_type(final_answer), 'answer_ambiguity_reason': extraction.ambiguity_reason,
        'answer_evidence_basis': extraction.evidence_basis,"""
if "'answer_extraction_json'" not in s:
    s = s.replace("'answer_parse_status': answer_status, 'correctness_status': 'unknown',", metadata)
# Numina retains its provided answer as provenance, while evidence comes from the solution.
if "'provided_answer_raw'" not in s:
    s = s.replace("'question_type': question_type,", "'provided_answer_raw': raw_text(row.get('answer', '')),\n        'question_type': question_type,")
if "'extraction_consistency_status':" not in s:
    s = s.replace("'answer_evidence_basis': extraction.evidence_basis,", "'answer_evidence_basis': extraction.evidence_basis,\n        'extraction_consistency_status': extraction_consistency_status(extraction, final_answer),\n        'solution_completeness_status': 'unknown',\n        'source_fields_origin': row.get('__source_fields_origin', 'pinned_upstream'),")
if "'numina_problem_validity_raw':" not in s:
    s = s.replace("'provided_answer_raw': raw_text(row.get('answer', '')),", "'provided_answer_raw': raw_text(row.get('answer', '')),\n        'numina_problem_validity_raw': raw_text(row.get('problem_is_valid', '')),\n        'numina_solution_validity_raw': raw_text(row.get('solution_is_valid', '')),")
replace(14, s)
s = source(26)
start = s.index('if tokenizer.eos_token_id') if 'if tokenizer.eos_token_id' in s else -1
if start >= 0:
    end = s.index('TEMPLATE_HASH =', start)
    s = s[:start] + """verify_tokenizer(tokenizer)
if TOKENIZER_ID != 'Qwen/Qwen2.5-0.5B' or tokenizer_revision != PINNED_TOKENIZER_REVISION:
    raise ValueError('Tokenizer identity/revision differs from the reference contract')
run_config_dir = RUN_ROOT / 'configs'
run_config_dir.mkdir(exist_ok=True)
asset_directory = HF_CACHE_DIR / 'models--Qwen--Qwen2.5-0.5B/snapshots' / tokenizer_revision
contract_spec = formatting_contract(tokenizer, asset_directory=asset_directory, max_total_tokens=MAX_TOTAL_TOKENS)
contract_spec['reference_encoder_sha256'] = sha256_text(CONTRACT_SOURCE)
(run_config_dir / 'formatting_contract.json').write_text(json.dumps(contract_spec, indent=2), encoding='utf-8')

""" + s[end:]
start, end = s.index('def batch_training_token_stats'), s.index('token_rows = batch_training_token_stats')
s = s[:start] + """def batch_training_token_stats(frame, batch_size=TOKENIZATION_BATCH_SIZE):
    token_rows = []
    for start in tqdm(range(0, len(frame), batch_size), desc='Encoding candidate labels'):
        for row in frame.iloc[start:start + batch_size].to_dict('records'):
            try:
                # No truncation here: the whole encoded row is then excluded if overlength.
                encoded = encode_example(tokenizer, row['problem'], row['solution'], row['final_answer'], max_total_tokens=None)
                stats = {key: value for key, value in encoded.items() if key not in {'input_ids', 'attention_mask', 'labels'}}
                stats.update(target_encoding_error=False, target_encoding_error_reason='')
            except ValueError as error:
                stats = dict(prompt_text=render_prompt(row['problem']), completion='', prompt_tokens=0,
                    completion_tokens=0, eos_tokens=0, supervised_tokens=0, total_tokens=0,
                    completion_start=0, boundary_mismatch=False, boundary_crossing_token=isinstance(error, BoundaryError),
                    target_encoding_error=True, target_encoding_error_reason=str(error))
            token_rows.append(stats)
    return token_rows

""" + s[end:]
s = s.replace("overlength_ids = set(candidate_df.loc[candidate_df['overlength'], 'row_id'])",
              "overlength_ids = set(candidate_df.loc[candidate_df['overlength'] | candidate_df['target_encoding_error'], 'row_id'])\ncandidate_df[candidate_df['target_encoding_error']].to_csv(REPORT_DIR / 'target_encoding_quarantine.csv', index=False)")
s = s.replace("if candidate_df['boundary_crossing_token'].any():\n    raise ValueError('At least one tokenizer token crosses the prompt/completion character boundary; inspect before training.')",
              "# Failed boundary/consistency examples are quarantined as whole rows.")
s = s.replace("candidate_df.groupby('dataset_id')['total_tokens']", "candidate_df[~candidate_df['target_encoding_error']].groupby('dataset_id')['total_tokens']")
s = s.replace("candidate_df[candidate_df['dataset_id'] == source].head(10)", "candidate_df[(candidate_df['dataset_id'] == source) & (~candidate_df['target_encoding_error'])].head(10)")
replace(26, s)
s = source(28).replace("(~candidate_df['overlength']) & (~candidate_df['boundary_crossing_token'])", "(~candidate_df['overlength']) & (~candidate_df['boundary_crossing_token']) & (~candidate_df['target_encoding_error'])")
# Avoid duplicate predicates on a repeat application.
s = s.replace(" & (~candidate_df['target_encoding_error']) & (~candidate_df['target_encoding_error'])", " & (~candidate_df['target_encoding_error'])")
if "'answer_extraction_json', 'answer_evidence_text'" not in s:
    s = s.replace("'answer_parse_status', 'artifact_status'", "'answer_parse_status', 'artifact_status',\n    'answer_extraction_json', 'answer_evidence_text', 'answer_evidence_start', 'answer_evidence_end',\n    'answer_confidence', 'answer_type', 'answer_ambiguity_reason', 'answer_evidence_basis',\n    'problem_raw', 'solution_raw', 'quality_flags', 'review_flags', 'eos_tokens'")
if "'extraction_consistency_status', 'solution_completeness_status'" not in s:
    s = s.replace("'quality_flags', 'review_flags', 'eos_tokens'", "'quality_flags', 'review_flags', 'eos_tokens',\n    'extraction_consistency_status', 'solution_completeness_status', 'source_fields_origin'")
replace(28, s)
s = source(30).replace("assert (~working_pool['boundary_crossing_token']).all()", "assert (~working_pool['boundary_crossing_token']).all()\nassert (~working_pool['target_encoding_error']).all()")
duplicate = "assert (~working_pool['target_encoding_error']).all()\nassert (~working_pool['target_encoding_error']).all()"
while duplicate in s:
    s = s.replace(duplicate, "assert (~working_pool['target_encoding_error']).all()")
s = s.replace("status = 'overlength_or_boundary_crossing'", "status = 'overlength_or_target_encoding_error'")
replace(30, s)
for cell in notebook['cells']:
    if cell['cell_type'] == 'code': cell['outputs'], cell['execution_count'] = [], None
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print('Aligned v2 configuration, canonical evidence, formatting and label-derived counts.')
