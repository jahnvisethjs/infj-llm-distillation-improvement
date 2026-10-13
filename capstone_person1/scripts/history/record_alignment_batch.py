"""Record checked first-batch evidence without changing historical run metrics."""
import json
from datetime import datetime, timezone
from pathlib import Path

from capstone_person1.pipeline_controls import file_sha256
from capstone_person1.scripts.verify_smoke_run import verify

ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / 'capstone_person1'
run_root = PROJECT / 'runs/full_run_v2_20261012T180206Z_eaf9409d'
frozen = json.loads((run_root / 'run_configuration.json').read_text(encoding='utf-8'))
if frozen['snapshot_binding']['notebook_sha256'] != file_sha256(ROOT / 'Person1_Full_Run_v2.ipynb'):
    raise ValueError('Code has progressed beyond this historical batch; preserve its recorded evidence instead of rerunning this recorder')
result = verify(run_root)
if result['source_rows_reused'] != 5000:
    raise ValueError('Expected all sampled rows to be reused during the verified resume')
reports = run_root / 'outputs/reports'
actual = json.loads((reports / 'reference_label_validation.json').read_text(encoding='utf-8'))
if actual['status'] != 'passed' or actual['tokenizer_class'] != 'Qwen2TokenizerFast':
    raise ValueError('Actual pinned tokenizer evidence missing')
if actual['actual_saved_manifest_validation']['train_n100_pilot.jsonl']['supervised_tokens'] != 31973:
    raise ValueError('Saved reference label budget changed')
archive = PROJECT / 'preserved_inputs/pre_alignment_dce3ef7348754718'
preservation = json.loads((archive / 'preservation.json').read_text(encoding='utf-8'))
if file_sha256(ROOT / 'Person1_Full_Run.ipynb') != preservation['notebook_sha256']:
    raise ValueError('Original input notebook changed')
spec = json.loads((PROJECT / 'configs/formatting_contract.json').read_text(encoding='utf-8'))
if spec['reference_encoder_sha256'] != file_sha256(PROJECT / 'person1_data_contract.py'):
    raise ValueError('Tokenizer contract verification refers to stale encoder code')
audit = json.loads((reports / 'audit_summary_pilot.json').read_text(encoding='utf-8'))
status = {
    'recorded_utc': datetime.now(timezone.utc).isoformat(),
    'status': 'DIAGNOSTIC_ONLY', 'data_ready_for_trainer': False,
    'main_run_ready': False, 'continuation_ready': False,
    'first_batch_code_verified': True,
    'test_count': 55,
    'test_command': 'python -m unittest capstone_person1.test_pipeline_controls capstone_person1.test_target_regressions capstone_person1.test_data_contract capstone_person1.test_snapshot_stream capstone_person1.test_snapshot_catalog capstone_person1.test_handoff_gate',
    'notebook_validation': ['nbformat_schema', 'magic_aware_python_syntax', 'embedded_controls_and_contract_agreement', 'cleared_outputs', 'original_input_hash_unchanged'],
    'preserved_notebook_sha256': preservation['notebook_sha256'],
    'untouched_original_colab_recovered': False,
    'code_sha256': {name: file_sha256(PROJECT / name) for name in [
        'pipeline_controls.py', 'person1_data_contract.py', 'snapshot_stream.py', 'snapshot_catalog.py',
        'streaming_rebuild.py', 'configuration.py', 'tokenizer_adapter.py']},
    'corrected_notebook_sha256': file_sha256(ROOT / 'Person1_Full_Run_v2.ipynb'),
    'successful_diagnostic': result,
    'initial_run_metrics_from_captured_execution': {'elapsed_seconds': 12.418359700008295, 'peak_rss_mib': 199.0859375},
    'counts': {key: audit[key] for key in ['rows_used_this_run', 'eligible_before_exact_dedup',
        'near_duplicate_scope_per_source', 'overlength_or_boundary_rows_removed', 'working_pool_rows', 'working_pool_source_counts', 'final_status_counts']},
    'actual_tokenizer_class': actual['tokenizer_class'],
    'actual_saved_manifest_reference_labels': actual['actual_saved_manifest_validation'],
    'evidence': {
        'reference_label_validation': str((reports / 'reference_label_validation.json').relative_to(PROJECT)),
        'formatting_contract': 'configs/formatting_contract.json',
        'handoff_readiness': str((reports / 'handoff_readiness.json').relative_to(PROJECT)),
        'preservation': str((archive / 'preservation.json').relative_to(PROJECT)),
    },
    'failed_diagnostics_retained': [
        {'run': 'full_run_v2_20261012T073609Z_fd250c80', 'reason': 'contiguous snapshot supply shortage under strict target encoding'},
        {'run': 'full_run_v2_20261012T180019Z_fe4b3b6e', 'reason': 'raw provenance dropped before manifest export; fixed and regression-tested'},
    ],
    'remaining': [
        'Person 3 manifest import and known/pending protection scope',
        'Instance/near-instance matching policy and final group mapping',
        'Disjoint 40K training, 10K selection, 1K loss banks and fixed source length bins',
        'Larger stratified reviews and actual applied decisions',
        'Independent data/main/continuation readiness and integration package',
        'Complete pinned upstream fields, representative upstream pilot and full regeneration',
    ],
}
(PROJECT / 'evidence/IMPLEMENTATION_STATUS.json').write_text(json.dumps(status, indent=2) + '\n', encoding='utf-8')
print('Saved IMPLEMENTATION_STATUS.json; current state DIAGNOSTIC_ONLY.')
