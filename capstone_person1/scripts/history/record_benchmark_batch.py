"""Update current implementation evidence while retaining the first batch record."""
import json
from datetime import datetime, timezone
from pathlib import Path

from capstone_person1.pipeline_controls import file_sha256
from capstone_person1.scripts.verify_smoke_run import verify

ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / 'capstone_person1'
run_root = PROJECT / 'runs/full_run_v2_20261012T181641Z_cfe8ce44'
reports = run_root / 'outputs/reports'
configuration = json.loads((run_root / 'run_configuration.json').read_text(encoding='utf-8'))
if configuration['snapshot_binding']['notebook_sha256'] != file_sha256(ROOT / 'Person1_Full_Run_v2.ipynb'):
    raise ValueError('Current notebook differs from the verified run')
if configuration['benchmark_manifest_source_sha256'] != file_sha256(PROJECT / 'benchmark_manifests.py'):
    raise ValueError('Question-manifest interface differs from the verified run')
result = verify(run_root)
if result['source_rows_reused'] != 5000:
    raise ValueError('Expected complete source checkpoint reuse')
scope = json.loads((reports / 'evaluation_protection_scope.json').read_text(encoding='utf-8'))
protected_path = run_root / 'outputs/processed/protected_questions.csv'
if file_sha256(protected_path) != scope['protected_questions_sha256']:
    raise ValueError('Protected-question output hash changed')
if scope['protected_question_entries'] != 1819 or scope['person3_development_coverage'] != 'pending':
    raise ValueError('Unexpected supplied evaluation coverage')
if scope['overlap_checks_status'] != 'bounded_exact_and_approximate_near_candidate_checks_completed':
    raise ValueError('Pipeline protection-check evidence is incomplete')
path = PROJECT / 'evidence/IMPLEMENTATION_STATUS.json'
previous = json.loads(path.read_text(encoding='utf-8'))
if previous['successful_diagnostic']['manifest_sha256'] != result['manifest_sha256']:
    raise ValueError('Saved diagnostic data no longer matches actual-tokenizer validation from the first batch')
prior = PROJECT / 'evidence/FIRST_BATCH_VERIFICATION.json'
if not prior.exists():
    prior.write_text(json.dumps(previous, indent=2) + '\n', encoding='utf-8')
first_batch = json.loads(prior.read_text(encoding='utf-8'))
status = dict(previous)
status.update(recorded_utc=datetime.now(timezone.utc).isoformat(),
              completed_code_batches=['initial_target_and_reference_encoder', 'person3_question_input_interface'],
              first_batch_evidence='evidence/FIRST_BATCH_VERIFICATION.json',
              previous_successful_diagnostic=first_batch['successful_diagnostic'],
              successful_diagnostic=result, test_count=63,
              corrected_notebook_sha256=file_sha256(ROOT / 'Person1_Full_Run_v2.ipynb'),
              evaluation_manifest_interface_verified=True,
              evaluation_protection_scope=scope,
              actual_tokenizer_validation_reuse_basis='identical manifest SHA-256 values to first-batch actual-tokenizer verification',
              latest_initial_run_metrics_from_captured_execution={'elapsed_seconds': 19.73491489997832, 'peak_rss_mib': 200.95703125})
status['code_sha256']['benchmark_manifests.py'] = file_sha256(PROJECT / 'benchmark_manifests.py')
status['test_command'] = 'python -m unittest discover -s capstone_person1/tests -t . -p "test_*.py"'
status['evidence']['evaluation_protection_scope'] = str((reports / 'evaluation_protection_scope.json').relative_to(PROJECT))
status['evidence']['protected_questions'] = str(protected_path.relative_to(PROJECT))
status['remaining'] = [
    'Instance/near-instance matching policy and final group mapping',
    'Disjoint 40K training, 10K selection, 1K loss banks and fixed source length bins',
    'Larger stratified reviews and actual applied decisions',
    'Independent data/main/continuation readiness and integration package',
    'Complete pinned upstream fields, representative upstream pilot and full regeneration',
    'Person 3 supplies/decides development and fast-development freezes plus final-test role confirmation',
]
path.write_text(json.dumps(status, indent=2) + '\n', encoding='utf-8')
print('Updated IMPLEMENTATION_STATUS.json; 63 tests; state remains DIAGNOSTIC_ONLY.')
