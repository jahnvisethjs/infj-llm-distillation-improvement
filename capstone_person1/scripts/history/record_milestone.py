"""Record observed milestone evidence and label preserved incomplete attempts."""
import contextlib
import io
import json
from pathlib import Path

from capstone_person1.pipeline_controls import CONTROL_VERSION
from capstone_person1.scripts.verify_smoke_run import verify

project = Path(__file__).resolve().parents[2]
incomplete = []
for run in (project / 'runs').glob('full_run_v2_*'):
    if not (run / 'outputs/reports/handoff_readiness.json').exists():
        (run / 'RUN_STATUS.json').write_text(json.dumps({
            'status': 'incomplete_attempt', 'research_ready': False,
            'note': 'Historical attempt before the streaming runner; no completed manifest handoff.',
        }, indent=2), encoding='utf-8')
        incomplete.append(run.name)
with contextlib.redirect_stdout(io.StringIO()):
    small = verify(project / 'runs/full_run_v2_20261012T004452Z_ffd466bf')
    larger_resume = verify(project / 'runs/full_run_v2_20261012T004834Z_c75c286d')
evidence = {
    'control_version': CONTROL_VERSION, 'regression_and_integration_tests_passed': 32,
    'smoke_1000': small, 'smoke_5000_resume': larger_resume,
    'smoke_5000_first_execution': {
        'elapsed_seconds_after_setup': 19.607591999985743, 'observed_peak_rss_mib': 214.34375,
        'measurement_source': 'Successful first-execution command output; metrics file now records the verified resume',
    },
    'incomplete_attempts_marked': incomplete,
    'full_million_row_run_started_for_this_milestone': False,
    'research_ready': False,
    'next_dependencies': ['frozen development questions', 'shared evaluator equivalence checker',
                          'complete pinned upstream fields', 'independent reviews', 'actual training labels and corrected parent'],
}
path = project / 'evidence/NEXT_STEP_VERIFICATION.json'
path.write_text(json.dumps(evidence, indent=2), encoding='utf-8')
print('Marked incomplete attempts:', len(incomplete))
print('Saved milestone evidence:', path)
