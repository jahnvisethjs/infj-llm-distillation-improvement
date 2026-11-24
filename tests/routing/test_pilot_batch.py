"""Exercise the real worker after relocation into a Slurm-like spool directory."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
import uuid

REPO = Path(__file__).resolve().parents[2]
BASH = Path('C:/Program Files/Git/bin/bash.exe')
if not BASH.exists():
    BASH = shutil.which('bash')


@unittest.skipUnless(BASH, 'Bash required for batch worker regression')
class BatchWorkerTests(unittest.TestCase):
    def test_spooled_worker_imports_from_explicit_deployed_code_directory(self):
        root = REPO / '.test_artifacts' / ('pilot-batch-' + uuid.uuid4().hex)
        code = root / 'deployment with spaces'
        package = code / 'capstone_routing'
        package.mkdir(parents=True)
        (package / '__init__.py').write_text('', encoding='utf-8')
        (package / 'pilot.py').write_text(
            'import json, os, sys\n'
            'with open(os.environ["PILOT_TEST_LOG"], "a", encoding="utf-8") as stream:\n'
            '    stream.write(json.dumps({"cwd": os.getcwd(), "args": sys.argv[1:]}) + "\\n")\n',
            encoding='utf-8')
        spool = root / 'slurm_spool/job123/slurm_script'
        spool.parent.mkdir(parents=True)
        spool.write_bytes((REPO / 'scripts/sol/routing_pilot_worker.sh').read_bytes())
        log = root / 'invocations.jsonl'
        env = dict(os.environ)
        env.pop('PYTHONPATH', None)
        env.pop('PYTHONHOME', None)
        env.update(ROUTING_DIR=(root / 'experiment').as_posix(),
                   ROUTING_PY=Path(sys.executable).as_posix(),
                   ROUTING_PILOT_PY=Path(sys.executable).as_posix(),
                   ROUTING_ENCODER_CACHE=(root / 'cache').as_posix(),
                   ROUTING_PILOT_CODE=code.as_posix(), PILOT_TEST_LOG=log.as_posix())
        for explicit in (True, False):
            with self.subTest(explicit_deployment=explicit):
                if not explicit:
                    env.pop('ROUTING_PILOT_CODE')
                stage_log = log.with_name(f'invocations-{explicit}.jsonl')
                env['PILOT_TEST_LOG'] = stage_log.as_posix()
                result = subprocess.run([str(BASH), spool.as_posix()], cwd=code, env=env,
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                calls = [json.loads(line) for line in stage_log.read_text().splitlines()]
                self.assertEqual(len(calls), 2)
                self.assertEqual([c['args'][0] for c in calls], ['label', 'fit'])
                self.assertTrue(all(Path(c['cwd']).resolve() == code.resolve() for c in calls))


if __name__ == '__main__':
    unittest.main()
