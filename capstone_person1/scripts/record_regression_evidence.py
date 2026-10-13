"""Run regressions and bind their result to the exact tested implementation."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

def main():
    project = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, '-X', 'utf8', '-m', 'unittest',
        'discover', '-s', str(project/'tests'), '-v'], capture_output=True,
        text=True, encoding='utf-8', cwd=project.parent)
    output = result.stdout + result.stderr
    (project/'evidence/RELEASE_REGRESSION_TESTS.txt').write_text(output, encoding='utf-8')
    paths = list(project.glob('*.py')) + list((project/'tests').glob('test_*.py'))
    evidence = dict(status='passed' if result.returncode == 0 else 'failed',
        command='python -X utf8 -m unittest discover -s capstone_person1/tests -v',
        code_sha256={path.relative_to(project).as_posix():hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
    (project/'evidence/RELEASE_REGRESSION_TESTS.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    print(output[-2500:])
    return result.returncode

if __name__ == '__main__':
    raise SystemExit(main())
