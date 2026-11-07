"""Run focused C0 checks and retain evidence without deleting test artifacts."""

import ast
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import sys
import time
import unittest
import uuid

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from capstone_person4.provenance import environment, source_hashes, write_bytes, write_json


def main():
    for path in (REPO / "capstone_person4").glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"))
    suite = unittest.defaultTestLoader.discover(str(REPO / "tests/person4"))
    captured = io.StringIO()
    started = time.perf_counter()
    with redirect_stdout(captured), redirect_stderr(captured):
        result = unittest.TextTestRunner(stream=captured, verbosity=2).run(suite)
    directory = REPO / ".test_artifacts/verification" / uuid.uuid4().hex
    report = {"passed": result.wasSuccessful(), "tests_run": result.testsRun, "failures": len(result.failures),
              "errors": len(result.errors), "skipped": len(result.skipped), "seconds": time.perf_counter() - started,
              "code": source_hashes(), "environment": environment(), "real_model_checked": False,
              "note": "Synthetic CPU fixtures exercise loss math and the reused training loop; no real parent or benchmark was executed"}
    write_json(directory / "validation.json", report)
    write_bytes(directory / "test_output.txt", captured.getvalue().encode())
    print(__import__("json").dumps({**report, "evidence_directory": str(directory)}, indent=2))
    if not result.wasSuccessful():
        print(captured.getvalue(), file=sys.stderr)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
