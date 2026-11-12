"""Retain E's focused validation evidence; never submit or train real models."""

import argparse
import ast
from contextlib import redirect_stderr, redirect_stdout
import importlib.metadata
import io
from pathlib import Path
import sys
import time
import unittest
import uuid

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-path", help="optional isolated dependency directory; leaves system packages unchanged")
    args = parser.parse_args()
    if args.solver_path:
        sys.path.insert(0, str(Path(args.solver_path).resolve()))
    from capstone_person4.provenance import source_hashes, write_bytes, write_json
    from capstone_person4_approachE.selection import code_hashes
    for path in (REPO / "capstone_person4_approachE").rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"))
    suite = unittest.defaultTestLoader.discover(str(REPO / "capstone_person4_approachE/tests"), top_level_dir=str(REPO))
    captured = io.StringIO()
    started = time.perf_counter()
    with redirect_stderr(captured), redirect_stdout(captured):
        result = unittest.TextTestRunner(stream=captured, verbosity=2).run(suite)
    try:
        solver = importlib.metadata.version("ortools")
    except importlib.metadata.PackageNotFoundError:
        solver = None
    report = {"passed": result.wasSuccessful(), "tests_run": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
              "seconds": time.perf_counter() - started, "ortools_version": solver,
              "e_code": code_hashes(), "reused_code": source_hashes(), "real_model_checked": False,
              "note": "Synthetic fixtures and scheduler stubs; real allocator on tiny fixtures when installed. No GPU jobs or benchmark evaluation."}
    directory = REPO / ".test_artifacts/verification" / ("e-" + uuid.uuid4().hex)
    write_json(directory / "validation.json", report)
    write_bytes(directory / "test_output.txt", captured.getvalue().encode("utf-8"))
    print(__import__("json").dumps({**report, "evidence_directory": str(directory)}, indent=2))
    if not result.wasSuccessful():
        print(captured.getvalue(), file=sys.stderr)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
