# Repository Guidelines

## Project Structure and Module Organization

- `capstone_person4/` contains the C0/B CLI, readiness checks, data preparation, reference scoring, selection, continuation, and evaluation interfaces.
- `tests/person4/` contains focused scientific-integrity and synthetic training tests. `scripts/validate_c0.py` retains validation evidence.
- `configs/` contains example configurations and unresolved-input templates. `docs/` contains implementation plans, execution guidance, and handoff records.
- `Person2_SOL_Seeds17_123_UpTo4000.ipynb` supplies existing training definitions. Reuse them through `person2_adapter.py`; preserve historical notebook behavior.
- Workspace siblings `../data/`, `../R1000_parents_seeds17_123_20261101T023416637853Z/`, and `../capstone_person1/` supply external data, parents, and inherited encoder code. Treat original inputs as immutable.

Before V4 changes, read `docs/C0_V4_IMPLEMENTATION_PLAN.md`. For command prerequisites and environment details, read `docs/C0_RUN_GUIDE.md`.

## Build, Test, and Development Commands

Run from the repository root with the intended Python environment activated. There is no separate build step.

```bash
python -m capstone_person4 --help
python -m capstone_person4 dry-run --output outputs/person4/synthetic-demo-current
python -m capstone_person4 audit --config configs/c0.example.json
python scripts/validate_c0.py
python -m unittest discover -s tests/person4 -v
```

The dry run uses synthetic fixtures. Audit writes readiness reports and returns exit code 2 when required inputs are unresolved. The validation runner retains logs under `.test_artifacts/`. Consult the run guide for Windows interpreter/MKL settings and the isolated GPU dependency stack in `requirements-person4.txt`.

## Coding Style and Naming Conventions

Use four-space Python indentation, `snake_case` functions/modules, and `PascalCase` classes. Keep GPU imports lazy so data-only commands work without model dependencies. Preserve canonical IDs, references, hashes, and tokenizer semantics. No formatter or linter is currently configured.

## Testing Guidelines

Use `unittest` and name files `test_*.py`. Add focused regressions for masking, causal alignment, freshness, deterministic selection, budget accounting, cache validity, and parent isolation. No coverage percentage is prescribed. Synthetic checks cannot establish real-model readiness or accuracy improvements.

## Commit and Pull Request Guidelines

History uses simple descriptive subjects without an enforced format. Prefer imperative subjects, such as `Add V4 candidate protection`. PRs should explain behavior, relevant plan/issue, validation commands/results, and remaining blockers. Keep datasets, weights, caches, and machine-specific configurations out of commits.

## Agent Workflow and Experiment Integrity

Inspect, explain the plan, and obtain approval before edits; approval covers the agreed scope. Read-only repository commands need no permission. Ask explicitly before every delete-related command, including cleanup. Obtain execution authorization before full GPU experiments. Report implementation, synthetic validation, actual training, and deferred evaluation separately. Handoffs should include run indexes, lineage, manifests, token totals, logs, and checkpoint paths.
