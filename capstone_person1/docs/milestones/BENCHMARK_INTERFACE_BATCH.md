# Evaluation-manifest interface batch — October 12, 2026

The Person 3 question input/protection interface is implemented and wired into v2. **The full corrected Person 1 release remains diagnostic-only.** This batch follows the [first alignment batch](FIRST_ALIGNMENT_BATCH.md).

## Delivered

- `benchmark_manifests.py` validates explicit benchmark roles, source/revision/split/selection provenance, hashes, counts and Person 3 confirmation.
- Full/fast development nesting is validated by IDs and normalized question text, including matching source provenance. Development/final intersections and official final-test reuse as development are rejected.
- Missing development files leave coverage pending and allow independent data preparation. An incomplete or invalid supplied freeze fails instead of being silently ignored.
- `import_person3_manifests.py` provides a standalone import; v2 embeds the same tested helper. Gold-answer columns are omitted from protection exports.
- `protected_questions.csv` and `evaluation_protection_scope.json` record known inputs, confirmation/coverage status and actual bounded exact/approximate-near checks. The historical reserved manifest is supported without claiming a new Person 3 sign-off.
- [BENCHMARK_INPUT_CONTRACT.md](../contracts/BENCHMARK_INPUT_CONTRACT.md) documents Person 3's inputs and ownership. The new metadata template is explicitly unsigned/example-only; no evaluation dataset was generated.

## Verification

**63 regression/integration tests pass**, including eight interface tests and execution of the actual notebook exact/near stages. Notebook schema, embedded-source consistency, syntax, cleared outputs and original notebook preservation pass.

Standalone import validates the existing 1,819 reserved questions and reports development, fast-development and explicit final-test confirmation as pending. Its import-only report correctly leaves overlap execution pending.

The end-to-end snapshot diagnostic `runs/full_run_v2_20261012T181641Z_cfe8ce44` completes with 5,000 sampled source rows, 278 checked candidates, 100 exported examples (50/source), and nested 32/64/100 manifests. Their content hashes equal those from the earlier run verified with the actual pinned tokenizer. The 100-row reference budget remains 31,973 supervised tokens. All outputs remain drafts; this does not establish representative upstream-pilot or full-release completion.

The run exports 1,819 protected question entries, completes bounded exact/approximate-near candidate checks and states that development coverage remains pending. No exact or retrieved near overlaps were found in this diagnostic universe; this is not a claim of global decontamination or guaranteed near-match recall.

## Next step

Align near matching with instance/near-instance decisions and export the final source-to-group mapping. Then implement the disjoint training/selection/loss banks, saved larger reviews, independent readiness states and packaging. Complete pinned source reload, representative upstream pilot and one full regeneration remain necessary for the corrected release.

> Historical milestone: execution commands and paths describe that milestone. Use [the current package README](../../README.md) for today's layout and commands.
