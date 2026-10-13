# First alignment batch — October 12, 2026

The initial code batch is implemented and tested. The corrected full Person 1 release is **not ready**. This batch establishes target safeguards and a shared reference encoder before the remaining matching, pool and packaging changes.

## Delivered code

- Preserved the actual pre-alignment local notebook and helper inputs under `preserved_inputs/pre_alignment_dce3ef7348754718/`, with checksums. `Person1_Full_Run.ipynb` remains unchanged. The archive explicitly records that this was an already-modified local input, not a recovered untouched original Colab notebook.
- Created `Person1_Full_Run_v2.ipynb`, wired the local runner and regression checks to it, and cleared execution outputs. Drive mounting is optional through `CAPSTONE_MOUNT_DRIVE=1`; `CAPSTONE_PROJECT_ROOT` overrides the project root.
- Added and wired `configs/person1_data_config.json`, immutable source/tokenizer revisions, quotas, seeds, target policies and frozen code/configuration bindings. The old matching engine remains explicitly diagnostic pending instance-policy alignment.
- Fixed interior/multiple final markers, null/NaN targets, explicit alternatives, partial arithmetic targets and leading decimals. Existing equivalent correct terminal final-answer blocks are preserved. Unsupported or unproven targets are quarantined.
- Added structured candidate/mode/span/evidence/confidence/type metadata, separate consistency/completeness/correctness fields, original text provenance and distinct inline versus external diagram flags. Snapshot-derived Numina validity provenance remains explicitly inferred; it cannot certify a final source release.
- Added `person1_data_contract.py`: exact plain-text prompt, supported completion construction, combined-text offsets, completion-only labels, one supervised EOS, positional padding, label-derived counts, whole-row length exclusion and saved-manifest validation. The notebook embeds the same tested source.
- Verified the actual pinned `Qwen2TokenizerFast`, EOS/PAD strings and ID 151643, local asset hashes, and reference labels. No model weights were loaded. Usage is documented in [REFERENCE_ENCODER.md](../contracts/REFERENCE_ENCODER.md).
- Added deterministic whole-snapshot random sampling to diagnostic runs, with seed/sample binding and interruption/resume tests. Raw provenance is preserved through the bounded catalog and manifest export.

## Verification and diagnostic outputs

**55 regression/integration tests passed.** Notebook schema, magic-aware code syntax, embedded-source agreement, cleared outputs and the unchanged original notebook hash passed. Actual historical Orca rows 71484 and 75823 still extract 6 and 1; row 185528 remains quarantined.

The successful diagnostic is `runs/full_run_v2_20261012T180206Z_eaf9409d`:

| Item | Result |
|---|---|
| Canonical rows processed | 5,000: 2,500 sampled across each source snapshot |
| Initial eligibility | 775; flags are preprocessing diagnostics, not mathematical certification |
| Checked candidate scope | 278: 139/source |
| Length/target encoding exclusions | 56 |
| Exported training pool | 100: 50/source |
| Nested diagnostic manifests | 32, 64, 100; these are overlapping prefixes, not separate banks |
| 100-row reference supervised budget | 31,973 tokens, including 100 genuine EOS labels |
| Initial processing time / observed peak RSS | About 12.4 seconds after setup / 199.1 MiB |
| Research readiness | `false` |

Saved manifests are read back and verified for hashes, IDs/groups, ordered nesting, target consistency, exact prompt bytes, lengths and label-derived counts. The run's `outputs/reports/reference_label_validation.json` records verification using the actual pinned tokenizer. Resume verification is recorded in [IMPLEMENTATION_STATUS.json](../../evidence/IMPLEMENTATION_STATUS.json).

Earlier new diagnostic attempts are retained: `full_run_v2_20261012T073609Z_fd250c80` stopped because the first contiguous Numina subset had no accepted targets; `full_run_v2_20261012T180019Z_fe4b3b6e` exposed dropped raw provenance and stopped before export. The latter defect is fixed and regression-tested. Their failed metrics were not rewritten as successes.

`SNAPSHOT_CONTRACT_SAMPLE.json` records a separate fixed-seed 100-row-per-source inspection across the historical snapshots. It is an extraction/encoding diagnostic, not completed independent mathematical review. Missing/ambiguous conclusions remain rejected; no quotas were filled by relaxing those checks.

## Remaining work in the alignment sequence

1. Complete the Person 3 frozen-manifest import/protection interface and report known versus pending coverage. Person 3 prepares its own evaluation data.
2. Align near matching to instance/near-instance decisions and export final group mappings. The current diagnostic retains the older conservative lexical matching policy.
3. Implement the disjoint 40K training, 10K selection and 1K held-out loss banks, common per-source length-bin edges and required R500/R1000/R2000/R4000 prefixes. Selection data must be exportable before a trained parent exists.
4. Increase stratified target review to approximately 100/source, save actual decisions and apply findings before final splitting/export. Current small review sheets remain unsigned.
5. Separate `DIAGNOSTIC_ONLY`, `DATA_READY_FOR_TRAINER`, `MAIN_RUN_READY` and `CONTINUATION_READY`, then package Person 2 verification/parent-binding interfaces.
6. Obtain complete pinned upstream fields, run the required representative upstream pilot, resolve supply/review findings, perform one full regeneration and validate/package the release.

The successful run uses historical snapshots. Old Numina answer truncation and absent original validity labels still require upstream reload. Its random sampling improves diagnostic coverage but **does not satisfy the required representative pilot using complete pinned upstream fields**. No corrected 40K/10K/1K release, training run, evaluation scorer or FIT ranking is claimed.

> Historical milestone: execution commands and paths describe that milestone. Use [the current package README](../../README.md) for today's layout and commands.
