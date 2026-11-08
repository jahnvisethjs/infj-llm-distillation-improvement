# C0 V4 implementation status

8 November 2026 (America/Phoenix): **the C0 runs described below are superseded.** They used a 5,000-row pool frozen by this pipeline; Method B (Suraj Shinde) selected from a different 6,409-row bank with its own strata and a CP-SAT selector, so the arms were not matched. The pipeline now takes B's bank as the explicit common universe, uses B's bins and his selector (vendored, reproduced exactly), and trains three continuation seeds (42, 137, 2026) per parent. See [C0_B_BANK_RERUN.md](C0_B_BANK_RERUN.md) for the change list, local evidence and what must be rerun. The old adapters remain on Sol under `/scratch/jseth2/c0/results` and in `c0-results-handoff.tar.gz`; they are a record, not comparison arms.

5 November 2026 (America/Phoenix).

The V4 software adaptations are implemented. Actual input binding and candidate protection ran locally. Token re-encoding, real-model scoring, continuation and mathematical accuracy evaluation remain incomplete. No accuracy improvement is claimed.

## Implemented

- `bindings.py`: V4 checksum inventory, inherited encoder/release verification, explicit parent-index selection, actual R1000 exposure, saved per-row labels, serialization and tokenizer-policy bindings.
- `protection.py`: hash-verified inherited matcher definitions; parent, protected-question, reserved-loss and internal boundaries; unioned group dispositions, conservative quarantine, exact duplicate clusters and scoped review handling. Original Orca synthetic-source metadata is preserved.
- Configuration and CLI: separate parent/candidate/selection/continuation seeds, stage dependency fingerprints, target-specific audit, `protect` and `finalize` commands. Later evaluator settings preserve earlier artifact identities.
- Preparation: deterministic parent-proportional 5,000-row freezing with explicit deficits, candidate/parent token verification and immutable provenance outputs.
- Scoring/selection/continuation: existing Person 2 stack retained; smoke source/length coverage, parent-specific raw token cache validation, common allocation, pairwise budget reporting, fresh optimizer initialization and value-level adapter-update verification.
- Completion: reconciled independent full arms produce a handoff with `evaluation_status: pending`; smoke or unreconciled runs cannot complete the phase.
- Four explicit model/parent configs cover 0.5B and 1.5B, each with parent seeds 17 and 123. The config generator accepts `--profile`; SOL launchers propagate profile overrides and include configuration/profile identity in job names. The candidate pool remains 5,000 and continuation seed remains 42.
- Source bundle builder/verifier, CPU preflight/preparation and explicit single-GPU SOL launchers. Seven stage `.sbatch` files target account `class_fse57081354fall2026`, partition `public`, QoS `class` (the only partition on which SOL accepts `class`); GPU stages request `--gpus=a100:1`. All stages request 4 CPU cores and 32 GB host RAM. Full GPU stages start with 4-hour caps and smoke checks with 30-minute caps. `submit_pipeline.sh` submits one configuration's eight-job `afterok` chain; `prefetch_models.sh` fills the Hugging Face cache on a login node so jobs run with `HF_HUB_OFFLINE=1`; `C0_ENV` activates a mamba environment inside each job. No jobs were submitted.

## Corrections made on 5 November 2026 (review against the Codex and V4 plans)

- **Partition/QoS**: scripts requested `general` with QoS `class`; SOL accepts `class` only on `public`, so every submission would have been rejected. All seven `.sbatch` files now use `public`/`class` and `--gpus=a100:1`.
- **Scoring smoke tolerance**: the padded-batch comparison used `atol=rtol=0.001`, far below bfloat16 kernel noise (logit rounding alone is ~0.06 at magnitude 16–32), so a correct real run would have failed the gate. All configurations now declare 0.05 nats; the smoke report records the largest observed per-token and per-row difference. Masking/padding bugs still fail by whole nats.
- **Smoke proof portability**: the measurement key includes the GPU name and runtime, so a smoke on one A100 variant could not unlock scoring on another. The full scoring job now runs the eight-row smoke inside its own allocation when no matching proof exists; proofs are stored per runtime key.
- **Tokenizer binding**: both the active tokenizer and the 1.5B guard's 0.5B reference tokenizer are read from the parent package and verified against its `SHA256SUMS.json`; compute nodes never need Hub access.
- **Fingerprint portability**: stage fingerprints, the measurement key and the method definition now bind content hashes only; absolute input locations (which differ between Windows, SOL login and compute mounts) no longer change artifact identities.
- **Scope**: B (random) and A (replay) are implemented and trained by other team members. The Sol chain therefore trains C0 only by default (`C0_ARMS="B C0"` opts B in), `finalize` accepts a C0-only run and records the token target/tolerance an external B must match, and `select` still produces the matched-budget B manifest for handover.
- **Environment**: `requirements-person4.txt` pins `huggingface-hub>=0.34,<1.0`, the mismatch that blocked local token re-encoding.

## Executed checks

**68 tests passed; zero failures, errors or skips.** The original 28 checks still pass. New checks cover the eight-job chain submitter and path-independent fingerprints. New checks cover V4 policy, real supplied input schemas/bindings, seed selection, historical-label and serialization mismatches, inherited-release mismatches, duplicate/quarantine propagation, stage reuse, smoke diversity, completion gates and adapter tensor changes. Model-expansion checks verify all four saved parents, separate measurement keys/artifact roots, the 1.5B token-ID compatibility guard, config generation and shell routing/submission arguments through recording stubs. Batch checks cover Slurm-spool execution, CPU/GPU stage routing, independent arms, explicit resume, safe finalization paths, requested class resources and the two-job GPU-minute budget.

[Validation JSON](../.test_artifacts/verification/7a2e96f000f44127a5ded441f39184b0/validation.json) and [test log](../.test_artifacts/verification/7a2e96f000f44127a5ded441f39184b0/test_output.txt) are retained locally in ignored runtime artifacts. CPU fixtures verify software behavior; they do not establish real Qwen loading or GPU compatibility.

All four explicit configurations pass the file-only input audit. The V4 checksum inventory, selected adapter/package records, exact 1,000-row exposure, historical 318,845 label total, inherited code and all five protected question manifests verify. GPU runtime and evaluator checks remain visibly unresolved without making the input target fail.

| Model | Parent seed | Local audit evidence |
| --- | ---: | --- |
| 0.5B | 17 | [Readiness report](../outputs/person4/v4/0.5B/054796e1758715a9/audit/eff51c57192ca75e/readiness_report.json) |
| 0.5B | 123 | [Readiness report](../outputs/person4/v4/0.5B/30dbd46b21413b7e/audit/94de6649ae3b6216/readiness_report.json) |
| 1.5B | 17 | [Readiness report](../outputs/person4/v4/1.5B/d82f35e58fd34d5c/audit/59e04486660a479c/readiness_report.json) |
| 1.5B | 123 | [Readiness report](../outputs/person4/v4/1.5B/8c9006891dc32ae3/audit/f89bea872d4f343e/readiness_report.json) |

All SOL shell helpers and `.sbatch` files pass `bash -n`. Python launch/transfer scripts parse successfully. The original Person 2 notebook is unchanged. The [SOL sbatch guide](C0_SOL_SBATCH_GUIDE.md) contains staged commands, user-supplied QoS limits and the required remote partition/account validation; no live SOL scheduler test has been performed.

## Actual candidate protection

The final scan processed the supplied 10,000-row selection bank with 2,819 unique protected questions (3,019 source entries including nested fast-development questions), the actual selected parent exposure, reserved loss data and internal matching.

| Disposition | Rows |
| --- | ---: |
| Eligible | 9,241 |
| Excluded | 54 |
| Quarantined | 705 |
| Total | 10,000 |

Twelve normalized duplicate clusters and 989 flagged directional pairs are recorded. Pair counts overlap across groups and boundaries; they must not be added to exclusion totals. The default policy quarantines unresolved internal near matches as well as external ambiguity. Approximate retrieval is not a proof of complete decontamination.

Evidence: [protection summary](../outputs/person4/v4/0.5B/b9022008184befae/protection/summary.json), flagged-pair/review tables and disposition JSONL in that directory; parent/release/encoder bindings are alongside it under `input_bindings/`.

A [capacity check](../outputs/person4/v4/0.5B/b9022008184befae/preparation_capacity_check.json) confirms that all twenty inherited strata can supply 250 rows. This is a capacity check, not a token-verified frozen manifest.

## Remaining execution requirements

`prepare` stopped at the installed environment's incompatible versions: Transformers 4.57.6 requires `huggingface-hub >=0.34.0,<1.0`, while 1.7.1 is installed. No package installation or environment replacement was performed. Use a compatible isolated tokenizer/model environment before re-encoding the candidates and replay anchor.

The current CPU environment also lacks PEFT, Accelerate and bitsandbytes. The account, partition (`public`), QoS (`class`) and initial resource caps are specified from the ASU partition table. Actual capstone SOL paths and the environment are set through `C0_*` variables at submission time. Actual GPU allocation, model smoke checks and measured time/memory requirements remain unresolved. GPU execution requires separate authorization.

No real scoring smoke, 5,000-row reference pass, disposable real training smoke, B/C0 checkpoint or phase-completion artifact exists yet. Evaluation software and accuracy comparisons remain deferred.

Use the [V4 run guide](C0_V4_RUN_GUIDE.md) for transfer, configuration, preparation, separate scoring/training jobs and final handoff. The [V4 plan](C0_V4_IMPLEMENTATION_PLAN.md) retains the full experiment completion criteria.
