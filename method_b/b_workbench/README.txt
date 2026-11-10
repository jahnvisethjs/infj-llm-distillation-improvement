APPROACH B WORKBENCH — WEIGHTS-FREE PREPARATION
29 October 2026

START HERE
Upload approach_b_workbench.zip into capstone_approach_b/inputs. Run the cell
in START_ON_SOL.txt, or extract it into the project root and open
b_workbench/Approach_B_Prepare.ipynb. Use a CPU Jupyter session for this batch.
It installs an isolated software environment, runs CPU checks, verifies both
model tokenizers, and caches public base assets. Several GB of downloads may
be required. It does not run a model, generate answers, or start training.

WHAT WAS COMPLETED IN THE AUTHORING WORKSPACE
- Real 1x, 2x, 4x B selections from the SAME 5000-row universe.
- 1x: 1000 rows / 318893 stored supervised tokens / target 318845.
- 2x: 1999 rows / 634906 stored supervised tokens / target 637690.
- 4x: 3999 rows / 1275227 stored supervised tokens / target 1275380.
- Every draw meets the unchanged +/-0.5% bound and exact integer stratum quotas.
- Whole original rows are preserved; selected row/group IDs are unique and fresh
  relative to the supplied parent groups. The 1x dataset hash is unchanged.
- 6000 actual parent/universe rows match the exact recovered prompt/completion
  renderer: zero prompt differences, completion differences, or render errors.
- 39 offline tests passed. See authoring_validation.json for execution scope.
- Pinned evaluator code and dev/test files, parent registry/metadata, an isolated
  environment installer, guarded continuation runner, comparison checks, and
  notebook entry point are implemented.

WHAT MUST EXECUTE ON SOL (NO PARENT ADAPTER REQUIRED)
run_weights_free.py runs these stages in order and saves logs/progress:
1. Create venv_b_training_py312 and install the recorded training versions plus
   pandas, which is imported by the pinned original encoder. The Jupyter host
   needs only the Python standard library to reach this installation stage.
2. Run offline tests using that prepared Python environment.
3. Reproduce all real selections and audits using the same prepared Python.
4. Four small CPU tensor tests of shifted loss, masked prompt/padding, EOS, and
   token-weighted accumulation. No model is instantiated.
5. Person 3's grader tests using math-verify 0.9.0.
6. Exact model-specific encoder verification of the 1000 parent + 5000 universe
   rows for 0.5B and 1.5B. The 1.5B IDs, labels, and attention masks must match
   the 0.5B reference, not merely have the same lengths.
7. Cache pinned PUBLIC base-model weights/tokenizers for both model sizes.
8. Prepare development-evaluation command plans and a stage readiness report.

Those installation/tokenizer/framework stages could not be executed in this
workspace: it has no remote SOL shell and lacks the tokenizer/framework assets.
They are implemented and ready to run, not falsely recorded as completed.
The user's earlier 1000-row 0.5B token check passed on SOL; the expanded check
adds new budgets, both profiles, and the exact recovered encoder interface.
The public base cache is separate from the missing trained parent adapters.

FILES
workbench.py          Main CLI for prepare/status/import/verify/train/evaluate.
allocator.py          Original shared deterministic B/C0/E/C1 allocation machinery.
continuation.py       Guarded adaptation of Person 2's training function.
bootstrap_env.py      Installs project software without modifying shared Jupyter.
run_weights_free.py   One-command CPU preparation batch.
compare.py            Strict paired-result checks; no placeholder accuracy.
Approach_B_Prepare.ipynb  Thin notebook that calls the same scripts.
config.json           Explicit draft common protocol and runtime recipe.
protocol_receipt_TEMPLATE.json  Unfilled shared-protocol record.
parent_registry.json  Four known parent identities and expected weight hashes.
reference_records/    Matching supplied parent configurations/seen groups/summaries.
source_provenance.json Vendor source pins and checksums.
vendor/              Pinned original Person 1/2/3 code and frozen evaluation files.
inputs_snapshot/     Verified parent data, common universe, renderer and protection
                     inputs, so CPU preparation can resume if earlier files are lost.
prepared_results/    Actual stored-token selection outputs produced here, for review.
                     The preparation command regenerates them under the SOL root.

SCIENTIFIC CONTROLS
The same 5000-row universe/hash and allocator settings must be used by B/C0/E/C1.
Selection seed 42, universe seed 1729, and training-order seed 42 are distinct
roles. Continuation training seed 42 is a proposed default, independently named.
Row counts are outcomes of token matching; 2x/4x do not force exactly 2000/4000.
Per-source token shares differ from the 50:50 row proportions and are reported.
No student loss, correctness, embedding, loss-band or test-score signal ranks B.
No examples are truncated, repeated, or padded with fake supervised tokens.
The matching algorithm is a bounded deterministic heuristic, not uniform sampling
of feasible subsets or a proof of globally optimal constrained selection.

Screening is limited to the documented exact checks and supplied near-match
review queue. Reference correctness is not established mathematically. Seven
related development rewrites in the already-trained parent must be disclosed.
No code here labels the parent completely decontaminated.
The tiny Person 1 package __init__ has a different recorded docstring hash; it
contains no executable imports or definitions. The three executable encoder,
control, and matching files exactly match the parent training fingerprints.

WHEN FENI SENDS ONE ZIP
Import all known adapters it contains in one operation:
  <B_PYTHON> b_workbench/workbench.py --root <ROOT> import-parents --zip <ZIP>
The importer recognizes weights by expected SHA-256, rejects unknown/mismatched
weights and unsafe paths, checks safetensors headers, and uses the matching
metadata already included. It never deserializes parent optimizer/pickle files.
Expected profiles are 0.5B/1.5B R1000, parent training seeds 17 and 123.

TRAINING INTERFACE — IMPLEMENTED, NOT YET GPU-VALIDATED
continuation.py reuses Person 2's QLoRA loader, token-weighted causal-loss loop,
accumulation, gradient clipping, checkpoint format, and reload probe. It changes
only the continuation-specific assumptions: known parent metadata eligibility,
parent weights with a fresh optimizer/scaler/RNG and 10-update warmup, and the
common global token budget. Original parent optimizer state is never loaded.
Own interrupted continuation checkpoints can resume, after fingerprint/hash
checks. A main run cannot resume from a foreign, parent, or smoke checkpoint.
Every budget starts independently from the original parent, not another B arm.
The saved final adapter is the end-of-budget checkpoint, with consumed-token and
unique-row reconciliation. Existing output folders are never overwritten.

Main training requires a frozen matching shared protocol receipt, a current real
model-specific label report, and a real matching load/update/reload smoke report.
The protocol receipt is deliberately draft/unfilled: no team agreement or manual
clearance has been invented. Record the actual shared agreement once, populate
protocol_evidence in config.json, and set protocol_status to frozen.
A disposable one-update smoke can run with a draft protocol once weights arrive;
it is marked debug and cannot be evaluated as a scientific main result.

EXAMPLE PARENT-DEPENDENT COMMANDS (DO NOT RUN UNTIL THEIR GATES PASS)
<B_PYTHON> b_workbench/workbench.py --root <ROOT> train --model 0.5B --parent-seed 17 --budget 1 --smoke
<B_PYTHON> b_workbench/workbench.py --root <ROOT> train --model 0.5B --parent-seed 17 --budget 1
<B_PYTHON> b_workbench/workbench.py --root <ROOT> evaluate --model 0.5B --parent-seed 17 --budget 1 --execute

Repeat only the models, parent seeds, continuation seeds, and budgets in the
agreed protocol. Use a new --run-id for a new run/resume segment. The own-run
checkpoint used with --resume is explicitly checked before pickle deserialization.

EVALUATION AND COMPARISON
The evaluator is the pinned Person 3 implementation. It reads each checkpoint's
base/revision and uses greedy fp32 merged-adapter generation on frozen questions.
Evaluate the parent again in the SAME installed runtime/GPU as B. Historical
outputs used torch 2.14.1/CUDA 13 and must not be mixed with a newly evaluated
B under torch 2.6/CUDA 12.4. All compared arms need one matching evaluation policy.
Development command plans are generated without loading models. Final tests
require a frozen shared receipt and a predeclared run list plus --final-test.
Real evaluation execution requires a GPU and a complete matching B checkpoint.
compare.py requires matching complete frozen question IDs, settings, runtime,
and evaluator/parser code hashes. Missing B predictions cannot create scores.

<B_PYTHON> b_workbench/compare.py --parent <parent_dev_gsm.jsonl> --B <B_dev_gsm.jsonl> --out <comparison.json> --paired-statistics
Use GSM and MATH separately. Paired question uncertainty is not training-seed
uncertainty. Parent-seed repetitions and continuation-seed repetitions are
separate factors; do not conflate either with the selection seed.

RESTORE/SESSION SAFETY
Project outputs go to <ROOT>/outputs/B_workbench_v1. Packages/models remain on
scratch disk between sessions but should be backed up appropriately. Save and
download your actual notebook and fresh SOL reports. Reopen the notebook and
recreate path variables after a session ends; installed environments/caches and
completed data selections need not be rebuilt when their files remain valid.

CAPACITY LIMIT
The current 5000-row universe supports the prepared 1x/2x/4x selections. Extending
to 8x requires a newly shared universe; 16x/40x exceed the entire available
screened bank's unique-token capacity and need additional screened fresh data.
The existing 500-to-40000 training-budget curves are a separate experiment.
