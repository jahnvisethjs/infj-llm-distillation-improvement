# Replay + C1: one 50/50 supervised-token pilot on Sol

This package adds one arm, `REPLAY_C1_50_50`, for Qwen2.5-0.5B, original R1000 parent17,
continuation seeds42/137/2026. It uses the same frozen selection for all three seeds.
There is no ratio sweep, automatic model expansion, new candidate generation or new
reference scoring. Local tests do not establish Sol access or real-model readiness.

## Inputs and exact lineage

The implementation reads Suraj's **native** screened experiment, not the earlier C1
export from Jahnvi's separate 6,409-bank pipeline. The native package remains in place:

| Input | Sol location |
|---|---|
| B root | `/scratch/sshind38/capstone_approach_b` |
| Work root | `/scratch/sshind38/capstone_c0_c1_work` |
| Native runner | `<work>/C0_C1_Overnight_Training_20261110/nightly.py` |
| Screened C1 profile | `<work>/outputs/C0_C1_overnight_training_v1/7376bbd91d80e726/profiles/0.5B_seed17/6078ed82118a83fc` |
| Parent examples | `<B root>/data/train_n1000.jsonl` |
| Original parent | `<B root>/parents/0.5B_seed17/final/adapter` |
| Fresh bank | `<work>/outputs/C0_C1_independent_screen_preview_v1/0.5B_seed17/35f8a5ecbf57cd37/candidate_bank.jsonl` |
| Reference scores | `<profile>/reference/imported/scores.jsonl` and native completion |
| Correctness outcomes | `<profile>/generation/` including original chunks and receipts |
| C1 evaluations | `<work>/outputs/C0_C1_development_evaluation_v1/931e8fb821c7d70f` |
| A evaluation sidecars | `/scratch/msures15/eval_results/A_0.5B_p17_s<seed>_gsm.config.json` |

The runner is pinned to SHA256
`50809b9e374fcc05a9b637c381396b46223ab85abcda3ca07cd4b6a5a4af882f`.
Its own package/source checks, actual-label proof, original-parent checks, chunk
reclassification and full-run consumption validation remain active. These are the
native files exported at GitHub commit `2c88b654a904ae6e1884d5b7ac1771e02a2eda7e`
on branch `Method-B,-C0,-C1-Latest-Files`.

The bank must have4955 examples and SHA256
`ef7f149cfc20df6421bac6f3dca67af7eabf14a08ddfcb322d1feb518b22300f`.
The parent manifest must have SHA256
`efa258c9c4a8bffa85bb3be2026c187e9035f3918fe044967c4dafadcfa7f303`.
The parent adapter SHA256 is
`11a9de70df8b102e57676fb1c4b8ed61f9f0d1e7f524c6c0f0169368ea17b1cb`.

## Set up the new code and configuration

Put this repository's new package and `scripts/sol/replay_c1*` scripts in a separate
checkout or working directory. **Do not overwrite `<work>/repo`, frozen native
packages, their fingerprints, or original inputs.** There is no need to check out
another branch in the existing scientific runtime.

The supplied `replay_c1_pilot.zip` is an additive deployment bundle: extract it to
a new directory and use that directory as `REPLAY_C1_REPO`. It includes the new
package, scripts, example config, this guide, and checksums. Full data and model
weights remain external on Sol.

Run the following in Bash on Sol, replacing only the repository location with the
actual new checkout. No job is submitted by these commands:

```bash
export REPLAY_C1_REPO=/actual/path/to/this/repository
export REPLAY_C1_PYTHON=/scratch/sshind38/capstone_approach_b/venv_b_training_py312/bin/python
cd "$REPLAY_C1_REPO"

"$REPLAY_C1_PYTHON" -B -m capstone_person4_approachReplayC1 configure \
  --config-output "/scratch/$USER/replay_c1_4955_pilot_config.json" \
  --output-root "/scratch/$USER/replay_c1_4955_pilot"

"$REPLAY_C1_PYTHON" -B -m capstone_person4_approachReplayC1 preflight \
  --config "/scratch/$USER/replay_c1_4955_pilot_config.json"
```

The native-runner installation path is an expected package location, not a claim
that this machine has inspected the live Sol filesystem. If that package was
installed elsewhere, pass `--native-runner /actual/package/nightly.py` to configure.
The pinned content and original scientific directory identities must still match.

Preflight is read-only and does not import model weights or submit jobs. It verifies
all six A/C1 control checkpoints and native records. A checkpoints are found through
their real evaluation sidecars; paths are not reconstructed from timestamp guesses.
Missing permissions, changed source, missing caches, ambiguous records or failed
diagnostics stop the pipeline with the offending input named.

The different A/C1 policy-file hashes must be reconciled. Preflight searches only
the A run directory, B's data directory and native parent-record directory for the
policy file with A's exact recorded hash. If unavailable, supply the original file
using `configure --a-policy /actual/tokenization_policy.json` and a new config-output
path. It must retain special-token, completion/EOS and maximum-length semantics.
Do not manufacture a replacement policy or edit expected hashes to bypass a stop.

The training and isolated solver environments must already exist and match their
recorded versions. Nothing is installed automatically. The solver subprocess uses
`<B root>/venv_b_solver_v3/bin/python`; preflight checks OR-Tools9.14.6206.

## Selection and training

The target is318845 additional supervised tokens, including EOS under the native
encoder. Each whole-example portion must contain158626–160219 tokens: within0.5%
of159422.5. This constrains the achieved replay fraction to approximately49.75–50.25%.
Fifty percent of example count alone is insufficient.

Both portions contain25 examples per frozen source/length stratum,500 examples
each. Each portion's stratum tokens must lie within5% of half the parent's stratum
tokens. An infeasible or unresolved solve stops; no counts/tolerances are relaxed.

Replay uses B-style seeded hash priorities over original parent examples. Fresh
selection uses all4955 rows and the native C1 rule
`round(1000000 * fit) * (1 + confirmed_mistake)`, with lambda1. Reference ranks are
computed over the full screened bank; the old selected1000 rows are not halved.
Native errors, unanswered/unparseable predictions and confirmed mistakes retain
the native classifier's distinct meanings. Failed diagnostic gates cannot be bypassed.

Canonical rows stay unchanged. A separate membership ledger supplies replay/fresh
labels and the frozen length-bin labels when historical parent bins differ.
Selection/initial order use seed42; the inherited trainer's shuffle uses each
continuation seed. The replay subset is frozen independently of C1 priorities.

For a CPU-only selection run:

```bash
"$REPLAY_C1_PYTHON" -B -m capstone_person4_approachReplayC1 select \
  --config "/scratch/$USER/replay_c1_4955_pilot_config.json"
```

The runtime adapter compiles the verified effective B/C1 training function and
changes its reporting label only. It supplies the new mixed manifest and separate
membership/budget guards. It retains fresh AdamW, LR1e-4, ten-update warmup,
accumulation8, microbatch1, bf16, one epoch, no packing, inherited LoRA/NF4 double
quantization, zero weight decay and gradient clip1.0. No parent optimizer is restored.
The16-example smoke uses eight examples from each portion, performs two optimizer
updates, and checks a saved-adapter reload before it can unlock full training.

The native trainer records smoke configuration mode as `debug`, although the
continuation launcher supplies branch initialization. Validation requires saved
mode `debug` for smoke and `branch` for full training. A completion receipt is
published only after the complete metadata, source, consumption and checkpoint
validation succeeds. Earlier bundles had an incorrect smoke-mode check and could
leave a premature `passed` receipt before failing the final check. Preserve those
attempts as history; after updating validation code, repeat selection and the small
smoke under the new code fingerprint before full training.

## Explicitly authorized execution

Only run this after approving GPU execution. The `--authorize-gpu` flag is mandatory
for submission, direct training, and evaluation; ordinary preflight cannot unlock it.

```bash
bash scripts/sol/replay_c1_submit.sh \
  "/scratch/$USER/replay_c1_4955_pilot_config.json" --authorize-gpu
```

This submits seven jobs in a serial `afterok` chain:

1. CPU selection.
2. One A100 smoke.
3. Three separate A100 continuations:42,137,2026.
4. One A100 evaluation job.
5. CPU finalization.

GPU jobs request one A100, four CPUs,32GB RAM and four-hour limits. Account/partition/
QoS are explicit config fields. Each submitted job is recorded immediately in an
immutable ledger. Duplicate active submissions are rejected. A failed stage blocks
its dependents; the scripts do not cancel jobs or submit automatic retry chains.

An incomplete attempt is preserved. Inspect failure evidence and select new
`--attempt` and/or `--evaluation-attempt` names when rerunning. Full training always
restarts from original parent weights. There is no guessed partial-state resume.
Completed compatible outputs are verified and reused.

For individual stages inside an authorized A100 allocation:

```bash
"$REPLAY_C1_PYTHON" -B -m capstone_person4_approachReplayC1 train \
  --config "/scratch/$USER/replay_c1_4955_pilot_config.json" \
  --smoke --training-seed 42 --authorize-gpu

"$REPLAY_C1_PYTHON" -B -m capstone_person4_approachReplayC1 train \
  --config "/scratch/$USER/replay_c1_4955_pilot_config.json" \
  --training-seed 42 --authorize-gpu
```

Repeat the full stage for137 and2026 only after successful earlier stages. The
default attempt is`attempt01`; use a new name to preserve a failed prior attempt.

## Evaluation and artifacts

Existing A predictions used Torch2.14.1. They are context, not reusable predictions
for this matched comparison. Re-evaluate the saved A adapters under C1's common
Torch2.6.0+cu124, Transformers4.57.1, PEFT0.17.1, CUDA12.4/A100 environment.
Reuse the verified native screened C1 predictions. Evaluate the three hybrid
adapters on the same500 GSM and500 MATH development questions, greedy fp32,
batch16,1024 new tokens, merged adapters and p3-parser-v6/math-verify0.9.0.
Every sidecar and question/checkpoint/source hash must match; a nominally similar
but incompatible result is rejected. Checker execution errors block completion.

The C1 result reader supports both the original `results/C1_0.5B_p17_t<seed>/`
layout and Suraj's published export at
`/scratch/sshind38/github_screened_4955_latest/C1/evaluation/0.5B_parent17_seed<seed>/`.
It uses the published copy only when the native receipt is absent; a present but
invalid receipt or permission error still blocks. Gzip prediction hashes are
verified against the uncompressed bytes recorded by the native receipt. Frozen
question files continue to come from the configured native evaluation root.

This requires three new full continuations, one small smoke and up to twelve
new benchmark evaluation files: six hybrid and six A. No A or C1 retraining is
scheduled. Native C1 predictions are copied losslessly into the new output tree
after receipt validation. Partial new evaluations use a new evaluation-attempt;
they are never appended blindly.

Outputs are under`<output>/0.5B_parent17/<input-and-code-identity>/`:

* `selection/`: immutable replay, fresh, combined and membership JSONL plus quotas,
  actual token ratio, solver status and optimization bounds.
* `training_inputs/`, `smokes/`, `runs/`: executed source, manifests, native logs,
  checkpoint/parent hashes, per-portion consumption and reload receipts.
* `evaluation/`: common-runtime predictions, sidecars and completion receipts.
* `reports/<evaluation-attempt>/`: run index, per-seed accuracies, paired differences,
  three-seed sample SDs, summary and provenance.
* `submissions/`: exact job commands, IDs and log paths.

Paired intervals use2000 question-bootstrap draws, seed0; combined intervals sample
within each benchmark and average the two. They are fixed-checkpoint uncertainty,
not confidence intervals over training seeds. Report three-seed means/SDs
descriptively. Only one ratio/model/parent is tested. Retain inherited exposure and
unresolved historical matcher-audit disclosures. Reserved final tests stay untouched.

## Local checks

```bash
python -B -m unittest discover -s capstone_person4_approachReplayC1/tests -t . -v
```

The tiny-fixture exhaustive oracle exercises selection and invariants without GPU
work. The additional real CP-SAT/oracle comparison runs only when the exact solver
is installed. For an additive-bundle installation, make the frozen selection
repository visible and use the existing Sol solver environment before a real solve:

```bash
PYTHONPATH="/scratch/sshind38/capstone_c0_c1_work/repo:$REPLAY_C1_REPO" \
  /scratch/sshind38/capstone_approach_b/venv_b_solver_v3/bin/python -B \
  -m unittest discover -s capstone_person4_approachReplayC1/tests -t . -v
```

Importing the CLI and running its help does not import Torch or launch work.
