# Adapter routing: recorded opportunity, held-out experiment, and live answers

The `capstone_routing` package is additive. It does not alter historical training,
frozen evaluators, or adapters. Every result directory is create-only and sealed
with file hashes and upstream receipts. Retain interrupted work and use a new
destination for repairs. `inspect` is read-only and prints JSON.

The first deliverable is **correctness matrices, rescue/risk counts, and oracle
accuracy** for the existing 0.5B A/B/C1 matched blocks. Router learning, the
A-only seed-diversity control, and live answering are separate explicit stages.
There is no automatic decision to spend GPU time after opportunity analysis.

For a five-checkpoint, approximately 450-question screen before full labeling,
use [the separate pilot guide](ROUTING_PILOT_GUIDE.md). It stops after a small
validation report and never starts the full experiment automatically.

Build a routing-only ZIP locally with:

```powershell
& 'C:/softwares/pythonAnaconda/python.exe' -B scripts/package_routing.py --output outputs/person4/routing_release/routing_v2.zip
```

Transfer it to SOL using your normal SSH/SCP workflow and extract it into a new
directory. The archive includes `ROUTING_SHA256SUMS.json`, all routing source,
the example configuration, requirements, and this guide. It contains no weights,
predictions, datasets, or credentials and does not replace existing deployments.

## Environments and experiment inputs

Run CPU inspection/opportunity/patterns with Python 3.10+ and the standard library.
For embeddings and fitting, install `requirements-routing.txt` in a **separate**
CPU environment, with a compatible PyTorch installation. Preserve the existing
native evaluator environment: fp32, greedy, merged adapters, 1,024-token cap,
batch 16, A100-SXM4-80GB, torch 2.6.0+cu124, transformers 4.57.1, peft 0.17.1,
math-verify 0.9.0, and the hash-pinned Person 3 evaluator/parser. `label` must be
invoked using that native GPU Python. It does not need sklearn or an encoder.
For a live demonstration, compute/freeze embeddings using the same PyTorch build
and encoder package versions that will be installed in the isolated inference
environment; CPU embedding execution can use the native CUDA wheel with
`device="cpu"`. A CPU-only wheel and a CUDA wheel are recorded as distinct builds.

Copy `configs/routing.sol.example.json` to a new local configuration and verify
the documented SOL locations. The template intentionally leaves unknown matcher
hashes, exposure inventory, and training-source proofs unresolved. No credentials,
weights, dataset downloads, or GPU jobs are included in this implementation.

Configuration paths are relative to the configuration file. Use absolute paths
inside separately supplied import-index files. Save outputs outside all input
directories; the CLI rejects overlapping destinations.

Candidate identities are `p17_A_t42`, `p17_B_t42`, `p17_C1_t42`, etc., across parents
17/123 and continuation seeds 42/137/2026. The original R1000 parent is not A:
**A is its REPEAT continuation checkpoint**. B accepts its recorded RANDOM arm.
The native screened C1 experiment is the intended C1 input.

### Explicit checkpoint and attempt indexes

Default discovery scans only the three configured experiment roots. It skips
known native/raw, smoke, and archive subdirectories and refuses ambiguous
prediction attempts. When multiple canonical exports exist, set `predictions`
explicitly. Do not pick an attempt based on correctness.

Each `checkpoints` entry may specify:

```json
{
  "checkpoint": "/actual/run/final/adapter",
  "method": "C1",
  "parent_seed": 17,
  "continuation_seed": 42,
  "run_config": "/actual/run/run_config.json",
  "training_summary": "/actual/run/training_summary.json",
  "parent_checkpoint": "/actual/original-parent/final/adapter",
  "parent_run_config": "/actual/original-parent/run_config.json",
  "training_source": {
    "path": "/actual/training_inputs/effective_train.py",
    "sha256": "REPLACE_WITH_64_CHARACTER_SHA256",
    "loader_function": "build_model",
    "save_function": "save_checkpoint",
    "format": "cumulative_lora_on_base",
    "parent_loaded_for_continuation": true
  }
}
```

Omit explicit paths that the standard `final/adapter` layout can resolve.
The training source must hash-match a source identity recorded by the actual
training run, and contain the audited existing-PEFT loader/full-adapter saver.
Do not set the declaration without auditing how that run initialized its model.
Unsupported incremental/merged save formats are rejected for generation. A
source proof can also be supplied by policy in `training_sources` for discovery;
use explicit checkpoint entries when different runs used different sources.
Offline complementarity can run without these generation-only source proofs.

### Native C1 hashes only the generated train function

The native screened C1 wrapper `50809b9e...` constructs `train()` in memory. Its
recorded `code_sha256=3b908d61...` is the SHA256 of that function text, not an
entire trainer module. The original `build_model` and `save_checkpoint` definitions
remain in the separate Person 2 core `638b4bef...`. This is the supported
`screened_c1_train_function_v1` representation.

The v2 routing bundle reproduces the inspected B adaptation, verifies its
intermediate `ccb26804...` function hash, applies only the native C1 reporting-label
replacement, and verifies the final `3b908d61...` hash. It verifies loader/save
definitions against the pinned original core separately. No training functions
or native launchers are executed while constructing this proof.

From a newly extracted v2 code directory, keep the existing experiment as data:

```bash
export ROUTING_DIR=/scratch/jseth2/routing_20261123_203728
export ROUTING_PY=/scratch/sshind38/capstone_approach_b/venv_b_training_py312/bin/python

"$ROUTING_PY" -B -m capstone_routing.proofs export-c1 \
  --core /scratch/sshind38/capstone_approach_b/b_workbench_500/vendor/person2/trainer_core.py \
  --output "$ROUTING_DIR/inputs/source_proofs/c1_v1"
```

The expected final train hash is
`3b908d61c27c600469984c05669f0ab9d851772a62f6dd3b5ed32239808ee6e8`.
Changing or appending a newline to the generated source changes that hash and is
rejected. The proof's original core file is also hash-checked on every use.

A's distinct trainer `8dee6f51...` still requires a readable archived source or
its exact reconstruction from the A run owner. Do not label B's source as A's.
Once the supported full A source is available, register all source proofs in a
**new** registry, without editing existing opportunity or question-pool receipts:

```bash
"$ROUTING_PY" -B -m capstone_routing.proofs register \
  --registry "$ROUTING_DIR/results/opportunity_matched01" \
  --a-source /actual/readable/archived/A/trainer_core.py \
  --b-source /scratch/sshind38/capstone_approach_b/b_workbench_500/vendor/person2/trainer_core.py \
  --c1-proof "$ROUTING_DIR/inputs/source_proofs/c1_v1" \
  --output "$ROUTING_DIR/results/registry_sources01"
```

Registration checks every checkpoint and source binding, including C1's native
wrapper identity. An unsupported A source layout requires its own audited proof;
it is not automatically accepted. Use the new registry for `label` and `train`,
and the existing `results/pool01` for question splits. Checkpoint weights and
training exposure identities must remain exactly equal to the original registry.
Because the v2 code directory is separate, pass absolute experiment-data paths
when running its commands. Installing this update does not require repeating the
completed opportunity analysis or question screening.

### GPU generation smoke after source registration

The standalone `scripts/smoke_routing_generation.py` is supplied separately from
the existing v2 ZIP. Upload that file without replacing the installed package.
From the extracted v2 code directory, set `ROUTING_CODE="$PWD"`, retain the
experiment-data variables above, and run this explicitly authorized GPU check:

```bash
export HF_HUB_CACHE=/scratch/sshind38/capstone_approach_b/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

srun --account=class_fse57081354fall2026 --partition=public --qos=class \
  --gpus=a100:1 --ntasks=1 --cpus-per-task=4 --mem=32G --time=01:00:00 \
  --chdir="$ROUTING_CODE" \
  "$ROUTING_PY" -B /scratch/jseth2/smoke_routing_generation.py \
  --code-root "$ROUTING_CODE" \
  --config "$ROUTING_DIR/configs/routing.pool.local-sources.json" \
  --registry "$ROUTING_DIR/results/registry_sources01" \
  --pool "$ROUTING_DIR/results/pool01" \
  --output "$ROUTING_DIR/results/generation_smoke01" --authorize-gpu
```

This checks A/B/C1 from parent 17, continuation seed 42. It deterministically
selects eight GSM and eight MATH questions from router training, then generates
48 responses through the unchanged native evaluator and 48 through the live
bridge. It checks exact completion text, generated-token counts, and stopping
reasons, with native batch 16 versus live batch 1. It preserves native logs,
checkpoint/settings evidence, live lineage/timing records, and a sealed summary.
No router is fitted and no final-test questions are used. Local mocked checks
are marked synthetic and cannot establish GPU parity. A real successful run
reports `complete: true`, `native_live_match: true`, and `mismatch_count: 0`.
Inspect any differences before proceeding; batch-size effects are possible.
Preserve interrupted outputs and choose a new attempt directory when retrying.

An explicit `predictions` entry is:

```json
{
  "candidate": "p17_C1_t42",
  "benchmark": "gsm",
  "predictions": "/actual/predictions.jsonl.gz",
  "config": "/actual/predictions.config.json",
  "receipt": "/actual/completion.json"
}
```

`receipt` is optional for historical outputs without one; when present, its hashes
and completion flag must pass. Checkpoint bytes, training/parent metadata, and
row-level evaluator identities are checked independently.

## 1. Inspect and measure existing opportunity

From a separate deployed copy of the repository on SOL:

```bash
python -B -m capstone_routing inspect --config configs/routing.local.json

python -B -m capstone_routing opportunity \
  --config configs/routing.local.json --output outputs/person4/routing/opportunity01

python -B -m capstone_routing patterns \
  --opportunity outputs/person4/routing/opportunity01 --output outputs/person4/routing/patterns01
```

`summary.json` names missing/invalid comparisons; incomplete blocks are not
replaced. The denominator remains all 500 questions per benchmark. Missing rows
and missing/unparseable answers are zero; checker failures and corrupt/mixed
inputs block the comparison. Pairwise rescues/risks and joint rescues are separate:
overlapping B/C1 rescues are counted once in the joint oracle.

`correctness_matrix.csv`, `opportunity.csv`, `run_index.csv`, and per-block matrices
preserve question text, benchmark-aware IDs, seed identity, lineage, and coverage.
The same command also analyzes the A-only three-seed pools, with their dependence
disclosed. Patterns use heuristic question cues only, mark support below 30, and
provide five deterministic examples per rescue/risk category. These descriptions
do not establish topic specialization.

## 2. Inventory exposures and prepare the independent pool

Populate `exposures` before declaring `exposure_inventory_complete: true`. Each
entry has `path`, `sha256`, `role`, and `checkpoint_ids` for affected checkpoints.
Roles are `parent_training`, `adapter_training`, `selection`, `reserved`,
`development`, and `official_test`. Development/final entries also name `benchmark`
(`gsm` or `math`). Actual parent/adapter manifests must match the hashes in their
training records. Include all historically exposed manifests, selection/scoring
banks, and reserved data, not merely the final selected rows.

```json
{
  "path": "/actual/train_A.jsonl",
  "sha256": "REPLACE_WITH_64_CHARACTER_SHA256",
  "role": "adapter_training",
  "checkpoint_ids": ["p17_A_t42", "p17_A_t137", "p17_A_t2026"]
}
```

Supply the SHA256 of the existing frozen `overlap_check.py` in `matcher`.
Its pure definitions are reused without running its CLI. All discovered related
families are connected transitively. Exact number-masked templates are grouped
even when short, a conservative addition to the historical long-question rule.
Unresolved exposure/template matches are excluded rather than admitted.

`sources` can point to hash-verified local JSONL exports with original question,
answer/solution, dataset ID/revision, and train split. Otherwise explicitly fetch
the pinned datasets:

```bash
python -B -m capstone_routing prepare-pool \
  --config configs/routing.local.json --registry outputs/person4/routing/opportunity01 \
  --output outputs/person4/routing/pool01 --download
```

If the pinned MATH source is unavailable from the Hub, supply a provenance-checked
local export of that revision; the pipeline does not substitute another dataset.
Preparation blocks if fewer than 1,500 eligible questions remain per benchmark.
It samples deterministically, then assigns entire families to approximately
60/20/20 splits. Group boundaries take precedence over exact split counts.

Question-only files contain exactly question ID, benchmark, text, split, and
family ID. Gold answers live separately under `grading/`. Keep that directory and
all future test labels out of manual model-selection work. The implemented split
checks cannot prove that a human has never inspected a public question, or that
the backbone never saw it during pretraining.

## 3. Label training/validation questions, embed, and fit

The following labeling commands run on a GPU allocation with the original native
evaluator Python. They submit no Slurm jobs themselves. Use an allocation with
the existing recorded A100 runtime and include this deployed repository on
`PYTHONPATH`. A complete experiment uses 18 existing checkpoints × 3,000 questions,
at most 54,000 responses. There is no new adapter training.

```bash
python -B -m capstone_routing label --config configs/routing.local.json \
  --registry outputs/person4/routing/opportunity01 --pool outputs/person4/routing/pool01 \
  --split train --output outputs/person4/routing/labels_train01 --authorize-gpu

python -B -m capstone_routing label --config configs/routing.local.json \
  --registry outputs/person4/routing/opportunity01 --pool outputs/person4/routing/pool01 \
  --split validation --output outputs/person4/routing/labels_validation01 --authorize-gpu
```

Completed 250-question chunks are hash-verified and reusable within an interrupted
attempt. An incomplete chunk is preserved and requires a new output attempt.
Pass `--reuse-labels /path/to/earlier/attempt` with the new output to reuse its
sealed chunks while generating only missing/incomplete chunks afresh. Reused
question/reference bytes and native prediction contracts must match; source
artifacts remain untouched. Keep `--authorize-gpu` because missing chunks can run.
To import previously completed predictions instead, use `--import-index` with an
absolute-path JSON list of entries containing `candidate`, `benchmark`, `split`,
`predictions`, `config`, optional `receipt`, and `questions: {path, sha256}`. The
question and gold-answer contents must match the prepared split. Imports require
no GPU authorization, but every required candidate/benchmark must be supplied.

In the separate CPU router environment:

```bash
python -B -m capstone_routing embed --pool outputs/person4/routing/pool01 \
  --output outputs/person4/routing/embeddings01

python -B -m capstone_routing train --registry outputs/person4/routing/opportunity01 \
  --pool outputs/person4/routing/pool01 --embeddings outputs/person4/routing/embeddings01 \
  --train-labels outputs/person4/routing/labels_train01 \
  --validation-labels outputs/person4/routing/labels_validation01 \
  --output outputs/person4/routing/training01

python -B -m capstone_routing freeze --training outputs/person4/routing/training01 \
  --output outputs/person4/routing/frozen01
```

The fixed MiniLM encoder uses all question tokens via bounded token windows,
content-token-weighted averaging, and L2 normalization. Existing sealed embeddings
are reusable. Logistic correctness classifiers use training questions only.
Validation chooses C from .01/.1/1/10, threshold from 0/.02/.05/.10/.15/.20, and
primary pools A/AB/AC1/ABC1. There is no JEV, reference loss, or generated-answer
feature. Single-class labels use a smoothed constant probability.
The encoder identity is pinned from its [published model repository](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
The MATH importer enumerates the dataset's [seven subject configurations](https://huggingface.co/datasets/EleutherAI/hendrycks_math/blob/21a5633873b6a120296cce3e2df9d5550074f4a3/README.md).

There are 18 frozen router scenarios: six primary, six fixed-three policy controls,
and six fixed-three A-only seed controls. For each parent/reference seed s,
`{A_s,B_s,C1_s}` is compared with `{A_42,A_137,A_2026}`. Both control pools retain
three candidates and share all data, preprocessing, searches, and inference rules.
They vary different checkpoint construction factors; their seed dependence is
reported rather than treated as independent replication.

## 4. Evaluate the frozen routing test once

After freezing, label the new router test split using native GPU Python:

```bash
python -B -m capstone_routing label --config configs/routing.local.json \
  --registry outputs/person4/routing/opportunity01 --pool outputs/person4/routing/pool01 \
  --split test --frozen outputs/person4/routing/frozen01 \
  --output outputs/person4/routing/labels_test01 --authorize-gpu
```

The evaluator's `--final_test` flag is used internally for this newly reserved
router split. The existing official GSM8K/MATH final benchmarks are not run.
Then, using CPU router Python:

```bash
python -B -m capstone_routing evaluate-test --frozen outputs/person4/routing/frozen01 \
  --pool outputs/person4/routing/pool01 --embeddings outputs/person4/routing/embeddings01 \
  --test-labels outputs/person4/routing/labels_test01 --output outputs/person4/routing/test01

python -B -m capstone_routing report --opportunity outputs/person4/routing/opportunity01 \
  --test outputs/person4/routing/test01 --output outputs/person4/routing/report01
```

Results include every switch, successful/harmful counts, paired family-bootstrap
intervals (2,000 draws), reference-seed variation, construction metadata, and a
random-routing control (1,000 permutations preserving test selection counts).
The primary contrast averages router-minus-A over six blocks using shared family
resamples. The secondary contrast compares fixed-three policy routing with
fixed-three A-only seed routing. No checkpoint pool is selected on test results.
Do not use these results to tune another version and call the same test untouched.

## 5. Route or answer a live question

```bash
python -B -m capstone_routing route --frozen outputs/person4/routing/frozen01 \
  --experiment primary_p17_t42 --question 'What is 25 percent of 80?'

python -B -m capstone_routing answer --config configs/routing.local.json \
  --frozen outputs/person4/routing/frozen01 --experiment primary_p17_t42 \
  --question 'What is 25 percent of 80?' --authorize-gpu
```

`route` needs the frozen encoder environment. `answer` additionally needs the
verified native GPU inference stack; install compatible router/encoder dependencies
in an isolated copy without changing the historical evaluator environment. The
frozen router validates recorded embedding versions, and the loader validates
the native GPU stack. Synthetic routers cannot be used for the live demonstration.

The answer result includes generated completion text, selected checkpoint/hash,
correctness scores, actual parent/adapter lineage, generated-token count, and
separate routing/loading/generation/end-to-end timings. Encoder initialization is
reported separately. Live generation uses batch size 1; offline labels use 16.
Do not infer accuracy or latency gains from a handful of demonstration questions.

The live bridge lifts the exact loader block from the hash-pinned evaluator and
uses its prompt renderer. It does not call the evaluator's scoring CLI or read a
gold answer. Each cache miss reconstructs the original backbone and loads the
verified cumulative final LoRA once. It never stacks a continuation adapter on an
already merged parent. Unknown save formats fail rather than guessing a lineage.

Python interfaces:

```python
from capstone_routing.router import FrozenRouter
from capstone_routing.live import Answerer

router = FrozenRouter(frozen_root, experiment="primary_p17_t42")
selection = router.route(question)

# Retain this object for its single-checkpoint cache. The evaluator dictionary
# is the resolved config['evaluator']; paths must be absolute here.
answerer = Answerer(frozen_root, evaluator, authorize_gpu=True,
                    experiment="primary_p17_t42")
result = answerer.answer(question)
```

## Local verification

```powershell
& 'C:/softwares/pythonAnaconda/python.exe' -B -m unittest discover -s tests/routing -v
```

Tests retain synthetic fixtures under `.test_artifacts/`. They cover scientific
invariants, fitting/freezing/final evaluation, native prompt/loader extraction,
and mocked live answers. They do not certify access to SOL, full pool eligibility,
encoder downloads, real-model reload parity, or actual routing improvements.
Before accepting the live demonstration, run the same non-test smoke questions
through both the native evaluator and the bridge on SOL and compare completion
tokens, prompt bytes, checkpoint hashes, and runtime. Preserve that evidence;
the local synthetic suite cannot replace a real GPU parity check.
