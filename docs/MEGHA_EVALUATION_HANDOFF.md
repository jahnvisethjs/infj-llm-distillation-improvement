# Jahnvi to Megha: C0 accuracy evaluation and C1 candidate-generation handoff

Prepared: 10 November 2026 (America/Phoenix).
Sender: Jahnvi, SOL account `jseth2`. Recipient: Megha, SOL account **`msures15`**.

## Start here: two tasks, two different outputs

**Task A: evaluate the completed C0 adapters on frozen GSM8K and MATH development
sets to measure accuracy improvement over their matching original R1000 parents.**

**Task B: make each of the four original R1000 parents answer all 6,409 training
candidates and grade those responses. Return the per-candidate outputs so Jahnvi
can select C1 data and train C1.**

Task B is required before C1 selection. It is **not** required before Task A.
Task A's benchmark accuracies and benchmark predictions are **not** inputs to C1
selection. The two tasks can be scheduled independently with available resources.

| Task | Models | Input questions | Required return |
| --- | --- | --- | --- |
| A: C0 accuracy | 4 original parents + 12 C0 adapters | Frozen `dev_gsm.jsonl` and `dev_math.jsonl`, 500 each | Per-question predictions/configs, accuracy summaries and matched C0-minus-parent comparisons |
| B: enable C1 | Only the 4 original parents | 6,409 candidates per parent, from its exported request | 4 complete predictions JSONL files + their 4 evaluator `.config.json` sidecars |

Jahnvi retains responsibility for ingesting Task B outputs, applying the C1
diagnostic gate, selecting C1 examples and training its 12 continuation runs.
Megha does not need to train C1 or manually label candidate questions.

## 1. Confirmed work and remaining verification

This document records the SOL output pasted in the working session. It does not
claim direct inspection of the remote filesystem or contents of an unreported
new archive.

| Work / handoff | Evidence and status |
| --- | --- |
| C0 training | All 12 full training jobs listed below reported `COMPLETED`, exit `0:0` |
| First three C0 finalization jobs | `965291`, `970419`, `973644` reported `COMPLETED`, exit `0:0` |
| Last C0 finalization | Job `989666`; Jahnvi confirmed completion after checking it |
| First three C1 request exports | Jobs `989592`, `989593`, `989595` reported `COMPLETED`, exit `0:0` |
| Last C1 request export | Original job `989597` failed because preparation had not yet run; replacement job `997618` emitted a 6,409-row request and Jahnvi confirmed completion |
| C1 candidate generations | Not yet provided; this is Task B for Megha |
| C0 benchmark evaluation | Not yet reported; this is Task A for Megha |
| C1 selection/training | Not yet reported; awaits Task B and passing diagnostics |
| Initial shared archive | A 582 MB archive was listed in the earlier folder below; it predates the last finalization and request export |
| Refreshed complete archive | Commands were supplied to require 12 C0 adapters and 4 requests before packaging; its new timestamp/path and successful output have **not yet been pasted** |
| Access | Read-only ACL commands were supplied for `msures15`; Megha must verify access using her own account |
| Frozen evaluation data | Commands were supplied to share dev GSM, dev MATH and the contract; verify their presence and hashes in the refreshed folder |

The cancelled `ood-jupyter` job is separate from C0 and is not evidence of a
training failure. The mamba banner in stderr is informational by itself.

### Completed C0 run mapping

Every run uses full attempt **`pilot01`** and starts independently from its
original matching R1000 parent. Continuation seeds are not parent seeds.

| Model | Parent seed | Continuation seed | Completed training job | Finalization job |
| --- | ---: | ---: | ---: | ---: |
| 0.5B | 17 | 42 | 965286 | 965291 |
| 0.5B | 17 | 137 | 965287 | 965291 |
| 0.5B | 17 | 2026 | 965290 | 965291 |
| 0.5B | 123 | 42 | 970416 | 970419 |
| 0.5B | 123 | 137 | 970417 | 970419 |
| 0.5B | 123 | 2026 | 970418 | 970419 |
| 1.5B | 17 | 42 | 973641 | 973644 |
| 1.5B | 17 | 137 | 973642 | 973644 |
| 1.5B | 17 | 2026 | 973643 | 973644 |
| 1.5B | 123 | 42 | 989663 | 989666 |
| 1.5B | 123 | 137 | 989664 | 989666 |
| 1.5B | 123 | 2026 | 989665 | 989666 |

## 2. Where Jahnvi's files are and what has been shared

### Shared transfer folders

The earlier folder whose files were actually listed is:

```text
/scratch/jseth2/megha_handoff_20261111T025744Z/
    models_and_requests.tar.gz          # 582 MB when listed
    models_and_requests.tar.gz.sha256
    repo_commit.txt
```

The date in this folder name is UTC. It is an **earlier snapshot**, not an
assertion that it contains the final 12-run/four-request inventory. Do not use
the smaller incomplete attempts `...025727Z` or `...025737Z`.

Jahnvi was given commands to build a replacement folder named:

```text
/scratch/jseth2/megha_handoff_complete_<UTC timestamp>/
```

**Jahnvi must send Megha the exact `NEW SHARED HANDOFF` path printed by that
successful command.** Do not type the timestamp placeholder literally or assume
that the initial folder was automatically refreshed. Read access to the old
folder does not automatically grant access to a new folder; the packaging
commands grant the new ACL when they finish successfully.

The corrected recipient is **`msures15`**, not `msuresh15`. Commands were supplied
to remove the earlier incorrect named grant and grant read-only access to the
correct account. No write permission is required for Megha on Jahnvi's handoff.

### Source-to-package-to-task map

The refreshed archive preserves paths relative to `/scratch/jseth2`. After
extraction, `<EXTRACTED>` is the directory where Megha unpacked it.

| Material | Jahnvi's source on SOL | Refreshed handoff / extracted location | Used for |
| --- | --- | --- | --- |
| 12 C0 adapters | `c0_c1_shared_bank_results/<size>/<train-fingerprint>/runs/C0/pilot01/final/adapter/` | `<EXTRACTED>/c0_c1_shared_bank_results/.../final/adapter/` | Task A |
| C0 run metadata | Same `pilot01/` run: `run_config.json`, `resolved_config.json`, `training_summary.json`, `token_reconciliation.json` | Same relative run paths in archive | Model/parent/seed mapping, integrity and eligibility |
| Original parent adapters | `c0/R1000_parents_seeds17_123_20261101T023416637853Z/<model-dir>/R1000_seed<parent>/final/adapter/` | `<EXTRACTED>/c0/R1000_parents_seeds17_123_20261101T023416637853Z/...` | Task A baselines; Task B generation |
| Parent index, checksums and training records | Inside that original parent package | Same relative paths in archive | Parent identity, revisions, training manifest and completeness |
| C0 handoffs | `c0_c1_shared_bank_results/<size>/<train-fingerprint>/handoffs/<id>/evaluation_handoff.json` | Same relative paths in archive | Maps parent, continuation seed, adapter and hashes |
| C0 selections | `c0_c1_shared_bank_results/<size>/<select-fingerprint>/selection/C0/` | Same relative paths in archive | Selected IDs, token budget and provenance |
| Four C1 requests | `c0_c1_shared_bank_results/<size>/<select-fingerprint>/generation/request/<request-name>/` | Same relative paths in archive | Task B input; each folder contains `questions.jsonl` and `request.json` |
| Frozen GSM development file | `/scratch/jseth2/c0/data/inputs/evaluation/data/dev_gsm.jsonl` | Refreshed handoff root: `dev_gsm.jsonl` | Task A, 500 questions |
| Frozen MATH development file | `/scratch/jseth2/c0/data/inputs/evaluation/data/dev_math.jsonl` | Refreshed handoff root: `dev_math.jsonl` | Task A, 500 questions |
| Evaluation contract | `/scratch/jseth2/c0/data/inputs/evaluation/handoff/eval_contract.json` | Refreshed handoff root: `eval_contract.json` | Frozen prompt, parser, settings and file hashes |
| SOL training configs | `/scratch/jseth2/infj-llm-distillation-improvement/configs/c0.local.sol.*.json` | `<EXTRACTED>/infj-llm-distillation-improvement/configs/` | Training provenance; absolute/relative source paths may need interpretation |
| Repository revision | Training checkout's Git HEAD at packaging | Handoff root: `repo_commit.txt` | Code version for the handoff snapshot |
| Transfer checksums | Produced after the refreshed package is built | Handoff root: `SHA256SUMS.txt` and archive `.sha256` | Verify copied files |

The archive omits optimizer states, intermediate training checkpoints, smoke
runs and model caches. It is not a complete copy of the immutable source
packages. Their original checksum inventories may list files intentionally not
included; use the transfer checksums for archive integrity and the relevant
original entries to verify supplied adapters/metadata.

Sharing the transfer folder does not imply access to Jahnvi's entire results
tree, parent tree or model cache. Megha should work from her own extracted copy.
Matching B adapters must be obtained from Suraj separately if a B comparison is
required; they are not created by these C0-only submissions.

### Repository map

Repository: <https://github.com/jahnvisethjs/infj-llm-distillation-improvement>.

| Branch / file | Role |
| --- | --- |
| `person4/c0-reference-fit` | C0/C1 selection, training, exports, ingestion and batch handoffs |
| `person_3-evaluation` | Megha's model generation, answer parsing, grading and comparison implementation |
| [C1_PERSON3_REQUIREMENTS.md](C1_PERSON3_REQUIREMENTS.md) | Detailed generation contract, metadata views, CLI example and error handling |
| [C1_SOL_SBATCH_GUIDE.md](C1_SOL_SBATCH_GUIDE.md) | Jahnvi's post-generation C1 submission procedure |

The evaluator inspection used commit
`7b9e496ee436a40b275771297a93085a8926cfeb`. A newer branch revision must be checked
and recorded before use. A shared local document must be pushed to the branch
or copied separately if it was created after the package.

## 3. Megha: receive and verify the updated package

Use your own SOL account. Replace the single `SHARED` placeholder with the exact
refreshed path Jahnvi supplies; run from a new destination to avoid merging an
old and new copy.

```bash
whoami                         # must be msures15
SHARED=/scratch/jseth2/megha_handoff_complete_REPLACE_WITH_CONFIRMED_TIMESTAMP
ls -lh "$SHARED"

cp -R "$SHARED" /scratch/msures15/
WORK="/scratch/msures15/$(basename "$SHARED")"
cd "$WORK"

sha256sum -c SHA256SUMS.txt
tar -xzf models_and_requests.tar.gz
```

All checksums should pass. The older archive checksum used an absolute sender
path; the refreshed package uses relative checksums so verification checks
Megha's copy. Stop on checksum failure or an incomplete archive.

Verify supplied model/request counts without loading a GPU:

```bash
python - "$WORK" <<'PY'
from pathlib import Path
import json
import sys

work = Path(sys.argv[1])
results = work / "c0_c1_shared_bank_results"
adapters = list(results.glob("*/*/runs/C0/pilot01/final/adapter/adapter_model.safetensors"))
requests = list(results.glob("*/*/generation/request/*/request.json"))
parents = work / "c0/R1000_parents_seeds17_123_20261101T023416637853Z"
parent_adapters = list(parents.glob("*/R1000_seed*/final/adapter/adapter_model.safetensors"))
print("C0 adapters:", len(adapters), "expected 12")
print("Original parents:", len(parent_adapters), "expected 4")
print("C1 requests:", len(requests), "expected 4")
assert len(adapters) == 12 and len(parent_adapters) == 4 and len(requests) == 4
expected = {f"{size}-parentseed{seed}" for size in ("0.5B", "1.5B") for seed in (17, 123)}
records = [json.loads(path.read_text()) for path in requests]
assert {record["request_name"] for record in records} == expected
for path, request in zip(requests, records):
    rows = [json.loads(line) for line in (path.parent / "questions.jsonl").read_text().splitlines() if line.strip()]
    assert request["rows"] == len(rows) == 6409
    assert len({row["problem_id"] for row in rows}) == 6409
    print(request["request_name"], "OK", path.parent)
PY
```

Also verify `dev_gsm.jsonl` and `dev_math.jsonl` against the corresponding
`file_sha256["data/<filename>"]` entries in `eval_contract.json`. Transfer checksums
verify copying; contract hashes verify the intended frozen datasets.

## 4. Build a model-to-parent mapping before evaluation

Do not infer parent seed from a C0 run's `run_config.json.seed`: that field is the
**continuation** seed. Use the parent in `resolved_config.json` and the handoff.

For each C0 run, record:

```text
model_size, parent_seed, continuation_seed, method,
adapter_path, adapter_sha256, parent_adapter_sha256,
model, revision, training_manifest_sha256, token_reconciliation_path
```

Use the 12-run table above to check the complete matrix. The parent adapter
SHA256 in each C0 reconciliation must match the original parent selected for
its accuracy comparison. The handoff's `arms.C0.checkpoint` and report paths
refer to Jahnvi's filesystem; rebase those paths onto Megha's extracted tree.
Check the resulting file's hash before execution. Do not edit original training
records just to replace their saved paths.

| Source field | Meaning / use |
| --- | --- |
| C0 `resolved_config.json.active_profile` | Model size |
| C0 `resolved_config.json.parent.training_seed` | Original parent seed, 17 or 123 |
| C0 `token_reconciliation.json.continuation_seed` | Continuation seed, 42/137/2026 |
| C0 `token_reconciliation.json.checkpoint_sha256` | Expected trained adapter hash |
| C0 `token_reconciliation.json.parent_fingerprint` | Expected original parent adapter hash |
| C0 `run_config.json.manifest_sha256` | C0 selected training manifest; use for that C0 adapter's evaluator guard |
| Parent `run_config.json.manifest_sha256` | Original R1000 manifest; use for original-parent evaluator guard |
| C1 `request.json.adapter_sha256` | Original parent required for candidate generation |
| C1 `request.json.model` / `revision` | Exact base model/revision for that request |

**Do not pass the R1000 manifest hash when evaluating a C0 adapter.** Its selected
continuation manifest is different. Each `--expect_train_manifest_sha256` must
come from the training record for the adapter actually being evaluated.

The inspected parent package stores `run_config.json` at the run root, while
the evaluator looks beside `final/adapter/`. Create a separate parent metadata
view in Megha's storage as described in the [requirements document](C1_PERSON3_REQUIREMENTS.md#parent-metadata-a-separate-view-is-needed-for-this-package-layout).
Keep the original adapter bytes and verify their hash. Check C0 metadata layout
too; the checkpoint normally includes `final/run_config.json`.

## 5. Task A: C0 accuracy on frozen GSM8K and MATH development sets

### What to run

Evaluate **16 adapters/models**: the 4 original R1000 parents and 12 C0 adapters.
The original parents are adapted R1000 models, not untouched base models.

For each, evaluate separately on:

- `dev_gsm.jsonl`: 500 GSM8K development questions.
- `dev_math.jsonl`: 500 MATH development questions.

This yields **32 complete prediction files and 32 config sidecars**, or 16,000
question responses before smokes/retries. Original-parent dev predictions are
reused across that parent's three C0 comparisons. Keep the same settings,
cleaning, parser/checker version, GPU type and runtime across compared models.
Do not mix historical L4 or different-runtime outputs into the comparison.

Use the frozen fp32/greedy/merged-adapter settings in the contract and
requirements document. Run a disposable GPU smoke before the full series.
Reserve final-test datasets for the separately agreed final evaluation.

### Commands: one model and one comparison

Use a separate evaluator checkout/environment. Below `EVALUATOR`, `ADAPTER`,
`MANIFEST_SHA` and `RUN` are populated from the verified mapping above. `WORK`
is Megha's extracted handoff directory from section 3.

```bash
EVALUATOR=/scratch/msures15/person3_evaluation/evaluation
OUT=/scratch/msures15/jahnvi_evaluation_results/c0_dev
mkdir -p "$OUT"

# Set these for ONE verified parent or C0 adapter before running.
ADAPTER=/actual/verified/adapter/or/metadata-view/final/adapter
MANIFEST_SHA=actual_manifest_sha256_from_this_adapters_run_config
RUN=C0_0.5B_p17_t42

python "$EVALUATOR/evaluate.py" \
  --eval_file "$WORK/dev_gsm.jsonl" \
  --adapter "$ADAPTER" --expect_train_manifest_sha256 "$MANIFEST_SHA" \
  --precision fp32 --question_text p1_clean --batch_size 16 --max_new_tokens 1024 \
  --run_id "${RUN}_dev_gsm" --out "$OUT/${RUN}_dev_gsm.jsonl"

python "$EVALUATOR/evaluate.py" \
  --eval_file "$WORK/dev_math.jsonl" \
  --adapter "$ADAPTER" --expect_train_manifest_sha256 "$MANIFEST_SHA" \
  --precision fp32 --question_text p1_clean --batch_size 16 --max_new_tokens 1024 \
  --run_id "${RUN}_dev_math" --out "$OUT/${RUN}_dev_math.jsonl"
```

These commands rely on the verified metadata view/checkpoint record to resolve
the model and full revision. Confirm the printed model, revision and checkpoint
before accepting the run; pass explicit `--model` and `--revision` from verified
metadata if needed. Do not fall back to the evaluator's default 0.5B model.

Name original-parent outputs, for example, `PARENT_0.5B_p17_dev_gsm.jsonl`.
Name C0 outputs `C0_0.5B_p17_t42_dev_gsm.jsonl`. Apply the same naming convention
to both sizes, both parents and all three continuation seeds.

After the full corresponding files are available:

```bash
python "$EVALUATOR/metrics.py" \
  "$OUT/PARENT_0.5B_p17_dev_gsm.jsonl" \
  "$OUT/C0_0.5B_p17_t42_dev_gsm.jsonl" --compare
```

The comparison reports the second file minus the first. Repeat separately for
MATH and each C0 run. Require all 500 expected IDs on both sides before accepting
a comparison; do not report a result from only an intersecting subset. Preserve
the evaluator's `.config.json` sidecars, since settings checks depend on them.

### What to return for Task A

Return all 32 prediction/config pairs, logs and an index connecting every file
to its method, model, parent and continuation seed. Include:

```text
c0_accuracy_summary.csv
c0_vs_parent_comparisons.csv
evaluation_run_index.csv
```

The comparison table should contain **24 rows**: 12 C0 runs times two benchmarks.
Suggested columns:

```csv
model_size,parent_seed,continuation_seed,benchmark,n,parent_accuracy_pct,c0_accuracy_pct,delta_pp,ci95_low_pp,ci95_high_pp,wrong_to_correct,correct_to_wrong,valid_answer_rate_pct,length_limit_rate_pct
```

Use `delta_pp = c0_accuracy_pct - parent_accuracy_pct`. Report per-seed results
and a separate per-model/per-parent/per-benchmark mean and sample standard
deviation across the three continuation seeds. A training-seed SD and a paired
question bootstrap CI describe different uncertainty; label them separately.
Do not pool model sizes or call a positive point estimate alone a confirmed
accuracy improvement. Keep GSM8K and MATH results separate.

Task A's outputs can be returned before Task B finishes. They are useful to
assess C0, but are not substituted into the C1 mistake indicator or used to
silently retune the frozen C1 bonus.

## 6. Task B: the exact output Jahnvi needs before selecting C1

### Original-parent/request mapping

Use **only original parents**, never C0-trained adapters for this task.

| Parent | Parent package relative adapter | Request name | Successful export job |
| --- | --- | --- | ---: |
| 0.5B / 17 | `qwen2_5_0_5b/R1000_seed17/final/adapter/` | `0.5B-parentseed17` | 989592 |
| 0.5B / 123 | `qwen2_5_0_5b/R1000_seed123/final/adapter/` | `0.5B-parentseed123` | 989593 |
| 1.5B / 17 | `qwen2_5_1_5b/R1000_seed17/final/adapter/` | `1.5B-parentseed17` | 989595 |
| 1.5B / 123 | `qwen2_5_1_5b/R1000_seed123/final/adapter/` | `1.5B-parentseed123` | 997618 |

The confirmed last request source is:

```text
/scratch/jseth2/c0_c1_shared_bank_results/1.5B/e7ec84e493362fe0/generation/request/1.5B-parentseed123/
```

Its confirmed parent adapter SHA256 is:

```text
849ca63fb2ebffc489144c68f00ac8513f1b9cae0c827c9fe4586cf6b38d370d
```

For the other three paths/hashes, use the request inventory and each
`request.json`; do not guess artifact fingerprints. The old failed job `989597`
is not an additional required run.

| Model | Frozen base revision |
| --- | --- |
| `Qwen/Qwen2.5-0.5B` | `060db6499f32faf8b98477b0a26969ef7d8b9987` |
| `Qwen/Qwen2.5-1.5B` | `8faed761d45a263340a0528343f099c05c9a4323` |

### Generation rules and command differences from Task A

There are four generation runs of 6,409 questions: **25,636 responses total**.
Each result is reused across continuation seeds 42, 137 and 2026; do not run
the candidate measurement twelve times.

Use the exact exported `questions.jsonl`, preserving its IDs and contents.
Only its question enters the model prompt; gold answers are for grading.
Settings are fp32, greedy, 1,024 new tokens, batch 16, `p1_clean`, merged
adapter, EOS 151643, no added special tokens, no extra question-stop option,
parser `p3-parser-v6`, and `math-verify==0.9.0`. The exported request is
authoritative for the model, revision, adapter hash and prompt hash.

Task B needs the following existing evaluator CLI options because the source
labels are `orca`/`numina`, whereas the evaluator accepts `gsm`/`math`:

```bash
--benchmark_field c1_grading_route --benchmark math
```

The export schema has no `c1_grading_route` field, so this selects the math
checker for both sources without rewriting the source request. Verify that the
field is absent. Preserve original source labels in the request for provenance.
**Do not apply this override to Task A's GSM8K evaluation**, which uses its
normal GSM grading route.

Use the full candidate-generation example and metadata-view procedure in
[C1_PERSON3_REQUIREMENTS.md](C1_PERSON3_REQUIREMENTS.md#7-running-on-jahnvis-sol-account),
substituting Megha's extracted paths and evaluator environment. Use distinct
output prefixes for smokes, full candidate runs and benchmark runs.

### Return these four pairs, with this mapping

Recommended output layout under Megha's own scratch:

```text
c1_candidates/
    0.5B_parent17/predictions.jsonl
    0.5B_parent17/predictions.config.json
    0.5B_parent123/predictions.jsonl
    0.5B_parent123/predictions.config.json
    1.5B_parent17/predictions.jsonl
    1.5B_parent17/predictions.config.json
    1.5B_parent123/predictions.jsonl
    1.5B_parent123/predictions.config.json
    candidate_run_index.csv
    SHA256SUMS.txt
```

Suggested `candidate_run_index.csv` columns:

```csv
model_size,parent_seed,request_name,generation_key,rows,predictions_path,config_path,parent_adapter_sha256,questions_sha256,evaluator_commit,parser_version,math_verify_version,gpu
```

Every prediction file must contain exactly its request's 6,409 unique IDs,
with no duplicate, missing or extra IDs. Keep the evaluator's full records.
One illustrative JSONL row is:

```json
{"problem_id":"<candidate ID>","parse_status":"ok","correct":false,"parsed_answer":"42","generated_tokens":123,"finish_reason":"eos","parser_version":"p3-parser-v6","model_revision":"<model>@<revision>","checkpoint_sha256":"<original parent adapter hash>"}
```

These placeholders are a format illustration, not values to fabricate. Also
retain `raw_output`, `gold_answer`, `benchmark`, `run_id`, `eval_config_hash`,
timing and other provenance emitted by the evaluator.

Return the **actual evaluator-emitted `.config.json`** for each predictions file.
It must match the request's `model`, `revision`, original-parent
`checkpoint_sha256`, `prompt_sha256`, parser version, precision, decoding,
token limit, EOS, special-token policy and merged-adapter mode. Retain batch
size, cleaning/stopping settings, input hash, checker version, source hashes,
package versions, CUDA and GPU details too. Do not replace it with a training
config, rename a summary into a config, or edit settings to make validation pass.

### Error and mistake policy

| Outcome | C1 treatment |
| --- | --- |
| `parse_status: ok`, `correct: false` | Confirmed mistake; bonus applies |
| Correct answer | No mistake bonus |
| No answer, unparseable answer, unresolved ambiguity | No confirmed-mistake bonus; retain original status |
| Length limit | Retain the flag; use grading status. A truncated response without a parsed answer is not automatically a confirmed mistake |
| Checker/execution failure | Resolve/retry before ingestion; never label it incorrect just to complete coverage |

The inspected parser can emit `verify_error`, which current C1 ingestion rejects.
Report these rows and resolve them with a documented retry/checked merge.
Ordinary evaluator resume skips already-written IDs, including failed checker
rows, so rerunning the same command is not a retry strategy for those rows.
Do not drop IDs or silently change grader policy. See the detailed error-handling
discussion in the requirements document.

## 7. What Jahnvi needs back before C1

**Required to select C1:** the four complete Task B prediction/config pairs,
their mapping to requests/parents, and sufficient provenance to validate them.
**Not required to select C1:** completed Task A benchmark metrics. Request both
tasks, but do not make one an artificial prerequisite for the other.

Megha should put returned results under her own output folder and give Jahnvi
read access or transfer a checksum-indexed copy. Shared input access is read-only;
Megha should not write evaluator outputs into Jahnvi's handoff folder.

For each parent, Jahnvi then runs:

```bash
cd /scratch/jseth2/infj-llm-distillation-improvement

# Replace these with the actual returned files for 0.5B / parent 17.
PREDICTIONS=/actual/returned/c1_candidates/0.5B_parent17/predictions.jsonl
REPORT=/actual/returned/c1_candidates/0.5B_parent17/predictions.config.json

bash scripts/sol/submit_c1_pipeline.sh \
  configs/c0.local.sol.0.5b.seed17.json q05-s17-c1 \
  "$PREDICTIONS" "$REPORT" pilot01 c1pilot01 c1smoke01
```

Map the other three parent pairs to their corresponding configs:

| Returned candidate pair | Jahnvi's config | C1 tag |
| --- | --- | --- |
| `0.5B_parent17` | `configs/c0.local.sol.0.5b.seed17.json` | `q05-s17-c1` |
| `0.5B_parent123` | `configs/c0.local.sol.0.5b.seed123.json` | `q05-s123-c1` |
| `1.5B_parent17` | `configs/c0.local.sol.1.5b.seed17.json` | `q15-s17-c1` |
| `1.5B_parent123` | `configs/c0.local.sol.1.5b.seed123.json` | `q15-s123-c1` |

The chain validates and ingests generations, runs C1 diagnostics and selection,
performs a smoke, trains three continuation seeds and produces paired handoffs.
The existing gate blocks a parent if confirmed mistakes are below 5% or above
95%, or C0/C1 selection Jaccard overlap is at least 95%. Successful candidate
generation alone does not guarantee the parent passes this gate. Keep failures
visible; do not relax the gate to force 12 runs.

C1 benchmark evaluation happens after C1 training, using the same frozen dev
question IDs and consistent evaluator settings. It will compare C1 to C0 and
the matching original parent; matching B can be added if its adapters and
evaluation provenance are supplied separately.

## 8. Completion checklist and ready-to-send message

### Jahnvi

- [x] Complete 12 C0 training jobs and confirm finalization for all four parents.
- [x] Complete all four C1 request exports, including retry `997618`.
- [ ] Confirm the refreshed archive's exact path, 12 C0 adapters, 4 parents and 4 requests.
- [ ] Confirm updated checksums, frozen dev files/contract and `msures15` read access.
- [ ] Send this document and [C1_PERSON3_REQUIREMENTS.md](C1_PERSON3_REQUIREMENTS.md), pushing/copying the new documents if needed.

### Megha

- [ ] Copy and verify the refreshed package in her own scratch.
- [ ] Build the model/parent/continuation registry and verify adapter hashes.
- [ ] Resolve parent metadata views and freeze one evaluator environment/GPU series.
- [ ] Run Task A and return 32 prediction/config pairs plus C0 accuracy comparisons.
- [ ] Run Task B and return 4 complete candidate prediction/config pairs plus mapping/checksums.
- [ ] Report and resolve checker errors without fabricating mistakes or dropping IDs.

### Message

> Megha, the 12 C0 training runs and four C1 request exports are complete. I am
> sharing the refreshed handoff at **[insert the confirmed new folder path]**,
> read-only for **msures15**. Please verify its checksums and inventory first.
> **Task A:** evaluate the 12 C0 adapters and their four matching original R1000
> parents on frozen dev GSM8K and dev MATH, and return per-question outputs,
> configs and accuracy improvements. **Task B:** have only the four original
> parents answer their 6,409 candidate questions and grade them, returning four
> complete prediction JSONL/config pairs. Task B enables my C1 selection; it is
> separate from Task A and does not need twelve candidate-generation runs.
> Please use this document for file mappings and return formats and the C1
> requirements document for the evaluator compatibility details.

## Sources and limits

Run/job status and shared-folder listings above come from Jahnvi's pasted SOL
outputs and confirmations. Remote ACLs, a refreshed archive's contents/path,
new evaluator execution and benchmark accuracy have not been independently
verified in this workspace. Do not report them as completed until checked.

The evaluator interface and comparison behavior were read from the locally
available [Person 3 commit](https://github.com/jahnvisethjs/infj-llm-distillation-improvement/tree/7b9e496ee436a40b275771297a93085a8926cfeb).
Training handoff fields are implemented in [handoff.py](../capstone_person4/handoff.py);
C1 output validation is implemented in [generation_scoring.py](../capstone_person4_approachC1/generation_scoring.py).
