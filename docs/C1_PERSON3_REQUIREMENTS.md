# C1 generation: requirements for Person 3 or execution on Jahnvi's SOL account

For Jahnvi's actual completed run IDs, shared-package status, file mappings and
Megha's separate C0-evaluation/C1-generation tasks, start with
[Megha's evaluation handoff](MEGHA_EVALUATION_HANDOFF.md).

Prepared: 10 November 2026. Scope: the current **6,409-row shared Method B bank**.

## Decision

**Jahnvi can run this work on her own SOL account using Person 3's evaluator.**
It does not require Person 3 personally, new manual labels, or a different
dataset. Person 3 supplies the established generation/grading implementation;
the person operating it needs access to the same parent checkpoints, candidate
requests, compatible environment, cached base models, and an allocated GPU.

The current C1 sbatch pipeline exports requests, ingests results, checks
diagnostics, selects examples, and trains. It **does not schedule candidate
generation**. The steps below explain what that missing stage must do and how
the existing evaluator can be used. This document adds no execution code and
does not establish that generation has run on SOL.

## 1. Why C1 needs these results

C0 ranks reference question/solution pairs by the parent's teacher-forced loss.
C1 combines that fit with an independently measured mistake indicator:

```text
C0 utility = round(1,000,000 * fit)
C1 utility = C0 utility * (1 + confirmed_mistake)   # lambda = 1
```

The **original R1000 parent** must answer each candidate question without seeing
the gold answer or reference solution. Its response is then graded against the
existing extracted answer key. Only a confirmed incorrect response receives
the bonus. A missing or unparseable answer is not automatically a mistake.

Do not use the C0-trained adapter, a C1-trained adapter, an unadapted base model,
or a different parent's outcomes for this measurement.

## 2. Workload and timing

| Original parent | Candidate generations | Later C1 continuation seeds |
| --- | ---: | --- |
| Qwen2.5-0.5B, parent seed 17 | 6,409 | 42, 137, 2026 |
| Qwen2.5-0.5B, parent seed 123 | 6,409 | 42, 137, 2026 |
| Qwen2.5-1.5B, parent seed 17 | 6,409 | 42, 137, 2026 |
| Qwen2.5-1.5B, parent seed 123 | 6,409 | 42, 137, 2026 |
| **Total** | **25,636** | **12 full C1 training runs** |

Generate once per original parent. All three continuation seeds reuse its
generation results and C1 selection. The total above counts required complete
responses; disposable smoke runs and retries add work.

Request export needs completed preparation. Generation depends on the original
parent and frozen candidates, so it can run before C0 training finishes if
resources allow. The existing C1 submission script separately requires matching
completed C0 runs for every continuation seed, for paired handoffs.

Runtime must be measured with a real generation smoke. Do not treat C0 training
times or an sbatch time limit as an estimate for 6,409 fp32 generations.

## 3. What to give Person 3

Provide one request folder per parent, plus access to the original parent package.

| Item | Required contents / purpose |
| --- | --- |
| `request.json` | Parent model/revision, adapter location and SHA256, bank/request hashes, prompt hash, generation settings, parser and checker |
| `questions.jsonl` | Exactly 6,409 records with `problem_id`, `question`, `gold_answer`, dataset/source metadata and `split: candidate` |
| Original parent adapter | Matching `adapter_model.safetensors` and `adapter_config.json` |
| Original parent metadata | `run_config.json`, `training_summary.json`, parent index and checksum inventory |
| Frozen evaluation contract | The file referenced by the SOL config's `generation.generation_contract` |
| Accessible output location | A distinct output prefix for each parent, separate from benchmark results and source inputs |
| Evaluator version | Pinned evaluator/parser files or a Git commit, plus the chosen environment and GPU type |

The answer key is supplied **for grading only**. The inference prompt must use
the question alone. Do not pass `completion`, solution text or gold answers into
the model's prompt. Canonical candidate IDs and questions must stay traceable
to the exported request.

From Jahnvi's current C0 checkout, export a request with:

```bash
cd /scratch/jseth2/infj-llm-distillation-improvement
CFG=configs/c0.local.sol.0.5b.seed17.json
sbatch --job-name=q05-s17-c1-request scripts/sol/c1_export.sbatch "$CFG"
```

The job output prints `request_directory`. Repeat for the other three SOL configs
listed in the [C1 sbatch guide](C1_SOL_SBATCH_GUIDE.md). The four requests differ
in parent binding even though they use the common bank.

## 4. Frozen generation and grading settings

The actual exported request and resolved SOL config are authoritative.

| Setting | Current C1 requirement |
| --- | --- |
| Base model/revision | Exact model and full revision from the matching parent/request |
| Adapter | Original R1000 parent; SHA256 must match the request |
| Adapter mode | Merged |
| Precision | `fp32` |
| Decoding | Greedy, one response per question |
| Maximum new tokens | 1,024 |
| Batch size | 16 |
| Question cleaning | `p1_clean` |
| Prompt | Person 1 template; SHA256 `22e10ad6ce4d2f2a959577326b6eca275528821da148bae57a9a4ad2bf6e64db` |
| Special tokens | `add_special_tokens=false`; EOS ID 151643 |
| Extra question stopping | `stop_at_question=false` |
| Parser | `p3-parser-v6` |
| Mathematical checker | `math-verify==0.9.0` |
| GPU | Use a consistent A100 type/runtime for the measurement series |

This is fp32 parent inference with a merged adapter, unlike the 4-bit training
loader. Do not substitute `--precision 4bit`, `--no_merge`, a different prompt,
or a smaller token limit just to make a run fit. Measure memory and time first;
if frozen settings cannot run, stop and resolve the protocol explicitly.

### Runtime consistency

The supplied evaluation contract records Torch `2.14.1`; Jahnvi's current `c0`
environment reports `2.6.0+cu124`. That difference is recorded evidence, not
proof that the two runtimes generate identical answers. Choose and record one
working evaluator environment for all four parents before the full series.
If using a different runtime from historical Person 3 results, disclose it and
do not mix old and new generations as though they were the same measurement.

Use a separate evaluation environment if dependencies need changing. Do not
upgrade the active C0 training environment while its jobs are queued/running.
Person 3's requirements leave Torch unpinned, so a blanket installation is not
an environment reproduction procedure. Record actual installed versions and
run the parser tests and a real GPU smoke before the full bank.

## 5. What Person 3 must return

Return these for **each of the four original parents**:

1. `predictions.jsonl`: one complete prediction row per candidate ID, exactly
   6,409 unique IDs, with no missing or unexpected IDs.
2. `predictions.config.json`: the evaluator's actual sidecar config, emitted
   alongside that prediction file. This is not the parent's training config
   and not an aggregate accuracy report.
3. The request identifier/hash, evaluator commit and source hashes, environment,
   GPU/runtime record and logs sufficient to connect the output to the request.

The evaluator generates the sidecar name from `--out`: an output named
`parent17_predictions.jsonl` produces `parent17_predictions.config.json`.

Prediction rows should preserve the evaluator's full output, including:

```text
problem_id, model_revision, checkpoint_sha256, parser_version,
raw_output, parsed_answer, gold_answer, parse_status, correct,
generated_tokens, finish_reason, eval_config_hash
```

C1 ingestion requires usable `problem_id`, `parse_status`, `correct` and
nonnegative integer `generated_tokens`. It retains `parsed_answer` and
`finish_reason`; model/checkpoint/parser fields, when supplied on rows, must
match. Supply those provenance fields rather than omitting them.

The sidecar must match `model`, `revision`, `checkpoint_sha256`, `prompt_sha256`,
`parser_version`, `precision`, `decoding`, `max_new_tokens`, `eos_token_id`,
`add_special_tokens` and `adapter_mode`. Also retain batch size, cleaning/stopping
settings, input file hash, evaluator/parser hashes, checker version, GPU, CUDA
and package versions. The existing ingest function does not enforce every one
of these additional fields; operator verification is still required.

An overall accuracy percentage, dev-set predictions, or a partial smoke output
cannot replace these per-candidate records.

## 6. Findings from Person 3's branch

Reviewed local branch snapshot:
[`person_3-evaluation`, commit `7b9e496ee436a40b275771297a93085a8926cfeb`](https://github.com/jahnvisethjs/infj-llm-distillation-improvement/tree/7b9e496ee436a40b275771297a93085a8926cfeb).
The live GitHub page was unavailable during inspection; a newer remote commit
has not been independently checked. Pin and inspect the version actually used.

### Dataset labels: supported through existing CLI options

C1 exports `benchmark: orca` or `benchmark: numina`, preserving source identity.
Person 3's `evaluate.py:load_rows` accepts only `gsm` or `math` and dispatches to
different graders. Passing the export directly with default arguments stops.

An input rewrite is **not required**: the existing evaluator supports choosing
the benchmark field and a fallback. C1's export schema has no `c1_grading_route`
field, so these options route both sources to the mathematical checker:

```bash
--benchmark_field c1_grading_route --benchmark math
```

Verify that the chosen routing field is absent from every request row. Preserve
the original request file and its `orca`/`numina` labels as the source record.
Evaluator prediction rows will say `benchmark: math`; original source remains
recoverable by candidate ID. This follows the mathematical-checker policy used
by the overnight C1 implementation. Record this routing policy for every parent;
do not silently route one source through the GSM numeric grader instead.

### Parent metadata: a separate view is needed for this package layout

The inspected parent package stores training metadata at:

```text
R1000_seed17/run_config.json
R1000_seed17/training_summary.json
R1000_seed17/final/adapter/
```

The evaluator looks for `run_config.json` beside the `adapter/` directory, under
`final/`. It warns if that file is absent, and manifest verification then fails.
It also looks for `training_summary.json` one level above `final/`.

Do not change the checksummed original package. A separate evaluation view can
link to its original adapter and metadata, providing the expected layout. Verify
the adapter hash through the view before execution.

### Checker failures: retries need explicit handling

`answer_parser.py:score_math` can emit `parse_status: verify_error` on a checker
exception. Current C1 `classify()` does not accept that status; ingestion stops.
It must not be converted into a confirmed mistake.

Preserve such rows and resolve/retry them before C1 ingestion. The evaluator's
ordinary resume skips every already-written ID, including error rows, so merely
rerunning the same command will not retry them. Targeted retries and a checked
replacement/merge procedure are needed if these errors occur. No such automatic
retry/merge helper is supplied by the current C1 sbatch chain. Repeated failures
require an explicit checker-policy decision, not fabricated correctness values.

### Use the evaluator directly, not its default benchmark launcher

`run_eval.py` defaults to dev GSM/MATH files. Person 3's existing sbatch scripts
also contain different account/partition and home-directory assumptions. Do not
submit them unchanged for candidate generation. Call `evaluate.py` on the C1
request with the required parent and settings in an appropriate SOL GPU job.
The actual parser code is v6 even where the branch README still says v5.

## 7. Running on Jahnvi's SOL account

### Available inputs

Locations reported in the current SOL session:

```text
C0 checkout:       /scratch/jseth2/infj-llm-distillation-improvement
V4 data:           /scratch/jseth2/c0/data
Parent package:    /scratch/jseth2/c0/R1000_parents_seeds17_123_20261101T023416637853Z
Inherited release:/scratch/jseth2/c0/person1_v3_20261012_ready
B bank root:       /scratch/sshind38/capstone_approach_b/outputs/B_method_B_document_v3
C0/C1 outputs:     /scratch/jseth2/c0_c1_shared_bank_results
Model cache:       /scratch/jseth2/hf-cache
```

Code and input presence have been checked; real candidate generation, its memory
requirements, and evaluator compatibility on this account remain unverified.
The Windows workspace can prepare code/documents, but successful inference here
requires the SOL GPU allocation and the environment checks described above.

### Obtain evaluation code without disturbing C0

Use a **separate checkout**. Do not switch branches or pull scientific code in
the checkout used by queued/running C0 jobs.

Example, from a SOL terminal with GitHub access:

```bash
git clone --branch person_3-evaluation --single-branch \
  https://github.com/jahnvisethjs/infj-llm-distillation-improvement.git \
  /scratch/jseth2/person3_evaluation

git -C /scratch/jseth2/person3_evaluation rev-parse HEAD
```

Compare that commit with the reviewed snapshot above. If different, inspect the
evaluator/parser interface and contract before adopting it. Cache the exact
requested base-model revisions before GPU jobs; compute execution should use
the existing `HF_HOME` with `HF_HUB_OFFLINE=1`.

### Example parent metadata view

These commands describe setup to perform after reviewing paths; they are not
executed by creating this document. They create new links outside the parent
package and do not overwrite existing destinations:

```bash
ORIGINAL_RUN=/scratch/jseth2/c0/R1000_parents_seeds17_123_20261101T023416637853Z/qwen2_5_0_5b/R1000_seed17
VIEW=/scratch/jseth2/c0_c1_shared_bank_results/evaluator_parent_views/0.5B_seed17

mkdir -p "$VIEW/final"
ln -s "$ORIGINAL_RUN/final/adapter" "$VIEW/final/adapter"
ln -s "$ORIGINAL_RUN/run_config.json" "$VIEW/final/run_config.json"
ln -s "$ORIGINAL_RUN/training_summary.json" "$VIEW/training_summary.json"
```

If a link already exists, inspect it instead of replacing it. Repeat with the
correct model/parent paths, and verify adapter bytes against `request.json`.

### Example evaluator command inside an allocated GPU job

This command is a template, not a complete generation sbatch wrapper. Replace
`REQUEST_DIR` with the exported directory. Activate the approved evaluator
environment and verify a single allocated CUDA GPU before running it.

```bash
EVALUATOR=/scratch/jseth2/person3_evaluation/evaluation
REQUEST_DIR=/actual/exported/request/directory
OUT=/scratch/jseth2/c0_c1_shared_bank_results/candidate_generations/0.5B_seed17
export HF_HOME=/scratch/jseth2/hf-cache
export HF_HUB_OFFLINE=1
mkdir -p "$OUT"

python "$EVALUATOR/evaluate.py" \
  --eval_file "$REQUEST_DIR/questions.jsonl" \
  --out "$OUT/predictions.jsonl" \
  --run_id C1_candidates_0.5B_parent17 \
  --adapter "$VIEW/final/adapter" \
  --model Qwen/Qwen2.5-0.5B \
  --revision 060db6499f32faf8b98477b0a26969ef7d8b9987 \
  --expect_train_manifest_sha256 efa258c9c4a8bffa85bb3be2026c187e9035f3918fe044967c4dafadcfa7f303 \
  --precision fp32 --batch_size 16 --max_new_tokens 1024 \
  --question_text p1_clean \
  --benchmark_field c1_grading_route --benchmark math
```

Take the revision, adapter hash and manifest hash from verified metadata for
each request, rather than reusing the 0.5B example for 1.5B. The verified 1.5B
revision is `8faed761d45a263340a0528343f099c05c9a4323`.

Run parser tests first:

```bash
python -m pytest -q "$EVALUATOR/test_answer_parser.py"
```

Use `--limit` and a **separate output prefix** for a disposable GPU smoke. Check
the actual model, checkpoint hash, merged mode, parser/checker versions, prompt,
generated responses, token counts, elapsed time and peak memory. Keep the full
run at the frozen batch/settings; do not ingest a limited smoke as a full run.
Ordinary interrupted generation can resume compatible completed IDs, but must
not mix environments, GPUs, settings or checkpoints. Handle checker errors as
described above.

### Return results to C1

Once all candidate IDs have validly resolved outcomes and the matching C0 runs
are complete, use the existing submission script:

```bash
cd /scratch/jseth2/infj-llm-distillation-improvement
bash scripts/sol/submit_c1_pipeline.sh \
  configs/c0.local.sol.0.5b.seed17.json q05-s17-c1 \
  "$OUT/predictions.jsonl" "$OUT/predictions.config.json" \
  pilot01 c1pilot01 c1smoke01
```

Keep the C0 environment exports for this C1 submission; the generation process
may use its own environment. The C1 ingest job validates the outputs before
downstream selection/training. Diagnostic failure blocks dependent jobs; do not
relax thresholds to fill the run table. See the [C1 SOL guide](C1_SOL_SBATCH_GUIDE.md)
for seed selection, outputs and recovery.

## 8. Later benchmark evaluation is a separate handoff

Candidate generation enables C1 selection; it does not measure held-out accuracy.
After training, Person 3 or the same evaluator operator will need:

- Four original parents and the 12 reconciled C0 plus 12 reconciled C1 adapters.
- Matching B checkpoints if reporting B-versus-C0-versus-C1 comparisons.
- Run indexes, adapter checksums, model/parent/continuation seeds, selected
  manifests, token reconciliation and evaluation handoffs.
- Frozen development question files and a single consistent evaluator runtime
  for every compared arm. Keep GSM and MATH metrics separate.
- Paired results for the same parent, continuation seed and question IDs;
  C1-minus-C0 is the primary contrast. Do not pool the two model sizes.
- Separate final-test authorization and the predeclared final model list before
  using reserved test sets.

Current configs leave benchmark evaluation settings unresolved. Those must be
bound before using the repository's `evaluate`/`compare` stages. Training
completion and candidate outcomes alone do not establish an accuracy gain.

## 9. Operator completion checklist

- [ ] Four original parents and their metadata/checksums identified.
- [ ] Four matching requests exported from the 6,409-row shared bank.
- [ ] Evaluator/parser commit, mathematical routing policy and runtime recorded.
- [ ] Metadata views verified without modifying original parent packages.
- [ ] Parser tests and real GPU generation smoke pass.
- [ ] Four full prediction files, each containing exactly its 6,409 unique IDs.
- [ ] All settings/checkpoint hashes match; no checker failures are disguised as mistakes.
- [ ] Four matching `.config.json` sidecars and execution logs retained.
- [ ] Matching C0 runs for seeds 42/137/2026 complete before C1 chain submission.
- [ ] C1 ingestion, diagnostic gates, training reconciliation and handoffs pass.
- [ ] Later benchmark evaluation is reported separately.

## Evidence and related files

- [Person 3 evaluator at the inspected commit](https://github.com/jahnvisethjs/infj-llm-distillation-improvement/blob/7b9e496ee436a40b275771297a93085a8926cfeb/evaluation/evaluate.py): CLI, parent metadata lookup, input routing, generation, resume and output schema.
- [Person 3 parser at the inspected commit](https://github.com/jahnvisethjs/infj-llm-distillation-improvement/blob/7b9e496ee436a40b275771297a93085a8926cfeb/evaluation/answer_parser.py): parser v6, mathematical grading and `verify_error`.
- [C1 request/ingest implementation](../capstone_person4_approachC1/generation_scoring.py): request schema, correctness mapping and ingestion checks.
- [C1 implementation plan](../capstone_person4_approachC1/docs/C1_IMPLEMENTATION_PLAN.md): experimental design and generation handoff.
- [C1 SOL sbatch guide](C1_SOL_SBATCH_GUIDE.md): the currently implemented post-generation pipeline.
- Frozen contract: `data/inputs/evaluation/handoff/eval_contract.json` in the external V4 release; read the actual SOL config for its location.
