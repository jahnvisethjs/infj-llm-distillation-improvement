# Evaluation notes (Person 3)

Decision log and pilot results. Each decision was made from model outputs or speed tests, never by choosing the setting that scored higher.

## Decisions

| # | Decision | Evidence |
|---|---|---|
| 1 | Dev sets from GSM8K **train** + MATH **train** (500 + 500, seed 1729); test sets untouched | Plan: dev for all decisions, test once at the end. 0 dev/test question overlap |
| 2 | 2 MATH problems with two different boxed answers replaced by the next problem of the same subject and level | Multi-answer problems can't be graded with one gold answer |
| 3 | Prompt = Person 2's training template exactly (incl. newline after `Solution:`), no special tokens | SHA256 `22e10ad6…` matches her recorded template |
| 4 | **fp32** instead of bf16 | 48 problems, batched vs one at a time: bf16 identical 17/48, fp32 identical 48/48. fp32 only ~10% slower |
| 5 | Keep **1,024**-token cap | 13% of baseline outputs hit the cap (plan trigger 5%); re-running them with 2,048 tokens recovered 1/200 answers, 23/24 still hit the limit (loops). Rule set in advance: switch only if ≥10/200 recovered |
| 6 | Grader v3: stop reading where the model starts a new task (`Question:`, `[Question]`, "You are an AI assistant") | Base model invents new problems after answering; 0 scores changed on re-grading |
| 7 | Grader v4: if the `Final answer:` line is empty, read the block below it | R500 writes `Final Answer:` + `\[\boxed{..}\]`; applied to all models; baseline unchanged, R500 MATH +3 correct |
| 8 | Adapters **merged** before generating | R500 on 200 problems: grading identical 200/200, 25% faster. Rule set in advance: ≥20% faster and ≥198/200 same |
| 9 | Grader v5: a `Final answer:` that starts a line beats one mid-sentence | Grading Person 1's 162 accepted training targets as if a model wrote them: v4 marked 1 wrong (`...to find the final answer: $$2.938 × 100 = 293.8$$ ... Final answer: 293.8`), v5 0 wrong. 59 tests pass |
| 10 | Question text cleaned like training prompts (Person 1 `clean_text`) | 134/500 GSM and 140/500 MATH dev questions had double spaces/tabs that training prompts never contain. Changes prompts, so all baselines are re-run |
| 11 | Grader version kept out of the generation config hash | Re-grading no longer makes a run look like a different configuration; `metrics.py --compare` refuses mixed grader versions |
| 12 | Adapters tied to their training record | Base model/revision read from Person 2's `run_config.json`; checkpoint SHA-256 and training-manifest SHA-256 saved on every row |
| 13 | Final tests locked behind `--final_test` | Plan E8: test results must not tune the pilot |
| 14 | All evaluations on **one GPU type (A100)**; GPU name + CUDA build are part of the generation config hash | Different GPU types can change greedy outputs; the hash stops a run from resuming on, or being mixed with, another platform. L4 pilot results are not compared with A100 results |
| 15 | Baselines re-run on A100 for both models (`base_a100` 0.5B, `base15` 1.5B); 1.5B adapters compared only with `base15` | Question-text cleaning (#10) and GPU change (#14) mean old baseline outputs can't just be re-graded |
| 16 | Grader v6: GSM numbers read a Unicode minus (`−5`) as negative and `.5` as 0.5 | Before v6 both lost the sign or decimal point and were graded wrong. Re-grading `base_a100_gsm` and `R500_05B_s42_gsm`: 0 changed. MATH unaffected (math_verify already handled both). 67 tests pass |
| 17 | Baseline vs fine-tuned comparison checks every setting except adapter on/off | `adapter_mode` is in the settings hash, so baseline and fine-tuned runs always had different hashes and `metrics.py --compare` always stopped. Now compares the `.config.json` settings (GPU, precision, prompt, versions, eval file…) and still refuses mixed grader versions. The summary table uses the same check and leaves out any unfair comparison |
| 18 | Each fine-tuned run names its baseline when a model has more than one | The summary silently used the last baseline in `runs.json` (could be the old L4 run). Now it stops and asks for `"baseline"` (`sol/add_run.py --baseline base_a100`) |
| 19 | Paired statistics: exact McNemar test, plus accuracy on problems where **both** models gave a valid answer | Plan promises paired tests; the both-valid comparison separates format gains from reasoning gains. R500 GSM: +14.0 pts overall (McNemar p≈5e-8) but +3.2 pts on both-valid (CI −1.9 to 8.5, p=0.28) |

## Grader validation

- 67 unit tests (v6); all 2,819 gold answers graded correct when given back as answers; 0 false positives with shuffled answers.
- Manual check of 20 baseline outputs (7 correct, 7 wrong, 6 invalid; random seed 7): grader agreed 20/20.
- All 14 MATH answers graded correct that contained several numbers were genuinely correct.

## Pilot results — SUPERSEDED (old adapters, unprotected data release, grader v4, raw question text)

Kept as a record only. Do not report or compare with new runs. Redo with Person 2's retrained adapters from the protected release.

| | Baseline | R500 | Change (95% paired bootstrap CI) |
|---|---|---|---|
| GSM accuracy | 40.0% | 53.6% | +13.6 pts (9.0 to 18.4) |
| GSM valid answers | 75.6% | 98.2% | |
| GSM accuracy among valid | 52.9% | 54.6% | |
| MATH accuracy | 15.2% | 25.8% | +10.6 pts (6.6 to 14.6) |
| MATH valid answers | 70.6% | 81.0% | |
| MATH accuracy among valid | 21.5% | 31.9% | |

Reading: on GSM the gain is mostly format (the model now answers in the expected form); on MATH accuracy among valid answers also rose ~10 points.

Run time per model, full dev set on L4: baseline ~36 min; R500 ~46 min unmerged (merged ~25% faster).

## Limitations and open items

- **Train/eval precision:** adapters are trained on a 4-bit base and evaluated on the fp32 base (merged). Same for every model, as the plan requires. One-time check pending: R500 on dev_fast with `--precision 4bit` vs fp32; record the agreement here.

- **Strict grading understates the baseline on GSM:** 77/500 answers are sentences with several numbers; in a pilot check about 4 in 10 contained the right number (up to ~6 pts).
- **MATH loops:** 19–22% of MATH outputs still hit the 1,024 cap for both models.
- **Failure review pending:** R500 got 44 GSM and 27 MATH problems wrong that the baseline got right.

## Results files

Saved in Google Drive `eval_results/` (not in git): one `.jsonl` row per problem plus a `.config.json` per run. Old bf16 runs and grader backups (`*.bak`) are kept as evidence.
