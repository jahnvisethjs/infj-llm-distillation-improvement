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

## Grader validation

- 55 unit tests; all 2,819 gold answers graded correct when given back as answers; 0 false positives with shuffled answers.
- Manual check of 20 baseline outputs (7 correct, 7 wrong, 6 invalid; random seed 7): grader agreed 20/20.
- All 14 MATH answers graded correct that contained several numbers were genuinely correct.

## Pilot results (dev set, fp32, L4, grader v4)

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

- **R500 is a pilot:** one training seed; dev set not yet checked against the training data (Person 1; MetaMath rows are the main risk). R500 was evaluated **unmerged**; re-run it merged if it is kept for final results.
- **Strict grading understates the baseline on GSM:** 77/500 answers are sentences with several numbers; in a pilot check about 4 in 10 contained the right number (up to ~6 pts).
- **MATH loops:** 19–22% of MATH outputs still hit the 1,024 cap for both models.
- **Failure review pending:** R500 got 44 GSM and 27 MATH problems wrong that the baseline got right.

## Results files

Saved in Google Drive `eval_results/` (not in git): one `.jsonl` row per problem plus a `.config.json` per run. Old bf16 runs and grader backups (`*.bak`) are kept as evidence.
