# Evaluation (Person 3)

Measures every model the same way on GSM8K and MATH. Decisions and pilot results: see `EVAL_NOTES.md`.

## Files

| File | What it does |
|---|---|
| `prepare_eval_data.py` | Downloads GSM8K + MATH, builds the dev and test question files (`data/`) |
| `make_dev_fast.py` | Builds the 200-problem quick-check set (`dev_fast`) |
| `check_eval_data.py` | Checks the question files (sizes, sources, no overlap, gold answers) |
| `answer_parser.py` | The grader: finds the model's final answer and checks it (version `p3-parser-v5`) |
| `test_answer_parser.py` | 59 grader tests: `pytest -q test_answer_parser.py` |
| `evaluate.py` | Runs a model on a question file and saves one result row per problem |
| `metrics.py` | Accuracy and other metrics; `--compare` = paired bootstrap between two models |
| `rescore.py` | Re-grades saved results with the current grader (no model rerun); updates `.config.json` |
| `make_eval_contract.py` | Freezes settings + file hashes into `handoff/eval_contract.json` for Person 2's readiness gate |

## Frozen evaluation settings (do not change between models)

| Setting | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-0.5B` @ `060db6499f32faf8b98477b0a26969ef7d8b9987` for the baseline; adapters use the model/revision in Person 2's `run_config.json` (so 1.5B adapters load the 1.5B base) |
| Prompt | Person 1's training template, checked by SHA256 inside `evaluate.py` |
| Question text | cleaned exactly like training prompts (Person 1 `clean_text`) |
| Decoding | greedy, one answer per problem, max 1,024 new tokens |
| Precision | fp32 (bf16 results depended on batch grouping) |
| Batch size | 16 |
| Adapters | merged into the model before generating (default) |
| GPU | Colab L4 for every evaluation |
| Grader | `p3-parser-v5` (strict: no "last number" guessing; line-start `Final answer:` first) |

All of these are the defaults, so the commands below need no extra flags.

## Data

| File | Size | Use |
|---|---|---|
| `dev_gsm`, `dev_math` | 500 + 500 (GSM8K train, MATH train) | every experiment |
| `dev_fast` | 100 + 100 from the dev sets | quick checks only |
| `test_gsm`, `test_math500` | 1,319 + 500 | final evaluation only, once |

The `.jsonl` files are not in git. Rebuild them exactly (pinned dataset versions, seed 1729):
```bash
python prepare_eval_data.py
python make_dev_fast.py --gsm data/dev_gsm.jsonl --math data/dev_math.jsonl --out data/dev_fast.jsonl
python check_eval_data.py        # expect: ALL CHECKS PASSED
```

## Evaluate a model (Colab, L4 GPU)

```bash
pip install -r requirements.txt
R=/content/drive/MyDrive/eval_results_v2      # NEW folder: old results used the old grader/prompt
N500=<train_n500 manifest sha256 from Person 1's NEW release>

# baseline (no adapter)
python evaluate.py --eval_file data/dev_gsm.jsonl  --run_id base05_gsm  --out $R/base05_gsm.jsonl
python evaluate.py --eval_file data/dev_math.jsonl --run_id base05_math --out $R/base05_math.jsonl

# a trained model: point at Person 2's <run>/final/adapter
A=<unzipped handoff>/run/final/adapter
python evaluate.py --eval_file data/dev_fast.jsonl --run_id R500_05_fast --adapter $A --expect_train_manifest_sha256 $N500 --out $R/R500_05_fast.jsonl
python evaluate.py --eval_file data/dev_gsm.jsonl  --run_id R500_05_gsm  --adapter $A --expect_train_manifest_sha256 $N500 --out $R/R500_05_gsm.jsonl
python evaluate.py --eval_file data/dev_math.jsonl --run_id R500_05_math --adapter $A --expect_train_manifest_sha256 $N500 --out $R/R500_05_math.jsonl

python metrics.py $R/R500_05_gsm.jsonl $R/R500_05_math.jsonl
cat $R/base05_gsm.jsonl $R/base05_math.jsonl   > $R/base05_dev.jsonl
cat $R/R500_05_gsm.jsonl $R/R500_05_math.jsonl > $R/R500_05_dev.jsonl
python metrics.py $R/base05_dev.jsonl $R/R500_05_dev.jsonl --compare
```

- Each run must start with `500 problems, 0 already done, 500 to run` and print the expected model.
- If Colab disconnects, rerun the same command: finished problems are skipped.
- `evaluate.py` STOPS if: the output file has other settings/another checkpoint; the adapter was
  trained on a different base model or (with `--expect_train_manifest_sha256`) different data;
  the adapter is a debug_32 or unfinished run; or the file is a final test without `--final_test`.
- `metrics.py --compare` STOPS if the two files used different grader versions or settings.
- 1.5B: also evaluate the untouched 1.5B base (`--model Qwen/Qwen2.5-1.5B --revision <from run_config.json>`)
  and compare 1.5B adapters only with that baseline.
- Report GSM and MATH separately: accuracy, valid-answer rate and accuracy among valid answers.
- If the grader changes again: bump `PARSER_VERSION`, run `rescore.py` on every result file, re-issue the contract.

## Many runs at once: registry + ASU Sol

`run_eval.py` runs every model listed in a registry (`runs.json`) on both dev sets, skips finished
ones, and writes `summary.csv` with the gain over the matching baseline and its 95% CI.
The model version comes from `pinned_revisions.json`; adapters are checked against Person 2's
`run_config.json`, and a `train_manifest_sha256` field makes the run stop if the adapter was trained on other data.

On Sol (interactive A100 session):
```bash
bash sol/setup_sol.sh <full 1.5B revision>          # once: env, checks, registry, model download
nohup bash sol/run_interactive.sh > logs/run_$(date +%m%d_%H%M).log 2>&1 &
python sol/add_run.py --run_id R500_05B_s42 --model 0.5B --adapter <run>/final/adapter \
    --budget 500 --seed 42 --train_manifest_sha256 <sha from Person 1 release>
```
`sol/first_run.sbatch` / `sol/run_eval.sbatch` do the same as batch jobs.
