# Evaluation (Person 3)

Measures every model the same way on GSM8K and MATH. Decisions and pilot results: see `EVAL_NOTES.md`.

## Files

| File | What it does |
|---|---|
| `prepare_eval_data.py` | Downloads GSM8K + MATH, builds the dev and test question files (`data/`) |
| `make_dev_fast.py` | Builds the 200-problem quick-check set (`dev_fast`) |
| `check_eval_data.py` | Checks the question files (sizes, sources, no overlap, gold answers) |
| `answer_parser.py` | The grader: finds the model's final answer and checks it (version `p3-parser-v4`) |
| `test_answer_parser.py` | 55 grader tests: `pytest -q test_answer_parser.py` |
| `evaluate.py` | Runs a model on a question file and saves one result row per problem |
| `metrics.py` | Accuracy and other metrics; `--compare` = paired bootstrap between two models |
| `rescore.py` | Re-grades saved results with the current grader (no model rerun) |

## Frozen evaluation settings (do not change between models)

| Setting | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-0.5B` @ `060db6499f32faf8b98477b0a26969ef7d8b9987` |
| Prompt | Person 2's training template, checked by SHA256 inside `evaluate.py` |
| Decoding | greedy, one answer per problem, max 1,024 new tokens |
| Precision | fp32 (bf16 results depended on batch grouping) |
| Batch size | 16 |
| Adapters | merged into the model before generating (default) |
| GPU | Colab L4 for every evaluation |
| Grader | `p3-parser-v4` (strict: no "last number" guessing) |

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
R=/content/drive/MyDrive/eval_results          # results folder

# baseline (no adapter)
python evaluate.py --eval_file data/dev_gsm.jsonl  --run_id base_gsm  --out $R/base_gsm.jsonl
python evaluate.py --eval_file data/dev_math.jsonl --run_id base_math --out $R/base_math.jsonl

# a trained model: same commands + --adapter
python evaluate.py --eval_file data/dev_gsm.jsonl  --run_id R1000_gsm  --adapter <path>/final/adapter --out $R/R1000_gsm.jsonl
python evaluate.py --eval_file data/dev_math.jsonl --run_id R1000_math --adapter <path>/final/adapter --out $R/R1000_math.jsonl

python metrics.py $R/R1000_gsm.jsonl $R/R1000_math.jsonl
cat $R/base_gsm.jsonl $R/base_math.jsonl  > $R/base_dev.jsonl
cat $R/R1000_gsm.jsonl $R/R1000_math.jsonl > $R/R1000_dev.jsonl
python metrics.py $R/base_dev.jsonl $R/R1000_dev.jsonl --compare
```

- Each run must start with `500 problems, 0 already done, 500 to run`.
- If Colab disconnects, rerun the same command: finished problems are skipped.
- `evaluate.py` refuses to resume into a file made with different settings or a different model.
- Report GSM and MATH separately: accuracy, valid-answer rate and accuracy among valid answers.
