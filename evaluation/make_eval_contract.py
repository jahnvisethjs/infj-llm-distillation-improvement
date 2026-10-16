"""
Freeze the evaluation contract (E3/E8) and hand it to Person 2's readiness gate.

  python make_eval_contract.py --who "Megha Suresh (Person 3)" --pins /scratch/$USER/eval_results/pinned_revisions.json

Writes handoff/eval_contract.json: frozen generation settings, grader version, and SHA-256 of
every file that defines the evaluation (code, question files, requirements). Person 2's
team_readiness.json references this file by path + sha256. If any listed file changes,
the hashes change and the contract must be re-issued (and affected runs redone).
"""
import argparse, datetime, hashlib, json, subprocess, sys
from pathlib import Path

import answer_parser as P
import evaluate as E

ap = argparse.ArgumentParser()
ap.add_argument("--who", required=True)
ap.add_argument("--out", default="handoff/eval_contract.json")
ap.add_argument("--gpu", default="NVIDIA A100", help="the ONE GPU type every evaluation runs on")
ap.add_argument("--pins", default=None, help="pinned_revisions.json (adds the 1.5B revision)")
a = ap.parse_args()

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "test_answer_parser.py"],
                       capture_output=True, text=True)
if tests.returncode != 0:
    sys.exit("STOP: grader tests fail -- fix before freezing.\n" + tests.stdout[-2000:])

defaults = E.build_parser().parse_args(["--eval_file", "x", "--out", "x", "--run_id", "x"])
defaults.model, defaults.revision = E.DEFAULT_MODEL
gen = E.generation_config(defaults)
for k in ("transformers", "peft", "torch", "model", "revision", "gpu", "cuda"):
    gen.pop(k)   # environment / per-model values are recorded per run, not frozen here

files = ["evaluate.py", "answer_parser.py", "metrics.py", "rescore.py", "run_eval.py", "test_answer_parser.py",
         "requirements.txt", "data/dev_gsm.jsonl", "data/dev_math.jsonl", "data/dev_fast.jsonl",
         "data/test_gsm.jsonl", "data/test_math500.jsonl", "data/eval_data_metadata.json",
         "handoff/frozen_development.csv", "handoff/frozen_development_metadata.json",
         "handoff/reserved_evaluation_metadata.json"]
pins = {E.DEFAULT_MODEL[0]: E.DEFAULT_MODEL[1]}
if a.pins:
    pins.update(json.loads(Path(a.pins).read_text()))
contract = {
    "schema_version": "person3-eval-contract-v1",
    "frozen_generation_settings": gen,
    "default_base_model": dict(zip(("model", "revision"), E.DEFAULT_MODEL)),
    "model_revisions": pins,
    "gpu_type": a.gpu,
    "software_versions": {p: E.version(p) for p in ("torch", "transformers", "peft", "accelerate", "math-verify")},
    "base_model_rule": "adapters are evaluated on the model/revision in Person 2's run_config.json",
    "prompt_template": E.PROMPT_TEMPLATE, "prompt_template_sha256": E.PROMPT_TEMPLATE_SHA256,
    "parser_version": P.PARSER_VERSION, "grader_tests": tests.stdout.strip().splitlines()[-1],
    "metrics": ["accuracy", "valid_answer_rate", "accuracy_among_valid", "length_limit_rate",
                "gen_tokens_mean/median/p95", "repetition_rate", "eval runtime"],
    "comparison": "paired bootstrap by problem_id, 2000 resamples, seed 0, 95% percentile CI",
    "dev_sets": {"dev_gsm": 500, "dev_math": 500, "dev_fast": 200},
    "final_tests": {"test_gsm": 1319, "test_math500": 500,
                    "rule": "evaluate.py refuses without --final_test; run once on a pre-declared model list"},
    "file_sha256": {f: sha(f) for f in files},
    "confirmed_by": a.who,
    "confirmed_utc": datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat(),
}
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(contract, indent=2) + "\n")
print("wrote", a.out)
print("sha256 for Person 2's team_readiness.json:", sha(a.out))
