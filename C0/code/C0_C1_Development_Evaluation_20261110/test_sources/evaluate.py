"""
Evaluator (Person 3, E3/E5/E6): model + eval file + config -> one JSONL row per problem.

Baseline (untrained base model):
  python evaluate.py --eval_file data/dev_gsm.jsonl --run_id base_gsm --out $R/base_gsm.jsonl

Trained checkpoint from Person 2 (point --adapter at <run>/final/adapter):
  python evaluate.py --eval_file data/dev_gsm.jsonl --run_id R500_0.5B_gsm \
      --adapter <run>/final/adapter --expect_train_manifest_sha256 <train_n500 sha from Person 1> \
      --out $R/R500_0.5B_gsm.jsonl
  The base model and revision are read from Person 2's <run>/final/run_config.json, so a
  1.5B adapter is automatically evaluated on the 1.5B base. A mismatch stops the run.

Final tests (test_gsm / test_math500) refuse to run unless --final_test is given.

Eval file format (JSONL, one problem per line):
  {"problem_id": "...", "benchmark": "gsm" | "math", "question": "...", "gold_answer": "...", ...}

Re-running with the same --out skips problems already done (safe after OOM / Colab disconnect).
Grading is versioned separately from generation: rescore.py can re-grade saved outputs
with a newer parser without regenerating.
"""
from __future__ import annotations

import argparse, hashlib, importlib.metadata, json, os, re, time, unicodedata
from pathlib import Path

import answer_parser as P

# Zero-shot prompt -- EXACT copy of Person 1's contract (person1_data_contract.PROMPT_TEMPLATE,
# contract person1-plain-text-labels-v3). Note the newline after "Solution:".
PROMPT_TEMPLATE = (
    "Question: {problem}\n"
    "Solve the problem step by step. End with: Final answer: <answer>.\n"
    "Solution:\n"
)
PROMPT_TEMPLATE_SHA256 = "22e10ad6ce4d2f2a959577326b6eca275528821da148bae57a9a4ad2bf6e64db"
assert hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest() == PROMPT_TEMPLATE_SHA256, \
    "Prompt differs from the training template"

EOS_ID = 151643  # <|endoftext|>: the token Person 1 appends to every training target
DEFAULT_MODEL = ("Qwen/Qwen2.5-0.5B", "060db6499f32faf8b98477b0a26969ef7d8b9987")
STOP_STRING = "\nQuestion:"


def p1_clean_text(value: str) -> str:
    """Copy of Person 1's pipeline_controls.clean_text (person1-v3.2).
    Training prompts were built from text cleaned this way (NFC, collapsed spaces/tabs,
    at most one blank line, stripped). Using it here keeps eval prompts in the training
    distribution: ~27% of GSM/MATH questions contain double spaces that training never saw."""
    text = unicodedata.normalize("NFC", str(value)).replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", text)).strip()


def render_prompt(question: str, mode: str) -> str:
    return PROMPT_TEMPLATE.format(problem=p1_clean_text(question) if mode == "p1_clean" else question)


def repetition_score(token_ids: list[int]) -> float:
    grams = [tuple(token_ids[i:i + 4]) for i in range(len(token_ids) - 3)]
    if len(grams) < 20:
        return 0.0
    return 1 - len(set(grams)) / len(grams)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def version(pkg: str) -> str:
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--model", default=None,
                    help="base model; default Qwen2.5-0.5B, or the model in the adapter's run_config.json")
    ap.add_argument("--revision", default=None, help="pinned base-model revision (same rule as --model)")
    ap.add_argument("--adapter", default=None, help="Person 2's <run>/final/adapter folder")
    ap.add_argument("--expect_train_manifest_sha256", default=None,
                    help="stop unless the adapter was trained on this exact Person 1 manifest")
    ap.add_argument("--no_merge", action="store_true",
                    help="keep the adapter separate instead of merging it (frozen default: merged)")
    ap.add_argument("--precision", choices=["bf16", "fp16", "fp32", "4bit"], default="fp32",
                    help="fp32 = frozen choice (batched == one-at-a-time 48/48). 4bit mirrors the "
                         "training quantization; use only for the one-time train/eval precision check.")
    ap.add_argument("--question_text", choices=["p1_clean", "raw"], default="p1_clean",
                    help="p1_clean = same text cleanup as training prompts (frozen default)")
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--stop_at_question", action="store_true",
                    help="also stop generation at '\\nQuestion:' (off = plan default)")
    ap.add_argument("--final_test", "--final", action="store_true",
                    help="required to evaluate test_gsm / test_math500 (once, at the end)")
    ap.add_argument("--limit", type=int, default=0, help="only first N problems (smoke test)")
    ap.add_argument("--id_field", default="problem_id")
    ap.add_argument("--question_field", default="question")
    ap.add_argument("--gold_field", default="gold_answer")
    ap.add_argument("--benchmark_field", default="benchmark")
    ap.add_argument("--benchmark", default=None, help="fallback if rows have no benchmark field")
    return ap


def generation_config(args) -> dict:
    """Everything that changes the generated text. The grader is NOT part of this hash:
    it is versioned per row (parser_version) and can be re-applied by rescore.py."""
    return dict(model=args.model, revision=args.revision, precision=args.precision,
                max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
                stop_at_question=args.stop_at_question, question_text=args.question_text,
                prompt_sha256=PROMPT_TEMPLATE_SHA256, add_special_tokens=False,
                decoding="greedy", eos_token_id=EOS_ID,
                adapter_mode=(None if not args.adapter else
                              "unmerged" if (args.no_merge or args.precision == "4bit") else "merged"),
                transformers=version("transformers"), peft=version("peft"), torch=version("torch"),
                **platform())


def platform() -> dict:
    """GPU type and CUDA build are part of the settings: greedy outputs can differ between GPU types,
    so a run can never be resumed (or silently mixed) on a different platform."""
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        return {"gpu": gpu, "cuda": torch.version.cuda}
    except ImportError:
        return {"gpu": "unknown", "cuda": None}


def resolve_checkpoint(args) -> dict:
    """Tie the adapter to Person 2's training record: base model, revision, data manifest, hash."""
    if not args.adapter:
        args.model = args.model or DEFAULT_MODEL[0]
        args.revision = args.revision or DEFAULT_MODEL[1]
        return {"checkpoint_id": "base", "checkpoint_sha256": None, "train_manifest_sha256": None}
    adapter = Path(args.adapter)
    weights = adapter / "adapter_model.safetensors"
    if not weights.is_file():
        raise SystemExit(f"STOP: {weights} not found. Point --adapter at <run>/final/adapter.")
    run_cfg_path = adapter.parent / "run_config.json"
    run_cfg = json.loads(run_cfg_path.read_text()) if run_cfg_path.is_file() else {}
    if not run_cfg:
        print(f"WARNING: {run_cfg_path} missing -- cannot confirm base model or training data.")
    train_model, train_rev = run_cfg.get("model"), run_cfg.get("revision")
    args.model = args.model or train_model or DEFAULT_MODEL[0]
    args.revision = args.revision or train_rev or DEFAULT_MODEL[1]
    if train_model and (args.model, args.revision) != (train_model, train_rev):
        raise SystemExit(f"STOP: adapter was trained on {train_model}@{train_rev}, "
                         f"but evaluation would load {args.model}@{args.revision}.")
    if run_cfg.get("mode") == "debug":
        raise SystemExit("STOP: this is a debug_32 adapter (discard it); evaluate curve/branch runs only.")
    manifest = run_cfg.get("manifest_sha256")
    if args.expect_train_manifest_sha256 and manifest != args.expect_train_manifest_sha256:
        raise SystemExit(f"STOP: adapter was trained on manifest {manifest}, "
                         f"expected {args.expect_train_manifest_sha256} (old/unprotected data release?).")
    summary = adapter.parent.parent / "training_summary.json"
    if summary.is_file() and json.loads(summary.read_text()).get("complete") is not True:
        raise SystemExit("STOP: training run is not complete (paused checkpoint).")
    return {"checkpoint_id": str(adapter), "checkpoint_sha256": sha256_file(weights),
            "train_manifest_sha256": manifest, "train_rows": run_cfg.get("rows"),
            "train_seed": run_cfg.get("seed"), "train_mode": run_cfg.get("mode"),
            "train_arm": run_cfg.get("arm")}


def load_rows(args) -> list[dict]:
    rows = []
    with open(args.eval_file, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            gold = r[args.gold_field]
            bench = r.get(args.benchmark_field, args.benchmark)
            if bench not in ("gsm", "math"):
                raise SystemExit(f"STOP: unknown benchmark {bench!r} for {r.get(args.id_field)}")
            if bench == "math" and "\\boxed" in str(gold):
                gold = P.math_gold({"solution": gold})
            rows.append({"problem_id": str(r[args.id_field]), "benchmark": bench,
                         "question": r[args.question_field], "gold": str(gold),
                         "split": r.get("split", "")})
    is_test = any(r["split"] == "test" for r in rows) or Path(args.eval_file).name.startswith("test_")
    if is_test and not args.final_test:
        raise SystemExit("STOP: this is a final-test file. Final tests are evaluated once, at the end, "
                         "on a pre-declared model list. Re-run with --final_test only then.")
    if args.limit:
        rows = rows[: args.limit]
    return rows


def finish_reason(ids: list[int], text: str, args) -> str:
    # PAD == EOS for Qwen, so a stop-string row is padded with EOS: check the text first.
    if args.stop_at_question and text.endswith(STOP_STRING):
        return "stop_string"
    if EOS_ID in ids:
        return "eos"
    return "length" if len(ids) >= args.max_new_tokens else "unknown"


def main():
    args = build_parser().parse_args()
    ckpt = resolve_checkpoint(args)
    rows = load_rows(args)

    gen_cfg = generation_config(args)
    cfg_hash = hashlib.sha256(json.dumps(gen_cfg, sort_keys=True).encode()).hexdigest()[:12]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            prev = [json.loads(l) for l in f if l.strip()]
        other_cfg = sorted({r.get("eval_config_hash") for r in prev} - {cfg_hash})
        other_ckpt = sorted({str(r.get("checkpoint_sha256")) for r in prev} - {str(ckpt["checkpoint_sha256"])})
        if other_cfg or other_ckpt:
            raise SystemExit(
                f"STOP: {args.out} already contains results from different settings ({other_cfg}) or a "
                f"different checkpoint ({other_ckpt}). Resuming would mix them. Use a new --out name.")
        done = {r["problem_id"] for r in prev}
    config_path = args.out.replace(".jsonl", "") + ".config.json"
    config = {**gen_cfg, "eval_config_hash": cfg_hash, "run_id": args.run_id, **ckpt,
              "parser_version": P.PARSER_VERSION, "cut_at_next_question": P.CUT_AT_NEXT_QUESTION,
              "math_verify": version("math-verify"), "eval_file": args.eval_file,
              "eval_file_sha256": sha256_file(args.eval_file),
              "evaluate_py_sha256": sha256_file(__file__),
              "answer_parser_sha256": sha256_file(P.__file__)}
    with open(config_path, "w") as f:   # written only after the resume check
        json.dump(config, f, indent=2)
    todo = [r for r in rows if r["problem_id"] not in done]
    print(f"{len(rows)} problems, {len(done)} already done, {len(todo)} to run | config {cfg_hash} | "
          f"model {args.model}@{args.revision[:8]} | checkpoint {ckpt['checkpoint_id']}")
    if not todo:
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision, use_fast=True)
    if tok.eos_token_id != EOS_ID:
        raise SystemExit(f"STOP: tokenizer EOS is {tok.eos_token_id}, training used {EOS_ID}")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"                      # decoder-only batching
    kw = {"revision": args.revision}
    if args.precision == "4bit":                   # mirrors Person 2's training quantization exactly
        from transformers import BitsAndBytesConfig
        compute = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute)
        kw["device_map"] = {"": 0}
    else:
        kw["dtype"] = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        if config["adapter_mode"] == "merged":
            model = model.merge_and_unload()      # adapter folded into the weights once
    if args.precision != "4bit":
        model.to(device)
    model.eval()
    load_seconds = time.time() - t_load
    print(f"model loaded in {load_seconds:.1f}s on {device}")
    ctx_limit = getattr(model.config, "max_position_embeddings", 32768)

    gen_kw = dict(max_new_tokens=args.max_new_tokens, do_sample=False,
                  pad_token_id=tok.pad_token_id, eos_token_id=EOS_ID)
    if args.stop_at_question:
        gen_kw.update(stop_strings=[STOP_STRING], tokenizer=tok)

    t_eval = time.time()
    with open(args.out, "a", encoding="utf-8") as fout:
        for b in range(0, len(todo), args.batch_size):
            batch = todo[b: b + args.batch_size]
            prompts = [render_prompt(r["question"], args.question_text) for r in batch]
            enc = tok(prompts, return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)  # same as training policy
            if enc.input_ids.shape[1] + args.max_new_tokens > ctx_limit:
                raise RuntimeError("prompt + max_new_tokens exceeds context; no silent truncation")
            t0 = time.time()
            with torch.inference_mode():
                gen = model.generate(**enc, **gen_kw)
            latency = (time.time() - t0) / len(batch)
            new = gen[:, enc.input_ids.shape[1]:]

            for r, ids in zip(batch, new.tolist()):
                raw_ids = ids
                ids = ids[: ids.index(EOS_ID)] if EOS_ID in ids else ids   # EOS itself not counted
                text = tok.decode(ids, skip_special_tokens=True)
                s = P.score_gsm(text, r["gold"]) if r["benchmark"] == "gsm" else P.score_math(text, r["gold"])
                fout.write(json.dumps({
                    "run_id": args.run_id, "model_revision": f"{args.model}@{args.revision}",
                    "checkpoint_id": ckpt["checkpoint_id"], "checkpoint_sha256": ckpt["checkpoint_sha256"],
                    "train_manifest_sha256": ckpt["train_manifest_sha256"],
                    "eval_config_hash": cfg_hash, "problem_id": r["problem_id"], "benchmark": r["benchmark"],
                    "raw_output": text, **s, "parser_version": P.PARSER_VERSION,
                    "generated_tokens": len(ids), "finish_reason": finish_reason(raw_ids, text, args),
                    "repetition_score": round(repetition_score(ids), 4),
                    "latency": round(latency, 3),
                }) + "\n")
            fout.flush()
            print(f"  {min(b + args.batch_size, len(todo))}/{len(todo)} done")

    eval_seconds = time.time() - t_eval
    config.update(load_seconds=round(load_seconds, 1), steady_eval_seconds=round(eval_seconds, 1),
                  problems_per_second=round(len(todo) / eval_seconds, 3), device=device,
                  gpu=torch.cuda.get_device_name(0) if device == "cuda" else device)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"steady eval time {eval_seconds:.1f}s -> run: python metrics.py {args.out}")


if __name__ == "__main__":
    main()
