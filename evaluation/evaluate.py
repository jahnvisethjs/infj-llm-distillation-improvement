"""
Evaluator (Person 3, E3/E5/E6): model + eval file + config -> one JSONL row per problem.

Example (untrained baseline on dev_fast):
  python evaluate.py --eval_file data/dev_fast.jsonl --run_id base_devfast \
      --model Qwen/Qwen2.5-0.5B --out results/base_devfast.jsonl

Fine-tuned checkpoint (QLoRA adapter from Person 2):
  python evaluate.py --eval_file data/dev_gsm.jsonl --run_id R500 \
      --model Qwen/Qwen2.5-0.5B --adapter /path/to/R500_adapter --out results/R500_gsm.jsonl

Eval file format (JSONL, one problem per line):
  {"problem_id": "...", "benchmark": "gsm" | "math", "question": "...", "gold_answer": "...", ...}
  - gsm gold may be the raw GSM8K answer (with ####) or just the number.
  - math gold may be the bare answer or a full solution containing \\boxed{...}.
  Use --*_field flags if Person 1's files use other column names.

Re-running with the same --out skips problems already done (safe after OOM / Colab disconnect).
"""
from __future__ import annotations

import argparse, hashlib, json, os, time
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import answer_parser as P

# Zero-shot prompt -- EXACT copy of Person 2's training template (config/prompt_template.json,
# extracted from Person 1's notebook cell 25). Note the newline after "Solution:".
PROMPT_TEMPLATE = (
    "Question: {problem}\n"
    "Solve the problem step by step. End with: Final answer: <answer>.\n"
    "Solution:\n"
)
PROMPT_TEMPLATE_SHA256 = "22e10ad6ce4d2f2a959577326b6eca275528821da148bae57a9a4ad2bf6e64db"
assert hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest() == PROMPT_TEMPLATE_SHA256, \
    "Prompt differs from the training template"


def repetition_score(token_ids: list[int]) -> float:
    grams = [tuple(token_ids[i:i + 4]) for i in range(len(token_ids) - 3)]
    if len(grams) < 20:
        return 0.0
    return 1 - len(set(grams)) / len(grams)


def load_rows(args) -> list[dict]:
    rows = []
    with open(args.eval_file) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            gold = r[args.gold_field]
            bench = r.get(args.benchmark_field, args.benchmark)
            if bench == "math" and "\\boxed" in str(gold):
                gold = P.math_gold({"solution": gold})
            rows.append({"problem_id": str(r[args.id_field]), "benchmark": bench,
                         "question": r[args.question_field], "gold": str(gold)})
    if args.limit:
        rows = rows[: args.limit]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--revision", default="060db6499f32faf8b98477b0a26969ef7d8b9987",
                    help="pinned base-model revision (same as Person 1 tokenizer and Person 2 training)")
    ap.add_argument("--adapter", default=None, help="path to a PEFT/QLoRA adapter (fine-tuned runs)")
    ap.add_argument("--no_merge", action="store_true",
                    help="keep the adapter separate instead of merging it (frozen default: merged; "
                         "R500 test: grading identical 200/200, 25%% faster)")
    ap.add_argument("--precision", choices=["bf16", "fp16", "fp32", "4bit"], default="fp32",
                    help="fp32 = frozen choice: batched and one-at-a-time outputs were identical (48/48), "
                         "bf16 was not (17/48). Must be the SAME for every evaluated model.")
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--stop_at_question", action="store_true",
                    help="also stop generation at '\\nQuestion:' (team decision; off = plan default)")
    ap.add_argument("--limit", type=int, default=0, help="only first N problems (smoke test)")
    ap.add_argument("--id_field", default="problem_id")
    ap.add_argument("--question_field", default="question")
    ap.add_argument("--gold_field", default="gold_answer")
    ap.add_argument("--benchmark_field", default="benchmark")
    ap.add_argument("--benchmark", default=None, help="fallback if rows have no benchmark field")
    args = ap.parse_args()

    # ---- config + hash (identifies the evaluation contract)
    cfg = {k: getattr(args, k) for k in
           ["model", "revision", "precision", "max_new_tokens", "stop_at_question", "batch_size"]}
    cfg.update(prompt=PROMPT_TEMPLATE, prompt_sha256=PROMPT_TEMPLATE_SHA256, add_special_tokens=False, parser=P.PARSER_VERSION,
               cut_at_next_question=P.CUT_AT_NEXT_QUESTION, decoding="greedy")
    try:
        import math_verify, transformers
        cfg["math_verify"] = getattr(math_verify, "__version__", "unknown")
        cfg["transformers"] = transformers.__version__
    except Exception:
        pass
    if args.adapter:
        cfg["adapter_mode"] = "unmerged" if args.no_merge else "merged"
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows = load_rows(args)
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            prev = [json.loads(l) for l in f if l.strip()]
        other = sorted({r.get("eval_config_hash") for r in prev} - {cfg_hash})
        other_model = sorted({r.get("checkpoint_id") for r in prev} - {args.adapter or "base"})
        if other or other_model:
            raise SystemExit(
                f"STOP: {args.out} already contains results from different settings or a different "
                f"model (config {other or cfg_hash}, model {other_model or args.adapter or 'base'}). "
                f"Resuming would mix them. Use a new --out file name, or rename the old file.")
        done = {r["problem_id"] for r in prev}
    with open(args.out.replace(".jsonl", "") + ".config.json", "w") as f:   # written only after the check
        json.dump({**cfg, "eval_config_hash": cfg_hash, "run_id": args.run_id,
                   "adapter": args.adapter, "eval_file": args.eval_file}, f, indent=2)
    todo = [r for r in rows if r["problem_id"] not in done]
    print(f"{len(rows)} problems, {len(done)} already done, {len(todo)} to run | config {cfg_hash}")
    if not todo:
        return

    # ---- model
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():   # Apple Silicon Mac (local testing only)
        device = "mps"
    else:
        device = "cpu"
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    tok.padding_side = "left"                      # decoder-only batching
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {"revision": args.revision}
    if args.precision == "4bit":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        kw["device_map"] = "auto"
    else:
        kw["dtype"] = {"bf16": torch.bfloat16, "fp16": torch.float16,
                             "fp32": torch.float32}[args.precision]
    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        if not args.no_merge:
            model = model.merge_and_unload()      # adapter folded into the weights once
    if args.precision != "4bit":
        model.to(device)
    model.eval()
    print(f"model loaded in {time.time() - t_load:.1f}s on {device}")
    ctx_limit = getattr(model.config, "max_position_embeddings", 32768)

    gen_kw = dict(max_new_tokens=args.max_new_tokens, do_sample=False,
                  pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    if args.stop_at_question:
        gen_kw.update(stop_strings=["\nQuestion:"], tokenizer=tok)

    t_eval = time.time()
    with open(args.out, "a") as fout:
        for b in range(0, len(todo), args.batch_size):
            batch = todo[b: b + args.batch_size]
            prompts = [PROMPT_TEMPLATE.format(problem=r["question"]) for r in batch]
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
                # trim padding / everything after the first EOS (EOS itself not counted)
                if tok.eos_token_id in ids:
                    ids = ids[: ids.index(tok.eos_token_id)]
                    finish = "eos"
                else:
                    ids = [t for t in ids if t != tok.pad_token_id]
                    finish = "length" if len(ids) >= args.max_new_tokens else "stop_string"
                text = tok.decode(ids, skip_special_tokens=True)
                if r["benchmark"] == "gsm":
                    s = P.score_gsm(text, r["gold"])
                else:
                    s = P.score_math(text, r["gold"])
                fout.write(json.dumps({
                    "run_id": args.run_id, "model_revision": f"{args.model}@{args.revision}",
                    "checkpoint_id": args.adapter or "base", "eval_config_hash": cfg_hash,
                    "problem_id": r["problem_id"], "benchmark": r["benchmark"],
                    "raw_output": text, **s, "parser_version": P.PARSER_VERSION,
                    "generated_tokens": len(ids), "finish_reason": finish,
                    "repetition_score": round(repetition_score(ids), 4),
                    "latency": round(latency, 3),
                }) + "\n")
            fout.flush()
            print(f"  {min(b + args.batch_size, len(todo))}/{len(todo)} done")

    print(f"steady eval time {time.time() - t_eval:.1f}s -> run: python metrics.py {args.out}")


if __name__ == "__main__":
    main()
