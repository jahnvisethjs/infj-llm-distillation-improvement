"""
Re-grade saved results with the CURRENT answer checker -- no model rerun needed.
Uses the saved raw_output of every problem. Keeps a backup of the old file and updates the
run's .config.json so the recorded grader always matches the rows.

  python rescore.py $R/base_gsm.jsonl $R/base_math.jsonl
"""
import json, os, shutil, sys
import answer_parser as P

for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    old_v = rows[0].get("parser_version", "p3-parser-v2 or earlier")
    if all(r.get("parser_version") == P.PARSER_VERSION for r in rows):
        print(f"{path}: already graded with {P.PARSER_VERSION}, skipped"); continue
    shutil.copy(path, path + f".{old_v.replace(' ', '_')}.bak")
    changed = 0
    for r in rows:
        score = P.score_gsm if r["benchmark"] == "gsm" else P.score_math
        s = score(r["raw_output"], r["gold_answer"])
        if (s["correct"], s["parse_status"]) != (r["correct"], r["parse_status"]):
            changed += 1
        r.update(parsed_answer=s["parsed_answer"], parse_status=s["parse_status"],
                 correct=s["correct"], parser_version=P.PARSER_VERSION)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    cfg_path = path.replace(".jsonl", "") + ".config.json"
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))
        cfg.setdefault("regrade_history", []).append({"from": old_v, "to": P.PARSER_VERSION, "changed": changed})
        cfg["parser_version"] = P.PARSER_VERSION
        json.dump(cfg, open(cfg_path, "w"), indent=2)
    print(f"{path}: re-graded {len(rows)} rows with {P.PARSER_VERSION}; {changed} changed (backup kept)")
