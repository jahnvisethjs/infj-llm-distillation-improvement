"""
Re-grade saved results with the CURRENT answer checker -- no model rerun needed.
Uses the saved raw_output of every problem. Keeps a backup of the old file.

  python rescore.py results/base_smoke.jsonl results/base_devfast.jsonl
"""
import json, shutil, sys
import answer_parser as P

for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    old_v = rows[0].get("parser_version", "p3-parser-v2 or earlier")
    if old_v == P.PARSER_VERSION:
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
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{path}: re-graded {len(rows)} rows with {P.PARSER_VERSION}; {changed} changed (backup kept)")
