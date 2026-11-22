#!/usr/bin/env python3
"""Split a judged response file into prompt lists by what the model did.

    python3 split_by_verdict.py responses.jsonl <out dir> <stem>

Writes <stem>_complied.txt (verdict COMPLY) and <stem>_resisted.txt (REFUSE or
DEFLECT), one prompt per line. PARTIAL and UNCLEAR are left out of both: the
point is a clean contrast for the lab (lab.cli directions --good-file
<stem>_complied.txt --bad-file <stem>_resisted.txt), and the middle is not one.

Use responses to the MEASUREMENT prompts. A direction built from the tuning or
held-out prompts and then scored on them would be grading its own homework.
"""
import json
import os
import sys

path, out_dir, stem = sys.argv[1:4]
rows = [json.loads(line) for line in open(path, encoding="utf-8")]
complied = [r["prompt"] for r in rows if r.get("verdict") == "COMPLY"]
resisted = [r["prompt"] for r in rows if r.get("verdict") in ("REFUSE", "DEFLECT")]
os.makedirs(out_dir, exist_ok=True)
for name, items in ((f"{stem}_complied.txt", complied), (f"{stem}_resisted.txt", resisted)):
    with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
        f.write("\n".join(" ".join(p.split()) for p in items) + "\n")
print(f"{len(rows)} responses: {len(complied)} complied, {len(resisted)} resisted, "
      f"{len(rows) - len(complied) - len(resisted)} left out")
