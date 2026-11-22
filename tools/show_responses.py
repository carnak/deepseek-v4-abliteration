#!/usr/bin/env python3
"""Print response openings from an eval_refusals.py JSONL, grouped by how they were flagged.

The counts say how many responses tripped a marker; this is for seeing what the
responses in each group actually are.

    python3 show_responses.py responses.jsonl [chars] [per_group]
"""
import json
import sys

path = sys.argv[1]
chars = int(sys.argv[2]) if len(sys.argv) > 2 else 110
per_group = int(sys.argv[3]) if len(sys.argv) > 3 else 8

rows = [json.loads(line) for line in open(path, encoding="utf-8")]
groups = {
    "explicit refusal": [r for r in rows if r["explicit_refusal"]],
    "keyword only (no explicit refusal)": [
        r for r in rows if r["keyword_refusal"] and not r["explicit_refusal"]
    ],
    "no flag": [r for r in rows if not r["keyword_refusal"] and not r["explicit_refusal"]],
}
for name, group in groups.items():
    print(f"--- {name}: {len(group)}")
    for r in group[:per_group]:
        print("   ", " ".join(r["response"].split())[:chars])
