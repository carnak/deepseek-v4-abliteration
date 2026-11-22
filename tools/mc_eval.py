#!/usr/bin/env python3
"""A broader capability check: multiple-choice questions across many subjects (MMLU).

GSM8K asks whether the model can still do multi-step arithmetic. It is blunt:
builds with three times the drift score the same on it. This asks factual and
reasoning questions across dozens of subjects, where damage to what the model
knows would show first.

    python3 mc_eval.py --model <id> --problems mmlu_test500.jsonl

Problems file: one JSON object per line with "question", "choices" (list of
four) and "answer" (index 0-3), plus an optional "subject".
Thinking is off: the measure is what the model knows, not how well it reasons
its way round a gap. Standard library only.
"""
import argparse
import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import thinking

LETTERS = "ABCD"


def ask(args, p):
    options = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(p["choices"]))
    body = {
        "model": args.model,
        "messages": [{
            "role": "user",
            "content": f"{p['question']}\n\n{options}\n\nAnswer with the letter of the correct option only.",
        }],
        "max_tokens": 8,
        "temperature": 0.0,
        "chat_template_kwargs": thinking.kwargs(False),
    }
    req = urllib.request.Request(
        args.api_base.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        text = json.load(resp)["choices"][0]["message"]["content"] or ""
    found = re.search(r"\b([ABCD])\b", text.upper())
    return found.group(1) if found else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--api-base", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--problems", required=True)
    ap.add_argument("--out", help="optional JSON of per-question results")
    args = ap.parse_args()

    problems = [json.loads(line) for line in open(args.problems, encoding="utf-8")]
    with ThreadPoolExecutor(8) as pool:
        answers = list(pool.map(lambda p: ask(args, p), problems))

    correct = [a == LETTERS[p["answer"]] for a, p in zip(answers, problems)]
    unanswered = sum(a is None for a in answers)
    if args.out:
        json.dump({"model": args.model, "answers": answers, "correct": correct}, open(args.out, "w"))
    print(f"{args.model}: {sum(correct)}/{len(problems)} correct, {unanswered} with no letter")


if __name__ == "__main__":
    main()
