#!/usr/bin/env python3
"""A small capability check: grade-school maths word problems (GSM8K) against a served model.

First-token KL says an abliterated model's outputs moved; it does not say
whether the model got worse at anything. This asks a fixed set of multi-step
arithmetic problems and counts exact final answers, so two builds can be
compared on something with a right answer.

    python3 gsm8k_eval.py --model <id> --problems gsm8k_test200.jsonl [--thinking]

Problems file: one JSON object per line with "question" and "answer" (the
dataset's own format, where the answer ends in "#### <number>").
Standard library only.
"""
import argparse
import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import thinking

INSTRUCTION = (
    "Solve the problem. Show your working briefly, then give the final answer "
    "on its own last line in the form 'Answer: <number>'."
)


def number(text):
    text = text.replace(",", "").replace("$", "")
    found = re.findall(r"-?\d+(?:\.\d+)?", text)
    return float(found[-1]) if found else None


def ask(args, question):
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": f"{INSTRUCTION}\n\n{question}"}],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": thinking.kwargs(args.thinking),
    }
    req = urllib.request.Request(
        args.api_base.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.load(resp)["choices"][0]["message"]["content"] or ""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--api-base", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--problems", required=True)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--out", help="optional JSONL of responses")
    args = ap.parse_args()
    if args.max_tokens is None:
        args.max_tokens = 4000 if args.thinking else 700

    problems = [json.loads(line) for line in open(args.problems, encoding="utf-8")]
    with ThreadPoolExecutor(8) as pool:
        responses = list(pool.map(lambda p: ask(args, p["question"]), problems))

    correct = unanswered = 0
    rows = []
    for p, r in zip(problems, responses):
        gold = number(p["answer"].split("####")[-1])
        marked = re.findall(r"Answer:\s*(.+)", r)
        got = number(marked[-1]) if marked else None
        if got is None:
            unanswered += 1
        ok = got is not None and abs(got - gold) < 1e-6
        correct += ok
        rows.append({"question": p["question"], "gold": gold, "got": got, "ok": ok, "response": r})

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n = len(problems)
    mode = "thinking on" if args.thinking else "thinking off"
    print(f"{args.model} ({mode}): {correct}/{n} correct, {unanswered} with no 'Answer:' line")


if __name__ == "__main__":
    main()
