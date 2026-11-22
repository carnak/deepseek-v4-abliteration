#!/usr/bin/env python3
"""Single-stream decode speed of a served model, in tokens per second.

An edit can cost speed without costing quality. DeepSeek-V4 is served with
speculative decoding from its own draft head (DSpark), which predicts the next
tokens from the main model's hidden states; an edit moves those states, and if
the draft's guesses are accepted less often the served model gets slower. Run
this on the control and on each candidate, one request at a time, so the number
is the decode rate a single user sees.

    python3 decode_speed.py --model <id> [--api-base URL] [--runs 3] [--tokens 512]

Prints "<tok/s> tok/s median of N" (choose.py reads the first number). Thinking
off, fixed prompts, temperature 0, ignore_eos so every run decodes the same length.
Standard library only.
"""
import argparse
import json
import statistics
import time
import urllib.request

import thinking

PROMPTS = [
    "Write a detailed history of the printing press, from Gutenberg to the twentieth century.",
    "Explain, step by step and with examples, how a hash table works and how it handles collisions.",
    "Describe the water cycle in detail, covering evaporation, condensation, precipitation and runoff.",
]


def one(args, prompt):
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "chat_template_kwargs": thinking.kwargs(False),
    }
    req = urllib.request.Request(args.api_base.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as resp:
        out = json.load(resp)
    dt = time.time() - t0
    return out["usage"]["completion_tokens"] / dt


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--api-base", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=512)
    args = ap.parse_args()
    one(args, "Say hello.")          # warm-up, not counted
    rates = [one(args, PROMPTS[i % len(PROMPTS)]) for i in range(args.runs)]
    print(f"{statistics.median(rates):.1f} tok/s median of {len(rates)} "
          f"({', '.join(f'{r:.1f}' for r in rates)}; {args.tokens} tokens each, includes prefill)")


if __name__ == "__main__":
    main()
