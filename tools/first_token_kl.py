#!/usr/bin/env python3
"""How far a served model's answers have drifted from a reference, on harmless prompts.

Heretic's second objective is the KL divergence between the original and the
abliterated model's first-token distributions over harmless prompts: a cheap
proxy for "did this damage the model". This measures the same thing against a
served endpoint, in two steps, because only one model fits on the box at once:

    # with the reference (stock) model served
    python3 first_token_kl.py dump --model <id> --prompts harmless.txt --out stock.json
    # with the modified model served
    python3 first_token_kl.py dump --model <id> --prompts harmless.txt --out mine.json
    python3 first_token_kl.py compare stock.json mine.json

The API returns only the top 20 tokens, so this is KL over the reference's
top-20 with the remaining mass lumped into one bucket -- a lower bound on the
true value, close to it when the distributions are peaked, and comparable
between runs made the same way. Not comparable to heretic's number digit for
digit.
"""
import argparse
import json
import math
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import thinking

TOP = 20


def first_token(args, prompt):
    body = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": TOP,
        "chat_template_kwargs": thinking.kwargs(False),
    }
    if args.token_ids:
        # Keys "token_id:<n>" instead of token text, so a dump can be compared with
        # one written outside the server (lab.cli dsv4-check) without matching strings.
        body["return_tokens_as_token_ids"] = True
    req = urllib.request.Request(
        args.api_base.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        top = json.load(resp)["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    return {t["token"]: t["logprob"] for t in top}


def dump(args):
    prompts = [p.strip() for p in open(args.prompts, encoding="utf-8") if p.strip()]
    with ThreadPoolExecutor(8) as pool:
        dists = list(pool.map(lambda p: first_token(args, p), prompts))
    json.dump({"model": args.model, "prompts": prompts, "dists": dists}, open(args.out, "w"))
    print(f"{len(dists)} first-token distributions from {args.model} -> {args.out}")


def kl(p, q):
    """KL(p || q) over p's top tokens plus one bucket for everything else."""
    floor = math.log(1e-6)
    total = 0.0
    p_mass = q_mass = 0.0
    for token, lp in p.items():
        lq = q.get(token, floor)
        total += math.exp(lp) * (lp - lq)
        p_mass += math.exp(lp)
        q_mass += math.exp(lq)
    p_rest, q_rest = max(1 - p_mass, 1e-9), max(1 - q_mass, 1e-9)
    return total + p_rest * (math.log(p_rest) - math.log(q_rest))


def compare(args):
    a, b = json.load(open(args.reference)), json.load(open(args.other))
    assert a["prompts"] == b["prompts"], "the two dumps used different prompts"
    values = [kl(p, q) for p, q in zip(a["dists"], b["dists"])]
    same_top = sum(
        max(p, key=p.get) == max(q, key=q.get) for p, q in zip(a["dists"], b["dists"])
    )
    values.sort()
    n = len(values)
    print(f"{b['model']} vs {a['model']}, {n} harmless prompts")
    print(f"  first-token KL: mean {sum(values) / n:.4f}  median {values[n // 2]:.4f}  max {values[-1]:.4f}")
    print(f"  same most-likely first token: {same_top}/{n}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--api-base", default="http://127.0.0.1:8001/v1")
    d.add_argument("--model", required=True)
    d.add_argument("--prompts", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--system-prompt", default="You are a helpful assistant.")
    d.add_argument("--token-ids", action="store_true", help="key tokens by id (token_id:<n>), not text")
    c = sub.add_parser("compare")
    c.add_argument("reference")
    c.add_argument("other")
    args = ap.parse_args()
    dump(args) if args.cmd == "dump" else compare(args)


if __name__ == "__main__":
    main()
