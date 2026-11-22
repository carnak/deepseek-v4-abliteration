#!/usr/bin/env python3
"""One table across experiments: what each configuration did, and what it cost.

    python3 results.py [--stock kl/stock.json] [experiments dir] [name ...]

Per experiment, from the files run_queue.sh leaves behind (verdicts as percentages of n):
  comply / partial / deflect / refuse   the judge's verdicts, thinking off and on
  un                                     no answer: reasoning used up its tokens
  KL                                     first-token drift from the stock model
The ranking is by comply with thinking on, then thinking off: that is the
behaviour being bought, and thinking on is the harder case.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from first_token_kl import kl  # noqa: E402

ORDER = ("COMPLY", "PARTIAL", "DEFLECT", "REFUSE")


def label(row):
    """A row's verdict, with empty responses pulled out as their own category.

    An empty response with thinking on is reasoning that ran out of tokens.
    Files judged before judge.py knew that carry REFUSE (or whatever the judge
    made of nothing) for those rows; reading them here corrects old results
    without re-judging.
    """
    return "UNFINISHED" if not row["response"].strip() else row["verdict"]


def verdicts(path):
    if not os.path.exists(path):
        return None
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    if not rows or "verdict" not in rows[0]:
        return None
    # Percentages, with the count last: runs on the 100-prompt screening set and
    # on the 300-prompt set have to be readable side by side.
    n = len(rows)
    return [round(100 * sum(label(r) == v for r in rows) / n) for v in (*ORDER, "UNFINISHED")] + [n]


def drift(path, stock):
    if not (stock and os.path.exists(path)):
        return None
    other = json.load(open(path))
    values = [kl(p, q) for p, q in zip(stock["dists"], other["dists"])]
    return sum(values) / len(values)


def cell(v):
    if not v:
        return "   (not judged)"
    return "  ".join(f"{x:>2}" for x in v[:5]) + f"  n={v[5]}"


def first_number(path):
    """The 'N/200 correct' figure from a gsm8k_eval.py output file."""
    if not os.path.exists(path):
        return "-"
    for token in open(path).read().split():
        if "/" in token and token.split("/")[0].isdigit():
            return token.split("/")[0]
    return "-"


def validation_table(root, names, stock):
    """Finalists: the independent prompt set, benign prompts, and capability."""
    rows = []
    for name in names or sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if not os.path.exists(os.path.join(d, "val-nothink.jsonl")):
            continue
        benign = verdicts(os.path.join(d, "benign.jsonl"))
        rows.append((
            name,
            verdicts(os.path.join(d, "val-nothink.jsonl")),
            verdicts(os.path.join(d, "val-thinking.jsonl")),
            f"{benign[0]:>3}" if benign else "  -",
            first_number(os.path.join(d, "gsm8k-nothink.txt")),
            first_number(os.path.join(d, "gsm8k-thinking.txt")),
            first_number(os.path.join(d, "mc.txt")),
            drift(os.path.join(d, "first-token.json"), stock),
        ))
    rows.sort(key=lambda r: (-(r[2][0] if r[2] else -1), -(r[1][0] if r[1] else -1)))
    width = max([len(r[0]) for r in rows] + [10])
    print(f"{'experiment':<{width}}  {'held-out, off':<25}  {'held-out, on':<25}  benign  gsm8k off/on  mmlu  {'KL':>6}")
    print(f"{'':<{width}}  {'co pa de re un  (%)':<25}  {'co pa de re un  (%)':<25}  comply%  (of 200)     /500")
    for name, off, on, benign, g_off, g_on, mc, k in rows:
        ks = f"{k:.3f}" if k is not None else "     -"
        print(f"{name:<{width}}  {cell(off):<25}  {cell(on):<25}  {benign:>6}  {g_off:>5} / {g_on:<4}  {mc:>4}  {ks:>6}")


SCENARIOS = ("system-cautious", "multi-turn", "long-context", "tools")


def robust_table(root, names):
    """Comply % per scenario: harmful prompts with thinking on, benign with it off.

    The plain column is the same 100 prompts taken from the held-out run, so a
    scenario's effect reads straight across the row. "82~9" is 82% complied and
    9 prompts got no answer because reasoning ran out of tokens.
    """
    def comply(path, prompts=None):
        if not os.path.exists(path):
            return "  -"
        rows = [json.loads(line) for line in open(path, encoding="utf-8")]
        if prompts is not None:
            rows = [r for r in rows if r["prompt"] in prompts]
        if not rows or "verdict" not in rows[0]:
            return "  ?"
        done = [r for r in rows if label(r) != "UNFINISHED"]
        # Comply %, then how many of the prompts never got an answer at all.
        cut = len(rows) - len(done)
        return f"{round(100 * sum(r['verdict'] == 'COMPLY' for r in done) / len(rows)):>3}" + (f"~{cut}" if cut else "")

    def tool_calls(path):
        if not os.path.exists(path):
            return "-"
        rows = [json.loads(line) for line in open(path, encoding="utf-8")]
        return str(sum(bool(r.get("tool_call")) for r in rows))

    rows = []
    for name in names or sorted(os.listdir(root)):
        d = os.path.join(root, name)
        first = os.path.join(d, f"rob-{SCENARIOS[0]}.jsonl")
        if not os.path.exists(first):
            continue
        subset = {json.loads(line)["prompt"] for line in open(first, encoding="utf-8")}
        rows.append((
            name,
            comply(os.path.join(d, "val-thinking.jsonl"), subset),
            [comply(os.path.join(d, f"rob-{s}.jsonl")) for s in SCENARIOS],
            [comply(os.path.join(d, f"robben-{s}.jsonl")) for s in SCENARIOS],
            tool_calls(os.path.join(d, "rob-tools.jsonl")),
            tool_calls(os.path.join(d, "robben-tools.jsonl")),
        ))
    width = max([len(r[0]) for r in rows] + [10])
    short = ["system", "turn-3", "long", "tools"]
    head = "  ".join(f"{s:>6}" for s in short)
    print(f"{'experiment':<{width}}  harmful, thinking on, comply %         benign, thinking off, comply %   tool calls")
    print(f"{'':<{width}}  {'plain':>6}  {head}    {head}   harmful/benign")
    for name, plain, harm, ben, tc_h, tc_b in rows:
        print(f"{name:<{width}}  {plain:>6}  " + "  ".join(f"{x:>6}" for x in harm)
              + "    " + "  ".join(f"{x:>6}" for x in ben) + f"   {tc_h}/{tc_b}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--robust", action="store_true",
                    help="show the robustness tier (VALIDATE=1 runs) instead")
    ap.add_argument("--validation", action="store_true",
                    help="show the finalists' table (VALIDATE=1 runs) instead")
    ap.add_argument("--stock", default="kl/stock.json")
    ap.add_argument("root", nargs="?", default="experiments")
    ap.add_argument("names", nargs="*")
    args = ap.parse_args()

    stock = json.load(open(args.stock)) if os.path.exists(args.stock) else None
    if args.validation:
        validation_table(args.root, args.names, stock)
        return
    if args.robust:
        robust_table(args.root, args.names)
        return
    rows = []
    for name in args.names or sorted(os.listdir(args.root)):
        d = os.path.join(args.root, name)
        if not os.path.isdir(d):
            continue
        spec = ""
        if os.path.exists(os.path.join(d, "ablate-args.txt")):
            spec = open(os.path.join(d, "ablate-args.txt")).read().strip()
        rows.append((name, verdicts(os.path.join(d, "nothink.jsonl")),
                     verdicts(os.path.join(d, "thinking.jsonl")),
                     drift(os.path.join(d, "first-token.json"), stock), spec))

    rows.sort(key=lambda r: (-(r[2][0] if r[2] else -1), -(r[1][0] if r[1] else -1)))
    width = max([len(r[0]) for r in rows] + [10])
    print(f"{'experiment':<{width}}  {'thinking off':<25}  {'thinking on':<25}  {'KL':>6}  arguments")
    print(f"{'':<{width}}  {'co pa de re un  (%)':<25}  {'co pa de re un  (%)':<25}")
    for name, off, on, k, spec in rows:
        ks = f"{k:.3f}" if k is not None else "     -"
        print(f"{name:<{width}}  {cell(off):<25}  {cell(on):<25}  {ks:>6}  {spec}")


if __name__ == "__main__":
    main()
