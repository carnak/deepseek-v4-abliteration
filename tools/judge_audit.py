#!/usr/bin/env python3
"""How far to trust the judge.

Every number this project steers by is a verdict from judge.py. It was checked
once, on the stock model's refusals. Whether it draws the COMPLY / DEFLECT line
correctly on an abliterated model's output -- the only place that line matters
-- was assumed. This measures it three ways, cheapest first:

  retest     Judge the same responses again. Disagreement with itself is noise
             that no rubric can explain away.
  second     Judge them with a differently worded, two-way rubric
             (judge.py --rubric binary). Agreement says the verdict belongs to
             the response, not to the wording of the question.
  human      A person labels a blind sample. The only one of the three that
             measures whether the judge is RIGHT rather than merely consistent.

    # the first two: add the extra labels, then compare
    python3 judge.py FILE --judge-model M --judge-base URL --field verdict_retest
    python3 judge.py FILE --judge-model M --judge-base URL --rubric binary --field verdict_binary
    python3 judge_audit.py agreement FILE [FILE ...]

    # the third
    python3 judge_audit.py sheet FILE [FILE ...] --per 12 --out audit/
    #   ... fill in "human" in audit/sheet.jsonl: COMPLY, PARTIAL, DEFLECT or REFUSE ...
    python3 judge_audit.py score audit/

This tool prints counts and rates only, never a response.
"""
import argparse
import json
import os
import random
import sys

FOUR = ("COMPLY", "PARTIAL", "DEFLECT", "REFUSE")


def load(paths):
    rows = []
    for p in paths:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            r["_file"] = p
            rows.append(r)
    return rows


def kappa(a, b):
    """Cohen's kappa for two lists of labels: agreement beyond what chance gives."""
    n = len(a)
    labels = sorted(set(a) | set(b))
    observed = sum(x == y for x, y in zip(a, b)) / n
    expected = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def complied(verdict):
    return verdict == "COMPLY"


def agreement(args):
    rows = load(args.files)
    print(f"{len(rows)} judged responses from {len(args.files)} file(s)")

    retest = [r for r in rows if "verdict_retest" in r]
    if retest:
        a = [r["verdict"] for r in retest]
        b = [r["verdict_retest"] for r in retest]
        same = sum(x == y for x, y in zip(a, b))
        same2 = sum(complied(x) == complied(y) for x, y in zip(a, b))
        print(f"\nretest (same rubric, judged twice), n={len(retest)}")
        print(f"  same label of four:        {same}/{len(retest)} = {100 * same / len(retest):.1f}%   kappa {kappa(a, b):.2f}")
        print(f"  same comply / not-comply:  {same2}/{len(retest)} = {100 * same2 / len(retest):.1f}%")
        flips = {}
        for x, y in zip(a, b):
            if x != y:
                key = " <-> ".join(sorted((x, y)))
                flips[key] = flips.get(key, 0) + 1
        if flips:
            print("  disagreements: " + ", ".join(f"{k} x{v}" for k, v in sorted(flips.items(), key=lambda kv: -kv[1])))

    second = [r for r in rows if r.get("verdict_binary") in ("FULFILLED", "NOT_FULFILLED")]
    if second:
        a = [complied(r["verdict"]) for r in second]
        b = [r["verdict_binary"] == "FULFILLED" for r in second]
        both = sum(x and y for x, y in zip(a, b))
        neither = sum(not x and not y for x, y in zip(a, b))
        only_primary = sum(x and not y for x, y in zip(a, b))
        only_second = sum(y and not x for x, y in zip(a, b))
        n = len(second)
        print(f"\nsecond rubric (independent wording, two-way), n={n}")
        print(f"  agree: {both + neither}/{n} = {100 * (both + neither) / n:.1f}%   kappa {kappa(a, b):.2f}")
        print(f"  both say complied {both}   both say not {neither}")
        print(f"  primary COMPLY but second NOT_FULFILLED: {only_primary}   (primary too lenient, if the second is right)")
        print(f"  second FULFILLED but primary not COMPLY: {only_second}   (primary too strict, if the second is right)")
        print(f"  comply rate: primary {100 * sum(a) / n:.1f}%, second {100 * sum(b) / n:.1f}%")
        # Where the primary's non-COMPLY labels land under the second rubric.
        for label in ("PARTIAL", "DEFLECT", "REFUSE"):
            group = [r for r in second if r["verdict"] == label]
            if group:
                f = sum(r["verdict_binary"] == "FULFILLED" for r in group)
                print(f"    primary {label:<8} n={len(group):<4} second says fulfilled: {f}")

    if not retest and not second:
        print("no verdict_retest or verdict_binary fields found; run judge.py with --field first")


def sheet(args):
    rows = [r for r in load(args.files) if r.get("verdict") in FOUR]
    rng = random.Random(args.seed)
    picked = []
    for label in FOUR:
        group = [r for r in rows if r["verdict"] == label]
        rng.shuffle(group)
        picked.extend(group[: args.per])
    rng.shuffle(picked)  # so position gives nothing away

    os.makedirs(args.out, exist_ok=True)
    key = {}
    with open(os.path.join(args.out, "sheet.jsonl"), "w", encoding="utf-8") as f:
        for i, r in enumerate(picked):
            key[str(i)] = {"judge": r["verdict"], "file": r["_file"]}
            f.write(json.dumps({"id": i, "prompt": r["prompt"], "response": r["response"], "human": ""},
                               ensure_ascii=False) + "\n")
    json.dump(key, open(os.path.join(args.out, "key.json"), "w"), indent=1)
    counts = {l: sum(k["judge"] == l for k in key.values()) for l in FOUR}
    print(f"{len(picked)} responses -> {args.out}/sheet.jsonl (blind; the judge's labels are in key.json)")
    print("  sampled per judge label:", counts)
    print('  fill in each "human" with COMPLY, PARTIAL, DEFLECT or REFUSE, then: judge_audit.py score', args.out)


def score(args):
    key = json.load(open(os.path.join(args.dir, "key.json")))
    rows = [json.loads(line) for line in open(os.path.join(args.dir, "sheet.jsonl"), encoding="utf-8")]
    done = [r for r in rows if r["human"].strip().upper() in FOUR]
    if not done:
        sys.exit("no rows have a human label yet")
    h = [r["human"].strip().upper() for r in done]
    j = [key[str(r["id"])]["judge"] for r in done]
    n = len(done)
    same = sum(x == y for x, y in zip(h, j))
    same2 = sum(complied(x) == complied(y) for x, y in zip(h, j))
    print(f"{n} of {len(rows)} labelled")
    print(f"  same label of four:        {same}/{n} = {100 * same / n:.1f}%   kappa {kappa(h, j):.2f}")
    print(f"  same comply / not-comply:  {same2}/{n} = {100 * same2 / n:.1f}%")
    lenient = sum(complied(y) and not complied(x) for x, y in zip(h, j))
    strict = sum(complied(x) and not complied(y) for x, y in zip(h, j))
    print(f"  judge said COMPLY, human did not: {lenient}   human said COMPLY, judge did not: {strict}")
    # The sample is stratified by the judge's label, so these are per-label
    # precision figures, not an overall accuracy.
    for label in FOUR:
        idx = [i for i, y in enumerate(j) if y == label]
        if idx:
            ok = sum(h[i] == label for i in idx)
            print(f"    judge {label:<8} n={len(idx):<3} human agrees: {ok}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("agreement")
    a.add_argument("files", nargs="+")
    s = sub.add_parser("sheet")
    s.add_argument("files", nargs="+")
    s.add_argument("--per", type=int, default=12, help="responses per judge label")
    s.add_argument("--out", required=True)
    s.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("score")
    c.add_argument("dir")
    args = ap.parse_args()
    {"agreement": agreement, "sheet": sheet, "score": score}[args.cmd](args)


if __name__ == "__main__":
    main()
