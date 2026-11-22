#!/usr/bin/env python3
"""Choose among experiments by rule, so that choosing is repeatable.

Two decisions, the same shape. Among the experiments that do the job about as
well as the best one does, take the one that changed the model least:

    choose.py screen <root>   which experiments go on to validation
    choose.py winner <root>   which validated experiment becomes the kept build

"Does the job" is comply %, taking the worse of thinking off and thinking on:
a build is only as good as its weaker mode. "About as well" is within
--tolerance points of the best, because two runs of one build differ by a few
points and a ranking finer than that is noise. "Changed least" is first-token
KL from the stock model.

For the winner there are also things a build must not have lost, measured on
the validation run: knowledge (MMLU against the control's), ordinary requests
still answered, and reasoning that still ends.

Prints the table with the decision marked, then one line per choice:
    FINALIST <name>      (screen)
    WINNER <name>        (winner)
Exit status 1 if nothing qualifies.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json  # noqa: E402

from results import drift, first_number, verdicts  # noqa: E402

CONTROL = "control"


def speed(path):
    """tok/s from a decode_speed.py output file, or None."""
    try:
        return float(open(path).read().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def load(root, stock, validation):
    off, on = ("val-nothink.jsonl", "val-thinking.jsonl") if validation else ("nothink.jsonl", "thinking.jsonl")
    rows = []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        v_off, v_on = verdicts(os.path.join(d, off)), verdicts(os.path.join(d, on))
        if not (v_off and v_on):
            continue
        benign = verdicts(os.path.join(d, "benign.jsonl"))
        mmlu = first_number(os.path.join(d, "mc.txt"))
        rows.append({
            "name": name,
            "off": v_off[0], "on": v_on[0], "score": min(v_off[0], v_on[0]),
            "unfinished": v_on[4],
            "kl": drift(os.path.join(d, "first-token.json"), stock),
            "benign": benign[0] if benign else None,
            "mmlu": int(mmlu) if mmlu.isdigit() else None,
            "speed": speed(os.path.join(d, "speed.txt")),
            "args": open(os.path.join(d, "ablate-args.txt")).read().strip(),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("screen", "winner"))
    ap.add_argument("root")
    ap.add_argument("--stock", default="kl/stock.json")
    ap.add_argument("--tolerance", type=float, default=3.0,
                    help="points of comply %% within which two builds count as equally good")
    ap.add_argument("--keep", type=int, default=2, help="screen: how many finalists")
    ap.add_argument("--mmlu-drop", type=int, default=5,
                    help="winner: most MMLU questions (of those asked) a build may lose against the control")
    ap.add_argument("--min-benign", type=float, default=95.0, help="winner: benign requests answered, %%")
    ap.add_argument("--max-unfinished", type=float, default=2.0,
                    help="winner: most thinking-on prompts (%%) whose reasoning never ended")
    ap.add_argument("--min-speed", type=float, default=0.0,
                    help="winner: least single-stream decode speed, as a fraction of the control's "
                    "(speed.txt, tools/decode_speed.py); 0 turns the check off")
    ap.add_argument("--write-queue", help="screen: write the finalists (and the control) as a queue file")
    args = ap.parse_args()

    stock = json.load(open(args.stock)) if os.path.exists(args.stock) else None
    rows = load(args.root, stock, validation=args.mode == "winner")
    control = next((r for r in rows if r["name"] == CONTROL), None)
    candidates = [r for r in rows if r["name"] != CONTROL and r["kl"] is not None]
    if not candidates:
        sys.exit(f"no judged experiments with a drift figure under {args.root}")

    for r in candidates:
        r["why_not"] = []
        if args.mode == "winner":
            if control and control["mmlu"] is not None and r["mmlu"] is not None \
                    and r["mmlu"] < control["mmlu"] - args.mmlu_drop:
                r["why_not"].append(f"MMLU {r['mmlu']} vs control {control['mmlu']}")
            if r["benign"] is not None and r["benign"] < args.min_benign:
                r["why_not"].append(f"benign {r['benign']}%")
            if r["unfinished"] > args.max_unfinished:
                r["why_not"].append(f"reasoning unfinished {r['unfinished']}%")
            if args.min_speed and control and control["speed"] and r["speed"] is not None \
                    and r["speed"] < args.min_speed * control["speed"]:
                r["why_not"].append(f"decode {r['speed']:.1f} tok/s vs control {control['speed']:.1f}")

    sound = [r for r in candidates if not r["why_not"]]
    best = max((r["score"] for r in sound), default=None)
    for r in candidates:
        if not r["why_not"] and r["score"] < best - args.tolerance:
            r["why_not"].append(f"{best - r['score']:.0f} below the best")
    eligible = sorted((r for r in candidates if not r["why_not"]), key=lambda r: (r["kl"], -r["score"]))
    chosen = eligible[: args.keep if args.mode == "screen" else 1]

    width = max(len(r["name"]) for r in rows)
    print(f"{'experiment':<{width}}  off   on  score  unfin    KL   mmlu  benign  tok/s")
    for r in sorted(rows, key=lambda r: (r["kl"] is None, r["kl"])):
        mark = "<= chosen" if r in chosen else ("   (control)" if r is control else
                                                 "   " + "; ".join(r.get("why_not", [])) if r.get("why_not") else "   qualifies")
        kl = f"{r['kl']:.3f}" if r["kl"] is not None else "    -"
        mmlu = r["mmlu"] if r["mmlu"] is not None else "-"
        benign = r["benign"] if r["benign"] is not None else "-"
        tps = f"{r['speed']:.1f}" if r["speed"] is not None else "-"
        print(f"{r['name']:<{width}}  {r['off']:>3}  {r['on']:>3}  {r['score']:>5}  {r['unfinished']:>5}  {kl}  "
              f"{mmlu:>5}  {benign:>6}  {tps:>5}  {mark}")
    print(f"\nbest sound score {best}; within {args.tolerance:g} points of it counts as equal; least drift wins")

    if not chosen:
        print("nothing qualifies")
        return 1
    for r in chosen:
        print(("FINALIST " if args.mode == "screen" else "WINNER ") + r["name"])
    if args.write_queue:
        with open(args.write_queue, "w") as f:
            f.write("# Written by choose.py: the screening run's finalists, and the control they are compared with.\n")
            if control:
                f.write(f"{control['name']} | {control['args']}\n")
            for r in chosen:
                f.write(f"{r['name']} | {r['args']}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
