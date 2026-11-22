#!/usr/bin/env python3
"""Assemble the prompt sets, from public benchmarks, with the three roles kept apart.

A prompt may play exactly one role:

  measure   what refusal directions are computed from
  tune      what experiments are ranked on while choosing a configuration
  held-out  what finalists are graded on; never looked at while choosing

Independence is by source, not just by row: each benchmark feeds one role, so a
configuration cannot look good on held-out prompts by having been tuned on its
neighbours from the same list. Exact duplicates across roles (several of these
benchmarks descend from the same older lists) are then removed, giving priority
measure > tune > held-out, so removing them only ever shrinks the later role.

    python3 build_prompt_sets.py <out dir>

Needs `datasets` and network access; run it in the heretic image. Deterministic:
no sampling, only fixed strides, so a rebuild gives the same files.

Writes one prompt per line:
  tuning300.txt    the first 100 lines are the original tuning set
                   (mlabonne/harmful_behaviors test[:100]), so results from
                   before this file existed stay comparable on that prefix.
  heldout300.txt   the first 100 lines are JailbreakBench's harmful behaviours,
                   likewise the original held-out set.
  manifest.json    what came from where, and what was dropped as a duplicate.
"""
import json
import os
import sys

from datasets import load_dataset


def norm(text):
    return " ".join(text.lower().split()).rstrip(".?! ")


def one_line(text):
    return " ".join(text.split())


def stride(items, n):
    """n items spread evenly through the list, in order."""
    if n >= len(items):
        return list(items)
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def per_group(rows, key, text, per):
    """`per` items from each group, groups in first-seen order."""
    groups = {}
    for r in rows:
        groups.setdefault(r[key], []).append(r[text])
    out = []
    for items in groups.values():
        out.extend(stride(items, per))
    return out, len(groups)


def main():
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    manifest = {"roles": {}, "dropped_duplicates": {}}

    # --- measure: unchanged from every direction exported so far ---------------
    measure = list(load_dataset("mlabonne/harmful_behaviors", split="train[:400]")["text"])
    seen = {norm(t) for t in measure}
    manifest["roles"]["measure"] = {"mlabonne/harmful_behaviors train[:400]": len(measure)}

    def take(role, parts, limit):
        kept, dropped, sources = [], 0, {}
        for label, items in parts:
            n = 0
            for t in items:
                t = one_line(t)
                if norm(t) in seen:
                    dropped += 1
                    continue
                if len(kept) >= limit:
                    break
                seen.add(norm(t))
                kept.append(t)
                n += 1
            sources[label] = n
        manifest["roles"][role] = sources
        manifest["dropped_duplicates"][role] = dropped
        return kept

    # --- tune ------------------------------------------------------------------
    legacy_tune = list(load_dataset("mlabonne/harmful_behaviors", split="test[:100]")["text"])
    cat = load_dataset("declare-lab/CategoricalHarmfulQA", split="en")
    cat_items, n_sub = per_group(cat, "Subcategory", "Question", 2)
    mal = list(load_dataset("walledai/MaliciousInstruct", split="train")["prompt"])
    tune = take("tune", [
        ("mlabonne/harmful_behaviors test[:100]", legacy_tune),
        (f"declare-lab/CategoricalHarmfulQA en, 2 per subcategory ({n_sub} subcategories)", cat_items),
        ("walledai/MaliciousInstruct", mal),
    ], 300)

    # --- held-out --------------------------------------------------------------
    jbb = list(load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")["Goal"])
    fq = load_dataset("TrustAIRLab/forbidden_question_set", split="train")
    fq_items, n_pol = per_group(fq, "content_policy_name", "question", 10)
    tdc = list(load_dataset("walledai/TDC23-RedTeaming", split="train")["prompt"])
    held = take("held-out", [
        ("JailbreakBench/JBB-Behaviors harmful", jbb),
        (f"TrustAIRLab/forbidden_question_set, 10 per policy ({n_pol} policies)", fq_items),
        ("walledai/TDC23-RedTeaming", tdc),
    ], 300)

    # The measurement prompts as files too, so the lab reads them without needing
    # `datasets`. The harmless half is what heretic's defaults use.
    harmless = list(load_dataset("mlabonne/harmless_alpaca", split="train[:400]")["text"])
    manifest["roles"]["measure"]["mlabonne/harmless_alpaca train[:400] (harmless)"] = len(harmless)

    for name, rows in (
        ("tuning300.txt", tune),
        ("tuning100.txt", tune[:100]),
        ("heldout300.txt", held),
        ("heldout100.txt", held[::3]),
        ("measure_harmful.txt", [one_line(t) for t in measure]),
        ("measure_harmless.txt", [one_line(t) for t in harmless]),
    ):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")
        manifest[name] = len(rows)

    # The promise the docstring makes, checked rather than assumed.
    assert tune[:100] == [one_line(t) for t in legacy_tune], "tuning prefix is not the original set"
    overlap = {norm(t) for t in tune} & {norm(t) for t in held}
    assert not overlap, f"{len(overlap)} prompts are in both tune and held-out"
    assert not ({norm(t) for t in measure} & ({norm(t) for t in tune} | {norm(t) for t in held}))
    manifest["legacy_heldout_prefix_intact"] = held[:100] == [one_line(t) for t in jbb]

    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
