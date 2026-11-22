#!/usr/bin/env python3
"""Per-layer geometry of a refusal-direction file, to choose where to edit.

Reads a residuals file (good_means, bad_means, directions; see lab/directions.py)
and reports, per layer: the raw separation ||bad - good||, its size relative to
the residual scale, and the cosine between the harmful and harmless means (low
cosine = the two are most separated there). With --model/--arch each layer is
labelled by block type, so for Nemotron you can see which MoE layers (the
fc2_latent_proj writer) carry the strongest refusal signal.

It then suggests a contiguous layer band: the run of layers whose relative
separation is at least --frac of the maximum. This is the knee idea without the
Kneedle dependency -- a threshold on the normalized per-layer strength.

    python3 geometry.py directions.safetensors [--arch nemotron_h --model DIR] [--frac 0.5]
"""
import argparse
import os
import sys

import torch
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arch as arch_registry  # noqa: E402


def block_types(arch_name, model_dir):
    """{layer index: 'mamba'|'attn'|'moe'} from the checkpoint, or {} if unavailable."""
    from check_formats import Checkpoint
    model = arch_registry.get(arch_name)
    bf16 = Checkpoint(model_dir)
    types = {}
    for layer in range(model.num_layers(model_dir)):
        comps = [c for c, *_ in model.targets(layer, bf16)]
        types[layer] = "moe" if "moe_latent" in comps else ("attn" if "attn" in comps else "mamba")
    return types


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("directions")
    ap.add_argument("--arch")
    ap.add_argument("--model", help="checkpoint dir, to label layers by block type")
    ap.add_argument("--frac", type=float, default=0.5, help="band threshold as a fraction of peak separation")
    ap.add_argument("--min-copy-cos", type=float, default=0.9,
                    help="DeepSeek-V4: lowest cosine between a copy's direction and the mean's that passes")
    args = ap.parse_args()

    t = load_file(args.directions)
    good, bad = t["good_means"], t["bad_means"]        # [L+1, H], index 0 = embeddings
    n = good.shape[0] - 1
    diff = (bad - good)[1:]                              # per layer, drop embedding row
    dnorm = diff.norm(dim=1)
    rel = dnorm / good[1:].norm(dim=1)
    cos = (torch.nn.functional.normalize(good[1:], dim=1) * torch.nn.functional.normalize(bad[1:], dim=1)).sum(1)

    types = {}
    if args.arch and args.model:
        types = block_types(args.arch, args.model)

    peak = rel.max().item()
    strong = (rel >= args.frac * peak)

    print(f"{'layer':>5} {'type':<5} {'||bad-good||':>12} {'rel':>7} {'cos(g,b)':>9}  band")
    for i in range(n):
        mark = "  <<" if strong[i] else ""
        print(f"{i:>5} {types.get(i,''):<5} {dnorm[i]:>12.3f} {rel[i]:>7.3f} {cos[i]:>9.4f}{mark}")

    idx = [i for i in range(n) if strong[i]]
    if idx:
        lo, hi = min(idx), max(idx)
        moe_in_band = [i for i in idx if types.get(i) == "moe"]
        print(f"\nsuggested band (rel >= {args.frac:.2f} x peak): {lo}-{hi}  "
              f"({len(idx)} layers; {len(moe_in_band)} of them MoE)")
        print(f"peak rel {peak:.3f} at layer {int(rel.argmax())}  "
              f"({types.get(int(rel.argmax()), '')})")

    if "copy_directions" in t and idx:
        hyper_connection_check(t, lo, hi, args.min_copy_cos)


def hyper_connection_check(t, lo, hi, threshold):
    """DeepSeek-V4: do the 4 residual copies carry the same refusal direction?

    The residual is 4 copies of the hidden state, and every writer's output is
    added to all of them. The edit removes one direction (measured on the mean
    of the copies) from what the writers write, which removes it from every copy
    alike. That is the right edit only if each copy's own direction is close to
    the mean's; if they differ, removing one direction leaves part of the
    refusal in the copies that point elsewhere.
    """
    d = torch.nn.functional.normalize(t["directions"][1:], dim=-1)               # [L, H]
    c = torch.nn.functional.normalize(t["copy_directions"][1:], dim=-1)         # [L, 4, H]
    cos = (c * d.unsqueeze(1)).sum(-1)                                           # [L, 4]
    print("\nhyper-connection copies: cosine of each copy's direction with the mean's")
    print(f"{'layer':>5}  " + "  ".join(f"copy{k}" for k in range(cos.shape[1])))
    for i in range(lo, hi + 1):
        print(f"{i:>5}  " + "  ".join(f"{x:>5.3f}" for x in cos[i].tolist()))
    worst = cos[lo: hi + 1].min().item()
    print(f"lowest in the band: {worst:.3f} (threshold {threshold})")
    if worst < threshold:
        print("COPIES DISAGREE: one direction removed from all four copies is not the measured refusal")


if __name__ == "__main__":
    main()
