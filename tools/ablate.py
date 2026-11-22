#!/usr/bin/env python3
"""Apply refusal directions straight to a quantized checkpoint, without heretic's search.

Heretic searches a constrained space: one direction per layer, a triangular
weight kernel over at most the upper ~60% of layers, weights capped at 1.5.
This applies directional ablation with whatever layers, weights and number of
directions are asked for, so a hypothesis can be built and served in minutes
instead of searched for overnight. On Qwen3.6-35B-A3B the configuration that
worked (layers 12-39 at full strength) is one heretic's kernel cannot express.

For every layer in range, every matrix that writes to the residual stream --
the attention output projection, the shared expert's down projection and each
routed expert's down projection -- becomes

    W' = W - weight * sum_d  v_d (v_d^T W)

over that layer's directions v_d, orthonormalized first so that several
directions are each removed exactly once. With weight 1 the layer can no
longer write along those directions at all; above 1 it writes their opposite.

Directions come from `heretic --export-residuals`. Each --directions file
contributes one direction per layer. By default layer i uses its own direction;
--global-layer N uses layer N's direction for every layer instead.

    python3 ablate.py --arch qwen3_5_moe --bf16 <dir> --quantized <dir> --out <dir> \\
        --directions nothink.safetensors --layers 12-39

Only the edited matrices are re-encoded (see quant_formats.py); every other
tensor of the quantized checkpoint is copied unchanged.
"""
import argparse
import os
import sys

import torch
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arch as arch_registry  # noqa: E402
from check_formats import Checkpoint  # noqa: E402
from quant_formats import (dsv4_decode, dsv4_encode, dsv4_encode_steered, dsv4_kind,  # noqa: E402
                           fp8_encode, nvfp4_encode)
from transplant import write_checkpoint  # noqa: E402


def layer_basis(direction_sets, index, scales):
    """Orthonormal basis [k, hidden] of the directions at one index, and each vector's scale.

    QR keeps the order: the first basis vector is the first direction, the
    second is what the second direction adds beyond the first, and so on. A
    scale therefore applies to what that --directions file contributes that the
    ones before it did not.
    """
    vs = torch.stack([d[index] for d in direction_sets]).to(torch.float64)
    q, r = torch.linalg.qr(vs.T)
    # A direction that is (nearly) a combination of the others adds nothing and
    # would only inject noise if normalized up to unit length.
    keep = r.diagonal().abs() > 1e-6
    return q.T[keep].to(torch.float32), torch.tensor(scales, dtype=torch.float32)[keep]


def ablate(w, basis, scales, weight):
    """w: [..., out, in]. Removes weight * scale_k of the component along basis vector k."""
    if weight == 0 or basis.numel() == 0:
        return w
    # sum_k s_k b_k b_k^T is the (scaled) projector onto the span, applied on the output side.
    return w - weight * torch.einsum("k,kh,kg,...gi->...hi", scales, basis, basis, w)


def encode(w, name, quant, replace):
    """Re-encode one [out, in] matrix in whatever format the checkpoint uses for it."""
    if name + ".weight_scale_2" in quant:
        q, s, s2 = nvfp4_encode(w)
        replace[name + ".weight"], replace[name + ".weight_scale"] = q, s
        replace[name + ".weight_scale_2"] = s2
    elif name + ".weight_scale" in quant:
        q, s = fp8_encode(w)
        replace[name + ".weight"], replace[name + ".weight_scale"] = q, s
    else:
        replace[name + ".weight"] = w.to(torch.bfloat16)


def survival(w0, w1, dec, basis, coef, probe):
    """How much of the intended edit survives re-encoding, per direction.

    Re-encoding rounds every element to the storage grid. For FP4 that grid is
    coarse (8 magnitudes per 32-wide group), and an edit is only useful if what
    is stored still lacks the direction. This is arXiv 2611.09382's caveat that
    some weight edits fall below storage resolution, measured rather than assumed.

    For basis vector b_k the edit makes b_k^T W1 = (1 - coef_k) b_k^T W0 exactly;
    what is stored gives b_k^T dec. Returned per direction, relative to the
    original component |b_k^T W0|:
      target  |1 - coef_k|                      what the edit asked for
      got     |b_k^T dec|                       what the stored matrix has
      err     |b_k^T (dec - W1)|                rounding error along b_k
      noise   |u^T (dec - W1)|                  rounding error along a random
                                                unit u orthogonal to the basis
    err near noise means rounding is not working against the edit, only adding
    the noise it adds everywhere; err well above noise means it is.
    """
    base = (basis @ w0).norm(dim=1).clamp(min=1e-30)          # [k]
    got = (basis @ dec).norm(dim=1) / base
    diff = dec - w1
    err = (basis @ diff).norm(dim=1) / base
    noise = (probe @ diff).norm() / base
    target = (1.0 - coef).abs()
    rel = (diff.norm() / w1.norm()).item()
    return {"target": target.tolist(), "got": got.tolist(), "err": err.tolist(),
            "noise": noise.tolist(), "quant_err": rel}


def build_native(args, model, plan, device):
    """Write the edited checkpoint for an arch with no BF16 original (DeepSeek-V4).

    plan: quantized name -> (component, basis [k, hidden], coef [k]), where
    W' = W - sum_k coef_k b_k (b_k^T W). The source of each edited matrix is the
    checkpoint itself, decoded; it is re-encoded in the format it came in.

    Shard by shard rather than collect-then-write: ~11k FP4 expert matrices
    would otherwise be held in memory together (~33 GB on a box whose memory
    the GPU shares). A shard is ~3 GB. Every tensor not in the plan is written
    back exactly as read.
    """
    import glob
    import json
    import shutil
    import statistics
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    os.makedirs(args.out, exist_ok=True)
    for f in sorted(os.listdir(args.quantized)):
        src = os.path.join(args.quantized, f)
        if os.path.isfile(src) and not f.endswith(".safetensors"):
            shutil.copy2(src, os.path.join(args.out, f))
        elif os.path.isdir(src) and f not in (".cache",):
            shutil.copytree(src, os.path.join(args.out, f), dirs_exist_ok=True)

    hidden = next(iter(plan.values()))[1].shape[1]
    gen = torch.Generator().manual_seed(0)
    stats = {}                                         # component -> [survival dict]
    done = 0
    for shard in sorted(glob.glob(os.path.join(args.quantized, "*.safetensors"))):
        with safe_open(shard, framework="pt") as f:
            meta = f.metadata()
        tensors = load_file(shard)
        hits = [k[: -len(".weight")] for k in tensors if k.endswith(".weight") and k[: -len(".weight")] in plan]
        for name in hits:
            component, basis, coef = plan[name]
            wkey, skey = name + ".weight", name + ".scale"
            if skey not in tensors:
                sys.exit(f"{skey} is not in the same shard as {wkey}; this build assumes it is")
            kind = dsv4_kind(tensors[wkey].dtype)
            b = basis.to(device)
            c = coef.to(device)
            w0 = dsv4_decode(tensors[wkey].to(device), tensors[skey].to(device))
            w1 = w0 - torch.einsum("k,kh,kg,gi->hi", c, b, b, w0)
            q, s = dsv4_encode_steered(w1, kind, b) if args.steer else dsv4_encode(w1, kind)
            dec = dsv4_decode(q, s)
            # A fixed random direction outside the edit, for the noise floor.
            u = torch.randn(hidden, generator=gen)
            u = u - basis.T @ (basis @ u)
            u = (u / u.norm()).to(device)
            stats.setdefault(component, []).append(survival(w0, w1, dec, b, c, u))
            for key, new in ((wkey, q), (skey, s)):
                old = tensors[key]
                assert old.dtype == new.dtype and old.shape == new.shape, (
                    f"{key}: {new.dtype}{list(new.shape)} would replace {old.dtype}{list(old.shape)}")
                tensors[key] = new.cpu()
        done += len(hits)
        save_file(tensors, os.path.join(args.out, os.path.basename(shard)), metadata=meta)
        print(f"{os.path.basename(shard)}: {len(tensors)} tensors, {len(hits)} matrices edited", flush=True)
        del tensors

    if done != len(plan):
        sys.exit(f"edited {done} matrices but planned {len(plan)} -- output is incomplete")

    # Edit survival, per component: medians over its matrices (first direction).
    print("\nedit survival (relative to each matrix's original component along the direction):")
    print(f"  {'component':<10} {'n':>6} {'target':>7} {'got':>7} {'err':>7} {'noise':>7} {'quant err':>9}")
    summary, worst = {}, 0.0
    for component, rows in stats.items():
        med = {k: statistics.median(r[k][0] if isinstance(r[k], list) else r[k] for r in rows)
               for k in ("target", "got", "err", "noise", "quant_err")}
        miss = statistics.median(abs(r["got"][0] - r["target"][0]) for r in rows)
        worst = max(worst, miss)
        summary[component] = {**med, "n": len(rows), "median_miss": miss}
        print(f"  {component:<10} {len(rows):>6} {med['target']:>7.3f} {med['got']:>7.3f} "
              f"{med['err']:>7.3f} {med['noise']:>7.3f} {med['quant_err']:>9.4f}   miss {miss:.3f}")
    if args.survival_out:
        json.dump(summary, open(args.survival_out, "w"), indent=1)
    if worst > args.survival_tol:
        sys.exit(f"edit survival: a component's median |got - target| is {worst:.3f}, above "
                 f"--survival-tol {args.survival_tol}; the stored weights do not carry the edit")
    return done


def main():
    # The set of residual-writer components is the arch's, so --arch is read
    # first and one --<component> flag is added per entry below.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--arch", default="qwen3_5_moe")
    model = arch_registry.get(pre.parse_known_args()[0].arch)
    components = model.COMPONENTS

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arch", default="qwen3_5_moe")
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--quantized", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--directions", action="append", required=True,
                    help="residuals file from heretic --export-residuals; repeatable")
    ap.add_argument("--scales", default=None,
                    help="comma-separated multiplier per --directions file, in order, applied on top of "
                    "the component weights; e.g. 1,0.5 removes the second direction at half strength")
    ap.add_argument("--place", action="append", default=None, metavar="A-B:" + ",".join(c.upper() for c in components),
                    help="where one direction goes: its layer range and one weight per component, in "
                    f"the order {','.join(components)}. Give one per --directions, in the same order, "
                    "e.g. --place 12-39:1.75,1,1 --place 20-39:1.3,0,0. Replaces "
                    "--layers, the --<component> flags, --weight and --scales")
    ap.add_argument("--weights-file", default=None,
                    help="JSON from `lab.cli learn`: a weight per layer, component and direction. "
                    "Replaces --layers, --attn/--shared/--experts, --weight and --scales")
    ap.add_argument("--layers", default="all", help="inclusive range, e.g. 12-39; default all")
    ap.add_argument("--global-layer", type=int, default=None,
                    help="use this layer's direction for every layer, instead of each layer's own")
    ap.add_argument("--weight", type=float, default=None, help="shorthand: one weight for all components")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="native-quant archs: where matrices are decoded, edited and re-encoded")
    ap.add_argument("--survival-tol", type=float, default=0.05,
                    help="native-quant archs: fail if a component's median |got - target| exceeds this")
    ap.add_argument("--survival-out", default=None, help="native-quant archs: write the survival summary here")
    ap.add_argument("--no-steer", dest="steer", action="store_false",
                    help="native-quant archs: round to nearest instead of direction-aware rounding "
                    "(quant_formats.dsv4_encode_steered); only to measure what steering buys")
    for c in components:
        # e.g. --attn / --moe-latent; stored under the component's own name.
        ap.add_argument(f"--{c.replace('_', '-')}", dest=c, type=float, default=1.0)
    args = ap.parse_args()

    weights = {c: getattr(args, c) for c in components}
    if args.weight is not None:
        weights = dict.fromkeys(components, args.weight)

    n_layers = model.num_layers(args.bf16)
    first, last = (0, n_layers - 1) if args.layers == "all" else (int(x) for x in args.layers.split("-"))

    # Learned weights: every layer has its own weight for each component and
    # each direction, so there is no single layer range or component weight.
    learned = None
    if args.weights_file:
        import json
        learned = {int(k): v for k, v in json.load(open(args.weights_file))["weights"].items()}
        first, last = min(learned), max(learned)
    direction_sets = [load_file(p)["directions"] for p in args.directions]

    # --place: each direction has its own layer range and component weights.
    # Expressed as the same per-layer table a learned-weights file holds, so the
    # two share everything downstream. A layer outside a direction's range gets
    # weight 0 for it.
    if args.place:
        if learned is not None:
            sys.exit("--place and --weights-file are two ways to say the same thing; give one")
        if len(args.place) != len(direction_sets):
            sys.exit(f"{len(args.place)} --place for {len(direction_sets)} --directions; give one each, in order")
        spans = []
        for spec in args.place:
            rng, _, vals = spec.partition(":")
            a, b = (int(x) for x in rng.split("-"))
            vals = [float(x) for x in vals.split(",")]
            if len(vals) != len(components):
                sys.exit(f"--place {spec!r}: {len(vals)} weights for components {components}")
            spans.append((a, b, dict(zip(components, vals))))
        first, last = min(a for a, _, _ in spans), max(b for _, b, _ in spans)
        learned = {
            layer: {c: [w[c] if a <= layer <= b else 0.0 for a, b, w in spans] for c in components}
            for layer in range(first, last + 1)
        }

    scales = [float(x) for x in args.scales.split(",")] if args.scales else [1.0] * len(direction_sets)
    if len(scales) != len(direction_sets):
        sys.exit(f"--scales has {len(scales)} value(s) for {len(direction_sets)} --directions")
    bf16, quant = Checkpoint(args.bf16), Checkpoint(args.quantized)
    scope = "each layer's own" if args.global_layer is None else f"layer {args.global_layer}'s"
    if learned is not None:
        source = f"learned weights from {args.weights_file}" if args.weights_file else f"placed {args.place}"
        print(f"{len(direction_sets)} direction(s) per layer ({scope}), layers {first}-{last}, {source}")
    else:
        print(f"{len(direction_sets)} direction(s) per layer ({scope}, scales {scales}), "
              f"layers {first}-{last}, weights {weights}")

    def rel(a, b):
        return ((a - b).norm() / b.norm()).item()

    native = getattr(model, "NATIVE_QUANT", False)
    plan = {}       # native archs: quantized name -> (component, basis, coef)
    replace = {}
    for layer in range(first, last + 1):
        # Index 0 of an exported tensor is the embedding output; layer i is at i + 1.
        source = layer if args.global_layer is None else args.global_layer
        basis, layer_scales = layer_basis(direction_sets, source + 1, scales)
        notes = [f"layer {layer:>2} rank {basis.shape[0]}"]
        if learned is not None and layer not in learned:
            continue

        if native:
            # Nothing is read here: build_native decodes, edits and re-encodes
            # each planned matrix shard by shard. coef_k = weight * scale_k.
            counts = {}
            for component, kind, bf16_key, qname in model.targets(layer, bf16):
                if basis.numel() == 0:
                    continue
                if learned is not None:
                    coef = torch.tensor(learned[layer][component], dtype=torch.float32)
                else:
                    coef = layer_scales * weights[component]
                if not coef.any():
                    continue
                plan[qname] = (component, basis, coef)
                counts[component] = counts.get(component, 0) + 1
            notes += [f"{c} x{n}" for c, n in counts.items()] or ["untouched"]
            print("  " + "  ".join(notes), flush=True)
            continue

        for component, kind, bf16_key, qname in model.targets(layer, bf16):
            # Skip a matrix this configuration does not change, so it is copied
            # byte-for-byte from the official checkpoint rather than re-encoded.
            # This matters when the encoders do not reproduce the checkpoint's
            # bytes exactly (Nemotron's FP8/NVFP4 are QAT, not a plain encoding of
            # the BF16 weights): re-encoding an unmodified matrix would then add
            # drift that is not the ablation's. The BF16 writers (moe_latent) are
            # reproduced exactly, so editing those stays clean.
            if basis.numel() == 0:
                continue
            if learned is not None:
                per_direction = torch.tensor(learned[layer][component], dtype=torch.float32)
                if not per_direction.any():
                    notes.append(f"{component} untouched")
                    continue
                if per_direction.numel() != basis.shape[0]:
                    sys.exit(f"layer {layer} {component}: {per_direction.numel()} learned weights "
                             f"for {basis.shape[0]} directions -- wrong --directions for this file?")
                w0 = bf16.get(bf16_key).to(torch.float32)
                w1 = ablate(w0, basis, per_direction, 1.0)
            else:
                if weights[component] == 0:
                    notes.append(f"{component} untouched")
                    continue
                w0 = bf16.get(bf16_key).to(torch.float32)
                w1 = ablate(w0, basis, layer_scales, weights[component])
            if kind == "fused":
                for e in range(w1.shape[0]):
                    encode(w1[e], model.expert_name(qname, e), quant, replace)
            else:
                encode(w1, qname, quant, replace)
            notes.append(f"{component} {rel(w1, w0):.4f}")
        print("  " + "  change ".join(notes), flush=True)

    if native:
        if not plan:
            # A control: nothing to edit, so the output is the checkpoint as shipped.
            import shutil
            print("no matrix edited (all weights 0): copying the checkpoint unchanged")
            shutil.copytree(args.quantized, args.out, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns(".cache"))
            return
        n = build_native(args, model, plan, args.device)
        print(f"done: {n} matrices edited -> {args.out}")
        return

    written = write_checkpoint(args.quantized, args.out, replace)
    print(f"done: {written} tensors replaced -> {args.out}")


if __name__ == "__main__":
    main()
