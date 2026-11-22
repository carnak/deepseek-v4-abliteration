#!/usr/bin/env python3
"""Build an abliterated NVFP4 checkpoint from the official one plus a heretic adapter.

Heretic expresses abliteration as a LoRA adapter over a few dozen matrices. The
official NVFP4 checkpoint is 124k tensors, nearly all of which abliteration
never touches. So instead of merging a 67 GB BF16 model and re-quantizing all
of it, this:

  1. reads each adapted module's ORIGINAL weight from the BF16 checkpoint,
  2. adds the adapter's delta (B @ A; heretic uses lora_alpha == r),
  3. re-encodes that one matrix in the format the official checkpoint uses for
     it (FP8 or NVFP4, see quant_formats.py),
  4. writes a copy of the official checkpoint with only those tensors replaced.

Routed MoE experts are handled the same way from heretic's sidecar file
fused_expert_ablations.safetensors, which sits next to the adapter because a
LoRA adapter cannot represent them.

Everything else -- untouched experts, the MTP head, activation scales, config
-- is byte-for-byte NVIDIA's.

    python3 transplant.py --bf16 <dir> --quantized <dir> --adapter <dir> --out <dir>
"""
import argparse
import glob
import json
import os
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_formats import Checkpoint  # noqa: E402
from quant_formats import fp8_decode, fp8_encode, nvfp4_decode, nvfp4_encode  # noqa: E402


def adapter_deltas(adapter_dir):
    """{base module name: (A, B)} from a PEFT LoRA directory."""
    path = os.path.join(adapter_dir, "adapter_model.safetensors")
    cfg = json.load(open(os.path.join(adapter_dir, "adapter_config.json")))
    r, alpha = cfg["r"], cfg["lora_alpha"]
    if alpha != r:
        # Heretic always sets them equal. Anything else is not a heretic adapter
        # and the scaling below would be wrong for it.
        sys.exit(f"adapter has lora_alpha {alpha} != r {r}; refusing to guess the scaling")

    sd = load_file(path)
    out = {}
    for key, a in sd.items():
        if not key.endswith(".lora_A.weight"):
            continue
        module = key[: -len(".lora_A.weight")]
        b = sd[module + ".lora_B.weight"]
        if not b.any():
            continue  # heretic leaves untouched modules at B == 0
        name = module.removeprefix("base_model.model.")
        out[name] = (a.to(torch.float32), b.to(torch.float32))
    return out


FUSED_FILE = "fused_expert_ablations.safetensors"


def write_checkpoint(quantized_dir, out_dir, replace):
    """Copy a checkpoint, substituting the tensors in `replace`. Returns the count replaced."""
    os.makedirs(out_dir, exist_ok=True)
    for f in sorted(os.listdir(quantized_dir)):
        src = os.path.join(quantized_dir, f)
        if os.path.isfile(src) and not f.endswith(".safetensors"):
            shutil.copy2(src, os.path.join(out_dir, f))

    written = 0
    for shard in sorted(glob.glob(os.path.join(quantized_dir, "*.safetensors"))):
        with safe_open(shard, framework="pt") as f:
            meta = f.metadata()
        tensors = load_file(shard)
        hits = [k for k in tensors if k in replace]
        for k in hits:
            assert tensors[k].dtype == replace[k].dtype and tensors[k].shape == replace[k].shape, k
            tensors[k] = replace[k]
        written += len(hits)
        save_file(tensors, os.path.join(out_dir, os.path.basename(shard)), metadata=meta)
        print(f"{os.path.basename(shard)}: {len(tensors)} tensors, {len(hits)} replaced")
        del tensors

    if written != len(replace):
        sys.exit(f"replaced {written} tensors but prepared {len(replace)} -- output is incomplete")
    return written


def fused_expert_replacements(adapter_dir, bf16, quant):
    """Re-encoded routed-expert matrices, from heretic's fused-expert sidecar.

    The BF16 checkpoint holds a layer's routed experts as one tensor
    [experts, out, in]; the NVFP4 checkpoint holds one matrix per expert. Heretic
    records, per layer, a unit direction v and a weight, meaning
    W_e -> W_e - weight * v (v^T W_e) for every expert e of that layer.
    """
    path = os.path.join(adapter_dir, FUSED_FILE)
    if not os.path.exists(path):
        return {}, 0

    sd = load_file(path)
    layers = sorted({int(k.split(".")[1]) for k in sd})
    replace = {}
    for layer in layers:
        v = sd[f"layers.{layer}.direction"].to(torch.float32)
        weight = sd[f"layers.{layer}.weight"].item()
        prefix = f"model.language_model.layers.{layer}.mlp.experts"
        w0 = bf16.get(prefix + ".down_proj").to(torch.float32)
        projection = torch.einsum("h,ehi->ei", v, w0)
        w1 = w0 - weight * v.view(1, -1, 1) * projection.unsqueeze(1)

        errs = []
        for e in range(w1.shape[0]):
            name = f"{prefix}.{e}.down_proj"
            q, s, s2 = nvfp4_encode(w1[e])
            for k, t in ((".weight", q), (".weight_scale", s), (".weight_scale_2", s2)):
                old = quant.get(name + k) if e == 0 else None  # shapes are uniform
                if old is not None:
                    assert old.dtype == t.dtype and old.shape == t.shape, name + k
                replace[name + k] = t
            if e % 64 == 0:
                errs.append(((nvfp4_decode(q, s, s2) - w1[e]).norm() / w1[e].norm()).item())

        change = ((w1 - w0).norm() / w0.norm()).item()
        print(
            f"  layer {layer:>2} routed experts x{w1.shape[0]:<4} weight {weight:+.3f}  "
            f"change {change:.4f}  quant err {sum(errs) / len(errs):.4f}"
        )
    return replace, len(layers)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bf16", required=True, help="original unquantized checkpoint")
    ap.add_argument("--quantized", required=True, help="official NVFP4 checkpoint")
    ap.add_argument("--adapter", required=True, help="heretic LoRA adapter directory")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    bf16, quant = Checkpoint(args.bf16), Checkpoint(args.quantized)
    deltas = adapter_deltas(args.adapter)
    print(f"adapter modifies {len(deltas)} modules")

    # Re-encode every modified matrix up front, so a problem with any one of
    # them stops the run before anything is written.
    replace = {}
    for name, (a, b) in sorted(deltas.items()):
        wkey = name + ".weight"
        if wkey not in bf16 or wkey not in quant:
            sys.exit(f"{name}: not present in both checkpoints -- adapter is for a different model?")
        w0 = bf16.get(wkey).to(torch.float32)
        w1 = w0 + b @ a

        if name + ".weight_scale_2" in quant:
            kind = "NVFP4"
            q, s, s2 = nvfp4_encode(w1)
            replace[wkey], replace[name + ".weight_scale"], replace[name + ".weight_scale_2"] = q, s, s2
            back = nvfp4_decode(q, s, s2)
        elif name + ".weight_scale" in quant:
            kind = "FP8"
            q, s = fp8_encode(w1)
            replace[wkey], replace[name + ".weight_scale"] = q, s
            back = fp8_decode(q, s)
        else:
            kind = "BF16"
            replace[wkey] = w1.to(torch.bfloat16)
            back = replace[wkey].to(torch.float32)

        for k in (wkey, name + ".weight_scale", name + ".weight_scale_2"):
            if k in replace:
                old = quant.get(k)
                assert old.dtype == replace[k].dtype and old.shape == replace[k].shape, (
                    f"{k}: {replace[k].dtype}{list(replace[k].shape)} would replace "
                    f"{old.dtype}{list(old.shape)}"
                )

        change = ((w1 - w0).norm() / w0.norm()).item()
        err = ((back - w1).norm() / w1.norm()).item()
        print(f"  {name:<58} {kind:<5} change {change:.4f}  quant err {err:.4f}")

    fused, n_fused_layers = fused_expert_replacements(args.adapter, bf16, quant)
    replace.update(fused)
    if n_fused_layers:
        print(f"fused-expert ablations cover {n_fused_layers} layers")
    else:
        print(f"no {FUSED_FILE} in the adapter directory: routed experts left as shipped")

    written = write_checkpoint(args.quantized, args.out, replace)
    print(
        f"done: {written} tensors replaced ({len(deltas)} adapted modules, "
        f"{n_fused_layers} layers of routed experts) -> {args.out}"
    )


if __name__ == "__main__":
    main()
