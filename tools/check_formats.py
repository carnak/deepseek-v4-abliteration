#!/usr/bin/env python3
"""Prove quant_formats.py against the official quantized checkpoint.

For matrices of the kinds abliteration modifies, encode the ORIGINAL BF16 weight
and compare with what NVIDIA shipped. Agreement here is what licenses
transplant.py / ablate.py to write re-encoded tensors into that checkpoint.

    python3 check_formats.py [--arch ARCH] <bf16_dir> <quantized_dir> [layer ...]

Reports, per matrix: how well the shipped tensors decode back to the original
(layout is right), and how many of our encoded bytes equal theirs (the encoder
is the same one). A matrix the quantized checkpoint stores unquantized (BF16,
e.g. Nemotron's fc2_latent_proj) is compared directly and must be byte-identical.
"""
import argparse
import glob
import json
import os
import struct
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arch as arch_registry  # noqa: E402
from quant_formats import fp8_decode, fp8_encode, nvfp4_decode, nvfp4_encode  # noqa: E402


class Checkpoint:
    """Random access to tensors of a sharded safetensors directory."""

    def __init__(self, path):
        self.path = path
        self.where = {}
        self.info = {}       # name -> {"dtype": "F8_E4M3", "shape": [...]}, from the headers
        for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
            with open(f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                for k, v in json.loads(fh.read(n)).items():
                    if k != "__metadata__":
                        self.where[k] = f
                        self.info[k] = v

    def __contains__(self, name):
        return name in self.where

    def stored_dtype(self, name):
        """The safetensors dtype string of a tensor (e.g. "I8", "F8_E4M3"), without reading it."""
        return self.info[name]["dtype"]

    def get(self, name):
        with safe_open(self.where[name], framework="pt") as f:
            return f.get_tensor(name)


def rel_err(a, b):
    return ((a - b).norm() / b.norm()).item()


def same(a, b):
    """Fraction of equal elements, comparing raw bytes for FP8."""
    if a.dtype == torch.float8_e4m3fn:
        a, b = a.view(torch.uint8), b.view(torch.uint8)
    return (a == b).float().mean().item()


def check_matrix(name, w, nv):
    """Verify one [out, in] matrix `name` against the shipped quantized tensors.

    Returns (kind, their_decode_err, our_decode_err, match_dict). The format is
    read from which scale tensors the checkpoint ships: both scales -> NVFP4, one
    -> FP8, none -> stored unquantized (compared byte for byte).
    """
    w = w.to(torch.float32)
    theirs = nv.get(name + ".weight")
    if name + ".weight_scale_2" in nv:
        kind = "NVFP4"
        t_scale, t_scale2 = nv.get(name + ".weight_scale"), nv.get(name + ".weight_scale_2")
        dec = rel_err(nvfp4_decode(theirs, t_scale, t_scale2), w)
        q, s, s2 = nvfp4_encode(w)
        ours = rel_err(nvfp4_decode(q, s, s2), w)
        match = {
            "weight": same(q, theirs),
            "weight_scale": same(s, t_scale),
            "weight_scale_2": float(torch.equal(s2, t_scale2)),
        }
    elif name + ".weight_scale" in nv:
        kind = "FP8"
        t_scale = nv.get(name + ".weight_scale")
        dec = rel_err(fp8_decode(theirs, t_scale), w)
        q, s = fp8_encode(w)
        ours = rel_err(fp8_decode(q, s), w)
        match = {
            "weight": same(q, theirs),
            "weight_scale": float(torch.equal(s, t_scale)),
        }
    else:
        # Stored unquantized in the "quantized" checkpoint: we write it back
        # verbatim, so the only thing to prove is that it is byte-identical.
        kind = "BF16"
        dec = ours = rel_err(theirs.to(torch.float32), w)
        match = {"weight": float(torch.equal(theirs, w.to(theirs.dtype)))}
    return kind, dec, ours, match


def check_native(model, ckpt, layers):
    """A natively quantized checkpoint (DeepSeek-V4): decode, re-encode, compare.

    There is no BF16 original to encode, so the question is narrower: does
    re-encoding what is stored give back what is stored? Bytes equal means the
    encoder is the one DeepSeek used and an unedited matrix would survive a
    rebuild untouched. Values equal with bytes differing is also safe: a block
    whose stored scale is larger than its contents need gets a smaller scale on
    re-encoding and the same numbers. Neither is required -- ablate.py writes an
    unedited matrix back verbatim -- but it says whether an edited matrix's
    re-encode adds anything beyond the edit's own rounding.
    """
    from quant_formats import dsv4_decode, dsv4_encode, dsv4_kind

    worst_values = 1.0
    for layer in layers:
        experts = []
        for component, _, key, qname in model.targets(layer, ckpt):
            if component == "experts":
                experts.append(qname)
                continue
            worst_values = min(worst_values, _check_one(layer, component, qname, ckpt,
                                                        dsv4_decode, dsv4_encode, dsv4_kind))
        for qname in (experts[0], experts[len(experts) // 2], experts[-1]):
            worst_values = min(worst_values, _check_one(layer, "experts", qname, ckpt,
                                                        dsv4_decode, dsv4_encode, dsv4_kind))
    print(f"\nworst value match across all checked tensors: {worst_values:.6f}")
    return 0 if worst_values > 0.9999 else 1


def _check_one(layer, component, qname, ckpt, decode, encode, kind_of):
    w, s = ckpt.get(qname + ".weight"), ckpt.get(qname + ".scale")
    kind = kind_of(w.dtype)
    dec = decode(w, s)
    q, s2 = encode(dec, kind)
    back = decode(q, s2)
    wb = (q.view(torch.uint8) == w.view(torch.uint8)).float().mean().item()
    sb = (s2.view(torch.uint8) == s.view(torch.uint8)).float().mean().item()
    vals = (back == dec).float().mean().item()
    tag = qname.rsplit(".", 2)[-2] if component == "experts" else ""
    print(f"layer {layer:>2} {component:<7}{tag:>4} {kind:<9} {list(dec.shape)}  "
          f"weight bytes {wb:.6f}  scale bytes {sb:.6f}  values {vals:.6f}")
    return vals


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arch", default="qwen3_5_moe")
    ap.add_argument("bf16")
    ap.add_argument("quantized")
    ap.add_argument("layers", nargs="*", type=int,
                    help="layer indices to check; default spans every block type of both archs")
    args = ap.parse_args()

    model = arch_registry.get(args.arch)
    if getattr(model, "NATIVE_QUANT", False):
        return check_native(model, Checkpoint(args.quantized), args.layers or [0, 3, 20, 42])
    bf16, nv = Checkpoint(args.bf16), Checkpoint(args.quantized)
    layers = args.layers or [0, 1, 7, 20]
    worst = 1.0

    for layer in layers:
        for component, kind, bf16_key, qname in model.targets(layer, bf16):
            if kind == "fused":
                # BF16 holds one [experts, out, in] tensor; the quantized
                # checkpoint, one matrix per expert. Sample a few experts.
                fused = bf16.get(bf16_key).to(torch.float32)
                for e in (0, fused.shape[0] // 2, fused.shape[0] - 1):
                    name = model.expert_name(qname, e)
                    k, dec, ours, match = check_matrix(name, fused[e], nv)
                    worst = min(worst, *match.values())
                    print(f"layer {layer:>2} {component}[{e}] {k:<5} {list(fused[e].shape)}  "
                          f"their decode err {dec:.4f}  our decode err {ours:.4f}  "
                          + "  ".join(f"{m} match {v:.6f}" for m, v in match.items()))
                continue

            if bf16_key not in bf16:
                continue  # layer does not have this writer (block-type specific)
            k, dec, ours, match = check_matrix(qname, bf16.get(bf16_key), nv)
            worst = min(worst, *match.values())
            print(f"layer {layer:>2} {component:<11} {k:<5} {list(bf16.get(bf16_key).shape)}  "
                  f"their decode err {dec:.4f}  our decode err {ours:.4f}  "
                  + "  ".join(f"{m} match {v:.6f}" for m, v in match.items()))

    print(f"\nworst match across all checked tensors: {worst:.6f}")
    # Identical bytes, scales and weights alike. Anything less means re-encoding
    # an unmodified matrix would change the model, and that shows up as drift
    # that is not the ablation's (it was KL 0.024 before the encoders matched).
    return 0 if worst > 0.9999 else 1


if __name__ == "__main__":
    sys.exit(main())
