#!/usr/bin/env python3
"""Checks for DeepSeek-V4's storage formats and the edit-survival fix, on synthetic
matrices (no checkpoint needed; CPU is fine).

    python3 tests/test_quant_formats.py

1. Round trip: a matrix already on the FP8-block / FP4 grid decodes, re-encodes
   and decodes to exactly the same values (the property check_formats.py checks
   on the real checkpoint).
2. The problem: project a direction out of an on-grid matrix (W1 = W0 - w v v^T W0)
   and re-encode it by rounding to nearest. Much of the removed component comes
   back, most of it for FP4.
3. The fix: dsv4_encode_steered re-encodes the same W1 so that the stored matrix's
   component along v is what the edit asked for, while every element stays on
   one of the two grid values either side of it.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
from quant_formats import dsv4_decode, dsv4_encode, dsv4_encode_steered  # noqa: E402

torch.manual_seed(0)
OUT, IN = 512, 256


def on_grid(kind):
    """A random weight matrix as the checkpoint stores it: already on the grid."""
    w = torch.randn(OUT, IN) * 0.02
    q, s = dsv4_encode(w, kind)
    return dsv4_decode(q, s)


def component(v, w, w0):
    return ((v @ w).norm() / (v @ w0).norm()).item()


def check(kind, weight, tol):
    w0 = on_grid(kind)

    # 1. round trip
    q, s = dsv4_encode(w0, kind)
    assert torch.equal(dsv4_decode(q, s), w0), f"{kind}: round trip changed values"

    # 2. and 3. an edit of the given weight along a random unit direction
    v = torch.randn(OUT)
    v /= v.norm()
    w1 = w0 - weight * torch.outer(v, v @ w0)
    target = abs(1.0 - weight)

    near = dsv4_decode(*dsv4_encode(w1, kind))
    steered = dsv4_decode(*dsv4_encode_steered(w1, kind, v[None, :]))
    got_near, got_steered = component(v, near, w0), component(v, steered, w0)
    # Signed, to show when rounding has flipped the edit back.
    signed = lambda w: ((v @ w) @ (v @ w0) / (v @ w0).norm() ** 2).item()  # noqa: E731
    print(f"{kind:9} weight {weight:3}: target component {target:.3f} "
          f"({1 - weight:+.2f})   nearest rounding {got_near:.3f} ({signed(near):+.2f})   "
          f"steered {got_steered:.3f} ({signed(steered):+.2f})")

    assert abs(got_steered - target) < tol, f"{kind}: steered rounding missed the edit"
    assert abs(got_near - target) > abs(got_steered - target), f"{kind}: steering did not help"
    # Every element ends on the grid: re-encoding the steered matrix changes nothing.
    assert torch.equal(dsv4_decode(*dsv4_encode(steered, kind)), steered), f"{kind}: off-grid result"


if __name__ == "__main__":
    for kind in ("fp8_block", "fp4"):
        check(kind, 1.0, tol=0.02)
        check(kind, 2.5, tol=0.02)
    print("ok")
