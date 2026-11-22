# Editing natively quantized weights

DeepSeek released V4 in the precision it was trained in. There is no BF16
checkpoint to edit and re-quantize, so the stock FP8/FP4 checkpoint is the
original. [`tools/ablate.py`](../tools/ablate.py) (`build_native`) works shard
by shard:

1. load the shard; for each tensor the plan edits, decode it to FP32
2. apply `W − w·v vᵀW` in FP32 (on the GPU)
3. re-encode it in the format it came in, with direction-aware rounding
4. measure how much of the edit survived, assert dtype and shape unchanged
5. write the shard back with every other tensor copied byte for byte

## The formats

[`tools/quant_formats.py`](../tools/quant_formats.py), following DeepSeek's
reference `inference/kernel.py`. Every scale is a power of two (UE8M0), chosen as
`2^ceil(log2(amax / max))` over its block, so the largest element fits.

| | FP8 block (`fp8_block`) | FP4 (`fp4`) |
|---|---|---|
| used for | attention `wo_b`, shared expert `w2` | routed experts' `w2` |
| element | E4M3 (`float8_e4m3fn`), max 448 | E2M1, max 6: magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6 |
| scale | one per 128×128 block | one per 32 consecutive inputs of a row |
| storage | `float8_e4m3fn` | two codes per `int8` (low nibble = even element) |

One subtlety: FP4 has a negative zero. A decoded −0.0 must re-encode as code 8,
not 0, or 12.5% of bytes change with no change in value
(`_e2m1_codes(..., signed_zero=True)`).

**Round trip.** `tools/check_formats.py --arch deepseek_v4` decodes and
re-encodes stock matrices from layers 0, 3, 20 and 42:

- decoded values match exactly in every matrix checked (the pipeline's
  preflight requires this)
- FP4 bytes match exactly
- FP8 bytes match 89-100%. Every differing byte belongs to a block whose stored
  scale is looser than its contents need. Re-encoding picks the tighter scale,
  which represents the same numbers.

## Rounding undoes the edit

The stock weights already sit exactly on the storage grid. An edit moves each
element by a small fraction of a grid step. Rounding to nearest puts most
elements straight back, and the removed direction comes back with them. This is
the caveat from arXiv 2611.09382 ("Exploring Weight-Space Programming") that
weight edits can fall below storage resolution. Here it is severe.

Measured on layer 20 with a random unit direction at weight 1 (target: no
component left). The entries are how much of the original component along `v`
the stored matrix still has:

| writer | nearest rounding | steered | rounding noise along a random direction |
|---|---|---|---|
| attn `wo_b` (FP8) | 0.37 | 0.001 | 0.010 → 0.014 |
| shared `w2` (FP8) | 0.37 | 0.001 | 0.010 → 0.014 |
| expert `w2` (FP4) | 0.89-0.92 | 0.005-0.006 | 0.02-0.03 → 0.03-0.06 |

At weight 2 (target: the component reversed, magnitude 1), nearest rounding
left the FP4 experts' component at **+0.7 of the original instead of −1**. The
edit was almost entirely undone.

## Direction-aware rounding

`dsv4_encode_steered(w1, kind, basis)`:

1. Start from nearest rounding, with the scales the plain encoder chose.
2. For each basis direction `b`, each column `j` has a residual
   `r_j = bᵀ(stored − intended)[:, j]`: how far the stored column is from the
   edit along `b`.
3. Every element can move to its other neighbouring grid value. Moving element
   `i` changes `r_j` by `b_i · Δ_ij · scale_ij`. A move is useful if it has the
   opposite sign to `r_j`.
4. Rank the useful moves by cost per unit gain. Cost is how much further the
   element ends from its exact value; elements near a midpoint are nearly free.
   Take them in order while each brings `|r_j|` closer to zero.
5. Two passes over the basis, then pack the result. It is exactly on the grid,
   with the same scales.

Every element still ends on one of the two grid values either side of its exact
value, so the per-element error stays within a grid step. The FP4 matrices'
total relative quantization error rises from about 0.025 to 0.04. In exchange,
the stored matrix carries the edit.

## The survival check

Every build prints, per component, relative to the original component
`|bᵀW0|` (`survival()` in `ablate.py`):

| column | meaning |
|---|---|
| target | `|1 − w|`, what the edit asked for |
| got | `|bᵀ stored|` |
| err | `|bᵀ (stored − W1)|`, rounding error along the direction |
| noise | the same along a random unit vector orthogonal to the basis |
| quant err | total relative rounding error of the matrix |

`err` near `noise` means rounding is not working against the edit, only adding
the noise it adds everywhere. A build fails if a component's median miss exceeds
`--survival-tol` (0.05). `--no-steer` turns steering off, to see the problem.

The kept build (5,934 matrices, layers 13-35, weight 2.5):

```
component       n  target     got     err   noise quant err
attn           23   1.500   1.500   0.001   0.019    0.0193   miss 0.000
shared         23   1.500   1.500   0.001   0.015    0.0196   miss 0.000
experts      5888   1.500   1.500   0.005   0.052    0.0585   miss 0.000
```

`tests/test_quant_formats.py` shows the same effect on synthetic matrices,
without the model. Synthetic weights are less clustered on the grid than real
ones, so nearest rounding loses less there (FP4: 0.56 of the component comes
back instead of 0.9). Steering still lands on the target.
