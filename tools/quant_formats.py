"""The two weight formats used by nvidia/Qwen3.6-35B-A3B-NVFP4, encode and decode.

Abliteration changes a small number of matrices. Rather than re-quantize a whole
model, transplant.py re-encodes only those and keeps every other tensor of the
official checkpoint byte-for-byte. These functions are that re-encoding, and
check_formats.py proves them against the official checkpoint: encoding the
ORIGINAL weights must reproduce NVIDIA's bytes.

Layout, read from the checkpoint's safetensors headers:

  FP8, per tensor (attention projections)
    <name>.weight         F8_E4M3  [out, in]
    <name>.weight_scale   F32      []        W ~= weight * weight_scale
    <name>.input_scale    F32      []        activation scale, from calibration

  NVFP4, W4A16, group size 16 (experts, shared expert)
    <name>.weight         U8       [out, in/2]    two 4-bit E2M1 codes per byte
    <name>.weight_scale   F8_E4M3  [out, in/16]   one scale per 16 inputs
    <name>.weight_scale_2 F32      []             global scale
    <name>.input_scale    F32      []
    W ~= e2m1(code) * weight_scale * weight_scale_2
"""

import torch

FP8_MAX = 448.0  # largest finite float8_e4m3fn
E2M1_MAX = 6.0
NVFP4_GROUP = 16

# The eight magnitudes a 3-bit E2M1 code can take; bit 3 of the nibble is sign.
_E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
# Midpoints between adjacent magnitudes.
_E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
# Round-half-to-even: at these midpoints the upper neighbour has the even
# mantissa, so a tie rounds up.
_E2M1_TIES_UP = torch.tensor([0.75, 1.75, 3.5])


def _as_stored(w: torch.Tensor) -> torch.Tensor:
    """The weight as a BF16 checkpoint would hold it, widened to FP32.

    An edited matrix arrives in FP32. The model it stands for is BF16, and that
    is what NVIDIA's tool was given, so round to BF16 first.
    """
    return w.to(torch.bfloat16).to(torch.float32)


def fp8_encode(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """-> (weight F8_E4M3, weight_scale F32 scalar)."""
    w = _as_stored(w)
    # Multiply by the reciprocal, as for NVFP4's global scale below.
    scale = w.abs().max() * (1.0 / FP8_MAX)
    # The division is done in BF16, not FP32. Established against the official
    # checkpoint (encoder_variants.py): this reproduces its bytes exactly, and
    # FP32 division only 96-97% of them.
    q = (w.to(torch.bfloat16) / scale.to(torch.bfloat16)).to(torch.float32)
    q = q.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale.to(torch.float32)


def fp8_decode(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.to(torch.float32) * scale.to(torch.float32)


def _e2m1_codes(x: torch.Tensor, signed_zero: bool = False) -> torch.Tensor:
    """Nearest E2M1 code (uint8, 0..15) for each element of an already-scaled x."""
    mag = x.abs()
    idx = torch.sum(mag.unsqueeze(-1) > _E2M1_BOUNDS.to(x.device), dim=-1)
    idx = idx + torch.isin(mag, _E2M1_TIES_UP.to(x.device)).to(idx.dtype)
    # A negative weight that rounds to zero keeps its sign bit (code 8, "-0").
    # It decodes to the same value as code 0, so this changes nothing but the
    # bytes -- and the official checkpoint's bytes are what it matches.
    # signed_zero also keeps it for an input that is already -0.0 (a decoded
    # code 8), so a decode/encode round trip gives back the same bytes.
    sign = torch.signbit(x) if signed_zero else x < 0
    return (idx + sign.to(idx.dtype) * 8).to(torch.uint8)


def nvfp4_encode(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """-> (weight U8 [out, in/2], weight_scale F8_E4M3 [out, in/16], weight_scale_2 F32)."""
    w = _as_stored(w)
    out_f, in_f = w.shape
    assert in_f % NVFP4_GROUP == 0, f"in_features {in_f} not a multiple of {NVFP4_GROUP}"

    # Multiply by the reciprocal, do not divide. In FP32 the two differ in the
    # last bit for about a third of matrices, and that bit decides which way
    # exact ties fall further down (a scaled weight of exactly 3.5, a block
    # scale exactly between two FP8 values). Established against the official
    # checkpoint: this form is bit-identical on every matrix checked.
    scale_2 = w.abs().max() * (1.0 / (E2M1_MAX * FP8_MAX))
    blocks = w.view(out_f, in_f // NVFP4_GROUP, NVFP4_GROUP)
    block_scale = blocks.abs().amax(dim=-1) / (E2M1_MAX * scale_2)
    block_scale = torch.where(block_scale == 0, torch.ones_like(block_scale), block_scale)
    block_scale_fp8 = block_scale.clamp(max=FP8_MAX).to(torch.float8_e4m3fn)

    # Quantize against the scale as it will be stored, not the ideal one.
    denom = block_scale_fp8.to(torch.float32).unsqueeze(-1) * scale_2
    codes = _e2m1_codes(blocks / denom).view(out_f, in_f)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed, block_scale_fp8, scale_2.to(torch.float32)


def nvfp4_decode(
    packed: torch.Tensor, block_scale: torch.Tensor, scale_2: torch.Tensor
) -> torch.Tensor:
    out_f, half = packed.shape
    codes = torch.empty((out_f, half * 2), dtype=torch.uint8)
    codes[:, 0::2] = packed & 0x0F
    codes[:, 1::2] = packed >> 4
    mag = _E2M1_VALUES[(codes & 0x07).long()]
    val = torch.where((codes & 0x08) != 0, -mag, mag)
    scale = block_scale.to(torch.float32).repeat_interleave(NVFP4_GROUP, dim=1)
    return val * scale * scale_2.to(torch.float32)


# --- DeepSeek-V4's native formats ----------------------------------------------
#
# DeepSeek ships V4 in the precision it was trained in, so there is no BF16
# original: these ARE the weights. Layout, read from the checkpoint:
#
#   FP8, 128x128 blocks (attention projections, shared expert)
#     <name>.weight   F8_E4M3  [out, in]
#     <name>.scale    F8_E8M0  [out/128, in/128]    W = weight * scale (per block)
#
#   FP4, 32-wide groups along the input (routed experts)
#     <name>.weight   I8       [out, in/2]   two E2M1 codes per byte, element 2j
#                                            in the low nibble, 2j+1 in the high
#     <name>.scale    F8_E8M0  [out, in/32]  W = e2m1(code) * scale (per group)
#
# Every scale is a power of two (E8M0 has no mantissa), chosen as DeepSeek's
# kernel.py does: 2^ceil(log2(amax / max)), max 448 for FP8 and 6 for E2M1.
# Dividing by a power of two is exact, so rounding to the grid is the only lossy
# step, and these encoders reproduce the checkpoint's arithmetic exactly.

DSV4_FP8_BLOCK = 128
DSV4_FP4_GROUP = 32
_E8M0 = torch.float8_e8m0fnu


def _pow2_ceil(x: torch.Tensor) -> torch.Tensor:
    """2^ceil(log2(x)) for positive FP32 x, as kernel.py's fast_round_scale computes it.

    That reads the IEEE exponent and adds one unless the mantissa is zero; frexp
    gives x = m * 2^e with m in [0.5, 1), where m == 0.5 is exactly a power of two.
    """
    m, e = torch.frexp(x)
    return torch.ldexp(torch.ones_like(x), torch.where(m == 0.5, e - 1, e))


def dsv4_fp8_encode(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[out, in] -> (weight F8_E4M3 [out, in], scale F8_E8M0 [out/128, in/128])."""
    w = w.to(torch.float32)
    out_f, in_f = w.shape
    b = DSV4_FP8_BLOCK
    assert out_f % b == 0 and in_f % b == 0, f"{list(w.shape)} not in {b}x{b} blocks"
    blocks = w.view(out_f // b, b, in_f // b, b)
    # The 1e-4 floor is act_quant's; it only matters for an all-zero block.
    amax = blocks.abs().amax(dim=(1, 3)).clamp(min=1e-4)
    scale = _pow2_ceil(amax * (1.0 / FP8_MAX))
    q = (blocks / scale[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.view(out_f, in_f), scale.to(_E8M0)


def dsv4_fp8_decode(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    out_f, in_f = q.shape
    b = DSV4_FP8_BLOCK
    s = scale.to(torch.float32)[:, None, :, None]
    return (q.to(torch.float32).view(out_f // b, b, in_f // b, b) * s).view(out_f, in_f)


def dsv4_fp4_encode(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[out, in] -> (weight I8 [out, in/2], scale F8_E8M0 [out, in/32])."""
    w = w.to(torch.float32)
    out_f, in_f = w.shape
    g = DSV4_FP4_GROUP
    assert in_f % g == 0, f"in_features {in_f} not a multiple of {g}"
    groups = w.view(out_f, in_f // g, g)
    # fp4_quant_kernel's floor: the smallest scale E8M0 can hold times 6.
    amax = groups.abs().amax(dim=-1).clamp(min=E2M1_MAX * 2.0**-126)
    scale = _pow2_ceil(amax * (1.0 / E2M1_MAX))
    x = (groups / scale.unsqueeze(-1)).clamp(-E2M1_MAX, E2M1_MAX)
    codes = _e2m1_codes(x, signed_zero=True).view(out_f, in_f)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed.view(torch.int8), scale.to(_E8M0)


def dsv4_fp4_decode(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    p = packed.view(torch.uint8)
    out_f, half = p.shape
    codes = torch.empty((out_f, half * 2), dtype=torch.uint8, device=p.device)
    codes[:, 0::2] = p & 0x0F
    codes[:, 1::2] = p >> 4
    table = _E2M1_VALUES.to(p.device)
    mag = table[(codes & 0x07).long()]
    val = torch.where((codes & 0x08) != 0, -mag, mag)
    return val * scale.to(torch.float32).repeat_interleave(DSV4_FP4_GROUP, dim=1)


def dsv4_kind(weight_dtype: torch.dtype) -> str:
    """Which DeepSeek-V4 format a <name>.weight of this dtype is stored in."""
    if weight_dtype == torch.float8_e4m3fn:
        return "fp8_block"
    if weight_dtype in (torch.int8, torch.uint8):
        return "fp4"
    raise ValueError(f"not a DeepSeek-V4 quantized weight dtype: {weight_dtype}")


def dsv4_decode(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if dsv4_kind(weight.dtype) == "fp8_block":
        return dsv4_fp8_decode(weight, scale)
    return dsv4_fp4_decode(weight, scale)


def dsv4_encode(w: torch.Tensor, kind: str) -> tuple[torch.Tensor, torch.Tensor]:
    return dsv4_fp8_encode(w) if kind == "fp8_block" else dsv4_fp4_encode(w)


# --- Direction-aware rounding -----------------------------------------------------
#
# DeepSeek's weights sit exactly on the storage grid. An abliteration edit moves
# each element by a fraction of a grid step, so rounding the edited matrix to
# nearest puts most elements straight back where they were -- and with them the
# direction the edit removed. Measured on layer 20 with a random direction at
# weight 1: 91% of the component survives in the FP4 experts, 37% in the FP8
# attention output, against 2-3% for a direction the edit did not touch.
#
# The fix keeps nearest rounding as the starting point and the scales as the
# encoder chose them, then, column by column, moves the elements that are
# cheapest to move (those closest to the midpoint between two grid values) to
# their other neighbour until the stored column's component along each edited
# direction b matches the edit's: b^T Q[:, j] ~= b^T W1[:, j]. Every element
# still ends on one of the two grid values either side of it, so the elementwise
# error stays under one grid step; only which side is chosen changes.

def _grid(kind: str, device) -> torch.Tensor:
    """Every finite value of the element format, sorted (scaled units, before the scale)."""
    if kind == "fp8_block":
        vals = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).to(torch.float32)
        vals = vals[torch.isfinite(vals)]
    else:
        vals = torch.cat([-_E2M1_VALUES, _E2M1_VALUES])
    return torch.unique(vals).to(device)        # one 0, ascending


def _element_scale(scale: torch.Tensor, kind: str, shape) -> torch.Tensor:
    s = scale.to(torch.float32)
    if kind == "fp8_block":
        b = DSV4_FP8_BLOCK
        return s.repeat_interleave(b, dim=0).repeat_interleave(b, dim=1)[: shape[0], : shape[1]]
    return s.repeat_interleave(DSV4_FP4_GROUP, dim=1)


def dsv4_encode_steered(w1: torch.Tensor, kind: str, basis: torch.Tensor,
                        passes: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode w1 [out, in] so the stored matrix keeps the edit along each row of basis [k, out]."""
    q, s = dsv4_encode(w1, kind)
    w1 = w1.to(torch.float32)
    se = _element_scale(s, kind, w1.shape).to(w1.device)
    x = w1 / se                                  # exact: se is a power of two
    v = dsv4_decode(q, s) / se                   # nearest grid value, scaled
    grid = _grid(kind, w1.device)
    top = grid.numel() - 1
    for _ in range(passes):
        for b in basis.to(w1.device, torch.float32):
            resid = b @ ((v - x) * se)           # [in]: stored minus intended, along b
            # Moving element i by delta changes resid_j by b_i * delta * se_ij;
            # the useful move has the opposite sign to resid_j.
            step = (-torch.sign(resid)[None, :] * torch.sign(b)[:, None]).to(torch.long)
            idx = torch.searchsorted(grid, v.contiguous())
            cidx = (idx + step).clamp(0, top)
            cand = grid[cidx]
            ok = (cidx != idx) & (step != 0)
            gain = torch.where(ok, (b[:, None] * (cand - v) * se).abs(), 0.0)
            cost = ((cand - x).abs() - (v - x).abs()) * se
            ratio = torch.where(ok & (gain > 0), cost / gain, torch.inf)
            order = torch.argsort(ratio, dim=0)
            g = torch.gather(gain, 0, order)
            before = torch.cumsum(g, dim=0) - g  # reduction already taken before this one
            need = resid.abs()[None, :]
            # Take a move while it brings |resid| closer to zero than leaving it.
            take_sorted = (before < need) & ((need - before - g).abs() < (need - before)) \
                & torch.isfinite(torch.gather(ratio, 0, order))
            take = torch.zeros_like(take_sorted)
            take.scatter_(0, order, take_sorted)
            v = torch.where(take, cand, v)
    if kind == "fp8_block":
        q = v.to(torch.float8_e4m3fn)            # exact: v is on the grid
    else:
        codes = _e2m1_codes(v, signed_zero=True)
        q = (codes[:, 0::2] | (codes[:, 1::2] << 4)).view(torch.int8)
    return q, s
