# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Double-buffered implicit-GEMM conv3d (BF16).

x: (N, C, D, H, W) bf16 NCDHW by default, weight: (K, C/groups, T, R, S) bf16 KCTRS.
Returns (N, K, Do, Ho, Wo) bf16 by default. ``input_layout`` / ``output_layout`` select
NCDHW or NDHWC independently; the GEMM itself is channels-last, so NDHWC input skips the
pre-transpose and NDHWC output is the raw row-major (npq, K) the epilogue produces.
Supports stride, padding (int, per-axis tuple, or torch's "same" / "valid"),
padding_mode, dilation, bias, groups, and split-K.
"""

import functools
import os
import weakref

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import rocdl as rocdl_ods
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from kernels.common import buffer_ops
from kernels.common.mem_ops import buffer_atomic_add

TILE_K = 32
STAGES = 2
WARP_SIZE = 64

# K tiles consumed between two barriers. Each one is MI_M * MI_N MFMAs, and that product
# is the only thing that hides global latency here -- see the PIPE_STAGES comment for why
# the pipeline depth cannot. Costs no LDS (the tiles are stages that already exist) and no
# extra ds_read/DMA traffic; it just halves the number of barriers. Worth +8..16% on the
# 3x3 conv2d/conv3d shapes at 2. Reaching the same ratio through TILE_K = 64 instead is a
# trap: it makes the LDS row stride 128B, exactly one bank rotation, and the resulting
# ds_read_b128 conflicts cost more than the batching wins (measured ~15% slower).
TILES_PER_BARRIER = 2

MFMA_M = 16
MFMA_N = 16
MFMA_A_VALUES = 8
MFMA_B_VALUES = 8
MFMA_C_VALUES = 4

LDG_VEC = 8

BF16_BYTES = 2

DEFAULT_TILE = (128, 128, 2, 4)

TILE_LADDER = ((128, 128, 2, 4), (64, 64, 2, 2), (32, 32, 1, 2))

# The bandwidth-bound rung, widest first. What every entry buys is a TILE_N that spans
# K/groups in one step. The x grid axis is dispatched first, so all of one N tile's M
# blocks run before the next N tile starts and none of the activation survives in cache
# between them: a second N tile is a second full pass over the input. These cost up to
# 128KB of LDS, so the rung is filtered against the device budget before it is used.
TILE_LADDER_WIDE = ((256, 256, 2, 4), (128, 256, 2, 4), (256, 128, 2, 4))

# Fallback when K/groups is wider than any tile: swizzle the grid so the N tiles run over
# the same rows at the same time and share them in cache instead. Recovers most, not all,
# of what the wide tile would have (measured 189us -> 159us where the tile gives 144us).
TILE_WIDE_WGM = 8

TILE_MIN_WAVES_PER_CU = 6

# Machine flop:byte ratio above which the MFMAs, not DRAM, set the kernel's time. The true
# ridge point is ~245 on CDNA3 and ~460 on CDNA4; one constant in between classifies every
# shape measured, the nearest being 2x away on either side, so the rung below does not have
# to know which chip it is running on.
TILE_RIDGE_FLOP_PER_BYTE = 300

PADDING_MODES = ("zeros", "reflect", "replicate", "circular")

CONV_COMPILE_HINTS = {}


def _as_stream(stream):
    return stream if hasattr(stream, "_is_stream_param") else fx.Stream(stream)


def _dispatch(exe, *args, stream=None):
    """Run a builder's launcher, pre-compiling on first use."""
    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = exe.compile(*args, stream=stream)
        return
    cf(*args, _as_stream(stream))


def _autotune_enabled():
    return os.environ.get("FLYDSL_CONV3D_AUTOTUNE", "0").lower() in ("1", "true", "yes")


_WEIGHT_CACHE = {}

_BIAS_PLACEHOLDER = {}


def _pad_channels(c):
    return (c + LDG_VEC - 1) // LDG_VEC * LDG_VEC


def _bias_placeholder(device):
    """The stand-in passed for ``bias`` when there is none.

    The kernel only builds a bias resource under ``const_expr(has_bias)``, so nothing
    ever reads this; one shared tensor per device keeps a small-shape conv from paying
    for an allocation the GPU will not touch.
    """
    t = _BIAS_PLACEHOLDER.get(device)
    if t is None:
        t = torch.empty(1, device=device, dtype=torch.float32)
        _BIAS_PLACEHOLDER[device] = t
    return t


# Layout names accepted per spatial rank.
LAYOUTS = {
    3: ("NCDHW", "NDHWC"),
    2: ("NCHW", "NHWC"),
    1: ("NCW", "NWC"),
}


def _check_layouts(rank, input_layout, output_layout):
    names = LAYOUTS[rank]
    for what, v in (("input_layout", input_layout), ("output_layout", output_layout)):
        assert v in names, f"{what} must be one of {names}, got {v!r}"


def _shape_ncdhw(x, ndhwc):
    """Unpack a 5-D input in either layout to (n, c, d, h, w)."""
    if ndhwc:
        n, d, h, w, c = x.shape
    else:
        n, c, d, h, w = x.shape
    return n, c, d, h, w


def _pad_spatial(x, ndhwc, pads, mode):
    """Materialize (D, H, W) padding, torch's (w_lo, w_hi, h_lo, h_hi, d_lo, d_hi) order.

    Only the non-zero padding modes reach this. Zero padding is entirely a matter of the
    gather's range mask and the output extent, so it never needs a padded copy of the
    activation, however lopsided the pad is.
    """
    if ndhwc:
        x = x.permute(0, 4, 1, 2, 3)
    return torch.nn.functional.pad(x, pads, mode=mode), False


def _big_in(n, c, groups, d, h, w, pt, ph, pw):
    """Whether the kernel's 64-bit BIG_IN address path would engage for this input."""
    cp = _pad_channels(c // groups) * groups
    return n * cp * (d + 2 * pt) * (h + 2 * ph) * (w + 2 * pw) > 0x7FFFFFFF


# --- reading an NCDHW input directly, instead of pre-transposing it to NDHWC ---
#
# Both MFMA operands want K contiguous per lane, and an NCDHW activation is contiguous
# along the output rows instead, so something has to transpose. `_ncdhw_to_ndhwc` does it
# in a separate pass over HBM; for a filter with taps to reuse that is amortized, but a
# 1x1 filter reads every input element exactly once and the extra pass costs more than
# the convolution (536MB against a 403MB problem on 1,512,512,512 x 256,512,1,1).
#
# gfx950 can do it for free on the way out of LDS instead. `ds_read_tr16_b64` redistributes
# a 128-byte LDS run across a 16-lane group, so if a run holds 4 K values of 16 M rows,
# K-major, one instruction hands each lane 4 consecutive K of one row. The DMA then only
# has to fill runs, which it can do straight from NCDHW: a run's 16 M rows at one channel
# are 16 consecutive input elements.
#
# A wave's `buffer_load_lds` writes 8 runs, and those runs have to be 8 M blocks of the
# same K group for the DMA to stay coalesced, hence the floor on TILE_M.
NCHW_A_MIN_TILE_M = 128


@functools.lru_cache(maxsize=1)
def _has_lds_read_transpose():
    """Whether the GPU has the gfx950 ds_read_tr16 LDS transpose."""
    try:
        from flydsl.runtime.device import get_rocm_arch

        return str(get_rocm_arch()).startswith("gfx95")
    except Exception:
        return False


def _nchw_a_ok(in_ndhwc, n, c, d, h, w, kt, kh, kw, st, sh, sw, pt, ph, pw, pad_mode):
    """Whether the kernel can read this NCDHW input as-is, skipping the pre-transpose.

    Only a 1x1x1 unit-stride unpadded filter qualifies, which is both where the transpose
    hurts most and where the A row is the output row: no im2col gather, so a lane's LDS
    run maps to one contiguous span of the input. Everything else keeps the NDHWC path.
    """
    return (
        not in_ndhwc
        and _has_lds_read_transpose()
        and (kt, kh, kw) == (1, 1, 1)
        and (st, sh, sw) == (1, 1, 1)
        and (pt, ph, pw) == (0, 0, 0)
        and pad_mode == "zeros"
        # The gather addresses the input with a 32-bit element offset -- BIG_IN's rebased
        # descriptor assumes NDHWC row order and does not carry over.
        and n * c * d * h * w <= 0x7FFFFFFF
        # A thread DMAs LDG_VEC consecutive output rows at one channel, so a batch
        # boundary inside that span would silently roll over into the next channel.
        and (n == 1 or (d * h * w) % LDG_VEC == 0)
    )


def _lds_read_transpose_frag(result_type, lds_byte_addr, run_stride_bytes):
    """One 8-element MFMA A fragment, read out of a transposed LDS tile (gfx950).

    ``ds_read_tr16_b64`` needs a 16-lane group's addresses to cover one 128-byte run and
    hands lane ``t`` elements ``t, t+16, t+32, t+48`` of it, so a run laid out as four K
    values of sixteen M rows, K-major, gives each lane four consecutive K of one row. A
    ``k=32`` MFMA wants eight, so this reads the run at ``lds_byte_addr`` for the low four
    and the run ``run_stride_bytes`` later for the high four, then concatenates.
    """
    ptr3 = ir.Type.parse("!llvm.ptr<3>")
    i16x4 = ir.VectorType.get([4], ir.IntegerType.get_signless(16))
    i16x8 = ir.VectorType.get([8], ir.IntegerType.get_signless(16))
    lo = rocdl_ods.ds_read_tr16_b64(i16x4, llvm.inttoptr(ptr3, arith.unwrap(lds_byte_addr)))
    hi = rocdl_ods.ds_read_tr16_b64(i16x4, llvm.inttoptr(ptr3, arith.unwrap(lds_byte_addr + run_stride_bytes)))
    return fx.Vector(llvm.bitcast(result_type, llvm.shufflevector(i16x8, lo, hi, [0, 1, 2, 3, 4, 5, 6, 7])))


def _evict_weight(key, _ref):
    """weakref callback: drop the entry the dead weight was pinning."""
    ent = _WEIGHT_CACHE.get(key)
    if ent is not None and ent[0]() is None:
        del _WEIGHT_CACHE[key]


def _prep_weight(w, k, kt, kh, kw, c):
    """Pack (K, C, T, R, S) -> (K, T*R*S*Cpad), memoized on the source weight."""
    anchor = w._base if w._base is not None else w
    key = w.data_ptr()
    stamp = (w._version, tuple(w.shape), w.stride(), w.dtype)
    ent = _WEIGHT_CACHE.get(key)
    if ent is not None and ent[0]() is anchor and ent[2] == stamp:
        return ent[1]
    cp = _pad_channels(c)
    wsrc = torch.nn.functional.pad(w, (0, 0, 0, 0, 0, 0, 0, cp - c)) if cp != c else w
    wk = wsrc.permute(0, 2, 3, 4, 1).contiguous().reshape(k, kt * kh * kw * cp)
    _WEIGHT_CACHE[key] = (weakref.ref(anchor, functools.partial(_evict_weight, key)), wk, stamp)
    return wk


TR_TILE = 64
TR_VEC = 8
TR_THREADS = 256
_TR_VPL = TR_TILE // TR_VEC
_TR_ITERS = (TR_TILE * TR_TILE) // (TR_VEC * TR_THREADS)
_TR_PAD = 8
_TR_LDS_S = TR_TILE + _TR_PAD

TR_MAX_BIG_S = (0x7FFFFFFF - (TR_TILE - TR_VEC)) // (TR_TILE - 1)


@functools.lru_cache(maxsize=64)
def compile_transpose_ncdhw_ndhwc(n, c, s, cp=None, groups=1):
    """Transpose flat (N, C, S) -> (N, S, Cp) (S == T*H*W), widening C to Cp with zeros.

    ``cp`` is the channel count the GEMM indexes, ``_pad_channels(c // groups) * groups``.
    Writing each group's zero tail here rather than with a separate ``F.pad`` is what
    keeps an unaligned-C input to one launch instead of a fill, a copy and this. Requires
    ``cp % 8 == 0``, which ``_pad_channels`` guarantees; ``cp == c`` is the plain
    transpose and compiles to what it always did.
    """
    cp = c if cp is None else cp
    cg, cgp = c // groups, cp // groups
    PADDED = cp != c
    # Only a spread-out group needs the source channel derived rather than copied, and
    # only then can a tile start past the end of the input -- see the read loop.
    REMAP = PADDED and groups > 1
    grid_s = (s + TR_TILE - 1) // TR_TILE
    grid_c = (cp + TR_TILE - 1) // TR_TILE
    elem_ty = fx.BFloat16
    BIG = (n * cp * s) > 0x7FFFFFFF

    @flyc.kernel(known_block_size=[TR_THREADS, 1, 1])
    def transpose_kernel(out: fx.Tensor, inp: fx.Tensor):
        # max_size: an exact num_records would zero the whole straddling tail read.
        in_rsrc = buffer_ops.create_buffer_resource(inp)
        out_rsrc = buffer_ops.create_buffer_resource(out)
        lds_alloc = fx.SharedAllocator(static=False)
        lds = lds_alloc.allocate(fx.Array[elem_ty, TR_TILE * _TR_LDS_S, 16]).peek()

        class BF16Ty:
            ir_type = elem_ty.ir_type

        tid = fx.thread_idx.x
        s0 = fx.block_idx.x * TR_TILE
        c0 = fx.block_idx.y * TR_TILE
        nb = fx.block_idx.z
        if const_expr(BIG):
            in_base_elem = fx.Index(nb) * fx.Index(c) * fx.Index(s) + fx.Index(c0) * fx.Index(s) + fx.Index(s0)
            in_addr = fx.Int64(buffer_ops.extract_base_index(inp)) + fx.Int64(in_base_elem) * fx.Int64(2)
            in_rsrc = buffer_ops.create_buffer_resource_from_addr(in_addr)
            out_base_elem = fx.Index(nb) * fx.Index(s) * fx.Index(cp) + fx.Index(s0) * fx.Index(cp) + fx.Index(c0)
            out_addr = fx.Int64(buffer_ops.extract_base_index(out)) + fx.Int64(out_base_elem) * fx.Int64(2)
            out_rsrc = buffer_ops.create_buffer_resource_from_addr(out_addr)
        else:
            in_base = nb * c * s
            out_base = nb * s * cp

        _lds_st_ptr_ty = fx.PointerType.get(elem_ty.ir_type, fx.AddressSpace.Shared, TR_VEC * BF16_BYTES)

        def lds_store_vec8(elem_offset, value):
            base = fx.Int64(fx.ptrtoint(lds.ptr)) + fx.Int64(elem_offset * 2)
            fx.ptr_store(value, fx.inttoptr(_lds_st_ptr_ty, base))

        def lds_load_scalar(elem_offset):
            u8 = fx.recast_iter(fx.Uint8, lds.ptr)
            return fx.ptr_load(u8 + fx.Int32(elem_offset * 2), result_type=BF16Ty)

        # Read: coalesced vec8 along contiguous S -> LDS[c_local][s_local]. `cc` is the
        # DESTINATION channel, so a widened group maps it back to the channel it came
        # from and hands LDS zeros for the tail -- the F.pad this kernel absorbs.
        for i in range_constexpr(_TR_ITERS):
            lin = tid + i * TR_THREADS
            rc = lin // _TR_VPL
            sv = (lin % _TR_VPL) * TR_VEC
            cc = c0 + rc
            ss = s0 + sv
            if const_expr(REMAP):
                ci = cc % cgp
                src_c = (cc // cgp) * cg + ci
                in_chan = (ci < cg) & (cc < cp)
            else:
                src_c = cc
                in_chan = cc < c
            valid = in_chan & (ss < s)
            if const_expr(BIG):
                # The rebase leans on src_c == cc, which REMAP breaks; a widened group
                # therefore never reaches the BIG path (see `_ncdhw_to_ndhwc`). It also
                # means c0 < c holds, so the rebased origin stays inside the input.
                g = fx.Int32(rc * s + sv)
            else:
                g = fx.Int32(in_base + src_c * s + ss)
            safe = arith.select(valid, g, fx.Int32(0))
            v = buffer_ops.buffer_load(in_rsrc, safe, vec_width=TR_VEC, dtype=elem_ty)
            if const_expr(PADDED):
                v = fx.Vector(arith.select(in_chan, v, fx.Vector.zeros_like(v)), dtype=elem_ty)
            lds_store_vec8(rc * _TR_LDS_S + sv, v)

        rocdl.s_waitcnt(lgkmcnt=0)
        rocdl.s_barrier()

        for i in range_constexpr(_TR_ITERS):
            lin = tid + i * TR_THREADS
            rs = lin // _TR_VPL
            cv = (lin % _TR_VPL) * TR_VEC
            ss = s0 + rs
            cc = c0 + cv
            scalars = [lds_load_scalar((cv + j) * _TR_LDS_S + rs) for j in range_constexpr(TR_VEC)]
            vv = fx.Vector.from_elements(scalars, dtype=elem_ty)
            valid = (ss < s) & (cc < cp)
            if valid:
                if const_expr(BIG):
                    go = fx.Int32(rs * cp + cv)
                else:
                    go = fx.Int32(out_base + ss * cp + cc)
                buffer_ops.buffer_store(vv, out_rsrc, go)

    @flyc.jit
    def launch_transpose(out: fx.Tensor, inp: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        transpose_kernel(out, inp).launch(
            grid=(grid_s, grid_c, n),
            block=(TR_THREADS, 1, 1),
            stream=stream,
        )

    def _launch(out, inp, stream=None):
        with CompilationContext.compile_hints(CONV_COMPILE_HINTS):
            return launch_transpose(out, inp, stream=_as_stream(stream))

    def _compile(out, inp, stream=None):
        with CompilationContext.compile_hints(CONV_COMPILE_HINTS):
            return flyc.compile(launch_transpose, out, inp, _as_stream(stream))

    _launch.compile = _compile
    return _launch


def _pad_channels_ncdhw(x, cp, groups):
    """Widen each group's channel block to ``cp // groups`` with zeros, in NCDHW."""
    n, c, d, h, w = x.shape
    cg, cgp = c // groups, cp // groups
    x = torch.nn.functional.pad(x.reshape(n, groups, cg, d, h, w), (0, 0, 0, 0, 0, 0, 0, cgp - cg))
    return x.reshape(n, cp, d, h, w)


def _ncdhw_to_ndhwc(x, stream, cp=None, groups=1):
    """Fast NCDHW->NDHWC via the tiled transpose kernel; falls back to torch.

    ``cp`` asks for each group's channels to be widened to ``cp // groups`` on the way
    out. Folding that into the transpose is what keeps an unaligned-C input to a single
    launch; only the paths that cannot fuse it fall back to a separate ``F.pad``.
    """
    n, c, t, h, w = x.shape
    cp = c if cp is None else cp
    s = t * h * w
    big = n * cp * s > 0x7FFFFFFF
    if (
        x.is_contiguous()
        and x.dtype == torch.bfloat16
        and cp % TR_VEC == 0
        # A rebased tile addresses its source by its own channel row, which tracks the
        # destination row only while the groups are not being spread apart.
        and (cp == c or groups == 1 or not big)
        and not (big and max(s, cp) > TR_MAX_BIG_S)
    ):
        out = torch.empty((n, t, h, w, cp), device=x.device, dtype=x.dtype)
        exe = compile_transpose_ncdhw_ndhwc(n, c, s, cp, groups)
        _dispatch(exe, out, x, stream=torch.cuda.current_stream() if stream is None else stream)
        return out
    if cp != c:
        # Widening it separately is the slow way round, but the transpose itself may
        # still be the kernel's to do.
        return _ncdhw_to_ndhwc(_pad_channels_ncdhw(x, cp, groups), stream)
    return x.permute(0, 2, 3, 4, 1).contiguous()


@functools.lru_cache(maxsize=256)
def compile_conv3d_implicit(
    n,
    c,
    d,
    h,
    w,
    k,
    kt,
    kh,
    kw,
    st,
    sh,
    sw,
    pt,
    ph,
    pw,
    dt=1,
    dh=1,
    dw=1,
    pad_mode="zeros",
    has_bias=False,
    splitk=1,
    tile=DEFAULT_TILE,
    wgm=1,
    groups=1,
    out_ndhwc=False,
    nchw_a=False,
    pad_hi=None,
):
    TILE_M, TILE_N, WAVE_M, WAVE_N = tile
    BLOCK_THREADS = WAVE_M * WAVE_N * WARP_SIZE
    # Per-wave MFMA grid (flat acc[mi * MI_N + ni]); WARP_M/N is the per-wave tile span.
    MI_M = TILE_M // WAVE_M // MFMA_M
    MI_N = TILE_N // WAVE_N // MFMA_N
    N_ACC = MI_M * MI_N
    WARP_M = MI_M * MFMA_M
    WARP_N = MI_N * MFMA_N
    BLOCK_VECS = LDG_VEC * BLOCK_THREADS
    LDG_A_COUNT = TILE_M * TILE_K // BLOCK_VECS
    LDG_B_COUNT = TILE_N * TILE_K // BLOCK_VECS

    # `c` is the padded TOTAL channel count and stays the NDHWC row stride. CGP is the
    # per-group channel count and is what the GEMM K axis decomposes against; the two
    # coincide only when groups == 1.
    CGP = c // groups
    KG = k // groups

    assert TILE_K == 32
    assert TILE_M % (WAVE_M * MFMA_M) == 0, f"TILE_M={TILE_M} not divisible by WAVE_M*16"
    assert TILE_N % (WAVE_N * MFMA_N) == 0, f"TILE_N={TILE_N} not divisible by WAVE_N*16"
    assert (TILE_M * TILE_K) % BLOCK_VECS == 0, f"A tile {TILE_M}x{TILE_K} not a multiple of {BLOCK_VECS} vecs"
    assert (TILE_N * TILE_K) % BLOCK_VECS == 0, f"B tile {TILE_N}x{TILE_K} not a multiple of {BLOCK_VECS} vecs"
    assert LDG_A_COUNT >= 1 and LDG_B_COUNT >= 1
    assert c % groups == 0, f"c={c} not divisible by groups={groups}"
    assert k % groups == 0, f"k={k} not divisible by groups={groups}"
    assert CGP % LDG_VEC == 0, f"c/groups={CGP} must be a multiple of LDG_VEC={LDG_VEC}; use _conv3d_impl to pad"
    assert BLOCK_THREADS <= 1024, f"BLOCK_THREADS={BLOCK_THREADS} exceeds 1024"

    # `pt/ph/pw` are the LOW pads and are the only ones the gather needs: it derives the
    # input coordinate as `out * stride - pad_lo` and then either range-masks it (zeros) or
    # folds it back into [0, ext) (every other mode). The high pad shows up here and
    # nowhere else, which is why an asymmetric pad costs a different `do` rather than a
    # padded copy of the activation.
    qt, qh, qw = (pt, ph, pw) if pad_hi is None else pad_hi

    # Dilation only stretches the filter's footprint; the K axis (CRS) is unchanged.
    do = (d + pt + qt - (dt * (kt - 1) + 1)) // st + 1
    ho = (h + ph + qh - (dh * (kh - 1) + 1)) // sh + 1
    wo = (w + pw + qw - (dw * (kw - 1) + 1)) // sw + 1
    dhw = do * ho * wo
    hw_o = ho * wo
    npq = n * dhw
    crs = CGP * kt * kh * kw
    k_tiles = (crs + TILE_K - 1) // TILE_K

    BIG_IN = (n * c * d * h * w) > 0x7FFFFFFF
    BIG_OUT = (n * k * do * ho * wo * BF16_BYTES) > 0x7FFFFFFF

    X_BYTES = n * c * d * h * w * BF16_BYTES
    W_BYTES = k * crs * BF16_BYTES
    OOB_SENTINEL_ELEM = 0x7FFFFF80  # *2 = 0xFFFFFF00 bytes (~4.2950 GB), just under 2^32
    OOB_SENTINEL_BYTES = OOB_SENTINEL_ELEM * BF16_BYTES
    BIG_IN_NR = 0x80000000  # 2 GB num_records for the rebased BIG_IN resource
    assert W_BYTES < OOB_SENTINEL_BYTES, f"weight {W_BYTES}B exceeds limit {OOB_SENTINEL_BYTES}B"
    assert X_BYTES < OOB_SENTINEL_BYTES or BIG_IN, f"input {X_BYTES}B exceeds limit"
    BIG_IN_N1 = BIG_IN and n == 1
    BIG_IN_NM = BIG_IN and n > 1

    _t_aligned = BIG_IN_N1 and hw_o % TILE_M == 0
    if BIG_IN_N1:
        _ot_span = (TILE_M - 1) // hw_o + (1 if _t_aligned else 2)
        _t_span = min(d - 1, (_ot_span - 1) * st + dt * (kt - 1))
        _h_span = min(h - 1, ((TILE_M - 1) // wo + 1) * sh + dh * (kh - 1)) if _t_aligned else h - 1
        _span = (((_t_span * h + _h_span) * w + (w - 1)) * c + c) * BF16_BYTES
        assert _span <= BIG_IN_NR, (
            f"input sample too large for the 32-bit gather: a {TILE_M}-row tile reaches "
            f"{_span / 2**30:.2f} GiB from its rebased origin, past the "
            f"{BIG_IN_NR / 2**30:.0f} GiB the buffer descriptor addresses. Split the batch "
            f"over N, or pass a narrower tile=(TILE_M, ...)."
        )

    assert pad_mode in PADDING_MODES, f"pad_mode must be one of {PADDING_MODES}, got {pad_mode!r}"
    assert pad_mode == "zeros" or not BIG_IN, "non-zero pad_mode requires the non-BIG_IN address path"
    X_SAMPLE_ELEMS = c * d * h * w

    tiles_per_group = (KG + TILE_N - 1) // TILE_N
    n_tail = KG % TILE_N != 0
    grid_n = groups * tiles_per_group

    splitk = max(1, min(splitk, k_tiles))
    tiles_per_split = k_tiles // splitk
    use_splitk = splitk > 1

    Y_BYTES = npq * k * (4 if use_splitk else BF16_BYTES)

    assert (
        not use_splitk or npq * k * 4 <= SPLITK_MAX_STAGING_BYTES
    ), f"split-K staging {npq * k * 4}B exceeds the {SPLITK_MAX_STAGING_BYTES}B buffer window"

    PIPE_STAGES = 2 * TILES_PER_BARRIER

    LDS_A_SIZE = PIPE_STAGES * TILE_M * TILE_K
    LDS_B_SIZE = PIPE_STAGES * TILE_N * TILE_K

    grid_m = (npq + TILE_M - 1) // TILE_M

    MAX_GRID_X = 0xFFFFFFFF // BLOCK_THREADS
    MAX_GRID_YZ = 65535
    grid_x = min(grid_m, MAX_GRID_X)
    m_chunks = (grid_m + grid_x - 1) // grid_x

    assert grid_n <= MAX_GRID_YZ, f"grid.y = {grid_n} exceeds the {MAX_GRID_YZ}-block limit"
    assert (
        m_chunks * splitk <= MAX_GRID_YZ
    ), f"grid.z = {m_chunks} M-chunks x {splitk} splits exceeds the {MAX_GRID_YZ}-block limit"

    WGM = 1 if m_chunks > 1 else max(1, int(wgm))
    elem_ty = fx.BFloat16
    mfma_fn = rocdl.mfma_f32_16x16x32_bf16
    temporal_only_fast = (
        kh == 1
        and kw == 1
        and st == 1
        and sh == 1
        and sw == 1
        and ph == 0
        and pw == 0
        and do == d
        and ho == h
        and wo == w
    )

    # NCDHW A tile: run `r` holds K values 4*(r // A_RUNS_M) .. +3 of M rows
    # 16*(r % A_RUNS_M) .. +15, K-major, and ds_read_tr16_b64 turns one run into four
    # lanes' worth of MFMA A. See the notes at NCHW_A_MIN_TILE_M.
    NCHW_A = bool(nchw_a)
    if NCHW_A:
        assert TILE_M % NCHW_A_MIN_TILE_M == 0, f"NCDHW A tile needs TILE_M % {NCHW_A_MIN_TILE_M} == 0"
        assert TILE_K == MFMA_A_VALUES * (WARP_SIZE // MFMA_M), "NCDHW A tile assumes one K octet per lane group"
    A_RUNS_M = TILE_M // MFMA_M
    A_RUN_ELEMS = MFMA_M * (MFMA_A_VALUES // 2)  # 64: 16 M rows x 4 K
    A_TR_STRIDE_BYTES = A_RUNS_M * A_RUN_ELEMS * BF16_BYTES

    # --- staging the epilogue through LDS, so the NCDHW store is contiguous ---
    #
    # An MFMA C fragment hands a lane 4 consecutive npq at one channel, and the 16 lanes of
    # a group sit at 16 different channels, so one NCDHW store instruction touches 16
    # disjoint 32-byte segments where the same bytes laid out along npq would take four
    # cache lines. Bouncing the tile through LDS fixes that at no cost in space -- the A
    # staging buffer is dead by the epilogue -- by re-landing one MFMA column block's
    # channels as whole npq runs: every lane then stores 16 bytes and 32 consecutive lanes
    # cover 512 contiguous bytes of one channel.
    EPI_VEC = 8  # bf16 per thread per global store: 16B, the widest buffer_store
    EPI_ROWS = WAVE_N * MFMA_N  # channels staged per round: one MFMA column block per wave
    # Row pad, in elements. 8 keeps the row stride 16B-aligned for the vec8 read-back and
    # spreads a wave's b64 writes evenly over all 32 banks; 0 would put every row on bank 0.
    EPI_PAD = 8
    EPI_STRIDE = TILE_M + EPI_PAD
    EPI_TOTAL = EPI_ROWS * TILE_M // EPI_VEC  # vec8 slots to drain per round
    LDS_EPILOGUE = (
        not out_ndhwc  # NDHWC output is already contiguous over the GEMM's own index space
        and not use_splitk  # split-K stores are f32 atomics, not plain stores
        and not BIG_OUT
        # dhw % EPI_VEC buys three things at once: a channel's rows start 16B-aligned, a
        # vec8 span never straddles a batch boundary, and npq % EPI_VEC == 0, so the tail
        # check stays uniform across a span.
        and dhw % EPI_VEC == 0
        and TILE_M % EPI_VEC == 0
        and EPI_ROWS * EPI_STRIDE <= LDS_A_SIZE
        and EPI_TOTAL % BLOCK_THREADS == 0
    )
    EPI_ITERS = EPI_TOTAL // BLOCK_THREADS if LDS_EPILOGUE else 0

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def conv3d_implicit_kernel(y: fx.Tensor, x: fx.Tensor, weight: fx.Tensor, bias: fx.Tensor):
        # The im2col gather addresses its source by a flat element index, so the DMA source
        # is a 1-D view over the buffer rather than the tensor's own n-D layout: dividing
        # that by a 1-element tile makes slice(src, (None, off)) exactly element `off`,
        # with no coordinate decomposition. `elems` only shapes the view -- the sentinel
        # offset that masks a tap deliberately points past it, and num_records (not the
        # layout) is what turns that into a zero-fill.
        def _dma_src(ptr, elems, num_records_bytes):
            buf = fx.rocdl.make_buffer_ptr(ptr, num_records_bytes=num_records_bytes)
            return fx.logical_divide(fx.make_view(buf, fx.make_layout(elems, 1)), fx.make_layout(1, 1))

        def _x_rebased(off_elems):
            # BIG_IN moves the descriptor base to the block's own origin so a 32-bit
            # voffset still reaches the tile; num_records bounds it at 2 GB from there.
            ptr = fx.add_offset(fx.get_iter(x), fx.make_int_tuple(off_elems))
            return _dma_src(ptr, BIG_IN_NR // BF16_BYTES, BIG_IN_NR)

        w_src = _dma_src(fx.get_iter(weight), W_BYTES // BF16_BYTES, W_BYTES)
        if const_expr(not BIG_IN):
            x_src = _dma_src(fx.get_iter(x), X_BYTES // BF16_BYTES, X_BYTES)
        y_rsrc = buffer_ops.create_buffer_resource(y, num_records_bytes=Y_BYTES)
        if const_expr(has_bias):
            bias_rsrc = buffer_ops.create_buffer_resource(bias)

        lds_alloc = fx.SharedAllocator(static=False)
        a_lds = lds_alloc.allocate(fx.Array[elem_ty, LDS_A_SIZE, 16]).peek()
        b_lds = lds_alloc.allocate(fx.Array[elem_ty, LDS_B_SIZE, 16]).peek()

        tid = fx.thread_idx.x
        if const_expr(m_chunks > 1):
            m_chunk = fx.Index(fx.block_idx.z) % fx.Index(m_chunks)
            m_offset = (fx.Index(fx.block_idx.x) + m_chunk * fx.Index(grid_x)) * TILE_M
            n_tile = fx.block_idx.y
        elif const_expr(WGM > 1):
            pid = fx.Index(fx.block_idx.x) + fx.Index(fx.block_idx.y) * fx.Index(grid_m)
            blocks_per_swizzle = fx.Index(WGM * grid_n)
            swizzle_id = pid // blocks_per_swizzle
            first_m = swizzle_id * fx.Index(WGM)
            swizzle_rows = fx.Index(grid_m) - first_m
            swizzle_rows = fx.Index(arith.select(swizzle_rows < fx.Index(WGM), swizzle_rows, fx.Index(WGM)))
            local = pid % blocks_per_swizzle
            m_offset = fx.Index(first_m + (local % swizzle_rows)) * TILE_M
            n_tile = fx.Index(local // swizzle_rows)
        else:
            m_offset = fx.block_idx.x * TILE_M
            n_tile = fx.block_idx.y

        if const_expr(groups > 1):
            gi = n_tile // tiles_per_group
            n_local = (n_tile % tiles_per_group) * TILE_N
            n_offset = gi * KG + n_local
            ch_base = gi * CGP
        else:
            n_offset = n_tile * TILE_N
            n_local = n_offset
        if const_expr(use_splitk):
            if const_expr(m_chunks > 1):
                split_idx = fx.Index(fx.block_idx.z) // fx.Index(m_chunks)
            else:
                split_idx = fx.Index(fx.block_idx.z)
            k_off = split_idx * (tiles_per_split * TILE_K)
        else:
            k_off = 0

        if const_expr(BIG_IN_N1):
            nbase = m_offset // dhw
            rem0 = m_offset % dhw
            ot_base0 = rem0 // hw_o

            base_t = ot_base0 * fx.Index(st) - fx.Index(pt)
            base_t = arith.select(base_t < fx.Index(0), fx.Index(0), base_t)
            if const_expr(_t_aligned):
                oh_base0 = (rem0 % hw_o) // wo
                base_h = oh_base0 * fx.Index(sh) - fx.Index(ph)
                base_h = arith.select(base_h < fx.Index(0), fx.Index(0), base_h)
            else:
                base_h = fx.Index(0)
            x_base_elem = ((nbase * fx.Index(d) + base_t) * fx.Index(h) + base_h) * fx.Index(w) * fx.Index(c)
            x_src = _x_rebased(fx.Int64(x_base_elem))

        wid = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        wave_m = wid // WAVE_N
        wave_n = wid % WAVE_N

        lane_m = lane % MFMA_M
        lane_n = lane % MFMA_N
        lane_k_a = lane // MFMA_M * MFMA_A_VALUES
        lane_k_b = lane // MFMA_N * MFMA_B_VALUES
        c_m_vec = lane // MFMA_N * MFMA_C_VALUES
        c_n = lane % MFMA_N

        Vec = fx.Vector

        class Vec8Ty:
            ir_type = Vec.make_type(8, elem_ty)

        acc0 = Vec.filled(MFMA_C_VALUES, 0.0, fx.Float32)
        acc = [acc0 for _ in range_constexpr(N_ACC)]

        def barrier(vmcnt=0, lgkmcnt=None):
            rocdl.s_waitcnt(vmcnt=vmcnt, lgkmcnt=lgkmcnt)
            rocdl.s_barrier()

        def lds_load_vec8(lds_array, elem_offset):
            u8_ptr = fx.recast_iter(fx.Uint8, lds_array.ptr)
            return fx.ptr_load(u8_ptr + fx.Int32(elem_offset * 2), result_type=Vec8Ty)

        def a_lds_off(stage, row, col):
            return (fx.Index(stage) * TILE_M + row) * TILE_K + col

        def b_lds_off(stage, row, col):
            return (fx.Index(stage) * TILE_N + row) * TILE_K + col

        def in_range(v, hi):
            return (v >= 0) & (v < fx.Index(hi))

        def dil(tap, factor):
            scaled = tap * factor if const_expr(factor != 1) else tap
            return scaled

        def pad_coord(v, ext, pad):
            """Tap coordinate -> in-bounds input coordinate; returns (coord, mask).

            "zeros" leaves the coordinate alone and returns a range mask, which the
            caller folds into the OOB-sentinel routing so the load reads as zero. Every
            other mode resolves the coordinate into [0, ext) instead and returns no mask
            """
            if const_expr(pad_mode == "zeros"):
                return v, in_range(v, ext)
            u = v + fx.Index(pad)
            low = u < fx.Index(pad)  # v < 0
            high = u >= fx.Index(pad + ext)  # v >= ext
            mid = u - fx.Index(pad)  # v, where in range
            if const_expr(pad_mode == "replicate"):
                r = arith.select(high, fx.Index(ext - 1), mid)
                r = arith.select(low, fx.Index(0), r)
            elif const_expr(pad_mode == "reflect"):
                # [a b c d e] pad 2 -> [c b a b c d e d c]: -v near, 2*(ext-1) - v far.
                r = arith.select(high, fx.Index(2 * (ext - 1) + pad) - u, mid)
                r = arith.select(low, fx.Index(pad) - u, r)
            else:  # circular: v + ext near, v - ext far
                r = arith.select(high, u - fx.Index(pad + ext), mid)
                r = arith.select(low, u + fx.Index(ext - pad), r)
            return fx.Index(r), None

        def gather_valid(base, *masks):
            for m in masks:
                if const_expr(m is not None):
                    base = base & m
            return base

        # ---- Per-thread row decomposition (loop-invariant across K) ----
        _row_dec = []  # per-i tuple of precomputed row terms
        for i in range_constexpr(LDG_A_COUNT):
            linear = (tid + i * BLOCK_THREADS) * LDG_VEC
            if const_expr(NCHW_A):
                # Invert the run layout: this thread's LDG_VEC contiguous LDS elements sit
                # at `pos` inside run `run`, which pins one K and LDG_VEC consecutive rows.
                run = linear // A_RUN_ELEMS
                pos = linear % A_RUN_ELEMS
                local_k = (run // A_RUNS_M) * (MFMA_A_VALUES // 2) + pos // MFMA_M
                local_m = (run % A_RUNS_M) * MFMA_M + pos % MFMA_M
                row = m_offset + local_m
                row_valid = row < fx.Index(npq)
                # NCDHW element offset of (row, channel 0); the channel adds cc * dhw.
                row_base = row if const_expr(n == 1) else (row // dhw) * fx.Index(c * dhw) + row % dhw
                _row_dec.append((local_k, row_base, row_valid))
                continue
            local_m = linear // TILE_K
            local_k = linear % TILE_K
            row = m_offset + local_m
            row_valid = row < fx.Index(npq)
            if const_expr(temporal_only_fast):
                out_t = (row // hw_o) % d
                _row_dec.append((local_k, row, row_valid, out_t))
            else:
                n_idx = row // dhw
                rem = row % dhw
                ot = rem // hw_o
                rem2 = rem % hw_o
                oh = rem2 // wo
                ow = rem2 % wo
                in_t0 = ot * st - pt
                in_h0 = oh * sh - ph
                in_w0 = ow * sw - pw
                if const_expr(BIG_IN_N1):
                    di = n_idx - nbase
                    _row_dec.append((local_k, row_valid, di, in_t0, in_h0, in_w0))
                elif const_expr(BIG_IN_NM):
                    _row_dec.append((local_k, row_valid, n_idx, in_t0, in_h0, in_w0))
                else:
                    _row_dec.append((local_k, row_valid, n_idx, in_t0, in_h0, in_w0))

        SCALAR_K = CGP % TILE_K == 0

        def _a_addr_nchw(i, kbase_i):
            """NCDHW gather: a 1x1 filter's K axis is the input channel, nothing else."""
            local_k, row_base, row_valid = _row_dec[i]
            k_abs = kbase_i + fx.Index(local_k)
            cc = (ch_base + k_abs) if const_expr(groups > 1) else k_abs
            valid = row_valid & (k_abs < fx.Index(crs))
            return fx.Int32(row_base + cc * fx.Index(dhw)), valid

        # ---- 3D im2col address math ----
        # The K axis decomposes against CGP (per-group channels) while every g_off below
        # keeps `c` (padded total channels) as the NDHWC row stride. `cc` is the absolute
        # input channel: the group base plus the offset within the group.
        def _a_addr(i, kbase_i, cc_base, ckk_base):
            dec = _row_dec[i]
            local_k = dec[0]
            k_abs = kbase_i + fx.Index(local_k)
            if const_expr(SCALAR_K):
                cc = cc_base + fx.Index(local_k)  # cc_base already carries ch_base
            else:
                cc = k_abs % CGP
                if const_expr(groups > 1):
                    cc = ch_base + cc
            k_valid = k_abs < fx.Index(crs)
            if const_expr(temporal_only_fast):
                _, row, row_valid, out_t = dec
                kt_i = ckk_base if const_expr(SCALAR_K) else k_abs // CGP
                temporal_delta = dil(kt_i, dt) - pt
                in_t, m_t = pad_coord(out_t + temporal_delta, d, pt)
                valid = gather_valid(row_valid & k_valid, m_t)

                delta = temporal_delta if const_expr(pad_mode == "zeros") else (in_t - out_t)
                if const_expr(BIG_IN_N1):
                    # base_h rides in x_base_elem whenever _t_aligned, so it has to come
                    # back out here as well -- kh == kw == 1 keeps the row otherwise flat.
                    rebased = (row + delta * hw_o) - (fx.Index(nbase) * dhw + base_t * hw_o)
                    g_off = rebased * c + cc - base_h * fx.Index(w * c)
                else:
                    g_off = (row + delta * hw_o) * c + cc
            else:
                ckk = ckk_base if const_expr(SCALAR_K) else k_abs // CGP
                kw_i = ckk % kw
                ckk2 = ckk // kw
                kh_i = ckk2 % kh
                kt_i = ckk2 // kh
                if const_expr(BIG_IN_N1):
                    _, row_valid, di, in_t0, in_h0, in_w0 = dec
                    in_t, m_t = pad_coord(in_t0 + dil(kt_i, dt), d, pt)
                    in_h, m_h = pad_coord(in_h0 + dil(kh_i, dh), h, ph)
                    in_w, m_w = pad_coord(in_w0 + dil(kw_i, dw), w, pw)
                    valid = gather_valid(row_valid & k_valid, m_t, m_h, m_w)
                    g_off = (((di * d + (in_t - base_t)) * h + (in_h - base_h)) * w + in_w) * c + cc
                elif const_expr(BIG_IN_NM):
                    _, row_valid, n_idx, in_t0, in_h0, in_w0 = dec
                    in_t, m_t = pad_coord(in_t0 + dil(kt_i, dt), d, pt)
                    in_h, m_h = pad_coord(in_h0 + dil(kh_i, dh), h, ph)
                    in_w, m_w = pad_coord(in_w0 + dil(kw_i, dw), w, pw)
                    valid = gather_valid(row_valid & k_valid, m_t, m_h, m_w)
                    g_off = ((in_t * h + in_h) * w + in_w) * c + cc
                    return fx.Int32(g_off), valid, n_idx
                else:
                    _, row_valid, n_idx, in_t0, in_h0, in_w0 = dec
                    in_t, m_t = pad_coord(in_t0 + dil(kt_i, dt), d, pt)
                    in_h, m_h = pad_coord(in_h0 + dil(kh_i, dh), h, ph)
                    in_w, m_w = pad_coord(in_w0 + dil(kw_i, dw), w, pw)
                    valid = gather_valid(row_valid & k_valid, m_t, m_h, m_w)
                    g_off = (((n_idx * d + in_t) * h + in_h) * w + in_w) * c + cc
            return fx.Int32(g_off), valid

        def _b_addr(i, k_base):
            linear = (tid + i * BLOCK_THREADS) * LDG_VEC
            local_n = linear // TILE_K
            local_k = linear % TILE_K
            col = n_offset + fx.Index(local_n)
            g_off = fx.Int32(col * crs + (fx.Index(k_base) + fx.Index(local_k)))
            # Tail is per group: the N grid is over-provisioned to groups*tiles_per_group.
            col_valid = ((n_local + fx.Index(local_n)) < fx.Index(KG)) if const_expr(n_tail) else None
            return g_off, col_valid

        # ---- global -> LDS DMA copy, masking via OOB routing ----
        DMA_BYTES = LDG_VEC * BF16_BYTES  # 16
        OOB_ELEM = fx.Int32(OOB_SENTINEL_ELEM)

        _lds_dma_ptr_ty = fx.PointerType.get(elem_ty.ir_type, fx.AddressSpace.Shared, DMA_BYTES)

        def sgpr(x):
            # Hoist a wave-uniform value into an SGPR (readfirstlane).
            return fx.Int64(rocdl.readfirstlane(T.i64, fx.Int64(x)))

        _dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), DMA_BYTES * 8)

        def _lds_dma_ptr(lds_array, stage_tile, i):
            # buffer_load_lds takes one wave-uniform LDS base and fans the wave's lanes
            # out from it, so the lane-0 address is the base the whole wave writes from.
            off_elems = fx.Index(stage_tile) + (fx.Index(tid) + fx.Index(i * BLOCK_THREADS)) * fx.Index(LDG_VEC)
            base_bytes = off_elems * fx.Index(BF16_BYTES)
            addr = fx.Int64(fx.ptrtoint(lds_array.ptr)) + fx.Int64(base_bytes)
            return fx.make_view(fx.inttoptr(_lds_dma_ptr_ty, sgpr(addr)), fx.make_layout(1, 1))

        def _dma_to_lds(src, dst, voff_elem):
            fx.copy(_dma_atom, fx.slice(src, (None, voff_elem)), dst)

        def _load_a(stage, k_base):
            kbase_i = fx.Index(k_base)
            stage_tile = fx.Index(stage) * TILE_M * TILE_K
            if const_expr(NCHW_A):
                for i in range_constexpr(LDG_A_COUNT):
                    g_off_i, valid = _a_addr_nchw(i, kbase_i)
                    voff = fx.Int32(arith.select(valid, g_off_i, OOB_ELEM))
                    _dma_to_lds(x_src, _lds_dma_ptr(a_lds, stage_tile, i), voff)
                return
            cc_base = ckk_base = None
            if const_expr(SCALAR_K):
                cc_base = kbase_i % CGP
                if const_expr(groups > 1):
                    cc_base = ch_base + cc_base
                ckk_base = kbase_i // CGP
            for i in range_constexpr(LDG_A_COUNT):
                if const_expr(BIG_IN_NM):
                    addr_ret = _a_addr(i, kbase_i, cc_base, ckk_base)
                    g_off_i, valid, n_idx_i = addr_ret
                    x_src_i = _x_rebased(fx.Int64(n_idx_i) * fx.Int64(X_SAMPLE_ELEMS))
                    voff = fx.Int32(arith.select(valid, g_off_i, OOB_ELEM))
                    _dma_to_lds(x_src_i, _lds_dma_ptr(a_lds, stage_tile, i), voff)
                else:
                    g_off_i, valid = _a_addr(i, kbase_i, cc_base, ckk_base)
                    voff = fx.Int32(arith.select(valid, g_off_i, OOB_ELEM))
                    _dma_to_lds(x_src, _lds_dma_ptr(a_lds, stage_tile, i), voff)

        def _load_b(stage, k_base):
            stage_tile = fx.Index(stage) * TILE_N * TILE_K
            for i in range_constexpr(LDG_B_COUNT):
                g_off, col_valid = _b_addr(i, k_base)
                if const_expr(n_tail):
                    voff = fx.Int32(arith.select(col_valid, g_off, OOB_ELEM))
                else:
                    voff = g_off
                _dma_to_lds(w_src, _lds_dma_ptr(b_lds, stage_tile, i), voff)

        # ---- single-vec ds_read (LDS -> register), indexed by per-wave MFMA row ----
        if const_expr(NCHW_A):
            # Address of run (lane_k_a // 4, wave_m * MI_M) at this lane, in bytes. The
            # per-mi run and the second half of the K octet are constant strides off it.
            a_tr_base = fx.Int32(fx.ptrtoint(a_lds.ptr)) + fx.Int32(
                (
                    (lane_k_a // (MFMA_A_VALUES // 2)) * fx.Index(A_RUNS_M * A_RUN_ELEMS)
                    + wave_m * fx.Index(MI_M * A_RUN_ELEMS)
                    + lane_m * fx.Index(MFMA_A_VALUES // 2)
                )
                * BF16_BYTES
            )

        def read_a_vec(stage, mi):
            if const_expr(NCHW_A):
                addr = a_tr_base + fx.Int32((stage * TILE_M * TILE_K + mi * A_RUN_ELEMS) * BF16_BYTES)
                return _lds_read_transpose_frag(Vec8Ty.ir_type, addr, A_TR_STRIDE_BYTES)
            a_row = wave_m * WARP_M + mi * MFMA_M + lane_m
            return lds_load_vec8(a_lds, a_lds_off(stage, fx.Index(a_row), fx.Index(lane_k_a)))

        def read_b_vec(stage, ni):
            b_row = wave_n * WARP_N + ni * MFMA_N + lane_n
            return lds_load_vec8(b_lds, b_lds_off(stage, fx.Index(b_row), fx.Index(lane_k_b)))

        def mfma_one(a_frag, b_frag, c_frag):
            return mfma_fn(
                T.vec(MFMA_C_VALUES, T.f32),
                [a_frag, b_frag, c_frag, 0, 0, 0],
            )

        def read_a_frags(stage):
            frags = [read_a_vec(stage, mi) for mi in range_constexpr(MI_M)]
            # The transposed path spends two ds_read_tr per fragment, not one ds_read_b128.
            rocdl.sched_dsrd(2 * MI_M if const_expr(NCHW_A) else MI_M)
            return frags

        def read_b_frags(stage):
            frags = [read_b_vec(stage, ni) for ni in range_constexpr(MI_N)]
            rocdl.sched_dsrd(MI_N)
            return frags

        def do_compute(acc_values, a_frag_values, b_frag_values):
            rocdl.s_setprio(1)
            for mi in range_constexpr(MI_M):
                for ni in range_constexpr(MI_N):
                    idx = mi * MI_N + ni
                    acc_values[idx] = mfma_one(a_frag_values[mi], b_frag_values[ni], acc_values[idx])
                rocdl.sched_mfma(MI_N)
            rocdl.s_setprio(0)
            return acc_values

        # global->LDS software pipeline
        # ---- prologue: fill the pipeline with the first PREFETCH tiles' DMAs ----
        PREFETCH = TILES_PER_BARRIER
        for s in range_constexpr(PREFETCH):
            if const_expr(s < tiles_per_split):
                _load_a(s, k_off + s * TILE_K)
                _load_b(s, k_off + s * TILE_K)

        # ---- main loop
        for kt_idx in range_constexpr(0, tiles_per_split, TILES_PER_BARRIER):
            batch = range_constexpr(kt_idx, min(kt_idx + TILES_PER_BARRIER, tiles_per_split))

            barrier(vmcnt=0, lgkmcnt=0)
            a_frags = [read_a_frags(kt % PIPE_STAGES) for kt in batch]
            b_frags = [read_b_frags(kt % PIPE_STAGES) for kt in batch]
            issued = 0
            for kt in batch:
                nxt = kt + PREFETCH
                if const_expr(nxt < tiles_per_split):
                    _load_a(nxt % PIPE_STAGES, k_off + nxt * TILE_K)
                    _load_b(nxt % PIPE_STAGES, k_off + nxt * TILE_K)
                    issued += LDG_A_COUNT + LDG_B_COUNT
            if const_expr(issued):
                rocdl.sched_vmem(issued)
            for j in range_constexpr(len(batch)):
                acc = do_compute(acc, a_frags[j], b_frags[j])

        _row_chk = (npq % TILE_M != 0) or (grid_x * m_chunks > grid_m)
        _need_chk = _row_chk or n_tail

        _vec_store = (n == 1) and (not use_splitk) and (dhw % MFMA_C_VALUES == 0) and (not BIG_OUT) and (not out_ndhwc)

        if const_expr(BIG_OUT):
            y_elem_base = fx.Int64(buffer_ops.extract_base_index(y))

        _big_st_ptr_ty = fx.PointerType.get(elem_ty.ir_type, fx.AddressSpace.Global, BF16_BYTES)

        def _big_store(off_nk_i64, value):
            addr = y_elem_base + off_nk_i64 * fx.Int64(BF16_BYTES)
            fx.ptr_store(value, fx.inttoptr(_big_st_ptr_ty, addr))

        def _valid_raw(row, col_loc):
            if const_expr(_row_chk and n_tail):
                return arith.andi(row < fx.Index(npq), col_loc < fx.Index(KG))
            if const_expr(_row_chk):
                v = row < fx.Index(npq)
                return arith.andi(v, v)
            v = col_loc < fx.Index(KG)
            return arith.andi(v, v)

        _route_store = _need_chk and not use_splitk and not BIG_OUT

        def _route(off, row, col_loc):
            if const_expr(not _route_store):
                return off
            return fx.Int32(arith.select(_valid_raw(row, col_loc), fx.Int32(off), OOB_ELEM))

        def _cols(ni):
            """Global out-channel for MFMA column block ni, and its index within the group."""
            col_off = fx.Index(wave_n * WARP_N + ni * MFMA_N + c_n)
            col = n_offset + col_off
            return col, ((n_local + col_off) if const_expr(groups > 1) else col)

        def _out_off(row, col):
            """Element offset of (npq row, out-channel col) in the NCDHW output."""
            if const_expr(n == 1):
                return col * dhw + row
            return (row // dhw) * (k * dhw) + col * dhw + row % dhw

        _epi_ptr_ty = fx.PointerType.get(elem_ty.ir_type, fx.AddressSpace.Shared, MFMA_C_VALUES * BF16_BYTES)

        def store_acc_lds():
            """Drain the accumulator through the (now dead) A staging buffer, one MFMA
            column block at a time, so each global store covers one channel's npq run."""
            # The tile lands where the last K tile's A did, so its DMAs and ds_reads have to
            # have retired before the first write.
            barrier(vmcnt=0, lgkmcnt=0)
            lds_row = fx.Index(wave_n * MFMA_N + c_n)
            _slots_per_row = TILE_M // EPI_VEC  # vec8 slots covering one staged channel

            for ni in range_constexpr(MI_N):
                if const_expr(has_bias):
                    col, col_loc = _cols(ni)
                    col_i = fx.Int32(col)  # bias is indexed by the global out-channel
                    if const_expr(n_tail):
                        col_i = arith.select(col_loc < fx.Index(KG), col_i, fx.Int32(0))
                    bias_val = fx.Float32(buffer_ops.buffer_load(bias_rsrc, col_i, vec_width=1, dtype=fx.Float32))
                if const_expr(ni > 0):
                    # The previous round's readers have to be done before their slots move.
                    barrier(vmcnt=None, lgkmcnt=0)

                for mi in range_constexpr(MI_M):
                    a = Vec(acc[mi * MI_N + ni])
                    vals = []
                    for i in range_constexpr(MFMA_C_VALUES):
                        cval = (a[i] + bias_val) if const_expr(has_bias) else a[i]
                        vals.append(cval.to(elem_ty))
                    lds_m = fx.Index(wave_m * WARP_M + mi * MFMA_M + c_m_vec)
                    addr = fx.Int64(fx.ptrtoint(a_lds.ptr)) + fx.Int64(lds_row * EPI_STRIDE + lds_m) * fx.Int64(
                        BF16_BYTES
                    )
                    fx.ptr_store(fx.Vector.from_elements(vals, dtype=elem_ty), fx.inttoptr(_epi_ptr_ty, addr))

                barrier(vmcnt=None, lgkmcnt=0)

                for it in range_constexpr(EPI_ITERS):
                    idx = tid + it * BLOCK_THREADS
                    r_row = idx // _slots_per_row  # staged channel slot
                    r_m = (idx % _slots_per_row) * EPI_VEC  # local npq in the tile
                    v = lds_load_vec8(a_lds, r_row * EPI_STRIDE + r_m)
                    # Undo the (wave_n, c_n) packing the writers used to get the channel back.
                    col_off = (r_row // MFMA_N) * WARP_N + r_row % MFMA_N + ni * MFMA_N
                    col = n_offset + col_off
                    col_loc = (n_local + col_off) if const_expr(groups > 1) else col
                    row = m_offset + r_m
                    buffer_ops.buffer_store(v, y_rsrc, _route(_out_off(row, col), row, col_loc))

        def store_acc():
            if const_expr(has_bias and not use_splitk):
                bias_vals = []
                for ni in range_constexpr(MI_N):
                    col, col_loc = _cols(ni)
                    col_i = fx.Int32(col)  # bias is indexed by the global out-channel
                    if const_expr(n_tail):
                        col_i = arith.select(col_loc < fx.Index(KG), col_i, fx.Int32(0))
                    bias_vals.append(
                        fx.Float32(buffer_ops.buffer_load(bias_rsrc, col_i, vec_width=1, dtype=fx.Float32))
                    )

            for mi in range_constexpr(MI_M):
                row_base = m_offset + wave_m * WARP_M + mi * MFMA_M + c_m_vec
                for ni in range_constexpr(MI_N):
                    col, col_loc = _cols(ni)
                    a = Vec(acc[mi * MI_N + ni])
                    if const_expr(has_bias and not use_splitk):
                        bias_val = bias_vals[ni]

                    if const_expr(_vec_store):
                        row0 = fx.Index(row_base)
                        off_nk0 = col * dhw + row0

                        def _emit_vec():
                            vals = []
                            for i in range_constexpr(MFMA_C_VALUES):
                                cval = (a[i] + bias_val) if const_expr(has_bias) else a[i]
                                vals.append(cval.to(elem_ty))
                            v4 = fx.Vector.from_elements(vals, dtype=elem_ty)
                            buffer_ops.buffer_store(v4, y_rsrc, _route(off_nk0, row0, col_loc))

                        if const_expr(_need_chk and not _route_store):
                            if _valid_raw(row0, col_loc):
                                _emit_vec()
                        else:
                            _emit_vec()
                        continue

                    for i in range_constexpr(MFMA_C_VALUES):
                        row = fx.Index(row_base + i)
                        off_sk = row * k + col

                        if const_expr(out_ndhwc):
                            # (n, do, ho, wo, k) is row-major over exactly the GEMM's own
                            # (npq, k) index space, so the scatter collapses to off_sk and
                            # the row decomposition disappears.
                            off_nk = off_sk
                        elif const_expr(n == 1):
                            off_nk = col * dhw + row
                        else:
                            n_idx = row // dhw
                            sp = row % dhw
                            off_nk = n_idx * (k * dhw) + col * dhw + sp

                        def _emit():
                            if const_expr(use_splitk):
                                off_b = fx.Int32(off_sk * 4)
                                z0 = fx.Int32(0)
                                buffer_atomic_add(a[i], y_rsrc, off_b, z0, z0)
                            else:
                                cval = (a[i] + bias_val).to(elem_ty) if const_expr(has_bias) else a[i].to(elem_ty)
                                if const_expr(BIG_OUT):
                                    _big_store(fx.Int64(off_nk), cval)
                                else:
                                    buffer_ops.buffer_store(cval, y_rsrc, _route(off_nk, row, col_loc))

                        if const_expr(_need_chk and not _route_store):
                            if _valid_raw(row, col_loc):
                                _emit()
                        else:
                            _emit()

        if const_expr(LDS_EPILOGUE):
            store_acc_lds()
        else:
            store_acc()

    @flyc.jit
    def launch(y: fx.Tensor, x: fx.Tensor, weight: fx.Tensor, bias: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        conv3d_implicit_kernel(y, x, weight, bias).launch(
            grid=(grid_x, grid_n, m_chunks * splitk), block=(BLOCK_THREADS, 1, 1), stream=stream
        )

    def _launch(y, x, weight, bias, stream=None):
        with CompilationContext.compile_hints(CONV_COMPILE_HINTS):
            return launch(y, x, weight, bias, stream=_as_stream(stream))

    def _compile(y, x, weight, bias, stream=None):
        with CompilationContext.compile_hints(CONV_COMPILE_HINTS):
            return flyc.compile(launch, y, x, weight, bias, _as_stream(stream))

    _launch.compile = _compile
    return _launch


SPLITK_MAX_STAGING_BYTES = 0xFFFFFFFF

# K tiles a one-wave grid must have before split-K's two extra launches pay for
# themselves. See the deep-K branch in _resolve_splitk for the measurement.
SPLITK_LONG_K_TILES = 256
SPLITK_MAX_DEEP = 16


@functools.lru_cache(maxsize=8)
def _num_cu(device):
    try:
        return torch.cuda.get_device_properties(device).multi_processor_count
    except Exception:
        return 256


@functools.lru_cache(maxsize=8)
def _lds_budget(device):
    """Bytes of LDS one block may allocate: 64KB on CDNA3, 160KB on CDNA4."""
    try:
        return torch.cuda.get_device_properties(device).shared_memory_per_block
    except Exception:
        return 64 * 1024


def _tile_lds_bytes(tile_m, tile_n):
    """What ``compile_conv3d_implicit`` allocates for this tile's A and B stages."""
    return 2 * TILES_PER_BARRIER * (tile_m + tile_n) * TILE_K * BF16_BYTES


@functools.lru_cache(maxsize=256)
def _pick_tile(npq, k, crs, x_bytes, groups, device):
    """(tile, wgm) for this problem: the smallest tile that fills the GPU, widened when
    the shape is bandwidth-bound.

    The base rule is occupancy: walk down ``TILE_LADDER`` and take the first tile whose
    grid reaches ``TILE_MIN_WAVES_PER_CU``, so a small problem gets a small tile rather
    than a few fat blocks on an idle GPU.

    That rule alone is blind to DRAM traffic, and one thing it misses is expensive. When
    ``TILE_N`` does not cover K/groups the activation is read from HBM once per N tile,
    and on a shape with little arithmetic to hide it behind, those extra passes are the
    whole runtime -- a 1x1 conv over 512 channels moves 536MB to do 69 GFLOP. So compare
    the two limits directly and, when DRAM wins, spend LDS on a tile that spans K in one
    step. Measured on gfx950 for 1,512,512,512 x 256,512,1,1: 189us at (128, 128, 2, 4),
    144us at (256, 256, 2, 4). Compute-bound shapes keep the narrow tile, which is what
    the ridge-point test is there to tell apart -- the 3x3 convs at the same input size
    sit a factor of two on the arithmetic side of it.
    """
    kg = k // groups
    target = TILE_MIN_WAVES_PER_CU * _num_cu(device)

    def fills_gpu(tile):
        blocks = ((npq + tile[0] - 1) // tile[0]) * groups * ((kg + tile[1] - 1) // tile[1])
        return blocks * tile[2] * tile[3] >= target

    legal = [t for t in TILE_LADDER if t[1] <= kg] or [TILE_LADDER[-1]]
    base = next((t for t in legal if fills_gpu(t)), legal[-1])

    n_tiles = (kg + base[1] - 1) // base[1]
    if n_tiles == 1:
        return base, 1
    dram_bytes = n_tiles * x_bytes + npq * k * BF16_BYTES
    if dram_bytes * TILE_RIDGE_FLOP_PER_BYTE <= 2 * npq * kg * groups * crs:
        return base, 1

    budget = _lds_budget(device)
    for tile in TILE_LADDER_WIDE:
        if tile[1] >= kg and _tile_lds_bytes(tile[0], tile[1]) <= budget and fills_gpu(tile):
            return tile, 1
    return base, TILE_WIDE_WGM


@functools.lru_cache(maxsize=256)
def _resolve_splitk(splitk, npq, crs, k, device, tile=DEFAULT_TILE, groups=1):
    k_tiles = (crs + TILE_K - 1) // TILE_K
    if npq * k * 4 > SPLITK_MAX_STAGING_BYTES:
        return 1
    if splitk is None:
        tile_m, tile_n = tile[0], tile[1]
        kg = k // groups
        base = ((npq + tile_m - 1) // tile_m) * groups * ((kg + tile_n - 1) // tile_n)
        num_cu = _num_cu(device)
        if crs % TILE_K != 0 or npq * k * 4 > 0x7FFFFFFF:
            sk = 1
        elif base < num_cu and k_tiles >= SPLITK_LONG_K_TILES:
            # Deep-K, few-block shapes -- small spatial extent over many channels -- never
            # fill the grid, so every block runs in one wave and the kernel costs one
            # block's entire K loop while the rest of the GPU waits. Split-K buys that
            # back for two extra launches (the fp32 staging memset and the epilogue that
            # narrows it), a fixed cost only a long K loop repays: measured on gfx950,
            # k_tiles=288 takes a 59us conv to 44us, while k_tiles=144 takes a 32us one
            # to 44us. Neither a small npq nor a ragged tile gates this the way it gates
            # the general case below -- the epilogue already masks both tails, and a grid
            # this empty has no occupancy left to lose to them.
            sk = min(SPLITK_MAX_DEEP, max(1, 4 * num_cu // base), k_tiles // 16)
        elif npq < 4096 or k_tiles < 16 or kg % tile_n != 0 or npq % tile_m != 0 or base >= (3 * num_cu) // 4:
            sk = 1
        else:
            sk = min(4, max(1, num_cu // base), k_tiles)
    else:
        sk = max(1, splitk)
    while sk > 1 and k_tiles % sk != 0:
        sk -= 1
    return sk


def _as_tuple(v, rank, name):
    if isinstance(v, int):
        return (v,) * rank
    t = tuple(v)
    if len(t) == 1:
        return t * rank
    assert len(t) == rank, f"{name} must be an int or a sequence of 1 or {rank} ints, got {tuple(v)}"
    return t


def _resolve_padding(padding, kernel, stride, dilation):
    """Normalize torch's ``padding`` argument to a (low, high) pair of per-axis triples."""
    if not isinstance(padding, str):
        p = _as_tuple(padding, 3, "padding")
        assert min(p) >= 0, f"negative padding is not supported, got (pt, ph, pw) = {p}"
        return p, p
    if padding == "valid":
        return (0, 0, 0), (0, 0, 0)
    if padding != "same":
        raise ValueError(f"padding string must be 'same' or 'valid', got {padding!r}")
    assert all(
        s == 1 for s in stride
    ), f"padding='same' is not supported for strided convolutions, got stride {tuple(stride)}"
    total = [dl * (kn - 1) for kn, dl in zip(kernel, dilation)]
    return tuple(t // 2 for t in total), tuple(t - t // 2 for t in total)


def _conv3d_impl(
    x,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    padding_mode="zeros",
    splitk=None,
    stream=None,
    tile=None,
    autotune=None,
    input_layout="NCDHW",
    output_layout="NCDHW",
    matmul_1x1=True,
):
    _check_layouts(3, input_layout, output_layout)

    in_ndhwc = input_layout == "NDHWC"
    out_ndhwc = output_layout == "NDHWC"
    n, c, d, h, w = _shape_ncdhw(x, in_ndhwc)
    k, wc, kt, kh, kw = weight.shape

    for name, t in (("x", x), ("weight", weight), ("bias", bias)):
        assert t is None or t.is_cuda, f"conv3d_implicit needs GPU tensors; {name} is on {t.device}"
    assert (
        x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
    ), f"conv3d_implicit is a bf16-only kernel; got x={x.dtype}, weight={weight.dtype}"
    assert bias is None or (bias.dim() == 1 and bias.numel() == k), (
        f"bias must be a 1-D tensor of {k} elements, one per output channel; " f"got shape {tuple(bias.shape)}"
    )
    groups = int(groups)
    assert groups >= 1, f"groups must be >= 1, got {groups}"
    assert c % groups == 0, f"in-channels {c} not divisible by groups {groups}"
    assert k % groups == 0, f"out-channels {k} not divisible by groups {groups}"
    assert wc == c // groups, f"weight in-channels {wc} != C/groups = {c // groups}"
    st, sh, sw = _as_tuple(stride, 3, "stride")

    assert min(st, sh, sw) >= 1, f"non-positive stride is not supported, got (st, sh, sw) = {(st, sh, sw)}"
    dt, dh, dw = _as_tuple(dilation, 3, "dilation")
    assert min(dt, dh, dw) >= 1, f"dilation must be >= 1, got {(dt, dh, dw)}"
    pad_lo, pad_hi = _resolve_padding(padding, (kt, kh, kw), (st, sh, sw), (dt, dh, dw))
    pt, ph, pw = pad_lo
    assert padding_mode in PADDING_MODES, f"padding_mode must be one of {PADDING_MODES}, got {padding_mode!r}"

    if padding_mode in ("reflect", "circular"):
        for ax, (p, ext) in enumerate(zip(map(max, pad_lo, pad_hi), (d, h, w))):
            if padding_mode == "reflect":
                assert p < ext, f"reflect padding {p} must be < input extent {ext} on spatial axis {ax}"
            else:
                assert p <= ext, f"circular padding {p} must be <= input extent {ext} on spatial axis {ax}"

    # An asymmetric pad (only 'same' produces one, when the dilated filter extent is even)
    # is handled by the kernel like any other: the gather reads the low pad alone, and the
    # high pad reaches it as part of the output extent. Nothing is materialized here.
    inline_pad = padding_mode != "zeros" and (any(pad_lo) or any(pad_hi))
    if inline_pad and _big_in(n, c, groups, d, h, w, *map(max, pad_lo, pad_hi)):
        pads = (pad_lo[2], pad_hi[2], pad_lo[1], pad_hi[1], pad_lo[0], pad_hi[0])
        x, in_ndhwc = _pad_spatial(x, in_ndhwc, pads, padding_mode)
        n, c, d, h, w = _shape_ncdhw(x, in_ndhwc)
        pad_lo = pad_hi = (0, 0, 0)
        pt, ph, pw = pad_lo
        inline_pad = False
    pad_mode = padding_mode if inline_pad else "zeros"

    if (
        matmul_1x1
        and groups == 1
        and kt == 1
        and kh == 1
        and kw == 1
        and st == 1
        and sh == 1
        and sw == 1
        and pt == 0
        and ph == 0
        and pw == 0
    ):
        wm = weight.reshape(k, c)
        if in_ndhwc:
            y = torch.matmul(x.reshape(n * d * h * w, c), wm.t()).reshape(n, d, h, w, k)
            if bias is not None:
                y = y + bias.to(y.dtype)
            return y if out_ndhwc else y.permute(0, 4, 1, 2, 3).contiguous()
        if n == 1:
            y = torch.matmul(wm, x.reshape(c, d * h * w)).reshape(n, k, d, h, w)
        else:
            y = torch.matmul(wm, x.reshape(n, c, d * h * w)).reshape(n, k, d, h, w)
        if bias is not None:
            y = y + bias.to(y.dtype).view(1, k, 1, 1, 1)
        return y.permute(0, 2, 3, 4, 1).contiguous() if out_ndhwc else y

    do = (d + pad_lo[0] + pad_hi[0] - (dt * (kt - 1) + 1)) // st + 1
    ho = (h + pad_lo[1] + pad_hi[1] - (dh * (kh - 1) + 1)) // sh + 1
    wo = (w + pad_lo[2] + pad_hi[2] - (dw * (kw - 1) + 1)) // sw + 1
    assert min(do, ho, wo) >= 1, f"dilated filter is larger than the padded input: output ({do}, {ho}, {wo})"
    npq = n * do * ho * wo

    if n == 0:
        empty = (0, do, ho, wo, k) if out_ndhwc else (0, k, do, ho, wo)
        return torch.empty(empty, device=x.device, dtype=torch.bfloat16)

    # `c` becomes the channel count the GEMM indexes; `c_src` is what `x` still holds.
    # The widening is not applied here -- the NCDHW route folds it into the transpose it
    # already runs, which is one launch instead of F.pad's fill and copy plus that.
    cg = c // groups
    cgp = _pad_channels(cg)
    c_src, c = c, groups * cgp
    crs = cgp * kt * kh * kw

    launch_stream = torch.cuda.current_stream() if stream is None else stream
    has_bias = bias is not None
    bias_arg = bias.to(torch.float32).contiguous() if has_bias else _bias_placeholder(x.device)

    # Whether the input is transposed depends on the tile, so the tile is settled first.
    # Autotune is the exception: its sweep spans tiles the NCDHW A layout cannot serve, so
    # it stays on the transposed input and picks among tiles on equal terms.
    # Reading NCDHW in place has no step to widen the channels in, and paying F.pad to
    # get them is no cheaper than the transpose it exists to avoid, so it stays off there.
    nchw_a = c == c_src and _nchw_a_ok(in_ndhwc, n, c, d, h, w, kt, kh, kw, st, sh, sw, pt, ph, pw, pad_mode)
    tuning = tile is None and (autotune or (autotune is None and _autotune_enabled()))
    if tile is not None:
        chosen_tile, chosen_wgm = tuple(tile), 1
    elif tuning:
        chosen_tile, chosen_wgm = None, 1
    else:
        chosen_tile, chosen_wgm = _pick_tile(npq, k, crs, n * c * d * h * w * BF16_BYTES, groups, x.device)
    nchw_a = nchw_a and not tuning and chosen_tile[0] % NCHW_A_MIN_TILE_M == 0

    # Hand the transpose the stream we already resolved; letting it default would cost a
    # second torch.cuda.current_stream(), which is ~2.7us of the small-shape host budget.
    if nchw_a:
        x_arg = x.contiguous()
    elif in_ndhwc:
        # Already channels-last, so there is no transpose to fold the widening into.
        if c != c_src:
            x = torch.nn.functional.pad(x.reshape(n, d, h, w, groups, cg), (0, cgp - cg))
            x = x.reshape(n, d, h, w, c)
        x_arg = x.contiguous()
    else:
        x_arg = _ncdhw_to_ndhwc(x, launch_stream, c, groups)
    w_packed = _prep_weight(weight, k, kt, kh, kw, wc)

    shape = (
        # fmt: off
        n, c, d, h, w, k, kt, kh, kw, st, sh, sw, pt, ph, pw, pad_hi,
        dt, dh, dw, pad_mode, has_bias, groups, out_ndhwc,
        # fmt: on
    )

    def _run(the_tile, the_wgm=1):
        sk = _resolve_splitk(splitk, npq, crs, k, x.device, the_tile, groups)
        if sk > 1:
            y = torch.zeros((npq, k), device=x.device, dtype=torch.float32)
        else:
            out_shape = (n, do, ho, wo, k) if out_ndhwc else (n, k, do, ho, wo)
            y = torch.empty(out_shape, device=x.device, dtype=torch.bfloat16)
        exe = compile_conv3d_implicit(
            n,
            c,
            d,
            h,
            w,
            k,
            kt,
            kh,
            kw,
            st,
            sh,
            sw,
            pt,
            ph,
            pw,
            dt,
            dh,
            dw,
            pad_mode,
            has_bias,
            sk,
            the_tile,
            the_wgm,
            groups,
            out_ndhwc,
            nchw_a,
            pad_hi,
        )
        _dispatch(exe, y, x_arg, w_packed, bias_arg, stream=launch_stream)
        return y, sk

    if tuning:
        from kernels.conv.conv3d_autotune import BF16_CANDIDATES, WGM_VALUES, autotune_conv3d

        candidates = [(t, w) for t in BF16_CANDIDATES for w in WGM_VALUES]
        best = autotune_conv3d("bf16", shape, "bf16", candidates, x.device, lambda tw: _run(tw[0], tw[1])[0])
        chosen_tile, chosen_wgm = best

    y, sk = _run(chosen_tile, chosen_wgm)
    if sk > 1:
        if has_bias:
            y = y + bias_arg.view(1, k)
        if out_ndhwc:
            return y.view(n, do, ho, wo, k).to(torch.bfloat16)
        out = torch.empty((n, k, do, ho, wo), device=x.device, dtype=torch.bfloat16)
        out.copy_(y.view(n, do, ho, wo, k).permute(0, 4, 1, 2, 3))
        return out
    return y


def _conv2d_impl(
    x, weight, bias=None, stride=1, padding=0, dilation=1, input_layout="NCHW", output_layout="NCHW", **kwargs
):
    assert x.dim() == 4 and weight.dim() == 4, "conv2d expects (N,C,H,W) / (K,C,R,S)"
    _check_layouts(2, input_layout, output_layout)
    sh, sw = _as_tuple(stride, 2, "stride")
    dh, dw = _as_tuple(dilation, 2, "dilation")

    if isinstance(padding, str):
        p3 = padding
    else:
        ph, pw = _as_tuple(padding, 2, "padding")
        p3 = (0, ph, pw)
    k, wc, r, s = weight.shape

    if input_layout == "NHWC":
        n, h, w, c = x.shape
        x5, in5 = x.reshape(n, 1, h, w, c), "NDHWC"
    else:
        n, c, h, w = x.shape
        x5, in5 = x.reshape(n, c, 1, h, w), "NCDHW"
    out5 = "NDHWC" if output_layout == "NHWC" else "NCDHW"
    w5 = weight.reshape(k, wc, 1, r, s)
    y5 = _conv3d_impl(
        x5,
        w5,
        bias=bias,
        stride=(1, sh, sw),
        padding=p3,
        dilation=(1, dh, dw),
        input_layout=in5,
        output_layout=out5,
        **kwargs,
    )
    if output_layout == "NHWC":
        return y5.reshape(y5.shape[0], y5.shape[2], y5.shape[3], y5.shape[4])
    return y5.reshape(y5.shape[0], y5.shape[1], y5.shape[3], y5.shape[4])


def _conv1d_impl(
    x, weight, bias=None, stride=1, padding=0, dilation=1, input_layout="NCW", output_layout="NCW", **kwargs
):
    assert x.dim() == 3 and weight.dim() == 3, "conv1d expects (N,C,W) / (K,C,S)"
    _check_layouts(1, input_layout, output_layout)
    (sw,) = _as_tuple(stride, 1, "stride")
    (dw,) = _as_tuple(dilation, 1, "dilation")
    if isinstance(padding, str):
        p3 = padding
    else:
        p3 = (0, 0, _as_tuple(padding, 1, "padding")[0])
    k, wc, s = weight.shape
    if input_layout == "NWC":
        n, w, c = x.shape
        x5, in5 = x.reshape(n, 1, 1, w, c), "NDHWC"
    else:
        n, c, w = x.shape
        x5, in5 = x.reshape(n, c, 1, 1, w), "NCDHW"
    out5 = "NDHWC" if output_layout == "NWC" else "NCDHW"
    w5 = weight.reshape(k, wc, 1, 1, s)
    y5 = _conv3d_impl(
        x5,
        w5,
        bias=bias,
        stride=(1, 1, sw),
        padding=p3,
        dilation=(1, 1, dw),
        input_layout=in5,
        output_layout=out5,
        **kwargs,
    )
    if output_layout == "NWC":
        return y5.reshape(y5.shape[0], y5.shape[3], y5.shape[4])
    return y5.reshape(y5.shape[0], y5.shape[1], y5.shape[4])


def conv3d_implicit(x, weight, bias=None, stride=1, padding=0, dilation=1, **kwargs):
    """Main implicit-GEMM conv entry; dispatches 1D/2D/3D by filter rank.

    Rank is taken from the filter (weight.dim() - 2): 3 -> 3D (N,C,D,H,W)/(K,C,T,R,S).

    ``input_layout`` and ``output_layout`` are independent and named per rank:
    "NCDHW"/"NDHWC", "NCHW"/"NHWC", "NCW"/"NWC". The weight stays KC*, and the batch axis
    leads in both, so an unbatched input works either way. Channels-last is the kernel's
    own layout on both sides: an NDHWC input skips the pre-transpose, and an NDHWC output
    is the (npq, K) index space the GEMM already writes, so it also skips the split-K
    epilogue's transpose. Channels-last output does give up the vectorized store on the
    ``n == 1`` fast path, since a lane's four accumulator values are four M rows and those
    are K apart once channels are innermost.

    ``padding`` takes an int, a per-axis tuple, or one of torch's two strings. "valid" is
    no padding. "same" pads so the output keeps the input's spatial extent, which needs
    ``dilation * (kernel - 1)`` elements per axis and, like torch, is only defined at
    stride 1. That total is normally even and splits evenly; an even-length filter under
    odd dilation makes it odd, and torch's rule puts the extra element on the high side.
    Either way it costs nothing at runtime: the gather only ever reads the low pad, so the
    high one is just a wider output extent -- unlike torch, which materializes a padded
    copy of the input and warns about it. ``padding_mode`` applies to "same" too.

    ``dilation`` follows torch semantics: it spaces the filter taps by that factor
    over the input, shrinking the output to
    ``(D + 2*pad - dilation*(T-1) - 1)//stride + 1`` per axis. It costs nothing in the
    GEMM -- the K axis is still C/groups*T*R*S -- it only stretches the im2col gather,
    so a dilated filter reads a wider input footprint per output row and gets less
    reuse out of cache than the same filter undilated.

    ``groups`` follows torch semantics: C and K must both be divisible by it and the
    weight's channel dim is C/groups. Groups map onto the N grid axis, one tile never
    spanning two groups, so efficiency tracks how well K/groups fills TILE_N. Measured
    on gfx950 vs torch/MIOpen, moderate cardinality wins across the board (1.5-2.0x for
    K/groups in [8, 256]). True depthwise (groups == C, so C/groups == 1) is the one
    weak case at ~0.5x: C/groups=1 pads to the gather's 8-wide vector, wasting 7/8 of
    the K axis, while K/groups=1 leaves all but one column of the N tile masked.
    Narrower tiles recover little there -- depthwise wants its own kernel, not this
    single-GEMM mapping.

    An unstrided unpadded 1x1 filter is a pure channel GEMM (im2col degenerates to a
    reshape), so it goes to ``torch.matmul`` by default. ``matmul_1x1=False`` forces those
    shapes through the kernel instead -- for benchmarking the kernel itself, or to avoid
    rocBLAS's 32-bit output indexing on an output above 2**32 elements. Grouped 1x1 is
    block-diagonal rather than a plain channel GEMM and always goes through the kernel.
    """
    spatial_rank = weight.dim() - 2
    if spatial_rank not in (1, 2, 3):
        raise ValueError(f"conv3d_implicit supports 1D/2D/3D; got filter rank {weight.dim()}")
    unbatched = x.dim() == weight.dim() - 1
    if unbatched:
        x = x.unsqueeze(0)
    assert x.dim() == weight.dim(), f"x rank {x.dim()} != weight rank {weight.dim()}"
    impl = {3: _conv3d_impl, 2: _conv2d_impl, 1: _conv1d_impl}[spatial_rank]
    y = impl(x, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, **kwargs)
    return y.squeeze(0) if unbatched else y
