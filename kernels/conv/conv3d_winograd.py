# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Winograd F(m x m, 3x3) convolution (BF16), for 3x3 / stride-1 / dilation-1 filters.

This is the classic Lavin & Gray decomposition, and it is deliberately the textbook
three-stage form rather than a fused kernel:

    1. weight transform   U = G g G^T          (host, torch, cached per weight tensor)
    2. input transform    V = B^T d B          (``winograd_input_transform_kernel``)
    3. batched GEMM       M[i] = V[i] @ U[i]   (``torch.bmm``, one call for all A*A points)
    4. output transform   Y = A^T M A          (``winograd_output_transform_kernel``)

``F(2x2,3x3)`` replaces 36 multiply-accumulates per 2x2 output patch with 16, a 2.25x
arithmetic reduction; ``F(4x4,3x3)`` replaces 144 with 36, a 4x reduction. What it costs
is memory traffic and launches: V is ``A*A / m*m`` times the size of the input patch grid
(4x for F(2x2,3x3), 2.25x for F(4x4,3x3)), and the pipeline is three device kernels where
the implicit-GEMM path in ``conv3d_implicit`` is one.

So this wins where the convolution is genuinely arithmetic-bound and loses where it is
launch- or bandwidth-bound. Measured on gfx950/MI350X over the 3x3 stride-1 shapes of
``scripts/run_conv.py``'s harness, geomean against ``conv3d_implicit``: 0.53x on wall time
and 0.65x on CUDA-graph device time. The transforms, not the GEMM, are the cost -- they are
55-72% of device time, and they move 4x the input tensor at roughly 2 TB/s.

It does win on deep-channel, small-spatial shapes, where the patch grid is small enough
that the 4x traffic is cheap and the arithmetic really is the bill: 1.5x over
``conv3d_implicit`` at 1x512x7x7 -> 512, and 1.1-1.4x over torch at 256c/14x14, 512c/7x7,
1024c/7x7 and 1024c/4x4. Everywhere else ``conv3d_implicit`` is the faster path.

Both paths produce the same output, so the choice is a benchmark question per shape, and
``winograd_supported`` exists so a caller can make it.

Numerics: the F(2x2,3x3) transforms are all +-1, so they are exact in fp32 and cost no
accuracy beyond the extra bf16 rounding of V and M. F(4x4,3x3)'s transforms span 1/24 to
8, which is a real conditioning loss -- with bf16's 8-bit mantissa it is measurably worse
than direct convolution and is offered for experiment, not as a default.

Restrictions relative to ``conv3d_implicit``: 3x3 filters only, stride 1, dilation 1,
groups 1, symmetric zero padding, and 2-D (a 3-D filter is accepted only when its depth
extent is 1). NHWC is the native layout on both sides.
"""

import functools
import weakref

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, range_constexpr
from kernels.common import buffer_ops
from kernels.conv.conv3d_implicit import (
    BF16_BYTES,
    CONV_COMPILE_HINTS,
    _as_stream,
    _as_tuple,
    _bias_placeholder,
    _dispatch,
    _ncdhw_to_ndhwc,
    _resolve_padding,
)

# Transform triples, indexed by variant name. ``m`` is the output tile edge, so the input
# tile edge is ``m + 2``. ``BT`` and ``AT`` are the kernel-side transforms and are integer
# by construction, which is why the kernels never multiply by a non-representable
# constant; ``G`` is the host-side weight transform and is the only one with fractions.
WINOGRAD_VARIANTS = {
    "F2x2_3x3": dict(
        m=2,
        BT=(
            (1, 0, -1, 0),
            (0, 1, 1, 0),
            (0, -1, 1, 0),
            (0, 1, 0, -1),
        ),
        G=(
            (1.0, 0.0, 0.0),
            (0.5, 0.5, 0.5),
            (0.5, -0.5, 0.5),
            (0.0, 0.0, 1.0),
        ),
        AT=(
            (1, 1, 1, 0),
            (0, 1, -1, -1),
        ),
    ),
    "F4x4_3x3": dict(
        m=4,
        BT=(
            (4, 0, -5, 0, 1, 0),
            (0, -4, -4, 1, 1, 0),
            (0, 4, -4, -1, 1, 0),
            (0, -2, -1, 2, 1, 0),
            (0, 2, -1, -2, 1, 0),
            (0, 4, 0, -5, 0, 1),
        ),
        G=(
            (1 / 4, 0.0, 0.0),
            (-1 / 6, -1 / 6, -1 / 6),
            (-1 / 6, 1 / 6, -1 / 6),
            (1 / 24, 1 / 12, 1 / 6),
            (1 / 24, -1 / 12, 1 / 6),
            (0.0, 0.0, 1.0),
        ),
        AT=(
            (1, 1, 1, 1, 1, 0),
            (0, 1, -1, 2, -2, 0),
            (0, 1, 1, 4, 4, 0),
            (0, 1, -1, 8, -8, 1),
        ),
    ),
}

DEFAULT_VARIANT = "F2x2_3x3"

BLOCK_THREADS = 256

# Threads spanning the channel (resp. output-channel) axis within a block. The remaining
# threads span tiles, so a block covers LANES * VEC channels and BLOCK_THREADS // LANES
# tiles, and the 8 lanes of one channel group issue one contiguous 128-byte request.
LANES = 8
TILES_PER_BLOCK = BLOCK_THREADS // LANES

# Widest vector a buffer instruction takes here is 16 bytes; F(4x4,3x3) holds 36 live
# transform vectors instead of 16, so it gets a narrower one to stay off the VGPR wall.
VEC_MAX = {"F2x2_3x3": 8, "F4x4_3x3": 4}

# Offset that reads past num_records, which is what makes an out-of-image tap zero-fill.
OOB_SENTINEL_ELEM = 0x7FFFFF80

MAX_GRID_YZ = 65535


def _vec_width(extent, cap):
    """Widest power-of-two vector up to ``cap`` that divides ``extent`` exactly."""
    v = min(cap, 8)
    while v > 1 and extent % v != 0:
        v //= 2
    return v


def _vload_f32(rsrc, off, width, dtype):
    """Buffer-load ``width`` elements as one f32 Vector.

    A width-1 buffer load yields a scalar rather than a ``vector<1x...>``, so it is
    re-wrapped: the transform code below is written once and only over Vectors.
    """
    raw = buffer_ops.buffer_load(rsrc, off, vec_width=width, dtype=dtype)
    vec = fx.Vector.from_elements([dtype(raw)], dtype=dtype) if width == 1 else fx.Vector(raw)
    return vec.to(fx.Float32)


def _vstore(vec, rsrc, off, width, dtype):
    """Store an f32 Vector as ``dtype``; a width-1 vector goes back out as a scalar."""
    out = vec.to(dtype)
    buffer_ops.buffer_store(out[0] if width == 1 else out, rsrc, off)


def _lincomb(row, vals):
    """Emit ``sum(row[i] * vals[i])`` at trace time, skipping zeros and folding +-1.

    ``vals`` are DSL values; ``row`` is a tuple of Python numbers. Every row of every
    transform in ``WINOGRAD_VARIANTS`` has a nonzero, so the accumulator is never empty.
    """
    acc = None
    for coeff, v in zip(row, vals):
        if coeff == 0:
            continue
        mag = abs(coeff)
        term = v if mag == 1 else v * fx.Float32(float(mag))
        if acc is None:
            acc = term if coeff > 0 else -term
        else:
            acc = acc + term if coeff > 0 else acc - term
    return acc


def _lincomb_all(mat, vals):
    return [_lincomb(row, vals) for row in mat]


def _transform2d(mat, d):
    """Emit ``mat @ d @ mat.T`` over a trace-time grid of DSL values.

    ``d`` is a list of rows. Applying ``mat`` down the columns and then across the rows
    of the result is the same operation twice, because ``(tmp @ mat.T)[i][j]`` is
    ``sum_k mat[j][k] * tmp[i][k]`` -- exactly ``_lincomb_all(mat, tmp[i])``.
    """
    inner = len(d)
    cols = [[d[r][cc] for r in range(inner)] for cc in range(len(d[0]))]
    tmp_cols = [_lincomb_all(mat, col) for col in cols]
    tmp = [[tmp_cols[cc][r] for cc in range(len(cols))] for r in range(len(mat))]
    return [_lincomb_all(mat, row) for row in tmp]


for _name, _var in WINOGRAD_VARIANTS.items():
    _a = _var["m"] + 2
    assert len(_var["BT"]) == _a and all(len(r) == _a for r in _var["BT"]), _name
    assert len(_var["G"]) == _a and all(len(r) == 3 for r in _var["G"]), _name
    assert len(_var["AT"]) == _var["m"] and all(len(r) == _a for r in _var["AT"]), _name
    assert all(any(cc for cc in r) for r in _var["BT"] + _var["AT"]), _name


@functools.lru_cache(maxsize=64)
def compile_winograd_input_transform(n, h, w, c, ph, pw, ho, wo, np_pad, variant):
    """Build ``x (N,H,W,C) -> V (A*A, np_pad, C)``, the B^T d B stage."""
    var = WINOGRAD_VARIANTS[variant]
    m_tile, bt = var["m"], var["BT"]
    a = m_tile + 2
    p_w = (wo + m_tile - 1) // m_tile
    p_t = ((ho + m_tile - 1) // m_tile) * p_w

    cvec = _vec_width(c, VEC_MAX[variant])
    ch_per_block = LANES * cvec
    grid_c = (c + ch_per_block - 1) // ch_per_block
    grid_p = np_pad // TILES_PER_BLOCK
    ch_tail = c % ch_per_block != 0

    elem_ty = fx.BFloat16
    x_bytes = n * h * w * c * BF16_BYTES
    v_bytes = a * a * np_pad * c * BF16_BYTES
    assert x_bytes < OOB_SENTINEL_ELEM * BF16_BYTES, f"input {x_bytes}B exceeds the buffer window"
    assert v_bytes <= 0xFFFFFFFF, f"transformed input {v_bytes}B exceeds the 4 GiB buffer window"
    assert a * a * np_pad * c <= 0x7FFFFFFF, "transformed input exceeds the 32-bit element offset"
    assert grid_c <= MAX_GRID_YZ, f"grid.y = {grid_c} exceeds the {MAX_GRID_YZ}-block limit"

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def winograd_input_transform_kernel(v: fx.Tensor, x: fx.Tensor):
        x_rsrc = buffer_ops.create_buffer_resource(x, num_records_bytes=x_bytes)
        v_rsrc = buffer_ops.create_buffer_resource(v, num_records_bytes=v_bytes)

        tid = fx.thread_idx.x
        ch = (fx.block_idx.y * LANES + tid % LANES) * cvec
        p = fx.block_idx.x * TILES_PER_BLOCK + tid // LANES

        b = p // p_t
        rem = p % p_t
        h0 = (rem // p_w) * m_tile - ph
        w0 = (rem % p_w) * m_tile - pw

        # Gather the a x a input patch. A tap outside the image -- padding, the ragged
        # tail of the tile grid, or a padded-up tile index whose batch runs past N --
        # is steered past num_records, and the buffer returns zero for it.
        oob = fx.Int32(OOB_SENTINEL_ELEM)
        d = []
        for i in range_constexpr(a):
            hh = h0 + i
            row = []
            for j in range_constexpr(a):
                ww = w0 + j
                inside = (hh >= 0) & (hh < h) & (ww >= 0) & (ww < w) & (b < n)
                off = fx.Int32(((b * h + hh) * w + ww) * c + ch)
                row.append(_vload_f32(x_rsrc, arith.select(inside, off, oob), cvec, elem_ty))
            d.append(row)

        out = _transform2d(bt, d)

        def store_all():
            for i in range_constexpr(a):
                for j in range_constexpr(a):
                    off = fx.Int32(((i * a + j) * np_pad + p) * c + ch)
                    _vstore(out[i][j], v_rsrc, off, cvec, elem_ty)

        # np_pad rounds the tile count up to a whole block, so the tile axis never needs a
        # guard -- those rows transform an all-zero patch into zeros the GEMM ignores. The
        # channel axis is not paddable that way, since C is the GEMM's contraction extent.
        if const_expr(ch_tail):
            if ch < c:
                store_all()
        else:
            store_all()

    @flyc.jit
    def launch_input_transform(v: fx.Tensor, x: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        winograd_input_transform_kernel(v, x).launch(
            grid=(grid_p, grid_c, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    def _launch(v, x, stream=None):
        with CompilationContext.compile_hints(CONV_COMPILE_HINTS):
            return launch_input_transform(v, x, stream=_as_stream(stream))

    def _compile(v, x, stream=None):
        with CompilationContext.compile_hints(CONV_COMPILE_HINTS):
            return flyc.compile(launch_input_transform, v, x, _as_stream(stream))

    _launch.compile = _compile
    return _launch


@functools.lru_cache(maxsize=64)
def compile_winograd_output_transform(n, ho, wo, k, np_pad, has_bias, variant):
    """Build ``M (A*A, np_pad, K) -> y (N, Ho, Wo, K)``, the A^T m A stage."""
    var = WINOGRAD_VARIANTS[variant]
    m_tile, at = var["m"], var["AT"]
    a = m_tile + 2
    p_w = (wo + m_tile - 1) // m_tile
    p_h = (ho + m_tile - 1) // m_tile
    p_t = p_h * p_w
    npq = n * p_t

    kvec = _vec_width(k, VEC_MAX[variant])
    k_per_block = LANES * kvec
    grid_k = (k + k_per_block - 1) // k_per_block
    grid_p = np_pad // TILES_PER_BLOCK
    k_tail = k % k_per_block != 0
    spatial_tail = ho % m_tile != 0 or wo % m_tile != 0 or npq != np_pad

    elem_ty = fx.BFloat16
    m_bytes = a * a * np_pad * k * BF16_BYTES
    y_bytes = n * ho * wo * k * BF16_BYTES
    assert m_bytes <= 0xFFFFFFFF, f"GEMM result {m_bytes}B exceeds the 4 GiB buffer window"
    assert a * a * np_pad * k <= 0x7FFFFFFF, "GEMM result exceeds the 32-bit element offset"
    assert grid_k <= MAX_GRID_YZ, f"grid.y = {grid_k} exceeds the {MAX_GRID_YZ}-block limit"

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def winograd_output_transform_kernel(y: fx.Tensor, mm: fx.Tensor, bias: fx.Tensor):
        m_rsrc = buffer_ops.create_buffer_resource(mm, num_records_bytes=m_bytes)
        y_rsrc = buffer_ops.create_buffer_resource(y, num_records_bytes=y_bytes)

        tid = fx.thread_idx.x
        kk = (fx.block_idx.y * LANES + tid % LANES) * kvec
        p = fx.block_idx.x * TILES_PER_BLOCK + tid // LANES

        b = p // p_t
        rem = p % p_t
        row0 = (rem // p_w) * m_tile
        col0 = (rem % p_w) * m_tile

        mvals = []
        for i in range_constexpr(a):
            row = []
            for j in range_constexpr(a):
                off = fx.Int32(((i * a + j) * np_pad + p) * k + kk)
                row.append(_vload_f32(m_rsrc, off, kvec, elem_ty))
            mvals.append(row)

        out = _transform2d(at, mvals)

        if const_expr(has_bias):
            # bias is f32, so kvec of it can exceed the 16-byte load; take it in dword4
            # chunks and stitch one kvec-wide vector out of the pieces.
            bias_rsrc = buffer_ops.create_buffer_resource(bias)
            scalars = []
            for lo in range_constexpr(0, kvec, 4):
                width = min(4, kvec - lo)
                chunk = _vload_f32(bias_rsrc, fx.Int32(kk + lo), width, fx.Float32)
                scalars.extend(chunk[e] for e in range_constexpr(width))
            bvec = fx.Vector.from_elements(scalars, dtype=fx.Float32)
            out = [[out[u][vv] + bvec for vv in range_constexpr(m_tile)] for u in range_constexpr(m_tile)]

        in_k = kk < k
        for u in range_constexpr(m_tile):
            for vv in range_constexpr(m_tile):
                rr = row0 + u
                cc = col0 + vv
                off = fx.Int32(((b * ho + rr) * wo + cc) * k + kk)
                if const_expr(spatial_tail and k_tail):
                    keep = in_k & (rr < ho) & (cc < wo) & (p < npq)
                elif const_expr(spatial_tail):
                    keep = (rr < ho) & (cc < wo) & (p < npq)
                elif const_expr(k_tail):
                    keep = in_k
                else:
                    keep = None
                if const_expr(keep is None):
                    _vstore(out[u][vv], y_rsrc, off, kvec, elem_ty)
                else:
                    if keep:
                        _vstore(out[u][vv], y_rsrc, off, kvec, elem_ty)

    @flyc.jit
    def launch_output_transform(y: fx.Tensor, mm: fx.Tensor, bias: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        winograd_output_transform_kernel(y, mm, bias).launch(
            grid=(grid_p, grid_k, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    def _launch(y, mm, bias, stream=None):
        with CompilationContext.compile_hints(CONV_COMPILE_HINTS):
            return launch_output_transform(y, mm, bias, stream=_as_stream(stream))

    def _compile(y, mm, bias, stream=None):
        with CompilationContext.compile_hints(CONV_COMPILE_HINTS):
            return flyc.compile(launch_output_transform, y, mm, bias, _as_stream(stream))

    _launch.compile = _compile
    return _launch


_U_CACHE = {}


def _evict_u(key, _ref):
    ent = _U_CACHE.get(key)
    if ent is not None and ent[0]() is None:
        del _U_CACHE[key]


def _prep_weight_winograd(w, variant):
    """Pack (K, C, 3, 3) -> U (A*A, C, K) = G g G^T, memoized on the source weight.

    Done in fp32 on the host because it is per-weight, not per-call: an inference weight
    is transformed once and reused, and doing it in fp32 keeps F(4x4,3x3)'s 1/24 and 1/12
    coefficients from being rounded before they are ever used.
    """
    anchor = w._base if w._base is not None else w
    key = (w.data_ptr(), variant)
    stamp = (w._version, tuple(w.shape), w.stride(), w.dtype)
    ent = _U_CACHE.get(key)
    if ent is not None and ent[0]() is anchor and ent[2] == stamp:
        return ent[1]
    k, c = w.shape[0], w.shape[1]
    g = torch.tensor(WINOGRAD_VARIANTS[variant]["G"], device=w.device, dtype=torch.float32)
    u = torch.einsum("ab,kcbd,ed->kcae", g, w.reshape(k, c, 3, 3).float(), g)
    a = g.shape[0]
    u = u.permute(2, 3, 1, 0).reshape(a * a, c, k).contiguous().to(torch.bfloat16)
    _U_CACHE[key] = (weakref.ref(anchor, functools.partial(_evict_u, key)), u, stamp)
    return u


def winograd_supported(x, weight, stride=1, padding=0, dilation=1, groups=1, padding_mode="zeros"):
    """Whether ``conv3d_winograd`` can run this problem at all.

    Cheap enough to call per convolution, so a caller can route between this and
    ``conv3d_implicit`` without catching an exception.
    """
    rank = weight.dim() - 2
    if rank not in (2, 3):
        return False
    ksize = tuple(weight.shape[2:])
    if rank == 3:
        if ksize[0] != 1:
            return False
        ksize = ksize[1:]
    if ksize != (3, 3) or groups != 1 or padding_mode != "zeros":
        return False
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        return False
    if any(v != 1 for v in _as_tuple(stride, rank, "stride")):
        return False
    if any(v != 1 for v in _as_tuple(dilation, rank, "dilation")):
        return False
    if isinstance(padding, str):
        return padding in ("same", "valid")
    return True


def _conv2d_winograd_impl(
    x,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    padding_mode="zeros",
    variant=DEFAULT_VARIANT,
    stream=None,
    input_layout="NCHW",
    output_layout="NCHW",
):
    assert variant in WINOGRAD_VARIANTS, f"variant must be one of {tuple(WINOGRAD_VARIANTS)}, got {variant!r}"
    assert input_layout in ("NCHW", "NHWC"), f"input_layout must be NCHW or NHWC, got {input_layout!r}"
    assert output_layout in ("NCHW", "NHWC"), f"output_layout must be NCHW or NHWC, got {output_layout!r}"

    k, wc, r, s = weight.shape
    assert (r, s) == (3, 3), f"conv3d_winograd is a 3x3 kernel; got filter {(r, s)}"
    assert groups == 1, "conv3d_winograd does not implement groups; use conv3d_implicit"
    assert padding_mode == "zeros", f"conv3d_winograd only implements zero padding, got {padding_mode!r}"
    for name, t in (("x", x), ("weight", weight), ("bias", bias)):
        assert t is None or t.is_cuda, f"conv3d_winograd needs GPU tensors; {name} is on {t.device}"
    assert (
        x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
    ), f"conv3d_winograd is a bf16-only kernel; got x={x.dtype}, weight={weight.dtype}"
    assert bias is None or (
        bias.dim() == 1 and bias.numel() == k
    ), f"bias must be a 1-D tensor of {k} elements, one per output channel"
    assert all(v == 1 for v in _as_tuple(stride, 2, "stride")), "conv3d_winograd is stride-1 only"
    assert all(v == 1 for v in _as_tuple(dilation, 2, "dilation")), "conv3d_winograd is dilation-1 only"

    if input_layout == "NHWC":
        n, h, w, c = x.shape
    else:
        n, c, h, w = x.shape
    assert wc == c, f"weight in-channels {wc} != C = {c}"

    # _resolve_padding works in the 5-D index space the sibling module uses, so a 2-D
    # padding is widened with a zero depth axis first, exactly as _conv2d_impl does.
    if isinstance(padding, str):
        p3 = padding
    else:
        p3 = (0,) + _as_tuple(padding, 2, "padding")
    pad_lo, pad_hi = _resolve_padding(p3, (1, 3, 3), (1, 1, 1), (1, 1, 1))
    assert pad_lo == pad_hi, f"conv3d_winograd needs symmetric padding, got {pad_lo} / {pad_hi}"
    ph, pw = pad_lo[1], pad_lo[2]

    ho = h + 2 * ph - 2
    wo = w + 2 * pw - 2
    assert min(ho, wo) >= 1, f"filter is larger than the padded input: output ({ho}, {wo})"

    m_tile = WINOGRAD_VARIANTS[variant]["m"]
    a = m_tile + 2
    p_t = ((ho + m_tile - 1) // m_tile) * ((wo + m_tile - 1) // m_tile)
    npq = n * p_t
    # Rounding the tile axis up to whole blocks buys the transforms a branchless tile
    # index; the extra rows are all-zero and cost only their share of the GEMM.
    np_pad = (npq + TILES_PER_BLOCK - 1) // TILES_PER_BLOCK * TILES_PER_BLOCK

    if n == 0:
        empty = (0, ho, wo, k) if output_layout == "NHWC" else (0, k, ho, wo)
        return torch.empty(empty, device=x.device, dtype=torch.bfloat16)

    launch_stream = torch.cuda.current_stream() if stream is None else stream
    if input_layout == "NHWC":
        x_nhwc = x.contiguous()
    else:
        x_nhwc = _ncdhw_to_ndhwc(x.reshape(n, c, 1, h, w).contiguous(), launch_stream)
    x_nhwc = x_nhwc.reshape(n, h, w, c)

    u = _prep_weight_winograd(weight, variant)

    v = torch.empty((a * a, np_pad, c), device=x.device, dtype=torch.bfloat16)
    _dispatch(
        compile_winograd_input_transform(n, h, w, c, ph, pw, ho, wo, np_pad, variant),
        v,
        x_nhwc,
        stream=launch_stream,
    )

    mm = torch.bmm(v, u)

    has_bias = bias is not None
    bias_arg = bias.to(torch.float32).contiguous() if has_bias else _bias_placeholder(x.device)
    y = torch.empty((n, ho, wo, k), device=x.device, dtype=torch.bfloat16)
    _dispatch(
        compile_winograd_output_transform(n, ho, wo, k, np_pad, has_bias, variant),
        y,
        mm,
        bias_arg,
        stream=launch_stream,
    )

    if output_layout == "NHWC":
        return y
    return y.permute(0, 3, 1, 2).contiguous()


def conv3d_winograd(x, weight, bias=None, stride=1, padding=0, dilation=1, **kwargs):
    """Winograd conv entry; dispatches by filter rank the way ``conv3d_implicit`` does.

    Accepts a rank-2 filter ``(K, C, 3, 3)`` directly, and a rank-3 filter
    ``(K, C, 1, 3, 3)`` -- the shape ``conv3d_implicit``'s 2-D path builds -- by dropping
    the depth axis. Everything else raises, since Winograd is only defined here for the
    3x3 / stride-1 / dilation-1 / groups-1 case; ``winograd_supported`` answers the same
    question without raising.

    ``variant`` selects the tile size: "F2x2_3x3" (default, exact +-1 transforms) or
    "F4x4_3x3" (4x fewer multiplies, materially worse in bf16). ``input_layout`` and
    ``output_layout`` are "NCHW"/"NHWC" and independent; NHWC is native on both sides, so
    an NCHW input costs a pre-transpose and an NCHW output a post-permute.
    """
    spatial_rank = weight.dim() - 2
    if spatial_rank not in (2, 3):
        raise ValueError(f"conv3d_winograd supports 2D/3D filters; got filter rank {weight.dim()}")
    unbatched = x.dim() == weight.dim() - 1
    if unbatched:
        x = x.unsqueeze(0)
    assert x.dim() == weight.dim(), f"x rank {x.dim()} != weight rank {weight.dim()}"

    # A depth-1 3-D problem is run as the 2-D one it is: the depth axis is dropped from
    # every rank-3 argument here and put back on the result, so the caller still sees the
    # 5-D shape torch.conv3d would have returned.
    squeeze_depth = spatial_rank == 3
    out_depth_axis = 0
    if squeeze_depth:
        if weight.shape[2] != 1:
            raise ValueError(
                f"conv3d_winograd implements 2-D Winograd only; a 3-D filter needs depth extent 1, "
                f"got {weight.shape[2]}"
            )
        weight = weight.reshape(weight.shape[0], weight.shape[1], *weight.shape[3:])
        in_layout = kwargs.get("input_layout", "NCDHW")
        out_layout = kwargs.get("output_layout", "NCDHW")
        in_depth_axis = 1 if in_layout == "NDHWC" else 2
        out_depth_axis = 1 if out_layout == "NDHWC" else 2
        assert x.shape[in_depth_axis] == 1, f"3-D input needs depth extent 1, got {x.shape[in_depth_axis]}"
        x = x.reshape(*x.shape[:in_depth_axis], *x.shape[in_depth_axis + 1 :])
        for name, layout in (("input_layout", in_layout), ("output_layout", out_layout)):
            if name in kwargs:
                kwargs[name] = layout.replace("D", "")
        stride = _as_tuple(stride, 3, "stride")[1:]
        dilation = _as_tuple(dilation, 3, "dilation")[1:]
        if not isinstance(padding, str):
            padding = _as_tuple(padding, 3, "padding")[1:]

    y = _conv2d_winograd_impl(x, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, **kwargs)
    if squeeze_depth:
        y = y.unsqueeze(out_depth_axis)
    return y.squeeze(0) if unbatched else y
