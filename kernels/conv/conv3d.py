# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Front door for the bf16 convolutions: picks implicit-GEMM or Winograd per problem.

``conv3d`` takes ``conv3d_implicit``'s signature and forwards to whichever backend is
faster for the shape in hand. ``conv3d_implicit`` is the general path -- every rank,
stride, dilation, group count and padding mode -- and ``conv3d_winograd`` is a specialist
that only exists for 3x3 / stride-1 / dilation-1 / dense / zero-padded 2-D convolutions.
So the first question is whether Winograd applies at all, and the second is whether it
wins.

The choice rule, and where it comes from
----------------------------------------
Measured on gfx950/MI350X over 137 shapes (N in 1..8, C and K in 64..2048, spatial
4..112), comparing the two backends directly:

1. **Input channels decide it.** Winograd wins iff ``C >= 512``, and K barely matters:
   at C=512 it wins for every K from 64 to 2048 (1.15-1.51x), and at C<=256 it loses for
   every K (0.47-1.00x). That asymmetry is structural -- Winograd's 4x traffic blowup is
   on V, which is sized by C, while implicit-GEMM's K-loop is also walked by C, so raising
   C helps one and hurts the other. The separation is clean: 40/40 rows on the C!=K sweep.

2. **Tiny tile counts break it above C=512.** ``np`` is rounded up to a whole block of 32
   tiles, and each wasted row still costs a full ``C x K`` column of the batched GEMM.
   At C=512 that is cheap enough to ignore, but at C >= 1024 a problem with fewer than 32
   real tiles pays more in padding than it saves in arithmetic.

3. **Eager mode has a host floor.** Winograd is three device kernels plus two temporaries
   where implicit-GEMM is one kernel, which pins its wall time near 62us no matter how
   small the problem (see ``conv-small-shapes-host-bound``). Under CUDA-graph capture that
   cost is amortized and rule 1 is the whole story; outside capture the convolution must
   be big enough that device time dominates. ``EAGER_MIN_MACS`` is where that crossover
   sits, chosen so no measured shape regresses.

Rules 1 and 2 are device-time rules and hold under capture. Rule 3 only applies eagerly.
Together they picked the faster backend on every measured shape; the cost is that two
eager wins (1.04x and 1.96x) are left on the table by rule 3's deliberately safe
threshold.

Pass ``impl="implicit"`` or ``impl="winograd"`` to bypass all of this, and
``conv3d_select`` to ask what ``impl="auto"`` would do without running anything.
"""

import math

import torch

from kernels.conv.conv3d_implicit import _as_tuple, conv3d_implicit
from kernels.conv.conv3d_winograd import (
    DEFAULT_VARIANT,
    TILES_PER_BLOCK,
    WINOGRAD_VARIANTS,
    conv3d_winograd,
    winograd_supported,
)

# Winograd's V blowup is sized by C, and so is implicit-GEMM's K-loop, so this one
# threshold separates the two backends across every K measured.
MIN_CHANNELS = 512

# Above MIN_CHANNELS, a problem with fewer real tiles than one block pays more for the
# padded-up rows of the batched GEMM than the transform saves.
MIN_TILES = TILES_PER_BLOCK

# Eager-mode floor: Winograd's two extra dispatches cost ~30us of host time, so outside
# CUDA-graph capture the convolution has to be large enough for device time to dominate.
EAGER_MIN_MACS = 1e10


def _spatial_dims(x, weight, layout):
    """(n, c, h, w) for a batched-or-not input in either layout."""
    rank = weight.dim() - 2
    batched = x.dim() == weight.dim()
    n = x.shape[0] if batched else 1
    if layout.endswith("C"):
        return n, x.shape[-1], x.shape[-3], x.shape[-2]
    return n, x.shape[-4] if rank == 3 else x.shape[-3], x.shape[-2], x.shape[-1]


def _hw_padding(padding, rank):
    """(ph, pw) for a 3x3 filter. Only called once Winograd applicability is established."""
    if isinstance(padding, str):
        return (1, 1) if padding == "same" else (0, 0)
    return _as_tuple(padding, rank, "padding")[-2:]


def conv3d_select(x, weight, stride=1, padding=0, dilation=1, groups=1, **kwargs):
    """Which backend ``conv3d(..., impl="auto")`` would use: "implicit" or "winograd".

    Pure inspection -- it launches nothing, so it is cheap enough to call per convolution
    and is the same code path the dispatch itself takes.
    """
    padding_mode = kwargs.get("padding_mode", "zeros")
    if not winograd_supported(x, weight, stride, padding, dilation, groups, padding_mode):
        return "implicit"

    rank = weight.dim() - 2
    layout = kwargs.get("input_layout", "NCDHW" if rank == 3 else "NCHW")
    n, c, h, w = _spatial_dims(x, weight, layout)
    if c < MIN_CHANNELS:
        return "implicit"

    ph, pw = _hw_padding(padding, rank)
    ho, wo = h + 2 * ph - 2, w + 2 * pw - 2
    if ho <= 0 or wo <= 0:
        return "implicit"

    m = WINOGRAD_VARIANTS[kwargs.get("variant", DEFAULT_VARIANT)]["m"]
    tiles = n * math.ceil(ho / m) * math.ceil(wo / m)
    if c > MIN_CHANNELS and tiles < MIN_TILES:
        return "implicit"

    # Under graph capture the extra dispatches are free, so the device-time rule stands.
    capturing = x.is_cuda and torch.cuda.is_current_stream_capturing()
    if not capturing:
        macs = n * ho * wo * c * weight.shape[0] * 9
        if macs < EAGER_MIN_MACS:
            return "implicit"
    return "winograd"


def conv3d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, impl="auto", **kwargs):
    """Convolution, routed to the faster of the implicit-GEMM and Winograd backends.

    Takes ``conv3d_implicit``'s arguments and returns what it would return, so this is a
    drop-in for it; ``impl`` is the only addition. Both backends produce the same result
    up to rounding, but not the same rounding: Winograd's transforms cost roughly 3x the
    relative error of a direct product in bf16 (~5e-3 against fp32, versus ~1.7e-3 for
    implicit-GEMM). If a caller needs bit-reproducible output across shapes, pin ``impl``.

    ``impl`` is "auto" (the measured rule, documented at the top of this module),
    "implicit", or "winograd". The explicit names bypass the rule but not the backend's
    own constraints -- asking for "winograd" on a 5x5 filter raises.

    Backend-specific keywords pass through: ``input_layout`` / ``output_layout`` are
    understood by both, ``variant`` only by Winograd, and ``autotune`` / ``tile`` only by
    implicit-GEMM. Sending a keyword to the backend that does not take it raises, so pin
    ``impl`` when passing one.
    """
    if impl not in ("auto", "implicit", "winograd"):
        raise ValueError(f"impl must be 'auto', 'implicit' or 'winograd', got {impl!r}")

    if impl == "auto":
        impl = conv3d_select(x, weight, stride, padding, dilation, groups, **kwargs)

    if impl == "winograd":
        # groups rides along so an explicit impl="winograd" on a grouped conv raises in the
        # backend rather than silently ignoring it.
        return conv3d_winograd(
            x,
            weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            **kwargs,
        )
    return conv3d_implicit(
        x,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        **kwargs,
    )
