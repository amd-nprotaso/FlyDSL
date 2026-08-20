#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Correctness test for the bf16 Winograd conv kernel.

Compares ``conv3d_winograd`` against ``torch.nn.functional.conv2d``. Winograd's transforms
lose precision relative to a direct product, so the tolerances here are looser than the
implicit-GEMM test's: F(2x2,3x3) lands around 5e-3 relative and F(4x4,3x3) around 3e-2,
which is the reason F(2x2,3x3) is the default variant.
"""

import pytest
import torch
import torch.nn.functional as F

from flydsl.runtime.device import get_rocm_arch
from kernels.conv.conv3d_winograd import conv3d_winograd, winograd_supported

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_ARCH = get_rocm_arch()
_skip_non_cdna4 = pytest.mark.skipif(
    not (isinstance(_ARCH, str) and _ARCH.startswith("gfx95")),
    reason=f"conv3d BF16 needs CDNA4 (gfx95x), got {_ARCH}",
)


def _rel(y, y_ref):
    a, r = y.float().flatten(), y_ref.float().flatten()
    return ((a - r).norm() / r.norm().clamp_min(1e-12)).item()


# (N, C, H, W, K, padding). Covers channel vectorization widths (c % 8, % 4, % 1), odd
# spatial extents that make a ragged tile grid, and padding 0 / 1 / 2.
@_skip_non_cdna4
@pytest.mark.parametrize(
    "n,c,h,w,k,padding",
    [
        (1, 64, 16, 16, 64, 1),
        (1, 64, 16, 16, 64, 0),
        (2, 32, 15, 17, 48, 1),
        (1, 3, 32, 32, 16, 1),
        (1, 20, 13, 13, 12, 1),
        (1, 64, 5, 9, 32, 2),
        (3, 96, 32, 32, 64, 1),
    ],
)
def test_winograd_vs_torch(n, c, h, w, k, padding):
    torch.manual_seed(3000 + h + w + k)
    x = torch.randn((n, c, h, w), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((k, c, 3, 3), device="cuda", dtype=torch.bfloat16) * 0.1

    y = conv3d_winograd(x, weight, padding=padding)
    y_ref = F.conv2d(x, weight, padding=padding)
    torch.cuda.synchronize()

    assert y.shape == y_ref.shape
    assert _rel(y, y_ref) < 2e-2


@_skip_non_cdna4
def test_winograd_f4x4_variant():
    torch.manual_seed(3100)
    x = torch.randn((2, 32, 15, 17), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((48, 32, 3, 3), device="cuda", dtype=torch.bfloat16) * 0.1

    y = conv3d_winograd(x, weight, padding=1, variant="F4x4_3x3")
    y_ref = F.conv2d(x, weight, padding=1)
    torch.cuda.synchronize()

    assert y.shape == y_ref.shape
    # F(4x4,3x3)'s transform coefficients span 1/24..8, so bf16 error is ~5x F(2x2,3x3)'s.
    assert _rel(y, y_ref) < 6e-2


@_skip_non_cdna4
@pytest.mark.parametrize("in_layout", ["NCHW", "NHWC"])
@pytest.mark.parametrize("out_layout", ["NCHW", "NHWC"])
def test_winograd_bias_and_layouts(in_layout, out_layout):
    torch.manual_seed(3200)
    x = torch.randn((2, 64, 16, 16), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((32, 64, 3, 3), device="cuda", dtype=torch.bfloat16) * 0.1
    bias = torch.randn((32,), device="cuda", dtype=torch.bfloat16)

    xi = x if in_layout == "NCHW" else x.permute(0, 2, 3, 1).contiguous()
    y = conv3d_winograd(xi, weight, bias=bias, padding=1, input_layout=in_layout, output_layout=out_layout)
    y_ref = F.conv2d(x, weight, bias=bias, padding=1)
    if out_layout == "NHWC":
        y = y.permute(0, 3, 1, 2)
    torch.cuda.synchronize()

    assert y.shape == y_ref.shape
    assert _rel(y, y_ref) < 2e-2


@_skip_non_cdna4
@pytest.mark.parametrize("padding", ["same", "valid"])
def test_winograd_padding_strings(padding):
    torch.manual_seed(3300)
    x = torch.randn((1, 64, 16, 16), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((64, 64, 3, 3), device="cuda", dtype=torch.bfloat16) * 0.1

    y = conv3d_winograd(x, weight, padding=padding)
    y_ref = F.conv2d(x, weight, padding=padding)
    torch.cuda.synchronize()

    assert y.shape == y_ref.shape
    assert _rel(y, y_ref) < 2e-2


@_skip_non_cdna4
def test_winograd_depth1_3d_filter():
    """A 3-D filter with depth extent 1 runs as 2-D and keeps torch's 5-D output shape."""
    torch.manual_seed(3400)
    x = torch.randn((1, 32, 1, 16, 16), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((48, 32, 1, 3, 3), device="cuda", dtype=torch.bfloat16) * 0.1

    y = conv3d_winograd(x, weight, padding=(0, 1, 1))
    y_ref = F.conv3d(x, weight, padding=(0, 1, 1))
    torch.cuda.synchronize()

    assert y.shape == y_ref.shape
    assert _rel(y, y_ref) < 2e-2


@_skip_non_cdna4
def test_winograd_unbatched():
    torch.manual_seed(3500)
    x = torch.randn((32, 16, 16), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((48, 32, 3, 3), device="cuda", dtype=torch.bfloat16) * 0.1

    y = conv3d_winograd(x, weight, padding=1)
    y_ref = F.conv2d(x, weight, padding=1)
    torch.cuda.synchronize()

    assert y.shape == y_ref.shape
    assert _rel(y, y_ref) < 2e-2


def test_winograd_supported_predicate():
    """The routing predicate accepts only 3x3 / stride 1 / dilation 1 / dense / zero-pad."""
    x = torch.empty((1, 32, 16, 16), dtype=torch.bfloat16)
    w33 = torch.empty((64, 32, 3, 3), dtype=torch.bfloat16)
    w55 = torch.empty((64, 32, 5, 5), dtype=torch.bfloat16)

    assert winograd_supported(x, w33, padding=1)
    assert not winograd_supported(x, w55, padding=2)
    assert not winograd_supported(x, w33, stride=2, padding=1)
    assert not winograd_supported(x, w33, dilation=2, padding=2)
    assert not winograd_supported(x, w33, padding=1, groups=2)
    assert not winograd_supported(x, w33, padding=1, padding_mode="reflect")
