#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Tests for the ``conv3d`` front door that routes between implicit-GEMM and Winograd.

Two things are checked separately: that the routing rule sends each problem to the
backend the measurements say is faster (pure inspection, no GPU work), and that both
routes produce torch's answer.
"""

import pytest
import torch
import torch.nn.functional as F

from flydsl.runtime.device import get_rocm_arch
from kernels.conv.conv3d import MIN_CHANNELS, conv3d, conv3d_select

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_ARCH = get_rocm_arch()
_skip_non_cdna4 = pytest.mark.skipif(
    not (isinstance(_ARCH, str) and _ARCH.startswith("gfx95")),
    reason=f"conv3d BF16 needs CDNA4 (gfx95x), got {_ARCH}",
)


def _meta(n, c, hw, k, rank=2):
    """Empty tensors of the right shape -- conv3d_select only reads metadata."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    xs = (n, c, hw, hw) if rank == 2 else (n, c, 1, hw, hw)
    ws = (k, c, 3, 3) if rank == 2 else (k, c, 1, 3, 3)
    return (
        torch.empty(xs, device=dev, dtype=torch.bfloat16),
        torch.empty(ws, device=dev, dtype=torch.bfloat16),
    )


# Everything Winograd cannot express has to fall through, whatever the channel count.
@pytest.mark.parametrize(
    "kernel,kwargs",
    [
        ((5, 5), {}),
        ((3, 3), {"stride": 2}),
        ((3, 3), {"dilation": 2}),
        ((3, 3), {"groups": 4}),
        ((3, 3), {"padding_mode": "reflect"}),
    ],
)
def test_select_falls_through_when_winograd_cannot_apply(kernel, kwargs):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.empty((8, 1024, 28, 28), device=dev, dtype=torch.bfloat16)
    c_in = 1024 // kwargs.get("groups", 1)
    w = torch.empty((1024, c_in, *kernel), device=dev, dtype=torch.bfloat16)
    assert conv3d_select(x, w, padding=1, **kwargs) == "implicit"


def test_select_fp32_falls_through():
    """The kernels are bf16-only, so an fp32 problem is not Winograd's to take."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.empty((8, 1024, 28, 28), device=dev, dtype=torch.float32)
    w = torch.empty((1024, 1024, 3, 3), device=dev, dtype=torch.float32)
    assert conv3d_select(x, w, padding=1) == "implicit"


# Below MIN_CHANNELS Winograd lost on every measured shape, for every K.
@pytest.mark.parametrize("c", [3, 64, 128, 256])
@pytest.mark.parametrize("k", [64, 512, 2048])
def test_select_shallow_channels_go_implicit(c, k):
    x, w = _meta(8, c, 28, k)
    assert conv3d_select(x, w, padding=1) == "implicit"


# Large and deep: winograd wins on both wall and device time.
@pytest.mark.parametrize(
    "n,c,hw,k",
    [(8, 1024, 14, 1024), (8, 2048, 14, 2048), (1, 2048, 28, 2048), (8, 512, 28, 512)],
)
def test_select_deep_and_large_goes_winograd(n, c, hw, k):
    x, w = _meta(n, c, hw, k)
    assert conv3d_select(x, w, padding=1) == "winograd"


def test_select_small_eager_problem_avoids_winograd_host_floor():
    """Deep but small: winograd is device-faster, yet its extra dispatches lose eagerly."""
    x, w = _meta(1, 512, 14, 512)
    assert conv3d_select(x, w, padding=1) == "implicit"


@pytest.fixture
def capturing(monkeypatch):
    """Pretend the stream is under graph capture, which drops the eager host-cost gate."""
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)


def test_select_channel_threshold_is_on_input_channels(capturing):
    """C decides, not K: C=512/K=64 is Winograd's, C=64/K=512 is not."""
    deep_in, w_narrow_out = _meta(8, MIN_CHANNELS, 28, 64)
    assert conv3d_select(deep_in, w_narrow_out, padding=1) == "winograd"
    shallow_in, w_wide_out = _meta(8, 64, 28, MIN_CHANNELS)
    assert conv3d_select(shallow_in, w_wide_out, padding=1) == "implicit"


def test_select_under_capture_drops_the_eager_size_gate():
    """The same deep-but-small problem routes differently eagerly and under capture."""
    x, w = _meta(1, 1024, 16, 1024)
    assert conv3d_select(x, w, padding=1) == "implicit"


def test_select_under_capture_takes_the_device_win(capturing):
    x, w = _meta(1, 1024, 16, 1024)
    assert conv3d_select(x, w, padding=1) == "winograd"


def test_select_tiny_tile_count_above_threshold_goes_implicit(capturing):
    """C>512 with fewer than one block of tiles pays more in GEMM padding than it saves."""
    x, w = _meta(1, 2048, 4, 2048)
    assert conv3d_select(x, w, padding=1) == "implicit"
    # C == MIN_CHANNELS is cheap enough per padded row that the same tile count still wins.
    x512, w512 = _meta(1, MIN_CHANNELS, 4, MIN_CHANNELS)
    assert conv3d_select(x512, w512, padding=1) == "winograd"


def test_select_honors_channels_last_layout():
    """An NHWC input carries C in the last axis; the rule must read the right one."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.empty((8, 28, 28, 1024), device=dev, dtype=torch.bfloat16)
    w = torch.empty((1024, 1024, 3, 3), device=dev, dtype=torch.bfloat16)
    assert conv3d_select(x, w, padding=1, input_layout="NHWC") == "winograd"


def test_select_unbatched_input():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.empty((64, 28, 28), device=dev, dtype=torch.bfloat16)
    w = torch.empty((512, 64, 3, 3), device=dev, dtype=torch.bfloat16)
    assert conv3d_select(x, w, padding=1) == "implicit"


def test_conv3d_rejects_unknown_impl():
    x, w = _meta(1, 64, 16, 64)
    with pytest.raises(ValueError, match="impl must be"):
        conv3d(x, w, padding=1, impl="winograd_v2")


@_skip_non_cdna4
@pytest.mark.parametrize("impl", ["auto", "implicit", "winograd"])
def test_conv3d_matches_torch(impl):
    torch.manual_seed(4000)
    x = torch.randn((2, 64, 16, 16), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((32, 64, 3, 3), device="cuda", dtype=torch.bfloat16) * 0.1
    bias = torch.randn((32,), device="cuda", dtype=torch.bfloat16)

    y = conv3d(x, weight, bias=bias, padding=1, impl=impl)
    y_ref = F.conv2d(x, weight, bias=bias, padding=1)
    torch.cuda.synchronize()

    assert y.shape == y_ref.shape
    rel = ((y.float() - y_ref.float()).norm() / y_ref.float().norm()).item()
    assert rel < 2e-2


@_skip_non_cdna4
def test_conv3d_auto_route_is_correct_on_a_winograd_shape():
    """A shape the rule sends to Winograd still has to produce torch's answer."""
    torch.manual_seed(4100)
    x = torch.randn((8, 1024, 14, 14), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((1024, 1024, 3, 3), device="cuda", dtype=torch.bfloat16) * 0.02

    assert conv3d_select(x, weight, padding=1) == "winograd"
    y = conv3d(x, weight, padding=1)
    y_ref = F.conv2d(x, weight, padding=1)
    torch.cuda.synchronize()

    assert y.shape == y_ref.shape
    rel = ((y.float() - y_ref.float()).norm() / y_ref.float().norm()).item()
    assert rel < 2e-2


@_skip_non_cdna4
def test_conv3d_forwards_general_problems_to_implicit():
    """A 3-D, strided, grouped conv is outside Winograd's domain but inside conv3d's."""
    torch.manual_seed(4200)
    x = torch.randn((1, 32, 8, 16, 16), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((64, 16, 3, 3, 3), device="cuda", dtype=torch.bfloat16) * 0.1

    y = conv3d(x, weight, stride=2, padding=1, groups=2)
    y_ref = F.conv3d(x, weight, stride=2, padding=1, groups=2)
    torch.cuda.synchronize()

    assert y.shape == y_ref.shape
    assert torch.allclose(y, y_ref, rtol=2e-2, atol=2e-2)


@_skip_non_cdna4
def test_conv3d_explicit_winograd_rejects_grouped():
    """Pinning impl bypasses the rule, not the backend's own constraints."""
    x, w = _meta(1, 512, 16, 512)
    with pytest.raises((AssertionError, ValueError)):
        conv3d(x, w, padding=1, groups=2, impl="winograd")
