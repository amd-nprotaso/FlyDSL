#!/usr/bin/env python3
"""Run the FlyDSL implicit-GEMM conv kernel on one shape, checked against torch.

Parameters use the same flag format as the traced shape lists under
``kernel-projects/convolution_benchmarking/microbench/shapes/*.txt``:

    conv1d --input 1,128,501 --weight 1536,128,7 --stride 1 --padding 3 \
        --dilation 1 --groups 1 --bias 1 --dtype bfloat16 --padding_mode zeros

Usage
-----
    python3 scripts/run_conv.py conv1d --input 1,128,501 --weight 1536,128,7 --padding 3 --bias 1
    python3 scripts/run_conv.py --line "conv2d --input 1,64,56,56 --weight 128,64,3,3 --padding 1"
    python3 scripts/run_conv.py --file .../microbench/shapes/test.txt

``--without-torch`` drops the reference conv and the accuracy check, so the only
convolution the device sees is FlyDSL's -- what a profiler run wants:

    rocprofv3 -i input.yaml -- python3 scripts/run_conv.py --without-torch \
        conv1d --input 1,128,501 --weight 1536,128,7 --padding 3 --bias 1

The kernel is forward-only and bf16-only, so ``conv_transpose*`` lines are reported as
SKIP and any other ``--dtype`` is run in bf16 anyway.
"""

import argparse
import pathlib
import re
import statistics
import sys

import torch
import torch.nn.functional as F

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from kernels.conv.conv3d_implicit import conv3d_implicit  # noqa: E402

FLAG = re.compile(r"--(\w+)\s+(\S+)")
CONV_FN = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}


def parse_line(line):
    """One shape line -> dict of problem parameters, or None for a comment/blank."""
    body = line.partition("#")[0].strip()
    if not body:
        return None
    op = body.split()[0]
    flags = {name: val for name, val in FLAG.findall(body)}
    if not op.startswith("conv") or "input" not in flags:
        raise ValueError(f"cannot parse line: {body[:80]}")
    rank = int(op[-2])

    def per_axis(name, default):
        vals = [int(v) for v in flags.get(name, str(default)).split(",")]
        return tuple(vals * rank) if len(vals) == 1 else tuple(vals)

    mode = flags.get("padding_mode", "zeros")
    return dict(
        op=op,
        rank=rank,
        raw=body,
        input=[int(v) for v in flags["input"].split(",")],
        weight=[int(v) for v in flags["weight"].split(",")],
        stride=per_axis("stride", 1),
        padding=per_axis("padding", 0),
        dilation=per_axis("dilation", 1),
        groups=int(flags.get("groups", 1)),
        bias=bool(int(flags.get("bias", 0))),
        dtype=flags.get("dtype", "bfloat16"),
        padding_mode="zeros" if mode in ("null", "none", "None") else mode,
    )


def flops(p, y):
    """2 * MACs, taken off the produced output extent."""
    spatial = 1
    for e in y.shape[2:]:
        spatial *= e
    kvol = 1
    for e in p["weight"][2:]:
        kvol *= e
    return 2 * y.shape[0] * y.shape[1] * spatial * (p["input"][1] // p["groups"]) * kvol


def time_ms(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(iters):
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop))
    return statistics.median(samples)


def reference(p, x, w, b):
    """torch's answer. Non-zero padding modes are materialized, as nn.ConvNd does."""
    conv = CONV_FN[p["rank"]]
    if p["padding_mode"] == "zeros":
        return conv(x, w, b, p["stride"], p["padding"], p["dilation"], p["groups"])
    pads = [v for e in reversed(p["padding"]) for v in (e, e)]
    xp = F.pad(x, pads, mode=p["padding_mode"])
    return conv(xp, w, b, p["stride"], 0, p["dilation"], p["groups"])


def run(p, args):
    if "transpose" in p["op"]:
        print(f"SKIP  {p['raw']}\n      conv3d_implicit has no transposed path")
        return None
    torch.manual_seed(0)
    dev = "cuda"
    x = torch.randn(p["input"], device=dev, dtype=torch.bfloat16)
    w = torch.randn(p["weight"], device=dev, dtype=torch.bfloat16) * 0.1
    b = torch.randn(p["weight"][0], device=dev, dtype=torch.bfloat16) if p["bias"] else None

    def fly():
        return conv3d_implicit(
            x,
            w,
            bias=b,
            stride=p["stride"],
            padding=p["padding"],
            dilation=p["dilation"],
            groups=p["groups"],
            padding_mode=p["padding_mode"],
            autotune=args.autotune,
        )

    y = fly()
    print(f"{p['raw']}")

    # --without-torch keeps the reference conv off the device entirely, so a profiler
    # trace holds this kernel's dispatches and nothing from MIOpen.
    if args.without_torch:
        torch.cuda.synchronize()
        print(f"  out       {tuple(y.shape)}")
        if not args.no_bench:
            ms_fly = time_ms(fly, args.warmup, args.iters)
            tf = flops(p, y) / 1e12
            print(f"  flydsl    {ms_fly:9.4f} ms  {tf / (ms_fly / 1e3):7.1f} TFLOP/s")
        return "ran"

    y_ref = reference(p, x, w, b)
    torch.cuda.synchronize()

    a, r = y.float().flatten(), y_ref.float().flatten()
    rel = ((a - r).norm() / r.norm().clamp_min(1e-12)).item()
    cos = torch.nn.functional.cosine_similarity(a, r, dim=0).item()
    ok = "OK" if (rel < 5e-2 and cos > 0.99) else "MISMATCH"

    print(f"  out       {tuple(y.shape)}  ref {tuple(y_ref.shape)}")
    print(f"  accuracy  {ok}  rel={rel:.2e} cos={cos:.6f}")
    if not args.no_bench:
        ms_fly = time_ms(fly, args.warmup, args.iters)
        ms_ref = time_ms(lambda: reference(p, x, w, b), args.warmup, args.iters)
        tf = flops(p, y) / 1e12
        print(
            f"  flydsl    {ms_fly:9.4f} ms  {tf / (ms_fly / 1e3):7.1f} TFLOP/s\n"
            f"  torch     {ms_ref:9.4f} ms  {tf / (ms_ref / 1e3):7.1f} TFLOP/s"
            f"   ({ms_ref / ms_fly:.2f}x)"
        )
    return ok == "OK"


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("op", nargs="?", help="conv1d / conv2d / conv3d")
    ap.add_argument("--input")
    ap.add_argument("--weight")
    ap.add_argument("--stride", default="1")
    ap.add_argument("--padding", default="0")
    ap.add_argument("--dilation", default="1")
    ap.add_argument("--groups", default="1")
    ap.add_argument("--bias", default="0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--padding_mode", default="zeros")
    ap.add_argument("--line", help="a whole shape line, quoted")
    ap.add_argument("--file", help="a shapes .txt; every line is run in order")
    ap.add_argument("--autotune", action="store_true", help="sweep the kernel's tile space")
    ap.add_argument(
        "--without-torch",
        action="store_true",
        help="run only the FlyDSL kernel: no reference conv, no accuracy check (for profile traces)",
    )
    ap.add_argument("--no-bench", action="store_true", help="correctness only")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("-h", "--help", action="help")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "conv3d_implicit needs a GPU"

    if args.file:
        lines = pathlib.Path(args.file).read_text().splitlines()
    elif args.line:
        lines = [args.line]
    else:
        assert args.op and args.input and args.weight, "need <op> --input --weight (or --line/--file)"
        lines = [
            f"{args.op} --input {args.input} --weight {args.weight} --stride {args.stride} "
            f"--padding {args.padding} --dilation {args.dilation} --groups {args.groups} "
            f"--bias {args.bias} --dtype {args.dtype} --padding_mode {args.padding_mode}"
        ]

    results = []
    for line in lines:
        p = parse_line(line)
        if p is None:
            continue
        results.append(run(p, args))
    bad = [r for r in results if r is False]
    tail = f", {results.count('ran')} unchecked" if results.count("ran") else ""
    print(
        f"\n{len(results)} line(s): {results.count(True)} ok, {len(bad)} mismatched, "
        f"{results.count(None)} skipped{tail}"
    )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
