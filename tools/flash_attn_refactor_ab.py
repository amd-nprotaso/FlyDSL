#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""A/B verification and benchmark harness for the ``flash_attn_utils`` refactor.

Implements the verification protocol from the ``flydsl-kernel-refactor`` skill
against a frozen git baseline:

  1. ISA diff (primary gate)          -- ``--phase isa``
  2. Resource counts (VGPR/spill/LDS) -- ``--phase isa``
  3. Numeric equivalence (max_abs)    -- ``--phase numerics``
  5. Interleaved A/B timing           -- ``--phase timing``

The baseline is materialised straight out of git (``--baseline-ref``, default
``HEAD``) into a shadow package ``kernels_base`` so the pre- and post-refactor
kernels can be driven from one harness without touching the working tree.

Why every measurement runs in a subprocess
------------------------------------------
``_jit_function_cache_key`` (flydsl/compiler/jit_function.py) folds dependency
*source text* into the JIT cache key, but only for objects that
``_get_underlying_func`` resolves to a function.  A helper **class** does not
resolve, and a plain ``@flyc.kernel`` body has no ``owner_cls``, so edits to
``DualwaveGemmHelper.qk`` and friends do **not** reliably invalidate the cache.
Running each side in its own process with its own ``FLYDSL_RUNTIME_CACHE_DIR``
makes the comparison sound regardless.  ``--phase selftest`` proves it.

Usage
-----
    python3 tools/flash_attn_refactor_ab.py --phase selftest   # prove the harness works
    python3 tools/flash_attn_refactor_ab.py --phase all        # run every gate
    python3 tools/flash_attn_refactor_ab.py --phase isa        # the primary gate
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Modules frozen into the shadow baseline package.  This is the transitive set
# that participates in the flash_attn_utils refactor: the file being refactored
# plus every module that imports it.
FROZEN_MODULES = (
    "flash_attn_utils",
    "flash_attn_gfx950",
    "flash_attn_fp8_gfx950",
    "flash_attn_generic",
    "swa_gfx950",
    "flash_attn_interface",
)

SHADOW_PKG = "kernels_base"

# Resource fields pulled out of the final ISA, per skill Verification Protocol s2.
RESOURCE_FIELDS = (
    "vgpr_count",
    "sgpr_count",
    "agpr_count",
    "vgpr_spill_count",
    "sgpr_spill_count",
    "group_segment_fixed_size",
)

# (label, batch, seq_len, num_heads, num_kv_heads, head_dim, causal, num_kv_splits)
DEFAULT_CONFIGS = [
    ("mha_1x1024_h32_d128_causal", 1, 1024, 32, 32, 128, True, 1),
    ("mha_2x2048_h16_d128_causal", 2, 2048, 16, 16, 128, True, 1),
    ("gqa_2x2048_h32kv8_d128_causal", 2, 2048, 32, 8, 128, True, 1),
    ("mha_1x4096_h16_d64_causal", 1, 4096, 16, 16, 64, True, 1),
    ("mha_1x2048_h16_d128_noncausal", 1, 2048, 16, 16, 128, False, 1),
    ("mha_1x1536_h16_d128_nonmult", 1, 1536, 16, 16, 128, True, 1),
    # split-K exercises DualwaveSplitKCombine{Context,Helper}.
    ("splitk_1x2048_h16_d128_causal", 1, 2048, 16, 16, 128, True, 4),
    # fp8 exercises the DualwaveFp8* classes, which share the refactored helpers.
    ("fp8_1x2048_h16_d128_causal", 1, 2048, 16, 16, 128, True, 1, "fp8"),
    ("fp8_1x2048_h16_d128_noncausal", 1, 2048, 16, 16, 128, False, 1, "fp8"),
]


# ---------------------------------------------------------------------------
# Baseline materialisation
# ---------------------------------------------------------------------------


def _git_show(ref: str, path: str) -> str:
    out = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        raise SystemExit(f"git show {ref}:{path} failed:\n{out.stderr}")
    return out.stdout


def _rewrite_imports(text: str) -> str:
    """Point intra-set imports at the shadow package, leave the rest alone.

    ``kernels.common.*`` is deliberately NOT rewritten: it is outside the
    refactor scope, so both sides must share the identical module.
    """
    for mod in FROZEN_MODULES:
        text = text.replace(f"kernels.attention.{mod}", f"{SHADOW_PKG}.attention.{mod}")
    # `from kernels.attention import flash_attn_utils` style.
    text = re.sub(
        r"from kernels\.attention import (" + "|".join(FROZEN_MODULES) + r")\b",
        rf"from {SHADOW_PKG}.attention import \1",
        text,
    )
    return text


def materialize_baseline(ref: str, workdir: Path, *, perturb: bool = False) -> Path:
    """Write a frozen copy of the pre-refactor kernels as an importable package."""
    pkg = workdir / SHADOW_PKG
    if pkg.exists():
        shutil.rmtree(pkg)
    (pkg / "attention").mkdir(parents=True)

    (pkg / "__init__.py").write_text(_git_show(ref, "kernels/__init__.py"))
    (pkg / "attention" / "__init__.py").write_text(_git_show(ref, "kernels/attention/__init__.py"))

    for mod in FROZEN_MODULES:
        src = _git_show(ref, f"kernels/attention/{mod}.py")
        src = _rewrite_imports(src)
        if perturb and mod == "flash_attn_utils":
            src = _inject_perturbation(src)
        (pkg / "attention" / f"{mod}.py").write_text(src)

    return workdir


def _inject_perturbation(src: str) -> str:
    """Nudge the softmax scale so a correct harness MUST see a difference.

    Used only by --phase selftest, to prove the A/B path is not silently
    comparing one binary against itself via a shared JIT cache entry.
    """
    needle = "_LOG2E = host_math.log2(host_math.e)"
    if needle not in src:
        raise SystemExit("selftest: could not find _LOG2E anchor to perturb")
    return src.replace(needle, needle + " * 1.0009765625  # SELFTEST PERTURBATION")


def check_scope(ref: str) -> list[str]:
    """Return working-tree changes under kernels/ that fall outside the frozen set."""
    out = subprocess.run(
        ["git", "diff", "--name-only", ref, "--", "kernels/"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    changed = [p for p in out.stdout.split() if p]
    allowed = {f"kernels/attention/{m}.py" for m in FROZEN_MODULES}
    return [p for p in changed if p not in allowed]


# ---------------------------------------------------------------------------
# Subprocess worker plumbing
# ---------------------------------------------------------------------------


def run_worker(side: str, task: str, payload: dict, workdir: Path, *, env_extra: dict | None = None) -> dict:
    """Run one measurement in an isolated process with its own JIT cache dir."""
    payload_path = workdir / f"_payload_{side}_{task}.json"
    result_path = workdir / f"_result_{side}_{task}.json"
    payload_path.write_text(json.dumps(payload))

    env = dict(os.environ)
    # Per-side cache isolation: the disk cache does not reliably invalidate on
    # helper-class method edits, so never let the two sides share one.
    env["FLYDSL_RUNTIME_CACHE_DIR"] = str(workdir / f"cache_{side}")
    env["PYTHONPATH"] = os.pathsep.join([str(REPO), str(workdir), env.get("PYTHONPATH", "")])
    if env_extra:
        env.update(env_extra)

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_worker",
        side,
        task,
        str(payload_path),
        str(result_path),
    ]
    proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    if proc.returncode != 0 or not result_path.exists():
        raise SystemExit(
            f"worker({side},{task}) failed rc={proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout[-4000:]}\n--- stderr ---\n{proc.stderr[-4000:]}"
        )
    return json.loads(result_path.read_text())


def _worker_main(side: str, task: str, payload_path: str, result_path: str) -> None:
    """Child-process entry point.  ``side`` selects which package to import."""
    import torch

    payload = json.loads(Path(payload_path).read_text())
    pkg = "kernels" if side == "new" else SHADOW_PKG
    mod = __import__(f"{pkg}.attention.flash_attn_interface", fromlist=["flydsl_flash_attn_func"])
    attn = mod.flydsl_flash_attn_func

    def _quant_fp8(x):
        """Per-tensor e4m3fn quantisation, matching the repo test helper."""
        finfo = torch.finfo(torch.float8_e4m3fn)
        amax = x.abs().max().to(torch.float32)
        descale = (amax / finfo.max).clamp(min=1e-12).view(1)
        return (x.to(torch.float32) / descale).to(torch.float8_e4m3fn).contiguous(), descale.contiguous()

    def make_inputs(cfg):
        b, s, h, hkv, d = cfg["batch"], cfg["seq_len"], cfg["heads"], cfg["kv_heads"], cfg["head_dim"]
        torch.manual_seed(cfg["seed"])
        dt = torch.bfloat16 if cfg["dtype"] == "bf16" else torch.float16
        def mk(nh):
            return (torch.rand((b, s, nh, d), device="cuda", dtype=torch.float32) * 2 - 1).to(dt)
        q, k, v = mk(h), mk(hkv), mk(hkv)
        if cfg.get("mode") == "fp8":
            (qq, qd), (kq, kd), (vq, vd) = _quant_fp8(q), _quant_fp8(k), _quant_fp8(v)
            return (qq, kq, vq), {"q_descale": qd, "k_descale": kd, "v_descale": vd}
        return (q, k, v), {}

    def call(inputs, extra, cfg):
        q, k, v = inputs
        return attn(
            q, k, v,
            causal=cfg["causal"],
            num_kv_heads=cfg["kv_heads"],
            num_kv_splits=cfg["splits"],
            **extra,
        )

    result: dict = {}

    if task == "numerics":
        cfg = payload
        inputs, extra = make_inputs(cfg)
        out = call(inputs, extra, cfg)
        torch.cuda.synchronize()
        torch.save(out.float().cpu(), payload["out_path"])
        result = {"ok": True, "shape": list(out.shape), "dtype": str(out.dtype)}

    elif task == "timing":
        cfg = payload
        inputs, extra = make_inputs(cfg)
        for _ in range(cfg["warmup"]):
            call(inputs, extra, cfg)
        torch.cuda.synchronize()
        times = []
        for _ in range(cfg["iters"]):
            st, en = torch.cuda.Event(True), torch.cuda.Event(True)
            st.record()
            call(inputs, extra, cfg)
            en.record()
            torch.cuda.synchronize()
            times.append(st.elapsed_time(en))
        result = {"times_ms": times}

    elif task == "isa":
        cfg = payload
        inputs, extra = make_inputs(cfg)
        call(inputs, extra, cfg)
        torch.cuda.synchronize()
        result = {"ok": True, "dump_dir": os.environ.get("FLYDSL_DUMP_DIR", "")}

    elif task == "reference":
        cfg = payload
        (q, k, v), _extra = make_inputs(cfg)
        if cfg.get("mode") == "fp8":
            # Dequantise (fp8_value * descale) so the reference is comparable.
            q = q.to(torch.float32) * _extra["q_descale"].to(torch.float32)
            k = k.to(torch.float32) * _extra["k_descale"].to(torch.float32)
            v = v.to(torch.float32) * _extra["v_descale"].to(torch.float32)
        import torch.nn.functional as F
        qf, kf, vf = (t.float().transpose(1, 2) for t in (q, k, v))
        if cfg["kv_heads"] != cfg["heads"]:
            rep = cfg["heads"] // cfg["kv_heads"]
            kf = kf.repeat_interleave(rep, dim=1)
            vf = vf.repeat_interleave(rep, dim=1)
        ref = F.scaled_dot_product_attention(qf, kf, vf, is_causal=cfg["causal"])
        torch.save(ref.transpose(1, 2).cpu(), payload["out_path"])
        result = {"ok": True}

    else:
        raise SystemExit(f"unknown worker task {task}")

    Path(result_path).write_text(json.dumps(result))


# ---------------------------------------------------------------------------
# ISA dump handling
# ---------------------------------------------------------------------------


def find_isa_files(dump_root: Path) -> list[Path]:
    return sorted(dump_root.rglob("*final_isa.s")) or sorted(dump_root.rglob("*.s"))


def parse_resources(isa_path: Path) -> dict:
    text = isa_path.read_text(errors="replace")
    found: dict[str, int] = {}
    for field in RESOURCE_FIELDS:
        m = re.findall(rf"\.{field}:\s*(\d+)", text)
        if m:
            found[field] = max(int(x) for x in m)
    return found


def isa_body(isa_path: Path) -> list[str]:
    """ISA text with volatile bits (paths, timestamps, symbol hashes) stripped."""
    lines = []
    for ln in isa_path.read_text(errors="replace").splitlines():
        if re.search(r"\.file|\.ident|/tmp/|\.amdgcn_target|kernel_[0-9a-f]{8,}", ln):
            continue
        lines.append(ln.rstrip())
    return lines


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------


def cfg_dict(row, args, seed=1234):
    label, b, s, h, hkv, d, causal, splits = row[:8]
    mode = row[8] if len(row) > 8 else "bf16"
    return {
        "label": label, "batch": b, "seq_len": s, "heads": h, "kv_heads": hkv,
        "head_dim": d, "causal": causal, "splits": splits, "mode": mode,
        "dtype": args.dtype, "seed": seed,
        "warmup": args.warmup, "iters": args.iters,
    }


def phase_numerics(args, workdir, configs, *, baseline_perturbed=False) -> bool:
    import torch

    print("\n=== Phase: numeric equivalence (required: max_abs == 0.0) ===")
    all_ok = True
    for row in configs:
        cfg = cfg_dict(row, args)
        outs = {}
        for side in ("base", "new"):
            p = workdir / f"out_{side}_{cfg['label']}.pt"
            c = dict(cfg, out_path=str(p))
            run_worker(side, "numerics", c, workdir)
            outs[side] = torch.load(p)

        diff = (outs["base"] - outs["new"]).abs().max().item()
        exact = torch.equal(outs["base"], outs["new"])

        refp = workdir / f"ref_{cfg['label']}.pt"
        run_worker("new", "reference", dict(cfg, out_path=str(refp)), workdir)
        ref = torch.load(refp)
        ref_err = (outs["new"] - ref).abs().max().item()

        ok = (diff == 0.0) and exact
        if baseline_perturbed:
            ok = diff != 0.0  # selftest inverts the expectation
        all_ok &= ok
        verdict = "PASS" if ok else "FAIL"
        print(
            f"  [{verdict}] {cfg['label']:<32} max_abs(base,new)={diff:.6g}  "
            f"bitwise_equal={exact}  max_abs(new,fp32_ref)={ref_err:.4g}"
        )
    return all_ok


def phase_isa(args, workdir, configs) -> bool:
    print("\n=== Phase: ISA diff + resource counts (primary gate) ===")
    all_ok = True
    for row in configs:
        cfg = cfg_dict(row, args)
        dumps = {}
        for side in ("base", "new"):
            d = workdir / f"isa_{side}_{cfg['label']}"
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)
            run_worker(
                side, "isa", cfg, workdir,
                env_extra={
                    "FLYDSL_DUMP_IR": "1",
                    "FLYDSL_DEBUG_DUMP_ASM": "1",
                    "FLYDSL_DUMP_DIR": str(d),
                },
            )
            dumps[side] = d

        fb, fn = find_isa_files(dumps["base"]), find_isa_files(dumps["new"])
        if not fb or not fn:
            print(f"  [WARN] {cfg['label']}: no ISA dumped (base={len(fb)} new={len(fn)})")
            all_ok = False
            continue

        by_base = {p.name: p for p in fb}
        by_new = {p.name: p for p in fn}
        for name in sorted(set(by_base) | set(by_new)):
            if name not in by_base or name not in by_new:
                print(f"  [FAIL] {cfg['label']}: ISA file {name} present on only one side")
                all_ok = False
                continue
            a, b = isa_body(by_base[name]), isa_body(by_new[name])
            if a == b:
                isa_ok, note = True, "identical"
            else:
                isa_ok = False
                d = list(difflib.unified_diff(a, b, "base", "new", n=2, lineterm=""))
                note = f"{sum(1 for x in d if x.startswith(('+', '-')) and not x.startswith(('+++', '---')))} changed lines"
                dp = workdir / f"isadiff_{cfg['label']}_{name}.diff"
                dp.write_text("\n".join(d))
                note += f" -> {dp}"

            ra, rb = parse_resources(by_base[name]), parse_resources(by_new[name])
            regressions = [
                f"{f}: {ra.get(f)} -> {rb.get(f)}"
                for f in RESOURCE_FIELDS
                if f in ra and f in rb and rb[f] > ra[f]
            ]
            res_ok = not regressions
            all_ok &= isa_ok and res_ok
            print(f"  [{'PASS' if isa_ok and res_ok else 'FAIL'}] {cfg['label']}/{name}: ISA {note}")
            print(f"           resources base={ra}")
            print(f"           resources new ={rb}")
            if regressions:
                print(f"           REGRESSIONS: {'; '.join(regressions)}")
    return all_ok


def phase_timing(args, workdir, configs) -> bool:
    print("\n=== Phase: interleaved A/B timing ===")
    print(f"  {args.reps} rounds x {args.iters} iters, alternating order per round\n")
    print(f"  {'config':<34}{'base ms':>12}{'new ms':>12}{'speedup':>10}{'min b/n':>18}")
    for row in configs:
        cfg = cfg_dict(row, args)
        times = {"base": [], "new": []}
        for rep in range(args.reps):
            order = ("base", "new") if rep % 2 == 0 else ("new", "base")
            for side in order:
                times[side].extend(run_worker(side, "timing", cfg, workdir)["times_ms"])
        mb, mn = statistics.median(times["base"]), statistics.median(times["new"])
        lb, ln = min(times["base"]), min(times["new"])
        print(
            f"  {cfg['label']:<34}{mb:>12.4f}{mn:>12.4f}{mb / mn:>9.3f}x"
            f"{f'{lb:.4f}/{ln:.4f}':>18}"
        )
    return True


def phase_selftest(args, workdir) -> bool:
    """Prove the harness detects a known-injected difference."""
    print("\n=== Phase: harness selftest ===")
    print("  Injecting a perturbation into the BASELINE side; numerics MUST differ.")
    print("  If this reports max_abs 0.0, the two sides are sharing a JIT cache")
    print("  entry and every other phase in this harness is meaningless.\n")

    sub = workdir / "selftest"
    sub.mkdir(parents=True, exist_ok=True)
    materialize_baseline(args.baseline_ref, sub, perturb=True)
    ok = phase_numerics(args, sub, DEFAULT_CONFIGS[:2], baseline_perturbed=True)
    print(f"\n  selftest {'PASSED - harness can detect changes' if ok else 'FAILED - harness is blind'}")
    return ok


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", default="all", choices=["all", "isa", "numerics", "timing", "selftest"])
    ap.add_argument("--baseline-ref", default="HEAD", help="git ref for the frozen pre-refactor baseline")
    ap.add_argument("--workdir", default="", help="scratch dir (default: a temp dir)")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "f16"])
    ap.add_argument("--reps", type=int, default=4, help="interleaved timing rounds per side")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--only", default="", help="substring filter on config label")
    ap.add_argument("--_worker", nargs=4, metavar=("SIDE", "TASK", "PAYLOAD", "RESULT"), help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._worker:
        _worker_main(*args._worker)
        return 0

    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="fa_ab_"))
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"workdir:       {workdir}")
    print(f"baseline ref:  {args.baseline_ref}")

    stray = check_scope(args.baseline_ref)
    if stray:
        print("\nWARNING: working-tree changes under kernels/ outside the frozen set;")
        print("         these are NOT captured in the baseline and will skew results:")
        for p in stray:
            print(f"           {p}")

    materialize_baseline(args.baseline_ref, workdir)
    print(f"baseline pkg:  {workdir / SHADOW_PKG}")

    configs = [c for c in DEFAULT_CONFIGS if args.only in c[0]]
    if not configs:
        raise SystemExit(f"no configs match --only {args.only!r}")

    ok = True
    if args.phase == "selftest":
        ok = phase_selftest(args, workdir)
    else:
        if args.phase in ("all", "isa"):
            ok &= phase_isa(args, workdir, configs)
        if args.phase in ("all", "numerics"):
            ok &= phase_numerics(args, workdir, configs)
        if args.phase in ("all", "timing"):
            ok &= phase_timing(args, workdir, configs)

    print(f"\n{'=' * 72}\nOVERALL: {'PASS' if ok else 'FAIL'}\n{'=' * 72}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
