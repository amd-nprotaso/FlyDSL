"""Benchmark the gfx950 dualwave flash-attention kernel WITH AN ATTENTION SINK
across every Q mode and KV source: dense, packed-varlen, paged KV, and split-K.

WHAT THIS MEASURES, AND WHY IT IS NOT SHAPED LIKE bench_flash_attn_aiter_bias.py.
There is no aiter sink baseline to compare against on this hardware: aiter
exposes the sink only through fmha_fwd_with_sink_asm / fmha_fwd_with_sink_varlen_asm,
which are gfx1250 ASM kernels and do not run on gfx950. So the primary axis here
is not fly-vs-aiter, it is the *cost of the sink itself*:

  sink ms    flash_attn_gfx950 built with has_sink=True.
  base ms    the same build with has_sink=False, same inputs, same shapes.
             This is a true like-for-like pair -- identical work, one extra
             epilogue term -- so `ovh` below is the honest cost of the feature.
  aiter ms   aiter flash-attn WITHOUT a sink. An external reference point only.
             It is NOT computing the same thing as the sink column; do not read
             `aiter TF` as a sink comparison. It is here so the fly numbers can
             be placed against a known kernel on the same shapes.

Columns:
  GFLOP     the work this row actually does, counting only unmasked
            (query, key) pairs. READ THIS BEFORE COMPARING ROWS -- varlen rows
            pack sequences of differing length (VARLEN_FRACTIONS), so a varlen
            row does strictly LESS work than the dense row on the same (B, seq)
            line and their ms are not comparable. TFLOPS is per-row work over
            per-row time, so TFLOPS is comparable; ms is not.
  ovh       sink_ms / base_ms. The sink is a single epilogue term -- one fmax,
            two exp2, one fma per row, hoisted out of the kv loop, plus one
            wave-uniform dword load per q head -- so this should sit at ~1.00
            and drift toward 1.00 as S grows and the O(S^2) inner loop dominates
            the O(1) epilogue. A row meaningfully above 1.00 at large S is a
            finding, not noise.
  vs dns    this row's sink TF over the *dense* sink TF at the same (B, seq,
            causal). Work-normalized, so it is meaningful for every mode
            including varlen. 1.00 by definition on dense rows; below 1.00 means
            the mode costs throughput relative to plain dense attention.

Unlike the bias benchmark there are no addressing SKIPs: the sink is an
(num_heads,) fp32 table, so it can never exceed the 4 GiB buffer-descriptor
clamp or the i32 offset range the way a dense (Sq, Skv) bias can. Only the
split-K workspace is capped.

FLOPs = 4 * unmasked_pairs * D * H (2 for QK^T + 2 for P@V), matching the repo's
_flops helper in tests/kernels/test_flash_attn_fwd.py, so causal rows are not
inflated and varlen sums per sequence. The sink adds one extra logit to the
softmax denominator, not a matmul, so it is not counted -- which is precisely
why `ovh` rather than TFLOPS is the number to read for its cost.

Because the timed path hand-builds the kernel's positional ABI (see make_args),
a selftest always runs first and asserts, per mode AND per build, that the
hand-built call is bitwise identical to the high-level launcher, aborting if not.
A wrong ABI slot would otherwise show up as a suspiciously fast row rather than
an error -- and here it would also silently corrupt `ovh`, since that ratio is
only meaningful if both halves ran the work they claim to.

Usage:
    PYTHONPATH=/var/home/FlyDSL python3 bench_flash_attn_sink.py
    PYTHONPATH=/var/home/FlyDSL python3 bench_flash_attn_sink.py 1 2   # batch sizes
"""

import sys

import torch

import flydsl.expr as fx
from kernels.attention.flash_attn_gfx950 import build_flash_attn_dualwave_swp_module
from kernels.attention.flash_attn_utils import dualwave_splitk_workspace_elems

try:
    from aiter.ops.mha import flash_attn_func as aiter_flash_attn_func
    from aiter.ops.mha import flash_attn_varlen_func as aiter_flash_attn_varlen_func
except Exception as e:  # aiter not installed / import failure
    aiter_flash_attn_func = None
    aiter_flash_attn_varlen_func = None
    print(f"[warn] aiter unavailable, skipping aiter comparison: {e}", file=sys.stderr)

torch.manual_seed(0)

OUT_PATH = "bench_flash_attn_sink.txt"
_out_fh = open(OUT_PATH, "w", buffering=1)  # line-buffered so rows survive a crash


def dbg(msg):
    # Flush immediately: a GPU memory-access fault aborts the process, so the
    # last flushed line tells us exactly which call died.
    print(f"[dbg] {msg}", file=sys.stderr, flush=True)


def emit(line):
    print(line, flush=True)
    _out_fh.write(line + "\n")
    _out_fh.flush()


D = 128
H = 32  # q heads
H_KV = 8  # kv heads (GQA)
dtype = torch.bfloat16

SEQ_LENS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
CAUSAL = [False, True]
MODES = ["dense", "varlen", "paged", "splitk"]
BATCHES = [int(x) for x in sys.argv[1:] if x.isdigit()] or [1, 2, 3]

# Per-batch seqlen fractions for varlen rows, cycled. The first is 1.0 so
# max_seqlen_q == seq_len and grid_y matches the dense row of the same seq_len.
VARLEN_FRACTIONS = [1.0, 0.5, 0.75, 0.25]

PAGE_SIZE = 64  # must equal traits.BLOCK_N; the block table is indexed by tile
SPLITK_SPLITS = 4

WORKSPACE_MAX_BYTES = 4 * 2**30  # cap the split-K workspace so it cannot OOM the run

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)


def iters_for(S):
    """Fewer repeats at large S: full attention is O(S^2)."""
    if S <= 4096:
        return 20, 30
    if S <= 16384:
        return 5, 10
    return 2, 5


def varlen_seqlens(B, S):
    """Deterministic per-batch Q/KV lengths for a varlen row of nominal length S."""
    return [max(1, int(S * VARLEN_FRACTIONS[i % len(VARLEN_FRACTIONS)])) for i in range(B)]


def splitk_fits(B, S):
    """Is the fp32 split-K workspace small enough to allocate?"""
    nbytes = dualwave_splitk_workspace_elems(B, H, S, SPLITK_SPLITS, head_dim=D) * 4
    if nbytes > WORKSPACE_MAX_BYTES:
        return False, f"split-K workspace {nbytes / 2**30:.1f} GiB exceeds {WORKSPACE_MAX_BYTES / 2**30:.0f} GiB cap"
    return True, ""


def row_fits(B, S, mode):
    """Per-row feasibility. The sink is (H,) fp32 and never constrains a row, so
    unlike the bias benchmark only the split-K workspace can rule one out."""
    if mode == "splitk":
        return splitk_fits(B, S)
    return True, ""


def flops(seqlens_q, causal):
    """FLOPs over unmasked (query, key) pairs; Sq == Skv per sequence here, so
    bottom-right causal keeps i+1 keys for query i. Varlen sums per sequence,
    which is why a varlen row's work differs from the dense row above it."""
    valid = sum((s * (s + 1) // 2) if causal else (s * s) for s in seqlens_q)
    return 4.0 * valid * D * H


def build_inputs(B, S, mode):
    """Allocate the tensors for one row and describe how to launch it.

    Returns a dict with the kernel inputs, the build kwargs, the launch kwargs,
    and `seqlens` (the per-sequence Q lengths that define this row's work).
    """
    dev = "cuda"
    varlen = mode == "varlen"
    paged = mode == "paged"

    # Per-q-head fp32 sink. The value does not affect timing (it is data, not
    # control flow); a moderate draw just keeps the epilogue off any denormal
    # slow path.
    sink = torch.randn(H, dtype=torch.float32, device=dev)

    if varlen:
        seqlens = varlen_seqlens(B, S)
        cu = [0]
        for s in seqlens:
            cu.append(cu[-1] + s)
        total = cu[-1]
        cu_t = torch.tensor(cu, dtype=torch.int32, device=dev)
        q = torch.randn(total, H, D, dtype=dtype, device=dev)
        k = torch.randn(total, H_KV, D, dtype=dtype, device=dev)
        v = torch.randn(total, H_KV, D, dtype=dtype, device=dev)
        out = torch.empty(total, H, D, dtype=dtype, device=dev)
        return dict(
            seqlens=seqlens,
            q=q,
            k=k,
            v=v,
            sink=sink,
            out=out,
            cu=cu_t,
            block_table=None,
            block_table_stride=0,
            workspace=None,
            build=dict(varlen=True),
            launch=dict(cu_seqlens_q=cu_t, cu_seqlens_kv=cu_t),
        )

    # dense / paged / splitk all use dense Q.
    seqlens = [S] * B
    q = torch.randn(B, S, H, D, dtype=dtype, device=dev)
    out = torch.empty(B, S, H, D, dtype=dtype, device=dev)

    if paged:
        # Paged KV: [num_pages, PAGE_SIZE, H_KV, D] plus a shuffled block table,
        # so a page-order bug cannot alias into a plausible timing.
        pages_per_seq = (S + PAGE_SIZE - 1) // PAGE_SIZE
        total_pages = B * pages_per_seq
        k = torch.randn(total_pages, PAGE_SIZE, H_KV, D, dtype=dtype, device=dev)
        v = torch.randn(total_pages, PAGE_SIZE, H_KV, D, dtype=dtype, device=dev)
        bt = torch.randperm(total_pages, device=dev).view(B, pages_per_seq).to(torch.int32)
        bt_flat = bt.contiguous().reshape(-1)
        return dict(
            seqlens=seqlens,
            q=q,
            k=k,
            v=v,
            sink=sink,
            out=out,
            cu=None,
            block_table=bt_flat,
            block_table_stride=pages_per_seq,
            workspace=None,
            build=dict(paged=True),
            launch=dict(block_table=bt_flat, block_table_stride=pages_per_seq),
        )

    k = torch.randn(B, S, H_KV, D, dtype=dtype, device=dev)
    v = torch.randn(B, S, H_KV, D, dtype=dtype, device=dev)
    if mode == "splitk":
        ws = torch.zeros(
            dualwave_splitk_workspace_elems(B, H, S, SPLITK_SPLITS, head_dim=D),
            dtype=torch.float32,
            device=dev,
        )
        return dict(
            seqlens=seqlens,
            q=q,
            k=k,
            v=v,
            sink=sink,
            out=out,
            cu=None,
            block_table=None,
            block_table_stride=0,
            workspace=ws,
            build=dict(num_kv_splits=SPLITK_SPLITS),
            launch=dict(workspace=ws),
        )

    return dict(
        seqlens=seqlens,
        q=q,
        k=k,
        v=v,
        sink=sink,
        out=out,
        cu=None,
        block_table=None,
        block_table_stride=0,
        workspace=None,
        build={},
        launch={},
    )


def make_exe(causal, inp, with_sink):
    return build_flash_attn_dualwave_swp_module(
        num_heads=H,
        head_dim=D,
        causal=causal,
        dtype_str="bf16",
        num_kv_heads=H_KV,
        has_sink=with_sink,
        **inp["build"],
    )


def make_args(B, S, inp, stream, with_sink):
    """The positional ABI that _compile builds internally.

    `out` is the placeholder for tensors this build never reads, matching the
    kernel's own convention; the cu_seqlens / BlockTable / workspace slots carry
    real tensors only in the modes that read them, and the Sink slot carries the
    real table only when has_sink=True (a has_sink=False build never reads it,
    exactly as _prep_sink(None, O) would arrange). Kept in one place so the
    selftest below covers every mode's and both builds' slot assignment.
    """
    out = inp["out"]
    return (
        inp["q"],
        inp["k"],
        inp["v"],
        out,
        out,  # LSE
        inp["workspace"] if inp["workspace"] is not None else out,  # DebugCounts / split-K workspace
        inp["cu"] if inp["cu"] is not None else out,  # CuSeqQ
        inp["cu"] if inp["cu"] is not None else out,  # CuSeqKv
        inp["block_table"] if inp["block_table"] is not None else out,
        out,  # Bias: these builds are has_bias=False, so it is never read
        out,  # AlibiSlopes: these builds are has_alibi=False, so it is never read
        inp["sink"] if with_sink else out,
        B,
        S,  # seq_len (varlen: max_seqlen_q, which sizes grid_y)
        S,  # seq_len_kv
        H * D,  # stride_q_n
        H_KV * D,  # stride_kv_n
        D,  # head_dim_runtime
        inp["block_table_stride"],
        0,  # bias_stride0
        0,  # alibi_stride_b
        fx.Stream(stream),
    )


def compiled_call(B, S, inp, stream, with_sink, causal):
    """Precompile one build and return a zero-overhead positional caller.

    Times the precompiled executable, not the _launch wrapper: the wrapper
    re-derives its JIT cache key by reflection on every call, which costs a flat
    ~0.12 ms regardless of S -- that swamps the GPU time below S~2048 and would
    report host dispatch cost instead of kernel performance. It would also land
    identically on both halves of `ovh` and wash the ratio out toward 1.00.
    """
    exe = make_exe(causal, inp, with_sink)
    launch = dict(inp["launch"], stream=stream)
    if with_sink:
        launch["sink"] = inp["sink"]
    compiled = exe.compile(inp["q"], inp["k"], inp["v"], inp["out"], B, S, **launch)
    args = make_args(B, S, inp, stream, with_sink)

    def call():
        compiled(*args)
        return inp["out"]

    return call


def time_interleaved(calls, warmup, iters):
    """Time several calls with their samples interleaved; returns a mean per call.

    Do NOT time one kernel to completion and then the next. The GPU clocks
    settle downward as a run heats up, so whichever call is measured first comes
    out systematically faster and the ratio between them is biased. Measured on
    this machine, dense S=8192 non-causal, the *same* sink kernel timed 0.9898 ms
    when measured first and 1.0488 ms when measured second -- a 6% swing that
    showed up as ovh 0.947 (sink "faster" than base) purely from ordering.
    Alternating the samples exposes every call to the same drift and brings that
    row to ovh 0.998.
    """
    for c in calls:
        for _ in range(warmup):
            c()
    torch.cuda.synchronize()
    acc = [[] for _ in calls]
    for _ in range(iters):
        for j, c in enumerate(calls):
            torch.cuda.synchronize()
            start.record()
            c()
            end.record()
            torch.cuda.synchronize()
            acc[j].append(start.elapsed_time(end))
    return [sum(a) / len(a) for a in acc]


def selftest(S=512, B=2, causal=True):
    """Assert the hand-built ABI matches the high-level launcher, per mode and
    per build.

    The timed path calls the precompiled executable positionally; a wrong slot
    would silently benchmark the wrong work instead of raising. Both the
    has_sink=True and has_sink=False builds are checked, because `ovh` compares
    them directly and is only meaningful if each ran what it claims to.
    """
    ok = True
    for mode in MODES:
        fits, why = row_fits(B, S, mode)
        if not fits:
            print(f"[selftest] {mode:>7}: SKIP ({why})", file=sys.stderr, flush=True)
            continue
        for with_sink in (True, False):
            inp = build_inputs(B, S, mode)
            exe = make_exe(causal, inp, with_sink)
            stream = torch.cuda.current_stream()
            launch = dict(inp["launch"], stream=stream)
            if with_sink:
                launch["sink"] = inp["sink"]
            exe(inp["q"], inp["k"], inp["v"], inp["out"], B, S, **launch)
            torch.cuda.synchronize()
            ref = inp["out"].clone()
            inp["out"].zero_()
            torch.cuda.synchronize()
            compiled = exe.compile(inp["q"], inp["k"], inp["v"], inp["out"], B, S, **launch)
            compiled(*make_args(B, S, inp, stream, with_sink))
            torch.cuda.synchronize()
            same = torch.equal(ref, inp["out"])
            ok &= same
            tag = "sink" if with_sink else "base"
            print(
                f"[selftest] {mode:>7} {tag}: {'ABI ok (bitwise identical)' if same else 'ABI MISMATCH'}",
                file=sys.stderr,
                flush=True,
            )
            del inp, ref
            torch.cuda.empty_cache()

    # The sink must actually change the output, or `ovh` would be measuring two
    # identical kernels and would report a meaningless 1.00.
    inp = build_inputs(B, S, "dense")
    stream = torch.cuda.current_stream()
    make_exe(causal, inp, True)(inp["q"], inp["k"], inp["v"], inp["out"], B, S, sink=inp["sink"], stream=stream)
    torch.cuda.synchronize()
    o_sink = inp["out"].clone()
    inp["out"].zero_()
    make_exe(causal, inp, False)(inp["q"], inp["k"], inp["v"], inp["out"], B, S, stream=stream)
    torch.cuda.synchronize()
    differs = not torch.equal(o_sink, inp["out"])
    ok &= differs
    verdict = "outputs differ (sink is live)" if differs else "IDENTICAL -- sink is a no-op"
    print(f"[selftest]   dense sink-vs-base: {verdict}", file=sys.stderr, flush=True)
    del inp, o_sink
    torch.cuda.empty_cache()
    return ok


def aiter_callable(S, causal, mode, inp):
    """aiter equivalent for this row, WITHOUT a sink, or None when aiter cannot
    run the mode.

    aiter's sink kernels (fmha_fwd_with_sink_asm / _varlen_asm) are gfx1250 ASM
    and do not run on gfx950, so there is no like-for-like aiter sink call to
    make. These are plain flash-attn timings, present only as a reference point.
    """
    if aiter_flash_attn_func is None:
        return None
    q, k, v = inp["q"], inp["k"], inp["v"]
    if mode == "varlen":
        cu = inp["cu"]
        return lambda: aiter_flash_attn_varlen_func(q, k, v, cu, cu, S, S, causal=causal)
    if mode == "paged":
        # Our paged row keeps Q dense and maps KV tiles through a block table;
        # aiter's paged entry point is a packed-Q batch-prefill, so timing it
        # here would compare two different shapes of work.
        return None
    return lambda: aiter_flash_attn_func(q, k, v, causal=causal)


def run_one(B, S, causal, mode):
    """Time one (batch, seq_len, causal, mode) point.

    Returns (sink_ms, base_ms, aiter_ms, seqlens)."""
    dbg(f"run_one begin B={B} S={S} causal={causal} mode={mode}")
    inp = build_inputs(B, S, mode)
    stream = torch.cuda.current_stream()

    dbg(f"build FlyDSL sink kernel (mode={mode})")
    sink_call = compiled_call(B, S, inp, stream, True, causal)
    dbg(f"build FlyDSL no-sink kernel (mode={mode})")
    base_call = compiled_call(B, S, inp, stream, False, causal)
    aiter_call = aiter_callable(S, causal, mode, inp)
    if aiter_call is not None:
        # Probe once outside the timing loop: a mid-loop throw would discard the
        # sink/base samples collected alongside it.
        try:
            aiter_call()
            torch.cuda.synchronize()
        except Exception as e:
            print(
                f"[warn] aiter call failed for B={B} S={S} causal={causal} mode={mode}: {e}",
                file=sys.stderr,
                flush=True,
            )
            aiter_call = None

    warmup, iters = iters_for(S)
    calls = [sink_call, base_call] + ([aiter_call] if aiter_call is not None else [])
    dbg(f"time {len(calls)} calls interleaved")
    times = time_interleaved(calls, warmup, iters)
    ms_sink, ms_base = times[0], times[1]
    ms_aiter = times[2] if aiter_call is not None else float("nan")
    dbg(f"run_one done B={B} S={S} causal={causal} mode={mode}")

    # No explicit del: the closures hold inp/compiled/args and die with run_one.
    result = (ms_sink, ms_base, ms_aiter, inp["seqlens"])
    torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    if not selftest():
        print("[fatal] ABI selftest failed; timings would be meaningless", file=sys.stderr)
        sys.exit(1)

    hdr = (
        f"{'B':>3} {'seq':>7} {'mode':>7} {'causal':>7} | "
        f"{'GFLOP':>9} | "
        f"{'sink ms':>10} {'base ms':>10} {'aiter ms':>10} | "
        f"{'ovh':>7} {'vs dns':>7} | "
        f"{'sink TF':>9} {'base TF':>9} {'aiter TF':>9}"
    )
    # sink TF of the dense row, keyed by (B, seq, causal); MODES puts dense first
    # so it is always populated before the modes that reference it.
    dense_tf = {}
    emit(f"D={D} H={H} H_KV={H_KV}  dtype={dtype}")
    emit("flash_attn_gfx950(has_sink=True) vs the same kernel with has_sink=False, on identical inputs")
    emit(f"modes: dense | varlen (packed, seqlen fractions {VARLEN_FRACTIONS} of seq)")
    emit(f"       paged (page_size={PAGE_SIZE}, shuffled block table) | splitk (num_kv_splits={SPLITK_SPLITS})")
    emit("sink: (num_heads,) fp32, one extra softmax logit per q head with no matching V row")
    emit("")
    emit("THE AITER COLUMN HAS NO SINK. aiter's sink kernels (fmha_fwd_with_sink_asm and its varlen")
    emit("twin) are gfx1250 ASM and do not run on gfx950, so no like-for-like aiter sink call exists.")
    emit("aiter ms/TF is plain flash-attn on the same shapes -- a reference point, not a comparison.")
    emit("")
    emit("ovh = sink_ms / base_ms, the real cost of the feature: same kernel, same work, one extra")
    emit("epilogue term. The sink is hoisted out of the kv loop, so ovh should sit near 1.00 and")
    emit("tend to 1.00 as S grows and the O(S^2) loop dwarfs the O(1) epilogue.")
    emit("All three columns are timed with their samples INTERLEAVED, not one call at a time:")
    emit("GPU clocks drift down over a run, so timing them sequentially makes whichever went first")
    emit("look up to ~6% faster and biased ovh below 1.00 for no physical reason.")
    emit("GFLOP is this row's actual work (unmasked q,k pairs only). Varlen rows pack shorter")
    emit("sequences, so they do LESS work than the dense row on the same (B,seq) line -- compare")
    emit("TFLOPS across rows, never ms. The sink adds no FLOPs, so it is not in GFLOP; read ovh.")
    emit("vs dns = this row's sink TF / the dense row's sink TF at the same (B,seq,causal).")
    emit("paged has no aiter number (aiter's paged entry point is a packed-Q batch-prefill, a")
    emit("different shape of work); judge it by vs dns.")
    emit("ms times the precompiled executable; the _launch wrapper adds a flat ~0.12 ms/call of")
    emit("host-side JIT cache-key resolution on top, which dominates below S~2048.\n")
    emit(hdr)
    emit("-" * len(hdr))
    for B in BATCHES:
        for S in SEQ_LENS:
            for mode in MODES:
                fits, why = row_fits(B, S, mode)
                if not fits:
                    emit(f"{B:>3} {S:>7} {mode:>7} {'*':>7} | SKIP  {why}")
                    continue
                for causal in CAUSAL:
                    try:
                        ms_s, ms_b, ms_a, seqlens = run_one(B, S, causal, mode)
                    except torch.cuda.OutOfMemoryError:
                        emit(f"{B:>3} {S:>7} {mode:>7} {int(causal):>7} | OOM")
                        torch.cuda.empty_cache()
                        continue
                    fl = flops(seqlens, causal)
                    tf_s = (fl / 1e12) / (ms_s / 1e3)
                    tf_b = (fl / 1e12) / (ms_b / 1e3)
                    have_a = ms_a == ms_a  # not NaN
                    tf_a = (fl / 1e12) / (ms_a / 1e3) if have_a else float("nan")
                    ovh = ms_s / ms_b
                    if mode == "dense":
                        dense_tf[(B, S, causal)] = tf_s
                    # Work-normalized, so varlen's shorter sequences do not skew it.
                    base = dense_tf.get((B, S, causal))
                    rel = tf_s / base if base else float("nan")
                    emit(
                        f"{B:>3} {S:>7} {mode:>7} {int(causal):>7} | "
                        f"{fl / 1e9:>9.1f} | "
                        f"{ms_s:>10.4f} {ms_b:>10.4f} {ms_a:>10.4f} | "
                        f"{ovh:>6.3f}x {rel:>6.2f}x | "
                        f"{tf_s:>9.1f} {tf_b:>9.1f} {tf_a:>9.1f}"
                    )
    _out_fh.close()
