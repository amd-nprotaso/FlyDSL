"""Benchmark the gfx950 dualwave flash-attention kernel WITH ALiBi across every
Q mode and KV source: dense, packed-varlen, paged KV, and split-K.

This measures two different things at once, because for ALiBi both are available
and neither alone is the whole story:

  alibi ms   flash_attn_gfx950 built with has_alibi=True.
  base ms    the same build with has_alibi=False, same inputs, same shapes.
             A true like-for-like pair -- identical work, one extra term folded
             into every score element -- so `ovh` is the honest cost of ALiBi.
  aiter ms   aiter flash-attn with THE SAME slopes. Unlike the sink benchmark
             (where no aiter sink kernel exists on gfx950), this is a real
             like-for-like comparison, so `fly/ait` is meaningful -- but only on
             the rows where aiter's ALiBi path is usable at all; see below.

ALiBi follows aiter's contract (ck-tile block_position_encoding.hpp):
alibi_slopes is fp32, positive, and score(i,j) += -slope * |i + Skv - Sq - j|,
added after the 1/sqrt(D) scale, bottom-right aligned like the causal mask.
Slopes here are the 1D (num_heads,) form, broadcast over batch. The 2D
(batch, num_heads) form is not timed separately: it differs only by a nonzero
alibi_stride_b on a single wave-uniform dword load per (batch, q head), which is
below the noise floor of every row in this table.

AITER'S CAUSAL ALiBi PATH IS NOT TRUSTWORTHY ON THIS BUILD. Measured on gfx950:
  - varlen + causal + alibi_slopes HARD-FAULTS the process (GPU core dump).
    Reproduced standalone with no FlyDSL kernel in the picture. It would abort
    this whole run mid-table, so those rows never call aiter at all -- gated by
    AITER_VARLEN_CAUSAL_ALIBI_FATAL below.
  - dense + causal + alibi_slopes runs and returns finite values, but the WRONG
    answer: cos 0.964 against an fp32 reference, degrading per head (see
    verify_flash_attn_alibi.py, which skips it for the same reason). It is timed
    here -- it is the real mha_fwd_bf16_alibi_mask_* kernel doing the real masked
    loop -- but read `fly/ait` on causal dense/splitk rows as indicative only.
    A kernel that is wrong could in principle be wrong because it skipped work.

Columns:
  GFLOP     the work this row actually does, counting only unmasked
            (query, key) pairs. READ THIS BEFORE COMPARING ROWS -- varlen rows
            pack sequences of differing length (VARLEN_FRACTIONS), so a varlen
            row does strictly LESS work than the dense row on the same (B, seq)
            line and their ms are not comparable. TFLOPS is per-row work over
            per-row time, so TFLOPS is comparable; ms is not.
  ovh       alibi_ms / base_ms. Unlike the sink -- which is one epilogue term
            hoisted out of the kv loop, so its overhead vanishes as S grows --
            ALiBi costs one sub plus one fma on EVERY score element, inside the
            kv loop, with no memory traffic (absf lowers to a VOP3 source
            modifier). That is O(S^2) work against an O(S^2) kernel, so ovh
            should be roughly FLAT in S, not decaying toward 1.00. A row where
            ovh climbs with S is a finding; a row near 1.00 means the fold hid
            behind the MFMA chain.
  vs dns    this row's alibi TF over the *dense* alibi TF at the same (B, seq,
            causal). Work-normalized, so it is meaningful for every mode
            including varlen, and it is the only baseline paged rows get.
            1.00 by definition on dense rows.
  fly/ait   alibi TF / aiter TF, both with the same slopes. Above 1.00 means fly
            is faster. nan on paged rows (aiter's paged entry point is a
            packed-Q batch-prefill, a different shape of work) and on varlen
            causal rows (fatal, see above).

Unlike the bias benchmark there are no addressing SKIPs: the slopes are a
(num_heads,) fp32 table, so they can never exceed the 4 GiB buffer-descriptor
clamp or the i32 offset range the way a dense (Sq, Skv) bias can. Only the
split-K workspace is capped.

FLOPs = 4 * unmasked_pairs * D * H (2 for QK^T + 2 for P@V), matching the repo's
_flops helper in tests/kernels/test_flash_attn_fwd.py, so causal rows are not
inflated and varlen sums per sequence. ALiBi adds an fma per score, not a
matmul, so it is not counted -- which is why `ovh` rather than TFLOPS is the
number to read for its cost.

Because the timed path hand-builds the kernel's positional ABI (see make_args),
a selftest always runs first and asserts, per mode AND per build, that the
hand-built call is bitwise identical to the high-level launcher, aborting if not.
A wrong ABI slot would otherwise show up as a suspiciously fast row rather than
an error -- and here it would also silently corrupt `ovh`, since that ratio is
only meaningful if both halves ran the work they claim to.

Usage:
    PYTHONPATH=/var/home/FlyDSL python3 bench_flash_attn_alibi.py
    PYTHONPATH=/var/home/FlyDSL python3 bench_flash_attn_alibi.py 1 2   # batch sizes
"""

import sys

import torch

import flydsl.expr as fx
from kernels.attention.flash_attn_gfx950_new import build_flash_attn_dualwave_swp_module
from kernels.attention.flash_attn_utils import dualwave_splitk_workspace_elems

try:
    from aiter.ops.mha import flash_attn_func as aiter_flash_attn_func
    from aiter.ops.mha import flash_attn_varlen_func as aiter_flash_attn_varlen_func
except Exception as e:  # aiter not installed / import failure
    aiter_flash_attn_func = None
    aiter_flash_attn_varlen_func = None
    print(f"[warn] aiter unavailable, skipping aiter comparison: {e}", file=sys.stderr)

torch.manual_seed(0)

# aiter's varlen + causal + alibi_slopes path faults the GPU (see module
# docstring), which aborts the process rather than raising, so it cannot be
# guarded with try/except -- those rows must never call it. Flip to False to
# re-measure if aiter is ever fixed.
AITER_VARLEN_CAUSAL_ALIBI_FATAL = True

OUT_PATH = "bench_flash_attn_alibi.txt"
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
    """Per-row feasibility. The slopes are (H,) fp32 and never constrain a row,
    so unlike the bias benchmark only the split-K workspace can rule one out."""
    if mode == "splitk":
        return splitk_fits(B, S)
    return True, ""


def flops(seqlens_q, causal):
    """FLOPs over unmasked (query, key) pairs; Sq == Skv per sequence here, so
    bottom-right causal keeps i+1 keys for query i. Varlen sums per sequence,
    which is why a varlen row's work differs from the dense row above it."""
    valid = sum((s * (s + 1) // 2) if causal else (s * s) for s in seqlens_q)
    return 4.0 * valid * D * H


def make_slopes():
    """The canonical geometric ALiBi ladder, 1D (num_heads,) fp32, positive.

    aiter takes positive slopes and negates internally; the kernel folds the
    negation and log2(e) into the loaded value. The magnitudes do not affect
    timing (they are data, not control flow) -- the ladder is used simply
    because it is what a real model passes.
    """
    return torch.tensor(
        [2.0 ** (-((h + 1) * 8.0 / H)) for h in range(H)],
        dtype=torch.float32,
        device="cuda",
    )


def build_inputs(B, S, mode):
    """Allocate the tensors for one row and describe how to launch it.

    Returns a dict with the kernel inputs, the build kwargs, the launch kwargs,
    and `seqlens` (the per-sequence Q lengths that define this row's work).
    """
    dev = "cuda"
    varlen = mode == "varlen"
    paged = mode == "paged"

    slopes = make_slopes()

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
            slopes=slopes,
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
            slopes=slopes,
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
            slopes=slopes,
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
        slopes=slopes,
        out=out,
        cu=None,
        block_table=None,
        block_table_stride=0,
        workspace=None,
        build={},
        launch={},
    )


def make_exe(causal, inp, with_alibi):
    return build_flash_attn_dualwave_swp_module(
        num_heads=H,
        head_dim=D,
        causal=causal,
        dtype_str="bf16",
        num_kv_heads=H_KV,
        has_alibi=with_alibi,
        **inp["build"],
    )


def make_args(B, S, inp, stream, with_alibi):
    """The positional ABI that _compile builds internally.

    `out` is the placeholder for tensors this build never reads, matching the
    kernel's own convention; the cu_seqlens / BlockTable / workspace slots carry
    real tensors only in the modes that read them, and the AlibiSlopes slot
    carries the real table only when has_alibi=True (a has_alibi=False build
    never reads it, exactly as _prep_alibi(None, O) would arrange). Kept in one
    place so the selftest below covers every mode's and both builds' slot
    assignment.

    alibi_stride_b is 0 for both builds: these slopes are 1D (num_heads,), which
    _prep_alibi broadcasts over batch by passing stride 0.
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
        inp["slopes"] if with_alibi else out,
        out,  # Sink: these builds are has_sink=False, so it is never read
        B,
        S,  # seq_len (varlen: max_seqlen_q, which sizes grid_y)
        S,  # seq_len_kv
        H * D,  # stride_q_n
        H_KV * D,  # stride_kv_n
        D,  # head_dim_runtime
        inp["block_table_stride"],
        0,  # bias_stride0
        0,  # alibi_stride_b (1D slopes broadcast over batch)
        fx.Stream(stream),
    )


def compiled_call(B, S, inp, stream, with_alibi, causal):
    """Precompile one build and return a zero-overhead positional caller.

    Times the precompiled executable, not the _launch wrapper: the wrapper
    re-derives its JIT cache key by reflection on every call, which costs a flat
    ~0.12 ms regardless of S -- that swamps the GPU time below S~2048 and would
    report host dispatch cost instead of kernel performance. It would also land
    identically on both halves of `ovh` and wash the ratio out toward 1.00.
    """
    exe = make_exe(causal, inp, with_alibi)
    launch = dict(inp["launch"], stream=stream)
    if with_alibi:
        launch["alibi_slopes"] = inp["slopes"]
    compiled = exe.compile(inp["q"], inp["k"], inp["v"], inp["out"], B, S, **launch)
    args = make_args(B, S, inp, stream, with_alibi)

    def call():
        compiled(*args)
        return inp["out"]

    return call


def time_interleaved(calls, warmup, iters):
    """Time several calls with their samples interleaved; returns a mean per call.

    Do NOT time one kernel to completion and then the next. The GPU clocks
    settle downward as a run heats up, so whichever call is measured first comes
    out systematically faster and the ratio between them is biased. Measured on
    this machine (in the sink benchmark, same harness), the *same* kernel timed
    0.9898 ms when measured first and 1.0488 ms when measured second -- a 6%
    swing that showed up as an ovh below 1.00 purely from ordering. Alternating
    the samples exposes every call to the same drift.
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
    has_alibi=True and has_alibi=False builds are checked, because `ovh`
    compares them directly and is only meaningful if each ran what it claims to.
    """
    ok = True
    for mode in MODES:
        fits, why = row_fits(B, S, mode)
        if not fits:
            print(f"[selftest] {mode:>7}: SKIP ({why})", file=sys.stderr, flush=True)
            continue
        for with_alibi in (True, False):
            inp = build_inputs(B, S, mode)
            exe = make_exe(causal, inp, with_alibi)
            stream = torch.cuda.current_stream()
            launch = dict(inp["launch"], stream=stream)
            if with_alibi:
                launch["alibi_slopes"] = inp["slopes"]
            exe(inp["q"], inp["k"], inp["v"], inp["out"], B, S, **launch)
            torch.cuda.synchronize()
            ref = inp["out"].clone()
            inp["out"].zero_()
            torch.cuda.synchronize()
            compiled = exe.compile(inp["q"], inp["k"], inp["v"], inp["out"], B, S, **launch)
            compiled(*make_args(B, S, inp, stream, with_alibi))
            torch.cuda.synchronize()
            same = torch.equal(ref, inp["out"])
            ok &= same
            tag = "alibi" if with_alibi else " base"
            print(
                f"[selftest] {mode:>7} {tag}: {'ABI ok (bitwise identical)' if same else 'ABI MISMATCH'}",
                file=sys.stderr,
                flush=True,
            )
            del inp, ref
            torch.cuda.empty_cache()

    # ALiBi must actually change the output, or `ovh` would be measuring two
    # identical kernels and would report a meaningless 1.00.
    inp = build_inputs(B, S, "dense")
    stream = torch.cuda.current_stream()
    make_exe(causal, inp, True)(
        inp["q"], inp["k"], inp["v"], inp["out"], B, S, alibi_slopes=inp["slopes"], stream=stream
    )
    torch.cuda.synchronize()
    o_alibi = inp["out"].clone()
    inp["out"].zero_()
    make_exe(causal, inp, False)(inp["q"], inp["k"], inp["v"], inp["out"], B, S, stream=stream)
    torch.cuda.synchronize()
    differs = not torch.equal(o_alibi, inp["out"])
    ok &= differs
    verdict = "outputs differ (ALiBi is live)" if differs else "IDENTICAL -- ALiBi is a no-op"
    print(f"[selftest]   dense alibi-vs-base: {verdict}", file=sys.stderr, flush=True)
    del inp, o_alibi
    torch.cuda.empty_cache()
    return ok


def aiter_callable(S, causal, mode, inp):
    """aiter equivalent for this row, WITH the same slopes, or None when aiter
    cannot run the mode.

    See the module docstring: varlen+causal is fatal and is never called; paged
    has no comparable entry point; dense/splitk causal are timed but are
    numerically wrong on this aiter build.
    """
    if aiter_flash_attn_func is None:
        return None
    q, k, v, slopes = inp["q"], inp["k"], inp["v"], inp["slopes"]
    if mode == "varlen":
        if causal and AITER_VARLEN_CAUSAL_ALIBI_FATAL:
            # Faults the GPU and aborts the process; try/except cannot catch it.
            return None
        cu = inp["cu"]
        return lambda: aiter_flash_attn_varlen_func(q, k, v, cu, cu, S, S, causal=causal, alibi_slopes=slopes)
    if mode == "paged":
        # Our paged row keeps Q dense and maps KV tiles through a block table;
        # aiter's paged entry point is a packed-Q batch-prefill, so timing it
        # here would compare two different shapes of work.
        return None
    return lambda: aiter_flash_attn_func(q, k, v, causal=causal, alibi_slopes=slopes)


def run_one(B, S, causal, mode):
    """Time one (batch, seq_len, causal, mode) point.

    Returns (alibi_ms, base_ms, aiter_ms, seqlens)."""
    dbg(f"run_one begin B={B} S={S} causal={causal} mode={mode}")
    inp = build_inputs(B, S, mode)
    stream = torch.cuda.current_stream()

    dbg(f"build FlyDSL alibi kernel (mode={mode})")
    alibi_call = compiled_call(B, S, inp, stream, True, causal)
    dbg(f"build FlyDSL no-alibi kernel (mode={mode})")
    base_call = compiled_call(B, S, inp, stream, False, causal)
    aiter_call = aiter_callable(S, causal, mode, inp)
    if aiter_call is not None:
        # Probe once outside the timing loop: a mid-loop throw would discard the
        # alibi/base samples collected alongside it.
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
    calls = [alibi_call, base_call] + ([aiter_call] if aiter_call is not None else [])
    dbg(f"time {len(calls)} calls interleaved")
    times = time_interleaved(calls, warmup, iters)
    ms_alibi, ms_base = times[0], times[1]
    ms_aiter = times[2] if aiter_call is not None else float("nan")
    dbg(f"run_one done B={B} S={S} causal={causal} mode={mode}")

    # No explicit del: the closures hold inp/compiled/args and die with run_one.
    result = (ms_alibi, ms_base, ms_aiter, inp["seqlens"])
    torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    if not selftest():
        print("[fatal] ABI selftest failed; timings would be meaningless", file=sys.stderr)
        sys.exit(1)

    hdr = (
        f"{'B':>3} {'seq':>7} {'mode':>7} {'causal':>7} | "
        f"{'GFLOP':>9} | "
        f"{'alibi ms':>10} {'base ms':>10} {'aiter ms':>10} | "
        f"{'ovh':>7} {'vs dns':>7} {'fly/ait':>8} | "
        f"{'alibi TF':>9} {'base TF':>9} {'aiter TF':>9}"
    )
    # alibi TF of the dense row, keyed by (B, seq, causal); MODES puts dense
    # first so it is always populated before the modes that reference it.
    dense_tf = {}
    emit(f"D={D} H={H} H_KV={H_KV}  dtype={dtype}")
    emit("flash_attn_gfx950(has_alibi=True) vs the same kernel with has_alibi=False, and vs aiter")
    emit("flash-attn with the same slopes, all on identical inputs")
    emit(f"modes: dense | varlen (packed, seqlen fractions {VARLEN_FRACTIONS} of seq)")
    emit(f"       paged (page_size={PAGE_SIZE}, shuffled block table) | splitk (num_kv_splits={SPLITK_SPLITS})")
    emit("alibi: (num_heads,) fp32 geometric ladder, positive, broadcast over batch;")
    emit("       score(i,j) += -slope * |i + Skv - Sq - j|, applied after the 1/sqrt(D) scale")
    emit("")
    emit("AITER'S CAUSAL ALiBi IS NOT TRUSTWORTHY ON THIS BUILD. varlen+causal+alibi faults the GPU")
    emit("and aborts the process, so those rows never call aiter (nan). dense/splitk causal DO run")
    emit("but return a wrong answer (cos 0.964 vs fp32; see verify_flash_attn_alibi.py) -- they are")
    emit("timed anyway, since it is the real masked alibi kernel, but treat fly/ait there as")
    emit("indicative only. Non-causal rows are a clean like-for-like comparison.")
    emit("")
    emit("ovh = alibi_ms / base_ms, the real cost of the feature: same kernel, same work, one extra")
    emit("term per score element. Unlike the sink (an epilogue term whose cost vanishes as S grows),")
    emit("ALiBi is one sub + one fma INSIDE the kv loop with no memory traffic, so ovh should be")
    emit("roughly flat in S. A row where ovh climbs with S is a finding.")
    emit("All columns are timed with their samples INTERLEAVED, not one call at a time: GPU clocks")
    emit("drift down over a run, so timing them sequentially makes whichever went first look up to")
    emit("~6% faster and biases every ratio for no physical reason.")
    emit("GFLOP is this row's actual work (unmasked q,k pairs only). Varlen rows pack shorter")
    emit("sequences, so they do LESS work than the dense row on the same (B,seq) line -- compare")
    emit("TFLOPS across rows, never ms. ALiBi adds no FLOPs, so it is not in GFLOP; read ovh.")
    emit("vs dns = this row's alibi TF / the dense row's alibi TF at the same (B,seq,causal).")
    emit("fly/ait = alibi TF / aiter TF, >1.00 meaning fly is faster.")
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
                        ms_l, ms_b, ms_a, seqlens = run_one(B, S, causal, mode)
                    except torch.cuda.OutOfMemoryError:
                        emit(f"{B:>3} {S:>7} {mode:>7} {int(causal):>7} | OOM")
                        torch.cuda.empty_cache()
                        continue
                    fl = flops(seqlens, causal)
                    tf_l = (fl / 1e12) / (ms_l / 1e3)
                    tf_b = (fl / 1e12) / (ms_b / 1e3)
                    have_a = ms_a == ms_a  # not NaN
                    tf_a = (fl / 1e12) / (ms_a / 1e3) if have_a else float("nan")
                    ovh = ms_l / ms_b
                    if mode == "dense":
                        dense_tf[(B, S, causal)] = tf_l
                    # Work-normalized, so varlen's shorter sequences do not skew it.
                    base = dense_tf.get((B, S, causal))
                    rel = tf_l / base if base else float("nan")
                    # tf_l / tf_a; same row's work cancels, so it is exactly ms_a / ms_l.
                    vs_aiter = tf_l / tf_a if have_a else float("nan")
                    emit(
                        f"{B:>3} {S:>7} {mode:>7} {int(causal):>7} | "
                        f"{fl / 1e9:>9.1f} | "
                        f"{ms_l:>10.4f} {ms_b:>10.4f} {ms_a:>10.4f} | "
                        f"{ovh:>6.3f}x {rel:>6.2f}x {vs_aiter:>7.2f}x | "
                        f"{tf_l:>9.1f} {tf_b:>9.1f} {tf_a:>9.1f}"
                    )
    _out_fh.close()
