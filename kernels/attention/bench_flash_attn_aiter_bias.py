"""Benchmark the gfx950 dualwave flash-attention kernel WITH ATTENTION BIAS
against aiter flash-attn with the same bias, across every Q mode and KV source
the bias path supports: dense, packed-varlen, paged KV, and split-K.

Bias follows aiter's contract, so where both backends run the work is identical:

  dense/paged/split-K   bias is (seqlen_q, seqlen_kv) bf16, broadcast over batch
                        and head.
  varlen                bias is packed (total_q, max_seqlen_kv): row = the
                        *global* packed q token index, column = the per-batch-
                        local key index, so batch b reads rows
                        [cu_seqlens_q[b], cu_seqlens_q[b+1]) and columns
                        [0, seqlen_kv[b]). Broadcast over head only. This is
                        what aiter/ck-tile group mode expects (see aiter
                        csrc/py_itfs_ck/mha_varlen_fwd_kernels.cu:
                        "bias:(total_q, max_seqlen_k)").

Columns:
  GFLOP     the work this row actually does, counting only unmasked
            (query, key) pairs. READ THIS BEFORE COMPARING ROWS -- varlen rows
            pack sequences of differing length (VARLEN_FRACTIONS), so a varlen
            row does strictly LESS work than the dense row on the same (B, seq)
            line and their ms are not comparable. TFLOPS is per-row work over
            per-row time, so TFLOPS is comparable; ms is not.
  fly TF    flash_attn_gfx950 built with has_bias=True.
  aiter TF  aiter flash-attn with the same bias, where aiter supports the mode.
  vs dns    this row's fly TF over the *dense* fly TF at the same (B, seq,
            causal). Work-normalized, so it is meaningful for every mode
            including varlen, and it is the only baseline paged rows get.
            1.00 by definition on dense rows; below 1.00 means this mode costs
            throughput relative to plain dense attention.

aiter coverage: dense/varlen/split-K compare against flash_attn_func /
flash_attn_varlen_func. Paged rows report nan for aiter, by necessity:
  - mha_varlen_fwd rejects bias outright for page attention
    ("Page attention does not supports bias for now").
  - mha_batch_prefill_func, aiter's paged prefill entry point, does not expose a
    bias argument at all (only alibi_slopes), even though the private
    _mha_batch_prefill beneath it accepts one. Reaching past the public API for
    a benchmark baseline would both be fragile across aiter versions and compare
    a packed-Q batch-prefill against our dense-Q paged row, so we don't.
Use `vs dns` to judge paged instead -- it isolates what paging costs inside the
same kernel, which is the more actionable number anyway.

Note aiter's varlen+bias path falls back to ck-tile (bias disqualifies its
fmha_v3 ASM path), so varlen aiter numbers are much slower than its dense ones
for the same work; that is an aiter routing artifact, not a work-accounting
difference.

FLOPs = 4 * unmasked_pairs * D * H (2 for QK^T + 2 for P@V), matching the repo's
_flops helper in tests/kernels/test_flash_attn_fwd.py, so causal rows are not
inflated and varlen sums per sequence. The bias itself adds one add per score,
not a matmul, so it is not counted.

Rows are SKIPped rather than silently dropped when the bias cannot be addressed
(FlyDSL buffer descriptors clamp num_records to 4 GiB and the kernel computes
bias element offsets in i32) or when the split-K workspace would not fit.

Because the timed path hand-builds the kernel's positional ABI (see make_args),
a selftest always runs first and asserts, per mode, that the hand-built call is
bitwise identical to the high-level launcher, aborting if not. A wrong ABI slot
would otherwise show up as a suspiciously fast row rather than an error.

Usage:
    PYTHONPATH=/var/home/bias/FlyDSL python3 bench_flash_attn_aiter_bias.py
    PYTHONPATH=/var/home/bias/FlyDSL python3 bench_flash_attn_aiter_bias.py 1 2   # batch sizes
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

OUT_PATH = "bench_flash_attn_aiter_bias.txt"
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

# Bias addressing limits, see module docstring.
BUFFER_MAX_BYTES = 0xFFFFFFFF  # flydsl/expr/buffer_ops.py clamps num_records here
I32_MAX_ELEMS = 2**31 - 1  # kernel computes bias element offsets in i32
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


def bias_fits(rows, cols):
    """Can a (rows, cols) bf16 bias be addressed by the kernel?"""
    elems = rows * cols
    if elems > I32_MAX_ELEMS:
        return False, f"bias {rows}x{cols} = {elems:.3g} elems exceeds i32 offset range"
    if elems * 2 > BUFFER_MAX_BYTES:
        return False, f"dense bias {elems * 2 / 2**30:.1f} GiB exceeds 4 GiB buffer limit"
    return True, ""


def splitk_fits(B, S):
    """Is the fp32 split-K workspace small enough to allocate?"""
    nbytes = dualwave_splitk_workspace_elems(B, H, S, SPLITK_SPLITS, head_dim=D) * 4
    if nbytes > WORKSPACE_MAX_BYTES:
        return False, f"split-K workspace {nbytes / 2**30:.1f} GiB exceeds {WORKSPACE_MAX_BYTES / 2**30:.0f} GiB cap"
    return True, ""


def row_fits(B, S, mode):
    """Per-row feasibility: bias addressing, plus split-K workspace."""
    bias_rows = sum(varlen_seqlens(B, S)) if mode == "varlen" else S
    ok, why = bias_fits(bias_rows, S)
    if not ok:
        return ok, why
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
        bias = torch.randn(total, S, dtype=dtype, device=dev)  # (total_q, max_seqlen_kv)
        out = torch.empty(total, H, D, dtype=dtype, device=dev)
        return dict(
            seqlens=seqlens,
            q=q,
            k=k,
            v=v,
            bias=bias,
            out=out,
            cu=cu_t,
            block_table=None,
            block_table_stride=0,
            workspace=None,
            build=dict(varlen=True),
            launch=dict(cu_seqlens_q=cu_t, cu_seqlens_kv=cu_t),
        )

    # dense / paged / splitk all use dense Q and a dense (S, S) bias.
    seqlens = [S] * B
    q = torch.randn(B, S, H, D, dtype=dtype, device=dev)
    bias = torch.randn(S, S, dtype=dtype, device=dev)
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
            bias=bias,
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
            bias=bias,
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
        bias=bias,
        out=out,
        cu=None,
        block_table=None,
        block_table_stride=0,
        workspace=None,
        build={},
        launch={},
    )


def make_exe(B, S, causal, mode, inp):
    return build_flash_attn_dualwave_swp_module(
        num_heads=H,
        head_dim=D,
        causal=causal,
        dtype_str="bf16",
        num_kv_heads=H_KV,
        has_bias=True,
        **inp["build"],
    )


def make_args(B, S, inp, stream):
    """The positional ABI that _compile builds internally.

    `out` is the placeholder for tensors this build never reads, matching the
    kernel's own convention; the cu_seqlens / BlockTable / workspace slots carry
    real tensors only in the modes that read them. Kept in one place so the
    selftest below covers every mode's slot assignment.
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
        inp["bias"].contiguous().view(-1),
        out,  # AlibiSlopes: this build is has_alibi=False, so it is never read
        B,
        S,  # seq_len (varlen: max_seqlen_q, which sizes grid_y)
        S,  # seq_len_kv
        H * D,  # stride_q_n
        H_KV * D,  # stride_kv_n
        D,  # head_dim_runtime
        inp["block_table_stride"],
        inp["bias"].stride(0),  # bias_stride0
        0,  # alibi_stride_b
        fx.Stream(stream),
    )


def time_call(call, warmup, iters):
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start.record()
        call()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return sum(times) / len(times)


def selftest(S=512, B=2, causal=True):
    """Assert the hand-built ABI matches the high-level launcher, per mode.

    The timed path calls the precompiled executable positionally; a wrong slot
    would silently benchmark the wrong work instead of raising.
    """
    ok = True
    for mode in MODES:
        fits, why = row_fits(B, S, mode)
        if not fits:
            print(f"[selftest] {mode:>7}: SKIP ({why})", file=sys.stderr, flush=True)
            continue
        inp = build_inputs(B, S, mode)
        exe = make_exe(B, S, causal, mode, inp)
        stream = torch.cuda.current_stream()
        launch = dict(inp["launch"], bias=inp["bias"], stream=stream)
        exe(inp["q"], inp["k"], inp["v"], inp["out"], B, S, **launch)
        torch.cuda.synchronize()
        ref = inp["out"].clone()
        inp["out"].zero_()
        torch.cuda.synchronize()
        compiled = exe.compile(inp["q"], inp["k"], inp["v"], inp["out"], B, S, **launch)
        compiled(*make_args(B, S, inp, stream))
        torch.cuda.synchronize()
        same = torch.equal(ref, inp["out"])
        ok &= same
        print(
            f"[selftest] {mode:>7}: {'ABI ok (bitwise identical)' if same else 'ABI MISMATCH'}",
            file=sys.stderr,
            flush=True,
        )
        del inp, ref
        torch.cuda.empty_cache()
    return ok


def aiter_callable(B, S, causal, mode, inp):
    """aiter equivalent for this row, or None when aiter cannot run it."""
    if aiter_flash_attn_func is None:
        return None
    q, k, v, bias = inp["q"], inp["k"], inp["v"], inp["bias"]
    if mode == "varlen":
        cu = inp["cu"]
        return lambda: aiter_flash_attn_varlen_func(q, k, v, cu, cu, S, S, causal=causal, bias=bias)
    if mode == "paged":
        # aiter: "Page attention does not supports bias for now".
        return None
    return lambda: aiter_flash_attn_func(q, k, v, causal=causal, bias=bias)


def run_one(B, S, causal, mode):
    """Time one (batch, seq_len, causal, mode) point. Returns (fly_ms, aiter_ms, seqlens)."""
    dbg(f"run_one begin B={B} S={S} causal={causal} mode={mode}")
    inp = build_inputs(B, S, mode)

    dbg(f"build FlyDSL bias kernel (mode={mode})")
    exe = make_exe(B, S, causal, mode, inp)
    stream = torch.cuda.current_stream()

    # Time the precompiled executable, not the _launch wrapper. The wrapper
    # re-derives its JIT cache key by reflection on every call, which costs a
    # flat ~0.12 ms regardless of S -- that swamps the GPU time below S~2048 and
    # would report host dispatch cost instead of kernel performance.
    launch = dict(inp["launch"], bias=inp["bias"], stream=stream)
    compiled = exe.compile(inp["q"], inp["k"], inp["v"], inp["out"], B, S, **launch)
    args = make_args(B, S, inp, stream)

    def fly_call():
        compiled(*args)
        return inp["out"]

    aiter_call = aiter_callable(B, S, causal, mode, inp)

    warmup, iters = iters_for(S)
    dbg("time fly_call")
    ms_fly = time_call(fly_call, warmup, iters)
    ms_aiter = float("nan")
    if aiter_call is not None:
        dbg("time aiter_call")
        try:
            ms_aiter = time_call(aiter_call, warmup, iters)
        except Exception as e:
            print(
                f"[warn] aiter call failed for B={B} S={S} causal={causal} mode={mode}: {e}",
                file=sys.stderr,
                flush=True,
            )
    dbg(f"run_one done B={B} S={S} causal={causal} mode={mode}")

    # No explicit del: fly_call/aiter_call close over inp/compiled/args, so they
    # are released when run_one returns and the closures die with it.
    result = (ms_fly, ms_aiter, inp["seqlens"])
    torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    if not selftest():
        print("[fatal] ABI selftest failed; timings would be meaningless", file=sys.stderr)
        sys.exit(1)

    hdr = (
        f"{'B':>3} {'seq':>7} {'mode':>7} {'causal':>7} | "
        f"{'GFLOP':>9} | "
        f"{'fly ms':>10} {'aiter ms':>10} | "
        f"{'fly/ait':>8} {'vs dns':>7} | "
        f"{'fly TF':>9} {'aiter TF':>9}"
    )
    # fly TF of the dense row, keyed by (B, seq, causal); MODES puts dense first
    # so it is always populated before the modes that reference it.
    dense_tf = {}
    emit(f"D={D} H={H} H_KV={H_KV}  dtype={dtype}")
    emit("flash_attn_gfx950(has_bias=True) vs aiter flash-attn, both with the same bf16 attention bias")
    emit(f"modes: dense | varlen (packed, seqlen fractions {VARLEN_FRACTIONS} of seq)")
    emit(f"       paged (page_size={PAGE_SIZE}, shuffled block table) | splitk (num_kv_splits={SPLITK_SPLITS})")
    emit("bias: dense/paged/splitk (Sq,Skv) broadcast over batch+head; varlen (total_q,max_Skv) packed rows")
    emit("")
    emit("GFLOP is this row's actual work (unmasked q,k pairs only). Varlen rows pack shorter")
    emit("sequences, so they do LESS work than the dense row on the same (B,seq) line -- compare")
    emit("TFLOPS across rows, never ms.")
    emit("vs dns = this row's fly TF / the dense row's fly TF at the same (B,seq,causal).")
    emit("paged has no aiter number (aiter exposes no public paged+bias entry point); judge it by vs dns.")
    emit("aiter varlen+bias falls back to ck-tile (bias disqualifies its fmha_v3 ASM path), so its")
    emit("varlen ms is far above its dense ms for identical work -- an aiter routing artifact.")
    emit("fly/ait = aiter_ms / fly_ms  (>1 means the FlyDSL kernel is faster)")
    emit("fly ms times the precompiled executable; the _launch wrapper adds a flat ~0.12 ms/call of")
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
                        ms_f, ms_a, seqlens = run_one(B, S, causal, mode)
                    except torch.cuda.OutOfMemoryError:
                        emit(f"{B:>3} {S:>7} {mode:>7} {int(causal):>7} | OOM")
                        torch.cuda.empty_cache()
                        continue
                    fl = flops(seqlens, causal)
                    tf_f = (fl / 1e12) / (ms_f / 1e3)
                    have_a = ms_a == ms_a  # not NaN
                    tf_a = (fl / 1e12) / (ms_a / 1e3) if have_a else float("nan")
                    sp = ms_a / ms_f if have_a else float("nan")
                    if mode == "dense":
                        dense_tf[(B, S, causal)] = tf_f
                    # Work-normalized, so varlen's shorter sequences do not skew it.
                    base = dense_tf.get((B, S, causal))
                    rel = tf_f / base if base else float("nan")
                    emit(
                        f"{B:>3} {S:>7} {mode:>7} {int(causal):>7} | "
                        f"{fl / 1e9:>9.1f} | "
                        f"{ms_f:>10.4f} {ms_a:>10.4f} | "
                        f"{sp:>7.2f}x {rel:>6.2f}x | "
                        f"{tf_f:>9.1f} {tf_a:>9.1f}"
                    )
    _out_fh.close()
