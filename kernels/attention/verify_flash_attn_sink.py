"""Correctness gate for the gfx950 dualwave flash-attention attention-sink path.

Checks build_flash_attn_dualwave_swp_module(has_sink=True) against an fp32 torch
reference. The sink is one extra softmax logit per q head with no matching V row:

    O = sum_j exp(s_j - m) v_j / (exp(sink - m) + sum_j exp(s_j - m))

It follows aiter's contract (aiter/ops/mha.py fmha_fwd_with_sink_asm): fp32,
1-D (num_heads,), consumed verbatim with no host-side scaling, so it lives in the
same post-softmax_scale logit space as the score.

There is no aiter cross-check: aiter exposes the sink only through
fmha_fwd_with_sink_asm / fmha_fwd_with_sink_varlen_asm, which are gfx1250 ASM
kernels and do not run on gfx950. Rows gate on the fp32 reference alone.

CALIBRATION -- why the sink is not just randn. The sink's share of the softmax
mass is sigmoid(sink - logsumexp(scores)), and logsumexp grows like ln(seqlen):
for D=128 randn inputs at S=1024 it sits near 7.4. So a sink drawn around 0 owns
~0.06% of the mass, which is *below the bf16 noise floor* -- such a row passes
just as happily with the sink dropped entirely. A very large sink is equally
useless: it takes 100% of the mass and drives O to 0, where cosine similarity
stops being sensitive to the denominator at all.

Every row therefore calibrates the sink to a target share of the softmax mass,
sink = mean_rows(logsumexp(scores)) + logit(share), and reports the share it
actually achieved. Shares in the 0.25-0.95 band put the sink well above bf16
noise, so a dropped, double-counted, or mis-scaled sink cannot pass. run_ablation
closes the loop by building the same shape with has_sink=False and requiring the
outputs to disagree.

The sink is applied in the epilogue and touches no score element, so it is
orthogonal to the Q mode and the KV source: dense, packed-varlen, paged (linear +
vectorized KV cache), paged+varlen and split-K are all covered. Split-K is the
one structurally different path -- per-split partials are written sink-free and
the combine pass folds the sink in once -- so a regression that counted the sink
NUM_KV_SPLITS times shows up only there, most sharply in LSE.

Usage:  PYTHONPATH=/var/home/FlyDSL python3 verify_flash_attn_sink.py
"""

import math
import sys

import torch

from kernels.attention.flash_attn_gfx950 import build_flash_attn_dualwave_swp_module
from kernels.attention.flash_attn_utils import dualwave_splitk_workspace_elems

torch.manual_seed(0)

D = 128
dtype = torch.bfloat16


def per_head_mean(lse0):
    """Mean of lse0 [..., H, Sq] over everything but H, ignoring -inf rows.

    Fully-masked rows (cross-seqlen causal) carry logsumexp = -inf and would
    otherwise poison the mean into -inf and the calibration with it.
    """
    h = lse0.shape[-2]
    x = lse0.movedim(-2, 0).reshape(h, -1)
    finite = torch.isfinite(x)
    return torch.where(finite, x, torch.zeros_like(x)).sum(1) / finite.sum(1).clamp(min=1)


def calibrate_sink(lse0, share):
    """Per-head sink placing `share` of the softmax mass on the sink.

    share = sigmoid(sink - logsumexp(scores)), so invert it per head.
    """
    return (per_head_mean(lse0) + math.log(share / (1.0 - share))).float().contiguous()


def achieved_share(sink_h, lse0):
    """Mean fraction of softmax mass actually taken by the sink, over finite rows."""
    s = sink_h.view(*([1] * (lse0.dim() - 2)), -1, 1)
    frac = torch.sigmoid(s - lse0)
    finite = torch.isfinite(lse0)
    return (frac[finite].mean().item()) if finite.any() else float("nan")


def sink_softmax(sc, sink_h):
    """softmax over [scores, sink] with the sink carrying no value row.

    sc is [..., H, Sq, Skv] fp32, sink_h is [H] fp32. Returns (probs, lse); probs
    sum to 1 - share (the sink holds the remainder) and lse is the natural-log
    denominator including the sink.

    A fully-masked row (all -inf) needs no special-casing: the max collapses to
    the sink, every score term is exp(-inf) = 0, and the row becomes all-sink,
    giving probs 0 and lse = sink -- exactly what the kernel produces. Without a
    sink this is the case that yields NaN and has to be patched up.
    """
    s = sink_h.view(*([1] * (sc.dim() - 3)), sink_h.shape[0], 1, 1)
    m = torch.maximum(sc.amax(dim=-1, keepdim=True), s)
    e = torch.exp(sc - m)
    den = e.sum(dim=-1, keepdim=True) + torch.exp(s - m)
    return e / den, (m + torch.log(den)).squeeze(-1)


def causal_mask(sc, sq, skv, device):
    """Bottom-right aligned causal mask: query i attends keys <= i + (skv - sq)."""
    i = torch.arange(sq, device=device)[:, None]
    j = torch.arange(skv, device=device)[None, :]
    return sc.masked_fill(j > i + (skv - sq), float("-inf"))


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def _report(label, c_ref, mx, share, extra=""):
    # A share this small means the row cannot distinguish a correct sink from no
    # sink at all, so treat it as a failure of the test rather than a pass.
    ok = c_ref > 0.999 and share > 0.01
    print(
        f"{label} | cos_ref={c_ref:.6f} maxabs={mx:.4f} sink_share={share * 100:5.1f}%{extra}"
        f"{'' if ok else '   <-- ACC FAIL'}",
        flush=True,
    )
    torch.cuda.empty_cache()
    return ok


# ---------------------------------------------------------------- dense


def dense_scores(q, k, v, causal):
    """fp32 [B, H, Sq, Skv] scores plus the GQA-expanded V the reference multiplies."""
    b_, s_, h_, dh = q.shape
    skv = k.shape[1]
    group = h_ // k.shape[2]
    qf = q.float().permute(0, 2, 1, 3)
    kf = k.float().permute(0, 2, 1, 3).repeat_interleave(group, dim=1)
    vf = v.float().permute(0, 2, 1, 3).repeat_interleave(group, dim=1)
    sc = torch.matmul(qf, kf.transpose(-1, -2)) * (1.0 / dh**0.5)
    return (causal_mask(sc, s_, skv, q.device) if causal else sc), vf


def ref_fp32(q, k, v, causal, share):
    """softmax([q @ k^T * scale, sink]) @ v in fp32; derives the calibrated sink."""
    sc, vf = dense_scores(q, k, v, causal)
    lse0 = torch.logsumexp(sc, dim=-1)
    sink = calibrate_sink(lse0, share)
    p, lse = sink_softmax(sc, sink)
    out = torch.matmul(p, vf).permute(0, 2, 1, 3).contiguous()
    return out, lse, sink, achieved_share(sink, lse0)


def _dense_inputs(b_, s_, h_, h_kv):
    return (
        torch.randn(b_, s_, h_, D, dtype=dtype, device="cuda"),
        torch.randn(b_, s_, h_kv, D, dtype=dtype, device="cuda"),
        torch.randn(b_, s_, h_kv, D, dtype=dtype, device="cuda"),
    )


def run(b_, s_, h_, h_kv, causal, share):
    q, k, v = _dense_inputs(b_, s_, h_, h_kv)
    ref, ref_lse, sink, got = ref_fp32(q, k, v, causal, share)

    out = torch.empty(b_, s_, h_, D, dtype=dtype, device="cuda")
    lse = torch.empty(b_, h_, s_, dtype=torch.float32, device="cuda")
    exe = build_flash_attn_dualwave_swp_module(
        num_heads=h_,
        head_dim=D,
        causal=causal,
        dtype_str="bf16",
        num_kv_heads=h_kv,
        return_lse=True,
        has_sink=True,
    )
    exe(q, k, v, out, b_, s_, sink=sink, lse=lse, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    # LSE is where a dropped or double-counted sink shows up most directly: it is
    # the log denominator, so the sink term is not normalized away as it is in O.
    lse_err = (ref_lse - lse).abs().max().item()
    return _report(
        f"dense  B={b_} S={s_:>5} H={h_:>3} Hkv={h_kv:>3} causal={int(causal)}",
        cos(ref, out),
        (ref.float() - out.float()).abs().max().item(),
        got,
        f" | lse_err={lse_err:.5f}",
    )


def run_ablation(b_, s_, h_, h_kv, causal, share):
    """has_sink=True vs has_sink=False on identical inputs; they must disagree.

    Guards against the whole feature being a no-op -- a kernel that silently
    ignored the sink would still match a sink reference on every other row here,
    because the sink only rescales O and cosine similarity is scale-invariant.
    So this row compares *magnitudes*, against a ratio taken from the fp32
    reference rather than from the mean share (per-row shares differ under a
    causal mask, and the norm ratio weights rows by ||O||, so 1 - mean(share) is
    not the right expectation).

    Dense self-attention only: no row is fully masked, so the no-sink reference
    is well defined.
    """
    q, k, v = _dense_inputs(b_, s_, h_, h_kv)
    sc, vf = dense_scores(q, k, v, causal)
    lse0 = torch.logsumexp(sc, dim=-1)
    sink = calibrate_sink(lse0, share)
    got = achieved_share(sink, lse0)

    p_sink, _ = sink_softmax(sc, sink)
    p_plain, _ = sink_softmax(sc, torch.full_like(sink, -1e30))
    ref_sink = torch.matmul(p_sink, vf).permute(0, 2, 1, 3).contiguous()
    ref_plain = torch.matmul(p_plain, vf).permute(0, 2, 1, 3).contiguous()
    expect = (ref_sink.norm() / ref_plain.norm()).item()

    out_sink = torch.empty(b_, s_, h_, D, dtype=dtype, device="cuda")
    build_flash_attn_dualwave_swp_module(
        num_heads=h_, head_dim=D, causal=causal, dtype_str="bf16", num_kv_heads=h_kv, has_sink=True
    )(q, k, v, out_sink, b_, s_, sink=sink, stream=torch.cuda.current_stream())

    out_plain = torch.empty(b_, s_, h_, D, dtype=dtype, device="cuda")
    build_flash_attn_dualwave_swp_module(
        num_heads=h_, head_dim=D, causal=causal, dtype_str="bf16", num_kv_heads=h_kv
    )(q, k, v, out_plain, b_, s_, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    ratio = (out_sink.float().norm() / out_plain.float().norm()).item()
    c_sink = cos(ref_sink, out_sink)
    c_plain = cos(ref_plain, out_plain)
    # expect < 0.95 keeps the row honest: if the sink barely changed the
    # magnitude there would be nothing here for the ratio check to catch.
    ok = c_sink > 0.999 and c_plain > 0.999 and expect < 0.95 and abs(ratio - expect) < 0.01
    print(
        f"ablation B={b_} S={s_:>5} H={h_:>3} causal={int(causal)} | cos_sink={c_sink:.6f} "
        f"cos_nosink={c_plain:.6f} sink_share={got * 100:5.1f}% "
        f"|O_sink|/|O_nosink|={ratio:.4f} (expect {expect:.4f})"
        f"{'' if ok else '   <-- ACC FAIL'}",
        flush=True,
    )
    torch.cuda.empty_cache()
    return ok


# ---------------------------------------------------------------- varlen


def ref_fp32_varlen(q, k, v, cu_q, cu_kv, causal, share):
    """Per-sequence fp32 varlen reference; two passes so the sink can be calibrated."""
    total_q, h_, dh = q.shape
    group = h_ // k.shape[1]
    n = len(cu_q) - 1

    def scores(b):
        sq, skv = cu_q[b + 1] - cu_q[b], cu_kv[b + 1] - cu_kv[b]
        qb = q[cu_q[b] : cu_q[b + 1]].float().permute(1, 0, 2)
        kb = k[cu_kv[b] : cu_kv[b + 1]].float().permute(1, 0, 2).repeat_interleave(group, dim=0)
        vb = v[cu_kv[b] : cu_kv[b + 1]].float().permute(1, 0, 2).repeat_interleave(group, dim=0)
        sc = torch.matmul(qb, kb.transpose(-1, -2)) * (1.0 / dh**0.5)
        return (causal_mask(sc, sq, skv, q.device) if causal else sc), vb

    lse0 = [torch.logsumexp(scores(b)[0], dim=-1) for b in range(n)]
    sink = calibrate_sink(torch.cat(lse0, dim=-1), share)

    out = torch.zeros(total_q, h_, dh, dtype=torch.float32, device=q.device)
    for b in range(n):
        sc, vb = scores(b)
        p, _ = sink_softmax(sc, sink)
        out[cu_q[b] : cu_q[b + 1]] = torch.matmul(p, vb).permute(1, 0, 2)
    return out, sink, achieved_share(sink, torch.cat(lse0, dim=-1))


def _cumsum(lengths):
    cu = [0]
    for s in lengths:
        cu.append(cu[-1] + s)
    return cu


def run_varlen(seqlens_q, seqlens_kv, h_, h_kv, causal, share):
    vq = list(seqlens_q)
    vkv = list(seqlens_kv) if seqlens_kv is not None else list(vq)
    b_ = len(vq)
    cross = any(a != b for a, b in zip(vq, vkv))
    cu_q, cu_kv = _cumsum(vq), _cumsum(vkv)
    total_q, total_kv = cu_q[-1], cu_kv[-1]
    max_sq, max_skv = max(vq), max(vkv)

    q = torch.randn(total_q, h_, D, dtype=dtype, device="cuda")
    k = torch.randn(total_kv, h_kv, D, dtype=dtype, device="cuda")
    v = torch.randn(total_kv, h_kv, D, dtype=dtype, device="cuda")
    ref, sink, got = ref_fp32_varlen(q, k, v, cu_q, cu_kv, causal, share)

    out = torch.empty(total_q, h_, D, dtype=dtype, device="cuda")
    exe = build_flash_attn_dualwave_swp_module(
        num_heads=h_,
        head_dim=D,
        causal=causal,
        dtype_str="bf16",
        num_kv_heads=h_kv,
        varlen=True,
        cross_seqlen=cross,
        has_sink=True,
    )
    kwargs = dict(
        cu_seqlens_q=torch.tensor(cu_q, dtype=torch.int32, device="cuda"),
        cu_seqlens_kv=torch.tensor(cu_kv, dtype=torch.int32, device="cuda"),
        sink=sink,
        stream=torch.cuda.current_stream(),
    )
    if cross:
        kwargs["seq_len_kv"] = max_skv
    exe(q, k, v, out, b_, max_sq, **kwargs)
    torch.cuda.synchronize()

    return _report(
        f"varlen q={str(vq):>24} kv={str(vkv) if seqlens_kv else '(self)':>24} "
        f"H={h_:>3} Hkv={h_kv:>3} causal={int(causal)}",
        cos(ref, out),
        (ref - out.float()).abs().max().item(),
        got,
    )


# ---------------------------------------------------------------- paged

PAGE_SIZE = 64  # must equal traits.BLOCK_N; the block table is indexed by tile


def _vectorize_paged_kv(k4, v4, h_kv, page_size, kvs):
    """aiter-style 5D paged K/V from the 4D [pages, page_size, Hkv, D] form."""
    k5 = k4.contiguous().view(-1, page_size, h_kv, D // kvs, kvs).permute(0, 2, 3, 1, 4).contiguous()
    v5 = v4.contiguous().view(-1, page_size // kvs, kvs, h_kv, D).permute(0, 3, 1, 4, 2).contiguous()
    return k5, v5


def _paged_cache(n_pages, h_kv, layout):
    """Physical K/V cache plus the 4D view the reference gathers through."""
    k4 = torch.randn(n_pages, PAGE_SIZE, h_kv, D, dtype=dtype, device="cuda")
    v4 = torch.randn(n_pages, PAGE_SIZE, h_kv, D, dtype=dtype, device="cuda")
    if layout == "vectorized":
        kvs = 16 // torch.empty((), dtype=dtype).element_size()
        return (*_vectorize_paged_kv(k4, v4, h_kv, PAGE_SIZE, kvs), k4, v4)
    return k4, v4, k4, v4


def _paged_perm(b_, n_pages_per_seq):
    """Deliberately non-identity logical->physical page map, plus spare pages, so a
    kernel that ignored the block table would fail loudly."""
    total_pages = b_ * n_pages_per_seq + 8
    perm = torch.randperm(total_pages, device="cuda")[: b_ * n_pages_per_seq].view(b_, n_pages_per_seq)
    return total_pages, perm


def _seq_scores(qb, kb, causal):
    """Scores for [..., Sq, D] x [..., Skv, D]; works for both the 3D per-sequence
    and 4D batched paged forms, so index the sequence axis from the end."""
    sc = torch.matmul(qb, kb.transpose(-1, -2)) * (1.0 / D**0.5)
    return causal_mask(sc, qb.shape[-2], kb.shape[-2], qb.device) if causal else sc


def run_paged(b_, sq, skv, h_, h_kv, causal, layout, share):
    n_pages_per_seq = (skv + PAGE_SIZE - 1) // PAGE_SIZE
    total_pages, perm = _paged_perm(b_, n_pages_per_seq)
    k_cache, v_cache, k4, v4 = _paged_cache(total_pages, h_kv, layout)
    q = torch.randn(b_, sq, h_, D, dtype=dtype, device="cuda")
    cross = skv != sq
    group = h_ // h_kv

    kd = k4[perm].reshape(b_, n_pages_per_seq * PAGE_SIZE, h_kv, D)[:, :skv]
    vd = v4[perm].reshape(b_, n_pages_per_seq * PAGE_SIZE, h_kv, D)[:, :skv]
    qf = q.float().permute(0, 2, 1, 3)
    kf = kd.float().permute(0, 2, 1, 3).repeat_interleave(group, dim=1)
    vf = vd.float().permute(0, 2, 1, 3).repeat_interleave(group, dim=1)
    sc = _seq_scores(qf, kf, causal)
    lse0 = torch.logsumexp(sc, dim=-1)
    sink = calibrate_sink(lse0, share)
    p, _ = sink_softmax(sc, sink)
    ref = torch.matmul(p, vf).permute(0, 2, 1, 3)

    out = torch.empty(b_, sq, h_, D, dtype=dtype, device="cuda")
    exe = build_flash_attn_dualwave_swp_module(
        num_heads=h_,
        head_dim=D,
        causal=causal,
        dtype_str="bf16",
        num_kv_heads=h_kv,
        paged=True,
        cross_seqlen=cross,
        kv_cache_layout=layout,
        has_sink=True,
    )
    kwargs = dict(
        block_table=perm.to(torch.int32).contiguous().reshape(-1),
        block_table_stride=n_pages_per_seq,
        sink=sink,
        stream=torch.cuda.current_stream(),
    )
    if cross:
        kwargs["seq_len_kv"] = skv
    exe(q, k_cache, v_cache, out, b_, sq, **kwargs)
    torch.cuda.synchronize()

    return _report(
        f"paged  {layout:>10} B={b_} Sq={sq:>5} Skv={skv:>5} H={h_:>3} Hkv={h_kv:>3} causal={int(causal)}",
        cos(ref, out),
        (ref - out.float()).abs().max().item(),
        achieved_share(sink, lse0),
    )


def run_paged_varlen(vq, vkv, h_, h_kv, causal, layout, share):
    b_ = len(vq)
    cu_q, cu_kv = _cumsum(vq), _cumsum(vkv)
    total_q, sq, skv = cu_q[-1], max(vq), max(vkv)
    n_pages_per_seq = (skv + PAGE_SIZE - 1) // PAGE_SIZE
    total_pages, perm = _paged_perm(b_, n_pages_per_seq)
    k_cache, v_cache, k4, v4 = _paged_cache(total_pages, h_kv, layout)
    q = torch.randn(total_q, h_, D, dtype=dtype, device="cuda")
    group = h_ // h_kv

    def scores(b):
        s_kv = cu_kv[b + 1] - cu_kv[b]
        kd = k4[perm[b]].reshape(-1, h_kv, D)[:s_kv]
        vd = v4[perm[b]].reshape(-1, h_kv, D)[:s_kv]
        qb = q[cu_q[b] : cu_q[b + 1]].float().permute(1, 0, 2)
        kb = kd.float().permute(1, 0, 2).repeat_interleave(group, dim=0)
        vb = vd.float().permute(1, 0, 2).repeat_interleave(group, dim=0)
        return _seq_scores(qb, kb, causal), vb

    lse0 = torch.cat([torch.logsumexp(scores(b)[0], dim=-1) for b in range(b_)], dim=-1)
    sink = calibrate_sink(lse0, share)
    ref = torch.zeros(total_q, h_, D, dtype=torch.float32, device="cuda")
    for b in range(b_):
        sc, vb = scores(b)
        p, _ = sink_softmax(sc, sink)
        ref[cu_q[b] : cu_q[b + 1]] = torch.matmul(p, vb).permute(1, 0, 2)

    out = torch.empty(total_q, h_, D, dtype=dtype, device="cuda")
    exe = build_flash_attn_dualwave_swp_module(
        num_heads=h_,
        head_dim=D,
        causal=causal,
        dtype_str="bf16",
        num_kv_heads=h_kv,
        paged=True,
        varlen=True,
        cross_seqlen=True,
        kv_cache_layout=layout,
        has_sink=True,
    )
    exe(
        q,
        k_cache,
        v_cache,
        out,
        b_,
        sq,
        seq_len_kv=skv,
        cu_seqlens_q=torch.tensor(cu_q, dtype=torch.int32, device="cuda"),
        cu_seqlens_kv=torch.tensor(cu_kv, dtype=torch.int32, device="cuda"),
        block_table=perm.to(torch.int32).contiguous().reshape(-1),
        block_table_stride=n_pages_per_seq,
        sink=sink,
        stream=torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()

    return _report(
        f"paged+varlen {layout:>10} q={str(vq):>22} causal={int(causal)}",
        cos(ref, out),
        (ref - out.float()).abs().max().item(),
        achieved_share(sink, lse0),
    )


# ---------------------------------------------------------------- split-K


def run_splitk(b_, s_, h_, h_kv, causal, splits, share):
    q, k, v = _dense_inputs(b_, s_, h_, h_kv)
    ref, ref_lse, sink, got = ref_fp32(q, k, v, causal, share)

    out = torch.empty(b_, s_, h_, D, dtype=dtype, device="cuda")
    lse = torch.empty(b_, h_, s_, dtype=torch.float32, device="cuda")
    ws = torch.zeros(
        dualwave_splitk_workspace_elems(b_, h_, s_, splits, head_dim=D),
        dtype=torch.float32,
        device="cuda",
    )
    exe = build_flash_attn_dualwave_swp_module(
        num_heads=h_,
        head_dim=D,
        causal=causal,
        dtype_str="bf16",
        num_kv_heads=h_kv,
        num_kv_splits=splits,
        return_lse=True,
        has_sink=True,
    )
    exe(q, k, v, out, b_, s_, workspace=ws, sink=sink, lse=lse, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    # A sink counted once per split rather than once per row inflates the
    # denominator by up to NUM_KV_SPLITS; LSE catches that even when the
    # normalized O still looks close.
    lse_err = (ref_lse - lse).abs().max().item()
    return _report(
        f"splitk splits={splits} B={b_} S={s_:>5} H={h_:>3} Hkv={h_kv:>3} causal={int(causal)}",
        cos(ref, out),
        (ref - out.float()).abs().max().item(),
        got,
        f" | lse_err={lse_err:.5f}",
    )


# (batch, seq_len, num_heads, num_kv_heads, causal, sink_share)
CONFIGS = [
    (1, 512, 8, 8, False, 0.5),
    (1, 1024, 16, 16, False, 0.25),
    (2, 1024, 16, 8, False, 0.75),
    (1, 1024, 16, 16, True, 0.5),
    (2, 2048, 32, 8, True, 0.25),
    (2, 2048, 32, 8, True, 0.95),
]

# (batch, seq_len, num_heads, num_kv_heads, causal, sink_share)
ABLATION_CONFIGS = [
    (1, 1024, 16, 16, False, 0.5),
    (2, 2048, 32, 8, True, 0.75),
]

# (seqlens_q, seqlens_kv or None for self-attention, num_heads, num_kv_heads, causal, sink_share)
VARLEN_CONFIGS = [
    ([512, 512], None, 8, 8, False, 0.5),
    ([1000, 377, 64], None, 16, 8, False, 0.25),
    ([31, 33, 65, 127], None, 8, 8, False, 0.5),
    ([1024, 512], [512, 1024], 16, 8, False, 0.75),
    ([1000, 377, 64], None, 16, 8, True, 0.5),
    ([31, 33, 65, 127], None, 8, 8, True, 0.25),
    # seqlen_q > seqlen_kv leaves leading rows fully masked -- with a sink those
    # rows are all-sink rather than degenerate, so this is the NaN-guard case.
    ([1024, 512], [512, 1024], 16, 8, True, 0.5),
    ([1024, 512], [512, 1024], 16, 8, True, 0.95),
]

# (batch, Sq, Skv, num_heads, num_kv_heads, causal, kv_cache_layout, sink_share)
PAGED_CONFIGS = [
    (1, 512, 512, 8, 8, False, "linear", 0.5),
    (2, 1024, 1024, 16, 8, False, "linear", 0.25),
    (1, 512, 1024, 8, 8, False, "linear", 0.75),
    (2, 1024, 768, 16, 8, True, "linear", 0.5),
    (2, 1024, 1024, 16, 8, True, "linear", 0.95),
    (1, 512, 512, 8, 8, False, "vectorized", 0.5),
    (2, 1024, 768, 16, 8, True, "vectorized", 0.25),
]

# (seqlens_q, seqlens_kv, num_heads, num_kv_heads, causal, kv_cache_layout, sink_share)
PAGED_VARLEN_CONFIGS = [
    ([512, 512], [512, 512], 8, 8, False, "linear", 0.5),
    ([377, 64, 1000], [512, 192, 640], 16, 8, True, "linear", 0.25),
    ([1024, 512, 768], [1024, 512, 768], 16, 8, True, "vectorized", 0.75),
]

# (batch, seq_len, num_heads, num_kv_heads, causal, num_kv_splits, sink_share)
SPLITK_CONFIGS = [
    (1, 2048, 8, 8, False, 2, 0.5),
    (1, 2048, 8, 8, False, 4, 0.25),
    (1, 2048, 8, 8, True, 2, 0.5),
    (2, 4096, 16, 8, True, 4, 0.75),
    # Split-count sensitivity: the same shape and share at 2 and 8 splits must
    # agree, which a per-split sink would not.
    (1, 2048, 8, 8, False, 8, 0.5),
    (1, 2048, 8, 8, False, 8, 0.95),
]


if __name__ == "__main__":
    ok = True
    for cfg in CONFIGS:
        ok &= run(*cfg)
    for cfg in ABLATION_CONFIGS:
        ok &= run_ablation(*cfg)
    for cfg in VARLEN_CONFIGS:
        ok &= run_varlen(*cfg)
    for cfg in PAGED_CONFIGS:
        ok &= run_paged(*cfg)
    for cfg in PAGED_VARLEN_CONFIGS:
        ok &= run_paged_varlen(*cfg)
    for cfg in SPLITK_CONFIGS:
        ok &= run_splitk(*cfg)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
