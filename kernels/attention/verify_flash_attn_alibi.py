"""Correctness gate for the gfx950 dualwave flash-attention ALiBi path.

Checks build_flash_attn_dualwave_swp_module(has_alibi=True) against an fp32 torch
reference, and against aiter's flash-attn with the same slopes. ALiBi is
orthogonal to the Q mode and to the KV source, so this covers dense,
packed-varlen, paged (linear + vectorized KV cache), paged+varlen, and split-K.

ALiBi follows aiter's contract (ck-tile block_position_encoding.hpp):

  alibi_slopes is fp32, (num_heads,) or (batch, num_heads), values positive, and

      score(i, j) += -slope * | i + seqlen_kv - seqlen_q - j |

  is added *after* the 1/sqrt(D) scaling (the slope is not divided by it). The
  distance is bottom-right aligned, the same alignment the causal mask uses, and
  it is an *absolute* value, so the bias decays in both directions.

  The (num_heads,) form broadcasts over batch; both forms are exercised here so
  the kernel's alibi_stride_b handling is covered.

Unlike the dense-bias path -- which aiter rejects for paged KV -- aiter supports
alibi_slopes on flash_attn_func and flash_attn_varlen_func, so the NON-CAUSAL
dense, varlen and split-K rows get a real second opinion on top of the torch
reference. Paged rows gate on the fp32 reference alone, with a deliberately
shuffled block table so a page-order bug cannot alias into a pass.

AITER_CAUSAL_ALIBI_BROKEN: aiter's causal + alibi_slopes path is unusable in
this build, so causal rows skip the aiter cross-check. Measured here on gfx950:

  - varlen causal + alibi_slopes *hard-faults the process* ("Memory access fault
    ... on address 0x1000"). Reproduced standalone with no FlyDSL kernel in the
    picture, so it would abort this whole gate rather than fail one row.
  - dense causal + alibi_slopes returns a wrong answer: cos 0.964 against the
    fp32 reference, degrading on every head. Sweeping a scale factor on the
    slope (log2e, 1/log2e, sqrt(D), ...) does not explain it -- the unscaled
    form is the closest match -- so it is not a convention mismatch.

Both torch formulations of ALiBi (the general -slope*|dist| and aiter's
AlibiMode::VERTICAL +slope*k_col causal shortcut) agree to cos 1.000000, as they
must: they differ by a per-row constant that softmax cancels. The kernel matches
both to 0.999998, so the causal path is covered by the reference even without
aiter. Only LSE differs between the two forms; we emit the general one.

Usage:  PYTHONPATH=/var/home/bias/FlyDSL python3 verify_flash_attn_alibi.py
"""

import sys

import torch

from kernels.attention.flash_attn_gfx950 import build_flash_attn_dualwave_swp_module
from kernels.attention.flash_attn_utils import dualwave_splitk_workspace_elems

try:
    from aiter.ops.mha import flash_attn_func as aiter_flash_attn_func
    from aiter.ops.mha import flash_attn_varlen_func as aiter_flash_attn_varlen_func
except Exception as e:  # aiter not installed / import failure
    aiter_flash_attn_func = None
    aiter_flash_attn_varlen_func = None
    print(f"[warn] aiter unavailable: {e}", file=sys.stderr)

torch.manual_seed(0)

# aiter's causal + alibi_slopes path is broken (see module docstring): varlen
# faults the process outright and dense returns a wrong answer, so it is not a
# usable baseline. Causal rows report nan for aiter and gate on the fp32
# reference alone. Flip to False to re-measure if aiter is ever fixed.
AITER_CAUSAL_ALIBI_BROKEN = True

D = 128
dtype = torch.bfloat16


def _aiter_usable(causal):
    return not (causal and AITER_CAUSAL_ALIBI_BROKEN)


def make_slopes(B, H, two_d):
    """Positive fp32 slopes, (batch, nheads) if two_d else (nheads,).

    Mixes the canonical geometric ALiBi ladder with a per-head jitter so a bug
    that swapped or broadcast the head index cannot alias into a pass.
    """
    base = torch.tensor([2.0 ** (-((h + 1) * 8.0 / H)) for h in range(H)], dtype=torch.float32, device="cuda")
    if not two_d:
        return base.contiguous()
    jitter = 1.0 + 0.25 * torch.arange(B, dtype=torch.float32, device="cuda")[:, None]
    return (base[None, :] * jitter).contiguous()


def alibi_matrix(slopes, Sq, Skv):
    """-slope * |i + Skv - Sq - j| -> (B or 1, H, Sq, Skv) fp32."""
    dev = slopes.device
    i = torch.arange(Sq, device=dev)[:, None]
    j = torch.arange(Skv, device=dev)[None, :]
    rel = (i + (Skv - Sq) - j).abs().to(torch.float32)
    s = slopes.float()
    if s.dim() == 1:
        s = s.unsqueeze(0)
    return -s[:, :, None, None] * rel


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def _report(label, c_ref, mx, c_ait=float("nan")):
    ok = c_ref > 0.999
    print(
        f"{label} | cos_ref={c_ref:.6f} maxabs={mx:.4f} | cos_aiter={c_ait:.6f}{'' if ok else '   <-- ACC FAIL'}",
        flush=True,
    )
    torch.cuda.empty_cache()
    return ok


def ref_fp32(q, k, v, slopes, causal, bias=None):
    """softmax(q @ k^T * scale + alibi [+ bias]) @ v in fp32, bottom-right causal."""
    B, S, H, Dh = q.shape
    Skv = k.shape[1]
    group = H // k.shape[2]
    qf = q.float().permute(0, 2, 1, 3)
    kf = k.float().permute(0, 2, 1, 3).repeat_interleave(group, dim=1)
    vf = v.float().permute(0, 2, 1, 3).repeat_interleave(group, dim=1)
    sc = torch.matmul(qf, kf.transpose(-1, -2)) * (1.0 / Dh**0.5)
    sc = sc + alibi_matrix(slopes, S, Skv)
    if bias is not None:
        sc = sc + bias.float().view(1, 1, S, Skv)
    if causal:
        # Bottom-right aligned: query i attends keys <= i + (Skv - S).
        i = torch.arange(S, device=q.device)[:, None]
        j = torch.arange(Skv, device=q.device)[None, :]
        sc = sc.masked_fill(j > i + (Skv - S), float("-inf"))
    out = torch.matmul(torch.softmax(sc, dim=-1), vf)
    return out.permute(0, 2, 1, 3).contiguous()


def run(B, S, H, H_KV, causal, two_d, with_bias=False):
    q = torch.randn(B, S, H, D, dtype=dtype, device="cuda")
    k = torch.randn(B, S, H_KV, D, dtype=dtype, device="cuda")
    v = torch.randn(B, S, H_KV, D, dtype=dtype, device="cuda")
    slopes = make_slopes(B, H, two_d)
    bias = torch.randn(S, S, dtype=dtype, device="cuda") if with_bias else None
    out = torch.empty(B, S, H, D, dtype=dtype, device="cuda")

    exe = build_flash_attn_dualwave_swp_module(
        num_heads=H,
        head_dim=D,
        causal=causal,
        dtype_str="bf16",
        num_kv_heads=H_KV,
        has_alibi=True,
        has_bias=with_bias,
    )
    kwargs = dict(alibi_slopes=slopes, stream=torch.cuda.current_stream())
    if with_bias:
        kwargs["bias"] = bias
    exe(q, k, v, out, B, S, **kwargs)
    torch.cuda.synchronize()

    ref = ref_fp32(q, k, v, slopes, causal, bias=bias)

    # aiter treats bias and alibi_slopes as mutually exclusive, so the combined
    # row gates on the torch reference alone.
    c_ait = float("nan")
    if aiter_flash_attn_func is not None and not with_bias and _aiter_usable(causal):
        try:
            oa = aiter_flash_attn_func(q, k, v, causal=causal, alibi_slopes=slopes)
            torch.cuda.synchronize()
            c_ait = cos(oa, out)
        except Exception as e:
            print(f"[warn] aiter failed: {e}", file=sys.stderr)

    label = (
        f"dense  B={B} S={S:>5} H={H:>3} Hkv={H_KV:>3} causal={int(causal)} "
        f"slopes={'2D' if two_d else '1D'}{' +bias' if with_bias else ''}"
    )
    r = _report(label, cos(ref, out), (ref.float() - out.float()).abs().max().item(), c_ait)
    del q, k, v, out, ref
    return r


def ref_fp32_varlen(q, k, v, slopes, cu_q, cu_kv, causal):
    """Per-sequence fp32 reference over packed varlen Q/K/V, bottom-right causal."""
    total_q, H, Dh = q.shape
    group = H // k.shape[1]
    out = torch.zeros(total_q, H, Dh, dtype=torch.float32, device=q.device)
    for b in range(len(cu_q) - 1):
        sq = cu_q[b + 1] - cu_q[b]
        skv = cu_kv[b + 1] - cu_kv[b]
        qb = q[cu_q[b] : cu_q[b + 1]].float().permute(1, 0, 2)
        kb = k[cu_kv[b] : cu_kv[b + 1]].float().permute(1, 0, 2).repeat_interleave(group, dim=0)
        vb = v[cu_kv[b] : cu_kv[b + 1]].float().permute(1, 0, 2).repeat_interleave(group, dim=0)
        sc = torch.matmul(qb, kb.transpose(-1, -2)) * (1.0 / Dh**0.5)
        # Per-sequence lengths drive the alignment, matching the kernel's
        # per-batch delta_i32 = seqlen_kv - seqlen_q.
        slopes_b = slopes[b] if slopes.dim() == 2 else slopes
        sc = sc + alibi_matrix(slopes_b, sq, skv)[0]
        if causal:
            i = torch.arange(sq, device=q.device)[:, None]
            j = torch.arange(skv, device=q.device)[None, :]
            masked = j > i + (skv - sq)
            p = torch.softmax(sc.masked_fill(masked, float("-inf")), dim=-1)
            # seqlen_q > seqlen_kv leaves leading rows fully masked; torch softmax
            # gives NaN there while the kernel zeroes the O block.
            p = p.masked_fill(masked.all(dim=-1)[None, :, None], 0.0)
        else:
            p = torch.softmax(sc, dim=-1)
        out[cu_q[b] : cu_q[b + 1]] = torch.matmul(p, vb).permute(1, 0, 2)
    return out


def run_varlen(seqlens_q, seqlens_kv, H, H_KV, causal, two_d):
    vq = list(seqlens_q)
    vkv = list(seqlens_kv) if seqlens_kv is not None else list(vq)
    B = len(vq)
    cross = any(a != b for a, b in zip(vq, vkv))
    cu_q, cu_kv = [0], [0]
    for s in vq:
        cu_q.append(cu_q[-1] + s)
    for s in vkv:
        cu_kv.append(cu_kv[-1] + s)
    total_q, total_kv = cu_q[-1], cu_kv[-1]
    max_sq, max_skv = max(vq), max(vkv)

    q = torch.randn(total_q, H, D, dtype=dtype, device="cuda")
    k = torch.randn(total_kv, H_KV, D, dtype=dtype, device="cuda")
    v = torch.randn(total_kv, H_KV, D, dtype=dtype, device="cuda")
    slopes = make_slopes(B, H, two_d)
    out = torch.empty(total_q, H, D, dtype=dtype, device="cuda")
    cu_q_t = torch.tensor(cu_q, dtype=torch.int32, device="cuda")
    cu_kv_t = torch.tensor(cu_kv, dtype=torch.int32, device="cuda")

    exe = build_flash_attn_dualwave_swp_module(
        num_heads=H,
        head_dim=D,
        causal=causal,
        dtype_str="bf16",
        num_kv_heads=H_KV,
        varlen=True,
        cross_seqlen=cross,
        has_alibi=True,
    )
    kwargs = dict(
        cu_seqlens_q=cu_q_t,
        cu_seqlens_kv=cu_kv_t,
        alibi_slopes=slopes,
        stream=torch.cuda.current_stream(),
    )
    if cross:
        kwargs["seq_len_kv"] = max_skv
    exe(q, k, v, out, B, max_sq, **kwargs)
    torch.cuda.synchronize()

    ref = ref_fp32_varlen(q, k, v, slopes, cu_q, cu_kv, causal)

    c_ait = float("nan")
    if aiter_flash_attn_varlen_func is not None and _aiter_usable(causal):
        try:
            oa = aiter_flash_attn_varlen_func(
                q, k, v, cu_q_t, cu_kv_t, max_sq, max_skv, causal=causal, alibi_slopes=slopes
            )
            if isinstance(oa, tuple):
                oa = oa[0]
            torch.cuda.synchronize()
            c_ait = cos(oa, out)
        except Exception as e:
            print(f"[warn] aiter varlen failed: {e}", file=sys.stderr)

    label = (
        f"varlen q={str(vq):>24} kv={str(vkv) if seqlens_kv else '(self)':>24} "
        f"H={H:>3} Hkv={H_KV:>3} causal={int(causal)} slopes={'2D' if two_d else '1D'}"
    )
    r = _report(label, cos(ref, out), (ref - out.float()).abs().max().item(), c_ait)
    del q, k, v, out, ref
    return r


PAGE_SIZE = 64  # must equal traits.BLOCK_N; the block table is indexed by tile


def _vectorize_paged_kv(k4, v4, h_kv, page_size, kvs):
    """aiter-style 5D paged K/V from the 4D [pages, page_size, Hkv, D] form."""
    k5 = k4.contiguous().view(-1, page_size, h_kv, D // kvs, kvs).permute(0, 2, 3, 1, 4).contiguous()
    v5 = v4.contiguous().view(-1, page_size // kvs, kvs, h_kv, D).permute(0, 3, 1, 4, 2).contiguous()
    return k5, v5


def _paged_cache(n_pages, H_KV, layout):
    """Physical K/V cache plus the 4D view the reference gathers through."""
    k4 = torch.randn(n_pages, PAGE_SIZE, H_KV, D, dtype=dtype, device="cuda")
    v4 = torch.randn(n_pages, PAGE_SIZE, H_KV, D, dtype=dtype, device="cuda")
    if layout == "vectorized":
        kvs = 16 // torch.empty((), dtype=dtype).element_size()
        return (*_vectorize_paged_kv(k4, v4, H_KV, PAGE_SIZE, kvs), k4, v4)
    return k4, v4, k4, v4


def _attn_ref(qb, kb, vb, alibi_b, causal):
    """fp32 [H, Sq, D] attention for one sequence; alibi_b is [H, Sq, Skv]."""
    sq, skv = qb.shape[1], kb.shape[1]
    sc = torch.matmul(qb, kb.transpose(-1, -2)) * (1.0 / D**0.5) + alibi_b
    if causal:
        i = torch.arange(sq, device=qb.device)[:, None]
        j = torch.arange(skv, device=qb.device)[None, :]
        masked = j > i + (skv - sq)
        p = torch.softmax(sc.masked_fill(masked, float("-inf")), dim=-1)
        # seqlen_q > seqlen_kv leaves leading rows fully masked; torch softmax
        # gives NaN there while the kernel zeroes the O block.
        p = p.masked_fill(masked.all(dim=-1)[None, :, None], 0.0)
    else:
        p = torch.softmax(sc, dim=-1)
    return torch.matmul(p, vb)


def run_paged(B, Sq, Skv, H, H_KV, causal, layout, two_d):
    n_pages_per_seq = (Skv + PAGE_SIZE - 1) // PAGE_SIZE
    # Spare pages + randperm: the logical->physical map is deliberately not the
    # identity, so a kernel that ignored the block table would fail loudly.
    total_pages = B * n_pages_per_seq + 8
    k_cache, v_cache, k4, v4 = _paged_cache(total_pages, H_KV, layout)
    perm = torch.randperm(total_pages, device="cuda")[: B * n_pages_per_seq].view(B, n_pages_per_seq)

    q = torch.randn(B, Sq, H, D, dtype=dtype, device="cuda")
    slopes = make_slopes(B, H, two_d)
    out = torch.empty(B, Sq, H, D, dtype=dtype, device="cuda")
    cross = Skv != Sq

    exe = build_flash_attn_dualwave_swp_module(
        num_heads=H,
        head_dim=D,
        causal=causal,
        dtype_str="bf16",
        num_kv_heads=H_KV,
        paged=True,
        cross_seqlen=cross,
        kv_cache_layout=layout,
        has_alibi=True,
    )
    kwargs = dict(
        block_table=perm.to(torch.int32).contiguous().reshape(-1),
        block_table_stride=n_pages_per_seq,
        alibi_slopes=slopes,
        stream=torch.cuda.current_stream(),
    )
    if cross:
        kwargs["seq_len_kv"] = Skv
    exe(q, k_cache, v_cache, out, B, Sq, **kwargs)
    torch.cuda.synchronize()

    group = H // H_KV
    kd = k4[perm].reshape(B, n_pages_per_seq * PAGE_SIZE, H_KV, D)[:, :Skv]
    vd = v4[perm].reshape(B, n_pages_per_seq * PAGE_SIZE, H_KV, D)[:, :Skv]
    alibi = alibi_matrix(slopes, Sq, Skv)
    ref = torch.empty(B, H, Sq, D, dtype=torch.float32, device="cuda")
    for b in range(B):
        ref[b] = _attn_ref(
            q[b].float().permute(1, 0, 2),
            kd[b].float().permute(1, 0, 2).repeat_interleave(group, dim=0),
            vd[b].float().permute(1, 0, 2).repeat_interleave(group, dim=0),
            alibi[b] if alibi.shape[0] > 1 else alibi[0],
            causal,
        )
    ref = ref.permute(0, 2, 1, 3)
    return _report(
        f"paged  {layout:>10} B={B} Sq={Sq:>5} Skv={Skv:>5} H={H:>3} Hkv={H_KV:>3} "
        f"causal={int(causal)} slopes={'2D' if two_d else '1D'}",
        cos(ref, out),
        (ref - out.float()).abs().max().item(),
    )


def run_paged_varlen(vq, vkv, H, H_KV, causal, layout, two_d):
    B = len(vq)
    cu_q, cu_kv = [0], [0]
    for s in vq:
        cu_q.append(cu_q[-1] + s)
    for s in vkv:
        cu_kv.append(cu_kv[-1] + s)
    total_q, Sq, Skv = cu_q[-1], max(vq), max(vkv)
    n_pages_per_seq = (Skv + PAGE_SIZE - 1) // PAGE_SIZE
    total_pages = B * n_pages_per_seq + 8
    k_cache, v_cache, k4, v4 = _paged_cache(total_pages, H_KV, layout)
    perm = torch.randperm(total_pages, device="cuda")[: B * n_pages_per_seq].view(B, n_pages_per_seq)

    q = torch.randn(total_q, H, D, dtype=dtype, device="cuda")
    slopes = make_slopes(B, H, two_d)
    out = torch.empty(total_q, H, D, dtype=dtype, device="cuda")

    exe = build_flash_attn_dualwave_swp_module(
        num_heads=H,
        head_dim=D,
        causal=causal,
        dtype_str="bf16",
        num_kv_heads=H_KV,
        paged=True,
        varlen=True,
        cross_seqlen=True,
        kv_cache_layout=layout,
        has_alibi=True,
    )
    exe(
        q,
        k_cache,
        v_cache,
        out,
        B,
        Sq,
        cu_seqlens_q=torch.tensor(cu_q, dtype=torch.int32, device="cuda"),
        cu_seqlens_kv=torch.tensor(cu_kv, dtype=torch.int32, device="cuda"),
        seq_len_kv=Skv,
        block_table=perm.to(torch.int32).contiguous().reshape(-1),
        block_table_stride=n_pages_per_seq,
        alibi_slopes=slopes,
        stream=torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()

    group = H // H_KV
    ref = torch.zeros(total_q, H, D, dtype=torch.float32, device="cuda")
    for b in range(B):
        skv = vkv[b]
        sq = vq[b]
        kb = k4[perm[b]].reshape(-1, H_KV, D)[:skv].float().permute(1, 0, 2).repeat_interleave(group, dim=0)
        vb = v4[perm[b]].reshape(-1, H_KV, D)[:skv].float().permute(1, 0, 2).repeat_interleave(group, dim=0)
        slopes_b = slopes[b] if slopes.dim() == 2 else slopes
        o = _attn_ref(
            q[cu_q[b] : cu_q[b + 1]].float().permute(1, 0, 2),
            kb,
            vb,
            alibi_matrix(slopes_b, sq, skv)[0],
            causal,
        )
        ref[cu_q[b] : cu_q[b + 1]] = o.permute(1, 0, 2)
    return _report(
        f"paged+varlen {layout:>10} q={str(vq):>22} causal={int(causal)} slopes={'2D' if two_d else '1D'}",
        cos(ref, out),
        (ref - out.float()).abs().max().item(),
    )


def run_splitk(B, S, H, H_KV, causal, splits, two_d):
    q = torch.randn(B, S, H, D, dtype=dtype, device="cuda")
    k = torch.randn(B, S, H_KV, D, dtype=dtype, device="cuda")
    v = torch.randn(B, S, H_KV, D, dtype=dtype, device="cuda")
    slopes = make_slopes(B, H, two_d)
    out = torch.empty(B, S, H, D, dtype=dtype, device="cuda")
    ws = torch.zeros(
        dualwave_splitk_workspace_elems(B, H, S, splits, head_dim=D),
        dtype=torch.float32,
        device="cuda",
    )
    exe = build_flash_attn_dualwave_swp_module(
        num_heads=H,
        head_dim=D,
        causal=causal,
        dtype_str="bf16",
        num_kv_heads=H_KV,
        num_kv_splits=splits,
        has_alibi=True,
    )
    exe(q, k, v, out, B, S, workspace=ws, alibi_slopes=slopes, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    ref = ref_fp32(q, k, v, slopes, causal)
    c_ait = float("nan")
    if aiter_flash_attn_func is not None and _aiter_usable(causal):
        try:
            oa = aiter_flash_attn_func(q, k, v, causal=causal, alibi_slopes=slopes)
            torch.cuda.synchronize()
            c_ait = cos(oa, out)
        except Exception as e:
            print(f"[warn] aiter failed: {e}", file=sys.stderr)
    return _report(
        f"splitk splits={splits} B={B} S={S:>5} H={H:>3} Hkv={H_KV:>3} "
        f"causal={int(causal)} slopes={'2D' if two_d else '1D'}",
        cos(ref, out),
        (ref - out.float()).abs().max().item(),
        c_ait,
    )


# (batch, seq_len, num_heads, num_kv_heads, causal, slopes_2d)
CONFIGS = [
    (1, 512, 8, 8, False, False),
    (1, 1024, 16, 16, False, True),
    (2, 1024, 16, 8, False, True),
    (1, 1024, 16, 16, True, False),
    (2, 2048, 32, 8, True, True),
]

# (seqlens_q, seqlens_kv or None for self-attention, num_heads, num_kv_heads, causal, slopes_2d)
# Covers ragged non-multiple-of-BLOCK_M lengths, sub-tile lengths, single-batch
# varlen, GQA, and cross-length (seqlen_q != seqlen_kv) packing. Cross-length
# rows are what actually exercise the per-batch delta in the ALiBi distance.
VARLEN_CONFIGS = [
    ([512, 512], None, 8, 8, False, False),
    ([1024, 512, 768], None, 16, 8, False, True),
    ([1000, 377, 64], None, 16, 8, False, True),
    ([2048], None, 8, 8, False, False),
    ([31, 33, 65, 127], None, 8, 8, False, True),
    ([1024, 512], [512, 1024], 16, 8, False, True),
    ([512, 512], None, 8, 8, True, False),
    ([1024, 512, 768], None, 16, 8, True, True),
    ([1000, 377, 64], None, 16, 8, True, True),
    ([2048], None, 8, 8, True, False),
    ([31, 33, 65, 127], None, 8, 8, True, True),
    ([1024, 512], [512, 1024], 16, 8, True, True),
]

# (batch, Sq, Skv, num_heads, num_kv_heads, causal, kv_cache_layout, slopes_2d)
PAGED_CONFIGS = [
    (1, 512, 512, 8, 8, False, "linear", False),
    (2, 1024, 1024, 16, 8, False, "linear", True),
    (1, 512, 1024, 8, 8, False, "linear", False),
    (2, 1024, 768, 16, 8, True, "linear", True),
    (1, 512, 512, 8, 8, True, "linear", False),
    (2, 1024, 1024, 16, 8, True, "linear", True),
    # The vectorized cache swizzles the score tile into runs of 8 columns; ALiBi
    # reuses the same _seq_pad_score_threshold helper the padding mask uses, and
    # these rows are what prove the two agree.
    (1, 512, 512, 8, 8, False, "vectorized", False),
    (2, 1024, 1024, 16, 8, False, "vectorized", True),
    (2, 1024, 768, 16, 8, True, "vectorized", True),
    (2, 1024, 1024, 16, 8, True, "vectorized", False),
]

# (seqlens_q, seqlens_kv, num_heads, num_kv_heads, causal, kv_cache_layout, slopes_2d)
PAGED_VARLEN_CONFIGS = [
    ([512, 512], [512, 512], 8, 8, False, "linear", False),
    ([1024, 512, 768], [1024, 512, 768], 16, 8, True, "linear", True),
    ([377, 64, 1000], [512, 192, 640], 16, 8, False, "linear", True),
    ([377, 64, 1000], [512, 192, 640], 16, 8, True, "linear", True),
    ([1024, 512, 768], [1024, 512, 768], 16, 8, True, "vectorized", False),
]

# (batch, seq_len, num_heads, num_kv_heads, causal, num_kv_splits, slopes_2d)
SPLITK_CONFIGS = [
    (1, 2048, 8, 8, False, 2, False),
    (1, 2048, 8, 8, False, 4, True),
    (1, 2048, 8, 8, True, 2, False),
    (1, 2048, 8, 8, True, 4, True),
    (2, 4096, 16, 8, True, 4, True),
]

# (batch, seq_len, num_heads, num_kv_heads, causal, slopes_2d) with a dense bias
# stacked on top of ALiBi -- aiter cannot express this, so torch reference only.
COMBINED_CONFIGS = [
    (1, 1024, 16, 16, False, False),
    (2, 1024, 16, 8, True, True),
]


if __name__ == "__main__":
    ok = True
    for cfg in CONFIGS:
        ok &= run(*cfg)
    for cfg in VARLEN_CONFIGS:
        ok &= run_varlen(*cfg)
    for cfg in PAGED_CONFIGS:
        ok &= run_paged(*cfg)
    for cfg in PAGED_VARLEN_CONFIGS:
        ok &= run_paged_varlen(*cfg)
    for cfg in SPLITK_CONFIGS:
        ok &= run_splitk(*cfg)
    for cfg in COMBINED_CONFIGS:
        ok &= run(*cfg, with_bias=True)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
