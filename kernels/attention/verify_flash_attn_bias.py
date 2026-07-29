"""Correctness gate for the gfx950 dualwave flash-attention bias path.

Checks build_flash_attn_dualwave_swp_module(has_bias=True) against an fp32 torch
reference, and against aiter's flash-attn with the same bias where aiter supports
the mode. Bias is orthogonal to the Q mode and to the KV source, so this covers
dense, packed-varlen, paged (linear + vectorized KV cache), paged+varlen, and
split-K.

Bias shapes follow aiter's contract:
  dense   (seqlen_q, seqlen_kv), broadcast over batch and head.
  varlen  (total_q, max_seqlen_kv), row = global packed q token index, column =
          per-batch-local key index, broadcast over head.

Paged rows have no aiter cross-check: aiter rejects bias for page attention
("Page attention does not supports bias for now"), so they gate on the fp32
torch reference alone, with a deliberately shuffled block table so a page-order
bug cannot alias into a pass.

Usage:  PYTHONPATH=/var/home/bias/FlyDSL python3 verify_flash_attn_bias.py
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

D = 128
dtype = torch.bfloat16

# (batch, seq_len, num_heads, num_kv_heads, causal)
CONFIGS = [
    (1, 512, 8, 8, False),
    (1, 1024, 16, 16, False),
    (2, 1024, 16, 8, False),
    (1, 1024, 16, 16, True),
    (2, 2048, 32, 8, True),
]

# (seqlens_q, seqlens_kv or None for self-attention, num_heads, num_kv_heads, causal)
# Covers ragged non-multiple-of-BLOCK_M lengths, sub-tile lengths, single-batch
# varlen, GQA, and cross-length (seqlen_q != seqlen_kv) packing.
VARLEN_CONFIGS = [
    ([512, 512], None, 8, 8, False),
    ([1024, 512, 768], None, 16, 8, False),
    ([1000, 377, 64], None, 16, 8, False),
    ([2048], None, 8, 8, False),
    ([31, 33, 65, 127], None, 8, 8, False),
    ([1024, 512], [512, 1024], 16, 8, False),
    ([512, 512], None, 8, 8, True),
    ([1024, 512, 768], None, 16, 8, True),
    ([1000, 377, 64], None, 16, 8, True),
    ([2048], None, 8, 8, True),
    ([31, 33, 65, 127], None, 8, 8, True),
    ([1024, 512], [512, 1024], 16, 8, True),
]


def ref_fp32(q, k, v, bias, causal):
    """softmax(q @ k^T * scale + bias) @ v in fp32, bottom-right causal."""
    B, S, H, Dh = q.shape
    Skv = k.shape[1]
    group = H // k.shape[2]
    qf = q.float().permute(0, 2, 1, 3)
    kf = k.float().permute(0, 2, 1, 3).repeat_interleave(group, dim=1)
    vf = v.float().permute(0, 2, 1, 3).repeat_interleave(group, dim=1)
    sc = torch.matmul(qf, kf.transpose(-1, -2)) * (1.0 / Dh**0.5)
    sc = sc + bias.float().view(1, 1, S, Skv)
    if causal:
        # Bottom-right aligned: query i attends keys <= i + (Skv - S).
        i = torch.arange(S, device=q.device)[:, None]
        j = torch.arange(Skv, device=q.device)[None, :]
        sc = sc.masked_fill(j > i + (Skv - S), float("-inf"))
    out = torch.matmul(torch.softmax(sc, dim=-1), vf)
    return out.permute(0, 2, 1, 3).contiguous()


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def run(B, S, H, H_KV, causal):
    q = torch.randn(B, S, H, D, dtype=dtype, device="cuda")
    k = torch.randn(B, S, H_KV, D, dtype=dtype, device="cuda")
    v = torch.randn(B, S, H_KV, D, dtype=dtype, device="cuda")
    bias = torch.randn(S, S, dtype=dtype, device="cuda")
    out = torch.empty(B, S, H, D, dtype=dtype, device="cuda")

    exe = build_flash_attn_dualwave_swp_module(
        num_heads=H,
        head_dim=D,
        causal=causal,
        dtype_str="bf16",
        num_kv_heads=H_KV,
        has_bias=True,
    )
    exe(q, k, v, out, B, S, bias=bias, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    ref = ref_fp32(q, k, v, bias, causal)
    c_ref = cos(ref, out)
    mx = (ref.float() - out.float()).abs().max().item()

    c_ait = float("nan")
    if aiter_flash_attn_func is not None:
        try:
            oa = aiter_flash_attn_func(q, k, v, causal=causal, bias=bias)
            torch.cuda.synchronize()
            c_ait = cos(oa, out)
        except Exception as e:
            print(f"[warn] aiter failed: {e}", file=sys.stderr)

    ok = c_ref > 0.999
    flag = "" if ok else "   <-- ACC FAIL"
    print(
        f"B={B} S={S:>5} H={H:>3} Hkv={H_KV:>3} causal={int(causal)} | "
        f"cos_ref={c_ref:.6f} maxabs={mx:.4f} | cos_aiter={c_ait:.6f}{flag}",
        flush=True,
    )

    del q, k, v, bias, out, ref
    torch.cuda.empty_cache()
    return ok


def ref_fp32_varlen(q, k, v, bias, cu_q, cu_kv, causal):
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
        # Rows are the batch's slice of the packed bias; columns are batch-local.
        sc = sc + bias[cu_q[b] : cu_q[b + 1], :skv].float()
        p = None
        if causal:
            i = torch.arange(sq, device=q.device)[:, None]
            j = torch.arange(skv, device=q.device)[None, :]
            masked = j > i + (skv - sq)
            sc = sc.masked_fill(masked, float("-inf"))
            p = torch.softmax(sc, dim=-1)
            # seqlen_q > seqlen_kv leaves leading rows fully masked; torch softmax
            # gives NaN there while the kernel zeroes the O block.
            p = p.masked_fill(masked.all(dim=-1)[None, :, None], 0.0)
        else:
            p = torch.softmax(sc, dim=-1)
        out[cu_q[b] : cu_q[b + 1]] = torch.matmul(p, vb).permute(1, 0, 2)
    return out


def run_varlen(seqlens_q, seqlens_kv, H, H_KV, causal):
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
    bias = torch.randn(total_q, max_skv, dtype=dtype, device="cuda")
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
        has_bias=True,
    )
    kwargs = dict(
        cu_seqlens_q=cu_q_t,
        cu_seqlens_kv=cu_kv_t,
        bias=bias,
        stream=torch.cuda.current_stream(),
    )
    if cross:
        kwargs["seq_len_kv"] = max_skv
    exe(q, k, v, out, B, max_sq, **kwargs)
    torch.cuda.synchronize()

    ref = ref_fp32_varlen(q, k, v, bias, cu_q, cu_kv, causal)
    c_ref = cos(ref, out)
    mx = (ref - out.float()).abs().max().item()

    c_ait = float("nan")
    if aiter_flash_attn_varlen_func is not None:
        try:
            oa = aiter_flash_attn_varlen_func(q, k, v, cu_q_t, cu_kv_t, max_sq, max_skv, causal=causal, bias=bias)
            if isinstance(oa, tuple):
                oa = oa[0]
            torch.cuda.synchronize()
            c_ait = cos(oa, out)
        except Exception as e:
            print(f"[warn] aiter varlen failed: {e}", file=sys.stderr)

    ok = c_ref > 0.999
    flag = "" if ok else "   <-- ACC FAIL"
    print(
        f"varlen q={str(vq):>24} kv={str(vkv) if seqlens_kv else '(self)':>24} "
        f"H={H:>3} Hkv={H_KV:>3} causal={int(causal)} | "
        f"cos_ref={c_ref:.6f} maxabs={mx:.4f} | cos_aiter={c_ait:.6f}{flag}",
        flush=True,
    )

    del q, k, v, bias, out, ref
    torch.cuda.empty_cache()
    return ok


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


def _attn_ref(qb, kb, vb, bias_b, causal):
    """fp32 [H, Sq, D] attention for one sequence; bias_b is [Sq, Skv]."""
    sq, skv = qb.shape[1], kb.shape[1]
    sc = torch.matmul(qb, kb.transpose(-1, -2)) * (1.0 / D**0.5) + bias_b.float()
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


def _report(label, c_ref, mx, c_ait=float("nan")):
    ok = c_ref > 0.999
    print(
        f"{label} | cos_ref={c_ref:.6f} maxabs={mx:.4f} | cos_aiter={c_ait:.6f}{'' if ok else '   <-- ACC FAIL'}",
        flush=True,
    )
    torch.cuda.empty_cache()
    return ok


def run_paged(B, Sq, Skv, H, H_KV, causal, layout):
    n_pages_per_seq = (Skv + PAGE_SIZE - 1) // PAGE_SIZE
    # Spare pages + randperm: the logical->physical map is deliberately not the
    # identity, so a kernel that ignored the block table would fail loudly.
    total_pages = B * n_pages_per_seq + 8
    k_cache, v_cache, k4, v4 = _paged_cache(total_pages, H_KV, layout)
    perm = torch.randperm(total_pages, device="cuda")[: B * n_pages_per_seq].view(B, n_pages_per_seq)

    q = torch.randn(B, Sq, H, D, dtype=dtype, device="cuda")
    bias = torch.randn(Sq, Skv, dtype=dtype, device="cuda")
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
        has_bias=True,
    )
    kwargs = dict(
        block_table=perm.to(torch.int32).contiguous().reshape(-1),
        block_table_stride=n_pages_per_seq,
        bias=bias,
        stream=torch.cuda.current_stream(),
    )
    if cross:
        kwargs["seq_len_kv"] = Skv
    exe(q, k_cache, v_cache, out, B, Sq, **kwargs)
    torch.cuda.synchronize()

    group = H // H_KV
    kd = k4[perm].reshape(B, n_pages_per_seq * PAGE_SIZE, H_KV, D)[:, :Skv]
    vd = v4[perm].reshape(B, n_pages_per_seq * PAGE_SIZE, H_KV, D)[:, :Skv]
    ref = torch.empty(B, H, Sq, D, dtype=torch.float32, device="cuda")
    for b in range(B):
        ref[b] = _attn_ref(
            q[b].float().permute(1, 0, 2),
            kd[b].float().permute(1, 0, 2).repeat_interleave(group, dim=0),
            vd[b].float().permute(1, 0, 2).repeat_interleave(group, dim=0),
            bias,
            causal,
        )
    ref = ref.permute(0, 2, 1, 3)
    return _report(
        f"paged  {layout:>10} B={B} Sq={Sq:>5} Skv={Skv:>5} H={H:>3} Hkv={H_KV:>3} causal={int(causal)}",
        cos(ref, out),
        (ref - out.float()).abs().max().item(),
    )


def run_paged_varlen(vq, vkv, H, H_KV, causal, layout):
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
    bias = torch.randn(total_q, Skv, dtype=dtype, device="cuda")
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
        has_bias=True,
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
        bias=bias,
        stream=torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()

    group = H // H_KV
    ref = torch.zeros(total_q, H, D, dtype=torch.float32, device="cuda")
    for b in range(B):
        skv = vkv[b]
        kb = k4[perm[b]].reshape(-1, H_KV, D)[:skv].float().permute(1, 0, 2).repeat_interleave(group, dim=0)
        vb = v4[perm[b]].reshape(-1, H_KV, D)[:skv].float().permute(1, 0, 2).repeat_interleave(group, dim=0)
        o = _attn_ref(
            q[cu_q[b] : cu_q[b + 1]].float().permute(1, 0, 2),
            kb,
            vb,
            bias[cu_q[b] : cu_q[b + 1], :skv],
            causal,
        )
        ref[cu_q[b] : cu_q[b + 1]] = o.permute(1, 0, 2)
    return _report(
        f"paged+varlen {layout:>10} q={str(vq):>22} causal={int(causal)}",
        cos(ref, out),
        (ref - out.float()).abs().max().item(),
    )


def run_splitk(B, S, H, H_KV, causal, splits):
    q = torch.randn(B, S, H, D, dtype=dtype, device="cuda")
    k = torch.randn(B, S, H_KV, D, dtype=dtype, device="cuda")
    v = torch.randn(B, S, H_KV, D, dtype=dtype, device="cuda")
    bias = torch.randn(S, S, dtype=dtype, device="cuda")
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
        has_bias=True,
    )
    exe(q, k, v, out, B, S, workspace=ws, bias=bias, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    ref = ref_fp32(q, k, v, bias, causal)
    c_ait = float("nan")
    if aiter_flash_attn_func is not None:
        try:
            oa = aiter_flash_attn_func(q, k, v, causal=causal, bias=bias)
            torch.cuda.synchronize()
            c_ait = cos(oa, out)
        except Exception as e:
            print(f"[warn] aiter failed: {e}", file=sys.stderr)
    return _report(
        f"splitk splits={splits} B={B} S={S:>5} H={H:>3} Hkv={H_KV:>3} causal={int(causal)}",
        cos(ref, out),
        (ref - out.float()).abs().max().item(),
        c_ait,
    )


# (batch, Sq, Skv, num_heads, num_kv_heads, causal, kv_cache_layout)
PAGED_CONFIGS = [
    (1, 512, 512, 8, 8, False, "linear"),
    (2, 1024, 1024, 16, 8, False, "linear"),
    (1, 512, 1024, 8, 8, False, "linear"),
    (2, 1024, 768, 16, 8, True, "linear"),
    (1, 512, 512, 8, 8, True, "linear"),
    (2, 1024, 1024, 16, 8, True, "linear"),
    # The vectorized cache swizzles the score tile into runs of 8 columns; this
    # is the case that caught _bias_half hardcoding the linear run-of-4 layout.
    (1, 512, 512, 8, 8, False, "vectorized"),
    (2, 1024, 1024, 16, 8, False, "vectorized"),
    (2, 1024, 768, 16, 8, True, "vectorized"),
    (2, 1024, 1024, 16, 8, True, "vectorized"),
]

# (seqlens_q, seqlens_kv, num_heads, num_kv_heads, causal, kv_cache_layout)
PAGED_VARLEN_CONFIGS = [
    ([512, 512], [512, 512], 8, 8, False, "linear"),
    ([1024, 512, 768], [1024, 512, 768], 16, 8, True, "linear"),
    ([377, 64, 1000], [512, 192, 640], 16, 8, False, "linear"),
    ([377, 64, 1000], [512, 192, 640], 16, 8, True, "linear"),
    ([1024, 512, 768], [1024, 512, 768], 16, 8, True, "vectorized"),
]

# (batch, seq_len, num_heads, num_kv_heads, causal, num_kv_splits)
SPLITK_CONFIGS = [
    (1, 2048, 8, 8, False, 2),
    (1, 2048, 8, 8, False, 4),
    (1, 2048, 8, 8, True, 2),
    (1, 2048, 8, 8, True, 4),
    (2, 4096, 16, 8, True, 4),
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
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
