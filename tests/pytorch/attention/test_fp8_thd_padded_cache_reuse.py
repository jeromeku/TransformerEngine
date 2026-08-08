"""Graph-cache reuse across contiguous and padded FP8 THD batches must not corrupt either.

Admitting physical gaps (see test_packed_padded_offsets.py) means a contiguous batch
(cu_seqlens_padded == cu_seqlens) and a padded batch (!=) with the same bucketed (b, s_q, s_kv) now
share one cached cuDNN graph. Before, padded was rejected, so only contiguous batches ever touched
this cache; that cross-type reuse is newly reachable and is what this file measures. The offsets are
supplied to the graph at runtime, so reuse should be correct -- but "should" is not evidence, and a
wrong graph reused across the boundary would corrupt only the second batch's tail documents with no
error, which is exactly the kind of silent fault a test has to catch.

Each scenario runs in a subprocess (the FP8 graph cache is process-lifetime, so a second test in the
same process would see the first's entries). The subprocess:

  * confirms the two batches collapse to ONE graph in the relevant direction -- otherwise the reuse
    under test never happened and the correctness assertion would be vacuous (fprop/bprop entries);
  * checks the batch that reused the other's graph against a per-document dense reference.

Both orders (contiguous-first, padded-first) for the forward, and the backward cache, are covered.
"""

import json
import os
import subprocess
import sys
import textwrap

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="device-only")

CACHE_STATS = "NVTE_FP8_ATTN_CACHE_STATS"
FORWARD_TOL = 0.12  # FP8 padded/contiguous vs BF16 dense reference; measured ~0.04
BACKWARD_TOL = 0.20  # FP8 grads vs BF16 padded grads; measured ~0.10

_WORKLOAD = textwrap.dedent(
    """
    import json, math, sys, torch
    import transformer_engine.pytorch as te
    import transformer_engine_torch as tex
    from transformer_engine.common import recipe

    mode = sys.argv[1]
    H, G, D, DEV = 8, 4, 128, "cuda"
    # Two documents in each batch; the longest of each lands in (512, 1024] so both bucket to the
    # same graph key. Contiguous: padded == actual. Padded: real gaps, longest padded 640.
    CONTIG = [400, 600]
    ACTUAL, PADDED = [350, 500], [512, 640]

    def module(fmt, mask):
        return te.DotProductAttention(
            num_attention_heads=H, kv_channels=D, num_gqa_groups=G, attention_dropout=0.0,
            qkv_format=fmt, attn_mask_type=mask, softmax_scale=1.0 / math.sqrt(D)).cuda()

    def cu(lens):
        return torch.tensor([0] + list(torch.cumsum(torch.tensor(lens), 0)),
                            dtype=torch.int32, device=DEV)

    def rand(n, c, gen):
        return torch.randn(n, c, D, generator=gen, device=DEV, dtype=torch.bfloat16) * 0.5

    def dense_reference(per_doc):
        m = module("bshd", "causal")
        return [m(qi.unsqueeze(0), ki.unsqueeze(0), vi.unsqueeze(0),
                  attn_mask_type="causal").reshape(qi.shape[0], H, D) for qi, ki, vi in per_doc]

    def per_doc_err(out, offs, lens, ref):
        e = []
        for i, a in enumerate(lens):
            b = int(offs[i]); g = out[b:b + a].float(); w = ref[i].float()
            e.append((g - w).pow(2).mean().sqrt().item() / max(w.pow(2).mean().sqrt().item(), 1e-12))
        return e

    def build(lens, padded_lens, seed):
        gen = torch.Generator(device=DEV).manual_seed(seed)
        offs = cu(padded_lens)
        q = torch.zeros(sum(padded_lens), H, D, device=DEV, dtype=torch.bfloat16)
        k = torch.zeros(sum(padded_lens), G, D, device=DEV, dtype=torch.bfloat16)
        v = torch.zeros(sum(padded_lens), G, D, device=DEV, dtype=torch.bfloat16)
        per_doc = []
        for a, base in zip(lens, offs[:-1].tolist()):
            qi, ki, vi = rand(a, H, gen), rand(a, G, gen), rand(a, G, gen)
            q[base:base + a], k[base:base + a], v[base:base + a] = qi, ki, vi
            per_doc.append((qi, ki, vi))
        return q, k, v, offs, per_doc

    def forward(lens, padded_lens, seed):
        q, k, v, offs, per_doc = build(lens, padded_lens, seed)
        with te.fp8_autocast(enabled=True, fp8_recipe=recipe.DelayedScaling(fp8_dpa=True)):
            out = module("thd", "padding_causal")(
                q, k, v, cu_seqlens_q=cu(lens), cu_seqlens_kv=cu(lens),
                cu_seqlens_q_padded=offs, cu_seqlens_kv_padded=offs,
                max_seqlen_q=max(padded_lens), max_seqlen_kv=max(padded_lens))
        return per_doc_err(out.reshape(sum(padded_lens), H, D), offs, lens, dense_reference(per_doc))

    def backward_err(lens, padded_lens, seed):
        q, k, v, offs, _ = build(lens, padded_lens, seed)
        grad = torch.zeros_like(q)
        g = torch.Generator(device=DEV).manual_seed(seed + 1)
        for a, base in zip(lens, offs[:-1].tolist()):
            grad[base:base + a] = rand(a, H, g)
        kwargs = dict(cu_seqlens_q=cu(lens), cu_seqlens_kv=cu(lens),
                      cu_seqlens_q_padded=offs, cu_seqlens_kv_padded=offs,
                      max_seqlen_q=max(padded_lens), max_seqlen_kv=max(padded_lens))
        def run(fp8):
            qg, kg, vg = (x.clone().detach().requires_grad_(True) for x in (q, k, v))
            if fp8:
                with te.fp8_autocast(enabled=True, fp8_recipe=recipe.DelayedScaling(fp8_dpa=True)):
                    out = module("thd", "padding_causal")(qg, kg, vg, **kwargs)
            else:
                out = module("thd", "padding_causal")(qg, kg, vg, **kwargs)
            out.reshape(sum(padded_lens), H, D).backward(grad)
            return qg.grad, kg.grad, vg.grad
        b16, f8 = run(False), run(True)
        out = {}
        for name, a8, a16 in zip(("dq", "dk", "dv"), f8, b16):
            out[name] = per_doc_err(a8, offs, lens,
                [a16[int(offs[j]):int(offs[j]) + lens[j]] for j in range(len(lens))])
        return out

    tex.reset_fused_attn_fp8_cache_stats()
    result = {}
    if mode == "fwd_contig_then_padded":
        forward(CONTIG, CONTIG, 1)                     # contiguous builds the graph
        result["err"] = forward(ACTUAL, PADDED, 2)     # padded reuses it
    elif mode == "fwd_padded_then_contig":
        forward(ACTUAL, PADDED, 2)                     # padded builds the graph
        result["err"] = forward(CONTIG, CONTIG, 1)     # contiguous reuses it
    elif mode == "bwd_contig_then_padded":
        backward_err(CONTIG, CONTIG, 1)                # contiguous builds fwd+bwd graphs
        result["err"] = backward_err(ACTUAL, PADDED, 2)  # padded reuses them
    torch.cuda.synchronize()
    result["stats"] = tex.get_fused_attn_fp8_cache_stats()
    print("result " + json.dumps(result))
    """
)


def _run(mode: str, timeout: int = 900) -> dict:
    env = dict(os.environ, **{CACHE_STATS: "1"})
    proc = subprocess.run(
        [sys.executable, "-c", _WORKLOAD, mode],
        capture_output=True, text=True, timeout=timeout, env=env,
    )
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("result ")), None)
    assert line is not None, f"workload failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    return json.loads(line[len("result ") :])


def _assert_single_graph(stats: dict, direction: str):
    assert stats["enabled"], "cache-stats collection is off; every count below would be a vacuous 0"
    entries = stats[f"{direction}_entries"]
    assert entries == 1, (
        f"the two batches built {entries} {direction} graphs, so they did not share one and the "
        "reuse under test never happened -- adjust the lengths so both land in one bucket"
    )
    assert stats[f"{direction}_hits"] > 0, f"no {direction} cache hit occurred"


def test_padded_reuses_contiguous_forward_graph():
    r = _run("fwd_contig_then_padded")
    _assert_single_graph(r["stats"], "fprop")
    for i, e in enumerate(r["err"]):
        assert e < FORWARD_TOL, f"padded doc {i} via reused contiguous graph: {e:.5f}"


def test_contiguous_reuses_padded_forward_graph():
    r = _run("fwd_padded_then_contig")
    _assert_single_graph(r["stats"], "fprop")
    for i, e in enumerate(r["err"]):
        assert e < FORWARD_TOL, f"contiguous doc {i} via reused padded graph: {e:.5f}"


def test_padded_reuses_contiguous_backward_graph():
    r = _run("bwd_contig_then_padded")
    _assert_single_graph(r["stats"], "bprop")
    for name, errs in r["err"].items():
        for i, e in enumerate(errs):
            assert e < BACKWARD_TOL, f"padded {name} doc {i} via reused contiguous graph: {e:.5f}"
