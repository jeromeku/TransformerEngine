"""FP8 THD attention with physical inter-sequence gaps (cu_seqlens_padded != cu_seqlens).

Ragged base offsets are built from cu_seqlens_*_padded, so each document is addressed at its padded
slot rather than at its actual cumulative offset. Three things are pinned:

  * BF16 over the padded layout reproduces a dense per-document reference exactly -- the addressing
    itself is correct, independent of quantization;
  * FP8 tracks that reference within quantization, forward and backward;
  * the negative control -- feeding the actual cu_seqlens where the padded offsets belong -- corrupts
    every document after the first, which is what makes the first two non-vacuous.

Per document, since a wrong base pointer corrupts only the tail documents and a whole-tensor norm
averages that away. The gap is real: data lives in the first actual[i] tokens of each padded slot,
the remainder is zeros.

    python3 -m pytest test_packed_padded_offsets.py -q -rs
"""

import os
import pathlib
import sys

import pytest
import torch

_current_file = pathlib.Path(__file__).resolve()
sys.path = [str(_current_file.parent), str(_current_file.parent.parent)] + sys.path

import transformer_engine.pytorch as te
from transformer_engine.common import recipe

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="device-only")

DEVICE = "cuda"
H, G, D = 8, 4, 128
ACTUAL = [400, 600, 300]
PADDED = [512, 640, 384]  # gaps of 112, 40, 84
SCALE = 0.5

BF16_EXACT_TOL = 1e-3  # BF16 padded vs dense reference; measured ~0
FORWARD_TOL = 0.12  # FP8 padded vs BF16 dense reference; measured ~0.043
BACKWARD_TOL = 0.20  # FP8 grads vs BF16 padded grads; measured ~0.10
CORRUPTION_FLOOR = 0.5  # a mis-addressed tail document; measured ~0.9-1.2


@pytest.fixture(autouse=True)
def _pin_fp8_backward():
    name = "NVTE_FP8_DPA_BWD"
    saved = os.environ.get(name)
    os.environ[name] = "1"
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = saved


def _cu(lengths):
    out = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=DEVICE)
    out[1:] = torch.cumsum(torch.tensor(lengths, dtype=torch.int32, device=DEVICE), 0)
    return out


def _module(qkv_format, mask):
    return te.DotProductAttention(
        num_attention_heads=H, kv_channels=D, num_gqa_groups=G, attention_dropout=0.0,
        qkv_format=qkv_format, attn_mask_type=mask, softmax_scale=1.0 / (D**0.5),
    ).cuda()


def _padded_batch(seed=0):
    """q/k/v in the padded physical layout, plus the per-document data for the dense reference."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    cu_padded = _cu(PADDED)
    q = torch.zeros(sum(PADDED), H, D, device=DEVICE, dtype=torch.bfloat16)
    k = torch.zeros(sum(PADDED), G, D, device=DEVICE, dtype=torch.bfloat16)
    v = torch.zeros(sum(PADDED), G, D, device=DEVICE, dtype=torch.bfloat16)
    per_doc = []
    for a, base in zip(ACTUAL, cu_padded[:-1].tolist()):
        qi = torch.randn(a, H, D, generator=gen, device=DEVICE, dtype=torch.bfloat16) * SCALE
        ki = torch.randn(a, G, D, generator=gen, device=DEVICE, dtype=torch.bfloat16) * SCALE
        vi = torch.randn(a, G, D, generator=gen, device=DEVICE, dtype=torch.bfloat16) * SCALE
        q[base : base + a], k[base : base + a], v[base : base + a] = qi, ki, vi
        per_doc.append((qi, ki, vi))
    return q, k, v, cu_padded, per_doc


def _reference(per_doc):
    """Per-document dense BF16, one sequence at a time -- no packing, no ragged offsets."""
    mod = _module("bshd", "causal")
    return [
        mod(qi.unsqueeze(0), ki.unsqueeze(0), vi.unsqueeze(0), attn_mask_type="causal").reshape(
            qi.shape[0], H, D
        )
        for qi, ki, vi in per_doc
    ]


def _forward_thd(q, k, v, cu_actual, cu_padded_offsets, fp8):
    mod = _module("thd", "padding_causal")
    kwargs = dict(
        cu_seqlens_q=cu_actual, cu_seqlens_kv=cu_actual,
        cu_seqlens_q_padded=cu_padded_offsets, cu_seqlens_kv_padded=cu_padded_offsets,
        max_seqlen_q=max(PADDED), max_seqlen_kv=max(PADDED), attn_mask_type="padding_causal",
    )
    if fp8:
        with te.fp8_autocast(enabled=True, fp8_recipe=recipe.DelayedScaling(fp8_dpa=True)):
            out = mod(q, k, v, **kwargs)
    else:
        out = mod(q, k, v, **kwargs)
    return out.reshape(sum(PADDED), H, D)


def _per_doc_rel_rms(out, cu_padded, reference):
    res = []
    for i, a in enumerate(ACTUAL):
        base = int(cu_padded[i])
        got, want = out[base : base + a].float(), reference[i].float()
        denom = max(want.pow(2).mean().sqrt().item(), 1e-12)
        res.append((got - want).pow(2).mean().sqrt().item() / denom)
    return res


def test_bf16_padded_matches_dense_reference():
    """The padded addressing is exact in BF16, so a real reference exists for the FP8 cases."""
    q, k, v, cu_padded, per_doc = _padded_batch()
    got = _forward_thd(q, k, v, _cu(ACTUAL), cu_padded, fp8=False)
    for i, rms in enumerate(_per_doc_rel_rms(got, cu_padded, _reference(per_doc))):
        assert rms < BF16_EXACT_TOL, f"BF16 doc {i}: {rms:.5f}"


def test_fp8_padded_forward_matches_reference():
    q, k, v, cu_padded, per_doc = _padded_batch()
    got = _forward_thd(q, k, v, _cu(ACTUAL), cu_padded, fp8=True)
    for i, rms in enumerate(_per_doc_rel_rms(got, cu_padded, _reference(per_doc))):
        assert rms < FORWARD_TOL, f"FP8 doc {i}: {rms:.5f}"


def test_fp8_padded_backward_matches_bf16():
    q, k, v, cu_padded, _ = _padded_batch()
    cu_actual = _cu(ACTUAL)
    torch.manual_seed(0)
    grad = torch.zeros_like(q)
    for a, base in zip(ACTUAL, cu_padded[:-1].tolist()):
        grad[base : base + a] = torch.randn(a, H, D, device=DEVICE, dtype=torch.bfloat16) * SCALE

    def grads(fp8):
        mod = _module("thd", "padding_causal")
        qg, kg, vg = (x.clone().detach().requires_grad_(True) for x in (q, k, v))
        kwargs = dict(
            cu_seqlens_q=cu_actual, cu_seqlens_kv=cu_actual,
            cu_seqlens_q_padded=cu_padded, cu_seqlens_kv_padded=cu_padded,
            max_seqlen_q=max(PADDED), max_seqlen_kv=max(PADDED), attn_mask_type="padding_causal",
        )
        if fp8:
            with te.fp8_autocast(enabled=True, fp8_recipe=recipe.DelayedScaling(fp8_dpa=True)):
                out = mod(qg, kg, vg, **kwargs)
        else:
            out = mod(qg, kg, vg, **kwargs)
        out.reshape(sum(PADDED), H, D).backward(grad)
        return qg.grad, kg.grad, vg.grad

    bf16, fp8 = grads(False), grads(True)
    for name, g8, g16 in zip(("dq", "dk", "dv"), fp8, bf16):
        for i, rms in enumerate(_per_doc_rel_rms(g8, cu_padded, [g16[int(cu_padded[j]):int(cu_padded[j]) + ACTUAL[j]] for j in range(len(ACTUAL))])):
            assert rms < BACKWARD_TOL, f"FP8 {name} doc {i}: {rms:.5f}"


def test_actual_for_padded_offsets_corrupts_tail_documents():
    """Negative control: feeding the actual cu_seqlens where the padded offsets belong addresses
    every document after the first into the previous slot. Makes the positive tests non-vacuous."""
    q, k, v, cu_padded, per_doc = _padded_batch()
    reference = _reference(per_doc)
    # cu_seqlens_padded == cu_seqlens (actual): the pre-fix behaviour, offsets from actual lengths.
    got = _forward_thd(q, k, v, _cu(ACTUAL), _cu(ACTUAL), fp8=True)
    rms = _per_doc_rel_rms(got, cu_padded, reference)
    assert rms[0] < FORWARD_TOL, f"doc 0 (offset 0) should be fine: {rms[0]:.5f}"
    for i in range(1, len(ACTUAL)):
        assert rms[i] > CORRUPTION_FLOOR, f"tail doc {i} should be corrupt without padded offsets: {rms[i]:.5f}"
