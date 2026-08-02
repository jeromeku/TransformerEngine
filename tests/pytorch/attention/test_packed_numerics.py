"""Packed (THD) attention computes the same result as the equivalent dense batch.

Two oracles, and the difference between them is the point.

The sharp one compares packed against dense-padded at the **same precision**. Quantization error
then cancels between the two sides, so the tolerance can be 1e-6 rather than the 0.2 a comparison
against a higher-precision reference forces. That is five orders of magnitude more sensitive to the
thing under test, which is the ragged addressing.

The coarse one compares FP8 against BF16 on the same packed input. It is necessarily loose because
it absorbs quantization error, but it is the only one that can see a fault shared by the packed and
dense paths, which the sharp oracle cannot because it compares them against each other.

Everything is compared per sequence rather than per tensor. A wrong offset usually corrupts only
the tail sequences, and a whole-tensor norm averages that away.

    python3 -m pytest test_packed_numerics.py -q -rs
"""

import math
import pathlib
import sys

import pytest
import torch

_current_file = pathlib.Path(__file__).resolve()
sys.path = [str(_current_file.parent), str(_current_file.parent.parent)] + sys.path

import transformer_engine.pytorch as te
from transformer_engine.common import recipe

from packed_input_utils import PACKED_CONFIGS, as_layout, constructible, make_packed_batch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="device-only")

# Packed against dense at the same precision: quantization cancels, so this bounds the addressing
# alone. Against a higher-precision reference the bound has to absorb quantization as well.
PACKING_TOLERANCE = 1e-6
PRECISION_TOLERANCE = 0.20
FORWARD_PRECISION_TOLERANCE = 0.12

# Delayed scaling has no amax history on a freshly constructed module, so the first step runs with
# default scales and measures the warmup rather than the kernel. Three steps is past the transient.
WARMUP_STEPS = 3
GRADIENT_SEED = 1234

RECIPES = {"delayed": recipe.DelayedScaling, "current": recipe.Float8CurrentScaling}


def _attention(config, qkv_format, mask):
    return te.DotProductAttention(
        num_attention_heads=config.num_heads, kv_channels=config.head_dim_qk,
        num_gqa_groups=config.num_gqa_groups, attention_dropout=0.0,
        qkv_format=qkv_format, attn_mask_type=mask,
        softmax_scale=1.0 / math.sqrt(config.head_dim_qk),
    ).cuda()


def upstream_gradient(batch, qkv_format):
    """The same logical upstream gradient, expressed in the requested format.

    `randn_like` on the output will not do: the packed output is [t, h, d] and the dense one is
    [b, s, h, d], so an identical seed yields different tensors and the comparison degrades to
    about sqrt(2), which looks exactly like a packing fault.
    """
    torch.manual_seed(GRADIENT_SEED)
    packed = torch.randn(batch.total_tokens, batch.config.num_heads, batch.config.head_dim_v,
                         device="cuda", dtype=torch.bfloat16)
    if qkv_format == "thd":
        return packed
    dense = torch.zeros(batch.batch_size, batch.max_seqlen, batch.config.num_heads,
                        batch.config.head_dim_v, device="cuda", dtype=torch.bfloat16)
    for i in range(batch.batch_size):
        dense[i, : batch.seqlens[i]] = batch.sequence(packed, i)
    return dense


def run_forward(batch, qkv_format="thd", layout="thd_thd_thd", mask="padding_causal",
                fp8=True, rec="delayed"):
    """One forward pass; returns the output reshaped to the packed or dense token layout."""
    if qkv_format == "thd":
        q, k, v = as_layout(batch, layout)
    else:
        q, k, v = batch.padded()
    module = _attention(batch.config, qkv_format, mask)
    kwargs = dict(cu_seqlens_q=batch.cu_seqlens, cu_seqlens_kv=batch.cu_seqlens,
                  max_seqlen_q=batch.max_seqlen, max_seqlen_kv=batch.max_seqlen,
                  attn_mask_type=mask)
    if not fp8:
        out = module(q, k, v, **kwargs)
    else:
        with te.fp8_autocast(enabled=True, fp8_recipe=RECIPES[rec](fp8_dpa=True)):
            out = module(q, k, v, **kwargs)
    if qkv_format == "thd":
        return out.reshape(batch.total_tokens, batch.config.num_heads, batch.config.head_dim_v)
    return out.reshape(batch.batch_size, batch.max_seqlen, batch.config.num_heads,
                       batch.config.head_dim_v)


def run_backward(batch, qkv_format="thd", layout="thd_thd_thd", mask="padding_causal",
                 fp8=True, rec="delayed", perturb=None, steps=WARMUP_STEPS, fp8_output=False):
    """`steps` forward and backward iterations; returns the final (dQ, dK, dV)."""
    if qkv_format == "thd":
        base = list(as_layout(batch, layout))
    else:
        base = list(batch.padded())

    if perturb is not None:
        lo, hi = batch.bounds(perturb)
        base[0] = base[0].clone()
        if qkv_format == "thd":
            base[0][lo:hi] = torch.randn_like(base[0][lo:hi]) * 0.5
        else:
            base[0][perturb, : batch.seqlens[perturb]] = torch.randn_like(
                base[0][perturb, : batch.seqlens[perturb]]) * 0.5

    module = _attention(batch.config, qkv_format, mask)
    kwargs = dict(cu_seqlens_q=batch.cu_seqlens, cu_seqlens_kv=batch.cu_seqlens,
                  max_seqlen_q=batch.max_seqlen, max_seqlen_kv=batch.max_seqlen,
                  attn_mask_type=mask)
    if fp8_output:
        kwargs["fp8_output"] = True
    grad_out = upstream_gradient(batch, qkv_format)

    for _ in range(steps):
        q, k, v = (x.clone().detach().requires_grad_(True) for x in base)
        if fp8:
            with te.fp8_autocast(enabled=True, fp8_recipe=RECIPES[rec](fp8_dpa=True)):
                out = module(q, k, v, **kwargs)
        else:
            out = module(q, k, v, **kwargs)
        out.backward(grad_out.reshape(tuple(out.shape)))

    grads = (q.grad, k.grad, v.grad)
    if qkv_format == "thd":
        return tuple(g.reshape(batch.total_tokens, -1, g.shape[-1]) for g in grads)
    return tuple(g.reshape(batch.batch_size, batch.max_seqlen, -1, g.shape[-1]) for g in grads)


def relative_rms(got, want):
    got, want = got.float(), want.float()
    return (got - want).pow(2).mean().sqrt().item() / max(
        want.pow(2).mean().sqrt().item(), 1e-12)


def compare_packed_vs_padded(batch, packed, padded, tol, label=""):
    """Every sequence's tensors agree between the packed and dense-padded runs."""
    for name, a, b in zip(("dQ", "dK", "dV"), packed, padded):
        for i in range(batch.batch_size):
            err = relative_rms(batch.sequence(a, i), b[i, : batch.seqlens[i]])
            assert err < tol, (
                f"{label}{name} sequence {i} (len {batch.seqlens[i]}): {err:.4g} exceeds {tol}"
            )


def compare_per_sequence(batch, got, want, tol, label=""):
    """Every sequence agrees, against a budget with both a relative and an absolute term.

    The absolute term is set by the tensor-wide magnitude and is not slack for its own sake. Under
    delayed scaling one scale covers the whole tensor, so the quantization step, and therefore the
    absolute error floor, is uniform across it and independent of any one sequence's magnitude. A
    sequence whose true gradient is near zero -- a length-one causal sequence has mathematically
    zero dQ and dK -- then shows an enormous relative error while its absolute error is ordinary.
    A purely relative bound would reject a result the dense path produces identically.
    """
    for name, a, b in zip(("dQ", "dK", "dV"), got, want):
        floor = b.float().pow(2).mean().sqrt().item()
        for i in range(batch.batch_size):
            got_seq = batch.sequence(a, i).float()
            want_seq = batch.sequence(b, i).float()
            err = (got_seq - want_seq).pow(2).mean().sqrt().item()
            budget = tol * (want_seq.pow(2).mean().sqrt().item() + floor)
            assert err <= budget, (
                f"{label}{name} sequence {i} (len {batch.seqlens[i]}): {err:.4g} exceeds budget "
                f"{budget:.4g}"
            )


# --------------------------------------------------------------------------------------
# forward
# --------------------------------------------------------------------------------------

FORWARD_CONFIGS = ["omnii_8b_tp1", "mha_d64", "mha_d128"]


@pytest.mark.parametrize("distribution", ["long_tailed_4k", "uniform_8k", "tile_aligned",
                                          "single_8k"])
@pytest.mark.parametrize("config_name", FORWARD_CONFIGS)
def test_forward_packing_equivalence(config_name, distribution):
    """Per-sequence FP8 against BF16 on the same packed input, across shapes and length mixes."""
    batch = make_packed_batch(PACKED_CONFIGS[config_name], distribution)
    fp8 = run_forward(batch)
    bf16 = run_forward(batch, fp8=False)
    for i in range(batch.batch_size):
        err = relative_rms(batch.sequence(fp8, i), batch.sequence(bf16, i))
        assert err < FORWARD_PRECISION_TOLERANCE, (
            f"sequence {i} (len {batch.seqlens[i]}): {err:.4f}"
        )


@pytest.mark.parametrize("config_name", ["omnii_8b_tp1", "mha_d64"])
def test_forward_packed_matches_padded(config_name):
    """FP8 packed against FP8 dense-padded on identical data.

    Catches a packing fault that affects FP8 and BF16 alike, which the comparison above cannot see.
    """
    batch = make_packed_batch(PACKED_CONFIGS[config_name], "tile_aligned")
    packed = run_forward(batch, mask="padding")
    padded = run_forward(batch, qkv_format="bshd", mask="padding")
    for i in range(batch.batch_size):
        err = relative_rms(batch.sequence(packed, i), padded[i, : batch.seqlens[i]])
        assert err < 0.05, f"sequence {i}: packed against dense {err:.4f}"


@pytest.mark.parametrize("config_name", ["omnii_8b_tp1", "gqa_8to1_d128"])
def test_forward_sequences_are_independent(config_name):
    """Perturbing one sequence's queries moves no other sequence's output.

    Catches offset-multiplier confusion under grouped-query attention, where the key and value
    offsets use the query head count. Every sequence stays individually plausible under that fault,
    so only cross-sequence independence exposes it.
    """
    batch = make_packed_batch(PACKED_CONFIGS[config_name], "long_tailed_4k")
    base = run_forward(batch)
    target = 1
    lo, hi = batch.bounds(target)
    batch.q[lo:hi] = torch.randn_like(batch.q[lo:hi]) * 0.5
    moved = run_forward(batch)

    for i in range(batch.batch_size):
        if i == target:
            continue
        err = relative_rms(batch.sequence(moved, i), batch.sequence(base, i))
        assert err < 1e-3, f"sequence {i} moved ({err:.2e}) when only {target} was perturbed"
    assert relative_rms(batch.sequence(moved, target),
                        batch.sequence(base, target)) > 1e-2, "the perturbed sequence must change"


@pytest.mark.parametrize("mask", ["padding", "padding_causal"])
def test_forward_both_padding_masks(mask):
    """Both admitted masks execute and agree with BF16."""
    batch = make_packed_batch(PACKED_CONFIGS["mha_d128"], "tile_aligned")
    fp8 = run_forward(batch, mask=mask)
    bf16 = run_forward(batch, mask=mask, fp8=False)
    for i in range(batch.batch_size):
        err = relative_rms(batch.sequence(fp8, i), batch.sequence(bf16, i))
        assert err < FORWARD_PRECISION_TOLERANCE, f"{mask} sequence {i}: {err:.4f}"


def test_forward_degenerate_sequences():
    """Length-one and sub-tile sequences packed beside long ones.

    Catches tail handling and tile-alignment assumptions.
    """
    batch = make_packed_batch(PACKED_CONFIGS["mha_d64"], [1, 37, 512, 3, 128])
    fp8 = run_forward(batch)
    bf16 = run_forward(batch, fp8=False)
    for i in range(batch.batch_size):
        err = relative_rms(batch.sequence(fp8, i), batch.sequence(bf16, i))
        assert err < 0.15, f"sequence {i} (len {batch.seqlens[i]}): {err:.4f}"


def test_forward_output_is_finite():
    """No non-finite values anywhere, including the tails next to a sequence boundary.

    Catches uninitialised reads past a sequence end, which surface as non-finite values rather than
    as a magnitude error and can hide inside a root-mean-square comparison.
    """
    batch = make_packed_batch(PACKED_CONFIGS["omnii_8b_tp1"], "long_tailed_4k")
    assert torch.isfinite(run_forward(batch).float()).all()


# --------------------------------------------------------------------------------------
# backward, sharp oracle
# --------------------------------------------------------------------------------------

BACKWARD_CONFIGS = ["omnii_8b_tp1", "mha_d64", "mha_d128", "omnii_8b_tp2", "omnii_15b_tp1",
                    "omnii_15b_tp2"]


@pytest.mark.parametrize("mask", ["padding", "padding_causal"])
@pytest.mark.parametrize("config_name", BACKWARD_CONFIGS)
def test_backward_packed_matches_padded(config_name, mask):
    """Packed gradients equal dense-padded gradients on identical data.

    Catches any error in the ragged write offsets for the gradients, which the backward derives
    from its own layout in a second launch the forward never performs. Both sides are FP8, so
    quantization cancels and the bound can be 1e-6.
    """
    batch = make_packed_batch(PACKED_CONFIGS[config_name], "tile_aligned")
    packed = run_backward(batch, qkv_format="thd", mask=mask)
    padded = run_backward(batch, qkv_format="bshd", mask=mask)
    compare_packed_vs_padded(batch, packed, padded, PACKING_TOLERANCE, label=f"{mask} ")


@pytest.mark.parametrize("config_name", ["omnii_8b_tp1", "gqa_8to1_d128"])
def test_backward_sequences_are_independent(config_name):
    """Perturbing one sequence's queries moves no other sequence's gradients."""
    batch = make_packed_batch(PACKED_CONFIGS[config_name], "long_tailed_4k")
    base = run_backward(batch)
    moved = run_backward(batch, perturb=1)
    for name, a, b in zip(("dQ", "dK", "dV"), moved, base):
        for i in range(batch.batch_size):
            if i == 1:
                continue
            err = relative_rms(batch.sequence(a, i), batch.sequence(b, i))
            assert err < 1e-3, f"{name} sequence {i} moved ({err:.2e}) when only 1 was perturbed"
    assert relative_rms(batch.sequence(moved[0], 1),
                        batch.sequence(base[0], 1)) > 1e-2, "the perturbed sequence must change"


@pytest.mark.parametrize("layout", [l for l, c in constructible() if c == "mha_d64"])
def test_backward_every_layout(layout):
    """Every constructible packed layout produces correct gradients.

    Catches a wrong multiplier in the gradient offsets for an interleaved layout, which carry three
    or two times the per-token product rather than one.
    """
    batch = make_packed_batch(PACKED_CONFIGS["mha_d64"], "tile_aligned")
    compare_per_sequence(batch, run_backward(batch, layout=layout),
                         run_backward(batch, layout=layout, fp8=False),
                         PRECISION_TOLERANCE, label=f"{layout} ")


# --------------------------------------------------------------------------------------
# backward, coarse oracle
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("distribution", ["long_tailed_4k", "uniform_8k", "tile_aligned",
                                          "single_8k"])
@pytest.mark.parametrize("config_name", FORWARD_CONFIGS)
def test_backward_packing_equivalence(config_name, distribution):
    """Per-sequence gradients, FP8 against BF16, across shapes and length mixes.

    Catches a fault shared by the packed and dense FP8 paths, which the sharp comparison cannot see
    because it checks them against each other.
    """
    batch = make_packed_batch(PACKED_CONFIGS[config_name], distribution)
    compare_per_sequence(batch, run_backward(batch), run_backward(batch, fp8=False),
                         PRECISION_TOLERANCE)


@pytest.mark.parametrize("mask", ["padding", "padding_causal"])
def test_backward_both_padding_masks(mask):
    """Both admitted masks produce correct gradients.

    The backward re-applies the mask when recomputing scores, so a diagonal misalignment can be
    right forward and wrong backward.
    """
    batch = make_packed_batch(PACKED_CONFIGS["mha_d128"], "tile_aligned")
    compare_per_sequence(batch, run_backward(batch, mask=mask),
                         run_backward(batch, mask=mask, fp8=False),
                         PRECISION_TOLERANCE, label=f"{mask} ")


def test_backward_current_scaling():
    """The other admitted recipe, which recomputes the scale per step from the tensors in hand
    rather than from history, and is the one that does not need the warmup above."""
    batch = make_packed_batch(PACKED_CONFIGS["mha_d128"], "tile_aligned")
    compare_per_sequence(batch, run_backward(batch, rec="current"),
                         run_backward(batch, fp8=False), PRECISION_TOLERANCE, label="current ")


def test_backward_degenerate_sequences():
    """Length-one, sub-tile and exact-tile sequences packed beside long ones.

    Catches tail handling in the gradient writes, where a partial tile at the end of a sequence is
    most likely to spill into its neighbour.
    """
    batch = make_packed_batch(PACKED_CONFIGS["mha_d64"], [1, 37, 512, 3, 128, 64])
    compare_per_sequence(batch, run_backward(batch), run_backward(batch, fp8=False),
                         PRECISION_TOLERANCE)


def test_backward_length_one_sequence_matches_padded():
    """A length-one causal sequence gives the same gradients packed and dense.

    Softmax over a single score is identically one, so mathematically no gradient reaches the
    queries or keys while the values receive the full upstream gradient. In FP8 that is exactly
    true only with cold scales; after the amax warmup a mathematically-zero gradient is quantized
    against a tensor-wide scale and comes out small but non-zero. The dense path produces the same
    value, so asserting equality with dense sidesteps the question of what the right magnitude is,
    which no comparison against a higher-precision reference can answer.
    """
    batch = make_packed_batch(PACKED_CONFIGS["mha_d64"], [1, 128, 64])
    packed = run_backward(batch, qkv_format="thd")
    padded = run_backward(batch, qkv_format="bshd")
    compare_packed_vs_padded(batch, packed, padded, PACKING_TOLERANCE, label="length one ")
    # An all-zero dV would satisfy the equality above while meaning the sequence was skipped.
    assert batch.sequence(packed[2], 0).abs().max().item() > 0.0, (
        "dV of the length-one sequence is all zero, so it was skipped rather than reduced"
    )


def test_backward_gradients_are_finite():
    """No non-finite values in any gradient, including the tails next to a boundary."""
    batch = make_packed_batch(PACKED_CONFIGS["omnii_8b_tp1"], "long_tailed_4k")
    for name, grad in zip(("dQ", "dK", "dV"), run_backward(batch)):
        assert torch.isfinite(grad.float()).all(), f"non-finite values in {name}"


def _output_dtype_reaching_the_backward(rec):
    """Describe the output tensor handed to the fused backward for the given recipe."""
    import transformer_engine_torch as tex

    seen = {}
    original = tex.fused_attn_bwd

    def wrapper(*args, **kwargs):
        tensors = [x for x in args if isinstance(x, torch.Tensor) or hasattr(x, "_data")]
        # Positional order: cu_seqlens_q, cu_seqlens_kv, Q, K, V, O, dO, rng_state
        seen.setdefault("output", tensors[5])
        return original(*args, **kwargs)

    tex.fused_attn_bwd = wrapper
    try:
        run_backward(make_packed_batch(PACKED_CONFIGS["mha_d64"], "tile_aligned"), rec=rec)
    finally:
        tex.fused_attn_bwd = original
    out = seen.get("output")
    return "fp8" if getattr(out, "_data", None) is not None else str(out.dtype)


@pytest.mark.parametrize("rec,expected", [("delayed", "fp8"), ("current", "torch.bfloat16")])
def test_each_recipe_drives_a_different_backward_branch(rec, expected):
    """Which output precision each recipe hands the backward.

    The FP8 backward builds a different graph depending on whether the output arrives quantized,
    and which side is taken falls out of the recipe rather than out of any test. Delayed scaling
    hands it a quantized output and current scaling a high-precision one, so the two recipes
    between them cover both branches.

    Catches that plumbing collapsing both recipes onto one branch, which would leave the other
    entirely unexercised while every numerical test above still passed. The coverage reads like a
    coincidence, so it is asserted rather than left implicit.
    """
    assert _output_dtype_reaching_the_backward(rec) == expected


def test_backward_with_a_quantized_output():
    """A step requesting a quantized output produces correct gradients.

    The returned tensor is quantized, so the upstream gradient has to be built against the
    dequantized shape, and the path from a quantized output back to the inputs is otherwise
    unexercised under packed input. This does not change what the backward sees for the output
    itself, which the recipe decides independently.
    """
    batch = make_packed_batch(PACKED_CONFIGS["mha_d128"], "tile_aligned")
    plain = run_backward(batch, fp8_output=False)
    quantized = run_backward(batch, fp8_output=True)
    compare_per_sequence(batch, quantized, plain, PRECISION_TOLERANCE, label="quantized output ")
