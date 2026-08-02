"""BF16 THD with a sliding window and/or a learnable softmax sink is correct.

    python3 -m pytest test_packed_sliding_window_sinks.py -q -rs

Oracle: under block-diagonal masking documents are independent, so a packed row must equal the
concatenation of per-document dense runs, and `softmax_offset.grad` must equal the sum of the
per-document dense sink gradients. Both sides run BF16, so quantisation error cancels and what
remains is packing error.

The matrix is window against softmax type. Both axes are otherwise untested under packed input:
elsewhere the window is always unbounded and the softmax type always left at its default.

Metrics differ by tensor on purpose:
    O, dQ, dK, dV   relative RMS
    dSink           max element-wise relative error, floored at 1e-3 of the largest entry.
                    dSink is one scalar per head spanning several orders of magnitude; relative
                    RMS over it is dominated by the largest entries and does not fail when a
                    small entry is wrong.
"""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="device-only")


@pytest.fixture(autouse=True)
def _nondeterministic_algos():
    """Pin NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 for every test in this module.

    The learnable sink needs the nondeterministic backend: under deterministic mode TE falls
    back to a path that carries the sink gradient at BF16 precision, which moves dSink from
    ~3e-4 to ~1e-1 against the same reference (see test_deterministic_mode_degrades_dsink).

    This is autouse because the setting is process-global and another module can turn it off at
    import time: pytest imports every test module during collection, so a module-level default
    set elsewhere lands before a single test here runs. Without this fixture these tests pass
    alone and fail in the full suite, which reads as a kernel fault and is not one.
    """
    saved = os.environ.get("NVTE_ALLOW_NONDETERMINISTIC_ALGO")
    os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "1"
    _invalidate_backend_cache()
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop("NVTE_ALLOW_NONDETERMINISTIC_ALGO", None)
        else:
            os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = saved
        _invalidate_backend_cache()


def _invalidate_backend_cache():
    """TE caches backend selection; an env change is invisible until the cache is dirtied."""
    from transformer_engine.pytorch.attention.dot_product_attention import (
        dot_product_attention as _dpa,
    )

    _dpa._attention_backends["backend_selection_requires_update"] = True

DEVICE = "cuda"
WINDOW = 4096
DOC_LENGTHS = [5812, 2183, 6389, 2000]
SHAPES = {
    "gqa_16_4_d128": dict(heads_q=16, heads_kv=4, head_dim=128),
    "mha_8_8_d128": dict(heads_q=8, heads_kv=8, head_dim=128),
    # Production per-rank geometries: head counts divided by the tensor-parallel size.
    # Gqa_16_4_d128 is omnii 8b at tensor parallel 2 and is not repeated.
    "omnii_8b_tp1": dict(heads_q=32, heads_kv=8, head_dim=128),
    "omnii_15b_tp1": dict(heads_q=40, heads_kv=10, head_dim=128),
    "omnii_15b_tp2": dict(heads_q=20, heads_kv=5, head_dim=128),
}
PACKING_TOL = 2e-2
SINK_TOL = 2e-2
# `None` is a legal window value ("no window"), so it cannot double as "not specified".
UNSET = object()


def relative_rms(actual, reference):
    actual, reference = actual.detach().float(), reference.detach().float()
    denominator = reference.pow(2).mean().sqrt()
    if denominator == 0:
        return float(actual.pow(2).mean().sqrt())
    return float((actual - reference).pow(2).mean().sqrt() / denominator)


def max_elementwise_relative(actual, reference):
    actual, reference = actual.detach().float(), reference.detach().float()
    floor = 1e-3 * reference.abs().max()
    return float(((actual - reference).abs() / (reference.abs() + floor)).max())


def build_module(shape, *, packed, window, softmax_type):
    import transformer_engine.pytorch as te

    kwargs = dict(num_attention_heads=shape["heads_q"], kv_channels=shape["head_dim"],
                  num_gqa_groups=shape["heads_kv"], attention_dropout=0.0,
                  softmax_type=softmax_type)
    if window is not None:
        kwargs["window_size"] = (window, 0)
    if packed:
        return te.DotProductAttention(qkv_format="thd", attn_mask_type="padding_causal",
                                      **kwargs).to(DEVICE)
    return te.DotProductAttention(qkv_format="bshd", attn_mask_type="causal", **kwargs).to(DEVICE)


def run_pair(shape, window, softmax_type, seed=0, reference_window=UNSET,
             reference_sink_scale=1.0):
    """Return (packed, dense) results as dicts of tensors.

    `reference_window` and `reference_sink_scale` exist so tripwire tests can perturb the dense
    reference; both default to matching the packed side exactly.
    """
    torch.manual_seed(seed)
    total = sum(DOC_LENGTHS)
    cu_seqlens = torch.tensor([0] + list(torch.tensor(DOC_LENGTHS).cumsum(0)),
                              dtype=torch.int32, device=DEVICE)

    def make(heads):
        return torch.randn(total, heads, shape["head_dim"], device=DEVICE, dtype=torch.bfloat16)

    query, key, value = make(shape["heads_q"]), make(shape["heads_kv"]), make(shape["heads_kv"])
    grad_output = torch.randn_like(query)
    learnable = softmax_type == "learnable"
    sink = (torch.randn(shape["heads_q"], device=DEVICE, dtype=torch.float32) * 0.1
            if learnable else None)

    packed_module = build_module(shape, packed=True, window=window, softmax_type=softmax_type)
    if learnable:
        with torch.no_grad():
            packed_module.softmax_offset.copy_(sink.reshape(packed_module.softmax_offset.shape))
    inputs = [t.clone().requires_grad_(True) for t in (query, key, value)]
    packed_out = packed_module(*inputs, cu_seqlens_q=cu_seqlens, cu_seqlens_kv=cu_seqlens,
                               max_seqlen_q=max(DOC_LENGTHS), max_seqlen_kv=max(DOC_LENGTHS))
    packed_out.backward(grad_output.reshape(packed_out.shape))
    packed = {"O": packed_out.reshape(total, -1)}
    for name, tensor in zip(("dQ", "dK", "dV"), inputs):
        packed[name] = tensor.grad.reshape(total, -1)
    if learnable:
        packed["dSink"] = packed_module.softmax_offset.grad.reshape(-1)

    dense_window = window if reference_window is UNSET else reference_window
    dense_module = build_module(shape, packed=False, window=dense_window,
                                softmax_type=softmax_type)
    if learnable:
        with torch.no_grad():
            dense_module.softmax_offset.copy_(
                (sink * reference_sink_scale).reshape(dense_module.softmax_offset.shape))
    dense_module.zero_grad(set_to_none=True)

    collected = {"O": [], "dQ": [], "dK": [], "dV": []}
    for index, length in enumerate(DOC_LENGTHS):
        start = int(cu_seqlens[index])
        slices = [t[start:start + length].unsqueeze(0).clone().requires_grad_(True)
                  for t in (query, key, value)]
        out = dense_module(*slices)
        out.backward(grad_output[start:start + length].reshape(out.shape))
        collected["O"].append(out.reshape(length, -1))
        for name, tensor in zip(("dQ", "dK", "dV"), slices):
            collected[name].append(tensor.grad.reshape(length, -1))
    dense = {name: torch.cat(parts) for name, parts in collected.items()}
    if learnable:
        dense["dSink"] = dense_module.softmax_offset.grad.reshape(-1)
    return packed, dense


def compare(packed, dense):
    errors = {name: relative_rms(packed[name], dense[name])
              for name in ("O", "dQ", "dK", "dV")}
    if "dSink" in packed:
        errors["dSink"] = max_elementwise_relative(packed["dSink"], dense["dSink"])
    return errors


# --------------------------------------------------------------------------------------
# The matrix: window x softmax_type, both axes previously ungated
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("softmax_type", ["vanilla", "learnable"])
@pytest.mark.parametrize("window", [None, WINDOW])
@pytest.mark.parametrize("shape_name", sorted(SHAPES))
def test_thd_matches_dense(shape_name, window, softmax_type):
    """Packed THD must equal per-document dense across the window x sink matrix."""
    packed, dense = run_pair(SHAPES[shape_name], window, softmax_type)
    errors = compare(packed, dense)
    for name, error in errors.items():
        tolerance = SINK_TOL if name == "dSink" else PACKING_TOL
        assert error < tolerance, f"{shape_name} window={window} {softmax_type}: {name}={error:.3e}"


@pytest.mark.parametrize("window", [None, WINDOW])
def test_sink_gradient_is_produced(window):
    """A learnable sink must yield a finite, non-zero gradient on the packed path."""
    packed, _ = run_pair(SHAPES["gqa_16_4_d128"], window, "learnable")
    grad = packed["dSink"]
    assert torch.isfinite(grad).all(), "dSink has non-finite entries"
    assert grad.abs().max() > 0, "dSink is identically zero"


@pytest.mark.parametrize("softmax_type", ["vanilla", "learnable"])
def test_window_changes_the_result(softmax_type):
    """The window must actually bind: windowed and unwindowed outputs must differ.

    Without this a silently-ignored `window_size` would leave every other gate green.
    """
    windowed, _ = run_pair(SHAPES["gqa_16_4_d128"], WINDOW, softmax_type)
    unwindowed, _ = run_pair(SHAPES["gqa_16_4_d128"], None, softmax_type)
    assert relative_rms(windowed["O"], unwindowed["O"]) > PACKING_TOL


def test_sink_changes_the_result():
    """The sink must actually bind: learnable and vanilla outputs must differ."""
    learnable, _ = run_pair(SHAPES["gqa_16_4_d128"], WINDOW, "learnable")
    vanilla, _ = run_pair(SHAPES["gqa_16_4_d128"], WINDOW, "vanilla")
    assert relative_rms(learnable["O"], vanilla["O"]) > PACKING_TOL


# --------------------------------------------------------------------------------------
# tripwires: the gates above must be able to fail
# --------------------------------------------------------------------------------------


def test_tripwire_window_mismatch_fails():
    """Dropping the window from the dense reference must break O/dQ/dK/dV."""
    packed, dense = run_pair(SHAPES["gqa_16_4_d128"], WINDOW, "vanilla", reference_window=None)
    errors = compare(packed, dense)
    assert max(errors[name] for name in ("O", "dQ", "dK", "dV")) > PACKING_TOL


def test_tripwire_sink_mismatch_fails():
    """Perturbing the dense reference's sink must break dSink specifically.

    A window mismatch does not move dSink much, so this is the perturbation that proves the dSink
    gate can fail.
    """
    packed, dense = run_pair(SHAPES["gqa_16_4_d128"], WINDOW, "learnable",
                             reference_sink_scale=1.1)
    assert max_elementwise_relative(packed["dSink"], dense["dSink"]) > SINK_TOL


def test_deterministic_mode_degrades_dsink():
    """Deterministic mode is not a supported configuration for the learnable sink.

    Gates the constraint that was previously folklore ("sinks need nondeterministic mode"):
    with NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 the sink gradient drops to BF16 precision and the
    packed-vs-dense agreement blows past SINK_TOL. Asserting the degradation (rather than
    skipping) means the day TE fixes it, this test fails and tells us.
    """
    saved = os.environ.get("NVTE_ALLOW_NONDETERMINISTIC_ALGO")
    os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
    _invalidate_backend_cache()
    try:
        packed, dense = run_pair(SHAPES["gqa_16_4_d128"], WINDOW, "learnable")
        degraded = max_elementwise_relative(packed["dSink"], dense["dSink"])
    finally:
        if saved is None:
            os.environ.pop("NVTE_ALLOW_NONDETERMINISTIC_ALGO", None)
        else:
            os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = saved
        _invalidate_backend_cache()
    assert degraded > SINK_TOL, (
        f"deterministic mode no longer degrades dSink ({degraded:.3e} <= {SINK_TOL}); if TE "
        "fixed this, drop the nondeterministic pin in _nondeterministic_algos"
    )
