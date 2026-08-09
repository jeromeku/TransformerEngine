"""Ragged offsets for packed (THD) input are sized correctly and their metadata is checked.

The offsets handed to cuDNN are a per-layout multiplier times the cumulative sequence length, so
the largest one scales with the total packed token count rather than with the longest sequence.
Sizing them from max_seqlen underestimates by roughly the batch size. Where the kernels use 32-bit
offsets that produces a wrapped base pointer: a read from the middle of another sequence, giving
plausible numbers and no error.

Three things are covered here, in increasing cost. The sizing itself is arithmetic and is tested
directly through `get_ragged_offset_dtype_bits`, which is also the only practical way -- a batch at
the 32-bit boundary needs about 2 GiB of queries alone, and the interesting cases are far larger.
Execution past the boundary is then checked on a batch that genuinely crosses it. Finally the
contents of cu_seqlens are checked, since it reaches the kernels as a trusted device pointer from
which every offset is derived.

    python3 -m pytest test_packed_ragged_offsets.py -q -rs
"""

import math
import os
import sys

import pytest
import torch

# transformer_engine must be imported before transformer_engine_torch: the package __init__ loads
# libtransformer_engine.so with RTLD_GLOBAL, and without it the extension cannot resolve symbols it
# expects to already be in the process.
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.common import recipe

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="device-only")

@pytest.fixture(autouse=True)
def _pin_fp8_backward():
    """Pin the FP8 backward on for every test here, and restore what was there before.

    The tolerances below were measured with it enabled. Another module in the same process can
    assign this variable to drive its own parametrization, and a comparison calibrated for one
    mode silently measures the other. Defence in depth: the module that assigns it also restores
    it, and this does not rely on that.
    """
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


INT32_MAX = 2**31 - 1

THD_THD_THD = tex.NVTE_QKV_Layout.NVTE_THD_THD_THD
T3HD = tex.NVTE_QKV_Layout.NVTE_T3HD
THD_T2HD = tex.NVTE_QKV_Layout.NVTE_THD_T2HD


def bits(layout=THD_THD_THD, h=1, hg=1, t_q=1, t_kv=1, d_qk=1, d_v=1):
    """Offset width in bits that the given extents require."""
    return tex.get_ragged_offset_dtype_bits(layout, h, hg, t_q, t_kv, d_qk, d_v)


# --------------------------------------------------------------------------------------
# the 32-bit boundary
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "total,expected",
    [
        (INT32_MAX - 1, 32),
        (INT32_MAX, 32),      # exactly representable, so still 32-bit
        (INT32_MAX + 1, 64),  # one past, must widen
    ],
)
def test_offset_width_at_the_boundary(total, expected):
    """32-bit up to and including INT32_MAX, 64-bit one past it.

    Catches an off-by-one in either direction. A `>=` where `>` belongs forces 64-bit offsets on
    configurations that are fine, which older cuDNN then rejects; a `>` where `>=` belongs admits
    one that overflows.
    """
    # One head and unit head dimension, so the token count is the offset. This isolates the
    # comparison from the per-layout multipliers tested below.
    assert bits(t_q=total, t_kv=1, h=1, d_qk=1, d_v=1) == expected


def test_o_offset_width_uses_head_dim_v_not_qk():
    """The O/dO offset width must use head_dim_v (F8).

    O inherits V's head dim, and the conversion kernel addresses the ragged O/dO base as
    num_attn_heads * head_dim_v * cu_seqlens_q_padded[i]. Sizing that entry with head_dim_qk (the
    prior bug) under-counts when d_v > d_qk, so the int32/int64 decision can admit a wrapped O
    offset. Reviewer's case: h=1, t_q=2**24, d_qk=1, d_v=128 -> the true O offset is
    1*128*2**24 = 2**31 = INT32_MAX + 1, which must widen to 64 bits. With the d_qk bug every other
    entry stays at 2**24 (32-bit) and the fault is silent.
    """
    o_offset = 1 * 128 * (2**24)
    assert o_offset == INT32_MAX + 1, "fixture: the O offset must cross the boundary by exactly one"
    assert bits(layout=THD_THD_THD, h=1, hg=1, t_q=2**24, t_kv=1, d_qk=1, d_v=128) == 64, (
        "O offset sized with head_dim_qk instead of head_dim_v -- admits a wrapped O/dO base pointer"
    )
    # Symmetric control: with d_qk == d_v == 128 the same extents give O = 128 * 2**23 = 2**30,
    # comfortably 32-bit, so my head_dim_v change does not spuriously widen the common d_qk==d_v case.
    assert bits(layout=THD_THD_THD, h=1, hg=1, t_q=2**23, t_kv=1, d_qk=128, d_v=128) == 32


def test_offset_width_uses_the_packed_token_count():
    """The width follows the physical extent, not the longest sequence.

    Catches sizing from max_seqlen. For batch 8, sequence length 32768, 64 heads and head dimension
    128, the longest sequence gives 268,435,456, comfortably 32-bit, while the packed batch gives
    2,147,483,648, which is INT32_MAX + 1 -- over by exactly one.
    """
    h, d, s, b = 64, 128, 32768, 8
    assert bits(h=h, hg=h, t_q=b * s, t_kv=b * s, d_qk=d, d_v=d) == 64, (
        "sized from the longest sequence instead of the packed token count"
    )
    assert bits(h=h, hg=h, t_q=s, t_kv=s, d_qk=d, d_v=d) == 32, (
        "the understated value must itself be 32-bit, which is why the fault is silent"
    )


def test_the_understatement_scales_with_batch_size():
    """Sizing from the longest sequence understates the offset by exactly the batch size.

    Pins the reason the arithmetic is wrong rather than one case that happens to trip it, so a
    partial fix using some other proxy for the extent is still caught.
    """
    # A single sequence sits just under INT32_MAX while any batch of two or more crosses it. With a
    # smaller sequence length the crossing never happens and every assertion below would be
    # vacuous, so the fixture asserts its own preconditions.
    h, d, s = 32, 128, 2**18
    one = h * d * s
    assert one <= INT32_MAX, "fixture broken: a single sequence must fit in 32 bits"

    checked = 0
    for b in (2, 8, 64):
        packed = h * d * b * s
        assert packed == one * b, "equal-length sequences give t = b * s"
        assert packed > INT32_MAX, "fixture broken: the packed batch must cross the boundary"
        assert bits(h=h, hg=h, t_q=s, t_kv=s, d_qk=d, d_v=d) == 32
        assert bits(h=h, hg=h, t_q=b * s, t_kv=b * s, d_qk=d, d_v=d) == 64
        checked += 1
    assert checked == 3, "every batch size must have been exercised"


# --------------------------------------------------------------------------------------
# per-layout multipliers
# --------------------------------------------------------------------------------------


def test_packed_layouts_carry_larger_multipliers():
    """Interleaved layouts cross the boundary sooner than separate buffers.

    The 3-way packed layouts carry three times the head-dimension product per token and the
    key-value packed layouts twice, so a token count that is safe for separate buffers is not safe
    for them. Catches the layout switch losing its packing factor, which would understate offsets
    for exactly the layouts that pack the most data.
    """
    h, d = 16, 128
    t = INT32_MAX // (h * d)          # just inside the limit for separate buffers
    assert bits(THD_THD_THD, h=h, hg=h, t_q=t, t_kv=t, d_qk=d, d_v=d) == 32
    assert bits(T3HD, h=h, hg=h, t_q=t, t_kv=t, d_qk=d, d_v=d) == 64
    assert bits(THD_T2HD, h=h, hg=h, t_q=t, t_kv=t, d_qk=d, d_v=d) == 64


def test_key_value_offsets_use_the_group_count():
    """Key and value offsets scale with the grouped-query group count, not the query head count.

    Catches using the query head count for the key and value entries, which overstates them under
    grouped-query attention and forces 64-bit offsets on configurations that are fine. That is the
    opposite failure to an overflow and no overflow test would reveal it.
    """
    d = 128
    t = INT32_MAX // (8 * d)          # sized so eight query heads sit right at the limit
    assert bits(h=8, hg=1, t_q=t, t_kv=t, d_qk=d, d_v=d) == 32
    assert bits(h=9, hg=1, t_q=t, t_kv=t, d_qk=d, d_v=d) == 64, "query heads must decide"
    assert bits(h=8, hg=2, t_q=t, t_kv=t, d_qk=d, d_v=d) == 32, "groups start far below the limit"


# --------------------------------------------------------------------------------------
# overflow safety of the check itself
# --------------------------------------------------------------------------------------


def test_absurd_extents_saturate_rather_than_wrap():
    """A pathological configuration widens to 64 bits rather than wrapping to a small value.

    Catches unchecked multiplication inside the helper. A product that wraps can land on a small
    positive number and answer that 32 bits suffice, which is the check failing open for precisely
    the most extreme input.
    """
    huge = 2**62
    assert bits(h=huge, hg=huge, t_q=huge, t_kv=huge, d_qk=huge, d_v=huge) == 64
    assert bits(h=2**40, hg=2**40, t_q=2**40, t_kv=2**40, d_qk=2**40, d_v=2**40) == 64


def test_zero_and_unit_extents_stay_32_bit():
    """Degenerate extents stay narrow rather than saturating.

    Catches an overflow check that treats zero as a sentinel, which would widen an empty batch and
    have older cuDNN reject it.
    """
    assert bits(h=0, hg=0, t_q=0, t_kv=0, d_qk=0, d_v=0) == 32
    assert bits(h=1, hg=1, t_q=1, t_kv=1, d_qk=1, d_v=1) == 32


# --------------------------------------------------------------------------------------
# execution past the boundary
# --------------------------------------------------------------------------------------

# Grouped-query keeps keys and values small so the queries dominate, and the token count is chosen
# so the query offsets clear INT32_MAX with margin.
BIG_BATCH, BIG_SEQLEN = 600, 1000
BIG_HEADS, BIG_GROUPS, BIG_DIM = 32, 4, 128
BIG_TOKENS = BIG_BATCH * BIG_SEQLEN
REQUIRED_BYTES = int(21.5 * 2**30)


def _enough_memory():
    if not torch.cuda.is_available():
        return False
    free, _ = torch.cuda.mem_get_info()
    return free >= REQUIRED_BYTES


requires_memory = pytest.mark.skipif(
    not _enough_memory(), reason=f"needs about {REQUIRED_BYTES / 2**30:.0f} GiB of free memory")


def _big_batch(dtype=torch.bfloat16):
    torch.manual_seed(0)
    q = torch.randn(BIG_TOKENS, BIG_HEADS, BIG_DIM, device="cuda", dtype=dtype) * 0.5
    k = torch.randn(BIG_TOKENS, BIG_GROUPS, BIG_DIM, device="cuda", dtype=dtype) * 0.5
    v = torch.randn(BIG_TOKENS, BIG_GROUPS, BIG_DIM, device="cuda", dtype=dtype) * 0.5
    cu_seqlens = torch.arange(0, BIG_TOKENS + 1, BIG_SEQLEN, dtype=torch.int32, device="cuda")
    return q, k, v, cu_seqlens


def _attention(heads, groups, dim):
    return te.DotProductAttention(
        num_attention_heads=heads, kv_channels=dim, num_gqa_groups=groups,
        attention_dropout=0.0, qkv_format="thd", attn_mask_type="padding_causal",
        softmax_scale=1.0 / math.sqrt(dim),
    ).cuda()


def _run(q, k, v, cu_seqlens, max_seqlen, fp8=False):
    module = _attention(q.shape[1], k.shape[1], q.shape[2])
    kwargs = dict(cu_seqlens_q=cu_seqlens, cu_seqlens_kv=cu_seqlens,
                  max_seqlen_q=max_seqlen, max_seqlen_kv=max_seqlen,
                  attn_mask_type="padding_causal")
    if not fp8:
        return module(q, k, v, **kwargs)
    with te.fp8_autocast(enabled=True, fp8_recipe=recipe.DelayedScaling(fp8_dpa=True)):
        return module(q, k, v, **kwargs)


def _run_one_sequence(q, k, v, index, fp8=False):
    """Sequence `index` on its own, as a batch of one, independent of every ragged offset."""
    lo, hi = index * BIG_SEQLEN, (index + 1) * BIG_SEQLEN
    # clone, not contiguous: a slice of a contiguous tensor is already contiguous, so contiguous()
    # returns the same view with a non-zero storage offset. The QKV layout is inferred from the
    # pointer relationships between q, k and v, and three views at arbitrary offsets match no
    # supported layout. Cloning gives three fresh allocations, which is what the inference expects.
    query, key, value = (x[lo:hi].clone() for x in (q, k, v))
    cu_seqlens = torch.tensor([0, BIG_SEQLEN], dtype=torch.int32, device="cuda")
    return _run(query, key, value, cu_seqlens, BIG_SEQLEN, fp8=fp8)


def _relative_rms(got, want):
    got, want = got.float(), want.float()
    return ((got - want).pow(2).mean().sqrt() / want.pow(2).mean().sqrt()).item()


def test_the_execution_fixture_really_crosses_the_boundary():
    """The batch below must need 64-bit offsets, and the longest sequence must not reveal that.

    Without this the execution tests could drift to a configuration that fits in 32 bits and pass
    while testing nothing. It also pins the gap itself: the width implied by the longest sequence
    disagrees with the width the packed batch actually needs.
    """
    truth = bits(THD_THD_THD, h=BIG_HEADS, hg=BIG_GROUPS, t_q=BIG_TOKENS, t_kv=BIG_TOKENS,
                 d_qk=BIG_DIM, d_v=BIG_DIM)
    from_longest = bits(THD_THD_THD, h=BIG_HEADS, hg=BIG_GROUPS, t_q=BIG_SEQLEN,
                        t_kv=BIG_SEQLEN, d_qk=BIG_DIM, d_v=BIG_DIM)
    assert truth == 64, "fixture no longer needs 64-bit offsets"
    assert from_longest == 32, "fixture no longer demonstrates the understatement"
    assert BIG_HEADS * BIG_DIM * BIG_TOKENS > INT32_MAX


@requires_memory
@pytest.mark.parametrize("index", [0, BIG_BATCH // 2, BIG_BATCH - 1])
def test_bf16_matches_a_standalone_run_past_the_boundary(index):
    """Each sampled sequence matches the same sequence run alone, at 64-bit offset scale.

    Sequence `BIG_BATCH - 1` carries the largest offset and is the first to wrap; sequence 0 has
    offset zero and would survive any wraparound, so it is the control showing the comparison
    itself is sound.
    """
    q, k, v, cu_seqlens = _big_batch()
    packed = _run(q, k, v, cu_seqlens, BIG_SEQLEN)
    alone = _run_one_sequence(q, k, v, index)
    error = _relative_rms(packed[index * BIG_SEQLEN : (index + 1) * BIG_SEQLEN], alone)
    assert error < 1e-3, f"sequence {index} diverged: relative RMS {error:.3e}"


@requires_memory
@pytest.mark.parametrize("index", [0, BIG_BATCH - 1])
def test_fp8_matches_a_standalone_run_past_the_boundary(index):
    """The same comparison in FP8, where every other test runs comfortably inside 32 bits."""
    q, k, v, cu_seqlens = _big_batch()
    packed = _run(q, k, v, cu_seqlens, BIG_SEQLEN, fp8=True)
    alone = _run_one_sequence(q, k, v, index, fp8=True)
    error = _relative_rms(packed[index * BIG_SEQLEN : (index + 1) * BIG_SEQLEN], alone)
    # Looser than BF16 because the packed run's amax covers every sequence and the standalone
    # run's covers one, so the two do not share a scale. An addressing fault is orders of magnitude
    # larger than this bound.
    assert error < 0.15, f"sequence {index} diverged under FP8: relative RMS {error:.4f}"


@requires_memory
def test_the_last_sequence_is_not_silently_empty():
    """The sequence at the largest offset carries real, finite output.

    Catches a wrapped offset landing outside the buffer and producing zeros or non-finite values at
    the tail, cheaply and for every run rather than only for sampled sequences.
    """
    q, k, v, cu_seqlens = _big_batch()
    out = _run(q, k, v, cu_seqlens, BIG_SEQLEN)
    tail = out[(BIG_BATCH - 1) * BIG_SEQLEN :]
    assert torch.isfinite(tail.float()).all(), "non-finite output in the last packed sequence"
    assert tail.abs().max().item() > 0.0, "the last packed sequence is all zeros"


# --------------------------------------------------------------------------------------
# contents of cu_seqlens
# --------------------------------------------------------------------------------------

SMALL_SEQLENS = [128, 64, 256, 64]
SMALL_TOKENS = sum(SMALL_SEQLENS)
WELL_FORMED = [0, 128, 192, 448, 512]
SMALL_HEADS, SMALL_DIM = 4, 128


def _run_with_metadata(cu_seqlens_list, validate=True, fp8=True):
    """One packed forward with the given cu_seqlens, with content validation on or off."""
    import transformer_engine.pytorch.attention.dot_product_attention.dot_product_attention as dpa

    previous = dpa._thd_validate_metadata
    dpa._thd_validate_metadata = validate
    try:
        q, k, v = (torch.randn(SMALL_TOKENS, SMALL_HEADS, SMALL_DIM,
                               device="cuda", dtype=torch.bfloat16) * 0.5 for _ in range(3))
        cu_seqlens = torch.tensor(cu_seqlens_list, dtype=torch.int32, device="cuda")
        return _run(q, k, v, cu_seqlens, 256, fp8=fp8)
    finally:
        dpa._thd_validate_metadata = previous


def test_well_formed_metadata_is_accepted():
    """The valid case runs clean with validation enabled.

    Without this every rejection below could pass by rejecting everything.
    """
    out = _run_with_metadata(WELL_FORMED, validate=True)
    assert torch.isfinite(out.float()).all()


@pytest.mark.parametrize(
    "name,cu_seqlens,message",
    [
        ("non_monotonic", [0, 128, 64, 448, 512], "non-decreasing"),
        ("negative", [0, -8, 192, 448, 512], "non-decreasing"),
        ("nonzero_start", [64, 128, 192, 448, 512], "must be 0"),
        ("beyond_the_token_count", [0, 128, 192, 448, 99999], "exceeds"),
    ],
)
def test_malformed_metadata_is_rejected(name, cu_seqlens, message):
    """Each malformation raises an error naming the problem.

    Three of these four produce no error at all without the guard and two return plausible finite
    numbers, so only an explicit rejection holds the contract. The message is asserted as well: an
    offset fault surfacing as a bare error several frames away is barely better than silence.
    """
    with pytest.raises(ValueError, match=message):
        _run_with_metadata(cu_seqlens, validate=True)


def test_a_short_cu_seqlens_is_accepted_by_construction():
    """A vector with one entry too few is accepted, and the limitation is recorded rather than
    assumed covered by the rejection tests above.

    The batch size is defined as the length of cu_seqlens minus one, so a short vector is
    indistinguishable from a smaller batch at this layer and the effect is a silently dropped
    trailing sequence. Detecting it would need a batch size supplied independently by the caller,
    which the interface does not carry.
    """
    out = _run_with_metadata([0, 128, 192, 448], validate=True)
    assert out is not None


@pytest.mark.parametrize("cu_seqlens", [[0, 128, 64, 448, 512], [64, 128, 192, 448, 512]])
def test_malformed_metadata_is_silent_without_the_guard(cu_seqlens):
    """These run to completion when validation is off, which is why the guard exists.

    Also catches the guard becoming unconditional. That would be a real regression, since it puts a
    device-to-host synchronisation in the training path, and no other test would notice because
    every other case here wants validation on.
    """
    out = _run_with_metadata(cu_seqlens, validate=False)
    assert out is not None, "expected the unguarded path to complete, however wrongly"
