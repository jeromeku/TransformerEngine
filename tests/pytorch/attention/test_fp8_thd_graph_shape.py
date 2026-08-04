"""FP8 THD rounds its cuDNN graph's sequence extents up to buckets, and keeps the batch exact.

`FADescriptor_v1` orders the fp8 graph caches on `b`, `s_q` and `s_kv`. Under THD those arrive as
the document count and the longest document in the micro-batch, both of which change nearly every
call, so copying them into the descriptor gives every micro-batch its own graph. The implementation
rounds `s_q` and `s_kv` up to buckets before the descriptor is built, and leaves `b` at the exact
document count.

Only the sequence extents. The F16 ragged path also replaces the batch with a bucketed capacity;
the FP8 workspace is sized from the declared batch as well as the declared extent, where the
flash-style F16 graph is insensitive to the over-description, so declaring a batch capacity here
multiplies the workspace. No closed-form cost model is assumed; the workspace assertions below are
ratios between measurements.

The bucket policy is therefore local to FP8: powers of two with a floor of 2 below 1024, and the
shared token buckets at and above 1024, bounding every extent at `s <= bucket(s) <= 2 * s`. The
bound is tight only at the floor, where `bucket(1) == 2`; above it the inequality is strict.

Because `b` is exact, `actual_b` and the graph batch are always equal today. The `(actual_b, b)`
plumbing in the conversion kernels is defence in depth for a future batch capacity, and **nothing
here exercises `actual_b < b`** -- the document-count cases below vary the exact batch, not a batch
bucket. What they do exercise is the `b + 1` terminal-offset launch extent, which is a real boundary
at 128, 256 and 512 documents.

These tests pin the properties that make the rounding safe, none of which are timing measurements:

  - raising the declared sequence bound changes no output, which is what lets it be rounded;
  - results are unchanged across the bucket boundaries the rounding introduces;
  - every document, including the last, is addressed correctly in both the forward and the backward
    at document counts that are exact multiples of the conversion block size;
  - short documents do not inflate the workspace, which a floored bucket would;
  - the FP8 THD zero fill performs no device-to-host read.

Throughput is not asserted here. These are correctness, safety and resource properties; the cache
behaviour they protect is a performance property, measured separately.

    NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 python3 -m pytest test_fp8_thd_graph_shape.py -q -rA
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import textwrap

import pytest
import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

from packed_input_utils import PACKED_CONFIGS, make_packed_batch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="device-only")

CONFIG = PACKED_CONFIGS["omnii_8b_tp1"]
TENSORS = ("out", "dq", "dk", "dv")

# The ragged-offset conversion writes b+1 entries, so a document count that is an exact multiple of
# the 128-thread conversion block leaves the terminal offset unwritten if that kernel reuses the
# sequence-length launch extent. 128 and 256 are those cases; their neighbours bracket them.
DOCUMENT_COUNTS = (1, 2, 31, 32, 33, 127, 128, 129, 255, 256, 257)

# Document counts where a lost terminal offset would corrupt the final document: the exact multiples
# of the 128-thread conversion block that the implementation calls out. The gradient comparison is
# restricted to these because it is the expensive check.
TERMINAL_OFFSET_COUNTS = (128, 256, 512)

# Sequence buckets are powers of two below 1024 and the shared token buckets above, so these
# straddle a bucket edge at constant document count.
TOKEN_COUNTS = (1023, 1024, 1025, 2048, 2049)


def attention(mask: str = "padding_causal") -> te.DotProductAttention:
    return te.DotProductAttention(
        num_attention_heads=CONFIG.num_heads,
        kv_channels=CONFIG.head_dim_qk,
        num_gqa_groups=CONFIG.num_gqa_groups,
        attention_dropout=0.0,
        qkv_format="thd",
        attn_mask_type=mask,
        softmax_scale=1.0 / math.sqrt(CONFIG.head_dim_qk),
    ).cuda()


def run(
    batch, module, fp8: bool = True, max_seqlen: int | None = None, fast_zero_fill: bool = True
) -> dict[str, torch.Tensor]:
    """One forward and backward; returns the output and the three input gradients."""
    q, k, v = (t.detach().clone().requires_grad_(True) for t in (batch.q, batch.k, batch.v))
    bound = batch.max_seqlen if max_seqlen is None else max_seqlen
    kwargs = dict(
        cu_seqlens_q=batch.cu_seqlens,
        cu_seqlens_kv=batch.cu_seqlens,
        max_seqlen_q=bound,
        max_seqlen_kv=bound,
        fast_zero_fill=fast_zero_fill,
    )
    if fp8:
        with te.fp8_autocast(enabled=True, fp8_recipe=recipe.DelayedScaling(fp8_dpa=True)):
            out = module(q, k, v, **kwargs)
    else:
        out = module(q, k, v, **kwargs)
    torch.manual_seed(0)
    grad = torch.randn_like(out)
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), grad)
    return {"out": out.detach(), "dq": dq, "dk": dk, "dv": dv}


def even_lengths(documents: int, per_document: int = 64) -> list[int]:
    return [per_document] * documents


@pytest.mark.parametrize("documents", DOCUMENT_COUNTS)
def test_padded_sequence_bound_is_inert(documents):
    """Raising max_seqlen to the packed capacity must change nothing, bit for bit.

    This is the property the quantization depends on: `s_q` and `s_kv` are declared upper bounds,
    so replacing them with a larger bucketed bound has to be invisible in the results. Every
    document is a different length so the true bound is well below the padded one.

    The padded bound is twice the longest document, which always crosses at least one bucket edge.
    It is deliberately not the full packed capacity: declaring the whole token count as the sequence
    extent is measured to exhaust device memory at these document counts, which is the reason the
    implementation buckets the longest document rather than adopting the F16 path's total-token
    form.
    """
    lengths = [64 + (i * 37) % 512 for i in range(documents)]
    batch = make_packed_batch(CONFIG, distribution=lengths, seed=documents)

    # A fresh module per variant, each seeing exactly one call. Delayed scaling advances its amax
    # history on every forward, so three calls on one module would compare the 1st against the 3rd
    # and disagree for a reason that has nothing to do with the bound under test.
    true_bound = run(batch, attention())
    padded = run(batch, attention(), max_seqlen=2 * batch.max_seqlen)
    control = run(batch, attention())

    for name in TENSORS:
        assert torch.equal(true_bound[name], control[name]), (
            f"{name}: two identical calls disagree, so bit-exactness is not a meaningful floor "
            "here and this test cannot distinguish the bound's effect from run-to-run noise"
        )
        assert torch.equal(true_bound[name], padded[name]), (
            f"{name}: declaring max_seqlen={2 * batch.max_seqlen} instead of {batch.max_seqlen} "
            "changed the result; the bound is not an upper bound in practice"
        )


def compare_per_document(batch, fp8, bf16, names):
    """Cosine and norm ratio for each document of each named tensor, against the BF16 reference.

    Two metrics, not one: cosine is invariant to a scalar, which is exactly the error class a
    per-tensor-descale fault produces, so the norm ratio has to be asserted alongside it.
    """
    for name in names:
        for index in range(batch.batch_size):
            a = batch.sequence(fp8[name], index).float().flatten()
            b = batch.sequence(bf16[name], index).float().flatten()
            cosine = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
            norm_ratio = (a.norm() / b.norm()).item()
            where = f"{name}, document {index} of {batch.batch_size}"
            assert cosine > 0.99, (
                f"{where}: cosine {cosine:.4f} against the BF16 reference; the packed base pointer "
                "for this document is wrong"
            )
            assert (
                0.9 < norm_ratio < 1.1
            ), f"{where}: norm ratio {norm_ratio:.4f} against the BF16 reference"


@pytest.mark.parametrize("documents", DOCUMENT_COUNTS)
def test_document_count_crosses_block_boundaries(documents):
    """Every document, including the last, is addressed correctly at each document count.

    The ragged-offset conversion writes `b + 1` entries, one more than the sequence-length
    conversion beside it, so it needs its own launch extent. Sharing the sequence-length grid leaves
    the terminal offset unwritten whenever `b` is an exact multiple of the 128-thread block, and the
    final document then reads from an uninitialised base pointer.

    BF16 over the identical packed batch is the reference. It takes the F16 ragged path, which has
    always sized that launch correctly, so it is independent of the code under test.

    Forward only here; the backward is checked at the boundary counts by the test below, which is
    the expensive one.
    """
    batch = make_packed_batch(CONFIG, distribution=even_lengths(documents), seed=documents)
    fp8 = run(batch, attention())
    bf16 = run(batch, attention(), fp8=False)

    for name in TENSORS:
        assert torch.isfinite(fp8[name]).all(), f"{name}: non-finite values"
    compare_per_document(batch, fp8, bf16, ("out",))


@pytest.mark.parametrize("documents", TERMINAL_OFFSET_COUNTS)
def test_gradients_at_the_terminal_offset_boundary(documents):
    """The backward addresses the final document correctly too.

    The forward and the backward convert ragged offsets at separate launch sites -- the backward
    converts twice, once per layout -- so a terminal-offset defect can exist in one and not the
    other. Checking only `out` would let a backward-only fault through as finite but wrong
    gradients for the last document, which is why this repeats the per-document comparison on dQ,
    dK and dV at exactly the document counts where the boundary bites.
    """
    batch = make_packed_batch(CONFIG, distribution=even_lengths(documents), seed=documents)
    fp8 = run(batch, attention())
    bf16 = run(batch, attention(), fp8=False)

    compare_per_document(batch, fp8, bf16, TENSORS)


@pytest.mark.parametrize("tokens", TOKEN_COUNTS)
def test_token_count_crosses_bucket_boundaries(tokens):
    """Results do not depend on which token bucket the packed row lands in.

    `get_max_tokens` rounds 1023 and 1024 to 1024 but 1025 to 2048, so two rows differing by two
    tokens are given graphs of different declared capacity. The answers must not differ in kind.
    """
    lengths = [tokens // 2, tokens - tokens // 2]
    batch = make_packed_batch(CONFIG, distribution=lengths, seed=tokens)
    fp8 = run(batch, attention())
    bf16 = run(batch, attention(), fp8=False)

    for name in TENSORS:
        assert torch.isfinite(fp8[name]).all(), f"{name}: non-finite values"
    a = fp8["out"].float().flatten()
    b = bf16["out"].float().flatten()
    cosine = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    assert cosine > 0.99, f"{tokens} tokens: cosine {cosine:.4f} against the BF16 reference"


def test_zero_fill_does_not_read_from_the_device():
    """The FP8 THD zero fill must not perform a device-to-host scalar read.

    It used to compute the unused suffix from `cu_seqlens[-1]` with `item<int32_t>()`, once for the
    output and once per input gradient. Each of those synchronizes and stops the host running ahead
    of the device. `set_sync_debug_mode("error")` turns any such read into an exception.
    """
    batch = make_packed_batch(CONFIG, distribution=even_lengths(4), seed=0)
    module = attention()
    run(batch, module)  # warm up outside the guarded region; setup is allowed to synchronize

    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        run(batch, module, fast_zero_fill=True)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()


def test_the_synchronization_detector_fires():
    """Negative control for the test above.

    Without this, that test passes identically if `set_sync_debug_mode` is unsupported, spelled
    wrong, or silently ignored for reads issued from C++.
    """
    scalar = torch.ones(1, device="cuda")
    torch.cuda.set_sync_debug_mode("error")
    try:
        with pytest.raises(RuntimeError):
            scalar.item()
    finally:
        torch.cuda.set_sync_debug_mode("default")


# --------------------------------------------------------------------------------------
# Workspace. The declared sequence extent is a resource decision, not only a correctness one, and
# nothing else in this suite would notice a bucket policy that inflates it: a policy can be correct
# on every value and still request orders of magnitude more memory than it needs. The assertions
# below are ratios between measurements and assume no cost model.
# --------------------------------------------------------------------------------------

_PEAK_WORKLOAD = textwrap.dedent(
    """
    import math, sys, torch
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe

    documents, per_document = int(sys.argv[1]), int(sys.argv[2])
    heads, gqa_groups, head_dim = 32, 8, 128
    total = documents * per_document
    cu_seqlens = torch.arange(0, total + 1, per_document, dtype=torch.int32, device="cuda")

    def draw(count):
        return torch.randn(total, count, head_dim, device="cuda",
                           dtype=torch.bfloat16).requires_grad_(True)

    q, k, v = draw(heads), draw(gqa_groups), draw(gqa_groups)
    module = te.DotProductAttention(
        num_attention_heads=heads, kv_channels=head_dim, num_gqa_groups=gqa_groups,
        attention_dropout=0.0, qkv_format="thd", attn_mask_type="padding_causal",
        softmax_scale=1.0 / math.sqrt(head_dim)).cuda()

    torch.cuda.reset_peak_memory_stats()
    with te.fp8_autocast(enabled=True, fp8_recipe=recipe.DelayedScaling(fp8_dpa=True)):
        out = module(q, k, v, cu_seqlens_q=cu_seqlens, cu_seqlens_kv=cu_seqlens,
                     max_seqlen_q=per_document, max_seqlen_kv=per_document)
    torch.autograd.grad(out, (q, k, v), torch.randn_like(out))
    torch.cuda.synchronize()
    print("peak-gib", torch.cuda.max_memory_allocated() / 2**30)
    """
)


def peak_gib(documents: int, per_document: int, timeout: int = 900) -> float:
    """Peak device memory for one packed FP8 attention call, in its own process.

    Isolated because a failure here is an out-of-memory error, which leaves the caching allocator
    in a state that perturbs every later measurement in the same process.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _PEAK_WORKLOAD, str(documents), str(per_document)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    line = next((l for l in proc.stdout.splitlines() if l.startswith("peak-gib")), None)
    assert line is not None, (
        f"{documents} documents of {per_document} tokens did not complete:\n"
        f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    )
    return float(line.split()[1])


def require_idle_device(fraction: float = 0.5):
    """Skip when the device is too busy for a peak-memory assertion to mean anything.

    These tests compare peak allocations, so a co-tenant holding most of the card turns a healthy
    run into an out-of-memory failure that says nothing about the code. Skipping is the honest
    outcome; the reason is visible under `-rA`.
    """
    # Release this process's cached-but-unused blocks first. The correctness tests above leave the
    # caching allocator holding most of the card, which the subprocess cannot use and which would
    # otherwise make this skip on an idle machine.
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    if free < fraction * total:
        pytest.skip(
            f"device has {free / 2**30:.1f} GiB free of {total / 2**30:.1f} GiB; "
            "peak-memory assertions need a substantially idle device"
        )


def test_one_token_documents_do_not_inflate_the_workspace():
    """One-token documents cost far less than the same count of 64-token ones.

    A bucket policy that floors short sequences describes every one-token document as a floor-sized
    sequence, so at a floor of 1,024 these two cases declare the same extent and cost the same.

    The assertion is a ratio between two measurements, so it carries no absolute memory constant and
    holds on any device large enough to run the reference.
    """
    require_idle_device()
    short = peak_gib(512, 1)
    reference = peak_gib(512, 64)
    assert short < reference / 4, (
        f"512 one-token documents peaked at {short:.3f} GiB against {reference:.3f} GiB for the "
        "same count at 64 tokens. The declared sequence extent is not tracking the actual one, "
        "which is what a floored bucket does"
    )


def test_many_one_token_documents_stay_linear_in_document_count():
    """4,096 one-token documents complete, and cost about eight times 512 of them.

    A bucket floor that rounds a one-token document up to a large extent makes this allocation fail
    outright, so completing at all is most of the assertion. The linearity bound then says the
    growth came from the document count and not from the sequence extent. Both measurements are
    well under a gigabyte, so this needs almost no free memory.
    """
    require_idle_device()
    small = peak_gib(512, 1)
    large = peak_gib(4096, 1)
    ratio = large / small
    assert ratio < 12.0, (
        f"4,096 one-token documents peaked at {large:.3f} GiB against {small:.3f} GiB for 512 of "
        f"them, a ratio of {ratio:.1f} for an eightfold document count. Growth that is faster than "
        "linear means the sequence extent is being inflated as well"
    )


# --------------------------------------------------------------------------------------
# Graph cache. The point of bucketing is that many distinct input shapes share one cuDNN graph.
# Until the counters existed that could only be inferred from timing, and timing said the wrong
# thing at least once: a window reporting 108 new input shapes turned out to contain exactly one
# cold graph. These read the caches directly.
#
# Each case runs in a subprocess. The caches are process-lifetime and resetting the counters does
# not evict them, so a second test in the same process would see the first one's entries.
# --------------------------------------------------------------------------------------

_CACHE_WORKLOAD = textwrap.dedent(
    """
    import json, math, sys, torch
    import transformer_engine.pytorch as te
    import transformer_engine_torch as tex
    from transformer_engine.common import recipe

    rows = json.loads(sys.argv[1])
    heads, gqa_groups, head_dim = 32, 8, 128
    module = te.DotProductAttention(
        num_attention_heads=heads, kv_channels=head_dim, num_gqa_groups=gqa_groups,
        attention_dropout=0.0, qkv_format="thd", attn_mask_type="padding_causal",
        softmax_scale=1.0 / math.sqrt(head_dim)).cuda()

    tex.reset_fused_attn_fp8_cache_stats()
    for lengths in rows:
        total = sum(lengths)
        cu_seqlens = torch.tensor([0] + list(torch.cumsum(torch.tensor(lengths), 0)),
                                  dtype=torch.int32, device="cuda")
        def draw(count):
            return torch.randn(total, count, head_dim, device="cuda",
                               dtype=torch.bfloat16).requires_grad_(True)
        q, k, v = draw(heads), draw(gqa_groups), draw(gqa_groups)
        with te.fp8_autocast(enabled=True, fp8_recipe=recipe.DelayedScaling(fp8_dpa=True)):
            out = module(q, k, v, cu_seqlens_q=cu_seqlens, cu_seqlens_kv=cu_seqlens,
                         max_seqlen_q=max(lengths), max_seqlen_kv=max(lengths))
        torch.autograd.grad(out, (q, k, v), torch.randn_like(out))
    torch.cuda.synchronize()
    print("stats " + json.dumps(tex.get_fused_attn_fp8_cache_stats()))
    """
)


CACHE_STATS = "NVTE_FP8_ATTN_CACHE_STATS"


def cache_stats(rows: list[list[int]], timeout: int = 900) -> dict:
    """Run these packed rows in a clean process and return the FP8 graph cache counters.

    Collection is off by default, so the subprocess sets the flag. The `enabled` field is asserted
    by every caller: without that check a disabled build reports zeros, and a zero entry count reads
    as "no graphs were built" rather than "nothing was counted".
    """
    env = dict(os.environ, **{CACHE_STATS: "1"})
    proc = subprocess.run(
        [sys.executable, "-c", _CACHE_WORKLOAD, json.dumps(rows)],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    line = next((l for l in proc.stdout.splitlines() if l.startswith("stats ")), None)
    assert (
        line is not None
    ), f"workload did not complete:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    return json.loads(line[len("stats ") :])


# Longest documents 1100 through 2000: ten distinct values, all inside the 2048 bucket.
ONE_BUCKET_ROWS = [[length, 2100 - length] for length in range(1100, 2001, 100)]

# Longest documents in the 2048, 4096 and 8192 buckets.
THREE_BUCKET_ROWS = [[1500, 500], [3000, 500], [5000, 500]]


def test_distinct_shapes_in_one_bucket_share_one_graph():
    """Ten different longest-document values build exactly one graph, in each direction.

    This is the property the whole fix exists to produce, asserted from the cache rather than
    inferred from a stopwatch. Before the fix these ten rows would have built ten graphs each way.
    """
    require_idle_device()
    stats = cache_stats(ONE_BUCKET_ROWS)
    assert stats["enabled"], "counter collection is off, so every count below would be a vacuous 0"
    assert stats["fprop_entries"] == 1, (
        f"ten distinct longest-document values built {stats['fprop_entries']} forward graphs; "
        "they all fall in the 2048 bucket and should share one"
    )
    assert (
        stats["bprop_entries"] == 1
    ), f"{stats['bprop_entries']} backward graphs for the same ten rows"
    # Without this the entry assertions would also pass if attention never ran at all.
    assert (
        stats["fprop_hits"] > 0 and stats["bprop_hits"] > 0
    ), f"no cache hits recorded, so the counters are not observing a live cache: {stats}"


def test_crossing_a_bucket_builds_another_graph():
    """Negative control for the test above.

    Without it, `fprop_entries == 1` passes just as well if the counter is stuck at one, if the
    cache key collapsed everything, or if bucketing were replaced by a constant. Three rows in
    three different buckets must produce three graphs.
    """
    require_idle_device()
    stats = cache_stats(THREE_BUCKET_ROWS)
    assert stats["enabled"], "counter collection is off, so every count below would be a vacuous 0"
    assert stats["fprop_entries"] == len(THREE_BUCKET_ROWS), (
        f"three rows in three different buckets built {stats['fprop_entries']} forward graphs, "
        f"expected {len(THREE_BUCKET_ROWS)}; the counter does not track graph creation"
    )
    assert stats["bprop_entries"] == len(
        THREE_BUCKET_ROWS
    ), f"{stats['bprop_entries']} backward graphs, expected {len(THREE_BUCKET_ROWS)}"


def test_cache_counters_are_off_by_default():
    """The counters are diagnostic and must not collect unless asked.

    Also the negative control for `enabled`: if the field were hardcoded true, or if the gate were
    ignored, this fails.
    """
    require_idle_device()
    proc = subprocess.run(
        [sys.executable, "-c", _CACHE_WORKLOAD, json.dumps(THREE_BUCKET_ROWS)],
        capture_output=True,
        text=True,
        timeout=900,
        env={k: v for k, v in os.environ.items() if k != CACHE_STATS},
    )
    line = next((l for l in proc.stdout.splitlines() if l.startswith("stats ")), None)
    assert line is not None, f"workload did not complete:\n{proc.stderr[-2000:]}"
    stats = json.loads(line[len("stats ") :])
    assert not stats["enabled"], "counters report enabled without the flag set"
    assert (
        stats["fprop_lookups"] == 0 and stats["bprop_lookups"] == 0
    ), f"counters collected without the flag set: {stats}"
