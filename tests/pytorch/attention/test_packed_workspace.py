"""Fused attention over packed (THD) input stays inside the workspace it asked for.

Both entry points query the required workspace size with a null pointer, allocate exactly that,
then run. Nothing otherwise checks that the kernel stays inside what it asked for. Packed input
grows that size by five ragged-offset buffers in the forward and ten in the backward, since
gradients are addressed separately from reads, so an under-count is a real risk.

It is also a silent one. The bytes just past the workspace are legally mapped caching-allocator
pool memory, so an overrun is a valid write as far as compute-sanitizer is concerned. Only a guard
band placed either side of the workspace can see it, which is what
`NVTE_FP8_THD_WORKSPACE_CANARY=1` installs.

The negative control comes first in this file, because without it every assertion below reduces to
"no exception was raised", which passes identically when the guard band is never allocated, never
checked, or spelled wrong.

Each case runs in a subprocess: the variables are read once into function-local state, so they
must be set before the process first touches attention.

    python3 -m pytest test_packed_workspace.py -q -rs
"""

import os
import subprocess
import sys
import textwrap

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="device-only")

CANARY = "NVTE_FP8_THD_WORKSPACE_CANARY"
SELF_TEST = "NVTE_FP8_THD_WORKSPACE_CANARY_SELFTEST"

# Adversarial lengths: a length-1, several sub-tile, an exact tile, a long one. The sequence count
# drives the offset-buffer size, at one entry per sequence plus one, which is the part of the
# workspace packed input actually grows.
_WORKLOAD = textwrap.dedent(
    """
    import math, sys, torch
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe

    fp8 = sys.argv[1] == "1"
    backward = sys.argv[2] == "1"
    seqlens = [1, 37, 64, 320, 3, 511, 128]
    heads, dim, total = 4, 128, sum(seqlens)

    def draw():
        x = torch.randn(total, heads, dim, device="cuda", dtype=torch.bfloat16) * 0.5
        return x.requires_grad_(True) if backward else x

    q, k, v = draw(), draw(), draw()
    cu_seqlens = torch.tensor([0] + list(torch.cumsum(torch.tensor(seqlens), 0)),
                              dtype=torch.int32, device="cuda")
    module = te.DotProductAttention(
        num_attention_heads=heads, kv_channels=dim, attention_dropout=0.0, qkv_format="thd",
        attn_mask_type="padding_causal", softmax_scale=1.0 / math.sqrt(dim)).cuda()
    kwargs = dict(cu_seqlens_q=cu_seqlens, cu_seqlens_kv=cu_seqlens,
                  max_seqlen_q=max(seqlens), max_seqlen_kv=max(seqlens),
                  attn_mask_type="padding_causal")
    if fp8:
        with te.fp8_autocast(enabled=True, fp8_recipe=recipe.DelayedScaling(fp8_dpa=True)):
            out = module(q, k, v, **kwargs)
    else:
        out = module(q, k, v, **kwargs)
    if backward:
        out.backward(torch.randn_like(out))
    torch.cuda.synchronize()
    assert torch.isfinite(out.float()).all()
    print("workload-ok")
    """
)


def run_workload(fp8=True, backward=False, canary="1", self_test=None, timeout=900):
    """Run one packed workload in a clean process, with the guard band configured as given."""
    env = {k: v for k, v in os.environ.items() if k not in (CANARY, SELF_TEST)}
    if canary is not None:
        env[CANARY] = canary
    if self_test is not None:
        env[SELF_TEST] = self_test
    proc = subprocess.run(
        [sys.executable, "-c", _WORKLOAD, "1" if fp8 else "0", "1" if backward else "0"],
        capture_output=True, text=True, env=env, timeout=timeout)
    return proc, proc.stdout + proc.stderr


# --------------------------------------------------------------------------------------
# the negative control, which is what makes the rest non-vacuous
# --------------------------------------------------------------------------------------


def test_the_guard_band_reports_a_corrupted_byte():
    """Poisoning one guard byte is detected and named.

    Without this the checks below assert only that nothing was raised, which is equally true of a
    guard band that was never allocated or never inspected. The self-test hook writes a single byte
    into the trailing guard, so a passing run proves the mechanism is live end to end.
    """
    proc, out = run_workload(canary="1", self_test="1")
    assert proc.returncode != 0, (
        f"a guard byte was poisoned and the run still succeeded, so the check is dead:"
        f"\n{out[-3000:]}"
    )
    assert "workspace canary corrupted" in out, f"expected the guard's diagnostic:\n{out[-3000:]}"
    assert "trailing guard" in out, f"expected the trailing guard to be named:\n{out[-2000:]}"


def test_the_self_test_hook_is_off_by_default():
    """Poisoning requires its own variable.

    A hook left permanently armed would turn every guarded run into a false positive and make the
    control above meaningless.
    """
    proc, out = run_workload(canary="1", self_test=None)
    assert proc.returncode == 0, f"the guard alone must not corrupt anything:\n{out[-3000:]}"


# --------------------------------------------------------------------------------------
# the workspace itself
# --------------------------------------------------------------------------------------


def test_the_fp8_forward_stays_inside_its_workspace():
    """The FP8 packed forward writes nothing past the workspace it requested.

    Catches under-sizing from the ragged-offset buffers packed input adds. A shortfall corrupts
    unrelated allocator memory with no crash and no sanitizer report, and the symptoms surface
    later and elsewhere as wrong numbers.
    """
    proc, out = run_workload(fp8=True, canary="1")
    assert "workload-ok" in out, f"workload did not complete:\n{out[-3000:]}"
    assert proc.returncode == 0, f"workspace overrun in the FP8 packed forward:\n{out[-3000:]}"


def test_the_bf16_forward_stays_inside_its_workspace():
    """The BF16 packed forward is clean too.

    This path predates the FP8 work and shares its plumbing. If it fails while the FP8 one passes,
    the fault is in common code, which is a materially different diagnosis.
    """
    proc, out = run_workload(fp8=False, canary="1")
    assert "workload-ok" in out, f"workload did not complete:\n{out[-3000:]}"
    assert proc.returncode == 0, f"workspace overrun in the BF16 packed forward:\n{out[-3000:]}"


def test_the_fp8_backward_stays_inside_its_workspace():
    """The FP8 packed backward writes nothing past the workspace it requested.

    The backward adds twice the offset buffers the forward does, because gradients are addressed by
    their own layout separately from the reads, so it is the more likely of the two to be
    under-counted.
    """
    proc, out = run_workload(fp8=True, backward=True, canary="1")
    assert "workload-ok" in out, f"workload did not complete:\n{out[-3000:]}"
    assert proc.returncode == 0, f"workspace overrun in the FP8 packed backward:\n{out[-3000:]}"


@pytest.mark.parametrize("flag", ["0", ""])
def test_the_guard_band_is_off_by_default(flag):
    """With the guard disabled the ordinary path runs unchanged.

    Catches the guarded branch becoming load-bearing. If the interior-pointer offset the guard
    introduces leaked into the normal path it would misalign every workspace in ordinary use.
    """
    proc, out = run_workload(fp8=True, canary=flag)
    assert "workload-ok" in out, f"workload did not complete with the guard off:\n{out[-3000:]}"
    assert proc.returncode == 0, f"the unguarded path is broken:\n{out[-3000:]}"
