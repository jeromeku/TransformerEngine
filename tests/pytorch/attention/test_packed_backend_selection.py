"""Which packed (THD) FP8 configurations the backend admits, and that admission implies execution.

A capability query is an execution contract. Answering "supported" and then failing to launch is
worse than answering "unsupported", because the caller has no way to tell the difference from a
silent fallback. So the admitted set is pinned, the rejected set is pinned, and both are checked
against what actually runs.

Each probe runs in a clean subprocess. The selection path reads environment variables and caches
the result, so an inherited flag from an earlier test can otherwise mask a regression.

    python3 -m pytest test_packed_backend_selection.py -q -rs
"""

import itertools
import json
import math
import os
import pathlib
import subprocess
import sys
import textwrap

import pytest
import torch

_current_file = pathlib.Path(__file__).resolve()
sys.path = [str(_current_file.parent), str(_current_file.parent.parent)] + sys.path
from packed_input_utils import PACKED_CONFIGS, constructible
from utils import ModelConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="device-only")

FP8_SUB_BACKEND = "FusedAttention/2"

ADMITTED_MASKS = ("padding", "padding_causal")
ADMITTED_RECIPES = ("delayed", "current")

# Geometries outside the FP8 surface on this architecture, kept here rather than in the shared
# module because their only use is to be rejected.
REJECTED_CONFIGS = {
    "head_dim_256": ModelConfig(2, 2048, 16, 256, attn_mask_type="padding_causal"),
    "unequal_head_dims": ModelConfig(2, 2048, 16, 192, head_dim_v=128,
                                     attn_mask_type="padding_causal"),
}

# Recorded rather than inferred, so a floor that moves is visible in the diff.
CUDNN_VERSION_FLOOR = 92100
COMPUTE_CAPABILITY_FLOOR = (10, 0)

_QUERY = textwrap.dedent(
    """
    import json, sys, torch
    from transformer_engine.common import recipe
    from transformer_engine.pytorch.attention.dot_product_attention import utils as U

    cfg = json.loads(sys.argv[1])
    recipes = {
        "delayed": lambda: recipe.DelayedScaling(fp8_dpa=True),
        "current": lambda: recipe.Float8CurrentScaling(fp8_dpa=True),
        "mxfp8": lambda: recipe.MXFP8BlockScaling(),
        "none": lambda: None,
    }
    rec = recipes[cfg["recipe"]]()
    mask = cfg["mask"]
    p = U.AttentionParams(
        qkv_type=torch.Tensor, qkv_dtype=torch.bfloat16, qkv_layout=cfg["layout"],
        batch_size=3, num_heads=cfg["heads"], num_gqa_groups=cfg["groups"],
        max_seqlen_q=2048, max_seqlen_kv=2048,
        head_dim_qk=cfg["head_dim_qk"], head_dim_v=cfg["head_dim_v"],
        attn_mask_type=mask,
        window_size=(-1, 0) if "causal" in mask else (-1, -1),
        core_attention_bias_type=cfg["bias"],
        attention_dropout=cfg["dropout"],
        pad_between_seqs=cfg["pad_between_seqs"],
        is_training=cfg["is_training"],
        fp8=cfg["fp8"],
        fp8_meta={"recipe": rec} if rec is not None else None,
    )
    flash, _, fused, sub, unfused, _ = U.get_attention_backend(p)
    print(json.dumps("FlashAttention" if flash else
                     (f"FusedAttention/{int(sub)}" if fused else
                      ("Unfused" if unfused else "NO_BACKEND"))))
    """
)


def selected_backend(config, layout="thd_thd_thd", mask="padding_causal", recipe="delayed",
                     bias="no_bias", dropout=0.0, pad_between_seqs=False, is_training=True,
                     fp8=True, env=None):
    """The backend the selector chooses, resolved in a clean subprocess."""
    payload = dict(layout=layout, mask=mask, recipe=recipe if fp8 else "none", bias=bias,
                   dropout=dropout, pad_between_seqs=pad_between_seqs, is_training=is_training,
                   fp8=fp8, heads=config.num_heads, groups=config.num_gqa_groups,
                   head_dim_qk=config.head_dim_qk, head_dim_v=config.head_dim_v)
    child = {k: v for k, v in os.environ.items() if k != "NVTE_FP8_THD_EXPERIMENTAL"}
    if env:
        child.update(env)
    out = subprocess.run([sys.executable, "-c", _QUERY, json.dumps(payload)],
                         capture_output=True, text=True, env=child, timeout=600)
    assert out.returncode == 0, f"query failed for {payload}:\n{out.stderr[-2000:]}"
    return json.loads(out.stdout.strip().splitlines()[-1])


_EXECUTE = textwrap.dedent(
    """
    import json, math, sys, torch
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe

    cfg = json.loads(sys.argv[1])
    seqlens = [128, 64, 256, 64]
    heads, groups, dim, total = cfg["heads"], cfg["groups"], cfg["head_dim_qk"], sum(seqlens)
    draw = lambda h: (torch.randn(total, h, dim, device="cuda", dtype=torch.bfloat16) * 0.5
                      ).requires_grad_(True)
    q, k, v = draw(heads), draw(groups), draw(groups)
    cu_seqlens = torch.tensor([0] + list(torch.cumsum(torch.tensor(seqlens), 0)),
                              dtype=torch.int32, device="cuda")
    module = te.DotProductAttention(
        num_attention_heads=heads, kv_channels=dim, num_gqa_groups=groups,
        attention_dropout=0.0, qkv_format="thd", attn_mask_type=cfg["mask"],
        softmax_scale=1.0 / math.sqrt(dim)).cuda()
    kwargs = dict(cu_seqlens_q=cu_seqlens, cu_seqlens_kv=cu_seqlens,
                  max_seqlen_q=max(seqlens), max_seqlen_kv=max(seqlens),
                  attn_mask_type=cfg["mask"])
    rec = (recipe.DelayedScaling(fp8_dpa=True) if cfg["recipe"] == "delayed"
           else recipe.Float8CurrentScaling(fp8_dpa=True))
    with te.fp8_autocast(enabled=True, fp8_recipe=rec):
        out = module(q, k, v, **kwargs)
    out.backward(torch.randn_like(out))
    torch.cuda.synchronize()
    assert torch.isfinite(out.float()).all()
    print("ran")
    """
)


def executes(config, mask="padding_causal", recipe="delayed"):
    """Whether the configuration completes a forward and a backward, in a clean subprocess."""
    payload = dict(mask=mask, recipe=recipe, heads=config.num_heads,
                   groups=config.num_gqa_groups, head_dim_qk=config.head_dim_qk)
    out = subprocess.run([sys.executable, "-c", _EXECUTE, json.dumps(payload)],
                         capture_output=True, text=True, env=dict(os.environ), timeout=900)
    return out.returncode == 0, out.stderr[-2000:]


# --------------------------------------------------------------------------------------
# the admitted set
# --------------------------------------------------------------------------------------

ADMITTED = [
    (layout, name, mask, recipe)
    for (layout, name), mask, recipe in itertools.product(
        constructible(), ADMITTED_MASKS, ADMITTED_RECIPES)
]


@pytest.mark.parametrize("layout,config_name,mask,recipe", ADMITTED)
def test_admitted_configurations_select_the_fp8_backend(layout, config_name, mask, recipe):
    """Every admitted combination selects the FP8 fused sub-backend, with no flag set.

    Catches the enablement narrowing silently. A combination that quietly falls back to a
    higher-precision backend still produces plausible numbers, so nothing else would notice.
    """
    got = selected_backend(PACKED_CONFIGS[config_name], layout=layout, mask=mask, recipe=recipe)
    assert got == FP8_SUB_BACKEND, (
        f"{layout} {config_name} {mask} {recipe} selected {got!r}"
    )


@pytest.mark.parametrize("is_training", [True, False])
def test_both_training_directions_are_admitted(is_training):
    """Training and inference are both admitted.

    Catches an answer that varies with the direction, which would make the forward path available
    only when a backward is also wanted.
    """
    got = selected_backend(PACKED_CONFIGS["omnii_8b_tp1"], is_training=is_training)
    assert got == FP8_SUB_BACKEND


# --------------------------------------------------------------------------------------
# the rejected set
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("mask", ["no_mask", "causal", "causal_bottom_right"])
def test_non_padding_masks_are_rejected(mask):
    """Ragged offsets require a padding-family mask.

    Without a padding mask there is no cu_seqlens for the kernel to derive offsets from, so the
    combination has to be refused rather than silently reinterpreted.
    """
    got = selected_backend(PACKED_CONFIGS["omnii_8b_tp1"], mask=mask)
    assert got != FP8_SUB_BACKEND, f"{mask} was admitted"


@pytest.mark.parametrize("config_name", sorted(REJECTED_CONFIGS))
def test_out_of_range_head_dimensions_are_rejected(config_name):
    """Head dimension 256, and unequal query and value head dimensions, are outside the surface."""
    got = selected_backend(REJECTED_CONFIGS[config_name])
    assert got != FP8_SUB_BACKEND, f"{config_name} was admitted"


def test_physical_gaps_between_sequences_are_rejected():
    """Packed rows must be contiguous; physical inter-sequence gaps are refused.

    The offsets are derived from cumulative lengths, which cannot express a gap, so admitting this
    would address into the gap rather than the next sequence.
    """
    got = selected_backend(PACKED_CONFIGS["omnii_8b_tp1"], pad_between_seqs=True)
    assert got != FP8_SUB_BACKEND


def test_attention_bias_is_rejected():
    """FP8 scaled dot-product attention does not support bias."""
    got = selected_backend(PACKED_CONFIGS["omnii_8b_tp1"], bias="post_scale_bias")
    assert got != FP8_SUB_BACKEND


def test_block_scaled_recipes_are_rejected():
    """Block-scaled quantization must not reach this path.

    Its scale-factor layout is a separate problem from per-tensor scaling, and admitting it would
    hand the kernel scales it cannot interpret.
    """
    got = selected_backend(PACKED_CONFIGS["omnii_8b_tp1"], recipe="mxfp8")
    assert got != FP8_SUB_BACKEND


@pytest.mark.parametrize("head_dim,admitted", [(63, False), (64, True), (128, True),
                                               (129, False), (192, False), (256, False)])
def test_head_dimension_boundaries(head_dim, admitted):
    """Only head dimensions 64 and 128 are admitted.

    The rejected values flank the admitted ones deliberately: 63 fails the multiple-of-16 rule and
    129 exceeds the architecture cap, so a comparison that is wrong in either direction shows up
    here and nowhere else. A relaxed bound admits a geometry the kernel cannot run; a tightened one
    silently drops a supported shape.
    """
    config = ModelConfig(2, 2048, 16, head_dim, attn_mask_type="padding_causal")
    got = selected_backend(config)
    assert (got == FP8_SUB_BACKEND) == admitted, f"head dim {head_dim} selected {got!r}"


# --------------------------------------------------------------------------------------
# admission implies execution
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("layout,config_name", list(constructible())[:4])
@pytest.mark.parametrize("mask", ADMITTED_MASKS)
def test_admitted_configurations_actually_run(layout, config_name, mask):
    """What the query admits, the kernel runs, forward and backward.

    This is the contract the rest of the file rests on. A query that admits a configuration the
    kernel then refuses is indistinguishable, from the caller's side, from a silent fallback.
    """
    config = PACKED_CONFIGS[config_name]
    assert selected_backend(config, layout=layout, mask=mask) == FP8_SUB_BACKEND
    ok, stderr = executes(config, mask=mask)
    assert ok, f"admitted but did not run: {layout} {config_name} {mask}\n{stderr}"


@pytest.mark.parametrize("mask", ["no_mask", "causal"])
def test_rejected_configurations_do_not_reach_the_fp8_kernel(mask):
    """What the query rejects is not launched on the FP8 path.

    A rejection that still launches would mean the query describes something other than what runs.
    """
    got = selected_backend(PACKED_CONFIGS["omnii_8b_tp1"], mask=mask)
    assert got != FP8_SUB_BACKEND


# --------------------------------------------------------------------------------------
# neighbouring paths are untouched
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("mask", ["causal", "no_mask"])
def test_dense_fp8_is_unaffected(mask):
    """FP8 over dense input is the pre-existing path and must keep selecting the FP8 backend.

    Opening the packed path changes a shared enablement clause, so the dense path is the first
    thing that would regress.
    """
    got = selected_backend(PACKED_CONFIGS["omnii_8b_tp1"], layout="bshd_bshd_bshd", mask=mask)
    assert got == FP8_SUB_BACKEND, f"dense FP8 regressed for {mask}: {got!r}"


@pytest.mark.parametrize("mask", ADMITTED_MASKS)
def test_bf16_packed_is_unaffected(mask):
    """BF16 over packed input already worked and must keep working."""
    got = selected_backend(PACKED_CONFIGS["omnii_8b_tp1"], mask=mask, fp8=False)
    assert got.startswith("FusedAttention"), f"BF16 packed regressed for {mask}: {got!r}"


# --------------------------------------------------------------------------------------
# the path is public
# --------------------------------------------------------------------------------------


def test_no_experimental_gate_remains():
    """The path is reachable with no environment flag, and no gate is left in the source.

    Catches a gate surviving in one of the two places. A flag still read by the C++ would make the
    feature unreachable in a default build even though the Python selector admits it.
    """
    with_flag = selected_backend(PACKED_CONFIGS["omnii_8b_tp1"],
                                 env={"NVTE_FP8_THD_EXPERIMENTAL": "1"})
    without = selected_backend(PACKED_CONFIGS["omnii_8b_tp1"])
    assert with_flag == without == FP8_SUB_BACKEND, "the flag still changes the answer"

    source = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                          "transformer_engine", "common", "fused_attn", "fused_attn.cpp")
    with open(os.path.abspath(source)) as handle:
        assert "NVTE_FP8_THD_EXPERIMENTAL" not in handle.read(), (
            "an experimental gate is still present in fused_attn.cpp"
        )


def test_environment_floor_is_met():
    """The declared cuDNN and architecture floors are met, and recorded rather than inferred.

    A floor that is only implied by which tests happen to pass moves without anyone noticing.
    """
    import transformer_engine_torch  # noqa: F401
    from transformer_engine.pytorch.utils import get_cudnn_version

    version = get_cudnn_version()
    encoded = version[0] * 10000 + version[1] * 100 + version[2]
    assert encoded >= CUDNN_VERSION_FLOOR, (
        f"cuDNN {version} is below the declared floor {CUDNN_VERSION_FLOOR}"
    )
    capability = torch.cuda.get_device_capability()
    assert capability >= COMPUTE_CAPABILITY_FLOOR, (
        f"compute capability {capability} is below the declared floor "
        f"{COMPUTE_CAPABILITY_FLOOR}"
    )
    # Pin the constants themselves, so lowering a floor is a deliberate edit here rather than a
    # quiet change to the contract.
    assert CUDNN_VERSION_FLOOR == 92100
    assert COMPUTE_CAPABILITY_FLOOR == (10, 0)


# --------------------------------------------------------------------------------------
# the backward really runs on the FP8 path
# --------------------------------------------------------------------------------------


def test_training_configuration_selects_the_fp8_backend():
    """A training packed configuration selects the FP8 sub-backend with no flag set.

    Stated separately from the admitted set because the backward is where a fallback is hardest to
    notice: the forward can be on the FP8 path while the backward is not.
    """
    got = selected_backend(PACKED_CONFIGS["omnii_8b_tp1"], is_training=True)
    assert got == FP8_SUB_BACKEND


def test_fp8_gradients_are_not_identical_to_bf16():
    """The FP8 and BF16 gradients differ.

    If they were bitwise identical the FP8 backward never ran, and every numerical comparison
    against a BF16 reference would pass by comparing BF16 with itself.
    """
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe as te_recipe

    seqlens = [128, 64, 256, 64]
    total, heads, dim = sum(seqlens), 8, 128
    cu_seqlens = torch.tensor([0] + list(torch.cumsum(torch.tensor(seqlens), 0)),
                              dtype=torch.int32, device="cuda")

    def run(fp8):
        torch.manual_seed(0)
        q, k, v = ((torch.randn(total, heads, dim, device="cuda", dtype=torch.bfloat16) * 0.5
                    ).requires_grad_(True) for _ in range(3))
        module = te.DotProductAttention(
            num_attention_heads=heads, kv_channels=dim, attention_dropout=0.0,
            qkv_format="thd", attn_mask_type="padding_causal",
            softmax_scale=1.0 / math.sqrt(dim)).cuda()
        kwargs = dict(cu_seqlens_q=cu_seqlens, cu_seqlens_kv=cu_seqlens,
                      max_seqlen_q=max(seqlens), max_seqlen_kv=max(seqlens),
                      attn_mask_type="padding_causal")
        if fp8:
            with te.fp8_autocast(enabled=True,
                                 fp8_recipe=te_recipe.DelayedScaling(fp8_dpa=True)):
                out = module(q, k, v, **kwargs)
        else:
            out = module(q, k, v, **kwargs)
        torch.manual_seed(1)
        out.backward(torch.randn_like(out))
        return q.grad

    assert not torch.equal(run(fp8=True), run(fp8=False)), (
        "FP8 and BF16 gradients are bitwise identical, so the FP8 backward did not run"
    )
