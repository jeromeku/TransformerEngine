"""Which context-parallel comm types the FP8 THD backend admits, and that the narrowing is specific.

Context parallelism pads each document to a multiple of 2*cp_size, so cu_seqlens_padded != cu_seqlens
on every CP call. The per-rank kernels thread those padded offsets, and a2a and (delayed-scaling) p2p
both handle the FP8 THD forward and backward. So a2a and delayed p2p are admitted for FP8 THD under CP;
current-scaling / mxfp8 p2p, all_gather and a2a+p2p are not.

Neither the BF16 THD path, the dense (BSHD) FP8 path, nor the non-CP FP8 path shares this narrowing,
so each is pinned unaffected.

    python3 -m pytest test_fp8_thd_cp_gate.py -q -rA
"""

import pathlib
import sys

import pytest
import torch

from transformer_engine.common import recipe

_current_file = pathlib.Path(__file__).resolve()
sys.path = [str(_current_file.parent), str(_current_file.parent.parent)] + sys.path
from packed_input_utils import PACKED_CONFIGS

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="device-only")

FP8_SUB_BACKEND = "FusedAttention/2"


def selected_backend(config, *, context_parallel, cp_comm_type, layout="thd_thd_thd", fp8=True,
                     mask="padding_causal", pad_between_seqs=True, softmax_type="vanilla",
                     rec=None, local_recipes=None):
    """The backend the selector chooses for this CP configuration.

    pad_between_seqs defaults True because CP always produces padded != actual; the selector's CP
    clauses are what is under test, not the contiguous packing already covered elsewhere.
    """
    from transformer_engine.pytorch.attention.dot_product_attention import utils as U
    from transformer_engine.pytorch.attention.dot_product_attention.dot_product_attention import (
        _attention_backends,
    )

    fp8_meta = None
    if fp8:
        fp8_meta = {"recipe": rec if rec is not None else recipe.DelayedScaling(fp8_dpa=True)}
        if local_recipes is not None:
            fp8_meta["local_recipes"] = local_recipes
    params = U.AttentionParams(
        qkv_type=torch.Tensor, qkv_dtype=torch.bfloat16, qkv_layout=layout,
        batch_size=3, num_heads=config.num_heads, num_gqa_groups=config.num_gqa_groups,
        max_seqlen_q=2048, max_seqlen_kv=2048,
        head_dim_qk=config.head_dim_qk, head_dim_v=config.head_dim_v,
        attn_mask_type=mask, window_size=(-1, 0) if "causal" in mask else (-1, -1),
        core_attention_bias_type="no_bias", attention_dropout=0.0,
        pad_between_seqs=pad_between_seqs, is_training=True, fp8=fp8,
        softmax_type=softmax_type,
        context_parallel=context_parallel, cp_comm_type=cp_comm_type,
        fp8_meta=fp8_meta,
    )
    # The chosen backend is cached; invalidate before querying so a previous answer is not returned,
    # and after, so the next attention call in this process selects for itself.
    _attention_backends["backend_selection_requires_update"] = True
    try:
        flash, _, fused, sub, unfused, _ = U.get_attention_backend(params)
    finally:
        _attention_backends["backend_selection_requires_update"] = True
    return ("FlashAttention" if flash else f"FusedAttention/{int(sub)}" if fused
            else "Unfused" if unfused else "NO_BACKEND")


# --------------------------------------------------------------------------------------
# the admitted comm type
# --------------------------------------------------------------------------------------


def test_fp8_thd_a2a_cp_selects_the_fp8_backend():
    """FP8 over packed input with a2a context parallelism selects the FP8 fused sub-backend.

    This is the enablement Phase 1.2 adds. a2a is the only THD+SWA/sink comm type, and the only one
    whose CP implementation threads cu_seqlens_*_padded into the per-rank kernel.
    """
    got = selected_backend(PACKED_CONFIGS["omnii_8b_tp1"], context_parallel=True, cp_comm_type="a2a")
    assert got == FP8_SUB_BACKEND, f"FP8 THD a2a CP selected {got!r}"


# --------------------------------------------------------------------------------------
# the refused comm types (the narrowing is specific)
# --------------------------------------------------------------------------------------


def test_fp8_thd_p2p_delayed_cp_selects_the_fp8_backend():
    """p2p with delayed scaling over FP8 packed input is admitted (the p2p enablement)."""
    got = selected_backend(PACKED_CONFIGS["omnii_8b_tp1"], context_parallel=True, cp_comm_type="p2p")
    assert got == FP8_SUB_BACKEND, f"FP8 THD p2p delayed CP selected {got!r}"


def test_fp8_thd_p2p_current_scaling_is_refused():
    got = selected_backend(
        PACKED_CONFIGS["omnii_8b_tp1"], context_parallel=True, cp_comm_type="p2p",
        rec=recipe.Float8CurrentScaling(fp8_dpa=True),
    )
    assert got != FP8_SUB_BACKEND, f"FP8 THD p2p current-scaling admitted: {got!r}"


def test_fp8_thd_p2p_mxfp8_is_refused():
    got = selected_backend(
        PACKED_CONFIGS["omnii_8b_tp1"], context_parallel=True, cp_comm_type="p2p",
        rec=recipe.MXFP8BlockScaling(fp8_dpa=True),
    )
    assert got != FP8_SUB_BACKEND, f"FP8 THD p2p mxfp8 admitted: {got!r}"


def test_fp8_thd_p2p_mixed_metadata_current_is_refused():
    """Effective recipe is local_recipes[0], so a delayed surrogate over current scaling stays refused."""
    got = selected_backend(
        PACKED_CONFIGS["omnii_8b_tp1"], context_parallel=True, cp_comm_type="p2p",
        rec=recipe.DelayedScaling(fp8_dpa=True),
        local_recipes=[recipe.Float8CurrentScaling(fp8_dpa=True)],
    )
    assert got != FP8_SUB_BACKEND, f"FP8 THD p2p mixed-metadata current admitted: {got!r}"


@pytest.mark.parametrize("cp_comm_type", ["all_gather", "a2a+p2p"])
def test_fp8_thd_other_comm_types_are_refused(cp_comm_type):
    """all_gather and a2a+p2p over FP8 packed input stay refused.

    THD is already unsupported for these two comm types regardless of precision; pinned here so the
    a2a admission is not misread as opening the whole THD CP surface.
    """
    got = selected_backend(PACKED_CONFIGS["omnii_8b_tp1"], context_parallel=True,
                           cp_comm_type=cp_comm_type)
    assert got != FP8_SUB_BACKEND, f"FP8 THD {cp_comm_type} CP was admitted: {got!r}"


def test_fp8_thd_a2a_requires_even_heads_and_groups():
    """a2a shards heads across ranks, so it needs num_heads and num_gqa_groups both even.

    omnii_15b at TP=2 has 5 KV groups; the a2a divisibility rule refuses it, and that refusal must
    survive the a2a admission above rather than being bypassed by it.
    """
    got = selected_backend(PACKED_CONFIGS["omnii_15b_tp2"], context_parallel=True,
                           cp_comm_type="a2a")
    assert got != FP8_SUB_BACKEND, f"FP8 THD a2a CP admitted odd groups: {got!r}"


# --------------------------------------------------------------------------------------
# neighbouring paths are untouched
# --------------------------------------------------------------------------------------


def test_bf16_thd_a2a_cp_is_unaffected():
    """BF16 packed input with a2a CP already worked and must keep selecting a fused backend.

    The narrowed clause is FP8-specific; a BF16 regression would mean it caught the wrong precision.
    """
    got = selected_backend(PACKED_CONFIGS["omnii_8b_tp1"], context_parallel=True, cp_comm_type="a2a",
                           fp8=False)
    assert got.startswith("FusedAttention"), f"BF16 THD a2a CP regressed: {got!r}"


def test_dense_fp8_a2a_cp_is_unaffected():
    """FP8 over dense (BSHD) input with a2a CP is a pre-existing path and must be unchanged.

    The narrowed clause keys on qkv_format == 'thd'; a dense regression would mean it caught the
    wrong format.
    """
    got = selected_backend(PACKED_CONFIGS["omnii_8b_tp1"], context_parallel=True, cp_comm_type="a2a",
                           layout="bshd_bshd_bshd")
    assert got == FP8_SUB_BACKEND, f"dense FP8 a2a CP regressed: {got!r}"


def test_fp8_thd_without_cp_is_unaffected():
    """FP8 packed input outside context parallelism keeps selecting the FP8 backend.

    The clause lives under `if context_parallel`; a leak into the non-CP path would disable the
    contiguous packing already in production.
    """
    got = selected_backend(PACKED_CONFIGS["omnii_8b_tp1"], context_parallel=False,
                           cp_comm_type="a2a")
    assert got == FP8_SUB_BACKEND, f"non-CP FP8 THD regressed: {got!r}"
