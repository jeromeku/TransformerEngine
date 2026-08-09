"""The native fused-attn selector rejects non-vanilla FP8 THD (F7).

``nvte_get_fused_attn_backend`` is the public C-API / extension entry point. The Python selector
disables FP8 fused attention for packed THD with a non-vanilla softmax because the packed FP8
softmax-offset (sink) gradient is wrong (dot_product_attention/utils.py). Before F7 the native
selector did not encode that restriction and returned NVTE_FP8 for vanilla, off-by-one and learnable
THD alike -- so a caller trusting the native selector was told a known-silent-wrong mode is
supported. This pins that the native answer now matches Python: vanilla THD FP8 -> NVTE_FP8,
non-vanilla THD FP8 -> NOT NVTE_FP8. Dense (BSHD) non-vanilla FP8 is unaffected (the defect is
THD-specific), which the last test guards.

    python3 -m pytest test_fp8_thd_native_selector.py -q -rA
"""

import pytest
import torch

import transformer_engine.pytorch  # noqa: F401 -- loads the core library before the torch extension
import transformer_engine_torch as tex


def _fp8_backend(softmax_type, layout=tex.NVTE_QKV_Layout.NVTE_T3HD):
    """The native selector's backend for an FP8 attention query at this softmax type / layout."""
    return tex.get_fused_attn_backend(
        True,
        tex.DType.kFloat8E4M3,
        tex.DType.kFloat8E4M3,
        layout,
        tex.NVTE_Bias_Type.NVTE_NO_BIAS,
        tex.NVTE_Mask_Type.NVTE_PADDING_CAUSAL_MASK,
        softmax_type,
        0.0,
        24,  # num_attn_heads
        4,   # num_gqa_groups
        2048,  # max_seqlen_q
        2048,  # max_seqlen_kv
        128,  # head_dim_qk
        128,  # head_dim_v
        -1,
        0,
        False,
        False,
        False,
    )


def _vanilla_thd_is_fp8():
    if not torch.cuda.is_available():
        return False
    try:
        return _fp8_backend(tex.NVTE_Softmax_Type.NVTE_VANILLA_SOFTMAX) == (
            tex.NVTE_Fused_Attn_Backend.NVTE_FP8
        )
    except Exception:  # noqa: BLE001 -- an environment that cannot answer cannot run these
        return False


pytestmark = pytest.mark.skipif(
    not _vanilla_thd_is_fp8(),
    reason="no native FP8 THD backend for vanilla softmax here (needs SM100 + cuDNN>=9.21 fork)",
)


def test_vanilla_thd_fp8_is_admitted():
    # Positive control: vanilla is the one softmax the packed FP8 THD path is correct for.
    assert _fp8_backend(tex.NVTE_Softmax_Type.NVTE_VANILLA_SOFTMAX) == (
        tex.NVTE_Fused_Attn_Backend.NVTE_FP8
    )


@pytest.mark.parametrize(
    "softmax_type",
    [tex.NVTE_Softmax_Type.NVTE_OFF_BY_ONE_SOFTMAX, tex.NVTE_Softmax_Type.NVTE_LEARNABLE_SOFTMAX],
)
def test_non_vanilla_thd_fp8_is_rejected_natively(softmax_type):
    # F7: the native selector must NOT advertise FP8 for non-vanilla THD (wrong packed sink gradient).
    # Before F7 all three returned NVTE_FP8; the Python layer caught it, a C-API caller did not.
    got = _fp8_backend(softmax_type)
    assert got != tex.NVTE_Fused_Attn_Backend.NVTE_FP8, (
        f"native selector admitted FP8 for non-vanilla THD softmax {softmax_type}: {got}"
    )


@pytest.mark.parametrize(
    "softmax_type",
    [tex.NVTE_Softmax_Type.NVTE_OFF_BY_ONE_SOFTMAX, tex.NVTE_Softmax_Type.NVTE_LEARNABLE_SOFTMAX],
)
def test_non_vanilla_dense_fp8_is_unaffected(softmax_type):
    # Control: the F7 restriction is THD-specific. Dense (BSHD) non-vanilla FP8 must be unchanged --
    # if this regressed, the native clause caught the wrong layout. (Skips if dense vanilla isn't FP8
    # here either, i.e. the environment has no FP8 backend at all for this shape.)
    if _fp8_backend(
        tex.NVTE_Softmax_Type.NVTE_VANILLA_SOFTMAX, layout=tex.NVTE_QKV_Layout.NVTE_BSHD_BSHD_BSHD
    ) != tex.NVTE_Fused_Attn_Backend.NVTE_FP8:
        pytest.skip("no native dense FP8 backend here to compare against")
    got = _fp8_backend(softmax_type, layout=tex.NVTE_QKV_Layout.NVTE_BSHD_BSHD_BSHD)
    assert got == tex.NVTE_Fused_Attn_Backend.NVTE_FP8, (
        f"dense non-vanilla FP8 regressed under the THD-specific F7 clause: {got}"
    )
