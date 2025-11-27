# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.pytorch import NVFP4Quantizer
from transformer_engine.pytorch.custom_recipes.quantization_nvfp4 import NVFP4QuantizerRef
from transformer_engine.pytorch.custom_recipes import utils


# largest power of 2 representable in `torch.float8_e4m3fn`
F8E4M3_LARGEST_POW2 = 8
# largest power of 2 representable in `torch.float4_e2m1fn_x2`
FP4E2M1FN_LARGEST_POW2 = 2.0
# max value of `torch.float8_e4m3fn` (448)
F8E4M3_MAX_VAL = torch.finfo(torch.float8_e4m3fn).max
# exponent bias of `torch.float8_e8m0fnu`
F8E8M0_EXP_BIAS = 127
# exponent and mantissa bits of `torch.float4_e2m1fn_x2`
FP4_EBITS, FP4_MBITS = 2, 1
FP4_MAX_VAL = 6.0


def data_to_nvfp4_with_global_scale(x, block_size):
    # Simple (slow) reference implementation of NVFP4 two-level-scaling
    from torch.testing._internal.common_quantized import _bfloat16_to_float4_e2m1fn_x2

    orig_shape = x.shape
    x = x.reshape(-1, block_size)

    # Per-block-amax
    block_max = torch.amax(torch.abs(x), 1) + 1e-12

    # Per-tensor max
    global_max = x.abs().max()

    # Constants
    # Global encoding scale for block-scales
    S_enc = FP4_MAX_VAL * F8E4M3_MAX_VAL / global_max
    S_dec = 1.0 / S_enc

    # Per-block decode-scale
    S_dec_b = block_max / FP4_MAX_VAL

    # Stored scaled-e4m3 per-block decode scales
    S_dec_b_e4m3 = (S_dec_b * S_enc).to(torch.float8_e4m3fn)

    # Actual per-block encoding scale
    S_enc_b = S_enc / S_dec_b_e4m3.float()

    # scale & reshape input, reshape scales
    x = (S_enc_b.unsqueeze(1) * x).bfloat16().reshape(orig_shape)
    S_dec_b_e4m3 = S_dec_b_e4m3.reshape(orig_shape[0], -1)

    # cast input
    x_fp4 = _bfloat16_to_float4_e2m1fn_x2(x)

    # fp4x2, fp8_e4m3, float respectively
    return x_fp4, S_dec_b_e4m3, S_dec.float()


def torch_native_gemm(x, w, y_ref, xq_ref, wq_ref, x_scales_ref, w_scales_ref, output_dtype):
    from torch.nn.functional import scaled_mm, ScalingType, SwizzleType
    from torch.testing._internal.common_quantized import to_blocked

    x_scale_ref, x_global_scale_ref = x_scales_ref
    w_scale_ref, w_global_scale_ref = w_scales_ref
    swizzle = [SwizzleType.SWIZZLE_32_4_4, SwizzleType.NO_SWIZZLE]
    xq, x_scale, x_global_scale = data_to_nvfp4_with_global_scale(x, 16)
    wq, w_scale, w_global_scale = data_to_nvfp4_with_global_scale(w, 16)
    x_scale_blocked = to_blocked(x_scale)
    w_scale_blocked = to_blocked(w_scale)
    RECIPE = [ScalingType.BlockWise1x16, ScalingType.TensorWise]

    out = scaled_mm(
        xq,
        wq.T,
        scale_a=[x_scale_blocked, x_global_scale],
        scale_recipe_a=RECIPE,
        scale_b=[w_scale_blocked, w_global_scale],
        scale_recipe_b=RECIPE,
        swizzle_a=swizzle,
        swizzle_b=swizzle,
        output_dtype=output_dtype,
    )
    return out


def check_nvfp4_gemm_versus_reference(
    x_dtype: torch.dtype,
    w_dtype: torch.dtype,
    out_dtype: torch.dtype,
    M: int,
    K: int,
    N: int,
    accumulate: bool,
    *,
    x_columnwise: bool = False,
    w_columnwise: bool = False,
):
    te_dtype = tex.DType.kFloat4E2M1

    # Setup device and random seed
    device = "cuda"
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Input tensors
    x_shape = (K, M) if x_columnwise else (M, K)
    w_shape = (K, N) if w_columnwise else (N, K)
    x = torch.randn(x_shape, dtype=x_dtype, device=device)
    w = torch.randn(w_shape, dtype=w_dtype, device=device)

    # Setup out tensor if accumulate is True
    if accumulate:
        out = torch.randn((M, N), dtype=out_dtype, device=device)
    else:
        out = None

    # Native TE NVFP4 quantization
    x_quantizer = NVFP4Quantizer(
        fp4_dtype=te_dtype,
        rowwise=True,
        columnwise=True,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
    )
    w_quantizer = NVFP4Quantizer(
        fp4_dtype=te_dtype,
        rowwise=True,
        columnwise=True,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
    )

    # Quantize x and w
    x_nvfp4_native = x_quantizer.make_empty(
        x_shape, dtype=x_dtype, device=device, requires_grad=False
    )
    x_nvfp4_native = x_quantizer.update_quantized(x, x_nvfp4_native)
    w_nvfp4_native = w_quantizer.make_empty(
        w_shape, dtype=w_dtype, device=device, requires_grad=False
    )
    w_nvfp4_native = w_quantizer.update_quantized(w, w_nvfp4_native)
    breakpoint()
    # Extract quantized data from native NVFP4Tensors
    qx_data = (
        x_nvfp4_native._columnwise_data.view(dtype=torch.uint8)
        if x_columnwise
        else x_nvfp4_native._rowwise_data.view(dtype=torch.uint8)
    )
    qw_data = (
        w_nvfp4_native._columnwise_data.view(dtype=torch.uint8)
        if w_columnwise
        else w_nvfp4_native._rowwise_data.view(dtype=torch.uint8)
    )
    sx_native = (
        x_nvfp4_native._columnwise_scale_inv if x_columnwise else x_nvfp4_native._rowwise_scale_inv
    )
    sw_native = (
        w_nvfp4_native._columnwise_scale_inv if w_columnwise else w_nvfp4_native._rowwise_scale_inv
    )
    breakpoint()
    # Trim quantized data to match the actual tensor dimensions (remove padding)
    qx_data = qx_data[:M, :]
    qw_data = qw_data[:N, :]

    # NVFP4 uses 16-element blocks, trim scales to remove padding
    block_length = 16  # NVFP4 uses 16-element blocks
    expected_sx_cols = expected_sw_cols = K // block_length
    # Trim the scales to remove padding
    sx_trimmed = sx_native[:M, :expected_sx_cols]
    sw_trimmed = sw_native[:N, :expected_sw_cols]

    # Native scales are stored as uint8 but need to be interpreted as float8_e4m3fn
    # for the reference GEMM to work correctly
    sx_trimmed = sx_trimmed.view(torch.float8_e4m3fn)
    sw_trimmed = sw_trimmed.view(torch.float8_e4m3fn)

    # Create reference quantizer for reference GEMM
    ref_quantizer = NVFP4QuantizerRef(
        dtype=utils.Fp4Formats.E2M1,
        rowwise=True,
        columnwise=True,
        pow_2_scales=False,
        eps=0.0,
        quant_tile_shape=(1, 16),
    )
    breakpoint()

    # Create reference quantized tensors needed by reference GEMM
    x_nvfp4_ref = ref_quantizer.quantize(x)
    w_nvfp4_ref = ref_quantizer.quantize(w)

    # Reference GEMM using quantizer's qgemm method
    y_ref = ref_quantizer.qgemm(
        qx=qx_data,
        qw=qw_data,
        m_params=None,  # MMParams not used in reference
        out_dtype=out_dtype,
        sx=sx_trimmed,
        sw=sw_trimmed,
        bias=None,  # No bias for this test
        out=out.clone() if accumulate else None,
        accumulate=accumulate,
        gemm_type=None,  # GEMMType not used in reference
        qresult_x=x_nvfp4_ref,
        qresult_w=w_nvfp4_ref,
    )

    x_ref_scales = [x_nvfp4_ref.scale, x_nvfp4_ref.global_amax_row]
    w_ref_scales = [w_nvfp4_ref.scale, x_nvfp4_ref.global_amax_row]
    torch_ref = torch_native_gemm(
        x,
        w,
        y_ref=y_ref,
        xq_ref=qx_data,
        wq_ref=qw_data,
        x_scales_ref=x_ref_scales,
        w_scales_ref=w_ref_scales,
        output_dtype=out_dtype,
    )
    
    # Native TE GEMM using tex.generic_gemm (cuBLAS GEMM)
    # Allocate cuBLAS workspace
    workspace = torch.empty(4, dtype=torch.uint8, device=device)

    transa = True if not w_columnwise else False
    transb = False if not x_columnwise else True
    out_quantizer = None
    bias = None
    bias_dtype = TE_DType[torch.bfloat16]
    use_gelu = False
    gelu_input = None
    use_grad = False
    use_split_accumulator = False

    # Native cuBLAS GEMM
    # return type is out, bias_grad, gelu_input, extra_output
    # We are just capturing out.
    y_native = tex.generic_gemm(
        w_nvfp4_native,
        transa,
        x_nvfp4_native,
        transb,
        out.clone() if accumulate else None,
        out_quantizer,
        TE_DType[out_dtype],
        bias,
        bias_dtype,
        use_gelu,
        gelu_input,
        use_grad,
        workspace,
        workspace.shape[0],
        accumulate,
        use_split_accumulator,
    )[0]

    # just in case of accumulation, make sure y_ref and y_native are not the same tensor
    assert y_ref is not y_native, "y_ref and y_native should not be the same tensor"
    # Reset nans to zeros because torch.assert_close does not assume nans to be equal
    assert not torch.isnan(y_ref.float()).all(), "All elements are nan"
    y_ref = torch.where(y_ref.isnan(), torch.zeros_like(y_ref), y_ref)
    y_native = torch.where(y_native.isnan(), torch.zeros_like(y_native), y_native)

    # Compare results with some tolerance
    torch.testing.assert_close(y_native, y_ref, atol=8e-3, rtol=8e-3)


SHAPES = [
    (128, 128, 128),
    (256, 128, 256),
    (256, 256, 256),
    (256, 1024, 256),
    (1024, 1024, 1024),
    (4096, 512, 3072),
    (112, 128, 96),
    (304, 640, 304),
    (1008, 3072, 992),
    (256, 64, 256),
    (128, 128, 112),
]


def test_nvfp4_gemm_versus_reference(
    M: int,
    K: int,
    N: int,
    x_dtype: torch.dtype,
    w_dtype: torch.dtype,
    out_dtype: torch.dtype,
    accumulate: bool,
    is_x_columnwise: bool = False,
    is_w_columnwise: bool = False,
):
    check_nvfp4_gemm_versus_reference(
        x_dtype=x_dtype,
        w_dtype=w_dtype,
        out_dtype=out_dtype,
        M=M,
        K=K,
        N=N,
        accumulate=accumulate,
        x_columnwise=is_x_columnwise,
        w_columnwise=is_w_columnwise,
    )


if __name__ == "__main__":
    x_dtype = w_dtype = torch.bfloat16
    out_dtype = torch.float32
    M, N, K = 256, 1024, 512
    accumulate = True
    test_nvfp4_gemm_versus_reference(
        M, K, N, x_dtype=x_dtype, w_dtype=w_dtype, out_dtype=out_dtype, accumulate=accumulate
    )
