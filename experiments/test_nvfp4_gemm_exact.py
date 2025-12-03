# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.pytorch import NVFP4Quantizer
from transformer_engine.pytorch.custom_recipes.quantization_nvfp4 import (
    NVFP4QuantizerRef,
    cast_from_fp4x2,
    cast_to_fp4x2,
    NVFP4TestOutputs,
)
from transformer_engine.pytorch.custom_recipes import utils
from torch_fp4_quant import (
    torch_quantize_to_nvfp4,
    FP4_MAX_VAL,
    FP8E4M3_MAX_VAL,
    torch_native_nvfp4_gemm,
    to_blocked,
    _f32_to_floatx_unpacked,
)
from dataclasses import dataclass


def unpack_fp4(t: torch.Tensor):
    """
    Unpacked packed fp4s

    Assumes low nibble encodes even elements, upper nibble encodes odd elements
    """
    assert t.dtype == torch.uint8
    assert t.ndim == 2
    rows, cols = t.shape
    result = torch.zeros(rows, cols * 2, dtype=torch.uint8, device=t.device)

    # Extract nibbles
    evens = t & 0xF
    odds = t >> 4

    result[:, ::2] = evens
    result[:, 1::2] = odds

    return result


def _cast_mx_to_float(t: torch.Tensor):
    return t.view(torch.uint8).float()


def mxfp_diff(t1: torch.Tensor, t2: torch.Tensor):
    t1, t2 = map(_cast_mx_to_float, [t1, t2])
    return t1.sub(t2).abs().max().item()


def clamp_to_fp4(t: torch.Tensor):
    return torch.clamp(t, -FP4_MAX_VAL, FP4_MAX_VAL)


def check_quantized_outputs(
    ref_outputs: NVFP4TestOutputs, test_outputs: NVFP4TestOutputs, precision: int = 8
):
    float_fmt = f".{precision}f"
    for f in NVFP4TestOutputs.__dataclass_fields__:
        ref, test = getattr(ref_outputs, f), getattr(test_outputs, f)
        print(f"{f}:")

        shapes = [ref.shape, test.shape]
        dtypes = [ref.dtype, test.dtype]
        print(f"{shapes=} {dtypes=}")

        if ref.shape != test.shape:
            try:
                test = test.reshape_as(ref)
            except:
                print(f"Shape mismatch {f}")
                continue

        if ref.dtype != test.dtype:
            print(f"Dtype mismatch: {f}")
            test = test.to(ref.dtype)

        if ref.dtype == torch.float32:
            diff = ref.sub(test).abs().max().item()
        else:
            diff = mxfp_diff(ref, test)

        print(f"{diff=:{float_fmt}}")

    # # Manually cast torch scaled x to fp4
    # torch_x_fp4 = cast_to_fp4x2(torch_x.clipped_x)
    # # Sanity check
    # te_fp4_check = cast_to_fp4x2(te_ref.clipped_x)
    # assert te_fp4_check.equal(te_ref.qx)

    # Unpack so that each fp4 lives in its own uint8
    unpacked_fp4_ref = unpack_fp4(ref_outputs.qx)
    unpacked_fp4_test = unpack_fp4(test_outputs.qx)

    mismatches = unpacked_fp4_ref != unpacked_fp4_test
    print(f"Num mismatches after fp4 cast: {torch.sum(mismatches).item()}")

    dq_ref = cast_from_fp4x2(ref_outputs.qx, torch.float32)
    dq_test = cast_from_fp4x2(test_outputs.qx, torch.float32)
    print(f"dqx diff: {dq_ref.sub(dq_test).abs().max().item():{float_fmt}}")

    # if mismatches > 0:
    #     ridx, cidx = torch.where(mismatches)
    #     idx = torch.nonzero(mismatches)

    #     # Num bits for E2M1
    #     EBITS = 2
    #     MBITS = 1
    #     breakpoint()
    #     torch_x_fp4_bitcast = _f32_to_floatx_unpacked(torch_test.scaled_x, EBITS, MBITS)
    #     te_x_fp4_bitcast = _f32_to_floatx_unpacked(te_ref.scaled_x, EBITS, MBITS)
    #     fp4_diff_bitcast = mxfp_diff(torch_x_fp4_bitcast, te_x_fp4_bitcast)
    #     print(f"Bitcast fp4 diff: {fp4_diff_bitcast:{float_fmt}}")

    #     # Sanity check
    #     breakpoint()
    #     unpacked_torch_bitcast_fp4 = unpack_fp4(torch_test.qx)
    #     torch_check = mxfp_diff(torch_x_fp4_bitcast, unpacked_torch_bitcast_fp4)
    #     print(f"Torch unpacked bitcast fp4 check: {torch_check:{float_fmt}}")


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
    precision: int = 8,
    debug: bool = False,
):
    float_fmt = f".{precision}f"
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

    TE_FP4_MAX_VAL = torch.tensor(6.0, device=device, dtype=torch.float32)
    TE_FP8_E4M3_MAX_VAL  = torch.tensor(448.0, device=device, dtype=torch.float32)

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

    # Create reference quantized tensors needed by reference GEMM
    x_nvfp4_ref = ref_quantizer.quantize(x)
    qx_ref = x_nvfp4_ref.data
    sx_ref = x_nvfp4_ref.scale

    w_nvfp4_ref = ref_quantizer.quantize(w)
    sw_ref = w_nvfp4_ref.scale
    wx_ref = w_nvfp4_ref.data

    te_scale_diff = mxfp_diff(sx_ref, sx_trimmed)
    print(f"TE Scale diff: {te_scale_diff:{float_fmt}}")
    te_qx_diff = mxfp_diff(qx_ref, qx_data)
    print(f"TE qx diff: {te_qx_diff:{float_fmt}}")

    # Repeat with extra returns
    quantize_ref = NVFP4QuantizerRef._quantize_blockwise_reference
    global_amax_x = torch.amax(torch.abs(x)).float()
    assert global_amax_x.float().equal(x_nvfp4_ref.global_amax_row.reshape_as(global_amax_x))

    te_ref_x: NVFP4TestOutputs = quantize_ref(
        x,
        x_nvfp4_ref.global_amax_row,
        tile_len_x=16,
        tile_len_y=1,
        pow_2_scales=False,
        debug=True,
    )
    te_ref_w: NVFP4TestOutputs = quantize_ref(
        w, w_nvfp4_ref.global_amax_row, tile_len_x=16, tile_len_y=1, pow_2_scales=False, debug=True
    )
    # sanity check
    print(f"Sanity check, qx_ref: {mxfp_diff(qx_ref, te_ref_x.qx):.4f}")
    print(f"Sanity check, sx_ref: {mxfp_diff(sx_ref, te_ref_x.quantized_decode_scales):.4f}")
    print(f"Sanity check, wx_ref: {mxfp_diff(wx_ref, te_ref_w.qx):.4f}")
    print(f"Sanity check, sw_ref: {mxfp_diff(sw_ref, te_ref_w.quantized_decode_scales):.4f}")

    torch_x: NVFP4TestOutputs = torch_quantize_to_nvfp4(
        x, cast_to_float=True, skip_bfloat16_cast_after_quant=True, invert_encode_scale=True
    )
    check_quantized_outputs(ref_outputs=te_ref_x, test_outputs=torch_x)

    breakpoint()
    
    """
    Note: use_te_max_norms is needed
    
    from torch_fp4_quant import FP4_MAX_VAL, FP8E4M3_MAX_VAL
    import numpy as np

    assert isinstance(FP4_MAX_VAL, float)
    assert isinstance(np.float64(FP4_MAX_VAL), float)

    Python's float is a C++ double (float64)

    TE explicitly constructs torch float32 scalars for these max norms whereas
    the pytorch test suite uses python builtin floats to represent these max norms.

    This results in small diffs when calculating scale factors, which compounds during quantization
    and results in catastrophic mismatches upon casting to fp4.

    Setting `use_te_max_norms=True` uses TE's numeric representation of max norms, which addresses these
    mismatches. 
    """
    
    torch_w: NVFP4TestOutputs = torch_quantize_to_nvfp4(
        w,
        cast_to_float=True,
        skip_bfloat16_cast_after_quant=True,
        invert_encode_scale=True,
        clamp_encode_scales=True,
        use_te_max_norms=True,
    )
    check_quantized_outputs(ref_outputs=te_ref_w, test_outputs=torch_w)

    breakpoint()
    x_scales_torch_blocked = to_blocked(torch_x.quantized_decode_scales)
    w_scales_torch_blocked = to_blocked(torch_w.quantized_decode_scales)

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

    breakpoint()
    torch_ref = torch_native_nvfp4_gemm(
        xq=torch_x.qx.view(torch.float4_e2m1fn_x2),
        x_scale_blocked=x_scales_torch_blocked,
        x_global_scale=torch_x.global_decode_scale.float(),
        wq=torch_w.qx.view(torch.float4_e2m1fn_x2),
        w_scale_blocked=w_scales_torch_blocked,
        w_global_scale=torch_w.global_decode_scale.float(),
        output_dtype=out_dtype,
    )
    breakpoint()
    diff = (torch_ref - y_ref).abs().max()
    print(f"Torch scale_mm vs TE Ref: {diff.cpu().item():.4f}")

    breakpoint()
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
    debug: bool = False,
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
        debug=debug,
    )


if __name__ == "__main__":
    x_dtype = w_dtype = torch.bfloat16
    out_dtype = torch.float32
    M, N, K = 256, 1024, 512
    accumulate = True
    debug = True
    test_nvfp4_gemm_versus_reference(
        M,
        K,
        N,
        x_dtype=x_dtype,
        w_dtype=w_dtype,
        out_dtype=out_dtype,
        accumulate=accumulate,
        debug=True,
    )
