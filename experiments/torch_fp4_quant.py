import torch
from torch.nn.functional import ScalingType, SwizzleType, scaled_mm
from dataclasses import dataclass
from transformer_engine.pytorch.custom_recipes.quantization_nvfp4 import NVFP4TestOutputs
"""
Adapted from torch.testing._internal.common_quantized
"""

# largest power of 2 representable in `torch.float8_e4m3fn`
F8E4M3_LARGEST_POW2 = 8
# largest power of 2 representable in `torch.float4_e2m1fn_x2`
FP4E2M1FN_LARGEST_POW2 = 2.0

# exponent bias of `torch.float8_e8m0fnu`
F8E8M0_EXP_BIAS = 127
# exponent and mantissa bits of `torch.float4_e2m1fn_x2`
FP4_EBITS, FP4_MBITS = 2, 1
FP4_MAX_VAL = 6.0

# max value of `torch.float8_e4m3fn` (448)
FP8E4M3_MAX_VAL = torch.finfo(torch.float8_e4m3fn).max

EBITS_F32, MBITS_F32 = 8, 23

def ceil_div(a, b):
    return (a + b - 1) // b

def to_blocked(input_matrix) -> torch.Tensor:
    """
    Rearrange a large matrix by breaking it into blocks and applying the rearrangement pattern.

    See:
        https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout

    Args:
        input_matrix: Input tensor of shape (H, W)

    Returns:
        Rearranged tensor of shape (32*ceil_div(H,128), 16*ceil_div(W,4))
    """
    rows, cols = input_matrix.shape
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)

    # Calculate the padded shape
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    padded = input_matrix
    # Ideally we would use torch.nn.pad but it doesn't support float8_e8m0fnu for now
    if (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros((padded_rows, padded_cols), device=input_matrix.device, dtype=input_matrix.dtype)
        padded[:rows, :cols] = input_matrix

    # Rearrange the blocks
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)

    return rearranged.flatten()


def _n_ones(n: int) -> int:
    return (1 << n) - 1


F32_EXP_BIAS = _n_ones(EBITS_F32 - 1)


def _f32_to_floatx_unpacked(x: torch.Tensor, ebits: int, mbits: int) -> torch.Tensor:
    """Convert FP32 numbers to sub-byte floating point numbers with the given
    number of exponent and mantissa bits.

    Input: torch.Tensor of dtype torch.float
    Output: torch.Tensor of dtype torch.uint8, where the bit encoding is stored
    in the least significant bits. e.g.
      fp4: bits 0-3 empty and bits 4-7 in fp4_e2m1 encoding
      fp6: bits 0-1 empty and bits 2-7 in fp6_e2m3 or fp6_e3m2 encoding

    Note: there are no special values (NaN, inf) support in this code. Values
    outside the representable range of Floatx after rounding are clamped to the
    maximum Floatx magnitude (sign is preserved).

    Code below is an adaptation of https://fburl.com/code/ciwofcg4

    Background 1: last answer in https://stackoverflow.com/q/8981913
    Background 2: Computer Organization and Design, RISC-V edition, Chapter 3.5
    """
    assert x.dtype == torch.float
    assert 1 + ebits + mbits <= 8

    # calculate constants
    exp_bias = _n_ones(ebits - 1)
    max_int = _n_ones(ebits + mbits)
    sign_mask = 1 << (ebits + mbits)

    # TODO document this better
    magic_adder = _n_ones(MBITS_F32 - mbits - 1)

    # all E bits and M bits are 1s
    max_normal = 2 ** (_n_ones(ebits) - exp_bias) * (_n_ones(mbits + 1) / (2**mbits))

    # E bits = 1, M bits = 0
    min_normal = 2 ** (1 - exp_bias)

    denorm_exp = (
        # exp bias conversion between formats
        (F32_EXP_BIAS - exp_bias)
        # mantissa length difference between formats
        + (MBITS_F32 - mbits)
        # add one to encoded exponent for denormalized numbers
        + 1
    )
    denorm_mask_int = denorm_exp << MBITS_F32

    # reinterpret int32 as float32
    denorm_mask_float = torch.tensor(denorm_mask_int, dtype=torch.int32).view(torch.float32)

    # save the sign
    # Note that we have torch.uint32, but some ops like cpu bit shifts
    # do not work on it. So, we stay in int32.
    x = x.view(torch.int32)
    sign = x & 0x80000000

    # set everything to positive, will add sign back at the end
    x = x ^ sign

    # TODO: can the branch floating point comparisons below be done without
    # converting to float? probably but need to verify
    x = x.view(torch.float)

    # rewrite saturate/denorm/norm branches without explicit data dependent
    # control flow, to be more compiler friendly
    saturate_mask = x >= max_normal
    denormal_mask = torch.logical_and(torch.logical_not(saturate_mask), x < min_normal)
    normal_mask = torch.logical_not(torch.logical_or(saturate_mask, denormal_mask))

    #
    # branch 1: saturate to max val - handled later in the code which combines
    #   the branches
    #

    #
    # branch 2: to conversion to denormal as well as rounding up to normal
    #
    denormal_x = x + denorm_mask_float
    denormal_x = denormal_x.view(torch.int32)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(torch.uint8)

    #
    # branch 3: stay in normal range, adjust the exponent and round
    #
    normal_x = x.view(torch.int32)
    # resulting mantissa is odd
    mant_odd = (normal_x >> (MBITS_F32 - mbits)) & 1
    # update exponent, rounding bias part 1
    val_to_add = ((exp_bias - F32_EXP_BIAS) << MBITS_F32) + magic_adder
    normal_x += val_to_add
    # rounding bias part 2
    normal_x += mant_odd
    # take the bits!
    normal_x = normal_x >> (MBITS_F32 - mbits)
    normal_x = normal_x.to(torch.uint8)

    #
    # combine the branches
    #
    x = torch.full_like(x, max_int, dtype=torch.uint8)
    x = torch.where(denormal_mask, denormal_x, x)
    x = torch.where(normal_mask, normal_x, x)

    # add sign back
    sign_lp = sign >> (MBITS_F32 + EBITS_F32 - mbits - ebits)
    sign_lp = sign_lp.to(torch.uint8)
    # Right shift of a negative signed integer can fill the least significant
    # bits with either 1s or 0s, depending on the implementation. Since PyTorch
    # doesn't have an uint32 dtype, we mask out these bits to get just the
    # f4 sign bit
    sign_lp = sign_lp & sign_mask
    x = x | sign_lp

    return x.to(torch.uint8)


def down_size(size):
    assert size[-1] % 2 == 0, f"{size} last dim not divisible by two"
    return (*size[:-1], size[-1] // 2)


def pack_uint4(uint8_data) -> torch.Tensor:
    # converting to uint8 for operations
    shape = uint8_data.shape
    assert shape[-1] % 2 == 0
    uint8_data = uint8_data.contiguous().view(-1)
    return (uint8_data[1::2] << 4 | uint8_data[::2]).view(down_size(shape))


def _float_to_float4_e2m1fn_x2(x):
#    assert x.dtype == torch.bfloat16
    x = _f32_to_floatx_unpacked(x, FP4_EBITS, FP4_MBITS)
    x = pack_uint4(x)
    x = x.view(torch.float4_e2m1fn_x2)
    return x


# https://github.com/pytorch/pytorch/blob/a5436a5e8e4ee42d1debf52c2786c7ae0043a434/test/test_scaled_matmul_cuda.py#L490
def torch_quantize_to_nvfp4(x, block_size: int = 16, cast_to_bfloat16: bool = True, eps: float = 1e-12):
    # Simple (slow) reference implementation of NVFP4 two-level-scaling

    orig_shape = x.shape
    x = x.reshape(-1, block_size)

    """
    Scale Factor and quantization
    - Calculate blockwise scales in HP, used for (HP -> FP4)
    - Calculate global scale factor for quantizing blockwise scales from HP -> FP8
    - Quantize blockwise scales from HP -> FP8 using global scale factor.  Blockwise scale factors now in FP8
    - Dequantize quantized scale factors from FP8 -> HP using global scale factor
    - Invert to get encoding factor for converting inputs from HP -> FP4
    - Blockwise quantize inputs from HP -> FP4
    - Return quantized inputs, quantized (FP8) blockwise scales, and global scale factor (needed for decoding quantized blockwise scales)
    """
    
    # Per-block-amax
    block_max = torch.amax(torch.abs(x), 1) + eps

    # Per-tensor max
    global_max = x.abs().max()

    # Constants
    # Global encoding scale for block-scales
    S_enc = FP4_MAX_VAL * FP8E4M3_MAX_VAL / global_max
    S_dec = 1.0 / S_enc

    # Per-block decode-scale
    S_dec_b = block_max / FP4_MAX_VAL

    # Stored scaled-e4m3 per-block decode scales
    S_dec_b_e4m3 = (S_dec_b * S_enc).to(torch.float8_e4m3fn)

    # Actual per-block encoding scale - dequantize the fp8 quantized block scales
    # = 1 / (S_dec * S_dec_b_e4m3) where S_dec dequantizes scales from FP8 -> FP32
    S_enc_b = (
        S_enc / S_dec_b_e4m3.float()
    )  

    # scale & reshape input, reshape scales
    x = (S_enc_b.unsqueeze(1) * x)
    if cast_to_bfloat16:
        x = x.bfloat16()
    else:
        assert x.dtype == torch.float32

    x = x.reshape(orig_shape)
    S_dec_b_e4m3 = S_dec_b_e4m3.reshape(orig_shape[0], -1)

    # cast input
    x_fp4 = _float_to_float4_e2m1fn_x2(x.float())

    # fp4x2, fp8_e4m3, float respectively
    return NVFP4TestOutputs(qx=x_fp4.view(torch.uint8),
                               global_amax=global_max,
                               blockwise_scales=S_dec_b,
                               global_encode_scale=S_enc,
                               global_decode_scale=S_dec,
                               quantized_decode_scales=S_dec_b_e4m3,
                               dequantized_encode_scales=S_enc_b)

# https://github.com/pytorch/pytorch/blob/a5436a5e8e4ee42d1debf52c2786c7ae0043a434/test/test_scaled_matmul_cuda.py#L1820
def torch_native_nvfp4_gemm(
    xq, x_scale_blocked, x_global_scale, wq, w_scale_blocked, w_global_scale, output_dtype
):

    RECIPE = [ScalingType.BlockWise1x16, ScalingType.TensorWise]
    swizzle = [SwizzleType.SWIZZLE_32_4_4, SwizzleType.NO_SWIZZLE]

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
