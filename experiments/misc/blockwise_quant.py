import torch
from transformer_engine.pytorch import NVFP4Quantizer, NVFP4Tensor
import te_quantize
from dataclasses import dataclass


@dataclass
class NVFP4QTensor:
    x: torch.Tensor
    qx: torch.Tensor
    decode_scales: torch.Tensor
    global_amax: torch.Tensor
    scaled_x: torch.Tensor = None


FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = 448.0
FLOAT32_MAX = torch.tensor(torch.finfo(torch.float32).max, dtype=torch.float32)
NVFP4_BLOCKSIZE = 16


def blockwise_quantize_nvfp4(x: torch.Tensor, tile_shape: list[int] = [1, NVFP4_BLOCKSIZE]):
    M, N = x.shape
    tile_y, tile_x = tile_shape

    # Step 1: Global amax
    global_amax = torch.amax(torch.abs(x))

    # Step 2: Global encode / decode scales
    global_encode_scale = torch.div(FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX, global_amax)
    global_encode_scale = torch.min(
        global_encode_scale,
        torch.tensor(
            torch.finfo(torch.float32).max,
            device=global_encode_scale.device,
            dtype=torch.float32,
        ),
    )
    if global_encode_scale == torch.tensor(0.0, device=x.device, dtype=torch.float32):
        global_encode_scale = torch.tensor(1.0, device=x.device, dtype=torch.float32)
    global_decode_scale = torch.div(1.0, global_encode_scale)

    # Step 3: Blockwise decode scales
    x_reshaped = x.view(M, N // tile_x, tile_x)
    blockwise_amax = torch.amax(torch.abs(x_reshaped), dim=-1, keepdim=True).to(torch.float32)
    decode_scales = torch.div(blockwise_amax, FLOAT4_E2M1_MAX)

    # Step 4: Encode blockwise scales to `E4M3`
    decode_scales = decode_scales * global_encode_scale
    decode_scales = torch.min(
        decode_scales,
        FLOAT32_MAX,
    )

    decode_scales = torch.clamp(decode_scales, min=-FLOAT8_E4M3_MAX, max=FLOAT8_E4M3_MAX)
    decode_scales = decode_scales.to(torch.float8_e4m3fn)

    encode_scale = torch.div(1.0, decode_scales.to(torch.float32) * global_decode_scale)
    encode_scale = torch.min(encode_scale, FLOAT32_MAX)

    # Quantize
    x = x.view(M, N // tile_x, tile_x)
    scaled_x = x.to(torch.float32) * encode_scale
    scaled_x = torch.clamp(scaled_x, -FLOAT4_E2M1_MAX, FLOAT4_E2M1_MAX).reshape(M, N)

    qx = round_to_nearest_even(scaled_x)

    return NVFP4QTensor(
        x=x, qx=qx, scaled_x=scaled_x, global_amax=global_amax, decode_scales=decode_scales
    )


def dequantize_fp4(
    qx: torch.Tensor,
    decode_scales: torch.Tensor,
    global_amax: float,
    tile_shape: list[int] = [1, NVFP4_BLOCKSIZE],
):
    """Dequantize from FP4 E2M1 -> FP32
    Args:
        qx: quantized tensor
        decode_scales: blockwise scale factors (FP8 E4M3)
        global_amax: float
    """
    tile_y, tile_x = tile_shape
    assert tile_y == 1
    (
        M,
        N,
    ) = qx.shape

    global_decode_scale = global_amax / (FLOAT4_E2M1_MAX * FLOAT8_E4M3_MAX)
    decoded_block_scales = decode_scales.to(torch.float32) * global_decode_scale
    qx = qx.view(M, N // tile_x, tile_x)
    assert qx.shape[0] == decoded_block_scales.shape[0], (
        f"Num rows mismatch: {qx.shape[0]} != {decoded_block_scales.shape[0]}"
    )
    assert qx.shape[1] == decoded_block_scales.shape[1], (
        f"Num scale factors mismatch: {qx.shape[1]} != {decoded_block_scales.shape[1]}"
    )

    if decoded_block_scales.ndim == 2:
        decoded_block_scales.unsqueeze_(-1)

    dq = qx * decoded_block_scales

    return dq.reshape(M, N)

# NOTE: Skip -0.0 to simplify stochastic rounding implementation    
FP4_E2M1_VALS = torch.tensor(
    [
        # Positives
        0.0,  # 0: 0000 
        0.5,  # 1: 0001 
        1.0,  # 2: 0010
        1.5,  # 3: 0011
        2.0,  # 4: 0100
        3.0,  # 5: 0101
        4.0,  # 6: 0110
        6.0,  # 7: 0111
        # Negatives - skip -0.0 to simplify stochastic rounding
        -0.0,  # 8: 1000  
        -0.5,  # 9: 1001 
        -1.0,  # 10: 1010
        -1.5,  # 11: 1011
        -2.0,  # 12: 1100
        -3.0,  # 13: 1101
        -4.0,  # 14: 1110
        -6.0,  # 15: 1111
    ],
    dtype=torch.float32,
)

def stochastic_round(x: torch.Tensor):
    dist = x.view(-1, 1).sub(FP4_E2M1_VALS.unsqueeze(0)).abs()
    nearest_dist, nearest_idx = dist.topk(k=2, dim=-1, largest=False)
    fp4_neighbors = FP4_E2M1_VALS[nearest_idx.view(-1).to(torch.long)].reshape(-1, 2)
    den = fp4_neighbors.diff().abs()
    den = torch.where(den == 0.0, 1.0, den)
    prob = (1 - (nearest_dist / den)).abs()
    select_idx = nearest_idx.gather(1, torch.multinomial(prob, 1).view(-1).to(torch.long).unsqueeze(-1))
    qx_sr = FP4_E2M1_VALS[select_idx.view(-1)].reshape(x.shape)
    return qx_sr

def round_to_nearest_even(x: torch.Tensor):
    """Round high precision tensor to nearest even E2M1 val"""

    result = torch.zeros_like(x, dtype=torch.float32)
    result[(x >= 0.0) & (x <= 0.25)] = 0.0
    result[(x > 0.25) & (x < 0.75)] = 0.5
    result[(x >= 0.75) & (x <= 1.25)] = 1.0
    result[(x > 1.25) & (x < 1.75)] = 1.5
    result[(x >= 1.75) & (x <= 2.5)] = 2.0
    result[(x > 2.5) & (x < 3.5)] = 3.0
    result[(x >= 3.5) & (x <= 5.0)] = 4.0
    result[x > 5.0] = 6.0

    result[(x >= -0.25) & (x < -0.0)] = -0.0
    result[(x < -0.25) & (x > -0.75)] = -0.5
    result[(x <= -0.75) & (x >= -1.25)] = -1.0
    result[(x < -1.25) & (x > -1.75)] = -1.5
    result[(x <= -1.75) & (x >= -2.5)] = -2.0
    result[(x < -2.5) & (x > -3.5)] = -3.0
    result[(x <= -3.5) & (x >= -5.0)] = -4.0
    result[x < -5.0] = -6.0

    return result


def quantize_fp4_ref(
    x: torch.Tensor,
    rowwise: bool = True,
    columnwise: bool = False,
    with_amax_reduction: bool = False,
    amax_reduction_group: int = None,
    with_rht: bool = False,
    with_post_rht_amax: bool = False,
    stochastic_rounding: bool = False,
    with_2d_quantization: bool = False,
):
    nvfp4_quantizer = NVFP4Quantizer(
        rowwise=rowwise,
        columnwise=columnwise,
        with_amax_reduction=with_amax_reduction,
        amax_reduction_group=amax_reduction_group,
        with_rht=with_rht,
        with_post_rht_amax=with_post_rht_amax,
        stochastic_rounding=stochastic_rounding,
        with_2d_quantization=with_2d_quantization,
    )
    x_nvfp4: NVFP4Tensor = nvfp4_quantizer(x.cuda())
    qx: torch.Tensor = x_nvfp4._rowwise_data.view(dtype=torch.uint8)
    sx: torch.Tensor = x_nvfp4._rowwise_scale_inv
    ref_amax = x_nvfp4._amax_rowwise

    return NVFP4QTensor(x=x, qx=qx, decode_scales=sx, global_amax=ref_amax)

x = torch.tensor(
    [0, 0.25, 0.5, 0.75, 1.25, 3.2, 4.5, 5.0, 0, 0.25, 0.5, 0.75, 1.25, 3.2, 4.5, 5.0]
).unsqueeze(0)

M, N = 128, 512
dtype = torch.float32
x = torch.randn((M, N), dtype=dtype) * 2 - 1

assert x.shape[1] % 16 == 0

# ref: NVFP4QTensor = quantize_fp4_ref(x)
q_nvfp4 = blockwise_quantize_nvfp4(x)
qx_rn = q_nvfp4.qx
decode_scales, global_amax = q_nvfp4.decode_scales, q_nvfp4.global_amax

dq_rn = dequantize_fp4(qx_rn, decode_scales, global_amax)
mse_rn = torch.sqrt(x.view(-1).sub(dq_rn.view(-1)).square().mean())

for n_iters in [1, 10, 50, 100, 1000]:
    total_sr = torch.zeros_like(dq_rn)

    for i in range(n_iters):
        qx_sr = stochastic_round(q_nvfp4.scaled_x)
        dq_sr = dequantize_fp4(qx_sr, decode_scales, global_amax)
        total_sr += dq_sr
    avg_dq_sr = total_sr / n_iters
    mse_sr = torch.sqrt(x.view(-1).sub(avg_dq_sr.view(-1)).square().mean())
    print(f"{n_iters=}")
    print(f"{mse_rn.item():.4f}")
    print(f"{mse_sr.item():.4f}")

breakpoint()
dq_test = dequantize_fp4(q_nvfp4.qx, q_nvfp4.decode_scales, q_nvfp4.global_amax)

# dq_ref = te_quantize.dequantize_fp4(ref.qx, ref.decode_scales, ref.global_amax)
# torch.testing.assert_close(dq_ref.to(dq_test.device).view(-1), dq_test.view(-1))
# for o, s, q, d in zip(x.view(-1), scaled_x.view(-1), qx.view(-1), dq.view(-1)):
#     print(f"{o:.2f} => {s:.2f} => {q:0.1f} => {d:0.2f}")
