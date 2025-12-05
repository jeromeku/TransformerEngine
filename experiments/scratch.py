import torch
from typing import List, TypeAlias, Tuple, Literal


def cast_to_fp4x2(x: torch.Tensor):
    """Quantize to E2M1 and pack into a byte tensor

    NOTE: Ties are rounded to the nearest even.
    """

    result = torch.zeros_like(x, dtype=torch.uint8)
    result[(x >= 0.0) & (x <= 0.25)] = 0
    result[(x > 0.25) & (x < 0.75)] = 1
    result[(x >= 0.75) & (x <= 1.25)] = 2
    result[(x > 1.25) & (x < 1.75)] = 3
    result[(x >= 1.75) & (x <= 2.5)] = 4
    result[(x > 2.5) & (x < 3.5)] = 5
    result[(x >= 3.5) & (x <= 5.0)] = 6
    result[x > 5.0] = 7

    result[(x >= -0.25) & (x < -0.0)] = 8
    result[(x < -0.25) & (x > -0.75)] = 9
    result[(x <= -0.75) & (x >= -1.25)] = 10
    result[(x < -1.25) & (x > -1.75)] = 11
    result[(x <= -1.75) & (x >= -2.5)] = 12
    result[(x < -2.5) & (x > -3.5)] = 13
    result[(x <= -3.5) & (x >= -5.0)] = 14
    result[x < -5.0] = 15

    return result[:, ::2] + result[:, 1::2] * 16


FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = 448.0


NVFP4_BLOCKSIZE = 16

NVFP4_1D_TILE = [NVFP4_BLOCKSIZE, 1]  # blockwise shape along reduction dim
NVFP4_2D_TILE = [
    NVFP4_BLOCKSIZE,
    NVFP4_BLOCKSIZE,
]  # symmetric quant for weight tensors and chain-rule consistency
NVFP4_TILESHAPE = NVFP4_1D_TILE | NVFP4_2D_TILE

def blockwise_quantize_nvfp4(x: torch.Tensor, tile_shape: "NVFP4_TILESHAPE"):
    M, N = x.shape
    assert tile_shape in [NVFP4_1D_TILE, NVFP4_2D_TILE]
    tile_len_x, tile_len_y = tile_shape

    using_2d_quantization = tile_shape == NVFP4_2D_TILE

    global_amax = torch.amax(torch.abs(x))

    if using_2d_quantization:
        x_blocks = (
            x.unfold(0, tile_len_y, tile_len_y).unfold(1, tile_len_x, tile_len_x).to(torch.float32)
        )
        block_amax = torch.amax(torch.abs(x_blocks), dim=(-1, -2))
        vec_max = block_amax.repeat_interleave(tile_len_y, dim=0).unsqueeze(-1)
    else:
        x_reshaped = x.view(M, N // tile_len_x, tile_len_x)
        vec_max = torch.amax(torch.abs(x_reshaped), dim=-1, keepdim=True).to(
            torch.float32
        )

    x = x.view(M, N // tile_len_x, tile_len_x)
    decode_scale = torch.div(vec_max, FLOAT4_E2M1_MAX)

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

    decode_scale = decode_scale * global_encode_scale
    decode_scale = torch.min(
        decode_scale,
        torch.tensor(
            torch.finfo(torch.float32).max,
            device=decode_scale.device,
            dtype=torch.float32,
        ),
    )
    decode_scale = torch.clamp(decode_scale, min=-FLOAT8_E4M3_MAX, max=FLOAT8_E4M3_MAX)
    decode_scale = decode_scale.to(torch.float8_e4m3fn)

    encode_scale = torch.min(
        torch.div(1.0, decode_scale.to(torch.float32) * global_decode_scale),
        torch.tensor(
            torch.finfo(torch.float32).max,
            device=decode_scale.device,
            dtype=torch.float32,
        ),
    )

    scaled_x = x.to(torch.float32) * encode_scale

    clipped_x = torch.clamp(scaled_x, -FLOAT4_E2M1_MAX, FLOAT4_E2M1_MAX).reshape(m, n)

    qx = cast_to_fp4x2(clipped_x)
    sx = decode_scale.squeeze(-1)

    return qx, sx
