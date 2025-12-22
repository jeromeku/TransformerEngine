"""
PyTorch reference implementations for NVFP4-style quantization
with and without stochastic rounding.

These functions are intentionally simple and are meant to illustrate:
  - How per-block scaling is computed
  - How FP4 E2M1 values can be encoded/decoded in Python
  - How stochastic rounding can be implemented by randomly
    choosing between the two nearest representable FP4 values.

They are closely related to NVFP4QuantizerRef in
transformer_engine.pytorch.custom_recipes.quantization_nvfp4, but
add a stochastic rounding path.
"""

from __future__ import annotations

from typing import Tuple

import torch

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = 448.0


# Same table as cast_from_fp4x2 in quantization_nvfp4.py
FP4_VALUES = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def _compute_block_scales(
    x: torch.Tensor, tile_len_x: int, tile_len_y: int, global_amax: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-block encode/decode scales (Python analogue of _quantize_blockwise_reference)."""
    assert x.ndim == 2
    m, n = x.shape
    using_2d_quantization = tile_len_x == 16 and tile_len_y == 16

    if using_2d_quantization:
        # x: (128,128) → (8,8,16,16)
        x_blocks = (
            x.unfold(0, tile_len_y, tile_len_y)
            .unfold(1, tile_len_x, tile_len_x)
            .to(torch.float32)
        )  # (8,8,16,16)
        block_amax = torch.amax(torch.abs(x_blocks), dim=(-1, -2))  # (8,8)
        vec_max = block_amax.repeat_interleave(tile_len_y, dim=0).unsqueeze(-1)  # (128,8,1)
    else:
        x_reshaped = x.view(m, n // tile_len_x, tile_len_x)  # (m, n/tile_len_x, tile_len_x)
        vec_max = torch.amax(torch.abs(x_reshaped), dim=-1, keepdim=True).to(torch.float32)

    FLOAT4 = torch.tensor(FLOAT4_E2M1_MAX, device=x.device, dtype=torch.float32)
    FLOAT8 = torch.tensor(FLOAT8_E4M3_MAX, device=x.device, dtype=torch.float32)

    decode_scale = vec_max / FLOAT4

    global_encode_scale = FLOAT8 * FLOAT4 / global_amax
    global_encode_scale = torch.min(
        global_encode_scale,
        torch.tensor(torch.finfo(torch.float32).max, device=x.device, dtype=torch.float32),
    )
    global_encode_scale = torch.where(
        global_encode_scale == 0.0,
        torch.tensor(1.0, device=x.device, dtype=torch.float32),
        global_encode_scale,
    )
    global_decode_scale = 1.0 / global_encode_scale

    # Encode decode_scale into FP8 E4M3
    decode_scale = decode_scale * global_encode_scale
    decode_scale = torch.min(
        decode_scale,
        torch.tensor(torch.finfo(torch.float32).max, device=x.device, dtype=torch.float32),
    )
    decode_scale = torch.clamp(decode_scale, min=-FLOAT8, max=FLOAT8)
    decode_scale_fp8 = decode_scale.to(torch.float8_e4m3fn)

    encode_scale = torch.min(
        1.0 / (decode_scale_fp8.to(torch.float32) * global_decode_scale),
        torch.tensor(torch.finfo(torch.float32).max, device=x.device, dtype=torch.float32),
    )

    return encode_scale, decode_scale_fp8.squeeze(-1)


def _deterministic_fp4_codes(scaled_x: torch.Tensor) -> torch.Tensor:
    """Deterministic FP4 coding: nearest neighbor in FP4_VALUES."""
    # scaled_x: (m, n)
    device = scaled_x.device
    vals = FP4_VALUES.to(device)
    # Broadcast over last dim
    diff = torch.abs(scaled_x.unsqueeze(-1) - vals.view(1, 1, -1))
    codes = diff.argmin(dim=-1).to(torch.uint8)  # (m,n)
    return codes


def _stochastic_fp4_codes(
    scaled_x: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Stochastic FP4 coding: randomly choose between the two nearest FP4 values."""
    device = scaled_x.device
    vals = FP4_VALUES.to(device)

    # Compute absolute differences to all 16 values
    diff = torch.abs(scaled_x.unsqueeze(-1) - vals.view(1, 1, -1))  # (m,n,16)
    # Indices of two smallest entries along last dim
    two_best = torch.topk(diff, k=2, dim=-1, largest=False).indices  # (m,n,2)
    idx_lo = torch.min(two_best[..., 0], two_best[..., 1])
    idx_hi = torch.max(two_best[..., 0], two_best[..., 1])

    v_lo = vals[idx_lo]
    v_hi = vals[idx_hi]

    # Avoid division by zero if v_lo == v_hi
    denom = (v_hi - v_lo).abs()
    denom = torch.where(denom == 0.0, torch.ones_like(denom), denom)

    t = (scaled_x - v_lo) / denom
    t = torch.clamp(t, 0.0, 1.0)

    # Draw uniform random numbers
    if generator is None:
        r = torch.rand_like(t)
    else:
        r = torch.rand_like(t, generator=generator)

    choose_hi = (r < t)
    codes = torch.where(choose_hi, idx_hi, idx_lo).to(torch.uint8)
    return codes


def nvfp4_quantize_reference(
    x: torch.Tensor,
    *,
    tile_len_x: int = 16,
    tile_len_y: int = 16,
    stochastic: bool = False,
    generator: torch.Generator | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reference NVFP4-style quantization in PyTorch.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor, shape (M, N).
    tile_len_x, tile_len_y : int
        Quantization tile dimensions. (16,16) mimics the NVFP4 2D weight
        quantization, (1,16) mimics rowwise activation/gradient tiles.
    stochastic : bool
        If True, uses stochastic rounding; else deterministic nearest neighbor.
    generator : torch.Generator, optional
        Optional PyTorch RNG generator used for stochastic rounding.

    Returns
    -------
    q_codes : torch.ByteTensor
        FP4 codes (0..15) for each element (shape (M,N)).
    decode_scale_fp8 : torch.Tensor
        FP8 E4M3 decode scales per tile, as in NVFP4.
    global_amax : torch.Tensor
        Global amax used to compute the second-stage scaling.
    """
    assert x.dim() == 2, "Only 2D tensors are supported in this reference."
    x = x.to(torch.float32)

    # Global amax for second-stage scaling
    global_amax = torch.max(torch.abs(x)).to(torch.float32).view(1)

    encode_scale, decode_scale_fp8 = _compute_block_scales(
        x, tile_len_x=tile_len_x, tile_len_y=tile_len_y, global_amax=global_amax
    )

    m, n = x.shape
    # Reshape into tiles of length tile_len_x
    x_blocks = x.view(m, n // tile_len_x, tile_len_x)
    scaled_x = x_blocks * encode_scale  # broadcasted over blocks

    # Clamp to FP4 range and reshape back
    scaled_x = torch.clamp(scaled_x, -FLOAT4_E2M1_MAX, FLOAT4_E2M1_MAX).view(m, n)

    if stochastic:
        q_codes = _stochastic_fp4_codes(scaled_x, generator=generator)
    else:
        q_codes = _deterministic_fp4_codes(scaled_x)

    return q_codes, decode_scale_fp8, global_amax

