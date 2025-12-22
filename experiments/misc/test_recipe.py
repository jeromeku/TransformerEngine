
from typing import Optional

import pytest
import torch
import warnings

import transformer_engine.common.recipe
import transformer_engine.pytorch as te
from transformer_engine.pytorch import (
    Float8BlockQuantizer,
    MXFP8Quantizer,
    Float8Quantizer,
    NVFP4Quantizer,
    quantized_model_init,
    Linear,
    LayerNormLinear,
    LayerNormMLP,
    GroupedLinear,
)

import transformer_engine_torch as tex
from transformer_engine.pytorch.quantization import (
    FP8GlobalStateManager,
    _amax_and_scale_update,
)
import transformer_engine.pytorch.ops as te_ops
from transformer_engine.common.recipe import DelayedScaling, Float8BlockScaling, MXFP8BlockScaling, NVFP4BlockScaling
import transformer_engine_torch as tex

# Check if FP8 is supported
fp8_available, reason_for_no_fp8 = te.is_fp8_available(return_reason=True)
mxfp8_available, reason_for_no_mxfp8 = te.is_mxfp8_available(return_reason=True)
fp8_block_scaling_available, reason_for_no_fp8_block_scaling = te.is_fp8_block_scaling_available(
    return_reason=True
)
fp4_available, reason_for_no_fp4 = te.is_nvfp4_available(return_reason=True)


def test_check_for_weight_tensor_and_recipe_correspondence(M=32, N=32, model_init_recipe=MXFP8BlockScaling(),
):
    with quantized_model_init(enabled=True, recipe=model_init_recipe):
        linear = Linear(M, N).cuda()

test_check_for_weight_tensor_and_recipe_correspondence(1024, 4096)
