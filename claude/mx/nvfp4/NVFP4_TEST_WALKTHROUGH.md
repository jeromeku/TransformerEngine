# NVFP4 Test Case Walkthroughs

This document provides annotated walkthroughs of NVFP4 test cases, tracing the complete call path from user-facing APIs down to kernel execution.

## Table of Contents

1. [Test File Organization](#test-file-organization)
2. [Basic Module Test Walkthrough](#basic-module-test-walkthrough)
3. [RHT Quantization Test](#rht-quantization-test)
4. [2D Quantization Test](#2d-quantization-test)
5. [GEMM Test Walkthrough](#gemm-test-walkthrough)

---

## Test File Organization

### NVFP4 Test Files

**Directory:** `/home/jeromeku/transformerengine/tests/pytorch/nvfp4/`

```
nvfp4/
├── test_nvfp4_module_exact.py          # ◄─ Main module test
│   - Tests te.Linear with NVFP4 recipe
│   - Compares against reference implementation
│   - Tests different quantization modes
│
├── test_nvfp4_gemm_exact.py            # ◄─ Direct GEMM tests
│   - Tests FP4 GEMM kernel execution
│   - Validates scaling factor handling
│
├── test_nvfp4_quantize_exact.py        # ◄─ Quantization kernel tests
│   - Tests quantization to FP4
│   - Tests dequantization
│   - Tests scaling factor computation
│
├── test_nvfp4_rht_quantize_exact.py    # ◄─ RHT-specific tests
│   - Tests random Hadamard transform
│   - Tests RHT + quantization pipeline
│
└── test_nvfp4_sr_quantize.py           # ◄─ Stochastic rounding tests
    - Tests stochastic rounding behavior
    - Tests gradient quantization
```

---

## Basic Module Test Walkthrough

### Test File: `test_nvfp4_module_exact.py`

**Lines 67-114: Factory Function Setup**

```python
def get_nvfp4_quantizer_factory(with_rht: bool = False, with_2d_quantization: bool = False):
    """
    Create a quantizer factory for NVFP4 reference implementation.
    
    This factory returns NVFP4QuantizerRef instances based on the role and configuration.
    Used with CustomRecipe to create reference quantizers.
    
    Args:
        with_rht: Whether to enable random Hadamard transform
        with_2d_quantization: Whether to use 2D quantization (16x16 tiles for weights)
    
    Returns:
        A factory function that takes a role string and returns a quantizer instance
    """
    
    def factory(role):
        if role == "linear_input":
            return quantization_nvfp4.NVFP4QuantizerRef(
                dtype=utils.Fp4Formats.E2M1,
                quant_tile_shape=(1, 16),           # ◄─ 1D quantization (block size 16)
                pow_2_scales=False,
                with_rht=with_rht,                  # ◄─ Random Hadamard transform
            )
        elif role == "linear_weight":
            return quantization_nvfp4.NVFP4QuantizerRef(
                dtype=utils.Fp4Formats.E2M1,
                quant_tile_shape=(16, 16) if with_2d_quantization else (1, 16),
                                                    # ◄─ 2D (16x16) or 1D (1x16)
                pow_2_scales=False,
                with_rht=False,                     # ◄─ No RHT for weights
            )
        elif role == "linear_output":
            return None                             # ◄─ Output quantization not used
        elif role == "linear_grad_output":
            return quantization_nvfp4.NVFP4QuantizerRef(
                dtype=utils.Fp4Formats.E2M1,
                quant_tile_shape=(1, 16),
                pow_2_scales=False,
                with_rht=with_rht,
            )
        elif role == "linear_grad_input":
            return None                             # ◄─ Grad input quantization not used
        else:
            return None
    
    return factory
```

### Complete Test Flow: `check_nvfp4_module_versus_reference()`

**Lines 123-200+: Test Execution**

```python
def check_nvfp4_module_versus_reference(
    module_class,
    in_features: int,
    out_features: int,
    bias: bool,
    x_dtype: torch.dtype,
    num_steps: int = 1,
    with_rht: bool = False,
    with_2d_quantization: bool = False,
):
    """
    Compare native NVFP4 module against reference implementation.
    
    Call Path:
    
    check_nvfp4_module_versus_reference()
    ├─ Setup: Create two identical modules (native vs reference)
    ├─ Quantization Recipe Setup
    ├─ Forward Pass (num_steps iterations)
    │  ├─ with te.autocast(enabled=True, recipe=NVFP4BlockScaling())
    │  │  │
    │  │  ├─ FP8GlobalStateManager.autocast_enter()
    │  │  │  ├─ FP8_ENABLED = True
    │  │  │  ├─ FP8_RECIPE = NVFP4BlockScaling()
    │  │  │  └─ check_nvfp4_support() ✓
    │  │  │
    │  │  ├─ Native Module Forward:
    │  │  │  └─ te.Linear.forward()
    │  │  │     ├─ Query: FP8GlobalStateManager.is_fp8_enabled() → True
    │  │  │     ├─ Recipe: FP8GlobalStateManager.get_fp8_recipe()
    │  │  │     │   → NVFP4BlockScaling()
    │  │  │     ├─ RecipeState.create(recipe, mode="forward", num_quantizers=3)
    │  │  │     │   → NVFP4BlockScalingRecipeState
    │  │  │     ├─ .make_quantizers() → [input_quant, weight_quant, output_quant]
    │  │  │     │
    │  │  │     ├─ _Linear.forward():
    │  │  │     │  ├─ input_quant(input)
    │  │  │     │  │  ├─ Apply RHT if enabled
    │  │  │     │  │  ├─ Compute amax per block (16 consecutive values)
    │  │  │     │  │  ├─ Compute E4M3 block scales
    │  │  │     │  │  ├─ Compute FP32 global scale
    │  │  │     │  │  ├─ Quantize to FP4 E2M1
    │  │  │     │  │  └─ Return QuantizedTensor storage
    │  │  │     │  │
    │  │  │     │  ├─ weight_quant(weight)
    │  │  │     │  │  ├─ Apply 2D quantization if enabled
    │  │  │     │  │  │  (16x16 tiles instead of 1x16 blocks)
    │  │  │     │  │  ├─ Compute amax per tile
    │  │  │     │  │  ├─ Compute E4M3 scales per tile
    │  │  │     │  │  ├─ Compute FP32 global scale
    │  │  │     │  │  └─ Return QuantizedTensor storage
    │  │  │     │  │
    │  │  │     │  ├─ GEMM: quantized_input @ quantized_weight
    │  │  │     │  │  ├─ Input: FP4 + E4M3 block scales + FP32 global scale
    │  │  │     │  │  ├─ Weight: FP4 + E4M3 block scales + FP32 global scale
    │  │  │     │  │  ├─ Kernel execution (cuBLASLt with FP4 tensor cores)
    │  │  │     │  │  │  - Dequantize on-the-fly using scales
    │  │  │     │  │  │  - Accumulate in FP32 or TF32
    │  │  │     │  │  │  - Output: FP32 then cast to BF16
    │  │  │     │  │  └─ Return dequantized output
    │  │  │     │  │
    │  │  │     │  ├─ Add bias if present
    │  │  │     │  ├─ Cast to output dtype (BF16)
    │  │  │     │  └─ Save for backward:
    │  │  │     │     ├─ input_quantizer
    │  │  │     │     ├─ weight
    │  │  │     │     └─ quantization metadata
    │  │  │     │
    │  │  │     └─ Return output (native_output)
    │  │  │
    │  │  ├─ Reference Module Forward:
    │  │  │  └─ (With reference NVFP4QuantizerRef instances)
    │  │  │     Same sequence but using reference quantizer
    │  │  │     → reference_output
    │  │  │
    │  │  └─ FP8GlobalStateManager.autocast_exit()
    │  │     └─ Restore previous state
    │  │
    │  ├─ Validation:
    │  │  ├─ torch.allclose(native_output, reference_output, rtol=1e-2, atol=1e-3)
    │  │  └─ Assert outputs match within tolerance
    │  │
    │  └─ Backward Pass (loss.backward()):
    │     ├─ Compute loss = output.sum()
    │     ├─ Call loss.backward()
    │     │
    │     ├─ _Linear.backward():
    │     │  ├─ grad_output from upstream
    │     │  │
    │     │  ├─ Create backward quantizers:
    │     │  │  RecipeState.create(recipe, mode="backward", num_quantizers=2)
    │     │  │  → [grad_output_quantizer, grad_input_quantizer]
    │     │  │
    │     │  ├─ grad_output_quant(grad_output):
    │     │  │  ├─ Apply RHT if enabled
    │     │  │  ├─ Apply stochastic rounding
    │     │  │  ├─ Compute amax per block
    │     │  │  ├─ Compute E4M3 block scales
    │     │  │  ├─ Compute FP32 global scale
    │     │  │  └─ Quantize to FP4 E2M1
    │     │  │
    │     │  ├─ dgrad GEMM:
    │     │  │  grad_output (FP4) @ weight.T (FP4)
    │     │  │  → grad_input (FP32 → BF16)
    │     │  │
    │     │  ├─ wgrad GEMM:
    │     │  │  input.T (FP4) @ grad_output (FP4)
    │     │  │  → grad_weight (FP32 → BF16)
    │     │  │
    │     │  └─ Return grad_input, grad_weight
    │     │
    │     └─ Update parameters using gradients
    │
    └─ Assertions:
       ├─ Check forward pass output correctness
       ├─ Check backward pass gradients correctness
       └─ Verify numerical stability
```

---

## RHT Quantization Test

### Test Function: `test_nvfp4_rht_quantization()`

**File:** `test_nvfp4_rht_quantize_exact.py`

#### Data Flow with RHT

```python
# Create recipe with RHT enabled
nvfp4_recipe = recipe.NVFP4BlockScaling()
# Internally sets:
# nvfp4_recipe.fp4_quant_fwd_inp.random_hadamard_transform = True
# nvfp4_recipe.fp4_quant_bwd_grad.random_hadamard_transform = True
# nvfp4_recipe.fp4_quant_fwd_weight.random_hadamard_transform = False

with te.autocast(enabled=True, recipe=nvfp4_recipe):
    output = module(input)
```

#### Forward Pass with RHT

```
INPUT TENSOR (BF16)
  │
  ├─► Input Quantizer with RHT enabled:
  │    │
  │    ├─ Step 1: Apply Random Hadamard Transform (RHT)
  │    │  │
  │    │  ├─ Transform shape: 16x16 random Hadamard matrix
  │    │  ├─ Purpose: Smooth out outliers in the distribution
  │    │  │  (makes distribution more uniform for better FP4 quantization)
  │    │  │
  │    │  ├─ Kernel execution:
  │    │  │  transformer_engine/pytorch/experimental/quantization_nvfp4.py
  │    │  │  CUDA kernel: apply_random_hadamard_transform()
  │    │  │
  │    │  └─ Output: Transformed tensor (values more uniformly distributed)
  │    │
  │    ├─ Step 2: Compute amax per block
  │    │  │
  │    │  ├─ Block size: 16 consecutive values (1D quantization)
  │    │  ├─ For each block of 16 consecutive values:
  │    │  │  ├─ amax[i] = max(abs(transformed_tensor[i*16:(i+1)*16]))
  │    │  │  └─ Store in amax tensor
  │    │  │
  │    │  └─ Output: amax tensor (shape: [num_blocks])
  │    │
  │    ├─ Step 3: Compute E4M3 block scaling factors
  │    │  │
  │    │  ├─ For each block i:
  │    │  │  ├─ E4M3_max = 6.0 (max value representable in E4M3)
  │    │  │  ├─ block_scale[i] = E4M3_max / amax[i]
  │    │  │  └─ Clamp to [2^-128, 2^127] (E4M3 exponent range)
  │    │  │
  │    │  └─ Output: block_scale tensor (dtype: E4M3, shape: [num_blocks])
  │    │
  │    ├─ Step 4: Compute FP32 global scaling factor
  │    │  │
  │    │  ├─ Combine all block scales:
  │    │  │  global_scale = compute_global_scale(block_scales)
  │    │  │
  │    │  └─ Output: global_scale (dtype: FP32, shape: [1])
  │    │
  │    ├─ Step 5: Quantize to FP4 E2M1
  │    │  │
  │    │  ├─ For each value v in transformed_tensor:
  │    │  │  ├─ block_idx = i // 16
  │    │  │  ├─ scaled_v = v * block_scale[block_idx]
  │    │  │  ├─ scaled_v = scaled_v * global_scale
  │    │  │  ├─ fp4_v = round_to_nearest_fp4(scaled_v)
  │    │  │  └─ Clamp to valid FP4 E2M1 range
  │    │  │
  │    │  └─ Output: quantized tensor (dtype: FP4 E2M1, shape: same as input)
  │    │
  │    └─ Return: QuantizedTensor{
  │         quantized_data=fp4_tensor,
  │         block_scales=block_scale,  # E4M3
  │         global_scale=global_scale, # FP32
  │         metadata={'with_rht': True, ...}
  │       }
  │
  └─► Next stage (weight quantization or GEMM)
```

#### Backward Pass with RHT + Stochastic Rounding

```
GRAD_OUTPUT TENSOR (BF16)
  │
  ├─► Grad Output Quantizer:
  │    │
  │    ├─ Step 1: Apply Random Hadamard Transform (RHT)
  │    │  │  (Same as forward)
  │    │  └─ Output: Transformed grad_output
  │    │
  │    ├─ Step 2: Apply Stochastic Rounding
  │    │  │
  │    │  ├─ For each value v in transformed_grad:
  │    │  │  ├─ Compute target FP4 values: lower, upper (two nearest FP4 values)
  │    │  │  ├─ Compute distance to each: d_lower, d_upper
  │    │  │  ├─ Probability of rounding up: p = d_lower / (d_lower + d_upper)
  │    │  │  ├─ Random sample r ~ Uniform(0, 1)
  │    │  │  ├─ If r < p: round to upper, else round to lower
  │    │  │  └─ Benefits:
  │    │  │     - Avoids bias toward lower or upper values
  │    │  │     - Better numerical properties for gradient accumulation
  │    │  │
  │    │  └─ Output: Stochastically rounded tensor
  │    │
  │    ├─ Step 3: Compute amax per block
  │    │  │  (Same as forward)
  │    │  └─ Output: amax tensor
  │    │
  │    ├─ Step 4: Compute scaling factors (E4M3 + FP32)
  │    │  │  (Same as forward)
  │    │  └─ Output: block_scale, global_scale
  │    │
  │    └─ Return: QuantizedTensor with stochastically rounded data
  │
  └─ GEMM: grad_output (FP4 + scales) @ weight.T (FP4 + scales)
     → grad_input (dequantized)
```

---

## 2D Quantization Test

### Test Function: `test_nvfp4_2d_quantization()`

#### Weight Quantization with 2D Blocks

**Key Difference:** Weights use 16x16 tile-based quantization instead of 1D blocks

```
WEIGHT TENSOR (BF16, shape: [out_features, in_features])
  │
  ├─► Weight Quantizer with 2D quantization:
  │    │
  │    ├─ Step 1: Divide tensor into 16x16 tiles
  │    │  │
  │    │  ├─ For a weight matrix of shape [1024, 1024]:
  │    │  │  ├─ Tile 1: weight[0:16, 0:16]
  │    │  │  ├─ Tile 2: weight[0:16, 16:32]
  │    │  │  ├─ ...
  │    │  │  ├─ Tile N: weight[(1024//16-1)*16:, (1024//16-1)*16:]
  │    │  │  │
  │    │  │  └─ Total tiles: (1024/16) * (1024/16) = 64 * 64 = 4096 tiles
  │    │  │
  │    │  └─ Output: Tile index mapping
  │    │
  │    ├─ Step 2: Compute amax per tile
  │    │  │
  │    │  ├─ For each 16x16 tile:
  │    │  │  ├─ amax[tile_idx] = max(abs(weight[tile_row_start:row_end, 
  │    │  │  │                           tile_col_start:col_end]))
  │    │  │  └─ This captures the maximum value in each tile
  │    │  │
  │    │  └─ Output: amax tensor (shape: [num_tile_rows, num_tile_cols])
  │    │
  │    ├─ Step 3: Compute E4M3 tile scaling factors
  │    │  │
  │    │  ├─ For each tile i,j:
  │    │  │  ├─ tile_scale[i,j] = E4M3_max / amax[i,j]
  │    │  │  └─ All values in tile use same scale
  │    │  │
  │    │  └─ Output: tile_scale tensor (dtype: E4M3, shape: [num_tile_rows, num_tile_cols])
  │    │
  │    ├─ Step 4: Compute FP32 global scaling factor
  │    │  │
  │    │  ├─ global_scale = compute_global_scale(all_tile_scales)
  │    │  │
  │    │  └─ Output: global_scale (dtype: FP32)
  │    │
  │    └─ Step 5: Quantize to FP4 E2M1 per tile
  │         │
  │         ├─ For each value v in tile (i, j):
  │         │  ├─ scaled_v = v * tile_scale[i,j]
  │         │  ├─ scaled_v = scaled_v * global_scale
  │         │  └─ fp4_v = round_to_nearest_fp4(scaled_v)
  │         │
  │         └─ Output: quantized weight (dtype: FP4 E2M1)
  │
  └─ Return: QuantizedTensor{
       quantized_data=fp4_weight,
       tile_scales=tile_scale,    # E4M3 (2D)
       global_scale=global_scale, # FP32
       metadata={'tile_shape': (16, 16), 'num_tiles': ...}
     }
```

#### GEMM with 2D Quantized Weights

```
FORWARD PASS:
  quantized_input (FP4, 1D quantization) @ quantized_weight (FP4, 2D quantization)
  │
  ├─ Dequantization in Kernel:
  │  │
  │  ├─ For input (1D quantization):
  │  │  ├─ dequant_input[i,j] = fp4_input[i,j] * block_scale[j//16] * global_scale_input
  │  │  └─ Block scale depends only on column index
  │  │
  │  ├─ For weight (2D quantization):
  │  │  ├─ tile_row = i // 16, tile_col = j // 16
  │  │  ├─ dequant_weight[i,j] = fp4_weight[i,j] 
  │  │  │                          * tile_scale[tile_row, tile_col]
  │  │  │                          * global_scale_weight
  │  │  └─ Scale depends on 2D tile location
  │  │
  │  └─ Output: FP32 values ready for GEMM
  │
  ├─ Tensor Core GEMM:
  │  ├─ Input dimensions: [M, K] @ [K, N] → [M, N]
  │  ├─ Tensor Core: 16x16 unit (matches quantization block size)
  │  ├─ Accumulation: FP32
  │  └─ Result: FP32 output
  │
  └─ Dequantization Output:
     ├─ Scale by scales
     ├─ Cast to output dtype (BF16)
     └─ Return final output
```

---

## GEMM Test Walkthrough

### Test File: `test_nvfp4_gemm_exact.py`

#### GEMM Kernel Execution Path

**File:** `/home/jeromeku/transformerengine/tests/pytorch/nvfp4/test_nvfp4_gemm_exact.py`

```python
# Test setup
m, n, k = 128, 256, 512
input_tensor = torch.randn(m, k, dtype=torch.bfloat16, device='cuda')
weight_tensor = torch.randn(n, k, dtype=torch.bfloat16, device='cuda')

# Create quantizers
input_quantizer = NVFP4Quantizer(
    fp4_dtype=tex.DType.kFloat4E2M1,
    rowwise=True,
    columnwise=True,
    with_rht=True,           # ◄─ Random Hadamard transform
    with_post_rht_amax=True,
    with_2d_quantization=False,  # ◄─ 1D for input
    stochastic_rounding=False
)

weight_quantizer = NVFP4Quantizer(
    fp4_dtype=tex.DType.kFloat4E2M1,
    rowwise=True,
    columnwise=True,
    with_rht=False,
    with_post_rht_amax=False,
    with_2d_quantization=True,  # ◄─ 2D for weight
    stochastic_rounding=False
)

with te.autocast(enabled=True, recipe=NVFP4BlockScaling()):
    # Call Path:
    # 1. Quantize input
    quantized_input = input_quantizer(input_tensor)
    #    ├─ Apply RHT
    #    ├─ Compute amax per block (1D: 512/16 = 32 blocks)
    #    ├─ Compute block scales (E4M3)
    #    ├─ Compute global scale (FP32)
    #    └─ Return QuantizedTensor
    
    # 2. Quantize weight
    quantized_weight = weight_quantizer(weight_tensor)
    #    ├─ NO RHT
    #    ├─ Apply 2D quantization
    #    ├─ Compute amax per 16x16 tile
    #    │  Tiles: (256/16) x (512/16) = 16 x 32 = 512 tiles
    #    ├─ Compute tile scales (E4M3)
    #    ├─ Compute global scale (FP32)
    #    └─ Return QuantizedTensor with 2D structure
    
    # 3. GEMM
    output = tex.general_gemm(
        quantized_input,
        quantized_weight,
        m=m, n=n, k=k,
        # ... other parameters
    )
    #    Call Path:
    #    tex.general_gemm()
    #    ├─ Detect FP4 + FP4 types
    #    │
    #    ├─ cuBLASLt kernel dispatch
    #    │  ├─ Create descriptors for FP4 input
    #    │  ├─ Create descriptors for FP4 weight
    #    │  ├─ Create descriptors for FP32 output
    #    │  └─ Configure scaling/dequantization callbacks
    #    │
    #    ├─ Kernel execution:
    #    │  ├─ Load FP4 input block (16 values → dequant using 1D scales)
    #    │  ├─ Load FP4 weight block (16x16 tile → dequant using 2D scales)
    #    │  ├─ Multiply dequantized values in FP32
    #    │  ├─ Accumulate to C_temp (FP32)
    #    │  ├─ Repeat for all tiles
    #    │  ├─ Post-process output (scale by output descriptor)
    #    │  └─ Store to global memory as FP32
    #    │
    #    ├─ Return FP32 output tensor
    #    │  (shape: [m, n] = [128, 256])
    #    │  (dtype: FP32 before casting to output format)
    #    │
    #    └─ Cast to BF16 if needed
```

#### GEMM Kernel Computational Flow

```
Tensor Core Execution (per 16x16 block):
│
├─ Input preparation (16x K):
│  ├─ Load FP4 input block
│  ├─ Load input block scales (amax computed before GEMM)
│  ├─ Load global input scale
│  ├─ In-kernel dequantization:
│  │  for each element (i,j) in block:
│  │      block_idx = j // 16
│  │      inp_dequant[i,j] = fp4_input[i,j] 
│  │                          * inp_block_scale[block_idx]
│  │                          * inp_global_scale
│  └─ Output: FP32 input block
│
├─ Weight preparation (K x 16):
│  ├─ Load FP4 weight block (may span multiple tiles)
│  ├─ Load weight tile scales (2D: 32x32 for K=512, N=256)
│  ├─ Load global weight scale
│  ├─ In-kernel dequantization:
│  │  for each element (i,j) in block:
│  │      tile_row = i // 16
│  │      tile_col = j // 16
│  │      weight_dequant[i,j] = fp4_weight[i,j]
│  │                             * weight_tile_scale[tile_row, tile_col]
│  │                             * weight_global_scale
│  └─ Output: FP32 weight block
│
├─ Matrix Multiply (using Tensor Core):
│  ├─ Input FP32 block: [16, K]
│  ├─ Weight FP32 block: [K, 16]
│  ├─ Accumulator: FP32, shape [16, 16]
│  │
│  ├─ For all K:
│  │  accumulator += matmul_16x16xK(input_fp32, weight_fp32)
│  │  (Uses Tensor Core hardware for efficient computation)
│  │
│  └─ Output: FP32 accumulator [16, 16]
│
└─ Output preparation:
   ├─ Accumulator values in FP32
   ├─ Store to global memory
   ├─ Cast to BF16 if final output requires it
   └─ Final output: [m, n] = [128, 256]
```

---

## Test Validation Mechanisms

### Forward Pass Validation

```python
with te.autocast(enabled=True, recipe=nvfp4_recipe):
    native_output = native_module(input)
    
reference_output = reference_module(input)  # Outside autocast for comparison

# Numerical validation
assert torch.allclose(
    native_output,
    reference_output,
    rtol=1e-2,  # Relative tolerance: 1%
    atol=1e-3,  # Absolute tolerance: 0.001
    equal_nan=False
), f"Output mismatch: max diff = {(native_output - reference_output).abs().max()}"
```

**Tolerance Rationale:**
- NVFP4 uses only 4 bits for mantissa
- E2M1 format: 2 bits exponent, 1 bit mantissa
- Quantization introduces ~1-2% error
- Tolerance of 1% RTol + 1e-3 ATol covers:
  - Quantization error from FP4 representation
  - Block scaling differences (1D vs 2D)
  - RHT approximation error (if enabled)
  - Stochastic rounding variance (if enabled)

### Backward Pass Validation

```python
# Compute loss and backward
loss_native = native_output.sum()
loss_native.backward()

loss_reference = reference_output.sum()
loss_reference.backward()

# Gradient validation
for (n_param, r_param) in zip(native_module.parameters(), 
                               reference_module.parameters()):
    if n_param.grad is not None:
        assert torch.allclose(
            n_param.grad,
            r_param.grad,
            rtol=1e-2,
            atol=1e-3
        ), f"Grad mismatch for {n_param.shape}"
```

---

## Environment Variable Controls

### NVFP4 Configuration via Environment Variables

```bash
# Control RHT (Random Hadamard Transform)
export NVTE_NVFP4_DISABLE_RHT=0    # Enable RHT (default)
export NVTE_NVFP4_DISABLE_RHT=1    # Disable RHT (faster but less accurate)

# Control stochastic rounding
export NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING=0  # Enable (default)
export NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING=1  # Disable

# Control 2D quantization
export NVTE_NVFP4_DISABLE_2D_QUANTIZATION=0  # Enable 2D for weights (default)
export NVTE_NVFP4_DISABLE_2D_QUANTIZATION=1  # Use 1D for all tensors
```

### Effect on Recipe Creation

```python
recipe = NVFP4BlockScaling()

# These are set during __post_init__
recipe.disable_rht = os.getenv("NVTE_NVFP4_DISABLE_RHT", "0") == "1"
recipe.disable_stochastic_rounding = os.getenv(...) == "1"
recipe.disable_2d_quantization = os.getenv(...) == "1"

# Which creates the QParams:
recipe.fp4_quant_fwd_inp = QParams(
    random_hadamard_transform=not recipe.disable_rht,
    stochastic_rounding=False,
    fp4_2d_quantization=False,
)
recipe.fp4_quant_fwd_weight = QParams(
    random_hadamard_transform=False,
    stochastic_rounding=False,
    fp4_2d_quantization=not recipe.disable_2d_quantization,  # ◄─ Controlled here
)
recipe.fp4_quant_bwd_grad = QParams(
    random_hadamard_transform=not recipe.disable_rht,
    stochastic_rounding=not recipe.disable_stochastic_rounding,
    fp4_2d_quantization=False,
)
```

---

## Summary

NVFP4 tests validate:

1. **Forward Pass:** Quantization → GEMM → Dequantization produces correct results
2. **Backward Pass:** Gradient computation with quantized tensors
3. **RHT Impact:** Random Hadamard transform improves distribution properties
4. **2D Quantization:** Tile-based scaling for weights vs 1D for inputs
5. **Numerical Stability:** Tolerance checks account for FP4 precision limits
6. **Functional Correctness:** Native implementation matches reference

The complete data flow traces from Python APIs through CUDA kernels, validating the entire quantization and computation pipeline.

