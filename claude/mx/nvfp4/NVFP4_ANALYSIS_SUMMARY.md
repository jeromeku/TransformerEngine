# NVFP4 Call Path Analysis - Executive Summary

## Quick Reference

This analysis documents the complete call path for `te.Linear` processing inputs under the NVFP4BlockScaling recipe, from Python API through C++ bindings to CUDA kernels.

## Key Findings

### 1. Recipe Detection Mechanism

**File**: [transformer_engine/common/recipe/__init__.py:387-481](../../../transformer_engine/common/recipe/__init__.py#L387)

The `NVFP4BlockScaling` recipe is detected via:
- `recipe.nvfp4()` → `isinstance(self, NVFP4BlockScaling)`
- Called during `Linear.set_meta_tensor(fwd, recipe)` initialization
- Triggers `_customize_quantizers_nvfp4()` for quantizer setup

### 2. Quantizer Architecture

**File**: [transformer_engine/pytorch/tensor/nvfp4_tensor.py:112-338](../../../transformer_engine/pytorch/tensor/nvfp4_tensor.py#L112)

Three NVFP4Quantizer instances per layer:
- **Input quantizer**: 1D block scaling (16-element blocks) with optional RHT
- **Weight quantizer**: 1D or 2D block scaling (16x16 for 2D mode)
- **Gradient quantizer**: 1D block scaling with optional stochastic rounding

### 3. Forward Pass Flow

**File**: [transformer_engine/pytorch/module/linear.py:1370-1500](../../../transformer_engine/pytorch/module/linear.py#L1370)

```
Linear.forward()
  ├─ Get quantizers from FP8GlobalStateManager
  ├─ _Linear.apply() [calls forward]
  │  ├─ Quantize input: input_quantizer(inp) → NVFP4Tensor
  │  ├─ Quantize weight: weight_quantizer(weight) → NVFP4Tensor
  │  └─ GEMM: general_gemm(weight_fp4, input_fp4, ...)
  └─ Return dequantized output
```

### 4. Quantization Pipeline

**File**: [transformer_engine/pytorch/tensor/nvfp4_tensor.py]

When quantizer is called (e.g., `input_quantizer(inp)`):

```python
inputmat = input_quantizer(inp)
  ↓
NVFP4Quantizer.quantize_impl(tensor)
  ↓
tex.quantize(tensor, self)  # C++ binding
```

### 5. CUDA Kernel Execution

Three main kernel families:

1. **Hadamard Transform** (optional, for column-wise data)
   - File: [hadamard_transform_cast_fusion.cu]
   - Smooths outliers via random Hadamard transform

2. **FP4 Quantization** (mandatory)
   - File: [quantize_transpose_vector_blockwise_fp4.cu]
   - 16-element blocks with E4M3 scaling factors
   - 2D variant for weights (16x16 blocks)

3. **GEMM Computation**
   - File: [cublaslt_gemm.cu]
   - cuBLASLt + CUTLASS kernel dispatch
   - FP4 x BF16 or FP4 x FP4 operations

### 6. Data Layout

**NVFP4Tensor Components**:
```
rowwise_data:           uint8  [M, K//2]  - FP4 data (2 per byte)
rowwise_scale_inv:      uint8  swizzled   - E4M3 block scales
amax_rowwise:          float32 [1]        - Per-tensor amax

columnwise_data:        uint8  [K, M//2]  - Transposed FP4 data
columnwise_scale_inv:   uint8  swizzled   - Transposed scales
amax_columnwise:       float32 [1]        - Column amax (if needed)
```

### 7. Distributed Training

**File**: [linear.py:1675-1696]

For tensor parallelism:
- Column-parallel: amax reduction across TP group for inputs
- Row-parallel: amax reduction across TP group for gradients
- Ensures scaling factor consistency across all ranks

### 8. Workspace Requirements

**File**: [module/base.py:77-92]

- Hopper (compute capability ≥ 9): 32 MiB + 1 KiB alignment
  - 32 MiB required for FP4 GEMM workspace
  - 1 KiB for alignment and misc scales
- Other architectures: 4 MiB

---

## Architecture Overview Diagram

```
┌────────────────────────────────────────────────────────┐
│ Python API Layer                                        │
│                                                         │
│  te.Linear.forward(inp)                               │
│    ├─ FP8GlobalStateManager.get_fp8_recipe()          │
│    │   └─ recipe.nvfp4() → NVFP4BlockScaling detected │
│    │                                                   │
│    ├─ _get_quantizers()                               │
│    │   └─ [NVFP4Quantizer, NVFP4Quantizer, ...]       │
│    │                                                   │
│    └─ _Linear.apply()                                 │
│       ├─ inputmat = input_quantizer(inp)              │
│       ├─ weightmat = weight_quantizer(weight)         │
│       └─ out = general_gemm(weightmat, inputmat, ...) │
└────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────┐
│ Quantizer Layer (Python)                               │
│                                                         │
│  NVFP4Quantizer.__call__(tensor)                       │
│    └─ quantize_impl(tensor)                            │
│       └─ tex.quantize(tensor, self)                    │
│                                                         │
│  Returns: NVFP4Tensor with FP4 data + scales           │
└────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────┐
│ C++ Binding Layer                                       │
│                                                         │
│  PyBind11: tex.quantize(tensor, quantizer)             │
│    ├─ Extract quantizer parameters                     │
│    │  (with_rht, with_2d_quantization, etc.)           │
│    │                                                   │
│    └─ Dispatch to appropriate kernel path              │
└────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────┐
│ CUDA Kernel Layer                                       │
│                                                         │
│  (Optional) Apply Random Hadamard Transform            │
│    └─ hadamard_transform_cast_fusion.cu                │
│                                                         │
│  FP4 Quantization (1D or 2D)                           │
│    └─ quantize_transpose_vector_blockwise_fp4.cu       │
│       ├─ Compute block amaxes                          │
│       ├─ Calculate scales (E4M3 format)                │
│       └─ Quantize to FP4                               │
│                                                         │
│  Return NVFP4Tensor with quantized data & scales       │
└────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────┐
│ GEMM Execution Layer                                    │
│                                                         │
│  general_gemm(weight_fp4, input_fp4, ...)              │
│    └─ tex.general_gemm()                               │
│       ├─ Extract FP4 data and scales from tensors      │
│       ├─ Setup cuBLASLt matmul descriptor              │
│       │  (data type: FP4, scaling: block scaling)      │
│       │                                                │
│       └─ Dispatch CUTLASS kernel                       │
│          └─ cublaslt_gemm.cu                           │
│             ├─ FP4 GEMM computation                    │
│             ├─ Apply per-block scales                  │
│             ├─ Apply global scale (from amax)          │
│             └─ Dequantize to output dtype (BF16/F32)   │
└────────────────────────────────────────────────────────┘
```

---

## Implementation Highlights

### 1. Recipe-Based Quantizer Selection

The recipe pattern allows runtime selection of quantizer type:

```python
# In Linear.set_meta_tensor()
if recipe.nvfp4():
    self._customize_quantizers_nvfp4(fwd, recipe)
elif recipe.float8_block_scaling():
    self._customize_quantizers_float8_blockwise_scaling(fwd, recipe)
```

This enables seamless switching between quantization strategies.

### 2. Lazy Quantization

Input/weight quantization happens at first use:
- Quantizers created once during module initialization
- Invoked on-demand during forward pass
- Cache strategies available (e.g., weight caching across microbatches)

### 3. Block Scaling Architecture

Two-level scaling system:
- **Level 1**: Per-block scales in E4M3 format (16 values per block)
- **Level 2**: Global per-tensor scale in FP32 (from amax values)

Combined scaling: `actual_scale = per_block_scale * global_scale`

### 4. RHT Optimization

Random Hadamard Transform applied selectively:
- **For inputs**: Applied to column-wise data (wgrad path)
- **For weights**: Not applied (affects numerical properties negatively)
- **For gradients**: Applied to column-wise data

Smooths value distributions to improve FP4 representation accuracy.

### 5. 2D Block Scaling

For weight matrices (optional):
- 16x16 block clustering instead of 16-element rows
- Reduces number of scaling factors
- More aggressive compression for weight-heavy models

---

## Performance Considerations

### Memory Efficiency

- **FP4 data**: 2x compression vs FP32 (4 bits vs 32 bits)
- **E4M3 scales**: Minimal overhead (1 byte per 16 values)
- **Amax tracking**: Single float32 per tensor

### Compute Efficiency

- **CUTLASS kernels**: Optimized for FP4 tensor cores (Hopper+)
- **Split accumulator**: FP32 intermediate accumulation for precision
- **Kernel fusion**: Hadamard + quantization potentially fused

### Communication Efficiency

- **Reduced precision**: Smaller tensors in collective operations
- **amax reduction**: Single value per tensor across TP group
- **Optional overlapping**: Can overlap quantization with communication

---

## Testing and Validation

**Test Files**:
- [tests/pytorch/nvfp4/test_nvfp4_module_exact.py] - Module-level testing
- [tests/pytorch/nvfp4/test_nvfp4_quantize_exact.py] - Quantization kernels
- [tests/pytorch/nvfp4/test_nvfp4_gemm_exact.py] - GEMM operations
- [tests/pytorch/nvfp4/test_nvfp4_rht_quantize_exact.py] - RHT functionality

---

## Key Code Locations

| Component | File | Lines |
|-----------|------|-------|
| Recipe Definition | recipe/__init__.py | 387-481 |
| Linear Module | module/linear.py | 1009-1500 |
| _Linear Function | module/linear.py | 77-482 |
| Quantizer | tensor/nvfp4_tensor.py | 112-338 |
| FP4 Quantization | tensor/nvfp4_tensor.py | 261-328 |
| CUDA Quantize | common/recipe/nvfp4.cu | 1-54 |
| Workspace | module/base.py | 77-92 |
| Global State | pytorch/quantization.py | Various |

---

## Future Extensions

1. **MXFP4 Support**: Similar architecture for mantissa-only scales
2. **Int4 Variants**: Extension to signed/unsigned integer 4-bit
3. **Custom Tile Sizes**: Runtime-configurable block sizes
4. **Activation Sparsity**: Integration with sparse kernels
5. **Mixed Precision**: Per-layer precision selection

---

Generated Document: `/home/jeromeku/transformerengine/claude/NVFP4_LINEAR_CALL_PATH.md`

