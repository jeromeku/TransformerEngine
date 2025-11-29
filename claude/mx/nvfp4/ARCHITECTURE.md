# TransformerEngine Low-Precision Quantization Architecture

> **Purpose**: A comprehensive guide to TransformerEngine's mixed-precision quantization system, focusing on MXFP8 and NVFP4 formats. This document traces the complete data flow from user-facing Python APIs through abstraction layers to CUDA kernels.
>
> **Target Audience**: Developers seeking to understand TransformerEngine's internals for debugging, optimization, or extension.

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Overview](#system-overview)
3. [Quantization Formats](#quantization-formats)
4. [Architecture Layers](#architecture-layers)
5. [MXFP8 Implementation Deep Dive](#mxfp8-implementation-deep-dive)
6. [NVFP4 Implementation Deep Dive](#nvfp4-implementation-deep-dive)
7. [Blockwise FP8 Implementation](#blockwise-fp8-implementation)
8. [Distributed Training Integration](#distributed-training-integration)
9. [Kernel Implementation Details](#kernel-implementation-details)
10. [API Reference & Call Paths](#api-reference--call-paths)

---

## Executive Summary

TransformerEngine (TE) provides highly optimized mixed-precision training for transformers using FP8, MXFP8, and NVFP4 data formats. The architecture consists of:

### **Key Components**

| Component | Purpose | Location |
|-----------|---------|----------|
| **Python API Layer** | User-facing interfaces, tensor wrappers | [`transformer_engine/pytorch/`](../transformer_engine/pytorch/) |
| **Quantizer Classes** | Quantization logic & configuration | [`transformer_engine/pytorch/tensor/`](../transformer_engine/pytorch/tensor/) |
| **C++ Binding Layer** | PyBind11 interface Python ↔ C++ | [`transformer_engine/pytorch/csrc/`](../transformer_engine/pytorch/csrc/) |
| **Recipe System** | Scaling strategies (delayed, block, current) | [`transformer_engine/common/recipe/`](../transformer_engine/common/recipe/) |
| **CUDA Kernels** | Low-level quantize/dequantize/GEMM ops | [`transformer_engine/common/`](../transformer_engine/common/) |

### **Supported Quantization Formats**

| Format | Block Size | Scale Type | Precision | Hardware | Use Case |
|--------|------------|------------|-----------|----------|----------|
| **FP8 E4M3** | Per-tensor | FP32 | 8-bit | H100+ | Forward pass activations, weights |
| **FP8 E5M2** | Per-tensor | FP32 | 8-bit | H100+ | Backward pass gradients |
| **MXFP8** | 32 elements | E8M0 (power-of-2) | 8-bit | Blackwell+ | All tensors in E4M3 |
| **NVFP4** | 16 elements (1D)<br>16×16 (2D weights) | E4M3 + FP32 (2-level) | 4-bit | Blackwell+ | Extreme memory savings |
| **Blockwise FP8** | 128 elements (1D)<br>128×128 (2D) | FP32 (power-of-2) | 8-bit | H100+ | Configurable granularity |

---

## System Overview

### High-Level Data Flow

```
User Code
    ↓
┌───────────────────────────────────────┐
│   Python API (te.autocast context)   │
│  - te.Linear, te.LayerNorm, etc.     │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│      Quantized Tensor Classes         │
│  - MXFP8Tensor                        │
│  - NVFP4Tensor                        │
│  - Float8BlockwiseQTensor             │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│         Quantizer Classes             │
│  - MXFP8Quantizer                     │
│  - NVFP4Quantizer                     │
│  - Float8BlockQuantizer               │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│      C++ Binding Layer (PyBind11)     │
│  - tex.quantize()                     │
│  - tex.generic_gemm()                 │
│  - tex.dequantize()                   │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│        C++ Core Implementation        │
│  - Dispatch to appropriate kernels    │
│  - Handle memory management           │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│          CUDA Kernels                 │
│  - Quantize/dequantize kernels        │
│  - cuBLAS/cuBLASLt GEMM               │
│  - Transpose/RHT kernels              │
└───────────────────────────────────────┘
```

### Key Design Patterns

1. **Wrapper Tensor Pattern**: All quantized types inherit from `torch.Tensor` via `_make_wrapper_subclass`, enabling transparent integration with PyTorch
2. **Quantizer Builder Pattern**: Separate quantizer objects configure and create quantized tensors
3. **Storage Separation**: Separate storage classes handle raw data, while tensor classes add autograd support
4. **Recipe System**: Pluggable scaling strategies (delayed, current, block-based)
5. **Dual-Layout Storage**: Row-wise and column-wise data/scales stored simultaneously to avoid requantization during transpose

---

## Quantization Formats

### MXFP8 (Microscaled FP8)

**Format Structure:**
- **Data**: FP8 E4M3 (1 sign, 4 exponent, 3 mantissa bits)
- **Scales**: E8M0 format (8-bit exponent representing powers of 2)
- **Block Size**: 32 consecutive elements
- **Scale Granularity**: One E8M0 scale per 32-element block

**Why MXFP8?**
1. **Higher precision**: All tensors use E4M3 (vs. E5M2 for gradients in standard FP8)
2. **Finer granularity**: Per-block scaling reduces quantization error
3. **Hardware efficiency**: Native Blackwell Tensor Core support

**Layout Requirements:**
```
Row-wise layout:    [M, K] → quantized as [M, K], scales [(M+127)/128, (K/32+3)/4]
Column-wise layout: [K, M] → quantized as [K, M], scales [(K+127)/128, (M/32+3)/4]
```

MXFP8 requires both layouts because:
- **Forward**: `W @ x` needs row-wise W, column-wise x
- **Backward (dW)**: `x.T @ grad` needs column-wise x
- **Backward (dx)**: `grad @ W.T` needs column-wise W

Transposing MXFP8 data requires requantization (unlike regular FP8), so TE precomputes both layouts.

---

### NVFP4 (4-bit Floating Point)

**Format Structure:**
- **Data**: FP4 E2M1 (1 sign, 2 exponent, 1 mantissa bit) → range ±6
- **Block Scales**: E4M3 (8-bit) per 16-element block (1D) or 16×16 block (2D)
- **Tensor Scale**: FP32 per-tensor scale (for overflow protection)
- **Block Size**:
  - Activations/Gradients: 16 elements (1D)
  - Weights: 16×16 elements (2D for symmetric transpose behavior)

**Two-Level Scaling:**
```python
dequantized_value = fp4_data * block_scale_e4m3 * tensor_scale_fp32
```

**Training Recipe Components:**

1. **Stochastic Rounding** (gradients only)
   - Probabilistic rounding avoids systematic bias
   - Essential for gradient accumulation accuracy

2. **Random Hadamard Transform (RHT)**
   - Applies 16×16 Hadamard matrix to smooth outliers
   - Makes distributions more Gaussian-like
   - Used for activations/gradients during weight gradient computation
   - Fixed random signs: `[1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1]`

3. **2D Scaling** (weights only)
   - 16×16 blocks ensure symmetric behavior under transpose
   - Critical for weight gradient computation accuracy

4. **Mixed Precision Layers**
   - Last few layers run in higher precision (MXFP8/BF16)
   - User must manually configure via nested `autocast` contexts

---

### Blockwise FP8

**Format Structure:**
- **Data**: FP8 E4M3 or E5M2
- **Scales**: FP32 (power-of-2 constrained)
- **Block Size**:
  - 1D: 128 elements
  - 2D: 128×128 elements
- **Data Format**: GEMM_READY (transposed, padded) or COMPACT

**Configurable Parameters:**
```python
Float8BlockQuantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,
    block_scaling_dim=2,          # 1D or 2D blocks
    force_pow_2_scales=True,      # Enforce power-of-2 scales
    amax_epsilon=0.0,             # Small constant added to amax
    all_gather_usage=False        # COMPACT format for comm
)
```

**Use Cases:**
- **DeepSeek-v3 style**: 2D weight quantization
- **Communication overlap**: COMPACT format reduces all-gather volume
- **Custom recipes**: Flexible block sizes and scale constraints

---

## Architecture Layers

### Layer 1: Python API (`transformer_engine/pytorch/`)

**Primary Entry Points:**

#### 1. `te.autocast()` Context Manager
[`transformer_engine/pytorch/fp8.py:110-250`](../transformer_engine/pytorch/fp8.py#L110-L250)

```python
with te.autocast(recipe=mxfp8_recipe):
    out = my_linear(inp)
```

**Responsibilities:**
- Set thread-local FP8 state
- Configure active recipe
- Manage amax history and scale updates
- Handle multi-GPU synchronization of scaling factors

#### 2. TE Module API (`te.Linear`, `te.LayerNorm`, etc.)
[`transformer_engine/pytorch/module/linear.py`](../transformer_engine/pytorch/module/linear.py)

**Core Modules:**
- **te.Linear**: [`linear.py:100-800`](../transformer_engine/pytorch/module/linear.py#L100-L800)
- **te.LayerNorm**: [`layernorm.py`](../transformer_engine/pytorch/module/layernorm.py)
- **te.RMSNorm**: [`rmsnorm.py`](../transformer_engine/pytorch/module/rmsnorm.py)
- **te.LayerNormLinear**: [`layernorm_linear.py`](../transformer_engine/pytorch/module/layernorm_linear.py)

These modules integrate with `autocast` to automatically quantize inputs/weights when FP8 is enabled.

---

### Layer 2: Quantized Tensor Classes

All quantized tensors inherit from:
1. **Storage class** (e.g., `MXFP8TensorStorage`) - holds raw data
2. **QuantizedTensor** interface - provides common methods
3. **torch.Tensor** (via `_make_wrapper_subclass`) - enables PyTorch integration

#### MXFP8Tensor Class Hierarchy

```
torch.Tensor (via _make_wrapper_subclass)
    ↑
MXFP8TensorStorage  ← Holds raw data buffers
    ↑
QuantizedTensor     ← Common interface
    ↑
MXFP8Tensor         ← Public API + autograd
```

**Key Files:**
- [`mxfp8_tensor.py:179-449`](../transformer_engine/pytorch/tensor/mxfp8_tensor.py#L179-L449) - MXFP8Tensor class
- [`mxfp8_tensor_storage.py`](../transformer_engine/pytorch/tensor/storage/mxfp8_tensor_storage.py) - Storage class
- [`quantized_tensor.py`](../transformer_engine/pytorch/tensor/quantized_tensor.py) - Base interface

**Data Members:**
```python
class MXFP8Tensor:
    _rowwise_data: torch.Tensor           # uint8, shape [M, K]
    _rowwise_scale_inv: torch.Tensor      # uint8 (E8M0), shape [(M+127)/128, (K/32+3)/4]
    _columnwise_data: torch.Tensor        # uint8, shape [M, K] (same as rowwise)
    _columnwise_scale_inv: torch.Tensor   # uint8 (E8M0), shape [(M/32+3)/4, (K+127)/128]
    _fp8_dtype: TE_DType                  # kFloat8E4M3
    _quantizer: MXFP8Quantizer            # Configuration
    dtype: torch.dtype                    # Nominal dtype (e.g., bfloat16)
    shape: torch.Size                     # Logical shape [M, K]
```

**Critical Methods:**

##### `dequantize()` → Returns high-precision tensor
[`mxfp8_tensor.py:232-245`](../transformer_engine/pytorch/tensor/mxfp8_tensor.py#L232-L245)

```python
def dequantize(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if dtype is None:
        dtype = self.dtype
    if torch.is_grad_enabled():
        return _FromMXFP8Func.apply(self, dtype)
    return _FromMXFP8Func.forward(None, self, dtype)
```

**Call trace:**
1. `_FromMXFP8Func.apply()` → Autograd-enabled path
2. `_FromMXFP8Func.forward()` → [`mxfp8_tensor_storage.py`](../transformer_engine/pytorch/tensor/storage/mxfp8_tensor_storage.py)
3. Calls C++ via PyBind11: `tex.dequantize()`
4. C++ implementation: [`extensions.cpp`](../transformer_engine/pytorch/csrc/extensions.cpp)
5. CUDA kernel: [`dequantize_kernels.cuh`](../transformer_engine/common/util/cuda_driver/kernels/dequantize_kernels.cuh)

##### `quantize_()` → In-place quantization
[`mxfp8_tensor.py:251-269`](../transformer_engine/pytorch/tensor/mxfp8_tensor.py#L251-L269)

```python
def quantize_(self, tensor: torch.Tensor, *, noop_flag: Optional[torch.Tensor] = None):
    if isinstance(tensor, QuantizedTensor):
        return self.quantize_(tensor.dequantize())
    return super().quantize_(tensor, noop_flag=noop_flag)
```

Delegates to quantizer's `update_quantized()`.

---

#### NVFP4Tensor Class Hierarchy

Similar structure to MXFP8:
```
torch.Tensor
    ↑
NVFP4TensorStorage
    ↑
QuantizedTensor
    ↑
NVFP4Tensor
```

**Key Files:**
- [`nvfp4_tensor.py:341-665`](../transformer_engine/pytorch/tensor/nvfp4_tensor.py#L341-L665) - NVFP4Tensor class
- [`nvfp4_tensor_storage.py`](../transformer_engine/pytorch/tensor/storage/nvfp4_tensor_storage.py) - Storage class

**Data Members:**
```python
class NVFP4Tensor:
    _rowwise_data: torch.Tensor           # uint8 (packed 4-bit), shape [M, K/2]
    _rowwise_scale_inv: torch.Tensor      # uint8 (E4M3), shape [(M+127)/128, (ceil(K/16)+3)/4]
    _amax_rowwise: torch.Tensor           # FP32 per-tensor scale, shape [1]
    _columnwise_data: torch.Tensor        # uint8, shape [K, M/2]
    _columnwise_scale_inv: torch.Tensor   # uint8 (E4M3), shape [(K+127)/128, (ceil(M/16)+3)/4]
    _amax_columnwise: torch.Tensor        # FP32, shape [1]
    _fp4_dtype: TE_DType                  # kFloat4E2M1
    _quantizer: NVFP4Quantizer
```

**Note on Data Packing:**
FP4 values are packed 2 per byte, so data tensors have `shape[-1] // 2`.

---

### Layer 3: Quantizer Classes

Quantizers are **builder objects** that:
1. Configure quantization parameters
2. Create empty quantized tensors
3. Perform quantization operations
4. Validate tensor shapes

#### MXFP8Quantizer

[`mxfp8_tensor.py:29-177`](../transformer_engine/pytorch/tensor/mxfp8_tensor.py#L29-L177)

```python
class MXFP8Quantizer(Quantizer):
    dtype: TE_DType                  # kFloat8E4M3
    rowwise: bool                    # Enable rowwise layout
    columnwise: bool                 # Enable columnwise layout

    def __init__(self, fp8_dtype: TE_DType, *, rowwise=True, columnwise=True):
        super().__init__(rowwise=rowwise, columnwise=columnwise)
        self.dtype = fp8_dtype
```

**Key Methods:**

##### `make_empty()` → Allocate quantized tensor
[`mxfp8_tensor.py:88-141`](../transformer_engine/pytorch/tensor/mxfp8_tensor.py#L88-L141)

```python
def make_empty(self, shape, *, dtype=torch.float32, device=None, requires_grad=False):
    # Validate shape is divisible by 32
    assert shape[-1] % 32 == 0 and math.prod(shape[:-1]) % 32 == 0

    # Allocate rowwise data (uint8)
    data = torch.empty(shape, dtype=torch.uint8, device=device)

    # Allocate rowwise scales (E8M0 = uint8)
    # Shape: [(M+127)/128, (K/32+3)/4] with padding for cuBLAS alignment
    scale_inv = torch.empty(
        round_up_to_nearest_multiple(math.prod(shape[:-1]), 128),
        round_up_to_nearest_multiple(shape[-1] // 32, 4),
        dtype=torch.uint8,
        device=device,
    )

    # Allocate columnwise if needed
    if self.columnwise_usage:
        columnwise_data = torch.empty_like(data)
        columnwise_scale_inv = torch.empty(
            round_up_to_nearest_multiple(math.prod(shape[:-1]) // 32, 4),
            round_up_to_nearest_multiple(shape[-1], 128),
            dtype=torch.uint8,
            device=device,
        )

    return MXFP8Tensor(...)  # Construct tensor with all buffers
```

**Scale Shape Calculation:**
- Scales are padded to multiples of 4 (inner dim) and 128 (outer dim)
- This satisfies cuBLAS alignment requirements for efficient GEMM

##### `update_quantized()` → Quantize into existing tensor
[`mxfp8_tensor.py:50-72`](../transformer_engine/pytorch/tensor/mxfp8_tensor.py#L50-L72)

```python
def update_quantized(self, src: torch.Tensor, dst: QuantizedTensor, *, noop_flag=None):
    assert isinstance(dst, MXFP8Tensor)

    # Ensure contiguous CUDA tensor
    if not devices_match(src.device, dst.device):
        src = src.to(device=dst.device)
    if not src.is_contiguous():
        src = src.contiguous()

    # Launch CUDA kernel via C++
    tex.quantize(src, self, dst, noop_flag)

    # Update dtype
    dst._fp8_dtype = self.dtype
    return dst
```

**Call Path:**
```
Python: tex.quantize(src, self, dst, noop_flag)
    ↓
C++: transformer_engine::pytorch::quantize()
    [pybind.cpp:119](../transformer_engine/pytorch/csrc/extensions/pybind.cpp#L119)
    ↓
Dispatch based on quantizer type (MXFP8Quantizer)
    ↓
CUDA: MXFP8 quantization kernel
    [fp8_block_scaling.cu](../transformer_engine/common/recipe/fp8_block_scaling.cu)
```

---

#### NVFP4Quantizer

[`nvfp4_tensor.py:112-339`](../transformer_engine/pytorch/tensor/nvfp4_tensor.py#L112-L339)

```python
class NVFP4Quantizer(Quantizer):
    dtype: TE_DType                          # kFloat4E2M1
    with_rht: bool                           # Enable Random Hadamard Transform
    with_post_rht_amax: bool                 # Compute amax after RHT
    with_amax_reduction: bool                # All-reduce amax across GPUs
    amax_reduction_group: Optional[dist_group_type]
    with_2d_quantization: bool               # 2D blocks (weights only)
    stochastic_rounding: bool                # SR for gradients
    rht_matrix: torch.Tensor                 # 16×16 Hadamard matrix
    rht_matrix_random_sign_mask_t: int       # Sign mask for RHT
```

**Configuration Options:**

##### Random Hadamard Transform Setup
[`nvfp4_tensor.py:91-110`](../transformer_engine/pytorch/tensor/nvfp4_tensor.py#L91-L110)

```python
@functools.lru_cache(maxsize=None)
def get_rht_matrix(with_random_sign_mask: bool) -> torch.Tensor:
    hadamard_dimension = 16
    if with_random_sign_mask:
        # Fixed random signs for reproducibility
        signs = get_wgrad_sign_vector()
    else:
        signs = get_no_random_sign_vector()  # All 1s

    sign_matrix = signs * torch.eye(16, dtype=torch.float32, device="cuda")
    rht_matrix = sign_matrix @ get_hadamard_matrix(16)
    return rht_matrix.to(dtype=torch.bfloat16)
```

The RHT matrix is cached and reused. Signs are hard-coded for determinism:
```python
[1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1]
```

##### Scale Shape Calculation (1D)
[`nvfp4_tensor.py:192-226`](../transformer_engine/pytorch/tensor/nvfp4_tensor.py#L192-L226)

```python
def get_scale_shape(self, shape, columnwise):
    M = math.prod(shape[:-1])
    K = shape[-1]

    if columnwise:
        outer = round_up_to_nearest_multiple(K, 128)
        inner = round_up_to_nearest_multiple(math.ceil(M / 16), 4)
        return (outer, inner)
    else:  # rowwise
        outer = round_up_to_nearest_multiple(M, 128)
        inner = round_up_to_nearest_multiple(math.ceil(K / 16), 4)
        return (outer, inner)
```

**Scale shape formula:**
- Rowwise: `[(M+127)/128, (ceil(K/16)+3)/4]`
- Columnwise: `[(K+127)/128, (ceil(M/16)+3)/4]`

##### `make_empty()` with 2-level scaling
[`nvfp4_tensor.py:261-328`](../transformer_engine/pytorch/tensor/nvfp4_tensor.py#L261-L328)

```python
def make_empty(self, shape, *, dtype=torch.float32, device=None, requires_grad=False):
    # Validate divisibility by 16
    assert shape[-1] % 16 == 0
    assert math.prod(shape[:-1]) % 16 == 0

    # Allocate rowwise
    if self.rowwise_usage:
        # FP4 data: packed 2 per byte
        data = torch.empty(self.convert_shape_for_fp4(shape), dtype=torch.uint8, device=device)

        # Block scales (E4M3 = uint8)
        scale_shape = self.get_scale_shape(shape, columnwise=False)
        scale_inv = torch.empty(scale_shape, dtype=torch.uint8, device=device)

        # Tensor-level scale (FP32)
        amax_rowwise = torch.zeros(1, dtype=torch.float32, device=device)

    # Similar for columnwise...

    return NVFP4Tensor(
        shape=shape,
        dtype=dtype,
        rowwise_data=data,
        rowwise_scale_inv=scale_inv,
        amax_rowwise=amax_rowwise,
        columnwise_data=columnwise_data,
        columnwise_scale_inv=columnwise_scale_inv,
        amax_columnwise=amax_columnwise,
        fp4_dtype=self.dtype,
        quantizer=self,
        requires_grad=requires_grad,
    )
```

---

#### Float8BlockQuantizer

[`float8_blockwise_tensor.py:27-279`](../transformer_engine/pytorch/tensor/float8_blockwise_tensor.py#L27-L279)

```python
class Float8BlockQuantizer(Quantizer):
    dtype: TE_DType                    # E4M3 or E5M2
    block_len: int = 128               # Fixed block size
    amax_epsilon: float                # Added to amax to avoid div-by-zero
    force_pow_2_scales: bool           # Quantize scales to powers of 2
    block_scaling_dim: int             # 1 (1D) or 2 (2D)
    all_gather_usage: bool             # COMPACT vs GEMM_READY format
```

**Scale Calculation (2D):**
[`float8_blockwise_tensor.py:112-172`](../transformer_engine/pytorch/tensor/float8_blockwise_tensor.py#L112-L172)

```python
def get_scale_shape(self, shape, columnwise):
    M = math.prod(shape[:-1])
    K = shape[-1]

    if self.block_scaling_dim == 2:  # 128×128 blocks
        if columnwise:
            outer = math.ceil(K / 128)
            inner = round_up_to_nearest_multiple(math.ceil(M / 128), 4)
            return (outer, inner)
        else:  # rowwise
            outer = math.ceil(M / 128)
            inner = round_up_to_nearest_multiple(math.ceil(K / 128), 4)
            return (outer, inner)

    elif self.block_scaling_dim == 1:  # 1×128 blocks
        if columnwise:
            outer = math.ceil(M / 128)
            inner = round_up_to_nearest_multiple(K, 4) if not self.all_gather_usage else K
            return (outer, inner)
        else:  # rowwise
            outer = math.ceil(K / 128)
            inner = round_up_to_nearest_multiple(M, 4) if not self.all_gather_usage else M
            return (outer, inner) if not self.all_gather_usage else (inner, outer)
```

**Data Format:**
- **GEMM_READY**: Data and scales transposed/swizzled for cuBLAS
- **COMPACT**: Minimal padding, used for communication (all-gather)

---

### Layer 4: C++ Binding Layer

**Entry Point:** [`pybind.cpp:117-133`](../transformer_engine/pytorch/csrc/extensions/pybind.cpp#L117-L133)

```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Quantization functions
  m.def("quantize", transformer_engine::pytorch::quantize,
        py::arg("tensor"), py::arg("quantizer"),
        py::arg("output") = py::none(), py::arg("noop") = py::none());

  m.def("dequantize", &transformer_engine::pytorch::dequantize,
        "Dequantize", py::arg("input"), py::arg("otype"));

  // GEMM
  m.def("generic_gemm", transformer_engine::pytorch::gemm,
        "Compute GEMM (matrix-matrix multiply)",
        py::arg("A"), py::arg("transA"), py::arg("B"), py::arg("transB"),
        py::arg("D"), py::arg("quantizer"), py::arg("output_dtype"),
        py::arg("bias"), py::arg("bias_type"), py::arg("gelu"),
        py::arg("gelu_in"), py::arg("grad"), py::arg("workspace"),
        py::arg("workspace_size"), py::arg("accumulate"),
        py::arg("use_split_accumulator"), ...);

  // ... many more functions
}
```

**Initialization:** [`pybind.cpp:38-111`](../transformer_engine/pytorch/csrc/extensions/pybind.cpp#L38-L111)

```cpp
void init_extension() {
  init_float8_extension();       // Load Float8Tensor classes
  init_mxfp8_extension();        // Load MXFP8Tensor classes
  init_float8blockwise_extension();  // Load Float8BlockwiseQTensor
  init_nvfp4_extensions();       // Load NVFP4Tensor classes
}

void init_mxfp8_extension() {
  // Import Python modules via PyBind11
  auto fp8_module = py::module_::import("transformer_engine.pytorch.tensor.mxfp8_tensor");

  // Get class objects for type checking
  MXFP8QuantizerClass = reinterpret_cast<PyTypeObject*>(
      PyObject_GetAttrString(fp8_module.ptr(), "MXFP8Quantizer"));
  MXFP8TensorPythonClass = reinterpret_cast<PyTypeObject*>(
      PyObject_GetAttrString(fp8_module.ptr(), "MXFP8Tensor"));
  // ... similar for storage classes
}
```

**Type Dispatch in `tex.quantize()`:**

The C++ `quantize()` function (not shown in the file snippets, but referenced) performs runtime type checking:

```cpp
// Pseudo-code (actual implementation in extensions.cpp)
py::object quantize(py::object tensor, py::object quantizer,
                    py::object output, py::object noop) {
  // Check quantizer type
  if (PyObject_IsInstance(quantizer.ptr(), (PyObject*)MXFP8QuantizerClass)) {
    return quantize_mxfp8(tensor, quantizer, output, noop);
  }
  else if (PyObject_IsInstance(quantizer.ptr(), (PyObject*)NVFP4QuantizerClass)) {
    return quantize_nvfp4(tensor, quantizer, output, noop);
  }
  else if (PyObject_IsInstance(quantizer.ptr(), (PyObject*)Float8BlockwiseQuantizerClass)) {
    return quantize_blockwise_fp8(tensor, quantizer, output, noop);
  }
  // ... other types
}
```

Each specialized function then:
1. Extracts configuration from quantizer
2. Allocates output tensor if needed (via Python constructor)
3. Launches appropriate CUDA kernel
4. Returns quantized tensor

---

### Layer 5: CUDA Kernel Layer

Kernels are organized by recipe type:

| Recipe | Kernel File | Purpose |
|--------|-------------|---------|
| **MXFP8** | [`fp8_block_scaling.cu`](../transformer_engine/common/recipe/fp8_block_scaling.cu) | MXFP8 quantize/dequantize |
| **NVFP4** | [`nvfp4.cu`](../transformer_engine/common/recipe/nvfp4.cu) | FP4 quantize/dequantize |
| **Delayed FP8** | [`delayed_scaling.cu`](../transformer_engine/common/recipe/delayed_scaling.cu) | Per-tensor FP8 |
| **Current FP8** | [`current_scaling.cu`](../transformer_engine/common/recipe/current_scaling.cu) | JIT FP8 scaling |
| **Blockwise FP8** | [`fp8_block_scaling.cu`](../transformer_engine/common/recipe/fp8_block_scaling.cu) | Block-scaled FP8 |

**Common Kernel Utilities:**
- [`cast_kernels.cuh`](../transformer_engine/common/util/cuda_driver/kernels/cast_kernels.cuh) - Type conversion kernels (98KB)
- [`dequantize_kernels.cuh`](../transformer_engine/common/util/cuda_driver/kernels/dequantize_kernels.cuh) - Dequantization (18KB)
- [`nvfp4_transpose.cuh`](../transformer_engine/common/util/cuda_driver/kernels/nvfp4_transpose.cuh) - FP4 transpose (68KB)

---

## MXFP8 Implementation Deep Dive

### Complete Call Path: User Code → CUDA

Let's trace a simple MXFP8 linear layer forward pass:

```python
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import MXFP8BlockScaling

# 1. Setup recipe and module
recipe = MXFP8BlockScaling(fp8_format=Format.E4M3)
linear = te.Linear(768, 768, bias=True)
inp = torch.randn(1024, 768, device='cuda', dtype=torch.bfloat16)

# 2. Run with autocast
with te.autocast(recipe=recipe):
    out = linear(inp)
```

#### Step-by-Step Trace

**Step 1: Autocast Context Entry**

Location: [`transformer_engine/pytorch/fp8.py:110-250`](../transformer_engine/pytorch/fp8.py#L110-L250)

```python
class autocast:
    def __enter__(self):
        # Set thread-local state
        get_fp8_context().enabled = True
        get_fp8_context().recipe = self.recipe
        # ... initialize amax/scale buffers if needed
        return self
```

**Step 2: Linear Forward Pass**

Location: [`transformer_engine/pytorch/module/linear.py:400-600`](../transformer_engine/pytorch/module/linear.py#L400-L600) (approximate)

```python
class Linear(torch.nn.Module):
    def forward(self, inp):
        # Check if FP8 enabled
        if is_fp8_enabled():
            # Quantize input
            inp_fp8 = quantize_tensor(inp, self.quantizer)

            # Quantize weight (if not already cached)
            if self.weight_fp8 is None:
                self.weight_fp8 = quantize_tensor(self.weight, self.weight_quantizer)

            # FP8 GEMM
            out = fp8_gemm(inp_fp8, self.weight_fp8, ...)
        else:
            # Regular matmul
            out = inp @ self.weight.T

        # Add bias if present
        if self.bias is not None:
            out = out + self.bias

        return out
```

**Step 3: Quantize Input**

The `quantize_tensor()` call internally does:

```python
# Python layer
def quantize_tensor(tensor, quantizer):
    # Create empty quantized tensor
    qtensor = quantizer.make_empty(
        tensor.shape,
        dtype=tensor.dtype,
        device=tensor.device
    )
    # Quantize into it
    quantizer.update_quantized(tensor, qtensor)
    return qtensor
```

**Step 3a: `make_empty()`** - Already detailed in Quantizer section

**Step 3b: `update_quantized()`**

[`mxfp8_tensor.py:50-72`](../transformer_engine/pytorch/tensor/mxfp8_tensor.py#L50-L72)

```python
def update_quantized(self, src, dst, *, noop_flag=None):
    # Ensure contiguous
    src = src.to(device=dst.device).contiguous()

    # Call C++ binding
    tex.quantize(src, self, dst, noop_flag)

    dst._fp8_dtype = self.dtype
    return dst
```

**Step 4: C++ Binding**

[`pybind.cpp:119`](../transformer_engine/pytorch/csrc/extensions/pybind.cpp#L119)

```cpp
m.def("quantize", transformer_engine::pytorch::quantize, ...);
```

Implementation (in [`extensions.cpp`](../transformer_engine/pytorch/csrc/extensions.cpp), not shown):

```cpp
py::object quantize(py::object src_tensor, py::object quantizer,
                    py::object dst_tensor, py::object noop_flag) {
  // Extract quantizer parameters
  auto quantizer_type = quantizer.get_type();

  if (is_mxfp8_quantizer(quantizer_type)) {
    // Extract MXFP8-specific params
    TE_DType dtype = quantizer.attr("dtype").cast<TE_DType>();
    bool rowwise = quantizer.attr("rowwise").cast<bool>();
    bool columnwise = quantizer.attr("columnwise").cast<bool>();

    // Get data pointers from tensors
    auto* src_data = tensor_data_ptr(src_tensor);
    auto* dst_data_rowwise = tensor_data_ptr(dst_tensor.attr("_rowwise_data"));
    auto* dst_scale_rowwise = tensor_data_ptr(dst_tensor.attr("_rowwise_scale_inv"));
    auto* dst_data_columnwise = tensor_data_ptr(dst_tensor.attr("_columnwise_data"));
    auto* dst_scale_columnwise = tensor_data_ptr(dst_tensor.attr("_columnwise_scale_inv"));

    // Launch CUDA kernel
    nvte_mxfp8_quantize(
        src_data, src_shape,
        dst_data_rowwise, dst_scale_rowwise,
        dst_data_columnwise, dst_scale_columnwise,
        dtype, stream
    );

    return dst_tensor;
  }
  // ... other quantizer types
}
```

**Step 5: CUDA Kernel**

Location: [`fp8_block_scaling.cu`](../transformer_engine/common/recipe/fp8_block_scaling.cu)

Pseudo-kernel code:
```cuda
__global__ void mxfp8_quantize_kernel(
    const __nv_bfloat16* src,      // [M, K]
    uint8_t* dst_data_rowwise,     // [M, K]
    uint8_t* dst_scale_rowwise,    // [(M+127)/128, (K/32+3)/4]
    uint8_t* dst_data_columnwise,  // [M, K]
    uint8_t* dst_scale_columnwise, // [(M/32+3)/4, (K+127)/128]
    int M, int K
) {
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;

    // Process rowwise quantization
    if (tidx < M * K / 32) {
        int block_id = tidx;
        int row = block_id / (K / 32);
        int block_in_row = block_id % (K / 32);

        // Compute amax for this 32-element block
        float amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            int col = block_in_row * 32 + i;
            float val = __bfloat162float(src[row * K + col]);
            amax = fmaxf(amax, fabsf(val));
        }

        // Compute E8M0 scale
        float fp_max = 448.0f;  // E4M3 max value
        uint8_t e8m0_scale = compute_e8m0_scale(amax, fp_max);

        // Store scale
        int scale_row = row;
        int scale_col = block_in_row;
        dst_scale_rowwise[scale_row * ((K/32+3)/4) + scale_col] = e8m0_scale;

        // Quantize and store 32 elements
        float scale_inv_fp32 = e8m0_to_fp32(e8m0_scale);
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            int col = block_in_row * 32 + i;
            float val = __bfloat162float(src[row * K + col]);
            float scaled = val / scale_inv_fp32;
            uint8_t quantized = float_to_fp8_e4m3(scaled);
            dst_data_rowwise[row * K + col] = quantized;
        }
    }

    // Process columnwise quantization (similar logic, different indexing)
    // ...
}
```

**Key Optimizations:**
1. **Vectorized loads**: Use `float4` to load 16 bytes at once
2. **Warp-level reductions**: Compute block amax using shuffle ops
3. **Shared memory**: Cache blocks for coalesced access
4. **Stream pipelining**: Overlap rowwise/columnwise kernels

**Step 6: FP8 GEMM**

After quantization, the GEMM is called:

```python
out = fp8_gemm(inp_fp8, weight_fp8, ...)
```

Which translates to:

```python
tex.generic_gemm(
    weight_fp8,        # A matrix
    True,              # transA (weight is [out, in], need [in, out])
    inp_fp8,           # B matrix
    False,             # transB
    None,              # D (output, allocated by cuBLAS)
    None,              # out_quantizer
    TE_DType.kBFloat16,  # output dtype
    bias,              # optional bias
    TE_DType.kBFloat16,  # bias dtype
    False,             # gelu
    None,              # gelu_in
    False,             # grad
    workspace,         # cuBLAS workspace
    workspace_size,
    False,             # accumulate
    True,              # use_split_accumulator
)
```

**GEMM Call Path:**

1. **Python**: `tex.generic_gemm()` → [`pybind.cpp:126`](../transformer_engine/pytorch/csrc/extensions/pybind.cpp#L126)
2. **C++**: `transformer_engine::pytorch::gemm()` → dispatches based on input types
3. **C++ GEMM wrapper**: Calls cuBLASLt for block-scaled FP8 GEMM
4. **cuBLASLt API**:
   ```cpp
   cublasLtMatmul(
       ltHandle,
       operationDesc,  // Configured for FP8 with block scaling
       alpha,
       A_mxfp8, A_desc,
       B_mxfp8, B_desc,
       beta,
       C_bf16, C_desc,
       workspace, workspace_size,
       stream
   );
   ```

5. **cuBLAS kernel**: Hardware-accelerated FP8 tensor core GEMM (vendor implementation)

**MXFP8 cuBLAS Configuration:**
```cpp
// Set matrix data type to FP8 E4M3
cublasLtMatmulDescSetAttribute(
    operationDesc, CUBLASLT_MATMUL_DESC_A_TYPE,
    &fp8_type, sizeof(fp8_type)
);

// Enable block scaling
cublasLtMatmulDescSetAttribute(
    operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
    &A_scale_ptr, sizeof(A_scale_ptr)
);

// Set scale format to E8M0
cudaDataType_t scale_type = CUDA_R_8I;  // Interpreted as E8M0 by cuBLAS
cublasLtMatmulDescSetAttribute(
    operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_TYPE,
    &scale_type, sizeof(scale_type)
);
```

**Step 7: Backward Pass (dequantize for user code)**

When user calls `.backward()`, gradients flow through custom autograd functions.

For MXFP8Tensor, the backward of operations auto-dequantizes:

```python
class _FromMXFP8Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mxfp8_tensor, dtype):
        # Dequantize for forward
        return tex.dequantize(mxfp8_tensor, dtype)

    @staticmethod
    def backward(ctx, grad_output):
        # Gradient flows through unchanged (casting is a no-op for gradients)
        return grad_output, None
```

**Dequantize Kernel Path:**

1. **Python**: `tex.dequantize(mxfp8_tensor, dtype)` → [`pybind.cpp:121`](../transformer_engine/pytorch/csrc/extensions/pybind.cpp#L121)
2. **C++**: `transformer_engine::pytorch::dequantize()` → extract layout (rowwise/columnwise)
3. **CUDA**: [`dequantize_kernels.cuh`](../transformer_engine/common/util/cuda_driver/kernels/dequantize_kernels.cuh)

Pseudo-kernel:
```cuda
__global__ void mxfp8_dequantize_kernel(
    const uint8_t* src_data,      // [M, K]
    const uint8_t* src_scale,     // [(M+127)/128, (K/32+3)/4]
    __nv_bfloat16* dst,           // [M, K]
    int M, int K
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * K) return;

    int row = idx / K;
    int col = idx % K;
    int block_col = col / 32;

    // Load E8M0 scale and convert to FP32
    int scale_row = row;
    int scale_col = block_col;
    uint8_t e8m0_scale = src_scale[scale_row * ((K/32+3)/4) + scale_col];
    float scale_inv = e8m0_to_fp32(e8m0_scale);

    // Load FP8 data, cast to FP32, apply scale
    uint8_t fp8_val = src_data[idx];
    float fp32_val = fp8_e4m3_to_float(fp8_val);
    float dequantized = fp32_val * scale_inv;

    // Convert to output dtype
    dst[idx] = __float2bfloat16(dequantized);
}
```

---

### MXFP8 Memory Layout Details

#### Rowwise Layout

**Data Tensor:** `[M, K]` in FP8 E4M3 (uint8)
```
Element (i, j) stored at: rowwise_data[i * K + j]
```

**Scale Tensor:** `[(M+127)/128, (K/32+3)/4]` in E8M0 (uint8)
```
Scale for block (i, j/32) stored at:
rowwise_scale_inv[(i // 128) * padded_scale_cols + (j/32)]

where padded_scale_cols = (K/32 + 3) / 4
```

#### Columnwise Layout

**Data Tensor:** `[M, K]` (same shape as rowwise, but computed differently)
```
Element (i, j) stored at: columnwise_data[i * K + j]

But block boundaries align differently:
- Rowwise: 32 consecutive elements in K dimension
- Columnwise: 32 consecutive elements in M dimension
```

**Scale Tensor:** `[(M/32+3)/4, (K+127)/128]` in E8M0
```
Scale for block (i/32, j) stored at:
columnwise_scale_inv[(i/32) * padded_scale_rows + (j // 128)]
```

#### Why Two Layouts?

Consider matrix multiply `C = A @ B.T`:
- **A needs rowwise**: Inner dimension K is contiguous for block scaling
- **B needs columnwise**: After transpose, inner dim K is contiguous

MXFP8 cannot simply transpose data+scales (unlike regular FP8) because:
1. Block boundaries would misalign
2. Scales would need recomputation (different amaxes)

So TE pre-computes both layouts during quantization.

---

### E8M0 Scale Format

E8M0 is an 8-bit representation of a power-of-2 scale:

**Encoding:**
```
value = 2^(exponent - 127)

where exponent is the 8-bit unsigned integer (0-255)
```

**Examples:**
- `exponent = 127` → scale = 2^0 = 1.0
- `exponent = 130` → scale = 2^3 = 8.0
- `exponent = 124` → scale = 2^-3 = 0.125

**Conversion Functions:**
```cpp
// Encode FP32 to E8M0
uint8_t fp32_to_e8m0(float scale_inv) {
    if (scale_inv == 0.0f) return 0;
    int exponent;
    frexpf(scale_inv, &exponent);  // Extract exponent
    return static_cast<uint8_t>(exponent + 126);  // Bias
}

// Decode E8M0 to FP32
float e8m0_to_fp32(uint8_t e8m0) {
    if (e8m0 == 0) return 0.0f;
    int exponent = static_cast<int>(e8m0) - 127;
    return ldexpf(1.0f, exponent);  // Compute 2^exponent
}
```

**Why E8M0?**
1. **Compact**: 1 byte per scale (vs. 4 bytes for FP32)
2. **Fast**: No mantissa bits → multiplication is a bitshift
3. **Sufficient range**: 2^-127 to 2^127 covers all practical scales
4. **Hardware support**: Blackwell Tensor Cores natively understand E8M0

---

## NVFP4 Implementation Deep Dive

### NVFP4 Training Recipe

Recall from earlier, NVFP4 training uses:
1. **Stochastic rounding** (gradients)
2. **Random Hadamard Transform** (activations/gradients for weight grad)
3. **2D block scaling** (weights)
4. **Mixed precision last layers**

Let's trace these through the code.

### Complete Call Path: Linear Layer with NVFP4

```python
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import NVFP4BlockScaling

# Setup
recipe = NVFP4BlockScaling()
linear1 = te.Linear(768, 768).bfloat16()  # FP4 layer
linear2 = te.Linear(768, 768).bfloat16()  # Higher precision
inp = torch.randn(1024, 768, device='cuda', dtype=torch.bfloat16)

# Forward with nested autocast
with te.autocast(recipe=recipe):
    x = linear1(inp)
    with te.autocast(recipe=MXFP8BlockScaling(Format.E4M3)):
        out = linear2(x)  # Override to MXFP8

loss = out.mean()
loss.backward()
```

### NVFP4 Quantization with RHT

**Setup in Quantizer:**

[`nvfp4_tensor.py:133-156`](../transformer_engine/pytorch/tensor/nvfp4_tensor.py#L133-L156)

```python
def __init__(self, ..., with_rht=False, with_random_sign_mask=True, ...):
    self.with_rht = with_rht
    self.rht_matrix_random_sign_mask_t = get_random_sign_mask_for_rht(with_random_sign_mask)
    self.rht_matrix = get_rht_matrix(with_random_sign_mask)
```

The RHT matrix is:
```python
rht_matrix = sign_matrix @ hadamard_matrix
```
where `sign_matrix = diag([1,1,1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,1,-1,-1])`

**Quantization with RHT:**

When `with_rht=True`, the quantizer configuration is passed to C++:

```python
# Python: tex.quantize(tensor, nvfp4_quantizer, output)
```

C++ extracts:
```cpp
bool with_rht = quantizer.attr("with_rht").cast<bool>();
int rht_sign_mask = quantizer.attr("rht_matrix_random_sign_mask_t").cast<int>();
auto rht_matrix = quantizer.attr("rht_matrix").cast<torch::Tensor>();
```

**CUDA Kernel (pseudo-code):**

```cuda
__global__ void nvfp4_quantize_with_rht_kernel(
    const __nv_bfloat16* src,       // [M, K], assume K = 16*n for simplicity
    uint8_t* dst_data,              // [M, K/2] (packed FP4)
    uint8_t* dst_scale,             // [M, K/16] (E4M3 block scales)
    float* dst_amax,                // [1] (tensor-level scale)
    const __nv_bfloat16* rht_mat,   // [16, 16]
    int rht_sign_mask,              // 16-bit mask
    int M, int K
) {
    int row = blockIdx.x;
    int block_start = threadIdx.x * 16;  // Each thread handles one 16-element block

    if (row >= M || block_start >= K) return;

    // Load 16 elements
    __nv_bfloat16 vals[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        vals[i] = src[row * K + block_start + i];
    }

    // Apply RHT: vals_transformed = rht_mat @ vals
    __nv_bfloat16 vals_transformed[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        float sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            sum += __bfloat162float(rht_mat[i * 16 + j]) * __bfloat162float(vals[j]);
        }
        vals_transformed[i] = __float2bfloat16(sum);
    }

    // Compute amax of transformed values
    float amax_block = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        amax_block = fmaxf(amax_block, fabsf(__bfloat162float(vals_transformed[i])));
    }

    // Reduce amax across all blocks for tensor-level scale
    atomicMaxFloat(dst_amax, amax_block);  // Simplified; real impl uses reductions

    // Compute block-level E4M3 scale
    float fp4_max = 6.0f;  // E2M1 max value
    float block_scale_fp32 = amax_block / fp4_max;
    uint8_t block_scale_e4m3 = float_to_fp8_e4m3(block_scale_fp32);
    dst_scale[row * (K/16) + (block_start/16)] = block_scale_e4m3;

    // Load tensor-level scale (computed in previous kernel or post-processing)
    float tensor_scale_inv = *dst_amax;

    // Quantize transformed values to FP4
    float combined_scale = fp8_e4m3_to_float(block_scale_e4m3) * tensor_scale_inv;
    #pragma unroll
    for (int i = 0; i < 16; i += 2) {
        // Quantize two FP4 values and pack into one byte
        float val0 = __bfloat162float(vals_transformed[i]) / combined_scale;
        float val1 = __bfloat162float(vals_transformed[i+1]) / combined_scale;

        uint8_t fp4_0 = float_to_fp4_e2m1(val0);
        uint8_t fp4_1 = float_to_fp4_e2m1(val1);

        // Pack: [fp4_1 (high 4 bits) | fp4_0 (low 4 bits)]
        uint8_t packed = (fp4_1 << 4) | fp4_0;
        dst_data[row * (K/2) + (block_start + i) / 2] = packed;
    }
}
```

**Key Points:**
1. RHT is applied **before** quantization to smooth distribution
2. Amax computed on **transformed** values (unless `with_post_rht_amax=False`)
3. Two-level scaling: block scale (E4M3) × tensor scale (FP32)
4. FP4 values packed 2 per byte

### Stochastic Rounding

For gradients, NVFP4 uses stochastic rounding to avoid bias.

**Configuration:**
```python
grad_quantizer = NVFP4Quantizer(
    stochastic_rounding=True,  # Enable SR
    with_rht=True,             # Also use RHT for gradients
    with_2d_quantization=False # Gradients use 1D
)
```

**SR Implementation (pseudo-code):**

```cuda
__device__ uint8_t stochastic_round_to_fp4(float val, float scale, uint32_t rng_state) {
    float scaled = val / scale;

    // Find two nearest FP4 values
    float fp4_low = floor_to_fp4(scaled);
    float fp4_high = ceil_to_fp4(scaled);

    // Compute probabilities
    float distance = scaled - fp4_low;
    float range = fp4_high - fp4_low;
    float p_high = distance / range;  // Probability of rounding up

    // Generate random number in [0, 1)
    float rand_val = curand_uniform(&rng_state);

    // Stochastic decision
    if (rand_val < p_high) {
        return float_to_fp4(fp4_high);
    } else {
        return float_to_fp4(fp4_low);
    }
}
```

**Why SR for gradients?**
- Deterministic rounding introduces systematic bias
- Over many iterations, biased gradients accumulate error
- SR is unbiased in expectation: `E[quantized_grad] = true_grad`

### 2D Quantization for Weights

Weights use 16×16 blocks to ensure symmetric transpose behavior.

**Quantizer Setup:**
```python
weight_quantizer = NVFP4Quantizer(
    with_2d_quantization=True,  # 2D blocks
    with_rht=False,             # Weights don't use RHT
    stochastic_rounding=False   # Deterministic for weights
)
```

**2D Block Quantization:**

Each 16×16 block shares one E4M3 scale:

```cuda
__global__ void nvfp4_quantize_2d_kernel(
    const __nv_bfloat16* src,  // [M, K]
    uint8_t* dst_data,         // [M, K/2]
    uint8_t* dst_scale,        // [M/16, K/16] (E4M3)
    float* dst_amax,           // [1] (FP32)
    int M, int K
) {
    // Each block processes one 16×16 tile
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    int row_offset = block_row * 16;
    int col_offset = block_col * 16;

    if (row_offset >= M || col_offset >= K) return;

    // Cooperatively compute amax over 16×16 block
    __shared__ float shared_amax;
    if (threadIdx.x == 0) shared_amax = 0.0f;
    __syncthreads();

    int tidx = threadIdx.x;  // 0-255 (16×16 elements)
    int local_row = tidx / 16;
    int local_col = tidx % 16;
    int global_row = row_offset + local_row;
    int global_col = col_offset + local_col;

    if (global_row < M && global_col < K) {
        float val = __bfloat162float(src[global_row * K + global_col]);
        atomicMaxFloat(&shared_amax, fabsf(val));
    }
    __syncthreads();

    // Compute block scale
    float fp4_max = 6.0f;
    float block_scale_fp32 = shared_amax / fp4_max;
    uint8_t block_scale_e4m3 = float_to_fp8_e4m3(block_scale_fp32);

    if (threadIdx.x == 0) {
        dst_scale[block_row * (K/16) + block_col] = block_scale_e4m3;
        atomicMaxFloat(dst_amax, shared_amax);  // Update tensor amax
    }
    __syncthreads();

    // Quantize with 2-level scale
    float tensor_scale_inv = *dst_amax;
    float combined_scale = fp8_e4m3_to_float(block_scale_e4m3) * tensor_scale_inv;

    if (global_row < M && global_col < K && (global_col % 2 == 0)) {
        // Pack two FP4 values
        float val0 = __bfloat162float(src[global_row * K + global_col]) / combined_scale;
        float val1 = (global_col + 1 < K)
            ? __bfloat162float(src[global_row * K + global_col + 1]) / combined_scale
            : 0.0f;

        uint8_t fp4_0 = float_to_fp4_e2m1(val0);
        uint8_t fp4_1 = float_to_fp4_e2m1(val1);
        uint8_t packed = (fp4_1 << 4) | fp4_0;

        dst_data[global_row * (K/2) + global_col/2] = packed;
    }
}
```

**Why 2D for weights?**

1D quantization (16×1 blocks):
```
Original W:  [N, K]
Transposed:  [K, N]

Problem: Block boundaries don't align after transpose
→ W.T has different quantization than if we quantized [K, N] directly
→ Weight gradient (computed using W.T) has errors
```

2D quantization (16×16 blocks):
```
Original W:  [N, K] with 16×16 blocks
Transposed:  [K, N] with 16×16 blocks (same blocks, just indexed differently)

→ W.T uses the same block scales as W
→ Symmetric under transpose → accurate gradients
```

### NVFP4 GEMM

NVFP4 uses cuBLASLt's block-scaled FP4 GEMM:

**Python Call:**
```python
tex.generic_gemm(
    weight_nvfp4,  # A: [out_features, in_features] in FP4
    True,          # transA
    inp_nvfp4,     # B: [batch, in_features] in FP4
    False,         # transB
    ...,
    use_split_accumulator=True  # Required for FP4
)
```

**C++ Configuration:**
```cpp
// cuBLAS GEMM descriptor
cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

// Set A (weight) to FP4 E2M1
cublasLtMatmulDescSetAttribute(
    operationDesc, CUBLASLT_MATMUL_DESC_A_TYPE,
    &fp4_type, sizeof(fp4_type)
);

// Set B (input) to FP4 E2M1
cublasLtMatmulDescSetAttribute(
    operationDesc, CUBLASLT_MATMUL_DESC_B_TYPE,
    &fp4_type, sizeof(fp4_type)
);

// Provide block scale pointers
cublasLtMatmulDescSetAttribute(
    operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
    &A_block_scale_ptr, sizeof(void*)
);
cublasLtMatmulDescSetAttribute(
    operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
    &B_block_scale_ptr, sizeof(void*)
);

// Set scale type to E4M3
cudaDataType_t scale_type = CUDA_R_8F_E4M3;
cublasLtMatmulDescSetAttribute(
    operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_TYPE,
    &scale_type, sizeof(scale_type)
);

// Provide tensor-level scales
cublasLtMatmulDescSetAttribute(
    operationDesc, CUBLASLT_MATMUL_DESC_AMAX_POINTER,
    &A_amax_ptr, sizeof(void*)
);
cublasLtMatmulDescSetAttribute(
    operationDesc, CUBLASLT_MATMUL_DESC_BMAX_POINTER,
    &B_amax_ptr, sizeof(void*)
);

// Enable split-K accumulation (required for FP4)
int split_k = 1;  // Typically 1 for FP4
cublasLtMatmulDescSetAttribute(
    operationDesc, CUBLASLT_MATMUL_DESC_SPLIT_K,
    &split_k, sizeof(split_k)
);

// Execute
cublasLtMatmul(ltHandle, operationDesc, &alpha,
               A_fp4, A_desc,
               B_fp4, B_desc,
               &beta,
               C_bf16, C_desc,
               workspace, workspace_size, stream);
```

**Hardware Execution:**

On Blackwell:
1. FP4 data loaded into Tensor Cores
2. Block scales (E4M3) loaded
3. Tensor scales (FP32) loaded
4. FP4 → FP32 promotion with 2-level scaling on-the-fly
5. FP32 accumulation in Tensor Core
6. Output cast to BF16

**Effective Operation:**
```
C[i,j] = Σ_k (A[i,k] * A_block_scale[i//16, k//16] * A_amax) *
             (B[k,j] * B_block_scale[k//16, j//16] * B_amax)
```

---

## Blockwise FP8 Implementation

Blockwise FP8 is a flexible format allowing 1D (1×128) or 2D (128×128) blocks.

### Configuration

```python
from transformer_engine.pytorch import Float8BlockQuantizer
import transformer_engine_torch as tex

# 1D quantization (activations/gradients)
act_quantizer = Float8BlockQuantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,
    rowwise=True,
    columnwise=True,
    block_scaling_dim=1,          # 1D blocks
    force_pow_2_scales=True,      # Constrain scales to powers of 2
    amax_epsilon=0.0,
    all_gather_usage=False        # GEMM_READY format
)

# 2D quantization (weights, DeepSeek-v3 style)
weight_quantizer = Float8BlockQuantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,
    rowwise=True,
    columnwise=True,
    block_scaling_dim=2,          # 2D blocks
    force_pow_2_scales=True,
    amax_epsilon=0.0,
    all_gather_usage=False
)
```

### Data Formats

#### GEMM_READY Format

**Purpose:** Data and scales pre-transposed/swizzled for cuBLAS

**Characteristics:**
- Row-wise data: `[M, K]`
- Row-wise scales: Transposed to cuBLAS layout
- Column-wise data: `[K, M]` (transposed)
- Column-wise scales: Transposed to cuBLAS layout
- Both layouts padded for alignment

**Use Case:** Default format for normal training

#### COMPACT Format

**Purpose:** Minimal padding for communication efficiency

**Characteristics:**
- Row-wise data: `[M, K]` (no transpose)
- Row-wise scales: Compact layout (no padding)
- Column-wise data: May not exist (depending on usage)
- Column-wise scales: Compact if present

**Use Case:** All-gather in distributed training (reduces communication volume)

**Conversion:**
```python
# Create COMPACT quantizer for all-gather
ag_quantizer = Float8BlockQuantizer(
    ...,
    all_gather_usage=True  # COMPACT format
)

# Quantize before all-gather
compact_tensor = ag_quantizer.quantize(tensor)

# All-gather (smaller payload)
gathered = torch.distributed.all_gather(compact_tensor, ...)

# Convert to GEMM_READY for computation
gemm_ready_quantizer = Float8BlockQuantizer(..., all_gather_usage=False)
gemm_ready_tensor = gemm_ready_quantizer.quantize(gathered)

# Use in GEMM
out = tex.generic_gemm(gemm_ready_tensor, ...)
```

### Scale Shape Calculations

#### 1D Blocks (1×128)

**For tensor `[M, K]`:**

Row-wise:
- **Data**: `[M, K]`
- **Scales**: `[ceil(K/128), round_to_4(M)]` for GEMM_READY
- **Scales**: `[M, ceil(K/128)]` for COMPACT

Column-wise:
- **Data**: `[K, M]`
- **Scales**: `[ceil(M/128), round_to_4(K)]` for GEMM_READY
- **Scales**: `[K, ceil(M/128)]` for COMPACT

#### 2D Blocks (128×128)

**For tensor `[M, K]`:**

Row-wise:
- **Data**: `[M, K]`
- **Scales**: `[ceil(M/128), round_to_4(ceil(K/128))]`

Column-wise:
- **Data**: `[K, M]`
- **Scales**: `[ceil(K/128), round_to_4(ceil(M/128))]`

### Quantization Kernel (1D)

```cuda
__global__ void fp8_block_quantize_1d_kernel(
    const __nv_bfloat16* src,  // [M, K]
    uint8_t* dst_data,         // [M, K]
    float* dst_scale,          // [ceil(K/128), round_to_4(M)]
    int M, int K,
    bool force_pow2
) {
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_blocks = M * (K / 128);

    if (tidx >= num_blocks) return;

    int row = tidx / (K / 128);
    int block_col = tidx % (K / 128);
    int col_start = block_col * 128;

    // Compute amax for this 128-element block
    float amax = 0.0f;
    #pragma unroll 16
    for (int i = 0; i < 128; i++) {
        float val = __bfloat162float(src[row * K + col_start + i]);
        amax = fmaxf(amax, fabsf(val));
    }

    // Compute scale
    float fp8_max = 448.0f;  // E4M3 max
    float scale_fp32 = amax / fp8_max;

    if (force_pow2) {
        // Round to nearest power of 2
        int exponent;
        frexpf(scale_fp32, &exponent);
        scale_fp32 = ldexpf(1.0f, exponent);
    }

    // Store scale (transposed for cuBLAS)
    int scale_row = block_col;
    int scale_col = row;
    dst_scale[scale_row * round_to_4(M) + scale_col] = scale_fp32;

    // Quantize and store
    float scale_inv = 1.0f / scale_fp32;
    #pragma unroll 16
    for (int i = 0; i < 128; i++) {
        float val = __bfloat162float(src[row * K + col_start + i]);
        float scaled = val * scale_inv;
        uint8_t quantized = float_to_fp8_e4m3(scaled);
        dst_data[row * K + col_start + i] = quantized;
    }
}
```

### Quantization Kernel (2D)

```cuda
__global__ void fp8_block_quantize_2d_kernel(
    const __nv_bfloat16* src,  // [M, K]
    uint8_t* dst_data,         // [M, K]
    float* dst_scale,          // [ceil(M/128), round_to_4(ceil(K/128))]
    int M, int K,
    bool force_pow2
) {
    // Each block processes one 128×128 tile
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    int row_offset = block_row * 128;
    int col_offset = block_col * 128;

    if (row_offset >= M || col_offset >= K) return;

    // Shared memory for block amax reduction
    __shared__ float shared_amax;
    if (threadIdx.x == 0) shared_amax = 0.0f;
    __syncthreads();

    // Each thread processes multiple elements
    int tidx = threadIdx.x;  // Assume 256 threads
    int elements_per_thread = (128 * 128) / 256;  // 64 elements per thread

    float local_amax = 0.0f;
    for (int i = 0; i < elements_per_thread; i++) {
        int flat_idx = tidx * elements_per_thread + i;
        int local_row = flat_idx / 128;
        int local_col = flat_idx % 128;
        int global_row = row_offset + local_row;
        int global_col = col_offset + local_col;

        if (global_row < M && global_col < K) {
            float val = __bfloat162float(src[global_row * K + global_col]);
            local_amax = fmaxf(local_amax, fabsf(val));
        }
    }

    // Reduce across threads
    atomicMaxFloat(&shared_amax, local_amax);
    __syncthreads();

    // Compute scale
    float amax = shared_amax;
    float fp8_max = 448.0f;
    float scale_fp32 = amax / fp8_max;

    if (force_pow2) {
        int exponent;
        frexpf(scale_fp32, &exponent);
        scale_fp32 = ldexpf(1.0f, exponent);
    }

    // Store scale
    if (threadIdx.x == 0) {
        int scale_row = block_row;
        int scale_col = block_col;
        dst_scale[scale_row * round_to_4(ceil(K/128)) + scale_col] = scale_fp32;
    }
    __syncthreads();

    // Quantize
    float scale_inv = 1.0f / scale_fp32;
    for (int i = 0; i < elements_per_thread; i++) {
        int flat_idx = tidx * elements_per_thread + i;
        int local_row = flat_idx / 128;
        int local_col = flat_idx % 128;
        int global_row = row_offset + local_row;
        int global_col = col_offset + local_col;

        if (global_row < M && global_col < K) {
            float val = __bfloat162float(src[global_row * K + global_col]);
            float scaled = val * scale_inv;
            uint8_t quantized = float_to_fp8_e4m3(scaled);
            dst_data[global_row * K + global_col] = quantized;
        }
    }
}
```

---

## Distributed Training Integration

TransformerEngine integrates with PyTorch distributed training primitives:
- **FSDP** (Fully Sharded Data Parallel)
- **TP** (Tensor Parallel)
- **SP** (Sequence Parallel)
- **CP** (Context Parallel)
- **EP** (Expert Parallel for MoE)
- **PP** (Pipeline Parallel)

### Key Integration Points

#### 1. FSDP2 Support

File: [`distributed.py:1-2500`](../transformer_engine/pytorch/distributed.py#L1-L2500)

**Capabilities:**
- FP8 parameter all-gather
- Master weight casting
- Gradient reduction with FP8
- Checkpoint/resume with FP8 state

**Example Integration:**
```python
from torch.distributed._composable.fsdp import fully_shard
import transformer_engine.pytorch as te

# Create model with TE modules
model = MyTransformer(...)

# Apply FSDP2 with FP8
for module in model.modules():
    if isinstance(module, te.Linear):
        fully_shard(module)

# FP8 training
with te.autocast(recipe=mxfp8_recipe):
    loss = model(input)
    loss.backward()
```

**Under the Hood:**

When FSDP all-gathers parameters:
1. **Before all-gather**: Parameters in master precision (BF16)
2. **Cast to FP8**: Using COMPACT format quantizer
3. **All-gather**: Smaller communication volume
4. **Unshard**: Reconstruct full FP8 parameter
5. **Use in forward**: Direct FP8 GEMM

**Implementation:**
```python
# Simplified FSDP integration
def fsdp_fp8_all_gather(sharded_param, quantizer):
    # Quantize local shard to FP8 COMPACT
    fp8_shard = quantizer.quantize(sharded_param)

    # All-gather FP8 shards
    fp8_full = torch.empty(world_size * fp8_shard.numel(), ...)
    torch.distributed.all_gather_into_tensor(fp8_full, fp8_shard)

    # Convert to GEMM_READY for computation
    gemm_ready_param = convert_compact_to_gemm_ready(fp8_full)

    return gemm_ready_param
```

#### 2. Tensor Parallel

**Column Parallel Linear:** [`module/linear.py`](../transformer_engine/pytorch/module/linear.py)

```python
class ColumnParallelLinear(te.Linear):
    def forward(self, inp):
        # Input is replicated across TP group
        # Weight is column-wise sharded

        # Local FP8 GEMM
        local_out = super().forward(inp)  # Uses FP8

        # No all-reduce needed (each rank has different output columns)
        return local_out
```

**Row Parallel Linear:**

```python
class RowParallelLinear(te.Linear):
    def forward(self, inp):
        # Input is column-wise sharded
        # Weight is row-wise sharded

        # Local FP8 GEMM
        local_out = super().forward(inp)  # Uses FP8

        # All-reduce across TP group
        torch.distributed.all_reduce(local_out, group=tp_group)

        return local_out
```

**Communication Overlap:**

TE supports overlapping communication with backward computation:

```python
# Enable comm overlap in GEMM
tex.generic_gemm(
    ...,
    comm_overlap=True,
    comm_type="all_gather"  # or "all_reduce"
)
```

This pipelines:
1. Start all-gather for next layer
2. Compute current layer backward
3. Wait for all-gather to complete
4. Proceed to next layer

#### 3. Sequence Parallel

Sequence dimension is sharded across ranks:

```python
# Input: [batch, seq_len // world_size, hidden]
# Local computation with FP8
local_out = te_linear(local_inp)

# All-gather in sequence dimension
full_out = torch.empty(batch, seq_len, hidden)
torch.distributed.all_gather_into_tensor(full_out, local_out, dim=1)
```

**With FP8:**
- Forward: All-gather activations in FP8 → GEMM
- Backward: Reduce-scatter gradients in FP8

#### 4. Amax Reduction

For stable FP8 training, amax values must be synchronized across ranks:

**Configuration:**
```python
nvfp4_quantizer = NVFP4Quantizer(
    with_amax_reduction=True,
    amax_reduction_group=my_process_group  # TP or DP group
)
```

**Implementation:**
```python
# After computing local amax
local_amax = compute_amax(tensor)

# All-reduce across group
torch.distributed.all_reduce(
    local_amax,
    op=torch.distributed.ReduceOp.MAX,
    group=amax_reduction_group
)

# Use global amax for scaling
scale = compute_scale(local_amax)
```

This ensures all ranks use consistent scaling factors.

---

## Kernel Implementation Details

### Common Kernel Patterns

All quantization kernels follow similar structure:

1. **Load data**: Coalesced reads from global memory
2. **Compute amax**: Warp-level or block-level reduction
3. **Compute scale**: Apply formula (amax / fp_max)
4. **Quantize**: Apply scale and cast to low-precision
5. **Store**: Coalesced writes

**Optimization Techniques:**

#### Vectorized Memory Access

```cuda
// Bad: Scalar loads
for (int i = 0; i < 128; i++) {
    float val = src[idx + i];  // 128 transactions
    // ...
}

// Good: Vectorized loads
float4* src_vec = reinterpret_cast<float4*>(src);
for (int i = 0; i < 32; i++) {
    float4 val = src_vec[idx/4 + i];  // 32 transactions (128 bytes each)
    // Process val.x, val.y, val.z, val.w
}
```

#### Warp-Level Reductions

```cuda
__device__ float warp_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Compute block amax
float local_amax = 0.0f;
for (int i = 0; i < elements_per_thread; i++) {
    local_amax = fmaxf(local_amax, fabsf(data[i]));
}
float block_amax = warp_max(local_amax);  // Fast!
```

#### Shared Memory for Scales

```cuda
__shared__ float shared_scales[128];  // Cache scales in SMEM

// Load scales once per block
if (threadIdx.x < num_scales) {
    shared_scales[threadIdx.x] = global_scales[threadIdx.x];
}
__syncthreads();

// Reuse from SMEM
for (int i = 0; i < elements_per_thread; i++) {
    int scale_idx = compute_scale_idx(i);
    float scale = shared_scales[scale_idx];  // Fast SMEM access
    // ...
}
```

### FP8/FP4 Conversion Routines

#### Float → FP8 E4M3

```cuda
__device__ uint8_t float_to_fp8_e4m3(float val) {
    // Handle special cases
    if (isnan(val)) return 0x7f;  // NaN
    if (val == 0.0f) return 0x00;

    // Extract sign
    uint32_t bits = __float_as_uint(val);
    uint8_t sign = (bits >> 31) & 0x1;

    // Get absolute value
    float abs_val = fabsf(val);

    // Clamp to E4M3 range [0, 448]
    if (abs_val > 448.0f) abs_val = 448.0f;

    // Extract exponent and mantissa
    int exp;
    float mantissa = frexpf(abs_val, &exp);  // mantissa in [0.5, 1.0)

    // Bias exponent for E4M3 (bias = 7)
    int biased_exp = exp + 7;
    if (biased_exp < 0) biased_exp = 0;       // Underflow
    if (biased_exp > 15) biased_exp = 15;     // Overflow (shouldn't happen after clamp)

    // Extract 3 mantissa bits
    uint8_t mantissa_bits = static_cast<uint8_t>((mantissa - 0.5f) * 8.0f);  // 3 bits

    // Pack: [sign (1 bit) | exp (4 bits) | mantissa (3 bits)]
    uint8_t fp8 = (sign << 7) | ((biased_exp & 0xf) << 3) | (mantissa_bits & 0x7);

    return fp8;
}
```

#### FP8 E4M3 → Float

```cuda
__device__ float fp8_e4m3_to_float(uint8_t fp8) {
    // Extract fields
    uint8_t sign = (fp8 >> 7) & 0x1;
    uint8_t exp = (fp8 >> 3) & 0xf;
    uint8_t mantissa = fp8 & 0x7;

    // Handle zero
    if (exp == 0 && mantissa == 0) return 0.0f;

    // Handle NaN
    if (exp == 15 && mantissa == 7) return NAN;

    // Unbias exponent
    int unbiased_exp = static_cast<int>(exp) - 7;

    // Reconstruct mantissa (add implicit leading 1)
    float mantissa_val = 1.0f + static_cast<float>(mantissa) / 8.0f;

    // Compute value
    float val = ldexpf(mantissa_val, unbiased_exp);

    // Apply sign
    if (sign) val = -val;

    return val;
}
```

#### Float → FP4 E2M1

```cuda
__device__ uint8_t float_to_fp4_e2m1(float val) {
    // Handle special cases
    if (isnan(val)) return 0x7;  // NaN (4-bit)
    if (val == 0.0f) return 0x0;

    // Extract sign
    uint8_t sign = (val < 0.0f) ? 1 : 0;
    float abs_val = fabsf(val);

    // Clamp to E2M1 range [0, 6]
    if (abs_val > 6.0f) abs_val = 6.0f;

    // Extract exponent and mantissa
    int exp;
    float mantissa = frexpf(abs_val, &exp);

    // Bias exponent for E2M1 (bias = 1)
    int biased_exp = exp + 1;
    if (biased_exp < 0) biased_exp = 0;
    if (biased_exp > 3) biased_exp = 3;

    // Extract 1 mantissa bit
    uint8_t mantissa_bit = (mantissa >= 0.75f) ? 1 : 0;

    // Pack: [sign (1 bit) | exp (2 bits) | mantissa (1 bit)]
    uint8_t fp4 = (sign << 3) | ((biased_exp & 0x3) << 1) | mantissa_bit;

    return fp4 & 0xf;  // Ensure 4 bits
}
```

#### FP4 E2M1 → Float

```cuda
__device__ float fp4_e2m1_to_float(uint8_t fp4) {
    // Extract fields
    uint8_t sign = (fp4 >> 3) & 0x1;
    uint8_t exp = (fp4 >> 1) & 0x3;
    uint8_t mantissa = fp4 & 0x1;

    // Handle zero
    if (exp == 0 && mantissa == 0) return 0.0f;

    // Handle NaN
    if (exp == 3 && mantissa == 1) return NAN;

    // Unbias exponent
    int unbiased_exp = static_cast<int>(exp) - 1;

    // Reconstruct mantissa
    float mantissa_val = 1.0f + static_cast<float>(mantissa) * 0.5f;  // 1 bit

    // Compute value
    float val = ldexpf(mantissa_val, unbiased_exp);

    // Apply sign
    if (sign) val = -val;

    return val;
}
```

### cuBLAS GEMM Integration

TransformerEngine wraps cuBLASLt for all low-precision GEMMs.

**Generic GEMM Function:**

Location: C++ implementation (referenced in [`pybind.cpp:126`](../transformer_engine/pytorch/csrc/extensions/pybind.cpp#L126))

```cpp
std::tuple<torch::Tensor, ...> gemm(
    torch::Tensor A,                    // Left matrix (possibly quantized)
    bool transA,                        // Transpose A?
    torch::Tensor B,                    // Right matrix (possibly quantized)
    bool transB,                        // Transpose B?
    torch::Tensor D,                    // Output/accumulation tensor
    py::object quantizer,               // Output quantizer (if quantizing output)
    DType output_dtype,                 // Output dtype
    torch::Tensor bias,                 // Optional bias
    DType bias_type,                    // Bias dtype
    bool gelu,                          // Apply GELU epilogue?
    torch::Tensor gelu_in,              // GELU auxiliary input
    bool grad,                          // Gradient of GELU?
    torch::Tensor workspace,            // cuBLAS workspace
    size_t workspace_size,              // Workspace size
    bool accumulate,                    // Accumulate into D?
    bool use_split_accumulator,         // Use split-K?
    ...
) {
    // Determine GEMM type from input types
    GEMMType gemm_type = infer_gemm_type(A, B);

    if (gemm_type == GEMMType::MXFP8) {
        return gemm_mxfp8(A, transA, B, transB, ...);
    } else if (gemm_type == GEMMType::NVFP4) {
        return gemm_nvfp4(A, transA, B, transB, ...);
    } else if (gemm_type == GEMMType::FP8_BLOCKWISE) {
        return gemm_fp8_blockwise(A, transA, B, transB, ...);
    } else {
        // Regular FP8, BF16, FP16, etc.
        return gemm_standard(A, transA, B, transB, ...);
    }
}
```

**MXFP8 GEMM:**

```cpp
torch::Tensor gemm_mxfp8(
    MXFP8Tensor A, bool transA,
    MXFP8Tensor B, bool transB,
    ...
) {
    // Extract row-wise or column-wise data based on transpose
    void* A_data = transA ? A.columnwise_data().data_ptr() : A.rowwise_data().data_ptr();
    void* A_scales = transA ? A.columnwise_scale_inv().data_ptr() : A.rowwise_scale_inv().data_ptr();

    void* B_data = transB ? B.columnwise_data().data_ptr() : B.rowwise_data().data_ptr();
    void* B_scales = transB ? B.columnwise_scale_inv().data_ptr() : B.rowwise_scale_inv().data_ptr();

    // Setup cuBLASLt operation descriptor
    cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

    // Configure for MXFP8
    cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_A_TYPE, &CUDA_R_8F_E4M3, ...);
    cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_B_TYPE, &CUDA_R_8F_E4M3, ...);
    cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &A_scales, ...);
    cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &B_scales, ...);
    cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_TYPE, &CUDA_R_8I, ...);  // E8M0
    cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_TYPE, &CUDA_R_8I, ...);

    // Configure bias/epilogue if needed
    if (bias) {
        cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, ...);
    }
    if (gelu) {
        cublasLtEpilogue_t epilogue = grad ? CUBLASLT_EPILOGUE_DGELU : CUBLASLT_EPILOGUE_GELU;
        cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, ...);
    }

    // Setup matrix layouts
    cublasLtMatrixLayoutCreate(&A_layout, CUDA_R_8F_E4M3, transA ? K : M, transA ? M : K, ...);
    cublasLtMatrixLayoutCreate(&B_layout, CUDA_R_8F_E4M3, transB ? N : K, transB ? K : N, ...);
    cublasLtMatrixLayoutCreate(&C_layout, output_dtype, M, N, ...);

    // Allocate output if needed
    if (!D.defined()) {
        D = torch::empty({M, N}, torch::dtype(output_dtype).device(A.device()));
    }

    // Execute
    float alpha = 1.0f, beta = accumulate ? 1.0f : 0.0f;
    cublasLtMatmul(
        cublas_lt_handle,
        op_desc,
        &alpha,
        A_data, A_layout,
        B_data, B_layout,
        &beta,
        D.data_ptr(), C_layout,
        workspace.data_ptr(), workspace_size,
        stream
    );

    return D;
}
```

---

## API Reference & Call Paths

### Summary Table: User Code → Kernel

| User API | Python Entry | C++ Binding | CUDA Kernel | Hardware |
|----------|--------------|-------------|-------------|----------|
| `te.autocast(recipe)` | [`fp8.py:110`](../transformer_engine/pytorch/fp8.py#L110) | N/A | N/A | N/A |
| `te.Linear(...)` | [`linear.py:100`](../transformer_engine/pytorch/module/linear.py#L100) | Multiple (quantize, gemm) | See below | Tensor Cores |
| `MXFP8Quantizer.quantize()` | [`mxfp8_tensor.py:74`](../transformer_engine/pytorch/tensor/mxfp8_tensor.py#L74) | [`pybind.cpp:119`](../transformer_engine/pytorch/csrc/extensions/pybind.cpp#L119) | [`fp8_block_scaling.cu`](../transformer_engine/common/recipe/fp8_block_scaling.cu) | CUDA cores |
| `NVFP4Quantizer.quantize()` | [`nvfp4_tensor.py:178`](../transformer_engine/pytorch/tensor/nvfp4_tensor.py#L178) | [`pybind.cpp:119`](../transformer_engine/pytorch/csrc/extensions/pybind.cpp#L119) | [`nvfp4.cu`](../transformer_engine/common/recipe/nvfp4.cu) | CUDA cores |
| `Float8BlockQuantizer.quantize()` | [`float8_blockwise_tensor.py:108`](../transformer_engine/pytorch/tensor/float8_blockwise_tensor.py#L108) | [`pybind.cpp:119`](../transformer_engine/pytorch/csrc/extensions/pybind.cpp#L119) | [`fp8_block_scaling.cu`](../transformer_engine/common/recipe/fp8_block_scaling.cu) | CUDA cores |
| `tensor.dequantize()` | `*_tensor.py::dequantize()` | [`pybind.cpp:121`](../transformer_engine/pytorch/csrc/extensions/pybind.cpp#L121) | [`dequantize_kernels.cuh`](../transformer_engine/common/util/cuda_driver/kernels/dequantize_kernels.cuh) | CUDA cores |
| `tex.generic_gemm()` | Direct C++ call | [`pybind.cpp:126`](../transformer_engine/pytorch/csrc/extensions/pybind.cpp#L126) | cuBLASLt | Tensor Cores |
| `te.LayerNorm(...)` | [`layernorm.py`](../transformer_engine/pytorch/module/layernorm.py) | [`pybind.cpp:236`](../transformer_engine/pytorch/csrc/extensions/pybind.cpp#L236) | LayerNorm kernels | CUDA cores |
| `te.RMSNorm(...)` | [`rmsnorm.py`](../transformer_engine/pytorch/module/rmsnorm.py) | [`pybind.cpp:240`](../transformer_engine/pytorch/csrc/extensions/pybind.cpp#L240) | RMSNorm kernels | CUDA cores |

### Detailed Call Paths

#### MXFP8 End-to-End

```
User:    with te.autocast(recipe=MXFP8BlockScaling()):
User:        out = te_linear(inp)

↓ te_linear.forward()
│
├─ Quantize Input:
│  Python:  quantizer.quantize(inp)
│  Python:  tex.quantize(inp, quantizer, output_tensor)
│  C++:     pybind.cpp:119 → quantize()
│  C++:     Dispatch to MXFP8 quantizer
│  CUDA:    fp8_block_scaling.cu::mxfp8_quantize_kernel()
│  Result:  MXFP8Tensor with rowwise/columnwise data+scales
│
├─ Quantize Weight (cached):
│  [Same as input quantization]
│
├─ GEMM:
│  Python:  tex.generic_gemm(weight_mxfp8, True, inp_mxfp8, False, ...)
│  C++:     pybind.cpp:126 → gemm()
│  C++:     Dispatch to gemm_mxfp8()
│  C++:     Setup cuBLASLt descriptor with E8M0 scales
│  cuBLAS:  cublasLtMatmul() with block-scaled FP8
│  GPU:     Blackwell Tensor Core GEMM
│  Result:  BF16 output tensor
│
└─ Return:  out (BF16)

Backward:
User:    loss.backward()

↓ Autograd
│
├─ Gradient of GEMM:
│  Autograd: Calls GEMM backward autograd function
│  GEMM:     tex.generic_gemm() for dW, dX
│  Result:   BF16 gradients (or FP8 if configured)
│
└─ Dequantize if needed:
   Python:  tensor.dequantize()
   C++:     pybind.cpp:121 → dequantize()
   CUDA:    dequantize_kernels.cuh::mxfp8_dequantize_kernel()
   Result:  BF16 tensor
```

#### NVFP4 End-to-End with RHT

```
User:    nvfp4_quantizer = NVFP4Quantizer(with_rht=True, stochastic_rounding=True)
User:    out_fp4 = nvfp4_quantizer.quantize(grad_tensor)

↓ quantizer.quantize()
│
├─ Python:  quantizer.quantize_impl(grad_tensor)
│  Python:  tex.quantize(grad_tensor, quantizer)
│  C++:     pybind.cpp:119 → quantize()
│  C++:     Extract quantizer config:
│             - with_rht = True
│             - stochastic_rounding = True
│             - rht_matrix, rht_sign_mask
│  CUDA:    nvfp4.cu::nvfp4_quantize_rht_sr_kernel()
│             1. Apply RHT: vals_transformed = rht_matrix @ vals
│             2. Compute amax on transformed vals
│             3. Compute block scales (E4M3) and tensor scale (FP32)
│             4. Quantize to FP4 E2M1 with stochastic rounding
│             5. Pack 2 FP4 values per byte
│  Result:  NVFP4Tensor with:
│             - rowwise_data: uint8 [M, K/2]
│             - rowwise_scale_inv: uint8 E4M3 [(M+127)/128, (K/16+3)/4]
│             - amax_rowwise: FP32 [1]
│             - columnwise_*: similar
│
└─ Return:  NVFP4Tensor

Usage in GEMM:
Python:  tex.generic_gemm(weight_nvfp4, True, act_nvfp4, False, ...)
C++:     gemm() → gemm_nvfp4()
C++:     Setup cuBLASLt with:
           - FP4 E2M1 matrix type
           - E4M3 block scales
           - FP32 tensor scales
           - Split-K accumulation (required)
cuBLAS:  cublasLtMatmul()
GPU:     Blackwell Tensor Core FP4 GEMM
Result:  BF16 output
```

#### Blockwise FP8 (2D) for Weights

```
User:    weight_quantizer = Float8BlockQuantizer(block_scaling_dim=2)
User:    weight_fp8 = weight_quantizer.quantize(weight_bf16)

↓ quantizer.quantize()
│
├─ Python:  quantizer.quantize_impl(weight_bf16)
│  Python:  tex.quantize(weight_bf16, quantizer)
│  C++:     pybind.cpp:119 → quantize()
│  C++:     Extract:
│             - block_scaling_dim = 2
│             - force_pow_2_scales = True
│  CUDA:    fp8_block_scaling.cu::fp8_block_quantize_2d_kernel()
│             1. Each block processes 128×128 tile
│             2. Compute tile amax cooperatively
│             3. Compute FP32 scale (round to pow2 if enabled)
│             4. Store scale at [tile_row, tile_col]
│             5. Quantize tile to FP8 E4M3
│  Result:  Float8BlockwiseQTensor with:
│             - rowwise_data: uint8 [M, K]
│             - rowwise_scale_inv: FP32 [ceil(M/128), round_to_4(ceil(K/128))]
│             - columnwise_data: uint8 [K, M]
│             - columnwise_scale_inv: FP32 [ceil(K/128), round_to_4(ceil(M/128))]
│             - is_2D_scaled: True
│             - data_format: GEMM_READY
│
└─ Return:  Float8BlockwiseQTensor

Usage in GEMM:
Python:  tex.generic_gemm(weight_fp8, True, inp_fp8, False, ...)
C++:     gemm() → gemm_fp8_blockwise()
C++:     Setup cuBLASLt with:
           - FP8 E4M3 matrix type
           - FP32 block scales (transposed for cuBLAS)
           - Split-K accumulation
cuBLAS:  cublasLtMatmul()
GPU:     H100/Hopper Tensor Core FP8 GEMM with block scaling
Result:  BF16 output
```

---

## Conclusion

This architecture document has traced the complete data flow of TransformerEngine's low-precision quantization system, from user-facing Python APIs through C++ bindings to CUDA kernel implementations.

### Key Takeaways

1. **Layered Architecture**: Clean separation between Python (user API), C++ (bindings), and CUDA (kernels)
2. **Quantizer Pattern**: Builder objects decouple configuration from tensor representation
3. **Dual-Layout Storage**: Pre-computing both row-wise and column-wise layouts avoids requantization
4. **Recipe System**: Pluggable scaling strategies (delayed, current, block-based) for different use cases
5. **Hardware Integration**: Direct cuBLASLt usage for maximum Tensor Core efficiency
6. **Distributed Support**: Deep integration with FSDP, TP, SP for large-scale training

### Next Steps

For detailed test walkthroughs demonstrating these concepts in action:
- See [`test_nvfp4_walkthrough.md`](test_nvfp4_walkthrough.md) for NVFP4 test analysis
- See [`test_blockwise_fp8_walkthrough.md`](test_blockwise_fp8_walkthrough.md) for blockwise FP8 tests

For implementation details:
- Read the source files linked throughout this document
- Experiment with the examples in [`fp8_primer.ipynb`](../docs/examples/fp8_primer.ipynb)
- Consult the [TransformerEngine documentation](https://docs.nvidia.com/deeplearning/transformer-engine/)

---

**Document Info:**
- Generated by: Claude (Anthropic)
- Date: 2025
- Based on: TransformerEngine commit bd55e7ba
- Purpose: Internal architecture documentation for developers
