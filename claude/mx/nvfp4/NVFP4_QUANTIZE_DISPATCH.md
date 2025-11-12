# NVFP4 Quantize Dispatch Path: Complete Call Stack Trace

This document provides a comprehensive line-by-line walkthrough of the complete dispatch path for NVFP4 quantization in TransformerEngine, from Python API through C++ bindings to CUDA kernels.

## Table of Contents

1. [Overview](#overview)
2. [Binding Mechanism](#binding-mechanism)
3. [Complete Call Stack](#complete-call-stack)
4. [Detailed Code Walkthrough](#detailed-code-walkthrough)
5. [Kernel Dispatch Logic](#kernel-dispatch-logic)

---

## Overview

The NVFP4 quantization dispatch follows this high-level flow:

```
Python API (tex.quantize)
    ↓
PyBind11 Binding Layer
    ↓
C++ Wrapper (quantize function)
    ↓
C++ Quantizer Abstraction (NVFP4Quantizer::quantize)
    ↓
TE Core API (nvte_quantize_v2)
    ↓
Kernel Dispatcher (quantize_helper)
    ↓
CUDA Kernel (nvfp4_quantize_transpose or fallback)
```

---

## Binding Mechanism

TransformerEngine uses **PyBind11** for Python-C++ bindings, not torch.library registration.

### Key Binding File

**File:** [transformer_engine/pytorch/csrc/extensions/pybind.cpp](../../../transformer_engine/pytorch/csrc/extensions/pybind.cpp)

```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  NVTE_DECLARE_COMMON_PYBIND11_HANDLES(m)
  m.def("quantize", transformer_engine::pytorch::quantize,
        py::arg("tensor"),
        py::arg("quantizer"),
        py::arg("output") = py::none(),
        py::arg("noop") = py::none());
  // ... other bindings
}
```

**Location:** [pybind.cpp:118-121](../../../transformer_engine/pytorch/csrc/extensions/pybind.cpp#L118-L121)

This creates the `tex.quantize` function available in Python as `transformer_engine_torch.quantize`.

---

## Complete Call Stack

### 1. Python Entry Point

**File:** [transformer_engine/pytorch/tensor/nvfp4_tensor.py:174](../../../transformer_engine/pytorch/tensor/nvfp4_tensor.py#L174)

```python
tex.quantize(src, self, dst, noop_flag)
```

Where:
- `tex` = `import transformer_engine_torch as tex` ([nvfp4_tensor.py:13](../../../transformer_engine/pytorch/tensor/nvfp4_tensor.py#L13))
- `src`: Input tensor (high precision, e.g., BF16)
- `self`: NVFP4Quantizer instance
- `dst`: Output NVFP4Tensor (optional)
- `noop_flag`: Optional flag to skip quantization

### 2. PyBind11 Binding

**File:** [transformer_engine/pytorch/csrc/extensions/pybind.cpp:120-121](../../../transformer_engine/pytorch/csrc/extensions/pybind.cpp#L120-L121)

```cpp
m.def("quantize", transformer_engine::pytorch::quantize,
      py::arg("tensor"), py::arg("quantizer"),
      py::arg("output") = py::none(), py::arg("noop") = py::none());
```

This binding dispatches to the C++ `quantize` function.

### 3. C++ Wrapper Function

**File:** [transformer_engine/pytorch/csrc/extensions/cast.cpp:33-79](../../../transformer_engine/pytorch/csrc/extensions/cast.cpp#L33-L79)

```cpp
py::object quantize(const at::Tensor &tensor, py::handle quantizer,
                    const py::object &output,
                    std::optional<at::Tensor> noop_flag) {
  // Convert quantizer to C++ object
  auto quantizer_cpp = convert_quantizer(quantizer);  // Line 36

  // Convert input tensor to C++ object
  auto input_contiguous = tensor.contiguous();  // Line 39
  auto input_cpp = makeTransformerEngineTensor(input_contiguous);  // Line 40

  // Set amax if use_existing_amax = true (only valid for CS)
  bool use_existing_amax = false;
  if (detail::IsFloat8CurrentScalingQuantizers(quantizer.ptr())) {
    // ... (lines 44-51)
  }

  // Initialize output tensor
  TensorWrapper output_cpp;
  py::object output_py;
  if (output.is_none()) {
    const auto shape = get_tensor_shape(input_cpp);
    const auto fake_dtype = input_cpp.dtype();
    std::tie(output_cpp, output_py) =
        quantizer_cpp->create_tensor(shape, fake_dtype);  // Lines 57-59
  } else {
    std::tie(output_cpp, output_py) =
        quantizer_cpp->convert_and_update_tensor(output);  // Line 61
  }

  // Initialize no-op flag
  std::optional<TensorWrapper> noop_flag_cpp;
  if (noop_flag.has_value()) {
    noop_flag_cpp = makeTransformerEngineTensor(*noop_flag);  // Line 67
  }

  // Perform quantization
  if (use_existing_amax) {
    auto *quantizer_cs = dynamic_cast<Float8CurrentScalingQuantizer *>(quantizer_cpp.get());
    quantizer_cs->quantize_with_amax(input_cpp, output_cpp, noop_flag_cpp);  // Line 73
  } else {
    quantizer_cpp->quantize(input_cpp, output_cpp, noop_flag_cpp);  // Line 75
  }

  return output_py;  // Line 78
}
```

**Key Operations:**
1. **Line 36**: Convert Python quantizer to C++ quantizer object via polymorphic dispatch
2. **Lines 39-40**: Convert PyTorch tensor to TE TensorWrapper
3. **Lines 57-61**: Create or convert output tensor
4. **Line 67**: Convert noop flag if present
5. **Line 75**: Call the quantizer's `quantize` method

### 4. Quantizer Conversion

**File:** [transformer_engine/pytorch/csrc/common.cpp](../../../transformer_engine/pytorch/csrc/common.cpp)

```cpp
std::unique_ptr<Quantizer> convert_quantizer(py::handle quantizer) {
  init_extension();
  if (quantizer.is_none()) {
    return std::make_unique<NoneQuantizer>(quantizer);
  }
  for (auto [_check_type, check_quantizer_type, _create_tensor, create_quantizer] :
       detail::custom_types_converters) {
    if (check_quantizer_type(quantizer.ptr())) {
      return create_quantizer(quantizer);  // Returns NVFP4Quantizer for NVFP4
    }
  }
  NVTE_ERROR("Unexpected type for quantizer");
}
```

This function uses a registry pattern to dispatch to the correct quantizer type constructor. For NVFP4, it returns a `std::unique_ptr<NVFP4Quantizer>`.

### 5. NVFP4Quantizer::quantize

**File:** [transformer_engine/pytorch/csrc/quantizer.cpp](../../../transformer_engine/pytorch/csrc/quantizer.cpp)

The NVFP4Quantizer has two quantize methods:

```cpp
void NVFP4Quantizer::quantize(const TensorWrapper& input, TensorWrapper& out,
                              const std::optional<TensorWrapper>& noop_flag) {
  this->quantize_impl(input, out, noop_flag, true);  // compute_amax=true
}

void NVFP4Quantizer::quantize_impl(const TensorWrapper& input, TensorWrapper& out,
                                   const std::optional<TensorWrapper>& noop_flag,
                                   bool compute_amax) {
  // Nothing to be done if input is empty
  if (input.numel() == 0) {
    return;
  }

  auto stream = at::cuda::getCurrentCUDAStream();

  QuantizationConfigWrapper quant_config;
  if (noop_flag) {
    quant_config.set_noop_tensor(noop_flag->data());
  }
  quant_config.set_nvfp4_2d_quantization(this->with_2d_quantization);
  quant_config.set_stochastic_rounding(this->stochastic_rounding);

  // Setup for Random Hadamard Transform (RHT) if enabled
  size_t rows = 1;
  for (size_t i = 0; i < input.ndim() - 1; ++i) {
    rows *= input.size(i);
  }
  size_t cols = input.size(input.ndim() - 1);

  // Setup stochastic rounding RNG state if needed
  TensorWrapper te_rng_state;
  if (this->stochastic_rounding) {
    // ... initialize RNG state
    quant_config.set_rng_state(te_rng_state.data());
  }

  // Check if eligible for optimized RHT+cast fusion kernel
  bool eligible_for_rht_cast_fusion =
      input.dtype() == DType::kBFloat16 && rows % 64 == 0 && cols % 128 == 0;

  // Compute amax and quantize
  if (this->with_rht) {
    // RHT path (Hadamard transform)
    // ... (complex logic for rowwise/columnwise with RHT)
  } else {
    // Standard quantization path (most common for NVFP4)
    NVTE_SCOPED_GIL_RELEASE({
      nvte_quantize_v2(input.data(), out.data(), quant_config, stream);
    });
  }
}
```

**Key Operations:**
1. Create `QuantizationConfigWrapper` with noop flag, 2D quantization flag, and stochastic rounding settings
2. Setup RNG state if stochastic rounding is enabled
3. Call `nvte_quantize_v2` (the core TE API)

### 6. TE Core API: nvte_quantize_v2

**File:** [transformer_engine/common/util/cast.cu:57-71](../../../transformer_engine/common/util/cast.cu#L57-L71)

```cpp
void nvte_quantize_v2(const NVTETensor input, NVTETensor output,
                      const NVTEQuantizationConfig quant_config, cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize_v2);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = false;
  constexpr bool IS_DACT = false;
  constexpr bool IS_ACT = false;
  constexpr NVTETensor dbias = nullptr;
  constexpr NVTETensor workspace = nullptr;
  constexpr const NVTETensor grad = nullptr;

  detail::quantize_helper<IS_DBIAS, IS_DACT, IS_ACT, Empty, nullptr>(
      input, grad, output, dbias, workspace, quant_config, stream);
}
```

This is a thin wrapper that calls the template dispatcher `quantize_helper` with compile-time flags set for forward quantization (no bias gradient, no activation backward).

### 7. Kernel Dispatcher: quantize_helper

**File:** [transformer_engine/common/util/cast_kernels.cuh:2034-2183](../../../transformer_engine/common/util/cast_kernels.cuh#L2034-L2183)

```cpp
template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename ParamOP,
          float (*OP)(float, float) = nullptr>
void quantize_helper(const NVTETensor input, const NVTETensor grad, NVTETensor output,
                     NVTETensor dbias, NVTETensor workspace,
                     const NVTEQuantizationConfig quant_config, cudaStream_t stream) {
  // Convert to internal Tensor types
  const Tensor *input_tensor;
  const Tensor *activation_input_tensor;
  if constexpr (IS_DBIAS || IS_DACT) {
    input_tensor = convertNVTETensorCheck(grad);
    activation_input_tensor = convertNVTETensor(input);
  } else {
    input_tensor = convertNVTETensorCheck(input);  // Line 2045
    activation_input_tensor = nullptr;
  }
  auto output_tensor = convertNVTETensorCheck(output);  // Line 2048
  auto dbias_tensor = convertNVTETensor(dbias);
  auto workspace_tensor = convertNVTETensor(workspace);

  // Quantization config
  QuantizationConfig quant_config_cpp;
  if (quant_config != nullptr) {
    quant_config_cpp = *reinterpret_cast<QuantizationConfig *>(quant_config);
  }

  // Noop flag
  Tensor dummy_tensor;
  Tensor *noop_tensor = &dummy_tensor;
  if (quant_config_cpp.noop_tensor != nullptr) {
    noop_tensor = convertNVTETensorCheck(quant_config_cpp.noop_tensor);
  }

  // Check for unsupported options
  if (quant_config_cpp.stochastic_rounding) {
    NVTE_CHECK(output_tensor->scaling_mode == NVTE_NVFP4_1D_SCALING,
               "Stochastic rounding is only supported for NVFP4 quantization.");
  }

  // Dispatch to quantization kernel depending on data format
  switch (output_tensor->scaling_mode) {  // Line 2072
    case NVTE_DELAYED_TENSOR_SCALING: {
      // FP8 delayed scaling path
      // ...
      break;
    }
    case NVTE_MXFP8_1D_SCALING: {
      // MXFP8 path
      mxfp8_quantize<IS_DBIAS, IS_DACT, IS_ACT, ParamOP, OP>(...);
      break;
    }
    case NVTE_NVFP4_1D_SCALING: {  // Line 2097 - NVFP4 PATH
      // Check tensors
      CheckNoopTensor(*noop_tensor, "cast_noop");
      CheckInputTensor(*input_tensor, "input");
      CheckOutputTensor(*output_tensor, "output", false);

      // Choose kernel
      int32_t rows = input_tensor->flat_first_dim();  // Line 2104
      int32_t cols = input_tensor->flat_last_dim();   // Line 2105
      auto dtype = input_tensor->dtype();
      bool use_optimized_kernel = dtype == DType::kBFloat16 &&
                                  rows % 32 == 0 && cols % 32 == 0 &&
                                  output_tensor->has_data();  // Lines 2107-2108

      // Launch NVFP4 quantize kernel
      if (use_optimized_kernel) {  // Line 2111
        if (quant_config_cpp.nvfp4_2d_quantization) {
          nvfp4_quantize_transpose<IS_ACT, ParamOP, OP, true>(
              *input_tensor, noop_tensor, output_tensor, &quant_config_cpp, stream);
        } else {
          nvfp4_quantize_transpose<IS_ACT, ParamOP, OP, false>(
              *input_tensor, noop_tensor, output_tensor, &quant_config_cpp, stream);
        }
      } else {  // Line 2119 - Fallback kernel
        auto &global_amax = (output_tensor->amax.dptr != nullptr)
                                ? output_tensor->amax
                                : output_tensor->columnwise_amax;
        NVTE_CHECK((!IS_DBIAS && !IS_DACT && !IS_ACT),
                   "IS_DBIAS, IS_DACT, and IS_ACT not implemented for NVTE_NVFP4_1D_SCALING for 2D quantization");
        quantize_transpose_vector_blockwise_fp4(
            /*input=*/input_tensor->data,
            /*global_amax=*/global_amax,
            /*scale_inv=*/output_tensor->scale_inv,
            /*scale_inv_t=*/output_tensor->columnwise_scale_inv,
            /*output=*/output_tensor->data,
            /*output_t=*/output_tensor->columnwise_data,
            /*epsilon=*/0.0f,
            /*return_identity=*/output_tensor->has_data(),
            /*return_transpose=*/output_tensor->has_columnwise_data(),
            /*pow2_scale=*/false,
            /*swizzled_scale=*/false,
            /*use_stochastic_rounding=*/quant_config_cpp.stochastic_rounding,
            /*rng_state=*/quant_config_cpp.rng_state,
            /*use_2d_quantization=*/quant_config_cpp.nvfp4_2d_quantization,
            /*noop_tensor=*/noop_tensor->data,
            /*stream=*/stream);  // Lines 2125-2136
      }
      break;
    }
    case NVTE_BLOCK_SCALING_2D: {
      // FP8 2D block scaling
      // ...
      break;
    }
    case NVTE_BLOCK_SCALING_1D: {
      // FP8 1D block scaling
      // ...
      break;
    }
    default:
      NVTE_ERROR("Not implemented scaling mode: " + to_string(output_tensor->scaling_mode) + ".");
  }
}
```

**Key Dispatch Logic:**

1. **Line 2072**: Switch on `output_tensor->scaling_mode`
2. **Line 2097**: Match `NVTE_NVFP4_1D_SCALING` case
3. **Lines 2104-2108**: Determine kernel selection criteria:
   - `rows % 32 == 0 && cols % 32 == 0`: Alignment for optimized kernel
   - `dtype == DType::kBFloat16`: Input must be BF16
   - `output_tensor->has_data()`: Must have rowwise output
4. **Line 2111**: If eligible, use optimized `nvfp4_quantize_transpose` kernel
5. **Line 2119**: Otherwise, use fallback `quantize_transpose_vector_blockwise_fp4`

---

## Kernel Dispatch Logic

### Optimized Kernel Path: nvfp4_quantize_transpose

**File:** [transformer_engine/common/util/nvfp4_transpose.cuh](../../../transformer_engine/common/util/nvfp4_transpose.cuh)

This kernel is used when:
- Input is BF16
- Dimensions are aligned to 32x32 tiles
- Rowwise output is required

**Template Signature:**
```cpp
template <bool IS_ACT, typename ParamOP, float (*OP)(float, float), bool WITH_2D_QUANTIZATION>
void nvfp4_quantize_transpose(const Tensor &input, Tensor *noop_tensor,
                               Tensor *output, QuantizationConfig *quant_config,
                               cudaStream_t stream)
```

**Key Features:**
- Fuses quantization and transpose operations
- Uses shared memory tiling (128x128 chunks processed by 128 threads)
- Computes scaling factors per 16-element block (1D or 2D)
- Optional stochastic rounding via cuRAND
- Outputs both rowwise and columnwise (transposed) FP4 data

**Kernel Constants:**
- `SCALE_DIM = 16`: NVFP4 block size
- `CHUNK_DIM_Y = 128`, `CHUNK_DIM_X = 128`: Tile dimensions
- `THREADS_NUM = 128`: Thread block size
- `TILE_DIM_Y = 32`, `TILE_DIM_X = 128`: Processing tile size

**Location:** [nvfp4_transpose.cuh:1-1516](../../../transformer_engine/common/util/nvfp4_transpose.cuh#L1-L1516)

### Fallback Kernel Path: quantize_transpose_vector_blockwise_fp4

Used when:
- Input is not BF16
- Dimensions are not aligned
- Columnwise-only output is needed

This kernel handles arbitrary input types and shapes, but is less optimized than the fused kernel.

---

## Summary of Key Files

| Layer | File | Lines | Description |
|-------|------|-------|-------------|
| **Python API** | [nvfp4_tensor.py](../../../transformer_engine/pytorch/tensor/nvfp4_tensor.py) | 174-175 | User-facing quantize call |
| **PyBind11** | [pybind.cpp](../../../transformer_engine/pytorch/csrc/extensions/pybind.cpp) | 120-121 | Python-C++ binding |
| **C++ Wrapper** | [cast.cpp](../../../transformer_engine/pytorch/csrc/extensions/cast.cpp) | 33-79 | Argument conversion & dispatch |
| **Quantizer** | [quantizer.cpp](../../../transformer_engine/pytorch/csrc/quantizer.cpp) | Various | NVFP4Quantizer implementation |
| **TE Core API** | [cast.cu](../../../transformer_engine/common/util/cast.cu) | 57-71 | nvte_quantize_v2 entry |
| **Dispatcher** | [cast_kernels.cuh](../../../transformer_engine/common/util/cast_kernels.cuh) | 2034-2183 | quantize_helper dispatcher |
| **CUDA Kernel** | [nvfp4_transpose.cuh](../../../transformer_engine/common/util/nvfp4_transpose.cuh) | 1-1516 | Optimized NVFP4 kernel |

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ Python: tex.quantize(src, self, dst, noop_flag)            │
│ File: nvfp4_tensor.py:174                                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ PyBind11: m.def("quantize", ...)                            │
│ File: pybind.cpp:120-121                                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ C++ Wrapper: quantize(tensor, quantizer, output, noop)     │
│ File: cast.cpp:33-79                                        │
│   • convert_quantizer() → NVFP4Quantizer                   │
│   • makeTransformerEngineTensor()                          │
│   • quantizer_cpp->quantize()                              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ NVFP4Quantizer::quantize()                                  │
│ File: quantizer.cpp                                         │
│   • Setup QuantizationConfig (noop, 2D, stochastic)        │
│   • Initialize RNG state if stochastic rounding            │
│   • Call nvte_quantize_v2()                                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ nvte_quantize_v2(input, output, quant_config, stream)      │
│ File: cast.cu:57-71                                         │
│   • Thin wrapper calling quantize_helper                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ quantize_helper<IS_DBIAS=false, IS_DACT=false, ...>()     │
│ File: cast_kernels.cuh:2034-2183                           │
│   • Switch on output_tensor->scaling_mode                  │
│   • Case NVTE_NVFP4_1D_SCALING:                            │
│     - Check alignment (rows%32==0, cols%32==0)             │
│     - Check dtype (BF16)                                   │
└────────────────────┬────────────────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
┌──────────────────────┐  ┌───────────────────────────────────┐
│ Optimized Path       │  │ Fallback Path                     │
│ (BF16, aligned)      │  │ (other dtypes, unaligned)         │
├──────────────────────┤  ├───────────────────────────────────┤
│ nvfp4_quantize_      │  │ quantize_transpose_vector_        │
│ transpose<1D/2D>()   │  │ blockwise_fp4()                   │
│                      │  │                                   │
│ File:                │  │ Generic FP4 quantization kernel   │
│ nvfp4_transpose.cuh  │  │                                   │
│                      │  │                                   │
│ • 128x128 tiles      │  │                                   │
│ • Shared memory      │  │                                   │
│ • Fused cast+        │  │                                   │
│   transpose          │  │                                   │
│ • Stochastic         │  │                                   │
│   rounding (cuRAND)  │  │                                   │
│ • 16-element blocks  │  │                                   │
└──────────────────────┘  └───────────────────────────────────┘
```

---

## Key Takeaways

1. **Binding Mechanism**: TransformerEngine uses **PyBind11** for all Python-C++ bindings, not torch.library or torchscript.

2. **Polymorphic Dispatch**: The C++ layer uses virtual function dispatch through the `Quantizer` base class, allowing runtime selection of quantization strategies (FP8, MXFP8, NVFP4, etc.).

3. **Template-Based Kernel Selection**: The dispatcher uses template metaprogramming to select between different kernel variants at compile time (e.g., with/without bias gradient, with/without activation backward).

4. **Two-Tier Kernel Selection**: For NVFP4, there are two kernels:
   - **Optimized**: `nvfp4_quantize_transpose` (fused, BF16-only, aligned)
   - **Fallback**: `quantize_transpose_vector_blockwise_fp4` (general-purpose)

5. **Configuration Propagation**: Quantization configuration (noop flag, 2D quantization, stochastic rounding, RNG state) flows through all layers via `QuantizationConfig` struct.

6. **Block-Based Quantization**: NVFP4 uses 16-element blocks for computing scaling factors:
   - **1D mode**: One scale per 16 elements along rows or columns
   - **2D mode**: One scale per 16x16 block (for Blackwell GPUs with native 2D block support)

7. **Stochastic Rounding**: Optional stochastic rounding uses cuRAND for generating random bits, improving quantization accuracy at the cost of reproducibility.

---

## Next Steps

To fully understand the NVFP4 implementation, explore:

1. **CUDA Kernel Details**: [nvfp4_transpose.cuh](../../../transformer_engine/common/util/nvfp4_transpose.cuh) for the optimized kernel implementation
2. **Scale Computation**: How FP8E4M3 scales are computed for 16-element blocks
3. **Memory Layout**: How FP4 data is packed (2 elements per byte)
4. **Hadamard Transform**: Optional RHT preprocessing for improved quantization quality
5. **Integration with GEMM**: How quantized tensors flow into FP4 GEMM operations

[recipe](../../../transformer_engine/common/recipe/nvfp4.cu)
[transpose](../../../transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh)
[cast](../../../transformer_engine/common/cast/nvfp4/core_nvfp4.cuh)
[quantize](../../../transformer_engine/common/cast/nvfp4/quantize_nvfp4.cuh)
[dequantize](../../../transformer_engine/common/cast/nvfp4/dequantize_nvfp4.cuh)
[tensor](../../../transformer_engine/pytorch/tensor/nvfp4_tensor.py)
[tensor storage](../../../transformer_engine/pytorch/tensor/storage/nvfp4_tensor_storage.py)
[quantization](../../../transformer_engine/pytorch/custom_recipes/quantization_nvfp4.py)