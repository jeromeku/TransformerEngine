# MXFP8 Quantize Dispatch Path: Complete Call Stack Trace

This document provides a comprehensive line-by-line walkthrough of the complete dispatch path for MXFP8 quantization in TransformerEngine, from Python API through C++ bindings to CUDA kernels.

## Table of Contents

1. [Overview](#overview)
2. [Binding Mechanism](#binding-mechanism)
3. [Complete Call Stack](#complete-call-stack)
4. [Detailed Code Walkthrough](#detailed-code-walkthrough)
5. [Kernel Dispatch Logic](#kernel-dispatch-logic)
6. [Comparison with NVFP4](#comparison-with-nvfp4)

---

## Overview

The MXFP8 quantization dispatch follows this high-level flow:

```
Python API (tex.quantize)
    ↓
PyBind11 Binding Layer
    ↓
C++ Wrapper (quantize function)
    ↓
C++ Quantizer Abstraction (MXFP8Quantizer::quantize)
    ↓
TE Core API (nvte_quantize_v2)
    ↓
Kernel Dispatcher (quantize_helper)
    ↓
CUDA Kernel (mxfp8_quantize - single path!)
```

**Key Difference from NVFP4**: MXFP8 has a much simpler dispatch with only one kernel path and no preprocessing steps like Random Hadamard Transform.

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

**Same binding is used for MXFP8 and NVFP4** - the dispatch to different implementations happens in C++ based on quantizer type.

---

## Complete Call Stack

### 1. Python Entry Point

**File:** [transformer_engine/pytorch/tensor/mxfp8_tensor.py:139-142](../../../transformer_engine/pytorch/tensor/mxfp8_tensor.py#L139-L142)

```python
def quantize_impl(self, tensor: torch.Tensor) -> QuantizedTensor:
    """Quantize tensor implementation"""
    # This invokes the C++ binding
    return tex.quantize(tensor, self)
```

Where:
- `tex` = `import transformer_engine_torch as tex` ([mxfp8_tensor.py:16](../../../transformer_engine/pytorch/tensor/mxfp8_tensor.py#L16))
- `tensor`: Input tensor (high precision, e.g., BF16)
- `self`: MXFP8Quantizer instance

**Key Difference from NVFP4**: MXFP8 has no `dst` or `noop_flag` parameters - simpler API!

### 2. PyBind11 Binding

**File:** [transformer_engine/pytorch/csrc/extensions/pybind.cpp:120-121](../../../transformer_engine/pytorch/csrc/extensions/pybind.cpp#L120-L121)

```cpp
m.def("quantize", transformer_engine::pytorch::quantize,
      py::arg("tensor"), py::arg("quantizer"),
      py::arg("output") = py::none(), py::arg("noop") = py::none());
```

This binding dispatches to the C++ `quantize` function.

**Same binding for all quantizer types** (FP8, MXFP8, NVFP4).

### 3. C++ Wrapper Function

**File:** [transformer_engine/pytorch/csrc/extensions/cast.cpp:33-79](../../../transformer_engine/pytorch/csrc/extensions/cast.cpp#L33-L79)

```cpp
py::object quantize(const at::Tensor &tensor, py::handle quantizer,
                    const py::object &output,
                    std::optional<at::Tensor> noop_flag) {
  // Convert quantizer to C++ object
  auto quantizer_cpp = convert_quantizer(quantizer);  // Line 36
  // Returns MXFP8Quantizer for MXFP8

  // Convert input tensor to C++ object
  auto input_contiguous = tensor.contiguous();  // Line 39
  auto input_cpp = makeTransformerEngineTensor(input_contiguous);  // Line 40

  // Set amax if use_existing_amax = true (only valid for CS, not MXFP8)
  bool use_existing_amax = false;
  if (detail::IsFloat8CurrentScalingQuantizers(quantizer.ptr())) {
    // ... (lines 44-51)
    // MXFP8 skips this - it's stateless, no amax tracking!
  }

  // Initialize output tensor
  TensorWrapper output_cpp;
  py::object output_py;
  if (output.is_none()) {
    const auto shape = get_tensor_shape(input_cpp);
    const auto fake_dtype = input_cpp.dtype();
    std::tie(output_cpp, output_py) =
        quantizer_cpp->create_tensor(shape, fake_dtype);  // Lines 57-59
    // For MXFP8: Creates MXFP8Tensor with rowwise/columnwise storage
  } else {
    std::tie(output_cpp, output_py) =
        quantizer_cpp->convert_and_update_tensor(output);  // Line 61
  }

  // Initialize no-op flag (optional, not commonly used for MXFP8)
  std::optional<TensorWrapper> noop_flag_cpp;
  if (noop_flag.has_value()) {
    noop_flag_cpp = makeTransformerEngineTensor(*noop_flag);  // Line 67
  }

  // Perform quantization
  if (use_existing_amax) {
    // Not used for MXFP8 (stateless)
    auto *quantizer_cs = dynamic_cast<Float8CurrentScalingQuantizer *>(quantizer_cpp.get());
    quantizer_cs->quantize_with_amax(input_cpp, output_cpp, noop_flag_cpp);  // Line 73
  } else {
    // MXFP8 takes this path
    quantizer_cpp->quantize(input_cpp, output_cpp, noop_flag_cpp);  // Line 75
  }

  return output_py;  // Line 78
}
```

**Key Operations:**
1. **Line 36**: Convert Python quantizer to C++ quantizer object (polymorphic dispatch)
2. **Lines 39-40**: Convert PyTorch tensor to TE TensorWrapper
3. **Lines 57-59**: Create MXFP8Tensor with allocated storage
4. **Line 75**: Call the quantizer's `quantize` method

**Key Difference from NVFP4**: No amax tracking logic, simpler output tensor creation.

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
      return create_quantizer(quantizer);  // Returns MXFP8Quantizer for MXFP8
    }
  }
  NVTE_ERROR("Unexpected type for quantizer");
}
```

This function uses a registry pattern to dispatch to the correct quantizer type constructor. For MXFP8, it returns a `std::unique_ptr<MXFP8Quantizer>`.

### 5. MXFP8Quantizer::quantize

**File:** [transformer_engine/pytorch/csrc/quantizer.cpp:1091-1103](../../../transformer_engine/pytorch/csrc/quantizer.cpp#L1091-L1103)

```cpp
void MXFP8Quantizer::quantize(const TensorWrapper& input, TensorWrapper& out,
                               const std::optional<TensorWrapper>& noop_flag) {
  // Nothing to be done if input is empty
  if (input.numel() == 0) {
    return;
  }

  auto stream = at::cuda::getCurrentCUDAStream();

  // Simple config setup - no complex features!
  QuantizationConfigWrapper quant_config;
  if (noop_flag) {
    quant_config.set_noop_tensor(noop_flag->data());
  }

  // No RHT setup
  // No stochastic rounding
  // No 2D quantization flag
  // No amax reduction group

  // Direct quantization call - no preprocessing!
  NVTE_SCOPED_GIL_RELEASE({
    nvte_quantize_v2(input.data(), out.data(), quant_config, stream);
  });
}
```

**Key Difference from NVFP4Quantizer**:
- No RHT (Random Hadamard Transform) setup
- No stochastic rounding configuration
- No 2D quantization flag
- No RNG state initialization
- No eligibility checks for fusion kernels
- **Much simpler: ~15 lines vs NVFP4's ~100+ lines**

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

**Same entry point for MXFP8 and NVFP4** - dispatch happens in `quantize_helper`.

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

  // Dispatch to quantization kernel depending on data format
  switch (output_tensor->scaling_mode) {  // Line 2072
    case NVTE_DELAYED_TENSOR_SCALING: {
      // FP8 delayed scaling path
      // ...
      break;
    }
    case NVTE_MXFP8_1D_SCALING: {  // ◄── MXFP8 PATH
      // Check tensors
      CheckNoopTensor(*noop_tensor, "cast_noop");
      CheckInputTensor(*input_tensor, "input");
      CheckOutputTensor(*output_tensor, "output", false);

      // Launch MXFP8 quantize kernel (single path!)
      mxfp8_quantize<IS_DBIAS, IS_DACT, IS_ACT, ParamOP, OP>(
          *input_tensor, activation_input_tensor, noop_tensor,
          output_tensor, dbias_tensor, workspace_tensor, stream);

      break;
    }
    case NVTE_NVFP4_1D_SCALING: {
      // NVFP4 path (complex with multiple kernel variants)
      // ...
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

**Key Dispatch Logic for MXFP8:**

1. **Line 2072**: Switch on `output_tensor->scaling_mode`
2. **Match `NVTE_MXFP8_1D_SCALING` case**
3. **No kernel selection logic** - only one kernel path!
4. **No alignment checks** - kernel handles all cases
5. **No dtype checks** - kernel supports all dtypes
6. **Direct dispatch to `mxfp8_quantize` template**

**Key Difference from NVFP4**:
- NVFP4 has 2 kernel paths (optimized + fallback)
- NVFP4 checks alignment (rows%32==0, cols%32==0)
- NVFP4 checks dtype (BF16 for optimized path)
- MXFP8 has 1 kernel path that handles everything

---

## Kernel Dispatch Logic

### MXFP8 Kernel: mxfp8_quantize

**File:** [transformer_engine/common/util/cast_kernels.cuh](../../../transformer_engine/common/util/cast_kernels.cuh)

This is the ONLY kernel path for MXFP8 quantization.

**Template Signature:**
```cpp
template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename ParamOP,
          float (*OP)(float, float)>
void mxfp8_quantize(const Tensor &input, const Tensor *activation_input,
                    Tensor *noop_tensor, Tensor *output, Tensor *dbias,
                    Tensor *workspace, cudaStream_t stream)
```

**Key Features:**
- Single unified kernel for all cases
- Handles all input dtypes (BF16, FP32, FP16, etc.)
- Supports both aligned and unaligned dimensions
- Computes E8M0 scales (power-of-2 only)
- Block size: 32 elements (fixed)
- No stochastic rounding
- No RHT preprocessing
- Outputs both rowwise and columnwise quantizations

**Kernel Constants:**
- `MXFP8_BLOCK_SIZE = 32`: Block size (fixed, not configurable)
- E8M0 scale format: 1 byte per 32-element block

**CUDA Kernel Implementation:**

**File:** [transformer_engine/common/cast/mxfp8/quantize_mxfp8.cuh:43-538](../../../transformer_engine/common/cast/mxfp8/quantize_mxfp8.cuh#L43-L538)

Key operations:
1. **Divide input into 32-element blocks**
2. **Compute amax per block**: `amax = max(abs(block[0..31]))`
3. **Encode E8M0 scale**:
   ```cpp
   // Power-of-2 scale: scale = 2^exponent
   int exponent = floor(log2(amax / 448.0))  // 448.0 = FP8 E4M3 max
   uint8_t scale_byte = exponent + 127       // Biased exponent
   ```
4. **Quantize to FP8 E4M3**:
   ```cpp
   float scale = 2^exponent
   fp8_value = round(input_value / scale)
   ```
5. **Store data and scales**:
   - Data: 1 byte per element (FP8 E4M3)
   - Scales: 1 byte per 32-element block (E8M0)

**No conditional branches based on features** - single code path!

---

## Comparison: MXFP8 vs NVFP4 Dispatch

### Dispatch Complexity

| Aspect | MXFP8 | NVFP4 |
|--------|-------|-------|
| **Kernel paths** | 1 | 2 (optimized + fallback) |
| **Preprocessing steps** | 0 | 1-2 (RHT, RNG setup) |
| **Alignment checks** | None | Yes (32x32 tiles) |
| **Dtype checks** | None | Yes (BF16 for optimized) |
| **Kernel selection logic** | None | Complex (if/else) |
| **Configuration flags** | 1 (noop) | 4 (noop, 2D, SR, RHT) |
| **Quantizer LOC** | ~15 | ~100+ |
| **Kernel variants** | 1 | 4+ (1D/2D, with/without SR) |

### Call Stack Depth

**MXFP8**:
```
Python: tex.quantize(tensor, quantizer)
  ↓
C++: quantize(tensor, quantizer, output, noop)
  ↓
MXFP8Quantizer::quantize(input, out, noop)
  ↓
nvte_quantize_v2(input, output, config, stream)
  ↓
quantize_helper() - switch on scaling_mode
  ↓
mxfp8_quantize() - single kernel
  ↓
CUDA: quantize_mxfp8 kernel
```

**NVFP4**:
```
Python: tex.quantize(src, quantizer, dst, noop)
  ↓
C++: quantize(tensor, quantizer, output, noop)
  ↓
NVFP4Quantizer::quantize_impl(input, out, noop, compute_amax)
  ├─ Setup RNG state for stochastic rounding
  ├─ Check RHT eligibility
  ├─ Optional: RHT preprocessing kernel
  ↓
nvte_quantize_v2(input, output, config, stream)
  ↓
quantize_helper() - switch on scaling_mode
  ├─ Check alignment (rows%32, cols%32)
  ├─ Check dtype (BF16)
  ├─ Check 2D quantization flag
  ↓
If optimized:
  nvfp4_quantize_transpose<1D/2D>(...)
    ↓
    CUDA: Fused quantize+transpose kernel
Else fallback:
  quantize_transpose_vector_blockwise_fp4(...)
    ↓
    CUDA: Generic FP4 quantization kernel
```

**MXFP8 is ~3-5× simpler!**

### Feature Comparison

| Feature | MXFP8 | NVFP4 |
|---------|-------|-------|
| **Block size** | 32 (fixed) | 16 (1D), 16×16 (2D) |
| **Scale format** | E8M0 (power-of-2) | E4M3 + FP32 (2-level) |
| **Random Hadamard Transform** | No | Yes (optional) |
| **Stochastic rounding** | No | Yes (optional) |
| **2D quantization** | No | Yes (for weights) |
| **Kernel fusion** | No | Yes (cast+transpose) |
| **Amax tracking** | No (stateless) | Yes (per-layer) |
| **RNG state** | No | Yes (for SR) |

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ Python: tex.quantize(tensor, self)                          │
│ File: mxfp8_tensor.py:139-142                               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ PyBind11: m.def("quantize", ...)                            │
│ File: pybind.cpp:120-121                                    │
│ (Same binding for FP8, MXFP8, NVFP4)                       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ C++ Wrapper: quantize(tensor, quantizer, output, noop)     │
│ File: cast.cpp:33-79                                        │
│   • convert_quantizer() → MXFP8Quantizer                   │
│   • makeTransformerEngineTensor()                          │
│   • quantizer_cpp->quantize()                              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ MXFP8Quantizer::quantize()                                  │
│ File: quantizer.cpp:1091-1103                               │
│   • Setup QuantizationConfig (noop only)                   │
│   • No RHT, no SR, no 2D                                   │
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
│   • Case NVTE_MXFP8_1D_SCALING:                            │
│     - No alignment checks                                  │
│     - No dtype checks                                      │
│     - Direct call to mxfp8_quantize()                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ mxfp8_quantize<IS_DBIAS, IS_DACT, IS_ACT, ...>()          │
│ File: cast_kernels.cuh                                      │
│   • Single unified kernel path                             │
│   • Handles all dtypes and shapes                          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ CUDA Kernel: quantize_mxfp8                                 │
│ File: quantize_mxfp8.cuh:43-538                            │
│                                                             │
│ • Divide into 32-element blocks                            │
│ • Compute amax per block                                   │
│ • Encode E8M0 scale (power-of-2):                          │
│   - exponent = floor(log2(amax/448))                       │
│   - scale_byte = exponent + 127                            │
│ • Quantize to FP8 E4M3:                                    │
│   - fp8_value = round(input / 2^exponent)                  │
│ • Store data (FP8) and scales (E8M0)                       │
│                                                             │
│ No preprocessing, no conditional branches!                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Summary of Key Files

| Layer | File | Lines | Description |
|-------|------|-------|-------------|
| **Python API** | [mxfp8_tensor.py](../../../transformer_engine/pytorch/tensor/mxfp8_tensor.py) | 139-142 | User-facing quantize call |
| **PyBind11** | [pybind.cpp](../../../transformer_engine/pytorch/csrc/extensions/pybind.cpp) | 120-121 | Python-C++ binding |
| **C++ Wrapper** | [cast.cpp](../../../transformer_engine/pytorch/csrc/extensions/cast.cpp) | 33-79 | Argument conversion & dispatch |
| **Quantizer** | [quantizer.cpp](../../../transformer_engine/pytorch/csrc/quantizer.cpp) | 1091-1103 | MXFP8Quantizer implementation |
| **TE Core API** | [cast.cu](../../../transformer_engine/common/util/cast.cu) | 57-71 | nvte_quantize_v2 entry |
| **Dispatcher** | [cast_kernels.cuh](../../../transformer_engine/common/util/cast_kernels.cuh) | 2034-2183 | quantize_helper dispatcher |
| **CUDA Kernel** | [quantize_mxfp8.cuh](../../../transformer_engine/common/cast/mxfp8/quantize_mxfp8.cuh) | 43-538 | MXFP8 quantization kernel |

---

## E8M0 Scale Encoding/Decoding

### Encoding (Quantization Time)

```cpp
// Input: amax = max(abs(block[0..31]))
// Output: scale_byte (uint8)

// FP8 E4M3 max value
constexpr float FP8_E4M3_MAX = 448.0f;

// Compute exponent for power-of-2 scale
float scale = amax / FP8_E4M3_MAX;
int exponent = floor(log2(scale));

// Encode as biased exponent
uint8_t scale_byte = exponent + 127;

// Store scale_byte (1 byte per 32-element block)
```

**Example**:
```
amax = 64.0
scale = 64.0 / 448.0 = 0.1428...
exponent = floor(log2(0.1428)) = floor(-2.807) = -3
scale_byte = -3 + 127 = 124

Actual scale used: 2^(-3) = 0.125
```

### Decoding (Dequantization Time)

```cpp
// Input: scale_byte (uint8)
// Output: scale (float, power-of-2)

// Decode biased exponent
int exponent = scale_byte - 127;

// Compute power-of-2 scale
float scale = exp2(exponent);  // = 2^exponent

// Dequantize: value = fp8_value * scale
```

**Example**:
```
scale_byte = 124
exponent = 124 - 127 = -3
scale = 2^(-3) = 0.125

fp8_value = 100 (FP8 E4M3)
dequantized_value = 100 * 0.125 = 12.5
```

### Why E8M0 (Power-of-2)?

1. **Simple encoding**: Just compute log2 and add bias
2. **Fast decoding**: exp2(x) is very efficient (bit shift)
3. **No mantissa storage**: Saves memory (1 byte vs 2 bytes)
4. **Hardware-friendly**: Power-of-2 multiply is fast
5. **Good accuracy**: For block-wise scaling, power-of-2 is sufficient

---

## Key Takeaways

1. **Single Kernel Path**: MXFP8 has only one kernel, no optimized/fallback variants like NVFP4.

2. **No Preprocessing**: No Random Hadamard Transform, no stochastic rounding setup.

3. **Simpler Dispatch**: Direct call to kernel, no alignment/dtype checks.

4. **Stateless Design**: No amax tracking, scales computed per-call.

5. **Power-of-2 Scales**: E8M0 format simplifies encoding/decoding vs NVFP4's E4M3+FP32.

6. **Fixed Block Size**: 32 elements (not configurable like NVFP4's 16).

7. **Unified Code Path**: Same kernel handles all dtypes, shapes, and features.

8. **~3-5× Less Code**: Quantizer implementation is ~15 lines vs NVFP4's ~100+ lines.

9. **Same Binding Layer**: PyBind11 binding is shared across all quantizer types.

10. **Polymorphic Dispatch**: C++ uses virtual function dispatch through `Quantizer` base class.

---

## Next Steps

To fully understand the MXFP8 implementation, explore:

1. **CUDA Kernel Details**: [quantize_mxfp8.cuh](../../../transformer_engine/common/cast/mxfp8/quantize_mxfp8.cuh) for the kernel implementation
2. **E8M0 Scale Computation**: How power-of-2 scales are computed and encoded
3. **Memory Layout**: How FP8 data and E8M0 scales are stored
4. **Integration with GEMM**: How quantized tensors flow into MXFP8 GEMM operations
5. **Dequantization**: How E8M0 scales are applied during GEMM execution

---

**Related Documents**:
- [AUTOCAST_FRAME_BY_FRAME.md](AUTOCAST_FRAME_BY_FRAME.md) - Detailed execution trace
- [MXFP8_LINEAR_CALL_PATH.md](MXFP8_LINEAR_CALL_PATH.md) - te.Linear call path
- [README.md](README.md) - Complete MXFP8 reference guide
- [NVFP4_QUANTIZE_DISPATCH.md](../nvfp4/NVFP4_QUANTIZE_DISPATCH.md) - NVFP4 comparison
