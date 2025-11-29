# MXFP8 Quantization Call Graph

This document provides visual call graphs and function dispatch diagrams for MXFP8 quantization.

## High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      Python User Code                            │
│  quantize(input_tensor, mxfp8_quantizer)                        │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│              C++ Entry Point (cast.cpp:33-79)                    │
│  py::object quantize(tensor, quantizer, output, noop_flag)      │
└─────────────────────┬───────────────────────────────────────────┘
                      │
      ┌───────────────┼───────────────┬───────────────────┐
      │               │               │                   │
      ▼               ▼               ▼                   ▼
  ┌────────┐    ┌─────────┐    ┌──────────┐      ┌─────────────┐
  │Convert │    │Wrap     │    │Allocate  │      │Quantize     │
  │Quantizer│    │Input    │    │Output    │      │(CUDA)       │
  └────────┘    └─────────┘    └──────────┘      └─────────────┘
      │               │               │                   │
      ▼               ▼               ▼                   ▼
┌─────────┐    ┌──────────┐   ┌──────────┐      ┌──────────────┐
│MXFP8    │    │Tensor    │   │MXFP8     │      │nvte_quantize │
│Quantizer│    │Wrapper   │   │Tensor    │      │_v2 (C API)   │
└─────────┘    └──────────┘   └──────────┘      └──────────────┘
                                                         │
                                                         ▼
                                                  ┌──────────────┐
                                                  │CUDA Kernel   │
                                                  │(cast.cu)     │
                                                  └──────────────┘
```

## Detailed Call Graph

### Full Function Call Sequence

```
quantize(tensor: at::Tensor, quantizer: py::handle, ...) → py::object
├─▶ convert_quantizer(quantizer: py::handle) → std::unique_ptr<Quantizer>
│   │   Location: common.cpp:51-64
│   │
│   ├─▶ init_extension()
│   │   └─▶ init_mxfp8_extension()
│   │       └─▶ Import Python classes (MXFP8Quantizer, MXFP8Tensor)
│   │           Location: extensions/pybind.cpp:55-68
│   │
│   ├─▶ Loop through custom_types_converters
│   │   └─▶ IsMXFP8Quantizers(quantizer.ptr()) → bool
│   │       │   Location: pybind.h:61
│   │       │   Checks: Py_TYPE(obj) == MXFP8QuantizerClass
│   │       │
│   │       └─▶ [MATCH FOUND]
│   │
│   └─▶ CreateQuantizer<MXFP8Quantizer>(quantizer)
│       └─▶ MXFP8Quantizer::MXFP8Quantizer(quantizer)
│           │   Location: quantizer.cpp:899-901
│           │
│           └─▶ Quantizer::Quantizer(quantizer) [base constructor]
│               │   Location: quantizer.cpp:50-61
│               │   Extracts: rowwise_usage, columnwise_usage, internal
│               │
│               └─▶ Extract dtype from Python quantizer
│
├─▶ tensor.contiguous() → at::Tensor
│   └─▶ [PyTorch internal] Ensure memory is contiguous
│
├─▶ makeTransformerEngineTensor(input_contiguous) → TensorWrapper
│   │   Location: common.cpp:120-127
│   │
│   ├─▶ GetTransformerEngineDType(tensor.scalar_type()) → DType
│   ├─▶ Extract shape: vector<size_t>
│   └─▶ makeTransformerEngineTensor(data_ptr, shape, dtype)
│       │   Location: common.cpp:115-118
│       │
│       └─▶ TensorWrapper(data_ptr, shape, dtype)
│           └─▶ [TE Core] Initialize tensor wrapper
│
├─▶ get_tensor_shape(input_cpp) → vector<size_t>
│   │   Location: cast.cpp:26-29
│   └─▶ Convert NVTEShape → std::vector<size_t>
│
├─▶ quantizer_cpp->create_tensor(shape, fake_dtype) → pair<TensorWrapper, py::object>
│   │
│   └─▶ MXFP8Quantizer::create_tensor(shape, dtype)
│       │   Location: quantizer.cpp:905-984
│       │
│       ├─▶ Validate dimensions
│       │   └─▶ NVTE_CHECK(flat_first_dim % 32 == 0 && flat_last_dim % 32 == 0)
│       │
│       ├─▶ get_scale_shape(shape, rowwise=false) → vector<size_t>
│       │   │   Location: quantizer.cpp:1105-1134
│       │   │
│       │   └─▶ Compute: [roundup(M, 128), roundup(N/32, 4)]
│       │
│       ├─▶ get_scale_shape(shape, columnwise=true) → vector<size_t>
│       │   └─▶ Compute: [roundup(M/32, 4), roundup(N, 128)]
│       │
│       ├─▶ Allocate rowwise buffers
│       │   ├─▶ at::empty([M, N], dtype=uint8) → rowwise_data
│       │   └─▶ at::empty([M, N/32], dtype=uint8) → rowwise_scale_inv
│       │
│       ├─▶ Allocate columnwise buffers
│       │   ├─▶ at::empty([M, N], dtype=uint8) → columnwise_data
│       │   └─▶ at::empty([M/32, N], dtype=uint8) → columnwise_scale_inv
│       │
│       ├─▶ MXFP8TensorClass(...) → py::object
│       │   └─▶ [Python] Construct MXFP8Tensor with allocated buffers
│       │       Source: transformer_engine/pytorch/tensor/mxfp8_tensor.py
│       │
│       └─▶ TensorWrapper(NVTE_MXFP8_1D_SCALING)
│           ├─▶ set_rowwise_data(ptr, dtype, shape)
│           ├─▶ set_rowwise_scale_inv(ptr, DType::kFloat8E8M0, scale_shape)
│           ├─▶ set_columnwise_data(ptr, dtype, shape)
│           └─▶ set_columnwise_scale_inv(ptr, DType::kFloat8E8M0, scale_shape)
│
└─▶ quantizer_cpp->quantize(input_cpp, output_cpp, noop_flag)
    │
    └─▶ MXFP8Quantizer::quantize(input, out, noop_flag)
        │   Location: quantizer.cpp:1091-1103
        │
        ├─▶ Check: input.numel() == 0 ? return : continue
        │
        ├─▶ QuantizationConfigWrapper() → config
        │   └─▶ [Optional] config.set_noop_tensor(noop_flag->data())
        │
        └─▶ NVTE_SCOPED_GIL_RELEASE
            └─▶ nvte_quantize_v2(input.data(), out.data(), config, stream)
                │   Location: transformer_engine/common/cast/cast.cu
                │   [C API - no Python/C++ dependencies]
                │
                ├─▶ Extract tensor metadata
                │   ├─▶ nvte_tensor_dtype(input) → NVTEDType
                │   ├─▶ nvte_tensor_scaling_mode(output) → NVTEScalingMode
                │   └─▶ nvte_tensor_shape(input) → NVTEShape
                │
                ├─▶ Dispatch: if (scaling_mode == NVTE_MXFP8_1D_SCALING)
                │
                ├─▶ launch_mxfp8_quantize_rowwise_kernel<<<blocks, threads, 0, stream>>>
                │   │   [CUDA Kernel Launch]
                │   │
                │   └─▶ mxfp8_quantize_rowwise_kernel
                │       │   [Executes on GPU]
                │       │
                │       └─▶ For each 32-element block:
                │           ├─▶ Compute: max_abs = max(|block[0]|, ..., |block[31]|)
                │           ├─▶ Compute: exponent = floor(log2(max_abs)) + 127
                │           ├─▶ Compute: scale_inv = 2^(127 - exponent)
                │           ├─▶ Store: output_scales[block_idx] = exponent (E8M0)
                │           └─▶ For each element i in block:
                │               └─▶ output_data[i] = float_to_fp8_e4m3(input[i] * scale_inv)
                │
                └─▶ launch_mxfp8_quantize_columnwise_kernel<<<blocks, threads, 0, stream>>>
                    └─▶ mxfp8_quantize_columnwise_kernel
                        └─▶ [Same as rowwise, columnwise memory layout]
```

## Type Dispatch Mechanism

### Quantizer Type Resolution

```
convert_quantizer(quantizer: py::handle)
│
├─▶ custom_types_converters array (pybind.h:102-112)
│
│   Index 0: Float8Quantizer
│   ├─▶ IsFloat8Quantizers(obj) ?
│   └─▶ CreateQuantizer<Float8Quantizer>
│
│   Index 1: Float8CurrentScalingQuantizer
│   ├─▶ IsFloat8CurrentScalingQuantizers(obj) ?
│   └─▶ CreateQuantizer<Float8CurrentScalingQuantizer>
│
│   Index 2: MXFP8Quantizer  ◀── MATCHED FOR MXFP8
│   ├─▶ IsMXFP8Quantizers(obj) ? ✓ TRUE
│   └─▶ CreateQuantizer<MXFP8Quantizer>
│       └─▶ return std::make_unique<MXFP8Quantizer>(quantizer)
│
│   Index 3: Float8BlockQuantizer
│   ├─▶ IsFloat8BlockwiseQuantizers(obj) ?
│   └─▶ CreateQuantizer<Float8BlockQuantizer>
│
│   Index 4: NVFP4Quantizer
│   ├─▶ IsNVFP4Quantizers(obj) ?
│   └─▶ CreateQuantizer<NVFP4Quantizer>
│
└─▶ NVTE_ERROR("Unexpected type for quantizer")
```

### Tensor Type Checking

```
IsMXFP8Quantizers(PyObject *obj)
│   Location: pybind.h:61
│
└─▶ Py_TYPE(obj) == MXFP8QuantizerClass ?
    │
    ├─▶ MXFP8QuantizerClass initialized by:
    │   init_mxfp8_extension()
    │   └─▶ py::module_::import("transformer_engine.pytorch.tensor.mxfp8_tensor")
    │       └─▶ PyObject_GetAttrString(module, "MXFP8Quantizer")
    │           └─▶ MXFP8QuantizerClass = <Python type pointer>
    │
    └─▶ Compare pointer equality
        ├─▶ If match: return true → use MXFP8 code path
        └─▶ If no match: return false → try next converter
```

## Virtual Method Dispatch

### Polymorphic `quantize()` Call

```
quantizer_cpp->quantize(input, output, noop_flag)
│   Type: std::unique_ptr<Quantizer> (abstract base class)
│   Actual type: MXFP8Quantizer (concrete derived class)
│
└─▶ Virtual function dispatch (C++ polymorphism)
    │
    ├─▶ if (dynamic_type == Float8Quantizer)
    │   └─▶ Float8Quantizer::quantize(...)
    │       └─▶ nvte_quantize_v2(...) with delayed scaling
    │
    ├─▶ if (dynamic_type == Float8CurrentScalingQuantizer)
    │   └─▶ Float8CurrentScalingQuantizer::quantize(...)
    │       └─▶ nvte_quantize_v2(...) with current scaling
    │
    ├─▶ if (dynamic_type == MXFP8Quantizer)  ◀── DISPATCHED HERE
    │   └─▶ MXFP8Quantizer::quantize(...)
    │       └─▶ nvte_quantize_v2(...) with MXFP8 1D scaling
    │
    ├─▶ if (dynamic_type == Float8BlockQuantizer)
    │   └─▶ Float8BlockQuantizer::quantize(...)
    │       └─▶ nvte_quantize_v2(...) with block scaling
    │
    └─▶ if (dynamic_type == NVFP4Quantizer)
        └─▶ NVFP4Quantizer::quantize(...)
            └─▶ nvte_quantize_v2(...) with NVFP4 scaling
```

## CUDA Kernel Launch Flow

### From C++ to GPU

```
MXFP8Quantizer::quantize(...)
│   [C++ PyTorch Extension]
│
└─▶ NVTE_SCOPED_GIL_RELEASE
    │   Macro expands to:
    │   {
    │     pybind11::gil_scoped_release release;
    │     ... code ...
    │   }
    │
    └─▶ nvte_quantize_v2(input.data(), out.data(), config, stream)
        │   [C API - pure C interface]
        │   Location: transformer_engine/common/cast/cast.cu
        │
        ├─▶ Determine kernel parameters
        │   ├─▶ Grid size: (num_blocks, 1, 1)
        │   ├─▶ Block size: (threads_per_block, 1, 1)
        │   └─▶ Shared memory: 0 (not used for MXFP8)
        │
        ├─▶ Kernel launch (rowwise)
        │   mxfp8_quantize_rowwise_kernel<<<grid, block, 0, stream>>>(
        │       input_ptr, output_data_ptr, output_scales_ptr,
        │       M, N, MXFP8_BLOCK_SIZE
        │   )
        │   │
        │   └─▶ [GPU Execution]
        │       ├─▶ Each thread processes one 32-element block
        │       ├─▶ Warp reduction for max_abs
        │       ├─▶ Compute shared exponent
        │       └─▶ Vectorized quantization (32 elements in parallel)
        │
        └─▶ Kernel launch (columnwise)
            mxfp8_quantize_columnwise_kernel<<<grid, block, 0, stream>>>(...)
            └─▶ [GPU Execution with columnwise memory access]
```

## Memory Flow

### Buffer Allocation and Usage

```
┌─────────────────────────────────────────────────────────────┐
│  Python Request: quantize(input, quantizer)                 │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │  Input Tensor (GPU Memory)    │
        │  [1024, 2048] float32         │
        │  Size: 8,388,608 bytes        │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────────────────┐
        │  MXFP8Quantizer::create_tensor()         │
        │                                           │
        │  Allocates 4 buffers:                    │
        │  ┌─────────────────────────────────────┐ │
        │  │ rowwise_data: [1024, 2048] uint8   │ │
        │  │   Size: 2,097,152 bytes             │ │
        │  └─────────────────────────────────────┘ │
        │  ┌─────────────────────────────────────┐ │
        │  │ rowwise_scales: [1024, 64] uint8   │ │
        │  │   Size: 65,536 bytes                │ │
        │  └─────────────────────────────────────┘ │
        │  ┌─────────────────────────────────────┐ │
        │  │ columnwise_data: [1024, 2048] uint8│ │
        │  │   Size: 2,097,152 bytes             │ │
        │  └─────────────────────────────────────┘ │
        │  ┌─────────────────────────────────────┐ │
        │  │ columnwise_scales: [32, 2048] uint8│ │
        │  │   Size: 65,536 bytes                │ │
        │  └─────────────────────────────────────┘ │
        │                                           │
        │  Total: 4,325,376 bytes (51% of input)   │
        └───────────────┬───────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────────────┐
        │  CUDA Kernel Execution                │
        │                                        │
        │  Read:  input [1024, 2048] float32    │
        │  Write: rowwise_data [1024, 2048]     │
        │  Write: rowwise_scales [1024, 64]     │
        │  Write: columnwise_data [1024, 2048]  │
        │  Write: columnwise_scales [32, 2048]  │
        └───────────────┬───────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────────────┐
        │  Return: MXFP8Tensor                  │
        │                                        │
        │  Wraps all 4 buffers                  │
        │  User sees: shape [1024, 2048]        │
        └───────────────────────────────────────┘
```

## Thread Execution Model

### GPU Parallelization

```
Input Tensor: [1024, 2048] = 2,097,152 elements
Block Size: 32 elements
Number of Blocks: 2,097,152 / 32 = 65,536 blocks

┌────────────────────────────────────────────────────────┐
│  CUDA Grid Configuration                               │
│                                                         │
│  Threads per block: 256                                │
│  Blocks per grid: ceil(65,536 / 256) = 256            │
│                                                         │
│  Each thread processes:                                │
│  - Read 32 input values (float32)                     │
│  - Compute max_abs (warp reduction)                   │
│  - Compute shared exponent (E8M0)                     │
│  - Quantize 32 values to FP8                          │
│  - Write 32 FP8 values + 1 E8M0 scale                 │
│                                                         │
│  Total threads active: 256 × 256 = 65,536             │
│  GPU utilization: High (one thread per block)         │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│  Warp-Level Execution (32 threads = 1 warp)           │
│                                                         │
│  Thread 0: Block [0:32]    → Quantize + Scale         │
│  Thread 1: Block [32:64]   → Quantize + Scale         │
│  Thread 2: Block [64:96]   → Quantize + Scale         │
│  ...                                                   │
│  Thread 31: Block [992:1024] → Quantize + Scale       │
│                                                         │
│  Warps execute in lockstep (SIMT)                     │
│  Memory coalescing: Each warp accesses 1KB            │
└────────────────────────────────────────────────────────┘
```

## Function Cross-Reference Table

| Function | Location | Purpose | Returns |
|----------|----------|---------|---------|
| `quantize` | [`cast.cpp:33-79`](../../../transformer_engine/pytorch/csrc/extensions/cast.cpp#L33-L79) | Main entry point | `py::object` (MXFP8Tensor) |
| `convert_quantizer` | [`common.cpp:51-64`](../../../transformer_engine/pytorch/csrc/common.cpp#L51-L64) | Dispatch to concrete quantizer | `unique_ptr<Quantizer>` |
| `MXFP8Quantizer::MXFP8Quantizer` | [`quantizer.cpp:899-901`](../../../transformer_engine/pytorch/csrc/quantizer.cpp#L899-L901) | Constructor | - |
| `makeTransformerEngineTensor` | [`common.cpp:120-127`](../../../transformer_engine/pytorch/csrc/common.cpp#L120-L127) | Wrap PyTorch tensor | `TensorWrapper` |
| `MXFP8Quantizer::create_tensor` | [`quantizer.cpp:905-984`](../../../transformer_engine/pytorch/csrc/quantizer.cpp#L905-L984) | Allocate output buffers | `pair<TensorWrapper, py::object>` |
| `MXFP8Quantizer::get_scale_shape` | [`quantizer.cpp:1105-1134`](../../../transformer_engine/pytorch/csrc/quantizer.cpp#L1105-L1134) | Compute scale buffer size | `vector<size_t>` |
| `MXFP8Quantizer::quantize` | [`quantizer.cpp:1091-1103`](../../../transformer_engine/pytorch/csrc/quantizer.cpp#L1091-L1103) | Invoke quantization | `void` |
| `nvte_quantize_v2` | [`cast.cu`](../../../transformer_engine/common/cast/cast.cu) | C API, dispatch to CUDA | `void` |
| `mxfp8_quantize_kernel` | [`cast.cu`](../../../transformer_engine/common/cast/cast.cu) | CUDA kernel | `void` |

## Related Documentation

- [QUANTIZE_FRAME_BY_FRAME.md](QUANTIZE_FRAME_BY_FRAME.md) - Detailed step-by-step trace
- [DESIGN_RATIONALE.md](DESIGN_RATIONALE.md) - Architecture decisions
- [README.md](README.md) - Overview and quick start
