# MXFP8 Quantization: Design Rationale

This document explains the architectural decisions, trade-offs, and design patterns used in the MXFP8 quantization implementation.

## Table of Contents

1. [MXFP8 Format Design](#mxfp8-format-design)
2. [Software Architecture](#software-architecture)
3. [Memory Layout Strategy](#memory-layout-strategy)
4. [Performance Optimizations](#performance-optimizations)
5. [API Design Principles](#api-design-principles)
6. [Trade-off Analysis](#trade-off-analysis)

---

## MXFP8 Format Design

### Why Block-Wise Quantization?

**Problem**: Per-tensor quantization loses too much precision for diverse value distributions within a tensor.

**Solution**: Block-wise quantization with 32-element blocks.

| Approach | Granularity | Scale Overhead | Precision | Hardware Efficiency |
|----------|-------------|----------------|-----------|---------------------|
| Per-tensor | 1 scale per tensor | ~0% | Low | Excellent |
| **Per-block (32)** | **1 scale per 32 elements** | **~3%** | **High** | **Excellent** |
| Per-element | 1 scale per element | 100% | Highest | Poor |

**Design Decision**: 32-element blocks
- **Hardware alignment**: NVIDIA GPU warp size = 32 threads
- **SIMD efficiency**: One warp processes one block in lockstep
- **Storage**: Only 3% overhead (1 byte scale / 32 bytes data)
- **Precision**: Fine enough for activation/weight distributions

### Why E8M0 (Exponent-Only) Scales?

**Alternatives Considered**:

| Format | Bits | Representable Scales | Storage | Speed |
|--------|------|---------------------|---------|-------|
| FP32 | 32 | Any real number | 4 bytes | Slow (FP multiply) |
| FP16 | 16 | Limited range | 2 bytes | Medium (FP multiply) |
| **E8M0** | **8** | **Powers of 2 only** | **1 byte** | **Fast (bit shift)** |
| INT8 | 8 | Integer multiples | 1 byte | Fast (INT multiply) |

**Design Decision**: E8M0 (8-bit exponent, 0-bit mantissa)

**Rationale**:
```cpp
// E8M0 value: 127 represents 2^0 = 1.0
// Scale = 2^(E8M0_value - 127)

// Quantization (bit shift, not multiply):
quantized = input >> (127 - E8M0_value)

// Dequantization (bit shift, not multiply):
output = quantized << (127 - E8M0_value)
```

**Advantages**:
1. **Hardware speed**: Bit shifts are faster than FP multiply on GPU
2. **Storage**: 1 byte per scale (minimal overhead)
3. **Range**: Covers 2^-127 to 2^127 (sufficient for NNs)
4. **Stability**: No rounding errors from FP arithmetic

**Trade-off**: Can only represent power-of-2 scales (acceptable for most NN distributions)

### Why FP8 E4M3 Data Format?

**Alternatives**:

| Format | Bits | Exponent | Mantissa | Range | Precision |
|--------|------|----------|----------|-------|-----------|
| FP32 | 32 | 8 | 23 | 10^-38 to 10^38 | High |
| FP16 | 16 | 5 | 10 | 10^-8 to 10^4 | Medium |
| BF16 | 16 | 8 | 7 | 10^-38 to 10^38 | Medium |
| **FP8 E4M3** | **8** | **4** | **3** | **-240 to 240** | **Low (adequate)** |
| FP8 E5M2 | 8 | 5 | 2 | -57344 to 57344 | Lower |

**Design Decision**: FP8 E4M3 for most cases (E5M2 for gradients)

**Rationale**:
- **E4M3**: Better precision (3-bit mantissa) for activations/weights
- **E5M2**: Better range (5-bit exponent) for gradients (wider distribution)
- **Memory**: 4× reduction vs FP32 (critical for large models)

**Trade-off**: Precision loss acceptable for forward pass (training uses FP32 accumulation)

---

## Software Architecture

### Polymorphic Quantizer Pattern

**Design**: Abstract base class with virtual methods

```cpp
class Quantizer {
 public:
  virtual void quantize(const TensorWrapper& input, TensorWrapper& out, ...) = 0;
  virtual pair<TensorWrapper, py::object> create_tensor(...) const = 0;
  virtual NVTEScalingMode get_scaling_mode() const = 0;
  // ...
};

class MXFP8Quantizer : public Quantizer {
  // MXFP8-specific implementation
};
```

**Benefits**:
1. **Extensibility**: Add new quantizer types without modifying entry point
2. **Type safety**: Compile-time dispatch to correct implementation
3. **Code reuse**: Generic code works for all quantizers
4. **Maintainability**: Each quantizer is self-contained

**Alternative Rejected**: Switch statement on quantizer type
```cpp
// ❌ REJECTED: Would need modification for each new type
if (quantizer.type == "mxfp8") {
  // MXFP8 code
} else if (quantizer.type == "fp8") {
  // FP8 code
} // ... and so on
```

### Type Dispatch Table

**Design**: `custom_types_converters` array for Python type → C++ class mapping

**Location**: [`../../../transformer_engine/pytorch/csrc/pybind.h:102-112`](../../../transformer_engine/pytorch/csrc/pybind.h#L102-L112)

```cpp
constexpr std::array custom_types_converters = {
  std::make_tuple(IsMXFP8Tensor, IsMXFP8Quantizers,
                  NVTETensorFromMXFP8Tensor, CreateQuantizer<MXFP8Quantizer>),
  // ... other types
};
```

**Benefits**:
1. **Data-driven**: No hardcoded if/else chains
2. **Compile-time**: Array is `constexpr` (zero runtime cost)
3. **Extensible**: New types just add array entry
4. **Symmetric**: Handles both quantizer and tensor types

**Alternative Rejected**: Factory pattern with registration
```cpp
// ❌ REJECTED: Runtime overhead, complex initialization
QuantizerFactory::register("MXFP8", [](...) { return new MXFP8Quantizer(...); });
```

### Three-Layer Architecture

**Design**: Separation of concerns across language boundaries

```
┌──────────────────────────────────────┐
│  Python Layer (User API)             │
│  - High-level interface              │
│  - Type checking                     │
│  - Documentation                     │
└─────────────┬────────────────────────┘
              │ pybind11
┌─────────────▼────────────────────────┐
│  C++ Layer (Glue Code)               │
│  - Type conversion                   │
│  - Memory management                 │
│  - Dispatch logic                    │
└─────────────┬────────────────────────┘
              │ C API
┌─────────────▼────────────────────────┐
│  CUDA Layer (Computation)            │
│  - Kernels                           │
│  - Pure CUDA/C (no framework deps)   │
└──────────────────────────────────────┘
```

**Benefits**:
1. **Framework independence**: CUDA layer works with JAX, TensorFlow, etc.
2. **Compilation speed**: CUDA doesn't recompile when Python/C++ changes
3. **Binary stability**: C API provides stable ABI
4. **Testability**: Each layer can be tested independently

**Alternative Rejected**: Monolithic Python extension
```cpp
// ❌ REJECTED: Hard to test, framework-locked
PYBIND11_MODULE(quantize, m) {
  m.def("quantize", [](py::array input) {
    // Directly call CUDA from Python binding (tight coupling)
  });
}
```

---

## Memory Layout Strategy

### Dual Layout: Rowwise + Columnwise

**Problem**: Matrix operations need different data layouts
- **A×B**: A needs rowwise layout
- **A×B^T**: A needs columnwise layout (or B needs transpose)

**Design Decision**: Store both layouts upfront

```
Input: [M, N]

Rowwise Layout:
  Data:   [M, N] in row-major order
  Scales: [M, N/32] - one scale per 32 elements in last dim

Columnwise Layout:
  Data:   [M, N] in row-major order (logically column-major)
  Scales: [M/32, N] - one scale per 32 elements in first dim
```

**Cost**: 2× data storage, 2× scale storage

**Benefit**: Zero-cost transpose operations

| Operation | With Dual Layout | Without Dual Layout |
|-----------|------------------|---------------------|
| A×B (TN) | Use rowwise | Use rowwise |
| A×B^T (NT) | Use columnwise | **Transpose B at runtime** |
| Memory | 2× | 1× |
| Transpose time | 0 ms | 5-10 ms (typical) |

**Justification**: Memory is cheap, compute is expensive
- Attention layers need both TN and NT
- Transpose B during forward pass = wasted compute
- 2× memory << inference latency savings

**Alternative Rejected**: Lazy transpose (on-demand)
```cpp
// ❌ REJECTED: Adds latency to critical path
if (need_columnwise && !has_columnwise) {
  transpose_tensor(rowwise_data);  // Expensive!
}
```

### Scale Buffer Padding

**Design**: Roundup scale dimensions to 128 and 4

```cpp
// quantizer.cpp:1105-1134
size_t sinv0 = roundup(numel / last_dim, 128);
size_t sinv1 = roundup(last_dim / MXFP8_BLOCK_SIZE, 4);
```

**Example**: For shape `[1000, 2000]`
- **Unpadded**: `[1000, 63]` (1000 rows, 2000/32 = 62.5 ≈ 63 blocks)
- **Padded**: `[1024, 64]` (roundup to 128 and 4)

**Rationale**:

| Padding | Alignment | Benefit | Cost |
|---------|-----------|---------|------|
| 128 | Cache line (128 bytes) | Coalesced memory access | ~5% extra memory |
| 4 | 128-bit vector (4×32-bit) | SIMD load/store | ~5% extra memory |

**GPU Memory Access Pattern**:
```
Without padding:
  Thread 0: Load scale[0] (address 0x1000)
  Thread 1: Load scale[1] (address 0x1001)  // Unaligned!
  Thread 2: Load scale[2] (address 0x1002)  // Unaligned!
  → 32 separate memory transactions

With padding (128-byte aligned):
  Warp: Load scale[0:31] (address 0x1000)
  → 1 coalesced memory transaction (32× faster!)
```

**Trade-off**: ~10% extra memory for 5-10× memory bandwidth improvement

---

## Performance Optimizations

### GIL Release During GPU Work

**Design**: `NVTE_SCOPED_GIL_RELEASE` macro

```cpp
// quantizer.cpp:1100-1102
NVTE_SCOPED_GIL_RELEASE({
  nvte_quantize_v2(input.data(), out.data(), config, stream);
});
```

**What happens**:
1. Acquire GIL (Python lock) before entering C++ function
2. Release GIL before GPU kernel launch
3. Python thread continues running while GPU works
4. Re-acquire GIL when returning to Python

**Benefit**: Overlap CPU and GPU work

| Without GIL Release | With GIL Release |
|---------------------|------------------|
| CPU: Idle (waiting for GPU) | CPU: Runs Python code |
| GPU: Quantize (10 ms) | GPU: Quantize (10 ms) |
| GPU: LayerNorm (5 ms) | CPU: Can prep LayerNorm |
| **Total: 15 ms** | **Total: ~10 ms** (overlapped) |

**Alternative Rejected**: Keep GIL locked
```cpp
// ❌ REJECTED: Blocks Python thread unnecessarily
nvte_quantize_v2(...);  // Python frozen during GPU work
```

### Lazy Quantization (Allocate Then Fill)

**Design**: Separate `create_tensor` from `quantize`

```cpp
// Step 1: Allocate empty buffers
auto [output_cpp, output_py] = quantizer->create_tensor(shape, dtype);

// Step 2: Fill buffers with quantized data
quantizer->quantize(input_cpp, output_cpp, noop_flag);
```

**Benefits**:
1. **Buffer reuse**: Pre-allocate output once, reuse for multiple quantizations
2. **Bulk allocation**: Allocate many tensors together (better memory layout)
3. **Flexibility**: Can allocate on CPU, transfer to GPU later

**Example** (bulk allocation in [`cast.cpp:347-492`](../../../transformer_engine/pytorch/csrc/extensions/cast.cpp#L347-L492)):
```cpp
// Allocate ALL output tensors in one large buffer
auto buffer = at::empty({total_size}, dtype=uint8);

// Create views into buffer (no individual allocations)
for (size_t i = 0; i < num_tensors; ++i) {
  output_data[i] = buffer.slice(offset[i], size[i]);
}
```

**Result**: Fewer CUDA allocations (faster), better memory locality

**Alternative Rejected**: Fused allocate+quantize
```cpp
// ❌ REJECTED: Can't reuse buffers, can't bulk allocate
auto output = quantizer->quantize(input);  // Always allocates new buffer
```

### Asynchronous CUDA Execution

**Design**: All GPU kernels launched on CUDA stream (non-blocking)

```cpp
nvte_quantize_v2(input.data(), out.data(), config,
                 at::cuda::getCurrentCUDAStream());  // Non-blocking!
```

**Flow**:
```
CPU Timeline:
├─▶ Launch quantize kernel (0.01 ms)
├─▶ Launch layernorm kernel (0.01 ms)
├─▶ Launch GEMM kernel (0.01 ms)
└─▶ Synchronize when accessing result (0 ms if GPU done)

GPU Timeline:
├─▶ Execute quantize kernel (10 ms)
├─▶ Execute layernorm kernel (5 ms)
└─▶ Execute GEMM kernel (20 ms)
```

**Benefit**: CPU doesn't block on GPU, can queue many operations

**Alternative Rejected**: Synchronous execution
```cpp
// ❌ REJECTED: CPU waits for each kernel
cudaDeviceSynchronize();  // Block until GPU finishes
```

### Contiguous Memory Layout

**Design**: Force contiguous layout before quantization

```cpp
auto input_contiguous = tensor.contiguous();
```

**Why?**

**Non-contiguous tensor** (e.g., transposed view):
```
Logical: [0, 1, 2, 3, 4, 5]
Physical memory: [0, 3, 6, 1, 4, 7]  (stride != 1)

GPU loads:
  Thread 0: Load elem[0] → address 0x1000
  Thread 1: Load elem[1] → address 0x1018 (gap!)
  Thread 2: Load elem[2] → address 0x1030 (gap!)
  → 6 separate memory transactions (slow!)
```

**Contiguous tensor**:
```
Logical: [0, 1, 2, 3, 4, 5]
Physical memory: [0, 1, 2, 3, 4, 5]  (stride = 1)

GPU loads:
  Warp: Load elem[0:31] → single address
  → 1 coalesced memory transaction (fast!)
```

**Trade-off**: Small copy cost << memory bandwidth savings

---

## API Design Principles

### Generic Entry Point

**Design**: Single `quantize()` function for all formats

```cpp
py::object quantize(const at::Tensor &tensor,
                    py::handle quantizer,
                    const py::object &output,
                    std::optional<at::Tensor> noop_flag)
```

**Benefits**:
1. **Consistency**: Same API for MXFP8, FP8, NVFP4, etc.
2. **Discoverability**: One function to learn, not 5
3. **Future-proof**: New formats just add quantizer type

**Alternative Rejected**: Format-specific functions
```python
# ❌ REJECTED: Confusing, hard to discover
output = quantize_mxfp8(input, quantizer)
output = quantize_fp8(input, quantizer)
output = quantize_nvfp4(input, quantizer)
```

### Optional Pre-allocated Output

**Design**: `output` parameter allows buffer reuse

```python
# Allocate once
output = MXFP8Tensor(shape, dtype)

# Reuse in loop (no allocation overhead)
for i in range(1000):
    quantize(inputs[i], quantizer, output=output)
```

**Benefit**: Eliminates allocation overhead in hot paths

**Trade-off**: Slightly more complex API (optional parameter)

### Conditional Quantization with noop_flag

**Design**: Optional `noop_flag` for selective quantization

```python
# Only quantize large gradients
noop_flag = (grad.abs() < 1e-6)
output = quantize(grad, quantizer, noop_flag=noop_flag)
```

**Use Case**: Sparse gradient optimization
- Small gradients stay FP32 (better precision)
- Large gradients quantized (memory savings)
- Single kernel call (fused logic)

**Alternative Rejected**: Separate quantize and mask operations
```python
# ❌ REJECTED: Two kernel launches, slower
large_grad_mask = grad.abs() >= 1e-6
quantized = quantize(grad, quantizer)
output = torch.where(large_grad_mask, quantized, grad)  # Extra kernel!
```

---

## Trade-off Analysis

### Memory vs. Compute Trade-offs

| Design Choice | Memory Cost | Compute Benefit | Decision |
|---------------|-------------|-----------------|----------|
| Dual layout (rowwise+columnwise) | 2× data storage | Eliminates runtime transpose | **Worth it** (compute >> memory) |
| E8M0 scales | 3% overhead | Bit shifts instead of FP multiply | **Worth it** (minimal cost) |
| Scale buffer padding | 10% extra scales | Coalesced memory access | **Worth it** (5-10× bandwidth) |
| Contiguous memory | Copy cost (one-time) | Faster kernel execution | **Worth it** (paid back in 1-2 kernels) |

### Precision vs. Speed Trade-offs

| Approach | Precision | Speed | Memory | Decision |
|----------|-----------|-------|--------|----------|
| FP32 (baseline) | High | 1× | 1× | Baseline |
| MXFP8 (32-elem blocks) | Medium-High | 4× | 0.25× | **Optimal** |
| MXFP8 (16-elem blocks) | Higher | 3.5× | 0.28× | Not worth finer granularity |
| Per-tensor FP8 | Low | 4× | 0.25× | Too coarse, precision loss |

### Code Complexity vs. Flexibility Trade-offs

| Design Pattern | Complexity | Flexibility | Decision |
|----------------|------------|-------------|----------|
| Polymorphic quantizers | Medium | High | **Worth it** (enables extensibility) |
| Dispatch table | Low | High | **Worth it** (data-driven, clean) |
| Three-layer architecture | Medium | High | **Worth it** (framework independence) |
| Monolithic design | Low | Low | **Rejected** (hard to extend) |

---

## Key Design Principles Summary

1. **Hardware-Aware**:
   - 32-element blocks align with GPU warp size
   - Memory padding enables coalesced access
   - Asynchronous execution hides latency

2. **Precision-Compute Balance**:
   - Block-wise scales balance precision and efficiency
   - E8M0 format trades exact scales for speed
   - FP8 E4M3 adequate for NN workloads

3. **Extensibility First**:
   - Polymorphic quantizers allow new formats
   - Dispatch table enables data-driven extensions
   - Three-layer architecture decouples frameworks

4. **Performance by Default**:
   - GIL release enables CPU/GPU overlap
   - Lazy allocation enables buffer reuse
   - Dual layout eliminates transpose overhead

5. **Trade-offs Favor Throughput**:
   - Memory overhead acceptable for compute savings
   - Small precision loss acceptable for 4× speedup
   - Code complexity acceptable for flexibility

---

## Related Documentation

- [QUANTIZE_FRAME_BY_FRAME.md](QUANTIZE_FRAME_BY_FRAME.md) - Step-by-step execution trace
- [CALL_GRAPH.md](CALL_GRAPH.md) - Function call visualization
- [README.md](README.md) - Overview and quick start
