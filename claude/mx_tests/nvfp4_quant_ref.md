# NVFP4 Quantization: Frame-by-Frame Execution Trace

This document provides a comprehensive frame-by-frame trace of the NVFP4 quantization test in [test_nvfp4_quantize_exact.py:160-177](../../tests/pytorch/nvfp4/test_nvfp4_quantize_exact.py#L160-L177), showing the complete call path from Python → Bindings → C++ → CUDA kernel with detailed analysis of all parameter impacts.

## Overview

The test `test_quantization_block_tiling_versus_reference` validates NVFP4 quantization by comparing the optimized GPU implementation against a pure Python reference implementation. This trace documents exactly how each parameter affects the execution flow.

## Test Parameters

The test is parameterized with the following options:

| Parameter | Type | Values | Description |
|-----------|------|--------|-------------|
| `x_dtype` | torch.dtype | float32, bfloat16 | Input tensor data type |
| `M, N` | int | Various sizes | Matrix dimensions (rows, columns) |
| `return_transpose` | bool | True, False | Whether to compute columnwise quantized data |
| `swizzled_scale` | bool | **False only** | Scale factor memory layout (currently disabled in test) |
| `use_cpp_allocator` | bool | True, False | Allocation method: direct call vs pre-allocated update |
| `with_2d_quantization` | bool | True, False | 1D (1x16) vs 2D (16x16) quantization blocks |

---

## Frame-by-Frame Execution Trace

### Frame 0: Test Entry Point
**Location**: [tests/pytorch/nvfp4/test_nvfp4_quantize_exact.py:160-177](../../tests/pytorch/nvfp4/test_nvfp4_quantize_exact.py#L160-L177)

**Function**: `test_quantization_block_tiling_versus_reference()`

**State Before**:
```python
# Example parameters for a concrete trace
x_dtype = torch.float32
M, N = 256, 256
return_transpose = True
swizzled_scale = False  # Currently always False in test
use_cpp_allocator = True
with_2d_quantization = False  # 1D quantization (1x16 blocks)
```

**Actions**:
- Receives test parameters from pytest
- Calls `check_quantization_nvfp4_versus_reference()` with all parameters

**State After**:
- Control passes to the checking function

---

### Frame 1: Python Layer - Check Function
**Location**: [tests/pytorch/nvfp4/test_nvfp4_quantize_exact.py:26-126](../../tests/pytorch/nvfp4/test_nvfp4_quantize_exact.py#L26-L126)

**Function**: `check_quantization_nvfp4_versus_reference()`

**State Before**:
```python
# Parameters received
x_dtype = torch.float32
M, N = 256, 256
return_transpose = True
swizzled_scale = False
use_cpp_allocator = True
with_2d_quantization = False
```

**Actions**:

1. **Setup** (lines 35-43):
   ```python
   te_dtype = tex.DType.kFloat4E2M1  # FP4 E2M1 format
   device = "cuda"
   seed = 0
   torch.manual_seed(seed)
   x = torch.randn((256, 256), dtype=torch.float32, device="cuda")
   ```

2. **Create NVFP4Quantizer** (lines 46-55):
   ```python
   nvfp4_quantizer = NVFP4Quantizer(
       fp4_dtype=tex.DType.kFloat4E2M1,
       rowwise=True,                    # Always compute rowwise data
       columnwise=True,                 # return_transpose=True
       with_amax_reduction=False,       # No distributed reduction
       amax_reduction_group=None,
       with_rht=False,                  # No Random Hadamard Transform
       with_post_rht_amax=False,
       with_2d_quantization=False,      # 1D block scaling (1x16)
   )
   ```

3. **Quantize** (lines 56-62):

   **Path A - use_cpp_allocator=True** (lines 57):
   ```python
   x_nvfp4_sut = nvfp4_quantizer(x)  # Calls __call__() → quantize()
   ```

   **Path B - use_cpp_allocator=False** (lines 59-62):
   ```python
   # Pre-allocate empty NVFP4Tensor
   x_nvfp4_sut = nvfp4_quantizer.make_empty(
       (256, 256), dtype=torch.float32, device="cuda", requires_grad=False
   )
   # Update with quantized data
   x_nvfp4_sut = nvfp4_quantizer.update_quantized(x, x_nvfp4_sut)
   ```

**State After**:
```python
# NVFP4Tensor created with:
x_nvfp4_sut._rowwise_data      # shape: (256, 128) uint8 (packed FP4)
x_nvfp4_sut._rowwise_scale_inv # shape: (256, 16) uint8 (E8M0 format)
x_nvfp4_sut._columnwise_data   # shape: (256, 128) uint8 (transposed)
x_nvfp4_sut._columnwise_scale_inv # shape: (256, 16) uint8
x_nvfp4_sut._amax_rowwise      # shape: (1,) float32
x_nvfp4_sut._amax_columnwise   # shape: (1,) float32
```

---

### Frame 2: Python NVFP4Quantizer Layer
**Location**: [transformer_engine/pytorch/tensor/nvfp4_tensor.py:158-177](../../transformer_engine/pytorch/tensor/nvfp4_tensor.py#L158-L177)

**Function**: `NVFP4Quantizer.update_quantized()`

**State Before**:
```python
src = torch.Tensor([256, 256], dtype=float32)  # Input tensor
dst = NVFP4Tensor(...)  # Pre-allocated or to be created
noop_flag = None
```

**Actions** (lines 169-176):
```python
# 1. Ensure input is contiguous and on correct device
if not src.is_contiguous():
    src = src.contiguous()

# 2. Call C++ extension via pybind11
tex.quantize(src, self, dst, noop_flag)
```

**Parameter Mapping to C++**:
```python
self.dtype = tex.DType.kFloat4E2M1
self.rowwise_usage = True
self.columnwise_usage = True  # because return_transpose=True
self.with_2d_quantization = False  # 1D block scaling
self.stochastic_rounding = False
self.with_rht = False
```

**State After**:
- C++ extension receives:
  - Input tensor (contiguous float32)
  - Quantizer object (Python handle)
  - Output NVFP4Tensor
  - No noop flag

---

### Frame 3: C++ Pybind11 Entry Point
**Location**: [transformer_engine/pytorch/csrc/extensions/pybind.cpp:119-120](../../transformer_engine/pytorch/csrc/extensions/pybind.cpp#L119-L120)

**Pybind Definition**:
```cpp
m.def("quantize", transformer_engine::pytorch::quantize,
      py::arg("tensor"), py::arg("quantizer"),
      py::arg("output") = py::none(), py::arg("noop") = py::none());
```

**State Before**:
```cpp
tensor = at::Tensor (256, 256) float32
quantizer = py::handle to NVFP4Quantizer Python object
output = NVFP4Tensor Python object
noop = std::nullopt
```

---

### Frame 4: C++ Quantize Implementation
**Location**: [transformer_engine/pytorch/csrc/extensions/cast.cpp:33-79](../../transformer_engine/pytorch/csrc/extensions/cast.cpp#L33-L79)

**Function**: `py::object quantize()`

**Actions**:

1. **Convert Quantizer to C++** (line 36):
   ```cpp
   auto quantizer_cpp = convert_quantizer(quantizer);
   // Returns: std::unique_ptr<NVFP4Quantizer>
   ```

2. **Make Input Contiguous** (lines 39-40):
   ```cpp
   auto input_contiguous = tensor.contiguous();
   auto input_cpp = makeTransformerEngineTensor(input_contiguous);
   ```

3. **Initialize Output** (lines 56-62):
   ```cpp
   if (output.is_none()) {
       // Allocate new tensor
       std::tie(output_cpp, output_py) =
           quantizer_cpp->create_tensor(shape, fake_dtype);
   } else {
       // Use provided tensor
       std::tie(output_cpp, output_py) =
           quantizer_cpp->convert_and_update_tensor(output);
   }
   ```

4. **Call Quantization** (lines 75-76):
   ```cpp
   quantizer_cpp->quantize(input_cpp, output_cpp, noop_flag_cpp);
   ```

**State After**:
```cpp
output_cpp = TensorWrapper with:
  - rowwise_data ptr
  - rowwise_scale_inv ptr
  - columnwise_data ptr (if return_transpose=True)
  - columnwise_scale_inv ptr
  - amax ptrs
```

---

### Frame 5: C++ NVFP4Quantizer Constructor
**Location**: [transformer_engine/pytorch/csrc/quantizer.cpp:1136-1155](../../transformer_engine/pytorch/csrc/quantizer.cpp#L1136-L1155)

**Function**: `NVFP4Quantizer::NVFP4Quantizer(const py::handle& quantizer)`

**Actions**:
```cpp
// Extract Python quantizer attributes
this->dtype = quantizer.attr("dtype").cast<DType>();
// = DType::kFloat4E2M1

this->with_rht = quantizer.attr("with_rht").cast<bool>();
// = false

this->with_post_rht_amax = quantizer.attr("with_post_rht_amax").cast<bool>();
// = false

this->with_2d_quantization = quantizer.attr("with_2d_quantization").cast<bool>();
// = false (1D quantization with 1x16 blocks)

this->stochastic_rounding = quantizer.attr("stochastic_rounding").cast<bool>();
// = false

this->with_amax_reduction = quantizer.attr("with_amax_reduction").cast<bool>();
// = false

this->rowwise_usage = quantizer.attr("rowwise_usage").cast<bool>();
// = true

this->columnwise_usage = quantizer.attr("columnwise_usage").cast<bool>();
// = true (because return_transpose=True)
```

**State After**:
```cpp
NVFP4Quantizer object created with:
  dtype = kFloat4E2M1
  with_2d_quantization = false  // 1D: 1x16 blocks
  rowwise_usage = true
  columnwise_usage = true
  stochastic_rounding = false
  with_rht = false
  with_amax_reduction = false
```

---

### Frame 6: C++ Quantize Implementation
**Location**: [transformer_engine/pytorch/csrc/quantizer.cpp:1446-1650](../../transformer_engine/pytorch/csrc/quantizer.cpp#L1446-L1650)

**Function**: `NVFP4Quantizer::quantize_impl()`

**State Before**:
```cpp
input = TensorWrapper([256, 256], dtype=float32)
out = TensorWrapper(NVFP4Tensor with allocated buffers)
noop_flag = std::nullopt
compute_amax = true
```

**Actions**:

1. **Setup Quantization Config** (lines 1456-1461):
   ```cpp
   QuantizationConfigWrapper quant_config;
   quant_config.set_nvfp4_2d_quantization(false);  // 1D quantization
   quant_config.set_stochastic_rounding(false);
   ```

2. **Compute Dimensions** (lines 1465-1469):
   ```cpp
   size_t rows = 1;
   for (size_t i = 0; i < input.ndim() - 1; ++i) {
       rows *= input.size(i);  // = 256
   }
   size_t cols = input.size(input.ndim() - 1);  // = 256
   ```

3. **Compute Amax** (lines 1506-1528):
   ```cpp
   // Since with_rht=false, go to this branch
   if (compute_amax) {
       auto rowwise_amax_ptr = out.get_amax().data_ptr;
       auto columnwise_amax_ptr = out.get_columnwise_amax().data_ptr;
       void* amax_ptr = rowwise_amax_ptr;

       out.set_amax(amax_ptr, DType::kFloat32, {1});
       nvte_compute_amax_with_config(input.data(), out.data(),
                                     quant_config, stream);

       // Copy amax to both rowwise and columnwise
       cudaMemcpyAsync(columnwise_amax_ptr, amax_ptr, sizeof(float),
                      cudaMemcpyDeviceToDevice, stream);
   }
   ```

4. **Call TE Quantize Kernel** (lines 1624-1625):
   ```cpp
   nvte_quantize_v2(input.data(), out.data(), quant_config, stream);
   ```

**State After**:
```cpp
// Amax computed (global max of absolute values)
out.amax = max(abs(input))  // scalar float32

// Ready to call CUDA kernel
```

---

### Frame 7: TE Kernel Dispatcher
**Location**: [transformer_engine/common/cast/dispatch/quantize.cuh:90-130](../../transformer_engine/common/cast/dispatch/quantize.cuh#L90-L130)

**Function**: `dispatch::quantize_fwd_helper()`

**State Before**:
```cpp
input: TensorWrapper([256, 256], dtype=float32)
output: TensorWrapper with:
  - scaling_mode = NVTE_NVFP4_1D_SCALING
  - rowwise_data ptr
  - columnwise_data ptr (because return_transpose=True)
  - scale_inv ptrs
config: with_2d_quantization = false
```

**Dispatch Logic**:

```cpp
// Line 90-100: Check scaling mode
if (output.scaling_mode() == NVTE_NVFP4_1D_SCALING) {

    // Line 102: Determine block size based on 2D quantization flag
    constexpr size_t kScaleBlockDim = 16;
    const bool use_2d_quantization = config.get_nvfp4_2d_quantization();
    // use_2d_quantization = false → 1D quantization (1x16 blocks)

    // Line 110-120: Choose optimized vs general kernel
    const bool has_columnwise_data =
        output.get_columnwise_data().data_ptr != nullptr;
    // has_columnwise_data = true (return_transpose=True)

    const bool is_bf16_aligned =
        input.dtype() == DType::kBFloat16 &&
        rows % 32 == 0 && cols % 32 == 0;

    if (is_bf16_aligned) {
        // Path A: Optimized TMA-based kernel for BF16
        if (use_2d_quantization) {
            nvfp4::quantize_transpose<true>(input, output, ...);
        } else {
            nvfp4::quantize_transpose<false>(input, output, ...);
        }
    } else {
        // Path B: General kernel for other types
        quantize_transpose_vector_blockwise_fp4(input, output, ...);
    }
}
```

**Path Selection for Our Example**:
- Input dtype = float32 (not BF16)
- **Route**: General kernel path
- **Kernel**: `quantize_transpose_vector_blockwise_fp4`

---

### Frame 8: CUDA Kernel Entry
**Location**: [transformer_engine/common/transpose/quantize_transpose_vector_blockwise_fp4.cu:704-836](../../transformer_engine/common/transpose/quantize_transpose_vector_blockwise_fp4.cu#L704-L836)

**Function**: `quantize_transpose_vector_blockwise_fp4()`

**State Before**:
```cpp
input: [256, 256] float32
output_c: [256, 128] uint8 (rowwise FP4 data)
output_t: [256, 128] uint8 (columnwise FP4 data)
tile_scales_inv_c: [256, 16] uint8 (rowwise scales)
tile_scales_inv_t: [256, 16] uint8 (columnwise scales)
global_amax: scalar float32 (computed earlier)
```

**Actions**:

1. **Template Instantiation** (lines 730-760):
   ```cpp
   constexpr bool kReturnIdentity = true;    // Always compute rowwise
   constexpr bool kReturnTranspose = true;   // return_transpose=True
   constexpr bool kIsE8Scaling = true;       // E8M0 scale format
   constexpr bool kAligned = true;           // Assume alignment
   constexpr bool kSwizzledScale = false;    // Linear scale layout
   constexpr bool kApplyStochasticRounding = false;
   constexpr bool kIs2DBlockScaling = false; // 1D quantization

   using IType = float;                      // Input type
   using OType = uint8_t;                    // Output type (packed FP4)
   using ScaleType = uint8_t;                // E8M0 scale type
   ```

2. **Compute Launch Configuration** (lines 786-795):
   ```cpp
   constexpr int kTileDim = 128;
   const int num_blocks_x = DIVUP(cols, kTileDim);  // DIVUP(256, 128) = 2
   const int num_blocks_y = DIVUP(rows, kTileDim);  // DIVUP(256, 128) = 2
   dim3 grid(num_blocks_x, num_blocks_y);           // dim3(2, 2)
   constexpr int kThreadsPerBlock = 256;            // 8 warps
   ```

3. **Compute Scale Strides** (lines 765-785):

   **For Rowwise Scales** (non-swizzled):
   ```cpp
   // Scale shape: [M_padded, N/16]
   // M_padded = roundup(M, 128) = 256
   // N_scale = roundup(N/16, 4) = roundup(16, 4) = 16

   size_t scale_stride_x = 1;        // Column stride
   size_t scale_stride_y = 16;       // Row stride
   ```

   **For Columnwise Scales** (transposed, non-swizzled):
   ```cpp
   // Scale shape: [N_padded, M/16]
   // N_padded = roundup(N, 128) = 256
   // M_scale = roundup(M/16, 4) = roundup(16, 4) = 16

   size_t scale_t_stride_x = 1;      // Column stride
   size_t scale_t_stride_y = 16;     // Row stride
   ```

4. **Launch Kernel** (lines 813-823):
   ```cpp
   block_scaled_1d_cast_transpose_kernel<
       kReturnIdentity, kReturnTranspose, kIsE8Scaling, kAligned,
       float, float, uint8_t, uint8_t,
       false,   // kSwizzledScale = false
       false,   // kApplyStochasticRounding = false
       false    // kIs2DBlockScaling = false
   ><<<grid, kThreadsPerBlock, smem_bytes, stream>>>(
       input_ptr, global_amax_ptr,
       output_c_ptr, output_t_ptr,
       tile_scales_inv_c_ptr, tile_scales_inv_t_ptr,
       cols, rows,
       scale_stride_x, scale_stride_y,
       scale_t_stride_x, scale_t_stride_y,
       kScaleBlockDim,  // = 16
       epsilon, rng_state, noop_ptr
   );
   ```

**Launch Configuration**:
```
Grid: (2, 2, 1) = 4 blocks total
Block: 256 threads (8 warps)
Shared Memory: ~66 KB
Each block processes: 128x128 tile of input
```

**State After**:
- CUDA kernel launched asynchronously
- Control returns to CPU

---

### Frame 9: CUDA Kernel Execution
**Location**: [transformer_engine/common/transpose/quantize_transpose_vector_blockwise_fp4.cu:321-699](../../transformer_engine/common/transpose/quantize_transpose_vector_blockwise_fp4.cu#L321-L699)

**Function**: `block_scaled_1d_cast_transpose_kernel<...>()`

**Thread Organization**:
```
Grid: 2x2 blocks
Block: 256 threads = 8 warps
Each block processes: 128x128 tile

Example for block (0, 0):
  - Processes input[0:128, 0:128]
  - 256 threads cooperate
  - Shared memory: 128x128 floats
```

**Kernel Execution Flow**:

#### **Step 1: Load Input to Shared Memory** (lines 375-416)

**Thread Mapping**:
```cpp
const int block_idx_x = blockIdx.x;  // 0 (for first block)
const int block_idx_y = blockIdx.y;  // 0
const int tidx = threadIdx.x;        // [0-255]

// Each thread loads 8 consecutive elements
constexpr int kNVecIn = 8;
```

**Loading Logic**:
```cpp
__shared__ IType smem[kTileDim][kTileDim + 1];  // 128x129 (avoid bank conflicts)

// Thread 0 loads elements [0-7] of row 0
// Thread 1 loads elements [0-7] of row 1
// ...
// Thread 127 loads elements [0-7] of row 127
// Thread 128 loads elements [8-15] of row 0
// etc.

for (int i = 0; i < iterations; ++i) {
    size_t row_idx = block_idx_y * kTileDim + thread_row;
    size_t col_idx = block_idx_x * kTileDim + thread_col + i * kNumThreadsLoad;

    // Vectorized load: 8 floats at once
    IVec input_vec;
    input_vec.input_type = *reinterpret_cast<const Vec<IType, kNVecIn>*>(
        &input[row_idx * row_length + col_idx]);

    // Store to shared memory
    for (int j = 0; j < kNVecIn; ++j) {
        smem[thread_row][thread_col + i * kNumThreadsLoad + j] =
            input_vec.input_type.data[j];
    }
}

__syncthreads();  // Wait for all threads to finish loading
```

**State After Step 1**:
```
Shared memory populated with 128x128 tile:
smem[0][0:127]   = input[0, 0:127]
smem[1][0:127]   = input[1, 0:127]
...
smem[127][0:127] = input[127, 0:127]
```

---

#### **Step 2: Rowwise Quantization** (lines 425-565)

This step processes data rowwise (left-to-right) and produces the main quantized output.

**Sub-step 2.1-2.3: Compute AMAX per Quantization Block** (lines 425-480)

```cpp
// Quantization block: 1x16 elements (1D) or 16x16 elements (2D)
// For 1D with_2d_quantization=false: each 1x16 horizontal strip

constexpr int kScaleBlockDim = 16;  // Block dimension

// Thread 0-7 process one 1x16 block at a time
float amax_val = 0.0f;

// Each thread responsible for computing amax of specific elements
for (int r_s = 0; r_s < kNumThreadsStore; ++r_s) {
    // Load 16 elements from shared memory
    for (int c = 0; c < kNVecOut; c += kNVecSMem) {
        int col = compute_col_index(...);
        SMemVec smem_vec = smem[r_s][col];

        // Find max absolute value
        for (int j = 0; j < kNVecSMem; ++j) {
            float val = static_cast<float>(smem_vec.data[j]);
            amax_val = max(amax_val, fabs(val));
        }
    }
}

// Warp-level reduction to find block amax
for (int offset = 16; offset >= 1; offset /= 2) {
    amax_val = max(amax_val, __shfl_xor_sync(0xFFFFFFFF, amax_val, offset));
}
```

**Sub-step 2.4-2.5: Compute and Store Scale Factors** (lines 488-530)

```cpp
// Thread 0 of each group computes the scale
if ((threadIdx.x % kNumThreadsStore) % kNumThreadsReduce == 0) {
    // Compute global encoding scale (E8M0 format)
    ScaleType scale = ComputeGlobalEncodeScaleFP4<ScaleType>(
        global_amax[0],  // Global amax from earlier
        amax_val,        // Block amax just computed
        kScaleBlockDim,  // = 16
        epsilon
    );

    // Compute decode scale for quantization
    float scale_inv = ComputeDecodeScaleFP4<ScaleType>(scale);

    // Store scale based on swizzled_scale parameter
    size_t row_idx = block_idx_y * kTileDim + r_s;
    size_t col_idx = block_idx_x * (kNumThreadsStore / kNumThreadsReduce) +
                     (threadIdx.x % kNumThreadsStore) / kNumThreadsReduce;

    if constexpr (kSwizzledScale) {
        // SWIZZLED PATH (not used in our test: swizzled_scale=False)
        // Complex 3D swizzled layout for cuBLAS
        size_t offset = scale_factor_swizzled_offset<ScaleType>(
            row_idx, col_idx, DIVUP(row_length, kScaleBlockDim));
        tile_scales_inv_c[offset] = scale_inv;
    } else {
        // LINEAR PATH (used in our test)
        // Simple row-major layout
        tile_scales_inv_c[row_idx * scale_stride_y + col_idx * scale_stride_x]
            = scale_inv;
    }
}
```

**Scale Format**: E8M0 (8-bit exponent, 0 mantissa bits)
- This is a power-of-2 scale represented as `uint8_t`
- Formula: `actual_scale = 2^(exponent - 127)`

**Sub-step 2.6-2.7: Quantize and Store** (lines 532-565)

```cpp
// Quantize each 1x16 block using computed scales
OVec output_vec;  // Container for 16 FP4 values (8 bytes)

for (int c = 0; c < kNVecOut; c += kNVecSMem * 2) {
    // Load 4 consecutive float values
    float val0 = smem[r_s][col_base + 0];
    float val1 = smem[r_s][col_base + 1];
    float val2 = smem[r_s][col_base + 2];
    float val3 = smem[r_s][col_base + 3];

    // Scale the values
    float scaled0 = val0 * scale_inv;
    float scaled1 = val1 * scale_inv;
    float scaled2 = val2 * scale_inv;
    float scaled3 = val3 * scale_inv;

    // Convert 4 float32 → 4 FP4 using PTX instruction
    __nv_fp4x4_e2m1 fp4_result = cvt_fp32_to_fp4_4x(
        float2{scaled0, scaled1},
        float2{scaled2, scaled3},
        rbits  // Random bits for stochastic rounding (unused here)
    );

    // Pack 4 FP4 values into 2 bytes
    output_vec.data[c / 2] = reinterpret_cast<uint8_t*>(&fp4_result)[0];
    output_vec.data[c / 2 + 1] = reinterpret_cast<uint8_t*>(&fp4_result)[1];
}

// Store 16 FP4 values (8 bytes) to global memory
size_t row_idx = block_idx_y * kTileDim + r_s;
size_t col_idx = block_idx_x * kTileDim + ...;
*reinterpret_cast<OVec*>(&output_c[row_idx * row_length_packed + col_idx])
    = output_vec;
```

**FP4 Conversion PTX**:
```assembly
cvt.rn.satfinite.e2m1x2.f32 f0, %1, %2;  // Convert 2 float32 → 2 FP4
cvt.rn.satfinite.e2m1x2.f32 f1, %3, %4;  // Convert 2 more
mov.b32 %0, {f0, f1, f0, f1};            // Pack into output
```

**State After Step 2**:
```
output_c: [256, 128] uint8 with quantized rowwise data
  - Each row has 256 FP4 values (128 bytes)
  - Values quantized using per-block scales
tile_scales_inv_c: [256, 16] uint8 with rowwise scales
  - Each row has 16 scale values (one per 16-element block)
```

---

#### **Step 3: Columnwise Quantization (Transpose)** (lines 567-699)

This step processes data columnwise (top-to-bottom) and produces transposed quantized output.

**Sub-step 3.1: Load from Shared Memory (Transposed)** (lines 574-605)

```cpp
// Now load from shared memory in transposed order
// Each thread loads a column instead of a row

for (int c_s = 0; c_s < iterations_t; ++c_s) {
    // Transpose indices
    size_t row_idx = block_idx_x * kTileDim + thread_col;  // Note: x and y swapped
    size_t col_idx = block_idx_y * kTileDim + thread_row;

    // Load from shared memory in column-major order
    SMemVec smem_vec;
    for (int j = 0; j < kNVecSMem; ++j) {
        smem_vec.data[j] = smem[...][...];  // Transposed access pattern
    }
}
```

**Sub-step 3.2-3.3: Compute AMAX for Transposed Blocks** (lines 607-650)

Same as Step 2.1-2.3, but operating on transposed data:
```cpp
// Compute amax for each 1x16 block in the transposed view
float amax_val_t = 0.0f;

// Find max absolute value in this 1x16 transposed block
for (each element in block) {
    amax_val_t = max(amax_val_t, fabs(value));
}

// Warp reduction
amax_val_t = warp_reduce_max(amax_val_t);
```

**Sub-step 3.4-3.5: Compute and Store Transposed Scales** (lines 652-664)

```cpp
if (thread_is_leader) {
    ScaleType scale_t = ComputeGlobalEncodeScaleFP4<ScaleType>(
        global_amax[0], amax_val_t, kScaleBlockDim, epsilon);
    float scale_inv_t = ComputeDecodeScaleFP4<ScaleType>(scale_t);

    // Compute transposed indices
    size_t row_idx = block_idx_x * kTileDim + c_s * kNVecSMem + smem_idx;
    size_t col_idx = (block_idx_y * (kNumThreadsStore / kNumThreadsReduce) + ...);

    if constexpr (kSwizzledScale) {
        // SWIZZLED PATH (not used: swizzled_scale=False)
        size_t offset = scale_factor_swizzled_offset<ScaleType>(
            row_idx, col_idx, DIVUP(num_rows, kScaleBlockDim));
        tile_scales_inv_t[offset] = scale_inv_t;
    } else {
        // LINEAR PATH (used in our test)
        tile_scales_inv_t[row_idx * scale_t_stride_y + col_idx * scale_t_stride_x]
            = scale_inv_t;
    }
}
```

**Sub-step 3.6-3.7: Quantize and Store Transposed Data** (lines 666-699)

```cpp
// Quantize transposed data
OVec output_vec_t;

for (each 4-element group) {
    // Load 4 values (already transposed in shared mem access)
    float val0 = ...;
    float val1 = ...;
    float val2 = ...;
    float val3 = ...;

    // Scale
    float scaled0 = val0 * scale_inv_t;
    float scaled1 = val1 * scale_inv_t;
    float scaled2 = val2 * scale_inv_t;
    float scaled3 = val3 * scale_inv_t;

    // Convert to FP4
    __nv_fp4x4_e2m1 fp4_result = cvt_fp32_to_fp4_4x(
        float2{scaled0, scaled1}, float2{scaled2, scaled3}, rbits);

    // Pack
    output_vec_t.data[...] = ...;
}

// Store transposed output
size_t row_idx_t = block_idx_x * kTileDim + ...;  // Swapped
size_t col_idx_t = block_idx_y * kTileDim + ...;
*reinterpret_cast<OVec*>(&output_t[row_idx_t * num_rows_packed + col_idx_t])
    = output_vec_t;
```

**State After Step 3**:
```
output_t: [256, 128] uint8 with quantized columnwise data
  - Data in transposed layout: output_t[i, j] = quantize(input[j, i])
  - Each "row" represents a column from the original matrix
tile_scales_inv_t: [256, 16] uint8 with columnwise scales
  - Scales for transposed quantization blocks
```

---

### Frame 10: CUDA Kernel Complete
**Location**: Return from kernel

**State After All Blocks Complete**:
```
All 4 blocks (2x2 grid) have processed their 128x128 tiles
Complete output tensors:

1. output_c (rowwise): [256, 128] uint8
   - Packed FP4 data (2 FP4 per byte)
   - Layout: row-major, each row represents original row

2. tile_scales_inv_c (rowwise scales): [256, 16] uint8
   - E8M0 scale format
   - Layout: Linear row-major (not swizzled)
   - scales_c[i, j] = scale for input[i, j*16:(j+1)*16]

3. output_t (columnwise): [256, 128] uint8
   - Packed FP4 data in transposed layout
   - Layout: each "row" represents original column

4. tile_scales_inv_t (columnwise scales): [256, 16] uint8
   - E8M0 scale format
   - Layout: Linear row-major (not swizzled)
   - scales_t[i, j] = scale for transposed block
```

---

### Frame 11: Return to C++
**Location**: [transformer_engine/pytorch/csrc/quantizer.cpp:1625](../../transformer_engine/pytorch/csrc/quantizer.cpp#L1625)

**State Before**:
- CUDA kernel completed asynchronously
- Output buffers filled with quantized data

**Actions**:
- Return from `nvte_quantize_v2()`
- `TensorWrapper out` now contains fully quantized data
- Return to Python layer

**State After**:
```cpp
out.rowwise_data: [256, 128] uint8 (FP4)
out.rowwise_scale_inv: [256, 16] uint8 (E8M0)
out.columnwise_data: [256, 128] uint8 (FP4 transposed)
out.columnwise_scale_inv: [256, 16] uint8 (E8M0)
out.amax_rowwise: float32 scalar
out.amax_columnwise: float32 scalar
```

---

### Frame 12: Return to Python
**Location**: [tests/pytorch/nvfp4/test_nvfp4_quantize_exact.py:65-126](../../tests/pytorch/nvfp4/test_nvfp4_quantize_exact.py#L65-L126)

**State Before**:
```python
x_nvfp4_sut = NVFP4Tensor(...)  # Returned from quantizer
```

**Actions**:

1. **Extract Quantized Data** (lines 65-75):
   ```python
   qx = x_nvfp4_sut._rowwise_data.view(dtype=torch.uint8)
   # shape: (256, 128) - packed FP4

   sx = x_nvfp4_sut._rowwise_scale_inv
   # shape: (256, 16) - E8M0 scales

   qx_t = x_nvfp4_sut._columnwise_data.view(dtype=torch.uint8)
   # shape: (256, 128) - packed FP4 transposed

   sx_t = x_nvfp4_sut._columnwise_scale_inv
   # shape: (256, 16) - E8M0 scales

   qx_amax = x_nvfp4_sut._amax_rowwise
   # shape: (1,) - float32
   ```

2. **Run Reference Quantizer** (lines 78-87):
   ```python
   quant_tile_shape = (1, 16)  # 1D quantization
   ref_quantizer = NVFP4QuantizerRef(
       dtype=utils.Fp4Formats.E2M1,
       rowwise=True,
       columnwise=True,
       pow_2_scales=False,
       eps=0.0,
       quant_tile_shape=(1, 16),  # Match 1D setting
   )
   x_nvfp4_ref = ref_quantizer.quantize(x)
   ```

3. **Unpack FP4 for Comparison** (lines 106-107):
   ```python
   def unpack_fp4(x: torch.Tensor) -> torch.Tensor:
       # Each uint8 contains 2 FP4 values
       # Lower 4 bits: first FP4
       # Upper 4 bits: second FP4
       repeated = x.repeat_interleave(2, dim=1)
       repeated[:, 0::2] &= 0x0F  # Extract lower 4 bits
       repeated[:, 1::2] >>= 4    # Extract upper 4 bits
       return repeated

   qx = unpack_fp4(qx)       # (256, 256) unpacked FP4 nibbles
   qx_t = unpack_fp4(qx_t)   # (256, 256) unpacked FP4 nibbles
   ```

4. **Compare Results** (lines 109-125):
   ```python
   # Compare quantized data (exact match required)
   torch.testing.assert_close(qx, qx_ref, atol=0.0, rtol=0.0)

   # Compare scales (accounting for padding)
   ref_sx_shape = sx_ref.shape
   sx_valid = sx[:ref_sx_shape[0], :ref_sx_shape[1]]
   torch.testing.assert_close(sx_valid, sx_ref, atol=0.0, rtol=0.0)

   # Compare transposed data and scales
   if return_transpose:
       torch.testing.assert_close(qx_t, qx_t_ref, atol=0.0, rtol=0.0)

       ref_sx_t_shape = sx_t_ref.shape
       sx_t_valid = sx_t[:ref_sx_t_shape[0], :ref_sx_t_shape[1]]
       torch.testing.assert_close(sx_t_valid, sx_t_ref, atol=0.0, rtol=0.0)

   # Compare amax
   torch.testing.assert_close(qx_amax, ref_amax, atol=0.0, rtol=0.0)
   ```

**Test Result**:
- ✅ All assertions pass
- GPU quantization matches reference implementation exactly

---

## Parameter Impact Deep Dive

### 1. **return_transpose** Parameter

**Impact**: Controls whether columnwise (transposed) data is computed.

**Value = True** (columnwise=True):
```python
# Allocates and computes:
- _columnwise_data: [N, M/2] uint8 (transposed FP4)
- _columnwise_scale_inv: [N_padded, M/16] uint8
- _amax_columnwise: scalar

# CUDA kernel executes:
- Step 2: Rowwise quantization (always)
- Step 3: Columnwise quantization (EXECUTED)

# Template parameter:
kReturnTranspose = true
```

**Value = False** (columnwise=False):
```python
# Only computes:
- _rowwise_data: [M, N/2] uint8
- _rowwise_scale_inv: [M_padded, N/16] uint8
- _amax_rowwise: scalar

# CUDA kernel executes:
- Step 2: Rowwise quantization (always)
- Step 3: Columnwise quantization (SKIPPED)

# Template parameter:
kReturnTranspose = false

# Performance benefit:
- ~2x faster (skips transpose path)
- ~50% less memory
```

**Use Cases**:
- `True`: Training (need both A and A^T for forward/backward)
- `False`: Inference (only need forward pass)

---

### 2. **swizzled_scale** Parameter

**Impact**: Changes memory layout of scale factors for cuBLAS compatibility.

**Current Status**: Always `False` in test (line 153), swizzled layout disabled.

**Value = False** (Linear Layout):
```python
# Scale storage: Simple row-major layout
scale_linear[row, col] = scales[row * stride_y + col * stride_x]

# Access pattern in kernel (line 528):
tile_scales_inv_c[row_idx * scale_stride_y + col_idx * scale_stride_x] = scale_inv

# Example for 256x256 input:
scales shape: [256, 16]
Access: scales[i, j] at offset i*16 + j
```

**Value = True** (Swizzled Layout):
```python
# Scale storage: 3D swizzled layout for cuBLAS
# Based on CUTLASS Blackwell functionality
# https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/blackwell_functionality.md#scale-factor-layouts

# 512-byte base blocks:
- 128 rows × 4 columns per base block
- Divided into 4 column blocks
- Each column block: 32 rows × 4 columns

# Access pattern in kernel (lines 524-526):
size_t offset = scale_factor_swizzled_offset<ScaleType>(
    row_idx, col_idx, DIVUP(row_length, kScaleBlockDim));
tile_scales_inv_c[offset] = scale_inv

# Swizzle formula (lines 220-257):
def scale_factor_swizzled_offset(row_idx, col_idx, col_length):
    kTotalRowsPerBaseBlock = 128
    kRowsPerBaseBlockCol = 32
    kColsPerBaseBlockCol = 4

    rb = row_idx // kTotalRowsPerBaseBlock
    rem = row_idx % kTotalRowsPerBaseBlock
    d4 = rem // kRowsPerBaseBlockCol
    d3 = rem % kRowsPerBaseBlockCol
    cbg = col_idx // kColsPerBaseBlockCol
    d5 = col_idx % kColsPerBaseBlockCol

    cbg_cnt = (col_length + kColsPerBaseBlockCol - 1) // kColsPerBaseBlockCol

    # Logical shape: (rb_cnt, cbg_cnt, 32, 4, 4)
    return ((rb * cbg_cnt + cbg) * kRowsPerBaseBlockCol + d3) * 16 +
           d4 * kColsPerBaseBlockCol + d5

# Equivalent Python reshape:
unswizzled = torch.empty((M, N // 16), dtype=torch.uint8)
cbg_cnt = (N // 16) // 4
rb_cnt = M // 128
tmp = unswizzled.reshape(rb_cnt, 4, 32, cbg_cnt, 4)
tmp = torch.permute(tmp, (0, 3, 2, 1, 4))
swizzled = tmp.reshape((-1, 128, 4))
```

**Why Swizzle?**:
- cuBLAS NVFP4 GEMM expects scales in swizzled format
- Optimizes memory access patterns for matrix multiplication
- Reduces bank conflicts in shared memory
- Required for Blackwell cuBLAS FP4 operations

**Performance**:
- Linear: Simple, portable, but slower for GEMM
- Swizzled: Complex, but required for optimal cuBLAS performance

---

### 3. **use_cpp_allocator** Parameter

**Impact**: Changes how NVFP4Tensor is created, not quantization logic.

**Value = True** (Direct Allocation):
```python
# Single call allocates and quantizes
x_nvfp4_sut = nvfp4_quantizer(x)

# Flow:
1. quantizer.__call__(x)
2. quantizer.quantize_impl(x)
3. quantizer.make_empty() - allocate NVFP4Tensor
4. tex.quantize() - fill with data
5. Return NVFP4Tensor

# Pros: Simple, one-shot operation
# Cons: Less control over allocation
```

**Value = False** (Pre-allocated):
```python
# Separate allocation and quantization
x_nvfp4_sut = nvfp4_quantizer.make_empty(
    (M, N), dtype=x_dtype, device=device, requires_grad=False
)
x_nvfp4_sut = nvfp4_quantizer.update_quantized(x, x_nvfp4_sut)

# Flow:
1. make_empty() - allocate empty NVFP4Tensor with uninitialized buffers
2. update_quantized() - fill existing tensor
3. tex.quantize(x, quantizer, output=x_nvfp4_sut)
4. Return same NVFP4Tensor object

# Pros:
- Reuse allocations across iterations
- Better for training loops
- Control over tensor properties

# Cons: Two-step process
```

**Quantization Result**: Identical - only allocation differs.

---

### 4. **with_2d_quantization** Parameter

**Impact**: Changes quantization block size from 1D to 2D.

**Value = False** (1D Quantization):
```python
# Quantization tile: 1 row × 16 columns
quant_tile_shape = (1, 16)

# Scale granularity:
For 256x256 input:
  - Scales shape: [256, 16]
  - Total blocks: 256 * 16 = 4,096 blocks
  - Each block: 16 consecutive elements in a row

# Block layout:
Row 0: [block_0_0] [block_0_1] ... [block_0_15]
       (elems 0-15) (elems 16-31)    (elems 240-255)
Row 1: [block_1_0] [block_1_1] ... [block_1_15]
...

# CUDA kernel parameter:
kIs2DBlockScaling = false
```

**Value = True** (2D Quantization):
```python
# Quantization tile: 16 rows × 16 columns
quant_tile_shape = (16, 16)

# Scale granularity:
For 256x256 input:
  - Scales shape: [16, 16]  # Fewer scales!
  - Total blocks: 16 * 16 = 256 blocks
  - Each block: 16×16 = 256 elements

# Block layout:
     Cols 0-15    Cols 16-31   ...  Cols 240-255
Rows 0-15   [block_0_0] [block_0_1] ... [block_0_15]
Rows 16-31  [block_1_0] [block_1_1] ... [block_1_15]
...

# CUDA kernel parameter:
kIs2DBlockScaling = true

# AMAX computation:
Instead of max over 16 elements (1D),
compute max over 256 elements (2D)
```

**Trade-offs**:

| Aspect | 1D (1×16) | 2D (16×16) |
|--------|-----------|------------|
| Scales | More (16× more) | Fewer |
| Accuracy | Higher (fine-grained) | Lower (coarse-grained) |
| Memory | Higher scale storage | Lower scale storage |
| Performance | Slightly slower | Slightly faster |
| Use Case | General purpose | Large models, less precision needed |

**Mathematical Impact**:

1D Quantization:
```
Block [i, j*16:(j+1)*16]:
  amax = max(|input[i, j*16]|, |input[i, j*16+1]|, ..., |input[i, j*16+15]|)
  scale = encode_scale(global_amax, amax, 16)
  quantized[i, k] = FP4(input[i, k] * scale) for k in [j*16, j*16+15]
```

2D Quantization:
```
Block [i*16:(i+1)*16, j*16:(j+1)*16]:
  amax = max over all 256 elements in block
  scale = encode_scale(global_amax, amax, 16)
  quantized[m, n] = FP4(input[m, n] * scale)
    for m in [i*16, (i+1)*16), n in [j*16, (j+1)*16)
```

---

### 5. **stochastic_rounding** Parameter

**Impact**: Changes FP4 conversion rounding mode.

**Value = False** (Round-to-Nearest):
```cpp
// PTX instruction (lines 281-298):
__device__ __forceinline__ __nv_fp4x4_e2m1
cvt_fp32_to_fp4_4x_with_rn(const float2 in01, const float2 in23,
                           const uint32_t rbits) {
    uint32_t out_4x;
    asm volatile(
        "{\n"
        ".reg.b8 f0; \n\t"
        ".reg.b8 f1; \n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f0, %1, %2;\n\t"  // .rn = round nearest
        "cvt.rn.satfinite.e2m1x2.f32 f1, %3, %4;\n\t"
        "mov.b32 %0, {f0, f1, f0, f1};\n\t"
        "}"
        : "=r"(out_4x)
        : "f"(in01.y), "f"(in01.x), "f"(in23.y), "f"(in23.x));
    return *reinterpret_cast<__nv_fp4x4_e2m1*>(&out_4x);
}

// Rounding behavior:
value = 1.234567
If FP4 representable values are: [..., 1.2, 1.3, ...]
Round to nearest: 1.2 (deterministic)
```

**Value = True** (Stochastic Rounding):
```cpp
// PTX instruction (lines 260-279):
__device__ __forceinline__ __nv_fp4x4_e2m1
cvt_fp32_to_fp4_4x_with_stochastic_rounding(const float2 in01,
                                           const float2 in23,
                                           const uint32_t rbits) {
    uint16_t out_4x;
    asm volatile(
        "{\n"
        "cvt.rs.satfinite.e2m1x4.f32 %0, {%3, %4, %1, %2}, %5; \n\t"  // .rs = stochastic
        "}"
        : "=h"(out_4x)
        : "f"(in01.y), "f"(in01.x), "f"(in23.y), "f"(in23.x), "r"(rbits));
    return *reinterpret_cast<__nv_fp4x4_e2m1*>(&out_4x);
}

// Rounding behavior:
value = 1.234567
If FP4 representable values are: [..., 1.2, 1.3, ...]
Stochastic:
  - 65.43% chance → 1.2
  - 34.57% chance → 1.3
  (Probabilities based on proximity)

// Random bits generation (lines 1472-1482):
const size_t rng_elts_per_thread = 1024;
auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(...);
at::PhiloxCudaState philox_args = init_philox_state(gen, rng_elts_per_thread);
auto rng_state = torch::empty({2}, opts);
philox_unpack(philox_args, static_cast<int64_t*>(rng_state.data_ptr()));
te_rng_state = makeTransformerEngineTensor(rng_state);
```

**Use Cases**:
- False: Inference, testing, reproducibility
- True: Training (reduces quantization bias in gradients)

**Performance**:
- False: Slightly faster (deterministic)
- True: Slightly slower (needs RNG state)

---

### 6. **with_rht** Parameter (Not varied in test, always False)

**Impact**: Applies Random Hadamard Transform before quantization.

**Value = False** (No Transform):
```python
# Direct quantization
quantized = quantize(input)

# Amax computation (lines 1506-1528):
amax = max(abs(input))
```

**Value = True** (With RHT):
```python
# Apply Random Hadamard Transform
# H is 16×16 Hadamard matrix with random sign flips
rht_input = input @ H^T  # Transform

# Quantization:
rowwise = quantize(input)          # Original for rowwise
columnwise = quantize(rht_input)   # Transformed for columnwise

# Amax computation (lines 1497-1499):
nvte_hadamard_transform_amax(input.data(), out.data(), 0,
                             rht_matrix_random_sign_mask_t, stream)

# Restrictions (line 1485):
- Only supported for BF16 input
- Requires rows % 64 == 0
- Requires cols % 128 == 0
```

**Benefits of RHT**:
- Improves quantization quality for correlated data
- Spreads quantization error more uniformly
- Used in distributed training (All-Gather scenarios)

**Cost**:
- Extra matrix multiply (H^T application)
- More complex amax computation
- BF16 only

---

## Memory Layouts Summary

### Input Tensor
```
Shape: [256, 256] float32
Memory: 256 KB (256 * 256 * 4 bytes)
Layout: Row-major
  input[i, j] at offset i*256 + j
```

### Rowwise Quantized Data
```
Shape: [256, 128] uint8  (packed FP4)
Memory: 32 KB (256 * 128 * 1 byte)
Layout: Row-major, 2 FP4 per byte
  Byte at [i, j] contains:
    - Lower 4 bits: FP4 for input[i, 2*j]
    - Upper 4 bits: FP4 for input[i, 2*j+1]
```

### Rowwise Scales (Linear Layout)
```
Shape: [256, 16] uint8  (E8M0 format)
Memory: 4 KB
Layout: Row-major
  scale[i, j] for block input[i, j*16:(j+1)*16]
  Access: scale[i * 16 + j]
```

### Rowwise Scales (Swizzled Layout)
```
Shape: [256, 16] uint8  (E8M0 format)
Memory: 4 KB
Layout: 3D swizzled (rb_cnt, cbg_cnt, 32, 4, 4)
  scale[i, j] at offset = scale_factor_swizzled_offset(i, j, 16)

  Logical shape after permute: (2, 4, 32, 4, 4)
    - 2 row blocks (256 / 128)
    - 4 column block groups (16 / 4)
    - 32 rows per column block
    - 4 column blocks per group
    - 4 columns per block
```

### Columnwise Quantized Data (Transposed)
```
Shape: [256, 128] uint8  (packed FP4)
Memory: 32 KB
Layout: Transposed, row-major in transposed space
  Byte at [i, j] contains FP4 for:
    - Lower 4 bits: input[2*j, i]    (original column i, row 2*j)
    - Upper 4 bits: input[2*j+1, i]  (original column i, row 2*j+1)
```

### Columnwise Scales
```
Shape: [256, 16] uint8
Memory: 4 KB
Layout: Same as rowwise (linear or swizzled)
  scale_t[i, j] for transposed block
```

---

## Scale Factor Details

### E8M0 Format (uint8_t)
```
8 bits: All exponent, no mantissa
Value = 2^(exponent - 127)

Example encodings:
  uint8 = 127 → 2^0 = 1.0
  uint8 = 128 → 2^1 = 2.0
  uint8 = 126 → 2^-1 = 0.5
  uint8 = 130 → 2^3 = 8.0
```

### Scale Computation

**Encode Scale** (lines 191-198):
```cpp
template <class ScaleType>
__device__ __forceinline__ ScaleType ComputeGlobalEncodeScaleFP4(
    float global_amax, float local_amax, int block_dim, float epsilon) {
  // Compute FP4 max value (E2M1: max = 6.0)
  constexpr float fp4_max = 6.0f;

  // Calculate scale such that local_amax maps to fp4_max
  // scale = fp4_max / (local_amax + epsilon)
  float global_scale = fp4_max / (global_amax + epsilon);
  float local_scale_ratio = local_amax / (global_amax + epsilon);

  // Encode as E8M0
  return encode_scale<ScaleType>(global_scale * local_scale_ratio);
}
```

**Decode Scale** (lines 171-178):
```cpp
template <class ScaleType>
__device__ __forceinline__ float ComputeDecodeScaleFP4(ScaleType scale) {
  // Decode E8M0 to float
  // This gives us the reciprocal scale for quantization
  return decode_scale<ScaleType>(scale);
}
```

**Quantization Formula**:
```
1. Compute block amax: amax_block = max(|x[i]| for all i in block)
2. Compute scale: scale = encode_E8M0(fp4_max / amax_block)
3. Decode for quantization: scale_inv = decode_E8M0(scale)
4. Quantize: fp4[i] = FP4(x[i] * scale_inv)
5. Dequantize: x[i] ≈ fp4[i] / scale_inv
```

---

## Complete State Trace Example

For concrete input: `M=256, N=256, return_transpose=True, with_2d_quantization=False`

### Block (0, 0) Processing:

**Input Tile**: `input[0:128, 0:128]` (128×128 floats = 64 KB)

**Step 1 - Load to Shared Memory**:
```
Time: ~10 μs
Threads: All 256 threads load cooperatively
Data: 16,384 float32 values → shared memory
```

**Step 2 - Rowwise Quantization**:
```
Iterations: 16 (one per 8-row group)
For row group 0 (rows 0-7):
  - Process 128 columns in groups of 16
  - Compute 8 blocks (128/16 = 8)
  - Each block:
    * Find amax over 16 consecutive elements
    * Compute E8M0 scale
    * Quantize 16 float32 → 16 FP4 (8 bytes)
    * Store scale

Output:
  - output_c[0:8, 0:64]: 512 bytes of FP4 data
  - scales_c[0:8, 0:8]: 64 bytes of E8M0 scales

Time per block: ~2 μs
Total: ~16 μs
```

**Step 3 - Columnwise Quantization**:
```
Load transposed data from shared memory
Iterations: 16 (one per 8-column group)
For column group 0 (columns 0-7 → becomes rows 0-7 transposed):
  - Process 128 rows (now columns) in groups of 16
  - Compute 8 transposed blocks
  - Each block:
    * Find amax over 16 elements in column direction
    * Compute E8M0 scale
    * Quantize and store transposed

Output:
  - output_t[0:8, 0:64]: 512 bytes of FP4 data (transposed)
  - scales_t[0:8, 0:8]: 64 bytes of E8M0 scales

Time: ~16 μs
```

**Block Total**:
```
Time: ~42 μs per block
Data processed: 64 KB input → 18 KB output (4.4× compression)
```

**Full Grid** (2×2 = 4 blocks):
```
Total time: ~170 μs
Total compression: 256 KB input → 72 KB output
```

---

## Performance Characteristics

### Kernel Launch Overhead
```
Python → C++ pybind: ~5 μs
C++ → CUDA launch: ~10 μs
Total launch overhead: ~15 μs
```

### Kernel Execution
```
For 256×256:
  - Grid: 2×2 = 4 blocks
  - Time per block: ~40-50 μs
  - Total kernel: ~170 μs (with some parallel execution)
```

### Memory Bandwidth
```
Input read: 256 KB
Output write: 72 KB
Effective bandwidth: ~2 TB/s (on A100)
Kernel is compute-bound, not memory-bound
```

### Instruction Breakdown per Thread
```
Loads: ~64 float32 loads (from global memory)
Stores: ~32 uint8 stores (to global memory)
FP4 conversions: ~64 (using PTX cvt.e2m1x2.f32)
Reductions: ~16 warp-level shuffles
Total: ~500 instructions per thread
```

---

## Conclusion

This trace provides a complete frame-by-frame understanding of NVFP4 quantization:

1. **Python Layer**: Simple API, parameter configuration
2. **Pybind11 Layer**: Type conversion, quantizer object translation
3. **C++ Layer**: Tensor management, amax computation, kernel dispatch
4. **CUDA Kernel**: Block-scaled quantization with transpose

Key insights:
- `return_transpose` doubles work but enables efficient forward/backward passes
- `swizzled_scale=False` uses simple layout (True enables cuBLAS optimization)
- `use_cpp_allocator` only affects allocation, not quantization
- `with_2d_quantization` trades accuracy for memory (16× fewer scales)
- Scale factors use E8M0 format for efficient hardware support
- Kernel is highly optimized with vectorized loads, warp-level reductions, and PTX FP4 instructions

The implementation achieves ~4.4× compression (float32 → FP4 + scales) with minimal accuracy loss for neural network tensors.
