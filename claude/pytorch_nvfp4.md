# Frame-by-Frame Trace of scaled_mm Call Path

Here's a complete trace from Python → Bindings → C++ → CUDA for the `scaled_mm` operation with BlockWise1x16 scaling and the specified dimensions (M=1025, K=512, N=2048).

## Frame 1: Python Entry Point

**Source:** [`torch/nn/functional.py:6691-6772`](https://github.com/pytorch/pytorch/blob/main/torch/nn/functional.py#L6691-L6772)

### Inputs

- `A`: shape `[1025, 256]` (Float4_e2m1fn_x2, packed K dimension, logical K=512)
- `B.t()`: shape `[256, 2048]` (Float4_e2m1fn_x2, transposed, logical K=512)
- `scale_a`: list of 2 tensors:
  - `A_scale`: Float8_e4m3fn scales for BlockWise1x16 (swizzled)
  - `A_global_scale`: Float32 scalar for TensorWise
- `scale_recipe_a`: `[ScalingType.BlockWise1x16, ScalingType.TensorWise]`
- `swizzle_a`: `[SwizzleType.SWIZZLE_32_4_4, SwizzleType.NO_SWIZZLE]`

### Logic

The function acts as a convenience wrapper that:

1. Normalizes inputs using `expand_single_value()` - converts single values or None to lists
2. Converts enums to integers using `enum_list_as_int_list()` because `native_functions.yaml` can't pass enum types directly
3. Routes to native op `torch._scaled_mm_v2`

```python
# Lines 6725-6756
def expand_single_value(v: _Any | list[_Any] | None) -> list[_Any]:
    if v is None:
        return []
    elif not isinstance(v, (list)):
        return [v,]
    else:
        return v

scale_a = expand_single_value(scale_a)  # [A_scale, A_global_scale]
scale_recipe_a = expand_single_value(scale_recipe_a)  # [BlockWise1x16, TensorWise]

def enum_list_as_int_list(l: _Any | list[_Any]) -> list[_Any]:
    if not isinstance(l, list):
        l = [l,]
    return [li.value for li in l]  # Extract .value from enum

# Line 6757
out = torch._scaled_mm_v2(
    mat_a,
    mat_b,
    scale_a,  # [A_scale, A_global_scale]
    enum_list_as_int_list(scale_recipe_a),  # [0, 1] as integers
    enum_list_as_int_list(list_or_empty(swizzle_a)),  # [1, 0] as integers
    scale_b,
    enum_list_as_int_list(scale_recipe_b),
    enum_list_as_int_list(list_or_empty(swizzle_b)),
    bias,
    output_dtype,
    contraction_dim,
    use_fast_accum,
)
```

### State After

- All inputs converted to proper types for C++ binding
- Enums converted to integer arrays `[0, 1]` and `[1, 0]`
## Frame 2: ATen Dispatcher

**Source:** [`aten/src/ATen/native/native_functions.yaml:7293-7296`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml#L7293-L7296)

### Logic

PyTorch's code generation system reads `native_functions.yaml` and auto-generates Python bindings for `torch._scaled_mm_v2`. The YAML file specifies:

- Function signature
- Dispatch key: routes CUDA tensors to `_scaled_mm_cuda_v2`

```yaml
- func: _scaled_mm_v2(Tensor self, Tensor mat2, Tensor[] scale_a, int[] recipe_a,
                     int[] swizzle_a, Tensor[] scale_b, int[] recipe_b, int[] swizzle_b,
                     Tensor? bias, ScalarType? out_dtype, int[] contraction_dim=[],
                     bool use_fast_accum=False) -> Tensor
  variants: function
  dispatch:
    CUDA: _scaled_mm_cuda_v2  # Routes to C++ implementation
```

### State After

- Dispatcher identifies CUDA backend
- Routes to `_scaled_mm_cuda_v2` in `ScaledBlas.cpp`
## Frame 3: CUDA Entry Point (Wrapper)

**Source:** [`aten/src/ATen/native/cuda/ScaledBlas.cpp:1474-1498`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/ScaledBlas.cpp#L1474-L1498)

### Logic

Simple wrapper that allocates output tensor and calls the `_out` variant:

```cpp
Tensor _scaled_mm_cuda_v2(
          const Tensor& mat_a, const Tensor& mat_b,
          ArrayRef<Tensor> scale_a,      // 2 tensors
          IntArrayRef scale_recipe_a,    // [0, 1]
          IntArrayRef swizzle_a,         // [1, 0]
          // ... same for B
          const std::optional<Tensor>& bias,
          const std::optional<c10::ScalarType> out_dtype,
          IntArrayRef contraction_dim,
          bool use_fast_accum) {
  const auto out_dtype_ = out_dtype.value_or(mat_a.scalar_type());
  Tensor out = at::empty({0}, mat_a.options().dtype(out_dtype_));  // Allocate empty output

  return _scaled_mm_cuda_v2_out(
      mat_a, mat_b, scale_a, scale_recipe_a, swizzle_a,
      scale_b, scale_recipe_b, swizzle_b,
      bias, out_dtype, contraction_dim, use_fast_accum,
      out);  // Pass output tensor by reference
}
```

### State After

- Empty output tensor allocated (will be resized later)
- Call forwarded to `_out` implementation
## Frame 4: Core Implementation

**Source:** [`aten/src/ATen/native/cuda/ScaledBlas.cpp:1279-1471`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/ScaledBlas.cpp#L1279-L1471)

### Logic

This is the main orchestration function that:

1. Validates inputs (device, shapes, dtypes)
2. Selects implementation from dispatch table
3. Routes to specialized handler

### Step 4a: Input Validation (Lines 1293-1400)

```cpp
Tensor& _scaled_mm_cuda_v2_out(..., Tensor& out) {
  // Device check
  bool allowed_device = _scaled_mm_allowed_device();  // SM >= 9.0 or 8.9
  TORCH_CHECK_NOT_IMPLEMENTED(allowed_device, "...");

  // Shape checks
  TORCH_CHECK_VALUE(mat_a.dim() == 2, "mat_a must be a matrix");  // [1025, 256]
  TORCH_CHECK_VALUE(mat_b.dim() == 2, "mat_b must be a matrix");  // [256, 2048]

  // For fp4, K dimension is packed 2x
  int K_multiplier = (mat_a.scalar_type() == ScalarType::Float4_e2m1fn_x2) ? 2 : 1;
  // So logical K = 256 * 2 = 512

  // Alignment checks (K must be divisible by 16)
  TORCH_CHECK_VALUE(K_multiplier * mat_a.sizes()[1] % 16 == 0, "...");  // 512 % 16 = 0 ✓
```

### Step 4b: Implementation Selection (Lines 1405-1442)

The code uses a dispatch table defined at `ScaledBlas.cpp:688-700`:

```cpp
// Convert integer arrays back to enums
auto scale_recipe_a_enum = convert_int_to_enum<ScalingType>(scale_recipe_a);
// [0, 1] → [BlockWise1x16, TensorWise]
auto swizzle_a_enum = convert_int_to_enum<SwizzleType>(swizzle_a);
// [1, 0] → [SWIZZLE_32_4_4, NO_SWIZZLE]

// Dispatch table - checked in order
std::array<std::tuple<std::string, acceptance_fn, ScaledGemmImplementation>, 9> scale_kernel_dispatch = {{
  { "tensorwise_tensorwise", ... },
  { "rowwise_rowwise", ... },
  { "block_1x128_128x128", ... },
  { "block_128x128_1x128", ... },
  { "block_1x128_1x128", ... },
  { "nvfp4_nvfp4", scaled_blas::check_nvfp4_recipe, ScaledGemmImplementation::NVFP4_NVFP4},  // ← MATCHES
  { "nvfp4_nvfp4_single_scale", ... },
  { "mxfp8_mxfp8", ... },
  { "mxfp4_mxfp4", ... }
}};

// Find matching implementation
for (const auto& fn_entry : scale_kernel_dispatch) {
    const auto [name, accept_fn, scaled_gemm_impl] = fn_entry;
    bool ok = accept_fn(mat_a.scalar_type(),     // Float4_e2m1fn_x2
                        scale_recipe_a_enum,     // [BlockWise1x16, TensorWise]
                        scale_a,                 // [e4m3fn tensor, fp32 scalar]
                        mat_b.scalar_type(),     // Float4_e2m1fn_x2
                        scale_recipe_b_enum,     // [BlockWise1x16, TensorWise]
                        scale_b);                // [e4m3fn tensor, fp32 scalar]
    if (ok) {
      gemm_impl = scaled_gemm_impl;  // NVFP4_NVFP4
      found_impl = true;
      break;
    }
}
```

The `check_nvfp4_recipe` function ([`CUDAScaledBlas.cpp:125-148`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/CUDAScaledBlas.cpp#L125-L148)) validates:

```cpp
bool check_nvfp4_recipe(...) {
  // Both inputs must be fp4
  if (type_a != Float4_e2m1fn_x2 || type_b != Float4_e2m1fn_x2) return false;

  // Need exactly 2 scales per input
  if (scales_a.size() != 2 || recipe_a.size() != 2 || ...) return false;

  // First scale: BlockWise1x16 with e4m3fn dtype
  if (recipe_a[0] != BlockWise1x16) return false;
  if (scales_a[0].scalar_type() != Float8_e4m3fn) return false;

  // Second scale: TensorWise with fp32 dtype
  if (recipe_a[1] != TensorWise) return false;
  if (scales_a[1].scalar_type() != Float) return false;

  return true;  // All checks passed!
}
```

### Step 4c: Dispatch to Handler (Lines 1449-1463)

```cpp
at::native::resize_output(out, {mat_a.size(0), mat_b.size(1)});  // Resize to [1025, 2048]

if (gemm_impl == ScaledGemmImplementation::NVFP4_NVFP4) {
    return _scaled_nvfp4_nvfp4(
        mat_a, mat_b,
        scale_a[0], swizzle_a_enum[0],  // BlockWise1x16 scale (e4m3fn, swizzled)
        scale_b[0], swizzle_b_enum[0],
        bias, out_dtype_, out,
        scale_a[1], scale_b[1]);        // TensorWise scales (fp32 scalars)
}
```

### State After

- Implementation identified: NVFP4_NVFP4
- Output tensor resized to `[1025, 2048]`
- Routing to NVFP4-specific handler
## Frame 5: NVFP4 Handler

**Source:** [`aten/src/ATen/native/cuda/ScaledBlas.cpp:1209-1254`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/ScaledBlas.cpp#L1209-L1254)

### Logic

Prepares two-level scaling and validates NVFP4-specific requirements:

```cpp
Tensor& _scaled_nvfp4_nvfp4(
          const Tensor& mat_a, const Tensor& mat_b,
          const Tensor& scale_a, const SwizzleType swizzle_a,  // BlockWise1x16
          const Tensor& scale_b, const SwizzleType swizzle_b,
          const std::optional<Tensor>& bias,
          const c10::ScalarType out_dtype,
          Tensor& out,
          const std::optional<Tensor>& global_scale_a,  // TensorWise
          const std::optional<Tensor>& global_scale_b) {

  // Combine global scales into alpha parameter
  std::optional<Tensor> alpha = std::nullopt;
  if (global_scale_a.has_value() && global_scale_b.has_value()) {
    TORCH_CHECK_VALUE(global_scale_a.has_value(), "...");
    TORCH_CHECK_VALUE(global_scale_b.has_value(), "...");
    alpha = global_scale_a.value().mul(global_scale_b.value());  // alpha = scale_a_global * scale_b_global
  }

  // Validate scales have correct swizzled layout
  // For M=1025, K=512 (packed 256), expected scale elements:
  auto scale_a_elems = round_up<int64_t>(1025, 128)  // M rounded to 128
                     * round_up<int64_t>(ceil_div<int64_t>(512, 16), 4);  // (K/16) rounded to 4
                     // = 1152 * 128 = 147,456 elements (actual: padded/swizzled layout)

  TORCH_CHECK_VALUE(swizzle_a == SwizzleType::SWIZZLE_32_4_4, "scale_a must be swizzled");
  TORCH_CHECK_VALUE(scale_a.is_contiguous(), "...");

  auto scaling_choice_a = ScalingType::BlockWise1x16;
  auto scaling_choice_b = ScalingType::BlockWise1x16;

  return _scaled_gemm(mat_a, mat_b, scale_a, scale_b,
                     scaling_choice_a, scaling_choice_b,
                     bias, false /* use_fast_accum */, out, alpha);
}
```

### State After

- Global scales combined: `alpha = A_global_scale * B_global_scale`
- Swizzle layout validated
- Routing to generic `_scaled_gemm`
## Frame 6: Scaled GEMM Orchestrator

**Source:** [`aten/src/ATen/native/cuda/ScaledBlas.cpp:384-446`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/ScaledBlas.cpp#L384-L446)

### Logic

Prepares cuBLAS arguments and routes to the appropriate backend (cuBLASLt or TunableOp):

```cpp
Tensor& _scaled_gemm(
          const Tensor& mat1, const Tensor& mat2,
          const Tensor& scale_a, const Tensor& scale_b,
          const ScalingType scaling_choice_a, const ScalingType scaling_choice_b,
          const std::optional<Tensor>& bias,
          const bool use_fast_accum,
          Tensor& out,
          const std::optional<Tensor>& alpha) {

  // Create cuBLAS argument wrapper
  cublasCommonArgs args(mat1, mat2, out, scale_a, scale_b, std::nullopt,
                       scaling_choice_a, scaling_choice_b);

  // Verify we're doing row-major × column-major
  TORCH_CHECK(args.transa == 't' && args.transb == 'n', "...");

  // ROCm uses TunableOp, CUDA uses cuBLASLt directly
  bool tunable_op_enabled = false;  // On CUDA

  if (!tunable_op_enabled) {
      at::cuda::blas::scaled_gemm(
          args.transa,              // 't' (transpose A)
          args.transb,              // 'n' (no transpose B)
          args.m,                   // 1025
          args.n,                   // 2048
          args.k,                   // 512 (logical, unpacked)
          args.mata->data_ptr(),    // A data
          args.scale_mata_ptr,      // scale_a pointer
          args.lda,                 // Leading dimension of A
          args.mata->scalar_type(), // Float4_e2m1fn_x2
          args.scale_mata_dtype.value(),    // Float8_e4m3fn
          args.scaling_mata_type.value(),   // BlockWise1x16
          args.matb->data_ptr(),    // B data
          args.scale_matb_ptr,      // scale_b pointer
          args.ldb,
          args.matb->scalar_type(), // Float4_e2m1fn_x2
          args.scale_matb_dtype.value(),    // Float8_e4m3fn
          args.scaling_matb_type.value(),   // BlockWise1x16
          bias ? bias->data_ptr(): nullptr,
          bias ? bias->scalar_type() : ...,
          args.result->data_ptr(),  // Output buffer
          args.scale_result_ptr,    // nullptr (no output scaling)
          args.result_ld,           // Leading dimension of output
          out_dtype,                // BFloat16
          use_fast_accum,           // false
          alpha);                   // Combined global scale
      return out;
  }
}
```

### State After

- Matrix dimensions prepared: M=1025, N=2048, K=512
- Transposition flags set: transa='t', transb='n'
- Routing to cuBLASLt API
## Frame 7: cuBLASLt Wrapper

**Source:** [`aten/src/ATen/cuda/CUDABlas.cpp:1967-2213`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/CUDABlas.cpp#L1967-L2213)

### Logic

Configures cuBLASLt descriptors and dispatches to the CUDA kernel:

### Step 7a: Scale Mode Selection (Lines 1900-1920)

```cpp
int get_scale_mode(ScalingType scaling_type, ScalarType scale_dtype, bool use_fast_accum) {
  switch (scaling_type) {
    case ScalingType::BlockWise1x16:
      TORCH_CHECK(scale_dtype == kFloat8_e4m3fn);  // ✓
      #if CUDA_VERSION >= 12080
        return CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;  // cuBLAS enum for 1x16 blocks
      #else
        TORCH_CHECK(false, "...only supported for CUDA 12.8+");
      #endif
    // ... other cases
  }
}
```

### Step 7b: cuBLASLt Configuration (Lines 2004-2110)

```cpp
void scaled_gemm(...) {
  const auto computeType = CUBLAS_COMPUTE_32F;  // FP32 accumulation
  const auto scaleType = CUDA_R_32F;
  float alpha_val = 1.0;  // Will be overwritten if alpha provided
  float beta_val = 0.0;

  // Create matrix multiply descriptor
  CuBlasLtMatmulDescriptor computeDesc(computeType, scaleType);
  computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSA, CUBLAS_OP_T);
  computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSB, CUBLAS_OP_N);

  // Set scale pointers
  computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, mat1_scale_ptr);
  computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, mat2_scale_ptr);

  // Set bias pointer if provided
  if (bias_ptr) {
    computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_BIAS_POINTER, bias_ptr);
    computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_EPILOGUE, CUBLASLT_EPILOGUE_BIAS);
    computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, ...);
  }

  // Handle device-side alpha (global scale)
  if (alpha.has_value() && alpha.value().is_cuda()) {
    float *user_alpha_ptr = at::cuda::detail::get_user_alpha_ptr();
    at::Tensor user_alpha = at::from_blob(user_alpha_ptr, {1}, ...);
    user_alpha.copy_(alpha.value());  // Copy to persistent buffer

    auto pointer_mode = CUBLASLT_POINTER_MODE_DEVICE;
    computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_POINTER_MODE, pointer_mode);
    alpha_ptr = user_alpha.data_ptr<float>();
  }

  // Set scale modes for BlockWise1x16
  int a_scale_mode = get_scale_mode(mat1_scaling_type, mat1_scale_dtype, use_fast_accum);
  // Returns CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3
  int b_scale_mode = get_scale_mode(mat2_scaling_type, mat2_scale_dtype, use_fast_accum);

  #if CUDA_VERSION >= 12080
    computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_A_SCALE_MODE, a_scale_mode);
    computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_B_SCALE_MODE, b_scale_mode);
  #endif

  // Create matrix layouts
  CuBlasLtMatrixLayout Adesc(CUDA_R_4F_E2M1, m, k, lda, true /* transposed */);
  CuBlasLtMatrixLayout Bdesc(CUDA_R_4F_E2M1, k, n, ldb, false /* not transposed */);
  CuBlasLtMatrixLayout Ddesc(CUDA_R_16BF, m, n, result_ld);  // BFloat16 output
```

### Step 7c: Algorithm Selection & Execution (Lines 2112-2198)

```cpp
  // Query cuBLASLt for best algorithm
  CuBlasLtMatmulPreference preference;
  auto ltworkspace = CublasLtWorkspace();  // Allocate workspace memory
  preference.setAttribute(CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, ltworkspace.size);

  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  int returnedResult = 0;
  cublasLtHandle_t ltHandle = at::cuda::getCurrentCUDABlasLtHandle();

  // Heuristic selection - cuBLASLt picks best kernel
  TORCH_CUDABLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
      ltHandle,
      computeDesc.descriptor(),
      Adesc.descriptor(),
      Bdesc.descriptor(),
      Cdesc.descriptor(),
      Ddesc.descriptor(),
      preference.descriptor(),
      1,
      &heuristicResult,
      &returnedResult));

  TORCH_CHECK(returnedResult > 0, "No algorithm found");

  // Execute the GEMM
  cublasStatus_t cublasStatus = cublasLtMatmul(
      ltHandle,
      computeDesc.descriptor(),
      alpha_ptr,                // Pointer to alpha (global scale)
      mat1_ptr,                 // A data [1025, 256] fp4
      Adesc.descriptor(),
      mat2_ptr,                 // B data [256, 2048] fp4
      Bdesc.descriptor(),
      beta_ptr,                 // = 0.0 (no C matrix)
      dummy_C_ptr,              // Unused (beta=0)
      Cdesc.descriptor(),
      result_ptr,               // Output [1025, 2048] bf16
      Ddesc.descriptor(),
      &heuristicResult.algo,    // Selected kernel
      ltworkspace.ptr,          // Workspace buffer
      ltworkspace.size,
      stream);                  // CUDA stream

  TORCH_CHECK(cublasStatus == CUBLAS_STATUS_SUCCESS, "...");
}
```

### State After

cuBLASLt descriptor fully configured with:

- Matrix shapes: A=[1025,512], B=[512,2048] (logical dimensions)
- Data types: FP4 inputs → BF16 output
- Scaling: BlockWise1x16 with e4m3fn scales + device-side global scale
- Epilogue: Bias addition if provided
- Kernel selected and launched by cuBLASLt heuristic
## Frame 8: CUDA Kernel Execution (cuBLASLt Internal)

### What Happens

cuBLASLt selects and launches a CUDA kernel based on:

- Matrix dimensions (M=1025, N=2048, K=512)
- Data types (FP4 → BF16)
- Scaling configuration (BlockWise1x16 with SWIZZLE_32_4_4)
- Hardware (SM version, tensor core availability)

### Kernel Behavior

1. Load tiles of A and B into shared memory
2. Decode FP4 → FP8/FP16 using BlockWise1x16 scales:
   - Each block of 16 elements in K dimension shares one e4m3fn scale
   - Scales are swizzled in SWIZZLE_32_4_4 layout for coalesced access
3. Multiply global scales (alpha parameter) during accumulation
4. Accumulate in FP32 using tensor cores
5. Add bias (if provided) during epilogue
6. Convert FP32 → BF16 and store to output

### Actual Library

The kernel is part of NVIDIA cuBLASLt (closed-source), specifically the FP4 GEMM implementation added in CUDA 12.8+.

### Key Performance Details

- Uses Hopper tensor cores (SM 9.0) for FP4 matrix multiply
- BlockWise1x16 allows fine-grained quantization (better accuracy than tensorwise)
- Swizzled scale layout optimizes memory coalescing
- Two-level scaling (block + global) enables better dynamic range
## Summary: Complete Call Path

```
Python (torch.nn.functional.scaled_mm)
  ↓ [Normalize inputs, convert enums to ints]

torch._scaled_mm_v2 (auto-generated binding)
  ↓ [ATen dispatcher routes CUDA tensors]

at::native::_scaled_mm_cuda_v2 (C++ entry)
  ↓ [Allocate output tensor]

at::native::_scaled_mm_cuda_v2_out (main orchestrator)
  ↓ [Validate inputs, dispatch table lookup]
  │  - Checks 9 implementations in order
  │  - Matches: check_nvfp4_recipe → NVFP4_NVFP4

at::native::_scaled_nvfp4_nvfp4 (NVFP4 handler)
  ↓ [Combine global scales, validate swizzle]

at::native::_scaled_gemm (GEMM orchestrator)
  ↓ [Prepare cuBLAS arguments, route to backend]

at::cuda::blas::scaled_gemm (cuBLASLt wrapper)
  ↓ [Configure descriptors, set scale modes]
  │  - Scale mode: CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3
  │  - Alpha: device-side pointer to global scale
  │  - Heuristic selection

cublasLtMatmul (NVIDIA cuBLASLt)
  ↓ [Launch CUDA kernel on GPU]

[CUDA Kernel: FP4 GEMM with BlockWise1x16 + Global Scaling]
  - Decode FP4 → FP8/FP16 using swizzled e4m3fn scales
  - Apply global scale during accumulation
  - Tensor core multiply-accumulate (FP32 accum)
  - Bias epilogue
  - Convert FP32 → BF16

Output: [1025, 2048] BFloat16 tensor
```

## Key Takeaways

- **Multi-level dispatch**: Python → ATen YAML → C++ dispatch table → specialized handler
- **Two-level scaling**: BlockWise1x16 (fine-grained) + TensorWise (global adjustment)
- **Swizzled layout**: SWIZZLE_32_4_4 ensures coalesced GPU memory access
- **cuBLASLt backend**: NVIDIA's library handles kernel selection and execution
- **CUDA 12.8+ required**: BlockWise1x16 with e4m3fn scales needs recent cuBLASLt
- **Tensor core acceleration**: Hopper (SM 9.0) FP4 tensor cores provide the compute