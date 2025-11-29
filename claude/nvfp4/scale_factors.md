# NVFP4 Scale Factor Shapes and GEMM Dispatch

This document provides a frame-by-frame trace of:
1. The quantization kernel call (`x_quantizer.update_quantized`) and the exact shapes of returned scale factors
2. The GEMM dispatch (`tex.generic_gemm`) including any scale factor reshaping prior to cuBLAS

## Test File Context

Starting from [experiments/test_nvfp4_gemm_exact.py](../../experiments/test_nvfp4_gemm_exact.py)

```python
# Line 154-161: Quantization
x_nvfp4_native = x_quantizer.make_empty(
    x_shape, dtype=x_dtype, device=device, requires_grad=False
)
x_nvfp4_native = x_quantizer.update_quantized(x, x_nvfp4_native)  # <--- FOCUS 1

# Line 266-283: GEMM
y_native = tex.generic_gemm(  # <--- FOCUS 2
    w_nvfp4_native,
    transa,
    x_nvfp4_native,
    transb,
    ...
)
```

---

## Part 1: Quantization - `x_quantizer.update_quantized(x, x_nvfp4_native)`

### Frame 1: Python Entry Point

**Location**: [transformer_engine/pytorch/tensor/nvfp4_tensor.py:160-179](../../transformer_engine/pytorch/tensor/nvfp4_tensor.py#L160-L179)

```python
def update_quantized(
    self,
    src: torch.Tensor,
    dst: QuantizedTensor,
    *,
    noop_flag: Optional[torch.Tensor] = None,
) -> QuantizedTensor:

    assert isinstance(dst, NVFP4Tensor), f"Cannot store quantized NVFP4 in {type(dst)} type."

    # Make sure input is in expected format
    if not devices_match(src.device, dst.device):
        src = src.to(device=dst.device)
    if not src.is_contiguous():
        src = src.contiguous()

    # Launch cast kernel
    tex.quantize(src, self, dst, noop_flag)  # <--- Goes to C++ binding

    return dst
```

**Key Points**:
- `src`: Input tensor (e.g., shape `[M, K]` where M=256, K=512)
- `dst`: Pre-allocated NVFP4Tensor with storage for:
  - `_rowwise_data`: FP4 data `[M, K//2]` (packed, 2 FP4 values per byte)
  - `_rowwise_scale_inv`: FP8 E4M3 scale factors
  - `_amax_rowwise`: Global amax (FP32 scalar)
- Calls into C++ binding: `tex.quantize`

---

### Frame 2: C++ Binding - `tex.quantize`

**Location**: [transformer_engine/pytorch/csrc/extensions/cast.cpp:34-79](../../transformer_engine/pytorch/csrc/extensions/cast.cpp#L34-L79)

```cpp
py::object quantize(const at::Tensor &tensor, py::handle quantizer, const py::object &output,
                    std::optional<at::Tensor> noop_flag) {
  // Convert quantizer to C++ object
  auto quantizer_cpp = convert_quantizer(quantizer);  // <--- Returns NVFP4Quantizer*

  // Convert input tensor to C++ object
  auto input_contiguous = tensor.contiguous();
  auto input_cpp = makeTransformerEngineTensor(input_contiguous);

  // Initialize output tensor
  TensorWrapper output_cpp;
  py::object output_py;
  if (output.is_none()) {
    const auto shape = get_tensor_shape(input_cpp);
    const auto fake_dtype = input_cpp.dtype();
    std::tie(output_cpp, output_py) = quantizer_cpp->create_tensor(shape, fake_dtype);
  } else {
    std::tie(output_cpp, output_py) = quantizer_cpp->convert_and_update_tensor(output);
  }

  // Initialize no-op flag
  std::optional<TensorWrapper> noop_flag_cpp;
  if (noop_flag.has_value()) {
    noop_flag_cpp = makeTransformerEngineTensor(*noop_flag);
  }

  // Perform quantization
  quantizer_cpp->quantize(input_cpp, output_cpp, noop_flag_cpp);  // <--- Dispatch

  return output_py;
}
```

**Key Points**:
- Converts Python quantizer → C++ `NVFP4Quantizer`
- Wraps input/output tensors in `TensorWrapper`
- Calls `quantizer_cpp->quantize(...)`

---

### Frame 3: NVFP4Quantizer C++ Implementation

**Location**: [transformer_engine/pytorch/csrc/quantizer.cpp:1100-1200](../../transformer_engine/pytorch/csrc/quantizer.cpp#L1100-L1200) (approximate, file is 1713 lines)

The NVFP4Quantizer inherits from `Quantizer` and implements:

```cpp
void NVFP4Quantizer::quantize(const TensorWrapper& input, TensorWrapper& out,
                               const std::optional<TensorWrapper>& noop_flag) {
  if (input.numel() == 0) {
    return;
  }

  QuantizationConfigWrapper quant_config;
  // Set up RHT (Randomized Hadamard Transform) if needed
  if (with_rht) {
    quant_config.set_rht_config(rht_matrix.data_ptr(), rht_matrix_random_sign_mask_t);
  }

  // Set 2D quantization flag
  quant_config.set_2d_quantization(with_2d_quantization);

  if (noop_flag) {
    quant_config.set_noop_tensor(noop_flag->data());
  }

  NVTE_SCOPED_GIL_RELEASE({
    // Call the main TE quantization function
    nvte_quantize_v2(input.data(), out.data(), quant_config,
                     at::cuda::getCurrentCUDAStream());
  });
}
```

**Key Points**:
- Configures RHT matrix if `with_rht=True`
- Sets 2D quantization mode if `with_2d_quantization=True`
- Calls into `nvte_quantize_v2` (TE core library function)

---

### Frame 4: TE Core - `nvte_quantize_v2`

**Location**: [transformer_engine/common/cast/cast.cu](../../transformer_engine/common/cast/cast.cu) (entry point)

This dispatches to the appropriate kernel based on dtype. For NVFP4, it routes to:

**Dispatch Location**: [transformer_engine/common/cast/dispatch/quantize.cuh](../../transformer_engine/common/cast/dispatch/quantize.cuh)

```cpp
// Dispatch based on input/output dtype
if (output_dtype == DType::kFloat4E2M1) {
  // NVFP4 quantization
  nvfp4::launch_quantize_kernel(input, output, config, stream);
}
```

---

### Frame 5: NVFP4 Kernel Launch

**Location**: [transformer_engine/common/cast/nvfp4/quantize_nvfp4.cuh](../../transformer_engine/common/cast/nvfp4/quantize_nvfp4.cuh)

```cpp
template <bool COMPUTE_ACTIVATIONS, typename ParamOP, float (*OP)(float, const ParamOP &),
          typename IType, typename OType, bool COLWISE_SCALING, size_t CHUNK_DIM_Y,
          size_t CHUNK_DIM_X, size_t THREADS_PER_CHUNK>
__global__ void __launch_bounds__(THREADS_PER_CHUNK)
    quantize_nvfp4_kernel(
        const __grid_constant__ CUtensorMap tensor_map_input,
        const __grid_constant__ CUtensorMap tensor_map_output_rowwise,
        const __grid_constant__ CUtensorMap tensor_map_output_colwise,
        fp8e4m3 *const scales_rowwise_e4m3,      // <--- Output scale factors (FP8 E4M3)
        e8m0_t *const scales_colwise_e8m0,       // <--- Output scale factors (FP8 E8M0)
        const float *noop,
        float *const amax_ptr,                    // <--- Global amax
        const float *const nvfp4_second_stage_scale_ptr, // <--- Global decode scale
        const size_t rows,
        const size_t cols,
        const size_t scale_stride_rowwise,       // <--- Stride between scale factor rows
        const size_t scale_stride_colwise) {
```

**Kernel Parameters**:

From lines 55-62, we see the kernel signature. The key scale-related parameters:

1. **`scales_rowwise_e4m3`**: Pointer to FP8 E4M3 scale factors (blockwise, 1×16 blocks)
2. **`scales_colwise_e8m0`**: Pointer to FP8 E8M0 scale factors (blockwise, 32×1 blocks for MXFP8)
3. **`amax_ptr`**: Global amax (single FP32 value)
4. **`nvfp4_second_stage_scale_ptr`**: Second-stage global scale factor (single FP32 value)

---

### Frame 6: Scale Factor Computation (Rowwise)

**Location**: [transformer_engine/common/cast/nvfp4/quantize_nvfp4.cuh:299-450](../../transformer_engine/common/cast/nvfp4/quantize_nvfp4.cuh#L299-L450)

The kernel computes scale factors in two passes:

#### Pass 1: Compute Local Block Amax

```cpp
// Lines 311-373: Find block amax
if constexpr (ROWWISE_SCALING) {
  const int stage_rowwise_scales_offset_Y = stage * BUFF_DIM_Y;
  #pragma unroll
  for (int it = 0; it < ITERATIONS_ROWWISE; ++it) {
    // ...
    block_amax = 0.0f;

    // 1. Read/Compute elements. Find NVFP4-block AMAX
    // Each thread processes SCALE_DIM_X=16 elements
    for (int w = 0; w < WAVES; ++w) {
      // Load 8 elements at a time (PACK_SIZE=8)
      // Compute abs max across all loaded elements
      block_amax = fmaxf(block_amax, fabsf(elt));
    }
```

#### Pass 2: Convert to FP8 E4M3 Scale Factors

```cpp
// Lines 450-480: Compute and store scale factors
// Reduce amax across threads in a warp
float block_amax_reduce = warp_reduce_max(block_amax);

// Compute global encoding scale: S_enc = (FP8_MAX * FP4_MAX) / global_amax
const float S_enc = (nvfp4_second_stage_scale_ptr == nullptr)
                    ? 1.0f
                    : 1.0f / (*nvfp4_second_stage_scale_ptr);

// Compute local decode scale: S_dec_b = block_amax / FP4_MAX
const float S_dec_b = block_amax_reduce / 6.0f;  // FP4_MAX = 6.0

// Quantize to FP8 E4M3: S_dec_b_e4m3 = (S_dec_b * S_enc) in FP8 E4M3
const float scaled_decode = S_dec_b * S_enc;
const fp8e4m3 S_dec_b_e4m3 = float_to_fp8e4m3(scaled_decode);

// Compute actual encoding scale: S_enc_b = S_enc / S_dec_b_e4m3
const float S_enc_b = S_enc / static_cast<float>(S_dec_b_e4m3);

// Store scale factor
const int global_scales_offset_Y = scales_offset_Y_rowwise + it_offset_Y;
const int global_scales_offset_X = scales_offset_X_rowwise;
const int scale_idx = global_scales_offset_Y * scale_stride_rowwise + global_scales_offset_X;

if (rowwise_scale_is_within_bounds) {
  scales_rowwise_e4m3[scale_idx] = S_dec_b_e4m3;  // <--- WRITE SCALE FACTOR
}
```

---

## Scale Factor Layout and Shapes

### Constants

From [quantize_nvfp4.cuh:34-48](../../transformer_engine/common/cast/nvfp4/quantize_nvfp4.cuh#L34-L48):

```cpp
constexpr size_t SCALE_DIM_Y = 32;  // Rows per scale factor block (for MXFP8/columnwise)
constexpr size_t SCALE_DIM_X = 16;  // Columns per scale factor block (for NVFP4/rowwise)
```

### Shape Calculation (Python)

**Location**: [transformer_engine/pytorch/tensor/nvfp4_tensor.py:215-249](../../transformer_engine/pytorch/tensor/nvfp4_tensor.py#L215-L249)

```python
def get_scale_shape(self, shape: Iterable[int], columnwise: bool) -> Tuple[int, int]:
    """Calculate the shape of the scaling tensor for NVFP4 1D blockwise quantization.

    Returns
    -------
    Tuple[int, int]
        Shape of the scaling tensor as (outer_dim, inner_dim)
        For NVFP4 1D blockwise quantization, blocksize is 16
        - If columnwise: (round_to_multiple(K, 128), round_to_multiple(roundup(M / 16), 4))
        - If rowwise: (round_to_multiple(M, 128), round_to_multiple(roundup(K / 16), 4))

    Swizzle kernel will be performed before GEMM to suit the need of CuBLAS.
    CuBLAS doc: https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout
    """
    M, K = 1, 1
    M = math.prod(shape[:-1])
    K = shape[-1]

    if columnwise:
        outer = round_up_to_nearest_multiple(K, 128)
        inner = round_up_to_nearest_multiple(math.ceil(M / NVFP4_BLOCK_SCALING_SIZE), 4)
        return (outer, inner)
    # rowwise
    outer = round_up_to_nearest_multiple(M, 128)
    inner = round_up_to_nearest_multiple(math.ceil(K / NVFP4_BLOCK_SCALING_SIZE), 4)
    return (outer, inner)
```

### Example: Input Shape [256, 512]

For `x` with shape `[M=256, K=512]`:

**Rowwise scaling** (1×16 blocks):
- Number of blocks in K dimension: `K // 16 = 512 // 16 = 32`
- Padded M: `round_up(256, 128) = 256`
- Padded K blocks: `round_up(32, 4) = 32`
- **Scale shape**: `[256, 32]` (stored as `uint8`, interpreted as FP8 E4M3)

Each scale factor covers one 1×16 block of input elements.

### Exact Memory Layout

**From the test file** [experiments/test_nvfp4_gemm_exact.py:174-192](../../experiments/test_nvfp4_gemm_exact.py#L174-L192):

```python
# Extract quantized data from native NVFP4Tensors
sx_native = (
    x_nvfp4_native._columnwise_scale_inv if x_columnwise
    else x_nvfp4_native._rowwise_scale_inv
)

# Trim quantized data to match the actual tensor dimensions (remove padding)
qx_data = qx_data[:M, :]

expected_sx_cols = K // BLOCK_LENGTH  # BLOCK_LENGTH = 16
# Trim the scales to remove padding
sx_trimmed = sx_native[:M, :expected_sx_cols]  # Shape: [256, 32]

# Native scales are stored as uint8 but need to be interpreted as float8_e4m3fn
# for the reference GEMM to work correctly
sx_trimmed = sx_trimmed.view(torch.float8_e4m3fn)  # Shape: [256, 32] as FP8 E4M3
```

**Key Observations**:
1. **Raw storage**: `[M_padded, K_blocks_padded]` = `[256, 32]` as `torch.uint8`
2. **After trimming**: `[M, K // 16]` = `[256, 32]` as `torch.uint8`
3. **After reinterpretation**: `[256, 32]` as `torch.float8_e4m3fn`

Each element in the scale tensor is an FP8 E4M3 value representing the decode scale for a 1×16 block.

---

## Part 2: GEMM - `tex.generic_gemm(...)`

### Frame 1: Python Entry Point

**Location**: [experiments/test_nvfp4_gemm_exact.py:266-283](../../experiments/test_nvfp4_gemm_exact.py#L266-L283)

```python
y_native = tex.generic_gemm(
    w_nvfp4_native,              # A: [N, K] NVFP4Tensor
    transa,                       # True (transpose A)
    x_nvfp4_native,              # B: [M, K] NVFP4Tensor
    transb,                       # False (no transpose B)
    out.clone() if accumulate else None,  # D: output tensor
    out_quantizer,                # None
    TE_DType[out_dtype],         # Output dtype (float32)
    bias,                         # None
    bias_dtype,                   # bfloat16
    use_gelu,                     # False
    gelu_input,                   # None
    use_grad,                     # False
    workspace,                    # Workspace buffer [4] uint8
    workspace.shape[0],          # Workspace size
    accumulate,                   # True/False
    use_split_accumulator,       # False
)[0]
```

**Matrix Dimensions**:
- W: `[N, K]` → After transpose: `[K, N]`
- X: `[M, K]` → No transpose: `[M, K]`
- Y: `[M, N]` (output)

**GEMM Layout**: Effectively `Y = X @ W^T` (computes `[M, K] @ [K, N] → [M, N]`)

---

### Frame 2: Python Wrapper

**Location**: [transformer_engine/pytorch/cpp_extensions/gemm.py:92-210](../../transformer_engine/pytorch/cpp_extensions/gemm.py#L92-L210)

```python
def general_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    out_dtype: Optional[torch.dtype] = None,
    ...
) -> Iterable[Optional[torch.Tensor]]:
    """GEMM supporting fp8 inputs."""

    assert layout in ("TN", "NN", "NT"), f"GEMM layout {layout} not supported."
    transa = layout[0] == "T"
    transb = layout[1] == "T"

    # ... validation ...

    workspace = get_cublas_workspace(get_tensor_device(A), ub is not None, False)

    # If A or B are custom tensors -> dispatch to quantizers's qgemm implementation
    if is_custom(A) or is_custom(B):
        return custom_gemm(A, B, workspace, ...)  # <--- NOT taken for NVFP4

    # ... setup bias, gelu, etc. ...

    args = (
        A, transa, B, transb, out, quantization_params, TE_DType[out_dtype],
        bias, bias_dtype, gelu, gelu_in, grad, workspace, workspace.shape[0],
        accumulate, use_split_accumulator,
    )
    kwargs = {...}

    out, bias_grad, gelu_input, extra_output = tex.generic_gemm(*args, **kwargs)

    return out, bias_grad, gelu_input, extra_output
```

**Key Points**:
- NVFP4Tensor is NOT a "custom" tensor in the sense checked by `is_custom()`
- Goes directly to `tex.generic_gemm` C++ binding

---

### Frame 3: C++ GEMM Entry

**Location**: [transformer_engine/pytorch/csrc/extensions/gemm.cpp:89-152](../../transformer_engine/pytorch/csrc/extensions/gemm.cpp#L89-L152)

```cpp
std::vector<py::object> gemm(
    py::handle A, bool transa,
    py::handle B, bool transb,
    py::object D,
    py::handle quantizer,
    std::optional<DType> out_dtype,
    MaybeTensor bias,
    DType bias_type,
    bool gelu,
    MaybeTensor gelu_in,
    bool grad,
    at::Tensor workspace,
    size_t workspaceSize,
    bool accumulate,
    bool use_split_accumulator,
    CommOverlapCore* comm_overlap,
    std::optional<CommOverlapType> comm_type,
    MaybeTensor extra_output,
    bool bulk_overlap,
    float alpha,
    std::optional<float> beta) {
  using namespace transformer_engine::pytorch::detail;

  // Input tensors
  NVTE_CHECK(!A.is_none(), "Tensor A has not been provided");
  NVTE_CHECK(!B.is_none(), "Tensor B has not been provided");
  auto none = py::none();
  TensorWrapper A_tensor = makeTransformerEngineTensor(A, none);  // <--- Convert to TE tensor
  TensorWrapper B_tensor = makeTransformerEngineTensor(B, none);  // <--- Convert to TE tensor

  // Check tensor dimensions
  const auto& A_shape = A_tensor.shape();
  const auto& B_shape = B_tensor.shape();
  const auto& D_shape = detail::getGemmOutputShape(A_shape, transa, B_shape, transb);

  // Output tensor
  TensorWrapper D_tensor;
  if (D.is_none()) {
    std::tie(D_tensor, D) = createOutputTensor(D_shape, output_dtype, quantizer);
  } else {
    D_tensor = makeTransformerEngineTensor(D, quantizer);
  }

  // ... setup workspace, bias, gelu tensors ...

  // Keep the swizzled scaling factor tensors alive during the GEMM.
  std::vector<std::optional<at::Tensor>> swizzled_scale_inverses_list;
  auto main_stream = at::cuda::getCurrentCUDAStream();
  if (A_tensor.numel() != 0 && B_tensor.numel() != 0) {
    // Optionally swizzle the scaling factors  <--- KEY: Scale factor preprocessing
    swizzled_scale_inverses_list.emplace_back(
        std::move(swizzle_scaling_factors(A_tensor, transa)));  // <--- SWIZZLE A scales
    swizzled_scale_inverses_list.emplace_back(
        std::move(swizzle_scaling_factors(B_tensor, !transb)));  // <--- SWIZZLE B scales
```

**Key Points**:
- Converts Python NVFP4Tensor → C++ `TensorWrapper`
- **Critical**: Calls `swizzle_scaling_factors()` for both A and B tensors
- This reshapes/reorders scale factors to match cuBLAS requirements

---

### Frame 4: Scale Factor Swizzling

**Location**: [transformer_engine/pytorch/csrc/util.cpp:12-83](../../transformer_engine/pytorch/csrc/util.cpp#L12-L83)

The `swizzle_scaling_factors` function transforms the scale factor layout from the quantization kernel's output format to cuBLAS's expected format.

**C++ Wrapper**:

```cpp
std::optional<at::Tensor> swizzle_scaling_factors(
    transformer_engine::TensorWrapper& input,
    bool rowwise) {
  using namespace transformer_engine::pytorch;

  if (input.scaling_mode() == NVTE_INVALID_SCALING) {
    NVTE_ERROR("Invalid scaling mode for swizzle.");
  } else if (input.scaling_mode() != NVTE_MXFP8_1D_SCALING &&
             input.scaling_mode() != NVTE_NVFP4_1D_SCALING) {
    return std::nullopt;  // No swizzling needed for per-tensor scaling
  }

  const auto nvfp4 = input.scaling_mode() == NVTE_NVFP4_1D_SCALING;

  // Get scale_inv from tensor
  NVTEBasicTensor scale_inv;
  if (rowwise) {
    scale_inv = input.get_rowwise_scale_inv();
  } else {
    scale_inv = input.get_columnwise_scale_inv();
  }

  auto scale_inv_shape = nvte_shape_to_vector(scale_inv.shape);

  // Allocate memory for swizzled output
  auto options = at::TensorOptions().dtype(torch::kByte).device(torch::kCUDA);
  auto swizzled_scale_inv = at::empty(scale_inv_shape, options);

  // Wrap tensors for TE API
  transformer_engine::TensorWrapper input_cu(input.scaling_mode());
  transformer_engine::TensorWrapper output_cu(input.scaling_mode());

  const auto scale_inv_dtype = (nvfp4)
    ? transformer_engine::DType::kFloat8E4M3
    : transformer_engine::DType::kFloat8E8M0;

  if (rowwise) {
    input_cu.set_rowwise_data(input.dptr(), input_dtype, input_shape);
    input_cu.set_rowwise_scale_inv(scale_inv.data_ptr, scale_inv_dtype, scale_inv_shape);
    output_cu.set_rowwise_data(input.dptr(), input_dtype, input_shape);
    output_cu.set_rowwise_scale_inv(swizzled_scale_inv.data_ptr(), scale_inv_dtype, scale_inv_shape);
  } else {
    // ... columnwise setup ...
  }

  // Launch kernel
  nvte_swizzle_scaling_factors(input_cu.data(), output_cu.data(),
                                at::cuda::getCurrentCUDAStream());

  // Update input tensor to point to swizzled scales
  if (rowwise) {
    input.set_rowwise_scale_inv(swizzled_scale_inv.data_ptr(), scale_inv_dtype, scale_inv_shape);
  } else {
    input.set_columnwise_scale_inv(swizzled_scale_inv.data_ptr(), scale_inv_dtype, scale_inv_shape);
  }

  return swizzled_scale_inv;
}
```

**CUDA Swizzling Kernel**:

**Location**: [transformer_engine/common/swizzle/swizzle.cu:173-240](../../transformer_engine/common/swizzle/swizzle.cu#L173-L240)

The rowwise swizzling kernel for NVFP4:

```cpp
template <typename LType, int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__device__ void swizzle_row_scaling_kernel_impl(
    const void* input,      // Input: [M, K/4] int32 (K-major, 4 scales per int)
    void* output,           // Output: [M, K/4] int32 (swizzled layout)
    const int M,            // Padded M dimension
    const int K,            // Padded K dimension (in scales, not elements)
    const int original_M,   // Original M before padding
    const int original_K,   // Original K before padding
    ...) {

  constexpr int N_TILE_PER_TD = sizeof(LType) / sizeof(int);
  constexpr int SF_TILE_SIZE_I32 = SF_TILE_DIM_M * SF_TILE_DIM_K / 4;

  // Load scales from global memory into registers
  // Input layout: K-major (scales stored column-by-column)
  LType regs_vec[N_SF_PER_TD_PER_TILE];
  for (int i = 0; i < N_SF_PER_TD_PER_TILE; i++) {
    const int thread_offset = (i * TB_DIM + threadIdx.y) * K_i32 + threadIdx.x * N_TILE_PER_TD;
    regs_vec[i] = __ldg(reinterpret_cast<const LType*>(input_i32 + thread_offset));
  }

  // Shuffle scales within registers
  // This reorders scales to match cuBLAS tile access pattern
  regs_shuffle<LType>(regs_vec);

  // Store shuffled scales to shared memory
  for (int i = 0; i < N_TILE_PER_TD; i++) {
    slm_v4i[(threadIdx.x * N_TILE_PER_TD + i) * SF_TILE_SIZE_I32 / 4 + threadIdx.y] =
        reinterpret_cast<int4*>(regs_vec)[i];
  }
  __syncthreads();

  // Copy from shared memory to global output
  // Now in swizzled layout suitable for cuBLAS
  int linear_id = threadIdx.y * blockDim.x + threadIdx.x;
  for (int i = linear_id; i < SF_TILE_SIZE_I32 * n_tiles_in_tb / 4; i += blockDim.x * blockDim.y) {
    output_v4i[i] = slm_v4i[i];
  }
}
```

**Swizzling Details**:

The swizzle operation performs a 2D transpose with tiling:

1. **Input Layout (K-major)**:
   - Scale factors stored in K-major order: `scales[m, k] = input[k * M + m]`
   - Each `int32` packs 4 consecutive FP8 E4M3 scales

2. **Tile Processing**:
   - Tiles of size `[SF_TILE_DIM_M=32, SF_TILE_DIM_K=16]`
   - Each thread block processes multiple tiles
   - Within each tile, scales are reordered using `regs_shuffle()`

3. **Output Layout (Swizzled)**:
   - Optimized for cuBLAS Tensor Core access pattern
   - Ensures coalesced memory accesses during GEMM
   - Still has shape `[M, K//16]` but internal ordering changed

4. **Why Swizzling?**:
   - cuBLAS loads FP4 data and scales in specific tile patterns (e.g., 128×128 GEMM tiles)
   - Swizzled layout ensures:
     - Coalesced memory accesses (32 consecutive threads access consecutive addresses)
     - Reduced bank conflicts in shared memory
     - Efficient scale factor broadcast to multiple warps

---

### Frame 5: cuBLAS GEMM Call

**Location**: [transformer_engine/pytorch/csrc/extensions/gemm.cpp:305-340](../../transformer_engine/pytorch/csrc/extensions/gemm.cpp#L305-L340)

After swizzling, the code calls the cuBLAS GEMM:

```cpp
// ... inside gemm() function ...

// Construct GEMM config
transformer_engine::MatmulConfigWrapper config;
if (grad) {
  config.set_dbias_tensor(bias_tensor.data());
  config.set_with_dgelu_epilogue(gelu);
} else {
  config.set_bias_tensor(bias_tensor.data());
  config.set_with_gelu_epilogue(gelu);
}
config.set_epilogue_aux_tensor(te_pre_gelu_out.data());
config.set_use_split_accumulator(use_split_accumulator);
config.set_sm_count(num_math_sms);

// ... (swizzling happens here) ...

// Call TE GEMM
NVTE_SCOPED_GIL_RELEASE({
  nvte_cublas_gemm(
      A_tensor.data(),  // Now has swizzled scale factors
      B_tensor.data(),  // Now has swizzled scale factors
      D_tensor.data(),
      bias_tensor.data(),
      te_pre_gelu_out.data(),
      transa, transb,
      grad,
      te_workspace.data(),
      accumulate,
      use_split_accumulator,
      &config,
      main_stream
  );
});
```

**Key Points**:
- Scale factors have been swizzled in-place in `A_tensor` and `B_tensor`
- cuBLAS accesses scale factors during GEMM via the pointers in `NVTETensor`
- cuBLAS interprets scales according to the NVFP4 block scaling format:
  - 1×16 blocks for rowwise data
  - FP8 E4M3 scale factors
  - Global FP32 decode scale

---

### Frame 6: cuBLAS Internal (Blackwell)

**Location**: NVIDIA cuBLAS library (closed source)

On Blackwell (SM 10.0+), cuBLAS uses the `scaled_mm` API with:

```
ScalingType::BlockWise1x16  (for NVFP4)
ScalingType::TensorWise     (for global scale)
```

The GEMM kernel:
1. Loads FP4 data in packed format (2 values per byte)
2. Loads FP8 E4M3 local scale factors (1 per 16 elements)
3. Loads FP32 global scale factor
4. Performs:
   - Dequantize: `x_block_fp32 = x_fp4 * scale_fp8_e4m3 * global_scale_fp32`
   - Matrix multiply accumulate in FP32
   - Write output in requested dtype (FP32/BF16)

---

## Summary: Scale Factor Flow

### Quantization Path

```
Input: [M, K] BF16 tensor
  ↓
Quantize kernel (CUDA)
  ├─ FP4 data: [M, K/2] packed (2 FP4 per byte)
  ├─ Local scales: [M, K/16] FP8 E4M3 (1 per 1×16 block)
  └─ Global scale: scalar FP32
  ↓
Stored in NVFP4Tensor._rowwise_scale_inv: [M_pad, K_blocks_pad] uint8
  where M_pad = roundup(M, 128), K_blocks_pad = roundup(K/16, 4)
```

### GEMM Path

```
NVFP4Tensor with scales [M_pad, K_blocks_pad]
  ↓
swizzle_scaling_factors()
  ├─ Input: [M_pad, K_blocks_pad] linear layout
  └─ Output: [M_pad, K_blocks_pad] swizzled layout (matches cuBLAS tiles)
  ↓
nvte_cublas_gemm()
  ├─ cuBLAS reads swizzled scales
  ├─ Performs FP4 GEMM with block scaling
  └─ Output: [M, N] FP32/BF16
```

### Key Shape Transformations

| Stage | Shape | Dtype | Notes |
|-------|-------|-------|-------|
| Input | `[M, K]` | BF16 | Original tensor |
| FP4 Data | `[M, K/2]` | uint8 | 2 FP4 values per byte |
| Local Scales (kernel output) | `[M, K/16]` | FP8 E4M3 | 1 scale per 16 elements |
| Local Scales (padded) | `[M_pad, K_blocks_pad]` | uint8 | M_pad=roundup(M,128), K_blocks_pad=roundup(K/16,4) |
| Local Scales (swizzled) | `[M_pad, K_blocks_pad]` | uint8 | Reordered for cuBLAS |
| Global Scale | `[1]` | FP32 | Single scalar |
| GEMM Output | `[M, N]` | FP32 | Final result |

---

## Code Reference Quick Links

### Quantization
- [experiments/test_nvfp4_gemm_exact.py:157](../../experiments/test_nvfp4_gemm_exact.py#L157) - Test entry point
- [transformer_engine/pytorch/tensor/nvfp4_tensor.py:160-179](../../transformer_engine/pytorch/tensor/nvfp4_tensor.py#L160-L179) - Python quantizer
- [transformer_engine/pytorch/csrc/extensions/cast.cpp:34-79](../../transformer_engine/pytorch/csrc/extensions/cast.cpp#L34-L79) - C++ binding
- [transformer_engine/common/cast/nvfp4/quantize_nvfp4.cuh:54-62](../../transformer_engine/common/cast/nvfp4/quantize_nvfp4.cuh#L54-L62) - CUDA kernel signature
- [transformer_engine/common/cast/nvfp4/quantize_nvfp4.cuh:450-480](../../transformer_engine/common/cast/nvfp4/quantize_nvfp4.cuh#L450-L480) - Scale computation

### Scale Shape Calculation
- [transformer_engine/pytorch/tensor/nvfp4_tensor.py:215-249](../../transformer_engine/pytorch/tensor/nvfp4_tensor.py#L215-L249) - `get_scale_shape()`
- [transformer_engine/pytorch/tensor/nvfp4_tensor.py:47](../../transformer_engine/pytorch/tensor/nvfp4_tensor.py#L47) - `NVFP4_BLOCK_SIZE = 16`

### GEMM
- [experiments/test_nvfp4_gemm_exact.py:266](../../experiments/test_nvfp4_gemm_exact.py#L266) - Test GEMM call
- [transformer_engine/pytorch/cpp_extensions/gemm.py:92-210](../../transformer_engine/pytorch/cpp_extensions/gemm.py#L92-L210) - Python wrapper
- [transformer_engine/pytorch/csrc/extensions/gemm.cpp:89-152](../../transformer_engine/pytorch/csrc/extensions/gemm.cpp#L89-L152) - C++ entry
- [transformer_engine/pytorch/csrc/extensions/gemm.cpp:238-245](../../transformer_engine/pytorch/csrc/extensions/gemm.cpp#L238-L245) - Scale swizzling call

### cuBLAS Documentation
- [NVIDIA cuBLAS Block Scaling Layout](https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout)

---

## Additional Notes

1. **Why Swizzling?**
   - cuBLAS Tensor Cores access data in specific tile patterns
   - Swizzled layout ensures coalesced memory access
   - Improves performance by reducing memory bottlenecks

2. **Two-Level Scaling**
   - **Local scales** (FP8 E4M3): Per-block decode scales
   - **Global scale** (FP32): Applied to all local scales
   - This provides better dynamic range and precision

3. **Padding**
   - Scales are padded to meet cuBLAS alignment requirements:
     - M dimension: multiple of 128
     - K blocks dimension: multiple of 4
   - Padding is handled transparently by TE

4. **Test Verification**
   - [experiments/test_nvfp4_gemm_exact.py:174-192](../../experiments/test_nvfp4_gemm_exact.py#L174-L192) shows how to extract and inspect scales
   - Scales are stored as `uint8` but reinterpreted as `float8_e4m3fn`
   - Global scale is accessed via `x_nvfp4_native._amax_rowwise` (FP32)
