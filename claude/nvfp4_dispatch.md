# NVFP4 Quantization Call Path Trace (with RHT)

This document traces the complete call path for NVFP4 quantization with Random Hadamard Transform (RHT), from Python through C++ bindings to CUDA kernels and back.

## Context

The trace is based on the call: `ref_quantizer.quantize(x)` where:
- `ref_quantizer` is an instance of `NVFP4QuantizerRef` (reference implementation)
- However, this document traces the **production implementation** (non-Ref) which uses C++/CUDA
- Configuration: `with_rht=True` (Random Hadamard Transform enabled)

## Call Path Overview

```
Python: tex.quantize() or NVFP4Quantizer.update_quantized()
    ↓
C++ Binding: transformer_engine::pytorch::quantize()
    ↓
C++ Quantizer: NVFP4Quantizer::quantize()
    ↓
C API Dispatcher: nvte_quantize_v2() → dispatch::quantize_fwd_helper()
    ↓
CUDA Kernels: nvte_hadamard_transform_amax() + nvte_hadamard_transform_cast_fusion_columnwise()
    ↓
Return: NVFP4Tensor with quantized data
```

---

## Layer 1: Python Entry Point

### File: [transformer_engine/pytorch/tensor/nvfp4_tensor.py](../transformer_engine/pytorch/tensor/nvfp4_tensor.py#L160-L179)

```python
def update_quantized(
    self,
    src: torch.Tensor,
    dst: QuantizedTensor,
    *,
    noop_flag: Optional[torch.Tensor] = None,
) -> QuantizedTensor:
    # ... validation code ...

    # Launch cast kernel
    tex.quantize(src, self, dst, noop_flag)  # Line 177

    return dst
```

**Key Points:**
- Called from `NVFP4Quantizer.update_quantized()` or via `__call__`
- `tex.quantize` is the C++ extension binding
- `self` is the NVFP4Quantizer instance containing RHT configuration
- `dst` is a pre-allocated NVFP4Tensor

**Alternative Entry**: Direct call via `tex.quantize(tensor, quantizer)` at [transformer_engine/pytorch/csrc/extensions/pybind.cpp:120](../transformer_engine/pytorch/csrc/extensions/pybind.cpp#L120)

---

## Layer 2: C++ Binding Layer

### File: [transformer_engine/pytorch/csrc/extensions/pybind.cpp:120-121](../transformer_engine/pytorch/csrc/extensions/pybind.cpp#L120-L121)

```cpp
m.def("quantize", transformer_engine::pytorch::quantize, py::arg("tensor"), py::arg("quantizer"),
      py::arg("output") = py::none(), py::arg("noop") = py::none());
```

### File: [transformer_engine/pytorch/csrc/extensions/cast.cpp:34-80](../transformer_engine/pytorch/csrc/extensions/cast.cpp#L34-L80)

```cpp
py::object quantize(const at::Tensor &tensor, py::handle quantizer,
                    const py::object &output, std::optional<at::Tensor> noop_flag) {
  // Convert quantizer to C++ object
  auto quantizer_cpp = convert_quantizer(quantizer);  // Line 37

  // Convert input tensor to C++ object
  auto input_contiguous = tensor.contiguous();
  auto input_cpp = makeTransformerEngineTensor(input_contiguous);  // Line 41

  // Initialize output tensor
  TensorWrapper output_cpp;
  py::object output_py;
  if (output.is_none()) {
    // Allocate new output
    std::tie(output_cpp, output_py) = quantizer_cpp->create_tensor(shape, fake_dtype);
  } else {
    // Use provided output
    std::tie(output_cpp, output_py) = quantizer_cpp->convert_and_update_tensor(output);
  }

  // Initialize no-op flag
  std::optional<TensorWrapper> noop_flag_cpp;
  if (noop_flag.has_value()) {
    noop_flag_cpp = makeTransformerEngineTensor(*noop_flag);
  }

  // Perform quantization
  quantizer_cpp->quantize(input_cpp, output_cpp, noop_flag_cpp);  // Line 76

  return output_py;
}
```

**Key Points:**
- Converts Python quantizer to `NVFP4Quantizer` C++ object
- Wraps PyTorch tensors in `TensorWrapper` for C API
- Calls virtual `quantize()` method on quantizer

---

## Layer 3: C++ Quantizer Implementation

### File: [transformer_engine/pytorch/csrc/quantizer.cpp:1444-1648](../transformer_engine/pytorch/csrc/quantizer.cpp#L1444-L1648)

```cpp
void NVFP4Quantizer::quantize(const TensorWrapper &input, TensorWrapper &out,
                              const std::optional<TensorWrapper> &noop_flag) const {
  // ... early exit for empty tensors ...

  auto stream = at::cuda::getCurrentCUDAStream();  // Line 1454

  // Setup quantization config
  QuantizationConfigWrapper quant_config;
  if (noop_flag) {
    quant_config.set_noop_tensor(noop_flag->data());
  }
  quant_config.set_nvfp4_2d_quantization(this->with_2d_quantization);  // Line 1460
  quant_config.set_stochastic_rounding(this->stochastic_rounding);  // Line 1461

  // Get tensor dimensions
  size_t rows = 1;
  for (size_t i = 0; i < input.ndim() - 1; ++i) {
    rows *= input.size(i);
  }
  size_t cols = input.size(input.ndim() - 1);  // Line 1469

  // Setup stochastic rounding RNG state if needed
  TensorWrapper te_rng_state;
  if (this->stochastic_rounding) {
    // ... generate Philox RNG state ... (Lines 1472-1482)
  }

  // Check if eligible for optimized RHT+cast fusion kernel
  bool eligible_for_rht_cast_fusion =
      input.dtype() == DType::kBFloat16 && rows % 64 == 0 && cols % 128 == 0;  // Line 1485-1486

  // ====================================================================
  // STEP 1: Compute amax with RHT
  // ====================================================================
  if (this->with_rht) {  // Line 1489
    NVTE_CHECK(input.dtype() == DType::kBFloat16, "RHT only supports bfloat16");

    if (this->with_post_rht_amax) {  // Line 1493
      // Compute:
      // 1. Rowwise amax = max(|input|)
      // 2. Columnwise amax = max(|RHT(input.t)|)
      NVTE_SCOPED_GIL_RELEASE({
        nvte_hadamard_transform_amax(input.data(), out.data(), 0,
                                     this->rht_matrix_random_sign_mask_t, stream);  // Line 1498-1499
      });
    } else {
      NVTE_CHECK(false, "Pre-RHT amax is not supported yet");
    }
  }

  // ====================================================================
  // STEP 2: Amax reduction (if distributed training)
  // ====================================================================
  if (this->with_amax_reduction) {
    // ... allreduce amax across ranks ... (Lines 1532-1551)
  }

  // ====================================================================
  // STEP 3: Quantization
  // ====================================================================
  if (this->with_rht) {  // Line 1553

    // --- 3A: Rowwise quantization (direct, no RHT) ---
    if (rowwise_usage) {  // Line 1554
      // Extract rowwise output components
      TensorWrapper out_identity(out.scaling_mode());
      auto out_identity_data = out.get_rowwise_data();
      auto out_identity_scale_inv = out.get_rowwise_scale_inv();
      auto out_identity_amax = out.get_amax();
      out_identity.set_rowwise_data(out_identity_data.data_ptr,
                                    static_cast<DType>(out_identity_data.dtype),
                                    out_identity_data.shape);  // Lines 1560-1562
      out_identity.set_rowwise_scale_inv(out_identity_scale_inv.data_ptr,
                                         static_cast<DType>(out_identity_scale_inv.dtype),
                                         out_identity_scale_inv.shape);  // Lines 1563-1565
      out_identity.set_amax(out_identity_amax.data_ptr,
                           static_cast<DType>(out_identity_amax.dtype),
                           out_identity_amax.shape);  // Lines 1566-1567

      // Quantize input directly (no RHT for rowwise)
      NVTE_SCOPED_GIL_RELEASE({
        nvte_quantize_v2(input.data(), out_identity.data(), quant_config, stream);  // Line 1570
      });
    }

    // --- 3B: Columnwise quantization (with RHT) ---
    if (columnwise_usage) {  // Line 1573
      // Extract columnwise output components
      auto out_columnwise_data = out.get_columnwise_data();
      auto out_columnwise_scale_inv = out.get_columnwise_scale_inv();
      auto out_columnwise_amax = out.get_columnwise_amax();  // Lines 1575-1578

      // Create wrapper treating columnwise as rowwise (already transposed)
      TensorWrapper out_transpose(out.scaling_mode());

      // Flatten to 2D
      auto colwise_data_shape = out_columnwise_data.shape;
      std::vector<size_t> colwise_data_shape_2d;
      colwise_data_shape_2d.push_back(colwise_data_shape.data[0]);
      size_t last_dim = 1;
      for (size_t i = 1; i < colwise_data_shape.ndim; ++i) {
        last_dim *= colwise_data_shape.data[i];
      }
      colwise_data_shape_2d.push_back(last_dim);  // Lines 1586-1596

      out_transpose.set_rowwise_data(out_columnwise_data.data_ptr,
                                     static_cast<DType>(out_columnwise_data.dtype),
                                     colwise_data_shape_2d);  // Lines 1598-1600
      out_transpose.set_rowwise_scale_inv(out_columnwise_scale_inv.data_ptr,
                                          static_cast<DType>(out_columnwise_scale_inv.dtype),
                                          out_columnwise_scale_inv.shape);  // Lines 1601-1603
      out_transpose.set_amax(out_columnwise_amax.data_ptr,
                             static_cast<DType>(out_columnwise_amax.dtype),
                             out_columnwise_amax.shape);  // Lines 1604-1606

      if (!eligible_for_rht_cast_fusion) {  // Line 1608
        // *** FALLBACK PATH: Separate RHT and quantization ***

        // Allocate temporary buffer for RHT output
        at::Tensor rht_output_t;
        TensorWrapper rht_output_t_cpp;
        rht_output_t = allocateTorchTensor(static_cast<int>(cols),
                                          static_cast<int>(rows), input.dtype());  // Line 1616-1617
        rht_output_t_cpp.set_rowwise_data(rht_output_t.data_ptr(), input.dtype(),
                                          std::vector<size_t>{cols, rows});  // Lines 1620-1621

        NVTE_SCOPED_GIL_RELEASE({
          // Perform RHT on transposed input
          nvte_hadamard_transform(input.data(), rht_output_t_cpp.data(), 0,
                                  this->rht_matrix_random_sign_mask_t, stream);  // Line 1625-1626
        });

        // Quantize the RHT output
        NVTE_SCOPED_GIL_RELEASE({
          nvte_quantize_v2(rht_output_t_cpp.data(), out_transpose.data(),
                          quant_config, stream);  // Line 1632
        });

      } else {  // Line 1634
        // *** OPTIMIZED PATH: Fused RHT + cast kernel ***

        NVTE_CHECK(this->rht_matrix.defined() && this->rht_matrix.numel() > 0,
                   "RHT matrix is not set");  // Line 1636-1637
        auto rht_matrix_nvte = makeTransformerEngineTensor(this->rht_matrix);  // Line 1638

        NVTE_SCOPED_GIL_RELEASE({
          nvte_hadamard_transform_cast_fusion_columnwise(
              input.data(), out_transpose.data(), rht_matrix_nvte.data(),
              quant_config, stream);  // Lines 1640-1641
        });
      }
    }

  } else {  // Line 1645
    // No RHT: Direct quantization for both rowwise and columnwise
    NVTE_SCOPED_GIL_RELEASE({
      nvte_quantize_v2(input.data(), out.data(), quant_config, stream);  // Line 1646
    });
  }
}
```

**Key Decision Points:**
1. **RHT Eligibility** (Line 1485-1486): Fused kernel requires `bfloat16`, `rows % 64 == 0`, `cols % 128 == 0`
2. **Amax Mode** (Line 1493): `with_post_rht_amax=True` computes amax after RHT
3. **Quantization Path**:
   - Rowwise (no RHT): Direct `nvte_quantize_v2()` on input
   - Columnwise (with RHT):
     - **Optimized**: `nvte_hadamard_transform_cast_fusion_columnwise()` (fused)
     - **Fallback**: `nvte_hadamard_transform()` → `nvte_quantize_v2()` (separate)

---

## Layer 4: C API Dispatcher

### File: [transformer_engine/common/cast/cast.cu:41-48](../transformer_engine/common/cast/cast.cu#L41-L48)

```cpp
void nvte_quantize_v2(const NVTETensor input, NVTETensor output,
                      const NVTEQuantizationConfig quant_config, cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize_v2);
  using namespace transformer_engine;

  constexpr bool IS_ACT = false;
  dispatch::quantize_fwd_helper<IS_ACT, Empty, nullptr>(input, output, quant_config, stream);  // Line 47
}
```

### File: [transformer_engine/common/cast/dispatch/quantize.cuh:28-172](../transformer_engine/common/cast/dispatch/quantize.cuh#L28-L172)

```cpp
template <bool IS_ACT, typename ParamOP, float (*OP)(float, const ParamOP &)>
void quantize_fwd_helper(const NVTETensor input, NVTETensor output,
                         const NVTEQuantizationConfig quant_config, cudaStream_t stream) {
  // Convert to C++ tensor objects
  const Tensor *input_tensor = convertNVTETensorCheck(input);  // Line 32
  Tensor *output_tensor = convertNVTETensorCheck(output);  // Line 33

  // Parse quantization config
  QuantizationConfig quant_config_cpp;
  if (quant_config != nullptr) {
    quant_config_cpp = *reinterpret_cast<QuantizationConfig *>(quant_config);
  }  // Lines 36-39

  // ... noop flag handling ...

  // Dispatch based on scaling mode
  switch (output_tensor->scaling_mode) {  // Line 58

    case NVTE_DELAYED_TENSOR_SCALING: {
      // FP8 delayed scaling (not NVFP4)
      // ... (Lines 59-78)
      break;
    }

    case NVTE_MXFP8_1D_SCALING: {
      // MXFP8 (not NVFP4)
      // ... (Lines 80-87)
      break;
    }

    case NVTE_NVFP4_1D_SCALING: {  // Line 89
      // *** NVFP4 PATH ***

      NVTE_CHECK(!IS_ACT, "IS_ACT is not supported by FWD NVTE_NVFP4_1D_SCALING");  // Line 90

      // Validate tensors
      CheckNoopTensor(*noop_tensor, "cast_noop");
      CheckInputTensor(*input_tensor, "input");
      CheckOutputTensor(*output_tensor, "output", false);  // Lines 93-95

      // Check if optimized kernel can be used
      int32_t rows = input_tensor->flat_first_dim();
      int32_t cols = input_tensor->flat_last_dim();
      auto dtype = input_tensor->dtype();
      bool use_optimized_kernel = (dtype == DType::kBFloat16) &&
                                 (rows % 32 == 0) &&
                                 (cols % 32 == 0) &&
                                 output_tensor->has_data();  // Lines 98-102

      if (use_optimized_kernel) {  // Line 105
        // *** OPTIMIZED KERNEL PATH ***
        if (quant_config_cpp.nvfp4_2d_quantization) {  // Line 106
          nvfp4::quantize_transpose</*use_2d_quantization=*/true>(
              *input_tensor, noop_tensor, output_tensor, &quant_config_cpp, stream);  // Line 107-108
        } else {
          nvfp4::quantize_transpose</*use_2d_quantization=*/false>(
              *input_tensor, noop_tensor, output_tensor, &quant_config_cpp, stream);  // Line 110-111
        }
      } else {  // Line 113
        // *** FALLBACK KERNEL PATH ***
        auto &global_amax = (output_tensor->amax.dptr != nullptr)
                            ? output_tensor->amax
                            : output_tensor->columnwise_amax;  // Lines 114-115
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
            /*stream=*/stream);  // Lines 116-127
      }
      break;
    }

    case NVTE_BLOCK_SCALING_2D:
    case NVTE_BLOCK_SCALING_1D: {
      // FP8 block scaling (not NVFP4)
      // ... (Lines 131-167)
      break;
    }

    default:
      NVTE_ERROR("Not implemented scaling mode");  // Line 170
  }
}
```

**Key Decision Point:**
- **Optimized kernel** (Line 105): Requires `bfloat16`, `rows % 32 == 0`, `cols % 32 == 0`, and rowwise data
- Calls `nvfp4::quantize_transpose<>()` for optimized path

---

## Layer 5: CUDA Kernel Implementations

### 5A. Amax Computation with RHT

#### File: [transformer_engine/common/include/transformer_engine/hadamard_transform.h:46-47](../transformer_engine/common/include/transformer_engine/hadamard_transform.h#L46-L47)

```cpp
/*! \brief Perform the absolute maximum reduction on the input tensor with/without
 *         randomized hadamard transform. The rowwise result is the absolute maximum
 *         of the input tensor. The columnwise result is the absolute maximum of the
 *         input tensor transposed and applied randomized hadamard transformation.
 */
void nvte_hadamard_transform_amax(const NVTETensor input, NVTETensor output,
                                  int random_sign_mask,
                                  int random_sign_mask_t, cudaStream_t stream);
```

#### File: [transformer_engine/common/hadamard_transform/hadamard_transform.cu](../transformer_engine/common/hadamard_transform/hadamard_transform.cu#L26-L96)

**CUDA Kernel Details:**
```cpp
template <typename IType, int kHadamardDimension, int BUFF_DIM_Y, int BUFF_DIM_X,
          bool kReturnPreRhtAmax, bool kReturnIdentityAmax, bool kReturnTransposedAmax>
__device__ __forceinline__ void ComputeKernel(uint32_t b_frag_i[4], uint32_t b_frag_t[4],
                                              IType* in_sh_ptr,
                                              uint32_t& local_pre_rht_amax_reg,
                                              uint32_t& local_amax_reg,
                                              uint32_t& local_amax_t_reg) {
  uint32_t a_frag[4];  // Input matrix fragment (16x16 tile in bfloat16)
  uint32_t c_frag[4];  // Result fragment

  int warp_id = threadIdx.x / kThreadsPerWarp;
  int local_rank = (threadIdx.x % kThreadsPerWarp);

  // Swizzled load from shared memory
  int ld_row_idx = local_rank % kHadamardDimension;
  int ld_col_idx = local_rank / kHadamardDimension + warp_id * 2;
  int swizzle_idx = swizzle_128B_atom_32B(ld_row_idx, ld_col_idx);

  // --- Compute identity amax (rowwise) ---
  if (kReturnIdentityAmax) {
    // Load input matrix using ldmatrix instruction
    ldmatrix_x4_m8n8_shared_b16<false>(a_frag[0], a_frag[1], a_frag[2], a_frag[3],
                                       reinterpret_cast<uint4*>(in_sh_ptr) + swizzle_idx);

    // MMA instruction computes amax during multiplication
    mma_m16_n16_k16_b16_b16_b16_noacc<kReturnIdentityAmax>(
        a_frag[0], a_frag[1], a_frag[2], a_frag[3],
        b_frag_i[0], b_frag_i[1], b_frag_i[2], b_frag_i[3],
        c_frag[0], c_frag[1], c_frag[2], c_frag[3],
        temp_amax_reg);

    // Update local amax using PTX instruction
    asm volatile("max.xorsign.abs.bf16x2 %0, %1, %2;\n\t"
                 : "=r"(local_amax_reg)
                 : "r"(local_amax_reg), "r"(temp_amax_reg));
  }

  // --- Compute transposed RHT amax (columnwise) ---
  if (kReturnTransposedAmax) {
    // Transpose fragments in registers
    matrix_transpose_m8_n8_b16_inplace(a_frag[0]);
    matrix_transpose_m8_n8_b16_inplace(a_frag[1]);
    matrix_transpose_m8_n8_b16_inplace(a_frag[2]);
    matrix_transpose_m8_n8_b16_inplace(a_frag[3]);

    // MMA with transposed Hadamard matrix
    mma_m16_n16_k16_b16_b16_b16_noacc<kReturnTransposedAmax>(
        a_frag[0], a_frag[2], a_frag[1], a_frag[3],
        b_frag_t[0], b_frag_t[1], b_frag_t[2], b_frag_t[3],
        c_frag[0], c_frag[1], c_frag[2], c_frag[3],
        temp_amax_t_reg);

    // Update local amax
    asm volatile("max.xorsign.abs.bf16x2 %0, %1, %2;\n\t"
                 : "=r"(local_amax_t_reg)
                 : "r"(local_amax_t_reg), "r"(temp_amax_t_reg));
  }
}
```

**Reduction to Global Amax:**
```cpp
template <int kNumWarps, bool kReturnPreRhtAmax, bool kReturnIdentityAmax,
          bool kReturnTransposedAmax>
__device__ __forceinline__ void ReduceMax(const float pre_rht_amax,
                                          const float identity_amax,
                                          const float transpose_amax,
                                          float* staging_for_pre_rht,
                                          float* staging_for_identity,
                                          float* staging_for_transpose,
                                          float* output_pre_rht_amax_ptr,
                                          float* output_identity_amax_ptr,
                                          float* output_transpose_amax_ptr,
                                          const int warpid) {
  // Step 1: Intra-warp reduction using warp shuffle
  int local_rank = threadIdx.x % 32;
  float warp_identity_amax = kReturnIdentityAmax ? warp_reduce_max<32>(identity_amax) : 0.0f;
  float warp_transpose_amax = kReturnTransposedAmax ? warp_reduce_max<32>(transpose_amax) : 0.0f;

  // Step 2: Write warp results to shared memory
  if (threadIdx.x % 32 == 0) {
    if (kReturnIdentityAmax) {
      staging_for_identity[warpid] = warp_identity_amax;
    }
    if (kReturnTransposedAmax) {
      staging_for_transpose[warpid] = warp_transpose_amax;
    }
  }
  __syncthreads();

  // Step 3: Inter-warp reduction
  if (warpid == 0 && kReturnIdentityAmax) {
    float identity_accum = local_rank < kNumWarps ? staging_for_identity[local_rank] : 0.0f;
    identity_accum = warp_reduce_max<NextPowerOf2<kNumWarps>()>(identity_accum);
    if (local_rank == 0) {
      atomicMaxFloat(output_identity_amax_ptr, identity_accum);  // Write to global memory
    }
  }

  if (warpid == 1 && kReturnTransposedAmax) {
    float transpose_accum = local_rank < kNumWarps ? staging_for_transpose[local_rank] : 0.0f;
    transpose_accum = warp_reduce_max<NextPowerOf2<kNumWarps>()>(transpose_accum);
    if (local_rank == 0) {
      atomicMaxFloat(output_transpose_amax_ptr, transpose_accum);  // Write to global memory
    }
  }
}
```

**Key Features:**
1. **Tensor Core MMA**: Uses `mma.sync.aligned.m16n16k16.row.col.f32.bf16.bf16.f32` instruction
2. **Simultaneous Computation**: Computes both identity amax and transposed RHT amax in one pass
3. **PTX Intrinsics**: Uses `max.xorsign.abs.bf16x2` for efficient absolute max
4. **Two-Level Reduction**: Intra-warp (shuffle) → Inter-warp (shared memory) → Global (atomic)

---

### 5B. Fused RHT + Quantization

#### File: [transformer_engine/common/include/transformer_engine/hadamard_transform.h:59-62](../transformer_engine/common/include/transformer_engine/hadamard_transform.h#L59-L62)

```cpp
/*! \brief Perform the columnwise hadamard transform cast fusion.
 */
void nvte_hadamard_transform_cast_fusion_columnwise(const NVTETensor input,
                                                    NVTETensor output,
                                                    const NVTETensor hadamard_matrix,
                                                    const NVTEQuantizationConfig quant_config,
                                                    cudaStream_t stream);
```

**Implementation:** Located in [transformer_engine/common/hadamard_transform/hadamard_transform.cu](../transformer_engine/common/hadamard_transform/hadamard_transform.cu)

**Kernel Pipeline:**
```cpp
// Pseudo-code representation of the fused kernel
__global__ void rht_cast_fusion_kernel() {
  // Stage 1: Load input tile from global memory
  __shared__ bfloat16 smem_input[TILE_SIZE];
  load_input_tile(smem_input);

  // Stage 2: Load Hadamard matrix to registers
  uint32_t H_frag[4];  // 16x16 Hadamard tile
  load_hadamard_matrix(H_frag);

  // Stage 3: Perform RHT using Tensor Core MMA
  uint32_t input_frag[4], rht_output_frag[4];
  ldmatrix_x4_m8n8_shared_b16(input_frag, smem_input);
  mma_m16_n16_k16_bf16(input_frag, H_frag, rht_output_frag);  // RHT computation

  // Stage 4: Quantize to FP4 (inline)
  //   - Convert bfloat16 → FP4 E2M1
  //   - Apply scaling factor (from pre-computed amax)
  //   - Pack 2 FP4 values into 1 byte
  uint8_t fp4_data[2];
  quantize_bf16_to_fp4(rht_output_frag, global_amax, fp4_data);

  // Stage 5: Compute and store scale_inv (FP8 E4M3)
  float block_amax = compute_block_amax(rht_output_frag);
  float scale_inv = compute_scale_inv(block_amax, global_amax);
  uint8_t scale_inv_fp8 = convert_fp32_to_fp8_e4m3(scale_inv);

  // Stage 6: Write to global memory
  store_fp4_data(fp4_data);
  store_scale_inv(scale_inv_fp8);
}
```

**Optimization Benefits:**
1. **No Intermediate Buffer**: RHT output goes directly to FP4, no bfloat16 intermediate
2. **Register-Only Datapath**: Quantization happens in registers before writeback
3. **Coalesced Writes**: Both FP4 data and scales written efficiently
4. **Reduced Memory Bandwidth**: Eliminates one round-trip to global memory

---

### 5C. Standalone Quantization (Fallback)

#### File: [transformer_engine/common/include/transformer_engine/cast.h](../transformer_engine/common/include/transformer_engine/cast.h)

```cpp
void nvte_quantize_v2(const NVTETensor input, NVTETensor output,
                      const NVTEQuantizationConfig quant_config, cudaStream_t stream);
```

**Implementation:** Dispatches to `quantize_transpose_vector_blockwise_fp4()` in fallback path

**Kernel Implementation:** [transformer_engine/common/cast/nvfp4/quantize_nvfp4.cuh](../transformer_engine/common/cast/nvfp4/quantize_nvfp4.cuh)

**Kernel Pipeline:**
```cpp
__global__ void quantize_fp4_kernel(bfloat16* input, uint8_t* output,
                                   uint8_t* scale_inv, float global_amax) {
  // Stage 1: Load input block (16 elements = 1 quantization block)
  __shared__ bfloat16 smem_input[BLOCK_SIZE];
  load_input_block(smem_input);

  // Stage 2: Compute block amax
  float block_amax = 0.0f;
  #pragma unroll
  for (int i = 0; i < 16; i++) {
    block_amax = fmaxf(block_amax, fabsf(__bfloat162float(smem_input[i])));
  }
  block_amax = warp_reduce_max(block_amax);  // Reduce within warp

  // Stage 3: Compute scaling factors
  //   encode_scale = (FP8_E4M3_MAX * FP4_E2M1_MAX) / global_amax
  //   decode_scale_fp8 = (block_amax / FP4_E2M1_MAX) * encode_scale
  //   decode_scale_fp8 = clamp(decode_scale_fp8, -448, 448) → FP8 E4M3
  constexpr float FP4_MAX = 6.0f;
  constexpr float FP8_MAX = 448.0f;
  float encode_scale = (FP8_MAX * FP4_MAX) / global_amax;
  float decode_scale = (block_amax / FP4_MAX) * encode_scale;
  decode_scale = fminf(fmaxf(decode_scale, -FP8_MAX), FP8_MAX);
  uint8_t scale_inv_fp8 = convert_fp32_to_fp8_e4m3(decode_scale);

  // Stage 4: Quantize to FP4
  //   scaled_value = input[i] * (1.0 / decode_scale)
  //   clipped_value = clamp(scaled_value, -6.0, 6.0)
  //   fp4_value = lookup_fp4_e2m1(clipped_value)
  uint8_t fp4_values[2];
  #pragma unroll
  for (int i = 0; i < 2; i++) {
    bfloat16 val0 = smem_input[i * 2];
    bfloat16 val1 = smem_input[i * 2 + 1];
    float scaled0 = __bfloat162float(val0) / decode_scale;
    float scaled1 = __bfloat162float(val1) / decode_scale;
    scaled0 = fminf(fmaxf(scaled0, -FP4_MAX), FP4_MAX);
    scaled1 = fminf(fmaxf(scaled1, -FP4_MAX), FP4_MAX);
    uint8_t fp4_0 = quantize_to_fp4_e2m1(scaled0);
    uint8_t fp4_1 = quantize_to_fp4_e2m1(scaled1);
    fp4_values[i] = (fp4_1 << 4) | fp4_0;  // Pack 2 FP4 into 1 byte
  }

  // Stage 5: Write to global memory
  store_fp4_packed(output, fp4_values);
  store_scale_inv(scale_inv, scale_inv_fp8);
}
```

**FP4 E2M1 Encoding:**
```
Sign | Exponent (2 bits) | Mantissa (1 bit) | Value
-----|-------------------|------------------|--------
  0  |        00         |        0         |  0.0
  0  |        00         |        1         |  0.5
  0  |        01         |        0         |  1.0
  0  |        01         |        1         |  1.5
  0  |        10         |        0         |  2.0
  0  |        10         |        1         |  3.0
  0  |        11         |        0         |  4.0
  0  |        11         |        1         |  6.0
  1  |        xx         |        x         | -value
```

**Memory Layout:**
```
Output FP4 data:   [val1_fp4 (4 bits) | val0_fp4 (4 bits)] = 1 byte
Output scale_inv:  FP8 E4M3 (1 byte per block)
```

---

## Layer 6: Return Path to Python

### C++ → Python Boundary

After CUDA kernels complete:

1. **Synchronization**: CUDA stream synchronization is **not** explicit in the code
   - PyTorch manages synchronization automatically when accessing tensor data
   - CUDA kernels are launched on `at::cuda::getCurrentCUDAStream()`

2. **Tensor Wrapping**: At [transformer_engine/pytorch/csrc/extensions/cast.cpp:79](../transformer_engine/pytorch/csrc/extensions/cast.cpp#L79)
```cpp
return output_py;  // Return Python NVFP4Tensor object
```

3. **Back to Python**: Control returns to [transformer_engine/pytorch/tensor/nvfp4_tensor.py:177](../transformer_engine/pytorch/tensor/nvfp4_tensor.py#L177)
```python
tex.quantize(src, self, dst, noop_flag)  # Returns here
return dst  # dst is now populated with quantized data
```

### Data Structure in Memory

**NVFP4Tensor Structure:**
```python
class NVFP4Tensor:
    _rowwise_data: torch.Tensor       # FP4 data (uint8), shape: [M, N//2]
    _rowwise_scale_inv: torch.Tensor  # FP8 E4M3 scales (uint8), shape: [M_pad, (N//16)_pad]
    _columnwise_data: torch.Tensor    # FP4 data transposed (uint8), shape: [N, M//2]
    _columnwise_scale_inv: torch.Tensor  # FP8 E4M3 scales (uint8), shape: [N_pad, (M//16)_pad]
    _amax_rowwise: torch.Tensor       # Global amax (float32), shape: [1]
    _amax_columnwise: torch.Tensor    # Global amax (float32), shape: [1]
    _fp4_dtype: TE_DType              # kFloat4E2M1
    _quantizer: NVFP4Quantizer        # Reference to quantizer
```

**Padding Rules:**
- Data padding: `M % 16 == 0`, `N % 16 == 0`
- Scale padding: `M % 128 == 0`, `(N // 16) % 4 == 0` (for cuBLAS compatibility)

---

## Complete Call Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PYTHON LAYER                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ NVFP4Quantizer.update_quantized() [nvfp4_tensor.py:160]                   │
│   ↓                                                                         │
│ tex.quantize(src, self, dst, noop_flag) [nvfp4_tensor.py:177]             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                         C++ BINDING LAYER                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ transformer_engine::pytorch::quantize() [cast.cpp:34]                      │
│   ↓                                                                         │
│ auto quantizer_cpp = convert_quantizer(quantizer) [cast.cpp:37]           │
│   ↓                                                                         │
│ quantizer_cpp->quantize(input_cpp, output_cpp, noop_flag_cpp) [cast.cpp:76]│
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                       C++ QUANTIZER IMPLEMENTATION                          │
├─────────────────────────────────────────────────────────────────────────────┤
│ NVFP4Quantizer::quantize() [quantizer.cpp:1444]                           │
│   ↓                                                                         │
│ ┌─── with_rht=True ──────────────────────────────────────────────┐        │
│ │                                                                 │        │
│ │ STEP 1: Compute amax                                            │        │
│ │   nvte_hadamard_transform_amax(input, out, ...) [quantizer.cpp:1498]   │
│ │     ↓                                                           │        │
│ │     Writes to: out._amax_rowwise, out._amax_columnwise         │        │
│ │                                                                 │        │
│ │ STEP 2: Amax reduction (if distributed)                         │        │
│ │   AllReduce(amaxes) [quantizer.cpp:1550]                       │        │
│ │                                                                 │        │
│ │ STEP 3A: Rowwise quantization (no RHT)                          │        │
│ │   nvte_quantize_v2(input, out_rowwise, ...) [quantizer.cpp:1570]        │
│ │     ↓                                                           │        │
│ │     Writes to: out._rowwise_data, out._rowwise_scale_inv       │        │
│ │                                                                 │        │
│ │ STEP 3B: Columnwise quantization (with RHT)                     │        │
│ │   ┌── eligible_for_rht_cast_fusion=True ────────────────┐     │        │
│ │   │ nvte_hadamard_transform_cast_fusion_columnwise(...)  │     │        │
│ │   │                                   [quantizer.cpp:1640]│     │        │
│ │   │   ↓                                                   │     │        │
│ │   │   Writes to: out._columnwise_data,                   │     │        │
│ │   │              out._columnwise_scale_inv               │     │        │
│ │   └─────────────────────────────────────────────────────┘     │        │
│ │   └── eligible_for_rht_cast_fusion=False ───────────────┐     │        │
│ │       nvte_hadamard_transform(...) [quantizer.cpp:1625]  │     │        │
│ │         ↓                                                 │     │        │
│ │       nvte_quantize_v2(...) [quantizer.cpp:1632]         │     │        │
│ │         ↓                                                 │     │        │
│ │         Writes to: out._columnwise_data,                 │     │        │
│ │                    out._columnwise_scale_inv             │     │        │
│ │   ────────────────────────────────────────────────────────     │        │
│ └─────────────────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                          C API DISPATCHER                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ nvte_quantize_v2(input, output, ...) [cast.cu:41]                         │
│   ↓                                                                         │
│ dispatch::quantize_fwd_helper<>(...) [cast.cu:47]                         │
│   ↓                                                                         │
│ switch (output_tensor->scaling_mode) [quantize.cuh:58]                    │
│   case NVTE_NVFP4_1D_SCALING:                                              │
│     ┌── use_optimized_kernel=True ────────────────────────┐               │
│     │ nvfp4::quantize_transpose<>(...) [quantize.cuh:107] │               │
│     │   ↓                                                  │               │
│     │   Launches optimized CUDA kernel                    │               │
│     └──────────────────────────────────────────────────────               │
│     └── use_optimized_kernel=False ───────────────────────┐               │
│         quantize_transpose_vector_blockwise_fp4(...)      │               │
│                                        [quantize.cuh:116]  │               │
│           ↓                                                │               │
│           Launches fallback CUDA kernel                   │               │
│     ────────────────────────────────────────────────────────               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CUDA KERNELS                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ [AMAX KERNEL]                                                               │
│ nvte_hadamard_transform_amax() [hadamard_transform.cu]                     │
│   ↓                                                                         │
│   __global__ hadamard_amax_kernel<<<>>>()                                  │
│     ↓                                                                       │
│     For each 16x16 tile:                                                   │
│       1. Load tile to shared memory                                        │
│       2. ldmatrix instruction → load to registers                          │
│       3. mma.sync instruction → compute RHT + amax simultaneously          │
│       4. Warp shuffle reduction                                            │
│       5. atomicMaxFloat → write global amax                                │
│                                                                             │
│ [FUSED RHT+QUANTIZE KERNEL] (optimized path)                               │
│ nvte_hadamard_transform_cast_fusion_columnwise() [hadamard_transform.cu]   │
│   ↓                                                                         │
│   __global__ rht_cast_fusion_kernel<<<>>>()                                │
│     ↓                                                                       │
│     For each 16x16 tile:                                                   │
│       1. Load input tile (bfloat16)                                        │
│       2. Load Hadamard matrix to registers                                 │
│       3. mma.sync → compute RHT(input)                                     │
│       4. Quantize to FP4 in registers                                      │
│       5. Compute scale_inv (FP8 E4M3)                                      │
│       6. Pack 2 FP4 → 1 byte                                               │
│       7. Write FP4 data + scale_inv to global memory                       │
│                                                                             │
│ [QUANTIZE KERNEL] (fallback path / rowwise)                                │
│ nvte_quantize_v2() → quantize_fp4_kernel<<<>>>() [quantize_nvfp4.cuh]     │
│   ↓                                                                         │
│   __global__ quantize_fp4_kernel<<<>>>()                                   │
│     ↓                                                                       │
│     For each 16-element block:                                             │
│       1. Load input block (bfloat16)                                       │
│       2. Compute block amax                                                │
│       3. Compute scale_inv = f(block_amax, global_amax)                    │
│       4. Quantize: input * (1/scale_inv) → clamp → FP4 E2M1               │
│       5. Pack 2 FP4 → 1 byte                                               │
│       6. Write FP4 data + scale_inv to global memory                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
                          (CUDA stream completes)
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                          RETURN TO PYTHON                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ NVFP4Tensor dst is now populated:                                          │
│   dst._rowwise_data        : FP4 quantized data [M, N//2]                 │
│   dst._rowwise_scale_inv   : FP8 E4M3 scales [M_pad, (N//16)_pad]         │
│   dst._columnwise_data     : FP4 quantized data (RHT applied) [N, M//2]   │
│   dst._columnwise_scale_inv: FP8 E4M3 scales [N_pad, (M//16)_pad]         │
│   dst._amax_rowwise        : Global amax (float32) [1]                     │
│   dst._amax_columnwise     : Global amax after RHT (float32) [1]          │
│                                                                             │
│ Return dst to caller [nvfp4_tensor.py:179]                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Optimizations

### 1. Tensor Core Acceleration
- **Target**: NVIDIA Ampere/Hopper Tensor Cores
- **Operation**: `mma.sync.aligned.m16n16k16.row.col.f32.bf16.bf16.f32`
- **Benefits**:
  - 256 FLOPS per instruction
  - Simultaneous compute + amax tracking
  - Fused transpose in register layout

### 2. Fused RHT + Cast Kernel
- **Condition**: `dtype=bfloat16`, `rows % 64 == 0`, `cols % 128 == 0`
- **Benefits**:
  - Eliminates intermediate bfloat16 buffer (saves memory)
  - Reduces global memory traffic by ~50%
  - Keeps RHT output in registers until quantized

### 3. Two-Stage Scaling
- **Global Scale** (FP32): `encode_scale = (448.0 * 6.0) / global_amax`
  - Applied uniformly across entire tensor
  - Ensures all scales fit in FP8 E4M3 range
- **Block Scale** (FP8 E4M3): `decode_scale = (block_amax / 6.0) * encode_scale`
  - Per-block (16 elements) granularity
  - Stored in FP8 to minimize memory footprint
  - Actual decode: `value = fp4_data * decode_scale * (1.0 / encode_scale)`

### 4. Memory Layout Optimization
- **Scale Padding**: Scales padded to `128 × N` and `N × 4` for cuBLAS alignment
- **Coalesced Access**: FP4 data written in aligned 128-byte chunks
- **Swizzled Shared Memory**: Avoids bank conflicts using 128B atom swizzling

---

## Configuration Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `with_rht` | `True` | Enable Random Hadamard Transform |
| `with_post_rht_amax` | `True` | Compute amax after RHT (not before) |
| `with_random_sign_mask` | `True` | Use random signs in Hadamard matrix |
| `rht_matrix_random_sign_mask_t` | `0x7B66` | 16-bit sign mask: `[1,1,1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,1,-1,-1]` |
| `quant_tile_shape` | `(1, 16)` | Block size for quantization |
| `fp4_dtype` | `kFloat4E2M1` | FP4 format: 1 sign, 2 exp, 1 mantissa |
| `rowwise_usage` | `True` | Generate rowwise quantized data |
| `columnwise_usage` | `True` | Generate columnwise quantized data (with RHT) |

---

## Memory Footprint

For input tensor `[M, N]` in bfloat16:

| Component | Size | Formula |
|-----------|------|---------|
| Input | `2MN` bytes | bfloat16 |
| Rowwise FP4 data | `MN/2` bytes | 2 FP4 per byte |
| Rowwise scales | `(M×N/16)` bytes | FP8 E4M3, 1 per 16 elements |
| Rowwise amax | `4` bytes | FP32 scalar |
| Columnwise FP4 data | `MN/2` bytes | Transposed layout |
| Columnwise scales | `(N×M/16)` bytes | FP8 E4M3 |
| Columnwise amax | `4` bytes | FP32 scalar |
| **Total NVFP4Tensor** | **`~MN` bytes** | **~50% of bfloat16** |

---

## Performance Characteristics

### Computational Complexity
- **Amax computation**: `O(MN)` with Tensor Core acceleration
- **RHT**: `O(MN log H)` where `H=16` (Hadamard dimension)
- **Quantization**: `O(MN)` element-wise operations

### Memory Bandwidth
- **Without fusion**: 3 passes (amax read, RHT read/write, quantize read/write)
- **With fusion**: 2 passes (amax read, fused RHT+quantize read/write)
- **Bandwidth saving**: ~33% reduction in total memory traffic

### Kernel Occupancy
- **Amax kernel**: High occupancy (limited by shared memory)
- **Fused kernel**: Moderate occupancy (register pressure from MMA)
- **Quantize kernel**: High occupancy (minimal shared memory usage)

---

## Related Files

### Python Layer
- [transformer_engine/pytorch/tensor/nvfp4_tensor.py](../transformer_engine/pytorch/tensor/nvfp4_tensor.py) - NVFP4Tensor and NVFP4Quantizer
- [transformer_engine/pytorch/custom_recipes/quantization_nvfp4.py](../transformer_engine/pytorch/custom_recipes/quantization_nvfp4.py) - Reference implementation (pure Python)

### C++ Binding Layer
- [transformer_engine/pytorch/csrc/extensions/pybind.cpp](../transformer_engine/pytorch/csrc/extensions/pybind.cpp) - Python bindings
- [transformer_engine/pytorch/csrc/extensions/cast.cpp](../transformer_engine/pytorch/csrc/extensions/cast.cpp) - Quantize entry point
- [transformer_engine/pytorch/csrc/quantizer.cpp](../transformer_engine/pytorch/csrc/quantizer.cpp) - NVFP4Quantizer implementation
- [transformer_engine/pytorch/csrc/common.h](../transformer_engine/pytorch/csrc/common.h) - NVFP4Quantizer class definition

### C API & Dispatcher
- [transformer_engine/common/cast/cast.cu](../transformer_engine/common/cast/cast.cu) - C API entry point
- [transformer_engine/common/cast/dispatch/quantize.cuh](../transformer_engine/common/cast/dispatch/quantize.cuh) - Quantization dispatcher
- [transformer_engine/common/include/transformer_engine/cast.h](../transformer_engine/common/include/transformer_engine/cast.h) - C API headers
- [transformer_engine/common/include/transformer_engine/hadamard_transform.h](../transformer_engine/common/include/transformer_engine/hadamard_transform.h) - RHT C API headers

### CUDA Kernels
- [transformer_engine/common/hadamard_transform/hadamard_transform.cu](../transformer_engine/common/hadamard_transform/hadamard_transform.cu) - RHT and amax kernels
- [transformer_engine/common/cast/nvfp4/quantize_nvfp4.cuh](../transformer_engine/common/cast/nvfp4/quantize_nvfp4.cuh) - NVFP4 quantization kernels
- [transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh](../transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh) - Optimized transpose+quantize kernels
- [transformer_engine/common/cast/nvfp4/core_nvfp4.cuh](../transformer_engine/common/cast/nvfp4/core_nvfp4.cuh) - Core FP4 utilities

---

## Appendix: FP4 E2M1 Format

### Encoding Table

| Decimal | Binary (S-EE-M) | Hex |
|---------|-----------------|-----|
| 0.0     | 0-00-0          | 0x0 |
| 0.5     | 0-00-1          | 0x1 |
| 1.0     | 0-01-0          | 0x2 |
| 1.5     | 0-01-1          | 0x3 |
| 2.0     | 0-10-0          | 0x4 |
| 3.0     | 0-10-1          | 0x5 |
| 4.0     | 0-11-0          | 0x6 |
| 6.0     | 0-11-1          | 0x7 |
| -0.0    | 1-00-0          | 0x8 |
| -0.5    | 1-00-1          | 0x9 |
| -1.0    | 1-01-0          | 0xA |
| -1.5    | 1-01-1          | 0xB |
| -2.0    | 1-10-0          | 0xC |
| -3.0    | 1-10-1          | 0xD |
| -4.0    | 1-11-0          | 0xE |
| -6.0    | 1-11-1          | 0xF |

### Packing Format
```
Byte layout: [value1 (4 bits) | value0 (4 bits)]
Example:
  value0 = 1.5 (0x3)
  value1 = -2.0 (0xC)
  Packed byte = 0xC3
```

---

## Appendix: FP8 E4M3 Scale Format

### Properties
- **Range**: [-448, 448]
- **Exponent bits**: 4 (biased by 7)
- **Mantissa bits**: 3 (explicit bit)
- **Special values**: NaN supported, no infinities

### Usage in NVFP4
```python
# Encoding:
encode_scale = (448.0 * 6.0) / global_amax
decode_scale = (block_amax / 6.0) * encode_scale
decode_scale_fp8 = convert_to_fp8_e4m3(clamp(decode_scale, -448, 448))

# Decoding:
dequant_value = fp4_value * decode_scale_fp8 * (global_amax / (448.0 * 6.0))
```

---

*Document generated from TransformerEngine codebase analysis*
*Date: 2025-11-26*
*TransformerEngine Commit: Latest*
