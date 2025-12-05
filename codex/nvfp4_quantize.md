# NVFP4 Quantize + Transpose (1D scaling, RETURN_TRANSPOSE=True)

Line‑by‑line walkthrough of `quantize_transpose` and the 1D kernel
`quantize_transpose_nvfp4_kernel` for a 1024×768 bf16 input. Both SR OFF/ON
behavior is called out explicitly. The order below mirrors source order in
`transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh` and does
not skip lines.

---
## 1. Launcher: `quantize_transpose` (1D path)
File: `transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh`

```cuda
template <bool use_2d_quantization>
void quantize_transpose(const Tensor &input, const Tensor *noop, Tensor *output,
                        const QuantizationConfig *quant_config, cudaStream_t stream) {
#if FP4_TYPE_SUPPORTED
  using namespace quantize_transpose_kernel;
  using namespace ptx;
  bool use_stochastic_rounding = quant_config ? quant_config->stochastic_rounding : false;

  bool return_transpose = output->has_columnwise_data();

  constexpr bool COMPUTE_ACTIVATIONS = false;
  using ParamOP = Empty;
  constexpr float (*OP)(float, const ParamOP &) = nullptr;
```
- Template gate picks 1D vs 2D; we enter 1D because `use_2d_quantization=false` from caller.
- Namespace aliases shorten later symbol use.
- SR flag pulled from `quant_config`; default false when config is null.
- `return_transpose` becomes true because the caller passes columnwise buffers; this triggers transpose path in kernel.
- Activations disabled (`COMPUTE_ACTIVATIONS=false`) so no fused nonlinear op.

```cuda
  checkCuDriverContext(stream);
  CheckNoopTensor(*noop, "cast_noop");
  CheckInputTensor(input, "input");
  CheckOutputTensor(*output, "output", false);

  NVTE_CHECK(input.has_data(), ...);
  NVTE_CHECK(output->has_data(), ...);
  NVTE_CHECK(is_fp4_dtype(output->data.dtype), ...);
  NVTE_CHECK(output->scale_inv.dptr != nullptr, ...);
  if (return_transpose) { ... }
```
- Runtime guards ensure driver context and tensor metadata are valid; these can early exit via NVTE error.
- Enforces output type is FP4 and scale buffers exist; when transpose requested, also checks transposed buffers.

```cuda
  const size_t rows = input.flat_first_dim();
  const size_t cols = input.flat_last_dim();

  NVTE_CHECK(rows % 32 == 0, ...);
  NVTE_CHECK(cols % 32 == 0, ...);

  const size_t blocks_Y = DIVUP(rows, CHUNK_DIM_Y);
  const size_t blocks_X = DIVUP(cols, CHUNK_DIM_X);
  const dim3 grid(blocks_X, blocks_Y);
  const size_t block_size = THREADS_NUM;
```
- Flattens input to 2D [rows, cols]; our shape 1024×768 satisfies 32‑alignment.
- `CHUNK_DIM_Y=128`, `CHUNK_DIM_X=128` ⇒ `blocks_Y=8`, `blocks_X=6` → grid (6,8); block size fixed at 128 threads.

```cuda
  const size_t scale_stride = output->scale_inv.shape[1];
  const size_t scale_stride_transpose = return_transpose ? output->columnwise_scale_inv.shape[1] : 0;

  nvfp4_scale_t *const scales_ptr = reinterpret_cast<nvfp4_scale_t *>(output->scale_inv.dptr);
  nvfp4_scale_t *const scales_transpose_ptr =
      reinterpret_cast<nvfp4_scale_t *>(output->columnwise_scale_inv.dptr);

  const float *noop_ptr = reinterpret_cast<const float *>(noop->data.dptr);
  const float *const amax_rowwise_ptr = reinterpret_cast<const float *>(output->amax.dptr);
  const float *const amax_colwise_ptr =
      reinterpret_cast<const float *>(output->columnwise_amax.dptr);
```
- Derives scale strides and pointers for rowwise and transpose outputs; these feed kernel scale writes.
- `noop_ptr` lets kernel early‑out if noop tensor requests; amax pointers carry running max stats.

```cuda
  const NVTETensor rng_state_tensor = (quant_config != nullptr) ? quant_config->rng_state : nullptr;
  const size_t *rng_state = nullptr;
  if (rng_state_tensor != nullptr) { ... rng_state = reinterpret_cast<const size_t *>(rng_state_te_tensor.data.dptr); }

  using IType = bf16;

  alignas(64) CUtensorMap tensor_map_input{};
  alignas(64) CUtensorMap tensor_map_output{};
  alignas(64) CUtensorMap tensor_map_output_transpose{};
```
- If SR enabled, kernel consumes a 2×uint64 RNG state; otherwise `rng_state=null` and SR logic becomes deterministic.
- Input element type fixed to bf16 for this build; tensor maps are aligned for TMA.

```cuda
  create_2D_tensor_map(tensor_map_input, input.data, rows, cols, BUFF_DIM_Y, BUFF_DIM_X, cols, 0,
                       sizeof(IType) * 8);

  create_2D_tensor_map(tensor_map_output, output->data, rows, cols, BUFF_DIM_Y, BUFF_DIM_X, cols, 0,
                       4);
  if (return_transpose) {
    create_2D_tensor_map(tensor_map_output_transpose, output->columnwise_data, cols, rows,
                         BUFF_DIM_X, BUFF_DIM_Y, rows, 0, 4);
  }
```
- Sets up TMA maps: input bf16 stride `cols`, output FP4 stride `cols`, transpose map swaps dims and tile shapes (128×32) for columnwise path.

```cuda
  constexpr size_t buff_elems = BUFF_DIM_Y * BUFF_DIM_X;
  constexpr size_t buff_elems_total = BUFFS_NUM * buff_elems;
  constexpr size_t buff_size_aligned_in = DIVUP_TO_MULTIPLE(buff_elems_total * sizeof(IType), TMA_SHMEM_ALIGNMENT);
  constexpr size_t buff_size_aligned_out = DIVUP_TO_MULTIPLE((buff_elems_total * 4) / 8, TMA_SHMEM_ALIGNMENT);
  constexpr size_t buff_size_scales = (CHUNK_DIM_Y * CHUNK_DIM_X) / 16 * sizeof(nvfp4_scale_t);
```
- Precomputes dynamic shared memory footprint: two input buffers, two output buffers, plus scales for transpose path; sizes are TMA‑alignment padded (128B).

```cuda
  constexpr size_t in_mem = buff_size_aligned_in;
  constexpr size_t out_data_mem = buff_size_aligned_out;
  constexpr size_t out_data_transpose_mem = buff_size_aligned_out;
  constexpr size_t out_scales_transpose_mem = buff_size_scales;

  constexpr size_t out_mem = out_data_mem + out_data_transpose_mem;

  constexpr size_t dshmem_size = in_mem + out_mem + out_scales_transpose_mem + TMA_SHMEM_ALIGNMENT;
```
- Computes final dynamic shared size (input + both outputs + scales + alignment padding) to pass to launch attributes.

```cuda
  TRANSFORMER_ENGINE_SWITCH_CONDITION(
      use_stochastic_rounding, USE_STOCHASTIC_ROUNDING,

      TRANSFORMER_ENGINE_SWITCH_CONDITION(return_transpose, RETURN_TRANSPOSE, {
        auto kernel = quantize_transpose_nvfp4_kernel<COMPUTE_ACTIVATIONS, ParamOP, OP, IType,
                                                      USE_STOCHASTIC_ROUNDING, RETURN_TRANSPOSE>;

        if constexpr (use_2d_quantization) {
          kernel = quantize_transpose_nvfp4_2D_kernel<...>;
        }

        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem_size);
        kernel<<<grid, block_size, dshmem_size, stream>>>(...);
      }););
#else
  NVTE_ERROR("FP4 support requires CUDA 12.8+, but compile-time CUDA version is ", CUDA_VERSION);
#endif
}
```
- Two macro switches instantiate template booleans for SR and transpose. Because `use_2d_quantization=false`, kernel remains the 1D path.
- Sets max dynamic shared size then launches with grid (6,8), block 128, and passes tensor maps, scales, amax, dimensions, strides, and RNG state.
- Compile‑time guard emits error on older CUDA.

---
## 2. Kernel signature and constants (1D)
`quantize_transpose_kernel::quantize_transpose_nvfp4_kernel`

```cuda
template <bool COMPUTE_ACTIVATIONS, typename ParamOP, float (*OP)(float, const ParamOP &),
          typename IType, bool USE_STOCHASTIC_ROUNDING, bool RETURN_TRANSPOSE>
__global__ void __launch_bounds__(THREADS_NUM)
    quantize_transpose_nvfp4_kernel(const __grid_constant__ CUtensorMap tensor_map_input,
                                    const __grid_constant__ CUtensorMap tensor_map_output,
                                    const __grid_constant__ CUtensorMap tensor_map_output_t,
                                    nvfp4_scale_t *const scales_ptr,
                                    nvfp4_scale_t *const scales_t_ptr, const float *noop,
                                    const float *const amax_rowwise_ptr,
                                    const float *const amax_colwise_ptr, const size_t rows,
                                    const size_t cols, const size_t scale_stride,
                                    const size_t scale_stride_t, const size_t *rng_state) {
```
- Templates control activation fusion, input type (bf16 here), SR, and whether transpose is emitted.
- `__launch_bounds__(128)` matches block size.
- Tensor maps are `__grid_constant__` (reside in global constant space for TMA). Scale pointers and RNG passed as plain pointers.

```cuda
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  constexpr bool NO_ACTIVATIONS_NOT_FP32_INPUT = (!COMPUTE_ACTIVATIONS) && (!std::is_same_v<IType, float>);

  using IType2 = typename ptx::FPx2<IType>;
```
- Guard requires Hopper+ (SM90) because the kernel uses TMA and FP4. `NO_ACTIVATIONS_NOT_FP32_INPUT` picks cheaper bf16 math path. `IType2` is a 2‑lane vector helper.

```cuda
  if constexpr (!COMPUTE_ACTIVATIONS) {
    if (noop != nullptr && noop[0] == 1.0f) {
      return;
    }
  }
```
- Early exit when caller set the `noop` flag (fast path to skip compute).

```cuda
  const size_t rng_sequence =
      threadIdx.x + blockIdx.x * THREADS_NUM + blockIdx.y * gridDim.x * THREADS_NUM;
  const size_t rng_seed = rng_state != nullptr ? rng_state[0] : 0;
  const size_t rng_offset = rng_state != nullptr ? rng_state[1] : 0;
  transformer_engine::curanddx::detail::philox4x32_native_state<10> rng;
  rng.init(rng_seed, rng_sequence, rng_offset);
  uint4 random_uint4 = USE_STOCHASTIC_ROUNDING ? rng.generate4() : uint4{0, 0, 0, 0};
  int rnd_idx = 0;
```
- Per‑thread Philox stream (unique sequence per CTA lane). If SR off, `random_uint4` is zeroed and downstream conversions become deterministic. `rnd_idx` cycles through 4 lanes inside `uint4` to amortize RNG calls.

```cuda
  constexpr bool IS_CACHED_ACT_OP = COMPUTE_ACTIVATIONS;

  const size_t block_offset_Y = blockIdx.y * CHUNK_DIM_Y;
  const size_t block_offset_X = blockIdx.x * CHUNK_DIM_X;

  const size_t block_offset_Y_t = blockIdx.x * CHUNK_DIM_X;
  const size_t block_offset_X_t = blockIdx.y * CHUNK_DIM_Y;

  const size_t chunk_rows = rows - block_offset_Y;
```
- Activation caching macro simplifies later branches (false here). Block offsets locate the 128×128 chunk in input; transpose offsets swap X/Y for transposed output. `chunk_rows` is remaining rows to handle tail CTAs; for 1024×768 it is always 128.

```cuda
  const size_t scales_block_offset_Y_rowwise = blockIdx.y * CHUNK_DIM_Y;
  const size_t scales_block_offset_X_rowwise = blockIdx.x * SCALES_PER_CHUNK_X;
  const size_t scales_block_offset_Y_t = blockIdx.x * CHUNK_DIM_X;
  const size_t scales_block_offset_X_t = blockIdx.y * SCALES_PER_CHUNK_Y;
```
- Precompute starting indices for writing scales: rowwise scales shaped (rows, cols/16); transpose scales shaped (cols, rows/16).

```cuda
  const size_t tid_Y_rowwise = threadIdx.x / THREADS_X_ROWWISE;   // 128 / 8 = 16 rows
  const size_t tid_X_rowwise = threadIdx.x % THREADS_X_ROWWISE;   // 0..7 columns
  const size_t tid_X_colwise = threadIdx.x;                       // full 0..127
  const size_t tid_Y_t = tid_X_colwise;                           // reused for transpose scale index
```
- Logical thread coordinates: rowwise view is 16×8 grid; columnwise uses linear thread id along X.

```cuda
  const size_t thread_offset_Y_rowwise = tid_Y_rowwise;
  const size_t thread_offset_X_rowwise = tid_X_rowwise * SCALE_DIM;  // 16‑wide slice
  const size_t thread_offset_X_colwise = tid_X_colwise;               // 1 column per thread

  const size_t row_base_rowwise = block_offset_Y + thread_offset_Y_rowwise;
  const size_t row_base_colwise = block_offset_Y;
  const size_t col_base_colwise = block_offset_X + thread_offset_X_colwise;

  const bool col_out_of_bounds_colwise = (col_base_colwise >= cols);
```
- Offsets into current chunk; rowwise threads start at their row within the chunk; columnwise threads track absolute column for bounds.

```cuda
  const size_t scales_offset_Y_rowwise = scales_block_offset_Y_rowwise + tid_Y_rowwise;
  const size_t scales_offset_X_rowwise = scales_block_offset_X_rowwise + tid_X_rowwise;
  const size_t scales_offset_Y_t = scales_block_offset_Y_t + tid_Y_t;
  const size_t scales_offset_X_t = scales_block_offset_X_t;

  const size_t SFs_per_row = cols / SCALE_DIM;

  const bool rowwise_scale_is_within_bounds_X = scales_offset_X_rowwise < SFs_per_row;
  const bool colwise_scale_is_within_bounds_Y = scales_offset_Y_t < cols;
```
- Computes global indices for scale tensors and precomputes X/Y bounds; for our shape all booleans true.

```cuda
  const int thread_lane = threadIdx.x % THREADS_PER_WARP;
  const int bank_group = thread_lane / THREADS_PER_BANK;

  constexpr size_t buff_elems = BUFF_DIM_Y * BUFF_IN_DIM_X;
  constexpr size_t buff_elems_total = BUFFS_NUM * buff_elems;

  constexpr size_t buff_size_aligned_in =
      DIVUP_TO_MULTIPLE(buff_elems_total * sizeof(IType), TMA_SHMEM_ALIGNMENT);
  constexpr size_t buff_size_aligned_out =
      DIVUP_TO_MULTIPLE((buff_elems_total * 4) / 8, TMA_SHMEM_ALIGNMENT);

  constexpr size_t in_mem = buff_size_aligned_in;

  constexpr size_t out_mem_rowwise_data = buff_size_aligned_out;
  constexpr size_t out_mem_colwise_data = buff_size_aligned_out;
  constexpr size_t out_mem_rowwise_scales = 0;
```
- `thread_lane`/`bank_group` used for swizzling to avoid SMEM bank conflicts. Recomputes buffer sizes locally for alignment.

```cuda
  extern __shared__ char dynamic_shmem[];
  uintptr_t base_shmem_ptr = reinterpret_cast<uintptr_t>(dynamic_shmem);
  uintptr_t dshmem = (base_shmem_ptr + TMA_SHMEM_ALIGNMENT - 1) &
                     ~(static_cast<uintptr_t>(TMA_SHMEM_ALIGNMENT - 1));

  IType *in_sh = reinterpret_cast<IType *>(dshmem);
  fp4e2m1x2 *out_data_sh = reinterpret_cast<fp4e2m1x2 *>(dshmem + in_mem);
  fp4e2m1x2 *out_t_data_sh = reinterpret_cast<fp4e2m1x2 *>(dshmem + in_mem + out_mem_rowwise_data);

  nvfp4_scale_t *out_rowwise_scales_sh = reinterpret_cast<nvfp4_scale_t *>(
      dshmem + in_mem + out_mem_rowwise_data + out_mem_colwise_data);
  nvfp4_scale_t *out_colwise_scales_sh = reinterpret_cast<nvfp4_scale_t *>(
      dshmem + in_mem + out_mem_rowwise_data + out_mem_colwise_data + out_mem_rowwise_scales);
  IType *cached_act_sh = in_sh;  // in_sh is used as a cache buffer

  constexpr size_t shmem_buff_size = buff_size_aligned_in / BUFFS_NUM;
```
- Aligns dynamic SMEM to 128B, partitions it: input buffers, rowwise fp4, transpose fp4, then scales. Reuses input buffer as activation cache when needed (not used here). `shmem_buff_size` is the per‑buffer TMA copy size.

```cuda
  const bool is_master_thread = (threadIdx.x == 0);

  const float S_enc_rowwise = (amax_rowwise_ptr == nullptr)
                                  ? 1.0f
                                  : compute_global_encode_scaling_factor_FP4(*amax_rowwise_ptr);
  const float S_dec_rowwise = 1.0 / S_enc_rowwise;

  const float S_enc_colwise = (amax_colwise_ptr == nullptr)
                                  ? S_enc_rowwise
                                  : compute_global_encode_scaling_factor_FP4(*amax_colwise_ptr);
  const float S_dec_colwise = 1.0 / S_enc_colwise;

  float thread_amax = 0.0f;
```
- `is_master_thread` gates TMA launches. Global encode/decoding factors derived from running amax (fallback 1). Separate rowwise/colwise to allow independent scaling. Initializes per‑thread max accumulator.

```cuda
  __shared__ alignas(8) uint64_t mbar[STAGES];

  initialize_barriers<STAGES, THREADS_NUM>(mbar, is_master_thread);

  copy_2d_to_shared(&in_sh[0], &tensor_map_input, block_offset_X, block_offset_Y, shmem_buff_size,
                    &mbar[0], is_master_thread);
```
- Allocates one mbarrier per stage (STAGES=4). Master thread arms them then kicks off first TMA read of the chunk’s first 32×128 tile into buffer 0.

---
## 3. Pipeline loop over stages
Each stage handles a 32×128 tile; there are 4 per CTA in Y.

```cuda
#pragma unroll
  for (size_t stage = 0; stage < STAGES; ++stage) {
    const size_t buff = stage % BUFFS_NUM;
    const size_t next_stage = stage + 1;
    const size_t stage_offset_Y = stage * BUFF_DIM_Y;

    const size_t buff_offset_in = buff * BUFF_IN_SIZE;
    const size_t buff_offset_out = buff * BUFF_OUT_SIZE;
    const size_t buff_offset_out_t = buff * BUFF_OUT_T_SIZE;

    if (next_stage < STAGES) {
      ptx::cp_async_bulk_wait_group_read<1>();

      const size_t next_buff = next_stage % BUFFS_NUM;
      const size_t next_stage_offset_Y = next_stage * BUFF_DIM_Y;
      const size_t global_offset_Y = block_offset_Y + next_stage_offset_Y;
      const size_t global_offset_X = block_offset_X;
      const size_t next_buff_offset = next_buff * BUFF_IN_SIZE;

      copy_2d_to_shared(&in_sh[next_buff_offset], &tensor_map_input, global_offset_X,
                        global_offset_Y, shmem_buff_size, &mbar[next_stage], is_master_thread);
    }

    ptx::fence_proxy_async_shared_cta();
    ptx::mbarrier_wait_parity(&mbar[stage], 0);

    float block_amax = 0.0f;
```
- Loop over 4 stages. Double‑buffer alternates `buff` 0/1. If a next stage exists, master waits for prior TMA read slot availability then issues the next async copy into the alternate buffer. `fence_proxy_async_shared_cta` enforces ordering before barrier wait; `mbarrier_wait_parity` blocks until data arrives. `block_amax` resets per stage.

### 3A. Columnwise + Transpose (only when RETURN_TRANSPOSE)
```cuda
    if constexpr (RETURN_TRANSPOSE) {
#pragma unroll
      for (size_t it = 0; it < ITERATIONS_TRANSPOSE; ++it) {
        const size_t in_thread_offset_Y = 0 + it * SCALE_DIM;
        const size_t in_thread_offset_X = thread_offset_X_colwise;

        const size_t out_t_thread_offset_Y = thread_offset_X_colwise;
        const size_t out_t_thread_offset_X = 0 + it * BUFF_OUT_IT_OFFSET;

        const size_t shmem_offset_base_colwise_in =
            buff_offset_in + in_thread_offset_Y * BUFF_IN_DIM_X + in_thread_offset_X;
        const size_t shmem_offset_base_colwise_out_t =
            buff_offset_out_t + out_t_thread_offset_Y * BUFF_OUT_T_DIM_X + out_t_thread_offset_X;

        block_amax = 0.0f;
        float in_compute_colwise[SCALE_DIM];
        IType in_colwise_IType[SCALE_DIM];
```
- Four iterations (`ITERATIONS_TRANSPOSE=4`) step through 32 rows in 16‑row bands. Each thread fixes a column, adjusts Y offset, and prepares local arrays for compute and bf16 raw values.

```cuda
        if constexpr (NO_ACTIVATIONS_NOT_FP32_INPUT) {
          IType block_amax_f16 = static_cast<IType>(0.0f);
#pragma unroll
          for (int i = 0; i < SCALE_DIM; ++i) {
            const int shmem_offset_colwise = shmem_offset_base_colwise_in + i * BUFF_IN_DIM_X;
            in_colwise_IType[i] = in_sh[shmem_offset_colwise];
            block_amax_f16 = __hmax(block_amax_f16, __habs(in_colwise_IType[i]));
          }
          block_amax = static_cast<float>(block_amax_f16);
        } else {
#pragma unroll
          for (int i = 0; i < SCALE_DIM; ++i) {
            const int shmem_offset_colwise = shmem_offset_base_colwise_in + i * BUFF_IN_DIM_X;
            float elt = static_cast<float>(in_sh[shmem_offset_colwise]);
            if constexpr (COMPUTE_ACTIVATIONS) { elt = OP(elt, {}); }
            if constexpr (!std::is_same_v<IType, float>) { elt = static_cast<float>(static_cast<IType>(elt)); }
            if constexpr (IS_CACHED_ACT_OP) { cached_act_sh[shmem_offset_colwise] = static_cast<IType>(elt); }
            const bool row_out_of_bounds_colwise = (row_base_colwise + stage_offset_Y + i >= rows);
            const bool out_of_bounds = (col_out_of_bounds_colwise || row_out_of_bounds_colwise);
            if (!out_of_bounds) { block_amax = fmaxf(block_amax, fabsf(elt)); }
            in_compute_colwise[i] = elt;
          }
        }
```
- Two paths: bf16 fast path (used here) keeps data bf16 and accumulates half max; FP32 path would run activation and cache. Bound checks protect tails (inactive for 1024×768). `block_amax` is max over 16 elements.

```cuda
        const nvfp4_scale_t S_dec_b_fp8 =
            compute_decoding_scaling_factor(block_amax, S_enc_colwise);

        const size_t scale_idx_sh = tid_Y_t * SCALES_PER_CHUNK_Y + stage * ITERATIONS_TRANSPOSE + it;
        out_colwise_scales_sh[scale_idx_sh] = S_dec_b_fp8;

        constexpr float float_max = detail::TypeExtrema<float>::max;
        const float block_scale_inverse = fminf(
            1.0f / (static_cast<float>(S_dec_b_fp8) * S_dec_colwise), float_max);
        const float2 block_scale_inverse_2x{block_scale_inverse, block_scale_inverse};
```
- Derives per‑block decode scale using block amax and global encode. Stores into shared scale buffer (later vector‑stored). Computes reciprocal encode factor, clamped to float max, and duplicates into float2 for vector PTX ops.

```cuda
        fp4e2m1x4 regs[SCALE_DIM / 4];

#pragma unroll
        for (int e = 0; e < SCALE_DIM / 4; ++e) {
          const uint32_t rbits = get_rbits(rng, random_uint4, rnd_idx);
          if constexpr (NO_ACTIVATIONS_NOT_FP32_INPUT) {
            const uint64_t elts = *reinterpret_cast<uint64_t *>(&in_colwise_IType[4 * e]);
            regs[e] = ptx::mul_cvt_bf16_to_fp4_4x<USE_STOCHASTIC_ROUNDING>(
                elts, block_scale_inverse_2x, rbits);
          } else {
            const float2 in01 = *reinterpret_cast<float2 *>(&in_compute_colwise[4 * e]);
            const float2 in23 = *reinterpret_cast<float2 *>(&in_compute_colwise[4 * e + 2]);
            regs[e] = ptx::mul_cvt_fp32_to_fp4_4x<USE_STOCHASTIC_ROUNDING>(
                in01, in23, block_scale_inverse_2x, rbits);
          }
        }
```
- Packs every 4 elements into a PTX intrinsic that scales then converts to fp4 (E2M1). SR ON injects random bits; SR OFF passes zero. Uses bf16 fast intrinsic in our case.

```cuda
        const int group = thread_lane / 16;
        uint32_t val[2];
        uint32_t *regs_4x = reinterpret_cast<uint32_t *>(regs);

        switch (group) {
          case 0:
            val[0] = regs_4x[0];
            val[1] = regs_4x[1];
            break;
          case 1:
            val[0] = regs_4x[1];
            val[1] = regs_4x[0];
            break;
        }
        uint32_t *out_t_data_sh_as_uint32_t =
            reinterpret_cast<uint32_t *>(&out_t_data_sh[shmem_offset_base_colwise_out_t]);
        out_t_data_sh_as_uint32_t[group] = val[0];
        out_t_data_sh_as_uint32_t[(group + 1) & 1] = val[1];
      }
    }
```
- Lanes split into two groups to permute 32‑bit fp4 packs, reducing bank conflicts. Each thread writes two 32‑bit words into transpose shared buffer at its column/row offset.

### 3B. Rowwise path (always runs)
```cuda
    {
      const size_t stage_rowwise_scales_offset_Y = stage * BUFF_DIM_Y;
#pragma unroll
      for (size_t it = 0; it < ITERATIONS_NORMAL; ++it) {
        const size_t it_thread_offset_Y_rowwise = thread_offset_Y_rowwise + it * THREADS_Y_ROWWISE;

        const size_t shmem_offset_base_rowwise_in =
            buff_offset_in + it_thread_offset_Y_rowwise * BUFF_IN_DIM_X;
        const size_t shmem_offset_base_rowwise_out =
            buff_offset_out + it_thread_offset_Y_rowwise * BUFF_OUT_DIM_X;

        const size_t it_offset_Y = stage_offset_Y + it * THREADS_Y_ROWWISE;

        block_amax = 0.0f;
        float in_compute_rowwise[SCALE_DIM];
        Vec<IType, PACK_SIZE> in_cached[WAVES];
        Vec<IType2, PACK_SIZE / 2> in_IType[WAVES];
```
- Rowwise scales computed in two iterations (16 rows each). Offsets pick the correct rows in shared buffers. Local arrays for compute or cached bf16 values allocated per thread.

```cuda
        if constexpr (NO_ACTIVATIONS_NOT_FP32_INPUT) {
          IType2 thread_amax_2x = {static_cast<IType>(0.0f), static_cast<IType>(0.0f)};
#pragma unroll
          for (int w = 0; w < WAVES; ++w) {
            const size_t swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM;
            const size_t swizzled_thread_idx = thread_offset_X_rowwise + swizzled_group_idx;
            const size_t shmem_offset_rowwise = shmem_offset_base_rowwise_in + swizzled_thread_idx;
            in_IType[w].load_from(&in_sh[shmem_offset_rowwise]);
#pragma unroll
            for (int e = 0; e < PACK_SIZE / 2; ++e) {
              ptx::abs_max_2x(thread_amax_2x, thread_amax_2x, in_IType[w].data.elt[e]);
            }
          }
          block_amax = static_cast<float>(__hmax(__habs(thread_amax_2x.x), __habs(thread_amax_2x.y)));
        } else if constexpr (IS_CACHED_ACT_OP) { ... } else { ... }
```
- Fast path: loads two 8‑element waves per thread with bank‑swizzled indices to reduce conflicts; accumulates vector abs‑max in bf16. Other branches (activation/cache) skipped here but included in code for completeness; they repeat compute with activation and caching.

```cuda
        const nvfp4_scale_t S_dec_b_fp8 =
            compute_decoding_scaling_factor(block_amax, S_enc_rowwise);

        const size_t scales_offset_Y =
            scales_offset_Y_rowwise + stage * BUFF_DIM_Y + it * THREADS_Y_ROWWISE;
        const size_t scales_offset_X = scales_offset_X_rowwise;
        const size_t scale_idx_global = scales_offset_Y * scale_stride + scales_offset_X;

        const bool rowwise_scale_is_within_bounds_Y =
            (stage_rowwise_scales_offset_Y + it * THREADS_Y_ROWWISE + tid_Y_rowwise) < chunk_rows;
        if (rowwise_scale_is_within_bounds_X && rowwise_scale_is_within_bounds_Y) {
          scales_ptr[scale_idx_global] = S_dec_b_fp8;
        }
```
- Derives rowwise decode scale and writes it if within bounds. For 1024×768 all threads store without branching.

```cuda
        constexpr float float_max = detail::TypeExtrema<float>::max;
        const float block_scale_inverse = fminf(
            1.0f / (static_cast<float>(S_dec_b_fp8) * S_dec_rowwise), float_max);
        const float2 block_scale_inverse_2x{block_scale_inverse, block_scale_inverse};

#pragma unroll
        for (int w = 0; w < WAVES; ++w) {
          Vec<fp4e2m1x4, PACK_SIZE / 4> out;
#pragma unroll
          for (int e = 0; e < PACK_SIZE / 4; ++e) {
            const uint32_t rbits = get_rbits(rng, random_uint4, rnd_idx);
            if constexpr (NO_ACTIVATIONS_NOT_FP32_INPUT) {
              const uint64_t elts = *reinterpret_cast<uint64_t *>(&in_IType[w].data.elt[2 * e]);
              out.data.elt[e] = ptx::mul_cvt_bf16_to_fp4_4x<USE_STOCHASTIC_ROUNDING>(
                  elts, block_scale_inverse_2x, rbits);
            } else if constexpr (IS_CACHED_ACT_OP) {
              const uint64_t elts = *reinterpret_cast<uint64_t *>(&in_cached[w].data.elt[4 * e]);
              out.data.elt[e] = ptx::mul_cvt_bf16_to_fp4_4x<USE_STOCHASTIC_ROUNDING>(
                  elts, block_scale_inverse_2x, rbits);
            } else {
              const int j = w * PACK_SIZE + 4 * e;
              const float2 in01 = make_float2(in_compute_rowwise[j], in_compute_rowwise[j + 1]);
              const float2 in23 = make_float2(in_compute_rowwise[j + 2], in_compute_rowwise[j + 3]);
              out.data.elt[e] = ptx::mul_cvt_fp32_to_fp4_4x<USE_STOCHASTIC_ROUNDING>(
                  in01, in23, block_scale_inverse_2x, rbits);
            }
          }

          const size_t swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM;
          const size_t swizzled_idx = swizzled_group_idx + thread_offset_X_rowwise;
          const size_t shmem_offset_rowwise = shmem_offset_base_rowwise_out + swizzled_idx / 2;
          out.store_to(&out_data_sh[shmem_offset_rowwise]);
        }
      }
    }
```
- Computes encode scale then converts each 8‑element wave into fp4 using SR or deterministic rounding. Swizzled indices spread writes across SMEM banks; each store writes packed fp4 (2 values/byte) into rowwise output buffer.

### 3C. Finalize stage and launch TMA writes
```cuda
    __builtin_assume(thread_amax >= 0);
    thread_amax = fmaxf(thread_amax, block_amax);

    ptx::fence_proxy_async_shared_cta();
    __syncthreads();

    if (is_master_thread) {
      const size_t global_offset_Y = block_offset_Y + stage_offset_Y;
      const size_t global_offset_X = block_offset_X;

      const size_t global_offset_Y_t = block_offset_Y_t;
      const size_t global_offset_X_t = block_offset_X_t + stage_offset_Y;

      ptx::cp_async_bulk_tensor_2d_shared_to_global(
          reinterpret_cast<const uint64_t *>(&tensor_map_output), global_offset_X, global_offset_Y,
          reinterpret_cast<uint64_t *>(&out_data_sh[buff_offset_out]));

      if constexpr (RETURN_TRANSPOSE) {
        ptx::cp_async_bulk_tensor_2d_shared_to_global(
            reinterpret_cast<const uint64_t *>(&tensor_map_output_t), global_offset_X_t,
            global_offset_Y_t, reinterpret_cast<uint64_t *>(&out_t_data_sh[buff_offset_out_t]));
      }

      ptx::cp_async_bulk_commit_group();
    }
  }  // end of stages
```
- Updates running amax (not used later but kept). Memory fence then CTA barrier to ensure shared writes visible to TMA. Master thread issues async bulk copies of rowwise and transpose outputs for this stage, then commits the async group. Loop repeats for next stage with overlapped loads/stores.

### 3D. Store transpose scales vectorized
```cuda
  if (RETURN_TRANSPOSE && colwise_scale_is_within_bounds_Y) {
    using ScalesVec = Vec<nvfp4_scale_t, SCALES_PER_CHUNK_Y>;
    const size_t scale_idx_sh = tid_Y_t * SCALES_PER_CHUNK_Y;
    ScalesVec &scales_vec = *reinterpret_cast<ScalesVec *>(&out_colwise_scales_sh[scale_idx_sh]);
    const size_t scale_idx_global = scales_offset_Y_t * scale_stride_t + scales_offset_X_t;
    const size_t count = (chunk_rows >= CHUNK_DIM_Y) ? SCALES_PER_CHUNK_Y : (chunk_rows / SCALE_DIM);
    nvfp4_scale_t *dst = &scales_t_ptr[scale_idx_global];
    constexpr size_t vec_bytes = SCALES_PER_CHUNK_Y * sizeof(nvfp4_scale_t);
    if (count == SCALES_PER_CHUNK_Y && (reinterpret_cast<uintptr_t>(dst) % vec_bytes == 0)) {
      scales_vec.store_to(dst);
    } else {
      scales_vec.store_to_elts(dst, 0, count);
    }
  }

  destroy_barriers<STAGES>(mbar, is_master_thread);
#else
  NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}
```
- After all stages, each thread writes its columnwise scales vector to global. Aligned case uses vectorized store; tails fall back to elementwise (not taken for 1024×768). Finally, master destroys barriers; compile guard mirrors start.

---
## 4. SR ON vs OFF (where behavior diverges)
- `random_uint4 = USE_STOCHASTIC_ROUNDING ? rng.generate4() : uint4{0,0,0,0};`
- `rbits = get_rbits(rng, random_uint4, rnd_idx);` inside both colwise and rowwise quant loops uses SR bits only when template boolean true.
- With SR OFF: rbits=0 → PTX `mul_cvt_*` performs deterministic round-to-nearest-even.
- With SR ON: Philox stream per thread; `rnd_idx` rotates 0..3 so one `uint4` serves four 4‑value conversions; extra RNG latency hides under unrolled math and TMA overlap.

---
## 5. Mapping for 1024×768
- Grid: 6 (X) × 8 (Y) CTAs. Each CTA covers 128×128 chunk; no boundary predicates fire.
- Stage loop: 4 stages (32 rows each) per CTA; X fixed per CTA because CHUNK_DIM_X=128.
- Threads: rowwise view 16×8 covers 32 rows × 128 cols per stage; columnwise view 128 threads cover 128 cols × 32 rows per stage for transpose.
- Scales: rowwise scales shape 1024×48; per CTA each thread writes one scale per row block and X block. Transpose scales shape 768×64; vector store path is used because destinations are aligned and full.

---
## 6. Key data/compute optimizations (as seen line‑by‑line)
- Double buffering (`buff = stage % 2`) plus `cp_async_bulk` + `mbarrier` overlaps load/compute/store every stage.
- SMEM alignment to 128B satisfies TMA; buffers laid out contiguously to allow single TMA per output per stage.
- Bank‑conflict mitigation via `bank_group` swizzle and columnwise pack permutation (group switch).
- Unrolled inner loops (`#pragma unroll`) expose ILP; per‑wave vector types (FPx2, fp4x4) keep conversion throughput high.
- Separate rowwise/colwise scaling lets each path choose its own amax and global encode factor for tighter dynamic range.

---
## 7. Takeaways
- The 1D kernel always produces both rowwise fp4 and (when requested) a fully transposed fp4 buffer, each with its own scale tensor.
- Control flow is identical for SR ON/OFF; only the random bits differ, so performance impact is minimal while improving rounding quality.
- For aligned shapes like 1024×768, all fast paths are taken: no tails, vectorized scale stores, fully coalesced TMA transactions.

