# NVFP4 Quantize-Transpose Kernel Deep Dive

**Analysis Date**: 2025-11-27
**Input Shape**: 128 × 1024 (rows × cols)
**Analyzed Cases**:
- 1D Scaling with `use_stochastic_rounding={True, False}` and `return_transpose=True`

---

## Table of Contents

1. [Kernel Entry Point](#1-kernel-entry-point)
2. [High-Level Design](#2-high-level-design)
3. [Kernel Configuration](#3-kernel-configuration)
4. [Pipeline Architecture](#4-pipeline-architecture)
5. [Thread/Warp/Block Mapping](#5-threadwarpblock-mapping)
6. [Step-by-Step Execution](#6-step-by-step-execution)
7. [PTX Intrinsics](#7-ptx-intrinsics)
8. [Memory Layout](#8-memory-layout)
9. [Performance Optimizations](#9-performance-optimizations)

---

## 1. Kernel Entry Point

### 1.1 Host-Side Launch Function

**File**: [transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh:1156-1281](../transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh#L1156-L1281)

```cpp
template <bool use_2d_quantization>
void quantize_transpose(const Tensor &input, const Tensor *noop, Tensor *output,
                        const QuantizationConfig *quant_config, cudaStream_t stream) {
  // For 1D quantization, use_2d_quantization = false
  // For our case: input is 128 × 1024 BF16

  bool use_stochastic_rounding = quant_config ? quant_config->stochastic_rounding : false;
  bool return_transpose = output->has_columnwise_data();  // true for our case

  const size_t rows = 128;
  const size_t cols = 1024;

  // Grid configuration
  const size_t blocks_Y = DIVUP(rows, CHUNK_DIM_Y);  // DIVUP(128, 128) = 1
  const size_t blocks_X = DIVUP(cols, CHUNK_DIM_X);  // DIVUP(1024, 128) = 8
  const dim3 grid(blocks_X, blocks_Y);  // grid(8, 1)
  const size_t block_size = THREADS_NUM;  // 128 threads per block

  // Select kernel based on template parameters
  auto kernel = quantize_transpose_nvfp4_kernel<COMPUTE_ACTIVATIONS=false, ParamOP, OP,
                                                IType=bf16, USE_STOCHASTIC_ROUNDING,
                                                RETURN_TRANSPOSE=true>;

  // Launch with dynamic shared memory
  constexpr size_t dshmem_size = in_mem + out_mem + out_scales_transpose_mem + TMA_SHMEM_ALIGNMENT;
  kernel<<<grid, block_size, dshmem_size, stream>>>(
      tensor_map_input, tensor_map_output, tensor_map_output_transpose,
      scales_ptr, scales_transpose_ptr, noop_ptr,
      amax_rowwise_ptr, amax_colwise_ptr, rows, cols,
      scale_stride, scale_stride_transpose, rng_state);
}
```

**For our 128 × 1024 input**:
- Grid: 8 blocks × 1 block (8 blocks in X dimension, 1 block in Y dimension)
- Each block: 128 threads
- Each block processes a 128 × 128 chunk of the input
- Block 0 processes columns [0, 128), Block 1 processes [128, 256), ..., Block 7 processes [896, 1024)

---

## 2. High-Level Design

### 2.1 Kernel Objectives

The `quantize_transpose_nvfp4_kernel` performs three main tasks simultaneously:

1. **Rowwise Quantization**: Quantize the input tensor row-by-row, producing identity output
2. **Columnwise Quantization + Transpose**: Quantize column-by-column while transposing, producing transposed output
3. **Scale Factor Computation**: Compute per-block E4M3 FP8 scaling factors for both orientations

### 2.2 Quantization Strategy

**Two-Stage Scaling**:
```
Original Value (BF16) → Scaled FP32 → Quantized NVFP4
```

**Scaling Formula**:
```
FP4_value = quantize(BF16_value * block_scale_inverse)

where:
  block_scale_inverse = 1 / (S_dec_b_fp8 * S_dec)
  S_dec_b_fp8 = per-block E4M3 FP8 scale (computed from block amax)
  S_dec = global FP32 scale (computed from global amax)
```

**NVFP4 Format** (E2M1):
- 1 sign bit, 2 exponent bits, 1 mantissa bit
- Range: [-6.0, 6.0]
- 16 distinct values (including zero, inf, NaN)

### 2.3 Key Design Principles

1. **Double Buffering**: Use 2 input buffers in shared memory for pipelining
2. **TMA (Tensor Memory Accelerator)**: Hardware-accelerated async copies between global and shared memory
3. **Staged Processing**: Break 128×128 chunk into 4 tiles of 32×128 for pipeline parallelism
4. **Bank Conflict Avoidance**: Manual swizzling to distribute memory accesses across SHMEM banks
5. **Fused Operations**: Combine scaling, conversion, and transposition in single passes

---

## 3. Kernel Configuration

### 3.1 Compile-Time Constants

**File**: [transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh:37-93](../transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh#L37-L93)

```cpp
constexpr size_t SCALE_DIM = 16;       // NVFP4 block size (16 elements per scale)

// Chunk processed by one thread block
constexpr size_t CHUNK_DIM_Y = 128;    // 128 rows
constexpr size_t CHUNK_DIM_X = 128;    // 128 cols
constexpr size_t THREADS_NUM = 128;    // 128 threads per block

// Scales per chunk
constexpr size_t SCALES_PER_CHUNK_Y = CHUNK_DIM_Y / SCALE_DIM;  // 128 / 16 = 8
constexpr size_t SCALES_PER_CHUNK_X = CHUNK_DIM_X / SCALE_DIM;  // 128 / 16 = 8

// Tile dimensions for pipelining
constexpr size_t TILE_DIM_Y = 32;      // 32 rows per tile
constexpr size_t TILE_DIM_X = 128;     // 128 cols per tile

// Pipeline stages
constexpr size_t TILES_Y = CHUNK_DIM_Y / TILE_DIM_Y;  // 128 / 32 = 4 tiles in Y
constexpr size_t TILES_X = CHUNK_DIM_X / TILE_DIM_X;  // 128 / 128 = 1 tile in X
constexpr size_t STAGES = TILES_Y * TILES_X;          // 4 × 1 = 4 stages

// Double buffering
constexpr size_t BUFFS_NUM = 2;        // 2 buffers for pipelining

// Buffer dimensions
constexpr size_t BUFF_DIM_Y = TILE_DIM_Y;  // 32 rows
constexpr size_t BUFF_DIM_X = TILE_DIM_X;  // 128 cols
constexpr size_t BUFF_SIZE = BUFF_DIM_Y * BUFF_DIM_X;  // 32 × 128 = 4096 elements

// Input buffer: BF16 elements
constexpr size_t BUFF_IN_SIZE = 32 × 128;  // 4096 BF16 = 8192 bytes

// Output buffer: NVFP4 elements (4 bits per element)
constexpr size_t BUFF_OUT_DIM_Y = 32;
constexpr size_t BUFF_OUT_DIM_X = (128 * 4) / 8;  // 128 elements × 4 bits / 8 = 64 bytes
constexpr size_t BUFF_OUT_SIZE = 32 × 64;  // 2048 bytes

// Transposed output buffer: NVFP4 elements
constexpr size_t BUFF_OUT_T_DIM_Y = 128;
constexpr size_t BUFF_OUT_T_DIM_X = (32 * 4) / 8;  // 32 elements × 4 bits / 8 = 16 bytes
constexpr size_t BUFF_OUT_T_SIZE = 128 × 16;  // 2048 bytes

// Bank conflict reduction
constexpr size_t PACK_SIZE = 8;        // 8 elements loaded per wave
constexpr size_t WAVES = SCALE_DIM / PACK_SIZE;  // 16 / 8 = 2 waves

// Thread organization for rowwise processing
constexpr size_t THREADS_X_ROWWISE = CHUNK_DIM_X / SCALE_DIM;  // 128 / 16 = 8 threads in X
constexpr size_t THREADS_Y_ROWWISE = THREADS_NUM / THREADS_X_ROWWISE;  // 128 / 8 = 16 threads in Y

// Iterations
constexpr size_t ITERATIONS_NORMAL = BUFF_DIM_Y / THREADS_Y_ROWWISE;  // 32 / 16 = 2
constexpr size_t ITERATIONS_TRANSPOSE = BUFF_IN_DIM_Y / SCALE_DIM;    // 32 / 16 = 2
```

### 3.2 For 128 × 1024 Input

```
Grid:           8 blocks × 1 block
Block Size:     128 threads
Chunk per block: 128 × 128
Total stages:    4 per block (process 32×128 per stage)
```

**Block Processing**:
```
Block 0: rows [0, 128), cols [0, 128)
Block 1: rows [0, 128), cols [128, 256)
Block 2: rows [0, 128), cols [256, 384)
Block 3: rows [0, 128), cols [384, 512)
Block 4: rows [0, 128), cols [512, 640)
Block 5: rows [0, 128), cols [640, 768)
Block 6: rows [0, 128), cols [768, 896)
Block 7: rows [0, 128), cols [896, 1024)
```

---

## 4. Pipeline Architecture

### 4.1 Double-Buffered Pipelining

The kernel uses **4 stages** with **2 buffers** to enable overlapping of:
- **TMA transfers** (Global Memory ↔ Shared Memory)
- **Compute** (SHMEM processing by threads)

```
Stage 0: TMA load buffer 0 → Wait → Compute on buffer 0 → TMA store buffer 0
Stage 1: TMA load buffer 1 → Wait → Compute on buffer 1 → TMA store buffer 1
Stage 2: TMA load buffer 0 → Wait → Compute on buffer 0 → TMA store buffer 0
Stage 3: TMA load buffer 1 → Wait → Compute on buffer 1 → TMA store buffer 1
```

### 4.2 Pipeline Timeline

```
Time →
Stage 0: |--TMA Load Buffer 0--|--Wait--|========Compute========|--TMA Store--|
Stage 1:                        |--TMA Load Buffer 1--|--Wait--|========Compute========|--TMA Store--|
Stage 2:                                                |--TMA Load Buffer 0--|--Wait--|========Compute========|--TMA Store--|
Stage 3:                                                                        |--TMA Load Buffer 1--|--Wait--|========Compute========|--TMA Store--|
```

**Key**: Compute on stage N overlaps with TMA load for stage N+1

### 4.3 Mbarrier Synchronization

**File**: [transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh:238-275](../transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh#L238-L275)

Each stage has its own **mbarrier** (shared memory barrier):

```cpp
__shared__ alignas(8) uint64_t mbar[STAGES];  // 4 barriers for 4 stages

// Initialize all barriers (only master thread)
initialize_barriers<STAGES, THREADS_NUM>(mbar, is_master_thread);

// Stage loop
for (size_t stage = 0; stage < STAGES; ++stage) {
  // Prefetch next stage
  if (next_stage < STAGES) {
    cp_async_bulk_wait_group_read<1>();  // Wait for previous TMA store to finish
    copy_2d_to_shared(&in_sh[next_buff_offset], &tensor_map_input,
                      global_offset_X, global_offset_Y,
                      shmem_buff_size, &mbar[next_stage], is_master_thread);
  }

  ptx::fence_proxy_async_shared_cta();

  // Wait for current stage's data to arrive
  ptx::mbarrier_wait_parity(&mbar[stage], 0);

  // === COMPUTE ON CURRENT BUFFER ===

  // Fence to ensure compute writes visible to TMA
  ptx::fence_proxy_async_shared_cta();
  __syncthreads();

  // TMA store (master thread only)
  if (is_master_thread) {
    ptx::cp_async_bulk_tensor_2d_shared_to_global(...);
    ptx::cp_async_bulk_commit_group();
  }
}
```

---

## 5. Thread/Warp/Block Mapping

### 5.1 Thread Organization

**128 threads** per block organized in multiple layouts depending on operation:

#### Rowwise Processing Layout

```cpp
const size_t tid_Y_rowwise = threadIdx.x / THREADS_X_ROWWISE;  // threadIdx.x / 8
const size_t tid_X_rowwise = threadIdx.x % THREADS_X_ROWWISE;  // threadIdx.x % 8

// Results in 16 × 8 thread grid:
// - 16 threads in Y (rows)
// - 8 threads in X (columns), each handling one 16-element block
```

**Example mapping**:
```
threadIdx.x=0  → tid_Y=0, tid_X=0  (processes row 0, block [0:16))
threadIdx.x=1  → tid_Y=0, tid_X=1  (processes row 0, block [16:32))
threadIdx.x=7  → tid_Y=0, tid_X=7  (processes row 0, block [112:128))
threadIdx.x=8  → tid_Y=1, tid_X=0  (processes row 1, block [0:16))
...
threadIdx.x=127 → tid_Y=15, tid_X=7 (processes row 15, block [112:128))
```

Each thread processes 2 rows (ITERATIONS_NORMAL=2):
- Thread with tid_Y=0 processes rows {0, 16}
- Thread with tid_Y=1 processes rows {1, 17}
- Thread with tid_Y=15 processes rows {15, 31}

#### Columnwise Processing Layout

```cpp
const size_t tid_X_colwise = threadIdx.x;  // 0 to 127

// Each thread processes one column
// Thread 0 → column 0
// Thread 1 → column 1
// ...
// Thread 127 → column 127
```

Each thread processes 2 blocks vertically (ITERATIONS_TRANSPOSE=2):
- Iteration 0: rows [0:16) in its column
- Iteration 1: rows [16:32) in its column

### 5.2 Warp Organization

```cpp
const int thread_lane = threadIdx.x % THREADS_PER_WARP;  // 0-31 within warp
const int bank_group = thread_lane / THREADS_PER_BANK;   // 0-3 (THREADS_PER_BANK=8)
```

**4 warps per block**:
- Warp 0: threads [0, 31]
- Warp 1: threads [32, 63]
- Warp 2: threads [64, 95]
- Warp 3: threads [96, 127]

---

## 6. Step-by-Step Execution

### 6.1 Kernel Initialization

**File**: [transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh:133-246](../transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh#L133-L246)

```cpp
__global__ void quantize_transpose_nvfp4_kernel(...) {
  // === STEP 1: Early exit check ===
  if (noop != nullptr && noop[0] == 1.0f) {
    return;  // Skip quantization if noop flag set
  }

  // === STEP 2: Initialize RNG for stochastic rounding ===
  const size_t rng_sequence = threadIdx.x + blockIdx.x * THREADS_NUM +
                              blockIdx.y * gridDim.x * THREADS_NUM;
  const size_t rng_seed = rng_state[0];
  const size_t rng_offset = rng_state[1];
  transformer_engine::curanddx::detail::philox4x32_native_state<10> rng;
  rng.init(rng_seed, rng_sequence, rng_offset);

  uint4 random_uint4 = USE_STOCHASTIC_ROUNDING ? rng.generate4() : uint4{0, 0, 0, 0};
  int rnd_idx = 0;  // Current index into random_uint4 (0-3)

  // === STEP 3: Compute block offsets ===
  const size_t block_offset_Y = blockIdx.y * CHUNK_DIM_Y;  // 0 for our case
  const size_t block_offset_X = blockIdx.x * CHUNK_DIM_X;  // 0, 128, 256, ..., 896

  const size_t block_offset_Y_t = blockIdx.x * CHUNK_DIM_X;  // Transposed: 0, 128, 256, ..., 896
  const size_t block_offset_X_t = blockIdx.y * CHUNK_DIM_Y;  // Transposed: 0

  // === STEP 4: Compute thread offsets ===
  const size_t tid_Y_rowwise = threadIdx.x / 8;  // 0-15
  const size_t tid_X_rowwise = threadIdx.x % 8;  // 0-7
  const size_t thread_offset_X_rowwise = tid_X_rowwise * SCALE_DIM;  // 0, 16, 32, ..., 112

  const size_t tid_X_colwise = threadIdx.x;  // 0-127

  // === STEP 5: Allocate shared memory ===
  extern __shared__ char dynamic_shmem[];
  uintptr_t dshmem = (base_shmem_ptr + TMA_SHMEM_ALIGNMENT - 1) &
                     ~(static_cast<uintptr_t>(TMA_SHMEM_ALIGNMENT - 1));

  IType *in_sh = reinterpret_cast<IType *>(dshmem);
  fp4e2m1x2 *out_data_sh = reinterpret_cast<fp4e2m1x2 *>(dshmem + in_mem);
  fp4e2m1x2 *out_t_data_sh = reinterpret_cast<fp4e2m1x2 *>(dshmem + in_mem + out_mem_rowwise_data);
  nvfp4_scale_t *out_colwise_scales_sh = reinterpret_cast<nvfp4_scale_t *>(...);

  // === STEP 6: Compute global scaling factors ===
  const float S_enc_rowwise = compute_global_encode_scaling_factor_FP4(*amax_rowwise_ptr);
  const float S_dec_rowwise = 1.0 / S_enc_rowwise;

  const float S_enc_colwise = compute_global_encode_scaling_factor_FP4(*amax_colwise_ptr);
  const float S_dec_colwise = 1.0 / S_enc_colwise;

  // === STEP 7: Initialize mbarriers ===
  __shared__ alignas(8) uint64_t mbar[STAGES];
  initialize_barriers<STAGES, THREADS_NUM>(mbar, is_master_thread);

  // === STEP 8: Prefetch first tile ===
  copy_2d_to_shared(&in_sh[0], &tensor_map_input, block_offset_X, block_offset_Y,
                    shmem_buff_size, &mbar[0], is_master_thread);
}
```

**Key Points**:
- RNG initialized per-thread with unique sequence number for stochastic rounding
- Global scaling factors computed once for entire kernel
- Shared memory carefully partitioned for input, output, transposed output, and scales
- First TMA transfer initiated before entering main loop

### 6.2 Main Stage Loop

**File**: [transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh:248-595](../transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh#L248-L595)

```cpp
#pragma unroll
for (size_t stage = 0; stage < STAGES; ++stage) {  // 4 stages
  const size_t buff = stage % BUFFS_NUM;  // 0, 1, 0, 1
  const size_t next_stage = stage + 1;
  const size_t stage_offset_Y = stage * BUFF_DIM_Y;  // 0, 32, 64, 96

  const size_t buff_offset_in = buff * BUFF_IN_SIZE;
  const size_t buff_offset_out = buff * BUFF_OUT_SIZE;
  const size_t buff_offset_out_t = buff * BUFF_OUT_T_SIZE;

  // === SUBSTEP 1: Prefetch next tile ===
  if (next_stage < STAGES) {
    cp_async_bulk_wait_group_read<1>();  // Wait for previous store to finish reading SHMEM

    const size_t next_buff = next_stage % BUFFS_NUM;
    const size_t next_stage_offset_Y = next_stage * BUFF_DIM_Y;
    const size_t global_offset_Y = block_offset_Y + next_stage_offset_Y;

    copy_2d_to_shared(&in_sh[next_buff_offset], &tensor_map_input,
                      block_offset_X, global_offset_Y, shmem_buff_size,
                      &mbar[next_stage], is_master_thread);
  }

  ptx::fence_proxy_async_shared_cta();

  // === SUBSTEP 2: Wait for current tile data ===
  ptx::mbarrier_wait_parity(&mbar[stage], 0);

  // === SUBSTEP 3: COLUMNWISE QUANTIZATION + TRANSPOSE ===
  if constexpr (RETURN_TRANSPOSE) {
    // Process 2 vertical blocks per thread
    for (size_t it = 0; it < ITERATIONS_TRANSPOSE; ++it) {  // 2 iterations
      // ... (detailed in 6.3)
    }
  }

  // === SUBSTEP 4: ROWWISE QUANTIZATION ===
  {
    // Process 2 rows per thread
    for (size_t it = 0; it < ITERATIONS_NORMAL; ++it) {  // 2 iterations
      // ... (detailed in 6.4)
    }
  }

  // === SUBSTEP 5: Synchronize and initiate TMA store ===
  ptx::fence_proxy_async_shared_cta();
  __syncthreads();

  if (is_master_thread) {
    const size_t global_offset_Y = block_offset_Y + stage_offset_Y;

    // Store rowwise quantized data
    ptx::cp_async_bulk_tensor_2d_shared_to_global(
        &tensor_map_output, global_offset_X, global_offset_Y,
        &out_data_sh[buff_offset_out]);

    // Store transposed quantized data
    if constexpr (RETURN_TRANSPOSE) {
      const size_t global_offset_Y_t = block_offset_Y_t;
      const size_t global_offset_X_t = block_offset_X_t + stage_offset_Y;

      ptx::cp_async_bulk_tensor_2d_shared_to_global(
          &tensor_map_output_t, global_offset_X_t, global_offset_Y_t,
          &out_t_data_sh[buff_offset_out_t]);
    }

    ptx::cp_async_bulk_commit_group();
  }
}
```

**Timeline for one stage** (e.g., Stage 1):
1. **Prefetch Stage 2** while **Compute Stage 1** can proceed
2. **Wait** for Stage 1 data to arrive in SHMEM
3. **Compute columnwise** quantization + transpose (all 128 threads)
4. **Compute rowwise** quantization (all 128 threads)
5. **Store** both outputs via TMA (master thread initiates)

### 6.3 Columnwise Quantization + Transpose (Detailed)

**File**: [transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh:280-391](../transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh#L280-L391)

Each of the 128 threads processes **one column** (tid_X_colwise = threadIdx.x).
Within that column, the thread processes **two 16-element blocks** vertically (ITERATIONS_TRANSPOSE=2).

```cpp
// COLWISE scaling (inside stage loop)
if constexpr (RETURN_TRANSPOSE) {
  #pragma unroll
  for (size_t it = 0; it < ITERATIONS_TRANSPOSE; ++it) {  // 2 iterations
    // === SUBSTEP 3.1: Compute SHMEM offsets ===
    const size_t in_thread_offset_Y = 0 + it * SCALE_DIM;  // 0 or 16
    const size_t in_thread_offset_X = thread_offset_X_colwise;  // threadIdx.x

    const size_t shmem_offset_base_colwise_in =
        buff_offset_in + in_thread_offset_Y * BUFF_IN_DIM_X + in_thread_offset_X;

    const size_t out_t_thread_offset_Y = thread_offset_X_colwise;  // threadIdx.x
    const size_t out_t_thread_offset_X = 0 + it * BUFF_OUT_IT_OFFSET;  // 0 or 8

    const size_t shmem_offset_base_colwise_out_t =
        buff_offset_out_t + out_t_thread_offset_Y * BUFF_OUT_T_DIM_X + out_t_thread_offset_X;

    // === SUBSTEP 3.2: Load 16 elements and compute block amax ===
    block_amax = 0.0f;
    float in_compute_colwise[SCALE_DIM];  // 16 elements

    // For NO_ACTIVATIONS_NOT_FP32_INPUT (BF16 input, no activations):
    IType block_amax_f16 = static_cast<IType>(0.0f);
    #pragma unroll
    for (int i = 0; i < SCALE_DIM; ++i) {  // 16 elements
      const int shmem_offset_colwise = shmem_offset_base_colwise_in + i * BUFF_IN_DIM_X;
      in_colwise_IType[i] = in_sh[shmem_offset_colwise];

      // Compute amax using BF16 arithmetic (faster)
      block_amax_f16 = __hmax(block_amax_f16, __habs(in_colwise_IType[i]));
    }
    block_amax = static_cast<float>(block_amax_f16);

    // === SUBSTEP 3.3: Compute E4M3 FP8 scaling factor ===
    const nvfp4_scale_t S_dec_b_fp8 =
        compute_decoding_scaling_factor(block_amax, S_enc_colwise);

    // Store to SHMEM for later writeback
    const size_t scale_idx_sh = tid_Y_t * SCALES_PER_CHUNK_Y + stage * ITERATIONS_TRANSPOSE + it;
    out_colwise_scales_sh[scale_idx_sh] = S_dec_b_fp8;

    // === SUBSTEP 3.4: Compute per-block encoding scale ===
    const float block_scale_inverse =
        fminf(1.0f / (static_cast<float>(S_dec_b_fp8) * S_dec_colwise), float_max);
    const float2 block_scale_inverse_2x{block_scale_inverse, block_scale_inverse};

    // === SUBSTEP 3.5: Quantize 16 elements to FP4 ===
    fp4e2m1x4 regs[SCALE_DIM / 4];  // 16 / 4 = 4 registers, each holds 4 FP4 values

    #pragma unroll
    for (int e = 0; e < SCALE_DIM / 4; ++e) {  // 4 iterations
      // Get random bits for stochastic rounding
      const uint32_t rbits = get_rbits(rng, random_uint4, rnd_idx);

      // Pack 4 BF16 values into uint64_t
      const uint64_t elts = *reinterpret_cast<uint64_t *>(&in_colwise_IType[4 * e]);

      // PTX intrinsic: mul + cvt fused (detailed in section 7)
      regs[e] = ptx::mul_cvt_bf16_to_fp4_4x<USE_STOCHASTIC_ROUNDING>(
          elts, block_scale_inverse_2x, rbits);
    }

    // === SUBSTEP 3.6: Write to SHMEM with transpose ===
    // Manual swizzling to reduce bank conflicts
    const int group = thread_lane / 16;  // 0 or 1 (half-warp)
    uint32_t val[2];
    uint32_t *regs_4x = reinterpret_cast<uint32_t *>(regs);

    // Swap order for group 1 to reduce conflicts
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

**Example for Thread 0, Iteration 0**:
- Reads column 0, rows [0:16) from SHMEM
- Computes amax = max(|in_sh[0][0]|, |in_sh[1][0]|, ..., |in_sh[15][0]|)
- Computes S_dec_b_fp8 from amax
- Quantizes 16 values to FP4 using PTX intrinsic
- Writes to transposed location: row 0, columns [0:16) of transposed output

**Example for Thread 0, Iteration 1**:
- Reads column 0, rows [16:32) from SHMEM
- Same process
- Writes to transposed location: row 0, columns [16:32) of transposed output

**Result**: After all 128 threads complete both iterations:
- Transposed output has 128 rows × 32 columns (matching BUFF_OUT_T_DIM_Y × BUFF_OUT_T_DIM_X)
- Each row in transposed output corresponds to a column in the original input
- Scaling factors stored in SHMEM for final writeback

### 6.4 Rowwise Quantization (Detailed)

**File**: [transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh:393-564](../transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh#L393-L564)

Each thread processes **2 rows** (ITERATIONS_NORMAL=2) and handles **one 16-element block** per row.

The 128 threads are organized as a 16 × 8 grid:
- 16 threads in Y (rows), each handling 2 rows
- 8 threads in X (columns), each handling one 16-element block

```cpp
// ROWWISE scaling (inside stage loop)
{
  const size_t stage_rowwise_scales_offset_Y = stage * BUFF_DIM_Y;

  #pragma unroll
  for (size_t it = 0; it < ITERATIONS_NORMAL; ++it) {  // 2 iterations
    // === SUBSTEP 4.1: Compute thread offsets ===
    const size_t it_thread_offset_Y_rowwise = thread_offset_Y_rowwise + it * THREADS_Y_ROWWISE;
    // Example: tid_Y_rowwise=0, it=0 → offset=0
    //          tid_Y_rowwise=0, it=1 → offset=16

    const size_t shmem_offset_base_rowwise_in =
        buff_offset_in + it_thread_offset_Y_rowwise * BUFF_IN_DIM_X;
    const size_t shmem_offset_base_rowwise_out =
        buff_offset_out + it_thread_offset_Y_rowwise * BUFF_OUT_DIM_X;

    // === SUBSTEP 4.2: Load 16 elements with swizzling and compute amax ===
    block_amax = 0.0f;
    Vec<IType2, PACK_SIZE / 2> in_IType[WAVES];  // 2 waves × 4 pairs = 16 elements

    // For NO_ACTIVATIONS_NOT_FP32_INPUT (BF16 input, no activations):
    IType2 thread_amax_2x = {static_cast<IType>(0.0f), static_cast<IType>(0.0f)};

    #pragma unroll
    for (int w = 0; w < WAVES; ++w) {  // 2 waves
      // Swizzle index to avoid bank conflicts
      const size_t swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM;
      const size_t swizzled_thread_idx = thread_offset_X_rowwise + swizzled_group_idx;
      const size_t shmem_offset_rowwise = shmem_offset_base_rowwise_in + swizzled_thread_idx;

      // Load 8 elements (PACK_SIZE=8) as 4 BF16x2 pairs
      in_IType[w].load_from(&in_sh[shmem_offset_rowwise]);

      // Compute amax on pairs using PTX intrinsic
      #pragma unroll
      for (int e = 0; e < PACK_SIZE / 2; ++e) {  // 4 pairs
        ptx::abs_max_2x(thread_amax_2x, thread_amax_2x, in_IType[w].data.elt[e]);
      }
    }

    // Reduce to scalar amax
    block_amax = static_cast<float>(__hmax(__habs(thread_amax_2x.x), __habs(thread_amax_2x.y)));

    // === SUBSTEP 4.3: Compute E4M3 FP8 scaling factor ===
    const nvfp4_scale_t S_dec_b_fp8 =
        compute_decoding_scaling_factor(block_amax, S_enc_rowwise);

    // === SUBSTEP 4.4: Write scale to global memory ===
    const size_t scales_offset_Y =
        scales_offset_Y_rowwise + stage * BUFF_DIM_Y + it * THREADS_Y_ROWWISE;
    const size_t scale_idx_global = scales_offset_Y * scale_stride + scales_offset_X_rowwise;

    const bool rowwise_scale_is_within_bounds_Y =
        (stage_rowwise_scales_offset_Y + it * THREADS_Y_ROWWISE + tid_Y_rowwise) < chunk_rows;
    if (rowwise_scale_is_within_bounds_X && rowwise_scale_is_within_bounds_Y) {
      scales_ptr[scale_idx_global] = S_dec_b_fp8;  // Direct write to global memory
    }

    // === SUBSTEP 4.5: Compute per-block encoding scale ===
    const float block_scale_inverse =
        fminf(1.0f / (static_cast<float>(S_dec_b_fp8) * S_dec_rowwise), float_max);
    const float2 block_scale_inverse_2x{block_scale_inverse, block_scale_inverse};

    // === SUBSTEP 4.6: Quantize 16 elements to FP4 ===
    #pragma unroll
    for (int w = 0; w < WAVES; ++w) {  // 2 waves
      Vec<fp4e2m1x4, PACK_SIZE / 4> out;  // 8 / 4 = 2 fp4x4 values

      #pragma unroll
      for (int e = 0; e < PACK_SIZE / 4; ++e) {  // 2 iterations
        const uint32_t rbits = get_rbits(rng, random_uint4, rnd_idx);

        // Pack 4 BF16 values into uint64_t
        const uint64_t elts = *reinterpret_cast<uint64_t *>(&in_IType[w].data.elt[2 * e]);

        // PTX intrinsic: mul + cvt fused
        out.data.elt[e] = ptx::mul_cvt_bf16_to_fp4_4x<USE_STOCHASTIC_ROUNDING>(
            elts, block_scale_inverse_2x, rbits);
      }

      // === SUBSTEP 4.7: Write to SHMEM with swizzling ===
      const size_t swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM;
      const size_t swizzled_idx = swizzled_group_idx + thread_offset_X_rowwise;
      const size_t shmem_offset_rowwise = shmem_offset_base_rowwise_out + swizzled_idx / 2;

      out.store_to(&out_data_sh[shmem_offset_rowwise]);
    }
  }
}
```

**Example for Thread 0 (tid_Y=0, tid_X=0), Iteration 0**:
- Processes row 0, columns [0:16)
- Loads 16 BF16 values in 2 waves (8+8) with swizzling
- Computes amax using BF16 arithmetic and PTX intrinsic
- Computes S_dec_b_fp8 and writes to global memory
- Quantizes 16 values to FP4
- Writes 16 FP4 values (8 bytes) to SHMEM

**Example for Thread 0, Iteration 1**:
- Processes row 16, columns [0:16)
- Same process

**Example for Thread 8 (tid_Y=1, tid_X=0), Iteration 0**:
- Processes row 1, columns [0:16)

**Result**: After all 128 threads complete both iterations:
- Identity output has 32 rows × 128 columns (matching BUFF_OUT_DIM_Y × BUFF_OUT_DIM_X bytes)
- Rowwise scaling factors written directly to global memory

### 6.5 Final Scale Writeback

**File**: [transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh:597-614](../transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh#L597-L614)

After all stages complete, columnwise scales need to be written from SHMEM to global memory:

```cpp
// Vectorized store scaling factors through SHMEM
if (RETURN_TRANSPOSE && colwise_scale_is_within_bounds_Y) {
  using ScalesVec = Vec<nvfp4_scale_t, SCALES_PER_CHUNK_Y>;  // 8 scales
  const size_t scale_idx_sh = tid_Y_t * SCALES_PER_CHUNK_Y;

  // Load 8 scales from SHMEM
  ScalesVec &scales_vec = *reinterpret_cast<ScalesVec *>(&out_colwise_scales_sh[scale_idx_sh]);

  const size_t scale_idx_global = scales_offset_Y_t * scale_stride_t + scales_offset_X_t;
  const size_t count = (chunk_rows >= CHUNK_DIM_Y) ? SCALES_PER_CHUNK_Y : (chunk_rows / SCALE_DIM);

  nvfp4_scale_t *dst = &scales_t_ptr[scale_idx_global];

  // Fast path: vectorized store if aligned
  if (count == SCALES_PER_CHUNK_Y && (reinterpret_cast<uintptr_t>(dst) % vec_bytes == 0)) {
    scales_vec.store_to(dst);  // 8 FP8 values = 8 bytes written
  } else {
    // Safe path: element-wise store
    scales_vec.store_to_elts(dst, 0, count);
  }
}
```

**Key Points**:
- Each thread writes 8 columnwise scales (one per column it processed)
- Vectorized write when possible (8 bytes = 64-bit store)
- Fallback to element-wise for tail cases

---

## 7. PTX Intrinsics

### 7.1 Multiply-Convert FP4 (Stochastic Rounding)

**File**: [transformer_engine/common/util/ptx.cuh:476-514](../transformer_engine/common/util/ptx.cuh#L476-L514)

```cpp
__device__ __forceinline__ fp4e2m1x4 mul_cvt_bf16_to_fp4_4x_with_stochastic_rounding(
    const uint64_t in_4x, const float2 scale, const uint32_t rbits) {
  uint16_t out_4x = 0;
  asm volatile(
      "{\n"
      ".reg.b64 v01; \n\t"              // 64-bit register for 2 FP32 values
      ".reg.b64 v23; \n\t"              // 64-bit register for 2 FP32 values
      ".reg.b16 v0_bf16; \n\t"          // 16-bit register for BF16
      ".reg.b16 v1_bf16; \n\t"
      ".reg.b16 v2_bf16; \n\t"
      ".reg.b16 v3_bf16; \n\t"
      ".reg.b32 v0; \n\t"               // 32-bit register for FP32
      ".reg.b32 v1; \n\t"
      ".reg.b32 v2; \n\t"
      ".reg.b32 v3; \n\t"

      // Extract 4 BF16 values from uint64_t
      "mov.b64 {v0_bf16, v1_bf16, v2_bf16, v3_bf16}, %1; \n\t"

      // Convert BF16 → FP32 (4 conversions)
      "cvt.f32.bf16 v0, v0_bf16; \n\t"
      "cvt.f32.bf16 v1, v1_bf16; \n\t"
      "cvt.f32.bf16 v2, v2_bf16; \n\t"
      "cvt.f32.bf16 v3, v3_bf16; \n\t"

      // Pack into 64-bit pairs
      "mov.b64 v01, {v0, v1}; \n\t"
      "mov.b64 v23, {v2, v3}; \n\t"

      // SIMD multiply: mul.f32x2 (2 FP32 muls in parallel)
      "mul.f32x2 v01, v01, %2; \n\t"    // v0 *= scale.x, v1 *= scale.y
      "mul.f32x2 v23, v23, %2; \n\t"    // v2 *= scale.x, v3 *= scale.y

      // Unpack (note shuffled order)
      "mov.b64 {v1, v0}, v01; \n\t"
      "mov.b64 {v3, v2}, v23; \n\t"

      // Convert to FP4 with stochastic rounding
      // cvt.rs.satfinite.e2m1x4.f32: 4 FP32 → 4 FP4 (16 bits total)
      // rs = round stochastically using rbits
      // satfinite = saturate to [-6, 6] and handle infinities
      // Order: v2, v3, v0, v1 (shuffled to match output layout)
      "cvt.rs.satfinite.e2m1x4.f32 %0, {v2, v3, v0, v1}, %3; \n\t"
      "}"
      : "=h"(out_4x)                    // Output: 16-bit value (4 × 4-bit FP4)
      : "l"(in_4x),                     // Input: 64-bit value (4 × 16-bit BF16)
        "l"(reinterpret_cast<const uint64_t &>(scale)),  // scale as float2
        "r"(rbits));                    // Random bits for stochastic rounding

  return *reinterpret_cast<fp4e2m1x4 *>(&out_4x);
}
```

**Breakdown**:
1. **BF16 → FP32**: 4 parallel conversions using `cvt.f32.bf16`
2. **Multiply**: 2 parallel FP32 multiplications using `mul.f32x2`
3. **FP32 → FP4**: 4 parallel conversions with stochastic rounding using `cvt.rs.satfinite.e2m1x4.f32`

**Stochastic Rounding**:
- `rbits` provides 32 random bits
- Each FP4 conversion uses ~8 bits from `rbits`
- Adds noise proportional to quantization step to break ties randomly

**Round-to-Nearest (No Stochastic Rounding)**:

**File**: [transformer_engine/common/util/ptx.cuh:516-560](../transformer_engine/common/util/ptx.cuh#L516-L560)

```cpp
__device__ __forceinline__ fp4e2m1x4 mul_cvt_bf16_to_fp4_4x_with_rn(
    const uint64_t in_4x, const float2 scale, const uint32_t rbits) {
  uint32_t out_4x = 0;
  asm volatile(
      "{\n"
      // ... (same setup as above)

      // Convert to FP4 with round-to-nearest
      // cvt.rn.satfinite.e2m1x2.f32: 2 FP32 → 2 FP4 (8 bits)
      "cvt.rn.satfinite.e2m1x2.f32 f0, v0, v1;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 f1, v2, v3;\n\t"

      // Pack into 32-bit (only using lower 16 bits)
      "mov.b32 %0, {f0, f1, f0, f1};\n\t"
      "}"
      : "=r"(out_4x)
      : "l"(in_4x), "l"(reinterpret_cast<const uint64_t &>(scale)));

  return reinterpret_cast<fp4e2m1x4 *>(&out_4x)[0];
}
```

**Key Difference**:
- `cvt.rn.*` (round-to-nearest) instead of `cvt.rs.*` (round stochastically)
- Deterministic rounding (ties to nearest even)

### 7.2 Absolute Maximum (BF16x2)

**File**: [transformer_engine/common/util/ptx.cuh:807-816](../transformer_engine/common/util/ptx.cuh#L807-L816)

```cpp
__device__ __forceinline__ void abs_max_2x(bf16x2 &dst, const bf16x2 &p1, const bf16x2 &p2) {
  asm volatile("max.xorsign.abs.bf16x2 %0, %1, %2;"
               : "=r"(reinterpret_cast<uint32_t &>(dst))
               : "r"(reinterpret_cast<const uint32_t &>(p1)),
                 "r"(reinterpret_cast<const uint32_t &>(p2)));
}
```

**Explanation**:
- `max.xorsign.abs.bf16x2`: SIMD instruction operating on 2 BF16 values simultaneously
- For each pair: `dst[i] = max(abs(p1[i]), abs(p2[i]))`
- Used for accumulating amax across multiple values

**Usage in Kernel**:
```cpp
IType2 thread_amax_2x = {0.0f, 0.0f};
for (int e = 0; e < PACK_SIZE / 2; ++e) {
  ptx::abs_max_2x(thread_amax_2x, thread_amax_2x, in_IType[w].data.elt[e]);
}
// Final reduction:
block_amax = static_cast<float>(__hmax(__habs(thread_amax_2x.x), __habs(thread_amax_2x.y)));
```

### 7.3 TMA (Tensor Memory Accelerator) Operations

#### Global → Shared (Async Load)

**File**: [transformer_engine/common/util/ptx.cuh:198-216](../transformer_engine/common/util/ptx.cuh#L198-L216)

```cpp
__device__ __forceinline__ void cp_async_bulk_tensor_2d_global_to_shared(
    uint64_t *dst_shmem, const uint64_t *tensor_map_ptr,
    const uint32_t offset_x, const uint32_t offset_y, uint64_t *mbar) {
  uint32_t dst_shmem_ptr = __cvta_generic_to_shared(dst_shmem);
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);

  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
      ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
      ::"r"(dst_shmem_ptr), "l"(tensor_map_ptr),
        "r"(offset_x), "r"(offset_y), "r"(mbar_ptr)
      : "memory");
}
```

**Explanation**:
- **Hardware-accelerated transfer**: TMA engine copies 2D tile from global to shared memory
- **Asynchronous**: Thread continues immediately; barrier tracks completion
- **Mbarrier integration**: Automatically decrements barrier's expected bytes counter
- **Tile-based**: Copies entire BUFF_DIM_Y × BUFF_DIM_X tile (32 × 128 = 4096 BF16 values)

#### Shared → Global (Async Store)

**File**: [transformer_engine/common/util/ptx.cuh:310-322](../transformer_engine/common/util/ptx.cuh#L310-L322)

```cpp
__device__ __forceinline__ void cp_async_bulk_tensor_2d_shared_to_global(
    const uint64_t *tensor_map_ptr, const uint32_t offset_x,
    const uint32_t offset_y, uint64_t *src_shmem) {
  uint32_t src_shmem_ptr = __cvta_generic_to_shared(src_shmem);

  asm volatile(
      "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
      ::"l"(tensor_map_ptr), "r"(offset_x), "r"(offset_y), "r"(src_shmem_ptr)
      : "memory");
}
```

**Explanation**:
- **Hardware-accelerated transfer**: TMA engine copies 2D tile from shared to global memory
- **Bulk group**: Part of async bulk operation group for synchronization

#### Mbarrier Operations

**File**: [transformer_engine/common/util/ptx.cuh:127-244](../transformer_engine/common/util/ptx.cuh#L127-L244)

```cpp
// Initialize barrier
__device__ __forceinline__ void mbarrier_init(uint64_t *mbar, const uint32_t count) {
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  asm volatile("mbarrier.init.shared.b64 [%0], %1;"
               ::"r"(mbar_ptr), "r"(count) : "memory");
}

// Arrive and notify expected bytes
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t *mbar, const uint32_t tx_count) {
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
               ::"r"(mbar_ptr), "r"(tx_count) : "memory");
}

// Wait for barrier with parity
__device__ __forceinline__ void mbarrier_wait_parity(uint64_t *mbar, const uint32_t parity) {
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  while (!mbarrier_try_wait_parity(mbar_ptr, parity)) {}
}
```

**Usage Pattern**:
```cpp
// Master thread: initiate TMA and arrive at barrier
if (is_master_thread) {
  mbarrier_arrive_expect_tx(&mbar[stage], num_bytes);
  cp_async_bulk_tensor_2d_global_to_shared(..., &mbar[stage]);
}

// All threads: wait for TMA to complete
mbarrier_wait_parity(&mbar[stage], 0);
```

### 7.4 Fence Operations

**File**: [transformer_engine/common/util/ptx.cuh:394-400](../transformer_engine/common/util/ptx.cuh#L394-L400)

```cpp
__device__ __forceinline__ void fence_proxy_async_shared_cta() {
  asm volatile("fence.proxy.async.shared::cta;");
}
```

**Explanation**:
- **Memory fence**: Ensures ordering of async operations
- **shared::cta scope**: Applies to shared memory within CTA (cooperative thread array / thread block)
- **Usage before TMA**: Ensures compute writes visible to TMA engine
- **Usage after TMA**: Ensures TMA writes visible to compute threads

---

## 8. Memory Layout

### 8.1 Shared Memory Layout

**Total SHMEM per block**: ~20 KB (exact depends on alignment)

```
┌─────────────────────────────────────────────────────────────────┐
│ Alignment Padding (up to 128 bytes)                            │
├─────────────────────────────────────────────────────────────────┤
│ Input Buffer 0: 32 × 128 BF16 = 8192 bytes                     │
├─────────────────────────────────────────────────────────────────┤
│ Input Buffer 1: 32 × 128 BF16 = 8192 bytes                     │
├─────────────────────────────────────────────────────────────────┤
│ Output Buffer 0: 32 × 64 NVFP4 bytes = 2048 bytes              │
├─────────────────────────────────────────────────────────────────┤
│ Output Buffer 1: 32 × 64 NVFP4 bytes = 2048 bytes              │
├─────────────────────────────────────────────────────────────────┤
│ Output Transpose Buffer 0: 128 × 16 NVFP4 bytes = 2048 bytes   │
├─────────────────────────────────────────────────────────────────┤
│ Output Transpose Buffer 1: 128 × 16 NVFP4 bytes = 2048 bytes   │
├─────────────────────────────────────────────────────────────────┤
│ Columnwise Scales: 128 × FP8 = 128 bytes                       │
├─────────────────────────────────────────────────────────────────┤
│ Mbarriers: 4 × 8 bytes = 32 bytes                              │
└─────────────────────────────────────────────────────────────────┘
```

**Note**: Input buffers can be aliased with output for activation caching.

### 8.2 Input Tensor Layout (Global Memory)

```
Input: 128 rows × 1024 cols (BF16)

Row-major layout:
[row0_col0, row0_col1, ..., row0_col1023,
 row1_col0, row1_col1, ..., row1_col1023,
 ...
 row127_col0, row127_col1, ..., row127_col1023]

Total size: 128 × 1024 × 2 bytes = 262,144 bytes = 256 KB
```

### 8.3 Output Tensor Layout (Identity - Global Memory)

```
Output (Identity): 128 rows × 1024 cols (NVFP4)

Row-major layout:
[row0_col0:col1, row0_col2:col3, ..., row0_col1022:col1023,
 row1_col0:col1, row1_col2:col3, ..., row1_col1022:col1023,
 ...
 row127_col0:col1, row127_col2:col3, ..., row127_col1022:col1023]

Each byte contains 2 NVFP4 values (4 bits each)
Total size: 128 × 1024 × 0.5 bytes = 65,536 bytes = 64 KB
```

### 8.4 Output Tensor Layout (Transposed - Global Memory)

```
Output (Transposed): 1024 rows × 128 cols (NVFP4)

Row-major layout (but transposed relative to input):
[col0_row0:row1, col0_row2:row3, ..., col0_row126:row127,
 col1_row0:row1, col1_row2:row3, ..., col1_row126:row127,
 ...
 col1023_row0:row1, col1023_row2:row3, ..., col1023_row126:row127]

Total size: 1024 × 128 × 0.5 bytes = 65,536 bytes = 64 KB
```

### 8.5 Scaling Factors Layout (Global Memory)

**Rowwise Scales**:
```
Shape: 128 rows × 64 scales (FP8 E4M3)
Layout: [row0_scale0, row0_scale1, ..., row0_scale63,
         row1_scale0, row1_scale1, ..., row1_scale63,
         ...
         row127_scale0, row127_scale1, ..., row127_scale63]

Total size: 128 × 64 × 1 byte = 8,192 bytes = 8 KB
```

**Columnwise Scales (Transposed)**:
```
Shape: 1024 rows × 8 scales (FP8 E4M3)
Layout: [col0_scale0, col0_scale1, ..., col0_scale7,
         col1_scale0, col1_scale1, ..., col1_scale7,
         ...
         col1023_scale0, col1023_scale1, ..., col1023_scale7]

Total size: 1024 × 8 × 1 byte = 8,192 bytes = 8 KB
```

### 8.6 Memory Access Patterns

#### Rowwise Access (Identity Output)

Each thread accesses:
- **Coalesced reads**: 16 consecutive BF16 values in same row
- **Coalesced writes**: 16 consecutive NVFP4 values (8 bytes) in same row

Example for tid_Y=0, tid_X=0:
```
Read:  in_sh[row0, col0:col16)
Write: out_sh[row0, col0:col16)]  (as 8 bytes)
```

#### Columnwise Access (Transposed Output)

Each thread accesses:
- **Strided reads**: 16 BF16 values in same column, different rows
- **Coalesced writes**: 16 consecutive NVFP4 values in transposed row

Example for threadIdx.x=0:
```
Read:  in_sh[row0:row16), col0]  (stride = 128 elements)
Write: out_t_sh[row0, col0:col16)]  (as 8 bytes, transposed)
```

---

## 9. Performance Optimizations

### 9.1 Compute Optimizations

1. **Fused Multiply-Convert**: Single PTX instruction performs `value * scale` and conversion to FP4
2. **SIMD Operations**:
   - `mul.f32x2`: 2 FP32 multiplies in parallel
   - `abs_max_2x`: 2 BF16 comparisons in parallel
3. **Reduced Precision Amax**: Use BF16 arithmetic for amax computation (faster than FP32)
4. **Activation Caching**: Reuse SHMEM buffer for cached activations to avoid recomputation

### 9.2 Memory Optimizations

#### Bank Conflict Avoidance (Rowwise)

**File**: [transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh:418-428](../transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh#L418-L428)

```cpp
const int bank_group = thread_lane / THREADS_PER_BANK;  // 0-3

for (int w = 0; w < WAVES; ++w) {
  // Swizzle index based on bank_group
  const size_t swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM;
  const size_t swizzled_thread_idx = thread_offset_X_rowwise + swizzled_group_idx;
  const size_t shmem_offset_rowwise = shmem_offset_base_rowwise_in + swizzled_thread_idx;

  in_IType[w].load_from(&in_sh[shmem_offset_rowwise]);
}
```

**Explanation**:
- Shared memory has 32 banks (4 bytes each)
- Without swizzling, threads in same warp accessing same 16-element block would cause 4-way conflicts
- Swizzling rotates access pattern by `bank_group`, distributing accesses across banks

**Example** (4 threads in warp, PACK_SIZE=8):
```
Without swizzling:
  Thread 0: accesses banks [0, 1, 2, 3, 4, 5, 6, 7]
  Thread 1: accesses banks [0, 1, 2, 3, 4, 5, 6, 7]  ← 2-way conflict
  Thread 2: accesses banks [0, 1, 2, 3, 4, 5, 6, 7]  ← 2-way conflict
  Thread 3: accesses banks [0, 1, 2, 3, 4, 5, 6, 7]  ← 2-way conflict

With swizzling (bank_group rotates):
  Thread 0 (bank_group=0): accesses banks [0, 1, 2, 3, 4, 5, 6, 7]
  Thread 1 (bank_group=0): accesses banks [8, 9, 10, 11, 12, 13, 14, 15]  ← no conflict
  Thread 8 (bank_group=1): accesses banks [8, 9, 10, 11, 12, 13, 14, 15]  ← no conflict
  Thread 9 (bank_group=1): accesses banks [0, 1, 2, 3, 4, 5, 6, 7]  ← no conflict
```

#### TMA Alignment

**Requirements**:
- Input/output pointers must be 128-byte aligned
- Tile dimensions must be multiples of 16 bytes
- For BF16: requires cols % 8 == 0
- For NVFP4: requires cols % 16 == 0 (since 16 values = 8 bytes)

**Validation**:
```cpp
NVTE_CHECK(rows % 32 == 0, "Number of tensor rows must be a multiple of 32");
NVTE_CHECK(cols % 32 == 0, "Number of tensor cols must be a multiple of 32");
```

### 9.3 Pipeline Optimizations

1. **Double Buffering**: Overlap TMA load of stage N+1 with compute of stage N
2. **TMA Prefetching**: Initiate next load before waiting on current stage
3. **Async Stores**: TMA stores happen concurrently with next stage's loads
4. **Mbarrier Pipelining**: Each stage has independent barrier for fine-grained synchronization

**Effective Timeline** (4 stages):
```
Stage 0: |--TMA Load--|--Compute--|--TMA Store--|
Stage 1:    |--TMA Load--|--Compute--|--TMA Store--|
Stage 2:       |--TMA Load--|--Compute--|--TMA Store--|
Stage 3:          |--TMA Load--|--Compute--|--TMA Store--|
```

Total time ≈ 1 × (TMA Load) + 4 × (Compute + TMA Store)

### 9.4 Quantization Optimizations

1. **Two-Stage Scaling**:
   - Global scale (FP32): Computed once per kernel
   - Block scale (FP8 E4M3): Computed per 16-element block
   - Reduces global memory for scales (FP8 vs FP32 = 4× savings)

2. **Stochastic Rounding**:
   - Uses Philox RNG (fast, parallel)
   - Generates 4 random values per call, reuses until exhausted
   - Adds unbiased noise to quantization errors

3. **Direct Scale Writeback** (Rowwise):
   - Rowwise scales written directly to global memory during compute
   - Avoids extra SHMEM → Global copy

4. **Vectorized Scale Writeback** (Columnwise):
   - 8 FP8 scales written as single 64-bit store when aligned
   - Reduces global memory transactions

---

## Summary for 128 × 1024 Input

### Grid Configuration
- **Grid**: 8 blocks × 1 block
- **Block size**: 128 threads
- **Chunks**: 8 chunks of 128 × 128

### Per-Block Processing
- **4 stages** of 32 × 128 tiles
- **Double buffering** with 2 SHMEM buffers
- **Pipeline overlap**: Load stage N+1 while computing stage N

### Per-Thread Work (Columnwise)
- Process 1 column
- 2 iterations × 16 elements = 32 elements total
- Compute 2 scales (FP8 E4M3)

### Per-Thread Work (Rowwise)
- Process 2 rows
- 1 block of 16 elements per row
- Compute 2 scales (FP8 E4M3)

### Memory Transfers
- **TMA Loads**: 8 blocks × 4 stages × 8192 bytes = 262,144 bytes (256 KB)
- **TMA Stores (Identity)**: 8 blocks × 4 stages × 2048 bytes = 65,536 bytes (64 KB)
- **TMA Stores (Transpose)**: 8 blocks × 4 stages × 2048 bytes = 65,536 bytes (64 KB)
- **Scale Writes**: 16,384 bytes (16 KB)

### Total Compute
- **Amax computations**: 128 × 1024 / 16 = 8,192 blocks
- **Scale computations**: 16,384 FP8 scales (8,192 rowwise + 8,192 columnwise)
- **FP4 conversions**: 131,072 values (128 × 1024)

---

## Conclusion

The `quantize_transpose_nvfp4_kernel` is a highly optimized fusion kernel that:

1. **Quantizes** BF16 → NVFP4 (4-bit) with per-16-element block scaling
2. **Transposes** while quantizing (columnwise path)
3. **Computes scales** in FP8 E4M3 format for compact storage
4. **Pipelines** 4 stages with double buffering for high throughput
5. **Leverages hardware**: TMA, mbarriers, SIMD PTX instructions

**Key innovations**:
- Fused multiply-convert PTX intrinsics
- Manual bank conflict avoidance via swizzling
- Two-stage scaling (global FP32 + block FP8)
- Async TMA with fine-grained mbarrier synchronization
- Stochastic rounding for unbiased quantization

For a 128 × 1024 tensor, the kernel achieves:
- **8× memory compression** (BF16 → NVFP4)
- **High memory bandwidth** via TMA and coalesced accesses
- **Low latency** via pipelining and compute-memory overlap
