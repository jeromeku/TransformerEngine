# HadamardAmaxTmaKernel Comprehensive Walkthrough

**Kernel Location**: [hadamard_transform.cu:355-498](../../../transformer_engine/common/hadamard_transform/hadamard_transform.cu#L355-L498)

## Executive Summary

`HadamardAmaxTmaKernel` is a Blackwell (SM 10.0+) GPU kernel that:
1. Loads input data from global memory using TMA (Tensor Memory Accelerator) with pipelined prefetching
2. Computes Hadamard transforms (both identity and transposed) on 16×16 tiles using tensor cores
3. Computes absolute maximum values (amax) of: pre-RHT input, identity transform, and transposed transform
4. Uses double-buffering and software pipelining to overlap data transfer with computation

## Concrete Example Parameters

**Input shape**: 1024 × 2048 (num_rows × row_length)
**Configuration**: `return_transposed_amax=true`, `return_pre_rht_amax=true`

### Template Parameters (Instantiated)
```cpp
template <
  typename IType = bf16,
  int kHadamardDimension = 16,
  int CHUNK_DIM_Y = 128,        // Block processes 128 rows
  int CHUNK_DIM_X = 128,        // Block processes 128 columns
  int BUFF_DIM_Y = 64,          // TMA loads 64 rows at a time
  int BUFF_DIM_X = 64,          // TMA loads 64 columns at a time
  int THREADS_PER_CHUNK = 128,  // 4 warps × 32 threads
  int THREADS_PER_Y = 1,        // 1 thread in Y dimension
  bool kReturnPreRhtAmax = true,
  bool kReturnIdentityAmax = false,
  bool kReturnTransposedAmax = true
>
```

### Grid and Block Configuration
```cpp
// From hadamard_transform.cu:817-819
dim3 block(128, 1);  // 128 threads (4 warps), 1 in Y
dim3 grid(16, 8);    // 16 blocks in X (2048/128), 8 blocks in Y (1024/128)

// Constants computed:
constexpr int kNumWarps = 4;              // (128 * 1) / 32
constexpr size_t STAGES_Y = 2;            // 128 / 64
constexpr size_t STAGES_X = 2;            // 128 / 64
constexpr int num_stages = 4;             // STAGES_X × STAGES_Y = 2 × 2
```

## Memory Layout and Addressing

### Thread and Warp Indexing
For a thread with `threadIdx.x = tid`:
```
warp_id = tid / 32                        // [0, 1, 2, 3] for our config
local_rank = tid % 32                     // [0..31] within warp
```

### Shared Memory Layout (per block)
```
Base Address: dynamic_shmem (aligned to 128 bytes)

├─ in_sh_0        [offset 0]           : 64×64×2 bytes = 8 KB (ping buffer)
├─ in_sh_1        [offset 8 KB]        : 64×64×2 bytes = 8 KB (pong buffer)
├─ mbar[0..3]     [offset 16 KB]       : 4×8 bytes = 32 bytes (barriers)
├─ max_staging_identity   [offset +32] : 4×4 bytes = 16 bytes (warp reduction)
├─ max_staging_transpose  [offset +48] : 4×4 bytes = 16 bytes (warp reduction)
└─ max_staging_pre_rht    [offset +64] : 4×4 bytes = 16 bytes (warp reduction)

Total: ~16 KB + 80 bytes + 128 byte alignment padding
```

### Block to Data Mapping
For `blockIdx.x = 0, blockIdx.y = 0`:
```
input_block_offset_X = 0 × 128 = 0       // Starting column
input_block_offset_Y = 0 × 128 = 0       // Starting row
→ This block processes input[0:128, 0:128]
```

For `blockIdx.x = 1, blockIdx.y = 2`:
```
input_block_offset_X = 1 × 128 = 128     // Starting column
input_block_offset_Y = 2 × 128 = 256     // Starting row
→ This block processes input[256:384, 128:256]
```

## Line-by-Line Kernel Execution

### Initialization Phase (Lines 355-422)

#### Line 361-372: Compute static configuration
```cpp
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  static_assert(CHUNK_DIM_Y >= BUFF_DIM_Y && CHUNK_DIM_Y % BUFF_DIM_Y == 0);
  static_assert(CHUNK_DIM_X >= BUFF_DIM_X && CHUNK_DIM_X % BUFF_DIM_X == 0);

  constexpr size_t STAGES_Y = CHUNK_DIM_Y / BUFF_DIM_Y;  // = 2
  constexpr size_t STAGES_X = CHUNK_DIM_X / BUFF_DIM_X;  // = 2

  constexpr int kNumWarps = (THREADS_PER_CHUNK * THREADS_PER_Y) / kThreadsPerWarp; // = 4

  const int input_block_offset_Y = blockIdx.y * CHUNK_DIM_Y;  // e.g., 0 for block(0,0)
  const int input_block_offset_X = blockIdx.x * CHUNK_DIM_X;  // e.g., 0 for block(0,0)
```

**Thread state**: All 128 threads execute this in parallel.

#### Lines 374-403: Allocate and initialize shared memory
```cpp
  extern __shared__ __align__(128) char dynamic_shmem[];
  uintptr_t base_shmem_ptr = reinterpret_cast<uintptr_t>(dynamic_shmem);
  // Manual 128-byte alignment
  uint8_t* dshmem = reinterpret_cast<uint8_t*>((base_shmem_ptr + 127) & ~127ULL);
```

**Thread state**: Each thread computes the same aligned pointer.

```cpp
  constexpr size_t in_buff_size = BUFF_DIM_X * BUFF_DIM_Y * sizeof(IType);  // = 8192 bytes
  IType* in_sh_0 = reinterpret_cast<IType*>(dshmem);
  dshmem += in_buff_size;
  IType* in_sh_1 = reinterpret_cast<IType*>(dshmem);
  dshmem += in_buff_size;

  IType* in_shs[2] = {in_sh_0, in_sh_1};  // Ping-pong buffer pointers
```

**Shared memory pointers**:
- `in_sh_0`: Address 0x... (ping buffer for stage 0, 2, ...)
- `in_sh_1`: Address 0x...+8KB (pong buffer for stage 1, 3, ...)

```cpp
  const bool is_master_thread = (threadIdx.x == 0 && threadIdx.y == 0);

  uint64_t* mbar = reinterpret_cast<uint64_t*>(dshmem);
  dshmem += sizeof(uint64_t) * (STAGES_X * STAGES_Y);  // 4 barriers

  float* max_staging_identity = reinterpret_cast<float*>(dshmem);
  dshmem += sizeof(float) * kNumWarps;
  float* max_staging_transpose = reinterpret_cast<float*>(dshmem);
  dshmem += sizeof(float) * kNumWarps;
  float* max_staging_pre_rht = reinterpret_cast<float*>(dshmem);
  dshmem += sizeof(float) * kNumWarps;
```

**Thread state**:
- **Thread 0**: `is_master_thread = true`
- **Threads 1-127**: `is_master_thread = false`

#### Lines 405-410: Initialize barriers and issue first TMA load
```cpp
  initialize_barriers<STAGES_X * STAGES_Y, THREADS_PER_CHUNK * THREADS_PER_Y>(mbar, is_master_thread);
```

**What happens inside `initialize_barriers`**:
- **Thread 0 only**: Initializes `mbar[0]`, `mbar[1]`, `mbar[2]`, `mbar[3]` with arrival count = 128
- **All threads**: Hit `__syncthreads()` barrier

**Barrier state after initialization**:
```
mbar[0]: phase=0, count=128, expected_bytes=0
mbar[1]: phase=0, count=128, expected_bytes=0
mbar[2]: phase=0, count=128, expected_bytes=0
mbar[3]: phase=0, count=128, expected_bytes=0
```

```cpp
  copy_2d_to_shared(in_shs[0], reinterpret_cast<const void*>(&tensor_map_input),
                    input_block_offset_X, input_block_offset_Y, shmem_buff_size, &mbar[0],
                    is_master_thread);
```

**What happens inside `copy_2d_to_shared`**:
- **Thread 0 only**:
  - Issues TMA instruction: `cp.async.bulk.tensor.2d.shared::cta.global[in_sh_0], [tensor_map + (0, 0)]`
    - This loads input[0:64, 0:64] → shared memory in_sh_0
  - Updates barrier: `mbarrier_arrive_expect_tx(&mbar[0], 8192)` - expects 8192 bytes
- **Threads 1-127**: Call `mbarrier_arrive(&mbar[0])` (decrement arrival count)

**Barrier state after first TMA**:
```
mbar[0]: phase=0, count=0, expected_bytes=8192, TMA in-flight
```

**Memory state**:
- `in_sh_0`: TMA loading input[0:64, 0:64] from global memory (asynchronous)
- `in_sh_1`: Uninitialized

#### Lines 412-422: Initialize Hadamard matrix fragments and amax registers
```cpp
  uint32_t had_frag_i[4];
  uint32_t had_frag_t[4];
  get_hadamard_matrix_fragment<kReturnIdentityAmax, kReturnTransposedAmax, false, false>(
      had_frag_i, random_sign_mask, had_frag_t, random_sign_mask_t);
```

**Per-thread execution**: Each thread computes its portion of the 16×16 Hadamard matrices.

**Thread 0 (warp 0, lane 0)**: Computes Hadamard coefficients for specific rows/columns
- `had_frag_i[0..3]`: Each holds 2×bf16 values (4 bf16 total) for identity transform
- `had_frag_t[0..3]`: Each holds 2×bf16 values (4 bf16 total) for transposed transform

**Hadamard Fragment Details**:
For a 16×16 Hadamard matrix, each thread holds fragments that, when combined across the warp, form the complete matrix needed for the MMA operation. The fragments are stored in a format compatible with tensor core operations.

```cpp
  float local_pre_rht_amax = 0.0;
  float local_amax = 0.0;
  float local_amax_t = 0.0;
  uint32_t local_pre_rht_amax_reg = *reinterpret_cast<uint32_t*>(&local_pre_rht_amax);
  uint32_t local_amax_reg = *reinterpret_cast<uint32_t*>(&local_amax);
  uint32_t local_amax_t_reg = *reinterpret_cast<uint32_t*>(&local_amax_t);
```

**Register state for each thread**:
```
local_pre_rht_amax_reg = 0x00000000 (float 0.0 as uint32)
local_amax_reg = 0x00000000
local_amax_t_reg = 0x00000000
```

These will accumulate maximum absolute values as we process tiles.

### Main Computation Loop (Lines 424-476)

The kernel processes the 128×128 block in 4 stages (2×2 grid of 64×64 buffers).

#### Stage 0: Load input[0:64, 0:64], Process input[0:64, 0:64]

**Entry to loop** (`stage_y=0, stage_x=0`):
```cpp
for (int stage_y = 0; stage_y < STAGES_Y; ++stage_y) {        // stage_y = 0
  for (int stage_x = 0; stage_x < STAGES_X; ++stage_x) {      // stage_x = 0
    int stage = STAGES_X * stage_y + stage_x;                 // stage = 0

    const int next_stage = stage + 1;                         // next_stage = 1
    const int next_stage_x = stage_x + 1 == STAGES_X ? 0 : stage_x + 1;  // = 1
    const int next_stage_y = stage_x + 1 == STAGES_X ? stage_y + 1 : stage_y;  // = 0
```

**Thread state**: All 128 threads computing same values.

##### Lines 432-440: Prefetch next stage
```cpp
    if (next_stage < STAGES_X * STAGES_Y) {                   // 1 < 4, true
      const int input_global_offset_Y = input_block_offset_Y + next_stage_y * BUFF_DIM_Y;
                                                              // = 0 + 0*64 = 0
      const int input_global_offset_X = input_block_offset_X + next_stage_x * BUFF_DIM_X;
                                                              // = 0 + 1*64 = 64

      copy_2d_to_shared(in_shs[next_stage % 2],              // in_sh_1 (stage 1 % 2)
                        reinterpret_cast<const void*>(&tensor_map_input),
                        input_global_offset_X,                // 64
                        input_global_offset_Y,                // 0
                        shmem_buff_size, &mbar[next_stage],   // &mbar[1]
                        is_master_thread);
    }
```

**TMA operation initiated**:
- **Thread 0**: Issues TMA load of input[0:64, 64:128] → in_sh_1
- **All threads**: Arrive at `mbar[1]`

**Memory state**:
- `in_sh_0`: Data input[0:64, 0:64] (TMA complete or completing)
- `in_sh_1`: TMA loading input[0:64, 64:128] (asynchronous)

**Barrier state**:
```
mbar[0]: TMA complete or nearly complete
mbar[1]: count=0, expected_bytes=8192, TMA in-flight for input[0:64, 64:128]
```

##### Lines 442-445: Wait for current stage data
```cpp
    ptx::fence_proxy_async_shared_cta();

    // Wait for the data to have arrived
    ptx::mbarrier_wait_parity(&mbar[stage], 0);              // Wait on mbar[0], phase 0
```

**Thread behavior**:
- **All 128 threads**: Block until TMA completes for `in_sh_0`
- Once complete, all threads can safely read from `in_sh_0`

**Memory state after wait**:
- `in_sh_0`: Contains input[0:64, 0:64] - READY for computation
- `in_sh_1`: Still loading input[0:64, 64:128]

##### Lines 447-453: Compute stage dimensions
```cpp
    const size_t compute_stage_x_num =
        BUFF_DIM_X / (kHadamardDimension * (THREADS_PER_CHUNK / kThreadsPerWarp));
        // = 64 / (16 * 4) = 1

    const size_t compute_stage_y_num = BUFF_DIM_Y / (kHadamardDimension * THREADS_PER_Y);
        // = 64 / (16 * 1) = 4

    const size_t in_row_stride = BUFF_DIM_X;                 // = 64

    IType* in_sh_ptr = in_shs[stage % 2];                    // in_sh_0
```

**Interpretation**: The 64×64 buffer is divided into:
- **Y direction**: 4 chunks (each 16 rows)
- **X direction**: 1 chunk (64 columns, but processed by 4 warps simultaneously)

##### Lines 456-474: Compute loop over 64×64 buffer

###### compute_stage_y = 0 (rows 0-15)

```cpp
for (size_t compute_stage_y = 0; compute_stage_y < compute_stage_y_num; compute_stage_y++) {
                                                              // compute_stage_y = 0
  const int row_idx_offset = (compute_stage_y * kHadamardDimension * THREADS_PER_Y +
                              threadIdx.y * kHadamardDimension);
                                                              // = 0*16*1 + 0*16 = 0
  const int in_row_offset = row_idx_offset * in_row_stride;  // = 0 * 64 = 0
```

**Thread-level state**:
- All threads in block process rows [0:16] of the 64×64 buffer

```cpp
  for (size_t compute_stage_x = 0; compute_stage_x < compute_stage_x_num; compute_stage_x++) {
                                                              // compute_stage_x = 0
    ComputeKernel<IType, kHadamardDimension, BUFF_DIM_Y, BUFF_DIM_X, kReturnPreRhtAmax,
                  kReturnIdentityAmax, kReturnTransposedAmax>(
        had_frag_i, had_frag_t,
        in_sh_ptr + in_row_offset +
            (compute_stage_x * kHadamardDimension * (THREADS_PER_CHUNK / kThreadsPerWarp)),
                                                              // in_sh_0 + 0 + 0*16*4 = in_sh_0
        local_pre_rht_amax_reg, local_amax_reg, local_amax_t_reg);
  }
```

**Memory address**: `in_sh_ptr = in_sh_0[0:16, 0:64]` (pointer to start)

#### Deep Dive: ComputeKernel Execution

Let me trace through `ComputeKernel` for compute_stage_y=0, compute_stage_x=0:

**Input**: `in_sh_ptr` points to in_sh_0[0:16, 0:64] containing input[0:16, 0:64]

##### Per-Warp Addressing (Lines 208-213)

```cpp
int warp_id = threadIdx.x / kThreadsPerWarp;
int local_rank = (threadIdx.x % kThreadsPerWarp);

int ld_row_idx = local_rank % kHadamardDimension;           // [0..15]
int ld_col_idx = local_rank / kHadamardDimension + warp_id * 2;
int swizzle_idx = swizzle_128B_atom_32B(ld_row_idx, ld_col_idx);
```

**Warp 0 thread mapping** (threadIdx.x = 0..31):
| Thread | warp_id | local_rank | ld_row_idx | ld_col_idx | Purpose |
|--------|---------|------------|------------|------------|---------|
| 0 | 0 | 0 | 0 | 0 | Load row 0, col 0-1 |
| 1 | 0 | 1 | 1 | 0 | Load row 1, col 0-1 |
| ... | 0 | ... | ... | 0 | ... |
| 15 | 0 | 15 | 15 | 0 | Load row 15, col 0-1 |
| 16 | 0 | 16 | 0 | 1 | Load row 0, col 2-3 |
| 17 | 0 | 17 | 1 | 1 | Load row 1, col 2-3 |
| ... | 0 | ... | ... | 1 | ... |
| 31 | 0 | 31 | 15 | 1 | Load row 15, col 2-3 |

**Warp 1 thread mapping** (threadIdx.x = 32..63):
| Thread | warp_id | local_rank | ld_row_idx | ld_col_idx | Purpose |
|--------|---------|------------|------------|------------|---------|
| 32 | 1 | 0 | 0 | 2 | Load row 0, col 4-5 |
| 33 | 1 | 1 | 1 | 2 | Load row 1, col 4-5 |
| ... | 1 | ... | ... | 2 | ... |

**Warp-level data distribution**: Each warp processes a different 16×4 column slice:
- **Warp 0**: columns [0:4]
- **Warp 1**: columns [4:8]
- **Warp 2**: columns [8:12]
- **Warp 3**: columns [12:16]

But wait - we have 64 columns in the buffer! Let me reconsider...

Actually, looking at the loop structure, the 4 warps process one 16×16 tile together. The 64 columns are processed by iterating through 4 separate 16×16 tiles.

Let me recalculate for the full 64 columns:

The buffer is 16 rows × 64 columns. This is processed as:
- **4 separate 16×16 tiles** (horizontally adjacent)
- Each 16×16 tile is processed by **all 4 warps working together**

But looking at `compute_stage_x_num = 1`, we only iterate once. This means the 4 warps together process multiple 16×16 tiles.

Let me re-examine the warp assignment more carefully...

Actually, with `ld_col_idx = local_rank / 16 + warp_id * 2`, we have:
- **Warp 0**: col_idx ∈ {0, 1}
- **Warp 1**: col_idx ∈ {2, 3}
- **Warp 2**: col_idx ∈ {4, 5}
- **Warp 3**: col_idx ∈ {6, 7}

And `ld_col_idx` indexes into 32-byte chunks (16 bf16 values). So:
- **Warp 0**: columns [0:16]
- **Warp 1**: columns [16:32]
- **Warp 2**: columns [32:48]
- **Warp 3**: columns [48:64]

**Each warp processes a separate 16×16 tile!**

##### Transposed Amax Computation (Lines 230-249)

Since `kReturnTransposedAmax = true`:

```cpp
if (kReturnTransposedAmax) {
  // Load data using ldmatrix instruction
  if (!kReturnIdentityAmax) {
    ldmatrix_x4_m8n8_shared_b16<false>(a_frag[0], a_frag[1], a_frag[2], a_frag[3],
                                       reinterpret_cast<uint4*>(in_sh_ptr) + swizzle_idx);
  }
```

**Warp 0, Thread 0** executes:
- `swizzle_idx = swizzle_128B_atom_32B(0, 0) = ...` (computes bank-conflict-free index)
- **ldmatrix** loads 4×8×8 = 256 bytes from shared memory into registers
- `a_frag[0..3]` now contains the thread's portion of a 16×16 input tile

**Data loaded by Warp 0**:
- Loads input[0:16, 0:16] in a swizzled, tensor-core-friendly layout
- Each thread gets a fragment (8 bf16 values spread across 4 registers)

```cpp
  matrix_transpose_m8_n8_b16_inplace(a_frag[0]);
  matrix_transpose_m8_n8_b16_inplace(a_frag[1]);
  matrix_transpose_m8_n8_b16_inplace(a_frag[2]);
  matrix_transpose_m8_n8_b16_inplace(a_frag[3]);
```

**In-register transpose**: Each thread transposes its 8×8 fragment in-place using the `movmatrix` PTX instruction.

```cpp
  mma_m16_n16_k16_b16_b16_b16_noacc<kReturnTransposedAmax>(
      a_frag[0], a_frag[2], a_frag[1], a_frag[3], b_frag_t[0], b_frag_t[1], b_frag_t[2],
      b_frag_t[3], c_frag[0], c_frag[1], c_frag[2], c_frag[3], temp_amax_t_reg);
```

**Tensor core MMA operation** (entire warp participates):
- Computes: `C = A.T @ Hadamard_T` where:
  - `A.T` is the transposed input (16×16)
  - `Hadamard_T` is the transposed Hadamard matrix (16×16)
- Result `c_frag` contains RHT-transformed data
- `temp_amax_t_reg` contains max(abs(c_frag)) as a packed bf16x2

**Warp 0, Thread 0 register state after MMA**:
```
c_frag[0] = <bf16x2 containing 2 output values>
c_frag[1] = <bf16x2 containing 2 output values>
c_frag[2] = <bf16x2 containing 2 output values>
c_frag[3] = <bf16x2 containing 2 output values>
temp_amax_t_reg = <bf16x2 packed max of the above 4 registers>
```

```cpp
  asm volatile("max.xorsign.abs.bf16x2 %0, %1, %2;\n\t"
               : "=r"(local_amax_t_reg)
               : "r"(local_amax_t_reg), "r"(temp_amax_t_reg));
}
```

**Accumulate max**: Takes element-wise max of current `local_amax_t_reg` and new `temp_amax_t_reg`.

**Thread 0 state**:
- `local_amax_t_reg` now contains max(abs(all RHT output values seen so far))

##### Pre-RHT Amax Computation (Lines 251-269)

Since `kReturnPreRhtAmax = true`:

```cpp
if (kReturnPreRhtAmax) {
  if (!kReturnIdentityAmax && !kReturnTransposedAmax) {
    ldmatrix_x4_m8n8_shared_b16<false>(a_frag[0], a_frag[1], a_frag[2], a_frag[3],
                                       reinterpret_cast<uint4*>(in_sh_ptr) + swizzle_idx);
  }
```

**Data reuse**: Since we already loaded data for transposed amax, we skip the load.

```cpp
  asm volatile("max.xorsign.abs.bf16x2 %0, %1, %2;\n\t"
               : "=r"(a_frag[0])
               : "r"(a_frag[0]), "r"(a_frag[1]));
  asm volatile("max.xorsign.abs.bf16x2 %0, %1, %2;\n\t"
               : "=r"(a_frag[2])
               : "r"(a_frag[2]), "r"(a_frag[3]));
  asm volatile("max.xorsign.abs.bf16x2 %0, %1, %2;\n\t"
               : "=r"(a_frag[0])
               : "r"(a_frag[0]), "r"(a_frag[2]));
  asm volatile("max.xorsign.abs.bf16x2 %0, %1, %2;\n\t"
               : "=r"(local_pre_rht_amax_reg)
               : "r"(a_frag[0]), "r"(local_pre_rht_amax_reg));
}
```

**Reduction tree**: Computes max(abs(a_frag[0..3])) in log2(4)=2 steps, then accumulates into `local_pre_rht_amax_reg`.

**Note**: `a_frag` contains the **original input data** (before transpose in this case, but after ldmatrix swizzling). The pre-RHT amax is computed from raw input.

##### End of ComputeKernel for compute_stage_y=0, compute_stage_x=0

**Thread-level register state** (example for Warp 0, Thread 0):
- `local_pre_rht_amax_reg`: max(abs(input[0:16, 0:16]))
- `local_amax_t_reg`: max(abs(RHT_transposed(input[0:16, 0:16])))

**Warp 1, 2, 3**: Process their respective 16×16 tiles independently.

##### Back to Main Loop: compute_stage_y = 1, 2, 3

```cpp
  // Ensure all threads have finished their computation before new data over-writes the shared
  // memory.
  __syncthreads();
}
```

**Synchronization**: All threads wait before proceeding to next `compute_stage_y`.

The loop continues for `compute_stage_y = 1, 2, 3`, processing rows [16:32], [32:48], [48:64] respectively.

**After compute_stage_y loop completes**:
- All 4 warps have processed all 4 horizontal strips of the 64×64 buffer
- Each warp has accumulated amax values across 4 vertical 16×16 tiles

**Thread-level state** (Warp 0, Thread 0):
- `local_pre_rht_amax_reg`: max(abs(input[0:64, 0:16]))
- `local_amax_t_reg`: max(abs(RHT_transposed(input[0:64, 0:16])))

**Memory state** after Stage 0:
- `in_sh_0`: Still contains input[0:64, 0:64]
- `in_sh_1`: Contains input[0:64, 64:128] (TMA completed during compute)

#### Stage 1: Load input[64:128, 0:64], Process input[0:64, 64:128]

```cpp
for (int stage_y = 0; stage_y < STAGES_Y; ++stage_y) {        // stage_y = 0
  for (int stage_x = 0; stage_x < STAGES_X; ++stage_x) {      // stage_x = 1 (second iteration)
    int stage = STAGES_X * stage_y + stage_x;                 // stage = 1

    const int next_stage = stage + 1;                         // next_stage = 2
    const int next_stage_x = stage_x + 1 == STAGES_X ? 0 : stage_x + 1;  // = 0
    const int next_stage_y = stage_x + 1 == STAGES_X ? stage_y + 1 : stage_y;  // = 1
```

##### Prefetch Stage 2

```cpp
    if (next_stage < STAGES_X * STAGES_Y) {                   // 2 < 4, true
      const int input_global_offset_Y = input_block_offset_Y + next_stage_y * BUFF_DIM_Y;
                                                              // = 0 + 1*64 = 64
      const int input_global_offset_X = input_block_offset_X + next_stage_x * BUFF_DIM_X;
                                                              // = 0 + 0*64 = 0

      copy_2d_to_shared(in_shs[next_stage % 2],              // in_sh_0 (stage 2 % 2)
                        reinterpret_cast<const void*>(&tensor_map_input),
                        input_global_offset_X,                // 0
                        input_global_offset_Y,                // 64
                        shmem_buff_size, &mbar[next_stage],   // &mbar[2]
                        is_master_thread);
    }
```

**TMA operation**: Load input[64:128, 0:64] → in_sh_0 (overwrites old data)

**Buffer state**:
- `in_sh_0`: TMA loading input[64:128, 0:64] (will overwrite previous data)
- `in_sh_1`: Contains input[0:64, 64:128] - ready for processing

##### Wait and Compute Stage 1

```cpp
    ptx::mbarrier_wait_parity(&mbar[stage], 0);              // Wait on mbar[1]

    IType* in_sh_ptr = in_shs[stage % 2];                    // in_sh_1
```

**Processing**: in_sh_1 contains input[0:64, 64:128]
- Warps process 16×16 tiles in columns [0:16], [16:32], [32:48], [48:64] of this buffer
- These correspond to **global columns [64:80], [80:96], [96:112], [112:128]**
- `local_amax_t_reg` continues accumulating max values

**Thread state after Stage 1**:
- `local_pre_rht_amax_reg`: max(abs(input[0:64, 0:64]))
- `local_amax_t_reg`: max(abs(RHT_transposed(input[0:64, 0:128])))

#### Stage 2: Load input[64:128, 64:128], Process input[64:128, 0:64]

```cpp
    // stage_y = 1, stage_x = 0
    int stage = STAGES_X * stage_y + stage_x;                 // stage = 2
    const int next_stage = stage + 1;                         // next_stage = 3
```

##### Prefetch Stage 3

```cpp
      const int input_global_offset_Y = 0 + 1*64 = 64
      const int input_global_offset_X = 0 + 1*64 = 64

      copy_2d_to_shared(in_shs[3 % 2],                       // in_sh_1
                        ..., 64, 64, ..., &mbar[3], ...);
```

**TMA operation**: Load input[64:128, 64:128] → in_sh_1

##### Process Stage 2

```cpp
    ptx::mbarrier_wait_parity(&mbar[2], 0);
    IType* in_sh_ptr = in_shs[2 % 2];                        // in_sh_0
```

**Processing**: in_sh_0 contains input[64:128, 0:64]
- Global rows [64:128], global columns [0:64]

**Thread state after Stage 2**:
- `local_pre_rht_amax_reg`: max(abs(input[0:128, 0:64]))
- `local_amax_t_reg`: max(abs(RHT_transposed(input[0:128, 0:128])))

Wait, that's not right. Let me reconsider the Y dimension.

Actually, looking back at the memory layout:
- Each stage loads and processes a 64×64 tile
- The block is responsible for 128×128 total

Let me reconsider Stage 0:
- Loads input[0:64, 0:64]
- Processes this 64×64 tile completely (4 compute_stage_y iterations)

Actually, I need to reconsider the row iteration. Looking at the code:

```cpp
const int row_idx_offset = (compute_stage_y * kHadamardDimension * THREADS_PER_Y +
                            threadIdx.y * kHadamardDimension);
```

With `THREADS_PER_Y = 1` and `threadIdx.y = 0`:
- compute_stage_y = 0: row_idx_offset = 0 (rows [0:16])
- compute_stage_y = 1: row_idx_offset = 16 (rows [16:32])
- compute_stage_y = 2: row_idx_offset = 32 (rows [32:48])
- compute_stage_y = 3: row_idx_offset = 48 (rows [48:64])

So yes, each stage processes the full 64×64 buffer vertically.

**Corrected understanding**:

- **Stage 0**: Process input[block_offset_Y+0:block_offset_Y+64, block_offset_X+0:block_offset_X+64]
  - For block(0,0): input[0:64, 0:64]
- **Stage 1**: Process input[block_offset_Y+0:block_offset_Y+64, block_offset_X+64:block_offset_X+128]
  - For block(0,0): input[0:64, 64:128]
- **Stage 2**: Process input[block_offset_Y+64:block_offset_Y+128, block_offset_X+0:block_offset_X+64]
  - For block(0,0): input[64:128, 0:64]
- **Stage 3**: Process input[block_offset_Y+64:block_offset_Y+128, block_offset_X+64:block_offset_X+128]
  - For block(0,0): input[64:128, 64:128]

**After all 4 stages complete**:

**Thread-level state** (each thread):
- `local_pre_rht_amax_reg`: max(abs(input[0:128, <thread's columns>]))
- `local_amax_t_reg`: max(abs(RHT_transposed(input[0:128, <thread's columns>])))

#### Stage 3: Process input[64:128, 64:128]

```cpp
    // stage_y = 1, stage_x = 1
    int stage = 3
    const int next_stage = 4

    if (next_stage < STAGES_X * STAGES_Y) {                   // 4 < 4, false
      // No prefetch
    }

    ptx::mbarrier_wait_parity(&mbar[3], 0);
    IType* in_sh_ptr = in_shs[3 % 2];                        // in_sh_1
```

**Processing**: in_sh_1 contains input[64:128, 64:128]

**After Stage 3 completes**:

**Thread-level state** (Warp 0, Thread 0 example):
- `local_pre_rht_amax_reg`: max(abs(input[0:128, 0:16])) - warp's column range
- `local_amax_t_reg`: max(abs(RHT_transposed(input[0:128, 0:16])))

### Reduction Phase (Lines 478-493)

```cpp
  const int warpid = (threadIdx.x + threadIdx.y * blockDim.x) / kThreadsPerWarp;

  if constexpr (kReturnPreRhtAmax) {
    unpack_max_of_packed_bf16(local_pre_rht_amax_reg, local_pre_rht_amax);
  }
  if constexpr (kReturnIdentityAmax) {
    unpack_max_of_packed_bf16(local_amax_reg, local_amax);
  }
  if constexpr (kReturnTransposedAmax) {
    unpack_max_of_packed_bf16(local_amax_t_reg, local_amax_t);
  }
```

**Warp ID computation**:
- Thread 0: warpid = 0
- Thread 32: warpid = 1
- Thread 64: warpid = 2
- Thread 96: warpid = 3

**Unpack packed bf16x2**: Convert from packed bf16x2 format to float.

**Example for Thread 0**:
- `local_pre_rht_amax` = `max(abs(input[0:128, 0:16]))` as float
- `local_amax_t` = `max(abs(RHT_transposed(input[0:128, 0:16])))` as float

```cpp
  ReduceMax<kNumWarps, kReturnPreRhtAmax, kReturnIdentityAmax, kReturnTransposedAmax>(
      local_pre_rht_amax, local_amax, local_amax_t, max_staging_pre_rht, max_staging_identity,
      max_staging_transpose, output_pre_rht_amax_ptr, output_identity_amax_ptr,
      output_transpose_amax_ptr, warpid);
```

#### Deep Dive: ReduceMax

**Input state** (each thread has):
- `pre_rht_amax`: Thread's local max of pre-RHT data
- `transpose_amax`: Thread's local max of transposed RHT output

##### Intra-warp reduction (Lines 290-293)

```cpp
  constexpr int kWarpSize = 32;
  int local_rank = threadIdx.x % 32;
  float warp_pre_rht_amax = kReturnPreRhtAmax ? warp_reduce_max<kWarpSize>(pre_rht_amax) : 0.0f;
  float warp_transpose_amax =
      kReturnTransposedAmax ? warp_reduce_max<kWarpSize>(transpose_amax) : 0.0f;
```

**Warp 0 reduction**:
- Input: Each of 32 threads has a float max value
- `warp_reduce_max`: Uses shuffle instructions to reduce across warp
  - Tree reduction: 16 pairs → 8 → 4 → 2 → 1
  - Final result in Lane 0

**After warp reduction** (Warp 0, Lane 0):
- `warp_pre_rht_amax`: max(abs(input[0:128, 0:16])) - entire warp's column range
- `warp_transpose_amax`: max(abs(RHT_transposed(input[0:128, 0:16])))

##### Inter-warp reduction (Lines 296-306)

```cpp
  if (threadIdx.x % 32 == 0) {
    if (kReturnPreRhtAmax) {
      staging_for_pre_rht[warpid] = warp_pre_rht_amax;
    }
    if (kReturnTransposedAmax) {
      staging_for_transpose[warpid] = warp_transpose_amax;
    }
  }
  __syncthreads();
```

**Lane 0 of each warp writes to shared memory**:
```
max_staging_pre_rht[0] = max(abs(input[0:128, 0:16]))
max_staging_pre_rht[1] = max(abs(input[0:128, 16:32]))
max_staging_pre_rht[2] = max(abs(input[0:128, 32:48]))
max_staging_pre_rht[3] = max(abs(input[0:128, 48:64]))

max_staging_transpose[0] = max(abs(RHT_t(input[0:128, 0:16])))
max_staging_transpose[1] = max(abs(RHT_t(input[0:128, 16:32])))
max_staging_transpose[2] = max(abs(RHT_t(input[0:128, 32:48])))
max_staging_transpose[3] = max(abs(RHT_t(input[0:128, 48:64])))
```

Actually wait - I need to reconsider which columns each warp processes. Let me trace back:

In `ComputeKernel`, with `ld_col_idx = local_rank / 16 + warp_id * 2`:
- Warp 0: ld_col_idx ∈ {0, 1} → loads from swizzled locations

The swizzle pattern and ldmatrix layout is complex. The key insight is:
- Each warp processes **one 16×16 tile** per `ComputeKernel` call
- With 4 warps and 64 columns, we have **4 warps × 16 columns = 64 columns**

So each warp processes every 4th tile horizontally as we iterate through stages.

Actually, looking more carefully: with `compute_stage_x_num = 1`, we only have one iteration in the X compute loop. So during each stage:
- All 4 warps work on the same 64 columns
- But each warp processes a different 16-column slice

Let me reconsider: `ld_col_idx` determines which **uint4** (16 bytes = 8 bf16) to load. With 64 bf16 columns:
- We need to load 64 × 2 = 128 bytes per row
- ldmatrix loads 128 bytes (16×16 bf16 tile)

So each warp loads one 16×16 tile. With 4 warps:
- 4 × 16 = 64 columns covered

**Corrected warp assignments per stage**:
- **Warp 0**: columns [base+0:base+16]
- **Warp 1**: columns [base+16:base+32]
- **Warp 2**: columns [base+32:base+48]
- **Warp 3**: columns [base+48:base+64]

Where `base` depends on the stage.

**After 4 stages**:
- **Warp 0**: processed columns [0:16, 64:80] and rows [0:64] + columns [0:16, 64:80] and rows [64:128]
  - Wait, this doesn't match the 128×128 grid...

I think I'm overcomplicating this. Let me think more carefully:

Each stage loads a 64×64 tile. For that tile:
- 4 warps each process 16 columns × 64 rows

Across 4 stages (2×2 spatial grid):
- Stage 0: rows [0:64], cols [0:64] → Warp i processes cols [i*16:(i+1)*16]
- Stage 1: rows [0:64], cols [64:128] → Warp i processes cols [64+i*16:64+(i+1)*16]
- Stage 2: rows [64:128], cols [0:64] → Warp i processes cols [i*16:(i+1)*16]
- Stage 3: rows [64:128], cols [64:128] → Warp i processes cols [64+i*16:64+(i+1)*16]

**After all stages, Warp 0 has processed**:
- rows [0:64], cols [0:16] (Stage 0)
- rows [0:64], cols [64:80] (Stage 1)
- rows [64:128], cols [0:16] (Stage 2)
- rows [64:128], cols [64:80] (Stage 3)

So Warp 0 has processed a **strided pattern** across the full 128×128 block.

**Shared memory after inter-warp write**:
```
max_staging_pre_rht[0] = max across Warp 0's tiles
max_staging_pre_rht[1] = max across Warp 1's tiles
max_staging_pre_rht[2] = max across Warp 2's tiles
max_staging_pre_rht[3] = max across Warp 3's tiles
```

##### Final reduction and atomic update (Lines 318-335)

```cpp
  constexpr int kNumWarpsPow2 = NextPowerOf2<kNumWarps>();   // = 4
  if (warpid == 1) {
    if (kReturnTransposedAmax) {
      float transpose_accum = local_rank < kNumWarps ? staging_for_transpose[local_rank] : 0.0f;
      transpose_accum = warp_reduce_max<kNumWarpsPow2>(transpose_accum);
      if (local_rank == 0) {
        atomicMaxFloat(output_transpose_amax_ptr, transpose_accum);
      }
    }
  }
  if (warpid == 2) {
    if (kReturnPreRhtAmax) {
      float pre_rht_accum = local_rank < kNumWarps ? staging_for_pre_rht[local_rank] : 0.0f;
      pre_rht_accum = warp_reduce_max<kNumWarpsPow2>(pre_rht_accum);
      if (local_rank == 0) {
        atomicMaxFloat(output_pre_rht_amax_ptr, pre_rht_accum);
      }
    }
  }
```

**Warp 1 (Threads 32-63)**:
- Lanes 0-3 load `max_staging_transpose[0..3]`
- Lanes 4-31 load 0.0f
- Warp reduce: finds max across these 32 values
- Lane 0 of Warp 1 (Thread 32): Atomically updates global memory with block's transpose_amax

**Warp 2 (Threads 64-95)**:
- Lanes 0-3 load `max_staging_pre_rht[0..3]`
- Warp reduce
- Lane 0 of Warp 2 (Thread 64): Atomically updates global memory with block's pre_rht_amax

**Global memory state after this block**:
- `output_pre_rht_amax_ptr`: max(abs(input[0:128, 0:128])) for this block
- `output_transpose_amax_ptr`: max(abs(RHT_transposed(input[0:128, 0:128]))) for this block

**Across all blocks**: Atomic max operations accumulate the global maximum.

### Cleanup Phase (Line 495)

```cpp
  destroy_barriers<STAGES_X * STAGES_Y>(mbar, is_master_thread);
```

**Thread 0**: Invalidates the 4 barriers in shared memory.

## Summary of Pipelined Execution

### Pipeline Visualization

```
Time →

Thread 0 Actions:
[Init]  [TMA Stage0]  [Wait0]  [TMA Stage1]  [Wait1]  [TMA Stage2]  [Wait2]  [TMA Stage3]  [Wait3]
                        ↓                        ↓                       ↓                       ↓
                        Compute Stage0           Compute Stage1          Compute Stage2          Compute Stage3

All Threads:
        [Init Hadamard fragments]
                                  [Compute 0]            [Compute 1]            [Compute 2]            [Compute 3]
                                                                                                                      [Reduce]

Buffers:
in_sh_0: [─TMA0─][■■■■Data0■■■■][─TMA2─][■■■■Data2■■■■]
in_sh_1:         [─TMA1─][■■■■Data1■■■■][─TMA3─][■■■■Data3■■■■]

Legend:
─TMA─ : TMA transfer in progress
■Data■ : Data ready for compute
```

### Per-Stage Buffer State

| Stage | Operation | in_sh_0 | in_sh_1 | Compute On |
|-------|-----------|---------|---------|------------|
| 0 | Load stage 0, Prefetch stage 1 | input[0:64,0:64] | Loading input[0:64,64:128] | in_sh_0 |
| 1 | Process stage 1, Prefetch stage 2 | Loading input[64:128,0:64] | input[0:64,64:128] | in_sh_1 |
| 2 | Process stage 2, Prefetch stage 3 | input[64:128,0:64] | Loading input[64:128,64:128] | in_sh_0 |
| 3 | Process stage 3 | input[64:128,0:64] | input[64:128,64:128] | in_sh_1 |

### Key Performance Optimizations

1. **Double Buffering**: Two shared memory buffers allow simultaneous compute and data transfer
2. **TMA Prefetching**: Next stage loads while current stage computes (1-2 stage lookahead)
3. **Async Barriers**: `mbarrier` enables efficient synchronization with TMA completion
4. **Tensor Core Utilization**: 16×16 matrix operations via WMMA instructions
5. **Memory Swizzling**: Bank conflict avoidance via `swizzle_128B_atom_32B`
6. **Warp-level Parallelism**: 4 warps process independent 16×16 tiles
7. **Register Accumulation**: Amax values accumulate in registers, minimizing shared memory traffic
8. **Hierarchical Reduction**: Intra-warp (shuffle) → Inter-warp (shared mem) → Inter-block (atomic)

## Thread/Warp/Block Hierarchy Summary

### For Input Shape 1024 × 2048

**Grid**: 16 × 8 blocks
- Each block: 128 × 128 tile of input

**Block**: 128 threads (4 warps)
- Each warp: 32 threads

**Processing per Block**:
- 4 stages (2×2 spatial grid of 64×64 buffers)
- Each stage: 4 warps × 4 vertical strips = 16 independent 16×16 tiles
- Total: 64 16×16 tiles per block (128×128 / (16×16))

**Processing per Warp per Stage**:
- 4 vertical iterations (compute_stage_y = 0..3)
- Each iteration: one 16×16 tile
- Total per stage: 4 tiles × 16 rows × 16 cols = 1024 elements
- Total across 4 stages: 4096 elements (64 × 64)

**Entire Grid**:
- 16 × 8 = 128 blocks
- 128 blocks × 4 warps = 512 warps
- 512 warps × 4096 elements/warp = 2,097,152 elements = 1024 × 2048 ✓

## Conclusion

This kernel achieves high throughput by:
1. Overlapping global → shared memory transfers with computation
2. Leveraging tensor cores for 16×16 matrix operations
3. Minimizing synchronization overhead with async barriers
4. Efficiently reducing amax values hierarchically across thread/warp/block levels
5. Computing multiple amax outputs (pre-RHT, identity, transposed) in a single pass

The pipelined architecture ensures that while one stage is computing on data in one buffer, the next stage's data is loading into the other buffer, maximizing GPU utilization.
