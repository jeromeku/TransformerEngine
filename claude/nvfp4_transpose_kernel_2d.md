# NVFP4 Quantize-Transpose Kernel Deep Dive (2D Scaling)

**Analysis Date**: 2025-11-27
**Input Shape**: 128 × 1024 (rows × cols)
**Configuration**:
- **2D Block Scaling**: 16×16 blocks (vs 1×16 in 1D mode)
- `return_transpose=True`
- `use_stochastic_rounding=False`

---

## Table of Contents

1. [2D vs 1D Scaling Overview](#1-2d-vs-1d-scaling-overview)
2. [Kernel Entry Point](#2-kernel-entry-point)
3. [High-Level Design Changes](#3-high-level-design-changes)
4. [2D Block Configuration](#4-2d-block-configuration)
5. [Step-by-Step Execution](#5-step-by-step-execution)
6. [2D Amax Computation](#6-2d-amax-computation)
7. [Warp Reduction Logic](#7-warp-reduction-logic)
8. [Memory and Compute Mapping](#8-memory-and-compute-mapping)
9. [Performance Comparison](#9-performance-comparison)

---

## 1. 2D vs 1D Scaling Overview

### 1.1 Scaling Granularity Comparison

**1D Scaling** (previous analysis):
- **Rowwise**: 1 row × 16 cols per scale
- **Columnwise**: 16 rows × 1 col per scale
- Each scale covers 16 elements in a line

**2D Scaling** (this analysis):
- **Both directions**: 16 rows × 16 cols per scale
- Each scale covers a 16×16 = 256-element block
- More spatial locality, better for 2D patterns

### 1.2 Visual Comparison

**1D Scaling Pattern**:
```
Rowwise (identity output):
┌────────────────────────────────────┐
│ S0  S0  ... S0  │ S1  S1  ... S1  │  ← Row 0
├────────────────────────────────────┤
│ S64 S64 ... S64 │ S65 S65 ... S65 │  ← Row 1
└────────────────────────────────────┘
  16 elements       16 elements

Each row has 64 scales (1024/16)
Total: 128 × 64 = 8,192 scales
```

**2D Scaling Pattern**:
```
2D blocks:
┌─────────────┬─────────────┬─────────
│   S0        │   S1        │   S2   ...
│   (16×16)   │   (16×16)   │
├─────────────┼─────────────┼─────────
│   S64       │   S65       │   S66  ...
│   (16×16)   │   (16×16)   │
├─────────────┼─────────────┼─────────

Each scale covers 16×16 = 256 elements
Scales per row: 1024/16 = 64
Scales per col: 128/16 = 8
Total: 64 × 8 = 512 scales (vs 8,192 in 1D)
```

### 1.3 Key Advantages of 2D Scaling

1. **Fewer scales**: 512 vs 8,192 scales → 16× reduction in scale storage
2. **Better spatial locality**: 16×16 blocks capture 2D patterns better
3. **Simpler scale management**: Fewer global memory writes
4. **More efficient**: Better quantization for structured data (images, feature maps)

---

## 2. Kernel Entry Point

### 2.1 Host-Side Launch

**File**: [transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh:1267-1270](../transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh#L1267-L1270)

```cpp
template <bool use_2d_quantization>
void quantize_transpose(const Tensor &input, ...) {
  // ...

  if constexpr (use_2d_quantization) {
    kernel = quantize_transpose_nvfp4_2D_kernel<COMPUTE_ACTIVATIONS, ParamOP, OP, IType,
                                                USE_STOCHASTIC_ROUNDING, RETURN_TRANSPOSE>;
  } else {
    kernel = quantize_transpose_nvfp4_kernel<...>;  // 1D version
  }

  // Same grid/block configuration
  const dim3 grid(blocks_X, blocks_Y);  // grid(8, 1) for 128×1024
  const size_t block_size = THREADS_NUM;  // 128 threads

  kernel<<<grid, block_size, dshmem_size, stream>>>(...);
}
```

**For our 128 × 1024 input**:
- Grid: 8 blocks × 1 block (same as 1D)
- Block size: 128 threads (same as 1D)
- Each block: processes 128×128 chunk
- Pipeline: 4 stages of 32×128 tiles

---

## 3. High-Level Design Changes

### 3.1 Kernel Structure Comparison

**1D Kernel**:
```
Stage Loop (4 stages):
  ├─ Prefetch next tile
  ├─ Wait for current tile
  ├─ COLUMNWISE: Per-thread vertical amax + quantize (16 elements)
  ├─ ROWWISE: Per-thread horizontal amax + quantize (16 elements)
  └─ TMA store
```

**2D Kernel**:
```
Stage Loop (4 stages):
  ├─ Prefetch next tile
  ├─ Wait for current tile
  ├─ 2D BLOCK AMAX: Warp-collaborative 16×16 block amax (2 iterations)
  │   └─ Store amax to shared memory matrix [2][8]
  ├─ COLUMNWISE: Lookup amax from SHMEM matrix + quantize
  ├─ ROWWISE: Lookup amax from SHMEM matrix + quantize
  └─ TMA store
```

### 3.2 Major Differences

| Aspect | 1D Kernel | 2D Kernel |
|--------|-----------|-----------|
| **Amax Computation** | Per-thread, inline during load | Warp-collaborative, separate pass |
| **Amax Storage** | Registers only | Shared memory matrix `[2][8]` |
| **Block Size** | 1×16 (rowwise) or 16×1 (colwise) | 16×16 (both) |
| **Thread Organization** | Different for row/col | Same for both passes |
| **Warp Reduction** | Not used | Used for 16×16 blocks |
| **Scale Count** | 8,192 (128×64) | 512 (8×64) |

---

## 4. 2D Block Configuration

### 4.1 2D Block Constants

**File**: [transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh:655-661](../transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh#L655-L661)

```cpp
// NEW: 2D Block-based scaling constants
constexpr size_t BLOCK_DIM = 16;  // 16×16 blocks
constexpr size_t BLOCKS_PER_TILE_Y = TILE_DIM_Y / BLOCK_DIM;  // 32/16 = 2
constexpr size_t BLOCKS_PER_TILE_X = TILE_DIM_X / BLOCK_DIM;  // 128/16 = 8

// For amax computation
constexpr size_t ITERATIONS_BLOCK = 2;  // 2 iterations to compute amaxes of 1 tile
constexpr size_t BLOCKS_PER_WARP = BLOCKS_PER_TILE_X / (THREADS_NUM / 32);  // 8 / 4 = 2
```

**Tile Structure** (32 × 128):
```
┌───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
│ B0,0  │ B0,1  │ B0,2  │ B0,3  │ B0,4  │ B0,5  │ B0,6  │ B0,7  │  ← Row 0-15
│ 16×16 │ 16×16 │ 16×16 │ 16×16 │ 16×16 │ 16×16 │ 16×16 │ 16×16 │
├───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
│ B1,0  │ B1,1  │ B1,2  │ B1,3  │ B1,4  │ B1,5  │ B1,6  │ B1,7  │  ← Row 16-31
│ 16×16 │ 16×16 │ 16×16 │ 16×16 │ 16×16 │ 16×16 │ 16×16 │ 16×16 │
└───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘

2 rows × 8 cols = 16 blocks per tile
```

### 4.2 Shared Memory for Amax

**File**: [transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh:757](../transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh#L757)

```cpp
__shared__ __align__(16) float block_amax_matrix[BLOCKS_PER_TILE_Y][BLOCKS_PER_TILE_X + 1];
//                                                 2                  8 + 1 (padding)
```

**Layout**:
```
block_amax_matrix[2][9]:
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬──────┐
│ B0,0│ B0,1│ B0,2│ B0,3│ B0,4│ B0,5│ B0,6│ B0,7│ PAD  │  ← Row 0 blocks
├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼──────┤
│ B1,0│ B1,1│ B1,2│ B1,3│ B1,4│ B1,5│ B1,6│ B1,7│ PAD  │  ← Row 1 blocks
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴──────┘

Size: 2 × 9 × 4 bytes = 72 bytes
Padding column avoids bank conflicts
```

### 4.3 Warp-to-Block Mapping

**128 threads = 4 warps**:
```
Warp 0 (threads 0-31):   Processes blocks in columns, distributed
Warp 1 (threads 32-63):  Processes blocks in columns, distributed
Warp 2 (threads 64-95):  Processes blocks in columns, distributed
Warp 3 (threads 96-127): Processes blocks in columns, distributed
```

**Each warp processes 2 blocks in X dimension** (BLOCKS_PER_WARP=2):
```
Warp 0: blocks [0, 1, 4, 5] (columns 0-31)
Warp 1: blocks [2, 3, 6, 7] (columns 32-63)
Warp 2: blocks [0, 1, 4, 5] (columns 64-95)
Warp 3: blocks [2, 3, 6, 7] (columns 96-127)
```

---

## 5. Step-by-Step Execution

### 5.1 Kernel Initialization

**File**: [transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh:634-772](../transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh#L634-L772)

```cpp
__global__ void quantize_transpose_nvfp4_2D_kernel(...) {
  // === STEP 1-6: Same as 1D kernel ===
  // - Early exit check
  // - RNG initialization (not used since USE_STOCHASTIC_ROUNDING=false)
  // - Block/thread offset computation
  // - SHMEM allocation
  // - Global scale computation
  // - Mbarrier initialization

  // === STEP 7: NEW - Compute warp/lane IDs for 2D blocks ===
  const size_t warp_id = threadIdx.x / 32;  // 0-3
  const size_t lane_id = threadIdx.x % 32;  // 0-31
  const size_t block_in_warp = lane_id / BLOCKS_PER_WARP;  // lane_id / 2 = 0-15

  // === STEP 8: NEW - Define warp reduction helper ===
  auto warp_reduce_amax = [](float thread_amax, int block_in_warp) -> float {
    #pragma unroll
    for (int delta = 8; delta >= 1; delta /= 2) {
      float other_amax = __shfl_xor_sync(0xffffffff, thread_amax, delta);
      thread_amax = fmaxf(thread_amax, other_amax);
    }
    return thread_amax;
  };

  // === STEP 9: Initialize shared memory amax matrix ===
  __shared__ __align__(16) float block_amax_matrix[BLOCKS_PER_TILE_Y][BLOCKS_PER_TILE_X + 1];

  // === STEP 10: Prefetch first tile ===
  copy_2d_to_shared(&in_sh[0], &tensor_map_input, block_offset_X, block_offset_Y,
                    shmem_buff_size, &mbar[0], is_master_thread);
}
```

**New Variables**:
- `warp_id`: Which warp (0-3) this thread belongs to
- `lane_id`: Position within warp (0-31)
- `block_in_warp`: Which 16×16 block this thread's lane processes (0-15)
- `warp_reduce_amax`: Lambda for warp-level amax reduction

### 5.2 Main Stage Loop (Overview)

**File**: [transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh:774-1129](../transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh#L774-L1129)

```cpp
#pragma unroll
for (size_t stage = 0; stage < STAGES; ++stage) {  // 4 stages
  const size_t buff = stage % BUFFS_NUM;  // 0, 1, 0, 1
  const size_t stage_offset_Y = stage * BUFF_DIM_Y;  // 0, 32, 64, 96

  // === SUBSTEP 1: Prefetch next tile (same as 1D) ===
  if (next_stage < STAGES) {
    cp_async_bulk_wait_group_read<1>();
    copy_2d_to_shared(..., &mbar[next_stage], is_master_thread);
  }

  ptx::fence_proxy_async_shared_cta();
  ptx::mbarrier_wait_parity(&mbar[stage], 0);

  // === SUBSTEP 2: NEW - 2D BLOCK AMAX COMPUTATION ===
  #pragma unroll
  for (size_t block_iter = 0; block_iter < ITERATIONS_BLOCK; ++block_iter) {  // 2 iterations
    // Compute amax for 16×16 blocks (detailed in 5.3)
  }

  __syncthreads();  // Ensure block_amax_matrix is ready

  // === SUBSTEP 3: COLUMNWISE QUANTIZATION ===
  if constexpr (RETURN_TRANSPOSE) {
    // Lookup amax from block_amax_matrix and quantize (detailed in 5.4)
  }

  // === SUBSTEP 4: ROWWISE QUANTIZATION ===
  {
    // Lookup amax from block_amax_matrix and quantize (detailed in 5.5)
  }

  // === SUBSTEP 5: TMA Store (same as 1D) ===
  ptx::fence_proxy_async_shared_cta();
  __syncthreads();
  if (is_master_thread) {
    cp_async_bulk_tensor_2d_shared_to_global(...);
    cp_async_bulk_commit_group();
  }
}
```

---

## 6. 2D Amax Computation

### 6.1 Overview

This is the key difference from 1D mode. Instead of computing amax inline during quantization, we have a **separate pass** that:
1. All threads collaboratively compute amax for each 16×16 block
2. Use warp reductions to aggregate amax across threads
3. Store results in shared memory matrix

**File**: [transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh:806-866](../transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh#L806-L866)

### 6.2 Thread-to-Block Mapping

For a 32×128 tile with 128 threads:

```cpp
const size_t warp_id = threadIdx.x / 32;      // 0-3
const size_t lane_id = threadIdx.x % 32;      // 0-31

// For each block iteration
const size_t block_in_tile_y = block_iter;    // 0 or 1
const size_t block_in_tile_x = threadIdx.x / BLOCK_DIM;  // threadIdx.x / 16 = 0-7
```

**Example mapping for block_iter=0** (top 16 rows):
```
Thread 0-15:   block_in_tile_x = 0  (block B0,0)
Thread 16-31:  block_in_tile_x = 1  (block B0,1)
Thread 32-47:  block_in_tile_x = 2  (block B0,2)
Thread 48-63:  block_in_tile_x = 3  (block B0,3)
Thread 64-79:  block_in_tile_x = 4  (block B0,4)
Thread 80-95:  block_in_tile_x = 5  (block B0,5)
Thread 96-111: block_in_tile_x = 6  (block B0,6)
Thread 112-127: block_in_tile_x = 7  (block B0,7)
```

**Each group of 16 threads** processes one 16×16 block:
- 16 threads × 16 rows = 256 elements per block
- Each thread processes 16 elements (one per row)

### 6.3 Detailed Amax Computation (NO_ACTIVATIONS_NOT_FP32_INPUT Path)

**File**: [transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh:812-829](../transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh#L812-L829)

```cpp
#pragma unroll
for (size_t block_iter = 0; block_iter < ITERATIONS_BLOCK; ++block_iter) {  // 2 iterations
  IType2 thread_amax_2x = {static_cast<IType>(0.0f), static_cast<IType>(0.0f)};
  const size_t block_in_tile_y = block_iter;  // 0 or 1
  const size_t block_in_tile_x = threadIdx.x / BLOCK_DIM;  // 0-7

  if constexpr (NO_ACTIVATIONS_NOT_FP32_INPUT) {  // BF16 input, no activations
    // === STEP 1: Each thread loads and computes amax for 16 elements (1 column) ===
    for (int elem = 0; elem < BLOCK_DIM; elem += 2) {  // 16 elements, 2 at a time
      // Compute element row/col indices
      const size_t elem_0_row = block_iter * BLOCK_DIM + elem;      // 0, 2, 4, ..., 14 (or 16, 18, ..., 30)
      const size_t elem_1_row = elem_0_row + 1;                     // 1, 3, 5, ..., 15 (or 17, 19, ..., 31)

      // Each thread handles one column within its block
      // warp_id * BLOCKS_PER_WARP * BLOCK_DIM: base column for this warp
      // lane_id: offset within warp's columns
      const size_t elem_0_col = warp_id * BLOCKS_PER_WARP * BLOCK_DIM + lane_id;
      const size_t elem_1_col = elem_0_col;

      // Compute SHMEM offsets
      const size_t shmem_offset_0 = buff_offset_in + elem_0_row * BUFF_IN_DIM_X + elem_0_col;
      const size_t shmem_offset_1 = buff_offset_in + elem_1_row * BUFF_IN_DIM_X + elem_1_col;

      // Load 2 BF16 values
      IType2 val_2x;
      val_2x.x = in_sh[shmem_offset_0];
      val_2x.y = in_sh[shmem_offset_1];

      // Update thread amax using PTX intrinsic
      ptx::abs_max_2x(thread_amax_2x, thread_amax_2x, val_2x);
    }

    // === STEP 2: Convert BF16 amax to float ===
    thread_amax = static_cast<float>(__hmax(__habs(thread_amax_2x.x), __habs(thread_amax_2x.y)));
  }

  // === STEP 3: Warp reduction (detailed in section 7) ===
  block_amax = warp_reduce_amax(thread_amax, block_in_warp);

  // === STEP 4: Write to SHMEM matrix (2 threads per block) ===
  if (lane_id == 0 || lane_id == 16) {
    block_amax_matrix[block_in_tile_y][block_in_tile_x] = block_amax;
  }
}

__syncthreads();  // Ensure all blocks' amax computed before quantization
```

### 6.4 Example Walkthrough (Block B0,0, block_iter=0)

**Block B0,0** is at tile position (0, 0), covering rows [0, 16) and columns [0, 16).

**Threads involved**: threads 0-15 (since block_in_tile_x = threadIdx.x / 16 = 0)

**Thread 0** (warp 0, lane 0):
- Processes column 0 of block B0,0
- Loads 16 elements: (row 0, col 0), (row 1, col 0), ..., (row 15, col 0)
- Computes thread_amax = max(|all 16 values|)

**Thread 1** (warp 0, lane 1):
- Processes column 1 of block B0,0
- Loads 16 elements: (row 0, col 1), (row 1, col 1), ..., (row 15, col 1)
- Computes thread_amax

...

**Thread 15** (warp 0, lane 15):
- Processes column 15 of block B0,0
- Loads 16 elements: (row 0, col 15), (row 1, col 15), ..., (row 15, col 15)
- Computes thread_amax

**Warp reduction**: 16 threads in warp 0 (lanes 0-15) reduce their amax values to get block amax

**Result**: `block_amax_matrix[0][0]` = max of all 256 elements in block B0,0

---

## 7. Warp Reduction Logic

### 7.1 Warp Shuffle Reduction

**File**: [transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh:760-767](../transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh#L760-L767)

```cpp
// Helper function for warp reduction
auto warp_reduce_amax = [](float thread_amax, int block_in_warp) -> float {
  #pragma unroll
  for (int delta = 8; delta >= 1; delta /= 2) {
    float other_amax = __shfl_xor_sync(0xffffffff, thread_amax, delta);
    thread_amax = fmaxf(thread_amax, other_amax);
  }
  return thread_amax;
};
```

### 7.2 Reduction Tree

Each 16×16 block is processed by 16 consecutive threads within a warp (or split across 2 half-warps).

**Example: Threads 0-15 reducing amax for Block B0,0**

```
Initial state:
T0:  amax0
T1:  amax1
T2:  amax2
T3:  amax3
T4:  amax4
T5:  amax5
T6:  amax6
T7:  amax7
T8:  amax8
T9:  amax9
T10: amax10
T11: amax11
T12: amax12
T13: amax13
T14: amax14
T15: amax15

Iteration 1 (delta=8):
  __shfl_xor_sync(mask, thread_amax, 8)
  T0 ← max(T0, T8)    T8 ← max(T8, T0)
  T1 ← max(T1, T9)    T9 ← max(T9, T1)
  T2 ← max(T2, T10)   T10 ← max(T10, T2)
  T3 ← max(T3, T11)   T11 ← max(T11, T3)
  T4 ← max(T4, T12)   T12 ← max(T12, T4)
  T5 ← max(T5, T13)   T13 ← max(T13, T5)
  T6 ← max(T6, T14)   T14 ← max(T14, T6)
  T7 ← max(T7, T15)   T15 ← max(T15, T7)

Iteration 2 (delta=4):
  __shfl_xor_sync(mask, thread_amax, 4)
  T0 ← max(T0, T4)    T4 ← max(T4, T0)
  T1 ← max(T1, T5)    T5 ← max(T5, T1)
  ...
  T11 ← max(T11, T15) T15 ← max(T15, T11)

Iteration 3 (delta=2):
  __shfl_xor_sync(mask, thread_amax, 2)
  T0 ← max(T0, T2)    T2 ← max(T2, T0)
  ...

Iteration 4 (delta=1):
  __shfl_xor_sync(mask, thread_amax, 1)
  T0 ← max(T0, T1)    T1 ← max(T1, T0)
  ...

Final state: ALL 16 threads have the same result = max(amax0, ..., amax15)
```

### 7.3 Multiple Blocks per Warp

Since BLOCKS_PER_WARP = 2, each warp processes 2 blocks in the X dimension.

**Warp 0 (lanes 0-31)** during block_iter=0:
- **Lanes 0-15**: Process block B0,0 (columns 0-15)
- **Lanes 16-31**: Process block B0,1 (columns 16-31)

**Reduction operates independently** on each half-warp:
- `delta=8` reduces within each 16-thread group
- Lanes 0-15 reduce to a single value (block B0,0 amax)
- Lanes 16-31 reduce to a single value (block B0,1 amax)

### 7.4 Writing Results to SHMEM

```cpp
if (lane_id == 0 || lane_id == 16) {
  block_amax_matrix[block_in_tile_y][block_in_tile_x] = block_amax;
}
```

**Why lane_id == 0 or 16?**
- Each 16-thread group has the same reduced amax
- Only need one thread to write: lane 0 (for first group) or lane 16 (for second group)
- Avoids redundant writes

**Example for Warp 0, block_iter=0**:
- Lane 0 writes `block_amax_matrix[0][0]` (block B0,0's amax)
- Lane 16 writes `block_amax_matrix[0][1]` (block B0,1's amax)

---

## 8. Memory and Compute Mapping

### 8.1 Columnwise Quantization (2D Mode)

**File**: [transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh:869-969](../transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh#L869-L969)

The columnwise path is **very similar to 1D mode**, except:
1. **Amax lookup** from `block_amax_matrix` instead of computing inline
2. **No amax computation** during load

```cpp
if constexpr (RETURN_TRANSPOSE) {
  #pragma unroll
  for (size_t it = 0; it < ITERATIONS_TRANSPOSE; ++it) {  // 2 iterations
    // === Compute which 16×16 block this iteration processes ===
    const size_t block_in_tile_y = it;  // 0 or 1
    const size_t block_in_tile_x = threadIdx.x / BLOCK_DIM;  // 0-7

    const size_t in_thread_offset_Y = 0 + it * SCALE_DIM;  // 0 or 16
    const size_t in_thread_offset_X = thread_offset_X_colwise;  // threadIdx.x

    const size_t shmem_offset_base_colwise_in =
        buff_offset_in + in_thread_offset_Y * BUFF_IN_DIM_X + in_thread_offset_X;
    const size_t shmem_offset_base_colwise_out_t =
        buff_offset_out_t + out_t_thread_offset_Y * BUFF_OUT_T_DIM_X + out_t_thread_offset_X;

    // === NEW: Lookup amax from SHMEM matrix ===
    block_amax = block_amax_matrix[block_in_tile_y][block_in_tile_x];

    // === Load 16 elements (NO amax computation) ===
    float in_compute_colwise[SCALE_DIM];
    IType in_colwise_IType[SCALE_DIM];

    if constexpr (NO_ACTIVATIONS_NOT_FP32_INPUT) {
      #pragma unroll
      for (int i = 0; i < SCALE_DIM; ++i) {
        const int shmem_offset_colwise = shmem_offset_base_colwise_in + i * BUFF_IN_DIM_X;
        in_colwise_IType[i] = in_sh[shmem_offset_colwise];
        // No amax computation here!
      }
    }

    // === Compute E4M3 scaling factor (same as 1D) ===
    const nvfp4_scale_t S_dec_b_fp8 =
        compute_decoding_scaling_factor(block_amax, S_enc_colwise);

    const size_t scale_idx_sh = tid_Y_t * SCALES_PER_CHUNK_Y + stage * ITERATIONS_TRANSPOSE + it;
    out_colwise_scales_sh[scale_idx_sh] = S_dec_b_fp8;

    // === Compute block encoding scale (same as 1D) ===
    const float block_scale_inverse =
        fminf(1.0f / (static_cast<float>(S_dec_b_fp8) * S_dec_colwise), float_max);
    const float2 block_scale_inverse_2x{block_scale_inverse, block_scale_inverse};

    // === Quantize to FP4 (same as 1D, but USE_STOCHASTIC_ROUNDING=false) ===
    fp4e2m1x4 regs[SCALE_DIM / 4];
    #pragma unroll
    for (int e = 0; e < SCALE_DIM / 4; ++e) {
      const uint32_t rbits = 0;  // Not used since stochastic rounding disabled
      const uint64_t elts = *reinterpret_cast<uint64_t *>(&in_colwise_IType[4 * e]);

      // Uses round-to-nearest (rn) since USE_STOCHASTIC_ROUNDING=false
      regs[e] = ptx::mul_cvt_bf16_to_fp4_4x<false>(elts, block_scale_inverse_2x, rbits);
    }

    // === Write to SHMEM with transpose (same as 1D) ===
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

**Key Difference Summary**:
```
1D Mode:
  Load → Compute amax → Store amax → Quantize

2D Mode:
  [Separate pass: Compute all amaxes → Store in SHMEM matrix]
  Load → Lookup amax → Quantize
```

### 8.2 Rowwise Quantization (2D Mode)

**File**: [transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh:971-1098](../transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh#L971-L1098)

Similarly, the rowwise path:
1. **Lookups amax** from `block_amax_matrix` instead of computing inline
2. **Loads elements** without amax computation

```cpp
{
  const size_t stage_rowwise_scales_offset_Y = stage * BUFF_DIM_Y;

  #pragma unroll
  for (size_t it = 0; it < ITERATIONS_NORMAL; ++it) {  // 2 iterations
    // === Compute which 16×16 block this iteration processes ===
    const size_t block_in_tile_y = it;  // 0 or 1
    const size_t block_in_tile_x = tid_X_rowwise;  // 0-7

    const size_t it_thread_offset_Y_rowwise = thread_offset_Y_rowwise + it * THREADS_Y_ROWWISE;

    const size_t shmem_offset_base_rowwise_in =
        buff_offset_in + it_thread_offset_Y_rowwise * BUFF_IN_DIM_X;
    const size_t shmem_offset_base_rowwise_out =
        buff_offset_out + it_thread_offset_Y_rowwise * BUFF_OUT_DIM_X;

    // === NEW: Lookup amax from SHMEM matrix ===
    block_amax = block_amax_matrix[block_in_tile_y][block_in_tile_x];

    // === Load 16 elements with swizzling (NO amax computation) ===
    Vec<IType, PACK_SIZE> in_cached[WAVES];
    Vec<IType2, PACK_SIZE / 2> in_IType[WAVES];

    if constexpr (NO_ACTIVATIONS_NOT_FP32_INPUT) {
      #pragma unroll
      for (int w = 0; w < WAVES; ++w) {
        const size_t swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM;
        const size_t swizzled_thread_idx = thread_offset_X_rowwise + swizzled_group_idx;
        const size_t shmem_offset_rowwise = shmem_offset_base_rowwise_in + swizzled_thread_idx;

        // Load elements (no amax computation)
        in_IType[w].load_from(&in_sh[shmem_offset_rowwise]);
      }
    }

    // === Compute E4M3 scaling factor (same as 1D) ===
    const nvfp4_scale_t S_dec_b_fp8 =
        compute_decoding_scaling_factor(block_amax, S_enc_rowwise);

    // === Write scale to global memory (same as 1D) ===
    const size_t scales_offset_Y =
        scales_offset_Y_rowwise + stage * BUFF_DIM_Y + it * THREADS_Y_ROWWISE;
    const size_t scale_idx_global = scales_offset_Y * scale_stride + scales_offset_X_rowwise;

    const bool rowwise_scale_is_within_bounds_Y =
        (stage_rowwise_scales_offset_Y + it * THREADS_Y_ROWWISE + tid_Y_rowwise) < chunk_rows;
    if (rowwise_scale_is_within_bounds_X && rowwise_scale_is_within_bounds_Y) {
      scales_ptr[scale_idx_global] = S_dec_b_fp8;
    }

    // === Compute block encoding scale (same as 1D) ===
    const float block_scale_inverse =
        fminf(1.0f / (static_cast<float>(S_dec_b_fp8) * S_dec_rowwise), float_max);
    const float2 block_scale_inverse_2x{block_scale_inverse, block_scale_inverse};

    // === Quantize to FP4 (same as 1D) ===
    #pragma unroll
    for (int w = 0; w < WAVES; ++w) {
      Vec<fp4e2m1x4, PACK_SIZE / 4> out;

      #pragma unroll
      for (int e = 0; e < PACK_SIZE / 4; ++e) {
        const uint32_t rbits = 0;  // Not used
        const uint64_t elts = *reinterpret_cast<uint64_t *>(&in_IType[w].data.elt[2 * e]);

        // Round-to-nearest since USE_STOCHASTIC_ROUNDING=false
        out.data.elt[e] = ptx::mul_cvt_bf16_to_fp4_4x<false>(elts, block_scale_inverse_2x, rbits);
      }

      // === Write to SHMEM with swizzling (same as 1D) ===
      const size_t swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM;
      const size_t swizzled_idx = swizzled_group_idx + thread_offset_X_rowwise;
      const size_t shmem_offset_rowwise = shmem_offset_base_rowwise_out + swizzled_idx / 2;

      out.store_to(&out_data_sh[shmem_offset_rowwise]);
    }
  }
}
```

**Key Difference Summary**:
- Same load/quantize logic as 1D
- Amax comes from `block_amax_matrix` instead of computed inline
- Removes amax computation overhead from load loops

### 8.3 Scale Storage Comparison

**1D Rowwise Scales**:
```
Shape: 128 rows × 64 scales
Each scale covers: 1 row × 16 cols
Total scales: 8,192
```

**2D Rowwise Scales**:
```
Shape: 8 block-rows × 64 scales
Each scale covers: 16 rows × 16 cols
Total scales: 512

But stored as: 128 rows × 64 scales (for compatibility)
- Rows 0-15 all use scale[0][*]
- Rows 16-31 all use scale[1][*]
- ...
- Rows 112-127 all use scale[7][*]
```

**Implementation**:
```cpp
// Each of 16 threads writes the same scale 16 times (once per row in block)
for (size_t it = 0; it < ITERATIONS_NORMAL; ++it) {  // 2 iterations
  const size_t scales_offset_Y =
      scales_offset_Y_rowwise + stage * BUFF_DIM_Y + it * THREADS_Y_ROWWISE;

  // Thread 0 writes scale for rows {0, 16}
  // Thread 1 writes scale for rows {1, 17}
  // ...
  // Thread 15 writes scale for rows {15, 31}

  scales_ptr[scale_idx_global] = S_dec_b_fp8;
}
```

**Result**: Each 16×16 block's scale is replicated 16 times (once per row) to maintain row-major format.

---

## 9. Performance Comparison

### 9.1 Compute Complexity

**1D Mode**:
- **Amax computation**: Inline during load (no extra overhead)
- **Per-thread work**: 16 elements → 1 amax
- **Total amax ops**: 8,192 blocks × 16 elements = 131,072 comparisons

**2D Mode**:
- **Amax computation**: Separate pass with warp reduction
- **Per-thread work**: 16 elements → 1 thread_amax → warp reduction
- **Total amax ops**: 512 blocks × 256 elements = 131,072 comparisons (same)
- **Extra work**: Warp reductions (4 shuffle ops per 16-thread group)

### 9.2 Memory Traffic

**1D Mode**:
- **Scale storage**: 8,192 FP8 scales × 2 (rowwise + colwise) = 16,384 bytes
- **Scale writes**: Direct to global memory (rowwise), buffered in SHMEM (colwise)

**2D Mode**:
- **Scale storage**: 512 FP8 scales (stored as 8,192 for compatibility) × 2 = 16,384 bytes
- **Amax SHMEM**: 72 bytes (block_amax_matrix)
- **Scale writes**: Same as 1D, but replicates each scale 16 times

**Memory Traffic Summary**:
```
                    1D Mode         2D Mode
Input (BF16)        262,144 bytes   262,144 bytes
Output (FP4)        131,072 bytes   131,072 bytes
Scales              16,384 bytes    16,384 bytes
Amax SHMEM          0 bytes         72 bytes per block
Total               409,600 bytes   409,672 bytes
```

### 9.3 Quantization Quality

**1D Scaling**:
- Adapts to row-wise or column-wise patterns
- Good for 1D structure (sequences, vectors)
- Can overfit to outliers in a single row/column

**2D Scaling**:
- Adapts to spatial patterns (images, feature maps)
- Better for structured 2D data
- Averages over 16×16 blocks → more robust to outliers
- Trade-off: Less granularity (256 elements per scale vs 16)

**Example Scenario**:
```
Data with outliers:
┌─────────────────────────────┐
│ 1  1  1  1  │ 1  1  1  1    │
│ 1  1  1  1  │ 1  1  1  1    │
│ 1  1  1 99  │ 1  1  1  1    │  ← Outlier at (2, 3)
│ 1  1  1  1  │ 1  1  1  1    │
└─────────────────────────────┘

1D Rowwise:
- Row 2 scale influenced by outlier 99
- All 64 values in row 2 quantized with inflated scale
- Poor quantization for other 63 values

2D Block:
- Block (0,0) scale influenced by outlier 99
- 256 values in block quantized with inflated scale
- But other blocks unaffected
- Better isolation
```

### 9.4 Performance Metrics (Estimated)

For 128 × 1024 BF16 input on H100:

**1D Mode**:
- **Compute**: ~5 μs (inline amax + quantization)
- **Memory**: ~10 μs (TMA transfers)
- **Total**: ~15 μs

**2D Mode**:
- **Compute**: ~6 μs (separate amax pass + warp reductions + quantization)
- **Memory**: ~10 μs (TMA transfers, same as 1D)
- **Total**: ~16 μs

**Overhead**: ~1 μs (6-7% slower) due to:
- Separate amax computation pass
- Warp reductions
- SHMEM matrix writes/reads

**Trade-off**: Slightly slower, but better quantization quality for 2D-structured data.

---

## Summary

### Key Differences: 1D vs 2D Mode

| Aspect | 1D Mode | 2D Mode |
|--------|---------|---------|
| **Block Size** | 1×16 or 16×1 | 16×16 |
| **Scales per Tensor** | 8,192 × 2 | 512 × 2 (stored as 8,192) |
| **Amax Computation** | Inline during load | Separate pass with warp reduction |
| **Amax Storage** | Registers only | SHMEM matrix [2][9] |
| **Warp Reduction** | Not used | Used (4 shuffle ops) |
| **Thread Organization** | Different for row/col | Same for both |
| **Quantization Quality** | Better for 1D patterns | Better for 2D patterns |
| **Performance** | Baseline | ~6-7% slower |

### When to Use 2D Mode

**Use 2D mode when**:
- Data has spatial structure (images, feature maps, matrices)
- Want fewer scales (memory savings)
- Outliers are spatially localized
- Can afford slight performance overhead

**Use 1D mode when**:
- Data has 1D structure (sequences, time series)
- Need finest quantization granularity
- Performance is critical
- Outliers are row-wise or column-wise

### Implementation Highlights

1. **Separate Amax Pass**: Compute all 16×16 block amaxes first, store in SHMEM matrix
2. **Warp Collaboration**: 16 threads per block collaborate via warp reductions
3. **SHMEM Reuse**: Same amax matrix used for both rowwise and columnwise quantization
4. **Scale Replication**: Each 16×16 block's scale written 16 times to maintain row-major format
5. **Same Pipeline**: 4-stage pipeline with double buffering (unchanged from 1D)

### Optimization Techniques

1. **Warp Shuffle Reduction**: Fast tree reduction using `__shfl_xor_sync` (4 iterations)
2. **SHMEM Padding**: `[2][9]` array with padding to avoid bank conflicts
3. **Minimal Writes**: Only lanes 0 and 16 write to SHMEM matrix
4. **Amax Lookup**: Fast SHMEM reads instead of recomputing amax
5. **Round-to-Nearest**: Deterministic rounding (no RNG overhead when `use_stochastic_rounding=false`)

### For 128 × 1024 Input

**Grid**: 8 blocks × 1 block (same as 1D)
**Per-block**:
- 4 stages × 2 block-rows × 8 block-cols = 64 16×16 blocks per block
- 64 amax computations (separate pass)
- 2048 FP4 conversions per stage (rowwise + columnwise)

**Total**:
- 512 unique 16×16 block amaxes
- 512 unique scales (replicated to 8,192 for storage)
- 131,072 FP4 conversions (same as 1D)

---

## Conclusion

The 2D quantization kernel introduces a **separate amax computation pass** that:
- Computes 16×16 block amaxes collaboratively using warp reductions
- Stores amaxes in a small SHMEM matrix for reuse
- Enables better quantization for spatially-structured data
- Trades ~6-7% performance for better quantization quality

The core pipeline and quantization logic remain similar to 1D mode, with the key innovation being the **warp-collaborative block amax computation** that enables efficient 2D block scaling.
