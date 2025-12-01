# NVFP4 Quantize+Transpose Kernel: Remaining Cases and 1D vs 2D Comparison

Source kernel (conceptual):

```cpp
quantize_transpose_nvfp4_kernel<
    COMPUTE_ACTIVATIONS,
    ParamOP,
    OP,
    IType,
    USE_STOCHASTIC_ROUNDING,
    RETURN_TRANSPOSE>
```

This document continues the previous walkthrough of the **1D block-scaling**, `RETURN_TRANSPOSE = true`, `USE_STOCHASTIC_ROUNDING = false` case.

Here we cover the remaining configurations:

1. `USE_STOCHASTIC_ROUNDING ∈ {false, true}`, `RETURN_TRANSPOSE = true` (1D scaling).
2. **2D block scaling** (`use_2d_quantization = true`) with `RETURN_TRANSPOSE = true`.
3. Key differences between **1D vs 2D** scaling paths.

We focus on:

* PTX intrinsics and how rounding bits are used.
* Data movement patterns (HBM ↔ TMA ↔ shared ↔ registers).
* Thread/value ownership and reduction patterns.
* Pipeline structure (TMA double-buffering, compute scheduling).
* Why these choices help latency and IPC and keep CUDA cores / tensor cores busy.

> Note: Code snippets below are representative / annotated pseudo-code reflecting typical TransformerEngine NVFP4 kernels. Names and shapes match what we’ve already reasoned about; the structure is accurate even if some details are simplified for clarity.

---

## 0. Quick Baseline Recap (1D, SR = false, RETURN_TRANSPOSE = true)

Baseline configuration (already analyzed in detail):

```cpp
using IType = nv_bfloat16;            // or half
constexpr bool COMPUTE_ACTIVATIONS      = false;
constexpr bool USE_STOCHASTIC_ROUNDING = false;
constexpr bool RETURN_TRANSPOSE        = true;
constexpr bool use_2d_quantization     = false;  // host-side choice
```

Tiling and threads:

* Block processes a **128 × 128** chunk of the `[rows × cols]` matrix.
* Chunk is split into **4 stages**, each a **32 × 128** tile.
* 128 threads per block:

  * **Rowwise view**: 16 (Y) × 8 (X) logical thread grid.
  * **Colwise view**: 128 threads mapped 1:1 to 128 columns.
* Block-scaling granularity: **16 elements** (`SCALE_DIM = 16`).

Key steps per stage:

1. TMA loads `32 × 128` tile into `in_sh` (double-buffered).
2. **Colwise path** (transpose): threads walk down columns in 16-element blocks, compute `block_amax`, derive per-block scale, quantize to FP4, write to `out_t_data_sh`.
3. **Rowwise path**: threads walk across rows in 16-element blocks, compute `block_amax`, write rowwise scales to `scales_ptr`, quantize to FP4 into `out_data_sh`.
4. TMA stores rowwise and transposed FP4 tiles from shared → global.
5. After all stages, colwise scales are vector-stored from `out_colwise_scales_sh` → global.

Global vs block scales:

```cpp
// from global amax
float S_enc_rowwise = compute_global_encode_scaling_factor_FP4(amax_rowwise);
float S_dec_rowwise = 1.0f / S_enc_rowwise;

// per-block decode scale
nvfp4_scale_t S_dec_b = compute_decoding_scaling_factor(block_amax, S_enc_rowwise);

// actual encode factor used in kernel
float block_scale_inverse = 1.0f / (float(S_dec_b) * S_dec_rowwise);
```

Then the kernel encodes each value as roughly:

```cpp
q = fp4_round( x * block_scale_inverse );
```

and at dequant time you multiply by `(S_dec_b * S_dec_rowwise)`.

The **stochastic rounding flag**, **2D vs 1D scaling**, and **RETURN_TRANSPOSE** control how this skeleton is specialized.

---

## 1. Stochastic Rounding Variants (1D Scaling, RETURN_TRANSPOSE = true)

We now fix:

```cpp
constexpr bool use_2d_quantization = false;  // 1D scaling
constexpr bool RETURN_TRANSPOSE    = true;
```

and vary `USE_STOCHASTIC_ROUNDING`.

### 1.1 RNG & PTX rounding bits

The kernel always constructs a Philox RNG state per thread:

```cpp
size_t rng_sequence = threadIdx.x
                    + blockIdx.x * THREADS_NUM
                    + blockIdx.y * gridDim.x * THREADS_NUM;
size_t rng_seed   = rng_state ? rng_state[0] : 0;
size_t rng_offset = rng_state ? rng_state[1] : 0;

philox4x32_native_state<10> rng;
rng.init(rng_seed, rng_sequence, rng_offset);

uint4 random_uint4 = USE_STOCHASTIC_ROUNDING
                       ? rng.generate4()
                       : uint4{0, 0, 0, 0};
int rnd_idx = 0;
```

The helper used inside conversion looks like (conceptually):

```cpp
__device__ inline uint32_t get_rbits(
    philox4x32_native_state<10>& rng,
    uint4 &random_uint4,
    int &rnd_idx) {
  if constexpr (!USE_STOCHASTIC_ROUNDING) {
    return 0u;                           // deterministic rounding
  } else {
    // Use 32-bit chunks from random_uint4; refresh when exhausted.
    if (rnd_idx == 4) {
      random_uint4 = rng.generate4();
      rnd_idx = 0;
    }
    uint32_t r = (&random_uint4.x)[rnd_idx++];
    return r;
  }
}
```

The `rbits` are then passed to the FP4 conversion PTX helpers.

### 1.2 PTX FP4 convert with/without stochastic rounding

Inside the quantization loops, the kernel calls helpers like:

```cpp
// For BF16 input (4 lanes packed in 64 bits)
fp4e2m1x4 out4 =
    ptx::mul_cvt_bf16_to_fp4_4x<USE_STOCHASTIC_ROUNDING>(
        elts,                // 4×BF16 packed as uint64_t
        block_scale_inverse_2x,
        rbits
    );
```

This PTX helper does roughly:

```cpp
// pseudo-implementation
__device__ inline fp4e2m1x4 mul_cvt_bf16_to_fp4_4x(
    uint64_t in_4x_bf16,
    float2 scale_2x,      // (scale, scale)
    uint32_t rbits) {
  // 1. Unpack 4 BF16 → 4 FP32
  float x0, x1, x2, x3 = unpack_bf16_to_f32(in_4x_bf16);

  // 2. Apply scaling (block_scale_inverse)
  x0 *= scale_2x.x; x1 *= scale_2x.y;
  x2 *= scale_2x.x; x3 *= scale_2x.y;

  // 3. Use `rbits` to perturb mantissa if stochastic rounding enabled.
  //    Implementation is inline PTX, something in spirit of:
  //    cvt.rn.satfinite.e2m1x2.f32  (round-to-nearest)
  //    or cvt.rz.satfinite.e2m1x2.f32 (round-towards-zero)
  //    with extra random offset into the sub-ULP region.

  // 4. Pack 4 FP4 values into fp4e2m1x4 container.
}
```

The important point:

* **`USE_STOCHASTIC_ROUNDING = false`** → `rbits = 0`, so PTX conversion behaves like pure **round-to-nearest-even** (or whichever deterministic mode is baked in) with saturation (`satfinite`) to handle overflows.
* **`USE_STOCHASTIC_ROUNDING = true`** → `rbits` becomes a **per-lane random offset** in the rounding decision; the hardware/PTX sequence ensures rounding is unbiased over time.

### 1.3 1D + RETURN_TRANSPOSE, SR = false (already covered)

We already walked this in detail:

* RNG is initialized but unused (rbits = 0).
* Rounding is deterministic.
* All reduction, scaling, TMA, and transpose logic is the same as in the baseline recap.

### 1.4 1D + RETURN_TRANSPOSE, SR = true

Now set:

```cpp
constexpr bool USE_STOCHASTIC_ROUNDING = true;
```

Everything else in the pipeline stays **identical**, except for how `rbits` are generated and consumed. We highlight the exact places where behavior changes.

#### 1.4.1 Colwise (transposed) path

```cpp
for (size_t it = 0; it < ITERATIONS_TRANSPOSE; ++it) {
  // 1. Each thread loads a column block of 16 BF16s.
  IType in_colwise_IType[SCALE_DIM];
  IType block_amax_f16 = 0;

  #pragma unroll
  for (int i = 0; i < SCALE_DIM; ++i) {
    int offset = shmem_offset_base_colwise_in + i * BUFF_IN_DIM_X;
    in_colwise_IType[i] = in_sh[offset];
    block_amax_f16 = __hmax(block_amax_f16, __habs(in_colwise_IType[i]));
  }

  // 2. Compute per-block scale (same as SR=false).
  float block_amax = static_cast<float>(block_amax_f16);
  nvfp4_scale_t S_dec_b = compute_decoding_scaling_factor(block_amax, S_enc_colwise);
  float block_scale_inverse = 1.0f / (float(S_dec_b) * S_dec_colwise);
  float2 block_scale_inverse_2x{block_scale_inverse, block_scale_inverse};

  // 3. Quantize 16 values using stochastic rounding.
  uint32_t regs_4x[2]; // packs 8 FP4s twice -> 16 values

  #pragma unroll
  for (int e = 0; e < SCALE_DIM / 4; ++e) {
    // *** DIFFERENCE: rbits are now random ***
    uint32_t rbits = get_rbits(rng, random_uint4, rnd_idx);

    uint64_t elts = *reinterpret_cast<uint64_t*>(&in_colwise_IType[4 * e]);
    fp4e2m1x4 out4 =
        ptx::mul_cvt_bf16_to_fp4_4x<true>(elts, block_scale_inverse_2x, rbits);

    // Pack into regs_4x[...] as before.
  }

  // 4. Store `regs_4x` into out_t_data_sh with same swizzle pattern.
}
```

**Everything outside the `rbits` path is untouched.** So locality, TMA behavior, and thread-chunk assignments remain identical.

#### 1.4.2 Rowwise path

Rowwise quantization loop becomes:

```cpp
for (int w = 0; w < WAVES; ++w) {
  Vec<fp4e2m1x4, PACK_SIZE / 4> out;

  #pragma unroll
  for (int e = 0; e < PACK_SIZE / 4; ++e) {
    uint32_t rbits = get_rbits(rng, random_uint4, rnd_idx); // random now

    uint64_t elts =
        *reinterpret_cast<uint64_t*>(&in_IType[w].data.elt[2 * e]);

    out.data.elt[e] =
        ptx::mul_cvt_bf16_to_fp4_4x<true>(elts, block_scale_inverse_2x, rbits);
  }

  // Same swizzled store into out_data_sh
}
```

All the **data movement**, **scale computation**, and **shared memory swizzling** stay exactly the same:

* Still 2 waves per 16-wide block (`PACK_SIZE = 8`, `WAVES = 2`).
* Still coalesced loads via vector types.
* Still double-buffered per-stage TMA.

#### 1.4.3 Performance impact of stochastic rounding

**Extra cost:**

* A couple of integer ops per 4 values:

  * `get_rbits` increments `rnd_idx`, occasionally calls `rng.generate4()`.
  * PTX conversion uses `rbits` to bias rounding; this may add 1–2 instructions.

**Why it doesn’t kill throughput:**

* The kernel is **heavily memory/latency amortized**:

  * Shared loads are vectorized and coalesced.
  * The work per value (scale + FP4 convert) dominates `get_rbits` overhead.
* Philox is **counter-based** and parallel-friendly, so no cross-thread dependencies.
* Global memory bandwidth is unchanged; TMA patterns are identical.

In practice, SR tends to reduce quantization bias at a very modest IPC cost.

---

## 2. 2D Block Scaling with `RETURN_TRANSPOSE = true`

Now we discuss the **2D block scaling** variant controlled by the host via:

```cpp
bool use_2d_quantization = true;
quantize_transpose<true>(...);   // calls 2D-scaling kernel variant
```

Conceptually:

* 1D scaling (previous section) uses **one scale per 16 contiguous elements along the feature axis**:

  * Scale tensor shape: `[rows, cols / 16]`.
* 2D scaling uses **one scale per 16×16 block**:

  * Scale tensor shape: `[rows / 16, cols / 16]`.

In other words, 2D scaling groups **both rows and columns** into blocks of 16, sharing a scale within each 16×16 patch.

We keep `RETURN_TRANSPOSE = true`, so we still produce both rowwise and transposed FP4 outputs.

### 2.1 Intuitive picture: 1D vs 2D blocks

For a `128 × 128` chunk, `SCALE_DIM = 16`:

* 1D scaling:

  * **Rowwise view**: 128 rows, each decomposed into 8 blocks of length 16.

    * 128 × 8 scales per chunk.
  * Scales stored as `scale[row, block_x]`.
* 2D scaling:

  * Chunk decomposed into **8 × 8 2D blocks**, each `16 × 16`.

    * 64 scales per chunk.
  * Scales stored as `scale[block_y, block_x]`.

So 2D scaling:

* **Reduces** number of scale values by 2x in this example.
* Ties blocks across both rows and columns, smoothing local variations but potentially increasing quantization error for very anisotropic patterns.

### 2.2 Thread/block mapping for 2D scales

We reuse the same **chunk/tiling and TMA pipeline** as 1D:

```cpp
CHUNK_DIM_Y = 128;
CHUNK_DIM_X = 128;
SCALE_DIM   = 16;

// Each 16x16 block is indexed by (by, bx):
// by = 0..7, bx = 0..7 in the chunk
```

Thread assignments adapt as follows (conceptually):

1. **Within each stage (32 × 128):**

   * We still load data into `in_sh` with TMA.
   * We split `32 × 128` into two 16-row slabs vertically:

     * Rows `0..15` → blocks `by = {stage_row / 16}`.
     * Rows `16..31` → blocks `by = {stage_row / 16 + 1}`.
2. Each 16×16 block inside the stage:

   * Owned collectively by a **warp or sub-warp**.
   * Threads in that warp compute **local amax** over subsets of the 16×16 block.
   * Warp-level reduction yields a single `block_amax` for the 16×16 patch.

A simplified row/col indexing for a 16×16 block:

```cpp
int block_y = (stage_offset_Y + local_row) / SCALE_DIM;  // 0..7
int block_x = local_col / SCALE_DIM;                     // 0..7
int block_idx = block_y * SCALES_PER_CHUNK_X + block_x;  // 0..63
```

### 2.3 Computing 2D block amax

Instead of per-thread-per-row max, we now compute a **block-level** max across all 256 elements (`16 × 16`) in the block.

Pseudo-code for rowwise amax in 2D scaling (within a warp handling one 16×16 block):

```cpp
// Each thread processes a subset of the 16x16 block.
float thread_block_amax = 0.0f;

for (int local_row = thread_row_start; local_row < 16; local_row += thread_row_stride) {
  int global_row = block_base_row + local_row;

  // Process columns in PACK_SIZE chunks for vectorization.
  for (int local_col = 0; local_col < 16; local_col += PACK_SIZE) {
    int global_col = block_base_col + local_col;

    // Vectorized load of PACK_SIZE values from shared.
    Vec<IType, PACK_SIZE> v;
    v.load_from(&in_sh[global_row * TILE_DIM_X + global_col]);

    // Convert to 2-lane vectors and accumulate abs-max.
    #pragma unroll
    for (int e = 0; e < PACK_SIZE / 2; ++e) {
      IType2 pair = v.data.elt[e];
      thread_block_amax = max(thread_block_amax,
                              max(abs(pair.x), abs(pair.y)));
    }
  }
}

// Warp-level reduction to obtain block_amax.
float block_amax = warp_allreduce_max(thread_block_amax);
```

**Differences vs 1D scaling:**

* 1D: `block_amax` is computed **independently per 16-wide horizontal row chunk** (no warp-wide reductions across rows).
* 2D: `block_amax` is computed **jointly for 16×16 block**, requiring warp-level reductions.

### 2.4 2D block scales and indexing

Once `block_amax` is known for a 16×16 block:

```cpp
nvfp4_scale_t S_dec_b = compute_decoding_scaling_factor(block_amax, S_enc_rowwise);
int block_y = global_row / SCALE_DIM;   // 0..rows/16 - 1
int block_x = global_col / SCALE_DIM;   // 0..cols/16 - 1

size_t idx = block_y * (cols / SCALE_DIM) + block_x;
rowwise_scales[idx] = S_dec_b;
```

For **colwise** (transpose) scales in 2D mode:

* Similar logic but with `block_y_t` and `block_x_t` defined in the transposed coordinates.
* A helpful mental model:

  * Rowwise 2D scales indexing: `scale_2d[row_block, col_block]`.
  * Colwise 2D scales (for transpose): `scale_2d_t[col_block, row_block]`.

### 2.5 Quantization loop in 2D scaling

After `S_dec_b` is known, the conversions are **identical** to 1D scaling, except that now `S_dec_b` is shared among all 256 values in the 16×16 block.

Rowwise quantization for a 16×16 block (pseudo):

```cpp
float block_scale_inverse = 1.0f / (float(S_dec_b) * S_dec_rowwise);
float2 block_scale_inverse_2x{block_scale_inverse, block_scale_inverse};

for (int local_row = thread_row_start; local_row < 16; local_row += thread_row_stride) {
  int global_row = block_base_row + local_row;

  for (int local_col = 0; local_col < 16; local_col += PACK_SIZE) {
    int global_col = block_base_col + local_col;

    Vec<IType, PACK_SIZE> v;
    v.load_from(&in_sh[global_row * TILE_DIM_X + global_col]);

    Vec<fp4e2m1x4, PACK_SIZE / 4> out;

    #pragma unroll
    for (int e = 0; e < PACK_SIZE / 4; ++e) {
      uint64_t elts = *reinterpret_cast<uint64_t*>(&v.data.elt[2 * e]);
      uint32_t rbits = get_rbits(rng, random_uint4, rnd_idx); // 0 or random

      out.data.elt[e] =
          ptx::mul_cvt_bf16_to_fp4_4x<USE_STOCHASTIC_ROUNDING>(
              elts, block_scale_inverse_2x, rbits);
    }

    // Swizzled store into out_data_sh for rowwise layout.
  }
}
```

Colwise/transposed quantization in 2D mode reuses **the same 2D block scale** `S_dec_b` but uses overall colwise encode factor `S_enc_colwise` if the kernel defines separate row/col global scales.

### 2.6 TMA and pipelining in 2D scaling

Critically, **TMA and pipeline structure do not change** between 1D and 2D scaling:

1. **Load** `32 × 128` tiles from global → shared using TMA and double buffering.
2. **Wait** for TMA completion using `mbarrier_wait_parity` and `fence_proxy_async_shared_cta`.
3. **Compute**:

   * In 1D mode: per-16-element horizontal and vertical blocks.
   * In 2D mode: per-16×16 2D blocks with warp-level reductions.
4. **Store** rowwise and transposed FP4 tiles using TMA shared → global.
5. **Write** scales using coalesced vector stores.

Thus the **HBM traffic pattern** and **global latency hiding** remain essentially the same:

* 2D scaling adds a bit more on-chip math (warp reductions) per element.
* This extra compute is usually overlapped with memory access, so IPC is still high.

---

## 3. Stochastic Rounding + 2D Scaling + RETURN_TRANSPOSE

Now consider:

```cpp
use_2d_quantization     = true;
RETURN_TRANSPOSE        = true;
USE_STOCHASTIC_ROUNDING = true;
```

Conceptually this is a **Cartesian product** of the features we’ve already described:

1. **2D scaling**: per 16×16 block scale `S_dec_b`.
2. **Stochastic rounding**: random `rbits` passed into `mul_cvt_*_to_fp4_4x`.
3. **Transpose**: colwise path uses TMA + shared swizzles to produce transposed FP4 tile.

### 3.1 Step-by-step outline per stage

Per stage (32×128 tile):

1. **TMA load** into shared buffer `in_sh[buff]`.
2. For each 16×16 block in the tile:

   * All threads in a warp:

     * Load subsets of the 16×16 block from `in_sh`.
     * Compute local abs-max.
   * Warp-level reduction → `block_amax` for the block.
   * Compute `S_dec_b = f(block_amax, S_enc_rowwise/S_enc_colwise)`.
   * Save `S_dec_b` in shared scale buffers.
3. For each value in the 16×16 block (rowwise view):

   * Load from shared.
   * Multiply by `block_scale_inverse`.
   * Fetch `rbits = get_rbits(rng, random_uint4, rnd_idx)`.
   * Call `mul_cvt_bf16_to_fp4_4x<true>`.
   * Store to `out_data_sh` with swizzled indexing.
4. For each value in the 16×16 block (colwise view / transposed path):

   * Similar steps but with colwise/global scaling factor if distinct.
   * Store to `out_t_data_sh` mapping `[row, col]` → `[col, row]`.
5. After all blocks in the stage:

   * `fence_proxy_async_shared_cta()` + `__syncthreads()`.
   * Master thread issues `cp_async_bulk_tensor_2d_shared_to_global` for both rowwise and transposed tiles.
6. After all stages:

   * Vectorized writes of 2D scales from `out_rowwise_scales_sh`/`out_colwise_scales_sh` → global.

### 3.2 IPC and hardware utilization

The overall performance picture:

* **Memory path**: identical TMA tiling and double buffering as 1D mode.

  * Still doing full tiles via bulk 2D transfers; no change in bandwidth usage.
* **Compute path**: more math per value vs 1D, but still:

  * Heavy use of vectorized loads/stores.
  * Fused convert+pack operations on 4 lanes per call.
  * Additional warp-level reductions for 2D blocks.
* **Stochastic rounding** adds random-number consumption:

  * Overhead is amortized over many values per RNG call (4 FP4 conversions per 32-bit rbits chunk).

Given that quantization is often **memory-influenced** and these kernels explicitly hide TMA latency with double buffering, the extra instructions for 2D scaling and stochastic rounding tend to **increase arithmetic intensity** without drastically reducing SM occupancy or IPC.

---

## 4. Comparison Matrix: 1D vs 2D, SR On/Off (RETURN_TRANSPOSE = true)

Assume:

* `SCALE_DIM = 16`.
* Chunk = `128 × 128`.

| Mode       | Scaling Pattern                        | Scales per 128×128 Chunk                                | Scale Tensor Shape       | Amax Reduction Pattern                                | Rounding                                 | Data Movement                               |
| ---------- | -------------------------------------- | ------------------------------------------------------- | ------------------------ | ----------------------------------------------------- | ---------------------------------------- | ------------------------------------------- |
| 1D, SR off | 1D along feature dim (rowwise/colwise) | 128 rows × 8 blocks/row = 1024 rowwise, similar colwise | `[rows, cols / 16]`      | Per-thread 16-element abs-max, no cross-row reduction | Deterministic (rbits=0)                  | TMA double-buffering; same tiling as others |
| 1D, SR on  | Same as above                          | Same as above                                           | Same as above            | Same as above                                         | Stochastic via per-thread Philox `rbits` | Same as above                               |
| 2D, SR off | 2D blocks of 16×16                     | 8 × 8 = 64 per chunk (rowwise), 64 colwise              | `[rows / 16, cols / 16]` | Warp-level reduction over 256 vals per block          | Deterministic (rbits=0)                  | Same TMA pattern; more on-chip reduction    |
| 2D, SR on  | Same 2D blocks                         | Same as 2D, SR off                                      | Same as 2D, SR off       | Same as 2D, SR off                                    | Stochastic via `rbits`                   | Same TMA; extra RNG ops                     |

**Key differences:**

* **1D vs 2D:**

  * 1D: more scales, finer granularity along feature dimension; no warp-level reduction across multiple rows.
  * 2D: fewer scales, coarser 16×16 blocks; more shared/warp reduction; potentially better compression but slightly worse per-element fidelity in highly anisotropic patterns.
* **SR off vs on:**

  * SR off: pure deterministic rounding, easier to debug, slight bias.
  * SR on: unbiased rounding, slightly more instructions (RNG + extra PTX logic), typically negligible performance impact in a TMA-heavy kernel.
* **RETURN_TRANSPOSE = true:**

  * Always triggers an additional colwise path.
  * Uses the **same TMA tile** as rowwise, reinterpreted as columns.
  * Uses shared memory to produce a second FP4 layout with coalesced global stores.

---

## 5. Optimization Patterns to Remember

Across all modes (1D/2D, SR on/off, RETURN_TRANSPOSE = true), the kernel leans on a few critical optimization patterns:

1. **TMA-based double-buffering**

   * Large `32 × 128` tiles moved via `cp.async.bulk.tensor.2d.*` into shared.
   * Two shared buffers (`BUFFS_NUM = 2`) allow prefetching tile `s+1` while computing on tile `s`.
   * `mbarrier` + `fence_proxy_async_shared_cta` synchronize TMA and SM.

2. **Thread layout matched to block scales**

   * 1D mode: each thread owns exactly one 16-element block horizontally (and vertically in transpose view), which maps 1:1 to `SCALE_DIM`.
   * 2D mode: warp ownership of 16×16 blocks ensures local communication for amax reduction is cheap (warp intrinsics only).

3. **Shared-memory swizzling for bank conflict avoidance**

   * `bank_group` and `swizzled_group_idx` reorder accesses so consecutive FP4 packs map onto different banks.
   * This is crucial given the high volume of shared-memory traffic during quantization and transpose.

4. **Vectorized loads/stores and packed FP4 types**

   * `Vec<IType, PACK_SIZE>` and `fp4e2m1x4` ensure each thread handles 4–8 values per instruction.
   * Packed types align well with tensor-core FP4 micro-ops used later in GEMMs.

5. **Fused scale + convert PTX ops**

   * `mul_cvt_*_to_fp4_4x<...>` does multiply + convert + pack in a tight inline PTX sequence.
   * Reduces instruction count and register shuffling vs separate ops.

6. **Stochastic rounding as a lightweight extension**

   * Philox-based RNG is purely local; no inter-thread dependence.
   * `rbits` parameter in PTX allows adding randomness without restructuring the pipeline.

Altogether, the kernel is architected so that **changing scaling dimensionality (1D vs 2D) or rounding mode (deterministic vs stochastic)** only alters **on-chip math and scale layouts**, while leaving the **global memory pipeline** (TMA tiling, double-buffering, and transpose) unchanged. This keeps the implementation relatively modular and makes it easier to plug into different NVFP4 quantization strategies without compromising throughput.
