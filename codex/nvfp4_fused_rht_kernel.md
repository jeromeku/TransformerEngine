# NVFP4 Fused RHT + Columnwise Quantization Kernel  

**Kernel family**: `hadamard_transform_cast_fusion_columnwise` → `detail::rht_gemm_ttt_wrapper` → `rht_gemm_device`  
**Role**: Apply Random Hadamard Transform (RHT) to activations and **directly** quantize into NVFP4 columnwise layout (including scales), with optional stochastic rounding.

This document is a companion to  
[`nvfp4_quantize_v2.md`](./nvfp4_quantize_v2.md), which covers the standalone NVFP4
quantize+transpose kernels. Here we focus on the **fused RHT+quantization path** used
for columnwise NVFP4 data when `with_rht=True` and `columnwise_usage=True`.

All links below are **relative to this file**.

---

## 0. Key Files and Functions

- **Fused RHT + quantization (host + device wrapper):**  
  `transformer_engine/common/hadamard_transform/hadamard_transform_cast_fusion.cu`
  - Host API used by the NVFP4 quantizer:  
    `hadamard_transform_cast_fusion_columnwise`  
    [`hadamard_transform_cast_fusion.cu:704–780`](../transformer_engine/common/hadamard_transform/hadamard_transform_cast_fusion.cu#L704-L780)
  - GEMM‑style wrapper that launches the device kernel:  
    `detail::rht_gemm_ttt_wrapper`  
    [`hadamard_transform_cast_fusion.cu:672–716`](../transformer_engine/common/hadamard_transform/hadamard_transform_cast_fusion.cu#L672-L716)
  - TMA + MMA setup for the device kernel:  
    `detail::rht_gemm_ntt_w_sfc` and `rht_gemm_device` (same file, above the snippet).

- **NVFP4 quantizer integration (where this fused kernel is called):**  
  `transformer_engine/pytorch/csrc/quantizer.cpp`
  - Columnwise path in `NVFP4Quantizer::quantize_impl`:  
    [`quantizer.cpp:1560–1685`](../transformer_engine/pytorch/csrc/quantizer.cpp#L1560-L1685)

- **Hadamard amax helper (for pre‑RHT / transposed amax):**  
  `transformer_engine/common/hadamard_transform/hadamard_transform.cu`
  - `HadamardAmaxTmaKernel` / `ComputeKernel`:  
    [`hadamard_transform.cu:20–180`](../transformer_engine/common/hadamard_transform/hadamard_transform.cu#L20-L180)

---

## 1. High‑Level Design (Fused Columnwise Path)

### 1.1 Where the Fused Kernel Fits

When we request both **rowwise and columnwise NVFP4** for a tensor with RHT enabled:

- Rowwise view is produced by the **standard NVFP4 kernels** documented in  
  [`nvfp4_quantize_v2.md`](./nvfp4_quantize_v2.md).
- Columnwise view can be produced in one of two ways:
  - **Fallback:** `nvte_hadamard_transform(input)` → BF16 `RHT(xᵀ)` → `nvte_quantize_v2` → NVFP4 columnwise.
  - **Fused:** `nvte_hadamard_transform_cast_fusion_columnwise(input, out_transpose, rht_matrix, quant_config)`  
    → RHT + per‑block amax + NVFP4 encode + scale computation in **one GEMM‑like kernel**.

The fused path avoids materializing the intermediate BF16 `RHT(xᵀ)` buffer and fuses:

- TMA‑based BF16 tile loads from `input`.
- TMA‑based BF16 tile loads from the Hadamard matrix `H` (16×16).
- Tensor Core MMA (`H` × BF16 tiles) to compute RHT(x) or RHT(xᵀ).
- Per‑block amax accumulation.
- NVFP4 scaling and encode to FP4 (`TC = cutlass::float_e2m1_t`).
- Writing per‑block scale factors (`TSFC = float_ue4m3_t`) and global amax.
- Optional stochastic rounding using per‑thread RNG state.

### 1.2 Data Layouts for Columnwise RHT Path

For an input activation tensor `x` with logical shape `[M, K]` (e.g. `M=1024`, `K=768`):

- **Input (`input_`):**
  - BF16, delayed scaling, arbitrary batch/sequence dims collapsed into `m`:
    - `ndim = input_.shape().size()`
    - `n = input.shape[ndim‑1]`  (inner dimension, multiple of 16)
    - `m = prod_{i<ndim‑1} input.shape[i]`
  - We can think of it as `[m, n]`.

- **Hadamard matrix (`hadamard_matrix_`):**
  - BF16, shape `[16, 16]`, representing the Hadamard transform on 16‑element blocks.

- **Output (`output_` for columnwise view):**
  - `output_.data` (`output_t`): FP4 NVFP4 data (columnwise), *conceptually* `[n, m]` in row‑major.
  - `output_.scale_inv` (`scale_inv_t`): FP8 per‑block **decode** scales, shape roughly `[n, m/16]`.
  - `output_.amax` (`global_amax`): single FP32 amax over all RHT‑transformed values.

Internally, `rht_gemm_ttt_wrapper` swaps `(m, n)` to match the columnwise, transposed RHT layout:

```cpp
// A: n x m (col-major), B: 16 x 16 (row-major)
// C: n x m (row-major),  SFC: n x (m/16) (row-major)
rht_gemm_ntt_w_sfc<TA, TB, TC, TSFC, kEnableStochasticRounding>(
    n, m,
    A, B, C,
    SFC, global_amax,
    rng_state,
    sm_count, stream,
    k_tile_size);
```

So the fused kernel computes something like:

```text
for each 16-wide block along inner dimension:
    C_block = RHT(A_block)  // Hadamard * A
    SFC_block = per-block decode scale
    C_block_fp4 = encode_NVFP4(C_block / SFC_block)
```

where `C_block_fp4` is stored in the **columnwise NVFP4 layout** used by cuBLAS for RHT‑ready NVFP4 GEMMs.

---

## 2. Host Entry: `hadamard_transform_cast_fusion_columnwise`

**Source**:  
[`hadamard_transform_cast_fusion.cu:704–780`](../transformer_engine/common/hadamard_transform/hadamard_transform_cast_fusion.cu#L704-L780)

### 2.1 Argument Checking and Unpacking

```cpp
void hadamard_transform_cast_fusion_columnwise(const Tensor &input_, Tensor &output_,
                                               const Tensor &hadamard_matrix_,
                                               QuantizationConfig quant_config,
                                               cudaStream_t stream) {
  NVTE_API_CALL(hadamard_transform_cast_fusion_columnwise);
```

- Registers the API call for logging / profiling.
- Parameters:
  - `input_`  : BF16 activations (`NVTE_DELAYED_TENSOR_SCALING`).
  - `output_` : NVFP4 columnwise tensor (FP4 data + FP8 scales + amax).
  - `hadamard_matrix_` : 16×16 BF16 Hadamard matrix.
  - `quant_config` : holds `stochastic_rounding` flag and RNG state pointer.

```cpp
  // Check input and output tensors
  NVTE_CHECK(input_.scaling_mode == NVTE_DELAYED_TENSOR_SCALING,
             "Input tensor must be BF16 tensor, but scaling mode is ",
             to_string(input_.scaling_mode), ".");
  NVTE_CHECK(input_.dtype() == transformer_engine::DType::kBFloat16,
             "Input tensor must be BF16 tensor, but dtype is ", to_string(input_.dtype()), ".");
  NVTE_CHECK(input_.dim() >= 2, "Input must be a 2D tensor.");
  const SimpleTensor &input = input_.data;
  SimpleTensor &global_amax = output_.amax;
  SimpleTensor &output_t = output_.data;
  SimpleTensor &scale_inv_t = output_.scale_inv;
```

- Ensures we have BF16 input with delayed scaling and at least 2D shape.
- Binds:
  - `input`      : raw BF16 data + shape `[d0, ..., d_{ndim-1}]`.
  - `global_amax`: FP32 scalar for overall amax of RHT(x).
  - `output_t`   : FP4 data buffer for columnwise NVFP4.
  - `scale_inv_t`: FP8 decode scales buffer.

### 2.2 Stochastic Rounding Config and RNG

```cpp
  // Stochastic rounding config
  const bool use_stochastic_rounding = quant_config.stochastic_rounding;
  const size_t *rng_state = nullptr;
  if (quant_config.rng_state != nullptr) {
    Tensor &rng_state_tensor = *convertNVTETensor(quant_config.rng_state);
    NVTE_CHECK(rng_state_tensor.dtype() == DType::kInt64,
               "RNG state should contain 2 64-bit values.");
    NVTE_CHECK(rng_state_tensor.data.shape == std::vector<size_t>{2},
               "Shape of the RNG state should be [2], but got ", rng_state_tensor.data.shape);
    rng_state = reinterpret_cast<const size_t *>(rng_state_tensor.data.dptr);
  }
```

- `rng_state` is a device pointer to `[seed, offset]` (2× int64), used inside the device kernel to seed Philox streams.
- `use_stochastic_rounding` selects the template parameter `kEnableStochasticRounding` in `rht_gemm_ttt_wrapper` → `rht_gemm_device`.

### 2.3 Type Aliases and Hadamard Matrix Checks

```cpp
  // Template arguments
  using TA = cute::bfloat16_t;
  using TB = cute::bfloat16_t;
  using TC = cutlass::float_e2m1_t;
  using TSFC = cutlass::float_ue4m3_t;

  checkCuDriverContext(stream);

  // Check Hadamard matrix
  constexpr int kHadamardDimension = 16;
  NVTE_CHECK(hadamard_matrix_.scaling_mode == NVTE_DELAYED_TENSOR_SCALING,
             "Hadamard matrix must be BF16 tensor, but scaling mode is ",
             to_string(hadamard_matrix_.scaling_mode), ".");
  NVTE_CHECK(hadamard_matrix_.dtype() == transformer_engine::DType::kBFloat16,
             "Hadamard matrix must be BF16 tensor, but dtype is ",
             to_string(hadamard_matrix_.dtype()), ".");
  const SimpleTensor &hadamard_matrix = hadamard_matrix_.data;
  NVTE_CHECK(
      (hadamard_matrix_.shape() == std::vector<size_t>{kHadamardDimension, kHadamardDimension}),
      "Hadamard matrix must have shape=",
      std::vector<size_t>{kHadamardDimension, kHadamardDimension},
      ", but got shape=", hadamard_matrix_.shape(), ".");
  const size_t hadamard_dimension = hadamard_matrix.shape[0];
```

- `TA` / `TB` = BF16 (Cute’s `bfloat16_t`).
- `TC` = `float_e2m1_t` (FP4 E2M1) → matches NVFP4 data type.
- `TSFC` = `float_ue4m3_t` (FP8 E4M3 unsigned exponent) → used for per‑block decode scales.
- The Hadamard matrix is strictly 16×16; the last dimension of `input` and the flattened batch dimension must both be multiples of 16.

### 2.4 Flattening Shape and Tile Size Selection

```cpp
  const size_t ndim = input.shape.size();
  const size_t n = input.shape[ndim - 1];
  size_t m = 1;
  for (size_t i = 0; i < ndim - 1; ++i) {
    m *= input.shape[i];
  }

  auto sm_count = transformer_engine::cuda::sm_count();

  NVTE_CHECK(n % hadamard_dimension == 0, "row_length must be divisible by hadamard_dimension.");

  NVTE_CHECK(m % hadamard_dimension == 0, "num_rows must be divisible by hadamard_dimension");
```

- `n` is the **inner dimension** (e.g. hidden size), must be divisible by 16.
- `m` is the product of all leading dimensions (e.g. batch × sequence), also divisible by 16.

Tile size heuristics:

```cpp
  int k_tile_size = 1024;

  if (m == 8192 && n == 5120) {
    k_tile_size = 512;
  } else if (m == 8192 && n == 10240) {
    k_tile_size = 1024;
  } ...
  else if (m < 1024 || n < 1024) {
    k_tile_size = 512;
  }
```

- Chooses `k_tile_size` (how many columns of `C` each kernel instance accumulates) based on `(m, n)` to balance:
  - The number of tiles (`tiles`).
  - SM occupancy (`sm_count`).
  - The size of K‑tiles along the inner dimension.

### 2.5 Launching the Fused GEMM‑style Kernel

```cpp
  TRANSFORMER_ENGINE_SWITCH_CONDITION(
      use_stochastic_rounding, kUseStochasticRounding,
      detail::rht_gemm_ttt_wrapper<TA, TB, TC, TSFC, kUseStochasticRounding>(
          /*m=*/m,
          /*n=*/n,
          /*A=*/reinterpret_cast<TA const *>(input.dptr),
          /*B=*/reinterpret_cast<TB const *>(hadamard_matrix.dptr),
          /*C=*/reinterpret_cast<TC *>(output_t.dptr),
          /*SFC=*/reinterpret_cast<TSFC *>(scale_inv_t.dptr),
          /*global_amax=*/reinterpret_cast<float const *>(global_amax.dptr),
          /*rng_state=*/rng_state,
          /*sm_count=*/sm_count,
          /*stream=*/stream,
          /*k_tile_size=*/k_tile_size););
}
```

- Uses a macro to instantiate either:
  - `kEnableStochasticRounding=false` (no SR).
  - `kEnableStochasticRounding=true` (SR enabled).
- Passes:
  - `A` = BF16 activations (`TA*`).
  - `B` = BF16 Hadamard matrix (`TB*`).
  - `C` = FP4 NVFP4 columnwise data (`TC*`).
  - `SFC` = FP8 decode scales array (`TSFC*`).
  - `global_amax` = pointer to single FP32 amax.
  - `rng_state` = RNG seed/offset.
  - `sm_count` & `k_tile_size` = used to choose grid shape.

---

## 3. Wrapper: `detail::rht_gemm_ttt_wrapper`

**Source**:  
[`hadamard_transform_cast_fusion.cu:672–716`](../transformer_engine/common/hadamard_transform/hadamard_transform_cast_fusion.cu#L672-L716)

```cpp
template <typename TA, typename TB, typename TC, typename TSFC, bool kEnableStochasticRounding>
void
rht_gemm_ttt_wrapper(int m, int n,
        TA const* A,
        TB const* B,
        TC      * C,
        TSFC    * SFC,
        float const* global_amax,
        const size_t* rng_state,
        uint32_t sm_count,
        cudaStream_t stream,
        int k_tile_size)
{
  // in addition to transpose the input tensor A
  // we also need to reshape m, n to at best
  // utilize as many SMs as possible while keeping
  // a relatively large contiguous dimension.
  // for example, after swapping m, n for transpose purposes,
  // the input / output tensor shapes for RHT-GEMM are:
  // A: n x m: col-major
  // B: 16 x 16: row-major
  // C: n x m: row-major
  // SFC: n x (m/16): row-major
  rht_gemm_ntt_w_sfc<TA, TB, TC, TSFC, kEnableStochasticRounding>(
    n, m,
    A, B, C,
    SFC, global_amax,
    rng_state,
    sm_count, stream,
    k_tile_size);
}
```

Key points:

- **Transpose semantics**:
  - The fused kernel is logically computing `RHT(xᵀ)` in a columnwise layout.
  - It swaps `(m, n)` and treats `A` as an `n×m` matrix in **column‑major** order.
  - `C` is `n×m` in **row‑major**; this matches the NVFP4 columnwise layout expected by cuBLAS.
  - `SFC` holds per‑block decode scales: shape `n × (m/16)`.

- All heavy lifting is done inside `rht_gemm_ntt_w_sfc`, which constructs the TMA pipeline,
  shared‑memory layouts, and launches `rht_gemm_device`.

---

## 4. Device Kernel Setup: `rht_gemm_ntt_w_sfc` → `rht_gemm_device`

**Source (excerpt)**:  
[`hadamard_transform_cast_fusion.cu:560–640`](../transformer_engine/common/hadamard_transform/hadamard_transform_cast_fusion.cu#L560-L640)

This section concentrates on how the wrapper prepares **tiling, TMA loads, and MMA mapping**,
because that determines how data moves through the kernel.

### 4.1 Tiling and CGA Shape

```cpp
  auto M = Int<m>{};
  auto N = Int<n>{};

  // TMA tile shape (M∙N∙K)
  auto cga_shape = Shape<_1,_1,_1>{}; // 1x1x1 cluster (per-kernel launch)
  auto cga_tile_shape = Shape<_128,_16,_16>{};
  auto cluster_tile_mainloop = Shape<_128,_16,_64>{};
```

- `cga_tile_shape = (128, 16, 16)`:
  - Each MMA tile covers:
    - `M_tile = 128` rows of `C` (and `A`).
    - `N_tile = 16` columns of `C`.
    - `K_tile = 16` elements along the inner dimension per Hadamard block.
- `cluster_tile_mainloop = (128, 16, 64)`:
  - Captures how many `K_tile` slices are processed together in the main loop:
    - Up to `64` along K per inner iteration (4 × 16 blocks).

The **CGA cluster** is currently `1×1×1`, but the layout is written generically so multiple CTAs could cooperate on a tile if needed.

### 4.2 MMA Construction and Shared‑Memory Layout

```cpp
  auto mma = make_tiled_mma(SM100_MMA_F16BF16_SS<TA, TB, float,
                                               128, 16,
                                               UMMA::Major::MN, UMMA::Major::MN>{},
                            Layout<Shape<_1,_1>>{});
```

- Uses **SM100** tensor cores in BF16×BF16→FP32 accumulation mode:
  - `TA` / `TB` = BF16 inputs (A, B).
  - Accumulator type = FP32.
  - Tile shape = 128×16×16 in (M×N×K).
  - Layout `UMMA::Major::MN` ensures that M and N are the “major” axes for tiling.

Shared‑memory layout shapes:

```cpp
  auto mma_shape_B = partition_shape_B(mma, make_shape(size<1>(cga_tile_shape), size<2>(cga_tile_shape)));

  using TiledMma = decltype(mma);
  using AtomThrID = typename TiledMma::AtomThrID;

  using SmemShape_M = decltype(shape_div(shape<0>(cga_tile_shape), shape_div(shape<0>(cga_tile_shape), size<0>(cga_tile_shape) / size(AtomThrID{}))));
  using SmemShape_N = decltype(shape_div(shape<1>(cga_tile_shape), shape_div(shape<1>(cga_tile_shape), size<1>(cga_tile_shape) / size(AtomThrID{}))));
  using SmemShape_K = decltype(cute::get<2>(cga_tile_shape));
```

- `SmemShape_M`, `SmemShape_N`, `SmemShape_K` describe how the MMA fragments are laid out in shared memory for `A` and `B`.
- These are sized so that:
  - Each warp (and each thread within a warp) gets a **contiguous slice** of the A/B tiles.
  - Shared‑memory accesses are coalesced and conflict‑free.

Cutlass helper for shared‑memory layout:

```cpp
  using SmemLayoutAtomB = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
      cute::UMMA::Major::MN, TB, SmemShape_N, SmemShape_K>());
```

- `SmemLayoutAtomB` encodes the bank‑friendly layout for `B` (Hadamard matrix tiles) in shared memory, matching the MMA’s expectations.

Similar for `A`:

```cpp
  auto mma_shape_A = partition_shape_A(mma, make_shape(size<0>(cluster_tile_mainloop), size<2>(cluster_tile_mainloop)));
  using SmemShape_M_A = decltype(shape_div(shape<0>(cluster_tile_mainloop), shape_div(shape<0>(cluster_tile_mainloop), size<0>(cluster_tile_mainloop) / size(AtomThrID{}))));
  using SmemShape_K_A = decltype(cute::get<2>(cluster_tile_mainloop));
  using SmemLayoutAtomA = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
      cute::UMMA::Major::MN, TA, SmemShape_M_A, SmemShape_K_A>());
```

- `SmemLayoutAtomA` matches `TA` (BF16 activations) to the MMA tile shape for `A`.

### 4.3 TMA Loaders for A and B

```cpp
  // Create GMEM tensors
  Tensor tensorA = make_tensor(A, make_layout(make_shape(M,N), dA));      // (M,N)
  Tensor tensorB = make_tensor(B, make_layout(make_shape(16,16), dB));    // (16,16)

  // Create the TiledCopy
  auto tma_load_a = make_tma_copy_A_sm100(
        SM90_TMA_LOAD{},
        tensorA,
        sA(_,_,_,0),
        cluster_tile_mainloop,
        mma);
  auto tma_load_b =  make_tma_copy_B_sm100(
        SM90_TMA_LOAD{},
        tensorB,
        sB(_,_,_,0),
        cga_tile_shape,
        mma);
```

- `tensorA` is the BF16 `[M, N]` view of `A`.
- `tensorB` is the 16×16 Hadamard matrix.
- `tma_load_a` / `tma_load_b` set up **TMA descriptors** that:
  - Copy rectangular tiles from `A`/`B` in global memory into the corresponding shared‑memory layout (`sA`, `sB`).
  - Use the same tiling as the MMA (`cga_tile_shape` / `cluster_tile_mainloop`).

**Data movement pattern:**

- For each tile:
  1. TMA engines load `A_tile` (size up to 128×K_chunk) into shared memory using `tma_load_a`.
  2. `B` (Hadamard matrix) is loaded once into shared memory (`tma_load_b`), then reused across tiles.
  3. A pipeline of TMA loads and MMAs keeps both compute and memory busy with double‑buffering in shared memory.

### 4.4 Launch Geometry and Shared‑Memory Size

```cpp
  // Assert checks on tile sizes -- no predication
  NVTE_CHECK(M % size<0>(cga_tile_shape) == 0,
             "Inner dimension must be divisible by ", static_cast<size_t>(size<0>(cga_tile_shape)), " but got ", M, ".");
  NVTE_CHECK(N % (4 * size<1>(cga_tile_shape)) == 0,
             "Outer dimension must be divisible by ", 4 * static_cast<size_t>(size<1>(cga_tile_shape)),
             " but got ", N, ".");

  uint32_t tiles = size(ceil_div(M, get<0>(cga_tile_shape))) * size(ceil_div(N, k_tile_size));

  tiles = (tiles < sm_count) ? tiles : sm_count;

  dim3 dimBlock(256);
  dim3 dimCluster(size<0>(cga_shape), size<1>(cga_shape), size<2>(cga_shape));
  dim3 dimGrid(tiles, 1, 1);

  int smem_size = sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>);
```

- The tile size constraints ensure **no predication** on the MMA:
  - `M` is divisible by 128.
  - `N` is divisible by `4×16 = 64` (because of packing / K dimension).
- `tiles` is the number of tiles along `(M, N/k_tile_size)` combined; capped by `sm_count`.
  - Each tile is mapped to one CTA (`dimBlock=256` threads).
- `smem_size` is determined by `SharedStorage` (holds `sA` and `sB` double‑buffered).

Launch:

```cpp
  auto* kernel_ptr = &rht_gemm_device</*many template params...*/, kEnableStochasticRounding>;

  bool status = cudaFuncSetAttribute(*kernel_ptr,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                smem_size);
  ...
  (*kernel_ptr)
      <<< dimGrid, dimBlock, smem_size, stream >>>
      (M,  N,  k_tile_size, cga_tile_shape,
       A, dA, sA, tma_load_a,
       B, dB, sB, tma_load_b,
       C, dC, sC,
       SFC,
       mma, global_amax,
       rng_state);
```

- Each CTA:
  - Processes a strip of `N` up to `k_tile_size` wide and `M` rows tall.
  - Uses 256 threads to drive TMA loads, MMA, and NVFP4 encode.
  - Stores results into:
    - `C` (FP4 data, `TC`).
    - `SFC` (FP8 scales).
    - `global_amax` (atomic or reduction writes).

---

## 5. Dataflow and Thread Mapping (Conceptual)

The full `rht_gemm_device` body is heavily template‑driven, but its dataflow mirrors a
standard CUTLASS GEMM with extra NVFP4 encode logic:

### 5.1 Conceptual Per‑CTA Flow

For each CTA (tile index `t`):

1. **Tile selection**:
   - Determine `(m0, m1)` and `(n0, n1)` ranges for this tile (`M_tile=128`, `N_chunk=k_tile_size`).
2. **TMA load A**:
   - Issue TMA to load BF16 `A[m0:m1, k0:k1]` into `sA` double buffer (K tiles).
3. **TMA load B**:
   - Once per kernel or per CTA, load BF16 `B[16×16]` into `sB`.
4. **MMA loop (RHT)**:
   - For each K‑slice:
     - All warps cooperatively:
       - Load A/B fragments from `sA`/`sB` into registers according to `AtomThrID` mapping.
       - Execute tensor core MMA:
         ```text
         C_frag += A_frag × B_frag   // A: activations, B: Hadamard
         ```
       - Accumulate FP32 results in `C_frag`.
5. **Per‑block amax & NVFP4 scaling**:
   - For each 16‑wide output block in `C_frag`:
     - Compute block‑wise amax (max |value| over 16 elements).
     - Combine with global amax and compute decode scale `S_dec_b` (FP8, `TSFC`).
     - Convert FP32 values to FP4 (`TC=E2M1`) with:
       - Scaling by `1 / S_dec_b`.
       - Optional SR using random bits from `rng_state`.
   - Write FP4 data into `C` in NVFP4 columnwise layout.
   - Write `S_dec_b` into `SFC` for this block.
6. **Global amax update**:
   - Each CTA writes its block amax contributions into `global_amax` (e.g., via atomic max or reduction inside the kernel).

### 5.2 Thread → Value Mapping

Although the exact warp layout is encoded in the CUTLASS `TiledMma`, the mapping roughly
follows:

- 256 threads per CTA, organized as 8 warps.
- Each warp is responsible for:
  - A strip of `M_tile` rows and `N_subtile` columns.
  - Loading contiguous slices of A and B from shared memory into registers.
- The cutlass `sm100_smem_selector` layouts guarantee that:
  - Threads of a warp access **contiguous or strided‑one** locations in shared memory.
  - All loads/stores are aligned to 16‑byte boundaries and avoid shared‑memory bank conflicts.

This is analogous to the NVFP4 quantize+transpose kernel’s **swizzled index** logic, but
here handled by CUTLASS’s SM100 helpers instead of hand‑rolled indexing.

---

## 6. Summary: Fused RHT vs Plain NVFP4 Quantize+Transpose

- **Plain NVFP4 kernels** (`quantize_transpose_nvfp4_kernel` and `_2D_kernel`):
  - Input: BF16 tiles (no RHT inside the kernel).
  - Compute per‑block amax, FP8 decode scales, FP4 encode for both rowwise and columnwise views.
  - Use custom TMA kernel with 128×128 tiles, 32×128 inner tiles, and manual swizzle.

- **Fused RHT kernel** (`hadamard_transform_cast_fusion_columnwise` → `rht_gemm_device`):
  - Input: BF16 `[m, n]` (flattened leading dims).
  - Performs:
    - RHT via BF16×BF16→FP32 Tensor Core MMAs with the 16×16 Hadamard matrix.
    - Per‑block amax + FP8 decode scale computation.
    - FP4 encode (NVFP4) with optional stochastic rounding.
    - Writes FP4 data + scales directly into **columnwise** NVFP4 layout.
  - Uses CUTLASS SM100 TMA + MMA infrastructure for:
    - High bandwidth global↔shared transfers (TMA).
    - High occupancy and high‑throughput tensor core usage.

From the NVFP4 quantizer’s perspective:

- Rowwise data: produced by `nvte_quantize_v2` → NVFP4 quantize+transpose kernels.
- Columnwise data (with RHT): produced either by:
  - Fallback RHT + NVFP4 quantize+transpose, or
  - This fused RHT GEMM kernel.

In all cases, the **GEMM kernels that consume columnwise NVFP4** see the same layout and
scales; the fused path simply provides a more efficient way to generate those tensors
while embedding the RHT step into the NVFP4 encode process.

---

## 7. Comparison with `HadamardAmaxTmaKernel`

The fused RHT kernel plays a similar “Hadamard + TMA + tensor core” game as
`HadamardAmaxTmaKernel`, but they solve **different problems** and therefore make
different choices in dataflow and implementation style.

### 7.1 Roles

- **Fused RHT + NVFP4 kernel**  
  - Entry: `hadamard_transform_cast_fusion_columnwise`  
    [`hadamard_transform_cast_fusion.cu:704–780`](../transformer_engine/common/hadamard_transform/hadamard_transform_cast_fusion.cu#L704-L780)  
  - Core: `detail::rht_gemm_ttt_wrapper` → `rht_gemm_ntt_w_sfc` → `rht_gemm_device`.  
  - Goal: compute `RHT(xᵀ)` and **materialize** the result as columnwise NVFP4:
    - BF16 input → RHT → FP32 accumulators → FP4 NVFP4 data + FP8 per-block scales + global amax.

- **HadamardAmaxTma kernel**  
  - `HadamardAmaxTmaKernel<IType,...>` and `ComputeKernel`  
    [`hadamard_transform.cu:20–180`](../transformer_engine/common/hadamard_transform/hadamard_transform.cu#L20-L180)  
  - Goal: compute **amax statistics only**:
    - Pre‑RHT amax, RHT(identity) amax, and RHT(transposed) amax.
    - It does not produce any transformed matrix; it only returns scalars.

### 7.2 Data Movement Patterns

**HadamardAmaxTma**

- Uses TMA with a CUtensorMap to stream BF16 tiles of the input into shared memory:
  - `copy_2d_to_shared(in_shs[0], tensor_map_input, ...)`  
    [`hadamard_transform.cu:120–140`](../transformer_engine/common/hadamard_transform/hadamard_transform.cu#L120-L140)
- Double‑buffered shared memory (`in_shs[2]`) + an array of barriers (`mbar`) + per‑warp staging
  buffers (`max_staging_identity`, `max_staging_transpose`, `max_staging_pre_rht`).
- For each tile:
  1. TMA fills `in_shs[buff]` with BF16 data.
  2. All warps run `ComputeKernel`:
     - Load BF16 fragments via `ldmatrix_x4_m8n8_shared_b16` from shared memory.
     - Apply Hadamard via tensor‑core MMA for the identity/transposed variants.
     - Update per‑thread amax registers.
  3. Reduce per‑thread amax across warp and block; finally write three FP32 scalars to global
     (pre‑RHT, identity, transposed amax).
- **No transformed matrix** is ever written to global memory; only amax accumulators leave the SM.

**Fused RHT GEMM (`rht_gemm_device`)**

- Also relies on TMA, but via CUTLASS/Cute:
  - `make_tma_copy_A_sm100(...)` and `make_tma_copy_B_sm100(...)` set up TMA descriptors for
    A (BF16 activations) and B (BF16 Hadamard matrix)  
    [`hadamard_transform_cast_fusion.cu:560–600`](../transformer_engine/common/hadamard_transform/hadamard_transform_cast_fusion.cu#L560-L600).
- For each CTA:
  1. TMA streams tiles of `A` (`M×N`) and `B` (16×16) into shared memory layouts `sA`, `sB`.
  2. Tensor cores compute RHT fragments into FP32 accumulators (`C_frag`).
  3. Per‑block amax is computed from `C_frag`; FP8 decode scales (`SFC`) are written.
  4. FP32 results are converted to FP4 NVFP4 and written into `C` in **columnwise NVFP4 layout**.
  5. A single FP32 `global_amax` is updated from all tiles.

**Summary of data movement**

- `HadamardAmaxTma`:
  - Global → shared: BF16 tiles (input).
  - Shared → registers: BF16 fragments for MMA.
  - Registers → global: **only a few FP32 scalars** (amax values).

- Fused RHT GEMM:
  - Global → shared: BF16 input tiles + BF16 Hadamard.
  - Shared → registers: fragments for MMA and NVFP4 encode.
  - Registers/shared → global: FP4 NVFP4 columnwise data + FP8 scales + one FP32 global amax.

### 7.3 Compute Patterns

**HadamardAmaxTma (`ComputeKernel`)**

- Per tile:
  - Loads BF16 fragments into `a_frag[4]` from shared memory:
    ```cpp
    ldmatrix_x4_m8n8_shared_b16<false>(a_frag[0], a_frag[1], a_frag[2], a_frag[3],
                                       reinterpret_cast<uint4*>(in_sh_ptr) + swizzle_idx);
    ```
  - For **identity amax**:
    ```cpp
    mma_m16_n16_k16_b16_b16_b16_noacc<kReturnIdentityAmax>(
        a_frag[0], a_frag[1], a_frag[2], a_frag[3],
        b_frag_i[0], b_frag_i[1], b_frag_i[2], b_frag_i[3],
        c_frag[0], c_frag[1], c_frag[2], c_frag[3],
        temp_amax_reg);
    asm volatile("max.xorsign.abs.bf16x2 %0, %1, %2;\n\t"
                 : "=r"(local_amax_reg)
                 : "r"(local_amax_reg), "r"(temp_amax_reg));
    ```
  - For **transposed amax**:
    - In‑register transpose of `a_frag` (`matrix_transpose_m8_n8_b16_inplace`).
    - Another MMA with transposed Hadamard fragments and amax update into `local_amax_t_reg`.
  - For **pre‑RHT amax**:
    - Simple max tree over BF16 elements in `a_frag` without MMA.
  - After all tiles:
    - Warp‑level reductions using shuffle.
    - Block‑level reductions into staging arrays.
    - Atomic `atomicMaxFloat` into three global FP32 amax pointers.

**Fused RHT GEMM**

- Per K‑tile:
  - Tensor cores execute BF16×BF16→FP32 SM100 MMAs via CUTLASS:
    ```cpp
    auto mma = make_tiled_mma(SM100_MMA_F16BF16_SS<TA, TB, float,
                                                 128, 16,
                                                 UMMA::Major::MN, UMMA::Major::MN>{},
                              Layout<Shape<_1,_1>>{});
    ```
  - A/B fragments are loaded from shared memory according to the MMA tiling.
- After MMA:
  - FP32 `C_frag` is processed per 16‑wide block:
    - Compute block amax.
    - Compute FP8 decode scale `S_dec_b` from block amax and `global_amax`.
    - Encode FP4 (NVFP4) with optional SR, writing FP4 into `C` and FP8 scales into `SFC`.
  - Global amax is updated from all block amaxes.

**Key difference**

- `HadamardAmaxTma` uses MMAs only as **temporary helpers** to generate values whose sole purpose is to feed amax reductions; no transformed matrix is kept.
- Fused RHT GEMM uses MMAs to compute the **actual operands** used later by GEMMs; every value is kept (after FP4 encode) and stored in columnwise NVFP4 layout.

### 7.4 Why CUTLASS for Fused RHT GEMM vs Inline PTX for HadamardAmaxTma

1. **Problem scope and complexity**
   - Fused RHT kernel:
     - Essentially a specialized GEMM on SM100:
       - Large `n×m` activations times a 16×16 Hadamard.
       - Requires careful control of TMA pipelines, multi‑stage shared memory, cluster shapes, and tile scheduling.
     - Needs near‑peak tensor‑core throughput across a wide range of shapes.
   - HadamardAmaxTma:
     - Small, fixed‑shape prepass whose only output is a few amax scalars.
     - The total FLOPs and memory traffic are relatively low; the kernel is short and highly specialized.

2. **Abstraction vs direct control**
   - Fused RHT GEMM:
     - CUTLASS/Cute gives:
       - SM100‑aware TMA and MMA tiling.
       - Proven bank‑conflict‑free shared layouts (`sm100_smem_selector`).
       - Easier tuning of tile sizes (`cga_tile_shape`, `k_tile_size`) and pipelines without rewriting PTX.
     - This is valuable for a kernel that sits on the **critical path of NVFP4 GEMMs**.
   - HadamardAmaxTma:
     - Inline PTX lets TE authors:
       - Precisely control instruction sequences (`ldmatrix`, `mma_m16_n16_k16`, `max.xorsign.abs`, `mbarrier`, `cp.async`).
       - Tailor the kernel to exactly the three amax flavors and tile sizes needed, with minimal overhead.

3. **Evolution and maintainability**
   - Fused RHT GEMM is written as a modern SM100 GEMM using the same infrastructure as other CUTLASS/TE GEMMs, so improvements in CUTLASS directly benefit this path.
   - HadamardAmaxTma predates or sits aside that infrastructure and is small enough that bespoke PTX is manageable and stable.

In short:

- The fused RHT kernel is a **full GEMM‑class kernel** whose complexity and performance requirements justify CUTLASS/Cute.
- `HadamardAmaxTma` is a **small, single‑purpose reduction kernel** where hand‑written PTX is sufficient and offers very fine‑grained control. 
