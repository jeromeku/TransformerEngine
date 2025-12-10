# TMEM Accumulator Pipeline on SM100 (RHT GEMM)  

This note explains why the fused RHT+NVFP4 kernel caps  
`AccumulatorPipelineStageCount` at 16 even though it allocates the full  
`128 x 512` TMEM capacity, and why the effective C‑accumulator fragment only
uses `128 x 256` of that space.

All links below are relative to this file.

---

## 1. Where the TMEM accumulator fragment comes from

In the fused RHT GEMM path we build a Blackwell UMMA kernel using
`SM100_MMA_F16BF16_SS` and Cute’s tiled MMA wrappers:

- MMA opcode and traits:  
  `3rdparty/cutlass/include/cute/arch/mma_sm100_umma.hpp`  
  `3rdparty/cutlass/include/cute/atom/mma_traits_sm100.hpp`
- RHT fused kernel (TE implementation):  
  `transformer_engine/common/hadamard_transform/hadamard_transform_cast_fusion.cu`
- Debug / playground version:  
  `experiments/rht_gemm.cu`

Inside the device kernel we construct the TMEM accumulator fragment with:

- In the TE kernel:  
  [`hadamard_transform_cast_fusion.cu:236–248`](../transformer_engine/common/hadamard_transform/hadamard_transform_cast_fusion.cu#L236-L248)

  ```cpp
  auto acc_shape_mma = partition_shape_C(TiledMMA{}, take<0,2>(ClusterTileShape{}));
  auto acc_shape_epilogue = partition_shape_C(TiledMmaEpilogue{}, take<0,2>(epilogue_tiler));

  auto bulk_tmem_mma = TiledMMA::make_fragment_C(
      append(acc_shape_mma, Int<AccumulatorPipelineStageCount>{}));

  auto bulk_tmem_epilogue = TiledMmaEpilogue::make_fragment_C(
      append(acc_shape_epilogue, Int<AccumulatorPipelineStageCount / 4>{}));
  ```

For the specific MMA in use (`M=128, N=16`), Cute infers the C‑fragment layout
based on the UMMA traits. The shape of `bulk_tmem_mma` in practice is:

```text
bulk_tmem_mma.shape == ((128,16), _1, _1, AccumulatorPipelineStageCount)
```

So for `AccumulatorPipelineStageCount = 16` we get a logical C‑fragment that
spans `128 x 16 x 16 = 128 x 256` TMEM elements.

The backing TMEM pointer is obtained by:

- In the TE kernel:  
  [`hadamard_transform_cast_fusion.cu:320–346`](../transformer_engine/common/hadamard_transform/hadamard_transform_cast_fusion.cu#L320-L346)

  ```cpp
  using TmemAllocator = cute::TMEM::Allocator1Sm;
  // ...
  tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns,
                          &shared_storage.tmem_base_ptr);
  __syncwarp();
  tmem_allocation_result_barrier.arrive();
  uint32_t tmem_base_ptr = shared_storage.tmem_base_ptr;
  bulk_tmem_mma.data() = tmem_base_ptr;
  ```

Here `Allocator1Sm::Sm100TmemCapacityColumns` is 512, so we are always
allocating the **full** 128×512 TMEM region for one SM and then pointing the
`bulk_tmem_mma` fragment descriptor at that base.

The same pattern is visible in the debug kernel, which prints the fragment
shapes:

- [`experiments/rht_gemm.cu:220–270`](../experiments/rht_gemm.cu#L220-L270)

  ```cpp
  auto acc_shape_mma = partition_shape_C(TiledMMA{}, take<0, 2>(ClusterTileShape{}));
  auto acc_shape_epilogue =
      partition_shape_C(TiledMmaEpilogue{}, take<0, 2>(epilogue_tiler));

  auto bulk_tmem_mma = TiledMMA::make_fragment_C(
      append(acc_shape_mma, Int<AccumulatorPipelineStageCount>{}));
  auto bulk_tmem_epilogue = TiledMmaEpilogue::make_fragment_C(
      append(acc_shape_epilogue, Int<AccumulatorPipelineStageCount / 4>{}));

  if (thread0()) {
    print_cute("acc_shape_mma", acc_shape_mma);
    print_cute("acc_shape_epilogue", acc_shape_epilogue);
    print_cute("bulk_tmem_mma", bulk_tmem_mma);
    print_cute("bulk_tmem_epilogue", bulk_tmem_epilogue);
  }
  ```

---

## 2. TMEM hardware capacity vs. fragment usage

The SM100 TMEM allocator encodes the hardware capacity as:

- [`tmem_allocator_sm100.hpp:32–44`](../3rdparty/cutlass/include/cute/arch/tmem_allocator_sm100.hpp#L32-L44)

  ```cpp
  // 128 DP x 512 COL x uint32_t-addressing
  using MAX_CAPACITY_BITS = Int<128*512*32>;
  ```

On the MMAs we use, the UMMA C‑fragment type is:

- [`mma_traits_sm100.hpp:1090–1108`](../3rdparty/cutlass/include/cute/atom/mma_traits_sm100.hpp#L1090-L1108)

  ```cpp
  template <class a_type, class b_type, class c_type,
            int M, int N, UMMA::Major a_major, UMMA::Major b_major,
            UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg>
  struct MMA_Traits<SM100_MMA_F16BF16_SS<a_type, b_type, c_type,
                                  M, N, a_major, b_major,
                                  a_neg, b_neg>> {
    using ValTypeC = c_type;
    using FrgTypeC = UMMA::tmem_frg_1sm<c_type>;
    // ...
  };
  ```

The `tmem_frg_1sm` helper builds a TMEM layout and checks that the requested
shape fits in `MAX_CAPACITY_BITS`:

- [`mma_traits_sm100.hpp:420–460`](../3rdparty/cutlass/include/cute/atom/mma_traits_sm100.hpp#L420-L460)

  ```cpp
  template <class ValueType, class StorageType, int N_SM, UMMA::TmemAllocMode TmemAlloc>
  struct tmem_frg : tmem_frg_base {
    template <class TmemShape>
    CUTE_HOST_DEVICE constexpr static auto
    make(TmemShape const& tmem_shape) {
      CUTE_STATIC_ASSERT_V(
          size(tmem_shape) * Int<int(sizeof_bits_v<StorageType>)>{}
          <= TMEM::MAX_CAPACITY_BITS{},
          "Requesting more TMEM than is available.");
      // ...
    }
  };
  ```

For our MMA (`M=128, N=16`) and `AccumulatorPipelineStageCount = 16`, the
total TMEM requested by the C‑fragment is:

```text
num_elements = 128 * 16 * AccumulatorPipelineStageCount
             = 128 * 16 * 16
             = 128 * 256
```

With a 32‑bit storage type, the bit requirement is:

```text
bits_used = 128 * 256 * 32
          = 128 * 512 * 16
          = 0.5 * MAX_CAPACITY_BITS
```

So the fragment only uses **half** of the available TMEM address space, even
though we allocated the full 128×512 region. From a pure hardware capacity
standpoint, we could go up to:

```text
128 * 16 * AccumulatorPipelineStageCount * 32 <= 128 * 512 * 32
⇒ AccumulatorPipelineStageCount <= 32
```

In other words, nothing in the TMEM capacity prohibits
`AccumulatorPipelineStageCount = 32`; the cap at 16 is a software / pipeline
design choice.

---

## 3. Why the pipeline stages are capped (4 effective stages)

The fused kernel does not use `AccumulatorPipelineStageCount` directly as the
pipeline depth. Instead it sets up the accumulator pipeline as:

- In the TE kernel:  
  [`hadamard_transform_cast_fusion.cu:287–304`](../transformer_engine/common/hadamard_transform/hadamard_transform_cast_fusion.cu#L287-L304)

  ```cpp
  static constexpr int AccumulatorPipelineStageCount = 16;

  using AccumulatorPipeline =
      cutlass::PipelineUmmaAsync<AccumulatorPipelineStageCount / 4,
                                 AtomThrShapeMNK>;
  using AccumulatorPipelineState = typename AccumulatorPipeline::PipelineState;
  ```

So the **pipeline depth** in `PipelineUmmaAsync` is:

```text
Stages = AccumulatorPipelineStageCount / 4 = 4
```

Down in the MMA loop, each pipeline stage owns **4 TMEM slots** in the last
dimension of `bulk_tmem_mma`:

- In the TE kernel:  
  [`hadamard_transform_cast_fusion.cu:338–355`](../transformer_engine/common/hadamard_transform/hadamard_transform_cast_fusion.cu#L338-L355)

  ```cpp
  for (int k_block = 0; k_block < size<2>(tCrA) / 4; ++k_block) {
    accumulator_pipeline.producer_acquire(accumulator_pipe_producer_state);
    CUTE_UNROLL
    for (int i = 0; i < 4; i++) {
      auto accumulators =
          bulk_tmem_mma(_,_,_,
                        accumulator_pipe_producer_state.index() * 4 + i);
      gemm(mma, tCrA_mk(_,_,k_block * 4 + i), tCrB_nk, accumulators);
    }

    accumulator_pipeline.producer_commit(accumulator_pipe_producer_state);
    ++accumulator_pipe_producer_state;
  }
  ```

Putting this together:

- `PipelineUmmaAsync` has `Stages = 4`.
- Each stage controls 4 distinct C‑slots in TMEM.
- Total logical accumulator slots in TMEM:

  ```text
  slots = Stages * 4 = 4 * 4 = 16
  ```

which is exactly `AccumulatorPipelineStageCount`. That is why
`bulk_tmem_mma` has a last dimension of size 16 even though the allocator
grabbed space for a potential 32‑stage pipeline.

This design mirrors the official CUTLASS SM100 GEMM builder, which
**deliberately** caps the accumulator pipeline stages:

- [`sm100_umma_builder.inl:252–266`](../3rdparty/cutlass/include/cutlass/gemm/collective/builders/sm100_umma_builder.inl#L252-L266)

  ```cpp
  static constexpr uint32_t TotalTmemRows = 128;
  static constexpr uint32_t Sm100TmemCapacityColumns = 512;
  static constexpr uint32_t TotalTmem = TotalTmemRows * Sm100TmemCapacityColumns;

  static constexpr uint32_t AccumulatorPipelineStageCount_ =
      (is_2sm || (!is_2sm && size(shape<0,0>(MmaShapeA_MK{}) > 64))) ?
        TotalTmem / (cute::size<0>(CtaTileShape_MNK{}) *
                     cute::size<1>(CtaTileShape_MNK{}))
      : (Sm100TmemCapacityColumns / cute::size<1>(CtaTileShape_MNK{})) * 2;

  // 4 accumulator stages works well to buffer the accumulators,
  // while also preventing overhead in the epilogue tail on small tile sizes.
  static constexpr uint32_t AccumulatorPipelineStageCount =
      cute::min(4u, AccumulatorPipelineStageCount_);
  ```

For our tile (`CtaTileShape_MNK = (128, 16, ...)`) and 1‑SM MMA,
`AccumulatorPipelineStageCount_` would be:

```text
TotalTmem / (CtaM * CtaN) = (128 * 512) / (128 * 16) = 32
```

but CUTLASS **clamps this to 4** accumulator stages. Our fused RHT kernel
follows the same principle, just with an extra factor of 4 folded into
`AccumulatorPipelineStageCount` and the indexing scheme (`index()*4 + i`).

So the cap at 16 does **not** come from TMEM capacity. It comes from CUTLASS’s
SM100 GEMM design, which:

- Uses 4 effective accumulator pipeline stages to balance mainloop progress
  and epilogue drain latency.
- Avoids deep accumulator pipelines that would increase epilogue tail cost
  without materially improving performance for typical tile sizes.
- Leaves some TMEM capacity unused (half of the 128×512 space) in exchange for
  simpler and more robust scheduling.

---

## 4. Summary

- Hardware TMEM capacity for a 1‑SM UMMA on SM100 is `128 x 512` lanes, modeled
  by `MAX_CAPACITY_BITS = 128*512*32`.
- Our fused RHT kernel allocates the **full** 128×512 columns via
  `Allocator1Sm::Sm100TmemCapacityColumns = 512`.
- The UMMA C‑fragment for `M=128, N=16` with
  `AccumulatorPipelineStageCount = 16` has shape
  `((128,16), _, _, 16)` and uses only `128 x 256` worth of TMEM.
- The effective accumulator pipeline depth is `Stages = 4`, each with 4 TMEM
  slots, giving 16 logical accumulator slots.
- The choice to cap the pipeline at 4 stages (16 slots) is inherited directly
  from CUTLASS’s SM100 UMMA builder, which clamps the accumulator pipeline
  for performance reasons even though the hardware could support a deeper
  pipeline that fully occupies TMEM.

