#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cute/algorithm/gemm.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/tensor.hpp>

#include "cute/atom/mma_traits.hpp"
#include "cute/config.hpp"
#include "cute/numeric/numeric_types.hpp"
#include "cute/pointer.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/builders/sm100_common.inl"
#include "cutlass/numeric_conversion.h"
#include "cutlass/pipeline/pipeline.hpp"

using namespace cute;

// M and N are transposed => original input is 1024 x 768
// This will compute
// A: n x m: col-major, col-major since original matrix was row-major
// B: 16 x 16: row-major, RHT is row-major by construction
// C: n x m: row-major => we want row-major (i.e, col-major M x N as output) for
// Blackwell FP4 GEMM which requires K-major matrices, where K is the reduction
// dim SFC: n x (m/16): row-major
// TODO: check where the RHT is constructed

template <typename T>
__host__ __device__ void print_cute(const char* msg, T obj) {
    printf("%s\n", msg);
    cute::print(obj);
    printf("\n");
}
template <typename T, class Fn>
void print_swizzle(int cols, int rows, Fn&& fn) {
    auto bytes = sizeof(T);
    const int effective_cols = bytes * cols;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            const int effective_col = bytes * j;
            const int coord = i * effective_cols + effective_col;
            const auto val = fn.apply(coord);
            printf("%-4u", static_cast<T>(val / bytes));
        }
        printf("\n");
    }
}
template <class ElementA, class ElementB, class ASmemLayout, class BSmemLayout>
struct SharedStorage {
    static constexpr int AccumulatorPipelineStageCount = 16;
    using AtomThrShapeMNK = cute::Shape<_1, _1, _1>;

    using AccumulatorPipeline =
        cutlass::PipelineUmmaAsync<AccumulatorPipelineStageCount / 4,
                                   AtomThrShapeMNK>;
    using AccumulatorPipelineStorage =
        typename AccumulatorPipeline::SharedStorage;

    static constexpr int MainloopPipelineStageCount = size<3>(ASmemLayout{});
    using MainloopPipeline =
        cutlass::PipelineTmaUmmaAsync<MainloopPipelineStageCount,
                                      Shape<_1, _1, _1>, AtomThrShapeMNK>;
    using MainloopPipelineStorage = typename MainloopPipeline::SharedStorage;

    alignas(16) AccumulatorPipelineStorage accumulator;
    alignas(16) MainloopPipelineStorage mainloop;
    alignas(16) cute::uint64_t tma_barrier[1];
    uint32_t tmem_base_ptr;

    struct TensorStorage : cute::aligned_struct<128, _1> {
        // cute::array_aligned<ElementA, cute::cosize_v<ASmemLayout>> smem_A;
        cute::array_aligned<ElementA, cute::cosize_v<ASmemLayout>> smem_A;
        cute::array_aligned<ElementB, cute::cosize_v<BSmemLayout>> smem_B;
    } tensors;
};
#define PRINT_DELIMITER                                                     \
    printf(                                                                 \
        "\n ------------------------------------------------------------- " \
        "\n");

template <class MShape, class NShape, class KShape, class ClusterTileShape,
          class TA, class AStride, class ASmemLayout, class TmaLoadA, class TB,
          class BStride, class BSmemLayout, class TmaLoadB, class TC_,
          class CStride, class CSmemLayout, class TSFC_, class TiledMMA,
          bool kEnableStochasticRounding = false>
__global__ static void rht_gemm_device(
    MShape M, NShape N, KShape K, ClusterTileShape cluster_tile, TA const* A,
    AStride dA, ASmemLayout sAlayout,
    CUTE_GRID_CONSTANT TmaLoadA const tma_load_a, TB const* B, BStride dB,
    BSmemLayout sBlayout, CUTE_GRID_CONSTANT TmaLoadB const tma_load_b, TC_* C,
    CStride dC, CSmemLayout, TSFC_* SFC, TiledMMA mma)
// float const* global_amax,
// const size_t* rng_state)
{
    using X = Underscore;
    using TC = cute::float_e2m1_t;
    using TSFC = cute::float_e4m3_t;

    // ClusterTileShape : cga_tile_shape: 128 x 16 x 16
    // static constexpr bool kApplyStochasticRounding = true;
    using ElementAccumulator = float;
    static constexpr int K_PIPE_MAX = size<3>(ASmemLayout{});
    using AtomThrShapeMNK =
        Shape<decltype(shape<0>(typename TiledMMA::ThrLayoutVMNK{})), _1, _1>;
    static constexpr uint32_t kTmaTransactionBytes = cutlass::bits_to_bytes(
        size(AtomThrShapeMNK{}) * cosize(take<0, 3>(ASmemLayout{})) *
        cute::sizeof_bits_v<TA>);

    static constexpr int kTmaRhtTensorTransactionBytes =
        cutlass::bits_to_bytes(16 * 16 * cute::sizeof_bits_v<TB>);
    static constexpr int AccumulatorPipelineStageCount = 16;

    static constexpr int MainloopPipelineStageCount = size<3>(ASmemLayout{});
    using MainloopPipeline =
        cutlass::PipelineTmaUmmaAsync<MainloopPipelineStageCount,
                                      Shape<_1, _1, _1>, AtomThrShapeMNK>;
    using MainloopPipelineState = typename MainloopPipeline::PipelineState;

    using TmemAllocator = cute::TMEM::Allocator1Sm;
    static constexpr int VectorSize = 16;
    //   const size_t rng_seed = rng_state != nullptr ? rng_state[0] : 0;
    //   const size_t rng_offset = rng_state != nullptr ? rng_state[1] : 0;
    // Preconditions
    //   CUTE_STATIC_ASSERT(is_static<ASmemLayout>::value);
    //   CUTE_STATIC_ASSERT(is_static<BSmemLayout>::value);
    //   CUTE_STATIC_ASSERT(is_static<CSmemLayout>::value);

    // Represent the full tensors
    Tensor mA = tma_load_a.get_tma_tensor(make_shape(M, N));
    Tensor mB = tma_load_b.get_tma_tensor(make_shape(16, 16));
    Tensor mC = make_tensor(cute::subbyte_iterator<TC>(C), make_shape(M, N),
                            dC);  // (M,N)

    auto sfc_shape =
        make_shape(M, make_shape(make_shape(Int<16>{}, _4{}), N / 64));

    auto sfc_stride =
        make_stride(N / 16, make_stride(make_stride(_0{}, _1{}), _4{}));
    auto sfc_layout = make_layout(sfc_shape, sfc_stride);

    if (thread0()) {
        PRINT_DELIMITER
        print_cute("mA", mA);
        print_cute("mB", mB);
        print_cute("mC", mC);
        print_cute("sfc_layout", sfc_layout);
    }

    Tensor mSFC = make_tensor(make_gmem_ptr(SFC), sfc_layout);

    auto cluster_shape = Shape<_1, _1, _1>{};

    // Get the appropriate blocks for this Cluster
    dim3 cluster_coord_in_grid = cluster_id_in_grid();

    // Total number of k-tiles
    // K = k_tile_size is heuristically chosen, defaults to 2048
    // K_TILE_MAX determines the "stride" along columns (N) in groups of 64 cols
    // Each
    const int K_TILE_MAX = min(N, K) / 64;
    uint32_t tiles_in_m =
        (M + size<0>(cluster_tile) - 1) / size<0>(cluster_tile);
    uint32_t tiles_in_n = (N + 64 - 1) / 64;
    uint32_t linear_tile_idx = blockIdx.x;
    uint32_t tile_idx_m = linear_tile_idx % tiles_in_m;
    uint32_t tile_idx_n = (linear_tile_idx / tiles_in_m) * K_TILE_MAX;

    auto mainloop_tiler = Shape<_128, _16, _64>{};
    auto epilogue_tiler = Shape<_128, _64, _64>{};
    Tensor gA_mk =
        local_tile(mA, mainloop_tiler, make_coord(_, _, _), Step<_1, X, _1>{});
    Tensor gB_nk = local_tile(mB, cluster_tile, make_coord(_, _, _),
                              Step<X, _1, _1>{});  // (BLK_N,BLK_K,k)
    Tensor gC_mn = local_tile(mC, epilogue_tiler, make_coord(_, _, _),
                              Step<_1, _1, X>{});  // (BLK_M,BLK_N)

    Tensor gSFC_mn = local_tile(mSFC, epilogue_tiler, make_coord(_, _, _),
                                Step<_1, _1, X>{});  // (BLK_M,BLK_N)

    // Allocate SMEM
    extern __shared__ char shared_memory[];
    using SharedStorage = SharedStorage<TA, TB, ASmemLayout, BSmemLayout>;
    SharedStorage& shared_storage =
        *reinterpret_cast<SharedStorage*>(shared_memory);
    Tensor tCsA =
        make_tensor(make_smem_ptr(shared_storage.tensors.smem_A.data()),
                    sAlayout);  // (MMA,MMA_M,MMA_N,PIPE)
    Tensor tCsB =
        make_tensor(make_smem_ptr(shared_storage.tensors.smem_B.data()),
                    sBlayout);  // (MMA,MMA_N,MMA_K,PIPE)
    if (thread0()) {
        PRINT_DELIMITER
        print_cute("K_TILE_MAX", K_TILE_MAX);
        print_cute("K_PIPE_MAX", K_PIPE_MAX);
        print_cute("tiles_in_m", tiles_in_m);
        print_cute("tiles_in_n", tiles_in_n);
        print_cute("gA_mk", gA_mk);
        print_cute("gB_nk", gB_nk);
        print_cute("gC_mn", gC_mn);
        print_cute("gSFC_mn", gSFC_mn);
        print_cute("tCsA", tCsA);
        print_cute("tCsB", tCsB);
    }

    //
    // MMA: Define C accumulators and A/B partitioning
    //

    int block_rank_in_cluster = cute::block_rank_in_cluster();
    ThrMMA thr_mma = mma.get_slice(block_rank_in_cluster);  // blk idx
    Tensor tCgB = thr_mma.partition_B(gB_nk);  // (MMA,MMA_N,MMA_K,k)

    auto mma_epilogue =
        make_tiled_mma(SM100_MMA_F16BF16_SS<TA, TB, ElementAccumulator, 128, 64,
                                            UMMA::Major::MN, UMMA::Major::MN>{},
                       Layout<Shape<_1, _1>>{});
    ThrMMA thr_mma_epilogue = mma_epilogue.get_slice(block_rank_in_cluster);

    using TiledMmaEpilogue = decltype(mma_epilogue);
    Tensor tCgA = thr_mma.partition_A(gA_mk);
    // Allocate "fragments" -- these are actually umma smem descriptors
    Tensor tCrA = thr_mma.make_fragment_A(tCsA);  // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);  // (MMA,MMA_M,MMA_K,PIPE)

    if (thread0()) {
        PRINT_DELIMITER
        print_cute("thr_mma", thr_mma);
        print_cute("tCgB", tCgB);
        print_cute("mma_epilogue", mma_epilogue);
        print_cute("thr_mma_epilogue", thr_mma_epilogue);
        print_cute("tCgA", tCgA);
        print_cute("tCrA", tCrA);
        print_cute("tCrB", tCrB);
    }
    auto acc_shape_mma =
        partition_shape_C(TiledMMA{}, take<0, 2>(ClusterTileShape{}));
    auto acc_shape_mma2 = partition_shape_C(TiledMMA{}, Shape<_128, _32>{});
    auto acc_shape_epilogue =
        partition_shape_C(TiledMmaEpilogue{}, take<0, 2>(epilogue_tiler));

    auto bulk_tmem_mma = TiledMMA::make_fragment_C(
        append(acc_shape_mma, Int<AccumulatorPipelineStageCount>{}));

    auto bulk_tmem_epilogue = TiledMmaEpilogue::make_fragment_C(
        append(acc_shape_epilogue, Int<AccumulatorPipelineStageCount / 4>{}));
    if (thread0()) {
        PRINT_DELIMITER
        print_cute("acc_shape_mma", acc_shape_mma);
        print_cute("acc_shape_mma2", acc_shape_mma2);
        print_cute("acc_shape_epilogue", acc_shape_epilogue);
        print_cute("bulk_tmem_mma", bulk_tmem_mma);
        print_cute("bulk_tmem_epilogue", bulk_tmem_epilogue);
    }
    TmemAllocator tmem_allocator{};
    cutlass::arch::NamedBarrier tmem_allocation_result_barrier(
        32 + 128, cutlass::arch::ReservedNamedBarriers::TmemAllocBarrier);

    Layout cta_layout_mnk = make_layout(cluster_shape);
    Layout cta_layout_vmnk =
        tiled_divide(cta_layout_mnk, make_tile(typename TiledMMA::AtomThrID{}));
    auto cta_coord_vmnk = cta_layout_vmnk.get_flat_coord(block_rank_in_cluster);

    auto [tAgA, tAsA] =
        tma_partition(tma_load_a, get<2>(cta_coord_vmnk),
                      make_layout(size<2>(cta_layout_vmnk)),
                      group_modes<0, 3>(tCsA), group_modes<0, 3>(tCgA));

    auto [tBgB, tBsB] =
        tma_partition(tma_load_b, get<1>(cta_coord_vmnk),
                      make_layout(size<1>(cta_layout_vmnk)),
                      group_modes<0, 3>(tCsB), group_modes<0, 3>(tCgB));

    if (thread0()) {
        PRINT_DELIMITER
        print_cute("tAgA", tAgA);
        print_cute("tAsA", tAsA);
        print_cute("tBgB", tBgB);
        print_cute("tBsB", tBgB);
        print_cute("AtomThrShapeMNK", AtomThrShapeMNK{});
    }

    uint16_t tma_mcast_mask_a =
        create_tma_multicast_mask<2>(cta_layout_vmnk, cta_coord_vmnk);
    uint16_t tma_mcast_mask_b =
        create_tma_multicast_mask<1>(cta_layout_vmnk, cta_coord_vmnk);

    int warp_idx = cutlass::canonical_warp_idx_sync();

    bool is_mma_warp = (warp_idx == 0);
    bool is_dma_warp = (warp_idx == 1);
    bool is_epilogue_warp = (warp_idx >= 4 && warp_idx <= 7);

    typename MainloopPipeline::Params mainloop_pipeline_params;
    if (is_dma_warp) {
        mainloop_pipeline_params.role =
            MainloopPipeline::ThreadCategory::Producer;
    }
    if (is_mma_warp) {
        mainloop_pipeline_params.role =
            MainloopPipeline::ThreadCategory::Consumer;
    }
    mainloop_pipeline_params.is_leader = cute::elect_one_sync() && is_dma_warp;
    mainloop_pipeline_params.transaction_bytes = kTmaTransactionBytes;
    mainloop_pipeline_params.initializing_warp = 0;
    MainloopPipeline mainloop_pipeline(
        shared_storage.mainloop, mainloop_pipeline_params, cluster_shape,
        cute::true_type{},   // Perform barrier init
        cute::true_type{});  // Delay mask calculation

    MainloopPipelineState mainloop_pipe_consumer_state;
    MainloopPipelineState mainloop_pipe_producer_state =
        cutlass::make_producer_start_state<MainloopPipeline>();

    using AccumulatorPipeline =
        cutlass::PipelineUmmaAsync<AccumulatorPipelineStageCount / 4,
                                   AtomThrShapeMNK>;
    using AccumulatorPipelineState =
        typename AccumulatorPipeline::PipelineState;

    AccumulatorPipelineState accumulator_pipe_consumer_state;
    AccumulatorPipelineState accumulator_pipe_producer_state =
        cutlass::make_producer_start_state<AccumulatorPipeline>();

    typename AccumulatorPipeline::Params accumulator_pipeline_params;
    if (is_mma_warp) {
        accumulator_pipeline_params.role =
            AccumulatorPipeline::ThreadCategory::Producer;
    }
    if (is_epilogue_warp) {
        accumulator_pipeline_params.role =
            AccumulatorPipeline::ThreadCategory::Consumer;
    }
    // Only one producer thread arrives on this barrier.
    accumulator_pipeline_params.producer_arv_count = 1;
    accumulator_pipeline_params.consumer_arv_count =
        size(AtomThrShapeMNK{}) * 128;
    accumulator_pipeline_params.initializing_warp = 1;
    AccumulatorPipeline accumulator_pipeline(
        shared_storage.accumulator, accumulator_pipeline_params, cluster_shape,
        cute::true_type{},   // Perform barrier init
        cute::true_type{});  // Delay mask calculation
    if (warp_idx == 2 && elect_one_sync()) {
        cute::initialize_barrier(shared_storage.tma_barrier[0],
                                 /* num_threads */ 1);
    }
    __syncthreads();
    using TMEM_LOAD_NEW = cute::SM100::TMEM::LOAD::SM100_TMEM_LOAD_32dp32b64x;

    if (is_dma_warp) {
        if (elect_one_sync()) {
            printf("Initialize TMA Load B...\n");
            cute::set_barrier_transaction_bytes(shared_storage.tma_barrier[0],
                                                kTmaRhtTensorTransactionBytes);
            copy(tma_load_b.with(shared_storage.tma_barrier[0],
                                 tma_mcast_mask_b),
                 tBgB(_, 0, 0), tBsB(_, 0));
        }
        cute::wait_barrier(shared_storage.tma_barrier[0], 0 /*tma_phase_bit*/);

        if (elect_one_sync()) {
            auto tAgA_mk = tAgA(_, 0, _);
            print_cute("DMA WARP: Loading tAgA_mk(_,k_tile_idx_n)",
                       tAgA_mk(_, 0));
            print_cute("DMA WARP: Loading tAsA(_,write_stage)", tAsA(_, 0));
        }

        do {
            bool is_first_wave = linear_tile_idx == blockIdx.x;
            uint32_t skip_wait = is_first_wave;
            auto tAgA_mk = tAgA(_, tile_idx_m, _);
            int k_tile = 0;
            auto barrier_token = mainloop_pipeline.producer_try_acquire(
                mainloop_pipe_producer_state, skip_wait);

            CUTE_NO_UNROLL
            while (k_tile < K_TILE_MAX && k_tile + tile_idx_n < tiles_in_n) {
                int k_tile_idx_n = tile_idx_n + k_tile;
                if (elect_one_sync()) {
                    printf(
                        "tile_idx_m, tile_idx_n, tiles_in_n, k_tile, "
                        "k_tile_idx_n,"
                        "K_TILE_MAX: %d, %d, %d, %d, %d, %d\n",
                        tile_idx_m, tile_idx_n, tiles_in_n, k_tile,
                        k_tile_idx_n, K_TILE_MAX);
                }

                ++k_tile;
                skip_wait =
                    (is_first_wave && k_tile < MainloopPipelineStageCount);

                // If barrier token is !BarrierStatus::WaitDone, waits on
                // empty_barrier for current stage Else arrive_expect_tx on
                // full_barrier
                // mainloop_pipeline.producer_acquire(mainloop_pipe_producer_state,
                //                                    barrier_token);
                // using BarrierType =
                //     typename MainloopPipeline::ProducerBarrierType;

                if (elect_one_sync()) {
                    printf(
                        "Mainloop producer getting barrier for stage, phase, "
                        "count: %d %d %d\n",
                        mainloop_pipe_producer_state.index(),
                        mainloop_pipe_producer_state.phase(),
                        mainloop_pipe_producer_state.count());
                }

                // BarrierType* tma_barrier =
                //     mainloop_pipeline.producer_get_barrier(
                //         mainloop_pipe_producer_state);

                int write_stage = mainloop_pipe_producer_state.index();

                // Advance write stage
                ++mainloop_pipe_producer_state;

                if (elect_one_sync()) {
                    printf(
                        "Mainloop producer acquiring arrival token for stage, "
                        "phase, count: %d %d %d\n",
                        mainloop_pipe_producer_state.index(),
                        mainloop_pipe_producer_state.phase(),
                        mainloop_pipe_producer_state.count());
                }

                // Acquire arrival token for the next stage, non-blocking
                // barrier_token = mainloop_pipeline.producer_try_acquire(
                //     mainloop_pipe_producer_state, skip_wait);

                // if (cute::elect_one_sync()) {
                //     copy(tma_load_a.with(*tma_barrier, tma_mcast_mask_a),
                //          tAgA_mk(_, k_tile_idx_n), tAsA(_, write_stage));
                // }
            }
            linear_tile_idx += gridDim.x;
            tile_idx_m = linear_tile_idx % tiles_in_m;
            tile_idx_n = (linear_tile_idx / tiles_in_m) * K_TILE_MAX;

        } while (tile_idx_m < tiles_in_m && tile_idx_n < tiles_in_n);
        // mainloop_pipeline.producer_tail(mainloop_pipe_producer_state);
    }
}

int main() {
    using TA = cute::bfloat16_t;
    using TB = TA;
    using TC = cutlass::float_e2m1_t;
    using TSFC = cutlass::float_ue4m3_t;

    int k_tile_size = 2048;

    constexpr int m = 128;   // 768;   // N
    constexpr int n = 1024;  // 1024;  // M
    // Define shapes (dynamic)
    auto M = static_cast<int>(m);
    auto N = static_cast<int>(n);

    // Define strides (mixed)
    auto dA = make_stride(Int<1>{}, m);   // (dM,dK)
    auto dB = make_stride(Int<1>{}, 16);  // (dN,dK)
    auto dC = make_stride(n, Int<1>{});   // (dM,dN)

    auto cga_shape = Shape<_1, _1, _1>{};
    auto cga_tile_shape = Shape<_128, _16, _16>{};
    auto cluster_tile_mainloop = Shape<_128, _16, _64>{};

    // Construct the MMA
    // 128 x 16 x 16
    using MMA_Op = SM100_MMA_F16BF16_SS<TA, TB, float, 128, 16, UMMA::Major::MN,
                                        UMMA::Major::MN>;
    using Traits = MMA_Traits<MMA_Op>;
    auto mma =
        make_tiled_mma(SM100_MMA_F16BF16_SS<TA, TB, float, 128, 16,
                                            UMMA::Major::MN, UMMA::Major::MN>{},
                       Layout<Shape<_1, _1>>{});
    using TMma = decltype(mma);
    print_cute("Traits K", Traits::K);

    print_cute("mmaK", TMma::AtomShape_MNK{});

    // MMA in CGA Layout XXX: Need to generalize synchro? {$nv-release-never}
    print_cute("mma", mma);

    // Assert that the TiledMMA uses all CTAs in the CGA.
    CUTE_STATIC_ASSERT_V(size(cga_shape) == size(mma));
    CUTE_STATIC_ASSERT_V(evenly_divides(cga_tile_shape, tile_shape(mma)));

    // Determine the A and B shapes
    auto mma_shape_B = partition_shape_B(
        mma, make_shape(size<1>(cga_tile_shape), size<2>(cga_tile_shape)));

    print_cute("mma_shape_B", mma_shape_B);

    using TiledMma = decltype(mma);
    using AtomThrID = typename TiledMma::AtomThrID;

    using SmemShape_M = decltype(shape_div(
        shape<0>(cga_tile_shape),
        shape_div(shape<0>(cga_tile_shape),
                  size<0>(cga_tile_shape) / size(AtomThrID{}))));
    auto smemShape_M = SmemShape_M{};
    auto atomThrID = AtomThrID{};

    using SmemShape_N = decltype(shape_div(
        shape<1>(cga_tile_shape),
        shape_div(shape<1>(cga_tile_shape),
                  size<1>(cga_tile_shape) / size(AtomThrID{}))));
    using SmemShape_K = decltype(cute::get<2>(cga_tile_shape));

    auto smemShapeM = SmemShape_M{};
    auto smemShapeN = SmemShape_N{};
    auto smemShapeK = SmemShape_K{};
    print_cute("smemShapeM", smemShapeM);
    print_cute("smemShapeN", smemShapeN);
    print_cute("smemShapeK", smemShapeK);

    using SmemLayoutAtomB =
        decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
                 cute::UMMA::Major::MN, TB, SmemShape_N, SmemShape_K>());
    using SmemLayoutAtomB_K =
        decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
                 cute::UMMA::Major::K, TB, SmemShape_N, SmemShape_K>());

    auto smemLayoutAtomB_Kmajor = SmemLayoutAtomB_K{};
    // mma 128 x 16 x 16
    // partition 128 x 64 by 128 x 16
    // ((128, 16), 1, 4)
    // The pipeline stages is then appended to this shape
    // Finally the smemLayoutAtomA (64, 8) is tiled to this shape
    // (((64, 2), (8, 2)), 1, 4, PIPE)
    // (((_64,_2),(_8,_2)),_1,_4,(_1,_13))
    auto mma_shape_A =
        partition_shape_A(mma, make_shape(size<0>(cluster_tile_mainloop),
                                          size<2>(cluster_tile_mainloop)));

    print_cute("mma_shape_A", mma_shape_A);

    auto smemLayoutAtomB = SmemLayoutAtomB{};
    print_cute("SmemLayoutAtomB", smemLayoutAtomB);

    using SmemShape_M_A = decltype(shape_div(
        shape<0>(cluster_tile_mainloop),
        shape_div(shape<0>(cluster_tile_mainloop),
                  size<0>(cluster_tile_mainloop) / size(AtomThrID{}))));
    using SmemShape_K_A = decltype(cute::get<2>(cluster_tile_mainloop));

    // 64 x 8, 1 x 64
    using SmemLayoutAtomA =
        decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
                 cute::UMMA::Major::MN, TA, SmemShape_M_A, SmemShape_K_A>());
    using SmemLayoutAtomA_Kmajor =
        decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
                 cute::UMMA::Major::K, TA, SmemShape_M_A, SmemShape_K_A>());
    auto smemLayoutAtomA = SmemLayoutAtomA{};

    auto layout_A = smemLayoutAtomB_Kmajor.layout_a();
    auto layout_B = smemLayoutAtomB_Kmajor.layout_b();
    auto atom_cols = size<1>(layout_B);
    auto atom_rows = size<0>(layout_B);
    constexpr int numel = atom_cols * atom_rows;
    thrust::host_vector<uint16_t> mat(numel);
    for (int i = 0; i < numel; i++) {
        mat[i] = i % atom_cols;
    }
    print_swizzle<uint16_t>(atom_cols, atom_rows, layout_A);
    // auto sA = tile_to_shape(
    //     smemLayoutAtomA_Kmajor, Shape<_128, _16>{});  //
    //     (MMA,MMA_M,MMA_K,PIPE)
    print_cute("smemLayoutAtomA", smemLayoutAtomA);
    print_cute("smemLayoutAtomB_Kmajor", smemLayoutAtomB_Kmajor);

    // print_cute("smemLayoutAtomA_Kmajor", smemLayoutAtomA_Kmajor);

    // Define the smem layouts (static)
    // Calculate max pipeline stages based on Blackwell SM100's 232KB shared
    // memory
    constexpr int kBlackwellSmemSize = 232448;  // 232KB in bytes
    constexpr int kBytesPerStage = cute::size(mma_shape_A) * sizeof(TA) +
                                   cute::size(mma_shape_B) * sizeof(TB);
    constexpr int kReservedBytes = 256;  // Reserve for barriers and other uses
    constexpr int kMaxStages =
        (kBlackwellSmemSize - kReservedBytes) / kBytesPerStage;
    auto sP = Int<kMaxStages>{};  // SMEM pipelines
    auto mma_tile_shape =
        append(mma_shape_A,
               sP);  // MMA = ((64, 2), (8, 2)), MMA_M=1, MMA_K=4, SP=(1, 13)
    auto sA = UMMA::tile_to_mma_shape(
        SmemLayoutAtomA{}, append(mma_shape_A, sP));  // (MMA,MMA_M,MMA_K,PIPE)

    // SmemLayoutAtomB (16, 8):(1, 16) - i.e., col major, 4 cols => 128 Bytes,
    // swizzle every 4 cols mma_shape_B: ((16, 16), 1, 1)
    auto sB = UMMA::tile_to_mma_shape(
        SmemLayoutAtomB{}, append(mma_shape_B, sP));  // (MMA,MMA_N,MMA_K,PIPE)

    auto sC = Layout<_1>{};  // XXX Dummy

    print_cute("sA", sA);
    print_cute("sB", sB);
    // Create GMEM tensors
    thrust::host_vector<TA> hA(M * N);
    thrust::host_vector<TB> hB(16 * 16);
    thrust::device_vector<TA> deviceA = hA;
    thrust::device_vector<TB> deviceB = hB;
    thrust::host_vector<uint8_t> hC(M * N);
    thrust::host_vector<uint8_t> sFC(M * N);
    thrust::device_vector<uint8_t> deviceC = hC;
    thrust::device_vector<uint8_t> deviceSFC = sFC;

    Tensor tensorA =
        make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(deviceA.data())),
                    make_layout(make_shape(M, N), dA));  // (M,N)
    Tensor tensorB =
        make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(deviceB.data())),
                    make_layout(make_shape(16, 16), dB));  // (16,16)

    // Create the TiledCopy

    auto tma_load_a = make_tma_copy_A_sm100(
        SM90_TMA_LOAD{}, tensorA, sA(_, _, _, 0), cluster_tile_mainloop, mma);
    auto tma_load_b = make_tma_copy_B_sm100(
        SM90_TMA_LOAD{}, tensorB, sB(_, _, _, 0), cga_tile_shape, mma);

    auto mainloop_tiler = Shape<_128, _16, _64>{};
    auto epilogue_tiler = Shape<_128, _64, _64>{};
    auto mA = tensorA;
    auto mB = tensorB;
    auto cluster_tile = cga_tile_shape;

    Tensor gA_mk =
        local_tile(mA, mainloop_tiler, make_coord(_, _, _), Step<_1, X, _1>{});
    Tensor gB_nk = local_tile(mB, cluster_tile, make_coord(_, _, _),
                              Step<X, _1, _1>{});  // (BLK_N,BLK_K,k)
    print_cute("gA_mk", gA_mk);
    print_cute("gB_nk", gB_nk);
    // Tensor gC_mn = local_tile(mC, epilogue_tiler, make_coord(_,_, _),
    // Step<_1,_1, X>{});  // (BLK_M,BLK_N)

    // Assert checks on tile sizes -- no predication
    // NVTE_CHECK(M % size<0>(cga_tile_shape) == 0,
    //             "Inner dimension must be divisible by ",
    //             static_cast<size_t>(size<0>(cga_tile_shape)), " but got ", M,
    //             ".");
    // NVTE_CHECK(N % (4 * size<1>(cga_tile_shape)) == 0,
    //             "Outer dimension must be divisible by ", 4 *
    //             static_cast<size_t>(size<1>(cga_tile_shape)), " but got ", N,
    //             ".");

    uint32_t tiles = size(ceil_div(M, get<0>(cga_tile_shape))) *
                     size(ceil_div(N, k_tile_size));

    // tiles = (tiles < sm_count) ? tiles : sm_count;

    dim3 dimBlock(256);
    dim3 dimCluster(size<0>(cga_shape), size<1>(cga_shape), size<2>(cga_shape));
    dim3 dimGrid(tiles, 1, 1);
    constexpr bool kEnableStochasticRounding = false;
    int smem_size = sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>);
    auto* kernel_ptr =
        &rht_gemm_device<decltype(M), decltype(N), decltype(k_tile_size),
                         decltype(cga_tile_shape), TA, decltype(dA),
                         decltype(sA), decltype(tma_load_a), TB, decltype(dB),
                         decltype(sB), decltype(tma_load_b), uint8_t,
                         decltype(dC), decltype(sC), uint8_t, decltype(mma),
                         kEnableStochasticRounding>;

    bool status = cudaFuncSetAttribute(
        *kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    if (status != cudaSuccess) {
        std::cerr << "Error: Failed to set Shared Memory size." << std::endl;
        return 1;
    }
    uint8_t* C =
        reinterpret_cast<uint8_t*>(thrust::raw_pointer_cast(deviceC.data()));
    uint8_t* SFC =
        reinterpret_cast<uint8_t*>(thrust::raw_pointer_cast(deviceSFC.data()));

    (*kernel_ptr)<<<dimGrid, dimBlock, smem_size>>>(
        M, N, k_tile_size, cga_tile_shape,
        thrust::raw_pointer_cast(deviceA.data()), dA, sA, tma_load_a,
        thrust::raw_pointer_cast(deviceB.data()), dB, sB, tma_load_b, C, dC, sC,
        SFC, mma);
}