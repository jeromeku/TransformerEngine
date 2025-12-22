#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cute/tensor.hpp>

#include "cute/arch/mma_sm90_desc.hpp"
#include "cute/numeric/numeric_types.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/mma_sm90.h"
#include "cutlass/cluster_launch.hpp"
#include "cutlass/device_kernel.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/util/print_error.hpp"
#include "debug.hpp"

using namespace cute;

template <class ElementA, class ElementB,
          class SmemLayoutA,  // (M,K,P)
          class SmemLayoutB>  // (N,K,P)
struct SharedStorage {
    alignas(128) cute::ArrayEngine<ElementA, cosize_v<SmemLayoutA>> A;
    alignas(128) cute::ArrayEngine<ElementB, cosize_v<SmemLayoutB>> B;

    uint64_t tma_barrier[size<2>(SmemLayoutA{})];
    uint64_t mma_barrier[size<2>(SmemLayoutA{})];
};

template <typename T>
__host__ __device__ void print_matrix(const T* mat, int num_rows, int num_cols) {
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            int idx = i * num_cols + j;  // assume row-major
            printf("%4d ", int(float((mat[idx]))));
        }
        printf("\n");
    }
}
template <class ProblemShape, class CtaTiler, class TA, class SmemLayoutA, class TmaA, class TB,
          class SmemLayoutB, class TmaB, class TC, class CStride, class TiledMma, class Alpha,
          class Beta>
__global__ static __launch_bounds__(decltype(size(TiledMma{}))::value) void gemm_device(
    ProblemShape shape_MNK, CtaTiler cta_tiler, TA const* A, CUTLASS_GRID_CONSTANT TmaA const tma_a,
    TB const* B, CUTLASS_GRID_CONSTANT TmaB const tma_b, TC* C, CStride dC, TiledMma mma,
    Alpha alpha, Beta beta) {
    // Preconditions
    CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});  // (M, N, K)
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});  // (BLK_M, BLK_N, BLK_K)

    static_assert(is_static<SmemLayoutA>::value);
    static_assert(is_static<SmemLayoutB>::value);

    CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutA{}) == size<0>(cta_tiler));  // BLK_M
    CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutB{}) == size<1>(cta_tiler));  // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutA{}) == size<2>(cta_tiler));  // BLK_K
    CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutB{}) == size<2>(cta_tiler));  // BLK_K

    CUTE_STATIC_ASSERT_V(congruent(select<0, 1>(shape_MNK), dC));  // dC strides for shape MN

    //
    // Full and Tiled Tensors
    //

    // Represent the full tensors
    auto [M, N, K] = shape_MNK;
    Tensor mA = tma_a.get_tma_tensor(make_shape(M, K));               // (M,K) TMA Tensor
    Tensor mB = tma_b.get_tma_tensor(make_shape(N, K));               // (N,K) TMA Tensor
    Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), dC);  // (M,N)

    // Get the appropriate blocks for this thread block
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);               // (m,n,k)
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});  // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});  // (BLK_N,BLK_K,k)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});  // (BLK_M,BLK_N)
    // Shared memory tensors
    extern __shared__ char shared_memory[];
    using SharedStorage = SharedStorage<TA, TB, SmemLayoutA, SmemLayoutB>;
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
    Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), SmemLayoutA{});  // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), SmemLayoutB{});  // (BLK_N,BLK_K,PIPE)

    //
    // Partition the copying of A and B tiles
    //
    // TUTORIAL:
    //   These are TMA partitionings, which have a dedicated custom partitioner.
    //   The Int<0>, Layout<_1> indicates that the TMAs are not multicasted.
    //     Any multicasting must be in conformance with tma_x constructed with make_tma_atom on host.
    //   The group_modes<0,2> transforms the (X,Y,Z)-shaped tensors into ((X,Y),Z)-shaped tensors
    //     with the understanding that the TMA is responsible for everything in mode-0.
    //   The tma_partition reorders and offsets mode-0 according to the tma_x atom and the multicast info.
    //

    auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sA),
                                      group_modes<0, 2>(gA));  // (TMA,k) and (TMA,PIPE)

    auto [tBgB, tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sB),
                                      group_modes<0, 2>(gB));  // (TMA,k) and (TMA,PIPE)

    // The TMA is responsible for copying everything in mode-0 of tAsA and tBsB
    constexpr int tma_transaction_bytes =
        sizeof(make_tensor_like(tensor<0>(tAsA))) + sizeof(make_tensor_like(tensor<0>(tBsB)));

    //
    // PREFETCH
    //

    auto K_PIPE_MAX = size<1>(tAsA);

    // Total count of tiles
    int k_tile_count = size<1>(tAgA);
    // Current tile index in gmem to read from
    int k_tile = 0;

    PRINT_CUTE(mA);
    PRINT_CUTE(gA);
    PRINT_CUTE(sA);
    PRINT_CUTE(tAgA);
    PRINT_CUTE(tAsA);
    PRINT_CUTE(K_PIPE_MAX);
    PRINT_CUTE(k_tile_count);

    // Initialize Barriers
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_predicate = cute::elect_one_sync();
    uint64_t* producer_mbar = smem.tma_barrier;
    uint64_t* consumer_mbar = smem.mma_barrier;

    using ProducerBarType = cutlass::arch::ClusterTransactionBarrier;  // TMA
    using ConsumerBarType = cutlass::arch::ClusterBarrier;             // MMA
    CUTE_UNROLL
    for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe) {
        if ((warp_idx == 0) && lane_predicate) {
            ProducerBarType::init(&producer_mbar[pipe], 1);
            ConsumerBarType::init(&consumer_mbar[pipe], 128);
        }
    }
    // Ensure barrier init is complete on all CTAs
    cluster_sync();

    // Start async loads for all pipes
    CUTE_UNROLL
    for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe) {
        if ((warp_idx == 0) && lane_predicate) {
            // Set expected Tx Bytes after each reset / init
            ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
            copy(tma_a.with(producer_mbar[pipe]), tAgA(_, k_tile), tAsA(_, pipe));
            copy(tma_b.with(producer_mbar[pipe]), tBgB(_, k_tile), tBsB(_, pipe));
        }
        --k_tile_count;
        ++k_tile;
    }
    auto read_state = cutlass::PipelineState<K_PIPE_MAX>();  // MMA  reads
    int read_pipe = read_state.index();

    ProducerBarType::wait(&producer_mbar[read_pipe], read_state.phase());

    printf("AFTER TMA: tAsA:\n");
    const TA* smemA = reinterpret_cast<const TA*>(smem.A.begin());
    auto num_rows = 128;
    auto num_cols = size<1>(sA);
    PRINT_CUTE(num_rows);
    PRINT_CUTE(num_cols);
    printf("gmemA:\n");
    print_matrix(A, num_rows, num_cols);
    printf("smemA:\n");
    print_matrix(smemA, num_rows, num_cols);
    //print_tensor(tAsA);

    //
    // Define A/B partitioning and C accumulators
    //
    // TUTORIAL:
    //   The tCrA and tCrB are actually Tensors of MMA Descriptors constructed as views of SMEM.
    //   The MMA Descriptor generation is automatic via inspection and validation of the SMEM Layouts.
    //   Because the MMA reads directly from SMEM and the fragments are descriptors rather than registers,
    //     there is no need for copy(tCsA, tCrA) in the mainloop.
    //

    ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);
    Tensor tCsA = thr_mma.partition_A(sA);  // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = thr_mma.partition_B(sB);  // (MMA,MMA_N,MMA_K,PIPE)
    Tensor tCgC = thr_mma.partition_C(gC);  // (MMA,MMA_M,MMA_N)

    PRINT_CUTE(thr_mma);
    PRINT_CUTE(tCsA);

    // Allocate accumulators and clear them
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);  // (MMA,MMA_M,MMA_N)
    clear(tCrC);

    // Allocate "fragments"
    using TMma =
        decltype(make_tiled_mma(SM90_64x64x16_F16F16F16_SS<GMMA::Major::K, GMMA::Major::K>{}));

    using FrgTypeA = typename TMma::FrgTypeA;
    
//     auto smemDescriptorA = MakeTensor<FrgTypeA>{}(tCsA);  //make_tensor<FrgTypeA>(tCsA);
    Tensor u128_tensor = recast<uint128_t const>(tCsA);
    // Result
    GmmaDescriptor desc;

    // Layout type
    using namespace cute::SM90::GMMA;
    constexpr LayoutType LAYOUT_TYPE = cute::SM90::GMMA::layout_type(u128_tensor);
    desc.bitfield.layout_type_ = uint8_t(LAYOUT_TYPE);

    // Start address (4LSB not included)
    uint32_t start_address = cast_smem_ptr_to_uint(raw_pointer_cast(u128_tensor.data()));
    desc.bitfield.start_address_ = static_cast<uint16_t>(start_address >> 4);

    constexpr uint8_t base_offset = 0;
    desc.bitfield.base_offset_ = base_offset;

    // LayoutType meta
    constexpr int W = LAYOUT_TYPE == LayoutType::INTERLEAVE ? 1
                      : LAYOUT_TYPE == LayoutType::B32      ? 2
                      : LAYOUT_TYPE == LayoutType::B64      ? 4
                      : LAYOUT_TYPE == LayoutType::B128     ? 8
                                                            : -1;
    Layout canonical_layout =
        logical_divide(layout(u128_tensor), Tile<Layout<_8, _1>, Layout<_2, _1>>{});

    PRINT_CUTE(W);
    PRINT_CUTE(u128_tensor);
    PRINT_CUTE(canonical_layout);

    // TMA copies in 8 x 64 fp16 chunks (or 8 x 8 in normalized 128b units). we can see this in the wgmma128.log, where 8 x 64 **logical** tiles are copied end to end (col major order)
    // WGMMA expects smem layout to be described in canonical layouts
    // For K-major 128B swizzle, these atoms are 8 x 2 in units of 128b (8 fp16).
    // The descriptor requires an SBO and LBO to be able to determine the strides between these atoms
    // Since TMA copies 8 x 64 where 64 elements are contiguous
    // a WGMMA instructions such as 64 x 16 for Operand A would require 8 of these tiles and 2 of these normalized cols from SMEM
    // So it needs to know the row stride between each smem tile (SBO) which is 8 x 8 in normalized units and a col stride of 1  
    // in make_fragment_A, the canonical layout does NOT depend on the shape of sA, but only on the shape of the WGMMA instruction
    // This is because of the tiling pattern of the TMA copy.  I.e., a 128 x 64 smemA vs a 128 x 128 would have the same SBO and LBO
    // the only difference is more 8 x 8 normalized unit tiles in the K direction.  The strides between row tiles and col units remains the same.
// Canonical layout for 128b K-major swizzle: ((8,m),(T,2k)):((8T,SBO),(1,T))
// SBO = stride_off << 4 = 64 << 4 = 64 * 16
//  8 x 64 * 2 = 8 x 64 is a smem tile, distance between tiles in **byte** is 8 x 64 * 2 = SBO.
// This is then right shifted by 4 to obtain the stride_off  

    Tensor tCrA = thr_mma.make_fragment_A(tCsA);                         // (MMA,MMA_M,MMA_K,PIPE)
    PRINT_CUTE(sA);
    PRINT_CUTE(tCsA);
    PRINT_CUTE(tCrA);
    auto descriptor = tCrA(_,_,_,0);
    auto gmma_descriptor = descriptor[0];
    PRINT_CUTE(descriptor);
    PRINT_CUTE(gmma_descriptor);
    // Tensor tCrB = thr_mma.make_fragment_B(tCsB);                         // (MMA,MMA_N,MMA_K,PIPE)
    // PRINT_CUTE(tCrA(_,_,_,0));
    //
    // PIPELINED MAIN LOOP
    //
    // TUTORIAL:
    //   Rather than interleaving the stages and instructions like in SM70 and SM80,
    //     the SM90 mainloops rely on explicit producer-consumer synchronization
    //     on the purely async instructions TMA and MMA.
    //   More advanced pipeline and warp-specialization strategies are available in CUTLASS mainloops.
    //

    // A PipelineState is a circular pipe index [.index()] and a pipe phase [.phase()]
    //   that flips each cycle through K_PIPE_MAX.
    // auto write_state = cutlass::PipelineState<K_PIPE_MAX>();  // TMA writes
    // auto read_state  = cutlass::PipelineState<K_PIPE_MAX>();             // MMA  reads

    // CUTE_NO_UNROLL

    //   while (k_tile_count > -K_PIPE_MAX)
    //   {
    //     // Wait for Producer to complete
    //     int read_pipe = read_state.index();
    //     ProducerBarType::wait(&producer_mbar[read_pipe], read_state.phase());

    //     // MMAs to cover 1 K_TILE
    //     warpgroup_arrive();
    //     gemm(mma, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrC);     // (V,M) x (V,N) => (V,M,N)
    //     warpgroup_commit_batch();

    //     // Wait for all MMAs in a K_TILE to complete
    //     warpgroup_wait<0>();

    //     // Notify that consumption is done
    //     ConsumerBarType::arrive(&consumer_mbar[read_pipe]);
    //     ++read_state;

    //     if ((warp_idx == 0) && lane_predicate)
    //     {
    //       int pipe = write_state.index();
    //       // Wait for Consumer to complete consumption
    //       ConsumerBarType::wait(&consumer_mbar[pipe], write_state.phase());
    //       // Set expected Tx Bytes after each reset / init
    //       ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
    //       copy(tma_a.with(producer_mbar[pipe]), tAgA(_,k_tile), tAsA(_,pipe));
    //       copy(tma_b.with(producer_mbar[pipe]), tBgB(_,k_tile), tBsB(_,pipe));
    //       ++write_state;
    //     }
    //     --k_tile_count;
    //     ++k_tile;
    //   }

    // #endif
}

// Setup params for a TN GEMM
template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm_tn(int m, int n, int k, Alpha alpha, TA const* A, int ldA, TB const* B, int ldB,
             Beta beta, TC* C, int ldC, cudaStream_t stream = 0) {
    // Define shapes (dynamic)
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);
    auto prob_shape = make_shape(M, N, K);  // (M, N, K)

    // Define TN strides (mixed)
    auto dA = make_stride(ldA, Int<1>{});  // (dM, dK)
    auto dB = make_stride(ldB, Int<1>{});  // (dN, dK)
    auto dC = make_stride(Int<1>{}, ldC);  // (dM, dN)

    // Define CTA tile sizes (static)
    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<128>{};
    auto cta_tiler = make_shape(bM, bN, bK);  // (BLK_M, BLK_N, BLK_K)
    auto bP = Int<1>{};                       // Pipeline

    // Define the smem layouts (static)
    auto atomA = GMMA::Layout_K_SW128_Atom<TA>{};
    PRINT_CUTE(atomA);
    auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<TA>{}, make_shape(bM, bK, bP));
    auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<TB>{}, make_shape(bN, bK, bP));

    // Define the MMA
    TiledMMA tiled_mma =
        make_tiled_mma(SM90_64x64x16_F16F16F16_SS<GMMA::Major::K, GMMA::Major::K>{});

    // Define the TMAs
    // Create Global memory tensors for TMA inspection
    Tensor mA = make_tensor(A, make_shape(M, K), dA);
    Tensor mB = make_tensor(B, make_shape(N, K), dB);

    // Create TMA Atoms with the desired copy operation on the source and destination
    Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_, _, 0), make_shape(bM, bK));
    Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_, _, 0), make_shape(bN, bK));

    //
    // Setup and Launch
    //

    int NUM_THREADS = 1;
    int NUM_BLOCKS = 1;
    // Launch parameter setup
    dim3 dimBlock(NUM_THREADS);
    dim3 dimCluster(1, 1, 1);
    dim3 dimGrid(NUM_BLOCKS);
    int smemBytes = sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>);

    auto* kernel_ptr =
        &gemm_device<decltype(prob_shape), decltype(cta_tiler), TA, decltype(sA), decltype(tmaA),
                     TB, decltype(sB), decltype(tmaB), TC, decltype(dC), decltype(tiled_mma),
                     decltype(alpha), decltype(beta)>;

    CUTE_CHECK_ERROR(
        cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes));

    // Kernel Launch
    cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smemBytes};
    cutlass::Status status =
        cutlass::launch_kernel_on_cluster(params, (void const*)kernel_ptr, prob_shape, cta_tiler, A,
                                          tmaA, B, tmaB, C, dC, tiled_mma, alpha, beta);
    CUTE_CHECK_LAST();

    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Error: Failed at kernel Launch" << std::endl;
    }
}

template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm(int m, int n, int k, Alpha alpha, TA const* A, int ldA, TB const* B, int ldB, Beta beta,
          TC* C, int ldC, cudaStream_t stream = 0) {
    return gemm_tn(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
}

int main(int argc, char** argv) {
    int m = 128;
    if (argc >= 2) sscanf(argv[1], "%d", &m);

    int n = 128;
    if (argc >= 3) sscanf(argv[2], "%d", &n);

    int k = 64;
    if (argc >= 4) sscanf(argv[3], "%d", &k);

    using TA = cute::half_t;
    using TB = TA;
    using TC = TA;
    using TI = TA;

    TI alpha = TI(1.0f);
    TI beta = TI(0.0f);

    thrust::host_vector<TA> h_A(m * k);
    thrust::host_vector<TB> h_B(n * k);
    thrust::host_vector<TC> h_C(m * n);

    // Initialize the tensors
    for (int j = 0; j < m * k; ++j) h_A[j] = TA(j);
    for (int j = 0; j < n * k; ++j) h_B[j] = TB(j);
    for (int j = 0; j < m * n; ++j) h_C[j] = TC(0);

    thrust::device_vector<TA> d_A = h_A;
    thrust::device_vector<TB> d_B = h_B;
    thrust::device_vector<TC> d_C = h_C;

    int ldA = 0, ldB = 0, ldC = m;

    ldA = k;

    ldB = k;

    // Run once
    d_C = h_C;
    gemm(m, n, k, alpha, d_A.data().get(), ldA, d_B.data().get(), ldB, beta, d_C.data().get(), ldC);
    CUTE_CHECK_LAST();

    return 0;
}
