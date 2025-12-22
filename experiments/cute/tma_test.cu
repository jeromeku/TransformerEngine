#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cstdint>
#include <cute/tensor.hpp>
#include <iostream>

#include "cute/arch/copy_sm90_tma.hpp"
#include "cute/atom/mma_traits_sm90_gmma.hpp"
#include "cute/numeric/int.hpp"
#include "cute/numeric/numeric_types.hpp"
#include "debug.hpp"

template <class ElementType, class SmemLayout>
struct SharedStorage {
    cute::ArrayEngine<ElementType, cute::cosize_v<SmemLayout>> smem;
    alignas(16) cute::uint64_t tma_load_mbar[1];
};
template <typename T>
void print_matrix(const T* mat, int num_rows, int num_cols) {
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            int idx = i * num_cols + j;  // assume row-major
            printf("%4d ", mat[idx]);
        }
        printf("\n");
    }
}
template <class T, class TiledCopy, class CTA_Tiler, class GmemLayout, class SmemLayout>
__global__ void tma_test_device_cute(T const* g_in, T* g_out,
                                     CUTE_GRID_CONSTANT TiledCopy const tma, CTA_Tiler cta_tiler,
                                     GmemLayout gmem_layout, SmemLayout smem_layout) {
    using namespace cute;
    CUTE_STATIC_ASSERT_V(product_each(shape(cta_tiler)) == product_each(shape(smem_layout)));

    // Use Shared Storage structure to allocate and distribute aligned SMEM addresses
    extern __shared__ char shared_memory[];
    using SharedStorage = SharedStorage<T, SmemLayout>;
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(shared_memory);

    // Construct SMEM tensor
    Tensor sA = make_tensor(make_smem_ptr(shared_storage.smem.begin()),
                            smem_layout);  // (CTA_TILE_M,CTA_TILE_N,...)
    // Shared memory barriers use 64bits in SMEM for synchronization
    uint64_t* tma_load_mbar = shared_storage.tma_load_mbar;
    T& smemA = reinterpret_cast<T&>(raw_pointer_cast(shared_storage.smem.begin()));
    printf("BEFORE LOAD:\n");
    print_matrix(smemA, 4, 64);

    int numRows = size<0>(smem_layout);
    int numCols = size<1>(smem_layout);

    // TMA requires special handling of strides to deal with coord codomain mapping
    // Represent the full tensors -- get these from TMA
    Tensor mA = tma.get_tma_tensor(shape(gmem_layout));
    Tensor mB = make_tensor(make_gmem_ptr<T>(g_out), gmem_layout);

    constexpr int R = rank_v<CTA_Tiler>;
    Tensor gA = flat_divide(mA, cta_tiler);  // (CTA_TILE_M,CTA_TILE_N,...REST_M,REST_N,...)
    Tensor gB = flat_divide(mB, cta_tiler);  // (CTA_TILE_M,CTA_TILE_N,...REST_M,REST_N,...)

    //
    // Prepare the TMA_LOAD
    //

    auto cta_tma = tma.get_slice(Int<0>{});   // CTA slice
    Tensor tAgA_x = cta_tma.partition_S(gA);  // (TMA,TMA_M,TMA_N,REST_M,REST_N)
    Tensor tAsA_x = cta_tma.partition_D(sA);  // (TMA,TMA_M,TMA_N)

    if (thread0()) {
        print(tma);
        print("TILE  :  ");
        print(cta_tiler);
        print("\n");
        print("  mA  :  ");
        print(mA);
        print("\n");
        print("  mB  :  ");
        print(mB);
        print("\n");
        print("  gA  :  ");
        print(gA);
        print("\n");
        print("  gB  :  ");
        print(gB);
        print("\n");
        print("  sA  :  ");
        print(sA);
        print("\n");
        print("tAgA_x:  ");
        print(tAgA_x);
        print("\n");
        print("tAsA_x:  ");
        print(tAsA_x);
        print("\n");
    }

    //
    // Perform the TMA_LOAD
    //

    // INPUT: Group the REST_X modes and the TMA_X modes to easily iterate through the tiles
    Tensor tAgA = group_modes<1, rank(tAgA_x)>(tAgA_x);  // (TMA,REST)
    Tensor tAsA = group_modes<1, rank(tAsA_x)>(tAsA_x);  // (TMA,REST)
    static_assert(size<1>(tAsA) == 1);

    // OUTPUT: Group the CTA_TILE_X modes and REST_X modes for output
    Tensor tBgB = group_modes<0, R>(group_modes<R, rank(gB)>(gB));  // (CTA_TILE, REST)
    ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);
    Tensor tCsA = thr_mma.partition_A(sA);  // (MMA,MMA_M,MMA_K,PIPE)

    if (thread0()) {
        print("tAgA  :  ");
        print(tAgA);
        print("\n");
        print("tAsA  :  ");
        print(tAsA);
        print("\n");
        print("tBgB  :  ");
        print(tBgB);
        print("\n");
    }
    // Test L2 prefetch
    if (threadIdx.x == 0) {
        prefetch(tma, tAgA);
    }

    // Loop over the TMA stages, using smem as our buffer
    for (int stage = 0; stage < size<1>(tAgA); ++stage) {
        // Set the bytes transferred in this TMA transaction (may involve multiple issues)
        constexpr int kTmaTransactionBytes = sizeof(make_tensor_like(tensor<0>(tAsA)));

        if (threadIdx.x == 0) {
            /// Initialize shared memory barrier
            tma_load_mbar[0] = 0;
            cute::initialize_barrier(tma_load_mbar[0], 1 /*numThreads*/);
            cute::set_barrier_transaction_bytes(tma_load_mbar[0], kTmaTransactionBytes);

            copy(tma.with(tma_load_mbar[0]), tAgA(_, stage), tAsA(_, 0));
        }
        __syncthreads();

        /// Wait on the shared memory barrier until the phase bit flips from kPhaseBit value
        constexpr int kPhaseBit = 0;
        cute::wait_barrier(tma_load_mbar[0], kPhaseBit);
        printf("AFTER LOAD:\n");
        print_matrix(smemA, 4, 64);

        // //
        // // Write out trivially smem -> gmem
        // //

        // // Subbyte elements could cause race conditions, so be even more conservative
        // if (thread0()) {
        //   copy(sA, tBgB(_,stage));
        // }

        __syncthreads();
    }
}

int main() {
    using namespace cute;
    using CopyOp = SM90_TMA_LOAD;
    using T = cute::uint16_t;
    using SwizzleAtom = cute::SM90::GMMA::Layout_K_SW128_Atom<T>;

    auto copy_op = CopyOp{};
    using TileShape = Shape<_128, _128>;
    auto atom_layout = SwizzleAtom{};
    auto smem_layout = tile_to_shape(atom_layout, TileShape{});
    auto cta_tile = product_each(shape(smem_layout));
    constexpr int num_tiles_m = 1;
    constexpr int num_tiles_n = 1;
    Layout gmem_layout = make_layout(make_shape(num_tiles_m * uint32_t(size<0>(smem_layout)),
                                                num_tiles_n * uint32_t(size<1>(smem_layout))),
                                     GenRowMajor{});

    // Allocate and initialize host test data
    size_t N = cosize(gmem_layout);
    thrust::host_vector<T> h_in(N);
    auto ncols = size<1>(gmem_layout);
    for (size_t i = 0; i < h_in.size(); ++i) {
        h_in[i] = i % ncols;
    }
    Tensor hA_in = make_tensor(recast_ptr<T>(h_in.data()), gmem_layout);

    // Allocate and initialize device test data
    thrust::device_vector<T> d_in = h_in;
    thrust::device_vector<T> d_out(h_in.size(), uint8_t(-1));  // overflow uint

    // Create TMA for this device Tensor
    Tensor gA = make_tensor(make_gmem_ptr<T>(raw_pointer_cast(d_in.data())), gmem_layout);
    PRINT_CUTE(gA);
    PRINT_CUTE(atom_layout);
    PRINT_CUTE(smem_layout);

    auto tma = make_tma_copy<T>(copy_op, gA, smem_layout, cta_tile, Int<1>{});
    PRINT_CUTE(tma);

    // Launch
    int smem_size = int(sizeof(SharedStorage<T, decltype(smem_layout)>));
    int num_blocks = 1;
    int num_threads = 1;
    // tma_test_device_cute<<<num_blocks, num_threads, smem_size>>>(
    //     reinterpret_cast<T const*>(raw_pointer_cast(d_in.data())),
    //     reinterpret_cast<T*>(raw_pointer_cast(d_out.data())), tma, cta_tile, gmem_layout,
    //     smem_layout);

    return 0;
}
