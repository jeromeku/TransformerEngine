/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_runtime.h>
#include <transformer_engine/fused_router.h>

#include "../common.h"
#include "../util/logging.h"
#include "../utils.cuh"

namespace transformer_engine {
namespace fused_router {

constexpr int kQBTileDim = 32;
constexpr int kQBBlockRows = 8;
constexpr int kQBRadixBits = 8;
constexpr int kQBRadixBuckets = 1 << kQBRadixBits;

/*
 * Stage 1: residual construction and transpose
 * --------------------------------------------
 * scores is row-major [M, E], while the order statistic consumes one expert
 * column at a time. A 32x8 CTA cooperatively moves a 32x32 tile: each thread
 * handles four rows on load and four experts on store. Consecutive x lanes
 * therefore read consecutive experts and write consecutive tokens.
 *
 * The shared tile has a 33-float stride. Padding the logical 32x32 tile by one
 * column prevents the transposed shared-memory access from mapping a warp to
 * the same bank repeatedly.
 */
__global__ void qb_residual_transpose_kernel(const float *scores, const float *alpha,
                                             int num_tokens, int num_experts,
                                             float *transposed_residual) {
  __shared__ float tile[kQBTileDim][kQBTileDim + 1];

  const int input_expert = blockIdx.x * kQBTileDim + threadIdx.x;
  const int input_token = blockIdx.y * kQBTileDim + threadIdx.y;
  for (int offset = 0; offset < kQBTileDim; offset += kQBBlockRows) {
    const int token = input_token + offset;
    if (input_expert < num_experts && token < num_tokens) {
      tile[threadIdx.y + offset][threadIdx.x] =
          scores[static_cast<size_t>(token) * num_experts + input_expert] - alpha[token];
    }
  }
  __syncthreads();

  const int output_token = blockIdx.y * kQBTileDim + threadIdx.x;
  const int output_expert = blockIdx.x * kQBTileDim + threadIdx.y;
  for (int offset = 0; offset < kQBTileDim; offset += kQBBlockRows) {
    const int expert = output_expert + offset;
    if (expert < num_experts && output_token < num_tokens) {
      transposed_residual[static_cast<size_t>(expert) * num_tokens + output_token] =
          tile[threadIdx.x][threadIdx.y + offset];
    }
  }
}

/*
 * Stage 2: exact per-expert order statistic
 * -----------------------------------------
 * One 256-thread CTA owns one expert. The block scans that expert's contiguous
 * token vector four times, resolving eight ordered-float bits per pass from
 * most significant to least significant. Thread i also owns histogram bucket
 * i during initialization; shared-memory atomicAdd combines counts from all
 * threads. The algorithm emits only the threshold value, not K values/indices.
 */
__global__ void qb_column_quantile_kernel(const float *transposed_residual, int num_tokens,
                                          int num_experts, int column_k,
                                          float *beta_candidate) {
  const int expert = blockIdx.x;
  if (expert >= num_experts) return;

  __shared__ unsigned int histogram[kQBRadixBuckets];
  __shared__ unsigned int desired;
  __shared__ unsigned int desired_mask;
  __shared__ int k_remaining;

  if (threadIdx.x == 0) {
    desired = 0;
    desired_mask = 0;
    k_remaining = column_k;
  }
  __syncthreads();

  const float *column = transposed_residual + static_cast<size_t>(expert) * num_tokens;
  for (int shift = 32 - kQBRadixBits; shift >= 0; shift -= kQBRadixBits) {
    histogram[threadIdx.x] = 0;
    __syncthreads();

    // Loop invariant: current_desired/current_mask describe the prefix of the
    // column_k-th largest ordered-float bit pattern fixed by earlier passes.
    const unsigned int current_desired = desired;
    const unsigned int current_mask = desired_mask;
    for (int token = threadIdx.x; token < num_tokens; token += blockDim.x) {
      const unsigned int bits = float_to_ordered_uint(column[token]);
      if ((bits & current_mask) == current_desired) {
        const unsigned int bucket = (bits >> shift) & (kQBRadixBuckets - 1);
        atomicAdd(histogram + bucket, 1u);
      }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      // Descending bucket scan skips values known to be larger, then retains
      // the rank within the selected bucket for the next eight-bit pass.
      int next_k = k_remaining;
      int selected_bucket = 0;
      for (int bucket = kQBRadixBuckets - 1; bucket >= 0; --bucket) {
        const int count = static_cast<int>(histogram[bucket]);
        if (count < next_k) {
          next_k -= count;
        } else {
          selected_bucket = bucket;
          break;
        }
      }
      desired |= static_cast<unsigned int>(selected_bucket) << shift;
      desired_mask |= static_cast<unsigned int>(kQBRadixBuckets - 1) << shift;
      k_remaining = next_k;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    beta_candidate[expert] = ordered_uint_to_float(desired);
  }
}

void fused_qb_column_quantile(const Tensor &scores, const Tensor &alpha, int num_tokens,
                              int num_experts, int column_k, Tensor &workspace,
                              Tensor &beta_candidate, cudaStream_t stream) {
  NVTE_CHECK(scores.data.dtype == DType::kFloat32, "QB column scores must have Float32 dtype.");
  NVTE_CHECK(alpha.data.dtype == DType::kFloat32, "QB column alpha must have Float32 dtype.");
  NVTE_CHECK(workspace.data.dtype == DType::kFloat32,
             "QB column workspace must have Float32 dtype.");
  NVTE_CHECK(beta_candidate.data.dtype == DType::kFloat32,
             "QB column output must have Float32 dtype.");
  NVTE_CHECK(column_k > 0 && column_k <= num_tokens, "QB column_k must be in [1, num_tokens].");

  const dim3 transpose_block(kQBTileDim, kQBBlockRows);
  const dim3 transpose_grid((num_experts + kQBTileDim - 1) / kQBTileDim,
                            (num_tokens + kQBTileDim - 1) / kQBTileDim);
  qb_residual_transpose_kernel<<<transpose_grid, transpose_block, 0, stream>>>(
      reinterpret_cast<const float *>(scores.data.dptr),
      reinterpret_cast<const float *>(alpha.data.dptr), num_tokens, num_experts,
      reinterpret_cast<float *>(workspace.data.dptr));
  NVTE_CHECK_CUDA(cudaGetLastError());

  // Launching both kernels on the same caller-provided stream establishes the
  // workspace dependency without a device- or host-wide synchronization.
  qb_column_quantile_kernel<<<num_experts, kQBRadixBuckets, 0, stream>>>(
      reinterpret_cast<const float *>(workspace.data.dptr), num_tokens, num_experts, column_k,
      reinterpret_cast<float *>(beta_candidate.data.dptr));
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace fused_router
}  // namespace transformer_engine

void nvte_fused_qb_column_quantile(const NVTETensor scores, const NVTETensor alpha, int num_tokens,
                                   int num_experts, int column_k, NVTETensor workspace,
                                   NVTETensor beta_candidate, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_qb_column_quantile);
  using namespace transformer_engine;
  fused_router::fused_qb_column_quantile(
      *convertNVTETensorCheck(scores), *convertNVTETensorCheck(alpha), num_tokens, num_experts,
      column_k, *convertNVTETensorCheck(workspace), *convertNVTETensorCheck(beta_candidate), stream);
}
