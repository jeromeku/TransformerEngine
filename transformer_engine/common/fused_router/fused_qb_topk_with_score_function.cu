/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_runtime.h>
#include <transformer_engine/fused_router.h>

#include "../common.h"
#include "../util/logging.h"
#include "utils.h"

namespace transformer_engine {
namespace fused_router {

template <typename DataType, TopkFuncType TopkFunc = TopkFuncType::Naive>
__global__ void fused_qb_topk_with_score_function_forward_kernel(
    const DataType *logits, const CompType *beta, int num_tokens, int num_experts, int topk,
    float scaling_factor, DataType *probs, bool *routing_map, CompType *alpha,
    CompType *intermediate_output) {
  const int num_tokens_per_block = blockDim.x / kThreadsPerWarp;
  const int warp_id = threadIdx.x / kThreadsPerWarp;
  const int lane_id = threadIdx.x % kThreadsPerWarp;
  const int selection_k = topk + 1;

  extern __shared__ float shmem[];
  CompType *adjusted_scores_buf = reinterpret_cast<CompType *>(shmem);
  CompType *selected_scores_buf =
      adjusted_scores_buf + num_experts * num_tokens_per_block;
  int *selected_indices_buf = reinterpret_cast<int *>(
      selected_scores_buf + selection_k * num_tokens_per_block);

  CompType *adjusted_scores = adjusted_scores_buf + warp_id * num_experts;
  CompType *selected_scores = selected_scores_buf + warp_id * selection_k;
  int *selected_indices = selected_indices_buf + warp_id * selection_k;

  const int total_rounds =
      (num_tokens + num_tokens_per_block - 1) / num_tokens_per_block;
  for (int round = blockIdx.x; round < total_rounds; round += gridDim.x) {
    const int token = round * num_tokens_per_block + warp_id;
    if (token >= num_tokens) break;

    const int token_offset = token * num_experts;
    for (int expert = lane_id; expert < num_experts; expert += kThreadsPerWarp) {
      const CompType raw_logit = static_cast<CompType>(logits[token_offset + expert]);
      adjusted_scores[expert] = raw_logit - beta[expert];
      probs[token_offset + expert] = 0.0f;
      routing_map[token_offset + expert] = false;
      intermediate_output[token_offset + expert] = 0.0f;
    }
    __syncwarp();

    topk_and_mask<TopkFunc>(adjusted_scores, num_experts, selection_k, selected_indices,
                            selected_scores, lane_id);
    __syncwarp();

    // Radix top-k returns the selected set in index order rather than score order.
    // Find and remove the minimum member so the remaining set is exactly top-k.
    // For equal boundary values, remove the largest index to retain TE's
    // deterministic (value DESC, index ASC) tie-breaking convention.
    if (lane_id == 0) {
      int alpha_position = 0;
      for (int i = 1; i < selection_k; ++i) {
        if (selected_scores[i] < selected_scores[alpha_position] ||
            (selected_scores[i] == selected_scores[alpha_position] &&
             selected_indices[i] > selected_indices[alpha_position])) {
          alpha_position = i;
        }
      }
      alpha[token] = selected_scores[alpha_position];
      for (int i = alpha_position; i < topk; ++i) {
        selected_scores[i] = selected_scores[i + 1];
        selected_indices[i] = selected_indices[i + 1];
      }
    }
    __syncwarp();

    // QB bias changes assignment only. Combine weights use the original logits.
    for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
      const int expert = selected_indices[i];
      const CompType raw_logit = static_cast<CompType>(logits[token_offset + expert]);
      const CompType sigmoid_score = 1.0f / (1.0f + expf(-raw_logit));
      selected_scores[i] = sigmoid_score;
      intermediate_output[token_offset + expert] = sigmoid_score;
    }
    __syncwarp();

    if (topk > 1) {
      const CompType score_sum =
          warp_reduce_on_shmem(selected_scores, topk, ReduceFuncType::SUM, lane_id);
      for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
        selected_scores[i] /= score_sum + epsilon;
      }
    }
    __syncwarp();

    for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
      const int expert = selected_indices[i];
      routing_map[token_offset + expert] = true;
      probs[token_offset + expert] = scaling_factor * selected_scores[i];
    }
    __syncwarp();
  }
}

template <typename DataType>
void fused_qb_topk_with_score_function_forward_kernel_launcher(
    const DataType *logits, const CompType *beta, int num_tokens, int num_experts, int topk,
    float scaling_factor, DataType *probs, bool *routing_map, CompType *alpha,
    CompType *intermediate_output, cudaStream_t stream) {
  const size_t num_tokens_per_block = kThreadsPerBlock / kThreadsPerWarp;
  const size_t grid_size =
      (num_tokens + num_tokens_per_block - 1) / num_tokens_per_block;
  const size_t selection_k = topk + 1;
  const size_t shared_memory_size =
      num_experts * num_tokens_per_block * sizeof(CompType) +
      selection_k * num_tokens_per_block * sizeof(CompType) +
      selection_k * num_tokens_per_block * sizeof(int);
  check_shared_memory_capacity_num_experts(shared_memory_size, num_experts);

  if (selection_k < 16) {
    NVTE_CHECK_CUDA(cudaFuncSetAttribute(
        fused_qb_topk_with_score_function_forward_kernel<DataType, TopkFuncType::Naive>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size));
    fused_qb_topk_with_score_function_forward_kernel<DataType, TopkFuncType::Naive>
        <<<grid_size, kThreadsPerBlock, shared_memory_size, stream>>>(
            logits, beta, num_tokens, num_experts, topk, scaling_factor, probs, routing_map, alpha,
            intermediate_output);
  } else {
    NVTE_CHECK_CUDA(cudaFuncSetAttribute(
        fused_qb_topk_with_score_function_forward_kernel<DataType, TopkFuncType::Radix>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size));
    fused_qb_topk_with_score_function_forward_kernel<DataType, TopkFuncType::Radix>
        <<<grid_size, kThreadsPerBlock, shared_memory_size, stream>>>(
            logits, beta, num_tokens, num_experts, topk, scaling_factor, probs, routing_map, alpha,
            intermediate_output);
  }
  NVTE_CHECK_CUDA(cudaGetLastError());
}

void fused_qb_topk_with_score_function_forward(
    const Tensor logits, const Tensor beta, int num_tokens, int num_experts, int topk,
    float scaling_factor, Tensor probs, Tensor routing_map, Tensor alpha,
    Tensor intermediate_output, cudaStream_t stream) {
  NVTE_CHECK(beta.data.dtype == DType::kFloat32, "QB beta must have Float32 dtype.");
  TE_ROUTER_PROBS_TYPE_SWITCH_ALL(
      logits.data.dtype, DataType,
      fused_qb_topk_with_score_function_forward_kernel_launcher<DataType>(
          reinterpret_cast<DataType *>(logits.data.dptr),
          reinterpret_cast<CompType *>(beta.data.dptr), num_tokens, num_experts, topk,
          scaling_factor, reinterpret_cast<DataType *>(probs.data.dptr),
          reinterpret_cast<bool *>(routing_map.data.dptr),
          reinterpret_cast<CompType *>(alpha.data.dptr),
          reinterpret_cast<CompType *>(intermediate_output.data.dptr), stream););
}

}  // namespace fused_router
}  // namespace transformer_engine

void nvte_fused_qb_topk_with_score_function_forward(
    const NVTETensor logits, const NVTETensor beta, int num_tokens, int num_experts, int topk,
    float scaling_factor, NVTETensor probs, NVTETensor routing_map, NVTETensor alpha,
    NVTETensor intermediate_output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_qb_topk_with_score_function_forward);
  using namespace transformer_engine;
  fused_router::fused_qb_topk_with_score_function_forward(
      *convertNVTETensorCheck(logits), *convertNVTETensorCheck(beta), num_tokens, num_experts, topk,
      scaling_factor, *convertNVTETensorCheck(probs), *convertNVTETensorCheck(routing_map),
      *convertNVTETensorCheck(alpha), *convertNVTETensorCheck(intermediate_output), stream);
}
