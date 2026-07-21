/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <limits>

#include "../extensions.h"
#include "common.h"

namespace transformer_engine::pytorch {

static std::map<std::string, int> score_function_map = {
    {"sigmoid", 0}, {"softmax", 1}, {"sqrtsoftplus", 2}};

std::tuple<at::Tensor, at::Tensor, at::Tensor> fused_topk_with_score_function_fwd(
    at::Tensor logits, int topk, bool use_pre_softmax, std::optional<int> num_groups,
    std::optional<int> group_topk, std::optional<float> scaling_factor, std::string score_function,
    std::optional<at::Tensor> expert_bias) {
  int num_tokens = logits.size(0);
  int num_experts = logits.size(1);
  // Check if the input is valid
  TORCH_CHECK(num_tokens > 0 && num_experts > 0,
              "num_tokens and num_experts must be greater than 0");
  // Expert bias only happens at the sigmoid case
  if (expert_bias.has_value()) {
    TORCH_CHECK(score_function == "sigmoid" || score_function == "sqrtsoftplus",
                "score_function must be sigmoid or sqrtsoftplus when expert_bias is not None");
    TORCH_CHECK(expert_bias.value().scalar_type() == at::kFloat,
                "expert_bias must be a float32 tensor");
  }
  // Check if the score function is valid
  TORCH_CHECK(score_function == "softmax" || score_function == "sigmoid" ||
                  score_function == "sqrtsoftplus",
              "score_function must be softmax, sigmoid or sqrtsoftplus for router fusion");
  if (score_function == "sigmoid" || score_function == "sqrtsoftplus") {
    use_pre_softmax = false;  // Pre-softmax only happens at the softmax case
  }

  // Reformat the input to make it compatible with the kernel
  int group_topk_value = group_topk.has_value() ? group_topk.value() : -1;
  int num_groups_value = num_groups.has_value() ? num_groups.value() : -1;
  float scaling_factor_value = scaling_factor.has_value() ? scaling_factor.value() : 1.0f;

  // Construct the output tensor
  at::Tensor probs =
      at::empty({num_tokens, num_experts}, at::dtype(logits.scalar_type()).device(at::kCUDA));
  at::Tensor routing_map =
      at::empty({num_tokens, num_experts}, at::dtype(at::kBool).device(at::kCUDA));
  // Intermediate output is used to store the output of the softmax/sigmoid function
  at::Tensor intermediate_output =
      at::empty({num_tokens, num_experts}, at::dtype(at::kFloat).device(at::kCUDA));

  auto logits_cu = makeTransformerEngineTensor(logits);
  auto probs_cu = makeTransformerEngineTensor(probs);
  auto routing_map_cu = makeTransformerEngineTensor(routing_map);
  auto intermediate_output_cu = makeTransformerEngineTensor(intermediate_output);
  auto expert_bias_cu = TensorWrapper();  // empty expert_bias_cu tensor
  if (expert_bias.has_value()) {
    expert_bias_cu = makeTransformerEngineTensor(expert_bias.value());
  }

  nvte_fused_topk_with_score_function_forward(
      logits_cu.data(), num_tokens, num_experts, topk, use_pre_softmax, num_groups_value,
      group_topk_value, scaling_factor_value, score_function_map[score_function],
      expert_bias_cu.data(), probs_cu.data(), routing_map_cu.data(), intermediate_output_cu.data(),
      at::cuda::getCurrentCUDAStream());

  return std::make_tuple(probs, routing_map, intermediate_output);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
fused_qb_topk_with_score_function_fwd(at::Tensor logits, at::Tensor beta, int topk,
                                      std::optional<float> scaling_factor,
                                      std::string score_function) {
  TORCH_CHECK(logits.dim() == 2, "logits must be a 2D tensor");
  TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor");
  TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");
  TORCH_CHECK(beta.is_cuda(), "beta must be a CUDA tensor");
  TORCH_CHECK(beta.is_contiguous(), "beta must be contiguous");
  TORCH_CHECK(beta.scalar_type() == at::kFloat, "beta must be a float32 tensor");
  TORCH_CHECK(beta.device() == logits.device(), "beta and logits must be on the same device");
  TORCH_CHECK(score_function == "sigmoid",
              "QB router fusion currently supports only the sigmoid score function");

  const int num_tokens = logits.size(0);
  const int num_experts = logits.size(1);
  TORCH_CHECK(num_tokens > 0 && num_experts > 0,
              "num_tokens and num_experts must be greater than 0");
  TORCH_CHECK(topk > 0 && topk < num_experts,
              "topk must be greater than 0 and smaller than num_experts");
  TORCH_CHECK(beta.dim() == 1 && beta.numel() == num_experts,
              "beta must have shape [num_experts]");

  const float scaling_factor_value = scaling_factor.has_value() ? scaling_factor.value() : 1.0f;
  at::Tensor probs = at::empty_like(logits);
  at::Tensor routing_map =
      at::empty({num_tokens, num_experts}, at::dtype(at::kBool).device(logits.device()));
  at::Tensor alpha =
      at::empty({num_tokens, 1}, at::dtype(at::kFloat).device(logits.device()));
  at::Tensor intermediate_output =
      at::empty({num_tokens, num_experts}, at::dtype(at::kFloat).device(logits.device()));

  auto logits_cu = makeTransformerEngineTensor(logits);
  auto beta_cu = makeTransformerEngineTensor(beta);
  auto probs_cu = makeTransformerEngineTensor(probs);
  auto routing_map_cu = makeTransformerEngineTensor(routing_map);
  auto alpha_cu = makeTransformerEngineTensor(alpha);
  auto intermediate_output_cu = makeTransformerEngineTensor(intermediate_output);

  nvte_fused_qb_topk_with_score_function_forward(
      logits_cu.data(), beta_cu.data(), num_tokens, num_experts, topk, scaling_factor_value,
      probs_cu.data(), routing_map_cu.data(), alpha_cu.data(), intermediate_output_cu.data(),
      at::cuda::getCurrentCUDAStream());

  return std::make_tuple(probs, routing_map, alpha, intermediate_output);
}

at::Tensor fused_qb_column_quantile(at::Tensor scores, at::Tensor alpha, int topk) {
  TORCH_CHECK(scores.dim() == 2, "scores must be a 2D tensor");
  TORCH_CHECK(scores.is_cuda(), "scores must be a CUDA tensor");
  TORCH_CHECK(scores.is_contiguous(), "scores must be contiguous");
  TORCH_CHECK(scores.scalar_type() == at::kFloat, "scores must be a float32 tensor");
  TORCH_CHECK(!scores.requires_grad(), "scores must be detached");
  TORCH_CHECK(alpha.is_cuda(), "alpha must be a CUDA tensor");
  TORCH_CHECK(alpha.is_contiguous(), "alpha must be contiguous");
  TORCH_CHECK(alpha.scalar_type() == at::kFloat, "alpha must be a float32 tensor");
  TORCH_CHECK(!alpha.requires_grad(), "alpha must be detached");
  TORCH_CHECK(alpha.device() == scores.device(), "alpha and scores must be on the same device");

  const int64_t num_tokens_64 = scores.size(0);
  const int64_t num_experts_64 = scores.size(1);
  TORCH_CHECK(num_tokens_64 > 0 && num_experts_64 > 0,
              "num_tokens and num_experts must be greater than 0");
  TORCH_CHECK(num_tokens_64 <= std::numeric_limits<int>::max() &&
                  num_experts_64 <= std::numeric_limits<int>::max(),
              "scores dimensions exceed the supported int32 range");
  TORCH_CHECK(topk > 0 && topk < num_experts_64,
              "topk must be greater than 0 and smaller than num_experts");
  TORCH_CHECK((alpha.dim() == 1 || (alpha.dim() == 2 && alpha.size(1) == 1)) &&
                  alpha.numel() == num_tokens_64,
              "alpha must have shape [num_tokens] or [num_tokens, 1]");

  const int64_t routed_slots = num_tokens_64 * topk;
  TORCH_CHECK(routed_slots % num_experts_64 == 0,
              "num_tokens * topk must be divisible by num_experts");
  const int64_t column_target = routed_slots / num_experts_64;
  const int num_tokens = static_cast<int>(num_tokens_64);
  const int num_experts = static_cast<int>(num_experts_64);
  const int column_k = static_cast<int>(column_target + 1);

  at::Tensor workspace = at::empty(
      {num_experts, num_tokens}, at::dtype(at::kFloat).device(scores.device()));
  at::Tensor beta_candidate =
      at::empty({num_experts}, at::dtype(at::kFloat).device(scores.device()));
  auto scores_cu = makeTransformerEngineTensor(scores);
  auto alpha_cu = makeTransformerEngineTensor(alpha);
  auto workspace_cu = makeTransformerEngineTensor(workspace);
  auto beta_candidate_cu = makeTransformerEngineTensor(beta_candidate);

  nvte_fused_qb_column_quantile(
      scores_cu.data(), alpha_cu.data(), num_tokens, num_experts, column_k, workspace_cu.data(),
      beta_candidate_cu.data(), at::cuda::getCurrentCUDAStream());
  return beta_candidate;
}

void fused_topk_with_score_function_bwd(int num_tokens, int num_experts, at::Tensor routing_map,
                                        at::Tensor intermediate_output, at::Tensor grad_probs,
                                        at::Tensor grad_logits, int topk, bool use_pre_softmax,
                                        std::optional<float> scaling_factor,
                                        std::string score_function) {
  // Get the value of the parameters
  auto scaling_factor_value = scaling_factor.has_value() ? scaling_factor.value() : 1.0f;
  auto score_function_value = score_function_map[score_function];

  auto routing_map_cu = makeTransformerEngineTensor(routing_map);
  auto intermediate_output_cu = makeTransformerEngineTensor(intermediate_output);
  auto grad_probs_cu = makeTransformerEngineTensor(grad_probs);
  auto grad_logits_cu = makeTransformerEngineTensor(grad_logits);

  nvte_fused_topk_with_score_function_backward(
      routing_map_cu.data(), intermediate_output_cu.data(), grad_probs_cu.data(), num_tokens,
      num_experts, topk, use_pre_softmax, scaling_factor_value, score_function_value,
      grad_logits_cu.data(), at::cuda::getCurrentCUDAStream());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> fused_score_for_moe_aux_loss_fwd(
    at::Tensor logits, int topk, std::string score_function) {
  int num_tokens = logits.size(0);
  int num_experts = logits.size(1);
  // Check if the input is valid
  TORCH_CHECK(num_tokens > 0 && num_experts > 0,
              "num_tokens and num_experts must be greater than 0");
  TORCH_CHECK(topk > 0, "topk must be greater than 0");
  // Check if the score function is valid
  TORCH_CHECK(score_function == "softmax" || score_function == "sigmoid" ||
                  score_function == "sqrtsoftplus",
              "score_function must be softmax, sigmoid or sqrtsoftplus for router fusion");
  int score_function_value = score_function_map[score_function];

  // Construct the output tensor
  at::Tensor scores = at::empty({num_tokens, num_experts}, at::dtype(at::kFloat).device(at::kCUDA));
  at::Tensor routing_map =
      at::empty({num_tokens, num_experts}, at::dtype(at::kBool).device(at::kCUDA));
  at::Tensor intermediate_output =
      at::empty({num_tokens, num_experts}, at::dtype(at::kFloat).device(at::kCUDA));

  auto logits_cu = makeTransformerEngineTensor(logits);
  auto scores_cu = makeTransformerEngineTensor(scores);
  auto routing_map_cu = makeTransformerEngineTensor(routing_map);
  auto intermediate_output_cu = makeTransformerEngineTensor(intermediate_output);

  nvte_fused_score_for_moe_aux_loss_forward(
      logits_cu.data(), num_tokens, num_experts, topk, score_function_value, scores_cu.data(),
      routing_map_cu.data(), intermediate_output_cu.data(), at::cuda::getCurrentCUDAStream());

  return std::make_tuple(scores, routing_map, intermediate_output);
}

void fused_score_for_moe_aux_loss_bwd(int num_tokens, int num_experts,
                                      at::Tensor intermediate_output, at::Tensor grad_scores,
                                      at::Tensor grad_logits, int topk,
                                      std::string score_function) {
  // Get the value of the parameters
  int score_function_value = score_function_map[score_function];

  auto intermediate_output_cu = makeTransformerEngineTensor(intermediate_output);
  auto grad_scores_cu = makeTransformerEngineTensor(grad_scores);
  auto grad_logits_cu = makeTransformerEngineTensor(grad_logits);

  nvte_fused_score_for_moe_aux_loss_backward(
      intermediate_output_cu.data(), grad_scores_cu.data(), num_tokens, num_experts, topk,
      score_function_value, grad_logits_cu.data(), at::cuda::getCurrentCUDAStream());
}

std::tuple<at::Tensor, at::Tensor> fused_moe_aux_loss_fwd(at::Tensor probs,
                                                          at::Tensor tokens_per_expert,
                                                          int total_num_tokens, int num_experts,
                                                          int num_rows, int num_cols, int topk,
                                                          float coeff) {
  TORCH_CHECK(topk > 0, "topk must be greater than 0");
  TORCH_CHECK(total_num_tokens > 0, "total_num_tokens must be greater than 0");
  TORCH_CHECK(num_experts > 0, "num_experts must be greater than 0");

  // Create the output tensor
  at::Tensor aux_loss = at::empty({}, at::dtype(probs.scalar_type()).device(at::kCUDA));
  at::Tensor Const_buf = at::empty({2}, at::dtype(at::kFloat).device(at::kCUDA));

  auto probs_cu = makeTransformerEngineTensor(probs);
  auto tokens_per_expert_cu = makeTransformerEngineTensor(tokens_per_expert);
  auto aux_loss_cu = makeTransformerEngineTensor(aux_loss);
  auto Const_buf_cu = makeTransformerEngineTensor(Const_buf);

  nvte_fused_moe_aux_loss_forward(probs_cu.data(), tokens_per_expert_cu.data(), total_num_tokens,
                                  num_experts, num_rows, num_cols, topk, coeff, aux_loss_cu.data(),
                                  Const_buf_cu.data(), at::cuda::getCurrentCUDAStream());

  return std::make_tuple(aux_loss, Const_buf);
}

at::Tensor fused_moe_aux_loss_bwd(at::Tensor Const_buf, at::Tensor tokens_per_expert, int num_rows,
                                  int num_cols, at::Tensor grad_aux_loss) {
  // Create the output tensor
  at::Tensor grad_probs =
      at::empty({num_rows, num_cols}, at::dtype(grad_aux_loss.scalar_type()).device(at::kCUDA));

  auto Const_buf_cu = makeTransformerEngineTensor(Const_buf);
  auto tokens_per_expert_cu = makeTransformerEngineTensor(tokens_per_expert);
  auto grad_aux_loss_cu = makeTransformerEngineTensor(grad_aux_loss);
  auto grad_probs_cu = makeTransformerEngineTensor(grad_probs);

  // Meta data for the kernel
  nvte_fused_moe_aux_loss_backward(Const_buf_cu.data(), tokens_per_expert_cu.data(), num_rows,
                                   num_cols, grad_aux_loss_cu.data(), grad_probs_cu.data(),
                                   at::cuda::getCurrentCUDAStream());

  return grad_probs;
}

}  // namespace transformer_engine::pytorch
