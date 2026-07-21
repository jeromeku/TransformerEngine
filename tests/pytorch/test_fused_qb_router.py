# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch

from transformer_engine.pytorch.router import fused_qb_topk_with_score_function


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _qb_reference(
    logits: torch.Tensor,
    beta: torch.Tensor,
    topk: int,
    scaling_factor: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    adjusted_logits = logits.detach().float() - beta
    topk_result = adjusted_logits.topk(topk + 1, dim=1)
    indices = topk_result.indices[:, :topk]
    alpha = topk_result.values[:, -1:]

    selected_scores = torch.sigmoid(logits.float()).gather(1, indices)
    if topk > 1:
        selected_scores = selected_scores / (
            selected_scores.sum(dim=1, keepdim=True) + 1e-20
        )
    if scaling_factor is not None:
        selected_scores = selected_scores * scaling_factor

    probs = torch.zeros_like(logits).scatter(
        1, indices, selected_scores.to(dtype=logits.dtype)
    )
    routing_map = torch.zeros_like(logits, dtype=torch.bool).scatter(1, indices, True)
    return probs, routing_map, alpha


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "num_tokens,num_experts,topk",
    [
        pytest.param(257, 32, 1, id="top1-naive"),
        pytest.param(257, 64, 8, id="top8-naive"),
        pytest.param(257, 128, 16, id="top16-radix"),
    ],
)
@pytest.mark.parametrize("scaling_factor", [None, 1.25])
def test_fused_qb_topk_forward_backward(
    dtype: torch.dtype,
    num_tokens: int,
    num_experts: int,
    topk: int,
    scaling_factor: float | None,
) -> None:
    torch.manual_seed(1234)
    logits = torch.randn(
        num_tokens, num_experts, device="cuda", dtype=dtype, requires_grad=True
    )
    logits_ref = logits.detach().clone().requires_grad_(True)
    beta = torch.randn(num_experts, device="cuda", dtype=torch.float32)
    beta_before = beta.clone()

    probs_ref, routing_map_ref, alpha_ref = _qb_reference(
        logits_ref, beta, topk, scaling_factor
    )
    probs, routing_map, alpha = fused_qb_topk_with_score_function(
        logits, beta, topk, scaling_factor=scaling_factor
    )

    assert torch.equal(routing_map, routing_map_ref)
    torch.testing.assert_close(alpha, alpha_ref, atol=0, rtol=0)
    assert not alpha.requires_grad
    torch.testing.assert_close(beta, beta_before, atol=0, rtol=0)

    tolerance = {
        torch.float32: (1e-6, 1e-6),
        torch.float16: (2e-3, 2e-3),
        torch.bfloat16: (2e-2, 2e-2),
    }[dtype]
    torch.testing.assert_close(probs, probs_ref, atol=tolerance[0], rtol=tolerance[1])

    grad_probs = torch.randn_like(probs)
    (probs * grad_probs).sum().backward()
    (probs_ref * grad_probs).sum().backward()
    torch.testing.assert_close(
        logits.grad, logits_ref.grad, atol=tolerance[0], rtol=tolerance[1]
    )


def test_fused_qb_alpha_drives_separate_column_update() -> None:
    torch.manual_seed(5678)
    num_tokens, num_experts, topk = 256, 64, 8
    logits = torch.randn(num_tokens, num_experts, device="cuda", dtype=torch.bfloat16)
    beta = torch.randn(num_experts, device="cuda", dtype=torch.float32)

    _, _, alpha = fused_qb_topk_with_score_function(logits, beta, topk)
    _, _, alpha_ref = _qb_reference(logits, beta, topk, None)

    col_target = num_tokens * topk // num_experts
    scores = logits.float()
    beta_candidate = (scores - alpha).topk(col_target + 1, dim=0).values[-1]
    beta_candidate_ref = (scores - alpha_ref).topk(col_target + 1, dim=0).values[-1]

    torch.testing.assert_close(beta_candidate, beta_candidate_ref, atol=0, rtol=0)


def test_fused_qb_topk_cuda_graph_capture() -> None:
    torch.manual_seed(9012)
    logits = torch.randn(256, 64, device="cuda", dtype=torch.bfloat16)
    beta = torch.randn(64, device="cuda", dtype=torch.float32)

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        fused_qb_topk_with_score_function(logits, beta, 8)
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = fused_qb_topk_with_score_function(logits, beta, 8)
    graph.replay()
    torch.cuda.synchronize()

    eager = fused_qb_topk_with_score_function(logits, beta, 8)
    torch.cuda.synchronize()
    for captured_output, eager_output in zip(captured, eager):
        torch.testing.assert_close(captured_output, eager_output, atol=0, rtol=0)
