# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch

from transformer_engine.pytorch.router import fused_qb_column_quantile


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _column_reference(
    scores: torch.Tensor,
    alpha: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    num_tokens, num_experts = scores.shape
    column_target = num_tokens * topk // num_experts
    residual = scores - alpha.reshape(num_tokens, 1)
    return residual.topk(column_target + 1, dim=0).values[-1]


@pytest.mark.parametrize(
    "num_tokens,num_experts,topk",
    [
        pytest.param(65, 13, 1, id="partial-tiles"),
        pytest.param(4096, 64, 8, id="m4096-e64-k8"),
        pytest.param(8192, 64, 8, id="m8192-e64-k8"),
        pytest.param(4096, 256, 8, id="m4096-e256-k8"),
        pytest.param(8192, 256, 8, id="m8192-e256-k8"),
    ],
)
@pytest.mark.parametrize("alpha_column", [False, True], ids=["alpha-vector", "alpha-column"])
def test_fused_qb_column_quantile_matches_reference(
    num_tokens: int,
    num_experts: int,
    topk: int,
    alpha_column: bool,
) -> None:
    torch.manual_seed(1234)
    scores = torch.randn(num_tokens, num_experts, device="cuda", dtype=torch.float32)
    alpha = torch.randn(num_tokens, device="cuda", dtype=torch.float32)
    if alpha_column:
        alpha = alpha.unsqueeze(1)

    expected = _column_reference(scores, alpha, topk)
    actual = fused_qb_column_quantile(scores, alpha, topk)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.shape == (num_experts,)
    assert actual.dtype == torch.float32
    assert not actual.requires_grad


def test_fused_qb_column_quantile_preserves_tie_threshold() -> None:
    num_tokens, num_experts, topk = 64, 16, 4
    scores = (
        torch.arange(num_tokens * num_experts, device="cuda", dtype=torch.float32)
        .remainder(7)
        .reshape(num_tokens, num_experts)
    )
    alpha = torch.arange(num_tokens, device="cuda", dtype=torch.float32).remainder(3)

    expected = _column_reference(scores, alpha, topk)
    actual = fused_qb_column_quantile(scores, alpha, topk)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_fused_qb_column_quantile_cuda_graph_capture() -> None:
    torch.manual_seed(9012)
    scores = torch.randn(256, 64, device="cuda", dtype=torch.float32)
    alpha = torch.randn(256, 1, device="cuda", dtype=torch.float32)

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        fused_qb_column_quantile(scores, alpha, 8)
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = fused_qb_column_quantile(scores, alpha, 8)
    graph.replay()
    torch.cuda.synchronize()

    eager = fused_qb_column_quantile(scores, alpha, 8)
    torch.testing.assert_close(captured, eager, atol=0, rtol=0)


@pytest.mark.parametrize(
    "scores_shape,alpha_shape,topk,error",
    [
        ((63, 16), (63,), 4, "divisible"),
        ((64, 16), (63,), 4, "alpha must have shape"),
        ((64, 16), (64,), 16, "smaller than num_experts"),
    ],
)
def test_fused_qb_column_quantile_rejects_invalid_shapes(
    scores_shape: tuple[int, int],
    alpha_shape: tuple[int, ...],
    topk: int,
    error: str,
) -> None:
    scores = torch.randn(*scores_shape, device="cuda", dtype=torch.float32)
    alpha = torch.randn(*alpha_shape, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match=error):
        fused_qb_column_quantile(scores, alpha, topk)


def test_fused_qb_column_quantile_requires_detached_fp32_inputs() -> None:
    scores = torch.randn(64, 16, device="cuda", dtype=torch.float32, requires_grad=True)
    alpha = torch.randn(64, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="scores must be detached"):
        fused_qb_column_quantile(scores, alpha, 4)

    with pytest.raises(RuntimeError, match="scores must be a float32 tensor"):
        fused_qb_column_quantile(scores.detach().bfloat16(), alpha, 4)
