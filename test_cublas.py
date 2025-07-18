from transformer_engine.pytorch.cpp_extensions import general_gemm
from transformer_engine.pytorch.module.base import get_workspace

import torch
from typing import Optional, List

def te_general_gemm(
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: Optional[torch.dtype] = None,
        layout: str = "TN",
        out: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        grad: bool = False,
        workspace = None
    ) -> List[torch.Tensor]:
        return general_gemm(
            A,
            B,
            workspace=workspace,
            out_dtype=out_dtype,
            quantization_params=None,
            gelu=None,
            gelu_in=None,
            accumulate=False,
            layout=layout,
            out=out,
            bias=bias,
            use_split_accumulator=False,
            grad=grad,
            ub=None,
            ub_type=None,
            extra_output=None,
            bulk_overlap=False,
        )

def main(layout="NN"):
    m = 1024
    k = 256
    n = 512
    dtype = torch.float16
    device = "cuda"
    A = torch.randn(m, k, dtype=dtype, device=device)
    """
    C = A @ B
    A: [M x K] row-major => "T", equivalent to [K x M] col-major => "N"
    B: [K x N] row-major => "T", equivalent to [N x K] col-major => "N"
    C: [M x N] row-major => "T", equivalent to [N x M] col-major => "N"

    
    A: [K x M] K-major
    B: [N x K] K-major
    C: [N x M] N-major
    N x M = (N x K) x (K x M) = N x M N-major

    "TN"

    Cublas
    "TTT" => "NNN" with A and B flipped
    """

    
    if layout == "NN":
    #    Cref2 = (B.T @ A.T).T
        B = torch.randn(k, n, dtype=dtype, device=device)
        Cref = A @ B
        workspace = get_workspace()
    
        out = te_general_gemm(B, A, layout=layout, out_dtype=dtype, workspace=workspace)


    elif layout == "TN":
        B = torch.randn(n, k, dtype=dtype, device=device)
        Cref = A @ B.T
        workspace = get_workspace()
        out = te_general_gemm(B, A, layout=layout, out_dtype=dtype, workspace=workspace)
    
    C = out[0]
    diff = (Cref - C).abs().max()
    print(f"{diff.item():.4f}")

main("TN")