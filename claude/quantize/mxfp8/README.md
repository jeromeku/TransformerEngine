# MXFP8 Quantization Deep Dive

This directory contains comprehensive documentation for MXFP8 quantization in Transformer Engine.

## Overview

MXFP8 (Microscaling FP8) is a block-wise quantization format that uses:
- **32-element blocks** as the quantization unit
- **Shared E8M0 exponent** (8-bit exponent, 0-bit mantissa) per block
- **Dual-layout tensors** (rowwise + columnwise) for efficient GEMM operations

## Documentation Files

1. **[QUANTIZE_FRAME_BY_FRAME.md](QUANTIZE_FRAME_BY_FRAME.md)** - Complete frame-by-frame trace of the `quantize()` method
2. **[CALL_GRAPH.md](CALL_GRAPH.md)** - Visual call graph and function dispatch
3. **[DESIGN_RATIONALE.md](DESIGN_RATIONALE.md)** - Architecture decisions and trade-offs

## Quick Start

The main entry point for MXFP8 quantization is:

```python
from transformer_engine.pytorch import quantize

# Input: torch.Tensor (e.g., shape [1024, 2048], dtype float32/bfloat16)
# Quantizer: MXFP8Quantizer instance
output = quantize(input_tensor, quantizer)
# Output: MXFP8Tensor with quantized data + scale factors
```

## Key Source Files

### Python API
- [`../../../transformer_engine/pytorch/__init__.py`](../../../transformer_engine/pytorch/__init__.py) - Python exports
- [`../../../transformer_engine/pytorch/quantization.py`](../../../transformer_engine/pytorch/quantization.py) - MXFP8Quantizer class
- [`../../../transformer_engine/pytorch/tensor/mxfp8_tensor.py`](../../../transformer_engine/pytorch/tensor/mxfp8_tensor.py) - MXFP8Tensor class

### C++/CUDA Implementation
- [`../../../transformer_engine/pytorch/csrc/extensions/cast.cpp:33-79`](../../../transformer_engine/pytorch/csrc/extensions/cast.cpp#L33-L79) - Main `quantize()` function
- [`../../../transformer_engine/pytorch/csrc/quantizer.cpp:899-1134`](../../../transformer_engine/pytorch/csrc/quantizer.cpp#L899-L1134) - MXFP8Quantizer implementation
- [`../../../transformer_engine/pytorch/csrc/common.cpp:51-64`](../../../transformer_engine/pytorch/csrc/common.cpp#L51-L64) - Quantizer dispatch
- [`../../../transformer_engine/pytorch/csrc/extensions/pybind.cpp:55-68`](../../../transformer_engine/pytorch/csrc/extensions/pybind.cpp#L55-L68) - Python binding initialization
- [`../../../transformer_engine/common/cast/cast.cu`](../../../transformer_engine/common/cast/cast.cu) - CUDA kernels

## Core Concepts

### Block-Wise Quantization

MXFP8 divides tensors into 32-element blocks, each with a shared exponent:

```
Input:  [0.5, -1.2, 0.8, ..., 0.3] (32 float32 values)
        ↓
Max:    1.2 → Shared exponent: E8M0(2^0)
        ↓
Output: [FP8(0.5), FP8(-1.2), FP8(0.8), ..., FP8(0.3)] + E8M0 scale
```

### Dual Layout Strategy

MXFP8 maintains both layouts for efficient matrix operations:

- **Rowwise**: For A×B where A is quantized
- **Columnwise**: For A×B^T (transpose) operations

This eliminates runtime transpose overhead in attention and feedforward layers.

### Scale Buffer Layout

For a tensor with shape `[M, N]`:

**Rowwise scales**: `[roundup(M, 128), roundup(N/32, 4)]`
- One scale per 32 elements in the last dimension
- Padded for 128-byte alignment

**Columnwise scales**: `[roundup(M/32, 4), roundup(N, 128)]`
- One scale per 32 elements in the first dimension
- Padded for coalesced memory access

## Performance Characteristics

| Aspect | Value |
|--------|-------|
| Block size | 32 elements |
| Scale storage | 1 byte per 32 elements (3% overhead) |
| Precision | Better than per-tensor FP8 |
| Compute | Optimized for NVIDIA SM_90+ |
| Memory bandwidth | ~4× reduction vs FP32 |

## Related Documentation

- MXFP8 usage examples: [`../../mx/mxfp8/`](../../mx/mxfp8/)
- Test walkthroughs: [`../../mx_tests/06_mxfp8_quantization.md`](../../mx_tests/06_mxfp8_quantization.md)
- NVFP4 comparison: [`../../mx/nvfp4/`](../../mx/nvfp4/)
