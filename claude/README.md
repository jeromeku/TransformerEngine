# TransformerEngine Architecture Documentation

> **Comprehensive documentation of TransformerEngine's low-precision quantization system (MXFP8, NVFP4, Blockwise FP8)**
>
> Generated for: Internal developer reference, debugging, and extension
>
> Date: 2025
>
> Base Commit: bd55e7ba

---

## Documentation Structure

This directory contains comprehensive architecture documentation for TransformerEngine's mixed-precision training system, with a focus on low-precision formats (FP8, MXFP8, NVFP4).

### 📄 Files

1. **[ARCHITECTURE.md](ARCHITECTURE.md)** (95KB+)
   - Complete system architecture overview
   - Layer-by-layer breakdown (Python → C++ → CUDA)
   - Detailed implementation guides for MXFP8, NVFP4, and Blockwise FP8
   - Kernel implementation details and optimizations
   - Distributed training integration (FSDP, TP, SP, etc.)
   - **Use when**: Understanding overall system design or tracing call paths

2. **[test_nvfp4_walkthrough.md](test_nvfp4_walkthrough.md)** (55KB+)
   - Line-by-line walkthrough of NVFP4 GEMM tests
   - Complete call stack traces from Python to GPU
   - Inline code snippets with source links
   - Demonstrates: 2-level scaling, RHT, stochastic rounding
   - **Use when**: Understanding NVFP4 implementation or debugging FP4 issues

3. **[test_blockwise_fp8_walkthrough.md](test_blockwise_fp8_walkthrough.md)** (48KB+)
   - Line-by-line walkthrough of blockwise FP8 GEMM tests
   - 1D and 2D block quantization explained
   - Mixed quantization strategies (1D×2D, 1D×1D, 2D×1D)
   - GEMM_READY vs COMPACT data formats
   - **Use when**: Understanding blockwise FP8 or implementing custom recipes

---

## Quick Navigation

### By Topic

| Topic | Primary Document | Section |
|-------|------------------|---------|
| **System Overview** | [ARCHITECTURE.md](ARCHITECTURE.md) | [System Overview](ARCHITECTURE.md#system-overview) |
| **MXFP8 Implementation** | [ARCHITECTURE.md](ARCHITECTURE.md) | [MXFP8 Deep Dive](ARCHITECTURE.md#mxfp8-implementation-deep-dive) |
| **NVFP4 Implementation** | [ARCHITECTURE.md](ARCHITECTURE.md) | [NVFP4 Deep Dive](ARCHITECTURE.md#nvfp4-implementation-deep-dive) |
| **Blockwise FP8** | [ARCHITECTURE.md](ARCHITECTURE.md) | [Blockwise FP8](ARCHITECTURE.md#blockwise-fp8-implementation) |
| **Distributed Training** | [ARCHITECTURE.md](ARCHITECTURE.md) | [Distributed Integration](ARCHITECTURE.md#distributed-training-integration) |
| **NVFP4 Testing** | [test_nvfp4_walkthrough.md](test_nvfp4_walkthrough.md) | Full document |
| **Blockwise Testing** | [test_blockwise_fp8_walkthrough.md](test_blockwise_fp8_walkthrough.md) | Full document |
| **Kernel Details** | [ARCHITECTURE.md](ARCHITECTURE.md) | [Kernel Implementation](ARCHITECTURE.md#kernel-implementation-details) |

### By Use Case

| Use Case | Recommended Reading |
|----------|---------------------|
| **New to TransformerEngine** | Start with [ARCHITECTURE.md § Executive Summary](ARCHITECTURE.md#executive-summary) |
| **Debugging quantization issues** | [ARCHITECTURE.md § Quantization Formats](ARCHITECTURE.md#quantization-formats) + relevant walkthrough |
| **Implementing custom recipe** | [ARCHITECTURE.md § Recipe System](ARCHITECTURE.md#architecture-layers) + [Blockwise walkthrough](test_blockwise_fp8_walkthrough.md) |
| **Optimizing kernels** | [ARCHITECTURE.md § Kernel Details](ARCHITECTURE.md#kernel-implementation-details) |
| **Understanding GEMM** | [NVFP4 walkthrough § Step 8](test_nvfp4_walkthrough.md#step-8-native-te-cublas-gemm) or [Blockwise § Step 5](test_blockwise_fp8_walkthrough.md#step-5-native-cublas-gemm) |
| **Distributed training integration** | [ARCHITECTURE.md § Distributed Training](ARCHITECTURE.md#distributed-training-integration) |
| **Adding new quantization format** | Read all three docs, focus on quantizer pattern in [ARCHITECTURE.md § Layer 3](ARCHITECTURE.md#layer-3-quantizer-classes) |

---

## Key Concepts Covered

### Quantization Formats

| Format | Docs Coverage | Key Characteristics |
|--------|---------------|---------------------|
| **MXFP8** | [ARCHITECTURE.md](ARCHITECTURE.md#mxfp8-microscaled-fp8) | 32-element blocks, E8M0 scales, dual layouts |
| **NVFP4** | [ARCHITECTURE.md](ARCHITECTURE.md#nvfp4-4-bit-floating-point) + [Walkthrough](test_nvfp4_walkthrough.md) | 16-element blocks, 2-level scaling (E4M3+FP32), RHT, SR |
| **Blockwise FP8** | [ARCHITECTURE.md](ARCHITECTURE.md#blockwise-fp8) + [Walkthrough](test_blockwise_fp8_walkthrough.md) | Configurable 1D/2D blocks, FP32 scales, GEMM_READY/COMPACT |
| **Standard FP8** | [ARCHITECTURE.md § Delayed Scaling](ARCHITECTURE.md#quantization-formats) | Per-tensor scales, delayed/current strategies |

### System Architecture

**5-Layer Architecture** (detailed in [ARCHITECTURE.md](ARCHITECTURE.md#architecture-layers)):
1. **Python API Layer**: `te.autocast()`, `te.Linear`, etc.
2. **Quantized Tensor Classes**: `MXFP8Tensor`, `NVFP4Tensor`, `Float8BlockwiseQTensor`
3. **Quantizer Classes**: Builder pattern for configuration
4. **C++ Binding Layer**: PyBind11 interface
5. **CUDA Kernel Layer**: Device code for quantization/GEMM

### Call Path Examples

**MXFP8 End-to-End**:
```
User: te.Linear(inp)  [ARCHITECTURE.md § MXFP8 Complete Call Path]
  ↓ Python: mxfp8_tensor.py
  ↓ C++: pybind.cpp → quantize()
  ↓ CUDA: fp8_block_scaling.cu → mxfp8_quantize_kernel()
  ↓ cuBLAS: cublasLtMatmul() with E8M0 scales
  ↓ GPU: Blackwell Tensor Cores
```

**NVFP4 with RHT**:
```
User: quantizer.quantize(grad)  [test_nvfp4_walkthrough.md § Step 4]
  ↓ Python: nvfp4_tensor.py → update_quantized()
  ↓ C++: pybind.cpp → quantize()
  ↓ CUDA: nvfp4.cu → nvfp4_quantize_rht_kernel()
      1. Apply 16×16 Hadamard transform
      2. Compute amax on transformed data
      3. Quantize to FP4 with 2-level scaling
  ↓ Result: NVFP4Tensor with smoothed distribution
```

**Blockwise FP8 (1D×2D)**:
```
User: tex.generic_gemm(w_2d, x_1d)  [test_blockwise_fp8_walkthrough.md § Step 5]
  ↓ Python: tex.generic_gemm()
  ↓ C++: gemm() → gemm_fp8_blockwise()
  ↓ cuBLAS: Mixed 128×128 (weight) and 1×128 (activation) blocks
  ↓ GPU: H100 Tensor Cores with per-block FP32 scales
```

---

## File Links and Line Numbers

All documents use **clickable VSCode-compatible links** with line numbers:

**Format**: `[filename.py:123-456](path/to/filename.py#L123-L456)`

**Examples**:
- [mxfp8_tensor.py:179-449](../transformer_engine/pytorch/tensor/mxfp8_tensor.py#L179-L449)
- [pybind.cpp:119](../transformer_engine/pytorch/csrc/extensions/pybind.cpp#L119)
- [fp8_block_scaling.cu](../transformer_engine/common/recipe/fp8_block_scaling.cu)

**How to use**:
1. Open any `.md` file in VSCode markdown preview (Ctrl+Shift+V / Cmd+Shift+V)
2. Click on links to jump to source code
3. Use "Go Back" (Alt+← / Ctrl+-) to return

---

## Visual Aids

All documents include:
- **Tables**: Quick reference for formats, APIs, parameters
- **Code Snippets**: Inline examples from actual source code
- **Pseudo-code**: Simplified kernel implementations for clarity
- **Call Stack Diagrams**: ASCII art flow diagrams

**Example Table** (from ARCHITECTURE.md):

| Format | Block Size | Scale Type | Hardware | Use Case |
|--------|------------|------------|----------|----------|
| MXFP8 | 32 | E8M0 | Blackwell+ | All tensors E4M3 |
| NVFP4 | 16 (1D), 16×16 (2D) | E4M3+FP32 | Blackwell+ | Extreme compression |
| Blockwise FP8 | 128 (1D), 128×128 (2D) | FP32 | H100+ | Configurable granularity |

---

## Testing Coverage

### NVFP4 Tests Documented

From `tests/pytorch/nvfp4/`:
- [test_nvfp4_gemm_exact.py](../tests/pytorch/nvfp4/test_nvfp4_gemm_exact.py) ✅ Fully documented in [test_nvfp4_walkthrough.md](test_nvfp4_walkthrough.md)
- test_nvfp4_quantize_exact.py
- test_nvfp4_sr_quantize.py (stochastic rounding)
- test_nvfp4_rht_quantize_exact.py (RHT)
- test_nvfp4_module_exact.py

### Blockwise FP8 Tests Documented

From `tests/pytorch/`:
- [test_float8_blockwise_gemm_exact.py](../tests/pytorch/test_float8_blockwise_gemm_exact.py) ✅ Fully documented in [test_blockwise_fp8_walkthrough.md](test_blockwise_fp8_walkthrough.md)
- test_float8_blockwise_scaling_exact.py
- test_float8blockwisetensor.py

---

## Development Workflow

### Adding New Quantization Format

1. **Design**: Read [ARCHITECTURE.md § Quantizer Pattern](ARCHITECTURE.md#layer-3-quantizer-classes)
2. **Implement Quantizer**: Follow `NVFP4Quantizer` as template
3. **Implement Tensor**: Inherit from `QuantizedTensor` and storage class
4. **Add C++ Binding**: Register in `pybind.cpp` (see [init_nvfp4_extensions](../transformer_engine/pytorch/csrc/extensions/pybind.cpp#L91-L104))
5. **Implement Kernels**: See [Kernel Details](ARCHITECTURE.md#kernel-implementation-details)
6. **Test**: Write tests following patterns in walkthroughs

### Debugging Quantization Issues

1. **Verify shapes**: Check [ARCHITECTURE.md § Memory Layouts](ARCHITECTURE.md#mxfp8-memory-layout-details)
2. **Check scales**: Validate scale computation (see walkthroughs § Step 4)
3. **Inspect data**: Use `tensor.dequantize()` to check values
4. **Compare to reference**: Write test like [test_nvfp4_gemm_exact.py](../tests/pytorch/nvfp4/test_nvfp4_gemm_exact.py)
5. **Trace call stack**: Follow call paths in ARCHITECTURE.md

### Optimizing Performance

1. **Profile**: Identify bottleneck (quantization vs GEMM vs communication)
2. **Kernel optimization**: See [ARCHITECTURE.md § Common Kernel Patterns](ARCHITECTURE.md#common-kernel-patterns)
3. **Layout optimization**: Consider COMPACT vs GEMM_READY ([Blockwise § GEMM_READY Format](test_blockwise_fp8_walkthrough.md#gemm_ready-vs-compact-formats))
4. **Recipe tuning**: Experiment with 1D vs 2D blocks ([ARCHITECTURE.md § Recipe System](ARCHITECTURE.md#architecture-layers))

---

## Related Resources

### TransformerEngine Documentation
- [Official Docs](https://docs.nvidia.com/deeplearning/transformer-engine/)
- [FP8 Primer](../docs/examples/fp8_primer.ipynb) - Introductory notebook
- [API Reference](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html)

### Papers and Specs
- **MXFP8**: [OCP Microscaling Formats Spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- **NVFP4**: [Pretraining LLMs with NVFP4](https://arxiv.org/abs/2509.25149v1)
- **Blockwise FP8**: [DeepSeek-v3 Paper](https://arxiv.org/abs/2412.19437v1)
- **FP8 Training**: [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)

### Hardware References
- **Blackwell Architecture**: NVIDIA Blackwell White Paper (search nvidia.com)
- **Hopper Architecture**: NVIDIA H100 Tensor Core GPU Architecture White Paper
- **cuBLASLt**: [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)

---

## Contributing

When updating this documentation:

1. **Maintain link format**: Use VSCode-compatible links with line numbers
2. **Update line numbers**: If source changes, verify line number references
3. **Add inline code**: Include relevant code snippets for new features
4. **Trace call stacks**: Document Python → C++ → CUDA paths
5. **Test links**: Verify all links work in VSCode markdown preview

---

## License

This documentation is part of TransformerEngine and follows the same license. See [../LICENSE](../LICENSE) for details.

---

## Acknowledgments

**Generated by**: Claude (Anthropic)
**Purpose**: Provide comprehensive internal documentation for TransformerEngine developers
**Scope**: Low-precision quantization system (MXFP8, NVFP4, Blockwise FP8)
**Style**: Literate programming with inline code and clickable links

---

## Document Statistics

- **Total Documentation**: ~200KB across 3 files
- **Source Files Referenced**: 50+ files traced
- **Call Stacks Documented**: 10+ complete traces
- **Code Snippets**: 100+ inline examples
- **Clickable Links**: 200+ VSCode-compatible source links

**Last Updated**: 2025
**Base Commit**: bd55e7ba
**Branch**: lowp
