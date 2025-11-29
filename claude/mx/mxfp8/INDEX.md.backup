# MXFP8 Documentation Index

This index provides a complete map of all MXFP8 documentation in the TransformerEngine repository.

---

## Documentation in This Directory (claude/mx/mxfp8/)

### Core Documentation Files

1. **[00_START_HERE.md](00_START_HERE.md)** - Start here!
   - Navigation guide
   - Key findings summary
   - Reading recommendations
   - Quick reference to all topics

2. **[README.md](README.md)** - Comprehensive overview
   - Quick start guide
   - Architecture overview
   - Complete call flow diagram
   - E8M0 scale format explanation
   - Comparison with other recipes
   - Performance characteristics
   - Troubleshooting and FAQ

---

## Detailed Frame-by-Frame Documentation (claude/mx_tests/tests/)

These documents provide detailed execution traces with line numbers and source code references:

### 1. MXFP8 Quantization Implementation

**[06_mxfp8_quantization.md](../../mx_tests/tests/06_mxfp8_quantization.md)**

Complete frame-by-frame trace of quantization pipeline:

- **Frame 1:** Python Entry Point (mxfp8_tensor.py:27-175)
  - MXFP8Quantizer initialization
  - Configuration parameters

- **Frame 2:** Memory Allocation (mxfp8_tensor.py:85-138)
  - MXFP8Tensor storage allocation
  - Rowwise and columnwise data/scales
  - Memory layout and padding

- **Frame 3:** Quantization Invocation (mxfp8_tensor.py:47-73)
  - update_quantized() method
  - tex.quantize() C++ binding call

- **Frame 4:** C++ Binding Layer (pybind.cpp:120)
  - PyBind11 quantize() binding
  - Type conversion and validation

- **Frame 5:** C++ Implementation (quantizer.cpp:1091-1103)
  - MXFP8Quantizer::quantize() method
  - Simple config setup (no complex features)
  - nvte_quantize_v2() dispatch

- **Frame 6:** CUDA Kernel Execution (quantize_mxfp8.cuh:43-538)
  - Block-wise amax computation (32 elements)
  - E8M0 scale generation
  - FP8 E4M3 quantization

**Topics Covered:**
- E8M0 scale format encoding/decoding
- Block size rationale (32 vs 16)
- Memory bandwidth analysis
- Quantization formula
- MXFP8 vs NVFP4 complexity comparison

---

### 2. MXFP8 Numerics and Module Integration

**[07_mxfp8_numerics.md](../../mx_tests/tests/07_mxfp8_numerics.md)**

End-to-end training tests with MXFP8:

- **Frame 1:** Recipe Initialization (test_numerics.py:155-157)
  - MXFP8BlockScaling instantiation
  - Recipe methods (mxfp8(), block_scaling())

- **Frame 2:** Test Setup with Autocast (test_numerics.py:1857-1868)
  - quantized_model_init context
  - Module creation with MXFP8 weights
  - Memory impact analysis

- **Frame 3:** Module Forward Pass (module/linear.py)
  - Linear.forward() with MXFP8
  - Quantizer creation
  - GEMM execution

- **Frame 4:** Quantizer Creation and Usage (test_numerics.py:2727-2756)
  - Direct MXFP8Quantizer usage
  - general_gemm API integration
  - MXFP8Tensor properties

- **Frame 5:** Backward Pass (test_numerics.py:1800-1818)
  - Gradient quantization flow
  - Weight gradient accumulation
  - Accuracy validation

**Topics Covered:**
- Module integration patterns
- Alignment requirements (32-element blocks)
- Forward + backward pass data flow
- Tolerance analysis (~1-2% relative error)
- Test patterns and usage examples
- Padding for alignment

---

### 3. MXFP8 Recipe Configuration and Management

**[08_mxfp8_recipe.md](../../mx_tests/tests/08_mxfp8_recipe.md)**

Recipe tests and configuration:

- **Frame 1:** Recipe Initialization (test_recipe.py:389-391)
  - MXFP8BlockScaling instantiation
  - Default configuration
  - Recipe methods and type checking

- **Frame 2:** Weight Tensor and Recipe Correspondence (test_recipe.py:401-409)
  - quantized_model_init context
  - MXFP8Tensor weight creation
  - Recipe mismatch detection

- **Frame 3:** Dynamic Recipe Switching (test_recipe.py:430-477)
  - DelayedScaling → MXFP8BlockScaling
  - Quantizer replacement
  - Warning messages

- **Frame 4:** Quantizer Type Validation
  - Quantizer inspection
  - Properties comparison (Float8Quantizer vs MXFP8Quantizer)

**Topics Covered:**
- Recipe state management
- Recipe lifecycle
- State persistence (none for MXFP8)
- MXFP8 vs DelayedScaling comparison
- Performance characteristics
- Implementation notes

---

## Test File Overview

**[00_overview.md](../../mx_tests/tests/00_overview.md)**

Summary of all test files covering MXFP8 and related features.

**[SUMMARY.md](../../mx_tests/tests/SUMMARY.md)**

High-level summary of findings across all tests.

---

## Source Code Reference

### Python Implementation

| Component | File | Lines |
|-----------|------|-------|
| **MXFP8Quantizer** | transformer_engine/pytorch/tensor/mxfp8_tensor.py | 27-175 |
| **MXFP8Tensor** | transformer_engine/pytorch/tensor/mxfp8_tensor.py | 177-943 |
| **MXFP8TensorStorage** | transformer_engine/pytorch/tensor/storage/mxfp8_tensor_storage.py | 50-257 |
| **MXFP8BlockScaling Recipe** | transformer_engine/common/recipe/__init__.py | 265-303 |
| **MXFP8BlockScalingRecipeState** | transformer_engine/pytorch/quantization.py | 1130-1162 |
| **autocast() Context** | transformer_engine/pytorch/quantization.py | 790-852 |
| **RecipeState Factory** | transformer_engine/pytorch/quantization.py | 967-1026 |
| **FP8GlobalStateManager** | transformer_engine/pytorch/quantization.py | 224-677 |

### C++ Implementation

| Component | File | Lines |
|-----------|------|-------|
| **quantize() Binding** | transformer_engine/pytorch/csrc/extensions/cast.cpp | 33-79, 347-492 |
| **MXFP8Quantizer C++** | transformer_engine/pytorch/csrc/common.h | 265-284 |
| **nvte_quantize API** | transformer_engine/common/include/transformer_engine/cast.h | 82-90 |

### CUDA Implementation

| Component | File | Description |
|-----------|------|-------------|
| **MXFP8 Quantization Kernel** | transformer_engine/common/cast/mxfp8/quantize_mxfp8.cuh | 32-element block quantization, E8M0 scales |
| **CUDA Tests** | tests/cpp/operator/test_cast_mxfp8.cu | Kernel validation tests |

### Test Files

| Test File | Focus | Lines |
|-----------|-------|-------|
| **test_sanity.py** | MXFP8 inference tests | 1091-1131 |
| **test_numerics.py** | Training accuracy tests | Multiple |
| **test_recipe.py** | Recipe configuration | 401-477 |

---

## Reading Paths

### Path 1: Quick Start (30 minutes)

1. [00_START_HERE.md](00_START_HERE.md) - Overview (5 min)
2. [README.md](README.md) - Quick start section (10 min)
3. [README.md](README.md) - E8M0 format section (10 min)
4. [README.md](README.md) - Comparison table (5 min)

**Result:** Understand what MXFP8 is, how to use it, and when to choose it.

---

### Path 2: Implementation Deep Dive (2 hours)

1. [00_START_HERE.md](00_START_HERE.md) - Full read (15 min)
2. [README.md](README.md) - Architecture and call flow (30 min)
3. [06_mxfp8_quantization.md](../../mx_tests/tests/06_mxfp8_quantization.md) - Frames 1-6 (45 min)
4. [07_mxfp8_numerics.md](../../mx_tests/tests/07_mxfp8_numerics.md) - Frames 1-5 (30 min)

**Result:** Understand complete implementation from Python API to CUDA kernels.

---

### Path 3: Testing and Integration (1.5 hours)

1. [README.md](README.md) - Test coverage section (10 min)
2. [07_mxfp8_numerics.md](../../mx_tests/tests/07_mxfp8_numerics.md) - Module integration (40 min)
3. [08_mxfp8_recipe.md](../../mx_tests/tests/08_mxfp8_recipe.md) - Recipe management (40 min)

**Result:** Understand how MXFP8 integrates with TE modules and how to test it.

---

### Path 4: Complete Mastery (4 hours)

Follow Path 2, then:

1. [08_mxfp8_recipe.md](../../mx_tests/tests/08_mxfp8_recipe.md) - Full read (1 hour)
2. [README.md](README.md) - Troubleshooting and FAQ (30 min)
3. Review source code files listed above (1.5 hours)

**Result:** Complete understanding of MXFP8 implementation, ready to extend or debug.

---

## Topic Index

### Quantization Mechanics
- **E8M0 Scale Format:** [README.md](README.md#e8m0-scale-format), [06_mxfp8_quantization.md](../../mx_tests/tests/06_mxfp8_quantization.md#e8m0-scale-format)
- **Block Size (32 elements):** [README.md](README.md#memory-layout), [06_mxfp8_quantization.md](../../mx_tests/tests/06_mxfp8_quantization.md#block-size-32-vs-16)
- **Quantization Pipeline:** [README.md](README.md#quantization-pipeline), [06_mxfp8_quantization.md](../../mx_tests/tests/06_mxfp8_quantization.md#frame-by-frame-execution-trace)

### Architecture
- **Recipe Definition:** [README.md](README.md#recipe-configuration), [00_START_HERE.md](00_START_HERE.md#2-mxfp8blockscaling-recipe)
- **Recipe State:** [08_mxfp8_recipe.md](../../mx_tests/tests/08_mxfp8_recipe.md#recipe-state-management)
- **Quantizer Class:** [README.md](README.md#1-mxfp8quantizer), [06_mxfp8_quantization.md](../../mx_tests/tests/06_mxfp8_quantization.md#frame-1-python-entry-point)
- **Tensor Storage:** [README.md](README.md#2-mxfp8tensor), [06_mxfp8_quantization.md](../../mx_tests/tests/06_mxfp8_quantization.md#frame-2-memory-allocation)

### Integration
- **autocast() Context:** [README.md](README.md#call-flow-user-api-to-kernel), [00_START_HERE.md](00_START_HERE.md#4-call-flow)
- **Module Integration:** [07_mxfp8_numerics.md](../../mx_tests/tests/07_mxfp8_numerics.md#frame-3-module-forward-pass)
- **GEMM Operations:** [07_mxfp8_numerics.md](../../mx_tests/tests/07_mxfp8_numerics.md#frame-3b-groupedlinear-with-mxfp8)

### Testing
- **Test Overview:** [README.md](README.md#test-coverage)
- **Inference Tests:** [00_overview.md](../../mx_tests/tests/00_overview.md)
- **Training Tests:** [07_mxfp8_numerics.md](../../mx_tests/tests/07_mxfp8_numerics.md)
- **Recipe Tests:** [08_mxfp8_recipe.md](../../mx_tests/tests/08_mxfp8_recipe.md)

### Comparisons
- **vs NVFP4:** [README.md](README.md#mxfp8-vs-other-recipes), [06_mxfp8_quantization.md](../../mx_tests/tests/06_mxfp8_quantization.md#mxfp8-vs-nvfp4-summary)
- **vs DelayedScaling:** [08_mxfp8_recipe.md](../../mx_tests/tests/08_mxfp8_recipe.md#mxfp8-vs-delayedscaling)
- **vs Float8BlockScaling:** [README.md](README.md#comparison-table)

### Performance
- **Memory Layout:** [README.md](README.md#memory-layout)
- **Bandwidth Analysis:** [README.md](README.md#memory-bandwidth), [06_mxfp8_quantization.md](../../mx_tests/tests/06_mxfp8_quantization.md#memory-bandwidth)
- **Accuracy Characteristics:** [README.md](README.md#accuracy), [07_mxfp8_numerics.md](../../mx_tests/tests/07_mxfp8_numerics.md#frame-5c-accuracy-validation)

### Practical Usage
- **Quick Start:** [README.md](README.md#quick-start)
- **Troubleshooting:** [README.md](README.md#troubleshooting)
- **FAQ:** [README.md](README.md#faq)
- **Device Requirements:** [README.md](README.md#device-requirements)

---

## Document History

- **2025-01-12:** Created comprehensive MXFP8 documentation
  - 00_START_HERE.md: Navigation and overview
  - README.md: Complete reference guide
  - INDEX.md: Documentation map
  - Links to existing detailed docs in claude/mx_tests/tests/

---

## Related Documentation

### NVFP4 Documentation (for comparison)
- [claude/NVFP4_TEST_WALKTHROUGH.md](../../NVFP4_TEST_WALKTHROUGH.md)
- [claude/TE_AUTOCAST_ANALYSIS.md](../../TE_AUTOCAST_ANALYSIS.md)

### TransformerEngine General
- [claude/README.md](../../README.md)
- [claude/ARCHITECTURE.md](../../ARCHITECTURE.md)

---

## Contributing

When adding new MXFP8 documentation:

1. **Update this index** with new file locations and descriptions
2. **Cross-reference** related documents
3. **Include line numbers** for source code references
4. **Follow the existing format:** Frame-by-frame traces with clear headings
5. **Test all code examples** before documenting

---

## Quick Reference Card

### MXFP8 Key Facts

```
Block Size:        32 elements (fixed)
Scale Format:      E8M0 (8-bit exponent, power-of-2)
Data Format:       FP8 E4M3 (default)
Compression:       3.88× vs FP32
Precision:         8 bits per element
Hardware:          Blackwell (CC 10.0+)
Features:          Simple (no RHT, SR, 2D)
State:             Stateless (no amax history)
Accuracy:          ~1-2% relative error vs FP32
Memory Overhead:   ~3% for scales
```

### Usage Pattern

```python
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

# Create recipe
mxfp8_recipe = recipe.MXFP8BlockScaling()

# Use with autocast
with te.autocast(enabled=True, recipe=mxfp8_recipe):
    output = model(input)
```

### File Lookup

```
Recipe:      transformer_engine/common/recipe/__init__.py:265-303
Quantizer:   transformer_engine/pytorch/tensor/mxfp8_tensor.py:27-175
Tensor:      transformer_engine/pytorch/tensor/mxfp8_tensor.py:177-943
autocast():  transformer_engine/pytorch/quantization.py:790-852
Tests:       tests/pytorch/test_sanity.py:1091-1131
```

---

**Navigate from here:** [Start Reading →](00_START_HERE.md)
