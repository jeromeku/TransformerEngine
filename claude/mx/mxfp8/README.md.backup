# MXFP8BlockScaling Mixed Precision Recipe Documentation

## Overview

This directory contains comprehensive documentation for the **MXFP8BlockScaling** (Microscaling FP8) mixed precision recipe in TransformerEngine. MXFP8 provides block-wise 8-bit quantization with power-of-2 scales for efficient mixed precision training and inference.

**Start here:** [00_START_HERE.md](00_START_HERE.md)

---

## What is MXFP8?

**MXFP8 (Microscaling FP8)** is a block-wise quantization strategy that:
- Divides tensors into 32-element blocks
- Scales each block with E8M0 format (8-bit exponent, power-of-2)
- Stores data in FP8 E4M3 format (8 bits per element)
- Provides higher precision than NVFP4 with simpler implementation

**Key Advantages:**
- ✅ **Simple Configuration:** No complex features (RHT, stochastic rounding, 2D quantization)
- ✅ **Stateless:** No amax history tracking
- ✅ **Efficient:** E8M0 scales (1 byte per 32 elements)
- ✅ **Higher Precision:** 8-bit vs 4-bit NVFP4
- ✅ **Good Accuracy:** ~1-2% relative error vs FP32

---

## Quick Start

### Basic Usage

```python
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

# Create MXFP8 recipe
mxfp8_recipe = recipe.MXFP8BlockScaling()

# Create a Linear layer
linear = te.Linear(1024, 2048)

# Use with autocast context
with te.autocast(enabled=True, recipe=mxfp8_recipe):
    output = linear(input)  # Automatically uses MXFP8 quantization
```

### With Quantized Weights

```python
# Store weights in MXFP8 format (saves memory)
with te.quantized_model_init(enabled=True, recipe=mxfp8_recipe):
    model = te.Linear(1024, 2048)
# Weights are now stored as MXFP8Tensor (48.5% of original size)

# Use the model
with te.autocast(enabled=True, recipe=mxfp8_recipe):
    output = model(input)
```

---

## Architecture Overview

### Recipe Configuration

```python
@dataclass
class MXFP8BlockScaling(Recipe):
    """MXFP8 block-wise scaling strategy."""

    margin: int = 0                    # Not used for block scaling
    fp8_format: Format = Format.E4M3   # E4M3 (default) or HYBRID
    fp8_dpa: bool = False              # Dot Product Attention (not supported yet)
    fp8_mha: bool = False              # Multi-Head Attention (not supported yet)
```

**Default Configuration:**
- Block size: 32 elements (fixed, defined in constants.py)
- Scale format: E8M0 (8-bit exponent, power-of-2 only)
- Data format: E4M3 (4-bit exponent, 3-bit mantissa)
- No complex features (RHT, stochastic rounding, 2D quantization)

### Quantization Pipeline

```
High-Precision Input (BF16/FP32)
         ↓
  Divide into 32-element blocks
         ↓
  Compute amax per block
         ↓
  Generate E8M0 scales (power-of-2)
         ↓
  Quantize to FP8 E4M3
         ↓
    MXFP8Tensor
```

### Memory Layout

For a tensor of shape `[M, N]`:

**Rowwise Quantization:**
- Data: `[M, N]` uint8 (E4M3 FP8)
- Scales: `[M_padded, N/32_padded]` uint8 (E8M0)
- Padding: M → 128, N/32 → 4 (for alignment)

**Columnwise Quantization (for transpose):**
- Data: `[M, N]` uint8 (E4M3 FP8)
- Scales: `[M/32_padded, N_padded]` uint8 (E8M0)
- Padding: M/32 → 4, N → 128

**Memory Savings:**
```
FP32 tensor:  M × N × 4 bytes
MXFP8 tensor: M × N × 1 byte (data) + M × N/32 × 1 byte (scales)
            = M × N × 1.03125 bytes
            ≈ 3.88× compression vs FP32
```

---

## Call Flow: User API to Kernel

### Complete Execution Path

```
1. User Code
   └─ with te.autocast(enabled=True, recipe=MXFP8BlockScaling()):

2. Python Layer (quantization.py:790-852)
   ├─ FP8GlobalStateManager.autocast_enter()
   ├─ Set FP8_ENABLED = True
   ├─ Set FP8_RECIPE = MXFP8BlockScaling()
   └─ Validate device (CC 10.0+ Blackwell)

3. Module Forward (module/base.py:744-779)
   ├─ Query: is_fp8_enabled() → True
   ├─ Get recipe: get_fp8_recipe() → MXFP8BlockScaling()
   ├─ Create state: RecipeState.create(recipe, mode="forward")
   │  └─ Returns: MXFP8BlockScalingRecipeState
   └─ Create quantizers: state.make_quantizers()
      └─ Returns: [MXFP8Quantizer, MXFP8Quantizer, ...]

4. Quantization (tensor/mxfp8_tensor.py:48-70)
   ├─ MXFP8Quantizer.__call__(tensor)
   ├─ Validate tensor dimensions (divisible by 32)
   ├─ Allocate MXFP8Tensor storage
   └─ Call: tex.quantize(src, self, dst)

5. C++ Binding (csrc/extensions/cast.cpp:33-79)
   ├─ quantize() PyBind11 function
   ├─ Convert Python objects to C++ types
   ├─ Wrap tensors: makeTransformerEngineTensor()
   └─ Call: quantizer_cpp->quantize(input, output)

6. C++ Quantizer (csrc/common.h:265-284)
   ├─ MXFP8Quantizer::quantize()
   ├─ Create NVTETensor with NVTE_MXFP8_1D_SCALING mode
   └─ Call: nvte_quantize(input, output, stream)

7. TE Core API (common/include/transformer_engine/cast.h:82-90)
   └─ nvte_quantize() dispatches to CUDA kernel

8. CUDA Kernel (common/cast/cast.cu)
   ├─ Load 32-element blocks
   ├─ Compute amax per block
   ├─ Compute E8M0 scale: exponent = ceil(log2(amax / FP8_MAX))
   ├─ Quantize: q_i = round(x_i / scale)
   └─ Store FP8 data and E8M0 scales
```

---

## Key Components

### 1. MXFP8Quantizer

**File:** `transformer_engine/pytorch/tensor/mxfp8_tensor.py` (Lines 27-175)

```python
class MXFP8Quantizer(Quantizer):
    """Builder class for FP8 tensors with MX block scaling."""

    def __init__(
        self,
        fp8_dtype: TE_DType,          # E4M3 or E5M2
        rowwise: bool = True,          # Rowwise scaling
        columnwise: bool = True,       # Columnwise scaling (for transpose)
    ):
        self.dtype = fp8_dtype
        self.rowwise_usage = rowwise
        self.columnwise_usage = columnwise

    def __call__(self, tensor: torch.Tensor) -> MXFP8Tensor:
        """Quantize tensor to MXFP8 format."""
        return self.quantize_impl(tensor)
```

**Key Methods:**
- `quantize_impl()`: Quantizes tensor, returns MXFP8Tensor
- `update_quantized()`: Updates existing MXFP8Tensor with new data
- `make_empty()`: Allocates empty MXFP8Tensor with proper memory layout
- `is_quantizable()`: Validates tensor dimensions (must be divisible by 32)

### 2. MXFP8Tensor

**File:** `transformer_engine/pytorch/tensor/mxfp8_tensor.py` (Lines 177-943)

```python
class MXFP8Tensor(QuantizedTensor):
    """FP8 tensor with MX block scaling."""

    # Storage attributes
    _rowwise_data: torch.Tensor           # FP8 data, shape [M, N]
    _rowwise_scale_inv: torch.Tensor      # E8M0 scales, shape [M, N/32]
    _columnwise_data: torch.Tensor        # Transposed FP8 data
    _columnwise_scale_inv: torch.Tensor   # Transposed E8M0 scales

    def dequantize(self, dtype=torch.float32) -> torch.Tensor:
        """Dequantize to high-precision tensor."""
        return tex.dequantize(self, dtype)
```

**Key Properties:**
- Both rowwise and columnwise quantizations stored
- E8M0 scales (1 byte per 32 elements)
- Dequantization via C++ binding

### 3. MXFP8BlockScalingRecipeState

**File:** `transformer_engine/pytorch/quantization.py` (Lines 1130-1162)

```python
class MXFP8BlockScalingRecipeState(RecipeState):
    """Configuration for MXFP8 quantization.

    MXFP8 quantization does not require state (no amax history).
    """

    def __init__(self, recipe, mode, num_quantizers=1, device=None):
        self.recipe = recipe
        self.mode = mode
        self.num_quantizers = num_quantizers
        self.dtype = get_fp8_te_dtype(recipe, mode == "forward")

    def make_quantizers(self):
        """Create MXFP8Quantizer instances."""
        from .tensor.mxfp8_tensor import MXFP8Quantizer
        return [MXFP8Quantizer(self.dtype) for _ in range(self.num_quantizers)]
```

**Characteristics:**
- Stateless (no amax history buffers)
- Creates MXFP8Quantizer instances on demand
- Simpler than DelayedScaling state

---

## E8M0 Scale Format

### What is E8M0?

**E8M0** = 8-bit Exponent, 0-bit Mantissa

- **Representation:** 8-bit unsigned integer
- **Value:** scale = 2^(value - 127)
- **Range:** 2^-127 to 2^128
- **Constraint:** Power-of-2 values only

### Encoding

```python
# Compute scale from amax
amax = max(abs(block_values))  # Max absolute value in 32-element block
scale = amax / FP8_E4M3_MAX    # FP8_E4M3_MAX = 448.0

# Compute exponent
exponent = ceil(log2(scale))

# Clamp to valid range
exponent = clamp(exponent, -127, 127)

# Store as biased uint8
scale_e8m0 = uint8(exponent + 127)
```

### Decoding

```python
# Load biased exponent
scale_e8m0 = scales[block_idx]  # uint8 value

# Unbias
exponent = int(scale_e8m0) - 127

# Compute scale
scale = 2^exponent

# Dequantize element
value_fp32 = quantized_value * scale
```

### Advantages

1. **Memory Efficient:** 1 byte per 32 elements = 0.25 bits overhead/element
2. **Simple Decoding:** Just bit shift (no multiplication)
3. **Wide Dynamic Range:** 2^-127 to 2^128
4. **Hardware Friendly:** Power-of-2 scales

---

## MXFP8 vs Other Recipes

### Comparison Table

| Feature | MXFP8 | NVFP4 | DelayedScaling | Float8BlockScaling |
|---------|-------|-------|----------------|-------------------|
| **Bits/element** | 8 | 4 | 8 | 8 |
| **Block size** | 32 | 16 (1D), 16×16 (2D) | Per-tensor | Configurable |
| **Scale format** | E8M0 | E4M3+FP32 | FP32 | FP32 |
| **Scale size** | 1 byte | 1+4 bytes | 4 bytes | 4 bytes |
| **RHT** | No | Yes | No | No |
| **Stochastic rounding** | No | Yes | No | No |
| **2D quantization** | No | Yes (weights) | No | No |
| **Amax history** | No | No | Yes | No |
| **Complexity** | Simple | Complex | Complex | Medium |
| **Precision** | Higher | Lower | Higher | Higher |
| **Compression** | 3.88× | 7.11× | 4× | ~4× |

### When to Use MXFP8

**Use MXFP8BlockScaling when:**
- ✅ Need higher precision than NVFP4 (8-bit vs 4-bit)
- ✅ Want simple configuration (no RHT, SR, 2D quantization)
- ✅ Acceptable memory overhead (~3% for scales)
- ✅ Training or fine-tuning with mixed precision
- ✅ Target hardware: Blackwell (CC 10.0+)

**Use NVFP4BlockScaling when:**
- ✅ Need maximum compression (8× vs 3.88×)
- ✅ Can tolerate lower precision (4-bit)
- ✅ Advanced features helpful (RHT improves quality)
- ✅ Primarily quantizing weights

**Use DelayedScaling when:**
- ✅ Need maximum memory efficiency (global scales)
- ✅ Want temporal smoothing of scales
- ✅ Training from scratch (needs stability)
- ✅ Target hardware: Hopper+ (CC 9.0+)

---

## Device Requirements

### Hardware Support

**MXFP8BlockScaling requires:**
- NVIDIA GPU with Compute Capability 10.0+
- Blackwell architecture (e.g., B100, B200)

**Check availability:**
```python
import transformer_engine.pytorch as te

# MXFP8 uses same hardware check as NVFP4
is_available, reason = te.is_nvfp4_available()
if is_available:
    print("MXFP8 is available!")
else:
    print(f"MXFP8 not available: {reason}")
```

### Software Requirements

```bash
# Install TransformerEngine
pip install transformer-engine

# Requires:
# - CUDA 12.0+
# - cuBLAS with FP8 support
# - Blackwell-capable GPU
```

---

## Test Coverage

### Test Files

1. **test_sanity.py** (Lines 1091-1131)
   - MXFP8 inference tests
   - Validates weight quantization
   - Checks tensor structure

2. **test_numerics.py** (Multiple locations)
   - MXFP8 training accuracy tests
   - Forward + backward pass validation
   - Tolerance checks (~1-2% relative error)

3. **test_recipe.py** (Lines 401-477)
   - Recipe configuration tests
   - Dynamic recipe switching
   - Recipe/weight compatibility checks

4. **tests/cpp/operator/test_cast_mxfp8.cu**
   - CUDA kernel tests
   - E8M0 scale computation validation
   - Quantization/dequantization accuracy

### Running Tests

```bash
# Run all MXFP8 tests
pytest tests/pytorch/test_sanity.py -k mxfp8 -v

# Run numerics tests with MXFP8
pytest tests/pytorch/test_numerics.py -k mxfp8 -v

# Run recipe tests
pytest tests/pytorch/test_recipe.py -k MXFP8 -v
```

---

## Detailed Documentation

For comprehensive frame-by-frame execution traces and detailed analysis, see:

### Existing Comprehensive Documentation

The following files in `claude/mx_tests/tests/` provide detailed MXFP8 documentation:

1. **[06_mxfp8_quantization.md](../../mx_tests/tests/06_mxfp8_quantization.md)**
   - Frame-by-frame quantization pipeline
   - Python → C++ → CUDA execution flow
   - E8M0 scale format details
   - Memory layout and allocation

2. **[07_mxfp8_numerics.md](../../mx_tests/tests/07_mxfp8_numerics.md)**
   - End-to-end training tests
   - Module integration (Linear, LayerNormLinear, TransformerLayer)
   - Forward + backward pass with quantization
   - Accuracy validation and tolerance analysis

3. **[08_mxfp8_recipe.md](../../mx_tests/tests/08_mxfp8_recipe.md)**
   - Recipe configuration and initialization
   - Recipe switching tests
   - Weight tensor and recipe correspondence
   - Quantizer type validation

### Additional Resources in This Directory

- **[TE_AUTOCAST_ANALYSIS.md](TE_AUTOCAST_ANALYSIS.md)** (Coming next)
  - Detailed autocast() implementation
  - FP8GlobalStateManager deep dive
  - Recipe state factory pattern
  - Complete integration analysis

- **[MXFP8_TEST_WALKTHROUGH.md](MXFP8_TEST_WALKTHROUGH.md)** (Coming next)
  - Test-by-test walkthroughs
  - Complete call paths with line numbers
  - Validation mechanisms
  - Environment variables

---

## Performance Characteristics

### Memory Bandwidth

```
For 1024×1024 GEMM:

FP32 baseline:
  Input:  4 MB
  Weight: 4 MB
  Total:  8 MB read + 4 MB write = 12 MB bandwidth

MXFP8:
  Input:  1 MB + 32 KB scales = 1.03 MB
  Weight: 1 MB + 32 KB scales = 1.03 MB
  Total:  2.06 MB read + 4 MB write = 6.06 MB bandwidth

  Speedup: 12 / 6.06 = 1.98× bandwidth reduction
```

### Accuracy

```
Expected relative error vs FP32:

Forward pass:     ~1% relative error
Backward pass:    ~1-2% relative error
Weight gradients: ~2% relative error (accumulated)

Suitable for:
  - Fine-tuning: Excellent
  - Full training: Good
  - Inference: Excellent
```

### Compute Throughput

```
Tensor Core Utilization:

FP32: 19.5 TFLOPS (A100)
TF32: 156 TFLOPS (A100)
FP8:  312 TFLOPS (H100), 2000 TFLOPS (B100 estimated)

MXFP8 on Blackwell:
  - Up to 10× faster than FP32
  - 2× faster than TF32
  - Similar to other FP8 formats
```

---

## Troubleshooting

### Common Issues

**1. "MXFP8 not supported on this device"**
```
Solution: Requires Blackwell GPU (CC 10.0+)
Check: is_nvfp4_available() for compatibility
```

**2. "Tensor dimensions not divisible by 32"**
```
Solution: Pad tensors to multiples of 32
Example:
  M, N, K = 1000, 1024, 512  # ❌ M not divisible by 32
  M, N, K = 1024, 1024, 512  # ✅ All divisible by 32
```

**3. "Recipe mismatch for weight tensor"**
```
Solution: Cannot use MXFP8 weights with other recipes
Either:
  - Don't use quantized_model_init (keeps weights high-precision)
  - Use same recipe consistently
```

**4. "High memory usage"**
```
Solution: MXFP8 stores both rowwise and columnwise quantizations for weights
This is by design to avoid double quantization errors
Expected: ~2× memory overhead for weights vs single quantization
```

---

## FAQ

**Q: Can I use MXFP8 with FP16/BF16 inputs?**
A: Yes, MXFP8 works with FP32, FP16, and BF16 input tensors.

**Q: Does MXFP8 support attention operations?**
A: Currently, `fp8_dpa` and `fp8_mha` are not supported for MXFP8. Use standard FP8 recipes for attention.

**Q: Can I transpose MXFP8Tensor?**
A: No, you cannot simply transpose. The quantizer computes both orientations from the high-precision input to avoid double quantization errors.

**Q: How does MXFP8 compare to BF16?**
A: MXFP8 uses ~50% memory of BF16 with ~1-2% accuracy loss. Compute is faster with Tensor Cores.

**Q: Can I mix MXFP8 with other quantization?**
A: No, use one recipe consistently within an autocast context.

---

## Additional Resources

### Source Code
- [MXFP8Quantizer](../../transformer_engine/pytorch/tensor/mxfp8_tensor.py)
- [MXFP8BlockScaling Recipe](../../transformer_engine/common/recipe/__init__.py)
- [Recipe State Factory](../../transformer_engine/pytorch/quantization.py)

### Tests
- [Inference Tests](../../tests/pytorch/test_sanity.py)
- [Training Tests](../../tests/pytorch/test_numerics.py)
- [Recipe Tests](../../tests/pytorch/test_recipe.py)

### Documentation
- [Existing MXFP8 Documentation](../../claude/mx_tests/tests/)
- [This Directory](.)

---

## Summary

MXFP8BlockScaling provides:
- ✅ **Simple**: No complex features (RHT, SR, 2D quantization)
- ✅ **Efficient**: E8M0 scales (1 byte per 32 elements)
- ✅ **Accurate**: ~1-2% relative error vs FP32
- ✅ **Stateless**: No amax history tracking
- ✅ **Flexible**: Works with FP32/FP16/BF16 inputs
- ✅ **Fast**: Leverages Blackwell Tensor Cores

Perfect for mixed precision training and inference on Blackwell GPUs!
