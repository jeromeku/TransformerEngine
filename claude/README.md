# TransformerEngine te.autocast() Analysis - Complete Documentation

This folder contains comprehensive documentation analyzing the `te.autocast()` implementation in TransformerEngine, with detailed focus on NVFP4BlockScaling recipe integration.

## Contents

### 1. TE_AUTOCAST_ANALYSIS.md
**Comprehensive architecture and implementation analysis**

Primary document covering:
- Architecture overview with data flow diagrams
- Core components and file structure
- Complete autocast() context manager implementation
- NVFP4BlockScaling recipe definition and integration
- Global state management via FP8GlobalStateManager
- Complete call flow from user API to kernel execution
- Integration with te.Linear module
- Support checks and device capability validation
- Key design patterns used

**Key Sections:**
- `autocast()` context manager (lines 789-852 in quantization.py)
- FP8GlobalStateManager state management (lines 224-252)
- NVFP4BlockScalingRecipeState factory (lines 1270-1343)
- Recipe support validation and device checks

**Best For:** Understanding the overall architecture, design patterns, and how different components interact.

---

### 2. NVFP4_TEST_WALKTHROUGH.md
**Detailed test case walkthroughs with data flow traces**

Test-focused documentation containing:
- Test file organization and structure
- Basic module test walkthrough with complete call paths
- RHT (Random Hadamard Transform) quantization data flow
- 2D quantization for weights explanation
- GEMM kernel execution path and computational flow
- Test validation mechanisms and tolerance rationale
- Environment variable controls
- Forward and backward pass specifications

**Test Files Covered:**
- `test_nvfp4_module_exact.py` - Main module test
- `test_nvfp4_gemm_exact.py` - Direct GEMM tests
- `test_nvfp4_rht_quantize_exact.py` - RHT-specific tests
- `test_nvfp4_quantize_exact.py` - Quantization kernel tests
- `test_nvfp4_sr_quantize.py` - Stochastic rounding tests

**Best For:** Understanding test execution, data flow during testing, and tracing specific features like RHT and 2D quantization.

---

## Key Files Referenced

### Core Implementation Files

1. **transformer_engine/pytorch/quantization.py** (1397 lines)
   - `autocast()` context manager (lines 789-852)
   - `FP8GlobalStateManager` class (lines 224-677)
   - `RecipeState` factory (lines 967-1026)
   - `NVFP4BlockScalingRecipeState` class (lines 1270-1343)

2. **transformer_engine/common/recipe/__init__.py** (515 lines)
   - `NVFP4BlockScaling` recipe definition (lines 386-481)
   - `Format` enum (lines 23-44)
   - `QParams` dataclass (lines 62-83)

3. **transformer_engine/pytorch/module/linear.py**
   - `_Linear` autograd.Function
   - Forward/backward pass integration with quantization

### Test Files

Located in: `/home/jeromeku/transformerengine/tests/pytorch/nvfp4/`

- `test_nvfp4_module_exact.py` (20,469 bytes)
- `test_nvfp4_gemm_exact.py` (7,745 bytes)
- `test_nvfp4_quantize_exact.py` (16,307 bytes)
- `test_nvfp4_rht_quantize_exact.py` (8,033 bytes)
- `test_nvfp4_sr_quantize.py` (7,968 bytes)

---

## Quick Reference: Autocast Usage Pattern

### Basic Usage

```python
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import NVFP4BlockScaling

# Create recipe
recipe = NVFP4BlockScaling()

# Use in autocast context
with te.autocast(enabled=True, recipe=recipe):
    output = model(input)
    loss = criterion(output, target)
    loss.backward()
```

### Data Flow Summary

```
User Code
├─ te.autocast(enabled=True, recipe=NVFP4BlockScaling())
│  └─ FP8GlobalStateManager.autocast_enter()
│
├─ model(input)
│  └─ te.Linear.forward()
│     ├─ Query FP8GlobalStateManager.is_fp8_enabled() → True
│     ├─ recipe = FP8GlobalStateManager.get_fp8_recipe()
│     ├─ RecipeState.create(recipe, mode="forward")
│     ├─ NVFP4BlockScalingRecipeState.make_quantizers()
│     └─ _Linear.forward() with quantization
│
└─ autocast_exit()
   └─ FP8GlobalStateManager.autocast_exit()
```

---

## Architecture Components

### 1. Context Manager (`autocast()`)
- **Purpose:** Enable/disable quantization context
- **Entry:** Set FP8GlobalStateManager state
- **Exit:** Restore previous state
- **File:** quantization.py, lines 789-852

### 2. Global State Manager
- **Purpose:** Track active quantization recipe and configuration globally
- **Class:** `FP8GlobalStateManager`
- **State Variables:** FP8_ENABLED, FP8_RECIPE, AUTOCAST_DEPTH
- **File:** quantization.py, lines 224-677

### 3. Recipe System
- **Base Class:** `Recipe` (common/recipe/__init__.py)
- **Implementations:** 
  - NVFP4BlockScaling (for FP4)
  - MXFP8BlockScaling (for MXFP8)
  - DelayedScaling (for FP8 legacy)
  - Float8CurrentScaling
  - Float8BlockScaling
- **Config:** QParams dataclass with quantization parameters

### 4. RecipeState Factory
- **Purpose:** Create recipe-specific quantizer instances
- **Pattern:** Abstract factory design
- **Dispatch Logic:** recipe.nvfp4() → NVFP4BlockScalingRecipeState
- **File:** quantization.py, lines 967-1026

### 5. Quantizers
- **NVFP4Quantizer:** 4-bit quantization with 2-level block scaling
- **Float8Quantizer:** 8-bit with delayed scaling
- **MXFP8Quantizer:** 8-bit with MXFP8 block scaling

---

## NVFP4BlockScaling Details

### Recipe Configuration

```python
recipe.fp4_format = Format.E2M1
recipe.fp8_format = Format.E4M3

# Quantization modes (configurable via environment variables)
recipe.fp4_quant_fwd_inp = QParams(
    random_hadamard_transform=True,  # Enabled by default
    stochastic_rounding=False,
    fp4_2d_quantization=False,       # 1D blocks for input
)
recipe.fp4_quant_fwd_weight = QParams(
    random_hadamard_transform=False,
    stochastic_rounding=False,
    fp4_2d_quantization=True,        # 2D 16x16 tiles for weights
)
recipe.fp4_quant_bwd_grad = QParams(
    random_hadamard_transform=True,
    stochastic_rounding=True,        # Enabled for gradients
    fp4_2d_quantization=False,       # 1D blocks for gradients
)
```

### Block Sizes

- **Forward Input:** 1D blocks of 16 consecutive values
- **Forward Weight:** 2D blocks of 16x16 (with RHT and 2D quantization)
- **Backward Gradient:** 1D blocks of 16 consecutive values with stochastic rounding

### Scaling Strategy (2-Level)

1. **Level 1 (Block Scales):** E4M3 format, one scale per block
   - For 1D: K/16 scales for tensor of length K
   - For 2D: (M/16) × (N/16) scales for M×N tensor

2. **Level 2 (Global Scale):** FP32 format, one scale for entire tensor

---

## Quantization Pipeline

### Forward Pass

```
Input (BF16)
├─ Apply RHT (if enabled)
├─ Compute amax per block
├─ Compute E4M3 block scales
├─ Compute FP32 global scale
├─ Quantize to FP4 E2M1
└─ Return QuantizedTensor

Weight (BF16)
├─ Apply 2D tile quantization (if enabled)
├─ Compute amax per tile
├─ Compute E4M3 tile scales
├─ Compute FP32 global scale
├─ Quantize to FP4 E2M1
└─ Return QuantizedTensor

GEMM: FP4_input @ FP4_weight
├─ In-kernel dequantization using scales
├─ Tensor Core computation in FP32
└─ Output (FP32 → cast to BF16)
```

### Backward Pass

```
grad_output (BF16)
├─ Apply RHT (if enabled)
├─ Apply stochastic rounding
├─ Compute amax per block
├─ Compute E4M3 block scales
├─ Compute FP32 global scale
├─ Quantize to FP4 E2M1
└─ Return QuantizedTensor

dgrad GEMM: grad_output @ weight.T
└─ grad_input (FP32 → cast to BF16)

wgrad GEMM: input.T @ grad_output
└─ grad_weight (FP32 → cast to BF16)
```

---

## Device Requirements

### Hardware Support

- **NVFP4:** Blackwell architecture (compute capability 10.0+)
- **FP8/MXFP8:** Ada (8.9+), Hopper (9.0+), Blackwell (10.0+)

### Validation Function

```python
def check_nvfp4_support() -> Tuple[bool, str]:
    if get_device_compute_capability() >= (10, 0):  # Blackwell+
        return True, ""
    return False, "Device compute capability 10.0+ required"
```

---

## Key Design Patterns

### 1. Context Manager Pattern
- Uses `@contextmanager` decorator
- Save/restore state for nested contexts
- Guaranteed cleanup with try/finally

### 2. Global State Manager Pattern
- Class variables track global state
- `AUTOCAST_DEPTH` for nested contexts
- Centralized query interface for modules

### 3. Factory Pattern
- `RecipeState.create()` dispatches to recipe-specific class
- `make_quantizers()` creates appropriate quantizers
- Abstracts quantizer creation from modules

### 4. Callable Quantizers
- Quantizers implement `__call__()` method
- Can be applied like functions: `quantizer(tensor)`
- Transparent API for quantization

---

## Testing Infrastructure

### Test Organization

```
tests/pytorch/nvfp4/
├── test_nvfp4_module_exact.py        # Main module tests
├── test_nvfp4_gemm_exact.py          # GEMM kernel tests
├── test_nvfp4_quantize_exact.py      # Quantization tests
├── test_nvfp4_rht_quantize_exact.py  # RHT tests
└── test_nvfp4_sr_quantize.py         # Stochastic rounding tests
```

### Validation Strategy

- Compare native implementation against reference
- Tolerance: rtol=1e-2 (1%), atol=1e-3
- Accounts for quantization error (~1-2%)
- Tests forward and backward passes

---

## Documentation Cross-References

### From TE_AUTOCAST_ANALYSIS.md

- **Section "autocast() Context Manager":** Implementation details (lines 789-852)
- **Section "NVFP4BlockScaling Recipe Integration":** Recipe definition and factory
- **Section "Call Flow: User to Kernel":** Complete data flow diagram
- **Section "Test Case Walkthrough":** Example test execution

### From NVFP4_TEST_WALKTHROUGH.md

- **Section "Basic Module Test Walkthrough":** Factory setup and test execution
- **Section "RHT Quantization Test":** RHT data flow with detailed steps
- **Section "2D Quantization Test":** Weight quantization with 2D blocks
- **Section "GEMM Test Walkthrough":** Kernel execution path
- **Section "Test Validation Mechanisms":** Tolerance rationale

---

## Getting Started

### To Understand autocast():

1. Read: TE_AUTOCAST_ANALYSIS.md, "Architecture Overview" section
2. Read: TE_AUTOCAST_ANALYSIS.md, "autocast() Context Manager" section
3. Reference: quantization.py, lines 789-852

### To Understand NVFP4BlockScaling:

1. Read: TE_AUTOCAST_ANALYSIS.md, "NVFP4BlockScaling Recipe Integration" section
2. Read: NVFP4_TEST_WALKTHROUGH.md, "RHT Quantization Test" and "2D Quantization Test"
3. Reference: common/recipe/__init__.py, lines 386-481

### To Trace Execution:

1. Read: TE_AUTOCAST_ANALYSIS.md, "Call Flow: User to Kernel" section
2. Read: NVFP4_TEST_WALKTHROUGH.md, "Basic Module Test Walkthrough"
3. Reference: Test file execution path diagrams

### To Run Tests:

```bash
cd /home/jeromeku/transformerengine
python -m pytest tests/pytorch/nvfp4/test_nvfp4_module_exact.py -v
```

---

## Environment Setup

### Required

- NVIDIA Blackwell GPU or compatible
- CUDA 12.1+
- cuBLASLt (for FP4 GEMM)
- Compute capability 10.0+

### Optional Environment Variables

```bash
# Control quantization features
export NVTE_NVFP4_DISABLE_RHT=0                    # Enable RHT
export NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING=0   # Enable SR
export NVTE_NVFP4_DISABLE_2D_QUANTIZATION=0       # Enable 2D

# Verify support
python -c "import transformer_engine.pytorch as te; print(te.is_nvfp4_available())"
```

---

## Additional Resources

### Related Files

- `docs/examples/fp8_primer.py` - FP8/FP4 concepts and examples
- `transformer_engine/pytorch/tensor/nvfp4_tensor.py` - NVFP4Quantizer implementation
- `transformer_engine/pytorch/experimental/quantization_nvfp4.py` - Reference implementation

### Papers

- NVFP4: "Pretraining Large Language Models with NVFP4" (https://arxiv.org/abs/2509.25149v1)
- MXFP8: OCP Microscaling Formats MX v1.0 Specification

---

## Document Maintenance

**Last Updated:** 2025-10-27
**Scope:** TransformerEngine with NVFP4BlockScaling recipe
**Coverage:**
- autocast() context manager implementation
- FP8GlobalStateManager state management
- RecipeState factory pattern
- NVFP4 quantization pipeline
- Test case walkthroughs
- Device capability checks

**Companion Documents:**
- TE_AUTOCAST_ANALYSIS.md (Architecture and implementation)
- NVFP4_TEST_WALKTHROUGH.md (Test case traces)

