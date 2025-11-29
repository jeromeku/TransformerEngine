# TransformerEngine MXFP8BlockScaling Analysis - START HERE

Welcome! This directory contains comprehensive documentation analyzing the MXFP8BlockScaling (Microscaling FP8) mixed precision recipe implementation in TransformerEngine, with detailed frame-by-frame execution traces.

## Quick Navigation

### I want to understand...

**The overall architecture:**
- Start with: [README.md](README.md) (Overview and quick reference)
- Then read: [TE_AUTOCAST_ANALYSIS.md](TE_AUTOCAST_ANALYSIS.md) (Detailed architecture)
- Reference: Source files via line numbers provided in documents

**How autocast() works with MXFP8:**
- Read: [TE_AUTOCAST_ANALYSIS.md](TE_AUTOCAST_ANALYSIS.md) → "autocast() Context Manager" section
- Source: `transformer_engine/pytorch/quantization.py:790-852`

**MXFP8BlockScaling recipe:**
- Read: [TE_AUTOCAST_ANALYSIS.md](TE_AUTOCAST_ANALYSIS.md) → "MXFP8BlockScaling Recipe Integration"
- Source: `transformer_engine/common/recipe/__init__.py:265-303`

**Test execution and data flows:**
- Read: [MXFP8_TEST_WALKTHROUGH.md](MXFP8_TEST_WALKTHROUGH.md)
- Source: `tests/pytorch/test_sanity.py`, `tests/pytorch/test_numerics.py`

**The complete call path (user API → kernel):**
- Read: [TE_AUTOCAST_ANALYSIS.md](TE_AUTOCAST_ANALYSIS.md) → "Call Flow: User to Kernel"
- Details: [MXFP8_TEST_WALKTHROUGH.md](MXFP8_TEST_WALKTHROUGH.md) → "Basic Module Test Walkthrough"

**How E8M0 block scaling works:**
- Read: [MXFP8_TEST_WALKTHROUGH.md](MXFP8_TEST_WALKTHROUGH.md) → "E8M0 Quantization Test"

**GEMM kernel execution:**
- Read: [MXFP8_TEST_WALKTHROUGH.md](MXFP8_TEST_WALKTHROUGH.md) → "GEMM Test Walkthrough"

---

## Documentation Structure

```
00_START_HERE.md                    ◄─ You are here
│
├─ README.md                        ◄─ Overview, quick reference, getting started
│
├─ TE_AUTOCAST_ANALYSIS.md         ◄─ Main architecture & implementation document
│  ├─ Architecture overview with diagrams
│  ├─ Core components and file structure
│  ├─ autocast() context manager (lines 790-852)
│  ├─ FP8GlobalStateManager details
│  ├─ MXFP8BlockScalingRecipeState factory
│  ├─ Complete call flow diagram
│  ├─ Integration with te.Linear
│  ├─ Support checks and device validation
│  └─ Key design patterns
│
└─ MXFP8_TEST_WALKTHROUGH.md       ◄─ Test case execution traces
   ├─ Test file organization
   ├─ Module test walkthrough with full call paths
   ├─ E8M0 quantization data flow
   ├─ Block-wise scaling (32 elements)
   ├─ GEMM kernel execution
   ├─ Test validation mechanisms
   ├─ Environment variable controls
   └─ Tolerance rationale
```

---

## Key Findings

### 1. Architecture

**te.autocast()** is a context manager that:
- Enables/disables low-precision quantization globally
- Manages nested contexts via `AUTOCAST_DEPTH`
- Saves/restores state on entry/exit
- Uses `FP8GlobalStateManager` for state tracking

**FP8GlobalStateManager** provides:
- Global quantization state (enabled, recipe, group)
- Query interface for modules
- Support for nested autocast contexts
- No amax tracking (unlike DelayedScaling)

**RecipeState factory** creates:
- Recipe-specific quantizer instances
- MXFP8BlockScalingRecipeState for MXFP8
- Stateless quantizers (no amax history)
- Dispatches to MXFP8Quantizer instances

### 2. MXFP8BlockScaling Recipe

**Features:**
- Block-wise scaling (32-element blocks)
- E8M0 scales (8-bit exponent, power-of-2 only)
- 8-bit FP8 E4M3 data format
- Both rowwise and columnwise quantization
- No Random Hadamard Transform (simpler than NVFP4)
- No stochastic rounding
- No amax history tracking

**Configuration:**
```python
recipe = MXFP8BlockScaling()
# Automatically configures:
# - Block size: 32 elements (fixed)
# - Scale format: E8M0 (power-of-2)
# - Data format: E4M3 (default)
# - No complex features (RHT, SR, etc.)
```

### 3. Quantization Pipeline

**Forward Pass:**
```
Input (BF16)
├─ Divide into 32-element blocks
├─ Compute amax per block
├─ Generate E8M0 scales (power-of-2)
└─ Quantize to FP8 E4M3

GEMM: FP8 @ FP8 → FP32 (with in-kernel dequantization)
└─ Cast to output dtype (BF16)
```

**Backward Pass:**
```
grad_output (BF16)
├─ Divide into 32-element blocks
├─ Compute amax per block
├─ Compute E8M0 scales
└─ Quantize to FP8 E4M3

Gradient GEMMs (dgrad, wgrad) with FP8 tensors
└─ Dequantize to FP32, cast to BF16
```

### 4. Call Flow

```
User Code with te.autocast()
│
├─ FP8GlobalStateManager.autocast_enter()
│  ├─ Set FP8_ENABLED, FP8_RECIPE
│  ├─ Validate device (Blackwell CC 10.0+)
│  └─ Check MXFP8 support
│
├─ te.Linear.forward()
│  ├─ Query: is_fp8_enabled() → True
│  ├─ Get recipe: get_fp8_recipe()
│  ├─ Create state: RecipeState.create(recipe, mode="forward")
│  │  → MXFP8BlockScalingRecipeState
│  ├─ Create quantizers: make_quantizers()
│  │  → MXFP8Quantizer instances
│  ├─ Quantize input/weight to MXFP8
│  ├─ GEMM with quantized tensors
│  └─ Dequantize output
│
└─ FP8GlobalStateManager.autocast_exit()
   └─ Restore previous state
```

### 5. Key Design Patterns

1. **Context Manager:** Save/restore state, guaranteed cleanup
2. **Global State Manager:** Class-level state, centralized queries
3. **Factory Pattern:** RecipeState.create() dispatches to MXFP8BlockScalingRecipeState
4. **Callable Objects:** Quantizers implement `__call__()` for transparent API
5. **Dataclass Recipes:** Type-safe, immutable recipe definitions
6. **Stateless Quantizers:** No amax history, scales computed per-call

---

## Most Important Source Files

### Core Implementation

1. **quantization.py** (1,397 lines)
   - Lines 224-677: FP8GlobalStateManager
   - Lines 790-852: autocast() context manager
   - Lines 967-1026: RecipeState factory
   - Lines 1130-1162: MXFP8BlockScalingRecipeState

2. **common/recipe/__init__.py** (515 lines)
   - Lines 265-303: MXFP8BlockScaling recipe definition
   - Lines 85-112: Recipe type checking methods
   - Lines 62-83: QParams configuration (not used for MXFP8)

3. **pytorch/tensor/mxfp8_tensor.py** (943 lines)
   - Lines 27-175: MXFP8Quantizer implementation
   - Lines 177-943: MXFP8Tensor class
   - MXFP8 quantization API

4. **pytorch/tensor/storage/mxfp8_tensor_storage.py** (257 lines)
   - Lines 50-257: MXFP8TensorStorage class
   - Memory layout and dequantization

5. **pytorch/csrc/extensions/cast.cpp** (C++ binding)
   - Lines 33-79: quantize() PyBind11 binding
   - Lines 347-492: MXFP8 tensor operations

6. **common/cast/cast.cu** (CUDA kernels)
   - MXFP8 quantization kernel
   - E8M0 scale computation
   - Block-wise amax reduction

### Tests

- `tests/pytorch/test_sanity.py` (MXFP8 inference tests)
- `tests/pytorch/test_numerics.py` (MXFP8 training accuracy tests)
- `tests/cpp/operator/test_cast_mxfp8.cu` (CUDA kernel tests)

---

## Reading Recommendations

### For Deep Understanding (Read in Order):

1. **README.md** (15 min) - Get oriented
2. **TE_AUTOCAST_ANALYSIS.md** "Architecture Overview" (10 min) - See big picture
3. **TE_AUTOCAST_ANALYSIS.md** "autocast() Context Manager" (15 min) - Understand entry/exit
4. **TE_AUTOCAST_ANALYSIS.md** "MXFP8BlockScaling Recipe Integration" (10 min) - Recipe specifics
5. **TE_AUTOCAST_ANALYSIS.md** "Call Flow: User to Kernel" (15 min) - Complete path
6. **MXFP8_TEST_WALKTHROUGH.md** "Basic Module Test Walkthrough" (20 min) - See it in action
7. **MXFP8_TEST_WALKTHROUGH.md** "E8M0 Quantization Test" (15 min) - Feature details

**Total: ~1.5 hours for comprehensive understanding**

### For Quick Reference:

- **README.md:** Architecture components, quick reference patterns
- **TE_AUTOCAST_ANALYSIS.md:** Key sections with line numbers
- **MXFP8_TEST_WALKTHROUGH.md:** Test case structure

---

## Code Verification

All line numbers and file paths have been verified against the actual codebase:

```
✓ transformer_engine/pytorch/quantization.py (1,397 lines)
✓ transformer_engine/common/recipe/__init__.py (515 lines)
✓ transformer_engine/pytorch/tensor/mxfp8_tensor.py (943 lines)
✓ transformer_engine/pytorch/tensor/storage/mxfp8_tensor_storage.py (257 lines)
✓ Tests in tests/pytorch/test_sanity.py, test_numerics.py
```

---

## Key Takeaways

1. **te.autocast() is lightweight:** Just manages global state and delegates to modules
2. **Recipes are immutable:** Configuration set at recipe creation time
3. **Quantizers are stateless:** No amax history, scales computed per-call
4. **Factory pattern enables extensibility:** Easy to add new recipes
5. **Device validation is critical:** Checks happen at autocast_enter()
6. **Nested contexts are supported:** AUTOCAST_DEPTH tracks nesting level
7. **State is always restored:** try/finally ensures cleanup even on errors
8. **Simpler than NVFP4:** No RHT, no 2D quantization, no stochastic rounding

---

## Device Requirements

- **MXFP8:** Compute capability 10.0+ (Blackwell)
- **NVFP4:** Compute capability 10.0+ (Blackwell)
- **FP8:** Compute capability 8.9+ (Ada) or 9.0+ (Hopper)

Check availability:
```python
import transformer_engine.pytorch as te
# MXFP8 uses same check as NVFP4
is_available, reason = te.is_nvfp4_available()
print(f"MXFP8 available: {is_available}, Reason: {reason}")
```

---

## MXFP8 vs NVFP4 Key Differences

| Feature | MXFP8 | NVFP4 |
|---------|-------|-------|
| **Bits per element** | 8 | 4 |
| **Block size** | 32 elements | 16 elements (1D), 16×16 (2D) |
| **Scale format** | E8M0 (power-of-2) | E4M3 + FP32 (2-level) |
| **Random Hadamard Transform** | No | Yes (optional) |
| **2D quantization** | No | Yes (for weights) |
| **Stochastic rounding** | No | Yes (for gradients) |
| **Complexity** | Simple | Complex |
| **Precision** | Higher (8-bit) | Lower (4-bit) |
| **Compression** | 4× vs FP32 | 8× vs FP32 |

---

## Next Steps

1. **To experiment:** See README.md "Quick Reference: Autocast Usage Pattern"
2. **To understand internals:** Follow "Reading Recommendations" section
3. **To debug issues:** Reference specific line numbers in TE_AUTOCAST_ANALYSIS.md
4. **To run tests:** See README.md "Getting Started" section

---

## Document Information

**Created:** 2025-01-12
**Scope:** TransformerEngine te.autocast() with MXFP8BlockScaling
**Coverage:**
- Architecture and design patterns
- Complete implementation details
- Full test walkthroughs with data flows
- Device capability requirements
- Configuration and environment variables

**Main Documents:**
- `README.md` (Overview)
- `TE_AUTOCAST_ANALYSIS.md` (Architecture & Implementation)
- `MXFP8_TEST_WALKTHROUGH.md` (Tests & Execution Traces)

---

## Questions Answered

- **Where is autocast() defined?** → quantization.py:790-852
- **How does it handle MXFP8BlockScaling?** → Via RecipeState factory pattern
- **What is the context manager implementation?** → @contextmanager decorator with save/restore
- **How does it affect te.Linear?** → Linear queries FP8GlobalStateManager for recipe
- **What is the complete call path?** → See "Call Flow: User to Kernel" diagram
- **How are quantizers created?** → RecipeState factory dispatches to MXFP8BlockScalingRecipeState
- **How does test execution work?** → See MXFP8_TEST_WALKTHROUGH.md
- **What's the difference from NVFP4?** → Simpler: no RHT, no 2D, no SR; 8-bit vs 4-bit

---

**Happy exploring! Start with README.md for a quick overview.**
