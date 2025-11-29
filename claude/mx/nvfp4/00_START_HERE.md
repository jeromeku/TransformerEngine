# TransformerEngine te.autocast() Analysis - START HERE

Welcome! This directory contains comprehensive documentation analyzing the `te.autocast()` implementation in TransformerEngine, with detailed focus on NVFP4BlockScaling recipe integration.

## Quick Navigation

### I want to understand...

**The overall architecture:**
- Start with: `README.md` (Overview and quick reference)
- Then read: `TE_AUTOCAST_ANALYSIS.md` (Detailed architecture)
- Reference: Source files via line numbers provided in documents

**How autocast() works:**
- Read: `TE_AUTOCAST_ANALYSIS.md` → "autocast() Context Manager" section
- Source: `transformer_engine/pytorch/quantization.py:789-852`

**NVFP4BlockScaling recipe:**
- Read: `TE_AUTOCAST_ANALYSIS.md` → "NVFP4BlockScaling Recipe Integration"
- Source: `transformer_engine/common/recipe/__init__.py:386-481`

**Test execution and data flows:**
- Read: `NVFP4_TEST_WALKTHROUGH.md`
- Source: `tests/pytorch/nvfp4/test_nvfp4_module_exact.py`

**The complete call path (user API → kernel):**
- Read: `TE_AUTOCAST_ANALYSIS.md` → "Call Flow: User to Kernel"
- Details: `NVFP4_TEST_WALKTHROUGH.md` → "Basic Module Test Walkthrough"

**How RHT and stochastic rounding work:**
- Read: `NVFP4_TEST_WALKTHROUGH.md` → "RHT Quantization Test"

**2D quantization for weights:**
- Read: `NVFP4_TEST_WALKTHROUGH.md` → "2D Quantization Test"

**GEMM kernel execution:**
- Read: `NVFP4_TEST_WALKTHROUGH.md` → "GEMM Test Walkthrough"

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
│  ├─ autocast() context manager (lines 789-852)
│  ├─ FP8GlobalStateManager details
│  ├─ NVFP4BlockScalingRecipeState factory
│  ├─ Complete call flow diagram
│  ├─ Integration with te.Linear
│  ├─ Support checks and device validation
│  └─ Key design patterns
│
├─ NVFP4_TEST_WALKTHROUGH.md       ◄─ Test case execution traces
│  ├─ Test file organization
│  ├─ Module test walkthrough with full call paths
│  ├─ RHT quantization data flow
│  ├─ 2D quantization for weights
│  ├─ GEMM kernel execution
│  ├─ Test validation mechanisms
│  ├─ Environment variable controls
│  └─ Tolerance rationale
│
└─ (Previously created documents)
   ├─ ARCHITECTURE.md              (Extensive overall architecture)
   ├─ NVFP4_LINEAR_CALL_PATH.md   (Linear-specific call paths)
   ├─ NVFP4_QUANTIZE_DISPATCH.md  (Quantization dispatch details)
   ├─ test_nvfp4_walkthrough.md   (Test walkthroughs)
   ├─ test_blockwise_fp8_walkthrough.md
   └─ INDEX.md                     (Document index)
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
- Amax reduction and scaling factor management
- Support for nested autocast contexts

**RecipeState factory** creates:
- Recipe-specific quantizer instances
- Appropriate configurations for forward/backward passes
- Dispatches to recipe-specific classes (e.g., NVFP4BlockScalingRecipeState)

### 2. NVFP4BlockScaling Recipe

**Features:**
- 2-level block scaling (E4M3 block scales + FP32 global scale)
- 1D blocks for inputs/gradients (16 consecutive values)
- 2D blocks for weights (16x16 tiles)
- Random Hadamard Transform (RHT) for distribution smoothing
- Stochastic rounding for gradients

**Configuration:**
```python
recipe = NVFP4BlockScaling()
# Automatically configures:
# - fp4_quant_fwd_inp: with RHT, 1D blocks
# - fp4_quant_fwd_weight: 2D blocks (16x16)
# - fp4_quant_bwd_grad: with RHT and stochastic rounding
```

### 3. Quantization Pipeline

**Forward Pass:**
```
Input/Weight (BF16)
├─ Apply transformations (RHT, etc.)
├─ Compute amax per block
├─ Generate E4M3 block scales
├─ Generate FP32 global scale
└─ Quantize to FP4 E2M1

GEMM: FP4 @ FP4 → FP32 (with in-kernel dequantization)
└─ Cast to output dtype (BF16)
```

**Backward Pass:**
```
grad_output (BF16)
├─ Apply RHT
├─ Apply stochastic rounding
├─ Compute scales
└─ Quantize to FP4 E2M1

Gradient GEMMs (dgrad, wgrad) with FP4 tensors
└─ Dequantize to FP32, cast to BF16
```

### 4. Call Flow

```
User Code with te.autocast()
│
├─ FP8GlobalStateManager.autocast_enter()
│  └─ Set FP8_ENABLED, FP8_RECIPE, validate device
│
├─ te.Linear.forward()
│  ├─ Query: is_fp8_enabled() → True
│  ├─ Get recipe: get_fp8_recipe()
│  ├─ Create state: RecipeState.create(recipe, mode="forward")
│  ├─ Create quantizers: NVFP4BlockScalingRecipeState.make_quantizers()
│  ├─ Quantize input/weight
│  ├─ GEMM with quantized tensors
│  └─ Dequantize output
│
└─ FP8GlobalStateManager.autocast_exit()
   └─ Restore previous state
```

### 5. Key Design Patterns

1. **Context Manager:** Save/restore state, guaranteed cleanup
2. **Global State Manager:** Class-level state, centralized queries
3. **Factory Pattern:** RecipeState.create() dispatches to specific class
4. **Callable Objects:** Quantizers implement `__call__()` for transparent API
5. **Dataclass Recipes:** Type-safe, immutable recipe definitions

---

## Most Important Source Files

### Core Implementation

1. **quantization.py** (1397 lines)
   - Lines 224-677: FP8GlobalStateManager
   - Lines 789-852: autocast() context manager
   - Lines 967-1026: RecipeState factory
   - Lines 1270-1343: NVFP4BlockScalingRecipeState

2. **common/recipe/__init__.py** (515 lines)
   - Lines 386-481: NVFP4BlockScaling recipe definition
   - Lines 62-83: QParams configuration
   - Lines 23-44: Format enum

3. **pytorch/module/linear.py**
   - _Linear autograd.Function
   - Forward/backward with quantization

### Tests

- `tests/pytorch/nvfp4/test_nvfp4_module_exact.py` (Main module tests)
- `tests/pytorch/nvfp4/test_nvfp4_gemm_exact.py` (GEMM tests)
- `tests/pytorch/nvfp4/test_nvfp4_quantize_exact.py` (Quantization tests)

---

## Reading Recommendations

### For Deep Understanding (Read in Order):

1. **README.md** (15 min) - Get oriented
2. **TE_AUTOCAST_ANALYSIS.md** "Architecture Overview" (10 min) - See big picture
3. **TE_AUTOCAST_ANALYSIS.md** "autocast() Context Manager" (15 min) - Understand entry/exit
4. **TE_AUTOCAST_ANALYSIS.md** "NVFP4BlockScaling Recipe Integration" (10 min) - Recipe specifics
5. **TE_AUTOCAST_ANALYSIS.md** "Call Flow: User to Kernel" (15 min) - Complete path
6. **NVFP4_TEST_WALKTHROUGH.md** "Basic Module Test Walkthrough" (20 min) - See it in action
7. **NVFP4_TEST_WALKTHROUGH.md** "RHT Quantization Test" + "2D Quantization Test" (15 min) - Feature details

**Total: ~1.5 hours for comprehensive understanding**

### For Quick Reference:

- **README.md:** Architecture components, quick reference patterns
- **TE_AUTOCAST_ANALYSIS.md:** Key sections with line numbers
- **NVFP4_TEST_WALKTHROUGH.md:** Test case structure

---

## Code Verification

All line numbers and file paths have been verified against the actual codebase:

```
✓ /home/jeromeku/transformerengine/transformer_engine/pytorch/quantization.py (1397 lines)
✓ /home/jeromeku/transformerengine/transformer_engine/common/recipe/__init__.py (515 lines)
✓ /home/jeromeku/transformerengine/transformer_engine/pytorch/module/linear.py
✓ Tests in /home/jeromeku/transformerengine/tests/pytorch/nvfp4/
```

---

## Key Takeaways

1. **te.autocast() is lightweight:** Just manages global state and delegates to modules
2. **Recipes are immutable:** Configuration set at recipe creation time
3. **Quantizers are stateless:** Can be created/destroyed per forward pass
4. **Factory pattern enables extensibility:** Easy to add new recipes
5. **Device validation is critical:** Checks happen at autocast_enter()
6. **Nested contexts are supported:** AUTOCAST_DEPTH tracks nesting level
7. **State is always restored:** try/finally ensures cleanup even on errors

---

## Device Requirements

- **NVFP4:** Compute capability 10.0+ (Blackwell)
- **FP8/MXFP8:** Compute capability 8.9+ (Ada) or 9.0+ (Hopper)

Check availability:
```python
import transformer_engine.pytorch as te
print(te.is_nvfp4_available())  # Returns (True, "") or (False, reason)
```

---

## Next Steps

1. **To experiment:** See README.md "Quick Reference: Autocast Usage Pattern"
2. **To understand internals:** Follow "Reading Recommendations" section
3. **To debug issues:** Reference specific line numbers in TE_AUTOCAST_ANALYSIS.md
4. **To run tests:** See README.md "Getting Started" section

---

## Document Information

**Created:** 2025-10-27  
**Scope:** TransformerEngine te.autocast() with NVFP4BlockScaling  
**Coverage:** 
- Architecture and design patterns
- Complete implementation details
- Full test walkthroughs with data flows
- Device capability requirements
- Configuration and environment variables

**Main Documents:**
- `README.md` (Overview)
- `TE_AUTOCAST_ANALYSIS.md` (Architecture & Implementation)
- `NVFP4_TEST_WALKTHROUGH.md` (Tests & Execution Traces)

---

## Questions Answered

- **Where is autocast() defined?** → quantization.py:789-852
- **How does it handle NVFP4BlockScaling?** → Via RecipeState factory pattern
- **What is the context manager implementation?** → @contextmanager decorator with save/restore
- **How does it affect te.Linear?** → Linear queries FP8GlobalStateManager for recipe
- **What is the complete call path?** → See "Call Flow: User to Kernel" diagram
- **How are quantizers created?** → RecipeState factory dispatches to recipe-specific class
- **How does test execution work?** → See NVFP4_TEST_WALKTHROUGH.md

---

**Happy exploring! Start with README.md for a quick overview.**

