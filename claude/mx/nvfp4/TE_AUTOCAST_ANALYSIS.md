# TransformerEngine `te.autocast()` Implementation Analysis

## Overview

This document provides a comprehensive analysis of the `te.autocast()` context manager implementation in TransformerEngine, with focus on how it enables NVFP4BlockScaling and other quantization recipes. The analysis covers the complete call path from user-facing APIs through context management to low-precision quantized computation.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [autocast() Context Manager](#autocast-context-manager)
4. [NVFP4BlockScaling Recipe Integration](#nvfp4blockscaling-recipe-integration)
5. [Global State Management](#global-state-management)
6. [Call Flow: User to Kernel](#call-flow-user-to-kernel)
7. [Integration with te.Linear](#integration-with-telinear)
8. [Test Case Walkthrough](#test-case-walkthrough)

---

## Architecture Overview

### High-Level Design

```
┌─────────────────────────────────────────────────────────┐
│  User Application                                       │
│  with te.autocast(recipe=NVFP4BlockScaling()):         │
│      output = model(input)                              │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│  FP8GlobalStateManager                                  │
│  - Manages global quantization context                  │
│  - Tracks FP8_ENABLED, FP8_RECIPE, autocast_arguments  │
│  - Handles amax reduction and scaling factor updates    │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│  te.Linear / TransformerEngine Modules                  │
│  - Query FP8GlobalStateManager for quantization config  │
│  - Create quantizers via RecipeState factories          │
│  - Apply quantization during forward/backward passes    │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│  Quantizer Implementations                              │
│  - NVFP4Quantizer (for FP4 tensors)                    │
│  - Float8Quantizer (for FP8 with delayed scaling)      │
│  - MXFP8Quantizer (for MXFP8 with block scaling)       │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│  CUDA Kernels & cuBLAS/cuBLASLt                        │
│  - Quantization kernels (cast to FP4/FP8)              │
│  - GEMM kernels (FP8/FP4 tensor cores)                 │
│  - Dequantization kernels                              │
└─────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. File Structure

```
transformer_engine/
├── pytorch/
│   ├── quantization.py                    # Core autocast implementation
│   │   ├── FP8GlobalStateManager          # Global state tracking
│   │   ├── autocast()                     # Context manager
│   │   ├── RecipeState (factory)          # Recipe state builders
│   │   └── NVFP4BlockScalingRecipeState   # NVFP4-specific state
│   │
│   ├── module/
│   │   ├── linear.py                      # Linear layer implementation
│   │   ├── base.py                        # Base module with quantization logic
│   │   └── _Linear                        # autograd.Function wrapper
│   │
│   └── tensor/
│       ├── nvfp4_tensor.py               # NVFP4Quantizer implementation
│       ├── float8_tensor.py              # FP8 quantizers
│       └── mxfp8_tensor.py               # MXFP8 quantizers
│
└── common/
    └── recipe/
        └── __init__.py                   # Recipe definitions (NVFP4BlockScaling, etc.)
```

---

## autocast() Context Manager

### Definition

**File:** [transformer_engine/pytorch/quantization.py](../../../../transformer_engine/pytorch/quantization.py#L789-L852)

```python
@contextmanager
def autocast(
    enabled: bool = True,
    calibrating: bool = False,
    recipe: Optional["Recipe"] = None,
    amax_reduction_group: Optional["dist_group_type"] = None,
    _graph: bool = False,
) -> None:
    """
    Context manager for quantization schemes like FP8 or FP4.
    
    Parameters
    ----------
    enabled: bool, default = `True`
             whether or not to enable low precision quantization (FP8/FP4).
    calibrating: bool, default = `False`
                 calibration mode allows collecting statistics (amax, scale)
                 even when executing without quantization enabled.
    recipe: recipe.Recipe, default = `None`
            recipe used for low precision quantization.
    amax_reduction_group: torch._C._distributed_c10d.ProcessGroup, default = `None`
                          distributed group over which amaxes are reduced.
    """
```

### Implementation Flow

**Step 1: Entry Logic (lines 835-847)**

```python
if enabled:
    check_recipe_support(recipe)              # ◄─ Validate recipe support

# Save current state so we always restore it on exit.
fp8_state = FP8GlobalStateManager.get_autocast_state()

FP8GlobalStateManager.autocast_enter(
    enabled=enabled,
    calibrating=calibrating,
    fp8_recipe=recipe,
    fp8_group=amax_reduction_group,
    _graph=_graph,
)
```

**Step 2: Yield (line 849)**
- Context is active, user code runs
- Modules query `FP8GlobalStateManager.is_fp8_enabled()` and `FP8GlobalStateManager.get_fp8_recipe()`
- Quantizers are created based on recipe
- Forward and backward passes execute with quantization enabled

**Step 3: Exit Logic (lines 850-852)**

```python
finally:
    FP8GlobalStateManager.set_autocast_state(fp8_state)  # Restore previous state
    FP8GlobalStateManager.autocast_exit(enabled, _graph=_graph)
```

### Key Methods in FP8GlobalStateManager

#### `autocast_enter()` [lines [553-589](../../../../transformer_engine/pytorch/quantization.py#L553-L589)]

```python
@classmethod
def autocast_enter(
    cls,
    enabled: bool = False,
    calibrating: bool = False,
    fp8_recipe: Optional[Recipe] = None,
    fp8_group: Optional[dist_group_type] = None,
    _graph: bool = False,
) -> None:
    """Set state and tracking variables for entry into FP8 region."""
    
    fp8_recipe = get_default_fp8_recipe() if fp8_recipe is None else fp8_recipe
    autocast_key = cls.get_unique_autocast_key(fp8_recipe, fp8_group)
    cls.autocast_arguments[autocast_key] = (fp8_recipe, fp8_group)
    
    cls.FP8_ENABLED = enabled
    cls.FP8_CALIBRATION = calibrating
    cls.FP8_RECIPE = fp8_recipe
    cls.FP8_DISTRIBUTED_GROUP = fp8_group
    cls.FP8_GRAPH_CAPTURING = _graph
    
    if cls.AUTOCAST_DEPTH == 0:
        cls.IS_FIRST_FP8_MODULE = True
    cls.AUTOCAST_DEPTH += 1
    
    # Recipe-specific validation
    if enabled:
        fp8_available, reason_for_no_fp8 = cls.is_fp8_available()
        assert fp8_available, reason_for_no_fp8
        if isinstance(fp8_recipe, MXFP8BlockScaling):
            mxfp8_available, reason_for_no_mxfp8 = cls.is_mxfp8_available()
            assert mxfp8_available, reason_for_no_mxfp8
        if isinstance(fp8_recipe, Float8BlockScaling):
            fp8_block_available, reason_for_no_fp8_block = cls.is_fp8_block_scaling_available()
            assert fp8_block_available, reason_for_no_fp8_block
        if isinstance(fp8_recipe, NVFP4BlockScaling):  # ◄─ NVFP4-specific
            nvfp4_available, reason_for_no_nvfp4 = cls.is_nvfp4_available()
            assert nvfp4_available, reason_for_no_nvfp4
```

#### `autocast_exit()` [lines [591-601](../../../../transformer_engine/pytorch/quantization.py#L591-L601)]

```python
@classmethod
def autocast_exit(cls, enabled: bool, _graph: bool) -> None:
    """Set state and tracking variables for exit from FP8 region."""
    cls.AUTOCAST_DEPTH -= 1
    # Reduce only the non-FP8 weight modules here.
    # FP8 weight modules are reduced at the end of the optimizer
    # step after the weight amax is populated.
    if enabled and cls.AUTOCAST_DEPTH == 0 and not _graph and torch.is_grad_enabled():
        # delayed scaling only function, for other recipes (current scaling with any granularity),
        # this is noop for other recipes because cls.global_amax_buffer is empty list
        cls.reduce_and_update_fp8_tensors(forward=True)
```

---

## NVFP4BlockScaling Recipe Integration

### Recipe Definition

**File:** [transformer_engine/common/recipe/__init__.py](../../../../transformer_engine/common/recipe/__init__.py#L386-L481)

```python
@dataclass()
class NVFP4BlockScaling(Recipe):
    """
    Use the NVFP4 scaling strategy.
    
    This is a 2-level block scaling strategy. In level 1, each group of
    16 consecutive values is scaled together using their own scaling
    factor. The type of the scaling factor is E4M3 (4 bits of exponent,
    3 bits of mantissa). In level 2, a global per tensor FP32 scaling
    factor is used to scale the entire tensor.
    """
    
    # Configuration envvars
    disable_rht: bool = os.getenv("NVTE_NVFP4_DISABLE_RHT", "0") == "1"
    disable_stochastic_rounding: bool = (
        os.getenv("NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING", "0") == "1"
    )
    disable_2d_quantization: bool = os.getenv("NVTE_NVFP4_DISABLE_2D_QUANTIZATION", "0") == "1"
    
    fp4_format: Format = Format.E2M1
    fp8_format: Format = Format.E4M3
    
    fp8_dpa: bool = False
    fp8_mha: bool = False
    
    def __post_init__(self) -> None:
        assert self.fp4_format == Format.E2M1, "Only E2M1 is supported for NVFP4 scaling"
        assert self.fp8_format == Format.E4M3, "Only E4M3 is supported for NVFP4 scaling"
        
        # Quantization params
        self.fp4_quant_fwd_inp = QParams(
            random_hadamard_transform=not self.disable_rht,
            stochastic_rounding=False,
            fp4_2d_quantization=False,
        )
        self.fp4_quant_fwd_weight = QParams(
            random_hadamard_transform=False,
            stochastic_rounding=False,
            fp4_2d_quantization=not self.disable_2d_quantization,
        )
        self.fp4_quant_bwd_grad = QParams(
            random_hadamard_transform=not self.disable_rht,
            stochastic_rounding=not self.disable_stochastic_rounding,
            fp4_2d_quantization=False,
        )
```

### RecipeState Factory

**File:** [transformer_engine/pytorch/quantization.py](../../../../transformer_engine/pytorch/quantization.py#L967-L1026)

```python
class RecipeState(abc.ABC):
    """Configuration and state for a quantization recipe.
    
    This is a builder class for quantizers, which are in turn builder
    classes for quantized tensors.
    """
    
    @staticmethod
    def create(
        recipe: Recipe,
        *,
        mode: str,
        num_quantizers: int = 1,
        device: Optional[torch.device] = None,
    ) -> RecipeState:
        """Factory method to create the state for a quantization recipe"""
        
        cls = None
        if recipe.delayed():
            cls = DelayedScalingRecipeState
        elif recipe.mxfp8():
            cls = MXFP8BlockScalingRecipeState
        elif recipe.float8_current_scaling():
            cls = Float8CurrentScalingRecipeState
        elif recipe.float8_block_scaling():
            cls = Float8BlockScalingRecipeState
        elif recipe.nvfp4():  # ◄─ NVFP4 path
            cls = NVFP4BlockScalingRecipeState
        elif recipe.custom():
            cls = CustomRecipeState
        else:
            raise ValueError(f"{recipe.__class__.__name__} is not supported")
        return cls(
            recipe,
            mode=mode,
            num_quantizers=num_quantizers,
            device=device,
        )
```

### NVFP4BlockScalingRecipeState

**File:** [transformer_engine/pytorch/quantization.py](../../../../transformer_engine/pytorch/quantization.py#L1270-L1343)

```python
class NVFP4BlockScalingRecipeState(RecipeState):
    """Configuration for NVFP4 quantization.
    
    NVFP4 quantization does not require state.
    """
    
    recipe: NVFP4BlockScaling
    mode: str
    dtype: tex.DType
    
    def __init__(
        self,
        recipe: NVFP4BlockScaling,
        *,
        mode: str,
        num_quantizers: int = 1,
        device: Optional[torch.device] = None,
    ) -> None:
        self.recipe = recipe
        self.mode = mode
        self.num_quantizers = num_quantizers
        self.dtype = get_fp4_te_dtype(recipe)  # Gets tex.DType.kFloat4E2M1
        
        if device is None:
            device = torch.device("cuda")
    
    def make_quantizers(self) -> list:
        """Create NVFP4Quantizer instances based on mode."""
        from .tensor.nvfp4_tensor import NVFP4Quantizer
        
        if self.mode == "forward":
            def _make_quantizer(idx: int) -> NVFP4Quantizer:
                qparams = (
                    self.recipe.fp4_quant_fwd_weight
                    if idx % 3 == 1
                    else self.recipe.fp4_quant_fwd_inp
                )
                return NVFP4Quantizer(
                    fp4_dtype=self.dtype,
                    rowwise=True,
                    columnwise=True,
                    with_rht=qparams.random_hadamard_transform,
                    with_post_rht_amax=qparams.random_hadamard_transform,
                    with_2d_quantization=qparams.fp4_2d_quantization,
                    stochastic_rounding=qparams.stochastic_rounding,
                )
            
            return [_make_quantizer(idx) for idx in range(self.num_quantizers)]
        
        if self.mode == "backward":
            return [
                NVFP4Quantizer(
                    fp4_dtype=self.dtype,
                    rowwise=True,
                    columnwise=True,
                    with_rht=self.recipe.fp4_quant_bwd_grad.random_hadamard_transform,
                    with_post_rht_amax=self.recipe.fp4_quant_bwd_grad.random_hadamard_transform,
                    with_2d_quantization=self.recipe.fp4_quant_bwd_grad.fp4_2d_quantization,
                    stochastic_rounding=self.recipe.fp4_quant_bwd_grad.stochastic_rounding,
                )
                for _ in range(self.num_quantizers)
            ]
        
        raise RuntimeError(f"Unexpected recipe mode ({self.mode})")
```

---

## Global State Management

### FP8GlobalStateManager Class Variables

**File:** [transformer_engine/pytorch/quantization.py](../../../../transformer_engine/pytorch/quantization.py#L224-L252)

```python
class FP8GlobalStateManager:
    """Class to keep track of and manipulate the global FP8 state at different stages."""
    
    # Core quantization state
    FP8_ENABLED = False                        # Is FP8/FP4 quantization active?
    FP8_CALIBRATION = False                    # Is calibration mode active?
    FP8_RECIPE = None                          # Current recipe instance
    FP8_DISTRIBUTED_GROUP = None               # Distributed group for amax reduction
    FP8_PARAMETERS = False                     # Should parameters be stored as FP8?
    HIGH_PRECISION_INIT_VAL = False            # Store high precision init values?
    IS_FIRST_FP8_MODULE = False                # Is this the first module in autocast?
    FP8_GRAPH_CAPTURING = False                # Is CUDA graph capturing active?
    AUTOCAST_DEPTH = 0                         # Nested autocast depth counter
    
    # Global buffers for delayed scaling (per recipe)
    global_amax_buffer = {}                    # Map: buffer_key -> [amax tensors]
    global_amax_history_buffer = {}            # Map: buffer_key -> [amax_history tensors]
    global_scale_buffer = {}                   # Map: buffer_key -> [scale tensors]
    
    # Recipe tracking
    fp8_tensors_recompute_buffer = []          # For activation checkpointing
    autocast_arguments = {}                    # Map: autocast_key -> (recipe, group)
    
    # Availability flags
    fp8_available = None
    reason_for_no_fp8 = ""
    mxfp8_available = None
    reason_for_no_mxfp8 = ""
    fp8_block_scaling_available = None
    reason_for_no_fp8_block_scaling = None
    nvfp4_available = None
    reason_for_no_nvfp4 = ""
```

### State Query Methods

```python
@classmethod
def is_fp8_enabled(cls) -> bool:
    """Is FP8 enabled"""
    return cls.FP8_ENABLED

@classmethod
def get_fp8_recipe(cls) -> Recipe:
    """Return the fp8 recipe"""
    if cls.FP8_RECIPE is not None:
        return cls.FP8_RECIPE
    return get_default_fp8_recipe()

@classmethod
def get_fp8_group(cls) -> Union[dist_group_type, None]:
    """Return the fp8 group for scale/amax comm"""
    return cls.FP8_DISTRIBUTED_GROUP

@classmethod
def is_first_fp8_module(cls):
    """Returns `True` only the first time when called multiple times
    from within the same `autocast` context."""
    tmp = cls.IS_FIRST_FP8_MODULE
    cls.IS_FIRST_FP8_MODULE = False
    return tmp
```

---

## Call Flow: User to Kernel

### Complete Flow Diagram

```
Application Code
│
├─ with te.autocast(enabled=True, recipe=NVFP4BlockScaling()):
│
├──► FP8GlobalStateManager.autocast_enter()
│    ├─ set FP8_ENABLED = True
│    ├─ set FP8_RECIPE = NVFP4BlockScaling()
│    ├─ check_nvfp4_support() ─► Verify Blackwell (CC >= 10.0)
│    └─ AUTOCAST_DEPTH += 1
│
├─ model(input)
│  │
│  ├─► te.Linear.forward()
│  │   │
│  │   ├─ Query FP8GlobalStateManager.is_fp8_enabled() ─► True
│  │   │
│  │   ├─ Create quantizers via RecipeState.create(recipe="NVFP4BlockScaling", mode="forward")
│  │   │  │
│  │   │  └─► NVFP4BlockScalingRecipeState.make_quantizers()
│  │   │      ├─ Create NVFP4Quantizer for input (with RHT)
│  │   │      ├─ Create NVFP4Quantizer for weight (with 2D quantization)
│  │   │      └─ Create NVFP4Quantizer for output
│  │   │
│  │   ├─ Call _Linear.forward() [autograd.Function]
│  │   │  │
│  │   │  ├─ Quantize input:
│  │   │  │  │ input_quantizer(input)
│  │   │  │  │   ├─ Apply random Hadamard transform (RHT)
│  │   │  │  │   ├─ Compute max-abs (amax) per block (16 consecutive values)
│  │   │  │  │   ├─ Compute block scaling factors (E4M3)
│  │   │  │  │   ├─ Compute global scale factor (FP32)
│  │   │  │  │   └─ Return quantized tensor (FP4)
│  │   │  │
│  │   │  ├─ Quantize weight:
│  │   │  │  │ weight_quantizer(weight)
│  │   │  │  │   ├─ Apply 2D block quantization (16x16 tiles)
│  │   │  │  │   ├─ Compute block scales per tile
│  │   │  │  │   ├─ Compute global scale
│  │   │  │  │   └─ Return quantized weight (FP4)
│  │   │  │
│  │   │  ├─ GEMM: FP4 @ FP4 → FP32
│  │   │  │  │ tex.general_gemm() / cuBLASLt
│  │   │  │  │   ├─ Input: quantized FP4 tensors + scales
│  │   │  │  │   ├─ Kernel: Tensor Core GEMM
│  │   │  │  │   └─ Output: FP32 accumulation
│  │   │  │
│  │   │  ├─ Dequantize output:
│  │   │  │  │ Scale FP32 output using block scales
│  │   │  │
│  │   │  └─ Return output (BF16 or other precision)
│  │   │
│  │   └─ Save tensors for backward:
│  │       ├─ input_quantizer (stores scale factors)
│  │       ├─ weight (FP4 or original)
│  │       └─ output (for gradient computation)
│  │
│  ├─► Backward Pass (when needed)
│  │   │
│  │   ├─ _Linear.backward()
│  │   │  │
│  │   │  ├─ Quantize grad_output:
│  │   │  │  │ grad_output_quantizer(grad_output)
│  │   │  │  │   ├─ Apply RHT and stochastic rounding
│  │   │  │  │   └─ Return quantized grad (FP4)
│  │   │  │
│  │   │  ├─ Compute grad_input:
│  │   │  │  │ GEMM: grad_output (FP4) @ weight.T (FP4) → grad_input (FP32)
│  │   │  │
│  │   │  ├─ Compute grad_weight:
│  │   │  │  │ GEMM: input.T (FP4) @ grad_output (FP4) → grad_weight (FP32)
│  │   │  │
│  │   │  └─ Dequantize gradients
│  │   │
│  │   └─ Create backward quantizers via RecipeState.create(mode="backward")
│  │       └─ NVFP4Quantizer with backward-specific settings
│  │
│  └─ loss.backward()
│
├─► FP8GlobalStateManager.autocast_exit()
│   ├─ AUTOCAST_DEPTH -= 1
│   └─ If AUTOCAST_DEPTH == 0:
│       └─ reduce_and_update_fp8_tensors()  [for delayed scaling only]
│
└─ Exit autocast context (restore previous state)
```

---

## Integration with te.Linear

### Linear Module Constructor

**File:** [transformer_engine/pytorch/module/linear.py](../../../transformer_engine/pytorch/module/linear.py)

The `te.Linear` module doesn't store quantizers - they are created dynamically based on the active autocast context and recipe.

### Forward Pass Integration

**Key Points:**

1. **Recipe Detection:** During forward pass, the module queries `FP8GlobalStateManager.is_fp8_enabled()`

2. **Quantizer Creation:** If FP8 is enabled:
   ```python
   recipe = FP8GlobalStateManager.get_fp8_recipe()  # Get active recipe
   recipe_state = RecipeState.create(recipe, mode="forward", num_quantizers=3)
   quantizers = recipe_state.make_quantizers()  # Returns list of quantizers
   ```

3. **Quantization Application:**
   ```python
   # Inside _Linear.forward()
   if fp8:
       # Quantize input
       input_quantizer = quantizers[0]
       quantized_input = input_quantizer(input)
       
       # Quantize weight
       weight_quantizer = quantizers[1]
       quantized_weight = weight_quantizer(weight)
       
       # Perform FP4 GEMM (for NVFP4)
       output = tex.general_gemm(
           quantized_input,
           quantized_weight,
           # ... other parameters
       )
   ```

### Backward Pass Integration

During backward:
1. Module creates backward quantizers: `RecipeState.create(recipe, mode="backward")`
2. Quantizes grad_output before computing gradients
3. Applies stochastic rounding (for NVFP4 gradients)
4. Computes grad_input and grad_weight as FP4 GEMMs

---

## Test Case Walkthrough

### Example: NVFP4 Module Test

**File:** [tests/pytorch/nvfp4/test_nvfp4_module_exact.py](../../../../tests/pytorch/nvfp4/test_nvfp4_module_exact.py)

#### Test Setup

```python
def check_nvfp4_module_versus_reference(
    module_class,
    in_features: int,
    out_features: int,
    bias: bool,
    x_dtype: torch.dtype,
    num_steps: int = 1,
    with_rht: bool = False,
    with_2d_quantization: bool = False,
):
    """Compare native NVFP4 module against reference implementation."""
    device = "cuda"
    batch_size = 32
    seq_len = 128
    
    # Step 1: Create recipe
    nvfp4_recipe = recipe.NVFP4BlockScaling()
    if with_rht:
        nvfp4_recipe.fp4_quant_fwd_inp = recipe.QParams(random_hadamard_transform=True)
        nvfp4_recipe.fp4_quant_bwd_grad = recipe.QParams(random_hadamard_transform=True)
    if with_2d_quantization:
        nvfp4_recipe.fp4_quant_fwd_weight = recipe.QParams(fp4_2d_quantization=True)
    
    # Step 2: Create module
    module = module_class(in_features, out_features, bias=bias, dtype=x_dtype, device=device)
    
    # Step 3: Create input
    x = torch.randn(batch_size, seq_len, in_features, dtype=x_dtype, device=device)
    
    # Step 4: Forward pass with autocast
    with te.autocast(enabled=True, recipe=nvfp4_recipe):
        output = module(x)
    
    # Step 5: Backward pass
    loss = output.sum()
    loss.backward()
```

#### Data Flow During Test

1. **Setup Phase:**
   - Recipe created with NVFP4BlockScaling configuration
   - Module parameters initialized in BF16 (or other dtype)

2. **Forward Pass:**
   - `autocast(enabled=True, recipe=nvfp4_recipe)` activates
   - `FP8GlobalStateManager.autocast_enter()` sets global state
   - `te.Linear.forward()` queries FP8 state:
     ```python
     if FP8GlobalStateManager.is_fp8_enabled():
         recipe = FP8GlobalStateManager.get_fp8_recipe()  # Returns NVFP4BlockScaling
     ```
   - Quantizers created:
     ```python
     recipe_state = RecipeState.create(recipe, mode="forward", num_quantizers=3)
     # Returns NVFP4BlockScalingRecipeState with 3 quantizers:
     # [0] = NVFP4Quantizer(input, with_rht=True)
     # [1] = NVFP4Quantizer(weight, fp4_2d_quantization=True)
     # [2] = NVFP4Quantizer(output)
     ```
   - Input quantization:
     - Apply RHT (random Hadamard transform) to smooth distribution
     - Compute amax per block (16 values)
     - Compute E4M3 block scales
     - Compute FP32 global scale
     - Quantize to FP4 E2M1
   
   - Weight quantization:
     - Apply 2D quantization (16x16 tiles)
     - Compute amax per tile
     - Compute E4M3 scales per tile
     - Compute FP32 global scale
     - Quantize to FP4 E2M1
   
   - GEMM computation:
     ```
     FP4(quantized_input) @ FP4(quantized_weight) 
     → FP32(accumulation in tensor core)
     → Dequantized using scales
     ```

3. **Backward Pass:**
   - `_Linear.backward()` called
   - Backward quantizers created:
     ```python
     recipe_state = RecipeState.create(recipe, mode="backward", num_quantizers=2)
     # Returns: [grad_output_quantizer, grad_input_quantizer]
     ```
   - grad_output quantization:
     - Apply RHT
     - Apply stochastic rounding (probabilistic rounding to nearest FP4 value)
     - Quantize to FP4
   
   - Gradient GEMMs:
     ```
     dgrad: grad_output (FP4) @ weight.T (FP4) → grad_input (FP32)
     wgrad: input.T (FP4) @ grad_output (FP4) → grad_weight (FP32)
     ```

4. **Exit Phase:**
   - `autocast_exit()` called
   - `FP8GlobalStateManager` state restored
   - Amax reduction (if delayed scaling; skipped for NVFP4)

---

## Support Checks

### Recipe Support Validation

**File:** [transformer_engine/pytorch/quantization.py](../../../../transformer_engine/pytorch/quantization.py#L89-L99)

```python
def check_recipe_support(recipe: Recipe) -> None:
    """Check if the given recipe is supported."""
    recipe_supported = True
    unsupported_reason = ""
    if isinstance(recipe, (DelayedScaling, Float8CurrentScaling)):
        recipe_supported, unsupported_reason = check_fp8_support()
    elif isinstance(recipe, Float8BlockScaling):
        recipe_supported, unsupported_reason = check_fp8_block_scaling_support()
    elif isinstance(recipe, MXFP8BlockScaling):
        recipe_supported, unsupported_reason = check_mxfp8_support()
    elif isinstance(recipe, NVFP4BlockScaling):  # ◄─ NVFP4 check
        # This is checked in autocast_enter, but included here for completeness
        recipe_supported, unsupported_reason = check_nvfp4_support()
    assert recipe_supported, unsupported_reason
```

### NVFP4 Availability Check

**File:** [transformer_engine/pytorch/quantization.py](../../../../transformer_engine/pytorch/quantization.py#L71-L75)

```python
@functools.lru_cache(maxsize=None)
def check_nvfp4_support() -> Tuple[bool, str]:
    """Return if nvfp4 support is available"""
    if get_device_compute_capability() >= (10, 0):  # blackwell and above
        return True, ""
    return False, "Device compute capability 10.0 or higher required for NVFP4 execution."
```

---

## Key Design Patterns

### 1. Context Manager for State Management

The `autocast` context manager uses Python's `contextmanager` decorator to:
- Save previous state on entry
- Activate new quantization state
- Ensure state restoration on exit (even with exceptions)

### 2. Global State Manager Pattern

`FP8GlobalStateManager` uses class variables and class methods to:
- Track global quantization context across the entire application
- Manage nested autocast contexts via `AUTOCAST_DEPTH`
- Provide query methods for modules to detect active quantization

### 3. Recipe State Factory Pattern

`RecipeState.create()` factory method:
- Dispatches to appropriate recipe state class based on recipe type
- Abstracts quantizer creation from modules
- Enables easy addition of new recipes

### 4. Quantizer as Callable

Quantizers are callable objects:
```python
quantizer = NVFP4Quantizer(...)
quantized_tensor = quantizer(input_tensor)  # __call__ method
```

This enables transparent quantization in module code.

---

## Summary

The `te.autocast()` implementation provides a clean, composable interface for enabling low-precision quantization in TransformerEngine:

1. **Entry Point:** User calls `with te.autocast(recipe=NVFP4BlockScaling()):`
2. **State Management:** Global state updated via `FP8GlobalStateManager`
3. **Recipe Resolution:** Active recipe queried by modules during forward/backward
4. **Quantizer Creation:** `RecipeState` factory creates appropriate quantizers
5. **Computation:** Quantizers applied to inputs/weights/gradients
6. **Kernel Execution:** FP4 GEMMs executed on Blackwell tensor cores
7. **Cleanup:** State restored on context exit

This design supports multiple recipes (FP8, MXFP8, NVFP4, Float8BlockScaling, etc.) with a unified interface and enables nested autocast contexts for advanced use cases.

