# Frame-by-Frame Execution Trace: `te.autocast` with NVFP4

This document provides a detailed, line-by-line trace of the execution path when using `te.autocast` with the NVFP4BlockScaling recipe, from the Python API down to CUDA kernels and back.

## Table of Contents
- [Example Code](#example-code)
- [Execution Overview](#execution-overview)
- [Frame-by-Frame Trace](#frame-by-frame-trace)
  - [Phase 1: Setup & Recipe Creation](#phase-1-setup--recipe-creation)
  - [Phase 2: Autocast Entry](#phase-2-autocast-entry)
  - [Phase 3: Forward Pass](#phase-3-forward-pass)
  - [Phase 4: Backward Pass](#phase-4-backward-pass)
  - [Phase 5: Autocast Exit](#phase-5-autocast-exit)
- [Data Flow Diagrams](#data-flow-diagrams)
- [Key Operations Deep Dive](#key-operations-deep-dive)

---

## Example Code

```python
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import NVFP4BlockScaling

# Setup
recipe = NVFP4BlockScaling()
linear1 = te.Linear(768, 768).bfloat16()  # FP4 layer
inp = torch.randn(1024, 768, device='cuda', dtype=torch.bfloat16)

# Forward with nested autocast
with te.autocast(recipe=recipe):
    out = linear1(inp)

loss = out.mean()
loss.backward()
```

---

## Execution Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    EXECUTION PHASES                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. Recipe Creation (NVFP4BlockScaling)                      │
│     ├─ Parse environment variables                           │
│     ├─ Create QParams for forward/backward                   │
│     └─ Store quantization configuration                      │
│                                                               │
│  2. Autocast Entry (Context Manager __enter__)               │
│     ├─ Save current FP8 state                                │
│     ├─ Set global recipe reference                           │
│     ├─ Increment autocast depth counter                      │
│     └─ Validate NVFP4 support                                │
│                                                               │
│  3. Forward Pass (linear1(inp))                              │
│     ├─ Query FP8GlobalStateManager for recipe               │
│     ├─ Create NVFP4 quantizers (input, weight, output)       │
│     ├─ Quantize input: BF16 → FP4                            │
│     │   ├─ Apply Random Hadamard Transform (RHT)            │
│     │   ├─ Compute amax per block                            │
│     │   ├─ Quantize to E2M1 (FP4) format                     │
│     │   └─ Store with E4M3 scales + FP32 amax                │
│     ├─ Quantize weight: BF16 → FP4                           │
│     │   ├─ Apply 2D block quantization (16x16)              │
│     │   ├─ Compute scales per block                          │
│     │   └─ Store in compressed format                        │
│     ├─ GEMM: FP4 × FP4 → BF16                                │
│     │   ├─ Call tex.generic_gemm                             │
│     │   ├─ Dispatch to cuBLAS FP4 GEMM kernel               │
│     │   └─ Accumulate in FP32, convert to BF16               │
│     └─ Return output                                         │
│                                                               │
│  4. Backward Pass (loss.backward())                          │
│     ├─ Compute grad_output (from loss)                       │
│     ├─ Quantize grad_output: BF16 → FP4                      │
│     │   ├─ Apply RHT                                         │
│     │   ├─ Apply stochastic rounding                         │
│     │   └─ Store quantized gradient                          │
│     ├─ Compute grad_input (DGRAD GEMM)                       │
│     │   └─ FP4 grad_output × FP4 weight^T → BF16            │
│     ├─ Compute grad_weight (WGRAD GEMM)                      │
│     │   └─ FP4 input^T × FP4 grad_output → BF16             │
│     └─ Accumulate gradients                                  │
│                                                               │
│  5. Autocast Exit (Context Manager __exit__)                 │
│     ├─ Restore saved FP8 state                               │
│     ├─ Decrement autocast depth counter                      │
│     └─ Clear recipe reference (if depth == 0)                │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Frame-by-Frame Trace

### Phase 1: Setup & Recipe Creation

#### Frame 1: Import TransformerEngine
```python
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import NVFP4BlockScaling
```

**Location:** [transformer_engine/pytorch/\_\_init\_\_.py:51](../transformer_engine/pytorch/__init__.py#L51)

**What happens:**
- Imports expose `autocast` function from `quantization.py`
- NVFP4BlockScaling class is loaded from recipe module

---

#### Frame 2: Create NVFP4BlockScaling Recipe
```python
recipe = NVFP4BlockScaling()
```

**Location:** [transformer_engine/common/recipe/\_\_init\_\_.py:386-481](../transformer_engine/common/recipe/__init__.py#L386-L481)

**Execution trace:**

1. **Class instantiation** (line 387)
   ```python
   @dataclass()
   class NVFP4BlockScaling(Recipe):
   ```

2. **Parse environment variables** (lines 434-438)
   ```python
   disable_rht: bool = os.getenv("NVTE_NVFP4_DISABLE_RHT", "0") == "1"
   disable_stochastic_rounding: bool = (
       os.getenv("NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING", "0") == "1"
   )
   disable_2d_quantization: bool = os.getenv("NVTE_NVFP4_DISABLE_2D_QUANTIZATION", "0") == "1"
   ```

3. **Set data formats** (lines 440-441)
   ```python
   fp4_format: Format = Format.E2M1  # 4-bit float: 2-bit exp, 1-bit mantissa
   fp8_format: Format = Format.E4M3  # 8-bit float for scales: 4-bit exp, 3-bit mantissa
   ```

4. **Run post-init validation** (lines 447-449)
   ```python
   def __post_init__(self) -> None:
       assert self.fp4_format == Format.E2M1, "Only E2M1 is supported for NVFP4 scaling"
       assert self.fp8_format == Format.E4M3, "Only E4M3 is supported for NVFP4 scaling"
   ```

5. **Create quantization parameters for forward pass inputs** (lines 454-458)
   ```python
   self.fp4_quant_fwd_inp = QParams(
       random_hadamard_transform=not self.disable_rht,      # Apply RHT: TRUE
       stochastic_rounding=False,                            # No SR for inputs: FALSE
       fp4_2d_quantization=False,                            # 1D for inputs: FALSE
   )
   ```
   - **RHT enabled**: Smooths outliers in activation distributions
   - **No stochastic rounding**: Not needed for forward pass
   - **1D quantization**: 16-element blocks

6. **Create quantization parameters for forward pass weights** (lines 459-463)
   ```python
   self.fp4_quant_fwd_weight = QParams(
       random_hadamard_transform=False,                      # No RHT for weights: FALSE
       stochastic_rounding=False,                            # No SR for weights: FALSE
       fp4_2d_quantization=not self.disable_2d_quantization, # 2D for weights: TRUE
   )
   ```
   - **No RHT**: Weights are static, don't benefit from RHT
   - **2D quantization**: 16×16 element blocks for better accuracy

7. **Create quantization parameters for backward pass gradients** (lines 464-468)
   ```python
   self.fp4_quant_bwd_grad = QParams(
       random_hadamard_transform=not self.disable_rht,                # Apply RHT: TRUE
       stochastic_rounding=not self.disable_stochastic_rounding,      # Apply SR: TRUE
       fp4_2d_quantization=False,                                     # 1D for grads: FALSE
   )
   ```
   - **RHT enabled**: Smooths gradient outliers
   - **Stochastic rounding**: Prevents bias in gradient quantization
   - **1D quantization**: Simpler for dynamic gradients

**Result:** Recipe object configured with quantization strategies for all tensor types

---

#### Frame 3: Create Linear Layer
```python
linear1 = te.Linear(768, 768).bfloat16()
```

**Location:** [transformer_engine/pytorch/module/linear.py:1-74](../transformer_engine/pytorch/module/linear.py#L1-L74)

**What happens:**
- Creates a Linear layer: 768 input features → 768 output features
- Initializes weight tensor: `[768, 768]` in BF16
- Initializes bias tensor (if `bias=True`)
- Layer is ready but quantizers not yet created (waiting for autocast)

---

#### Frame 4: Create Input Tensor
```python
inp = torch.randn(1024, 768, device='cuda', dtype=torch.bfloat16)
```

**Result:** Tensor shape `[1024, 768]` in BF16 on GPU

---

### Phase 2: Autocast Entry

#### Frame 5: Enter Autocast Context
```python
with te.autocast(recipe=recipe):
```

**Location:** [transformer_engine/pytorch/quantization.py:789-852](../transformer_engine/pytorch/quantization.py#L789-L852)

**Execution trace:**

1. **Function call** (line 790-796)
   ```python
   @contextmanager
   def autocast(
       enabled: bool = True,         # Quantization enabled
       calibrating: bool = False,    # Not calibrating
       recipe: Optional["Recipe"] = None,  # Our NVFP4BlockScaling recipe
       amax_reduction_group: Optional["dist_group_type"] = None,  # No distributed
       _graph: bool = False,
   ) -> None:
   ```

2. **Check recipe support** (line 836)
   ```python
   if enabled:
       check_recipe_support(recipe)
   ```
   - Validates NVFP4 support is available on current GPU

3. **Save current state** (line 839)
   ```python
   fp8_state = FP8GlobalStateManager.get_autocast_state()
   ```
   **Location:** [transformer_engine/pytorch/quantization.py:224-251](../transformer_engine/pytorch/quantization.py#L224-L251)

   Returns dict with:
   ```python
   {
       'FP8_ENABLED': False,           # Not currently in autocast
       'FP8_CALIBRATION': False,
       'FP8_RECIPE': None,
       'FP8_DISTRIBUTED_GROUP': None,
       'AUTOCAST_DEPTH': 0,
       # ... other state variables
   }
   ```

4. **Enter autocast state** (lines 841-847)
   ```python
   FP8GlobalStateManager.autocast_enter(
       enabled=True,
       calibrating=False,
       fp8_recipe=recipe,
       fp8_group=None,
       _graph=False,
   )
   ```
   **Location:** [transformer_engine/pytorch/quantization.py:552-579](../transformer_engine/pytorch/quantization.py#L552-L579)

5. **Inside autocast_enter:** (lines 563-575)
   ```python
   # Get default recipe if none provided (we provided one, so skip)
   fp8_recipe = get_default_fp8_recipe() if fp8_recipe is None else fp8_recipe

   # Create unique key for this autocast context
   autocast_key = cls.get_unique_autocast_key(fp8_recipe, fp8_group)
   # Key: "recipe_type=NVFP4BlockScaling, ...:hash(None)"

   # Store recipe reference for later retrieval
   cls.autocast_arguments[autocast_key] = (fp8_recipe, fp8_group)

   # Set global state
   cls.FP8_ENABLED = True                        # Enable quantization
   cls.FP8_CALIBRATION = False                   # Not calibrating
   cls.FP8_RECIPE = fp8_recipe                   # Store recipe reference
   cls.FP8_DISTRIBUTED_GROUP = None              # No distributed group
   cls.FP8_GRAPH_CAPTURING = False               # Not capturing CUDA graph

   # Track nesting depth
   if cls.AUTOCAST_DEPTH == 0:
       cls.IS_FIRST_FP8_MODULE = True
   cls.AUTOCAST_DEPTH += 1                       # Now depth = 1
   ```

6. **Validate FP8 availability** (lines 577-579)
   ```python
   if enabled:
       fp8_available, reason_for_no_fp8 = cls.is_fp8_available()
       assert fp8_available, reason_for_no_fp8
   ```
   - Checks GPU architecture (requires SM 8.9+ for NVFP4)
   - Validates CUDA/cuBLAS versions

**Global State After Entry:**
```python
FP8GlobalStateManager.FP8_ENABLED = True
FP8GlobalStateManager.FP8_RECIPE = NVFP4BlockScaling(...)
FP8GlobalStateManager.AUTOCAST_DEPTH = 1
FP8GlobalStateManager.IS_FIRST_FP8_MODULE = True
```

---

### Phase 3: Forward Pass

#### Frame 6: Call Linear Forward
```python
out = linear1(inp)
```

**Location:** [transformer_engine/pytorch/module/linear.py:1300+](../transformer_engine/pytorch/module/linear.py#L1300)

**Forward method calls `_Linear.forward` (autograd function):**

**Location:** [transformer_engine/pytorch/module/linear.py:77-121](../transformer_engine/pytorch/module/linear.py#L77-L121)

---

#### Frame 7: Create Quantizers

**Location:** Linear module's `forward()` method queries state manager

1. **Check if FP8 is enabled** (module's forward method)
   ```python
   fp8_enabled = FP8GlobalStateManager.is_fp8_enabled()
   # Returns: True

   recipe = FP8GlobalStateManager.get_fp8_recipe()
   # Returns: NVFP4BlockScaling instance
   ```

2. **Get quantizers from recipe state**
   The Linear module uses its `set_meta_tensor` method to create quantizers:

   **Location:** [transformer_engine/pytorch/module/base.py](../transformer_engine/pytorch/module/base.py)

   This calls into the recipe state factory:

   **Location:** [transformer_engine/pytorch/quantization.py:967-1026](../transformer_engine/pytorch/quantization.py#L967-L1026)

   ```python
   def _make_recipe_state(
       recipe: Recipe,
       *,
       mode: str,  # "forward"
       num_quantizers: int = 1,
       device: Optional[torch.device] = None,
   ) -> RecipeState:
       # ... dispatch logic ...
       if recipe.nvfp4():  # Check if recipe is NVFP4BlockScaling
           return NVFP4BlockScalingRecipeState(
               recipe,
               mode=mode,
               num_quantizers=num_quantizers,
               device=device,
           )
   ```

3. **NVFP4BlockScalingRecipeState creation**

   **Location:** [transformer_engine/pytorch/quantization.py:1270-1343](../transformer_engine/pytorch/quantization.py#L1270-L1343)

   ```python
   class NVFP4BlockScalingRecipeState(RecipeState):
       def __init__(
           self,
           recipe: NVFP4BlockScaling,
           *,
           mode: str,  # "forward"
           num_quantizers: int = 1,
           device: Optional[torch.device] = None,
       ) -> None:
           self.recipe = recipe
           self.mode = mode
           self.num_quantizers = num_quantizers  # 3 for Linear (input, weight, output)
           self.dtype = get_fp4_te_dtype(recipe)  # Returns TE_DType.kFloat4E2M1

           if device is None:
               device = torch.device("cuda")
   ```

4. **Create quantizers** (lines 1298-1327)
   ```python
   def make_quantizers(self) -> list:
       from .tensor.nvfp4_tensor import NVFP4Quantizer

       if self.mode == "forward":
           def _make_quantizer(idx: int) -> NVFP4Quantizer:
               # For Linear: idx=0 (input), idx=1 (weight), idx=2 (output)
               qparams = (
                   self.recipe.fp4_quant_fwd_weight  # idx=1 (weight)
                   if idx % 3 == 1
                   else self.recipe.fp4_quant_fwd_inp  # idx=0,2 (input, output)
               )
               return NVFP4Quantizer(
                   fp4_dtype=self.dtype,              # kFloat4E2M1
                   rowwise=True,                       # Create rowwise data
                   columnwise=True,                    # Create columnwise data
                   with_rht=qparams.random_hadamard_transform,
                   with_post_rht_amax=qparams.random_hadamard_transform,
                   with_2d_quantization=qparams.fp4_2d_quantization,
                   stochastic_rounding=qparams.stochastic_rounding,
               )

           return [_make_quantizer(idx) for idx in range(self.num_quantizers)]
   ```

5. **Result: Three quantizers created**

   **Input Quantizer (idx=0):**
   ```python
   NVFP4Quantizer(
       fp4_dtype=kFloat4E2M1,
       rowwise=True,
       columnwise=True,
       with_rht=True,                    # Apply Random Hadamard Transform
       with_post_rht_amax=True,          # Track amax after RHT
       with_2d_quantization=False,       # Use 1D block scaling
       stochastic_rounding=False,        # No stochastic rounding for inputs
   )
   ```

   **Weight Quantizer (idx=1):**
   ```python
   NVFP4Quantizer(
       fp4_dtype=kFloat4E2M1,
       rowwise=True,
       columnwise=True,
       with_rht=False,                   # No RHT for weights
       with_post_rht_amax=False,
       with_2d_quantization=True,        # Use 2D block scaling (16x16)
       stochastic_rounding=False,        # No stochastic rounding for weights
   )
   ```

   **Output Quantizer (idx=2):**
   ```python
   NVFP4Quantizer(
       fp4_dtype=kFloat4E2M1,
       rowwise=True,
       columnwise=True,
       with_rht=True,                    # Apply RHT
       with_post_rht_amax=True,
       with_2d_quantization=False,       # Use 1D block scaling
       stochastic_rounding=False,
   )
   ```

---

#### Frame 8: Prepare Input Tensor

**Location:** [transformer_engine/pytorch/module/linear.py:159-235](../transformer_engine/pytorch/module/linear.py#L159-L235)

```python
# ------------------------------------------------------
# Prepare input tensor
# ------------------------------------------------------

# Input shape: [1024, 768], dtype: bfloat16
inputmat = inp

# Check if FP8/FP4 quantization is needed
if fp8:  # True
    assert_dim_for_fp8_exec(inputmat, weight)  # Verify dims divisible by 16

    # Since we're not doing all-gather (no tensor parallelism), go to else branch
    if with_input_all_gather_nccl or ub_overlap_ag_fprop:
        # ... skipped in our case
        pass
    else:  # Our path (lines 217-231)
        if fp8 or debug:
            if isinstance(inputmat, QuantizedTensorStorage):
                # Already quantized - not our case
                inputmat.update_usage(rowwise_usage=True)
            else:
                # Quantize input now
                if input_quantizer is None:
                    raise ValueError("Missing quantizer for input tensor")

                # Configure quantizer usage
                input_quantizer.set_usage(
                    rowwise=True,                    # Create rowwise data
                    columnwise=backward_needs_input  # Create columnwise if backward needed
                )
                # backward_needs_input = True (requires_grad=True)

                # QUANTIZE INPUT
                inputmat = input_quantizer(inputmat)
                own_quantized_input = True

        inputmat_total = inputmat  # No all-gather needed
```

---

#### Frame 9: Quantize Input (BF16 → FP4)

**Location:** [transformer_engine/pytorch/tensor/nvfp4_tensor.py:112-180](../transformer_engine/pytorch/tensor/nvfp4_tensor.py#L112-L180)

**Calling `input_quantizer(inputmat)`:**

1. **NVFP4Quantizer.__call__** (inherited from Quantizer base class)
   Calls `quantize_impl`:

   ```python
   def quantize_impl(self, tensor: torch.Tensor) -> QuantizedTensor:
       """Quantize tensor implementation"""
       return tex.quantize(tensor, self)  # Call C++ extension
   ```
   **Location:** Line 178-180

2. **C++ Extension Call: `tex.quantize`**

   **Python → C++ Binding:** This calls into the PyTorch C++ extension

   **Location:** `transformer_engine/pytorch/csrc/extensions/quantization.cpp`

   The quantize function:
   ```cpp
   torch::Tensor quantize(
       torch::Tensor input,           // [1024, 768] BF16
       NVFP4Quantizer quantizer       // Quantizer config
   ) {
       // Allocate output storage
       NVFP4Tensor output = quantizer.make_empty(
           input.sizes(),
           dtype=input.dtype,
           device=input.device,
           requires_grad=input.requires_grad
       );

       // Dispatch to quantization kernel
       return quantizer.update_quantized(input, output);
   }
   ```

3. **Allocate FP4 Storage** (make_empty)

   **Location:** [transformer_engine/pytorch/tensor/nvfp4_tensor.py:261-325](../transformer_engine/pytorch/tensor/nvfp4_tensor.py#L261-L325)

   ```python
   def make_empty(
       self,
       shape: Iterable[int],  # [1024, 768]
       *,
       dtype: torch.dtype = torch.float32,
       device: Optional[torch.device] = None,
       requires_grad: bool = False,
   ) -> NVFP4Tensor:

       # Validate shape divisibility by 16
       assert shape[-1] % 16 == 0  # 768 % 16 = 0 ✓
       flat_first_dim = math.prod(shape[:-1])  # 1024
       assert flat_first_dim % 16 == 0  # 1024 % 16 = 0 ✓

       # Allocate rowwise FP4 data
       data = None
       scale_inv = None
       amax_rowwise = None
       if self.rowwise_usage:  # True
           # FP4 data: 2 values per byte, so divide by 2
           data = torch.empty(
               self.convert_shape_for_fp4(shape),  # [1024, 384]
               dtype=torch.uint8,
               device='cuda'
           )
           # Scale shape for rowwise
           scale_shape = self.get_scale_shape(shape, columnwise=False)
           # Returns: (1024, 48) padded to (1024, 48)
           # Explanation: 768/16 = 48 blocks per row
           scale_inv = torch.empty(scale_shape, dtype=torch.uint8, device='cuda')

           # Per-tensor amax in FP32
           amax_rowwise = torch.zeros(1, dtype=torch.float32, device='cuda')

       # Allocate columnwise FP4 data (for backward pass)
       columnwise_data = None
       columnwise_scale_inv = None
       amax_columnwise = None
       if self.columnwise_usage:  # True (backward needs it)
           columnwise_shape = self.get_columnwise_shape(shape)
           # Returns: [768, 1024] (transposed)
           columnwise_data = torch.empty(
               self.convert_shape_for_fp4(columnwise_shape),  # [768, 512]
               dtype=torch.uint8,
               device='cuda'
           )
           scale_shape = self.get_scale_shape(shape, columnwise=True)
           # Returns: (768, 64) padded
           # Explanation: 1024/16 = 64 blocks per column
           columnwise_scale_inv = torch.empty(scale_shape, dtype=torch.uint8, device='cuda')
           amax_columnwise = torch.zeros(1, dtype=torch.float32, device='cuda')

       # Create NVFP4Tensor storage object
       return NVFP4Tensor(
           data=data,
           scale_inv=scale_inv,
           amax=amax_rowwise,
           columnwise_data=columnwise_data,
           columnwise_scale_inv=columnwise_scale_inv,
           columnwise_amax=amax_columnwise,
           shape=shape,
           dtype=dtype,
           device=device,
           requires_grad=requires_grad,
       )
   ```

4. **Launch Quantization Kernel** (update_quantized)

   **Location:** [transformer_engine/pytorch/tensor/nvfp4_tensor.py:157-176](../transformer_engine/pytorch/tensor/nvfp4_tensor.py#L157-L176)

   ```python
   def update_quantized(
       self,
       src: torch.Tensor,        # Input BF16 [1024, 768]
       dst: QuantizedTensor,     # Output NVFP4Tensor
       *,
       noop_flag: Optional[torch.Tensor] = None,
   ) -> QuantizedTensor:

       assert isinstance(dst, NVFP4Tensor)

       # Ensure input is contiguous and on correct device
       if not devices_match(src.device, dst.device):
           src = src.to(device=dst.device)
       if not src.is_contiguous():
           src = src.contiguous()

       # Launch cast kernel (C++ call)
       tex.quantize(src, self, dst, noop_flag)

       return dst
   ```

5. **C++ Quantization Kernel Dispatch**

   **Location:** `transformer_engine/pytorch/csrc/extensions/quantization.cpp`

   ```cpp
   void quantize(
       torch::Tensor src,         // [1024, 768] BF16
       NVFP4Quantizer quantizer,  // Config
       NVFP4Tensor dst,           // Output storage
       torch::Tensor noop_flag    // Optional skip flag
   ) {
       // Get quantizer parameters
       bool with_rht = quantizer.with_rht;  // true
       bool with_2d_quantization = quantizer.with_2d_quantization;  // false
       int rht_mask = quantizer.rht_matrix_random_sign_mask_t;

       // Dispatch to appropriate kernel based on config
       if (with_rht && !with_2d_quantization) {
           // Our path: 1D quantization with RHT
           nvfp4_quantize_rht_1d(
               src.data_ptr<at::BFloat16>(),
               dst.data.data_ptr<uint8_t>(),
               dst.scale_inv.data_ptr<uint8_t>(),
               dst.amax.data_ptr<float>(),
               dst.columnwise_data.data_ptr<uint8_t>(),
               dst.columnwise_scale_inv.data_ptr<uint8_t>(),
               dst.columnwise_amax.data_ptr<float>(),
               src.size(0),  // M = 1024
               src.size(1),  // K = 768
               rht_mask,
               at::cuda::getCurrentCUDAStream()
           );
       }
       // ... other dispatch paths for different configs
   }
   ```

6. **CUDA Kernel: nvfp4_quantize_rht_1d**

   **Location:** `transformer_engine/common/transpose/quantize_transpose_vector_blockwise_fp4.cu`

   ```cuda
   __global__ void nvfp4_quantize_rht_1d_kernel(
       const __nv_bfloat16* input,     // [1024, 768]
       uint8_t* output,                 // [1024, 384] (2 FP4 per byte)
       uint8_t* scale_inv,              // [1024, 48] E4M3 scales
       float* amax,                     // [1] global amax
       uint8_t* columnwise_output,      // [768, 512]
       uint8_t* columnwise_scale_inv,   // [768, 64]
       float* columnwise_amax,          // [1]
       int M,                           // 1024
       int K,                           // 768
       int rht_mask                     // Random sign mask
   ) {
       // Thread/block indices
       int tid = threadIdx.x;
       int block_id = blockIdx.x;

       // Each block processes 16 consecutive elements (blocksize)
       int elements_per_block = 16;
       int block_start = block_id * elements_per_block;

       // Load 16 elements into shared memory
       __shared__ float shared_data[16];
       if (block_start + tid < M * K) {
           shared_data[tid] = __bfloat162float(input[block_start + tid]);
       }
       __syncthreads();

       // STEP 1: Apply Random Hadamard Transform (RHT)
       // This smooths outliers in the distribution

       // Apply random signs based on rht_mask
       if ((rht_mask >> tid) & 1) {
           shared_data[tid] = -shared_data[tid];
       }
       __syncthreads();

       // Apply Hadamard matrix multiplication (16x16)
       // H16 = kronecker product of H2 x H2 x H2 x H2
       // Can be computed efficiently with butterfly pattern

       float hadamard_scale = 1.0f / sqrtf(16.0f);  // 1/4

       // Butterfly pattern for Hadamard transform
       for (int stride = 1; stride < 16; stride *= 2) {
           int pair_idx = tid ^ stride;
           float a = shared_data[tid];
           float b = shared_data[pair_idx];
           __syncthreads();
           if (tid < pair_idx) {
               shared_data[tid] = a + b;
               shared_data[pair_idx] = a - b;
           }
           __syncthreads();
       }

       // Scale by 1/sqrt(16)
       shared_data[tid] *= hadamard_scale;
       __syncthreads();

       // STEP 2: Compute block amax (absolute maximum)
       __shared__ float block_amax;
       if (tid == 0) block_amax = 0.0f;
       __syncthreads();

       float abs_val = fabsf(shared_data[tid]);
       atomicMax_float(&block_amax, abs_val);
       __syncthreads();

       // STEP 3: Compute scale from amax
       // E4M3 max value = 448.0
       // FP4 (E2M1) max value = 6.0
       const float fp4_max = 6.0f;
       float scale = block_amax / fp4_max;

       // Quantize scale to E4M3 (8-bit) for storage
       uint8_t scale_e4m3 = float_to_e4m3(scale);

       // Store scale (only thread 0)
       if (tid == 0) {
           int scale_idx = block_id;
           scale_inv[scale_idx] = scale_e4m3;
       }

       // STEP 4: Quantize to FP4 (E2M1)
       float scaled_val = shared_data[tid] / scale;
       uint8_t fp4_val = float_to_e2m1(scaled_val);

       // STEP 5: Pack two FP4 values into one byte
       // Even threads pack low nibble, odd threads pack high nibble
       __shared__ uint8_t packed_output[8];  // 16 FP4 = 8 bytes

       if (tid % 2 == 0) {
           // Low nibble
           packed_output[tid / 2] = fp4_val & 0x0F;
       } else {
           // High nibble
           atomicOr(&packed_output[tid / 2], (fp4_val & 0x0F) << 4);
       }
       __syncthreads();

       // STEP 6: Write packed output
       if (tid < 8) {
           output[block_start / 2 + tid] = packed_output[tid];
       }

       // STEP 7: Update global amax
       if (tid == 0) {
           atomicMax_float(amax, block_amax);
       }

       // STEP 8: Create columnwise version (transpose)
       // Similar process but with transposed indices
       // ... (code for columnwise quantization)
   }
   ```

7. **FP4 Data Format (E2M1)**

   **4-bit floating point:**
   - 1 sign bit
   - 2 exponent bits
   - 1 mantissa bit

   **Representable values:**
   ```
   E2M1 format can represent:
   - 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6
   - Plus denormals and special values
   - Range: approximately ±6.0
   ```

   **Storage:**
   - 2 FP4 values packed per uint8 byte
   - Low nibble (bits 0-3): first value
   - High nibble (bits 4-7): second value

8. **Scale Format (E4M3)**

   **8-bit floating point:**
   - 1 sign bit
   - 4 exponent bits
   - 3 mantissa bits

   **Range:** approximately 2^-6 to 448

   **Purpose:** Stores per-block scale factors for dequantization

**Result after input quantization:**

```
Input Tensor (Quantized):
├─ Rowwise data: [1024, 384] uint8 (FP4 packed)
├─ Rowwise scales: [1024, 48] uint8 (E4M3)
├─ Rowwise amax: [1] float32
├─ Columnwise data: [768, 512] uint8 (FP4 packed, transposed)
├─ Columnwise scales: [768, 64] uint8 (E4M3)
└─ Columnwise amax: [1] float32

Memory savings:
- Original BF16: 1024 × 768 × 2 bytes = 1,572,864 bytes
- FP4 data: 1024 × 384 + 768 × 512 = 786,432 bytes
- Scales: (1024 × 48) + (768 × 64) = 98,304 bytes
- Total: 884,736 bytes (56% of original)
```

---

#### Frame 10: Prepare Weight Tensor

**Location:** [transformer_engine/pytorch/module/linear.py:237-269](../transformer_engine/pytorch/module/linear.py#L237-L269)

```python
# ------------------------------------------------------
# Prepare weight tensor
# ------------------------------------------------------
weightmat = weight  # [768, 768] BF16

if fp8 or debug:  # True
    # Configure quantizer
    if weight_quantizer is not None:  # True
        columnwise_usage = is_grad_enabled and inp.requires_grad  # True

        # Configure usage
        weight_quantizer.set_usage(rowwise=True, columnwise=columnwise_usage)

    # Get quantized weight
    update_workspace = is_first_microbatch is None or is_first_microbatch  # True
    weightmat = module.get_weight_workspace(
        tensor=weight,
        quantizer=weight_quantizer,
        cache_name=(None if is_first_microbatch is None else "weight"),
        update_workspace=update_workspace,
        skip_update_flag=skip_fp8_weight_update,
        fsdp_group=fsdp_group,
        workspace_dtype=activation_dtype,
    )
    weightmat.update_usage(rowwise_usage=True)
```

---

#### Frame 11: Quantize Weight (BF16 → FP4)

**Weight quantization follows similar path to input but with 2D block scaling:**

**Location:** [transformer_engine/pytorch/tensor/nvfp4_tensor.py:178-180](../transformer_engine/pytorch/tensor/nvfp4_tensor.py#L178-L180)

**Key difference: 2D block quantization**

```cuda
__global__ void nvfp4_quantize_2d_kernel(
    const __nv_bfloat16* input,     // [768, 768]
    uint8_t* output,                 // [768, 384]
    uint8_t* scale_inv,              // [48, 48] E4M3 scales (2D!)
    float* amax,
    int M,                           // 768
    int K                            // 768
) {
    // Each thread block processes a 16x16 tile
    int tile_i = blockIdx.y;
    int tile_j = blockIdx.x;
    int tid = threadIdx.x;

    // Load 16x16 tile into shared memory
    __shared__ float tile[16][16];

    // Load data (each thread loads one element)
    int global_i = tile_i * 16 + tid / 16;
    int global_j = tile_j * 16 + tid % 16;
    tile[tid / 16][tid % 16] = __bfloat162float(input[global_i * K + global_j]);
    __syncthreads();

    // Compute amax for entire 16x16 block
    __shared__ float block_amax;
    if (tid == 0) block_amax = 0.0f;
    __syncthreads();

    float abs_val = fabsf(tile[tid / 16][tid % 16]);
    atomicMax_float(&block_amax, abs_val);
    __syncthreads();

    // Compute single scale for entire 16x16 block
    const float fp4_max = 6.0f;
    float scale = block_amax / fp4_max;
    uint8_t scale_e4m3 = float_to_e4m3(scale);

    // Store scale (only thread 0)
    if (tid == 0) {
        scale_inv[tile_i * (K/16) + tile_j] = scale_e4m3;
    }

    // Quantize element to FP4
    float scaled_val = tile[tid / 16][tid % 16] / scale;
    uint8_t fp4_val = float_to_e2m1(scaled_val);

    // Pack and store
    // ... (packing logic similar to 1D case)
}
```

**2D vs 1D Quantization:**

```
1D Quantization (Inputs/Gradients):
┌─────────────────────┐
│ Block: 16 elements  │  One scale per 16 elements
│ [x₀, x₁, ..., x₁₅] │  Scale: s₀
└─────────────────────┘

2D Quantization (Weights):
┌─────────────────────┐
│ Block: 16×16 tile   │  One scale per 256 elements
│ ┌───┬───┬───┬───┐   │  Better for static weights
│ │   │   │   │   │   │  Reduces scale storage
│ └───┴───┴───┴───┘   │
│       ...           │
└─────────────────────┘
```

**Result after weight quantization:**

```
Weight Tensor (Quantized):
├─ Rowwise data: [768, 384] uint8 (FP4 packed)
├─ Rowwise scales: [48, 48] uint8 (E4M3, 2D blocks!)
├─ Rowwise amax: [1] float32
├─ Columnwise data: [768, 384] uint8
├─ Columnwise scales: [48, 48] uint8 (E4M3, 2D blocks!)
└─ Columnwise amax: [1] float32
```

---

#### Frame 12: Forward GEMM

**Location:** [transformer_engine/pytorch/module/linear.py:304-324](../transformer_engine/pytorch/module/linear.py#L304-L324)

```python
# ------------------------------------------------------
# Forward GEMM: y = x * w^T
# ------------------------------------------------------

# Choose whether to use split accumulator
use_split_accumulator = _2X_ACC_FPROP
if fp8:
    recipe = FP8GlobalStateManager.get_fp8_recipe()
    if hasattr(recipe, "fp8_gemm_fprop"):
        use_split_accumulator = recipe.fp8_gemm_fprop.use_split_accumulator

# Configure output quantizer (if needed)
if output_quantizer is not None:
    output_quantizer.set_usage(rowwise=True, columnwise=False)

nvtx_range_push("transformer_engine._Linear.forward.gemm")

gemm_out, *_, reduce_scatter_out = general_gemm(
    weightmat,           # [768, 768] FP4
    inputmat_total,      # [1024, 768] FP4
    get_workspace(),     # Workspace buffer
    quantization_params=output_quantizer,
    out_dtype=activation_dtype,  # bfloat16
    bias=bias,
    use_split_accumulator=use_split_accumulator,
    ub=None,             # No comm overlap
    ub_type=None,
    extra_output=None,
)

nvtx_range_pop("transformer_engine._Linear.forward.gemm")
```

---

#### Frame 13: GEMM Dispatch

**Location:** [transformer_engine/pytorch/cpp_extensions/gemm.py:35-154](../transformer_engine/pytorch/cpp_extensions/gemm.py#L35-L154)

```python
def general_gemm(
    A: torch.Tensor,         # weightmat [768, 768] FP4
    B: torch.Tensor,         # inputmat [1024, 768] FP4
    workspace: torch.Tensor,
    out_dtype: Optional[torch.dtype] = None,  # bfloat16
    quantization_params: Optional[Quantizer] = None,
    gelu: bool = False,
    gelu_in: torch.Tensor = None,
    alpha: float = 1.0,
    beta: Optional[float] = None,
    accumulate: bool = False,
    layout: str = "TN",      # Transpose A, Normal B
    out: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    use_split_accumulator: bool = False,
    grad: bool = False,
    ub: Union[tex.CommOverlap, tex.CommOverlapP2P] = None,
    ub_type: tex.CommOverlapType = None,
    extra_output: Optional[torch.Tensor] = None,
    bulk_overlap: bool = False,
) -> Iterable[Optional[torch.Tensor]]:

    # Validate layout
    assert layout in ("TN", "NN", "NT")
    transa = layout[0] == "T"  # True (transpose weight)
    transb = layout[1] == "T"  # False (don't transpose input)

    # Prepare arguments
    args = (
        A,                    # Weight [768, 768] FP4
        transa,               # True
        B,                    # Input [1024, 768] FP4
        transb,               # False
        out,                  # None (will be allocated)
        quantization_params,  # output_quantizer
        TE_DType[out_dtype],  # TE_DType.kBFloat16
        bias,                 # Optional bias
        TE_DType[torch.bfloat16],  # bias_dtype
        gelu,                 # False
        gelu_in,              # None
        grad,                 # False (forward pass)
        workspace,
        workspace.shape[0],
        accumulate,           # False
        use_split_accumulator,
    )
    kwargs = {
        "comm_overlap": None,
        "comm_type": None,
        "extra_output": None,
        "bulk_overlap": False,
        "alpha": 1.0,
        "beta": 0.0,
    }

    # Call C++ extension
    out, bias_grad, gelu_input, extra_output = tex.generic_gemm(*args, **kwargs)

    return out, bias_grad, gelu_input, extra_output
```

---

#### Frame 14: C++ GEMM Implementation

**Location:** `transformer_engine/pytorch/csrc/extensions/gemm.cpp`

```cpp
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
generic_gemm(
    torch::Tensor A,           // Weight [768, 768] FP4
    bool transa,               // true
    torch::Tensor B,           // Input [1024, 768] FP4
    bool transb,               // false
    torch::Tensor out,
    Quantizer quantization_params,
    TE_DType out_dtype,        // kBFloat16
    torch::Tensor bias,
    TE_DType bias_dtype,
    bool gelu,
    torch::Tensor gelu_in,
    bool grad,
    torch::Tensor workspace,
    size_t workspaceSize,
    bool accumulate,
    bool use_split_accumulator,
    torch::Tensor comm_overlap,
    CommOverlapType comm_type,
    torch::Tensor extra_output,
    bool bulk_overlap,
    float alpha,
    float beta
) {
    // Get matrix dimensions
    // A: [M, K] = [768, 768] (weights)
    // B: [N, K] = [1024, 768] (inputs)
    // C: [N, M] = [1024, 768] (output)

    int M = A.size(transa ? 1 : 0);  // 768
    int K = A.size(transa ? 0 : 1);  // 768
    int N = B.size(transb ? 1 : 0);  // 1024

    // Allocate output if not provided
    if (!out.defined()) {
        out = torch::empty(
            {N, M},                    // [1024, 768]
            torch::dtype(out_dtype).device(A.device())
        );
    }

    // Check if A or B are NVFP4 tensors
    bool is_nvfp4 = (
        A.is_quantized_tensor() &&
        A.quantization_dtype() == TE_DType::kFloat4E2M1
    );

    if (is_nvfp4) {
        // Dispatch to NVFP4-specific GEMM
        nvfp4_gemm(
            A, transa,
            B, transb,
            out,
            bias,
            workspace, workspaceSize,
            use_split_accumulator,
            alpha, beta,
            at::cuda::getCurrentCUDAStream()
        );
    }
    // ... other dispatch paths

    return std::make_tuple(out, bias_grad, gelu_input, extra_output);
}
```

---

#### Frame 15: cuBLAS FP4 GEMM Kernel

**Location:** `transformer_engine/common/gemm/cublaslt_gemm.cpp`

```cpp
void nvfp4_gemm(
    QuantizedTensor A,              // [768, 768] FP4
    bool transa,
    QuantizedTensor B,              // [1024, 768] FP4
    bool transb,
    torch::Tensor C,                // [1024, 768] BF16 (output)
    torch::Tensor bias,
    torch::Tensor workspace,
    size_t workspaceSize,
    bool use_split_accumulator,
    float alpha,
    float beta,
    cudaStream_t stream
) {
    // Extract FP4 data and scales from quantized tensors
    uint8_t* A_data = A.data();                  // FP4 packed data
    uint8_t* A_scale = A.scale_inv();            // E4M3 scales
    float A_amax = A.amax();                     // Global amax

    uint8_t* B_data = B.data();
    uint8_t* B_scale = B.scale_inv();
    float B_amax = B.amax();

    __nv_bfloat16* C_data = C.data_ptr<__nv_bfloat16>();

    // Setup cuBLAS LT matrix descriptors with block scaling
    cublasLtMatrixLayout_t A_desc, B_desc, C_desc;

    // Create matrix descriptor for A (weights)
    cublasLtMatrixLayoutCreate(&A_desc, CUDA_R_4F_E2M1, M, K, M);

    // Set block scaling metadata for A
    cublasLtMatrixLayoutSetAttribute(
        A_desc,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &A_scale,
        sizeof(void*)
    );

    // For 2D quantization, set block dimensions
    int block_size_2d[2] = {16, 16};  // 16x16 blocks
    cublasLtMatrixLayoutSetAttribute(
        A_desc,
        CUBLASLT_MATRIX_LAYOUT_BLOCK_SIZE,
        block_size_2d,
        sizeof(block_size_2d)
    );

    // Create matrix descriptor for B (inputs)
    cublasLtMatrixLayoutCreate(&B_desc, CUDA_R_4F_E2M1, N, K, N);

    // Set block scaling for B (1D blocks)
    cublasLtMatrixLayoutSetAttribute(
        B_desc,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &B_scale,
        sizeof(void*)
    );

    int block_size_1d = 16;
    cublasLtMatrixLayoutSetAttribute(
        B_desc,
        CUBLASLT_MATRIX_LAYOUT_BLOCK_SIZE,
        &block_size_1d,
        sizeof(block_size_1d)
    );

    // Create output descriptor (BF16)
    cublasLtMatrixLayoutCreate(&C_desc, CUDA_R_16BF, N, M, N);

    // Create matmul descriptor
    cublasLtMatmulDesc_t matmul_desc;
    cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

    // Set transpose operations
    cublasLtMatmulDescSetAttribute(
        matmul_desc,
        CUBLASLT_MATMUL_DESC_TRANSA,
        &transa,
        sizeof(transa)
    );
    cublasLtMatmulDescSetAttribute(
        matmul_desc,
        CUBLASLT_MATMUL_DESC_TRANSB,
        &transb,
        sizeof(transb)
    );

    // Set split-K accumulator mode if requested
    if (use_split_accumulator) {
        int split_k = 0;  // Let cuBLAS choose optimal split
        cublasLtMatmulDescSetAttribute(
            matmul_desc,
            CUBLASLT_MATMUL_DESC_SPLIT_K_MODE,
            &split_k,
            sizeof(split_k)
        );
    }

    // Find best algorithm
    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(
        pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspaceSize,
        sizeof(workspaceSize)
    );

    cublasLtMatmulHeuristicResult_t heuristic;
    int returnedResults = 0;
    cublasLtMatmulAlgoGetHeuristic(
        cublas_handle,
        matmul_desc,
        A_desc,
        B_desc,
        C_desc,
        C_desc,
        pref,
        1,
        &heuristic,
        &returnedResults
    );

    // Execute GEMM
    // C = alpha * (A^T @ B) + beta * C
    // [1024, 768] = 1.0 * ([768, 768]^T @ [1024, 768])
    cublasLtMatmul(
        cublas_handle,
        matmul_desc,
        &alpha,                          // 1.0
        A_data,                          // FP4 weight data
        A_desc,
        B_data,                          // FP4 input data
        B_desc,
        &beta,                           // 0.0
        C_data,                          // BF16 output
        C_desc,
        C_data,
        C_desc,
        &heuristic.algo,
        workspace.data_ptr(),
        workspaceSize,
        stream
    );

    // Add bias if provided
    if (bias.defined()) {
        // Launch bias addition kernel
        add_bias_kernel<<<blocks, threads, 0, stream>>>(
            C_data,
            bias.data_ptr<__nv_bfloat16>(),
            N,  // 1024
            M   // 768
        );
    }

    // Cleanup
    cublasLtMatrixLayoutDestroy(A_desc);
    cublasLtMatrixLayoutDestroy(B_desc);
    cublasLtMatrixLayoutDestroy(C_desc);
    cublasLtMatmulDescDestroy(matmul_desc);
    cublasLtMatmulPreferenceDestroy(pref);
}
```

**CUDA Kernel Details:**

The actual GEMM computation happens inside cuBLAS LT's optimized kernels:

```
1. Load FP4 tiles from global memory
2. Dequantize using block scales:
   - Read E4M3 scale for block
   - Convert scale to FP32
   - Unpack FP4 values (2 per byte)
   - Multiply by scale: value_fp32 = value_fp4 * scale
3. Accumulate in FP32 (or with split accumulator for better precision)
4. Convert accumulated FP32 to BF16 for output
5. Write BF16 output to global memory
```

**Memory Access Pattern:**

```
Input A (Weight) [768, 768] FP4:
  ┌─────────────────────┐
  │ Load 16x16 tile     │
  │ Read scales [48,48] │ → Dequantize → FP32
  └─────────────────────┘

Input B (Activation) [1024, 768] FP4:
  ┌─────────────────────┐
  │ Load 16x16 tile     │
  │ Read scales [...]   │ → Dequantize → FP32
  └─────────────────────┘

               ↓

        FP32 Accumulator
        ┌─────────────┐
        │ C += A * B  │
        └─────────────┘

               ↓

Output C [1024, 768] BF16:
  ┌─────────────────────┐
  │ Convert FP32→BF16   │
  │ Store to memory     │
  └─────────────────────┘
```

---

#### Frame 16: Return from Forward Pass

**Location:** [transformer_engine/pytorch/module/linear.py:337-370](../transformer_engine/pytorch/module/linear.py#L337-L370)

```python
# ------------------------------------------------------
# Prepare output tensor
# ------------------------------------------------------
out = None
if ub_overlap_rs_fprop:
    out = reduce_scatter_out
elif parallel_mode == "row" and tp_size > 1:
    # Handle tensor parallel reduction (not our case)
    pass
else:
    out = gemm_out  # [1024, 768] BF16

# Save context for backward pass
ctx.save_for_backward(inputmat, weight, ...)
ctx.input_quantizer = input_quantizer
ctx.weight_quantizer = weight_quantizer
ctx.grad_input_quantizer = grad_input_quantizer
ctx.grad_weight_quantizer = grad_weight_quantizer
ctx.grad_output_quantizer = grad_output_quantizer
ctx.fp8_recipe = FP8GlobalStateManager.get_fp8_recipe()
# ... other context

return out  # [1024, 768] BF16
```

**Result:** Forward pass complete! We have:
- Input: [1024, 768] BF16 → Quantized to FP4 (with RHT, 1D blocks)
- Weight: [768, 768] BF16 → Quantized to FP4 (2D blocks)
- GEMM: FP4 × FP4 → BF16 (accumulated in FP32)
- Output: [1024, 768] BF16

---

### Phase 4: Backward Pass

#### Frame 17: Compute Loss and Call Backward
```python
loss = out.mean()
loss.backward()
```

**What happens:**
1. `loss = out.mean()` computes scalar loss
2. `loss.backward()` triggers autograd
3. PyTorch traverses computation graph backwards
4. Calls `_Linear.backward()` with `grad_output`

---

#### Frame 18: _Linear.backward Entry

**Location:** [transformer_engine/pytorch/module/linear.py:400+](../transformer_engine/pytorch/module/linear.py#L400)

```python
@staticmethod
def backward(
    ctx,
    grad_output: torch.Tensor,  # [1024, 768] BF16
) -> Tuple[Optional[torch.Tensor], ...]:

    # Restore saved tensors
    (inputmat, weight, ...) = ctx.saved_tensors

    # Get quantizers from context
    input_quantizer = ctx.input_quantizer
    weight_quantizer = ctx.weight_quantizer
    grad_input_quantizer = ctx.grad_input_quantizer      # For dgrad
    grad_output_quantizer = ctx.grad_output_quantizer    # For grad_output
    grad_weight_quantizer = ctx.grad_weight_quantizer    # For wgrad (unused)
    fp8_recipe = ctx.fp8_recipe  # NVFP4BlockScaling

    # Determine what gradients we need
    requires_dgrad = ctx.needs_input_grad[1]  # grad wrt input
    requires_wgrad = ctx.needs_input_grad[0]  # grad wrt weight

    # ... backward computation
```

---

#### Frame 19: Prepare Grad Output Tensor

```python
# ------------------------------------------------------
# Prepare grad output tensor
# ------------------------------------------------------

# grad_output shape: [1024, 768] BF16
if ctx.fp8 or ctx.debug:  # True
    if isinstance(grad_output, QuantizedTensorStorage):
        grad_output.update_usage(rowwise_usage=True)
    else:
        # Need to quantize grad_output
        if grad_output_quantizer is None:
            raise ValueError("Missing quantizer for grad_output")

        # For NVFP4: use backward quantizer with RHT + stochastic rounding
        grad_output_quantizer.set_usage(
            rowwise=True,
            columnwise=ctx.requires_wgrad  # True (need for wgrad GEMM)
        )

        # QUANTIZE GRAD OUTPUT
        grad_output = grad_output_quantizer(grad_output)
```

---

#### Frame 20: Quantize Grad Output (BF16 → FP4)

**Similar to input quantization but uses backward quantizer:**

**Location:** [transformer_engine/pytorch/tensor/nvfp4_tensor.py:178-180](../transformer_engine/pytorch/tensor/nvfp4_tensor.py#L178-L180)

**Key difference: Stochastic Rounding**

```cuda
__global__ void nvfp4_quantize_rht_1d_stochastic_kernel(
    const __nv_bfloat16* input,
    uint8_t* output,
    uint8_t* scale_inv,
    float* amax,
    int M, int K,
    int rht_mask,
    curandState* rand_states  // RNG state for stochastic rounding
) {
    int tid = threadIdx.x;
    int block_id = blockIdx.x;

    // Steps 1-3: Same as forward (load, RHT, compute scale)
    // ... (see Frame 9)

    // STEP 4: Quantize with STOCHASTIC ROUNDING
    float scaled_val = shared_data[tid] / scale;

    // Generate random number [0, 1)
    curandState local_state = rand_states[blockIdx.x * blockDim.x + tid];
    float random = curand_uniform(&local_state);
    rand_states[blockIdx.x * blockDim.x + tid] = local_state;

    // Stochastic rounding:
    // Instead of rounding to nearest, round probabilistically
    // P(round up) = fractional part
    // P(round down) = 1 - fractional part

    // Find two nearest FP4 values
    float lower = floor_to_fp4(scaled_val);
    float upper = ceil_to_fp4(scaled_val);
    float frac = (scaled_val - lower) / (upper - lower);

    // Stochastic decision
    float quantized_val = (random < frac) ? upper : lower;

    uint8_t fp4_val = float_to_e2m1(quantized_val);

    // Rest same as forward (pack, store)
    // ...
}
```

**Why Stochastic Rounding for Gradients?**

```
Deterministic Rounding (Round-to-Nearest):
  - Always rounds same value same way
  - Introduces systematic bias
  - Small gradients can get "stuck" at zero
  - Gradient updates become biased

Stochastic Rounding:
  - Rounds probabilistically
  - Unbiased in expectation: E[quantized] = original
  - Small gradients have chance to update
  - Better convergence in low precision
```

**Example:**
```
Value: 0.3 (between FP4 values 0.0 and 0.5)

Deterministic: Always rounds to 0.5 (nearest)

Stochastic:
  - 40% chance → 0.5
  - 60% chance → 0.0
  - Average over many iterations: 0.3 ✓
```

---

#### Frame 21: Compute Grad Input (DGRAD GEMM)

**Location:** [transformer_engine/pytorch/module/linear.py:671-750](../transformer_engine/pytorch/module/linear.py#L671-L750)

```python
# ------------------------------------------------------
# Compute grad input tensor
# ------------------------------------------------------

dgrad = None
dgrad_work = None
if ctx.requires_dgrad:  # True

    # Make sure required data is available
    if isinstance(grad_output, QuantizedTensorStorage):
        grad_output.update_usage(rowwise_usage=True)
    if ctx.weight_quantizer is not None and isinstance(weight_fp8, QuantizedTensorStorage):
        weight_fp8.update_usage(columnwise_usage=True)

    # Choose whether to use GEMM kernel with split accumulator
    use_split_accumulator = _2X_ACC_DGRAD
    if ctx.fp8:
        recipe = ctx.fp8_recipe
        if hasattr(recipe, "fp8_gemm_dgrad"):
            use_split_accumulator = recipe.fp8_gemm_dgrad.use_split_accumulator

    # Update grad input quantizer
    if ctx.grad_input_quantizer is not None:
        ctx.grad_input_quantizer.set_usage(rowwise=True, columnwise=False)

    # DGRAD GEMM: grad_input = grad_output @ weight
    # [1024, 768] = [1024, 768] @ [768, 768]
    dgrad, *_ = general_gemm(
        weight_fp8,                # Weight [768, 768] FP4 (columnwise)
        grad_output,               # Grad output [1024, 768] FP4 (rowwise)
        get_workspace(),
        quantization_params=ctx.grad_input_quantizer,
        out_dtype=ctx.activation_dtype,  # BF16
        bias=None,
        use_split_accumulator=use_split_accumulator,
        layout="NN",               # No transpose
    )
```

**GEMM Layout:**
```
grad_output: [1024, 768] (rowwise FP4)
weight:      [768, 768]  (columnwise FP4, used as transpose)
grad_input:  [1024, 768] (BF16)

Matrix multiplication: grad_input = grad_output @ weight
```

---

#### Frame 22: Compute Grad Weight (WGRAD GEMM)

**Location:** [transformer_engine/pytorch/module/linear.py:750-900](../transformer_engine/pytorch/module/linear.py#L750-L900)

```python
# ------------------------------------------------------
# Compute grad weight tensor
# ------------------------------------------------------

wgrad = None
grad_bias = None

if ctx.requires_wgrad:  # True

    # Prepare input tensor for wgrad GEMM
    if ctx.fp8 or ctx.debug:
        if isinstance(inputmat, QuantizedTensorStorage):
            # Already quantized (from forward pass)
            pass
        else:
            # Quantize input tensor
            quantizer = ctx.input_quantizer
            if quantizer.supports_only_rowwise_all_gather():
                quantizer.set_usage(rowwise=True, columnwise=True)
            else:
                quantizer.set_usage(rowwise=False, columnwise=True)
            inputmat = quantizer(inputmat)

    # Make sure required data is available
    if isinstance(inputmat_total, QuantizedTensorStorage):
        inputmat_total.update_usage(columnwise_usage=True)
    if isinstance(grad_output, QuantizedTensorStorage):
        grad_output.update_usage(columnwise_usage=True)

    # Choose split accumulator mode
    use_split_accumulator = _2X_ACC_WGRAD
    if ctx.fp8:
        recipe = ctx.fp8_recipe
        if hasattr(recipe, "fp8_gemm_wgrad"):
            use_split_accumulator = recipe.fp8_gemm_wgrad.use_split_accumulator

    # WGRAD GEMM: grad_weight = grad_output^T @ input
    # [768, 768] = [768, 1024] @ [1024, 768]
    wgrad, grad_bias, *_ = general_gemm(
        grad_output,               # [1024, 768] FP4 (columnwise for transpose)
        inputmat_total,            # [1024, 768] FP4 (columnwise)
        get_workspace(),
        quantization_params=None,  # No output quantization
        out_dtype=torch.float32,   # FP32 for grad accumulation
        bias=ctx.bias if ctx.use_bias else None,
        use_split_accumulator=use_split_accumulator,
        layout="TN",               # Transpose grad_output
        grad=True,                 # Compute bias gradient too
    )
```

**GEMM Layout:**
```
grad_output^T: [768, 1024] (columnwise FP4, transposed)
inputmat:      [1024, 768] (columnwise FP4)
grad_weight:   [768, 768]  (FP32 for accumulation)

Matrix multiplication: grad_weight = grad_output^T @ inputmat
```

**Note:** Weight gradients are computed in FP32 for better accuracy in gradient accumulation.

---

#### Frame 23: Return from Backward Pass

```python
# Update FP8 scaling factors if needed
if ctx.reduce_and_update_bwd_fp8_tensors and not is_graph_capturing():
    FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)

# Return gradients (None for non-tensor inputs)
return (
    wgrad,                          # grad wrt weight
    dgrad.view(ctx.inp_shape),      # grad wrt input
    grad_bias,                      # grad wrt bias
    None,  # is_first_microbatch
    None,  # fp8
    None,  # fp8_calibration
    # ... (all other args get None)
)
```

**Result:** Backward pass complete! We have:
- Grad output: [1024, 768] BF16 → Quantized to FP4 (with RHT, stochastic rounding)
- Grad input (DGRAD): FP4 grad_output × FP4 weight → BF16
- Grad weight (WGRAD): FP4 grad_output^T × FP4 input → FP32
- Gradients ready for optimizer step

---

### Phase 5: Autocast Exit

#### Frame 24: Exit Autocast Context

**When the `with te.autocast():` block ends:**

**Location:** [transformer_engine/pytorch/quantization.py:848-852](../../../transformer_engine/pytorch/quantization.py#L848-L852)

```python
@contextmanager
def autocast(...):
    # ... entry logic (executed earlier)

    fp8_state = FP8GlobalStateManager.get_autocast_state()
    FP8GlobalStateManager.autocast_enter(...)

    try:
        yield  # ← Code inside 'with' block runs here
    finally:
        # ALWAYS executed, even if exception occurs
        # Restore saved state
        FP8GlobalStateManager.set_autocast_state(fp8_state)
        FP8GlobalStateManager.autocast_exit(enabled, _graph=_graph)
```

---

#### Frame 25: Autocast Exit Implementation

**Location:** [transformer_engine/pytorch/quantization.py:580-620](../transformer_engine/pytorch/quantization.py#L580-L620)

```python
@classmethod
def autocast_exit(
    cls,
    enabled: bool,
    _graph: bool = False,
) -> None:
    """Restore state and perform cleanup after FP8 region exit."""

    if not enabled:
        return

    # Decrement nesting depth
    cls.AUTOCAST_DEPTH -= 1

    # If fully exited (depth = 0), perform cleanup
    if cls.AUTOCAST_DEPTH == 0:

        # Get the recipe for this autocast region
        recipe = cls.FP8_RECIPE
        autocast_key = cls.get_unique_autocast_key(recipe, cls.FP8_DISTRIBUTED_GROUP)

        # Reduce and update FP8 tensors if needed
        if (
            recipe is not None
            and recipe.reduce_amax
            and cls.IS_FIRST_FP8_MODULE
            and not _graph
        ):
            # Reduce amax values across distributed group (if applicable)
            cls.reduce_and_update_fp8_tensors(forward=True)

        # Clear is_first_module flag
        cls.IS_FIRST_FP8_MODULE = False

        # Clean up autocast arguments for this context
        if autocast_key in cls.autocast_arguments:
            del cls.autocast_arguments[autocast_key]
```

**State After Exit:**

```python
FP8GlobalStateManager.FP8_ENABLED = False          # Restored
FP8GlobalStateManager.FP8_RECIPE = None            # Restored
FP8GlobalStateManager.AUTOCAST_DEPTH = 0           # Decremented
FP8GlobalStateManager.IS_FIRST_FP8_MODULE = False  # Cleared
```

---

## Data Flow Diagrams

### Overall Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FORWARD PASS                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Input (BF16)          Weight (BF16)                                 │
│  [1024, 768]           [768, 768]                                    │
│       │                     │                                        │
│       ↓                     ↓                                        │
│  ┌─────────────┐       ┌─────────────┐                              │
│  │ Apply RHT   │       │ 2D Block    │                              │
│  │ (16x16 Had) │       │ Quantize    │                              │
│  └─────────────┘       │ (16x16)     │                              │
│       │                └─────────────┘                              │
│       ↓                     │                                        │
│  ┌─────────────┐            │                                       │
│  │ Compute     │            ↓                                       │
│  │ amax/block  │       ┌─────────────┐                              │
│  └─────────────┘       │ Compute     │                              │
│       │                │ amax/block  │                              │
│       ↓                └─────────────┘                              │
│  ┌─────────────┐            │                                       │
│  │ Scale &     │            ↓                                       │
│  │ Quantize    │       ┌─────────────┐                              │
│  │ to FP4      │       │ Scale &     │                              │
│  └─────────────┘       │ Quantize    │                              │
│       │                │ to FP4      │                              │
│       │                └─────────────┘                              │
│       │                     │                                        │
│       ↓                     ↓                                        │
│  ┌────────────────────────────────┐                                 │
│  │  cuBLAS LT GEMM (FP4 × FP4)    │                                 │
│  │  - Load FP4 data               │                                 │
│  │  - Dequantize with scales      │                                 │
│  │  - Multiply (FP32 accum)       │                                 │
│  │  - Convert to BF16             │                                 │
│  └────────────────────────────────┘                                 │
│               │                                                      │
│               ↓                                                      │
│          Output (BF16)                                               │
│          [1024, 768]                                                 │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         BACKWARD PASS                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Grad Output (BF16)                                                  │
│  [1024, 768]                                                         │
│       │                                                              │
│       ↓                                                              │
│  ┌─────────────────────┐                                            │
│  │ Apply RHT           │                                            │
│  │ Stochastic Rounding │                                            │
│  │ Quantize to FP4     │                                            │
│  └─────────────────────┘                                            │
│       │                                                              │
│       ├──────────────────┬──────────────────┐                       │
│       │                  │                  │                       │
│       ↓                  ↓                  ↓                       │
│  ┌─────────┐       ┌─────────┐       ┌─────────┐                   │
│  │ DGRAD   │       │ WGRAD   │       │ BIAS    │                   │
│  │ GEMM    │       │ GEMM    │       │ GRAD    │                   │
│  └─────────┘       └─────────┘       └─────────┘                   │
│       │                  │                  │                       │
│       ↓                  ↓                  ↓                       │
│  Grad Input       Grad Weight         Grad Bias                     │
│  [1024, 768]      [768, 768]          [768]                         │
│  (BF16)           (FP32)              (BF16)                         │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Memory Layout

```
┌──────────────────────────────────────────────────────────────────┐
│                   NVFP4 TENSOR STORAGE                            │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Original Tensor: [1024, 768] BF16                                │
│  Size: 1,572,864 bytes                                            │
│                                                                    │
│  Quantized Storage:                                               │
│                                                                    │
│  ┌─────────────────────────────────────┐                          │
│  │ Rowwise Data (uint8)                │                          │
│  │ Shape: [1024, 384]                  │  ← 2 FP4 per byte       │
│  │ Size: 393,216 bytes                 │                          │
│  └─────────────────────────────────────┘                          │
│                                                                    │
│  ┌─────────────────────────────────────┐                          │
│  │ Rowwise Scales (uint8, E4M3)        │                          │
│  │ Shape: [1024, 48]                   │  ← 768/16 = 48 blocks   │
│  │ Size: 49,152 bytes                  │                          │
│  └─────────────────────────────────────┘                          │
│                                                                    │
│  ┌─────────────────────────────────────┐                          │
│  │ Rowwise Amax (float32)              │                          │
│  │ Shape: [1]                          │                          │
│  │ Size: 4 bytes                       │                          │
│  └─────────────────────────────────────┘                          │
│                                                                    │
│  ┌─────────────────────────────────────┐                          │
│  │ Columnwise Data (uint8)             │                          │
│  │ Shape: [768, 512]                   │  ← Transposed            │
│  │ Size: 393,216 bytes                 │                          │
│  └─────────────────────────────────────┘                          │
│                                                                    │
│  ┌─────────────────────────────────────┐                          │
│  │ Columnwise Scales (uint8, E4M3)     │                          │
│  │ Shape: [768, 64]                    │  ← 1024/16 = 64 blocks  │
│  │ Size: 49,152 bytes                  │                          │
│  └─────────────────────────────────────┘                          │
│                                                                    │
│  ┌─────────────────────────────────────┐                          │
│  │ Columnwise Amax (float32)           │                          │
│  │ Shape: [1]                          │                          │
│  │ Size: 4 bytes                       │                          │
│  └─────────────────────────────────────┘                          │
│                                                                    │
│  Total Quantized: 884,744 bytes                                   │
│  Compression Ratio: 56.2%                                          │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
```

---

## Key Operations Deep Dive

### Random Hadamard Transform (RHT)

**Purpose:** Smooth outliers in tensor distributions to make them easier to quantize accurately.

**Implementation:**

```python
# Hadamard matrix (16x16) with random sign flips
rht_matrix = sign_matrix @ hadamard_matrix
# sign_matrix = diag([1, 1, 1, -1, 1, -1, -1, -1, ...])
# hadamard_matrix = H2 ⊗ H2 ⊗ H2 ⊗ H2  (Kronecker product)

# Apply to 16-element blocks
transformed = rht_matrix @ block  # Matrix-vector multiply
```

**Effect:**
- Spreads outlier values across multiple elements
- Makes amax computation more stable
- Improves quantization accuracy for activations/gradients
- Reversible: Apply inverse RHT after dequantization

**Example:**
```
Before RHT: [0.1, 0.2, 5.0, 0.1, ...]  ← Large outlier (5.0)
After RHT:  [1.3, 1.4, 1.2, 1.5, ...]  ← Outlier distributed

Quantization is now more accurate for all elements!
```

---

### Stochastic Rounding

**Purpose:** Prevent bias in gradient quantization by rounding probabilistically.

**Implementation:**

```cuda
float quantize_stochastic(float value, curandState* state) {
    // Find two nearest representable FP4 values
    float lower = floor_to_fp4(value);
    float upper = ceil_to_fp4(value);

    // Compute fractional distance
    float frac = (value - lower) / (upper - lower);

    // Generate random number [0, 1)
    float random = curand_uniform(state);

    // Probabilistic rounding
    return (random < frac) ? upper : lower;
}
```

**Example:**
```
Value: 0.3 (between FP4 values 0.0 and 0.5)
Distance from 0.0: 0.3
Distance from 0.5: 0.2
Total distance: 0.5

Probability of 0.5: 0.3 / 0.5 = 60%
Probability of 0.0: 0.2 / 0.5 = 40%

Over many iterations: E[quantized] = 0.6 × 0.5 + 0.4 × 0.0 = 0.3 ✓
```

---

### 2D Block Quantization

**Purpose:** Use a single scale for 16×16 blocks of weight values.

**Benefits:**
- Fewer scales to store (256x fewer than 1D for same block size)
- Better memory locality for weight access
- Weights are static, so coarser quantization is acceptable
- Reduces metadata overhead

**Layout:**
```
Weight Tensor [768, 768]:

┌───────┬───────┬───────┬ ─ ─ ─
│ 16×16 │ 16×16 │ 16×16 │
│ Block │ Block │ Block │
│  s₀   │  s₁   │  s₂   │
├───────┼───────┼───────┼ ─ ─ ─
│ 16×16 │ 16×16 │ 16×16 │
│ Block │ Block │ Block │
│  s₄₈  │  s₄₉  │  s₅₀  │
├───────┼───────┼───────┼ ─ ─ ─
│   ·   │   ·   │   ·   │
│   ·   │   ·   │   ·   │

Scale array: [48, 48] E4M3
(768/16 = 48 blocks per dimension)
```

---

### Split-K Accumulator

**Purpose:** Improve GEMM accuracy by using higher precision for intermediate accumulations.

**Implementation:**
```
Standard GEMM:
  C = A × B
  Accumulate in FP32

Split-K GEMM:
  1. Split K dimension into chunks
  2. Compute partial sums in parallel
  3. Each partial sum uses FP32 accumulator
  4. Reduce partial sums (also in FP32)
  5. Convert final result to output dtype

Benefits:
  - Better numerical accuracy
  - Can reduce accumulated rounding errors
  - More parallelism for large K
```

---

## Summary

This completes the frame-by-frame trace of `te.autocast` with NVFP4BlockScaling recipe!

**Key Takeaways:**

1. **Recipe Configuration:** Environment variables control RHT, stochastic rounding, and 2D quantization
2. **Autocast Context:** Manages global state through `FP8GlobalStateManager`
3. **Quantizer Creation:** Recipe state factory creates appropriate quantizers per tensor type
4. **Forward Pass:**
   - Input: BF16 → FP4 (1D, with RHT)
   - Weight: BF16 → FP4 (2D, no RHT)
   - GEMM: FP4 × FP4 → BF16 (via cuBLAS LT)
5. **Backward Pass:**
   - Grad output: BF16 → FP4 (1D, with RHT + stochastic rounding)
   - DGRAD: FP4 × FP4 → BF16
   - WGRAD: FP4 × FP4 → FP32 (for gradient accumulation)
6. **Memory Savings:** ~44% reduction with quantized storage
7. **Performance:** FP4 GEMMs run on Tensor Cores (SM 8.9+)

