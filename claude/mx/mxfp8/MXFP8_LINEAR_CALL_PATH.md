# te.Linear MXFP8 Call Path Analysis

## Overview

This document traces the complete call path for `te.Linear` when processing inputs under the **MXFP8BlockScaling** recipe. It covers:

1. Python API entry point ([te.Linear.forward](../../../transformer_engine/pytorch/module/linear.py#L1370))
2. MXFP8 recipe detection and quantizer initialization
3. Forward pass execution through quantization and GEMM
4. C++ bindings and CUDA kernel invocations
5. Data flow from user-facing APIs through to GPU kernels

---

## Part 1: Recipe Detection and Context

### Recipe Class Hierarchy

```
Recipe (base class)
├── DelayedScaling
├── Float8CurrentScaling
├── Float8BlockScaling
├── MXFP8BlockScaling      ◄── OUR FOCUS
├── NVFP4BlockScaling
└── CustomRecipe
```

**File**: [transformer_engine/common/recipe/__init__.py](../../../transformer_engine/common/recipe/__init__.py#L86-L515)

### MXFP8BlockScaling Recipe Definition

```python
@dataclass()
class MXFP8BlockScaling(Recipe):
    """
    Use the MXFP8 scaling strategy.

    This is a block scaling strategy with E8M0 scales (power-of-2):
    - Block size: 32 consecutive values per scaling factor
    - Scale format: E8M0 (8-bit exponent, power-of-2 only)
    - Data format: FP8 E4M3 (8-bit floating point)
    """

    # Configuration
    margin: int = 0
    fp8_format: Format = Format.E4M3  # FP8 data type (8-bit)
    fp8_dpa: bool = False
    fp8_mha: bool = False

    # No RHT, stochastic rounding, or 2D quantization
    # Much simpler than NVFP4!
```

**File**: [recipe/__init__.py:265-303](../../../transformer_engine/common/recipe/__init__.py#L265-L303)

---

## Part 2: te.Linear Class Structure

### Class Definition

```python
class Linear(TransformerEngineBaseModule):
    """Applies linear transformation: y = xA^T + b"""

    def __init__(self, in_features: int, out_features: int, ...):
        super().__init__()
        # Weight and bias initialization
        self.in_features = in_features
        self.out_features = out_features
        # ... parameter registration and initialization ...

        if with_fp8_params:
            self.init_fp8_metadata()
```

**File**: [transformer_engine/pytorch/module/linear.py:1009-1334](../../../transformer_engine/pytorch/module/linear.py#L1009-L1334)

### Key Initialization Method: `set_meta_tensor`

This method is called during FP8 setup to configure quantizers based on the recipe type:

```python
def set_meta_tensor(self, fwd: bool, recipe: Recipe) -> None:
    """Init scales and amaxes for fwd | bwd."""
    super().set_meta_tensor(fwd, recipe)

    # Customize quantizers based on recipe type
    recipe = FP8GlobalStateManager.get_fp8_recipe()
    if recipe.float8_current_scaling():
        self._customize_quantizers_float8_current_scaling(fwd, recipe)
    elif recipe.float8_block_scaling():
        self._customize_quantizers_float8_blockwise_scaling(fwd, recipe)
    elif recipe.mxfp8():  # ◄── MXFP8 DETECTION
        self._customize_quantizers_mxfp8(fwd, recipe)
    elif recipe.nvfp4():
        self._customize_quantizers_nvfp4(fwd, recipe)
```

**File**: [linear.py:1335-1347](../../../transformer_engine/pytorch/module/linear.py#L1335-L1347)

### MXFP8-Specific Quantizer Customization

```python
def _customize_quantizers_mxfp8(self, fwd: bool, recipe: Recipe) -> None:
    """Customize quantizers based on MXFP8 recipe + linear layer."""
    assert recipe.mxfp8(), "Incorrect recipe."

    # MXFP8 is stateless - no per-layer customization needed
    # No amax reduction groups
    # No RHT configuration
    # No 2D quantization flags

    # Much simpler than NVFP4!
    pass
```

**File**: [linear.py:1698-1707](../../../transformer_engine/pytorch/module/linear.py#L1698-L1707)

**Key Difference from NVFP4**: MXFP8 requires no special configuration for distributed training or advanced features. The quantizers work out-of-the-box with default settings.

---

## Part 3: Forward Pass Call Stack

### Entry Point: `Linear.forward`

```python
@no_torch_dynamo()
def forward(
    self,
    inp: torch.Tensor,
    is_first_microbatch: Optional[bool] = None,
    fp8_output: Optional[bool] = False,
    fp8_grad: Optional[bool] = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """Apply linear transformation to input."""

    # 1. Check if in ONNX export mode
    if is_in_onnx_export_mode():
        return self.onnx_forward(inp, fp8_output)

    # 2. Get weight and bias tensors
    weight_tensor, bias_tensor = self._get_weight_and_bias_tensors()

    # 3. Get quantizers (critical for MXFP8!)
    quantizers = (
        self._get_quantizers(fp8_output, fp8_grad)
        if not debug
        else self._get_debug_quantizers(fp8_output, fp8_grad)
    )

    (
        input_quantizer,       # MXFP8Quantizer
        weight_quantizer,      # MXFP8Quantizer
        output_quantizer,
        grad_input_quantizer,
        grad_weight_quantizer,
        grad_output_quantizer,
    ) = quantizers

    # 4. Call _Linear.apply (PyTorch autograd function)
    linear_fn = _Linear.apply if torch.is_grad_enabled() else _Linear.forward
    out = linear_fn(
        weight_tensor,
        inp,
        bias_tensor if (self.apply_bias and not self.gemm_bias_unfused_add) else None,
        is_first_microbatch,
        self.fp8,
        self.fp8_calibration,
        self.wgrad_store,
        input_quantizer,      # MXFP8Quantizer for inputs
        weight_quantizer,     # MXFP8Quantizer for weights
        output_quantizer,
        grad_input_quantizer,
        grad_weight_quantizer,
        grad_output_quantizer,
        # ... more parameters ...
    )

    return out
```

**File**: [linear.py:1370-1500](../../../transformer_engine/pytorch/module/linear.py#L1370-L1500)

### Quantizer Retrieval

```python
def _get_quantizers(self, fp8_output, fp8_grad):
    """Get quantizers from FP8 global state manager."""
    if not self.fp8:
        return [None] * 6

    # For MXFP8, these quantizers are MXFP8Quantizer instances
    input_quantizer = self.quantizers["scaling_fwd"][tex.FP8FwdTensors.GEMM1_INPUT]
    input_quantizer.internal = True

    (weight_quantizer,) = self._get_weight_quantizers()

    # ... setup output and gradient quantizers ...

    return (
        input_quantizer,        # MXFP8Quantizer ◄─── INPUT
        weight_quantizer,       # MXFP8Quantizer ◄─── WEIGHT
        output_quantizer,
        grad_input_quantizer,
        grad_weight_quantizer,
        grad_output_quantizer,
    )
```

**File**: [linear.py:1502-1526](../../../transformer_engine/pytorch/module/linear.py#L1502-L1526)

---

## Part 4: _Linear Autograd Function

This is the core computation function that handles the actual forward and backward passes.

### Forward Pass Core Logic

```python
class _Linear(torch.autograd.Function):
    """Linear semi-top level module. Calls custom CUDA extensions."""

    @staticmethod
    def forward(
        ctx,
        weight: torch.Tensor,
        inp: torch.Tensor,
        bias: Optional[torch.Tensor],
        # ... many parameters ...
        input_quantizer: Optional[Quantizer],  # ◄── MXFP8Quantizer for input
        weight_quantizer: Optional[Quantizer], # ◄── MXFP8Quantizer for weight
        # ... more quantizers ...
    ) -> torch.Tensor:

        # ==================================================
        # PHASE 1: PREPARE INPUT TENSOR
        # ==================================================

        nvtx_label = "transformer_engine._Linear.forward"

        # Dimension checks
        out_features, in_features = weight.shape
        assert inp.shape[-1] == in_features, "GEMM not possible"

        # Input preprocessing
        inputmat = inp
        if fp8:  # fp8 == self.fp8 (recipe is MXFP8)
            assert_dim_for_fp8_exec(inputmat, weight)

            if with_input_all_gather_nccl or ub_overlap_ag_fprop:
                # Cast local input tensor if needed
                if not isinstance(inputmat, QuantizedTensorStorage) and not experimental:
                    own_quantized_input = True

                    # CRITICAL: Set quantizer usage pattern
                    # MXFP8 supports both rowwise and columnwise quantization
                    input_quantizer.set_usage(
                        rowwise=True,
                        columnwise=backward_needs_input
                    )

                    # QUANTIZATION KERNEL CALL #1: Quantize input
                    # This calls MXFP8Quantizer.__call__() -> tex.quantize()
                    inputmat = input_quantizer(inputmat)
            else:  # No all-gather needed
                if fp8:
                    if isinstance(inputmat, QuantizedTensorStorage):
                        inputmat.update_usage(rowwise_usage=True)
                    else:
                        input_quantizer.set_usage(
                            rowwise=True,
                            columnwise=backward_needs_input and not save_original_input
                        )
                        # QUANTIZATION KERNEL CALL #1: Quantize input
                        inputmat = input_quantizer(inputmat)
                        own_quantized_input = True
                inputmat_total = inputmat

        # ==================================================
        # PHASE 2: PREPARE WEIGHT TENSOR
        # ==================================================

        weightmat = weight
        if fp8:
            if weight_quantizer is not None:
                # Configure for column-wise usage if needed for gradient
                columnwise_usage = is_grad_enabled and inp.requires_grad
                weight_quantizer.set_usage(rowwise=True, columnwise=columnwise_usage)

            # Get quantized weight (cached or freshly quantized)
            # MXFP8Tensor stores both rowwise and columnwise quantizations
            weightmat = module.get_weight_workspace(
                tensor=weight,
                quantizer=weight_quantizer,
                cache_name=None if is_first_microbatch is None else "weight",
                update_workspace=is_first_microbatch is None or is_first_microbatch,
                skip_update_flag=skip_fp8_weight_update,
                fsdp_group=fsdp_group,
                workspace_dtype=activation_dtype,
            )
            weightmat.update_usage(rowwise_usage=True)

        # ==================================================
        # PHASE 3: FORWARD GEMM
        # ==================================================

        # Determine accumulator strategy
        use_split_accumulator = _2X_ACC_FPROP
        if fp8:
            recipe = FP8GlobalStateManager.get_fp8_recipe()
            if hasattr(recipe, "fp8_gemm_fprop"):
                use_split_accumulator = recipe.fp8_gemm_fprop.use_split_accumulator

        # Configure output quantizer
        if output_quantizer is not None:
            output_quantizer.set_usage(rowwise=True, columnwise=False)

        # CRITICAL: Invoke GEMM with quantized tensors
        gemm_out, *_, reduce_scatter_out = general_gemm(
            weightmat,           # Quantized weight (MXFP8Tensor)
            inputmat_total,      # Quantized input (MXFP8Tensor)
            get_workspace(),     # cuBLAS workspace (32MB for Blackwell)
            quantization_params=output_quantizer,
            out_dtype=activation_dtype,
            bias=bias,
            use_split_accumulator=use_split_accumulator,
            ub=ub_obj,
            ub_type=ub_type,
            extra_output=reduce_scatter_out,
        )

        # ==================================================
        # PHASE 4: CACHE STATE FOR BACKWARD
        # ==================================================

        if is_grad_enabled:
            ctx.weight_quantizer = weight_quantizer
            ctx.fp8 = fp8
            ctx.fp8_recipe = FP8GlobalStateManager.get_fp8_recipe() if fp8 else None
            ctx.input_quantizer = input_quantizer
            ctx.grad_input_quantizer = grad_input_quantizer
            ctx.grad_weight_quantizer = grad_weight_quantizer
            ctx.grad_output_quantizer = grad_output_quantizer
            # ... save more state ...
            ctx.save_for_backward(tensors_to_save)

        return out
```

**File**: [linear.py:77-482](../../../transformer_engine/pytorch/module/linear.py#L77-L482)

---

## Part 5: Quantizer Implementation - MXFP8Quantizer

### MXFP8Quantizer Class

**File**: [transformer_engine/pytorch/tensor/mxfp8_tensor.py:27-175](../../../transformer_engine/pytorch/tensor/mxfp8_tensor.py#L27-L175)

```python
class MXFP8Quantizer(Quantizer):
    """Builder class for MXFP8 tensors with block scaling"""

    dtype: TE_DType  # tex.DType.kFloat8E4M3 (default)

    def __init__(
        self,
        fp8_dtype: TE_DType = tex.DType.kFloat8E4M3,
        rowwise: bool = True,
        columnwise: bool = True,
    ) -> None:
        super().__init__(rowwise=rowwise, columnwise=columnwise)
        self.dtype = fp8_dtype

        # No RHT matrix needed
        # No stochastic rounding flag
        # No 2D quantization configuration
        # No amax reduction group

        # MUCH SIMPLER than NVFP4Quantizer!
```

**Key Differences from NVFP4**:
- No Random Hadamard Transform (RHT)
- No stochastic rounding
- No 2D quantization option
- No amax reduction groups for distributed training
- Stateless: scales computed per-call

### Quantization Call: `__call__` -> `quantize_impl`

When the quantizer is invoked (e.g., `inputmat = input_quantizer(inputmat)`), it calls:

```python
def quantize_impl(self, tensor: torch.Tensor) -> QuantizedTensor:
    """Quantize tensor implementation"""
    # This invokes the C++ binding
    return tex.quantize(tensor, self)
```

**File**: [mxfp8_tensor.py:139-142](../../../transformer_engine/pytorch/tensor/mxfp8_tensor.py#L139-L142)

This routes to the **C++ quantize dispatcher**.

### MXFP8Tensor Creation

```python
def make_empty(
    self,
    shape: Iterable[int],
    *,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
) -> MXFP8Tensor:
    """Create empty MXFP8Tensor with allocated storage."""

    # Calculate storage shape
    # FP8 uses 1 byte per value (no packing like FP4)
    M, K = shape[-2], shape[-1]

    # Allocate rowwise data
    data_shape = shape  # Same shape as input
    data = torch.empty(data_shape, dtype=torch.uint8, device=device)

    # Allocate E8M0 scaling factors
    # Block size = 32 elements
    scale_shape = self.get_scale_shape(shape, columnwise=False)
    # scale_shape = (..., M, ceil(K/32))
    scale_inv = torch.empty(scale_shape, dtype=torch.uint8, device=device)

    # Similarly for columnwise if needed
    if self.columnwise:
        columnwise_data = torch.empty(
            (*shape[:-2], K, M), dtype=torch.uint8, device=device
        )
        columnwise_scale_inv = torch.empty(
            self.get_scale_shape((K, M), columnwise=True),
            dtype=torch.uint8,
            device=device,
        )
    else:
        columnwise_data = None
        columnwise_scale_inv = None

    return MXFP8Tensor(
        shape=shape,
        dtype=dtype,
        rowwise_data=data,
        rowwise_scale_inv=scale_inv,
        columnwise_data=columnwise_data,
        columnwise_scale_inv=columnwise_scale_inv,
        fp8_dtype=self.dtype,
        quantizer=self,
        requires_grad=requires_grad,
    )
```

**File**: [mxfp8_tensor.py:85-138](../../../transformer_engine/pytorch/tensor/mxfp8_tensor.py#L85-L138)

---

## Part 6: C++ Bindings and Quantization Dispatch

### Entry Point: `tex.quantize` Call

**File**: [transformer_engine/pytorch/csrc/extensions/cast.cpp:33-79](../../../transformer_engine/pytorch/csrc/extensions/cast.cpp#L33-L79)

The Python call `tex.quantize(tensor, self)` routes to C++ code:

```cpp
// Simplified representation
at::Tensor quantize(
    const at::Tensor& tensor,
    const MXFP8Quantizer& quantizer
) {
    // 1. Extract quantizer parameters
    TE_DType fp8_dtype = quantizer.dtype;  // kFloat8E4M3
    bool rowwise = quantizer.rowwise;
    bool columnwise = quantizer.columnwise;

    // 2. Invoke MXFP8 quantization kernel
    // No RHT needed
    // No stochastic rounding
    // Direct block-wise quantization

    return nvte_quantize_mxfp8(
        tensor,
        fp8_dtype,
        rowwise,
        columnwise
    );
}
```

### Key Quantization Kernels

These CUDA kernels are invoked during quantization:

**File**: [transformer_engine/common/cast/mxfp8/quantize_mxfp8.cuh:43-538](../../../transformer_engine/common/cast/mxfp8/quantize_mxfp8.cuh#L43-L538)

**MXFP8 Quantization Kernel**:
- Divides input into 32-element blocks
- Computes amax per block
- Encodes scale as E8M0 (8-bit exponent, power-of-2)
- Quantizes values to FP8 E4M3

**No additional kernels needed** (unlike NVFP4):
- No Hadamard Transform
- No stochastic rounding
- No 2D quantization variant

**Much simpler pipeline than NVFP4!**

---

## Part 7: GEMM Execution

### `general_gemm` Call

**File**: [transformer_engine/pytorch/cpp_extensions/__init__.py](../../../transformer_engine/pytorch/cpp_extensions/__init__.py)

The `general_gemm` function invokes cuBLASLt with FP8 MXFP8 kernel selection:

```python
def general_gemm(
    weightmat,           # MXFP8Tensor with quantized weights
    inputmat_total,      # MXFP8Tensor with quantized inputs
    workspace,
    quantization_params=None,
    out_dtype=torch.bfloat16,
    bias=None,
    use_split_accumulator=True,
    ub=None,
    ub_type=None,
    extra_output=None,
):
    """General GEMM supporting quantized tensors."""

    # For MXFP8 tensors:
    # 1. Extract quantized data from MXFP8Tensor
    # 2. Extract E8M0 scaling factors
    # 3. Call cuBLASLt with MXFP8 descriptor
    # 4. Output in higher precision (typically BF16 or F32)

    return tex.general_gemm(
        weightmat,
        inputmat_total,
        workspace,
        quantization_params,
        out_dtype,
        bias,
        use_split_accumulator,
        ub,
        ub_type,
        extra_output,
    )
```

### GEMM Kernel Selection (C++ Side)

**File**: [transformer_engine/common/gemm/cublaslt_gemm.cu](../../../transformer_engine/common/gemm/cublaslt_gemm.cu)

Key steps:

1. **Descriptor Setup**: Configure cuBLASLt matmul descriptor with:
   - Input data types: FP8 E4M3 (MXFP8)
   - Scaling modes: Block scaling with E8M0 scales
   - Block size: 32 elements
   - Accumulation precision: FP32 (split accumulator)

2. **Scale Application**:
   - Per-block E8M0 scales (from quantized tensor)
   - Power-of-2 scales simplify computation

3. **Kernel Dispatch**: cuBLASLt selects appropriate CUTLASS kernel:
   - MXFP8 FP8 x FP8 GEMM with block scaling
   - Blackwell-optimized kernels (CC 10.0+)

4. **Output**: Result dequantized to higher precision

---

## Part 8: Complete Call Graph

### Forward Pass Call Sequence

```
Python:
  te.Linear.forward()
  ├─ Get quantizers (MXFP8Quantizer instances)
  ├─ _Linear.apply()
  │  └─ _Linear.forward()
  │     ├─ input_quantizer(inp)        ◄── Call #1: Quantize input
  │     │  └─ MXFP8Quantizer.quantize_impl()
  │     │     └─ tex.quantize(inp, self)
  │     │
  │     ├─ weight_quantizer(weight)    ◄── Call #2: Quantize weight
  │     │  └─ MXFP8Quantizer.quantize_impl()
  │     │     └─ tex.quantize(weight, self)
  │     │
  │     └─ general_gemm(
  │        weight_mxfp8,
  │        input_mxfp8,
  │        workspace,
  │        ...
  │     )   ◄── Call #3: Quantized GEMM
  │        └─ tex.general_gemm()

C++:
  tex.quantize()
  ├─ For MXFP8:
  │  ├─ Block-wise MXFP8 Quantization Kernel
  │  │  └─ nvte_quantize_mxfp8()
  │  │     ├─ Divide into 32-element blocks
  │  │     ├─ Compute amax per block
  │  │     ├─ Generate E8M0 scale (power-of-2)
  │  │     └─ Quantize to FP8 E4M3
  │  │
  │  └─ No additional kernels needed!
  │     (No RHT, no SR, no 2D quantization)
  │
  └─ Return MXFP8Tensor
     ├─ rowwise_data: uint8 (FP8 E4M3 values)
     ├─ rowwise_scale_inv: uint8 (E8M0 scales per 32-element block)
     └─ [columnwise variants if needed]

GEMM:
  tex.general_gemm()
  ├─ Extract MXFP8 data and E8M0 scales
  ├─ Setup cuBLASLt matmul descriptor
  │  ├─ Data format: FP8 E4M3
  │  ├─ Scaling: Block scaling (32 elements per scale)
  │  ├─ Scale format: E8M0 (power-of-2)
  │  └─ Accumulation: FP32
  │
  └─ Dispatch CUTLASS kernel
     ├─ MXFP8 GEMM computation
     ├─ E8M0 scale application (efficient power-of-2 multiply)
     └─ Dequantize to output dtype (BF16/F32)
```

---

## Part 9: Data Structures

### MXFP8Tensor Storage Layout

**File**: [transformer_engine/pytorch/tensor/storage/mxfp8_tensor_storage.py:50-257](../../../transformer_engine/pytorch/tensor/storage/mxfp8_tensor_storage.py#L50-L257)

```python
class MXFP8Tensor(MXFP8TensorStorage, QuantizedTensor):
    """Quantized tensor class with FP8 data and E8M0 scales"""

    # Storage for rowwise quantization
    _rowwise_data: torch.Tensor            # Shape: [M, K], dtype: uint8
    _rowwise_scale_inv: torch.Tensor       # Shape: [M, ceil(K/32)], dtype: uint8

    # Storage for columnwise quantization (for gradients)
    _columnwise_data: torch.Tensor         # Shape: [K, M], dtype: uint8
    _columnwise_scale_inv: torch.Tensor    # Shape: [K, ceil(M/32)], dtype: uint8

    _fp8_dtype: TE_DType                   # tex.DType.kFloat8E4M3
    _quantizer: Quantizer                  # Reference to MXFP8Quantizer
```

**Key Differences from NVFP4Tensor**:
- FP8 uses 1 byte per value (not packed like FP4's 2 values per byte)
- E8M0 scales: 1 byte per 32-element block (not E4M3+FP32 2-level scales)
- No amax storage needed (scales are self-contained)
- Simpler memory layout

### Scale Shape Calculation

For input shape `(M, K)`:

**Rowwise (block_size=32)**:
- Data shape: `(M, K)` (1 FP8 value per byte)
- Scale shape: `(M, ceil(K/32))` (1 E8M0 scale per 32 elements)

**Columnwise (after transpose)**:
- Data shape: `(K, M)` after transpose
- Scale shape: `(K, ceil(M/32))`

**No 2D variant**: MXFP8 only supports 1D block scaling

---

## Part 10: Key Constants and Configuration

### MXFP8 Constants

```python
MXFP8_BLOCK_SCALING_SIZE = 32  # Block size for quantization (fixed)

# From transformer_engine/pytorch/constants.py
class FP8FwdTensors(Enum):
    GEMM1_INPUT = 0    # Input tensor index
    GEMM1_WEIGHT = 1   # Weight tensor index
    GEMM1_OUTPUT = 2   # Output tensor index (if quantized)

class FP8BwdTensors(Enum):
    GRAD_OUTPUT1 = 0
    GRAD_INPUT1 = 1
```

### FP8 Data Type

```python
# From transformer_engine/pytorch/tensor/mxfp8_tensor.py
dtype = tex.DType.kFloat8E4M3  # 8-bit FP format, 4 exponent bits, 3 mantissa bits

# FP8 E4M3 value range
Max positive: 448.0
Min positive: 2^-9 (very small subnormal)
```

### E8M0 Scale Format

```python
# E8M0: 8-bit exponent only (power-of-2)
# Stored as biased exponent: scale_byte = exponent + 127

# Examples:
scale_byte = 127  # exponent = 0   → scale = 2^0 = 1.0
scale_byte = 130  # exponent = 3   → scale = 2^3 = 8.0
scale_byte = 120  # exponent = -7  → scale = 2^-7 = 0.0078125

# Decoding:
exponent = scale_byte - 127
scale = 2^exponent
```

### Workspace Requirements

```python
# From transformer_engine/pytorch/module/base.py:81
if compute_capability >= 10:  # Blackwell
    workspace_size = 32 * 1024 * 1024 + 1024  # 32 MiB (for MXFP8 GEMM) + 1 KiB
else:
    # MXFP8 not supported on older GPUs
    raise RuntimeError("MXFP8 requires Blackwell (CC 10.0+)")
```

---

## Part 11: Distributed Training Integration

### MXFP8 with Tensor Parallelism

**File**: [linear.py:1698-1707](../../../transformer_engine/pytorch/module/linear.py#L1698-L1707)

```python
def _customize_quantizers_mxfp8(self, fwd: bool, recipe: Recipe) -> None:
    """Customize quantizers for distributed training."""

    # MXFP8 is stateless - no amax tracking needed
    # No amax reduction across TP groups
    # No special configuration required

    # Works out-of-the-box for distributed training!
    pass
```

**Key Difference from NVFP4**: MXFP8 doesn't track amax history, so there's no need for amax reduction across distributed processes. Scales are computed independently per-call, making it much simpler for distributed training.

---

## Part 12: Backward Pass

### Gradient Computation

**File**: [linear.py:484-1006](../../../transformer_engine/pytorch/module/linear.py#L484-L1006)

Similar to forward pass but with gradient quantizers:

1. **Grad-output quantization** (for dgrad GEMM)
2. **Input quantization** (for wgrad GEMM)
3. **Output quantization** (for grad tracking)

Key differences for MXFP8:
- Same simple quantization as forward pass
- No stochastic rounding
- No random Hadamard transform
- Per-block E8M0 scaling factors for gradient tensors

**Much simpler backward pass than NVFP4!**

---

## Part 13: MXFP8 vs NVFP4 Comparison

### Complexity Comparison

| Feature | MXFP8 | NVFP4 |
|---------|-------|-------|
| **Bits per element** | 8 | 4 |
| **Block size** | 32 elements (fixed) | 16 elements (1D), 16×16 (2D) |
| **Scale format** | E8M0 (power-of-2) | E4M3 + FP32 (2-level) |
| **Random Hadamard Transform** | No | Yes (optional) |
| **2D quantization** | No | Yes (for weights) |
| **Stochastic rounding** | No | Yes (for gradients) |
| **Amax tracking** | No (stateless) | Yes (per-layer history) |
| **Distributed training config** | None needed | Amax reduction groups |
| **Quantizer complexity** | ~50 lines | ~200 lines |
| **Kernel complexity** | 1 kernel | 3+ kernels |

### Call Path Complexity

**MXFP8 quantization path**:
```
Python quantizer(tensor)
  └─ C++ nvte_quantize_mxfp8()
     └─ CUDA quantize_mxfp8 kernel
        └─ Done!
```

**NVFP4 quantization path**:
```
Python quantizer(tensor)
  └─ C++ nvte_quantize_nvfp4()
     ├─ Optional: Hadamard transform kernel
     ├─ CUDA quantize_nvfp4 kernel (1D or 2D variant)
     ├─ Optional: Stochastic rounding
     ├─ Amax computation kernel
     └─ Optional: Amax reduction across distributed group
```

**MXFP8 is ~3-5× simpler in code and execution!**

---

## Summary: Key Callpoints for MXFP8 Processing

| Step | Python Function | C++ Binding | Kernel File |
|------|-----------------|-------------|------------|
| 1. Recipe Detection | `Linear.set_meta_tensor()` | - | - |
| 2. Quantizer Init | `_customize_quantizers_mxfp8()` | - | - |
| 3. Input Quantization | `input_quantizer(inp)` | `tex.quantize()` | `quantize_mxfp8.cuh` |
| 4. Weight Quantization | `weight_quantizer(weight)` | `tex.quantize()` | `quantize_mxfp8.cuh` |
| 5. GEMM | `general_gemm()` | `tex.general_gemm()` | `cublaslt_gemm.cu` (CUTLASS) |

**That's it! No Hadamard transform, no stochastic rounding, no amax reduction.**

---

## File Reference Summary

### Python Files
- [transformer_engine/pytorch/module/linear.py](../../../transformer_engine/pytorch/module/linear.py) - te.Linear class
- [transformer_engine/pytorch/tensor/mxfp8_tensor.py](../../../transformer_engine/pytorch/tensor/mxfp8_tensor.py) - MXFP8 quantizer & tensor
- [transformer_engine/pytorch/tensor/storage/mxfp8_tensor_storage.py](../../../transformer_engine/pytorch/tensor/storage/mxfp8_tensor_storage.py) - MXFP8 storage
- [transformer_engine/common/recipe/__init__.py](../../../transformer_engine/common/recipe/__init__.py) - Recipe definitions
- [transformer_engine/pytorch/quantization.py](../../../transformer_engine/pytorch/quantization.py) - FP8 state management

### C++ Files
- [transformer_engine/pytorch/csrc/extensions/cast.cpp](../../../transformer_engine/pytorch/csrc/extensions/cast.cpp) - quantize() PyBind11 binding
- [transformer_engine/common/include/transformer_engine/cast.h](../../../transformer_engine/common/include/transformer_engine/cast.h) - nvte_quantize API

### CUDA Files
- [transformer_engine/common/cast/mxfp8/quantize_mxfp8.cuh](../../../transformer_engine/common/cast/mxfp8/quantize_mxfp8.cuh) - MXFP8 quantization kernel
- [transformer_engine/common/gemm/cublaslt_gemm.cu](../../../transformer_engine/common/gemm/cublaslt_gemm.cu) - GEMM kernel dispatch

### Test Files
- [tests/pytorch/test_sanity.py](../../../tests/pytorch/test_sanity.py) - MXFP8 inference tests
- [tests/pytorch/test_numerics.py](../../../tests/pytorch/test_numerics.py) - MXFP8 training accuracy tests
- [tests/cpp/operator/test_cast_mxfp8.cu](../../../tests/cpp/operator/test_cast_mxfp8.cu) - CUDA kernel tests

---

## Key Takeaways

1. **MXFP8 is much simpler than NVFP4**: No RHT, SR, 2D quantization, or amax tracking
2. **Stateless design**: Scales computed per-call, no history tracking
3. **Power-of-2 scales**: E8M0 format simplifies scale application in kernels
4. **Blackwell required**: CC 10.0+ for MXFP8 support
5. **Single kernel path**: Direct quantization without preprocessing
6. **Easy distributed training**: No special configuration needed
7. **Higher precision**: 8-bit vs 4-bit provides better accuracy
8. **Dual orientation storage**: Both rowwise and columnwise quantizations cached to avoid double quantization errors

---

**Related Documents**:
- [AUTOCAST_FRAME_BY_FRAME.md](AUTOCAST_FRAME_BY_FRAME.md) - Detailed execution trace
- [MXFP8_QUANTIZE_DISPATCH.md](MXFP8_QUANTIZE_DISPATCH.md) - Quantization dispatch mechanisms
- [README.md](README.md) - Complete MXFP8 reference guide
