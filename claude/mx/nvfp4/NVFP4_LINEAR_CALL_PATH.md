# te.Linear NVFP4 FP4 Call Path Analysis

## Overview

This document traces the complete call path for `te.Linear` when processing inputs under the **NVFP4BlockScaling** recipe. It covers:

1. Python API entry point (`te.Linear.forward`)
2. FP4 recipe detection and quantizer initialization
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
├── MXFP8BlockScaling
├── NVFP4BlockScaling ◄── OUR FOCUS
└── CustomRecipe
```

**File**: `/home/jeromeku/transformerengine/transformer_engine/common/recipe/__init__.py` [lines 86-515]

### NVFP4BlockScaling Recipe Definition

```python
@dataclass()
class NVFP4BlockScaling(Recipe):
    """
    Use the NVFP4 scaling strategy.
    
    This is a 2-level block scaling strategy:
    - Level 1: Each group of 16 consecutive values has its own scaling factor (E4M3)
    - Level 2: Global per-tensor FP32 scaling factor
    """
    
    # Configuration
    disable_rht: bool = False  # Random Hadamard Transform
    disable_stochastic_rounding: bool = False
    disable_2d_quantization: bool = False
    
    fp4_format: Format = Format.E2M1  # FP4 data type
    fp8_format: Format = Format.E4M3  # Scaling factor type
    
    # Quantization parameters set during __post_init__
    self.fp4_quant_fwd_inp: QParams  # Input quantization config
    self.fp4_quant_fwd_weight: QParams  # Weight quantization config
    self.fp4_quant_bwd_grad: QParams  # Gradient quantization config
```

**File**: [recipe/__init__.py:387-481](transformer_engine/common/recipe/__init__.py#L387)

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

**File**: [transformer_engine/pytorch/module/linear.py:1009-1334](transformer_engine/pytorch/module/linear.py#L1009)

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
    elif recipe.nvfp4():  # ◄── NVFP4 DETECTION
        self._customize_quantizers_nvfp4(fwd, recipe)
```

**File**: [linear.py:1335-1347](transformer_engine/pytorch/module/linear.py#L1335)

### FP4-Specific Quantizer Customization

```python
def _customize_quantizers_nvfp4(self, fwd: bool, recipe: Recipe) -> None:
    """Customize quantizers based on NVFP4 recipe + linear layer."""
    assert recipe.nvfp4(), "Incorrect recipe."
    
    if fwd:
        if self.sequence_parallel and self.parallel_mode == "column":
            # Set amax reduction group for distributed training
            self.quantizers["scaling_fwd"][
                tex.FP8FwdTensors.GEMM1_INPUT
            ].with_amax_reduction = True
            self.quantizers["scaling_fwd"][
                tex.FP8FwdTensors.GEMM1_INPUT
            ].amax_reduction_group = self.tp_group
    else:
        if self.sequence_parallel and self.parallel_mode == "row":
            # Configure backward pass quantizers
            self.quantizers["scaling_bwd"][
                tex.FP8BwdTensors.GRAD_OUTPUT1
            ].with_amax_reduction = True
            self.quantizers["scaling_bwd"][
                tex.FP8BwdTensors.GRAD_OUTPUT1
            ].amax_reduction_group = self.tp_group
```

**File**: [linear.py:1675-1696](transformer_engine/pytorch/module/linear.py#L1675)

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
    
    # 3. Get quantizers (critical for FP4!)
    quantizers = (
        self._get_quantizers(fp8_output, fp8_grad)
        if not debug
        else self._get_debug_quantizers(fp8_output, fp8_grad)
    )
    
    (
        input_quantizer,
        weight_quantizer,
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
        input_quantizer,      # NVFP4 quantizer for inputs
        weight_quantizer,     # NVFP4 quantizer for weights
        output_quantizer,
        grad_input_quantizer,
        grad_weight_quantizer,
        grad_output_quantizer,
        # ... more parameters ...
    )
    
    return out
```

**File**: [linear.py:1370-1500](transformer_engine/pytorch/module/linear.py#L1370)

### Quantizer Retrieval

```python
def _get_quantizers(self, fp8_output, fp8_grad):
    """Get quantizers from FP8 global state manager."""
    if not self.fp8:
        return [None] * 6
    
    # For NVFP4, these quantizers are NVFP4Quantizer instances
    input_quantizer = self.quantizers["scaling_fwd"][tex.FP8FwdTensors.GEMM1_INPUT]
    input_quantizer.internal = True
    
    (weight_quantizer,) = self._get_weight_quantizers()
    
    # ... setup output and gradient quantizers ...
    
    return (
        input_quantizer,        # NVFP4Quantizer ◄─── INPUT
        weight_quantizer,       # NVFP4Quantizer ◄─── WEIGHT
        output_quantizer,
        grad_input_quantizer,
        grad_weight_quantizer,
        grad_output_quantizer,
    )
```

**File**: [linear.py:1502-1526](transformer_engine/pytorch/module/linear.py#L1502)

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
        input_quantizer: Optional[Quantizer],  # ◄── NVFP4Quantizer for input
        weight_quantizer: Optional[Quantizer], # ◄── NVFP4Quantizer for weight
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
        if fp8:  # fp8 == self.fp8 (recipe is FP4)
            assert_dim_for_fp8_exec(inputmat, weight)
            
            if with_input_all_gather_nccl or ub_overlap_ag_fprop:
                # Cast local input tensor if needed
                if not isinstance(inputmat, QuantizedTensorStorage) and not experimental:
                    own_quantized_input = True
                    
                    # CRITICAL: Set quantizer usage pattern
                    input_quantizer.set_usage(
                        rowwise=True,
                        columnwise=backward_needs_input
                    )
                    
                    # QUANTIZATION KERNEL CALL #1: Quantize input
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
            weightmat,           # Quantized weight (NVFP4Tensor)
            inputmat_total,      # Quantized input (NVFP4Tensor)
            get_workspace(),     # cuBLAS workspace (32MB for Hopper)
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

**File**: [linear.py:77-482](transformer_engine/pytorch/module/linear.py#L77)

---

## Part 5: Quantizer Implementation - NVFP4Quantizer

### NVFP4Quantizer Class

**File**: [transformer_engine/pytorch/tensor/nvfp4_tensor.py:112-338](transformer_engine/pytorch/tensor/nvfp4_tensor.py#L112)

```python
class NVFP4Quantizer(Quantizer):
    """Builder class for NVFP4 tensors with NV block scaling"""
    
    dtype: TE_DType  # tex.DType.kFloat4E2M1
    with_rht: bool   # Random Hadamard Transform
    with_post_rht_amax: bool
    with_amax_reduction: bool
    amax_reduction_group: Optional[dist_group_type]
    with_2d_quantization: bool  # 2D block scaling for weights
    stochastic_rounding: bool
    rht_matrix_random_sign_mask_t: int
    rht_matrix: torch.Tensor
    
    def __init__(
        self,
        fp4_dtype: TE_DType = tex.DType.kFloat4E2M1,
        rowwise: bool = True,
        columnwise: bool = True,
        with_amax_reduction: bool = False,
        amax_reduction_group: Optional[dist_group_type] = None,
        with_rht: bool = False,
        with_post_rht_amax: bool = False,
        with_2d_quantization: bool = False,
        stochastic_rounding: bool = False,
        with_random_sign_mask: bool = True,
    ) -> None:
        super().__init__(rowwise=rowwise, columnwise=columnwise)
        self.dtype = fp4_dtype
        self.with_rht = with_rht
        self.with_post_rht_amax = with_post_rht_amax
        self.with_amax_reduction = with_amax_reduction
        self.amax_reduction_group = amax_reduction_group
        self.with_2d_quantization = with_2d_quantization
        self.stochastic_rounding = stochastic_rounding
        self.rht_matrix_random_sign_mask_t = get_random_sign_mask_for_rht(with_random_sign_mask)
        self.rht_matrix = get_rht_matrix(with_random_sign_mask)
```

### Quantization Call: `__call__` -> `quantize_impl`

When the quantizer is invoked (e.g., `inputmat = input_quantizer(inputmat)`), it calls:

```python
def quantize_impl(self, tensor: torch.Tensor) -> QuantizedTensor:
    """Quantize tensor implementation"""
    # This invokes the C++ binding
    return tex.quantize(tensor, self)
```

This routes to the **C++ quantize dispatcher**.

### NVFP4Tensor Creation

```python
def make_empty(
    self,
    shape: Iterable[int],
    *,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
) -> NVFP4Tensor:
    """Create empty NVFP4Tensor with allocated storage."""
    
    # Calculate storage shape
    # FP4 packs 2 values per byte, so last dim is halved
    data_shape = self.convert_shape_for_fp4(shape)  # [M, K//2] for rowwise
    
    # Allocate rowwise data
    data = torch.empty(data_shape, dtype=torch.uint8, device=device)
    
    # Allocate scaling factors (shape depends on block scaling)
    scale_shape = self.get_scale_shape(shape, columnwise=False)
    scale_inv = torch.empty(scale_shape, dtype=torch.uint8, device=device)
    
    # Per-tensor amax
    amax_rowwise = torch.zeros(1, dtype=torch.float32, device=device)
    
    # Similarly for columnwise if needed
    # ...
    
    return NVFP4Tensor(
        shape=shape,
        dtype=dtype,
        rowwise_data=data,
        rowwise_scale_inv=scale_inv,
        columnwise_data=columnwise_data,
        columnwise_scale_inv=columnwise_scale_inv,
        amax_rowwise=amax_rowwise,
        amax_columnwise=amax_columnwise,
        fp4_dtype=self.dtype,
        quantizer=self,
        requires_grad=requires_grad,
    )
```

---

## Part 6: C++ Bindings and Quantization Dispatch

### Entry Point: `tex.quantize` Call

**File**: [transformer_engine/pytorch/csrc/extensions/recipe.cpp]

The Python call `tex.quantize(tensor, self)` routes to C++ code:

```cpp
// Simplified representation
void quantize(
    const torch::Tensor& tensor,
    const NVFP4Quantizer& quantizer
) {
    // 1. Extract quantizer parameters
    bool with_rht = quantizer.with_rht;
    bool with_2d_quantization = quantizer.with_2d_quantization;
    bool stochastic_rounding = quantizer.stochastic_rounding;
    
    // 2. Select appropriate kernel based on configuration
    if (with_rht) {
        // Apply random Hadamard transform
        apply_random_hadamard_transform(tensor, quantizer.rht_matrix);
    }
    
    // 3. Quantize to FP4 with block scaling
    fp4_quantize_kernel(
        tensor,
        quantizer.rowwise_data,
        quantizer.rowwise_scale_inv,
        with_2d_quantization,
        stochastic_rounding
    );
    
    // 4. Compute amax values
    compute_amax_kernel(tensor, quantizer.amax_rowwise);
}
```

### Key Quantization Kernels

These CUDA kernels are invoked during quantization:

1. **Hadamard Transform**: [transformer_engine/common/hadamard_transform/hadamard_transform_cast_fusion.cu]
   - Applies random Hadamard matrix multiplication to smooth distributions
   - Only used for column-wise quantization to improve numerical properties

2. **FP4 Quantization**: [transformer_engine/common/transpose/quantize_transpose_vector_blockwise_fp4.cu]
   - Performs 16-element block scaling quantization
   - 2D variant for weight matrices (16x16 blocks)
   - Computes per-block scales in E4M3 format
   - Quantizes values to FP4 (4-bit format)

3. **Amax Computation**: Integrated in quantization kernels
   - Tracks per-block maximum absolute values
   - Used for scaling factor calculation

---

## Part 7: GEMM Execution

### `general_gemm` Call

**File**: [transformer_engine/pytorch/cpp_extensions/__init__.py]

The `general_gemm` function invokes cuBLASLt with FP4 kernel selection:

```python
def general_gemm(
    weightmat,           # NVFP4Tensor with quantized weights
    inputmat_total,      # NVFP4Tensor with quantized inputs
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
    
    # For NVFP4 tensors:
    # 1. Extract quantized data from NVFP4Tensor
    # 2. Extract scaling factors
    # 3. Call cuBLASLt with FP4 descriptor
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

**File**: [transformer_engine/common/gemm/cublaslt_gemm.cu]

Key steps:

1. **Descriptor Setup**: Configure cuBLASLt matmul descriptor with:
   - Input data types: FP4 (NVFP4)
   - Scaling modes: Block scaling (NVFP4)
   - Accumulation precision: FP32 (split accumulator)

2. **Scale Application**: 
   - Per-block FP4 scales (from quantized tensor)
   - Optional global scale for NVFP4 (from amax values)

3. **Kernel Dispatch**: cuBLASLt selects appropriate CUTLASS kernel:
   - `72a_blackwell_nvfp4_bf16_gemm.cu` (FP4 x BF16)
   - `72b_blackwell_nvfp4_nvfp4_gemm.cu` (FP4 x FP4)

4. **Output**: Result dequantized to higher precision

---

## Part 8: Complete Call Graph

### Forward Pass Call Sequence

```
Python:
  te.Linear.forward()
  ├─ Get quantizers (NVFP4Quantizer instances)
  ├─ _Linear.apply()
  │  └─ _Linear.forward()
  │     ├─ input_quantizer(inp)        ◄── Call #1: Quantize input
  │     │  └─ NVFP4Quantizer.quantize_impl()
  │     │     └─ tex.quantize(inp, self)
  │     │
  │     ├─ weight_quantizer(weight)    ◄── Call #2: Quantize weight
  │     │  └─ NVFP4Quantizer.quantize_impl()
  │     │     └─ tex.quantize(weight, self)
  │     │
  │     └─ general_gemm(
  │        weight_fp4,
  │        input_fp4,
  │        workspace,
  │        ...
  │     )   ◄── Call #3: Quantized GEMM
  │        └─ tex.general_gemm()

C++:
  tex.quantize()
  ├─ For NVFP4 with RHT:
  │  ├─ Hadamard Transform Kernel
  │  │  └─ apply_random_hadamard_transform()
  │  │
  │  ├─ FP4 Quantization Kernel (2D variant for weights)
  │  │  └─ quantize_transpose_vector_blockwise_fp4()
  │  │
  │  └─ Amax Computation
  │     └─ compute_amax()
  │
  └─ Return NVFP4Tensor
     ├─ rowwise_data: uint8 (2 FP4 values packed per byte)
     ├─ rowwise_scale_inv: uint8 (E4M3 per-block scales)
     ├─ amax_rowwise: float32
     └─ [columnwise variants if needed]

GEMM:
  tex.general_gemm()
  ├─ Extract NVFP4 data and scales
  ├─ Setup cuBLASLt matmul descriptor
  │  ├─ Data format: FP4 (NVFP4)
  │  ├─ Scaling: Block scaling
  │  └─ Accumulation: FP32
  │
  └─ Dispatch CUTLASS kernel
     ├─ FP4 GEMM computation
     ├─ Scale application (per-block + global)
     └─ Dequantize to output dtype (BF16/F32)
```

---

## Part 9: Data Structures

### NVFP4Tensor Storage Layout

**File**: [transformer_engine/pytorch/tensor/storage/nvfp4_tensor_storage.py]

```python
class NVFP4Tensor(NVFP4TensorStorage, QuantizedTensor):
    """Quantized tensor class with FP4 data"""
    
    # Storage for rowwise quantization
    _rowwise_data: torch.Tensor            # Shape: [M, K//2], dtype: uint8
    _rowwise_scale_inv: torch.Tensor       # Shape: swizzled, dtype: uint8
    _amax_rowwise: torch.Tensor            # Shape: [1], dtype: float32
    
    # Storage for columnwise quantization (for gradients)
    _columnwise_data: torch.Tensor         # Shape: [K, M//2], dtype: uint8
    _columnwise_scale_inv: torch.Tensor    # Shape: swizzled, dtype: uint8
    _amax_columnwise: torch.Tensor         # Shape: [1], dtype: float32
    
    _fp4_dtype: TE_DType                   # tex.DType.kFloat4E2M1
    _quantizer: Quantizer                  # Reference to NVFP4Quantizer
```

### Scale Shape Calculation

For input shape `(M, K)`:

**Rowwise (1D block scaling, block_size=16)**:
- Data shape: `(M, K//2)` (2 FP4 values per byte)
- Scale shape: `(round_up_to_128(M), round_up_to_4(ceil(K/16)))`

**Columnwise (1D block scaling after transpose)**:
- Data shape: `(K, M//2)` after transpose
- Scale shape: `(round_up_to_128(K), round_up_to_4(ceil(M/16)))`

**2D block scaling (for weights, 16x16 blocks)**:
- Data shape: `(M, K//2)`
- Scale shape: `(round_up_to_128(M), round_up_to_4(ceil(K/16)))` per-block scale

---

## Part 10: Key Constants and Configuration

### NVFP4 Constants

```python
NVFP4_BLOCK_SCALING_SIZE = 16  # Block size for quantization

# From transformer_engine/pytorch/constants.py
class FP8FwdTensors(Enum):
    GEMM1_INPUT = 0    # Input tensor index
    GEMM1_WEIGHT = 1   # Weight tensor index
    GEMM1_OUTPUT = 2   # Output tensor index (if quantized)

class FP8BwdTensors(Enum):
    GRAD_OUTPUT1 = 0
    GRAD_INPUT1 = 1
```

### FP4 Data Type

```cpp
// From transformer_engine/pytorch/tensor/nvfp4_tensor.py
dtype = tex.DType.kFloat4E2M1  // 4-bit FP format, 2 exponent bits, 1 mantissa bit

// FP4 E2M1 value range
Max positive: 6.0
Min positive: 0.5
```

### Workspace Requirements

```cpp
// From transformer_engine/pytorch/module/base.py:81
if compute_capability >= 9:  // Hopper
    workspace_size = 32 * 1024 * 1024 + 1024  // 32 MiB (for NVFP4 GEMM) + 1 KiB
else:
    workspace_size = 4_194_304  // 4 MiB
```

---

## Part 11: Distributed Training Integration

### FP4 with Tensor Parallelism

**File**: [linear.py:1679-1695]

```python
def _customize_quantizers_nvfp4(self, fwd: bool, recipe: Recipe) -> None:
    """Customize quantizers for distributed training."""
    
    if fwd:
        if self.sequence_parallel and self.parallel_mode == "column":
            # Column-parallel: reduce amax across TP group
            self.quantizers["scaling_fwd"][
                tex.FP8FwdTensors.GEMM1_INPUT
            ].with_amax_reduction = True
            self.quantizers["scaling_fwd"][
                tex.FP8FwdTensors.GEMM1_INPUT
            ].amax_reduction_group = self.tp_group
    else:
        if self.sequence_parallel and self.parallel_mode == "row":
            # Row-parallel: reduce amax across TP group
            self.quantizers["scaling_bwd"][
                tex.FP8BwdTensors.GRAD_OUTPUT1
            ].with_amax_reduction = True
            self.quantizers["scaling_bwd"][
                tex.FP8BwdTensors.GRAD_OUTPUT1
            ].amax_reduction_group = self.tp_group
```

This ensures amax values are synchronized across distributed processes for consistent scaling.

---

## Part 12: Backward Pass

### Gradient Computation

**File**: [linear.py:484-1006]

Similar to forward pass but with gradient quantizers:

1. **Grad-output quantization** (for dgrad GEMM)
2. **Input quantization** (for wgrad GEMM)
3. **Output quantization** (for grad tracking)

Key differences for NVFP4:
- Stochastic rounding applied to gradients (if enabled)
- Optional random Hadamard transform for gradient inputs
- Per-block scaling factors for gradient tensors

---

## Summary: Key Callpoints for FP4 Processing

| Step | Python Function | C++ Binding | Kernel File |
|------|-----------------|-------------|------------|
| 1. Recipe Detection | `Linear.set_meta_tensor()` | - | - |
| 2. Quantizer Init | `_customize_quantizers_nvfp4()` | - | - |
| 3. Input Quantization | `input_quantizer(inp)` | `tex.quantize()` | `quantize_transpose_vector_blockwise_fp4.cu` |
| 4. Weight Quantization | `weight_quantizer(weight)` | `tex.quantize()` | `quantize_transpose_vector_blockwise_fp4.cu` |
| 5. Hadamard Transform | - | `tex.quantize()` | `hadamard_transform_cast_fusion.cu` |
| 6. GEMM | `general_gemm()` | `tex.general_gemm()` | `cublaslt_gemm.cu` (CUTLASS) |
| 7. Scale Computation | - | `nvte_nvfp4_compute_per_tensor_scale()` | `nvfp4.cu` |

---

## File Reference Summary

### Python Files
- `/home/jeromeku/transformerengine/transformer_engine/pytorch/module/linear.py` - te.Linear class
- `/home/jeromeku/transformerengine/transformer_engine/pytorch/tensor/nvfp4_tensor.py` - NVFP4 quantizer & tensor
- `/home/jeromeku/transformerengine/transformer_engine/common/recipe/__init__.py` - Recipe definitions
- `/home/jeromeku/transformerengine/transformer_engine/pytorch/quantization.py` - FP8 state management

### CUDA/C++ Files
- `/home/jeromeku/transformerengine/transformer_engine/common/recipe/nvfp4.cu` - Per-tensor scale computation
- `/home/jeromeku/transformerengine/transformer_engine/common/transpose/quantize_transpose_vector_blockwise_fp4.cu` - FP4 quantization
- `/home/jeromeku/transformerengine/transformer_engine/common/gemm/cublaslt_gemm.cu` - GEMM kernel dispatch
- `/home/jeromeku/transformerengine/transformer_engine/common/hadamard_transform/hadamard_transform_cast_fusion.cu` - RHT

