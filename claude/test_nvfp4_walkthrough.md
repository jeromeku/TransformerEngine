# NVFP4 Test Walkthrough

> **Purpose**: Annotated line-by-line walkthrough of NVFP4 quantization tests, tracing the complete call stack from test setup through Python APIs to C++ bindings and CUDA kernels.

## Table of Contents

1. [Test Overview](#test-overview)
2. [Test: test_nvfp4_gemm_versus_reference](#test-test_nvfp4_gemm_versus_reference)
3. [Complete Call Stack Trace](#complete-call-stack-trace)
4. [Key Concepts Demonstrated](#key-concepts-demonstrated)

---

## Test Overview

**Test File**: [`tests/pytorch/nvfp4/test_nvfp4_gemm_exact.py`](../../tests/pytorch/nvfp4/test_nvfp4_gemm_exact.py)

**Purpose**: Verify that TransformerEngine's native NVFP4 GEMM (using cuBLASLt) produces identical results to a reference implementation.

**Key Components Tested**:
1. NVFP4 quantization (Python → C++ → CUDA)
2. cuBLASLt GEMM with FP4 + 2-level block scaling
3. Correctness of quantized data layout and scales
4. Tensor unpacking and scale interpretation

---

## Test: test_nvfp4_gemm_versus_reference

### Test Function Signature

[`test_nvfp4_gemm_exact.py:221-242`](../../tests/pytorch/nvfp4/test_nvfp4_gemm_exact.py#L221-L242)

```python
@pytest.mark.parametrize("M, K, N", [(128, 128, 128), (256, 128, 256), ...])
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize("w_dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32], ids=str)
@pytest.mark.parametrize("accumulate", [True, False])
@pytest.mark.parametrize("is_x_columnwise, is_w_columnwise", [(False, False)])
def test_nvfp4_gemm_versus_reference(
    M, K, N, x_dtype, w_dtype, out_dtype, accumulate,
    is_x_columnwise, is_w_columnwise
):
    check_nvfp4_gemm_versus_reference(...)
```

**What it tests**: GEMM operation `Y = W @ X + (accumulate ? Out : 0)` where W and X are quantized to NVFP4.

**Parametrization**:
- **Matrix sizes**: Various shapes from 128×128×128 to 1024×1024×1024
- **Input dtypes**: FP32 and BF16
- **Output dtype**: BF16 or FP32
- **Accumulate mode**: Test both `Y = W @ X` and `Y += W @ X`
- **Layouts**: Currently only row-wise × row-wise (reference limitation)

---

### Test Implementation: check_nvfp4_gemm_versus_reference

**Location**: [`test_nvfp4_gemm_exact.py:18-189`](../../tests/pytorch/nvfp4/test_nvfp4_gemm_exact.py#L18-L189)

Let's trace this step-by-step with inline code and source links.

---

#### Step 1: Test Setup

[`test_nvfp4_gemm_exact.py:30-48`](../../tests/pytorch/nvfp4/test_nvfp4_gemm_exact.py#L30-L48)

```python
def check_nvfp4_gemm_versus_reference(...):
    te_dtype = tex.DType.kFloat4E2M1  # NVFP4 E2M1 format

    # Setup device and seed for reproducibility
    device = "cuda"
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Create input tensors
    x_shape = (K, M) if x_columnwise else (M, K)
    w_shape = (K, N) if w_columnwise else (N, K)
    x = torch.randn(x_shape, dtype=x_dtype, device=device)
    w = torch.randn(w_shape, dtype=w_dtype, device=device)

    # Setup output tensor if accumulating
    if accumulate:
        out = torch.randn((M, N), dtype=out_dtype, device=device)
    else:
        out = None
```

**Key Points**:
- Uses standard PyTorch tensor creation
- Shapes vary based on layout (row-wise vs column-wise)
- For this test: `x_columnwise=False`, `w_columnwise=False`, so shapes are `(M, K)` and `(N, K)`

**Data Flow**:
```
User code creates BF16/FP32 tensors
    ↓
x: [M=256, K=128] in BF16
w: [N=256, K=128] in BF16
```

---

#### Step 2: Create Native TE Quantizers

[`test_nvfp4_gemm_exact.py:50-68`](../../tests/pytorch/nvfp4/test_nvfp4_gemm_exact.py#L50-L68)

```python
# Native TE NVFP4 quantization
x_quantizer = NVFP4Quantizer(
    fp4_dtype=te_dtype,               # kFloat4E2M1
    rowwise=True,                     # Enable rowwise layout
    columnwise=True,                  # Enable columnwise layout
    with_amax_reduction=False,        # No multi-GPU amax sync
    amax_reduction_group=None,
    with_rht=False,                   # No Random Hadamard Transform
    with_post_rht_amax=False,
)
w_quantizer = NVFP4Quantizer(
    fp4_dtype=te_dtype,
    rowwise=True,
    columnwise=True,
    with_amax_reduction=False,
    amax_reduction_group=None,
    with_rht=False,
    with_post_rht_amax=False,
)
```

**What this creates**:
- Two `NVFP4Quantizer` instances (one for activations, one for weights)
- Configuration: Basic NVFP4 without RHT or stochastic rounding (for exact testing)
- Both layouts enabled since we may need either depending on GEMM orientation

**Call Stack**:
```
Python: NVFP4Quantizer.__init__()
    Location: transformer_engine/pytorch/tensor/nvfp4_tensor.py:133-156
    ↓
Initializes:
  - self.dtype = tex.DType.kFloat4E2M1
  - self.with_rht = False
  - self.rht_matrix = get_rht_matrix(False)  # No random signs
  - self.rht_matrix_random_sign_mask_t = 0
  - self.with_2d_quantization = False  # Default: 1D blocks (16 elements)
  - self.stochastic_rounding = False
```

---

#### Step 3: Allocate Empty Quantized Tensors

[`test_nvfp4_gemm_exact.py:71-78`](../../tests/pytorch/nvfp4/test_nvfp4_gemm_exact.py#L71-L78)

```python
# Quantize x and w
x_nvfp4_native = x_quantizer.make_empty(
    x_shape, dtype=x_dtype, device=device, requires_grad=False
)
x_nvfp4_native = x_quantizer.update_quantized(x, x_nvfp4_native)

w_nvfp4_native = w_quantizer.make_empty(
    w_shape, dtype=w_dtype, device=device, requires_grad=False
)
w_nvfp4_native = w_quantizer.update_quantized(w, w_nvfp4_native)
```

**Call Stack for `make_empty()`**:

```
Python: x_quantizer.make_empty((M, K), dtype=BF16, device=cuda, requires_grad=False)
    Location: transformer_engine/pytorch/tensor/nvfp4_tensor.py:261-328
    ↓
Validates shape:
  assert shape[-1] % 16 == 0  # K must be divisible by 16
  assert math.prod(shape[:-1]) % 16 == 0  # M must be divisible by 16
    ↓
Allocates rowwise buffers:
  rowwise_data = torch.empty([M, K/2], dtype=uint8, device=cuda)
      # FP4 packed 2 per byte

  scale_shape = get_scale_shape((M, K), columnwise=False)
      # Returns: [(M+127)/128, (ceil(K/16)+3)/4]
      # For M=256, K=128: [256, 2] → padded [256, 2]
  rowwise_scale_inv = torch.empty(scale_shape, dtype=uint8, device=cuda)
      # E4M3 block scales

  amax_rowwise = torch.zeros([1], dtype=float32, device=cuda)
      # Tensor-level FP32 scale
    ↓
Allocates columnwise buffers (similar):
  columnwise_data = torch.empty([K, M/2], dtype=uint8, device=cuda)
  columnwise_scale_inv = torch.empty(columnwise_scale_shape, dtype=uint8, device=cuda)
  amax_columnwise = torch.zeros([1], dtype=float32, device=cuda)
    ↓
Constructs NVFP4Tensor:
  Location: transformer_engine/pytorch/tensor/nvfp4_tensor.py:316-328

  return NVFP4Tensor(
      shape=(M, K),
      dtype=BF16,
      rowwise_data=rowwise_data,
      rowwise_scale_inv=rowwise_scale_inv,
      columnwise_data=columnwise_data,
      columnwise_scale_inv=columnwise_scale_inv,
      amax_rowwise=amax_rowwise,
      amax_columnwise=amax_columnwise,
      fp4_dtype=kFloat4E2M1,
      quantizer=self,
      requires_grad=False,
  )
```

**Result**: `x_nvfp4_native` is now a `NVFP4Tensor` wrapper with **uninitialized** data buffers.

---

#### Step 4: Quantize Input Data

[`test_nvfp4_gemm_exact.py:74`](../../tests/pytorch/nvfp4/test_nvfp4_gemm_exact.py#L74)

```python
x_nvfp4_native = x_quantizer.update_quantized(x, x_nvfp4_native)
```

**Call Stack**:

```
Python: x_quantizer.update_quantized(x, x_nvfp4_native)
    Location: transformer_engine/pytorch/tensor/nvfp4_tensor.py:157-176
    ↓
Checks:
  assert isinstance(x_nvfp4_native, NVFP4Tensor)
  if not devices_match(x.device, x_nvfp4_native.device):
      x = x.to(device=x_nvfp4_native.device)
  if not x.is_contiguous():
      x = x.contiguous()
    ↓
Calls C++ via PyBind11:
  tex.quantize(x, self, x_nvfp4_native, noop_flag=None)
      Location (Python): Imported from transformer_engine_torch
      Location (C++): transformer_engine/pytorch/csrc/extensions/pybind.cpp:119
    ↓
C++ Function: transformer_engine::pytorch::quantize()
    ↓
Extracts quantizer configuration:
  bool rowwise = quantizer.attr("rowwise").cast<bool>()  // True
  bool columnwise = quantizer.attr("columnwise").cast<bool>()  // True
  bool with_rht = quantizer.attr("with_rht").cast<bool>()  // False
  bool with_2d = quantizer.attr("with_2d_quantization").cast<bool>()  // False
  bool stochastic = quantizer.attr("stochastic_rounding").cast<bool>()  // False
  torch::Tensor rht_matrix = quantizer.attr("rht_matrix").cast<torch::Tensor>()
  int rht_sign_mask = quantizer.attr("rht_matrix_random_sign_mask_t").cast<int>()  // 0
    ↓
Extracts tensor pointers:
  void* src_data = x.data_ptr()  // [M, K] in BF16

  void* dst_rowwise_data = x_nvfp4_native.attr("_rowwise_data").data_ptr()  // [M, K/2] uint8
  void* dst_rowwise_scale = x_nvfp4_native.attr("_rowwise_scale_inv").data_ptr()  // E4M3 scales
  void* dst_rowwise_amax = x_nvfp4_native.attr("_amax_rowwise").data_ptr()  // FP32 amax

  void* dst_columnwise_data = x_nvfp4_native.attr("_columnwise_data").data_ptr()
  void* dst_columnwise_scale = x_nvfp4_native.attr("_columnwise_scale_inv").data_ptr()
  void* dst_columnwise_amax = x_nvfp4_native.attr("_amax_columnwise").data_ptr()
    ↓
Launches CUDA kernel:
  nvte_nvfp4_quantize(
      src_data, src_shape,
      dst_rowwise_data, dst_rowwise_scale, dst_rowwise_amax,
      dst_columnwise_data, dst_columnwise_scale, dst_columnwise_amax,
      with_rht, rht_matrix_ptr, rht_sign_mask,
      stochastic,
      stream
  );
    Location: transformer_engine/common/recipe/nvfp4.cu
    ↓
CUDA Kernel: nvfp4_quantize_kernel<<<...>>>(...)
    Pseudo-code (simplified for clarity):

    __global__ void nvfp4_quantize_kernel(
        const __nv_bfloat16* src,  // [M, K]
        uint8_t* dst_data,          // [M, K/2] packed FP4
        uint8_t* dst_scale,         // E4M3 block scales
        float* dst_amax,            // FP32 tensor scale
        int M, int K
    ) {
        int tidx = blockIdx.x * blockDim.x + threadIdx.x;
        int num_blocks = M * (K / 16);  // 16-element blocks

        if (tidx >= num_blocks) return;

        int row = tidx / (K / 16);
        int block_col = tidx % (K / 16);
        int col_start = block_col * 16;

        // 1. Load 16 elements from src
        __nv_bfloat16 vals[16];
        for (int i = 0; i < 16; i++) {
            vals[i] = src[row * K + col_start + i];
        }

        // 2. Compute amax of block
        float amax_block = 0.0f;
        for (int i = 0; i < 16; i++) {
            amax_block = fmaxf(amax_block, fabsf(__bfloat162float(vals[i])));
        }

        // 3. Update tensor-level amax (global reduction)
        atomicMaxFloat(dst_amax, amax_block);

        // 4. Compute block-level E4M3 scale
        float fp4_max = 6.0f;  // E2M1 max value
        float block_scale_fp32 = amax_block / fp4_max;
        uint8_t block_scale_e4m3 = float_to_fp8_e4m3(block_scale_fp32);

        // 5. Store block scale
        int scale_idx = row * (K/16) + block_col;
        dst_scale[scale_idx] = block_scale_e4m3;

        // 6. Quantize 16 values to FP4 (using 2-level scale)
        // Note: Tensor scale computed in a separate pass or post-processing
        float tensor_scale_inv = *dst_amax;  // Read tensor amax
        float combined_scale = fp8_e4m3_to_float(block_scale_e4m3) * tensor_scale_inv;

        for (int i = 0; i < 16; i += 2) {
            float val0 = __bfloat162float(vals[i]) / combined_scale;
            float val1 = __bfloat162float(vals[i+1]) / combined_scale;

            // Quantize to FP4 E2M1
            uint8_t fp4_0 = float_to_fp4_e2m1(val0);
            uint8_t fp4_1 = float_to_fp4_e2m1(val1);

            // Pack two FP4 values into one byte
            // [high 4 bits: fp4_1 | low 4 bits: fp4_0]
            uint8_t packed = (fp4_1 << 4) | fp4_0;

            int data_idx = row * (K/2) + (col_start + i) / 2;
            dst_data[data_idx] = packed;
        }
    }

    // Similar kernel for columnwise layout (transpose + quantize)
    ↓
Returns to Python:
  x_nvfp4_native now has populated:
    - _rowwise_data: Quantized FP4 data
    - _rowwise_scale_inv: E4M3 block scales
    - _amax_rowwise: FP32 tensor scale
    - (same for columnwise)
```

**Result**: `x_nvfp4_native` and `w_nvfp4_native` are fully quantized NVFP4 tensors.

---

#### Step 5: Extract Quantized Data for Reference GEMM

[`test_nvfp4_gemm_exact.py:80-112`](../../tests/pytorch/nvfp4/test_nvfp4_gemm_exact.py#L80-L112)

```python
# Extract quantized data from native NVFP4Tensors
qx_data = (
    x_nvfp4_native._columnwise_data.view(dtype=torch.uint8)
    if x_columnwise
    else x_nvfp4_native._rowwise_data.view(dtype=torch.uint8)
)
qw_data = (
    w_nvfp4_native._columnwise_data.view(dtype=torch.uint8)
    if w_columnwise
    else w_nvfp4_native._rowwise_data.view(dtype=torch.uint8)
)
sx_native = (
    x_nvfp4_native._columnwise_scale_inv if x_columnwise
    else x_nvfp4_native._rowwise_scale_inv
)
sw_native = (
    w_nvfp4_native._columnwise_scale_inv if w_columnwise
    else w_nvfp4_native._rowwise_scale_inv
)

# Trim quantized data to remove padding
qx_data = qx_data[:M, :]  # Keep only M rows
qw_data = qw_data[:N, :]  # Keep only N rows

# NVFP4 uses 16-element blocks, trim scales to remove padding
block_length = 16
expected_sx_cols = expected_sw_cols = K // block_length  # K/16 scales

# Trim the scales to remove padding (inner dim padded to multiple of 4)
sx_trimmed = sx_native[:M, :expected_sx_cols]
sw_trimmed = sw_native[:N, :expected_sw_cols]

# Native scales are stored as uint8 but need to be interpreted as float8_e4m3fn
# for the reference GEMM to work correctly
sx_trimmed = sx_trimmed.view(torch.float8_e4m3fn)
sw_trimmed = sw_trimmed.view(torch.float8_e4m3fn)
```

**What's happening**:
1. **Select layout**: Since test uses rowwise×rowwise, extract `_rowwise_data` and `_rowwise_scale_inv`
2. **Trim data**: Remove padding (TE pads dimensions, but reference expects exact sizes)
3. **Trim scales**:
   - TE stores scales with shape `[(M+127)/128, (ceil(K/16)+3)/4]` (padded for cuBLAS)
   - Reference expects `[M, K/16]`
   - So trim to `[:M, :K/16]`
4. **Reinterpret scale dtype**:
   - TE stores E4M3 as `uint8`
   - Reference expects `torch.float8_e4m3fn`
   - `.view()` reinterprets bytes without changing data

**Data after this step**:
```
qx_data:    [M, K/2] uint8 (packed FP4)
qw_data:    [N, K/2] uint8 (packed FP4)
sx_trimmed: [M, K/16] float8_e4m3fn (block scales)
sw_trimmed: [N, K/16] float8_e4m3fn (block scales)
```

---

#### Step 6: Create Reference Quantizer and Quantized Tensors

[`test_nvfp4_gemm_exact.py:114-126`](../../tests/pytorch/nvfp4/test_nvfp4_gemm_exact.py#L114-L126)

```python
# Create reference quantizer for reference GEMM
ref_quantizer = NVFP4QuantizerRef(
    dtype=utils.Fp4Formats.E2M1,
    rowwise=True,
    columnwise=True,
    pow_2_scales=False,
    eps=0.0,
    quant_tile_shape=(1, 16),  # 1D blocks of 16 elements
)

# Create reference quantized tensors needed by reference GEMM
x_nvfp4_ref = ref_quantizer.quantize(x)
w_nvfp4_ref = ref_quantizer.quantize(w)
```

**What this does**:
- `NVFP4QuantizerRef` is a **reference implementation** (pure Python/PyTorch) for testing
- Located in: [`transformer_engine/pytorch/experimental/quantization_nvfp4.py`](../../transformer_engine/pytorch/experimental/quantization_nvfp4.py)
- Purpose: Compute ground truth quantization to compare against native TE implementation
- `x_nvfp4_ref` and `w_nvfp4_ref` hold reference quantized tensors with known-correct values

---

#### Step 7: Reference GEMM

[`test_nvfp4_gemm_exact.py:128-142`](../../tests/pytorch/nvfp4/test_nvfp4_gemm_exact.py#L128-L142)

```python
# Reference GEMM using quantizer's qgemm method
y_ref = ref_quantizer.qgemm(
    qx=qx_data,                          # [M, K/2] packed FP4 from TE
    qw=qw_data,                          # [N, K/2] packed FP4 from TE
    m_params=None,                       # Not used
    out_dtype=out_dtype,                 # BF16 or FP32
    sx=sx_trimmed,                       # [M, K/16] E4M3 scales from TE
    sw=sw_trimmed,                       # [N, K/16] E4M3 scales from TE
    bias=None,                           # No bias
    out=out.clone() if accumulate else None,  # Accumulation target
    accumulate=accumulate,               # True/False
    gemm_type=None,                      # Not used
    qresult_x=x_nvfp4_ref,              # Reference quantized X (for tensor scales)
    qresult_w=w_nvfp4_ref,              # Reference quantized W (for tensor scales)
)
```

**What `qgemm()` does** (simplified):
```python
def qgemm(self, qx, qw, sx, sw, qresult_x, qresult_w, ...):
    # 1. Unpack FP4 data to FP32
    x_fp32 = unpack_fp4(qx)  # [M, K]
    w_fp32 = unpack_fp4(qw)  # [N, K]

    # 2. Apply 2-level scaling
    #    Scale format: block_scale (E4M3) * tensor_scale (FP32)
    for i in range(M):
        for k in range(K):
            block_idx = k // 16
            x_fp32[i, k] *= sx[i, block_idx]  # Block scale
            x_fp32[i, k] *= qresult_x.tensor_scale  # Tensor scale

    # Similar for w_fp32

    # 3. Perform FP32 GEMM
    y_ref = x_fp32 @ w_fp32.T  # [M, N]

    # 4. Accumulate if needed
    if accumulate:
        y_ref += out

    # 5. Cast to output dtype
    y_ref = y_ref.to(out_dtype)

    return y_ref
```

**Result**: `y_ref` is the reference output computed in high precision.

---

#### Step 8: Native TE cuBLAS GEMM

[`test_nvfp4_gemm_exact.py:144-178`](../../tests/pytorch/nvfp4/test_nvfp4_gemm_exact.py#L144-L178)

```python
# Native TE GEMM using tex.generic_gemm (cuBLAS GEMM)
workspace = torch.empty(4, dtype=torch.uint8, device=device)

transa = True if not w_columnwise else False  # True (need to transpose W)
transb = False if not x_columnwise else True  # False
out_quantizer = None
bias = None
bias_dtype = TE_DType[torch.bfloat16]
use_gelu = False
gelu_input = None
use_grad = False
use_split_accumulator = False

# Native cuBLAS GEMM
# Returns: (out, bias_grad, gelu_input, extra_output)
# We only need out
y_native = tex.generic_gemm(
    w_nvfp4_native,                      # A matrix (weight) [N, K] in NVFP4
    transa,                              # True (transpose to [K, N])
    x_nvfp4_native,                      # B matrix (input) [M, K] in NVFP4
    transb,                              # False (no transpose)
    out.clone() if accumulate else None, # D (output/accumulation)
    out_quantizer,                       # None (output in BF16/FP32)
    TE_DType[out_dtype],                # Output dtype
    bias,                                # None
    bias_dtype,                          # BF16 (unused)
    use_gelu,                            # False
    gelu_input,                          # None
    use_grad,                            # False
    workspace,                           # cuBLAS workspace
    workspace.shape[0],                  # Workspace size
    accumulate,                          # True/False
    use_split_accumulator,               # False
)[0]  # Extract output tensor
```

**Call Stack**:

```
Python: tex.generic_gemm(w_nvfp4_native, True, x_nvfp4_native, False, ...)
    Location (binding): transformer_engine/pytorch/csrc/extensions/pybind.cpp:126
    ↓
C++: transformer_engine::pytorch::gemm(...)
    ↓
Determine GEMM type:
  Input A: NVFP4Tensor
  Input B: NVFP4Tensor
  → gemm_type = NVFP4_GEMM
    ↓
Dispatch to gemm_nvfp4():
  Extract data pointers:
    void* A_data = (transA ? w._columnwise_data : w._rowwise_data).data_ptr()
        // Since transA=True, use columnwise_data
    void* A_scale = (transA ? w._columnwise_scale_inv : w._rowwise_scale_inv).data_ptr()
    void* A_amax = (transA ? w._amax_columnwise : w._amax_rowwise).data_ptr()

    // Similar for B (x)
    ↓
Setup cuBLASLt operation descriptor:
  cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

  // Set matrix A (weight) type
  cudaDataType_t fp4_type = CUDA_R_4F_E2M1;  // FP4 E2M1
  cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_A_TYPE,
      &fp4_type, sizeof(fp4_type)
  );

  // Set matrix B (input) type
  cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_B_TYPE,
      &fp4_type, sizeof(fp4_type)
  );

  // Provide block scale pointers (E4M3)
  cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
      &A_scale, sizeof(void*)
  );
  cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
      &B_scale, sizeof(void*)
  );

  // Set scale type (E4M3)
  cudaDataType_t scale_type = CUDA_R_8F_E4M3;
  cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_TYPE,
      &scale_type, sizeof(scale_type)
  );
  cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_TYPE,
      &scale_type, sizeof(scale_type)
  );

  // Provide tensor-level amax pointers (FP32)
  cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_AMAX_POINTER,
      &A_amax, sizeof(void*)
  );
  cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_BMAX_POINTER,
      &B_amax, sizeof(void*)
  );

  // Enable split-K (required for FP4)
  // Note: test has use_split_accumulator=False, but cuBLAS may still use split-K internally
  int split_k = 1;
  cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_SPLIT_K,
      &split_k, sizeof(split_k)
  );
    ↓
Setup matrix layouts:
  // A: [N, K] transposed → effective [K, N] for GEMM
  cublasLtMatrixLayoutCreate(&A_layout, fp4_type, K, N, K);

  // B: [M, K] not transposed
  cublasLtMatrixLayoutCreate(&B_layout, fp4_type, K, M, K);

  // C: [M, N]
  cublasLtMatrixLayoutCreate(&C_layout, output_dtype, N, M, N);
    ↓
Allocate output if needed:
  if (!D.defined()) {
      D = torch::empty({M, N}, torch::dtype(output_dtype).device(cuda));
  }
    ↓
Execute cuBLASLt GEMM:
  float alpha = 1.0f;
  float beta = accumulate ? 1.0f : 0.0f;  // beta=1 for accumulation

  cublasLtMatmul(
      cublas_lt_handle,
      op_desc,
      &alpha,
      A_data, A_layout,  // Weight FP4 data
      B_data, B_layout,  // Input FP4 data
      &beta,
      D.data_ptr(), C_layout,  // Output BF16/FP32
      workspace.data_ptr(), workspace_size,
      stream
  );
    ↓
GPU Execution (Blackwell Tensor Cores):
  For each output element C[i, j]:
    1. Load FP4 values from A and B
    2. Unpack FP4 → intermediate format
    3. Load block scales (E4M3) and tensor scales (FP32)
    4. Apply 2-level scaling: value * block_scale * tensor_scale
    5. Accumulate in FP32
    6. Apply beta (accumulation if beta=1)
    7. Cast to output dtype (BF16/FP32)
    8. Store C[i, j]
    ↓
Returns to Python:
  y_native: [M, N] in out_dtype (BF16 or FP32)
```

**Result**: `y_native` is the output from cuBLASLt FP4 GEMM.

---

#### Step 9: Compare Results

[`test_nvfp4_gemm_exact.py:180-188`](../../tests/pytorch/nvfp4/test_nvfp4_gemm_exact.py#L180-L188)

```python
# Ensure y_ref and y_native are different tensors
assert y_ref is not y_native, "y_ref and y_native should not be the same tensor"

# Reset nans to zeros (torch.assert_close treats nans as not equal)
assert not torch.isnan(y_ref.float()).all(), "All elements are nan"
y_ref = torch.where(y_ref.isnan(), torch.zeros_like(y_ref), y_ref)
y_native = torch.where(y_native.isnan(), torch.zeros_like(y_native), y_native)

# Compare results with tolerance
torch.testing.assert_close(y_native, y_ref, atol=8e-3, rtol=8e-3)
```

**What this validates**:
- Native cuBLASLt GEMM matches reference implementation
- Tolerance: absolute=8e-3, relative=8e-3
  - Accounts for:
    1. Quantization error (FP4 has very limited precision)
    2. Different accumulation order (cuBLAS may use different blocking)
    3. FP32 vs BF16 intermediate precision

**If this passes**: TE's NVFP4 implementation is correct!

---

## Complete Call Stack Trace

### Quantization Path

```
User Test Code
    ↓
quantizer = NVFP4Quantizer(...)
    Python: transformer_engine/pytorch/tensor/nvfp4_tensor.py:133-156
    ↓
tensor_fp4 = quantizer.quantize(tensor_bf16)
    Python: transformer_engine/pytorch/tensor/nvfp4_tensor.py:178-180
    ↓
quantizer.quantize_impl(tensor_bf16)
    Python: transformer_engine/pytorch/tensor/nvfp4_tensor.py:178-180
    ↓
tex.quantize(tensor_bf16, quantizer)
    Python: Imported from transformer_engine_torch
    C++ binding: transformer_engine/pytorch/csrc/extensions/pybind.cpp:119
    ↓
transformer_engine::pytorch::quantize(tensor, quantizer, output, noop)
    C++: transformer_engine/pytorch/csrc/extensions.cpp (implementation)
    ↓
Dispatch to NVFP4 quantization based on quantizer type
    C++: Check isinstance(quantizer, NVFP4Quantizer)
    ↓
Extract configuration from quantizer:
    - fp4_dtype, rowwise, columnwise
    - with_rht, with_2d_quantization, stochastic_rounding
    - rht_matrix, rht_sign_mask
    ↓
Extract tensor pointers:
    - src_data, dst_rowwise_data, dst_rowwise_scale, dst_amax_rowwise
    - dst_columnwise_data, dst_columnwise_scale, dst_amax_columnwise
    ↓
nvte_nvfp4_quantize(...)
    C++: Call into TE common library
    CUDA: transformer_engine/common/recipe/nvfp4.cu
    ↓
nvfp4_quantize_kernel<<<...>>>()
    CUDA Kernel Launch
    ↓
Kernel Execution (per thread):
    1. Load 16-element block from input
    2. Apply RHT if enabled (matrix multiply)
    3. Compute block amax
    4. Update tensor amax (atomic operation)
    5. Compute E4M3 block scale
    6. Quantize 16 values to FP4 E2M1
    7. Pack 2 FP4 values per byte
    8. Store quantized data and scales
    ↓
GPU Synchronization
    ↓
Return to Python: NVFP4Tensor populated with quantized data
```

### GEMM Path

```
User Test Code
    ↓
tex.generic_gemm(w_nvfp4, True, x_nvfp4, False, ...)
    C++ binding: transformer_engine/pytorch/csrc/extensions/pybind.cpp:126
    ↓
transformer_engine::pytorch::gemm(A, transA, B, transB, D, ...)
    C++: transformer_engine/pytorch/csrc/extensions.cpp
    ↓
Infer GEMM type from input tensors:
    Check A type: NVFP4Tensor
    Check B type: NVFP4Tensor
    → gemm_type = NVFP4_GEMM
    ↓
gemm_nvfp4(A, transA, B, transB, D, ...)
    C++: Specialized NVFP4 GEMM function
    ↓
Extract layout-appropriate data pointers:
    A_data = transA ? A._columnwise_data : A._rowwise_data
    A_scale = transA ? A._columnwise_scale_inv : A._rowwise_scale_inv
    A_amax = transA ? A._amax_columnwise : A._amax_rowwise
    (same for B)
    ↓
Setup cuBLASLt operation descriptor:
    cublasLtMatmulDescCreate(&op_desc, COMPUTE_32F, CUDA_R_32F)
    ↓
Configure for FP4:
    - A_TYPE: CUDA_R_4F_E2M1
    - B_TYPE: CUDA_R_4F_E2M1
    - A_SCALE_POINTER: A_scale
    - B_SCALE_POINTER: B_scale
    - A_SCALE_TYPE: CUDA_R_8F_E4M3
    - B_SCALE_TYPE: CUDA_R_8F_E4M3
    - AMAX_POINTER: A_amax
    - BMAX_POINTER: B_amax
    - SPLIT_K: 1 (or auto)
    ↓
Setup matrix layouts:
    cublasLtMatrixLayoutCreate(&A_layout, FP4, ...)
    cublasLtMatrixLayoutCreate(&B_layout, FP4, ...)
    cublasLtMatrixLayoutCreate(&C_layout, out_dtype, ...)
    ↓
Allocate output tensor if needed:
    D = torch::empty({M, N}, dtype=out_dtype, device=cuda)
    ↓
cublasLtMatmul(handle, op_desc, &alpha,
               A_data, A_layout,
               B_data, B_layout,
               &beta, D.data_ptr(), C_layout,
               workspace, workspace_size, stream)
    cuBLAS: Library call
    ↓
GPU Execution (Blackwell Tensor Cores):
    Parallel processing of output tiles:
        For each tile:
            1. Load FP4 data blocks for A and B
            2. Unpack FP4 → intermediate precision
            3. Load corresponding E4M3 block scales
            4. Load FP32 tensor scales
            5. Apply 2-level scaling
            6. FP32 tensor core accumulation
            7. Apply beta for accumulation
            8. Cast to output dtype
            9. Store result
    ↓
GPU Synchronization
    ↓
Return to C++: D tensor populated with BF16/FP32 output
    ↓
Return to Python: y_native
```

---

## Key Concepts Demonstrated

### 1. Two-Level Scaling in NVFP4

NVFP4 uses **block-level E4M3 scales** + **tensor-level FP32 scale**:

```
dequantized_value = fp4_data * block_scale_e4m3 * tensor_scale_fp32
```

**Why two levels?**
- **E4M3 block scales**: Provide finer granularity (per 16 elements), reduce quantization error
- **FP32 tensor scale**: Extend dynamic range beyond E4M3 (which has limited range)
- **Combined**: E4M3 offers compact storage (1 byte), FP32 prevents overflow

**Example**:
```
Tensor with values: [0.001, 0.002, ..., 1000.0, 2000.0]
                     ↑small values↑        ↑large values↑

Without tensor scale:
  E4M3 range: ~[2^-9, 2^6] = [0.002, 64]
  → Large values overflow, small values underflow

With tensor scale:
  1. Compute tensor_scale = max_abs / fp4_max = 2000.0 / 6.0 ≈ 333.3
  2. Scale tensor down: [0.001/333.3, ..., 2000.0/333.3] = [3e-6, ..., 6.0]
  3. Now all values fit in FP4 range after block scaling
```

### 2. Data Packing

FP4 values are packed 2 per byte to save memory:

```python
# High-level packing
val0_fp4 = 0b0110  # 4 bits
val1_fp4 = 0b1001  # 4 bits
packed = (val1_fp4 << 4) | val0_fp4  # 0b10010110

# Unpacking
val0 = packed & 0x0F      # 0b0110
val1 = (packed >> 4) & 0x0F  # 0b1001
```

**Storage savings**:
- FP32: 4 bytes per value → 1024 values = 4KB
- FP4: 0.5 bytes per value → 1024 values = 512 bytes
- **8× reduction** (before considering scales)

### 3. Scale Padding

TE pads scale tensors for cuBLAS alignment:

```python
# Logical shape: [M, K/16] for M=256, K=128
# → [256, 8]

# cuBLAS requires:
# - Outer dim: multiple of 128
# - Inner dim: multiple of 4

# Padded shape: [256, 8] (already satisfies both)
# If M=200, K=128:
#   Logical: [200, 8]
#   Padded: [256, 8]  # Pad outer to 256 (next multiple of 128)
```

**Why pad?**
- cuBLAS uses vectorized loads (16 bytes at a time)
- Padding ensures memory accesses are aligned
- Performance critical for GEMM

### 4. Layout Selection (Rowwise vs Columnwise)

GEMM computation `C = A @ B` requires:
- **A reduction dimension** contiguous in memory
- **B reduction dimension** contiguous in memory

For FP4 with block scaling:
- Blocks must align with reduction dimension
- 16 consecutive elements in reduction dim = 1 block

**Example**:
```
GEMM: C[M,N] = A[M,K] @ B[K,N]

A needs blocks along K dimension:
  → Use rowwise layout (K is rightmost dim)

B needs blocks along K dimension:
  → K is leftmost dim, so use columnwise layout
  → (columnwise flips dims: [K,N] stored as [N,K])
```

Test uses rowwise × rowwise because reference implementation limitation.

### 5. Tolerance in Comparisons

Test uses `atol=8e-3, rtol=8e-3`:

**Why such large tolerance?**
1. **FP4 quantization error**: E2M1 has only 1 mantissa bit → large rounding
2. **2-level scaling**: Compounds error from both block and tensor scales
3. **Different accumulation order**: cuBLAS may use different tile sizes than reference
4. **Output dtype**: BF16 has 8-bit mantissa, adds rounding error

**Example error sources**:
```
True value: 1.234567
After FP4 quant: 1.25 (nearest representable)
After E4M3 scale: 1.2480 (E4M3 rounding)
After FP32 scale: 1.2479 (FP32 rounding)
After GEMM acc: 1.248 (BF16 rounding)

Total error: |1.248 - 1.234567| = 0.013433 ≈ 1.1% relative error
```

Multiple such errors accumulate in GEMM, hence 0.8% tolerance.

---

## Summary

This test demonstrates:
1. **NVFP4 quantization** from Python through C++ to CUDA
2. **cuBLASLt integration** for FP4 GEMM with 2-level block scaling
3. **Data layout management** (rowwise vs columnwise)
4. **Correctness verification** against reference implementation

**Key files traced**:
- Python API: [`nvfp4_tensor.py`](../../transformer_engine/pytorch/tensor/nvfp4_tensor.py)
- C++ bindings: [`pybind.cpp`](../../transformer_engine/pytorch/csrc/extensions/pybind.cpp)
- CUDA kernels: [`nvfp4.cu`](../../transformer_engine/common/recipe/nvfp4.cu)
- Test file: [`test_nvfp4_gemm_exact.py`](../../tests/pytorch/nvfp4/test_nvfp4_gemm_exact.py)

**Complete call depth**: Python → PyBind11 → C++ → cuBLASLt → Blackwell Tensor Cores → GPU memory

This level of detail enables debugging, optimization, and extension of the NVFP4 implementation.
