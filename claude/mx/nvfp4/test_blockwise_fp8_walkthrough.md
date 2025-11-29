# Blockwise FP8 Test Walkthrough

> **Purpose**: Annotated line-by-line walkthrough of blockwise FP8 GEMM tests, demonstrating 1D and 2D block scaling, data format handling, and cuBLASLt integration.

## Table of Contents

1. [Test Overview](#test-overview)
2. [Test: test_cublas_gemm_fp8_blockwise](#test-test_cublas_gemm_fp8_blockwise)
3. [Complete Call Stack Trace](#complete-call-stack-trace)
4. [Key Concepts Demonstrated](#key-concepts-demonstrated)

---

## Test Overview

**Test File**: [`tests/pytorch/test_float8_blockwise_gemm_exact.py`](../../tests/pytorch/test_float8_blockwise_gemm_exact.py)

**Purpose**: Verify blockwise FP8 GEMM correctness for:
- **1D block scaling** (1×128 blocks)
- **2D block scaling** (128×128 blocks)
- **Mixed quantization**: 1D×2D, 1D×1D, 2D×1D combinations
- **Various GEMM features**: Bias, GELU, split accumulator, columnwise layouts

**Test Matrix**:
- Block scaling combinations: 1D×2D, 1D×1D, 2D×1D
- Matrix sizes: From 128×128×128 to 4096×4096×4096
- Dtypes: FP8 E4M3, FP8 E5M2 (not both E5M2)
- Output dtypes: BF16, FP32
- Features: Accumulation, bias, GELU epilogue, columnwise

---

## Test: test_cublas_gemm_fp8_blockwise

### Test Function

[`test_float8_blockwise_gemm_exact.py:351-380`](../../tests/pytorch/test_float8_blockwise_gemm_exact.py#L351-L380)

```python
@pytest.mark.parametrize("M, K, N", [(128, 128, 128), (256, 128, 256), ...])
@pytest.mark.parametrize("x_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("w_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("noise_type", ["normal"])
@pytest.mark.parametrize("x_magnitude", [1])
@pytest.mark.parametrize("w_magnitude", [1])
@pytest.mark.parametrize("accumulate", [False])
@pytest.mark.parametrize("use_split_accumulator", [True])
@pytest.mark.parametrize("is_x_1d_scaled, is_w_1d_scaled",
    [(True, False), (True, True), (False, True)])
def test_cublas_gemm_fp8_blockwise_shape_varying(...):
    cublas_gemm_fp8_blockwise_case(
        x_dtype, w_dtype, out_dtype, M, K, N,
        noise_type, x_magnitude, w_magnitude,
        accumulate, use_split_accumulator,
        is_x_1d_scaled, is_w_1d_scaled,
    )
```

**What it tests**: GEMM `Y = W @ X` with various block scaling combinations:
- **1D×2D**: Input uses 1×128 blocks, weight uses 128×128 blocks
- **1D×1D**: Both use 1×128 blocks
- **2D×1D**: Weight uses 128×128 blocks, input uses 1×128 blocks

---

### Test Implementation: cublas_gemm_fp8_blockwise_case

**Location**: [`test_float8_blockwise_gemm_exact.py:25-207`](../../tests/pytorch/test_float8_blockwise_gemm_exact.py#L25-L207)

Let's trace through a specific example: `M=256, K=128, N=256`, `is_x_1d_scaled=True`, `is_w_1d_scaled=False` (1D×2D).

---

#### Step 1: Test Setup and Input Generation

[`test_float8_blockwise_gemm_exact.py:48-76`](../../tests/pytorch/test_float8_blockwise_gemm_exact.py#L48-L76)

```python
def cublas_gemm_fp8_blockwise_case(...):
    # Check unsupported dtype combination
    if x_dtype == torch.float8_e5m2 and w_dtype == torch.float8_e5m2:
        pytest.skip("FP8 GEMM doesn't support both a and b types being e5m2")

    # Check unsupported 2D×2D
    if not (is_x_1d_scaled or is_w_1d_scaled):
        pytest.skip("FP8 GEMM doesn't support 2D×2D block scaling")

    # Check hardware support
    if not fp8_blockwise_gemm_supported():
        pytest.skip("CUDA version does not support blockwise FP8 gemm.")

    # Setup
    device = "cuda"
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Determine shapes based on layout
    x_shape = (K, M) if x_columnwise else (M, K)  # (256, 128) for rowwise
    w_shape = (K, N) if w_columnwise else (N, K)  # (256, 128) for rowwise

    # Generate random inputs
    if noise_type == "uniform":
        x = torch.rand(x_shape, dtype=torch.float32, device=device) * x_magnitude * 2 - x_magnitude
        w = torch.rand(w_shape, dtype=torch.float32, device=device) * w_magnitude * 2 - w_magnitude
    elif noise_type == "normal":
        x = torch.randn(x_shape, dtype=torch.float32, device=device) * x_magnitude
        w = torch.randn(w_shape, dtype=torch.float32, device=device) * w_magnitude

    # Setup accumulation output if needed
    if accumulate:
        out = torch.randn((M, N), dtype=out_dtype, device=device) * x_magnitude
    else:
        out = None
```

**Data after this step**:
```
x: [256, 128] in FP32 (will be quantized to FP8)
w: [256, 128] in FP32 (will be quantized to FP8)
out: None (or [256, 256] in BF16 if accumulating)
```

---

#### Step 2: Create Quantizers

[`test_float8_blockwise_gemm_exact.py:79-101`](../../tests/pytorch/test_float8_blockwise_gemm_exact.py#L79-L101)

```python
# Set quantization parameters
x_quant_tile_shape = (1, 128) if is_x_1d_scaled else (128, 128)  # (1, 128) for 1D
w_quant_tile_shape = (1, 128) if is_w_1d_scaled else (128, 128)  # (128, 128) for 2D
x_block_scaling_dim = 1 if is_x_1d_scaled else 2  # 1
w_block_scaling_dim = 1 if is_w_1d_scaled else 2  # 2

# Convert PyTorch dtypes to TE dtypes
x_te_dtype = TE_DType[x_dtype]  # TE_DType.kFloat8E4M3 or kFloat8E5M2
w_te_dtype = TE_DType[w_dtype]

# Create quantizers
x_quantizer = Float8BlockQuantizer(
    fp8_dtype=x_te_dtype,          # E4M3 or E5M2
    rowwise=True,                  # Enable rowwise layout
    columnwise=True,               # Enable columnwise layout
    amax_epsilon=0.0,              # No epsilon added to amax
    force_pow_2_scales=True,       # Round scales to powers of 2
    block_scaling_dim=x_block_scaling_dim,  # 1 (1D blocks)
)
w_quantizer = Float8BlockQuantizer(
    fp8_dtype=w_te_dtype,
    rowwise=True,
    columnwise=True,
    amax_epsilon=0.0,
    force_pow_2_scales=True,
    block_scaling_dim=w_block_scaling_dim,  # 2 (2D blocks)
)
```

**Quantizer configuration**:
- **x_quantizer**: 1D blocks (1×128), force power-of-2 scales
- **w_quantizer**: 2D blocks (128×128), force power-of-2 scales

**Call Stack for Quantizer Creation**:
```
Python: Float8BlockQuantizer(fp8_dtype=E4M3, block_scaling_dim=1, ...)
    Location: transformer_engine/pytorch/tensor/float8_blockwise_tensor.py:44-61
    ↓
Initializes:
    self.dtype = tex.DType.kFloat8E4M3
    self.block_len = 128  # Fixed
    self.force_pow_2_scales = True
    self.amax_epsilon = 0.0
    self.block_scaling_dim = 1  # 1D or 2D
    self.all_gather_usage = False  # GEMM_READY format
```

---

#### Step 3: Quantize Inputs

[`test_float8_blockwise_gemm_exact.py:103-107`](../../tests/pytorch/test_float8_blockwise_gemm_exact.py#L103-L107)

```python
# Quantize x and w
qx = x_quantizer.make_empty(x_shape, dtype=x_dtype, device=device, requires_grad=False)
qx = x_quantizer.update_quantized(x, qx)

qw = w_quantizer.make_empty(w_shape, dtype=w_dtype, device=device, requires_grad=False)
qw = w_quantizer.update_quantized(w, qw)
```

**Call Stack for `make_empty()` (1D blocks)**:

```
Python: x_quantizer.make_empty((256, 128), dtype=E4M3, device=cuda, ...)
    Location: transformer_engine/pytorch/tensor/float8_blockwise_tensor.py:213-270
    ↓
Determine data format:
    data_format = GEMM_READY  # (all_gather_usage=False)
    ↓
Allocate rowwise buffers:
    data = torch.empty([256, 128], dtype=uint8, device=cuda)
        # FP8 data, same shape as input

    scale_shape = get_scale_shape((256, 128), columnwise=False)
        Location: float8_blockwise_tensor.py:112-172
        ↓
        For 1D rowwise:
            outer = ceil(K / 128) = ceil(128 / 128) = 1
            inner = round_to_4(M) = round_to_4(256) = 256
            return (1, 256)  # Transposed for cuBLAS

    scale_inv = torch.empty([1, 256], dtype=float32, device=cuda)
        # FP32 scales (power-of-2 constrained)
    ↓
Allocate columnwise buffers:
    columnwise_data = torch.empty([128, 256], dtype=uint8, device=cuda)
        # Transposed shape

    columnwise_scale_shape = get_scale_shape((256, 128), columnwise=True)
        For 1D columnwise:
            outer = ceil(M / 128) = ceil(256 / 128) = 2
            inner = round_to_4(K) = round_to_4(128) = 128
            return (2, 128)

    columnwise_scale_inv = torch.empty([2, 128], dtype=float32, device=cuda)
    ↓
Construct Float8BlockwiseQTensor:
    Location: float8_blockwise_tensor.py:258-270

    return Float8BlockwiseQTensor(
        shape=(256, 128),
        dtype=E4M3,
        fp8_dtype=E4M3,
        rowwise_data=data,                      # [256, 128]
        rowwise_scale_inv=scale_inv,            # [1, 256]
        columnwise_data=columnwise_data,        # [128, 256]
        columnwise_scale_inv=columnwise_scale_inv,  # [2, 128]
        quantizer=self,
        is_2D_scaled=False,  # 1D blocks
        data_format=GEMM_READY,
        requires_grad=False,
    )
```

**Call Stack for `make_empty()` (2D blocks)**:

```
Python: w_quantizer.make_empty((256, 128), dtype=E4M3, device=cuda, ...)
    ↓
Allocate rowwise buffers:
    data = torch.empty([256, 128], dtype=uint8, device=cuda)

    scale_shape = get_scale_shape((256, 128), columnwise=False)
        For 2D rowwise:
            outer = ceil(M / 128) = ceil(256 / 128) = 2
            inner = round_to_4(ceil(K / 128)) = round_to_4(ceil(128/128)) = round_to_4(1) = 4
            return (2, 4)  # Padded for alignment

    scale_inv = torch.empty([2, 4], dtype=float32, device=cuda)
    ↓
Allocate columnwise buffers:
    columnwise_data = torch.empty([128, 256], dtype=uint8, device=cuda)

    columnwise_scale_shape = get_scale_shape((256, 128), columnwise=True)
        For 2D columnwise:
            outer = ceil(K / 128) = 1
            inner = round_to_4(ceil(M / 128)) = round_to_4(2) = 4
            return (1, 4)

    columnwise_scale_inv = torch.empty([1, 4], dtype=float32, device=cuda)
    ↓
Construct Float8BlockwiseQTensor:
    return Float8BlockwiseQTensor(
        shape=(256, 128),
        dtype=E4M3,
        fp8_dtype=E4M3,
        rowwise_data=data,                      # [256, 128]
        rowwise_scale_inv=scale_inv,            # [2, 4]
        columnwise_data=columnwise_data,        # [128, 256]
        columnwise_scale_inv=columnwise_scale_inv,  # [1, 4]
        quantizer=self,
        is_2D_scaled=True,  # 2D blocks
        data_format=GEMM_READY,
        requires_grad=False,
    )
```

**Call Stack for `update_quantized()`**:

```
Python: x_quantizer.update_quantized(x, qx)
    Location: transformer_engine/pytorch/tensor/float8_blockwise_tensor.py:63-106
    ↓
Ensure contiguous:
    if not devices_match(x.device, qx.device):
        x = x.to(device=qx.device)
    if not x.is_contiguous():
        x = x.contiguous()
    ↓
Call C++ via PyBind11:
    tex.quantize(x, self, qx, noop_flag=None)
        Location: transformer_engine/pytorch/csrc/extensions/pybind.cpp:119
    ↓
C++: transformer_engine::pytorch::quantize()
    ↓
Dispatch to blockwise FP8 quantization:
    Check quantizer type: Float8BlockQuantizer
    ↓
Extract configuration:
    int block_scaling_dim = quantizer.attr("block_scaling_dim").cast<int>()  // 1 or 2
    bool force_pow2 = quantizer.attr("force_pow_2_scales").cast<bool>()  // True
    TE_DType dtype = quantizer.attr("dtype").cast<TE_DType>()  // E4M3
    ↓
Extract tensor pointers:
    void* src = x.data_ptr()  // [256, 128] FP32

    void* dst_rowwise_data = qx.attr("_rowwise_data").data_ptr()
    void* dst_rowwise_scale = qx.attr("_rowwise_scale_inv").data_ptr()
    void* dst_columnwise_data = qx.attr("_columnwise_data").data_ptr()
    void* dst_columnwise_scale = qx.attr("_columnwise_scale_inv").data_ptr()
    ↓
Launch CUDA kernel:
    if (block_scaling_dim == 1) {
        nvte_fp8_block_quantize_1d(src, dst_rowwise_data, dst_rowwise_scale,
                                     dst_columnwise_data, dst_columnwise_scale,
                                     M, K, force_pow2, dtype, stream);
    } else {
        nvte_fp8_block_quantize_2d(src, dst_rowwise_data, dst_rowwise_scale,
                                     dst_columnwise_data, dst_columnwise_scale,
                                     M, K, force_pow2, dtype, stream);
    }
    Location: transformer_engine/common/recipe/fp8_block_scaling.cu
    ↓
CUDA Kernel (1D example):

__global__ void fp8_block_quantize_1d_kernel(
    const float* src,        // [M, K] = [256, 128]
    uint8_t* dst_data,       // [M, K] = [256, 128]
    float* dst_scale,        // [ceil(K/128), round_to_4(M)] = [1, 256]
    int M, int K,
    bool force_pow2
) {
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_blocks = M * (K / 128);  // 256 * 1 = 256 blocks

    if (tidx >= num_blocks) return;

    int row = tidx / (K / 128);  // Which row
    int block_col = tidx % (K / 128);  // Which 128-element block (0 for K=128)
    int col_start = block_col * 128;  // Column offset

    // 1. Load 128 elements
    float vals[128];
    for (int i = 0; i < 128; i++) {
        vals[i] = src[row * K + col_start + i];
    }

    // 2. Compute amax
    float amax = 0.0f;
    for (int i = 0; i < 128; i++) {
        amax = fmaxf(amax, fabsf(vals[i]));
    }

    // 3. Compute scale
    float fp8_max = 448.0f;  // E4M3 max value
    float scale_fp32 = amax / fp8_max;

    if (force_pow2) {
        // Round to nearest power of 2
        int exponent;
        frexpf(scale_fp32, &exponent);
        scale_fp32 = ldexpf(1.0f, exponent);
    }

    // 4. Store scale (transposed for cuBLAS)
    // cuBLAS expects scales in transposed format
    int scale_row = block_col;  // 0
    int scale_col = row;        // 0-255
    dst_scale[scale_row * round_to_4(M) + scale_col] = scale_fp32;

    // 5. Quantize and store
    float scale_inv = 1.0f / scale_fp32;
    for (int i = 0; i < 128; i++) {
        float val = vals[i];
        float scaled = val * scale_inv;
        uint8_t quantized = float_to_fp8_e4m3(scaled);
        dst_data[row * K + col_start + i] = quantized;
    }

    // Similar kernel launch for columnwise layout (transpose then quantize)
}

CUDA Kernel (2D example for weight):

__global__ void fp8_block_quantize_2d_kernel(
    const float* src,        // [M, K] = [256, 128]
    uint8_t* dst_data,       // [M, K] = [256, 128]
    float* dst_scale,        // [ceil(M/128), round_to_4(ceil(K/128))] = [2, 4]
    int M, int K,
    bool force_pow2
) {
    // Grid: 2D grid of 128×128 tiles
    int block_row = blockIdx.y;  // 0 or 1 (for M=256)
    int block_col = blockIdx.x;  // 0 (for K=128)

    int row_offset = block_row * 128;
    int col_offset = block_col * 128;

    if (row_offset >= M || col_offset >= K) return;

    // Shared memory for reduction
    __shared__ float shared_amax;
    if (threadIdx.x == 0) shared_amax = 0.0f;
    __syncthreads();

    // Each thread processes multiple elements
    int tidx = threadIdx.x;
    int elements_per_thread = (128 * 128) / blockDim.x;

    float local_amax = 0.0f;
    for (int i = 0; i < elements_per_thread; i++) {
        int flat_idx = tidx * elements_per_thread + i;
        int local_row = flat_idx / 128;
        int local_col = flat_idx % 128;
        int global_row = row_offset + local_row;
        int global_col = col_offset + local_col;

        if (global_row < M && global_col < K) {
            float val = src[global_row * K + global_col];
            local_amax = fmaxf(local_amax, fabsf(val));
        }
    }

    // Reduce to shared_amax
    atomicMaxFloat(&shared_amax, local_amax);
    __syncthreads();

    // Compute scale
    float amax = shared_amax;
    float fp8_max = 448.0f;
    float scale_fp32 = amax / fp8_max;

    if (force_pow2) {
        int exponent;
        frexpf(scale_fp32, &exponent);
        scale_fp32 = ldexpf(1.0f, exponent);
    }

    // Store scale
    if (threadIdx.x == 0) {
        int scale_row = block_row;
        int scale_col = block_col;
        dst_scale[scale_row * 4 + scale_col] = scale_fp32;
    }
    __syncthreads();

    // Quantize
    float scale_inv = 1.0f / scale_fp32;
    for (int i = 0; i < elements_per_thread; i++) {
        int flat_idx = tidx * elements_per_thread + i;
        int local_row = flat_idx / 128;
        int local_col = flat_idx % 128;
        int global_row = row_offset + local_row;
        int global_col = col_offset + local_col;

        if (global_row < M && global_col < K) {
            float val = src[global_row * K + global_col];
            float scaled = val * scale_inv;
            uint8_t quantized = float_to_fp8_e4m3(scaled);
            dst_data[global_row * K + global_col] = quantized;
        }
    }
}

Returns to Python:
    qx and qw now fully quantized with data and scales populated
```

**Result**: `qx` (1D) and `qw` (2D) are quantized blockwise FP8 tensors.

---

#### Step 4: Setup Reference GEMM

[`test_float8_blockwise_gemm_exact.py:114-145`](../../tests/pytorch/test_float8_blockwise_gemm_exact.py#L114-L145)

```python
# Reference GEMM
ref_gemm = CuBLASRefBlockwiseGemm()
scale_decoder = CuBLASScaleMunger()

# Extract data and scales based on layout
qx_data = (
    qx._columnwise_data.view(dtype=x_dtype)
    if x_columnwise
    else qx._rowwise_data.view(dtype=x_dtype)
)
qw_data = (
    qw._columnwise_data.view(dtype=w_dtype)
    if w_columnwise
    else qw._rowwise_data.view(dtype=w_dtype)
)
ref_scales_x = qx._columnwise_scale_inv if x_columnwise else qx._rowwise_scale_inv
ref_scales_w = qw._columnwise_scale_inv if w_columnwise else qw._rowwise_scale_inv

# Reference GEMM
y_ref = ref_gemm.qgemm(
    qx=qx_data,                          # [256, 128] E4M3
    qw=qw_data,                          # [256, 128] E4M3
    out_dtype=out_dtype,                 # BF16 or FP32
    demunged_sx=CuBLASScaleMunger.demunge_scale_shape_from_backend(
        qtensor_shape=(M, K),
        scales=ref_scales_x,
        tile_shape=x_quant_tile_shape   # (1, 128)
    ),
    demunged_sw=CuBLASScaleMunger.demunge_scale_shape_from_backend(
        qtensor_shape=(N, K),
        scales=ref_scales_w,
        tile_shape=w_quant_tile_shape   # (128, 128)
    ),
    quant_tile_shape_x=x_quant_tile_shape,  # (1, 128)
    quant_tile_shape_w=w_quant_tile_shape,  # (128, 128)
    bias=bias,                           # None or [1, N]
    out=out.clone() if accumulate else None,
    accumulate=accumulate,
    use_split_accumulator=use_split_accumulator,
)
```

**What this does**:
- **CuBLASRefBlockwiseGemm**: Reference implementation that mimics cuBLAS behavior
  - Located in: [`tests/pytorch/references/blockwise_fp8_gemm_reference.py`](../../tests/pytorch/references/blockwise_fp8_gemm_reference.py)
- **demunge_scale_shape_from_backend**: Converts TE's transposed scale layout to logical layout
  - TE stores scales transposed for cuBLAS efficiency
  - Reference needs logical shape for correctness checking

**Reference GEMM operation (simplified)**:
```python
def qgemm(self, qx, qw, demunged_sx, demunged_sw, ...):
    # 1. Cast FP8 data to FP32
    x_fp32 = qx.to(torch.float32)  # [M, K]
    w_fp32 = qw.to(torch.float32)  # [N, K]

    # 2. Apply block scaling
    for i in range(M):
        for k in range(K):
            # Determine which block (i,k) belongs to
            block_row, block_col = get_block_indices(i, k, x_quant_tile_shape)
            scale = demunged_sx[block_row, block_col]
            x_fp32[i, k] *= scale

    # Similar for w_fp32 with 2D blocks

    # 3. GEMM in FP32
    y_ref = x_fp32 @ w_fp32.T  # [M, N]

    # 4. Add bias if present
    if bias is not None:
        y_ref += bias

    # 5. Accumulate if needed
    if accumulate:
        y_ref += out

    # 6. Cast to output dtype
    return y_ref.to(out_dtype)
```

---

#### Step 5: Native cuBLAS GEMM

[`test_float8_blockwise_gemm_exact.py:147-179`](../../tests/pytorch/test_float8_blockwise_gemm_exact.py#L147-L179)

```python
# Allocate cuBLAS workspace
workspace_size = 0
workspace = torch.empty(0, dtype=torch.uint8, device=device)

transa = True if not w_columnwise else False  # True (transpose W)
transb = False if not x_columnwise else True  # False
out_quantizer = None
aux_tensor = torch.randn((M, N), dtype=out_dtype, device=device) if use_gelu else None
bias_dtype = TE_DType[torch.bfloat16 if bias is None else bias.dtype]

# cuBLAS GEMM
# Returns: (out, bias_grad, gelu_input, extra_output)
y = tex.generic_gemm(
    qw,                                  # A: [256, 128] blockwise FP8 (2D)
    transa,                              # True
    qx,                                  # B: [256, 128] blockwise FP8 (1D)
    transb,                              # False
    out.clone() if accumulate else None, # D
    out_quantizer,                       # None
    TE_DType[out_dtype],                # BF16 or FP32
    bias,                                # None or [1, N]
    bias_dtype,                          # BF16
    use_gelu,                            # False
    aux_tensor,                          # None
    use_grad,                            # False
    workspace,                           # Empty workspace
    workspace.shape[0],                  # 0
    accumulate,                          # False
    use_split_accumulator,               # True
)[0]
```

**Call Stack**:

```
Python: tex.generic_gemm(qw, True, qx, False, ...)
    Location: transformer_engine/pytorch/csrc/extensions/pybind.cpp:126
    ↓
C++: transformer_engine::pytorch::gemm(...)
    ↓
Determine GEMM type:
    Check A: Float8BlockwiseQTensor (2D scaled)
    Check B: Float8BlockwiseQTensor (1D scaled)
    → gemm_type = FP8_BLOCKWISE_GEMM
    ↓
Dispatch to gemm_fp8_blockwise():
    ↓
Extract data pointers based on transpose flags:
    // A (weight) with transA=True → use columnwise
    void* A_data = qw._columnwise_data.data_ptr()  // [128, 256]
    void* A_scale = qw._columnwise_scale_inv.data_ptr()  // [1, 4] for 2D

    // B (input) with transB=False → use rowwise
    void* B_data = qx._rowwise_data.data_ptr()  // [256, 128]
    void* B_scale = qx._rowwise_scale_inv.data_ptr()  // [1, 256] for 1D

    // Check block scaling dims
    bool A_is_2d = qw._is_2D_scaled  // True
    bool B_is_2d = qx._is_2D_scaled  // False
    ↓
Setup cuBLASLt operation descriptor:
    cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

    // Set matrix types
    cudaDataType_t fp8_type = (A.dtype == E4M3) ? CUDA_R_8F_E4M3FN : CUDA_R_8F_E5M2;
    cublasLtMatmulDescSetAttribute(
        op_desc, CUBLASLT_MATMUL_DESC_A_TYPE,
        &fp8_type, sizeof(fp8_type)
    );
    cublasLtMatmulDescSetAttribute(
        op_desc, CUBLASLT_MATMUL_DESC_B_TYPE,
        &fp8_type, sizeof(fp8_type)
    );

    // Provide scale pointers
    cublasLtMatmulDescSetAttribute(
        op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
        &A_scale, sizeof(void*)
    );
    cublasLtMatmulDescSetAttribute(
        op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
        &B_scale, sizeof(void*)
    );

    // Set scale format (FP32 for blockwise FP8)
    cudaDataType_t scale_type = CUDA_R_32F;
    cublasLtMatmulDescSetAttribute(
        op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_TYPE,
        &scale_type, sizeof(scale_type)
    );
    cublasLtMatmulDescSetAttribute(
        op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_TYPE,
        &scale_type, sizeof(scale_type)
    );

    // Enable split-K accumulator (required for blockwise)
    if (use_split_accumulator) {
        int split_k = 1;  // Can be auto-tuned
        cublasLtMatmulDescSetAttribute(
            op_desc, CUBLASLT_MATMUL_DESC_SPLIT_K,
            &split_k, sizeof(split_k)
        );
    }

    // Configure epilogue (bias, GELU, etc.) if needed
    if (bias) {
        cublasLtMatmulDescSetAttribute(
            op_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
            &bias_ptr, sizeof(void*)
        );
    }
    ↓
Setup matrix layouts:
    // A: columnwise [K, N] with leading dim K
    cublasLtMatrixLayoutCreate(&A_layout, fp8_type, K, N, K);

    // B: rowwise [K, M] with leading dim K
    cublasLtMatrixLayoutCreate(&B_layout, fp8_type, K, M, K);

    // C: [N, M] with leading dim N (cuBLAS column-major)
    cublasLtMatrixLayoutCreate(&C_layout, output_dtype, N, M, N);
    ↓
Allocate output if needed:
    if (!D.defined()) {
        D = torch::empty({M, N}, torch::dtype(output_dtype).device(cuda));
    }
    ↓
Execute cuBLASLt GEMM:
    float alpha = 1.0f;
    float beta = accumulate ? 1.0f : 0.0f;

    cublasLtMatmul(
        cublas_lt_handle,
        op_desc,
        &alpha,
        A_data, A_layout,  // Weight columnwise FP8
        B_data, B_layout,  // Input rowwise FP8
        &beta,
        D.data_ptr(), C_layout,  // Output BF16/FP32
        workspace.data_ptr(), workspace_size,
        stream
    );
    ↓
GPU Execution (H100 Tensor Cores):
    Parallel tile processing:
        For each output tile C[tile_i, tile_j]:
            1. Load FP8 tiles from A and B
            2. Load corresponding scales
               - A: 128×128 blocks → 1 scale per tile
               - B: 1×128 blocks → 1 scale per 128 elements
            3. Unpack FP8 → higher precision
            4. Apply scales
            5. FP32 accumulation in Tensor Cores
            6. Apply beta (accumulation)
            7. Cast to output dtype
            8. Store tile
    ↓
Returns to Python:
    y: [256, 256] in BF16 or FP32
```

---

#### Step 6: Validate Results

[`test_float8_blockwise_gemm_exact.py:181-206`](../../tests/pytorch/test_float8_blockwise_gemm_exact.py#L181-L206)

```python
# Ensure different tensors
assert y_ref is not y, "y_ref and y should not be the same tensor"

# Handle NaNs
assert not torch.isnan(y_ref.float()).all(), "All elements are nan"
y_ref = torch.where(y_ref.isnan(), torch.zeros_like(y_ref), y_ref)
y = torch.where(y.isnan(), torch.zeros_like(y), y)

# Compare
torch.testing.assert_close(y, y_ref, atol=atol, rtol=rtol)
```

**Default tolerances**: `atol=0.0, rtol=0.0` (exact match expected)

**Why exact match possible?**
1. Both use same quantized FP8 data
2. Both use same FP32 scales
3. Both accumulate in FP32
4. FP8 quantization is deterministic (no stochastic rounding)
5. cuBLAS and reference use same accumulation order

**If using BF16 output**: Small differences possible due to final cast, so tests use tolerances

---

## Complete Call Stack Trace

### Quantization Path (1D Blocks)

```
User Test
    ↓
quantizer = Float8BlockQuantizer(block_scaling_dim=1, force_pow_2_scales=True)
    Python: transformer_engine/pytorch/tensor/float8_blockwise_tensor.py:44-61
    ↓
qtensor = quantizer.make_empty((M, K), dtype=E4M3, device=cuda)
    Python: float8_blockwise_tensor.py:213-270
    ↓
Allocate data and scale tensors:
    - rowwise_data: [M, K] uint8
    - rowwise_scale_inv: [ceil(K/128), round_to_4(M)] float32 (transposed)
    - columnwise_data: [K, M] uint8
    - columnwise_scale_inv: [ceil(M/128), round_to_4(K)] float32
    ↓
qtensor = quantizer.update_quantized(tensor_fp32, qtensor)
    Python: float8_blockwise_tensor.py:63-106
    ↓
tex.quantize(tensor_fp32, quantizer, qtensor)
    C++: transformer_engine/pytorch/csrc/extensions/pybind.cpp:119
    ↓
transformer_engine::pytorch::quantize()
    C++: Dispatch to blockwise FP8 handler
    ↓
Extract configuration:
    - block_scaling_dim = 1
    - force_pow_2_scales = True
    - dtype = E4M3
    ↓
nvte_fp8_block_quantize_1d(...)
    CUDA: transformer_engine/common/recipe/fp8_block_scaling.cu
    ↓
fp8_block_quantize_1d_kernel<<<...>>>()
    Per thread:
        1. Load 128-element block
        2. Compute block amax
        3. Compute scale = amax / fp8_max
        4. Round scale to power of 2 if force_pow2=True
        5. Store scale (transposed layout for cuBLAS)
        6. Quantize 128 elements to FP8
        7. Store quantized data
    ↓
Launch columnwise quantization kernel (transpose + quantize)
    ↓
Return to Python: Float8BlockwiseQTensor fully populated
```

### Quantization Path (2D Blocks)

```
quantizer = Float8BlockQuantizer(block_scaling_dim=2, ...)
    ↓
qtensor = quantizer.make_empty((M, K), ...)
    Allocate:
        - rowwise_data: [M, K]
        - rowwise_scale_inv: [ceil(M/128), round_to_4(ceil(K/128))]  # 2D
        - columnwise_data: [K, M]
        - columnwise_scale_inv: [ceil(K/128), round_to_4(ceil(M/128))]
    ↓
qtensor = quantizer.update_quantized(tensor_fp32, qtensor)
    ↓
tex.quantize(...)
    ↓
nvte_fp8_block_quantize_2d(...)
    ↓
fp8_block_quantize_2d_kernel<<<...>>>()
    Grid: 2D grid of 128×128 tiles
    Per block:
        1. Threads cooperatively compute 128×128 amax
        2. Compute tile scale
        3. Round to power of 2 if needed
        4. Store scale (one per 128×128 tile)
        5. Threads quantize their elements
        6. Store quantized data
    ↓
Return to Python: Float8BlockwiseQTensor (2D scaled)
```

### GEMM Path (1D×2D Mixed)

```
User Test
    ↓
tex.generic_gemm(qw_2d, True, qx_1d, False, ...)
    C++: transformer_engine/pytorch/csrc/extensions/pybind.cpp:126
    ↓
transformer_engine::pytorch::gemm(...)
    ↓
Infer GEMM type:
    A: Float8BlockwiseQTensor (2D scaled)
    B: Float8BlockwiseQTensor (1D scaled)
    → FP8_BLOCKWISE_GEMM
    ↓
gemm_fp8_blockwise(...)
    ↓
Select layout based on transpose:
    A (transA=True): Use columnwise layout
        - data: [K, N]
        - scales: [ceil(K/128), round_to_4(ceil(N/128))]
    B (transB=False): Use rowwise layout
        - data: [M, K]
        - scales: [ceil(K/128), round_to_4(M)]
    ↓
Setup cuBLASLt descriptor:
    cublasLtMatmulDescCreate(&op_desc, COMPUTE_32F, CUDA_R_32F)
    ↓
Configure for FP8 blockwise:
    - A_TYPE: CUDA_R_8F_E4M3FN or E5M2
    - B_TYPE: CUDA_R_8F_E4M3FN or E5M2
    - A_SCALE_POINTER: A_scale
    - B_SCALE_POINTER: B_scale
    - A_SCALE_TYPE: CUDA_R_32F  # FP32 scales
    - B_SCALE_TYPE: CUDA_R_32F
    - SPLIT_K: 1 (or auto)
    - EPILOGUE: BIAS, GELU, etc. if configured
    ↓
Setup layouts:
    cublasLtMatrixLayoutCreate(&A_layout, fp8_type, K, N, K)
    cublasLtMatrixLayoutCreate(&B_layout, fp8_type, K, M, K)
    cublasLtMatrixLayoutCreate(&C_layout, out_dtype, N, M, N)
    ↓
cublasLtMatmul(handle, op_desc, &alpha,
               A_data, A_layout,
               B_data, B_layout,
               &beta, D.data_ptr(), C_layout,
               workspace, workspace_size, stream)
    ↓
GPU (H100 Tensor Cores):
    Tile-based GEMM:
        For each output tile:
            1. Load FP8 data tiles
            2. Load scales:
               - A: One FP32 scale per 128×128 tile
               - B: One FP32 scale per 1×128 block
            3. Unpack FP8 to higher precision
            4. Apply scales
            5. FP32 accumulation
            6. Cast to output dtype
            7. Store tile
    ↓
Return to Python: output tensor [M, N]
```

---

## Key Concepts Demonstrated

### 1. Mixed Block Quantization (1D×2D)

**Why mix 1D and 2D?**

Different tensors have different characteristics:

**Activations/Gradients** → 1D blocks:
- Change every iteration (dynamic)
- Need fast quantization
- 1×128 blocks: Simpler, faster kernel
- Good enough precision for most use cases

**Weights** → 2D blocks:
- Static or slowly changing
- Can afford slower quantization
- 128×128 blocks: Better precision, especially for gradients
- Symmetric under transpose (important for weight grad)

**Test combinations**:
- **1D×2D**: Typical case (activation × weight)
- **1D×1D**: All tensors use 1D (fastest, lower precision)
- **2D×1D**: Reversed roles (less common)

---

### 2. Power-of-2 Scale Constraint

`force_pow_2_scales=True` rounds scales to nearest power of 2:

```python
# Without constraint
amax = 123.45
scale = amax / 448.0 = 0.2755...  # Arbitrary FP32

# With constraint
import math
exponent = math.frexp(0.2755)[1]  # Extract exponent
scale_pow2 = 2 ** exponent = 0.25  # Nearest power of 2
```

**Advantages**:
1. **Hardware efficiency**: Multiplication by power-of-2 is a bitshift
2. **Reduced precision loss**: No mantissa bits in scale itself
3. **Simpler debugging**: Easier to reason about quantized values

**Disadvantage**:
- Slight precision loss from rounding scale (but negligible compared to FP8 precision)

---

### 3. GEMM_READY vs COMPACT Formats

**GEMM_READY** (default):
```python
# Data: Potentially transposed
rowwise_data: [M, K]
columnwise_data: [K, M]  # Transposed

# Scales: Transposed and padded for cuBLAS
rowwise_scale: [ceil(K/128), round_to_4(M)]  # Note: K and M swapped
columnwise_scale: [ceil(M/128), round_to_4(K)]
```

**Why transpose scales?**
- cuBLAS expects scales in specific layout for efficient access
- TE pre-transposes to match cuBLAS requirements
- Avoids runtime transpose

**COMPACT**:
```python
# Data: Logical layout
rowwise_data: [M, K]
columnwise_data: May not exist

# Scales: Logical layout, minimal padding
rowwise_scale: [M, ceil(K/128)]  # Direct mapping
```

**Use case**: All-gather in distributed training (reduces communication)

---

### 4. Split-K Accumulation

`use_split_accumulator=True` enables split-K in cuBLAS:

**What is split-K?**
```
Normal GEMM: C[i,j] = Σ(k=0 to K-1) A[i,k] * B[k,j]
                      ↑ Single accumulation

Split-K: Partition K dimension into chunks:
    C[i,j] = Σ(k=0 to K1-1) A[i,k]*B[k,j]  +
             Σ(k=K1 to K2-1) A[i,k]*B[k,j]  +
             ...
             ↑ Parallel partial sums, then reduce
```

**Benefits**:
1. **Higher occupancy**: More tiles can run concurrently
2. **Better load balance**: Especially for small M or N
3. **Required for blockwise FP8**: cuBLAS may internally mandate split-K for numerical stability

---

### 5. Scale Layout "Munging"

Test uses `CuBLASScaleMunger` to convert between TE and reference layouts:

**TE layout (GEMM_READY)**:
```
For 1D rowwise with (M=256, K=128):
    TE stores: [1, 256]  # Transposed
    Logical:   [256, 1]  # What you'd expect
```

**Reference needs logical layout**:
```python
demunged_sx = CuBLASScaleMunger.demunge_scale_shape_from_backend(
    qtensor_shape=(M, K),
    scales=ref_scales_x,  # [1, 256] from TE
    tile_shape=(1, 128)
)
# Returns: [256, 1] (logical shape)
```

**Why this complexity?**
- TE optimizes for cuBLAS performance (transposed layout)
- Reference optimizes for readability (logical layout)
- Munger bridges the two

---

### 6. Bias and GELU Epilogues

Tests include bias and GELU to validate epilogue fusion:

**Bias**:
```python
bias = torch.randn((1, N), dtype=torch.bfloat16, device=device)
y = tex.generic_gemm(..., bias=bias, ...)

# cuBLAS fuses: Y = (A @ B) + bias
# vs. separate: Y = A @ B; Y += bias
```

**GELU**:
```python
aux_tensor = torch.randn((M, N), ...)
y = tex.generic_gemm(..., use_gelu=True, gelu_input=aux_tensor, ...)

# Fused: Y = GELU(A @ B)
# Stores pre-GELU aux for backward
```

**Benefits of fusion**:
1. Fewer kernel launches
2. Reduced memory traffic (no intermediate writes)
3. Higher throughput

---

### 7. Columnwise Layout GEMM

Some tests use `x_columnwise=True` or `w_columnwise=True`:

**Why test columnwise?**
```
Some distributed strategies (e.g., sequence parallel) shard tensors
such that the natural layout is columnwise (K dimension first).

TE must handle:
    C = A @ B where A or B is in columnwise layout

Example:
    A: [N, K] rowwise
    B: [K, M] columnwise (stored as [M, K])
    C = A @ B^T  (use transB=True)
```

**Test validates**:
- Correct layout selection (rowwise vs columnwise data/scales)
- Proper transpose flags to cuBLAS
- Scale tensor selection based on layout

---

## Summary

This test demonstrates:
1. **Flexible block quantization**: 1D (1×128) and 2D (128×128) blocks
2. **Mixed quantization**: Different formats for different tensors
3. **cuBLASLt integration**: FP8 GEMM with FP32 block scales
4. **Layout management**: GEMM_READY format with transposed scales
5. **Feature testing**: Bias, GELU, split-K, accumulation
6. **Correctness**: Exact match to reference implementation

**Key files traced**:
- Python API: [`float8_blockwise_tensor.py`](../../transformer_engine/pytorch/tensor/float8_blockwise_tensor.py)
- C++ bindings: [`pybind.cpp`](../../transformer_engine/pytorch/csrc/extensions/pybind.cpp)
- CUDA kernels: [`fp8_block_scaling.cu`](../../transformer_engine/common/recipe/fp8_block_scaling.cu)
- Test file: [`test_float8_blockwise_gemm_exact.py`](../../tests/pytorch/test_float8_blockwise_gemm_exact.py)
- Reference: [`blockwise_fp8_gemm_reference.py`](../../tests/pytorch/references/blockwise_fp8_gemm_reference.py)

**Complete trace depth**: Python → PyBind11 → C++ → cuBLASLt → H100 Tensor Cores → GPU memory

This comprehensive walkthrough enables understanding, debugging, and extending the blockwise FP8 implementation.
