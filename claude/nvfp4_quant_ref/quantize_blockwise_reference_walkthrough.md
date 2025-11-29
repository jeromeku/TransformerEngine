# NVFP4 Blockwise Quantization Reference - Complete Walkthrough

**Function Location**: [quantization_nvfp4.py:438-521](../../../transformer_engine/pytorch/custom_recipes/quantization_nvfp4.py#L438-L521)

## Executive Summary

`_quantize_blockwise_reference` converts a high-precision (bf16/fp32) tensor into NVFP4 (4-bit floating point) format with blockwise scaling factors. The function supports two quantization modes:
- **1D quantization**: Each 1×16 row-chunk gets its own scale (tile_len_x=16, tile_len_y=1)
- **2D quantization**: Each 16×16 block gets its own scale (tile_len_x=16, tile_len_y=16)

The quantization uses a **two-level scaling hierarchy**:
1. **Blockwise scales** (FP8 E4M3): Per-block normalization factors
2. **Global scale** (scalar): Single scaling factor for the entire tensor

## Concrete Example

Let's trace through with:
- **Input tensor `x`**: shape (128, 128), dtype=bf16
- **Quantization mode**: 1D (tile_len_x=16, tile_len_y=1)
- **Global amax**: `global_amax = 10.0` (max absolute value in tensor)

## Function Signature

```python
@classmethod
def _quantize_blockwise_reference(
    cls,
    x: torch.Tensor,              # Input: (128, 128) bf16
    global_amax: torch.Tensor,    # Scalar: max(abs(x))
    tile_len_x: int,              # 16 (block width)
    tile_len_y: int,              # 1 (block height)
    *,
    pow_2_scales: bool,           # False (use FP8 scales, not power-of-2)
    eps: float,                   # Unused in this path
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Returns: (quantized_data, decode_scales)
```

## Step-by-Step Execution

### Step 1: Input Validation and Mode Detection (Lines 449-451)

```python
assert x.ndim == 2
using_2d_quantization = tile_len_x == 16 and tile_len_y == 16
m, n = x.shape  # m=128, n=128
```

**State**:
- `using_2d_quantization = False` (since tile_len_y=1)
- `m = 128`, `n = 128`

**Data types**:
- `x`: (128, 128) bf16/fp32

---

### Step 2: Compute Per-Block Amax Values (Lines 452-470)

This step computes the maximum absolute value for each quantization block.

#### For 1D Quantization (Our Example)

```python
# x shape: (128, 128)
x_reshaped = x.view(m, n // tile_len_x, tile_len_x)  # (128, 8, 16)
vec_max = torch.amax(torch.abs(x_reshaped), dim=-1, keepdim=True).to(
    torch.float32
)  # (128, 8, 1)
```

**What's happening**:
1. **Reshape**: Split 128 columns into 8 chunks of 16
   - Original: `x[i, j]` where i∈[0,128), j∈[0,128)
   - Reshaped: `x_reshaped[i, k, l]` where i∈[0,128), k∈[0,8), l∈[0,16)
   - Mapping: `x[i, j] = x_reshaped[i, j//16, j%16]`

2. **Compute amax**: For each (i, k) block, find max(abs(values))
   - `vec_max[i, k, 0]` = max(abs(x[i, k*16:(k+1)*16]))

**Example values**:
```
x_reshaped[0, 0, :] = [0.5, -1.2, 3.0, ..., 2.1]  # 16 values from row 0, cols [0:16]
vec_max[0, 0, 0] = 3.0  # max absolute value in this block

x_reshaped[0, 1, :] = [0.1, 0.8, -0.5, ..., 1.0]  # 16 values from row 0, cols [16:32]
vec_max[0, 1, 0] = 1.0

... (similarly for all 128 rows × 8 blocks)
```

**State after Step 2**:
- `vec_max`: (128, 8, 1) float32 - max absolute value per 1×16 block
- `x_reshaped`: (128, 8, 16) - reshaped view of input

**Data types**:
- `vec_max`: (128, 8, 1) **float32**

#### For 2D Quantization (Alternative Path)

If we were using 2D quantization (tile_len_x=16, tile_len_y=16):

```python
# x shape: (128, 128)
x_blocks = (
    x.unfold(0, tile_len_y, tile_len_y)
    .unfold(1, tile_len_x, tile_len_x)
    .to(torch.float32)
)  # (8, 8, 16, 16)
block_amax = torch.amax(torch.abs(x_blocks), dim=(-1, -2))  # (8, 8)
vec_max = block_amax.repeat_interleave(tile_len_y, dim=0).unsqueeze(-1)  # (128, 8, 1)
```

**What this does**:
- Creates 8×8 grid of 16×16 tiles
- Computes amax for each tile: `block_amax[i, j]` = max(abs(x[i*16:(i+1)*16, j*16:(j+1)*16]))
- Replicates each block's amax across all 16 rows in that block

---

### Step 3: Prepare Constants (Lines 471-474)

```python
x = x.view(m, n // tile_len_x, tile_len_x)  # (128, 8, 16) - same as x_reshaped
FLOAT4_E2M1_MAX = torch.tensor(6.0, device=x.device, dtype=torch.float32)
FLOAT8_E4M3_MAX = torch.tensor(448.0, device=x.device, dtype=torch.float32)
decode_scale = torch.div(vec_max, FLOAT4_E2M1_MAX)  # (128, 8, 1)
```

**What's happening**:
- `FLOAT4_E2M1_MAX = 6.0`: Maximum representable value in FP4 E2M1 format
- `FLOAT8_E4M3_MAX = 448.0`: Maximum representable value in FP8 E4M3 format
- `decode_scale`: Initial scale factor to map block amax to FP4 range

**Calculation**:
```
decode_scale[i, k, 0] = vec_max[i, k, 0] / 6.0
```

**Example**:
```
vec_max[0, 0, 0] = 3.0
decode_scale[0, 0, 0] = 3.0 / 6.0 = 0.5

vec_max[10, 5, 0] = 8.4
decode_scale[10, 5, 0] = 8.4 / 6.0 = 1.4
```

**State**:
- `decode_scale`: (128, 8, 1) float32 - raw blockwise decode scales

**Physical meaning**: To dequantize, we'll multiply FP4 values by these scales.

---

### Step 4: Scale Computation with Global Normalization (Lines 476-515)

This is the **two-level scaling** logic. We'll trace the `pow_2_scales=False` path (NVFP4).

```python
if pow_2_scales:
    # ... (MXFP4 path - not taken)
else:
    # NVFP4 path with global scale
```

#### Step 4a: Compute Global Encode Scale (Lines 483-494)

```python
global_encode_scale = torch.div(FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX, global_amax)
# = 448.0 * 6.0 / global_amax
# = 2688.0 / global_amax
```

**With our example** (`global_amax = 10.0`):
```
global_encode_scale = 2688.0 / 10.0 = 268.8
```

**Safety clamps**:
```python
global_encode_scale = torch.min(
    global_encode_scale,
    torch.tensor(
        torch.finfo(torch.float32).max,  # ~3.4e38
        device=global_encode_scale.device,
        dtype=torch.float32,
    ),
)
if global_encode_scale == torch.tensor(0.0, device=x.device, dtype=torch.float32):
    global_encode_scale = torch.tensor(1.0, device=x.device, dtype=torch.float32)
```

**Result**: `global_encode_scale = 268.8` (scalar float32)

#### Step 4b: Compute Global Decode Scale (Line 494)

```python
global_decode_scale = torch.div(1.0, global_encode_scale)
# = 1.0 / 268.8
# ≈ 0.003720
```

**Result**: `global_decode_scale ≈ 0.003720` (scalar float32)

#### Step 4c: Normalize Blockwise Scales to FP8 E4M3 (Lines 496-506)

```python
decode_scale = decode_scale * global_encode_scale
```

**Calculation**:
```
decode_scale[i, k, 0] = (vec_max[i, k, 0] / 6.0) * 268.8
                       = vec_max[i, k, 0] * 44.8
```

**Example blocks**:
```
Block [0, 0]: vec_max=3.0
  decode_scale = 3.0 * 44.8 = 134.4

Block [10, 5]: vec_max=8.4
  decode_scale = 8.4 * 44.8 = 376.32
```

**Clamp to FP8 range and convert**:
```python
decode_scale = torch.min(
    decode_scale,
    torch.tensor(
        torch.finfo(torch.float32).max,
        device=decode_scale.device,
        dtype=torch.float32,
    ),
)
decode_scale = torch.clamp(decode_scale, min=-FLOAT8_E4M3_MAX, max=FLOAT8_E4M3_MAX)
# Clamp to [-448.0, 448.0]
decode_scale = decode_scale.to(torch.float8_e4m3fn)
```

**After clamping and conversion**:
```
Block [0, 0]: 134.4 → clamped to 134.4 → FP8 E4M3
Block [10, 5]: 376.32 → clamped to 376.32 → FP8 E4M3
```

**State**:
- `decode_scale`: (128, 8, 1) **torch.float8_e4m3fn**

#### Step 4d: Compute Final Encode Scale (Lines 508-515)

```python
encode_scale = torch.min(
    torch.div(1.0, decode_scale.to(torch.float32) * global_decode_scale),
    torch.tensor(
        torch.finfo(torch.float32).max,
        device=decode_scale.device,
        dtype=torch.float32,
    ),
)
```

**Calculation**:
```
encode_scale[i, k, 0] = 1.0 / (decode_scale[i, k, 0] * global_decode_scale)
```

**Example**:
```
Block [0, 0]:
  decode_scale = 134.4 (FP8, converted back to float32 for math)
  encode_scale = 1.0 / (134.4 * 0.003720) ≈ 1.0 / 0.5 ≈ 2.0

Block [10, 5]:
  decode_scale = 376.32
  encode_scale = 1.0 / (376.32 * 0.003720) ≈ 1.0 / 1.4 ≈ 0.714
```

**State**:
- `encode_scale`: (128, 8, 1) float32 - factors to multiply input by before quantizing

**Physical meaning**:
- `encode_scale` maps input values to FP4 range [-6, 6]
- `decode_scale` (FP8) maps FP4 values back toward original range
- `global_decode_scale` provides final scaling to recover original magnitude

---

### Step 5: Apply Encoding Scale and Quantize (Lines 517-521)

#### Step 5a: Scale Input (Line 517)

```python
scaled_x = x.to(torch.float32) * encode_scale
```

**Broadcasting**:
```
x: (128, 8, 16) float32
encode_scale: (128, 8, 1) float32
scaled_x: (128, 8, 16) float32
```

**Calculation**:
```
scaled_x[i, k, l] = x[i, k, l] * encode_scale[i, k, 0]
```

**Example values**:
```
Block [0, 0], element [0, 0, 5]:
  x[0, 0, 5] = 2.5 (original value)
  encode_scale[0, 0, 0] = 2.0
  scaled_x[0, 0, 5] = 2.5 * 2.0 = 5.0

Block [10, 5], element [10, 5, 3]:
  x[10, 5, 3] = -3.6
  encode_scale[10, 5, 0] = 0.714
  scaled_x[10, 5, 3] = -3.6 * 0.714 ≈ -2.57
```

**State**:
- `scaled_x`: (128, 8, 16) float32 - input scaled to FP4 range

#### Step 5b: Clamp to FP4 Range (Line 519)

```python
clipped_x = torch.clamp(scaled_x, -FLOAT4_E2M1_MAX, FLOAT4_E2M1_MAX).reshape(m, n)
# Clamp to [-6.0, 6.0] and reshape to (128, 128)
```

**Effect**:
```
scaled_x[0, 0, 5] = 5.0 → clipped to 5.0 ✓
scaled_x[10, 5, 3] = -2.57 → clipped to -2.57 ✓

If scaled_x[i, j, k] = 8.5 → clipped to 6.0
If scaled_x[i, j, k] = -7.2 → clipped to -6.0
```

**State**:
- `clipped_x`: (128, 128) float32 - values in [-6.0, 6.0]

#### Step 5c: Convert to FP4 (Line 521)

```python
return cast_to_fp4x2(clipped_x), decode_scale.squeeze(-1)
```

**`cast_to_fp4x2` function** (lines 50-72):

This function maps float32 values to 4-bit FP4 E2M1 format and packs two FP4 values per byte.

**FP4 E2M1 Format**:
- 1 sign bit, 2 exponent bits, 1 mantissa bit
- 16 representable values (including +/-0):

| FP4 Code | Value | FP4 Code | Value |
|----------|-------|----------|-------|
| 0 | 0.0 | 8 | -0.0 |
| 1 | 0.5 | 9 | -0.5 |
| 2 | 1.0 | 10 | -1.0 |
| 3 | 1.5 | 11 | -1.5 |
| 4 | 2.0 | 12 | -2.0 |
| 5 | 3.0 | 13 | -3.0 |
| 6 | 4.0 | 14 | -4.0 |
| 7 | 6.0 | 15 | -6.0 |

**Quantization thresholds**:
```python
result = torch.zeros_like(x, dtype=torch.uint8)
result[(x >= 0.0) & (x <= 0.25)] = 0    # [0.0, 0.25] → 0 (represents 0.0)
result[(x > 0.25) & (x < 0.75)] = 1     # (0.25, 0.75) → 1 (represents 0.5)
result[(x >= 0.75) & (x <= 1.25)] = 2   # [0.75, 1.25] → 2 (represents 1.0)
result[(x > 1.25) & (x < 1.75)] = 3     # (1.25, 1.75) → 3 (represents 1.5)
result[(x >= 1.75) & (x <= 2.5)] = 4    # [1.75, 2.5] → 4 (represents 2.0)
result[(x > 2.5) & (x < 3.5)] = 5       # (2.5, 3.5) → 5 (represents 3.0)
result[(x >= 3.5) & (x <= 5.0)] = 6     # [3.5, 5.0] → 6 (represents 4.0)
result[x > 5.0] = 7                     # (5.0, ∞) → 7 (represents 6.0)

# Negative values (codes 8-15)
result[(x >= -0.25) & (x < -0.0)] = 8   # [-0.25, 0) → 8 (represents -0.0)
result[(x < -0.25) & (x > -0.75)] = 9   # (-0.75, -0.25) → 9 (represents -0.5)
result[(x <= -0.75) & (x >= -1.25)] = 10  # [-1.25, -0.75] → 10 (represents -1.0)
# ... (similar for -1.5, -2.0, -3.0, -4.0, -6.0)
```

**Example quantization**:
```
clipped_x[0, 5] = 5.0 → code 7 (represents 6.0 in FP4)
clipped_x[0, 6] = -2.57 → code 13 (represents -3.0 in FP4)
clipped_x[0, 7] = 1.1 → code 2 (represents 1.0 in FP4)
clipped_x[0, 8] = 0.3 → code 1 (represents 0.5 in FP4)
```

**Packing two FP4 values per byte** (line 72):
```python
return result[:, ::2] + result[:, 1::2] * 16
```

This packs even and odd columns together:
```
Row 0, columns [0, 1]:
  result[0, 0] = 7 (even)
  result[0, 1] = 13 (odd)
  packed[0, 0] = 7 + 13*16 = 7 + 208 = 215

Row 0, columns [2, 3]:
  result[0, 2] = 2 (even)
  result[0, 3] = 1 (odd)
  packed[0, 1] = 2 + 1*16 = 2 + 16 = 18
```

**Final packed tensor**:
- **Input**: (128, 128) float32
- **Output**: (128, 64) uint8 (each byte contains two 4-bit values)

**State**:
- Quantized data: (128, 64) **uint8**
- Decode scales: (128, 8) **torch.float8_e4m3fn** (squeezed from (128, 8, 1))

---

## Final Return Values

```python
return cast_to_fp4x2(clipped_x), decode_scale.squeeze(-1)
```

**Returns**:
1. **Quantized tensor**: (128, 64) uint8
   - Each byte contains two packed FP4 values
   - Represents original (128, 128) tensor

2. **Decode scales**: (128, 8) torch.float8_e4m3fn
   - One scale per 1×16 block
   - Used for dequantization: `dequant[i, k*16:(k+1)*16] = fp4_values * decode_scale[i, k] * global_decode_scale`

---

## Two-Level Scaling Hierarchy Explained

### Why Two Levels?

The design uses **blockwise FP8 scales** and a **global scalar** to balance:
1. **Dynamic range**: Adapt to per-block statistics
2. **Precision**: Store scales in FP8 (1 byte) instead of FP32 (4 bytes)
3. **Accuracy**: Global scale compensates for FP8 quantization error

### Roles of Each Scale

#### 1. Blockwise Decode Scale (FP8 E4M3)

**Stored in**: `decode_scale` tensor, shape (128, 8), dtype=torch.float8_e4m3fn

**Purpose**: Provides per-block normalization to handle varying magnitudes across the tensor.

**Encoding**:
```
decode_scale[i, k] = (vec_max[i, k] / 6.0) * global_encode_scale
                   = vec_max[i, k] * (448.0 / global_amax)
```

**Decoding** (conceptual):
```
fp4_values[i, k*16:(k+1)*16] are in range [-6, 6]
scaled_values = fp4_values * decode_scale[i, k]
```

**Range**: [-448, 448] (FP8 E4M3 range)

#### 2. Global Decode Scale (Scalar FP32)

**Stored implicitly**: Can be computed as `1.0 / global_encode_scale`

**Purpose**: Final scaling factor to map from FP8-scaled values back to original magnitude.

**Value**:
```
global_decode_scale = global_amax / (448.0 * 6.0)
                    = global_amax / 2688.0
```

**With our example** (`global_amax = 10.0`):
```
global_decode_scale = 10.0 / 2688.0 ≈ 0.003720
```

**In our example**: `global_decode_scale ≈ 0.003720`

#### 3. Full Dequantization Formula

```
original_value[i, j] ≈ fp4_code[i, j] * decode_scale[i, j//16] * global_decode_scale
```

**Step-by-step**:
1. **FP4 → Float**: `fp4_code` → FP4 representation (e.g., 7 → 6.0)
2. **Apply blockwise scale**: Multiply by `decode_scale[i, k]` (FP8 value)
3. **Apply global scale**: Multiply by `global_decode_scale`

**Example reconstruction**:
```
Block [0, 0], position [0, 5]:
  fp4_code = 7 → fp4_value = 6.0
  decode_scale[0, 0] = 134.4 (FP8)
  global_decode_scale = 0.003720

  reconstructed = 6.0 * 134.4 * 0.003720
                ≈ 6.0 * 0.5
                ≈ 3.0

  (Compare to original: x[0, 0, 5] = 2.5, quantization error ≈ 0.5)
```

### Why Not Just Use FP32 Scales?

**Memory savings**:
- FP32 scales: (128, 8) × 4 bytes = 4 KB
- FP8 scales: (128, 8) × 1 byte = 1 KB
- **75% reduction** in scale storage

**Tradeoff**: Small loss in precision, compensated by the global scale.

---

## Data Type Summary at Each Stage

| Step | Variable | Shape | Dtype | Description |
|------|----------|-------|-------|-------------|
| Input | `x` | (128, 128) | bf16/fp32 | Original tensor |
| Input | `global_amax` | (1,) | float32 | Max(abs(x)) |
| 2 | `x_reshaped` | (128, 8, 16) | bf16/fp32 | Reshaped for blockwise ops |
| 2 | `vec_max` | (128, 8, 1) | float32 | Per-block amax |
| 3 | `decode_scale` (init) | (128, 8, 1) | float32 | vec_max / 6.0 |
| 4a | `global_encode_scale` | (1,) | float32 | 2688.0 / global_amax |
| 4b | `global_decode_scale` | (1,) | float32 | 1.0 / global_encode_scale |
| 4c | `decode_scale` (norm) | (128, 8, 1) | **torch.float8_e4m3fn** | Blockwise scales in FP8 |
| 4d | `encode_scale` | (128, 8, 1) | float32 | 1 / (decode_scale × global_decode_scale) |
| 5a | `scaled_x` | (128, 8, 16) | float32 | Input scaled to FP4 range |
| 5b | `clipped_x` | (128, 128) | float32 | Clamped to [-6, 6] |
| 5c | **Quantized output** | **(128, 64)** | **uint8** | **Packed FP4 values** |
| 5c | **Decode scales output** | **(128, 8)** | **torch.float8_e4m3fn** | **Blockwise decode scales** |

---

## Key Insights

1. **Blockwise quantization** adapts to local statistics, preventing outliers in one block from affecting others.

2. **Two-level scaling** (blockwise FP8 + global scalar) provides:
   - Fine-grained adaptation (per-block)
   - Memory efficiency (FP8 storage)
   - Accurate reconstruction (global correction)

3. **FP4 E2M1** has only 16 representable values, so:
   - Quantization error is ~±0.25 at small values
   - Quantization error is ~±1.0 at large values
   - Total reconstruction error depends on both FP4 quantization and scale quantization

4. **Packing** reduces memory by 2× (two 4-bit values per byte)

5. **Global scale is NOT stored** in the returned tuple but is **implicitly known** from `global_amax` and can be recomputed during dequantization or GEMM operations.

---

## Comparison: 1D vs 2D Quantization

| Aspect | 1D Quantization (1×16) | 2D Quantization (16×16) |
|--------|------------------------|-------------------------|
| Block size | 1 row × 16 columns | 16 rows × 16 columns |
| Scales shape (for 128×128) | (128, 8) | (128, 8) (replicated per row) |
| Granularity | Fine (per row-chunk) | Coarser (per square block) |
| Typical use case | Activations (row-major) | Weights (spatial locality) |
| Memory overhead | Higher (more scales) | Lower (fewer unique scales) |

For a 128×128 tensor:
- **1D**: 128 × 8 = 1024 unique scales
- **2D**: 8 × 8 = 64 unique scales, replicated to (128, 8) shape

---

## Usage in Full Quantization Pipeline

From the calling function `_quantize` (line 625):

```python
qx, sx = self._quantize_blockwise_reference(
    x_padded,
    global_amax_row,
    self.quant_tile_shape[1],  # tile_len_x = 16
    self.quant_tile_shape[0],  # tile_len_y = 1 or 16
    pow_2_scales=self.pow_2_scales,
    eps=self.eps,
)
```

The returned values are stored in `NVFP4TensorRef`:
- `qx` → `data` (quantized tensor)
- `sx` → `scale` (decode scales)
- `global_amax_row` → `global_amax_row` (stored separately)

During GEMM (line 789-887), the dequantization uses:
```python
high_precision_x = cast_from_fp4x2(qx, out_dtype)
sx = sx.to(torch.float32)  # Convert FP8 scales to FP32
alpha = partial_alpha / (6.0 * 6.0 * 448.0 * 448.0)
# partial_alpha contains the global amax information
```

The scales are applied during the GEMM accumulation (line 871):
```python
y += torch.outer(sx_block, sw_block) * high_precision_gemm_ref(...)
```

And the global scale is applied at the end (line 877):
```python
y = alpha * y
```

This demonstrates the **three-stage dequantization**:
1. FP4 → float (nominal [-6, 6] range)
2. Multiply by blockwise FP8 scales
3. Multiply by global scalar (embedded in `alpha`)
