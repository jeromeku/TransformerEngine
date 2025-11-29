# NVFP4 Two-Level Scaling Hierarchy - Visual Explanation

## Overview

NVFP4 quantization uses a **two-level scaling hierarchy** to achieve both fine-grained adaptation and memory efficiency:

1. **Blockwise scales** (FP8 E4M3): Per-block normalization stored with the quantized data
2. **Global scale** (scalar): Single scaling factor computed from global statistics

## Visual Representation

### Input Tensor and Block Structure

```
Input Tensor x: (128, 128) bf16/fp32
Global amax: 10.0 (max absolute value in entire tensor)

┌─────────────────────────────────────────────────────────────────┐
│  Block Structure (1D quantization: 1×16 blocks)                 │
│                                                                  │
│  Row 0:  [Block 0,0] [Block 0,1] [Block 0,2] ... [Block 0,7]   │
│          ← 16 cols→  ← 16 cols→  ← 16 cols→       ← 16 cols→    │
│                                                                  │
│  Row 1:  [Block 1,0] [Block 1,1] [Block 1,2] ... [Block 1,7]   │
│  ...                                                             │
│  Row 127:[Block 127,0] ... ... ... ... ... ... [Block 127,7]   │
│                                                                  │
│  Total: 128 rows × 8 blocks/row = 1024 blocks                   │
└─────────────────────────────────────────────────────────────────┘
```

### Per-Block Statistics

```
For each block [i, k] (row i, block k):

  ┌─────────────────────┐
  │  Block [i, k]       │
  │  16 elements        │
  │                     │
  │  [e₀, e₁, ..., e₁₅] │
  └─────────────────────┘
           ↓
     Compute amax
           ↓
  vec_max[i, k] = max(|e₀|, |e₁|, ..., |e₁₅|)

Example:
  Block [0, 0]: vec_max = 3.0
  Block [0, 1]: vec_max = 1.0
  Block [10, 5]: vec_max = 8.4
  Block [127, 7]: vec_max = 0.5
```

## Scaling Computation Flow

### Step 1: Compute Initial Decode Scale

```
For each block [i, k]:

  vec_max[i, k]
       ↓
  divide by FP4_MAX (6.0)
       ↓
  decode_scale_init[i, k] = vec_max[i, k] / 6.0

Example:
  Block [0, 0]: 3.0 / 6.0 = 0.5
  Block [0, 1]: 1.0 / 6.0 ≈ 0.167
  Block [10, 5]: 8.4 / 6.0 = 1.4
```

**Purpose**: Map block's amax to the FP4 range [-6, 6].

### Step 2: Compute Global Scales

```
Global statistics (entire tensor):

  global_amax = 10.0 (given)

       ↓

  global_encode_scale = (FP8_MAX × FP4_MAX) / global_amax
                      = (448.0 × 6.0) / 10.0
                      = 2688.0 / 10.0
                      = 268.8
       ↓

  global_decode_scale = 1.0 / global_encode_scale
                      = 1.0 / 268.8
                      ≈ 0.003720
```

**Purpose**:
- `global_encode_scale`: Normalizes the entire tensor's range to fit FP8×FP4 combined range
- `global_decode_scale`: Reverses this normalization during dequantization

### Step 3: Normalize Blockwise Scales to FP8

```
For each block [i, k]:

  decode_scale_init[i, k]
       ↓
  multiply by global_encode_scale
       ↓
  decode_scale_fp32[i, k] = decode_scale_init[i, k] × global_encode_scale
       ↓
  clamp to [-448.0, 448.0]
       ↓
  convert to FP8 E4M3
       ↓
  decode_scale[i, k] ∈ torch.float8_e4m3fn

Example:
  Block [0, 0]: 0.5 × 268.8 = 134.4 → FP8(134.4)
  Block [0, 1]: 0.167 × 268.8 ≈ 44.8 → FP8(44.8)
  Block [10, 5]: 1.4 × 268.8 = 376.32 → FP8(376.32)
```

**Purpose**: Store blockwise scales in memory-efficient FP8 format while preserving relative magnitudes.

### Step 4: Compute Encode Scale for Quantization

```
For each block [i, k]:

  decode_scale[i, k] (FP8)
       ↓
  convert to float32
       ↓
  encode_scale[i, k] = 1.0 / (decode_scale[i, k] × global_decode_scale)

Example:
  Block [0, 0]:
    decode_scale = 134.4
    encode_scale = 1.0 / (134.4 × 0.003720) = 1.0 / 0.5 = 2.0

  Block [0, 1]:
    decode_scale = 44.8
    encode_scale = 1.0 / (44.8 × 0.003720) ≈ 1.0 / 0.167 ≈ 6.0

  Block [10, 5]:
    decode_scale = 376.32
    encode_scale = 1.0 / (376.32 × 0.003720) ≈ 1.0 / 1.4 ≈ 0.714
```

**Purpose**: Scale each block's input values to the FP4 range [-6, 6] before quantization.

## Complete Quantization Pipeline

```
Original Value x[i, j] in Block [i, k] where j ∈ [k×16, (k+1)×16)
       ↓
─────────────────────────────────────────────────────────────
ENCODING (Quantization)
─────────────────────────────────────────────────────────────
       ↓
[1] Multiply by encode_scale[i, k]
       ↓
    scaled_x = x[i, j] × encode_scale[i, k]
       ↓
[2] Clamp to [-6.0, 6.0]
       ↓
    clipped_x = clamp(scaled_x, -6.0, 6.0)
       ↓
[3] Quantize to FP4 E2M1 (4 bits)
       ↓
    fp4_code ∈ {0, 1, ..., 15}
       ↓
[4] Pack two FP4 values per byte
       ↓
    quantized_data: uint8

─────────────────────────────────────────────────────────────
STORED DATA
─────────────────────────────────────────────────────────────

    • quantized_data: (128, 64) uint8
    • decode_scale: (128, 8) torch.float8_e4m3fn
    • global_amax: scalar float32 (stored separately)

─────────────────────────────────────────────────────────────
DECODING (Dequantization)
─────────────────────────────────────────────────────────────
       ↓
[1] Unpack FP4 codes from bytes
       ↓
    fp4_code → two 4-bit values per byte
       ↓
[2] Convert FP4 code to FP4 value
       ↓
    fp4_value ∈ {0.0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0}
       ↓
[3] Multiply by blockwise decode scale (FP8 → FP32)
       ↓
    intermediate = fp4_value × decode_scale[i, k]
       ↓
[4] Multiply by global decode scale
       ↓
    reconstructed = intermediate × global_decode_scale
       ↓
       ↓
Reconstructed Value ≈ x[i, j]
```

## Numerical Example: End-to-End

### Block [0, 0] - Element at position [0, 5]

```
ENCODING:
─────────
Original: x[0, 5] = 2.5
Block stats: vec_max[0, 0] = 3.0
Global stats: global_amax = 10.0

Computed scales:
  global_encode_scale = 268.8
  global_decode_scale = 0.003720
  decode_scale[0, 0] = 134.4 (FP8)
  encode_scale[0, 0] = 2.0

Quantization:
  scaled = 2.5 × 2.0 = 5.0
  clipped = clamp(5.0, -6.0, 6.0) = 5.0
  fp4_code = 5 (represents 3.0 in FP4)

DECODING:
─────────
  fp4_value = 3.0 (from code 5)
  intermediate = 3.0 × 134.4 = 403.2
  reconstructed = 403.2 × 0.003720 ≈ 1.5

Quantization error: |2.5 - 1.5| = 1.0
```

Wait, that doesn't look right. Let me recalculate...

Actually, looking at the FP4 thresholds:
```python
result[(x > 2.5) & (x < 3.5)] = 5  # Maps to FP4 value 3.0
```

But `scaled = 5.0` should map to:
```python
result[(x >= 3.5) & (x <= 5.0)] = 6  # Maps to FP4 value 4.0
```

Let me redo this:

```
ENCODING (corrected):
─────────
Original: x[0, 5] = 2.5
encode_scale[0, 0] = 2.0

Quantization:
  scaled = 2.5 × 2.0 = 5.0
  clipped = 5.0
  fp4_code = 6 (since 5.0 falls in [3.5, 5.0] → code 6 → FP4 value 4.0)

DECODING:
─────────
  fp4_value = 4.0 (from code 6)
  intermediate = 4.0 × 134.4 = 537.6

Wait, this is also wrong. Let me reconsider...
```

Actually, I think I'm confusing the encode/decode relationship. Let me trace through more carefully:

The key insight is:
```
encode_scale = 1.0 / (decode_scale × global_decode_scale)
```

So:
```
encode_scale[0, 0] = 1.0 / (134.4 × 0.003720)
                   = 1.0 / 0.5
                   = 2.0
```

And the full decode formula is:
```
reconstructed = fp4_value × decode_scale × global_decode_scale
              = fp4_value × 134.4 × 0.003720
              = fp4_value × 0.5
```

So if `fp4_value = 4.0`:
```
reconstructed = 4.0 × 0.5 = 2.0
```

And if `fp4_value = 6.0`:
```
reconstructed = 6.0 × 0.5 = 3.0
```

Let me redo the example properly:

```
ENCODING (corrected):
──────────────────
Original: x[0, 5] = 2.5
encode_scale[0, 0] = 2.0

Quantization:
  scaled = 2.5 × 2.0 = 5.0
  clipped = 5.0

  FP4 quantization: 5.0 falls in [3.5, 5.0] → code 6
  fp4_code = 6 → represents FP4 value 4.0

DECODING:
─────────
  fp4_value = 4.0 (from code 6)
  decode_scale[0, 0] = 134.4 (FP8)
  global_decode_scale = 0.003720

  reconstructed = 4.0 × 134.4 × 0.003720
                = 4.0 × 0.5
                = 2.0

Quantization error: |2.5 - 2.0| = 0.5
Relative error: 0.5 / 2.5 = 20%
```

### Block [10, 5] - Element at position [10, 85]

```
ENCODING:
─────────
Original: x[10, 85] = -3.6 (in block [10, 5]: cols [80:96])
Block stats: vec_max[10, 5] = 8.4
Global stats: global_amax = 10.0

Computed scales:
  decode_scale[10, 5] = 376.32 (FP8)
  encode_scale[10, 5] = 1.0 / (376.32 × 0.003720) = 1.0 / 1.4 ≈ 0.714

Quantization:
  scaled = -3.6 × 0.714 ≈ -2.57
  clipped = -2.57

  FP4 quantization: -2.57 falls in (-3.5, -2.5) → code 13
  fp4_code = 13 → represents FP4 value -3.0

DECODING:
─────────
  fp4_value = -3.0
  reconstructed = -3.0 × 376.32 × 0.003720
                = -3.0 × 1.4
                ≈ -4.2

Quantization error: |-3.6 - (-4.2)| = 0.6
Relative error: 0.6 / 3.6 ≈ 17%
```

## Scale Relationships

### Mathematical Identities

```
encode_scale × decode_scale × global_decode_scale = 1.0

Proof:
  encode_scale = 1.0 / (decode_scale × global_decode_scale)

  encode_scale × decode_scale × global_decode_scale
    = [1.0 / (decode_scale × global_decode_scale)] × decode_scale × global_decode_scale
    = 1.0 ✓
```

This ensures that encoding followed by decoding (ignoring FP4 quantization error) recovers the original value:

```
x → (×encode_scale) → scaled_x → [FP4 quant] → fp4_value →
  (×decode_scale) → (×global_decode_scale) → reconstructed ≈ x
```

### Why This Design?

1. **Block adaptation**: Each block has its own `encode_scale` based on local statistics
2. **Global normalization**: The global scale ensures all blocks are normalized consistently
3. **FP8 storage**: Storing `decode_scale` in FP8 saves memory (1 byte vs 4 bytes)
4. **Numerical stability**: The two-level design prevents overflow/underflow in either scale

## Memory Breakdown

For a 128×128 tensor:

### Original Tensor
```
128 × 128 × 2 bytes (bf16) = 32 KB
```

### Quantized Representation
```
Quantized data:  128 × 64 × 1 byte (uint8)     = 8 KB
Blockwise scales: 128 × 8 × 1 byte (FP8 E4M3) = 1 KB
Global amax:     1 × 4 bytes (fp32)            = 4 bytes
                                        Total  ≈ 9 KB
```

### Compression Ratio
```
32 KB / 9 KB ≈ 3.5× compression
```

### If We Used FP32 Blockwise Scales Instead
```
Quantized data:  8 KB
Blockwise scales: 128 × 8 × 4 bytes (fp32)    = 4 KB
Global amax:     4 bytes
                                        Total  = 12 KB

Compression ratio: 32 KB / 12 KB ≈ 2.7×
```

**FP8 scales provide an additional 1.3× compression** on the metadata.

## Summary

The two-level scaling hierarchy achieves:

✓ **Fine-grained adaptation**: Per-block scales handle varying magnitudes
✓ **Memory efficiency**: FP8 scales instead of FP32 (75% reduction in scale storage)
✓ **Numerical accuracy**: Global scale compensates for FP8 quantization error
✓ **Simple decode**: Three multiplications (FP4 → FP32 → FP32 → result)
✓ **Hardware friendly**: FP8 operations are accelerated on modern GPUs

The design balances **accuracy**, **memory footprint**, and **computational efficiency** for deep learning workloads.
