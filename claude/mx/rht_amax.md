# Random Hadamard Transform Amax Analysis

## Purpose of `with_post_rht_amax`

The `with_post_rht_amax` flag controls **when** the absolute maximum (amax) is computed relative to the Random Hadamard Transform (RHT) during NVFP4 quantization.

## What Happens When Set to True

When `with_post_rht_amax=True` is set (as it is for `scaling_fwd` input and output), the quantizer calls a specialized fused kernel at [quantizer.cpp:1493-1500](../../transformer_engine/pytorch/csrc/quantizer.cpp#L1493-L1500):

```cpp
if (this->with_post_rht_amax) {
    // We need:
    // 1. Rowwise amax = amax for input
    // 2. Columnwise amax = amax for RHT(input.t)
    NVTE_SCOPED_GIL_RELEASE({
        nvte_hadamard_transform_amax(input.data(), out.data(), 0,
                                     this->rht_matrix_random_sign_mask_t, stream);
    });
}
```

## The Two Amax Values Computed

The `nvte_hadamard_transform_amax` kernel (implemented in [hadamard_transform.cu:743-856](../../transformer_engine/common/hadamard_transform/hadamard_transform.cu#L743-L856)) simultaneously computes **two different amax values**:

### 1. **Rowwise Amax** (Identity)
```
amax_rowwise = max(abs(input))
```
- Computes the absolute maximum of the **original untransformed input**
- Used for quantizing the tensor in its original row-major layout
- No RHT applied to this computation

### 2. **Columnwise Amax** (Transposed + RHT)
```
amax_columnwise = max(abs(RHT(input.transpose())))
```
- **First transposes** the input matrix
- **Then applies** the Random Hadamard Transform
- **Then computes** the absolute maximum of the result
- Used for quantizing the transposed view that will be used in the backward pass

## Why Computing Amax AFTER RHT Matters

### The Problem with Pre-RHT Amax

If you computed amax before applying RHT (the "pre-RHT amax" path, which errors out at [quantizer.cpp:1503](../../transformer_engine/pytorch/csrc/quantizer.cpp#L1503)):

```cpp
// NOT SUPPORTED:
amax = max(abs(input))           // Compute amax first
transformed = RHT(input)         // Then transform
quantized = quantize(transformed, amax)  // ❌ Scale doesn't match!
```

**The issue**: The RHT redistributes values across the matrix to spread out outliers. The maximum value **changes** after transformation, so scales based on the pre-RHT amax would be suboptimal.

### The Solution: Post-RHT Amax

With `with_post_rht_amax=True`:

```cpp
transformed = RHT(input)                    // Transform first
amax = max(abs(transformed))                // Then compute amax
quantized = quantize(transformed, amax)     // ✓ Scale matches!
```

**The benefit**: Scaling factors are computed based on the **actual data distribution after transformation**, ensuring optimal use of the limited FP4 range.

## Implementation Details

The kernel uses highly optimized GPU operations ([hadamard_transform.cu:199-270](../../transformer_engine/common/hadamard_transform/hadamard_transform.cu#L199-L270)):

1. **Loads 16×16 tile** via TMA (Tensor Memory Accelerator)
2. **Applies Hadamard matrix multiply** using Tensor Cores
3. **Simultaneously tracks amax** during the transform using inline PTX:
   ```cpp
   asm volatile("max.xorsign.abs.bf16x2 %0, %1, %2;\n\t"
                : "=r"(local_amax_reg)
                : "r"(local_amax_reg), "r"(temp_amax_reg));
   ```
4. **Transposes on-the-fly** for columnwise amax
5. **Reduces across warps** using shuffle operations
6. **Combines results atomically** across thread blocks

## Why Both Rowwise and Columnwise?

NVFP4 uses **2D block quantization** with separate rowwise and columnwise scale factors:

- **Rowwise scales**: Used for forward pass computation (original layout)
- **Columnwise scales**: Used for backward pass computation (transposed layout)

By computing both amax values in a single fused kernel, the implementation:
- Avoids multiple passes over the data
- Leverages the same memory loads for both computations
- Maximizes GPU utilization

## Trace Through the Code Path

1. **Recipe sets the flag** ([quantization.py:1322](../../transformer_engine/pytorch/quantization.py#L1322)):
   ```python
   with_post_rht_amax=qparams.random_hadamard_transform
   ```

2. **C++ extracts the flag** ([quantizer.cpp:1139](../../transformer_engine/pytorch/csrc/quantizer.cpp#L1139)):
   ```cpp
   this->with_post_rht_amax = quantizer.attr("with_post_rht_amax").cast<bool>();
   ```

3. **Quantization checks the flag** ([quantizer.cpp:1493](../../transformer_engine/pytorch/csrc/quantizer.cpp#L1493)):
   ```cpp
   if (this->with_post_rht_amax) {
       nvte_hadamard_transform_amax(...);  // Fused kernel
   }
   ```

4. **CUDA kernel executes** ([hadamard_transform.cu:355-499](../../transformer_engine/common/hadamard_transform/hadamard_transform.cu#L355-L499)):
   - Computes RHT via Tensor Core matrix multiply
   - Tracks amax for both identity and transposed views
   - Uses atomic operations to write final amax values

## Comparison Table

| Aspect | Pre-RHT Amax ❌ | Post-RHT Amax ✓ |
|--------|----------------|-----------------|
| **Rowwise amax** | `max(abs(input))` | `max(abs(input))` |
| **Columnwise amax** | `max(abs(input.T))` | `max(abs(RHT(input.T)))` |
| **Advantage** | Simpler (1 kernel) | Accurate scales for transformed data |
| **Problem** | Scales don't match RHT distribution | None |
| **Status** | Not implemented | ✓ Only supported path |
| **Performance** | N/A | Fused kernel is very efficient |

## Key Code Locations

- **Flag definition**: [nvfp4_tensor.py:119](../../transformer_engine/pytorch/tensor/nvfp4_tensor.py#L119)
- **Recipe configuration**: [quantization.py:1322](../../transformer_engine/pytorch/quantization.py#L1322), [quantization.py:1336](../../transformer_engine/pytorch/quantization.py#L1336)
- **C++ quantizer setup**: [quantizer.cpp:1139](../../transformer_engine/pytorch/csrc/quantizer.cpp#L1139)
- **Amax computation dispatch**: [quantizer.cpp:1493-1500](../../transformer_engine/pytorch/csrc/quantizer.cpp#L1493-L1500)
- **Kernel implementation**: [hadamard_transform.cu:743-856](../../transformer_engine/common/hadamard_transform/hadamard_transform.cu#L743-L856)
- **Kernel header**: [hadamard_transform.h:46-47](../../transformer_engine/common/include/transformer_engine/hadamard_transform.h#L46-L47)

## Summary

**Bottom line**: `with_post_rht_amax=True` ensures that quantization scales are computed based on the **actual post-transform data range**, which is essential for optimal FP4 quantization quality when using Random Hadamard Transforms. The RHT redistributes outliers across the matrix, changing the data distribution, so amax must be computed after the transform to ensure the scaling factors accurately reflect the range of values that will actually be quantized.
