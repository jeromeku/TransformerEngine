# Sequence Parallelism and Amax Reduction

## Why Sequence Parallel + Column/Row Parallel Requires Amax Reduction

This document explains why TransformerEngine requires `with_amax_reduction=True` for specific tensor parallelism configurations with sequence parallelism enabled.

## Background: Tensor Parallelism vs Sequence Parallelism

### Tensor Parallelism (TP)
- **Splits the weight matrix** across GPUs
- **Column-parallel**: Weight matrix split column-wise → `out_features = out_features // tp_size`
- **Row-parallel**: Weight matrix split row-wise → `in_features = in_features // tp_size`
- See [linear.py:1158-1160](../../transformer_engine/pytorch/module/linear.py#L1158-L1160)

### Sequence Parallelism (SP)
- **Splits the sequence dimension** of activations across GPUs (in addition to TP)
- Only enabled when `sequence_parallel=True` AND `tp_size > 1`
- Each GPU holds: `[batch, seq_len // tp_size, hidden]` instead of full sequence
- Reduces activation memory by `tp_size` compared to standard TP
- See [linear.py:1162](../../transformer_engine/pytorch/module/linear.py#L1162)

**Key insight**: Sequence parallelism works **on top of** tensor parallelism, not as an alternative.

## The Problem: Inconsistent Quantization Scales

### Case 1: Column-Parallel Forward Pass (Input Quantization)

**Data flow with sequence parallelism** ([linear.py:136-138](../../transformer_engine/pytorch/module/linear.py#L136-L138)):

```python
with_input_all_gather_nccl = (
    parallel_mode == "column" and sequence_parallel and not ub_overlap_ag_fprop
)
```

When `sequence_parallel=True` and `parallel_mode="column"`:

```
Before All-Gather:
GPU0: input[0:N/4, :]     → local_amax0 = max(abs(input[0:N/4, :]))
GPU1: input[N/4:N/2, :]   → local_amax1 = max(abs(input[N/4:N/2, :]))
GPU2: input[N/2:3N/4, :]  → local_amax2 = max(abs(input[N/2:3N/4, :]))
GPU3: input[3N/4:N, :]    → local_amax3 = max(abs(input[3N/4:N, :]))
```

**Without amax reduction**:
- Each GPU quantizes its shard using **different scales** (based on different local amax values)
- All-gather concatenates these inconsistently quantized shards
- Result: **Numerical corruption** when dequantizing the full tensor

**The fix** ([linear.py:1680-1687](../../transformer_engine/pytorch/module/linear.py#L1680-L1687)):

```python
if self.sequence_parallel and self.parallel_mode == "column":
    # customize input_quantizer with amax reduction TP group
    self.quantizers["scaling_fwd"][
        tex.FP8FwdTensors.GEMM1_INPUT
    ].with_amax_reduction = True
    self.quantizers["scaling_fwd"][
        tex.FP8FwdTensors.GEMM1_INPUT
    ].amax_reduction_group = self.tp_group
```

### Case 2: Row-Parallel Backward Pass (Grad Output Quantization)

**Data flow with sequence parallelism** ([base.py:1147](../../transformer_engine/pytorch/module/base.py#L1147)):

```python
gather_grad_output = row_parallel_mode and ctx.sequence_parallel
```

When `sequence_parallel=True` and `parallel_mode="row"`:

```
Backward Pass:
GPU0: grad_output[0:N/4, :]     → local_amax0
GPU1: grad_output[N/4:N/2, :]   → local_amax1
GPU2: grad_output[N/2:3N/4, :]  → local_amax2
GPU3: grad_output[3N/4:N, :]    → local_amax3

All-gather needed before computing weight gradients
```

**The fix** ([linear.py:1689-1696](../../transformer_engine/pytorch/module/linear.py#L1689-L1696)):

```python
if self.sequence_parallel and self.parallel_mode == "row":
    # customize grad_output_quantizer with amax reduction TP group
    self.quantizers["scaling_bwd"][
        tex.FP8BwdTensors.GRAD_OUTPUT1
    ].with_amax_reduction = True
    self.quantizers["scaling_bwd"][
        tex.FP8BwdTensors.GRAD_OUTPUT1
    ].amax_reduction_group = self.tp_group
```

## Complete Call Path: How Amax Reduction Works

### Step 1: Setup in Linear Module

**File**: [linear.py:1679-1697](../../transformer_engine/pytorch/module/linear.py#L1679-L1697)

During `Linear` module initialization, the flags are set:

```python
def _setup_for_fsdp(self, fwd: bool) -> None:
    if fwd:
        if self.sequence_parallel and self.parallel_mode == "column":
            # Forward: quantize input before all-gather
            self.quantizers["scaling_fwd"][
                tex.FP8FwdTensors.GEMM1_INPUT
            ].with_amax_reduction = True
            self.quantizers["scaling_fwd"][
                tex.FP8FwdTensors.GEMM1_INPUT
            ].amax_reduction_group = self.tp_group
    else:
        if self.sequence_parallel and self.parallel_mode == "row":
            # Backward: quantize grad_output before all-gather
            self.quantizers["scaling_bwd"][
                tex.FP8BwdTensors.GRAD_OUTPUT1
            ].with_amax_reduction = True
            self.quantizers["scaling_bwd"][
                tex.FP8BwdTensors.GRAD_OUTPUT1
            ].amax_reduction_group = self.tp_group
```

### Step 2: Quantizer Initialization (Python → C++)

**File**: [quantizer.cpp:1136-1152](../../transformer_engine/pytorch/csrc/quantizer.cpp#L1136-L1152)

The C++ quantizer reads the flags from the Python object:

```cpp
NVFP4Quantizer::NVFP4Quantizer(const py::handle& quantizer) : Quantizer(quantizer) {
    this->dtype = quantizer.attr("dtype").cast<DType>();
    this->with_rht = quantizer.attr("with_rht").cast<bool>();
    this->with_post_rht_amax = quantizer.attr("with_post_rht_amax").cast<bool>();
    this->with_2d_quantization = quantizer.attr("with_2d_quantization").cast<bool>();
    this->stochastic_rounding = quantizer.attr("stochastic_rounding").cast<bool>();

    // Get amax reduction group if needed for NVFP4 AG
    const bool with_amax_reduction = quantizer.attr("with_amax_reduction").cast<bool>();
    c10::intrusive_ptr<dist_group_type> amax_reduction_group;
    if (with_amax_reduction) {
        auto group = quantizer.attr("_canonicalized_amax_reduction_group")();
        NVTE_CHECK(!group.is_none(), "NVFP4Quantizer could not canonicalize amax reduction group");
        amax_reduction_group = group.cast<c10::intrusive_ptr<dist_group_type>>();
    }
    this->with_amax_reduction = with_amax_reduction;
    this->amax_reduction_group = amax_reduction_group;
    // ...
}
```

The `_canonicalized_amax_reduction_group()` method ([nvfp4_tensor.py:334-336](../../transformer_engine/pytorch/tensor/nvfp4_tensor.py#L334-L336)) converts the process group to a C++ compatible format.

### Step 3: Local Amax Computation

**File**: [quantizer.cpp:1485-1528](../../transformer_engine/pytorch/csrc/quantizer.cpp#L1485-L1528)

During quantization, each GPU computes its **local amax** on its input shard:

```cpp
// Compute amax (local to this GPU's shard)
if (this->with_rht) {
    if (this->with_post_rht_amax) {
        // Compute amax after RHT transform
        nvte_hadamard_transform_amax(input.data(), out.data(), 0,
                                     this->rht_matrix_random_sign_mask_t, stream);
    }
} else {
    // Standard amax computation (no RHT)
    nvte_compute_amax_with_config(input.data(), out.data(), quant_config, stream);
}

// Copy amax to both rowwise and columnwise pointers if needed
if (rowwise_amax_ptr != amax_ptr && rowwise_amax_ptr != nullptr) {
    cudaMemcpyAsync(rowwise_amax_ptr, amax_ptr, sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);
}
if (columnwise_amax_ptr != amax_ptr && columnwise_amax_ptr != nullptr) {
    cudaMemcpyAsync(columnwise_amax_ptr, amax_ptr, sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);
}
```

At this point, each GPU has computed `local_amax_i = max(abs(shard_i))`.

### Step 4: AllReduce MAX Across TP Group

**File**: [quantizer.cpp:1531-1551](../../transformer_engine/pytorch/csrc/quantizer.cpp#L1531-L1551)

**This is the critical step** where amax values are synchronized:

```cpp
// amax reduction
if (this->with_amax_reduction) {
    std::vector<at::Tensor> amax_tensors;

    // Create tensor wrappers for amax values (no-op deleter, doesn't own data)
    auto make_amax_tensor = [](void* data_ptr) {
        return at::from_blob(
            data_ptr, std::vector<int64_t>{1},
            [](void*) {},  // deleter doing nothing since it doesn't own the data
            at::device(at::kCUDA).dtype(torch::kFloat32));
    };

    // Add rowwise and/or columnwise amax to reduction list
    if (rowwise_usage) {
        amax_tensors.push_back(make_amax_tensor(out.get_amax().data_ptr));
    }
    if (columnwise_usage) {
        amax_tensors.push_back(make_amax_tensor(out.get_columnwise_amax().data_ptr));
    }

    // Perform AllReduce(MAX) across the tensor parallel group
    c10d::AllreduceCoalescedOptions opts;
    opts.reduceOp = c10d::ReduceOp::MAX;  // ← Take MAX across all ranks
    NVTE_SCOPED_GIL_RELEASE({
        this->amax_reduction_group->allreduce_coalesced(amax_tensors, opts)->wait();
    });
}
```

**What happens**:
1. Each GPU sends its local amax to all other GPUs in the TP group
2. NCCL performs a `MAX` reduction: `global_amax = max(local_amax0, local_amax1, ..., local_amaxN)`
3. All GPUs receive the **same global_amax** value
4. All GPUs now use this **synchronized amax** to compute quantization scales

### Step 5: Quantization with Synchronized Scale

After amax reduction, all GPUs compute the same scale factor:

```
scale = compute_scale_from_amax(global_amax, dtype_max_value)
```

Now when quantizing:
```
GPU0: quantized_shard0 = quantize(shard0, scale=global_scale)
GPU1: quantized_shard1 = quantize(shard1, scale=global_scale)
GPU2: quantized_shard2 = quantize(shard2, scale=global_scale)
GPU3: quantized_shard3 = quantize(shard3, scale=global_scale)
```

### Step 6: All-Gather with Consistent Scales

When all-gathering the quantized shards:

```python
# All shards quantized with the same scale!
gathered = all_gather([quantized_shard0, quantized_shard1, quantized_shard2, quantized_shard3])

# Dequantization produces correct values
dequantized = dequantize(gathered, scale=global_scale)
```

## What Happens Without Amax Reduction?

### Numerical Corruption Example

Without amax reduction, each GPU uses different scales:

```
GPU0: shard0 has amax=1.0  → scale0=1.0  → quantized with scale0
GPU1: shard1 has amax=2.0  → scale1=2.0  → quantized with scale1
GPU2: shard2 has amax=0.5  → scale2=0.5  → quantized with scale2
GPU3: shard3 has amax=1.5  → scale3=1.5  → quantized with scale3

# After all-gather, the shards have inconsistent scaling!
gathered = concat([shard0_scaled_by_1.0, shard1_scaled_by_2.0,
                   shard2_scaled_by_0.5, shard3_scaled_by_1.5])

# Dequantization with any single scale produces wrong values
dequantized = dequantize(gathered, scale=???)  # Which scale to use?!
```

From test comments ([run_numerics.py:194-196](../../tests/pytorch/distributed/run_numerics.py#L194-L196)):

```python
# loose tolerances for fp8_cs because of sequence parallel & amax reduction
# so that each rank has a different scale_inv for computing Y when we have
# row parallel & sequence parallel, because we do the all_gather in backward pass
```

### Impact on Training

Without proper amax reduction:
1. **Forward pass errors**: Incorrect activations fed to next layer
2. **Backward pass errors**: Incorrect gradients computed
3. **Weight update errors**: Weights updated in wrong direction
4. **Training divergence**: Model fails to converge or produces NaN/Inf values

## Summary Table

| Configuration | Affected Tensor | Reason |
|---------------|----------------|--------|
| **Column-parallel + SP (forward)** | `GEMM1_INPUT` | Input shards all-gathered before GEMM |
| **Row-parallel + SP (backward)** | `GRAD_OUTPUT1` | Grad output shards all-gathered before weight gradient GEMM |

| Without Amax Reduction | With Amax Reduction |
|------------------------|---------------------|
| ❌ Each GPU: different local amax | ✓ All GPUs: same global amax |
| ❌ Each GPU: different scale | ✓ All GPUs: same scale |
| ❌ Inconsistent quantization | ✓ Consistent quantization |
| ❌ Numerical corruption after all-gather | ✓ Correct values after all-gather |
| ❌ Training divergence | ✓ Stable training |

## Key Code Locations

1. **Setup**: [linear.py:1679-1697](../../transformer_engine/pytorch/module/linear.py#L1679-L1697)
2. **Column-parallel check**: [linear.py:136-138](../../transformer_engine/pytorch/module/linear.py#L136-L138)
3. **Row-parallel check**: [base.py:1147](../../transformer_engine/pytorch/module/base.py#L1147)
4. **C++ initialization**: [quantizer.cpp:1136-1152](../../transformer_engine/pytorch/csrc/quantizer.cpp#L1136-L1152)
5. **AllReduce MAX**: [quantizer.cpp:1531-1551](../../transformer_engine/pytorch/csrc/quantizer.cpp#L1531-L1551)
6. **Process group canonicalization**: [nvfp4_tensor.py:334-336](../../transformer_engine/pytorch/tensor/nvfp4_tensor.py#L334-L336)

## Why Use TP Group (Not DP Group)?

The tensor parallel group is used because:
- **Sequence sharding happens within the TP group**: Different TP ranks hold different sequence shards of the same logical tensor
- **DP ranks hold different data**: Different batches, no need to synchronize amax
- **Communication pattern**: Only TP ranks participate in all-gather, so only they need consistent scales

## Bottom Line

**Sequence parallelism splits activations across the sequence dimension within the tensor parallel group. Since each TP rank computes amax on only its shard, an AllReduce(MAX) across the TP group is essential to ensure all ranks use the same quantization scale. This prevents numerical corruption when the shards are all-gathered back into the full tensor.**
