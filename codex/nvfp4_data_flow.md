# NVFP4 Data Flow for `experiments/test_mp.py` (FProp / DGrad / WGrad)

**Script**: [`experiments/test_mp.py`](../experiments/test_mp.py)  
**Layer**: `te.Linear(768, 2048, params_dtype=torch.bfloat16, bias=False)`  
**Input**: `inp ∈ BF16[1024, 768]`  
**Recipe**: [`NVFP4BlockScaling`](../transformer_engine/common/recipe/__init__.py#L360-L420)

We trace, step by step, the NVFP4 quantization and RHT/SR logic for the three GEMMs:

- **FProp GEMM**: `Y = X · Wᵀ`
- **DGrad GEMM**: `dX = dY · W`
- **WGrad GEMM**: `dW = dYᵀ · X`

For each GEMM we show:

- How activations / weights / grad outputs are prepared (quantized)  
- Where amax, RHT, SR, and FP4 quantization are applied  
- Which fused kernels are used and with what inputs / template params  
- The full call chain down to CUDA kernels (where relevant)

For detailed kernel internals (`quantize_transpose_nvfp4_kernel` / `_2D_kernel`) see  
[`codex/nvfp4_quantize_v2.md`](nvfp4_quantize_v2.md).

---

## 0. Quick Reference

### 0.1 Key Files (Click-Through Links)

- Test script: [`experiments/test_mp.py`](../experiments/test_mp.py#L1-L32)
- NVFP4 training recipe: [`NVFP4BlockScaling`](../transformer_engine/common/recipe/__init__.py#L360-L420)
- Quantizer factory (NVFP4): [`NVFP4BlockScalingRecipeState`](../transformer_engine/pytorch/quantization.py#L1280-L1360)
- Python NVFP4 quantizer:
  - Class & config: [`transformer_engine/pytorch/tensor/nvfp4_tensor.py:108-220`](../transformer_engine/pytorch/tensor/nvfp4_tensor.py#L108-L220)
- PyTorch Quantizer base:
  - `Quantizer` interface & `set_usage`, `quantize`, `quantize_impl`:  
    [`transformer_engine/pytorch/quantized_tensor.py:169-260`](../transformer_engine/pytorch/quantized_tensor.py#L169-L260)
- Linear module:
  - Forward autograd function (`_Linear.forward`):  
    [`transformer_engine/pytorch/module/linear.py:88-260`](../transformer_engine/pytorch/module/linear.py#L88-L260)
  - Backward autograd function (`_Linear.backward`):  
    [`transformer_engine/pytorch/module/linear.py:520-880`](../transformer_engine/pytorch/module/linear.py#L520-L880)
- C++ NVFP4 quantizer:
  - `NVFP4Quantizer` ctor & config import from Python:  
    [`transformer_engine/pytorch/csrc/quantizer.cpp:1136-1156`](../transformer_engine/pytorch/csrc/quantizer.cpp#L1136-L1156)
  - `NVFP4Quantizer::quantize_impl` (RHT, amax, SR, quantize):  
    [`transformer_engine/pytorch/csrc/quantizer.cpp:1446-1647`](../transformer_engine/pytorch/csrc/quantizer.cpp#L1446-L1647)
- Dispatcher:
  - Forward dispatcher `quantize_fwd_helper`:  
    [`transformer_engine/common/cast/dispatch/quantize.cuh:22-97`](../transformer_engine/common/cast/dispatch/quantize.cuh#L22-L97)
  - Backward dispatcher `quantize_bwd_helper`:  
    [`transformer_engine/common/cast/dispatch/quantize.cuh:116-206`](../transformer_engine/common/cast/dispatch/quantize.cuh#L116-L206)
- NVFP4 CUDA kernels:
  - 1D/2D kernels + launcher:  
    [`transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh`](../transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh)
- Fused multi-tensor NVFP4 quantization (not used in this single-GEMM example, but referenced):
  - `split_quantize_nvfp4_impl`:  
    [`transformer_engine/pytorch/csrc/extensions/cast.cpp:712-880`](../transformer_engine/pytorch/csrc/extensions/cast.cpp#L712-L880)

### 0.2 Roles & Recipe (NVFP4BlockScaling)

NVFP4 recipe configuration (simplified) — see  
[`NVFP4BlockScaling.__post_init__`](../transformer_engine/common/recipe/__init__.py#L392-L420):

- **Forward input / output (`fp4_quant_fwd_inp`)**:
  - `random_hadamard_transform = True`
  - `stochastic_rounding = False`
  - `fp4_2d_quantization = False`
- **Forward weights (`fp4_quant_fwd_weight`)**:
  - `random_hadamard_transform = False`
  - `stochastic_rounding = False`
  - `fp4_2d_quantization = True` (16×16 2D blocks)
- **Backward gradients (`fp4_quant_bwd_grad`)**:
  - `random_hadamard_transform = True`
  - `stochastic_rounding = True`
  - `fp4_2d_quantization = False`

The `NVFP4BlockScalingRecipeState` converts these into `NVFP4Quantizer` instances:

- Forward mode (`mode="forward"`): [`quantization.py:1280-1319`](../transformer_engine/pytorch/quantization.py#L1280-L1319)

  ```python
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
  ```

  - Index 0 → GEMM input quantizer: `with_rht=True`, `with_2d_quantization=False`, `SR=False`.
  - Index 1 → GEMM weight quantizer: `with_rht=False`, `with_2d_quantization=True` (2D weights), `SR=False`.
  - Index 2 → GEMM output quantizer: same QParams as input.

- Backward mode (`mode="backward"`): same config for all gradient quantizers:

  ```python
  if self.mode == "backward":
      return [
          NVFP4Quantizer(
              fp4_dtype=self.dtype,
              rowwise=True,
              columnwise=True,
              with_rht=self.recipe.fp4_quant_bwd_grad.random_hadamard_transform,  # True
              with_post_rht_amax=self.recipe.fp4_quant_bwd_grad.random_hadamard_transform,
              with_2d_quantization=self.recipe.fp4_quant_bwd_grad.fp4_2d_quantization,  # False
              stochastic_rounding=self.recipe.fp4_quant_bwd_grad.stochastic_rounding,   # True
          )
          for _ in range(self.num_quantizers)
      ]
  ```

So, for **our Linear**:

- FProp GEMM:
  - **Activation quantizer**: RHT ON (columnwise), SR OFF, 1D scaling.
  - **Weight quantizer**: RHT OFF, SR OFF, **2D scaling** (weights).
  - **Output quantizer**: RHT ON, SR OFF, 1D scaling (columnwise not used here).
- DGrad + WGrad GEMMs:
  - Gradient quantizers: RHT ON (columnwise), SR ON, 1D scaling.

---

## 1. High-Level Call Chain for `test_mp.py`

**Script**: [`experiments/test_mp.py`](../experiments/test_mp.py#L1-L32)

```python
fp8_recipe = NVFP4BlockScaling()
high_precision_dtype = torch.bfloat16
my_linear = te.Linear(768, 2048, params_dtype=high_precision_dtype, bias=False)

inp = torch.rand((1024, 768), dtype=high_precision_dtype, requires_grad=True).cuda()
with te.autocast(enabled=True, recipe=fp8_recipe):
    out_fp8 = my_linear(inp)
loss = out_fp8.mean()
loss.backward()
```

Shapes:

- `X = inp ∈ BF16[1024, 768]`
- `W ∈ BF16[2048, 768]`
- `Y = X · Wᵀ ∈ BF16[1024, 2048]` (stored as NVFP4 internally under autocast)

### 1.1 Mermaid Sequence Diagram (End-to-End)

```mermaid
sequenceDiagram
    participant Py as test_mp.py
    participant L as te.Linear
    participant F as _Linear.forward (autograd)
    participant QPy as NVFP4Quantizer (Python)
    participant QC as NVFP4Quantizer (C++)
    participant CAPI as nvte_quantize_v2
    participant Dsp as quantize_fwd/bwd_helper
    participant NV as nvfp4::quantize_transpose (1D/2D)

    Py->>L: my_linear(inp)  # FProp
    L->>F: _Linear.forward(..., input_quantizer, weight_quantizer, output_quantizer, ...)

    note over F: FProp GEMM: Y = X · Wᵀ
    F->>QPy: input_quantizer(inputmat)\n(activations, fwd)
    QPy->>QC: NVFP4Quantizer::quantize(...)  # C++ wrapper
    QC->>QC: quantize_impl(input, out, compute_amax=true)
    QC->>CAPI: nvte_quantize_v2(input, out, quant_config)
    CAPI->>Dsp: quantize_fwd_helper(..., scaling_mode=NVTE_NVFP4_1D_SCALING)
    Dsp->>NV: nvfp4::quantize_transpose<use_2d=false>(...)

    F->>QPy: weight_quantizer(weight)\n(weights, fwd 2D)
    QPy->>QC: NVFP4Quantizer::quantize(...)
    QC->>QC: quantize_impl(weight, out, compute_amax=true)
    QC->>CAPI: nvte_quantize_v2(weight, out, quant_config)
    Dsp->>NV: nvfp4::quantize_transpose<use_2d=true>(...)  # 2D 16×16 blocks

    F->>F: general_gemm(weightmat, inputmat_total,\nquantization_params=output_quantizer)
    F-->>Py: out_fp8

    Py->>Py: loss = out_fp8.mean(); loss.backward()

    Py->>F: _Linear.backward(grad_output)
    note over F: DGrad GEMM: dX = dY · W
    F->>QPy: grad_output_quantizer(grad_output)\n(DGrad path)
    QPy->>QC: NVFP4Quantizer::quantize(...)
    QC->>CAPI: nvte_quantize_v2(grad_output, out, quant_config)\n(SR ON, RHT ON (columnwise))
    Dsp->>NV: nvfp4::quantize_transpose<use_2d=false>(...)

    note over F: WGrad GEMM: dW = dYᵀ · X
    F->>QPy: input_quantizer(inputmat_total)\n(columnwise for WGrad)
    F->>QPy: grad_output_quantizer(grad_output)\n(columnwise for WGrad)
    QPy->>QC: NVFP4Quantizer::quantize(...)  # RHT+SR for columnwise
    QC->>CAPI: nvte_quantize_v2(...)
    Dsp->>NV: nvfp4::quantize_transpose<use_2d=false>(...)
```

### 1.2 Mermaid Flowchart (Data Transform per Role)

```mermaid
flowchart TD
    X[BF16 activations X: 1024×768]
    W[BF16 weights W: 2048×768]
    dY[BF16 grad_output dY: 1024×2048]

    subgraph FProp
      X --> Qx[NVFP4Quantizer (fwd_inp)\nwith_rht=True, SR=False]
      Qx --> X_rw[NVFP4 rowwise X_q\n1D scaling, no RHT]
      Qx --> X_cw[NVFP4 columnwise X_q^T\nRHT+quant, fused]

      W --> Qw[NVFP4Quantizer (fwd_weight)\nwith_2d_quantization=True]
      Qw --> W_rw[NVFP4 rowwise W_q\n2D 16×16 blocks]
      Qw --> W_cw[NVFP4 columnwise W_q^T\n2D 16×16 blocks]

      X_rw & W_rw --> GEMM_F[FProp GEMM\nY = X · Wᵀ]
      GEMM_F --> Y[BF16/NVFP4 output]
    end

    subgraph DGrad
      dY --> Qgy[NVFP4Quantizer (bwd_grad)\nwith_rht=True, SR=True]
      Qgy --> dY_rw[NVFP4 rowwise dY_q\n1D scaling, SR]
      Qgy --> dY_cw[NVFP4 columnwise dY_q^T\nRHT+SR (for WGrad)]

      dY_rw & W_rw --> GEMM_D[dX GEMM\n dX = dY · W]
      GEMM_D --> dX[BF16 dX]
    end

    subgraph WGrad
      X --> Qx_w[NVFP4Quantizer (fwd_inp)\ncolumnwise usage only]
      Qx_w --> X_cw_w[NVFP4 columnwise X_q^T\nRHT+quant (if enabled)]

      dY --> Qgy_w[NVFP4Quantizer (bwd_grad)\ncolumnwise usage]
      Qgy_w --> dY_cw_w[NVFP4 columnwise dY_q^T\nRHT+SR]

      dY_cw_w & X_cw_w --> GEMM_W[dW GEMM\n dW = dYᵀ · X]
      GEMM_W --> dW[BF16 dW or NVFP4 dW_q]
    end
```

### 1.3 Mermaid Class Diagram (Components)

```mermaid
classDiagram
    class NVFP4BlockScaling {
        +fp4_quant_fwd_inp: QParams
        +fp4_quant_fwd_weight: QParams
        +fp4_quant_bwd_grad: QParams
    }

    class NVFP4BlockScalingRecipeState {
        +make_quantizers()
    }

    class NVFP4Quantizer(Python) {
        +with_rht: bool
        +with_2d_quantization: bool
        +stochastic_rounding: bool
        +quantize_impl(tensor)
    }

    class NVFP4QuantizerCpp {
        +quantize_impl(input, out, noop_flag, compute_amax)
    }

    class Linear {
        +forward(...)
        +backward(...)
        +_get_quantizers()
    }

    class QuantizeDispatch {
        +quantize_fwd_helper()
        +quantize_bwd_helper()
    }

    class NVFPRuntime {
        +quantize_transpose_nvfp4_kernel()
        +quantize_transpose_nvfp4_2D_kernel()
    }

    NVFP4BlockScaling --> NVFP4BlockScalingRecipeState : configures
    NVFP4BlockScalingRecipeState --> NVFP4Quantizer : builds
    NVFP4Quantizer --> NVFP4QuantizerCpp : via tex.quantize()
    Linear --> NVFP4Quantizer : uses for input/weight/grad
    NVFP4QuantizerCpp --> QuantizeDispatch : calls
    QuantizeDispatch --> NVFPRuntime : launches kernels
```

---

## 2. FProp GEMM (`Y = X · Wᵀ`)

We now trace FProp end-to-end for the single GEMM in `test_mp.py`.

### 2.1 Linear Forward: Preparing Input & Weight

**Entry**: [`linear.py:88-140`](../transformer_engine/pytorch/module/linear.py#L88-L140)

```python
def forward(
    ctx,
    weight: torch.Tensor,
    inp: torch.Tensor,
    bias: Optional[torch.Tensor],
    non_tensor_args: Tuple,
) -> torch.Tensor:
    (
        is_first_microbatch,
        fp8,
        fp8_calibration,
        wgrad_store,
        input_quantizer,
        weight_quantizer,
        output_quantizer,
        grad_input_quantizer,
        grad_weight_quantizer,
        grad_output_quantizer,
        ...
    ) = non_tensor_args

    out_features, in_features = weight.shape
    assert inp.shape[-1] == in_features, "GEMM not possible"
```

For `te.Linear(768, 2048)`:

- `weight.shape = [2048, 768]`, `inp.shape = [1024, 768]`.
- `fp8=True` under `te.autocast` with NVFP4 recipe, so quantizers are active.

#### 2.1.1 Input Activation Quantization (FProp)

Still in `forward`, **no TP / UB overlap** in this simple script (defaults), so execution goes to the “Do not all-gather input tensor” branch:  
[`linear.py:140-184`](../transformer_engine/pytorch/module/linear.py#L140-L184)

```python
else:  # Do not all-gather input tensor
    if fp8 or debug:
        if isinstance(inputmat, QuantizedTensorStorage):
            inputmat.update_usage(rowwise_usage=True)
        else:
            if input_quantizer is None:
                raise ValueError("Missing quantizer for input tensor")
            input_quantizer.set_usage(
                rowwise=True,
                columnwise=backward_needs_input and not save_original_input,
            )
            inputmat = input_quantizer(inputmat)
            own_quantized_input = True
    else:
        inputmat = cast_if_needed(inp, activation_dtype)
    inputmat_total = inputmat
```

Here:

- `backward_needs_input=True` (weights require grad, and `is_grad_enabled=True`).
- `save_original_input=False` by default.
- So `input_quantizer.set_usage(rowwise=True, columnwise=True)` is called.
- Then `input_quantizer(inputmat)` invokes the **Python NVFP4Quantizer**:

```python
class NVFP4Quantizer(Quantizer):
    ...
    def quantize_impl(self, tensor: torch.Tensor) -> QuantizedTensor:
        """Quantize tensor implementation"""
        return tex.quantize(tensor, self)
```

**Effect**:

- `tex.quantize` calls into the C++ NVFP4 quantizer, which in turn calls `NVFP4Quantizer::quantize` and `NVFP4Quantizer::quantize_impl` (C++).

#### 2.1.2 Weight Quantization (FProp, 2D)

Forward then prepares the weight tensor:  
[`linear.py:260-320`](../transformer_engine/pytorch/module/linear.py#L260-L320)

```python
weightmat = weight
if fp8 or debug:
    if weight_quantizer is not None and not isinstance(weight, QuantizedTensor):
        columnwise_usage = is_grad_enabled and inp.requires_grad
        if not columnwise_usage:
            columnwise_usage = (
                is_fp8_activation_recompute_enabled()
                and not in_fp8_activation_recompute_phase()
            )
        weight_quantizer.set_usage(rowwise=True, columnwise=columnwise_usage)
    elif isinstance(weight, QuantizedTensor):
        weight_quantizer = weight._quantizer

    # Quantization happens in get_weight_workspace
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

- In our test:
  - `columnwise_usage=True` because we will compute WGrad later and recomputation is not enabled.
  - `weight_quantizer` is an NVFP4Quantizer built with `fp4_quant_fwd_weight` (2D scaling).
- `get_weight_workspace` ultimately calls the generic FP8 workspace helper in  
  [`base.py:1360-1425`](../transformer_engine/pytorch/module/base.py#L1360-L1425):

```python
if out is None:
    ...
    out = quantizer.quantize(tensor, dtype=workspace_dtype)
    ...
```

So both the **activation** and **weight** paths go through:

1. Python NVFP4Quantizer (`quantize_impl` → `tex.quantize`)
2. C++ NVFP4Quantizer (`NVFP4Quantizer::quantize_impl`)
3. `nvte_quantize_v2` → `quantize_fwd_helper` → NVFP4 kernels

### 2.2 C++ NVFP4Quantizer: FProp Activation vs Weight

#### 2.2.1 Ctor: Importing Config from Python

**Source**: [`quantizer.cpp:1136-1156`](../transformer_engine/pytorch/csrc/quantizer.cpp#L1136-L1156)

```cpp
NVFP4Quantizer::NVFP4Quantizer(const py::handle& quantizer) : Quantizer(quantizer) {
  this->dtype = quantizer.attr("dtype").cast<DType>();
  this->with_rht = quantizer.attr("with_rht").cast<bool>();
  this->with_post_rht_amax = quantizer.attr("with_post_rht_amax").cast<bool>();
  this->with_2d_quantization = quantizer.attr("with_2d_quantization").cast<bool>();
  this->stochastic_rounding = quantizer.attr("stochastic_rounding").cast<bool>();
  ...
  this->rht_matrix_random_sign_mask_t = quantizer.attr("rht_matrix_random_sign_mask_t").cast<int>();
  this->rht_matrix = quantizer.attr("rht_matrix").cast<at::Tensor>();
}
```

For our FProp roles:

- **Activation quantizer**:
  - `with_rht = True`
  - `with_post_rht_amax = True`
  - `with_2d_quantization = False`
  - `stochastic_rounding = False`
- **Weight quantizer**:
  - `with_rht = False`
  - `with_post_rht_amax = False`
  - `with_2d_quantization = True` (2D 16×16)
  - `stochastic_rounding = False`

#### 2.2.2 `quantize_impl`: Common Setup

**Source**: [`quantizer.cpp:1446-1487`](../transformer_engine/pytorch/csrc/quantizer.cpp#L1446-L1487)

```cpp
void NVFP4Quantizer::quantize_impl(const TensorWrapper& input, TensorWrapper& out,
                                   const std::optional<TensorWrapper>& noop_flag,
                                   bool compute_amax) {
  if (input.numel() == 0) {
    return;
  }

  auto stream = at::cuda::getCurrentCUDAStream();

  QuantizationConfigWrapper quant_config;
  if (noop_flag) {
    quant_config.set_noop_tensor(noop_flag->data());
  }
  quant_config.set_nvfp4_2d_quantization(this->with_2d_quantization);
  quant_config.set_stochastic_rounding(this->stochastic_rounding);

  size_t rows = 1;
  for (size_t i = 0; i < input.ndim() - 1; ++i) {
    rows *= input.size(i);
  }
  size_t cols = input.size(input.ndim() - 1);
```

For activations / weights from `test_mp.py`:

- `rows = 1024`, `cols = 768` for X.
- `rows = 2048`, `cols = 768` for W.

If SR is enabled (not the case for fwd), RNG state is initialized; for fwd:

- `stochastic_rounding=false` → no RNG state, deterministic quantization.

#### 2.2.3 FProp amax + RHT Path (Activation, with_rht=True)

**Source**: [`quantizer.cpp:1488-1529`](../transformer_engine/pytorch/csrc/quantizer.cpp#L1488-L1529)

```cpp
  bool eligible_for_rht_cast_fusion =
      input.dtype() == DType::kBFloat16 && rows % 64 == 0 && cols % 128 == 0;

  // Compute amax.
  if (this->with_rht) {
    if (input.dtype() != DType::kBFloat16) {
      NVTE_CHECK(false, "RHT is only supported for bfloat16 input");
    }
    if (this->with_post_rht_amax) {
      // We need:
      // 1. Rowwise amax = amax for input
      // 2. Columnwise amax = amax for RHT(input.t)
      NVTE_SCOPED_GIL_RELEASE({
        nvte_hadamard_transform_amax(input.data(), out.data(), 0,
                                     this->rht_matrix_random_sign_mask_t, stream);
      });
    } else {
      NVTE_CHECK(false, "Pre-RHT amax is not supported yet");
    }
  } else {  // Without RHT
    if (compute_amax) {
      ...
      NVTE_SCOPED_GIL_RELEASE(
          { nvte_compute_amax_with_config(input.data(), out.data(), quant_config, stream); });
      ...
    }
  }
```

For FProp activations (X):

- `input.dtype() == BF16`, `rows=1024`, `cols=768` satisfy fusion constraints:
  - `rows % 64 == 0`, `cols % 128 == 0` → **`eligible_for_rht_cast_fusion = true`**.
- Because `with_rht=true`, `with_post_rht_amax=true`:
  - `nvte_hadamard_transform_amax` computes:
    - Rowwise amax = amax(X) (no RHT).
    - Columnwise amax = amax(RHT(Xᵀ)) (RHT applied to transpose).

For FProp weights (W):

- `with_rht=false`, so we follow the `else` branch:
  - `nvte_compute_amax_with_config(W, out, quant_config)` computes a single amax(W).
  - Rowwise and columnwise amax pointers are forced to share this value.

#### 2.2.4 FProp Quantization: 1D vs 2D

At the tail of `quantize_impl`, after amax reduction and optional amax allreduce, we perform the actual quantization:

**Activation path (with_rht=true, 1D)**: [`quantizer.cpp:1553-1571`](../transformer_engine/pytorch/csrc/quantizer.cpp#L1553-L1571)

```cpp
  if (this->with_rht) {
    if (rowwise_usage) {
      TensorWrapper out_identity(out.scaling_mode());
      auto out_identity_data = out.get_rowwise_data();
      auto out_identity_scale_inv = out.get_rowwise_scale_inv();
      auto out_identity_amax = out.get_amax();
      out_identity.set_rowwise_data(...);
      out_identity.set_rowwise_scale_inv(...);
      out_identity.set_amax(...);

      NVTE_SCOPED_GIL_RELEASE(
          { nvte_quantize_v2(input.data(), out_identity.data(), quant_config, stream); });
    }

    if (columnwise_usage) {
      ...
      if (!eligible_for_rht_cast_fusion) {
        // allocate RHT output, call nvte_hadamard_transform, then nvte_quantize_v2
      } else {
        // RHT cast fusion kernel.
        auto rht_matrix_nvte = makeTransformerEngineTensor(this->rht_matrix);
        NVTE_SCOPED_GIL_RELEASE({
          nvte_hadamard_transform_cast_fusion_columnwise(
              input.data(), out_transpose.data(), rht_matrix_nvte.data(), quant_config, stream);
        });
      }
    }
  } else {
    NVTE_SCOPED_GIL_RELEASE({ nvte_quantize_v2(input.data(), out.data(), quant_config, stream); });
  }
```

So for **activation X** during FProp:

- Rowwise usage:
  - Direct quantization **without RHT**:
    - `nvte_quantize_v2(X, out_identity, quant_config)` with
      - `nvfp4_2d_quantization=false`
      - `stochastic_rounding=false`
  - Dispatcher (`quantize_fwd_helper`) chooses:

    ```cpp
    case NVTE_NVFP4_1D_SCALING: {
      rows = input_tensor->flat_first_dim();  // 1024
      cols = input_tensor->flat_last_dim();   // 768
      use_optimized_kernel = (dtype == BF16) && rows%32==0 && cols%32==0 && has_data;
      if (use_optimized_kernel) {
        nvfp4::quantize_transpose</*use_2d_quantization=*/false>(...);
      } else {
        quantize_transpose_vector_blockwise_fp4(...);
      }
    }
    ```

  - This is the **1D NVFP4 kernel** described in `nvfp4_quantize_v2.md`.

- Columnwise usage:
  - Uses **fused RHT + quantization**:
    - `nvte_hadamard_transform_cast_fusion_columnwise(X, out_transpose, rht_matrix, quant_config)`.
  - This kernel:
    - Applies a 16×16 RHT to chunks of `Xᵀ`.
    - Computes amax for RHT(Xᵀ) and writes columnwise amax.
    - Quantizes RHT(Xᵀ) to NVFP4 with the same scaling logic (1D across columns).

For **weights W** during FProp:

- `with_rht=false`, `with_2d_quantization=true`:
  - `quant_config.nvfp4_2d_quantization=true`.
  - The code calls:

    ```cpp
    nvte_quantize_v2(W, out, quant_config);
    ```

  - Dispatcher (`quantize_fwd_helper`) sees `nvfp4_2d_quantization=true` and uses:

    ```cpp
    nvfp4::quantize_transpose</*use_2d_quantization=*/true>(...);
    ```

  - This path uses the **2D NVFP4 kernel** (`quantize_transpose_nvfp4_2D_kernel`) with:
    - 16×16 2D scaling blocks (weights recipe requirement).
    - No RHT, no SR.

### 2.3 FProp GEMM Call

Once `inputmat_total` and `weightmat` are quantized, forward calls the GEMM:

**Source**: [`linear.py:320-360`](../transformer_engine/pytorch/module/linear.py#L320-L360)

```python
nvtx_range_push(f"{nvtx_label}.gemm")
gemm_out, *_, reduce_scatter_out = general_gemm(
    weightmat,
    inputmat_total,
    quantization_params=output_quantizer,
    out_dtype=activation_dtype,
    bias=bias,
    use_split_accumulator=use_split_accumulator,
    ub=ub_obj,
    ub_type=ub_type,
    extra_output=reduce_scatter_out,
)
nvtx_range_pop(f"{nvtx_label}.gemm")
```

- For `test_mp.py`:
  - `weightmat` is NVFP4 (possibly compact) weight tensor (rowwise) of shape ≈ `[2048, 768/2]` plus scales etc.
  - `inputmat_total` is NVFP4 activation tensor (`NVFP4TensorStorage`) of shape ≈ `[1024, 768/2]` plus scales.
  - `quantization_params=output_quantizer` is another NVFP4Quantizer:
    - `with_rht=True`, `stochastic_rounding=False`, `with_2d_quantization=False`.
    - But `output_quantizer.set_usage(rowwise=True, columnwise=False)`, so **no RHT** is applied in practice (RHT only affects columnwise path).

`general_gemm` then:

- Dequantizes/feeds NVFP4 inputs to a GEMM kernel using Tensor Cores.
- Optionally quantizes the GEMM output via the `output_quantizer` (NVFP4), but with `columnwise_usage=False`, we only care about rowwise FP4 data if stored.

---

## 3. DGrad GEMM (`dX = dY · W`)

In backward, `_Linear.backward` computes gradients w.r.t. input and weights. We focus first on the **DGrad GEMM** and how `dY` is quantized.

### 3.1 Preparing Grad Output for DGrad

**Source**: [`linear.py:520-620`](../transformer_engine/pytorch/module/linear.py#L520-L620)

```python
def backward(ctx, grad_output: torch.Tensor) -> Tuple[...]:
    ...
    # Configure quantizer for grad output tensor
    if ctx.grad_output_quantizer is not None:
        quantizer = ctx.grad_output_quantizer
        quantizer.set_usage(rowwise=True, columnwise=True)
        if ctx.ub_overlap_ag:
            quantizer.set_usage(columnwise=False)

    # Adjust quantization direction if no wgrad needed (not our case)
    if (
        not ctx.use_bias
        and not ctx.requires_wgrad
        and ctx.grad_output_quantizer is not None
    ):
        ctx.grad_output_quantizer.set_usage(columnwise=False)

    # Prepare grad output tensor
    nvtx_range_push(f"{nvtx_label}.grad_output_preprocess")
    (
        grad_output,
        grad_bias,
    ) = TransformerEngineBaseModule.grad_output_preprocess(
        ctx,
        grad_output,
        ctx.parallel_mode == "row",
        ctx.grad_output_quantizer,
    )
    nvtx_range_pop(f"{nvtx_label}.grad_output_preprocess")
```

In `test_mp.py`:

- `ctx.requires_wgrad=True` and no UB overlap → `grad_output_quantizer.set_usage(rowwise=True, columnwise=True)`.
- `grad_output_preprocess`:
  - Applies any TP communication.
  - Calls `tex.quantize(grad_output, grad_output_quantizer)` if FP8/NVFP4 is enabled → same quantization path as in FProp, but using **backward** NVFP4 quantizers.

### 3.2 Gradient NVFP4Quantizer Configuration (Backward)

From `NVFP4BlockScalingRecipeState` in backward mode (see 0.2):

- `with_rht=True`
- `with_post_rht_amax=True`
- `with_2d_quantization=False`
- `stochastic_rounding=True`

Thus in C++:

- `this->with_rht = true`
- `this->stochastic_rounding = true`
- `quant_config.set_stochastic_rounding(true)`
- `quant_config.set_nvfp4_2d_quantization(false)`

Inside `quantize_impl`:

- RNG state is allocated for SR:

```cpp
if (this->stochastic_rounding) {
  const size_t rng_elts_per_thread = 1024;
  auto gen = ...;
  at::PhiloxCudaState philox_args = init_philox_state(gen, rng_elts_per_thread);
  auto rng_state = torch::empty({2}, opts);
  philox_unpack(philox_args, static_cast<int64_t*>(rng_state.data_ptr()));
  te_rng_state = makeTransformerEngineTensor(rng_state);
  quant_config.set_rng_state(te_rng_state.data());
}
```

- Amax and RHT:
  - Same `with_rht` branch as FProp activations, using `nvte_hadamard_transform_amax`.
  - Rowwise amax = amax(dY); columnwise amax = amax(RHT(dYᵀ)).
- Quantization:
  - Rowwise: `nvte_quantize_v2(dY, out_identity, quant_config)` → 1D NVFP4 kernel with **SR ON**.
  - Columnwise: fused `nvte_hadamard_transform_cast_fusion_columnwise` or fallback RHT+quant, also with SR ON.

### 3.3 Dispatcher: Bwd Path for Grad

`nvte_quantize_v2` uses the same dispatcher but the **backward helper**:  
[`quantize.cuh:116-206`](../transformer_engine/common/cast/dispatch/quantize.cuh#L116-L206)

```cpp
void quantize_bwd_helper(..., const NVTEQuantizationConfig quant_config, cudaStream_t stream) {
  ...
  switch (output_tensor->scaling_mode) {
    ...
    case NVTE_NVFP4_1D_SCALING: {
      NVTE_CHECK((!IS_DBIAS && !IS_DACT),
                 "IS_DBIAS and IS_DACT are not supported by BWD NVTE_NVFP4_1D_SCALING");

      CheckNoopTensor(*noop_tensor, "cast_noop");
      CheckInputTensor(*grad_tensor, "input");
      CheckOutputTensor(*output_tensor, "output", false);

      int32_t rows = grad_tensor->flat_first_dim();
      int32_t cols = grad_tensor->flat_last_dim();
      auto dtype = grad_tensor->dtype();
      bool use_optimized_kernel = (dtype == DType::kBFloat16) &&
                                  (rows % 32 == 0) && (cols % 32 == 0) &&
                                  output_tensor->has_data();

      if (use_optimized_kernel) {
        if (quant_config_cpp.nvfp4_2d_quantization) {
          nvfp4::quantize_transpose</*use_2d_quantization=*/true>(...);
        } else {
          nvfp4::quantize_transpose</*use_2d_quantization*/ false>(...);
        }
      } else {
        quantize_transpose_vector_blockwise_fp4(...);
      }
      break;
    }
    ...
  }
}
```

For `dY ∈ BF16[1024, 2048]`:

- `rows = 1024`, `cols = 2048`, both multiples of 32 → optimized 1D NVFP4 kernel.
- `stochastic_rounding=true` and `rng_state` is set → SR active inside NVFP4 kernel.

### 3.4 DGrad GEMM Itself

After quantization, `_Linear.backward` configures and calls the DGrad GEMM (simplified):

- Inputs:
  - `grad_output` (now NVFP4 rowwise) as GEMM input.
  - `weightmat` (same NVFP4 weight workspace from FProp) as GEMM weight.
- GEMM call structure mirrors FProp but with `grad_input_quantizer` and appropriate layout arguments.

We can summarize:

- **dY**:
  - amax: `nvte_hadamard_transform_amax(dY)` → rowwise + columnwise amax.
  - RHT: used internally to compute columnwise amax and columnwise quantization (for WGrad), not for DGrad rowwise path.
  - SR: ON via Philox rng in NVFP4 kernels.
  - Quantized: 1D NVFP4, both rowwise (for DGrad GEMM) and columnwise (for WGrad GEMM).

---

## 4. WGrad GEMM (`dW = dYᵀ · X`)

The WGrad GEMM uses **columnwise NVFP4 representations** of both the activations and the grad output.

### 4.1 Preparing Activations & Grad Output for WGrad

The relevant part of `_Linear.backward` (wgrad section) shows how activations and grad output are re-prepared:

**Activation (inputmat_total)**: [`linear.py:760-808`](../transformer_engine/pytorch/module/linear.py#L760-L808)

```python
if ctx.fp8 or ctx.debug:
    if isinstance(inputmat_total, QuantizedTensorStorage):
        inputmat_total.update_usage(columnwise_usage=True)
    else:
        ctx.input_quantizer.set_usage(rowwise=False, columnwise=True)
        inputmat_total = ctx.input_quantizer(inputmat_total)
```

- This ensures we have **columnwise NVFP4 X** (possibly reusing what was computed in FProp).
- Since `input_quantizer` is the same NVFP4Quantizer used in FProp (with `with_rht=True`, `with_2d_quantization=False`, SR OFF), the columnwise path uses:
  - `nvte_hadamard_transform_amax`, then
  - `nvte_hadamard_transform_cast_fusion_columnwise(X, out_transpose, rht_matrix, quant_config)` → RHT + quantization to columnwise NVFP4 (no SR).

**Grad output (grad_output)**: [`linear.py:808-860`](../transformer_engine/pytorch/module/linear.py#L808-L860)

```python
if ctx.fp8 or ctx.debug:
    if isinstance(grad_output, QuantizedTensorStorage):
        grad_output.update_usage(columnwise_usage=True)
    else:
        ctx.grad_output_quantizer.set_usage(rowwise=False, columnwise=True)
        grad_output = ctx.grad_output_quantizer(grad_output)
```

- Here `grad_output_quantizer` is the **backward NVFP4Quantizer** (RHT ON, SR ON).
- Columnwise-only usage triggers:
  - `nvte_hadamard_transform_amax(dY)` (rowwise + columnwise amax).
  - Fused `nvte_hadamard_transform_cast_fusion_columnwise(dY, out_transpose, rht_matrix, quant_config)` with **SR ON**.
- This yields **columnwise NVFP4 dY** suitable for the WGrad GEMM.

### 4.2 WGrad GEMM Call

`wgrad_gemm` closure: [`linear.py:880-920`](../transformer_engine/pytorch/module/linear.py#L880-L920)

```python
def wgrad_gemm(
    x: torch.Tensor,
    dy: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Perform wgrad GEMM: dw = dy^T * x"""
    nvtx_range_push(f"{nvtx_label}.wgrad_gemm")
    dw, db, *_ = general_gemm(x, dy, **wgrad_gemm_kwargs)
    nvtx_range_pop(f"{nvtx_label}.wgrad_gemm")
    return dw, db
```

Where `wgrad_gemm_kwargs` includes:

```python
wgrad_gemm_kwargs = {
    "out_dtype": ...,
    "quantization_params": ctx.grad_weight_quantizer,
    "layout": "NT",
    "grad": True,
    ...
}
```

- `x` is the columnwise NVFP4 activation (`X_cw_w`).
- `dy` is the columnwise NVFP4 grad output (`dY_cw_w`).
- `grad_weight_quantizer` is another NVFP4Quantizer using `fp4_quant_bwd_grad`:
  - `with_rht=True`, `stochastic_rounding=True`, `with_2d_quantization=False`.
- `general_gemm` will:
  - Dequantize/consume columnwise NVFP4 inputs appropriately.
  - Compute `dW = dYᵀ · X`.
  - Optionally **quantize `dW`** using `grad_weight_quantizer` (NVFP4, SR ON, RHT applied to columnwise view if columnwise usage is requested for dW).

### 4.3 Where amax / RHT / SR / Quantization Happen for WGrad

Putting it together:

- **Activations (X)**:
  - amax:
    - FProp: `nvte_hadamard_transform_amax(X)` (when with_rht=True).
    - WGrad: reusing the same quantizer, columnwise path uses fused RHT+quant; amax already computed; `quantize_impl` with `compute_amax=false` might be used depending on call (when using `quantize_with_amax` disjoint path).
  - RHT:
    - Only for columnwise usage: `nvte_hadamard_transform_cast_fusion_columnwise`.
  - SR:
    - OFF (forward input quantization uses `stochastic_rounding=False`).
  - Quantization:
    - 1D NVFP4 (use_2d_quantization=false) via `quantize_transpose_nvfp4_kernel`.

- **Grad output (dY)**:
  - amax:
    - `nvte_hadamard_transform_amax(dY)` in backward quantizer `quantize_impl`.
  - RHT:
    - Columnwise-only path for WGrad: fused `nvte_hadamard_transform_cast_fusion_columnwise`.
  - SR:
    - ON (`stochastic_rounding=True` in backward quantizer).
    - NVFP4 kernels use Philox-based random bits to stochastically round to E2M1.
  - Quantization:
    - 1D NVFP4 (use_2d_quantization=false) for both rowwise (DGrad) and columnwise (WGrad).

- **Weights (W)**:
  - amax:
    - FProp weight quantizer computes amax(W) (no RHT).
  - RHT:
    - Never used on weights in this recipe.
  - SR:
    - OFF (forward weight quantizer uses deterministic RN).
  - Quantization:
    - 2D NVFP4 via `quantize_transpose_nvfp4_2D_kernel` (16×16 blocks) — both rowwise and columnwise layouts are derived from BF16 W once per FProp.

---

## 5. Summary per GEMM (for `test_mp.py`)

**FProp GEMM (Y = X · Wᵀ)**:

- Inputs:
  - X: BF16[1024, 768] → NVFP4, 1D scaling, rowwise+columnwise; RHT only for columnwise; SR OFF.
  - W: BF16[2048, 768] → NVFP4, **2D 16×16 scaling**, rowwise+columnwise; no RHT; SR OFF.
- amax:
  - X: via `nvte_hadamard_transform_amax` (RHT used internally for columnwise amax).
  - W: via `nvte_compute_amax_with_config`.
- Kernels:
  - Activations: `nvfp4::quantize_transpose<use_2d=false>` (1D NVFP4).
  - Weights: `nvfp4::quantize_transpose<use_2d=true>` (2D NVFP4).
  - Optional fused RHT+quant for X columnwise: `nvte_hadamard_transform_cast_fusion_columnwise`.

**DGrad GEMM (dX = dY · W)**:

- Inputs:
  - dY: BF16[1024, 2048] → NVFP4, 1D scaling, rowwise+columnwise; RHT for columnwise; **SR ON**.
  - W: same NVFP4 2D-quantized weights as FProp.
- amax:
  - dY: via `nvte_hadamard_transform_amax` (RHT-based for columnwise).
- Kernels:
  - Grad_output quantization: `nvfp4::quantize_transpose<use_2d=false>` with SR ON.
  - DGrad GEMM uses rowwise NVFP4 dY and rowwise NVFP4 W.

**WGrad GEMM (dW = dYᵀ · X)**:

- Inputs:
  - X: columnwise NVFP4, produced via activation quantizer (RHT+quant, SR OFF).
  - dY: columnwise NVFP4, produced via gradient quantizer (RHT+quant, SR ON).
- amax:
  - X & dY: computed once per role via `nvte_hadamard_transform_amax` in their quantizers.
- Kernels:
  - For columnwise paths, both X and dY use the **fused RHT+quant kernel**:
    - `nvte_hadamard_transform_cast_fusion_columnwise(input, out_transpose, rht_matrix, quant_config)`.
  - GEMM itself: `general_gemm` with `quantization_params=grad_weight_quantizer` controlling optional NVFP4 quantization of `dW`.

Overall, `test_mp.py` exercises:

- **1D NVFP4** scaling for activations and gradients.
- **2D NVFP4** scaling for weights (16×16 blocks).
- **RHT**:
  - For activations and gradients, but **only for columnwise usage** (i.e., for WGrad-related paths).
- **Stochastic rounding**:
  - OFF for FProp activations and weights.
  - ON for gradients (dY), as prescribed by the NVFP4 training recipe.

