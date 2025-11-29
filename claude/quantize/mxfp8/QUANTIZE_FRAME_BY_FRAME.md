# MXFP8 Quantize Method: Frame-by-Frame Walkthrough

This document provides a detailed frame-by-frame trace of the `quantize` method in [`../../../transformer_engine/pytorch/csrc/extensions/cast.cpp:33-79`](../../../transformer_engine/pytorch/csrc/extensions/cast.cpp#L33-L79), specifically for MXFP8 quantization.

## Table of Contents

1. [Frame 1: Entry Point](#frame-1-entry-point---quantize-function)
2. [Frame 2: Convert Python Quantizer → C++ Quantizer](#frame-2-convert-python-quantizer--c-quantizer)
3. [Frame 3: Convert Input Tensor](#frame-3-convert-input-tensor-to-contiguous-layout)
4. [Frame 4: Skip Float8CurrentScaling Path](#frame-4-skip-float8currentscaling-path-mxfp8-specific)
5. [Frame 5: Initialize Output Tensor](#frame-5-initialize-output-tensor)
6. [Frame 6: Handle noop_flag](#frame-6-handle-noop_flag-optional)
7. [Frame 7: Perform Quantization](#frame-7-perform-quantization)
8. [Frame 8: Return Python Object](#frame-8-return-python-object)

---

## Frame 1: Entry Point - `quantize` Function

**Location**: [`../../../transformer_engine/pytorch/csrc/extensions/cast.cpp:33-79`](../../../transformer_engine/pytorch/csrc/extensions/cast.cpp#L33-L79)

**Function Signature**:
```cpp
py::object quantize(const at::Tensor &tensor,
                    py::handle quantizer,
                    const py::object &output,
                    std::optional<at::Tensor> noop_flag)
```

### Input State (for MXFP8)

| Parameter | Type | Example Value | Purpose |
|-----------|------|---------------|---------|
| `tensor` | `at::Tensor` | shape `[1024, 2048]`, dtype `float32` | Input tensor to quantize |
| `quantizer` | `py::handle` | Python `MXFP8Quantizer` object | Quantization configuration |
| `output` | `py::object` | `None` (typical) | Optional pre-allocated output |
| `noop_flag` | `std::optional<at::Tensor>` | `std::nullopt` | Optional conditional quantization flag |

### Design Rationale

This function is the main Python-exposed API for quantization. It's **generic** and works with multiple quantizer types through polymorphism:
- FP8 Delayed Scaling
- FP8 Current Scaling
- MXFP8 (this trace)
- NVFP4
- Float8 Block-wise

The generic design allows new quantization formats to be added without modifying the entry point.

---

## Frame 2: Convert Python Quantizer → C++ Quantizer

**Location**: [`../../../transformer_engine/pytorch/csrc/extensions/cast.cpp:36`](../../../transformer_engine/pytorch/csrc/extensions/cast.cpp#L36)

```cpp
auto quantizer_cpp = convert_quantizer(quantizer);
```

### Frame 2a: `convert_quantizer` Function

**Location**: [`../../../transformer_engine/pytorch/csrc/common.cpp:51-64`](../../../transformer_engine/pytorch/csrc/common.cpp#L51-L64)

```cpp
std::unique_ptr<Quantizer> convert_quantizer(py::handle quantizer) {
  init_extension();
  if (quantizer.is_none()) {
    return std::make_unique<NoneQuantizer>(quantizer);
  }
  for (auto [_check_type, check_quantizer_type, _create_tensor, create_quantizer] :
       detail::custom_types_converters) {
    if (check_quantizer_type(quantizer.ptr())) {
      return create_quantizer(quantizer);
    }
  }

  NVTE_ERROR("Unexpected type for quantizer");
}
```

### Dispatch Mechanism

The function iterates through `detail::custom_types_converters` array, defined in [`../../../transformer_engine/pytorch/csrc/pybind.h:102-112`](../../../transformer_engine/pytorch/csrc/pybind.h#L102-L112):

```cpp
constexpr std::array custom_types_converters = {
    std::make_tuple(IsFloat8Tensor, IsFloat8Quantizers, NVTETensorFromFloat8Tensor,
                    CreateQuantizer<Float8Quantizer>),
    std::make_tuple(IsFloat8Tensor, IsFloat8CurrentScalingQuantizers, NVTETensorFromFloat8Tensor,
                    CreateQuantizer<Float8CurrentScalingQuantizer>),
    std::make_tuple(IsMXFP8Tensor, IsMXFP8Quantizers, NVTETensorFromMXFP8Tensor,
                    CreateQuantizer<MXFP8Quantizer>),  // ← MXFP8 entry
    std::make_tuple(IsFloat8BlockwiseQTensor, IsFloat8BlockwiseQuantizers,
                    NVTETensorFromFloat8BlockwiseQTensor, CreateQuantizer<Float8BlockQuantizer>),
    std::make_tuple(IsNVFP4Tensor, IsNVFP4Quantizers, NVTETensorFromNVFP4Tensor,
                    CreateQuantizer<NVFP4Quantizer>)
};
```

### For MXFP8, the dispatch flow is:

1. **Check**: `IsMXFP8Quantizers(quantizer.ptr())` returns `true`
   - Implementation ([`pybind.h:61`](../../../transformer_engine/pytorch/csrc/pybind.h#L61)):
     ```cpp
     inline bool IsMXFP8Quantizers(PyObject *obj) {
       return Py_TYPE(obj) == MXFP8QuantizerClass;
     }
     ```
   - Compares Python object type against `MXFP8QuantizerClass` pointer

2. **Create**: Calls `create_quantizer(quantizer)` → `CreateQuantizer<MXFP8Quantizer>(quantizer)`

3. **Construct**: Invokes MXFP8Quantizer constructor

### Frame 2b: MXFP8Quantizer Constructor

**Location**: [`../../../transformer_engine/pytorch/csrc/quantizer.cpp:899-901`](../../../transformer_engine/pytorch/csrc/quantizer.cpp#L899-L901)

```cpp
MXFP8Quantizer::MXFP8Quantizer(const py::handle& quantizer) : Quantizer(quantizer) {
  this->dtype = quantizer.attr("dtype").cast<DType>();
}
```

### Parent Constructor

**Location**: [`../../../transformer_engine/pytorch/csrc/quantizer.cpp:50-61`](../../../transformer_engine/pytorch/csrc/quantizer.cpp#L50-L61)

```cpp
Quantizer::Quantizer(const py::handle& quantizer) {
  if (quantizer.is_none()) {
    this->rowwise_usage = true;
    this->columnwise_usage = true;
    this->internal = false;
  } else {
    this->rowwise_usage = quantizer.attr("rowwise_usage").cast<bool>();
    this->columnwise_usage = quantizer.attr("columnwise_usage").cast<bool>();
    this->internal = quantizer.attr("internal").cast<bool>();
    this->quantizer = quantizer;
  }
}
```

### Extracted Attributes

| Attribute | Python Source | Typical Value | Purpose |
|-----------|---------------|---------------|---------|
| `dtype` | `quantizer.dtype` | `DType::kFloat8E4M3` | Target FP8 format |
| `rowwise_usage` | `quantizer.rowwise_usage` | `true` | Enable rowwise layout |
| `columnwise_usage` | `quantizer.columnwise_usage` | `true` | Enable columnwise layout |
| `internal` | `quantizer.internal` | `false` | Internal tensor format flag |

### State After Frame 2

- **Output**: `quantizer_cpp` is a `std::unique_ptr<MXFP8Quantizer>` with:
  - FP8 dtype configured (e.g., E4M3)
  - Usage flags set (rowwise/columnwise)
  - Reference to Python quantizer object

### Design Rationale

**Why the converter pattern?**
- **Type Safety**: Ensures Python objects are correctly typed before C++ operations
- **Decoupling**: C++ code doesn't depend on Python types directly
- **Extensibility**: New quantizer types just add an entry to `custom_types_converters`
- **Performance**: Avoids repeated Python type checks in hot paths

**Why initialize from Python attributes?**
- Python holds user-facing configuration (dtype, usage flags)
- C++ extracts only what it needs for computation
- Keeps quantizer state in one place (Python object)

---

## Frame 3: Convert Input Tensor to Contiguous Layout

**Location**: [`../../../transformer_engine/pytorch/csrc/extensions/cast.cpp:39-40`](../../../transformer_engine/pytorch/csrc/extensions/cast.cpp#L39-L40)

```cpp
auto input_contiguous = tensor.contiguous();
auto input_cpp = makeTransformerEngineTensor(input_contiguous);
```

### Step 1: Ensure Contiguous Memory

```cpp
auto input_contiguous = tensor.contiguous();
```

**What happens**:
- Checks if tensor has contiguous memory layout
- If not, creates a copy with contiguous layout
- If yes, returns reference to original tensor (no copy)

**Why necessary**: CUDA kernels expect sequential memory for efficient coalesced access.

### Step 2: Wrap in TensorWrapper

### Frame 3a: `makeTransformerEngineTensor` (Simple Overload)

**Location**: [`../../../transformer_engine/pytorch/csrc/common.cpp:120-127`](../../../transformer_engine/pytorch/csrc/common.cpp#L120-L127)

```cpp
transformer_engine::TensorWrapper makeTransformerEngineTensor(at::Tensor tensor) {
  transformer_engine::DType dtype = GetTransformerEngineDType(tensor.scalar_type());
  std::vector<size_t> shape;
  for (auto s : tensor.sizes()) {
    shape.push_back(s);
  }
  return makeTransformerEngineTensor(tensor.data_ptr(), shape, dtype);
}
```

**Extracts**:
- **Data pointer**: `tensor.data_ptr()` → GPU memory address
- **Shape**: `[1024, 2048]` (example)
- **DType**: Converts PyTorch dtype → TE dtype

| PyTorch dtype | TE DType |
|---------------|----------|
| `at::kFloat` | `DType::kFloat32` |
| `at::kHalf` | `DType::kFloat16` |
| `at::kBFloat16` | `DType::kBFloat16` |

### Frame 3b: `makeTransformerEngineTensor` (Data Pointer Overload)

**Location**: [`../../../transformer_engine/pytorch/csrc/common.cpp:115-118`](../../../transformer_engine/pytorch/csrc/common.cpp#L115-L118)

```cpp
transformer_engine::TensorWrapper makeTransformerEngineTensor(
    void* data_ptr, const std::vector<size_t>& shape,
    const transformer_engine::DType type) {
  return transformer_engine::TensorWrapper(data_ptr, shape, type);
}
```

**Creates**: `TensorWrapper` object (TE's C++ tensor abstraction)

### State After Frame 3

| Property | Value |
|----------|-------|
| `input_cpp` type | `TensorWrapper` |
| Data pointer | Points to GPU memory (e.g., `0x7f8a40000000`) |
| Shape | `[1024, 2048]` |
| DType | `DType::kFloat32` |
| Scaling mode | Default (none yet) |

### Design Rationale

**Why TensorWrapper?**
- Unified abstraction for tensors across TE
- Stores metadata: shape, dtype, scaling mode
- Holds pointers to: data, scales, amax, scale_inv
- Enables passing complex tensor layouts (FP8, MXFP8, NVFP4) through C++ APIs

**Why contiguous() first?**
- CUDA kernels are optimized for sequential access
- Strided/transposed tensors cause uncoalesced memory access
- Small copy cost << performance loss from slow memory patterns

---

## Frame 4: Skip Float8CurrentScaling Path (MXFP8 specific)

**Location**: [`../../../transformer_engine/pytorch/csrc/extensions/cast.cpp:43-51`](../../../transformer_engine/pytorch/csrc/extensions/cast.cpp#L43-L51)

```cpp
bool use_existing_amax = false;
if (detail::IsFloat8CurrentScalingQuantizers(quantizer.ptr())) {
  use_existing_amax = quantizer.attr("use_existing_amax").cast<bool>();
  if (use_existing_amax) {
    const at::Tensor &amax = quantizer.attr("amax").cast<at::Tensor>();
    input_cpp.set_amax(amax.data_ptr(), GetTransformerEngineDType(amax.scalar_type()),
                       getTensorShape(amax));
  }
}
```

### For MXFP8

**Condition**: `detail::IsFloat8CurrentScalingQuantizers(quantizer.ptr())` returns **`false`**
- Implementation: `Py_TYPE(obj) == Float8CurrentScalingQuantizerClass`
- MXFP8 has type `MXFP8QuantizerClass`, not `Float8CurrentScalingQuantizerClass`

**Result**: Entire block is skipped, `use_existing_amax` remains `false`

### Design Rationale

**What is "current scaling"?**
- FP8 Current Scaling: Pre-computed amax from previous operations
- Used to skip amax computation during quantization
- Optimizes fused kernels (e.g., GEMM + quantize)

**Why doesn't MXFP8 use this?**
- MXFP8 uses **per-block scaling** computed on-the-fly
- Each 32-element block gets its own scale during quantization
- No global amax to reuse

---

## Frame 5: Initialize Output Tensor

**Location**: [`../../../transformer_engine/pytorch/csrc/extensions/cast.cpp:54-62`](../../../transformer_engine/pytorch/csrc/extensions/cast.cpp#L54-L62)

```cpp
TensorWrapper output_cpp;
py::object output_py;
if (output.is_none()) {
  const auto shape = get_tensor_shape(input_cpp);
  const auto fake_dtype = input_cpp.dtype();
  std::tie(output_cpp, output_py) = quantizer_cpp->create_tensor(shape, fake_dtype);
} else {
  std::tie(output_cpp, output_py) = quantizer_cpp->convert_and_update_tensor(output);
}
```

### Typical Case: `output.is_none() == true`

This is the standard path when no pre-allocated output is provided.

### Step 1: Extract Input Shape

**Location**: [`../../../transformer_engine/pytorch/csrc/extensions/cast.cpp:26-29`](../../../transformer_engine/pytorch/csrc/extensions/cast.cpp#L26-L29)

```cpp
std::vector<size_t> get_tensor_shape(const TensorWrapper &tensor) {
  const auto &shape = tensor.shape();
  return std::vector<size_t>(shape.data, shape.data + shape.ndim);
}
```

**Result**: `shape = [1024, 2048]` (example)

### Step 2: Call `MXFP8Quantizer::create_tensor`

**Location**: [`../../../transformer_engine/pytorch/csrc/quantizer.cpp:905-984`](../../../transformer_engine/pytorch/csrc/quantizer.cpp#L905-L984)

This is a **critical function**—let me break it down step by step.

---

### Frame 5a: Validate Dimensions

**Location**: [`quantizer.cpp:909-920`](../../../transformer_engine/pytorch/csrc/quantizer.cpp#L909-L920)

```cpp
const std::vector<int64_t> shape_int64(shape.begin(), shape.end());
size_t flat_first_dim = 1;
if (shape.size() > 0) {
  for (size_t i = 0; i < shape.size() - 1; ++i) {
    flat_first_dim *= shape[i];
  }
}
const size_t flat_last_dim = shape.size() > 0 ? shape.back() : 1;
NVTE_CHECK(flat_first_dim % MXFP8_BLOCK_SIZE == 0 && flat_last_dim % MXFP8_BLOCK_SIZE == 0,
           "MXFP8 requires tensor dims that are divisible by ", MXFP8_BLOCK_SIZE,
           " (got shape=", shape, ")");
```

**Constants**: `MXFP8_BLOCK_SIZE = 32` ([`quantizer.cpp:48`](../../../transformer_engine/pytorch/csrc/quantizer.cpp#L48))

**Example Validation** for shape `[1024, 2048]`:
- `flat_first_dim = 1024` → `1024 % 32 == 0` ✓
- `flat_last_dim = 2048` → `2048 % 32 == 0` ✓

**If validation fails**: Throws error with shape information

### Design Rationale

**Why 32-element blocks?**
1. **GPU warp size**: 32 threads = 1 warp (NVIDIA GPUs)
2. **Vectorization**: Enables efficient SIMD operations
3. **Storage**: 1 scale per 32 elements = only 3% overhead
4. **Precision**: Fine enough granularity to preserve accuracy

**Why require divisibility?**
- Simplifies CUDA kernel logic (no partial blocks)
- Enables fixed-size threadblock configurations
- Most NN layers have dimensions divisible by 32

---

### Frame 5b: Compute Scale Shapes

**Location**: [`quantizer.cpp:921-922`](../../../transformer_engine/pytorch/csrc/quantizer.cpp#L921-L922)

```cpp
const auto rowwise_scale_inv_shape = get_scale_shape(shape, false);
const auto columnwise_scale_inv_shape = get_scale_shape(shape, true);
```

### `MXFP8Quantizer::get_scale_shape` Implementation

**Location**: [`../../../transformer_engine/pytorch/csrc/quantizer.cpp:1105-1134`](../../../transformer_engine/pytorch/csrc/quantizer.cpp#L1105-L1134)

```cpp
std::vector<size_t> MXFP8Quantizer::get_scale_shape(const std::vector<size_t>& shape,
                                                    bool columnwise) const {
  size_t numel = 1;
  for (auto s : shape) {
    numel *= s;
  }
  auto last_dim = shape.back();

  NVTE_CHECK(last_dim % MXFP8_BLOCK_SIZE == 0 && (numel / last_dim) % MXFP8_BLOCK_SIZE == 0,
             "MXFP8 requires tensor dims that are divisible by ", MXFP8_BLOCK_SIZE,
             " (got shape=", shape, ")");

  std::vector<size_t> scale_shape;
  bool rowwise_usage = !columnwise;

  if (rowwise_usage) {
    // rowwise scaling factor shape
    size_t sinv0 = roundup(numel / last_dim, 128);
    size_t sinv1 = roundup(last_dim / MXFP8_BLOCK_SIZE, 4);
    scale_shape = {sinv0, sinv1};
  } else {
    // columnwise scaling factor shape
    size_t sinv0 = roundup(numel / (last_dim * MXFP8_BLOCK_SIZE), 4);
    size_t sinv1 = roundup(last_dim, 128);
    scale_shape = {sinv0, sinv1};
  }
  return scale_shape;
}
```

### Example Calculation for shape `[1024, 2048]`

#### Rowwise Scale Shape

```
numel = 1024 × 2048 = 2,097,152
last_dim = 2048

sinv0 = roundup(2,097,152 / 2048, 128)
      = roundup(1024, 128)
      = 1024

sinv1 = roundup(2048 / 32, 4)
      = roundup(64, 4)
      = 64

Result: [1024, 64]
```

**Interpretation**:
- 1024 rows
- Each row has 2048 elements = 2048/32 = 64 blocks
- One E8M0 scale per block

#### Columnwise Scale Shape

```
sinv0 = roundup(2,097,152 / (2048 × 32), 4)
      = roundup(32, 4)
      = 32

sinv1 = roundup(2048, 128)
      = 2048

Result: [32, 2048]
```

**Interpretation**:
- 2048 columns
- 1024 rows = 1024/32 = 32 blocks in column direction
- One E8M0 scale per block

### Design Rationale

**Why roundup to 128 and 4?**
- **128-byte alignment**: GPU cache line size
  - Ensures coalesced memory access
  - Reduces memory transactions
- **4-element alignment**: Vectorized loads (128-bit = 4×32-bit)
  - Enables SIMD instructions
  - Improves kernel performance

**Why both rowwise and columnwise?**
- **Rowwise**: For matrix A in A×B (standard GEMM)
- **Columnwise**: For matrix B^T in A×B^T (attention patterns)
- Avoids runtime transpose of scale buffers
- Enables TN, NT, NN GEMM layouts without overhead

---

### Frame 5c: Allocate Tensors

**Location**: [`quantizer.cpp:924-939`](../../../transformer_engine/pytorch/csrc/quantizer.cpp#L924-L939)

```cpp
at::Tensor rowwise_data_tensor, rowwise_scale_inv_tensor;
at::Tensor columnwise_data_tensor, columnwise_scale_inv_tensor;
const auto uint8_tensor_opts = at::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);

if (rowwise_usage) {
  const std::vector<int64_t> scale_inv_shape_int64(rowwise_scale_inv_shape.begin(),
                                                   rowwise_scale_inv_shape.end());
  rowwise_data_tensor = at::empty(shape_int64, uint8_tensor_opts);
  rowwise_scale_inv_tensor = at::empty(scale_inv_shape_int64, uint8_tensor_opts);
}
if (columnwise_usage) {
  const std::vector<int64_t> scale_inv_shape_int64(columnwise_scale_inv_shape.begin(),
                                                   columnwise_scale_inv_shape.end());
  columnwise_data_tensor = at::empty(shape_int64, uint8_tensor_opts);
  columnwise_scale_inv_tensor = at::empty(scale_inv_shape_int64, uint8_tensor_opts);
}
```

### Allocated Buffers (assuming both usages = true)

| Buffer | Shape | DType | Size (bytes) | Purpose |
|--------|-------|-------|--------------|---------|
| `rowwise_data` | `[1024, 2048]` | `uint8` | 2,097,152 | FP8 quantized values |
| `rowwise_scale_inv` | `[1024, 64]` | `uint8` | 65,536 | E8M0 exponents (rowwise) |
| `columnwise_data` | `[1024, 2048]` | `uint8` | 2,097,152 | FP8 quantized values (col layout) |
| `columnwise_scale_inv` | `[32, 2048]` | `uint8` | 65,536 | E8M0 exponents (columnwise) |

**Total**: ~4.3 MB for this example (vs 8.4 MB for FP32 input)

### Design Rationale

**Why uint8 for data?**
- FP8 (E4M3 or E5M2) is 8-bit format
- Stored as raw bytes (reinterpreted by CUDA kernels)
- PyTorch lacks native FP8 scalar type (uses uint8 storage)

**Why uint8 for scales?**
- E8M0 (8-bit exponent only, no mantissa) format
- 1 byte per scale = minimal overhead
- Represents powers of 2: `2^(exponent - 127)`

**Why at::empty() not at::zeros()?**
- Allocates GPU memory without initialization
- Faster (no memset kernel)
- Values will be filled by quantization kernel

**Why separate data buffers?**
- Rowwise and columnwise may need different memory layouts
- GEMM kernels read different patterns
- Separate buffers = better cache locality

---

### Frame 5d: Convert to Python Objects

**Location**: [`quantizer.cpp:941-949`](../../../transformer_engine/pytorch/csrc/quantizer.cpp#L941-L949)

```cpp
auto py_cast = [](at::Tensor& tensor, bool need_cast) -> py::object {
  return need_cast ? py::cast(tensor) : py::none();
};
auto rowwise_data_py = py_cast(rowwise_data_tensor, rowwise_usage);
auto rowwise_scale_inv_py = py_cast(rowwise_scale_inv_tensor, rowwise_usage);
auto columnwise_data_py = py_cast(columnwise_data_tensor, columnwise_usage);
auto columnwise_scale_inv_py = py_cast(columnwise_scale_inv_tensor, columnwise_usage);
```

**Purpose**: Wrap C++ `at::Tensor` objects in Python `torch.Tensor` wrappers

---

### Frame 5e: Create Python MXFP8Tensor Object

**Location**: [`quantizer.cpp:950-967`](../../../transformer_engine/pytorch/csrc/quantizer.cpp#L950-L967)

```cpp
py::object out_py;
if (internal) {
  py::handle MXFP8TensorClass(reinterpret_cast<PyObject*>(MXFP8TensorStoragePythonClass));
  out_py = MXFP8TensorClass("rowwise_data"_a = rowwise_data_py,
                            "columnwise_data"_a = columnwise_data_py,
                            "rowwise_scale_inv"_a = rowwise_scale_inv_py,
                            "columnwise_scale_inv"_a = columnwise_scale_inv_py,
                            "fp8_dtype"_a = this->dtype, "quantizer"_a = this->quantizer);
} else {
  py::handle MXFP8TensorClass(reinterpret_cast<PyObject*>(MXFP8TensorPythonClass));
  out_py = MXFP8TensorClass("shape"_a = shape_int64, "dtype"_a = GetATenDType(dtype),
                            "rowwise_data"_a = rowwise_data_py,
                            "columnwise_data"_a = columnwise_data_py,
                            "rowwise_scale_inv"_a = rowwise_scale_inv_py,
                            "columnwise_scale_inv"_a = columnwise_scale_inv_py,
                            "fp8_dtype"_a = this->dtype, "quantizer"_a = this->quantizer);
}
```

### Python Class References

These are initialized in [`../../../transformer_engine/pytorch/csrc/extensions/pybind.cpp:55-68`](../../../transformer_engine/pytorch/csrc/extensions/pybind.cpp#L55-L68):

```cpp
void init_mxfp8_extension() {
  if (MXFP8TensorPythonClass) return;
  auto fp8_module = py::module_::import("transformer_engine.pytorch.tensor.mxfp8_tensor");
  MXFP8QuantizerClass =
      reinterpret_cast<PyTypeObject *>(PyObject_GetAttrString(fp8_module.ptr(), "MXFP8Quantizer"));
  MXFP8TensorPythonClass =
      reinterpret_cast<PyTypeObject *>(PyObject_GetAttrString(fp8_module.ptr(), "MXFP8Tensor"));
  auto fp8_base_module =
      py::module_::import("transformer_engine.pytorch.tensor.storage.mxfp8_tensor_storage");
  MXFP8TensorStoragePythonClass = reinterpret_cast<PyTypeObject *>(
      PyObject_GetAttrString(fp8_base_module.ptr(), "MXFP8TensorStorage"));
  NVTE_CHECK(MXFP8TensorPythonClass != nullptr,
             "Internal error: could not initialize pyTorch MXFP8 extension.");
}
```

### Python Class Hierarchy

- **MXFP8Tensor**: User-facing class with high-level API
  - Source: [`../../../transformer_engine/pytorch/tensor/mxfp8_tensor.py`](../../../transformer_engine/pytorch/tensor/mxfp8_tensor.py)
- **MXFP8TensorStorage**: Internal storage class with raw buffers
  - Source: `../../../transformer_engine/pytorch/tensor/storage/mxfp8_tensor_storage.py`

### Design Rationale

**Why two classes?**
- **Separation of concerns**: Storage vs. behavior
- **Internal flag**: Allows C++ to create storage-only objects for intermediate values
- **Flexibility**: User-facing class can add methods without affecting C++ code

---

### Frame 5f: Create C++ TensorWrapper

**Location**: [`quantizer.cpp:969-981`](../../../transformer_engine/pytorch/csrc/quantizer.cpp#L969-L981)

```cpp
TensorWrapper out_cpp(NVTE_MXFP8_1D_SCALING);
if (rowwise_usage) {
  out_cpp.set_rowwise_data(rowwise_data_tensor.data_ptr(), this->dtype, shape);
  out_cpp.set_rowwise_scale_inv(rowwise_scale_inv_tensor.data_ptr(), DType::kFloat8E8M0,
                                rowwise_scale_inv_shape);
}
if (columnwise_usage) {
  out_cpp.set_columnwise_data(columnwise_data_tensor.data_ptr(), this->dtype, shape);
  out_cpp.set_columnwise_scale_inv(columnwise_scale_inv_tensor.data_ptr(), DType::kFloat8E8M0,
                                   columnwise_scale_inv_shape);
}
this->set_quantization_params(&out_cpp);
```

### TensorWrapper Configuration

**Constructor**: `TensorWrapper(NVTE_MXFP8_1D_SCALING)`
- Sets scaling mode for the wrapper
- Tells downstream kernels this is MXFP8 format

**Rowwise configuration**:
- **Data**: Pointer to `[1024, 2048]` buffer, dtype = FP8 (E4M3/E5M2)
- **Scale inv**: Pointer to `[1024, 64]` buffer, dtype = E8M0

**Columnwise configuration**:
- **Data**: Pointer to `[1024, 2048]` buffer, dtype = FP8
- **Scale inv**: Pointer to `[32, 2048]` buffer, dtype = E8M0

**Final call**: `set_quantization_params(&out_cpp)`
- Location: [`quantizer.cpp:903`](../../../transformer_engine/pytorch/csrc/quantizer.cpp#L903)
- Implementation: `void MXFP8Quantizer::set_quantization_params(TensorWrapper* tensor) const {}`
- **No-op for MXFP8** (all config already set)

### Design Rationale

**Why E8M0 dtype?**
- Custom format: 8-bit exponent, 0-bit mantissa
- Represents power-of-2 scales: `scale = 2^(E8M0_value - 127)`
- Efficient GPU computation (shifts instead of multiplies)

**Why "scale_inv" not "scale"?**
- Stores **inverse** scale (reciprocal)
- Quantization: `quantized = input * scale_inv`
- Dequantization: `output = quantized / scale_inv`
- Avoids division in forward pass (faster)

---

### State After Frame 5

| Object | Type | Contents |
|--------|------|----------|
| `output_cpp` | `TensorWrapper` | Configured with pointers to all buffers |
| `output_py` | `py::object` | Python MXFP8Tensor wrapping the same buffers |

**Important**: Memory is **allocated but not filled** yet. Values are uninitialized.

---

## Frame 6: Handle noop_flag (Optional)

**Location**: [`../../../transformer_engine/pytorch/csrc/extensions/cast.cpp:65-68`](../../../transformer_engine/pytorch/csrc/extensions/cast.cpp#L65-L68)

```cpp
std::optional<TensorWrapper> noop_flag_cpp;
if (noop_flag.has_value()) {
  noop_flag_cpp = makeTransformerEngineTensor(*noop_flag);
}
```

### For Typical Case: `noop_flag` not provided

- `noop_flag.has_value()` returns `false`
- Block is skipped
- `noop_flag_cpp` remains `std::nullopt`

### When noop_flag IS Provided

**Purpose**: Conditional quantization
- `noop_flag` is a boolean tensor (same shape as input, or broadcastable)
- If `noop_flag[i] == true`, skip quantizing input[i]
- Used in advanced training recipes (e.g., only quantize large gradients)

**Example Use Case**:
```python
# Only quantize gradients above threshold
noop_flag = (grad.abs() < 1e-6)  # Skip tiny gradients
quantized = quantize(grad, quantizer, noop_flag=noop_flag)
```

---

## Frame 7: Perform Quantization

**Location**: [`../../../transformer_engine/pytorch/csrc/extensions/cast.cpp:71-76`](../../../transformer_engine/pytorch/csrc/extensions/cast.cpp#L71-L76)

```cpp
if (use_existing_amax) {
  auto *quantizer_cs = dynamic_cast<Float8CurrentScalingQuantizer *>(quantizer_cpp.get());
  quantizer_cs->quantize_with_amax(input_cpp, output_cpp, noop_flag_cpp);
} else {
  quantizer_cpp->quantize(input_cpp, output_cpp, noop_flag_cpp);
}
```

### For MXFP8

- `use_existing_amax == false` (set in Frame 4)
- Takes **else branch**
- Calls virtual method: `quantizer_cpp->quantize(...)`
- Dispatches to `MXFP8Quantizer::quantize`

---

### Frame 7a: `MXFP8Quantizer::quantize`

**Location**: [`../../../transformer_engine/pytorch/csrc/quantizer.cpp:1091-1103`](../../../transformer_engine/pytorch/csrc/quantizer.cpp#L1091-L1103)

```cpp
void MXFP8Quantizer::quantize(const TensorWrapper& input, TensorWrapper& out,
                              const std::optional<TensorWrapper>& noop_flag) {
  if (input.numel() == 0) {
    return;
  }
  QuantizationConfigWrapper quant_config;
  if (noop_flag) {
    quant_config.set_noop_tensor(noop_flag->data());
  }
  NVTE_SCOPED_GIL_RELEASE({
    nvte_quantize_v2(input.data(), out.data(), quant_config, at::cuda::getCurrentCUDAStream());
  });
}
```

### Step-by-Step Execution

#### 1. Check Empty Input
```cpp
if (input.numel() == 0) {
  return;
}
```
- Early return for zero-element tensors
- Avoids launching empty CUDA kernels

#### 2. Create Config Wrapper
```cpp
QuantizationConfigWrapper quant_config;
```
- Wraps quantization parameters for C API
- Default constructor: no special config for MXFP8

#### 3. Set noop Flag (if provided)
```cpp
if (noop_flag) {
  quant_config.set_noop_tensor(noop_flag->data());
}
```
- Passes conditional quantization flag to kernel

#### 4. Release GIL and Call C API
```cpp
NVTE_SCOPED_GIL_RELEASE({
  nvte_quantize_v2(input.data(), out.data(), quant_config, at::cuda::getCurrentCUDAStream());
});
```

**Critical Macro**: `NVTE_SCOPED_GIL_RELEASE`
- Releases Python Global Interpreter Lock (GIL)
- Allows Python threads to run while GPU works
- Automatically re-acquires GIL when scope exits

**C API Call**: `nvte_quantize_v2(...)`
- Location: [`../../../transformer_engine/common/cast/cast.cu`](../../../transformer_engine/common/cast/cast.cu)
- Pure C interface (no C++ or Python dependencies)
- Dispatches to CUDA kernels

### Design Rationale

**Why release GIL?**
- GPU work is asynchronous (non-blocking)
- Python thread shouldn't wait for GPU
- Enables Python/CPU work to overlap with GPU kernels
- Critical for training throughput

**Why `nvte_quantize_v2` not direct kernel call?**
- C API provides stable ABI (binary compatibility)
- Allows linking from JAX, TensorFlow, other frameworks
- Isolates PyTorch-specific code from CUDA kernels

---

### Frame 7b: `nvte_quantize_v2` (C API and CUDA Kernels)

**Location**: [`../../../transformer_engine/common/cast/cast.cu`](../../../transformer_engine/common/cast/cast.cu)

### High-Level Pseudo-Logic

```cpp
void nvte_quantize_v2(NVTETensor input, NVTETensor output,
                      NVTEQuantizationConfig config, cudaStream_t stream) {
  // Extract tensor metadata
  auto input_dtype = nvte_tensor_dtype(input);
  auto output_scaling_mode = nvte_tensor_scaling_mode(output);

  // Dispatch based on output format
  if (output_scaling_mode == NVTE_MXFP8_1D_SCALING) {
    // MXFP8 path
    auto output_rowwise_data = nvte_tensor_data(output);
    auto output_rowwise_scales = nvte_tensor_rowwise_data(output);
    auto output_columnwise_data = nvte_tensor_columnwise_data(output);
    auto output_columnwise_scales = nvte_tensor_columnwise_scale_inv(output);

    if (output_rowwise_data && output_rowwise_scales) {
      launch_mxfp8_quantize_rowwise_kernel(input, output, stream);
    }
    if (output_columnwise_data && output_columnwise_scales) {
      launch_mxfp8_quantize_columnwise_kernel(input, output, stream);
    }
  } else {
    // Other quantization formats (FP8 delayed/current scaling, NVFP4, etc.)
    // ...
  }
}
```

### MXFP8 CUDA Kernel Behavior (Conceptual)

#### Per-Block Quantization Algorithm

For each **32-element block** in input tensor:

```cuda
__global__ void mxfp8_quantize_kernel(
    const float* input,     // [M, N] input data
    uint8_t* output_data,   // [M, N] FP8 output
    uint8_t* output_scales, // [M, N/32] E8M0 scales
    int M, int N
) {
  // Thread indexing
  int block_id = blockIdx.x * blockDim.x + threadIdx.x;
  int blocks_per_row = N / 32;
  int row = block_id / blocks_per_row;
  int block_in_row = block_id % blocks_per_row;

  // Each thread handles one 32-element block
  const float* block_input = &input[row * N + block_in_row * 32];
  uint8_t* block_output = &output_data[row * N + block_in_row * 32];

  // Step 1: Find max absolute value in block
  float max_abs = 0.0f;
  for (int i = 0; i < 32; i++) {
    max_abs = fmaxf(max_abs, fabsf(block_input[i]));
  }

  // Step 2: Compute shared exponent (E8M0)
  // E8M0 = floor(log2(max_abs)) + bias
  int exponent = (max_abs > 0) ? (int)floor(log2f(max_abs)) + 127 : 0;
  float scale = powf(2.0f, exponent - 127);
  float scale_inv = 1.0f / scale;

  // Store E8M0 scale
  output_scales[row * blocks_per_row + block_in_row] = (uint8_t)exponent;

  // Step 3: Quantize each element to FP8
  for (int i = 0; i < 32; i++) {
    float normalized = block_input[i] * scale_inv;
    block_output[i] = float_to_fp8_e4m3(normalized);  // Hardware intrinsic
  }
}
```

### Example Execution

**Input block** (32 float32 values):
```
[0.5, -1.2, 0.8, 0.3, ..., -0.9]
```

**Step 1**: Find `max_abs = 1.2`

**Step 2**: Compute scale
```
exponent = floor(log2(1.2)) + 127 = 0 + 127 = 127
scale = 2^(127-127) = 2^0 = 1.0
scale_inv = 1.0
E8M0 value: 127 (stored as uint8)
```

**Step 3**: Quantize elements
```
normalized[0] = 0.5 * 1.0 = 0.5
fp8[0] = float_to_fp8_e4m3(0.5) = 0x38  (FP8 representation)

normalized[1] = -1.2 * 1.0 = -1.2
fp8[1] = float_to_fp8_e4m3(-1.2) = 0xB9

... (repeat for all 32 elements)
```

**Output**:
- **Data**: `[0x38, 0xB9, 0x40, ...]` (32 uint8 FP8 values)
- **Scale**: `127` (1 uint8 E8M0 value)

### Kernel Launch Configuration

```cpp
// Typical configuration for [1024, 2048] tensor
int blocks_total = (1024 * 2048) / 32;  // 65,536 blocks
int threads_per_block = 256;
int num_blocks = (blocks_total + threads_per_block - 1) / threads_per_block;

launch_mxfp8_quantize_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
    input_data, output_data, output_scales, 1024, 2048
);
```

### State After Frame 7

**GPU memory updated**:
- `rowwise_data`: Contains FP8 quantized values
- `rowwise_scale_inv`: Contains E8M0 scales (one per 32 elements)
- `columnwise_data`: Contains FP8 quantized values (column-major layout)
- `columnwise_scale_inv`: Contains E8M0 scales (column-major blocks)

**CUDA stream**:
- Kernel launched asynchronously
- May still be executing when function returns
- Synchronized by PyTorch's CUDA stream management

### Design Rationale

**Why per-block scaling?**
- **Precision**: Better dynamic range than per-tensor scaling
- **Efficiency**: Coarse enough for fast computation
- **Hardware**: Aligns with GPU warp size (32 threads)

**Why E8M0 (exponent-only)?**
- **Simplicity**: Only power-of-2 scales
- **Speed**: Hardware bit shifts instead of FP multiply
- **Storage**: 1 byte per scale = minimal overhead
- **Accuracy**: Sufficient for most NN workloads

**Why dual layout (rowwise + columnwise)?**
- **GEMM flexibility**: Supports TN, NT, NN layouts
- **Zero overhead**: No runtime transpose of scales
- **Memory locality**: Scales match data access patterns

**Why asynchronous kernel launch?**
- **Overlapping**: CPU continues while GPU quantizes
- **Throughput**: Multiple operations in flight
- **Latency hiding**: Masks slow operations

---

## Frame 8: Return Python Object

**Location**: [`../../../transformer_engine/pytorch/csrc/extensions/cast.cpp:78`](../../../transformer_engine/pytorch/csrc/extensions/cast.cpp#L78)

```cpp
return output_py;
```

### What's Returned

**Type**: `py::object` (Python MXFP8Tensor)

**Attributes** (for user):
- `.shape`: `[1024, 2048]` (logical shape)
- `.dtype`: `torch.float32` (logical dtype)
- `._rowwise_data`: `torch.Tensor([1024, 2048], dtype=uint8)` (FP8 values)
- `._rowwise_scale_inv`: `torch.Tensor([1024, 64], dtype=uint8)` (E8M0 scales)
- `._columnwise_data`: `torch.Tensor([1024, 2048], dtype=uint8)` (FP8 values)
- `._columnwise_scale_inv`: `torch.Tensor([32, 2048], dtype=uint8)` (E8M0 scales)
- `.quantizer`: Reference to MXFP8Quantizer

### User Workflow

```python
import torch
from transformer_engine.pytorch import MXFP8Quantizer, quantize

# Create input
input_tensor = torch.randn(1024, 2048, device='cuda', dtype=torch.float32)

# Create quantizer
quantizer = MXFP8Quantizer(dtype=torch.float8_e4m3fn)

# Quantize (Frame 1-8 execution)
output = quantize(input_tensor, quantizer)

# Use quantized tensor
print(output.shape)  # torch.Size([1024, 2048])
print(type(output))  # <class 'MXFP8Tensor'>

# Dequantize (not covered in this trace)
dequantized = output.dequantize()  # Back to float32
```

---

## Complete Call Graph Summary

```
quantize(tensor, quantizer=MXFP8Quantizer, output=None, noop_flag=None)
│
├─▶ convert_quantizer(quantizer)                          [Frame 2]
│   ├─▶ IsMXFP8Quantizers(quantizer.ptr())               [Type check]
│   └─▶ CreateQuantizer<MXFP8Quantizer>(quantizer)
│       └─▶ MXFP8Quantizer::MXFP8Quantizer(quantizer)    [Constructor]
│           └─▶ Quantizer::Quantizer(quantizer)          [Base constructor]
│
├─▶ tensor.contiguous()                                    [Frame 3]
├─▶ makeTransformerEngineTensor(input_contiguous)
│   └─▶ TensorWrapper(data_ptr, shape, dtype)
│
├─▶ quantizer_cpp->create_tensor(shape, dtype)            [Frame 5]
│   └─▶ MXFP8Quantizer::create_tensor(shape, dtype)
│       ├─▶ Validate dimensions (divisible by 32)
│       ├─▶ get_scale_shape(shape, rowwise=false)        [Compute rowwise scale shape]
│       ├─▶ get_scale_shape(shape, columnwise=true)      [Compute columnwise scale shape]
│       ├─▶ at::empty([M, N], uint8)                      [Allocate rowwise data]
│       ├─▶ at::empty([M, N/32], uint8)                   [Allocate rowwise scales]
│       ├─▶ at::empty([M, N], uint8)                      [Allocate columnwise data]
│       ├─▶ at::empty([M/32, N], uint8)                   [Allocate columnwise scales]
│       ├─▶ MXFP8TensorClass(...)                         [Construct Python object]
│       └─▶ TensorWrapper(NVTE_MXFP8_1D_SCALING)         [Construct C++ wrapper]
│           ├─▶ set_rowwise_data(ptr, dtype, shape)
│           ├─▶ set_rowwise_scale_inv(ptr, E8M0, shape)
│           ├─▶ set_columnwise_data(ptr, dtype, shape)
│           └─▶ set_columnwise_scale_inv(ptr, E8M0, shape)
│
└─▶ quantizer_cpp->quantize(input_cpp, output_cpp, noop_flag)  [Frame 7]
    └─▶ MXFP8Quantizer::quantize(input, out, noop_flag)
        └─▶ NVTE_SCOPED_GIL_RELEASE
            └─▶ nvte_quantize_v2(input.data(), out.data(), config, stream)  [C API]
                ├─▶ launch_mxfp8_quantize_rowwise_kernel<<<>>>()            [CUDA]
                │   └─▶ For each 32-element block:
                │       ├─▶ Compute max_abs
                │       ├─▶ Compute E8M0 scale = log2(max_abs)
                │       ├─▶ Store E8M0 in scale buffer
                │       └─▶ Quantize each element: fp8 = float_to_fp8(value / scale)
                │
                └─▶ launch_mxfp8_quantize_columnwise_kernel<<<>>>()         [CUDA]
                    └─▶ (Same as above, columnwise layout)
```

---

## State Transitions Summary

| Frame | Input State | Output State |
|-------|-------------|--------------|
| **1** (Entry) | Python tensor + quantizer | - |
| **2** (Convert quantizer) | Python `MXFP8Quantizer` | C++ `std::unique_ptr<MXFP8Quantizer>` |
| **3** (Wrap input) | PyTorch `at::Tensor` | C++ `TensorWrapper` (input) |
| **4** (Skip FP8 CS path) | - | `use_existing_amax = false` |
| **5** (Allocate output) | Input shape `[M, N]` | Python `MXFP8Tensor` + C++ `TensorWrapper` (uninitialized) |
| **6** (noop flag) | Optional noop tensor | `std::optional<TensorWrapper>` |
| **7** (Quantize) | Float32 input, empty output | FP8 data + E8M0 scales (GPU) |
| **8** (Return) | - | Python `MXFP8Tensor` (quantized) |

---

## Key Takeaways

### MXFP8 Architecture

1. **Block size**: 32 elements (GPU warp size)
2. **Shared exponent**: E8M0 (8-bit exponent only)
3. **Dual layout**: Rowwise + columnwise for efficient GEMM
4. **Scale overhead**: ~3% (1 byte per 32 elements)

### Design Principles

1. **Polymorphism**: Generic `quantize()` supports multiple formats
2. **Type safety**: Dispatch table ensures correct quantizer/tensor pairing
3. **Lazy initialization**: Allocate first, quantize later
4. **GIL management**: Release during GPU work for concurrency
5. **Separation of concerns**: Python (API) ↔ C++ (glue) ↔ CUDA (compute)

### Performance Optimizations

1. **Contiguous layout**: Ensures coalesced memory access
2. **Buffer alignment**: Roundup to 128/4 for cache efficiency
3. **Asynchronous execution**: GPU work overlaps with CPU
4. **Dual layout**: Avoids runtime transpose overhead
5. **E8M0 scales**: Hardware shifts instead of FP multiplies

### Code Organization

- **Python API**: User-facing, high-level
- **C++ glue**: Type conversion, dispatch, memory management
- **C API**: Stable ABI for multi-framework support
- **CUDA kernels**: Pure computation, no framework dependencies

---

## Related Documentation

- [CALL_GRAPH.md](CALL_GRAPH.md) - Visual call graph
- [DESIGN_RATIONALE.md](DESIGN_RATIONALE.md) - Architecture decisions
- [../../mx/mxfp8/](../../mx/mxfp8/) - MXFP8 usage and tests
