# NVFP4 Quantize+Transpose Kernel Walkthrough (1D/2D, SR ON/OFF, RETURN_TRANSPOSE=TRUE)

Source file (as attached):

```cpp
// transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh
```

We assume **input shape = 128 × 1024** (rows × cols), and we analyze:

- **1D scaling** (use_2d_quantization = false)
  - `USE_STOCHASTIC_ROUNDING = false`, `RETURN_TRANSPOSE = true`
  - `USE_STOCHASTIC_ROUNDING = true`, `RETURN_TRANSPOSE = true`
- **2D scaling** (use_2d_quantization = true)
  - `RETURN_TRANSPOSE = true` (SR on/off both covered where relevant)

For each variant:

- Start at the **host launcher** `quantize_transpose<use_2d_quantization>`.
- Then walk the device kernels:
  - `quantize_transpose_nvfp4_kernel` (1D)
  - `quantize_transpose_nvfp4_2D_kernel` (2D)
- Emphasize thread/block mapping, shared-memory layout, TMA pipeline, FP4 PTX helpers, and SR behavior.

---

## 0. Common Setup: Constants and Shapes

From the header:

```cpp
constexpr size_t SCALE_DIM   = 16;   // NVFP4 block (16 elements)
constexpr size_t CHUNK_DIM_Y = 128;  // rows per threadblock
constexpr size_t CHUNK_DIM_X = 128;  // cols per threadblock
constexpr size_t THREADS_NUM = 128;  // threads per block

constexpr size_t TILE_DIM_Y = 32;    // rows per pipeline stage
constexpr size_t TILE_DIM_X = 128;   // cols per stage

constexpr size_t BUFFS_NUM   = 2;    // double-buffered shared tiles
constexpr size_t BUFF_DIM_Y  = TILE_DIM_Y; // 32
constexpr size_t BUFF_DIM_X  = TILE_DIM_X; // 128

constexpr size_t BUFF_IN_DIM_Y = BUFF_DIM_Y;
constexpr size_t BUFF_IN_DIM_X = BUFF_DIM_X;
constexpr size_t BUFF_IN_SIZE  = BUFF_IN_DIM_Y * BUFF_IN_DIM_X; // 32*128

constexpr size_t BUFF_OUT_DIM_Y = BUFF_DIM_Y;
constexpr size_t BUFF_OUT_DIM_X = BUFF_DIM_X / 2; // FP4x2 packs
constexpr size_t BUFF_OUT_SIZE  = BUFF_OUT_DIM_Y * BUFF_OUT_DIM_X;
```

Tiling and stages:

```cpp
constexpr size_t SCALES_PER_TILE_Y = TILE_DIM_Y / SCALE_DIM; // 32/16 = 2
constexpr size_t SCALES_PER_TILE_X = TILE_DIM_X / SCALE_DIM; // 128/16 = 8

constexpr size_t TILES_Y = CHUNK_DIM_Y / TILE_DIM_Y; // 128/32 = 4
constexpr size_t TILES_X = CHUNK_DIM_X / TILE_DIM_X; // 128/128 = 1
constexpr size_t STAGES  = TILES_Y * TILES_X;        // 4
```

So per threadblock we process a **128×128 chunk** as **4 stages** of size **32×128**.

For our input **128×1024**:

```cpp
blocks_Y = DIVUP(rows, CHUNK_DIM_Y) = DIVUP(128,128) = 1;
blocks_X = DIVUP(cols, CHUNK_DIM_X) = DIVUP(1024,128) = 8;

// CUDA grid and block
grid      = dim3(blocks_X, blocks_Y) = (8, 1, 1);
blockDim  = dim3(THREADS_NUM)        = (128, 1, 1);
```

Each block handles rows `[0..127]` and a 128-wide column slice of the input.

---

## 1. Host Launcher: `quantize_transpose<use_2d_quantization>`

### 1.1 High-level logic

Signature:

```cpp
template <bool use_2d_quantization>
void quantize_transpose(const Tensor &input,
                        const Tensor *noop,
                        Tensor *output,
                        const QuantizationConfig *quant_config,
                        cudaStream_t stream);
```

Key bits:

```cpp
using namespace quantize_transpose_kernel;
using namespace ptx;

bool use_stochastic_rounding = quant_config
                             ? quant_config->stochastic_rounding
                             : false;

// Return transposed data only if transposed output buffers are allocated
bool return_transpose = output->has_columnwise_data();

constexpr bool COMPUTE_ACTIVATIONS = false;
using ParamOP = Empty;
constexpr float (*OP)(float, const ParamOP &) = nullptr;
```

- **`use_stochastic_rounding`** → selects `USE_STOCHASTIC_ROUNDING` template param.
- **`return_transpose`** → selects `RETURN_TRANSPOSE` template param.
- `COMPUTE_ACTIVATIONS = false` for this path: we’re **only quantizing**, not applying extra activation functions.

Shape and sanity checks:

```cpp
const size_t rows = input.flat_first_dim();
const size_t cols = input.flat_last_dim();

NVTE_CHECK(rows % 32 == 0);
NVTE_CHECK(cols % 32 == 0);

const size_t blocks_Y = DIVUP(rows, CHUNK_DIM_Y);
const size_t blocks_X = DIVUP(cols, CHUNK_DIM_X);
const dim3 grid(blocks_X, blocks_Y);
const size_t block_size = THREADS_NUM;  // 128
```

Scale tensor geometry:

```cpp
const size_t scale_stride = output->scale_inv.shape[1];
const size_t scale_stride_transpose =
    return_transpose ? output->columnwise_scale_inv.shape[1] : 0;

nvfp4_scale_t *scales_ptr         = reinterpret_cast<nvfp4_scale_t *>(output->scale_inv.dptr);
nvfp4_scale_t *scales_transpose   = reinterpret_cast<nvfp4_scale_t *>(output->columnwise_scale_inv.dptr);
```

- `scales_ptr` → rowwise scales (for non-transposed layout).
- `scales_transpose` → columnwise scales (for the transposed layout).

RNG state for stochastic rounding:

```cpp
const NVTETensor rng_state_tensor =
    (quant_config != nullptr) ? quant_config->rng_state : nullptr;

const size_t *rng_state = nullptr;
if (rng_state_tensor != nullptr) {
  Tensor &rng_state_te_tensor = *convertNVTETensor(rng_state_tensor);
  NVTE_CHECK(rng_state_te_tensor.dtype() == DType::kInt64);
  NVTE_CHECK(rng_state_te_tensor.data.shape == std::vector<size_t>{2});
  rng_state = reinterpret_cast<const size_t *>(rng_state_te_tensor.data.dptr);
}
```

- `rng_state[0]` = seed, `rng_state[1]` = global offset.

For NVFP4, `IType` is BF16 in this version:

```cpp
using IType = bf16;
```

Then the host builds **TMA tensor maps** for input, output, and transposed output:

```cpp
alignas(64) CUtensorMap tensor_map_input{};
alignas(64) CUtensorMap tensor_map_output{};
alignas(64) CUtensorMap tensor_map_output_transpose{};

create_2D_tensor_map(tensor_map_input,
                     input.data,
                     rows, cols,
                     BUFF_DIM_Y, BUFF_DIM_X,
                     cols, 0,
                     sizeof(IType) * 8);

create_2D_tensor_map(tensor_map_output,
                     output->data,
                     rows, cols,
                     BUFF_DIM_Y, BUFF_DIM_X,
                     cols, 0,
                     /*bits_per_element=*/4);

if (return_transpose) {
  create_2D_tensor_map(tensor_map_output_transpose,
                       output->columnwise_data,
                       cols, rows,      // note swapped dims
                       BUFF_DIM_X, BUFF_DIM_Y,
                       rows, 0,
                       /*bits_per_element=*/4);
}
```

**Interpretation:**

- TMA will move **BF16 tiles** from `tensor_map_input` to shared.
- TMA will move **FP4-packed tiles** from shared to `tensor_map_output`.
- For transpose, TMA writes to `tensor_map_output_transpose` with swapped dims.

Dynamic shared memory layout size:

```cpp
constexpr size_t buff_elems        = BUFF_DIM_Y * BUFF_DIM_X; // 32*128
constexpr size_t buff_elems_total  = BUFFS_NUM * buff_elems;  // 2 tiles

constexpr size_t buff_size_aligned_in  = ...; // BF16 input tiles
constexpr size_t buff_size_aligned_out = ...; // FP4 output tiles
constexpr size_t buff_size_scales      = (CHUNK_DIM_Y * CHUNK_DIM_X)/16 * sizeof(nvfp4_scale_t);

constexpr size_t in_mem                = buff_size_aligned_in;
constexpr size_t out_data_mem          = buff_size_aligned_out;
constexpr size_t out_data_transpose_mem= buff_size_aligned_out;
constexpr size_t out_scales_transpose_mem = buff_size_scales;

size_t dshmem_size = in_mem + out_data_mem + out_data_transpose_mem
                   + out_scales_transpose_mem + TMA_SHMEM_ALIGNMENT;
```

Then comes the key **switch**:

```cpp
TRANSFORMER_ENGINE_SWITCH_CONDITION(
    use_stochastic_rounding, USE_STOCHASTIC_ROUNDING,
    TRANSFORMER_ENGINE_SWITCH_CONDITION(return_transpose, RETURN_TRANSPOSE, {

      auto kernel = quantize_transpose_nvfp4_kernel<
          COMPUTE_ACTIVATIONS, ParamOP, OP, IType,
          USE_STOCHASTIC_ROUNDING, RETURN_TRANSPOSE>;

      if constexpr (use_2d_quantization) {
        kernel = quantize_transpose_nvfp4_2D_kernel<
          COMPUTE_ACTIVATIONS, ParamOP, OP, IType,
          USE_STOCHASTIC_ROUNDING, RETURN_TRANSPOSE>;
      }

      cudaFuncSetAttribute(kernel,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           dshmem_size);

      kernel<<<grid, block_size, dshmem_size, stream>>>(
          tensor_map_input,
          tensor_map_output,
          tensor_map_output_transpose,
          scales_ptr,
          scales_transpose_ptr,
          noop_ptr,
          amax_rowwise_ptr,
          amax_colwise_ptr,
          rows, cols,
          scale_stride, scale_stride_transpose,
          rng_state);
    }););
```

So:

- `use_2d_quantization = false` → **1D kernel** `quantize_transpose_nvfp4_kernel`.
- `use_2d_quantization = true`  → **2D kernel** `quantize_transpose_nvfp4_2D_kernel`.
- `use_stochastic_rounding` and `return_transpose` become compile-time template flags.

Everything after this is device-side and specialized per configuration.

---

## 2. 1D Scaling, RETURN_TRANSPOSE = true

Template:

```cpp
template <bool COMPUTE_ACTIVATIONS, typename ParamOP, float (*OP)(float,const ParamOP &),
          typename IType, bool USE_STOCHASTIC_ROUNDING, bool RETURN_TRANSPOSE>
__global__ void __launch_bounds__(THREADS_NUM)
quantize_transpose_nvfp4_kernel(...);
```

We focus on:

- `COMPUTE_ACTIVATIONS = false`
- `IType = bf16`
- `RETURN_TRANSPOSE = true`
- `USE_STOCHASTIC_ROUNDING = false` and `true`.

### 2.1 Block & thread indexing

First, common compile-time helpers:

```cpp
constexpr bool NO_ACTIVATIONS_NOT_FP32_INPUT =
    (!COMPUTE_ACTIVATIONS) && (!std::is_same_v<IType, float>);

using IType2 = typename ptx::FPx2<IType>;  // bf16x2
```

The kernel can early-exit if a `noop` tensor says to skip quantization:

```cpp
if constexpr (!COMPUTE_ACTIVATIONS) {
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }
}
```

RNG setup is shared between SR on/off cases:

```cpp
size_t rng_sequence = threadIdx.x
                    + blockIdx.x * THREADS_NUM
                    + blockIdx.y * gridDim.x * THREADS_NUM;

size_t rng_seed   = rng_state ? rng_state[0] : 0;
size_t rng_offset = rng_state ? rng_state[1] : 0;

philox4x32_native_state<10> rng;
rng.init(rng_seed, rng_sequence, rng_offset);

uint4 random_uint4 = USE_STOCHASTIC_ROUNDING ? rng.generate4()
                                             : uint4{0, 0, 0, 0};
int rnd_idx = 0;   // which 32-bit lane in random_uint4 we’re using
```

The helper for rounding bits:

```cpp
__device__ inline uint32_t get_rbits(auto &rng, uint4 &random_uint4, int &rnd_idx) {
  if constexpr (!USE_STOCHASTIC_ROUNDING) return 0u;

  if (rnd_idx == 4) {
    random_uint4 = rng.generate4();
    rnd_idx = 0;
  }
  uint32_t r = (&random_uint4.x)[rnd_idx++];
  return r;
}
```

- **SR OFF**: `USE_STOCHASTIC_ROUNDING = false` → `rbits = 0` everywhere.
- **SR ON**: `USE_STOCHASTIC_ROUNDING = true`  → each group of 4 values gets its own random `rbits`.

Block offsets in rowwise layout:

```cpp
const size_t block_offset_Y = blockIdx.y * CHUNK_DIM_Y; // 128 * 0 = 0
const size_t block_offset_X = blockIdx.x * CHUNK_DIM_X; // 128 * bx

const size_t chunk_rows = rows - block_offset_Y;         // = 128 here
```

Block offsets in transposed layout:

```cpp
const size_t block_offset_Y_t = blockIdx.x * CHUNK_DIM_X; // 128 * bx
const size_t block_offset_X_t = blockIdx.y * CHUNK_DIM_Y; // 0
```

So each block handles:

- **Rowwise view**: rows `[0..127]`, cols `[bx*128 .. bx*128+127]`.
- **Transposed view**: rows `[bx*128 .. bx*128+127]`, cols `[0..127]`.

### 2.2 Thread decomposition (1D scaling)

Rowwise mapping:

```cpp
constexpr size_t THREADS_PER_BANK      = TOTAL_BANKS_WIDTH / SCALE_DIM; // e.g. 8
constexpr size_t THREADS_X_ROWWISE     = SCALES_PER_TILE_X;             // 8
constexpr size_t THREADS_Y_ROWWISE     = THREADS_NUM / THREADS_X_ROWWISE; // 128 / 8 = 16

const size_t tid_Y_rowwise      = threadIdx.x / THREADS_X_ROWWISE;   // 0..15
const size_t tid_X_rowwise      = threadIdx.x % THREADS_X_ROWWISE;   // 0..7

const size_t thread_offset_Y_rowwise = tid_Y_rowwise;                 // row in tile
const size_t thread_offset_X_rowwise = tid_X_rowwise * SCALE_DIM;     // 0,16,...,112
```

Interpretation:

- Logical 2D grid of threads: **16 (rows) × 8 (scale-block columns)**.
- Each thread owns a **16-element contiguous block** along the X dimension in a row.

Colwise mapping:

```cpp
const size_t tid_X_colwise      = threadIdx.x;      // 0..127
const size_t thread_offset_X_colwise = tid_X_colwise; // column index in 128-wide tile
```

So for the **transposed path**, each thread is in charge of one column in the 32×128 tile.

Scale indexing helpers:

```cpp
constexpr size_t SCALES_PER_CHUNK_X = CHUNK_DIM_X / SCALE_DIM; // 128/16 = 8

const size_t scales_block_offset_Y_rowwise = blockIdx.y * CHUNK_DIM_Y;       // 0
const size_t scales_block_offset_X_rowwise = blockIdx.x * SCALES_PER_CHUNK_X;// bx*8

const size_t scales_offset_Y_rowwise = scales_block_offset_Y_rowwise + tid_Y_rowwise;
const size_t scales_offset_X_rowwise = scales_block_offset_X_rowwise;

const size_t SFs_per_row = cols / SCALE_DIM; // 1024 / 16 = 64

const bool rowwise_scale_is_within_bounds_X = (scales_offset_X_rowwise < SFs_per_row);

// Transposed scales
const size_t scales_block_offset_Y_t = blockIdx.x * CHUNK_DIM_X;      // bx*128
const size_t scales_block_offset_X_t = blockIdx.y * SCALES_PER_CHUNK_Y;

const size_t tid_Y_t       = tid_Y_rowwise;
const size_t scales_offset_Y_t = scales_block_offset_Y_t + tid_Y_t;

const bool colwise_scale_is_within_bounds_Y = (scales_offset_Y_t < cols);
```

### 2.3 Shared memory layout & global scaling factors

Shared memory is manually aligned for TMA:

```cpp
extern __shared__ char dynamic_shmem[];
uintptr_t base_shmem_ptr = reinterpret_cast<uintptr_t>(dynamic_shmem);
uintptr_t dshmem = (base_shmem_ptr + TMA_SHMEM_ALIGNMENT - 1)
                 & ~static_cast<uintptr_t>(TMA_SHMEM_ALIGNMENT - 1);

IType      *in_sh         = reinterpret_cast<IType *>(dshmem);
fp4e2m1x2  *out_data_sh   = reinterpret_cast<fp4e2m1x2 *>(dshmem + in_mem);
fp4e2m1x2  *out_t_data_sh = reinterpret_cast<fp4e2m1x2 *>(dshmem + in_mem + out_mem_rowwise_data);

nvfp4_scale_t *out_rowwise_scales_sh   =
    reinterpret_cast<nvfp4_scale_t *>(dshmem + in_mem + out_mem_rowwise_data + out_mem_colwise_data);
nvfp4_scale_t *out_colwise_scales_sh   =
    reinterpret_cast<nvfp4_scale_t *>(dshmem + in_mem + out_mem_rowwise_data
                                      + out_mem_colwise_data + out_mem_rowwise_scales);
```

So for each block we have:

- **Input BF16 tiles**: `in_sh[BUFFS_NUM][32×128]`.
- **Rowwise FP4 tiles**: `out_data_sh[BUFFS_NUM][32×64]` (FP4x2 packs).
- **Transposed FP4 tiles**: `out_t_data_sh[BUFFS_NUM][32×64]`.
- **Staged scales** for colwise path in shared.

Global encode/decode scaling factors (tensor-level):

```cpp
float S_enc_rowwise = (amax_rowwise_ptr == nullptr)
                        ? 1.0f
                        : compute_global_encode_scaling_factor_FP4(*amax_rowwise_ptr);
float S_dec_rowwise = 1.0f / S_enc_rowwise;

float S_enc_colwise = (amax_colwise_ptr == nullptr)
                        ? S_enc_rowwise
                        : compute_global_encode_scaling_factor_FP4(*amax_colwise_ptr);
float S_dec_colwise = 1.0f / S_enc_colwise;
```

Per-block scales will be derived from these.

### 2.4 Pipeline setup: barriers + first TMA load

```cpp
float thread_amax = 0.0f;

__shared__ alignas(8) uint64_t mbar[STAGES];
initialize_barriers<STAGES, THREADS_NUM>(mbar, is_master_thread);

copy_2d_to_shared(&in_sh[0],
                  &tensor_map_input,
                  block_offset_X,
                  block_offset_Y,
                  shmem_buff_size,
                  &mbar[0],
                  is_master_thread);
```

- `initialize_barriers` sets up per-stage **memory barriers** for TMA.
- `copy_2d_to_shared` kicks off a **TMA bulk tensor copy** for the first 32×128 tile.
- Only `is_master_thread` (probably `threadIdx.x == 0`) issues the TMA command.

From here we enter the main **stage loop**:

```cpp
for (size_t stage = 0; stage < STAGES; ++stage) {
  const size_t buff              = stage % BUFFS_NUM;  // ping-pong
  const size_t next_stage        = stage + 1;
  const size_t stage_offset_Y    = stage * BUFF_DIM_Y; // 0,32,64,96

  const size_t buff_offset_in    = buff * BUFF_IN_SIZE;
  const size_t buff_offset_out   = buff * BUFF_OUT_SIZE;
  const size_t buff_offset_out_t = buff * BUFF_OUT_T_SIZE;

  // prefetch next tile if any
  ...
}
```

### 2.5 Prefetch next tile (global → shared via TMA)

Inside the loop:

```cpp
if (next_stage < STAGES) {
  // Wait until shared buffer is free to be reused
  ptx::cp_async_bulk_wait_group_read<1>();

  const size_t next_buff           = next_stage % BUFFS_NUM;
  const size_t next_stage_offset_Y = next_stage * BUFF_DIM_Y;
  const size_t global_offset_Y     = block_offset_Y + next_stage_offset_Y;
  const size_t global_offset_X     = block_offset_X;
  const size_t next_buff_offset    = next_buff * BUFF_IN_SIZE;

  copy_2d_to_shared(&in_sh[next_buff_offset],
                    &tensor_map_input,
                    global_offset_X,
                    global_offset_Y,
                    shmem_buff_size,
                    &mbar[next_stage],
                    is_master_thread);
}

ptx::fence_proxy_async_shared_cta();
ptx::mbarrier_wait_parity(&mbar[stage], 0);
```

**PTX primitives:**

- `cp_async_bulk_wait_group_read<1>()` → wait for any prior bulk copies that *read from this shared buffer* to finish (SHMEM is safe to write again).
- `mbarrier_wait_parity` + `fence_proxy_async_shared_cta` → ensure TMA **global→shared** copy for this stage has completed and data is visible to threads.

At this point, tile `stage` is ready in `in_sh[buff]` for compute.

### 2.6 Colwise path (transpose) – 1D scaling

Executed only when `RETURN_TRANSPOSE = true`:

```cpp
if constexpr (RETURN_TRANSPOSE) {
#pragma unroll
  for (size_t it = 0; it < ITERATIONS_TRANSPOSE; ++it) {
    const size_t in_thread_offset_Y  = 0 + it * SCALE_DIM;        // 0 or 16
    const size_t in_thread_offset_X  = thread_offset_X_colwise;   // 0..127

    const size_t out_t_thread_offset_Y = thread_offset_X_colwise; // col → row
    const size_t out_t_thread_offset_X = 0 + it * BUFF_OUT_IT_OFFSET;

    const size_t shmem_offset_base_colwise_in =
        buff_offset_in + in_thread_offset_Y * BUFF_IN_DIM_X + in_thread_offset_X;
    const size_t shmem_offset_base_colwise_out_t =
        buff_offset_out_t + out_t_thread_offset_Y * BUFF_OUT_T_DIM_X + out_t_thread_offset_X;

    float  in_compute_colwise[SCALE_DIM];
    IType  in_colwise_IType[SCALE_DIM];
    float  block_amax = 0.0f;
```

Here:

- `ITERATIONS_TRANSPOSE = BUFF_IN_DIM_Y / SCALE_DIM = 32 / 16 = 2`.
- Each iteration, each thread processes **16 elements down its column**.

Load data and compute per-block amax for 1D **colwise** scaling:

```cpp
if constexpr (NO_ACTIVATIONS_NOT_FP32_INPUT) {
  IType block_amax_f16 = static_cast<IType>(0.0f);
#pragma unroll
  for (int i = 0; i < SCALE_DIM; ++i) {
    int shmem_offset_colwise = shmem_offset_base_colwise_in + i * BUFF_IN_DIM_X;
    in_colwise_IType[i] = in_sh[shmem_offset_colwise];           // BF16
    block_amax_f16 = __hmax(block_amax_f16, __habs(in_colwise_IType[i]));
  }
  block_amax = static_cast<float>(block_amax_f16);
} else {
  // FP32 path (not used here)
}
```

Compute and store scale for this 16-element block:

```cpp
nvfp4_scale_t S_dec_b_fp8 =
    compute_decoding_scaling_factor(block_amax, S_enc_colwise);

size_t scale_idx_sh =
    tid_Y_t * SCALES_PER_CHUNK_Y + stage * ITERATIONS_TRANSPOSE + it;
out_colwise_scales_sh[scale_idx_sh] = S_dec_b_fp8;

constexpr float float_max = detail::TypeExtrema<float>::max;
float block_scale_inverse = fminf(
    1.0f / (static_cast<float>(S_dec_b_fp8) * S_dec_colwise), float_max);
float2 block_scale_inverse_2x{block_scale_inverse, block_scale_inverse};
```

Now quantize **16 inputs → FP4**:

```cpp
fp4e2m1x4 regs[SCALE_DIM / 4];  // 16/4 = 4 vector packs
#pragma unroll
for (int e = 0; e < SCALE_DIM / 4; ++e) {
  uint32_t rbits = get_rbits(rng, random_uint4, rnd_idx);  // 0 or random

  uint64_t elts = *reinterpret_cast<uint64_t*>(&in_colwise_IType[4 * e]);
  regs[e] = ptx::mul_cvt_bf16_to_fp4_4x<USE_STOCHASTIC_ROUNDING>(
               elts, block_scale_inverse_2x, rbits);
}
```

**PTX helper behavior: `mul_cvt_bf16_to_fp4_4x`**

- **Unpacks** 4 BF16 values into FP32.
- **Multiplies** by `block_scale_inverse`.
- Uses `rbits` to optionally perturb rounding (if SR enabled).
- Issues one or more `cvt.rn.satfinite.e2m1x2.f32`-style ops to produce FP4.
- **Packs** 4 FP4s into an `fp4e2m1x4` container.

Finally, write to transposed buffer with a bank-friendly pattern:

```cpp
const int group = thread_lane / 16;       // 0 or 1
uint32_t val[2];
uint32_t *regs_4x = reinterpret_cast<uint32_t *>(regs);

val[0] = regs_4x[0];
val[1] = regs_4x[1];

uint32_t *out_t_data_sh_as_uint32_t =
    reinterpret_cast<uint32_t *>(&out_t_data_sh[shmem_offset_base_colwise_out_t]);

out_t_data_sh_as_uint32_t[group]          = val[0];
out_t_data_sh_as_uint32_t[(group + 1) & 1] = val[1];  // swap index to reduce bank conflicts
```

So for the transpose:

- Each thread writes two 32-bit words corresponding to 8 FP4 values.
- Grouping by `thread_lane / 16` helps distribute accesses across shared memory banks.

### 2.7 Rowwise path – 1D scaling

Rowwise block scaling uses two iterations per stage:

```cpp
const size_t stage_rowwise_scales_offset_Y = stage * BUFF_DIM_Y;
constexpr size_t ITERATIONS_NORMAL = BUFF_DIM_Y / THREADS_Y_ROWWISE; // 32 / 16 = 2

#pragma unroll
for (size_t it = 0; it < ITERATIONS_NORMAL; ++it) {
  const size_t it_thread_offset_Y_rowwise =
      thread_offset_Y_rowwise + it * THREADS_Y_ROWWISE;  // row within tile

  const size_t shmem_offset_base_rowwise_in =
      buff_offset_in + it_thread_offset_Y_rowwise * BUFF_IN_DIM_X;
  const size_t shmem_offset_base_rowwise_out =
      buff_offset_out + it_thread_offset_Y_rowwise * BUFF_OUT_DIM_X;

  float in_compute_rowwise[SCALE_DIM];
  Vec<IType,  PACK_SIZE>    in_cached[WAVES];
  Vec<IType2, PACK_SIZE/2>  in_IType[WAVES];
  float block_amax = 0.0f;
```

`PACK_SIZE` and `WAVES` are sized so that each thread processes its 16-wide block in multiple vectorized waves, e.g.: `PACK_SIZE=8`, `WAVES=2` (2×8=16).

1D rowwise amax (BF16 path):

```cpp
if constexpr (NO_ACTIVATIONS_NOT_FP32_INPUT) {
  IType2 thread_amax_2x = {IType(0.0f), IType(0.0f)};

#pragma unroll
  for (int w = 0; w < WAVES; ++w) {
    const size_t swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM;
    const size_t swizzled_thread_idx = thread_offset_X_rowwise + swizzled_group_idx;
    const size_t shmem_offset_rowwise = shmem_offset_base_rowwise_in + swizzled_thread_idx;

    in_IType[w].load_from(&in_sh[shmem_offset_rowwise]); // load 8 BF16 values

#pragma unroll
    for (int e = 0; e < PACK_SIZE / 2; ++e) {
      const IType2 pair = in_IType[w].data.elt[e];  // 2 lanes
      thread_amax_2x.x = __hmax(thread_amax_2x.x, __habs(pair.x));
      thread_amax_2x.y = __hmax(thread_amax_2x.y, __habs(pair.y));
    }
  }

  block_amax = static_cast<float>(
      __hmax(__habs(thread_amax_2x.x), __habs(thread_amax_2x.y)));
} else {
  // FP32 / activation path
}
```

- `swizzled_group_idx` + `bank_group` spreads thread accesses across shared-memory banks.
- Each thread computes **amax over 16 elements** of its rowwise block.

Then per-block scale:

```cpp
nvfp4_scale_t S_dec_b_fp8 =
    compute_decoding_scaling_factor(block_amax, S_enc_rowwise);

size_t scales_offset_Y =
    scales_offset_Y_rowwise + stage * BUFF_DIM_Y + it * THREADS_Y_ROWWISE;
size_t scales_offset_X = scales_offset_X_rowwise;
size_t scale_idx_global = scales_offset_Y * scale_stride + scales_offset_X;

bool rowwise_scale_is_within_bounds_Y =
    (stage_rowwise_scales_offset_Y + it * THREADS_Y_ROWWISE + tid_Y_rowwise) < chunk_rows;

if (rowwise_scale_is_within_bounds_X && rowwise_scale_is_within_bounds_Y) {
  scales_ptr[scale_idx_global] = S_dec_b_fp8;
}

constexpr float float_max = detail::TypeExtrema<float>::max;
float block_scale_inverse = fminf(
    1.0f / (static_cast<float>(S_dec_b_fp8) * S_dec_rowwise), float_max);
float2 block_scale_inverse_2x{block_scale_inverse, block_scale_inverse};
```

Rowwise quantization for 1D scaling:

```cpp
Vec<fp4e2m1x4, PACK_SIZE / 4> out;

#pragma unroll
for (int w = 0; w < WAVES; ++w) {
#pragma unroll
  for (int e = 0; e < PACK_SIZE / 4; ++e) {
    uint32_t rbits = get_rbits(rng, random_uint4, rnd_idx);  // 0 or random

    uint64_t elts = *reinterpret_cast<uint64_t*>(
        &in_IType[w].data.elt[2 * e]);  // 4 BF16 packed

    out.data.elt[e] = ptx::mul_cvt_bf16_to_fp4_4x<USE_STOCHASTIC_ROUNDING>(
        elts, block_scale_inverse_2x, rbits);
  }

  const size_t swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM;
  const size_t swizzled_idx       = swizzled_group_idx + thread_offset_X_rowwise;
  const size_t shmem_offset_rowwise =
      shmem_offset_base_rowwise_out + swizzled_idx / 2; // FP4x2 packs

  out.store_to(&out_data_sh[shmem_offset_rowwise]);
}
```

Again:

- **SR OFF** → `rbits = 0` → deterministic rounding.
- **SR ON**  → `rbits` from Philox, per 4 values.
- Swizzled indexing keeps shared-memory bank conflicts down.

### 2.8 Completing the stage: TMA stores

After rowwise + colwise quantization for this stage:

```cpp
ptx::fence_proxy_async_shared_cta();
__syncthreads();

if (is_master_thread) {
  const size_t global_offset_Y   = block_offset_Y   + stage_offset_Y;
  const size_t global_offset_X   = block_offset_X;

  const size_t global_offset_Y_t = block_offset_Y_t;
  const size_t global_offset_X_t = block_offset_X_t + stage_offset_Y;

  ptx::cp_async_bulk_tensor_2d_shared_to_global(
      reinterpret_cast<const uint64_t*>(&tensor_map_output),
      global_offset_X, global_offset_Y,
      reinterpret_cast<uint64_t*>(&out_data_sh[buff_offset_out]));

  if constexpr (RETURN_TRANSPOSE) {
    ptx::cp_async_bulk_tensor_2d_shared_to_global(
        reinterpret_cast<const uint64_t*>(&tensor_map_output_t),
        global_offset_X_t, global_offset_Y_t,
        reinterpret_cast<uint64_t*>(&out_t_data_sh[buff_offset_out_t]));
  }

  ptx::cp_async_bulk_commit_group();
}
```

- One bulk TMA store for rowwise tile.
- One bulk TMA store for transposed tile.
- `cp_async_bulk_commit_group()` groups them so `cp_async_bulk_wait_group_read` in the next stage can know when SHMEM is free.

### 2.9 Colwise scales writeback

After all stages:

```cpp
if (RETURN_TRANSPOSE && colwise_scale_is_within_bounds_Y) {
  using ScalesVec = Vec<nvfp4_scale_t, SCALES_PER_CHUNK_Y>; // 8

  size_t scale_idx_sh = tid_Y_t * SCALES_PER_CHUNK_Y;
  ScalesVec &scales_vec =
      *reinterpret_cast<ScalesVec*>(&out_colwise_scales_sh[scale_idx_sh]);

  size_t scale_idx_global =
      scales_offset_Y_t * scale_stride_t + scales_offset_X_t;

  size_t count = (chunk_rows >= CHUNK_DIM_Y)
                   ? SCALES_PER_CHUNK_Y
                   : (chunk_rows / SCALE_DIM);

  nvfp4_scale_t *dst = &scales_t_ptr[scale_idx_global];
  constexpr size_t vec_bytes = SCALES_PER_CHUNK_Y * sizeof(nvfp4_scale_t);

  if (count == SCALES_PER_CHUNK_Y &&
      (reinterpret_cast<uintptr_t>(dst) % vec_bytes == 0)) {
    scales_vec.store_to(dst);  // vectorized
  } else {
    scales_vec.store_to_elts(dst, 0, count); // scalar fallback
  }
}

destroy_barriers<STAGES>(mbar, is_master_thread);
```


### 2.10 Summary: 1D SR OFF vs SR ON

- **Same dataflow** for both SR OFF and SR ON:
  - Global → shared via TMA, double-buffered.
  - Shared → registers in vector packs.
  - Per-16-element block amax, per-block scale.
  - Shared → global via TMA.
- **Only difference**: `get_rbits` and how `mul_cvt_*_to_fp4_4x` uses those bits.
  - SR OFF: pure deterministic rounding → good performance, slightly biased.
  - SR ON: unbiased rounding → extra integer ops and RNG calls; overhead is modest and well hidden under memory latency / vectorized compute.

---

## 3. 2D Scaling, RETURN_TRANSPOSE = true

Now the host chooses:

```cpp
quantize_transpose<true>(...);
```

which activates:

```cpp
kernel = quantize_transpose_nvfp4_2D_kernel<...>;
```

Same grid, same TMA maps, same shared memory sizes.

### 3.1 Extra 2D-block constants

Inside `quantize_transpose_nvfp4_2D_kernel`:

```cpp
constexpr size_t BLOCK_DIM         = 16;  // 16x16 blocks
constexpr size_t BLOCKS_PER_TILE_Y = TILE_DIM_Y / BLOCK_DIM;  // 32/16 = 2
constexpr size_t BLOCKS_PER_TILE_X = TILE_DIM_X / BLOCK_DIM;  // 128/16 = 8

constexpr size_t ITERATIONS_BLOCK  = 2;  // passes to cover 2 blocks vertically in a tile
constexpr size_t BLOCKS_PER_WARP   = BLOCKS_PER_TILE_X / (THREADS_NUM / 32);
// For THREADS_NUM=128: 8 / (128/32) = 2

const size_t warp_id  = threadIdx.x / 32;
const size_t lane_id  = threadIdx.x % 32;
float thread_amax     = 0.0f;
const size_t block_in_warp = lane_id / BLOCKS_PER_WARP; // 0 or 1

__shared__ alignas(8) uint64_t mbar[STAGES];
__shared__ __align__(16) float block_amax_matrix[BLOCKS_PER_TILE_Y][BLOCKS_PER_TILE_X + 1];
```

- **BLOCK_DIM = 16**: each block covers **16×16** elements.
- Per 32×128 tile we have:
  - `BLOCKS_PER_TILE_Y = 2`, `BLOCKS_PER_TILE_X = 8` → 16 blocks per tile.
- With `THREADS_NUM = 128`, there are 4 warps, each handling some 16×16 blocks.
- `block_amax_matrix[2][9]` holds per-2D-block amax values (extra padding for bank alignment).

Warp reduction helper:

```cpp
auto warp_reduce_amax = [](float thread_amax, int block_in_warp) {
#pragma unroll
  for (int delta = 8; delta >= 1; delta /= 2) {
    float other_amax = __shfl_xor_sync(0xffffffff, thread_amax, delta);
    thread_amax = fmaxf(thread_amax, other_amax);
  }
  return thread_amax;
};
```

This uses warp shuffles to avoid shared-memory reductions.

### 3.2 Pipeline structure

The **pipeline skeleton** is almost identical to the 1D case:

```cpp
initialize_barriers<STAGES, THREADS_NUM>(mbar, is_master_thread);

copy_2d_to_shared(&in_sh[0], &tensor_map_input,
                  block_offset_X, block_offset_Y,
                  shmem_buff_size, &mbar[0], is_master_thread);

for (size_t stage = 0; stage < STAGES; ++stage) {
  const size_t buff         = stage % BUFFS_NUM;
  const size_t next_stage   = stage + 1;
  const size_t stage_offset_Y = stage * BUFF_DIM_Y;

  const size_t buff_offset_in    = buff * BUFF_IN_SIZE;
  const size_t buff_offset_out   = buff * BUFF_OUT_SIZE;
  const size_t buff_offset_out_t = buff * BUFF_OUT_T_SIZE;

  if (next_stage < STAGES) {
    ptx::cp_async_bulk_wait_group_read<1>();
    ...
    copy_2d_to_shared(&in_sh[next_buff_offset], ...);
  }

  ptx::fence_proxy_async_shared_cta();
  ptx::mbarrier_wait_parity(&mbar[stage], 0);

  // 2D block amax computation + quantization
  ...
}
```

Key difference: **how we compute block_amax and scales**.

### 3.3 2D block amax computation

Goal: compute `block_amax` for each 16×16 block in the current 32×128 tile.

The code (simplified):

```cpp
// For each tile-stage: iterate over blocks in Y (2) and X (8)
#pragma unroll
for (size_t it = 0; it < ITERATIONS_BLOCK; ++it) {   // 0..1 (two block rows)
  const size_t block_in_tile_y = it;                 // which 16-row half

  // Each warp covers some subset of blocks in X
  for (size_t block_in_tile_x = 0; block_in_tile_x < BLOCKS_PER_TILE_X; ++block_in_tile_x) {

    // Compute logical coords of this 16x16 block in the tile
    const size_t block_base_row_in_tile = block_in_tile_y * BLOCK_DIM;  // 0 or 16
    const size_t block_base_col_in_tile = block_in_tile_x * BLOCK_DIM;  // 0,16,...,112

    // Local row & column traversal for this thread
    for (size_t row_in_block = lane_id % BLOCK_DIM; row_in_block < BLOCK_DIM; row_in_block += 32) {
      const size_t row = block_base_row_in_tile + row_in_block;

      // Process 16 columns per block in PACK_SIZE chunks
      for (size_t col_in_block = 0; col_in_block < BLOCK_DIM; col_in_block += PACK_SIZE) {
        const size_t col = block_base_col_in_tile + col_in_block;

        // Vectorized load
        Vec<IType, PACK_SIZE> v;
        v.load_from(&in_sh[buff_offset_in + row * BUFF_IN_DIM_X + col]);

        // Accumulate abs-max into thread_amax
        ...
      }
    }

    // Warp-reduce thread_amax for this block
    float block_amax = warp_reduce_amax(thread_amax, block_in_warp);

    // Write am
    if (lane_id == 0 || lane_id == 16) {
      block_amax_matrix[block_in_tile_y][block_in_tile_x] = block_amax;
    }
  }
}

__syncthreads();  // ensure block_amax_matrix is fully populated
```

The actual code slices blocks slightly differently, but the **intent** is:

- Each warp’s threads cooperate to walk all 256 elements of a 16×16 block.
- They compute **max |x|** for that block and write it into `block_amax_matrix[by][bx]`.
- This is done **for all 16 blocks per 32×128 tile**:
  - `BLOCKS_PER_TILE_Y = 2` (two 16-row halves).
  - `BLOCKS_PER_TILE_X = 8` (eight 16-column segments).

So in 2D mode, block scales cover **16×16 patches**, not 16×1 or 1×16 strips.

### 3.4 Colwise path – 2D scaling + RETURN_TRANSPOSE

Now we reuse the computed `block_amax_matrix` for colwise work.

```cpp
if constexpr (RETURN_TRANSPOSE) {
#pragma unroll
  for (size_t it = 0; it < ITERATIONS_TRANSPOSE; ++it) {
    const size_t block_in_tile_y = it;                         // 0 or 1
    const size_t block_in_tile_x = threadIdx.x / BLOCK_DIM;    // 0..7

    const size_t in_thread_offset_Y  = 0 + it * SCALE_DIM;     // 0 or 16
    const size_t in_thread_offset_X  = thread_offset_X_colwise;// 0..127

    const size_t out_t_thread_offset_Y = thread_offset_X_colwise;
    const size_t out_t_thread_offset_X = 0 + it * BUFF_OUT_IT_OFFSET;

    const size_t shmem_offset_base_colwise_in =
        buff_offset_in + in_thread_offset_Y * BUFF_IN_DIM_X + in_thread_offset_X;
    const size_t shmem_offset_base_colwise_out_t =
        buff_offset_out_t + out_t_thread_offset_Y * BUFF_OUT_T_DIM_X + out_t_thread_offset_X;

    float block_amax = block_amax_matrix[block_in_tile_y][block_in_tile_x];
```

So each thread:

- Determines which **16×16 block** it belongs to via `block_in_tile_x` and `it`.
- Looks up `block_amax` for that block.
- Then loads its **16 elements down a column** exactly like in 1D mode:

```cpp
    float in_compute_colwise[SCALE_DIM];
    IType in_colwise_IType[SCALE_DIM];

    if constexpr (NO_ACTIVATIONS_NOT_FP32_INPUT) {
#pragma unroll
      for (int i = 0; i < SCALE_DIM; ++i) {
        int shmem_offset_colwise = shmem_offset_base_colwise_in + i * BUFF_IN_DIM_X;
        in_colwise_IType[i] = in_sh[shmem_offset_colwise];
      }
    } else {
      // FP32/activation path
    }

    nvfp4_scale_t S_dec_b_fp8 =
        compute_decoding_scaling_factor(block_amax, S_enc_colwise);

    size_t scale_idx_sh =
        tid_Y_t * SCALES_PER_CHUNK_Y + stage * ITERATIONS_TRANSPOSE + it;
    out_colwise_scales_sh[scale_idx_sh] = S_dec_b_fp8;

    constexpr float float_max = detail::TypeExtrema<float>::max;
    float block_scale_inverse = fminf(
        1.0f / (static_cast<float>(S_dec_b_fp8) * S_dec_colwise), float_max);
    float2 block_scale_inverse_2x{block_scale_inverse, block_scale_inverse};

    fp4e2m1x4 regs[SCALE_DIM / 4];
#pragma unroll
    for (int e = 0; e < SCALE_DIM / 4; ++e) {
      uint32_t rbits = get_rbits(rng, random_uint4, rnd_idx);
      uint64_t elts = *reinterpret_cast<uint64_t*>(&in_colwise_IType[4 * e]);
      regs[e] = ptx::mul_cvt_bf16_to_fp4_4x<USE_STOCHASTIC_ROUNDING>(
                   elts, block_scale_inverse_2x, rbits);
    }

    // same bank-avoiding swizzle as 1D
    const int group = thread_lane / 16;
    uint32_t *out_t_data_sh_as_uint32_t =
        reinterpret_cast<uint32_t *>(&out_t_data_sh[shmem_offset_base_colwise_out_t]);

    ...
  }
}
```

So compared to 1D:

- **Same per-thread data path** (16 elements down a column).
- **Different scale**: in 2D, scale is shared across a full 16×16 block.

### 3.5 Rowwise path – 2D scaling

Rowwise code is similar to 1D, but uses per-block `block_amax_matrix[block_in_tile_y][block_in_tile_x]`.

```cpp
const size_t stage_rowwise_scales_offset_Y = stage * BUFF_DIM_Y;

#pragma unroll
for (size_t it = 0; it < ITERATIONS_NORMAL; ++it) {
  const size_t block_in_tile_y = it;           // 0 or 1
  const size_t block_in_tile_x = tid_X_rowwise;// 0..7

  const size_t it_thread_offset_Y_rowwise =
      thread_offset_Y_rowwise + it * THREADS_Y_ROWWISE;

  const size_t shmem_offset_base_rowwise_in =
      buff_offset_in + it_thread_offset_Y_rowwise * BUFF_IN_DIM_X;
  const size_t shmem_offset_base_rowwise_out =
      buff_offset_out + it_thread_offset_Y_rowwise * BUFF_OUT_DIM_X;

  float block_amax = block_amax_matrix[block_in_tile_y][block_in_tile_x];
  float in_compute_rowwise[SCALE_DIM];
  Vec<IType,  PACK_SIZE>   in_cached[WAVES];
  Vec<IType2, PACK_SIZE/2> in_IType[WAVES];
```

Same BF16 path for reads as 1D, but now `block_amax` does **not** depend on the specific row segment; it’s shared across the 16×16 block.

Then rowwise scale computation and writeback:

```cpp
nvfp4_scale_t S_dec_b_fp8 =
    compute_decoding_scaling_factor(block_amax, S_enc_rowwise);

size_t scales_offset_Y =
    scales_offset_Y_rowwise + stage * BUFF_DIM_Y + it * THREADS_Y_ROWWISE;
size_t scales_offset_X = scales_offset_X_rowwise;
size_t scale_idx_global = scales_offset_Y * scale_stride + scales_offset_X;

bool rowwise_scale_is_within_bounds_Y =
    (stage_rowwise_scales_offset_Y + it * THREADS_Y_ROWWISE + tid_Y_rowwise) < chunk_rows;

if (rowwise_scale_is_within_bounds_X && rowwise_scale_is_within_bounds_Y) {
  scales_ptr[scale_idx_global] = S_dec_b_fp8;
}

float block_scale_inverse =
    1.0f / (static_cast<float>(S_dec_b_fp8) * S_dec_rowwise);
float2 block_scale_inverse_2x{block_scale_inverse, block_scale_inverse};
```

And quantization again uses `mul_cvt_bf16_to_fp4_4x<USE_STOCHASTIC_ROUNDING>` with vectorized swizzled stores, just like the 1D kernel.

The rest of the pipeline (TMA stores, colwise scale writeback, barrier destruction) is identical in structure to the 1D kernel.

### 3.6 SR ON vs OFF in 2D

The SR behavior is exactly analogous to 1D:

- **SR OFF**:
  - `random_uint4 = {0,0,0,0}`; `get_rbits` returns 0.
  - Conversion is deterministic.
- **SR ON**:
  - `random_uint4` seeded via Philox per thread.
  - Every 4 FP4 values share one 32-bit random `rbits`.

In 2D mode, SR interacts with larger blocks (16×16) but doesn’t change pipeline structure.

---

## 4. 1D vs 2D Scaling: Key Differences

Assume **128×128 chunk** per block and `SCALE_DIM = 16`.

### 4.1 Block granularity

- **1D scaling**:
  - Per-block scale covers **16 contiguous elements** along the feature axis.
  - Rowwise: each row of 128 columns → 8 scales (`128 / 16 = 8`).
  - 128 rows × 8 scales = 1024 rowwise scales per chunk.
  - Colwise scales computed similarly, for transposed layout.

- **2D scaling**:
  - Per-block scale covers **16×16 = 256 elements**.
  - Chunk decomposed into `8 × 8 = 64` blocks.
  - Scales shape: `[rows/16, cols/16]` instead of `[rows, cols/16]`.

### 4.2 Amax computation

- **1D**:
  - For each 16-element **rowwise block**, per-thread `thread_amax_2x` over its 16 values.
  - No cross-row warp reduction.
  - For colwise, each thread does 16 values down a column, again per-thread amax.

- **2D**:
  - For each 16×16 block:
    - Threads in a warp cooperatively sweep the 256 values.
    - Use **warp shuffles** (`__shfl_xor_sync`) to reduce `thread_amax` to `block_amax`.
    - Store `block_amax` into `block_amax_matrix[by][bx]`.
  - Rowwise and colwise passes reuse this 2D `block_amax` table.

So 2D performs **more intra-warp collaboration** and uses 2D structure of the data, while 1D treats each 16-wide slice independently.

### 4.3 Scale tensor geometry

- **1D rowwise scales**: `scale[row, col_block]` with `col_block = col / 16`.
- **1D colwise scales**: `scale_t[col, row_block]` with `row_block = row / 16`.
- **2D rowwise scales**: `scale[row_block, col_block]` (one per 16×16 block).
- **2D colwise scales**: `scale_t[col_block, row_block]` for transposed layout.

### 4.4 Performance implications

1. **Memory bandwidth & latency**
   - All modes share **identical TMA tiling** and **double-buffering**.
   - HBM → SM and SM → HBM traffic is the same shape.
   - 2D scaling reduces the number of scale values written (fewer scales per chunk), which slightly reduces scale memory traffic.

2. **Compute intensity**
   - 2D scaling adds **warp reductions** and uses `block_amax_matrix` lookups.
   - This increases arithmetic intensity (more math per byte), often good for hiding memory latency.

3. **Quantization quality vs compression**
   - 1D is more fine-grained (per 16 contiguous elements) and can adapt better to local magnitude variations along the feature dimension.
   - 2D shares scale across 256 values; slightly less precise but fewer scales and potentially better hardware utilization when fused with later GEMMs.

4. **Stochastic rounding**
   - SR overhead is similar in all modes: a few integer ops + `rng.generate4()` per ~16 values.
   - In these TMA-heavy kernels, SR overhead is modest compared to memory and FP4 convert cost.

### 4.5 ILP and SM utilization

Across all variants, the kernel keeps IPC and SM utilization high by:

- **Vectorization**:
  - Use of `Vec<IType, PACK_SIZE>` and `fp4e2m1x4` ensures 4–8 values are processed per instruction.
  - `mul_cvt_*_to_fp4_4x` fuses multiply, convert, and pack.

- **Double-buffered TMA**:
  - While one tile is being quantized, the next tile is being fetched into the other buffer.
  - `cp_async_bulk_wait_group_read` + `cp_async_bulk_commit_group` choreograph buffer reuse.

- **Warp-friendly work decomposition**:
  - 1D: each thread handles one `SCALE_DIM` block; no inter-thread sync required for amax.
  - 2D: each warp owns a set of 16×16 blocks and uses warp shuffles only (no SMEM) to reduce amax.

- **Shared-memory swizzling**:
  - `bank_group`, `swizzled_group_idx`, and the group-based transposed store pattern reduce bank conflicts.

- **Minimal divergence**:
  - All threads execute the same loops; boundary checks are coarse-grained (chunk tails), negligible in the well-aligned case (128×1024).

In short: 1D vs 2D and SR on/off are **orthogonal knobs** that affect scale geometry and rounding behavior, while the **core TMA + shared-memory + vectorized FP4 conversion pipeline** stays consistent and tuned for high throughput on Hopper/Blackwell-class GPUs.

