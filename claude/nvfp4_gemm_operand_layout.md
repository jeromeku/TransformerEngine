# NVFP4 GEMM Operand Layout Explanation

## Question
Why is `w_nvfp4_native` passed as **operand A** with `transa=True` and `x_nvfp4_native` passed as **operand B** with `transb=False`?

---

## Quick Answer

The operands are swapped and transposed to match **cuBLAS GEMM conventions** while computing the desired mathematical operation:

**Desired**: `Y = X @ W.T` (where X is M×K, W is N×K)

**cuBLAS call**: `Y = op(A) @ op(B)` where A=W, B=X, with appropriate transposes

This is a common pattern in deep learning frameworks to efficiently compute `XW^T` using weight matrices stored in row-major format.

---

## Detailed Explanation

### 1. Problem Setup

**File**: [experiments/test_nvfp4_gemm_exact.py:35-38](../experiments/test_nvfp4_gemm_exact.py#L35-L38)

```python
# Input tensor shapes
x_shape = (K, M) if x_columnwise else (M, K)  # x_columnwise=False → (M, K)
w_shape = (K, N) if w_columnwise else (N, K)  # w_columnwise=False → (N, K)

x = torch.randn(x_shape, dtype=x_dtype, device=device)  # Shape: (M, K)
w = torch.randn(w_shape, dtype=w_dtype, device=device)  # Shape: (N, K)
```

**For default case** (x_columnwise=False, w_columnwise=False):
- `x`: **(M × K)** - Input activations, M examples with K features each
- `w`: **(N × K)** - Weight matrix, N output neurons with K input features each

### 2. Desired Mathematical Operation

The goal is to compute a **linear transformation**:

```
Y = X @ W^T
```

Where:
- `X`: (M × K)
- `W^T`: (K × N) - transposed weight matrix
- `Y`: (M × N) - output

**Expanded form**:
```
Y[i, j] = Σ(k=0 to K-1) X[i, k] * W[j, k]
```

This is the standard **fully-connected layer** operation in neural networks.

### 3. cuBLAS GEMM Interface

**cuBLAS GEMM** computes:
```
C = α * op(A) @ op(B) + β * C
```

Where `op(X)` can be:
- `op(X) = X` (no transpose)
- `op(X) = X^T` (transpose)

**Function signature** (conceptual):
```cpp
cublas_gemm(
    opA,      // transpose operation on A
    opB,      // transpose operation on B
    M,        // rows of op(A) and C
    N,        // columns of op(B) and C
    K,        // columns of op(A) and rows of op(B)
    A,        // first operand
    B,        // second operand
    C         // output
)
```

### 4. Why Swap Operands?

**Option 1: Direct approach (seems natural)**
```python
# Try to do: Y = X @ W^T
# Pass: A=X (M×K), B=W (N×K), transb=True
# Compute: Y = X @ W^T
```

**Problem**: This would require:
- `A = X` with `transa=False` → op(A) = X (M×K)
- `B = W` with `transb=True` → op(B) = W^T (K×N)
- Result: `Y = op(A) @ op(B) = X @ W^T` ✓

**However**, there's a **memory layout issue**...

### 5. Memory Layout Considerations

**PyTorch/NumPy default**: Row-major (C-order)
- Matrix A[M][K] stored as: `[row0_col0, row0_col1, ..., row0_colK-1, row1_col0, ...]`
- Consecutive elements in a row are contiguous in memory

**cuBLAS default**: Column-major (Fortran-order)
- Matrix A[M][K] stored as: `[col0_row0, col1_row0, ..., colK-1_row0, col0_row1, ...]`
- Consecutive elements in a column are contiguous in memory

**Key insight**: A row-major (M×K) matrix looks like a column-major (K×M) transposed matrix!

```
Row-major X[M][K]:        Column-major interpretation:
┌─────────┐               ┌─────────┐
│ x00 x01 │               │ x00 x10 │  ← Looks like X^T[K][M]
│ x10 x11 │  stored as    │ x01 x11 │     in column-major
└─────────┘               └─────────┘
[x00, x01, x10, x11]      [x00, x10, x01, x11] (col-major view)
```

### 6. The Swap Trick

To compute `Y = X @ W^T` using cuBLAS efficiently:

**Mathematical identity**:
```
Y = X @ W^T

Taking transpose of both sides:
Y^T = (X @ W^T)^T
Y^T = W @ X^T
```

So instead of computing `Y` directly, compute `Y^T` and reinterpret!

**In cuBLAS (column-major)**:
```
# Compute Y^T = W @ X^T
# Pass W as operand A, X as operand B
cublas_gemm(
    W,        # A: (N × K) in row-major = (K × N)^T in col-major
    transa=True,   # op(A) = W^T: (K × N)
    X,        # B: (M × K) in row-major = (K × M)^T in col-major
    transb=False,  # op(B) = X^T: (K × M)
    result    # (N × M) in col-major = (M × N)^T, which when viewed as row-major gives us Y!
)
```

**Result**: The output, when interpreted as row-major, is exactly `Y = X @ W^T`!

### 7. Code Implementation

**File**: [experiments/test_nvfp4_gemm_exact.py:145-162](../experiments/test_nvfp4_gemm_exact.py#L145-L162)

```python
# Determine transpose flags
transa = True if not w_columnwise else False    # transa = True (since w_columnwise=False)
transb = False if not x_columnwise else True    # transb = False (since x_columnwise=False)

# cuBLAS GEMM call
y_native = tex.generic_gemm(
    w_nvfp4_native,   # Operand A: W (N × K)
    transa,           # True: apply transpose to W → W^T (K × N)
    x_nvfp4_native,   # Operand B: X (M × K)
    transb,           # False: treat X as X^T in column-major → (K × M)
    ...
)
```

**What cuBLAS actually computes** (in column-major view):
```
# In column-major interpretation:
# A = W (N × K) in row-major → (K × N)^T in col-major
# op(A) = (K × N)^T with transa=True → (K × N) [transpose cancels]
#       = W^T in our row-major view

# B = X (M × K) in row-major → (K × M)^T in col-major
# op(B) = (K × M)^T with transb=False → (K × M)^T
#       = X^T in our row-major view

# Result = op(A) @ op(B) = W^T @ X^T in col-major
#        = (X @ W)^T in row-major
#        = X @ W^T when reinterpreted!  ✓
```

Wait, let me reconsider this more carefully...

### 8. Correct Analysis

Let's think step-by-step with explicit memory layouts:

**Given**:
- `X`: (M × K) in row-major PyTorch
- `W`: (N × K) in row-major PyTorch
- Want: `Y = X @ W^T` (M × N)

**cuBLAS expects column-major**, so when we pass row-major tensors:
- Row-major (M × K) → cuBLAS sees it as column-major (K × M)

**Without any transposes**:
```
PyTorch row-major X (M × K) → cuBLAS interprets as X^T (K × M) in column-major
PyTorch row-major W (N × K) → cuBLAS interprets as W^T (K × N) in column-major
```

**Now apply the GEMM operation**:

```python
# Pass: A=W, B=X, transa=True, transb=False

# cuBLAS sees (in its column-major view):
# A = W (N × K) row-major → W^T (K × N) col-major
# op(A) with transa=True: (W^T)^T = W (N × K) col-major
#                       = W^T (K × N) row-major (our actual weight transpose!)

# B = X (M × K) row-major → X^T (K × M) col-major
# op(B) with transb=False: X^T (K × M) col-major
#                        = X (M × K) row-major (our actual input!)

# cuBLAS computes (in column-major):
# C = op(A) @ op(B)
#   = W @ X^T  (in column-major view)

# But this is stored in column-major format (N × M)
# When we interpret this as row-major (PyTorch's view):
# C_rowmajor = (W @ X^T)^T = X @ W^T  ✓
```

Actually, I think I'm overcomplicating. Let me use the standard GEMM swap trick explanation:

### 9. Standard Explanation: The Row/Column-Major GEMM Swap

**Core principle**:
```
(A @ B)^T = B^T @ A^T
```

**Problem**: Compute `Y = X @ W^T` in row-major (PyTorch)

**Solution**: Use column-major GEMM (cuBLAS) to compute `Y^T`:
```
Y = X @ W^T

Taking transpose:
Y^T = (X @ W^T)^T
Y^T = W @ X^T
```

**In cuBLAS column-major**:
- Pass W as matrix A
- Pass X as matrix B
- Apply transa=True to get W^T from W
- X is already implicitly transposed due to row→col major reinterpretation

**Result**: cuBLAS computes and stores `Y^T` in column-major, which is `Y` in row-major!

### 10. Visual Walkthrough

```
Step 1: Original tensors (row-major)
X (M × K):          W (N × K):
┌───────┐           ┌───────┐
│ • • • │ M rows    │ • • • │ N rows
│ • • • │           │ • • • │
└───────┘           └───────┘
  K cols              K cols

Step 2: Desired operation
Y = X @ W^T (M × N):
┌─────────┐
│ • • • • │ M rows
│ • • • • │
└─────────┘
    N cols

Step 3: cuBLAS sees (column-major interpretation of row-major data)
A = W (N × K) row-major → sees as (K × N) with different layout
B = X (M × K) row-major → sees as (K × M) with different layout

Step 4: Apply transposes
transa=True on A:  (K × N) → (N × K) after transpose
transb=False on B: (K × M) stays (K × M)

Wait, this is getting too confusing. Let me use the simplest explanation:
```

### 11. Simplest Explanation

**The key insight**: cuBLAS is column-major, PyTorch is row-major.

When you pass a PyTorch row-major (M×K) matrix to cuBLAS:
- cuBLAS interprets it as a (K×M) column-major matrix
- This is equivalent to the transpose!

**To compute `Y = X @ W^T`**:

Instead, compute `Y^T = W @ X^T` using:
```python
tex.generic_gemm(
    W,            # (N × K) → cuBLAS sees as (K × N) col-major ≈ W in our view
    transa=True,  # Apply transpose: W^T
    X,            # (M × K) → cuBLAS sees as (K × M) col-major ≈ X^T in our view
    transb=False, # Don't transpose: stays X^T
    ...
)
```

**cuBLAS computes**: `W^T @ X^T = (X @ W)^T = Y^T` (in column-major)

**PyTorch receives**: The column-major result, interprets it as row-major → gets `Y` ✓

---

## Summary Table

| Item | Value | Interpretation |
|------|-------|----------------|
| **Input X** | (M × K) row-major | Activations |
| **Weight W** | (N × K) row-major | Weight matrix |
| **Desired output** | Y = X @ W^T (M × N) | Linear layer output |
| **Operand A** | W (N × K) | Weight passed first |
| **transa** | True | Transpose W → W^T |
| **Operand B** | X (M × K) | Input passed second |
| **transb** | False | Let col-major reinterpret as X^T |
| **cuBLAS computes** | W^T @ X^T (col-major) | Equivalent to (X @ W)^T |
| **PyTorch receives** | (M × N) row-major | Reinterpreted as Y = X @ W^T ✓ |

---

## Why This Matters for NVFP4

**File**: [experiments/test_nvfp4_gemm_exact.py:77-92](../experiments/test_nvfp4_gemm_exact.py#L77-L92)

```python
# Extract quantized data based on memory layout
qx_data = (
    x_nvfp4_native._columnwise_data.view(dtype=torch.uint8)
    if x_columnwise
    else x_nvfp4_native._rowwise_data.view(dtype=torch.uint8)  # Used when x_columnwise=False
)
qw_data = (
    w_nvfp4_native._columnwise_data.view(dtype=torch.uint8)
    if w_columnwise
    else w_nvfp4_native._rowwise_data.view(dtype=torch.uint8)  # Used when w_columnwise=False
)
```

**NVFP4 tensors** store two versions:
1. **Rowwise data**: Quantized with row-major layout
2. **Columnwise data**: Quantized with column-major layout (transposed)

The `transa` and `transb` flags determine which version to use:
- `transa=True` on W (row-major) → effectively uses W^T
- `transb=False` on X (row-major) → cuBLAS's col-major view gives X^T

This ensures the correct quantized data and scales are accessed for efficient GEMM computation!

---

## References

- **cuBLAS GEMM documentation**: https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemm
- **Row-major vs Column-major**: https://en.wikipedia.org/wiki/Row-_and_column-major_order
- **Matrix transpose identity**: (AB)^T = B^T A^T
