# Memory Swizzling in HadamardAmaxTmaKernel

## The Swizzle Function

**Location**: [hadamard_transform.cu:191-197](../../../transformer_engine/common/hadamard_transform/hadamard_transform.cu#L191-L197)

```cpp
__device__ __forceinline__ uint32_t swizzle_128B_atom_32B(uint32_t gmem_row_idx,
                                                          uint32_t gmem_col_idx) {
  uint32_t smem_row_idx = gmem_row_idx;
  uint32_t xor_factor = (smem_row_idx * 2) % 8;
  uint32_t smem_col_idx = gmem_col_idx ^ xor_factor;
  return smem_row_idx * 8 + smem_col_idx;
}
```

## What is Memory Swizzling?

Memory swizzling is a technique to avoid **shared memory bank conflicts** on GPUs. Shared memory is divided into banks (32 on modern GPUs), and when multiple threads in a warp access the same bank simultaneously, it causes serialization.

## The Problem Without Swizzling

When threads load a matrix in row-major order:
- Thread 0 loads column 0
- Thread 1 loads column 1
- Thread 2 loads column 2
- ...

If columns are laid out consecutively, and each column index maps to the same bank, we get **many-way bank conflicts**.

## How This Swizzle Works

### Input Parameters

For a thread loading from a 16×16 tile:
- `gmem_row_idx`: Which row (0-15)
- `gmem_col_idx`: Which column group (in units of 32 bytes)

### Calculation Example

Let's trace through several examples:

#### Row 0, Column 0
```
gmem_row_idx = 0
gmem_col_idx = 0

smem_row_idx = 0
xor_factor = (0 * 2) % 8 = 0
smem_col_idx = 0 ^ 0 = 0
return 0 * 8 + 0 = 0
```

#### Row 1, Column 0
```
gmem_row_idx = 1
gmem_col_idx = 0

smem_row_idx = 1
xor_factor = (1 * 2) % 8 = 2
smem_col_idx = 0 ^ 2 = 2
return 1 * 8 + 2 = 10
```

#### Row 2, Column 0
```
gmem_row_idx = 2
gmem_col_idx = 0

smem_row_idx = 2
xor_factor = (2 * 2) % 8 = 4
smem_col_idx = 0 ^ 4 = 4
return 2 * 8 + 4 = 20
```

#### Row 3, Column 0
```
gmem_row_idx = 3
gmem_col_idx = 0

smem_row_idx = 3
xor_factor = (3 * 2) % 8 = 6
smem_col_idx = 0 ^ 6 = 6
return 3 * 8 + 6 = 30
```

### Full Swizzle Pattern Table

| Row | Col | xor_factor | smem_col | Linear Index | Bank (assuming 4-byte words, 32 banks) |
|-----|-----|------------|----------|--------------|----------------------------------------|
| 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | 0 | 2 | 2 | 10 | 10 |
| 2 | 0 | 4 | 4 | 20 | 20 |
| 3 | 0 | 6 | 6 | 30 | 30 |
| 4 | 0 | 0 | 0 | 32 | 0 |
| 5 | 0 | 2 | 2 | 42 | 10 |
| 6 | 0 | 4 | 4 | 52 | 20 |
| 7 | 0 | 6 | 6 | 62 | 30 |
| 8 | 0 | 0 | 0 | 64 | 0 |
| ... | | | | | |

### Why This Pattern Avoids Conflicts

The XOR with `(row * 2) % 8` creates a pattern where:
1. Consecutive rows access different column offsets
2. The pattern repeats every 4 rows (since `(4 * 2) % 8 = 0`)
3. Within a warp, threads access distributed banks

## Swizzle Pattern Visualization

### Unswizzled (row-major, causes bank conflicts):
```
Col:  0   1   2   3   4   5   6   7
Row 0: [A] [ ] [ ] [ ] [ ] [ ] [ ] [ ]
Row 1: [B] [ ] [ ] [ ] [ ] [ ] [ ] [ ]
Row 2: [C] [ ] [ ] [ ] [ ] [ ] [ ] [ ]
Row 3: [D] [ ] [ ] [ ] [ ] [ ] [ ] [ ]

If threads A,B,C,D access column 0 → all access bank 0 → 4-way conflict
```

### Swizzled (XOR pattern, avoids conflicts):
```
Col:  0   1   2   3   4   5   6   7
Row 0: [A] [ ] [ ] [ ] [ ] [ ] [ ] [ ]   → smem_col = 0
Row 1: [ ] [ ] [B] [ ] [ ] [ ] [ ] [ ]   → smem_col = 2
Row 2: [ ] [ ] [ ] [ ] [C] [ ] [ ] [ ]   → smem_col = 4
Row 3: [ ] [ ] [ ] [ ] [ ] [ ] [D] [ ]   → smem_col = 6

Threads A,B,C,D access banks 0, 2, 4, 6 → no conflict!
```

## Integration with TMA

The swizzle pattern specified in the TensorMap configuration:
```cpp
CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B
```

This tells TMA hardware to:
- Use 128-byte atoms (groups of consecutive data)
- Apply 32-byte swizzle granularity
- Automatically apply the XOR swizzle pattern when loading from global to shared memory

This means the data arrives in shared memory **already swizzled**, and the `swizzle_128B_atom_32B` function computes the **correct swizzled address** for threads to read from.

## Why "128B Atom, 32B Swizzle"?

- **128B atom**: TMA loads data in 128-byte chunks (atomic units of transfer)
- **32B swizzle**: The XOR pattern operates on 32-byte boundaries
  - 32 bytes = 16 × bf16 = one row of a 16×16 tile

## Relation to ldmatrix

After data is swizzled in shared memory, `ldmatrix` reads it:

```cpp
ldmatrix_x4_m8n8_shared_b16<false>(a_frag[0], a_frag[1], a_frag[2], a_frag[3],
                                   reinterpret_cast<uint4*>(in_sh_ptr) + swizzle_idx);
```

The `swizzle_idx` tells ldmatrix **where to start reading** in the swizzled layout. `ldmatrix` is designed to work with this swizzled pattern and loads data in the format expected by tensor cores.

## Key Takeaway

The swizzle pattern ensures that:
1. TMA loads data into shared memory in a bank-conflict-free layout
2. Threads compute swizzled addresses to access their data
3. ldmatrix reads from these swizzled addresses efficiently
4. Tensor cores receive data in the correct layout for computation

This entire pipeline is carefully designed to maximize memory bandwidth and avoid any serialization from bank conflicts.
