"""Bit-exact + zero-sync oracle for the THD a2a chunk-reorder permutations.

The two helpers reorder packed-THD sequence chunks around the a2a collective. They are correctness-
critical permutations, so any efficiency refactor (vectorizing the index build) must produce a
bit-for-bit identical result and must not add a device->host sync. This file is the gate:

  * `_ref_before` / `_ref_after` are faithful copies of the original Python index construction, used
    as the reference oracle -- run against the library functions, they must match exactly.
  * The docstring ground-truth examples (cu_seqlens=[0,8,16,24,40], cp=4) are implementation-
    independent expected outputs, checked directly.
  * before/after are inverses given the matching seq_chunk_ids -> a round-trip identity.
  * The library helpers must do zero device->host syncs (set_sync_debug_mode("error")).

Run first against the current (Python-loop) implementation to confirm the oracle is faithful, then
against the vectorized implementation.

    python3 -m pytest test_thd_a2a_reorder.py -q -rA
"""

import itertools

import pytest
import torch

from transformer_engine.pytorch.attention.dot_product_attention.context_parallel import (
    get_seq_chunk_ids_for_reordering_after_attn,
    reorder_seq_chunks_after_a2a_before_attn_thd,
    reorder_seq_chunks_before_a2a_after_attn_thd,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="device-only")


# --- reference oracle: faithful copies of the original Python index construction ----------------


def _ref_before(x, cu_seqlens, cp_size, seq_dim=0):
    total_slices_of_any_sequence = 2 * cp_size
    slice_sizes = (cu_seqlens[1:] - cu_seqlens[:-1]) // total_slices_of_any_sequence
    indices = [
        (
            torch.arange(
                seq_start + (cp_rank * slice_size),
                seq_start + ((cp_rank + 1) * slice_size),
                device=cu_seqlens.device,
            ),
            torch.arange(
                seq_start + ((total_slices_of_any_sequence - cp_rank - 1) * slice_size),
                seq_start + ((total_slices_of_any_sequence - cp_rank) * slice_size),
                device=cu_seqlens.device,
            ),
        )
        for cp_rank in range(cp_size)
        for slice_size, seq_start in zip(slice_sizes, cu_seqlens[:-1])
    ]
    indices = list(itertools.chain(*indices))
    indices = torch.cat(indices)
    return x.index_select(seq_dim, indices)


def _ref_after(x, cu_seqlens, seq_chunk_ids, cp_size, seq_dim=0):
    max_cum_seqlen_per_cp_rank = cu_seqlens[-1] // cp_size
    cu_seqlens_on_any_cp_rank = cu_seqlens // cp_size
    indices = [
        torch.arange(
            (
                start + max_cum_seqlen_per_cp_rank * (chunk_id // 2)
                if loc < cp_size
                else (start + end) // 2 + max_cum_seqlen_per_cp_rank * (chunk_id // 2)
            ),
            (
                (start + end) // 2 + max_cum_seqlen_per_cp_rank * (chunk_id // 2)
                if loc < cp_size
                else end + max_cum_seqlen_per_cp_rank * (chunk_id // 2)
            ),
            device=cu_seqlens.device,
        )
        for start, end in zip(cu_seqlens_on_any_cp_rank[:-1], cu_seqlens_on_any_cp_rank[1:])
        for loc, chunk_id in enumerate(seq_chunk_ids)
    ]
    indices = torch.cat(indices)
    return x.index_select(seq_dim, indices)


# --- fixtures ------------------------------------------------------------------------------------


def _payload(total_tokens, hd=(3, 8)):
    # A distinctly-valued [T, h, d] tensor so index_select mistakes are visible.
    return torch.arange(total_tokens * hd[0] * hd[1], device="cuda", dtype=torch.float32).view(
        total_tokens, *hd
    )


def _cu(doc_lens):
    return torch.tensor([0, *itertools.accumulate(doc_lens)], device="cuda", dtype=torch.int32)


# cp_size -> document lengths, each a multiple of 2*cp_size (the CP padding invariant).
CASES = [
    (4, [8, 8, 8, 16]),      # the docstring example
    (2, [4, 8, 12]),
    (4, [8, 24, 16]),
    (8, [16, 32, 48, 16]),
    (2, [16]),               # single document
]


# --- the gate ------------------------------------------------------------------------------------


@pytest.mark.parametrize("cp_size,doc_lens", CASES)
def test_before_matches_reference(cp_size, doc_lens):
    cu = _cu(doc_lens)
    x = _payload(sum(doc_lens))
    got = reorder_seq_chunks_before_a2a_after_attn_thd(x, cu, cp_size)
    ref = _ref_before(x, cu, cp_size)
    assert torch.equal(got, ref), f"before-a2a permutation differs from reference (cp={cp_size})"


@pytest.mark.parametrize("cp_size,doc_lens", CASES)
def test_after_matches_reference(cp_size, doc_lens):
    cu = _cu(doc_lens)
    x = _payload(sum(doc_lens))
    chunk_ids = get_seq_chunk_ids_for_reordering_after_attn(cp_size, x.device)
    got = reorder_seq_chunks_after_a2a_before_attn_thd(x, cu, chunk_ids, cp_size)
    ref = _ref_after(x, cu, chunk_ids, cp_size)
    assert torch.equal(got, ref), f"after-a2a permutation differs from reference (cp={cp_size})"


def test_before_docstring_ground_truth():
    # Implementation-independent: the values printed in the before-a2a helper docstring. Documents
    # are [0..7], [0..7], [0..7], [0..15] (cu_seqlens=[0,8,16,24,40]); the fourth is 16 tokens.
    cp_size = 4
    cu = _cu([8, 8, 8, 16])
    x = torch.tensor(
        [i for _ in range(3) for i in range(8)] + list(range(16)),
        device="cuda",
        dtype=torch.float32,
    ).unsqueeze(-1)
    before = reorder_seq_chunks_before_a2a_after_attn_thd(x, cu, cp_size).squeeze(-1)
    expected_before = torch.tensor(
        [0, 7, 0, 7, 0, 7, 0, 1, 14, 15, 1, 6, 1, 6, 1, 6, 2, 3, 12, 13,
         2, 5, 2, 5, 2, 5, 4, 5, 10, 11, 3, 4, 3, 4, 3, 4, 6, 7, 8, 9],
        device="cuda", dtype=torch.float32,
    )
    assert torch.equal(before, expected_before)


@pytest.mark.parametrize("helper", ["before", "after"])
def test_no_device_to_host_sync(helper):
    # The efficiency point: the index build must be fully on-device. sync_debug_mode("error")
    # raises if the helper reads a CUDA scalar to host. This is the gate that fails on the original
    # Python-loop implementation (arange over CUDA-scalar bounds) and must pass after vectorizing.
    cp_size = 4
    cu = _cu([8, 24, 16])
    x = _payload(48)
    torch.cuda.synchronize()
    prev = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        if helper == "before":
            reorder_seq_chunks_before_a2a_after_attn_thd(x, cu, cp_size)
        else:
            chunk_ids = get_seq_chunk_ids_for_reordering_after_attn(cp_size, x.device)
            reorder_seq_chunks_after_a2a_before_attn_thd(x, cu, chunk_ids, cp_size)
    finally:
        torch.cuda.set_sync_debug_mode(prev)
