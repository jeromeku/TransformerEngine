import pytest
import torch

import transformer_engine.pytorch  # noqa: F401  loads libtransformer_engine.so before the torch ext
import transformer_engine_torch as tex

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

# (op_first_half, op_second_half): copy the half-extent partial into one half, leave the other.
COPY_OPS = [("none", "copy"), ("copy", "none")]
# Even padded slots (varied lengths + a physical-gap-style large slot), so each half is integral.
SLOTS = [16, 48, 16, 96, 32]
H, D = 4, 128  # H*D*2 % 16 == 0 for both the bf16 and the byte view


def _cu_seqlens(slots, device):
    cu = torch.zeros(len(slots) + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.tensor(slots, dtype=torch.int32, device=device).cumsum(0)
    return cu


def _make(shape_kind, total, half, device):
    if shape_kind == "q":
        grad = torch.randn(total, H, D, device=device, dtype=torch.bfloat16)
        gps = torch.randn(half, H, D, device=device, dtype=torch.bfloat16)
    else:
        grad = torch.randn(2, total, H, D, device=device, dtype=torch.bfloat16)
        gps = torch.randn(2, half, H, D, device=device, dtype=torch.bfloat16)
    return grad.contiguous(), gps.contiguous()


@requires_cuda
@pytest.mark.parametrize("shape_kind", ["q", "kv"])
@pytest.mark.parametrize("ops", COPY_OPS)
def test_byte_path_matches_bf16_path(shape_kind, ops):
    device = "cuda"
    cu = _cu_seqlens(SLOTS, device)
    total, half = sum(SLOTS), sum(SLOTS) // 2
    grad0, gps = _make(shape_kind, total, half, device)

    ref = grad0.clone()
    tex.thd_grad_correction(ref, gps, cu, ops[0], ops[1])

    test = grad0.clone().view(torch.uint8)
    tex.thd_grad_correction(test, gps.view(torch.uint8), cu, ops[0], ops[1])

    assert torch.equal(test.view(torch.bfloat16), ref)


@requires_cuda
@pytest.mark.parametrize("shape_kind", ["q", "kv"])
def test_fill_then_scatter_zeroes_absent_half(shape_kind):
    device = "cuda"
    cu = _cu_seqlens(SLOTS, device)
    total, half = sum(SLOTS), sum(SLOTS) // 2
    grad0, gps = _make(shape_kind, total, half, device)

    # The delayed-fp8 usage: zero the slot, then copy the partial into the second half.
    dst = grad0.clone().view(torch.uint8)
    dst.fill_(0)
    tex.thd_grad_correction(dst, gps.view(torch.uint8), cu, "none", "copy")
    got = dst.view(torch.bfloat16)

    seq_dim = 0 if shape_kind == "q" else 1
    off = 0
    for s in SLOTS:
        h = s // 2
        first = got.narrow(seq_dim, off, h)
        second = got.narrow(seq_dim, off + h, h)
        assert torch.equal(first, torch.zeros_like(first))
        assert torch.equal(second, gps.narrow(seq_dim, off // 2, h))
        off += s


@requires_cuda
def test_add_op_rejected_for_fp8_bytes():
    device = "cuda"
    cu = _cu_seqlens(SLOTS, device)
    total, half = sum(SLOTS), sum(SLOTS) // 2
    grad0, gps = _make("q", total, half, device)
    with pytest.raises(RuntimeError, match="add"):
        tex.thd_grad_correction(
            grad0.view(torch.uint8), gps.view(torch.uint8), cu, "add", "copy"
        )
