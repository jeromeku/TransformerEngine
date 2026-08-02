"""Compare packed (THD) FP8 attention sharded across ranks against the same problem on one rank.

Launched under torchrun by test_packed_attention_sharded.py, which asserts on the exit status. It
is a plain script rather than a pytest module because per-test setup, fixtures and skips inside
ranks that are also inside collectives deadlock as soon as one rank takes a different path from the
others.

Attention is sharded by head. Each rank receives a contiguous slice of the query heads and the
key-value groups they map to, and the outputs are gathered along the head dimension. `--config` is
the whole model's geometry; the per-rank geometry that results is checked against the named one the
deployment runs, so a sharding that produces a shape nobody runs is caught rather than measured.

The two scaling recipes behave differently and both are asserted:

  delayed   the sharded result is bitwise identical to the unsharded one. Heads do not interact,
            and each rank's contiguous query slice maps to its own key-value groups, so per-head
            arithmetic and the accumulation order within each group are both preserved.
  current   the scale comes from the amax of the tensor in hand, so a rank holding half the heads
            scales differently from one holding all of them and the results differ. The difference
            is bounded rather than eliminated. Supplying an amax reduction group does not change
            it, which --no-amax-reduction demonstrates.

Exit status is 0 only if every rank agrees.

    torchrun --nproc_per_node=2 run_packed_attention_sharded.py --config omnii_8b_tp1
"""

import argparse
import math
import os
import pathlib
import sys

import torch
import torch.distributed as dist

_current_file = pathlib.Path(__file__).resolve()
sys.path = [str(_current_file.parent.parent / "attention"),
            str(_current_file.parent.parent)] + sys.path

import transformer_engine.pytorch as te
from transformer_engine.common import recipe

from packed_input_utils import PACKED_CONFIGS, make_packed_batch

RECIPES = {"delayed": recipe.DelayedScaling, "current": recipe.Float8CurrentScaling}
TENSOR_NAMES = ("out", "dq", "dk", "dv")
WARMUP_STEPS = 3
GRADIENT_SEED = 1234


def head_slice(total_heads, world_size, rank):
    per_rank = total_heads // world_size
    return slice(rank * per_rank, (rank + 1) * per_rank)


def upstream_gradient(batch, heads):
    torch.manual_seed(GRADIENT_SEED)
    return torch.randn(batch.total_tokens, batch.config.num_heads, batch.config.head_dim_v,
                       device="cuda", dtype=torch.bfloat16)[:, :heads]


def run(batch, heads, groups, q, k, v, grad_out, mask, rec, fp8_group):
    """`WARMUP_STEPS` forward and backward iterations over the given head counts."""
    module = te.DotProductAttention(
        num_attention_heads=heads, kv_channels=batch.config.head_dim_qk, num_gqa_groups=groups,
        attention_dropout=0.0, qkv_format="thd", attn_mask_type=mask,
        softmax_scale=1.0 / math.sqrt(batch.config.head_dim_qk),
    ).cuda()
    kwargs = dict(cu_seqlens_q=batch.cu_seqlens, cu_seqlens_kv=batch.cu_seqlens,
                  max_seqlen_q=batch.max_seqlen, max_seqlen_kv=batch.max_seqlen,
                  attn_mask_type=mask)
    for _ in range(WARMUP_STEPS):
        qq, kk, vv = (x.clone().detach().requires_grad_(True) for x in (q, k, v))
        with te.fp8_autocast(enabled=True, fp8_recipe=RECIPES[rec](fp8_dpa=True),
                             fp8_group=fp8_group):
            out = module(qq, kk, vv, **kwargs)
        out.backward(grad_out.reshape(tuple(out.shape)))
    return (out.detach(), qq.grad, kk.grad, vv.grad)


def relative_rms(got, want):
    got, want = got.float(), want.float()
    return (got - want).pow(2).mean().sqrt().item() / max(
        want.pow(2).mean().sqrt().item(), 1e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="omnii_8b_tp1", help="the whole model's geometry")
    parser.add_argument("--expect-per-rank-config", default=None,
                        help="name of the geometry each rank should end up with")
    parser.add_argument("--mask", default="padding_causal")
    parser.add_argument("--recipe", default="delayed")
    parser.add_argument("--require-exact", action="store_true")
    parser.add_argument("--max-relative-rms", type=float, default=0.15)
    parser.add_argument("--no-amax-reduction", action="store_true",
                        help="withhold the amax reduction group, so each rank scales from its own "
                             "heads")
    parser.add_argument("--wrong-head-slice", action="store_true",
                        help="give every rank the first rank's heads, so the shard is wrong")
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl")
    group = dist.new_group(backend="nccl")

    config = PACKED_CONFIGS[args.config]
    if config.num_heads % world_size or config.num_gqa_groups % world_size:
        if rank == 0:
            print(f"{args.config} has {config.num_heads} query and {config.num_gqa_groups} "
                  f"key-value heads, which do not divide across {world_size} ranks",
                  file=sys.stderr)
        dist.destroy_process_group()
        raise SystemExit(2)

    batch = make_packed_batch(config, "long_tailed_4k", seed=0)
    grad_full = upstream_gradient(batch, config.num_heads)

    # Every head on one rank, computed identically on each rank so no broadcast is needed.
    single = run(batch, config.num_heads, config.num_gqa_groups, batch.q, batch.k, batch.v,
                 grad_full, args.mask, args.recipe, None)

    source = 0 if args.wrong_head_slice else rank
    q_heads = head_slice(config.num_heads, world_size, source)
    kv_heads = head_slice(config.num_gqa_groups, world_size, source)
    local_q, local_kv = config.num_heads // world_size, config.num_gqa_groups // world_size

    # Assert the input is really sharded before asserting anything about its numbers.
    assert batch.q[:, q_heads].shape[1] == local_q, "query heads were not sharded as expected"
    assert batch.k[:, kv_heads].shape[1] == local_kv, "key-value heads were not sharded"
    assert world_size > 1 and local_q < config.num_heads, "the heads were not sharded at all"
    if args.expect_per_rank_config:
        want = PACKED_CONFIGS[args.expect_per_rank_config]
        assert (local_q, local_kv) == (want.num_heads, want.num_gqa_groups), (
            f"sharding {args.config} across {world_size} ranks gives {local_q}/{local_kv} heads, "
            f"but {args.expect_per_rank_config} is {want.num_heads}/{want.num_gqa_groups}"
        )

    sharded = run(batch, local_q, local_kv, batch.q[:, q_heads], batch.k[:, kv_heads],
                  batch.v[:, kv_heads], grad_full[:, q_heads], args.mask, args.recipe,
                  None if args.no_amax_reduction else group)

    failed = torch.zeros(1, dtype=torch.uint8, device="cuda")
    for i, name in enumerate(TENSOR_NAMES):
        gathered = [torch.empty_like(sharded[i]) for _ in range(world_size)]
        dist.all_gather(gathered, sharded[i].contiguous(), group=group)
        joined = torch.cat(gathered, dim=1)
        error = relative_rms(joined, single[i])
        identical = torch.equal(joined, single[i])

        if not torch.isfinite(joined.float()).all():
            failed[0] = 1
            print(f"rank {rank} {name}: not finite", file=sys.stderr)
        elif args.require_exact and not identical:
            failed[0] = 1
            print(f"rank {rank} {name}: not bitwise equal, relative RMS {error:.3e}",
                  file=sys.stderr)
        elif error >= args.max_relative_rms:
            failed[0] = 1
            print(f"rank {rank} {name}: relative RMS {error:.6f} exceeds "
                  f"{args.max_relative_rms}", file=sys.stderr)
        if rank == 0:
            print(f"{name:4s} relative RMS {error:.3e}  bitwise {identical}")

    # Reduce the verdict so every rank exits the same way. A rank asserting locally while the
    # others wait in the next collective would hang instead of failing.
    dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=group)
    dist.destroy_process_group()
    raise SystemExit(int(failed.item()))


if __name__ == "__main__":
    main()
