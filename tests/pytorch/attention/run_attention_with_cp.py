# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import copy
import json
import os
import statistics
import sys
import logging
from contextlib import nullcontext
import torch
import torch.distributed as dist
from transformer_engine.pytorch.attention.dot_product_attention.context_parallel import (
    get_cu_seqlens_on_cp_rank,
    get_a2a_kv_head_replication_factor,
)
from transformer_engine.pytorch.attention.dot_product_attention.utils import combine_and_quantize
import transformer_engine_torch as tex
from test_attention_with_cp import model_configs_flash_attn, model_configs_fused_attn
from transformer_engine.pytorch import (
    autocast,
    DotProductAttention,
    Float8Quantizer,
    Float8CurrentScalingQuantizer,
    MXFP8Quantizer,
)
from transformer_engine.common.recipe import (
    DelayedScaling,
    Float8CurrentScaling,
    MXFP8BlockScaling,
    Format,
)
from utils import ModelConfig
from utils import compare_and_assert as _compare_and_assert

# Pool mode (NVTE_CP_POOL_PG=1) only: shared CP collective groups, created once
# per pool by run_attention_with_cp_pool.main() and reused across every case in
# that pool. world_size and the rank set don't change per case, so re-creating
# these per call would be wasted NCCL setup (~50-100 ms each). Single-shot
# subprocess mode leaves these None / [] and run_dpa_with_cp creates/destroys
# its own groups inline.
_pool_cp_comm_group = None
_pool_cp_comm_sub_groups: list = []

dtypes = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp8": torch.bfloat16}


def _bench_cp_attention(
    core_attn, q_, k_, v_, dout_, bias_, config, fp8_context, is_training, fp8_bwd, fp8_mha,
    dout_quantizer, cu_seqlens_q, cu_seqlens_kv, cu_seqlens_q_padded, cu_seqlens_kv_padded,
    model, dtype, qkv_format, cp_comm_type, world_size, rank, bench_iters, bench_warmup, bench_out,
    bench_profile,
):
    """Time the CP forward/backward and write a JSON row (rank 0).

    Reuses the exact CP inputs and module built by run_dpa_with_cp, so the measured path is the
    validated one. fwd and bwd are timed separately with CUDA events; peak memory is measured after a
    reset that excludes warmup + setup. fwd/bwd/step are rank-0's GPU timeline (the a2a/p2p
    collectives synchronise ranks, so this includes communication wait); step_ms_slowest is the
    max median step across ranks, which is what actually gates a training step.

    When bench_profile > 0, a torch.profiler pass attributes per-iter device (kernel) time to
    attention / comm(nccl) / quantize / other categories. Note: kernels on different streams overlap
    (a2a on cp_stream vs attention on the main stream), so the category device times are the *work
    distribution*, and their sum can exceed the wall-clock step; prof_total_dev_ms vs step_ms shows
    the overlap. This is measured device work, not an assumed attribution.
    """
    common = dict(
        core_attention_bias_type=config.attn_bias_type,
        core_attention_bias=bias_,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        cu_seqlens_q_padded=cu_seqlens_q_padded,
        cu_seqlens_kv_padded=cu_seqlens_kv_padded,
        fp8_output=fp8_mha,
    )
    dout_bwd = dout_quantizer(dout_) if (is_training and fp8_bwd and fp8_mha) else dout_

    def _step(measure):
        for x in (q_, k_, v_):
            x.grad = None
        if config.softmax_type != "vanilla":
            core_attn.softmax_offset.grad = None
        if measure:
            f0, f1, b0, b1 = (torch.cuda.Event(enable_timing=True) for _ in range(4))
        with fp8_context:
            if measure:
                f0.record()
            out = core_attn(q_, k_, v_, **common)
            if config.return_max_logit:
                out = out[0]
            if measure:
                f1.record()
            if is_training:
                if measure:
                    b0.record()
                out.backward(dout_bwd)
                if measure:
                    b1.record()
        if measure:
            torch.cuda.synchronize()
            return f0.elapsed_time(f1), (b0.elapsed_time(b1) if is_training else 0.0)
        return None, None

    for _ in range(bench_warmup):
        _step(False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    fwd_ms, bwd_ms = [], []
    for _ in range(bench_iters):
        f, b = _step(True)
        fwd_ms.append(f)
        bwd_ms.append(b)
    peak_mib = torch.cuda.max_memory_allocated() / (1024**2)

    prof_cats = {}
    if bench_profile:
        import torch.profiler as _tp

        cats = {"attention": 0.0, "comm": 0.0, "quantize": 0.0, "other": 0.0}
        with _tp.profile(activities=[_tp.ProfilerActivity.CUDA]) as prof:
            for _ in range(bench_profile):
                _step(False)
            torch.cuda.synchronize()
        for evt in prof.key_averages():
            dev_us = float(evt.self_device_time_total)
            if dev_us <= 0:
                continue
            n = evt.key.lower()
            if any(s in n for s in ("fmha", "attn", "flash", "cudnn", "fused_attention")):
                cats["attention"] += dev_us
            elif any(s in n for s in ("nccl", "alltoall", "all_to_all", "sendrecv", "reduce")):
                cats["comm"] += dev_us
            elif any(s in n for s in ("quantize", "dequant", "amax", "fp8", "e4m3", "e5m2")):
                cats["quantize"] += dev_us
            else:
                cats["other"] += dev_us
        prof_cats = {f"prof_{k}_ms": round(v / 1000.0 / bench_profile, 4) for k, v in cats.items()}
        prof_cats["prof_total_dev_ms"] = round(sum(cats.values()) / 1000.0 / bench_profile, 4)

    step_med = statistics.median([a + b for a, b in zip(fwd_ms, bwd_ms)])
    slowest = torch.tensor([step_med], device="cuda")
    dist.all_reduce(slowest, op=dist.ReduceOp.MAX)

    replication = (
        get_a2a_kv_head_replication_factor(config.num_heads, config.num_gqa_groups, world_size)
        if cp_comm_type in ("a2a", "a2a+p2p")
        else 1
    )
    row = dict(
        model=model,
        dtype=dtype,
        qkv_format=qkv_format,
        cp_comm_type=cp_comm_type,
        cp_size=world_size,
        seqlen=config.max_seqlen_q,
        total_tokens=int(cu_seqlens_q_padded[-1]),
        num_heads=config.num_heads,
        num_gqa_groups=config.num_gqa_groups,
        kv_replication_factor=replication,
        fwd_ms=round(statistics.median(fwd_ms), 4),
        bwd_ms=round(statistics.median(bwd_ms), 4),
        step_ms=round(step_med, 4),
        step_ms_slowest=round(slowest.item(), 4),
        peak_mib=round(peak_mib, 1),
        iters=bench_iters,
        warmup=bench_warmup,
        **prof_cats,
    )
    if rank == 0 and bench_out:
        with open(bench_out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
    logging.info("[Rank %d] bench %s", rank, row)


def generate_input_shapes(
    qkv_format: str,
    config: ModelConfig,
    world_size: int,
    kernel_backend: str,
    fa_pad_between_seqs: str = "False",
    bench: bool = False,
):
    if qkv_format == "bshd":
        q_input_shape = (
            config.batch_size,
            config.max_seqlen_q,
            config.num_heads,
            config.head_dim_qk,
        )
        k_input_shape = (
            config.batch_size,
            config.max_seqlen_kv,
            config.num_gqa_groups,
            config.head_dim_qk,
        )
        v_input_shape = (
            config.batch_size,
            config.max_seqlen_kv,
            config.num_gqa_groups,
            config.head_dim_v,
        )
        attn_output_shape = (
            config.batch_size,
            config.max_seqlen_q,
            config.num_heads * config.head_dim_v,
        )
        cu_seqlens_q = None
        cu_seqlens_kv = None
        cu_seqlens_q_padded = None
        cu_seqlens_kv_padded = None
    elif qkv_format == "sbhd":
        q_input_shape = (
            config.max_seqlen_q,
            config.batch_size,
            config.num_heads,
            config.head_dim_qk,
        )
        k_input_shape = (
            config.max_seqlen_kv,
            config.batch_size,
            config.num_gqa_groups,
            config.head_dim_qk,
        )
        v_input_shape = (
            config.max_seqlen_kv,
            config.batch_size,
            config.num_gqa_groups,
            config.head_dim_v,
        )
        attn_output_shape = (
            config.max_seqlen_q,
            config.batch_size,
            config.num_heads * config.head_dim_v,
        )
        cu_seqlens_q = None
        cu_seqlens_kv = None
        cu_seqlens_q_padded = None
        cu_seqlens_kv_padded = None
    elif qkv_format == "thd":
        if bench:
            # Deterministic full-length documents so total tokens = batch * seqlen exactly. The
            # correctness default draws random per-doc lengths, which is right for varied test cases
            # but makes benchmark workload (and thus timing/memory) random per run.
            seqlens_q = torch.full(
                [config.batch_size], config.max_seqlen_q, dtype=torch.int32
            )
        else:
            seqlens_q = torch.randint(
                0, config.max_seqlen_q + 1, [config.batch_size]
            ).to(torch.int32)
        seqlens_q_padded = (seqlens_q + 2 * world_size - 1) // (world_size * 2) * (world_size * 2)
        cu_seqlens_q_padded = torch.cat(
            [
                torch.zeros([1], dtype=torch.int32),
                seqlens_q_padded.cumsum(0, dtype=torch.int32),
            ]
        ).cuda()
        cu_seqlens_q = torch.clone(cu_seqlens_q_padded)

        # Generate padded data (cu_seqlens_q reflects non-padded lengths, so it
        # differs from cu_seqlens_q_padded) for FusedAttention always, and for
        # FlashAttention only when its test param requests it. DPA auto-detects
        # pad_between_seqs downstream from the cu_seqlens_q vs cu_seqlens_q_padded
        # mismatch.
        if kernel_backend == "FusedAttention" or fa_pad_between_seqs == "True":
            cu_seqlens_q[1:] = seqlens_q.cumsum(0, dtype=torch.int32).cuda()

        # NOTE: In case of Cross-Attention, `cu_seqlens_kv` and `cu_seqlens_kv_padded`
        # will not be the same as `cu_seqlens_q` and `cu_seqlens_q_padded` respectively.
        cu_seqlens_kv = cu_seqlens_q
        cu_seqlens_kv_padded = cu_seqlens_q_padded

        total_tokens = cu_seqlens_q_padded[-1]

        q_input_shape = (
            total_tokens,
            config.num_heads,
            config.head_dim_qk,
        )
        k_input_shape = (
            total_tokens,
            config.num_gqa_groups,
            config.head_dim_qk,
        )
        v_input_shape = (
            total_tokens,
            config.num_gqa_groups,
            config.head_dim_v,
        )
        attn_output_shape = (
            total_tokens,
            config.num_heads * config.head_dim_v,
        )
    else:
        assert False, f"{qkv_format=} is not supported!"

    return (
        q_input_shape,
        k_input_shape,
        v_input_shape,
        attn_output_shape,
        cu_seqlens_q,
        cu_seqlens_kv,
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
    )


def get_tols(config, dtype):
    if dtype == "bf16":
        if config.num_heads == config.num_gqa_groups:
            atol = 2.5e-2
            rtol = 2.5e-2
        else:
            atol = 3.5e-2
            rtol = 3.5e-2
        rmse_tol = 0.01
    elif dtype == "fp16":
        atol = 5e-3
        rtol = 5e-3
        rmse_tol = 0.01
    elif dtype == "fp8":
        atol = 5e-1
        rtol = 5e-1
        rmse_tol = 0.15
    else:
        assert False, f"{dtype=} is not supported!"

    return atol, rtol, rmse_tol


def _reset_delayed_scaling_state(module):
    reset = 0
    fp8_meta = getattr(module, "fp8_meta", {}) or {}
    for name in ("scaling_fwd", "scaling_bwd"):
        state = fp8_meta.get(name)
        if state is None:
            continue
        for attr, value in (("scale", 1.0), ("amax_history", 0.0)):
            tensor = getattr(state, attr, None)
            if isinstance(tensor, torch.Tensor):
                tensor.fill_(value)
                reset += 1
    for group in (getattr(module, "quantizers", {}) or {}).values():
        for quantizer in group:
            for attr, value in (("scale", 1.0), ("amax", 0.0)):
                tensor = getattr(quantizer, attr, None)
                if isinstance(tensor, torch.Tensor):
                    tensor.fill_(value)
                    reset += 1
    return reset


def run_dpa_with_cp(
    dtype="bf16",
    model=None,
    qkv_format="bshd",
    kernel_backend="FlashAttention",
    cp_comm_type="p2p",
    fp8_bwd="True",
    fp8_dpa="False",
    fp8_mha="False",
    scaling_mode="delayed",
    f16_O="False",
    is_training="True",
    fa_pad_between_seqs="False",
    deterministic="False",
    log_level=logging.WARNING,
    bench_iters="0",
    bench_warmup="5",
    bench_out="",
    bench_seqlen="0",
    bench_batch="0",
    bench_profile="0",
):
    """Test DotProductAttention module with context parallelism.

    When bench_iters > 0, the reference (no-CP) run and the correctness comparison are skipped;
    instead the CP forward/backward is timed over bench_iters iterations (after bench_warmup) and a
    JSON row (timings + peak memory + replication factor) is written to bench_out by rank 0.
    bench_seqlen > 0 overrides the config's max_seqlen_q/kv (to sweep sequence length).
    """
    logging.root.setLevel(log_level)
    # When is_training is False, gradient outputs are None.
    is_training = is_training == "True"
    bench_iters = int(bench_iters)
    bench_warmup = int(bench_warmup)
    bench_seqlen = int(bench_seqlen)
    bench_batch = int(bench_batch)
    bench_profile = int(bench_profile)

    # set up environment variables and config
    if deterministic == "True":
        os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
    else:
        os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "1"
    fp8_bwd = fp8_bwd == "True" and dtype == "fp8"
    os.environ["NVTE_FP8_DPA_BWD"] = "1" if fp8_bwd else "0"
    fp8_dpa = fp8_dpa == "True" and dtype == "fp8"
    fp8_mha = fp8_mha == "True" and dtype == "fp8" and scaling_mode != "mxfp8"
    f16_O = dtype == "fp8" and scaling_mode in ["current", "mxfp8"] and f16_O == "True"
    os.environ["NVTE_DPA_FP8CS_O_in_F16"] = "1" if f16_O else "0"
    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    if kernel_backend == "FlashAttention":
        os.environ["NVTE_FLASH_ATTN"] = "1"
        # Deep-copy: the module-level dict is shared across pool cases; the
        # THD branch below rewrites attn_mask_type in place, which would
        # otherwise leak into subsequent cases reusing the same model key.
        config = copy.deepcopy(model_configs_flash_attn[model])
    if kernel_backend == "FusedAttention":
        os.environ["NVTE_FUSED_ATTN"] = "1"
        config = copy.deepcopy(model_configs_fused_attn[model])
    assert config.attn_mask_type in [
        "causal",
        "no_mask",
    ], f"{config.attn_mask_type=} is not supported!"
    if qkv_format == "thd":
        if "causal" in config.attn_mask_type:
            config.attn_mask_type = "padding_causal"
        else:
            config.attn_mask_type = "padding"
    if bench_seqlen:
        config.max_seqlen_q = bench_seqlen
        config.max_seqlen_kv = bench_seqlen
    if bench_batch:
        config.batch_size = bench_batch

    # set up distributed group
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    # When NVTE_CP_POOL_PG=1, the pool runner owns the lifecycle of the main
    # process group across many cases; here we only reuse it.
    _pool_managed_pg = os.getenv("NVTE_CP_POOL_PG", "0") == "1"
    if dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        device_count = torch.cuda.device_count()
        device = rank % device_count
        torch.cuda.set_device(device)
    logging.info(f"[Rank {rank}] Setup: world_size {world_size}")
    if not _pool_managed_pg:
        dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)

    # Set up communication group for CP. In pool mode, the pool worker has
    # already pre-created world-scoped and a2a+p2p sub-groups once and stashed
    # them in module-level pointers; we reuse those and the pool destroys them
    # at shutdown. In single-shot mode we create them per call and destroy in
    # the finally below.
    cp_comm_ranks = range(world_size)
    assert rank in cp_comm_ranks
    _reusing_pool_groups = _pool_managed_pg and _pool_cp_comm_group is not None
    cp_comm_group = None
    cp_comm_sub_groups: list = []
    if _reusing_pool_groups:
        cp_comm_group = _pool_cp_comm_group
        cp_comm_sub_groups = _pool_cp_comm_sub_groups if cp_comm_type == "a2a+p2p" else []
    else:
        cp_comm_group = dist.new_group(cp_comm_ranks, backend="nccl")
        if cp_comm_type == "a2a+p2p":
            assert world_size % 2 == 0, (
                "{cp_comm_type=} requires world_size % 2 = 0 as it assumes the a2a level has"
                " cp_size = 2."
            )
            cp_comm_sub_ranks = [range(i * 2, (i + 1) * 2) for i in range(world_size // 2)]
            cp_comm_sub_ranks += [range(i, world_size, 2) for i in range(2)]
            for sub_ranks in cp_comm_sub_ranks:
                sub_group = dist.new_group(sub_ranks, backend="nccl")
                if rank in sub_ranks:
                    cp_comm_sub_groups.append(sub_group)
    if dtype == "fp8":
        if scaling_mode == "delayed":
            fp8_recipe = DelayedScaling(fp8_dpa=fp8_dpa, fp8_mha=fp8_mha)
        if scaling_mode == "current":
            fp8_recipe = Float8CurrentScaling(fp8_dpa=fp8_dpa, fp8_mha=fp8_mha)
        if scaling_mode == "mxfp8":
            fp8_recipe = MXFP8BlockScaling(fp8_format=Format.E4M3, fp8_dpa=fp8_dpa, fp8_mha=fp8_mha)

    # instantiate attention module
    core_attn = DotProductAttention(
        config.num_heads,
        (config.head_dim_qk, config.head_dim_v),
        num_gqa_groups=config.num_gqa_groups,
        attention_dropout=config.dropout_p,
        qkv_format=qkv_format,
        attn_mask_type=config.attn_mask_type,
        window_size=config.window_size,
        softmax_type=config.softmax_type,
        return_max_logit=config.return_max_logit,
    ).cuda()
    if not is_training:
        core_attn.eval()
    if is_training and config.softmax_type != "vanilla":
        core_attn.softmax_offset.requires_grad = True

    # generate attention inputs. Seed deterministically so the CP-vs-no-CP comparison is reproducible
    # and independent of prior RNG state (e.g. earlier cases in the same pool worker).
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    (
        q_input_shape,
        k_input_shape,
        v_input_shape,
        attn_output_shape,
        cu_seqlens_q,
        cu_seqlens_kv,
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
    ) = generate_input_shapes(
        qkv_format, config, world_size, kernel_backend, fa_pad_between_seqs, bench=bool(bench_iters)
    )
    q_orig = torch.clamp(torch.randn(q_input_shape, dtype=dtypes[dtype]), min=-1, max=1).cuda()
    k_orig = torch.clamp(torch.randn(k_input_shape, dtype=dtypes[dtype]), min=-1, max=1).cuda()
    v_orig = torch.clamp(torch.randn(v_input_shape, dtype=dtypes[dtype]), min=-1, max=1).cuda()
    dout_orig = torch.clamp(
        torch.randn(attn_output_shape, dtype=dtypes[dtype]), min=-1, max=1
    ).cuda()
    if scaling_mode == "delayed":
        qkv_quantizer = Float8Quantizer(
            fp8_dtype=tex.DType.kFloat8E4M3,
            scale=torch.tensor([1], dtype=torch.float32).cuda(),
            amax=torch.tensor([0], dtype=torch.float32).cuda(),
        )
        dout_quantizer = Float8Quantizer(
            fp8_dtype=tex.DType.kFloat8E5M2,
            scale=torch.tensor([1], dtype=torch.float32).cuda(),
            amax=torch.tensor([0], dtype=torch.float32).cuda(),
        )
    if scaling_mode == "current":
        qkv_quantizer = Float8CurrentScalingQuantizer(
            fp8_dtype=tex.DType.kFloat8E4M3,
            device="cuda",
        )
        dout_quantizer = Float8CurrentScalingQuantizer(
            fp8_dtype=tex.DType.kFloat8E5M2,
            device="cuda",
        )
    if scaling_mode == "mxfp8":
        qkv_quantizer = MXFP8Quantizer(
            fp8_dtype=tex.DType.kFloat8E4M3,
            rowwise=True,
            columnwise=True,
        )
        qkv_quantizer.optimize_for_gemm = True
        qkv_quantizer.internal = False
        dout_quantizer = MXFP8Quantizer(
            fp8_dtype=tex.DType.kFloat8E5M2,
            rowwise=True,
            columnwise=True,
        )
        dout_quantizer.optimize_for_gemm = True
        dout_quantizer.internal = False
    qkv_layout = "_".join([qkv_format] * 3)
    q, k, v, dout = [x.clone().detach() for x in [q_orig, k_orig, v_orig, dout_orig]]
    if fp8_mha:
        q, k, v, qkv_layout, _ = combine_and_quantize(qkv_layout, q, k, v, qkv_quantizer)
    for x in [q, k, v]:
        x.requires_grad = True

    if config.attn_bias_type not in ["no_bias", "alibi"]:
        bias_shape_map = {
            "1hss": (1, config.num_heads, config.max_seqlen_q, config.max_seqlen_kv),
            "11ss": (1, 1, config.max_seqlen_q, config.max_seqlen_kv),
            "b1ss": (config.batch_size, 1, config.max_seqlen_q, config.max_seqlen_kv),
            "bhss": (
                config.batch_size,
                config.num_heads,
                config.max_seqlen_q,
                config.max_seqlen_kv,
            ),
            "111s": (1, 1, 1, config.max_seqlen_kv),
        }
        attn_bias_shape = bias_shape_map.get(config.bias_shape)
        if attn_bias_shape is None:
            assert False, f"cuDNN does not support {config.bias_shape=}"
        bias = torch.randn(*attn_bias_shape, dtype=dtypes[dtype]).cuda()
        # cuDNN does not support dbias calculation for 111s as of cuDNN 9.18
        # TODO(KshitijLakhani): Set requires_grad to True for all shapes once 111s is supported
        bias.requires_grad = True if config.bias_shape != "111s" else False
    else:
        bias = None

    ############ run without CP ############
    # Skipped under benchmarking: there is no comparison, and running it would leave the no-CP
    # tensors resident, inflating the CP peak-memory measurement.
    dq = dk = dv = dbias = d_softmax_offset = max_logit = out = None
    if not bench_iters:
        logging.info(f"[Rank {rank}] Run without context parallelism")
        if dtype == "fp8":
            if not bench_iters and scaling_mode == "delayed":
                # Match the delayed-scaling state the CP arm is reset to, so the reference and CP runs
                # are compared at the same FP8 scale regardless of prior state in this (pool) process.
                _reset_delayed_scaling_state(core_attn)
            fp8_context = autocast(
                enabled=True, recipe=fp8_recipe, amax_reduction_group=cp_comm_group
            )
        else:
            fp8_context = nullcontext()
        with fp8_context:
            # q, k, v, out in FP8; dout in F16
            out = core_attn(
                q,
                k,
                v,
                core_attention_bias_type=config.attn_bias_type,
                core_attention_bias=bias,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                cu_seqlens_q_padded=cu_seqlens_q_padded,
                cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                fp8_output=fp8_mha,
            )
            if config.return_max_logit:
                out, max_logit = out
            if is_training:
                if fp8_bwd and fp8_mha:
                    dout_fp8 = dout_quantizer(dout)
                    out.backward(dout_fp8)
                else:
                    out.backward(dout)
        if is_training:
            dq, dk, dv, dbias = q.grad, k.grad, v.grad, bias.grad if bias is not None else None
            d_softmax_offset = (
                core_attn.softmax_offset.grad if config.softmax_type != "vanilla" else None
            )

    ############ run with CP ############
    logging.info(f"[Rank {rank}] Run with context parallelism")

    # set up inputs
    q_, k_, v_, dout_, *rest = [
        x.clone().detach()
        for x in [q_orig, k_orig, v_orig, dout_orig] + ([] if bias is None else [bias])
    ]
    bias_ = rest[0] if len(rest) else None
    if qkv_format == "bshd" or qkv_format == "sbhd":
        seq_dim = qkv_format.index("s")
        q_, k_, v_, dout_ = [
            x.view(
                *x.shape[:seq_dim],
                2 * world_size,
                x.shape[seq_dim] // (2 * world_size),
                *x.shape[(seq_dim + 1) :],
            )
            for x in [q_, k_, v_, dout_]
        ]
        seq_idx = torch.tensor([rank, 2 * world_size - rank - 1], device=q_.device)
        q_, k_, v_, dout_ = [x.index_select(seq_dim, seq_idx) for x in [q_, k_, v_, dout_]]
        q_, k_, v_, dout_ = [
            x.view(*x.shape[:seq_dim], -1, *x.shape[(seq_dim + 2) :]) for x in [q_, k_, v_, dout_]
        ]
    elif qkv_format == "thd":
        seq_idx_q = tex.thd_get_partitioned_indices(
            cu_seqlens_q_padded, q_.shape[0], world_size, rank
        )
        seq_idx_kv = tex.thd_get_partitioned_indices(
            cu_seqlens_kv_padded, k_.shape[0], world_size, rank
        )
        q_, dout_ = [x.index_select(0, seq_idx_q) for x in [q_, dout_]]
        k_, v_ = [x.index_select(0, seq_idx_kv) for x in [k_, v_]]
    else:
        assert False, f"{qkv_format} is an unsupported qkv_format!"
    q_, k_, v_, dout_ = [x.contiguous() for x in [q_, k_, v_, dout_]]
    if scaling_mode == "delayed":
        qkv_quantizer.scale.fill_(1.0)
        qkv_quantizer.amax.fill_(0.0)
        dout_quantizer.scale.fill_(1.0)
        dout_quantizer.amax.fill_(0.0)
    if fp8_mha:
        q_, k_, v_, qkv_layout, _ = combine_and_quantize(qkv_layout, q_, k_, v_, qkv_quantizer)
    if is_training:
        q_, k_, v_ = [x.requires_grad_() for x in [q_, k_, v_]]
    if bias_ is not None:
        ndim = bias_.ndim
        seq_q_dim = ndim - 2
        if qkv_format == "thd":
            bias_seq_idx = seq_idx_q
        else:
            bias_seq_idx = seq_idx
        shape_before_seq = bias_.shape[:seq_q_dim]
        seq_q_size = bias_.shape[seq_q_dim]
        seq_kv_size = bias_.shape[-1]
        if seq_q_size == 1:
            # TODO(KshitijLakhani): Set to True always once cuDNN supports dbias for 111s
            bias_.requires_grad = False
            # Bias is broadcast, no need to partition along sequence dimension
            pass
        else:
            bias_ = bias_.view(
                *shape_before_seq, 2 * world_size, seq_q_size // (2 * world_size), seq_kv_size
            )
            bias_ = bias_.index_select(seq_q_dim, bias_seq_idx)
            bias_ = bias_.view(*shape_before_seq, -1, seq_kv_size)
            bias_.requires_grad = True
    # set up environment
    core_attn.set_context_parallel_group(
        cp_comm_sub_groups if cp_comm_type == "a2a+p2p" else cp_comm_group,
        cp_comm_ranks,
        torch.cuda.Stream(),
        cp_comm_type,
    )
    if config.softmax_type != "vanilla":
        core_attn.softmax_offset.grad.zero_()
    if dtype == "fp8":
        core_attn.fp8_initialized = False
        core_attn.fp8_meta_tensors_initialized = False
        if not bench_iters and scaling_mode == "delayed":
            assert _reset_delayed_scaling_state(core_attn) > 0, (
                "no delayed FP8 scaling state was reset before the CP run; the reference and CP runs "
                "would be compared at different scales"
            )
        fp8_context = autocast(enabled=True, recipe=fp8_recipe, amax_reduction_group=cp_comm_group)
    else:
        fp8_context = nullcontext()

    if bench_iters:
        _bench_cp_attention(
            core_attn, q_, k_, v_, dout_, bias_, config, fp8_context, is_training,
            fp8_bwd, fp8_mha, dout_quantizer if dtype == "fp8" else None,
            cu_seqlens_q, cu_seqlens_kv, cu_seqlens_q_padded, cu_seqlens_kv_padded,
            model, dtype, qkv_format, cp_comm_type, world_size, rank, bench_iters, bench_warmup,
            bench_out, bench_profile,
        )
        if not _pool_managed_pg:
            dist.destroy_process_group()
        return

    # run attention
    max_logit_ = None
    with fp8_context:
        # q, k, v, out in FP8; dout in F16
        out_ = core_attn(
            q_,
            k_,
            v_,
            core_attention_bias_type=config.attn_bias_type,
            core_attention_bias=bias_,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            cu_seqlens_q_padded=cu_seqlens_q_padded,
            cu_seqlens_kv_padded=cu_seqlens_kv_padded,
            fp8_output=fp8_mha,
        )
        if config.return_max_logit:
            out_, max_logit_ = out_
        if is_training:
            if fp8_bwd and fp8_mha:
                dout_fp8_ = dout_quantizer(dout_)
                out_.backward(dout_fp8_)
            else:
                out_.backward(dout_)
    if is_training:
        dq_, dk_, dv_, dbias_ = (
            q_.grad,
            k_.grad,
            v_.grad,
            bias_.grad if bias_ is not None else None,
        )
        d_softmax_offset_ = (
            core_attn.softmax_offset.grad.clone() if config.softmax_type != "vanilla" else None
        )
    else:
        dq_, dk_, dv_, dbias_ = None, None, None, None
        d_softmax_offset_ = None

    # get outputs
    tensors = [out, dq, dk, dv, dbias, out_, dq_, dk_, dv_, dbias_]
    names = ["out", "dq", "dk", "dv", "dbias", "out_cp", "dq_cp", "dk_cp", "dv_cp", "dbias_cp"]
    if fp8_mha:
        tensors_to_deq = [out, out_] if not fp8_bwd else tensors
        for i, tensor in enumerate(tensors_to_deq):
            # dbias/dbias_ could be None, so skip check for it
            if tensor is not None:
                tensors_to_deq[i] = tensor.dequantize()
        if not fp8_bwd:
            tensors[0], tensors[5] = tensors_to_deq
    for tensor, name in zip(tensors, names):
        # dbias/dbias_ could be None, so skip check for it
        if tensor is not None:
            assert torch.all(~torch.isnan(tensor)), f"{name} has nan values"
            assert torch.all(~torch.isinf(tensor)), f"{name} has inf values"
    out, dq, dk, dv, dbias, out_, dq_, dk_, dv_, dbias_ = tensors

    ############  compare results between CP and no-CP ############
    if qkv_format == "bshd" or qkv_format == "sbhd":
        if is_training:
            dq, dk, dv, out = [
                x.view(
                    *x.shape[:seq_dim],
                    2 * world_size,
                    x.shape[seq_dim] // (2 * world_size),
                    *x.shape[(seq_dim + 1) :],
                )
                for x in [dq, dk, dv, out]
            ]
            dq, dk, dv, out = [x.index_select(seq_dim, seq_idx) for x in [dq, dk, dv, out]]
            dq_, dk_, dv_, out_ = [
                x.view(*x.shape[:seq_dim], 2, x.shape[seq_dim] // 2, *x.shape[(seq_dim + 1) :])
                for x in [dq_, dk_, dv_, out_]
            ]
            if dbias is not None and dbias_ is not None:
                ndim = dbias.ndim
                # Query seq is at dim -2
                seq_q_dim = ndim - 2
                shape_before_seq = dbias.shape[:seq_q_dim]
                seq_q_size = dbias.shape[seq_q_dim]
                seq_kv_size = dbias.shape[-1]
                # Reshape to split seq_q dimension
                dbias = dbias.view(
                    *shape_before_seq,
                    2 * world_size,
                    seq_q_size // (2 * world_size),
                    seq_kv_size,
                )
                # Index select on the newly created dimension (now at position seq_q_dim)
                dbias = dbias.index_select(seq_q_dim, seq_idx)
                dbias_ = dbias_.view(
                    *shape_before_seq, 2, dbias_.shape[seq_q_dim] // 2, seq_kv_size
                )
        else:
            # Forward-only: reshape only out/out_ for comparison
            out = out.view(
                *out.shape[:seq_dim],
                2 * world_size,
                out.shape[seq_dim] // (2 * world_size),
                *out.shape[(seq_dim + 1) :],
            )
            out = out.index_select(seq_dim, seq_idx)
            out_ = out_.view(
                *out_.shape[:seq_dim], 2, out_.shape[seq_dim] // 2, *out_.shape[(seq_dim + 1) :]
            )

    elif qkv_format == "thd":
        if is_training:
            dq, out = [x.index_select(0, seq_idx_q).contiguous() for x in [dq, out]]
            dk, dv = [x.index_select(0, seq_idx_kv).contiguous() for x in [dk, dv]]
            cu_seqlens_q_padded = cu_seqlens_q_padded // world_size
            cu_seqlens_q = get_cu_seqlens_on_cp_rank(
                cu_seqlens_q, cu_seqlens_q_padded, world_size, rank, True, True
            )
            num_pads_q = (cu_seqlens_q_padded - cu_seqlens_q)[1:] - (
                cu_seqlens_q_padded - cu_seqlens_q
            )[:-1]
            cu_seqlens_kv_padded = cu_seqlens_kv_padded // world_size
            cu_seqlens_kv = get_cu_seqlens_on_cp_rank(
                cu_seqlens_kv, cu_seqlens_kv_padded, world_size, rank, True, True
            )
            num_pads_kv = (cu_seqlens_kv_padded - cu_seqlens_kv)[1:] - (
                cu_seqlens_kv_padded - cu_seqlens_kv
            )[:-1]
            # FA3 leaves garbage at padding positions despite seqused_q/k (tile spillover).
            # Forward out_ can't be pre-zeroed because FA3's custom op returns out_ as an
            # output rather than mutating it in-place, triggering PyTorch's aliasing constraint.
            # Backward dq/dk/dv CAN be pre-zeroed because FA3 marks them as mutated inputs.
            if fa_pad_between_seqs == "True":
                # out_ is a view inside the CP custom autograd Function, so in-place
                # zeroing is blocked by PyTorch. Clone to break the view relationship.
                out_ = out_.clone()
                for x in [out, out_, dq]:
                    for b in range(config.batch_size):
                        x[
                            cu_seqlens_q_padded[b + 1] - num_pads_q[b] : cu_seqlens_q_padded[b + 1]
                        ] = 0.0
                    x[cu_seqlens_q_padded[-1] :] = 0.0
                for x in [dk, dv]:
                    for b in range(config.batch_size):
                        x[
                            cu_seqlens_kv_padded[b + 1]
                            - num_pads_kv[b] : cu_seqlens_kv_padded[b + 1]
                        ] = 0.0
                    x[cu_seqlens_kv_padded[-1] :] = 0.0
                # Verify CP backward tensors have clean padding (pre-zeroed in context_parallel.py).
                for xname, x, cu, np_ in [
                    ("dq_", dq_, cu_seqlens_q_padded, num_pads_q),
                    ("dk_", dk_, cu_seqlens_kv_padded, num_pads_kv),
                    ("dv_", dv_, cu_seqlens_kv_padded, num_pads_kv),
                ]:
                    nnz = torch.count_nonzero(x[cu[-1] :]).item()
                    assert nnz == 0, (
                        f"{xname} has {nnz} nonzero values in tail padding — "
                        "context_parallel.py should zero padding positions"
                    )
                    for b in range(config.batch_size):
                        if np_[b] > 0:
                            nnz = torch.count_nonzero(x[cu[b + 1] - np_[b] : cu[b + 1]]).item()
                            assert nnz == 0, (
                                f"{xname} has {nnz} nonzero values in batch {b} padding — "
                                "context_parallel.py should zero padding positions"
                            )
        else:
            out = out.index_select(0, seq_idx_q).contiguous()
            out_ = out_

    atol, rtol, rmse_tol = get_tols(config, dtype)
    tensors_cp = [out_, dq_, dk_, dv_, dbias_, d_softmax_offset_, max_logit_]
    tensors_no_cp = [out, dq, dk, dv, dbias, d_softmax_offset, max_logit]
    names = ["out", "dq", "dk", "dv", "dbias", "d_softmax_offset", "max_logit"]
    names_cp = [x + "_cp" for x in names]
    names_no_cp = [x + "_no_cp" for x in names]
    is_fp8 = dtype == "fp8"

    # F3: the FP8 magnitude gate (utils.compare_and_assert) is enabled for the THD CP paths whose
    # backward can silently mis-scale a gradient: a2a (a dropped dK/dV replica-sum) and p2p (a wrong
    # THD half placement in the delayed-FP8 ring). For every other comm type / layout / scaling it
    # over-fires on legitimate noise (e.g. current-scaling sbhd dV, ratio dev ~0.358 > 0.35), so it
    # stays off there. This local wrapper bakes the scope in so the comparison call sites are unchanged.
    _check_magnitude = is_fp8 and cp_comm_type in ("a2a", "p2p") and qkv_format == "thd"

    def compare_and_assert(a, b, name_a, name_b, atol, rtol, rmse_tol, is_fp8):
        _compare_and_assert(
            a, b, name_a, name_b, atol, rtol, rmse_tol, is_fp8, check_magnitude=_check_magnitude
        )

    for i, t in enumerate(tensors_no_cp):
        if t is not None:
            if "softmax_offset" not in names[i] and "max_logit" not in names[i]:
                if qkv_format == "bshd":
                    # Compare the two sequence chunks separately
                    # Compare dbias
                    if names[i] == "dbias":
                        # Compare the two chunks along dimension 2 (the split sequence dimension)
                        seq_q_dim_bias = 2
                        ndim_bias = t.ndim
                        slice_0 = [slice(None)] * ndim_bias
                        slice_0[seq_q_dim_bias] = 0
                        slice_1 = [slice(None)] * ndim_bias
                        slice_1[seq_q_dim_bias] = 1
                        compare_and_assert(
                            t[tuple(slice_0)],
                            tensors_cp[i][tuple(slice_0)],
                            names_no_cp[i],
                            names_cp[i],
                            atol,
                            rtol,
                            rmse_tol,
                            is_fp8,
                        )
                        compare_and_assert(
                            t[tuple(slice_1)],
                            tensors_cp[i][tuple(slice_1)],
                            names_no_cp[i],
                            names_cp[i],
                            atol,
                            rtol,
                            rmse_tol,
                            is_fp8,
                        )
                    # Compare Q/K/V/out
                    else:
                        #  Compare the two chunks along dimension 1 (the split sequence dimension)
                        compare_and_assert(
                            t[:, 0],
                            tensors_cp[i][:, 0],
                            names_no_cp[i],
                            names_cp[i],
                            atol,
                            rtol,
                            rmse_tol,
                            is_fp8,
                        )
                        compare_and_assert(
                            t[:, 1],
                            tensors_cp[i][:, 1],
                            names_no_cp[i],
                            names_cp[i],
                            atol,
                            rtol,
                            rmse_tol,
                            is_fp8,
                        )
                elif qkv_format == "sbhd":
                    # Compare the two sequence chunks separately
                    # Compare dbias (same as BSHD)
                    if names[i] == "dbias":
                        # Same as bshd: Compare the two chunks along dimension 2 (the split sequence dimension)
                        seq_q_dim_bias = 2
                        ndim_bias = t.ndim
                        slice_0 = [slice(None)] * ndim_bias
                        slice_0[seq_q_dim_bias] = 0
                        slice_1 = [slice(None)] * ndim_bias
                        slice_1[seq_q_dim_bias] = 1
                        compare_and_assert(
                            t[tuple(slice_0)],
                            tensors_cp[i][tuple(slice_0)],
                            names_no_cp[i],
                            names_cp[i],
                            atol,
                            rtol,
                            rmse_tol,
                            is_fp8,
                        )
                        compare_and_assert(
                            t[tuple(slice_1)],
                            tensors_cp[i][tuple(slice_1)],
                            names_no_cp[i],
                            names_cp[i],
                            atol,
                            rtol,
                            rmse_tol,
                            is_fp8,
                        )
                    # Compare Q/K/V/out
                    else:
                        #  Compare the two chunks along dimension 0 (the split sequence dimension)
                        compare_and_assert(
                            t[0],
                            tensors_cp[i][0],
                            names_no_cp[i],
                            names_cp[i],
                            atol,
                            rtol,
                            rmse_tol,
                            is_fp8,
                        )
                        compare_and_assert(
                            t[1],
                            tensors_cp[i][1],
                            names_no_cp[i],
                            names_cp[i],
                            atol,
                            rtol,
                            rmse_tol,
                            is_fp8,
                        )
                elif qkv_format == "thd":
                    compare_and_assert(
                        t,
                        tensors_cp[i],
                        names_no_cp[i],
                        names_cp[i],
                        atol,
                        rtol,
                        rmse_tol,
                        is_fp8,
                    )
            else:
                compare_and_assert(
                    t, tensors_cp[i], names_no_cp[i], names_cp[i], atol, rtol, rmse_tol, is_fp8
                )
            logging.info(f"[Rank {rank}] CP vs no-CP: {names[i]} matches")

    # Teardown on the success path. Pool mode: cp_comm_group / cp_comm_sub_groups
    # point at pool-shared groups owned by the pool runner (which destroys them
    # at pool shutdown), and the main PG is also pool-owned — both branches
    # below are no-ops. Single-shot mode: destroy what we created here. If the
    # body above raises, we skip this — the subprocess dies at function return
    # and NCCL releases the communicators with the process.
    if not _reusing_pool_groups:
        if cp_comm_group is not None:
            try:
                dist.destroy_process_group(cp_comm_group)
            except Exception:
                pass
        for g in cp_comm_sub_groups:
            try:
                dist.destroy_process_group(g)
            except Exception:
                pass
    if not _pool_managed_pg:
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def main(**kwargs):
    run_dpa_with_cp(**kwargs)


if __name__ == "__main__":
    kwargs = dict(arg.split("=") for arg in sys.argv[2:])
    main(**kwargs)
