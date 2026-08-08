# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import json
import os
import select
import signal
import subprocess
import sys
import threading
import time
import pathlib
import logging
import copy
from collections import deque
import pytest
import torch
from transformer_engine.pytorch import (
    get_device_compute_capability,
    get_cudnn_version,
)
from transformer_engine.common.recipe import (
    DelayedScaling,
    Float8CurrentScaling,
    MXFP8BlockScaling,
    Format,
)
from transformer_engine.pytorch.attention.dot_product_attention.utils import (
    FlashAttentionUtils,
)

_current_file = pathlib.Path(__file__).resolve()
sys.path.append(str(_current_file.parent.parent))
from utils import ModelConfig, get_available_attention_backends

pytest_logging_level = logging.getLevelName(logging.root.level)

# Get determinism
_deterministic = (
    not bool(int(os.getenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1")))
    or torch.are_deterministic_algorithms_enabled()
)

# Initialize RNG state
seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

test_essential = True

model_configs_flash_attn = {
    # test: ModelConfig(b, sq, hq, dqk)
    "cp_1_0": ModelConfig(2, 4096, 12, 128, attn_mask_type="causal"),  # MHA
    "cp_1_1": ModelConfig(2, 4096, 12, 128),  # MHA
    "cp_1_2": ModelConfig(
        2, 4096, 12, 128, attn_mask_type="causal", window_size=(512, 0)
    ),  # MHA
    "cp_1_3": ModelConfig(2, 4096, 12, 128, window_size=(512, 512)),  # MHA
    "cp_2_0": ModelConfig(
        2, 4096, 32, 128, num_gqa_groups=4, attn_mask_type="causal"
    ),  # GQA
    "cp_2_1": ModelConfig(2, 4096, 12, 128, num_gqa_groups=2),  # GQA
    "cp_2_2": ModelConfig(
        2, 4096, 32, 128, attn_mask_type="causal", window_size=(128, 0)
    ),  # GQA
    "cp_2_3": ModelConfig(
        2, 4096, 12, 128, num_gqa_groups=2, window_size=(512, 512)
    ),  # GQA
    "cp_3_0": ModelConfig(
        2, 4096, 128, 192, attn_mask_type="causal", head_dim_v=128
    ),  # MLA
    "cp_3_1": ModelConfig(2, 4096, 12, 192, head_dim_v=128),  # MLA
    "cp_3_2": ModelConfig(
        2, 4096, 12, 192, attn_mask_type="causal", window_size=(512, 0), head_dim_v=128
    ),  # MLA
    "cp_3_3": ModelConfig(
        2, 4096, 12, 192, window_size=(512, 512), head_dim_v=128
    ),  # MLA
}


# --- Persistent pool runner -----------------------------------------------
#
# Each (world_size) is served by one long-lived torchrun running
# run_attention_with_cp_pool.py. We submit one work item per pytest case over
# rank-0 stdin and read one JSON response from rank-0 stdout. Replaces
# the per-case torchrun launch path; init/destroy NCCL once per pool, not
# once per case.
#
# Why two pool sizes: cp_comm_type="a2a+p2p" needs world_size=4; everything
# else uses world_size=2. We can't resize an active PG, so we keep one pool
# per world_size and route each case to the right one. Pools are spawned
# lazily on first use so a session that only exercises 2-GPU cases never
# pays the 4-GPU init cost.

# Per-case wall is ~5 s p50 / ~15 s max on H100 (test_essential=True).
# 90 s gives ~6× headroom over the slowest observed case while still detecting
# a genuine hang within ~1.5 min instead of ~10 min. Override with the env var
# if a slower machine or expanded test matrix needs more room.
POOL_SUBMIT_TIMEOUT_SEC = float(os.getenv("NVTE_CP_POOL_TIMEOUT_SEC", "90"))


class PoolWorker:
    # Crash-path AssertionErrors include the tail of the worker's stderr so CI
    # JUnit XML shows the actual failure cause (NCCL/CUDA messages, Python
    # traceback) inline with the failing test, not just "pool worker died".
    # Equivalent in spirit to PR #2965's run_distributed() stderr capture.
    _STDERR_BUFFER_LINES = 200  # ring cap (~40 KB ceiling)
    _STDERR_TAIL_CHARS = 4000  # how much to attach to the AssertionError

    def __init__(self, world_size: int):
        self.world_size = world_size
        self.proc: subprocess.Popen | None = None
        self._stderr_buf: deque[str] = deque(maxlen=self._STDERR_BUFFER_LINES)

    def _spawn(self) -> None:
        te_path = os.getenv("TE_PATH", "/opt/transformerengine")
        worker = os.path.join(
            te_path, "tests/pytorch/attention/run_attention_with_cp_pool.py"
        )
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc-per-node={self.world_size}",
            "--standalone",  # picks a free rendezvous port
            worker,
        ]
        # stderr=PIPE so we can capture the tail for crash-path AssertionErrors;
        # a daemon drainer thread also echoes each line to sys.stderr so pytest's
        # per-test stderr capture still works. The thread is daemon, so it
        # self-terminates when the pipe closes — no tracking needed.
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            # Own process group so _kill can killpg all ranks in one shot;
            # without this, terminating the launcher PID leaves rank workers
            # as orphans holding CUDA/NCCL state.
            start_new_session=True,
        )
        self._stderr_buf.clear()
        threading.Thread(target=self._drain_stderr, daemon=True).start()

    def _drain_stderr(self) -> None:
        proc = self.proc
        if proc is None or proc.stderr is None:
            return
        for line in iter(proc.stderr.readline, ""):
            self._stderr_buf.append(line)
            sys.stderr.write(line)
            sys.stderr.flush()

    def _diag(self, msg: str) -> str:
        tail = "".join(self._stderr_buf)[-self._STDERR_TAIL_CHARS :]
        if not tail.strip():
            return msg
        return f"{msg}\n\n--- pool worker stderr (tail) ---\n{tail}"

    def _ensure_alive(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            self._spawn()

    def _killpg(self, sig: int) -> None:
        try:
            os.killpg(self.proc.pid, sig)
        except ProcessLookupError:
            pass

    def _kill(self) -> None:
        # Kill the whole process group so rank workers don't survive as orphans.
        if self.proc and self.proc.poll() is None:
            self._killpg(signal.SIGTERM)
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._killpg(signal.SIGKILL)
                self.proc.wait()
        self.proc = None

    # One retry on pool-infrastructure failures (worker died / timed out / broken
    # pipe). Test-assertion failures from the worker carry the full per-rank
    # traceback in resp["error"] and propagate without retry. Every retry leaves
    # a [POOL-RETRY] line in stderr so pytest's <system-err> capture surfaces
    # flake patterns in JUnit XML for offline analysis.
    _MAX_RETRIES = 1

    def submit(self, kwargs: dict, timeout: float = POOL_SUBMIT_TIMEOUT_SEC) -> None:
        first_err = None
        for attempt in range(self._MAX_RETRIES + 1):
            try:
                return self._submit_once(kwargs, timeout)
            except AssertionError as e:
                msg_head = str(e).splitlines()[0]
                infrastructure_flake = (
                    "pool worker died" in msg_head
                    or "timed out" in msg_head
                    or "before request could be sent" in msg_head
                )
                if not infrastructure_flake or attempt == self._MAX_RETRIES:
                    if first_err is not None:
                        sys.stderr.write(
                            f"[POOL-RETRY-FAIL] world_size={self.world_size}: "
                            "both attempts died; first error was: "
                            f"{str(first_err).splitlines()[0]!r}\n"
                        )
                        sys.stderr.flush()
                    raise
                first_err = e
                sys.stderr.write(
                    f"[POOL-RETRY] world_size={self.world_size} attempt {attempt + 1} "
                    f"died: {msg_head!r}; respawning pool and retrying\n"
                )
                sys.stderr.flush()
        raise first_err  # unreachable; loop either returns or raises

    def _submit_once(self, kwargs: dict, timeout: float) -> None:
        self._ensure_alive()
        req = json.dumps({"op": "run", "kwargs": kwargs}) + "\n"
        try:
            self.proc.stdin.write(req)
            self.proc.stdin.flush()
        except BrokenPipeError:
            msg = self._diag("pool worker died before request could be sent")
            self._kill()
            raise AssertionError(msg)

        # Worker redirects non-rank-0 stdout to /dev/null at fd level, so
        # rank 0's JSON line is the only thing that arrives on this pipe.
        # select() on a pipe fd is Linux/macOS only — on Windows the select
        # module only accepts sockets. CP attention tests run on Linux GPU
        # hosts so this is fine; flag if portability is ever needed.
        ready, _, _ = select.select([self.proc.stdout], [], [], timeout)
        if not ready:
            msg = self._diag(
                f"pool worker (world_size={self.world_size}) timed out after "
                f"{timeout}s; pool killed and will be respawned for the next case"
            )
            self._kill()
            raise AssertionError(msg)

        line = self.proc.stdout.readline()
        if not line:
            msg = self._diag("pool worker died mid-request")
            self._kill()
            raise AssertionError(msg)

        # A stray non-JSON line from rank 0 would desynchronize the protocol;
        # turn it into a clear test failure rather than a raw JSONDecodeError.
        try:
            resp = json.loads(line)
        except json.JSONDecodeError as e:
            self._kill()
            raise AssertionError(
                self._diag(f"pool worker JSON protocol broke: {e!r}; line={line!r}")
            )

        if not resp["ok"]:
            # Discard the pool so half-aborted CUDA/NCCL/FP8 state from the
            # failed case doesn't leak into the next. resp["error"] already
            # carries the per-rank traceback via gather_object.
            self._kill()
            raise AssertionError(resp["error"])

    def shutdown(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.write(json.dumps({"op": "shutdown"}) + "\n")
                self.proc.stdin.flush()
                self.proc.stdin.close()
            except BrokenPipeError:
                pass
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._kill()
        self.proc = None


@pytest.fixture(scope="session")
def cp_pool():
    """Returns a callable: cp_pool(world_size) -> PoolWorker."""
    pools: dict[int, PoolWorker] = {}

    def _get(world_size: int) -> PoolWorker:
        if world_size > torch.cuda.device_count():
            pytest.skip(
                f"Test requires {world_size} GPUs, but found {torch.cuda.device_count()}"
            )
        if world_size not in pools:
            pools[world_size] = PoolWorker(world_size)
        return pools[world_size]

    yield _get
    for p in pools.values():
        p.shutdown()


def _submit(pool: PoolWorker, **kwargs) -> None:
    # run_dpa_with_cp expects all kwargs as strings (it does e.g.
    # `fp8_bwd == "True"`), matching the old argv-based path. Serialize
    # everything as strings so we don't accidentally change semantics.
    pool.submit({k: str(v) for k, v in kwargs.items()})


dtypes = ["bf16", "fp16"]
qkv_formats = ["bshd", "sbhd", "thd"]
cp_comm_types = ["p2p", "all_gather", "a2a", "a2a+p2p"]
if test_essential:
    configs = ["cp_2_0", "cp_2_2", "cp_3_0", "cp_3_3"]
    model_configs_flash_attn = {k: model_configs_flash_attn[k] for k in configs}
    dtypes = ["bf16"]
    qkv_formats = ["sbhd", "thd"]


@pytest.mark.skipif(
    not FlashAttentionUtils.v2_plus, reason="Flash-attn 2.0+ is required."
)
@pytest.mark.skipif(
    get_device_compute_capability() < (8, 0), reason="CP tests require sm80+."
)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("model", model_configs_flash_attn.keys())
@pytest.mark.parametrize("qkv_format", qkv_formats)
@pytest.mark.parametrize("cp_comm_type", cp_comm_types)
@pytest.mark.parametrize("pad_between_seqs", [False, True])
def test_cp_with_flash_attention(
    cp_pool, dtype, model, qkv_format, cp_comm_type, pad_between_seqs
):
    num_gpus = 4 if cp_comm_type == "a2a+p2p" else 2
    pool = cp_pool(num_gpus)

    if pad_between_seqs:
        if qkv_format != "thd":
            pytest.skip("pad_between_seqs only applies to THD format!")
        if (
            not FlashAttentionUtils.v3_is_installed
            or get_device_compute_capability() > (9, 0)
        ):
            pytest.skip(
                "pad_between_seqs with CP requires Flash Attention v3 on Hopper (sm90)!"
            )
        if cp_comm_type == "a2a+p2p":
            pytest.skip(
                "pad_between_seqs is not yet supported with A2A+P2P CP comm type!"
            )

    config = model_configs_flash_attn[model]
    config.context_parallel = True
    config.cp_comm_type = cp_comm_type

    if config.attn_bias_type != "no_bias" and qkv_format == "thd":
        pytest.skip("No support for bias with THD format!")
    if config.attn_bias_type != "no_bias" and cp_comm_type in [
        "all_gather",
        "a2a",
        "a2a+p2p",
    ]:
        pytest.skip("No support for bias with cp_comm_type={all_gather, a2a, a2a+p2p}!")

    if qkv_format == "thd" and cp_comm_type in ["all_gather", "a2a+p2p"]:
        pytest.skip(
            "No support for THD format with cp_comm_type={all_gather, a2a+p2p}!"
        )

    if (
        config.window_size != (-1, 0)
        and config.window_size != (-1, -1)
        and cp_comm_type
        in [
            "p2p",
            "a2a+p2p",
        ]
    ):
        pytest.skip("No support for SWA with cp_comm_type={p2p, a2a+p2p}!")

    if cp_comm_type in ["a2a", "a2a+p2p"] and (
        config.num_heads % 2 != 0 or config.num_gqa_groups % 2 != 0
    ):
        pytest.skip(
            f"cp_comm_type=a2a requires num_heads ({config.num_heads}) and"
            f" num_gqa_groups ({config.num_gqa_groups}) divisible by 2!"
        )

    # FlashAttention / CP implementation specific: MLA only with KV P2P
    if "p2p" not in cp_comm_type and config.head_dim_qk != config.head_dim_v:
        pytest.skip("MLA CP currently only support KV P2P!")
    dtypes = {"fp16": torch.float16, "bf16": torch.bfloat16}
    available_backends, *_ = get_available_attention_backends(
        config,
        qkv_dtype=dtypes[dtype],
        qkv_layout="_".join([qkv_format] * 3),
    )
    flash_attn_supported, *_ = available_backends
    if not flash_attn_supported:
        pytest.skip("No attention backend available.")

    _submit(
        pool,
        dtype=dtype,
        model=model,
        qkv_format=qkv_format,
        kernel_backend="FlashAttention",
        cp_comm_type=cp_comm_type,
        fa_pad_between_seqs=pad_between_seqs,
        log_level=pytest_logging_level,
    )


model_configs_fused_attn = {
    # test: ModelConfig(b, sq, hq, dqk)
    "cp_1_0": ModelConfig(
        2, 4096, 12, 128, attn_mask_type="causal", return_max_logit=True
    ),  # MHA
    "cp_1_1": ModelConfig(2, 4096, 12, 128, return_max_logit=True),  # MHA
    "cp_1_2": ModelConfig(
        2, 4096, 12, 128, attn_mask_type="causal", attn_bias_type="post_scale_bias"
    ),  # MHA
    "cp_1_3": ModelConfig(2, 4096, 12, 128, attn_bias_type="post_scale_bias"),  # MHA
    "cp_1_4": ModelConfig(
        2, 4096, 12, 128, attn_bias_type="post_scale_bias", bias_shape="bhss"
    ),  # MHA
    "cp_1_5": ModelConfig(
        2, 4096, 12, 128, attn_mask_type="causal", window_size=(512, 512)
    ),  # MHA
    "cp_2_0": ModelConfig(
        2,
        4096,
        32,
        128,
        num_gqa_groups=4,
        attn_mask_type="causal",
    ),  # GQA
    "cp_2_1": ModelConfig(
        2,
        4096,
        32,
        128,
        attn_mask_type="causal",
        window_size=(128, 0),
    ),  # GQA
    "cp_2_2": ModelConfig(
        2,
        4096,
        12,
        128,
        num_gqa_groups=2,
        attn_mask_type="causal",
        attn_bias_type="post_scale_bias",
    ),  # GQA
    "cp_2_3": ModelConfig(
        2,
        4096,
        12,
        128,
        num_gqa_groups=2,
        attn_mask_type="causal",
        attn_bias_type="post_scale_bias",
        bias_shape="11ss",
    ),  # GQA
    "cp_2_4": ModelConfig(
        2,
        4096,
        12,
        128,
        num_gqa_groups=2,
        attn_mask_type="causal",
        attn_bias_type="post_scale_bias",
        bias_shape="111s",
        return_max_logit=True,
    ),  # GQA
    "cp_2_5": ModelConfig(
        2, 4096, 12, 128, num_gqa_groups=2, attn_bias_type="post_scale_bias"
    ),  # GQA
    "cp_2_6": ModelConfig(
        2,
        4096,
        12,
        128,
        num_gqa_groups=2,
        attn_mask_type="causal",
        window_size=(512, 512),
    ),  # GQA
    "cp_3_0": ModelConfig(
        2, 4096, 12, 128, attn_mask_type="causal", head_dim_v=64
    ),  # MLA
    "cp_3_1": ModelConfig(
        2, 4096, 128, 192, head_dim_v=128, attn_mask_type="causal"
    ),  # MLA
    "cp_3_2": ModelConfig(
        2,
        4096,
        12,
        128,
        attn_mask_type="causal",
        attn_bias_type="post_scale_bias",
        head_dim_v=64,
    ),  # MLA
    "cp_3_3": ModelConfig(
        2, 4096, 12, 128, attn_bias_type="post_scale_bias", head_dim_v=64
    ),  # MLA
    "cp_3_4": ModelConfig(
        2,
        4096,
        12,
        128,
        attn_bias_type="post_scale_bias",
        bias_shape="b1ss",
        head_dim_v=64,
    ),  # MLA
    "cp_4_0": ModelConfig(
        2,
        4096,
        64,
        64,
        num_gqa_groups=8,
        attn_mask_type="causal",
        softmax_type="vanilla",
    ),  # GQA
    "cp_4_1": ModelConfig(
        2,
        4096,
        64,
        64,
        num_gqa_groups=8,
        attn_mask_type="causal",
        softmax_type="off-by-one",
    ),  # GQA
    "cp_4_2": ModelConfig(
        2,
        4096,
        64,
        64,
        num_gqa_groups=8,
        attn_mask_type="causal",
        softmax_type="learnable",
    ),  # GQA
    "cp_4_3": ModelConfig(
        2,
        4096,
        64,
        64,
        attn_mask_type="causal",
        window_size=(128, 0),
        softmax_type="learnable",
    ),  # GQA
    # The two layer types of a packed hybrid SWA model, as one row each: GQA 36:6 at head_dim 128,
    # which is the shape the 8b/15b configs run. Every ingredient below already has a row above, but
    # this combination -- padded THD *and* a sliding window *and* a sink *and* GQA -- does not, and
    # that is the composition a packed context-parallel training run depends on. Declared "causal":
    # the runner promotes it to padding_causal under qkv_format="thd" and pads between documents
    # itself for FusedAttention, so asking for padding_causal here is rejected.
    "cp_5_0": ModelConfig(
        2,
        4096,
        36,
        128,
        num_gqa_groups=6,
        attn_mask_type="causal",
        window_size=(4096, 0),
        softmax_type="learnable",
    ),  # GQA, sliding-window layer with a learnable sink (a2a only)
    "cp_5_1": ModelConfig(
        2,
        4096,
        36,
        128,
        num_gqa_groups=6,
        attn_mask_type="causal",
    ),  # GQA, global-attention layer, vanilla softmax (p2p)
    # The rest of the attention-variant matrix the scaling arms sweep, {SWA, global} x {vanilla,
    # off-by-one, learnable}, at the same shape. A sink constrains the comm type -- it is a2a-only --
    # so these exist to prove every variant the 8b/15b configs can select survives packed CP, not
    # just the two that ship today.
    "cp_5_2": ModelConfig(
        2,
        4096,
        36,
        128,
        num_gqa_groups=6,
        attn_mask_type="causal",
        window_size=(4096, 0),
    ),  # GQA, sliding window, vanilla softmax
    "cp_5_3": ModelConfig(
        2,
        4096,
        36,
        128,
        num_gqa_groups=6,
        attn_mask_type="causal",
        window_size=(4096, 0),
        softmax_type="off-by-one",
    ),  # GQA, sliding window, fixed sink
    "cp_5_4": ModelConfig(
        2,
        4096,
        36,
        128,
        num_gqa_groups=6,
        attn_mask_type="causal",
        softmax_type="off-by-one",
    ),  # GQA, global attention, fixed sink
    "cp_5_5": ModelConfig(
        2,
        4096,
        36,
        128,
        num_gqa_groups=6,
        attn_mask_type="causal",
        softmax_type="learnable",
    ),  # GQA, global attention, learnable sink
    # The 15b arm's geometry (40 heads, 10 KV groups). The 36:6 rows above validate the 8b only;
    # 40:10 CP=2 feasibility was arithmetic, not measured, until these two. Same two representative
    # layers: a2a sliding-window + learnable sink, and p2p global vanilla.
    "cp_5_6": ModelConfig(
        2,
        4096,
        40,
        128,
        num_gqa_groups=10,
        attn_mask_type="causal",
        window_size=(4096, 0),
        softmax_type="learnable",
    ),  # 15b: GQA, sliding-window layer with a learnable sink (a2a only)
    "cp_5_7": ModelConfig(
        2,
        4096,
        40,
        128,
        num_gqa_groups=10,
        attn_mask_type="causal",
    ),  # 15b: GQA, global-attention layer, vanilla softmax (p2p)
}


dtypes = ["bf16", "fp16", "fp8"]
qkv_formats = ["bshd", "sbhd", "thd"]
cp_comm_types = ["p2p", "all_gather", "a2a", "a2a+p2p"]
if test_essential:
    configs = [
        "cp_1_0",
        "cp_2_0",
        "cp_2_1",
        "cp_2_2",
        "cp_2_4",
        "cp_3_1",
        "cp_3_2",
        "cp_3_4",
        "cp_4_2",
        "cp_4_3",
        "cp_5_0",
        "cp_5_1",
        "cp_5_2",
        "cp_5_3",
        "cp_5_4",
        "cp_5_5",
        "cp_5_6",
        "cp_5_7",
    ]
    model_configs_fused_attn = {k: model_configs_fused_attn[k] for k in configs}
    dtypes = ["bf16", "fp8"]
    qkv_formats = ["sbhd", "thd"]


@pytest.mark.skipif(get_cudnn_version() < (8, 9, 7), reason="cuDNN 8.9.7+ is required.")
@pytest.mark.skipif(
    get_device_compute_capability() < (8, 0), reason="CP tests require sm80+."
)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("model", model_configs_fused_attn.keys())
@pytest.mark.parametrize("qkv_format", qkv_formats)
@pytest.mark.parametrize("cp_comm_type", cp_comm_types)
@pytest.mark.parametrize("fp8_bwd", [True, False])
@pytest.mark.parametrize("fp8_mha", [True, False])
@pytest.mark.parametrize("fp8_dpa", [True, False])
@pytest.mark.parametrize("scaling_mode", [None, "delayed", "current", "mxfp8"])
@pytest.mark.parametrize("f16_O", [True, False])
def test_cp_with_fused_attention(
    cp_pool,
    dtype,
    model,
    qkv_format,
    cp_comm_type,
    fp8_bwd,
    fp8_mha,
    fp8_dpa,
    scaling_mode,
    f16_O,
):
    config = model_configs_fused_attn[model]
    config.context_parallel = True
    config.cp_comm_type = cp_comm_type

    num_gpus = 4 if cp_comm_type == "a2a+p2p" else 2
    pool = cp_pool(num_gpus)

    if get_device_compute_capability() < (9, 0) and qkv_format == "thd":
        pytest.skip("Only sm90+ architectures support THD format!")
    if get_device_compute_capability() < (9, 0) and dtype == "fp8":
        pytest.skip("Only sm90+ architectures support FP8 attention!")

    if dtype == "fp8" and not (fp8_mha or fp8_dpa):
        pytest.skip("dtype=fp8 requires fp8_dpa=True or fp8_mha=True!")
    if dtype == "fp8" and not fp8_dpa and fp8_mha:
        pytest.skip("Duplicate tests to fp8_dpa=True and fp8_mha=True!")
    if dtype != "fp8" and fp8_bwd:
        pytest.skip("fp8_bwd=True requires dtype=fp8!")
    if dtype != "fp8" and (fp8_mha or fp8_dpa):
        pytest.skip("dtype!=fp8 requires fp8_dpa=False and fp8_mha=False!")

    if dtype == "fp8" and qkv_format == "thd" and cp_comm_type != "a2a":
        # FP8 THD CP is enabled for a2a only: its per-rank kernel receives cu_seqlens_*_padded
        # (Phase 1.1). p2p / all_gather / a2a+p2p FP8 THD are not yet validated; the selector
        # refuses them too, so they also skip at the "No attention backend available" check below.
        pytest.skip("FP8 THD context parallelism is only supported with cp_comm_type=a2a!")
    if dtype == "fp8" and config.attn_bias_type != "no_bias":
        pytest.skip("No support for FP8 attention with bias!")

    if config.attn_bias_type != "no_bias" and qkv_format == "thd":
        pytest.skip("No support for bias with THD format!")
    if config.attn_bias_type != "no_bias" and cp_comm_type in [
        "all_gather",
        "a2a",
        "a2a+p2p",
    ]:
        pytest.skip("No support for bias with cp_comm_type={all_gather, a2a, a2a+p2p}!")

    if qkv_format == "thd" and cp_comm_type in ["all_gather", "a2a+p2p"]:
        pytest.skip(
            "No support for THD format with cp_comm_type={all_gather, a2a+p2p}!"
        )

    if (
        config.window_size[0] != -1 or config.window_size[1] not in [-1, 0]
    ) and cp_comm_type in [
        "p2p",
        "a2a+p2p",
    ]:
        pytest.skip("No support for SWA with cp_comm_type={p2p, a2a+p2p}!")

    if cp_comm_type in ["a2a", "a2a+p2p"] and (
        config.num_heads % 2 != 0 or config.num_gqa_groups % 2 != 0
    ):
        pytest.skip(
            f"cp_comm_type=a2a requires num_heads ({config.num_heads}) and"
            f" num_gqa_groups ({config.num_gqa_groups}) divisible by 2!"
        )

    if config.softmax_type != "vanilla" and cp_comm_type != "a2a":
        pytest.skip(
            f"No support for non-vanilla softmax with cp_comm_type={cp_comm_type}!"
        )
    if (
        config.softmax_type != "vanilla"
        and qkv_format == "thd"
        and get_cudnn_version() < (9, 18, 0)
    ):
        pytest.skip(
            "No support for non-vanilla softmax with THD format and cuDNN < 9.18.0!"
        )

    if dtype == "fp8" and scaling_mode is None:
        pytest.skip("dtype=fp8 requires scaling_mode != None!")
    if dtype != "fp8" and scaling_mode is not None:
        pytest.skip("dtype!=fp8 requires scaling_mode = None!")
    if dtype != "fp8" and not f16_O:
        pytest.skip("dtype!=fp8 requires f16_O=True!")
    if scaling_mode == "delayed" and f16_O:
        pytest.skip("scaling_mode=delayed requires f16_O=False!")
    if scaling_mode == "mxfp8" and not f16_O:
        pytest.skip("scaling_mode=mxfp8 requires f16_O=True!")
    if scaling_mode == "mxfp8" and fp8_mha:
        pytest.skip("No support for scaling_mode=mxfp8 with fp8_mha=True!")

    dtypes = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp8": torch.bfloat16}

    if qkv_format == "thd":
        config = copy.deepcopy(config)
        if "causal" in config.attn_mask_type:
            config.attn_mask_type = "padding_causal"
        else:
            config.attn_mask_type = "padding"

    fp8_meta = {}
    fp8_meta["recipe"] = None
    fp8_meta["local_recipes"] = []
    fp8 = dtype == "fp8" and (fp8_dpa or fp8_mha)
    if fp8 and scaling_mode == "delayed":
        fp8_meta["recipe"] = DelayedScaling(fp8_dpa=True)
        fp8_meta["local_recipes"] = [DelayedScaling(fp8_dpa=True)]
    if fp8 and scaling_mode == "current":
        fp8_meta["recipe"] = DelayedScaling(fp8_dpa=True)
        fp8_meta["local_recipes"] = [
            Float8CurrentScaling(fp8_dpa=True),
            DelayedScaling(fp8_dpa=True),
        ]
    if fp8 and scaling_mode == "mxfp8":
        fp8_meta["recipe"] = MXFP8BlockScaling(fp8_format=Format.E4M3, fp8_dpa=True)
        fp8_meta["local_recipes"] = [
            MXFP8BlockScaling(fp8_format=Format.E4M3, fp8_dpa=True),
        ]

    # For 111s, dbias calculation is not supported as of cuDNN 9.18, hence, test fwd only for 111s.
    is_training = False if config.bias_shape == "111s" else True
    available_backends, _, fused_attn_backends = get_available_attention_backends(
        config,
        qkv_dtype=dtypes[dtype] if dtype != "fp8" else torch.float8_e4m3fn,
        qkv_layout="_".join([qkv_format] * 3),
        fp8=fp8,
        fp8_meta=fp8_meta,
        is_training=is_training,
        deterministic=_deterministic,
    )

    _, fused_attn_supported, _ = available_backends
    # The bottom-right re-query gates on padding_causal_bottom_right support because the
    # sequence-sharding ring path implements CP causal via a bottom-right-aligned causal on the last
    # chunk, so its reference needs that variant. a2a is excluded: it gathers the full sequence on
    # each rank and runs plain (padding_)causal (AttnFuncWithCPAndQKVOA2A passes the original
    # attn_mask_type to the per-rank kernel), so it never uses the bottom-right variant. FP8 does not
    # implement padding_causal_bottom_right over packed input, so without this exclusion every FP8
    # THD a2a case would be skipped here despite the a2a path being available and correct.
    if (
        fused_attn_supported
        and config.attn_mask_type in ["causal", "padding_causal"]
        and cp_comm_type != "a2a"
    ):
        config_copy = copy.deepcopy(config)
        config_copy.context_parallel = False
        config_copy.attn_mask_type = config.attn_mask_type + "_bottom_right"
        available_backends, _, fused_attn_backends = get_available_attention_backends(
            config_copy,
            qkv_dtype=dtypes[dtype] if dtype != "fp8" else torch.float8_e4m3fn,
            qkv_layout="_".join([qkv_format] * 3),
            fp8=fp8,
            fp8_meta=fp8_meta,
            is_training=is_training,
            deterministic=_deterministic,
        )
        _, fused_attn_supported, _ = available_backends
    if not fused_attn_supported:
        pytest.skip("No attention backend available.")

    if _deterministic and config.softmax_type != "vanilla":
        pytest.skip(
            "Deterministic mode does not support non-vanilla softmax with FusedAttention"
        )
    if _deterministic and config.attn_bias_type == "post_scale_bias" and is_training:
        pytest.skip(
            "Deterministic mode does not support post_scale_bias with requires_grad"
        )
    # Observed: cuDNN det THD backward asks for ~128 * bHSS bytes of workspace
    # on sm90; at 1<<30 that's 128 GiB, won't fit on H100's 80 GB. Held exactly
    # at b=2 + power-of-2 S in our sweep; for b>=3 the workspace was observed to
    # grow super-linearly (b=4 took ~4x the b=2 amount, not 2x) — revisit if a
    # config uses b>2.
    SM90_DET_FUSED_THD_BWD_MAX_BHSS = 1 << 30
    if (
        _deterministic
        and qkv_format == "thd"
        and get_device_compute_capability() == (9, 0)
        and config.batch_size
        * config.num_heads
        * config.max_seqlen_q
        * config.max_seqlen_kv
        >= SM90_DET_FUSED_THD_BWD_MAX_BHSS
    ):
        pytest.skip(
            "Deterministic FusedAttention backward with THD format OOMs on sm90"
            " for large bHSS configs (known cuDNN issue)."
        )

    _submit(
        pool,
        dtype=dtype,
        model=model,
        qkv_format=qkv_format,
        kernel_backend="FusedAttention",
        cp_comm_type=cp_comm_type,
        fp8_bwd=fp8_bwd,
        fp8_dpa=fp8_dpa,
        fp8_mha=fp8_mha,
        scaling_mode=scaling_mode,
        f16_O=f16_O,
        is_training=is_training,
        deterministic=_deterministic,
        log_level=pytest_logging_level,
    )


# --------------------------------------------------------------------------------------
# a2a GQA-aware KV-head replication (raises the a2a CP ceiling above 2 for GQA models)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_q_heads,num_kv_heads,cp_size,expected",
    [
        (36, 6, 2, 1),  # 8b, already divisible
        (36, 6, 4, 2),  # 8b at CP=4: 6 not divisible by 4 -> replicate x2 -> 12
        (36, 6, 8, 1),  # 8b at CP=8: no factor divides fanout 6 with (f*6)%8==0 -> 1 (assert fires)
        (40, 10, 2, 1),  # 15b, already divisible
        (40, 10, 4, 2),  # 15b at CP=4: 10 -> x2 -> 20
        (40, 10, 8, 4),  # 15b at CP=8: 10 -> x4 -> 40
        (16, 16, 4, 1),  # MHA, already divisible
        (16, 3, 4, 1),  # fanout 16/3 non-integer -> not clean GQA -> 1
    ],
)
def test_a2a_kv_head_replication_factor_table(num_q_heads, num_kv_heads, cp_size, expected):
    """Pin the structured replication factor. Mirrors radical's _smallest_kv_replication_factor;
    the two must agree (radical's own suite pins its copy). A factor that stops dividing the GQA
    fanout, or stops making (f*num_kv_heads) divisible by cp_size, breaks the a2a head split."""
    from transformer_engine.pytorch.attention.dot_product_attention.context_parallel import (
        get_a2a_kv_head_replication_factor,
    )

    got = get_a2a_kv_head_replication_factor(num_q_heads, num_kv_heads, cp_size)
    assert got == expected, (
        f"factor({num_q_heads}, {num_kv_heads}, {cp_size}) = {got}, expected {expected}"
    )
    if got > 1:
        assert num_q_heads % num_kv_heads == 0 and (num_q_heads // num_kv_heads) % got == 0, (
            "factor must divide the GQA fanout to keep an integer Q:KV ratio"
        )
        assert (got * num_kv_heads) % cp_size == 0, "replicated KV heads must be cp_size-divisible"


@pytest.mark.skipif(get_cudnn_version() < (8, 9, 7), reason="cuDNN 8.9.7+ is required.")
@pytest.mark.skipif(get_device_compute_capability() < (9, 0), reason="THD requires sm90+.")
@pytest.mark.parametrize("model", ["cp_5_1", "cp_5_7", "cp_5_0", "cp_5_6"])
@pytest.mark.parametrize("dtype", ["bf16", "fp8"])
@pytest.mark.parametrize("cp_size", [4, 8])
def test_a2a_kv_head_replication(cp_pool, cp_size, dtype, model):
    """a2a at CP>2 for GQA models whose KV-head count is not divisible by cp_size (8b 36:6, 15b 40:10).

    Without KV replication the a2a head split asserts (e.g. 6 % 4 != 0, 10 % 8 != 0); with it, each
    KV head is replicated f times (f = smallest that divides the GQA fanout and makes f*num_kv %
    cp_size == 0) to reach an integer per-rank count. run_dpa_with_cp compares the CP forward AND
    every input gradient against a single-GPU reference, so this exercises the correctness-critical
    backward reduction: a dK/dV too small by f fails the dk/dv check (BF16 assert_close, FP8 norm-
    ratio magnitude gate). Shown able to fail by disabling the reduction (see PHASE2 worklog).

    Coverage: 15b 40:10 replicates at CP=4 (f=2) and CP=8 (f=4); 8b 36:6 at CP=4 (f=2) but NOT CP=8
    (36 % 8 != 0 is a Q-side limit replication cannot fix -- skipped). cp_5_1/cp_5_7 are global-
    vanilla, cp_5_0/cp_5_6 add a sliding window + learnable sink to prove replication composes.

    FP8 uses delayed scaling (the shipping hybrid recipe) with fp8_dpa. FP8 refuses non-vanilla
    softmax on packed input, so the sink rows (cp_5_0/cp_5_6) skip under FP8 (they ship BF16).
    """
    if torch.cuda.device_count() < cp_size:
        pytest.skip(f"CP={cp_size} requires {cp_size} GPUs")
    config = model_configs_fused_attn[model]
    if config.num_heads % cp_size != 0:
        pytest.skip(f"{model}: num_heads {config.num_heads} not divisible by CP={cp_size} (Q-side)")
    if config.num_gqa_groups % cp_size == 0:
        pytest.skip(f"{model}: {config.num_gqa_groups} KV groups already divide CP={cp_size} (no "
                    "replication to exercise)")
    fp8 = dtype == "fp8"
    if fp8 and config.softmax_type != "vanilla":
        # FP8 refuses non-vanilla softmax on packed input (the sink offset-gradient bug), so the
        # sink layers ship in BF16 -- there is no FP8 sink layer to replicate KV for.
        pytest.skip(f"{model}: FP8 does not support softmax_type={config.softmax_type} on packed")
    pool = cp_pool(cp_size)
    _submit(
        pool,
        dtype=dtype,
        model=model,
        qkv_format="thd",
        kernel_backend="FusedAttention",
        cp_comm_type="a2a",
        fp8_bwd=fp8,
        fp8_dpa=fp8,
        fp8_mha=False,
        scaling_mode="delayed" if fp8 else None,
        f16_O=not fp8,
        is_training=True,
        deterministic=False,
        log_level=pytest_logging_level,
    )
