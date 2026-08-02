"""Packed (THD) FP8 attention gives the same answer sharded across ranks as it does on one.

Every other test of this path runs on a single device, so nothing else establishes that the sharded
numerical result matches the unsharded one. The two shardings covered are the ones a deployment
uses: 32 query and 8 key-value heads becoming 16 and 4 per rank, and 40 and 10 becoming 20 and 5.

The rank work runs in a subprocess under torchrun rather than in pytest itself. Per-test setup,
fixtures and skips inside ranks that are also inside collectives deadlock as soon as one rank takes
a different path, so the ranks run a plain script and pytest asserts on its exit status.

    python3 -m pytest test_packed_attention_sharded.py -q -rs
"""

import os
import pathlib
import socket
import subprocess

import pytest
import torch

TEST_ROOT = pathlib.Path(__file__).parent.resolve()
RUNNER = TEST_ROOT / "run_packed_attention_sharded.py"

# Whole-model geometry, and the per-rank geometry two-way sharding must produce.
SHARDINGS = [
    ("omnii_8b_tp1", "omnii_8b_tp2"),
    ("omnii_15b_tp1", "omnii_15b_tp2"),
]

# Under current scaling each rank scales from the amax of its own heads, so the sharded result
# differs from the unsharded one. Measured maximum over both shardings is 0.098; this bounds it
# rather than pretending it is zero.
CURRENT_SCALING_TOLERANCE = 0.15

pytestmark = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two devices")


def free_port():
    """A port the kernel has just confirmed is free.

    A fixed port number is a flaky test waiting to happen: a previous run's socket in TIME_WAIT
    makes the rendezvous fail, and the resulting rank-zero exit looks identical to a numerical
    failure.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def launch(*args, timeout=1800):
    env = dict(os.environ)
    env.setdefault("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1")
    cmd = ["torchrun", "--nproc_per_node=2", f"--master_port={free_port()}", str(RUNNER), *args]
    return subprocess.run(cmd, cwd=TEST_ROOT, env=env, capture_output=True, text=True,
                          timeout=timeout)


@pytest.mark.parametrize("full,per_rank", SHARDINGS)
def test_delayed_scaling_sharding_is_exact(full, per_rank):
    """Sharding by head changes nothing at all under delayed scaling.

    Heads do not interact, and each rank's contiguous query slice maps to its own key-value groups,
    so both the per-head arithmetic and the accumulation order within each group are preserved.
    Bitwise equality is therefore the right assertion, and a weaker tolerance would hide a future
    change that made the scale depend on how many heads share a tensor.
    """
    result = launch("--config", full, "--expect-per-rank-config", per_rank,
                    "--recipe", "delayed", "--require-exact")
    assert result.returncode == 0, f"{result.stdout[-3000:]}\n{result.stderr[-3000:]}"


@pytest.mark.parametrize("full,per_rank", SHARDINGS)
def test_current_scaling_sharding_stays_within_tolerance(full, per_rank):
    """Under current scaling the sharded result differs, and the difference is bounded.

    The scale comes from the amax of the tensor in hand, so a rank holding half the heads scales
    differently from one holding all of them. That is a property of the recipe rather than a
    defect, and the bound is what keeps it from growing unnoticed.
    """
    result = launch("--config", full, "--expect-per-rank-config", per_rank,
                    "--recipe", "current", "--max-relative-rms", str(CURRENT_SCALING_TOLERANCE))
    assert result.returncode == 0, f"{result.stdout[-3000:]}\n{result.stderr[-3000:]}"


def test_a_wrong_head_slice_is_detected():
    """The comparison fails when the ranks are given the wrong heads.

    Without this, a comparison that passed for every input would be indistinguishable from one that
    works. Every rank is handed the first rank's heads, so the gathered result cannot reconstruct
    the unsharded one.
    """
    result = launch("--config", "omnii_8b_tp1", "--recipe", "delayed", "--require-exact",
                    "--wrong-head-slice")
    assert result.returncode != 0, f"a wrong shard was accepted:\n{result.stdout[-2000:]}"


def test_an_indivisible_geometry_is_refused():
    """A geometry whose heads do not divide across the ranks is refused, not truncated.

    The per-rank geometries are already the result of sharding, so asking to shard one again is a
    configuration error. Integer division would quietly produce a geometry nothing runs.
    """
    result = launch("--config", "omnii_15b_tp2", "--recipe", "delayed")
    assert result.returncode != 0
    assert "do not divide" in result.stderr, f"unexpected failure:\n{result.stderr[-2000:]}"


def test_amax_reduction_does_not_change_the_result():
    """Supplying an amax reduction group makes no difference to attention quantization.

    Recorded because the opposite is the natural assumption: reducing the amax across ranks sounds
    like what would keep a sharded FP8 result equal to an unsharded one. Measured, the results are
    identical either way, so the group is not what makes delayed scaling exact and its absence is
    not what makes current scaling differ. Anyone tempted to explain either result by amax
    reduction should see this fail first.
    """
    with_group = launch("--config", "omnii_8b_tp1", "--recipe", "current",
                        "--max-relative-rms", "999")
    without = launch("--config", "omnii_8b_tp1", "--recipe", "current",
                     "--max-relative-rms", "999", "--no-amax-reduction")
    assert with_group.returncode == 0 and without.returncode == 0

    def measurements(out):
        return [line for line in out.splitlines()
                if line.startswith(("out ", "dq ", "dk ", "dv "))]

    assert measurements(with_group.stdout) == measurements(without.stdout), (
        f"amax reduction changed the result:\nwith:\n{measurements(with_group.stdout)}\n"
        f"without:\n{measurements(without.stdout)}"
    )
