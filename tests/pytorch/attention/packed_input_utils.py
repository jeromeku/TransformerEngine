"""Construction of packed (THD) attention inputs, and the geometries worth packing.

A packed batch is a single token axis holding several sequences end to end, described by a
cumulative-length vector rather than by padding. The tests that use this module care about the
ragged addressing that implies, so the batches here deliberately contain a length-1 sequence and a
sequence whose length is not a multiple of any tile, both of which are where offset arithmetic
tends to fail.

`as_layout` is the reason this module exists rather than a few inline tensor allocations.
TransformerEngine recovers the QKV layout from data pointers, strides and storage offsets, so a
layout is only exercised when the tensors genuinely share a buffer. Each layout carries a different
ragged-offset multiplier, which is what makes the distinction worth testing.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from utils import ModelConfig

DEVICE = "cuda"

THD_LAYOUTS = ("thd_thd_thd", "t3hd", "th3d", "thd_t2hd", "thd_th2d")

# Per-rank geometries, named for the model and tensor-parallel size that produce them. Per-rank
# head counts are the model's divided by the tensor-parallel size, so one model contributes a
# different geometry at each parallel size and both need testing.
PACKED_CONFIGS = {
    #    name:              ModelConfig(b, sq, hq, dqk, num_gqa_groups=...)
    "mha_d64": ModelConfig(4, 2048, 16, 64, attn_mask_type="padding_causal"),
    "mha_d128": ModelConfig(4, 2048, 16, 128, attn_mask_type="padding_causal"),
    "gqa_8to1_d128": ModelConfig(4, 2048, 64, 128, num_gqa_groups=8,
                                 attn_mask_type="padding_causal"),
    "omnii_8b_tp1": ModelConfig(4, 2048, 32, 128, num_gqa_groups=8,
                                attn_mask_type="padding_causal"),
    "omnii_8b_tp2": ModelConfig(4, 2048, 16, 128, num_gqa_groups=4,
                                attn_mask_type="padding_causal"),
    "omnii_15b_tp1": ModelConfig(4, 2048, 40, 128, num_gqa_groups=10,
                                 attn_mask_type="padding_causal"),
    "omnii_15b_tp2": ModelConfig(4, 2048, 20, 128, num_gqa_groups=5,
                                 attn_mask_type="padding_causal"),
}

# The geometries a deployment actually runs, as opposed to those included to vary a multiplier.
DEPLOYMENT_CONFIGS = ("omnii_8b_tp1", "omnii_8b_tp2", "omnii_15b_tp1", "omnii_15b_tp2")


def is_gqa(config: ModelConfig) -> bool:
    return config.num_heads != config.num_gqa_groups


def head_dims_equal(config: ModelConfig) -> bool:
    return config.head_dim_qk == config.head_dim_v


def layout_supports(layout: str, config: ModelConfig) -> bool:
    """Can this layout physically represent this geometry?

    The 3-way packed layouts interleave Q, K and V in one buffer, so they need equal query and
    key-value head counts and equal head dimensions. The KV-packed layouts interleave K and V only,
    so grouped-query is fine but the head dimensions must still match. Crossing the layout list
    with the geometry list otherwise produces tuples that cannot be built.
    """
    if layout in ("t3hd", "th3d"):
        return not is_gqa(config) and head_dims_equal(config)
    if layout in ("thd_t2hd", "thd_th2d"):
        return head_dims_equal(config)
    if layout == "thd_thd_thd":
        return True
    raise ValueError(f"not a THD layout: {layout!r}")


def constructible(config_names=tuple(PACKED_CONFIGS), layouts=THD_LAYOUTS):
    """(layout, config name) pairs that can actually be built. Use this to parametrize."""
    for name in config_names:
        for layout in layouts:
            if layout_supports(layout, PACKED_CONFIGS[name]):
                yield layout, name


def _long_tailed(budget: int, rng: random.Random, lo: int = 1, hi: int = 2048) -> list[int]:
    """Lengths summing to at most `budget`, with a heavy tail.

    A length-1 and a length-37 sequence are always present: the first has mathematically zero
    query and key gradients and is where relative error measures degenerate, and the second is not
    a multiple of any tile size, so it exercises the tail handling that a round length hides.
    """
    lengths = [1, 37]
    while sum(lengths) < budget:
        nxt = min(hi, max(lo, int(rng.paretovariate(1.1) * 64)))
        if sum(lengths) + nxt > budget:
            break
        lengths.append(nxt)
    rng.shuffle(lengths)
    return lengths


def length_distribution(name: str, seed: int = 0) -> list[int]:
    """Named sequence-length distributions, each probing a different hazard."""
    rng = random.Random(seed)
    if name == "long_tailed_4k":
        return _long_tailed(4096, rng)
    if name == "uniform_8k":
        lengths, total = [], 0
        while total < 8192 - 1100:
            nxt = rng.randint(900, 1100)
            lengths.append(nxt)
            total += nxt
        return lengths
    if name == "one_dominant_64k":
        return [60000] + [rng.randint(32, 256) for _ in range(20)]
    if name == "single_8k":
        return [8192]
    if name == "tile_aligned":
        return [64 * rng.randint(1, 16) for _ in range(6)]
    raise ValueError(f"unknown distribution {name!r}")


@dataclass
class PackedBatch:
    """One packed row: several sequences on a single token axis, plus their boundaries."""

    seqlens: list[int]
    config: ModelConfig
    dtype: torch.dtype
    q: torch.Tensor          # [t, num_heads, head_dim_qk]
    k: torch.Tensor          # [t, num_gqa_groups, head_dim_qk]
    v: torch.Tensor          # [t, num_gqa_groups, head_dim_v]
    cu_seqlens: torch.Tensor  # [b + 1] int32

    @property
    def batch_size(self) -> int:
        return len(self.seqlens)

    @property
    def total_tokens(self) -> int:
        return sum(self.seqlens)

    @property
    def max_seqlen(self) -> int:
        return max(self.seqlens)

    def bounds(self, i: int) -> tuple[int, int]:
        lo = int(self.cu_seqlens[i])
        return lo, lo + self.seqlens[i]

    def sequence(self, x: torch.Tensor, i: int) -> torch.Tensor:
        """Sequence `i`'s rows of any tensor indexed by packed token on dimension 0."""
        lo, hi = self.bounds(i)
        return x[lo:hi]

    def padded(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The same data as [b, s, h, d], zero-padded.

        Zeros rather than noise: padding must not be able to move a tensor-wide amax, or a padded
        reference disagrees with the packed one for a reason unrelated to packing.
        """
        s = self.max_seqlen
        out = []
        for x, h in ((self.q, self.config.num_heads),
                     (self.k, self.config.num_gqa_groups),
                     (self.v, self.config.num_gqa_groups)):
            dense = torch.zeros(self.batch_size, s, h, x.shape[-1],
                                dtype=x.dtype, device=x.device)
            for i in range(self.batch_size):
                dense[i, : self.seqlens[i]] = self.sequence(x, i)
            out.append(dense)
        return tuple(out)


def make_packed_batch(config: ModelConfig, distribution="long_tailed_4k",
                      dtype: torch.dtype = torch.bfloat16, seed: int = 0,
                      scale: float = 0.5) -> PackedBatch:
    """Build a packed batch for `config`. `distribution` is a name or an explicit length list."""
    seqlens = (length_distribution(distribution, seed) if isinstance(distribution, str)
               else list(distribution))
    total = sum(seqlens)
    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    def draw(heads: int, dim: int) -> torch.Tensor:
        return torch.randn(total, heads, dim, generator=generator,
                           device=DEVICE, dtype=dtype) * scale

    cu_seqlens = torch.zeros(len(seqlens) + 1, dtype=torch.int32, device=DEVICE)
    cu_seqlens[1:] = torch.cumsum(
        torch.tensor(seqlens, dtype=torch.int32, device=DEVICE), 0)
    return PackedBatch(
        seqlens=seqlens, config=config, dtype=dtype,
        q=draw(config.num_heads, config.head_dim_qk),
        k=draw(config.num_gqa_groups, config.head_dim_qk),
        v=draw(config.num_gqa_groups, config.head_dim_v),
        cu_seqlens=cu_seqlens,
    )


def as_layout(batch: PackedBatch, layout: str):
    """Return (q, k, v) views that TransformerEngine will recognise as `layout`.

    The tensors must genuinely share a buffer, since the layout is recovered from pointers and
    strides rather than declared.
    """
    config, total = batch.config, batch.total_tokens
    heads, groups = config.num_heads, config.num_gqa_groups
    dim = config.head_dim_qk

    if layout == "thd_thd_thd":
        return batch.q, batch.k, batch.v

    if not layout_supports(layout, config):
        raise ValueError(f"{layout!r} cannot represent this geometry")

    if layout == "t3hd":                                     # [t, 3, h, d]
        buf = torch.empty(total, 3, heads, dim, dtype=batch.dtype, device=DEVICE)
        buf[:, 0], buf[:, 1], buf[:, 2] = batch.q, batch.k, batch.v
        return buf[:, 0], buf[:, 1], buf[:, 2]
    if layout == "th3d":                                     # [t, h, 3, d]
        buf = torch.empty(total, heads, 3, dim, dtype=batch.dtype, device=DEVICE)
        buf[:, :, 0], buf[:, :, 1], buf[:, :, 2] = batch.q, batch.k, batch.v
        return buf[:, :, 0], buf[:, :, 1], buf[:, :, 2]
    if layout == "thd_t2hd":                                 # q separate, kv [t, 2, hg, d]
        kv = torch.empty(total, 2, groups, dim, dtype=batch.dtype, device=DEVICE)
        kv[:, 0], kv[:, 1] = batch.k, batch.v
        return batch.q, kv[:, 0], kv[:, 1]
    if layout == "thd_th2d":                                 # q separate, kv [t, hg, 2, d]
        kv = torch.empty(total, groups, 2, dim, dtype=batch.dtype, device=DEVICE)
        kv[:, :, 0], kv[:, :, 1] = batch.k, batch.v
        return batch.q, kv[:, :, 0], kv[:, :, 1]

    raise ValueError(f"unknown THD layout {layout!r}")
