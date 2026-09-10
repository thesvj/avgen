"""Axial rotary position embeddings driven by physical coordinates.

Rotary embeddings encode position as a rotation applied to pairs of query and
key channels, so attention sees *relative* position for free. The design
decision this module encodes is what "position" means.

**Frequencies come from the coordinate, not from the token index.** The usual
implementation builds a table indexed by sequence position, which quietly bakes
three assumptions into the weights: that the sequence is contiguous, that the
frame rate is fixed, and that the resolution is fixed. All three are false for
video training worth doing. :class:`~avgen.core.tokens.TokenStream` carries a
``(batch, length, 3)`` coordinate tensor in physical units — seconds, latent
rows, latent columns — and this module multiplies those numbers by the inverse
frequencies directly. The consequences are the entire point:

* A 24 fps clip and a 30 fps clip land in the same phase space, so mixed-frame-
  rate batches train one time axis instead of two.
* An audio stream at 43 latent frames per second and a video stream at 6 land
  in the same phase space too, which is what makes audio-video attention
  temporally meaningful rather than an index coincidence.
* A model trained at 256 px extrapolates to 512 px, because column 40 is column
  40 at either resolution instead of being "80% of the way along the row".
* Packed, shuffled, or context-parallel-sharded sequences still know where each
  token came from.

**Context parallelism needs no special case here.** The tables are built from
whatever coordinate slice the rank is holding. After
:func:`~avgen.parallel.context.shard_stream` hands rank 3 of 8 the tokens for
its slice of the sequence, those tokens arrive with *their own* coordinates, and
the rotation computed from them is exactly the rotation the global model would
have applied. No offset arithmetic, no rank-aware branch, nothing to get wrong.

**The rotation is interleaved (GPT-J style), not split-half (NeoX style).**
Head channels are partitioned into a time budget, a height budget, and a width
budget that are concatenated; a split-half rotation would pair a time channel
with a width channel across the halves and mix the axes. Interleaved pairing
keeps each rotation inside one axis. The two conventions are not
interchangeable: a checkpoint trained under one is noise under the other.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

__all__ = [
    "RoPEScaling",
    "RotaryEmbedding",
    "RotaryTables",
    "apply_rotary",
    "build_rotary_tables",
    "rope_axis_pairs",
]

#: Frequency-scaling modes understood by :class:`RoPEScaling`.
ROPE_SCALING_MODES: tuple[str, ...] = ("none", "linear", "ntk")


@dataclass(frozen=True, slots=True)
class RoPEScaling:
    """Frequency scaling for extrapolating beyond the trained extent.

    Training happens on short clips at modest resolution because that is what
    fits; sampling wants long clips at high resolution. Without scaling, a
    coordinate beyond the trained range produces a rotation phase the model has
    never seen, and the sample degenerates — repeated frames, drifting
    geometry, or a hard loss of temporal coherence part-way through.

    Two remedies, both cheap and both applied at table-build time so no weight
    changes:

    * ``"linear"`` — position interpolation. Divide the coordinate by the
        factor, squeezing the longer sequence back into the trained phase
        range. Exact and safe, but it compresses high-frequency detail
        everywhere, including at short lengths, so a model sampled this way is
        slightly blurrier in time.
    * ``"ntk"`` — NTK-aware base scaling. Raise the rotary base instead, by
        ``factor ** (dim / (dim - 2))``, which stretches the low-frequency
        channels that carry long-range position while leaving the high-frequency
        channels that carry local detail nearly untouched. Usually the better
        trade at the same factor, and the reason it is the default recommendation
        for length extrapolation.

    Time and space are scaled separately because they extrapolate for different
    reasons: longer clips stretch time, larger frames stretch space, and the two
    factors are rarely equal.

    Args:
        mode: One of :data:`ROPE_SCALING_MODES`.
        time_factor: Extrapolation factor for the temporal axis, as a multiple
            of the trained duration.
        spatial_factor: Extrapolation factor for the row and column axes, as a
            multiple of the trained side length.

    Raises:
        ValueError: If the mode is unknown or a factor is not positive.
    """

    mode: str = "none"
    time_factor: float = 1.0
    spatial_factor: float = 1.0

    def __post_init__(self) -> None:
        """Validate the mode and the factors."""
        if self.mode not in ROPE_SCALING_MODES:
            raise ValueError(
                f"mode must be one of {ROPE_SCALING_MODES}; got {self.mode!r}"
            )
        for name in ("time_factor", "spatial_factor"):
            value = float(getattr(self, name))
            if not value > 0.0:
                raise ValueError(f"{name} must be positive; got {value!r}")

    @property
    def is_identity(self) -> bool:
        """Whether this configuration leaves the frequencies untouched."""
        return self.mode == "none" or (
            self.time_factor == 1.0 and self.spatial_factor == 1.0
        )

    def factors(self) -> tuple[float, float, float]:
        """Return the per-axis factors as ``(time, height, width)``."""
        return (self.time_factor, self.spatial_factor, self.spatial_factor)


def rope_axis_pairs(head_dim: int) -> tuple[int, int, int]:
    """Split a head dimension into ``(time, height, width)`` rotation budgets.

    Each rotation consumes two channels, so the head is first halved into
    pairs and the pairs are split three ways. **Time takes the remainder.** A
    head dimension of 64 gives 32 pairs and a ``(12, 10, 10)`` split; the extra
    resolution goes to the axis where getting position wrong is most visible,
    because a temporal phase error shows up as motion incoherence across the
    whole clip while a spatial one shows up as a local artefact.

    Args:
        head_dim: Channels per attention head.

    Returns:
        Rotation-pair counts for time, height, and width. They sum to
        ``head_dim // 2``.

    Raises:
        ValueError: If ``head_dim`` is not a positive multiple of six. Six is
            required so that every axis receives at least one pair and the
            three budgets tile the head exactly.
    """
    if head_dim < 6 or head_dim % 2 != 0:
        raise ValueError(
            f"head_dim must be an even integer of at least 6 so each rotary axis "
            f"gets at least one channel pair; got {head_dim!r}"
        )
    pairs = head_dim // 2
    spatial = pairs // 3
    return (pairs - 2 * spatial, spatial, spatial)


@dataclass(frozen=True, slots=True)
class RotaryTables:
    """Precomputed rotation cosines and sines for one token sequence.

    Stored at *pair* resolution — ``head_dim // 2`` entries, not ``head_dim`` —
    because the duplicated-channel layout that some implementations use doubles
    a tensor that is already ``batch * length * head_dim`` floats. At 100k
    tokens and a 128-wide head that is the difference between 25 MB and 50 MB
    of fp32 held live across the whole forward pass.

    The head axis is materialised as a singleton so the tables broadcast
    against ``(batch, heads, length, pairs)`` with no reshape at the call site.

    Args:
        cos: ``(batch, 1, length, pairs)`` float32 cosines.
        sin: ``(batch, 1, length, pairs)`` float32 sines.
    """

    cos: torch.Tensor
    sin: torch.Tensor

    @property
    def pairs(self) -> int:
        """Number of rotation pairs, i.e. half the head dimension."""
        return int(self.cos.shape[-1])

    @property
    def length(self) -> int:
        """Local sequence length these tables cover."""
        return int(self.cos.shape[-2])

    @property
    def device(self) -> torch.device:
        """Device the tables live on."""
        return self.cos.device

    def validate(self) -> None:
        """Validate rank, agreement, and dtype.

        Raises:
            ValueError: If the tables disagree in shape or are not rank 4.
            TypeError: If the tables are not float32.
        """
        if self.cos.ndim != 4 or self.sin.ndim != 4:
            raise ValueError(
                f"rotary tables must have rank 4 (batch, 1, length, pairs); got "
                f"{tuple(self.cos.shape)} and {tuple(self.sin.shape)}"
            )
        if self.cos.shape != self.sin.shape:
            raise ValueError(
                f"cos and sin must have the same shape; got {tuple(self.cos.shape)} "
                f"and {tuple(self.sin.shape)}"
            )
        for name, tensor in (("cos", self.cos), ("sin", self.sin)):
            if tensor.dtype is not torch.float32:
                raise TypeError(f"{name} must be float32; got {tensor.dtype}")


def build_rotary_tables(
    coords: torch.Tensor,
    *,
    head_dim: int,
    theta: float = 10000.0,
    time_scale: float = 1.0,
    scaling: RoPEScaling | None = None,
) -> RotaryTables:
    """Build rotation tables from physical token coordinates.

    Args:
        coords: ``(batch, length, 3)`` float32 coordinates as
            ``(seconds, row, column)``. Under context parallelism this is the
            **local** shard's coordinates, which is exactly what makes the
            result correct without any rank-dependent offset.
        head_dim: Channels per attention head.
        theta: Rotary base. Larger bases put more channels at low frequency and
            extend the unambiguous position range; 10000 is the value every
            pretrained tower uses and changing it invalidates a checkpoint.
        time_scale: Multiplier applied to the seconds axis before the
            frequencies. Setting it so that one unit of scaled time is roughly
            one token of the *denser* stream is what puts video and audio in a
            comparable phase range; leaving them incomparable makes cross-modal
            attention rotate the two streams past each other.
        scaling: Optional frequency scaling for extrapolation.

    Returns:
        Tables covering exactly the tokens in ``coords``.

    Raises:
        ValueError: If ``coords`` is not ``(batch, length, 3)``.
        TypeError: If ``coords`` is not float32.
    """
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError(
            f"coords must be (batch, length, 3) as (seconds, row, column); "
            f"got {tuple(coords.shape)}"
        )
    if coords.dtype is not torch.float32:
        raise TypeError(f"coords must be float32; got {coords.dtype}")

    plan = scaling if scaling is not None else RoPEScaling()
    budgets = rope_axis_pairs(head_dim)
    factors = plan.factors()
    device = coords.device

    angles: list[torch.Tensor] = []
    for axis, (pairs, factor) in enumerate(zip(budgets, factors, strict=True)):
        axis_theta = theta
        # Phase is computed in float32 regardless of the compute dtype. A bf16
        # coordinate has ~3 decimal digits, and at 100k tokens the accumulated
        # phase error is larger than the rotation between adjacent positions.
        position = coords[..., axis]
        if axis == 0 and time_scale != 1.0:
            position = position * time_scale

        if plan.mode == "linear" and factor != 1.0:
            # Position interpolation: squeeze the longer sequence back inside
            # the phase range the weights were actually trained on.
            position = position / factor
        elif plan.mode == "ntk" and factor != 1.0:
            # NTK-aware scaling stretches the low-frequency channels only. The
            # exponent dim/(dim-2) is what makes the *highest*-frequency channel
            # land on the same wavelength it had before scaling, which is why
            # local detail survives where linear interpolation blurs it.
            dim = 2 * pairs
            if dim > 2:
                axis_theta = theta * factor ** (dim / (dim - 2))

        exponent = torch.arange(pairs, device=device, dtype=torch.float32) / pairs
        inverse_frequency = torch.pow(
            torch.tensor(axis_theta, device=device, dtype=torch.float32), -exponent
        )
        angles.append(position[..., None] * inverse_frequency)

    phase = torch.cat(angles, dim=-1)[:, None]
    return RotaryTables(cos=torch.cos(phase), sin=torch.sin(phase))


def apply_rotary(x: torch.Tensor, tables: RotaryTables) -> torch.Tensor:
    """Rotate query or key channels in place of an additive position embedding.

    Args:
        x: ``(batch, heads, length, head_dim)`` projected queries or keys.
        tables: Tables built by :func:`build_rotary_tables` for this exact
            token slice.

    Returns:
        The rotated tensor, in ``x``'s dtype.

    Raises:
        ValueError: If ``x`` is not rank 4 or its head dimension does not match
            the tables.
    """
    if x.ndim != 4:
        raise ValueError(
            f"rotary input must be (batch, heads, length, head_dim); "
            f"got {tuple(x.shape)}"
        )
    if x.shape[-1] != 2 * tables.pairs:
        raise ValueError(
            f"head_dim {x.shape[-1]} does not match rotary tables built for "
            f"head_dim {2 * tables.pairs}"
        )
    if x.shape[-2] != tables.length:
        raise ValueError(
            f"sequence length {x.shape[-2]} does not match rotary tables built "
            f"for length {tables.length}; under context parallelism the tables "
            "must be built from the local coordinate shard"
        )

    # Rotate in float32 and cast back once. The rotation is a rank-2 orthogonal
    # transform per pair, so it is norm-preserving in exact arithmetic; doing it
    # in bf16 loses that property and the drift compounds across every block.
    pairs = x.float().unflatten(-1, (tables.pairs, 2))
    even = pairs[..., 0]
    odd = pairs[..., 1]
    cos = tables.cos
    sin = tables.sin
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """Stateless module that turns token coordinates into rotation tables.

    Holds no parameters and no buffers by design. A precomputed table would
    have to be sized for the longest sequence the model will ever see, would be
    wrong the moment the frame rate changed, and — being a buffer — would be
    silently filled with garbage by the ``to_empty()`` step of meta-device
    construction. Computing from coordinates costs a few elementwise kernels
    once per forward and removes all three failure modes.

    Args:
        head_dim: Channels per attention head.
        theta: Rotary base.
        time_scale: Multiplier on the seconds axis.
        scaling: Optional frequency scaling for extrapolation.
    """

    head_dim: int
    theta: float
    time_scale: float

    def __init__(
        self,
        head_dim: int,
        *,
        theta: float = 10000.0,
        time_scale: float = 1.0,
        scaling: RoPEScaling | None = None,
    ) -> None:
        super().__init__()
        rope_axis_pairs(head_dim)  # validate eagerly, not on the first forward
        self.head_dim = head_dim
        self.theta = theta
        self.time_scale = time_scale
        self.scaling = scaling if scaling is not None else RoPEScaling()

    def forward(self, coords: torch.Tensor) -> RotaryTables:
        """Build tables for one batch of token coordinates.

        Args:
            coords: ``(batch, length, 3)`` physical coordinates, local to this
                context-parallel rank.

        Returns:
            The rotation tables.
        """
        return build_rotary_tables(
            coords,
            head_dim=self.head_dim,
            theta=self.theta,
            time_scale=self.time_scale,
            scaling=self.scaling,
        )

    def extra_repr(self) -> str:
        """Describe the geometry in module printouts."""
        budgets = rope_axis_pairs(self.head_dim)
        return (
            f"head_dim={self.head_dim}, pairs(t,h,w)={budgets}, theta={self.theta}, "
            f"time_scale={self.time_scale}, scaling={self.scaling.mode}"
        )
