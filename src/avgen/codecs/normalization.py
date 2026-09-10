"""Latent normalisation: the step that makes flow matching well conditioned.

A VAE emits latents on whatever scale its own training happened to settle on.
For the SD-family autoencoders that is roughly a standard deviation of 5-6 with a
non-zero per-channel mean; for several video VAEs the per-channel standard
deviations differ from each other by more than an order of magnitude.

That is fatal for rectified flow, and the reason is worth spelling out because it
is not obvious from the loss expression. Flow matching interpolates
``x_t = (1 - t) * clean + t * noise`` with ``noise ~ N(0, I)`` and regresses the
velocity ``v = noise - clean``. Two things break when ``clean`` is not O(1):

1. **The interpolation stops interpolating.** If ``clean`` has scale 6 and the
   noise has scale 1, then for every ``t`` below roughly 0.85 the noisy sample is
   dominated by the clean signal. The model is never shown the high-noise regime
   it must actually solve at sampling time, and samples come out as structured
   noise.
2. **The loss is captured by a few channels.** The velocity target's per-channel
   variance is ``var(noise) + var(clean_c)``. An unnormalised MSE therefore
   weights channel ``c`` by its own variance, so a handful of high-variance
   channels absorb essentially the whole gradient and the rest are never learned.

Normalising to zero mean and unit variance *per channel* fixes both at once, and
it costs one multiply-add at the data boundary. It must be applied in exactly one
place — here — and the same statistics must be used to denormalise before
decoding, or the decoder is handed latents from a distribution it has never seen.

The statistics travel with the codec identity (see
:mod:`avgen.codecs.protocols`), because latents normalised with one VAE's
statistics are meaningless to another's decoder.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Self

import torch

__all__ = [
    "LatentStatistics",
    "denormalize_latents",
    "normalize_latents",
    "resolve_latent_normalization",
]


@dataclass(frozen=True, slots=True)
class LatentStatistics:
    """Per-channel mean and standard deviation of a codec's latent space.

    Stored as plain Python floats rather than tensors so the object is hashable,
    JSON-serialisable, and safe to put in a checkpoint manifest or a pinned
    generation config. Tensors are materialised on demand, on the device that
    needs them.

    Per-channel rather than a single scalar is a deliberate choice. The scalar
    ``scaling_factor`` that the diffusers ecosystem popularised is a special case
    (all channels share one std, mean zero), and it is adequate for image VAEs
    whose channels happen to be balanced. Video VAEs with 16 or 48 latent
    channels routinely are not balanced, and the scalar form then leaves the
    per-channel imbalance described in the module docstring fully intact.

    Args:
        mean: Per-channel mean, one entry per latent channel.
        std: Per-channel standard deviation, one entry per latent channel. Every
            entry must be finite and strictly positive.
        channel_dim: Which tensor axis the channels live on. ``1`` for both the
            ``(B, C, T, H, W)`` video layout and the ``(B, C, T)`` audio layout.

    Raises:
        ValueError: If the sequences differ in length, are empty, or contain a
            non-finite value or a non-positive standard deviation.
    """

    mean: tuple[float, ...]
    std: tuple[float, ...]
    channel_dim: int = 1

    def __post_init__(self) -> None:
        """Validate the statistics at construction rather than at first use."""
        if len(self.mean) != len(self.std):
            raise ValueError(
                "mean and std must have one entry per channel; got "
                f"mean={len(self.mean)}, std={len(self.std)}"
            )
        if not self.mean:
            raise ValueError("mean must have at least one channel; got ()")
        if isinstance(self.channel_dim, bool) or self.channel_dim < 0:
            raise ValueError(
                f"channel_dim must be a non-negative integer; got {self.channel_dim!r}"
            )
        for index, value in enumerate(self.mean):
            if not math.isfinite(value):
                raise ValueError(f"mean[{index}] must be finite; got {value!r}")
        for index, value in enumerate(self.std):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"std[{index}] must be finite and positive; got {value!r}"
                )

    @property
    def channels(self) -> int:
        """Number of latent channels these statistics describe."""
        return len(self.mean)

    @property
    def is_identity(self) -> bool:
        """Whether applying these statistics is a no-op.

        Checked exactly rather than approximately: the point of the flag is to
        skip an elementwise kernel, and a codec that genuinely wants a 1.0001
        scale should pay for it.
        """
        return all(value == 0.0 for value in self.mean) and all(
            value == 1.0 for value in self.std
        )

    def tensors(
        self,
        *,
        ndim: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialise mean and std broadcast-shaped for a latent tensor.

        Args:
            ndim: Rank of the tensor the statistics will be applied to.
            device: Device to allocate on.
            dtype: Dtype to allocate in.

        Returns:
            ``(mean, std)`` each shaped with the channel axis populated and every
            other axis of size one, so a single broadcast multiply applies them.

        Raises:
            ValueError: If ``channel_dim`` is not a valid axis of a rank-``ndim``
                tensor.
        """
        if not 0 <= self.channel_dim < ndim:
            raise ValueError(
                f"channel_dim={self.channel_dim} is not a valid axis of a rank-"
                f"{ndim} tensor"
            )
        shape = [1] * ndim
        shape[self.channel_dim] = self.channels
        mean = torch.tensor(self.mean, device=device, dtype=dtype).reshape(shape)
        std = torch.tensor(self.std, device=device, dtype=dtype).reshape(shape)
        return mean, std

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""
        return {
            "mean": list(self.mean),
            "std": list(self.std),
            "channel_dim": self.channel_dim,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> Self:
        """Rebuild statistics from :meth:`to_dict` output.

        Args:
            values: Mapping with ``mean``, ``std``, and optionally
                ``channel_dim``.

        Returns:
            The restored statistics.

        Raises:
            KeyError: If ``mean`` or ``std`` is absent.
        """
        return cls(
            mean=tuple(float(value) for value in values["mean"]),
            std=tuple(float(value) for value in values["std"]),
            channel_dim=int(values.get("channel_dim", 1)),
        )

    @classmethod
    def identity(cls, channels: int, *, channel_dim: int = 1) -> Self:
        """Return statistics that leave latents untouched.

        Used by codecs whose latent space is already standardised — the
        reference codecs, and any VAE trained with a unit-variance latent prior.

        Args:
            channels: Number of latent channels.
            channel_dim: Channel axis.

        Returns:
            Zero-mean unit-std statistics.

        Raises:
            ValueError: If ``channels`` is not positive.
        """
        if isinstance(channels, bool) or channels < 1:
            raise ValueError(f"channels must be a positive integer; got {channels!r}")
        return cls(
            mean=(0.0,) * channels, std=(1.0,) * channels, channel_dim=channel_dim
        )

    @classmethod
    def from_scalar(
        cls,
        channels: int,
        *,
        mean: float = 0.0,
        std: float = 1.0,
        channel_dim: int = 1,
    ) -> Self:
        """Return statistics that apply one shift and scale to every channel.

        This is the diffusers ``shift_factor`` / ``scaling_factor`` convention,
        expressed in the per-channel form so that the rest of the framework has a
        single code path.

        Args:
            channels: Number of latent channels.
            mean: Shift applied to every channel.
            std: Scale applied to every channel.
            channel_dim: Channel axis.

        Returns:
            The broadcast statistics.

        Raises:
            ValueError: If ``channels`` is not positive.
        """
        if isinstance(channels, bool) or channels < 1:
            raise ValueError(f"channels must be a positive integer; got {channels!r}")
        return cls(
            mean=(float(mean),) * channels,
            std=(float(std),) * channels,
            channel_dim=channel_dim,
        )

    @classmethod
    def from_latents(cls, latents: torch.Tensor, *, channel_dim: int = 1) -> Self:
        """Measure statistics from a sample of latents.

        Intended for the offline job that computes a new codec's statistics once
        over a few thousand clips, not for use inside a training step. Measuring
        per batch would make the normalisation depend on batch composition and
        destroy run-to-run reproducibility.

        Args:
            latents: Latent tensor of any rank at least two.
            channel_dim: Channel axis.

        Returns:
            The measured statistics.

        Raises:
            ValueError: If ``latents`` has fewer than two dimensions or a
                measured standard deviation is not positive.
        """
        if latents.ndim < 2:
            raise ValueError(f"latents must have rank >= 2; got {tuple(latents.shape)}")
        reduce_dims = tuple(i for i in range(latents.ndim) if i != channel_dim)
        values = latents.to(torch.float64)
        mean = values.mean(dim=reduce_dims)
        std = values.std(dim=reduce_dims, unbiased=False)
        return cls(
            mean=tuple(float(value) for value in mean.tolist()),
            std=tuple(float(value) for value in std.tolist()),
            channel_dim=channel_dim,
        )


def normalize_latents(
    latents: torch.Tensor,
    statistics: LatentStatistics,
) -> torch.Tensor:
    """Map codec latents onto a zero-mean unit-variance space.

    Args:
        latents: Latents as the codec emits them.
        statistics: Statistics of that codec's latent space.

    Returns:
        Normalised latents, in the input dtype so an fp16/bf16 latent cache stays
        in its storage dtype.

    Raises:
        ValueError: If the channel axis does not match the statistics.
    """
    if statistics.is_identity:
        # Skipping is not just an optimisation: it keeps a bf16 latent cache
        # bit-identical rather than round-tripping through a no-op multiply.
        return latents
    _require_channels(latents, statistics)
    mean, std = statistics.tensors(
        ndim=latents.ndim, device=latents.device, dtype=torch.float32
    )
    return ((latents.to(torch.float32) - mean) / std).to(latents.dtype)


def denormalize_latents(
    latents: torch.Tensor,
    statistics: LatentStatistics,
) -> torch.Tensor:
    """Map normalised latents back onto the codec's own scale.

    The exact inverse of :func:`normalize_latents`. Forgetting this call before
    decoding is the single most common cause of "the model trains fine but the
    decoded video is grey mush".

    Args:
        latents: Normalised latents.
        statistics: Statistics of the target codec's latent space.

    Returns:
        Latents on the codec's scale, in the input dtype.

    Raises:
        ValueError: If the channel axis does not match the statistics.
    """
    if statistics.is_identity:
        return latents
    _require_channels(latents, statistics)
    mean, std = statistics.tensors(
        ndim=latents.ndim, device=latents.device, dtype=torch.float32
    )
    return (latents.to(torch.float32) * std + mean).to(latents.dtype)


def resolve_latent_normalization(
    source: object,
    *,
    channels: int | None = None,
    channel_dim: int = 1,
) -> LatentStatistics:
    """Obtain latent statistics from a codec, a config mapping, or nothing.

    Inference and training both need to answer "what normalisation belongs to
    these latents", and the answer arrives in several shapes depending on whether
    the caller has a live codec object, a deserialised config, or only a channel
    count. Centralising the coercion here keeps the fallback ordering identical
    everywhere, which matters because a silent fallback to identity in one path
    and to a scaling factor in another is a train/inference skew bug that only
    shows up as slightly washed-out samples.

    Resolution order:

    1. Already a :class:`LatentStatistics` — returned unchanged.
    2. An object exposing ``latent_statistics`` — a codec. Recursed into.
    3. A mapping with ``mean``/``std`` — per-channel statistics.
    4. A mapping with ``scaling_factor`` (and optionally ``shift_factor``) — the
       diffusers scalar convention. Note the inversion: diffusers *multiplies* by
       ``scaling_factor`` on encode, so it plays the role of ``1 / std``.
    5. An object exposing ``config.scaling_factor`` — a raw diffusers module.
    6. ``None`` — identity, which requires ``channels`` so the result still has a
       definite width.

    Args:
        source: The statistics, codec, config mapping, or ``None``.
        channels: Channel count, required when ``source`` does not carry one.
        channel_dim: Channel axis for the constructed statistics.

    Returns:
        Resolved per-channel statistics.

    Raises:
        TypeError: If ``source`` is of a shape this function does not understand.
        ValueError: If a channel count is needed and was not supplied.
    """
    if isinstance(source, LatentStatistics):
        return source

    nested = getattr(source, "latent_statistics", None)
    if nested is not None and not isinstance(nested, property):
        return resolve_latent_normalization(
            nested, channels=channels, channel_dim=channel_dim
        )

    if isinstance(source, Mapping):
        return _from_mapping(source, channels=channels, channel_dim=channel_dim)

    config = getattr(source, "config", None)
    if config is not None:
        scaling = getattr(config, "scaling_factor", None)
        if scaling is not None:
            width = channels or getattr(config, "latent_channels", None)
            if width is None:
                raise ValueError(
                    "channels is required to expand a scalar scaling_factor into "
                    "per-channel statistics"
                )
            shift = float(getattr(config, "shift_factor", 0.0) or 0.0)
            return LatentStatistics.from_scalar(
                int(width),
                mean=shift,
                std=1.0 / float(scaling),
                channel_dim=channel_dim,
            )

    if source is None:
        if channels is None:
            raise ValueError(
                "channels is required to build identity latent statistics from "
                "source=None"
            )
        return LatentStatistics.identity(channels, channel_dim=channel_dim)

    raise TypeError(
        "cannot resolve latent statistics from "
        f"{type(source).__name__}; pass a LatentStatistics, a codec exposing "
        "latent_statistics, a mapping with mean/std or scaling_factor, or None"
    )


def _from_mapping(
    source: Mapping[str, Any],
    *,
    channels: int | None,
    channel_dim: int,
) -> LatentStatistics:
    """Build statistics from a config mapping."""
    if "mean" in source and "std" in source:
        return LatentStatistics(
            mean=_floats(source["mean"]),
            std=_floats(source["std"]),
            channel_dim=int(source.get("channel_dim", channel_dim)),
        )
    if "scaling_factor" in source:
        width = channels or source.get("latent_channels")
        if width is None:
            raise ValueError(
                "channels is required to expand a scalar scaling_factor into "
                "per-channel statistics"
            )
        return LatentStatistics.from_scalar(
            int(width),
            mean=float(source.get("shift_factor", 0.0) or 0.0),
            std=1.0 / float(source["scaling_factor"]),
            channel_dim=channel_dim,
        )
    raise TypeError(
        "latent statistics mapping must contain either mean and std, or "
        f"scaling_factor; got keys {sorted(source)}"
    )


def _floats(values: object) -> tuple[float, ...]:
    """Coerce a scalar or sequence of numbers into a float tuple."""
    if isinstance(values, (int, float)):
        return (float(values),)
    if isinstance(values, Sequence):
        return tuple(float(value) for value in values)
    if isinstance(values, torch.Tensor):
        return tuple(float(value) for value in values.flatten().tolist())
    raise TypeError(f"expected a number or a sequence of numbers; got {type(values)}")


def _require_channels(latents: torch.Tensor, statistics: LatentStatistics) -> None:
    """Reject a channel-count mismatch before it becomes a broadcast surprise."""
    axis = statistics.channel_dim
    if not 0 <= axis < latents.ndim:
        raise ValueError(
            f"channel_dim={axis} is not a valid axis of a rank-{latents.ndim} tensor"
        )
    observed = latents.shape[axis]
    if observed != statistics.channels:
        raise ValueError(
            f"latents have {observed} channels on axis {axis} but the statistics "
            f"describe {statistics.channels}; latents from a different codec must "
            "never be normalised with these statistics"
        )
