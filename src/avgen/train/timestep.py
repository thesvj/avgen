"""How noise levels are drawn, and why the distribution decides sample quality.

A flow-matching trainer chooses a noise level ``t`` for every sample, forms
``x_t = (1 - t) x_0 + t eps``, and asks the model to predict the velocity. The
*distribution* ``t`` is drawn from is not a detail — it is the allocation of a
fixed training budget across the denoising trajectory, and it is one of the few
knobs that visibly changes what the model can do:

* Near ``t = 1`` the model decides **global structure**: layout, subject
  identity, camera motion, whether the clip is coherent at all.
* Near ``t = 0`` it decides **texture**: grain, edges, high-frequency detail.

Uniform sampling spends the budget evenly, which under-trains the structural
regime because those steps are the ones the sampler cannot recover from. SD3
showed that a logit-normal density — peaked in the middle, thin at both ends —
beats uniform reliably, on the argument that both endpoints are nearly trivial
(at ``t = 0`` the answer is roughly the input, at ``t = 1`` there is no signal
to condition on) while the middle is where the hard decisions live.

**Resolution changes the answer.** This is the part that is easy to miss and
expensive to get wrong. Adding independent noise to ``L`` tokens destroys
signal at a rate that grows with ``L``: at a fixed ``t``, a 100k-token 720p clip
retains far more recoverable structure than a 1k-token 256px clip, because the
redundancy between neighbouring tokens is higher. Train both at the same ``t``
distribution and the high-resolution model spends almost all of its budget in
an easy regime and never learns to build structure from nothing.

The fix, introduced by SD3 and made sequence-length-dependent by LTX-Video and
Flux, is to *shift* the sampled ``t`` towards 1 by an amount that grows with the
sequence length::

    t' = shift * t / (1 + (shift - 1) * t)

with ``shift`` linearly interpolated in sequence length between a low-resolution
anchor and a high-resolution one. At high resolution this is the single largest
quality win available from the training objective, and it costs one extra
elementwise operation per step.

See :class:`ShiftedLogitNormalSampler` for the arithmetic and the defaults.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, runtime_checkable

import torch

from avgen.core._validate import require_finite, require_positive
from avgen.train._random import normal, uniform

__all__ = [
    "LogitNormalSampler",
    "ModeSampler",
    "ShiftedLogitNormalSampler",
    "TimestepSampler",
    "UniformSampler",
    "build_timestep_sampler",
    "list_timestep_samplers",
    "register_timestep_sampler",
    "resolution_shift",
    "shift_timesteps",
]


@runtime_checkable
class TimestepSampler(Protocol):
    """Draws per-sample noise levels in ``[0, 1]``.

    Implement this to change the training-budget allocation without touching
    the objective. Every implementation must be a pure function of the supplied
    generator: given the same generator state, batch size, and sequence length,
    it must return the same tensor, on any rank and after any restart.
    """

    def sample(
        self,
        batch: int,
        *,
        device: torch.device | str,
        generator: torch.Generator,
        sequence_length: int | None = None,
    ) -> torch.Tensor:
        """Draw ``(batch,)`` float32 noise levels in ``[0, 1]``."""
        ...


_SAMPLERS: dict[str, Callable[..., TimestepSampler]] = {}

_T = TypeVar("_T", bound=Callable[..., TimestepSampler])


def register_timestep_sampler(name: str) -> Callable[[_T], _T]:
    """Register a timestep sampler class under a configuration name.

    Args:
        name: Name used in configuration files and by
            :func:`build_timestep_sampler`.

    Returns:
        A decorator that registers and returns the class unchanged.

    Raises:
        ValueError: If ``name`` is already registered. Silent replacement is
            rejected because two modules registering the same name makes the
            active implementation depend on import order, which is exactly the
            kind of difference that makes a run irreproducible.
    """

    def decorate(factory: _T) -> _T:
        if name in _SAMPLERS:
            raise ValueError(f"timestep sampler {name!r} is already registered")
        _SAMPLERS[name] = factory
        return factory

    return decorate


def build_timestep_sampler(
    name: str,
    config: Mapping[str, Any] | None = None,
) -> TimestepSampler:
    """Construct a registered timestep sampler.

    Args:
        name: Registered name.
        config: Keyword arguments for the sampler's constructor.

    Returns:
        The constructed sampler.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    if name not in _SAMPLERS:
        available = ", ".join(list_timestep_samplers())
        raise KeyError(f"unknown timestep sampler {name!r}; available: {available}")
    return _SAMPLERS[name](**dict(config or {}))


def list_timestep_samplers() -> tuple[str, ...]:
    """Return every registered sampler name, sorted.

    Returns:
        Sorted registered names.
    """
    return tuple(sorted(_SAMPLERS))


def resolution_shift(
    sequence_length: int | None,
    *,
    base_seq_len: int,
    base_shift: float,
    max_seq_len: int,
    max_shift: float,
) -> float:
    """Interpolate the shift strength for a sequence length.

    The interpolation is linear in *tokens*, not in ``log`` tokens, because that
    is what SD3, LTX-Video, and Flux calibrated their anchors against; changing
    the interpolant silently invalidates the published constants.

    Args:
        sequence_length: Tokens the model will see per sample, or ``None`` to
            fall back to the low-resolution anchor.
        base_seq_len: Sequence length of the low-resolution anchor.
        base_shift: Shift at the low-resolution anchor.
        max_seq_len: Sequence length of the high-resolution anchor.
        max_shift: Shift at the high-resolution anchor.

    Returns:
        The shift strength, clamped to the anchor interval.
    """
    if sequence_length is None:
        return base_shift
    span = max_seq_len - base_seq_len
    slope = (max_shift - base_shift) / span
    value = base_shift + slope * (sequence_length - base_seq_len)
    # Clamping rather than extrapolating: a 4-token debug batch and a 500k-token
    # experimental batch both stay inside a regime the anchors actually describe,
    # and neither can produce a non-positive shift that would invert the map.
    low, high = min(base_shift, max_shift), max(base_shift, max_shift)
    return min(max(value, low), high)


def shift_timesteps(timesteps: torch.Tensor, shift: float) -> torch.Tensor:
    """Apply the SD3 timestep shift.

    ``t' = shift * t / (1 + (shift - 1) * t)`` is a monotone bijection of
    ``[0, 1]`` onto itself that fixes both endpoints. ``shift > 1`` moves mass
    towards 1 (more high-noise training), ``shift < 1`` towards 0. Because it is
    a bijection, the *sampler* at inference time can apply the identical map to
    its step schedule and stay consistent with training — which is why a
    reparameterisation is used here instead of simply re-fitting the density.

    Args:
        timesteps: ``(batch,)`` values in ``[0, 1]``.
        shift: Positive shift strength.

    Returns:
        The shifted timesteps.

    Raises:
        ValueError: If ``shift`` is not positive, which would make the map
            non-monotone and let a timestep leave ``[0, 1]``.
    """
    if not shift > 0.0:
        raise ValueError(f"shift must be positive; got {shift!r}")
    return shift * timesteps / (1.0 + (shift - 1.0) * timesteps)


@register_timestep_sampler("uniform")
@dataclass(frozen=True, slots=True)
class UniformSampler:
    """Draw ``t`` uniformly from ``[0, 1)``.

    The honest baseline: it makes no assumption about which part of the
    trajectory is hard. Keep it for ablations and for debugging a suspected
    sampler bug, where an exactly-known density is worth more than quality.

    Args:
        low: Lower bound of the interval.
        high: Upper bound of the interval.
    """

    low: float = 0.0
    high: float = 1.0

    def __post_init__(self) -> None:
        """Validate that the interval is a non-empty subset of ``[0, 1]``."""
        for name in ("low", "high"):
            require_finite(name, getattr(self, name))
        if not 0.0 <= self.low < self.high <= 1.0:
            raise ValueError(
                "require 0 <= low < high <= 1; got "
                f"low={self.low!r}, high={self.high!r}"
            )

    def sample(
        self,
        batch: int,
        *,
        device: torch.device | str,
        generator: torch.Generator,
        sequence_length: int | None = None,
    ) -> torch.Tensor:
        """Draw ``(batch,)`` uniform noise levels.

        Args:
            batch: Number of samples.
            device: Device the result must end up on.
            generator: Generator supplying the randomness.
            sequence_length: Ignored; uniform sampling is resolution blind.

        Returns:
            ``(batch,)`` float32 noise levels.
        """
        del sequence_length
        draw = uniform((batch,), generator=generator, device=device)
        return self.low + (self.high - self.low) * draw


@register_timestep_sampler("logit_normal")
@dataclass(frozen=True, slots=True)
class LogitNormalSampler:
    """Draw ``t = sigmoid(u)`` with ``u ~ Normal(mean, std)`` — the SD3 default.

    The density is peaked in the middle of the trajectory and thin at both
    ends, which matches where the learning signal actually is: at ``t`` near 0
    the target velocity is almost determined by the input, and at ``t`` near 1
    there is no input left to condition on. Both extremes are cheap to fit and
    consume budget that the middle needs.

    The alternative that lost was a discrete importance-sampling table over
    binned timesteps. It adapts during training, but it introduces a second
    stateful thing to checkpoint, makes two runs with different bin counts
    incomparable, and in SD3's own ablations did not beat a fixed logit-normal.

    ``mean`` is the useful knob: negative values bias towards low noise (good
    for a detail-focused fine-tune), positive towards high noise (good when the
    model produces incoherent global structure).

    Args:
        mean: Mean of the underlying normal, in logit space.
        std: Standard deviation of the underlying normal.
    """

    mean: float = 0.0
    std: float = 1.0

    def __post_init__(self) -> None:
        """Validate the normal parameters."""
        require_finite("mean", self.mean)
        require_finite("std", self.std)
        if self.std <= 0.0:
            raise ValueError(f"std must be positive; got {self.std!r}")

    def sample(
        self,
        batch: int,
        *,
        device: torch.device | str,
        generator: torch.Generator,
        sequence_length: int | None = None,
    ) -> torch.Tensor:
        """Draw ``(batch,)`` logit-normal noise levels.

        Args:
            batch: Number of samples.
            device: Device the result must end up on.
            generator: Generator supplying the randomness.
            sequence_length: Ignored; use
                :class:`ShiftedLogitNormalSampler` to make the density depend on
                resolution.

        Returns:
            ``(batch,)`` float32 noise levels in ``(0, 1)``.
        """
        del sequence_length
        latent = normal((batch,), generator=generator, device=device)
        return torch.sigmoid(self.mean + self.std * latent)


@register_timestep_sampler("shifted_logit_normal")
@dataclass(frozen=True, slots=True)
class ShiftedLogitNormalSampler:
    """Logit-normal sampling with a sequence-length-dependent shift.

    This is the sampler a real high-resolution run should use, and the one worth
    understanding in full.

    Adding independent Gaussian noise to a token sequence destroys *relative*
    signal at a rate that grows with the sequence length. A 256px clip at 1k
    tokens and a 720p clip at 100k tokens, both noised to ``t = 0.7``, are not
    equally hard: the long sequence has far more redundancy between neighbouring
    tokens, so far more of the original structure survives. Train the long
    sequence on the same ``t`` distribution and it spends nearly its whole
    budget in a regime where the answer is already visible, then fails at
    inference exactly where the sampler starts — at pure noise, where it has
    barely trained.

    The correction is to push the sampled ``t`` towards 1 by an amount that
    grows with the sequence length::

        shift = lerp((base_seq_len, base_shift) -> (max_seq_len, max_shift))
        t'    = shift * t / (1 + (shift - 1) * t)

    The map is a monotone bijection fixing 0 and 1, so it re-weights the density
    without ever producing an out-of-range timestep, and the inference sampler
    can apply the identical map to its step schedule. Switching a 720p run from
    unshifted to shifted timesteps is, empirically, the largest single quality
    change available from the training objective — larger than most
    architectural changes of comparable cost.

    The defaults are the LTX-Video/Flux anchors. Note ``base_shift < 1``: at
    1024 tokens the correction is very slightly towards *low* noise, because
    that is the resolution the unshifted logit-normal density was fitted at.

    The alternative that lost was training a separate model per resolution.
    It sidesteps the problem entirely and is strictly worse: it forfeits the
    transfer from the cheap low-resolution regime, which is where most of the
    semantics are learned for a fraction of the FLOPs.

    Args:
        mean: Mean of the underlying normal, in logit space.
        std: Standard deviation of the underlying normal.
        base_seq_len: Token count of the low-resolution anchor.
        base_shift: Shift applied at ``base_seq_len``.
        max_seq_len: Token count of the high-resolution anchor.
        max_shift: Shift applied at ``max_seq_len``.
    """

    mean: float = 0.0
    std: float = 1.0
    base_seq_len: int = 1024
    base_shift: float = 0.95
    max_seq_len: int = 32768
    max_shift: float = 2.05

    def __post_init__(self) -> None:
        """Validate the normal parameters and the shift anchors."""
        require_finite("mean", self.mean)
        require_finite("std", self.std)
        if self.std <= 0.0:
            raise ValueError(f"std must be positive; got {self.std!r}")
        require_positive("base_seq_len", self.base_seq_len)
        require_positive("max_seq_len", self.max_seq_len)
        if self.base_seq_len >= self.max_seq_len:
            raise ValueError(
                "base_seq_len must be below max_seq_len; got "
                f"base_seq_len={self.base_seq_len!r}, max_seq_len={self.max_seq_len!r}"
            )
        for name in ("base_shift", "max_shift"):
            value = float(getattr(self, name))
            require_finite(name, value)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive; got {value!r}")

    def shift_for(self, sequence_length: int | None) -> float:
        """Return the shift strength this sampler would apply.

        Exposed because the inference sampler must apply the *same* shift to its
        step schedule; a mismatch between the training density and the sampling
        schedule shows up as washed-out or over-sharpened output that is very
        hard to attribute.

        Args:
            sequence_length: Tokens per sample, or ``None`` for the anchor.

        Returns:
            The shift strength.
        """
        return resolution_shift(
            sequence_length,
            base_seq_len=self.base_seq_len,
            base_shift=self.base_shift,
            max_seq_len=self.max_seq_len,
            max_shift=self.max_shift,
        )

    def sample(
        self,
        batch: int,
        *,
        device: torch.device | str,
        generator: torch.Generator,
        sequence_length: int | None = None,
    ) -> torch.Tensor:
        """Draw ``(batch,)`` shifted logit-normal noise levels.

        Args:
            batch: Number of samples.
            device: Device the result must end up on.
            generator: Generator supplying the randomness.
            sequence_length: Tokens the model will see per sample, *after*
                patchification and summed across generative modalities. Passing
                the pre-patch latent count instead silently under-shifts by the
                patch area, which for 2x2 patching is a factor of four in
                sequence length.

        Returns:
            ``(batch,)`` float32 noise levels in ``(0, 1)``.
        """
        latent = normal((batch,), generator=generator, device=device)
        timesteps = torch.sigmoid(self.mean + self.std * latent)
        return shift_timesteps(timesteps, self.shift_for(sequence_length))


@register_timestep_sampler("mode")
@dataclass(frozen=True, slots=True)
class ModeSampler:
    """SD3 mode sampling: a tunable mode with deliberately heavy tails.

    Logit-normal's tails vanish exponentially, so the endpoints are visited
    almost never. That is usually right, but it is fatal for a model that must
    also behave sensibly at ``t`` exactly 0 or 1 — an image-to-video model whose
    conditioning frames sit at ``t = 0``, or a distillation student trained to
    jump from pure noise in one step.

    Mode sampling keeps a controllable peak while retaining polynomial tails::

        u ~ U[0, 1)
        t = 1 - u - scale * (cos^2(pi u / 2) - 1 + u)

    ``scale = 0`` degenerates to uniform. Positive values move the mode towards
    high noise; negative towards low noise. SD3 found ``1.29`` best in its
    sweep, and that is the default here.

    Args:
        scale: Mode strength. Outside roughly ``[-1, 2.3]`` the map stops being
            monotone in ``u``, which folds the density back on itself.
    """

    scale: float = 1.29

    def __post_init__(self) -> None:
        """Validate that the mode scale keeps the map monotone."""
        require_finite("scale", self.scale)
        # d t / d u stays single-signed on [0, 1] only inside this window; past
        # it the map folds and two values of u give the same t, which quietly
        # doubles the density at the fold and halves it elsewhere.
        if not -1.0 <= self.scale <= 2.0 * math.pi / (math.pi - 2.0):
            raise ValueError(
                f"scale must be in [-1, {2.0 * math.pi / (math.pi - 2.0):.4f}] to keep "
                f"the mode map monotone; got {self.scale!r}"
            )

    def sample(
        self,
        batch: int,
        *,
        device: torch.device | str,
        generator: torch.Generator,
        sequence_length: int | None = None,
    ) -> torch.Tensor:
        """Draw ``(batch,)`` mode-sampled noise levels.

        Args:
            batch: Number of samples.
            device: Device the result must end up on.
            generator: Generator supplying the randomness.
            sequence_length: Ignored; mode sampling is resolution blind.

        Returns:
            ``(batch,)`` float32 noise levels in ``[0, 1]``.
        """
        del sequence_length
        draw = uniform((batch,), generator=generator, device=device)
        cosine = torch.cos(0.5 * math.pi * draw)
        timesteps = 1.0 - draw - self.scale * (cosine.pow(2) - 1.0 + draw)
        # Floating-point error at the endpoints can push the result a few ulp
        # outside the interval; the objective divides by (1 - t) in its SNR
        # weighting, so clamp rather than trusting exactness.
        return timesteps.clamp(0.0, 1.0)
