"""Noise schedules: the sequence of noise levels a sampler steps through.

avgen trains rectified flow, so there is one convention and everything obeys it:

* ``sigma`` is the flow time ``t`` in ``[0, 1]``. ``1`` is pure noise, ``0`` is
  clean. It is the same quantity as
  :attr:`~avgen.core.tokens.TokenStream.noise_level`.
* The noisy sample is the straight-line interpolant
  ``x_t = (1 - t) * clean + t * noise``.
* The model predicts the velocity ``v = noise - clean``, which is exactly
  ``dx/dt`` along that line.

A schedule is therefore just a decreasing list of ``t`` values from
``sigma_max`` down to ``0``, and a sampler integrates the ODE backwards along it.
This is *not* the variance-exploding sigma of an EDM model, and the two must not
be confused: a Karras sigma runs to 80 or higher, a flow sigma never exceeds 1.

Why the spacing matters at all, given that the ODE is exactly linear for a
perfectly trained rectified flow: the model is not perfectly trained, and its
error is not uniform in ``t``. Almost all of the perceptually important structure
is decided in the high-noise region, where the velocity field is still deciding
*what* the video is rather than sharpening it. Spending steps there and
economising near ``t = 0`` is what lets a 20-step sample look like a 50-step one.

.. warning::

   **The inference shift must equal the training shift.** ``shift`` reparametrises
   the time axis by ``t' = shift * t / (1 + (shift - 1) * t)``, and training uses
   it to decide which noise levels the model is *shown*. A model trained with
   ``shift=3`` has seen its high-noise region stretched; sampling it on an
   unshifted schedule steps through noise levels at a density the model was never
   trained for, and the result is systematically over- or under-denoised —
   washed-out and low-contrast when the inference shift is too low, oversaturated
   and over-sharpened when it is too high. It is not a subtle degradation and it
   is not a bug you will find by reading the sampler. Pin the shift next to the
   checkpoint, and if the schedule uses a sequence-length-dependent shift, use the
   same interpolation endpoints that :mod:`avgen.train.timestep` used.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, replace
from typing import Any

import torch

__all__ = [
    "ScheduleConfig",
    "SigmaSchedule",
    "apply_shift",
    "build_sigma_schedule",
    "karras_sigmas",
    "linear_quadratic_sigmas",
    "linear_sigmas",
    "list_sigma_schedules",
    "register_sigma_schedule",
    "resolution_shift",
]


def apply_shift(sigmas: torch.Tensor, shift: float) -> torch.Tensor:
    """Reparametrise flow time by the standard resolution shift.

    ``t' = shift * t / (1 + (shift - 1) * t)``.

    The map fixes both endpoints — ``0 -> 0`` and ``1 -> 1`` — and is strictly
    monotone for ``shift > 0``, so a shifted schedule is still a valid decreasing
    path from pure noise to clean. For ``shift > 1`` it pushes interior values
    *up*, concentrating the schedule in the high-noise region.

    The reason it exists is a property of high-resolution latents rather than of
    the sampler. Adding noise of a fixed variance destroys less information in a
    sequence of 100,000 tokens than in one of 1,000, because the redundancy
    between neighbouring tokens lets the signal be recovered from its neighbours.
    At a fixed ``t`` a long sequence is therefore *effectively less noisy*, and
    without the shift a high-resolution model spends most of its capacity on
    noise levels that are trivially easy.

    Args:
        sigmas: Flow times in ``[0, 1]``.
        shift: Shift factor. ``1.0`` is the identity.

    Returns:
        The shifted times, same shape and dtype.

    Raises:
        ValueError: If ``shift`` is not finite and positive.
    """
    if not math.isfinite(shift) or shift <= 0.0:
        raise ValueError(f"shift must be finite and positive; got {shift!r}")
    if shift == 1.0:
        return sigmas
    return shift * sigmas / (1.0 + (shift - 1.0) * sigmas)


def resolution_shift(
    sequence_length: int,
    *,
    base_length: int = 256,
    base_shift: float = 1.0,
    max_length: int = 4096,
    max_shift: float = 3.0,
    exponential: bool = False,
) -> float:
    """Interpolate a shift factor for a sequence length.

    Linear in the sequence length between the two anchor points, clamped outside
    them. This must reproduce whatever :mod:`avgen.train.timestep` did for the
    checkpoint being sampled — see the module warning.

    Args:
        sequence_length: Token count of the sample being generated.
        base_length: Sequence length at which ``base_shift`` applies.
        base_shift: Shift at ``base_length``.
        max_length: Sequence length at which ``max_shift`` applies.
        max_shift: Shift at ``max_length``.
        exponential: When true, the interpolated value is treated as the ``mu``
            of the Flux/SD3 parameterisation and exponentiated, so the anchors
            are ``log`` shifts rather than shifts. Use this only if the training
            run used that parameterisation; the two disagree by a lot.

    Returns:
        The shift factor to pass to :func:`apply_shift`.

    Raises:
        ValueError: If the anchors coincide or a length is not positive.
    """
    for name, value in (
        ("sequence_length", sequence_length),
        ("base_length", base_length),
        ("max_length", max_length),
    ):
        if isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer; got {value!r}")
    if base_length == max_length:
        raise ValueError(
            f"base_length and max_length must differ; both are {base_length}"
        )
    slope = (max_shift - base_shift) / (max_length - base_length)
    value = base_shift + slope * (sequence_length - base_length)
    low, high = min(base_shift, max_shift), max(base_shift, max_shift)
    value = min(max(value, low), high)
    return math.exp(value) if exponential else value


@dataclass(frozen=True, slots=True)
class SigmaSchedule:
    """The noise levels one sampling run steps through.

    Holds ``num_steps + 1`` values: a sampler consumes them pairwise, so step
    ``i`` moves the sample from ``sigmas[i]`` to ``sigmas[i + 1]``. Storing the
    boundaries rather than the steps is what lets an arbitrary sampler decide for
    itself how to traverse a pair, including the two-evaluation and multistep
    solvers that need the endpoints explicitly.

    The tail is exactly ``0.0``. Stopping short leaves residual noise that no
    later stage removes, and the difference between a final sigma of ``0.0`` and
    one of ``0.02`` is visible as a faint grain over the whole frame.

    Args:
        sigmas: ``(num_steps + 1,)`` float32, strictly decreasing, starting at
            most at 1.0 and ending at exactly 0.0.
        shift: The shift factor that was applied, recorded so it can be pinned
            into a generation config and compared against the checkpoint's.
        name: The schedule family, for the same reason.

    Raises:
        ValueError: If the tensor is not a rank-1 strictly-decreasing float
            sequence in ``[0, 1]`` ending at zero.
    """

    sigmas: torch.Tensor
    shift: float = 1.0
    name: str = "linear"

    def __post_init__(self) -> None:
        """Validate monotonicity and the endpoints once, at construction."""
        if self.sigmas.ndim != 1:
            raise ValueError(f"sigmas must have rank 1; got {tuple(self.sigmas.shape)}")
        if self.sigmas.numel() < 2:
            raise ValueError(
                f"sigmas must hold at least two boundaries (one step); got "
                f"{self.sigmas.numel()}"
            )
        if not self.sigmas.is_floating_point():
            raise TypeError(f"sigmas must be floating point; got {self.sigmas.dtype}")
        if not bool(torch.isfinite(self.sigmas).all()):
            raise ValueError("sigmas must be finite")
        differences = self.sigmas[1:] - self.sigmas[:-1]
        if not bool((differences < 0).all()):
            raise ValueError(
                "sigmas must be strictly decreasing; a flat or rising step makes "
                f"the solver stall or run backwards. Got {self.sigmas.tolist()}"
            )
        if float(self.sigmas[0]) > 1.0 or float(self.sigmas[-1]) != 0.0:
            raise ValueError(
                "sigmas must start at or below 1.0 and end at exactly 0.0; got "
                f"[{float(self.sigmas[0])}, ..., {float(self.sigmas[-1])}]"
            )

    @property
    def num_steps(self) -> int:
        """Number of sampler steps."""
        return int(self.sigmas.numel()) - 1

    @property
    def device(self) -> torch.device:
        """Device the schedule lives on."""
        return self.sigmas.device

    def __len__(self) -> int:
        """Number of sampler steps."""
        return self.num_steps

    def __iter__(self) -> Iterator[tuple[int, float, float]]:
        """Iterate ``(index, sigma, sigma_next)`` over the steps.

        Yields Python floats rather than tensors deliberately: a sampler that
        indexed a device tensor per step would force a host synchronisation on
        every step to make the value usable in control flow.

        Yields:
            One triple per step.
        """
        values = self.sigmas.tolist()
        for index in range(self.num_steps):
            yield index, float(values[index]), float(values[index + 1])

    def to(self, device: torch.device | str) -> SigmaSchedule:
        """Return a copy on another device.

        Args:
            device: Target device.

        Returns:
            The moved schedule.
        """
        return replace(self, sigmas=self.sigmas.to(device))

    def timesteps(self) -> torch.Tensor:
        """Return the noise levels the model is actually evaluated at.

        Returns:
            ``(num_steps,)`` — the boundaries excluding the final ``0.0``, at
            which no evaluation happens.
        """
        return self.sigmas[:-1]

    def tolist(self) -> list[float]:
        """Return the boundaries as Python floats."""
        return [float(value) for value in self.sigmas.tolist()]


_SCHEDULES: dict[str, Callable[..., torch.Tensor]] = {}


def register_sigma_schedule(
    name: str,
) -> Callable[[Callable[..., torch.Tensor]], Callable[..., torch.Tensor]]:
    """Register a schedule family under a name.

    Args:
        name: Registry key.

    Returns:
        A decorator that registers and returns the function unchanged.

    Raises:
        ValueError: If ``name`` is empty, untrimmed, or already registered.
            Silent replacement is refused because a config naming a schedule
            must resolve to the same curve for the life of a checkpoint.
    """
    if not name or name.strip() != name:
        raise ValueError(f"name must be a non-empty trimmed string; got {name!r}")
    if name in _SCHEDULES:
        raise ValueError(f"sigma schedule {name!r} is already registered")

    def decorator(builder: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        _SCHEDULES[name] = builder
        return builder

    return decorator


def list_sigma_schedules() -> tuple[str, ...]:
    """Return the registered schedule names, sorted."""
    return tuple(sorted(_SCHEDULES))


@register_sigma_schedule("linear")
def linear_sigmas(
    steps: int,
    *,
    sigma_max: float = 1.0,
    sigma_min: float | None = None,
    device: torch.device | str = "cpu",
    **_: Any,
) -> torch.Tensor:
    """Return evenly spaced flow times from ``sigma_max`` to zero.

    The natural schedule for rectified flow: the probability path is a straight
    line, so uniform spacing in ``t`` is uniform spacing along the path. Combined
    with a shift it is what most production flow models actually ship.

    Args:
        steps: Number of sampler steps.
        sigma_max: Starting noise level.
        sigma_min: Noise level of the last *evaluated* boundary. Defaults to
            ``sigma_max / steps``, which makes the final jump to zero the same
            size as every other step; that is the spacing every production flow
            sampler uses. Raising it truncates the schedule early and leaves the
            last step doing more work than the rest.
        device: Device to build on.
        **_: Ignored, so every family accepts the same keyword set.

    Returns:
        ``(steps + 1,)`` boundaries.

    Raises:
        ValueError: If ``steps`` is not positive or the range is invalid.
    """
    lowest = sigma_max / steps if sigma_min is None else sigma_min
    _require_range(steps, lowest, sigma_max)
    body = torch.linspace(sigma_max, lowest, steps, dtype=torch.float32, device=device)
    return _close(body)


@register_sigma_schedule("karras")
def karras_sigmas(
    steps: int,
    *,
    sigma_max: float = 1.0,
    sigma_min: float | None = None,
    rho: float = 7.0,
    device: torch.device | str = "cpu",
    **_: Any,
) -> torch.Tensor:
    """Return Karras-style ``rho``-spaced flow times.

    Karras et al. derived this spacing for variance-exploding diffusion by
    minimising the truncation error of the discretised ODE, and the result is a
    curve that is dense near ``sigma_min`` and sparse near ``sigma_max``. Applied
    to flow times it is a heuristic rather than a derivation — the underlying ODE
    is different — but it is a useful one, and it is what a user who types
    ``karras`` expects.

    Note that its bias is the *opposite* of the shift's: it concentrates steps at
    low noise. Stacking a large ``rho`` on a large ``shift`` mostly cancels, which
    is worth knowing before tuning both at once.

    Args:
        steps: Number of sampler steps.
        sigma_max: Starting noise level.
        sigma_min: Smallest evaluated noise level. Must be positive: the
            construction takes a root of it and zero would collapse the curve.
            Defaults to ``0.002``.
        rho: Curvature. Higher concentrates more strongly at low noise. ``7.0``
            is the value from the paper.
        device: Device to build on.
        **_: Ignored.

    Returns:
        ``(steps + 1,)`` boundaries.

    Raises:
        ValueError: If ``steps`` is not positive, ``sigma_min`` is not positive,
            or ``rho`` is not positive.
    """
    lowest = 0.002 if sigma_min is None else sigma_min
    _require_range(steps, lowest, sigma_max)
    if not math.isfinite(rho) or rho <= 0.0:
        raise ValueError(f"rho must be finite and positive; got {rho!r}")
    ramp = torch.linspace(0.0, 1.0, steps, dtype=torch.float32, device=device)
    min_inv = lowest ** (1.0 / rho)
    max_inv = sigma_max ** (1.0 / rho)
    body = (max_inv + ramp * (min_inv - max_inv)) ** rho
    return _close(body)


@register_sigma_schedule("linear_quadratic")
def linear_quadratic_sigmas(
    steps: int,
    *,
    sigma_max: float = 1.0,
    linear_steps: int | None = None,
    threshold_noise: float = 0.025,
    device: torch.device | str = "cpu",
    **_: Any,
) -> torch.Tensor:
    """Return a schedule that is linear at high noise and quadratic at low noise.

    Designed for the few-step regime, where the usual assumption behind uniform
    spacing breaks down. With 50 steps the discretisation error at any single
    step is small and the spacing barely matters; with 8 steps each step is a
    large jump and the solver's error is dominated by the low-noise end, where
    the velocity field changes fastest as the sample commits to detail.

    The fix is to spend the first half of the budget linearly across the
    high-noise region — which is nearly linear in ``t`` anyway for a rectified
    flow — and then compress the remaining noise range quadratically, so the last
    few steps take progressively smaller bites. This is the schedule LTX-Video
    introduced for its distilled few-step mode.

    Args:
        steps: Number of sampler steps.
        sigma_max: Starting noise level.
        linear_steps: Steps spent in the linear region. Defaults to half.
        threshold_noise: Noise level at which the linear region hands over to the
            quadratic one.
        device: Device to build on.
        **_: Ignored.

    Returns:
        ``(steps + 1,)`` boundaries.

    Raises:
        ValueError: If ``steps`` is not positive, ``linear_steps`` is out of
            range, or ``threshold_noise`` is not strictly between 0 and
            ``sigma_max``.
    """
    _require_range(steps, sigma_max / (steps + 1), sigma_max)
    split = max(steps // 2, 1) if linear_steps is None else linear_steps
    if isinstance(split, bool) or not 0 < split < steps:
        raise ValueError(f"linear_steps must be in (0, steps={steps}); got {split!r}")
    if not 0.0 < threshold_noise < sigma_max:
        raise ValueError(
            f"threshold_noise must be in (0, sigma_max={sigma_max}); "
            f"got {threshold_noise!r}"
        )
    # Built ascending in "noise added so far" and reversed at the end, because
    # the quadratic region is naturally expressed from the clean end.
    linear_slope = threshold_noise / split
    ascending = [index * linear_slope for index in range(split)]
    remaining = steps - split
    # Coefficients of the quadratic continuation, fixed by requiring that it
    # meets the linear region at (split, threshold_noise) with a matching value
    # and reaches sigma_max at the last boundary.
    span = steps - split
    quadratic_a = (sigma_max - threshold_noise) / (span * span)
    for index in range(remaining + 1):
        offset = index
        ascending.append(threshold_noise + quadratic_a * offset * offset)
    values = torch.tensor(ascending, dtype=torch.float32, device=device)
    descending = torch.flip(values, dims=(0,))
    # The construction ends at exactly 0.0 by design, but float arithmetic in the
    # linear region can leave a denormal; force it.
    descending = descending.clone()
    descending[-1] = 0.0
    return descending


def build_sigma_schedule(
    config: ScheduleConfig,
    *,
    sequence_length: int | None = None,
    device: torch.device | str = "cpu",
) -> SigmaSchedule:
    """Build a schedule from a config, resolving the shift.

    Args:
        config: The schedule configuration.
        sequence_length: Token count of the sample, required when
            ``config.dynamic_shift`` is set. This is the number the shift
            interpolation is a function of, and it must be the *global* sequence
            length, not a context-parallel shard's.
        device: Device to build the schedule on.

    Returns:
        The schedule, with the resolved shift recorded on it.

    Raises:
        KeyError: If the schedule name is not registered.
        ValueError: If a dynamic shift is requested without a sequence length.
    """
    builder = _SCHEDULES.get(config.name)
    if builder is None:
        raise KeyError(
            f"unknown sigma schedule {config.name!r}; registered: "
            f"{', '.join(list_sigma_schedules())}"
        )
    sigmas = builder(
        config.steps,
        sigma_max=config.sigma_max,
        sigma_min=config.sigma_min,
        rho=config.rho,
        linear_steps=config.linear_steps,
        threshold_noise=config.threshold_noise,
        device=device,
    )
    shift = config.resolve_shift(sequence_length)
    if shift != 1.0:
        shifted = apply_shift(sigmas, shift)
        # apply_shift fixes 0 and 1 exactly in exact arithmetic; in float32 the
        # endpoint can drift by an ulp, and the schedule contract requires an
        # exact zero.
        shifted = shifted.clone()
        shifted[-1] = 0.0
        sigmas = shifted
    return SigmaSchedule(sigmas=sigmas, shift=shift, name=config.name)


@dataclass(frozen=True, slots=True)
class ScheduleConfig:
    """Declarative description of a schedule, safe to pin into a run config.

    Args:
        name: Registered schedule family.
        steps: Number of sampler steps.
        shift: Static shift factor. Ignored when ``dynamic_shift`` is true.
        dynamic_shift: Whether to interpolate the shift from the sequence length
            instead of using ``shift``.
        base_length: Sequence length anchor for the dynamic shift.
        base_shift: Shift at ``base_length``.
        max_length: Second sequence length anchor.
        max_shift: Shift at ``max_length``.
        exponential_shift: Whether the dynamic anchors are ``log`` shifts.
        sigma_max: Starting noise level.
        sigma_min: Smallest evaluated noise level. ``None`` uses each family's
            own default.
        rho: Karras curvature.
        linear_steps: Linear-region length for ``linear_quadratic``.
        threshold_noise: Hand-over noise level for ``linear_quadratic``.

    Raises:
        ValueError: If ``steps`` is not positive or ``name`` is empty.
    """

    name: str = "linear"
    steps: int = 30
    shift: float = 1.0
    dynamic_shift: bool = False
    base_length: int = 256
    base_shift: float = 1.0
    max_length: int = 4096
    max_shift: float = 3.0
    exponential_shift: bool = False
    sigma_max: float = 1.0
    sigma_min: float | None = None
    rho: float = 7.0
    linear_steps: int | None = None
    threshold_noise: float = 0.025

    def __post_init__(self) -> None:
        """Validate the fields that do not depend on the family."""
        if not self.name or self.name.strip() != self.name:
            raise ValueError(
                f"name must be a non-empty trimmed string; got {self.name!r}"
            )
        if isinstance(self.steps, bool) or self.steps < 1:
            raise ValueError(f"steps must be a positive integer; got {self.steps!r}")
        if not math.isfinite(self.shift) or self.shift <= 0.0:
            raise ValueError(f"shift must be finite and positive; got {self.shift!r}")

    def resolve_shift(self, sequence_length: int | None) -> float:
        """Return the shift factor this config implies.

        Args:
            sequence_length: Global token count, required for a dynamic shift.

        Returns:
            The shift factor.

        Raises:
            ValueError: If a dynamic shift is requested without a length.
        """
        if not self.dynamic_shift:
            return self.shift
        if sequence_length is None:
            raise ValueError(
                "dynamic_shift requires a sequence_length; pass the global token "
                "count of the sample being generated"
            )
        return resolution_shift(
            sequence_length,
            base_length=self.base_length,
            base_shift=self.base_shift,
            max_length=self.max_length,
            max_shift=self.max_shift,
            exponential=self.exponential_shift,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation for pinning into a sample."""
        return {
            "name": self.name,
            "steps": self.steps,
            "shift": self.shift,
            "dynamic_shift": self.dynamic_shift,
            "base_length": self.base_length,
            "base_shift": self.base_shift,
            "max_length": self.max_length,
            "max_shift": self.max_shift,
            "exponential_shift": self.exponential_shift,
            "sigma_max": self.sigma_max,
            "sigma_min": self.sigma_min,
            "rho": self.rho,
            "linear_steps": self.linear_steps,
            "threshold_noise": self.threshold_noise,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> ScheduleConfig:
        """Rebuild a config from :meth:`to_dict` output.

        Args:
            values: The mapping.

        Returns:
            The restored config.
        """
        fields = {
            key: values[key]
            for key in (
                "name",
                "steps",
                "shift",
                "dynamic_shift",
                "base_length",
                "base_shift",
                "max_length",
                "max_shift",
                "exponential_shift",
                "sigma_max",
                "sigma_min",
                "rho",
                "linear_steps",
                "threshold_noise",
            )
            if key in values
        }
        return cls(**fields)


def _require_range(steps: int, sigma_min: float, sigma_max: float) -> None:
    """Validate the arguments every schedule family shares."""
    if isinstance(steps, bool) or steps < 1:
        raise ValueError(f"steps must be a positive integer; got {steps!r}")
    if not math.isfinite(sigma_max) or not 0.0 < sigma_max <= 1.0:
        raise ValueError(
            f"sigma_max must be in (0, 1] for a flow schedule; got {sigma_max!r}. "
            "Variance-exploding sigmas above 1 belong to EDM, not rectified flow."
        )
    if not math.isfinite(sigma_min) or not 0.0 < sigma_min < sigma_max:
        raise ValueError(
            f"sigma_min must be in (0, sigma_max={sigma_max}); got {sigma_min!r}. "
            "The terminal 0.0 boundary is appended separately, so the last "
            "evaluated noise level must be strictly positive."
        )


def _close(body: torch.Tensor) -> torch.Tensor:
    """Append the terminal zero boundary to a schedule body."""
    return torch.cat((body, body.new_zeros(1)))
