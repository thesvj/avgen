"""Generator-bound draws that survive a device mismatch.

Every random draw in training must come from an explicit
:class:`torch.Generator` so that a run is reproducible from its seed alone.
PyTorch makes that awkward in one specific way: a generator is *device bound*,
and ``torch.rand(..., generator=g, device=d)`` raises when ``g.device != d``.

A trainer hits that constantly. :class:`~avgen.core.rng.RNGStreams` is often
built on CPU — it is created before the device is chosen, it round-trips through
a checkpoint as CPU byte tensors, and CPU generators give bit-identical draws
across GPU models and driver versions, which a CUDA generator does not. The
tensors being filled, meanwhile, live on the accelerator.

The helpers here resolve that by drawing on the generator's own device and
moving the result. The moved tensors are per-sample or per-token metadata —
timesteps, dropout masks, task labels — measured in kilobytes, so the copy is
free relative to a step. The one draw where this would *not* be acceptable is
the noise tensor itself, which is the size of the activations; that one is
drawn directly on the compute device, and the objective documents the
consequence.
"""

from __future__ import annotations

import torch

__all__ = ["bernoulli", "categorical", "normal", "uniform"]


def _relocate(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move a freshly drawn tensor onto the target device when it differs."""
    return value if value.device == device else value.to(device)


def uniform(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Draw from ``U[0, 1)``.

    Args:
        shape: Shape of the result.
        generator: Generator supplying the randomness.
        device: Device the result must end up on.
        dtype: Floating dtype of the result.

    Returns:
        A tensor of the requested shape on ``device``.
    """
    target = torch.device(device)
    value = torch.rand(shape, generator=generator, device=generator.device, dtype=dtype)
    return _relocate(value, target)


def normal(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Draw from the standard normal distribution.

    Args:
        shape: Shape of the result.
        generator: Generator supplying the randomness.
        device: Device the result must end up on.
        dtype: Floating dtype of the result.

    Returns:
        A tensor of the requested shape on ``device``.
    """
    target = torch.device(device)
    value = torch.randn(
        shape, generator=generator, device=generator.device, dtype=dtype
    )
    return _relocate(value, target)


def bernoulli(
    shape: tuple[int, ...],
    probability: float,
    *,
    generator: torch.Generator,
    device: torch.device | str,
) -> torch.Tensor:
    """Draw an independent boolean mask that is true with a fixed probability.

    Args:
        shape: Shape of the result.
        probability: Probability of ``True`` per element.
        generator: Generator supplying the randomness.
        device: Device the result must end up on.

    Returns:
        A boolean tensor of the requested shape on ``device``.
    """
    draw = uniform(shape, generator=generator, device=device)
    # Strict ``<`` rather than ``<=`` so probability 0.0 is exactly never, which
    # a caller disabling dropout is entitled to rely on.
    return draw < probability


def categorical(
    weights: torch.Tensor,
    count: int,
    *,
    generator: torch.Generator,
    device: torch.device | str,
) -> torch.Tensor:
    """Sample category indices proportionally to non-negative weights.

    Args:
        weights: ``(categories,)`` non-negative, not necessarily normalised.
        count: Number of independent samples to draw.
        generator: Generator supplying the randomness.
        device: Device the result must end up on.

    Returns:
        ``(count,)`` int64 indices on ``device``.

    Raises:
        ValueError: If every weight is zero, which would leave nothing to draw.
    """
    source = weights.to(generator.device, dtype=torch.float32)
    total = float(source.sum())
    if total <= 0.0:
        raise ValueError("categorical weights must contain at least one positive entry")
    drawn = torch.multinomial(source, count, replacement=True, generator=generator)
    return _relocate(drawn.to(torch.int64), torch.device(device))
