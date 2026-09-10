"""Shared argument validators used across the tensor and batch contracts.

Every public dataclass in :mod:`avgen.core` validates its own arguments at the
boundary rather than trusting the caller. The helpers here keep the error
messages uniform: they always name the offending field and echo the value that
was rejected, which is what makes a failure at rank 743 of a 1024-rank job
diagnosable from a log line.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

__all__ = [
    "require_dimension",
    "require_dtype",
    "require_finite",
    "require_floating",
    "require_name",
    "require_positive",
    "require_probability",
    "require_same_device",
    "require_shape",
    "require_unique_names",
]


def require_positive(name: str, value: int) -> None:
    """Reject non-positive or boolean integers with a precise message.

    Args:
        name: Field name echoed in the error message.
        value: Candidate value.

    Raises:
        ValueError: If ``value`` is a bool or is less than one.
    """
    if isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")


def require_dimension(name: str, value: int, *, allow_zero: bool = False) -> None:
    """Reject invalid tensor dimensions.

    Args:
        name: Field name echoed in the error message.
        value: Candidate dimension.
        allow_zero: Whether a zero-length dimension is admissible. Zero is legal
            for optional modalities (a video-only batch carries zero audio
            frames) but never for a batch size.

    Raises:
        ValueError: If ``value`` is a bool or below the permitted lower bound.
    """
    lower = 0 if allow_zero else 1
    if isinstance(value, bool) or value < lower:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer; got {value!r}")


def require_name(name: str, value: str) -> None:
    """Require a non-empty string with no leading or trailing whitespace.

    Args:
        name: Field name echoed in the error message.
        value: Candidate string.

    Raises:
        ValueError: If ``value`` is empty or is not already trimmed.
    """
    if not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty trimmed string; got {value!r}")


def require_unique_names(name: str, values: Sequence[str]) -> None:
    """Require every entry to be a valid, distinct name.

    Args:
        name: Collection name echoed in the error message.
        values: Candidate names.

    Raises:
        ValueError: If any entry is invalid or appears more than once.
    """
    for index, value in enumerate(values):
        require_name(f"{name}[{index}]", value)
    duplicates = sorted({value for value in values if list(values).count(value) > 1})
    if duplicates:
        raise ValueError(f"{name} must be unique; duplicates: {', '.join(duplicates)}")


def require_finite(name: str, value: float) -> None:
    """Require a finite float.

    Args:
        name: Field name echoed in the error message.
        value: Candidate value.

    Raises:
        ValueError: If ``value`` is NaN or infinite.
    """
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite; got {value!r}")


def require_probability(name: str, value: float) -> None:
    """Require a finite float in ``[0, 1]``.

    Args:
        name: Field name echoed in the error message.
        value: Candidate value.

    Raises:
        ValueError: If ``value`` is not finite or falls outside ``[0, 1]``.
    """
    require_finite(name, value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]; got {value!r}")


def require_shape(
    name: str,
    tensor: torch.Tensor,
    expected: tuple[int, ...],
) -> None:
    """Require an exact tensor shape.

    Args:
        name: Tensor name echoed in the error message.
        tensor: Candidate tensor.
        expected: Required shape.

    Raises:
        ValueError: If the observed shape differs from ``expected``.
    """
    observed = tuple(tensor.shape)
    if observed != expected:
        raise ValueError(f"{name} shape must be {expected}; got {observed}")


def require_floating(name: str, tensor: torch.Tensor) -> None:
    """Require a floating-point tensor.

    Args:
        name: Tensor name echoed in the error message.
        tensor: Candidate tensor.

    Raises:
        TypeError: If the tensor is not floating point.
    """
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must be floating point; got {tensor.dtype}")


def require_dtype(name: str, tensor: torch.Tensor, expected: torch.dtype) -> None:
    """Require an exact tensor dtype.

    Args:
        name: Tensor name echoed in the error message.
        tensor: Candidate tensor.
        expected: Required dtype.

    Raises:
        TypeError: If the observed dtype differs from ``expected``.
    """
    if tensor.dtype is not expected:
        raise TypeError(f"{name} dtype must be {expected}; got {tensor.dtype}")


def require_same_device(
    reference_name: str,
    reference: torch.Tensor,
    tensors: Sequence[tuple[str, torch.Tensor]],
) -> None:
    """Require every tensor to live on the reference tensor's device.

    A silent host-to-device copy inside a training step is one of the most
    expensive mistakes available at scale, so device coherence is checked at the
    contract boundary instead of being fixed up implicitly.

    Args:
        reference_name: Name of the reference tensor.
        reference: Tensor whose device is authoritative.
        tensors: ``(name, tensor)`` pairs to check.

    Raises:
        ValueError: If any tensor lives on a different device.
    """
    for name, tensor in tensors:
        if tensor.device != reference.device:
            raise ValueError(
                f"{name} must be on {reference_name} device {reference.device}; "
                f"got {tensor.device}"
            )
