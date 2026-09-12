"""Torch-free validators shared by every configuration dataclass.

:mod:`avgen.core._validate` does the same job for the tensor contracts, but it
imports ``torch`` at module scope. The configuration schema must stay importable
without torch — ``avgen plan`` and ``avgen --help`` are meant to work on a
laptop with nothing installed but ``pyyaml`` — so the handful of checks the
schema needs are duplicated here rather than borrowed.

Every message follows the same shape: the field name, what was required, and
the value that was rejected. A config error read off a cluster log at 3am is
only actionable if it says *which* field and *what* value.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = [
    "require_choice",
    "require_fraction",
    "require_non_negative",
    "require_non_negative_int",
    "require_path_like",
    "require_positive",
    "require_positive_int",
    "require_unique",
]


def require_positive_int(name: str, value: int) -> None:
    """Require a strictly positive, non-boolean integer.

    Args:
        name: Dotted field name echoed in the message.
        value: Candidate value.

    Raises:
        TypeError: If ``value`` is a bool or not an ``int``.
        ValueError: If ``value`` is below one.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer; got {value!r}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1; got {value!r}")


def require_non_negative_int(name: str, value: int) -> None:
    """Require a non-negative, non-boolean integer.

    Args:
        name: Dotted field name echoed in the message.
        value: Candidate value.

    Raises:
        TypeError: If ``value`` is a bool or not an ``int``.
        ValueError: If ``value`` is negative.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer; got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0; got {value!r}")


def require_positive(name: str, value: float) -> None:
    """Require a finite, strictly positive real number.

    Args:
        name: Dotted field name echoed in the message.
        value: Candidate value.

    Raises:
        TypeError: If ``value`` is not a real number.
        ValueError: If ``value`` is non-finite or non-positive.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number; got {value!r}")
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and > 0; got {value!r}")


def require_non_negative(name: str, value: float) -> None:
    """Require a finite, non-negative real number.

    Args:
        name: Dotted field name echoed in the message.
        value: Candidate value.

    Raises:
        TypeError: If ``value`` is not a real number.
        ValueError: If ``value`` is non-finite or negative.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number; got {value!r}")
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and >= 0; got {value!r}")


def require_fraction(name: str, value: float, *, upper: float = 1.0) -> None:
    """Require a value inside ``[0, upper]``.

    Args:
        name: Dotted field name echoed in the message.
        value: Candidate value.
        upper: Inclusive upper bound.

    Raises:
        TypeError: If ``value`` is not a real number.
        ValueError: If ``value`` falls outside the interval.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number; got {value!r}")
    if not math.isfinite(value) or not 0.0 <= value <= upper:
        raise ValueError(f"{name} must be in [0, {upper}]; got {value!r}")


def require_choice(name: str, value: str, options: Sequence[str]) -> None:
    """Require membership in a fixed set of names.

    Args:
        name: Dotted field name echoed in the message.
        value: Candidate value.
        options: Permitted values, listed in the message when rejected.

    Raises:
        ValueError: If ``value`` is not one of ``options``.
    """
    if value not in options:
        raise ValueError(
            f"{name} must be one of {', '.join(sorted(options))}; got {value!r}"
        )


def require_unique(name: str, values: Sequence[str]) -> None:
    """Require a sequence of names with no duplicates.

    Duplicates in a metric or logger list are always a copy-paste mistake, and
    silently de-duplicating them hides the mistake until someone wonders why
    their second entry has no effect.

    Args:
        name: Dotted field name echoed in the message.
        values: Candidate sequence.

    Raises:
        ValueError: If any value repeats.
    """
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"{name} contains a duplicate entry {value!r}")
        seen.add(value)


def require_path_like(name: str, value: str) -> None:
    """Require a non-empty, trimmed path string.

    Args:
        name: Dotted field name echoed in the message.
        value: Candidate value.

    Raises:
        TypeError: If ``value`` is not a string.
        ValueError: If ``value`` is empty or carries surrounding whitespace.
    """
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string path; got {value!r}")
    if not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty trimmed path; got {value!r}")
