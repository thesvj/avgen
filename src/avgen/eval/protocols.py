"""The metric contract, the registry, and the accumulation base class.

**Why a streaming ``update``/``compute``/``reset`` contract rather than a
function.** An evaluation suite generates far more media than fits in memory,
and it runs across data-parallel ranks that must combine their partial results.
A metric that is a pure function of a whole dataset forces both problems on the
caller. A metric that accumulates sufficient statistics solves both: samples
stream past in microbatches, and two ranks' states merge by summing their
accumulators. Every built-in here therefore stores sums and counts, never the
samples.

**Why keyword-only inputs.** Metrics need different things — one wants video,
one wants video and audio, one wants a reference frame — and a positional
signature forces every caller to know which. ``update(**inputs)`` lets the suite
hand every metric the same bundle and lets each take what it needs. The cost is
that a typo in an input name would be silently ignored, so
:class:`RunningMetric` checks its required inputs are present and raises with
the metric name and the keys it actually received.

Tensor conventions, used by every built-in:

======================  =============================================
``video``               ``(batch, channels, frames, height, width)``
``audio``               ``(batch, channels, frames)``
``reference``           same shape as ``video``
``mask``                ``(batch, 1, frames, height, width)``, True where
                        the model generated the content
======================  =============================================

Metrics work on **either latents or decoded pixels**. Which one is correct is a
per-metric question and each one says so in its docstring; the two that are
pixel-only (:class:`~avgen.eval.video.SaturationClipping` and the sharpness
proxy's absolute scale) declare ``pixel_space_only`` so the suite can skip them
with a note instead of reporting a number that means nothing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any, Protocol, runtime_checkable

import torch

__all__ = [
    "Metric",
    "MetricError",
    "RunningMetric",
    "build_metric",
    "describe_metrics",
    "list_metrics",
    "metric_class",
    "register_metric",
    "require_audio",
    "require_video",
]


class MetricError(ValueError):
    """A metric was asked for something it cannot compute.

    Distinct from ``ValueError`` so the CLI can print it as one line: a missing
    input or a wrong tensor rank is a caller error, not a bug to traceback.
    """


@runtime_checkable
class Metric(Protocol):
    """Streaming evaluation metric.

    Implementations accumulate sufficient statistics in :meth:`update` and
    reduce them in :meth:`compute`. :meth:`compute` must be callable more than
    once without changing the result, so a report can be rendered and then
    written.
    """

    name: str

    def update(self, **inputs: Any) -> None:
        """Fold one batch of samples into the accumulator."""
        ...

    def compute(self) -> dict[str, float]:
        """Reduce the accumulator to named scalars."""
        ...

    def reset(self) -> None:
        """Discard all accumulated state."""
        ...


_REGISTRY: dict[str, type[RunningMetric]] = {}


def register_metric(name: str) -> Callable[[type[RunningMetric]], type[RunningMetric]]:
    """Register a metric class under a name.

    Args:
        name: Registry key, used in ``eval.metrics`` in a config file.

    Returns:
        A class decorator.

    Raises:
        ValueError: If the name is already taken. Silently replacing a metric
            means two runs can report the same metric name computed two
            different ways, which is worse than a crash at import time.
    """

    def decorate(cls: type[RunningMetric]) -> type[RunningMetric]:
        if name in _REGISTRY:
            raise ValueError(
                f"metric {name!r} is already registered to "
                f"{_REGISTRY[name].__name__}; pick a different name rather than "
                "shadowing it, or two runs will report incomparable numbers "
                "under one label"
            )
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return decorate


def list_metrics() -> tuple[str, ...]:
    """Return every registered dependency-free metric name, sorted.

    Returns:
        The names accepted by :func:`build_metric`.
    """
    return tuple(sorted(_REGISTRY))


def metric_class(name: str) -> type[RunningMetric]:
    """Look up a metric class by name.

    Args:
        name: Registry key.

    Returns:
        The class.

    Raises:
        MetricError: If the name is unknown.
    """
    if name not in _REGISTRY:
        raise MetricError(
            f"unknown metric {name!r}; available: {', '.join(list_metrics())}. "
            "Learned metrics (fvd, fid, clip_score, ...) live in "
            "avgen.eval.learned and are requested the same way."
        )
    return _REGISTRY[name]


def build_metric(name: str, **kwargs: Any) -> RunningMetric:
    """Construct a registered metric.

    Args:
        name: Registry key.
        **kwargs: Constructor arguments for that metric.

    Returns:
        A fresh metric instance.

    Raises:
        MetricError: If the name is unknown or the arguments are rejected.
    """
    cls = metric_class(name)
    try:
        return cls(**kwargs)
    except TypeError as error:
        raise MetricError(
            f"cannot build metric {name!r} with {kwargs!r}: {error}"
        ) from error


def describe_metrics() -> dict[str, str]:
    """Return one-line descriptions of every registered metric.

    Returns:
        Name to the first line of the class docstring.
    """
    described: dict[str, str] = {}
    for name, cls in sorted(_REGISTRY.items()):
        doc = (cls.__doc__ or "").strip().splitlines()
        described[name] = doc[0] if doc else ""
    return described


class RunningMetric(ABC):
    """Base class handling accumulation, input checking, and shape validation.

    Subclasses declare which inputs they need and implement
    :meth:`observe`, which returns a mapping of statistic name to a
    ``(sum, count)`` pair. The base class does the rest, so a new metric is
    usually fifteen lines and cannot get the averaging wrong.

    The alternative — each metric keeping a list of per-batch scalars and
    averaging at the end — was rejected because it averages *batches* rather
    than *samples*, so a final short batch is over-weighted. With a large
    evaluation set that is a fraction of a percent; with the 50-prompt sets
    people actually use it is several percent, which is the same size as the
    differences being reported.

    Args:
        device: Device the accumulators live on. Statistics are kept as fp64
            because an evaluation loop accumulates tens of thousands of terms
            and fp32 summation visibly drifts over that many additions.
    """

    #: Registry name, set by :func:`register_metric`.
    name: str = "unnamed"

    #: Input keys :meth:`observe` requires. Checked before every call.
    required_inputs: tuple[str, ...] = ("video",)

    #: Whether the metric is only meaningful on decoded pixels. The suite skips
    #: these with an explicit note when evaluating in latent space rather than
    #: reporting a number that does not mean what its name says.
    pixel_space_only: bool = False

    def __init__(self, *, device: torch.device | str = "cpu") -> None:
        self._device = torch.device(device)
        self._sums: dict[str, torch.Tensor] = {}
        self._counts: dict[str, torch.Tensor] = {}

    # -- subclass hook ---------------------------------------------------

    @abstractmethod
    def observe(self, **inputs: Any) -> Mapping[str, tuple[torch.Tensor, int]]:
        """Return per-statistic ``(sum, count)`` contributions for one batch.

        Args:
            **inputs: The tensors named in :attr:`required_inputs`, plus
                whatever else the caller supplied.

        Returns:
            Statistic name to a zero-dimensional sum and the number of items
            that sum covers.
        """

    # -- Metric protocol -------------------------------------------------

    def update(self, **inputs: Any) -> None:
        """Fold one batch into the accumulator.

        Args:
            **inputs: Named tensors; see the module docstring for shapes.

        Raises:
            MetricError: If a required input is absent.
        """
        missing = [key for key in self.required_inputs if inputs.get(key) is None]
        if missing:
            raise MetricError(
                f"metric {self.name!r} requires {', '.join(missing)} but was "
                f"given {', '.join(sorted(inputs)) or '<nothing>'}"
            )
        for key, (value, count) in self.observe(**inputs).items():
            if count <= 0:
                continue
            contribution = value.detach().to(self._device, torch.float64)
            if key in self._sums:
                self._sums[key] += contribution
                self._counts[key] += count
            else:
                self._sums[key] = contribution
                self._counts[key] = torch.tensor(
                    float(count), dtype=torch.float64, device=self._device
                )

    def compute(self) -> dict[str, float]:
        """Reduce accumulated statistics to named means.

        Returns:
            Statistic name to value. An empty mapping means :meth:`update` was
            never called with usable data — the suite reports that as a skipped
            metric rather than as a zero.
        """
        return {
            f"{self.name}/{key}": float(total / self._counts[key])
            for key, total in self._sums.items()
            if float(self._counts[key]) > 0
        }

    def reset(self) -> None:
        """Discard all accumulated state."""
        self._sums.clear()
        self._counts.clear()

    def state(self) -> dict[str, tuple[float, float]]:
        """Return raw ``(sum, count)`` pairs, for cross-rank reduction.

        A distributed suite gathers these from every data rank and sums them,
        which gives exactly the answer a single-rank run would have produced.
        Averaging per-rank ``compute()`` outputs instead is wrong whenever the
        ranks saw different sample counts, which they do as soon as the prompt
        count is not divisible by the world size.

        Returns:
            Statistic name to ``(sum, count)``.
        """
        return {
            key: (float(total), float(self._counts[key]))
            for key, total in self._sums.items()
        }

    def merge(self, other: Mapping[str, tuple[float, float]]) -> None:
        """Fold another rank's :meth:`state` into this one.

        Args:
            other: A state mapping from :meth:`state`.
        """
        for key, (total, count) in other.items():
            value = torch.tensor(total, dtype=torch.float64, device=self._device)
            if key in self._sums:
                self._sums[key] += value
                self._counts[key] += count
            else:
                self._sums[key] = value
                self._counts[key] = torch.tensor(
                    float(count), dtype=torch.float64, device=self._device
                )


def require_video(tensor: Any, *, metric: str, argument: str = "video") -> torch.Tensor:
    """Validate a 5-D video tensor and return it as float32.

    Args:
        tensor: Candidate tensor.
        metric: Metric name, for the error message.
        argument: Argument name, for the error message.

    Returns:
        The tensor, detached and in float32.

    Raises:
        MetricError: If the input is not a 5-D tensor.
    """
    if not isinstance(tensor, torch.Tensor):
        raise MetricError(
            f"{metric}: {argument} must be a tensor; got {type(tensor).__name__}"
        )
    if tensor.ndim != 5:
        raise MetricError(
            f"{metric}: {argument} must be (batch, channels, frames, height, "
            f"width); got shape {tuple(tensor.shape)}"
        )
    return tensor.detach().float()


def require_audio(tensor: Any, *, metric: str, argument: str = "audio") -> torch.Tensor:
    """Validate a 3-D audio tensor and return it as float32.

    Args:
        tensor: Candidate tensor.
        metric: Metric name, for the error message.
        argument: Argument name, for the error message.

    Returns:
        The tensor, detached and in float32.

    Raises:
        MetricError: If the input is not a 3-D tensor.
    """
    if not isinstance(tensor, torch.Tensor):
        raise MetricError(
            f"{metric}: {argument} must be a tensor; got {type(tensor).__name__}"
        )
    if tensor.ndim != 3:
        raise MetricError(
            f"{metric}: {argument} must be (batch, channels, frames); got shape "
            f"{tuple(tensor.shape)}"
        )
    return tensor.detach().float()
