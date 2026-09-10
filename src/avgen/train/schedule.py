"""Learning-rate schedules, and why the default is not cosine.

Every schedule here is a multiplier on the optimizer's configured learning rate,
expressed as a function of the step count and wrapped in ``LambdaLR``. Building
on the PyTorch scheduler rather than a bespoke class is deliberate: ``LambdaLR``
already satisfies :class:`~avgen.core.state.LRSchedule`, already round-trips
through a checkpoint, and already handles multiple parameter groups, so there is
nothing left for a custom class to add except a new set of bugs.

**Warmup exists because Adam's second moment starts wrong.** For the first few
hundred steps the running variance estimate is built from a handful of samples,
so the update size is essentially arbitrary — which for a deep pre-norm
transformer means the residual stream can be knocked far enough off scale that
the run never recovers. Warmup makes those steps small enough not to matter.

**Why WSD rather than cosine.** Cosine requires committing to ``total_steps``
before the run starts, and that commitment is load-bearing: the schedule's shape
at step 50k depends on whether you declared 100k or 500k steps. A real
pretraining run does not know its horizon. It gets extended because the loss is
still falling, or cut short because the cluster is needed, or branched to
compare two data mixtures at three budgets. Under cosine every one of those is a
different schedule and therefore a different, non-comparable run: extending past
the declared horizon leaves the learning rate pinned at the floor, and stopping
early leaves it un-annealed, which costs a large fraction of the final quality.

Warmup-stable-decay makes the horizon a *decision at the end*. The learning rate
holds flat through the stable phase, so a checkpoint from the middle of it is a
valid starting point for any budget; when a budget is chosen, a short decay
phase — ten percent of the steps is the usual figure — anneals to the floor and
recovers the quality cosine would have given. That yields several fully-annealed
models from one stable trunk, which is what makes a scaling study affordable,
and it matches or beats cosine at matched compute in every published comparison.
The cost is that the loss curve has a visible cliff at the decay, which looks
alarming and is not.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from avgen.core.state import LRSchedule

__all__ = [
    "build_schedule",
    "list_schedules",
    "register_schedule",
]

#: A schedule factory maps ``(total_steps, warmup_steps, options)`` to the
#: step-to-multiplier function ``LambdaLR`` consumes.
ScheduleFactory = Callable[[int, int, Mapping[str, Any]], Callable[[int], float]]

_SCHEDULES: dict[str, ScheduleFactory] = {}


def register_schedule(name: str) -> Callable[[ScheduleFactory], ScheduleFactory]:
    """Register a schedule factory under a configuration name.

    Args:
        name: Name used in configuration files and by :func:`build_schedule`.

    Returns:
        A decorator registering and returning the factory unchanged.

    Raises:
        ValueError: If ``name`` is already registered. Replacement is rejected
            so the active schedule can never depend on import order.
    """

    def decorate(factory: ScheduleFactory) -> ScheduleFactory:
        if name in _SCHEDULES:
            raise ValueError(f"schedule {name!r} is already registered")
        _SCHEDULES[name] = factory
        return factory

    return decorate


def list_schedules() -> tuple[str, ...]:
    """Return every registered schedule name, sorted.

    Returns:
        Sorted registered names.
    """
    return tuple(sorted(_SCHEDULES))


def build_schedule(
    name: str,
    optimizer: Optimizer,
    *,
    total_steps: int,
    warmup_steps: int = 0,
    **options: Any,
) -> LRSchedule:
    """Construct a learning-rate schedule for an optimizer.

    Args:
        name: One of :func:`list_schedules` — ``"constant"``, ``"linear"``,
            ``"cosine"``, or ``"wsd"``.
        optimizer: The optimizer whose groups are scheduled.
        total_steps: Planned optimizer steps. Used by every schedule except
            ``"constant"``. For ``"wsd"`` it decides only where the decay
            starts, so a run may safely exceed it — the learning rate simply
            stays at the floor.
        warmup_steps: Steps of linear warmup from zero.
        **options: Schedule-specific settings; see the individual factories.

    Returns:
        A checkpointable schedule.

    Raises:
        KeyError: If ``name`` is not registered.
        ValueError: If the step counts are inconsistent.
    """
    if name not in _SCHEDULES:
        available = ", ".join(list_schedules())
        raise KeyError(f"unknown schedule {name!r}; available: {available}")
    if isinstance(total_steps, bool) or total_steps < 1:
        raise ValueError(f"total_steps must be positive; got {total_steps!r}")
    if isinstance(warmup_steps, bool) or warmup_steps < 0:
        raise ValueError(f"warmup_steps must be non-negative; got {warmup_steps!r}")
    if warmup_steps >= total_steps:
        raise ValueError(
            f"warmup_steps={warmup_steps} must be below total_steps={total_steps}; "
            "a run that is entirely warmup never trains at its configured rate"
        )
    lambda_fn = _SCHEDULES[name](total_steps, warmup_steps, options)
    return LambdaLR(optimizer, lr_lambda=lambda_fn)


def _warmup_factor(step: int, warmup_steps: int) -> float:
    """Return the linear warmup multiplier for a step.

    The numerator is ``step + 1`` so the very first optimizer step has a small
    but non-zero learning rate. Starting at exactly zero wastes a step and, more
    annoyingly, makes the first logged learning rate zero, which reads as a
    misconfiguration.

    Args:
        step: Zero-based optimizer step.
        warmup_steps: Length of the warmup.

    Returns:
        A multiplier in ``(0, 1]``.
    """
    return float(step + 1) / float(max(warmup_steps, 1))


def _floor(options: Mapping[str, Any]) -> float:
    """Read and validate the ``min_lr_ratio`` option.

    Args:
        options: Schedule options.

    Returns:
        The floor as a fraction of the peak learning rate.

    Raises:
        ValueError: If the ratio is outside ``[0, 1]``.
    """
    ratio = float(options.get("min_lr_ratio", 0.0))
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"min_lr_ratio must be in [0, 1]; got {ratio!r}")
    return ratio


@register_schedule("constant")
def _constant(
    total_steps: int,
    warmup_steps: int,
    options: Mapping[str, Any],
) -> Callable[[int], float]:
    """Warm up, then hold the peak learning rate forever.

    The right choice for a fine-tune short enough that annealing would eat a
    meaningful fraction of it, and the right control when a schedule is
    suspected of causing an effect.

    Args:
        total_steps: Ignored.
        warmup_steps: Length of the linear warmup.
        options: Ignored.

    Returns:
        The step-to-multiplier function.
    """
    del total_steps, options

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return _warmup_factor(step, warmup_steps)
        return 1.0

    return multiplier


@register_schedule("linear")
def _linear(
    total_steps: int,
    warmup_steps: int,
    options: Mapping[str, Any],
) -> Callable[[int], float]:
    """Warm up, then decay linearly to the floor at ``total_steps``.

    Args:
        total_steps: Step at which the floor is reached.
        warmup_steps: Length of the linear warmup.
        options: ``min_lr_ratio`` — the floor as a fraction of the peak.

    Returns:
        The step-to-multiplier function.
    """
    floor = _floor(options)
    span = max(total_steps - warmup_steps, 1)

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return _warmup_factor(step, warmup_steps)
        progress = min(float(step - warmup_steps) / span, 1.0)
        return floor + (1.0 - floor) * (1.0 - progress)

    return multiplier


@register_schedule("cosine")
def _cosine(
    total_steps: int,
    warmup_steps: int,
    options: Mapping[str, Any],
) -> Callable[[int], float]:
    """Warm up, then follow a half cosine down to the floor.

    Spends more of the budget near the peak than linear does and anneals
    smoothly at the end, which is why it became the default everywhere. Its
    weakness is structural rather than numerical: the shape depends on
    ``total_steps``, so the schedule is only meaningful if the horizon is known
    in advance. See the module docstring and ``"wsd"``.

    Args:
        total_steps: Step at which the floor is reached.
        warmup_steps: Length of the linear warmup.
        options: ``min_lr_ratio`` — the floor as a fraction of the peak.

    Returns:
        The step-to-multiplier function.
    """
    floor = _floor(options)
    span = max(total_steps - warmup_steps, 1)

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return _warmup_factor(step, warmup_steps)
        progress = min(float(step - warmup_steps) / span, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return floor + (1.0 - floor) * cosine

    return multiplier


@register_schedule("wsd")
def _warmup_stable_decay(
    total_steps: int,
    warmup_steps: int,
    options: Mapping[str, Any],
) -> Callable[[int], float]:
    """Warm up, hold flat, then anneal over the final fraction of the budget.

    The stable phase is the point: every checkpoint taken during it is a valid
    trunk for *any* remaining budget, so one run can produce several
    fully-annealed models by branching a short decay wherever a budget lands.
    Cosine cannot do that — its shape is fixed by the horizon declared at step
    zero — and that is what makes it the wrong default for a run whose length is
    not known in advance, which is every pretraining run.

    ``decay_shape="1-sqrt"`` is the shape reported to anneal best in the WSD
    literature; it drops slowly at first and steepens, which spends more of the
    decay budget at a useful learning rate than a linear ramp does.

    Args:
        total_steps: Budget the decay is sized against. Exceeding it is safe:
            the multiplier stays at the floor.
        warmup_steps: Length of the linear warmup.
        options: ``decay_fraction`` (default ``0.1``), ``min_lr_ratio``
            (default ``0.0``), and ``decay_shape`` — ``"1-sqrt"`` (default),
            ``"linear"``, or ``"cosine"``.

    Returns:
        The step-to-multiplier function.

    Raises:
        ValueError: If the decay fraction or shape is invalid.
    """
    floor = _floor(options)
    fraction = float(options.get("decay_fraction", 0.1))
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"decay_fraction must be in (0, 1]; got {fraction!r}")
    shape = str(options.get("decay_shape", "1-sqrt"))
    if shape not in {"1-sqrt", "linear", "cosine"}:
        raise ValueError(
            f"decay_shape must be one of 1-sqrt, linear, cosine; got {shape!r}"
        )
    decay_steps = max(round(total_steps * fraction), 1)
    decay_start = max(total_steps - decay_steps, warmup_steps)
    span = max(total_steps - decay_start, 1)

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return _warmup_factor(step, warmup_steps)
        if step < decay_start:
            return 1.0
        progress = min(float(step - decay_start) / span, 1.0)
        if shape == "linear":
            remaining = 1.0 - progress
        elif shape == "cosine":
            remaining = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            remaining = 1.0 - math.sqrt(progress)
        return floor + (1.0 - floor) * remaining

    return multiplier
