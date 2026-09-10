"""Persistence protocols and the mutable state a training step advances.

Everything that must survive a restart implements the same two-method contract,
which is deliberately the one PyTorch already uses
(``torch.distributed.checkpoint.stateful.Stateful``). A learning-rate schedule,
an exponential moving average, a dataloader cursor, and the model itself are all
just objects with ``state_dict`` and ``load_state_dict``, so the checkpoint layer
can save an open-ended set of components without knowing what any of them are.

That matters more at scale than it looks. A thousand-GPU run that resumes with
the correct weights but the wrong dataloader position silently retrains on data
it has already seen, and nothing in the loss curve will tell you.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from torch import nn
from torch.optim import Optimizer

from avgen.core.rng import RNGStreams

__all__ = [
    "EMA",
    "DataCursor",
    "LRSchedule",
    "Stateful",
    "TrainState",
]


@runtime_checkable
class Stateful(Protocol):
    """Minimal persistence contract shared by every checkpointable component.

    Structurally compatible with ``torch.distributed.checkpoint.Stateful``, so
    an object satisfying this protocol can be handed straight to DCP.
    """

    def state_dict(self) -> Mapping[str, Any]:
        """Return serialisable component state."""
        ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore component state."""
        ...


@runtime_checkable
class LRSchedule(Stateful, Protocol):
    """A checkpointable learning-rate schedule, advanced after an optimizer step."""

    def step(self) -> object:
        """Advance the schedule by one optimizer step."""
        ...

    def get_last_lr(self) -> list[float]:
        """Return the learning rate of each parameter group."""
        ...


@runtime_checkable
class EMA(Stateful, Protocol):
    """A checkpointable exponential moving average of model weights.

    EMA weights are what almost every published diffusion sample is drawn from;
    the raw training weights are noticeably worse. Treating EMA as a first-class
    checkpointable component rather than an afterthought is the difference
    between being able to reproduce a sample and not.
    """

    def update(self, model: nn.Module) -> object:
        """Fold the current model weights into the average."""
        ...


@runtime_checkable
class DataCursor(Stateful, Protocol):
    """A checkpointable position in a data stream.

    Resuming a run means resuming the *data*, not just the weights. A cursor
    that round-trips through a checkpoint is what makes a preempted job
    equivalent to an uninterrupted one.
    """

    def advance(self, samples: int) -> None:
        """Record that ``samples`` more samples have been consumed."""
        ...


@dataclass(slots=True)
class TrainState:
    """Mutable model, optimizer, RNG, and progress state for one training step.

    This is intentionally a plain mutable dataclass rather than a class with
    behaviour. The training step is a function of ``(state, batch, objective)``,
    which makes it trivially testable without constructing a trainer, a mesh, or
    a process group — and the ``Trainer`` in :mod:`avgen.train` is then a thin
    orchestration layer over it rather than a god object.

    Args:
        model: The module being trained. Under FSDP2 this is the sharded root
            module; under DDP it is the wrapper. Use
            :func:`avgen.parallel.unwrap_model` to reach the original.
        optimizer: The optimizer stepping ``model``.
        schedule: Optional learning-rate schedule.
        ema: Optional exponential moving average.
        rng: Purpose-separated random streams.
        step: Optimizer steps completed.
        samples_seen: Samples consumed across the whole job, not just this rank.
        tokens_seen: Generative tokens consumed across the whole job. The
            honest scaling-law x-axis for a video model, since samples of
            different resolution and duration are not comparable units.
        epoch: Passes completed over the dataset, when the dataset is finite.
    """

    model: nn.Module
    optimizer: Optimizer
    rng: RNGStreams
    schedule: LRSchedule | None = None
    ema: EMA | None = None
    step: int = 0
    samples_seen: int = 0
    tokens_seen: int = 0
    epoch: int = 0
    extras: dict[str, Stateful] = field(default_factory=dict)
    """Extra checkpointable components — a data cursor, a curriculum, a custom
    metric accumulator — saved and restored alongside the core state without the
    checkpoint layer needing to know what they are."""

    def validate(self) -> None:
        """Validate non-negative progress counters and coherent RNG devices.

        Raises:
            ValueError: If a counter is negative or the RNG streams straddle
                devices.
        """
        for name in ("step", "samples_seen", "tokens_seen", "epoch"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"{name} must be a non-negative integer; got {value!r}"
                )
        self.rng.validate()

    def progress(self) -> dict[str, int]:
        """Return the progress counters as a plain mapping."""
        return {
            "step": self.step,
            "samples_seen": self.samples_seen,
            "tokens_seen": self.tokens_seen,
            "epoch": self.epoch,
        }

    def load_progress(self, values: Mapping[str, int]) -> None:
        """Restore the progress counters from a mapping.

        Args:
            values: Mapping produced by :meth:`progress`.

        Raises:
            KeyError: If a counter is missing.
        """
        for name in ("step", "samples_seen", "tokens_seen", "epoch"):
            if name not in values:
                raise KeyError(f"missing progress counter {name!r}")
            setattr(self, name, int(values[name]))
