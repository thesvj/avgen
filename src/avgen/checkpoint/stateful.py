"""Adapters that present avgen's training state to DCP as ``Stateful`` objects.

``torch.distributed.checkpoint`` has exactly one extension point: an object with
``state_dict``/``load_state_dict``. Give it those two methods and it will plan,
shard, write, and *reshard* the contents without knowing what they are. This
module is the thin layer that turns a :class:`~avgen.core.state.TrainState` into
a flat mapping of such objects.

Two properties of DCP drive every decision here.

**Only the top level is unwrapped.** ``dcp.save`` calls ``state_dict()`` on the
values of the *outermost* dict and nothing deeper. A ``Stateful`` buried inside
a nested dict is pickled as an opaque object instead of being sharded, which
silently produces a checkpoint that cannot be resharded. Hence
:func:`build_stateful` returns a *flat* ``dict[str, Stateful]``.

**Non-tensor values are deduplicated across ranks.** The default save planner
lets exactly one rank write each byte-serialised object. That is correct for
progress counters, which are identical everywhere, and *wrong* for RNG
generator state, which must differ per data-parallel rank. :class:`RNGState`
therefore namespaces its payload by ``data_rank`` so every rank writes a
distinct key — see its docstring for what happens on a resharded resume.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)
from torch.optim import Optimizer

from avgen.core.rng import RNGStreams
from avgen.core.state import Stateful, TrainState

__all__ = [
    "EMA_KEY",
    "EXTRAS_KEY",
    "MODEL_OPTIMIZER_KEY",
    "PROGRESS_KEY",
    "RNG_KEY",
    "SCHEDULE_KEY",
    "ExtrasState",
    "ModelOptimizerState",
    "ProgressState",
    "RNGState",
    "build_stateful",
]

#: Top-level checkpoint keys. They are part of the on-disk format: renaming one
#: makes every existing checkpoint unloadable, so they live here as constants
#: rather than as string literals scattered through the manager.
MODEL_OPTIMIZER_KEY = "model_optimizer"
RNG_KEY = "rng"
PROGRESS_KEY = "progress"
EXTRAS_KEY = "extras"

#: Key prefix for the optional components. Kept distinct from the required ones
#: so a checkpoint saved without an EMA can still be loaded by a run that has
#: one (the manager simply omits the key it cannot find).
SCHEDULE_KEY = "schedule"
EMA_KEY = "ema"


class ModelOptimizerState:
    """Model parameters and optimizer state as one resharding-safe unit.

    The model and the optimizer are adapted *together*, not separately, because
    optimizer state is only meaningful relative to the parameters it tracks.
    ``get_state_dict`` walks both at once and rewrites the optimizer's integer
    parameter indices into fully qualified parameter names; without that step an
    optimizer state dict is keyed by position in ``param_groups``, and any
    change to sharding, to parameter ordering, or to the rank count reassigns
    those positions and silently pairs Adam moments with the wrong weights.

    The rejected alternative was to save ``model.state_dict()`` and
    ``optimizer.state_dict()`` directly. It is simpler and it works on a single
    GPU, which is exactly why it is dangerous: it fails only once the run is
    large enough for FSDP to shard the parameters, and it fails by producing a
    checkpoint that loads without error and trains worse.

    Args:
        model: The (possibly sharded) module being trained.
        optimizer: The optimizer stepping ``model``.
        strict: Whether a key mismatch on load is an error. Turn it off only
            for a deliberate partial restore, such as warm-starting a model
            whose head has changed shape.
        cpu_offload: Whether to materialise the state dict on CPU. Halves the
            transient GPU memory of a save at the cost of a device-to-host copy;
            the async path stages to CPU anyway, so leave it off there.
        flatten_optimizer_state_dict: Whether to flatten the ``param_groups``
            nesting. Required for resharding — a nested optimizer state dict is
            stored as one opaque blob per group, and a blob cannot be split
            across a different number of ranks.
    """

    __slots__ = ("_model", "_optimizer", "_options")

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        *,
        strict: bool = True,
        cpu_offload: bool = False,
        flatten_optimizer_state_dict: bool = True,
    ) -> None:
        self._model = model
        self._optimizer = optimizer
        self._options = StateDictOptions(
            strict=strict,
            cpu_offload=cpu_offload,
            flatten_optimizer_state_dict=flatten_optimizer_state_dict,
        )

    @property
    def model(self) -> nn.Module:
        """The wrapped module."""
        return self._model

    @property
    def optimizer(self) -> Optimizer:
        """The wrapped optimizer."""
        return self._optimizer

    def state_dict(self) -> dict[str, Any]:
        """Return FQN-keyed model and optimizer state.

        Returns:
            Mapping with ``"model"`` and ``"optimizer"`` entries whose leaves
            are local tensors on a single device and ``DTensor`` shards under
            FSDP/TP. DCP writes only the local shard of each.
        """
        model_state, optimizer_state = get_state_dict(
            self._model, self._optimizer, options=self._options
        )
        return {"model": model_state, "optimizer": optimizer_state}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore model and optimizer in place.

        Args:
            state: Mapping produced by :meth:`state_dict`, already populated
                with this rank's shard by ``dcp.load``.

        Raises:
            KeyError: If either half is missing, which means the checkpoint was
                written by a different format version.
        """
        for name in ("model", "optimizer"):
            if name not in state:
                raise KeyError(
                    f"checkpoint entry {name!r} missing from {MODEL_OPTIMIZER_KEY!r}"
                )
        set_state_dict(
            self._model,
            self._optimizer,
            model_state_dict=state["model"],
            optim_state_dict=state["optimizer"],
            options=self._options,
        )


class RNGState:
    """Per-data-rank generator state, namespaced so ranks do not overwrite it.

    Generator state is a small CPU ``uint8`` tensor, so the temptation is to
    save it under a single key and be done. That is wrong at scale in a way
    that never raises: DCP deduplicates each key across ranks, so one rank's
    state would be written and *every* rank would restore it. All data-parallel
    ranks would then draw identical noise, the effective batch would collapse to
    one distinct sample repeated ``dp_size`` times, and the loss curve would
    look plausible while the run quietly stopped scaling.

    Namespacing by ``data_rank`` — and only ``data_rank``, matching
    :meth:`~avgen.core.rng.RNGStreams.for_rank` — gives each rank its own key.
    Tensor- and context-parallel ranks share a ``data_rank`` and therefore share
    a key, which is exactly right: they hold shards of the same sample and must
    draw the same noise.

    The cost is that this entry is the one part of the checkpoint that is *not*
    reshardable. A resume at a different data-parallel width has no key for its
    new ranks, so :class:`~avgen.checkpoint.manager.CheckpointManager` omits
    this component and re-derives the streams from the seed instead. That is a
    real, documented behaviour change on a resharded resume: the noise sequence
    differs from the one the original run would have drawn. Weights, optimizer
    state, and progress counters are unaffected.

    Args:
        rng: The live streams to save and restore.
        data_rank: This rank's index along the data-parallel axis, from
            :meth:`~avgen.parallel.dims.ParallelDims.data_coordinates`.

    Raises:
        ValueError: If ``data_rank`` is negative.
    """

    __slots__ = ("_data_rank", "_rng")

    def __init__(self, rng: RNGStreams, *, data_rank: int = 0) -> None:
        if isinstance(data_rank, bool) or data_rank < 0:
            raise ValueError(f"data_rank must be non-negative; got {data_rank!r}")
        self._rng = rng
        self._data_rank = data_rank

    @property
    def key(self) -> str:
        """The namespaced sub-key this rank owns inside the RNG entry."""
        return f"data_rank_{self._data_rank}"

    def state_dict(self) -> dict[str, Any]:
        """Return this rank's four generator states under its namespaced key."""
        return {self.key: dict(self._rng.state_dict())}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore this rank's generator states.

        A missing key is tolerated rather than fatal: the manager only includes
        this component when the data-parallel width matches, and a caller who
        bypasses the manager should get a re-seeded run rather than a crash.

        Args:
            state: Mapping produced by :meth:`state_dict`.
        """
        payload = state.get(self.key)
        if payload is None:
            return
        self._rng.load_state_dict(payload)


class ProgressState:
    """The four counters that say where in the run we are.

    These are plain Python integers, identical on every rank, so DCP stores them
    once as a byte blob. They are separated from the model entry because they
    are the part a human reads: ``latest_step`` and the resume banner come from
    here, and keeping them out of the tensor plan means they can be recovered
    from a checkpoint whose tensors are unreadable.

    Restoring ``samples_seen`` and ``tokens_seen`` matters as much as restoring
    ``step``. They are the x-axis of every scaling plot, and a resumed run that
    resets them produces a curve that cannot be compared against the run it
    continues.

    Args:
        state: The training state whose counters are saved and restored.
    """

    __slots__ = ("_state",)

    def __init__(self, state: TrainState) -> None:
        self._state = state

    def state_dict(self) -> dict[str, Any]:
        """Return the progress counters."""
        return dict(self._state.progress())

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore the progress counters.

        Args:
            state: Mapping produced by :meth:`state_dict`.

        Raises:
            KeyError: If a counter is missing from the checkpoint.
        """
        self._state.load_progress(state)


class ExtrasState:
    """Passthrough for ``TrainState.extras`` — components the layer never names.

    A data cursor, a curriculum schedule, a fault-tolerance token: the
    checkpoint layer must persist them without depending on them. Each value is
    already :class:`~avgen.core.state.Stateful`, so this class only has to keep
    them addressable by name and iterate them in a stable order.

    The ordering is not cosmetic. DCP builds its save plan from the traversal
    order of the state dict, and plans must agree across ranks; a ``dict`` built
    from an unordered source (a ``set``, a hash-ordered scan) can enumerate
    differently on different ranks and hang the collective. Sorting the names
    here makes that impossible.

    Args:
        extras: The named checkpointable components.
    """

    __slots__ = ("_extras",)

    def __init__(self, extras: Mapping[str, Stateful]) -> None:
        self._extras = dict(extras)

    def state_dict(self) -> dict[str, Any]:
        """Return each component's state, keyed by name in sorted order."""
        return {
            name: dict(self._extras[name].state_dict()) for name in sorted(self._extras)
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore each component present in both the checkpoint and the run.

        A component in the checkpoint but not in the run is ignored, and vice
        versa. That asymmetry is deliberate: adding an extra mid-project should
        not invalidate every checkpoint written before it existed.

        Args:
            state: Mapping produced by :meth:`state_dict`.
        """
        for name in sorted(self._extras):
            payload = state.get(name)
            if payload is not None:
                self._extras[name].load_state_dict(payload)


def build_stateful(
    state: TrainState,
    *,
    data_rank: int = 0,
    include_rng: bool = True,
    strict: bool = True,
    cpu_offload: bool = False,
) -> dict[str, object]:
    """Adapt a :class:`~avgen.core.state.TrainState` into DCP's flat contract.

    Optional components are included only when present, so a run without an EMA
    writes no EMA key at all rather than an empty one. The manager compares the
    key sets on load and reports what it skipped.

    Args:
        state: The live training state.
        data_rank: Index along the data-parallel axis, used to namespace RNG.
        include_rng: Whether to save or restore generator state. The manager
            sets this to false when resuming at a different data-parallel width,
            where per-rank streams cannot be resharded.
        strict: Whether model/optimizer key mismatches are fatal.
        cpu_offload: Whether to materialise model/optimizer state on CPU.

    Returns:
        A flat mapping from checkpoint key to a DCP-compatible ``Stateful``.
    """
    entries: dict[str, object] = {
        MODEL_OPTIMIZER_KEY: ModelOptimizerState(
            state.model,
            state.optimizer,
            strict=strict,
            cpu_offload=cpu_offload,
        ),
        PROGRESS_KEY: ProgressState(state),
    }
    if include_rng:
        entries[RNG_KEY] = RNGState(state.rng, data_rank=data_rank)
    if state.schedule is not None:
        entries[SCHEDULE_KEY] = state.schedule
    if state.ema is not None:
        entries[EMA_KEY] = state.ema
    if state.extras:
        entries[EXTRAS_KEY] = ExtrasState(state.extras)
    return entries
