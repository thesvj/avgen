"""The training step, and the loop that drives it.

The step is a **function**, not a method: ``train_step(state, batch, objective)``
takes a :class:`~avgen.core.state.TrainState` and returns the advanced state and
the metrics. That shape is what makes a trainer testable — a test constructs a
state, a batch, and an objective and calls one function, with no mesh, no
process group, and no config tree. :class:`Trainer` is a thin orchestrator over
it that owns cadence, callbacks, and the curriculum, and nothing else.

Two decisions in the step are worth explaining, because both are consequences of
scale rather than taste.

**Nothing synchronises with the host.** Reading a device scalar with ``.item()``
stalls the pipeline until the GPU drains, and at a thousand ranks every rank
then waits for the slowest one; done once per step, it is a measurable fraction
of throughput. So the loss, the gradient norm, and the finiteness flags all stay
as zero-dimensional device tensors, and the only synchronisation is
:meth:`~avgen.core.metrics.StepMetrics.to_mapping` at logging cadence.

**A bad microbatch neutralises its step instead of ending the run.** One
non-finite activation somewhere in a thousand-GPU job is routine — a corrupt
sample, a transient ECC error, a rare overflow at an extreme timestep — and a
crash there costs the whole run. But *skipping* the optimizer step would require
a host-side branch on a device value, which is the synchronisation just ruled
out. The resolution is to neutralise the update on device: the gradients are
replaced with zeros when the norm is non-finite, so the step still executes but
moves the weights only by one step of weight decay and moment decay, and the
event is recorded in ``StepMetrics.nonfinite`` and ``StepMetrics.skipped`` for
the logger to surface. The residual is negligible; a NaN reaching the weights is
not, because it is unrecoverable and silent until the loss curve flatlines.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch
from torch.distributed.device_mesh import DeviceMesh

from avgen.core._validate import require_name, require_positive
from avgen.core.batch import MediaBatch
from avgen.core.metrics import StepMetrics
from avgen.core.patchify import GridPatchifier, Patchifier
from avgen.core.state import TrainState
from avgen.parallel.apply import ParallelModel
from avgen.parallel.comm import clip_grad_norm
from avgen.parallel.env import unwrap_model
from avgen.train.objective import Objective, ObjectiveOutput

__all__ = [
    "Curriculum",
    "CurriculumStage",
    "MetricSink",
    "ResolutionBucket",
    "Trainer",
    "TrainerCallback",
    "TrainerConfig",
    "train_step",
]

_LOGGER = logging.getLogger("avgen.train")


@runtime_checkable
class MetricSink(Protocol):
    """Where step metrics go.

    Declared structurally here rather than imported from :mod:`avgen.telemetry`
    so that the training subsystem has no build-order dependency on the
    telemetry subsystem. Any object with this method — including
    ``avgen.telemetry.Logger`` — satisfies it.
    """

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Record one metric snapshot."""
        ...


@runtime_checkable
class TrainerCallback(Protocol):
    """A hook invoked at fixed points of the loop.

    A list of small objects rather than a base class to subclass. Inheritance
    would force every hook through one type, so a checkpointer and a profiler
    could not be composed without a diamond, and overriding a method would
    silently replace behaviour rather than adding to it. Callbacks are also
    *partial*: implement only the methods you need and the dispatcher skips the
    rest.
    """

    def on_step_end(
        self, trainer: Trainer, state: TrainState, metrics: StepMetrics
    ) -> None:
        """Called after every optimizer step."""
        ...

    def on_checkpoint(self, trainer: Trainer, state: TrainState) -> None:
        """Called when the checkpoint cadence fires."""
        ...

    def on_eval(self, trainer: Trainer, state: TrainState) -> None:
        """Called when the evaluation cadence fires."""
        ...


@dataclass(frozen=True, slots=True)
class ResolutionBucket:
    """One resolution and duration the curriculum can ask the loader for.

    Args:
        name: Stable identifier, used in logs and by the data subsystem to
            resolve an actual bucket.
        frames: Latent frames per sample.
        height: Latent rows per sample.
        width: Latent columns per sample.
    """

    name: str
    frames: int
    height: int
    width: int

    def __post_init__(self) -> None:
        """Validate the identifier and the extents."""
        require_name("name", self.name)
        for field_name in ("frames", "height", "width"):
            require_positive(field_name, getattr(self, field_name))

    @property
    def latents(self) -> int:
        """Latent positions per sample, before patchification."""
        return self.frames * self.height * self.width


@dataclass(frozen=True, slots=True)
class CurriculumStage:
    """A phase of the progressive-resolution curriculum.

    Args:
        name: Stage identifier for logs.
        until_step: The stage is active while ``state.step < until_step``.
        buckets: ``(bucket, weight)`` pairs sampled within the stage.
    """

    name: str
    until_step: int
    buckets: tuple[tuple[ResolutionBucket, float], ...]

    def __post_init__(self) -> None:
        """Validate the identifier, the boundary, and the bucket weights."""
        require_name("name", self.name)
        require_positive("until_step", self.until_step)
        if not self.buckets:
            raise ValueError(f"stage {self.name!r} must declare at least one bucket")
        if sum(weight for _, weight in self.buckets) <= 0.0:
            raise ValueError(f"stage {self.name!r} must have a positive total weight")
        for bucket, weight in self.buckets:
            if weight < 0.0:
                raise ValueError(
                    f"weight for {bucket.name!r} in stage {self.name!r} must be "
                    f">= 0; got {weight!r}"
                )


@dataclass(frozen=True, slots=True)
class Curriculum:
    """An ordered progressive-resolution schedule.

    Training a video model at its target resolution from step zero wastes most
    of the budget. Semantics — what a dog looks like, how a camera pans, what
    follows a prompt — are learned almost as well at 256px as at 720p, and cost
    an order of magnitude less per step because attention is quadratic in the
    token count. So the standard recipe is to spend most steps low and cheap,
    then move up.

    Each stage keeps a *weighted mixture* rather than a single resolution, and
    that is the part that is easy to drop and expensive to lose: a stage that
    abandons the previous resolution entirely produces a model that has
    forgotten it, which is visible immediately if the model is ever asked for a
    shorter or smaller clip. Retaining ten to twenty percent of the earlier
    bucket costs almost nothing and keeps the whole range usable.

    The mixture is *declared* here and *realised* by the data subsystem, which
    owns bucketing. The trainer hands the active stage to any loader exposing
    ``set_curriculum``; a loader without it is unaffected and the stage is
    advisory.

    Args:
        stages: Stages in ascending ``until_step`` order.
    """

    stages: tuple[CurriculumStage, ...] = ()

    def __post_init__(self) -> None:
        """Validate that the stage boundaries are strictly increasing."""
        boundaries = [stage.until_step for stage in self.stages]
        if boundaries != sorted(set(boundaries)):
            raise ValueError(
                "curriculum stages must have strictly increasing until_step; got "
                f"{boundaries}"
            )

    @property
    def is_empty(self) -> bool:
        """Whether no curriculum is configured."""
        return not self.stages

    def stage_for(self, step: int) -> CurriculumStage | None:
        """Return the stage active at a step.

        Args:
            step: Optimizer steps completed.

        Returns:
            The active stage, the final stage once every boundary has passed,
            or ``None`` when no curriculum is configured.
        """
        if not self.stages:
            return None
        for stage in self.stages:
            if step < stage.until_step:
                return stage
        # Past the last boundary the run continues at the final mixture rather
        # than stopping or falling back to the first stage.
        return self.stages[-1]


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    """Cadences, clipping, and curriculum for a run.

    Args:
        gradient_accumulation_steps: Microbatches per optimizer step. Use
            :meth:`~avgen.parallel.dims.ParallelDims.gradient_accumulation_for`
            to derive it from a global batch size rather than setting it by
            hand.
        max_grad_norm: Global gradient-norm clip, or ``None`` to disable. Never
            disable it on a long run: a single outlier batch can produce an
            update large enough to destroy the model, and the clip is the only
            thing standing between that batch and a restart from the last
            checkpoint.
        log_every: Steps between metric syncs. This is the only place the
            training loop synchronises with the host, so a small value is a
            real throughput cost.
        eval_every: Steps between evaluation callbacks; ``0`` disables.
        checkpoint_every: Steps between checkpoint callbacks; ``0`` disables.
        autocast_dtype: Mixed-precision dtype for the forward pass on CUDA.
            bfloat16 rather than float16 because it has the same exponent range
            as fp32, so the flow-matching loss cannot overflow at extreme
            timesteps and no gradient scaler is needed.
        curriculum: Progressive-resolution schedule.
        data_world: Number of data-parallel ranks, used to convert this rank's
            sample and token counts into job-wide totals.
    """

    gradient_accumulation_steps: int = 1
    max_grad_norm: float | None = 1.0
    log_every: int = 10
    eval_every: int = 0
    checkpoint_every: int = 0
    autocast_dtype: str = "bfloat16"
    curriculum: Curriculum = field(default_factory=Curriculum)
    data_world: int = 1

    def __post_init__(self) -> None:
        """Validate cadences, the clip threshold, and the autocast dtype."""
        require_positive(
            "gradient_accumulation_steps", self.gradient_accumulation_steps
        )
        require_positive("log_every", self.log_every)
        require_positive("data_world", self.data_world)
        for name in ("eval_every", "checkpoint_every"):
            value = getattr(self, name)
            if isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be non-negative; got {value!r}")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0.0:
            raise ValueError(
                f"max_grad_norm must be positive or None; got {self.max_grad_norm!r}"
            )
        if not hasattr(torch, self.autocast_dtype):
            raise ValueError(f"unknown autocast dtype {self.autocast_dtype!r}")

    @property
    def torch_autocast_dtype(self) -> torch.dtype:
        """The autocast dtype as a ``torch.dtype``."""
        resolved = getattr(torch, self.autocast_dtype)
        if not isinstance(resolved, torch.dtype):
            raise TypeError(f"{self.autocast_dtype!r} does not name a torch dtype")
        return resolved


def _autocast(device: torch.device, dtype: torch.dtype) -> Any:
    """Return the mixed-precision context for a device.

    Autocast is enabled on CUDA only. On CPU a bf16 forward is emulated and
    slower than fp32, so a CPU smoke test would be measuring the emulation
    rather than the code.

    Args:
        device: Device the step runs on.
        dtype: Autocast dtype.

    Returns:
        A context manager.
    """
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def _set_gradient_sync(model: torch.nn.Module, enabled: bool) -> None:
    """Enable or disable gradient synchronisation for accumulation.

    FSDP2 reduce-scatters gradients at the end of every backward. During
    gradient accumulation all but the last microbatch would pay for a reduction
    whose result is immediately added to, so disabling it turns an
    ``accumulation``-fold communication cost into a single one. Guarded by
    ``hasattr`` because a plain module, a DDP wrapper, and an FSDP2 module all
    reach this path.

    Args:
        model: The (possibly wrapped) model.
        enabled: Whether this microbatch should synchronise.
    """
    setter = getattr(model, "set_requires_gradient_sync", None)
    if callable(setter):
        setter(enabled)


def _neutralise_nonfinite(
    gradients: Sequence[torch.Tensor], finite: torch.Tensor
) -> None:
    """Replace every gradient with zeros when the step is not finite.

    ``mul_`` by a zero scalar is not enough: ``NaN * 0`` is ``NaN``, so the
    poison would survive the scaling and reach the weights. ``where`` selects a
    fresh zero instead, which is the only elementwise form that actually removes
    a NaN, and it keeps the decision on device so no host synchronisation is
    needed.

    Args:
        gradients: The gradients to sanitise, modified in place.
        finite: Zero-dimensional bool tensor; ``True`` leaves them untouched.
    """
    with torch.no_grad():
        for gradient in gradients:
            zero = torch.zeros((), dtype=gradient.dtype, device=gradient.device)
            torch.where(finite, gradient, zero, out=gradient)


def train_step(
    state: TrainState,
    batch: MediaBatch,
    objective: Objective,
    *,
    patchifier: Patchifier,
    gradient_accumulation_steps: int = 1,
    microbatch_index: int = 0,
    max_grad_norm: float | None = None,
    cp_mesh: DeviceMesh | None = None,
    pp_mesh: DeviceMesh | None = None,
    autocast_dtype: torch.dtype = torch.bfloat16,
    data_world: int = 1,
) -> tuple[TrainState, StepMetrics]:
    """Run one microbatch of training and, on the last one, an optimizer step.

    ``state`` is advanced in place and returned, so the call reads functionally
    without copying an optimizer's state every step.

    Args:
        state: Model, optimizer, schedule, EMA, RNG, and progress counters.
        batch: One microbatch, already on the compute device.
        objective: Computes the loss from the model and the batch.
        patchifier: Converts the video grid into tokens.
        gradient_accumulation_steps: Microbatches per optimizer step.
        microbatch_index: Zero-based index within the accumulation group. The
            gradients are zeroed at index 0 and the optimizer steps at the last
            index.
        max_grad_norm: Global gradient-norm clip, or ``None``.
        cp_mesh: Context-parallel sub-mesh, forwarded to the objective.
        pp_mesh: Pipeline sub-mesh, needed for a correct global gradient norm
            because pipeline stages hold disjoint parameters.
        autocast_dtype: Mixed-precision dtype, applied on CUDA only.
        data_world: Data-parallel rank count, used to scale this rank's sample
            and token counts into job-wide totals.

    Returns:
        The advanced state and the metrics for this microbatch.

    Raises:
        ValueError: If ``microbatch_index`` is outside the accumulation group.
    """
    if not 0 <= microbatch_index < gradient_accumulation_steps:
        raise ValueError(
            f"microbatch_index={microbatch_index} must lie in "
            f"[0, {gradient_accumulation_steps})"
        )
    device = batch.device
    is_first = microbatch_index == 0
    is_last = microbatch_index == gradient_accumulation_steps - 1

    if is_first:
        state.optimizer.zero_grad(set_to_none=True)
    _set_gradient_sync(state.model, is_last)

    with _autocast(device, autocast_dtype):
        output: ObjectiveOutput = objective(
            state.model, batch, state.rng, patchifier=patchifier, cp_mesh=cp_mesh
        )

    # Dividing by the accumulation count here, rather than scaling the gradients
    # afterwards, keeps the accumulated gradient equal to the mean over the whole
    # global batch — which is what makes a learning rate transferable between two
    # runs that reach the same global batch size with different microbatching.
    (output.loss / gradient_accumulation_steps).backward()

    parameters = [p for p in state.model.parameters() if p.grad is not None]
    grad_norm = torch.zeros((), dtype=torch.float32, device=device)
    finite = torch.isfinite(output.loss.detach())

    if is_last:
        if max_grad_norm is not None and parameters:
            grad_norm = clip_grad_norm(parameters, max_grad_norm, pp_mesh=pp_mesh).to(
                torch.float32
            )
        elif parameters:
            grad_norm = torch.nn.utils.get_total_norm(
                [p.grad for p in parameters if p.grad is not None],
                norm_type=2.0,
                error_if_nonfinite=False,
            ).to(torch.float32)
        finite = finite & torch.isfinite(grad_norm)
        _neutralise_nonfinite(
            [p.grad for p in parameters if p.grad is not None], finite
        )
        state.optimizer.step()
        if state.schedule is not None:
            state.schedule.step()
        if state.ema is not None:
            state.ema.update(state.model)
        state.step += 1

    # Counted from the static spec rather than from the device-side valid-token
    # tensors: the honest per-token count would need a synchronisation every
    # step, and this one is exact for the admitted (padded) sequence, which is
    # also the number that FLOPs and throughput are actually spent on.
    state.samples_seen += batch.spec.batch_size * data_world
    state.tokens_seen += batch.spec.batch_size * batch.spec.sequence_length * data_world

    skipped = ~finite if is_last else torch.zeros((), dtype=torch.bool, device=device)
    metrics = StepMetrics(
        loss=output.loss.detach().to(torch.float32),
        video_loss=output.video_loss.to(torch.float32),
        audio_loss=output.audio_loss.to(torch.float32),
        valid_video_tokens=output.valid_video_tokens,
        valid_audio_tokens=output.valid_audio_tokens,
        grad_norm=grad_norm,
        nonfinite=~finite,
        skipped=skipped,
    )
    return state, metrics


class Trainer:
    """Wires a parallel model, an objective, and a loader into a run.

    Everything the trainer does beyond calling :func:`train_step` is cadence:
    when to log, when to evaluate, when to checkpoint, and which curriculum
    stage is active. Checkpointing and telemetry are *callbacks* rather than
    methods, so this class has no dependency on either subsystem and a run can
    be assembled from whichever of them exist.

    Args:
        state: The mutable training state.
        objective: The loss.
        parallel: The parallelised model and its sub-meshes.
        config: Cadences and curriculum.
        patchifier: Grid-to-token converter. Defaults to the model's own
            ``patchifier`` property when it exposes one, because the model and
            the objective must agree on the patch geometry or the predicted
            token width will not match the target's.
        callbacks: Hooks invoked at step, eval, and checkpoint boundaries.
        logger: Where metric snapshots go. Defaults to the standard library
            logger, so a run without the telemetry subsystem still reports.
    """

    __slots__ = (
        "_stage_name",
        "callbacks",
        "config",
        "logger",
        "objective",
        "parallel",
        "patchifier",
        "state",
    )

    def __init__(
        self,
        state: TrainState,
        objective: Objective,
        parallel: ParallelModel,
        config: TrainerConfig,
        *,
        patchifier: Patchifier | None = None,
        callbacks: Sequence[TrainerCallback] = (),
        logger: MetricSink | None = None,
    ) -> None:
        self.state = state
        self.objective = objective
        self.parallel = parallel
        self.config = config
        self.patchifier = patchifier or _model_patchifier(parallel)
        self.callbacks = tuple(callbacks)
        self.logger = logger
        self._stage_name: str | None = None

    @property
    def device(self) -> torch.device:
        """Device the model's parameters live on."""
        for parameter in self.state.model.parameters():
            return parameter.device
        return torch.device("cpu")

    def train_step(
        self,
        batch: MediaBatch,
        *,
        microbatch_index: int = 0,
        accumulation: int | None = None,
    ) -> StepMetrics:
        """Run one microbatch through the functional step.

        Args:
            batch: One microbatch on the compute device.
            microbatch_index: Index within the accumulation group.
            accumulation: Microbatches per optimizer step; defaults to the
                configured value.

        Returns:
            The metrics for this microbatch.
        """
        _, metrics = train_step(
            self.state,
            batch,
            self.objective,
            patchifier=self.patchifier,
            gradient_accumulation_steps=(
                accumulation or self.config.gradient_accumulation_steps
            ),
            microbatch_index=microbatch_index,
            max_grad_norm=self.config.max_grad_norm,
            cp_mesh=self.parallel.cp_mesh,
            pp_mesh=self.parallel.pp_mesh,
            autocast_dtype=self.config.torch_autocast_dtype,
            data_world=self.config.data_world,
        )
        return metrics

    def fit(
        self,
        loader: Iterable[MediaBatch],
        total_steps: int | None = None,
    ) -> None:
        """Train until the step budget is exhausted or the loader runs dry.

        Args:
            loader: Yields microbatches. Sharded on ``data_rank`` by the data
                subsystem; the trainer does no sharding of its own.
            total_steps: Optimizer steps to run to. ``None`` trains until the
                loader is exhausted.

        Raises:
            ValueError: If ``total_steps`` is not positive.
        """
        if total_steps is not None and (
            isinstance(total_steps, bool) or total_steps < 1
        ):
            raise ValueError(f"total_steps must be positive; got {total_steps!r}")
        accumulation = self.config.gradient_accumulation_steps
        device = self.device
        self._announce_stage(loader)

        for index, batch in enumerate(_iterate(loader, device)):
            metrics = self.train_step(
                batch,
                microbatch_index=index % accumulation,
                accumulation=accumulation,
            )
            if index % accumulation != accumulation - 1:
                continue
            self._on_step_end(metrics)
            self._announce_stage(loader)
            if total_steps is not None and self.state.step >= total_steps:
                return

    def _on_step_end(self, metrics: StepMetrics) -> None:
        """Fire the step callbacks and the cadence-driven ones."""
        step = self.state.step
        _dispatch(self.callbacks, "on_step_end", self, self.state, metrics)
        if step % self.config.log_every == 0:
            # The single host synchronisation in the loop.
            snapshot = metrics.to_mapping()
            snapshot.update(
                {f"progress/{k}": float(v) for k, v in self.state.progress().items()}
            )
            if self.state.schedule is not None:
                snapshot["lr"] = self.state.schedule.get_last_lr()[0]
            if self.logger is not None:
                self.logger.log_metrics(snapshot, step)
            else:
                _LOGGER.info("step %d %s", step, snapshot)
        if self.config.eval_every and step % self.config.eval_every == 0:
            _dispatch(self.callbacks, "on_eval", self, self.state)
        if self.config.checkpoint_every and step % self.config.checkpoint_every == 0:
            _dispatch(self.callbacks, "on_checkpoint", self, self.state)

    def _announce_stage(self, loader: object) -> None:
        """Push the active curriculum stage to the loader when it changes."""
        stage = self.config.curriculum.stage_for(self.state.step)
        if stage is None or stage.name == self._stage_name:
            return
        self._stage_name = stage.name
        setter = getattr(loader, "set_curriculum", None)
        if callable(setter):
            setter(stage)
        _LOGGER.info(
            "curriculum stage %r active at step %d: %s",
            stage.name,
            self.state.step,
            ", ".join(f"{b.name}={w:g}" for b, w in stage.buckets),
        )


def _iterate(
    loader: Iterable[MediaBatch], device: torch.device
) -> Iterator[MediaBatch]:
    """Yield batches placed on the compute device.

    Args:
        loader: The data source.
        device: Target device.

    Yields:
        Batches on ``device``.
    """
    for batch in loader:
        yield batch if batch.device == device else batch.to(device)


def _dispatch(callbacks: Sequence[TrainerCallback], hook: str, *args: object) -> None:
    """Invoke one hook on every callback that implements it.

    Args:
        callbacks: The registered callbacks.
        hook: Method name to call.
        *args: Arguments forwarded to the hook.
    """
    for callback in callbacks:
        method = getattr(callback, hook, None)
        if callable(method):
            method(*args)


def _model_patchifier(parallel: ParallelModel) -> Patchifier:
    """Return the model's patchifier, or the default.

    Args:
        parallel: The parallelised model.

    Returns:
        The model's own patchifier when it exposes one, else a
        :class:`~avgen.core.patchify.GridPatchifier` with default patch sizes.
    """
    model = unwrap_model(parallel.model)
    candidate = getattr(model, "patchifier", None)
    if isinstance(candidate, Patchifier):
        return candidate
    return GridPatchifier()
