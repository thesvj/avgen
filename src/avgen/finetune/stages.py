"""Fine-tuning recipes as data, not as code paths.

A production video model is not trained once. It goes through a sequence of
stages — low-resolution pretraining, high-resolution adaptation, duration
extension, an aesthetic fine-tune, a control adapter, a few-step distillation —
and each stage differs from the last in a handful of settings: which parameters
move, at what learning rate, with which timestep distribution, over which mix of
conditioning tasks.

The usual way this is organised is one training script per stage. Six scripts
that are 95% identical, that drift apart, and where the answer to "what exactly
was different about the 720p run" lives in a diff nobody kept. A
:class:`FinetuneStage` is that difference, written down as a value: it is
serialisable, diffable, loggable next to the checkpoint, and comparable between
two runs by equality.

The one piece of real arithmetic in this module is the resolution shift, because
it is the setting that most often gets forgotten and it is invisible in the loss
curve when it is wrong.

Timestep shift under resolution change
--------------------------------------

Rectified flow noises a latent as ``x_s = (1-s)·x_0 + s·eps``. At a fixed noise
level ``s``, how much *information* survives depends on how many correlated
tokens there are: a 720p frame has four times the tokens of a 360p frame, and
because natural video is spatially redundant, averaging over four times as many
noisy tokens recovers the signal far better. The same nominal ``s`` is therefore
a much easier denoising problem at high resolution than at low.

Left uncorrected, a model fine-tuned at high resolution spends almost all of its
training samples on problems it finds easy, never learns to construct global
structure from near-pure noise, and produces the characteristic failure: correct
local texture, incoherent composition and motion — plus a sampler whose early
steps do almost nothing.

The fix, from the SD3 paper and used by every large video model since, is to
shift the timestep distribution toward higher noise as the sequence grows::

    s' = shift * s / (1 + (shift - 1) * s)

with ``shift`` interpolated in the sequence length between a base and a maximum.
:func:`resolution_shift` computes it and :func:`shift_timesteps` applies it; both
must be consistent between training and sampling, or the sampler will be
integrating a different SDE from the one that was trained.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import torch

from avgen.finetune.adapters import ControlAdapterConfig
from avgen.finetune.lora import LoRAConfig

__all__ = [
    "DistillationConfig",
    "FinetuneStage",
    "TimestepPlan",
    "apply_stage",
    "get_stage",
    "list_stages",
    "register_stage",
    "resolution_shift",
    "shift_timesteps",
    "stage_names_for",
    "standard_stages",
]


def resolution_shift(
    sequence_length: int,
    *,
    base_length: int = 4096,
    base_shift: float = 1.0,
    max_length: int = 65536,
    max_shift: float = 3.0,
) -> float:
    """Interpolate the timestep shift for a sequence length.

    Linear in the sequence length between the two anchor points, and clamped
    outside them. Linear rather than log-linear because that is what the
    reference implementations do and because the anchors, not the interpolant,
    carry the information — the value is calibrated by measuring, at each
    resolution, the noise level at which a model can no longer recover global
    structure.

    Args:
        sequence_length: Tokens per sample at the target resolution and
            duration. This is the correct variable, not the pixel count: it is
            what actually determines how much redundancy the denoiser can
            average over, and it makes duration extension and resolution
            increase the same adjustment.
        base_length: Sequence length the base shift was calibrated at.
        base_shift: Shift at ``base_length``. 1.0 means no shift.
        max_length: Sequence length the maximum shift was calibrated at.
        max_shift: Shift at ``max_length``.

    Returns:
        The shift factor, at least 1.0.

    Raises:
        ValueError: If the anchors are not positive and ordered, or if a shift
            is below 1.0 (a shift below 1 moves probability toward *low* noise,
            which is the opposite of the correction).
    """
    if sequence_length < 1:
        raise ValueError(f"sequence_length must be positive; got {sequence_length!r}")
    if base_length < 1 or max_length <= base_length:
        raise ValueError(
            "anchors must satisfy 0 < base_length < max_length; got "
            f"base_length={base_length}, max_length={max_length}"
        )
    if base_shift < 1.0 or max_shift < 1.0:
        raise ValueError(
            "shifts must be at least 1.0 (below 1 shifts toward low noise, "
            f"the wrong direction); got base_shift={base_shift}, max_shift={max_shift}"
        )
    slope = (max_shift - base_shift) / (max_length - base_length)
    value = base_shift + slope * (sequence_length - base_length)
    return float(
        min(max(value, min(base_shift, max_shift)), max(base_shift, max_shift))
    )


def shift_timesteps(noise_level: torch.Tensor, shift: float) -> torch.Tensor:
    """Apply the SD3 timestep shift to sampled noise levels.

    ``s' = shift*s / (1 + (shift-1)*s)`` maps ``[0, 1]`` onto itself, fixes both
    endpoints, and pushes mass toward 1 (more noise) for ``shift > 1``. It is
    monotone and invertible, which is what makes it safe to apply to a sampler's
    step schedule as well as to a training distribution — and applying it to one
    but not the other is the single most common way this goes wrong.

    Args:
        noise_level: Noise levels in ``[0, 1]``, ``0`` meaning clean, matching
            :attr:`avgen.core.tokens.TokenStream.noise_level`.
        shift: Shift factor from :func:`resolution_shift`.

    Returns:
        Shifted noise levels, same shape and dtype.

    Raises:
        ValueError: If ``shift`` is not positive.
    """
    if shift <= 0.0:
        raise ValueError(f"shift must be positive; got {shift!r}")
    return shift * noise_level / (1.0 + (shift - 1.0) * noise_level)


@dataclass(frozen=True, slots=True)
class TimestepPlan:
    """Which noise levels a stage trains on.

    Args:
        sampler: Registered sampler name from :mod:`avgen.train.timestep` —
            ``"uniform"``, ``"logit_normal"``, ``"shifted_logit_normal"`` or
            ``"mode"``.
        options: Keyword arguments for the sampler, as ordered pairs so the plan
            stays hashable and its serialisation is byte-stable.
        shift: Explicit shift factor, or ``None`` to derive it from the
            sequence length with :func:`resolution_shift`.
        base_length: Anchor passed to :func:`resolution_shift`.
        base_shift: Anchor passed to :func:`resolution_shift`.
        max_length: Anchor passed to :func:`resolution_shift`.
        max_shift: Anchor passed to :func:`resolution_shift`.
    """

    sampler: str = "logit_normal"
    options: tuple[tuple[str, float], ...] = ()
    shift: float | None = None
    base_length: int = 4096
    base_shift: float = 1.0
    max_length: int = 65536
    max_shift: float = 3.0

    def shift_for(self, sequence_length: int) -> float:
        """Return the shift this plan applies at a given sequence length.

        Args:
            sequence_length: Tokens per sample.

        Returns:
            The explicit shift if one was set, otherwise the interpolated one.
        """
        if self.shift is not None:
            return self.shift
        return resolution_shift(
            sequence_length,
            base_length=self.base_length,
            base_shift=self.base_shift,
            max_length=self.max_length,
            max_shift=self.max_shift,
        )

    def as_kwargs(self) -> dict[str, float]:
        """Return the sampler options as a keyword mapping."""
        return dict(self.options)


@dataclass(frozen=True, slots=True)
class DistillationConfig:
    """Settings for a step-reduction or guidance-distillation stage.

    Scaffolding: this records *what* a distillation stage is, so a recipe can be
    written and diffed today, and so the trainer has a single place to read the
    teacher from once the objectives land in :mod:`avgen.train`. No objective is
    implemented here — the loss belongs next to the other objectives, not in the
    fine-tuning package, or the two would drift.

    Args:
        method: ``"progressive"`` (halve the step count repeatedly),
            ``"consistency"`` (self-consistency along the flow trajectory),
            ``"dmd"`` (distribution matching against a teacher score) or
            ``"lcm"`` (latent consistency).
        teacher_checkpoint: Path or identifier of the frozen teacher. ``None``
            means the teacher is a frozen copy of the student's initial weights,
            which is what progressive distillation does.
        student_steps: Sampler steps the student must match the teacher in.
        distill_guidance: Whether the student also absorbs classifier-free
            guidance, so inference needs one forward per step instead of two.
            This is a larger win than the step reduction on most deployments and
            is usually done in the same stage.
        teacher_steps: Sampler steps the teacher is run at.

    Raises:
        ValueError: On an unknown method or a non-positive step count.
    """

    method: str = "progressive"
    teacher_checkpoint: str | None = None
    student_steps: int = 4
    distill_guidance: bool = True
    teacher_steps: int = 50

    def __post_init__(self) -> None:
        """Validate the method name and step counts.

        Raises:
            ValueError: On an unknown method or non-positive counts.
        """
        if self.method not in {"progressive", "consistency", "dmd", "lcm"}:
            raise ValueError(
                "method must be 'progressive', 'consistency', 'dmd' or 'lcm'; "
                f"got {self.method!r}"
            )
        for name in ("student_steps", "teacher_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer; got {value!r}")
        if self.student_steps > self.teacher_steps:
            raise ValueError(
                f"student_steps={self.student_steps} exceeds "
                f"teacher_steps={self.teacher_steps}; distillation reduces steps"
            )


@dataclass(frozen=True, slots=True)
class FinetuneStage:
    """One fine-tuning recipe, complete enough to run from.

    Everything that distinguishes this stage from another one is a field here.
    Nothing that distinguishes it is a code path, which is the point: a user
    adapts a recipe by writing ``replace(stage, learning_rate=5e-6)`` in a config
    file, not by copying a training script.

    Args:
        name: Registry key. Unique.
        description: One line explaining what the stage is for. It ends up in
            the run log next to the checkpoint, which is where someone will read
            it a year later.
        trainable_patterns: Parameter patterns to keep trainable. ``()`` means
            everything trains — full fine-tuning.
        frozen_patterns: Parameter patterns to freeze, applied after the
            trainable selection, so "everything except the text tower" is one
            line.
        lora: Adapter config, or ``None`` for no adapter.
        control: Control tower config, or ``None``.
        distillation: Distillation config, or ``None``.
        learning_rate: Peak learning rate. Adapter stages tolerate — and need —
            an order of magnitude more than full fine-tuning, because the
            effective update is scaled by ``alpha/rank`` and because a low-rank
            update cannot damage the model in the ways a full one can.
        weight_decay: Decoupled weight decay. Zero on adapter stages: decaying
            ``lora_b`` toward zero pulls the adapter back toward the identity it
            started at, which is a strong and usually unintended prior.
        warmup_steps: Linear warmup length.
        schedule: Registered schedule name from :mod:`avgen.train.schedule`.
        max_grad_norm: Gradient clipping threshold.
        ema_decay: EMA decay, or ``None`` to skip the EMA. Short stages should
            skip it: an EMA with a 10k-step time constant over a 2k-step stage
            is mostly the weights you started from.
        timesteps: Which noise levels to train on.
        conditioning_mix: ``(ConditionMode name, weight)`` pairs giving the task
            mixture. Weights are normalised by the conditioning sampler.
        batch_size: Global batch size in samples, or ``None`` to inherit.
        total_steps: Optimizer steps, or ``None`` to inherit.
        notes: Free-form guidance for whoever runs the stage.

    Raises:
        ValueError: On a non-positive learning rate, a negative decay, or a
            conditioning mix that does not sum to something positive.
    """

    name: str
    description: str
    trainable_patterns: tuple[str, ...] = ()
    frozen_patterns: tuple[str, ...] = ()
    lora: LoRAConfig | None = None
    control: ControlAdapterConfig | None = None
    distillation: DistillationConfig | None = None
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    warmup_steps: int = 500
    schedule: str = "constant"
    max_grad_norm: float = 1.0
    ema_decay: float | None = None
    timesteps: TimestepPlan = field(default_factory=TimestepPlan)
    conditioning_mix: tuple[tuple[str, float], ...] = (("JOINT", 1.0),)
    batch_size: int | None = None
    total_steps: int | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        """Validate the numeric fields and the conditioning mixture.

        Raises:
            ValueError: On an out-of-range field or an empty/negative mixture.
        """
        if not self.name:
            raise ValueError("stage name must be non-empty")
        if self.learning_rate <= 0.0:
            raise ValueError(
                f"learning_rate must be positive; got {self.learning_rate!r}"
            )
        if self.weight_decay < 0.0:
            raise ValueError(
                f"weight_decay must be non-negative; got {self.weight_decay!r}"
            )
        if self.warmup_steps < 0:
            raise ValueError(
                f"warmup_steps must be non-negative; got {self.warmup_steps!r}"
            )
        if self.max_grad_norm <= 0.0:
            raise ValueError(
                f"max_grad_norm must be positive; got {self.max_grad_norm!r}"
            )
        if self.ema_decay is not None and not 0.0 < self.ema_decay < 1.0:
            raise ValueError(
                f"ema_decay must be in (0, 1) or None; got {self.ema_decay!r}"
            )
        total = sum(weight for _, weight in self.conditioning_mix)
        if not self.conditioning_mix or total <= 0.0:
            raise ValueError(
                "conditioning_mix must be non-empty with a positive total weight; "
                f"got {self.conditioning_mix!r}"
            )
        if any(weight < 0.0 for _, weight in self.conditioning_mix):
            raise ValueError(
                f"conditioning_mix weights must be non-negative; got "
                f"{self.conditioning_mix!r}"
            )

    @property
    def uses_adapter(self) -> bool:
        """Whether the stage trains an adapter rather than base weights."""
        return self.lora is not None or self.control is not None

    def normalized_mix(self) -> dict[str, float]:
        """Return the conditioning mixture as probabilities summing to one.

        Returns:
            Mode name to probability, in declaration order.
        """
        total = sum(weight for _, weight in self.conditioning_mix)
        return {name: weight / total for name, weight in self.conditioning_mix}

    def with_overrides(self, **changes: Any) -> FinetuneStage:
        """Return a copy with fields replaced.

        Args:
            **changes: Field names and new values.

        Returns:
            The modified stage. The original is unchanged, so a shipped recipe
            can never be mutated by a run that borrows it.
        """
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe description for logging next to a checkpoint.

        Returns:
            A plain mapping. Nested configs become mappings of their own fields
            so a diff between two stages is readable.
        """
        payload: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "trainable_patterns": list(self.trainable_patterns),
            "frozen_patterns": list(self.frozen_patterns),
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "schedule": self.schedule,
            "max_grad_norm": self.max_grad_norm,
            "ema_decay": self.ema_decay,
            "timestep_sampler": self.timesteps.sampler,
            "timestep_options": dict(self.timesteps.options),
            "timestep_shift": self.timesteps.shift,
            "conditioning_mix": self.normalized_mix(),
            "batch_size": self.batch_size,
            "total_steps": self.total_steps,
            "uses_lora": self.lora is not None,
            "uses_control": self.control is not None,
            "uses_distillation": self.distillation is not None,
            "notes": self.notes,
        }
        if self.lora is not None:
            payload["lora"] = {
                "rank": self.lora.rank,
                "alpha": self.lora.alpha,
                "dropout": self.lora.dropout,
                "use_dora": self.lora.use_dora,
                "target_modules": list(self.lora.target_modules),
            }
        if self.control is not None:
            payload["control"] = {
                "control_width": self.control.control_width,
                "num_blocks": self.control.num_blocks,
                "injection_stride": self.control.injection_stride,
                "conditioning_scale": self.control.conditioning_scale,
            }
        if self.distillation is not None:
            payload["distillation"] = {
                "method": self.distillation.method,
                "student_steps": self.distillation.student_steps,
                "teacher_steps": self.distillation.teacher_steps,
                "distill_guidance": self.distillation.distill_guidance,
            }
        return payload


FULL = FinetuneStage(
    name="full",
    description="Full fine-tuning of every parameter on a new data distribution.",
    # An order of magnitude below pretraining. The model is already at a good
    # minimum; the job is to move it, not to search.
    learning_rate=1e-5,
    weight_decay=0.01,
    warmup_steps=1000,
    schedule="constant",
    ema_decay=0.9999,
    timesteps=TimestepPlan(
        sampler="logit_normal", options=(("mean", 0.0), ("std", 1.0))
    ),
    notes=(
        "Costs full optimizer state. Prefer 'lora' unless the target "
        "distribution is genuinely far from pretraining."
    ),
)

LORA = FinetuneStage(
    name="lora",
    description="Low-rank adaptation of the attention and FFN projections.",
    lora=LoRAConfig(rank=32, alpha=64.0, dropout=0.0),
    # Two orders of magnitude above full fine-tuning: the update is scaled by
    # alpha/rank and confined to a rank-32 subspace, so it cannot move the model
    # as far per unit of learning rate.
    learning_rate=1e-4,
    # Zero on purpose: decay on lora_b pulls the adapter back toward the exact
    # identity it was initialised at, which is a prior nobody asked for.
    weight_decay=0.0,
    warmup_steps=200,
    ema_decay=None,
    timesteps=TimestepPlan(
        sampler="logit_normal", options=(("mean", 0.0), ("std", 1.0))
    ),
    notes=(
        "Pair with FSDPConfig(ignore_frozen_params=True) so the frozen base is "
        "not all-gathered every step."
    ),
)

CONTROL = FinetuneStage(
    name="control",
    description="ControlNet-style side tower over a frozen base.",
    control=ControlAdapterConfig(control_width=16, num_blocks=6),
    frozen_patterns=(
        "blocks.*",
        "patch_embed*",
        "time_embed*",
        "text_proj*",
        "final_*",
    ),
    learning_rate=1e-4,
    weight_decay=0.0,
    warmup_steps=500,
    ema_decay=0.999,
    # Uniform rather than logit-normal: a control signal constrains structure,
    # which is decided at high noise, so the stage must not concentrate its
    # samples in the mid-noise band a generation-quality stage prefers.
    timesteps=TimestepPlan(sampler="uniform"),
    notes="Only the side tower trains; the base stays byte-identical.",
)

RESOLUTION_ADAPTATION = FinetuneStage(
    name="resolution_adaptation",
    description="Adapt a low-resolution model to a higher resolution.",
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=500,
    ema_decay=0.9999,
    # The load-bearing setting. shifted_logit_normal moves the training noise
    # distribution toward high noise in proportion to the sequence length; see
    # the module docstring for what goes wrong without it.
    timesteps=TimestepPlan(
        sampler="shifted_logit_normal",
        options=(("mean", 0.0), ("std", 1.0)),
        base_length=4096,
        base_shift=1.0,
        max_length=65536,
        max_shift=3.0,
    ),
    notes=(
        "The same shift MUST be applied to the inference step schedule. "
        "Training with a shift and sampling without it is the classic cause of "
        "a high-resolution fine-tune that looks worse than the model it started "
        "from."
    ),
)

DURATION_EXTENSION = FinetuneStage(
    name="duration_extension",
    description="Extend a short-clip model to longer clips.",
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=500,
    ema_decay=0.9999,
    # Duration raises the token count exactly as resolution does, so it needs
    # the same shift; the anchors are the same because the correction depends on
    # sequence length, not on which axis grew.
    timesteps=TimestepPlan(
        sampler="shifted_logit_normal",
        options=(("mean", 0.0), ("std", 1.0)),
        max_shift=3.0,
    ),
    # Continuation and inpainting carry most of the weight: a longer clip is
    # generated in practice by conditioning on what came before, so the stage
    # must train the task it will actually be asked to do.
    conditioning_mix=(("JOINT", 0.5), ("CONTINUATION", 0.35), ("INPAINT", 0.15)),
    notes=(
        "Coordinates are physical seconds, so extending duration does not "
        "invalidate the time axis the model already learned."
    ),
)

DISTILLATION = FinetuneStage(
    name="distillation",
    description="Few-step and guidance distillation from a frozen teacher.",
    distillation=DistillationConfig(
        method="progressive", student_steps=4, distill_guidance=True
    ),
    learning_rate=5e-6,
    weight_decay=0.0,
    warmup_steps=100,
    schedule="constant",
    ema_decay=0.999,
    timesteps=TimestepPlan(sampler="uniform"),
    notes=(
        "SCAFFOLDING: the objective is not implemented. This stage records the "
        "recipe; the loss belongs in avgen.train alongside the other objectives."
    ),
)

_STAGES: dict[str, FinetuneStage] = {
    stage.name: stage
    for stage in (
        FULL,
        LORA,
        CONTROL,
        RESOLUTION_ADAPTATION,
        DURATION_EXTENSION,
        DISTILLATION,
    )
}


def register_stage(stage: FinetuneStage, *, overwrite: bool = False) -> FinetuneStage:
    """Add a stage to the registry.

    Args:
        stage: The recipe to register.
        overwrite: Whether replacing an existing name is allowed. Off by
            default because a silent overwrite means two runs that both claim to
            be ``"lora"`` were not the same thing.

    Returns:
        The registered stage.

    Raises:
        KeyError: If the name is taken and ``overwrite`` is false.
    """
    if not overwrite and stage.name in _STAGES:
        raise KeyError(
            f"stage {stage.name!r} is already registered; pass overwrite=True "
            "if replacing it is intended"
        )
    _STAGES[stage.name] = stage
    return stage


def get_stage(name: str) -> FinetuneStage:
    """Return a registered stage by name.

    Args:
        name: Registry key.

    Returns:
        The recipe.

    Raises:
        KeyError: If the name is unknown, with the available names listed.
    """
    try:
        return _STAGES[name]
    except KeyError:
        raise KeyError(
            f"unknown stage {name!r}; registered stages are {list_stages()}"
        ) from None


def list_stages() -> tuple[str, ...]:
    """Return the registered stage names in sorted order.

    Returns:
        Sorted names. Sorted rather than insertion-ordered so the listing is
        identical regardless of which modules imported first.
    """
    return tuple(sorted(_STAGES))


def standard_stages() -> Mapping[str, FinetuneStage]:
    """Return a read-only view of the registry.

    Returns:
        Name to recipe. A copy, so a caller cannot mutate the shipped recipes.
    """
    return dict(_STAGES)


def apply_stage(model: Any, stage: FinetuneStage) -> Any:
    """Apply a stage's parameter selection and adapter injection to a model.

    The order is fixed and matters: adapters are injected first so their
    parameters exist to be selected, then the freeze rules run, and only then
    should the caller parallelise. See :mod:`avgen.finetune.freeze` for why
    freezing before ``parallelize`` is what lets FSDP skip the frozen base.

    Args:
        model: The model to prepare, modified in place.
        stage: The recipe to apply.

    Returns:
        The same model.
    """
    from avgen.finetune.freeze import freeze_except, freeze_matching
    from avgen.finetune.lora import apply_lora, mark_only_lora_trainable

    if stage.lora is not None:
        apply_lora(model, stage.lora)
        mark_only_lora_trainable(model, extra_trainable=stage.trainable_patterns)
    elif stage.trainable_patterns:
        freeze_except(model, stage.trainable_patterns)
    if stage.frozen_patterns:
        freeze_matching(model, stage.frozen_patterns)
    return model


def stage_names_for(modes: Sequence[str]) -> tuple[str, ...]:
    """Return the stages whose conditioning mix touches every listed mode.

    A small convenience for planning: "which shipped recipe already trains the
    continuation task?".

    Args:
        modes: :class:`~avgen.core.batch.ConditionMode` names.

    Returns:
        Matching stage names, sorted.
    """
    wanted = tuple(modes)
    return tuple(
        sorted(
            name
            for name, stage in _STAGES.items()
            if all(mode in stage.normalized_mix() for mode in wanted)
        )
    )
