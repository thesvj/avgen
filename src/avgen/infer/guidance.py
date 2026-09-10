"""Classifier-free guidance and its repairs.

Classifier-free guidance is the single most effective knob in conditional
generation and also the single most reliable source of artifacts. This module
implements the basic mechanism and the three fixes that a video model actually
needs, and documents what each one is repairing, because "turn the scale down"
is the wrong answer to all three.

**The mechanism.** The model is evaluated twice — once with the prompt, once with
a null prompt — and the difference between the two predictions is the direction
in which conditioning pulls. Extrapolating along that direction::

    guided = uncond + scale * (cond - uncond)

amplifies the conditional signal. It is an extrapolation, not an interpolation,
for any ``scale > 1``, and everything that goes wrong with it goes wrong for that
reason: nothing constrains the extrapolated point to lie on the data manifold.

**What goes wrong, and the three repairs:**

1. **Over-saturation.** The guided prediction's *norm* grows roughly linearly in
   the scale, so at ``scale = 12`` the implied clean sample has a standard
   deviation far larger than any real latent. Decoded, that reads as blown-out
   highlights and posterised colour. :func:`rescale_guidance` (Lin et al., "Common
   Diffusion Noise Schedules and Sample Steps are Flawed") rescales the guided
   prediction back to the conditional prediction's standard deviation.
2. **The direction is wrong, not just the magnitude.** Rescaling fixes the norm
   but keeps a guidance update whose component *parallel* to the conditional
   prediction is what inflated it in the first place — that component only makes
   the model more confident about what it already predicted.
   :func:`adaptive_projected_guidance` (APG) projects the update onto the
   component orthogonal to the conditional prediction, keeping the part that
   changes *what* is generated and discarding the part that only changes how
   loudly. This is the better default at high scales.
3. **A constant scale is the wrong shape.** See :meth:`GuidanceConfig.scale_at`.

**Modality-aware guidance.** A single avgen model handles text-to-video,
image-to-video, video-to-audio and audio-to-video, and those conditionings are
independent: a user may want the prompt followed loosely while the supplied first
frame is honoured exactly. :func:`apply_modality_guidance` composes a chain of
progressively-more-conditioned branches so that each conditioning source gets its
own scale. That is what makes one multi-task model controllable at inference
instead of being one blunt slider.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from avgen.core.model_input import ModelInput

__all__ = [
    "GuidanceConfig",
    "adaptive_projected_guidance",
    "apply_guidance",
    "apply_modality_guidance",
    "classifier_free_guidance",
    "rescale_guidance",
    "unconditional_branch",
]

#: Guidance modalities that carry an independent scale. Ordered from the most
#: general conditioning to the most specific, which is the order
#: :func:`apply_modality_guidance` composes them in.
_MODALITIES = ("text", "video", "audio")

#: Supported shapes for the guidance scale over the sampling trajectory.
_SCHEDULE_NAMES = ("constant", "linear", "cosine", "power", "interval")


@dataclass(frozen=True, slots=True)
class GuidanceConfig:
    """Everything that parameterises guidance for one generation.

    Args:
        scale: Base guidance scale. ``1.0`` disables guidance entirely and makes
            :func:`apply_guidance` return the conditional prediction unchanged,
            skipping the null forward pass — the property that lets a caller
            switch guidance off without a separate code path.
        text_scale: Multiplier on ``scale`` for text conditioning. ``None``
            means 1.0.
        video_scale: Multiplier on ``scale`` for video-anchor conditioning — a
            first frame, a temporal prefix, an inpainting mask.
        audio_scale: Multiplier on ``scale`` for audio-anchor conditioning.
        rescale: Strength of the Lin et al. standard-deviation rescale, in
            ``[0, 1]``. ``0.0`` disables it; ``0.7`` is the usual value when it
            is used at all.
        projection: Whether to use adaptive projected guidance instead of plain
            extrapolation.
        projection_eta: How much of the parallel component APG keeps. ``0.0``
            discards it entirely and is the paper's recommendation; small
            positive values soften the effect.
        projection_threshold: Norm ceiling applied to the guidance update before
            projection. ``0.0`` disables the clamp. Positive values bound the
            worst-case step when a single timestep produces an outlier
            difference.
        schedule: Scale schedule over the sampling trajectory. One of
            ``"constant"``, ``"linear"``, ``"cosine"``, ``"power"``, or
            ``"interval"``.
        schedule_power: Exponent for the ``"power"`` schedule.
        min_scale: Floor the schedule decays towards, and the value used outside
            the interval for the ``"interval"`` schedule.
        start_fraction: Start of the guided interval, as a fraction of the
            sampling trajectory.
        end_fraction: End of the guided interval.

    Raises:
        ValueError: If a scale is not finite and positive, a fraction is out of
            range, or the schedule name is unknown.
    """

    scale: float = 1.0
    text_scale: float | None = None
    video_scale: float | None = None
    audio_scale: float | None = None
    rescale: float = 0.0
    projection: bool = False
    projection_eta: float = 0.0
    projection_threshold: float = 0.0
    schedule: str = "constant"
    schedule_power: float = 1.0
    min_scale: float = 1.0
    start_fraction: float = 0.0
    end_fraction: float = 1.0

    def __post_init__(self) -> None:
        """Validate every coefficient at construction."""
        for name in ("scale", "min_scale"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive; got {value!r}")
        for name in ("text_scale", "video_scale", "audio_scale"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be finite and positive; got {value!r}")
        if not math.isfinite(self.rescale) or not 0.0 <= self.rescale <= 1.0:
            raise ValueError(f"rescale must be in [0, 1]; got {self.rescale!r}")
        if not math.isfinite(self.projection_eta):
            raise ValueError(
                f"projection_eta must be finite; got {self.projection_eta!r}"
            )
        if (
            not math.isfinite(self.projection_threshold)
            or self.projection_threshold < 0.0
        ):
            raise ValueError(
                "projection_threshold must be finite and non-negative; got "
                f"{self.projection_threshold!r}"
            )
        if self.schedule not in _SCHEDULE_NAMES:
            raise ValueError(
                f"unknown guidance schedule {self.schedule!r}; supported: "
                f"{', '.join(_SCHEDULE_NAMES)}"
            )
        if not math.isfinite(self.schedule_power) or self.schedule_power <= 0.0:
            raise ValueError(
                f"schedule_power must be finite and positive; got "
                f"{self.schedule_power!r}"
            )
        for name in ("start_fraction", "end_fraction"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]; got {value!r}")
        if self.start_fraction >= self.end_fraction:
            raise ValueError(
                f"start_fraction={self.start_fraction} must be below "
                f"end_fraction={self.end_fraction}"
            )

    @property
    def is_enabled(self) -> bool:
        """Whether guidance does anything at all.

        A base scale of exactly 1.0 with no modality override makes the guided
        prediction equal the conditional one, so the null forward pass can be
        skipped — halving the cost of a generation.
        """
        if self.scale != 1.0:
            return True
        return any(
            getattr(self, f"{modality}_scale") not in (None, 1.0)
            for modality in _MODALITIES
        )

    def scale_at(self, progress: float) -> float:
        """Return the base scale at a point in the sampling trajectory.

        ``progress`` runs from ``0.0`` at the first step (highest noise) to
        ``1.0`` at the last (lowest noise).

        **Why a constant scale wastes the early steps.** Guidance amplifies the
        difference between the conditional and unconditional predictions, and at
        the top of the schedule there is almost no difference to amplify: the
        input is nearly pure noise, both branches predict something close to the
        dataset mean, and their difference is dominated by model noise rather
        than by the prompt. Multiplying that by 8 injects high-variance junk into
        the step that decides the sample's global structure — which is
        empirically where guidance artifacts such as duplicated subjects and
        blown-out contrast are born. Conversely the *late* steps are where a
        large scale over-saturates, since the predictions there are confident and
        the extrapolation runs furthest off-manifold.

        The result (Kynkäänniemi et al. on guidance intervals, and the several
        cosine-decay variants that video models ship) is that the useful work
        happens in the middle of the trajectory, and both ends are better served
        by a scale near 1.

        Args:
            progress: Trajectory position in ``[0, 1]``.

        Returns:
            The scale to use at this point.

        Raises:
            ValueError: If ``progress`` is outside ``[0, 1]``.
        """
        if not math.isfinite(progress) or not 0.0 <= progress <= 1.0:
            raise ValueError(f"progress must be in [0, 1]; got {progress!r}")
        if self.schedule == "constant":
            return self.scale
        if self.schedule == "interval":
            inside = self.start_fraction <= progress <= self.end_fraction
            return self.scale if inside else self.min_scale
        span = self.scale - self.min_scale
        if self.schedule == "linear":
            return self.scale - span * progress
        if self.schedule == "cosine":
            # Half a cosine from 1 to 0 across the trajectory: flat at both ends,
            # steepest in the middle, so neither endpoint changes abruptly.
            return self.min_scale + span * 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_scale + span * (1.0 - progress) ** self.schedule_power

    def modality_scale(self, modality: str, progress: float = 0.0) -> float:
        """Return the effective scale for one conditioning modality.

        Composition is multiplicative: the base scale sets the overall strength
        of guidance and each modality factor says how much *more or less* that
        modality is trusted relative to it. That is what makes the knobs
        independent in practice — halving ``scale`` halves everything uniformly
        and leaves the relative balance between prompt and anchor untouched,
        which an additive composition would not.

        Args:
            modality: One of ``"text"``, ``"video"``, ``"audio"``.
            progress: Trajectory position, forwarded to :meth:`scale_at`.

        Returns:
            The composed scale.

        Raises:
            ValueError: If ``modality`` is not recognised.
        """
        if modality not in _MODALITIES:
            raise ValueError(
                f"unknown guidance modality {modality!r}; supported: "
                f"{', '.join(_MODALITIES)}"
            )
        factor = getattr(self, f"{modality}_scale")
        return self.scale_at(progress) * (1.0 if factor is None else factor)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation for pinning into a sample."""
        return {
            "scale": self.scale,
            "text_scale": self.text_scale,
            "video_scale": self.video_scale,
            "audio_scale": self.audio_scale,
            "rescale": self.rescale,
            "projection": self.projection,
            "projection_eta": self.projection_eta,
            "projection_threshold": self.projection_threshold,
            "schedule": self.schedule,
            "schedule_power": self.schedule_power,
            "min_scale": self.min_scale,
            "start_fraction": self.start_fraction,
            "end_fraction": self.end_fraction,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> GuidanceConfig:
        """Rebuild a config from :meth:`to_dict` output.

        Args:
            values: The mapping.

        Returns:
            The restored config.
        """
        keys = (
            "scale",
            "text_scale",
            "video_scale",
            "audio_scale",
            "rescale",
            "projection",
            "projection_eta",
            "projection_threshold",
            "schedule",
            "schedule_power",
            "min_scale",
            "start_fraction",
            "end_fraction",
        )
        return cls(**{key: values[key] for key in keys if key in values})


def unconditional_branch(inputs: ModelInput) -> ModelInput:
    """Build the null branch for classifier-free guidance.

    Delegates to :meth:`~avgen.core.model_input.ModelInput.unconditional`, which
    nulls the *text* context and deliberately preserves structural conditioning.
    That asymmetry is the whole reason this is a one-line wrapper rather than
    something the caller assembles: dropping the clean first frame from the null
    branch would make it a different task rather than the same task without a
    prompt, and the difference between the branches would then measure "with
    versus without an anchor" instead of "with versus without a prompt".

    Args:
        inputs: The conditional input.

    Returns:
        The matching unconditional input.
    """
    return inputs.unconditional()


def classifier_free_guidance(
    conditional: torch.Tensor,
    unconditional: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Extrapolate along the conditioning direction.

    Args:
        conditional: Prediction with conditioning.
        unconditional: Prediction with the null branch.
        scale: Guidance scale. ``1.0`` returns ``conditional`` exactly.

    Returns:
        The guided prediction.

    Raises:
        ValueError: If the two predictions disagree in shape.
    """
    _require_match(conditional, unconditional)
    if scale == 1.0:
        # Returning the tensor itself rather than computing uncond + 1 * delta
        # keeps the scale-1 path bit-identical to the unguided path, which is
        # what makes the equivalence testable rather than approximate.
        return conditional
    return unconditional + scale * (conditional - unconditional)


def rescale_guidance(
    guided: torch.Tensor,
    conditional: torch.Tensor,
    factor: float,
) -> torch.Tensor:
    """Pull an over-scaled guided prediction back to a sane magnitude.

    The mechanism, precisely: guidance extrapolates, so
    ``std(guided) ≈ scale * std(cond - uncond) + ...`` grows with the scale while
    the conditional prediction's own standard deviation does not. The implied
    clean sample therefore lands outside the range the decoder was trained on,
    and the decoder maps out-of-range latents to clipped, posterised colour —
    the "everything looks like an over-processed stock photo" failure.

    The fix rescales the guided prediction so its per-sample standard deviation
    matches the conditional prediction's, then interpolates back towards the
    unrescaled version by ``factor``. Full rescaling (``factor = 1``) tends to
    flatten the image, hence the interpolation; ``0.7`` is the value the paper
    settles on.

    Statistics are computed per sample over every non-batch axis. Per sample
    rather than over the whole batch, because two prompts in one batch have no
    reason to share a magnitude and coupling them would make a sample depend on
    what it was batched with.

    Args:
        guided: The guided prediction.
        conditional: The conditional prediction, whose magnitude is the target.
        factor: Interpolation weight in ``[0, 1]``. ``0.0`` is a no-op.

    Returns:
        The rescaled prediction.

    Raises:
        ValueError: If the shapes disagree or ``factor`` is out of range.
    """
    _require_match(guided, conditional)
    if not math.isfinite(factor) or not 0.0 <= factor <= 1.0:
        raise ValueError(f"factor must be in [0, 1]; got {factor!r}")
    if factor == 0.0:
        return guided
    dims = tuple(range(1, guided.ndim))
    std_conditional = conditional.std(dim=dims, keepdim=True)
    std_guided = guided.std(dim=dims, keepdim=True)
    # A degenerate (constant) guided prediction has zero spread and nothing to
    # rescale; clamping leaves it untouched instead of producing a NaN.
    rescaled = guided * (std_conditional / torch.clamp(std_guided, min=1e-8))
    return factor * rescaled + (1.0 - factor) * guided


def adaptive_projected_guidance(
    conditional: torch.Tensor,
    unconditional: torch.Tensor,
    scale: float,
    *,
    eta: float = 0.0,
    norm_threshold: float = 0.0,
) -> torch.Tensor:
    """Guide along the component orthogonal to the conditional prediction.

    Decompose the guidance update ``delta = cond - uncond`` relative to the
    conditional prediction::

        delta = delta_parallel + delta_orthogonal

    The parallel component points along what the model already predicts, so
    amplifying it only scales the prediction up — it is precisely the term that
    inflates the magnitude and produces the over-saturation that
    :func:`rescale_guidance` then has to undo. The orthogonal component is the
    part that changes *what* is generated: it moves the prediction towards
    prompt-consistent content rather than towards more of the same content.

    APG keeps the orthogonal component in full and admits only ``eta`` of the
    parallel one::

        guided = cond + (scale - 1) * (delta_orthogonal + eta * delta_parallel)

    With ``eta = 0`` the magnitude is left essentially alone, so a scale of 15
    behaves like a scale of 15 on *content* without behaving like a scale of 15
    on *contrast*. That is why it is the better default at high scales, and why
    it is preferable to rescaling: it removes the cause rather than correcting
    the symptom afterwards.

    The optional norm threshold clamps ``delta`` before the decomposition. A
    single timestep occasionally produces a difference far larger than its
    neighbours — a discretisation artifact rather than signal — and bounding it
    stops one step from dominating the trajectory.

    Args:
        conditional: Prediction with conditioning.
        unconditional: Prediction with the null branch.
        scale: Guidance scale. ``1.0`` returns ``conditional`` exactly.
        eta: Fraction of the parallel component to keep.
        norm_threshold: Per-sample ceiling on the update norm. ``0.0`` disables.

    Returns:
        The guided prediction.

    Raises:
        ValueError: If the shapes disagree or ``norm_threshold`` is negative.
    """
    _require_match(conditional, unconditional)
    if not math.isfinite(norm_threshold) or norm_threshold < 0.0:
        raise ValueError(
            f"norm_threshold must be finite and non-negative; got {norm_threshold!r}"
        )
    if scale == 1.0:
        return conditional
    delta = conditional - unconditional
    dims = tuple(range(1, conditional.ndim))
    if norm_threshold > 0.0:
        norm = torch.linalg.vector_norm(delta, dim=dims, keepdim=True)
        limit = norm_threshold / torch.clamp(norm, min=1e-8)
        delta = delta * torch.clamp(limit, max=1.0)
    # Project delta onto cond: the parallel part is (delta . cond_hat) cond_hat.
    denominator = torch.clamp(
        (conditional * conditional).sum(dim=dims, keepdim=True), min=1e-8
    )
    coefficient = (delta * conditional).sum(dim=dims, keepdim=True) / denominator
    parallel = coefficient * conditional
    orthogonal = delta - parallel
    return conditional + (scale - 1.0) * (orthogonal + eta * parallel)


def apply_guidance(
    conditional: torch.Tensor,
    unconditional: torch.Tensor | None = None,
    *,
    config: GuidanceConfig | None = None,
    progress: float = 0.0,
    modality: str = "text",
    scale: float | None = None,
) -> torch.Tensor:
    """Apply the configured guidance to a pair of predictions.

    Composes the schedule, the modality scale, the projection choice, and the
    rescale in the one order that makes sense: the scale is resolved first, the
    guided prediction is formed by either plain extrapolation or APG, and the
    magnitude repair is applied last because it is a correction to the result.

    Args:
        conditional: Prediction with conditioning.
        unconditional: Prediction with the null branch. May be ``None`` when
            guidance is disabled, which is how a caller skips the null forward
            pass entirely.
        config: Guidance configuration. ``None`` means unguided.
        progress: Trajectory position in ``[0, 1]``, for the scale schedule.
        modality: Which modality's scale to use.
        scale: Explicit scale override, bypassing the schedule and the modality
            factor. Used by :func:`apply_modality_guidance`, which has already
            resolved the scale for the branch it is composing.

    Returns:
        The guided prediction.

    Raises:
        ValueError: If guidance is enabled but ``unconditional`` is ``None``.
    """
    if config is None or not config.is_enabled:
        return conditional
    resolved = config.modality_scale(modality, progress) if scale is None else scale
    if resolved == 1.0:
        return conditional
    if unconditional is None:
        raise ValueError(
            "guidance is enabled but no unconditional prediction was supplied; "
            "evaluate the model on ModelInput.unconditional() as well, or set "
            "the guidance scale to 1.0"
        )
    if config.projection:
        guided = adaptive_projected_guidance(
            conditional,
            unconditional,
            resolved,
            eta=config.projection_eta,
            norm_threshold=config.projection_threshold,
        )
    else:
        guided = classifier_free_guidance(conditional, unconditional, resolved)
    return rescale_guidance(guided, conditional, config.rescale)


def apply_modality_guidance(
    branches: Sequence[tuple[str, torch.Tensor]],
    *,
    config: GuidanceConfig,
    progress: float = 0.0,
) -> torch.Tensor:
    """Compose guidance over several independently-scaled conditioning sources.

    ``branches`` is an ordered chain from the most null prediction to the most
    conditioned one, each labelled with the conditioning that was *added* to
    reach it. The first entry is the base and carries no scale. Each subsequent
    entry contributes its own increment::

        guided = base + sum_i scale_i * (pred_i - pred_{i-1})

    Each ``scale_i`` comes from :meth:`GuidanceConfig.modality_scale`, so it is
    the base scale multiplied by that modality's factor — the multiplicative
    composition described in :meth:`GuidanceConfig.modality_scale`.

    The chain form rather than independent differences from a shared null is
    deliberate: it makes each increment measure the marginal effect of *that*
    conditioning given the ones already applied, which is what an independent
    knob should control. Differences from a shared null double-count whatever the
    conditionings have in common, and the resulting scales interact.

    The cost is one forward pass per branch, so a three-way chain costs three
    evaluations per step against two for ordinary guidance. Chain only the
    conditionings a user actually needs to control separately.

    Args:
        branches: ``(modality, prediction)`` pairs, base first. The base entry's
            modality label is ignored.
        config: Guidance configuration.
        progress: Trajectory position in ``[0, 1]``.

    Returns:
        The composed prediction.

    Raises:
        ValueError: If fewer than two branches are given or the shapes disagree.
    """
    if len(branches) < 2:
        raise ValueError(
            f"apply_modality_guidance needs at least a base and one conditioned "
            f"branch; got {len(branches)}"
        )
    base = branches[0][1]
    guided = base
    previous = base
    for modality, prediction in branches[1:]:
        _require_match(prediction, base)
        scale = config.modality_scale(modality, progress)
        guided = guided + scale * (prediction - previous)
        previous = prediction
    # The most-conditioned branch is the magnitude reference, matching what
    # rescale_guidance does in the two-branch case.
    return rescale_guidance(guided, branches[-1][1], config.rescale)


def _require_match(first: torch.Tensor, second: torch.Tensor) -> None:
    """Reject a shape or device mismatch between two prediction branches."""
    if tuple(first.shape) != tuple(second.shape):
        raise ValueError(
            "guidance branches must have the same shape; got "
            f"{tuple(first.shape)} and {tuple(second.shape)}"
        )
    if first.device != second.device:
        raise ValueError(
            "guidance branches must be on the same device; got "
            f"{first.device} and {second.device}"
        )
