"""Which task each sample is trained on, and which of its tokens stay clean.

A video model that only ever learns "text in, video out" is strictly weaker than
the same model trained on a mixture of conditional tasks, and the reason is
worth stating plainly: **image-to-video, continuation, inpainting, and
audio-conditioned generation are all the same computation.** Each of them hands
the model a set of *clean anchor tokens* and asks it to denoise the rest in a
way that is consistent with them. Train on the mixture and the model learns what
consistency with a clean anchor means; train on ``JOINT`` alone and it never
sees a clean token at training time, so at inference it treats the conditioning
frame as just another slightly odd input and drifts away from it within a
second of video.

The mechanism is one boolean mask per modality. A token marked ``conditioned``
is handed to the model at noise level 0 and excluded from the loss (see
:meth:`~avgen.core.tokens.TokenStream.loss_mask`). Nothing else in the model
changes; the task label rides along in ``condition_mode`` so the model can
modulate on it, but the anchor mask is what actually carries the information.

**Anchor dropout is what makes modality-aware guidance possible.** With some
probability the anchor for one modality is cleared while the task label is left
alone, so the model sees the same requested task both with and without its
structural conditioning. That pair is exactly the null and conditional branch a
classifier-free-guidance step needs, which is what lets inference dial up
"follow the conditioning image" independently of "follow the prompt". Clearing
the *label* as well would have been the obvious alternative, and it is wrong:
the two branches would then be different tasks, and their difference would
measure the task change rather than the strength of the conditioning.

Every draw in this module comes from ``rng.conditioning`` and from nowhere else.
Sharing the timestep or noise stream would couple the task mixture to the noise
schedule, so changing the dropout rate would silently change every timestep in
the run and no two configurations would be comparable.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import torch

from avgen.core._validate import require_dtype, require_probability, require_shape
from avgen.core.batch import ConditionMode, MediaBatch
from avgen.core.rng import RNGStreams
from avgen.train._random import bernoulli, categorical, uniform

__all__ = [
    "ConditioningPlan",
    "ConditioningSampler",
    "MultiTaskConditioning",
    "default_mode_weights",
]

#: Modes whose anchor is the audio stream, or which are meaningless without one.
#: A video-only bucket must never draw these, so their weights are zeroed rather
#: than producing a sample that asks for audio conditioning it does not have.
_AUDIO_MODES: frozenset[ConditionMode] = frozenset(
    {ConditionMode.VIDEO_TO_AUDIO, ConditionMode.AUDIO_TO_VIDEO}
)


def default_mode_weights() -> tuple[tuple[ConditionMode, float], ...]:
    """Return the shipped task mixture.

    The mixture is deliberately dominated by ``JOINT``: the conditional tasks
    are cheap to learn because they are strictly easier — part of the answer is
    given — so a small fraction of the budget buys most of the benefit, and
    spending more starves the unconditional task that every other capability is
    built on.

    Returns:
        ``(mode, weight)`` pairs ordered by mode value. Ordered rather than a
        mapping so the multinomial draw is byte-identical across processes; a
        dict literal would be stable in CPython but nothing guarantees that a
        config loader preserves it.
    """
    return (
        (ConditionMode.JOINT, 0.55),
        (ConditionMode.VIDEO_TO_AUDIO, 0.05),
        (ConditionMode.AUDIO_TO_VIDEO, 0.05),
        (ConditionMode.CONTINUATION, 0.10),
        (ConditionMode.INPAINT, 0.05),
        (ConditionMode.IMAGE_TO_VIDEO, 0.20),
    )


@dataclass(frozen=True, slots=True)
class ConditioningPlan:
    """The per-sample task and clean-anchor masks for one microbatch.

    The two anchor masks are in *latent grid* space, matching
    ``MediaBatch.video_mask`` and ``MediaBatch.audio_mask`` exactly, rather than
    in token space. That is on purpose: the patchifier is the only component
    that knows the patch geometry, and it already reduces a grid mask to a token
    mask with the correct ``all`` semantics — a patch is a clean anchor only if
    every latent inside it is clean. Producing token-space masks here would
    duplicate that geometry and let it drift.

    Args:
        condition_mode: ``(batch,)`` int64 :class:`ConditionMode` per sample.
        video_conditioned: ``(batch, frames, height, width)`` bool. True marks a
            latent supplied clean.
        audio_conditioned: ``(batch, frames)`` bool.
        drop_text: ``(batch,)`` bool. True marks a sample whose text context is
            nulled for classifier-free guidance.
    """

    condition_mode: torch.Tensor
    video_conditioned: torch.Tensor
    audio_conditioned: torch.Tensor
    drop_text: torch.Tensor

    @property
    def batch_size(self) -> int:
        """Number of samples covered by this plan."""
        return int(self.condition_mode.shape[0])

    def validate(self, batch: MediaBatch) -> None:
        """Check the plan against the batch it was drawn for.

        Args:
            batch: The batch this plan conditions.

        Raises:
            ValueError: On a shape, device, or mode-range violation.
            TypeError: On a dtype violation.
        """
        size = batch.spec.batch_size
        _, _, frames, height, width = batch.spec.video_shape
        require_shape("condition_mode", self.condition_mode, (size,))
        require_shape(
            "video_conditioned", self.video_conditioned, (size, frames, height, width)
        )
        require_shape(
            "audio_conditioned", self.audio_conditioned, (size, batch.spec.audio_tokens)
        )
        require_shape("drop_text", self.drop_text, (size,))
        require_dtype("condition_mode", self.condition_mode, torch.int64)
        require_dtype("video_conditioned", self.video_conditioned, torch.bool)
        require_dtype("audio_conditioned", self.audio_conditioned, torch.bool)
        require_dtype("drop_text", self.drop_text, torch.bool)
        valid = torch.zeros_like(self.condition_mode, dtype=torch.bool)
        for mode in ConditionMode:
            valid |= self.condition_mode == int(mode)
        if not bool(valid.all()):
            raise ValueError("condition_mode contains unsupported values")
        for name in ("video_conditioned", "audio_conditioned", "drop_text"):
            tensor: torch.Tensor = getattr(self, name)
            if tensor.device != batch.device:
                raise ValueError(
                    f"{name} must be on batch device {batch.device}; "
                    f"got {tensor.device}"
                )


@runtime_checkable
class ConditioningSampler(Protocol):
    """Draws the task mixture and the clean-anchor masks for a batch.

    Implement this to change the curriculum of tasks — to add a new conditional
    mode, to anneal the mixture over training, to make the mixture depend on the
    bucket — without touching the objective or the model.
    """

    def sample(self, batch: MediaBatch, *, rng: RNGStreams) -> ConditioningPlan:
        """Draw a plan for one microbatch."""
        ...


@dataclass(frozen=True, slots=True)
class MultiTaskConditioning:
    """The shipped sampler: a weighted task mixture with per-modality dropout.

    Args:
        mode_weights: ``(mode, weight)`` pairs. Weights need not sum to one and
            are renormalised, so a config can express "twice as much I2V" by
            changing one number.
        continuation_prefix: ``(low, high)`` fraction of frames kept clean for
            ``CONTINUATION``. Drawn per sample rather than fixed so the model
            learns to extend from any amount of context; a fixed prefix produces
            a model that only extends well from exactly that length.
        inpaint_ratio: ``(low, high)`` fraction of latents kept clean for
            ``INPAINT``.
        image_frames: Latent frames kept clean for ``IMAGE_TO_VIDEO``. One
            latent frame, not one pixel frame — a temporally compressing VAE
            folds several pixel frames into the first latent frame, and the
            model conditions on the latent.
        text_dropout: Probability of nulling the text context, enabling
            classifier-free guidance on the prompt.
        video_anchor_dropout: Probability of clearing the video anchor while
            keeping the task label, enabling guidance on the video anchor.
        audio_anchor_dropout: Probability of clearing the audio anchor.
    """

    mode_weights: tuple[tuple[ConditionMode, float], ...] = field(
        default_factory=default_mode_weights
    )
    continuation_prefix: tuple[float, float] = (0.125, 0.5)
    inpaint_ratio: tuple[float, float] = (0.1, 0.6)
    image_frames: int = 1
    text_dropout: float = 0.1
    video_anchor_dropout: float = 0.1
    audio_anchor_dropout: float = 0.1

    def __post_init__(self) -> None:
        """Validate the mixture, the anchor ranges, and the dropout rates."""
        if not self.mode_weights:
            raise ValueError("mode_weights must contain at least one entry")
        seen: list[ConditionMode] = []
        for mode, weight in self.mode_weights:
            if not isinstance(mode, ConditionMode):
                raise TypeError(
                    f"mode_weights key must be a ConditionMode; got {mode!r}"
                )
            if mode in seen:
                raise ValueError(f"mode_weights contains {mode.name} twice")
            seen.append(mode)
            if weight < 0.0:
                raise ValueError(f"weight for {mode.name} must be >= 0; got {weight!r}")
        if sum(weight for _, weight in self.mode_weights) <= 0.0:
            raise ValueError("mode_weights must contain at least one positive weight")
        for name in ("continuation_prefix", "inpaint_ratio"):
            low, high = getattr(self, name)
            require_probability(f"{name}[0]", low)
            require_probability(f"{name}[1]", high)
            if low > high:
                raise ValueError(
                    f"{name} must be ordered (low, high); got {(low, high)}"
                )
        if isinstance(self.image_frames, bool) or self.image_frames < 1:
            raise ValueError(f"image_frames must be >= 1; got {self.image_frames!r}")
        for name in ("text_dropout", "video_anchor_dropout", "audio_anchor_dropout"):
            require_probability(name, getattr(self, name))

    @classmethod
    def from_weights(
        cls,
        weights: Mapping[ConditionMode | str | int, float],
        **options: float,
    ) -> MultiTaskConditioning:
        """Build a sampler from a config-friendly mapping of task weights.

        Args:
            weights: Mode name, value, or enum member to weight.
            **options: Remaining constructor arguments.

        Returns:
            The constructed sampler.

        Raises:
            KeyError: If a key does not name a :class:`ConditionMode`.
        """
        resolved: list[tuple[ConditionMode, float]] = []
        for key, weight in weights.items():
            if isinstance(key, ConditionMode):
                mode = key
            elif isinstance(key, int):
                mode = ConditionMode(key)
            else:
                try:
                    mode = ConditionMode[key.upper()]
                except KeyError as error:
                    names = ", ".join(m.name for m in ConditionMode)
                    raise KeyError(
                        f"unknown condition mode {key!r}; expected one of {names}"
                    ) from error
            resolved.append((mode, float(weight)))
        # Sorted so the multinomial category order is a function of the mode set
        # alone, never of the mapping's insertion order.
        resolved.sort(key=lambda pair: int(pair[0]))
        return cls(mode_weights=tuple(resolved), **options)  # type: ignore[arg-type]

    def sample(self, batch: MediaBatch, *, rng: RNGStreams) -> ConditioningPlan:
        """Draw the task mixture and clean-anchor masks for one microbatch.

        Args:
            batch: The batch to condition. Only its shapes, masks, and spec are
                read; the latents themselves are not touched.
            rng: Purpose-separated streams. Only ``rng.conditioning`` is used.

        Returns:
            The plan.
        """
        device = batch.device
        generator = rng.conditioning
        size = batch.spec.batch_size
        _, _, frames, height, width = batch.spec.video_shape
        audio_frames = batch.spec.audio_tokens

        modes = self._draw_modes(size, batch=batch, generator=generator, device=device)
        video = torch.zeros(
            (size, frames, height, width), dtype=torch.bool, device=device
        )
        audio = torch.zeros((size, audio_frames), dtype=torch.bool, device=device)

        frame_index = torch.arange(frames, device=device)[None, :]

        # IMAGE_TO_VIDEO and CONTINUATION differ only in how many leading latent
        # frames are clean, so one prefix-length vector covers both. Building it
        # for the whole batch and selecting with `where` keeps the whole draw
        # branch-free, which matters because a per-sample Python loop here would
        # launch O(batch) kernels every step.
        prefix = torch.zeros((size,), dtype=torch.int64, device=device)
        low, high = self.continuation_prefix
        ratio = low + (high - low) * uniform(
            (size,), generator=generator, device=device
        )
        # At least one clean frame (an empty prefix is just JOINT) and at most
        # frames - 1 (a full prefix leaves nothing to supervise, so the sample
        # would contribute an exactly-zero loss and waste a slot in the batch).
        continuation = (ratio * frames).to(torch.int64).clamp(1, max(frames - 1, 1))
        image = torch.full_like(prefix, min(self.image_frames, max(frames - 1, 1)))
        prefix = torch.where(
            modes == int(ConditionMode.CONTINUATION), continuation, prefix
        )
        prefix = torch.where(modes == int(ConditionMode.IMAGE_TO_VIDEO), image, prefix)
        video |= (frame_index < prefix[:, None])[:, :, None, None]

        # INPAINT: an independent Bernoulli draw per latent at a per-sample rate.
        # The alternative, choosing exactly k positions, needs a sort or a
        # randperm over the whole grid every step and buys nothing: the model
        # never sees k, only the realised mask.
        low, high = self.inpaint_ratio
        keep = low + (high - low) * uniform((size,), generator=generator, device=device)
        draw = uniform(
            (size, frames, height, width), generator=generator, device=device
        )
        inpaint = draw < keep[:, None, None, None]
        video |= inpaint & (modes == int(ConditionMode.INPAINT))[:, None, None, None]

        # Whole-stream anchors for the cross-modal tasks.
        video |= (modes == int(ConditionMode.VIDEO_TO_AUDIO))[:, None, None, None]
        audio |= (modes == int(ConditionMode.AUDIO_TO_VIDEO))[:, None]

        # Independent per-modality dropout. Applied after the masks are built so
        # a dropped sample keeps its task label: the model then sees the same
        # requested task with and without its anchor, which is precisely the
        # conditional/null pair modality-aware guidance needs at inference.
        video &= ~bernoulli(
            (size,), self.video_anchor_dropout, generator=generator, device=device
        )[:, None, None, None]
        audio &= ~bernoulli(
            (size,), self.audio_anchor_dropout, generator=generator, device=device
        )[:, None]

        # A padding latent can never be a clean anchor: it carries no signal, and
        # marking it conditioned would remove it from the loss twice over while
        # telling the model to attend to zeros as if they were content.
        video &= batch.video_mask
        audio &= batch.audio_mask

        drop_text = bernoulli(
            (size,), self.text_dropout, generator=generator, device=device
        )
        return ConditioningPlan(
            condition_mode=modes,
            video_conditioned=video,
            audio_conditioned=audio,
            drop_text=drop_text,
        )

    def _draw_modes(
        self,
        size: int,
        *,
        batch: MediaBatch,
        generator: torch.Generator,
        device: torch.device,
    ) -> torch.Tensor:
        """Draw one :class:`ConditionMode` per sample from the mixture.

        Args:
            size: Batch size.
            batch: The batch, consulted for whether audio is present.
            generator: The conditioning generator.
            device: Device for the result.

        Returns:
            ``(size,)`` int64 mode values.
        """
        modes = [mode for mode, _ in self.mode_weights]
        weights = [weight for _, weight in self.mode_weights]
        if not batch.spec.has_audio:
            # A video-only bucket cannot express an audio anchor. Zeroing the
            # weight and renormalising is the only safe response: silently
            # remapping to JOINT would misreport the realised mixture, and
            # raising would make one video-only bucket fatal to a mixed run.
            weights = [
                0.0 if mode in _AUDIO_MODES else weight
                for mode, weight in zip(modes, weights, strict=True)
            ]
        table = torch.tensor(weights, dtype=torch.float32)
        indices = categorical(table, size, generator=generator, device=device)
        lookup = torch.tensor(
            [int(mode) for mode in modes], dtype=torch.int64, device=device
        )
        return lookup[indices]
