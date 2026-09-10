"""Building a :class:`~avgen.core.model_input.ModelInput` for each inference task.

This module is the reason avgen has no train/inference skew, and the claim is
worth being precise about rather than asserting.

At training time, a conditioning sampler picks a task per sample and marks some
latents as clean anchors. The objective then builds the model's input by:

1. mixing the clean anchors back into the noisy latents at the anchored
   positions, so the model sees ``x_t`` everywhere except the anchors and the
   *clean* value there;
2. building a per-element noise level that is ``sigma`` at free positions and
   exactly ``0`` at anchored ones;
3. patchifying with ``conditioned=`` set to the anchor mask, so the patchifier
   reduces it with ``all`` and the loss excludes those tokens.

At inference time, the functions here do **exactly the same three things, in the
same order, through the same** :class:`~avgen.core.patchify.Patchifier`. There is
no second implementation to drift. A first frame supplied to
:func:`condition_first_frame` produces byte-for-byte the tensor layout that the
training-time image-to-video task produced, which is the only reason a model
trained on that task behaves at inference the way its training loss suggested it
would.

The failure this prevents is specific and common: an inference path that anchors
by *overwriting the prediction* after each step, rather than by handing the model
a clean input with a zero noise level, teaches the model nothing about the anchor
during the step. The model then denoises the anchored region as if it were noise,
the overwrite hides the mismatch at the region boundary, and the result is a
video whose first frame is correct and whose second frame is unrelated to it.

**Conventions.** ``sigma`` is flow time in ``[0, 1]``. Positions are physical
seconds. The video grid is ``(batch, channels, frames, height, width)``; the
audio grid is ``(batch, channels, frames)`` and is lifted to a rank-5 grid with
unit spatial extent so that one patchifier code path serves both.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import cast

import torch

from avgen.core.batch import ConditionMode
from avgen.core.model_input import ModelInput
from avgen.core.patchify import GridPatchifier, Patchifier
from avgen.core.tokens import PatchLayout, TextContext, TokenStream

__all__ = [
    "StreamState",
    "build_model_input",
    "condition_first_frame",
    "condition_mask",
    "condition_stream",
    "condition_temporal_prefix",
    "default_audio_patchifier",
    "first_frame_mask",
    "temporal_prefix_mask",
    "unconditional_input",
]


def default_audio_patchifier() -> GridPatchifier:
    """Return the patchifier used for the audio stream.

    Unit spatial patches, because the audio grid has unit spatial extent. Audio
    reuses the video patchifier machinery rather than getting its own, so that
    the coordinate convention, the mask reduction, and the conditioning reduction
    are literally the same code for both modalities.

    Returns:
        A patchifier with unit spatial patch size.
    """
    return GridPatchifier(patch_frames=1, patch_height=1, patch_width=1)


@dataclass(frozen=True, slots=True)
class StreamState:
    """One modality's sampler state plus whatever is clean about it.

    Args:
        latents: Current noisy latents. ``(batch, channels, frames, height,
            width)`` for video, ``(batch, channels, frames)`` for audio.
        fps: Latent frames per second. Physical, not an index rate — see
            :mod:`avgen.core.patchify` for why the framework carries seconds.
        anchor: Clean latent values, shaped like ``latents``. ``None`` when the
            stream has no anchors. Values at unanchored positions are ignored.
        anchor_mask: Boolean mask of anchored positions, shaped like ``latents``
            with the channel axis dropped. ``None`` when there are no anchors.
        start_seconds: Physical time of the first latent frame. Non-zero when
            continuing a clip, so that a continuation's coordinates carry on from
            where the prefix ended rather than restarting at zero.
        valid_mask: Optional validity mask shaped like ``anchor_mask``, marking
            real (non-padding) latents. Defaults to all-valid.

    Raises:
        ValueError: If ``fps`` is not positive, only one of ``anchor`` and
            ``anchor_mask`` is given, or a mask does not match the latents.
    """

    latents: torch.Tensor
    fps: float
    anchor: torch.Tensor | None = None
    anchor_mask: torch.Tensor | None = None
    start_seconds: float = 0.0
    valid_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        """Validate the stream geometry at construction."""
        if not math.isfinite(self.fps) or self.fps <= 0.0:
            raise ValueError(f"fps must be finite and positive; got {self.fps!r}")
        if not math.isfinite(self.start_seconds):
            raise ValueError(
                f"start_seconds must be finite; got {self.start_seconds!r}"
            )
        if self.latents.ndim not in (3, 5):
            raise ValueError(
                "latents must be a rank-5 video grid or a rank-3 audio grid; got "
                f"{tuple(self.latents.shape)}"
            )
        if (self.anchor is None) != (self.anchor_mask is None):
            raise ValueError(
                "anchor and anchor_mask must be supplied together; got "
                f"anchor={'set' if self.anchor is not None else 'None'}, "
                f"anchor_mask={'set' if self.anchor_mask is not None else 'None'}"
            )
        if self.anchor is not None and tuple(self.anchor.shape) != tuple(
            self.latents.shape
        ):
            raise ValueError(
                f"anchor shape must be {tuple(self.latents.shape)}; got "
                f"{tuple(self.anchor.shape)}"
            )
        expected = self.element_shape
        for name in ("anchor_mask", "valid_mask"):
            mask: torch.Tensor | None = getattr(self, name)
            if mask is None:
                continue
            if tuple(mask.shape) != expected:
                raise ValueError(
                    f"{name} shape must be {expected}; got {tuple(mask.shape)}"
                )
            if mask.dtype is not torch.bool:
                raise TypeError(f"{name} must be bool; got {mask.dtype}")

    @property
    def is_audio(self) -> bool:
        """Whether this is the rank-3 audio layout."""
        return cast("bool", self.latents.ndim == 3)

    @property
    def batch_size(self) -> int:
        """Number of samples."""
        return int(self.latents.shape[0])

    @property
    def frames(self) -> int:
        """Latent frames."""
        return int(self.latents.shape[2])

    @property
    def element_shape(self) -> tuple[int, ...]:
        """Shape a per-latent mask must have: the grid without its channel axis."""
        shape = tuple(int(value) for value in self.latents.shape)
        return (shape[0], *shape[2:])

    @property
    def has_anchor(self) -> bool:
        """Whether any position is clean."""
        return self.anchor_mask is not None

    def grid(self) -> torch.Tensor:
        """Return the latents as a rank-5 grid.

        Returns:
            ``(batch, channels, frames, height, width)``. The audio layout gains
            unit spatial extent, which is what lets one patchifier serve both
            modalities instead of two near-duplicate implementations.
        """
        if self.is_audio:
            return self.latents[..., None, None]
        return self.latents

    def with_latents(self, latents: torch.Tensor) -> StreamState:
        """Return a copy carrying new latents.

        Args:
            latents: Replacement latents of the same shape.

        Returns:
            The updated state.
        """
        return replace(self, latents=latents)

    def positions(self) -> torch.Tensor:
        """Return ``(batch, frames)`` physical frame times in seconds.

        Returns:
            Float32 seconds, uniformly spaced at :attr:`fps` starting from
            :attr:`start_seconds`.
        """
        index = torch.arange(
            self.frames, device=self.latents.device, dtype=torch.float32
        )
        row = self.start_seconds + index / self.fps
        return row[None].expand(self.batch_size, self.frames)


def first_frame_mask(
    state: StreamState, *, frames: int = 1, device: torch.device | None = None
) -> torch.Tensor:
    """Return a mask anchoring the leading latent frames of a video grid.

    Args:
        state: The stream whose geometry the mask must match.
        frames: How many leading latent frames are clean. One latent frame is
            *not* one pixel frame: a VAE with 4x temporal compression turns four
            pixel frames into one latent frame, so anchoring a single supplied
            image usually means anchoring exactly one latent frame.
        device: Device for the mask. Defaults to the latents' device.

    Returns:
        A boolean mask shaped like the grid without its channel axis.

    Raises:
        ValueError: If ``frames`` is not in ``[1, state.frames]``.
    """
    if isinstance(frames, bool) or not 1 <= frames <= state.frames:
        raise ValueError(f"frames must be in [1, {state.frames}]; got {frames!r}")
    mask = torch.zeros(
        state.element_shape,
        dtype=torch.bool,
        device=device or state.latents.device,
    )
    mask[:, :frames] = True
    return mask


def temporal_prefix_mask(
    state: StreamState, *, frames: int, device: torch.device | None = None
) -> torch.Tensor:
    """Return a mask anchoring a temporal prefix.

    Identical in construction to :func:`first_frame_mask`; kept as a separate
    name because the *task* is different and the condition mode it implies is
    different, and a reader following a continuation should not have to notice
    that it went through a function named for image-to-video.

    Args:
        state: The stream whose geometry the mask must match.
        frames: Length of the clean prefix in latent frames.
        device: Device for the mask.

    Returns:
        A boolean mask shaped like the grid without its channel axis.

    Raises:
        ValueError: If ``frames`` is not in ``[1, state.frames]``.
    """
    return first_frame_mask(state, frames=frames, device=device)


def build_model_input(
    *,
    video: StreamState,
    sigma: float,
    text: TextContext,
    patchifier: Patchifier,
    condition_mode: ConditionMode = ConditionMode.JOINT,
    audio: StreamState | None = None,
    audio_patchifier: Patchifier | None = None,
    audio_sigma: float | None = None,
) -> ModelInput:
    """Assemble the model input for one sampler step.

    This is the single implementation every task helper routes through, and it is
    deliberately the *same* three-step recipe the training-time objective uses.
    See the module docstring.

    Args:
        video: Video stream state, including any clean anchors.
        sigma: Flow time of the free (unanchored) video positions.
        text: Text conditioning context.
        patchifier: Video patchifier. Must be the one the model exposes, or the
            token layout will not match the weights.
        condition_mode: The task, written into the input so that a model with
            per-task embeddings can condition on it.
        audio: Audio stream state, or ``None`` for a video-only generation.
        audio_patchifier: Audio patchifier. Defaults to
            :func:`default_audio_patchifier`.
        audio_sigma: Flow time of the free audio positions. Defaults to
            ``sigma``. It is separate because video-to-audio holds video at
            ``sigma = 0`` while audio runs down the schedule, and a single shared
            value cannot express that.

    Returns:
        The assembled input, validated.

    Raises:
        ValueError: If ``sigma`` is outside ``[0, 1]`` or the streams disagree on
            batch size.
    """
    _require_sigma("sigma", sigma)
    resolved_audio_sigma = sigma if audio_sigma is None else audio_sigma
    _require_sigma("audio_sigma", resolved_audio_sigma)

    video_stream = _to_stream(video, sigma, patchifier)
    if audio is None:
        audio_stream = TokenStream.empty_like(
            video.batch_size,
            PatchLayout.empty().patch_dim,
            device=video.latents.device,
            dtype=video.latents.dtype,
        )
    else:
        if audio.batch_size != video.batch_size:
            raise ValueError(
                f"audio batch size {audio.batch_size} must match video "
                f"{video.batch_size}"
            )
        audio_stream = _to_stream(
            audio,
            resolved_audio_sigma,
            audio_patchifier or default_audio_patchifier(),
        )

    modes = torch.full(
        (video.batch_size,),
        int(condition_mode),
        dtype=torch.int64,
        device=video.latents.device,
    )
    inputs = ModelInput(
        video=video_stream,
        audio=audio_stream,
        text=text,
        condition_mode=modes,
    )
    # Validation at the boundary, not in the hot loop: this runs once per sampler
    # step, which is far outside any compiled region and cheap next to a forward
    # pass, and it turns a silent shape bug into an error naming the field.
    inputs.validate()
    return inputs


def unconditional_input(
    video: StreamState,
    *,
    sigma: float,
    text: TextContext,
    patchifier: Patchifier,
    audio: StreamState | None = None,
    audio_patchifier: Patchifier | None = None,
) -> ModelInput:
    """Build the input for generation from text alone.

    No structural anchors: every latent is free and carries the same noise level.

    .. note::

       This is **not** the classifier-free-guidance null branch. That is
       :meth:`~avgen.core.model_input.ModelInput.unconditional`, which nulls the
       *text* while keeping the structural conditioning intact. This function
       does the opposite: it keeps the text and supplies no structure. Confusing
       the two produces a guidance difference that measures the wrong thing.

    Args:
        video: Video stream state. Any anchors on it are ignored.
        sigma: Flow time.
        text: Text conditioning context.
        patchifier: Video patchifier.
        audio: Audio stream state, or ``None``.
        audio_patchifier: Audio patchifier.

    Returns:
        The assembled input.
    """
    mode = ConditionMode.JOINT if audio is not None else ConditionMode.VIDEO_ONLY
    return build_model_input(
        video=replace(video, anchor=None, anchor_mask=None),
        sigma=sigma,
        text=text,
        patchifier=patchifier,
        condition_mode=mode,
        audio=(
            replace(audio, anchor=None, anchor_mask=None) if audio is not None else None
        ),
        audio_patchifier=audio_patchifier,
    )


def condition_first_frame(
    video: StreamState,
    first_frame: torch.Tensor,
    *,
    sigma: float,
    text: TextContext,
    patchifier: Patchifier,
    latent_frames: int = 1,
    audio: StreamState | None = None,
    audio_patchifier: Patchifier | None = None,
) -> ModelInput:
    """Build the image-to-video input from a clean first frame.

    Args:
        video: Video stream state carrying the current noisy latents.
        first_frame: Clean latents for the leading frames, shaped either
            ``(batch, channels, latent_frames, height, width)`` or
            ``(batch, channels, height, width)`` for a single frame.
        sigma: Flow time of the free positions.
        text: Text conditioning context.
        patchifier: Video patchifier.
        latent_frames: How many leading *latent* frames the anchor covers.
        audio: Audio stream state, or ``None``.
        audio_patchifier: Audio patchifier.

    Returns:
        The assembled input, in condition mode ``IMAGE_TO_VIDEO``.

    Raises:
        ValueError: If the anchor does not match the video geometry.
    """
    anchored = _place_prefix(video, first_frame, latent_frames)
    return build_model_input(
        video=anchored,
        sigma=sigma,
        text=text,
        patchifier=patchifier,
        condition_mode=ConditionMode.IMAGE_TO_VIDEO,
        audio=audio,
        audio_patchifier=audio_patchifier,
    )


def condition_temporal_prefix(
    video: StreamState,
    prefix: torch.Tensor,
    *,
    sigma: float,
    text: TextContext,
    patchifier: Patchifier,
    audio: StreamState | None = None,
    audio_patchifier: Patchifier | None = None,
) -> ModelInput:
    """Build the continuation input from a clean temporal prefix.

    Extension works by anchoring the tail of the previous clip as the head of the
    next one and generating the rest. The prefix must be long enough for the
    model to infer motion — a single frame fixes appearance but says nothing
    about velocity, which is why a continuation with a one-frame prefix produces
    a plausible video that does not continue the motion it was given.

    Args:
        video: Video stream state carrying the current noisy latents.
        prefix: Clean latents ``(batch, channels, prefix_frames, height, width)``
            occupying the leading frames of the canvas.
        sigma: Flow time of the free positions.
        text: Text conditioning context.
        patchifier: Video patchifier.
        audio: Audio stream state, or ``None``.
        audio_patchifier: Audio patchifier.

    Returns:
        The assembled input, in condition mode ``CONTINUATION``.

    Raises:
        ValueError: If the prefix does not match the video geometry.
    """
    if prefix.ndim != 5:
        raise ValueError(
            "prefix must be a rank-5 grid (batch, channels, frames, height, "
            f"width); got {tuple(prefix.shape)}"
        )
    anchored = _place_prefix(video, prefix, int(prefix.shape[2]))
    return build_model_input(
        video=anchored,
        sigma=sigma,
        text=text,
        patchifier=patchifier,
        condition_mode=ConditionMode.CONTINUATION,
        audio=audio,
        audio_patchifier=audio_patchifier,
    )


def condition_mask(
    video: StreamState,
    known: torch.Tensor,
    mask: torch.Tensor,
    *,
    sigma: float,
    text: TextContext,
    patchifier: Patchifier,
    audio: StreamState | None = None,
    audio_patchifier: Patchifier | None = None,
) -> ModelInput:
    """Build the inpainting input from an arbitrary clean-region mask.

    The most general structural conditioning: any subset of latents can be
    anchored. First-frame and prefix conditioning are special cases, and they
    exist as separate functions only because their condition modes differ and a
    model may embed the mode.

    Note that the mask is over *latents*, not pixels. A VAE's spatial and
    temporal compression means an edit region drawn in pixel space must be
    downsampled — and dilated, because the decoder's receptive field spreads each
    latent over several pixels, so a tight latent mask leaves a halo of
    unmodified pixels around the edit.

    Args:
        video: Video stream state carrying the current noisy latents.
        known: Clean latents shaped like ``video.latents``. Values outside the
            mask are ignored.
        mask: Boolean mask of clean positions, shaped like the grid without its
            channel axis.
        sigma: Flow time of the free positions.
        text: Text conditioning context.
        patchifier: Video patchifier.
        audio: Audio stream state, or ``None``.
        audio_patchifier: Audio patchifier.

    Returns:
        The assembled input, in condition mode ``INPAINT``.

    Raises:
        ValueError: If the mask or the known latents do not match the geometry.
    """
    anchored = replace(video, anchor=known, anchor_mask=mask)
    return build_model_input(
        video=anchored,
        sigma=sigma,
        text=text,
        patchifier=patchifier,
        condition_mode=ConditionMode.INPAINT,
        audio=audio,
        audio_patchifier=audio_patchifier,
    )


def condition_stream(
    *,
    video: StreamState,
    audio: StreamState,
    generate: str,
    sigma: float,
    text: TextContext,
    patchifier: Patchifier,
    audio_patchifier: Patchifier | None = None,
) -> ModelInput:
    """Build the cross-modal input for video-to-audio or audio-to-video.

    One modality is held entirely clean and the other is generated. The clean
    modality is anchored the same way a first frame is — a full-coverage anchor
    mask and a zero noise level — rather than by being routed into a separate
    conditioning encoder. That is what lets a single model do V2A and A2V without
    a second architecture: the task is expressed in the data, not in the graph.

    The anchored stream's noise level is exactly zero, and it must be: a stream
    at ``sigma = 1e-3`` instead of ``0`` is one the model was never shown as a
    conditioning signal during training, and cross-modal alignment degrades
    sharply.

    Args:
        video: Video stream state.
        audio: Audio stream state.
        generate: Which modality to generate — ``"audio"`` or ``"video"``.
        sigma: Flow time of the generated modality.
        text: Text conditioning context.
        patchifier: Video patchifier.
        audio_patchifier: Audio patchifier.

    Returns:
        The assembled input, in condition mode ``VIDEO_TO_AUDIO`` or
        ``AUDIO_TO_VIDEO``.

    Raises:
        ValueError: If ``generate`` is not ``"audio"`` or ``"video"``.
    """
    if generate == "audio":
        anchored_video = _anchor_everything(video)
        return build_model_input(
            video=anchored_video,
            sigma=0.0,
            text=text,
            patchifier=patchifier,
            condition_mode=ConditionMode.VIDEO_TO_AUDIO,
            audio=audio,
            audio_patchifier=audio_patchifier,
            audio_sigma=sigma,
        )
    if generate == "video":
        anchored_audio = _anchor_everything(audio)
        return build_model_input(
            video=video,
            sigma=sigma,
            text=text,
            patchifier=patchifier,
            condition_mode=ConditionMode.AUDIO_TO_VIDEO,
            audio=anchored_audio,
            audio_patchifier=audio_patchifier,
            audio_sigma=0.0,
        )
    raise ValueError(f"generate must be 'audio' or 'video'; got {generate!r}")


def _anchor_everything(state: StreamState) -> StreamState:
    """Mark every latent of a stream as a clean anchor."""
    mask = torch.ones(
        state.element_shape, dtype=torch.bool, device=state.latents.device
    )
    return replace(state, anchor=state.latents, anchor_mask=mask)


def _place_prefix(state: StreamState, anchor: torch.Tensor, frames: int) -> StreamState:
    """Place a clean prefix onto a full-length canvas and mark it anchored."""
    if anchor.ndim == 4:
        anchor = anchor[:, :, None]
    if anchor.ndim != 5:
        raise ValueError(
            "anchor must be a rank-4 single frame or a rank-5 grid; got "
            f"{tuple(anchor.shape)}"
        )
    if int(anchor.shape[2]) != frames:
        raise ValueError(
            f"anchor covers {int(anchor.shape[2])} latent frames but "
            f"latent_frames={frames} was requested"
        )
    canvas = state.latents.clone()
    shape = tuple(int(value) for value in state.latents.shape)
    expected = (shape[0], shape[1], frames, *shape[3:])
    if tuple(anchor.shape) != expected:
        raise ValueError(f"anchor shape must be {expected}; got {tuple(anchor.shape)}")
    canvas[:, :, :frames] = anchor.to(canvas.dtype)
    mask = first_frame_mask(state, frames=frames)
    return replace(state, anchor=canvas, anchor_mask=mask)


def _to_stream(state: StreamState, sigma: float, patchifier: Patchifier) -> TokenStream:
    """Mix anchors into the latents and patchify, exactly as training does.

    The three steps, in the order the training objective performs them:

    1. ``torch.where`` puts the *clean* value at anchored positions. The model
       must receive the clean latent there, not the noisy one with a flag
       alongside it — the flag is redundant information the model would have to
       learn to trust, and every published conditioning scheme that works feeds
       the clean value directly.
    2. The per-element noise level is ``sigma`` at free positions and exactly
       ``0`` at anchored ones. That is what
       :meth:`~avgen.core.tokens.TokenStream.expanded_noise` and the model's
       timestep embedding read, so the model *knows* which regions are clean
       rather than having to infer it from their statistics.
    3. ``conditioned=`` carries the anchor mask into the patchifier, which
       reduces it with ``all`` — a patch is a clean anchor only if every latent
       inside it is clean — and
       :meth:`~avgen.core.tokens.TokenStream.loss_mask` then excludes those
       tokens from any training loss computed over this input.

    When there are no anchors, the per-sample ``(batch,)`` noise fast path is
    used rather than a materialised per-element tensor, which is again exactly
    what training does for the unanchored tasks.

    Args:
        state: The stream state.
        sigma: Flow time of the free positions.
        patchifier: The patchifier to convert through.

    Returns:
        The token stream.
    """
    grid = state.grid()
    batch = state.batch_size
    device = grid.device
    element_shape = tuple(grid.shape[:1] + grid.shape[2:])

    valid = state.valid_mask
    if valid is None:
        valid_grid = torch.ones(element_shape, dtype=torch.bool, device=device)
    else:
        valid_grid = valid.reshape(element_shape)

    if state.anchor_mask is None or state.anchor is None:
        noise_level = torch.full(
            (batch,), float(sigma), dtype=torch.float32, device=device
        )
        return patchifier.to_tokens(
            grid,
            positions=state.positions(),
            mask=valid_grid,
            noise_level=noise_level,
            conditioned=None,
        )

    anchor_mask = state.anchor_mask.reshape(element_shape)
    anchor_grid = state.anchor.reshape(grid.shape)
    mixed = torch.where(anchor_mask[:, None], anchor_grid.to(grid.dtype), grid)
    noise_grid = torch.where(
        anchor_mask,
        torch.zeros((), dtype=torch.float32, device=device),
        torch.full((), float(sigma), dtype=torch.float32, device=device),
    )
    return patchifier.to_tokens(
        mixed,
        positions=state.positions(),
        mask=valid_grid,
        noise_level=noise_grid,
        conditioned=anchor_mask,
    )


def _require_sigma(name: str, value: float) -> None:
    """Reject a flow time outside the closed unit interval."""
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{name} must be a flow time in [0, 1]; got {value!r}. Rectified "
            "flow sigmas never exceed 1; a variance-exploding sigma belongs to "
            "an EDM schedule."
        )
