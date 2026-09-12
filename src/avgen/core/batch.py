"""The dense batch contract at the data boundary.

:class:`MediaBatch` is what a data source produces: clean latents on a dense
grid, exactly as a VAE emits them and as a shard file stores them. It is the
last place in the framework where a latent is a grid rather than a sequence — a
:class:`~avgen.core.patchify.Patchifier` converts it into
:class:`~avgen.core.tokens.TokenStream` objects before any model sees it, and
:class:`~avgen.core.model_input.ModelInput` is what a model actually consumes.

Keeping the boundary dense and the interior sequential is deliberate. Dense is
the right shape for storage and for a convolutional autoencoder; sequential is
the only shape that context parallelism, variable-resolution batching, and
sequence packing can work with.

The batch is registered as a pytree node, so ``torch.compile``, FSDP hooks, and
distributed collectives can traverse it without special-casing.

**Audio is optional throughout.** A text-to-video model passes an audio tensor
with zero frames; every audio code path degenerates to a no-op with no branch in
the model's public signature. That is what lets one architecture cover T2V,
V2A, A2V, and joint audio-video generation without four separate trainers.

**Positions are physical time, in seconds, not indices.** Video at 24 fps and
audio at 86 latent frames per second have no common index space, and resampling
one to the other throws away the alignment that audio-video generation is
entirely about. Carrying float seconds lets rotary embeddings put both streams
in one phase space, and lets a single model train on mixed frame rates.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field, replace
from enum import IntEnum
from typing import Any, Self, cast

import torch
from torch.utils import _pytree

from avgen.core._validate import (
    require_dimension,
    require_dtype,
    require_floating,
    require_name,
    require_same_device,
    require_shape,
)
from avgen.core.tensors import TensorBundle, TensorBundleSpec

__all__ = [
    "ConditionMode",
    "MediaBatch",
    "MediaBatchSpec",
    "null_text_conditioning",
    "stack_batches",
]


class ConditionMode(IntEnum):
    """Per-sample generation task.

    One model trained across all of these is strictly stronger than a model
    trained on ``JOINT`` alone: the conditional tasks teach it what a clean
    anchor looks like, which is the same signal that makes editing, extension,
    and image-to-video work at inference time.

    New modes append at the end. The integer value is written into checkpoints
    and shard headers, so existing values never change.
    """

    JOINT = 0
    """Generate every modality from text alone."""

    VIDEO_TO_AUDIO = 1
    """Video is a clean anchor; generate audio."""

    AUDIO_TO_VIDEO = 2
    """Audio is a clean anchor; generate video."""

    VIDEO_ONLY = 3
    """Generate video; ignore the audio stream entirely."""

    CONTINUATION = 4
    """A temporal prefix is clean; extend it forwards."""

    INPAINT = 5
    """An arbitrary token mask is clean; fill the rest."""

    IMAGE_TO_VIDEO = 6
    """The first video frame is clean; animate it."""

    VIDEO_TO_VIDEO = 7
    """A low-fidelity or partially-noised video guides generation."""


@dataclass(frozen=True, slots=True)
class MediaBatchSpec:
    """Static schema for one admitted batch bucket.

    The spec is the *static* half of a batch: shapes, timebases, and the identity
    of the codecs that produced the latents. It is hashable and JSON-safe, and it
    is what ``torch.compile`` keys on. Two batches with the same spec reuse one
    compiled graph; a new spec is a new bucket and a new compile.

    Timebases are exact rationals rather than floats. ``30000/1001`` (NTSC) is
    not representable in binary floating point, and a drift of one part in
    100,000 accumulated over a 10-second clip is enough to visibly desynchronise
    audio from video.

    Args:
        schema_version: Version of this contract. Bumped only on a breaking
            change to the field set.
        bucket_id: Identifier of the resolution/duration bucket this batch
            belongs to. Distinct buckets keep distinct compiled graphs.
        video_shape: ``(batch, channels, frames, height, width)``.
        audio_shape: ``(batch, channels, frames)``. Zero frames means no audio.
        text_shape: ``(batch, tokens, width)``. Zero tokens means no text.
        video_timebase_num: Numerator of the video latent frame rate.
        video_timebase_den: Denominator of the video latent frame rate.
        audio_timebase_num: Numerator of the audio latent frame rate.
        audio_timebase_den: Denominator of the audio latent frame rate.
        video_codec_id: Fingerprint of the video autoencoder. Latents from two
            different VAEs are not interchangeable and must never be mixed into
            one batch.
        audio_codec_id: Fingerprint of the audio autoencoder.
        target_spec: Schema for optional auxiliary supervision targets.
    """

    schema_version: int
    bucket_id: int
    video_shape: tuple[int, ...]
    audio_shape: tuple[int, ...]
    text_shape: tuple[int, ...]
    video_timebase_num: int
    video_timebase_den: int
    audio_timebase_num: int
    audio_timebase_den: int
    video_codec_id: str
    audio_codec_id: str
    target_spec: TensorBundleSpec = field(default_factory=TensorBundleSpec)

    def __post_init__(self) -> None:
        """Validate ranks, dimensions, timebases, and codec identity."""
        require_dimension("schema_version", self.schema_version)
        require_dimension("bucket_id", self.bucket_id, allow_zero=True)
        self._validate_shape("video_shape", self.video_shape, rank=5)
        self._validate_shape(
            "audio_shape", self.audio_shape, rank=3, zero_allowed_index=2
        )
        self._validate_shape(
            "text_shape", self.text_shape, rank=3, zero_allowed_index=1
        )
        batch_sizes = {self.video_shape[0], self.audio_shape[0], self.text_shape[0]}
        if len(batch_sizes) != 1:
            raise ValueError(
                "video, audio, and text batch dimensions must match; got "
                f"{self.video_shape[0]}, {self.audio_shape[0]}, {self.text_shape[0]}"
            )
        for name in (
            "video_timebase_num",
            "video_timebase_den",
            "audio_timebase_num",
            "audio_timebase_den",
        ):
            require_dimension(name, getattr(self, name))
        for name in ("video_codec_id", "audio_codec_id"):
            require_name(name, getattr(self, name))

    @staticmethod
    def _validate_shape(
        name: str,
        shape: tuple[int, ...],
        *,
        rank: int,
        zero_allowed_index: int | None = None,
    ) -> None:
        if len(shape) != rank:
            raise ValueError(f"{name} must have rank {rank}; got {shape}")
        for index, value in enumerate(shape):
            require_dimension(
                f"{name}[{index}]", value, allow_zero=index == zero_allowed_index
            )

    @property
    def batch_size(self) -> int:
        """Number of samples in the batch."""
        return self.video_shape[0]

    @property
    def video_fps(self) -> float:
        """Video latent frame rate in frames per second."""
        return self.video_timebase_num / self.video_timebase_den

    @property
    def audio_fps(self) -> float:
        """Audio latent frame rate in frames per second."""
        return self.audio_timebase_num / self.audio_timebase_den

    @property
    def has_audio(self) -> bool:
        """Whether this bucket carries an audio stream."""
        return self.audio_shape[2] > 0

    @property
    def has_text(self) -> bool:
        """Whether this bucket carries a text conditioning stream."""
        return self.text_shape[1] > 0

    @property
    def video_tokens(self) -> int:
        """Video tokens per sample, before any patchification."""
        _, _, frames, height, width = self.video_shape
        return frames * height * width

    @property
    def audio_tokens(self) -> int:
        """Audio tokens per sample."""
        return self.audio_shape[2]

    @property
    def sequence_length(self) -> int:
        """Total tokens per sample across every generative modality.

        This is the number that decides whether a batch needs context
        parallelism: attention cost is quadratic in it, and activation memory is
        linear in it.
        """
        return self.video_tokens + self.audio_tokens

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation."""
        values = asdict(self)
        for key in ("video_shape", "audio_shape", "text_shape"):
            values[key] = list(cast("tuple[int, ...]", values[key]))
        values["target_spec"] = self.target_spec.to_dict()
        return values

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> Self:
        """Construct a schema from its JSON-safe representation.

        Args:
            values: Mapping produced by :meth:`to_dict`.

        Returns:
            The restored spec.
        """
        converted = dict(values)
        for key in ("video_shape", "audio_shape", "text_shape"):
            converted[key] = tuple(cast("list[int]", converted[key]))
        converted["target_spec"] = (
            TensorBundleSpec.from_dict(
                cast("dict[str, object]", converted["target_spec"])
            )
            if "target_spec" in converted
            else TensorBundleSpec()
        )
        return cls(**cast("dict[str, Any]", converted))


@dataclass(frozen=True, slots=True)
class MediaBatch:
    """Clean, aligned latents produced by a data source.

    Args:
        video: ``(batch, channels, frames, height, width)`` clean video latents.
        audio: ``(batch, channels, frames)`` clean audio latents; may be empty.
        text: ``(batch, tokens, width)`` frozen text-encoder features.
        video_mask: ``(batch, frames, height, width)`` validity of each video
            token. False marks padding introduced by bucketing.
        audio_mask: ``(batch, frames)`` validity of each audio token.
        video_positions: ``(batch, frames)`` physical time in seconds.
        audio_positions: ``(batch, frames)`` physical time in seconds.
        sample_ids: ``(batch,)`` stable identifiers, kept on CPU so that logging
            a sample id never forces a device synchronisation inside a step.
        spec: The static schema this batch conforms to.
        text_mask: ``(batch, tokens)`` validity of each text token. Defaults to
            all-valid.
        targets: Optional auxiliary supervision, interpreted by
            ``spec.target_spec``.
    """

    video: torch.Tensor
    audio: torch.Tensor
    text: torch.Tensor
    video_mask: torch.Tensor
    audio_mask: torch.Tensor
    video_positions: torch.Tensor
    audio_positions: torch.Tensor
    sample_ids: torch.Tensor
    spec: MediaBatchSpec
    text_mask: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.bool)
    )
    targets: TensorBundle = field(default_factory=TensorBundle)

    def __post_init__(self) -> None:
        """Fill an omitted text mask with all-valid."""
        if self.text_mask.numel() == 0:
            object.__setattr__(
                self,
                "text_mask",
                torch.ones(
                    self.spec.text_shape[:2], dtype=torch.bool, device=self.text.device
                ),
            )

    @property
    def device(self) -> torch.device:
        """Device the latent tensors live on."""
        return self.video.device

    def to(
        self, device: torch.device | str, *, non_blocking: bool = True
    ) -> MediaBatch:
        """Move every device tensor, leaving ``sample_ids`` on the host.

        Args:
            device: Target device.
            non_blocking: Whether to issue asynchronous copies. Safe and
                strongly preferred when the source tensors are pinned, which is
                what the shipped loaders do.

        Returns:
            A batch on the target device.
        """
        target = torch.device(device)

        def _move(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.to(target, non_blocking=non_blocking)

        return MediaBatch(
            video=_move(self.video),
            audio=_move(self.audio),
            text=_move(self.text),
            video_mask=_move(self.video_mask),
            audio_mask=_move(self.audio_mask),
            video_positions=_move(self.video_positions),
            audio_positions=_move(self.audio_positions),
            sample_ids=self.sample_ids,
            spec=self.spec,
            text_mask=_move(self.text_mask),
            targets=TensorBundle(tuple(_move(value) for value in self.targets.values)),
        )

    def validate(self) -> None:
        """Validate shapes, dtypes, and devices at a data/runtime boundary.

        Called once per batch at the loader boundary, never inside a compiled
        region. The cost is a few microseconds; the alternative is a shape bug
        surfacing as a NaN two hours into a thousand-GPU run.

        Raises:
            ValueError: On a shape, device, or placement violation.
            TypeError: On a dtype violation.
        """
        batch, _, video_time, height, width = self.spec.video_shape
        _, _, audio_time = self.spec.audio_shape
        require_shape("video", self.video, self.spec.video_shape)
        require_shape("audio", self.audio, self.spec.audio_shape)
        require_shape("text", self.text, self.spec.text_shape)
        require_shape("video_mask", self.video_mask, (batch, video_time, height, width))
        require_shape("audio_mask", self.audio_mask, (batch, audio_time))
        require_shape("text_mask", self.text_mask, (batch, self.spec.text_shape[1]))
        require_shape("video_positions", self.video_positions, (batch, video_time))
        require_shape("audio_positions", self.audio_positions, (batch, audio_time))
        require_shape("sample_ids", self.sample_ids, (batch,))

        for name, tensor in (
            ("video", self.video),
            ("audio", self.audio),
            ("text", self.text),
        ):
            require_floating(name, tensor)
        require_dtype("video_mask", self.video_mask, torch.bool)
        require_dtype("audio_mask", self.audio_mask, torch.bool)
        require_dtype("text_mask", self.text_mask, torch.bool)
        require_dtype("video_positions", self.video_positions, torch.float32)
        require_dtype("audio_positions", self.audio_positions, torch.float32)
        require_dtype("sample_ids", self.sample_ids, torch.int64)
        if self.sample_ids.device.type != "cpu":
            raise ValueError(
                f"sample_ids must remain on CPU; got {self.sample_ids.device}"
            )
        require_same_device(
            "video",
            self.video,
            (
                ("audio", self.audio),
                ("text", self.text),
                ("video_mask", self.video_mask),
                ("audio_mask", self.audio_mask),
                ("text_mask", self.text_mask),
                ("video_positions", self.video_positions),
                ("audio_positions", self.audio_positions),
            ),
        )
        self.targets.validate(self.spec.target_spec, device=self.video.device)


def stack_batches(batches: Sequence[MediaBatch]) -> MediaBatch:
    """Concatenate same-spec batches along the batch dimension.

    Args:
        batches: One or more batches sharing an identical spec.

    Returns:
        A single batch whose batch dimension is the sum of the inputs'.

    Raises:
        ValueError: If the sequence is empty, the specs differ, or the target
            bundles have different widths.
    """
    if not batches:
        raise ValueError("stack_batches requires at least one batch")
    if len(batches) == 1:
        return batches[0]
    base = batches[0].spec
    for batch in batches[1:]:
        if batch.spec != base:
            raise ValueError("stack_batches requires an identical MediaBatchSpec")
    target_widths = {len(batch.targets) for batch in batches}
    if len(target_widths) != 1:
        raise ValueError("stack_batches requires matching target bundles")
    total = sum(batch.spec.batch_size for batch in batches)
    spec = replace(
        base,
        video_shape=(total, *base.video_shape[1:]),
        audio_shape=(total, *base.audio_shape[1:]),
        text_shape=(total, *base.text_shape[1:]),
        target_spec=base.target_spec.with_batch_size(total),
    )

    def _cat(name: str) -> torch.Tensor:
        return torch.cat([getattr(batch, name) for batch in batches], dim=0)

    targets = TensorBundle(
        tuple(
            torch.cat([batch.targets.values[index] for batch in batches], dim=0)
            for index in range(next(iter(target_widths)))
        )
    )
    return MediaBatch(
        video=_cat("video"),
        audio=_cat("audio"),
        text=_cat("text"),
        video_mask=_cat("video_mask"),
        audio_mask=_cat("audio_mask"),
        text_mask=_cat("text_mask"),
        video_positions=_cat("video_positions"),
        audio_positions=_cat("audio_positions"),
        sample_ids=_cat("sample_ids"),
        targets=targets,
        spec=spec,
    )


def null_text_conditioning(
    text: torch.Tensor,
    text_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the unconditional text context matching a conditional one.

    Classifier-free guidance needs an unconditional branch that the model has
    actually been trained on. Zeroing both the features and the mask — rather
    than encoding an empty string — gives an exact, encoder-independent null
    that costs nothing to produce at inference time.

    Args:
        text: Conditional text features.
        text_mask: Conditional text mask.

    Returns:
        The zeroed features and mask.
    """
    return torch.zeros_like(text), torch.zeros_like(text_mask)


def _flatten_batch(batch: MediaBatch) -> tuple[list[torch.Tensor], MediaBatchSpec]:
    return [
        batch.video,
        batch.audio,
        batch.text,
        batch.video_mask,
        batch.audio_mask,
        batch.text_mask,
        batch.video_positions,
        batch.audio_positions,
        batch.sample_ids,
        *batch.targets.values,
    ], batch.spec


def _unflatten_batch(values: Iterable[object], spec: object) -> MediaBatch:
    tensors = [cast("torch.Tensor", value) for value in values]
    return MediaBatch(
        video=tensors[0],
        audio=tensors[1],
        text=tensors[2],
        video_mask=tensors[3],
        audio_mask=tensors[4],
        text_mask=tensors[5],
        video_positions=tensors[6],
        audio_positions=tensors[7],
        sample_ids=tensors[8],
        targets=TensorBundle(tuple(tensors[9:])),
        spec=cast("MediaBatchSpec", spec),
    )


_pytree.register_pytree_node(
    MediaBatch,
    _flatten_batch,
    _unflatten_batch,
    serialized_type_name="avgen.MediaBatch",
    to_dumpable_context=MediaBatchSpec.to_dict,
    from_dumpable_context=MediaBatchSpec.from_dict,
)
