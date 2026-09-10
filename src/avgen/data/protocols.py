"""Record types, buckets, and the protocols every data source satisfies.

This module is the seam between *storage* and *training*. On the storage side a
sample is a small pile of dense tensors with some provenance attached
(:class:`LatentSample`); on the training side a step consumes
:class:`~avgen.core.batch.MediaBatch`. :func:`collate_samples` is the only place
that crosses, and it is deliberately the only place, so there is exactly one
implementation of "what a batch means" to audit.

Three ideas carry most of the weight here.

**A sample is described before it is read.** :class:`SampleDescriptor` carries
the shapes, timebases, and codec identity of a sample without touching its
tensors. Bucketing a corpus of a hundred million clips must not require a full
read pass over the latents, so every store answers ``descriptor(i)`` from its
manifest and only materialises tensors when a batch is actually being built.

**Buckets are the unit of shape stability.** :class:`Bucket` is a
(duration, resolution) class; a batch never mixes two of them. That is what
gives ``torch.compile`` a small finite set of static shapes to specialise on
instead of a new graph per clip.

**Provenance is part of the sample, not a side channel.** Two latents produced
by two different autoencoders are not the same kind of number, and averaging a
loss over both trains a model on a blend of two representations that neither
decoder can invert. The codec fingerprint travels with the tensor and is
checked at every join.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch

from avgen.core._validate import (
    require_dimension,
    require_name,
    require_positive,
)
from avgen.core.batch import MediaBatch, MediaBatchSpec
from avgen.core.tensors import TensorBundle, TensorBundleSpec

__all__ = [
    "Bucket",
    "BucketPlan",
    "DataSource",
    "LatentSample",
    "SampleDescriptor",
    "SampleStore",
    "collate_samples",
]

#: Version of the record layout in this module. Written into shard manifests and
#: into :class:`~avgen.core.batch.MediaBatchSpec`, so a reader can refuse data it
#: does not understand instead of misinterpreting it.
DATA_SCHEMA_VERSION: int = 1


@dataclass(frozen=True, slots=True)
class Bucket:
    """One (duration, resolution) class that batches are built inside.

    A bucket is pure static geometry in *latent* units — the units a shard
    actually stores — not pixel units. Converting to pixels would bake the
    autoencoder's downsampling factor into the sampler, and that factor differs
    between codecs.

    The design decision this class encodes is **bucketing instead of padding**.
    The rejected alternative is to pad every clip to the largest shape in the
    corpus and rely on the attention mask. That is correct and unaffordable:
    attention cost is quadratic in sequence length, so padding a 256x256 clip up
    to 1024x1024 does sixteen times the token work and two hundred and fifty six
    times the attention work, essentially all of it on tokens that are masked
    out. Bucketing makes cost proportional to real content, and as a bonus
    hands ``torch.compile`` a handful of static shapes rather than one per clip.

    Args:
        bucket_id: Stable identifier. Written into batch specs and shard
            manifests, so a value is never reused for a different geometry.
        frames: Latent frames per sample.
        height: Latent rows per sample.
        width: Latent columns per sample.
        audio_frames: Latent audio frames per sample. Zero means the bucket is
            video-only.
        patch_frames: Temporal patch size the model will apply.
        patch_height: Spatial patch height the model will apply.
        patch_width: Spatial patch width the model will apply.
        audio_patch_frames: Temporal patch size for the audio stream.

    Raises:
        ValueError: If a dimension is non-positive, or if a patch size does not
            divide its axis. Divisibility is checked here rather than in the
            model because a bucket that cannot be patchified is a configuration
            error, and configuration errors must fail before a job is launched.
    """

    bucket_id: int
    frames: int
    height: int
    width: int
    audio_frames: int = 0
    patch_frames: int = 1
    patch_height: int = 2
    patch_width: int = 2
    audio_patch_frames: int = 1

    def __post_init__(self) -> None:
        """Validate dimensions and patch divisibility."""
        require_dimension("bucket_id", self.bucket_id, allow_zero=True)
        for name in ("frames", "height", "width"):
            require_positive(name, getattr(self, name))
        require_dimension("audio_frames", self.audio_frames, allow_zero=True)
        for name in (
            "patch_frames",
            "patch_height",
            "patch_width",
            "audio_patch_frames",
        ):
            require_positive(name, getattr(self, name))
        for axis, patch in (
            ("frames", "patch_frames"),
            ("height", "patch_height"),
            ("width", "patch_width"),
        ):
            extent = getattr(self, axis)
            size = getattr(self, patch)
            if extent % size != 0:
                raise ValueError(
                    f"bucket {self.bucket_id}: {axis}={extent} must be divisible "
                    f"by {patch}={size}; choose bucket geometry that the model's "
                    "patch size tiles exactly rather than cropping at run time"
                )
        if self.audio_frames % self.audio_patch_frames != 0:
            raise ValueError(
                f"bucket {self.bucket_id}: audio_frames={self.audio_frames} must "
                f"be divisible by audio_patch_frames={self.audio_patch_frames}"
            )

    @property
    def aspect_ratio(self) -> float:
        """Width divided by height, in latent cells."""
        return self.width / self.height

    @property
    def latent_cells(self) -> int:
        """Latent cells per sample, before patchification."""
        return self.frames * self.height * self.width

    @property
    def video_tokens(self) -> int:
        """Video tokens per sample after patchification."""
        return (
            (self.frames // self.patch_frames)
            * (self.height // self.patch_height)
            * (self.width // self.patch_width)
        )

    @property
    def audio_tokens(self) -> int:
        """Audio tokens per sample after patchification."""
        return self.audio_frames // self.audio_patch_frames

    @property
    def tokens(self) -> int:
        """Total generative tokens per sample.

        This is the number that decides whether a bucket needs context
        parallelism, and it is the honest x-axis unit for a video scaling law:
        two samples of different resolution are not comparable units of data,
        but their tokens are.
        """
        return self.video_tokens + self.audio_tokens

    @property
    def has_audio(self) -> bool:
        """Whether this bucket carries an audio stream."""
        return self.audio_frames > 0

    def duration_seconds(self, *, timebase_num: int, timebase_den: int) -> float:
        """Return the clip duration implied by a latent frame rate.

        Args:
            timebase_num: Numerator of the latent frame rate.
            timebase_den: Denominator of the latent frame rate.

        Returns:
            Duration in seconds.

        Raises:
            ValueError: If either component of the timebase is non-positive.
        """
        require_positive("timebase_num", timebase_num)
        require_positive("timebase_den", timebase_den)
        return self.frames * timebase_den / timebase_num

    def describe(self) -> str:
        """Return a one-line human summary for logs and plan tables."""
        audio = f" a={self.audio_frames}" if self.has_audio else ""
        return (
            f"bucket {self.bucket_id}: {self.frames}x{self.height}x{self.width}"
            f"{audio} ar={self.aspect_ratio:.3f} tokens={self.tokens}"
        )


@dataclass(frozen=True, slots=True)
class BucketPlan:
    """An ordered set of buckets and the mixture weights over them.

    Weights are the *target* proportion of batches drawn from each bucket, not
    the proportion of samples available. Those two are almost never the same: a
    corpus is usually dominated by short low-resolution clips, while the model
    you want at the end needs to have seen long high-resolution ones. Decoupling
    the mixture from the corpus census is the whole point.

    Args:
        buckets: Buckets in a stable order. Order is part of the contract
            because curriculum weight vectors are positional.
        weights: One non-negative weight per bucket. Empty means uniform.
            Normalised on construction, so callers may pass counts, fractions,
            or arbitrary positive numbers.

    Raises:
        ValueError: If the plan is empty, ids repeat, the weight vector has the
            wrong length, a weight is negative or non-finite, or every weight is
            zero.
    """

    buckets: tuple[Bucket, ...]
    weights: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        """Validate uniqueness and normalise the mixture weights."""
        if not self.buckets:
            raise ValueError("BucketPlan requires at least one bucket")
        ids = [bucket.bucket_id for bucket in self.buckets]
        duplicates = sorted({value for value in ids if ids.count(value) > 1})
        if duplicates:
            raise ValueError(
                f"bucket ids must be unique; duplicates: {duplicates}. A reused id "
                "makes two different geometries share one compiled graph."
            )
        weights = self.weights or tuple(1.0 for _ in self.buckets)
        if len(weights) != len(self.buckets):
            raise ValueError(
                f"weights must have one entry per bucket; got {len(weights)} "
                f"weights for {len(self.buckets)} buckets"
            )
        for index, weight in enumerate(weights):
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError(
                    f"weights[{index}] must be a finite non-negative number; "
                    f"got {weight!r}"
                )
        total = math.fsum(weights)
        if total <= 0.0:
            raise ValueError("bucket weights must not all be zero")
        object.__setattr__(
            self, "weights", tuple(weight / total for weight in weights)
        )

    def __len__(self) -> int:
        """Return the number of buckets."""
        return len(self.buckets)

    def __iter__(self) -> Iterator[Bucket]:
        """Iterate buckets in plan order."""
        return iter(self.buckets)

    def index_of(self, bucket_id: int) -> int:
        """Return the plan position of a bucket id.

        Args:
            bucket_id: Identifier to look up.

        Returns:
            Zero-based position in :attr:`buckets`.

        Raises:
            KeyError: If no bucket carries that id.
        """
        for index, bucket in enumerate(self.buckets):
            if bucket.bucket_id == bucket_id:
                return index
        known = ", ".join(str(bucket.bucket_id) for bucket in self.buckets)
        raise KeyError(f"unknown bucket_id {bucket_id}; known ids: {known}")

    def get(self, bucket_id: int) -> Bucket:
        """Return the bucket carrying an id.

        Args:
            bucket_id: Identifier to look up.

        Returns:
            The matching bucket.

        Raises:
            KeyError: If no bucket carries that id.
        """
        return self.buckets[self.index_of(bucket_id)]

    @property
    def token_budget(self) -> int:
        """Tokens per sample in the largest bucket.

        Memory planning and context-parallel degree selection are sized off the
        worst case, not the average: a job that fits on average and OOMs on the
        long bucket is a job that dies four hours in.
        """
        return max(bucket.tokens for bucket in self.buckets)

    def nearest(self, *, frames: int, height: int, width: int) -> Bucket:
        """Return the bucket a raw clip geometry should be admitted into.

        The heuristic, in priority order:

        1. **Aspect ratio**, compared in log space so 2:1 and 1:2 are equally
           far from 1:1. Getting this wrong means either letterboxing (teaching
           the model to draw black bars) or anisotropic squashing (teaching it
           that faces are ovals).
        2. **Do not upscale.** A bucket larger than the source invents detail
           the encoder never saw, and the model learns to reproduce resampling
           artefacts.
        3. **Duration proximity**, again in log space.

        Ties break on ``bucket_id`` so the assignment is deterministic; a
        ``set``-ordered or dict-ordered tiebreak would make two runs with the
        same seed disagree.

        Args:
            frames: Latent frames available in the source clip.
            height: Latent rows available in the source clip.
            width: Latent columns available in the source clip.

        Returns:
            The chosen bucket.

        Raises:
            ValueError: If any argument is non-positive.
        """
        for name, value in (("frames", frames), ("height", height), ("width", width)):
            require_positive(name, value)
        source_ratio = width / height
        source_cells = height * width

        def _score(bucket: Bucket) -> tuple[float, float, float, int]:
            ratio_gap = abs(math.log(bucket.aspect_ratio / source_ratio))
            upscale = max(
                0.0, math.log((bucket.height * bucket.width) / source_cells)
            )
            frame_gap = abs(math.log(bucket.frames / frames))
            return (ratio_gap, upscale, frame_gap, bucket.bucket_id)

        return min(self.buckets, key=_score)

    def describe(self) -> str:
        """Return a multi-line summary of the plan and its mixture."""
        lines = [f"BucketPlan({len(self.buckets)} buckets)"]
        for bucket, weight in zip(self.buckets, self.weights, strict=True):
            lines.append(f"  {bucket.describe()} weight={weight:.4f}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class SampleDescriptor:
    """Everything needed to place a sample in a bucket, without reading it.

    A store answers this from its manifest. The rejected alternative — deriving
    shapes by opening each sample — costs a full read pass over the corpus every
    time a job starts, which on a real dataset is hours of I/O to compute
    numbers that were already known at write time.

    Args:
        sample_id: Stable global identifier, unique within a corpus.
        video_frames: Latent video frames stored for this sample.
        height: Latent rows.
        width: Latent columns.
        audio_frames: Latent audio frames; zero when the sample has no audio.
        text_tokens: Text-encoder tokens; zero when the sample has no caption.
        text_width: Text-encoder feature width. Must stay positive even when
            ``text_tokens`` is zero, because the batch spec carries the width.
        video_channels: Latent video channels.
        audio_channels: Latent audio channels.
        video_timebase_num: Numerator of the video latent frame rate.
        video_timebase_den: Denominator of the video latent frame rate.
        audio_timebase_num: Numerator of the audio latent frame rate.
        audio_timebase_den: Denominator of the audio latent frame rate.
        video_codec_id: Fingerprint of the autoencoder that produced the video
            latents.
        audio_codec_id: Fingerprint of the autoencoder that produced the audio
            latents.
    """

    sample_id: int
    video_frames: int
    height: int
    width: int
    audio_frames: int
    text_tokens: int
    text_width: int
    video_channels: int
    audio_channels: int
    video_timebase_num: int
    video_timebase_den: int
    audio_timebase_num: int
    audio_timebase_den: int
    video_codec_id: str
    audio_codec_id: str

    def __post_init__(self) -> None:
        """Validate identifiers, dimensions, timebases, and codec names."""
        require_dimension("sample_id", self.sample_id, allow_zero=True)
        for name in (
            "video_frames",
            "height",
            "width",
            "text_width",
            "video_channels",
            "audio_channels",
            "video_timebase_num",
            "video_timebase_den",
            "audio_timebase_num",
            "audio_timebase_den",
        ):
            require_positive(name, getattr(self, name))
        for name in ("audio_frames", "text_tokens"):
            require_dimension(name, getattr(self, name), allow_zero=True)
        for name in ("video_codec_id", "audio_codec_id"):
            require_name(name, getattr(self, name))

    @property
    def codec_key(self) -> tuple[str, str, int, int, int, int]:
        """Identity that two samples must share before they may be batched.

        Mixing latents from two autoencoders, or from two frame rates, produces
        a batch whose entries are not the same kind of number. The loss still
        goes down; the decoded video is mush. This key is what makes that
        mistake impossible rather than merely unlikely.
        """
        return (
            self.video_codec_id,
            self.audio_codec_id,
            self.video_timebase_num,
            self.video_timebase_den,
            self.audio_timebase_num,
            self.audio_timebase_den,
        )


@dataclass(frozen=True, slots=True)
class LatentSample:
    """One sample's dense latents, exactly as a shard stores them.

    Tensors are unbatched: video is ``(channels, frames, height, width)``, not
    ``(1, channels, frames, height, width)``. Carrying the leading batch axis in
    storage would mean every read allocates a view whose only purpose is to be
    squeezed again in :func:`collate_samples`.

    Args:
        descriptor: Static description of this sample.
        video: ``(channels, frames, height, width)`` clean video latents.
        audio: ``(channels, frames)`` clean audio latents; may have zero frames.
        text: ``(tokens, width)`` frozen text-encoder features; may have zero
            tokens.
        video_positions: ``(frames,)`` float32 physical time in seconds.
        audio_positions: ``(frames,)`` float32 physical time in seconds.
        text_mask: ``(tokens,)`` bool validity of each text token.
        valid_video_frames: How many leading video frames are real content. The
            remainder is bucket padding. ``-1`` means every frame is real.
        valid_audio_frames: Same, for audio.
        targets: Optional auxiliary supervision, interpreted by a
            :class:`~avgen.core.tensors.TensorBundleSpec`.
        target_spec: Spec interpreting ``targets`` for a single sample.
    """

    descriptor: SampleDescriptor
    video: torch.Tensor
    audio: torch.Tensor
    text: torch.Tensor
    video_positions: torch.Tensor
    audio_positions: torch.Tensor
    text_mask: torch.Tensor
    valid_video_frames: int = -1
    valid_audio_frames: int = -1
    targets: TensorBundle = field(default_factory=TensorBundle)
    target_spec: TensorBundleSpec = field(default_factory=TensorBundleSpec)

    @property
    def sample_id(self) -> int:
        """Stable global identifier."""
        return self.descriptor.sample_id

    def validate(self) -> None:
        """Check that the tensors agree with the descriptor.

        Called at the store boundary, once per sample read, never in a hot loop.
        A shape disagreement caught here is a one-line error; the same
        disagreement caught downstream is a silent broadcast that trains for
        hours before anyone notices.

        Raises:
            ValueError: On a shape, dtype, or device disagreement.
        """
        info = self.descriptor
        expected: tuple[tuple[str, torch.Tensor, tuple[int, ...]], ...] = (
            (
                "video",
                self.video,
                (info.video_channels, info.video_frames, info.height, info.width),
            ),
            ("audio", self.audio, (info.audio_channels, info.audio_frames)),
            ("text", self.text, (info.text_tokens, info.text_width)),
            ("video_positions", self.video_positions, (info.video_frames,)),
            ("audio_positions", self.audio_positions, (info.audio_frames,)),
            ("text_mask", self.text_mask, (info.text_tokens,)),
        )
        for name, tensor, shape in expected:
            if tuple(tensor.shape) != shape:
                raise ValueError(
                    f"sample {info.sample_id}: {name} shape must be {shape}; "
                    f"got {tuple(tensor.shape)}"
                )
        if self.text_mask.dtype is not torch.bool:
            raise ValueError(
                f"sample {info.sample_id}: text_mask must be bool; "
                f"got {self.text_mask.dtype}"
            )
        for name in ("video_positions", "audio_positions"):
            tensor = getattr(self, name)
            if tensor.dtype is not torch.float32:
                raise ValueError(
                    f"sample {info.sample_id}: {name} must be float32 seconds; "
                    f"got {tensor.dtype}"
                )
        for name, limit in (
            ("valid_video_frames", info.video_frames),
            ("valid_audio_frames", info.audio_frames),
        ):
            value = getattr(self, name)
            if value != -1 and not 0 <= value <= limit:
                raise ValueError(
                    f"sample {info.sample_id}: {name}={value} must be -1 or within "
                    f"[0, {limit}]"
                )
        self.targets.validate(self.target_spec)


@runtime_checkable
class SampleStore(Protocol):
    """Random-access, read-only access to a corpus of latent samples.

    Random access rather than iteration is what makes deterministic resumable
    shuffling possible: a resumable stream must be able to jump to sample
    ``n`` of a permutation, and a generator-based reader cannot.
    """

    def __len__(self) -> int:
        """Return the number of samples in the store."""
        ...

    def __getitem__(self, index: int) -> LatentSample:
        """Return one materialised sample.

        Args:
            index: Position within the store.

        Returns:
            The sample's dense latents.
        """
        ...

    def descriptor(self, index: int) -> SampleDescriptor:
        """Return one sample's static description without reading its tensors.

        Args:
            index: Position within the store.

        Returns:
            The sample's descriptor.
        """
        ...


@runtime_checkable
class DataSource(Protocol):
    """A resumable stream of training batches.

    Satisfies :class:`avgen.core.state.Stateful`, so the checkpoint layer can
    save a data source's position without knowing what kind of source it is.
    That is not a convenience: a run that resumes with the right weights and the
    wrong data position silently retrains on samples it has already seen, and
    nothing in the loss curve reports it.
    """

    def __iter__(self) -> Iterator[MediaBatch]:
        """Iterate batches from the current cursor position."""
        ...

    def state_dict(self) -> Mapping[str, Any]:
        """Return the cursor state needed to resume this stream exactly."""
        ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a cursor produced by :meth:`state_dict`.

        Args:
            state: Mapping produced by :meth:`state_dict`.
        """
        ...


def collate_samples(
    samples: Sequence[LatentSample],
    *,
    bucket_id: int,
    schema_version: int = DATA_SCHEMA_VERSION,
) -> MediaBatch:
    """Stack same-geometry samples into one dense batch.

    Every sample in the sequence must share a geometry *and* a
    :attr:`SampleDescriptor.codec_key`. Both are checked, and both raise rather
    than coercing: silently padding a mismatched sample would hide a bucketing
    bug, and silently mixing codecs would hide a much worse one.

    Masks are derived from ``valid_video_frames`` / ``valid_audio_frames``, so a
    clip that is slightly shorter than its bucket contributes its real frames
    and nothing else. The padded frames still occupy memory — that is the price
    of a static shape — but they contribute no gradient.

    Args:
        samples: One or more samples with identical descriptors apart from
            ``sample_id`` and the valid-frame counts.
        bucket_id: Bucket the batch belongs to, recorded in the spec so
            ``torch.compile`` keys on it.
        schema_version: Version stamped into the batch spec.

    Returns:
        A dense :class:`~avgen.core.batch.MediaBatch` on CPU.

    Raises:
        ValueError: If the sequence is empty, geometries disagree, or codec
            fingerprints disagree.
    """
    if not samples:
        raise ValueError("collate_samples requires at least one sample")
    head = samples[0].descriptor
    for sample in samples[1:]:
        other = sample.descriptor
        if other.codec_key != head.codec_key:
            raise ValueError(
                "cannot batch latents from different codecs or timebases; "
                f"sample {head.sample_id} has {head.codec_key} and sample "
                f"{other.sample_id} has {other.codec_key}"
            )
        geometry = (
            other.video_frames,
            other.height,
            other.width,
            other.audio_frames,
            other.text_tokens,
            other.text_width,
            other.video_channels,
            other.audio_channels,
        )
        expected = (
            head.video_frames,
            head.height,
            head.width,
            head.audio_frames,
            head.text_tokens,
            head.text_width,
            head.video_channels,
            head.audio_channels,
        )
        if geometry != expected:
            raise ValueError(
                "collate_samples requires identical geometry within a bucket; "
                f"sample {other.sample_id} has {geometry}, expected {expected}"
            )

    batch = len(samples)
    video = torch.stack([sample.video for sample in samples], dim=0)
    audio = torch.stack([sample.audio for sample in samples], dim=0)
    text = torch.stack([sample.text for sample in samples], dim=0)
    text_mask = torch.stack([sample.text_mask for sample in samples], dim=0)
    video_positions = torch.stack(
        [sample.video_positions for sample in samples], dim=0
    )
    audio_positions = torch.stack(
        [sample.audio_positions for sample in samples], dim=0
    )
    # sample_ids stay on CPU for the life of the batch: logging one must never
    # be a reason to synchronise the device inside a training step.
    sample_ids = torch.tensor(
        [sample.sample_id for sample in samples], dtype=torch.int64
    )

    frame_index = torch.arange(head.video_frames).view(1, -1)
    valid_video = torch.tensor(
        [
            sample.descriptor.video_frames
            if sample.valid_video_frames < 0
            else sample.valid_video_frames
            for sample in samples
        ],
        dtype=torch.int64,
    ).view(-1, 1)
    video_mask = (frame_index < valid_video).view(batch, head.video_frames, 1, 1)
    video_mask = video_mask.expand(batch, head.video_frames, head.height, head.width)

    audio_index = torch.arange(head.audio_frames).view(1, -1)
    valid_audio = torch.tensor(
        [
            sample.descriptor.audio_frames
            if sample.valid_audio_frames < 0
            else sample.valid_audio_frames
            for sample in samples
        ],
        dtype=torch.int64,
    ).view(-1, 1)
    audio_mask = audio_index < valid_audio

    target_spec = samples[0].target_spec.with_batch_size(batch)
    targets = TensorBundle(
        tuple(
            torch.stack(
                [sample.targets.values[index] for sample in samples], dim=0
            )
            for index in range(len(samples[0].target_spec))
        )
    )

    spec = MediaBatchSpec(
        schema_version=schema_version,
        bucket_id=bucket_id,
        video_shape=(
            batch,
            head.video_channels,
            head.video_frames,
            head.height,
            head.width,
        ),
        audio_shape=(batch, head.audio_channels, head.audio_frames),
        text_shape=(batch, head.text_tokens, head.text_width),
        video_timebase_num=head.video_timebase_num,
        video_timebase_den=head.video_timebase_den,
        audio_timebase_num=head.audio_timebase_num,
        audio_timebase_den=head.audio_timebase_den,
        video_codec_id=head.video_codec_id,
        audio_codec_id=head.audio_codec_id,
        target_spec=target_spec,
    )
    return MediaBatch(
        video=video,
        audio=audio,
        text=text,
        video_mask=video_mask.contiguous(),
        audio_mask=audio_mask.contiguous(),
        video_positions=video_positions,
        audio_positions=audio_positions,
        sample_ids=sample_ids,
        spec=spec,
        text_mask=text_mask,
        targets=targets,
    )
