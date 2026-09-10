"""Offline pixel-to-latent ingest: quality gates and the bridge to shards.

Training reads latents, never pixels. Encoding is done once, offline, and the
result is written to shards — because a video autoencoder forward pass costs
more than the diffusion step it feeds, and paying it every epoch would make the
VAE, not the model, the thing you are training.

This module is the *scaffolding* for that offline pass: the shape of the
decision, not the decision itself.

**On the defaults, which is the important part of this file.** Every numeric
gate below is disabled by default. That is deliberate and it is not laziness.
Quality thresholds are properties of a corpus and a goal, not of a framework: a
minimum duration that is right for a model trained on continuous shots is wrong
for one trained on cuts, a silence floor that is right for speech is wrong for
ambient audio, and a resolution floor that is right for a 720p model throws away
most of the usable data for a 256p one. A framework that shipped tuned numbers
would be shipping one lab's dataset decisions as if they were universal, and
every user who did not read the source would inherit them silently.

So the shipped configuration rejects only what is *definitionally* unusable — a
file that does not decode, a clip with no frames, an exact duplicate — and
everything else is a number you set, with a docstring here explaining which way
each one trades. Measure your corpus, choose, and record the choice in the
config you commit.

**On optional dependencies.** Media decoding and columnar manifest reading are
in the ``data`` extra and are imported inside the functions that need them, so a
training container that only reads shards never installs a demuxer. The
quality-gate logic itself has no dependencies at all, which is what makes it
testable without a video file.
"""

from __future__ import annotations

import hashlib
import importlib
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import torch

from avgen.core._validate import require_dimension, require_positive
from avgen.data.protocols import BucketPlan, LatentSample, SampleDescriptor

__all__ = [
    "ClipProbe",
    "ClipQA",
    "ClipReport",
    "ClipSpec",
    "IngestSummary",
    "QualityGate",
    "RejectionReason",
    "build_latent_sample",
    "content_fingerprint",
    "iter_clip_table",
    "plan_clip_spec",
    "probe_clip",
    "summarise_reports",
    "tensor_fingerprint",
]

#: Decibels below full scale reported for a signal that is exactly zero. A true
#: digital silence has no logarithm, and returning -inf makes every comparison
#: and every mean containing it useless.
SILENT_DBFS: float = -120.0


def _require_module(name: str, extra: str) -> Any:
    """Import an optional dependency, naming the extra that provides it.

    Args:
        name: Module to import.
        extra: The avgen extra that installs it.

    Returns:
        The imported module.

    Raises:
        RuntimeError: If the module is not installed.
    """
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise RuntimeError(
            f"{name} is required for this operation but is not installed; "
            f"install it with: pip install 'avgen[{extra}]'"
        ) from error


class RejectionReason(StrEnum):
    """Why a clip was excluded from a corpus.

    A string enum rather than a free-text message so rejection counts can be
    tallied across a hundred million clips and compared between ingest runs. The
    single most useful artefact of an ingest pass is the histogram of these:
    when a corpus comes out a tenth of the expected size, it says which gate ate
    it.
    """

    DECODE_FAILED = "decode_failed"
    """The container or codec could not be read at all."""

    NO_FRAMES = "no_frames"
    """The stream decoded but contained no video frames."""

    TOO_SHORT = "too_short"
    """Below the configured minimum duration."""

    TOO_LONG = "too_long"
    """Above the configured maximum duration."""

    RESOLUTION_TOO_SMALL = "resolution_too_small"
    """The short side is below the configured minimum."""

    FRAME_RATE_OUT_OF_RANGE = "frame_rate_out_of_range"
    """The frame rate falls outside the configured range."""

    ASPECT_RATIO_OUT_OF_RANGE = "aspect_ratio_out_of_range"
    """The aspect ratio falls outside the configured range."""

    AUDIO_MISSING = "audio_missing"
    """Audio was required and the file has none."""

    AUDIO_SILENT = "audio_silent"
    """The audio track is below the configured loudness floor."""

    DUPLICATE = "duplicate"
    """An identical clip was already admitted."""


@dataclass(frozen=True, slots=True)
class ClipProbe:
    """What a decoder reports about one source clip.

    Deliberately a plain data record with no decoder attached. The quality gates
    consume this and nothing else, so the whole gate layer is testable by
    constructing probes by hand, with no media files, no codecs, and no optional
    dependencies in the test environment.

    Args:
        path: Source location, for reporting only.
        decoded: Whether the file could be read.
        duration_seconds: Duration of the video stream.
        frame_count: Decoded video frames.
        height: Pixel height.
        width: Pixel width.
        frame_rate_num: Numerator of the exact frame rate.
        frame_rate_den: Denominator of the exact frame rate. Kept rational
            because common broadcast rates (30000/1001 and friends) are not
            representable in binary floating point, and a rounded rate
            accumulates into visible audio-video drift over a long clip.
        has_audio: Whether an audio stream is present.
        audio_sample_rate: Audio sample rate in hertz; zero when absent.
        audio_channels: Audio channel count; zero when absent.
        audio_rms_dbfs: Root-mean-square level of the audio in decibels below
            full scale, or :data:`SILENT_DBFS` for digital silence.
        content_hash: Stable digest of the clip's content, used for exact
            duplicate detection.
        decode_error: Decoder message when ``decoded`` is false.
    """

    path: str
    decoded: bool = True
    duration_seconds: float = 0.0
    frame_count: int = 0
    height: int = 0
    width: int = 0
    frame_rate_num: int = 0
    frame_rate_den: int = 1
    has_audio: bool = False
    audio_sample_rate: int = 0
    audio_channels: int = 0
    audio_rms_dbfs: float = SILENT_DBFS
    content_hash: str = ""
    decode_error: str = ""

    @property
    def frame_rate(self) -> float:
        """Frames per second, or zero when unknown."""
        if self.frame_rate_den == 0:
            return 0.0
        return self.frame_rate_num / self.frame_rate_den

    @property
    def aspect_ratio(self) -> float:
        """Width divided by height, or zero when unknown."""
        return self.width / self.height if self.height else 0.0

    @property
    def short_side(self) -> int:
        """Smaller of the two pixel dimensions."""
        return min(self.height, self.width)


@dataclass(frozen=True, slots=True)
class ClipQA:
    """Corpus admission gates.

    **Every numeric gate defaults to disabled.** These are placeholders for the
    numbers you will choose after measuring your own corpus, not values tuned on
    anyone else's. Each field's documentation says which way it trades so that
    choice can be made from evidence.

    Args:
        require_decodable: Reject files the decoder cannot open. Leave on; a
            file that does not decode has no content to trade off.
        min_duration_seconds: Reject clips shorter than this. Trade-off: short
            clips are cheap and numerous, and they teach texture and appearance
            well, but a model trained mostly on them never sees enough temporal
            context to learn motion that persists. Raising this cuts the corpus
            hard, because the duration distribution of scraped video has a very
            heavy short tail. ``None`` disables the gate.
        max_duration_seconds: Reject clips longer than this. Trade-off: long
            clips are the only source of long-horizon structure, but they are
            expensive (attention is quadratic in duration) and they usually
            contain scene cuts, which teach the model to change scene abruptly
            mid-generation. Most pipelines cut long sources into shots rather
            than rejecting them. ``None`` disables the gate.
        min_short_side: Reject clips whose smaller pixel dimension is below
            this. Trade-off: upscaling low-resolution sources to the training
            resolution teaches the model to reproduce upscaling artefacts,
            which are then baked into every sample it generates. Setting this
            above the training resolution's short side is the safe choice and
            the expensive one. ``None`` disables the gate.
        min_frame_rate: Reject clips below this frame rate. Trade-off: a low
            frame rate means large inter-frame motion, which reads as
            judder and is hard to model; but rejecting it discards animation
            and archival footage entirely. ``None`` disables the gate.
        max_frame_rate: Reject clips above this frame rate. High-rate sources
            are usually fine after temporal resampling, so this is mostly a
            guard against malformed metadata. ``None`` disables the gate.
        aspect_ratio_range: Reject clips whose aspect ratio falls outside this
            ``(low, high)`` range. Trade-off: extreme ratios cost a bucket of
            their own or get cropped, and cropping systematically removes
            whatever the framing put at the edges. ``None`` disables the gate.
        require_audio: Reject clips with no audio stream. Set this for
            audio-video training; leave it off for text-to-video, where
            discarding silent footage removes a large fraction of usable data
            for no benefit.
        min_audio_rms_dbfs: Reject clips whose audio is quieter than this.
            Trade-off: a track that is silent or near-silent teaches an
            audio-video model that video predicts silence, which is a strong
            prior and a wrong one; but a floor set too high rejects quiet
            ambient audio, which is exactly the content an environmental model
            needs. Measure the distribution before choosing. ``None`` disables
            the gate.
        detect_duplicates: Reject a clip whose content hash has already been
            admitted. Leave on. Duplicates are not a matter of taste: an
            over-represented clip is memorised rather than learned, they inflate
            any evaluation that shares a source with training, and in a scraped
            corpus they are common enough to matter.

    Raises:
        ValueError: If a bound is not finite, a range is inverted, or a size is
            negative.
    """

    require_decodable: bool = True
    min_duration_seconds: float | None = None
    max_duration_seconds: float | None = None
    min_short_side: int | None = None
    min_frame_rate: float | None = None
    max_frame_rate: float | None = None
    aspect_ratio_range: tuple[float, float] | None = None
    require_audio: bool = False
    min_audio_rms_dbfs: float | None = None
    detect_duplicates: bool = True

    def __post_init__(self) -> None:
        """Validate that every configured bound is finite and ordered."""
        for name in (
            "min_duration_seconds",
            "max_duration_seconds",
            "min_frame_rate",
            "max_frame_rate",
            "min_audio_rms_dbfs",
        ):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite when set; got {value!r}")
        if self.min_short_side is not None:
            require_positive("min_short_side", self.min_short_side)
        for low_name, high_name in (
            ("min_duration_seconds", "max_duration_seconds"),
            ("min_frame_rate", "max_frame_rate"),
        ):
            low = getattr(self, low_name)
            high = getattr(self, high_name)
            if low is not None and high is not None and low > high:
                raise ValueError(
                    f"{low_name}={low} must not exceed {high_name}={high}"
                )
        if self.aspect_ratio_range is not None:
            low, high = self.aspect_ratio_range
            if not (math.isfinite(low) and math.isfinite(high)) or low <= 0.0:
                raise ValueError(
                    f"aspect_ratio_range must be finite and positive; got "
                    f"{self.aspect_ratio_range!r}"
                )
            if low > high:
                raise ValueError(
                    f"aspect_ratio_range is inverted; got {self.aspect_ratio_range!r}"
                )


@dataclass(frozen=True, slots=True)
class ClipReport:
    """The verdict on one clip.

    Args:
        path: Source location.
        accepted: Whether the clip passed every gate.
        reasons: Every gate the clip failed, in evaluation order. All of them,
            not just the first: a corpus report that stops at the first failure
            makes it look as though one gate is responsible for a rejection that
            several gates independently agree on.
        content_hash: The clip's content digest, when known.
        duplicate_of: Path of the previously admitted clip this one duplicates.
    """

    path: str
    accepted: bool
    reasons: tuple[RejectionReason, ...] = ()
    content_hash: str = ""
    duplicate_of: str = ""


class QualityGate:
    """Applies :class:`ClipQA` across a corpus, tracking duplicates as it goes.

    Stateful because duplicate detection is inherently stateful: whether a clip
    is a duplicate is a property of what came before it. Everything else is a
    pure function of the probe.

    Ordering matters and is the caller's responsibility. When two identical
    clips are seen, the *first* is admitted, so the ingest order decides which
    copy survives. Feed paths in a deterministic order (sorted, or from a
    manifest) if you want the admitted set to be reproducible.

    Args:
        config: Gates to apply. Defaults to the permissive shipped
            configuration, which rejects only undecodable, empty, and duplicate
            clips.
    """

    __slots__ = ("_config", "_seen")

    def __init__(self, config: ClipQA | None = None) -> None:
        self._config = config or ClipQA()
        self._seen: dict[str, str] = {}

    @property
    def config(self) -> ClipQA:
        """The gates being applied."""
        return self._config

    @property
    def admitted(self) -> int:
        """Number of distinct clips admitted so far."""
        return len(self._seen)

    def reset(self) -> None:
        """Forget every seen content hash.

        Call this between corpora, never within one: a reset mid-pass makes the
        duplicate gate silently stop working from that point on.
        """
        self._seen.clear()

    def evaluate(self, probe: ClipProbe) -> ClipReport:
        """Apply every gate to one clip.

        Args:
            probe: What the decoder reported.

        Returns:
            The verdict, listing every failed gate.
        """
        config = self._config
        reasons: list[RejectionReason] = []

        if config.require_decodable and not probe.decoded:
            # Nothing downstream is meaningful on a file that did not decode, so
            # this is the one gate that short-circuits.
            return ClipReport(
                path=probe.path,
                accepted=False,
                reasons=(RejectionReason.DECODE_FAILED,),
            )
        if probe.frame_count <= 0:
            reasons.append(RejectionReason.NO_FRAMES)
        if (
            config.min_duration_seconds is not None
            and probe.duration_seconds < config.min_duration_seconds
        ):
            reasons.append(RejectionReason.TOO_SHORT)
        if (
            config.max_duration_seconds is not None
            and probe.duration_seconds > config.max_duration_seconds
        ):
            reasons.append(RejectionReason.TOO_LONG)
        if (
            config.min_short_side is not None
            and probe.short_side < config.min_short_side
        ):
            reasons.append(RejectionReason.RESOLUTION_TOO_SMALL)
        if (
            config.min_frame_rate is not None
            and probe.frame_rate < config.min_frame_rate
        ) or (
            config.max_frame_rate is not None
            and probe.frame_rate > config.max_frame_rate
        ):
            reasons.append(RejectionReason.FRAME_RATE_OUT_OF_RANGE)
        if config.aspect_ratio_range is not None:
            low, high = config.aspect_ratio_range
            if not low <= probe.aspect_ratio <= high:
                reasons.append(RejectionReason.ASPECT_RATIO_OUT_OF_RANGE)
        if config.require_audio and not probe.has_audio:
            reasons.append(RejectionReason.AUDIO_MISSING)
        if (
            config.min_audio_rms_dbfs is not None
            and probe.has_audio
            and probe.audio_rms_dbfs < config.min_audio_rms_dbfs
        ):
            reasons.append(RejectionReason.AUDIO_SILENT)

        duplicate_of = ""
        if config.detect_duplicates and probe.content_hash:
            existing = self._seen.get(probe.content_hash)
            if existing is not None:
                duplicate_of = existing
                reasons.append(RejectionReason.DUPLICATE)

        accepted = not reasons
        if accepted and probe.content_hash:
            self._seen[probe.content_hash] = probe.path
        return ClipReport(
            path=probe.path,
            accepted=accepted,
            reasons=tuple(reasons),
            content_hash=probe.content_hash,
            duplicate_of=duplicate_of,
        )


def content_fingerprint(payload: bytes | Sequence[bytes]) -> str:
    """Return a stable digest of raw bytes.

    Exact-match only. It detects a byte-identical or pixel-identical copy and
    nothing else: a re-encode, a crop, a watermark, or a one-frame trim all
    produce a different hash. Near-duplicate detection needs a perceptual
    embedding and an index, which is a component this framework does not ship
    and does not pretend to.

    Args:
        payload: Bytes, or a sequence of byte chunks hashed in order.

    Returns:
        A hex digest.
    """
    digest = hashlib.sha256()
    if isinstance(payload, bytes):
        digest.update(payload)
    else:
        for chunk in payload:
            digest.update(chunk)
    return digest.hexdigest()


def tensor_fingerprint(tensor: torch.Tensor) -> str:
    """Return a stable digest of a decoded tensor's contents.

    Hashes the dtype, shape, and raw buffer, so two tensors with the same values
    but different shapes do not collide. The tensor is made contiguous first;
    hashing a strided view would digest whatever happens to sit between its
    elements.

    Args:
        tensor: Decoded pixels, latents, or any other tensor.

    Returns:
        A hex digest.
    """
    contiguous = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(str(tuple(contiguous.shape)).encode("utf-8"))
    digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ClipSpec:
    """The target geometry one source clip should be resampled to.

    This is the output of the ingest decision and the input to the codec: it
    says which window of the source to take, at what rate, and at what size. It
    is in *pixel* units, because that is what a decoder speaks; the latent
    geometry it becomes is this divided by the autoencoder's downsampling
    factors, which is what :func:`plan_clip_spec` computes against a bucket.

    Args:
        frames: Video frames to produce.
        height: Pixel height to produce.
        width: Pixel width to produce.
        frame_rate_num: Numerator of the output frame rate.
        frame_rate_den: Denominator of the output frame rate.
        start_seconds: Offset into the source at which the window begins.
        audio_sample_rate: Output audio rate in hertz; zero drops the audio.
        audio_channels: Output audio channels.
        bucket_id: Bucket this clip is being ingested for, recorded so a shard
            can be checked against the plan that produced it.

    Raises:
        ValueError: If a dimension or rate is non-positive, or the start offset
            is negative or not finite.
    """

    frames: int
    height: int
    width: int
    frame_rate_num: int
    frame_rate_den: int = 1
    start_seconds: float = 0.0
    audio_sample_rate: int = 0
    audio_channels: int = 1
    bucket_id: int = 0

    def __post_init__(self) -> None:
        """Validate geometry, rate, and window offset."""
        for name in ("frames", "height", "width", "frame_rate_num", "frame_rate_den"):
            require_positive(name, getattr(self, name))
        require_dimension("audio_sample_rate", self.audio_sample_rate, allow_zero=True)
        require_positive("audio_channels", self.audio_channels)
        require_dimension("bucket_id", self.bucket_id, allow_zero=True)
        if not math.isfinite(self.start_seconds) or self.start_seconds < 0.0:
            raise ValueError(
                f"start_seconds must be finite and non-negative; "
                f"got {self.start_seconds!r}"
            )

    @property
    def frame_rate(self) -> float:
        """Output frames per second."""
        return self.frame_rate_num / self.frame_rate_den

    @property
    def duration_seconds(self) -> float:
        """Duration of the output window in seconds."""
        return self.frames * self.frame_rate_den / self.frame_rate_num


def plan_clip_spec(
    probe: ClipProbe,
    plan: BucketPlan,
    *,
    spatial_downsample: int,
    temporal_downsample: int,
    start_seconds: float = 0.0,
    audio_sample_rate: int = 0,
    audio_channels: int = 1,
) -> ClipSpec:
    """Choose the bucket a clip belongs in and the pixel geometry it implies.

    This is where :meth:`BucketPlan.nearest` is consulted — at ingest time, once
    per clip, offline. By the time the loader runs, every stored latent already
    matches a bucket exactly, which is why
    :func:`~avgen.data.loader.assign_buckets` can insist on an exact match
    rather than re-deciding at training time. Deciding twice is how a corpus and
    a plan drift apart.

    Args:
        probe: What the decoder reported about the source.
        plan: The bucket plan the corpus is being built for.
        spatial_downsample: The video autoencoder's spatial compression factor,
            typically 8 or 16.
        temporal_downsample: Its temporal compression factor, typically 1, 4,
            or 8.
        start_seconds: Offset into the source at which to take the window.
        audio_sample_rate: Output audio rate; zero drops audio.
        audio_channels: Output audio channels.

    Returns:
        The target geometry, in pixels, for the chosen bucket.

    Raises:
        ValueError: If the probe has no usable geometry, or the downsampling
            factors are non-positive.
    """
    require_positive("spatial_downsample", spatial_downsample)
    require_positive("temporal_downsample", temporal_downsample)
    if probe.height <= 0 or probe.width <= 0 or probe.frame_count <= 0:
        raise ValueError(
            f"cannot plan an ingest for {probe.path!r}: the probe reports no usable "
            f"geometry ({probe.frame_count} frames, {probe.width}x{probe.height})"
        )
    # Convert the source into latent units before asking the plan, because
    # buckets are latent geometry. Round up so a source that is one pixel short
    # of a bucket is still considered for it rather than falling to the tier
    # below.
    source = plan.nearest(
        frames=max(1, -(-probe.frame_count // temporal_downsample)),
        height=max(1, -(-probe.height // spatial_downsample)),
        width=max(1, -(-probe.width // spatial_downsample)),
    )
    return ClipSpec(
        frames=source.frames * temporal_downsample,
        height=source.height * spatial_downsample,
        width=source.width * spatial_downsample,
        frame_rate_num=probe.frame_rate_num or 1,
        frame_rate_den=probe.frame_rate_den or 1,
        start_seconds=start_seconds,
        audio_sample_rate=audio_sample_rate,
        audio_channels=audio_channels,
        bucket_id=source.bucket_id,
    )


def build_latent_sample(
    *,
    sample_id: int,
    video: torch.Tensor,
    video_timebase: tuple[int, int],
    video_codec_id: str,
    audio: torch.Tensor | None = None,
    audio_timebase: tuple[int, int] = (1, 1),
    audio_codec_id: str = "none",
    text: torch.Tensor | None = None,
    text_mask: torch.Tensor | None = None,
    text_width: int = 1,
    start_seconds: float = 0.0,
    audio_start_seconds: float | None = None,
    valid_video_frames: int = -1,
    valid_audio_frames: int = -1,
) -> LatentSample:
    """Assemble a storable sample from encoded latents.

    The bridge between :mod:`avgen.codecs` and :mod:`avgen.data.shard`. Its only
    real work is deriving the position tensors, and that is the part worth doing
    in one place: positions are **physical seconds**, derived from the exact
    rational timebase, and computing them per-caller is how a pipeline ends up
    with a video stream indexed in frames and an audio stream indexed in samples
    that agree about nothing.

    Args:
        sample_id: Stable global identifier.
        video: ``(channels, frames, height, width)`` encoded video latents.
        video_timebase: ``(numerator, denominator)`` of the video latent rate.
        video_codec_id: Fingerprint of the video autoencoder. Use something
            that changes when the weights change; a name alone is not enough,
            because two checkpoints of "the same" VAE produce incompatible
            latents.
        audio: ``(channels, frames)`` encoded audio latents, or ``None``.
        audio_timebase: ``(numerator, denominator)`` of the audio latent rate.
        audio_codec_id: Fingerprint of the audio autoencoder.
        text: ``(tokens, width)`` frozen text-encoder features, or ``None``.
        text_mask: ``(tokens,)`` validity, defaulting to all-valid.
        text_width: Feature width to record when ``text`` is absent.
        start_seconds: Physical time of the first video latent frame. Non-zero
            when the clip is a window into a longer source, which is what makes
            a continuation task expressible.
        audio_start_seconds: Physical time of the first audio latent frame,
            defaulting to ``start_seconds``. Give it explicitly when the encoder
            introduces a modality-dependent offset; that offset is exactly the
            audio-video desynchronisation users report and nobody can find.
        valid_video_frames: Real video frames, or ``-1`` when all are.
        valid_audio_frames: Real audio frames, or ``-1`` when all are.

    Returns:
        The assembled sample, validated.

    Raises:
        ValueError: If the video tensor is not rank four, a timebase is
            non-positive, or a tensor disagrees with another.
    """
    if video.ndim != 4:
        raise ValueError(
            f"video latents must be (channels, frames, height, width); got "
            f"{tuple(video.shape)}"
        )
    video_num, video_den = video_timebase
    audio_num, audio_den = audio_timebase
    require_positive("video_timebase[0]", video_num)
    require_positive("video_timebase[1]", video_den)
    require_positive("audio_timebase[0]", audio_num)
    require_positive("audio_timebase[1]", audio_den)

    channels, frames, height, width = (int(size) for size in video.shape)
    audio_tensor = (
        audio if audio is not None else torch.zeros((1, 0), dtype=video.dtype)
    )
    if audio_tensor.ndim != 2:
        raise ValueError(
            f"audio latents must be (channels, frames); got "
            f"{tuple(audio_tensor.shape)}"
        )
    audio_channels, audio_frames = (int(size) for size in audio_tensor.shape)

    text_tensor = (
        text if text is not None else torch.zeros((0, text_width), dtype=video.dtype)
    )
    if text_tensor.ndim != 2:
        raise ValueError(
            f"text features must be (tokens, width); got {tuple(text_tensor.shape)}"
        )
    text_tokens, resolved_text_width = (int(size) for size in text_tensor.shape)
    resolved_mask = (
        text_mask
        if text_mask is not None
        else torch.ones(text_tokens, dtype=torch.bool)
    )

    video_positions = (
        torch.arange(frames, dtype=torch.float32) * video_den / video_num
        + start_seconds
    )
    audio_origin = start_seconds if audio_start_seconds is None else audio_start_seconds
    audio_positions = (
        torch.arange(audio_frames, dtype=torch.float32) * audio_den / audio_num
        + audio_origin
    )

    sample = LatentSample(
        descriptor=SampleDescriptor(
            sample_id=sample_id,
            video_frames=frames,
            height=height,
            width=width,
            audio_frames=audio_frames,
            text_tokens=text_tokens,
            text_width=max(resolved_text_width, 1),
            video_channels=channels,
            audio_channels=audio_channels,
            video_timebase_num=video_num,
            video_timebase_den=video_den,
            audio_timebase_num=audio_num,
            audio_timebase_den=audio_den,
            video_codec_id=video_codec_id,
            audio_codec_id=audio_codec_id,
        ),
        video=video,
        audio=audio_tensor,
        text=text_tensor,
        video_positions=video_positions,
        audio_positions=audio_positions,
        text_mask=resolved_mask,
        valid_video_frames=valid_video_frames,
        valid_audio_frames=valid_audio_frames,
    )
    sample.validate()
    return sample


def probe_clip(
    path: Path | str,
    *,
    hash_bytes: int = 1 << 20,
    audio_probe_seconds: float = 10.0,
) -> ClipProbe:
    """Read one media file's metadata and audio level without decoding it fully.

    A full decode of every candidate in a large corpus is days of compute, so
    this reads container metadata and decodes only enough audio to measure a
    level. That is the deliberate trade: it cannot detect a file that decodes
    for two seconds and then fails, and a pipeline that cares about that must do
    a full decode pass separately.

    The content hash is over a bounded prefix of the file bytes, not the decoded
    pixels. It catches the exact byte-level duplicates that dominate scraped
    corpora at a cost independent of clip length. For pixel-level duplicates
    that differ in container metadata, hash the decoded frames with
    :func:`tensor_fingerprint` instead.

    Args:
        path: Media file to probe.
        hash_bytes: Bytes of file prefix to include in the content hash. Larger
            is more discriminating and slower.
        audio_probe_seconds: Seconds of audio to decode when measuring the
            level. The level of a prefix is not the level of the clip; a file
            that opens on silence and then plays will read as quiet.

    Returns:
        The probe. A file that fails to open returns ``decoded=False`` with the
        decoder's message rather than raising, because in a corpus pass an
        unreadable file is data, not an error.

    Raises:
        RuntimeError: If the ``data`` extra is not installed.
        ValueError: If ``hash_bytes`` is non-positive.
    """
    require_positive("hash_bytes", hash_bytes)
    av = _require_module("av", "data")
    source = Path(path)

    try:
        prefix = source.open("rb").read(hash_bytes)
    except OSError as error:
        return ClipProbe(path=str(source), decoded=False, decode_error=str(error))
    content_hash = content_fingerprint(prefix)

    try:
        container = av.open(str(source))
    except Exception as error:  # noqa: BLE001 - any decoder failure is a rejection
        return ClipProbe(
            path=str(source),
            decoded=False,
            content_hash=content_hash,
            decode_error=str(error),
        )

    try:
        video_streams = container.streams.video
        audio_streams = container.streams.audio
        if not video_streams:
            return ClipProbe(
                path=str(source),
                decoded=True,
                content_hash=content_hash,
                decode_error="no video stream",
            )
        stream = video_streams[0]
        rate = stream.average_rate or stream.guessed_rate
        frame_rate_num = int(rate.numerator) if rate is not None else 0
        frame_rate_den = int(rate.denominator) if rate is not None else 1
        duration = float(stream.duration * stream.time_base) if stream.duration else 0.0
        frames = int(stream.frames) if stream.frames else int(duration * (rate or 0))

        has_audio = bool(audio_streams)
        audio_rate = int(audio_streams[0].rate) if has_audio else 0
        audio_channels = int(audio_streams[0].channels) if has_audio else 0
        rms_dbfs = (
            _audio_rms_dbfs(container, audio_streams[0], audio_probe_seconds)
            if has_audio
            else SILENT_DBFS
        )
        return ClipProbe(
            path=str(source),
            decoded=True,
            duration_seconds=duration,
            frame_count=frames,
            height=int(stream.height),
            width=int(stream.width),
            frame_rate_num=frame_rate_num,
            frame_rate_den=frame_rate_den or 1,
            has_audio=has_audio,
            audio_sample_rate=audio_rate,
            audio_channels=audio_channels,
            audio_rms_dbfs=rms_dbfs,
            content_hash=content_hash,
        )
    finally:
        container.close()


def _audio_rms_dbfs(container: Any, stream: Any, seconds: float) -> float:
    """Measure the RMS level of an audio stream prefix in dBFS.

    Args:
        container: An open PyAV container.
        stream: The audio stream to measure.
        seconds: How much audio to decode.

    Returns:
        The level in decibels below full scale, or :data:`SILENT_DBFS` for
        digital silence.
    """
    total = 0.0
    count = 0
    limit = max(seconds, 0.0)
    for frame in container.decode(stream):
        samples = torch.from_numpy(frame.to_ndarray()).float()
        total += float((samples * samples).sum())
        count += samples.numel()
        if frame.time is not None and frame.time >= limit:
            break
    if count == 0 or total <= 0.0:
        return SILENT_DBFS
    return 20.0 * math.log10(math.sqrt(total / count))


def iter_clip_table(
    path: Path | str,
    *,
    columns: Sequence[str] | None = None,
    batch_size: int = 4096,
) -> Iterator[Mapping[str, Any]]:
    """Stream rows from a columnar clip manifest.

    A corpus manifest — paths, captions, whatever provenance a pipeline records
    — is millions of rows, so it is read in record batches rather than
    materialised. Column projection is the other half of that: a manifest with
    an embedded caption embedding is orders of magnitude larger than the columns
    an ingest pass actually needs.

    Args:
        path: Parquet file or dataset directory.
        columns: Columns to read, or ``None`` for all.
        batch_size: Rows per record batch.

    Yields:
        One mapping per row.

    Raises:
        RuntimeError: If the ``data`` extra is not installed.
        ValueError: If ``batch_size`` is non-positive.
    """
    require_positive("batch_size", batch_size)
    dataset_module = _require_module("pyarrow.dataset", "data")
    dataset = dataset_module.dataset(str(path))
    for record_batch in dataset.to_batches(
        columns=list(columns) if columns is not None else None,
        batch_size=batch_size,
    ):
        yield from record_batch.to_pylist()


@dataclass(frozen=True, slots=True)
class IngestSummary:
    """Aggregate outcome of one ingest pass, for the run log.

    The rejection histogram is the point. A corpus that comes out a tenth of the
    expected size is a normal event, and the only way to tell a correctly strict
    gate from a misconfigured one is to see which gate did it.

    Args:
        examined: Clips probed.
        accepted: Clips admitted.
        rejections: Count per rejection reason.
    """

    examined: int = 0
    accepted: int = 0
    rejections: Mapping[RejectionReason, int] = field(default_factory=dict)

    @property
    def acceptance_rate(self) -> float:
        """Fraction of examined clips that were admitted."""
        return self.accepted / self.examined if self.examined else 0.0

    def describe(self) -> str:
        """Return a multi-line summary suitable for a run log."""
        lines = [
            f"examined={self.examined} accepted={self.accepted} "
            f"rate={self.acceptance_rate:.4f}"
        ]
        for reason in sorted(self.rejections, key=lambda value: value.value):
            lines.append(f"  {reason.value}: {self.rejections[reason]}")
        return "\n".join(lines)


def summarise_reports(reports: Sequence[ClipReport]) -> IngestSummary:
    """Aggregate per-clip verdicts into a run summary.

    Args:
        reports: Verdicts from :meth:`QualityGate.evaluate`.

    Returns:
        The aggregate.
    """
    rejections: dict[RejectionReason, int] = {}
    accepted = 0
    for report in reports:
        if report.accepted:
            accepted += 1
        for reason in report.reasons:
            rejections[reason] = rejections.get(reason, 0) + 1
    return IngestSummary(
        examined=len(reports), accepted=accepted, rejections=rejections
    )
