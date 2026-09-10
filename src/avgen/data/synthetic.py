"""Procedurally generated latents with a known, learnable structure.

Every framework ships a "random tensors" data source and every one of them is
useless for anything except checking that the shapes line up. Gaussian noise has
no temporal structure, so a model that has learned nothing and a model that has
learned everything produce indistinguishable losses, and an entire class of bugs
— a mis-ordered time axis, a broken positional embedding, an audio stream wired
to the wrong sample — passes the smoke test.

:class:`SyntheticSource` is the alternative: a closed-form, deterministic signal
with *real* temporal structure and a *known* audio-video correlation.

* The video is a sum of drifting sinusoids and a translating gradient. Both
  depend on physical time, so predicting frame ``t`` requires actually using
  ``t``; a model that ignores the time axis cannot fit it.
* One video channel is built so its **spatial mean is exactly a known driving
  waveform**, and the audio stream is that same waveform, optionally delayed.
  So audio-video synchrony is not merely present, it is *measurable*:
  :class:`AlignmentTruth` computes the correlation the data was built to have,
  and a test can assert both that the data has it and that a trained model
  reproduced it.

There are no optional dependencies here and no I/O. This is what CI, the
throughput simulator, and the quickstart all run on, so it must be fast,
allocation-light, and reproducible from a seed on any machine.

Determinism is per *sample*, not per iteration order: sample ``k`` is the same
tensor no matter which rank draws it, in which epoch, or after how many
restarts. That is what lets the resumability tests compare batches by value.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from avgen.core._validate import (
    require_dimension,
    require_finite,
    require_name,
    require_positive,
)
from avgen.core.batch import MediaBatch
from avgen.data.protocols import (
    DATA_SCHEMA_VERSION,
    LatentSample,
    SampleDescriptor,
    collate_samples,
)

__all__ = ["AlignmentTruth", "SyntheticConfig", "SyntheticSource"]

_U64 = (1 << 64) - 1
_MAX_SEED = (1 << 63) - 1


def _mix64(value: int) -> int:
    """Return a splitmix64 avalanche of a 64-bit value.

    Sample seeds are derived by mixing rather than by addition. Adding the
    sample index to a base seed makes sample ``k`` of seed ``s`` identical to
    sample ``k-1`` of seed ``s+1``, which turns two "independent" runs into the
    same data shifted by one.

    Args:
        value: Any integer; only the low 64 bits are used.

    Returns:
        A well-mixed 64-bit integer.
    """
    value = (value + 0x9E3779B97F4A7C15) & _U64
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & _U64
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & _U64
    value ^= value >> 31
    return value


@dataclass(frozen=True, slots=True)
class SyntheticConfig:
    """Shape and signal parameters for the procedural source.

    The defaults are deliberately tiny — a few thousand elements per sample — so
    the full test suite runs on a laptop CPU in seconds. Scale them up for a
    throughput benchmark; the signal construction is shape-independent.

    Args:
        seed: Base seed. Two sources with the same seed and config emit
            bit-identical samples.
        num_samples: Size of the synthetic corpus. Finite rather than infinite
            so that epoch boundaries, reshuffling, and resumption are all
            exercisable.
        video_channels: Latent video channels. Must be at least one; channel
            zero is reserved as the audio-video drive channel.
        frames: Latent video frames per sample.
        height: Latent rows per sample.
        width: Latent columns per sample.
        audio_channels: Latent audio channels.
        audio_frames: Latent audio frames per sample. Zero produces a
            video-only corpus, which is the text-to-video training case.
        text_tokens: Text-encoder tokens per sample. Zero produces an
            unconditional corpus.
        text_width: Text-encoder feature width. Stays positive even when
            ``text_tokens`` is zero, because the batch spec carries the width.
        video_timebase_num: Numerator of the video latent frame rate.
        video_timebase_den: Denominator of the video latent frame rate.
        audio_timebase_num: Numerator of the audio latent frame rate.
        audio_timebase_den: Denominator of the audio latent frame rate.
        drive_frequency_hz: Centre frequency of the waveform shared by the drive
            video channel and the audio stream. Keep it well below the Nyquist
            limit of the *video* frame rate, or the video side of the
            correlation aliases and the ground truth stops being recoverable.
        drive_frequency_jitter: Fractional spread of the drive frequency across
            samples, in ``[0, 1)``. This is what makes the ground truth
            *discriminative* rather than merely present: with one shared
            frequency, every sample's waveform is a phase shift of every other
            one, so pairing a clip with the wrong audio still scores a high
            correlation and the alignment assertion proves nothing. Per-sample
            frequencies make a mismatched pair decorrelate.
        drive_harmonic_gain: Amplitude of the second harmonic folded into the
            drive waveform, relative to the fundamental. A pure sinusoid over a
            short window is a nearly one-dimensional family; adding a harmonic
            with an independent phase widens it, which again is about making a
            mismatch detectable.
        audio_lag_seconds: Delay applied to the audio copy of the drive
            waveform. Non-zero is the interesting case: a model can score well
            on "is there audio" while being completely wrong about *when*, and a
            deliberate lag separates those two skills.
        motion_scale: Peak drift speed of the spatial patterns, in latent cells
            per second.
        audio_gain: Amplitude of the audio stream.
        texture_scale: Amplitude of the zero-mean spatial texture added to the
            drive channel. Purely cosmetic for the correlation, which uses the
            spatial mean, but it stops the channel from being spatially
            constant and therefore trivially compressible.
        dtype: Dtype of the emitted latents.
        video_codec_id: Fingerprint recorded on every sample.
        audio_codec_id: Fingerprint recorded on every sample.

    Raises:
        ValueError: If a dimension is non-positive, a rate is non-positive, or a
            signal parameter is not finite.
    """

    seed: int = 0
    num_samples: int = 64
    video_channels: int = 4
    frames: int = 8
    height: int = 8
    width: int = 8
    audio_channels: int = 2
    audio_frames: int = 32
    text_tokens: int = 4
    text_width: int = 16
    video_timebase_num: int = 8
    video_timebase_den: int = 1
    audio_timebase_num: int = 32
    audio_timebase_den: int = 1
    drive_frequency_hz: float = 0.5
    drive_frequency_jitter: float = 0.6
    drive_harmonic_gain: float = 0.6
    audio_lag_seconds: float = 0.0
    motion_scale: float = 1.0
    audio_gain: float = 1.0
    texture_scale: float = 0.25
    dtype: torch.dtype = torch.float32
    video_codec_id: str = "synthetic-video-v1"
    audio_codec_id: str = "synthetic-audio-v1"

    def __post_init__(self) -> None:
        """Validate shapes, rates, and signal parameters."""
        require_dimension("seed", self.seed, allow_zero=True)
        for name in (
            "num_samples",
            "video_channels",
            "frames",
            "height",
            "width",
            "audio_channels",
            "text_width",
            "video_timebase_num",
            "video_timebase_den",
            "audio_timebase_num",
            "audio_timebase_den",
        ):
            require_positive(name, getattr(self, name))
        for name in ("audio_frames", "text_tokens"):
            require_dimension(name, getattr(self, name), allow_zero=True)
        for name in (
            "drive_frequency_hz",
            "drive_frequency_jitter",
            "drive_harmonic_gain",
            "audio_lag_seconds",
            "motion_scale",
            "audio_gain",
            "texture_scale",
        ):
            require_finite(name, getattr(self, name))
        if self.drive_frequency_hz <= 0.0:
            raise ValueError(
                f"drive_frequency_hz must be positive; got {self.drive_frequency_hz}"
            )
        if not 0.0 <= self.drive_frequency_jitter < 1.0:
            raise ValueError(
                "drive_frequency_jitter must be in [0, 1); got "
                f"{self.drive_frequency_jitter}"
            )
        if self.drive_harmonic_gain < 0.0:
            raise ValueError(
                f"drive_harmonic_gain must be non-negative; "
                f"got {self.drive_harmonic_gain}"
            )
        for name in ("video_codec_id", "audio_codec_id"):
            require_name(name, getattr(self, name))
        # Nyquist on the *video* rate, not the audio rate: the correlation is
        # measured on the video time grid, so that is the grid that must resolve
        # the waveform. Two samples per period is the theoretical floor; below
        # four the estimate is too noisy to assert on.
        highest = self.max_drive_frequency_hz
        if highest * 4.0 > self.video_fps:
            raise ValueError(
                f"the drive waveform reaches {highest} Hz (frequency "
                f"{self.drive_frequency_hz} widened by jitter "
                f"{self.drive_frequency_jitter} and doubled by the harmonic) and "
                "needs at least four video samples per period, but the video rate "
                f"is {self.video_fps} fps; lower the frequency, drop the harmonic, "
                "or raise the frame rate, otherwise the ground-truth correlation "
                "is not recoverable from the video"
            )

    @property
    def max_drive_frequency_hz(self) -> float:
        """Highest frequency present in any sample's drive waveform."""
        top = self.drive_frequency_hz * (1.0 + self.drive_frequency_jitter)
        return top * 2.0 if self.drive_harmonic_gain > 0.0 else top

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
        """Whether the corpus carries an audio stream."""
        return self.audio_frames > 0


@dataclass(frozen=True, slots=True)
class AlignmentTruth:
    """The audio-video correlation the synthetic data was built to contain.

    This exists so a test can make a claim stronger than "the loss went down".
    Because the source knows the exact waveform it shared between the two
    modalities, a test can assert that the *data* carries the correlation (a
    check on the generator) and, separately, that a model's *output* carries it
    (a check on the model). The latter is the only cheap end-to-end assertion
    that audio-video sync is being learned rather than merely represented.

    The measurement is a Pearson correlation between the spatial mean of the
    drive video channel and the audio drive channel resampled onto the video
    time grid. Correlation rather than mean-squared error because it is
    invariant to the arbitrary scale and offset a model's output head applies.

    Args:
        drive_frequency_hz: Frequency of the shared waveform.
        audio_lag_seconds: Delay applied to the audio copy.
        video_drive_channel: Video channel whose spatial mean is the waveform.
        audio_drive_channel: Audio channel that carries the waveform.
        expected_correlation: Correlation the clean data achieves. One, up to
            interpolation and floating-point error.
        tolerance: How far below :attr:`expected_correlation` a measurement may
            fall and still be considered a match for the clean data.
    """

    drive_frequency_hz: float
    audio_lag_seconds: float
    video_drive_channel: int
    audio_drive_channel: int
    expected_correlation: float = 1.0
    tolerance: float = 1e-2

    def video_drive(self, video: torch.Tensor) -> torch.Tensor:
        """Return the video-side waveform estimate.

        Args:
            video: ``(batch, channels, frames, height, width)`` latents.

        Returns:
            ``(batch, frames)`` spatial mean of the drive channel.
        """
        return video[:, self.video_drive_channel].mean(dim=(-2, -1)).float()

    def audio_drive_at(
        self,
        audio: torch.Tensor,
        video_positions: torch.Tensor,
        *,
        audio_fps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resample the audio drive channel onto the video time grid.

        Linear interpolation, not nearest-neighbour: at realistic latent frame
        rates the audio grid is only a few times denser than the video grid, and
        nearest-neighbour quantisation alone can cost several percent of
        correlation, which would swamp the effect being measured.

        A validity mask is returned alongside the values, and it matters
        whenever the lag is non-zero. Video frame ``t`` needs the audio at
        ``t + lag``, and for the last ``lag`` seconds of a clip that sample does
        not exist. Clamping to the final audio frame — the obvious thing — turns
        the tail of the comparison into a constant, which drags the correlation
        down by several percent for reasons that have nothing to do with
        alignment. Those positions are excluded instead.

        Args:
            audio: ``(batch, channels, frames)`` audio latents.
            video_positions: ``(batch, frames)`` video times in seconds.
            audio_fps: Audio latent frame rate.

        Returns:
            ``(batch, video_frames)`` audio drive sampled at the video times,
            and a bool mask of the positions the audio actually covers.

        Raises:
            ValueError: If the audio stream has no frames, which means there is
                no alignment to measure.
        """
        if audio.shape[-1] == 0:
            raise ValueError(
                "cannot measure alignment on a batch with no audio frames"
            )
        track = audio[:, self.audio_drive_channel].float()
        last = track.shape[-1] - 1
        # Shift forward by the lag so that video time t is compared against the
        # audio sample that carries the waveform value from time t.
        position = (video_positions.float() + self.audio_lag_seconds) * audio_fps
        covered = (position >= 0.0) & (position <= float(last))
        position = position.clamp(0.0, float(last))
        lower = position.floor().to(torch.int64)
        upper = (lower + 1).clamp(max=last)
        weight = position - lower.to(position.dtype)
        left = torch.gather(track, 1, lower)
        right = torch.gather(track, 1, upper)
        return left + (right - left) * weight, covered

    def correlation(self, batch: MediaBatch) -> torch.Tensor:
        """Return the per-sample Pearson correlation carried by a batch.

        Args:
            batch: A batch produced by any source, including model output that
                has been packed back into batch form.

        Returns:
            ``(batch,)`` float32 correlations in ``[-1, 1]``. A sample with
            fewer than three usable positions, or with a constant signal on
            either side, reports zero rather than a NaN that would poison any
            aggregate computed over it.

        Raises:
            ValueError: If the batch carries no audio.
        """
        video_signal = self.video_drive(batch.video)
        audio_signal, covered = self.audio_drive_at(
            batch.audio, batch.video_positions, audio_fps=batch.spec.audio_fps
        )
        weights = covered.to(torch.float32)
        count = weights.sum(dim=1, keepdim=True)
        safe = count.clamp(min=1.0)
        video_centred = (video_signal - (video_signal * weights).sum(1, True) / safe)
        audio_centred = (audio_signal - (audio_signal * weights).sum(1, True) / safe)
        video_centred = video_centred * weights
        audio_centred = audio_centred * weights
        numerator = (video_centred * audio_centred).sum(dim=1)
        denominator = video_centred.norm(dim=1) * audio_centred.norm(dim=1)
        # Three points is the floor at which a correlation means anything; below
        # it any two signals agree almost perfectly by construction.
        usable = (denominator > 0.0) & (count.squeeze(1) >= 3.0)
        return torch.where(
            usable, numerator / denominator.clamp(min=1e-12),
            torch.zeros_like(numerator),
        )

    def holds(self, batch: MediaBatch) -> bool:
        """Whether every sample in a batch carries the expected correlation.

        Args:
            batch: The batch to check.

        Returns:
            True if the minimum per-sample correlation is within
            :attr:`tolerance` of :attr:`expected_correlation`.
        """
        worst = float(self.correlation(batch).min())
        return worst >= self.expected_correlation - self.tolerance


class SyntheticSource:
    """A deterministic procedural corpus that is both a store and a stream.

    It implements :class:`~avgen.data.protocols.SampleStore` so
    :func:`~avgen.data.loader.build_loader` can shuffle, bucket, and shard it
    like any on-disk corpus, and it implements
    :class:`~avgen.data.protocols.DataSource` so a quickstart can iterate it
    directly with no loader at all. The rejected alternative was two classes;
    they shared every line of the signal construction and drifted apart within a
    week.

    Args:
        config: Shape and signal parameters.
        batch_size: Samples per batch when iterated directly as a data source.
        data_rank: This rank's index within the data-parallel dimension, used
            only for direct iteration. Ranks sharing a ``data_rank`` iterate the
            identical sample sequence.
        data_world: Size of the data-parallel dimension.
        bucket_id: Bucket identifier stamped into the emitted batch specs.

    Raises:
        ValueError: If ``batch_size`` is non-positive, the rank coordinates are
            inconsistent, or the corpus is too small to fill one batch per rank.
    """

    __slots__ = (
        "_batch_size",
        "_bucket_id",
        "_config",
        "_cursor",
        "_data_rank",
        "_data_world",
        "_epoch",
        "_local_indices",
    )

    def __init__(
        self,
        config: SyntheticConfig | None = None,
        *,
        batch_size: int = 1,
        data_rank: int = 0,
        data_world: int = 1,
        bucket_id: int = 0,
    ) -> None:
        self._config = config or SyntheticConfig()
        require_positive("batch_size", batch_size)
        require_positive("data_world", data_world)
        require_dimension("data_rank", data_rank, allow_zero=True)
        require_dimension("bucket_id", bucket_id, allow_zero=True)
        if data_rank >= data_world:
            raise ValueError(
                f"data_rank={data_rank} must be below data_world={data_world}"
            )
        self._batch_size = batch_size
        self._data_rank = data_rank
        self._data_world = data_world
        self._bucket_id = bucket_id
        # A strided slice, not a contiguous block: with a contiguous split, rank
        # zero would only ever see the low sample ids, so any property that
        # correlates with id (write order, and therefore recording session,
        # source, or capture date in a real corpus) becomes a per-rank bias.
        self._local_indices = tuple(
            range(data_rank, self._config.num_samples, data_world)
        )
        if not self._local_indices:
            raise ValueError(
                f"num_samples={self._config.num_samples} is smaller than "
                f"data_world={data_world}; every rank must own at least one sample"
            )
        self._cursor = 0
        self._epoch = 0

    @property
    def config(self) -> SyntheticConfig:
        """The configuration this source was built from."""
        return self._config

    @property
    def alignment_truth(self) -> AlignmentTruth:
        """The audio-video correlation this source guarantees.

        Raises:
            ValueError: If the corpus has no audio, in which case there is no
                alignment to describe.
        """
        if not self._config.has_audio:
            raise ValueError(
                "alignment_truth is undefined for a video-only synthetic corpus; "
                "set audio_frames > 0"
            )
        return AlignmentTruth(
            drive_frequency_hz=self._config.drive_frequency_hz,
            audio_lag_seconds=self._config.audio_lag_seconds,
            video_drive_channel=0,
            audio_drive_channel=0,
        )

    # ------------------------------------------------------------------
    # SampleStore
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the size of the synthetic corpus."""
        return self._config.num_samples

    def descriptor(self, index: int) -> SampleDescriptor:
        """Return a sample's static description without generating tensors.

        Args:
            index: Position within the corpus.

        Returns:
            The descriptor. Every synthetic sample shares one geometry, so this
            is a cheap constant apart from the id.

        Raises:
            IndexError: If ``index`` is out of range.
        """
        config = self._config
        return SampleDescriptor(
            sample_id=self._checked_index(index),
            video_frames=config.frames,
            height=config.height,
            width=config.width,
            audio_frames=config.audio_frames,
            text_tokens=config.text_tokens,
            text_width=config.text_width,
            video_channels=config.video_channels,
            audio_channels=config.audio_channels,
            video_timebase_num=config.video_timebase_num,
            video_timebase_den=config.video_timebase_den,
            audio_timebase_num=config.audio_timebase_num,
            audio_timebase_den=config.audio_timebase_den,
            video_codec_id=config.video_codec_id,
            audio_codec_id=config.audio_codec_id,
        )

    def __getitem__(self, index: int) -> LatentSample:
        """Generate one sample.

        Args:
            index: Position within the corpus, which is also the sample id.

        Returns:
            The generated sample.

        Raises:
            IndexError: If ``index`` is out of range.
        """
        sample_id = self._checked_index(index)
        config = self._config
        parameters = self._parameters_for(sample_id)

        video_time = (
            torch.arange(config.frames, dtype=torch.float32)
            * config.video_timebase_den
            / config.video_timebase_num
        )
        rows = torch.arange(config.height, dtype=torch.float32)
        cols = torch.arange(config.width, dtype=torch.float32)

        video = self._build_video(parameters, video_time, rows, cols)

        if config.has_audio:
            audio_time = (
                torch.arange(config.audio_frames, dtype=torch.float32)
                * config.audio_timebase_den
                / config.audio_timebase_num
            )
            audio = self._build_audio(parameters, audio_time)
        else:
            audio_time = torch.zeros(0, dtype=torch.float32)
            audio = torch.zeros(
                (config.audio_channels, 0), dtype=torch.float32
            )

        text = parameters["text"]
        text_mask = torch.ones(config.text_tokens, dtype=torch.bool)

        return LatentSample(
            descriptor=self.descriptor(sample_id),
            video=video.to(config.dtype),
            audio=audio.to(config.dtype),
            text=text.to(config.dtype),
            video_positions=video_time,
            audio_positions=audio_time,
            text_mask=text_mask,
        )

    # ------------------------------------------------------------------
    # DataSource
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[MediaBatch]:
        """Yield batches from this rank's slice, wrapping at epoch boundaries.

        Iteration order is the natural index order. Shuffling is deliberately
        *not* done here: it belongs to the loader, which owns the epoch seed and
        the resumable cursor, and duplicating it would give two sources of truth
        for sample order.

        Yields:
            Dense batches on CPU.
        """
        while True:
            if self._cursor + self._batch_size > len(self._local_indices):
                self._cursor = 0
                self._epoch += 1
            window = self._local_indices[
                self._cursor : self._cursor + self._batch_size
            ]
            self._cursor += self._batch_size
            yield collate_samples(
                [self[index] for index in window],
                bucket_id=self._bucket_id,
                schema_version=DATA_SCHEMA_VERSION,
            )

    def state_dict(self) -> Mapping[str, Any]:
        """Return the cursor needed to resume this stream exactly."""
        return {
            "schema_version": DATA_SCHEMA_VERSION,
            "cursor": self._cursor,
            "epoch": self._epoch,
            "data_rank": self._data_rank,
            "data_world": self._data_world,
            "batch_size": self._batch_size,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a cursor produced by :meth:`state_dict`.

        Args:
            state: Mapping produced by :meth:`state_dict`.

        Raises:
            ValueError: If the state was written by a different rank layout or
                batch size, which would make the replayed sample order differ
                from the recorded one.
        """
        for name, current in (
            ("data_rank", self._data_rank),
            ("data_world", self._data_world),
            ("batch_size", self._batch_size),
        ):
            recorded = int(state[name])
            if recorded != current:
                raise ValueError(
                    f"cannot resume a synthetic source with {name}={current} from a "
                    f"checkpoint written with {name}={recorded}; the sample order "
                    "would silently differ from the one being resumed"
                )
        self._cursor = int(state["cursor"])
        self._epoch = int(state["epoch"])

    # ------------------------------------------------------------------
    # Signal construction
    # ------------------------------------------------------------------

    def _checked_index(self, index: int) -> int:
        """Bounds-check an index against the corpus size.

        Args:
            index: Candidate index.

        Returns:
            The index unchanged.

        Raises:
            IndexError: If it falls outside the corpus.
        """
        if not 0 <= index < self._config.num_samples:
            raise IndexError(
                f"sample index {index} out of range for a corpus of "
                f"{self._config.num_samples} samples"
            )
        return index

    def _parameters_for(self, sample_id: int) -> dict[str, torch.Tensor]:
        """Draw this sample's per-sample signal parameters.

        Every draw comes from a generator seeded by mixing the base seed with
        the sample id, so the parameters depend on the sample and on nothing
        else — not on iteration order, not on how many samples were drawn
        before, not on how many ranks are running.

        Args:
            sample_id: Stable sample identifier.

        Returns:
            A mapping of parameter tensors, all on CPU in float32.
        """
        config = self._config
        seed = _mix64((config.seed << 32) ^ _mix64(sample_id)) % _MAX_SEED
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        def _uniform(shape: tuple[int, ...], low: float, high: float) -> torch.Tensor:
            return (
                torch.rand(shape, generator=generator, dtype=torch.float32)
                * (high - low)
                + low
            )

        channels = config.video_channels
        return {
            # Spatial frequencies in cycles per latent cell. Bounded well below
            # 0.5 (the spatial Nyquist limit) so the pattern is resolved rather
            # than aliased into a different, lower-frequency pattern.
            "freq_row": _uniform((channels,), 0.04, 0.30),
            "freq_col": _uniform((channels,), 0.04, 0.30),
            "phase": _uniform((channels,), 0.0, 2.0 * math.pi),
            # Drift velocities in latent cells per second. Signed, so patterns
            # move in both directions and a model cannot learn one direction.
            "velocity_row": _uniform((channels,), -1.0, 1.0) * config.motion_scale,
            "velocity_col": _uniform((channels,), -1.0, 1.0) * config.motion_scale,
            "gradient_velocity": _uniform((channels,), -0.5, 0.5) * config.motion_scale,
            "drive_phase": _uniform((), 0.0, 2.0 * math.pi),
            "drive_harmonic_phase": _uniform((), 0.0, 2.0 * math.pi),
            "drive_frequency": (
                config.drive_frequency_hz
                * (1.0 + config.drive_frequency_jitter * _uniform((), -1.0, 1.0))
            ),
            "audio_phase": _uniform((max(config.audio_channels, 1),), 0.0, 2 * math.pi),
            "audio_freq": _uniform((max(config.audio_channels, 1),), 0.2, 1.5),
            "text": _uniform((config.text_tokens, config.text_width), -1.0, 1.0),
        }

    def _drive(
        self, parameters: Mapping[str, torch.Tensor], time: torch.Tensor
    ) -> torch.Tensor:
        """Return the shared audio-video waveform at the given times.

        A fundamental at a per-sample frequency plus an independently phased
        second harmonic. Both of those exist to make the waveform *identify* its
        sample: a single shared frequency would make every sample a phase shift
        of every other, so a clip paired with the wrong audio would still
        correlate highly and the alignment assertion would be vacuous.

        Args:
            parameters: Per-sample parameters.
            time: Times in seconds.

        Returns:
            The waveform, same shape as ``time``, with unit peak amplitude.
        """
        config = self._config
        frequency = float(parameters["drive_frequency"])
        fundamental = torch.sin(
            2.0 * math.pi * frequency * time + float(parameters["drive_phase"])
        )
        gain = config.drive_harmonic_gain
        if gain <= 0.0:
            return fundamental
        harmonic = torch.sin(
            2.0 * math.pi * 2.0 * frequency * time
            + float(parameters["drive_harmonic_phase"])
        )
        # Normalise by the sum of the amplitudes so the waveform stays inside
        # [-1, 1] regardless of the harmonic gain; an unnormalised sum would make
        # the drive channel's dynamic range depend on a config knob and therefore
        # make two configurations produce differently scaled latents.
        return (fundamental + gain * harmonic) / (1.0 + gain)

    def _build_video(
        self,
        parameters: Mapping[str, torch.Tensor],
        time: torch.Tensor,
        rows: torch.Tensor,
        cols: torch.Tensor,
    ) -> torch.Tensor:
        """Construct the video latents in closed form.

        Channel zero is special: it is the drive waveform plus a texture that is
        forced to have exactly zero spatial mean, so the channel's spatial mean
        *is* the waveform. Everything else is a drifting sinusoid plus a
        translating triangular gradient, which together give both a
        high-frequency and a low-frequency structure that move at different
        speeds — enough that a model must represent motion rather than a single
        static average frame.

        Args:
            parameters: Per-sample parameters.
            time: ``(frames,)`` times in seconds.
            rows: ``(height,)`` row indices.
            cols: ``(width,)`` column indices.

        Returns:
            ``(channels, frames, height, width)`` float32 latents.
        """
        config = self._config
        # Broadcast axes: (channel, frame, row, col).
        frame_axis = time.view(1, -1, 1, 1)
        row_axis = rows.view(1, 1, -1, 1)
        col_axis = cols.view(1, 1, 1, -1)

        def _channel_axis(name: str) -> torch.Tensor:
            return parameters[name].view(-1, 1, 1, 1)

        # Advected coordinates: the pattern is stationary in a frame that
        # translates, which is exactly "an object moving across the screen".
        row_phase = row_axis - _channel_axis("velocity_row") * frame_axis
        col_phase = col_axis - _channel_axis("velocity_col") * frame_axis
        wave = torch.sin(
            2.0
            * math.pi
            * (
                _channel_axis("freq_row") * row_phase
                + _channel_axis("freq_col") * col_phase
            )
            + _channel_axis("phase")
        )

        # A translating triangular ramp. Triangular rather than sawtooth so the
        # signal is continuous: a wrap discontinuity would put an unlearnable
        # step edge in the data and dominate any reconstruction loss.
        ramp = (row_axis / config.height + col_axis / config.width) * 0.5
        ramp = ramp + _channel_axis("gradient_velocity") * frame_axis
        gradient = 2.0 * torch.abs(2.0 * (ramp - torch.floor(ramp)) - 1.0) - 1.0

        video = 0.7 * wave + 0.3 * gradient

        # Force channel zero to carry the drive waveform in its spatial mean.
        texture = config.texture_scale * wave[0:1]
        texture = texture - texture.mean(dim=(-2, -1), keepdim=True)
        video[0:1] = self._drive(parameters, time).view(1, -1, 1, 1) + texture
        return video

    def _build_audio(
        self, parameters: Mapping[str, torch.Tensor], time: torch.Tensor
    ) -> torch.Tensor:
        """Construct the audio latents in closed form.

        Channel zero is the drive waveform, delayed by
        :attr:`SyntheticConfig.audio_lag_seconds`. Remaining channels are half
        drive and half an uncorrelated tone, so a model cannot solve the
        alignment task by copying every channel indiscriminately.

        Args:
            parameters: Per-sample parameters.
            time: ``(frames,)`` times in seconds.

        Returns:
            ``(channels, frames)`` float32 latents.
        """
        config = self._config
        # Evaluating the drive at (t - lag) means the value the video showed at
        # time t appears in the audio at time t + lag: a genuine delay, not a
        # relabelling of the time axis.
        delayed = self._drive(parameters, time - config.audio_lag_seconds)
        channels = [config.audio_gain * delayed]
        for channel in range(1, config.audio_channels):
            tone = torch.sin(
                2.0 * math.pi * float(parameters["audio_freq"][channel]) * time
                + float(parameters["audio_phase"][channel])
            )
            channels.append(config.audio_gain * (0.5 * delayed + 0.5 * tone))
        return torch.stack(channels, dim=0)
