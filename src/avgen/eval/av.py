"""Audio-video metrics: a sync **proxy**, plus the audio degeneracy checks.

The sync metric here is a proxy and the module says so in three places because
people quote it as if it were not. What it computes is the peak of the
cross-correlation between an **audio envelope** (per-frame energy) and a **video
motion energy** signal (per-frame change), over a window of candidate lags. That
catches the thing an AV generator most often gets wrong — audio and video that
are simply not aligned in time — and it is blind to everything a human means by
"in sync":

* It sees energy, not events. A door slam and a hand clap at the same instant
  are the same signal to it.
* It has no notion of *what* is making the sound. Lip sync, in particular, is a
  fine-grained phoneme-to-viseme correspondence that a global energy
  correlation cannot represent at all. Do not report this as a lip-sync number.
* It rewards any content whose loudness happens to track its motion, which
  includes plenty of unrelated pairings.

The honest use is a **regression detector**: when this correlation drops or the
lag shifts between two checkpoints, something in the AV fusion path changed.
For a perceptual claim you need a learned synchronisation model — SyncNet,
AVST, or similar — and those are gated in :mod:`avgen.eval.learned`.

The remaining metrics exist because the failure they detect is embarrassing and
easy to miss: a model that emits **silence** scores well on many published sync
metrics, because a flat signal correlates with nothing and the metric's
denominator is doing the work.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from avgen.eval.protocols import (
    RunningMetric,
    register_metric,
    require_audio,
    require_video,
)

__all__ = ["AudioBandwidth", "AudioSilence", "AVSyncProxy"]

_EPS = 1e-8


def _standardise(signal: torch.Tensor) -> torch.Tensor:
    """Zero-mean, unit-variance a ``(batch, length)`` signal along its length."""
    centred = signal - signal.mean(dim=-1, keepdim=True)
    return centred / centred.std(dim=-1, keepdim=True).clamp_min(_EPS)


def _resample(signal: torch.Tensor, length: int) -> torch.Tensor:
    """Linearly resample ``(batch, n)`` to ``(batch, length)``.

    Audio and video latents are almost never at the same frame rate — a typical
    AV model runs video at 8-24 latent fps and audio at 50-100 — so the two
    envelopes must be brought onto a common grid before they can be correlated
    at all. Linear interpolation is the right tool because both signals are
    already heavily smoothed energy envelopes; anything fancier would be
    modelling noise.
    """
    if signal.shape[-1] == length:
        return signal
    return torch.nn.functional.interpolate(
        signal.unsqueeze(1), size=length, mode="linear", align_corners=False
    ).squeeze(1)


@register_metric("av_sync_proxy")
class AVSyncProxy(RunningMetric):
    """Peak lagged correlation between audio energy and video motion energy.

    **This is a proxy, not a perceptual metric.** See the module docstring for
    what that means in practice; the short version is that it detects gross
    temporal misalignment and cannot detect lip sync.

    **What it measures.** Both modalities are reduced to a one-dimensional
    per-frame energy envelope, resampled onto the video frame grid,
    standardised, and cross-correlated over lags in ``[-max_lag, +max_lag]``
    video frames. Reported: the peak correlation, the lag at which it occurs
    (in frames and in seconds), and the zero-lag correlation. The gap between
    peak and zero-lag is the interesting one — a high peak at a non-zero lag
    means the model learned the association and put it in the wrong place,
    which is a *fixable* bug, while a low peak everywhere means it learned no
    association at all.

    **What it does not measure.** Which sound goes with which object, phoneme
    alignment, audio quality, or whether either stream is plausible on its own.

    Args:
        max_lag: Search window in video frames, each side of zero. Wider is
            slower and more prone to spurious peaks; four frames covers the
            misalignments that a broken AV fusion path produces.
        video_fps: Video latent frame rate, used only to report the lag in
            seconds. Zero suppresses the seconds figure rather than reporting
            a wrong one.
        device: Accumulator device.
    """

    required_inputs = ("video", "audio")

    def __init__(
        self,
        *,
        max_lag: int = 4,
        video_fps: float = 0.0,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(device=device)
        if max_lag < 1:
            raise ValueError(f"max_lag must be >= 1; got {max_lag!r}")
        if video_fps < 0.0:
            raise ValueError(f"video_fps must be >= 0; got {video_fps!r}")
        self._max_lag = max_lag
        self._fps = video_fps

    def observe(self, **inputs: Any) -> Mapping[str, tuple[torch.Tensor, int]]:
        """Accumulate peak correlation, best lag, and zero-lag correlation.

        Args:
            **inputs: Must contain ``video`` and ``audio``.

        Returns:
            ``peak_correlation``, ``zero_lag_correlation``, ``lag_frames``,
            and — only when ``video_fps`` was supplied — ``lag_seconds``.
        """
        video = require_video(inputs["video"], metric=self.name)
        audio = require_audio(inputs["audio"], metric=self.name)
        if video.shape[2] < 3 or audio.shape[2] < 3:
            return {}

        # Motion energy is defined on frame *pairs*, so it is one shorter than
        # the clip; the audio envelope is resampled onto that same grid.
        motion = (video[:, :, 1:] - video[:, :, :-1]).abs().mean(dim=(1, 3, 4))
        envelope = audio.pow(2).mean(dim=1).sqrt()
        length = motion.shape[-1]
        lag_limit = min(self._max_lag, length - 1)
        if lag_limit < 1:
            return {}

        motion = _standardise(motion)
        envelope = _standardise(_resample(envelope, length))

        correlations = []
        for lag in range(-lag_limit, lag_limit + 1):
            if lag < 0:
                left, right = motion[:, -lag:], envelope[:, : length + lag]
            elif lag > 0:
                left, right = motion[:, : length - lag], envelope[:, lag:]
            else:
                left, right = motion, envelope
            # Normalise by the *overlap* length, not the full length, or long
            # lags are penalised purely for having fewer terms.
            correlations.append((left * right).sum(dim=-1) / max(1, left.shape[-1]))
        stacked = torch.stack(correlations, dim=1)

        peak, index = stacked.max(dim=1)
        lag_frames = (index - lag_limit).float()
        zero_lag = stacked[:, lag_limit]
        batch = int(peak.numel())
        results: dict[str, tuple[torch.Tensor, int]] = {
            "peak_correlation": (peak.sum(), batch),
            "zero_lag_correlation": (zero_lag.sum(), batch),
            "lag_frames": (lag_frames.abs().sum(), batch),
        }
        if self._fps > 0.0:
            results["lag_seconds"] = (lag_frames.abs().sum() / self._fps, batch)
        return results


@register_metric("audio_silence")
class AudioSilence(RunningMetric):
    """Silent-frame rate, silent-clip rate, and clipping rate.

    **What it measures.** The two degenerate audio outputs. Silence is the
    trained-to-convergence failure of an AV model whose audio loss was easier
    to minimise by predicting the mean; clipping is the guidance failure, the
    audio analogue of a blown-out image.

    Both are reported as fractions so they read the same at any clip length.
    ``clip_rate`` — the fraction of *clips* that are entirely silent — is the
    one that matters, because a model that emits silence half the time and
    audio the other half has a frame-level silence rate that looks unremarkable.

    **What it does not measure.** Audio quality, intelligibility, or whether
    the sound suits the video. It is a liveness check.

    Args:
        silence_threshold: RMS below which a frame is silent, relative to the
            clip's own peak RMS. Relative rather than absolute because latent
            audio has no fixed scale.
        clip_threshold: Absolute value at or above which a sample counts as
            clipped. ``0.999`` matches a decoder producing ``[-1, 1]`` audio.
        device: Accumulator device.
    """

    required_inputs = ("audio",)

    def __init__(
        self,
        *,
        silence_threshold: float = 0.01,
        clip_threshold: float = 0.999,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(device=device)
        if not 0.0 < silence_threshold < 1.0:
            raise ValueError(
                f"silence_threshold must be in (0, 1); got {silence_threshold!r}"
            )
        if clip_threshold <= 0.0:
            raise ValueError(f"clip_threshold must be > 0; got {clip_threshold!r}")
        self._silence = silence_threshold
        self._clip = clip_threshold

    def observe(self, **inputs: Any) -> Mapping[str, tuple[torch.Tensor, int]]:
        """Accumulate silence and clipping rates.

        Args:
            **inputs: Must contain ``audio``.

        Returns:
            ``frame_rate`` (fraction of silent frames), ``clip_rate`` (fraction
            of entirely silent clips), and ``clipped_fraction``.
        """
        audio = require_audio(inputs["audio"], metric=self.name)
        rms = audio.pow(2).mean(dim=1).sqrt()
        peak = rms.max(dim=-1, keepdim=True).values.clamp_min(_EPS)
        silent = (rms / peak < self._silence).float()
        silent_clips = (rms.max(dim=-1).values < _EPS).float()
        clipped = (audio.abs() >= self._clip).float()
        return {
            "frame_rate": (silent.sum(), int(silent.numel())),
            "clip_rate": (silent_clips.sum(), int(silent_clips.numel())),
            "clipped_fraction": (clipped.sum(), int(clipped.numel())),
        }


@register_metric("audio_bandwidth")
class AudioBandwidth(RunningMetric):
    """Spectral centroid and rolloff of the generated audio.

    **What it measures.** Where the energy sits in the spectrum, as a fraction
    of Nyquist. It detects the muffled output — a model that has learned the
    low-frequency envelope and given up on everything above it, which is what
    an under-trained or over-regularised audio branch produces. The number to
    watch is ``rolloff``: the normalised frequency below which
    ``rolloff_fraction`` of the energy lies. A real-audio reference sits well
    above a generated one when this failure is present.

    **What it does not measure.** Anything perceptual. Two clips with identical
    spectra can sound completely different, and white noise scores an excellent
    bandwidth.

    **Applicability.** On a *waveform* this is a real spectral measurement. On a
    learned audio *latent* the frequency axis is whatever the codec's channel
    ordering happens to be, so the absolute value is meaningless — but the
    metric is still a valid change detector between two checkpoints of the same
    codec, which is how a training loop uses it.

    Args:
        rolloff_fraction: Energy fraction defining the rolloff point.
        sample_rate: Audio sample rate. When given, ``centroid_hz`` and
            ``rolloff_hz`` are reported alongside the normalised figures.
        device: Accumulator device.
    """

    required_inputs = ("audio",)

    def __init__(
        self,
        *,
        rolloff_fraction: float = 0.95,
        sample_rate: float = 0.0,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(device=device)
        if not 0.0 < rolloff_fraction < 1.0:
            raise ValueError(
                f"rolloff_fraction must be in (0, 1); got {rolloff_fraction!r}"
            )
        if sample_rate < 0.0:
            raise ValueError(f"sample_rate must be >= 0; got {sample_rate!r}")
        self._fraction = rolloff_fraction
        self._sample_rate = sample_rate

    def observe(self, **inputs: Any) -> Mapping[str, tuple[torch.Tensor, int]]:
        """Accumulate spectral centroid and rolloff.

        Args:
            **inputs: Must contain ``audio``.

        Returns:
            ``centroid`` and ``rolloff`` as fractions of Nyquist, plus their
            Hz equivalents when a sample rate was supplied.
        """
        audio = require_audio(inputs["audio"], metric=self.name)
        if audio.shape[2] < 4:
            return {}
        mono = audio.mean(dim=1)
        spectrum = torch.fft.rfft(mono, dim=-1).abs().pow(2)
        bins = spectrum.shape[-1]
        # Bin 0 is DC; including it drags the centroid toward zero for any
        # signal with an offset, which is a property of the offset and not of
        # the bandwidth.
        spectrum = spectrum[:, 1:]
        if spectrum.shape[-1] < 2:
            return {}
        frequency = torch.linspace(
            1.0 / (bins - 1), 1.0, spectrum.shape[-1], device=audio.device
        )
        total = spectrum.sum(dim=-1, keepdim=True).clamp_min(_EPS)
        centroid = (spectrum * frequency).sum(dim=-1) / total.squeeze(-1)
        cumulative = spectrum.cumsum(dim=-1) / total
        reached = (cumulative >= self._fraction).float().argmax(dim=-1)
        rolloff = frequency[reached]
        batch = int(centroid.numel())
        results: dict[str, tuple[torch.Tensor, int]] = {
            "centroid": (centroid.sum(), batch),
            "rolloff": (rolloff.sum(), batch),
        }
        if self._sample_rate > 0.0:
            nyquist = self._sample_rate / 2.0
            results["centroid_hz"] = (centroid.sum() * nyquist, batch)
            results["rolloff_hz"] = (rolloff.sum() * nyquist, batch)
        return results
