"""Mel filterbank and STFT frontend, in pure torch.

Audio conditioning and audio generation both need a mel spectrogram, and the
obvious way to get one is ``torchaudio``. avgen does not take that dependency in
this path, for two reasons that are practical rather than ideological:

* ``torchaudio`` is version-locked to ``torch``. A cluster image that pins
  ``torch`` for a driver reason then cannot upgrade, and a mel spectrogram is not
  worth that constraint on the whole stack.
* Mel is a filterbank matmul over an STFT. It is sixty lines. Owning them means
  the exact convention — which mel scale, which normalisation, which padding — is
  visible in this repository rather than inferred from another project's
  defaults, and *that* matters because a vocoder trained against one convention
  produces noise when fed a spectrogram built with another.

**Slaney versus HTK.** Two mel scales are in circulation and they disagree by
enough to matter. HTK is a single logarithmic curve. Slaney (the one used by
librosa's default, by every HiFi-GAN/BigVGAN checkpoint in common use, and
therefore by essentially every audio generation model you might want to pair with
this framework) is *linear* below 1kHz and logarithmic above it, on the grounds
that pitch perception really is close to linear at low frequencies. avgen
implements Slaney, and the area normalisation that goes with it, because the
whole point of a shared frontend is to match the vocoder you did not train.

Everything here is a pure function of its config, so a fingerprint over the
config fully determines the output.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

import torch

__all__ = [
    "MelConfig",
    "MelFrontend",
    "hz_to_mel",
    "mel_filterbank",
    "mel_to_hz",
    "stft_magnitude",
]

#: Slaney break point: below this the scale is linear, above it logarithmic.
_BREAK_HZ = 1000.0
#: Hertz per mel in the linear region. 200/3 is the constant Slaney chose so that
#: 1000 Hz lands exactly on mel 15, which is what makes the two regions join
#: without a discontinuity.
_LINEAR_SLOPE_HZ_PER_MEL = 200.0 / 3.0
_BREAK_MEL = _BREAK_HZ / _LINEAR_SLOPE_HZ_PER_MEL
#: Logarithmic step above the break point: 6.4x in frequency over 27 mels.
_LOG_STEP = math.log(6.4) / 27.0


@dataclass(frozen=True, slots=True)
class MelConfig:
    """Everything that determines a mel spectrogram.

    Args:
        sample_rate: Waveform sample rate in hertz.
        n_fft: FFT size. Also the analysis window length unless ``win_length``
            overrides it.
        hop_length: Samples between consecutive frames.
        win_length: Analysis window length. Defaults to ``n_fft``.
        n_mels: Number of mel bands.
        f_min: Lowest band edge in hertz.
        f_max: Highest band edge in hertz. Defaults to the Nyquist frequency.
        power: Exponent applied to the STFT magnitude. ``1.0`` gives amplitude,
            ``2.0`` gives power. Vocoders overwhelmingly expect ``1.0``.
        center: Whether to pad the signal so frame ``k`` is centred on sample
            ``k * hop_length``. True matches librosa and every vocoder trained
            against it; False loses half a window at each end.
        normalized: Whether the STFT is normalised by the window energy.
        log_epsilon: Floor added before the log so silence maps to a finite
            value instead of ``-inf``.

    Raises:
        ValueError: If a size is non-positive, the window exceeds ``n_fft``, or
            the frequency range is empty or exceeds Nyquist.
    """

    sample_rate: int = 24000
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int | None = None
    n_mels: int = 80
    f_min: float = 0.0
    f_max: float | None = None
    power: float = 1.0
    center: bool = True
    normalized: bool = False
    log_epsilon: float = 1e-5

    def __post_init__(self) -> None:
        """Validate sizes and the frequency range."""
        for name in ("sample_rate", "n_fft", "hop_length", "n_mels"):
            value = getattr(self, name)
            if isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer; got {value!r}")
        if self.win_length is not None and not 1 <= self.win_length <= self.n_fft:
            raise ValueError(
                f"win_length must be in [1, n_fft={self.n_fft}]; "
                f"got {self.win_length!r}"
            )
        if not math.isfinite(self.power) or self.power <= 0.0:
            raise ValueError(f"power must be finite and positive; got {self.power!r}")
        if not math.isfinite(self.log_epsilon) or self.log_epsilon <= 0.0:
            raise ValueError(
                f"log_epsilon must be finite and positive; got {self.log_epsilon!r}"
            )
        if self.f_min < 0.0 or not math.isfinite(self.f_min):
            raise ValueError(
                f"f_min must be finite and non-negative; got {self.f_min!r}"
            )
        upper = self.nyquist if self.f_max is None else self.f_max
        if not math.isfinite(upper) or upper <= self.f_min:
            raise ValueError(
                f"f_max must be finite and greater than f_min={self.f_min}; "
                f"got {self.f_max!r}"
            )
        if upper > self.nyquist:
            raise ValueError(
                f"f_max={upper} exceeds the Nyquist frequency {self.nyquist}; "
                "the bands above it would be empty"
            )

    @property
    def nyquist(self) -> float:
        """Highest representable frequency."""
        return self.sample_rate / 2.0

    @property
    def window_length(self) -> int:
        """Effective analysis window length."""
        return self.n_fft if self.win_length is None else self.win_length

    @property
    def upper_hz(self) -> float:
        """Effective highest band edge."""
        return self.nyquist if self.f_max is None else self.f_max

    @property
    def num_bins(self) -> int:
        """Number of one-sided STFT bins."""
        return self.n_fft // 2 + 1

    def frames_for(self, samples: int) -> int:
        """Return the frame count produced for a sample count.

        Args:
            samples: Waveform length in samples.

        Returns:
            Number of STFT frames.

        Raises:
            ValueError: If ``samples`` is negative, or is shorter than one
                window when ``center`` is false.
        """
        if isinstance(samples, bool) or samples < 0:
            raise ValueError(f"samples must be non-negative; got {samples!r}")
        if self.center:
            return samples // self.hop_length + 1
        if samples < self.n_fft:
            raise ValueError(
                f"samples={samples} is shorter than n_fft={self.n_fft} and "
                "center=False leaves no complete frame"
            )
        return (samples - self.n_fft) // self.hop_length + 1

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""
        return asdict(self)


def hz_to_mel(frequency: torch.Tensor) -> torch.Tensor:
    """Convert hertz to Slaney mels.

    Args:
        frequency: Frequencies in hertz.

    Returns:
        Mel values, same shape and dtype.
    """
    linear = frequency / _LINEAR_SLOPE_HZ_PER_MEL
    # torch.where evaluates both branches, so the log argument is clamped away
    # from zero even where it is discarded; without the clamp a 0 Hz entry
    # produces a NaN gradient and an inf that survives the select.
    logarithmic = _BREAK_MEL + torch.log(
        torch.clamp(frequency / _BREAK_HZ, min=1e-10)
    ) / _LOG_STEP
    return torch.where(frequency >= _BREAK_HZ, logarithmic, linear)


def mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    """Convert Slaney mels to hertz.

    Args:
        mel: Mel values.

    Returns:
        Frequencies in hertz, same shape and dtype.
    """
    linear = mel * _LINEAR_SLOPE_HZ_PER_MEL
    logarithmic = _BREAK_HZ * torch.exp(_LOG_STEP * (mel - _BREAK_MEL))
    return torch.where(mel >= _BREAK_MEL, logarithmic, linear)


@lru_cache(maxsize=16)
def _filterbank_cpu(
    num_bins: int,
    n_mels: int,
    sample_rate: int,
    f_min: float,
    f_max: float,
) -> torch.Tensor:
    """Build the Slaney-normalised triangular filterbank on CPU."""
    bin_hz = torch.linspace(0.0, sample_rate / 2.0, num_bins, dtype=torch.float64)
    edges_mel = torch.linspace(
        float(hz_to_mel(torch.tensor(f_min, dtype=torch.float64))),
        float(hz_to_mel(torch.tensor(f_max, dtype=torch.float64))),
        n_mels + 2,
        dtype=torch.float64,
    )
    edges_hz = mel_to_hz(edges_mel)

    # Band m is the triangle rising from edge m to edge m+1 and falling to
    # edge m+2, so consecutive bands overlap by half and every frequency in
    # range is covered by exactly two bands.
    differences = edges_hz[1:] - edges_hz[:-1]
    offsets = edges_hz[None, :] - bin_hz[:, None]
    lower = -offsets[:, :-2] / differences[None, :-1]
    upper = offsets[:, 2:] / differences[None, 1:]
    weights = torch.clamp(torch.minimum(lower, upper), min=0.0)

    # Slaney normalisation: each band is scaled to unit *area* rather than unit
    # peak, so a wide high-frequency band does not dominate a narrow low one
    # purely by covering more bins. Skipping this is the usual reason a
    # home-rolled mel does not match librosa.
    enorm = 2.0 / (edges_hz[2 : n_mels + 2] - edges_hz[:n_mels])
    weights = weights * enorm[None, :]
    return weights.transpose(0, 1).to(torch.float32).contiguous()


def mel_filterbank(
    config: MelConfig,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the mel projection matrix for a config.

    Args:
        config: Frontend configuration.
        device: Device to place the matrix on.
        dtype: Dtype of the returned matrix.

    Returns:
        ``(n_mels, num_bins)`` filterbank. Built once in float64 and cached, so
        repeated calls are a device copy rather than a rebuild.
    """
    bank = _filterbank_cpu(
        config.num_bins,
        config.n_mels,
        config.sample_rate,
        config.f_min,
        config.upper_hz,
    )
    return bank.to(device=device, dtype=dtype)


@lru_cache(maxsize=16)
def _window_cpu(window_length: int, n_fft: int) -> torch.Tensor:
    """Return a periodic Hann window zero-padded to the FFT size."""
    window = torch.hann_window(window_length, periodic=True, dtype=torch.float32)
    if window_length == n_fft:
        return window
    # Centre the shorter window inside the FFT frame, matching librosa; padding
    # only on the right would shift every frame by half the difference and
    # smear the phase, which a vocoder hears.
    left = (n_fft - window_length) // 2
    return torch.nn.functional.pad(window, (left, n_fft - window_length - left))


def stft_magnitude(waveform: torch.Tensor, config: MelConfig) -> torch.Tensor:
    """Return the one-sided STFT magnitude of a waveform.

    Args:
        waveform: ``(batch, samples)``, ``(batch, 1, samples)``, or
            ``(samples,)``. A multi-channel waveform must be downmixed by the
            caller; silently averaging channels here would hide a stereo file
            reaching a mono pipeline.
        config: Frontend configuration.

    Returns:
        ``(batch, num_bins, frames)`` magnitude raised to ``config.power``.

    Raises:
        ValueError: If the waveform rank is unsupported or it carries more than
            one channel.
        TypeError: If the waveform is not floating point.
    """
    if not waveform.is_floating_point():
        raise TypeError(f"waveform must be floating point; got {waveform.dtype}")
    if waveform.ndim == 1:
        signal = waveform[None, :]
    elif waveform.ndim == 2:
        signal = waveform
    elif waveform.ndim == 3:
        if waveform.shape[1] != 1:
            raise ValueError(
                f"stft_magnitude expects one channel; got {waveform.shape[1]}. "
                "Downmix explicitly at the data boundary."
            )
        signal = waveform[:, 0]
    else:
        raise ValueError(
            f"waveform must have rank 1, 2, or 3; got {tuple(waveform.shape)}"
        )
    window = _window_cpu(config.window_length, config.n_fft).to(
        device=signal.device, dtype=signal.dtype
    )
    spectrum = torch.stft(
        signal,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.n_fft,
        window=window,
        center=config.center,
        # Reflect padding rather than zeros: a zero pad injects a step
        # discontinuity at the clip boundary that shows up as a broadband click
        # in the first and last frames.
        pad_mode="reflect",
        normalized=config.normalized,
        onesided=True,
        return_complex=True,
    )
    magnitude = torch.abs(spectrum)
    if config.power != 1.0:
        magnitude = magnitude.pow(config.power)
    return magnitude


class MelFrontend:
    """A configured mel spectrogram transform.

    Holds the config and caches the filterbank per device, so the steady-state
    call is one STFT plus one matmul with no allocation of the filterbank.

    Args:
        config: Frontend configuration.
    """

    __slots__ = ("_banks", "config")

    def __init__(self, config: MelConfig | None = None) -> None:
        self.config = config or MelConfig()
        self._banks: dict[tuple[str, torch.dtype], torch.Tensor] = {}

    @property
    def mel_bins(self) -> int:
        """Number of mel bands produced."""
        return self.config.n_mels

    @property
    def sample_rate(self) -> int:
        """Expected input sample rate."""
        return self.config.sample_rate

    def filterbank(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return the cached filterbank for a device and dtype.

        Args:
            device: Target device.
            dtype: Target dtype.

        Returns:
            ``(n_mels, num_bins)`` filterbank.
        """
        key = (str(device), dtype)
        bank = self._banks.get(key)
        if bank is None:
            bank = mel_filterbank(self.config, device=device, dtype=dtype)
            self._banks[key] = bank
        return bank

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute a linear-amplitude mel spectrogram.

        Args:
            waveform: ``(batch, samples)`` or ``(batch, 1, samples)``.

        Returns:
            ``(batch, n_mels, frames)``.
        """
        magnitude = stft_magnitude(waveform, self.config)
        bank = self.filterbank(device=magnitude.device, dtype=magnitude.dtype)
        return torch.matmul(bank, magnitude)

    def log_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute a natural-log mel spectrogram.

        Natural log rather than decibels because that is what HiFi-GAN,
        BigVGAN, and the diffusion audio models built on them consume. A dB
        spectrogram differs by a constant factor and a per-utterance reference,
        and feeding one to a vocoder trained on the other produces a plausible
        but wrong timbre rather than an obvious failure.

        Args:
            waveform: ``(batch, samples)`` or ``(batch, 1, samples)``.

        Returns:
            ``(batch, n_mels, frames)``.
        """
        return torch.log(torch.clamp(self(waveform), min=self.config.log_epsilon))
