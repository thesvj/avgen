"""Dependency-free reference codecs: exactly invertible, no learned weights.

Tests, CI, the simulator, and every smoke run need a codec. Downloading a 400MB
VAE to check that a sampler loop advances is absurd, and a randomly initialised
stand-in is worse than useless — it gives a round-trip test nothing to assert, so
a genuine bug in the encode/decode plumbing sails through green.

The reference codecs are therefore **analytic and exactly invertible**. Encode is
a rearrangement followed by an orthogonal channel mixing; decode applies the
inverse of both. Round-tripping any tensor returns it to within floating-point
epsilon, so ``assert_close(codec.decode(codec.encode(x)), x)`` is a real test of
a real property, and any future change that breaks the geometry fails it.

Two design choices are worth explaining.

**Why a rearrangement rather than a learned encoder.** Pixel-unshuffle moves
spatial and temporal extent into channels. It has exactly the shape signature of
a real video VAE — it compresses ``T``, ``H``, and ``W`` by fixed factors and
widens the channel axis — so latent shapes, patch layouts, token counts, and
memory estimates computed against it are structurally identical to the ones a
real codec produces. Nothing downstream can tell the difference.

**Why an orthogonal mixing on top.** Without it, latent channel ``k`` would be a
verbatim copy of a pixel, and a bug that silently transposed or dropped channels
would still round-trip. The mixing is a Householder reflection ``Q = I - 2vvᵀ``:
exactly orthogonal by construction, its own inverse (so decode and encode share
one matrix), one matmul to apply, and fully determined by a seed.

The text encoder is not invertible — no text encoder is — but it is
deterministic, byte-level, and dependency-free, which is what a test needs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import torch

from avgen.codecs.normalization import LatentStatistics
from avgen.codecs.protocols import codec_fingerprint

__all__ = [
    "ReferenceAudioCodec",
    "ReferenceTextEncoder",
    "ReferenceVideoCodec",
]


@lru_cache(maxsize=32)
def _householder(width: int, seed: int) -> torch.Tensor:
    """Return a deterministic symmetric orthogonal mixing matrix.

    ``Q = I - 2vvᵀ`` for a unit vector ``v``. Symmetric and orthogonal, hence its
    own inverse, which is why encode and decode can share one matrix instead of
    storing a factorisation. Built in float64 and kept there so that the
    orthogonality error is ~1e-16 rather than ~1e-7; callers cast at use.

    Args:
        width: Matrix size.
        seed: Seed determining ``v``.

    Returns:
        ``(width, width)`` float64 matrix on CPU.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    vector = torch.randn((width,), generator=generator, dtype=torch.float64)
    # A zero draw is astronomically unlikely but would produce a NaN direction,
    # and a codec that NaNs one time in 2^64 is not a codec you can debug.
    norm = torch.linalg.vector_norm(vector)
    if float(norm) == 0.0:
        vector = torch.ones((width,), dtype=torch.float64)
        norm = torch.linalg.vector_norm(vector)
    vector = vector / norm
    return torch.eye(width, dtype=torch.float64) - 2.0 * torch.outer(vector, vector)


def _mix_channels(tensor: torch.Tensor, seed: int) -> torch.Tensor:
    """Apply the Householder mixing along dimension one.

    Args:
        tensor: ``(batch, channels, ...)``.
        seed: Seed determining the reflection.

    Returns:
        The mixed tensor, same shape and dtype.
    """
    width = tensor.shape[1]
    matrix = _householder(width, seed).to(device=tensor.device, dtype=tensor.dtype)
    # Channels must be the contracted axis; moving them last lets one matmul do
    # the job for any rank, which is what keeps the video (rank 5) and audio
    # (rank 3) paths identical.
    return (tensor.movedim(1, -1) @ matrix).movedim(-1, 1)


@dataclass(frozen=True, slots=True)
class ReferenceVideoCodec:
    """An analytic, exactly invertible stand-in for a video VAE.

    Encode is ``pixel-unshuffle -> orthogonal channel mix -> scale``; decode is
    the exact inverse. There are no parameters, no downloads, and no
    nondeterminism.

    Args:
        channels: Pixel channel count. Three for RGB.
        temporal_compression: Pixel frames per latent frame.
        spatial_compression: Pixel rows and columns per latent row and column.
        scale: Multiplier applied after mixing. Left at 1.0 by default so the
            latent space is already standardised; set it to something else to
            exercise the normalisation path in a test.
        seed: Determines the channel mixing.

    Raises:
        ValueError: If any factor is non-positive or ``scale`` is not a positive
            finite number.
    """

    channels: int = 3
    temporal_compression: int = 1
    spatial_compression: int = 2
    scale: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        """Validate the compression factors and the scale."""
        for name in ("channels", "temporal_compression", "spatial_compression"):
            value = getattr(self, name)
            if isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer; got {value!r}")
        if not math.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError(f"scale must be finite and positive; got {self.scale!r}")

    @property
    def fingerprint(self) -> str:
        """Stable identity + config hash."""
        return codec_fingerprint(
            "video",
            "avgen.reference",
            {
                "channels": self.channels,
                "temporal_compression": self.temporal_compression,
                "spatial_compression": self.spatial_compression,
                "scale": self.scale,
                "seed": self.seed,
            },
        )

    @property
    def latent_channels(self) -> int:
        """Latent channel count.

        The rearrangement is lossless, so every pixel value has to land
        somewhere: the channel axis widens by exactly the product of the
        compression factors.
        """
        return (
            self.channels
            * self.temporal_compression
            * self.spatial_compression
            * self.spatial_compression
        )

    @property
    def latent_statistics(self) -> LatentStatistics:
        """Per-channel statistics of this codec's latent space.

        An orthogonal mixing preserves variance, so the latents are on the same
        scale as the pixels apart from the explicit :attr:`scale` factor. That
        makes the statistics exactly ``std = scale`` with zero mean, and there is
        nothing to measure empirically.
        """
        return LatentStatistics.from_scalar(
            self.latent_channels, mean=0.0, std=self.scale
        )

    def latent_shape(
        self, pixel_shape: Sequence[int]
    ) -> tuple[int, int, int, int, int]:
        """Return the latent shape produced for a pixel shape.

        Args:
            pixel_shape: ``(batch, channels, frames, height, width)``.

        Returns:
            The latent shape.

        Raises:
            ValueError: If the shape is not rank 5, the channel count disagrees,
                or an axis is not evenly compressible.
        """
        shape = tuple(int(value) for value in pixel_shape)
        if len(shape) != 5:
            raise ValueError(
                "pixel shape must be rank 5 (batch, channels, frames, height, "
                f"width); got {shape}"
            )
        batch, channels, frames, height, width = shape
        if channels != self.channels:
            raise ValueError(f"pixel channels must be {self.channels}; got {channels}")
        for name, extent, factor in (
            ("frames", frames, self.temporal_compression),
            ("height", height, self.spatial_compression),
            ("width", width, self.spatial_compression),
        ):
            if extent % factor != 0:
                raise ValueError(
                    f"{name}={extent} must be divisible by the compression factor "
                    f"{factor}; pad or crop at the data boundary"
                )
        return (
            batch,
            self.latent_channels,
            frames // self.temporal_compression,
            height // self.spatial_compression,
            width // self.spatial_compression,
        )

    def pixel_shape(
        self, latent_shape: Sequence[int]
    ) -> tuple[int, int, int, int, int]:
        """Return the pixel shape a latent shape decodes to.

        Args:
            latent_shape: ``(batch, latent_channels, latent_frames,
                latent_height, latent_width)``.

        Returns:
            The pixel shape.

        Raises:
            ValueError: If the shape is not rank 5 or the channel count
                disagrees.
        """
        shape = tuple(int(value) for value in latent_shape)
        if len(shape) != 5:
            raise ValueError(f"latent shape must be rank 5; got {shape}")
        batch, channels, frames, height, width = shape
        if channels != self.latent_channels:
            raise ValueError(
                f"latent channels must be {self.latent_channels}; got {channels}"
            )
        return (
            batch,
            self.channels,
            frames * self.temporal_compression,
            height * self.spatial_compression,
            width * self.spatial_compression,
        )

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """Compress pixels to latents.

        Args:
            pixels: ``(batch, channels, frames, height, width)``.

        Returns:
            ``(batch, latent_channels, latent_frames, latent_height,
            latent_width)``.

        Raises:
            ValueError: On a shape this codec cannot compress exactly.
            TypeError: If ``pixels`` is not floating point.
        """
        if not pixels.is_floating_point():
            raise TypeError(f"pixels must be floating point; got {pixels.dtype}")
        batch, _, latent_frames, latent_height, latent_width = self.latent_shape(
            tuple(pixels.shape)
        )
        temporal = self.temporal_compression
        spatial = self.spatial_compression
        folded = pixels.reshape(
            batch,
            self.channels,
            latent_frames,
            temporal,
            latent_height,
            spatial,
            latent_width,
            spatial,
        )
        # (b, c, lf, pt, lh, ps, lw, ps) -> (b, c, pt, ps, ps, lf, lh, lw): the
        # sub-sampled offsets become the fastest-varying part of the channel
        # axis, exactly as pixel-unshuffle defines it, so the inverse is a
        # single permutation back.
        permuted = folded.permute(0, 1, 3, 5, 7, 2, 4, 6)
        latents = permuted.reshape(
            batch, self.latent_channels, latent_frames, latent_height, latent_width
        )
        return _mix_channels(latents, self.seed) * self.scale

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Reconstruct pixels from latents.

        Args:
            latents: ``(batch, latent_channels, latent_frames, latent_height,
                latent_width)``.

        Returns:
            ``(batch, channels, frames, height, width)``.

        Raises:
            ValueError: On a shape mismatch with this codec's latent geometry.
            TypeError: If ``latents`` is not floating point.
        """
        if not latents.is_floating_point():
            raise TypeError(f"latents must be floating point; got {latents.dtype}")
        batch, _, frames, height, width = self.pixel_shape(tuple(latents.shape))
        temporal = self.temporal_compression
        spatial = self.spatial_compression
        # The reflection is its own inverse, so decode reuses the encode matrix.
        mixed = _mix_channels(latents / self.scale, self.seed)
        unfolded = mixed.reshape(
            batch,
            self.channels,
            temporal,
            spatial,
            spatial,
            frames // temporal,
            height // spatial,
            width // spatial,
        )
        permuted = unfolded.permute(0, 1, 5, 2, 6, 3, 7, 4)
        return permuted.reshape(batch, self.channels, frames, height, width)


@dataclass(frozen=True, slots=True)
class ReferenceAudioCodec:
    """An analytic, exactly invertible stand-in for a waveform audio codec.

    Frames the waveform into non-overlapping hops and mixes the resulting channel
    axis orthogonally — the one-dimensional analogue of
    :class:`ReferenceVideoCodec`, and invertible for the same reason.

    Non-overlapping framing rather than an STFT is deliberate: an STFT with
    ``center=True`` and a non-rectangular window is only invertible up to edge
    effects and a COLA condition, which would make the round-trip test assert
    something weaker than exactness.

    Args:
        channels: Waveform channel count. One for mono.
        hop_length: Waveform samples per latent frame.
        sample_rate: Waveform sample rate in hertz, carried for metadata only.
        scale: Multiplier applied after mixing.
        seed: Determines the channel mixing.

    Raises:
        ValueError: If any size is non-positive or ``scale`` is not positive.
    """

    channels: int = 1
    hop_length: int = 256
    sample_rate: int = 24000
    scale: float = 1.0
    seed: int = 1

    def __post_init__(self) -> None:
        """Validate sizes and the scale."""
        for name in ("channels", "hop_length", "sample_rate"):
            value = getattr(self, name)
            if isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer; got {value!r}")
        if not math.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError(f"scale must be finite and positive; got {self.scale!r}")

    @property
    def fingerprint(self) -> str:
        """Stable identity + config hash."""
        return codec_fingerprint(
            "audio",
            "avgen.reference",
            {
                "channels": self.channels,
                "hop_length": self.hop_length,
                "sample_rate": self.sample_rate,
                "scale": self.scale,
                "seed": self.seed,
            },
        )

    @property
    def latent_channels(self) -> int:
        """Latent channel count: one per waveform channel per sample in a hop."""
        return self.channels * self.hop_length

    @property
    def latent_rate(self) -> float:
        """Latent frames per second."""
        return self.sample_rate / self.hop_length

    @property
    def latent_statistics(self) -> LatentStatistics:
        """Per-channel statistics of this codec's latent space."""
        return LatentStatistics.from_scalar(
            self.latent_channels, mean=0.0, std=self.scale
        )

    def latent_frames(self, samples: int) -> int:
        """Return the latent frame count for a sample count.

        Args:
            samples: Waveform length in samples.

        Returns:
            Latent frames.

        Raises:
            ValueError: If ``samples`` is not a whole number of hops. Implicit
                truncation is rejected because it would desynchronise audio from
                video by a fraction of a frame, which is exactly the error class
                the exact-rational timebases in ``MediaBatchSpec`` exist to
                prevent.
        """
        if isinstance(samples, bool) or samples < 0:
            raise ValueError(f"samples must be non-negative; got {samples!r}")
        if samples % self.hop_length != 0:
            raise ValueError(
                f"samples={samples} must be divisible by hop_length="
                f"{self.hop_length}; pad the waveform at the data boundary"
            )
        return samples // self.hop_length

    def num_samples(self, frames: int) -> int:
        """Return the sample count a latent frame count decodes to.

        Args:
            frames: Latent frames.

        Returns:
            Waveform samples.
        """
        return int(frames) * self.hop_length

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compress a waveform to latents.

        Args:
            waveform: ``(batch, channels, samples)``.

        Returns:
            ``(batch, latent_channels, latent_frames)``.

        Raises:
            ValueError: On a shape this codec cannot compress exactly.
            TypeError: If ``waveform`` is not floating point.
        """
        if not waveform.is_floating_point():
            raise TypeError(f"waveform must be floating point; got {waveform.dtype}")
        if waveform.ndim != 3:
            raise ValueError(
                "waveform must have rank 3 (batch, channels, samples); got "
                f"{tuple(waveform.shape)}"
            )
        batch, channels, samples = tuple(waveform.shape)
        if channels != self.channels:
            raise ValueError(
                f"waveform channels must be {self.channels}; got {channels}"
            )
        frames = self.latent_frames(samples)
        folded = waveform.reshape(batch, channels, frames, self.hop_length)
        latents = folded.permute(0, 1, 3, 2).reshape(
            batch, self.latent_channels, frames
        )
        return _mix_channels(latents, self.seed) * self.scale

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Reconstruct a waveform from latents.

        Args:
            latents: ``(batch, latent_channels, latent_frames)``.

        Returns:
            ``(batch, channels, samples)``.

        Raises:
            ValueError: On a shape mismatch with this codec's latent geometry.
            TypeError: If ``latents`` is not floating point.
        """
        if not latents.is_floating_point():
            raise TypeError(f"latents must be floating point; got {latents.dtype}")
        if latents.ndim != 3:
            raise ValueError(
                "latents must have rank 3 (batch, channels, frames); got "
                f"{tuple(latents.shape)}"
            )
        batch, channels, frames = tuple(latents.shape)
        if channels != self.latent_channels:
            raise ValueError(
                f"latent channels must be {self.latent_channels}; got {channels}"
            )
        mixed = _mix_channels(latents / self.scale, self.seed)
        unfolded = mixed.reshape(batch, self.channels, self.hop_length, frames)
        return unfolded.permute(0, 1, 3, 2).reshape(
            batch, self.channels, self.num_samples(frames)
        )


@lru_cache(maxsize=8)
def _byte_table(width: int, seed: int) -> torch.Tensor:
    """Return a deterministic byte embedding table.

    Args:
        width: Feature width.
        seed: Seed determining the table.

    Returns:
        ``(256, width)`` float32 table on CPU, unit-scaled per row so the
        features are O(1) regardless of width.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    table = torch.randn((256, width), generator=generator, dtype=torch.float32)
    return table / math.sqrt(width)


@dataclass(frozen=True, slots=True)
class ReferenceTextEncoder:
    """A deterministic byte-level text encoder with no dependencies.

    Every UTF-8 byte of the prompt indexes a fixed pseudo-random embedding, and a
    sinusoidal position code is added so that two prompts sharing a bag of bytes
    but differing in order produce different features. That is enough structure
    for a test to assert the obvious properties — same prompt gives same
    features, different prompts give different features, an empty prompt gives
    the null context — without loading a 4.7B-parameter T5.

    The rejected alternative was a hash of the whole prompt broadcast across the
    token axis. It is simpler, but it makes every token of a prompt identical,
    so a cross-attention bug that collapsed the text axis would still pass.

    Args:
        width: Feature width. Must match the model's text projection input.
        max_length: Maximum tokens. Longer prompts are truncated at the byte
            level, which can split a multi-byte character; that is acceptable
            because this encoder is a fixture, not a language model.
        seed: Determines the embedding table.
        device: Device the features are allocated on.

    Raises:
        ValueError: If ``width`` or ``max_length`` is not positive.
    """

    width: int = 64
    max_length: int = 32
    seed: int = 2
    device: str = "cpu"

    def __post_init__(self) -> None:
        """Validate the feature geometry."""
        for name in ("width", "max_length"):
            value = getattr(self, name)
            if isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer; got {value!r}")

    @property
    def fingerprint(self) -> str:
        """Stable identity + config hash."""
        return codec_fingerprint(
            "text",
            "avgen.reference",
            {"width": self.width, "max_length": self.max_length, "seed": self.seed},
        )

    def encode(self, prompts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of prompts.

        Args:
            prompts: One prompt per sample.

        Returns:
            ``(features, mask)`` shaped ``(batch, max_length, width)`` and
            ``(batch, max_length)``. Padding positions carry zero features and a
            false mask, so an all-empty batch is bit-identical to the
            classifier-free-guidance null branch.

        Raises:
            ValueError: If ``prompts`` is empty.
            TypeError: If any entry is not a string.
        """
        if len(prompts) == 0:
            raise ValueError("prompts must contain at least one entry")
        for index, prompt in enumerate(prompts):
            if not isinstance(prompt, str):
                raise TypeError(
                    f"prompts[{index}] must be a string; got {type(prompt).__name__}"
                )
        device = torch.device(self.device)
        table = _byte_table(self.width, self.seed).to(device)
        batch = len(prompts)
        features = torch.zeros(
            (batch, self.max_length, self.width), dtype=torch.float32, device=device
        )
        mask = torch.zeros((batch, self.max_length), dtype=torch.bool, device=device)
        positions = _position_code(self.max_length, self.width, device)
        for index, prompt in enumerate(prompts):
            payload = prompt.encode("utf-8")[: self.max_length]
            if not payload:
                continue
            indices = torch.tensor(list(payload), dtype=torch.long, device=device)
            length = int(indices.shape[0])
            features[index, :length] = table[indices] + positions[:length]
            mask[index, :length] = True
        return features, mask


@lru_cache(maxsize=8)
def _position_code_cpu(length: int, width: int) -> torch.Tensor:
    """Return sinusoidal position features on CPU."""
    position = torch.arange(length, dtype=torch.float32)[:, None]
    half = max(width // 2, 1)
    frequency = torch.exp(
        torch.arange(half, dtype=torch.float32) * (-math.log(10000.0) / half)
    )
    angles = position * frequency[None, :]
    code = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
    if code.shape[1] < width:
        code = torch.nn.functional.pad(code, (0, width - code.shape[1]))
    return code[:, :width].contiguous()


def _position_code(length: int, width: int, device: torch.device) -> torch.Tensor:
    """Return sinusoidal position features on the requested device."""
    return _position_code_cpu(length, width).to(device)
