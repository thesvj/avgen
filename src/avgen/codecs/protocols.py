"""Codec interfaces: the boundary between pixels and the sequence interior.

A codec is the only thing in avgen that knows what a pixel is. Everything above
it — the objective, the model, the sampler — sees latents, and everything below
it sees files. Keeping that boundary at a Protocol rather than a base class means
a frozen pretrained VAE, an analytic reference transform, and a future
tokeniser-based codec are interchangeable without a shared ancestor.

**Fingerprints are the load-bearing part of this module.** Every codec exposes a
stable string that identifies both *which* codec it is and *how* it is
configured, and that string is written into
:attr:`~avgen.core.batch.MediaBatchSpec.video_codec_id` and
``audio_codec_id``. The failure it prevents is quiet and expensive: latents from
two different VAEs, or from the same VAE at two different tiling or dtype
settings, have incompatible geometry and incompatible statistics. Mixed into one
batch they produce a model that trains to a mediocre loss forever and generates
nothing recognisable, with no error anywhere. A batch assembler that compares
fingerprints catches it in the first step.

The fingerprint must be a pure function of identity and config — never of the
process, the device, or the wall clock — because it is compared across machines
and across months.

Shape contracts, stated once here and repeated in every implementation:

* Video pixels: ``(batch, channels, frames, height, width)``, float, nominally
  in ``[-1, 1]``.
* Video latents: ``(batch, latent_channels, latent_frames, latent_height,
  latent_width)``.
* Audio waveform: ``(batch, channels, samples)``, float in ``[-1, 1]``.
* Audio latents: ``(batch, latent_channels, latent_frames)``.
* Mel spectrogram: ``(batch, mel_bins, frames)``.
* Text features: ``(batch, tokens, width)`` with a ``(batch, tokens)`` bool mask.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import torch

from avgen.codecs.normalization import LatentStatistics

__all__ = [
    "AudioCodec",
    "TextEncoder",
    "VideoCodec",
    "Vocoder",
    "codec_fingerprint",
]


def codec_fingerprint(
    kind: str,
    identity: str,
    config: Mapping[str, Any] | None = None,
) -> str:
    """Return a stable identifier for a codec and its configuration.

    The digest is taken over canonical JSON with sorted keys, so two processes
    that build the same codec from the same arguments agree byte for byte, and a
    changed patch size or latent width produces a different fingerprint.

    Only 16 hex characters of the digest are kept. That is 64 bits: the birthday
    bound over even a million distinct codec configurations leaves a collision
    probability around 3e-8, and the string has to stay short enough to sit in a
    shard header and a log line without dominating either.

    Args:
        kind: Codec family, for example ``"video"``, ``"audio"``, or ``"text"``.
        identity: What this codec *is* — a model id, a class name, a checkpoint
            hash. Two codecs that decode differently must differ here.
        config: Configuration affecting the latents. Values must be JSON-safe;
            anything else is rendered via ``repr`` so the digest stays defined.

    Returns:
        A string of the form ``"<kind>:<identity>:<16 hex chars>"``.

    Raises:
        ValueError: If ``kind`` or ``identity`` is empty or untrimmed, which
            would produce an id that :class:`~avgen.core.batch.MediaBatchSpec`
            rejects.
    """
    for name, value in (("kind", kind), ("identity", identity)):
        if not value or value.strip() != value:
            raise ValueError(
                f"{name} must be a non-empty trimmed string; got {value!r}"
            )
    payload = json.dumps(
        dict(config or {}),
        sort_keys=True,
        separators=(",", ":"),
        default=repr,
    )
    digest = hashlib.sha256(f"{kind}\0{identity}\0{payload}".encode()).hexdigest()
    return f"{kind}:{identity}:{digest[:16]}"


@runtime_checkable
class VideoCodec(Protocol):
    """Compresses video pixels to latents and back.

    Implementations are frozen at training time: the codec is not optimised, and
    its weights are not part of the checkpoint the trainer writes. That is why
    the interface has no ``parameters()`` and no ``train()`` — a codec is
    infrastructure, not a model.

    The rejected alternative was folding the codec into the model as a first and
    last stage. It loses on three counts: latents can then not be cached to disk
    (and re-encoding pixels every epoch dominates the step time for video), the
    VAE decoder's activations dwarf the transformer's at high resolution, and the
    same trained transformer can no longer be paired with an improved decoder.
    """

    @property
    def fingerprint(self) -> str:
        """Stable identity + config hash written into ``MediaBatchSpec``."""
        ...

    @property
    def latent_channels(self) -> int:
        """Channel count of the emitted latents."""
        ...

    @property
    def temporal_compression(self) -> int:
        """Pixel frames per latent frame."""
        ...

    @property
    def spatial_compression(self) -> int:
        """Pixel rows (and columns) per latent row (and column)."""
        ...

    @property
    def latent_statistics(self) -> LatentStatistics:
        """Per-channel statistics used to standardise these latents."""
        ...

    def latent_shape(
        self, pixel_shape: Sequence[int]
    ) -> tuple[int, int, int, int, int]:
        """Return the latent shape produced for a pixel shape.

        Args:
            pixel_shape: ``(batch, channels, frames, height, width)``.

        Returns:
            ``(batch, latent_channels, latent_frames, latent_height,
            latent_width)``.

        Raises:
            ValueError: If the pixel shape is not rank 5 or is not compressible
                by this codec's factors.
        """
        ...

    def pixel_shape(
        self, latent_shape: Sequence[int]
    ) -> tuple[int, int, int, int, int]:
        """Return the pixel shape a latent shape decodes to.

        Args:
            latent_shape: ``(batch, latent_channels, latent_frames,
            latent_height, latent_width)``.

        Returns:
            ``(batch, channels, frames, height, width)``.

        Raises:
            ValueError: If the latent shape is not rank 5.
        """
        ...

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """Compress pixels to latents.

        Args:
            pixels: ``(batch, channels, frames, height, width)`` float tensor,
                nominally in ``[-1, 1]``.

        Returns:
            ``(batch, latent_channels, latent_frames, latent_height,
            latent_width)`` latents on the codec's own scale — *not*
            normalised. Normalisation is applied separately by
            :func:`~avgen.codecs.normalization.normalize_latents` so that the
            statistics stay visible and versioned rather than baked into the
            codec.

        Raises:
            ValueError: On a shape that this codec cannot compress exactly.
        """
        ...

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Reconstruct pixels from latents on the codec's own scale.

        Args:
            latents: ``(batch, latent_channels, latent_frames, latent_height,
                latent_width)``, denormalised.

        Returns:
            ``(batch, channels, frames, height, width)`` pixels.

        Raises:
            ValueError: On a shape mismatch with this codec's latent geometry.
        """
        ...


@runtime_checkable
class AudioCodec(Protocol):
    """Compresses an audio waveform to latents and back.

    Audio latents are one-dimensional in time, which is why
    :class:`~avgen.core.tokens.PatchLayout` has a
    :meth:`~avgen.core.tokens.PatchLayout.temporal` constructor: the audio stream
    is a video stream with unit spatial extent, and every downstream code path is
    shared rather than duplicated.
    """

    @property
    def fingerprint(self) -> str:
        """Stable identity + config hash written into ``MediaBatchSpec``."""
        ...

    @property
    def latent_channels(self) -> int:
        """Channel count of the emitted latents."""
        ...

    @property
    def sample_rate(self) -> int:
        """Waveform sample rate in hertz."""
        ...

    @property
    def hop_length(self) -> int:
        """Waveform samples per latent frame."""
        ...

    @property
    def latent_statistics(self) -> LatentStatistics:
        """Per-channel statistics used to standardise these latents."""
        ...

    def latent_frames(self, samples: int) -> int:
        """Return the latent frame count for a sample count.

        Args:
            samples: Waveform length in samples.

        Returns:
            Latent frames.

        Raises:
            ValueError: If ``samples`` is not a whole number of hops.
        """
        ...

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compress a waveform to latents.

        Args:
            waveform: ``(batch, channels, samples)`` float tensor in ``[-1, 1]``.

        Returns:
            ``(batch, latent_channels, latent_frames)`` latents on the codec's
            own scale.

        Raises:
            ValueError: On a shape this codec cannot compress exactly.
        """
        ...

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Reconstruct a waveform from latents on the codec's own scale.

        Args:
            latents: ``(batch, latent_channels, latent_frames)``, denormalised.

        Returns:
            ``(batch, channels, samples)`` waveform.

        Raises:
            ValueError: On a shape mismatch with this codec's latent geometry.
        """
        ...


@runtime_checkable
class Vocoder(Protocol):
    """Turns a mel spectrogram into a waveform.

    Kept separate from :class:`AudioCodec` because the two compose rather than
    substitute: a mel-domain audio model pairs a mel frontend with a vocoder,
    while a waveform-latent model uses an :class:`AudioCodec` and no vocoder at
    all. Merging them would force every waveform codec to pretend it has a mel
    bin count.
    """

    @property
    def fingerprint(self) -> str:
        """Stable identity + config hash."""
        ...

    @property
    def sample_rate(self) -> int:
        """Output sample rate in hertz."""
        ...

    @property
    def mel_bins(self) -> int:
        """Expected number of mel bins on the input."""
        ...

    def decode(self, mel: torch.Tensor) -> torch.Tensor:
        """Synthesise a waveform from a mel spectrogram.

        Args:
            mel: ``(batch, mel_bins, frames)``, log-scaled unless the
                implementation documents otherwise.

        Returns:
            ``(batch, 1, samples)`` waveform in ``[-1, 1]``.

        Raises:
            ValueError: If the mel bin count does not match :attr:`mel_bins`.
        """
        ...


@runtime_checkable
class TextEncoder(Protocol):
    """Turns prompts into frozen cross-attention features.

    The encoder is frozen and its output is cacheable, which is the whole reason
    :class:`~avgen.core.tokens.TextContext` exists as a separate contract from
    :class:`~avgen.core.tokens.TokenStream`. A production training run encodes
    its captions once, offline, and never loads a text tower on a training node —
    a 4.7B-parameter T5 that contributes no gradients has no business occupying
    memory next to the model being trained.

    The mask matters as much as the features. Classifier-free guidance needs a
    null branch, and avgen defines the null as *zeroed features with an all-false
    mask* (see :meth:`~avgen.core.tokens.TextContext.nullified`) rather than the
    encoding of an empty string, so that the null is encoder-independent and free
    to construct.
    """

    @property
    def fingerprint(self) -> str:
        """Stable identity + config hash."""
        ...

    @property
    def width(self) -> int:
        """Feature width of the emitted context."""
        ...

    @property
    def max_length(self) -> int:
        """Maximum token count; longer prompts are truncated."""
        ...

    def encode(self, prompts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of prompts.

        Args:
            prompts: One prompt per sample. An empty string is legal and must
                produce an all-false mask, matching the guidance null branch.

        Returns:
            ``(features, mask)`` shaped ``(batch, tokens, width)`` and
            ``(batch, tokens)`` with ``mask`` of dtype ``torch.bool``.

        Raises:
            ValueError: If ``prompts`` is empty.
        """
        ...
