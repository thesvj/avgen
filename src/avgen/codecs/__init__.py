"""Codecs: the only place in avgen that knows what a pixel is.

Above this layer everything is latents and tokens; below it, files and waveforms.
Three ideas hold the subsystem together:

* **Protocols, not base classes** (:mod:`avgen.codecs.protocols`). A frozen
  diffusers VAE, an analytic reference transform, and a future learned tokeniser
  are interchangeable without a shared ancestor.
* **Fingerprints**. Every codec carries a stable hash of its identity and config,
  written into :class:`~avgen.core.batch.MediaBatchSpec`, so latents from two
  different autoencoders can never be silently mixed into one batch.
* **Normalisation is explicit** (:mod:`avgen.codecs.normalization`). It lives
  next to the codec, versioned and inspectable, rather than being baked into
  ``encode`` where nobody can see which statistics a cached latent shard was
  built with.

Optional dependencies (``diffusers``, ``transformers``) are imported inside the
functions that need them, in :mod:`avgen.codecs.adapters`. Importing this package
requires nothing but ``torch``.
"""

from avgen.codecs.adapters import (
    DiffusersVideoCodec,
    TilingConfig,
    TransformersTextEncoder,
    blend_tiles,
)
from avgen.codecs.mel import (
    MelConfig,
    MelFrontend,
    hz_to_mel,
    mel_filterbank,
    mel_to_hz,
    stft_magnitude,
)
from avgen.codecs.normalization import (
    LatentStatistics,
    denormalize_latents,
    normalize_latents,
    resolve_latent_normalization,
)
from avgen.codecs.protocols import (
    AudioCodec,
    TextEncoder,
    VideoCodec,
    Vocoder,
    codec_fingerprint,
)
from avgen.codecs.reference import (
    ReferenceAudioCodec,
    ReferenceTextEncoder,
    ReferenceVideoCodec,
)

__all__ = [
    "AudioCodec",
    "DiffusersVideoCodec",
    "LatentStatistics",
    "MelConfig",
    "MelFrontend",
    "ReferenceAudioCodec",
    "ReferenceTextEncoder",
    "ReferenceVideoCodec",
    "TextEncoder",
    "TilingConfig",
    "TransformersTextEncoder",
    "VideoCodec",
    "Vocoder",
    "blend_tiles",
    "codec_fingerprint",
    "denormalize_latents",
    "hz_to_mel",
    "mel_filterbank",
    "mel_to_hz",
    "normalize_latents",
    "resolve_latent_normalization",
    "stft_magnitude",
]
