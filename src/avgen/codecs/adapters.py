"""Lazy adapters for pretrained towers, plus tiled encode/decode.

Nothing in this module is imported when ``avgen.codecs`` is imported. The heavy
dependencies — ``diffusers`` for VAEs, ``transformers`` for text towers — are
imported inside the function that first needs them, and a missing one raises a
:class:`RuntimeError` naming the extra to install. That rule exists because the
steady-state training hot path must never touch an optional dependency, and
because a cluster image that does not run text encoding should not have to carry
a 2GB dependency tree to import the package.

**Tiling.** A video VAE decoder is the single largest activation consumer in a
generation pipeline, and it is not close. Decoding one second of 720p video from
a 8x8x4-compressing latent materialises intermediate feature maps at full pixel
resolution across every decoder stage; at 4-8 GB per second of video it will OOM
a card that trained the transformer comfortably. Tiling decodes overlapping
spatial windows and blends them, trading time for memory at a roughly constant
total FLOP count.

The trade-off is real and worth stating plainly: **tiles produce seams.** A
convolutional decoder's receptive field spans the tile boundary, so a tile
decoded without its neighbours' context differs slightly from the same region
decoded whole, and an abrupt tile join shows up as a visible grid — most
obviously in flat gradients like sky. Overlapping the tiles and cross-fading the
overlap with a linear ramp hides it, at the cost of decoding the overlap region
twice. An overlap of roughly a quarter of the tile is the usual place to land:
smaller and the seam returns, larger and the redundant compute dominates.

The same argument applies along time. A temporal chunk boundary produces a
visible flicker at the join, and the same overlap-and-blend fix applies.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from avgen.codecs.normalization import LatentStatistics, resolve_latent_normalization
from avgen.codecs.protocols import codec_fingerprint

__all__ = [
    "DiffusersVideoCodec",
    "TilingConfig",
    "TransformersTextEncoder",
    "blend_tiles",
]


@dataclass(frozen=True, slots=True)
class TilingConfig:
    """Spatial and temporal chunking for a memory-bound codec.

    Sizes are given in the *pixel* domain and converted to latents by the codec's
    compression factors, because the memory that forces tiling is pixel-domain
    memory and reasoning about it in latent units means multiplying by 8 in your
    head every time.

    Args:
        enabled: Whether to tile at all. Off by default: tiling is slower and
            slightly lossy, so it should be an explicit response to an OOM
            rather than an always-on default.
        tile_size: Pixel edge length of one square tile.
        tile_overlap: Pixel overlap between neighbouring tiles. Must be smaller
            than ``tile_size``, or the tiles never advance.
        temporal_chunk: Pixel frames per temporal chunk. Zero disables temporal
            chunking.
        temporal_overlap: Frames of overlap between temporal chunks.

    Raises:
        ValueError: If a size is negative, an overlap is not smaller than its
            extent, or ``tile_size`` is zero while tiling is enabled.
    """

    enabled: bool = False
    tile_size: int = 512
    tile_overlap: int = 128
    temporal_chunk: int = 0
    temporal_overlap: int = 0

    def __post_init__(self) -> None:
        """Validate the geometry."""
        names = ("tile_size", "tile_overlap", "temporal_chunk", "temporal_overlap")
        for name in names:
            value = getattr(self, name)
            if isinstance(value, bool) or value < 0:
                raise ValueError(
                    f"{name} must be a non-negative integer; got {value!r}"
                )
        if self.enabled and self.tile_size < 1:
            raise ValueError("tile_size must be positive when tiling is enabled")
        if self.tile_size and self.tile_overlap >= self.tile_size:
            raise ValueError(
                f"tile_overlap={self.tile_overlap} must be smaller than "
                f"tile_size={self.tile_size}; equal or larger means the window "
                "never advances"
            )
        if self.temporal_chunk and self.temporal_overlap >= self.temporal_chunk:
            raise ValueError(
                f"temporal_overlap={self.temporal_overlap} must be smaller than "
                f"temporal_chunk={self.temporal_chunk}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""
        return {
            "enabled": self.enabled,
            "tile_size": self.tile_size,
            "tile_overlap": self.tile_overlap,
            "temporal_chunk": self.temporal_chunk,
            "temporal_overlap": self.temporal_overlap,
        }


def blend_tiles(
    total: int,
    tile: int,
    overlap: int,
) -> list[tuple[int, int]]:
    """Return the ``(start, stop)`` windows covering an axis.

    The last window is pulled back to end exactly at ``total`` rather than being
    allowed to run past it, so no window is short and every position is covered.
    That matters for the blending weights: a short final tile would get a ramp of
    the wrong length and leave a visible bright or dark band at the edge.

    Args:
        total: Axis extent.
        tile: Window length.
        overlap: Overlap between consecutive windows.

    Returns:
        Windows in increasing order, at least one, covering ``[0, total)``.

    Raises:
        ValueError: If ``tile`` is not positive or ``overlap`` is not smaller
            than ``tile``.
    """
    if tile < 1:
        raise ValueError(f"tile must be positive; got {tile!r}")
    if overlap >= tile:
        raise ValueError(f"overlap={overlap} must be smaller than tile={tile}")
    if total <= tile:
        return [(0, total)]
    stride = tile - overlap
    starts = list(range(0, max(total - tile, 0) + 1, stride))
    if starts[-1] + tile < total:
        starts.append(total - tile)
    return [(start, start + tile) for start in starts]


def _ramp(
    length: int,
    ramp_in: int,
    ramp_out: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a 1-D blending weight with linear fades at the ends.

    The interior is 1.0 and the fades are strictly positive, so the accumulated
    weight never reaches zero anywhere and the normalising division is always
    safe.
    """
    weight = torch.ones(length, device=device, dtype=dtype)
    if ramp_in > 0:
        weight[:ramp_in] = torch.linspace(
            1.0 / (ramp_in + 1), 1.0, ramp_in, device=device, dtype=dtype
        )
    if ramp_out > 0:
        weight[length - ramp_out :] = torch.linspace(
            1.0, 1.0 / (ramp_out + 1), ramp_out, device=device, dtype=dtype
        )
    return weight


def _tiled_apply(
    tensor: torch.Tensor,
    transform: Callable[[torch.Tensor], torch.Tensor],
    *,
    tiling: TilingConfig,
    scale_time: float,
    scale_space: float,
) -> torch.Tensor:
    """Apply ``transform`` over overlapping windows and blend the results.

    Accumulates ``output * weight`` and ``weight`` separately and divides at the
    end. That is more memory than writing tiles into place, but it is the only
    formulation where the blend is exactly a partition of unity regardless of how
    the windows land, including the pulled-back final window whose overlap with
    its predecessor is not the nominal one.

    Args:
        tensor: ``(batch, channels, frames, height, width)`` input.
        transform: The untiled operation.
        tiling: Window geometry, in *input* units.
        scale_time: Output frames per input frame.
        scale_space: Output pixels per input pixel on each spatial axis.

    Returns:
        The blended output.
    """
    _, _, frames, height, width = tuple(tensor.shape)
    time_windows = (
        blend_tiles(frames, tiling.temporal_chunk, tiling.temporal_overlap)
        if tiling.temporal_chunk
        else [(0, frames)]
    )
    row_windows = blend_tiles(height, tiling.tile_size, tiling.tile_overlap)
    column_windows = blend_tiles(width, tiling.tile_size, tiling.tile_overlap)

    buffers: tuple[torch.Tensor, torch.Tensor] | None = None
    for time_start, time_stop in time_windows:
        for row_start, row_stop in row_windows:
            for column_start, column_stop in column_windows:
                window = tensor[
                    :,
                    :,
                    time_start:time_stop,
                    row_start:row_stop,
                    column_start:column_stop,
                ]
                output = transform(window)
                if buffers is None:
                    empty = tensor.new_zeros(
                        (
                            output.shape[0],
                            output.shape[1],
                            _scaled(frames, scale_time),
                            _scaled(height, scale_space),
                            _scaled(width, scale_space),
                        ),
                        dtype=output.dtype,
                    )
                    buffers = (empty, torch.zeros_like(empty))
                accumulator, weights = buffers
                offsets = (
                    _scaled(time_start, scale_time),
                    _scaled(row_start, scale_space),
                    _scaled(column_start, scale_space),
                )
                weight = _window_weight(
                    output,
                    time_first=time_start == time_windows[0][0],
                    time_last=time_stop == frames,
                    row_first=row_start == 0,
                    row_last=row_stop == height,
                    column_first=column_start == 0,
                    column_last=column_stop == width,
                    time_overlap=_scaled(tiling.temporal_overlap, scale_time),
                    space_overlap=_scaled(tiling.tile_overlap, scale_space),
                )
                slices = (
                    slice(None),
                    slice(None),
                    slice(offsets[0], offsets[0] + output.shape[2]),
                    slice(offsets[1], offsets[1] + output.shape[3]),
                    slice(offsets[2], offsets[2] + output.shape[4]),
                )
                accumulator[slices] += output * weight
                weights[slices] += weight
    if buffers is None:
        raise ValueError("tiling produced no windows; check the tiling configuration")
    accumulator, weights = buffers
    return accumulator / torch.clamp(weights, min=1e-6)


def _scaled(value: int, factor: float) -> int:
    """Scale an extent, requiring the result to be a whole number."""
    scaled = value * factor
    rounded = round(scaled)
    if not math.isclose(scaled, rounded, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(
            f"tile extent {value} does not scale to a whole number under factor "
            f"{factor}; choose a tile size that is a multiple of the compression "
            "factor"
        )
    return rounded


def _window_weight(
    output: torch.Tensor,
    *,
    time_first: bool,
    time_last: bool,
    row_first: bool,
    row_last: bool,
    column_first: bool,
    column_last: bool,
    time_overlap: int,
    space_overlap: int,
) -> torch.Tensor:
    """Return the separable blending weight for one window.

    Fades are applied only on interior edges. Fading the outer boundary of the
    whole tensor would darken the frame border, since there is no neighbour to
    supply the missing weight.
    """
    _, _, frames, height, width = tuple(output.shape)
    device, dtype = output.device, output.dtype
    time_weight = _ramp(
        frames,
        0 if time_first else min(time_overlap, frames),
        0 if time_last else min(time_overlap, frames),
        device=device,
        dtype=dtype,
    )
    row_weight = _ramp(
        height,
        0 if row_first else min(space_overlap, height),
        0 if row_last else min(space_overlap, height),
        device=device,
        dtype=dtype,
    )
    column_weight = _ramp(
        width,
        0 if column_first else min(space_overlap, width),
        0 if column_last else min(space_overlap, width),
        device=device,
        dtype=dtype,
    )
    return (
        time_weight[None, None, :, None, None]
        * row_weight[None, None, None, :, None]
        * column_weight[None, None, None, None, :]
    )


def _require(module: str, extra: str) -> Any:
    """Import an optional dependency or raise a message that names the fix.

    Args:
        module: Module to import.
        extra: The avgen extra that provides it.

    Returns:
        The imported module.

    Raises:
        RuntimeError: If the module is not installed.
    """
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as error:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            f"{module} is required for this adapter but is not installed; "
            f"install it with: pip install 'avgen[{extra}]'"
        ) from error


class DiffusersVideoCodec:
    """Wraps any ``diffusers`` AutoencoderKL-family video VAE.

    The adapter exists so that the rest of avgen never learns diffusers' calling
    conventions — ``.latent_dist.sample()`` versus ``.latent_dist.mode()``,
    ``scaling_factor`` versus ``shift_factor``, ``AutoencoderKL`` versus
    ``AutoencoderKLCogVideoX`` versus ``AutoencoderKLWan`` — and so that swapping
    the VAE is a config change rather than a code change.

    The mode of the posterior is used rather than a sample. Sampling the VAE
    posterior at *inference* would inject noise that the sampler then has to
    denoise a second time; sampling at *training* time is a legitimate choice, but
    it belongs to the offline latent-preparation job, where it is reproducible
    from a seed, not to a live codec call.

    Args:
        model_id: Hub id or local path.
        subfolder: Subfolder inside the repository, typically ``"vae"``.
        dtype: Torch dtype name for the loaded weights.
        device: Device to place the module on.
        tiling: Spatial and temporal chunking.
        statistics: Latent statistics override. When omitted they are derived
            from the loaded config's ``scaling_factor`` and ``shift_factor``.
        revision: Repository revision to pin. Pinning is strongly advised: an
            upstream reupload changes the latents and silently invalidates every
            cached latent shard.

    Raises:
        ValueError: If ``model_id`` is empty.
    """

    __slots__ = (
        "_module",
        "_statistics",
        "device",
        "dtype",
        "model_id",
        "revision",
        "statistics_override",
        "subfolder",
        "tiling",
    )

    def __init__(
        self,
        model_id: str,
        *,
        subfolder: str | None = "vae",
        dtype: str = "float32",
        device: str = "cpu",
        tiling: TilingConfig | None = None,
        statistics: LatentStatistics | None = None,
        revision: str | None = None,
    ) -> None:
        if not model_id or model_id.strip() != model_id:
            raise ValueError(
                f"model_id must be a non-empty trimmed string; got {model_id!r}"
            )
        self.model_id = model_id
        self.subfolder = subfolder
        self.dtype = dtype
        self.device = device
        self.tiling = tiling or TilingConfig()
        self.revision = revision
        self.statistics_override = statistics
        self._module: Any = None
        self._statistics: LatentStatistics | None = statistics

    @property
    def fingerprint(self) -> str:
        """Stable identity + config hash.

        Computed from the *declared* configuration only, so it is available
        before the weights are downloaded and is identical on a machine that
        never loads them — a shard writer and a training node must agree on the
        id without both paying to instantiate the VAE.
        """
        return codec_fingerprint(
            "video",
            f"diffusers:{self.model_id}",
            {
                "subfolder": self.subfolder,
                "dtype": self.dtype,
                "revision": self.revision,
                "tiling": self.tiling.to_dict(),
                "statistics": (
                    self.statistics_override.to_dict()
                    if self.statistics_override is not None
                    else None
                ),
            },
        )

    def module(self) -> Any:
        """Return the loaded VAE, loading it on first use.

        Returns:
            The diffusers autoencoder module, in eval mode with gradients
            disabled. A frozen codec that still builds an autograd graph is a
            silent memory leak of exactly the size of its activations.

        Raises:
            RuntimeError: If ``diffusers`` is not installed.
        """
        if self._module is None:
            diffusers = _require("diffusers", "codecs")
            loader = getattr(diffusers, "AutoencoderKL", None)
            automodel = getattr(diffusers, "AutoModel", None)
            factory = automodel if automodel is not None else loader
            if factory is None:  # pragma: no cover - defensive
                raise RuntimeError(
                    "the installed diffusers exposes neither AutoModel nor "
                    "AutoencoderKL; install it with: pip install 'avgen[codecs]'"
                )
            module = factory.from_pretrained(
                self.model_id,
                subfolder=self.subfolder,
                revision=self.revision,
                torch_dtype=getattr(torch, self.dtype),
            )
            module.eval().requires_grad_(False)
            self._module = module.to(self.device)
        return self._module

    @property
    def latent_channels(self) -> int:
        """Channel count of the emitted latents."""
        return int(self.module().config.latent_channels)

    @property
    def temporal_compression(self) -> int:
        """Pixel frames per latent frame.

        Read from whichever attribute the particular VAE class uses, defaulting
        to 1 for an image VAE applied frame by frame.
        """
        config = self.module().config
        for name in (
            "temporal_compression_ratio",
            "temporal_downsample_factor",
            "time_compression_ratio",
        ):
            value = getattr(config, name, None)
            if value is not None:
                return int(value)
        return 1

    @property
    def spatial_compression(self) -> int:
        """Pixel rows and columns per latent row and column."""
        config = self.module().config
        for name in ("spatial_compression_ratio", "spatial_downsample_factor"):
            value = getattr(config, name, None)
            if value is not None:
                return int(value)
        # The classic AutoencoderKL has no such field: its factor is implied by
        # the number of downsampling blocks, each halving the resolution.
        blocks = getattr(config, "block_out_channels", None)
        return 2 ** (len(blocks) - 1) if blocks else 8

    @property
    def latent_statistics(self) -> LatentStatistics:
        """Per-channel statistics for this VAE's latent space."""
        if self._statistics is None:
            self._statistics = resolve_latent_normalization(
                self.module(), channels=self.latent_channels
            )
        return self._statistics

    def latent_shape(
        self, pixel_shape: Sequence[int]
    ) -> tuple[int, int, int, int, int]:
        """Return the latent shape produced for a pixel shape.

        Args:
            pixel_shape: ``(batch, channels, frames, height, width)``.

        Returns:
            The latent shape.

        Raises:
            ValueError: If the shape is not rank 5 or an axis is not evenly
                compressible.
        """
        shape = tuple(int(value) for value in pixel_shape)
        if len(shape) != 5:
            raise ValueError(f"pixel shape must be rank 5; got {shape}")
        batch, _, frames, height, width = shape
        temporal = self.temporal_compression
        spatial = self.spatial_compression
        for name, extent, factor in (
            ("height", height, spatial),
            ("width", width, spatial),
        ):
            if extent % factor != 0:
                raise ValueError(f"{name}={extent} must be divisible by {factor}")
        # Causal video VAEs emit an extra latent frame for the first pixel frame,
        # so the count is ceil rather than exact division; getting this wrong by
        # one frame shifts every audio-video alignment downstream.
        latent_frames = (frames - 1) // temporal + 1 if temporal > 1 else frames
        return (
            batch,
            self.latent_channels,
            latent_frames,
            height // spatial,
            width // spatial,
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
            ValueError: If the shape is not rank 5.
        """
        shape = tuple(int(value) for value in latent_shape)
        if len(shape) != 5:
            raise ValueError(f"latent shape must be rank 5; got {shape}")
        batch, _, frames, height, width = shape
        temporal = self.temporal_compression
        spatial = self.spatial_compression
        pixel_frames = (frames - 1) * temporal + 1 if temporal > 1 else frames
        return (
            batch,
            int(getattr(self.module().config, "out_channels", 3)),
            pixel_frames,
            height * spatial,
            width * spatial,
        )

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """Compress pixels to latents on the VAE's own scale.

        Args:
            pixels: ``(batch, channels, frames, height, width)`` in ``[-1, 1]``.

        Returns:
            ``(batch, latent_channels, latent_frames, latent_height,
            latent_width)``.

        Raises:
            RuntimeError: If ``diffusers`` is not installed.
        """
        if self.tiling.enabled:
            ratio = 1.0 / self.spatial_compression
            time_ratio = 1.0 / self.temporal_compression
            return _tiled_apply(
                pixels,
                self._encode_whole,
                tiling=self.tiling,
                scale_time=time_ratio,
                scale_space=ratio,
            )
        return self._encode_whole(pixels)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Reconstruct pixels from latents on the VAE's own scale.

        Args:
            latents: ``(batch, latent_channels, latent_frames, latent_height,
                latent_width)``, denormalised.

        Returns:
            ``(batch, channels, frames, height, width)``.

        Raises:
            RuntimeError: If ``diffusers`` is not installed.
        """
        if self.tiling.enabled:
            spatial = self.spatial_compression
            temporal = self.temporal_compression
            latent_tiling = TilingConfig(
                enabled=True,
                tile_size=max(self.tiling.tile_size // spatial, 1),
                tile_overlap=self.tiling.tile_overlap // spatial,
                temporal_chunk=self.tiling.temporal_chunk // max(temporal, 1),
                temporal_overlap=self.tiling.temporal_overlap // max(temporal, 1),
            )
            return _tiled_apply(
                latents,
                self._decode_whole,
                tiling=latent_tiling,
                scale_time=float(temporal),
                scale_space=float(spatial),
            )
        return self._decode_whole(latents)

    def _encode_whole(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode without tiling."""
        module = self.module()
        with torch.no_grad():
            posterior = module.encode(pixels.to(module.dtype))
        distribution = getattr(posterior, "latent_dist", None)
        if distribution is not None:
            return distribution.mode().to(pixels.dtype)
        return getattr(posterior, "latents", posterior).to(pixels.dtype)

    def _decode_whole(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode without tiling."""
        module = self.module()
        with torch.no_grad():
            decoded = module.decode(latents.to(module.dtype))
        return getattr(decoded, "sample", decoded).to(latents.dtype)


class TransformersTextEncoder:
    """Wraps any Hugging Face text tower — T5, Gemma, Qwen, and relatives.

    Returns the final hidden states and the attention mask, not a pooled vector.
    Video DiTs cross-attend to the full token sequence, and pooling discards
    exactly the compositional detail — which adjective binds to which noun, the
    order of events in a long prompt — that a video model needs and an image
    model can often get away with losing.

    The tower is frozen and run under ``no_grad``. In production it should not run
    on a training node at all: encode captions offline into a shard and let the
    trainer read features.

    Args:
        model_id: Hub id or local path.
        max_length: Token budget. Prompts are padded to exactly this length so
            every batch has one static shape and ``torch.compile`` sees one
            graph.
        dtype: Torch dtype name for the loaded weights.
        device: Device to place the tower on.
        revision: Repository revision to pin.
        subfolder: Subfolder inside the repository, for combined pipelines.

    Raises:
        ValueError: If ``model_id`` is empty or ``max_length`` is not positive.
    """

    __slots__ = (
        "_model",
        "_tokenizer",
        "device",
        "dtype",
        "max_length",
        "model_id",
        "revision",
        "subfolder",
    )

    def __init__(
        self,
        model_id: str,
        *,
        max_length: int = 226,
        dtype: str = "float32",
        device: str = "cpu",
        revision: str | None = None,
        subfolder: str | None = None,
    ) -> None:
        if not model_id or model_id.strip() != model_id:
            raise ValueError(
                f"model_id must be a non-empty trimmed string; got {model_id!r}"
            )
        if isinstance(max_length, bool) or max_length < 1:
            raise ValueError(
                f"max_length must be a positive integer; got {max_length!r}"
            )
        self.model_id = model_id
        self.max_length = max_length
        self.dtype = dtype
        self.device = device
        self.revision = revision
        self.subfolder = subfolder
        self._model: Any = None
        self._tokenizer: Any = None

    @property
    def fingerprint(self) -> str:
        """Stable identity + config hash."""
        return codec_fingerprint(
            "text",
            f"transformers:{self.model_id}",
            {
                "max_length": self.max_length,
                "dtype": self.dtype,
                "revision": self.revision,
                "subfolder": self.subfolder,
            },
        )

    @property
    def width(self) -> int:
        """Feature width of the emitted context."""
        config = self.model().config
        for name in ("d_model", "hidden_size"):
            value = getattr(config, name, None)
            if value is not None:
                return int(value)
        raise RuntimeError(  # pragma: no cover - defensive
            f"cannot determine the hidden width of {self.model_id}"
        )

    def tokenizer(self) -> Any:
        """Return the tokenizer, loading it on first use.

        Returns:
            The Hugging Face tokenizer.

        Raises:
            RuntimeError: If ``transformers`` is not installed.
        """
        if self._tokenizer is None:
            transformers = _require("transformers", "text")
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_id, revision=self.revision, subfolder=self.subfolder or ""
            )
        return self._tokenizer

    def model(self) -> Any:
        """Return the text tower, loading it on first use.

        Returns:
            The Hugging Face model, in eval mode with gradients disabled.

        Raises:
            RuntimeError: If ``transformers`` is not installed.
        """
        if self._model is None:
            transformers = _require("transformers", "text")
            # T5 ships as an encoder-decoder; loading the whole thing wastes half
            # the parameters, so prefer the encoder-only class when it applies.
            factory = getattr(transformers, "AutoModel", None)
            if "t5" in self.model_id.lower():
                factory = getattr(transformers, "T5EncoderModel", factory)
            if factory is None:  # pragma: no cover - defensive
                raise RuntimeError(
                    "the installed transformers exposes no AutoModel; install it "
                    "with: pip install 'avgen[text]'"
                )
            model = factory.from_pretrained(
                self.model_id,
                revision=self.revision,
                subfolder=self.subfolder or "",
                torch_dtype=getattr(torch, self.dtype),
            )
            model.eval().requires_grad_(False)
            self._model = model.to(self.device)
        return self._model

    def encode(self, prompts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of prompts.

        Args:
            prompts: One prompt per sample.

        Returns:
            ``(features, mask)`` shaped ``(batch, max_length, width)`` and
            ``(batch, max_length)``.

        Raises:
            RuntimeError: If ``transformers`` is not installed.
            ValueError: If ``prompts`` is empty.
        """
        if len(prompts) == 0:
            raise ValueError("prompts must contain at least one entry")
        tokenizer = self.tokenizer()
        model = self.model()
        encoded = tokenizer(
            list(prompts),
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention = encoded["attention_mask"].to(self.device)
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention)
        features = getattr(output, "last_hidden_state", None)
        if features is None:  # pragma: no cover - defensive
            features = output[0]
        # Zeroing padded positions is redundant given the mask, but it makes the
        # features themselves safe to pool or to compare, and it keeps a padded
        # batch bit-identical to the same prompts batched differently.
        features = features * attention[..., None].to(features.dtype)
        return features.to(torch.float32), attention.to(torch.bool)
