"""Token streams: the sequence-first representation every model interior uses.

A video latent is naturally a dense 5-D grid, and that is how a VAE emits it and
how a shard file stores it. But a transformer does not see a grid — it sees a
sequence — and *everything* that makes large-scale video training possible
depends on the model interior being an explicit sequence:

* **Context parallelism** shards the sequence dimension across ranks. You cannot
  shard ``(B, C, T, H, W)`` along "part of T and part of H"; you can trivially
  shard ``(B, L, D)`` along ``L``. At 40k-200k tokens per clip this is the axis
  that decides whether a model fits at all.
* **Variable resolution in one batch** requires that two samples of different
  spatial extent become sequences of the same length class, distinguished only
  by their coordinates and mask.
* **Sequence packing** — putting a 2-second clip and an 8-second clip in one
  batch without padding to the longer — is only expressible on a sequence.
* **Ring and Ulysses attention** are sequence-dimension algorithms.

So: dense grids live at the data and codec boundary, and nowhere else. A
:class:`Patchifier` converts once on the way in and once on the way out, and
every block in between sees :class:`TokenStream`.

Coordinates are carried explicitly per token, in physical units — seconds for
time, latent pixels for space — rather than being implied by tensor position.
That is what lets a shuffled, packed, or CP-sharded sequence still know where
each of its tokens came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch
from torch.utils import _pytree

from avgen.core._validate import require_dtype, require_positive, require_shape

__all__ = ["PatchLayout", "TextContext", "TokenStream"]


@dataclass(frozen=True, slots=True)
class PatchLayout:
    """Static geometry needed to fold a token sequence back into a grid.

    This is metadata, not data: it holds no tensors, is hashable, and is safe to
    treat as a compile-time constant. Two batches with the same layout share a
    compiled graph.

    Args:
        frames: Latent frames before patchification.
        height: Latent rows before patchification.
        width: Latent columns before patchification.
        patch_frames: Temporal patch size.
        patch_height: Spatial patch height.
        patch_width: Spatial patch width.
        channels: Latent channels before patchification.

    Raises:
        ValueError: If any dimension is non-positive or a patch size does not
            divide its axis.
    """

    frames: int
    height: int
    width: int
    patch_frames: int = 1
    patch_height: int = 2
    patch_width: int = 2
    channels: int = 1

    def __post_init__(self) -> None:
        """Validate divisibility so unpatchify is always exact."""
        for name in (
            "frames",
            "height",
            "width",
            "patch_frames",
            "patch_height",
            "patch_width",
            "channels",
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
                    f"{axis}={extent} must be divisible by {patch}={size}; "
                    "pad or crop the latent at the data boundary instead of "
                    "silently truncating tokens"
                )

    @property
    def grid(self) -> tuple[int, int, int]:
        """Post-patch grid as ``(frames, height, width)``."""
        return (
            self.frames // self.patch_frames,
            self.height // self.patch_height,
            self.width // self.patch_width,
        )

    @property
    def num_tokens(self) -> int:
        """Sequence length produced by this layout."""
        frames, height, width = self.grid
        return frames * height * width

    @property
    def patch_dim(self) -> int:
        """Feature width of one packed patch."""
        return self.channels * self.patch_frames * self.patch_height * self.patch_width

    def is_temporal_only(self) -> bool:
        """Whether this layout describes a 1-D (audio-like) stream."""
        return self.height == 1 and self.width == 1

    @classmethod
    def temporal(cls, frames: int, channels: int, patch_frames: int = 1) -> PatchLayout:
        """Build a 1-D layout for a waveform-like latent stream.

        Args:
            frames: Latent frames.
            channels: Latent channels.
            patch_frames: Temporal patch size.

        Returns:
            A layout with unit spatial extent.
        """
        return cls(
            frames=frames,
            height=1,
            width=1,
            patch_frames=patch_frames,
            patch_height=1,
            patch_width=1,
            channels=channels,
        )

    @classmethod
    def empty(cls) -> PatchLayout:
        """Return the layout of an absent stream (one degenerate token).

        An absent stream still needs a valid layout so that shape arithmetic
        never special-cases ``None``; the accompanying mask is all-false, so no
        token contributes to attention or to the loss.

        Every patch size is one: the default 2x2 spatial patch would not divide
        a 1x1 grid, and a degenerate layout must still satisfy the same
        divisibility contract as a real one.
        """
        return cls(
            frames=1,
            height=1,
            width=1,
            patch_frames=1,
            patch_height=1,
            patch_width=1,
            channels=1,
        )


@dataclass(frozen=True, slots=True)
class TokenStream:
    """One modality's tokens, coordinates, mask, and noise state.

    All tensors are sequence-major: dimension 1 is the token axis and is the
    axis that context parallelism shards.

    Args:
        tokens: ``(batch, length, width)`` token features.
        coords: ``(batch, length, 3)`` physical coordinates as
            ``(seconds, row, column)``. Time is in seconds so streams at
            different frame rates share one phase space; space is in latent
            pixel units so a 480p and a 720p sample agree about what "one pixel
            apart" means.
        mask: ``(batch, length)`` token validity. False marks padding.
        noise_level: Noise level in ``[0, 1]``, either ``(batch,)`` for uniform
            per-sample noise or ``(batch, length)`` for exact per-token noise.
            Per-sample is the fast path and covers ordinary generation;
            per-token is what makes inpainting and continuation exact rather
            than mask-approximated.
        conditioned: ``(batch, length)`` marking tokens supplied as clean
            conditioning. These are attended to but excluded from the loss.
        layout: Static geometry for unpatchification.
    """

    tokens: torch.Tensor
    coords: torch.Tensor
    mask: torch.Tensor
    noise_level: torch.Tensor
    layout: PatchLayout
    conditioned: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.bool)
    )

    def __post_init__(self) -> None:
        """Materialise an omitted conditioning mask as all-false."""
        if self.conditioned.numel() == 0:
            object.__setattr__(
                self,
                "conditioned",
                torch.zeros(
                    self.tokens.shape[:2], dtype=torch.bool, device=self.tokens.device
                ),
            )

    @property
    def batch_size(self) -> int:
        """Number of samples."""
        return self.tokens.shape[0]

    @property
    def length(self) -> int:
        """Sequence length held by this rank.

        Under context parallelism this is the *local* shard length, not the
        global sequence length. Code that needs the global length must read it
        from ``layout.num_tokens``.
        """
        return self.tokens.shape[1]

    @property
    def width(self) -> int:
        """Token feature width."""
        return self.tokens.shape[2]

    @property
    def device(self) -> torch.device:
        """Device the tensors live on."""
        return self.tokens.device

    @property
    def is_empty(self) -> bool:
        """Whether the stream carries no valid tokens at all."""
        return self.tokens.shape[1] == 0 or not bool(self.mask.any())

    @property
    def per_token_noise(self) -> bool:
        """Whether the noise level varies within a sample."""
        return self.noise_level.ndim == 2

    def expanded_noise(self) -> torch.Tensor:
        """Return the noise level broadcast to ``(batch, length)``.

        Returns:
            A per-token noise tensor, materialised only when the stream is on
            the per-sample fast path.
        """
        if self.per_token_noise:
            return self.noise_level
        return self.noise_level[:, None].expand(self.tokens.shape[:2])

    def with_tokens(self, tokens: torch.Tensor) -> TokenStream:
        """Return a copy carrying different token features.

        Args:
            tokens: Replacement features, same batch and length.

        Returns:
            The updated stream.
        """
        return replace(self, tokens=tokens)

    def masked(self) -> torch.Tensor:
        """Return the tokens with invalid positions zeroed.

        Zeroing rather than trusting attention masks keeps padding from leaking
        into residual streams, normalisation statistics, or pooled outputs.

        Returns:
            The zero-padded token features.
        """
        return self.tokens * self.mask.unsqueeze(-1).to(self.tokens.dtype)

    def loss_mask(self) -> torch.Tensor:
        """Return the mask of tokens that contribute to the objective.

        A token counts if it is real (not padding) and was not handed to the
        model as clean conditioning. Supervising conditioning tokens teaches the
        model to reproduce its own input, which is the classic way an
        image-to-video model learns to output a still frame.

        Returns:
            ``(batch, length)`` boolean loss mask.
        """
        return self.mask & ~self.conditioned

    def validate(self) -> None:
        """Validate ranks, shapes, dtypes, and devices.

        Raises:
            ValueError: On a rank, shape, or device violation.
            TypeError: On a dtype violation.
        """
        if self.tokens.ndim != 3:
            raise ValueError(
                f"tokens must have rank 3 (batch, length, width); "
                f"got {tuple(self.tokens.shape)}"
            )
        batch, length, _ = tuple(self.tokens.shape)
        require_shape("coords", self.coords, (batch, length, 3))
        require_shape("mask", self.mask, (batch, length))
        require_shape("conditioned", self.conditioned, (batch, length))
        if self.noise_level.shape not in {(batch,), (batch, length)}:
            raise ValueError(
                f"noise_level must be ({batch},) or ({batch}, {length}); "
                f"got {tuple(self.noise_level.shape)}"
            )
        if not self.tokens.is_floating_point():
            raise TypeError(f"tokens must be floating point; got {self.tokens.dtype}")
        require_dtype("coords", self.coords, torch.float32)
        require_dtype("mask", self.mask, torch.bool)
        require_dtype("conditioned", self.conditioned, torch.bool)
        require_dtype("noise_level", self.noise_level, torch.float32)
        for name in ("coords", "mask", "conditioned", "noise_level"):
            tensor: torch.Tensor = getattr(self, name)
            if tensor.device != self.tokens.device:
                raise ValueError(
                    f"{name} must be on tokens device {self.tokens.device}; "
                    f"got {tensor.device}"
                )

    @classmethod
    def empty_like(
        cls,
        batch: int,
        width: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> TokenStream:
        """Return a zero-length stream standing in for an absent modality.

        A text-to-video model passes one of these for audio. Every audio code
        path then degenerates to a no-op without a single ``if audio is None``
        in the model.

        Args:
            batch: Batch size, which must still match the other streams.
            width: Token feature width.
            device: Device to allocate on.
            dtype: Token dtype.

        Returns:
            An empty stream.
        """
        target = torch.device(device)
        return cls(
            tokens=torch.zeros((batch, 0, width), dtype=dtype, device=target),
            coords=torch.zeros((batch, 0, 3), dtype=torch.float32, device=target),
            mask=torch.zeros((batch, 0), dtype=torch.bool, device=target),
            noise_level=torch.zeros((batch,), dtype=torch.float32, device=target),
            layout=PatchLayout.empty(),
            conditioned=torch.zeros((batch, 0), dtype=torch.bool, device=target),
        )


@dataclass(frozen=True, slots=True)
class TextContext:
    """Frozen text-encoder features used as cross-attention context.

    Kept separate from :class:`TokenStream` because text is never denoised: it
    carries no noise level, no coordinates, and no conditioning mask, and it is
    never sharded by context parallelism.

    Args:
        features: ``(batch, tokens, width)`` encoder hidden states.
        mask: ``(batch, tokens)`` token validity.
    """

    features: torch.Tensor
    mask: torch.Tensor

    @property
    def is_empty(self) -> bool:
        """Whether there is no usable text conditioning."""
        return self.features.shape[1] == 0

    def validate(self) -> None:
        """Validate rank, shape, dtype, and device.

        Raises:
            ValueError: On a rank, shape, or device violation.
            TypeError: On a dtype violation.
        """
        if self.features.ndim != 3:
            raise ValueError(
                f"text features must have rank 3; got {tuple(self.features.shape)}"
            )
        batch, tokens, _ = tuple(self.features.shape)
        require_shape("text mask", self.mask, (batch, tokens))
        require_dtype("text mask", self.mask, torch.bool)
        if self.mask.device != self.features.device:
            raise ValueError(
                f"text mask must be on features device {self.features.device}; "
                f"got {self.mask.device}"
            )

    def nullified(self) -> TextContext:
        """Return the unconditional context for classifier-free guidance.

        Zeroing both features and mask gives an exact, encoder-independent null
        that the model has actually been trained on, and costs nothing at
        inference time compared with re-encoding an empty string.

        Returns:
            The zeroed context.
        """
        return TextContext(
            features=torch.zeros_like(self.features),
            mask=torch.zeros_like(self.mask),
        )

    @classmethod
    def empty(
        cls,
        batch: int,
        width: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> TextContext:
        """Return an empty context for an unconditional model.

        Args:
            batch: Batch size.
            width: Encoder width.
            device: Device to allocate on.
            dtype: Feature dtype.

        Returns:
            An empty context.
        """
        target = torch.device(device)
        return cls(
            features=torch.zeros((batch, 0, width), dtype=dtype, device=target),
            mask=torch.zeros((batch, 0), dtype=torch.bool, device=target),
        )


def _flatten_stream(
    stream: TokenStream,
) -> tuple[list[torch.Tensor], PatchLayout]:
    return [
        stream.tokens,
        stream.coords,
        stream.mask,
        stream.noise_level,
        stream.conditioned,
    ], stream.layout


def _unflatten_stream(values: list[torch.Tensor], layout: PatchLayout) -> TokenStream:
    tokens, coords, mask, noise_level, conditioned = values
    return TokenStream(
        tokens=tokens,
        coords=coords,
        mask=mask,
        noise_level=noise_level,
        layout=layout,
        conditioned=conditioned,
    )


_pytree.register_pytree_node(
    TokenStream,
    _flatten_stream,
    _unflatten_stream,
    serialized_type_name="avgen.TokenStream",
)
_pytree.register_dataclass(TextContext, serialized_type_name="avgen.TextContext")
