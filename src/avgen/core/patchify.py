"""Conversion between dense latent grids and token sequences.

This module owns the only two places in the framework where a dense grid and a
token sequence meet. Everything upstream (codecs, shard files, dataloaders)
speaks grids; everything downstream (models, parallelism, attention) speaks
sequences.

The patchifier also builds the coordinate tensor, and that is the part worth
reading carefully. Coordinates are physical, not positional:

* **Time is seconds.** A video latent at 6.25 frames/second and an audio latent
  at 43 frames/second have no shared index space. Carrying seconds means one
  rotary embedding puts both in the same phase space, an 8-second clip and a
  2-second clip agree about what "one second later" means, and a model trained
  at 24 fps can be fine-tuned at 30 fps without relearning its time axis.
* **Space is latent pixels, optionally normalised.** Absolute pixel coordinates
  let a model trained at 256px extrapolate to 512px; normalised coordinates make
  aspect-ratio changes benign. Both are supported and the choice is explicit.

Padding tokens receive coordinate zero and a false mask. They are never attended
to and never contribute to a loss, so their coordinate value is irrelevant — but
it is fixed rather than uninitialised so that two runs with different padding
produce bit-identical results for the real tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from avgen.core.tokens import PatchLayout, TokenStream

__all__ = [
    "GridPatchifier",
    "Patchifier",
    "build_spatial_coords",
    "build_temporal_coords",
    "patchify_grid",
    "unpatchify_grid",
]


def build_temporal_coords(
    positions: torch.Tensor,
    layout: PatchLayout,
    *,
    normalize_space: bool = False,
) -> torch.Tensor:
    """Build per-token ``(seconds, row, column)`` coordinates for a grid.

    Args:
        positions: ``(batch, frames)`` physical time in seconds for each latent
            frame, *before* temporal patching.
        layout: Grid and patch geometry.
        normalize_space: When true, spatial coordinates are scaled into
            ``[0, 1]`` by the grid extent. Use for models that must generalise
            across aspect ratios; leave false for models that should extrapolate
            to higher resolution at a fixed pixel pitch.

    Returns:
        ``(batch, num_tokens, 3)`` float32 coordinates.

    Raises:
        ValueError: If ``positions`` does not match the layout's frame count.
    """
    batch, frames = tuple(positions.shape)
    if frames != layout.frames:
        raise ValueError(
            f"positions has {frames} frames but layout declares {layout.frames}"
        )
    device = positions.device
    grid_frames, grid_height, grid_width = layout.grid

    # A temporal patch spans several latent frames; its coordinate is the mean
    # of the frames it covers, which keeps the mapping exact for patch size 1
    # and centred for larger patches.
    folded = positions.reshape(batch, grid_frames, layout.patch_frames)
    time = folded.mean(dim=2)

    rows = torch.arange(grid_height, device=device, dtype=torch.float32)
    columns = torch.arange(grid_width, device=device, dtype=torch.float32)
    if normalize_space:
        rows = rows / max(grid_height - 1, 1)
        columns = columns / max(grid_width - 1, 1)
    else:
        # Scale by patch size so a coordinate step of one means one latent
        # pixel regardless of patch size; a model fine-tuned with a different
        # patch size then keeps the same spatial frequency response.
        rows = rows * layout.patch_height
        columns = columns * layout.patch_width

    shape = (batch, grid_frames, grid_height, grid_width)
    time_grid = time[:, :, None, None].expand(shape)
    row_grid = rows[None, None, :, None].expand(shape)
    column_grid = columns[None, None, None, :].expand(shape)
    return torch.stack(
        (
            time_grid.reshape(batch, -1),
            row_grid.reshape(batch, -1),
            column_grid.reshape(batch, -1),
        ),
        dim=-1,
    ).to(torch.float32)


def build_spatial_coords(
    layout: PatchLayout,
    *,
    batch: int,
    fps: float,
    device: torch.device | str = "cpu",
    start_seconds: float = 0.0,
) -> torch.Tensor:
    """Build coordinates for a uniformly sampled grid.

    A convenience wrapper for the common case where frame times are a regular
    sequence rather than an arbitrary per-sample schedule.

    Args:
        layout: Grid and patch geometry.
        batch: Batch size.
        fps: Latent frames per second.
        device: Device to allocate on.
        start_seconds: Time of the first latent frame.

    Returns:
        ``(batch, num_tokens, 3)`` float32 coordinates.

    Raises:
        ValueError: If ``fps`` is not positive.
    """
    if fps <= 0.0:
        raise ValueError(f"fps must be positive; got {fps!r}")
    frames = torch.arange(layout.frames, device=device, dtype=torch.float32)
    positions = (start_seconds + frames / fps)[None].expand(batch, layout.frames)
    return build_temporal_coords(positions, layout)


def patchify_grid(grid: torch.Tensor, layout: PatchLayout) -> torch.Tensor:
    """Fold a dense latent grid into a token sequence.

    Args:
        grid: ``(batch, channels, frames, height, width)`` latents.
        layout: Geometry describing ``grid``.

    Returns:
        ``(batch, num_tokens, patch_dim)`` packed patches, ordered
        time-major then row-major then column-major. That ordering matters:
        every coordinate builder, every unpatchify, and every context-parallel
        shard boundary assumes it.

    Raises:
        ValueError: If ``grid`` does not match the layout.
    """
    expected = (
        grid.shape[0],
        layout.channels,
        layout.frames,
        layout.height,
        layout.width,
    )
    if tuple(grid.shape) != expected:
        raise ValueError(f"grid shape must be {expected}; got {tuple(grid.shape)}")
    batch = grid.shape[0]
    grid_frames, grid_height, grid_width = layout.grid
    reshaped = grid.reshape(
        batch,
        layout.channels,
        grid_frames,
        layout.patch_frames,
        grid_height,
        layout.patch_height,
        grid_width,
        layout.patch_width,
    )
    # (b, c, gf, pf, gh, ph, gw, pw) -> (b, gf, gh, gw, c, pf, ph, pw)
    permuted = reshaped.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return permuted.reshape(batch, layout.num_tokens, layout.patch_dim)


def unpatchify_grid(tokens: torch.Tensor, layout: PatchLayout) -> torch.Tensor:
    """Unfold a token sequence back into a dense latent grid.

    Exactly inverts :func:`patchify_grid`.

    Args:
        tokens: ``(batch, num_tokens, patch_dim)`` packed patches.
        layout: Geometry to restore.

    Returns:
        ``(batch, channels, frames, height, width)`` latents.

    Raises:
        ValueError: If ``tokens`` does not match the layout.
    """
    expected = (tokens.shape[0], layout.num_tokens, layout.patch_dim)
    if tuple(tokens.shape) != expected:
        raise ValueError(f"tokens shape must be {expected}; got {tuple(tokens.shape)}")
    batch = tokens.shape[0]
    grid_frames, grid_height, grid_width = layout.grid
    reshaped = tokens.reshape(
        batch,
        grid_frames,
        grid_height,
        grid_width,
        layout.channels,
        layout.patch_frames,
        layout.patch_height,
        layout.patch_width,
    )
    # (b, gf, gh, gw, c, pf, ph, pw) -> (b, c, gf, pf, gh, ph, gw, pw)
    permuted = reshaped.permute(0, 4, 1, 5, 2, 6, 3, 7)
    return permuted.reshape(
        batch, layout.channels, layout.frames, layout.height, layout.width
    )


@runtime_checkable
class Patchifier(Protocol):
    """Converts between dense latent grids and token streams.

    Implement this to change how a model consumes latents — a different patch
    size, a learned tokenizer, a wavelet decomposition — without touching the
    model, the objective, or the parallelism layer.
    """

    def to_tokens(
        self,
        grid: torch.Tensor,
        *,
        positions: torch.Tensor,
        mask: torch.Tensor,
        noise_level: torch.Tensor,
        conditioned: torch.Tensor | None = None,
    ) -> TokenStream:
        """Convert a dense grid and its metadata into a token stream."""
        ...

    def to_grid(self, stream: TokenStream) -> torch.Tensor:
        """Convert a token stream back into a dense grid."""
        ...

    def layout_for(self, grid_shape: tuple[int, ...]) -> PatchLayout:
        """Return the layout this patchifier would produce for a grid shape."""
        ...


@dataclass(frozen=True, slots=True)
class GridPatchifier:
    """The default patchifier: fixed-size non-overlapping patches.

    This is the "pixel-shuffle" patchification used by essentially every
    production video DiT. It is parameter-free, exactly invertible, and cheap,
    which is what you want at the boundary of a hot loop.

    Args:
        patch_frames: Temporal patch size. Left at 1 by default because modern
            video VAEs already compress time 4-8x, and compressing it further
            in the transformer costs temporal detail that cannot be recovered.
        patch_height: Spatial patch height.
        patch_width: Spatial patch width.
        normalize_space: Whether spatial coordinates are normalised to
            ``[0, 1]`` rather than kept in latent pixels.
    """

    patch_frames: int = 1
    patch_height: int = 2
    patch_width: int = 2
    normalize_space: bool = False

    def layout_for(self, grid_shape: tuple[int, ...]) -> PatchLayout:
        """Return the layout produced for a grid of the given shape.

        Args:
            grid_shape: ``(batch, channels, frames, height, width)``.

        Returns:
            The corresponding layout.

        Raises:
            ValueError: If the shape is not rank 5.
        """
        if len(grid_shape) != 5:
            raise ValueError(
                f"grid shape must be rank 5 (batch, channels, frames, height, "
                f"width); got {grid_shape}"
            )
        _, channels, frames, height, width = grid_shape
        return PatchLayout(
            frames=frames,
            height=height,
            width=width,
            patch_frames=self.patch_frames,
            patch_height=self.patch_height,
            patch_width=self.patch_width,
            channels=channels,
        )

    def to_tokens(
        self,
        grid: torch.Tensor,
        *,
        positions: torch.Tensor,
        mask: torch.Tensor,
        noise_level: torch.Tensor,
        conditioned: torch.Tensor | None = None,
    ) -> TokenStream:
        """Convert a dense grid and its metadata into a token stream.

        Masks are reduced over each patch with ``any`` rather than ``all``: a
        patch that contains even one real latent is a real token, because
        discarding it would lose signal at the padded edge of a bucketed sample.
        Conditioning masks reduce with ``all``, because a patch is only a clean
        anchor if every latent inside it is clean.

        Args:
            grid: ``(batch, channels, frames, height, width)`` latents.
            positions: ``(batch, frames)`` physical time in seconds.
            mask: ``(batch, frames, height, width)`` latent validity.
            noise_level: ``(batch,)`` or ``(batch, frames, height, width)``
                noise level.
            conditioned: Optional clean-anchor mask shaped like ``mask``.

        Returns:
            The token stream.

        Raises:
            ValueError: If any input disagrees with the derived layout.
        """
        layout = self.layout_for(tuple(grid.shape))
        batch = grid.shape[0]
        tokens = patchify_grid(grid, layout)
        coords = build_temporal_coords(
            positions, layout, normalize_space=self.normalize_space
        )
        token_mask = self._reduce_mask(mask, layout, batch, reduce_all=False)
        token_conditioned = (
            self._reduce_mask(conditioned, layout, batch, reduce_all=True)
            if conditioned is not None
            else torch.zeros(
                (batch, layout.num_tokens), dtype=torch.bool, device=grid.device
            )
        )
        if noise_level.ndim == 1:
            token_noise = noise_level
        else:
            token_noise = self._reduce_noise(noise_level, layout, batch)
        return TokenStream(
            tokens=tokens,
            coords=coords,
            mask=token_mask,
            noise_level=token_noise,
            layout=layout,
            conditioned=token_conditioned,
        )

    def to_grid(self, stream: TokenStream) -> torch.Tensor:
        """Convert a token stream back into a dense grid.

        Args:
            stream: The stream to unfold.

        Returns:
            ``(batch, channels, frames, height, width)`` latents.
        """
        return unpatchify_grid(stream.tokens, stream.layout)

    @staticmethod
    def _fold(
        values: torch.Tensor,
        layout: PatchLayout,
        batch: int,
    ) -> torch.Tensor:
        """Reshape ``(batch, frames, height, width)`` into per-patch groups."""
        grid_frames, grid_height, grid_width = layout.grid
        return (
            values.reshape(
                batch,
                grid_frames,
                layout.patch_frames,
                grid_height,
                layout.patch_height,
                grid_width,
                layout.patch_width,
            )
            .permute(0, 1, 3, 5, 2, 4, 6)
            .reshape(batch, layout.num_tokens, -1)
        )

    @classmethod
    def _reduce_mask(
        cls,
        values: torch.Tensor,
        layout: PatchLayout,
        batch: int,
        *,
        reduce_all: bool,
    ) -> torch.Tensor:
        expected = (batch, layout.frames, layout.height, layout.width)
        if tuple(values.shape) != expected:
            raise ValueError(
                f"mask shape must be {expected}; got {tuple(values.shape)}"
            )
        folded = cls._fold(values, layout, batch)
        return folded.all(dim=-1) if reduce_all else folded.any(dim=-1)

    @classmethod
    def _reduce_noise(
        cls,
        values: torch.Tensor,
        layout: PatchLayout,
        batch: int,
    ) -> torch.Tensor:
        expected = (batch, layout.frames, layout.height, layout.width)
        if tuple(values.shape) != expected:
            raise ValueError(
                f"per-token noise level shape must be {expected}; "
                f"got {tuple(values.shape)}"
            )
        return cls._fold(values, layout, batch).mean(dim=-1).to(torch.float32)
