"""Variable-length sequence packing with block-diagonal attention.

Bucketing (see :mod:`avgen.data.bucket`) makes a batch shape-homogeneous, but
within a bucket the *valid* lengths still vary: a 4.2-second clip and a
5.0-second clip both land in the five-second bucket, and the shorter one carries
16% padding. Across a realistic duration distribution that residual padding is
worth a large fraction of a training run.

Packing removes it. Several short sequences are concatenated into one sequence
of fixed capacity, and an attention mask keeps them from seeing each other.

**The throughput win.** Padding a batch costs the *maximum* length for every
sample; packing costs the *mean*. For a batch whose lengths are uniform on
``[L/4, L]`` that is a 37% saving on the token axis and more on attention, which
is quadratic. For the heavy-tailed duration distributions real corpora actually
have — a few long clips among many short ones — the saving is routinely over
half the compute. Packing is also what makes long clips affordable at all: a
batch of eight sequences padded to the longest is eight times the memory of one
packed sequence holding the same tokens.

**The correctness requirement, and it is absolute.** Two clips that share a
packed sequence must not attend to each other. If they do, the model learns to
predict frames of clip A from frames of clip B, which is both a nonsense task
and a training-time information leak that inflates every metric. Three things
have to hold together:

1. **Attention is block-diagonal.** A query in segment *i* attends only to keys
   in segment *i*. :meth:`PackedLayout.attention_mask` builds the dense form;
   :attr:`PackedLayout.cu_seqlens` is the same information in the cumulative
   form that variable-length fused attention kernels take.
2. **Noise levels become per-token.** Each packed clip is denoised at its own
   timestep, so a per-sample noise level is no longer expressible. This is not
   an implementation detail to be papered over — a packed stream that carries a
   single noise level is silently training every segment at the first segment's
   timestep.
3. **The loss normalises per segment.** Summing over the packed sequence and
   dividing by its length weights a long clip more than a short one. Whether
   that is wanted is a modelling choice, but it must be a *choice*;
   :meth:`PackedLayout.segment_weights` provides the per-token weights that make
   every segment count equally.

**The known wart.** :class:`~avgen.core.tokens.TokenStream` carries one static
:class:`~avgen.core.tokens.PatchLayout`, and a packed stream contains several.
The packed stream's ``layout`` describes its first segment only; the full set
lives in :attr:`PackedLayout.layouts`, and unpacking must go through
:func:`split_packed` rather than through ``unpatchify_grid`` on the whole
sequence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from avgen.core._validate import require_positive
from avgen.core.tokens import PatchLayout, TokenStream

__all__ = [
    "PackPlan",
    "PackedLayout",
    "pack_streams",
    "plan_packing",
    "split_packed",
]

#: Marks a position that belongs to no segment. Negative so it can never be
#: confused with a real segment index, and so an equality test against it is a
#: single comparison rather than a mask lookup.
PADDING_SEGMENT: int = -1


@dataclass(frozen=True, slots=True)
class PackedLayout:
    """Segment structure of one packed sequence.

    Args:
        lengths: Token count of each segment, in packed order.
        capacity: Total length of the packed sequence, including trailing
            padding.
        sample_indices: Index of each segment's source sample, so a packed
            sequence can be traced back to the clips it holds.
        layouts: Patch geometry of each segment, needed to unpatchify it.

    Raises:
        ValueError: If there are no segments, a length is non-positive, the
            segments overflow the capacity, or the metadata tuples disagree in
            length.
    """

    lengths: tuple[int, ...]
    capacity: int
    sample_indices: tuple[int, ...] = ()
    layouts: tuple[PatchLayout, ...] = ()

    def __post_init__(self) -> None:
        """Validate the segment structure against the capacity."""
        require_positive("capacity", self.capacity)
        if not self.lengths:
            raise ValueError("a packed layout requires at least one segment")
        for index, length in enumerate(self.lengths):
            require_positive(f"lengths[{index}]", length)
        total = sum(self.lengths)
        if total > self.capacity:
            raise ValueError(
                f"segments total {total} tokens but the capacity is "
                f"{self.capacity}; plan_packing is what guarantees they fit"
            )
        for name in ("sample_indices", "layouts"):
            values = getattr(self, name)
            if values and len(values) != len(self.lengths):
                raise ValueError(
                    f"{name} must have one entry per segment; got {len(values)} "
                    f"for {len(self.lengths)} segments"
                )

    @property
    def segment_count(self) -> int:
        """Number of independent sequences packed together."""
        return len(self.lengths)

    @property
    def offsets(self) -> tuple[int, ...]:
        """Start position of each segment within the packed sequence."""
        running = 0
        starts: list[int] = []
        for length in self.lengths:
            starts.append(running)
            running += length
        return tuple(starts)

    @property
    def occupancy(self) -> float:
        """Fraction of the capacity carrying real tokens.

        This is the number to watch when tuning a packing strategy. Occupancy
        below roughly 0.9 means the bin-packing is leaving compute on the table;
        occupancy at 1.0 with many segments means the capacity is too small for
        the corpus's long tail and long clips are being excluded.
        """
        return sum(self.lengths) / self.capacity

    def segment_ids(
        self, *, device: torch.device | str = "cpu"
    ) -> torch.Tensor:
        """Return the per-token segment index, with padding marked.

        Args:
            device: Device to allocate on.

        Returns:
            ``(capacity,)`` int32 tensor; :data:`PADDING_SEGMENT` at unused
            positions.
        """
        ids = torch.full(
            (self.capacity,), PADDING_SEGMENT, dtype=torch.int32, device=device
        )
        for segment, (start, length) in enumerate(
            zip(self.offsets, self.lengths, strict=True)
        ):
            ids[start : start + length] = segment
        return ids

    @property
    def cu_seqlens(self) -> torch.Tensor:
        """Cumulative segment boundaries, the varlen-attention calling convention.

        Fused variable-length attention kernels take exactly this: an int32
        tensor of length ``segment_count + 1`` whose consecutive entries bound
        each segment. Passing it instead of a dense mask is what keeps packing a
        throughput win — a dense ``capacity x capacity`` mask is quadratic
        memory, which is the cost packing exists to avoid.

        Returns:
            ``(segment_count + 1,)`` int32 tensor on CPU.
        """
        bounds = [0]
        for length in self.lengths:
            bounds.append(bounds[-1] + length)
        return torch.tensor(bounds, dtype=torch.int32)

    def attention_mask(
        self, *, device: torch.device | str = "cpu"
    ) -> torch.Tensor:
        """Return the dense block-diagonal mask.

        Use this for a reference implementation, a test, or an attention backend
        with no varlen path. Prefer :attr:`cu_seqlens` in production: this
        tensor is quadratic in the capacity, so at a 64k capacity it is four
        gigabytes on its own.

        Args:
            device: Device to allocate on.

        Returns:
            ``(capacity, capacity)`` bool tensor; True where attention is
            permitted. Padding positions attend to nothing and are attended to
            by nothing.
        """
        ids = self.segment_ids(device=device)
        valid = ids != PADDING_SEGMENT
        same = ids.unsqueeze(1) == ids.unsqueeze(0)
        return same & valid.unsqueeze(1) & valid.unsqueeze(0)

    def segment_weights(
        self, *, device: torch.device | str = "cpu"
    ) -> torch.Tensor:
        """Return per-token loss weights that make every segment count equally.

        Each token in a segment of length ``n`` gets weight ``1/n``, so the
        weights of a segment sum to one regardless of its length. Without this,
        a packed sequence holding one 8-second clip and four 2-second clips
        gives the long clip half the gradient, purely because of how the bin
        packer happened to fill the bin — a data-dependent loss weighting that
        changes when the shuffle changes.

        Args:
            device: Device to allocate on.

        Returns:
            ``(capacity,)`` float32 weights; zero at padding positions.
        """
        weights = torch.zeros(self.capacity, dtype=torch.float32, device=device)
        for start, length in zip(self.offsets, self.lengths, strict=True):
            weights[start : start + length] = 1.0 / length
        return weights

    def valid_mask(self, *, device: torch.device | str = "cpu") -> torch.Tensor:
        """Return the non-padding mask.

        Args:
            device: Device to allocate on.

        Returns:
            ``(capacity,)`` bool tensor, True at real tokens.
        """
        return self.segment_ids(device=device) != PADDING_SEGMENT


@dataclass(frozen=True, slots=True)
class PackPlan:
    """An assignment of samples to packed sequences.

    Args:
        bins: Each entry lists the sample indices packed into one sequence, in
            packed order.
        lengths: Token count of every sample, indexed globally.
        capacity: Token budget of one packed sequence.

    Raises:
        ValueError: If there are no bins, or a bin overflows the capacity.
    """

    bins: tuple[tuple[int, ...], ...]
    lengths: tuple[int, ...]
    capacity: int

    def __post_init__(self) -> None:
        """Validate that every bin fits."""
        require_positive("capacity", self.capacity)
        if not self.bins:
            raise ValueError("a pack plan requires at least one bin")
        for position, contents in enumerate(self.bins):
            total = sum(self.lengths[index] for index in contents)
            if total > self.capacity:
                raise ValueError(
                    f"bin {position} holds {total} tokens, over the capacity of "
                    f"{self.capacity}"
                )

    def __len__(self) -> int:
        """Return the number of packed sequences."""
        return len(self.bins)

    @property
    def occupancy(self) -> float:
        """Fraction of the planned capacity carrying real tokens.

        The single number that says whether packing is worth its complexity for
        a given corpus and capacity. Compare it against the occupancy of plain
        padding, which is ``mean(length) / max(length)`` within a batch.
        """
        used = sum(self.lengths[index] for contents in self.bins for index in contents)
        return used / (len(self.bins) * self.capacity)

    def layout_for(
        self,
        bin_index: int,
        *,
        layouts: Sequence[PatchLayout] | None = None,
    ) -> PackedLayout:
        """Return the segment structure of one bin.

        Args:
            bin_index: Which packed sequence to describe.
            layouts: Per-sample patch geometry, indexed globally. Optional; a
                layout is only needed to unpatchify.

        Returns:
            The packed layout.

        Raises:
            IndexError: If ``bin_index`` is out of range.
        """
        contents = self.bins[bin_index]
        return PackedLayout(
            lengths=tuple(self.lengths[index] for index in contents),
            capacity=self.capacity,
            sample_indices=contents,
            layouts=(
                tuple(layouts[index] for index in contents)
                if layouts is not None
                else ()
            ),
        )


def plan_packing(
    lengths: Sequence[int],
    *,
    capacity: int,
    strategy: str = "first_fit_decreasing",
) -> PackPlan:
    """Assign samples to packed sequences under a token budget.

    Two strategies, both deterministic:

    ``first_fit_decreasing``
        Sort by length descending, then place each sample in the first bin it
        fits. The classic bin-packing heuristic; provably within 11/9 of optimal
        plus a constant, and in practice within a percent or two on realistic
        length distributions. It reorders samples, which is fine because the
        shuffle upstream already made the order arbitrary.

    ``sequential``
        Fill bins in the given order, opening a new one when the next sample
        does not fit. Worse occupancy, but it preserves order, which matters
        when the caller has deliberately ordered samples — a curriculum within
        an epoch, or a deterministic replay being compared against a recording.

    Args:
        lengths: Token count of each sample.
        capacity: Token budget per packed sequence.
        strategy: ``"first_fit_decreasing"`` or ``"sequential"``.

    Returns:
        The plan.

    Raises:
        ValueError: If ``lengths`` is empty, a length is non-positive, a single
            sample exceeds the capacity, or the strategy is unknown.
    """
    require_positive("capacity", capacity)
    if not lengths:
        raise ValueError("plan_packing requires at least one sample")
    for index, length in enumerate(lengths):
        require_positive(f"lengths[{index}]", length)
        if length > capacity:
            raise ValueError(
                f"sample {index} needs {length} tokens but the packing capacity is "
                f"{capacity}; a sequence that does not fit in one bin cannot be "
                "packed at all — raise the capacity, shorten the clip, or shard "
                "the sequence with context parallelism instead"
            )
    if strategy == "first_fit_decreasing":
        # Descending length, index ascending on ties: the tiebreak keeps the
        # plan a pure function of the length vector, so two ranks planning the
        # same epoch produce the same bins.
        order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    elif strategy == "sequential":
        order = list(range(len(lengths)))
    else:
        raise ValueError(
            f"unknown packing strategy {strategy!r}; expected "
            "'first_fit_decreasing' or 'sequential'"
        )

    bins: list[list[int]] = []
    remaining: list[int] = []
    for index in order:
        length = lengths[index]
        placed = False
        for position, free in enumerate(remaining):
            if free >= length:
                bins[position].append(index)
                remaining[position] = free - length
                placed = True
                break
        if not placed:
            bins.append([index])
            remaining.append(capacity - length)
    return PackPlan(
        bins=tuple(tuple(contents) for contents in bins),
        lengths=tuple(lengths),
        capacity=capacity,
    )


def pack_streams(
    streams: Sequence[TokenStream],
    *,
    capacity: int,
    sample_indices: Sequence[int] | None = None,
) -> tuple[TokenStream, PackedLayout]:
    """Concatenate single-sample token streams into one packed stream.

    Every input must have batch size one: packing operates on individual
    sequences, and a batched input would be ambiguous about whether the batch
    axis or the token axis is being merged.

    The output's noise level is always per-token, never per-sample. That is
    forced, not a preference — each packed segment carries its own timestep, and
    a per-sample noise level cannot express that.

    Args:
        streams: Streams to pack, each with batch size one.
        capacity: Token budget of the packed sequence. Trailing positions are
            padded and masked out.
        sample_indices: Optional source sample index of each stream, recorded in
            the returned layout for traceability.

    Returns:
        The packed stream, shaped ``(1, capacity, width)``, and its layout.

    Raises:
        ValueError: If the sequence is empty, a stream is batched, widths or
            dtypes or devices disagree, or the segments overflow the capacity.
    """
    require_positive("capacity", capacity)
    if not streams:
        raise ValueError("pack_streams requires at least one stream")
    head = streams[0]
    for position, stream in enumerate(streams):
        if stream.batch_size != 1:
            raise ValueError(
                f"streams[{position}] has batch size {stream.batch_size}; packing "
                "operates on single sequences, so split the batch first"
            )
        if stream.width != head.width:
            raise ValueError(
                f"streams[{position}] has width {stream.width}, expected "
                f"{head.width}; packing concatenates along the token axis and "
                "cannot reconcile two feature widths"
            )
        if stream.tokens.dtype is not head.tokens.dtype:
            raise ValueError(
                f"streams[{position}] has dtype {stream.tokens.dtype}, expected "
                f"{head.tokens.dtype}"
            )
        if stream.device != head.device:
            raise ValueError(
                f"streams[{position}] is on {stream.device}, expected {head.device}"
            )
        if stream.length == 0:
            raise ValueError(f"streams[{position}] is empty and cannot be packed")

    lengths = tuple(stream.length for stream in streams)
    total = sum(lengths)
    if total > capacity:
        raise ValueError(
            f"streams total {total} tokens but the capacity is {capacity}; call "
            "plan_packing first so the bins are known to fit"
        )

    device = head.device
    width = head.width
    tokens = torch.zeros((1, capacity, width), dtype=head.tokens.dtype, device=device)
    coords = torch.zeros((1, capacity, 3), dtype=torch.float32, device=device)
    mask = torch.zeros((1, capacity), dtype=torch.bool, device=device)
    conditioned = torch.zeros((1, capacity), dtype=torch.bool, device=device)
    noise = torch.zeros((1, capacity), dtype=torch.float32, device=device)

    start = 0
    for stream in streams:
        stop = start + stream.length
        tokens[:, start:stop] = stream.tokens
        coords[:, start:stop] = stream.coords
        mask[:, start:stop] = stream.mask
        conditioned[:, start:stop] = stream.conditioned
        # expanded_noise materialises the per-sample fast path into per-token
        # form; without it a per-sample stream would broadcast its single
        # timestep across the whole packed sequence.
        noise[:, start:stop] = stream.expanded_noise()
        start = stop

    layout = PackedLayout(
        lengths=lengths,
        capacity=capacity,
        sample_indices=tuple(sample_indices) if sample_indices is not None else (),
        layouts=tuple(stream.layout for stream in streams),
    )
    packed = TokenStream(
        tokens=tokens,
        coords=coords,
        mask=mask,
        noise_level=noise,
        # Describes the FIRST segment only. See the module docstring: a packed
        # stream has no single patch geometry, and unpatchifying it as a whole
        # would silently reinterpret every segment after the first.
        layout=head.layout,
        conditioned=conditioned,
    )
    return packed, layout


def split_packed(
    tensor: torch.Tensor,
    layout: PackedLayout,
) -> tuple[torch.Tensor, ...]:
    """Split a packed-sequence tensor back into its segments.

    Args:
        tensor: ``(batch, capacity, ...)`` tensor whose second axis is the
            packed token axis. Model outputs and packed streams both qualify.
        layout: The layout the tensor was packed with.

    Returns:
        One view per segment, trailing padding excluded.

    Raises:
        ValueError: If the tensor's token axis does not match the capacity.
    """
    if tensor.ndim < 2 or tensor.shape[1] != layout.capacity:
        raise ValueError(
            f"expected a tensor whose second axis is the capacity {layout.capacity}; "
            f"got shape {tuple(tensor.shape)}"
        )
    return tuple(
        tensor[:, start : start + length]
        for start, length in zip(layout.offsets, layout.lengths, strict=True)
    )
