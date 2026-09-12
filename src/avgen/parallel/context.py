"""Context parallelism: sharding the token sequence across ranks.

This is the axis that decides whether a video model is trainable at all.

A 10-second clip at 720p through a 8x8x4-compressing VAE with 2x2 patching is
roughly 100,000 tokens. Attention is quadratic in that; activation memory is
linear in it and multiplied by depth. No amount of parameter sharding helps,
because the problem is not the parameters. Context parallelism splits the
sequence itself: with ``cp=8`` each rank holds 12,500 tokens and attention is
computed collaboratively across the group.

**Diffusion transformers get an advantage here that language models do not.**
Causal attention creates severe load imbalance under naive contiguous sharding —
the rank holding the last chunk attends over the whole sequence while the rank
holding the first attends over almost nothing — which is why LLM stacks need
zigzag or striped assignment to rebalance. Video DiTs use *bidirectional*
attention: every query attends to every key regardless of position, so every
shard does exactly the same work. Contiguous sharding is therefore both optimal
and simplest, and it preserves spatial locality, which keeps rotary embeddings
and any windowed attention pattern cheap to compute locally.

Two mechanisms are provided:

* :func:`context_parallel_region` wraps PyTorch's native context-parallel SDPA,
  which implements ring attention over the sequence shards. Use it when the
  model's attention is standard SDPA.
* :func:`shard_stream` / :func:`gather_stream` shard the avgen token contracts
  themselves, for the parts of a step that live outside attention — the
  objective, the loss reduction, the sampler.

The load-balance check is not decoration. An imbalanced shard silently makes
every rank wait for the slowest one, and it shows up as "our scaling efficiency
is 60% and we do not know why".
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import replace

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from avgen.core.tokens import TokenStream

__all__ = [
    "context_parallel_region",
    "gather_stream",
    "gather_tokens",
    "pad_to_multiple",
    "shard_stream",
    "sharded_length",
]


def sharded_length(total: int, shards: int) -> int:
    """Return the per-shard sequence length, requiring an exact split.

    Args:
        total: Global sequence length.
        shards: Number of context-parallel ranks.

    Returns:
        Tokens per shard.

    Raises:
        ValueError: If the sequence does not divide evenly. Ragged shards are
            rejected rather than padded implicitly, because an implicit pad
            changes the token count that the loss normalises by and makes two
            otherwise-identical runs disagree.
    """
    if total % shards != 0:
        raise ValueError(
            f"sequence length {total} is not divisible by context-parallel "
            f"degree {shards}; pad the sequence with pad_to_multiple() at the "
            "data boundary so the padding is explicit and masked"
        )
    return total // shards


def pad_to_multiple(stream: TokenStream, multiple: int) -> TokenStream:
    """Right-pad a stream so its length divides evenly.

    Padding is appended with a false mask, so padded tokens are excluded from
    attention and from the loss. Their coordinates are zero and their features
    are zero, which keeps the result bit-reproducible.

    Args:
        stream: The stream to pad.
        multiple: Required divisor of the resulting length.

    Returns:
        The padded stream, or the original when it already divides evenly.

    Raises:
        ValueError: If ``multiple`` is not positive.
    """
    if multiple < 1:
        raise ValueError(f"multiple must be positive; got {multiple!r}")
    length = stream.length
    remainder = length % multiple
    if remainder == 0:
        return stream
    pad = multiple - remainder
    batch = stream.batch_size

    def _pad(tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
        shape = list(tensor.shape)
        shape[dim] = pad
        return torch.cat(
            (tensor, torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)),
            dim=dim,
        )

    noise = stream.noise_level
    if stream.per_token_noise:
        noise = _pad(noise)
    return TokenStream(
        tokens=_pad(stream.tokens),
        coords=_pad(stream.coords),
        mask=torch.cat(
            (
                stream.mask,
                torch.zeros((batch, pad), dtype=torch.bool, device=stream.device),
            ),
            dim=1,
        ),
        noise_level=noise,
        layout=stream.layout,
        conditioned=torch.cat(
            (
                stream.conditioned,
                torch.zeros((batch, pad), dtype=torch.bool, device=stream.device),
            ),
            dim=1,
        ),
    )


def shard_stream(stream: TokenStream, mesh: DeviceMesh | None) -> TokenStream:
    """Take this rank's contiguous slice of a token stream.

    The stream's ``layout`` is preserved untouched, so the global geometry
    remains known even though this rank holds only part of the sequence. That is
    what lets :func:`gather_stream` and the unpatchifier reassemble the result
    without a separate bookkeeping structure.

    Args:
        stream: The full-sequence stream.
        mesh: The ``cp`` sub-mesh, or ``None`` to return the stream unchanged.

    Returns:
        This rank's shard.
    """
    if mesh is None or mesh.size() == 1:
        return stream
    shards = mesh.size()
    index = mesh.get_local_rank()
    local = sharded_length(stream.length, shards)
    start = index * local
    stop = start + local

    noise = stream.noise_level
    if stream.per_token_noise:
        noise = noise[:, start:stop]
    return replace(
        stream,
        tokens=stream.tokens[:, start:stop],
        coords=stream.coords[:, start:stop],
        mask=stream.mask[:, start:stop],
        noise_level=noise,
        conditioned=stream.conditioned[:, start:stop],
    )


def gather_stream(stream: TokenStream, mesh: DeviceMesh | None) -> TokenStream:
    """Reassemble a sharded stream onto every rank.

    Args:
        stream: This rank's shard.
        mesh: The ``cp`` sub-mesh, or ``None`` to return the stream unchanged.

    Returns:
        The full-sequence stream.
    """
    if mesh is None or mesh.size() == 1:
        return stream
    if not (dist.is_available() and dist.is_initialized()):
        return stream
    group = mesh.get_group()
    shards = mesh.size()

    def _gather(tensor: torch.Tensor) -> torch.Tensor:
        parts = [torch.empty_like(tensor) for _ in range(shards)]
        dist.all_gather(parts, tensor.contiguous(), group=group)
        return torch.cat(parts, dim=1)

    noise = stream.noise_level
    if stream.per_token_noise:
        noise = _gather(noise)
    return replace(
        stream,
        tokens=_gather(stream.tokens),
        coords=_gather(stream.coords),
        mask=_gather(stream.mask),
        noise_level=noise,
        conditioned=_gather(stream.conditioned),
    )


def gather_tokens(tensor: torch.Tensor, mesh: DeviceMesh | None) -> torch.Tensor:
    """All-gather a sequence-sharded tensor along dimension one.

    Args:
        tensor: ``(batch, local_length, ...)`` shard.
        mesh: The ``cp`` sub-mesh, or ``None``.

    Returns:
        The full-sequence tensor.
    """
    if mesh is None or mesh.size() == 1:
        return tensor
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    parts = [torch.empty_like(tensor) for _ in range(mesh.size())]
    dist.all_gather(parts, tensor.contiguous(), group=mesh.get_group())
    return torch.cat(parts, dim=1)


@contextmanager
def context_parallel_region(
    mesh: DeviceMesh | None,
    *,
    buffers: Sequence[torch.Tensor] | None = None,
    buffer_seq_dims: Sequence[int] | None = None,
) -> Iterator[None]:
    """Enable ring attention over the sequence dimension inside the block.

    Within this region, ``F.scaled_dot_product_attention`` computes attention
    collaboratively across the mesh: each rank holds a shard of the keys and
    values, and they are rotated around the ring so every query eventually sees
    every key. The result is numerically equivalent to unsharded attention, and
    per-rank activation memory falls by the mesh size.

    Args:
        mesh: The ``cp`` sub-mesh, or ``None`` to run without context
            parallelism.
        buffers: Tensors that must be sharded alongside the sequence — the
            rotary tables and the attention mask, principally. Forgetting one
            is the classic context-parallel bug: attention then silently pairs
            queries with the wrong positions.
        buffer_seq_dims: The sequence dimension of each buffer.

    Yields:
        None.

    Raises:
        ValueError: If ``buffers`` and ``buffer_seq_dims`` disagree in length.
    """
    if mesh is None or mesh.size() == 1:
        yield
        return
    if (
        buffers is not None
        and buffer_seq_dims is not None
        and len(buffers) != len(buffer_seq_dims)
    ):
        raise ValueError(
            f"buffers ({len(buffers)}) and buffer_seq_dims "
            f"({len(buffer_seq_dims)}) must have the same length"
        )
    from torch.distributed.tensor.experimental import context_parallel

    with context_parallel(
        mesh,
        buffers=list(buffers) if buffers is not None else None,
        buffer_seq_dims=list(buffer_seq_dims) if buffer_seq_dims is not None else None,
    ):
        yield


def maybe_context_parallel(
    mesh: DeviceMesh | None,
    **kwargs: object,
) -> Iterator[None]:
    """Return a context-parallel region, or a null context when disabled.

    A convenience for call sites that would otherwise need an ``if`` around a
    ``with``.

    Args:
        mesh: The ``cp`` sub-mesh, or ``None``.
        **kwargs: Forwarded to :func:`context_parallel_region`.

    Returns:
        A context manager.
    """
    if mesh is None or mesh.size() == 1:
        return nullcontext()  # type: ignore[return-value]
    return context_parallel_region(mesh, **kwargs)  # type: ignore[arg-type,return-value]
