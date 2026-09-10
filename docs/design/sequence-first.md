# Sequence-first

The single most consequential decision in avgen: **dense grids live at the data
and codec boundary, and nowhere else.**

A video latent is naturally a 5-D grid `(B, C, T, H, W)`, and that is how a VAE
emits it and how a shard file stores it. But a transformer does not see a grid —
it sees a sequence — and everything that makes large-scale video training
possible depends on the model interior being an *explicit* sequence.

## The four things this buys

### Context parallelism

You cannot shard `(B, C, T, H, W)` along "part of `T` and part of `H`". You can
trivially shard `(B, L, W)` along `L`.

At 40k–500k tokens per clip this is the axis that decides whether a model fits
at all. Anything that keeps the grid shape inside the model has to invent a
bespoke spatial-temporal sharding scheme, get its halo exchanges right, and
redo the work for every new attention pattern. A sequence needs none of that.

### Variable resolution in one batch

Two samples of different spatial extent become sequences of the same length
class, distinguished only by their coordinates and their mask. A 480p clip and a
720p clip can share a batch without either being padded to a common grid.

### Sequence packing

Putting a 2-second clip and an 8-second clip in one batch without padding to the
longer is only expressible on a sequence. On a grid, the shorter sample is
padding all the way up to the longer one's frame count, and you pay full
attention cost on that padding.

### Ring and Ulysses attention are sequence algorithms

They are defined over a sharded sequence dimension. Handing them a grid means
flattening it anyway — the only question is whether the flattening is an
explicit, coordinate-carrying representation or an implicit one buried in a
`reshape`.

## The representation

```python
@dataclass(frozen=True, slots=True)
class TokenStream:
    tokens: Tensor        # (B, L, W)   sequence-major; L is the CP-shard axis
    coords: Tensor        # (B, L, 3)   float32 (seconds, row, col) PHYSICAL units
    mask: Tensor          # (B, L)      bool validity
    noise_level: Tensor   # (B,) or (B, L)  float32 in [0, 1]; 0 = clean
    layout: PatchLayout   # static geometry, survives CP sharding
    conditioned: Tensor   # (B, L)      bool clean-anchor mask
```

### Coordinates are physical, not indices

`(seconds, row, col)` — **not** `(frame_index, y_index, x_index)`.

This is the detail that makes the rest work. A shuffled, packed, or CP-sharded
sequence still knows where each of its tokens came from, because position is
data carried alongside the token rather than implied by its offset in a tensor.

Physical units specifically, because:

- **Two clips at different frame rates are comparable.** Token 40 of a 16 fps
  clip and token 60 of a 24 fps clip are both at 2.5 seconds. If coordinates
  were indices, the model would have to learn a separate temporal geometry per
  frame rate.
- **Two clips at different resolutions are comparable.** Row 22 of a 45-row
  latent and row 33 of a 68-row latent describe different absolute positions,
  and the model should know that.
- **A continuation starting at t = 8 s is expressible** without renumbering
  anything.

### The mask is not padding bookkeeping

`mask` marks which tokens are real. `loss_mask()` combines it with `conditioned`
so that clean anchor tokens are excluded from the objective, and the
normalisation divides by the count of tokens that actually contributed. Get this
wrong and your loss silently scales with your padding fraction — which varies by
batch, so it looks like noise.

### `noise_level` is per-sample or per-token

Per-sample is the ordinary diffusion case. Per-token is what makes inpainting,
continuation, and image-to-video a single code path instead of three: clean
anchor tokens get noise level zero and are marked in `conditioned`, noised
tokens get the sampled level, and the model reads `expanded_noise()` without
caring which mode produced it.

### `PatchLayout` is metadata, not data

It holds no tensors, is hashable, and is safe to treat as a compile-time
constant. Two batches with the same layout share a compiled graph — which is
what makes bucketed variable-resolution training compatible with
`torch.compile`. It also survives CP sharding unchanged, so the unpatchifier can
fold the gathered sequence back into a grid.

## The boundary

```python
class Patchifier(Protocol):
    def to_tokens(self, grid, *, positions, mask, noise_level, conditioned=None) -> TokenStream: ...
    def to_grid(self, stream: TokenStream) -> Tensor: ...
    def layout_for(self, grid_shape: tuple[int, ...]) -> PatchLayout: ...
```

One conversion in, one conversion out, and the round trip is exact. Patch
ordering is time-major, then row, then column — fixed, because a checkpoint
written under one ordering and read under another produces a model that is
subtly, unfixably wrong.

## What was rejected

**Keeping the grid and adding sharding hooks.** This is what most video training
code does, and it works up to one node. It fails at the point where you want a
new attention pattern, a new resolution mix, or a new conditioning mode, because
each of those needs its own grid-aware sharding logic. The sequence
representation makes all three the same problem.

**Implicit positions via tensor order.** Cheaper — no `coords` tensor — and it
breaks the moment anything reorders, packs, or shards the sequence. The memory
cost of three float32 per token is under 1% of the token itself.

**A `Sequence` base class with subclasses per modality.** Video and audio
streams have identical structure; the differences are in the layout
(`is_temporal_only()`) and the coordinates. A single frozen dataclass keeps the
parallelism code monomorphic, which matters when it is shard-and-gather code
that must be exactly right.

## What it costs

Honestly: some ergonomic friction. Writing a block means thinking in `(B, L, W)`
and consulting `coords` for anything spatial, rather than indexing a grid. A
convolution over the spatial dimensions is awkward — which is a real cost, and
part of why avgen targets transformers rather than U-Nets.

The trade is deliberate. Convenience inside a block is worth much less than
being able to shard the sequence at all.

## Further reading

- [Context parallelism for video](../guides/context-parallel-for-video.md)
- [Architecture](architecture.md)
- [Adding a model](adding-a-model.md)
