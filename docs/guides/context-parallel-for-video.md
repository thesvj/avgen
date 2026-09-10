# Context parallelism for video

Context parallelism shards the token sequence across ranks. For language models
it is a niche technique for long-context finetuning. For video it is the axis
that decides whether the model fits at all, and it is the reason avgen's model
interior is a sequence rather than a grid.

## Why video needs it and language mostly does not

| | Tokens per sample | Attention share of FLOPs |
|---|---|---|
| LLM pretraining, 4k context | 4,096 | ~10% |
| 480p, 5 s, 16 fps (20 × 30 × 54) | 32,400 | 63% |
| 720p, 5 s, 16 fps (20 × 45 × 80) | **72,000** | **79%** |
| 720p, 10 s, 24 fps (60 × 45 × 80) | 216,000 | 92% |
| 1080p, 10 s, 24 fps (60 × 68 × 120) | 489,600 | 96% |

Two things follow. Activation memory is linear in sequence length, so a 72,000
token clip needs 176 GiB of stored activations per rank without checkpointing —
more than two H100s, for a *2.7B* model. And attention is quadratic, so at 720p
almost all of your arithmetic is attention, which means splitting the sequence
splits nearly all of the work.

Data parallelism does not help with either. Only context parallelism reduces
per-rank sequence length.

## What makes it possible

You cannot shard `(B, C, T, H, W)` along "part of `T` and part of `H`". You can
trivially shard `(B, L, W)` along `L`.

That is the whole argument for
[the sequence-first interior](../design/sequence-first.md). Dense grids live at
the data and codec boundary; a `Patchifier` converts once on the way in and once
on the way out; every block in between sees a `TokenStream`, and every token
carries its own coordinates in physical units. A CP-sharded sequence still knows
exactly where each of its tokens came from, because position is data, not tensor
index.

```python
from avgen.parallel.context import pad_to_multiple, shard_stream

stream = pad_to_multiple(stream, dims.context)   # ragged shards are rejected
local = shard_stream(stream, mesh["cp"])
print(stream.length, local.length)
# 72000 9000
```

`PatchLayout` survives sharding unchanged — it is static geometry, not data —
so the unpatchifier can still fold the gathered sequence back into a grid.

## How ring attention works here

Each rank holds a contiguous shard of queries, keys, and values. Attention needs
every query to see every key, so the K/V shards rotate around a ring: on each of
`cp - 1` steps, every rank passes its K/V block to its neighbour and accumulates
partial attention with online softmax rescaling. After a full rotation every
query has attended to the full sequence, and no rank ever materialised the
`L × L` score matrix.

avgen wraps PyTorch's implementation:

```python
from avgen.parallel.context import context_parallel_region

with context_parallel_region(mesh["cp"], buffers=[q, k, v], buffer_seq_dims=[1, 1, 1]):
    out = attention(q, k, v)
```

Everything that is indexed by sequence position must be sharded consistently —
that is what the `buffers` argument is for. The classic bug is sharding `q`, `k`,
and `v` but forgetting the rotary tables or a positional bias, so rank 3's
tokens get rank 0's positions. There is no error; the model just learns a
scrambled geometry.

## Video is *better* at ring attention than language is

This is worth stating plainly, because the ring-attention literature is written
about causal LLMs and its main difficulty does not apply here.

With causal masking, rank 0 holds the first tokens and attends to almost
nothing, while the last rank attends to everything. The load imbalance is 2× and
fixing it needs striped or zigzag sharding, which complicates every index.

**Video diffusion attention is bidirectional.** Every token attends to every
other token. So a contiguous split is perfectly balanced by construction, and
avgen shards contiguously — the simplest scheme is also the correct one. The
consolation prize for quadratic attention over 72,000 tokens is that at least it
parallelises cleanly.

## The limit: ring volume is bounded, compute is not

Per rank per block, the ring moves `2 × B × (L / cp) × W × bytes` for each of
`cp - 1` hops, in the forward pass and again in the backward. The `(cp - 1)`
and the `1 / cp` cancel, so:

> **Ring volume per rank approaches a constant as `cp` grows, while per-rank
> compute keeps falling.**

For the 2.7B model at 72,000 tokens, on H100 at 45% of peak, NVLink 4 at an
achievable 356 GiB/s:

| `cp` | Local sequence | Ring volume / rank / step | Ring time | Compute | Ring / compute |
|---|---|---|---|---|---|
| 2 | 36,000 | 17.6 GiB | 0.05 s | 5.77 s | 0.9% |
| 4 | 18,000 | 26.4 GiB | 0.07 s | 2.89 s | 2.6% |
| 8 | 9,000 | 30.8 GiB | 0.09 s | 1.44 s | 6.0% |
| 16 | 4,500 | 33.0 GiB | 0.09 s | 0.72 s | 12.9% |
| 32 | 2,250 | 34.0 GiB | 0.10 s | 0.36 s | 26.5% |

Doubling `cp` roughly halves compute and leaves communication flat, so the ratio
doubles every step. That is the ceiling on context parallelism, and it is a
property of the algorithm, not of the implementation.

And that table assumes NVLink. Over InfiniBand NDR at an achievable 35 GiB/s,
the `cp=8` row becomes **0.88 s of communication against 1.44 s of compute** —
ten times worse, and no longer hideable behind compute.

## The rule that follows

**Keep the context-parallel group inside a node.**

With 8 GPUs per node, `cp ≤ 8` costs you a few percent. `cp = 16` crosses a node
boundary and costs you a third of your throughput. If you need more sequence
reduction than `cp = 8` gives, the next lever is activation checkpointing, not
more `cp` — see [Memory](memory.md).

avgen's plan search knows this: it prices the ring against the fabric tier the
group would actually land on, given the mesh ordering. Which is exactly why the
mesh is built `(pp, dp_replicate, dp_shard, cp, tp)` — with `tp` innermost and
`cp` next, both stay intra-node for the common `tp × cp ≤ gpus_per_node` case.

## `cp` composes with FSDP for free

FSDP2 shards parameters across the flattened `dp_shard_cp` mesh, not just
`dp_shard`. CP ranks hold different tokens of the same sample, so they can also
hold different parameter shards. With `dp_shard=128` and `cp=8`, parameters
shard across all 1024 ranks rather than 128.

Not flattening here would leave the CP dimension's memory saving on the table,
and for a model where parameters are a small fraction of the footprint it would
not matter much — but it is free, so avgen takes it.

## Things that go wrong

**The sequence does not divide by `cp`.** Ragged shards are rejected upstream.
Call `pad_to_multiple(stream, cp)`; the mask marks the padding invalid, the loss
ignores it, and `loss_mask()` keeps the normalisation honest.

**A positional table is not sharded.** Symptom: the model trains, slowly, to a
worse loss than the same config at `cp=1`. Always A/B a new plan against `cp=1`
on a small config; the losses should match within noise for the first few
hundred steps.

**Data or noise differs across CP ranks.** They hold shards of *one* sample and
must see identical data and identical noise. `RNGStreams.for_rank` varies on
`data_rank` only, and the loader shards on `data_rank` only. If you write a new
source, this is the rule to get right — see
[Data pipeline](data-pipeline.md).

**Loss reduced over the wrong mesh.** Reduce over `dp_cp`, via
`avgen.parallel.data_mesh(mesh)`. Reducing over the world divides by the TP and
PP ranks, which hold the same loss.

**Metrics gathered per step.** Gathering the full sequence to compute a metric
inside the step undoes the entire point of sharding it. Accumulate on device,
sync at log cadence.

## Checking it

The simulator applies the real CP plan at any world size on one CPU machine:

```bash
avgen simulate --config configs/av_2b_720p.yaml --world-size 512
```

Look at `bottleneck`. If it says `cp_ring_attention`, you have over-sharded the
sequence — lower `cp` and spend the ranks on `dp_shard` instead.

## Further reading

- [Parallelism](parallelism.md) — where `cp` sits among the five axes.
- [Sequence-first](../design/sequence-first.md) — why the interior is a sequence.
- [Memory](memory.md) — the other lever on activation memory.
