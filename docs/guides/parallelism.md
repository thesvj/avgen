# Parallelism

This is the guide to read before you launch anything larger than one node.

A thousand-GPU video training job is a five-dimensional problem. The single most
useful thing a framework can do is make those five dimensions explicit, named,
and validated in one place, instead of scattering `world_size // something`
arithmetic through the codebase. That is what `ParallelDims` is.

```python
from avgen.parallel import ParallelDims

dims = ParallelDims(
    world_size=1024,
    dp_replicate=1,  # replication for throughput
    dp_shard=-1,  # FSDP2; -1 means "whatever is left over"
    context=8,  # shard the token sequence
    tensor=8,  # shard each layer's weights and activations
    pipeline=1,  # shard model depth
)
print(dims.describe())
# world=1024 dp_shard=16 cp=8 tp=8
```

## The five axes

| Axis | Name in the mesh | What it reduces | What it costs |
|---|---|---|---|
| Replicated data parallel | `dp_replicate` | Nothing — it buys throughput | Gradient all-reduce per step |
| Sharded data parallel (FSDP2) | `dp_shard` | Parameter, gradient, and optimizer memory | All-gather per block per forward, reduce-scatter per block per backward |
| Context parallel | `cp` | **Per-rank sequence length**, so activation memory *and* attention FLOPs | KV rotation around a ring, once per block |
| Tensor parallel | `tp` | Per-layer weight and activation memory | Two collectives per block, on the critical path |
| Pipeline parallel | `pp` | Model depth held per rank | Bubbles, and stage-imbalance stalls |

Read that table as a cost model, not a menu. Every axis reduces something and
charges for it; the job is to spend the cheapest currency you have.

## The order to reach for them

**This ordering is the guide.** It holds for video diffusion transformers
specifically, because sequence length is the binding constraint and attention is
most of the arithmetic.

### 1. `dp_shard` — always, and first

FSDP2 is nearly free. A 2.7B model in bf16 is 5.0 GiB of parameters, another 5.0
of gradients, and 30 GiB of AdamW state in fp32 — 40 GiB before a single
activation exists. Shard across 64 ranks and that becomes 0.7 GiB. The
communication is per-block all-gathers that overlap with compute almost
perfectly.

Start with `dp_shard=-1` and let it absorb whatever the other axes leave.

### 2. `cp` — as soon as the sequence is long

For video this is the axis that decides whether the model fits at all, and it is
the axis that does not exist in an LLM framework's default recipe because LLM
sequences are 4k–32k, not 72k–500k.

Here is a 2.7B DiT on five seconds of 720p at 16 fps — a latent grid of
20 × 90 × 160, patched 1 × 2 × 2, giving **72,000 tokens** — on H100 80 GB, with
no activation checkpointing, `dp_shard` absorbing the rest of a 64-rank world:

| `cp` | Local sequence | Activation memory | Total per rank | Fits in 80 GB? |
|---|---|---|---|---|
| 1 | 72,000 | 175.8 GiB | 178.8 GiB | no |
| 2 | 36,000 | 88.0 GiB | 90.9 GiB | no |
| 4 | 18,000 | 44.0 GiB | 47.0 GiB | yes |
| 8 | 9,000 | 22.0 GiB | 25.0 GiB | comfortably |
| 16 | 4,500 | 11.0 GiB | 14.0 GiB | comfortably |

Parameters, gradients, and optimizer state together are under 1 GiB at these
shard counts. **Activations are 98% of the footprint.** No amount of `dp_shard`
touches them; only `cp` does.

Context parallelism also divides the attention work, which at this length is
79% of all FLOPs. Per-rank compute for one optimizer step, at 45% of H100 peak:

| `cp` | Per-rank compute |
|---|---|
| 1 | 11.5 s |
| 4 | 2.9 s |
| 8 | 1.44 s |
| 16 | 0.72 s |

See [Context parallelism for video](context-parallel-for-video.md) for how ring
attention actually works and where the limits are.

### 3. `tp` — when `cp` runs out, and only inside a node

Tensor parallelism shards the weights and activations of each layer. It is the
right answer when the model is wide, and it composes with sequence parallelism
so that the normalisation and residual paths are sharded along the sequence too.

Its cost is two collectives per block **on the critical path** — they cannot be
overlapped away, because the next matmul needs the result. Over NVLink that is
tolerable. Over InfiniBand it is not: the same collective that costs
microseconds intra-node costs milliseconds inter-node, thirty-two times per
step, in the forward and again in the backward.

**Therefore: `tp ≤ gpus_per_node`, always.** avgen's plan search enforces this
and the mesh ordering is designed to make it true (see below).

### 4. `dp_replicate` — when `dp_shard` gets too wide

Sharding parameters across 1024 ranks means an all-gather whose ring spans the
entire cluster. Past a few hundred ranks it is better to shard within a group
and replicate across groups — HSDP. The expensive all-gather stays inside a
node or a rack; the cheaper gradient all-reduce crosses the slow fabric.

### 5. `pp` — last

Pipeline parallelism is the only axis that reduces the *depth* held per rank, so
it is what you reach for when a model does not fit even with everything else
maxed. It costs bubbles: with `pp=4` and a naive schedule you lose a fraction of
throughput proportional to `(stages - 1) / microbatches`, and interleaved
schedules only reduce it. It also demands enough blocks per stage — a two-block
stage is nearly all bubble, which is why avgen's plan search rejects splits
thinner than four blocks by default.

For a 2–14B video DiT you almost certainly do not need it. Try it when `cp × tp`
has run out and the model still does not fit.

## Mesh ordering is a performance decision

The mesh is built in the order:

```text
(pp, dp_replicate, dp_shard, cp, tp)
```

Rank ordering makes the **last** dimension vary fastest. So `tp` ranks are
adjacent — landing inside one NVLink domain — and `pp` ranks are furthest apart.
That is correct, because pipeline traffic is a small point-to-point activation
handoff while tensor-parallel traffic is an all-reduce on every layer.

**Getting this order wrong is worth a large fraction of your throughput and
produces no error message.** Nothing crashes. The job just runs at half speed,
and it is nearly impossible to diagnose from inside the training loop. This is
why the ordering is asserted in CI by the
[simulator workflow](simulation.md#the-ci-gate).

Only dimensions greater than one appear in the mesh: a degenerate dimension of
size one costs a process group and buys nothing, and its absence is what lets
the same plan code run unchanged on one GPU.

Two flattened views are registered because they are needed constantly:

- **`dp_shard_cp`** — FSDP2 shards parameters across *both* the sharded data
  dimension and the context dimension. CP ranks hold different tokens of the
  same sample, so they can also hold different parameter shards. Not flattening
  here would leave the CP dimension's memory saving on the table.
- **`dp_cp`** — the ranks holding different data or different sequence shards.
  **This is the mesh losses and metrics reduce over.** Reducing over the whole
  world would divide by the TP and PP ranks, which hold the same loss, and your
  reported loss would silently be wrong by a constant factor.

## Data and RNG follow `data_rank`, not global rank

This is the rule that produces the worst bugs when it is broken, because
everything appears to work.

```python
data_rank, data_world = dims.data_coordinates(mesh)
```

`data_world` is `dp_replicate × dp_shard`, **not** `world_size`. Context- and
tensor-parallel ranks hold shards of the *same* sample. They must receive:

- **identical data** — otherwise rank 3 is computing attention over a quarter of
  a different clip;
- **identical noise** — otherwise the diffusion target differs across shards of
  one sample and the gradient is meaningless.

```python
from avgen.core.rng import RNGStreams

rng = RNGStreams.for_rank(seed, data_rank=data_rank)
```

The failure mode is subtle: the loss still goes down, just to a worse place, and
you only notice when a scaled-up run underperforms a small one. See
[RNG and determinism](../design/rng-and-determinism.md).

## Worked example: 1024 GPUs, 720p, 5 seconds

Sequence 72,000 tokens. 2.7B parameters, 32 blocks, width 2048. H100 80 GB,
NVLink 4 intra-node, InfiniBand NDR inter-node, 8 GPUs per node.

**Step 1 — how much `cp` do you need?** From the table above, `cp=8` brings
activations to 22 GiB with no checkpointing, or 9.5 GiB with `selective_op`.
`cp=8` also fits inside one node, so the ring never crosses the slow fabric.
Take `cp=8`.

**Step 2 — what is left?** 1024 / 8 = 128 ranks. With `tp=1`, that is
`dp_shard=128`.

**Step 3 — memory.** At `cp=8`, `dp_shard=128`, `selective_op` checkpointing:

| Term | GiB |
|---|---|
| Parameters (bf16, sharded over 1024) | 0.005 |
| Gradients | 0.005 |
| Optimizer state (AdamW, fp32 master + moments) | 0.03 |
| Activations (`selective_op`) | 9.5 |
| FSDP gather peak (prefetch 1) | 0.3 |
| Workspace and allocator overhead | 2.0 |
| **Total** | **≈ 12 GiB** |

Twelve gigabytes of eighty. The obvious next move is to spend that headroom on a
larger micro-batch, which raises MFU by giving the collectives more compute to
hide behind.

**Step 4 — communication.** Per rank per step, in round numbers:

| Collective | Volume | Tier | Time |
|---|---|---|---|
| CP ring attention (K, V, fwd + bwd) | ≈ 31 GiB | NVLink, 356 GiB/s | ≈ 0.09 s |
| FSDP all-gather (fwd + bwd) | ≈ 10 GiB | InfiniBand, 35 GiB/s | ≈ 0.29 s |
| FSDP reduce-scatter (fp32 grads) | ≈ 10 GiB | InfiniBand, 35 GiB/s | ≈ 0.29 s |

Against 1.44 s of per-rank compute, with the usual ~0.8 overlap efficiency, most
of that hides. The ring is cheap *because it stayed inside the node* — the same
31 GiB over InfiniBand would be 0.9 s, comparable to the entire compute time.

That single comparison is the most important number on this page.

**Step 5 — check it, do not trust it.**

```bash
avgen plan --model 2b --world-size 1024 --seq-len 72000
```

The numbers above come from the same analytical models the plan search uses;
they are estimates calibrated against published hardware peaks, not
measurements of your cluster. Calibrate the interconnect from `nccl-tests`
before you trust a projection:

```python
from avgen.simulate.comms import calibrate_from_busbw

nvlink = calibrate_from_busbw("NVLink 4", measured_busbw_gbps=372.0, peak_gbps=450.0)
```

## Applying a plan

```python
from avgen.parallel import ParallelConfig, ParallelDims, parallelize
from avgen.parallel.activation import ActivationCheckpointConfig
from avgen.parallel.precision import PrecisionConfig

dims = ParallelDims(world_size=1024, dp_shard=-1, context=8)
mesh = dims.build_mesh("cuda")

parallel = parallelize(
    model,
    dims,
    mesh,
    ParallelConfig(
        precision=PrecisionConfig(param_dtype="bfloat16", reduce_dtype="float32"),
        activation_checkpoint=ActivationCheckpointConfig(mode="selective_op"),
        sequence_parallel=True,
        compile_blocks=True,
    ),
)
```

`parallelize` applies the transformations in the only order that works:
tensor parallel, then activation checkpointing, then `torch.compile`, then
FSDP2. Applying FSDP before TP would shard parameters that TP then tries to
shard again; compiling after FSDP compiles the communication instead of the
compute. The returned `ParallelModel` carries the mesh and the submeshes the
training loop needs (`.cp_mesh`, `.pp_mesh`).

## Quick reference

| Symptom | Reach for |
|---|---|
| OOM, activations dominate | `cp`, then activation checkpointing |
| OOM, optimizer state dominates | `dp_shard` |
| OOM at every `cp`, model is very wide | `tp`, capped at `gpus_per_node` |
| OOM with everything maxed, model is very deep | `pp` |
| Fits, but MFU is low and `bottleneck` is FSDP | `dp_replicate` (HSDP), or a larger micro-batch |
| Fits, but `bottleneck` is `cp_ring_attention` | Lower `cp`; you over-sharded the sequence |
| Fits, but `bottleneck` is `tp_all_reduce` | `tp` is crossing nodes — it must not |

## Further reading

- [Context parallelism for video](context-parallel-for-video.md)
- [Memory](memory.md)
- [Simulation](simulation.md)
- [Scaling to 1000 GPUs](scaling-to-1000-gpus.md)
- [Adding a parallelism plan](../design/adding-a-parallelism-plan.md)
