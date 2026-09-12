# Your first training run

This page walks one real run from nothing to a checkpoint you can sample from,
on a single 8-GPU node. It is deliberately small. The point is to see every
stage work and to learn what the log lines mean, so that when the 512-GPU
version misbehaves you know which number went wrong.

Budget: about an hour of wall clock, most of it data preparation.

## 0. What you need

- One node with 8 GPUs (any 40 GB+ card; the numbers below assume H100 80 GB).
- A few hundred short video clips. A thousand is better. Quality matters more
  than quantity for a smoke run.
- `avgen[data,codecs,text]` on this machine.

## 1. Prepare data offline

Do the expensive, non-deterministic work **once**, outside the training loop.
Encoding video with a VAE inside the training step is the single most common way
to build a pipeline that is slow, memory-hungry, and irreproducible.

```bash
avgen data ingest \
  --input /data/raw_clips \
  --output /data/avgen/ingested \
  --min-seconds 2 --max-seconds 5 --fps 16

avgen data shard \
  --input /data/avgen/ingested \
  --output /data/avgen/shards_480p \
  --codec vae-8x8x4 \
  --text-encoder t5-base \
  --resolution 480p
```

This writes safetensors shards with a sha256 manifest and an atomic commit
marker. A half-written shard is never visible to a reader — which matters,
because ingest jobs get preempted and a partially written shard that a loader
happily reads is a data-corruption bug you will not find for weeks.

Check what you got:

```bash
avgen data inspect /data/avgen/shards_480p
```

Look at the bucket histogram. If 90% of your clips land in one bucket, your
batches will be uniform and your model will learn one resolution — fine for a
smoke run, worth fixing before a real one.

## 2. Sanity-check on synthetic data first

Before touching real data, prove the plan runs:

```bash
torchrun --standalone --nproc-per-node 8 \
  -m avgen.cli.main train --config configs/train/smoke_cpu.yaml \
  train.steps=50
```

`SyntheticSource` is deterministic and needs no optional dependency. If this
fails, the problem is the plan or the environment, not your data — a much
smaller search space.

## 3. Simulate the plan you intend to run

```bash
avgen simulate --config configs/train/node_8gpu.yaml --world-size 8
```

Read three numbers before launching:

- **`memory_gib`** — must sit comfortably below your device memory. If it is at
  76 of 80 GB, you will OOM on the step whose allocator bucket is slightly
  larger. See [Memory](../guides/memory.md).
- **`mfu`** — 0.35–0.50 is healthy for a video DiT at this scale. Below 0.2,
  something is wrong with the plan, not with your patience.
- **`bottleneck`** — which collective costs the most. On one node it should be
  `fsdp_all_gather`; if it says `cp_ring_attention`, your context-parallel
  degree is larger than the sequence justifies.

## 4. Launch

```bash
torchrun --standalone --nproc-per-node 8 \
  -m avgen.cli.main train \
  --config configs/train/node_8gpu.yaml \
  data.root=/data/avgen/shards_480p \
  train.steps=20000 \
  parallel.context=4 \
  parallel.dp_shard=2
```

For 32,400 tokens (480p, 5 s, 16 fps) on 8 GPUs, `cp=4 × dp_shard=2` is a
reasonable starting point: context parallelism cuts the activation memory that
binds, and the remaining two ranks shard parameters and give you two independent
data groups.

## 5. Read the log

The first lines are the ones worth reading carefully:

```text
[avgen] world=8 dp_shard=2 cp=4  mesh=(dp_shard, cp)
[avgen] data_rank=0/2  sequence_shard=0/4
[avgen] model=video_dit params=2.68B  blocks=32  width=2048
[avgen] tokens/sample=32400  micro_batch=1  grad_accum=4  global_batch=8
[avgen] precision=bfloat16 reduce=float32  activation_checkpoint=selective_op
[avgen] resumed=False  seed=1234
```

Check that:

- **`data_rank=0/2`, not `0/8`.** There are two data groups, because CP ranks
  hold shards of the *same* sample and must receive identical data. If this says
  `0/8` your loader is sharding on global rank, and each rank is training on a
  quarter of a clip as if it were a whole one.
- **`global_batch`** is what you think it is. `dp_size × micro_batch ×
  grad_accum`. This is the number two runs must share to be comparable.
- **`grad_accum`** is computed, not guessed. `ParallelDims.gradient_accumulation_for`
  derives it from the global batch you asked for and the local batch that fits.

Then the steady-state lines:

```text
step 100  loss 0.8123  v 0.8123  a 0.0000  |g| 0.94  lr 4.0e-05  tok/s 118k  mfu 0.38  mem 41.2/80.0 GiB  step 1.10s
```

| Field | What it means | What is wrong if it looks odd |
|---|---|---|
| `loss` | Flow-matching velocity MSE over valid tokens | Flat from step 0: LR too low, or the loss mask is empty |
| `\|g\|` | Global gradient norm, through `clip_grad_norm` | Growing without bound: LR too high, or a spike is coming |
| `tok/s` | Valid tokens per second across the job | Falling over time: a data stall, not a model problem |
| `mfu` | Model FLOPs utilisation | Far below the simulator's prediction: communication is not overlapping |
| `mem` | Peak allocated / device total | Climbing step over step: something is retained across steps |
| `step` | Wall-clock seconds | p99 ≫ p50: stragglers, or checkpoint saves on the critical path |

Metrics are accumulated **on device** and synchronised once per log interval.
There is no `.item()` in the training step, because a per-step host sync
serialises every rank against the slowest one and costs more than the metric is
worth.

## 6. Watch for the two failures that matter early

**Loss spikes.** A flow-matching loss that jumps and recovers is normal. One
that jumps and stays up has usually taken a bad step. Check `|g|` in the
preceding steps: an order-of-magnitude spike before the loss spike means
clipping is too loose. See
[Troubleshooting](../guides/troubleshooting.md#loss-spikes).

**Memory climbing.** Peak memory should be flat after the first few steps. If it
rises monotonically, something is holding a reference across steps — usually a
metric tensor accumulated without `detach()`.

## 7. Checkpoint and resume

Checkpoints are Distributed Checkpoint directories, saved asynchronously so the
training step does not block on I/O:

```text
runs/av_2b_480p/
  config.resolved.yaml
  step_2000/
  step_4000/
  latest -> step_4000
```

Resume by pointing at the run directory; avgen picks up `latest`:

```bash
torchrun --standalone --nproc-per-node 8 \
  -m avgen.cli.main train --config configs/train/node_8gpu.yaml \
  checkpoint.resume=runs/av_2b_480p
```

The important property: **a DCP checkpoint reshards.** You can save on 8 ranks
and resume on 64, or save with `cp=4` and resume with `cp=8`. The optimizer
state, the EMA, the LR schedule, and the data cursor all come back. See
[Checkpointing](../guides/checkpointing.md).

Verify the resume rather than trusting it. The first loss after resume should
match the last loss before it to within noise. If it jumps, your data cursor did
not restore — see
[Troubleshooting](../guides/troubleshooting.md#dataloader-resume-mismatch).

## 8. Sample

```bash
avgen generate \
  --checkpoint runs/av_2b_480p/latest \
  --prompt "a paper boat drifting down a rain gutter, close up" \
  --steps 30 --guidance 4.5 --seed 0 \
  --output samples/
```

At 20,000 steps on a few thousand clips, expect plausible motion and mush for
detail. That is the correct outcome. What you are checking is that the sampler,
the guidance, and the codec decode path all work — not that the model is good.

## Next

- [Parallelism](../guides/parallelism.md) — moving beyond one node.
- [Scaling to 1000 GPUs](../guides/scaling-to-1000-gpus.md) — what changes, and
  what breaks, between 8 and 1024 ranks.
- [Data pipeline](../guides/data-pipeline.md) — buckets, packing, and resumable
  sources.
