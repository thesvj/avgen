# Data pipeline

The data layer has one job at scale: deliver identical tensors to every rank
that holds a shard of the same sample, resumably, without becoming the
bottleneck. Everything below follows from that.

## Do the expensive work offline

Encode video to latents and prompts to text features **once**, into shards, and
let the training job read tensors.

Running a VAE inside the training step is the most common way to build a
pipeline that is slow, memory-hungry, and irreproducible. It serialises against
your data rate, competes for the memory the model needs, drags `diffusers` onto
every rank, and makes your run depend on a version of a library that will change
under you.

```bash
avgen data ingest --input /data/raw --output /data/ingested --fps 16 \
  --min-seconds 2 --max-seconds 5
avgen data shard  --input /data/ingested --output /data/shards_720p \
  --codec vae-8x8x4 --text-encoder t5-base --resolution 720p
avgen data inspect /data/shards_720p
```

Ingest needs `avgen[data,codecs,text]`. **The training cluster needs none of
them.**

## Shards

`write_shard` / `read_shard` are mmap plus safetensors, with a sha256 manifest
and an atomic commit marker.

The atomicity matters more than it sounds. Ingest jobs get preempted. A
half-written shard that a reader happily consumes is a data-corruption bug that
surfaces weeks later as an unexplained loss plateau. A shard without its COMMIT
marker is invisible to readers.

Nothing is loaded eagerly: mmap means a shard costs address space, not RAM, so a
rank can hold thousands of shards open.

## Buckets

Video datasets are not uniform. Clips differ in duration and aspect ratio, and
padding everything to the largest wastes most of your compute on mask.

`BucketSampler` groups samples into `(resolution, duration)` buckets with
aspect-ratio groups, and emits batches from within one bucket. Two consequences:

- **Every sample in a batch has the same token count**, so nothing is wasted on
  padding and the shapes repeat — which keeps the CUDA allocator from
  fragmenting and lets `torch.compile` reuse a graph.
- **Different batches have different shapes**, which is fine because the model
  interior is a sequence and `PatchLayout` is static per batch.

Check your bucket histogram after ingest. If 90% of clips land in one bucket,
you have effectively a single-resolution dataset.

## The rule: shard on `data_rank` only

```python
data_rank, data_world = dims.data_coordinates(mesh)
source = build_loader(path, data_rank=data_rank, data_world=data_world)
```

`data_world` is `dp_replicate × dp_shard`, **not** `world_size`.

Context- and tensor-parallel ranks hold shards of the *same* sample. If they
receive different data, rank 3 computes attention over a quarter of a different
clip and calls it a sequence shard. The loss still decreases — to a worse place.

The corresponding rule for noise is in
[RNG and determinism](../design/rng-and-determinism.md): `RNGStreams.for_rank`
varies on `data_rank` only, for the same reason.

## Resumability is not optional

```python
class DataSource(Protocol):
    def __iter__(self) -> Iterator[MediaBatch]: ...
    def state_dict(self) -> Mapping[str, Any]: ...
    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...
```

A source that cannot restore its cursor turns every preemption into lost samples
and a silently different data order. At 1024 ranks, preemption is a scheduled
event, so "we do not resume data" means "we replay the first shard forever".

Three things must be in the state, and all three are easy to forget:

1. the position within the shard list, with a **stable ordering** — never a
   `set`, never an unsorted directory listing;
2. the position within the current shard;
3. the bucket sampler's own position, so the resolution mix does not restart.

Verify a resume rather than trusting it: see
[Troubleshooting](troubleshooting.md#dataloader-resume-mismatch).

## `MediaBatch`

The dense, data-edge representation. It carries video, audio, and text tensors
with their masks, per-sample positions, `sample_ids`, and a `MediaBatchSpec`
describing shapes and exact rational timebases.

Two details worth knowing:

- **Timebases are exact rationals** (`video_timebase_num/den`), not floats. A
  29.97 fps clip is 30000/1001, and storing that as a float accumulates drift
  that eventually misaligns audio from video by a frame.
- **`sample_ids` stay on CPU.** They are int64 identifiers, never used in
  computation, and moving them to device would add a transfer per batch and a
  sync every time you wanted to log one.

`MediaBatch` is where `validate()` is called — at the boundary, once per batch,
never in the hot loop.

## Synthetic data

`SyntheticSource` is deterministic and needs no optional dependency. Use it to
separate "the plan is wrong" from "the data is wrong":

```bash
avgen train --config configs/train/smoke_cpu.yaml train.steps=50
```

It is also how you measure whether the loader is your bottleneck: if synthetic
is much faster than real, the problem is I/O.

## Throughput checklist

- [ ] Latents and text features precomputed; no codec in the training loop.
- [ ] Shards on a filesystem that survives 1024 concurrent readers.
- [ ] Bucket histogram is not degenerate.
- [ ] Loader sharded on `data_rank`, verified in the startup log
      (`data_rank=0/128`, not `0/1024`).
- [ ] `state_dict` round-trips; resume verified against an uninterrupted run.
- [ ] Prefetch deep enough that the loader is never the critical path.

## Further reading

- [RNG and determinism](../design/rng-and-determinism.md)
- [Checkpointing](checkpointing.md)
- [Troubleshooting](troubleshooting.md)
