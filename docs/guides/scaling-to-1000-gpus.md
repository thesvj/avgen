# Scaling to 1000 GPUs

What changes between 8 ranks and 1024, and what breaks. Read
[Parallelism](parallelism.md) first; this page assumes you know what the five
axes do.

## The scaling ladder

Each rung has a different binding constraint, and the plan that was right on the
rung below is usually wrong on the next one.

| Ranks | Binding constraint | Plan that usually works | What newly bites |
|---|---|---|---|
| 1 | Fits at all | `cp=1`, `full` checkpointing, tiny clips | Nothing distributed |
| 8 (1 node) | Activation memory | `cp=4`, `dp_shard=2`, `selective_op` | Nothing crosses a node yet |
| 64 (8 nodes) | Inter-node bandwidth | `cp=8`, `dp_shard=8` | FSDP all-gather now crosses the fabric |
| 512 (64 nodes) | FSDP ring width, stragglers | `cp=8`, `dp_shard=64`, maybe HSDP | Slowest rank sets the pace; checkpoints get slow |
| 1024+ (128 nodes) | Failures, and the global batch | `cp=8`, `dp_shard=128`, HSDP | Something fails every few hours; batch size stops fitting the token budget |

## What breaks, in the order you meet it

### Crossing the first node boundary (8 → 16)

Everything that was NVLink is now InfiniBand: roughly **10× less bandwidth and
3× more latency**. The collectives that were free stop being free.

The two rules that follow are the same two rules from
[Parallelism](parallelism.md), and this is where they start being enforced by
physics rather than by policy:

- **`tp ≤ gpus_per_node`.** TP's two collectives per block are on the critical
  path and cannot be hidden.
- **`cp ≤ gpus_per_node`, in practice.** Ring volume per rank is bounded, but at
  35 GiB/s instead of 356 it stops hiding behind compute. See
  [the ring volume table](context-parallel-for-video.md#the-limit-ring-volume-is-bounded-compute-is-not).

The mesh ordering `(pp, dp_replicate, dp_shard, cp, tp)` is what makes both
achievable: `tp` innermost and `cp` next means `tp × cp ≤ gpus_per_node` keeps
both inside one node automatically.

### The FSDP ring gets too wide (256 →)

Sharding parameters across 1024 ranks means an all-gather whose ring spans the
whole cluster. Latency accumulates per hop, and the tail dominates.

Switch to **HSDP**: shard within a group, replicate across groups.

```python
dims = ParallelDims(world_size=1024, dp_replicate=8, dp_shard=16, context=8)
```

The expensive per-block all-gather now stays inside 16 ranks — often within a
node or a rack — and only the cheap gradient all-reduce crosses the slow fabric,
once per step rather than once per block.

The trade: `dp_replicate` groups each hold a full copy of the sharded state, so
memory per rank rises by the replication factor's inverse. Check the simulator
before assuming it still fits.

### Stragglers (512 →)

At 512 ranks, every step waits for the slowest one. A single GPU running 5%
slow — thermal throttling, a bad card, an ECC-correcting DIMM — costs you 5% of
the whole job, and it is invisible in the average.

Watch **p99 step time against p50**. `ThroughputMeter` reports both. A p99 more
than about 20% above p50 means a straggler, not noise. Find it by logging
per-rank step time once a minute and looking at the distribution's tail; then
drain that node.

### Checkpoints stop being free (512 →)

A 2.7B model with AdamW state is roughly 40 GB of tensors. Writing that
synchronously from 512 ranks to a shared filesystem stalls the job for minutes.

- Use `async_save=True`, so the copy to host memory blocks briefly and the write
  proceeds in the background.
- Save less often than feels safe, and rely on fast resume instead.
- Test the resume path *before* you need it. A checkpoint you have never
  restored is not a checkpoint.

See [Checkpointing](checkpointing.md).

### Something fails every few hours (1024 →)

At 128 nodes, hardware failure is a scheduled event, not an exception. Plan for
it:

- **torchelastic** restarts the job on a node failure; combined with
  resume-latest, the cost is the time since the last checkpoint.
- **`avgen[fault-tolerance]`** (torchft) goes further, recovering per-step
  without a full restart. It changes process-group semantics, which is why it is
  a separate extra and an explicit choice.
- **NCCL flight recorder** is what turns a hang into a diagnosis. Turn it on
  before you need it — see
  [Troubleshooting](troubleshooting.md#nccl-timeouts-and-the-flight-recorder).

### The global batch stops being a free parameter (1024 →)

With `dp_size = 128` and a micro-batch of 1, the smallest global batch you can
express without idling ranks is 128 samples. At 72,000 tokens each, that is 9.2M
tokens per step. If your token budget calls for a smaller batch, ranks sit idle
or you accumulate to a batch you did not want.

`gradient_accumulation_for` derives the accumulation from the global batch you
asked for and the local batch that fits:

```python
accumulation = dims.gradient_accumulation_for(
    global_batch_size=256, local_batch_size=1
)
```

If the global batch does not factor across the data dimension, avgen raises
rather than silently rounding — because a silently changed effective batch means
two runs are no longer comparable, and that is worse than a crash.

## A worked plan at 1024

2.7B parameters, 32 blocks, width 2048, 72,000 tokens (720p, 5 s, 16 fps).
H100 80 GB, 8 per node, NVLink 4 and InfiniBand NDR.

```python
dims = ParallelDims(world_size=1024, dp_replicate=1, dp_shard=-1, context=8)
# world=1024 dp_shard=128 cp=8
```

| | |
|---|---|
| Per-rank memory (`selective_op`) | ≈ 12 GiB of 80 |
| Per-rank compute per step | ≈ 1.44 s |
| CP ring volume | ≈ 31 GiB/step, intra-node, ≈ 0.09 s |
| FSDP traffic | ≈ 20 GiB/step, inter-node, ≈ 0.6 s, mostly overlapped |
| Data groups | 128 |

Twelve gigabytes of eighty is a lot of headroom. The right move is to spend it:
raise the micro-batch to 2 or 4, which roughly doubles or quadruples the compute
each collective has to hide behind and raises MFU. Then re-simulate.

## The pre-launch checklist

Before a run that costs more than a few GPU-hours:

- [ ] `make simulate`, or `avgen plan`, for the exact shape and world size.
- [ ] Memory prediction below 80% of device memory.
- [ ] `bottleneck` is `fsdp_all_gather`, not `cp_ring_attention` or
      `tp_all_reduce`. The latter two mean an axis is over-sharded or crossing
      nodes.
- [ ] `tp × cp ≤ gpus_per_node`.
- [ ] Global batch factors across `dp_size`.
- [ ] The same config ran for 50 steps at 8 ranks and the loss looked sane.
- [ ] Resume tested: save at step 20, kill, resume, and confirm the loss
      continues rather than jumping.
- [ ] Flight recorder enabled (`TORCH_NCCL_TRACE_BUFFER_SIZE`,
      `TORCH_NCCL_DUMP_ON_TIMEOUT`).
- [ ] Collective timeout set high enough for the first step, which includes
      compilation and lazy NCCL initialisation, and low enough that a hang is
      caught in minutes rather than hours.
- [ ] Telemetry logging p50 and p99 step time, tokens/s, MFU, and peak memory.

## What does *not* change with scale

Worth saying, because it is the point of the design:

- The model code. `TokenStream` in, `ModelOutput` out, at 1 rank or 1024.
- The objective, the sampler, the metrics.
- The checkpoint format. DCP reshards; you can move from 512 to 1024 ranks
  mid-run.
- The config. Parallelism is a handful of integers, overridable from the command
  line.

If scaling required editing model code, the abstraction would have failed.

## Further reading

- [Parallelism](parallelism.md)
- [Simulation](simulation.md)
- [Troubleshooting](troubleshooting.md)
- [Checkpointing](checkpointing.md)
