# Troubleshooting

The failures that actually happen, in the order they usually happen, with the
diagnosis before the fix.

---

## NCCL timeouts and the flight recorder

**Symptom.** The job stops producing log lines. Some minutes later:

```text
Watchdog caught collective operation timeout: WorkNCCL(SeqNum=8213, OpType=ALLGATHER,
NumelIn=..., Timeout(ms)=600000) ran for 600018 milliseconds before timing out
```

**What this actually means.** One rank did not reach a collective the others
did. NCCL cannot tell you which, because from its point of view everyone else is
simply waiting. The rank that timed out is almost never the rank that caused it.

**Diagnose it — before you need to.** The NCCL flight recorder keeps a ring
buffer of recent collectives per rank and dumps it on timeout. Turn it on for
every multi-node run:

```bash
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=/shared/run/nccl_trace_rank_
```

After a timeout, compare the last recorded sequence number per rank. The ranks
that are *ahead* are waiting; the one that is *behind* is the culprit. That
single comparison usually identifies the node in a minute.

Set the collective timeout deliberately:

```python
from avgen.parallel.env import init_distributed

env = init_distributed(timeout_seconds=600, init_timeout_seconds=1800)
```

The first step is slow — `torch.compile`, lazy NCCL channel setup, the first
all-gather — so the init timeout must be generous. Steady-state should not be:
ten minutes is long enough to survive a hiccup and short enough that a hang
costs you ten minutes instead of a night.

**Common causes, most to least likely.**

| Cause | Tell |
|---|---|
| Rank divergence: a `if rank == 0` branch that calls a collective | Flight recorder shows one rank missing one op |
| Data-dependent control flow — different shapes on different ranks | The lagging rank differs run to run |
| An actual hardware or fabric fault | `dmesg`, `nvidia-smi -q`, IB counters |
| An OOM on one rank that killed it | The process is gone; check `dmesg` for the OOM killer |
| Checkpoint save blocking one rank past the timeout | Timeout happens right after a save step |
| Wrong interface selected | Fails on the first collective, every time |

For the last one:

```bash
export NCCL_SOCKET_IFNAME=ib0        # not the management interface
export NCCL_IB_HCA=mlx5              # the right adapters
export NCCL_DEBUG=INFO               # once, to confirm what it chose
```

---

## OOM triage order

**Do not start turning knobs.** Read the numbers first; the fix depends on which
of three different problems you have.

### Step 1: read the allocator summary

```text
CUDA out of memory. Tried to allocate 2.20 GiB.
GPU 0 has a total capacity of 79.15 GiB of which 1.31 GiB is free.
Of the allocated memory 61.42 GiB is allocated by PyTorch,
and 14.83 GiB is reserved by PyTorch but unallocated.
```

- **`reserved but unallocated` is large (> ~10%)** → fragmentation, not
  capacity. See step 4.
- **`allocated` is close to capacity** → genuinely too big. Go to step 3.
- **The failing allocation is huge relative to the model** → something
  materialised that should not have. A gathered sequence, an `L × L` score
  matrix, a full-precision copy.

### Step 2: ask the simulator what it expected

```bash
avgen simulate --config your.yaml --world-size <N>
```

If prediction and reality agree, you have a capacity problem and the levers in
[Memory](memory.md) apply. If reality is much worse, something structural is
wrong — a custom attention storing scores, a metric holding a graph, a codec
left inside the training loop.

### Step 3: is the peak flat?

Log peak memory every step for fifty steps.

- **Flat** → capacity. Apply the levers in order: raise `cp`, enable
  `selective_op`, lower the micro-batch, raise `tp`.
- **Climbing** → a leak. Almost always a metric accumulated without `detach()`,
  or a Python list holding tensors across steps. `StepMetrics` fields are
  detached fp32 scalars for exactly this reason.
- **Spiking on one step in N** → it is the checkpoint save, the EMA update, or a
  validation pass. Those need headroom too.

### Step 4: fragmentation

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

The deeper fix is fixed shapes. Variable-resolution batching is a fragmentation
generator unless the bucket sampler produces a small, repeating set of shapes —
which is what `BucketSampler` is for. If you are seeing fragmentation, check how
many distinct shapes your loader actually emits.

### Step 5: OOM on one rank only

That is not a memory problem, it is an imbalance problem.

- With `pp`, stage 0 holds more activations than the last stage. Rebalance the
  split points.
- With uneven bucketing, one data group can draw longer sequences than another.
- With a per-rank validation or logging path, rank 0 does extra work.

---

## Loss spikes

**Normal:** a flow-matching loss that jumps and recovers within tens of steps.
Diffusion losses are noisy because the timestep is sampled; a batch that lands
in a hard timestep region genuinely has a higher loss.

**Not normal:** a jump that does not recover, or one followed by NaN.

### Diagnose

Look at the **gradient norm in the steps *before* the spike**, not the spike
itself. The pattern is nearly always: `|g|` climbs an order of magnitude over a
few steps, the optimizer takes one bad step, and the loss follows.

`StepMetrics` carries `grad_norm`, `nonfinite`, and `skipped` for exactly this.
A rising `skipped` count means the guard is catching bad steps — good, but it is
telling you something upstream is wrong.

### Causes, in order

1. **Clipping too loose.** `max_grad_norm=1.0` is a reasonable default; if your
   observed `|g|` is routinely 5 and you clip at 100, you are not clipping.
2. **LR too high for the current batch size.** Spikes cluster early in training
   and after a warmup that was too short.
3. **A bad batch.** Corrupt samples, an all-black clip, a mislabelled duration.
   Log `sample_ids` at the spike — they stay on CPU precisely so this is cheap —
   and go look at the data.
4. **bf16 accumulation.** Ensure `reduce_dtype="float32"`. Gradient reduction in
   bf16 across 1024 ranks loses meaningful precision.
5. **Timestep sampler mismatched to sequence length.** The shifted logit-normal
   sampler's shift is resolution-dependent for a reason: reusing the 256px shift
   at 720p concentrates sampling in a region where the velocity target is large,
   and the loss becomes spiky. Check that `base_len`/`max_len` bracket your
   actual sequence length.
6. **EMA divergence.** If EMA-evaluated samples are fine and raw ones are not,
   the model is oscillating; lower the LR rather than raising the decay.

### If it is NaN, not a spike

`nonfinite` goes true. Check, in this order: the loss mask is not empty
(`loss_mask()` sums to zero if every token is invalid); no division by a
zero-length count; `noise_level` is in `[0, 1]`; no fp16 anywhere — avgen uses
bf16 for a reason.

---

## Dataloader resume mismatch

**Symptom.** After a resume, the first loss does not match the last loss before
the checkpoint. Or training gets suspiciously good, then plateaus.

**What it means.** The data cursor did not restore. The job is either replaying
data it has already seen — a fast path to memorisation — or skipping data.

### Diagnose

Log `samples_seen` and `tokens_seen` from `TrainState.progress()` immediately
before saving and immediately after resuming. They must match exactly. If they
do, the cursor restored and the problem is elsewhere.

Then check the RNG. `RNGStreams.state_dict()` round-trips through the
checkpoint; if the noise stream restarts, every sample gets different noise than
it would have, and the loss shifts even with perfect data restoration.

### Causes

1. **A `DataSource` without a real `state_dict`.** The protocol requires
   resumability. A source that returns `{}` will not raise, and it will silently
   restart from the top of the shard list.
2. **Sharding on global rank instead of `data_rank`.** After a rank-count change
   the assignment is different, so a resumed job reads a different subset. The
   first symptom is often a loss *drop* — the job is now seeing data it never
   saw — followed by worse validation.
3. **Shard list ordering not stable.** If shards are enumerated from a `set` or
   an unsorted directory listing, the cursor index means something different on
   every restart. Nothing in avgen that affects computation iterates a `set`,
   and neither should your source.
4. **Bucket state not saved.** The bucket sampler has a position too. If it
   restarts, the resolution mix changes at the resume point.

### Verify a resume properly

Save at step 20, kill the job, resume, and compare steps 21–25 against an
uninterrupted run with the same seed. They should match to within bf16 noise. If
they match exactly for one step and then diverge, the RNG restored but the data
cursor did not.

---

## Mesh ordering mistakes

**Symptom.** Nothing crashes. The job runs at half the throughput the simulator
predicted, and profiling shows collectives taking far longer than the message
size justifies.

**What it means.** A parallelism dimension landed on the wrong fabric — almost
always tensor parallel spanning two nodes.

### Diagnose

```python
print(mesh.mesh_dim_names)     # must be ordered (pp, dp_replicate, dp_shard, cp, tp)
```

Only dimensions greater than one appear, so at `dp_shard=128, cp=8, tp=1` you
should see `('dp_shard', 'cp')`. What must never happen is `tp` appearing before
`cp`, or `dp_shard` after `cp`.

Then check the arithmetic: with 8 GPUs per node, `tp × cp ≤ 8` keeps both
innermost dimensions inside one node. `tp=8, cp=2` means the CP ring spans two
nodes; `tp=16` means TP itself does, which is much worse.

Confirm with the topology:

```bash
nvidia-smi topo -m
```

Ranks in the same TP group must show `NV#` (NVLink) between them, not `SYS` or
`PHB`.

### Fix

Do not reorder the mesh. The order `(pp, dp_replicate, dp_shard, cp, tp)` is
correct and is asserted in CI. Change the *degrees* so that `tp × cp` fits
inside a node, and re-run the simulator:

```bash
make simulate
```

### Related: reducing over the wrong mesh

Losses and metrics reduce over `dp_cp`, never the whole world. Reducing over the
world divides by the TP and PP ranks, which hold the *same* loss, so your
reported loss is wrong by a constant factor — which looks like a hyperparameter
problem and is not.

```python
from avgen.parallel import all_reduce_mean, data_mesh

loss = all_reduce_mean(loss, data_mesh(mesh))
```

---

## Slow first step, then fine

Expected. The first step pays for `torch.compile`, lazy NCCL channel setup, the
first FSDP all-gather, and CUDA context creation. On a large job it can be
minutes. This is why the init timeout and the collective timeout are separate
settings.

If it is slow *every* step, and the simulator disagrees, profile one step and
look at whether communication is actually overlapping with compute.

---

## Throughput drops over time

Peak memory flat, MFU falling, tokens/s falling.

1. **A data stall.** Check whether the loader is the bottleneck: run the same
   config against `SyntheticSource` and compare. If synthetic is fast, the
   problem is I/O, not the model.
2. **A straggler that got slower.** Compare p99 to p50 step time. Thermal
   throttling develops over hours.
3. **Filesystem contention** from checkpoint saves that overlap the next save.
4. **Fragmentation** raising allocator pressure. Reserved climbs while allocated
   does not.

---

## Getting help

If none of this resolves it, open a
[bug report](https://github.com/avgen-project/avgen/issues/new?template=bug_report.yml).
The form asks for the avgen version, torch version, GPU, world size, parallelism
degrees, and the exact resolved config — all six, because a distributed training
bug is not reproducible without them. Include the simulator output for your plan
and, if you have one, the flight-recorder dump.
