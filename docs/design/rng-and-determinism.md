# RNG and determinism

> The same seed, config, and rank count reproduce the same loss curve.

That is the contract. It is not a nicety — without it, "did my change help?" is
unanswerable, because you cannot distinguish a real improvement from run-to-run
variance you have no way to measure.

## Separated streams

```python
class RNGStreams:
    noise: torch.Generator
    timestep: torch.Generator
    conditioning: torch.Generator
    sampler: torch.Generator
```

Four generators, not one.

With a single global generator, every random draw is coupled to every other. Add
a dropout call somewhere, or change how many timesteps you sample, and the noise
for every subsequent sample shifts. Two runs that differ only in a logging change
diverge, and you spend a day discovering that.

Separated streams mean:

- changing the conditioning dropout rate does not perturb the noise;
- changing the sampler at evaluation time does not perturb training;
- adding a draw to one stream does not renumber another.

```python
generator = rng.fork("my_new_thing")
```

`fork()` derives a fresh generator for a new purpose without disturbing the
existing four — which is how you add randomness without breaking every
in-progress comparison.

## The rule: vary on `data_rank` only

```python
data_rank, data_world = dims.data_coordinates(mesh)
rng = RNGStreams.for_rank(seed, data_rank=data_rank)
```

**Not global rank. Not `world_size`.**

Context-parallel and tensor-parallel ranks hold shards of the *same* sample.
They must draw **identical noise**, because the diffusion target is defined per
sample: if rank 0 and rank 1 hold two halves of one clip and noise them
differently, they are training on two different targets and calling the result
one gradient.

### What the failure looks like

Nothing crashes. No shape is wrong. The loss decreases.

It decreases to a worse place, and you notice — if you notice — when a scaled-up
run underperforms a smaller one that used less context parallelism. By then you
have a month of runs to re-examine.

This is why the rule is on the pull-request checklist and why `for_rank` takes
`data_rank` as a keyword-only argument: it is hard to pass the wrong thing by
accident.

The same rule governs the loader — see
[Data pipeline](../guides/data-pipeline.md).

## Checkpointing the RNG

`RNGStreams` implements `Stateful`, so all four generators round-trip through
DCP. If they did not, every resume would give the same samples different noise
than they would have received, and the loss would step at every resume — small,
persistent, and easy to misread as a data problem.

Verify: after a resume, the first loss should match the last loss before saving,
to within bf16 noise.

## What breaks determinism, legitimately

Determinism holds for a **fixed rank count and parallelism plan**. It does not
hold across changes to either, and it should not be expected to:

- **Different rank counts** change the order of floating-point reductions.
  Addition is not associative in floating point, so the sums differ in the last
  bits, and over thousands of steps that diverges.
- **Different `cp` or `tp`** change how the sequence and the layers are split,
  so the same arithmetic happens in a different order.
- **`torch.compile`** may fuse differently across versions.
- **Non-deterministic kernels.** Some backward kernels use atomics.
  `torch.use_deterministic_algorithms(True)` removes them at a throughput cost;
  it is a debugging tool, not a default.

The honest statement is: *bitwise* reproducibility requires everything fixed;
*statistical* reproducibility — the loss curve within noise — survives a
rank-count change. When you change the plan mid-run, record it, because the run
is no longer comparable to itself before the change.

## What breaks determinism, illegitimately

These are bugs:

- **Iterating a `set`** anywhere that affects computation. Python's set ordering
  varies with insertion history and hash randomisation. Nothing in avgen that
  affects computation iterates a set, and neither should your code.
- **Unsorted directory listings** for shard order. The cursor index means
  something different on every restart.
- **`dict` ordering assumptions** across processes where the dicts were built by
  different code paths.
- **Time-, PID-, or hostname-seeded anything.**
- **Reducing over the wrong mesh**, which makes the result depend on the TP
  degree.

## Verifying determinism

The cheapest useful test, and worth running before any comparison you intend to
believe:

```bash
# Same seed, same rank count, twice.
avgen train --config configs/smoke.yaml train.total_steps=50 train.seed=0 \
  telemetry.jsonl=runs/a.jsonl
avgen train --config configs/smoke.yaml train.total_steps=50 train.seed=0 \
  telemetry.jsonl=runs/b.jsonl
diff <(jq .loss runs/a.jsonl) <(jq .loss runs/b.jsonl)
```

Identical losses mean the pipeline is deterministic. If they differ, find out why
before running anything expensive — every experiment you run afterwards inherits
the uncertainty.

A second test worth having: the same config at `cp=1` and `cp=2` should give the
same loss to within noise for the first few hundred steps. If it does not,
something in the context-parallel path is not sharding consistently — usually a
positional table.

## Further reading

- [Contracts](contracts.md)
- [Data pipeline](../guides/data-pipeline.md)
- [Parallelism](../guides/parallelism.md)
