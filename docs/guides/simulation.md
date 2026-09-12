# Simulation

A parallelism plan for a 1024-GPU job has to be right **before** you spend 1024
GPU-hours discovering that it is not.

That is the whole argument. Everything below is how avgen makes it true.

## The problem

The failure modes of a large plan are the ones that do not show up at small
scale:

- a mesh dimension ordered so tensor-parallel traffic crosses nodes — no error,
  just a job that runs at half speed;
- an FSDP wrap that produces one giant all-gather instead of per-block ones,
  destroying overlap;
- a context-parallel shard that forgets to shard the rotary tables — the model
  trains, to a worse loss;
- an activation-checkpoint policy that leaves the model 4 GB over budget at
  world size 512 but fits fine at 8.

Each costs a launch to discover, and a launch on 128 nodes is an hour of
queueing plus an hour of startup before the error appears.

## What PyTorch gives you

Everything needed to check all of that without a cluster:

| Tool | What it provides |
|---|---|
| `FakeProcessGroup` | A backend whose collectives complete instantly and return uninitialised data. Enough to build a real 1024-rank `DeviceMesh` and run real sharding logic. |
| `FakeTensorMode`, `meta` device | Tensors with real shapes, dtypes, and strides but no storage. A 30B model costs nothing to construct. |
| `CommDebugMode` | Counts and attributes every collective, by module. |
| `MemTracker`, `FSDPMemTracker` | Module-wise memory accounting. |
| `RuntimeEstimator` | Per-operator time estimates from a roofline or from measured kernels. |

avgen wraps these in `avgen.simulate` and adds closed-form memory,
communication, and compute models on top, so that a plan can be *priced* and not
merely *executed*.

## What it is exact about, and what it is not

This distinction matters more than any feature in this guide.

**Exact:** shapes, sharding, memory accounting, collective counts, collective
sizes, mesh topology, whether a plan applies at all.

**Estimated:** time. The models do not know about your fabric's congestion, your
scheduler's placement, or the neighbouring job saturating the same rail.

**Silent:** numerics. Fake collectives return garbage, so a simulated run's loss
is meaningless.

Use it to answer *"does this plan fit and is it shaped right"*. Never *"does
this model learn"*.

## Four things you can do with it

### 1. Search the plan space

Given a model shape, a world size, and a machine, there are usually a few dozen
valid ways to factor the ranks. Only a handful fit, and among those the
throughput spread is routinely 2–3×. Picking by intuition leaves most of a
cluster on the floor.

```python
from avgen.simulate.compute import H100_SXM
from avgen.simulate.memory import ModelShape
from avgen.simulate.plan import SearchSpace, render_plan_table, search_parallel_plan

shape = ModelShape(
    parameters=2_684_354_560, depth=32, width=2048,
    num_heads=16, sequence_length=72_000, micro_batch_size=1, text_tokens=256,
)
space = SearchSpace(
    world_size=1024,
    gpus_per_node=8,
    max_tensor=8,        # never exceeds gpus_per_node, whatever you ask for
    max_context=16,
    max_pipeline=1,      # pipeline is the last axis to reach for
    global_batch_size=256,
)
candidates = search_parallel_plan(shape, space, accelerator=H100_SXM)
print(render_plan_table(candidates, H100_SXM))
```

A sweep over a 1024-GPU space takes well under a second and needs no GPU.

The constraints it encodes are the ones that come from hardware, not from taste:
TP must not cross a node; CP must divide the sequence; PP must divide the depth
and wants several blocks per stage; the global batch must factor across the data
dimension, or the effective batch silently changes and two runs stop being
comparable.

When nothing fits, the table says so and ranks the fixes — raise `cp`, shorten
the clip, get a bigger device, shrink the model — in order of how much each
helps.

### 2. Build the real mesh at the real world size

```python
from avgen.parallel import ParallelDims
from avgen.simulate.world import fake_world

dims = ParallelDims(world_size=1024, dp_shard=-1, context=8, tensor=8)
with fake_world(dims, rank=0) as world:
    mesh = world.require_mesh()
    print(mesh.mesh_dim_names)     # ('dp_shard', 'cp', 'tp')
    print(world.describe())
```

This is a real `DeviceMesh` with real process groups and real submeshes. Rank 0
is the usual choice; simulating a middle rank is useful for checking that
pipeline stage assignment is balanced.

### 3. Apply the plan and see what each rank holds

```python
from avgen.parallel import parallelize
from avgen.simulate.world import simulate_rank

report = simulate_rank(
    dims,
    build=lambda: VideoDiT(config),          # built under meta_init(), so free
    apply_plan=lambda m, d, mesh: parallelize(m, d, mesh).model,
)
print(report["parameters_per_rank"], report["shard_efficiency"])
```

This is the smallest useful end-to-end check: it proves the plan applies cleanly
at the target scale and reports what each rank ends up holding. Run it in CI on
every change to a model or a plan and a whole class of "it worked on 8 GPUs"
regressions stops reaching the cluster.

### 4. Export a trace for a network simulator

The analytical communication model is good enough to rank plans. It is not good
enough to answer questions about the *network*: what happens when a fat tree
oversubscribes, whether a rail-optimised topology beats a flat one, how much a
congested neighbour costs you.

Those need a real network simulator. The industry entry point is
[ASTRA-sim](https://astra-sim.github.io/) driven by an MLCommons
[Chakra](https://mlcommons.org/working-groups/research/chakra/) execution trace,
and PyTorch emits the upstream half natively:

```text
avgen trace  →  PyTorch ET (JSON)
             →  chakra_trace_link    (merge with a Kineto trace for timing)
             →  chakra_converter     (PyTorch ET → Chakra ET protobuf)
             →  ASTRA-sim            (network + system simulation)
```

```python
from avgen.simulate.chakra import capture_execution_trace

with capture_execution_trace("traces/step.et.json") as path:
    trainer.train_step(batch, microbatch_index=0, accumulation=1)
print(path)
```

avgen owns the capture step and documents the rest rather than vendoring a
converter that would rot against Chakra's schema. `TraceArtifacts.conversion_commands()`
prints the exact command line for your world size.

**Capture on rank 0 only** unless you specifically need per-rank divergence: a
full trace of a 1024-rank job is hundreds of gigabytes, and the ranks are nearly
identical by construction.

## Calibrate, or the numbers are fiction

The shipped hardware profiles are plausible starting points, not measurements of
your cluster. They exist so a first simulation runs, not so it is accurate.

**The single most valuable calibration** converts every downstream time estimate
from a guess into a projection:

```bash
./build/all_reduce_perf -b 1G -e 8G -f 2 -g 8     # busbw 372 GB/s
```

```python
from avgen.simulate.comms import calibrate_from_busbw

nvlink = calibrate_from_busbw("NVLink 4", measured_busbw_gbps=372.0, peak_gbps=450.0)
```

`nccl-tests` reports *bus bandwidth*, which already folds in the `2(n-1)/n` ring
factor, so dividing by the hardware peak gives the efficiency term directly.

Two other knobs worth calibrating from one real run:

- **`achieved_fraction`** (default 0.45) — the fraction of peak FLOPs your job
  actually reaches. 0.45 is realistic for a well-tuned large transformer, not
  the 0.8 a naive roofline suggests. The gap is kernel launch gaps, non-matmul
  work, memory-bound norms, and imperfect overlap.
- **`workspace_gib`** (default 2.0) — allocator overhead and kernel workspaces.

```python
from avgen.simulate.memory import calibration_error, measure_memory

print(calibration_error(estimate, measure_memory(model, optimizer, batch)))
```

## The CI gate

This is avgen's headline CI trick, and it runs on every pull request.

`.github/workflows/simulate.yml` runs
`.github/scripts/check_reference_plans.py` at world sizes **8, 64, 512, and
1024**. For each reference plan it:

1. **searches** the whole space and fails if nothing fits any more,
2. **prices the pinned plan** a real job would launch with, and fails if memory
   exceeds the recorded ceiling or predicted MFU or scaling efficiency falls
   below the floor,
3. **builds the real mesh** under `FakeProcessGroup` and asserts the dimension
   order is `(pp, dp_replicate, dp_shard, cp, tp)` — the ordering that keeps
   tensor-parallel traffic inside an NVLink domain, and whose violation produces
   no error at runtime.

Runtime: seconds, on a free CPU runner. No GPU is involved anywhere.

The floors live in `.github/reference_plans.json` and are deliberately loose.
They are a tripwire, not a target — tightening them is a maintainer decision,
made once a real run has calibrated the cost models.

A failure does not mean your change is wrong. It means your change moved a
number other people depend on, and the pull request has to say so. Run it
yourself first:

```bash
make simulate
```

and paste the before/after table into the PR, as the template asks.

## Recipes

**"Will 720p 10-second clips fit anywhere on my cluster?"**

```bash
avgen plan --model 2b --world-size 512 --seq-len 216000
```

**"I changed the model. Did anything stop fitting?"**

```bash
make simulate
```

**"Why is my real MFU half what the simulator predicted?"**

Calibrate `achieved_fraction` and the interconnects from the real run first —
the model may simply be optimistic. If the gap survives calibration, the
prediction assumed overlap you are not getting; profile a step and look at
whether the FSDP all-gather is actually running concurrently with compute.

**"Is my new parallel plan shaped right at 1024 ranks?"**

`simulate_rank` with your `apply_plan`, then check `shard_efficiency` and the
collective counts from `CommDebugMode`.

## Further reading

- [Parallelism](parallelism.md)
- [Scaling to 1000 GPUs](scaling-to-1000-gpus.md)
- [Adding a parallelism plan](../design/adding-a-parallelism-plan.md)
