# avgen

A PyTorch-native framework for training video and audio-video generation models
at scale: pretraining, finetuning, inference and RL post-training.

```bash
uv add avgen           # torch, numpy, pyyaml, safetensors. Nothing else.
```

---

## What avgen is

avgen is a training stack for diffusion transformers over video, written on the
assumption that the regime worth caring about is **long sequences spread over
many GPUs**. A five-second 720p clip works out to 72,000 latent tokens. Ten
seconds of 1080p comes to nearly half a million. At those lengths, attention
takes up 80–96% of the arithmetic and activation memory becomes the binding
constraint. Adding more data parallelism does not help at all here. The sequence
itself has to be split.

Three decisions follow from this, and they are what set avgen apart:

**The interior of the model is a sequence, not a grid.** Dense 5-D latents
appear only at the data and codec boundary. Everything sitting between the
patchifier and the unpatchifier is a
[`TokenStream`](design/sequence-first.md): `(B, L, W)` tokens, along with
explicit per-token coordinates kept in physical units, seconds for time and
latent pixels for space. The reason this matters is simple enough. You cannot
shard `(B, C, T, H, W)` along "part of T and part of H", whereas sharding
`(B, L, W)` along `L` is trivial. This one choice is what makes context
parallelism, variable-resolution batching and sequence packing possible in the
first place.

**Distribution is PyTorch itself, not a framework sitting on top of PyTorch.**
DeviceMesh and DTensor, FSDP2 `fully_shard`, tensor parallel with sequence
parallelism, context parallel over ring attention, pipeline parallel,
Distributed Checkpoint, torchrun and torchelastic. **Ray, DeepSpeed, Megatron,
Accelerate and Lightning are not dependencies in any form.** The benefit shows
up when something breaks at 3am on 128 nodes: the stack you have to debug is the
one PyTorch documents.

**A parallelism plan is validated before it is ever launched.** avgen ships a
[simulator](guides/simulation.md) which builds a real 1024-rank `DeviceMesh` on
a single CPU machine using `FakeProcessGroup`, applies the actual parallel plans
under `FakeTensorMode`, and then prices the result using closed-form memory,
communication and compute models. A full search over a 1024-GPU configuration
space finishes in under a second. It runs in CI on every pull request, and it
fails the build if a reference plan stops fitting or its predicted MFU
regresses.

---

## What avgen gives you

A complete training stack for video and audio-video diffusion transformers,
built on PyTorch primitives and nothing else.

### Train at scale

Five parallelism axes compose together in one fixed mesh order: FSDP2 sharding,
HSDP replication, context parallelism, tensor parallelism and pipeline
parallelism. Tensor parallelism always lands inside a node, since the mesh order
takes care of it. You do not have to remember to get it right in the config.

### Handle long sequences

Context parallelism shards the sequence itself. It is the only axis that reduces
the per-rank sequence length, and for video that is usually the constraint which
binds. You get ring attention, sharded rotary tables, and a loss reduction that
stays correct across the shard.

### Know the cost before you pay it

`avgen plan` ranks every viable parallelism plan for your model and cluster and
prices each one: memory, step time, MFU, and the single collective that
dominates the step. Analytically, in milliseconds, with no GPU. `avgen simulate`
does the same for a config you already have.

### Resume anywhere

Distributed checkpoints reshard on load, so a run saved on 512 ranks resumes on
64. Saving is asynchronous and overlaps the next step.

### Post-train, not just pretrain

LoRA and DoRA implemented natively on DTensor, so an adapter attaches to a model
already sharded five ways without gathering it first. GRPO and Diffusion-DPO for
reward and preference optimisation. Full finetuning stages with per-stage
freezing.

### Generate

Five samplers, three sigma schedules, classifier-free and adaptive-projected
guidance, and a timestep shift that varies with the sequence length. That last
point matters, because a schedule calibrated on images will under-resolve a long
clip.

### One model, many tasks

Each modality carries its own noise level, and that independence is enough to
collapse text-to-video, image-to-video, continuation, inpainting and audio-video
into one model with one objective. You are not left maintaining five separate
checkpoints that disagree with each other.

### Extend it without forking

Models, metrics, samplers and rewards are all registries which also read entry
points, so your architecture can ship in your own package and simply be named in
a config.

### Run it on a laptop first

Every one of these paths has a CPU smoke path. Reviewing a 1024-GPU parallelism
plan needs no GPU at all.

## What it looks like

First declare the five axes. Setting `dp_shard=-1` means "use whatever is left
over", which is almost always what you want:

```python
from avgen.parallel import ParallelDims, parallelize

dims = ParallelDims(world_size=1024, dp_shard=-1, context=8, tensor=8)
mesh = dims.build_mesh("cuda")
model = parallelize(model, dims, mesh=mesh).model
```

Then price it before launching:

```bash
avgen plan --world-size 1024 --seq-len 72000 --params 2e9 --depth 32 --width 2560
```

```text
plan                               activation_checkpoint  memory_gib  step_seconds  mfu     bottleneck
---------------------------------- ---------------------- ----------- ------------- ------- -----------------------
world=1024 dp_shard=128 cp=8       selective_op           12.4        1.53          0.42    fsdp_all_gather
world=1024 dp_shard=64 cp=8 tp=2   selective_op           8.6         1.61          0.40    tp_all_reduce
world=1024 dp_shard=256 cp=4       selective_op           23.1        1.58          0.41    cp_ring_attention
```

Then run it:

```bash
torchrun --nnodes 128 --nproc-per-node 8 \
  -m avgen.cli.main train --config configs/train/multinode_512.yaml parallel.context=8
```

---

## Where to go next

<div class="grid cards" markdown>

- **New here?** Start with [Installation](getting-started/installation.md) →
  [Quickstart](getting-started/quickstart.md) →
  [Your first training run](getting-started/first-training-run.md)

- **Choosing a parallelism plan.** [Parallelism](guides/parallelism.md) is the
  main guide: which axis to use, in what order, and why. After that, read
  [Context parallelism for video](guides/context-parallel-for-video.md).

- **Making it fit.** Read [Memory](guides/memory.md), then use
  [Simulation](guides/simulation.md) to check the plan, then
  [Scaling to 1000 GPUs](guides/scaling-to-1000-gpus.md).

- **Something is broken.** [Troubleshooting](guides/troubleshooting.md) covers
  NCCL timeouts, OOM triage, loss spikes, dataloader resume mismatch and mesh
  ordering mistakes.

- **Understanding the design.** See [Architecture](design/architecture.md),
  [Sequence-first](design/sequence-first.md) and
  [Contracts](design/contracts.md).

- **Extending it.** See [Adding a model](design/adding-a-model.md) and
  [Adding a parallelism plan](design/adding-a-parallelism-plan.md).

</div>

---

## Install

The core package depends on **torch, numpy, pyyaml and safetensors, and nothing
else**. Anything heavier is an opt-in extra. That keeps a cluster image small,
and it means the steady-state training hot path never imports an optional
dependency.

```bash
uv add avgen                    # core
uv add 'avgen[text,codecs]'     # frozen text towers and VAEs
uv add 'avgen[all]'             # everything except fault-tolerance
```

Requires Python ≥ 3.11 and `torch >= 2.6`. See
[Installation](getting-started/installation.md) for what each extra pulls in and
when you need it.

---

## License and citation

Apache-2.0. If avgen is useful in published work, see
[`CITATION.cff`](https://github.com/thesvj/avgen/blob/main/CITATION.cff).
