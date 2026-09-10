# avgen

A PyTorch-native framework for training video and audio-video generation models
at scale — pretraining, finetuning, inference, and RL post-training.

```bash
uv add avgen           # torch, numpy, pyyaml, safetensors. Nothing else.
```

---

## What avgen is

avgen is the training stack for diffusion transformers over video, built on the
assumption that the interesting regime is **long sequences on many GPUs**. A
five-second 720p clip is 72,000 latent tokens. Ten seconds of 1080p is close to
half a million. At those lengths attention is 80–96% of the arithmetic,
activation memory is the binding constraint, and no amount of data parallelism
helps — the sequence itself has to be split.

Three decisions follow from that, and they are what distinguish avgen:

**The model interior is a sequence, not a grid.** Dense 5-D latents exist only
at the data and codec boundary. Everything between the patchifier and the
unpatchifier is a [`TokenStream`](design/sequence-first.md): `(B, L, W)` tokens
plus explicit per-token coordinates in physical units — seconds for time,
latent pixels for space. You cannot shard `(B, C, T, H, W)` along "part of T and
part of H"; you can trivially shard `(B, L, W)` along `L`. That single choice is
what makes context parallelism, variable-resolution batching, and sequence
packing possible at all.

**Distribution is PyTorch, not a framework on top of PyTorch.** DeviceMesh and
DTensor, FSDP2 `fully_shard`, tensor parallel with sequence parallelism, context
parallel over ring attention, pipeline parallel, Distributed Checkpoint,
torchrun and torchelastic. **Ray, DeepSpeed, Megatron, Accelerate, and Lightning
are not dependencies** — Ray appears only as an optional launcher adapter. When
something breaks at 3am on 128 nodes, the stack you debug is the one PyTorch
documents.

**A parallelism plan is validated before it is launched.** avgen ships a
[simulator](guides/simulation.md) that builds a real 1024-rank `DeviceMesh` on
one CPU machine using `FakeProcessGroup`, applies the real parallel plans under
`FakeTensorMode`, and prices the result with closed-form memory, communication,
and compute models. A full search over a 1024-GPU configuration space takes
under a second. It runs in CI on every pull request, and it fails the build when
a reference plan stops fitting or its predicted MFU regresses.

---

## The gap this fills

There is good open-source work on video generation. Almost none of it is a
*training framework for large-scale video*, and the pieces that exist do not
compose.

| | Open weights + inference | LoRA / small finetune | Multi-node pretraining | Context parallel **for training** | Checkpoints that reshard | Plan simulator | Audio-video joint |
|---|---|---|---|---|---|---|---|
| **LTX-Video-Trainer** | uses LTX weights | **yes**, its purpose | no | no | no | no | no |
| **Mochi 1** (+ finetuner) | **yes** | **yes** | no | inference only | no | no | no |
| **Open-Sora** | **yes** | **yes** | **yes** | sequence parallel, via ColossalAI | partial | no | no |
| **FastVideo** | **yes** | distillation | limited | inference / distill focus | no | no | no |
| **torchtitan** | n/a (LLM) | n/a | **yes**, exemplary | **yes** | **yes** | no | n/a |
| **avgen** | consumes any codec | **yes** | **yes** | **yes** | **yes** | **yes** | **yes** |

Read the table as scope, not as a quality judgement — every project in it is
good at what it set out to do, and each one's scope may have moved since this
was written. Check before relying on a cell.

What the row structure says:

- **The video projects are finetuning and inference stacks.** LTX-Video-Trainer
  and the Mochi finetuner exist to adapt an existing checkpoint on one node.
  They are excellent at that and were never trying to be a pretraining
  framework. Neither shards a sequence across ranks during training, because at
  their target scale nothing needs it.
- **FastVideo optimises the other end.** Sliding-tile attention, distillation,
  fast sampling. If your problem is that generation is slow, that is the project
  to read. It is not where you go to train a model from scratch on 512 GPUs.
- **Open-Sora is the closest thing to a full recipe**, and it genuinely does
  multi-node training with sequence parallelism — through ColossalAI. That is a
  real dependency choice with real consequences: the parallelism semantics,
  checkpoint format, and debugging surface are ColossalAI's, not PyTorch's.
- **torchtitan is the architecture avgen follows**, and openly so. Five explicit
  axes, DeviceMesh built in a deliberate order, FSDP2, DCP resharding — avgen's
  parallel layer is torchtitan's design applied to a different problem.
  torchtitan is an LLM pretraining reference; it has no notion of a video
  latent, a flow-matching objective, a resolution bucket, or a codec boundary.

**Nobody ships multi-node FSDP + context-parallel training for video, with
checkpoints that reshard across a changed rank count, and a simulator that tells
you whether the plan fits before you launch it.** That is the gap.

---

## What it looks like

Declare the five axes. `dp_shard=-1` means "use whatever is left over", which is
almost always what you want.

```python
from avgen.parallel import ParallelDims, parallelize

dims = ParallelDims(world_size=1024, dp_shard=-1, context=8, tensor=8)
mesh = dims.build_mesh("cuda")
model = parallelize(model, dims, mesh).model
```

Price it before you launch it:

```bash
avgen plan --model 2b --world-size 1024 --seq-len 72000
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
  -m avgen.cli.main train --config configs/av_2b_720p.yaml parallel.context=8
```

---

## Where to go next

<div class="grid cards" markdown>

- **New here** — [Installation](getting-started/installation.md) →
  [Quickstart](getting-started/quickstart.md) →
  [Your first training run](getting-started/first-training-run.md)

- **Choosing a parallelism plan** — [Parallelism](guides/parallelism.md) is the
  flagship guide: which axis, in what order, and why. Then
  [Context parallelism for video](guides/context-parallel-for-video.md).

- **Making it fit** — [Memory](guides/memory.md), then
  [Simulation](guides/simulation.md) to check the plan, then
  [Scaling to 1000 GPUs](guides/scaling-to-1000-gpus.md).

- **Something is broken** — [Troubleshooting](guides/troubleshooting.md) covers
  NCCL timeouts, OOM triage, loss spikes, dataloader resume mismatch, and mesh
  ordering mistakes.

- **Understanding the design** — [Architecture](design/architecture.md),
  [Sequence-first](design/sequence-first.md),
  [Contracts](design/contracts.md).

- **Extending it** — [Adding a model](design/adding-a-model.md),
  [Adding a parallelism plan](design/adding-a-parallelism-plan.md).

</div>

---

## Install

The core package depends on **torch, numpy, pyyaml, and safetensors, and nothing
else**. Everything heavier is an opt-in extra, so a cluster image stays small and
the steady-state training hot path never imports an optional dependency.

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
[`CITATION.cff`](https://github.com/avgen-project/avgen/blob/main/CITATION.cff).
