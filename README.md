<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
  <img src="docs/assets/logo-light.svg" alt="avgen" width="96" height="96">
</picture>

# avgen

**Train video generation models at scale, from pretraining through to RL, on PyTorch primitives and nothing else.**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/pytorch-%E2%89%A5%202.6-ee4c2c.svg)](https://pytorch.org)

[Documentation](https://thesvj.github.io/avgen) ·
[Quickstart](#quickstart) ·
[Examples](examples/) ·
[Features](#what-avgen-gives-you) ·
[Design](#design-commitments) ·
[Contributing](CONTRIBUTING.md)

</div>

---

## What avgen gives you

A complete training stack for video and audio-video diffusion transformers,
built on PyTorch primitives and nothing else.

**Training at scale.** Five parallelism axes compose together in one fixed mesh
order: FSDP2 sharding, HSDP replication, context parallelism, tensor parallelism
and pipeline parallelism. Tensor parallelism always lands inside a node, since
the mesh order takes care of it. You do not have to remember to set it up
correctly in the config.

**Long sequences.** Context parallelism shards the sequence itself. It is the
only axis that reduces the per-rank sequence length, and for video that is
usually the constraint which binds. You get ring attention, sharded rotary
tables, and a loss reduction that stays correct across the shard.

**Knowing the cost before you pay it.** `avgen plan` ranks every viable
parallelism plan for your model and your cluster, and prices each one: memory,
step time, MFU, and the single collective that dominates the step. This is done
analytically, in milliseconds, with no GPU. If you already have a config,
`avgen simulate` does the same for it.

**Resuming anywhere.** Distributed checkpoints reshard when they are loaded, so
a run saved on 512 ranks can resume on 64. Saving happens asynchronously and
overlaps with the next step.

**Post-training, not only pretraining.** LoRA and DoRA are implemented directly
on DTensor, so an adapter can attach to a model that is already sharded five
ways. For reward and preference optimisation there is GRPO and Diffusion-DPO.
Full finetuning is supported through staged recipes with per-stage freezing.

**Generation.** Five samplers, three sigma schedules, classifier-free and
adaptive-projected guidance, and a timestep shift that varies with the sequence
length. That last one matters because a schedule calibrated on images will
under-resolve a long clip.

**One model for many tasks.** Each modality carries its own noise level, and
that independence is enough to collapse text-to-video, image-to-video,
continuation, inpainting and audio-video into a single model with a single
objective. You do not end up maintaining five checkpoints that disagree with
each other.

**Extending it without forking.** Models, metrics, samplers and rewards are all
registries which also read entry points, so your architecture can ship in your
own package and simply be named in a config.

**Starting on a laptop.** Every one of these paths has a CPU smoke path. You can
review a 1024-GPU parallelism plan without touching a GPU.

## What makes video different

The model architecture is not really the difference. The **sequence length** is.

A 10-second 720p clip, passed through an 8×8×4 VAE with 2×2 patching, comes to
roughly 130,000 tokens. Attention is quadratic in that number, and activation
memory is linear in it and then multiplied by the depth. To put it concretely,
for a 14B model at 131k tokens on H100s:

- attention accounts for **74% of the forward FLOPs**, so the usual
  language-model habit of ignoring it will understate the actual work several
  times over
- sharding the parameters further does not help, because the parameters were
  never the problem in the first place
- **context parallelism is therefore not an optimisation. It is the axis that
  makes the job possible at all.**

The rest of avgen follows from taking this seriously.

## Quickstart

```bash
pip install avgen                    # core: torch, numpy, pyyaml, safetensors
pip install 'avgen[codecs,text]'     # + pretrained VAEs and text towers
```

Before you allocate anything, do check what your plan is going to cost:

```bash
avgen plan --world-size 512 --seq-len 65536 --params 2e9 --depth 32 --width 2560
```

```
plan                                    activation_checkpoint  memory_gib  step_seconds  samples/s  mfu     scaling_eff  bottleneck
------------------------------------------------------------------------------------------------------------------------------------
world=512 dp_replicate=8 dp_shard=2 cp=4 tp=8   none            8.82        12.92         39.63      0.448   0.995        tensor_parallel.activation
world=512 dp_shard=32 cp=2 tp=8                 none           14.67        13.05         39.22      0.443   0.985        fsdp.all_gather
world=512 dp_replicate=2 dp_shard=16 cp=2 tp=8  none           14.72        13.05         39.23      0.443   0.985        fsdp.all_gather
```

This needs no GPU and finishes in under a second. Once you are satisfied with
the plan, start the training:

```bash
torchrun --nnodes 64 --nproc-per-node 8 -m avgen.cli.main train \
    --config configs/train/multinode_512.yaml \
    parallel.context=4 parallel.tensor=8
```

Or from Python:

```python
import avgen
from avgen.config.resolve import model_kwargs

config = avgen.load_config("configs/train/multinode_512.yaml")
model = avgen.build_model(config.model.name, model_kwargs(config))

# Check it fits, and where the time goes, before touching the cluster.
dims = avgen.ParallelDims(world_size=512, context=4, tensor=8)
print(avgen.simulate_config(model.model_shape(sequence_length=65_536), dims).render())

# Under torchrun, ParallelDims.from_env() reads the world size for you.
parallel = avgen.parallelize(model, dims)
avgen.Trainer(state, objective, parallel, trainer_config).fit(loader, total_steps=100_000)
```

## The simulator

Normally the way to find out whether a configuration works is to launch the job
and wait for it to run out of memory. The simulator answers those same questions
on a laptop instead:

```python
from avgen.simulate import simulate_config, ModelShape```python
from avgen.simulate import simulate_config, ModelShape
from avgen.parallel import ParallelDims

shape = ModelShape(parameters=2e9, depth=32, width=2560,
                   sequence_length=65_536, num_heads=20, text_tokens=256)
print(simulate_config(shape, ParallelDims(world_size=512, context=8, tensor=2)).render())
```

```
========================================================================
avgen simulation — world=512 dp_shard=32 cp=8 tp=2
========================================================================
  device                H100 SXM (80 GiB)
  parameters            2.00 B
  sequence length       65,536 tokens (4,096 per rank)

MEMORY  14.8 GiB per rank   FITS
    activation              12.58 GiB
    workspace                2.00 GiB

COMPUTE  803.8 ms/microbatch   MFU 42.7%   recompute overhead 0.0%

COMMUNICATION  9.77 GiB/step   exposed 44.1 ms   scaling efficiency 94.8%
    fsdp.all_gather                    106.76 ms
    fsdp.reduce_scatter                106.51 ms
    tensor_parallel.activation           3.89 ms
    context_parallel.ring_kv             3.26 ms

THROUGHPUT  0.848 s/step   2.47 M tokens/s

NOTES
    - attention is 74% of forward FLOPs at 65,536 tokens; context parallelism
      and an efficient attention kernel matter more here than parameter sharding
========================================================================
```

These numbers are not guesswork. Each one comes from PyTorch's own tooling:

| Question | Mechanism |
|---|---|
| Does the plan even apply at 1024 ranks? | `FakeProcessGroup` + `FakeTensorMode`: a real mesh, real sharding, no GPUs |
| How much memory per rank? | closed-form model, calibrated against `MemTracker` / `FSDPMemTracker` |
| Which collectives, from which module? | `CommDebugMode` |
| How long per operator? | `RuntimeEstimator` |
| What does the *network* do? | `ExecutionTraceObserver` → Chakra ET → [ASTRA-sim](https://astra-sim.github.io/) |

Since the whole thing runs in about a second on CPU, it can also run in CI.
`SimulationReport.assert_no_regression()` turns the question "has somebody just
made our 1024-GPU job 20% slower" into an ordinary failing test on the pull
request, which is where you want to find out.

## Design commitments

**Sequence-first.** Dense grids live only at the data and codec boundary.
Everywhere else, what the model touches is a `TokenStream`: shape
`(batch, length, width)`, with explicit per-token coordinates carried alongside.
This matters because context parallelism, mixed-resolution batching and sequence
packing simply cannot be expressed on a 5-D grid.

**Physical coordinates.** Time is carried in seconds and space in latent
pixels, never as indices. Because of this, a model trained at 24 fps and 256px
can sample at 30 fps and 512px: its positional encoding was never told what a
frame index is. The same property is what allows video at 6 fps and audio at
43 fps to share one rotary phase space.

**One model ABI.** Training and sampling both construct a `ModelInput` and read
back a `ModelOutput`. There is no separate inference path that can quietly drift
out of sync with the training one.

**Independent per-modality noise levels.** Text-to-video, image-to-video,
continuation, inpainting, video-to-audio and audio-to-video all collapse into a
single model with a per-sample task label. This works because "keep one stream
clean while the other is noised" is something the representation can express
directly.

**PyTorch itself, not a layer on top of it.** We use `DeviceMesh`, `DTensor`,
`fully_shard`, `parallelize_module`, `context_parallel`,
`torch.distributed.checkpoint` and `torchrun` directly. There is no Ray,
DeepSpeed, Megatron, Accelerate or Lightning anywhere in the dependency tree.
A cluster image needs `torch` plus four small pure-Python packages.

**Correct when the axes are combined.** The RNG varies on the *data* rank only,
so tensor- and context-parallel ranks that hold shards of the same sample will
draw identical noise. Losses reduce over the `dp_cp` mesh rather than the whole
world. Gradient norms are computed across DTensor shards and pipeline stages.
None of these show up in a small test run. They show up on a cluster, after you
have already spent the allocation.

## Parallelism

The five axes are composed in a fixed order, so that tensor-parallel traffic
stays inside a node and pipeline traffic sits furthest apart:

```
mesh = (pp, dp_replicate, dp_shard, cp, tp)
```

| Axis | What it buys | When to reach for it |
|---|---|---|
| `cp` — context | **sequence length** | first, for video; this is the binding constraint |
| `dp_shard` — FSDP2 | parameter + optimizer memory | second |
| `tp` — tensor (+ sequence) | per-layer weights *and* activations | third; never across nodes |
| `dp_replicate` — HSDP | keeps all-gather on NVLink | multi-node, above ~256 GPUs |
| `pp` — pipeline | model depth | last; the bubble is real |

```python
dims = ParallelDims(world_size=1024, dp_shard=32, context=4, tensor=8)
mesh = dims.build_mesh()          # registers dp_shard_cp and dp_cp views too
parallel = parallelize(model, dims, config=ParallelConfig(
    precision=PrecisionConfig(param_dtype="bfloat16", reduce_dtype="float32"),
    activation_checkpoint=ActivationCheckpointConfig(mode="selective_op"),
    compile_blocks=True,
))
```

## Full lifecycle

| Stage | Module | What you get |
|---|---|---|
| **Pretrain** | `avgen.train` | rectified flow, resolution-shifted timesteps, multi-task conditioning, progressive curriculum, sharded EMA |
| **Finetune** | `avgen.finetune` | LoRA/DoRA (DTensor-safe), control adapters, resolution and duration adaptation, staged recipes |
| **RL** | `avgen.rl` | Flow-GRPO (ODE→SDE with exact transition log-probs), MixGRPO windows, group-relative advantage, Diffusion-DPO |
| **Infer** | `avgen.infer` | Euler / Heun / DPM++ 2M / res-multistep, CFG + rescale + APG, modality-aware guidance, context-parallel sampling |
| **Evaluate** | `avgen.eval` | dependency-free metrics, gated learned backends, reports that refuse to compare mismatched pins |
| **Verify** | `avgen.simulate` | plan search, memory, collectives, MFU, Chakra export, CI regression gate |

## Extending it

You can add a model, a metric, a sampler or a timestep law without touching the
core at all:

```python
from avgen.models import register_model

@register_model("my_dit")
class MyDiT(nn.Module):
    def forward(self, inputs: ModelInput) -> ModelOutput: ...
    def tensor_parallel_plan(self, *, sequence_parallel): ...
```

Alternatively, ship it in your own package and advertise it through an entry
point. No fork is needed, and no import is required in the training script:

```toml
# your package's pyproject.toml
[project.entry-points."avgen.models"]
my_dit = "my_package:MyDiT"
```

There are four groups: `avgen.models`, `avgen.metrics`, `avgen.samplers` and
`avgen.rewards`. If a plugin fails to load, it raises an error naming the group,
the entry-point name and its value. We are strict about this because a plugin
that quietly fails to load looks exactly like a typo in a config file, and you
will only find out hours later, on a cluster.

## Status

Version `0.1.0`. The distributed layer, the contracts and the simulator are the
most mature parts as of now. No pretrained weights are being released. Where the
docs quote benchmark numbers, each one is labelled either as a prediction from
the simulator or as an actual measurement. We never mix the two.

## Citing

See [`CITATION.cff`](CITATION.cff).

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
