<div align="center">

# avgen

**Train video generation models at scale — pretraining through RL — on PyTorch primitives and nothing else.**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/pytorch-%E2%89%A5%202.6-ee4c2c.svg)](https://pytorch.org)

[Documentation](https://avgen-project.github.io/avgen) ·
[Quickstart](#quickstart) ·
[Why this exists](#the-gap) ·
[Design](#design-commitments) ·
[Contributing](CONTRIBUTING.md)

</div>

---

## The gap

Every open video-generation release ships inference. Almost none ship the thing
that produced the weights.

| Project | Inference | Finetune | **Multi-node pretraining** | Long-sequence (CP) | Resharding checkpoints | RL |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| LTX-Video | ✅ | LoRA, 1 node | ❌ | ❌ | ❌ | ❌ |
| Mochi 1 | ✅ | LoRA, 1 GPU | ❌ | inference only | ❌ | ❌ |
| HunyuanVideo / Wan | ✅ | partial | ❌ | partial | ❌ | ❌ |
| Open-Sora | ✅ | ✅ | partial | ✅ | ❌ | ❌ |
| torchtitan | — | — | ✅ | ✅ | ✅ | ❌ |
| **avgen** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

torchtitan solved the distributed half — for language models. The video half has
no equivalent: no project gives you FSDP2 + tensor + context parallelism over a
diffusion transformer, checkpoints that reshard onto a different rank count, a
data pipeline that buckets by resolution and duration, and a way to find out
whether your plan fits *before* you spend the allocation.

avgen is that stack.

## What makes video different

Not the model. The **sequence length**.

A 10-second 720p clip through an 8×8×4 VAE with 2×2 patching is ~130,000 tokens.
Attention is quadratic in that; activation memory is linear in it and multiplied
by depth. Concretely, for a 14B model at 131k tokens on H100s:

- attention is **74% of forward FLOPs** — the language-model shortcut of ignoring it understates the work several-fold
- no amount of parameter sharding helps, because parameters were never the problem
- **context parallelism is not an optimisation, it is the enabling axis**

Everything in avgen follows from taking that seriously.

## Quickstart

```bash
pip install avgen                    # core: torch, numpy, pyyaml, safetensors
pip install 'avgen[codecs,text]'     # + pretrained VAEs and text towers
```

**Before you allocate anything**, ask what your plan costs:

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

No GPU needed. Under a second. Then train:

```bash
torchrun --nnodes 64 --nproc-per-node 8 -m avgen.cli.main train \
    --config configs/train/multinode_512.yaml \
    parallel.context=4 parallel.tensor=8
```

Or from Python:

```python
import avgen

dims  = avgen.ParallelDims.from_env(context=4, tensor=8)
model = avgen.build_model("video_dit", avgen.load_config("configs/model/dit_2b.yaml").model)

# Check it fits, and where the time goes, before touching the cluster.
print(avgen.simulate_config(model.model_shape(sequence_length=65_536), dims).render())

parallel = avgen.parallelize(model, dims)
avgen.Trainer(state, objective, parallel, config).fit(loader, total_steps=100_000)
```

## The simulator

The feature that separates avgen from a repository of good intentions.

Every question you would normally answer by launching a job and watching it OOM,
answered on a laptop:

```python
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

It is not a guess. Built on PyTorch's own tooling:

| Question | Mechanism |
|---|---|
| Does the plan even apply at 1024 ranks? | `FakeProcessGroup` + `FakeTensorMode` — a real mesh, real sharding, no GPUs |
| How much memory per rank? | closed-form model, calibrated against `MemTracker` / `FSDPMemTracker` |
| Which collectives, from which module? | `CommDebugMode` |
| How long per operator? | `RuntimeEstimator` |
| What does the *network* do? | `ExecutionTraceObserver` → Chakra ET → [ASTRA-sim](https://astra-sim.github.io/) |

And because it runs in a second on CPU, it runs in CI:
`SimulationReport.assert_no_regression()` turns "did someone just make our
1024-GPU job 20% slower" into a failing test on the pull request.

## Design commitments

**Sequence-first.** Dense grids exist at the data and codec boundary and nowhere
else. Everything a model touches is a `TokenStream` — `(batch, length, width)`
plus explicit per-token coordinates. Context parallelism, mixed-resolution
batching, and sequence packing are all inexpressible on a 5-D grid.

**Physical coordinates.** Time in seconds, space in latent pixels — never
indices. A model trained at 24 fps / 256px can sample at 30 fps / 512px because
its positional encoding was never told what a frame index was. It is also what
lets video at 6 fps and audio at 43 fps share one rotary phase space.

**One model ABI.** Training and sampling both build a `ModelInput` and read a
`ModelOutput`. No second inference path to drift out of sync.

**Independent per-modality noise levels.** Text-to-video, image-to-video,
continuation, inpainting, video-to-audio and audio-to-video collapse into *one*
model with a per-sample task label, because "one stream clean while the other is
noised" is directly expressible.

**PyTorch, not a layer over it.** `DeviceMesh`, `DTensor`, `fully_shard`,
`parallelize_module`, `context_parallel`, `torch.distributed.checkpoint`,
`torchrun`. No Ray, no DeepSpeed, no Megatron, no Accelerate, no Lightning in
the dependency tree. A cluster image needs `torch` and four small pure-Python
packages.

**Correct under composition.** RNG varies on the *data* rank only, so tensor-
and context-parallel ranks holding shards of one sample draw identical noise.
Losses reduce over the `dp_cp` mesh, not the world. Gradient norms are computed
across DTensor shards and pipeline stages. These are the bugs that are invisible
until they are expensive.

## Parallelism

Five axes, composed in a fixed order so tensor-parallel traffic lands inside a
node and pipeline traffic is furthest apart:

```
mesh = (pp, dp_replicate, dp_shard, cp, tp)
```

| Axis | What it buys | When to reach for it |
|---|---|---|
| `cp` — context | **sequence length** | first, for video — it is the binding constraint |
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
| **Evaluate** | `avgen.eval` | dependency-free metrics, gated learned backends, reports that *refuse* to compare mismatched pins |
| **Verify** | `avgen.simulate` | plan search, memory, collectives, MFU, Chakra export, CI regression gate |

## Extending it

Add a model, a metric, a sampler, or a timestep law without touching core:

```python
from avgen.models import register_model

@register_model("my_dit")
class MyDiT(nn.Module):
    def forward(self, inputs: ModelInput) -> ModelOutput: ...
    def tensor_parallel_plan(self, *, sequence_parallel): ...
```

Or ship it in your own package via an `avgen.models` entry point — no fork
required. Same for `avgen.metrics`, `avgen.samplers`, `avgen.rewards`.

## Status

`0.1.0`. The distributed layer, contracts, and simulator are the mature parts.
No pretrained weights are released. Benchmarks in the docs are *predictions*
from the simulator where marked, and measurements where marked — the two are
never mixed.

## Citing

See [`CITATION.cff`](CITATION.cff).

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
