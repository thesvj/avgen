# Architecture

avgen is organised around one boundary and one ABI.

The **boundary** is where dense grids become sequences. The **ABI** is
`forward(ModelInput) -> ModelOutput`. Everything else is a subsystem that can be
replaced without touching the others.

## The layers

```text
        configs (YAML + dotted overrides)          avgen.config
                        │
                        ▼
   ┌──────────────── avgen.cli ────────────────┐
   │  train · generate · simulate · plan · eval │
   └────────────────────┬───────────────────────┘
                        ▼
   data ──► codecs ──► ┃ PATCHIFIER ┃ ──► model ──► objective ──► optimizer
   (dense grids)       ┃ (boundary) ┃    (TokenStream everywhere)
                        │
                        ├── avgen.parallel   (mesh, FSDP2, TP, CP, PP)
                        ├── avgen.checkpoint (DCP, resharding)
                        ├── avgen.telemetry  (metrics, throughput, memory)
                        └── avgen.simulate   (price a plan before running it)
```

Dependencies point inward. `avgen.core` imports nothing from the rest of avgen;
`avgen.parallel` imports `core`; everything else imports both. There is no cycle
anywhere, which is what lets eight people work on eight subsystems at once
against a written contract.

## `avgen.core` — the frozen centre

| Module | What it owns |
|---|---|
| `tokens` | `TokenStream`, `PatchLayout`, `TextContext` |
| `patchify` | The grid ↔ sequence boundary |
| `model_input` | `ModelInput` / `ModelOutput` — the model ABI |
| `batch` | `MediaBatch`, `MediaBatchSpec`, `ConditionMode` |
| `tensors` | `TensorBundle`, dtype and shape specs |
| `rng` | `RNGStreams` — separated, rank-aware generators |
| `state` | `Stateful`, `TrainState`, and the LR/EMA/cursor protocols |
| `metrics` | `StepMetrics` — on-device accumulation, one sync per log |

These are frozen. They are what every other subsystem imports, so changing one
is a governance act, not a refactor — see
[Contracts](contracts.md).

Everything here is a `@dataclass(frozen=True, slots=True)` with validation in
`__post_init__`. Frozen because a training step that mutates its input is a bug
that reproduces only under retry; `slots` because these objects are created per
batch and per block.

## `avgen.parallel` — five axes, one mesh

`ParallelDims` validates the degrees and builds the mesh in the order
`(pp, dp_replicate, dp_shard, cp, tp)`. `parallelize()` applies the
transformations in the only order that works: tensor parallel → activation
checkpointing → `torch.compile` → FSDP2.

The subsystem is deliberately thin. It composes PyTorch's own primitives —
`DeviceMesh`, `DTensor`, `fully_shard`, `parallelize_module`, the
context-parallel APIs — rather than reimplementing them. When something breaks
at 3am on 128 nodes, the stack you debug is the one PyTorch documents.

See [Parallelism](../guides/parallelism.md).

## `avgen.simulate` — the unusual one

Most frameworks let you discover at launch time that a plan does not fit. avgen
prices the plan first, on a CPU, in under a second, and gates the CI on it.

| Module | What it answers |
|---|---|
| `world` | Does the plan *apply* at 1024 ranks? What does each rank hold? |
| `memory` | What is the per-rank peak, broken down by term? |
| `comms` | What collectives run, how big, on which fabric tier, how much hides? |
| `compute` | How many FLOPs, how long, what MFU? |
| `plan` | Of every valid factorisation, which fit, and which is fastest? |
| `chakra` | Export a trace for ASTRA-sim network simulation. |

See [Simulation](../guides/simulation.md).

## The training subsystems

| Package | Responsibility |
|---|---|
| `models` | `VideoDiT`, `AVDiT`, and the model registry |
| `train` | Objective, timestep sampling, conditioning, optimizer, schedule, EMA, trainer |
| `data` | Sources, buckets, shards, the loader |
| `checkpoint` | DCP save/load with resharding, export |
| `telemetry` | Loggers, throughput, memory reporting, profiling |
| `infer` | Samplers, guidance, the generation pipeline |
| `codecs` | Video/audio codecs and text encoders, all optional-dep-gated |
| `finetune` | LoRA, DoRA, freezing, control adapters |
| `rl` | Flow-GRPO, DPO, reward registry |
| `eval` | Metrics and reports |
| `config` | Dataclass + YAML, dotted overrides |
| `cli` | The `avgen` command |

## Registries, not inheritance

Models, timestep samplers, schedules, samplers, rewards, and metrics are all
registry-based:

```python
@register_model("video_dit")
class VideoDiT(nn.Module): ...
```

The point is that adding one is a **pure addition**. If adding a model requires
editing `avgen.core`, either the registry is missing a hook or the change
belongs somewhere else. That is a design rule, not a preference: the frozen
centre only stays stable if extension never touches it.

## Configuration

Dataclasses plus YAML plus dotted CLI overrides. **No Hydra, no OmegaConf, no
Pydantic.**

```bash
avgen train --config configs/av_2b_720p.yaml train.lr=1e-4 parallel.context=8
```

The reasoning: a config system whose semantics you have to learn is a config
system that will surprise you at 3am. A frozen dataclass validates in
`__post_init__`, type-checks under mypy, and has exactly one way to be wrong.
`config_diff(a, b)` answers "what actually differs between these two runs",
which is the question you ask most often and the one an interpolating config
system makes hardest.

The resolved config is written into the run directory, so a run is reproducible
from its own output rather than from whatever the YAML says today.

## Optional dependencies

The core package depends on **torch, numpy, pyyaml, safetensors**. Everything
else is an extra, imported inside the function that needs it, raising a
`RuntimeError` naming the extra when missing.

This is enforced by a CI job that installs the core only and fails if any
optional package appears in `sys.modules` after `import avgen`. It sounds
pedantic; it is the difference between a 400 MB cluster image and a 4 GB one,
multiplied by every node in the job.

## What avgen deliberately does not do

- **No custom kernels.** avgen composes PyTorch's attention backends. A custom
  kernel is a maintenance liability and a portability problem, and the framework
  is not where the win is.
- **No distributed runtime of its own.** torchrun and torchelastic. Ray is an
  optional launcher adapter, not a dependency.
- **No model zoo.** avgen trains models; it does not host weights.
- **No data.** avgen reads shards you produced.

## Further reading

- [Sequence-first](sequence-first.md) — the central representation decision.
- [Contracts](contracts.md) — what is frozen and why.
- [Adding a model](adding-a-model.md)
