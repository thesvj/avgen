# Quickstart

Five minutes, one machine, no GPU required for the first three sections.

## 1. Look at a token stream

Everything in avgen's model interior is a `TokenStream`. Understanding it is
most of understanding the framework.

```python
import torch

from avgen.core.patchify import GridPatchifier

# A tiny latent video: batch 2, 4 channels, 8 latent frames, 16x16 latent pixels.
latents = torch.randn(2, 4, 8, 16, 16)

patchifier = GridPatchifier(patch_frames=1, patch_height=2, patch_width=2)
layout = patchifier.layout_for(tuple(latents.shape))
print(layout.grid, layout.num_tokens, layout.patch_dim)
# (8, 8, 8) 512 16
```

The grid `(8, 8, 8)` collapses into a single sequence of 512 tokens, each 16
features wide. What you hand `to_tokens` is metadata about the *grid*: a
timestamp per frame, in seconds, and a validity mask per latent.

```python
# Physical time, in seconds. Eight frames at 8fps -> 0.000 .. 0.875.
times = torch.arange(layout.frames).float().div(8.0).expand(2, layout.frames)
mask = torch.ones(2, layout.frames, layout.height, layout.width, dtype=torch.bool)

stream = patchifier.to_tokens(
    latents,
    positions=times,
    mask=mask,
    noise_level=torch.zeros(2),
)
print(stream.tokens.shape, stream.length, stream.width)
# torch.Size([2, 512, 16]) 512 16
```

Coordinates come out carried explicitly, in **physical units**:

```python
print(stream.coords.shape, stream.coords[0, 0], stream.coords[0, -1])
# torch.Size([2, 512, 3])  tensor([0., 0., 0.])  tensor([ 0.8750, 14., 14.])
```

The last token sits at 0.875 seconds, latent row 14, latent column 14 — seconds
and latent pixels, not indices. (Row 14, not 7, because a 2x2 patch is addressed
by the latent pixel it starts at, so the model can extrapolate to a higher
resolution at the same pixel pitch.) That is what lets a shuffled, packed, or
context-parallel sharded sequence still know where each token came from.

```python
# The round trip is exact.
assert torch.allclose(patchifier.to_grid(stream), latents)
```

## 2. Choose a parallelism plan, without a cluster

```python
from avgen.parallel import ParallelDims

dims = ParallelDims(world_size=1024, dp_shard=-1, context=8, tensor=8)
print(dims.describe())
# world=1024 dp_shard=16 cp=8 tp=8
print(dims.dp_size, dims.sequence_shard_size, dims.model_shard_size)
```

`dp_shard=-1` means "use whatever is left over after the other axes": 1024 /
(8 × 8) = 16. Getting the arithmetic wrong is the most common configuration
error at scale, so avgen does it for you and validates the result.

## 3. Price the plan before you launch it

```python
from avgen.simulate.compute import H100_SXM
from avgen.simulate.memory import ModelShape
from avgen.simulate.plan import SearchSpace, render_plan_table, search_parallel_plan

# 2.7B DiT, five seconds of 720p at 16fps: 20 x 45 x 80 = 72,000 tokens.
shape = ModelShape(
    parameters=2_684_354_560,
    depth=32,
    width=2048,
    num_heads=16,
    sequence_length=72_000,
    micro_batch_size=1,
    text_tokens=256,
)

candidates = search_parallel_plan(
    shape,
    SearchSpace(world_size=1024, gpus_per_node=8, max_context=16),
    accelerator=H100_SXM,
)
print(render_plan_table(candidates, H100_SXM))
```

This enumerates every valid factorisation of 1024 ranks, prices each one for
memory, communication, and compute, discards what does not fit, and ranks the
rest. It takes well under a second and needs no GPU.

The same thing from the shell:

```bash
avgen plan --model 2b --world-size 1024 --seq-len 72000
```

If nothing fits, the table says so and tells you which lever to pull first —
context parallelism, almost always, because sequence length is what binds.

## 4. Train on synthetic data

`SyntheticSource` is deterministic and needs no optional dependency, which makes
it the right way to check that a plan actually runs before you point it at real
data.

```bash
avgen train --config configs/train/smoke_cpu.yaml train.steps=20
```

Or in Python:

```python
from avgen.config import load_config

config = load_config("configs/train/smoke_cpu.yaml", overrides=["train.steps=20"])
```

Expect a loss that decreases and nothing else — the point is that the plumbing
works end to end: data → patchifier → model → flow-matching objective →
optimizer → checkpoint.

## 5. Single-node, multi-GPU

```bash
torchrun --standalone --nproc-per-node 8 \
  -m avgen.cli.main train --config configs/train/node_8gpu.yaml \
  parallel.context=4 parallel.dp_shard=2
```

Overrides use dotted paths and are applied on top of the YAML. Everything
`ParallelDims` accepts is settable this way, so you can sweep a plan without
editing a file.

## 6. Generate

```bash
avgen generate \
  --checkpoint runs/av_2b_480p/step_10000 \
  --prompt "a paper boat drifting down a rain gutter, close up" \
  --steps 30 --guidance 4.5 --seed 0
```

Inference shares `ModelInput` with training — the same tokens, the same
coordinates, the same condition modes. There is no separate inference-time model
definition to drift out of sync.

## Next

- [Your first training run](first-training-run.md) — a real run, end to end,
  including what the log lines mean.
- [Parallelism](../guides/parallelism.md) — which axis to reach for, in which
  order, and why.
- [Simulation](../guides/simulation.md) — what the simulator is exact about and
  what it only estimates.
