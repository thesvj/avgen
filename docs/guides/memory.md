# Memory

For a video diffusion transformer, memory is activations. Everything else is
rounding error once FSDP is on. This guide is about knowing that number before
you launch, and about the order in which to attack it when it is too big.

## The six terms

```python
from avgen.parallel import ParallelDims
from avgen.simulate.memory import ModelShape, estimate_memory

shape = ModelShape(
    parameters=2_684_354_560,
    depth=32,
    width=2048,
    num_heads=16,
    sequence_length=72_000,
    micro_batch_size=1,
    text_tokens=256,
)
dims = ParallelDims(world_size=1024, dp_shard=-1, context=8)

estimate = estimate_memory(shape, dims)
print(estimate.total_gib, estimate.dominant_term())
print(estimate.breakdown_gib())
```

| Term | What it is | Sharded by |
|---|---|---|
| `parameter_bytes` | Weights in compute precision | `dp_shard × cp × tp × pp` |
| `gradient_bytes` | Gradients, same precision | same |
| `optimizer_bytes` | fp32 master + moments (12 bytes/param for AdamW) | same |
| `activation_bytes` | Stored tensors the backward pass needs | `cp × tp` (sequence), `pp` (depth) |
| `gather_bytes` | Transient FSDP all-gather of in-flight blocks | `tp × pp` |
| `workspace_bytes` | Allocator overhead, fragmentation, kernel workspaces | nothing |

`dominant_term()` tells you which one to attack. It is almost always
`activation_bytes`.

## Why activations dominate

For the model above at 72,000 tokens, `cp=8`, `dp_shard=128`, no checkpointing:

| Term | GiB | Share |
|---|---|---|
| Parameters | 0.005 | 0.02% |
| Gradients | 0.005 | 0.02% |
| Optimizer state | 0.03 | 0.1% |
| **Activations** | **22.0** | **88%** |
| FSDP gather peak | 0.3 | 1.2% |
| Workspace | 2.0 | 8% |

Parameters, gradients, and optimizer state together are forty megabytes. FSDP
has already solved that problem completely. What remains is a linear function of
sequence length, and the only two levers on it are `cp` and activation
checkpointing.

### What is actually stored

Per unchecked block, avgen's model counts the tensors backward needs: the block
input, the normalised activations, `q`/`k`/`v`, the attention output, the second
norm, and the two feed-forward intermediates at the expanded width.

**Attention scores are not in that list.** Every kernel avgen uses — flash,
memory-efficient, cuDNN — recomputes the `L × L` score matrix in the backward
pass rather than storing it. At 72,000 tokens an fp16 score matrix would be
9.7 GiB *per head per block*. That single property is what makes long-sequence
video training possible at all; if you write a custom attention that
materialises scores, nothing else in this guide will save you.

## The order to attack it

### 1. Raise `cp`

Linear reduction in activations, and it divides the attention FLOPs too. This is
the only lever that makes the job faster while making it fit. Cap it at
`gpus_per_node` — see
[Context parallelism for video](context-parallel-for-video.md).

### 2. Turn on selective-op activation checkpointing

```python
from avgen.parallel.activation import ActivationCheckpointConfig

ActivationCheckpointConfig(mode="selective_op", save_op_frequency=1)
```

At `cp=8`, 72,000 tokens:

| Policy | Activations | Total | Recompute cost |
|---|---|---|---|
| `none` | 22.0 GiB | 25.0 GiB | 0% |
| `selective_op`, frequency 1 | 9.5 GiB | 12.4 GiB | ~10–15% |
| `selective_op`, frequency 2 | 5.6 GiB | 8.6 GiB | ~20% |
| `full` | 1.8 GiB | 4.7 GiB | ~33% |

`selective_op` keeps the expensive-to-recompute outputs — matmuls, attention —
and drops the cheap elementwise ones. It buys most of the memory of full
checkpointing for a fraction of the recompute, which is why it is the default
recommendation and why `full` is a last resort. Full checkpointing costs you an
entire extra forward pass: a third of your compute, permanently.

### 3. Reduce the micro-batch

Activations are linear in micro-batch size. Going from 2 to 1 halves them.
Gradient accumulation keeps the global batch — and therefore the optimization
trajectory — unchanged:

```python
accumulation = dims.gradient_accumulation_for(global_batch_size=256, local_batch_size=1)
```

This is the cheapest lever, but it also lowers MFU: a smaller micro-batch gives
the collectives less compute to hide behind.

### 4. Raise `tp`

Tensor parallelism shards the per-layer activations too, and with sequence
parallelism it shards the norm and residual paths along the sequence. Stay
inside a node.

### 5. `pp`, or a shorter clip

If none of the above fits, either shard depth or admit that the clip is too long
for this cluster. The plan search says so explicitly when nothing fits, and
ranks the options.

## Keep headroom

`fits_in()` defaults to keeping **10% free**, and that margin is not
superstition:

- the CUDA caching allocator fragments, and a long run's peak drifts up;
- cuDNN and NCCL workspaces sit outside PyTorch's accounting;
- one step's bucket is always slightly larger than the others.

A job that peaks at 99% of device memory will OOM — not on step 1, which would
be convenient, but on step 40,000.

```python
estimate.fits_in(80.0, headroom=0.10)
```

## Calibrate the model against reality

The closed-form estimate is a model. Check it once on real hardware and every
downstream projection improves:

```python
from avgen.simulate.memory import calibration_error, measure_memory

measured = measure_memory(model, optimizer, batch)
print(calibration_error(estimate, measured))
```

If the error is above roughly 15%, the usual causes are a custom attention that
stores scores, a `workspace_gib` that is too low for your kernels, or a model
whose blocks are not uniform.

## Diagnosing a real OOM

See [Troubleshooting](troubleshooting.md#oom-triage-order) for the full
procedure. In short:

1. **Read the allocator summary in the traceback**, not just the failing
   allocation. `reserved` far above `allocated` means fragmentation, which is a
   different problem with a different fix.
2. **Ask the simulator what it expected.** A large gap between prediction and
   reality points at something structural, not at a tuning knob.
3. **Check whether the peak is flat.** Memory that climbs every step is a leak —
   usually a metric tensor accumulated without `detach()`, or a reference held
   across steps.
4. **Only then** turn knobs, in the order above.

## Reading the report at runtime

`avgen.telemetry.MemoryReporter` logs peak, reserved, and fragmentation, and
warns when the job is close enough to the ceiling that a fluctuation will kill
it. Watch `reserved - allocated`: a large and growing gap is fragmentation, and
the fix is usually a fixed bucket shape rather than a smaller model.

## Further reading

- [Parallelism](parallelism.md)
- [Simulation](simulation.md)
- [Troubleshooting](troubleshooting.md)
