# Checkpointing

avgen checkpoints with PyTorch Distributed Checkpoint (DCP). The property that
matters, and the reason DCP is worth its complexity:

> **A checkpoint saved at one rank count and parallelism plan can be loaded at a
> different one.**

Save on 512 ranks with `cp=8`, resume on 1024 with `cp=4`. Save with `tp=1`,
resume with `tp=8`. Nothing is reshaped by hand, because DCP stores tensors with
their sharding metadata rather than as a flat blob assembled by rank 0.

## Save and load

```python
from avgen.checkpoint import load, save

save("runs/exp/step_2000", state, parallel=parallel, async_save=True)
load("runs/exp/step_2000", state, parallel=parallel)
```

`state` is a `TrainState`: model, optimizer, RNG streams, LR schedule, EMA, step
counters, and anything in `extras` that implements `Stateful`. DCP saves it
without knowing what any of it is — which is why `Stateful` is a protocol and
not a base class.

## What must be in the checkpoint

A checkpoint that restores the model but not the rest is not a resumable
checkpoint. All of these change the trajectory:

| | Why it matters |
|---|---|
| Model parameters | Obviously |
| Optimizer state | AdamW moments; losing them is a warm restart, not a resume |
| RNG streams | Otherwise the same sample gets different noise after resume |
| LR schedule position | Otherwise the schedule restarts and the LR jumps |
| EMA weights | Otherwise your evaluation model is gone |
| `samples_seen`, `tokens_seen`, `step`, `epoch` | Progress accounting |
| Data cursor | Otherwise you replay data — see [Data pipeline](data-pipeline.md) |

Anything you add to the training loop that carries state goes in `extras` and
implements `Stateful`. If it is not in the checkpoint, it silently resets.

## Async save

```python
save(path, state, parallel=parallel, async_save=True)
```

The tensors are copied to host memory — which blocks briefly — and the write
proceeds in the background while training continues. At 512 ranks writing ~40 GB
to a shared filesystem, a synchronous save stalls the whole job for minutes,
every save.

Two things to watch: the next save must not start before the previous one
finishes, and a save in flight when the job dies leaves an incomplete directory.
`CheckpointManager` handles both, and retention:

```python
from avgen.checkpoint import CheckpointManager

manager = CheckpointManager(root="runs/exp", keep_last_n=3, keep_every=10_000)
```

Keep the last few for crash recovery and a sparse history for archaeology.
Keeping everything fills the filesystem; keeping only the last one means a
corrupted save loses the run.

## Resume

```bash
avgen train --config configs/av_2b_720p.yaml checkpoint.resume=runs/exp
```

Pointing at the run directory picks up `latest`. Pointing at a specific step
directory loads that step.

**Verify the resume.** The first loss after resuming should match the last loss
before saving, to within bf16 noise. If it jumps, something did not restore —
see [Troubleshooting](troubleshooting.md#dataloader-resume-mismatch).

A checkpoint you have never restored is not a checkpoint. Test the path on a
small run before you depend on it at scale.

## Export for release

DCP is a training format: sharded, plan-aware, not something an inference user
should have to understand.

```python
from avgen.checkpoint import export_huggingface, export_safetensors

export_safetensors("release/model.safetensors", model, dtype=torch.bfloat16)
export_huggingface("release/hf", model)
```

`safetensors` is the default for a reason: it executes no code on load. A pickle
checkpoint from an untrusted source is remote code execution by design, which is
why avgen never reaches for `torch.load` with `weights_only=False` on a file the
user did not explicitly opt into (see
[`SECURITY.md`](https://github.com/avgen-project/avgen/blob/main/SECURITY.md)).

Export the EMA weights, not the raw ones, unless you have a reason not to.

## Inspect and convert

```bash
avgen checkpoint inspect runs/exp/step_2000
avgen checkpoint convert runs/exp/step_2000 --to safetensors --out release/
avgen checkpoint export  runs/exp/latest --format huggingface --out release/hf
```

`inspect` prints the tensor inventory, shard layout, and the pinned config — the
fastest way to answer "what plan produced this, and does it contain an EMA".

## Failure modes

**Save times out one rank.** A slow filesystem can push a synchronous save past
the NCCL collective timeout, and the failure looks like a hang, not an I/O
problem. Use async save; if it still happens, the filesystem cannot support the
job.

**Resume at a different rank count changes the loss.** DCP reshards the tensors
correctly, but the *data* assignment also changes, because `data_rank` is
different. That is expected and not a bug — the run is no longer bitwise
comparable to the pre-resume run. Note it in the run log.

**EMA missing after resume.** The EMA must be in `TrainState.ema` before `load`
is called; DCP restores into an existing structure, it does not conjure one.

**A partially written directory.** Only `latest` is treated as authoritative,
and it is updated after the write completes.

## Further reading

- [Scaling to 1000 GPUs](scaling-to-1000-gpus.md)
- [Data pipeline](data-pipeline.md)
