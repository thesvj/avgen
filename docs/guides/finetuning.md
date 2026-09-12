# Finetuning

Finetuning is the same training stack with fewer trainable parameters. The
objective, the parallelism, the checkpoint format, and the data pipeline are
unchanged — which is the point: a LoRA run and a pretraining run differ by a
config section, not by a codebase.

## LoRA and DoRA

```python
from avgen.finetune import LoRAConfig, apply_lora

model = apply_lora(
    model,
    LoRAConfig(
        rank=64,
        alpha=64,
        dropout=0.0,
        targets=("attention.q_proj", "attention.v_proj", "feed_forward.gate_proj"),
        use_dora=False,
    ),
)
```

Rank 32–64 is the useful range for a 2–14B video DiT. Below 16 the adapter
cannot represent much beyond a style shift; above 128 you are close enough to
full finetuning that you should compare against it before paying the complexity.

`alpha` scales the update by `alpha / rank`. Setting `alpha = rank` keeps the
effective learning rate stable when you sweep rank, which is usually what you
want.

**DoRA** (`use_dora=True`) decomposes the update into magnitude and direction.
It typically tracks full finetuning more closely at the same rank, at a modest
throughput cost. Try LoRA first; reach for DoRA when LoRA underfits.

### Which modules to target

Attention projections first, feed-forward second. Targeting the norms or the
patch embedding rarely helps and makes merging awkward. If motion quality is the
problem rather than appearance, include the temporal attention path — for many
video DiTs that is where motion actually lives.

## Freezing

```python
from avgen.finetune import freeze_except

freeze_except(model, patterns=("blocks.28", "blocks.29", "blocks.30", "blocks.31"))
```

Last-blocks-only finetuning is a reasonable middle ground between LoRA and full
finetuning: more capacity than an adapter, far less optimizer state than the
whole model.

Note the memory consequence. Frozen parameters need no gradients and no
optimizer state, so the FSDP memory picture changes: parameters still shard,
but the AdamW state shrinks proportionally to the trainable fraction. Tell the
simulator the trainable count, not the total, or it will over-predict.

## Merge, save, load

```python
from avgen.finetune import load_adapter, merge_lora, save_adapter

save_adapter("runs/style/adapter", model)     # small; ship this
load_adapter("runs/style/adapter", model)
merged = merge_lora(model)                    # fold into base weights for inference
```

Merging removes the adapter's runtime cost, which matters for inference. Keep
the unmerged adapter too — it is the only artifact you can compose with another
one.

## Control adapters

```python
from avgen.finetune import ControlAdapter
```

A ControlNet-style side tower conditioned on an auxiliary signal — depth, pose,
edges, a reference frame. The side tower reads the same `TokenStream`, so
control tokens carry their own physical coordinates and shard under context
parallelism exactly like the main stream. There is no separate control-path
sharding to get wrong.

## What changes for the training loop

**Almost nothing.** Same objective, same timestep sampler, same parallelism.
Three practical differences:

1. **Learning rate.** LoRA wants roughly 10× the LR of full finetuning; 1e-4 is
   a normal starting point where the base model trained at 1e-5.
2. **Warmup can be short.** There is much less state to stabilise.
3. **EMA is usually unnecessary** for short adapter runs, and it doubles the
   memory of the trainable parameters. For longer runs it still helps.

## Parallelism for finetuning

Sequence length has not changed, so **context parallelism is still the axis that
decides whether it fits**. What has changed is that parameter, gradient, and
optimizer memory are now negligible, which means `dp_shard` buys you almost
nothing and `cp` buys you everything.

A finetune that fits on 8 GPUs at 480p will still need `cp=8` at 720p. Run the
simulator with the trainable parameter count:

```bash
avgen plan --model 2b --world-size 8 --seq-len 32400
```

## Condition modes

Finetuning is often about teaching a new *conditioning*, not a new style.
`ConditionMode` covers image-to-video, continuation, inpainting,
video-to-video, and the audio-video directions, and the conditioning sampler
mixes them during training.

The `TokenStream.conditioned` mask is what makes this clean: conditioned tokens
are clean anchors, excluded from the loss and given noise level zero, and the
mask survives context-parallel sharding. You do not need a separate code path
for each mode.

## Further reading

- [Reinforcement learning](rl.md) — post-training beyond supervised finetuning.
- [Parallelism](parallelism.md)
- [Checkpointing](checkpointing.md)
