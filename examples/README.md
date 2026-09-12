# Examples

Five self-contained programs, in the order they are worth reading. Each one runs
on a CPU in a few seconds, with no dataset, no download and no GPU. All of them
are executed by the test suite, so if a signature here has drifted from the
framework, CI will tell you rather than you finding out.

```bash
python examples/01_training_loop_from_scratch.py
```

| | What it shows |
|---|---|
| [`01_training_loop_from_scratch.py`](01_training_loop_from_scratch.py) | The whole framework in about fifty lines: build a model, parallelize it, step it. Everything else is this loop with more machinery around it. |
| [`02_choose_a_parallelism_plan.py`](02_choose_a_parallelism_plan.py) | Rank and price every 5D parallelism plan for 1024 GPUs, on a laptop, in milliseconds, before you book an allocation. |
| [`03_register_a_custom_model.py`](03_register_a_custom_model.py) | The entire model ABI. Satisfy it and your architecture gets FSDP, context parallelism, sharded checkpoints, LoRA and RL for free. |
| [`04_lora_finetune.py`](04_lora_finetune.py) | LoRA and DoRA on DTensor, and merging the adapter away so inference costs nothing extra. |
| [`05_sampling_and_schedules.py`](05_sampling_and_schedules.py) | Why the sigma schedule's shift depends on sequence length, and what each sampler actually buys you. |

## Going further

- For a real training run, see
  [`docs/getting-started/first-training-run.md`](../docs/getting-started/first-training-run.md).
- For multi-node, the annotated configs in [`configs/train/`](../configs/train/)
  explain every parallelism choice and why it was made in that order.
- [`benchmarks/`](../benchmarks/) measures a real step and compares it against
  what the simulator predicted.
