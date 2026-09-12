"""Measure a real training step, and compare it against the simulator.

This is the benchmark that keeps the simulator honest. `avgen plan` predicts
step time and memory from analytical models; those predictions are only worth
anything if someone periodically checks them against a step that actually ran.

Run:
    python benchmarks/bench_step.py                      # tiny, CPU, seconds
    python benchmarks/bench_step.py --model dit_2b --device cuda --steps 20

Reports measured step time, tokens/second, peak memory, and — when the shapes
are ones the simulator can price — the predicted values beside them.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from avgen import (
    RNGStreams,
    TrainState,
    build_model,
    load_config,
    train_step,
)
from avgen.config.resolve import model_kwargs
from avgen.core.patchify import GridPatchifier
from avgen.data.synthetic import SyntheticConfig, SyntheticSource
from avgen.train import (
    FlowMatchingConfig,
    FlowMatchingObjective,
    MultiTaskConditioning,
    build_optimizer,
    build_schedule,
    build_timestep_sampler,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="tiny", help="configs/model/<name>.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--batch", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    config = load_config(f"configs/model/{args.model}.yaml")

    torch.manual_seed(0)
    model = build_model(config.model.name, model_kwargs(config)).to(device)
    parameters = sum(p.numel() for p in model.parameters())

    optimizer = build_optimizer(model, lr=1e-4, weight_decay=0.0)
    state = TrainState(
        model=model,
        optimizer=optimizer,
        rng=RNGStreams.from_seed(0),
        schedule=build_schedule("constant", optimizer, total_steps=10_000),
    )
    objective = FlowMatchingObjective(
        FlowMatchingConfig(),
        build_timestep_sampler("shifted_logit_normal"),
        MultiTaskConditioning(),
    )
    patchifier = GridPatchifier(
        patch_frames=config.model.patch_frames,
        patch_height=config.model.patch_height,
        patch_width=config.model.patch_width,
    )
    source = SyntheticSource(
        SyntheticConfig(
            video_channels=config.model.in_channels,
            frames=args.frames,
            height=args.height,
            width=args.width,
            text_width=config.data.text_width,
            text_tokens=config.data.text_tokens,
        ),
        batch_size=args.batch,
    )
    layout = patchifier.layout_for(
        (args.batch, config.model.in_channels, args.frames, args.height, args.width)
    )

    print(f"model      {config.model.name} — {parameters / 1e6:.1f}M parameters")
    print(f"device     {device}")
    print(f"sequence   {layout.num_tokens:,} tokens x batch {args.batch}")

    autocast = torch.bfloat16 if device.type == "cuda" else torch.float32
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    durations: list[float] = []
    batches = iter(source)
    for index in range(args.warmup + args.steps):
        batch = next(batches).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        state, _ = train_step(
            state, batch, objective, patchifier=patchifier, autocast_dtype=autocast
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        # Warmup steps are discarded rather than averaged in: the first step
        # pays for lazy kernel selection, allocator growth and, under torchrun,
        # NCCL channel setup. Averaging them in understates steady-state speed
        # by a factor that depends on how many steps you happened to run.
        if index >= args.warmup:
            durations.append(time.perf_counter() - started)

    median = statistics.median(durations)
    tokens = layout.num_tokens * args.batch
    print()
    print(f"step time  {median * 1e3:8.2f} ms  (median of {len(durations)})")
    print(f"           {min(durations) * 1e3:8.2f} ms  (fastest)")
    print(f"throughput {tokens / median:10,.0f} tokens/s")
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) / 1024**3
        print(f"peak mem   {peak:8.2f} GiB")

    print(
        "\nCompare against the prediction for the same shape:\n"
        f"  avgen plan --world-size 1 --seq-len {layout.num_tokens} "
        f"--params {parameters} --depth {config.model.depth} "
        f"--width {config.model.width}"
    )


if __name__ == "__main__":
    main()
