"""Train a video diffusion transformer in fifty lines, on a CPU, with no data.

This is the whole framework in one file: build a model, wrap it for whatever
parallelism the machine has, and step it. Everything else in avgen — 5D
parallelism, sharded checkpoints, RL post-training — is this loop with more
machinery around it, not a different loop.

Run:
    python examples/01_training_loop_from_scratch.py

Needs no GPU, no dataset and no download. It finishes in seconds.
"""

from __future__ import annotations

import torch

from avgen import (
    RNGStreams,
    SyntheticSource,
    TrainState,
    build_model,
    load_config,
    parallelize,
    train_step,
)
from avgen.config.resolve import (
    build_parallel_config,
    build_parallel_dims,
    model_kwargs,
)
from avgen.core.patchify import GridPatchifier
from avgen.data.synthetic import SyntheticConfig
from avgen.train import (
    FlowMatchingConfig,
    FlowMatchingObjective,
    MultiTaskConditioning,
    build_optimizer,
    build_schedule,
    build_timestep_sampler,
)


def main() -> None:
    # A config is plain dataclasses over plain YAML. Loading none gives the
    # schema defaults; the overrides here shrink it to something a laptop runs.
    config = load_config(
        None,
        overrides=[
            "model.name=video_dit",
            "model.depth=2",
            "model.width=64",
            "model.num_heads=4",
            "data.source=synthetic",
            "train.steps=10",
            "train.warmup_steps=2",
            "train.micro_batch_size=2",
            "train.global_batch_size=2",
        ],
    )

    torch.manual_seed(config.seed)
    model = build_model(config.model.name, model_kwargs(config))
    print(f"model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # parallelize() is the single seam between "a model" and "a model running on
    # a cluster". With no distributed environment it is a no-op wrapper, which
    # is exactly why the same script runs on a laptop and on 1024 GPUs.
    dims = build_parallel_dims(config, world_size=1)
    parallel = parallelize(model, dims, config=build_parallel_config(config))
    print(f"parallelism: {dims.describe()}")

    # Everything downstream uses parallel.model, not the model you passed in.
    # Under FSDP those are the same object with sharded parameters; under tensor
    # or pipeline parallelism they are not, and building the optimizer over the
    # original is how a run silently optimises the wrong tensors.
    optimizer = build_optimizer(parallel.model, lr=config.train.lr, weight_decay=0.01)
    state = TrainState(
        model=parallel.model,
        optimizer=optimizer,
        rng=RNGStreams.from_seed(config.seed),
        schedule=build_schedule(
            "constant", optimizer, total_steps=config.train.steps, warmup_steps=2
        ),
    )

    # Rectified flow: the model predicts the velocity along a straight path from
    # noise to data. The objective owns the timestep sampling and the loss.
    objective = FlowMatchingObjective(
        FlowMatchingConfig(),
        # The timestep shift is a function of sequence length: a 100k-token clip
        # needs more probability mass at high noise than an image does, or the
        # model never learns global structure. This is the sampler that does it.
        build_timestep_sampler("shifted_logit_normal"),
        MultiTaskConditioning(text_dropout=config.data.caption_dropout),
    )
    patchifier = GridPatchifier(
        patch_frames=config.model.patch_frames,
        patch_height=config.model.patch_height,
        patch_width=config.model.patch_width,
    )

    # Deterministic, dependency-free batches. Swap this for build_loader() over
    # a shard directory and nothing else in the loop changes.
    source = SyntheticSource(
        SyntheticConfig(
            seed=config.seed,
            video_channels=config.data.latent_channels,
            text_tokens=config.data.text_tokens,
            text_width=config.data.text_width,
        ),
        batch_size=config.train.micro_batch_size,
    )

    for step, batch in enumerate(source):
        if step >= config.train.steps:
            break
        state, metrics = train_step(
            state,
            batch,
            objective,
            patchifier=patchifier,
            max_grad_norm=config.train.max_grad_norm,
            autocast_dtype=torch.float32,  # CPU bf16 matmuls are slow
        )
        # .item() here is fine; inside a real training step it is a device sync
        # on every rank, which is why the trainer accumulates and syncs once per
        # log interval instead.
        print(f"step {step:>3}  loss {float(metrics.loss):.4f}")

    print("done — the same script runs unchanged under torchrun on 1024 GPUs")


if __name__ == "__main__":
    main()
