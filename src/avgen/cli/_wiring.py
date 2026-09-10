"""Assemble a live training stack from a :class:`~avgen.config.RunConfig`.

This is the seam between the declarative config and the subsystems that do the
work, and it is deliberately thin: it constructs, in the one order that is
correct, and it owns no policy of its own.

The order is not arbitrary. Each step assumes the previous one happened:

1. **Distributed environment**, so the device and the rank exist.
2. **Parallelism degrees**, validated against the real world size — a job whose
   degrees do not factor the world size must die here, before allocation, not
   after the first all-gather.
3. **Model on meta**, so a 14B model costs nothing to construct. Nothing ever
   has to fit unsharded.
4. **``parallelize``**, which applies float8, tensor parallel, checkpointing,
   compile, and FSDP in the order :mod:`avgen.parallel.apply` documents.
5. **Optimizer and schedule**, after sharding, so the optimizer sees DTensor
   parameters and its state is sharded with them.
6. **RNG for this rank's data coordinate**, which varies on ``data_rank`` only:
   context- and tensor-parallel ranks hold shards of the same sample and must
   draw identical noise.
7. **Loader**, sharded on ``data_rank`` for the same reason.

Every import is local to the function that needs it. These subsystems are the
largest in the package and are developed independently, so a missing one must
produce a precise message naming the contract rather than an ``ImportError``
traceback from four frames down.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from avgen.config.schema import RunConfig

__all__ = ["TrainingStack", "build_training_stack", "require_subsystem"]


def require_subsystem(module: str, symbol: str) -> Any:
    """Import a symbol from a concurrently developed subsystem, or explain.

    Args:
        module: Dotted module path, e.g. ``"avgen.train.optimizer"``.
        symbol: Name to fetch from it.

    Returns:
        The symbol.

    Raises:
        RuntimeError: If the module or symbol is unavailable, naming the
            contract that declares it so the reader knows whether this is a
            broken install or an unfinished subsystem.
    """
    import importlib

    try:
        loaded = importlib.import_module(module)
    except ImportError as error:
        raise RuntimeError(
            f"{module} is not available ({error}). It is declared in "
            "CONTRACTS.md §4. If avgen is installed, this is a broken install; "
            "in a source checkout it means that subsystem is not finished yet. "
            "'avgen info' lists which subsystems import."
        ) from error
    try:
        return getattr(loaded, symbol)
    except AttributeError as error:
        raise RuntimeError(
            f"{module}.{symbol} does not exist. CONTRACTS.md §4 declares it; "
            "either the contract moved or that subsystem is incomplete."
        ) from error


@dataclass(slots=True)
class TrainingStack:
    """Everything a training loop needs, constructed and wired.

    Args:
        config: The configuration this was built from.
        env: The distributed environment.
        dims: Validated parallelism degrees.
        parallel: The parallelised model plus its sub-meshes.
        state: Mutable training state (model, optimizer, rng, schedule, ema).
        loader: The data source, already sharded on ``data_rank``.
        objective: The training objective.
        gradient_accumulation: Microbatches per optimizer step.
        data_rank: This rank's index within the data-parallel product.
        data_world: Size of that product.
    """

    config: RunConfig
    env: Any
    dims: Any
    parallel: Any
    state: Any
    loader: Any
    objective: Any
    gradient_accumulation: int
    data_rank: int
    data_world: int


def build_training_stack(config: RunConfig) -> TrainingStack:
    """Construct the full training stack in dependency order.

    Args:
        config: The validated run configuration.

    Returns:
        The assembled stack.

    Raises:
        RuntimeError: If a required subsystem is unavailable.
        ValueError: If the parallelism degrees do not factor the world size, or
            the global batch size does not factor across the data dimension.
    """
    import torch

    from avgen.config.resolve import (
        build_parallel_config,
        build_parallel_dims,
        model_kwargs,
    )
    from avgen.core.rng import RNGStreams
    from avgen.core.state import TrainState
    from avgen.parallel.apply import parallelize
    from avgen.parallel.env import init_distributed

    env = init_distributed()
    dims = build_parallel_dims(config, world_size=env.world_size)

    # 3. Build on meta. A 14B model constructed on meta costs no memory; the
    # parallel plans then materialise only this rank's shard.
    build_model = require_subsystem("avgen.models.registry", "build_model")
    with torch.device("meta"):
        model = build_model(config.model.name, model_kwargs(config))

    parallel = parallelize(model, dims, config=build_parallel_config(config))

    build_optimizer = require_subsystem("avgen.train.optimizer", "build_optimizer")
    optimizer = build_optimizer(
        parallel.model,
        name=config.train.optimizer,
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
        betas=config.train.betas,
        eps=config.train.eps,
    )
    build_schedule = require_subsystem("avgen.train.schedule", "build_schedule")
    schedule = build_schedule(
        config.train.schedule,
        optimizer,
        total_steps=config.train.steps,
        warmup_steps=config.train.warmup_steps,
    )

    ema = None
    if config.train.ema_enabled:
        sharded_ema = require_subsystem("avgen.train.ema", "ShardedEMA")
        ema = sharded_ema(
            parallel.model,
            decay=config.train.ema_decay,
            warmup_steps=config.train.ema_warmup_steps,
        )

    mesh = parallel.mesh
    data_rank, data_world = dims.data_coordinates(mesh) if mesh is not None else (0, 1)
    # Rule 1 from CONTRACTS.md §6: RNG varies on data_rank ONLY. Ranks holding
    # shards of one sample must draw identical noise, or the sample is denoised
    # toward two different targets and the gradient is silently wrong.
    rng = RNGStreams.for_rank(config.seed, data_rank=data_rank, device=env.device)

    state = TrainState(
        model=parallel.model,
        optimizer=optimizer,
        rng=rng,
        schedule=schedule,
        ema=ema,
    )

    build_loader = require_subsystem("avgen.data.loader", "build_loader")
    loader = build_loader(config.data, data_rank=data_rank, data_world=data_world)

    objective = _build_objective(config)
    accumulation = dims.gradient_accumulation_for(
        global_batch_size=config.train.global_batch_size,
        local_batch_size=config.micro_batch_size,
    )

    return TrainingStack(
        config=config,
        env=env,
        dims=dims,
        parallel=parallel,
        state=state,
        loader=loader,
        objective=objective,
        gradient_accumulation=accumulation,
        data_rank=data_rank,
        data_world=data_world,
    )


def _build_objective(config: RunConfig) -> Any:
    """Construct the training objective and its timestep/conditioning samplers.

    The shifted timestep sampler is the default for a reason worth restating:
    the same nominal noise level destroys far more information in a
    100k-token sequence than in a 4k one, so a single fixed schedule trains the
    two ends of the resolution range badly. The shift interpolates between a
    calibration point at ``base_seq_len`` and one at ``max_seq_len``.
    """
    build_sampler = require_subsystem("avgen.train.timestep", "build_timestep_sampler")
    timestep = build_sampler(
        config.train.timestep_sampler,
        mean=config.train.logit_normal_mean,
        std=config.train.logit_normal_std,
        base_shift=config.train.base_shift,
        max_shift=config.train.max_shift,
        base_seq_len=config.train.base_seq_len,
        max_seq_len=config.train.max_seq_len,
    )
    build_conditioning = require_subsystem(
        "avgen.train.conditioning", "build_conditioning_sampler"
    )
    conditioning = build_conditioning(
        caption_dropout=config.data.caption_dropout,
        has_audio=config.data.has_audio,
    )
    flow_config_class = require_subsystem("avgen.train.objective", "FlowMatchingConfig")
    objective_class = require_subsystem(
        "avgen.train.objective", "FlowMatchingObjective"
    )
    return objective_class(
        flow_config_class(audio_loss_weight=config.train.audio_loss_weight),
        timestep,
        conditioning,
    )
