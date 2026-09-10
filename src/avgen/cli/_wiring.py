"""Assemble a live training or generation stack from a :class:`RunConfig`.

This is the seam between the declarative config and the subsystems that do the
work. It is deliberately thin: it constructs, in the one order that is correct,
and it owns no policy of its own.

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
   draw identical noise (CONTRACTS.md §6, rule 1).
7. **Loader**, sharded on ``data_rank`` for the same reason (rule 2).

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

__all__ = [
    "TrainingStack",
    "build_generation_pipeline",
    "build_logger_for",
    "build_objective",
    "build_source",
    "build_trainer_config",
    "build_training_stack",
    "require_subsystem",
]


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
            "CONTRACTS.md section 4. If avgen is installed, this is a broken "
            "install; in a source checkout it means that subsystem is not "
            "finished yet. 'avgen info' lists which subsystems import."
        ) from error
    try:
        return getattr(loaded, symbol)
    except AttributeError as error:
        raise RuntimeError(
            f"{module}.{symbol} does not exist. CONTRACTS.md section 4 declares "
            "it; either the contract moved or that subsystem is incomplete."
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
        source: The data source, already sharded on ``data_rank``.
        objective: The training objective.
        logger: The telemetry sink.
        gradient_accumulation: Microbatches per optimizer step.
        data_rank: This rank's index within the data-parallel product.
        data_world: Size of that product.
    """

    config: RunConfig
    env: Any
    dims: Any
    parallel: Any
    state: Any
    source: Any
    objective: Any
    logger: Any
    gradient_accumulation: int
    data_rank: int
    data_world: int


# ---------------------------------------------------------------------------
# Pieces
# ---------------------------------------------------------------------------


def _timestep_options(config: RunConfig) -> dict[str, Any]:
    """Return only the options the chosen timestep sampler actually accepts.

    ``build_timestep_sampler`` forwards its mapping as keyword arguments, so
    handing every sampler the union of all their options raises on the simple
    ones. Selecting per sampler keeps the config's single
    ``train.timestep_sampler`` key usable for all four.
    """
    train = config.train
    name = train.timestep_sampler
    if name in ("uniform", "mode"):
        return {}
    common: dict[str, Any] = {
        "mean": train.logit_normal_mean,
        "std": train.logit_normal_std,
    }
    if name == "logit_normal":
        return common
    # shifted_logit_normal: the shift is interpolated between two calibration
    # points, because the same nominal noise level destroys far more information
    # in a 100k-token sequence than in a 4k one.
    return {
        **common,
        "base_seq_len": train.base_seq_len,
        "base_shift": train.base_shift,
        "max_seq_len": train.max_seq_len,
        "max_shift": train.max_shift,
    }


def build_objective(config: RunConfig) -> Any:
    """Construct the flow-matching objective and its samplers.

    Args:
        config: The run configuration.

    Returns:
        The objective.

    Raises:
        RuntimeError: If ``avgen.train`` is unavailable.
    """
    build_timestep_sampler = require_subsystem(
        "avgen.train.timestep", "build_timestep_sampler"
    )
    multi_task = require_subsystem("avgen.train.conditioning", "MultiTaskConditioning")
    flow_config = require_subsystem("avgen.train.objective", "FlowMatchingConfig")
    flow_objective = require_subsystem("avgen.train.objective", "FlowMatchingObjective")
    timestep = build_timestep_sampler(
        config.train.timestep_sampler, _timestep_options(config)
    )
    conditioning = multi_task(text_dropout=config.data.caption_dropout)
    return flow_objective(
        flow_config(audio_weight=config.train.audio_loss_weight),
        timestep,
        conditioning,
    )


def build_source(
    config: RunConfig,
    *,
    data_rank: int = 0,
    data_world: int = 1,
) -> Any:
    """Build the data source, sharded on ``data_rank`` only.

    Sharding on ``data_rank`` rather than global rank is the rule that makes a
    multi-dimensional parallel trainer correct: context- and tensor-parallel
    ranks hold shards of the *same* samples and must receive byte-identical
    batches. Getting it wrong is invisible — the loss curve looks fine while the
    effective batch size is a fraction of what the config says.

    Args:
        config: The run configuration.
        data_rank: This rank's index in the data-parallel product.
        data_world: Size of that product.

    Returns:
        An iterable of :class:`~avgen.core.MediaBatch`.

    Raises:
        RuntimeError: If ``avgen.data`` is unavailable.
        FileNotFoundError: If a shard root is configured but holds no shards.
    """
    bucket = config.data.largest_bucket()
    micro = config.micro_batch_size

    if config.data.source == "synthetic":
        synthetic_source = require_subsystem("avgen.data.synthetic", "SyntheticSource")
        synthetic_config = require_subsystem("avgen.data.synthetic", "SyntheticConfig")
        settings = synthetic_config(
            seed=config.data.seed,
            video_channels=config.data.latent_channels,
            frames=bucket.frames,
            height=bucket.height,
            width=bucket.width,
            audio_channels=config.data.audio_channels,
            audio_frames=config.data.audio_frames,
            text_tokens=config.data.text_tokens,
            text_width=config.data.text_width,
        )
        return synthetic_source(
            settings,
            batch_size=micro,
            data_rank=data_rank,
            data_world=data_world,
        )

    from pathlib import Path

    build_loader = require_subsystem("avgen.data.loader", "build_loader")
    concat_reader = require_subsystem("avgen.data.shard", "ConcatShardReader")
    shard_reader = require_subsystem("avgen.data.shard", "ShardReader")

    from avgen.cli.data import shard_directories

    root = Path(config.data.root)
    shards = shard_directories(root) if root.is_dir() else []
    if not shards:
        raise FileNotFoundError(
            f"data.root={root} contains no committed shards. avgen trains on "
            "pre-encoded latents; write them with avgen.data.write_shard (see "
            "'avgen data synthesize' for a worked example), or set "
            "data.source=synthetic for a smoke run. An uncommitted shard is "
            "invisible on purpose — a half-written one must never be read."
        )
    # NOTE (avgen.data defect, not this module's): a shard-backed loader that
    # has yielded at least one batch aborts the interpreter with SIGABRT at
    # shutdown — "terminate called without an active exception" — after the run
    # has already succeeded. Closing the readers, dropping every reference, and
    # forcing a collection before close all fail to prevent it, so the mmap
    # lifetime has to be fixed inside avgen.data.shard. Reproducer:
    #
    #   store = ConcatShardReader([ShardReader(<shard dir>)])
    #   for _ in build_loader(store, batch_size=2): break
    #
    # A training run completes and writes its checkpoints normally; only the
    # exit code is wrong, so do not treat a 134 from a finished run as a
    # training failure until this is fixed.
    store = concat_reader([shard_reader(path) for path in shards])
    return build_loader(
        store,
        batch_size=micro,
        data_rank=data_rank,
        data_world=data_world,
    )


def build_trainer_config(
    config: RunConfig,
    *,
    accumulation: int,
    data_world: int,
) -> Any:
    """Build the trainer's own configuration.

    Args:
        config: The run configuration.
        accumulation: Microbatches per optimizer step.
        data_world: Size of the data-parallel product, so the trainer's
            throughput accounting knows how many samples a step really covers.

    Returns:
        A :class:`avgen.train.trainer.TrainerConfig`.
    """
    trainer_config = require_subsystem("avgen.train.trainer", "TrainerConfig")
    return trainer_config(
        gradient_accumulation_steps=accumulation,
        # Zero means "no clipping" in the config; the trainer spells that None.
        max_grad_norm=config.train.max_grad_norm or None,
        log_every=config.train.log_every,
        eval_every=config.train.eval_every,
        checkpoint_every=config.checkpoint.save_every,
        autocast_dtype=config.parallel.precision.param_dtype,
        data_world=data_world,
    )


def build_logger_for(config: RunConfig, *, rank: int = 0) -> Any:
    """Build the telemetry sink described by ``telemetry.loggers``.

    The resolved config is handed to the logger so that a tracking backend
    records what produced the curve. A metric series whose configuration lives
    only in shell history documents nothing.

    Args:
        config: The run configuration.
        rank: Global rank, so only rank 0 writes.

    Returns:
        A :class:`avgen.telemetry.Logger`.
    """
    from pathlib import Path

    from avgen.config.diff import to_mapping

    build_logger = require_subsystem("avgen.telemetry", "build_logger")
    telemetry = config.telemetry
    return build_logger(
        telemetry.loggers,
        log_dir=Path(config.output_dir),
        run_name=telemetry.run_name or config.resolved_run_name(),
        project=telemetry.project,
        config=to_mapping(config),
        rank=rank,
        jsonl_path=telemetry.jsonl_path or None,
        tensorboard_dir=telemetry.tensorboard_dir or None,
    )


# ---------------------------------------------------------------------------
# The whole stack
# ---------------------------------------------------------------------------


def _materialize(model: Any, device: Any, *, seed: int) -> None:
    """Move a meta-device model onto real storage and initialise it.

    Building on meta is free but leaves every parameter storage-less, so it has
    to be materialised before the optimizer is constructed. avgen models keep
    initialisation out of ``__init__`` and expose ``init_weights()`` for exactly
    this sequence — build on meta, shard, ``to_empty()``, then fill — which is
    the only order in which a model too large for one device can be created at
    all. Their determinism comes from the ambient torch seed, so the seed is set
    here, before the call (CONTRACTS.md section 6, rule 12).

    Args:
        model: The parallelised model.
        device: Device to materialise on.
        seed: Master seed for initialisation.
    """
    import torch

    if not any(parameter.is_meta for parameter in model.parameters()):
        return
    model.to_empty(device=device)
    # Seed BEFORE initialising, not after: avgen models take their determinism
    # from the ambient torch seed, so setting it afterwards would make the
    # weights depend on whatever ran before this call.
    torch.manual_seed(seed)

    from avgen.parallel.env import unwrap_model

    target = unwrap_model(model)
    for hook in ("init_weights", "reset_parameters"):
        method = getattr(target, hook, None)
        if callable(method):
            method()
            return
    # No model-level hook: fall back to per-module initialisation. This leaves
    # any custom parameter that owns no reset_parameters as uninitialised
    # garbage, which surfaces immediately as a NaN loss rather than as a
    # subtly worse model — the failure mode worth having.
    for module in target.modules():
        module_reset = getattr(module, "reset_parameters", None)
        if callable(module_reset) and module is not target:
            module_reset()


def build_training_stack(
    config: RunConfig,
    *,
    adapt: Any = None,
) -> TrainingStack:
    """Construct the full training stack in dependency order.

    Args:
        config: The validated run configuration.
        adapt: Optional model transformation applied between tensor parallelism
            and FSDP — the seam adapter fine-tuning needs. See
            :func:`avgen.parallel.apply.parallelize`.

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

    # Build on meta: a 14B model constructed here costs no memory, and the
    # parallel plans then materialise only this rank's shard. This is how a
    # large job should initialise, not merely how it is simulated.
    build_model = require_subsystem("avgen.models.registry", "build_model")
    with torch.device("meta"):
        model = build_model(config.model.name, model_kwargs(config))

    parallel = parallelize(
        model, dims, config=build_parallel_config(config), adapt=adapt
    )
    _materialize(parallel.model, env.device, seed=config.seed)

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
    # CONTRACTS.md section 6, rule 1: RNG varies on data_rank ONLY. Ranks holding
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
        source=build_source(config, data_rank=data_rank, data_world=data_world),
        objective=build_objective(config),
        logger=build_logger_for(config, rank=env.rank),
        gradient_accumulation=accumulation,
        data_rank=data_rank,
        data_world=data_world,
    )


def build_generation_pipeline(config: RunConfig, checkpoint: str | None = None) -> Any:
    """Build a generation pipeline, loading weights when a checkpoint is given.

    The codecs default to the **reference** implementations, which are
    dependency-free and deterministic. They produce structurally valid media and
    are not a real autoencoder: use them for smoke tests and wiring checks, and
    configure a real codec for anything whose output you intend to look at.

    Args:
        config: The run configuration.
        checkpoint: Checkpoint to load weights from, or ``None`` for a randomly
            initialised model, which is useful only for a smoke test.

    Returns:
        A :class:`avgen.infer.GenerationPipeline`.

    Raises:
        RuntimeError: If ``avgen.infer``, ``avgen.models``, or ``avgen.codecs``
            is unavailable.
    """
    from avgen.config.resolve import model_kwargs

    build_model = require_subsystem("avgen.models.registry", "build_model")
    pipeline_class = require_subsystem("avgen.infer", "GenerationPipeline")
    generation_config = require_subsystem("avgen.infer", "GenerationConfig")
    guidance_config = require_subsystem("avgen.infer", "GuidanceConfig")
    sampler_config = require_subsystem("avgen.infer", "SamplerConfig")
    reference_video = require_subsystem("avgen.codecs", "ReferenceVideoCodec")
    reference_text = require_subsystem("avgen.codecs", "ReferenceTextEncoder")

    model = build_model(config.model.name, model_kwargs(config))
    if checkpoint:
        load_weights = require_subsystem("avgen.checkpoint", "import_safetensors")
        load_weights(checkpoint, model)

    inference = config.inference
    settings = generation_config(
        steps=inference.steps,
        seed=max(0, inference.seed),
        height=inference.height,
        width=inference.width,
        num_frames=inference.frames,
        fps=inference.fps,
        guidance=guidance_config(
            scale=inference.guidance, rescale=inference.guidance_rescale
        ),
        sampler=sampler_config(name=inference.sampler),
        negative_prompt=inference.negative_prompt or None,
    )
    return pipeline_class(
        model,
        video_codec=reference_video(
            channels=config.model.in_channels, spatial_compression=1
        ),
        text_encoder=reference_text(
            width=config.data.text_width, max_length=config.data.text_tokens
        ),
        config=settings,
    )
