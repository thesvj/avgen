"""Turn a declarative :class:`RunConfig` into live objects.

The split between this module and :mod:`avgen.config.schema` is the whole point
of the config layer's design: **the schema knows no torch, and the resolver
knows no YAML.** ``avgen plan`` and ``avgen --help`` import only the schema and
start instantly on a machine with no GPU; the moment a command actually needs a
device mesh or a model, it calls in here.

The second rule is that everything a *concurrently developed* subsystem owns is
imported inside the function that needs it. ``avgen info`` must still run when
``avgen.models`` is half-written, and a user with only the core dependencies
installed must still be able to plan a cluster job.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from avgen.config.schema import BucketConfig, RunConfig

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch import nn

    from avgen.parallel.activation import ActivationCheckpointConfig
    from avgen.parallel.apply import ParallelConfig
    from avgen.parallel.dims import ParallelDims
    from avgen.parallel.fsdp import FSDPConfig
    from avgen.parallel.pipeline import PipelineConfig
    from avgen.parallel.precision import PrecisionConfig
    from avgen.simulate.comms import Interconnect
    from avgen.simulate.compute import Accelerator
    from avgen.simulate.memory import ModelShape

__all__ = [
    "build_accelerator",
    "build_activation_checkpoint",
    "build_fsdp_config",
    "build_interconnects",
    "build_model",
    "build_model_shape",
    "build_parallel_config",
    "build_parallel_dims",
    "build_pipeline_config",
    "build_precision_config",
    "gradient_accumulation",
    "model_kwargs",
    "sequence_length_for",
]


# ---------------------------------------------------------------------------
# Parallelism
# ---------------------------------------------------------------------------


def build_parallel_dims(
    config: RunConfig,
    *,
    world_size: int | None = None,
) -> ParallelDims:
    """Build the validated parallelism degrees for a run.

    Args:
        config: The run configuration.
        world_size: Ranks in the job. ``None`` reads ``WORLD_SIZE`` from the
            launcher environment, defaulting to one.

    Returns:
        The dims, which validate that the degrees multiply to ``world_size``.

    Raises:
        ValueError: If the degrees do not factor the world size. Deliberately
            fatal — silently reducing a degree to make the arithmetic work
            changes the effective batch size and the memory footprint at once.
    """
    import os

    from avgen.parallel.dims import ParallelDims

    spec = config.parallel
    size = (
        world_size if world_size is not None else int(os.environ.get("WORLD_SIZE", "1"))
    )
    return ParallelDims(
        world_size=size,
        dp_replicate=spec.dp_replicate,
        dp_shard=spec.dp_shard,
        tensor=spec.tensor,
        context=spec.context,
        pipeline=spec.pipeline,
        enable_loss_parallel=spec.enable_loss_parallel,
    )


def build_precision_config(config: RunConfig) -> PrecisionConfig:
    """Build the mixed-precision policy.

    Args:
        config: The run configuration.

    Returns:
        The live precision config.
    """
    from avgen.parallel.precision import PrecisionConfig

    spec = config.parallel.precision
    return PrecisionConfig(
        param_dtype=spec.param_dtype,  # type: ignore[arg-type]
        reduce_dtype=spec.reduce_dtype,  # type: ignore[arg-type]
        enable_float8=spec.enable_float8,
        float8_min_features=spec.float8_min_features,
        float8_recipe=spec.float8_recipe,  # type: ignore[arg-type]
    )


def build_activation_checkpoint(config: RunConfig) -> ActivationCheckpointConfig:
    """Build the activation-checkpointing policy.

    Args:
        config: The run configuration.

    Returns:
        The live checkpointing config.
    """
    from avgen.parallel.activation import ActivationCheckpointConfig

    spec = config.parallel.activation
    return ActivationCheckpointConfig(
        mode=spec.mode,  # type: ignore[arg-type]
        layer_interval=spec.layer_interval,
        save_op_frequency=spec.save_op_frequency,
    )


def build_fsdp_config(config: RunConfig) -> FSDPConfig:
    """Build the FSDP2 sharding options.

    Args:
        config: The run configuration.

    Returns:
        The live FSDP config.
    """
    from avgen.parallel.fsdp import FSDPConfig

    spec = config.parallel.fsdp
    return FSDPConfig(
        reshard_after_forward=spec.reshard_after_forward,
        cpu_offload=spec.cpu_offload,
        shard_last_block_after_forward=spec.shard_last_block_after_forward,
        ignore_frozen_params=spec.ignore_frozen_params,
    )


def build_pipeline_config(config: RunConfig) -> PipelineConfig:
    """Build the pipeline schedule options.

    Args:
        config: The run configuration.

    Returns:
        The live pipeline config.
    """
    from avgen.parallel.pipeline import PipelineConfig

    spec = config.parallel.pipeline_schedule
    return PipelineConfig(
        schedule=spec.schedule,  # type: ignore[arg-type]
        microbatches=spec.microbatches,
        stages_per_rank=spec.stages_per_rank,
    )


def build_parallel_config(config: RunConfig) -> ParallelConfig:
    """Assemble the full parallelisation policy handed to ``parallelize``.

    Args:
        config: The run configuration.

    Returns:
        The live parallel config.
    """
    from avgen.parallel.apply import ParallelConfig

    return ParallelConfig(
        precision=build_precision_config(config),
        activation_checkpoint=build_activation_checkpoint(config),
        fsdp=build_fsdp_config(config),
        compile_blocks=config.parallel.compile_blocks,
        compile_mode=config.parallel.compile_mode,
        sequence_parallel=config.parallel.sequence_parallel,
    )


def gradient_accumulation(config: RunConfig, *, world_size: int | None = None) -> int:
    """Return the microbatches per optimizer step implied by the config.

    Args:
        config: The run configuration.
        world_size: Ranks in the job.

    Returns:
        The accumulation factor.

    Raises:
        ValueError: If the global batch size does not factor across the data
            dimension and the microbatch size.
    """
    dims = build_parallel_dims(config, world_size=world_size)
    return dims.gradient_accumulation_for(
        global_batch_size=config.train.global_batch_size,
        local_batch_size=config.micro_batch_size,
    )


# ---------------------------------------------------------------------------
# Shapes and hardware profiles
# ---------------------------------------------------------------------------


def sequence_length_for(config: RunConfig, bucket: BucketConfig | None = None) -> int:
    """Return the token count one sample produces.

    Args:
        config: The run configuration.
        bucket: A specific bucket. ``None`` uses the largest, which is the one
            that decides whether the run fits.

    Returns:
        Tokens per sample, before context-parallel sharding.
    """
    chosen = bucket or config.data.largest_bucket()
    return chosen.tokens(config.model)


def build_model_shape(
    config: RunConfig,
    *,
    bucket: BucketConfig | None = None,
    parameters: int | None = None,
) -> ModelShape:
    """Build the geometry the memory, compute, and comms estimators price.

    Audio tokens are added to the video sequence length rather than tracked
    separately: an AV DiT attends over the concatenation, so the quadratic
    attention term is a function of the *total*, and pricing the two streams
    independently understates it.

    Args:
        config: The run configuration.
        bucket: Bucket to price. ``None`` uses the largest.
        parameters: Override the parameter count. ``None`` uses the config's
            explicit count, or the estimate derived from depth and width.

    Returns:
        The model shape.
    """
    from avgen.simulate.memory import ModelShape

    chosen = bucket or config.data.largest_bucket()
    tokens = chosen.tokens(config.model)
    if config.model.has_audio and config.data.audio_frames:
        tokens += config.data.audio_frames // config.model.patch_frames
    micro = (
        config.train.micro_batch_size
        if config.train.micro_batch_size != -1
        else chosen.micro_batch_size
    )
    return ModelShape(
        parameters=parameters or config.model.estimated_parameters(),
        depth=config.model.depth,
        width=config.model.width,
        sequence_length=tokens,
        micro_batch_size=micro,
        mlp_ratio=config.model.mlp_ratio,
        num_heads=config.model.num_heads,
        text_tokens=config.data.text_tokens if config.model.cross_attention else 0,
    )


def build_accelerator(name: str) -> Accelerator:
    """Look up a device profile by short name.

    Args:
        name: ``a100``, ``h100``, ``h200``, or ``b200``.

    Returns:
        The accelerator profile.

    Raises:
        ValueError: If the name is unknown. The profiles are published dense
            peaks, not measurements on your silicon — calibrate before trusting
            an MFU derived from them.
    """
    from avgen.simulate.compute import A100_80GB, B200, H100_SXM, H200_SXM

    profiles = {
        "a100": A100_80GB,
        "h100": H100_SXM,
        "h200": H200_SXM,
        "b200": B200,
    }
    key = name.strip().lower()
    if key not in profiles:
        raise ValueError(f"unknown gpu {name!r}; known: {', '.join(sorted(profiles))}")
    return profiles[key]


def build_interconnects(gpu: str) -> tuple[Interconnect, Interconnect]:
    """Return plausible intra-node and inter-node fabrics for a device.

    Blackwell parts ship with NVLink 5, which roughly doubles the intra-node
    bandwidth; pairing a B200 with an NVLink 4 profile understates every
    tensor- and context-parallel plan. The inter-node tier stays InfiniBand NDR
    because that is what most clusters actually have, and it is the tier that
    usually decides the answer.

    Args:
        gpu: Device short name.

    Returns:
        ``(intra_node, inter_node)`` profiles. **Calibrate them** with
        :func:`avgen.simulate.comms.calibrate_from_busbw` before trusting a
        predicted step time.
    """
    from avgen.simulate.comms import INFINIBAND_NDR, NVLINK4, NVLINK5

    intra = NVLINK5 if gpu.strip().lower() == "b200" else NVLINK4
    return intra, INFINIBAND_NDR


# ---------------------------------------------------------------------------
# Model construction (lazy: avgen.models is developed independently)
# ---------------------------------------------------------------------------


def model_kwargs(config: RunConfig) -> dict[str, Any]:
    """Return the keyword arguments a model constructor receives.

    Args:
        config: The run configuration.

    Returns:
        A mapping of architecture fields plus the ``extra`` escape hatch,
        flattened. ``extra`` wins on conflict, which is what makes it usable
        for overriding a field the shared schema names but a particular model
        interprets differently.
    """
    model = config.model
    base: dict[str, Any] = {
        "depth": model.depth,
        "width": model.width,
        "num_heads": model.num_heads,
        "mlp_ratio": model.mlp_ratio,
        "patch_frames": model.patch_frames,
        "patch_height": model.patch_height,
        "patch_width": model.patch_width,
        "in_channels": model.in_channels,
        "out_channels": model.latent_out_channels,
        "cross_attention": model.cross_attention,
        "text_width": config.data.text_width,
        "rope_theta": model.rope_theta,
    }
    if model.has_audio:
        base["audio_width"] = model.audio_width
        base["audio_channels"] = config.data.audio_channels
    base.update(model.extra)
    return base


def build_model(config: RunConfig, *, meta: bool = False) -> nn.Module:
    """Construct the configured model.

    Args:
        config: The run configuration.
        meta: Whether to build on the ``meta`` device, which gives real shapes
            and dtypes at zero memory cost. This is how a large model should be
            constructed even in production: build on meta, apply the parallel
            plans, materialise only this rank's shard, so nothing ever has to
            fit unsharded.

    Returns:
        The model.

    Raises:
        RuntimeError: If ``avgen.models`` is unavailable, with the registry
            name that was requested so the failure is actionable.
    """
    import contextlib

    import torch

    try:
        from avgen.models.registry import build_model as _build
    except ImportError as error:  # pragma: no cover - depends on install state
        raise RuntimeError(
            f"cannot build model {config.model.name!r}: avgen.models is not "
            f"importable ({error}). Install avgen fully, or use "
            "'avgen plan' / 'avgen simulate', which price a run analytically "
            "and need no model instance."
        ) from error

    context = torch.device("meta") if meta else contextlib.nullcontext()
    with context:
        return _build(config.model.name, model_kwargs(config))
