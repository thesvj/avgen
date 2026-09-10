"""Configuration: plain dataclasses, YAML, and dotted overrides. Nothing else.

The design decision, stated once: **a config system that requires a framework is
a config system users cannot read.** avgen's config is a tree of frozen
dataclasses (:mod:`avgen.config.schema`), a YAML loader with ``_base_``
composition and environment interpolation (:mod:`avgen.config.loader`), a
resolver that turns declarations into torch objects
(:mod:`avgen.config.resolve`), and a differ that keeps runs comparable
(:mod:`avgen.config.diff`). The only dependency is ``pyyaml``. Hydra, OmegaConf
and pydantic were all considered and all rejected for the same reason: the file
is read far more often by people than by the framework.

Typical use::

    from avgen.config import load_config, save_config

    cfg = load_config("configs/train/node_8gpu.yaml", overrides=["train.lr=1e-4"])
    save_config(cfg, "runs/exp/config.yaml")
"""

from avgen.config.diff import config_diff, format_diff, save_config, to_mapping
from avgen.config.loader import (
    ConfigError,
    apply_overrides,
    load_config,
    load_mapping,
    parse_override,
)
from avgen.config.resolve import (
    build_accelerator,
    build_activation_checkpoint,
    build_fsdp_config,
    build_interconnects,
    build_model,
    build_model_shape,
    build_parallel_config,
    build_parallel_dims,
    build_pipeline_config,
    build_precision_config,
    gradient_accumulation,
    model_kwargs,
    sequence_length_for,
)
from avgen.config.schema import (
    SCHEMA_VERSION,
    ActivationSpec,
    BucketConfig,
    CheckpointConfig,
    DataConfig,
    EvalConfig,
    FinetuneConfig,
    FSDPSpec,
    InferenceConfig,
    ModelConfig,
    ParallelConfigSpec,
    PipelineSpec,
    PrecisionSpec,
    RLConfig,
    RunConfig,
    TelemetryConfig,
    TrainConfig,
)

__all__ = [
    "SCHEMA_VERSION",
    "ActivationSpec",
    "BucketConfig",
    "CheckpointConfig",
    "ConfigError",
    "DataConfig",
    "EvalConfig",
    "FSDPSpec",
    "FinetuneConfig",
    "InferenceConfig",
    "ModelConfig",
    "ParallelConfigSpec",
    "PipelineSpec",
    "PrecisionSpec",
    "RLConfig",
    "RunConfig",
    "TelemetryConfig",
    "TrainConfig",
    "apply_overrides",
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
    "config_diff",
    "format_diff",
    "gradient_accumulation",
    "load_config",
    "load_mapping",
    "model_kwargs",
    "parse_override",
    "save_config",
    "sequence_length_for",
    "to_mapping",
]
