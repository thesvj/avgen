"""The frozen dataclass tree that describes one avgen run.

**Why plain dataclasses and not a config framework.** Hydra, OmegaConf, and
pydantic all solve real problems, and all of them make a configuration file into
something you have to learn a library to read. A training config is the single
most-read artifact in a research project: it is pasted into issues, diffed
across runs, attached to papers, and read by people who will never install the
framework. So avgen's config is a YAML file whose keys are dataclass field
names, and the only dependency is ``pyyaml``. Interpolation, composition, and
overrides are implemented here — a few hundred lines in :mod:`avgen.config.
loader` — because that is cheaper than the reader's time.

Three properties follow from that decision and are worth defending:

* **The schema is torch-free.** ``import avgen.config.schema`` costs
  milliseconds and works on a machine with no GPU and no torch. ``avgen plan``
  depends on this. Turning config values into live torch objects is
  :mod:`avgen.config.resolve`'s job.
* **Every field validates itself.** A config error is caught before a single
  rank is allocated, and the message names the field and echoes the value.
* **Unknown keys are fatal, not ignored.** See :mod:`avgen.config.loader`.

Structure::

    RunConfig
      ├── model:      ModelConfig
      ├── data:       DataConfig      ── buckets: tuple[BucketConfig, ...]
      ├── train:      TrainConfig
      ├── parallel:   ParallelConfigSpec
      │                 ├── precision:   PrecisionSpec
      │                 ├── activation:  ActivationSpec
      │                 ├── fsdp:        FSDPSpec
      │                 └── pipeline:    PipelineSpec
      ├── checkpoint: CheckpointConfig
      ├── telemetry:  TelemetryConfig
      ├── eval:       EvalConfig
      ├── inference:  InferenceConfig
      ├── finetune:   FinetuneConfig
      └── rl:         RLConfig
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from avgen.config._validate import (
    require_choice,
    require_fraction,
    require_non_negative,
    require_non_negative_int,
    require_path_like,
    require_positive,
    require_positive_int,
    require_unique,
)

__all__ = [
    "ActivationSpec",
    "BucketConfig",
    "CheckpointConfig",
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
]

#: Bumped only when a change to this tree breaks an existing YAML file. A
#: config written by an older avgen carries its version, so the loader can say
#: "this file predates the rename of X" instead of "unknown key X".
SCHEMA_VERSION = 1

DTYPE_NAMES: tuple[str, ...] = ("float32", "bfloat16", "float16")
AC_MODES: tuple[str, ...] = ("none", "full", "selective_op", "selective_layer")
OPTIMIZER_NAMES: tuple[str, ...] = ("adamw", "adamw_bf16_state", "sgd_momentum", "muon")
SCHEDULE_NAMES: tuple[str, ...] = ("constant", "cosine", "linear", "wsd")
TIMESTEP_SAMPLERS: tuple[str, ...] = (
    "uniform",
    "logit_normal",
    "shifted_logit_normal",
    "mode",
)
SAMPLER_NAMES: tuple[str, ...] = ("euler", "heun", "dpmpp_2m", "res_multistep")
PIPELINE_SCHEDULES: tuple[str, ...] = (
    "gpipe",
    "1f1b",
    "interleaved_1f1b",
    "zero_bubble",
)
DATA_SOURCES: tuple[str, ...] = ("synthetic", "latent_shards")
LOGGER_NAMES: tuple[str, ...] = ("console", "jsonl", "tensorboard", "wandb", "noop")
FINETUNE_MODES: tuple[str, ...] = ("lora", "full", "control")
RL_ALGORITHMS: tuple[str, ...] = ("grpo", "dpo")
GPU_NAMES: tuple[str, ...] = ("a100", "h100", "h200", "b200", "cpu")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Architecture of the denoiser.

    The fields here are the ones every avgen model has and that the planner,
    the simulator, and the memory estimator all need to price a run. Anything
    architecture-specific goes in ``extra``, which is passed through to
    :func:`avgen.models.build_model` untouched. That escape hatch is explicit
    rather than implicit: a typo at the top level is an error, a deliberate
    experiment lives under ``extra:`` where a reader can see it is not part of
    the shared contract.

    Args:
        name: Registry name, e.g. ``"video_dit"`` or ``"av_dit"``.
        depth: Number of transformer blocks.
        width: Hidden width.
        num_heads: Attention heads. Must divide ``width``.
        mlp_ratio: Feed-forward expansion factor.
        patch_frames: Temporal patch size applied to the latent.
        patch_height: Spatial patch height applied to the latent.
        patch_width: Spatial patch width applied to the latent.
        in_channels: Latent channels the model consumes.
        out_channels: Latent channels the model predicts. ``-1`` mirrors
            ``in_channels``, which is what flow matching wants.
        cross_attention: Whether blocks carry a text cross-attention sublayer.
            ``False`` means text is injected through adaptive modulation only,
            which is cheaper and materially weaker at prompt following.
        audio_width: Width of the audio stream in an AV model. ``0`` disables
            the audio tower.
        rope_theta: Rotary base frequency. Raise it when training on longer
            sequences than the model was designed for.
        parameters: Optional explicit parameter count. Leave at ``0`` and the
            planner estimates it from depth and width; set it when you know the
            true number and want the simulator to price exactly that.
        extra: Architecture-specific keyword arguments forwarded to the model
            constructor. Deliberately unvalidated.

    Raises:
        ValueError: If a dimension is non-positive, ``num_heads`` does not
            divide ``width``, or a name is empty.
    """

    name: str = "video_dit"
    depth: int = 12
    width: int = 768
    num_heads: int = 12
    mlp_ratio: int = 4
    patch_frames: int = 1
    patch_height: int = 2
    patch_width: int = 2
    in_channels: int = 16
    out_channels: int = -1
    cross_attention: bool = True
    audio_width: int = 0
    rope_theta: float = 10000.0
    parameters: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate architecture dimensions and divisibility."""
        require_path_like("model.name", self.name)
        for name in (
            "depth",
            "width",
            "num_heads",
            "mlp_ratio",
            "patch_frames",
            "patch_height",
            "patch_width",
            "in_channels",
        ):
            require_positive_int(f"model.{name}", getattr(self, name))
        require_non_negative_int("model.audio_width", self.audio_width)
        require_non_negative_int("model.parameters", self.parameters)
        require_positive("model.rope_theta", self.rope_theta)
        if self.out_channels != -1:
            require_positive_int("model.out_channels", self.out_channels)
        if self.width % self.num_heads != 0:
            raise ValueError(
                f"model.num_heads must divide model.width; "
                f"got width={self.width} num_heads={self.num_heads} "
                f"(remainder {self.width % self.num_heads})"
            )
        if not isinstance(self.extra, dict):
            raise TypeError(f"model.extra must be a mapping; got {self.extra!r}")

    @property
    def head_dim(self) -> int:
        """Width of one attention head."""
        return self.width // self.num_heads

    @property
    def latent_out_channels(self) -> int:
        """Resolved output channel count."""
        return self.in_channels if self.out_channels == -1 else self.out_channels

    @property
    def has_audio(self) -> bool:
        """Whether this configuration describes an audio-video model."""
        return self.audio_width > 0

    def estimated_parameters(self) -> int:
        """Estimate the parameter count from the architecture.

        Counts, per block: the four attention projections (``4 w^2``), the
        SwiGLU feed-forward (``3 r w^2``), optional cross-attention
        (``4 w^2``), and adaptive-layernorm modulation, which produces six
        ``w``-sized vectors from the conditioning embedding and therefore costs
        ``6 w^2``. That last term is the one people forget; on a DiT it is a
        fifth of the model.

        Returns:
            The explicit ``parameters`` field when set, otherwise the estimate.
        """
        if self.parameters:
            return self.parameters
        width = float(self.width)
        per_block = 4.0 * width * width  # q, k, v, out
        per_block += 3.0 * self.mlp_ratio * width * width  # gate, up, down
        per_block += 6.0 * width * width  # adaLN modulation
        if self.cross_attention:
            per_block += 4.0 * width * width
        patch_dim = (
            self.in_channels * self.patch_frames * self.patch_height * self.patch_width
        )
        embedding = patch_dim * width + width * patch_dim  # patch embed + final proj
        embedding += 4.0 * width * width  # time embedding MLP + text projection
        return int(per_block * self.depth + embedding)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BucketConfig:
    """One resolution/duration bucket the loader may emit.

    Bucketing exists because padding a 2-second clip out to 10 seconds wastes
    80% of the compute on masked tokens. Samples are grouped so that every
    microbatch has one shape, and the shapes are chosen so that token counts
    are comparable across buckets — otherwise step time swings with whichever
    bucket the sampler happened to draw.

    Args:
        name: Label used in logs and in the bucket sampler's statistics.
        frames: Latent frames (post temporal compression).
        height: Latent rows (post spatial compression).
        width: Latent columns (post spatial compression).
        micro_batch_size: Samples per rank per microbatch for this bucket.
        weight: Relative sampling probability. Normalised across buckets.

    Raises:
        ValueError: If any extent is non-positive or the weight is negative.
    """

    name: str = "default"
    frames: int = 16
    height: int = 32
    width: int = 32
    micro_batch_size: int = 1
    weight: float = 1.0

    def __post_init__(self) -> None:
        """Validate the bucket geometry."""
        require_path_like("data.buckets[].name", self.name)
        for name in ("frames", "height", "width", "micro_batch_size"):
            require_positive_int(
                f"data.buckets[{self.name}].{name}", getattr(self, name)
            )
        require_non_negative(f"data.buckets[{self.name}].weight", self.weight)

    def tokens(self, model: ModelConfig) -> int:
        """Sequence length this bucket produces for a given patch size.

        Args:
            model: Model config supplying the patch sizes.

        Returns:
            Token count for one sample.

        Raises:
            ValueError: If a patch size does not divide its axis. Ragged
                patchification is rejected here rather than producing a
                truncated sequence three hours into a run.
        """
        for axis, patch in (
            ("frames", model.patch_frames),
            ("height", model.patch_height),
            ("width", model.patch_width),
        ):
            extent = getattr(self, axis)
            if extent % patch != 0:
                raise ValueError(
                    f"data.buckets[{self.name}].{axis}={extent} is not divisible "
                    f"by model.patch_{axis if axis != 'frames' else 'frames'}"
                    f"={patch}; adjust the bucket or the patch size"
                )
        return (
            (self.frames // model.patch_frames)
            * (self.height // model.patch_height)
            * (self.width // model.patch_width)
        )


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Where training samples come from and what shape they arrive in.

    This file ships **templates only**. Paths, dataset names, and mixture
    weights are yours to fill in; nothing dataset-specific is baked into avgen.

    Args:
        source: ``"synthetic"`` for the dependency-free generator used by
            smoke tests, ``"latent_shards"`` for pre-encoded shards on disk.
        root: Directory holding shards. Ignored by the synthetic source.
        manifest: Optional shard manifest file. When absent the loader scans
            ``root``, which is slower and non-deterministic across filesystems.
        buckets: Resolution/duration buckets. At least one.
        num_workers: Loader worker processes per rank.
        prefetch_batches: Batches queued ahead of the trainer.
        shuffle_buffer: Samples held for reservoir shuffling. Zero disables
            shuffling, which is only correct if the shards are already shuffled.
        seed: Shuffling seed. Combined with ``data_rank`` so that ranks see
            disjoint data.
        drop_last: Whether to discard a trailing partial batch. Keep it true:
            a short final batch changes the effective batch size for one step.
        latent_channels: Channels in a stored video latent.
        text_tokens: Length of the frozen text-encoder context.
        text_width: Width of the frozen text-encoder features.
        audio_frames: Latent audio frames per sample. Zero means video only.
        audio_channels: Channels in a stored audio latent.
        caption_dropout: Probability a sample's text is replaced by the null
            embedding during training, which is what makes classifier-free
            guidance possible at inference. Below ~0.05 guidance is weak; above
            ~0.2 prompt following degrades.
        repeat: Number of passes over the dataset before it is exhausted. Zero
            means stream forever, which is the norm for large runs.

    Raises:
        ValueError: If a count is negative, no bucket is defined, or bucket
            names repeat.
    """

    source: str = "synthetic"
    root: str = ""
    manifest: str = ""
    buckets: tuple[BucketConfig, ...] = (BucketConfig(),)
    num_workers: int = 2
    prefetch_batches: int = 2
    shuffle_buffer: int = 1024
    seed: int = 0
    drop_last: bool = True
    latent_channels: int = 16
    text_tokens: int = 226
    text_width: int = 4096
    audio_frames: int = 0
    audio_channels: int = 8
    caption_dropout: float = 0.1
    repeat: int = 0

    def __post_init__(self) -> None:
        """Validate the source, the buckets, and the loader counts."""
        require_choice("data.source", self.source, DATA_SOURCES)
        for name in ("num_workers", "prefetch_batches", "shuffle_buffer", "repeat"):
            require_non_negative_int(f"data.{name}", getattr(self, name))
        require_non_negative_int("data.seed", self.seed)
        for name in ("latent_channels", "text_tokens", "text_width", "audio_channels"):
            require_positive_int(f"data.{name}", getattr(self, name))
        require_non_negative_int("data.audio_frames", self.audio_frames)
        require_fraction("data.caption_dropout", self.caption_dropout)
        if not self.buckets:
            raise ValueError("data.buckets must contain at least one bucket")
        require_unique("data.buckets", [bucket.name for bucket in self.buckets])
        if self.source == "latent_shards" and not self.root:
            raise ValueError(
                "data.root must be set when data.source='latent_shards'; "
                "point it at the directory holding your encoded shards"
            )

    @property
    def has_audio(self) -> bool:
        """Whether batches carry an audio stream."""
        return self.audio_frames > 0

    def bucket(self, name: str) -> BucketConfig:
        """Look up a bucket by name.

        Args:
            name: Bucket name.

        Returns:
            The bucket.

        Raises:
            KeyError: If no bucket carries that name.
        """
        for bucket in self.buckets:
            if bucket.name == name:
                return bucket
        raise KeyError(
            f"no bucket named {name!r}; defined: "
            f"{', '.join(b.name for b in self.buckets)}"
        )

    def largest_bucket(self) -> BucketConfig:
        """Return the bucket with the most latent elements.

        This is the bucket that decides whether the run fits in memory, so it
        is the one the planner and the simulator price.

        Returns:
            The largest bucket by element count.
        """
        return max(self.buckets, key=lambda b: b.frames * b.height * b.width)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """The optimisation loop.

    Args:
        steps: Optimizer steps to run.
        global_batch_size: Samples per optimizer step across the whole job.
            The gradient accumulation factor is derived from this and the
            per-rank microbatch, never the other way round — deriving it the
            other way lets the effective batch size drift when the world size
            changes, and two runs stop being comparable.
        micro_batch_size: Samples per rank per microbatch. ``-1`` takes the
            value from the active data bucket.
        optimizer: Optimizer family.
        lr: Peak learning rate.
        weight_decay: Decoupled weight decay. Never applied to norms, biases,
            or modulation parameters; see
            :func:`avgen.train.optimizer.build_optimizer`.
        beta1: First moment decay.
        beta2: Second moment decay. 0.95 rather than 0.999 is the usual choice
            for diffusion transformers: the gradient distribution is heavier
            tailed than in language modelling and a long second-moment memory
            makes the optimizer slow to react to a bad batch.
        eps: Optimizer epsilon.
        max_grad_norm: Global gradient-norm clip. Zero disables clipping.
        schedule: Learning-rate schedule name.
        warmup_steps: Steps of linear warmup.
        decay_ratio: For ``wsd``, the fraction of total steps spent decaying.
        min_lr_ratio: Floor of the schedule as a fraction of ``lr``.
        objective: Training objective name.
        timestep_sampler: Timestep distribution name.
        logit_normal_mean: Mean of the logit-normal timestep distribution.
        logit_normal_std: Standard deviation of the same.
        base_shift: Timestep shift at ``base_seq_len``.
        max_shift: Timestep shift at ``max_seq_len``. Resolution-dependent
            shift is not optional for video: the same noise level destroys far
            more information in a 100k-token sequence than in a 4k one, so a
            single fixed schedule trains the two ends of the range badly.
        base_seq_len: Sequence length the base shift is calibrated at.
        max_seq_len: Sequence length the max shift is calibrated at.
        audio_loss_weight: Weight of the audio term in an AV objective.
        ema_decay: EMA decay. Zero disables the EMA entirely.
        ema_warmup_steps: Steps before the EMA starts tracking.
        seed: Master seed. Varies on ``data_rank`` only.
        gradient_checkpointing: Deprecated alias kept out; use
            ``parallel.activation.mode``.
        log_every: Steps between metric syncs. Every sync costs a device
            synchronisation, so this is a real throughput knob.
        eval_every: Steps between in-training evaluations. Zero disables.
        seed_deterministic: Whether to force deterministic kernels. Costs
            throughput; buys bit-exact reproduction.

    Raises:
        ValueError: If a rate is non-positive, a count is negative, or the
            batch sizes are inconsistent.
    """

    steps: int = 1000
    global_batch_size: int = 8
    micro_batch_size: int = -1
    optimizer: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    schedule: str = "cosine"
    warmup_steps: int = 100
    decay_ratio: float = 0.1
    min_lr_ratio: float = 0.1
    objective: str = "flow_matching"
    timestep_sampler: str = "shifted_logit_normal"
    logit_normal_mean: float = 0.0
    logit_normal_std: float = 1.0
    base_shift: float = 0.5
    max_shift: float = 1.15
    base_seq_len: int = 256
    max_seq_len: int = 4096
    audio_loss_weight: float = 1.0
    ema_decay: float = 0.9999
    ema_warmup_steps: int = 0
    seed: int = 0
    log_every: int = 10
    eval_every: int = 0
    seed_deterministic: bool = False

    def __post_init__(self) -> None:
        """Validate rates, counts, and schedule parameters."""
        require_positive_int("train.steps", self.steps)
        require_positive_int("train.global_batch_size", self.global_batch_size)
        if self.micro_batch_size != -1:
            require_positive_int("train.micro_batch_size", self.micro_batch_size)
        require_choice("train.optimizer", self.optimizer, OPTIMIZER_NAMES)
        require_choice("train.schedule", self.schedule, SCHEDULE_NAMES)
        require_choice(
            "train.timestep_sampler", self.timestep_sampler, TIMESTEP_SAMPLERS
        )
        require_positive("train.lr", self.lr)
        require_positive("train.eps", self.eps)
        require_non_negative("train.weight_decay", self.weight_decay)
        require_non_negative("train.max_grad_norm", self.max_grad_norm)
        require_non_negative("train.audio_loss_weight", self.audio_loss_weight)
        require_fraction("train.beta1", self.beta1)
        require_fraction("train.beta2", self.beta2)
        require_fraction("train.decay_ratio", self.decay_ratio)
        require_fraction("train.min_lr_ratio", self.min_lr_ratio)
        require_fraction("train.ema_decay", self.ema_decay)
        require_positive("train.logit_normal_std", self.logit_normal_std)
        require_positive("train.base_shift", self.base_shift)
        require_positive("train.max_shift", self.max_shift)
        require_positive_int("train.base_seq_len", self.base_seq_len)
        require_positive_int("train.max_seq_len", self.max_seq_len)
        for name in ("warmup_steps", "ema_warmup_steps", "eval_every", "seed"):
            require_non_negative_int(f"train.{name}", getattr(self, name))
        require_positive_int("train.log_every", self.log_every)
        if self.warmup_steps >= self.steps:
            raise ValueError(
                f"train.warmup_steps={self.warmup_steps} must be less than "
                f"train.steps={self.steps}; the schedule would never leave warmup"
            )
        if self.max_seq_len <= self.base_seq_len:
            raise ValueError(
                f"train.max_seq_len={self.max_seq_len} must exceed "
                f"train.base_seq_len={self.base_seq_len}; the shift is "
                "interpolated between the two and a degenerate interval makes "
                "it undefined"
            )
        if self.beta2 <= self.beta1:
            raise ValueError(
                f"train.beta2={self.beta2} must exceed train.beta1={self.beta1}"
            )

    @property
    def ema_enabled(self) -> bool:
        """Whether an exponential moving average is maintained."""
        return self.ema_decay > 0.0

    @property
    def betas(self) -> tuple[float, float]:
        """Optimizer betas as the tuple torch expects."""
        return (self.beta1, self.beta2)


# ---------------------------------------------------------------------------
# Parallelism
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PrecisionSpec:
    """Mixed-precision policy, mirroring :class:`avgen.parallel.PrecisionConfig`.

    Args:
        param_dtype: Compute dtype for weights and activations.
        reduce_dtype: Dtype gradients are reduced in. Keep it float32: the
            reduction tree is ``log(world_size)`` deep, so bf16 reduction error
            grows with job size, which is exactly the wrong direction.
        enable_float8: Whether eligible linear layers are swapped for float8.
        float8_min_features: Size threshold below which float8 loses.
        float8_recipe: ``tensorwise`` (robust) or ``rowwise`` (accurate).

    Raises:
        ValueError: If a dtype name is unknown.
    """

    param_dtype: str = "bfloat16"
    reduce_dtype: str = "float32"
    enable_float8: bool = False
    float8_min_features: int = 1024
    float8_recipe: str = "tensorwise"

    def __post_init__(self) -> None:
        """Validate dtype names and the float8 threshold."""
        require_choice("parallel.precision.param_dtype", self.param_dtype, DTYPE_NAMES)
        require_choice(
            "parallel.precision.reduce_dtype", self.reduce_dtype, DTYPE_NAMES
        )
        require_choice(
            "parallel.precision.float8_recipe",
            self.float8_recipe,
            ("tensorwise", "rowwise"),
        )
        require_positive_int(
            "parallel.precision.float8_min_features", self.float8_min_features
        )

    @property
    def param_dtype_bytes(self) -> int:
        """Bytes per parameter in compute precision, for the memory estimator."""
        return 4 if self.param_dtype == "float32" else 2


@dataclass(frozen=True, slots=True)
class ActivationSpec:
    """Activation-checkpointing policy.

    Args:
        mode: One of ``none``, ``selective_op``, ``selective_layer``, ``full``.
        layer_interval: For ``selective_layer``, checkpoint one block in every
            ``layer_interval``.
        save_op_frequency: For ``selective_op``, save every n-th occurrence of
            a saveable op.

    Raises:
        ValueError: If the mode is unknown or an interval is below one.
    """

    mode: str = "selective_op"
    layer_interval: int = 2
    save_op_frequency: int = 1

    def __post_init__(self) -> None:
        """Validate the mode and its intervals."""
        require_choice("parallel.activation.mode", self.mode, AC_MODES)
        require_positive_int("parallel.activation.layer_interval", self.layer_interval)
        require_positive_int(
            "parallel.activation.save_op_frequency", self.save_op_frequency
        )


@dataclass(frozen=True, slots=True)
class FSDPSpec:
    """FSDP2 sharding options.

    Args:
        reshard_after_forward: ZeRO-3 behaviour. False is ZeRO-2: faster, and
            the model must fit unsharded on one rank.
        cpu_offload: Keep sharded parameters in host memory. A last resort that
            moves the bottleneck to PCIe.
        shard_last_block_after_forward: Whether the final block reshards. Left
            false because backward needs its parameters immediately.
        ignore_frozen_params: Exclude frozen parameters from sharding. A large
            win for adapter fine-tuning, where the frozen base dominates.
    """

    reshard_after_forward: bool = True
    cpu_offload: bool = False
    shard_last_block_after_forward: bool = False
    ignore_frozen_params: bool = False


@dataclass(frozen=True, slots=True)
class PipelineSpec:
    """Pipeline schedule options.

    Args:
        schedule: Schedule name.
        microbatches: Microbatches per optimizer step. Must be at least the
            number of stages or the bubble dominates.
        stages_per_rank: Virtual stages per rank; greater than one requires an
            interleaved schedule.

    Raises:
        ValueError: If the schedule is unknown or a count is below one.
    """

    schedule: str = "1f1b"
    microbatches: int = 8
    stages_per_rank: int = 1

    def __post_init__(self) -> None:
        """Validate the schedule name and its counts."""
        require_choice("parallel.pipeline.schedule", self.schedule, PIPELINE_SCHEDULES)
        require_positive_int("parallel.pipeline.microbatches", self.microbatches)
        require_positive_int("parallel.pipeline.stages_per_rank", self.stages_per_rank)
        if self.stages_per_rank > 1 and self.schedule in ("gpipe", "1f1b"):
            raise ValueError(
                f"parallel.pipeline.schedule={self.schedule!r} is single-stage "
                f"but stages_per_rank={self.stages_per_rank}; use "
                "'interleaved_1f1b' or 'zero_bubble'"
            )


@dataclass(frozen=True, slots=True)
class ParallelConfigSpec:
    """The five parallelism degrees plus the policies that ride on them.

    Named ``ParallelConfigSpec`` rather than ``ParallelConfig`` on purpose:
    :class:`avgen.parallel.ParallelConfig` is the *live* object holding torch
    types, and this is its torch-free description. Keeping the two names
    distinct stops an import cycle and stops a reader assuming a YAML key maps
    to a torch dtype.

    Args:
        dp_replicate: Replicated data-parallel degree. Greater than one is HSDP.
        dp_shard: Sharded data-parallel degree; ``-1`` infers the remainder.
        tensor: Tensor-parallel degree. Should never exceed GPUs per node.
        context: Context-parallel degree. The axis that matters for video.
        pipeline: Pipeline-parallel degree. The last axis to reach for.
        enable_loss_parallel: Compute the loss on sharded activations.
        sequence_parallel: Shard norms and residuals along the sequence under
            tensor parallelism. Leave on — without it TP saves weight memory
            but not activation memory, and for video the activations are the
            problem.
        compile_blocks: ``torch.compile`` each transformer block. Compiling per
            block rather than whole-model reuses one graph ``depth`` times.
        compile_mode: Inductor mode.
        precision: Mixed-precision policy.
        activation: Activation-checkpointing policy.
        fsdp: Sharding options.
        pipeline_schedule: Pipeline schedule options.
        gpus_per_node: Ranks sharing the fast fabric. Used by the planner to
            keep tensor parallelism inside a node.

    Raises:
        ValueError: If a degree is invalid or tensor parallelism would cross a
            node boundary.
    """

    dp_replicate: int = 1
    dp_shard: int = -1
    tensor: int = 1
    context: int = 1
    pipeline: int = 1
    enable_loss_parallel: bool = True
    sequence_parallel: bool = True
    compile_blocks: bool = False
    compile_mode: str = "default"
    gpus_per_node: int = 8
    precision: PrecisionSpec = field(default_factory=PrecisionSpec)
    activation: ActivationSpec = field(default_factory=ActivationSpec)
    fsdp: FSDPSpec = field(default_factory=FSDPSpec)
    pipeline_schedule: PipelineSpec = field(default_factory=PipelineSpec)

    def __post_init__(self) -> None:
        """Validate the degrees and the intra-node tensor-parallel constraint."""
        for name in ("dp_replicate", "tensor", "context", "pipeline", "gpus_per_node"):
            require_positive_int(f"parallel.{name}", getattr(self, name))
        if self.dp_shard != -1:
            require_positive_int("parallel.dp_shard", self.dp_shard)
        require_choice(
            "parallel.compile_mode",
            self.compile_mode,
            (
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ),
        )
        if self.tensor > self.gpus_per_node:
            raise ValueError(
                f"parallel.tensor={self.tensor} exceeds "
                f"parallel.gpus_per_node={self.gpus_per_node}; tensor parallelism "
                "communicates twice per block on the critical path and must stay "
                "inside one NVLink domain"
            )

    @property
    def model_parallel_degree(self) -> int:
        """Product of the axes a single model replica is spread across."""
        return self.tensor * self.context * self.pipeline


# ---------------------------------------------------------------------------
# Checkpointing, telemetry
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CheckpointConfig:
    """Where state is written and how much of it is kept.

    Args:
        dir: Checkpoint directory. Empty means ``<output_dir>/checkpoints``.
        save_every: Steps between checkpoints. Zero disables saving, which is
            only sane for a smoke test.
        keep_last_n: Recent checkpoints retained. Zero keeps all.
        keep_every_n_steps: Additionally retain every n-th checkpoint forever, so a
            long run leaves a usable trajectory rather than only its tail.
        async_save: Whether to stage the save and return immediately. On a
            large model this is the difference between a 90-second stall and a
            2-second one, every save.
        resume: ``"latest"``, an explicit path, or empty for a fresh run.
        load_model_only: Restore weights but not optimizer, schedule, or data
            cursor. What you want when starting a fine-tune from a base model;
            never what you want when resuming an interrupted run.
        export_dtype: dtype for ``avgen checkpoint export``. Empty keeps the
            training dtype.
        save_config: Write the fully resolved config next to every checkpoint.
            Leave this on. A checkpoint whose config is not beside it is a
            checkpoint nobody can reproduce.

    Raises:
        ValueError: If a count is negative or the dtype name is unknown.
    """

    dir: str = ""
    save_every: int = 1000
    keep_last_n: int = 3
    keep_every_n_steps: int = 0
    async_save: bool = True
    resume: str = ""
    load_model_only: bool = False
    export_dtype: str = ""
    save_config: bool = True

    def __post_init__(self) -> None:
        """Validate retention counts and the export dtype."""
        for name in ("save_every", "keep_last_n", "keep_every_n_steps"):
            require_non_negative_int(f"checkpoint.{name}", getattr(self, name))
        if self.export_dtype:
            require_choice("checkpoint.export_dtype", self.export_dtype, DTYPE_NAMES)


@dataclass(frozen=True, slots=True)
class TelemetryConfig:
    """Where metrics go and how often.

    Args:
        loggers: Backend names. ``console`` and ``jsonl`` have no optional
            dependency; ``tensorboard`` and ``wandb`` do and will raise a
            precise error if selected without the extra installed.
        project: Project name for the tracking backends.
        run_name: Run name. Empty derives one from the config and the date.
        log_every: Steps between metric syncs. Should match
            ``train.log_every``; when they disagree the larger one wins because
            a sync is the expensive part.
        jsonl_path: Destination for the jsonl logger. Empty means
            ``<output_dir>/metrics.jsonl``.
        tensorboard_dir: Destination for tensorboard events.
        profile_steps: Steps to capture with ``torch.profiler``. Zero disables.
        profile_warmup: Steps skipped before profiling. Never profile step
            zero: it contains lazy init, autotuning, and the first compile,
            none of which recur.
        memory_report_every: Steps between memory reports. Zero disables.
        report_mfu: Whether to compute and log Model FLOPs Utilisation.

    Raises:
        ValueError: If a logger name is unknown or repeated.
    """

    loggers: tuple[str, ...] = ("console", "jsonl")
    project: str = "avgen"
    run_name: str = ""
    log_every: int = 10
    jsonl_path: str = ""
    tensorboard_dir: str = ""
    profile_steps: int = 0
    profile_warmup: int = 5
    memory_report_every: int = 0
    report_mfu: bool = True

    def __post_init__(self) -> None:
        """Validate logger names and cadences."""
        require_unique("telemetry.loggers", self.loggers)
        for name in self.loggers:
            require_choice("telemetry.loggers[]", name, LOGGER_NAMES)
        require_positive_int("telemetry.log_every", self.log_every)
        for name in ("profile_steps", "profile_warmup", "memory_report_every"):
            require_non_negative_int(f"telemetry.{name}", getattr(self, name))


# ---------------------------------------------------------------------------
# Evaluation and inference
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EvalConfig:
    """What is measured, on what, and under exactly which sampling settings.

    Every field in the "pin" group below is recorded in the
    :class:`avgen.eval.EvalReport` and checked before two reports are compared.
    That is not bureaucracy: a metric computed at 30 sampling steps and one
    computed at 50 are different numbers, and the published video-generation
    literature is full of tables that compare them anyway.

    Args:
        metrics: Metric names to compute. Dependency-free built-ins are always
            available; gated backends raise a named error if their extra is not
            installed, and are never silently substituted.
        prompts_file: Newline-delimited prompt file. Empty uses the built-in
            deterministic prompt list, which is fine for smoke tests and
            meaningless for a paper number.
        num_prompts: Prompts to evaluate. Zero uses all of them.
        batch_size: Prompts generated per forward pass.
        seed: Sampling seed. Part of the pin.
        steps: Sampler steps. Part of the pin.
        guidance: Classifier-free guidance scale. Part of the pin.
        sampler: Sampler name. Part of the pin.
        negative_prompt: Negative prompt used for the unconditional branch.
            Part of the pin — a comparison against a run with a different
            negative prompt is not a comparison of the models.
        frames: Latent frames to generate. Part of the pin.
        height: Latent rows to generate. Part of the pin.
        width: Latent columns to generate. Part of the pin.
        decode: Whether latents are decoded to pixels before metrics run.
            Pixel-space metrics (saturation, sharpness) are meaningless
            otherwise and are skipped with a note rather than reported wrong.
        output_dir: Where reports and samples land.
        save_samples: Whether generated media is written alongside the report.
        baseline_report: A previous report to compare against. The comparison
            refuses if the pins differ.

    Raises:
        ValueError: If a metric name repeats or a sampling parameter is invalid.
    """

    metrics: tuple[str, ...] = (
        "temporal_consistency",
        "motion_magnitude",
        "flicker_index",
        "static_frames",
    )
    prompts_file: str = ""
    num_prompts: int = 0
    batch_size: int = 1
    seed: int = 0
    steps: int = 30
    guidance: float = 5.0
    sampler: str = "euler"
    negative_prompt: str = ""
    frames: int = 16
    height: int = 32
    width: int = 32
    decode: bool = False
    output_dir: str = ""
    save_samples: bool = False
    baseline_report: str = ""

    def __post_init__(self) -> None:
        """Validate the metric list and every pinned sampling parameter."""
        require_unique("eval.metrics", self.metrics)
        if not self.metrics:
            raise ValueError("eval.metrics must name at least one metric")
        require_choice("eval.sampler", self.sampler, SAMPLER_NAMES)
        require_positive_int("eval.steps", self.steps)
        require_positive_int("eval.batch_size", self.batch_size)
        for name in ("frames", "height", "width"):
            require_positive_int(f"eval.{name}", getattr(self, name))
        require_non_negative_int("eval.num_prompts", self.num_prompts)
        require_non_negative_int("eval.seed", self.seed)
        require_non_negative("eval.guidance", self.guidance)


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    """Defaults for ``avgen generate``.

    Command-line flags override these; the config exists so that a project can
    pin its house sampling settings once instead of retyping them.

    Args:
        sampler: Sampler name.
        steps: Sampler steps.
        guidance: Classifier-free guidance scale. 1.0 disables guidance and
            halves the cost, because the unconditional branch is no longer
            evaluated.
        guidance_rescale: CFG-rescale factor, which counteracts the
            over-saturation high guidance produces. Zero disables it.
        negative_prompt: Default negative prompt.
        seed: Default seed. Negative means draw a fresh one per call.
        frames: Latent frames to generate.
        height: Latent rows.
        width: Latent columns.
        fps: Frames per second written into the output container.
        batch_size: Prompts per forward pass.
        dtype: Inference dtype.
        output: Default output path.

    Raises:
        ValueError: If a sampling parameter is invalid.
    """

    sampler: str = "euler"
    steps: int = 30
    guidance: float = 5.0
    guidance_rescale: float = 0.0
    negative_prompt: str = ""
    seed: int = -1
    frames: int = 16
    height: int = 32
    width: int = 32
    fps: float = 16.0
    batch_size: int = 1
    dtype: str = "bfloat16"
    output: str = "samples"

    def __post_init__(self) -> None:
        """Validate sampler settings and output geometry."""
        require_choice("inference.sampler", self.sampler, SAMPLER_NAMES)
        require_choice("inference.dtype", self.dtype, DTYPE_NAMES)
        require_positive_int("inference.steps", self.steps)
        require_positive_int("inference.batch_size", self.batch_size)
        for name in ("frames", "height", "width"):
            require_positive_int(f"inference.{name}", getattr(self, name))
        require_non_negative("inference.guidance", self.guidance)
        require_fraction("inference.guidance_rescale", self.guidance_rescale)
        require_positive("inference.fps", self.fps)


# ---------------------------------------------------------------------------
# Fine-tuning and RL
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FinetuneConfig:
    """Adapter, full, and control-tower fine-tuning.

    Args:
        mode: ``lora``, ``full``, or ``control``.
        base_checkpoint: Checkpoint the fine-tune starts from. Required for
            every mode — a fine-tune without a base is just training.
        lora_rank: LoRA rank. The dominant quality/size knob; 16-64 covers most
            of the useful range for a video DiT.
        lora_alpha: LoRA scaling numerator. The effective scale is
            ``alpha / rank``, so raising rank without raising alpha quietly
            lowers the adapter's influence.
        lora_dropout: Dropout on the adapter path.
        lora_targets: Substrings of module names the adapter attaches to.
        use_dora: Whether to use weight-decomposed LoRA, which separates
            magnitude from direction and closes much of the gap to full
            fine-tuning at the cost of a slower step.
        freeze_patterns: Substrings of parameter names frozen in ``full`` mode.
        control_channels: Input channels of the control signal.
        control_blocks: Blocks the control tower copies. ``-1`` copies half the
            depth, which is the usual ControlNet arrangement.
        control_scale: Scale applied to the control tower's residuals.
        train_norms: Whether normalisation parameters stay trainable during
            adapter training. Cheap, and usually worth a visible quality gain.

    Raises:
        ValueError: If the mode is unknown or a rank is non-positive.
    """

    mode: str = "lora"
    base_checkpoint: str = ""
    lora_rank: int = 32
    lora_alpha: float = 32.0
    lora_dropout: float = 0.0
    lora_targets: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    use_dora: bool = False
    freeze_patterns: tuple[str, ...] = ()
    control_channels: int = 3
    control_blocks: int = -1
    control_scale: float = 1.0
    train_norms: bool = True

    def __post_init__(self) -> None:
        """Validate the mode and the adapter hyper-parameters."""
        require_choice("finetune.mode", self.mode, FINETUNE_MODES)
        require_positive_int("finetune.lora_rank", self.lora_rank)
        require_positive("finetune.lora_alpha", self.lora_alpha)
        require_fraction("finetune.lora_dropout", self.lora_dropout)
        require_positive_int("finetune.control_channels", self.control_channels)
        require_non_negative("finetune.control_scale", self.control_scale)
        if self.control_blocks != -1:
            require_positive_int("finetune.control_blocks", self.control_blocks)
        if self.mode == "lora" and not self.lora_targets:
            raise ValueError(
                "finetune.lora_targets is empty; a LoRA fine-tune with no target "
                "modules trains nothing"
            )
        require_unique("finetune.lora_targets", self.lora_targets)


@dataclass(frozen=True, slots=True)
class RLConfig:
    """Preference and reward-based post-training.

    Args:
        algorithm: ``grpo`` (group-relative policy optimisation over an
            SDE-converted sampler) or ``dpo`` (pairwise preference loss).
        reference_checkpoint: Frozen reference policy. Empty uses a snapshot of
            the initial weights, which is what the KL term is measured against.
        group_size: Samples per prompt for GRPO. The advantage is computed
            within the group, so a group of one has zero variance signal and
            learns nothing; four is the practical floor.
        sampler_steps: Denoising steps used to roll out a sample. The dominant
            cost of RL post-training, and the first thing to cut.
        kl_coefficient: Weight of the KL-to-reference penalty. The knob that
            decides whether reward hacking or reward learning happens.
        clip_range: PPO-style ratio clip.
        advantage_normalize: Whether advantages are standardised within a group.
        mixgrpo_window: Number of consecutive denoising steps optimised per
            update. Zero optimises all of them, which is correct and expensive;
            a window trades a little bias for a large speedup.
        sde_noise_scale: Noise injected when converting the deterministic ODE
            sampler into an SDE. Exploration is impossible without it.
        dpo_beta: Temperature of the DPO loss. Larger keeps the policy closer
            to the reference.
        rewards: Reward model names.
        reward_weights: Weight per reward. Must match ``rewards`` in length.

    Raises:
        ValueError: If the algorithm is unknown or the reward weights do not
            line up with the reward names.
    """

    algorithm: str = "grpo"
    reference_checkpoint: str = ""
    group_size: int = 8
    sampler_steps: int = 16
    kl_coefficient: float = 0.04
    clip_range: float = 0.2
    advantage_normalize: bool = True
    mixgrpo_window: int = 0
    sde_noise_scale: float = 0.7
    dpo_beta: float = 0.1
    rewards: tuple[str, ...] = ()
    reward_weights: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        """Validate the algorithm and the reward composition."""
        require_choice("rl.algorithm", self.algorithm, RL_ALGORITHMS)
        require_positive_int("rl.group_size", self.group_size)
        require_positive_int("rl.sampler_steps", self.sampler_steps)
        require_non_negative("rl.kl_coefficient", self.kl_coefficient)
        require_positive("rl.clip_range", self.clip_range)
        require_non_negative("rl.sde_noise_scale", self.sde_noise_scale)
        require_positive("rl.dpo_beta", self.dpo_beta)
        require_non_negative_int("rl.mixgrpo_window", self.mixgrpo_window)
        require_unique("rl.rewards", self.rewards)
        if self.reward_weights and len(self.reward_weights) != len(self.rewards):
            raise ValueError(
                f"rl.reward_weights has {len(self.reward_weights)} entries but "
                f"rl.rewards has {len(self.rewards)}; a composite reward with a "
                "mismatched weight vector silently reweights the objective"
            )
        if self.algorithm == "grpo" and self.group_size < 2:
            raise ValueError(
                f"rl.group_size={self.group_size} gives GRPO no within-group "
                "variance to form an advantage from; use at least 4"
            )


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunConfig:
    """One complete avgen run.

    Args:
        schema_version: Version of this dataclass tree. The loader refuses a
            file from a newer schema rather than dropping the fields it does
            not recognise.
        run_name: Human label for the run. Empty derives one.
        output_dir: Root for checkpoints, logs, samples, and the resolved
            config copy.
        seed: Master seed. Varies on ``data_rank`` only; CP and TP ranks
            holding shards of one sample must draw identical noise.
        model: Architecture.
        data: Input pipeline.
        train: Optimisation loop.
        parallel: Parallelism degrees and policies.
        checkpoint: Persistence.
        telemetry: Metrics and profiling.
        eval: Evaluation suite.
        inference: Generation defaults.
        finetune: Fine-tuning options, used by ``avgen finetune``.
        rl: Post-training options, used by ``avgen rl``.
        notes: Free-text provenance, carried into the saved config. Use it for
            the thing you will want to know in six months.

    Raises:
        ValueError: If the schema version is unsupported or a cross-section
            constraint is violated.
    """

    schema_version: int = SCHEMA_VERSION
    run_name: str = ""
    output_dir: str = "runs/default"
    seed: int = 0
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    parallel: ParallelConfigSpec = field(default_factory=ParallelConfigSpec)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    notes: str = ""

    def __post_init__(self) -> None:
        """Validate the schema version and the cross-section constraints."""
        require_positive_int("schema_version", self.schema_version)
        if self.schema_version > SCHEMA_VERSION:
            raise ValueError(
                f"schema_version={self.schema_version} is newer than this avgen "
                f"understands ({SCHEMA_VERSION}); upgrade avgen rather than "
                "editing the file, because the missing fields have defaults you "
                "did not choose"
            )
        require_non_negative_int("seed", self.seed)
        require_path_like("output_dir", self.output_dir)

        # An AV model with no audio in the batch trains its whole audio tower
        # on zero-length tensors and reports a plausible-looking video loss.
        if self.model.has_audio and not self.data.has_audio:
            raise ValueError(
                f"model.audio_width={self.model.audio_width} declares an audio "
                "tower but data.audio_frames=0; the audio branch would train on "
                "empty tensors. Set data.audio_frames or model.audio_width=0"
            )
        # Every bucket must patchify exactly; raise now, not at step 1. A
        # ragged patch grid is otherwise discovered by the first forward pass,
        # after the allocator, the loader, and the mesh have all been built.
        for bucket in self.data.buckets:
            bucket.tokens(self.model)

    @property
    def sequence_length(self) -> int:
        """Token count of the largest bucket, which is what has to fit."""
        return self.data.largest_bucket().tokens(self.model)

    @property
    def micro_batch_size(self) -> int:
        """Resolved per-rank microbatch size."""
        if self.train.micro_batch_size != -1:
            return self.train.micro_batch_size
        return self.data.largest_bucket().micro_batch_size

    def resolved_run_name(self) -> str:
        """Return the run name, deriving one from the architecture when unset.

        Returns:
            A filesystem-safe run name.
        """
        if self.run_name:
            return self.run_name
        params = self.model.estimated_parameters()
        return f"{self.model.name}-{params / 1e9:.1f}b-d{self.model.depth}"
