"""Predict per-rank memory before spending a cluster-hour finding out.

Two independent estimators, deliberately kept separate:

:func:`estimate_memory`
    A closed-form model. Runs in microseconds, needs no GPU and no model
    instance — only shapes and degrees. Fast enough to sweep thousands of
    configurations and find the ones worth measuring.

:func:`measure_memory`
    Instruments an actual forward and backward with PyTorch's ``MemTracker`` /
    ``FSDPMemTracker`` and reports what really happened, module by module. Needs
    a GPU (or fake tensors), and is the ground truth the analytical model is
    calibrated against.

The analytical model exists because the measurement is too slow to sweep and
the sweep is too coarse to trust. Use the first to shortlist, the second to
confirm, and :func:`calibration_error` to keep the first honest.

**The accounting that people get wrong.** Optimizer state for AdamW is not "2x
parameters". In mixed precision it is an fp32 master copy plus two fp32 moments
— 12 bytes per parameter, not 8 — and it dominates parameter memory by 3x. On a
2B model that is 24 GB before a single activation exists, which is why FSDP
sharding of *optimizer state*, not just weights, is what makes large training
possible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from avgen.parallel.activation import ActivationCheckpointConfig
from avgen.parallel.dims import ParallelDims

__all__ = [
    "MemoryEstimate",
    "ModelShape",
    "calibration_error",
    "estimate_memory",
    "measure_memory",
]

_GIB = 1024.0**3

#: Bytes of optimizer state per parameter, by optimizer family, assuming an
#: fp32 master weight is kept alongside the moments.
OPTIMIZER_BYTES: dict[str, int] = {
    "sgd": 4,  # fp32 master only
    "sgd_momentum": 8,  # master + momentum
    "adamw": 12,  # master + exp_avg + exp_avg_sq
    "adamw_bf16_state": 8,  # master fp32 + bf16 moments
    "adafactor": 5,  # master + factored second moment
    "muon": 8,  # master + momentum
}


@dataclass(frozen=True, slots=True)
class ModelShape:
    """The handful of numbers that determine a transformer's memory footprint.

    Args:
        parameters: Total parameter count, unsharded.
        depth: Number of transformer blocks.
        width: Hidden width.
        sequence_length: Tokens per sample, before context-parallel sharding.
        micro_batch_size: Samples per rank per microbatch.
        mlp_ratio: Feed-forward expansion factor.
        num_heads: Attention heads.
        text_tokens: Cross-attention context length.
    """

    parameters: int
    depth: int
    width: int
    sequence_length: int
    micro_batch_size: int = 1
    mlp_ratio: int = 4
    num_heads: int = 16
    text_tokens: int = 0


@dataclass(frozen=True, slots=True)
class MemoryEstimate:
    """Predicted per-rank memory, broken down by what consumes it.

    Args:
        parameter_bytes: Sharded weights in compute precision.
        gradient_bytes: Sharded gradients.
        optimizer_bytes: Sharded optimizer state including the master copy.
        activation_bytes: Peak stored activations after the checkpointing policy.
        gather_bytes: Transient peak from FSDP all-gathering unsharded
            parameters for the currently executing blocks.
        workspace_bytes: Allocator overhead, fragmentation, and kernel
            workspaces.
    """

    parameter_bytes: float
    gradient_bytes: float
    optimizer_bytes: float
    activation_bytes: float
    gather_bytes: float
    workspace_bytes: float

    @property
    def total_bytes(self) -> float:
        """Predicted peak allocation on one rank."""
        return (
            self.parameter_bytes
            + self.gradient_bytes
            + self.optimizer_bytes
            + self.activation_bytes
            + self.gather_bytes
            + self.workspace_bytes
        )

    @property
    def total_gib(self) -> float:
        """Predicted peak in gibibytes."""
        return self.total_bytes / _GIB

    def fits_in(self, device_gib: float, *, headroom: float = 0.10) -> bool:
        """Whether this configuration fits, keeping a safety margin.

        The margin is not superstition. Allocator fragmentation, a cuDNN
        workspace, and the NCCL buffers all sit outside this accounting, and a
        job that peaks at 99% of device memory will OOM on the one step whose
        bucket is slightly larger.

        Args:
            device_gib: Device memory in gibibytes.
            headroom: Fraction to keep free.

        Returns:
            Whether the estimate fits within the budget.
        """
        return self.total_gib <= device_gib * (1.0 - headroom)

    def breakdown_gib(self) -> dict[str, float]:
        """Return every component in gibibytes, largest first."""
        items = {key: value / _GIB for key, value in asdict(self).items()}
        return dict(sorted(items.items(), key=lambda kv: kv[1], reverse=True))

    def dominant_term(self) -> str:
        """Return the component to attack first when this does not fit."""
        return max(asdict(self).items(), key=lambda kv: kv[1])[0]


def _activation_bytes_per_block(
    shape: ModelShape,
    *,
    local_sequence: int,
    dtype_bytes: int,
) -> float:
    """Stored activation bytes for one unchecked transformer block.

    Counts the tensors a backward pass needs: the block input, the normalised
    activations, the q/k/v projections, the attention output, and the two
    feed-forward intermediates. Attention scores are excluded because every
    kernel avgen uses (flash, memory-efficient, cuDNN) recomputes them in
    backward rather than storing an ``O(seq^2)`` matrix — that single property
    is what makes long-sequence video training possible at all.
    """
    tokens = shape.micro_batch_size * local_sequence
    hidden = tokens * shape.width * dtype_bytes
    # input + attn_norm_out + q + k + v + attn_out + ffn_norm_out
    attention = hidden * 7.0
    # gate and up projections at the expanded width, plus the SiLU product
    feed_forward = tokens * shape.width * shape.mlp_ratio * dtype_bytes * 3.0
    cross_attention = 0.0
    if shape.text_tokens:
        # q at sequence length, k/v at text length
        cross_attention = hidden + 2.0 * (
            shape.micro_batch_size * shape.text_tokens * shape.width * dtype_bytes
        )
    return attention + feed_forward + cross_attention


def estimate_memory(
    shape: ModelShape,
    dims: ParallelDims,
    *,
    activation_checkpoint: ActivationCheckpointConfig | None = None,
    param_dtype_bytes: int = 2,
    optimizer: str = "adamw",
    fsdp_prefetch_depth: int = 1,
    workspace_gib: float = 2.0,
) -> MemoryEstimate:
    """Predict per-rank peak memory in closed form.

    Args:
        shape: Model and batch geometry.
        dims: Parallelism degrees.
        activation_checkpoint: The checkpointing policy. ``None`` means store
            everything.
        param_dtype_bytes: Bytes per parameter in compute precision.
        optimizer: Key into :data:`OPTIMIZER_BYTES`.
        fsdp_prefetch_depth: Blocks whose parameters are gathered at once.
        workspace_gib: Fixed allowance for allocator overhead and kernel
            workspaces.

    Returns:
        The estimate.

    Raises:
        KeyError: If the optimizer is unknown.
    """
    if optimizer not in OPTIMIZER_BYTES:
        raise KeyError(
            f"unknown optimizer {optimizer!r}; "
            f"known: {', '.join(sorted(OPTIMIZER_BYTES))}"
        )
    policy = activation_checkpoint or ActivationCheckpointConfig(mode="none")

    # Parameters shard across FSDP, tensor parallel, and pipeline depth. Context
    # parallelism shards parameters too, because avgen flattens dp_shard with cp
    # for the FSDP mesh.
    param_shards = dims.dp_shard * dims.context * dims.tensor * dims.pipeline
    local_params = shape.parameters / max(1, param_shards)

    parameter_bytes = local_params * param_dtype_bytes
    gradient_bytes = local_params * param_dtype_bytes
    optimizer_bytes = local_params * OPTIMIZER_BYTES[optimizer]

    # The sequence is split by context parallelism, and again by tensor
    # parallelism wherever sequence parallelism applies.
    local_sequence = shape.sequence_length / max(1, dims.sequence_shard_size)
    per_block = _activation_bytes_per_block(
        shape, local_sequence=local_sequence, dtype_bytes=param_dtype_bytes
    )

    local_depth = shape.depth / max(1, dims.pipeline)
    if policy.mode == "none":
        stored_blocks = local_depth
    elif policy.mode == "full":
        # Only the block boundaries survive, plus one block live at a time.
        stored_blocks = 1.0
    elif policy.mode == "selective_layer":
        checkpointed = local_depth / policy.layer_interval
        stored_blocks = (local_depth - checkpointed) + 1.0
    else:
        # Selective-op keeps roughly the matmul and attention outputs: about a
        # third of a full block's stored tensors in a standard pre-norm block,
        # scaled by how often those outputs are actually saved.
        retained = 0.35 / policy.save_op_frequency
        stored_blocks = local_depth * retained + 1.0

    activation_bytes = per_block * stored_blocks
    if policy.mode != "none":
        # Every checkpointed block still stores its input for recomputation.
        boundary = (
            shape.micro_batch_size * local_sequence * shape.width * param_dtype_bytes
        )
        activation_bytes += boundary * local_depth

    # FSDP transiently holds unsharded parameters for the blocks in flight.
    gather_bytes = 0.0
    if dims.dp_shard_enabled and shape.depth:
        params_per_block = shape.parameters / shape.depth
        gather_bytes = (
            params_per_block
            * param_dtype_bytes
            * (fsdp_prefetch_depth + 1)
            / max(1, dims.tensor * dims.pipeline)
        )

    return MemoryEstimate(
        parameter_bytes=parameter_bytes,
        gradient_bytes=gradient_bytes,
        optimizer_bytes=optimizer_bytes,
        activation_bytes=activation_bytes,
        gather_bytes=gather_bytes,
        workspace_bytes=workspace_gib * _GIB,
    )


def measure_memory(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: Any,
    *,
    is_fsdp: bool = False,
    display: bool = False,
) -> dict[str, float]:
    """Measure real per-module memory across one training step.

    Args:
        model: The model, already parallelised as it will be in production.
        optimizer: The optimizer, so its state is attributed correctly.
        step: A zero-argument callable running one forward and backward.
        is_fsdp: Whether the model has been sharded with FSDP2, which selects
            the tracker that understands all-gather transients.
        display: Whether to print the module-wise breakdown.

    Returns:
        Peak bytes by category.
    """
    if is_fsdp:
        from torch.distributed._tools.fsdp2_mem_tracker import FSDPMemTracker

        tracker: Any = FSDPMemTracker(model, optimizer)
    else:
        from torch.distributed._tools.mem_tracker import MemTracker

        tracker = MemTracker()
        tracker.track_external(model, optimizer)

    with tracker:
        step()
        if is_fsdp:
            tracker.reset_mod_stats()
            step()

    if display:
        tracker.display_modulewise_snapshots(depth=3, units="GiB")

    snapshot = tracker.get_tracker_snapshot("peak")
    return {
        str(device): float(stats.get("Total", 0.0))
        for device, stats in snapshot.items()
    }


def calibration_error(
    predicted: MemoryEstimate,
    measured_bytes: float,
) -> dict[str, float]:
    """Compare a prediction against a measurement.

    Run this whenever you change the analytical model, and on any new
    architecture. A relative error above roughly 15% means the closed-form model
    has stopped describing your architecture and its sweeps should not be
    trusted until the term that drifted is fixed.

    Args:
        predicted: Output of :func:`estimate_memory`.
        measured_bytes: Peak bytes from :func:`measure_memory`.

    Returns:
        Absolute and relative error, and the component to look at first.
    """
    absolute = predicted.total_bytes - measured_bytes
    relative = absolute / measured_bytes if measured_bytes else float("inf")
    return {
        "predicted_gib": predicted.total_gib,
        "measured_gib": measured_bytes / _GIB,
        "absolute_error_gib": absolute / _GIB,
        "relative_error": relative,
        "dominant_term": predicted.dominant_term(),  # type: ignore[dict-item]
    }
