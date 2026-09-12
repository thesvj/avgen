"""The one function that turns a plain model into a parallel one.

**Order is not negotiable.** Each transformation assumes the previous one has
already happened, and applying them in a different order produces failures that
range from a clear exception to a silently wrong model:

1. **float8 conversion** — swaps ``nn.Linear`` for a float8 module. Must precede
   sharding, because sharding a module and then replacing it leaves DTensor
   shards pointing at a module that no longer exists.
2. **Tensor parallel** — converts parameters to DTensor with ``Shard``
   placements on the ``tp`` mesh. Must precede FSDP so FSDP composes its own
   sharding on top, yielding a 2-D sharded DTensor rather than fighting over
   the same parameter.
3. **Activation checkpointing** — wraps blocks. Must precede compile so the
   compiler sees the checkpoint boundaries and can fuse within them; wrapping a
   compiled module instead means recompiling the recomputation.
4. **Compile** — per block, not the whole model. A video DiT is ``depth``
   identical blocks, so one compile is reused ``depth`` times: compile time
   drops by roughly that factor, and a shape change in one bucket does not
   invalidate the whole graph.
5. **FSDP** — last. It must see the final module structure, because the set of
   modules it wraps is the set of communication boundaries.

Pipeline parallelism is applied by the caller before any of this, since it
changes which blocks exist on this rank at all.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from avgen.parallel.activation import (
    ActivationCheckpointConfig,
    apply_activation_checkpointing,
)
from avgen.parallel.dims import ParallelDims, submesh
from avgen.parallel.fsdp import FSDPConfig, apply_fsdp
from avgen.parallel.precision import PrecisionConfig, convert_to_float8
from avgen.parallel.tensor import apply_tensor_parallel

__all__ = ["ParallelConfig", "ParallelModel", "parallelize"]


@dataclass(frozen=True, slots=True)
class ParallelConfig:
    """Everything about how a model is distributed, in one place.

    Args:
        precision: Mixed-precision and float8 policy.
        activation_checkpoint: Activation-memory policy.
        fsdp: Sharding options.
        compile_blocks: Whether to ``torch.compile`` each transformer block.
        compile_mode: Inductor mode. ``default`` is the safe choice;
            ``max-autotune`` spends minutes searching for kernels and is worth
            it for a run measured in days, not for a smoke test.
        block_attribute: Name of the model's ``ModuleList`` of blocks.
        sequence_parallel: Whether tensor parallelism also shards the norm and
            residual regions along the sequence. Leave on — without it, tensor
            parallelism saves weight memory but not activation memory, and for
            video the activations are the problem.
    """

    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    activation_checkpoint: ActivationCheckpointConfig = field(
        default_factory=ActivationCheckpointConfig
    )
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    compile_blocks: bool = False
    compile_mode: str = "default"
    block_attribute: str = "blocks"
    sequence_parallel: bool = True


@dataclass(frozen=True, slots=True)
class ParallelModel:
    """A parallelised model plus the sub-meshes the trainer needs.

    Args:
        model: The transformed model.
        dims: The parallelism degrees in force.
        mesh: The full device mesh, or ``None`` on a single device.
        applied: Human-readable list of transformations that ran, for logs and
            for the simulator's report.
    """

    model: nn.Module
    dims: ParallelDims
    mesh: DeviceMesh | None
    applied: tuple[str, ...]

    def submesh(self, name: str) -> DeviceMesh | None:
        """Return a named sub-mesh, or ``None`` when that axis is inactive.

        Args:
            name: A mesh dimension name such as ``"cp"`` or ``"dp_cp"``.

        Returns:
            The sub-mesh, or ``None``.
        """
        return submesh(self.mesh, name)

    @property
    def cp_mesh(self) -> DeviceMesh | None:
        """The context-parallel sub-mesh."""
        return self.submesh("cp")

    @property
    def pp_mesh(self) -> DeviceMesh | None:
        """The pipeline sub-mesh."""
        return self.submesh("pp")


def _fsdp_target(dims: ParallelDims, mesh: DeviceMesh) -> DeviceMesh | None:
    """Choose the mesh FSDP2 shards parameters across.

    Two cases, and they are not the same shape:

    * **Plain FSDP** wants one flat dimension, and that dimension is
      ``dp_shard_cp``, not ``dp_shard``. Context-parallel ranks hold different
      *tokens* of the same sample, so there is no reason for them to also hold
      identical *parameter* copies. Folding cp into the sharding dimension is
      free memory, and forgetting to is the difference between a 1024-way and a
      256-way shard of the optimizer state.
    * **HSDP** wants the 2-D ``(dp_replicate, dp_shard_cp)`` view, because FSDP2
      all-gathers along the shard dimension and all-reduces along the replicate
      one. Both views are built once by
      :meth:`~avgen.parallel.dims.ParallelDims.build_mesh` and looked up here by
      name.

    Args:
        dims: Parallelism degrees.
        mesh: The full device mesh.

    Returns:
        The mesh to hand to ``fully_shard``, or ``None`` when nothing is
        sharded.
    """
    if dims.dp_replicate_enabled:
        hsdp = submesh(mesh, "hsdp")
        if hsdp is not None:
            return hsdp
    for candidate in ("dp_shard_cp", "dp_shard", "cp"):
        found = submesh(mesh, candidate)
        if found is not None:
            return found
    return None


def parallelize(
    model: nn.Module,
    dims: ParallelDims,
    *,
    mesh: DeviceMesh | None = None,
    config: ParallelConfig | None = None,
    adapt: Callable[[nn.Module], nn.Module] | None = None,
) -> ParallelModel:
    """Apply every configured parallelism transformation, in the correct order.

    Args:
        model: The model to transform, modified in place.
        dims: Parallelism degrees.
        mesh: A pre-built mesh, or ``None`` to build one from ``dims``. Passing
            one lets the simulator supply a fake mesh.
        config: How to apply each transformation.
        adapt: Optional hook invoked after tensor parallelism and before
            activation checkpointing, compile and FSDP. This is the only correct
            seam for injecting LoRA or a control adapter: the base weights are
            already DTensors on the tensor-parallel mesh, so an adapter can
            derive its own placements from them, and FSDP has not yet run, so the
            new parameters are still picked up and gradient-reduced. Injecting
            after FSDP leaves the adapter unmanaged and every rank silently
            learns a different one.

    Returns:
        The transformed model together with the meshes the trainer will need.
    """
    settings = config or ParallelConfig()
    applied: list[str] = []

    if mesh is None and dims.world_size > 1:
        mesh = dims.build_mesh()

    converted = convert_to_float8(model, settings.precision)
    if converted:
        applied.append(f"float8({converted} linears)")

    if dims.tp_enabled and mesh is not None:
        apply_tensor_parallel(
            model,
            mesh["tp"],
            sequence_parallel=settings.sequence_parallel,
            block_attribute=settings.block_attribute,
        )
        applied.append(
            f"tensor_parallel(tp={dims.tensor}, "
            f"sequence_parallel={settings.sequence_parallel})"
        )

    if adapt is not None:
        model = adapt(model)
        applied.append("adapt")

    if settings.activation_checkpoint.enabled:
        apply_activation_checkpointing(
            model,
            settings.activation_checkpoint,
            block_attribute=settings.block_attribute,
        )
        applied.append(f"activation_checkpoint({settings.activation_checkpoint.mode})")

    if settings.compile_blocks:
        blocks = getattr(model, settings.block_attribute, None)
        if isinstance(blocks, nn.ModuleList):
            for index, block in enumerate(blocks):
                blocks[index] = torch.compile(block, mode=settings.compile_mode)
            applied.append(f"compile({len(blocks)} blocks, {settings.compile_mode})")

    # The gate is "is there a mesh to shard and reduce over", not "is dp_shard
    # enabled": a cp-only job has no data-parallel dimension but still needs its
    # partial gradients summed. See _fsdp_target.
    if mesh is not None and (dims.dp_shard_enabled or dims.cp_enabled):
        target = _fsdp_target(dims, mesh)
        if target is not None:
            apply_fsdp(
                model,
                target,
                precision=settings.precision,
                config=settings.fsdp,
            )
            if dims.dp_replicate_enabled:
                label = f"hsdp(dp_replicate={dims.dp_replicate}, "
                label += f"dp_shard={dims.dp_shard})"
            elif dims.dp_shard_enabled:
                label = f"fsdp(dp_shard={dims.dp_shard}"
                label += f", cp={dims.context})" if dims.cp_enabled else ")"
            else:
                label = f"fsdp(cp={dims.context})"
            applied.append(label)
    elif dims.dp_replicate_enabled and mesh is not None:
        # Replication without sharding: DDP is the right tool, and it is
        # cheaper than a degenerate FSDP mesh of size one.
        from torch.nn.parallel import DistributedDataParallel

        model = DistributedDataParallel(
            model,
            device_mesh=mesh["dp_replicate"],
            gradient_as_bucket_view=True,
        )
        applied.append(f"ddp(dp_replicate={dims.dp_replicate})")

    if not applied:
        applied.append("single-device")

    return ParallelModel(model=model, dims=dims, mesh=mesh, applied=tuple(applied))
