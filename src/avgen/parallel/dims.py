"""Parallelism dimensions and the device mesh they build.

A thousand-GPU video training job is a five-dimensional problem, and the single
most useful thing a framework can do is make those five dimensions *explicit,
named, and validated in one place* instead of scattering ``world_size //
something`` arithmetic through the codebase.

The five axes, and what each one buys:

======================  =========================================================
Axis                    What it solves
======================  =========================================================
``dp_replicate``        Throughput. Plain replication; gradients all-reduced.
``dp_shard``            Parameter/optimizer memory. FSDP2 shards state across it.
``context`` (CP)        **Sequence length.** The axis that matters most for
                        video: a 10-second 720p clip is 100k+ tokens, attention
                        is quadratic in that, and activation memory is linear.
                        No other axis reduces per-rank sequence length.
``tensor`` (TP)         Per-layer weight and activation memory; needs the
                        highest bandwidth, so it is placed innermost (intra-node).
``pipeline`` (PP)       Model depth beyond what one node can hold. Last resort:
                        bubbles and stage imbalance cost real throughput.
======================  =========================================================

**Ordering is a performance decision, not a formality.** The mesh is built as
``(pp, dp_replicate, dp_shard, cp, tp)``. Rank ordering makes the *last* dim
vary fastest, so TP ranks are adjacent — landing inside one NVLink domain — and
PP ranks are furthest apart, which is correct because pipeline traffic is a
small point-to-point activation handoff while TP traffic is an all-reduce on
every layer. Getting this order wrong is worth a large fraction of your
throughput and produces no error message.

**Sharding vs replication for data.** The data-parallel *product*
``dp_replicate * dp_shard`` is how many distinct sample groups exist. CP and TP
ranks hold shards of the *same* samples, so they must receive identical data and
identical noise; see :meth:`ParallelDims.data_coordinates`.
"""

from __future__ import annotations

import os
import warnings
import weakref
from dataclasses import dataclass

import torch
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

__all__ = ["ParallelDims", "submesh"]

# Flattened meshes (``dp_shard_cp``, ``dp_cp``) are derived views, not real
# dimensions, so PyTorch does not list them in ``mesh_dim_names`` and slicing
# them back off the root mesh is deprecated from PT 2.11 — the guidance is to
# bookkeep them yourself. This registry is that bookkeeping: it is keyed weakly
# on the root mesh, so it holds nothing alive and needs no cleanup.
_FLATTENED: weakref.WeakKeyDictionary[DeviceMesh, dict[str, DeviceMesh]] = (
    weakref.WeakKeyDictionary()
)


def submesh(mesh: DeviceMesh | None, name: str) -> DeviceMesh | None:
    """Return a named sub-mesh, including flattened views, or ``None``.

    Looks in the flattened-mesh registry first, then at the mesh's real
    dimensions. Returns ``None`` rather than raising when the axis is inactive,
    because "this parallelism dimension is not enabled" is the normal case on a
    single GPU and every call site would otherwise need a guard.

    Args:
        mesh: The root device mesh, or ``None`` when running single-device.
        name: A dimension name such as ``"cp"``, or a flattened view such as
            ``"dp_shard_cp"``.

    Returns:
        The sub-mesh, or ``None`` when it does not exist.
    """
    if mesh is None:
        return None
    flattened = _FLATTENED.get(mesh)
    if flattened is not None and name in flattened:
        return flattened[name]
    if name in tuple(mesh.mesh_dim_names or ()):
        return mesh[name]
    return None


#: Mesh dimension names, outermost first. Kept as a module constant because the
#: parallel plans, the checkpoint layer, and the simulator all index the mesh by
#: these exact names.
MESH_DIM_ORDER: tuple[str, ...] = ("pp", "dp_replicate", "dp_shard", "cp", "tp")


@dataclass(frozen=True, slots=True)
class ParallelDims:
    """Validated parallelism degrees plus the mesh they induce.

    Args:
        world_size: Total number of ranks in the job.
        dp_replicate: Replicated data-parallel degree. Greater than one enables
            HSDP: shard within a group, replicate across groups, which keeps the
            expensive all-gather inside a node and the cheap all-reduce between
            nodes.
        dp_shard: Sharded data-parallel (FSDP2) degree. ``-1`` means "use
            whatever is left over", which is almost always what you want.
        tensor: Tensor-parallel degree. Should not exceed the number of GPUs
            per node.
        context: Context-parallel degree, sharding the token sequence.
        pipeline: Pipeline-parallel degree, sharding model depth.
        enable_loss_parallel: Whether the loss is computed on sharded
            activations rather than gathering them first.

    Raises:
        ValueError: If any degree is non-positive, if ``dp_shard`` cannot be
            inferred cleanly, or if the product of degrees does not equal
            ``world_size``.
    """

    world_size: int
    dp_replicate: int = 1
    dp_shard: int = -1
    tensor: int = 1
    context: int = 1
    pipeline: int = 1
    enable_loss_parallel: bool = True

    def __post_init__(self) -> None:
        """Infer ``dp_shard`` when requested and validate the factorisation."""
        if isinstance(self.world_size, bool) or self.world_size < 1:
            raise ValueError(f"world_size must be positive; got {self.world_size!r}")
        for name in ("dp_replicate", "tensor", "context", "pipeline"):
            value = getattr(self, name)
            if isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be >= 1; got {value!r}")
        if isinstance(self.dp_shard, bool) or (
            self.dp_shard < 1 and self.dp_shard != -1
        ):
            raise ValueError(f"dp_shard must be >= 1 or -1; got {self.dp_shard!r}")

        others = self.dp_replicate * self.tensor * self.context * self.pipeline
        if self.dp_shard == -1:
            if self.world_size % others != 0:
                raise ValueError(
                    f"cannot infer dp_shard: world_size={self.world_size} is not "
                    f"divisible by dp_replicate*tensor*context*pipeline={others}"
                )
            object.__setattr__(self, "dp_shard", self.world_size // others)

        product = others * self.dp_shard
        if product != self.world_size:
            raise ValueError(
                "parallelism degrees must multiply to world_size; "
                f"dp_replicate={self.dp_replicate} * dp_shard={self.dp_shard} * "
                f"tensor={self.tensor} * context={self.context} * "
                f"pipeline={self.pipeline} = {product} != {self.world_size}"
            )

    # ------------------------------------------------------------------
    # Enablement flags. Plans consult these instead of comparing to 1, so the
    # meaning of "enabled" lives in exactly one place.
    # ------------------------------------------------------------------

    @property
    def dp_replicate_enabled(self) -> bool:
        """Whether replicated data parallelism is active."""
        return self.dp_replicate > 1

    @property
    def dp_shard_enabled(self) -> bool:
        """Whether sharded data parallelism (FSDP2) is active."""
        return self.dp_shard > 1

    @property
    def dp_enabled(self) -> bool:
        """Whether any data parallelism is active."""
        return self.dp_replicate_enabled or self.dp_shard_enabled

    @property
    def tp_enabled(self) -> bool:
        """Whether tensor parallelism is active."""
        return self.tensor > 1

    @property
    def cp_enabled(self) -> bool:
        """Whether context parallelism is active."""
        return self.context > 1

    @property
    def pp_enabled(self) -> bool:
        """Whether pipeline parallelism is active."""
        return self.pipeline > 1

    @property
    def loss_parallel_enabled(self) -> bool:
        """Whether the loss should be computed on sharded activations."""
        return self.tp_enabled and self.enable_loss_parallel

    # ------------------------------------------------------------------
    # Derived sizes.
    # ------------------------------------------------------------------

    @property
    def dp_size(self) -> int:
        """Number of distinct sample groups.

        This is the number the global batch size must be divisible by, and the
        number of independent data shards the loader must produce.
        """
        return self.dp_replicate * self.dp_shard

    @property
    def sequence_shard_size(self) -> int:
        """Factor by which the token sequence is split across ranks.

        Context parallelism splits the sequence directly. Tensor parallelism
        splits it too whenever sequence parallelism is applied to the norm and
        residual regions, which the shipped plans always do — that is where the
        activation-memory win of TP actually comes from.
        """
        return self.context * self.tensor

    @property
    def model_shard_size(self) -> int:
        """Number of ranks a single model replica is spread across."""
        return self.dp_shard * self.context * self.tensor * self.pipeline

    def gradient_accumulation_for(
        self,
        *,
        global_batch_size: int,
        local_batch_size: int,
    ) -> int:
        """Return the accumulation steps that realise a global batch size.

        Args:
            global_batch_size: Samples per optimizer step across the whole job.
            local_batch_size: Samples per data-parallel rank per microbatch.

        Returns:
            Number of microbatches per optimizer step.

        Raises:
            ValueError: If the global batch does not factor cleanly. This is
                deliberately fatal: silently rounding it changes the effective
                learning rate and makes two runs incomparable.
        """
        denominator = self.dp_size * local_batch_size
        if global_batch_size % denominator != 0:
            raise ValueError(
                f"global_batch_size={global_batch_size} must be divisible by "
                f"dp_size*local_batch_size={denominator}; adjust one of them "
                "rather than letting the effective batch size drift"
            )
        return global_batch_size // denominator

    # ------------------------------------------------------------------
    # Mesh construction.
    # ------------------------------------------------------------------

    @property
    def _active_dims(self) -> tuple[tuple[str, int], ...]:
        """Named degrees greater than one, in mesh order.

        Recomputed rather than cached: the class is ``slots=True`` (so there is
        no ``__dict__`` for ``cached_property`` to write into) and the
        computation is five dictionary lookups.
        """
        degrees = {
            "pp": self.pipeline,
            "dp_replicate": self.dp_replicate,
            "dp_shard": self.dp_shard,
            "cp": self.context,
            "tp": self.tensor,
        }
        return tuple(
            (name, degrees[name]) for name in MESH_DIM_ORDER if degrees[name] > 1
        )

    def build_mesh(self, device_type: str | None = None) -> DeviceMesh:
        """Build the device mesh, including the flattened helper meshes.

        Only dimensions greater than one appear in the mesh: a degenerate
        dimension of size one costs a process group and buys nothing, and its
        absence is what lets the same plan code run unchanged on one GPU.

        Two flattened views are registered because they are needed constantly:

        * ``dp_shard_cp`` — FSDP2 shards parameters across *both* the sharded
          data dimension and the context dimension. CP ranks hold different
          tokens of the same sample, so they can also hold different parameter
          shards; not flattening here would leave the CP dimension's memory
          saving on the table.
        * ``dp_cp`` — the set of ranks holding different data or different
          tokens. Loss and metric reductions must span exactly this set:
          reducing over TP too would double-count, and reducing over less would
          report one shard's loss as the job's.

        Args:
            device_type: ``"cuda"``, ``"cpu"``, or ``None`` to auto-detect.

        Returns:
            The named device mesh.

        Raises:
            RuntimeError: If no dimension is greater than one, which means a
                mesh is not needed at all.
        """
        resolved = device_type or ("cuda" if torch.cuda.is_available() else "cpu")
        active = self._active_dims
        if not active:
            raise RuntimeError(
                "no parallelism dimension exceeds 1; run single-device without a mesh"
            )
        names = tuple(name for name, _ in active)
        shape = tuple(size for _, size in active)
        mesh = init_device_mesh(resolved, shape, mesh_dim_names=names)

        flattened: dict[str, DeviceMesh] = {}
        shard_group = [name for name in ("dp_shard", "cp") if name in names]
        if len(shard_group) > 1:
            flattened["dp_shard_cp"] = mesh[tuple(shard_group)]._flatten(
                mesh_dim_name="dp_shard_cp"
            )

        data_group = [
            name for name in ("dp_replicate", "dp_shard", "cp") if name in names
        ]
        if len(data_group) > 1:
            flattened["dp_cp"] = mesh[tuple(data_group)]._flatten(mesh_dim_name="dp_cp")
        # HSDP needs a 2-D (replicate, shard) mesh: FSDP2 all-gathers along the
        # shard dimension and all-reduces along the replicate one, so the two
        # must stay distinct. Building it means slicing a flattened dimension
        # off the root, which PyTorch deprecates in favour of bookkeeping the
        # result yourself — which is exactly what the registry below is. Doing
        # it once here, rather than on every lookup, keeps that to a single
        # suppressed warning at startup instead of one per call.
        if "dp_replicate" in names and "dp_shard_cp" in flattened:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*flattened dim.*")
                flattened["hsdp"] = mesh[("dp_replicate", "dp_shard_cp")]
        elif "dp_replicate" in names and "dp_shard" in names:
            flattened["hsdp"] = mesh[("dp_replicate", "dp_shard")]

        if flattened:
            _FLATTENED[mesh] = flattened
        return mesh

    def data_coordinates(self, mesh: DeviceMesh) -> tuple[int, int]:
        """Return this rank's ``(data_rank, data_world)`` for sharding.

        Every rank sharing a ``data_rank`` must receive **byte-identical**
        batches and draw **identical** noise, because they hold shards of the
        same samples. Ranks with different ``data_rank`` must receive disjoint
        data. Getting this wrong is the single most common correctness bug in a
        multi-dimensional parallel trainer, and it is invisible: the loss curve
        looks fine while the effective batch size is a fraction of what you
        think it is.

        Args:
            mesh: The mesh returned by :meth:`build_mesh`.

        Returns:
            This rank's index within the data-parallel product, and the size of
            that product.
        """
        names = tuple(mesh.mesh_dim_names or ())
        rank = 0
        world = 1
        # Iterate outermost-to-innermost so the composed index is stable and
        # matches the mesh's own rank ordering.
        for name in ("dp_replicate", "dp_shard"):
            if name in names:
                dim_mesh = mesh[name]
                rank = rank * dim_mesh.size() + dim_mesh.get_local_rank()
                world *= dim_mesh.size()
        return rank, world

    def sequence_coordinates(self, mesh: DeviceMesh) -> tuple[int, int]:
        """Return this rank's ``(shard_index, shard_count)`` along the sequence.

        Args:
            mesh: The mesh returned by :meth:`build_mesh`.

        Returns:
            Position within the context-parallel group and its size.
        """
        names = tuple(mesh.mesh_dim_names or ())
        if "cp" not in names:
            return 0, 1
        cp_mesh = mesh["cp"]
        return cp_mesh.get_local_rank(), cp_mesh.size()

    def describe(self) -> str:
        """Return a one-line human summary for logs and error messages."""
        parts = [f"world={self.world_size}"]
        for name, size in (
            ("dp_replicate", self.dp_replicate),
            ("dp_shard", self.dp_shard),
            ("cp", self.context),
            ("tp", self.tensor),
            ("pp", self.pipeline),
        ):
            if size > 1:
                parts.append(f"{name}={size}")
        if len(parts) == 1:
            parts.append("single-device")
        return " ".join(parts)

    @classmethod
    def from_env(
        cls,
        *,
        dp_replicate: int = 1,
        dp_shard: int = -1,
        tensor: int = 1,
        context: int = 1,
        pipeline: int = 1,
        enable_loss_parallel: bool = True,
    ) -> ParallelDims:
        """Build dims using ``WORLD_SIZE`` from the launcher environment.

        Args:
            dp_replicate: Replicated data-parallel degree.
            dp_shard: Sharded data-parallel degree, or ``-1`` to infer.
            tensor: Tensor-parallel degree.
            context: Context-parallel degree.
            pipeline: Pipeline-parallel degree.
            enable_loss_parallel: Whether to compute loss on sharded activations.

        Returns:
            The validated dims.
        """
        return cls(
            world_size=int(os.environ.get("WORLD_SIZE", "1")),
            dp_replicate=dp_replicate,
            dp_shard=dp_shard,
            tensor=tensor,
            context=context,
            pipeline=pipeline,
            enable_loss_parallel=enable_loss_parallel,
        )
