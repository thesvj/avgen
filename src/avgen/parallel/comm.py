"""Mesh-aware reductions.

Reducing a scalar across "all ranks" is wrong the moment more than one
parallelism axis is active, and it is wrong *silently*.

* Reducing a loss over the full world when TP is on divides by ``tp`` too many
  ranks — every TP rank computed the same loss for the same tokens, so they
  triple-count in a 3-way TP job.
* Reducing only over ``dp_shard`` when CP is on ignores the fact that CP ranks
  hold *different tokens of the same sample*, so the reported loss is one
  shard's loss, not the sample's.

The correct set is ``dp_cp``: every rank holding either different data or
different tokens. That mesh is registered by
:meth:`~avgen.parallel.dims.ParallelDims.build_mesh` precisely so this module
can ask for it by name.

Gradient norms have the same problem in a harder form. Under FSDP2 and TP the
parameters are :class:`~torch.distributed.tensor.DTensor` shards, and the true
global norm is not the norm of any local shard — it is the root of the summed
squares across shards. ``torch.nn.utils.clip_grad_norm_`` gets this right for
DTensor already, but only if you let it see DTensors; converting to local
tensors first, which is a tempting simplification, produces a norm that is too
small by roughly the square root of the shard count and therefore a clip that
never fires.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from avgen.parallel.dims import submesh

__all__ = [
    "all_gather_object",
    "all_reduce_max",
    "all_reduce_mean",
    "all_reduce_sum",
    "broadcast_object",
    "clip_grad_norm",
    "data_mesh",
    "gather_object",
]


def data_mesh(mesh: DeviceMesh | None) -> DeviceMesh | None:
    """Return the sub-mesh across which losses and metrics must be reduced.

    Args:
        mesh: The full device mesh, or ``None`` when running single-device.

    Returns:
        The ``dp_cp`` flattened mesh when present, else the single active data
        or context dimension, else ``None`` when no reduction is needed.
    """
    if mesh is None:
        return None
    for candidate in ("dp_cp", "dp_shard", "dp_replicate", "cp"):
        found = submesh(mesh, candidate)
        if found is not None:
            return found
    return None


def _reduce(
    value: torch.Tensor,
    mesh: DeviceMesh | None,
    op: dist.ReduceOp.RedOpType,
) -> torch.Tensor:
    if mesh is None or mesh.size() == 1:
        return value
    if not (dist.is_available() and dist.is_initialized()):
        return value
    reduced = value.detach().clone()
    dist.all_reduce(reduced, op=op, group=mesh.get_group())
    return reduced


def all_reduce_sum(value: torch.Tensor, mesh: DeviceMesh | None) -> torch.Tensor:
    """Sum a tensor across a mesh dimension.

    Args:
        value: Tensor to reduce. Not modified.
        mesh: Sub-mesh to reduce across, typically from :func:`data_mesh`.

    Returns:
        The summed tensor.
    """
    return _reduce(value, mesh, dist.ReduceOp.SUM)


def all_reduce_max(value: torch.Tensor, mesh: DeviceMesh | None) -> torch.Tensor:
    """Take the elementwise maximum across a mesh dimension.

    The right reduction for anything where the worst rank is what matters:
    peak memory, step latency, a straggler's dataloader wait.

    Args:
        value: Tensor to reduce.
        mesh: Sub-mesh to reduce across.

    Returns:
        The maximum tensor.
    """
    return _reduce(value, mesh, dist.ReduceOp.MAX)


def all_reduce_mean(value: torch.Tensor, mesh: DeviceMesh | None) -> torch.Tensor:
    """Average a tensor across a mesh dimension.

    Args:
        value: Tensor to reduce.
        mesh: Sub-mesh to reduce across.

    Returns:
        The averaged tensor.
    """
    if mesh is None or mesh.size() == 1:
        return value
    return all_reduce_sum(value, mesh) / mesh.size()


def clip_grad_norm(
    parameters: Iterable[torch.nn.Parameter],
    max_norm: float,
    *,
    pp_mesh: DeviceMesh | None = None,
    foreach: bool = True,
    error_if_nonfinite: bool = False,
) -> torch.Tensor:
    """Clip gradients to a global norm that is correct under any mesh.

    Data, tensor, and context parallelism are handled by ``clip_grad_norm_``
    itself, because gradients on those axes are DTensors that know their own
    placement. Pipeline parallelism is not: each stage holds a *disjoint* set of
    parameters as ordinary local tensors, so the per-stage norms must be
    combined explicitly. Skipping that step gives every stage a different, too
    small norm, and the clip effectively stops working just as the model gets
    deep enough to need it.

    Args:
        parameters: Parameters whose gradients to clip.
        max_norm: Maximum global 2-norm.
        pp_mesh: The pipeline mesh, when pipeline parallelism is active.
        foreach: Whether to use the fused multi-tensor path.
        error_if_nonfinite: Whether a non-finite norm should raise. Left false
            by default because the trainer prefers to skip the step and keep
            going: at 1000 GPUs, one bad microbatch should not end a run.

    Returns:
        The pre-clip global gradient norm as a scalar tensor.
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = torch.nn.utils.get_total_norm(
        grads, norm_type=2.0, error_if_nonfinite=error_if_nonfinite, foreach=foreach
    )

    if isinstance(total_norm, DTensor):
        # The norm of a sharded gradient is itself sharded; realise it before
        # it is used as a scalar multiplier.
        total_norm = total_norm.full_tensor()

    if pp_mesh is not None and pp_mesh.size() > 1:
        # Stages hold disjoint parameters, so combine sum-of-squares.
        squared = total_norm.pow(2)
        dist.all_reduce(squared, op=dist.ReduceOp.SUM, group=pp_mesh.get_group())
        total_norm = squared.sqrt()

    torch.nn.utils.clip_grads_with_norm_(
        [p for p in parameters if p.grad is not None],
        max_norm,
        total_norm,
        foreach=foreach,
    )
    return total_norm


def broadcast_object(obj: object, *, src: int = 0) -> object:
    """Broadcast a picklable object from one rank to all others.

    Used for agreeing on things that must be identical everywhere but are only
    known on one rank: a resolved checkpoint path, a run identifier, a data
    manifest hash.

    Args:
        obj: The object to send. Ignored on non-source ranks.
        src: Source global rank.

    Returns:
        The broadcast object.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return obj
    payload: list[object] = [obj if dist.get_rank() == src else None]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]


def all_gather_object(obj: object) -> list[object]:
    """Gather one picklable object per rank onto *every* rank.

    The counterpart to :func:`gather_object`, and the one an evaluation wants.
    ``gather_object`` returns ``None`` everywhere except the destination, so a
    caller that iterates the result crashes on every other rank — which is
    exactly what happened to the documented ``run_eval_suite(gather=...)``
    recipe. This returns a full list on all ranks, so the same code path works
    regardless of who is asking.

    Args:
        obj: This rank's object.

    Returns:
        One entry per rank, in rank order. A single-element list when not
        running distributed, so callers need no special case.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return [obj]
    output: list[object] = [None] * dist.get_world_size()
    dist.all_gather_object(output, obj)
    return output


def gather_object(obj: object, *, dst: int = 0) -> list[object] | None:
    """Gather one picklable object per rank onto a destination rank.

    Args:
        obj: This rank's object.
        dst: Destination global rank.

    Returns:
        The list of objects on ``dst``; ``None`` elsewhere.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return [obj]
    world_size = dist.get_world_size()
    output: list[object] | None = (
        [None] * world_size if dist.get_rank() == dst else None
    )
    dist.gather_object(obj, output, dst=dst)
    return output
