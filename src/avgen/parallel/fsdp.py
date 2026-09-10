"""FSDP2 sharding.

FSDP2 (``fully_shard``) replaces parameters with :class:`~torch.distributed.
tensor.DTensor` shards in place rather than wrapping the module. Everything
downstream benefits: state-dict keys keep their original names, tensor
parallelism composes on the same parameters, and ``torch.compile`` sees the real
module rather than a wrapper.

Two decisions in this module carry most of the performance:

**Wrap per block, not once at the root.** The unit of ``fully_shard`` is the
unit of communication: each wrapped module all-gathers its parameters just
before its forward and frees them just after. Wrapping only the root means one
enormous all-gather that cannot overlap with anything and a peak memory equal to
the whole model. Wrapping per block gives ``depth`` small all-gathers that
prefetch under the previous block's compute — the entire point of FSDP.

**Do not reshard the last block after forward.** Its parameters are needed again
almost immediately by the first backward step, so freeing and re-gathering them
is a round trip for nothing. It is a one-line change worth a measurable
percentage of step time on a deep model.

For multi-node jobs, ``dp_replicate > 1`` gives HSDP: shard inside a replica
group, replicate across groups. The all-gather then stays on NVLink and only a
reduce-scatter crosses the slower inter-node fabric, which is usually the right
trade above a few hundred GPUs.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, OffloadPolicy, fully_shard

from avgen.parallel.precision import PrecisionConfig

__all__ = ["FSDPConfig", "apply_fsdp"]


@dataclass(frozen=True, slots=True)
class FSDPConfig:
    """How FSDP2 shards a model.

    Args:
        reshard_after_forward: Whether to free gathered parameters after the
            forward pass. True is ZeRO-3 behaviour and the memory-optimal
            choice. False keeps parameters gathered, which is ZeRO-2 behaviour:
            faster, but the model must fit unsharded on one rank.
        cpu_offload: Whether to keep sharded parameters in host memory. A last
            resort — it moves the bottleneck to PCIe — but it is the difference
            between "cannot run" and "runs slowly", and sometimes that matters.
        block_attribute: Name of the ``ModuleList`` of transformer blocks.
        shard_last_block_after_forward: Whether the final block reshards after
            forward. Left false because backward needs it immediately.
        ignore_frozen_params: Whether parameters with ``requires_grad=False``
            are excluded from sharding. During adapter fine-tuning the frozen
            base is the overwhelming majority of parameters and sharding it
            still pays the all-gather cost every step for weights that never
            change; excluding them is a large win for LoRA-style training.
    """

    reshard_after_forward: bool = True
    cpu_offload: bool = False
    block_attribute: str = "blocks"
    shard_last_block_after_forward: bool = False
    ignore_frozen_params: bool = False


def _offload_policy(config: FSDPConfig) -> OffloadPolicy:
    return CPUOffloadPolicy(pin_memory=True) if config.cpu_offload else OffloadPolicy()


def apply_fsdp(
    model: nn.Module,
    mesh: DeviceMesh,
    *,
    precision: PrecisionConfig,
    config: FSDPConfig | None = None,
) -> nn.Module:
    """Shard a model's blocks and root across a data-parallel mesh.

    Args:
        model: The model to shard, modified in place.
        mesh: The mesh to shard across. Pass the ``dp_shard_cp`` flattened mesh
            when context parallelism is active, so parameter shards also spread
            across the context dimension.
        precision: Mixed-precision policy.
        config: Sharding options. Pass the 2-D ``(dp_replicate, dp_shard_cp)``
            mesh for HSDP; ``fully_shard`` reads the two dimensions itself.

    Returns:
        The same model, sharded in place.

    Raises:
        AttributeError: If the model exposes no block list under
            ``config.block_attribute``.
    """
    settings = config or FSDPConfig()
    blocks = getattr(model, settings.block_attribute, None)
    if not isinstance(blocks, nn.ModuleList):
        raise AttributeError(
            f"model {type(model).__name__} has no nn.ModuleList attribute "
            f"{settings.block_attribute!r}; FSDP needs per-block wrapping to "
            "overlap communication with compute"
        )

    ignored: set[nn.Parameter] | None = None
    if settings.ignore_frozen_params:
        frozen = {p for p in model.parameters() if not p.requires_grad}
        ignored = frozen or None

    shared: dict[str, object] = {
        "mesh": mesh,
        "mp_policy": precision.fsdp_policy(),
        "offload_policy": _offload_policy(settings),
    }
    if ignored is not None:
        shared["ignored_params"] = ignored

    last = len(blocks) - 1
    for index, block in enumerate(blocks):
        reshard = settings.reshard_after_forward
        if index == last and not settings.shard_last_block_after_forward:
            reshard = False
        fully_shard(block, reshard_after_forward=reshard, **shared)  # type: ignore[arg-type]

    # The root gathers whatever is left: embeddings, projections, norms. It must
    # be wrapped last so FSDP2 sees the already-sharded blocks as leaves.
    fully_shard(
        model,
        reshard_after_forward=settings.reshard_after_forward,
        **shared,  # type: ignore[arg-type]
    )
    return model


def set_prefetch_depth(
    model: nn.Module, *, forward: int = 1, backward: int = 1
) -> None:
    """Tune how many blocks ahead FSDP2 prefetches parameters.

    Deeper prefetch hides more latency and holds more parameters resident at
    once. One is the safe default. Raising it helps on a slow fabric where a
    single block's compute is not long enough to cover an all-gather; it hurts
    when memory is already tight, because ``depth`` blocks of unsharded
    parameters are live simultaneously.

    Args:
        model: A model already passed through :func:`apply_fsdp`.
        forward: Blocks to prefetch ahead during forward.
        backward: Blocks to prefetch ahead during backward.

    Raises:
        ValueError: If either depth is below one.
    """
    if forward < 1 or backward < 1:
        raise ValueError(
            f"prefetch depth must be >= 1; got forward={forward}, backward={backward}"
        )
    modules = [
        m for m in model.modules() if hasattr(m, "set_modules_to_forward_prefetch")
    ]
    for index, module in enumerate(modules):
        ahead = modules[index + 1 : index + 1 + forward]
        behind = modules[max(0, index - backward) : index][::-1]
        if ahead:
            module.set_modules_to_forward_prefetch(ahead)  # type: ignore[attr-defined]
        if behind and hasattr(module, "set_modules_to_backward_prefetch"):
            module.set_modules_to_backward_prefetch(behind)  # type: ignore[attr-defined]


def summarize_sharding(model: nn.Module) -> dict[str, int]:
    """Return per-rank parameter counts, split by sharded and replicated.

    A fast sanity check that sharding actually happened. If ``sharded`` is zero
    after :func:`apply_fsdp`, the mesh was wrong or the block attribute did not
    match, and the job will run — just with every rank holding a full copy.

    Args:
        model: The model to inspect.

    Returns:
        Counts of local elements, sharded parameters, and replicated parameters.
    """
    from torch.distributed.tensor import DTensor

    local_elements = 0
    sharded = 0
    replicated = 0
    for parameter in model.parameters():
        local_elements += (
            parameter.numel()
            if not isinstance(parameter, DTensor)
            else parameter.to_local().numel()
        )
        if isinstance(parameter, DTensor):
            sharded += 1
        else:
            replicated += 1
    return {
        "local_elements": local_elements,
        "sharded_params": sharded,
        "replicated_params": replicated,
    }
