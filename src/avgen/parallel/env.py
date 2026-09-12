"""Process-group lifecycle, device binding, and collective health at scale.

Three things in this module exist specifically because of failures that only
appear past a few hundred ranks:

1. **Device binding before ``init_process_group``.** NCCL picks a device from
   the current CUDA context. If the context is not already bound to this rank's
   local GPU, every rank on a node can end up initialising communicators on
   device 0 — which either deadlocks or silently serialises everything.

2. **Finite collective timeouts, with a longer one for initialisation.** The
   default 30-minute timeout means a single hung rank stalls a 1024-GPU job for
   half an hour before anyone finds out. A short steady-state timeout surfaces a
   straggler quickly; a generous init timeout tolerates the genuinely slow
   rendezvous of a large job starting on a cold filesystem.

3. **The flight recorder.** NCCL's trace buffer records in-flight collectives so
   that when a job does hang, you can identify *which* collective on *which*
   rank never completed. Without it, a hang at scale is nearly undiagnosable.
   It costs a small ring buffer per rank; enable it always.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn

__all__ = [
    "DistributedEnv",
    "barrier",
    "init_distributed",
    "is_distributed_launch",
    "local_rank_device",
    "shutdown_distributed",
    "unwrap_model",
]

#: Steady-state collective timeout. Long enough for a large all-gather on a
#: congested fabric, short enough that a dead rank is noticed within minutes.
DEFAULT_TIMEOUT = timedelta(minutes=10)

#: Rendezvous timeout. A 1024-rank job on a cold shared filesystem can take
#: several minutes just to agree that everyone has arrived.
DEFAULT_INIT_TIMEOUT = timedelta(minutes=30)


@dataclass(frozen=True, slots=True)
class DistributedEnv:
    """Identity of one rank within a job.

    Args:
        rank: Global rank.
        local_rank: Rank within this node, which is also the CUDA device index.
        world_size: Total ranks.
        local_world_size: Ranks on this node.
        backend: Active collective backend, or ``"none"`` when single-process.
        device: The device this rank owns.
    """

    rank: int
    local_rank: int
    world_size: int
    local_world_size: int
    backend: str
    device: torch.device

    @property
    def is_main(self) -> bool:
        """Whether this rank writes checkpoints, reports, and logs."""
        return self.rank == 0

    @property
    def is_local_main(self) -> bool:
        """Whether this rank is first on its node.

        The right granularity for node-local work: populating a page cache,
        extracting an archive to local NVMe, or writing a per-node profile.
        """
        return self.local_rank == 0

    @property
    def is_distributed(self) -> bool:
        """Whether more than one process participates."""
        return self.world_size > 1

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the job."""
        return max(1, self.world_size // max(1, self.local_world_size))


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as error:
        raise ValueError(
            f"environment variable {name}={raw!r} is not an int"
        ) from error


def is_distributed_launch() -> bool:
    """Return whether the standard launcher variables are set."""
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def local_rank_device() -> torch.device:
    """Return this rank's device without initialising a process group.

    Safe to call before ``init_distributed``; useful for allocating a model on
    the right device during a dry run or a simulation.

    Returns:
        This rank's CUDA device, or CPU when no GPU is visible.
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")
    local_rank = _env_int("LOCAL_RANK", 0)
    return torch.device("cuda", local_rank % max(1, torch.cuda.device_count()))


def init_distributed(
    *,
    backend: str | None = None,
    timeout: timedelta = DEFAULT_TIMEOUT,
    init_timeout: timedelta = DEFAULT_INIT_TIMEOUT,
    enable_flight_recorder: bool = True,
    trace_buffer_size: int = 2000,
) -> DistributedEnv:
    """Initialise the process group and bind this rank's device.

    Returns a single-process environment unchanged when not launched under
    ``torchrun``, so the same script runs on a laptop and on a cluster.

    Args:
        backend: Collective backend. Defaults to ``cpu:gloo,cuda:nccl`` on CUDA
            and ``gloo`` otherwise. The CPU half is required by asynchronous
            distributed checkpointing; overriding this with plain ``"nccl"``
            will break every checkpoint save.
        timeout: Steady-state collective timeout.
        init_timeout: Rendezvous timeout, applied by temporarily raising the
            store timeout during initialisation.
        enable_flight_recorder: Whether to enable NCCL's trace buffer. Leave on:
            it is the only practical way to diagnose a hang at scale.
        trace_buffer_size: Number of collectives retained in the ring buffer.

    Returns:
        This rank's environment.

    Raises:
        ValueError: If the launcher variables are inconsistent.
    """
    if not is_distributed_launch():
        return DistributedEnv(
            rank=0,
            local_rank=0,
            world_size=1,
            local_world_size=1,
            backend="none",
            device=local_rank_device(),
        )

    rank = _env_int("RANK", 0)
    world_size = _env_int("WORLD_SIZE", 1)
    local_rank = _env_int("LOCAL_RANK", 0)
    local_world_size = _env_int("LOCAL_WORLD_SIZE", 1)
    if world_size < 1:
        raise ValueError(f"WORLD_SIZE must be positive; got {world_size}")
    if not 0 <= rank < world_size:
        raise ValueError(f"RANK must be in [0, {world_size}); got {rank}")

    if enable_flight_recorder:
        # Set before the first communicator is created; NCCL reads these once.
        os.environ.setdefault("TORCH_NCCL_TRACE_BUFFER_SIZE", str(trace_buffer_size))
        os.environ.setdefault("TORCH_NCCL_DUMP_ON_TIMEOUT", "1")
        # Ask the watchdog to abort the process rather than leaving a zombie
        # rank that holds the job's allocation until the scheduler kills it.
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

    # A CPU backend must be present alongside NCCL, and this is not optional.
    # torch.distributed.checkpoint's async_save stages the state dict and then
    # runs its planning collectives on a CPU process group; with a NCCL-only
    # group it asserts "A CPU backend must be enabled for async save" and every
    # checkpoint on a GPU job fails. The multi-backend spelling creates both
    # groups from one call and costs nothing when the CPU one is unused.
    resolved_backend = backend or (
        "cpu:gloo,cuda:nccl" if torch.cuda.is_available() else "gloo"
    )

    # Bind the device *before* init_process_group so NCCL builds communicators
    # on the right GPU rather than inheriting device 0 from an unset context.
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if not dist.is_initialized():
        previous = os.environ.get("TORCH_NCCL_BLOCKING_WAIT")
        os.environ["TORCH_DISTRIBUTED_STORE_TIMEOUT"] = str(
            int(init_timeout.total_seconds())
        )
        try:
            kwargs: dict[str, object] = {
                "backend": resolved_backend,
                "rank": rank,
                "world_size": world_size,
                "timeout": timeout,
            }
            if device.type == "cuda":
                # Pins the default device for this process group so barriers and
                # object collectives do not warn or guess.
                kwargs["device_id"] = device
            dist.init_process_group(**kwargs)
        finally:
            if previous is None:
                os.environ.pop("TORCH_NCCL_BLOCKING_WAIT", None)

    return DistributedEnv(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        local_world_size=local_world_size,
        backend=resolved_backend,
        device=device,
    )


def barrier(env: DistributedEnv | None = None) -> None:
    """Synchronise every rank; a no-op when single-process.

    Args:
        env: The environment, used to pass the correct device to NCCL.
    """
    if env is not None and not env.is_distributed:
        return
    if not (dist.is_available() and dist.is_initialized()):
        return
    if env is not None and env.device.type == "cuda":
        dist.barrier(device_ids=[env.local_rank])
    else:
        dist.barrier()


def shutdown_distributed() -> None:
    """Destroy the process group if one exists.

    Always call this on a clean exit. Skipping it leaves NCCL communicators
    open, which on some fabrics delays the next job's allocation.
    """
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the innermost module beneath any parallel wrappers.

    Handles DDP, activation-checkpoint wrappers, and ``torch.compile``'s
    ``OptimizedModule``, in any nesting order. FSDP2 needs no unwrapping — it
    mutates the module in place rather than wrapping it, which is one of its
    quieter advantages.

    Args:
        model: A possibly wrapped module.

    Returns:
        The underlying module.
    """
    # Each wrapper stores its inner module under a different attribute, and they
    # nest in any order: compile(checkpoint(ddp(model))) is as legal as the
    # reverse. Looping over all three names until nothing changes handles every
    # ordering without encoding one.
    attributes = ("_orig_mod", "_checkpoint_wrapped_module", "module")
    current = model
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        for attribute in attributes:
            inner = getattr(current, attribute, None)
            if isinstance(inner, nn.Module):
                current = inner
                break
        else:
            break
    return current


@contextmanager
def collective_timeout(seconds: float) -> Iterator[None]:
    """Temporarily shorten the collective timeout for a fragile region.

    Useful around a checkpoint save, where a hung filesystem should surface in
    seconds rather than after the steady-state timeout expires.

    Args:
        seconds: Timeout to apply inside the block.

    Yields:
        None.
    """
    if not (dist.is_available() and dist.is_initialized()):
        yield
        return
    group = dist.distributed_c10d._get_default_group()
    try:
        dist.distributed_c10d._set_pg_timeout(timedelta(seconds=seconds), group)
        yield
    finally:
        dist.distributed_c10d._set_pg_timeout(DEFAULT_TIMEOUT, group)
