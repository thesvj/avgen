"""Simulate a thousand-GPU world on one machine.

The problem this solves is concrete. A parallelism plan for a 1024-GPU job has
to be right *before* you spend 1024 GPU-hours discovering it is not, and the
failure modes are the ones that do not show up at small scale: a mesh dimension
ordered so tensor-parallel traffic crosses nodes, an FSDP wrap that produces one
giant all-gather, a context-parallel shard that forgets to shard the rotary
tables, an activation-checkpoint policy that leaves the model 4 GB over budget.

PyTorch ships the pieces to check all of that without a cluster:

* ``FakeProcessGroup`` — a backend whose collectives complete instantly and
  return uninitialised data. Enough to build a real 1024-rank ``DeviceMesh``,
  run real sharding logic, and observe real communication *patterns*.
* ``FakeTensorMode`` and the ``meta`` device — tensors with real shapes, dtypes,
  and strides but no storage. A 30B-parameter model costs no memory to
  construct.
* ``CommDebugMode`` — counts and attributes every collective by module.
* ``MemTracker`` / ``FSDPMemTracker`` — module-wise memory accounting.
* ``RuntimeEstimator`` — per-operator time estimates from a roofline model or
  from measured kernel benchmarks.

**What a simulation does and does not tell you.** It is exact about shapes,
sharding, memory accounting, collective counts, and collective sizes. It is an
*estimate* for time, because it does not model your fabric's congestion, your
scheduler's placement, or your neighbours' traffic. And it is silent about
numerics: fake collectives return garbage, so a simulated run's loss is
meaningless. Use it to answer "does this plan fit and is it shaped right",
never "does this model learn".
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from avgen.parallel.dims import ParallelDims

__all__ = ["FakeWorld", "fake_world", "simulate_rank"]


@dataclass(slots=True)
class FakeWorld:
    """A simulated distributed world backed by ``FakeProcessGroup``.

    Args:
        dims: The parallelism degrees being simulated.
        rank: Which rank this simulation is impersonating. Rank 0 is the usual
            choice; simulating a middle rank is useful when checking that
            pipeline stage assignment is balanced.
        device_type: Device type the mesh is built on. ``None`` auto-detects:
            ``"cuda"`` when a GPU is present, ``"cpu"`` otherwise. The choice
            does not change what the simulation measures — shapes, sharding,
            collective counts and message sizes are identical either way — but
            ``fully_shard`` queries the current device of the mesh's device
            type, so asking for ``"cuda"`` on a CPU-only machine fails inside
            PyTorch rather than in avgen.

    Attributes:
        mesh: The device mesh, populated on entry and ``None`` outside the
            context. Not a constructor argument: it is built from ``dims`` when
            the context is entered, because a mesh outlives no rendezvous.
    """

    dims: ParallelDims
    rank: int = 0
    device_type: str | None = None
    mesh: DeviceMesh | None = field(default=None, init=False)
    _entered: bool = field(default=False, init=False, repr=False)
    _saved_env: dict[str, str | None] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Validate the impersonated rank and resolve the device type."""
        if not 0 <= self.rank < self.dims.world_size:
            raise ValueError(
                f"rank must be in [0, {self.dims.world_size}); got {self.rank!r}"
            )
        if self.device_type is None:
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
            object.__setattr__(self, "device_type", resolved)

    def __enter__(self) -> FakeWorld:
        """Initialise the fake process group and build the mesh."""
        if dist.is_initialized():
            raise RuntimeError(
                "a real process group is already initialised; run the simulator "
                "in a fresh process, not inside a live training job"
            )
        from torch.testing._internal.distributed.fake_pg import FakeStore

        for name, value in (
            ("RANK", str(self.rank)),
            ("WORLD_SIZE", str(self.dims.world_size)),
            ("LOCAL_RANK", str(self.rank % 8)),
            ("LOCAL_WORLD_SIZE", "8"),
        ):
            self._saved_env[name] = os.environ.get(name)
            os.environ[name] = value

        dist.init_process_group(
            backend="fake",
            store=FakeStore(),
            rank=self.rank,
            world_size=self.dims.world_size,
        )
        self._entered = True
        self.mesh = self.dims.build_mesh(self.device_type)
        return self

    def __exit__(self, *_exc: object) -> None:
        """Tear down the fake process group and restore the environment."""
        if self._entered:
            with contextlib.suppress(Exception):
                dist.destroy_process_group()
            self._entered = False
        for name, value in self._saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        self._saved_env.clear()
        self.mesh = None

    @property
    def resolved_device_type(self) -> str:
        """The device type actually in use after auto-detection."""
        return self.device_type or "cpu"

    def require_mesh(self) -> DeviceMesh:
        """Return the mesh, raising if the world is not active.

        Returns:
            The device mesh.

        Raises:
            RuntimeError: If used outside the ``with`` block.
        """
        if self.mesh is None:
            raise RuntimeError("FakeWorld must be used as a context manager")
        return self.mesh

    def describe(self) -> dict[str, Any]:
        """Return a JSON-safe description of the simulated topology.

        Returns:
            Mesh dimension names and sizes, plus derived coordinates.
        """
        mesh = self.require_mesh()
        names = tuple(mesh.mesh_dim_names or ())
        data_rank, data_world = self.dims.data_coordinates(mesh)
        seq_index, seq_count = self.dims.sequence_coordinates(mesh)
        return {
            "world_size": self.dims.world_size,
            "rank": self.rank,
            "mesh_dims": {name: mesh[name].size() for name in names},
            "mesh_order": list(names),
            "data_rank": data_rank,
            "data_world": data_world,
            "sequence_shard": seq_index,
            "sequence_shards": seq_count,
            "dp_size": self.dims.dp_size,
            "model_shard_size": self.dims.model_shard_size,
        }


@contextlib.contextmanager
def fake_world(
    dims: ParallelDims,
    *,
    rank: int = 0,
    device_type: str | None = None,
) -> Iterator[FakeWorld]:
    """Open a simulated world for the duration of a block.

    Args:
        dims: Parallelism degrees to simulate.
        rank: Rank to impersonate.
        device_type: Device type for the mesh, or ``None`` to auto-detect.

    Yields:
        The active world.
    """
    world = FakeWorld(dims=dims, rank=rank, device_type=device_type)
    with world:
        yield world


@contextlib.contextmanager
def meta_init() -> Iterator[None]:
    """Construct modules on the ``meta`` device.

    Parameters get real shapes and dtypes but no storage, so a 30B-parameter
    model is free to build. This is also how a real large-scale job should
    initialise: build on meta, apply the parallel plans, then materialise only
    this rank's shard. Nothing ever has to fit unsharded.

    Yields:
        None.
    """
    with torch.device("meta"):
        yield


@contextlib.contextmanager
def fake_tensors(allow_fallback: bool = True) -> Iterator[Any]:
    """Run under ``FakeTensorMode`` so operations execute symbolically.

    Unlike ``meta``, fake tensors carry a device, so code that branches on
    ``tensor.is_cuda`` or calls ``torch.cuda`` APIs behaves as it would on a
    real GPU. That makes it the right mode for exercising a full training step
    rather than just constructing a model.

    Args:
        allow_fallback: Whether operators without a meta kernel may fall back
            to real execution on a small tensor.

    Yields:
        The active fake mode.
    """
    from torch._subclasses.fake_tensor import FakeTensorMode

    mode = FakeTensorMode(allow_fallback_kernels=allow_fallback)
    with mode:
        yield mode


def simulate_rank(
    dims: ParallelDims,
    build: Callable[[], nn.Module],
    apply_plan: Callable[[nn.Module, ParallelDims, DeviceMesh], nn.Module],
    *,
    rank: int = 0,
    device_type: str | None = None,
) -> dict[str, Any]:
    """Build and parallelise a model inside a simulated world.

    This is the smallest useful end-to-end check: it proves the plan applies
    cleanly at the target scale, and reports what each rank ends up holding.
    Run it in CI on every change to a model or a plan, and a whole class of
    "it worked on 8 GPUs" regressions stops reaching the cluster.

    Args:
        dims: Parallelism degrees to simulate.
        build: Callable returning a fresh, unsharded model. Invoked under
            ``meta_init`` so its size does not matter.
        apply_plan: Callable applying the parallelism transformations.
        rank: Rank to impersonate.
        device_type: Device type for the mesh, or ``None`` to auto-detect.

    Returns:
        Topology description plus per-rank parameter accounting.
    """
    from avgen.parallel.fsdp import summarize_sharding

    with fake_world(dims, rank=rank, device_type=device_type) as world:
        mesh = world.require_mesh()
        with meta_init():
            model = build()
        total_params = sum(p.numel() for p in model.parameters())
        model = apply_plan(model, dims, mesh)
        report = world.describe()
        report["global_parameters"] = total_params
        report.update(summarize_sharding(model))
        report["parameters_per_rank"] = report.pop("local_elements")
        report["shard_efficiency"] = (
            total_params / (report["parameters_per_rank"] * dims.model_shard_size)
            if report["parameters_per_rank"]
            else 0.0
        )
        return report
