"""Search the parallelism space instead of guessing at it.

Given a model shape, a world size, and a machine, there are usually a few dozen
valid ways to factor the ranks across data, context, tensor, and pipeline
parallelism. Only a handful fit in memory, and among those the throughput spread
is routinely 2-3x. Picking by intuition means leaving most of a cluster on the
floor, and finding out takes a full job launch per guess.

This module enumerates the space, prices every candidate with the memory,
communication, and compute models, and ranks what survives. A sweep over a
1024-GPU search space takes well under a second and needs no GPU.

The constraints encoded here are the ones that come from hardware rather than
from taste:

* **Tensor parallelism must not cross a node.** It communicates twice per block
  on the critical path; over a slower inter-node fabric that is catastrophic
  rather than merely bad.
* **Context parallelism must divide the sequence.** Ragged shards are rejected
  upstream, so a degree that does not divide is simply invalid.
* **Pipeline parallelism must divide the depth**, and wants several blocks per
  stage — a two-block stage is nearly all bubble.
* **Global batch size must factor across the data dimension**, or the effective
  batch silently changes and two runs stop being comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from avgen.parallel.activation import ActivationCheckpointConfig
from avgen.parallel.dims import ParallelDims
from avgen.simulate.comms import (
    INFINIBAND_NDR,
    NVLINK4,
    CommunicationEstimate,
    Interconnect,
    estimate_step_communication,
)
from avgen.simulate.compute import (
    H100_SXM,
    Accelerator,
    ComputeEstimate,
    estimate_compute,
)
from avgen.simulate.memory import MemoryEstimate, ModelShape, estimate_memory

__all__ = ["PlanCandidate", "SearchSpace", "search_parallel_plan"]


def _divisors(value: int, *, maximum: int | None = None) -> list[int]:
    limit = value if maximum is None else min(value, maximum)
    return [d for d in range(1, limit + 1) if value % d == 0]


@dataclass(frozen=True, slots=True)
class SearchSpace:
    """Bounds and constraints on the parallelism search.

    Args:
        world_size: Total ranks available.
        gpus_per_node: Ranks sharing the fast intra-node fabric.
        max_tensor: Cap on tensor-parallel degree. Never exceeds
            ``gpus_per_node`` regardless of what is requested here.
        max_context: Cap on context-parallel degree.
        max_pipeline: Cap on pipeline-parallel degree. Defaults to one:
            pipeline is the last axis to reach for.
        allow_hsdp: Whether to consider replicated-over-sharded data
            parallelism.
        global_batch_size: Samples per optimizer step, if it must factor.
        min_blocks_per_stage: Refuse pipeline splits thinner than this.
        activation_policies: Checkpointing policies to try for each candidate.
    """

    world_size: int
    gpus_per_node: int = 8
    max_tensor: int = 8
    max_context: int = 16
    max_pipeline: int = 1
    allow_hsdp: bool = True
    global_batch_size: int | None = None
    min_blocks_per_stage: int = 4
    activation_policies: tuple[ActivationCheckpointConfig, ...] = (
        ActivationCheckpointConfig(mode="none"),
        ActivationCheckpointConfig(mode="selective_op", save_op_frequency=1),
        ActivationCheckpointConfig(mode="selective_op", save_op_frequency=2),
        ActivationCheckpointConfig(mode="full"),
    )

    def __post_init__(self) -> None:
        """Validate the world and node sizes."""
        if self.world_size < 1:
            raise ValueError(f"world_size must be positive; got {self.world_size!r}")
        if self.gpus_per_node < 1:
            raise ValueError(
                f"gpus_per_node must be positive; got {self.gpus_per_node!r}"
            )


@dataclass(frozen=True, slots=True)
class PlanCandidate:
    """One priced parallelism configuration.

    Args:
        dims: The degrees.
        activation_checkpoint: The checkpointing policy assumed.
        memory: Predicted per-rank memory.
        compute: Predicted per-rank compute.
        communication: Predicted per-step communication.
        micro_batch_size: Samples per rank per microbatch.
        gradient_accumulation: Microbatches per optimizer step.
    """

    dims: ParallelDims
    activation_checkpoint: ActivationCheckpointConfig
    memory: MemoryEstimate
    compute: ComputeEstimate
    communication: CommunicationEstimate
    micro_batch_size: int
    gradient_accumulation: int = 1

    @property
    def step_seconds(self) -> float:
        """Predicted wall-clock seconds per optimizer step."""
        return self.communication.estimated_step_seconds * self.gradient_accumulation

    @property
    def samples_per_second(self) -> float:
        """Predicted global throughput in samples per second."""
        if self.step_seconds <= 0:
            return 0.0
        per_step = (
            self.dims.dp_size * self.micro_batch_size * self.gradient_accumulation
        )
        return per_step / self.step_seconds

    def mfu(self, accelerator: Accelerator, *, use_fp8: bool = False) -> float:
        """Predicted Model FLOPs Utilisation.

        Args:
            accelerator: Device profile.
            use_fp8: Whether to price against the fp8 peak.

        Returns:
            MFU in ``[0, 1]``.
        """
        return self.compute.mfu(
            self.step_seconds / max(1, self.gradient_accumulation),
            accelerator.peak_flops(fp8=use_fp8),
        )

    def summary(self, accelerator: Accelerator) -> dict[str, Any]:
        """Return a flat, JSON-safe summary row.

        Args:
            accelerator: Device profile, for the MFU column.

        Returns:
            The row.
        """
        return {
            "plan": self.dims.describe(),
            "activation_checkpoint": self.activation_checkpoint.mode,
            "micro_batch": self.micro_batch_size,
            "grad_accum": self.gradient_accumulation,
            "memory_gib": round(self.memory.total_gib, 2),
            "memory_headroom_gib": round(
                accelerator.memory_gib - self.memory.total_gib, 2
            ),
            "step_seconds": round(self.step_seconds, 4),
            "samples_per_second": round(self.samples_per_second, 3),
            "mfu": round(self.mfu(accelerator), 4),
            "scaling_efficiency": round(self.communication.scaling_efficiency, 4),
            "comm_gib_per_step": round(self.communication.total_wire_gib, 3),
            "bottleneck": self.communication.bottleneck(),
            "dominant_memory": self.memory.dominant_term(),
        }


def search_parallel_plan(
    shape: ModelShape,
    space: SearchSpace,
    *,
    accelerator: Accelerator = H100_SXM,
    intra_node: Interconnect = NVLINK4,
    inter_node: Interconnect = INFINIBAND_NDR,
    achieved_fraction: float = 0.45,
    headroom: float = 0.10,
    top_k: int = 10,
    throughput_tie_tolerance: float = 0.02,
) -> list[PlanCandidate]:
    """Enumerate and rank every viable parallelism configuration.

    Args:
        shape: Model and batch geometry. ``micro_batch_size`` is the per-rank
            microbatch the search holds fixed.
        space: Search bounds and constraints.
        accelerator: Device profile.
        intra_node: Fast fabric profile.
        inter_node: Slow fabric profile.
        achieved_fraction: Fraction of peak FLOPs assumed.
        headroom: Device memory fraction to keep free.
        top_k: How many candidates to return.
        throughput_tie_tolerance: Relative throughput window within which plans
            are considered tied and ranked by memory headroom instead.

    Returns:
        The best candidates by predicted throughput, fastest first. An empty
        list means nothing fits — the fix is more ranks, a shorter sequence, or
        a smaller model, not a different checkpointing policy.
    """
    tensor_cap = min(space.max_tensor, space.gpus_per_node, space.world_size)
    candidates: list[PlanCandidate] = []

    for tensor in _divisors(space.world_size, maximum=tensor_cap):
        # Tensor parallelism must stay inside one node.
        if tensor > space.gpus_per_node:
            continue
        for pipeline in _divisors(space.world_size, maximum=space.max_pipeline):
            if shape.depth % pipeline != 0:
                continue
            if shape.depth // pipeline < space.min_blocks_per_stage:
                continue
            remaining_after_tp = space.world_size // (tensor * pipeline)
            for context in _divisors(remaining_after_tp, maximum=space.max_context):
                # The sequence must split evenly across cp and the sequence
                # parallel factor from tp.
                if shape.sequence_length % (context * tensor) != 0:
                    continue
                remaining = remaining_after_tp // context
                replicate_options = _divisors(remaining) if space.allow_hsdp else [1]
                for replicate in replicate_options:
                    shard = remaining // replicate
                    # HSDP with a shard group of one is just replication; skip
                    # the duplicate encoding of the same plan.
                    if replicate > 1 and shard == 1:
                        continue
                    try:
                        dims = ParallelDims(
                            world_size=space.world_size,
                            dp_replicate=replicate,
                            dp_shard=shard,
                            tensor=tensor,
                            context=context,
                            pipeline=pipeline,
                        )
                    except ValueError:
                        continue

                    accumulation = 1
                    if space.global_batch_size is not None:
                        try:
                            accumulation = dims.gradient_accumulation_for(
                                global_batch_size=space.global_batch_size,
                                local_batch_size=shape.micro_batch_size,
                            )
                        except ValueError:
                            continue

                    for policy in space.activation_policies:
                        memory = estimate_memory(
                            shape, dims, activation_checkpoint=policy
                        )
                        if not memory.fits_in(
                            accelerator.memory_gib, headroom=headroom
                        ):
                            continue
                        compute = estimate_compute(
                            shape,
                            dims,
                            accelerator=accelerator,
                            activation_checkpoint=policy,
                            achieved_fraction=achieved_fraction,
                        )
                        communication = estimate_step_communication(
                            shape,
                            dims,
                            intra_node=intra_node,
                            inter_node=inter_node,
                            compute_seconds=compute.seconds,
                        )
                        candidates.append(
                            PlanCandidate(
                                dims=dims,
                                activation_checkpoint=policy,
                                memory=memory,
                                compute=compute,
                                communication=communication,
                                micro_batch_size=shape.micro_batch_size,
                                gradient_accumulation=accumulation,
                            )
                        )
                        # The policies are ordered cheapest-first; once one
                        # fits, a more aggressive one only costs compute.
                        break

    if not candidates:
        return []

    # When the global batch size is pinned, throughput is nearly identical
    # across plans by construction: a larger data dimension simply means less
    # gradient accumulation, and the two cancel. The differences that remain are
    # a fraction of a percent of communication overhead, which is well inside
    # the error bars of any analytical model. Ranking on that alone would be
    # noise, so plans within a small tolerance of the best are treated as tied
    # and broken on memory: among equally fast plans, the one with the most
    # headroom is strictly better — it tolerates a longer clip, a larger
    # microbatch, or an unlucky allocator without falling over.
    best_throughput = max(c.samples_per_second for c in candidates)
    tolerance = best_throughput * (1.0 - throughput_tie_tolerance)

    def _rank(candidate: PlanCandidate) -> tuple[int, float, float]:
        tied = candidate.samples_per_second >= tolerance
        return (
            0 if tied else 1,
            candidate.memory.total_gib if tied else -candidate.samples_per_second,
            -candidate.samples_per_second,
        )

    candidates.sort(key=_rank)
    return candidates[:top_k]


def render_plan_table(
    candidates: list[PlanCandidate],
    accelerator: Accelerator,
) -> str:
    """Render ranked candidates as a fixed-width table.

    Args:
        candidates: Output of :func:`search_parallel_plan`.
        accelerator: Device profile, for the MFU column.

    Returns:
        A printable table, or an explanatory line when nothing fits.
    """
    if not candidates:
        return (
            "No configuration fits. Options, in order of how much they help:\n"
            "  1. Raise context parallelism (needs more ranks) — sequence "
            "length is almost always the binding constraint for video.\n"
            "  2. Shorten the clip or lower the resolution.\n"
            "  3. Use a device with more memory.\n"
            "  4. Reduce depth or width."
        )
    columns = [
        ("plan", 34),
        ("activation_checkpoint", 20),
        ("memory_gib", 11),
        ("step_seconds", 13),
        ("samples_per_second", 19),
        ("mfu", 7),
        ("scaling_efficiency", 19),
        ("bottleneck", 26),
    ]
    header = " ".join(name.ljust(width) for name, width in columns)
    lines = [header, "-" * len(header)]
    for candidate in candidates:
        row = candidate.summary(accelerator)
        lines.append(" ".join(str(row[name]).ljust(width) for name, width in columns))
    return "\n".join(lines)
