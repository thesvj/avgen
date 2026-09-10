"""Communication accounting: what gets sent, how much, and how long it takes.

Two halves again.

**Observed.** :func:`count_collectives` runs a real step under
``CommDebugMode``, which intercepts every functional collective and attributes
it to the module that issued it. Combined with a fake process group this works
at any simulated world size on one machine. It answers questions that are
otherwise guesswork: is FSDP issuing one all-gather per block or one per
parameter? Did the tensor-parallel plan actually insert reduce-scatter, or is it
silently all-gathering and throwing the result away?

**Predicted.** :func:`estimate_step_communication` applies textbook ring
formulas to the message sizes a plan implies. For a ring algorithm over ``n``
ranks moving ``S`` bytes:

===================  =========================  ==============================
Collective            Bytes on the wire          Time at bus bandwidth ``B``
===================  =========================  ==============================
all-reduce            ``2S(n-1)/n``              ``2S(n-1)/(nB)``
reduce-scatter        ``S(n-1)/n``               ``S(n-1)/(nB)``
all-gather            ``S(n-1)/n``               ``S(n-1)/(nB)``
all-to-all            ``S(n-1)/n``               ``S(n-1)/(nB)``
===================  =========================  ==============================

Those formulas are asymptotic. Real collectives pay a per-message latency and
fall short of peak bandwidth, so :class:`Interconnect` carries an achievable
fraction and a latency term rather than a headline number. **Calibrate it.** The
defaults are plausible for common hardware, not measured on yours; run
``nccl-tests`` or :func:`calibrate_from_busbw` once and the whole predictive
layer becomes trustworthy.

The one number worth internalising: at ``cp=8`` on a 100k-token sequence, ring
attention moves the entire key/value tensor around the ring once per block. That
is the dominant communication cost of a video DiT, and it is invisible in any
LLM-derived cost model.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from avgen.parallel.dims import ParallelDims
from avgen.simulate.memory import ModelShape

__all__ = [
    "CollectiveCost",
    "CommunicationEstimate",
    "Interconnect",
    "calibrate_from_busbw",
    "count_collectives",
    "estimate_step_communication",
]

_GIB = 1024.0**3


@dataclass(frozen=True, slots=True)
class Interconnect:
    """Achievable bandwidth and latency of one communication tier.

    Args:
        name: Human label, e.g. ``"NVLink 4"`` or ``"InfiniBand NDR"``.
        peak_gbps: Unidirectional peak bandwidth in gigabytes per second.
        efficiency: Fraction of peak a real collective achieves. 0.7-0.85 is
            typical for a well-tuned NCCL ring; below 0.5 means something is
            misconfigured, usually a missing NCCL topology hint.
        latency_us: Per-collective fixed cost in microseconds. Dominates for
            small messages, which is exactly what gradient buckets are if you
            let them get too small.
        ranks_per_group: Ranks that share this tier, e.g. GPUs per node for
            NVLink.
    """

    name: str
    peak_gbps: float
    efficiency: float = 0.8
    latency_us: float = 5.0
    ranks_per_group: int = 8

    def __post_init__(self) -> None:
        """Validate the bandwidth and efficiency."""
        if self.peak_gbps <= 0.0:
            raise ValueError(f"peak_gbps must be positive; got {self.peak_gbps!r}")
        if not 0.0 < self.efficiency <= 1.0:
            raise ValueError(f"efficiency must be in (0, 1]; got {self.efficiency!r}")

    @property
    def achievable_bytes_per_second(self) -> float:
        """Bandwidth actually available to a collective, in bytes per second."""
        return self.peak_gbps * 1e9 * self.efficiency

    def transfer_seconds(self, byte_count: float) -> float:
        """Time to move a number of bytes across this tier.

        Args:
            byte_count: Bytes on the wire.

        Returns:
            Seconds, including the fixed latency term.
        """
        return self.latency_us * 1e-6 + byte_count / self.achievable_bytes_per_second


#: Plausible starting points. **Replace these with measurements** — they exist
#: so a first simulation runs, not so it is accurate on your fabric.
NVLINK4 = Interconnect(
    "NVLink 4 (H100)", peak_gbps=450.0, efficiency=0.85, latency_us=3.0
)
NVLINK5 = Interconnect(
    "NVLink 5 (B200)", peak_gbps=900.0, efficiency=0.85, latency_us=3.0
)
INFINIBAND_NDR = Interconnect(
    "InfiniBand NDR 400G",
    peak_gbps=50.0,
    efficiency=0.75,
    latency_us=8.0,
    ranks_per_group=8,
)
ETHERNET_200G = Interconnect(
    "RoCE 200G", peak_gbps=25.0, efficiency=0.6, latency_us=15.0, ranks_per_group=8
)


def calibrate_from_busbw(
    name: str,
    *,
    measured_busbw_gbps: float,
    peak_gbps: float,
    latency_us: float = 5.0,
    ranks_per_group: int = 8,
) -> Interconnect:
    """Build a calibrated interconnect from an ``nccl-tests`` bus-bandwidth run.

    ``all_reduce_perf`` reports *bus bandwidth*, which already folds in the
    ``2(n-1)/n`` ring factor. Dividing it by the hardware peak gives the
    efficiency term directly, which is the single most valuable calibration you
    can do — it converts every downstream time estimate from a guess into a
    projection.

    Example::

        # ./build/all_reduce_perf -b 1G -e 8G -f 2 -g 8   ->  busbw 372 GB/s
        nvlink = calibrate_from_busbw(
            "NVLink 4", measured_busbw_gbps=372.0, peak_gbps=450.0
        )

    Args:
        name: Label for the tier.
        measured_busbw_gbps: Bus bandwidth reported by nccl-tests, in GB/s.
        peak_gbps: Hardware peak for the tier.
        latency_us: Measured small-message latency.
        ranks_per_group: Ranks sharing the tier.

    Returns:
        A calibrated interconnect.
    """
    return Interconnect(
        name=name,
        peak_gbps=peak_gbps,
        efficiency=min(1.0, measured_busbw_gbps / peak_gbps),
        latency_us=latency_us,
        ranks_per_group=ranks_per_group,
    )


@dataclass(frozen=True, slots=True)
class CollectiveCost:
    """One collective's contribution to a step.

    Args:
        label: What issues it, e.g. ``"fsdp.all_gather"``.
        kind: Collective type.
        payload_bytes: Logical tensor size before the ring factor.
        wire_bytes: Bytes actually on the wire.
        ranks: Participating ranks.
        count: Times per step.
        tier: Interconnect it crosses.
        seconds: Estimated total time per step.
    """

    label: str
    kind: str
    payload_bytes: float
    wire_bytes: float
    ranks: int
    count: int
    tier: str
    seconds: float


def _ring_wire_bytes(kind: str, payload: float, ranks: int) -> float:
    if ranks <= 1:
        return 0.0
    factor = (ranks - 1) / ranks
    if kind == "all_reduce":
        return 2.0 * payload * factor
    return payload * factor


@dataclass(frozen=True, slots=True)
class CommunicationEstimate:
    """Predicted per-step communication for a whole parallelism plan.

    Args:
        collectives: Every modelled collective.
        compute_seconds: Compute time the communication is overlapped against.
        overlap_efficiency: Fraction of communication hidden behind compute.
    """

    collectives: tuple[CollectiveCost, ...]
    compute_seconds: float = 0.0
    overlap_efficiency: float = 0.8

    @property
    def total_seconds(self) -> float:
        """Total communication time if nothing overlapped."""
        return sum(item.seconds for item in self.collectives)

    @property
    def total_wire_gib(self) -> float:
        """Total bytes on the wire per step, in gibibytes."""
        return sum(item.wire_bytes * item.count for item in self.collectives) / _GIB

    @property
    def exposed_seconds(self) -> float:
        """Communication time that compute cannot hide.

        This is the number that actually costs throughput. Communication fully
        hidden behind compute is free; what is left over is pure loss.
        """
        hidden = min(self.total_seconds * self.overlap_efficiency, self.compute_seconds)
        return max(0.0, self.total_seconds - hidden)

    @property
    def estimated_step_seconds(self) -> float:
        """Predicted wall-clock time for one optimizer step."""
        return self.compute_seconds + self.exposed_seconds

    @property
    def scaling_efficiency(self) -> float:
        """Fraction of ideal throughput retained after communication.

        1.0 means communication is entirely hidden. Below roughly 0.8, the plan
        is spending more on moving data than on arithmetic and a different
        factorisation of the same world size will beat it.
        """
        total = self.estimated_step_seconds
        return self.compute_seconds / total if total > 0 else 1.0

    def by_label(self) -> dict[str, float]:
        """Return seconds per collective label, largest first."""
        totals: dict[str, float] = {}
        for item in self.collectives:
            totals[item.label] = totals.get(item.label, 0.0) + item.seconds
        return dict(sorted(totals.items(), key=lambda kv: kv[1], reverse=True))

    def bottleneck(self) -> str:
        """Return the collective label costing the most time per step."""
        ordered = self.by_label()
        return next(iter(ordered), "none")


def estimate_step_communication(
    shape: ModelShape,
    dims: ParallelDims,
    *,
    intra_node: Interconnect = NVLINK4,
    inter_node: Interconnect = INFINIBAND_NDR,
    param_dtype_bytes: int = 2,
    reduce_dtype_bytes: int = 4,
    compute_seconds: float = 0.0,
    overlap_efficiency: float = 0.8,
) -> CommunicationEstimate:
    """Predict the collectives one optimizer step will issue, and their cost.

    Models, in order of usual significance for a video DiT:

    * **Context-parallel ring attention** — key and value shards rotate around
      the ring once per block. Scales with sequence length and depth, and is
      almost always the largest term once ``cp > 1``.
    * **FSDP all-gather** — unsharded parameters per block per forward, plus
      again in backward when ``reshard_after_forward`` is on.
    * **FSDP reduce-scatter** — gradients, once per block, in reduce dtype.
    * **Tensor-parallel all-reduce** (or reduce-scatter/all-gather pairs under
      sequence parallelism) — twice per block on the activations.
    * **HSDP all-reduce** — gradients across replica groups, inter-node.

    Args:
        shape: Model and batch geometry.
        dims: Parallelism degrees.
        intra_node: Tier used by dimensions that fit inside a node.
        inter_node: Tier used by dimensions that span nodes.
        param_dtype_bytes: Bytes per parameter in compute precision.
        reduce_dtype_bytes: Bytes per gradient element during reduction.
        compute_seconds: Estimated compute time, for the overlap calculation.
        overlap_efficiency: Fraction of communication hidden behind compute.

    Returns:
        The estimate.
    """
    items: list[CollectiveCost] = []
    ranks_per_node = intra_node.ranks_per_group

    def add(
        label: str,
        kind: str,
        payload: float,
        ranks: int,
        count: int,
    ) -> None:
        if ranks <= 1 or count <= 0 or payload <= 0:
            return
        tier = intra_node if ranks <= ranks_per_node else inter_node
        wire = _ring_wire_bytes(kind, payload, ranks)
        items.append(
            CollectiveCost(
                label=label,
                kind=kind,
                payload_bytes=payload,
                wire_bytes=wire,
                ranks=ranks,
                count=count,
                tier=tier.name,
                seconds=tier.transfer_seconds(wire) * count,
            )
        )

    params_per_block = shape.parameters / max(1, shape.depth)
    local_depth = max(1, shape.depth // max(1, dims.pipeline))
    local_sequence = shape.sequence_length / max(1, dims.sequence_shard_size)

    # --- Context parallel: the video-specific term. -----------------------
    if dims.cp_enabled:
        head_dim = shape.width // max(1, shape.num_heads)
        kv_elements = (
            shape.micro_batch_size
            * local_sequence
            * shape.num_heads
            * head_dim
            * 2.0  # keys and values
        )
        kv_bytes = kv_elements * param_dtype_bytes / max(1, dims.tensor)
        # Ring attention passes each shard to every other rank, forward and
        # again in backward.
        add(
            "context_parallel.ring_kv",
            "all_to_all",
            kv_bytes,
            dims.context,
            local_depth * 2,
        )

    # --- FSDP2. -----------------------------------------------------------
    if dims.dp_shard_enabled:
        gather_ranks = dims.dp_shard * dims.context
        block_bytes = params_per_block * param_dtype_bytes / max(1, dims.tensor)
        # Forward gather, plus a backward re-gather when resharding.
        add("fsdp.all_gather", "all_gather", block_bytes, gather_ranks, local_depth * 2)
        add(
            "fsdp.reduce_scatter",
            "reduce_scatter",
            params_per_block * reduce_dtype_bytes / max(1, dims.tensor),
            gather_ranks,
            local_depth,
        )

    # --- HSDP: gradients across replica groups. ---------------------------
    if dims.dp_replicate_enabled:
        local_params = shape.parameters / max(
            1, dims.dp_shard * dims.context * dims.tensor * dims.pipeline
        )
        add(
            "hsdp.all_reduce",
            "all_reduce",
            local_params * reduce_dtype_bytes,
            dims.dp_replicate,
            1,
        )

    # --- Tensor parallel: activations, twice per block. -------------------
    if dims.tp_enabled:
        activation_bytes = (
            shape.micro_batch_size * local_sequence * shape.width * param_dtype_bytes
        )
        kind = "reduce_scatter" if dims.sequence_shard_size > 1 else "all_reduce"
        # Attention output and feed-forward output, forward and backward.
        add(
            "tensor_parallel.activation",
            kind,
            activation_bytes,
            dims.tensor,
            local_depth * 4,
        )

    # --- Pipeline: activation handoff between stages. ---------------------
    if dims.pp_enabled:
        boundary = (
            shape.micro_batch_size * local_sequence * shape.width * param_dtype_bytes
        )
        add("pipeline.p2p", "all_gather", boundary, 2, dims.pipeline - 1)

    return CommunicationEstimate(
        collectives=tuple(items),
        compute_seconds=compute_seconds,
        overlap_efficiency=overlap_efficiency,
    )


def count_collectives(
    step: Callable[[], Any],
    *,
    trace_file: str | None = None,
) -> dict[str, Any]:
    """Run a step under ``CommDebugMode`` and report every collective issued.

    This is the ground truth for the predictive model above, and it works under
    a fake process group — so a 1024-rank communication profile can be produced
    on a laptop.

    Args:
        step: Zero-argument callable running one forward and backward.
        trace_file: Optional path for a module-wise tracing table, which shows
            *where* each collective came from rather than only how many there
            were.

    Returns:
        Total counts by collective, per-module counts, and parameter sharding
        information.
    """
    from torch.distributed.tensor.debug import CommDebugMode

    mode = CommDebugMode()
    with mode:
        step()

    result: dict[str, Any] = {
        "total_counts": {str(k): v for k, v in mode.get_total_counts().items()}
        if callable(getattr(mode, "get_total_counts", None))
        else {},
        "module_counts": {
            str(module): {str(op): count for op, count in ops.items()}
            for module, ops in mode.get_comm_counts().items()
        },
    }
    with contextlib_suppress():
        result["sharding"] = {
            str(k): str(v) for k, v in mode.get_sharding_info().items()
        }
    if trace_file is not None:
        mode.log_comm_debug_tracing_table_to_file(file_name=trace_file)
        result["trace_file"] = trace_file
    return result


def contextlib_suppress() -> Any:
    """Return a suppress-all context manager.

    ``CommDebugMode``'s sharding introspection is only populated when DTensors
    were involved; asking for it otherwise raises. Suppressing keeps the report
    useful for non-DTensor models rather than failing the whole call.

    Returns:
        A context manager that swallows exceptions.
    """
    import contextlib

    return contextlib.suppress(Exception)
