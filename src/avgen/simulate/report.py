"""One report that answers "will this run, and how fast".

Ties the four estimators together into a single artifact a person can read, a CI
job can assert against, and a reviewer can attach to a design discussion.

The regression check is the part that earns its keep. Parallelism performance
regresses silently: someone adds a norm in the wrong place and suddenly tensor
parallelism all-gathers where it used to reduce-scatter, and nobody notices
until the next thousand-GPU run costs 20% more. Because the whole report is
produced on CPU in under a second, it can run on every pull request —
:meth:`SimulationReport.assert_no_regression` turns a performance property into
a test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
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
    transformer_flops,
)
from avgen.simulate.memory import MemoryEstimate, ModelShape, estimate_memory

__all__ = ["SimulationReport", "simulate_config"]

_GIB = 1024.0**3


@dataclass(frozen=True, slots=True)
class SimulationReport:
    """Everything the estimators know about one configuration.

    Args:
        shape: Model and batch geometry.
        dims: Parallelism degrees.
        activation_checkpoint: Checkpointing policy assumed.
        accelerator: Device profile.
        memory: Predicted per-rank memory.
        compute: Predicted per-rank compute.
        communication: Predicted per-step communication.
        gradient_accumulation: Microbatches per optimizer step.
        notes: Warnings and advisories raised during analysis.
    """

    shape: ModelShape
    dims: ParallelDims
    activation_checkpoint: ActivationCheckpointConfig
    accelerator: Accelerator
    memory: MemoryEstimate
    compute: ComputeEstimate
    communication: CommunicationEstimate
    gradient_accumulation: int = 1
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def fits(self) -> bool:
        """Whether the configuration is predicted to fit in device memory."""
        return self.memory.fits_in(self.accelerator.memory_gib)

    @property
    def step_seconds(self) -> float:
        """Predicted wall-clock seconds per optimizer step."""
        return self.communication.estimated_step_seconds * self.gradient_accumulation

    @property
    def global_batch_size(self) -> int:
        """Samples per optimizer step across the whole job."""
        return (
            self.dims.dp_size * self.shape.micro_batch_size * self.gradient_accumulation
        )

    @property
    def tokens_per_second(self) -> float:
        """Predicted global generative-token throughput."""
        if self.step_seconds <= 0:
            return 0.0
        return self.global_batch_size * self.shape.sequence_length / self.step_seconds

    @property
    def mfu(self) -> float:
        """Predicted Model FLOPs Utilisation."""
        return self.compute.mfu(
            self.step_seconds / max(1, self.gradient_accumulation),
            self.accelerator.peak_flops(),
        )

    def days_for_tokens(self, target_tokens: float) -> float:
        """Days of wall clock to consume a token budget at this throughput.

        The number that turns a parallelism decision into a budget decision.

        Args:
            target_tokens: Total generative tokens to train on.

        Returns:
            Days, or infinity when throughput is zero.
        """
        rate = self.tokens_per_second
        return target_tokens / (rate * 86400.0) if rate > 0 else float("inf")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe summary suitable for CI assertions.

        Returns:
            A flat-ish mapping of every headline number.
        """
        return {
            "plan": self.dims.describe(),
            "world_size": self.dims.world_size,
            "accelerator": self.accelerator.name,
            "sequence_length": self.shape.sequence_length,
            "parameters": self.shape.parameters,
            "micro_batch_size": self.shape.micro_batch_size,
            "gradient_accumulation": self.gradient_accumulation,
            "global_batch_size": self.global_batch_size,
            "activation_checkpoint": self.activation_checkpoint.mode,
            "fits": self.fits,
            "memory_gib": round(self.memory.total_gib, 3),
            "memory_breakdown_gib": {
                key: round(value, 3)
                for key, value in self.memory.breakdown_gib().items()
            },
            "dominant_memory_term": self.memory.dominant_term(),
            "step_seconds": round(self.step_seconds, 5),
            "compute_seconds": round(self.compute.seconds, 5),
            "exposed_comm_seconds": round(self.communication.exposed_seconds, 5),
            "comm_gib_per_step": round(self.communication.total_wire_gib, 4),
            "comm_by_label_seconds": {
                key: round(value, 5)
                for key, value in self.communication.by_label().items()
            },
            "comm_bottleneck": self.communication.bottleneck(),
            "scaling_efficiency": round(self.communication.scaling_efficiency, 4),
            "mfu": round(self.mfu, 4),
            "tokens_per_second": round(self.tokens_per_second, 1),
            "recompute_overhead": round(self.compute.recompute_overhead, 4),
            "notes": list(self.notes),
        }

    def save(self, path: str | Path) -> Path:
        """Write the report as JSON.

        Args:
            path: Destination file.

        Returns:
            The written path.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return destination

    def render(self) -> str:
        """Render a human-readable report.

        Returns:
            A multi-line string laid out for a terminal.
        """
        memory = self.memory.breakdown_gib()
        local_sequence = self.shape.sequence_length // max(
            1, self.dims.sequence_shard_size
        )
        lines = [
            "=" * 72,
            f"avgen simulation — {self.dims.describe()}",
            "=" * 72,
            f"  device                {self.accelerator.name} "
            f"({self.accelerator.memory_gib:.0f} GiB)",
            f"  parameters            {self.shape.parameters / 1e9:.2f} B",
            f"  depth x width         {self.shape.depth} x {self.shape.width}",
            f"  sequence length       {self.shape.sequence_length:,} tokens "
            f"({local_sequence:,} per rank)",
            f"  global batch          {self.global_batch_size} "
            f"(micro {self.shape.micro_batch_size} x accum {self.gradient_accumulation}"
            f" x dp {self.dims.dp_size})",
            "",
            f"MEMORY  {self.memory.total_gib:.1f} GiB per rank   "
            f"{'FITS' if self.fits else 'DOES NOT FIT'}",
        ]
        for name, value in memory.items():
            if name.endswith("_bytes") and value > 0.01:
                lines.append(f"    {name[:-6]:<20} {value:8.2f} GiB")
        lines += [
            "",
            f"COMPUTE  {self.compute.seconds * 1e3:.1f} ms/microbatch   "
            f"MFU {self.mfu:.1%}   recompute overhead "
            f"{self.compute.recompute_overhead:.1%}",
            "",
            f"COMMUNICATION  {self.communication.total_wire_gib:.2f} GiB/step   "
            f"exposed {self.communication.exposed_seconds * 1e3:.1f} ms   "
            f"scaling efficiency {self.communication.scaling_efficiency:.1%}",
        ]
        for label, seconds in self.communication.by_label().items():
            lines.append(f"    {label:<32} {seconds * 1e3:8.2f} ms")
        lines += [
            "",
            f"THROUGHPUT  {self.step_seconds:.3f} s/step   "
            f"{self.tokens_per_second / 1e6:.2f} M tokens/s",
        ]
        if self.notes:
            lines.append("")
            lines.append("NOTES")
            lines.extend(f"    - {note}" for note in self.notes)
        lines.append("=" * 72)
        return "\n".join(lines)

    def assert_no_regression(
        self,
        baseline: dict[str, Any],
        *,
        mfu_tolerance: float = 0.05,
        memory_tolerance_gib: float = 2.0,
    ) -> None:
        """Fail if this configuration got worse than a recorded baseline.

        Designed to be called from a CI job with a committed baseline JSON. It
        catches the two regressions that otherwise reach a cluster silently: a
        plan that stopped fitting, and a plan whose communication pattern got
        worse.

        Args:
            baseline: A previously saved :meth:`to_dict` output.
            mfu_tolerance: Allowed relative drop in predicted MFU.
            memory_tolerance_gib: Allowed absolute increase in per-rank memory.

        Raises:
            AssertionError: If the configuration regressed or stopped fitting.
        """
        current = self.to_dict()
        if baseline.get("fits") and not current["fits"]:
            raise AssertionError(
                f"configuration no longer fits: {current['memory_gib']:.1f} GiB "
                f"exceeds {self.accelerator.memory_gib:.0f} GiB "
                f"(was {baseline.get('memory_gib')} GiB). Dominant term: "
                f"{current['dominant_memory_term']}"
            )
        memory_delta = current["memory_gib"] - float(baseline.get("memory_gib", 0.0))
        if memory_delta > memory_tolerance_gib:
            raise AssertionError(
                f"per-rank memory grew by {memory_delta:.2f} GiB "
                f"(tolerance {memory_tolerance_gib} GiB); dominant term is "
                f"{current['dominant_memory_term']}"
            )
        baseline_mfu = float(baseline.get("mfu", 0.0))
        if baseline_mfu > 0:
            drop = (baseline_mfu - current["mfu"]) / baseline_mfu
            if drop > mfu_tolerance:
                raise AssertionError(
                    f"predicted MFU fell {drop:.1%} "
                    f"({baseline_mfu:.3f} -> {current['mfu']:.3f}); "
                    f"communication bottleneck is now {current['comm_bottleneck']}"
                )


def _advisories(
    shape: ModelShape,
    dims: ParallelDims,
    memory: MemoryEstimate,
    communication: CommunicationEstimate,
    accelerator: Accelerator,
) -> tuple[str, ...]:
    """Collect warnings a reviewer would otherwise have to notice by eye."""
    notes: list[str] = []
    if dims.tensor > 8:
        notes.append(
            f"tensor parallel degree {dims.tensor} likely spans nodes; TP "
            "communicates twice per block on the critical path and should stay "
            "inside one NVLink domain"
        )
    if dims.pp_enabled and dims.pipeline > shape.depth // 4:
        notes.append(
            f"pipeline degree {dims.pipeline} leaves fewer than 4 blocks per "
            "stage; the bubble will dominate"
        )
    if not memory.fits_in(accelerator.memory_gib):
        notes.append(
            f"does not fit: {memory.total_gib:.1f} GiB needed, "
            f"{accelerator.memory_gib:.0f} GiB available; largest term is "
            f"{memory.dominant_term()}"
        )
    if communication.scaling_efficiency < 0.8:
        notes.append(
            f"scaling efficiency {communication.scaling_efficiency:.0%}: "
            f"{communication.bottleneck()} is not hidden behind compute"
        )
    flops = transformer_flops(shape)
    if flops["attention_fraction"] > 0.5:
        notes.append(
            f"attention is {flops['attention_fraction']:.0%} of forward FLOPs at "
            f"{shape.sequence_length:,} tokens; context parallelism and an "
            "efficient attention kernel matter more here than parameter sharding"
        )
    if dims.cp_enabled and shape.sequence_length % dims.sequence_shard_size != 0:
        notes.append(
            f"sequence length {shape.sequence_length} does not divide evenly by "
            f"cp*tp = {dims.sequence_shard_size}; pad at the data boundary"
        )
    return tuple(notes)


def simulate_config(
    shape: ModelShape,
    dims: ParallelDims,
    *,
    accelerator: Accelerator = H100_SXM,
    activation_checkpoint: ActivationCheckpointConfig | None = None,
    intra_node: Interconnect = NVLINK4,
    inter_node: Interconnect = INFINIBAND_NDR,
    achieved_fraction: float = 0.45,
    gradient_accumulation: int = 1,
) -> SimulationReport:
    """Price one configuration end to end.

    Args:
        shape: Model and batch geometry.
        dims: Parallelism degrees.
        accelerator: Device profile.
        activation_checkpoint: Checkpointing policy. ``None`` means store
            everything.
        intra_node: Fast fabric profile.
        inter_node: Slow fabric profile.
        achieved_fraction: Fraction of peak FLOPs assumed achievable.
        gradient_accumulation: Microbatches per optimizer step.

    Returns:
        The full report.
    """
    policy = activation_checkpoint or ActivationCheckpointConfig(mode="none")
    memory = estimate_memory(shape, dims, activation_checkpoint=policy)
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
    return SimulationReport(
        shape=shape,
        dims=dims,
        activation_checkpoint=policy,
        accelerator=accelerator,
        memory=memory,
        compute=compute,
        communication=communication,
        gradient_accumulation=gradient_accumulation,
        notes=_advisories(shape, dims, memory, communication, accelerator),
    )
