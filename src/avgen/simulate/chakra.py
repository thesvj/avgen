"""Export execution traces for external network simulators.

The analytical model in :mod:`avgen.simulate.comms` is good enough to rank
plans. It is not good enough to answer questions about the *network*: what
happens when a fat-tree oversubscribes, whether a rail-optimised topology beats
a flat one, how much a congested neighbour costs you. Those need a real network
simulator, and the industry standard entry point is
`ASTRA-sim <https://astra-sim.github.io/>`_ driven by an MLCommons
`Chakra <https://mlcommons.org/working-groups/research/chakra/>`_ execution
trace.

PyTorch emits the upstream half natively via ``ExecutionTraceObserver``. The
pipeline is::

    avgen trace  →  PyTorch ET (JSON)
                 →  chakra_trace_link   (merge with a Kineto trace for timing)
                 →  chakra_converter    (PyTorch ET → Chakra ET protobuf)
                 →  ASTRA-sim           (network + system simulation)

This module owns the first step and documents the rest, rather than vendoring a
converter that would rot against Chakra's schema. Capture on **rank 0 only**
unless you specifically need per-rank divergence: a full trace of a 1024-rank
job is hundreds of gigabytes and the ranks are nearly identical by construction.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["TraceArtifacts", "capture_execution_trace", "capture_with_timing"]

CONVERSION_GUIDE = """\
Convert the captured PyTorch execution trace for ASTRA-sim:

  # 1. Install the Chakra toolchain
  pip install "chakra @ git+https://github.com/mlcommons/chakra.git"

  # 2. Link the execution trace with the Kineto timing trace
  chakra_trace_link \\
      --chakra-host-trace {et_path} \\
      --chakra-device-trace {kineto_path} \\
      --output-file linked.json

  # 3. Convert to the Chakra ET protobuf
  chakra_converter PyTorch \\
      --input linked.json \\
      --output chakra.et \\
      --num-ranks {world_size}

  # 4. Run ASTRA-sim against your network description
  astra-sim \\
      --workload-configuration=chakra \\
      --system-configuration=system.json \\
      --network-configuration=network.yml \\
      --remote-memory-configuration=memory.json
"""


@dataclass(frozen=True, slots=True)
class TraceArtifacts:
    """Files produced by a trace capture.

    Args:
        execution_trace: PyTorch execution-trace JSON, the operator graph.
        kineto_trace: Kineto profiler trace with real kernel timings, or
            ``None`` when only the graph was captured.
        world_size: World size the trace was captured at, needed by the
            converter.
    """

    execution_trace: Path
    kineto_trace: Path | None
    world_size: int

    def conversion_commands(self) -> str:
        """Return the exact commands to turn this into a Chakra trace.

        Returns:
            A shell snippet with the paths filled in.
        """
        return CONVERSION_GUIDE.format(
            et_path=self.execution_trace,
            kineto_path=self.kineto_trace or "<run capture_with_timing to produce one>",
            world_size=self.world_size,
        )

    def summarize(self) -> dict[str, Any]:
        """Return counts of recorded nodes by operator type.

        A quick sanity check that the trace captured what you expect: if the
        collective count is zero, the observer was registered outside the
        distributed region and the trace is useless for network simulation.

        Returns:
            Node totals and the ten most frequent operators.
        """
        payload = json.loads(self.execution_trace.read_text())
        nodes = payload.get("nodes", []) if isinstance(payload, dict) else []
        counts: dict[str, int] = {}
        collectives = 0
        for node in nodes:
            name = str(node.get("name", "?"))
            counts[name] = counts.get(name, 0) + 1
            if "c10d" in name or "nccl" in name or "all_" in name or "reduce" in name:
                collectives += 1
        top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
        return {
            "total_nodes": len(nodes),
            "distinct_ops": len(counts),
            "collective_nodes": collectives,
            "top_operators": dict(top),
        }


@contextmanager
def capture_execution_trace(output: str | Path) -> Iterator[Path]:
    """Record a PyTorch execution trace for the enclosed block.

    Capture *steady-state* steps, never the first one. Step zero includes lazy
    module initialisation, autotuning, and the first ``torch.compile``, none of
    which recur, and all of which make the trace unrepresentative of the run.

    Args:
        output: Path for the trace JSON.

    Yields:
        The output path.
    """
    from torch.profiler import ExecutionTraceObserver

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    observer = ExecutionTraceObserver()
    observer.register_callback(str(path))
    observer.start()
    try:
        yield path
    finally:
        observer.stop()
        observer.unregister_callback()


def capture_with_timing(
    step: Callable[[], Any],
    output_dir: str | Path,
    *,
    world_size: int,
    warmup_steps: int = 3,
    active_steps: int = 1,
) -> TraceArtifacts:
    """Capture both the operator graph and real kernel timings.

    ASTRA-sim needs both: the execution trace gives the dependency graph, the
    Kineto trace gives how long each node actually took. Linking them produces a
    workload description that reflects your real kernels rather than a cost
    model's guess at them.

    Args:
        step: Zero-argument callable running one full training step.
        output_dir: Directory for both traces.
        world_size: World size being traced, recorded for the converter.
        warmup_steps: Steps to run before recording, to get past compilation
            and autotuning.
        active_steps: Steps to record.

    Returns:
        Paths to both traces plus the conversion recipe.
    """
    import torch
    from torch.profiler import ProfilerActivity, profile, schedule

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    et_path = directory / "execution_trace.json"
    kineto_path = directory / "kineto_trace.json"

    for _ in range(warmup_steps):
        step()

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with capture_execution_trace(et_path):
        with profile(
            activities=activities,
            schedule=schedule(wait=0, warmup=0, active=active_steps),
            record_shapes=True,
            with_stack=False,
        ) as profiler:
            for _ in range(active_steps):
                step()
                profiler.step()
        profiler.export_chrome_trace(str(kineto_path))

    return TraceArtifacts(
        execution_trace=et_path,
        kineto_trace=kineto_path,
        world_size=world_size,
    )
