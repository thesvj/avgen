"""Validate a parallelism plan before spending a cluster on it.

The premise: every question you would normally answer by launching a job and
watching it OOM can be answered on a laptop in under a second, and the ones that
cannot be answered analytically can be *measured* on a laptop using PyTorch's
fake process group and fake tensors.

Four things this package does that are hard to do any other way:

**Build a 1024-rank mesh on one machine.** ``FakeProcessGroup`` completes
collectives instantly, so real sharding logic, real ``DeviceMesh`` construction,
and real FSDP/TP/CP plans all execute at the target world size. If the plan is
malformed, it fails here instead of at minute forty of a cluster allocation.

**Predict per-rank memory before allocating any.** A closed-form model fast
enough to sweep the whole configuration space, calibrated against PyTorch's own
``MemTracker`` so the sweep stays honest.

**Count the collectives you will actually issue.** ``CommDebugMode`` attributes
every collective to the module that issued it. This is how you find out that
your tensor-parallel plan is all-gathering where it should reduce-scatter.

**Export a trace for a real network simulator.** ``ExecutionTraceObserver``
produces a graph that converts to an MLCommons Chakra trace, which ASTRA-sim
consumes — for the topology questions an analytical model cannot answer.

Quickstart::

    from avgen.parallel import ParallelDims
    from avgen.simulate import ModelShape, SearchSpace, search_parallel_plan
    from avgen.simulate.plan import render_plan_table
    from avgen.simulate.compute import H100_SXM

    shape = ModelShape(
        parameters=2_000_000_000, depth=32, width=2560,
        sequence_length=65_536, micro_batch_size=1, num_heads=20,
    )
    best = search_parallel_plan(shape, SearchSpace(world_size=512))
    print(render_plan_table(best, H100_SXM))

Nothing here tells you whether a model *learns*. Fake collectives return
uninitialised data, so a simulated loss is meaningless. This package answers
"does it fit, is it shaped right, how fast will it be" — and those are the
questions that cost the most to answer any other way.
"""

from avgen.simulate.chakra import (
    TraceArtifacts,
    capture_execution_trace,
    capture_with_timing,
)
from avgen.simulate.comms import (
    ETHERNET_200G,
    INFINIBAND_NDR,
    NVLINK4,
    NVLINK5,
    CollectiveCost,
    CommunicationEstimate,
    Interconnect,
    calibrate_from_busbw,
    count_collectives,
    estimate_step_communication,
)
from avgen.simulate.compute import (
    A100_80GB,
    B200,
    H100_SXM,
    H200_SXM,
    Accelerator,
    ComputeEstimate,
    estimate_compute,
    measure_runtime,
    suggest_activation_policy,
    transformer_flops,
)
from avgen.simulate.memory import (
    OPTIMIZER_BYTES,
    MemoryEstimate,
    ModelShape,
    calibration_error,
    estimate_memory,
    measure_memory,
)
from avgen.simulate.plan import (
    PlanCandidate,
    SearchSpace,
    render_plan_table,
    search_parallel_plan,
)
from avgen.simulate.report import SimulationReport, simulate_config
from avgen.simulate.world import (
    FakeWorld,
    fake_tensors,
    fake_world,
    meta_init,
    simulate_rank,
)

__all__ = [
    "A100_80GB",
    "B200",
    "ETHERNET_200G",
    "H100_SXM",
    "H200_SXM",
    "INFINIBAND_NDR",
    "NVLINK4",
    "NVLINK5",
    "OPTIMIZER_BYTES",
    "Accelerator",
    "CollectiveCost",
    "CommunicationEstimate",
    "ComputeEstimate",
    "FakeWorld",
    "Interconnect",
    "MemoryEstimate",
    "ModelShape",
    "PlanCandidate",
    "SearchSpace",
    "SimulationReport",
    "TraceArtifacts",
    "calibrate_from_busbw",
    "calibration_error",
    "capture_execution_trace",
    "capture_with_timing",
    "count_collectives",
    "estimate_compute",
    "estimate_memory",
    "estimate_step_communication",
    "fake_tensors",
    "fake_world",
    "measure_memory",
    "measure_runtime",
    "meta_init",
    "render_plan_table",
    "search_parallel_plan",
    "simulate_config",
    "simulate_rank",
    "suggest_activation_policy",
    "transformer_flops",
]
