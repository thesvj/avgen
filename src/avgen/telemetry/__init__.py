"""Metrics, logging, and the diagnostics that make a large run debuggable.

Two rules govern everything in this package, and both exist because telemetry
that is naive about scale is worse than no telemetry at all.

**One rank logs.** A thousand ranks writing the same line produces a thousand
times the IO and a file no tool can read. Rank gating lives in the logger base
class, not at the call sites.

**One synchronisation per log interval, not per step.** Metrics accumulate on
device and are reduced across the ``dp_cp`` mesh exactly once, at log cadence.
A ``.item()`` in the training step stalls the host/device pipeline every step
for numbers nobody reads until the interval ends.

Typical use::

    from avgen.telemetry import MetricAccumulator, ThroughputMeter, build_logger
    from avgen.parallel import data_mesh

    logger = build_logger(["console", "jsonl"], log_dir=run_dir)
    accumulator = MetricAccumulator(device, mesh=data_mesh(parallel.mesh))
    meter = ThroughputMeter(peak_flops_per_s=peak, flops_per_step=flops)

    for step in range(total_steps):
        metrics = trainer.train_step(batch, ...)
        accumulator.update(metrics)          # no sync
        meter.step(samples=global_batch, tokens=global_tokens)
        if step % log_every == 0:
            logger.log_metrics(
                merge_metrics(accumulator.flush(), meter.report().to_mapping()),
                step,
            )
"""

from avgen.telemetry.logger import (
    ConsoleLogger,
    JSONLLogger,
    Logger,
    MultiLogger,
    NoOpLogger,
    RankGatedLogger,
    TensorBoardLogger,
    WandbLogger,
    build_logger,
)
from avgen.telemetry.metrics import (
    MemoryReport,
    MemoryReporter,
    MetricAccumulator,
    ThroughputMeter,
    ThroughputReport,
    merge_metrics,
)
from avgen.telemetry.profiler import (
    flight_recorder_dump,
    memory_snapshot,
    profile_steps,
    should_profile_rank,
)

__all__ = [
    "ConsoleLogger",
    "JSONLLogger",
    "Logger",
    "MemoryReport",
    "MemoryReporter",
    "MetricAccumulator",
    "MultiLogger",
    "NoOpLogger",
    "RankGatedLogger",
    "TensorBoardLogger",
    "ThroughputMeter",
    "ThroughputReport",
    "WandbLogger",
    "build_logger",
    "flight_recorder_dump",
    "memory_snapshot",
    "merge_metrics",
    "profile_steps",
    "should_profile_rank",
]
