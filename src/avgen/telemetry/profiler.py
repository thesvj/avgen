"""The three tools that actually diagnose a slow, fat, or hung training job.

Each answers a different question, and reaching for the wrong one wastes a day.

**"Where is the time going?"** — :func:`profile_steps`. A kernel trace of a
handful of steps, viewable in Perfetto or ``chrome://tracing``. It shows the
gaps: a launch-bound region where the GPU idles between tiny kernels, an
all-gather that did not overlap with compute, a dataloader stall that looks like
a slow forward pass. Profiling is *expensive* — it can halve step time and the
trace for a 1024-rank job would be terabytes — so it is scheduled (a few steps,
once) and rank-gated (one rank per node at most).

**"Why did it run out of memory?"** — :func:`memory_snapshot`. A peak-memory
number tells you that you ran out; it does not tell you what was holding the
memory or why the allocator could not find a contiguous block. The snapshot
records every allocation with its Python stack, and the viewer at
``pytorch.org/memory_viz`` draws the address space over time. Fragmentation is
visible there as a striped pattern of small live blocks between large free
gaps — and it is essentially invisible in any aggregate statistic. This is the
tool that finds the bucketed-video-loader fragmentation problem described in
:class:`~avgen.telemetry.metrics.MemoryReporter`.

**"Why did it hang?"** — :func:`flight_recorder_dump`. A hang at scale gives you
nothing: no exception, no log line, 1024 processes sitting in ``ncclAllReduce``.
The flight recorder is a per-rank ring buffer of recent collectives, and
comparing the dumps across ranks identifies the collective that some ranks
entered and others did not. That mismatch is nearly always the actual bug — a
rank that took a different branch and skipped a collective, which is why every
docstring in this package that describes a collective says *every rank must call
this*.

Nothing here is enabled by default. Profiling changes the thing it measures, and
a framework that profiles unconditionally is lying about its own throughput.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

__all__ = [
    "flight_recorder_dump",
    "memory_snapshot",
    "profile_steps",
    "should_profile_rank",
]

_LOG = logging.getLogger("avgen.telemetry")

#: Allocation records kept by the memory recorder. 100k covers several steps of
#: a large model; the buffer is host memory, so a much larger value is a real
#: cost for a marginal amount of extra history.
DEFAULT_MEMORY_HISTORY_ENTRIES = 100_000


def should_profile_rank(*, mode: str = "global_zero") -> bool:
    """Decide whether this rank participates in profiling.

    Args:
        mode: ``"global_zero"`` profiles rank 0 only — the right default, since
            one trace answers most questions and N traces answer the same one N
            times. ``"local_zero"`` profiles one rank per node, which is what
            you want when investigating a node-local problem (a bad NIC, a
            throttling GPU, an NVMe that is slower on one host). ``"all"``
            profiles everything and should only ever be used on a job of a
            handful of ranks.

    Returns:
        Whether to profile on this process.

    Raises:
        ValueError: If ``mode`` is not recognised.
    """
    if mode == "all":
        return True
    if mode == "global_zero":
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank()) == 0
        return True
    if mode == "local_zero":
        raw = os.environ.get("LOCAL_RANK", "0")
        return raw.lstrip("-").isdigit() and int(raw) == 0
    raise ValueError(
        f"mode must be 'global_zero', 'local_zero', or 'all'; got {mode!r}"
    )


@contextmanager
def profile_steps(
    output_dir: str | os.PathLike[str],
    *,
    enabled: bool = True,
    wait: int = 1,
    warmup: int = 1,
    active: int = 3,
    repeat: int = 1,
    rank_mode: str = "global_zero",
    record_shapes: bool = False,
    profile_memory: bool = False,
    with_stack: bool = False,
    with_flops: bool = False,
) -> Iterator[Any]:
    """Trace a few training steps and write a Chrome/Perfetto trace.

    The caller must call ``profiler.step()`` once per training step on the
    yielded object; that is what advances the schedule. Yielding the profiler
    rather than driving it internally keeps this usable around any loop shape,
    including gradient accumulation where "a step" is several forward passes.

    **The schedule is the whole point.** ``wait`` skips steps so the loop
    reaches steady state; ``warmup`` runs the profiler without recording so
    CUPTI's own first-call overhead and any lazy kernel compilation land
    outside the measurement; ``active`` records. Profiling from step zero
    instead measures ``torch.compile``, cuDNN autotuning, and a cold allocator,
    and produces a trace whose hot spot is a one-time cost.

    Args:
        output_dir: Directory for the trace files. One file per rank.
        enabled: Master switch. When false this yields a null object with a
            no-op ``step``, so the training loop needs no conditional.
        wait: Steps to skip at the start of each cycle.
        warmup: Steps to run the profiler on without recording.
        active: Steps to record.
        repeat: Number of ``wait``/``warmup``/``active`` cycles. ``1`` is right
            for a diagnostic; more only to compare early and late behaviour.
        rank_mode: Which ranks profile; see :func:`should_profile_rank`.
        record_shapes: Record input shapes. Necessary to attribute time to a
            specific bucket in a variable-resolution video run; roughly 10%
            extra overhead.
        profile_memory: Record allocator events. Prefer
            :func:`memory_snapshot` for a memory investigation — it is both
            cheaper and far more informative.
        with_stack: Record Python stacks. Expensive, and the only way to map a
            kernel back to the model code that launched it.
        with_flops: Estimate FLOPs per operator. A sanity check on the analytic
            model in :mod:`avgen.simulate.compute`, not a substitute for it.

    Yields:
        The active ``torch.profiler.profile``, or a null object with a no-op
        ``step`` method when profiling is disabled on this rank.
    """
    if not enabled or not should_profile_rank(mode=rank_mode):
        with nullcontext(_NullProfiler()) as null:
            yield null
        return

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    def _export(profiler: Any) -> None:
        # Named by rank and by the profiler's own step counter so repeated
        # cycles do not overwrite each other, and so traces from different
        # ranks can be loaded side by side.
        target = directory / f"trace_rank{rank}_step{profiler.step_num}.json.gz"
        profiler.export_chrome_trace(str(target))
        _LOG.info("wrote profiler trace -> %s", target)

    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=wait, warmup=warmup, active=active, repeat=repeat
        ),
        on_trace_ready=_export,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
        with_flops=with_flops,
    ) as profiler:
        yield profiler


class _NullProfiler:
    """Stand-in with the profiler's ``step`` method and none of its cost."""

    __slots__ = ()

    def step(self) -> None:
        """Do nothing, so the training loop is rank- and config-agnostic."""

    def export_chrome_trace(self, path: str) -> None:
        """Do nothing.

        Args:
            path: Ignored.
        """


@contextmanager
def memory_snapshot(
    path: str | os.PathLike[str],
    *,
    enabled: bool = True,
    max_entries: int = DEFAULT_MEMORY_HISTORY_ENTRIES,
    rank_mode: str = "global_zero",
    stacks: str = "python",
) -> Iterator[None]:
    """Record every allocation inside the block and dump it for the viewer.

    Open the result at https://pytorch.org/memory_viz. The picture it draws —
    the device address space on one axis and time on the other — is the only
    practical way to see *fragmentation* as opposed to *usage*. Two runs with
    an identical peak-allocated figure look completely different here when one
    of them is interleaving long-lived and short-lived blocks.

    Use it around the step that OOMs, not around the whole run: the recorder
    keeps a bounded history and a long block simply discards the early part.
    The idiomatic use is a few steps at the start (to see the steady-state
    pattern) and a few steps at the largest sequence bucket (to see the spike).

    Args:
        path: Destination ``.pickle`` file. Parent directories are created.
        enabled: Master switch.
        max_entries: Allocation records retained.
        rank_mode: Which ranks record; see :func:`should_profile_rank`.
        stacks: Which stacks to capture — ``"python"`` or ``"all"``.
            ``"all"`` includes C++ frames and is much slower to record and to
            render; ``"python"`` is enough to identify the offending module.

    Yields:
        None.
    """
    if not (enabled and torch.cuda.is_available()):
        yield
        return
    if not should_profile_rank(mode=rank_mode):
        yield
        return

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.memory._record_memory_history(max_entries=max_entries, stacks=stacks)
    try:
        yield
    finally:
        try:
            torch.cuda.memory._dump_snapshot(str(destination))
            _LOG.info("wrote memory snapshot -> %s", destination)
        except Exception:
            # This block usually runs while unwinding an OOM. Losing the
            # snapshot is bad; replacing the OOM traceback with a dump failure
            # is worse, because the OOM is the thing being debugged.
            _LOG.warning(
                "failed to dump memory snapshot to %s", destination, exc_info=True
            )
        # Recording is global process state, so it must be turned off even on
        # the failure path or every later allocation keeps paying for it.
        torch.cuda.memory._record_memory_history(enabled=None)


def flight_recorder_dump(path: str | os.PathLike[str]) -> Path | None:
    """Dump this rank's NCCL flight recorder buffer for hang analysis.

    **How to actually use this.** The recorder is armed by
    :func:`avgen.parallel.init_distributed`, which sets
    ``TORCH_NCCL_TRACE_BUFFER_SIZE`` and ``TORCH_NCCL_DUMP_ON_TIMEOUT`` before
    the first communicator exists (they are read once, at communicator
    creation — setting them later does nothing). From then on every collective
    is recorded with its sequence number, input and output sizes, and the state
    it reached: enqueued, started, or completed.

    When a job hangs, each rank's dump is a list of recent collectives. Line
    them up by ``seq_id`` and one of three patterns appears:

    * **Some ranks are missing a collective entirely.** Those ranks took a
      different branch — an ``if rank == 0`` around a save, a metric logged on
      one rank, an early ``break`` on a rank whose shard ran out of data. This
      is the common case and the bug is in the training loop, not in NCCL.
    * **Every rank entered the same collective; some never completed it.** A
      hardware or fabric problem on the ranks that are stuck. The dump names the
      device.
    * **Ranks are on different collectives with the same ``seq_id``.** The
      collectives were issued in different orders on different ranks — usually
      an unordered iteration over a ``dict`` or ``set`` that happens to hash
      differently, which is why CONTRACTS forbids ``set`` iteration order in
      anything affecting computation.

    On a timeout the watchdog dumps automatically to
    ``TORCH_NCCL_DEBUG_INFO_TEMP_FILE``. This function is the manual trigger,
    for a job that is hung but has not yet hit its timeout — which is when you
    want the data, since a shortened timeout will abort the process before you
    can attach a debugger.

    Args:
        path: Destination file for this rank's dump.

    Returns:
        The written path, or ``None`` when the recorder is unavailable (no
        process group, a non-NCCL backend, or a build without the trace
        buffer).
    """
    if not (dist.is_available() and dist.is_initialized()):
        return None
    dump = getattr(torch._C._distributed_c10d, "_dump_nccl_trace", None)
    if dump is None:
        _LOG.warning("this torch build has no NCCL flight recorder")
        return None
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = dump()
    except Exception:
        # A hung job must still get whatever diagnostic is available; raising
        # here would replace "the job is hung" with "the dump failed".
        _LOG.warning("NCCL flight recorder dump failed", exc_info=True)
        return None
    destination.write_bytes(payload if isinstance(payload, bytes) else bytes(payload))
    _LOG.info("wrote NCCL flight recorder dump -> %s", destination)
    return destination
