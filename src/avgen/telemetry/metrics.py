"""On-device metric accumulation, throughput accounting, and memory headroom.

The unifying rule of this module is: **never synchronise inside a training
step**.

``float(tensor)``, ``tensor.item()``, ``print(loss)`` and every ``if
loss > threshold`` on a CUDA tensor all do the same thing — they block the host
until the device has finished every kernel queued so far. In steady state the
host runs ahead of the device by tens of milliseconds, queuing the next step's
kernels while the current one executes; one synchronisation collapses that
pipeline and the GPU idles until the host catches up again. Measured on a video
DiT step, a single per-step ``.item()`` costs low single-digit percent of
throughput. Eight of them — one per :class:`~avgen.core.metrics.StepMetrics`
field — cost real money, every step, for the entire run, to produce numbers
nobody reads until the log interval anyway.

:class:`MetricAccumulator` therefore keeps running sums as device tensors, does
**one** collective over the ``dp_cp`` mesh at log cadence, and **one** transfer
to host for the whole batch of metrics. At a log interval of 50 steps that is
two synchronisations per 50 steps instead of 400.

The ``dp_cp`` mesh rather than the whole world is not an optimisation, it is
correctness — see :mod:`avgen.parallel.comm`. Tensor-parallel ranks computed the
same loss for the same tokens; averaging over them too triple-counts a 3-way TP
job and reports a number that is right by accident only when TP is 1.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields

import torch
from torch.distributed.device_mesh import DeviceMesh

from avgen.core.metrics import StepMetrics
from avgen.parallel.comm import all_reduce_max, all_reduce_sum

__all__ = [
    "MemoryReport",
    "MemoryReporter",
    "MetricAccumulator",
    "ThroughputMeter",
    "ThroughputReport",
    "merge_metrics",
]

_LOG = logging.getLogger("avgen.telemetry")

#: Fields averaged over steps rather than summed. Token counts and the skip
#: counter are totals; losses and gradient norms are means.
_MEAN_FIELDS = ("loss", "video_loss", "audio_loss", "grad_norm")
_SUM_FIELDS = ("valid_video_tokens", "valid_audio_tokens", "nonfinite", "skipped")


class MetricAccumulator:
    """Sums :class:`~avgen.core.metrics.StepMetrics` on device, syncs once.

    Every accumulated field lives in a single flat float64 buffer rather than
    eight separate scalars. That is what makes the flush cheap: one
    ``all_reduce`` of one contiguous tensor instead of eight round trips through
    the collective library, and one host transfer instead of eight. At 1024
    ranks the difference between one collective and eight per log interval is
    the difference between logging being free and logging being a straggler
    amplifier.

    float64 for the accumulator, not float32: a run logging every 50 steps with
    a loss around 0.03 accumulates values whose sum stays small, but token
    counts reach 10^10 within a day and float32 stops representing consecutive
    integers past 2^24. Losing count of tokens corrupts the x-axis of every
    scaling plot the run exists to produce.

    The rejected alternative was to append each step's metrics to a Python list
    and reduce at flush time. It avoids the arithmetic subtlety and it forces a
    synchronisation per step to get the value out of the tensor, which is the
    exact cost this class exists to avoid.

    Args:
        device: Device the accumulator lives on. Must match the device the
            training step produces metrics on.
        mesh: The mesh to reduce across, normally
            :func:`avgen.parallel.data_mesh` applied to the full mesh. ``None``
            means single-replica and skips the collective entirely.
    """

    __slots__ = ("_count", "_device", "_mesh", "_names", "_steps", "_totals")

    def __init__(
        self,
        device: torch.device | str = "cpu",
        *,
        mesh: DeviceMesh | None = None,
    ) -> None:
        self._device = torch.device(device)
        self._mesh = mesh
        # A host-side mirror of the step count, so `is_empty` can be answered
        # without reading the device buffer and synchronising.
        self._steps = 0
        # Field order is taken from the dataclass, so adding a metric to
        # StepMetrics needs no change here and cannot silently misalign the
        # buffer against the names.
        self._names = tuple(spec.name for spec in fields(StepMetrics))
        self._totals = torch.zeros(
            len(self._names), dtype=torch.float64, device=self._device
        )
        self._count = torch.zeros((), dtype=torch.float64, device=self._device)

    @property
    def device(self) -> torch.device:
        """Device the running sums live on."""
        return self._device

    @property
    def is_empty(self) -> bool:
        """Whether anything has been accumulated since the last flush.

        Reads a host-side mirror rather than the device counter, so asking is
        free and does not itself synchronise.
        """
        return self._steps == 0

    def update(self, metrics: StepMetrics, *, weight: float = 1.0) -> None:
        """Fold one step's metrics into the running sums. No synchronisation.

        Args:
            metrics: The step's detached scalar tensors.
            weight: Relative weight of this step, for the mean fields. Use the
                microbatch's token count when microbatches differ in size,
                otherwise the mean is a mean of means and over-weights the
                small ones.
        """
        values = torch.stack(
            [getattr(metrics, name).detach().to(torch.float64) for name in self._names]
        )
        # `to(self._device)` is a no-op when they already match, and a cheap
        # async copy when a caller accumulates CPU metrics on a CUDA buffer.
        self._totals += values.to(self._device) * weight
        self._count += weight
        self._steps += 1

    def reset(self) -> None:
        """Zero the running sums without allocating."""
        self._totals.zero_()
        self._count.zero_()
        self._steps = 0

    def flush(self, *, reset: bool = True) -> dict[str, float]:
        """Reduce across the mesh, transfer to host once, and return the means.

        This is the **only** synchronisation point in the metric path. Call it
        at log cadence; calling it every step reintroduces exactly the cost the
        class exists to remove.

        The reduction is a sum over ranks of (sum over steps), divided by a
        summed count, rather than a mean of per-rank means. Those differ
        whenever ranks accumulated different numbers of steps, which happens the
        moment one rank skips a microbatch for a non-finite loss.

        Args:
            reset: Whether to zero the accumulator afterwards. The normal case.

        Returns:
            Metric name to value: mean over steps and ranks for losses and
            gradient norms, total over steps and ranks for token counts and
            skip counters, plus ``"steps"`` — the number of steps folded in.
            Empty when nothing was accumulated.
        """
        # One buffer for the payload and the count so the collective sees a
        # single contiguous tensor.
        packed = torch.cat([self._totals, self._count.reshape(1)])
        reduced = all_reduce_sum(packed, self._mesh)
        # The single device-to-host transfer for the whole interval.
        host = reduced.to("cpu", dtype=torch.float64).tolist()
        count = host[-1]
        if count <= 0:
            if reset:
                self.reset()
            return {}
        result: dict[str, float] = {}
        for index, name in enumerate(self._names):
            total = host[index]
            result[name] = total / count if name in _MEAN_FIELDS else total
        result["steps"] = count / (self._mesh.size() if self._mesh is not None else 1)
        if reset:
            self.reset()
        return result


@dataclass(frozen=True, slots=True)
class ThroughputReport:
    """One interval's throughput accounting.

    Args:
        steps: Optimizer steps in the interval.
        elapsed_s: Wall-clock seconds the interval covered.
        step_time_s: Mean step time.
        step_time_p50_s: Median step time.
        step_time_p99_s: 99th-percentile step time.
        samples_per_s: Global samples per second.
        tokens_per_s: Global generative tokens per second.
        mfu: Model FLOPs utilisation, or ``None`` when unpriced.
        hfu: Hardware FLOPs utilisation, or ``None`` when unpriced.
    """

    steps: int
    elapsed_s: float
    step_time_s: float
    step_time_p50_s: float
    step_time_p99_s: float
    samples_per_s: float
    tokens_per_s: float
    mfu: float | None
    hfu: float | None

    def to_mapping(self) -> dict[str, float]:
        """Return a flat mapping suitable for a logger, dropping unset ratios."""
        payload = {
            "throughput/steps": float(self.steps),
            "throughput/elapsed_s": self.elapsed_s,
            "throughput/step_time_s": self.step_time_s,
            "throughput/step_time_p50_s": self.step_time_p50_s,
            "throughput/step_time_p99_s": self.step_time_p99_s,
            "throughput/samples_per_s": self.samples_per_s,
            "throughput/tokens_per_s": self.tokens_per_s,
        }
        if self.mfu is not None:
            payload["throughput/mfu"] = self.mfu
        if self.hfu is not None:
            payload["throughput/hfu"] = self.hfu
        return payload


class ThroughputMeter:
    """Step timing, token rate, and FLOPs utilisation over a log interval.

    **p99, not just the mean.** A mean step time hides the failure mode that
    actually costs throughput at scale: most steps are fast and a few are
    catastrophically slow, because one rank waited on a cold data shard or a
    checkpoint upload saturated the fabric. Every rank waits for the slowest, so
    a p99 twice the median means the job is running at roughly half the speed
    the mean suggests. The gap between p50 and p99 is the single most useful
    diagnostic number in a training log.

    **MFU and HFU are different numbers and both are needed.** MFU prices the
    FLOPs the *model* mathematically requires; HFU prices the FLOPs the
    *hardware* actually executed, which is larger whenever activation
    checkpointing recomputes a forward pass. A run at 35% MFU and 50% HFU is
    spending a third of its compute on recomputation — that is the signal to
    move from full to selective checkpointing, and it is invisible if you track
    only one of the two.

    Timing uses ``time.perf_counter`` on the host. That measures the *training
    loop*, which is what the user cares about, and it is correct as long as the
    loop is not measured across an unsynchronised device boundary — which is
    why the meter is marked at the top of a step, after the previous step's
    optimizer step has already forced ordering.

    Args:
        peak_flops_per_s: Device peak FLOP/s for the training dtype, summed
            across the whole job, or ``None`` to skip MFU/HFU. Use the dense
            (non-sparse) number from the vendor spec sheet; the sparsity figure
            is unreachable for dense training and inflates MFU by 2x.
        flops_per_step: Model FLOPs for one global optimizer step, from
            :mod:`avgen.simulate.compute`. ``None`` skips MFU.
        hardware_flops_per_step: Hardware FLOPs for one global step, including
            activation recomputation. ``None`` skips HFU.
        window: Number of recent step times kept for percentiles.

    Raises:
        ValueError: If ``window`` is smaller than one.
    """

    __slots__ = (
        "_flops_per_step",
        "_hardware_flops_per_step",
        "_interval_start",
        "_last_mark",
        "_peak_flops_per_s",
        "_samples",
        "_step_times",
        "_steps",
        "_tokens",
        "_window",
    )

    def __init__(
        self,
        *,
        peak_flops_per_s: float | None = None,
        flops_per_step: float | None = None,
        hardware_flops_per_step: float | None = None,
        window: int = 1000,
    ) -> None:
        if window < 1:
            raise ValueError(f"window must be >= 1; got {window}")
        self._peak_flops_per_s = peak_flops_per_s
        self._flops_per_step = flops_per_step
        self._hardware_flops_per_step = hardware_flops_per_step
        self._window = window
        self._step_times: list[float] = []
        self._steps = 0
        self._samples = 0
        self._tokens = 0
        now = time.perf_counter()
        self._interval_start = now
        self._last_mark = now

    def start(self) -> None:
        """Reset the clock without discarding the percentile window.

        Call after a warm-up phase or after a resume: the first few steps of a
        run include compilation, autotuning, and a cold page cache, and folding
        them into the steady-state rate makes every later comparison wrong.
        """
        now = time.perf_counter()
        self._interval_start = now
        self._last_mark = now
        self._steps = 0
        self._samples = 0
        self._tokens = 0

    def step(self, *, samples: int = 0, tokens: int = 0) -> float:
        """Record one completed optimizer step.

        Args:
            samples: Global samples consumed by this step, across all data-
                parallel ranks — not this rank's share. Reporting a local count
                understates throughput by ``dp_size``.
            tokens: Global generative tokens consumed by this step.

        Returns:
            The measured duration of the step, in seconds.
        """
        now = time.perf_counter()
        duration = now - self._last_mark
        self._last_mark = now
        self._step_times.append(duration)
        if len(self._step_times) > self._window:
            # Bounded so a week-long run does not accumulate a list of millions
            # of floats purely to compute a percentile over the recent past.
            del self._step_times[: len(self._step_times) - self._window]
        self._steps += 1
        self._samples += samples
        self._tokens += tokens
        return duration

    @staticmethod
    def _percentile(sorted_values: list[float], fraction: float) -> float:
        if not sorted_values:
            return 0.0
        index = min(
            len(sorted_values) - 1,
            max(0, round(fraction * (len(sorted_values) - 1))),
        )
        return sorted_values[index]

    def report(self, *, reset: bool = True) -> ThroughputReport:
        """Summarise the interval since the last report.

        Args:
            reset: Whether to start a new interval. The percentile window is
                kept either way, since it is deliberately a rolling view.

        Returns:
            The interval's throughput accounting.
        """
        now = time.perf_counter()
        elapsed = max(now - self._interval_start, 1e-9)
        ordered = sorted(self._step_times)
        mean_step = elapsed / self._steps if self._steps else 0.0

        mfu: float | None = None
        hfu: float | None = None
        if self._peak_flops_per_s and self._steps:
            achieved_steps_per_s = self._steps / elapsed
            if self._flops_per_step:
                mfu = (
                    self._flops_per_step * achieved_steps_per_s
                ) / self._peak_flops_per_s
            if self._hardware_flops_per_step:
                hfu = (
                    self._hardware_flops_per_step * achieved_steps_per_s
                ) / self._peak_flops_per_s

        report = ThroughputReport(
            steps=self._steps,
            elapsed_s=elapsed,
            step_time_s=mean_step,
            step_time_p50_s=self._percentile(ordered, 0.50),
            step_time_p99_s=self._percentile(ordered, 0.99),
            samples_per_s=self._samples / elapsed,
            tokens_per_s=self._tokens / elapsed,
            mfu=mfu,
            hfu=hfu,
        )
        if reset:
            self._interval_start = now
            self._steps = 0
            self._samples = 0
            self._tokens = 0
        return report


@dataclass(frozen=True, slots=True)
class MemoryReport:
    """A snapshot of one device's allocator state.

    Args:
        allocated_gib: Live tensor bytes right now.
        peak_allocated_gib: High-water mark of live tensor bytes.
        reserved_gib: Bytes the caching allocator holds from the driver.
        peak_reserved_gib: High-water mark of reserved bytes.
        capacity_gib: Total device memory.
        fragmentation: ``1 - peak_allocated / peak_reserved``. The share of
            reserved memory that is held but unusable.
        headroom: ``1 - peak_reserved / capacity``.
        num_alloc_retries: Times the allocator had to free cached blocks to
            satisfy a request. Nonzero means it is already struggling.
        num_ooms: Out-of-memory events survived.
    """

    allocated_gib: float
    peak_allocated_gib: float
    reserved_gib: float
    peak_reserved_gib: float
    capacity_gib: float
    fragmentation: float
    headroom: float
    num_alloc_retries: int
    num_ooms: int

    def to_mapping(self) -> dict[str, float]:
        """Return a flat mapping suitable for a logger."""
        return {
            f"memory/{spec.name}": float(getattr(self, spec.name))
            for spec in fields(self)
        }


class MemoryReporter:
    """Peak memory, fragmentation, and the warning that precedes an OOM.

    The number that matters is **reserved**, not allocated. The caching
    allocator takes memory from the driver and keeps it; a run whose allocated
    peak is 40 GiB but whose reserved peak is 78 GiB on an 80 GiB device is one
    unlucky allocation away from dying, and its *allocated* figure looks
    perfectly healthy. The gap between the two is fragmentation: memory that is
    reserved, free, and in blocks too small for the tensor that needs it.

    Two things make that gap grow during a run rather than at the start, which
    is why a job can train for six hours and then OOM on a step no different
    from the first: variable sequence lengths (a bucketed video loader allocates
    a different shape every batch) and any allocation pattern that interleaves
    long- and short-lived tensors. Both are normal for video training, so this
    reporter's warning threshold is the intended early signal, not a formality.

    ``num_alloc_retries`` is the sharpest of these signals. It counts the times
    the allocator had to release cached blocks back to the driver to satisfy a
    request — an expensive, synchronising operation. It is zero on a healthy
    run and becomes nonzero *before* the first OOM, usually by minutes.

    Args:
        device: Device to report on. Defaults to the current CUDA device.
        headroom_warning: Warn when free capacity drops below this fraction.
            0.10 is a deliberate choice: the transient spike of a checkpoint
            gather or an evaluation pass is comfortably larger than 5%.
        fragmentation_warning: Warn when this fraction of reserved memory is
            unusable.

    Raises:
        ValueError: If a threshold is outside ``[0, 1]``.
    """

    __slots__ = ("_device", "_fragmentation_warning", "_headroom_warning", "_warned")

    def __init__(
        self,
        device: torch.device | str | None = None,
        *,
        headroom_warning: float = 0.10,
        fragmentation_warning: float = 0.20,
    ) -> None:
        for name, value in (
            ("headroom_warning", headroom_warning),
            ("fragmentation_warning", fragmentation_warning),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]; got {value}")
        # Resolving the device lazily keeps this constructible on a CPU-only
        # machine, which CONTRACTS requires of every module.
        self._device = torch.device(device) if device is not None else None
        self._headroom_warning = headroom_warning
        self._fragmentation_warning = fragmentation_warning
        self._warned = False

    @property
    def available(self) -> bool:
        """Whether there is a CUDA allocator to report on."""
        return bool(torch.cuda.is_available())

    def reset_peaks(self) -> None:
        """Reset the high-water marks so the next interval is measured alone.

        Without this, the peak reported at step 50000 is still the peak from
        step 3, when compilation and autotuning briefly allocated far more than
        steady state ever will.
        """
        if self.available:
            torch.cuda.reset_peak_memory_stats(self._device)

    def report(self) -> MemoryReport | None:
        """Read the allocator, derive fragmentation and headroom, warn if tight.

        Returns:
            The snapshot, or ``None`` on a device with no CUDA allocator.
        """
        if not self.available:
            return None
        stats = torch.cuda.memory_stats(self._device)
        capacity = float(torch.cuda.get_device_properties(self._device).total_memory)
        peak_allocated = float(stats.get("allocated_bytes.all.peak", 0))
        peak_reserved = float(stats.get("reserved_bytes.all.peak", 0))
        fragmentation = (
            1.0 - peak_allocated / peak_reserved if peak_reserved > 0 else 0.0
        )
        headroom = 1.0 - peak_reserved / capacity if capacity > 0 else 0.0
        report = MemoryReport(
            allocated_gib=float(stats.get("allocated_bytes.all.current", 0)) / 1024**3,
            peak_allocated_gib=peak_allocated / 1024**3,
            reserved_gib=float(stats.get("reserved_bytes.all.current", 0)) / 1024**3,
            peak_reserved_gib=peak_reserved / 1024**3,
            capacity_gib=capacity / 1024**3,
            fragmentation=fragmentation,
            headroom=headroom,
            num_alloc_retries=int(stats.get("num_alloc_retries", 0)),
            num_ooms=int(stats.get("num_ooms", 0)),
        )
        self._maybe_warn(report)
        return report

    def _maybe_warn(self, report: MemoryReport) -> None:
        """Emit the pre-OOM warning at most once per reset."""
        if self._warned:
            return
        if report.headroom < self._headroom_warning:
            self._warned = True
            _LOG.warning(
                "memory headroom %.1f%% below the %.1f%% threshold "
                "(peak reserved %.1f GiB of %.1f GiB, %d alloc retries). "
                "A longer sequence bucket or an eval pass will OOM.",
                report.headroom * 100,
                self._headroom_warning * 100,
                report.peak_reserved_gib,
                report.capacity_gib,
                report.num_alloc_retries,
            )
        elif report.fragmentation > self._fragmentation_warning:
            self._warned = True
            _LOG.warning(
                "%.1f%% of reserved memory is fragmented "
                "(peak allocated %.1f GiB, peak reserved %.1f GiB). "
                "Capture a snapshot with avgen.telemetry.memory_snapshot to see "
                "which allocation pattern is responsible.",
                report.fragmentation * 100,
                report.peak_allocated_gib,
                report.peak_reserved_gib,
            )

    def clear_warning(self) -> None:
        """Re-arm the one-shot warning, normally after a configuration change."""
        self._warned = False

    def peak_across(self, mesh: DeviceMesh | None) -> MemoryReport | None:
        """Return the worst rank's report, reduced across a mesh.

        Memory is the one metric where the mean is useless: the job dies when
        *one* rank runs out. Pipeline stages and the ranks holding an uneven
        FSDP shard routinely differ by tens of percent.

        Args:
            mesh: Mesh to reduce across, or ``None`` for a local report.

        Returns:
            The elementwise-maximum report, or ``None`` without CUDA.
        """
        local = self.report()
        if local is None or mesh is None:
            return local
        names = tuple(spec.name for spec in fields(MemoryReport))
        packed = torch.tensor(
            [float(getattr(local, name)) for name in names],
            dtype=torch.float64,
            device=self._device or torch.cuda.current_device(),
        )
        worst = all_reduce_max(packed, mesh).to("cpu").tolist()
        values: dict[str, float] = dict(zip(names, worst, strict=True))
        return MemoryReport(
            allocated_gib=values["allocated_gib"],
            peak_allocated_gib=values["peak_allocated_gib"],
            reserved_gib=values["reserved_gib"],
            peak_reserved_gib=values["peak_reserved_gib"],
            capacity_gib=values["capacity_gib"],
            fragmentation=values["fragmentation"],
            headroom=values["headroom"],
            num_alloc_retries=int(values["num_alloc_retries"]),
            num_ooms=int(values["num_ooms"]),
        )


def merge_metrics(
    *sources: Mapping[str, float] | Iterable[tuple[str, float]],
) -> dict[str, float]:
    """Merge metric mappings into one payload for a single logger call.

    One call per step keeps the JSONL stream to one object per step, which is
    what makes it trivially loadable as a dataframe. Later sources win on a key
    collision.

    Args:
        *sources: Mappings or key/value iterables to merge.

    Returns:
        The merged mapping.
    """
    merged: dict[str, float] = {}
    for source in sources:
        items = source.items() if isinstance(source, Mapping) else source
        for key, value in items:
            merged[key] = float(value)
    return merged
