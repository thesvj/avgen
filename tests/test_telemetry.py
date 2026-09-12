"""Tests for the telemetry subsystem.

Telemetry that is naive about scale is worse than no telemetry at all, and the
two rules that follow from that are the two things asserted hardest here.

**One rank logs.** At 1024 ranks an ungated logger is 1024x the IO and a file
no tool can read. The gate lives in the base class, so the tests point a
``RANK=3`` environment at a rank-0 logger and assert that nothing at all
reaches the disk.

**One synchronisation per log interval, not per step.** A single ``.item()`` in
the training step collapses the host/device pipeline every step for numbers
nobody reads until the interval ends. :class:`MetricAccumulator` is therefore
tested by *counting* collectives and host transfers, not only by checking the
arithmetic.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import logging
import math
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import torch

import avgen.telemetry.metrics as metrics_module
from avgen.core.metrics import StepMetrics
from avgen.telemetry import (
    ConsoleLogger,
    JSONLLogger,
    MemoryReporter,
    MetricAccumulator,
    MultiLogger,
    NoOpLogger,
    RankGatedLogger,
    ThroughputMeter,
    ThroughputReport,
    build_logger,
    flight_recorder_dump,
    memory_snapshot,
    merge_metrics,
    profile_steps,
    should_profile_rank,
)


def importlib_available(module: str) -> bool:
    """Whether an optional backend is installed, without importing it."""
    return importlib.util.find_spec(module) is not None


def step_metrics(
    *,
    loss: float = 1.0,
    video_loss: float = 1.0,
    audio_loss: float = 0.0,
    video_tokens: int = 1000,
    audio_tokens: int = 0,
    grad_norm: float = 0.5,
    nonfinite: bool = False,
    skipped: bool = False,
) -> StepMetrics:
    """Build one step's detached scalar metrics on the CPU."""
    return StepMetrics(
        loss=torch.tensor(loss, dtype=torch.float32),
        video_loss=torch.tensor(video_loss, dtype=torch.float32),
        audio_loss=torch.tensor(audio_loss, dtype=torch.float32),
        valid_video_tokens=torch.tensor(video_tokens, dtype=torch.int64),
        valid_audio_tokens=torch.tensor(audio_tokens, dtype=torch.int64),
        grad_norm=torch.tensor(grad_norm, dtype=torch.float32),
        nonfinite=torch.tensor(nonfinite, dtype=torch.bool),
        skipped=torch.tensor(skipped, dtype=torch.bool),
    )


@pytest.fixture
def fake_clock(monkeypatch: pytest.MonkeyPatch) -> Callable[[list[float]], None]:
    """Replace ``time.perf_counter`` with a scripted sequence of readings.

    Throughput arithmetic is only checkable against a hand-computed answer if
    the clock is deterministic; a real clock turns every assertion into a
    tolerance nobody can justify.
    """

    def install(readings: list[float]) -> None:
        supply: Iterator[float] = iter(readings)
        last = [readings[-1]]

        def perf_counter() -> float:
            with contextlib.suppress(StopIteration):
                last[0] = next(supply)
            return last[0]

        monkeypatch.setattr(time, "perf_counter", perf_counter)

    return install


@pytest.fixture
def sync_counter(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Count device collectives and device-to-host transfers.

    ``tolist`` is the transfer the accumulator makes at flush time; counting it
    is how "syncs once per interval" becomes a testable claim rather than a
    comment.
    """
    counts = {"collective": 0, "transfer": 0}
    real_all_reduce = metrics_module.all_reduce_sum
    real_tolist = torch.Tensor.tolist

    def counting_all_reduce(value: torch.Tensor, mesh: Any) -> torch.Tensor:
        counts["collective"] += 1
        return real_all_reduce(value, mesh)

    def counting_tolist(self: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        counts["transfer"] += 1
        return real_tolist(self, *args, **kwargs)

    monkeypatch.setattr(metrics_module, "all_reduce_sum", counting_all_reduce)
    monkeypatch.setattr(torch.Tensor, "tolist", counting_tolist)
    return counts


class TestMetricAccumulator:
    """Sums on device, syncs once, and gets the averaging right.

    The rejected alternative — appending each step's metrics to a Python list —
    forces a synchronisation per step to get the value out of the tensor, which
    is the exact cost this class exists to avoid.
    """

    def test_losses_are_meaned_and_token_counts_are_summed(self) -> None:
        accumulator = MetricAccumulator()
        for index in range(4):
            accumulator.update(step_metrics(loss=float(index), video_tokens=100))
        flushed = accumulator.flush()
        assert flushed["loss"] == pytest.approx(1.5)  # mean of 0, 1, 2, 3
        assert flushed["valid_video_tokens"] == pytest.approx(400.0)  # total
        assert flushed["steps"] == pytest.approx(4.0)

    def test_every_step_metrics_field_is_reported(self) -> None:
        # The field order is taken from the dataclass, so a new metric cannot
        # silently misalign the buffer against the names.
        accumulator = MetricAccumulator()
        accumulator.update(step_metrics())
        flushed = accumulator.flush()
        for name in (
            "loss",
            "video_loss",
            "audio_loss",
            "valid_video_tokens",
            "valid_audio_tokens",
            "grad_norm",
            "nonfinite",
            "skipped",
        ):
            assert name in flushed

    def test_boolean_counters_accumulate_as_totals(self) -> None:
        # A handful of skipped steps is survivable; a rising rate is the
        # earliest signal a run is diverging, so it must be a count.
        accumulator = MetricAccumulator()
        for index in range(10):
            accumulator.update(step_metrics(skipped=index % 5 == 0))
        assert accumulator.flush()["skipped"] == pytest.approx(2.0)

    def test_a_weighted_mean_over_many_steps(self) -> None:
        # Weighting by token count is what stops a mean of means over-weighting
        # a small microbatch.
        accumulator = MetricAccumulator()
        accumulator.update(step_metrics(loss=1.0), weight=2.0)
        accumulator.update(step_metrics(loss=3.0), weight=6.0)
        assert accumulator.flush()["loss"] == pytest.approx((2 * 1.0 + 6 * 3.0) / 8)

    def test_a_long_interval_stays_exact_in_float64(self) -> None:
        # Token counts reach 10^10 within a day, and float32 stops representing
        # consecutive integers past 2^24 — losing count of tokens corrupts the
        # x-axis of every scaling plot the run exists to produce.
        accumulator = MetricAccumulator()
        for _ in range(1000):
            accumulator.update(step_metrics(loss=0.03, video_tokens=20_000_000))
        flushed = accumulator.flush()
        # Exactly, not approximately: fp32 would have lost the low bits of this
        # total long before a day of training was over.
        assert flushed["valid_video_tokens"] == 2e10
        single = float(torch.tensor(0.03, dtype=torch.float32))
        assert flushed["loss"] == pytest.approx(single, rel=1e-12)

    def test_an_empty_flush_is_safe(self) -> None:
        # A log interval in which every step was skipped must not divide by zero.
        assert MetricAccumulator().flush() == {}

    def test_is_empty_before_and_after(self) -> None:
        accumulator = MetricAccumulator()
        assert accumulator.is_empty
        accumulator.update(step_metrics())
        assert not accumulator.is_empty
        accumulator.flush()
        assert accumulator.is_empty

    def test_flush_without_reset_keeps_the_interval(self) -> None:
        accumulator = MetricAccumulator()
        accumulator.update(step_metrics(loss=2.0))
        assert accumulator.flush(reset=False)["loss"] == pytest.approx(2.0)
        assert not accumulator.is_empty
        assert accumulator.flush()["loss"] == pytest.approx(2.0)

    def test_reset_zeroes_without_flushing(self) -> None:
        accumulator = MetricAccumulator()
        accumulator.update(step_metrics(loss=5.0))
        accumulator.reset()
        assert accumulator.is_empty
        assert accumulator.flush() == {}

    def test_the_device_is_reported(self) -> None:
        assert MetricAccumulator("cpu").device == torch.device("cpu")

    def test_updates_do_not_synchronise_and_flush_does_so_once(
        self, sync_counter: dict[str, int]
    ) -> None:
        # This is the entire reason the class exists. At a log interval of 50
        # steps it is two synchronisations per 50 steps instead of 400.
        accumulator = MetricAccumulator()
        for _ in range(50):
            accumulator.update(step_metrics())
        assert accumulator.is_empty is False
        assert sync_counter == {"collective": 0, "transfer": 0}

        accumulator.flush()
        assert sync_counter["collective"] == 1
        assert sync_counter["transfer"] == 1

    def test_an_empty_flush_still_costs_only_one_collective(
        self, sync_counter: dict[str, int]
    ) -> None:
        MetricAccumulator().flush()
        assert sync_counter["collective"] == 1

    def test_is_empty_never_reads_the_device_buffer(
        self, sync_counter: dict[str, int]
    ) -> None:
        accumulator = MetricAccumulator()
        accumulator.update(step_metrics())
        for _ in range(100):
            assert accumulator.is_empty is False
        assert sync_counter["transfer"] == 0


class TestMergeMetrics:
    """One logger call per step keeps the JSONL stream one object per step."""

    def test_mappings_are_merged_and_coerced_to_float(self) -> None:
        merged = merge_metrics({"a": 1}, {"b": 2.5})
        assert merged == {"a": 1.0, "b": 2.5}
        assert all(isinstance(value, float) for value in merged.values())

    def test_later_sources_win_a_collision(self) -> None:
        assert merge_metrics({"a": 1.0}, {"a": 2.0}) == {"a": 2.0}

    def test_key_value_iterables_are_accepted(self) -> None:
        assert merge_metrics([("a", 1.0)], {"b": 2.0}) == {"a": 1.0, "b": 2.0}

    def test_no_sources_is_an_empty_payload(self) -> None:
        assert merge_metrics() == {}


class TestThroughputMeter:
    """Hand-computed rates against a scripted clock, and p50 next to p99.

    The gap between p50 and p99 is the single most useful diagnostic number in
    a training log: every rank waits for the slowest, so a p99 twice the median
    means the job runs at roughly half the speed the mean suggests.
    """

    def test_rates_and_utilisation_against_a_known_clock(
        self, fake_clock: Callable[[list[float]], None]
    ) -> None:
        # Construction reads 0.0; four steps land at 1, 2, 3, 4; the report
        # reads 4.0. So elapsed = 4 s over 4 steps of 1 s each.
        fake_clock([0.0, 1.0, 2.0, 3.0, 4.0, 4.0])
        meter = ThroughputMeter(
            peak_flops_per_s=1000.0,
            flops_per_step=100.0,
            hardware_flops_per_step=150.0,
        )
        for _ in range(4):
            meter.step(samples=8, tokens=2000)
        report = meter.report()

        assert report.steps == 4
        assert report.elapsed_s == pytest.approx(4.0)
        assert report.step_time_s == pytest.approx(1.0)
        assert report.samples_per_s == pytest.approx(32 / 4.0)
        assert report.tokens_per_s == pytest.approx(8000 / 4.0)
        # 4 steps / 4 s = 1 step/s; 100 model FLOPs per step against a 1000
        # FLOP/s peak is 10% MFU, and 150 hardware FLOPs is 15% HFU.
        assert report.mfu == pytest.approx(0.10)
        assert report.hfu == pytest.approx(0.15)

    def test_the_hfu_mfu_gap_prices_recomputation(
        self, fake_clock: Callable[[list[float]], None]
    ) -> None:
        # A run at 35% MFU and 50% HFU spends a third of its compute on
        # activation recomputation; that is invisible if you track only one.
        fake_clock([0.0, 1.0, 1.0])
        meter = ThroughputMeter(
            peak_flops_per_s=100.0, flops_per_step=35.0, hardware_flops_per_step=50.0
        )
        meter.step()
        report = meter.report()
        assert report.mfu == pytest.approx(0.35)
        assert report.hfu == pytest.approx(0.50)

    def test_step_returns_the_measured_duration(
        self, fake_clock: Callable[[list[float]], None]
    ) -> None:
        fake_clock([0.0, 0.25, 1.25])
        meter = ThroughputMeter()
        assert meter.step() == pytest.approx(0.25)
        assert meter.step() == pytest.approx(1.0)

    def test_percentiles_over_a_hundred_steps(
        self, fake_clock: Callable[[list[float]], None]
    ) -> None:
        # Step i takes i+1 seconds, so the sorted durations are 1..100. The
        # percentile index is round(fraction * (n - 1)): 50 and 98.
        readings = [0.0]
        cursor = 0.0
        for index in range(100):
            cursor += index + 1
            readings.append(cursor)
        readings.append(cursor)
        fake_clock(readings)

        meter = ThroughputMeter()
        for _ in range(100):
            meter.step()
        report = meter.report()
        assert report.step_time_p50_s == pytest.approx(51.0)
        assert report.step_time_p99_s == pytest.approx(99.0)
        assert report.step_time_p99_s > report.step_time_p50_s

    def test_a_straggler_tail_moves_p99_and_not_p50(
        self, fake_clock: Callable[[list[float]], None]
    ) -> None:
        # The failure mode a mean hides: most steps fast, a few catastrophically
        # slow because one rank waited on a cold shard. Every rank waits for the
        # slowest, so this is the job's real speed.
        readings = [0.0]
        cursor = 0.0
        for index in range(100):
            cursor += 100.0 if index >= 95 else 1.0
            readings.append(cursor)
        readings.append(cursor)
        fake_clock(readings)

        meter = ThroughputMeter()
        for _ in range(100):
            meter.step()
        report = meter.report()
        assert report.step_time_p50_s == pytest.approx(1.0)
        assert report.step_time_p99_s == pytest.approx(100.0)
        assert report.step_time_p99_s > 50 * report.step_time_p50_s

    def test_the_percentile_window_is_bounded(
        self, fake_clock: Callable[[list[float]], None]
    ) -> None:
        # A week-long run must not accumulate a list of millions of floats
        # purely to compute a percentile over the recent past.
        fake_clock([float(index) for index in range(60)])
        meter = ThroughputMeter(window=8)
        for _ in range(50):
            meter.step()
        assert len(meter._step_times) == 8

    def test_a_window_below_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"window must be >= 1"):
            ThroughputMeter(window=0)

    def test_start_discards_the_warm_up_interval(
        self, fake_clock: Callable[[list[float]], None]
    ) -> None:
        # The first steps of a run include compilation, autotuning, and a cold
        # page cache; folding them in makes every later comparison wrong.
        fake_clock([0.0, 10.0, 10.0, 11.0, 11.0])
        meter = ThroughputMeter()
        meter.step(samples=4)
        meter.start()
        meter.step(samples=4)
        report = meter.report()
        assert report.steps == 1
        assert report.elapsed_s == pytest.approx(1.0)
        assert report.samples_per_s == pytest.approx(4.0)

    def test_utilisation_is_none_when_unpriced(
        self, fake_clock: Callable[[list[float]], None]
    ) -> None:
        fake_clock([0.0, 1.0, 1.0])
        meter = ThroughputMeter()
        meter.step()
        report = meter.report()
        assert report.mfu is None and report.hfu is None
        assert "throughput/mfu" not in report.to_mapping()

    def test_mfu_without_hfu_when_only_model_flops_are_known(
        self, fake_clock: Callable[[list[float]], None]
    ) -> None:
        fake_clock([0.0, 1.0, 1.0])
        meter = ThroughputMeter(peak_flops_per_s=100.0, flops_per_step=25.0)
        meter.step()
        report = meter.report()
        assert report.mfu == pytest.approx(0.25)
        assert report.hfu is None

    def test_an_interval_with_no_steps_reports_zeroes(
        self, fake_clock: Callable[[list[float]], None]
    ) -> None:
        fake_clock([0.0, 1.0])
        report = ThroughputMeter(peak_flops_per_s=1.0, flops_per_step=1.0).report()
        assert report.steps == 0
        assert report.step_time_s == 0.0
        assert report.mfu is None

    def test_report_without_reset_keeps_accumulating(
        self, fake_clock: Callable[[list[float]], None]
    ) -> None:
        fake_clock([0.0, 1.0, 1.0, 2.0, 2.0])
        meter = ThroughputMeter()
        meter.step()
        assert meter.report(reset=False).steps == 1
        meter.step()
        assert meter.report().steps == 2

    def test_the_mapping_is_flat_and_namespaced(self) -> None:
        mapping = ThroughputReport(
            steps=2,
            elapsed_s=1.0,
            step_time_s=0.5,
            step_time_p50_s=0.5,
            step_time_p99_s=0.9,
            samples_per_s=4.0,
            tokens_per_s=8.0,
            mfu=0.4,
            hfu=0.6,
        ).to_mapping()
        assert mapping["throughput/steps"] == 2.0
        assert mapping["throughput/mfu"] == 0.4
        assert all(key.startswith("throughput/") for key in mapping)
        assert all(isinstance(value, float) for value in mapping.values())


class TestJSONLLogger:
    """JSONL is the format of record because it is the only one that survives.

    It needs no service, no account and no human; it appends atomically and
    tolerates truncation at the end. Everything else is a convenience layered
    on top of it.
    """

    def test_every_line_parses(self, tmp_path: Path) -> None:
        logger = JSONLLogger(tmp_path / "metrics.jsonl")
        for step in range(3):
            logger.log_metrics({"loss": 1.0 / (step + 1)}, step)
        logger.close()

        lines = logger.path.read_text(encoding="utf-8").splitlines()
        records = [json.loads(line) for line in lines]
        assert [record["step"] for record in records] == [0, 1, 2]
        assert all(record["type"] == "metrics" for record in records)
        assert records[0]["loss"] == 1.0

    def test_parent_directories_are_created(self, tmp_path: Path) -> None:
        logger = JSONLLogger(tmp_path / "runs" / "exp" / "metrics.jsonl")
        logger.log_metrics({"a": 1.0}, 0)
        logger.close()
        assert (tmp_path / "runs" / "exp" / "metrics.jsonl").is_file()

    def test_a_resumed_run_appends_rather_than_truncating(self, tmp_path: Path) -> None:
        # The step field is what a reader uses to drop the overlap when a job
        # is preempted mid-interval; truncating would throw the history away.
        path = tmp_path / "metrics.jsonl"
        first = JSONLLogger(path)
        first.log_metrics({"loss": 1.0}, 0)
        first.close()

        second = JSONLLogger(path)
        second.log_metrics({"loss": 0.5}, 1)
        second.close()

        records = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        ]
        assert [record["step"] for record in records] == [0, 1]

    def test_each_line_is_flushed_immediately(self, tmp_path: Path) -> None:
        # Buffered output is lost when a job is killed, and a job being killed
        # is precisely when the last few lines matter most.
        logger = JSONLLogger(tmp_path / "metrics.jsonl")
        logger.log_metrics({"loss": 1.0}, 0)
        assert logger.path.read_text(encoding="utf-8").strip()
        logger.close()

    def test_config_and_artifact_records_are_typed(self, tmp_path: Path) -> None:
        logger = JSONLLogger(tmp_path / "metrics.jsonl")
        logger.log_config({"train": {"lr": 1e-4}})
        logger.log_artifact(tmp_path / "sample.mp4")
        logger.close()

        records = [
            json.loads(line)
            for line in logger.path.read_text(encoding="utf-8").splitlines()
        ]
        assert records[0]["type"] == "config"
        assert records[0]["config"]["train"]["lr"] == 1e-4
        assert records[1]["type"] == "artifact"
        assert records[1]["path"].endswith("sample.mp4")

    def test_an_unserialisable_value_never_kills_the_run(self, tmp_path: Path) -> None:
        # Losing a training run to the logger is never acceptable.
        logger = JSONLLogger(tmp_path / "metrics.jsonl")
        logger.log_config({"dtype": torch.float32, "path": tmp_path})
        logger.close()
        record = json.loads(logger.path.read_text(encoding="utf-8").splitlines()[0])
        assert record["config"]["dtype"] == "torch.float32"

    def test_it_works_as_a_context_manager(self, tmp_path: Path) -> None:
        with JSONLLogger(tmp_path / "metrics.jsonl") as logger:
            logger.log_metrics({"a": 1.0}, 0)
        assert (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").strip()

    def test_closing_twice_is_safe(self, tmp_path: Path) -> None:
        logger = JSONLLogger(tmp_path / "metrics.jsonl")
        logger.close()
        logger.close()


class TestRankGating:
    """Exactly one rank writes. At 1024 ranks this is the whole ballgame.

    A thousand ranks writing the same line produces a thousand times the IO, a
    file no tool can read, and — with a network backend — a thousand sockets to
    a service that will rate-limit the job into a stall.
    """

    def test_a_non_logging_rank_writes_nothing_at_all(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RANK", "3")
        path = tmp_path / "metrics.jsonl"
        logger = JSONLLogger(path, rank=0)
        assert logger.enabled is False
        logger.log_metrics({"loss": 1.0}, 0)
        logger.log_config({"a": 1})
        logger.log_artifact(tmp_path / "x.mp4")
        logger.close()
        # Not merely empty: the file is never even opened.
        assert not path.exists()

    def test_the_logging_rank_writes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RANK", "0")
        logger = JSONLLogger(tmp_path / "metrics.jsonl", rank=0)
        assert logger.enabled is True
        logger.log_metrics({"loss": 1.0}, 0)
        logger.close()
        assert (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").strip()

    def test_the_logging_rank_is_configurable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RANK", "3")
        logger = JSONLLogger(tmp_path / "metrics.jsonl", rank=3)
        assert logger.enabled is True
        logger.close()

    def test_enabled_false_overrides_the_rank(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # build_logger uses this to build a structurally identical logger tree
        # on every rank, so trainer control flow never differs by rank.
        monkeypatch.setenv("RANK", "0")
        logger = JSONLLogger(tmp_path / "metrics.jsonl", rank=0, enabled=False)
        assert logger.enabled is False
        assert not (tmp_path / "metrics.jsonl").exists()

    def test_a_console_logger_on_another_rank_prints_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RANK", "7")
        stream = io.StringIO()
        logger = ConsoleLogger(stream=stream, rank=0)
        logger.log_metrics({"loss": 1.0}, 0)
        assert stream.getvalue() == ""

    @pytest.mark.parametrize("raw", ["", "not-a-number", "3.5"])
    def test_a_malformed_rank_variable_defaults_to_zero(
        self, raw: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A launcher that exports something odd must not silence rank 0.
        monkeypatch.setenv("RANK", raw)
        assert RankGatedLogger(rank=0).enabled is True

    def test_no_rank_variable_at_all_means_rank_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("RANK", raising=False)
        assert RankGatedLogger(rank=0).enabled is True
        assert RankGatedLogger(rank=1).enabled is False


class TestConsoleLogger:
    """A fixed-width table for a human watching a job start."""

    def test_a_header_precedes_the_first_row(self) -> None:
        stream = io.StringIO()
        logger = ConsoleLogger(stream=stream, keys=("loss", "grad_norm"))
        logger.log_metrics({"loss": 0.5, "grad_norm": 1.0}, 100)
        lines = stream.getvalue().splitlines()
        assert "loss" in lines[0] and "grad_norm" in lines[0]
        assert set(lines[1]) == {"-"}
        assert lines[2].split()[0] == "100"

    def test_the_header_is_reprinted_periodically(self) -> None:
        # The reader has scrolled past the first one.
        stream = io.StringIO()
        logger = ConsoleLogger(stream=stream, keys=("loss",), header_every=2)
        for step in range(4):
            logger.log_metrics({"loss": 1.0}, step)
        assert stream.getvalue().count("step") == 2

    def test_the_column_set_is_fixed_by_the_first_call(self) -> None:
        # A table whose columns move is unreadable at a glance, which is the
        # only thing a console log is for.
        stream = io.StringIO()
        logger = ConsoleLogger(stream=stream)
        logger.log_metrics({"loss": 1.0}, 0)
        logger.log_metrics({"loss": 1.0, "brand_new": 2.0}, 1)
        assert "brand_new" not in stream.getvalue()

    def test_an_absent_key_renders_as_a_dash(self) -> None:
        stream = io.StringIO()
        logger = ConsoleLogger(stream=stream, keys=("loss", "grad_norm"))
        logger.log_metrics({"loss": 1.0}, 0)
        assert "-" in stream.getvalue().splitlines()[2]

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (3.0, "3"),
            (0.125, "0.125"),
            (1.2345e-4, "1.2345e-04"),
            (5e6, "5000000"),
            (1.5e6, "1500000"),
            (123456.75, "1.2346e+05"),
            (2e9, "2.0000e+09"),
            (float("nan"), "NaN"),
            (float("inf"), "+inf"),
            (float("-inf"), "-inf"),
        ],
    )
    def test_number_formatting_across_ten_decades(
        self, value: float, expected: str
    ) -> None:
        # NaN and inf are the two values a reader most needs to notice.
        assert ConsoleLogger._format(value) == expected

    def test_a_namespaced_key_is_shortened_to_its_leaf(self) -> None:
        stream = io.StringIO()
        logger = ConsoleLogger(stream=stream, keys=("throughput/mfu",))
        logger.log_metrics({"throughput/mfu": 1.0}, 0)
        header = stream.getvalue().splitlines()[0]
        assert "mfu" in header
        assert "throughput/" not in header

    def test_a_long_leaf_is_truncated_from_the_left(self) -> None:
        # The units are the half that differs between neighbouring columns, so
        # they are the half that must survive.
        stream = io.StringIO()
        logger = ConsoleLogger(stream=stream, keys=("throughput/samples_per_s",))
        logger.log_metrics({"throughput/samples_per_s": 1.0}, 0)
        header = stream.getvalue().splitlines()[0]
        assert "amples_per_s" in header
        assert "samples_per_s" not in header

    def test_config_and_artifacts_reach_the_stream(self, tmp_path: Path) -> None:
        stream = io.StringIO()
        logger = ConsoleLogger(stream=stream)
        logger.log_config({"lr": 1e-4})
        logger.log_artifact(tmp_path / "grid.png")
        text = stream.getvalue()
        assert '"lr"' in text
        assert "artifact:" in text


class TestMultiLogger:
    """One call fans out, and one backend's failure never ends a run."""

    def test_it_forwards_to_every_backend(self, tmp_path: Path) -> None:
        first = JSONLLogger(tmp_path / "a.jsonl")
        second = JSONLLogger(tmp_path / "b.jsonl")
        multi = MultiLogger([first, second])
        multi.log_metrics({"loss": 1.0}, 0)
        multi.log_config({"a": 1})
        multi.log_artifact(tmp_path / "x.mp4")
        multi.close()

        for path in (tmp_path / "a.jsonl", tmp_path / "b.jsonl"):
            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            assert [record["type"] for record in records] == [
                "metrics",
                "config",
                "artifact",
            ]

    def test_the_wrapped_backends_are_exposed(self, tmp_path: Path) -> None:
        logger = JSONLLogger(tmp_path / "a.jsonl")
        assert MultiLogger([logger]).loggers == (logger,)
        logger.close()

    def test_a_failing_backend_is_warned_about_and_stepped_over(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # A full disk or an expired W&B token is survivable; the training job
        # is worth orders of magnitude more than the metric it failed to record.
        class Exploding:
            def log_metrics(self, metrics: Any, step: int) -> None:
                raise RuntimeError("disk full")

            def log_config(self, config: Any) -> None:
                raise RuntimeError("disk full")

            def log_artifact(self, path: Any) -> None:
                raise RuntimeError("disk full")

            def close(self) -> None:
                raise RuntimeError("disk full")

        healthy = JSONLLogger(tmp_path / "a.jsonl")
        multi = MultiLogger([Exploding(), healthy])
        with caplog.at_level(logging.WARNING, logger="avgen.telemetry"):
            multi.log_metrics({"loss": 1.0}, 0)
            multi.close()

        assert "Exploding.log_metrics failed" in caplog.text
        # The healthy backend still received the line.
        assert (
            json.loads(
                (tmp_path / "a.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )["loss"]
            == 1.0
        )

    def test_a_failing_backend_is_retried_on_the_next_call(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # It says so once and then keeps being tried in case it recovers.
        calls: list[int] = []

        class Flaky:
            def log_metrics(self, metrics: Any, step: int) -> None:
                calls.append(step)
                raise RuntimeError("transient")

        multi = MultiLogger([Flaky()])
        with caplog.at_level(logging.WARNING, logger="avgen.telemetry"):
            multi.log_metrics({"a": 1.0}, 0)
            multi.log_metrics({"a": 1.0}, 1)
        assert calls == [0, 1]

    def test_it_works_as_a_context_manager(self, tmp_path: Path) -> None:
        with MultiLogger([JSONLLogger(tmp_path / "a.jsonl")]) as multi:
            multi.log_metrics({"a": 1.0}, 0)
        assert (tmp_path / "a.jsonl").read_text(encoding="utf-8").strip()

    def test_an_empty_fan_out_is_harmless(self) -> None:
        multi = MultiLogger([])
        multi.log_metrics({"a": 1.0}, 0)
        multi.close()


class TestNoOpLogger:
    """A real object rather than ``None``, so the trainer never guards a call."""

    def test_it_is_disabled_and_accepts_everything(self, tmp_path: Path) -> None:
        logger = NoOpLogger()
        assert logger.enabled is False
        logger.log_metrics({"loss": 1.0}, 0)
        logger.log_config({"a": 1})
        logger.log_artifact(tmp_path / "x.mp4")
        logger.close()

    def test_it_is_disabled_even_on_rank_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RANK", "0")
        assert NoOpLogger().enabled is False

    def test_it_works_as_a_context_manager(self) -> None:
        with NoOpLogger() as logger:
            logger.log_metrics({"a": 1.0}, 0)


class TestBuildLogger:
    """Construction is identical on every rank, and a bad name fails loudly."""

    def test_an_empty_selection_is_a_no_op(self) -> None:
        assert isinstance(build_logger([]), NoOpLogger)
        assert isinstance(build_logger(""), NoOpLogger)
        assert isinstance(build_logger(["noop"]), NoOpLogger)
        assert isinstance(build_logger("noop"), NoOpLogger)

    def test_a_single_backend_is_returned_bare(self, tmp_path: Path) -> None:
        logger = build_logger(["jsonl"], log_dir=tmp_path)
        assert isinstance(logger, JSONLLogger)
        logger.close()

    def test_several_backends_become_a_multi_logger(self, tmp_path: Path) -> None:
        logger = build_logger(["console", "jsonl"], log_dir=tmp_path)
        assert isinstance(logger, MultiLogger)
        assert len(logger.loggers) == 2
        logger.close()

    def test_a_comma_separated_string_is_accepted(self, tmp_path: Path) -> None:
        logger = build_logger("console, jsonl", log_dir=tmp_path)
        assert isinstance(logger, MultiLogger)
        logger.close()

    def test_noop_is_dropped_from_a_mixed_selection(self, tmp_path: Path) -> None:
        logger = build_logger(["noop", "jsonl"], log_dir=tmp_path)
        assert isinstance(logger, JSONLLogger)
        logger.close()

    def test_the_jsonl_destination_defaults_under_the_log_dir(
        self, tmp_path: Path
    ) -> None:
        logger = build_logger(["jsonl"], log_dir=tmp_path)
        assert logger.path == tmp_path / "metrics.jsonl"
        logger.close()

    def test_an_explicit_jsonl_path_overrides_the_log_dir(self, tmp_path: Path) -> None:
        target = tmp_path / "elsewhere" / "stream.jsonl"
        logger = build_logger(["jsonl"], log_dir=tmp_path, jsonl_path=target)
        assert logger.path == target
        logger.close()

    def test_a_custom_jsonl_filename(self, tmp_path: Path) -> None:
        logger = build_logger(["jsonl"], log_dir=tmp_path, jsonl_filename="run.jsonl")
        assert logger.path == tmp_path / "run.jsonl"
        logger.close()

    def test_a_file_backend_without_a_destination_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"'jsonl' logger requires log_dir"):
            build_logger(["jsonl"])
        with pytest.raises(ValueError, match=r"'tensorboard' logger requires log_dir"):
            build_logger(["tensorboard"])

    def test_an_unknown_backend_lists_the_known_ones(self) -> None:
        with pytest.raises(ValueError, match=r"unknown logger 'mlflow'") as caught:
            build_logger(["mlflow"], log_dir="/tmp")
        assert "'console', 'jsonl', 'tensorboard', 'wandb', 'noop'" in str(caught.value)

    def test_a_non_logging_rank_gets_the_same_object_graph(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Control flow that differs by rank is how a collective gets skipped
        # and a job hangs.
        monkeypatch.setenv("RANK", "5")
        logger = build_logger(["console", "jsonl"], log_dir=tmp_path, rank=0)
        assert isinstance(logger, MultiLogger)
        assert len(logger.loggers) == 2
        assert all(not backend.enabled for backend in logger.loggers)
        logger.log_metrics({"loss": 1.0}, 0)
        assert not (tmp_path / "metrics.jsonl").exists()

    @pytest.mark.skipif(
        importlib_available("tensorboard"),
        reason="tensorboard is installed, so the gate cannot be observed",
    )
    def test_an_unavailable_tensorboard_backend_names_the_extra(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(RuntimeError, match=r'install "avgen\[tracking\]"'):
            build_logger(["tensorboard"], log_dir=tmp_path)

    @pytest.mark.skipif(
        importlib_available("wandb"),
        reason="wandb is installed, so the gate cannot be observed",
    )
    def test_an_unavailable_wandb_backend_names_the_install(self) -> None:
        with pytest.raises(RuntimeError, match=r"pip install wandb"):
            build_logger(["wandb"])

    @pytest.mark.skipif(
        importlib_available("wandb"),
        reason="wandb is installed, so the gate cannot be observed",
    )
    def test_a_gated_backend_is_not_constructed_off_the_logging_rank(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A disabled backend must not need its optional dependency, or a
        # 1024-rank job cannot start on an image that lacks it.
        monkeypatch.setenv("RANK", "5")
        assert build_logger(["wandb"], rank=0) is not None


class TestMemoryReporter:
    """Constructible on a CPU-only machine; silent when there is no allocator."""

    def test_thresholds_are_validated(self) -> None:
        with pytest.raises(ValueError, match=r"headroom_warning must be in \[0, 1\]"):
            MemoryReporter(headroom_warning=1.5)
        with pytest.raises(
            ValueError, match=r"fragmentation_warning must be in \[0, 1\]"
        ):
            MemoryReporter(fragmentation_warning=-0.1)

    @pytest.mark.skipif(
        torch.cuda.is_available(), reason="asserts the CPU-only fallback path"
    )
    def test_it_is_a_no_op_without_cuda(self) -> None:
        reporter = MemoryReporter()
        assert reporter.available is False
        assert reporter.report() is None
        assert reporter.peak_across(None) is None
        reporter.reset_peaks()  # must not raise
        reporter.clear_warning()


class TestProfiler:
    """Rank selection and the disabled paths, which must work with no CUDA."""

    def test_global_zero_profiles_a_single_process_job(self) -> None:
        assert should_profile_rank(mode="global_zero") is True

    def test_all_profiles_everything(self) -> None:
        assert should_profile_rank(mode="all") is True

    def test_local_zero_reads_the_launcher_variable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LOCAL_RANK", "0")
        assert should_profile_rank(mode="local_zero") is True
        monkeypatch.setenv("LOCAL_RANK", "3")
        assert should_profile_rank(mode="local_zero") is False
        monkeypatch.setenv("LOCAL_RANK", "garbage")
        assert should_profile_rank(mode="local_zero") is False

    def test_an_unknown_mode_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"mode must be"):
            should_profile_rank(mode="every_other_tuesday")

    def test_a_disabled_profiler_yields_a_null_object(self, tmp_path: Path) -> None:
        # So the training loop needs no conditional around profiler.step().
        with profile_steps(tmp_path, enabled=False) as profiler:
            profiler.step()
            profiler.export_chrome_trace(str(tmp_path / "trace.json"))
        assert not any(tmp_path.iterdir())

    def test_a_non_profiling_rank_yields_a_null_object(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LOCAL_RANK", "2")
        with profile_steps(tmp_path, rank_mode="local_zero") as profiler:
            profiler.step()
        assert not any(tmp_path.iterdir())

    def test_memory_snapshot_is_a_no_op_when_disabled(self, tmp_path: Path) -> None:
        with memory_snapshot(tmp_path / "snap.pickle", enabled=False):
            pass
        assert not (tmp_path / "snap.pickle").exists()

    @pytest.mark.skipif(
        torch.cuda.is_available(), reason="asserts the CPU-only fallback path"
    )
    def test_memory_snapshot_is_a_no_op_without_cuda(self, tmp_path: Path) -> None:
        with memory_snapshot(tmp_path / "snap.pickle"):
            pass
        assert not (tmp_path / "snap.pickle").exists()

    def test_the_flight_recorder_needs_a_process_group(self, tmp_path: Path) -> None:
        assert flight_recorder_dump(tmp_path / "nccl.dump") is None


class TestEndToEnd:
    """The loop from the module docstring, wired together on the CPU."""

    def test_accumulate_meter_and_log_one_interval(
        self, tmp_path: Path, fake_clock: Callable[[list[float]], None]
    ) -> None:
        fake_clock([0.0, 1.0, 2.0, 3.0, 4.0, 4.0])
        accumulator = MetricAccumulator()
        meter = ThroughputMeter(peak_flops_per_s=1000.0, flops_per_step=100.0)
        logger = build_logger(["jsonl"], log_dir=tmp_path)

        for step in range(4):
            accumulator.update(step_metrics(loss=1.0 / (step + 1), video_tokens=1000))
            meter.step(samples=8, tokens=1000)
        logger.log_metrics(
            merge_metrics(accumulator.flush(), meter.report().to_mapping()), 4
        )
        logger.close()

        record = json.loads(
            (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )
        assert record["step"] == 4
        assert record["valid_video_tokens"] == 4000.0
        assert record["throughput/steps"] == 4.0
        assert record["throughput/mfu"] == pytest.approx(0.10)
        assert math.isfinite(record["loss"])


class TestWeightingSemantics:
    """The two halves of an accumulated metric behave differently on purpose.

    Mean fields (loss, gradient norm, learning rate) are weighted averages, so a
    step that saw more tokens counts for more. Sum fields (token counts,
    non-finite counts, skipped steps) are totals: a weight would make them
    quadratic in the token count, since the docstring tells callers to pass the
    token count as the weight. Both of these were wrong once; they stay tested.
    """

    def test_summed_fields_ignore_the_weight(self) -> None:
        accumulator = MetricAccumulator()
        accumulator.update(step_metrics(video_tokens=100), weight=2.0)
        accumulator.update(step_metrics(video_tokens=100), weight=6.0)
        assert accumulator.flush()["valid_video_tokens"] == pytest.approx(200.0)

    def test_the_step_count_is_a_step_count(self) -> None:
        accumulator = MetricAccumulator()
        accumulator.update(step_metrics(), weight=2.0)
        accumulator.update(step_metrics(), weight=6.0)
        assert accumulator.flush()["steps"] == pytest.approx(2.0)

    def test_mean_fields_are_weighted_by_the_token_count(self) -> None:
        """A step over more tokens must pull the mean further."""
        accumulator = MetricAccumulator()
        accumulator.update(step_metrics(loss=1.0), weight=1.0)
        accumulator.update(step_metrics(loss=5.0), weight=3.0)
        assert accumulator.flush()["loss"] == pytest.approx(4.0)
