"""Where metrics go: console, JSONL, TensorBoard, Weights & Biases, or nowhere.

Three design decisions shape this module, and all three come from running jobs
big enough that logging itself becomes a problem.

**Rank gating is built into the base class, not left to the caller.** A
1024-rank job where every rank logs writes 1024 lines per step to the same
filesystem, produces a file no tool can read, and — with a network backend —
opens 1024 sockets to a service that will rate-limit the job into a stall. The
correct behaviour is that exactly one rank logs, and the correct place to
enforce it is here, once, rather than at every call site where somebody will
eventually forget. :class:`RankGatedLogger` makes non-logging ranks return
immediately without touching the payload.

That gate is also why every metric must already be *reduced* before it arrives.
Rank 0's local loss is not the job's loss. Reduction happens in
:class:`~avgen.telemetry.metrics.MetricAccumulator` over the ``dp_cp`` mesh; by
the time a number reaches a logger it is a global number and rank 0 is merely
the one that writes it down.

**JSONL is the format of record.** Not because it is nice to look at, but
because it is the only one that survives: a TensorBoard event file needs
TensorBoard, a W&B run needs an account and a network, and a console log needs a
human. One JSON object per line needs nothing, appends atomically, tolerates
truncation at the end (the last line is dropped, the rest still parses), and can
be read by ``jq``, pandas, or a shell loop. Every other backend is a
convenience layered on top of it.

**Optional backends import lazily.** ``tensorboard`` and ``wandb`` are extras.
Importing them at module scope would make ``avgen.telemetry`` unimportable on a
cluster image that does not have them, which is most cluster images.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import IO, Any, Protocol, runtime_checkable

import torch.distributed as dist

__all__ = [
    "ConsoleLogger",
    "JSONLLogger",
    "Logger",
    "MultiLogger",
    "NoOpLogger",
    "RankGatedLogger",
    "TensorBoardLogger",
    "WandbLogger",
    "build_logger",
]

_LOG = logging.getLogger("avgen.telemetry")

#: Column width for the console table. Wide enough for ``1.2345e-04`` plus a
#: sign, narrow enough that eight metrics still fit an 80-column terminal.
_CONSOLE_COLUMN = 12


@runtime_checkable
class Logger(Protocol):
    """The whole logging surface: four methods, no state the caller can see.

    Deliberately narrower than any backend's own API. A logger that exposes
    ``watch``, ``define_metric``, ``add_histogram`` and so on forces every call
    site to know which backend it is talking to, and the trainer then cannot be
    reconfigured from a YAML file. Anything a specific backend can do that this
    protocol cannot express belongs in that backend's construction arguments.
    """

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        """Record already-reduced scalar metrics for one step."""
        ...

    def log_config(self, config: Mapping[str, Any]) -> None:
        """Record the run's configuration, once, at startup."""
        ...

    def log_artifact(self, path: str | os.PathLike[str]) -> None:
        """Record a file produced by the run — a sample grid, an eval report."""
        ...

    def close(self) -> None:
        """Flush and release resources."""
        ...


def _current_rank() -> int:
    """This process's global rank, or ``0`` when not launched distributed.

    Reads the process group when there is one and falls back to ``RANK`` so a
    logger constructed before ``init_distributed`` still gates correctly.
    """
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    raw = os.environ.get("RANK")
    if raw is None or not raw.lstrip("-").isdigit():
        return 0
    return int(raw)


class RankGatedLogger:
    """Base class that turns every method into a no-op off the logging rank.

    Subclasses implement the ``_write_*`` hooks and never think about ranks
    again. The gate is evaluated once, at construction, rather than per call:
    the rank of a process does not change, and a per-call check is a branch on
    the hot path for no benefit.

    Args:
        rank: The rank that logs. ``0`` is almost always right.
        enabled: Force-disable regardless of rank, used by :func:`build_logger`
            to build a structurally identical logger tree on every rank.
    """

    __slots__ = ("_enabled",)

    def __init__(self, *, rank: int = 0, enabled: bool = True) -> None:
        self._enabled = enabled and _current_rank() == rank

    @property
    def enabled(self) -> bool:
        """Whether this process actually writes anything."""
        return self._enabled

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        """Record metrics if this rank logs.

        Args:
            metrics: Already-reduced scalars.
            step: Optimizer step the metrics belong to.
        """
        if self._enabled:
            self._write_metrics(metrics, step)

    def log_config(self, config: Mapping[str, Any]) -> None:
        """Record the run configuration if this rank logs.

        Args:
            config: JSON-serialisable configuration mapping.
        """
        if self._enabled:
            self._write_config(config)

    def log_artifact(self, path: str | os.PathLike[str]) -> None:
        """Record an output file if this rank logs.

        Args:
            path: Path to the artifact.
        """
        if self._enabled:
            self._write_artifact(Path(path))

    def close(self) -> None:
        """Flush and release backend resources if this rank logs."""
        if self._enabled:
            self._close()

    def __enter__(self) -> RankGatedLogger:
        """Return self so a logger can scope a run."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close on exit, including on an exception."""
        self.close()

    def _write_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        """Backend hook for metrics."""

    def _write_config(self, config: Mapping[str, Any]) -> None:
        """Backend hook for configuration."""

    def _write_artifact(self, path: Path) -> None:
        """Backend hook for artifacts."""

    def _close(self) -> None:
        """Backend hook for shutdown."""


class NoOpLogger(RankGatedLogger):
    """Discards everything.

    Not a placeholder: it is the correct logger for a unit test, a simulation,
    and every rank that is not rank 0. Having a real object rather than ``None``
    means the trainer never guards a logging call, which is one fewer branch and
    one fewer place to forget the guard.
    """

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(enabled=False)


class ConsoleLogger(RankGatedLogger):
    """Fixed-width table for a human watching a job start.

    The header is reprinted periodically because the reader has scrolled past
    the first one, and columns are fixed width because a table whose columns
    move is unreadable at a glance — which is the only thing a console log is
    for. Anything that needs to be parsed should come from the JSONL file.

    Args:
        stream: Where to write. Defaults to stdout.
        keys: Metric names to show, in order. ``None`` shows whatever the first
            call provides, sorted, which keeps the column set stable thereafter.
        header_every: Reprint the header every this many rows.
        rank: The rank that logs.
        enabled: Force-disable.
    """

    __slots__ = ("_header_every", "_keys", "_rows", "_stream")

    def __init__(
        self,
        *,
        stream: IO[str] | None = None,
        keys: Sequence[str] | None = None,
        header_every: int = 25,
        rank: int = 0,
        enabled: bool = True,
    ) -> None:
        super().__init__(rank=rank, enabled=enabled)
        self._stream = stream if stream is not None else sys.stdout
        self._keys: tuple[str, ...] | None = tuple(keys) if keys is not None else None
        self._header_every = max(1, header_every)
        self._rows = 0

    @staticmethod
    def _format(value: float) -> str:
        """Render one number in a fixed width, readable across ten decades."""
        if isinstance(value, bool):
            return "yes" if value else "no"
        if not math.isfinite(value):
            # NaN and inf are the two values a reader most needs to notice.
            return "NaN" if math.isnan(value) else ("+inf" if value > 0 else "-inf")
        magnitude = abs(value)
        if value == int(value) and magnitude < 1e9:
            return f"{int(value):d}"
        if magnitude >= 1e5 or (magnitude < 1e-3 and magnitude > 0):
            return f"{value:.4e}"
        return f"{value:.5g}"

    def _write_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        if self._keys is None:
            self._keys = tuple(sorted(metrics))
        if self._rows % self._header_every == 0:
            header = "step".rjust(9) + "".join(
                _column_label(key).rjust(_CONSOLE_COLUMN + 1) for key in self._keys
            )
            self._stream.write(header + "\n")
            self._stream.write("-" * len(header) + "\n")
        row = f"{step:>9d}" + "".join(
            self._format(float(metrics[key])).rjust(_CONSOLE_COLUMN + 1)
            if key in metrics
            else "-".rjust(_CONSOLE_COLUMN + 1)
            for key in self._keys
        )
        self._stream.write(row + "\n")
        self._stream.flush()
        self._rows += 1

    def _write_config(self, config: Mapping[str, Any]) -> None:
        self._stream.write(json.dumps(dict(config), indent=2, sort_keys=True) + "\n")
        self._stream.flush()

    def _write_artifact(self, path: Path) -> None:
        self._stream.write(f"artifact: {path}\n")
        self._stream.flush()


def _column_label(key: str) -> str:
    """Shorten a namespaced metric name to something a column can hold.

    ``throughput/samples_per_s`` becomes ``samples_per_s``: the namespace is
    what disambiguates keys in a file, and the leaf is what identifies them to
    a human reading a terminal. Longer leaves are truncated from the left,
    keeping the units, which is the half that differs between neighbours.
    """
    leaf = key.rsplit("/", 1)[-1]
    return leaf if len(leaf) <= _CONSOLE_COLUMN else leaf[-_CONSOLE_COLUMN:]


class JSONLLogger(RankGatedLogger):
    """One JSON object per line — the format everything else can read.

    The file is opened in append mode so a resumed run continues the same
    history instead of truncating it; the ``step`` field is what a reader uses
    to detect and drop the overlap when a job is preempted mid-interval.

    Flushing every line is deliberate. Buffered output is lost when a job is
    killed, and a job being killed is precisely when the last few lines matter
    most. The cost is one small write per log interval, not per step.

    Args:
        path: File to append to. Parent directories are created.
        rank: The rank that logs.
        enabled: Force-disable.
    """

    __slots__ = ("_handle", "_path")

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        rank: int = 0,
        enabled: bool = True,
    ) -> None:
        super().__init__(rank=rank, enabled=enabled)
        self._path = Path(path)
        self._handle: IO[str] | None = None
        if self.enabled:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self._path.open("a", encoding="utf-8")

    @property
    def path(self) -> Path:
        """The file being appended to."""
        return self._path

    def _emit(self, record: Mapping[str, Any]) -> None:
        if self._handle is None:
            return
        # `default=str` keeps a stray Path or dtype from killing a training run
        # over a log line. Losing a run to the logger is never acceptable.
        self._handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        self._handle.flush()

    def _write_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        self._emit({"type": "metrics", "step": step, **dict(metrics)})

    def _write_config(self, config: Mapping[str, Any]) -> None:
        self._emit({"type": "config", "config": dict(config)})

    def _write_artifact(self, path: Path) -> None:
        self._emit({"type": "artifact", "path": str(path)})

    def _close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


class TensorBoardLogger(RankGatedLogger):
    """Scalars to a TensorBoard event file.

    Args:
        log_dir: Directory for the event file.
        rank: The rank that logs.
        enabled: Force-disable.
        flush_secs: How often the writer flushes to disk.

    Raises:
        RuntimeError: If ``tensorboard`` is not installed.
    """

    __slots__ = ("_writer",)

    def __init__(
        self,
        log_dir: str | os.PathLike[str],
        *,
        rank: int = 0,
        enabled: bool = True,
        flush_secs: int = 30,
    ) -> None:
        super().__init__(rank=rank, enabled=enabled)
        self._writer: Any = None
        if not self.enabled:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as error:
            raise RuntimeError(
                "tensorboard logging requires the tracking extra; "
                'install "avgen[tracking]"'
            ) from error
        self._writer = SummaryWriter(log_dir=str(log_dir), flush_secs=flush_secs)

    def _write_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        for key in sorted(metrics):
            self._writer.add_scalar(key, float(metrics[key]), global_step=step)

    def _write_config(self, config: Mapping[str, Any]) -> None:
        # `add_text` rather than `add_hparams`: hparams needs a metric to pair
        # with and silently drops non-scalar values, which is most of a config.
        rendered = json.dumps(dict(config), indent=2)
        self._writer.add_text("config", f"```\n{rendered}\n```")

    def _write_artifact(self, path: Path) -> None:
        self._writer.add_text("artifacts", str(path))

    def _close(self) -> None:
        self._writer.flush()
        self._writer.close()


class WandbLogger(RankGatedLogger):
    """Scalars, config, and files to a Weights & Biases run.

    Args:
        project: W&B project name.
        name: Run name. ``None`` lets the service generate one.
        config: Configuration logged at ``init`` time.
        rank: The rank that logs.
        enabled: Force-disable.
        mode: ``"online"``, ``"offline"``, or ``"disabled"``. Use ``"offline"``
            on a cluster whose compute nodes have no egress — the run syncs
            later from the login node — rather than letting every step block on
            a socket that will never connect.

    Raises:
        RuntimeError: If ``wandb`` is not installed.
    """

    __slots__ = ("_run", "_wandb")

    def __init__(
        self,
        *,
        project: str = "avgen",
        name: str | None = None,
        config: Mapping[str, Any] | None = None,
        rank: int = 0,
        enabled: bool = True,
        mode: str = "online",
    ) -> None:
        super().__init__(rank=rank, enabled=enabled)
        self._wandb: Any = None
        self._run: Any = None
        if not self.enabled:
            return
        try:
            # wandb is not a declared extra, so it has no stub in the mypy
            # override list; the lazy import is guarded either way.
            import wandb  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError(
                "wandb logging requires wandb; install it with 'pip install wandb'"
            ) from error
        self._wandb = wandb
        self._run = wandb.init(
            project=project, name=name, config=dict(config or {}), mode=mode
        )

    def _write_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        self._run.log(dict(metrics), step=step)

    def _write_config(self, config: Mapping[str, Any]) -> None:
        self._run.config.update(dict(config), allow_val_change=True)

    def _write_artifact(self, path: Path) -> None:
        self._run.save(str(path), policy="now")

    def _close(self) -> None:
        self._run.finish()


class MultiLogger:
    """Fans one call out to several loggers, isolating each one's failures.

    A backend raising must not end a training run. A W&B socket timeout, a full
    disk on the TensorBoard volume, a permissions change on the log directory:
    all of these are logged as warnings and stepped over, because the run is
    worth more than the telemetry. The one exception is that failures are not
    swallowed silently — a backend that has failed says so, once, and then keeps
    being tried in case it recovers.

    Args:
        loggers: The backends to fan out to.
    """

    __slots__ = ("_loggers",)

    def __init__(self, loggers: Iterable[Logger]) -> None:
        self._loggers = tuple(loggers)

    @property
    def loggers(self) -> tuple[Logger, ...]:
        """The wrapped backends."""
        return self._loggers

    def _each(self, action: str, *args: Any) -> None:
        for logger in self._loggers:
            try:
                getattr(logger, action)(*args)
            except Exception:
                # Telemetry must never kill a run: a full disk, an expired W&B
                # token, or a revoked directory permission are all survivable,
                # and the training job is worth orders of magnitude more than
                # the metric it failed to record.
                _LOG.warning(
                    "%s.%s failed; continuing",
                    type(logger).__name__,
                    action,
                    exc_info=True,
                )

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        """Forward metrics to every backend.

        Args:
            metrics: Already-reduced scalars.
            step: Optimizer step.
        """
        self._each("log_metrics", metrics, step)

    def log_config(self, config: Mapping[str, Any]) -> None:
        """Forward the configuration to every backend.

        Args:
            config: JSON-serialisable configuration.
        """
        self._each("log_config", config)

    def log_artifact(self, path: str | os.PathLike[str]) -> None:
        """Forward an artifact path to every backend.

        Args:
            path: Path to the artifact.
        """
        self._each("log_artifact", path)

    def close(self) -> None:
        """Close every backend."""
        self._each("close")

    def __enter__(self) -> MultiLogger:
        """Return self so the fan-out can scope a run."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close every backend on exit."""
        self.close()


def build_logger(
    names: Sequence[str] | str,
    *,
    log_dir: str | os.PathLike[str] | None = None,
    run_name: str | None = None,
    project: str = "avgen",
    config: Mapping[str, Any] | None = None,
    rank: int = 0,
    jsonl_filename: str = "metrics.jsonl",
    jsonl_path: str | os.PathLike[str] | None = None,
    tensorboard_dir: str | os.PathLike[str] | None = None,
    console_keys: Sequence[str] | None = None,
    wandb_mode: str = "online",
) -> Logger:
    """Construct the configured logger fan-out.

    Called identically on every rank. Non-logging ranks get the same object
    graph with every backend disabled, so the trainer's control flow does not
    depend on rank — which matters more than it sounds, because control flow
    that differs by rank is how a collective gets skipped and a job hangs.

    Args:
        names: Backend names, or one comma-separated string. Recognised:
            ``"console"``, ``"jsonl"``, ``"tensorboard"``, ``"wandb"``,
            ``"noop"``. An empty selection yields a no-op logger.
        log_dir: Directory for file-backed backends. Required by ``jsonl`` and
            ``tensorboard``.
        run_name: Run name for W&B.
        project: Project name for W&B.
        config: Configuration passed to W&B at init.
        rank: The rank that logs.
        jsonl_filename: Filename inside ``log_dir`` for the JSONL stream.
        jsonl_path: Explicit destination for the JSONL stream, overriding
            ``log_dir``/``jsonl_filename``. Lets a run put its metric stream
            somewhere other than beside its checkpoints.
        tensorboard_dir: Explicit destination for TensorBoard events,
            overriding ``log_dir``/``tensorboard``. Useful when one event
            directory aggregates several runs for comparison.
        console_keys: Fixed console column order.
        wandb_mode: ``"online"``, ``"offline"``, or ``"disabled"``.

    Returns:
        A single logger; a :class:`MultiLogger` when more than one was asked for.

    Raises:
        ValueError: If a name is unknown, or a file backend was asked for
            without a destination.
    """
    selected = (
        [part.strip() for part in names.split(",") if part.strip()]
        if isinstance(names, str)
        else [str(part).strip() for part in names if str(part).strip()]
    )
    if not selected or selected == ["noop"]:
        return NoOpLogger()

    directory = Path(log_dir) if log_dir is not None else None
    built: list[Logger] = []
    for name in selected:
        if name == "noop":
            continue
        if name == "console":
            built.append(ConsoleLogger(keys=console_keys, rank=rank))
        elif name == "jsonl":
            if jsonl_path is not None:
                target = Path(jsonl_path)
            elif directory is not None:
                target = directory / jsonl_filename
            else:
                raise ValueError("the 'jsonl' logger requires log_dir or jsonl_path")
            built.append(JSONLLogger(target, rank=rank))
        elif name == "tensorboard":
            if tensorboard_dir is not None:
                events = Path(tensorboard_dir)
            elif directory is not None:
                events = directory / "tensorboard"
            else:
                raise ValueError(
                    "the 'tensorboard' logger requires log_dir or tensorboard_dir"
                )
            built.append(TensorBoardLogger(events, rank=rank))
        elif name == "wandb":
            built.append(
                WandbLogger(
                    project=project,
                    name=run_name,
                    config=config,
                    rank=rank,
                    mode=wandb_mode,
                )
            )
        else:
            raise ValueError(
                f"unknown logger {name!r}; expected one of "
                "'console', 'jsonl', 'tensorboard', 'wandb', 'noop'"
            )
    if not built:
        return NoOpLogger()
    if len(built) == 1:
        return built[0]
    return MultiLogger(built)
