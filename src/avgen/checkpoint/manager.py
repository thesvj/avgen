"""Sharded, resumable, self-pruning checkpoints for long distributed runs.

A checkpoint layer for a job that runs for weeks on hardware that is preempted
weekly has three jobs, in this order of importance.

**1. Never hand back a checkpoint that is not finished.** A save at 1024 ranks
writes thousands of files. If the job dies halfway through, the directory looks
almost exactly like a complete one — same name, same layout, most of the files
present. Resuming from it fails somewhere deep inside the load planner, or, far
worse, succeeds with a subset of the tensors. The defence is a *completion
marker written atomically after every rank has finished*: a directory without
``avgen_checkpoint.json`` is invisible to
:meth:`CheckpointManager.list_checkpoints`, and therefore invisible to resume.

**2. Never let a save block training longer than it has to, and never let it
corrupt the tensors it is writing.** Those two goals fight each other, and the
resolution is the single most important property in this module:

    An asynchronous save must copy ("stage") the state dict **synchronously**,
    and background only the IO.

``dcp.async_save`` does exactly that, and it matters because the alternative is
a bug that cannot be found from a stack trace. If the writer thread reads the
live parameter tensors while training continues, the optimizer step that runs
half a second later mutates those tensors *mid-write*. The resulting checkpoint
is internally inconsistent — some tensors from step ``N``, some from step
``N+1``, some torn between the two — and it loads without a single error. The
run resumes, the loss is subtly wrong, and nothing in the logs says why. The
staging copy is the price of correctness; it is paid once, synchronously, and it
is not optional.

The same reasoning forbids two overlapping saves. If save ``N+1`` begins
staging while save ``N`` is still uploading, the two share a storage writer and
a thread pool and race on the destination directory.
:meth:`CheckpointManager.save` therefore blocks on the previous future before
starting, and says so in the logs when it does — that stall is real information,
it means the filesystem is not keeping up with the training loop.

**3. Reshard.** DCP stores each tensor as a set of chunks with explicit global
offsets, so a load at a different rank count re-plans the reads and stitches the
chunks back together. This is the whole reason avgen does not hand-roll sharded
IO. A rank-count-locked checkpoint format — one file per rank, restored
positionally — means a job preempted off 512 GPUs cannot restart on the 256 the
scheduler is offering. The run waits for the original allocation instead, which
in practice is how a week of compute gets lost.

The one component that does not reshard is per-rank RNG generator state; see
:class:`~avgen.checkpoint.stateful.RNGState`. The manager detects the width
change and re-seeds rather than restoring it.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import get_model_state_dict

from avgen.checkpoint.stateful import (
    EMA_KEY,
    EXTRAS_KEY,
    MODEL_OPTIMIZER_KEY,
    PROGRESS_KEY,
    RNG_KEY,
    SCHEDULE_KEY,
    build_stateful,
)
from avgen.core.state import TrainState
from avgen.parallel.env import collective_timeout

if TYPE_CHECKING:  # pragma: no cover - typing only
    from avgen.parallel.apply import ParallelModel

__all__ = [
    "CHECKPOINT_DIR_PREFIX",
    "CHECKPOINT_FORMAT_VERSION",
    "MARKER_FILENAME",
    "CheckpointEntry",
    "CheckpointManager",
    "load",
    "load_ema",
    "load_model",
    "save",
]

_LOG = logging.getLogger("avgen.checkpoint")

#: Directory naming. Zero padding to 10 digits keeps lexicographic order equal
#: to numeric order, so `ls` and a glob agree with `sorted(by step)` even for a
#: run that reaches a billion steps.
CHECKPOINT_DIR_PREFIX = "step_"
_STEP_DIGITS = 10

#: Presence of this file is the definition of "complete". It is written last,
#: by rank 0 only, via an atomic rename. Its contents are the human- and
#: tool-readable description of the checkpoint.
MARKER_FILENAME = "avgen_checkpoint.json"

#: Bumped whenever the top-level key layout changes. Recorded in the marker so
#: a future loader can refuse a format it does not understand instead of
#: half-loading it.
CHECKPOINT_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class CheckpointEntry:
    """One complete checkpoint on disk.

    Args:
        step: Optimizer step the checkpoint was taken at.
        path: Directory holding the DCP shards and the marker.
        metadata: Decoded contents of the completion marker.
    """

    step: int
    path: Path
    metadata: dict[str, Any]

    @property
    def data_world(self) -> int:
        """Data-parallel width the checkpoint was written at, or ``1``."""
        return int(self.metadata.get("data_world", 1))

    @property
    def world_size(self) -> int:
        """Total rank count the checkpoint was written at, or ``1``."""
        return int(self.metadata.get("world_size", 1))


def _step_dirname(step: int) -> str:
    return f"{CHECKPOINT_DIR_PREFIX}{step:0{_STEP_DIGITS}d}"


def _parse_step(name: str) -> int | None:
    if not name.startswith(CHECKPOINT_DIR_PREFIX):
        return None
    suffix = name[len(CHECKPOINT_DIR_PREFIX) :]
    if not suffix.isdigit():
        return None
    return int(suffix)


def _world_size() -> int:
    """Total rank count, or ``1`` when there is no process group."""
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_world_size())
    return 1


def _is_main_rank() -> bool:
    """Whether this process writes metadata and deletes directories.

    Deliberately asks ``torch.distributed`` rather than taking a
    :class:`~avgen.parallel.env.DistributedEnv`: the manager is also used by
    offline tools (``avgen checkpoint inspect``) that never initialise a group.
    """
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank()) == 0
    return True


def _data_coordinates(parallel: ParallelModel | None) -> tuple[int, int]:
    """Return ``(data_rank, data_world)`` for RNG namespacing.

    Falls back to the single-replica coordinate when there is no mesh, which is
    the correct answer for a single-process run and for a CPU smoke test.
    """
    if parallel is None or parallel.mesh is None:
        return 0, 1
    coordinates: tuple[int, int] = parallel.dims.data_coordinates(parallel.mesh)
    return coordinates


class CheckpointManager:
    """Owns a directory of checkpoints: writing, resuming, and pruning it.

    One instance per run, held by the trainer for the life of the job. It is
    stateful on purpose — it tracks the in-flight asynchronous save so that
    exactly one can be outstanding, and so that the completion marker is written
    only after that save has actually landed.

    The rejected alternative was a set of free functions plus a marker written
    eagerly at save time. It is stateless and it is wrong: with an async save
    the marker would appear before the bytes did, and a job that died in between
    would leave a checkpoint that advertises itself as complete and is not.

    Args:
        root: Directory holding one subdirectory per checkpoint. Created on
            demand by rank 0.
        keep_last_n: Number of most recent checkpoints to retain. ``0`` keeps
            all of them.
        keep_every_n_steps: Additionally retain every checkpoint whose step is a
            multiple of this. ``0`` disables it. The two policies compose: the
            usual configuration is a short rolling window for preemption
            recovery plus a sparse permanent series for post-hoc analysis, and
            the sparse series must survive the rolling window's pruning.
        save_timeout_s: Wall-clock bound on the collective portion of a save.
            Applied by shortening the process-group timeout, so a hung shared
            filesystem raises in minutes instead of stalling every rank until
            the steady-state timeout expires.
        strict: Whether model/optimizer key mismatches on load are fatal.
        include_rng: Whether generator state participates at all. Restoring it
            is still skipped automatically when the data-parallel width changed.
        cpu_offload: Whether to materialise model state on CPU before writing.
            Redundant with async staging; useful for a synchronous save on a
            device that is close to its memory ceiling.
        thread_count: Writer threads per rank. More threads help on an object
            store with per-request latency, not on a local NVMe.

    Raises:
        ValueError: If a retention or timeout setting is negative.
    """

    __slots__ = (
        "_cpu_offload",
        "_include_rng",
        "_keep_every_n_steps",
        "_keep_last_n",
        "_pending_future",
        "_pending_marker",
        "_pending_step",
        "_root",
        "_save_timeout_s",
        "_strict",
        "_thread_count",
    )

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        keep_last_n: int = 3,
        keep_every_n_steps: int = 0,
        save_timeout_s: float = 600.0,
        strict: bool = True,
        include_rng: bool = True,
        cpu_offload: bool = False,
        thread_count: int = 1,
    ) -> None:
        if keep_last_n < 0:
            raise ValueError(f"keep_last_n must be non-negative; got {keep_last_n}")
        if keep_every_n_steps < 0:
            raise ValueError(
                f"keep_every_n_steps must be non-negative; got {keep_every_n_steps}"
            )
        if save_timeout_s <= 0:
            raise ValueError(f"save_timeout_s must be positive; got {save_timeout_s}")
        if thread_count < 1:
            raise ValueError(f"thread_count must be >= 1; got {thread_count}")
        self._root = Path(root)
        self._keep_last_n = keep_last_n
        self._keep_every_n_steps = keep_every_n_steps
        self._save_timeout_s = save_timeout_s
        self._strict = strict
        self._include_rng = include_rng
        self._cpu_offload = cpu_offload
        self._thread_count = thread_count
        self._pending_future: Future[Any] | None = None
        self._pending_step: int | None = None
        self._pending_marker: dict[str, Any] | None = None

    @property
    def root(self) -> Path:
        """The directory this manager owns."""
        return self._root

    @property
    def has_pending_save(self) -> bool:
        """Whether an asynchronous save is still uploading."""
        return self._pending_future is not None

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def save(
        self,
        step: int,
        state: TrainState,
        *,
        parallel: ParallelModel | None = None,
        async_save: bool = True,
        extras: dict[str, Any] | None = None,
    ) -> Path:
        """Write a checkpoint for ``step``.

        **Every rank must call this.** The save is a collective: ranks agree on
        a write plan, deduplicate replicated tensors, and each write their own
        shards. A rank that skips the call hangs the job at the next collective,
        so the usual ``if rank == 0:`` guard around checkpointing is a bug. Only
        the *metadata* write and the *pruning* are rank-0 work, and they happen
        inside this method.

        With ``async_save`` the state dict is staged to host memory before this
        returns, and only the upload continues in the background. See the module
        docstring for why that ordering is not negotiable.

        Args:
            step: Optimizer step being checkpointed. Names the directory.
            state: The live training state.
            parallel: The parallelised model, used for the data-parallel
                coordinates recorded in the marker and used to namespace RNG.
                ``None`` means single-replica.
            async_save: Whether to background the IO. Set false for the final
                checkpoint of a run, where there is no training left to overlap
                with and a synchronous write is one less thing to get wrong.
            extras: Arbitrary JSON-serialisable fields to record in the marker —
                the run id, the config hash, the git SHA. Not part of the tensor
                state; this is what makes a checkpoint directory
                self-describing six months later.

        Returns:
            The checkpoint directory.

        Raises:
            ValueError: If ``step`` is negative.
        """
        if isinstance(step, bool) or step < 0:
            raise ValueError(f"step must be a non-negative integer; got {step!r}")

        # Draining first is what guarantees a single writer. It also surfaces an
        # exception from the *previous* save at a point where we can still act
        # on it, rather than at process exit.
        self.wait()

        data_rank, data_world = _data_coordinates(parallel)
        directory = self._root / _step_dirname(step)

        if _is_main_rank():
            directory.mkdir(parents=True, exist_ok=True)

        entries = build_stateful(
            state,
            data_rank=data_rank,
            include_rng=self._include_rng,
            strict=self._strict,
            cpu_offload=self._cpu_offload,
        )
        marker = self._build_marker(
            step=step,
            state=state,
            entries=entries,
            data_world=data_world,
            parallel=parallel,
            extras=extras,
        )

        writer = dcp.FileSystemWriter(
            directory,
            thread_count=self._thread_count,
            # A per-rank file is one open handle and one metadata entry per rank
            # instead of one per tensor. At four figures of ranks that is the
            # difference between a save and a distributed filesystem incident.
            single_file_per_rank=True,
            # Retrying a save into an existing directory must overwrite, or a
            # preempted-and-restarted job dies on its first checkpoint.
            overwrite=True,
        )

        started = time.monotonic()
        if async_save:
            # The shortened timeout covers staging and the planning collective,
            # which is where a hung filesystem shows up first. The upload
            # thread's own collectives run under the steady-state timeout; the
            # bound on those is the `future.result(timeout=...)` in `wait`.
            with collective_timeout(self._save_timeout_s):
                future = dcp.async_save(entries, storage_writer=writer)
            self._pending_future = _upload_future(future)
            self._pending_step = step
            self._pending_marker = marker
            _LOG.info(
                "checkpoint step=%d staged in %.2fs, uploading in background -> %s",
                step,
                time.monotonic() - started,
                directory,
            )
        else:
            with collective_timeout(self._save_timeout_s):
                dcp.save(entries, storage_writer=writer)
            self._finalize(step, directory, marker)
            _LOG.info(
                "checkpoint step=%d written in %.2fs -> %s",
                step,
                time.monotonic() - started,
                directory,
            )
        return directory

    def wait(self) -> None:
        """Block until any in-flight save has landed, then mark it complete.

        Safe and cheap to call when nothing is pending. Call it before reading
        the checkpoint directory, before exiting, and — the manager does this
        itself — before starting another save.

        Raises:
            TimeoutError: If the upload does not finish within
                ``save_timeout_s``. The partially written directory is left in
                place without a marker, so it is ignored by resume rather than
                silently used.
        """
        future = self._pending_future
        if future is None:
            return
        step = self._pending_step
        marker = self._pending_marker
        self._pending_future = None
        self._pending_step = None
        self._pending_marker = None

        started = time.monotonic()
        try:
            future.result(timeout=self._save_timeout_s)
        except TimeoutError as error:  # pragma: no cover - needs a stuck filesystem
            raise TimeoutError(
                f"checkpoint upload for step {step} exceeded "
                f"{self._save_timeout_s}s; directory left unmarked and will be "
                "ignored on resume"
            ) from error
        waited = time.monotonic() - started
        if waited > 1.0:
            # A long wait here means the previous save had not finished by the
            # time the next one started: IO is the bottleneck, not compute.
            _LOG.warning(
                "training stalled %.2fs waiting for checkpoint step=%s to upload; "
                "increase the save interval or the writer thread count",
                waited,
                step,
            )
        if step is not None and marker is not None:
            self._finalize(step, self._root / _step_dirname(step), marker)

    def _finalize(self, step: int, directory: Path, marker: dict[str, Any]) -> None:
        """Publish the completion marker, then prune, then synchronise.

        The barrier at the end is not decoration. Rank 0 deletes directories
        during pruning; a rank still reading one of them would see it vanish.
        Every rank leaves this method with the same view of the directory.
        """
        if _is_main_rank():
            payload = dict(marker)
            payload["completed_at"] = time.time()
            temporary = directory / f".{MARKER_FILENAME}.tmp"
            temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
            # `Path.replace` is an atomic rename within a filesystem: a reader
            # sees either no marker or a whole one, never a truncated document.
            temporary.replace(directory / MARKER_FILENAME)
            self._prune(keep_step=step)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def _build_marker(
        self,
        *,
        step: int,
        state: TrainState,
        entries: dict[str, object],
        data_world: int,
        parallel: ParallelModel | None,
        extras: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Assemble the self-describing metadata written beside the shards."""
        world_size = _world_size()
        marker: dict[str, Any] = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "step": step,
            "progress": dict(state.progress()),
            "keys": sorted(entries),
            "world_size": world_size,
            "data_world": data_world,
            "torch_version": torch.__version__,
        }
        if parallel is not None:
            marker["parallel"] = {
                "describe": parallel.dims.describe(),
                "dp_replicate": parallel.dims.dp_replicate,
                "dp_shard": parallel.dims.dp_shard,
                "context": parallel.dims.context,
                "tensor": parallel.dims.tensor,
                "pipeline": parallel.dims.pipeline,
                "applied": list(parallel.applied),
            }
        if extras:
            marker["extras"] = dict(extras)
        return marker

    # ------------------------------------------------------------------
    # Retention
    # ------------------------------------------------------------------

    def _prune(self, *, keep_step: int) -> None:
        """Delete checkpoints no policy asks to keep. Rank 0 only.

        ``keep_step`` is protected unconditionally, on top of every configured
        policy. Deleting the newest checkpoint is the one mistake a retention
        policy must never make: it turns a preemption from a lost hour into a
        lost run, and a misconfiguration that makes it possible (``keep_last_n``
        of zero meaning "keep none" instead of "keep all") is easy to write.
        """
        entries = self.list_checkpoints()
        if not entries:
            return
        keep: set[int] = {keep_step, entries[-1].step}
        if self._keep_last_n > 0:
            keep.update(entry.step for entry in entries[-self._keep_last_n :])
        else:
            # Zero means "unbounded", not "delete everything".
            keep.update(entry.step for entry in entries)
        if self._keep_every_n_steps > 0:
            keep.update(
                entry.step
                for entry in entries
                if entry.step % self._keep_every_n_steps == 0
            )
        for entry in entries:
            if entry.step not in keep:
                shutil.rmtree(entry.path, ignore_errors=True)
                _LOG.info("pruned checkpoint step=%d at %s", entry.step, entry.path)
        self._prune_incomplete(newest_complete=entries[-1].step, keep_step=keep_step)

    def _prune_incomplete(self, *, newest_complete: int, keep_step: int) -> None:
        """Remove abandoned partial directories older than the newest complete one.

        These are the debris of preempted jobs. They are never selected for
        resume, but they are full-size and they accumulate one per preemption,
        which on a large model is terabytes a month. Anything at or after the
        newest complete step is left alone — it may be an upload still in
        flight, in this process or in a concurrent evaluation job.
        """
        for candidate in self._scan_directories():
            step, directory = candidate
            if step >= newest_complete or step == keep_step:
                continue
            if (directory / MARKER_FILENAME).exists():
                continue
            shutil.rmtree(directory, ignore_errors=True)
            _LOG.warning("removed incomplete checkpoint step=%d at %s", step, directory)

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _scan_directories(
        self, path: str | os.PathLike[str] | None = None
    ) -> list[tuple[int, Path]]:
        root = Path(path) if path is not None else self._root
        if not root.is_dir():
            return []
        found: list[tuple[int, Path]] = []
        for child in root.iterdir():
            if not child.is_dir():
                continue
            step = _parse_step(child.name)
            if step is not None:
                found.append((step, child))
        found.sort(key=lambda item: item[0])
        return found

    def list_checkpoints(
        self, path: str | os.PathLike[str] | None = None
    ) -> tuple[CheckpointEntry, ...]:
        """Return every **complete** checkpoint, oldest first.

        A directory without a readable completion marker is skipped, silently
        and by design: an incomplete checkpoint must be invisible to everything
        that chooses what to resume from.

        Args:
            path: Directory to scan. Defaults to this manager's root.

        Returns:
            Complete checkpoints in ascending step order.
        """
        entries: list[CheckpointEntry] = []
        for step, directory in self._scan_directories(path):
            marker = directory / MARKER_FILENAME
            if not marker.is_file():
                continue
            try:
                metadata = json.loads(marker.read_text())
            except (OSError, json.JSONDecodeError):
                # A marker we cannot parse is not a marker. Treat the
                # checkpoint as incomplete rather than guessing at its layout.
                _LOG.warning("unreadable checkpoint marker at %s; skipping", marker)
                continue
            if not isinstance(metadata, dict):
                continue
            entries.append(
                CheckpointEntry(step=step, path=directory, metadata=metadata)
            )
        return tuple(entries)

    def latest_step(self, path: str | os.PathLike[str] | None = None) -> int | None:
        """Return the newest complete checkpoint's step, or ``None``.

        Args:
            path: Directory to scan. Defaults to this manager's root.

        Returns:
            The step, or ``None`` when there is nothing to resume from.
        """
        entries = self.list_checkpoints(path)
        return entries[-1].step if entries else None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(
        self,
        state: TrainState,
        *,
        parallel: ParallelModel | None = None,
        step: int | None = None,
    ) -> int | None:
        """Restore ``state`` in place from a checkpoint, resuming the latest.

        **Every rank must call this**, for the same reason as :meth:`save`.

        The load re-plans reads from the chunk offsets recorded at save time,
        so the rank count may differ from the one that wrote the checkpoint.
        That is the property the whole design exists to provide: a job preempted
        off 512 GPUs restarts on 256, or on 1024, with bit-identical parameters
        and correctly resharded optimizer moments.

        Generator state is the exception. It is per-data-rank and cannot be
        split or merged, so when the data-parallel width has changed this method
        skips it and leaves the streams as the caller seeded them — which should
        be :meth:`~avgen.core.rng.RNGStreams.for_rank` with the run's base seed.
        A warning is logged, because it means this resume is not bit-identical
        to the run it continues even though the weights are.

        Args:
            state: The training state to fill. Its model and optimizer must
                already be built and parallelised the way this run wants them,
                which is what makes resharding possible at all.
            parallel: The parallelised model, for data-parallel coordinates.
            step: Step to resume from. Defaults to the newest complete one.

        Returns:
            The step that was restored, or ``None`` when the directory holds no
            complete checkpoint. ``None`` is the normal answer for a fresh run,
            so a trainer can call this unconditionally.

        Raises:
            FileNotFoundError: If ``step`` was given explicitly and there is no
                complete checkpoint at it. An explicit request that cannot be
                satisfied is an error; an implicit one is a fresh start.
        """
        self.wait()
        entries = self.list_checkpoints()
        if step is None:
            if not entries:
                _LOG.info("no complete checkpoint under %s; starting fresh", self._root)
                return None
            entry = entries[-1]
        else:
            matches = [candidate for candidate in entries if candidate.step == step]
            if not matches:
                raise FileNotFoundError(
                    f"no complete checkpoint for step {step} under {self._root}"
                )
            entry = matches[0]

        version = int(entry.metadata.get("format_version", 0))
        if version > CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                f"checkpoint at {entry.path} uses format version {version}, "
                f"newer than the supported {CHECKPOINT_FORMAT_VERSION}"
            )

        data_rank, data_world = _data_coordinates(parallel)
        saved_keys = set(entry.metadata.get("keys", ()))
        include_rng = self._include_rng and RNG_KEY in saved_keys
        if include_rng and entry.data_world != data_world:
            _LOG.warning(
                "resuming at data-parallel width %d from a checkpoint written at "
                "%d; per-rank RNG state cannot be resharded and will be re-derived "
                "from the seed. Weights and optimizer state are unaffected.",
                data_world,
                entry.data_world,
            )
            include_rng = False

        entries_to_load = build_stateful(
            state,
            data_rank=data_rank,
            include_rng=include_rng,
            strict=self._strict,
            cpu_offload=self._cpu_offload,
        )
        # Asking DCP for a key the checkpoint does not have is a hard error, so
        # drop anything this run has that the checkpoint did not: adding an EMA
        # or a data cursor mid-project must not invalidate earlier checkpoints.
        for key in (SCHEDULE_KEY, EMA_KEY, EXTRAS_KEY, RNG_KEY):
            if key in entries_to_load and saved_keys and key not in saved_keys:
                _LOG.warning(
                    "component %r is active in this run but absent from the "
                    "checkpoint at %s; it keeps its freshly initialised state",
                    key,
                    entry.path,
                )
                entries_to_load.pop(key)
        required = {MODEL_OPTIMIZER_KEY, PROGRESS_KEY}
        missing = saved_keys - set(entries_to_load) - required
        if missing:
            _LOG.warning(
                "checkpoint at %s contains %s which this run does not use; skipped",
                entry.path,
                sorted(missing),
            )

        reader = dcp.FileSystemReader(entry.path)
        started = time.monotonic()
        with collective_timeout(self._save_timeout_s):
            dcp.load(entries_to_load, storage_reader=reader)
        _LOG.info(
            "restored checkpoint step=%d from %s in %.2fs (saved at world=%d, now %d)",
            entry.step,
            entry.path,
            time.monotonic() - started,
            entry.world_size,
            _world_size(),
        )
        return entry.step

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Drain any pending save. Always call this before the process exits.

        A background upload that is still running when the interpreter tears
        down loses the completion marker, which throws away a checkpoint that
        was very nearly finished.
        """
        self.wait()

    def __enter__(self) -> CheckpointManager:
        """Return self so the manager can guard a training loop."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Drain the pending save even when the loop is exiting on an error."""
        self.close()


def _upload_future(result: Any) -> Future[Any]:
    """Normalise the two shapes ``dcp.async_save`` can return.

    With a synchronous stager it returns the upload future directly; with an
    asynchronous ("zero overhead") stager it returns an ``AsyncSaveResponse``
    carrying separate staging and upload futures. Only the upload future gates
    the completion marker — staging is already done by the time either object
    reaches us, which is the property the whole async path depends on.
    """
    upload = getattr(result, "upload_completion", None)
    if upload is not None:
        return upload
    return result


def save(
    path: str | os.PathLike[str],
    state: TrainState,
    *,
    parallel: ParallelModel | None = None,
    async_save: bool = True,
    extras: dict[str, Any] | None = None,
) -> Path:
    """Write one checkpoint without configuring retention.

    A convenience for scripts and tests. A real training loop should hold a
    :class:`CheckpointManager`, because this function has to drain its
    single-use manager synchronously and therefore cannot overlap the upload
    with anything.

    Args:
        path: Root directory for checkpoints.
        state: The training state to save.
        parallel: The parallelised model, or ``None`` for single-replica.
        async_save: Whether to stage-then-upload. The upload is still awaited
            before this function returns.
        extras: JSON-serialisable metadata for the marker.

    Returns:
        The checkpoint directory.
    """
    manager = CheckpointManager(path, keep_last_n=0)
    directory = manager.save(
        state.step, state, parallel=parallel, async_save=async_save, extras=extras
    )
    manager.close()
    return directory


def load(
    path: str | os.PathLike[str],
    state: TrainState,
    *,
    parallel: ParallelModel | None = None,
    step: int | None = None,
) -> int | None:
    """Restore the latest (or a named) checkpoint from ``path`` in place.

    Args:
        path: Root directory for checkpoints.
        state: The training state to fill.
        parallel: The parallelised model, or ``None`` for single-replica.
        step: Step to resume from, or ``None`` for the newest complete one.

    Returns:
        The restored step, or ``None`` when there is nothing to resume from.
    """
    return CheckpointManager(path, keep_last_n=0).load(
        state, parallel=parallel, step=step
    )


def load_model(path: str | Path, model: nn.Module, *, strict: bool = True) -> None:
    """Load only the model weights from a training checkpoint.

    The full :func:`load` restores a whole :class:`~avgen.core.TrainState` —
    optimizer moments, RNG streams, progress counters — and needs a live
    optimizer and a parallelised model to do it. Everything downstream of
    training wants something smaller: a model, and the weights that go in it.
    Exporting a release artifact, running an evaluation, or serving a checkpoint
    all need exactly this and nothing else.

    Without it each caller reimplements a partial DCP load and gets a slightly
    different answer about what to do with a missing key.

    Args:
        path: Checkpoint directory written by :class:`CheckpointManager`.
        model: The model to load into, already constructed with the right
            architecture. Under FSDP its parameters may be DTensors; DCP
            reshards into whatever layout this model has.
        strict: Whether a key in the model but absent from the checkpoint is an
            error. Left true: a silently unfilled tensor is a model that
            produces plausible noise.

    Raises:
        FileNotFoundError: If the directory holds no readable checkpoint.
        RuntimeError: If the checkpoint's model entry cannot be read.
    """
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.api import CheckpointException
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        set_model_state_dict,
    )

    directory = Path(path)
    if not directory.is_dir():
        raise FileNotFoundError(f"no checkpoint directory at {directory}")

    from avgen.checkpoint.stateful import MODEL_OPTIMIZER_KEY

    options = StateDictOptions(full_state_dict=False, strict=strict)
    holder: dict[str, Any] = {
        MODEL_OPTIMIZER_KEY: {
            "model": get_model_state_dict(model, options=options),
        }
    }
    try:
        dcp.load(holder, checkpoint_id=str(directory))
    except (Exception, CheckpointException) as error:
        # CheckpointException derives from BaseException, not Exception, so it
        # slips past every ordinary handler including the CLI's. Naming it here
        # is what turns "missing key in checkpoint state_dict" into a sentence
        # about this checkpoint.
        raise RuntimeError(
            f"could not read model weights from {directory}: {error}"
        ) from error
    set_model_state_dict(model, holder[MODEL_OPTIMIZER_KEY]["model"], options=options)


def load_ema(path: str | Path, model: nn.Module, *, decay: float = 0.9999) -> None:
    """Load a checkpoint's EMA weights **into** the given model, in place.

    Published diffusion samples come from EMA weights; the raw training weights
    are visibly worse. Exporting a release artifact or reproducing a sample
    therefore needs the average, not the live parameters — and the average is
    stored under its own checkpoint entry, shaped like the EMA's state rather
    than like a model state dict.

    The shadow buffers are materialised from ``model`` before the load, which is
    what gives DCP a target to reshard into: the checkpoint may have been
    written at a different rank count, and it is the live parameter layout that
    decides the destination sharding.

    Args:
        path: Checkpoint directory written by :class:`CheckpointManager`.
        model: The model to overwrite with the averaged weights. Must have the
            architecture the checkpoint was written from.
        decay: Decay used to construct the tracker. Immaterial to a load — the
            stored tensors are copied verbatim — but it has to be positive for
            the tracker to hold any shadows at all.

    Raises:
        FileNotFoundError: If the directory holds no readable checkpoint.
        RuntimeError: If the checkpoint carries no EMA entry. That is a real
            answer, not a fallback: silently exporting the live weights when the
            caller asked for the average produces a worse model with no sign
            that anything went wrong.
    """
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.api import CheckpointException

    from avgen.train.ema import ShardedEMA

    directory = Path(path)
    if not directory.is_dir():
        raise FileNotFoundError(f"no checkpoint directory at {directory}")

    ema = ShardedEMA(model, decay=decay)
    holder: dict[str, Any] = {EMA_KEY: ema.state_dict()}
    try:
        dcp.load(holder, checkpoint_id=str(directory))
    except (Exception, CheckpointException) as error:
        raise RuntimeError(
            f"could not read EMA weights from {directory}: {error}. A checkpoint "
            "saved with train.ema_decay = 0 carries none; export without --ema, "
            "or point at a checkpoint that has them."
        ) from error
    ema.load_state_dict(holder[EMA_KEY])
    ema.copy_to(model)
