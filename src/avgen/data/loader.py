"""The resumable, rank-sharded batch loader.

Two rules govern this file. Both are invisible when broken, which is why they
are stated here and repeated at the code that implements them.

**Rule one: shard on ``data_rank``, and on nothing else.**

A five-dimensional parallel job has ranks that hold *different samples* and
ranks that hold *pieces of the same sample*. Data-parallel ranks are the first
kind. Context-parallel, tensor-parallel, and pipeline ranks are the second: they
hold shards of one sample's sequence, one layer's weights, or one stage's
depth, and every one of them must receive **byte-identical batches**.

Sharding on the global rank instead is the single most common correctness bug in
a multi-dimensional trainer, and it produces no error. With ``cp=8``, the eight
CP ranks would each get a different sample, then each compute an attention shard
of a *different* sequence, then all-gather them into a sequence that is eight
unrelated clips glued together. The loss goes down. The gradients are garbage.
Nothing in any log says so.

The mechanism here is deliberately the simplest one that cannot get this wrong:
**every rank plans the entire epoch identically**, from the same seed and the
same corpus, and then takes a strided slice by ``data_rank``. Ranks sharing a
``data_rank`` compute the same plan and take the same slice, so they cannot
diverge; ranks with different ``data_rank`` take disjoint slices, so no sample is
seen twice per step. There is no broadcast, no rank-zero coordinator, and no
opportunity for the ranks to disagree.

**Rule two: the cursor must round-trip exactly.**

A run that resumes with the right weights and the wrong data position retrains
on samples it has already seen. This is not a small error and it is not a loud
one. The loss curve looks *better* than it should, because a model re-shown data
it has memorised fits it faster; the training metric improves while the
validation gap widens, and by the time that shows up the run has burned days.

So the cursor is a plain integer count into a plan that is itself a pure
function of ``(seed, epoch, step, corpus)``. Resuming does not restore an
iterator's internal state — it recomputes the plan and skips forward. That is
slower to start by milliseconds and correct by construction, which is the right
trade at any scale.
"""

from __future__ import annotations

import hashlib
import queue
import threading
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from avgen.core._validate import require_dimension, require_positive
from avgen.core.batch import MediaBatch
from avgen.core.tensors import TensorBundle
from avgen.data.bucket import BucketBatch, BucketCurriculum, BucketSampler
from avgen.data.protocols import (
    DATA_SCHEMA_VERSION,
    Bucket,
    BucketPlan,
    SampleDescriptor,
    SampleStore,
    collate_samples,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance for type checking
    from torch.distributed.device_mesh import DeviceMesh

    from avgen.parallel.dims import ParallelDims

__all__ = [
    "LoaderConfig",
    "LoaderCursor",
    "ShardedLoader",
    "assign_buckets",
    "bucket_from_descriptor",
    "build_loader",
]


def bucket_from_descriptor(
    descriptor: SampleDescriptor,
    *,
    bucket_id: int = 0,
    patch_frames: int = 1,
    patch_height: int = 2,
    patch_width: int = 2,
    audio_patch_frames: int = 1,
) -> Bucket:
    """Derive a single bucket matching a sample's geometry.

    The convenience path for a homogeneous corpus — the synthetic source, a
    fixed-resolution benchmark, a smoke test — where declaring a bucket plan by
    hand would be ceremony.

    Args:
        descriptor: Sample whose geometry defines the bucket.
        bucket_id: Identifier for the derived bucket.
        patch_frames: Temporal patch size the model will apply.
        patch_height: Spatial patch height the model will apply.
        patch_width: Spatial patch width the model will apply.
        audio_patch_frames: Temporal patch size for audio.

    Returns:
        A bucket with exactly this sample's latent geometry.
    """
    return Bucket(
        bucket_id=bucket_id,
        frames=descriptor.video_frames,
        height=descriptor.height,
        width=descriptor.width,
        audio_frames=descriptor.audio_frames,
        patch_frames=patch_frames,
        patch_height=patch_height,
        patch_width=patch_width,
        audio_patch_frames=audio_patch_frames,
    )


def assign_buckets(store: SampleStore, plan: BucketPlan) -> tuple[int, ...]:
    """Map every sample in a store to the bucket whose geometry it matches.

    Matching is **exact**, not nearest. By the time latents are on disk the
    resizing decision has already been made — that happened in
    :mod:`avgen.data.ingest`, against :meth:`BucketPlan.nearest` — so a stored
    sample that does not match any bucket exactly means the plan and the corpus
    were built from different configurations. Silently rounding it to the
    nearest bucket would produce a batch whose samples have different shapes,
    which :func:`~avgen.data.protocols.collate_samples` would then reject with a
    much less informative message.

    Reads descriptors only; no tensor is touched, so this is cheap even on a
    corpus that does not fit in memory.

    Args:
        store: Corpus to classify.
        plan: Buckets to classify into.

    Returns:
        One bucket id per sample, in global index order.

    Raises:
        ValueError: If a sample matches no bucket.
    """
    geometry: dict[tuple[int, int, int, int], int] = {}
    for bucket in plan:
        geometry[(bucket.frames, bucket.height, bucket.width, bucket.audio_frames)] = (
            bucket.bucket_id
        )
    assignments: list[int] = []
    for index in range(len(store)):
        info = store.descriptor(index)
        key = (info.video_frames, info.height, info.width, info.audio_frames)
        bucket_id = geometry.get(key)
        if bucket_id is None:
            suggestion = plan.nearest(
                frames=info.video_frames, height=info.height, width=info.width
            )
            raise ValueError(
                f"sample {info.sample_id} has latent geometry {key} which matches "
                "no bucket in the plan; the nearest is "
                f"{suggestion.describe()}. Re-ingest against this plan, or add a "
                "bucket for this geometry."
            )
        assignments.append(bucket_id)
    return tuple(assignments)


@dataclass(frozen=True, slots=True)
class LoaderConfig:
    """Behavioural knobs for :class:`ShardedLoader`.

    Args:
        batch_size: Samples per batch **per data rank**, not per job. The global
            batch is this times the data-parallel degree times the gradient
            accumulation steps; see
            :meth:`avgen.parallel.dims.ParallelDims.gradient_accumulation_for`.
        seed: Base seed. Combined with the epoch to derive every shuffle.
        max_epochs: Stop after this many passes, or ``None`` to stream forever.
            Forever is the usual setting for a large corpus: the run ends on a
            step budget, not on a data budget.
        device: Device to prefetch onto, or ``None`` to leave batches on CPU.
        pin_memory: Whether to stage batches in pinned host memory. Pinned
            memory is what makes a host-to-device copy asynchronous; from
            pageable memory the copy is synchronous no matter what
            ``non_blocking`` says, and the whole prefetch pipeline collapses
            into a serial one.
        prefetch_depth: Batches to keep in flight ahead of the consumer. Two is
            enough to hide collation behind a step; more buys nothing and costs
            host memory proportional to the batch size.
        prefetch: Whether to run collation on a background thread at all.
            Disabling it makes iteration single-threaded and easier to debug.
        allow_reshard: Whether resuming with a different data-parallel degree is
            permitted. It changes the sample order, so it is off by default and
            must be an explicit, logged decision.
        schema_version: Version stamped into emitted batch specs.

    Raises:
        ValueError: If a size is non-positive or ``max_epochs`` is negative.
    """

    batch_size: int
    seed: int = 0
    max_epochs: int | None = None
    device: str | None = None
    pin_memory: bool = True
    prefetch_depth: int = 2
    prefetch: bool = True
    allow_reshard: bool = False
    schema_version: int = DATA_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate sizes and the epoch budget."""
        require_positive("batch_size", self.batch_size)
        require_positive("prefetch_depth", self.prefetch_depth)
        require_dimension("seed", self.seed, allow_zero=True)
        require_positive("schema_version", self.schema_version)
        if self.max_epochs is not None:
            require_positive("max_epochs", self.max_epochs)


class LoaderCursor:
    """Position in a data stream, expressed as counts rather than iterator state.

    Implements :class:`avgen.core.state.DataCursor`, so the checkpoint layer
    saves it alongside the model without knowing what it is.

    Everything here is a small integer. That is the design decision: the
    alternative is serialising a generator or a shuffled index list, which is
    fragile across versions, large on a big corpus, and impossible to inspect
    when a resume goes wrong. Counts into a reproducible plan are none of those
    things.

    Args:
        epoch: Completed passes over the corpus.
        batch_index: Batches consumed within the current epoch, counted in this
            rank's own slice of the plan.
        samples_seen: Samples consumed by this rank across the whole run.
        step: Optimizer step, used to evaluate the bucket curriculum. Owned by
            the trainer and pushed in; the loader cannot know it otherwise.
    """

    __slots__ = ("batch_index", "epoch", "samples_seen", "step")

    def __init__(
        self,
        *,
        epoch: int = 0,
        batch_index: int = 0,
        samples_seen: int = 0,
        step: int = 0,
    ) -> None:
        self.epoch = epoch
        self.batch_index = batch_index
        self.samples_seen = samples_seen
        self.step = step

    def advance(self, samples: int) -> None:
        """Record that more samples have been consumed.

        Args:
            samples: Number of samples consumed.

        Raises:
            ValueError: If ``samples`` is negative.
        """
        require_dimension("samples", samples, allow_zero=True)
        self.samples_seen += samples

    def state_dict(self) -> dict[str, Any]:
        """Return the cursor as a plain mapping."""
        return {
            "epoch": self.epoch,
            "batch_index": self.batch_index,
            "samples_seen": self.samples_seen,
            "step": self.step,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a cursor produced by :meth:`state_dict`.

        Args:
            state: Mapping produced by :meth:`state_dict`.

        Raises:
            KeyError: If a counter is missing.
        """
        for name in ("epoch", "batch_index", "samples_seen", "step"):
            if name not in state:
                raise KeyError(f"missing cursor field {name!r}")
            setattr(self, name, int(state[name]))

    def __repr__(self) -> str:
        """Return a compact developer representation."""
        return (
            f"LoaderCursor(epoch={self.epoch}, batch_index={self.batch_index}, "
            f"samples_seen={self.samples_seen}, step={self.step})"
        )


class ShardedLoader:
    """A resumable, curriculum-aware, rank-sharded stream of media batches.

    Args:
        store: Random-access corpus.
        sampler: Bucket sampler that plans each epoch.
        config: Behavioural knobs.
        data_rank: This rank's index within the data-parallel dimension, from
            :meth:`avgen.parallel.dims.ParallelDims.data_coordinates`.
        data_world: Size of the data-parallel dimension, from the same call.

    Raises:
        ValueError: If the rank coordinates are inconsistent, or the corpus
            cannot fill one batch per data rank.
    """

    __slots__ = (
        "_config",
        "_cursor",
        "_data_rank",
        "_data_world",
        "_fingerprint",
        "_sampler",
        "_store",
    )

    def __init__(
        self,
        store: SampleStore,
        sampler: BucketSampler,
        *,
        config: LoaderConfig,
        data_rank: int = 0,
        data_world: int = 1,
    ) -> None:
        require_positive("data_world", data_world)
        require_dimension("data_rank", data_rank, allow_zero=True)
        if data_rank >= data_world:
            raise ValueError(
                f"data_rank={data_rank} must be below data_world={data_world}"
            )
        if sampler.batches_per_epoch() < data_world:
            raise ValueError(
                f"the corpus yields {sampler.batches_per_epoch()} batches per epoch "
                f"but there are {data_world} data-parallel ranks; every rank must "
                "get at least one batch or the job deadlocks at the first "
                "collective"
            )
        self._store = store
        self._sampler = sampler
        self._config = config
        self._data_rank = data_rank
        self._data_world = data_world
        self._cursor = LoaderCursor()
        self._fingerprint = _corpus_fingerprint(store, sampler)

    @property
    def cursor(self) -> LoaderCursor:
        """The live cursor. Mutating it changes where the next iteration starts."""
        return self._cursor

    @property
    def sampler(self) -> BucketSampler:
        """The bucket sampler planning each epoch."""
        return self._sampler

    @property
    def config(self) -> LoaderConfig:
        """The loader's configuration."""
        return self._config

    @property
    def data_coordinates(self) -> tuple[int, int]:
        """This loader's ``(data_rank, data_world)``."""
        return self._data_rank, self._data_world

    def set_step(self, step: int) -> None:
        """Tell the loader which optimizer step training is on.

        Only the curriculum uses this, and it is read once per epoch, so pushing
        it every step is unnecessary — pushing it before each epoch boundary is
        enough. It is stored in the cursor so a resumed run evaluates the
        curriculum at the step it left off at rather than at zero.

        Args:
            step: Current optimizer step.

        Raises:
            ValueError: If ``step`` is negative.
        """
        require_dimension("step", step, allow_zero=True)
        self._cursor.step = step

    def batches_per_epoch(self) -> int:
        """Return how many batches **this rank** yields per epoch.

        Every rank yields the same count. The plan is truncated to a multiple of
        ``data_world`` before slicing, so ranks never disagree about how many
        steps an epoch has — a disagreement that surfaces as a hang at the next
        collective, minutes after the rank that ran out went quiet.
        """
        return self._sampler.batches_per_epoch() // self._data_world

    def epoch_plan(
        self, epoch: int, *, step: int | None = None
    ) -> tuple[BucketBatch, ...]:
        """Return this rank's batches for one epoch, in order.

        Args:
            epoch: Pass number over the corpus.
            step: Optimizer step used to evaluate the curriculum; defaults to
                the cursor's step.

        Returns:
            This rank's slice of the global epoch plan.
        """
        resolved_step = self._cursor.step if step is None else step
        global_plan = self._sampler.plan_epoch(epoch=epoch, step=resolved_step)
        # ---------------------------------------------------------------
        # THE RULE. Every rank in the job just computed `global_plan`, and
        # computed it identically: plan_epoch is a pure function of
        # (seed, epoch, step, assignments), none of which vary by rank.
        #
        # The slice below is the ONLY place rank enters the data path, and it
        # uses data_rank -- never the global rank, never dist.get_rank().
        # Context-, tensor-, and pipeline-parallel ranks share a data_rank, so
        # they take the identical slice and receive byte-identical batches,
        # which is what they need because they hold shards of the SAME sample.
        #
        # Truncating to a multiple of data_world first keeps every rank's slice
        # the same length. Without the truncation, the first few ranks get one
        # extra batch and the job hangs on the collective at the end of the
        # epoch.
        # ---------------------------------------------------------------
        usable = (len(global_plan) // self._data_world) * self._data_world
        return global_plan[self._data_rank : usable : self._data_world]

    def build_batch(self, planned: BucketBatch) -> MediaBatch:
        """Materialise one planned batch from the store.

        Args:
            planned: Sample indices and their bucket.

        Returns:
            The dense batch, validated, on CPU.
        """
        samples = [self._store[index] for index in planned.indices]
        batch = collate_samples(
            samples,
            bucket_id=planned.bucket_id,
            schema_version=self._config.schema_version,
        )
        # Validation at the boundary, once per batch, never in the hot loop.
        # This is the last point where a shape bug is attributable to the data.
        batch.validate()
        return batch

    def __iter__(self) -> Iterator[MediaBatch]:
        """Yield batches from the cursor's position, advancing it as it goes.

        Yields:
            Batches on the configured device.
        """
        source = self._plan_iterator()
        if self._config.prefetch:
            source = _prefetch(source, depth=self._config.prefetch_depth)
        device = (
            torch.device(self._config.device)
            if self._config.device is not None
            else None
        )
        for batch in source:
            if device is not None:
                # The host-to-device copy is issued HERE, on the consumer's
                # thread and therefore on the consumer's stream. Issuing it in
                # the producer thread would put it on that thread's current
                # stream, and the compute stream would then read the destination
                # before the copy retired unless an explicit event was recorded.
                # Keeping the copy on the consumer side removes that hazard
                # entirely at no cost.
                yield batch.to(device, non_blocking=True)
            else:
                yield batch

    def _plan_iterator(self) -> Iterator[MediaBatch]:
        """Walk epochs and batches from the cursor, doing all CPU-side work.

        Yields:
            Pinned CPU batches.
        """
        config = self._config
        while config.max_epochs is None or self._cursor.epoch < config.max_epochs:
            planned = self.epoch_plan(self._cursor.epoch)
            while self._cursor.batch_index < len(planned):
                entry = planned[self._cursor.batch_index]
                batch = self.build_batch(entry)
                if config.pin_memory:
                    batch = _pin_batch(batch)
                self._cursor.batch_index += 1
                self._cursor.advance(len(entry))
                yield batch
            self._cursor.epoch += 1
            self._cursor.batch_index = 0

    # ------------------------------------------------------------------
    # Stateful
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Return everything needed to resume this stream at the same sample."""
        return {
            "schema_version": DATA_SCHEMA_VERSION,
            "cursor": self._cursor.state_dict(),
            "seed": self._config.seed,
            "batch_size": self._config.batch_size,
            "data_rank": self._data_rank,
            "data_world": self._data_world,
            "corpus_fingerprint": self._fingerprint,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a cursor, refusing any change that would alter sample order.

        Three things are checked, and each one is fatal by default because each
        one silently changes *which samples the run sees next* while leaving
        every observable metric looking normal:

        * **The seed**, because it seeds every shuffle.
        * **The corpus fingerprint**, because adding or removing shards
          renumbers the global index space, so the same cursor points at
          different data.
        * **The data-parallel degree and batch size**, because they determine
          how the plan is sliced. Resharding is sometimes genuinely wanted, so
          :attr:`LoaderConfig.allow_reshard` permits it — but as a decision
          someone made and can find in a log, not as a default.

        Args:
            state: Mapping produced by :meth:`state_dict`.

        Raises:
            ValueError: On a mismatch that would change the sample order.
            KeyError: If the state is missing a required field.
        """
        recorded_seed = int(state["seed"])
        if recorded_seed != self._config.seed:
            raise ValueError(
                f"cannot resume: checkpoint was written with seed={recorded_seed} "
                f"but this loader has seed={self._config.seed}; every shuffle would "
                "differ and the run would silently revisit data it has seen"
            )
        recorded_fingerprint = str(state["corpus_fingerprint"])
        if recorded_fingerprint != self._fingerprint:
            raise ValueError(
                "cannot resume: the corpus changed since the checkpoint "
                f"({recorded_fingerprint[:16]} -> {self._fingerprint[:16]}). The "
                "cursor is an index into a plan over this exact corpus, so it now "
                "points at different samples."
            )
        recorded_world = int(state["data_world"])
        recorded_batch = int(state["batch_size"])
        if (
            recorded_world != self._data_world
            or recorded_batch != self._config.batch_size
        ) and not self._config.allow_reshard:
            raise ValueError(
                "cannot resume: the checkpoint was written with "
                f"data_world={recorded_world}, batch_size={recorded_batch} but this "
                f"loader has data_world={self._data_world}, "
                f"batch_size={self._config.batch_size}. The epoch plan is sliced by "
                "both, so the resumed order would not match. Set "
                "LoaderConfig.allow_reshard=True to accept an approximate resume."
            )
        self._cursor.load_state_dict(state["cursor"])
        resliced = (
            recorded_world != self._data_world
            or recorded_batch != self._config.batch_size
        )
        if resliced:
            # Approximate resume: the within-epoch position is meaningless under
            # a different slicing, so keep the epoch (which controls the shuffle)
            # and restart the epoch rather than pretending to land on the same
            # sample.
            self._cursor.batch_index = 0


def _corpus_fingerprint(store: SampleStore, sampler: BucketSampler) -> str:
    """Return a digest identifying the corpus and the plan drawn over it.

    Built from the corpus size, the codec identity, the bucket geometry, and the
    bucket populations — everything that changes the meaning of a cursor. It
    deliberately does *not* read the sample tensors: this runs at loader
    construction on every rank of every job, and a full pass over the corpus
    would make startup time scale with dataset size.

    Args:
        store: The corpus.
        sampler: The sampler drawing from it.

    Returns:
        A hex digest.
    """
    digest = hashlib.sha256()
    digest.update(str(len(store)).encode("utf-8"))
    if len(store) > 0:
        digest.update(repr(store.descriptor(0).codec_key).encode("utf-8"))
    for bucket in sampler.plan:
        digest.update(bucket.describe().encode("utf-8"))
        digest.update(str(sampler.pool_size(bucket.bucket_id)).encode("utf-8"))
    digest.update(str(sampler.batch_size).encode("utf-8"))
    return digest.hexdigest()


def _pin_batch(batch: MediaBatch) -> MediaBatch:
    """Stage a batch's device-bound tensors in pinned host memory.

    Pinning is what makes ``non_blocking=True`` mean anything: a copy out of
    pageable memory is synchronous regardless of the flag, because the driver
    has to stage it through a pinned bounce buffer first. Without this the
    prefetch thread and the compute stream serialise and the loader stops
    hiding any latency at all.

    A no-op when CUDA is unavailable, so the same code path runs on a CPU-only
    machine. ``sample_ids`` are left alone: they never move to the device.

    Args:
        batch: Batch to pin.

    Returns:
        The pinned batch, or the original when pinning is unavailable.
    """
    if not torch.cuda.is_available():
        return batch

    def _pin(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.pin_memory() if not tensor.is_pinned() else tensor

    return MediaBatch(
        video=_pin(batch.video),
        audio=_pin(batch.audio),
        text=_pin(batch.text),
        video_mask=_pin(batch.video_mask),
        audio_mask=_pin(batch.audio_mask),
        video_positions=_pin(batch.video_positions),
        audio_positions=_pin(batch.audio_positions),
        sample_ids=batch.sample_ids,
        spec=batch.spec,
        text_mask=_pin(batch.text_mask),
        targets=TensorBundle(tuple(_pin(value) for value in batch.targets.values)),
    )


def _prefetch(source: Iterator[MediaBatch], *, depth: int) -> Iterator[MediaBatch]:
    """Run a batch iterator on a background thread with a bounded queue.

    One producer thread, not a pool. Order is part of the contract — the
    resumable cursor is an index into a fixed sequence — and a pool would
    reorder batches under load, which would make a resumed run diverge from the
    run it is resuming. A single thread preserves order exactly, and since the
    work here is tensor copies that release the GIL, one thread is enough to
    keep a step's worth of collation off the critical path.

    Args:
        source: Iterator producing batches.
        depth: Maximum batches held in the queue.

    Yields:
        Batches in the source's order.

    Raises:
        BaseException: Whatever the source raised, re-raised on the consumer's
            thread so a data error is not swallowed by a dying daemon.
    """
    channel: queue.Queue[tuple[MediaBatch | None, BaseException | None]] = queue.Queue(
        maxsize=depth
    )
    stop = threading.Event()

    def _produce() -> None:
        try:
            for batch in source:
                if stop.is_set():
                    break
                channel.put((batch, None))
        except BaseException as error:  # forwarded to the consumer's thread
            channel.put((None, error))
        finally:
            channel.put((None, None))

    worker = threading.Thread(target=_produce, name="avgen-data-prefetch", daemon=True)
    worker.start()
    try:
        while True:
            batch, error = channel.get()
            if error is not None:
                raise error
            if batch is None:
                return
            yield batch
    finally:
        # Tell the producer to stop and drain whatever it already queued, so the
        # thread cannot block forever on a full queue when the consumer walks
        # away mid-epoch (an early `break`, or an exception in the training step).
        #
        # Draining once is not enough, and the failure is nasty. The producer may
        # be blocked inside `channel.put` at the moment we drain; it then wakes,
        # puts one more batch, and only notices `stop` on the following
        # iteration. If the generator has already returned by then, a live daemon
        # thread is still holding memory-mapped shard tensors when the
        # interpreter tears down, and the C++ runtime aborts the process with
        # "terminate called without an active exception" — exit code 134 after a
        # training run that actually succeeded. Any scheduler or CI job reads
        # that as a failed job.
        #
        # So drain *and* join: keep making room until the thread has genuinely
        # finished. This terminates because the producer breaks out of its loop
        # on the first `stop` check after its pending put completes.
        stop.set()
        while worker.is_alive():
            try:
                channel.get_nowait()
            except queue.Empty:
                worker.join(timeout=0.05)
        while not channel.empty():
            channel.get_nowait()


def build_loader(
    store: SampleStore,
    *,
    batch_size: int,
    plan: BucketPlan | None = None,
    assignments: Sequence[int] | None = None,
    curriculum: BucketCurriculum | None = None,
    data_rank: int = 0,
    data_world: int = 1,
    dims: ParallelDims | None = None,
    mesh: DeviceMesh | None = None,
    seed: int = 0,
    max_epochs: int | None = None,
    device: torch.device | str | None = None,
    pin_memory: bool = True,
    prefetch: bool = True,
    prefetch_depth: int = 2,
    allow_oversampling: bool = True,
    allow_reshard: bool = False,
    schema_version: int = DATA_SCHEMA_VERSION,
) -> ShardedLoader:
    """Assemble a resumable data source over a corpus.

    Rank coordinates come from one of two places, and passing both is an error:
    give ``data_rank``/``data_world`` directly, or give ``dims`` and ``mesh``
    and let :meth:`~avgen.parallel.dims.ParallelDims.data_coordinates` derive
    them. The second form is preferred in a real job, because it is the call
    that encodes the "data-parallel coordinate only" rule; deriving the
    coordinates by hand from a global rank is exactly the mistake this loader
    exists to prevent.

    Args:
        store: Random-access corpus. A
            :class:`~avgen.data.shard.ConcatShardReader` or a
            :class:`~avgen.data.synthetic.SyntheticSource` both qualify.
        batch_size: Samples per batch per data rank.
        plan: Bucket plan. Defaults to a single bucket matching the first
            sample's geometry, which is right for a homogeneous corpus and
            wrong for anything else.
        assignments: Bucket id per sample. Defaults to
            :func:`assign_buckets`.
        curriculum: Optional step-varying bucket mixture.
        data_rank: This rank's data-parallel coordinate.
        data_world: Size of the data-parallel dimension.
        dims: Parallelism degrees, used with ``mesh`` to derive the
            coordinates.
        mesh: The device mesh, used with ``dims``.
        seed: Base seed for every shuffle.
        max_epochs: Passes to run, or ``None`` to stream forever.
        device: Device to prefetch onto, or ``None`` for CPU.
        pin_memory: Whether to stage batches in pinned host memory.
        prefetch: Whether to collate on a background thread.
        prefetch_depth: Batches to keep in flight.
        allow_oversampling: Whether a bucket may be drawn from more often than
            it has samples, which a curriculum normally needs.
        allow_reshard: Whether resuming under a different rank layout is
            permitted.
        schema_version: Version stamped into emitted batch specs.

    Returns:
        The loader, positioned at epoch zero.

    Raises:
        ValueError: If the corpus is empty, both coordinate forms are given, or
            only one of ``dims`` and ``mesh`` is given.
    """
    if len(store) == 0:
        raise ValueError("build_loader requires a non-empty corpus")
    if (dims is None) != (mesh is None):
        raise ValueError(
            "dims and mesh must be given together; data_coordinates needs both"
        )
    if dims is not None and mesh is not None:
        if (data_rank, data_world) != (0, 1):
            raise ValueError(
                "pass either dims/mesh or data_rank/data_world, not both; two "
                "sources of truth for the data coordinate is how they end up "
                "disagreeing"
            )
        data_rank, data_world = dims.data_coordinates(mesh)

    resolved_plan = plan or BucketPlan(
        buckets=(bucket_from_descriptor(store.descriptor(0)),)
    )
    resolved_assignments = (
        tuple(assignments)
        if assignments is not None
        else assign_buckets(store, resolved_plan)
    )
    sampler = BucketSampler(
        resolved_plan,
        resolved_assignments,
        batch_size=batch_size,
        seed=seed,
        curriculum=curriculum,
        allow_oversampling=allow_oversampling,
    )
    config = LoaderConfig(
        batch_size=batch_size,
        seed=seed,
        max_epochs=max_epochs,
        device=None if device is None else str(device),
        pin_memory=pin_memory,
        prefetch=prefetch,
        prefetch_depth=prefetch_depth,
        allow_reshard=allow_reshard,
        schema_version=schema_version,
    )
    return ShardedLoader(
        store,
        sampler,
        config=config,
        data_rank=data_rank,
        data_world=data_world,
    )
