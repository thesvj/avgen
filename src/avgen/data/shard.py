"""The on-disk latent shard format: container, manifest, and commit marker.

A shard is a directory holding exactly three files:

============================  ================================================
``data.safetensors``          Every tensor of every sample, in one container.
``manifest.json``             Shapes, provenance, and the container's digest.
``COMMIT``                    Written last. Its presence means "readable".
============================  ================================================

**Why a commit marker.** Preemption is the normal case, not the exception. A
shard writer on a spot instance or a scheduler with a wall clock will be killed
mid-write, and it will be killed at the worst possible moment often enough that
"often enough" is every day. Without a marker, the next reader finds a
plausible-looking file that is short by a few megabytes, and either crashes with
an inscrutable deserialisation error or — much worse — reads a truncated tensor
as valid data. With a marker, a half-written shard is simply invisible: the data
and manifest land under temporary names, are flushed to stable storage, are
renamed into place, and only then does a small ``COMMIT`` file appear. Rename is
atomic within a filesystem, so at no instant does a reader see a partially
written shard. A crash leaves stray temporary files and nothing else, and
re-running the writer is safe.

**Why safetensors.** The container needs three properties: it must be
mmap-able so a reader touches only the samples it uses rather than paging in the
whole shard; it must be zero-copy so a batch does not cost a full memcpy of
every latent; and it must have no code execution path, because a shard is a file
that gets copied between clusters and ``pickle`` is a remote code execution
primitive. ``safetensors`` is exactly those three properties, and using its
format rather than a bespoke one means shards are readable by anything in the
ecosystem.

**Why the digests.** A shard is written once and read ten thousand times, often
after being copied across a network or sitting on a disk for a year. Silent
corruption at that scale is not hypothetical. Two digests are recorded — one
over the container bytes, one over the manifest bytes — and
:func:`validate_shard` checks both. The manifest digest lives in ``COMMIT`` and
the data digest lives in the manifest, so the chain is anchored by the file that
was written last.

**Why unknown manifest keys survive a round trip.** A shard outlives the version
of avgen that wrote it. A newer writer records a field this reader has never
heard of; dropping it would mean that reading and rewriting a shard silently
destroys information. Unknown keys are collected into ``extra`` and re-emitted.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, Self, cast

import torch

from avgen.core._validate import require_dimension, require_positive
from avgen.core.tensors import TensorBundle, TensorBundleSpec
from avgen.data.protocols import (
    DATA_SCHEMA_VERSION,
    LatentSample,
    SampleDescriptor,
)

__all__ = [
    "COMMIT_FILENAME",
    "DATA_FILENAME",
    "MANIFEST_FILENAME",
    "SHARD_SCHEMA_VERSION",
    "ConcatShardReader",
    "ShardCorruptionError",
    "ShardManifest",
    "ShardReader",
    "ShardRecord",
    "validate_shard",
    "write_shard",
]

#: Version of the shard layout. Bumped only when an old reader would
#: misinterpret a new shard; adding an optional manifest key does not qualify,
#: because unknown keys are preserved rather than misread.
SHARD_SCHEMA_VERSION: int = 1

DATA_FILENAME = "data.safetensors"
MANIFEST_FILENAME = "manifest.json"
COMMIT_FILENAME = "COMMIT"

#: Digesting is done in chunks so a shard larger than memory can still be
#: validated. 4 MiB is comfortably above the point where syscall overhead stops
#: mattering and well below anything that would strain a page cache.
_DIGEST_CHUNK_BYTES = 4 * 1024 * 1024

#: Fields stored per sample. Order is fixed so the container's key order is
#: reproducible, which makes two shards written from the same samples
#: byte-identical and therefore comparable by digest.
_TENSOR_FIELDS: tuple[str, ...] = (
    "video",
    "audio",
    "text",
    "text_mask",
    "video_positions",
    "audio_positions",
)


class ShardCorruptionError(RuntimeError):
    """Raised when a shard fails a structural or digest check.

    A distinct type rather than a bare ``RuntimeError`` because the correct
    response is specific: quarantine this shard, log its path, and continue with
    the rest of the corpus. A training job that dies because one shard out of
    fifty thousand rotted is a job that will not finish.
    """


@dataclass(frozen=True, slots=True)
class ShardRecord:
    """Manifest entry describing one stored sample.

    Args:
        descriptor: Shapes, timebases, and codec fingerprints.
        valid_video_frames: Leading video frames that are real content, or
            ``-1`` when every frame is.
        valid_audio_frames: Same, for audio.
        target_spec: Spec interpreting this sample's auxiliary target tensors.
        extra: Manifest keys this version of avgen does not recognise,
            preserved verbatim.
    """

    descriptor: SampleDescriptor
    valid_video_frames: int = -1
    valid_audio_frames: int = -1
    target_spec: TensorBundleSpec = field(default_factory=TensorBundleSpec)
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation, unknown keys included."""
        known: dict[str, Any] = {
            "sample_id": self.descriptor.sample_id,
            "video_frames": self.descriptor.video_frames,
            "height": self.descriptor.height,
            "width": self.descriptor.width,
            "audio_frames": self.descriptor.audio_frames,
            "text_tokens": self.descriptor.text_tokens,
            "text_width": self.descriptor.text_width,
            "video_channels": self.descriptor.video_channels,
            "audio_channels": self.descriptor.audio_channels,
            "video_timebase_num": self.descriptor.video_timebase_num,
            "video_timebase_den": self.descriptor.video_timebase_den,
            "audio_timebase_num": self.descriptor.audio_timebase_num,
            "audio_timebase_den": self.descriptor.audio_timebase_den,
            "video_codec_id": self.descriptor.video_codec_id,
            "audio_codec_id": self.descriptor.audio_codec_id,
            "valid_video_frames": self.valid_video_frames,
            "valid_audio_frames": self.valid_audio_frames,
            "target_spec": self.target_spec.to_dict(),
        }
        return {**dict(self.extra), **known}

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> Self:
        """Restore a record, collecting unrecognised keys into ``extra``.

        Args:
            values: One entry of the manifest's ``samples`` list.

        Returns:
            The restored record.

        Raises:
            ShardCorruptionError: If a required key is missing or malformed.
        """
        descriptor_fields = (
            "sample_id",
            "video_frames",
            "height",
            "width",
            "audio_frames",
            "text_tokens",
            "text_width",
            "video_channels",
            "audio_channels",
            "video_timebase_num",
            "video_timebase_den",
            "audio_timebase_num",
            "audio_timebase_den",
            "video_codec_id",
            "audio_codec_id",
        )
        known = {*descriptor_fields, "valid_video_frames", "valid_audio_frames"}
        known.add("target_spec")
        try:
            descriptor = SampleDescriptor(
                **{name: values[name] for name in descriptor_fields}
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ShardCorruptionError(
                f"malformed sample record in shard manifest: {error}"
            ) from error
        spec_values = values.get("target_spec")
        target_spec = (
            TensorBundleSpec.from_dict(cast("dict[str, object]", spec_values))
            if spec_values
            else TensorBundleSpec()
        )
        return cls(
            descriptor=descriptor,
            valid_video_frames=int(values.get("valid_video_frames", -1)),
            valid_audio_frames=int(values.get("valid_audio_frames", -1)),
            target_spec=target_spec,
            extra={key: value for key, value in values.items() if key not in known},
        )


@dataclass(frozen=True, slots=True)
class ShardManifest:
    """The index and provenance of one shard.

    Args:
        shard_schema_version: Layout version of the shard on disk.
        data_schema_version: Version of the record layout in
            :mod:`avgen.data.protocols`.
        records: One entry per sample, in container order.
        data_sha256: Digest of ``data.safetensors``.
        data_bytes: Size of ``data.safetensors``.
        created_unix: Write time, for provenance only. Never used for ordering:
            clocks on a cluster disagree, and a shard's identity must not depend
            on which machine wrote it.
        extra: Manifest keys this version of avgen does not recognise.

    Raises:
        ValueError: If the shard holds no samples, or if its samples disagree
            about which codec produced them.
    """

    shard_schema_version: int
    data_schema_version: int
    records: tuple[ShardRecord, ...]
    data_sha256: str
    data_bytes: int
    created_unix: int
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate that the shard is non-empty and codec-homogeneous."""
        if not self.records:
            raise ValueError(
                "a shard must contain at least one sample; an empty shard is a "
                "writer bug that would otherwise show up as an off-by-one in the "
                "corpus size"
            )
        require_positive("shard_schema_version", self.shard_schema_version)
        require_positive("data_schema_version", self.data_schema_version)
        require_dimension("data_bytes", self.data_bytes, allow_zero=True)
        keys = {record.descriptor.codec_key for record in self.records}
        if len(keys) != 1:
            raise ValueError(
                "every sample in a shard must come from the same codecs and "
                f"timebases; found {len(keys)} distinct combinations. Write one "
                "shard per codec rather than mixing representations on disk."
            )

    def __len__(self) -> int:
        """Return the number of samples in the shard."""
        return len(self.records)

    @property
    def codec_key(self) -> tuple[str, str, int, int, int, int]:
        """The codec and timebase identity shared by every sample."""
        return self.records[0].descriptor.codec_key

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation, unknown keys included."""
        known: dict[str, Any] = {
            "shard_schema_version": self.shard_schema_version,
            "data_schema_version": self.data_schema_version,
            "created_unix": self.created_unix,
            "sample_count": len(self.records),
            "data_sha256": self.data_sha256,
            "data_bytes": self.data_bytes,
            "samples": [record.to_dict() for record in self.records],
        }
        return {**dict(self.extra), **known}

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> Self:
        """Restore a manifest, collecting unrecognised keys into ``extra``.

        Args:
            values: The parsed ``manifest.json``.

        Returns:
            The restored manifest.

        Raises:
            ShardCorruptionError: If a required key is missing, or if the
                declared sample count disagrees with the entries present.
        """
        known = {
            "shard_schema_version",
            "data_schema_version",
            "created_unix",
            "sample_count",
            "data_sha256",
            "data_bytes",
            "samples",
        }
        try:
            entries = cast("list[Mapping[str, Any]]", values["samples"])
            records = tuple(ShardRecord.from_dict(entry) for entry in entries)
            declared = int(values["sample_count"])
            manifest = cls(
                shard_schema_version=int(values["shard_schema_version"]),
                data_schema_version=int(values["data_schema_version"]),
                records=records,
                data_sha256=str(values["data_sha256"]),
                data_bytes=int(values["data_bytes"]),
                created_unix=int(values.get("created_unix", 0)),
                extra={key: value for key, value in values.items() if key not in known},
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ShardCorruptionError(f"malformed shard manifest: {error}") from error
        if declared != len(records):
            raise ShardCorruptionError(
                f"manifest declares {declared} samples but lists {len(records)}"
            )
        return manifest


def _tensor_key(index: int, field_name: str) -> str:
    """Return the container key for one field of one sample.

    Args:
        index: Position of the sample within the shard.
        field_name: Field name, one of :data:`_TENSOR_FIELDS` or a target slot.

    Returns:
        The container key. Zero-padded so lexical order matches numeric order,
        which keeps the container's byte layout in sample order and therefore
        keeps sequential reads sequential on disk.
    """
    return f"{index:08d}.{field_name}"


def _digest_file(path: Path) -> tuple[str, int]:
    """Return the sha256 and byte length of a file, streaming it in chunks.

    Args:
        path: File to digest.

    Returns:
        Hex digest and size in bytes.
    """
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_DIGEST_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _serialise_manifest(manifest: ShardManifest) -> bytes:
    """Serialise a manifest to canonical bytes.

    Keys are sorted and separators are tight so the encoding is a pure function
    of the content. Without that, the manifest digest would depend on dictionary
    insertion order and two writers producing identical shards would disagree.

    Args:
        manifest: Manifest to encode.

    Returns:
        UTF-8 encoded JSON.
    """
    return json.dumps(
        manifest.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    """Flush a directory entry so a rename survives a power loss.

    Renaming a file is atomic, but on most filesystems the *directory entry*
    recording the rename is not durable until the directory itself is synced.
    Skipping this is what turns "the commit marker exists" into "the commit
    marker existed until the node rebooted".

    Args:
        path: Directory to sync.
    """
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_bytes(target: Path, payload: bytes) -> None:
    """Write bytes so a reader never observes a partial file.

    Args:
        target: Final path.
        payload: Bytes to write.
    """
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(target)


def write_shard(
    path: Path | str,
    samples: Sequence[LatentSample],
    *,
    extra: Mapping[str, Any] | None = None,
    validate: bool = True,
) -> ShardManifest:
    """Write samples into a new shard directory, committing atomically.

    The ordering is the whole point and is not negotiable: container, then
    manifest, then marker, each flushed to stable storage before the next
    begins. Reversing any two of them creates a window in which a reader can
    find a marker pointing at data that is not all there.

    Args:
        path: Directory to create. Must not already contain a committed shard.
        samples: Samples to store, in the order they should be read back.
        extra: Additional manifest keys. Use this for corpus-specific
            provenance — a pipeline version, a source identifier, a licence tag.
            Nothing here is interpreted by avgen, and everything here survives a
            read/write round trip.
        validate: Whether to validate each sample against its descriptor before
            writing. Leave it on: the cost is microseconds per sample and the
            alternative is discovering a shape bug after a hundred terabytes
            have been written.

    Returns:
        The manifest that was committed.

    Raises:
        FileExistsError: If ``path`` already holds a committed shard. Shards are
            immutable; rewriting one in place would make the digest of a copy
            depend on when it was copied.
        ValueError: If ``samples`` is empty or a sample fails validation.
    """
    directory = Path(path)
    if (directory / COMMIT_FILENAME).exists():
        raise FileExistsError(
            f"{directory} already holds a committed shard; shards are immutable, "
            "write a new one and retire the old"
        )
    if not samples:
        raise ValueError("write_shard requires at least one sample")
    directory.mkdir(parents=True, exist_ok=True)

    tensors: dict[str, torch.Tensor] = {}
    records: list[ShardRecord] = []
    for index, sample in enumerate(samples):
        if validate:
            sample.validate()
        for name in _TENSOR_FIELDS:
            tensor = cast("torch.Tensor", getattr(sample, name))
            # A zero-element tensor carries no bytes, and not every container
            # implementation round-trips one faithfully. The shape is already in
            # the manifest, so the empty case is reconstructed rather than
            # stored: fewer keys, and no dependence on that edge case.
            if tensor.numel() == 0:
                continue
            tensors[_tensor_key(index, name)] = tensor.detach().contiguous()
        for slot, value in enumerate(sample.targets.values):
            if value.numel() == 0:
                continue
            tensors[_tensor_key(index, f"target{slot}")] = value.detach().contiguous()
        records.append(
            ShardRecord(
                descriptor=sample.descriptor,
                valid_video_frames=sample.valid_video_frames,
                valid_audio_frames=sample.valid_audio_frames,
                target_spec=sample.target_spec,
            )
        )

    # Imported here rather than at module scope only to keep the failure message
    # local; safetensors is a hard dependency of avgen, not an extra.
    from safetensors.torch import save_file

    data_temporary = directory / (DATA_FILENAME + ".tmp")
    save_file(tensors, str(data_temporary))
    with data_temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    data_sha256, data_bytes = _digest_file(data_temporary)
    data_temporary.replace(directory / DATA_FILENAME)

    manifest = ShardManifest(
        shard_schema_version=SHARD_SCHEMA_VERSION,
        data_schema_version=DATA_SCHEMA_VERSION,
        records=tuple(records),
        data_sha256=data_sha256,
        data_bytes=data_bytes,
        created_unix=int(time.time()),
        extra=dict(extra or {}),
    )
    manifest_bytes = _serialise_manifest(manifest)
    _atomic_write_bytes(directory / MANIFEST_FILENAME, manifest_bytes)

    commit = {
        "shard_schema_version": SHARD_SCHEMA_VERSION,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "data_sha256": data_sha256,
        "committed_unix": int(time.time()),
    }
    _atomic_write_bytes(
        directory / COMMIT_FILENAME,
        json.dumps(commit, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    )
    _fsync_directory(directory)
    return manifest


def _read_commit(directory: Path) -> dict[str, Any]:
    """Read and parse a shard's commit marker.

    Args:
        directory: Shard directory.

    Returns:
        The parsed marker.

    Raises:
        ShardCorruptionError: If the marker is absent or unparseable. Absence is
            the expected outcome for a shard whose writer was preempted, and it
            is reported as corruption rather than as a missing file so a caller
            has one exception type to quarantine on.
    """
    commit_path = directory / COMMIT_FILENAME
    if not commit_path.exists():
        raise ShardCorruptionError(
            f"{directory} has no {COMMIT_FILENAME}; the shard was never committed "
            "(most likely its writer was preempted) and must not be read"
        )
    try:
        return cast("dict[str, Any]", json.loads(commit_path.read_bytes()))
    except (OSError, json.JSONDecodeError) as error:
        raise ShardCorruptionError(
            f"{commit_path} is not readable JSON: {error}"
        ) from error


def _read_manifest(directory: Path) -> tuple[ShardManifest, bytes]:
    """Read a shard manifest and its exact on-disk bytes.

    The raw bytes are returned alongside the parsed object because the digest is
    over what was written, not over what a re-serialisation would produce.

    Args:
        directory: Shard directory.

    Returns:
        The parsed manifest and its bytes.

    Raises:
        ShardCorruptionError: If the manifest is absent or unparseable.
    """
    manifest_path = directory / MANIFEST_FILENAME
    try:
        payload = manifest_path.read_bytes()
    except OSError as error:
        raise ShardCorruptionError(f"{manifest_path} is unreadable: {error}") from error
    try:
        values = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ShardCorruptionError(
            f"{manifest_path} is not valid JSON: {error}"
        ) from error
    return ShardManifest.from_dict(values), payload


def validate_shard(path: Path | str, *, check_data: bool = True) -> ShardManifest:
    """Verify a shard's commit marker, manifest digest, and data digest.

    Both digests are checked. The manifest digest recorded in ``COMMIT`` catches
    a manifest that was edited or truncated after the fact; the data digest
    recorded in the manifest catches bit rot in the container. Checking only one
    leaves half the file unprotected.

    Args:
        path: Shard directory.
        check_data: Whether to re-digest the container. This reads the whole
            shard, so a corpus-wide sweep is expensive; leaving it off still
            verifies the marker, the manifest digest, and the container's size,
            which is enough to catch a truncated write.

    Returns:
        The validated manifest.

    Raises:
        ShardCorruptionError: On a missing marker, a digest mismatch, or a size
            mismatch.
    """
    directory = Path(path)
    commit = _read_commit(directory)
    manifest, manifest_bytes = _read_manifest(directory)

    observed_manifest = hashlib.sha256(manifest_bytes).hexdigest()
    expected_manifest = str(commit.get("manifest_sha256", ""))
    if observed_manifest != expected_manifest:
        raise ShardCorruptionError(
            f"{directory}: manifest digest mismatch; COMMIT records "
            f"{expected_manifest or '(absent)'} but the manifest hashes to "
            f"{observed_manifest}"
        )

    data_path = directory / DATA_FILENAME
    if not data_path.exists():
        raise ShardCorruptionError(f"{directory}: {DATA_FILENAME} is missing")
    size = data_path.stat().st_size
    if size != manifest.data_bytes:
        raise ShardCorruptionError(
            f"{directory}: {DATA_FILENAME} is {size} bytes but the manifest "
            f"records {manifest.data_bytes}; the container is truncated or grew"
        )
    if check_data:
        observed_data, _ = _digest_file(data_path)
        if observed_data != manifest.data_sha256:
            raise ShardCorruptionError(
                f"{directory}: data digest mismatch; manifest records "
                f"{manifest.data_sha256} but the container hashes to "
                f"{observed_data}"
            )
    return manifest


class ShardReader:
    """Memory-mapped random access to one committed shard.

    The container is opened once and held open for the reader's lifetime. Every
    tensor handed out is a view onto the mapping, so reading a sample costs a
    page fault per touched page and no copy at all. That is what makes it
    affordable for a data loader to hold thousands of shards open and read a
    scattered permutation of samples out of them.

    The consequence, and it matters: **a returned tensor aliases the file.** It
    is valid only while the reader is open, and mutating it in place is either a
    segmentation fault or a corrupted shard depending on the mapping mode.
    Collation copies (``torch.stack`` allocates), so the batch a loader emits is
    independent of the mapping; anything else that wants to keep a tensor past
    the reader's lifetime must clone it.

    Args:
        path: Shard directory.
        validate: Whether to verify digests on open. Digesting a whole shard on
            every open is too slow for a training job that opens thousands, so
            the default only checks the commit marker, the manifest digest, and
            the container size. Run the full check in an offline sweep.

    Raises:
        ShardCorruptionError: If the shard is uncommitted or fails its checks.
    """

    __slots__ = ("_container", "_handle", "_manifest", "_path")

    def __init__(self, path: Path | str, *, validate: bool = False) -> None:
        self._path = Path(path)
        self._manifest = validate_shard(self._path, check_data=validate)
        # safetensors is a hard dependency, but the import stays local so the
        # module imports on a machine where only the manifest is being read.
        from safetensors import safe_open

        self._container = safe_open(
            str(self._path / DATA_FILENAME), framework="pt", device="cpu"
        )
        # The Python binding exposes the mapping through the context-manager
        # protocol; entering it explicitly lets the reader own the lifetime
        # rather than forcing every caller into a ``with`` block.
        self._handle = self._container.__enter__()

    @property
    def path(self) -> Path:
        """Directory this reader was opened on."""
        return self._path

    @property
    def manifest(self) -> ShardManifest:
        """The shard's validated manifest."""
        return self._manifest

    def __len__(self) -> int:
        """Return the number of samples in the shard."""
        return len(self._manifest.records)

    def __enter__(self) -> Self:
        """Enter a context that closes the mapping on exit."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the mapping."""
        self.close()

    def close(self) -> None:
        """Release the memory mapping.

        Tensors previously handed out alias the mapping and must not be used
        afterwards.
        """
        container = self._container
        if container is not None:
            container.__exit__(None, None, None)
            self._container = None

    def descriptor(self, index: int) -> SampleDescriptor:
        """Return a sample's description without touching the container.

        Args:
            index: Position within the shard.

        Returns:
            The descriptor.

        Raises:
            IndexError: If ``index`` is out of range.
        """
        return self._record(index).descriptor

    def record(self, index: int) -> ShardRecord:
        """Return a sample's full manifest entry.

        Args:
            index: Position within the shard.

        Returns:
            The record, including any unrecognised manifest keys.

        Raises:
            IndexError: If ``index`` is out of range.
        """
        return self._record(index)

    def __getitem__(self, index: int) -> LatentSample:
        """Return one sample as views onto the memory mapping.

        Args:
            index: Position within the shard.

        Returns:
            The sample.

        Raises:
            IndexError: If ``index`` is out of range.
            ShardCorruptionError: If the manifest promises a tensor the
                container does not hold.
        """
        record = self._record(index)
        info = record.descriptor
        shapes: dict[str, tuple[int, ...]] = {
            "video": (info.video_channels, info.video_frames, info.height, info.width),
            "audio": (info.audio_channels, info.audio_frames),
            "text": (info.text_tokens, info.text_width),
            "text_mask": (info.text_tokens,),
            "video_positions": (info.video_frames,),
            "audio_positions": (info.audio_frames,),
        }
        dtypes: dict[str, torch.dtype] = {
            "text_mask": torch.bool,
            "video_positions": torch.float32,
            "audio_positions": torch.float32,
        }
        values: dict[str, torch.Tensor] = {}
        for name in _TENSOR_FIELDS:
            shape = shapes[name]
            values[name] = self._tensor(
                index,
                name,
                shape,
                dtypes.get(name, torch.float32),
            )
        targets = tuple(
            self._tensor(index, f"target{slot}", shape, dtype.torch_dtype)
            for slot, (shape, dtype) in enumerate(
                zip(record.target_spec.shapes, record.target_spec.dtypes, strict=True)
            )
        )
        return LatentSample(
            descriptor=info,
            video=values["video"],
            audio=values["audio"],
            text=values["text"],
            video_positions=values["video_positions"],
            audio_positions=values["audio_positions"],
            text_mask=values["text_mask"],
            valid_video_frames=record.valid_video_frames,
            valid_audio_frames=record.valid_audio_frames,
            targets=TensorBundle(targets),
            target_spec=record.target_spec,
        )

    def _record(self, index: int) -> ShardRecord:
        """Bounds-check an index and return its record.

        Args:
            index: Position within the shard.

        Returns:
            The record.

        Raises:
            IndexError: If ``index`` is out of range.
        """
        if not 0 <= index < len(self._manifest.records):
            raise IndexError(
                f"sample index {index} out of range for shard {self._path} with "
                f"{len(self._manifest.records)} samples"
            )
        return self._manifest.records[index]

    def _tensor(
        self,
        index: int,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Fetch one tensor, materialising the empty case from the manifest.

        Args:
            index: Sample position.
            name: Field name.
            shape: Shape the manifest promises.
            dtype: Dtype to use when the tensor is empty and therefore absent.

        Returns:
            The tensor, as a view onto the mapping when it has content.

        Raises:
            ShardCorruptionError: If a non-empty tensor is missing, or the
                stored shape disagrees with the manifest.
        """
        if 0 in shape:
            return torch.zeros(shape, dtype=dtype)
        key = _tensor_key(index, name)
        try:
            tensor = self._handle.get_tensor(key)
        except Exception as error:  # the binding raises an untyped error
            raise ShardCorruptionError(
                f"{self._path}: manifest promises tensor {key!r} but the container "
                f"does not hold it ({error})"
            ) from error
        if tuple(tensor.shape) != shape:
            raise ShardCorruptionError(
                f"{self._path}: tensor {key!r} has shape {tuple(tensor.shape)} but "
                f"the manifest records {shape}"
            )
        return tensor


class ConcatShardReader:
    """Many shards presented as one flat, indexable corpus.

    A corpus is thousands of shards; a sampler wants one index space. This is
    the adapter, and it is a pure index translation — no copying, no caching, no
    reordering. A global index is resolved to a shard by binary search over
    cumulative sizes, which keeps lookup logarithmic in the number of shards
    rather than linear.

    Shard order is the order given, and it is part of the reproducibility
    contract: the global index of a sample must not change between runs, so
    callers must pass a sorted, stable list rather than the output of a
    directory glob. :meth:`from_paths` sorts for exactly this reason.

    Args:
        readers: Open shard readers, in a stable order.

    Raises:
        ValueError: If no readers are given, or if two shards disagree about
            which codec produced their latents.
    """

    __slots__ = ("_offsets", "_readers", "_total")

    def __init__(self, readers: Sequence[ShardReader]) -> None:
        if not readers:
            raise ValueError("ConcatShardReader requires at least one shard")
        keys = {reader.manifest.codec_key for reader in readers}
        if len(keys) != 1:
            details = "\n".join(
                f"  {reader.path}: {reader.manifest.codec_key}" for reader in readers
            )
            raise ValueError(
                "every shard in a corpus must come from the same codecs and "
                "timebases; a batch drawn across two representations trains on a "
                f"blend neither decoder can invert:\n{details}"
            )
        self._readers = tuple(readers)
        offsets: list[int] = []
        total = 0
        for reader in self._readers:
            offsets.append(total)
            total += len(reader)
        self._offsets = tuple(offsets)
        self._total = total

    @classmethod
    def from_paths(
        cls,
        paths: Sequence[Path | str],
        *,
        validate: bool = False,
        skip_uncommitted: bool = False,
    ) -> Self:
        """Open a list of shard directories as one corpus.

        Args:
            paths: Shard directories. Sorted by string form before opening, so
                the global index space does not depend on filesystem ordering.
            validate: Whether to fully digest each shard on open.
            skip_uncommitted: Whether to silently ignore directories with no
                commit marker. Useful when reading a corpus while a writer is
                still filling it; dangerous otherwise, because it turns a
                corrupted shard into a quietly smaller dataset.

        Returns:
            The concatenated reader.

        Raises:
            ShardCorruptionError: If a shard fails its checks and
                ``skip_uncommitted`` is false.
            ValueError: If no shard could be opened.
        """
        readers: list[ShardReader] = []
        for path in sorted(str(candidate) for candidate in paths):
            try:
                readers.append(ShardReader(path, validate=validate))
            except ShardCorruptionError:
                if not skip_uncommitted:
                    raise
        if not readers:
            raise ValueError("no committed shards found among the given paths")
        return cls(readers)

    @property
    def readers(self) -> tuple[ShardReader, ...]:
        """The underlying shard readers, in index order."""
        return self._readers

    def __len__(self) -> int:
        """Return the total number of samples across every shard."""
        return self._total

    def __enter__(self) -> Self:
        """Enter a context that closes every shard on exit."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close every shard."""
        self.close()

    def close(self) -> None:
        """Close every underlying shard reader."""
        for reader in self._readers:
            reader.close()

    def locate(self, index: int) -> tuple[int, int]:
        """Resolve a global index to ``(shard position, local index)``.

        Args:
            index: Global sample index.

        Returns:
            The shard's position in :attr:`readers` and the index within it.

        Raises:
            IndexError: If ``index`` is out of range.
        """
        if not 0 <= index < self._total:
            raise IndexError(
                f"sample index {index} out of range for a corpus of {self._total} "
                "samples"
            )
        shard = bisect.bisect_right(self._offsets, index) - 1
        return shard, index - self._offsets[shard]

    def __getitem__(self, index: int) -> LatentSample:
        """Return one sample by global index.

        Args:
            index: Global sample index.

        Returns:
            The sample.

        Raises:
            IndexError: If ``index`` is out of range.
        """
        shard, local = self.locate(index)
        return self._readers[shard][local]

    def descriptor(self, index: int) -> SampleDescriptor:
        """Return one sample's description by global index.

        Args:
            index: Global sample index.

        Returns:
            The descriptor.

        Raises:
            IndexError: If ``index`` is out of range.
        """
        shard, local = self.locate(index)
        return self._readers[shard].descriptor(local)
