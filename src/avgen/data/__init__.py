"""Data: buckets, shards, packing, and the resumable loader.

The subsystem is layered, and the layering is what keeps it testable:

``protocols``
    The record types and the two protocols (:class:`SampleStore`,
    :class:`DataSource`) everything else is written against. Depends on
    :mod:`avgen.core` and nothing else in this package.
``synthetic``
    A dependency-free procedural corpus with a known audio-video correlation.
    What CI, the simulator, and the quickstart run on.
``shard``
    The on-disk latent format: one mmap-able container, a JSON manifest, and an
    atomic commit marker so a preempted writer leaves nothing readable.
``bucket``
    Resolution and duration bucketing with a step-varying curriculum, which is
    the only affordable way to train on mixed-shape video.
``packing``
    Concatenating short clips into one sequence under a block-diagonal mask,
    which removes the padding bucketing leaves behind.
``loader``
    Puts them together: shards on ``data_rank`` only, resumes on an exact
    cursor, prefetches to the device.
``ingest``
    The offline pixel-to-latent pass, with generic quality gates whose numeric
    thresholds are all disabled by default and meant to be set per corpus.
"""

from avgen.data.bucket import (
    BucketBatch,
    BucketCurriculum,
    BucketSampler,
    CurriculumPhase,
    largest_remainder,
)
from avgen.data.ingest import (
    ClipProbe,
    ClipQA,
    ClipReport,
    ClipSpec,
    IngestSummary,
    QualityGate,
    RejectionReason,
    build_latent_sample,
    content_fingerprint,
    iter_clip_table,
    plan_clip_spec,
    probe_clip,
    summarise_reports,
    tensor_fingerprint,
)
from avgen.data.loader import (
    LoaderConfig,
    LoaderCursor,
    ShardedLoader,
    assign_buckets,
    bucket_from_descriptor,
    build_loader,
)
from avgen.data.packing import (
    PackedLayout,
    PackPlan,
    pack_streams,
    plan_packing,
    split_packed,
)
from avgen.data.protocols import (
    Bucket,
    BucketPlan,
    DataSource,
    LatentSample,
    SampleDescriptor,
    SampleStore,
    collate_samples,
)
from avgen.data.shard import (
    ConcatShardReader,
    ShardCorruptionError,
    ShardManifest,
    ShardReader,
    ShardRecord,
    validate_shard,
    write_shard,
)
from avgen.data.synthetic import AlignmentTruth, SyntheticConfig, SyntheticSource

__all__ = [
    "AlignmentTruth",
    "Bucket",
    "BucketBatch",
    "BucketCurriculum",
    "BucketPlan",
    "BucketSampler",
    "ClipProbe",
    "ClipQA",
    "ClipReport",
    "ClipSpec",
    "ConcatShardReader",
    "CurriculumPhase",
    "DataSource",
    "IngestSummary",
    "LatentSample",
    "LoaderConfig",
    "LoaderCursor",
    "PackPlan",
    "PackedLayout",
    "QualityGate",
    "RejectionReason",
    "SampleDescriptor",
    "SampleStore",
    "ShardCorruptionError",
    "ShardManifest",
    "ShardReader",
    "ShardRecord",
    "ShardedLoader",
    "SyntheticConfig",
    "SyntheticSource",
    "assign_buckets",
    "bucket_from_descriptor",
    "build_latent_sample",
    "build_loader",
    "collate_samples",
    "content_fingerprint",
    "iter_clip_table",
    "largest_remainder",
    "pack_streams",
    "plan_clip_spec",
    "plan_packing",
    "probe_clip",
    "split_packed",
    "summarise_reports",
    "tensor_fingerprint",
    "validate_shard",
    "write_shard",
]
