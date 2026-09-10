"""Core contracts: the types every other subsystem is written against.

Nothing in this package imports from any other avgen subsystem. That is the
rule that keeps the dependency graph a tree rather than a knot: models,
parallelism, data, training, and inference all depend on ``avgen.core``, and
``avgen.core`` depends on nothing but ``torch``.
"""

from avgen.core.batch import (
    ConditionMode,
    MediaBatch,
    MediaBatchSpec,
    null_text_conditioning,
    stack_batches,
)
from avgen.core.metrics import StepMetrics
from avgen.core.model_input import ModelInput, ModelOutput
from avgen.core.patchify import (
    GridPatchifier,
    Patchifier,
    build_spatial_coords,
    build_temporal_coords,
    patchify_grid,
    unpatchify_grid,
)
from avgen.core.rng import RNGStreams
from avgen.core.state import EMA, DataCursor, LRSchedule, Stateful, TrainState
from avgen.core.tensors import TensorBundle, TensorBundleSpec, TensorDType
from avgen.core.tokens import PatchLayout, TextContext, TokenStream

__all__ = [
    "EMA",
    "ConditionMode",
    "DataCursor",
    "GridPatchifier",
    "LRSchedule",
    "MediaBatch",
    "MediaBatchSpec",
    "ModelInput",
    "ModelOutput",
    "PatchLayout",
    "Patchifier",
    "RNGStreams",
    "Stateful",
    "StepMetrics",
    "TensorBundle",
    "TensorBundleSpec",
    "TensorDType",
    "TextContext",
    "TokenStream",
    "TrainState",
    "build_spatial_coords",
    "build_temporal_coords",
    "null_text_conditioning",
    "patchify_grid",
    "stack_batches",
    "unpatchify_grid",
]
