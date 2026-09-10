"""Sharded training checkpoints and consolidated release artifacts.

Two formats, two purposes, one module.

``CheckpointManager`` writes the *training* checkpoint: sharded, asynchronous,
self-pruning, and — the point of the whole design — reshardable, so a job
preempted off 512 GPUs restarts on whatever the scheduler offers next.

``export_safetensors`` and ``export_huggingface`` write the *release* artifact:
consolidated full tensors that anything can read, with no optimizer state, no
RNG, and no dependency on avgen.

Typical use::

    from avgen.checkpoint import CheckpointManager

    manager = CheckpointManager(run_dir / "checkpoints", keep_last_n=3)
    start_step = manager.load(state, parallel=parallel) or 0
    ...
    manager.save(step, state, parallel=parallel)   # every rank calls this
    manager.close()                                # drains the last upload
"""

from avgen.checkpoint.export import (
    DEFAULT_MAX_SHARD_BYTES,
    INDEX_FILENAME,
    WEIGHTS_FILENAME,
    convert,
    export_huggingface,
    export_safetensors,
    import_safetensors,
)
from avgen.checkpoint.manager import (
    CHECKPOINT_DIR_PREFIX,
    CHECKPOINT_FORMAT_VERSION,
    MARKER_FILENAME,
    CheckpointEntry,
    CheckpointManager,
    load,
    save,
)
from avgen.checkpoint.stateful import (
    EMA_KEY,
    EXTRAS_KEY,
    MODEL_OPTIMIZER_KEY,
    PROGRESS_KEY,
    RNG_KEY,
    SCHEDULE_KEY,
    ExtrasState,
    ModelOptimizerState,
    ProgressState,
    RNGState,
    build_stateful,
)

__all__ = [
    "CHECKPOINT_DIR_PREFIX",
    "CHECKPOINT_FORMAT_VERSION",
    "DEFAULT_MAX_SHARD_BYTES",
    "EMA_KEY",
    "EXTRAS_KEY",
    "INDEX_FILENAME",
    "MARKER_FILENAME",
    "MODEL_OPTIMIZER_KEY",
    "PROGRESS_KEY",
    "RNG_KEY",
    "SCHEDULE_KEY",
    "WEIGHTS_FILENAME",
    "CheckpointEntry",
    "CheckpointManager",
    "ExtrasState",
    "ModelOptimizerState",
    "ProgressState",
    "RNGState",
    "build_stateful",
    "convert",
    "export_huggingface",
    "export_safetensors",
    "import_safetensors",
    "load",
    "save",
]
