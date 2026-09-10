"""avgen — a PyTorch-native framework for training video generation models at scale.

The whole surface, in one place::

    import avgen

    # 1. Decide how to spread the job across the cluster, and check it first.
    dims = avgen.ParallelDims.from_env(context=8, tensor=2)
    report = avgen.simulate_config(model.model_shape(sequence_length=65_536), dims)
    print(report.render())          # fits? how fast? what is the bottleneck?

    # 2. Apply the plan.
    parallel = avgen.parallelize(model, dims)

    # 3. Train.
    trainer = avgen.Trainer(state, objective, parallel, config)
    trainer.fit(loader, total_steps=100_000)

Design commitments, stated once so the rest of the code does not have to argue
for them:

**Sequence-first.** Dense latent grids exist only at the data and codec
boundary. Everything a model touches is a :class:`~avgen.core.TokenStream`, a
flat ``(batch, length, width)`` sequence with explicit per-token coordinates.
This is what makes context parallelism, variable-resolution batching, and
sequence packing possible at all — none of which can be expressed on a
five-dimensional grid.

**Physical coordinates.** Time is seconds, space is latent pixels. Not indices.
A model trained at 24 fps and 256px can be sampled at 30 fps and 512px because
its positional encoding was never told what a frame index was.

**One model ABI.** Training and inference both build a
:class:`~avgen.core.ModelInput` and both read a
:class:`~avgen.core.ModelOutput`. There is no separate inference path to drift
out of sync, which removes the single most expensive class of bug in generative
codebases.

**PyTorch, not a framework on top of PyTorch.** Meshes are ``DeviceMesh``,
shards are ``DTensor``, sharding is ``fully_shard``, checkpoints are
``torch.distributed.checkpoint``, launching is ``torchrun``. No Ray, no
DeepSpeed, no Megatron, no Accelerate in the dependency tree. The core package
needs ``torch``, ``numpy``, ``pyyaml``, and ``safetensors``, and nothing else.

**Simulate before you allocate.** :mod:`avgen.simulate` builds a 1024-rank mesh
on a laptop, prices the memory, counts the collectives, and predicts the step
time — so a parallelism decision is a measurement rather than a guess.

Submodules are imported lazily, so ``import avgen`` stays fast and a machine
without an optional dependency can still import the package.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

# Eager: the contracts and the distributed layer. Both are pure-torch, cheap to
# import, and needed by essentially every entry point.
from avgen.core import (
    ConditionMode,
    GridPatchifier,
    MediaBatch,
    MediaBatchSpec,
    ModelInput,
    ModelOutput,
    Patchifier,
    PatchLayout,
    RNGStreams,
    StepMetrics,
    TextContext,
    TokenStream,
    TrainState,
)
from avgen.parallel import (
    ActivationCheckpointConfig,
    DistributedEnv,
    FSDPConfig,
    ParallelConfig,
    ParallelDims,
    ParallelModel,
    PipelineConfig,
    PrecisionConfig,
    init_distributed,
    parallelize,
    shutdown_distributed,
)

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from avgen.checkpoint import CheckpointManager
    from avgen.config import RunConfig, load_config
    from avgen.data import SyntheticSource, build_loader
    from avgen.eval import EvalReport, run_eval_suite
    from avgen.infer import GenerationPipeline
    from avgen.models import build_model, list_models
    from avgen.simulate import ModelShape, SimulationReport, simulate_config
    from avgen.train import FlowMatchingObjective, Trainer, train_step

#: Lazily resolved re-exports: attribute name to the module that defines it.
#: Keeping these out of the eager import means ``import avgen`` does not pull in
#: the sampler, the eval metrics, or anything that touches an optional
#: dependency — which matters when a cluster image has only the core package.
_LAZY: dict[str, str] = {
    "CheckpointManager": "avgen.checkpoint",
    "EvalReport": "avgen.eval",
    "FlowMatchingObjective": "avgen.train",
    "GenerationPipeline": "avgen.infer",
    "ModelShape": "avgen.simulate",
    "RunConfig": "avgen.config",
    "SimulationReport": "avgen.simulate",
    "SyntheticSource": "avgen.data",
    "Trainer": "avgen.train",
    "build_loader": "avgen.data",
    "build_model": "avgen.models",
    "list_models": "avgen.models",
    "load_config": "avgen.config",
    "run_eval_suite": "avgen.eval",
    "simulate_config": "avgen.simulate",
    "train_step": "avgen.train",
}


def __getattr__(name: str) -> Any:
    """Resolve a lazily exported symbol on first access.

    Args:
        name: Attribute name.

    Returns:
        The resolved object.

    Raises:
        AttributeError: If the name is not part of the public surface.
    """
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value  # cache so the next lookup skips this path
    return value


def __dir__() -> list[str]:
    """Return the full public surface, including lazily loaded names."""
    return sorted(__all__)


__all__ = [
    "ActivationCheckpointConfig",
    "CheckpointManager",
    "ConditionMode",
    "DistributedEnv",
    "EvalReport",
    "FSDPConfig",
    "FlowMatchingObjective",
    "GenerationPipeline",
    "GridPatchifier",
    "MediaBatch",
    "MediaBatchSpec",
    "ModelInput",
    "ModelOutput",
    "ModelShape",
    "ParallelConfig",
    "ParallelDims",
    "ParallelModel",
    "PatchLayout",
    "Patchifier",
    "PipelineConfig",
    "PrecisionConfig",
    "RNGStreams",
    "RunConfig",
    "SimulationReport",
    "StepMetrics",
    "SyntheticSource",
    "TextContext",
    "TokenStream",
    "TrainState",
    "Trainer",
    "__version__",
    "build_loader",
    "build_model",
    "init_distributed",
    "list_models",
    "load_config",
    "parallelize",
    "run_eval_suite",
    "shutdown_distributed",
    "simulate_config",
    "train_step",
]
