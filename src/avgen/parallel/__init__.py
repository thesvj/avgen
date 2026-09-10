"""Distributed execution: meshes, sharding plans, and mesh-aware collectives.

Everything here is built on ``torch.distributed`` primitives — ``DeviceMesh``,
``DTensor``, ``fully_shard``, ``parallelize_module``, ``context_parallel``,
``pipelining``. avgen adds the parts PyTorch deliberately leaves to the
framework: which axes exist, in what order they are composed, which mesh a
reduction belongs on, and how a video-shaped model maps onto all of it.

Typical use::

    from avgen.parallel import ParallelConfig, ParallelDims, parallelize

    dims = ParallelDims.from_env(context=8, tensor=2)
    parallel = parallelize(model, dims, config=ParallelConfig())
"""

from avgen.parallel.activation import (
    ACMode,
    ActivationCheckpointConfig,
    apply_activation_checkpointing,
)
from avgen.parallel.apply import ParallelConfig, ParallelModel, parallelize
from avgen.parallel.comm import (
    all_reduce_max,
    all_reduce_mean,
    all_reduce_sum,
    broadcast_object,
    clip_grad_norm,
    data_mesh,
    gather_object,
)
from avgen.parallel.context import (
    context_parallel_region,
    gather_stream,
    pad_to_multiple,
    shard_stream,
    sharded_length,
)
from avgen.parallel.dims import ParallelDims, submesh
from avgen.parallel.env import (
    DistributedEnv,
    barrier,
    init_distributed,
    is_distributed_launch,
    local_rank_device,
    shutdown_distributed,
    unwrap_model,
)
from avgen.parallel.fsdp import FSDPConfig, apply_fsdp, summarize_sharding
from avgen.parallel.pipeline import (
    PipelineConfig,
    balanced_split_points,
    build_pipeline_schedule,
    split_model,
)
from avgen.parallel.precision import (
    PrecisionConfig,
    convert_to_float8,
    float8_available,
)
from avgen.parallel.tensor import (
    TensorParallelizable,
    TensorParallelPlan,
    apply_tensor_parallel,
    standard_block_plan,
    standard_root_plan,
)

__all__ = [
    "ACMode",
    "ActivationCheckpointConfig",
    "DistributedEnv",
    "FSDPConfig",
    "ParallelConfig",
    "ParallelDims",
    "ParallelModel",
    "PipelineConfig",
    "PrecisionConfig",
    "TensorParallelPlan",
    "TensorParallelizable",
    "all_reduce_max",
    "all_reduce_mean",
    "all_reduce_sum",
    "apply_activation_checkpointing",
    "apply_fsdp",
    "apply_tensor_parallel",
    "balanced_split_points",
    "barrier",
    "broadcast_object",
    "build_pipeline_schedule",
    "clip_grad_norm",
    "context_parallel_region",
    "convert_to_float8",
    "data_mesh",
    "float8_available",
    "gather_object",
    "gather_stream",
    "init_distributed",
    "is_distributed_launch",
    "local_rank_device",
    "pad_to_multiple",
    "parallelize",
    "shard_stream",
    "sharded_length",
    "shutdown_distributed",
    "split_model",
    "standard_block_plan",
    "standard_root_plan",
    "submesh",
    "summarize_sharding",
    "unwrap_model",
]
