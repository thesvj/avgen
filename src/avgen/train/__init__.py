"""Training: the objective, the step, and the loop that drives them.

The subsystem is deliberately layered so that each piece can be replaced alone:

* :mod:`avgen.train.timestep` decides *where on the trajectory* to train, and
  is the module that makes high-resolution training work at all.
* :mod:`avgen.train.conditioning` decides *what task* each sample is, which is
  what turns one set of weights into a text-to-video, image-to-video,
  continuation, inpainting, and audio-conditioned model.
* :mod:`avgen.train.objective` turns a batch into a differentiable scalar,
  including patchification, the model call, and a context-parallel-correct
  reduction.
* :mod:`avgen.train.optimizer`, :mod:`avgen.train.schedule`, and
  :mod:`avgen.train.ema` supply the update rule, its learning-rate envelope,
  and the averaged weights that samples are actually drawn from.
* :mod:`avgen.train.trainer` holds the functional step and a thin orchestrator.

Typical use::

    from avgen.train import (
        FlowMatchingConfig, FlowMatchingObjective, MultiTaskConditioning,
        Trainer, TrainerConfig, build_optimizer, build_schedule,
        build_timestep_sampler,
    )

    objective = FlowMatchingObjective(
        FlowMatchingConfig(),
        build_timestep_sampler("shifted_logit_normal"),
        MultiTaskConditioning(),
    )
    trainer = Trainer(state, objective, parallel, TrainerConfig())
    trainer.fit(loader, total_steps=100_000)
"""

from avgen.train.conditioning import (
    ConditioningPlan,
    ConditioningSampler,
    MultiTaskConditioning,
    default_mode_weights,
)
from avgen.train.ema import ShardedEMA
from avgen.train.objective import (
    FlowMatchingConfig,
    FlowMatchingObjective,
    Objective,
    ObjectiveOutput,
)
from avgen.train.optimizer import (
    DEFAULT_NO_DECAY_PATTERNS,
    build_optimizer,
    build_param_groups,
    list_optimizers,
)
from avgen.train.schedule import build_schedule, list_schedules, register_schedule
from avgen.train.timestep import (
    LogitNormalSampler,
    ModeSampler,
    ShiftedLogitNormalSampler,
    TimestepSampler,
    UniformSampler,
    build_timestep_sampler,
    list_timestep_samplers,
    register_timestep_sampler,
    resolution_shift,
    shift_timesteps,
)
from avgen.train.trainer import (
    Curriculum,
    CurriculumStage,
    MetricSink,
    ResolutionBucket,
    Trainer,
    TrainerCallback,
    TrainerConfig,
    train_step,
)

__all__ = [
    "DEFAULT_NO_DECAY_PATTERNS",
    "ConditioningPlan",
    "ConditioningSampler",
    "Curriculum",
    "CurriculumStage",
    "FlowMatchingConfig",
    "FlowMatchingObjective",
    "LogitNormalSampler",
    "MetricSink",
    "ModeSampler",
    "MultiTaskConditioning",
    "Objective",
    "ObjectiveOutput",
    "ResolutionBucket",
    "ShardedEMA",
    "ShiftedLogitNormalSampler",
    "TimestepSampler",
    "Trainer",
    "TrainerCallback",
    "TrainerConfig",
    "UniformSampler",
    "build_optimizer",
    "build_param_groups",
    "build_schedule",
    "build_timestep_sampler",
    "default_mode_weights",
    "list_optimizers",
    "list_schedules",
    "list_timestep_samplers",
    "register_schedule",
    "register_timestep_sampler",
    "resolution_shift",
    "shift_timesteps",
    "train_step",
]
