"""Reinforcement learning and preference optimisation for flow-matching models.

Two paths, sharing the reward layer:

* **Online RL** (:mod:`avgen.rl.grpo`). Generate groups of samples, score them,
  and take a group-relative policy-gradient step. Needs a reward model and a lot
  of sampling; can exceed the quality of anything in the training data, because
  it optimises the reward rather than imitating examples.
* **Preference optimisation** (:mod:`avgen.rl.dpo`). Train directly on
  ``(winner, loser)`` pairs against a frozen reference. No reward model, no
  rollouts, and it plugs into the existing supervised trainer unchanged; bounded
  above by the preferences it was given.

The enabling piece for the first path is :mod:`avgen.rl.sde`: a deterministic
flow sampler is not a policy at all, and converting the sampling ODE into a
marginal-preserving SDE is what gives the trajectory a probability that can be
differentiated.

Typical online use::

    from avgen.rl import CompositeReward, GRPOConfig, GRPOTrainer, WindowSchedule

    trainer = GRPOTrainer(state, velocity_fn, GRPOConfig(
        group_size=12, window=WindowSchedule(window=4, total_steps=25)))
    buffer = trainer.rollout(prompts, sigmas=sigmas,
                             latent_shape=shape, reward=reward)
    trainer.update(buffer)
"""

from avgen.rl.dpo import DPOConfig, DPOObjective, dpo_loss
from avgen.rl.grpo import (
    GRPOConfig,
    GRPOMetrics,
    GRPOTrainer,
    WindowSchedule,
    clipped_surrogate,
    kl_penalty,
    regulated_log_ratio,
)
from avgen.rl.rewards import (
    AVSyncProxyReward,
    CompositeReward,
    HPSv2Reward,
    MotionMagnitudeReward,
    PickScoreReward,
    RewardModel,
    RunningNormalizer,
    TemporalConsistencyReward,
    VideoScoreReward,
    build_reward,
    list_rewards,
    media_tensors,
    register_reward,
)
from avgen.rl.rollout import (
    RolloutBatch,
    RolloutBuffer,
    VelocityFn,
    distribute_prompts,
    gather_rewards,
    generate_group,
    group_advantages,
    model_velocity_fn,
    rollout_velocity_batch,
)
from avgen.rl.sde import (
    SDEStep,
    diffusion_coefficient,
    gaussian_log_prob,
    ode_step,
    to_sde,
)

__all__ = [
    "AVSyncProxyReward",
    "CompositeReward",
    "DPOConfig",
    "DPOObjective",
    "GRPOConfig",
    "GRPOMetrics",
    "GRPOTrainer",
    "HPSv2Reward",
    "MotionMagnitudeReward",
    "PickScoreReward",
    "RewardModel",
    "RolloutBatch",
    "RolloutBuffer",
    "RunningNormalizer",
    "SDEStep",
    "TemporalConsistencyReward",
    "VelocityFn",
    "VideoScoreReward",
    "WindowSchedule",
    "build_reward",
    "clipped_surrogate",
    "diffusion_coefficient",
    "distribute_prompts",
    "dpo_loss",
    "gather_rewards",
    "gaussian_log_prob",
    "generate_group",
    "group_advantages",
    "kl_penalty",
    "list_rewards",
    "media_tensors",
    "model_velocity_fn",
    "ode_step",
    "register_reward",
    "regulated_log_ratio",
    "rollout_velocity_batch",
    "to_sde",
]
