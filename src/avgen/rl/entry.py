"""The config-driven entry point behind ``avgen rl``.

Two algorithms share this door, and they are shaped differently on purpose.

**DPO** is an objective. It consumes preference pairs, computes a loss, and
otherwise behaves exactly like supervised training — so it runs through the same
:class:`~avgen.train.trainer.Trainer` as pretraining, with
:class:`~avgen.rl.dpo.DPOObjective` swapped in. That compatibility is the design
goal, not a coincidence: anything that works for a pretraining run (checkpoint
resume, curricula, telemetry, gradient accumulation) works unchanged.

**GRPO** is not an objective. A supervised objective takes a batch of clean
latents; a policy-gradient step takes a *stored trajectory* — the states, the
per-step transition log-probabilities, and a reward that is only known once the
whole sample has been generated and scored. Pretending otherwise would mean
faking a `MediaBatch` that carries none of that. So GRPO gets its own trainer,
which reuses the supervised step's *semantics* (accumulate, clip, skip on
non-finite, step, advance schedule, update EMA) without reusing its interface.

The reward is the part most likely to be misconfigured, and the failure is
quiet: with no reward, every sample in a group scores identically, the
group-relative advantage is identically zero, and the run is an expensive no-op
that looks like it is working.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from avgen.config import RunConfig

__all__ = ["build_rewards", "rl_from_config"]

_LOG = logging.getLogger("avgen.rl")


def build_rewards(config: RunConfig) -> Any:
    """Assemble the reward model this run optimises.

    Args:
        config: The validated run configuration.

    Returns:
        A single reward when one is configured, or a
        :class:`~avgen.rl.rewards.CompositeReward` over the weighted set.

    Raises:
        ValueError: If no reward is configured, or if the weights do not match
            the rewards one-for-one.
        RuntimeError: If a named reward needs an optional dependency that is not
            installed. The message names the extra.
    """
    from avgen.rl.rewards import CompositeReward, build_reward

    names = tuple(config.rl.rewards)
    if not names:
        raise ValueError(
            "rl.rewards is empty. GRPO optimises a reward; with none configured "
            "every sample in a group scores identically, the group-relative "
            "advantage is identically zero, and the run is an expensive no-op."
        )
    weights = tuple(config.rl.reward_weights) or (1.0,) * len(names)
    if len(weights) != len(names):
        raise ValueError(
            f"rl.reward_weights has {len(weights)} entries for {len(names)} "
            "rewards; they must correspond one-for-one"
        )

    rewards = [build_reward(name) for name in names]
    if len(rewards) == 1:
        return rewards[0]
    return CompositeReward(tuple(zip(rewards, weights, strict=True)))


def _run_dpo(config: RunConfig) -> None:
    """Run preference post-training through the ordinary supervised trainer."""
    from avgen.checkpoint import load as load_checkpoint
    from avgen.cli._wiring import build_trainer_config, build_training_stack
    from avgen.rl.dpo import DPOConfig, DPOObjective
    from avgen.train.trainer import Trainer

    stack = build_training_stack(config)

    reference = None
    if config.rl.reference_checkpoint:
        from avgen.checkpoint.export import import_safetensors
        from avgen.config.resolve import model_kwargs
        from avgen.models.registry import build_model

        reference = build_model(config.model.name, model_kwargs(config))
        import_safetensors(config.rl.reference_checkpoint, reference, strict=False)
        reference.eval()
        for parameter in reference.parameters():
            parameter.requires_grad_(False)

    objective = DPOObjective(
        DPOConfig(beta=config.rl.dpo_beta),
        reference_model=reference,
    )
    if config.checkpoint.resume:
        load_checkpoint(config.checkpoint.resume, stack.state, parallel=stack.parallel)

    trainer = Trainer(
        stack.state,
        objective,
        stack.parallel,
        build_trainer_config(
            config,
            accumulation=stack.gradient_accumulation,
            data_world=stack.data_world,
        ),
        logger=stack.logger,
    )
    try:
        trainer.fit(stack.source, total_steps=config.train.steps)
    finally:
        close = getattr(stack.logger, "close", None)
        if callable(close):
            close()


def _run_grpo(config: RunConfig) -> None:
    """Run reward post-training with group-relative policy optimisation."""
    from avgen.checkpoint import load as load_checkpoint
    from avgen.cli._wiring import build_training_stack
    from avgen.rl.grpo import GRPOConfig, GRPOTrainer

    stack = build_training_stack(config)
    reward = build_rewards(config)

    settings = config.rl
    grpo = GRPOConfig(
        group_size=settings.group_size,
        sampler_steps=settings.sampler_steps,
        kl_coefficient=settings.kl_coefficient,
        clip_range=settings.clip_range,
        normalize_advantage=settings.advantage_normalize,
        mixgrpo_window=settings.mixgrpo_window,
        sde_noise_scale=settings.sde_noise_scale,
    )
    _LOG.info(
        "grpo group_size=%d steps=%d kl=%g rewards=%s",
        settings.group_size,
        settings.sampler_steps,
        settings.kl_coefficient,
        ", ".join(settings.rewards),
    )

    if config.checkpoint.resume:
        load_checkpoint(config.checkpoint.resume, stack.state, parallel=stack.parallel)

    trainer = GRPOTrainer(
        stack.state,
        stack.parallel,
        grpo,
        reward=reward,
        reference_checkpoint=settings.reference_checkpoint or None,
        logger=stack.logger,
    )
    try:
        trainer.fit(stack.source, total_steps=config.train.steps)
    finally:
        close = getattr(stack.logger, "close", None)
        if callable(close):
            close()


def rl_from_config(config: RunConfig) -> None:
    """Run reward or preference post-training described by a configuration.

    Args:
        config: The validated run configuration. ``config.rl.algorithm``
            selects between ``grpo`` and ``dpo``.

    Raises:
        ValueError: If the algorithm is unknown or the reward set is invalid.
        RuntimeError: If a required subsystem is unavailable.
    """
    algorithm = config.rl.algorithm
    if algorithm == "dpo":
        _run_dpo(config)
        return
    if algorithm == "grpo":
        _run_grpo(config)
        return
    raise ValueError(
        f"unknown rl.algorithm {algorithm!r}; expected one of: grpo, dpo"
    )
