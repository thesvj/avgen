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


def _load_prompts(path: str) -> list[str]:
    """Read the prompt set GRPO rolls out from.

    Args:
        path: Newline-delimited prompt file.

    Returns:
        The non-empty prompts, in file order.

    Raises:
        ValueError: If the path is unset or the file yields no prompts. GRPO
            generates from prompts rather than reading clean latents, so there
            is nothing for the data source to supply and no sensible default.
    """
    from pathlib import Path

    if not path:
        raise ValueError(
            "rl.prompts_file is unset. GRPO rolls out from prompts rather than "
            "reading clean latents, so the data source supplies nothing and "
            "there is no default to fall back on."
        )
    lines = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        raise ValueError(f"rl.prompts_file {path!r} contains no prompts")
    return lines


def _run_grpo(config: RunConfig) -> None:
    """Run reward post-training with group-relative policy optimisation.

    GRPO is driven explicitly rather than through ``Trainer.fit`` because its
    step is a different shape: roll out a group of trajectories per prompt,
    score them, compute a group-relative advantage, then take several inner
    optimisation passes over the stored trajectory. There is no batch of clean
    latents anywhere in that loop, which is why it cannot reuse the supervised
    objective interface.
    """
    import torch

    from avgen.checkpoint import load as load_checkpoint
    from avgen.cli._wiring import build_training_stack
    from avgen.infer.schedule import ScheduleConfig, build_sigma_schedule
    from avgen.rl.grpo import GRPOConfig, GRPOTrainer, WindowSchedule
    from avgen.rl.rollout import model_velocity_fn

    settings = config.rl
    prompts = _load_prompts(settings.prompts_file)
    reward = build_rewards(config)
    stack = build_training_stack(config)

    grpo = GRPOConfig(
        group_size=settings.group_size,
        clip_range=settings.clip_range,
        kl_coefficient=settings.kl_coefficient,
        normalize_advantage_by_std=settings.advantage_normalize,
        noise_level=settings.sde_noise_scale,
        # MixGRPO: only a sliding window of denoising steps takes the SDE path
        # and receives gradient; the rest stay on the deterministic ODE. Zero
        # means optimise every step — correct, and proportionally expensive.
        window=(
            WindowSchedule(
                window=settings.mixgrpo_window, total_steps=settings.sampler_steps
            )
            if settings.mixgrpo_window
            else None
        ),
    )

    bucket = config.data.buckets[0]
    patchifier = stack.objective.patchifier
    device = stack.env.device
    batch = next(iter(stack.source)).to(device)

    velocity = model_velocity_fn(
        stack.parallel.model,
        patchifier=patchifier,
        positions=batch.video_positions,
        mask=batch.video_mask,
        text_features=batch.text,
        text_mask=batch.text_mask,
    )
    sigmas = build_sigma_schedule(
        ScheduleConfig(name="linear", steps=settings.sampler_steps), device=device
    ).sigmas
    latent_shape = (
        config.data.latent_channels,
        bucket.frames,
        bucket.height,
        bucket.width,
    )

    if config.checkpoint.resume:
        load_checkpoint(config.checkpoint.resume, stack.state, parallel=stack.parallel)

    _LOG.info(
        "grpo prompts=%d group_size=%d sampler_steps=%d kl=%g rewards=%s",
        len(prompts),
        settings.group_size,
        settings.sampler_steps,
        settings.kl_coefficient,
        ", ".join(settings.rewards),
    )

    trainer = GRPOTrainer(stack.state, velocity, grpo)
    per_step = max(1, config.train.global_batch_size)
    try:
        for step in range(config.train.steps):
            # Cycle the prompt set; a group is drawn per prompt every step.
            offset = (step * per_step) % len(prompts)
            selected = [prompts[(offset + i) % len(prompts)] for i in range(per_step)]
            buffer = trainer.rollout(
                selected,
                sigmas=sigmas,
                latent_shape=latent_shape,
                reward=reward,
                device=device,
                dtype=torch.float32,
            )
            metrics = trainer.update(buffer)
            if metrics and step % config.telemetry.log_every == 0:
                last = metrics[-1]
                record = {
                    "rl/loss": float(last.loss),
                    "rl/kl": float(last.kl),
                    "rl/clip_fraction": float(last.clip_fraction),
                    "rl/advantage_std": float(last.advantage_std),
                }
                if buffer.rewards is not None:
                    record["rl/reward"] = float(buffer.rewards.mean())
                stack.logger.log_metrics(record, step)
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
    raise ValueError(f"unknown rl.algorithm {algorithm!r}; expected one of: grpo, dpo")
