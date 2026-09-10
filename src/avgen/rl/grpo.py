"""Group Relative Policy Optimization for flow-matching video models.

The pieces, and where each one comes from:

* **Group-relative advantage** (Flow-GRPO, arXiv:2505.05470, Eq. 4). No value
  network: ``A_i = (r_i - mean(r_group)) / (std(r_group) + eps)``. See
  :func:`avgen.rl.rollout.group_advantages` for why the absence of a critic is
  the point rather than an economy.
* **PPO-style clipped surrogate** (Eq. 5). The ratio of the new policy's
  transition density to the rollout policy's, clipped so a single step cannot
  move the policy arbitrarily far off the distribution the data was collected
  under.
* **Regulated clipping** (GRPO-Guard, arXiv:2510.22319). Plain clipping does not
  work on a diffusion policy, for a reason that is specific to this setting and
  is documented at :func:`regulated_log_ratio`.
* **KL to a frozen reference** (Flow-GRPO Eq. 6, ``beta`` 0.01-0.04). DanceGRPO
  and GRPO-Guard both drop it; it is off by default here and available because
  a long run without it drifts.
* **MixGRPO's sliding window** (arXiv:2507.21802). Only ``w`` of the ``T``
  denoising steps use the SDE path and carry gradient; the rest stay on the
  deterministic ODE. Reported 48% training-time reduction at ``w = 4``, ``T = 25``.

Implicit over-optimisation, and what it looks like
--------------------------------------------------

The failure this file spends the most code preventing does not announce itself.
The reward curve rises smoothly and monotonically. The samples get worse:
colours saturate, textures smooth out, motion becomes repetitive and small, and
prompt adherence quietly degrades on everything the reward model does not
measure. The policy has not learned the behaviour the reward was a proxy for; it
has found the reward model's blind spot, and the reward curve is the one
diagnostic that cannot see it.

GRPO-Guard identifies a mechanism specific to flow models. The importance ratio
for a diffusion policy has a *timestep-dependent* distribution: its log has mean
``-||dmu||^2 / (2 s^2)``, which is systematically negative and whose magnitude
scales with ``1 / (s^2)``, i.e. differently at every denoising step. So a single
clip range ``[1-eps, 1+eps]`` is a tight constraint at some timesteps and no
constraint at all at others; positive-advantage samples at the loose timesteps
never enter the clip region, and the mechanism that was supposed to bound the
update simply does not engage there. The fix is to normalise the ratio before
clipping, so that one ``eps`` means the same thing at every step.

Monitor ``clip_fraction`` and ``kl`` alongside the reward. A reward that climbs
while the clip fraction stays near zero is the signature: nothing is being
constrained.

Relationship to :mod:`avgen.train`
-----------------------------------

The optimizer mechanics — accumulation, ``avgen.parallel.clip_grad_norm``,
non-finite rejection, :class:`~avgen.core.metrics.StepMetrics`, EMA and schedule
advancement — are the same as a supervised step and are reused through
:class:`~avgen.core.state.TrainState` rather than reimplemented.

What could *not* be reused is the ``Objective`` protocol, and the reason is
structural rather than incidental: an ``Objective`` is called with a
:class:`~avgen.core.batch.MediaBatch` of clean latents, while an RL update's
input is a stored trajectory of noisy states with their transition statistics
and a terminal reward. There is no honest way to express one as the other, and
pretending otherwise would mean smuggling a rollout buffer through a field that
claims to be a batch of training data. :mod:`avgen.rl.dpo` *does* satisfy the
protocol, because preference data genuinely is a batch of clean latents.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from avgen.rl.rollout import RolloutBatch, RolloutBuffer, VelocityFn
from avgen.rl.sde import to_sde

__all__ = [
    "GRPOConfig",
    "GRPOMetrics",
    "GRPOTrainer",
    "WindowSchedule",
    "clipped_surrogate",
    "kl_penalty",
    "regulated_log_ratio",
]


@dataclass(frozen=True, slots=True)
class WindowSchedule:
    """MixGRPO's sliding window of gradient-carrying denoising steps.

    Only steps inside the window use the SDE path, get their transitions stored,
    and receive gradient. Everything outside it is a deterministic ODE step: one
    forward pass, no stored state, no backward.

    The efficiency argument is direct. Cost per rollout scales with the number
    of stored steps, and so does the memory that holds them; with ``T = 25`` and
    ``w = 4`` that is a 6x reduction in both, which is what turns a 291
    seconds-per-iteration baseline into 151 (MixGRPO Table 1). The reason it does
    not cost quality is that the window *moves*: over a training run every
    timestep spends time inside it, so no part of the trajectory goes
    permanently unoptimised — it is a schedule over which steps are optimised
    when, not a decision to ignore most of them.

    The window starts at the high-noise end and moves toward clean. That
    direction is deliberate: early steps decide global structure and composition,
    which is what a reward model responds to most strongly, so the window spends
    its first and most influential training steps where the leverage is.

    Args:
        window: Steps inside the window, ``w``. MixGRPO uses 4.
        stride: How far the window moves each time it moves, ``s``. MixGRPO
            uses 1.
        interval: Optimizer steps between window moves, ``tau``. MixGRPO uses 25.
        total_steps: Denoising steps in a rollout, ``T``.
        exponential: Whether the interval shrinks geometrically, so the window
            sweeps the trajectory faster and faster. This is MixGRPO-Flash's
            variant; it trades some late-trajectory optimisation for a further
            speedup.
        decay: Geometric factor applied to the interval when ``exponential``.

    Raises:
        ValueError: If any count is non-positive or the window exceeds the
            trajectory.
    """

    window: int = 4
    stride: int = 1
    interval: int = 25
    total_steps: int = 25
    exponential: bool = False
    decay: float = 0.8

    def __post_init__(self) -> None:
        """Validate the schedule.

        Raises:
            ValueError: On a non-positive count, an out-of-range decay, or a
                window wider than the trajectory.
        """
        for name in ("window", "stride", "interval", "total_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer; got {value!r}")
        if self.window > self.total_steps:
            raise ValueError(
                f"window={self.window} exceeds total_steps={self.total_steps}"
            )
        if not 0.0 < self.decay <= 1.0:
            raise ValueError(f"decay must be in (0, 1]; got {self.decay!r}")

    def start_for(self, optimizer_step: int) -> int:
        """Return the window's leading step index at a training step.

        Args:
            optimizer_step: Optimizer steps completed.

        Returns:
            The window start, clamped so the window never runs off the end.
        """
        if optimizer_step < 0:
            raise ValueError(
                f"optimizer_step must be non-negative; got {optimizer_step}"
            )
        limit = self.total_steps - self.window
        if not self.exponential:
            moves = optimizer_step // self.interval
            return int(min(moves * self.stride, limit))
        # Geometric interval: sum of a shrinking series, so the window reaches
        # the end of the trajectory in bounded time however long the run is.
        moves = 0
        elapsed = 0.0
        span = float(self.interval)
        while elapsed + span <= optimizer_step and moves * self.stride < limit:
            elapsed += span
            span = max(span * self.decay, 1.0)
            moves += 1
        return int(min(moves * self.stride, limit))

    def steps_for(self, optimizer_step: int) -> tuple[int, ...]:
        """Return the denoising-step indices that carry gradient.

        Args:
            optimizer_step: Optimizer steps completed.

        Returns:
            A contiguous run of indices of length ``window``.
        """
        start = self.start_for(optimizer_step)
        return tuple(range(start, start + self.window))


@dataclass(frozen=True, slots=True)
class GRPOConfig:
    """Settings for one GRPO training run.

    Args:
        group_size: ``G``, trajectories per prompt. Flow-GRPO and GRPO-Guard use
            24 for SD3.5-M; DanceGRPO and MixGRPO use 12. Below about 8 the
            group mean is too noisy a baseline to be worth the sampling cost.
        clip_range: Lower clip epsilon. The default is small because it applies
            to the **normalised** ratio when ``regulated_clip`` is on, where
            GRPO-Guard reports 2e-6 for SD3.5-M. With ``regulated_clip`` off,
            raise it to the raw-ratio scale — DanceGRPO reports 1e-4 — and note
            that PPO's familiar 0.2 is not a meaningful value for a diffusion
            policy at either scale.
        clip_range_high: Upper clip epsilon, allowed to differ. An asymmetric
            range (DAPO's "clip-higher") gives low-probability improvements more
            room to be reinforced, which counteracts the entropy collapse that
            symmetric clipping causes. ``None`` mirrors ``clip_range``.
        regulated_clip: Whether to normalise the ratio before clipping, per
            GRPO-Guard. Strongly recommended; see :func:`regulated_log_ratio`.
        timestep_weighting: Whether to reweight each step's contribution by
            ``1/|ds|``, equalising gradient magnitude across the trajectory.
            The weights are renormalised to average one so the loss scale does
            not depend on the number of denoising steps.
        kl_coefficient: Weight on the KL penalty to the reference policy. Zero
            disables it, which is what DanceGRPO, MixGRPO and GRPO-Guard do.
            Flow-GRPO uses 0.04 with a rule-based reward and 0.01 with a learned
            one — the stronger the reward model, the more the policy needs
            holding back.
        advantage_epsilon: Added to the group standard deviation.
        advantage_clip: Symmetric advantage clamp, or ``None``.
        normalize_advantage_by_std: Whether to divide by the group standard
            deviation.
        noise_level: SDE scale ``a``. 0.7 in Flow-GRPO and MixGRPO; DanceGRPO
            reports 0.3.
        sde_schedule: Diffusion-coefficient schedule.
        inner_epochs: Passes over the rollout buffer per round. More than one
            makes the update off-policy, which is exactly what the ratio and the
            clipping exist to permit; beyond about four the stored data is too
            stale for the clip range to contain.
        minibatch_size: Transitions per optimizer step.
        max_grad_norm: Gradient clipping threshold.
        window: MixGRPO window schedule, or ``None`` for full-trajectory SDE.

    Raises:
        ValueError: On a negative or otherwise invalid field.
    """

    group_size: int = 12
    clip_range: float = 2e-6
    clip_range_high: float | None = None
    regulated_clip: bool = True
    timestep_weighting: bool = True
    kl_coefficient: float = 0.0
    advantage_epsilon: float = 1e-4
    advantage_clip: float | None = 5.0
    normalize_advantage_by_std: bool = True
    noise_level: float = 0.7
    sde_schedule: str = "flow_grpo"
    inner_epochs: int = 1
    minibatch_size: int = 4
    max_grad_norm: float = 1.0
    window: WindowSchedule | None = None

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: On an out-of-range field.
        """
        if self.group_size < 2:
            raise ValueError(
                f"group_size must be at least 2; got {self.group_size}. A group "
                "of one has no baseline"
            )
        if self.clip_range <= 0.0:
            raise ValueError(f"clip_range must be positive; got {self.clip_range!r}")
        if self.clip_range_high is not None and self.clip_range_high <= 0.0:
            raise ValueError(
                "clip_range_high must be positive or None; got "
                f"{self.clip_range_high!r}"
            )
        if self.kl_coefficient < 0.0:
            raise ValueError(
                f"kl_coefficient must be non-negative; got {self.kl_coefficient!r}"
            )
        if self.noise_level <= 0.0:
            raise ValueError(
                f"noise_level must be positive for GRPO; got {self.noise_level!r}. "
                "A deterministic policy has zero entropy and zero policy gradient"
            )
        for name in ("inner_epochs", "minibatch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer; got {value!r}")
        if self.max_grad_norm <= 0.0:
            raise ValueError(
                f"max_grad_norm must be positive; got {self.max_grad_norm!r}"
            )

    @property
    def high(self) -> float:
        """Upper clip epsilon, mirroring the lower one when unset."""
        return self.clip_range if self.clip_range_high is None else self.clip_range_high


@dataclass(frozen=True, slots=True)
class GRPOMetrics:
    """Diagnostics for one GRPO minibatch, kept on device.

    ``clip_fraction`` is the one to watch. Near zero means the clip range is not
    binding and nothing constrains the update — the precondition for implicit
    over-optimisation. Near one means every sample is clipped and almost no
    gradient flows.

    Args:
        loss: The surrogate objective, including the KL penalty.
        policy_loss: The surrogate objective alone.
        kl: Mean KL to the reference policy.
        clip_fraction: Fraction of transitions whose ratio hit the clip.
        ratio_mean: Mean importance ratio. Should sit near 1; a systematic drift
            below 1 is the bias GRPO-Guard's normalisation removes.
        advantage_mean: Mean advantage, which is zero by construction within a
            full group and is logged as a correctness check on the grouping.
        advantage_std: Standard deviation of the advantage.
    """

    loss: torch.Tensor
    policy_loss: torch.Tensor
    kl: torch.Tensor
    clip_fraction: torch.Tensor
    ratio_mean: torch.Tensor
    advantage_mean: torch.Tensor
    advantage_std: torch.Tensor

    def to_mapping(self) -> dict[str, float]:
        """Return host-side floats, synchronising once.

        Returns:
            Metric name to value. Call at logging cadence only.
        """
        return {
            "grpo/loss": float(self.loss),
            "grpo/policy_loss": float(self.policy_loss),
            "grpo/kl": float(self.kl),
            "grpo/clip_fraction": float(self.clip_fraction),
            "grpo/ratio_mean": float(self.ratio_mean),
            "grpo/advantage_mean": float(self.advantage_mean),
            "grpo/advantage_std": float(self.advantage_std),
        }


def regulated_log_ratio(
    mean_new: torch.Tensor,
    mean_old: torch.Tensor,
    std_old: torch.Tensor,
    sample: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """GRPO-Guard's RatioNorm: the timestep-debiased log importance ratio.

    Both transitions are isotropic Gaussians with the *same* standard deviation
    ``s`` (the diffusion coefficient does not depend on the policy) and different
    means. Writing ``dmu = mu_old - mu_new`` and noting that the stored sample
    was drawn as ``x = mu_old + s * eps``, the raw log ratio is exactly::

        log r = -||dmu||^2 / (2 s^2) - (dmu . eps) / s

    The first term is deterministic, always negative, and scales as ``1/s^2`` —
    so it differs by orders of magnitude between an early denoising step and a
    late one. It biases the whole ratio distribution below 1, and it does so by
    a different amount at every timestep. A fixed clip range therefore binds
    hard at some timesteps and not at all at others, and at the loose timesteps
    positive-advantage samples never reach the clip region, so the constraint
    that is supposed to prevent an over-large update simply is not there. That
    is the mechanism behind implicit over-optimisation.

    Multiplying by ``s`` and adding back the bias term removes both problems at
    once and leaves::

        log r_hat = -(dmu . eps)

    which has zero mean, unit-free scale, and the same distribution at every
    timestep — so one ``eps`` means one thing everywhere. This is GRPO-Guard
    Eq. 8, and it is an exact algebraic identity, not an approximation.

    Args:
        mean_new: ``(batch, ...)`` transition mean under the current policy.
        mean_old: ``(batch, ...)`` transition mean under the rollout policy.
        std_old: ``(batch,)`` transition standard deviation.
        sample: ``(batch, ...)`` state actually visited.
        mask: ``(batch, ...)`` validity mask.

    Returns:
        ``(batch,)`` normalised log ratios.
    """
    sigma = std_old.reshape(-1, *((1,) * (sample.ndim - 1))).clamp_min(1e-12)
    delta_mean = mean_old - mean_new
    epsilon = (sample - mean_old) / sigma
    product = -(delta_mean * epsilon)
    if mask is not None:
        product = product * mask.to(product.dtype)
    return product.flatten(1).sum(dim=1)


def clipped_surrogate(
    log_ratio: torch.Tensor,
    advantage: torch.Tensor,
    *,
    clip_low: float,
    clip_high: float,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PPO's pessimistic clipped objective, as a loss to minimise.

    ``min`` of the clipped and unclipped terms is a *lower bound* on the true
    objective, which is why it is safe to maximise: an update that looks good
    only because the ratio drifted far from 1 is not credited.

    Args:
        log_ratio: ``(batch,)`` log importance ratio.
        advantage: ``(batch,)`` advantage.
        clip_low: Lower epsilon; the ratio floor is ``1 - clip_low``.
        clip_high: Upper epsilon; the ratio ceiling is ``1 + clip_high``.
        weights: ``(batch,)`` per-transition weights, already normalised.

    Returns:
        ``(loss, clip_fraction, ratio)``.
    """
    ratio = torch.exp(log_ratio)
    clipped = ratio.clamp(1.0 - clip_low, 1.0 + clip_high)
    surrogate = torch.minimum(ratio * advantage, clipped * advantage)
    if weights is not None:
        surrogate = surrogate * weights
    was_clipped = (ratio < 1.0 - clip_low) | (ratio > 1.0 + clip_high)
    return -surrogate.mean(), was_clipped.float().mean(), ratio


def kl_penalty(
    log_prob_new: torch.Tensor, log_prob_reference: torch.Tensor
) -> torch.Tensor:
    """Schulman's k3 estimator of ``KL(pi_new || pi_ref)``.

    ``exp(d) - d - 1`` with ``d = log pi_ref - log pi_new``. Unlike the naive
    ``log pi_new - log pi_ref``, this estimator is non-negative for every sample,
    not merely in expectation. That matters because the penalty is added to a
    loss: a per-sample-negative KL estimate would occasionally *pay* the policy
    for moving away from the reference, and the variance of the naive estimator
    is large enough that this happens often.

    Args:
        log_prob_new: ``(batch,)`` log-probability under the current policy.
        log_prob_reference: ``(batch,)`` log-probability under the reference.

    Returns:
        ``(batch,)`` non-negative KL estimates.
    """
    difference = log_prob_reference - log_prob_new
    return torch.exp(difference) - difference - 1.0


class GRPOTrainer:
    """Runs GRPO updates over rollout buffers.

    Args:
        state: The mutable training state. Its ``model`` is the policy, its
            ``optimizer`` steps it, and its ``rng.sampler`` stream drives the
            rollouts, so a run reproduces from the seed alone.
        velocity_fn: The current policy's velocity predictor, closing over
            ``state.model``.
        config: GRPO settings.
        reference_velocity_fn: The frozen reference policy, required when
            ``kl_coefficient`` is non-zero.
        cp_mesh: Context-parallel mesh, forwarded to gradient clipping.
        pp_mesh: Pipeline mesh, forwarded to gradient clipping.
    """

    __slots__ = (
        "_accumulated",
        "config",
        "cp_mesh",
        "pp_mesh",
        "reference_velocity_fn",
        "state",
        "velocity_fn",
    )

    def __init__(
        self,
        state: Any,
        velocity_fn: VelocityFn,
        config: GRPOConfig,
        *,
        reference_velocity_fn: VelocityFn | None = None,
        cp_mesh: Any | None = None,
        pp_mesh: Any | None = None,
    ) -> None:
        if config.kl_coefficient > 0.0 and reference_velocity_fn is None:
            raise ValueError(
                "kl_coefficient is non-zero but no reference policy was given; "
                "a KL penalty needs something to be relative to"
            )
        self.state = state
        self.velocity_fn = velocity_fn
        self.config = config
        self.reference_velocity_fn = reference_velocity_fn
        self.cp_mesh = cp_mesh
        self.pp_mesh = pp_mesh
        self._accumulated = 0

    def gradient_steps(self) -> tuple[int, ...] | None:
        """Return the denoising steps that carry gradient at the current step.

        Returns:
            The MixGRPO window, or ``None`` when the whole trajectory is used.
        """
        if self.config.window is None:
            return None
        return self.config.window.steps_for(int(self.state.step))

    def _step_weights(self, batch: RolloutBatch) -> torch.Tensor | None:
        """Return per-transition weights that equalise gradient across timesteps.

        GRPO-Guard's ``delta = 1/dt``. The published form is used verbatim and
        then renormalised to mean one, which the paper does not do: without the
        renormalisation the loss magnitude depends on the discretisation, so
        changing the rollout step count silently changes the effective learning
        rate.

        Args:
            batch: The minibatch.

        Returns:
            ``(batch,)`` weights, or ``None`` when weighting is off.
        """
        if not self.config.timestep_weighting:
            return None
        interval = (batch.sigma - batch.sigma_next).abs().clamp_min(1e-8)
        weights = 1.0 / interval
        return weights / weights.mean().clamp_min(1e-12)

    def loss(self, batch: RolloutBatch) -> tuple[torch.Tensor, GRPOMetrics]:
        """Compute the GRPO surrogate loss for one minibatch.

        Args:
            batch: Stored transitions with their advantages.

        Returns:
            ``(loss, metrics)``. The loss carries gradient; every metric is
            detached and stays on device.
        """
        config = self.config
        velocity = self.velocity_fn(
            batch.sample_before, batch.sigma, prompt_index=batch.prompt_index
        )
        step = to_sde(
            velocity,
            batch.sample_before,
            batch.sigma,
            batch.sigma_next,
            noise_level=config.noise_level,
            schedule=config.sde_schedule,
            prev_sample=batch.sample_after,
        )
        if config.regulated_clip:
            log_ratio = regulated_log_ratio(
                step.mean.float(),
                batch.mean_old.float(),
                batch.std_old.float(),
                batch.sample_after.float(),
            )
        else:
            log_ratio = step.log_prob - batch.log_prob_old

        policy_loss, clip_fraction, ratio = clipped_surrogate(
            log_ratio,
            batch.advantage,
            clip_low=config.clip_range,
            clip_high=config.high,
            weights=self._step_weights(batch),
        )

        kl = torch.zeros((), device=policy_loss.device, dtype=policy_loss.dtype)
        if config.kl_coefficient > 0.0 and self.reference_velocity_fn is not None:
            with torch.no_grad():
                reference_velocity = self.reference_velocity_fn(
                    batch.sample_before, batch.sigma, prompt_index=batch.prompt_index
                )
                reference = to_sde(
                    reference_velocity,
                    batch.sample_before,
                    batch.sigma,
                    batch.sigma_next,
                    noise_level=config.noise_level,
                    schedule=config.sde_schedule,
                    prev_sample=batch.sample_after,
                )
            # The reference log-prob is detached but the policy's is not, so the
            # penalty pulls the policy toward the reference rather than the
            # reverse.
            kl = kl_penalty(step.log_prob, reference.log_prob).mean()

        total = policy_loss + config.kl_coefficient * kl
        metrics = GRPOMetrics(
            loss=total.detach(),
            policy_loss=policy_loss.detach(),
            kl=kl.detach(),
            clip_fraction=clip_fraction.detach(),
            ratio_mean=ratio.detach().mean(),
            advantage_mean=batch.advantage.detach().mean(),
            advantage_std=batch.advantage.detach().std(unbiased=False),
        )
        return total, metrics

    def update(self, buffer: RolloutBuffer) -> list[GRPOMetrics]:
        """Run the configured inner epochs over a rollout buffer.

        Args:
            buffer: A buffer with rewards already attached.

        Returns:
            One metrics record per optimizer step taken.

        Raises:
            RuntimeError: If the buffer has no rewards.
        """
        if buffer.rewards is None:
            raise RuntimeError("attach rewards with set_rewards() before updating")
        buffer.compute_advantages(
            epsilon=self.config.advantage_epsilon,
            normalize_by_std=self.config.normalize_advantage_by_std,
            clip=self.config.advantage_clip,
        )
        collected: list[GRPOMetrics] = []
        for _ in range(self.config.inner_epochs):
            for batch in buffer.iter_minibatches(
                self.config.minibatch_size, generator=self.state.rng.sampler
            ):
                collected.append(self.train_step(batch))
        return collected

    def train_step(self, batch: RolloutBatch) -> GRPOMetrics:
        """Take one optimizer step on a minibatch.

        Mirrors the supervised step in :mod:`avgen.train`: backward, mesh-aware
        gradient clipping through :func:`avgen.parallel.clip_grad_norm`,
        non-finite rejection, optimizer step, schedule step, EMA update. The
        non-finite check is a device-side comparison folded into the returned
        metrics rather than an ``.item()``, so the step never synchronises.

        Args:
            batch: Stored transitions.

        Returns:
            Diagnostics for the step.
        """
        from avgen.parallel.comm import clip_grad_norm

        loss, metrics = self.loss(batch)
        loss.backward()
        parameters = [p for p in self.state.model.parameters() if p.requires_grad]
        grad_norm = clip_grad_norm(
            parameters, self.config.max_grad_norm, pp_mesh=self.pp_mesh
        )
        nonfinite = ~torch.isfinite(grad_norm)
        if not bool(nonfinite):
            self.state.optimizer.step()
            if self.state.schedule is not None:
                self.state.schedule.step()
            if self.state.ema is not None:
                self.state.ema.update(self.state.model)
        self.state.optimizer.zero_grad(set_to_none=True)
        self.state.step += 1
        return metrics

    def rollout(
        self,
        prompts: Sequence[str],
        *,
        sigmas: torch.Tensor,
        latent_shape: tuple[int, ...],
        reward: Any,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        initial_latents: torch.Tensor | None = None,
    ) -> RolloutBuffer:
        """Generate a group of trajectories per prompt and score them.

        Args:
            prompts: One prompt per group.
            sigmas: ``(steps + 1,)`` decreasing noise-level schedule.
            latent_shape: Per-sample latent shape without the batch dimension.
            reward: Anything satisfying :class:`avgen.rl.rewards.RewardModel`.
            device: Generation device.
            dtype: Latent dtype.
            initial_latents: Pre-drawn per-prompt initial noise.

        Returns:
            A buffer with rewards attached, ready for :meth:`update`.
        """
        from avgen.rl.rollout import generate_group

        buffer = generate_group(
            self.velocity_fn,
            prompts=prompts,
            group_size=self.config.group_size,
            sigmas=sigmas,
            latent_shape=latent_shape,
            device=device,
            dtype=dtype,
            noise_level=self.config.noise_level,
            generator=self.state.rng.sampler,
            gradient_steps=self.gradient_steps(),
            initial_latents=initial_latents,
            schedule=self.config.sde_schedule,
        )
        expanded = [prompt for prompt in prompts for _ in range(self.config.group_size)]
        assert buffer.media is not None
        scores = reward.score(buffer.media, expanded)
        buffer.set_rewards(scores, media=buffer.media)
        return buffer
