"""Group rollouts: generating the trajectories a group-relative update needs.

GRPO has no value network. The baseline it subtracts is the mean reward of a
*group* of samples drawn from the same prompt, which is only a valid baseline if
the group members are genuinely comparable — same prompt, same conditions, same
everything except the policy's own stochastic choices.

Shared initial noise
--------------------

:func:`generate_group` draws **one** initial latent per prompt and repeats it
across the group. This is DanceGRPO's trick and it matters more than it looks.

Decompose a sample's reward into what the initial noise decided and what the
policy's per-step actions decided. With independent initial noise, the
within-group reward spread is dominated by the first term: some seeds are simply
easier prompts to satisfy. The group-relative advantage then rewards the policy
for a draw it did not make, and since the advantage is normalised by the group
standard deviation, that noise variance also *shrinks* the signal from the
actions that the policy did choose. The gradient is not merely noisier; its
signal-to-noise ratio is reduced twice over.

Fixing the initial noise within a group cancels the first term exactly. Every
reward difference inside a group is then attributable to the SDE noise injected
during sampling — which is precisely the quantity the policy controls and the
quantity the log-probability ratio is written in terms of.

Memory
------

Every SDE step must be replayable, so the buffer stores the state before and
after each step, plus the transition's mean and standard deviation. For a video
model this is the dominant cost of GRPO and the reason MixGRPO exists: with a
window of ``W`` gradient-carrying steps out of ``T``, only ``W`` steps need to
be stored, and both memory and backward cost fall by ``T/W``. Steps outside the
window are recorded with ``requires_grad`` false and no stored statistics.

Distribution
------------

**A group never spans ranks.** Prompts are distributed whole, with all ``G``
members of a group generated on the same rank, so the group statistics are
computed locally with no collective at all. The alternative — sharding a group
across ranks and all-reducing its mean and variance — costs a synchronisation
per group per step and makes the advantage depend on the world size, which
breaks the reproducibility rule. Rewards are still gathered, but only for
logging and for the global reward statistics.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

from avgen.rl.sde import SDEStep, ode_step, to_sde

__all__ = [
    "RolloutBatch",
    "RolloutBuffer",
    "VelocityFn",
    "distribute_prompts",
    "gather_rewards",
    "generate_group",
    "group_advantages",
    "model_velocity_fn",
    "rollout_velocity_batch",
]


class VelocityFn(Protocol):
    """Predicts the flow velocity at a state and noise level.

    The seam between the RL package and the model. Everything upstream of this
    protocol — how a ``ModelInput`` is assembled, how text is encoded, whether
    guidance is applied — is the caller's business, which keeps rollouts
    testable against a twenty-line stub and keeps this module from depending on
    :mod:`avgen.models`.
    """

    def __call__(
        self, sample: torch.Tensor, sigma: torch.Tensor, *, prompt_index: torch.Tensor
    ) -> torch.Tensor:
        """Return ``(batch, ...)`` velocity for a batch of states."""
        ...


def model_velocity_fn(
    model: Any,
    *,
    patchifier: Any,
    positions: torch.Tensor,
    mask: torch.Tensor,
    text_features: torch.Tensor,
    text_mask: torch.Tensor,
) -> VelocityFn:
    """Adapt an avgen model to the :class:`VelocityFn` protocol.

    Builds a :class:`~avgen.core.model_input.ModelInput` per call and unfolds the
    model's token-space output back into the latent grid the sampler works in.
    Imported lazily so this module stands alone.

    Args:
        model: Any module implementing ``forward(ModelInput) -> ModelOutput``.
        patchifier: The model's patchifier.
        positions: ``(batch, frames)`` physical frame times in seconds.
        mask: ``(batch, frames, height, width)`` token validity.
        text_features: ``(batch, tokens, width)`` frozen text context.
        text_mask: ``(batch, tokens)`` text validity.

    Returns:
        A callable satisfying :class:`VelocityFn`.
    """
    from avgen.core.model_input import ModelInput
    from avgen.core.patchify import unpatchify_grid
    from avgen.core.tokens import TextContext, TokenStream

    def _velocity(
        sample: torch.Tensor, sigma: torch.Tensor, *, prompt_index: torch.Tensor
    ) -> torch.Tensor:
        del prompt_index  # conditioning is already expanded to the group
        stream = patchifier.to_tokens(
            sample,
            positions=positions,
            mask=mask,
            noise_level=sigma.to(torch.float32),
        )
        inputs = ModelInput(
            video=stream,
            audio=TokenStream.empty_like(
                sample.shape[0], stream.width, device=sample.device
            ),
            text=TextContext(features=text_features, mask=text_mask),
        )
        output = model(inputs)
        return unpatchify_grid(output.video, stream.layout)

    return _velocity


@dataclass(frozen=True, slots=True)
class RolloutBatch:
    """One minibatch of stored transitions, ready for a policy-gradient update.

    Args:
        sample_before: ``(batch, ...)`` state entering the step.
        sample_after: ``(batch, ...)`` state the rollout policy chose.
        sigma: ``(batch,)`` noise level at the step.
        sigma_next: ``(batch,)`` noise level after the step.
        log_prob_old: ``(batch,)`` log-probability under the rollout policy.
        mean_old: ``(batch, ...)`` transition mean under the rollout policy.
        std_old: ``(batch,)`` transition standard deviation.
        advantage: ``(batch,)`` group-relative advantage of the trajectory this
            step belongs to. Constant along a trajectory: GRPO assigns the whole
            trajectory's advantage to every step, because the reward is only
            observed at the end and there is no value function to bootstrap an
            intermediate credit assignment from.
        prompt_index: ``(batch,)`` which prompt each step came from.
        step_index: ``(batch,)`` position of the step in its trajectory,
            needed by GRPO-Guard's per-timestep reweighting.
    """

    sample_before: torch.Tensor
    sample_after: torch.Tensor
    sigma: torch.Tensor
    sigma_next: torch.Tensor
    log_prob_old: torch.Tensor
    mean_old: torch.Tensor
    std_old: torch.Tensor
    advantage: torch.Tensor
    prompt_index: torch.Tensor
    step_index: torch.Tensor

    @property
    def size(self) -> int:
        """Number of transitions in this minibatch."""
        return int(self.sample_before.shape[0])


def group_advantages(
    rewards: torch.Tensor,
    group_size: int,
    *,
    epsilon: float = 1e-4,
    normalize_by_std: bool = True,
    clip: float | None = 5.0,
) -> torch.Tensor:
    """Compute the group-relative advantage ``(r - mean) / (std + eps)``.

    This is the whole of GRPO's credit assignment, and the absence of a value
    network is the point rather than an economy. A learned critic for a video
    diffusion policy would be a second video-sized model, trained on a reward
    signal that arrives once per trajectory, and its bias would be
    indistinguishable in the loss from the policy's own error. The group mean is
    an unbiased baseline that costs no parameters and no extra training signal —
    it costs ``G`` samples per prompt instead, which for a diffusion model is a
    trade worth making because sampling parallelises and critic training does
    not.

    Args:
        rewards: ``(num_prompts * group_size,)`` rewards, grouped contiguously.
        group_size: ``G``.
        epsilon: Added to the group standard deviation.
        normalize_by_std: Whether to divide by the group standard deviation.
            Dividing is standard GRPO and is what makes one clip range work
            across rewards of different scales. Not dividing (Dr. GRPO's
            correction) removes a bias that favours prompts the policy already
            answers consistently — a low-variance group gets its small
            differences amplified, so easy prompts contribute gradient out of
            proportion to what they teach.
        clip: Symmetric clamp on the advantage, or ``None``. MixGRPO clips to
            5.0; an unclipped advantage from a near-degenerate group can be
            arbitrarily large and will dominate a whole batch.

    Returns:
        ``(num_prompts * group_size,)`` advantages, zero-mean within each group.

    Raises:
        ValueError: If the reward count is not a multiple of the group size, or
            the group size is below 2 (a group of one has no baseline).
    """
    if group_size < 2:
        raise ValueError(
            f"group_size must be at least 2; got {group_size}. A group of one "
            "has no baseline to be relative to, and every advantage is zero"
        )
    if rewards.numel() % group_size != 0:
        raise ValueError(
            f"reward count {rewards.numel()} is not a multiple of "
            f"group_size={group_size}"
        )
    grouped = rewards.float().reshape(-1, group_size)
    centred = grouped - grouped.mean(dim=1, keepdim=True)
    if normalize_by_std:
        centred = centred / (grouped.std(dim=1, unbiased=False, keepdim=True) + epsilon)
    if clip is not None:
        centred = centred.clamp(-clip, clip)
    return centred.reshape(-1)


@dataclass(slots=True)
class RolloutBuffer:
    """Stored transitions for one round of rollouts.

    Args:
        group_size: ``G``, the number of samples per prompt.
        prompts: The prompts, one per group.
        rewards: ``(num_prompts * group_size,)`` terminal rewards, filled by
            :meth:`set_rewards`.
        advantages: Group-relative advantages, filled by
            :meth:`compute_advantages`.
        media: The generated media, kept so a reward can be recomputed or a
            sample logged without regenerating.
    """

    group_size: int
    prompts: tuple[str, ...] = ()
    rewards: torch.Tensor | None = None
    advantages: torch.Tensor | None = None
    media: torch.Tensor | None = None
    _steps: list[dict[str, torch.Tensor]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        """Validate the group size.

        Raises:
            ValueError: If the group size is below 2.
        """
        if self.group_size < 2:
            raise ValueError(f"group_size must be at least 2; got {self.group_size}")

    def __len__(self) -> int:
        """Number of stored transitions."""
        return len(self._steps)

    @property
    def num_samples(self) -> int:
        """Trajectories in the buffer."""
        if not self._steps:
            return 0
        return int(self._steps[0]["sample_before"].shape[0])

    def add_step(
        self,
        *,
        step_index: int,
        sample_before: torch.Tensor,
        step: SDEStep,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
    ) -> None:
        """Record one gradient-carrying transition.

        Everything is detached. The rollout is generated under ``no_grad``; the
        gradient enters later, when the update recomputes the velocity for these
        stored states under the current policy. Storing an attached tensor here
        would retain the whole rollout's autograd graph, which for a 25-step
        video trajectory is the fastest way to run out of memory.

        Args:
            step_index: Position in the trajectory.
            sample_before: State entering the step.
            step: The transition taken.
            sigma: Noise level at the step.
            sigma_next: Noise level after the step.
        """
        count = sample_before.shape[0]
        self._steps.append(
            {
                "sample_before": sample_before.detach(),
                "sample_after": step.sample.detach(),
                "sigma": sigma.detach(),
                "sigma_next": sigma_next.detach(),
                "log_prob_old": step.log_prob.detach(),
                "mean_old": step.mean.detach(),
                "std_old": step.std.detach(),
                "step_index": torch.full(
                    (count,), step_index, dtype=torch.int64, device=sample_before.device
                ),
            }
        )

    def set_rewards(self, rewards: torch.Tensor, *, media: torch.Tensor | None) -> None:
        """Attach terminal rewards to the stored trajectories.

        Args:
            rewards: ``(num_samples,)`` rewards.
            media: The decoded media the rewards were computed from.

        Raises:
            ValueError: If the reward count does not match the trajectory count.
        """
        if rewards.numel() != self.num_samples:
            raise ValueError(
                f"expected {self.num_samples} rewards to match the trajectories; "
                f"got {rewards.numel()}"
            )
        self.rewards = rewards.detach().float()
        self.media = media

    def compute_advantages(
        self,
        *,
        epsilon: float = 1e-4,
        normalize_by_std: bool = True,
        clip: float | None = 5.0,
    ) -> torch.Tensor:
        """Compute and store the group-relative advantages.

        Args:
            epsilon: Added to the group standard deviation.
            normalize_by_std: Whether to divide by the group standard deviation.
            clip: Symmetric advantage clamp, or ``None``.

        Returns:
            ``(num_samples,)`` advantages.

        Raises:
            RuntimeError: If rewards have not been set.
        """
        if self.rewards is None:
            raise RuntimeError("call set_rewards() before compute_advantages()")
        self.advantages = group_advantages(
            self.rewards,
            self.group_size,
            epsilon=epsilon,
            normalize_by_std=normalize_by_std,
            clip=clip,
        )
        return self.advantages

    def iter_minibatches(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
        shuffle: bool = True,
    ) -> Iterator[RolloutBatch]:
        """Yield shuffled minibatches over the flattened (trajectory, step) grid.

        Flattening across steps as well as trajectories is deliberate. Taking a
        whole trajectory as one minibatch correlates every sample in it — the
        same reward, the same advantage, adjacent states — and the resulting
        gradient estimate has far higher variance than the batch size suggests.

        Args:
            batch_size: Transitions per minibatch.
            generator: RNG for the shuffle.
            shuffle: Whether to shuffle. Off gives a deterministic sweep, which
                is what a test wants.

        Yields:
            Minibatches of transitions.

        Raises:
            RuntimeError: If advantages have not been computed.
            ValueError: If ``batch_size`` is not positive.
        """
        if self.advantages is None:
            raise RuntimeError("call compute_advantages() before iterating")
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive; got {batch_size}")
        flat = self._flatten()
        total = int(flat["sample_before"].shape[0])
        order = (
            torch.randperm(total, generator=generator, device="cpu")
            if shuffle
            else torch.arange(total, device="cpu")
        )
        order = order.to(flat["sample_before"].device)
        for start in range(0, total, batch_size):
            index = order[start : start + batch_size]
            yield RolloutBatch(
                sample_before=flat["sample_before"][index],
                sample_after=flat["sample_after"][index],
                sigma=flat["sigma"][index],
                sigma_next=flat["sigma_next"][index],
                log_prob_old=flat["log_prob_old"][index],
                mean_old=flat["mean_old"][index],
                std_old=flat["std_old"][index],
                advantage=flat["advantage"][index],
                prompt_index=flat["prompt_index"][index],
                step_index=flat["step_index"][index],
            )

    def _flatten(self) -> dict[str, torch.Tensor]:
        """Concatenate every stored step into one flat transition table."""
        assert self.advantages is not None
        samples = self.num_samples
        device = self._steps[0]["sample_before"].device
        prompt_index = torch.arange(samples, device=device) // self.group_size
        flat: dict[str, torch.Tensor] = {}
        for key in (
            "sample_before",
            "sample_after",
            "sigma",
            "sigma_next",
            "log_prob_old",
            "mean_old",
            "std_old",
            "step_index",
        ):
            flat[key] = torch.cat([step[key] for step in self._steps], dim=0)
        repeats = len(self._steps)
        flat["advantage"] = self.advantages.to(device).repeat(repeats)
        flat["prompt_index"] = prompt_index.repeat(repeats)
        return flat


def distribute_prompts(
    prompts: Sequence[str], *, data_rank: int, data_world: int
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Split prompts across data-parallel ranks, keeping groups whole.

    Whole prompts, never fractions of a group: see the module docstring on why
    a group that spans ranks turns every advantage computation into a
    collective and makes the result depend on the world size.

    Args:
        prompts: Every prompt in the round, identical on every rank.
        data_rank: This rank's index in the data-parallel dimension.
        data_world: Size of the data-parallel dimension.

    Returns:
        ``(local_prompts, global_indices)``.

    Raises:
        ValueError: If the coordinates are inconsistent.
    """
    if data_world < 1 or not 0 <= data_rank < data_world:
        raise ValueError(
            f"invalid data coordinates: data_rank={data_rank}, data_world={data_world}"
        )
    indices = tuple(range(data_rank, len(prompts), data_world))
    return tuple(prompts[i] for i in indices), indices


def gather_rewards(rewards: torch.Tensor, mesh: Any | None = None) -> torch.Tensor:
    """Gather per-rank rewards across the data mesh for logging.

    Only for statistics. The advantage itself stays local by construction, so
    this is never on the critical path of the update and a failure here degrades
    a log line rather than the run.

    Args:
        rewards: ``(local_samples,)`` rewards on this rank.
        mesh: The data mesh, or ``None`` for the single-process case.

    Returns:
        ``(world * local_samples,)`` rewards, or the input unchanged when not
        distributed.
    """
    import torch.distributed as dist

    if mesh is None or not (dist.is_available() and dist.is_initialized()):
        return rewards
    group = mesh.get_group()
    world = dist.get_world_size(group)
    if world == 1:
        return rewards
    buffer = [torch.empty_like(rewards) for _ in range(world)]
    dist.all_gather(buffer, rewards.contiguous(), group=group)
    return torch.cat(buffer, dim=0)


def generate_group(
    velocity_fn: VelocityFn,
    *,
    prompts: Sequence[str],
    group_size: int,
    sigmas: torch.Tensor,
    latent_shape: tuple[int, ...],
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    noise_level: float = 0.7,
    generator: torch.Generator | None = None,
    gradient_steps: Sequence[int] | None = None,
    initial_latents: torch.Tensor | None = None,
    schedule: str = "flow_grpo",
) -> RolloutBuffer:
    """Sample ``group_size`` trajectories per prompt with shared initial noise.

    Args:
        velocity_fn: The policy's velocity predictor.
        prompts: One prompt per group.
        group_size: ``G``.
        sigmas: ``(steps + 1,)`` monotonically decreasing noise-level schedule,
            from 1 (pure noise) down to 0 (clean).
        latent_shape: Per-sample latent shape, without the batch dimension.
        device: Device to generate on.
        dtype: Latent dtype.
        noise_level: SDE noise scale ``a``.
        generator: RNG for both the initial noise and the SDE draws.
        gradient_steps: Step indices that use the SDE path and are stored for
            the update. ``None`` means every step — plain Flow-GRPO. A
            contiguous window is MixGRPO; see
            :class:`avgen.rl.grpo.WindowSchedule`.
        initial_latents: ``(num_prompts, *latent_shape)`` pre-drawn initial
            noise, one per *prompt* not per sample. Supply this to reuse the
            same seeds across training iterations, which removes seed variance
            from the comparison between two policy versions.
        schedule: Diffusion-coefficient schedule name.

    Returns:
        A buffer holding the stored transitions and the final latents, with
        rewards still to be attached.

    Raises:
        ValueError: If the sigma schedule is not decreasing or too short, or if
            supplied initial latents have the wrong shape.
    """
    if sigmas.ndim != 1 or sigmas.numel() < 2:
        raise ValueError(
            f"sigmas must be a 1-D schedule with at least two entries; got "
            f"{tuple(sigmas.shape)}"
        )
    if bool((sigmas[1:] > sigmas[:-1]).any()):
        raise ValueError("sigmas must be non-increasing (1 = noise down to 0 = clean)")
    num_prompts = len(prompts)
    num_steps = int(sigmas.numel()) - 1
    target = torch.device(device)
    active = (
        frozenset(range(num_steps))
        if gradient_steps is None
        else frozenset(int(i) for i in gradient_steps)
    )

    if initial_latents is None:
        seed_noise = torch.randn(
            (num_prompts, *latent_shape),
            device=target,
            dtype=dtype,
            generator=generator,
        )
    else:
        if tuple(initial_latents.shape) != (num_prompts, *latent_shape):
            raise ValueError(
                f"initial_latents must be {(num_prompts, *latent_shape)}; got "
                f"{tuple(initial_latents.shape)}"
            )
        seed_noise = initial_latents.to(device=target, dtype=dtype)
    # repeat_interleave, not repeat: group members must be adjacent so that
    # reshape(-1, group_size) in group_advantages groups the right samples. A
    # plain repeat would interleave prompts and silently compute each advantage
    # against the wrong baseline.
    state = seed_noise.repeat_interleave(group_size, dim=0)

    buffer = RolloutBuffer(group_size=group_size, prompts=tuple(prompts))
    prompt_index = torch.arange(num_prompts, device=target).repeat_interleave(
        group_size
    )
    batch = state.shape[0]

    with torch.no_grad():
        for index in range(num_steps):
            sigma = sigmas[index].to(target).expand(batch).contiguous()
            sigma_next = sigmas[index + 1].to(target).expand(batch).contiguous()
            velocity = velocity_fn(state, sigma, prompt_index=prompt_index)
            if index not in active:
                # Outside the window: deterministic, unstored, no gradient. This
                # is the MixGRPO saving — the step still advances the trajectory
                # but costs nothing beyond one forward pass.
                state = ode_step(velocity, state, sigma, sigma_next)
                continue
            step = to_sde(
                velocity,
                state,
                sigma,
                sigma_next,
                noise_level=noise_level,
                generator=generator,
                schedule=schedule,
            )
            buffer.add_step(
                step_index=index,
                sample_before=state,
                step=step,
                sigma=sigma,
                sigma_next=sigma_next,
            )
            state = step.sample
    buffer.media = state
    return buffer


def rollout_velocity_batch(
    velocity_fn: VelocityFn, batch: RolloutBatch
) -> torch.Tensor:
    """Evaluate the current policy on a stored minibatch.

    Args:
        velocity_fn: The current policy.
        batch: Stored transitions.

    Returns:
        ``(batch, ...)`` velocities, with gradient.
    """
    return velocity_fn(
        batch.sample_before, batch.sigma, prompt_index=batch.prompt_index
    )
