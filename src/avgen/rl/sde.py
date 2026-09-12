"""ODE to SDE conversion: what makes a flow model a policy at all.

Policy gradient needs a policy — a *distribution* over actions whose probability
can be differentiated. A rectified-flow sampler has none. Its update

    x_{s+ds} = x_s + v_theta(x_s, s) * ds

is a deterministic map, so the induced distribution over trajectories is a point
mass, its entropy is zero, and the log-probability of the transition it took is
the same under every policy. The importance ratio is identically one and the
gradient is identically zero. This is not a numerical difficulty; there is
genuinely nothing to estimate.

The fix (Flow-GRPO, arXiv:2505.05470) is to replace the probability-flow ODE
with a stochastic differential equation that has the *same marginals* at every
noise level. Such an SDE exists for any diffusion path::

    dx = [v(x, s) - (g(s)^2 / 2) * score(x, s)] ds + g(s) dw

for any diffusion coefficient ``g``. Every choice of ``g`` produces a different
distribution over *trajectories* while producing the identical distribution over
*samples*, which is exactly the freedom needed: sampling quality is unchanged,
but the policy now has entropy, and the entropy is a tunable knob.

Deriving the score for rectified flow
-------------------------------------

avgen's convention (:attr:`avgen.core.tokens.TokenStream.noise_level`) is
``s in [0, 1]`` with ``s = 0`` clean, and the forward path::

    x_s = (1 - s) * x_0 + s * eps,        v = eps - x_0

Those two equations invert exactly, with no learned quantity and no
approximation::

    x_0 = x_s - s * v
    eps = x_s + (1 - s) * v

For a Gaussian path the score is ``-eps_hat / s``, so::

    score(x_s, s) = -(x_s + (1 - s) * v(x_s, s)) / s

Substituting into the SDE and grouping terms gives the form Flow-GRPO states as
its Eq. 8 (their ``t`` is this module's ``s``, same orientation)::

    dx = [v + (g^2 / (2s)) * (x + (1 - s) * v)] ds + g dw

which is what :func:`to_sde` discretises. The Euler-Maruyama step over an
interval ``ds = s_next - s`` (negative, since sampling walks ``s`` down to 0) is
Gaussian::

    mean = x + [v + (g^2 / (2s)) * (x + (1 - s) * v)] * ds
    std  = g * sqrt(|ds|)
    x_next ~ Normal(mean, std^2 I)

and that Gaussian, evaluated at the state actually visited, is
``log pi_theta(x_next | x)``: the quantity the policy ratio is built from.

The diffusion coefficient
-------------------------

Flow-GRPO sets ``g(s) = a * sqrt(s / (1 - s))`` with ``a = 0.7`` (MixGRPO reuses
``eta = 0.7``). It diverges as ``s -> 1``, which is where sampling *starts*, so a
clamp on ``1 - s`` is not optional — without it the first step of every rollout
injects unbounded noise. The clamp is exposed rather than hidden because its
value changes the policy, and a policy that differs between the rollout and the
update is the single most common silent bug in diffusion RL.

A constant schedule is also provided. It is the conservative choice: it has no
singularity, it makes the per-step noise independent of where in the trajectory
the step falls, and it is a reasonable default for anyone who has not tuned
``a``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

__all__ = [
    "SDEStep",
    "diffusion_coefficient",
    "gaussian_log_prob",
    "ode_step",
    "to_sde",
]

#: Flow-GRPO's noise scale, reused by MixGRPO as ``eta``. Not a magic number:
#: it is the value both papers report for SD3.5-M and FLUX.1-dev.
DEFAULT_NOISE_LEVEL = 0.7

#: Floor on ``1 - s`` inside the ``sqrt(s / (1 - s))`` schedule. At s = 1 the
#: coefficient is infinite and the first rollout step would destroy the state.
#: 1e-3 keeps g below ~32*a, which is large but finite.
DEFAULT_SIGMA_FLOOR = 1e-3


def diffusion_coefficient(
    noise_level_at: torch.Tensor,
    *,
    noise_level: float,
    schedule: str = "flow_grpo",
    floor: float = DEFAULT_SIGMA_FLOOR,
) -> torch.Tensor:
    """Return ``g(s)``, the SDE's diffusion coefficient, per sample.

    Args:
        noise_level_at: ``(batch,)`` current noise level ``s`` in ``[0, 1]``.
        noise_level: Scale ``a``. Zero recovers the deterministic ODE exactly.
        schedule: ``"flow_grpo"`` for ``a*sqrt(s/(1-s))``, or ``"constant"``
            for ``a``.
        floor: Lower clamp on ``1 - s`` and on ``s`` for the flow_grpo
            schedule.

    Returns:
        ``(batch,)`` non-negative coefficients.

    Raises:
        ValueError: On a negative scale, a non-positive floor, or an unknown
            schedule name.
    """
    if noise_level < 0.0:
        raise ValueError(f"noise_level must be non-negative; got {noise_level!r}")
    if not 0.0 < floor < 1.0:
        raise ValueError(f"floor must be in (0, 1); got {floor!r}")
    if schedule == "constant":
        return torch.full_like(noise_level_at, noise_level)
    if schedule != "flow_grpo":
        raise ValueError(
            f"unknown schedule {schedule!r}; expected 'flow_grpo' or 'constant'"
        )
    sigma = noise_level_at.clamp(min=floor, max=1.0 - floor)
    return noise_level * torch.sqrt(sigma / (1.0 - sigma))


@dataclass(frozen=True, slots=True)
class SDEStep:
    """One stochastic sampler transition and everything the update needs.

    The mean and standard deviation are kept, not just the log-probability,
    because GRPO-Guard's ratio normalisation is expressed in terms of the
    *difference of means* rather than the difference of log-probabilities, and
    recovering the mean from a log-probability is not possible.

    Args:
        sample: ``(batch, ...)`` state after the step.
        log_prob: ``(batch,)`` log-density of ``sample`` under this transition,
            summed over every non-batch dimension.
        mean: ``(batch, ...)`` transition mean.
        std: ``(batch,)`` transition standard deviation, isotropic.
        deterministic: Whether the step was taken with zero noise, in which case
            ``log_prob`` is zero by convention rather than infinite.
    """

    sample: torch.Tensor
    log_prob: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    deterministic: bool = False


def _broadcast(value: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Reshape a per-sample vector so it broadcasts against a batched tensor."""
    return value.reshape(value.shape[0], *((1,) * (like.ndim - 1)))


def gaussian_log_prob(
    value: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Log-density of an isotropic Gaussian, summed over the sample dimensions.

    The normalising constant is included even though it cancels in every
    importance ratio. Two reasons: the value is then a real log-density that can
    be compared across steps and logged as a diagnostic, and the KL term against
    a reference policy uses log-probabilities from *different* transitions,
    where the constants do not cancel unless both were computed the same way.

    Args:
        value: ``(batch, ...)`` point to evaluate.
        mean: ``(batch, ...)`` distribution mean.
        std: ``(batch,)`` or broadcastable standard deviation.
        mask: ``(batch, ...)`` optional validity mask. Padding tokens must be
            excluded or the log-probability grows with the padding, and two
            samples in different buckets stop being comparable.

    Returns:
        ``(batch,)`` log-densities.

    Raises:
        ValueError: If ``value`` and ``mean`` disagree in shape.
    """
    if value.shape != mean.shape:
        raise ValueError(
            f"value shape {tuple(value.shape)} must match mean shape "
            f"{tuple(mean.shape)}"
        )
    sigma = _broadcast(std, value) if std.ndim == 1 else std
    residual = (value - mean) / sigma
    # log(2*pi)/2 per element; kept explicit so the expression reads as the
    # density it is rather than as an unnormalised score.
    per_element = (
        -0.5 * residual * residual - torch.log(sigma) - 0.5 * math.log(2.0 * math.pi)
    )
    if mask is not None:
        per_element = per_element * mask.to(per_element.dtype)
    return per_element.flatten(1).sum(dim=1)


def ode_step(
    velocity: torch.Tensor,
    x_t: torch.Tensor,
    sigma_t: torch.Tensor,
    sigma_next: torch.Tensor,
) -> torch.Tensor:
    """Take one deterministic Euler step of the probability-flow ODE.

    The path a MixGRPO trajectory follows outside its window, and the limit
    :func:`to_sde` reduces to at zero noise.

    Args:
        velocity: ``(batch, ...)`` predicted velocity ``v = eps - x_0``.
        x_t: ``(batch, ...)`` current state.
        sigma_t: ``(batch,)`` current noise level.
        sigma_next: ``(batch,)`` next noise level, below ``sigma_t``.

    Returns:
        ``(batch, ...)`` next state.
    """
    delta = _broadcast(sigma_next - sigma_t, x_t)
    return x_t + velocity * delta


def to_sde(
    velocity: torch.Tensor,
    x_t: torch.Tensor,
    sigma_t: torch.Tensor,
    sigma_next: torch.Tensor,
    *,
    noise_level: float = DEFAULT_NOISE_LEVEL,
    generator: torch.Generator | None = None,
    prev_sample: torch.Tensor | None = None,
    schedule: str = "flow_grpo",
    floor: float = DEFAULT_SIGMA_FLOOR,
    mask: torch.Tensor | None = None,
) -> SDEStep:
    """Take one Euler-Maruyama step of the marginal-preserving SDE.

    Two modes, and the difference between them is the whole mechanism of
    on-policy diffusion RL:

    * ``prev_sample is None`` — **rollout**. A state is sampled from the
      transition kernel and its log-probability under that kernel is returned.
      This is ``log pi_old``, stored in the buffer.
    * ``prev_sample`` given — **update**. No sampling happens. The transition
      kernel is rebuilt from the *current* policy's velocity and the stored
      state is scored under it. This is ``log pi_theta``, and the ratio
      ``exp(log pi_theta - log pi_old)`` is what the surrogate objective
      clips. Re-sampling here instead would make the update off-policy against
      a trajectory nobody visited.

    Args:
        velocity: ``(batch, ...)`` predicted velocity at ``x_t``.
        x_t: ``(batch, ...)`` current state.
        sigma_t: ``(batch,)`` current noise level in ``[0, 1]``, 0 = clean.
        sigma_next: ``(batch,)`` next noise level. Must not exceed ``sigma_t``;
            sampling walks the noise level down.
        noise_level: Scale ``a`` of the diffusion coefficient. ``0.0`` gives the
            deterministic ODE exactly.
        generator: RNG for the rollout draw. Pass
            ``RNGStreams.sampler`` so a rollout is reproducible.
        prev_sample: The stored next state, for the update mode.
        schedule: Diffusion-coefficient schedule, see
            :func:`diffusion_coefficient`.
        floor: Clamp for the ``flow_grpo`` schedule.
        mask: ``(batch, ...)`` validity mask excluding padding from the
            log-probability.

    Returns:
        The transition, its log-probability, and its Gaussian parameters.

    Raises:
        ValueError: If the noise level rises rather than falls, or if
            ``prev_sample`` does not match ``x_t`` in shape.
    """
    if bool((sigma_next > sigma_t + 1e-6).any()):
        raise ValueError(
            "sigma_next must not exceed sigma_t; sampling integrates the noise "
            "level downward toward 0 (clean)"
        )
    if prev_sample is not None and prev_sample.shape != x_t.shape:
        raise ValueError(
            f"prev_sample shape {tuple(prev_sample.shape)} must match x_t "
            f"{tuple(x_t.shape)}"
        )
    delta_scalar = sigma_next - sigma_t
    delta = _broadcast(delta_scalar, x_t)
    g = diffusion_coefficient(
        sigma_t, noise_level=noise_level, schedule=schedule, floor=floor
    )

    if noise_level == 0.0:
        # The deterministic limit, taken as an exact branch rather than as a
        # limit of the Gaussian: the Dirac transition has infinite log-density,
        # and the honest value for the *ratio* it induces is 1, i.e. a
        # log-probability of 0 that contributes no policy gradient. That is not
        # a numerical dodge — it is precisely why a deterministic sampler cannot
        # be improved by policy gradient, which is what this module exists to
        # fix.
        mean = ode_step(velocity, x_t, sigma_t, sigma_next)
        sample = mean if prev_sample is None else prev_sample
        zeros = torch.zeros(x_t.shape[0], device=x_t.device, dtype=torch.float32)
        return SDEStep(
            sample=sample,
            log_prob=zeros,
            mean=mean,
            std=zeros,
            deterministic=True,
        )

    # score(x, s) = -(x + (1 - s) v) / s, exact for rectified flow; see module
    # docstring. Clamped from below because s = 0 is the clean endpoint, where
    # the score is undefined and no SDE step is ever taken.
    sigma = _broadcast(sigma_t.clamp(min=floor), x_t)
    eps_hat = x_t + (1.0 - sigma) * velocity
    g_squared = _broadcast(g * g, x_t)
    drift = velocity + (g_squared / (2.0 * sigma)) * eps_hat
    mean = x_t + drift * delta
    # |ds| because sampling runs the noise level downward: ds is negative and
    # a variance cannot be.
    std = g * torch.sqrt(delta_scalar.abs().clamp_min(0.0))

    if prev_sample is None:
        noise = torch.randn(
            x_t.shape, device=x_t.device, dtype=x_t.dtype, generator=generator
        )
        sample = mean + _broadcast(std, x_t) * noise
    else:
        sample = prev_sample

    log_prob = gaussian_log_prob(
        sample.float(), mean.float(), std.float().clamp_min(1e-12), mask=mask
    )
    return SDEStep(sample=sample, log_prob=log_prob, mean=mean, std=std)
