"""Diffusion-DPO / flow-DPO: preference learning without a reward model.

GRPO needs a reward model, a rollout loop, a reference policy and a stored
trajectory buffer. DPO needs a pile of ``(winner, loser)`` pairs and one extra
frozen forward pass. When preference pairs already exist — and for a video model
they usually do, because they fall out of the same human evaluation that decides
which checkpoint ships — DPO is a fraction of the machinery for a comparable
result. It cannot exceed the preferences it was given, which GRPO can; that is
the trade.

The objective (Diffusion-DPO, arXiv:2311.12908 Eq. 14) is::

    L = -log sigmoid( -beta * T * omega(t) * [
            (||eps_w - eps_theta(x_w)||^2 - ||eps_w - eps_ref(x_w)||^2)
          - (||eps_l - eps_theta(x_l)||^2 - ||eps_l - eps_ref(x_l)||^2) ] )

Read the bracket as "how much better the policy got at the winner, relative to
the reference, minus how much better it got at the loser". Pushing it negative —
improving on the winner more than on the loser — drives the sigmoid toward 1 and
the loss toward zero. The reference terms are what stop the model from
satisfying this by simply becoming better at *everything*, which would be a
capability gain rather than a preference.

The flow-matching form
----------------------

The paper derives only the epsilon-prediction loss; a velocity-prediction
version is not stated in it. It does not need to be guessed, because the
substitution is exact. For rectified flow, ``x_s = (1-s) x_0 + s eps`` and
``v = eps - x_0`` give the identity ``eps = x_s + (1-s) v``, so for a prediction
``v_theta`` at the same state::

    ||eps - eps_theta||^2 = (1 - s)^2 * ||v - v_theta||^2

Substituting velocities for noises therefore changes the objective only by a
per-timestep factor ``(1-s)^2``, which is absorbed into the ``omega(t)`` weight
the paper already sets to a constant in practice. :class:`DPOObjective`
implements the velocity form and exposes ``snr_weighting`` for anyone who wants
the factor back.

Interface
---------

:class:`DPOObjective` satisfies :class:`avgen.train.Objective` — the same
``(model, batch, rng, patchifier=, cp_mesh=) -> ObjectiveOutput`` signature the
flow-matching objective uses. **That compatibility is the design goal of this
module.** Preference tuning then needs no RL trainer, no rollout buffer and no
new loop: it is ``Trainer(state, DPOObjective(...), parallel, config)`` and every
piece of infrastructure that already exists — checkpointing, EMA, gradient
accumulation, context parallelism, telemetry — applies unchanged. Anything that
required a bespoke trainer would be a second code path to keep in sync with the
first, and the second one is always the one that rots.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

__all__ = ["DPOConfig", "DPOObjective", "ObjectiveOutput", "dpo_loss"]


@dataclass(frozen=True, slots=True)
class ObjectiveOutput:
    """Loss components returned by an objective.

    Structurally identical to :class:`avgen.train.objective.ObjectiveOutput` and
    used only when that module is not importable, so this package can be tested
    and linted before the training subsystem lands. :meth:`DPOObjective.__call__`
    returns the real type whenever it is available; delete this once
    :mod:`avgen.train` is merged.

    Args:
        loss: Total objective, an fp32 scalar carrying gradient.
        video_loss: Video component.
        audio_loss: Audio component.
    """

    loss: torch.Tensor
    video_loss: torch.Tensor
    audio_loss: torch.Tensor


def _objective_output(
    loss: torch.Tensor, video_loss: torch.Tensor, audio_loss: torch.Tensor
) -> Any:
    """Build the framework's ObjectiveOutput, falling back to the local one.

    Args:
        loss: Total loss.
        video_loss: Video component.
        audio_loss: Audio component.

    Returns:
        An ``ObjectiveOutput`` from :mod:`avgen.train` when importable, else the
        local stand-in.
    """
    try:
        from avgen.train.objective import ObjectiveOutput as TrainObjectiveOutput
    except ImportError:
        return ObjectiveOutput(loss=loss, video_loss=video_loss, audio_loss=audio_loss)
    return TrainObjectiveOutput(loss=loss, video_loss=video_loss, audio_loss=audio_loss)


@dataclass(frozen=True, slots=True)
class DPOConfig:
    """Settings for the pairwise preference objective.

    Args:
        beta: Preference sharpness, and simultaneously the implicit KL weight
            against the reference. **Its scale depends entirely on how the
            squared error is reduced.** The reference implementation averages
            over the latent dimensions and reports beta in the 2000-5000 range
            (2000 for SD1.5, 5000 for SDXL); the same beta with a summed
            reduction would be larger by the latent size and would saturate the
            sigmoid on the first batch. This implementation uses the mean
            reduction, so those published values apply.
        loss_type: ``"sigmoid"`` for the published logistic loss, ``"hinge"``
            for a margin form that stops pushing once the pair is separated, or
            ``"ipo"`` for the squared form that is less prone to over-fitting a
            small preference set.
        label_smoothing: Probability mass assigned to the pair being labelled
            the wrong way round. Non-zero smoothing bounds the gradient of a
            confidently-wrong pair, which matters because human preference data
            has a real and irreducible label-noise rate.
        share_noise: Whether the winner and loser see the same noise draw. The
            paper draws them independently. Sharing couples the two branches, so
            the difference of their errors reflects only the difference between
            the *samples* rather than also the difference between two noise
            draws — a large variance reduction. It is not a free one: the loss
            is nonlinear in each term, so the coupled estimator is not equal in
            expectation to the independent one. Shared is the default here
            because it is what every implementation does and because it makes
            the "identical pair implies no signal" property exact.
        center_loss: Whether to subtract ``log 2``, the loss of a pair carrying
            no preference information. Purely a constant, so gradients are
            unchanged; it makes zero mean "at chance" instead of 0.693, which is
            worth it on a curve someone reads every day.
        snr_weighting: Whether to restore the ``(1-s)^2`` factor that converts
            the velocity residual back into a noise residual. Off by default,
            matching the paper's constant ``omega``.
        beta_scales_with_steps: Whether ``beta`` is multiplied by the nominal
            step count ``T``, as the paper's ``beta * T`` written literally. Off
            by default because the published beta values already absorb it.
        timestep_count: The ``T`` used when ``beta_scales_with_steps`` is on.

    Raises:
        ValueError: On a non-positive beta, an unknown loss type, or a
            smoothing outside ``[0, 0.5)``.
    """

    beta: float = 5000.0
    loss_type: str = "sigmoid"
    label_smoothing: float = 0.0
    share_noise: bool = True
    center_loss: bool = True
    snr_weighting: bool = False
    beta_scales_with_steps: bool = False
    timestep_count: int = 1000

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: On an out-of-range field.
        """
        if self.beta <= 0.0:
            raise ValueError(f"beta must be positive; got {self.beta!r}")
        if self.loss_type not in {"sigmoid", "hinge", "ipo"}:
            raise ValueError(
                f"loss_type must be 'sigmoid', 'hinge' or 'ipo'; got {self.loss_type!r}"
            )
        if not 0.0 <= self.label_smoothing < 0.5:
            raise ValueError(
                f"label_smoothing must be in [0, 0.5); got {self.label_smoothing!r}"
            )
        if isinstance(self.timestep_count, bool) or self.timestep_count < 1:
            raise ValueError(
                "timestep_count must be a positive integer; got "
                f"{self.timestep_count!r}"
            )

    @property
    def effective_beta(self) -> float:
        """Beta including the optional step-count factor."""
        if self.beta_scales_with_steps:
            return self.beta * self.timestep_count
        return self.beta


def _squared_error(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
    """Return the per-sample mean squared error over the latent dimensions.

    Mean rather than sum, matching the reference implementation, because beta is
    calibrated against it and because a mean keeps the loss scale independent of
    resolution — otherwise a 720p pair and a 360p pair enter the same batch with
    a four-to-one weighting nobody asked for.

    Args:
        prediction: ``(batch, ...)`` model output.
        target: ``(batch, ...)`` regression target.
        mask: ``(batch, ...)`` validity mask, or ``None``.

    Returns:
        ``(batch,)`` errors.
    """
    squared = (prediction.float() - target.float()) ** 2
    if mask is None:
        return squared.flatten(1).mean(dim=1)
    weights = mask.to(squared.dtype)
    total = weights.flatten(1).sum(dim=1).clamp_min(1.0)
    return (squared * weights).flatten(1).sum(dim=1) / total


def dpo_loss(
    policy_winner_error: torch.Tensor,
    policy_loser_error: torch.Tensor,
    reference_winner_error: torch.Tensor,
    reference_loser_error: torch.Tensor,
    config: DPOConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the pairwise preference loss from four regression errors.

    Args:
        policy_winner_error: ``(pairs,)`` policy error on the preferred sample.
        policy_loser_error: ``(pairs,)`` policy error on the rejected sample.
        reference_winner_error: ``(pairs,)`` reference error on the preferred
            sample.
        reference_loser_error: ``(pairs,)`` reference error on the rejected
            sample.
        config: Loss settings.

    Returns:
        ``(loss, margin)``. The margin is the logit inside the sigmoid: positive
        means the policy already prefers the winner more than the reference
        does, and its sign agreement rate is the accuracy metric to log.
    """
    policy_difference = policy_winner_error - policy_loser_error
    reference_difference = reference_winner_error - reference_loser_error
    # The overall negative sign is the paper's: a *lower* policy error on the
    # winner must give a *higher* logit, and the errors enter with their own
    # sign.
    margin = -config.effective_beta * (policy_difference - reference_difference)
    if config.loss_type == "hinge":
        # No reward once the pair is separated by one unit, which stops a
        # well-separated pair from dominating the batch gradient forever.
        per_pair = torch.relu(1.0 - margin)
    elif config.loss_type == "ipo":
        # IPO regresses the margin onto a fixed target instead of maximising it,
        # which is what removes DPO's tendency to drive the margin to infinity
        # on a small preference set.
        per_pair = (margin - 0.5) ** 2
    else:
        smoothing = config.label_smoothing
        per_pair = -(
            (1.0 - smoothing) * torch.nn.functional.logsigmoid(margin)
            + smoothing * torch.nn.functional.logsigmoid(-margin)
        )
        if config.center_loss:
            # log 2 is the loss of a pair carrying no information (margin 0).
            # Subtracting it is a constant shift: gradients are untouched, but
            # "zero" now means "at chance" on the logged curve.
            floor = -((1.0 - smoothing) * math.log(0.5) + smoothing * math.log(0.5))
            per_pair = per_pair - floor
    return per_pair.mean(), margin


class DPOObjective:
    """Diffusion-DPO as an :class:`avgen.train.Objective`.

    Drops into the existing trainer unchanged — that is the point of the class.
    See the module docstring.

    **Batch convention.** A preference pair arrives as two samples in one
    :class:`~avgen.core.batch.MediaBatch`: the first half of the batch is the
    winners and the second half is the losers, index-aligned, sharing a prompt.
    This is the layout the reference implementation uses, and it is what lets
    both branches go through the model in a single forward pass — halving the
    number of kernel launches and letting the shared text conditioning be
    computed once.

    Args:
        config: Loss settings.
        reference_model: The frozen reference. Usually a deep copy of the
            initial policy. Put it in ``eval()`` and freeze it; this class does
            not mutate it.
        timestep_sampler: Anything with
            ``sample(batch, *, device, generator, sequence_length=None)``.
            ``None`` uses a uniform sampler, which is what the paper uses.

    Attributes:
        last_margin: ``(pairs,)`` logits from the most recent call, detached,
            for logging preference accuracy.
    """

    __slots__ = ("config", "last_margin", "reference_model", "timestep_sampler")

    def __init__(
        self,
        config: DPOConfig,
        reference_model: Any,
        *,
        timestep_sampler: Any | None = None,
    ) -> None:
        self.config = config
        self.reference_model = reference_model
        self.timestep_sampler = timestep_sampler
        #: Mean logit of the last batch, detached. Its sign agreement rate is
        #: the preference accuracy: a margin that keeps growing while accuracy
        #: is flat means the model is inflating a separation it already has
        #: rather than learning new preferences.
        self.last_margin: torch.Tensor | None = None

    def _sample_noise_levels(
        self, pairs: int, *, device: torch.device, generator: torch.Generator | None
    ) -> torch.Tensor:
        """Draw one noise level per *pair*, shared by winner and loser.

        Sharing the timestep is not optional. The two error terms are subtracted
        from each other, and the regression error of a flow model varies by
        orders of magnitude across the noise schedule; drawing independently
        would make that variation, rather than the preference, the dominant term
        in the difference.

        Args:
            pairs: Number of preference pairs.
            device: Device to draw on.
            generator: RNG stream.

        Returns:
            ``(pairs,)`` noise levels in ``[0, 1]``.
        """
        if self.timestep_sampler is not None:
            return self.timestep_sampler.sample(
                pairs, device=device, generator=generator
            ).to(torch.float32)
        return torch.rand(
            pairs, device=device, generator=generator, dtype=torch.float32
        )

    def __call__(
        self,
        model: Any,
        batch: Any,
        rng: Any,
        *,
        patchifier: Any,
        cp_mesh: Any | None = None,
    ) -> Any:
        """Evaluate the preference loss on a batch of stacked pairs.

        Args:
            model: The policy.
            batch: A :class:`~avgen.core.batch.MediaBatch` whose first half is
                winners and second half losers.
            rng: :class:`~avgen.core.rng.RNGStreams`.
            patchifier: The model's patchifier.
            cp_mesh: Context-parallel mesh, applied to both branches so the two
                error terms are computed over the same token shard.

        Returns:
            An ``ObjectiveOutput``.

        Raises:
            ValueError: If the batch size is odd, which means the pairs do not
                line up and every preference would be computed against the wrong
                partner.
        """
        from avgen.core.model_input import ModelInput
        from avgen.core.tokens import TextContext, TokenStream

        total = int(batch.video.shape[0])
        if total % 2 != 0:
            raise ValueError(
                f"DPO needs an even batch of stacked (winner, loser) pairs; got {total}"
            )
        pairs = total // 2
        device = batch.video.device
        winner = batch.video[:pairs]
        loser = batch.video[pairs:]

        sigma = self._sample_noise_levels(pairs, device=device, generator=rng.timestep)
        noise_winner = torch.randn(
            winner.shape, device=device, dtype=winner.dtype, generator=rng.noise
        )
        noise_loser = (
            noise_winner
            if self.config.share_noise
            else torch.randn(
                loser.shape, device=device, dtype=loser.dtype, generator=rng.noise
            )
        )

        def _noise(clean: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
            shape = (pairs, *((1,) * (clean.ndim - 1)))
            level = sigma.reshape(shape).to(clean.dtype)
            return (1.0 - level) * clean + level * eps

        noisy = torch.cat((_noise(winner, noise_winner), _noise(loser, noise_loser)))
        # v = eps - x_0, the frozen rectified-flow target from CONTRACTS.
        target = torch.cat((noise_winner - winner, noise_loser - loser))
        doubled_sigma = torch.cat((sigma, sigma))

        stream = patchifier.to_tokens(
            noisy,
            positions=batch.video_positions,
            mask=batch.video_mask,
            noise_level=doubled_sigma,
        )
        target_tokens = patchifier.to_tokens(
            target,
            positions=batch.video_positions,
            mask=batch.video_mask,
            noise_level=doubled_sigma,
        ).tokens
        inputs = ModelInput(
            video=stream,
            audio=TokenStream.empty_like(total, stream.width, device=device),
            text=TextContext(features=batch.text, mask=batch.text_mask),
        )
        if cp_mesh is not None:
            from avgen.parallel.context import shard_stream

            inputs = inputs.replace_streams(video=shard_stream(stream, cp_mesh))
            target_tokens = shard_stream(
                stream.with_tokens(target_tokens), cp_mesh
            ).tokens

        loss_mask = inputs.video.loss_mask().unsqueeze(-1)
        policy = model(inputs).video
        with torch.no_grad():
            self.reference_model.eval()
            reference = self.reference_model(inputs).video

        weights = None
        if self.config.snr_weighting:
            # (1-s)^2 converts the velocity residual into the noise residual the
            # paper's objective is written in; see the module docstring.
            factor = (1.0 - doubled_sigma) ** 2
            weights = factor.reshape(total, *((1,) * (policy.ndim - 1)))

        def _error(prediction: torch.Tensor) -> torch.Tensor:
            residual = prediction if weights is None else prediction * weights.sqrt()
            goal = target_tokens if weights is None else target_tokens * weights.sqrt()
            return _squared_error(residual, goal, loss_mask)

        policy_error = _error(policy)
        reference_error = _error(reference)
        loss, margin = dpo_loss(
            policy_error[:pairs],
            policy_error[pairs:],
            reference_error[:pairs],
            reference_error[pairs:],
            self.config,
        )
        self.last_margin = margin.detach().float()
        zero = torch.zeros((), device=device, dtype=torch.float32)
        return _objective_output(
            loss=loss.float(),
            video_loss=loss.detach().float(),
            audio_loss=zero,
        )
